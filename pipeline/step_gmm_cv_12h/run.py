from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cv_monthly import build_monthly_folds
from src.gmm_regimes import assign_regimes, compute_regime_stats, fit_gmm, select_k_bic
from src.io_utils import ensure_dir, save_json
from src.labeling_12h import LabelingParams, label_candles, label_distribution
from src.plots import (
    plot_close_by_regime,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_regime_counts,
    plot_regime_stats,
)
from src.preprocess import (
    data_quality_report,
    detect_ohlcv_columns,
    detect_timestamp_column,
    fit_preprocess,
    numeric_feature_columns,
)
from src.report import build_report
from src.train_eval import train_and_evaluate


LABEL_ORDER = ["long", "short", "skip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="12h GMM regime discovery + rolling CV.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gmm-k", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--tp-points", type=float, default=2000.0)
    parser.add_argument("--sl-points", type=float, default=1000.0)
    parser.add_argument("--base-horizon-minutes", type=int, default=60)
    parser.add_argument(
        "--no-compare-regime-feature",
        action="store_true",
        help="Disable comparison with regime_id as a feature.",
    )
    return parser.parse_args()


def flatten_report(report: dict, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_precision": report[prefix]["precision"],
        f"{prefix}_recall": report[prefix]["recall"],
        f"{prefix}_f1": report[prefix]["f1-score"],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    root = Path(__file__).resolve().parent
    output_dir = root / "output"
    models_dir = root / "models"
    plots_dir = root / "plots"
    ensure_dir(output_dir)
    ensure_dir(models_dir)
    ensure_dir(plots_dir)
    ensure_dir(output_dir / "folds_confusion_matrices")

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    timestamp_col = detect_timestamp_column(df.columns)
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    else:
        print("WARNING: No timestamp column detected; using index-based ordering.")

    ohlcv_cols = detect_ohlcv_columns(df.columns)

    quality = data_quality_report(df, timestamp_col)
    save_json(quality, output_dir / "data_quality.json")

    label_params = LabelingParams(
        base_horizon_minutes=args.base_horizon_minutes,
        tp_points=args.tp_points,
        sl_points=args.sl_points,
    )
    labels = label_candles(
        df,
        open_col=ohlcv_cols["open"],
        high_col=ohlcv_cols["high"],
        low_col=ohlcv_cols["low"],
        params=label_params,
    )
    df["Candle_type"] = labels
    full_label_dist = label_distribution(labels)
    print(f"Label distribution: {full_label_dist}")
    df.to_csv(output_dir / "labeled_full.csv", index=False)

    exclude_cols = list(ohlcv_cols.values()) + ["Candle_type"]
    if timestamp_col:
        exclude_cols.append(timestamp_col)
    feature_cols = numeric_feature_columns(df, exclude_cols)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for modeling.")

    folds, month_col = build_monthly_folds(df, timestamp_col, min_train_months=2)
    if len(folds) == 0:
        raise ValueError("Insufficient months for rolling CV (need at least 3).")

    # Use first fold train window to select global k via BIC.
    first_fold = folds[0]
    X_scaled, imputer_global, scaler_global = fit_preprocess(df, feature_cols, first_fold.train_idx)
    X_train_first = X_scaled[first_fold.train_idx]
    gmm_selection = select_k_bic(X_train_first, args.gmm_k, args.random_state)
    selected_k = gmm_selection.best_k
    save_json(gmm_selection.bic_scores, output_dir / "gmm_bic_scores.json")
    print(f"Selected GMM k={selected_k} using BIC on earliest training window.")

    # Representative regime assignment for plots (trained on first fold only).
    gmm_global = fit_gmm(X_train_first, selected_k, args.random_state)
    regimes_full = assign_regimes(gmm_global, X_scaled)
    df["regime_id"] = regimes_full
    df.to_csv(output_dir / "regimes_full.csv", index=False)

    # Optional per-regime splits for inspection.
    per_regime_dir = output_dir / "per_regime_splits"
    ensure_dir(per_regime_dir)
    for regime_id in sorted(df["regime_id"].unique()):
        df[df["regime_id"] == regime_id].to_csv(
            per_regime_dir / f"regime_{regime_id}.csv", index=False
        )

    stats_df = compute_regime_stats(
        df,
        regime_col="regime_id",
        close_col=ohlcv_cols["close"],
        high_col=ohlcv_cols["high"],
        low_col=ohlcv_cols["low"],
        label_col="Candle_type",
    )
    stats_df.to_csv(output_dir / "gmm_regime_stats.csv", index=False)

    label_dist_by_regime = (
        df.groupby(["regime_id", "Candle_type"]).size().groupby(level=0).apply(lambda s: s / s.sum())
    ).unstack(fill_value=0.0)

    plot_close_by_regime(
        df,
        timestamp_col=timestamp_col,
        close_col=ohlcv_cols["close"],
        regime_col="regime_id",
        output_path=plots_dir / "close_by_regime.png",
    )
    plot_regime_counts(df["regime_id"].value_counts(), plots_dir / "regime_counts.png")
    plot_regime_stats(stats_df, plots_dir / "gmm_cluster_stats.png")
    plot_label_distribution(label_dist_by_regime, plots_dir / "label_distribution_by_regime.png")

    folds_metrics: list[dict[str, object]] = []
    fold_table: list[dict[str, object]] = []
    compare_regime = not args.no_compare_regime_feature

    for fold in folds:
        print(f"Fold {fold.fold_id}: train months={len(fold.train_months)} test={fold.test_month}")
        fold_table.append(
            {
                "fold_id": fold.fold_id,
                "train_months": ",".join(fold.train_months),
                "test_month": fold.test_month,
                "train_size": len(fold.train_idx),
                "test_size": len(fold.test_idx),
            }
        )

        X_scaled, imputer, scaler = fit_preprocess(df, feature_cols, fold.train_idx)
        X_train = X_scaled[fold.train_idx]
        X_test = X_scaled[fold.test_idx]

        gmm_fold = fit_gmm(X_train, selected_k, args.random_state)
        regime_train = assign_regimes(gmm_fold, X_train)
        regime_test = assign_regimes(gmm_fold, X_test)

        y = df["Candle_type"].map({label: idx for idx, label in enumerate(LABEL_ORDER)}).to_numpy()
        y_train = y[fold.train_idx]
        y_test = y[fold.test_idx]
        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold {fold.fold_id}: only one class in training data.")
            continue

        results_per_setting: list[tuple[bool, dict, object, np.ndarray]] = []
        for use_regime in [False, True]:
            if use_regime and not compare_regime:
                continue
            X_train_use = X_train
            X_test_use = X_test
            if use_regime:
                X_train_use = np.column_stack([X_train, regime_train])
                X_test_use = np.column_stack([X_test, regime_test])
            result = train_and_evaluate(
                X_train_use,
                y_train,
                X_test_use,
                y_test,
                labels=LABEL_ORDER,
                random_state=args.random_state,
            )
            report = result.report
            metrics_row = {
                "fold_id": fold.fold_id,
                "use_regime": use_regime,
                "train_months": ",".join(fold.train_months),
                "test_month": fold.test_month,
                "train_size": len(fold.train_idx),
                "test_size": len(fold.test_idx),
                "gmm_k": selected_k,
                "accuracy": report["accuracy"],
            }
            metrics_row.update(flatten_report(report, "macro avg"))
            metrics_row.update(flatten_report(report, "weighted avg"))
            for label in LABEL_ORDER:
                metrics_row.update(
                    {
                        f"{label}_precision": report[label]["precision"],
                        f"{label}_recall": report[label]["recall"],
                        f"{label}_f1": report[label]["f1-score"],
                    }
                )
            folds_metrics.append(metrics_row)
            results_per_setting.append((use_regime, report, result.model, result.confusion))

        # Choose best setting by weighted F1.
        best_use_regime = False
        best_report = None
        best_model = None
        best_confusion = None
        best_weighted_f1 = -1.0
        for use_regime, report, model, confusion in results_per_setting:
            weighted_f1 = report["weighted avg"]["f1-score"]
            if weighted_f1 > best_weighted_f1:
                best_weighted_f1 = weighted_f1
                best_use_regime = use_regime
                best_report = report
                best_model = model
                best_confusion = confusion

        fold_model_dir = models_dir / f"fold_{fold.fold_id}"
        ensure_dir(fold_model_dir)
        joblib.dump(best_model, fold_model_dir / "model.joblib")
        joblib.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "use_regime": best_use_regime,
                "gmm_k": selected_k,
            },
            fold_model_dir / "preprocess.joblib",
        )

        if best_confusion is not None:
            plot_confusion_matrix(
                best_confusion,
                LABEL_ORDER,
                plots_dir / f"confusion_matrix_fold_{fold.fold_id}.png",
            )
            cm_df = pd.DataFrame(best_confusion, index=LABEL_ORDER, columns=LABEL_ORDER)
            cm_df.to_csv(output_dir / "folds_confusion_matrices" / f"fold_{fold.fold_id}.csv")

    folds_df = pd.DataFrame(folds_metrics)
    folds_df.to_csv(output_dir / "folds_metrics.csv", index=False)
    fold_table_df = pd.DataFrame(fold_table)
    fold_table_df.to_csv(output_dir / "folds_table.csv", index=False)

    avg_metrics = {}
    for use_regime, group in folds_df.groupby("use_regime"):
        key = "with_regime" if use_regime else "without_regime"
        avg_metrics[key] = {
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std()),
            "macro_f1_mean": float(group["macro avg_f1"].mean()),
            "macro_f1_std": float(group["macro avg_f1"].std()),
            "weighted_f1_mean": float(group["weighted avg_f1"].mean()),
            "weighted_f1_std": float(group["weighted avg_f1"].std()),
        }
        for label in LABEL_ORDER:
            avg_metrics[key][f"{label}_f1_mean"] = float(group[f"{label}_f1"].mean())
            avg_metrics[key][f"{label}_f1_std"] = float(group[f"{label}_f1"].std())

    best_setting = max(avg_metrics.items(), key=lambda item: item[1]["weighted_f1_mean"])[0]

    report_summary = {
        "input_path": str(input_path),
        "rows": len(df),
        "columns": df.shape[1],
        "timestamp_col": timestamp_col,
        "labeling": {
            "base_horizon": label_params.base_horizon_minutes,
            "scaled_horizon": label_params.horizon_minutes,
            "tp_points": label_params.tp_points,
            "sl_points": label_params.sl_points,
        },
        "gmm": {
            "k_strategy": "global_k_from_first_fold",
            "selected_k": selected_k,
        },
        "folds_count": len(folds),
        "best_setting": best_setting,
        "avg_metrics": avg_metrics,
    }
    save_json(report_summary, output_dir / "report_summary.json")
    build_report(report_summary, root / "REPORT.md")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
