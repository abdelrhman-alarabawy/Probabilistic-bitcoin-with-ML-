from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .baseline_models import train_multiclass_model
from .config import ArtifactPaths, BetaTestConfig
from .io import load_labeled_csv, missingness_summary, select_feature_columns
from .label_cleaning import apply_label_cleaning
from .range_ambiguity_probe import run_range_ambiguity_probe
from .plots import (
    plot_confusion_matrix,
    plot_gate_precision_coverage_curve,
    plot_label_distribution_per_setting,
)
from .report import write_report_v3
from .splits import build_split_table, holdout_indices, rolling_splits
from .thresholding import (
    build_probability_sample,
    multiclass_prob_stats,
    sweep_thresholds_multiclass,
)
from .two_stage import run_two_stage


LABELS_ALLOWED = ["long", "short", "skip"]


def _ensure_dirs(paths: ArtifactPaths) -> None:
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)


def _save_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _setting_tag(min_range_pct: Optional[float]) -> str:
    if min_range_pct is None:
        return "raw"
    return f"{min_range_pct:.4f}".replace(".", "p")


def _direction_debug_sample(
    timestamps: pd.Series,
    y_true: List[str],
    p_long: np.ndarray,
    preds: List[str],
    sample_size: int = 30,
) -> pd.DataFrame:
    if len(y_true) == 0:
        return pd.DataFrame(columns=["timestamp", "true_label", "p_long", "predicted_label"])

    sample_idx = np.linspace(0, len(y_true) - 1, num=min(sample_size, len(y_true)), dtype=int)
    rows = []
    for idx in sample_idx:
        rows.append(
            {
                "timestamp": timestamps.iloc[idx],
                "true_label": y_true[idx],
                "p_long": float(p_long[idx]),
                "predicted_label": preds[idx],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = BetaTestConfig()
    paths = ArtifactPaths()
    _ensure_dirs(paths)

    df, time_col, dataset_summary = load_labeled_csv(
        config.labeled_csv_path,
        config.time_col_candidates,
    )

    if config.label_col not in df.columns:
        raise ValueError(f"Missing label column '{config.label_col}' in labeled CSV.")

    label_values = set(df[config.label_col].dropna().unique())
    if not label_values.issubset(set(LABELS_ALLOWED)):
        raise ValueError(f"Unexpected labels detected: {label_values - set(LABELS_ALLOWED)}")

    run_range_ambiguity_probe(
        df,
        label_col=config.label_col,
        ambig_col=config.ambig_col,
        tables_dir=paths.tables_dir,
        min_range_pcts=[0.0, 0.005, 0.01, 0.015, 0.02],
    )

    missing_top = missingness_summary(df)
    missing_top.to_csv(paths.tables_dir / "missingness_top20.csv", index=False)

    holdout = holdout_indices(len(df), config.holdout_test_frac)
    rolling = rolling_splits(len(df), config.cv_splits)
    split_table = build_split_table(df, time_col, config.label_col, LABELS_ALLOWED, holdout, rolling)
    split_table.to_csv(paths.tables_dir / "split_summary.csv", index=False)

    feature_cols, missing_table = select_feature_columns(
        df,
        time_col=time_col,
        label_col=config.label_col,
        ambig_col=config.ambig_col,
        drop_missing_above=config.drop_missing_above,
    )
    missing_table.to_csv(paths.tables_dir / "feature_missingness.csv", index=False)

    if not feature_cols:
        raise ValueError("No feature columns available after filtering.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    train_idx, test_idx = holdout
    X_train = X.iloc[train_idx].to_numpy()
    X_test = X.iloc[test_idx].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    if config.use_label_cleaning:
        settings = list(config.min_range_pct_list)
    else:
        settings = [None]

    cleaning_rows = []
    settings_results = []
    label_counts_per_setting = []
    setting_labels = []
    prob_stats_rows = []

    best_setting = None
    best_score = -1.0
    best_coverage = 0.0

    for min_range_pct in settings:
        if min_range_pct is None:
            force_ambiguous = False
            min_range_filter = False
            min_range_value = 0.0
        else:
            force_ambiguous = config.force_ambiguous_to_skip
            min_range_filter = config.min_range_filter
            min_range_value = float(min_range_pct)

        cleaned_df, stats = apply_label_cleaning(
            df,
            label_col=config.label_col,
            ambig_col=config.ambig_col,
            force_ambiguous_to_skip=force_ambiguous,
            min_range_filter=min_range_filter,
            min_range_pct=min_range_value,
        )

        label_counts = stats["after_counts"]
        label_counts_per_setting.append(label_counts)
        setting_labels.append(_setting_tag(min_range_pct))

        cleaning_rows.append(
            {
                "min_range_pct": min_range_pct,
                "count_long_before": stats["before_counts"]["long"],
                "count_short_before": stats["before_counts"]["short"],
                "count_skip_before": stats["before_counts"]["skip"],
                "count_long_after": stats["after_counts"]["long"],
                "count_short_after": stats["after_counts"]["short"],
                "count_skip_after": stats["after_counts"]["skip"],
                "ambiguous_true": stats["ambiguous_true"],
                "num_changed_by_ambiguous": stats["num_changed_by_ambiguous"],
                "num_changed_by_min_range": stats["num_changed_by_min_range"],
                "total_rows_changed": stats["total_rows_changed"],
                "range_pct_min": stats["range_pct_min"],
                "range_pct_median": stats["range_pct_median"],
                "range_pct_max": stats["range_pct_max"],
                "range_filtered": stats["range_filtered"],
            }
        )

        y = cleaned_df[config.label_col].astype(str)
        y_train = y.iloc[train_idx].reset_index(drop=True).to_numpy()
        y_test = y.iloc[test_idx].reset_index(drop=True).to_numpy()

        two_stage = run_two_stage(
            X_train,
            y_train,
            X_test,
            y_test,
            config.gate_thresh_grid,
            config.dir_thresh_grid,
            config.dir_long_high_list,
            config.dir_short_low_list,
            config.min_coverage_trade,
            config.min_dir_samples,
            config.min_coverage_total,
            config.seed,
        )

        tag = _setting_tag(min_range_pct)

        gate_grid = pd.DataFrame(two_stage["gate_grid"])
        if not gate_grid.empty:
            gate_grid.to_csv(paths.tables_dir / f"gate_threshold_grid_{tag}.csv", index=False)

        direction_grid = pd.DataFrame(two_stage["direction_grid"])
        if not direction_grid.empty:
            direction_grid.to_csv(paths.tables_dir / f"direction_threshold_grid_{tag}.csv", index=False)

        band_grid = pd.DataFrame(two_stage["direction_band_grid"])
        if not band_grid.empty:
            band_grid.to_csv(paths.tables_dir / f"direction_band_grid_{tag}.csv", index=False)

        multiclass_output = train_multiclass_model(
            X_train,
            y_train,
            X_test,
            y_test,
            LABELS_ALLOWED,
            config.seed,
        )
        multiclass_report = multiclass_output.metrics["report"]
        macro_f1 = multiclass_report.get("macro avg", {}).get("f1-score", 0.0)

        prob_stats = multiclass_prob_stats(multiclass_output.proba, LABELS_ALLOWED)
        prob_stats["min_range_pct"] = min_range_pct
        prob_stats_rows.append(prob_stats)

        grid_df, best_thresholds = sweep_thresholds_multiclass(
            multiclass_output.proba,
            y_test.tolist(),
            LABELS_ALLOWED,
            config.thresh_grid,
            config.min_coverage,
        )

        if not grid_df.empty:
            grid_df.to_csv(paths.tables_dir / f"multiclass_threshold_grid_{tag}.csv", index=False)

        thresholds_summary = None
        if best_thresholds is not None:
            thresholds_summary = {
                "th_long": best_thresholds["th_long"],
                "th_short": best_thresholds["th_short"],
                "precision_long": best_thresholds["precision_long"],
                "precision_short": best_thresholds["precision_short"],
                "coverage": best_thresholds["coverage"],
                "trade_precision": best_thresholds["trade_precision"],
                "score": best_thresholds["score"],
            }
            _save_json(thresholds_summary, paths.reports_dir / f"best_thresholds_{tag}.json")
        else:
            argmax_trade = prob_stats["argmax_long_pct"] + prob_stats["argmax_short_pct"]
            if argmax_trade >= config.min_coverage:
                print(
                    "ERROR: argmax trade coverage meets MIN_COVERAGE but no feasible thresholds found."
                )

        if best_thresholds is None:
            sample_th_long = max(config.thresh_grid)
            sample_th_short = max(config.thresh_grid)
        else:
            sample_th_long = best_thresholds["th_long"]
            sample_th_short = best_thresholds["th_short"]

        sample_df = build_probability_sample(
            cleaned_df.iloc[test_idx].reset_index(drop=True),
            y_test.tolist(),
            multiclass_output.proba,
            LABELS_ALLOWED,
            sample_th_long,
            sample_th_short,
            time_col,
        )
        sample_df.to_csv(paths.predictions_dir / f"probability_sanity_sample_{tag}.csv", index=False)

        end_to_end = two_stage.get("end_to_end")
        gate_best = two_stage.get("gate_best")
        direction_best = two_stage.get("direction_band_best") or two_stage.get("direction_best")

        if end_to_end:
            score = (end_to_end["precision_long"] + end_to_end["precision_short"]) / 2
            coverage_total = end_to_end["coverage_total"]
            if coverage_total >= config.min_coverage_total and (
                score > best_score or (score == best_score and coverage_total > best_coverage)
            ):
                best_score = score
                best_coverage = coverage_total
                best_setting = {
                    "min_range_pct": min_range_pct,
                    "tag": tag,
                    "end_to_end": end_to_end,
                    "gate_best": gate_best,
                    "direction_best": direction_best,
                    "y_test": y_test.tolist(),
                    "gated_idx": two_stage.get("gated_idx"),
                    "p_long_gated": two_stage.get("p_long_gated"),
                    "y_test_gated": two_stage.get("y_test_gated"),
                    "cleaned_df": cleaned_df,
                }

        settings_results.append(
            {
                "min_range_pct": min_range_pct,
                "label_counts": label_counts,
                "ambiguous_true": stats["ambiguous_true"],
                "gate": {"best": gate_best},
                "direction": {
                    "best": direction_best,
                    "gated_count": two_stage.get("gated_count", 0),
                },
                "end_to_end": end_to_end,
                "multiclass": {
                    "model": multiclass_output.model_name,
                    "macro_f1": macro_f1,
                    "thresholds": thresholds_summary,
                },
            }
        )

    pd.DataFrame(cleaning_rows).to_csv(
        paths.tables_dir / "label_cleaning_effects.csv", index=False
    )

    counts_unique = {
        (
            row["count_long_after"],
            row["count_short_after"],
            row["count_skip_after"],
        )
        for row in cleaning_rows
    }
    if len(counts_unique) == 1 and config.use_label_cleaning:
        print("WARNING: Label counts identical across min_range_pct settings. Cleaning may be ineffective.")

    plot_label_distribution_per_setting(
        setting_labels,
        label_counts_per_setting,
        str(paths.figures_dir / "label_distribution_per_setting.png"),
    )

    if prob_stats_rows:
        pd.DataFrame(prob_stats_rows).to_csv(
            paths.tables_dir / "multiclass_prob_stats.csv", index=False
        )

    direction_debug_path = paths.tables_dir / "direction_debug_sample.csv"
    if best_setting and best_setting.get("p_long_gated") is not None:
        tag = best_setting["tag"]
        gate_best = best_setting.get("gate_best")
        end_to_end = best_setting.get("end_to_end")

        gate_grid_path = paths.tables_dir / f"gate_threshold_grid_{tag}.csv"
        if gate_best and gate_grid_path.exists():
            gate_grid_df = pd.read_csv(gate_grid_path)
            if not gate_grid_df.empty:
                plot_gate_precision_coverage_curve(
                    gate_grid_df["coverage_trade"].tolist(),
                    gate_grid_df["precision_trade"].tolist(),
                    str(paths.figures_dir / "gate_precision_coverage_curve.png"),
                )

        if end_to_end is not None:
            preds = end_to_end["preds"]
            y_test_best = best_setting.get("y_test", [])
            confusion_best = pd.crosstab(
                pd.Series(y_test_best, name="actual"),
                pd.Series(preds, name="predicted"),
                dropna=False,
            ).reindex(index=LABELS_ALLOWED, columns=LABELS_ALLOWED, fill_value=0)
            confusion_best.to_csv(paths.tables_dir / "confusion_end_to_end_best.csv")
            plot_confusion_matrix(
                confusion_best.to_numpy(),
                LABELS_ALLOWED,
                str(paths.figures_dir / "confusion_matrix_end_to_end.png"),
            )

            pred_long = end_to_end["pred_long"]
            pred_short = end_to_end["pred_short"]
            pred_total = pred_long + pred_short
            if pred_total > 0:
                share_long = pred_long / pred_total
                share_short = pred_short / pred_total
                if share_long > 0.9 or share_short > 0.9:
                    print("WARNING: Direction predictions are highly imbalanced (>90% one class).")

        if best_setting.get("gated_idx") is not None:
            gated_idx = best_setting["gated_idx"]
            cleaned_df = best_setting["cleaned_df"]
            timestamps = cleaned_df.iloc[test_idx].reset_index(drop=True).iloc[gated_idx][time_col]
            p_long = best_setting["p_long_gated"]
            y_true_gated = best_setting.get("y_test_gated") or []

            if best_setting.get("direction_best"):
                dir_best = best_setting["direction_best"]
                if "dir_long_high" in dir_best and "dir_short_low" in dir_best:
                    long_high = dir_best["dir_long_high"]
                    short_low = dir_best["dir_short_low"]
                    preds = [
                        "long" if p >= long_high else "short" if p <= short_low else "skip"
                        for p in p_long
                    ]
                elif "dir_threshold" in dir_best:
                    th = dir_best["dir_threshold"]
                    preds = ["long" if p >= th else "short" for p in p_long]
                else:
                    preds = ["skip" for _ in p_long]
            else:
                preds = ["skip" for _ in p_long]

            debug_df = _direction_debug_sample(
                timestamps.reset_index(drop=True),
                y_true_gated,
                p_long,
                preds,
            )
            debug_df.to_csv(direction_debug_path, index=False)
    else:
        pd.DataFrame(
            columns=["timestamp", "true_label", "p_long", "predicted_label"]
        ).to_csv(direction_debug_path, index=False)

    notes = [
        "If trade-vs-skip is near random but direction on trades is better, entry timing may be weak.",
        "If direction is random but trade-vs-skip is strong, direction labeling or features are weak.",
        "If probabilities are skewed to skip, the model may be collapsing to skip.",
        "If argmax differs from decision at high rates, thresholding or class order may be off.",
    ]

    report_path = paths.reports_dir / "BetaTest_Report_v3.md"
    write_report_v3(report_path, dataset_summary, split_table, settings_results, notes)

    if best_setting and best_setting.get("end_to_end"):
        end_to_end = best_setting["end_to_end"]
        gate_best = best_setting.get("gate_best")
        direction_best = best_setting.get("direction_best")

        print("==== BETATEST V3 SUMMARY ====")
        print(f"Best min_range_pct = {best_setting['min_range_pct']}")
        if gate_best:
            print(
                "Gate: precision_trade={:.3f}, coverage_trade={:.3f}".format(
                    gate_best["precision_trade"], gate_best["coverage_trade"]
                )
            )
        else:
            print("Gate: precision_trade=0.000, coverage_trade=0.000")

        if direction_best:
            print(
                "Direction (gated): precision_long={:.3f}, precision_short={:.3f}".format(
                    direction_best.get("precision_long", 0.0),
                    direction_best.get("precision_short", 0.0),
                )
            )
        else:
            print("Direction (gated): precision_long=0.000, precision_short=0.000")

        print(
            "End-to-end: precision_long={:.3f}, precision_short={:.3f}, coverage_total={:.3f}".format(
                end_to_end["precision_long"],
                end_to_end["precision_short"],
                end_to_end["coverage_total"],
            )
        )
        print(
            "Predictions: long={}, short={}, skip={}".format(
                end_to_end["pred_long"], end_to_end["pred_short"], end_to_end["pred_skip"]
            )
        )
        print(f"See report: {report_path}")
        print("=============================")
    else:
        print("==== BETATEST V3 SUMMARY ====")
        print("Best min_range_pct = None")
        print("Gate: precision_trade=0.000, coverage_trade=0.000")
        print("Direction (gated): precision_long=0.000, precision_short=0.000")
        print("End-to-end: precision_long=0.000, precision_short=0.000, coverage_total=0.000")
        print("Predictions: long=0, short=0, skip=0")
        print(f"See report: {report_path}")
        print("=============================")


if __name__ == "__main__":
    main()
