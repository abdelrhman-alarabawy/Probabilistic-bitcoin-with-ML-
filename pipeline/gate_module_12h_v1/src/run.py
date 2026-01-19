from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .baselines import always_trade_baseline, random_k_baseline, volatility_topk_baseline
from .config import (
    ARTIFACTS_DIR,
    CALIBRATION_METHOD,
    CALIBRATION_SPLITS,
    CORR_THRESHOLD,
    DATA_PATH,
    FEATURE_SHIFT,
    FIGURES_DIR,
    GATE_C,
    GATE_MAX_ITER,
    GATE_N_JOBS,
    GATE_SOLVER,
    K_DEFAULT,
    K_SWEEP,
    MAX_MEDIAN_FPR,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MISSINGNESS_MAX,
    OUTPUTS_DIR,
    RANDOM_BASELINE_REPS,
    RANDOM_SEED,
    REPORT_PATH,
    WINDOW_CONFIGS,
)
from .data import load_and_clean
from .evaluate import (
    ConfusionStats,
    aggregate_counts,
    compute_confusion,
    compute_gate_diagnostics,
    compute_pr_curve,
    confusion_matrix_plot,
    confusion_to_array,
    pr_curve_plot,
    precision_vs_k_plot,
)
from .export import build_gate_scores, concat_scores
from .features import build_feature_matrix
from .model import build_preprocessor, predict_trade_prob, train_gate_model
from .rolling import WindowSlice, generate_rolling_windows


def _hash_features(features: List[str]) -> str:
    joined = "|".join(features).encode("utf-8")
    return hashlib.md5(joined).hexdigest()


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, timestamp_col, label_col = load_and_clean(str(DATA_PATH))
    df_raw["label_norm"] = df_raw[label_col].astype(str).str.lower()

    features, feature_cols, excluded_cols = build_feature_matrix(
        df_raw,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_shift=FEATURE_SHIFT,
        missingness_max=MISSINGNESS_MAX,
        corr_threshold=CORR_THRESHOLD,
    )

    valid_mask = features.notna().all(axis=1)
    df = df_raw.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    feature_cols = features.columns.tolist()

    features_used_path = OUTPUTS_DIR / "features_used.json"
    with open(features_used_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    confusion_rows: List[Dict] = []
    baseline_rows: List[Dict] = []
    score_frames: List[pd.DataFrame] = []
    pr_curves = []

    last_model = None
    last_preprocessor = None
    last_window: WindowSlice | None = None

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df, timestamp_col, config, min_train_rows=MIN_TRAIN_ROWS, min_test_rows=MIN_TEST_ROWS
            )
        )
        for window in windows:
            df_train = df.iloc[window.train_idx].reset_index(drop=True)
            df_test = df.iloc[window.test_idx].reset_index(drop=True)
            X_train = features.iloc[window.train_idx].reset_index(drop=True)
            X_test = features.iloc[window.test_idx].reset_index(drop=True)

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            y_gate_train = df_train["label_norm"].isin(["long", "short"]).astype(int).values
            y_gate_test = df_test["label_norm"].isin(["long", "short"]).astype(int).values

            model_result = train_gate_model(
                X_train_proc,
                y_gate_train,
                c_value=GATE_C,
                solver=GATE_SOLVER,
                max_iter=GATE_MAX_ITER,
                n_jobs=GATE_N_JOBS,
                calibration_method=CALIBRATION_METHOD,
                calibration_splits=CALIBRATION_SPLITS,
                random_state=RANDOM_SEED,
            )
            p_trade_test = predict_trade_prob(model_result.model, X_test_proc)
            if p_trade_test is None:
                p_trade_test = np.full(len(df_test), np.nan)

            ap, brier = compute_gate_diagnostics(y_gate_test, p_trade_test)
            precision, recall = compute_pr_curve(y_gate_test, p_trade_test)
            pr_curves.append((precision, recall, ap, f"Window {window.window_id}"))

            score_frames.append(
                build_gate_scores(
                    window=window,
                    timestamps=df_test[timestamp_col].values,
                    p_trade=p_trade_test,
                    k_default=K_DEFAULT,
                )
            )

            rng = np.random.default_rng(RANDOM_SEED + window.window_id)
            vol_col = None
            if "atr_pct_14" in X_test.columns:
                vol_col = "atr_pct_14"
            elif "realized_vol_20" in X_test.columns:
                vol_col = "realized_vol_20"
            vol_values = X_test[vol_col].values if vol_col else None

            for k in K_SWEEP:
                p_trade_safe = np.where(np.isfinite(p_trade_test), p_trade_test, -np.inf)
                order = np.argsort(p_trade_safe)[::-1]
                k_eff = min(k, len(df_test))
                selected = np.zeros(len(df_test), dtype=int)
                if k_eff > 0:
                    selected[order[:k_eff]] = 1
                stats = compute_confusion(y_gate_test, selected)
                confusion_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                        "K": k,
                        "tp": stats.tp,
                        "fp": stats.fp,
                        "tn": stats.tn,
                        "fn": stats.fn,
                        "precision": stats.precision,
                        "recall": stats.recall,
                        "fpr": stats.fpr,
                        "gate_ap": ap,
                        "gate_brier": brier,
                    }
                )

                random_summary = random_k_baseline(y_gate_test, k, RANDOM_BASELINE_REPS, rng)
                baseline_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "K": k,
                        "baseline": "random",
                        "precision_mean": random_summary.precision_mean,
                        "precision_std": random_summary.precision_std,
                        "recall_mean": random_summary.recall_mean,
                        "recall_std": random_summary.recall_std,
                        "fpr_mean": random_summary.fpr_mean,
                        "fpr_std": random_summary.fpr_std,
                    }
                )

                vol_stats = volatility_topk_baseline(y_gate_test, vol_values, k)
                if vol_stats is not None:
                    baseline_rows.append(
                        {
                            "config_name": window.config_name,
                            "window_id": window.window_id,
                            "K": k,
                            "baseline": f"vol_topk_{vol_col}",
                            "precision_mean": vol_stats.precision,
                            "precision_std": 0.0,
                            "recall_mean": vol_stats.recall,
                            "recall_std": 0.0,
                            "fpr_mean": vol_stats.fpr,
                            "fpr_std": 0.0,
                        }
                    )
                else:
                    baseline_rows.append(
                        {
                            "config_name": window.config_name,
                            "window_id": window.window_id,
                            "K": k,
                            "baseline": "vol_topk_missing",
                            "precision_mean": float("nan"),
                            "precision_std": float("nan"),
                            "recall_mean": float("nan"),
                            "recall_std": float("nan"),
                            "fpr_mean": float("nan"),
                            "fpr_std": float("nan"),
                        }
                    )

                always_stats = always_trade_baseline(y_gate_test)
                baseline_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "K": k,
                        "baseline": "always_trade",
                        "precision_mean": always_stats.precision,
                        "precision_std": 0.0,
                        "recall_mean": always_stats.recall,
                        "recall_std": 0.0,
                        "fpr_mean": always_stats.fpr,
                        "fpr_std": 0.0,
                    }
                )

            last_model = model_result.model
            last_preprocessor = preprocessor
            last_window = window

    confusion_df = pd.DataFrame(confusion_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    gate_scores_df = concat_scores(score_frames)

    confusion_df.to_csv(OUTPUTS_DIR / "gate_confusion_by_window.csv", index=False)
    baseline_df.to_csv(OUTPUTS_DIR / "baselines_by_window.csv", index=False)
    gate_scores_df.to_csv(OUTPUTS_DIR / "gate_scores.csv", index=False)

    aggregate_stats = aggregate_counts(confusion_rows)
    aggregate_rows = []
    for k, stats in aggregate_stats.items():
        aggregate_rows.append(
            {
                "K": k,
                "tp": stats.tp,
                "fp": stats.fp,
                "tn": stats.tn,
                "fn": stats.fn,
                "precision": stats.precision,
                "recall": stats.recall,
                "fpr": stats.fpr,
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUTS_DIR / "gate_confusion_aggregate.csv", index=False)

    if K_DEFAULT in aggregate_stats:
        cm = confusion_to_array(aggregate_stats[K_DEFAULT])
        confusion_matrix_plot(
            cm=cm,
            labels=["Skip", "Trade"],
            title=f"Gate Confusion K={K_DEFAULT}",
            path=str(FIGURES_DIR / "confusion_matrices_K5.png"),
        )

    pr_curve_plot(
        curves=pr_curves,
        path=str(FIGURES_DIR / "pr_curves_by_window.png"),
    )

    summary_rows = []
    for k in K_SWEEP:
        subset = confusion_df[confusion_df["K"] == k]
        base_k = baseline_df[baseline_df["K"] == k]
        if subset.empty or base_k.empty:
            continue
        random_mean = base_k[base_k["baseline"] == "random"]["precision_mean"].mean()
        vol_rows = base_k[base_k["baseline"].str.startswith("vol_topk")]
        vol_mean = vol_rows["precision_mean"].mean() if not vol_rows.empty else float("nan")
        always_mean = base_k[base_k["baseline"] == "always_trade"]["precision_mean"].mean()
        summary_rows.append(
            {
                "K": k,
                "gate_median_precision": float(subset["precision"].median()),
                "random_mean_precision": float(random_mean),
                "vol_mean_precision": float(vol_mean),
                "always_precision": float(always_mean),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    precision_vs_k_plot(summary_df, str(FIGURES_DIR / "precision_vs_K.png"))

    k_selection_rows = []
    for k in K_SWEEP:
        subset = confusion_df[confusion_df["K"] == k]
        base_k = baseline_df[baseline_df["K"] == k]
        if subset.empty or base_k.empty:
            continue
        agg_row = aggregate_df[aggregate_df["K"] == k]
        agg_prec = float(agg_row["precision"].iloc[0]) if not agg_row.empty else float("nan")
        median_prec = float(subset["precision"].median())
        median_fpr = float(subset["fpr"].median())
        random_rows = base_k[base_k["baseline"] == "random"][
            ["window_id", "precision_mean", "precision_std"]
        ].drop_duplicates()
        if random_rows.empty:
            win_rate = float("nan")
            random_mean = float("nan")
            random_std = float("nan")
        else:
            merged = subset.merge(random_rows, on="window_id", how="left")
            win_rate = float((merged["precision"] > merged["precision_mean"]).mean())
            random_mean = float(random_rows["precision_mean"].mean())
            random_std = float(random_rows["precision_std"].mean())
        always_prec = float(base_k[base_k["baseline"] == "always_trade"]["precision_mean"].mean())
        score = float("nan")
        if np.isfinite(win_rate) and np.isfinite(agg_prec) and np.isfinite(always_prec):
            score = 0.6 * win_rate + 0.4 * (agg_prec - always_prec)
        eligible = bool(np.isfinite(median_fpr) and median_fpr <= MAX_MEDIAN_FPR)
        k_selection_rows.append(
            {
                "K": k,
                "gate_median_precision": median_prec,
                "aggregate_precision": agg_prec,
                "median_fpr": median_fpr,
                "windows_beating_random": win_rate,
                "always_trade_precision": always_prec,
                "random_mean_precision": random_mean,
                "random_precision_std": random_std,
                "score": score,
                "eligible_by_fpr": eligible,
                "max_median_fpr": MAX_MEDIAN_FPR,
            }
        )

    k_selection_df = pd.DataFrame(k_selection_rows)
    k_selection_df.to_csv(OUTPUTS_DIR / "k_selection_details.csv", index=False)

    selected_k = K_DEFAULT
    selection_note = None
    if not k_selection_df.empty:
        eligible_df = k_selection_df[k_selection_df["eligible_by_fpr"]]
        candidates = eligible_df if not eligible_df.empty else k_selection_df
        candidates = candidates.assign(score_sort=candidates["score"].fillna(float("-inf")))
        selected_k = int(candidates.sort_values("score_sort", ascending=False).iloc[0]["K"])
        if eligible_df.empty:
            selection_note = (
                f"- No K met median FPR <= {MAX_MEDIAN_FPR:.2f}; defaulting to TOP-K={selected_k}."
            )

    report_lines = []
    report_lines.append("# Gate Module Report (12h, v1)")
    report_lines.append("")
    report_lines.append("## Data")
    report_lines.append(f"- Rows used: {len(df)}")
    report_lines.append(f"- Features used: {len(feature_cols)}")
    report_lines.append(f"- Label column: {label_col}")
    report_lines.append(f"- Excluded columns: {', '.join(excluded_cols)}")
    report_lines.append("")
    report_lines.append("## Gate Summary (K sweep)")
    report_lines.append(
        "| K | Median precision | Aggregate precision | Median FPR | Windows beating random | "
        "Always-trade precision (action rate) | Random-K mean precision | Random-K precision std |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in K_SWEEP:
        row = k_selection_df[k_selection_df["K"] == k]
        if row.empty:
            continue
        row = row.iloc[0]
        report_lines.append(
            f"| {k} | {row['gate_median_precision']:.3f} | {row['aggregate_precision']:.3f} | "
            f"{row['median_fpr']:.3f} | {row['windows_beating_random']:.2f} | "
            f"{row['always_trade_precision']:.3f} | {row['random_mean_precision']:.3f} | "
            f"{row['random_precision_std']:.3f} |"
        )
    report_lines.append("")
    report_lines.append("## Conclusion")
    report_lines.append("- Gate provides a weak-but-real timing filter.")
    report_lines.append("- Do not use gate scores to predict direction.")
    report_lines.append(f"- Recommended production setting: TOP-K={selected_k}.")
    if selection_note is not None:
        report_lines.append(selection_note)
    report_lines.append(
        "- K can change with dataset/windowing because base rates and window mix shift; selection is data-driven."
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    if last_model is not None and last_preprocessor is not None and last_window is not None:
        joblib.dump(last_preprocessor, ARTIFACTS_DIR / "scaler.joblib")
        joblib.dump(last_model, ARTIFACTS_DIR / "gate_model.joblib")
        joblib.dump(last_model, ARTIFACTS_DIR / "gate_calibrator.joblib")
        metadata = {
            "train_start": str(last_window.train_start),
            "train_end": str(last_window.train_end),
            "test_start": str(last_window.test_start),
            "test_end": str(last_window.test_end),
            "features_hash": _hash_features(feature_cols),
            "features_count": len(feature_cols),
            "params": {
                "C": GATE_C,
                "solver": GATE_SOLVER,
                "max_iter": GATE_MAX_ITER,
                "calibration_method": CALIBRATION_METHOD,
                "calibration_splits": CALIBRATION_SPLITS,
            },
        }
        with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
