from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import (
    CORR_THRESHOLD,
    DATA_PATH,
    FEATURE_SHIFT,
    FIGURES_DIR,
    GATE_CALIBRATION_METHOD,
    GATE_CALIBRATION_SPLITS,
    GATE_C,
    GATE_FEATURES_PATH,
    GATE_MAX_ITER,
    GATE_N_JOBS,
    GATE_SOLVER,
    GATE_TOPK,
    MIN_COVERAGE_TIGHT,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    OUTPUTS_DIR,
    QUANTILE_PAIRS,
    RANDOM_SEED,
    REACH_HORIZON,
    REPORT_PATH,
    SUMMARY_QUANTILE_PAIR,
    USE_GATE,
    WIDTH_THRESHOLDS,
    WINDOW_CONFIGS,
    MISSINGNESS_MAX,
)
from .data import load_and_prepare_with_counts
from .eval import ConfusionCounts, summarize_interval
from .features import build_feature_matrix, load_features_used, select_features_from_used
from .gate import build_preprocessor as build_gate_preprocessor
from .gate import predict_trade_prob, select_topk, train_gate_model
from .models import select_primary_family, train_quantile_model
from .plots import plot_confusion_matrix, plot_frontier, plot_gate_vs_all, plot_precision_tight, plot_yearly_summary
from .rolling import generate_rolling_windows


def build_interval_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def _format_threshold(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not GATE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing gate features file: {GATE_FEATURES_PATH}")

    df, timestamp_col, label_col, prep_counts = load_and_prepare_with_counts(
        str(DATA_PATH), REACH_HORIZON
    )
    df["label_norm"] = df[label_col].astype(str).str.lower()

    features, feature_cols, excluded_cols = build_feature_matrix(
        df,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_shift=FEATURE_SHIFT,
        missingness_max=MISSINGNESS_MAX,
        corr_threshold=CORR_THRESHOLD,
    )

    missing_feature_counts = features.isna().sum()
    missing_feature_counts = missing_feature_counts[missing_feature_counts > 0].sort_values(ascending=False)

    features_used = load_features_used(GATE_FEATURES_PATH)
    features = select_features_from_used(features, features_used)
    feature_cols = features.columns.tolist()

    valid_mask = features.notna().all(axis=1)
    df = df.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    n_after_features = len(df)

    per_window_rows: List[Dict] = []
    confusion_totals: Dict[Tuple[str, float, float, bool, float], ConfusionCounts] = {}
    sample_store: Dict[Tuple[str, float, float, bool], Dict[str, List[np.ndarray]]] = {}
    summary_rows: List[Dict] = []
    gate_skipped_windows: List[int] = []

    primary_family = select_primary_family(("lightgbm", "sklearn"))
    nominal_coverages = {pair: pair[1] - pair[0] for pair in QUANTILE_PAIRS}

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df,
                timestamp_col,
                config,
                min_train_rows=MIN_TRAIN_ROWS,
                min_test_rows=MIN_TEST_ROWS,
            )
        )
        for window in windows:
            train_idx = window.train_idx
            test_idx = window.test_idx

            X_train = features.iloc[train_idx].values
            X_test = features.iloc[test_idx].values

            y_high_train = df["y_high_reach"].iloc[train_idx].values
            y_high_test = df["y_high_reach"].iloc[test_idx].values
            y_low_train = df["y_low_reach"].iloc[train_idx].values
            y_low_test = df["y_low_reach"].iloc[test_idx].values

            close_test = df["close"].iloc[test_idx].values
            ts_test = df[timestamp_col].iloc[test_idx].values

            interval_pre = build_interval_preprocessor()
            X_train_proc = interval_pre.fit_transform(X_train)
            X_test_proc = interval_pre.transform(X_test)

            gate_mask = None
            if USE_GATE:
                label_valid = df["label_norm"].isin(["long", "short", "skip"]).values
                train_label_mask = label_valid[train_idx]
                if train_label_mask.any():
                    y_gate_train = df["label_norm"].iloc[train_idx].isin(["long", "short"]).astype(int).values
                    y_gate_train = y_gate_train[train_label_mask]
                    X_gate_train = X_train[train_label_mask]
                    X_gate_test = X_test

                    gate_pre = build_gate_preprocessor()
                    X_gate_train = gate_pre.fit_transform(X_gate_train)
                    X_gate_test = gate_pre.transform(X_gate_test)
                    gate_model = train_gate_model(
                        X_gate_train,
                        y_gate_train,
                        c_value=GATE_C,
                        solver=GATE_SOLVER,
                        max_iter=GATE_MAX_ITER,
                        n_jobs=GATE_N_JOBS,
                        calibration_method=GATE_CALIBRATION_METHOD,
                        calibration_splits=GATE_CALIBRATION_SPLITS,
                        random_state=RANDOM_SEED,
                    )
                    p_trade_test = predict_trade_prob(gate_model.model, X_gate_test)
                    if p_trade_test is not None:
                        gate_mask, _ = select_topk(p_trade_test, GATE_TOPK)
                    else:
                        gate_skipped_windows.append(window.window_id)
                else:
                    gate_skipped_windows.append(window.window_id)

            for target_name, y_train, y_test in [
                ("high_reach", y_high_train, y_high_test),
                ("low_reach", y_low_train, y_low_test),
            ]:
                for q_low, q_high in QUANTILE_PAIRS:
                    low_model = train_quantile_model(X_train_proc, y_train, q_low, primary_family)
                    high_model = train_quantile_model(X_train_proc, y_train, q_high, primary_family)
                    pred_low = low_model.predict(X_test_proc)
                    pred_high = high_model.predict(X_test_proc)

                    for gate_flag, mask in [
                        (False, np.ones(len(y_test), dtype=bool)),
                        (True, gate_mask if gate_mask is not None else None),
                    ]:
                        if mask is None:
                            continue
                        y_eval = y_test[mask]
                        pred_low_eval = pred_low[mask]
                        pred_high_eval = pred_high[mask]
                        close_eval = close_test[mask]

                        metrics, confusions, tight_metrics = summarize_interval(
                            y_eval,
                            pred_low_eval,
                            pred_high_eval,
                            close_eval,
                            WIDTH_THRESHOLDS,
                            nominal_coverages[(q_low, q_high)],
                        )

                        row = {
                            "window_id": window.window_id,
                            "config_name": window.config_name,
                            "train_start": window.train_start,
                            "train_end": window.train_end,
                            "test_start": window.test_start,
                            "test_end": window.test_end,
                            "target": target_name,
                            "q_low": q_low,
                            "q_high": q_high,
                            "gate_flag": gate_flag,
                            "n_samples": len(y_eval),
                            **metrics,
                        }

                        for threshold, tm in tight_metrics.items():
                            key = _format_threshold(threshold)
                            row[f"precision_tight_{key}"] = tm["precision_tight"]
                            row[f"coverage_tight_{key}"] = tm["coverage_tight"]

                        per_window_rows.append(row)

                        key_base = (target_name, q_low, q_high, gate_flag)
                        store = sample_store.setdefault(
                            key_base, {"hits": [], "width_pct": [], "width": []}
                        )
                        lower = np.minimum(pred_low_eval, pred_high_eval)
                        upper = np.maximum(pred_low_eval, pred_high_eval)
                        hits = (y_eval >= lower) & (y_eval <= upper)
                        width = upper - lower
                        width_pct = width / np.where(close_eval == 0, np.nan, close_eval)
                        store["hits"].append(hits.astype(float))
                        store["width_pct"].append(width_pct)
                        store["width"].append(width)

                        if (q_low, q_high) == SUMMARY_QUANTILE_PAIR and not gate_flag:
                            for ts_val, hit_val, width_val in zip(ts_test[mask], hits, width_pct):
                                summary_rows.append(
                                    {
                                        "timestamp": ts_val,
                                        "target": target_name,
                                        "hit": float(hit_val),
                                        "width_pct": float(width_val),
                                    }
                                )

                        for threshold, confusion in confusions.items():
                            key = (target_name, q_low, q_high, gate_flag, threshold)
                            prev = confusion_totals.get(key)
                            if prev is None:
                                confusion_totals[key] = confusion
                            else:
                                confusion_totals[key] = ConfusionCounts(
                                    tp=prev.tp + confusion.tp,
                                    fp=prev.fp + confusion.fp,
                                    tn=prev.tn + confusion.tn,
                                    fn=prev.fn + confusion.fn,
                                )

            print(
                f"Window {window.window_id}: train {window.train_start} -> {window.train_end}, "
                f"test {window.test_start} -> {window.test_end}"
            )

    per_window_df = pd.DataFrame(per_window_rows)
    per_window_df.to_csv(OUTPUTS_DIR / "per_window_metrics.csv", index=False)

    aggregate_rows: List[Dict] = []
    for key, store in sample_store.items():
        target, q_low, q_high, gate_flag = key
        hits = np.concatenate(store["hits"]) if store["hits"] else np.array([])
        width_pct = np.concatenate(store["width_pct"]) if store["width_pct"] else np.array([])
        width = np.concatenate(store["width"]) if store["width"] else np.array([])

        agg_row = {
            "target": target,
            "q_low": q_low,
            "q_high": q_high,
            "gate_flag": gate_flag,
            "coverage": _safe_mean(hits),
            "width_pct_mean": _safe_mean(width_pct),
            "width_pct_median": float(np.median(width_pct)) if len(width_pct) else float("nan"),
            "width_mean": _safe_mean(width),
            "width_median": float(np.median(width)) if len(width) else float("nan"),
        }
        agg_row["coverage_gap"] = agg_row["coverage"] - nominal_coverages[(q_low, q_high)]

        for threshold in WIDTH_THRESHOLDS:
            confusion = confusion_totals.get((target, q_low, q_high, gate_flag, threshold))
            if confusion is None:
                continue
            key_name = _format_threshold(threshold)
            agg_row[f"precision_tight_{key_name}"] = confusion.precision
            agg_row[f"recall_tight_{key_name}"] = confusion.recall
            tight = per_window_df[
                (per_window_df["target"] == target)
                & (per_window_df["q_low"] == q_low)
                & (per_window_df["q_high"] == q_high)
                & (per_window_df["gate_flag"] == gate_flag)
            ]
            coverage_tight = _safe_mean(tight.get(f"coverage_tight_{key_name}", np.array([])).values)
            agg_row[f"coverage_tight_{key_name}"] = coverage_tight

        aggregate_rows.append(agg_row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUTS_DIR / "aggregate_metrics.csv", index=False)

    frontier_high = aggregate_df[aggregate_df["target"] == "high_reach"][
        ["q_low", "q_high", "gate_flag", "coverage", "width_pct_mean", "width_pct_median", "coverage_gap"]
    ]
    frontier_low = aggregate_df[aggregate_df["target"] == "low_reach"][
        ["q_low", "q_high", "gate_flag", "coverage", "width_pct_mean", "width_pct_median", "coverage_gap"]
    ]
    frontier_high.to_csv(OUTPUTS_DIR / "frontier_high_reach.csv", index=False)
    frontier_low.to_csv(OUTPUTS_DIR / "frontier_low_reach.csv", index=False)

    confusion_records: Dict[str, List[Dict]] = {"high_reach": [], "low_reach": []}
    for (target, q_low, q_high, gate_flag, threshold), confusion in confusion_totals.items():
        confusion_records[target].append(
            {
                "q_low": q_low,
                "q_high": q_high,
                "gate_flag": gate_flag,
                "width_threshold": threshold,
                "TP": confusion.tp,
                "FP": confusion.fp,
                "TN": confusion.tn,
                "FN": confusion.fn,
                "precision": confusion.precision,
                "recall": confusion.recall,
            }
        )

    for target, rows in confusion_records.items():
        if not rows:
            continue
        df_conf = pd.DataFrame(rows)
        for threshold in WIDTH_THRESHOLDS:
            key = _format_threshold(threshold)
            subset_all = df_conf[df_conf["width_threshold"] == threshold]
            subset_all.to_csv(OUTPUTS_DIR / f"confusion_{target}_W{key}.csv", index=False)

            subset = subset_all[
                (subset_all["q_low"] == SUMMARY_QUANTILE_PAIR[0])
                & (subset_all["q_high"] == SUMMARY_QUANTILE_PAIR[1])
            ]

            for gate_flag in [False, True]:
                gate_subset = subset[subset["gate_flag"] == gate_flag]
                if gate_subset.empty:
                    continue
                cm = np.array(
                    [
                        [gate_subset["TP"].sum(), gate_subset["FN"].sum()],
                        [gate_subset["FP"].sum(), gate_subset["TN"].sum()],
                    ],
                    dtype=int,
                )
                suffix = "gate" if gate_flag else ""
                plot_confusion_matrix(
                    cm,
                    x_labels=["Tight", "Wide"],
                    y_labels=["Hit", "Miss"],
                    title=(
                        f"{target.upper()} Tight vs Wide (W={threshold:.3f}, "
                        f"{'gate' if gate_flag else 'all'})"
                    ),
                    path=str(
                        FIGURES_DIR
                        / f"confusion_matrix_{target}_W{key}{'_' + suffix if suffix else ''}.png"
                    ),
                )

    plot_frontier(
        frontier_high.sort_values("width_pct_mean"),
        str(FIGURES_DIR / "coverage_vs_width_high_reach.png"),
        "Coverage vs Width (High Reach)",
    )
    plot_frontier(
        frontier_low.sort_values("width_pct_mean"),
        str(FIGURES_DIR / "coverage_vs_width_low_reach.png"),
        "Coverage vs Width (Low Reach)",
    )

    precision_plot_rows: List[Dict] = []
    for target in ["high_reach", "low_reach"]:
        subset = aggregate_df[aggregate_df["target"] == target]
        for gate_flag in [False, True]:
            sub_gate = subset[subset["gate_flag"] == gate_flag]
            for threshold in WIDTH_THRESHOLDS:
                key = _format_threshold(threshold)
                col = f"precision_tight_{key}"
                if col not in sub_gate.columns or sub_gate.empty:
                    continue
                precision_best = sub_gate[col].max()
                precision_plot_rows.append(
                    {
                        "target": target,
                        "gate_flag": gate_flag,
                        "width_threshold": threshold,
                        "precision_tight": precision_best,
                    }
                )

    precision_plot_df = pd.DataFrame(precision_plot_rows)
    plot_precision_tight(
        precision_plot_df,
        str(FIGURES_DIR / "tight_precision_vs_W.png"),
        "Precision (Tight) vs Width Threshold",
    )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["year"] = pd.to_datetime(summary_df["timestamp"]).dt.year
        summary_group = (
            summary_df.groupby(["year", "target"])
            .agg(coverage=("hit", "mean"), width_pct_mean=("width_pct", "mean"))
            .reset_index()
        )
        plot_yearly_summary(
            summary_group,
            str(FIGURES_DIR / "yearly_summary.png"),
        )

    if not aggregate_df.empty:
        plot_gate_vs_all(
            aggregate_df,
            str(FIGURES_DIR / "gate_vs_all_comparison.png"),
            "Gate vs All (Median Coverage)",
        )

    report_lines: List[str] = []
    report_lines.append("# Reachable Range Interval Report (12h, v1)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append(f"- Reach horizon (bars): {REACH_HORIZON}")
    report_lines.append(
        f"- Windows: train={WINDOW_CONFIGS[0].train_months}m, "
        f"test={WINDOW_CONFIGS[0].test_months}m, step={WINDOW_CONFIGS[0].step_months}m"
    )
    report_lines.append(f"- Features used: {len(feature_cols)} (from gate_module_12h_v1)")
    report_lines.append(f"- Quantile pairs: {QUANTILE_PAIRS}")
    report_lines.append(f"- Width thresholds: {WIDTH_THRESHOLDS}")
    report_lines.append(f"- Gate enabled: {USE_GATE}")
    report_lines.append(
        f"- Rows: raw={prep_counts['raw']}, after_clean={prep_counts['after_clean']}, "
        f"after_targets={prep_counts['after_targets']}, after_features={n_after_features}."
    )
    report_lines.append("")

    report_lines.append("## Aggregate Coverage + Width")
    if aggregate_df.empty:
        report_lines.append("- No windows produced.")
    else:
        report_lines.append("| Target | q_low | q_high | Gate | Coverage | Width% mean | Width% median | Gap vs nominal |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in aggregate_df.sort_values(["target", "q_low", "q_high", "gate_flag"]).iterrows():
            report_lines.append(
                f"| {row['target']} | {row['q_low']:.2f} | {row['q_high']:.2f} | "
                f"{'gate' if row['gate_flag'] else 'all'} | {row['coverage']:.3f} | "
                f"{row['width_pct_mean']:.3f} | {row['width_pct_median']:.3f} | "
                f"{row['coverage_gap']:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Tight Precision (Aggregate)")
    if aggregate_df.empty:
        report_lines.append("- No aggregate metrics available.")
    else:
        report_lines.append("| Target | q_low | q_high | Gate | " + " | ".join(
            [f"Prec@W{t:.3f}" for t in WIDTH_THRESHOLDS]
        ) + " |")
        report_lines.append("| --- | --- | --- | --- | " + " | ".join(["---"] * len(WIDTH_THRESHOLDS)) + " |")
        for _, row in aggregate_df.sort_values(["target", "q_low", "q_high", "gate_flag"]).iterrows():
            precs = []
            for threshold in WIDTH_THRESHOLDS:
                key = _format_threshold(threshold)
                precs.append(f"{row.get(f'precision_tight_{key}', float('nan')):.3f}")
            report_lines.append(
                f"| {row['target']} | {row['q_low']:.2f} | {row['q_high']:.2f} | "
                f"{'gate' if row['gate_flag'] else 'all'} | " + " | ".join(precs) + " |"
            )
    report_lines.append("")

    report_lines.append("## Best High-Precision Regimes")
    if not aggregate_df.empty:
        for target in ["high_reach", "low_reach"]:
            rows = aggregate_df[aggregate_df["target"] == target]
            best_desc = []
            for threshold in WIDTH_THRESHOLDS:
                key = _format_threshold(threshold)
                col = f"precision_tight_{key}"
                cov_col = f"coverage_tight_{key}"
                if col not in rows.columns or rows.empty:
                    continue
                viable = rows[rows[cov_col] >= MIN_COVERAGE_TIGHT]
                pick = viable.loc[viable[col].idxmax()] if not viable.empty else rows.loc[rows[col].idxmax()]
                best_desc.append(
                    f"W={threshold:.3f}: q=({pick['q_low']:.2f},{pick['q_high']:.2f}), "
                    f"{'gate' if pick['gate_flag'] else 'all'} "
                    f"(prec={pick[col]:.3f}, cov_tight={pick[cov_col]:.3f})"
                )
            if best_desc:
                report_lines.append(f"- {target}: " + "; ".join(best_desc) + ".")
    report_lines.append("")

    report_lines.append("## Commentary")
    report_lines.append("- Reachable range targets reduce path noise but do not guarantee high tight precision.")
    report_lines.append("- Compare tight-precision tables vs the original next-candle interval report.")
    report_lines.append("")

    report_lines.append("## Notes")
    excluded_timestamp = prep_counts["raw"] - prep_counts["after_clean"]
    excluded_targets = prep_counts["after_clean"] - prep_counts["after_targets"]
    excluded_features = prep_counts["after_targets"] - n_after_features
    report_lines.append(
        f"- Exclusions: timestamp/duplicates={excluded_timestamp}, "
        f"target tail={excluded_targets}, missing features={excluded_features}."
    )
    if missing_feature_counts.empty:
        report_lines.append("- No missing features after feature engineering.")
    else:
        top_missing = missing_feature_counts.head(10)
        missing_desc = ", ".join([f"{name} (n={int(count)})" for name, count in top_missing.items()])
        report_lines.append(f"- Top missing features: {missing_desc}.")
    if gate_skipped_windows:
        skipped = ", ".join([str(w) for w in gate_skipped_windows[:10]])
        report_lines.append(f"- Gate skipped on windows: {skipped}.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
