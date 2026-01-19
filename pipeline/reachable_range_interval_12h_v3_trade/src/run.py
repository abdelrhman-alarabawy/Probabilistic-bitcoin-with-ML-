from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    ALPHA_HIGH,
    ALPHA_LOW,
    CAL_FRACTION_MAX,
    CAL_FRACTION_MIN,
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
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REACH_HORIZON,
    REPORT_PATH,
    TARGET_COVERAGE,
    TIGHTNESS_LEVELS,
    USE_GATE,
    VOL_FEATURE_CANDIDATES,
    VOL_K,
    WINDOW_CONFIGS,
    LIQ_POLICY,
    MIN_TRADES_PER_WINDOW,
    WEIGHT_GRID_STEP,
    MAX_KNN_WEIGHT,
)
from .cqr_ensemble import available_models, build_weight_grid, fit_cqr_ensemble, predict_cqr
from .data import load_and_prepare_with_counts
from .eval import ConfusionCounts, summarize_interval
from .features import (
    apply_liquidity_policy,
    build_feature_matrix,
    build_preprocessor,
    load_features_used,
    select_features_from_used,
)
from .gate import build_preprocessor as build_gate_preprocessor
from .gate import predict_trade_prob, select_topk, train_gate_model
from .plots import plot_confusion_matrix
from .rolling import generate_rolling_windows


def _calibration_fraction(n_rows: int) -> float:
    if n_rows <= 0:
        return CAL_FRACTION_MIN
    if n_rows <= 800:
        return CAL_FRACTION_MAX
    if n_rows >= 2000:
        return CAL_FRACTION_MIN
    slope = (CAL_FRACTION_MIN - CAL_FRACTION_MAX) / (2000 - 800)
    return CAL_FRACTION_MAX + slope * (n_rows - 800)


def _calibration_split_idx(n: int) -> int:
    if n <= 1:
        return max(0, n - 1)
    frac = _calibration_fraction(n)
    cal_size = int(n * frac)
    cal_size = max(50, cal_size) if n >= 100 else max(1, cal_size)
    cal_size = min(cal_size, max(1, n - 50)) if n >= 100 else min(cal_size, n - 1)
    return n - cal_size


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _percentile_thresholds(values: np.ndarray) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if len(values) == 0:
        return {key: float("nan") for key in TIGHTNESS_LEVELS}
    for key in TIGHTNESS_LEVELS:
        pct = float(key.replace("p", ""))
        thresholds[key] = float(np.nanpercentile(values, pct))
    return thresholds


def _pick_vol_column(features: pd.DataFrame) -> str | None:
    for name in VOL_FEATURE_CANDIDATES:
        if name in features.columns:
            return name
    return None


def _trade_rule(
    rule: str,
    width_high: np.ndarray,
    width_low: np.ndarray,
    thr_high: float,
    thr_low: float,
    vol: np.ndarray | None,
) -> np.ndarray:
    if rule == "intersection":
        return (width_high <= thr_high) & (width_low <= thr_low)
    if rule == "either":
        return (width_high <= thr_high) | (width_low <= thr_low)
    if rule == "vol_adjusted":
        if vol is None:
            return (width_high <= thr_high) & (width_low <= thr_low)
        scale = 1.0 + VOL_K * vol
        return (width_high <= thr_high * scale) & (width_low <= thr_low * scale)
    raise ValueError(f"Unknown trade rule: {rule}")


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
    )

    features, added_flags, dropped_liq = apply_liquidity_policy(features)
    features_used = load_features_used(GATE_FEATURES_PATH)
    features, features_used_final = select_features_from_used(
        features, features_used, added_flags, dropped_liq
    )

    missing_feature_counts = features.isna().sum()
    missing_feature_counts = missing_feature_counts[missing_feature_counts > 0].sort_values(ascending=False)

    available = available_models()
    model_names = [name for name in ["lightgbm", "xgboost", "catboost"] if available.get(name, False)]
    model_names.append("knn")
    weight_grid = build_weight_grid(model_names, step=WEIGHT_GRID_STEP, max_knn=MAX_KNN_WEIGHT)

    per_window_interval_rows: List[Dict] = []
    per_window_trade_rows: List[Dict] = []
    threshold_rows: List[Dict] = []
    confusion_totals: Dict[Tuple[str, str, bool], ConfusionCounts] = {}

    rules = ["intersection", "either", "vol_adjusted"]
    vol_col = _pick_vol_column(features)

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

            y_high_train = df["y_high"].iloc[train_idx].values
            y_high_test = df["y_high"].iloc[test_idx].values
            y_low_train = df["y_low"].iloc[train_idx].values
            y_low_test = df["y_low"].iloc[test_idx].values

            actual_trade = df["label_norm"].iloc[test_idx].isin(["long", "short"]).astype(int).values

            split_idx = _calibration_split_idx(len(X_train))
            X_train_tr = X_train[:split_idx]
            X_train_cal = X_train[split_idx:]
            y_high_tr = y_high_train[:split_idx]
            y_high_cal = y_high_train[split_idx:]
            y_low_tr = y_low_train[:split_idx]
            y_low_cal = y_low_train[split_idx:]

            pre = build_preprocessor()
            X_train_tr_proc = pre.fit_transform(X_train_tr)
            X_train_cal_proc = pre.transform(X_train_cal)
            X_test_proc = pre.transform(X_test)
            vol_train = features.iloc[train_idx][vol_col].values if vol_col else None
            vol_test = features.iloc[test_idx][vol_col].values if vol_col else None
            if vol_col:
                vol_median = np.nanmedian(vol_train[:split_idx]) if len(vol_train[:split_idx]) else 0.0
                vol_test = np.where(np.isfinite(vol_test), vol_test, vol_median)

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
                if gate_mask is None:
                    gate_mask = np.zeros(len(test_idx), dtype=bool)

            thresholds_high: Dict[str, float] = {}
            thresholds_low: Dict[str, float] = {}

            cqr_high = fit_cqr_ensemble(
                X_train_tr_proc,
                y_high_tr,
                X_train_cal_proc,
                y_high_cal,
                ALPHA_LOW,
                ALPHA_HIGH,
                weight_grid,
                target_coverage=TARGET_COVERAGE,
            )
            cqr_low = fit_cqr_ensemble(
                X_train_tr_proc,
                y_low_tr,
                X_train_cal_proc,
                y_low_cal,
                ALPHA_LOW,
                ALPHA_HIGH,
                weight_grid,
                target_coverage=TARGET_COVERAGE,
            )

            high_cal_lo, high_cal_hi = predict_cqr(cqr_high, X_train_cal_proc, ALPHA_LOW, ALPHA_HIGH)
            low_cal_lo, low_cal_hi = predict_cqr(cqr_low, X_train_cal_proc, ALPHA_LOW, ALPHA_HIGH)
            width_high_cal = np.maximum(high_cal_hi, high_cal_lo) - np.minimum(high_cal_hi, high_cal_lo)
            width_low_cal = np.maximum(low_cal_hi, low_cal_lo) - np.minimum(low_cal_hi, low_cal_lo)
            thresholds_high = _percentile_thresholds(width_high_cal)
            thresholds_low = _percentile_thresholds(width_low_cal)

            for key in TIGHTNESS_LEVELS:
                threshold_rows.append(
                    {
                        "window_id": window.window_id,
                        "target": "high_reach",
                        "threshold_key": key,
                        "threshold_value": thresholds_high[key],
                    }
                )
                threshold_rows.append(
                    {
                        "window_id": window.window_id,
                        "target": "low_reach",
                        "threshold_key": key,
                        "threshold_value": thresholds_low[key],
                    }
                )

            high_test_lo, high_test_hi = predict_cqr(cqr_high, X_test_proc, ALPHA_LOW, ALPHA_HIGH)
            low_test_lo, low_test_hi = predict_cqr(cqr_low, X_test_proc, ALPHA_LOW, ALPHA_HIGH)
            width_high_test = np.maximum(high_test_hi, high_test_lo) - np.minimum(high_test_hi, high_test_lo)
            width_low_test = np.maximum(low_test_hi, low_test_lo) - np.minimum(low_test_hi, low_test_lo)

            for gate_flag, mask in [
                (False, np.ones(len(test_idx), dtype=bool)),
                (True, gate_mask),
            ]:
                y_high_eval = y_high_test[mask]
                y_low_eval = y_low_test[mask]
                actual_trade_eval = actual_trade[mask]

                high_lo_eval = high_test_lo[mask]
                high_hi_eval = high_test_hi[mask]
                low_lo_eval = low_test_lo[mask]
                low_hi_eval = low_test_hi[mask]
                width_high_eval = width_high_test[mask]
                width_low_eval = width_low_test[mask]
                vol_eval = vol_test[mask] if vol_test is not None else None

                metrics_high, conf_high, tight_high = summarize_interval(
                    y_high_eval,
                    high_lo_eval,
                    high_hi_eval,
                    TARGET_COVERAGE,
                    thresholds_high,
                )
                metrics_low, conf_low, tight_low = summarize_interval(
                    y_low_eval,
                    low_lo_eval,
                    low_hi_eval,
                    TARGET_COVERAGE,
                    thresholds_low,
                )

                per_window_interval_rows.append(
                    {
                        "window_id": window.window_id,
                        "gate_flag": gate_flag,
                        "target": "high_reach",
                        **metrics_high,
                        **{f"precision_tight_{k}": v["precision_tight"] for k, v in tight_high.items()},
                        **{f"coverage_tight_{k}": v["coverage_tight"] for k, v in tight_high.items()},
                    }
                )
                per_window_interval_rows.append(
                    {
                        "window_id": window.window_id,
                        "gate_flag": gate_flag,
                        "target": "low_reach",
                        **metrics_low,
                        **{f"precision_tight_{k}": v["precision_tight"] for k, v in tight_low.items()},
                        **{f"coverage_tight_{k}": v["coverage_tight"] for k, v in tight_low.items()},
                    }
                )

                for thr_key in TIGHTNESS_LEVELS:
                    thr_high = thresholds_high[thr_key]
                    thr_low = thresholds_low[thr_key]
                    for rule in rules:
                        trade_pred = _trade_rule(
                            rule,
                            width_high_eval,
                            width_low_eval,
                            thr_high,
                            thr_low,
                            vol_eval if rule == "vol_adjusted" else None,
                        ).astype(int)
                        tp = int(np.sum((trade_pred == 1) & (actual_trade_eval == 1)))
                        fp = int(np.sum((trade_pred == 1) & (actual_trade_eval == 0)))
                        fn = int(np.sum((trade_pred == 0) & (actual_trade_eval == 1)))
                        tn = int(np.sum((trade_pred == 0) & (actual_trade_eval == 0)))
                        confusion = ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)
                        per_window_trade_rows.append(
                            {
                                "window_id": window.window_id,
                                "train_start": window.train_start,
                                "train_end": window.train_end,
                                "test_start": window.test_start,
                                "test_end": window.test_end,
                                "gate_flag": gate_flag,
                                "rule": rule,
                                "threshold_key": thr_key,
                                "threshold_high": thr_high,
                                "threshold_low": thr_low,
                                "TP": tp,
                                "FP": fp,
                                "TN": tn,
                                "FN": fn,
                                "precision": confusion.precision,
                                "recall": confusion.recall,
                                "fpr": confusion.fpr,
                                "n_trades": int(trade_pred.sum()),
                                "n_samples": int(len(actual_trade_eval)),
                            }
                        )
                        gate_label = bool(gate_flag)
                        prev = confusion_totals.get((rule, thr_key, gate_label))
                        if prev is None:
                            confusion_totals[(rule, thr_key, gate_label)] = confusion
                        else:
                            confusion_totals[(rule, thr_key, gate_label)] = ConfusionCounts(
                                tp=prev.tp + tp,
                                fp=prev.fp + fp,
                                tn=prev.tn + tn,
                                fn=prev.fn + fn,
                            )

    per_window_df = pd.DataFrame(per_window_trade_rows)
    per_window_df.to_csv(OUTPUTS_DIR / "trade_confusion_by_window.csv", index=False)

    interval_df = pd.DataFrame(per_window_interval_rows)
    interval_df.to_csv(OUTPUTS_DIR / "interval_metrics_by_window.csv", index=False)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUTPUTS_DIR / "tightness_thresholds.csv", index=False)

    aggregate_rows: List[Dict] = []
    for (rule, thr_key, gate_flag), conf in confusion_totals.items():
        subset = per_window_df[
            (per_window_df["rule"] == rule)
            & (per_window_df["threshold_key"] == thr_key)
            & (per_window_df["gate_flag"] == gate_flag)
        ]
        aggregate_rows.append(
            {
                "rule": rule,
                "threshold_key": thr_key,
                "gate_flag": gate_flag,
                "TP": conf.tp,
                "FP": conf.fp,
                "TN": conf.tn,
                "FN": conf.fn,
                "precision": conf.precision,
                "recall": conf.recall,
                "fpr": conf.fpr,
                "n_trades": int(subset["n_trades"].sum()) if not subset.empty else 0,
                "n_samples": int(subset["n_samples"].sum()) if not subset.empty else 0,
            }
        )
        cm = np.array([[conf.tp, conf.fn], [conf.fp, conf.tn]], dtype=int)
        plot_confusion_matrix(
            cm,
            x_labels=["Trade", "NoTrade"],
            y_labels=["Actual Trade", "Actual NoTrade"],
            title=f"{rule} {thr_key} ({'gate' if gate_flag else 'all'})",
            path=str(
                FIGURES_DIR
                / f"confusion_trade_{rule}_{thr_key}_{'gate' if gate_flag else 'all'}.png"
            ),
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUTS_DIR / "trade_confusion_aggregate.csv", index=False)

    recommendation = None
    if not per_window_df.empty:
        grouped = per_window_df.groupby(["rule", "threshold_key", "gate_flag"])["n_trades"].min().reset_index()
        eligible = grouped[grouped["n_trades"] >= MIN_TRADES_PER_WINDOW]
        merged = aggregate_df.merge(
            eligible[["rule", "threshold_key", "gate_flag"]], on=["rule", "threshold_key", "gate_flag"], how="inner"
        )
        if merged.empty:
            merged = aggregate_df.copy()
        if not merged.empty:
            best = merged.sort_values(["precision", "recall"], ascending=False).iloc[0]
            recommendation = best

    report_lines: List[str] = []
    report_lines.append("# Reachable Range Trade Report (12h, v3)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append(f"- Reach horizon (bars): {REACH_HORIZON}")
    report_lines.append(f"- Coverage target: {TARGET_COVERAGE:.2f}")
    report_lines.append(f"- LIQ_POLICY: {LIQ_POLICY}")
    report_lines.append(f"- Gate enabled: {USE_GATE}")
    report_lines.append(f"- Features used: {len(features_used_final)}")
    report_lines.append(f"- Volatility feature: {vol_col if vol_col else 'none'} (k={VOL_K:.2f})")
    report_lines.append(
        f"- Rows: raw={prep_counts['raw']}, after_clean={prep_counts['after_clean']}, "
        f"after_targets={prep_counts['after_targets']}."
    )
    report_lines.append("")
    report_lines.append("## Trade Rules")
    report_lines.append("- intersection: width_high<=thr_high AND width_low<=thr_low")
    report_lines.append("- either: width_high<=thr_high OR width_low<=thr_low")
    report_lines.append("- vol_adjusted: width_high<=thr_high*(1+k*vol) AND width_low<=thr_low*(1+k*vol)")
    report_lines.append("")

    report_lines.append("## Interval Coverage + Width (per target)")
    if interval_df.empty:
        report_lines.append("- No interval metrics available.")
    else:
        agg_interval = interval_df.groupby(["target", "gate_flag"]).agg(
            coverage=("coverage", "mean"),
            width_pct_mean=("width_pct_mean", "mean"),
            width_pct_median=("width_pct_median", "median"),
        ).reset_index()
        report_lines.append("| Target | Gate | Coverage | Width% mean | Width% median |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for _, row in agg_interval.iterrows():
            report_lines.append(
                f"| {row['target']} | {'gate' if row['gate_flag'] else 'all'} | "
                f"{row['coverage']:.3f} | {row['width_pct_mean']:.3f} | {row['width_pct_median']:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Tightness Thresholds (median over windows)")
    if threshold_df.empty:
        report_lines.append("- No thresholds available.")
    else:
        agg_thresh = threshold_df.groupby(["target", "threshold_key"])["threshold_value"].median().reset_index()
        report_lines.append("| Target | Threshold | Median value |")
        report_lines.append("| --- | --- | --- |")
        for _, row in agg_thresh.iterrows():
            report_lines.append(
                f"| {row['target']} | {row['threshold_key']} | {row['threshold_value']:.4f} |"
            )
    report_lines.append("")

    report_lines.append("## Tight Precision (intervals)")
    if interval_df.empty:
        report_lines.append("- No tight precision metrics.")
    else:
        report_lines.append("| Target | Gate | " + " | ".join(TIGHTNESS_LEVELS) + " |")
        report_lines.append("| --- | --- | " + " | ".join(["---"] * len(TIGHTNESS_LEVELS)) + " |")
        for target in ["high_reach", "low_reach"]:
            for gate_flag in [False, True]:
                subset = interval_df[(interval_df["target"] == target) & (interval_df["gate_flag"] == gate_flag)]
                if subset.empty:
                    continue
                vals = []
                for key in TIGHTNESS_LEVELS:
                    col = f"precision_tight_{key}"
                    vals.append(f"{_safe_mean(subset[col].values):.3f}" if col in subset else "nan")
                report_lines.append(
                    f"| {target} | {'gate' if gate_flag else 'all'} | " + " | ".join(vals) + " |"
                )
    report_lines.append("")

    report_lines.append("## Trade Confusion (Aggregate)")
    if aggregate_df.empty:
        report_lines.append("- No trade confusion available.")
    else:
        report_lines.append("| Rule | Thr | Gate | Precision | Recall | FPR | Trades |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in aggregate_df.sort_values(["precision", "recall"], ascending=False).iterrows():
            report_lines.append(
                f"| {row['rule']} | {row['threshold_key']} | {'gate' if row['gate_flag'] else 'all'} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['fpr']:.3f} | {row['n_trades']} |"
            )
    report_lines.append("")

    if recommendation is not None:
        report_lines.append("## Recommendation")
        report_lines.append(
            f"- Best rule under MIN_TRADES={MIN_TRADES_PER_WINDOW}: "
            f"{recommendation['rule']} @ {recommendation['threshold_key']} "
            f"({'gate' if recommendation['gate_flag'] else 'all'}) "
            f"(precision={recommendation['precision']:.3f}, recall={recommendation['recall']:.3f})."
        )
        report_lines.append("")

    if not aggregate_df.empty:
        best_gate = aggregate_df[aggregate_df["gate_flag"]].sort_values(
            ["precision", "recall"], ascending=False
        ).head(1)
        best_all = aggregate_df[~aggregate_df["gate_flag"]].sort_values(
            ["precision", "recall"], ascending=False
        ).head(1)
        if not best_gate.empty and not best_all.empty:
            gate_prec = best_gate["precision"].iloc[0]
            all_prec = best_all["precision"].iloc[0]
            direction = "helps" if gate_prec > all_prec else "hurts"
            report_lines.append("## Gate Impact")
            report_lines.append(
                f"- Best precision gate={gate_prec:.3f}, all={all_prec:.3f}; gate {direction} precision."
            )
            report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("- actual_trade = candle_type in {long, short}; skip = no-trade.")
    if missing_feature_counts.empty:
        report_lines.append("- No missing features after feature engineering.")
    else:
        top_missing = missing_feature_counts.head(10)
        missing_desc = ", ".join([f"{name} (n={int(count)})" for name, count in top_missing.items()])
        report_lines.append(f"- Top missing features: {missing_desc}.")
    if dropped_liq:
        report_lines.append(f"- Dropped liq features: {len(dropped_liq)} columns.")
    if added_flags:
        report_lines.append(f"- Added liq missingness flags: {len(added_flags)} columns.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
