from __future__ import annotations

from typing import Dict, List

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
    USE_GATE,
    WEIGHT_GRID_STEP,
    WIDTH_THRESHOLDS,
    WINDOW_CONFIGS,
    MAX_KNN_WEIGHT,
)
from .cqr_ensemble import (
    available_models,
    build_weight_grid,
    fit_cqr_ensemble,
    predict_cqr,
)
from .data import load_and_prepare_with_counts
from .eval import ConfusionCounts, summarize_interval
from .features import build_feature_matrix, build_preprocessor, load_features_used, select_features_from_used
from .gate import build_preprocessor as build_gate_preprocessor
from .gate import predict_trade_prob, select_topk, train_gate_model
from .plots import plot_confusion_matrix, plot_gate_vs_all, plot_timeline
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


def _format_threshold(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _price_width_from_pct(
    close: np.ndarray,
    low_pct: np.ndarray,
    high_pct: np.ndarray,
    target: str,
) -> np.ndarray:
    if target == "high_reach":
        price_lo = close * (1.0 + low_pct)
        price_hi = close * (1.0 + high_pct)
        return price_hi - price_lo
    price_lo = close * (1.0 - high_pct)
    price_hi = close * (1.0 - low_pct)
    return price_hi - price_lo


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

    missing_feature_counts = features.isna().sum()
    missing_feature_counts = missing_feature_counts[missing_feature_counts > 0].sort_values(ascending=False)

    features_used = load_features_used(GATE_FEATURES_PATH)
    features = select_features_from_used(features, features_used)
    feature_cols = features.columns.tolist()

    per_window_rows: List[Dict] = []
    confusion_totals: Dict[Tuple[str, bool, str], ConfusionCounts] = {}
    sample_store: Dict[Tuple[str, bool], Dict[str, List[np.ndarray]]] = {}
    gate_skipped_windows: List[int] = []

    available = available_models()
    model_names = [name for name in ["lightgbm", "xgboost", "catboost"] if available.get(name, False)]
    model_names.append("knn")
    weight_grid = build_weight_grid(model_names, WEIGHT_GRID_STEP, MAX_KNN_WEIGHT)

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

            close_test = df["close"].iloc[test_idx].values
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

            for target_name, y_tr, y_cal, y_test in [
                ("high_reach", y_high_tr, y_high_cal, y_high_test),
                ("low_reach", y_low_tr, y_low_cal, y_low_test),
            ]:
                cqr_state = fit_cqr_ensemble(
                    X_train_tr_proc,
                    y_tr,
                    X_train_cal_proc,
                    y_cal,
                    ALPHA_LOW,
                    ALPHA_HIGH,
                    weight_grid,
                    target_coverage=TARGET_COVERAGE,
                )

                pred_low, pred_high = predict_cqr(cqr_state, X_test_proc, ALPHA_LOW, ALPHA_HIGH)

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
                        WIDTH_THRESHOLDS,
                        TARGET_COVERAGE,
                    )
                    width_price = _price_width_from_pct(close_eval, pred_low_eval, pred_high_eval, target_name)
                    metrics["width_price_mean"] = _safe_mean(width_price)
                    metrics["width_price_median"] = float(np.median(width_price)) if len(width_price) else float("nan")

                    row = {
                        "window_id": window.window_id,
                        "config_name": window.config_name,
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                        "target": target_name,
                        "gate_flag": gate_flag,
                        "n_samples": len(y_eval),
                        "qhat": cqr_state.qhat,
                        "cal_coverage": cqr_state.coverage_cal,
                        "cal_width": cqr_state.width_cal,
                        "weight_lgbm": cqr_state.weights.get("lightgbm", 0.0),
                        "weight_xgb": cqr_state.weights.get("xgboost", 0.0),
                        "weight_cat": cqr_state.weights.get("catboost", 0.0),
                        "weight_knn": cqr_state.weights.get("knn", 0.0),
                        **metrics,
                    }

                    for key, tm in tight_metrics.items():
                        row[f"precision_tight_{key}"] = tm["precision_tight"]
                        row[f"coverage_tight_{key}"] = tm["coverage_tight"]

                    per_window_rows.append(row)

                    store = sample_store.setdefault(
                        (target_name, gate_flag), {"hits": [], "width_pct": [], "width_price": []}
                    )
                    lower = np.minimum(pred_low_eval, pred_high_eval)
                    upper = np.maximum(pred_low_eval, pred_high_eval)
                    hits = (y_eval >= lower) & (y_eval <= upper)
                    width_pct = upper - lower
                    store["hits"].append(hits.astype(float))
                    store["width_pct"].append(width_pct)
                    store["width_price"].append(width_price)

                    for key, confusion in confusions.items():
                        conf_key = (target_name, gate_flag, key)
                        prev = confusion_totals.get(conf_key)
                        if prev is None:
                            confusion_totals[conf_key] = confusion
                        else:
                            confusion_totals[conf_key] = ConfusionCounts(
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
    per_window_df.to_csv(OUTPUTS_DIR / "per_window.csv", index=False)

    aggregate_rows: List[Dict] = []
    for (target, gate_flag), store in sample_store.items():
        hits = np.concatenate(store["hits"]) if store["hits"] else np.array([])
        width_pct = np.concatenate(store["width_pct"]) if store["width_pct"] else np.array([])
        width_price = np.concatenate(store["width_price"]) if store["width_price"] else np.array([])

        agg_row = {
            "target": target,
            "gate_flag": gate_flag,
            "coverage": _safe_mean(hits),
            "width_pct_mean": _safe_mean(width_pct),
            "width_pct_median": float(np.median(width_pct)) if len(width_pct) else float("nan"),
            "width_price_mean": _safe_mean(width_price),
            "width_price_median": float(np.median(width_price)) if len(width_price) else float("nan"),
            "coverage_gap": _safe_mean(hits) - TARGET_COVERAGE,
        }

        for key in ["p10", "p25"] + [f"W{value:.3f}" for value in WIDTH_THRESHOLDS]:
            conf = confusion_totals.get((target, gate_flag, key))
            if conf is None:
                continue
            agg_row[f"precision_tight_{key}"] = conf.precision
            agg_row[f"recall_tight_{key}"] = conf.recall
        aggregate_rows.append(agg_row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUTS_DIR / "aggregate.csv", index=False)

    for target in ["high_reach", "low_reach"]:
        conf_rows = []
        for key, conf in confusion_totals.items():
            t_name, gate_flag, threshold_key = key
            if t_name != target:
                continue
            conf_rows.append(
                {
                    "gate_flag": gate_flag,
                    "threshold": threshold_key,
                    "TP": conf.tp,
                    "FP": conf.fp,
                    "TN": conf.tn,
                    "FN": conf.fn,
                    "precision": conf.precision,
                    "recall": conf.recall,
                }
            )
        if conf_rows:
            df_conf = pd.DataFrame(conf_rows)
            df_conf.to_csv(OUTPUTS_DIR / f"confusion_{target}.csv", index=False)

    plot_timeline(
        per_window_df,
        "coverage",
        str(FIGURES_DIR / "coverage_timeline.png"),
        "Coverage by Window",
    )
    plot_timeline(
        per_window_df,
        "width_pct_mean",
        str(FIGURES_DIR / "width_timeline.png"),
        "Width% Mean by Window",
    )

    if not aggregate_df.empty:
        plot_gate_vs_all(
            aggregate_df,
            str(FIGURES_DIR / "gate_vs_all.png"),
            "Gate vs All (Median Coverage)",
        )

    for target in ["high_reach", "low_reach"]:
        key = (target, False, "p10")
        conf = confusion_totals.get(key)
        if conf is None:
            continue
        cm = np.array([[conf.tp, conf.fn], [conf.fp, conf.tn]], dtype=int)
        plot_confusion_matrix(
            cm,
            x_labels=["Tight", "Wide"],
            y_labels=["Hit", "Miss"],
            title=f"{target} Tight10 Confusion (all)",
            path=str(FIGURES_DIR / f"confusion_matrix_{target}.png"),
        )

    report_lines: List[str] = []
    report_lines.append("# Reachable Range Interval Report (12h, v2 CQR)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append(f"- Reach horizon (bars): {REACH_HORIZON}")
    report_lines.append(
        f"- Windows: train={WINDOW_CONFIGS[0].train_months}m, "
        f"test={WINDOW_CONFIGS[0].test_months}m, step={WINDOW_CONFIGS[0].step_months}m"
    )
    report_lines.append(f"- Features used: {len(feature_cols)} (from gate_module_12h_v1)")
    report_lines.append(
        f"- Coverage target: {TARGET_COVERAGE:.2f} "
        f"(alpha_low={ALPHA_LOW:.2f}, alpha_high={ALPHA_HIGH:.2f})"
    )
    report_lines.append(f"- Gate enabled: {USE_GATE}")
    report_lines.append(
        f"- Rows: raw={prep_counts['raw']}, after_clean={prep_counts['after_clean']}, "
        f"after_targets={prep_counts['after_targets']}."
    )
    report_lines.append("")

    report_lines.append("## Aggregate Coverage + Width")
    if aggregate_df.empty:
        report_lines.append("- No windows produced.")
    else:
        report_lines.append("| Target | Gate | Coverage | Width% mean | Width% median | Width price mean | Gap vs nominal |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in aggregate_df.sort_values(["target", "gate_flag"]).iterrows():
            report_lines.append(
                f"| {row['target']} | {'gate' if row['gate_flag'] else 'all'} | "
                f"{row['coverage']:.3f} | {row['width_pct_mean']:.3f} | "
                f"{row['width_pct_median']:.3f} | {row['width_price_mean']:.3f} | "
                f"{row['coverage_gap']:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Tight Precision")
    if aggregate_df.empty:
        report_lines.append("- No aggregate metrics available.")
    else:
        report_lines.append("| Target | Gate | Prec@p10 | Prec@p25 | " + " | ".join(
            [f"Prec@{w:.3f}" for w in WIDTH_THRESHOLDS]
        ) + " |")
        report_lines.append("| --- | --- | --- | --- | " + " | ".join(["---"] * len(WIDTH_THRESHOLDS)) + " |")
        for _, row in aggregate_df.sort_values(["target", "gate_flag"]).iterrows():
            precs = []
            for key in [f"W{w:.3f}" for w in WIDTH_THRESHOLDS]:
                precs.append(f"{row.get(f'precision_tight_{key}', float('nan')):.3f}")
            report_lines.append(
                f"| {row['target']} | {'gate' if row['gate_flag'] else 'all'} | "
                f"{row.get('precision_tight_p10', float('nan')):.3f} | "
                f"{row.get('precision_tight_p25', float('nan')):.3f} | "
                + " | ".join(precs)
                + " |"
            )
    report_lines.append("")

    report_lines.append("## Best High-Precision Regimes")
    if not aggregate_df.empty:
        for target in ["high_reach", "low_reach"]:
            rows = aggregate_df[aggregate_df["target"] == target]
            if rows.empty:
                continue
            desc = []
            for key in ["p10", "p25"]:
                col = f"precision_tight_{key}"
                if col not in rows.columns:
                    continue
                best = rows.loc[rows[col].idxmax()]
                desc.append(
                    f"{key}: {'gate' if best['gate_flag'] else 'all'} (prec={best[col]:.3f})"
                )
            if desc:
                report_lines.append(f"- {target}: " + "; ".join(desc) + ".")
    report_lines.append("")

    report_lines.append("## Commentary")
    report_lines.append(
        "- Reachable range targets smooth path noise; CQR calibration targets coverage but tight precision may remain low."
    )
    report_lines.append("")

    report_lines.append("## Notes")
    excluded_timestamp = prep_counts["raw"] - prep_counts["after_clean"]
    excluded_targets = prep_counts["after_clean"] - prep_counts["after_targets"]
    report_lines.append(
        f"- Exclusions: timestamp/duplicates={excluded_timestamp}, target tail={excluded_targets}."
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
    if not aggregate_df.empty:
        high_all = aggregate_df[(aggregate_df["target"] == "high_reach") & (~aggregate_df["gate_flag"])]
        high_gate = aggregate_df[(aggregate_df["target"] == "high_reach") & (aggregate_df["gate_flag"])]
        low_all = aggregate_df[(aggregate_df["target"] == "low_reach") & (~aggregate_df["gate_flag"])]
        low_gate = aggregate_df[(aggregate_df["target"] == "low_reach") & (aggregate_df["gate_flag"])]
        if not high_all.empty:
            print(f"High reach coverage (all): {high_all['coverage'].iloc[0]:.3f}")
        if not high_gate.empty:
            print(f"High reach coverage (gate): {high_gate['coverage'].iloc[0]:.3f}")
        if not low_all.empty:
            print(f"Low reach coverage (all): {low_all['coverage'].iloc[0]:.3f}")
        if not low_gate.empty:
            print(f"Low reach coverage (gate): {low_gate['coverage'].iloc[0]:.3f}")


if __name__ == "__main__":
    run()
