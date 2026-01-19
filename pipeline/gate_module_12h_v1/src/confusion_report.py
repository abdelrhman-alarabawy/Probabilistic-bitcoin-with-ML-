from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    CALIBRATION_METHOD,
    CALIBRATION_SPLITS,
    CONFUSION_AGG_FIG_PATH,
    CONFUSION_AGGREGATE_PATH,
    CONFUSION_BY_WINDOW_PATH,
    CONFUSION_FIGURES_DIR,
    CONFUSION_K,
    CONFUSION_SUMMARY_FIG_PATH,
    CORR_THRESHOLD,
    DATA_PATH,
    FEATURE_SHIFT,
    FIGURES_DIR,
    GATE_C,
    GATE_MAX_ITER,
    GATE_N_JOBS,
    GATE_SOLVER,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MISSINGNESS_MAX,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REPORT_PATH,
    WINDOW_CONFIGS,
)
from .data import LABEL_CANDIDATES, detect_timestamp_column
from .evaluate import ConfusionStats, compute_confusion, confusion_matrix_plot
from .features import build_feature_matrix
from .model import build_preprocessor, predict_trade_prob, train_gate_model
from .rolling import generate_rolling_windows


ALLOWED_LABELS = {"long", "short", "skip"}


def detect_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        lower = col.lower()
        if lower in LABEL_CANDIDATES or "candle_type" in lower:
            return col
    raise ValueError("Could not detect label column.")


def _mark_timestamp_issues(
    df: pd.DataFrame,
    timestamp_col: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    has_ts = ts.notna()
    if not has_ts.any():
        dup_mask = pd.Series(False, index=df.index)
        return ts, has_ts, dup_mask
    df_ts = df.loc[has_ts].copy()
    df_ts["_ts"] = ts.loc[has_ts].values
    df_ts = df_ts.sort_values("_ts")
    dup = df_ts.duplicated(subset="_ts", keep="last")
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.loc[df_ts.index] = dup.values
    return ts, has_ts, dup_mask


def _confusion_trade_notrade(stats: ConfusionStats) -> np.ndarray:
    return np.array([[stats.tp, stats.fn], [stats.fp, stats.tn]], dtype=int)


def _write_window_figures(
    rows: List[Dict],
    figures_dir: Path,
) -> None:
    if not rows:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        cm = np.array([[row["TP"], row["FN"]], [row["FP"], row["TN"]]], dtype=int)
        title = f"Trade vs Notrade Confusion (Window {row['window_id']})"
        path = str(figures_dir / f"confusion_window_{row['window_id']}.png")
        confusion_matrix_plot(cm=cm, labels=["Trade", "Notrade"], title=title, path=path)


def _write_summary_figure(rows: List[Dict], figures_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("window_id")
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(df["window_id"], df["precision"], marker="o")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Precision by Window (Top-K=20)")

    axes[1].bar(df["window_id"], df["N_cm_used"], color="tab:blue")
    axes[1].set_ylabel("N_cm_used")
    axes[1].set_xlabel("Window ID")
    axes[1].set_title("Samples Used by Window")

    fig.tight_layout()
    figures_dir.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir, dpi=150)
    plt.close(fig)


def _insert_report_section(lines: List[str], section: List[str]) -> List[str]:
    header = "## Trade vs Not-Trade Confusion (Top-K=20)"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is not None:
        end = start + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        lines = lines[:start] + lines[end:]

    insert_at = None
    for i, line in enumerate(lines):
        if line.strip() == "## Conclusion":
            insert_at = i
            break
    if insert_at is None:
        insert_at = len(lines)
    return lines[:insert_at] + section + [""] + lines[insert_at:]


def _compute_missing_feature_report(features: pd.DataFrame, top_n: int = 10) -> List[str]:
    missing_counts = features.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_counts.empty:
        return ["- No missing features after feature engineering."]
    top = missing_counts.head(top_n)
    formatted = ", ".join([f"{name} (n={int(count)})" for name, count in top.items()])
    return [f"- Top missing features: {formatted}."]


def _compute_confusion_safe(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionStats:
    if len(y_true) == 0:
        return ConfusionStats(tp=0, fp=0, tn=0, fn=0)
    return compute_confusion(y_true, y_pred)


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    timestamp_col = detect_timestamp_column(df_raw.columns)
    label_col = detect_label_column(df_raw)

    ts, has_ts, dup_mask = _mark_timestamp_issues(df_raw, timestamp_col)
    df_raw = df_raw.copy()
    df_raw[timestamp_col] = ts

    raw_missing_timestamp = int((~has_ts).sum())
    df_clean = df_raw.loc[has_ts & ~dup_mask].copy()
    df_clean = df_clean.sort_values(timestamp_col).reset_index(drop=True)

    label_series = df_clean[label_col]
    label_norm = label_series.astype(str).str.lower()
    label_missing = label_series.isna()
    label_valid = label_series.notna() & label_norm.isin(ALLOWED_LABELS)

    features, feature_cols, excluded_cols = build_feature_matrix(
        df_clean,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_shift=FEATURE_SHIFT,
        missingness_max=MISSINGNESS_MAX,
        corr_threshold=CORR_THRESHOLD,
    )

    features_used_path = OUTPUTS_DIR / "features_used.json"
    if not features_used_path.exists():
        raise FileNotFoundError(f"Missing features_used.json at {features_used_path}")
    with open(features_used_path, "r", encoding="utf-8") as f:
        features_used = json.load(f)
    missing_features = [col for col in features_used if col not in features.columns]
    if missing_features:
        raise ValueError(f"Missing features from features_used.json: {missing_features[:10]}")
    features = features[features_used]

    feature_missing = features.isna().any(axis=1)
    valid_feature_mask = ~feature_missing

    df_valid = df_clean.loc[valid_feature_mask].reset_index(drop=True)
    features_valid = features.loc[valid_feature_mask].reset_index(drop=True)
    label_norm_valid = label_norm.loc[valid_feature_mask].reset_index(drop=True)
    label_valid_valid = label_valid.loc[valid_feature_mask].reset_index(drop=True)

    per_window_rows: List[Dict] = []
    k_shortfall_windows: List[int] = []
    mismatch_windows: List[int] = []

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df_valid,
                timestamp_col,
                config,
                min_train_rows=MIN_TRAIN_ROWS,
                min_test_rows=MIN_TEST_ROWS,
            )
        )
        for window in windows:
            test_start = window.test_start
            test_end = window.test_end

            raw_in_window = has_ts & (ts >= test_start) & (ts < test_end)
            n_test_raw = int(raw_in_window.sum())
            n_excluded_timestamp = int((raw_in_window & dup_mask).sum())

            clean_ts = df_clean[timestamp_col]
            clean_in_window = (clean_ts >= test_start) & (clean_ts < test_end)
            n_excluded_missing_label = int((clean_in_window & (~label_valid)).sum())
            n_excluded_missing_features = int(
                (clean_in_window & label_valid & feature_missing).sum()
            )

            n_scored = int((clean_in_window & label_valid & ~feature_missing).sum())
            n_excluded_other = n_test_raw - (
                n_excluded_timestamp + n_excluded_missing_label + n_excluded_missing_features + n_scored
            )
            if n_excluded_other < 0:
                n_excluded_other = 0
            n_excluded_total = (
                n_excluded_timestamp + n_excluded_missing_label + n_excluded_missing_features + n_excluded_other
            )
            n_cm_used = n_scored

            train_idx = window.train_idx
            test_idx = window.test_idx

            train_mask = label_valid_valid.iloc[train_idx].values
            test_mask = label_valid_valid.iloc[test_idx].values

            X_train = features_valid.iloc[train_idx].values[train_mask]
            X_test = features_valid.iloc[test_idx].values[test_mask]
            y_train = label_norm_valid.iloc[train_idx].isin(["long", "short"]).astype(int).values[train_mask]
            y_test = label_norm_valid.iloc[test_idx].isin(["long", "short"]).astype(int).values[test_mask]

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            model_result = train_gate_model(
                X_train_proc,
                y_train,
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
                p_trade_test = np.full(len(y_test), np.nan)

            k_effective = min(CONFUSION_K, len(y_test))
            if len(y_test) < CONFUSION_K:
                k_shortfall_windows.append(window.window_id)
            selected = np.zeros(len(y_test), dtype=int)
            if k_effective > 0:
                p_trade_safe = np.where(np.isfinite(p_trade_test), p_trade_test, -np.inf)
                order = np.argsort(p_trade_safe)[::-1]
                selected[order[:k_effective]] = 1

            stats = _compute_confusion_safe(y_test, selected)
            precision = stats.precision
            recall = stats.recall
            fpr = stats.fpr

            if n_cm_used != len(y_test):
                mismatch_windows.append(window.window_id)

            per_window_rows.append(
                {
                    "window_id": window.window_id,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "N_test_raw": n_test_raw,
                    "N_excluded_total": n_excluded_total,
                    "N_scored": n_scored,
                    "K_effective": k_effective,
                    "N_cm_used": n_cm_used,
                    "TP": stats.tp,
                    "FP": stats.fp,
                    "TN": stats.tn,
                    "FN": stats.fn,
                    "precision": precision,
                    "recall": recall,
                    "FPR": fpr,
                    "N_excluded_timestamp": n_excluded_timestamp,
                    "N_excluded_missing_label": n_excluded_missing_label,
                    "N_excluded_missing_features": n_excluded_missing_features,
                    "N_excluded_other": n_excluded_other,
                    "N_pred_trade": k_effective,
                    "N_pred_notrade": n_cm_used - k_effective,
                }
            )

    confusion_df = pd.DataFrame(per_window_rows)
    confusion_df.to_csv(CONFUSION_BY_WINDOW_PATH, index=False)

    agg_stats = ConfusionStats(tp=0, fp=0, tn=0, fn=0)
    total_used = int(confusion_df["N_cm_used"].sum()) if not confusion_df.empty else 0
    total_excluded = int(confusion_df["N_excluded_total"].sum()) if not confusion_df.empty else 0
    total_raw = int(confusion_df["N_test_raw"].sum()) if not confusion_df.empty else 0
    total_ex_timestamp = int(confusion_df["N_excluded_timestamp"].sum()) if not confusion_df.empty else 0
    total_ex_label = int(confusion_df["N_excluded_missing_label"].sum()) if not confusion_df.empty else 0
    total_ex_features = int(confusion_df["N_excluded_missing_features"].sum()) if not confusion_df.empty else 0
    total_ex_other = int(confusion_df["N_excluded_other"].sum()) if not confusion_df.empty else 0

    if not confusion_df.empty:
        agg_stats = ConfusionStats(
            tp=int(confusion_df["TP"].sum()),
            fp=int(confusion_df["FP"].sum()),
            tn=int(confusion_df["TN"].sum()),
            fn=int(confusion_df["FN"].sum()),
        )

    aggregate_precision = agg_stats.precision
    aggregate_recall = agg_stats.recall
    aggregate_fpr = agg_stats.fpr

    aggregate_df = pd.DataFrame(
        [
            {
                "TP": agg_stats.tp,
                "FP": agg_stats.fp,
                "TN": agg_stats.tn,
                "FN": agg_stats.fn,
                "precision": aggregate_precision,
                "recall": aggregate_recall,
                "FPR": aggregate_fpr,
                "total_used": total_used,
            }
        ]
    )
    aggregate_df.to_csv(CONFUSION_AGGREGATE_PATH, index=False)

    if not confusion_df.empty:
        cm = _confusion_trade_notrade(agg_stats)
        confusion_matrix_plot(
            cm=cm,
            labels=["Trade", "Notrade"],
            title="Trade vs Notrade Confusion (Aggregate)",
            path=str(CONFUSION_AGG_FIG_PATH),
        )

    _write_window_figures(per_window_rows, CONFUSION_FIGURES_DIR)
    _write_summary_figure(per_window_rows, CONFUSION_SUMMARY_FIG_PATH)

    report_lines = REPORT_PATH.read_text(encoding="utf-8").splitlines() if REPORT_PATH.exists() else []
    section = []
    section.append(f"## Trade vs Not-Trade Confusion (Top-K={CONFUSION_K})")
    section.append("")
    section.append("| Actual \\ Predicted | Trade | Notrade |")
    section.append("| --- | --- | --- |")
    section.append(f"| Trade | {agg_stats.tp} | {agg_stats.fn} |")
    section.append(f"| Notrade | {agg_stats.fp} | {agg_stats.tn} |")
    section.append("")
    section.append(
        f"- Aggregate precision: {aggregate_precision:.3f}, recall: {aggregate_recall:.3f}, "
        f"FPR: {aggregate_fpr:.3f}."
    )
    section.append(
        f"- Total used: {total_used}, total excluded: {total_excluded} "
        f"(raw test rows={total_raw})."
    )
    section.append(
        f"- Exclusions by reason: timestamp={total_ex_timestamp}, "
        f"missing label={total_ex_label}, missing features={total_ex_features}, other={total_ex_other}."
    )
    if raw_missing_timestamp > 0:
        section.append(
            f"- {raw_missing_timestamp} rows had missing timestamps and could not be assigned to windows."
        )
    if mismatch_windows:
        mismatch_desc = ", ".join(str(w) for w in mismatch_windows[:10])
        section.append(
            f"- N_scored differs from N_cm_used in windows: {mismatch_desc}."
        )
    else:
        section.append("- N_scored matches N_cm_used for all windows.")
    if k_shortfall_windows:
        shortfall_desc = ", ".join(str(w) for w in k_shortfall_windows[:10])
        section.append(f"- K_effective < 20 in windows: {shortfall_desc}.")
    section.extend(_compute_missing_feature_report(features))

    report_lines = _insert_report_section(report_lines, section)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    if confusion_df.empty:
        print("No windows produced.")
        return

    median_precision = float(confusion_df["precision"].median())
    worst_windows = (
        confusion_df.sort_values("precision")
        .head(3)[["window_id", "precision", "test_start", "test_end"]]
        .values
        .tolist()
    )

    print(
        "Aggregate confusion matrix (Trade vs Notrade): "
        f"TP={agg_stats.tp}, FP={agg_stats.fp}, TN={agg_stats.tn}, FN={agg_stats.fn}"
    )
    print(
        f"Total used={total_used}, total excluded={total_excluded}, raw test rows={total_raw}"
    )
    print(f"Median per-window precision={median_precision:.3f}")
    worst_desc = ", ".join(
        [f"window {w[0]} (precision={w[1]:.3f})" for w in worst_windows]
    )
    print(f"Worst 3 windows: {worst_desc}")


if __name__ == "__main__":
    run()
