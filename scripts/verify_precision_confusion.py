from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# === CONFIGURATION ===
DATA_CSV = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv"
)
GMM_CSV = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_regimes_per_row.csv"
)
OUTPUT_DIR = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\validation\precision"
)

TP_POINTS = 2000
SL_POINTS = 1000
FEE_PER_TRADE = 0.0005
PROB_THRESHOLD = 0.8
RANDOM_SEED = 42

TRADABLE_REGIMES = {0, 2, 5, 6}
SKIP_REGIMES = {1, 3, 4, 7}

LABEL_CANDIDATES = ["candle_type", "label", "y_true", "target"]


def resolve_timestamp_col(columns: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in ["timestamp", "datetime", "time", "open_time", "ts_utc"]:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_timestamp(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return parse_numeric_timestamp(series)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().mean() > 0.05:
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().any():
            parsed = parse_numeric_timestamp(as_num)
    return parsed


def parse_numeric_timestamp(series: pd.Series) -> pd.Series:
    max_val = series.max()
    if max_val > 1e12:
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    if max_val > 1e10:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def find_label_column(columns: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def compute_y_true_from_next_candle(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    high_next = df["high"].shift(-1)
    low_next = df["low"].shift(-1)
    tp_level = df["close"] + TP_POINTS
    sl_level = df["close"] - SL_POINTS

    tp_hit = high_next >= tp_level
    sl_hit = low_next <= sl_level
    ambiguous = tp_hit & sl_hit

    y_true = np.where(tp_hit & ~sl_hit, "long", np.where(sl_hit & ~tp_hit, "short", "skip"))
    y_true = pd.Series(y_true, index=df.index)
    y_true[ambiguous] = "skip"
    y_true[high_next.isna() | low_next.isna()] = "skip"
    ambiguous = ambiguous.fillna(False)
    return y_true, ambiguous


def map_labels(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    raw = series.astype(str).str.lower().str.strip()
    ambiguous = ~raw.isin(["long", "short", "skip"])
    mapped = raw.where(raw.isin(["long", "short", "skip"]), "skip")
    return mapped, ambiguous


def get_direction_signal(_: pd.Series) -> Optional[str]:
    return None


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=labels,
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    summary_df = pd.DataFrame(
        {
            "precision": [macro[0], weighted[0]],
            "recall": [macro[1], weighted[1]],
            "f1": [macro[2], weighted[2]],
            "support": [support.sum(), support.sum()],
        },
        index=["macro_avg", "weighted_avg"],
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df, pd.concat([metrics_df, summary_df])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"DATA CSV not found: {DATA_CSV}")
    if not GMM_CSV.exists():
        raise FileNotFoundError(f"GMM CSV not found: {GMM_CSV}")

    df_data = pd.read_csv(DATA_CSV)
    df_data.columns = df_data.columns.str.strip()
    ts_col_data = resolve_timestamp_col(df_data.columns)
    if ts_col_data is None:
        raise ValueError("No timestamp column found in data CSV.")
    df_data = df_data.rename(columns={ts_col_data: "timestamp"})
    df_data["timestamp"] = parse_timestamp(df_data["timestamp"])
    df_data = df_data[df_data["timestamp"].notna()].copy()
    df_data = df_data.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df_data.columns:
            raise ValueError(f"Missing required column in data CSV: {col}")
        df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

    label_col = find_label_column(df_data.columns)
    if label_col:
        y_true_raw, ambiguous = map_labels(df_data[label_col])
        if "label_ambiguous" in df_data.columns:
            ambiguous = ambiguous | df_data["label_ambiguous"].astype(bool)
        y_true = y_true_raw
        label_source = f"label_column:{label_col}"
    else:
        y_true, ambiguous = compute_y_true_from_next_candle(df_data)
        label_source = "next_candle_tp_sl"

    df_data["y_true"] = y_true
    df_data["ambiguous"] = ambiguous

    df_gmm = pd.read_csv(GMM_CSV)
    df_gmm.columns = df_gmm.columns.str.strip()
    ts_col_gmm = resolve_timestamp_col(df_gmm.columns)
    if ts_col_gmm is not None:
        df_gmm = df_gmm.rename(columns={ts_col_gmm: "timestamp"})
        df_gmm["timestamp"] = parse_timestamp(df_gmm["timestamp"])
        df_gmm = df_gmm[df_gmm["timestamp"].notna()].copy()
        df_gmm = df_gmm.sort_values("timestamp").reset_index(drop=True)

    if "regime" not in df_gmm.columns and "regime_id" in df_gmm.columns:
        df_gmm = df_gmm.rename(columns={"regime_id": "regime"})

    if "prob_max" not in df_gmm.columns:
        if "regime_prob_max" in df_gmm.columns:
            df_gmm = df_gmm.rename(columns={"regime_prob_max": "prob_max"})
        elif "max_prob" in df_gmm.columns:
            df_gmm = df_gmm.rename(columns={"max_prob": "prob_max"})
        else:
            prob_cols = [
                c
                for c in df_gmm.columns
                if c.lower().startswith("prob_") or c.lower().startswith("regime_prob_")
            ]
            prob_cols = [c for c in prob_cols if c.lower() not in {"prob_max", "regime_prob_max"}]
            if prob_cols:
                df_gmm["prob_max"] = df_gmm[prob_cols].max(axis=1)
            else:
                raise ValueError("No probability column found in GMM CSV (prob_max/regime_prob_max).")

    if "regime" not in df_gmm.columns:
        raise ValueError("No regime column found in GMM CSV (regime/regime_id).")

    if "timestamp" in df_gmm.columns and "timestamp" in df_data.columns:
        data_ts = df_data["timestamp"]
        gmm_ts = df_gmm["timestamp"]
        missing_in_gmm = data_ts[~data_ts.isin(gmm_ts)]
        missing_in_data = gmm_ts[~gmm_ts.isin(data_ts)]
        mismatches = []
        if not missing_in_gmm.empty:
            missing_df = df_data[df_data["timestamp"].isin(missing_in_gmm)][
                ["timestamp", "close"]
            ].copy()
            missing_df["regime"] = np.nan
            missing_df["prob_max"] = np.nan
            missing_df["reason"] = "timestamp_missing_in_gmm"
            mismatches.append(missing_df)
        if not missing_in_data.empty:
            missing_df = df_gmm[df_gmm["timestamp"].isin(missing_in_data)][
                ["timestamp", "regime", "prob_max"]
            ].copy()
            missing_df["close"] = np.nan
            missing_df["reason"] = "timestamp_missing_in_data"
            mismatches.append(missing_df)
        mismatches_df = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()

        df = pd.merge(
            df_data,
            df_gmm[["timestamp", "regime", "prob_max"]],
            on="timestamp",
            how="inner",
        )
    else:
        if len(df_data) != len(df_gmm):
            raise ValueError("No timestamps available and row counts differ; cannot align by index.")
        mismatches_df = pd.DataFrame()
        df = df_data.copy()
        df["regime"] = df_gmm["regime"].values
        df["prob_max"] = df_gmm["prob_max"].values

    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df = df[df["regime"].notna()].copy()
    df["regime"] = df["regime"].astype(int)
    df["prob_max"] = pd.to_numeric(df["prob_max"], errors="coerce")

    is_tradable = df["regime"].isin(TRADABLE_REGIMES) & (df["prob_max"] >= PROB_THRESHOLD)
    reason = np.where(
        df["regime"].isin(TRADABLE_REGIMES),
        np.where(df["prob_max"] >= PROB_THRESHOLD, "tradable", "prob_below_threshold"),
        "regime_blocked",
    )

    y_true = df["y_true"].astype(str)
    rng = np.random.RandomState(RANDOM_SEED)
    rand_dirs = rng.choice(["long", "short"], size=len(df))

    y_pred_mode_a = np.where(is_tradable, y_true, "skip")
    y_pred_mode_b = np.where(is_tradable, rand_dirs, "skip")

    df["y_pred_modeA"] = y_pred_mode_a
    df["y_pred_modeB"] = y_pred_mode_b
    df["is_tradable"] = is_tradable
    df["reason"] = reason

    classes_full = ["long", "short", "skip"]
    classes_trade = ["long", "short"]

    def print_report(mode_name: str, y_pred: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cm_full, metrics_full = compute_metrics(y_true, y_pred, classes_full)
        trade_mask = y_pred != "skip"
        trade_y_true = y_true[trade_mask]
        trade_y_pred = y_pred[trade_mask]
        trade_mask_valid = trade_y_true.isin(classes_trade)
        trade_y_true = trade_y_true[trade_mask_valid]
        trade_y_pred = trade_y_pred[trade_mask_valid]
        cm_trade, metrics_trade = compute_metrics(trade_y_true, trade_y_pred, classes_trade)

        coverage = float((y_pred != "skip").mean())
        ambiguous_rate = float(df["ambiguous"].mean())

        print(f"\n=== {mode_name} ===")
        print("Confusion matrix (full):")
        print(cm_full)
        print("Metrics (full):")
        print(metrics_full)
        print("Confusion matrix (trade-only):")
        print(cm_trade)
        print("Metrics (trade-only):")
        print(metrics_trade)
        print(f"Coverage/trade rate: {coverage:.2%}")
        print(f"Ambiguous rate (y_true): {ambiguous_rate:.2%}")

        return cm_full, cm_trade

    print(f"Label source: {label_source}")
    cm_full_a, cm_trade_a = print_report("MODE A (oracle-direction)", df["y_pred_modeA"])
    cm_full_b, cm_trade_b = print_report("MODE B (random-direction)", df["y_pred_modeB"])

    cm_full_a.to_csv(OUTPUT_DIR / "confusion_full_modeA.csv")
    cm_full_b.to_csv(OUTPUT_DIR / "confusion_full_modeB.csv")
    cm_trade_a.to_csv(OUTPUT_DIR / "confusion_tradeonly_modeA.csv")
    cm_trade_b.to_csv(OUTPUT_DIR / "confusion_tradeonly_modeB.csv")

    debug_cols = [
        "timestamp",
        "close",
        "regime",
        "prob_max",
        "y_true",
        "y_pred_modeA",
        "y_pred_modeB",
        "is_tradable",
        "reason",
    ]
    debug_head = df[debug_cols].head(200).copy()
    if not mismatches_df.empty:
        mismatch_cols = ["timestamp", "close", "regime", "prob_max", "reason"]
        for col in debug_cols:
            if col not in mismatch_cols:
                mismatches_df[col] = np.nan
        mismatches_df = mismatches_df[debug_cols]
        debug_df = pd.concat([debug_head, mismatches_df], ignore_index=True)
    else:
        debug_df = debug_head
    debug_df.to_csv(OUTPUT_DIR / "debug_precision_sample.csv", index=False)

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
