from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

TIMESTAMP_PATTERNS = ("timestamp", "nts", "time", "datetime")
LABEL_CANDIDATES = ("candle_type", "label")
EXCLUDE_PATTERNS = ("future", "target", "lead", "ahead", "t+")


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(pat in lower for pat in patterns)


def detect_timestamp_column(columns: Iterable[str]) -> str:
    candidates = [col for col in columns if _matches_any(col, TIMESTAMP_PATTERNS)]
    if not candidates:
        raise ValueError("Could not detect timestamp column.")
    return candidates[0]


def detect_label_column(columns: Iterable[str]) -> str:
    for col in columns:
        lower = col.lower()
        if lower in LABEL_CANDIDATES or "candle_type" in lower:
            return col
    raise ValueError("Could not detect label column.")


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")
    return df.reset_index(drop=True)


def add_reachable_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    if horizon < 1:
        raise ValueError("Reachable horizon must be >= 1.")
    high_shift = df["high"].shift(-1)
    low_shift = df["low"].shift(-1)
    high_reach = high_shift.rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    low_reach = low_shift.rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    df["high_reach_price"] = high_reach
    df["low_reach_price"] = low_reach
    df["y_high"] = (high_reach / df["close"]) - 1.0
    df["y_low"] = 1.0 - (low_reach / df["close"])
    df["y_range_reach"] = high_reach - low_reach
    df = df.dropna(subset=["y_high", "y_low"])
    return df.reset_index(drop=True)


def load_and_prepare_with_counts(
    path: str,
    horizon: int,
) -> Tuple[pd.DataFrame, str, str, dict]:
    df_raw = load_raw(path)
    timestamp_col = detect_timestamp_column(df_raw.columns)
    label_col = detect_label_column(df_raw.columns)
    n_raw = len(df_raw)
    df_clean = clean_timestamp(df_raw, timestamp_col)
    n_after_clean = len(df_clean)
    df_targets = add_reachable_targets(df_clean, horizon)
    n_after_targets = len(df_targets)
    counts = {
        "raw": n_raw,
        "after_clean": n_after_clean,
        "after_targets": n_after_targets,
    }
    return df_targets, timestamp_col, label_col, counts
