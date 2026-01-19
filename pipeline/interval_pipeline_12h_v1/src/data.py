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


def add_targets(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    df = df.copy()
    df["y_high"] = df["high"].shift(-horizon)
    df["y_low"] = df["low"].shift(-horizon)
    df["y_close"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["y_high", "y_low"])
    return df.reset_index(drop=True)


def load_and_prepare(path: str, horizon: int) -> Tuple[pd.DataFrame, str, str]:
    df = load_raw(path)
    timestamp_col = detect_timestamp_column(df.columns)
    label_col = detect_label_column(df.columns)
    df = clean_timestamp(df, timestamp_col)
    df = add_targets(df, horizon)
    return df, timestamp_col, label_col
