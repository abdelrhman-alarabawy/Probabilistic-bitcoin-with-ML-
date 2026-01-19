from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


LABEL_PATTERNS = (
    "label",
    "target",
    "candle_type",
    "class",
    "signal",
)

SUSPICIOUS_PATTERNS = (
    "future",
    "next",
    "t+",
    "lead",
    "ahead",
    "target",
    "label",
)

TIMESTAMP_PATTERNS = (
    "timestamp",
    "nts",
    "time",
    "datetime",
)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    features: pd.DataFrame
    feature_columns: list[str]
    timestamp_column: str
    excluded_columns: list[str]
    suspicious_columns: list[str]


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(pat in lower for pat in patterns)


def detect_timestamp_column(columns: Iterable[str]) -> str:
    candidates = [col for col in columns if _matches_any(col, TIMESTAMP_PATTERNS)]
    if not candidates:
        raise ValueError("Could not detect timestamp column.")
    return candidates[0]


def _label_like_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if _matches_any(col, LABEL_PATTERNS)]


def _suspicious_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if _matches_any(col, SUSPICIOUS_PATTERNS)]


def load_and_clean_data(path: Optional[Union[str, Path]]) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    timestamp_col = detect_timestamp_column(df.columns)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")
    df = df.reset_index(drop=True)
    return df, timestamp_col


def compute_future_returns(
    df: pd.DataFrame,
    horizon: int,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    if close_col not in df.columns:
        raise ValueError(f"Missing close column: {close_col}")
    close = df[close_col]
    future_return = close.shift(-horizon) / close - 1.0

    highs = pd.concat([df[high_col].shift(-k) for k in range(1, horizon + 1)], axis=1)
    lows = pd.concat([df[low_col].shift(-k) for k in range(1, horizon + 1)], axis=1)
    future_high = highs.max(axis=1)
    future_low = lows.min(axis=1)
    mfe = future_high / close - 1.0
    mae = future_low / close - 1.0

    return pd.DataFrame(
        {
            "future_return": future_return,
            "mfe": mfe,
            "mae": mae,
        }
    )


def build_dataset(
    path: str,
    horizon: int,
    feature_shift: int,
    enable_leakage_audit: bool = True,
) -> DatasetBundle:
    df, timestamp_col = load_and_clean_data(path)
    label_cols = _label_like_columns(df.columns)
    suspicious_cols = _suspicious_columns(df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    excluded_cols = set(label_cols + suspicious_cols + [timestamp_col])
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]

    future_df = compute_future_returns(df, horizon)
    df = pd.concat([df, future_df], axis=1)
    df["past_return"] = df["close"].pct_change()

    if feature_shift > 0:
        df[feature_cols] = df[feature_cols].shift(feature_shift)

    leakage_cols = []
    if enable_leakage_audit:
        for col in feature_cols:
            series = df[col]
            if series.isna().all():
                continue
            corr = series.corr(df["future_return"])
            if corr is not None and np.isfinite(corr) and abs(corr) > 0.99:
                leakage_cols.append(col)

    if leakage_cols:
        feature_cols = [col for col in feature_cols if col not in leakage_cols]
        excluded_cols.update(leakage_cols)

    df = df.dropna(subset=feature_cols + ["future_return", "mfe", "mae", "past_return"])
    df = df.reset_index(drop=True)

    features = df[feature_cols].copy()

    return DatasetBundle(
        df=df,
        features=features,
        feature_columns=feature_cols,
        timestamp_column=timestamp_col,
        excluded_columns=sorted(excluded_cols),
        suspicious_columns=sorted(set(suspicious_cols + leakage_cols)),
    )
