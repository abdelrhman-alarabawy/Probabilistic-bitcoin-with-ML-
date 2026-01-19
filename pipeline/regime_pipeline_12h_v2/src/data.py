from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union, List, Tuple

import numpy as np
import pandas as pd


TIMESTAMP_PATTERNS = ("timestamp", "nts", "time", "datetime")
LABEL_CANDIDATES = ("candle_type", "label")
EXCLUDE_PATTERNS = ("future", "target", "lead", "ahead", "t+")


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    features: pd.DataFrame
    feature_columns: List[str]
    timestamp_column: str
    label_column: str
    excluded_columns: List[str]


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(pat in lower for pat in patterns)


def detect_timestamp_column(columns: Iterable[str]) -> str:
    candidates = [col for col in columns if _matches_any(col, TIMESTAMP_PATTERNS)]
    if not candidates:
        raise ValueError("Could not detect timestamp column.")
    return candidates[0]


def detect_label_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if lower in LABEL_CANDIDATES or "candle_type" in lower:
            candidates.append(col)
    if not candidates:
        raise ValueError("Could not detect label column.")
    label_col = candidates[0]
    values = df[label_col].dropna().astype(str).str.lower().unique()
    allowed = {"long", "short", "skip"}
    if not set(values).issubset(allowed):
        raise ValueError(f"Label column {label_col} has unexpected values: {values}")
    return label_col


def load_and_clean(path: Union[str, Path]) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    timestamp_col = detect_timestamp_column(df.columns)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")
    df = df.reset_index(drop=True)
    label_col = detect_label_column(df)
    return df, timestamp_col, label_col


def build_base_features(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_shift: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_cols = [timestamp_col, label_col, "open", "high", "low", "close", "volume"]
    for col in df.columns:
        if col.lower() == "label_ambiguous":
            excluded_cols.append(col)
        if _matches_any(col, EXCLUDE_PATTERNS):
            excluded_cols.append(col)

    excluded_cols = list(dict.fromkeys(excluded_cols))
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]

    features = df[feature_cols].copy()
    if feature_shift > 0:
        features = features.shift(feature_shift)

    return features, feature_cols, excluded_cols


def drop_constant_columns(features: pd.DataFrame) -> pd.DataFrame:
    nunique = features.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    return features[keep_cols]
