from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .data import EXCLUDE_PATTERNS, _matches_any
from .features_extended import build_features_extended


def build_base_features(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_shift: int,
) -> Tuple[pd.DataFrame, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_cols = [timestamp_col, label_col, "open", "high", "low", "close", "volume"]
    for col in df.columns:
        if col.lower() == "label_ambiguous":
            excluded_cols.append(col)
        if col.lower().startswith("y_"):
            excluded_cols.append(col)
        if _matches_any(col, EXCLUDE_PATTERNS):
            excluded_cols.append(col)
    excluded_cols = list(dict.fromkeys(excluded_cols))
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    features = df[feature_cols].copy()
    if feature_shift > 0:
        features = features.shift(feature_shift)
    return features, excluded_cols


def drop_constant_columns(features: pd.DataFrame) -> pd.DataFrame:
    nunique = features.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    return features[keep_cols]


def drop_missing_columns(features: pd.DataFrame, max_missing: float) -> pd.DataFrame:
    missing_frac = features.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing].index.tolist()
    return features[keep_cols]


def drop_high_corr(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if features.empty:
        return features
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return features.drop(columns=drop_cols)


def build_feature_matrix(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_shift: int,
    missingness_max: float,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_features, excluded_cols = build_base_features(
        df,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_shift=feature_shift,
    )
    extra_features, _ = build_features_extended(df, feature_shift=feature_shift)
    features = pd.concat([base_features, extra_features], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]

    features = drop_missing_columns(features, missingness_max)
    features = drop_constant_columns(features)
    features = drop_high_corr(features, corr_threshold)
    feature_cols = features.columns.tolist()
    return features, feature_cols, excluded_cols


def load_features_used(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_features_from_used(features: pd.DataFrame, features_used: List[str]) -> pd.DataFrame:
    missing = [col for col in features_used if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing[:10]}")
    return features[features_used]
