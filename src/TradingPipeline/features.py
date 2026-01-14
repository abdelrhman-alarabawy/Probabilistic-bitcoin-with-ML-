from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import AMBIGUOUS_COL, DECISION_TIME, LABEL_COL, OHLCV_COLS


LEAKAGE_PATTERNS = ("t+1", "future", "next")


@dataclass
class FeatureArtifacts:
    features: pd.DataFrame
    feature_cols: list[str]
    imputer: SimpleImputer


def _is_datetime_like(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series)


def _drop_leakage_columns(columns: Iterable[str]) -> list[str]:
    keep = []
    for col in columns:
        col_l = col.lower()
        if any(token in col_l for token in LEAKAGE_PATTERNS):
            continue
        keep.append(col)
    return keep


def select_feature_columns(df: pd.DataFrame, timestamp_col: str) -> list[str]:
    excluded = {timestamp_col, LABEL_COL, AMBIGUOUS_COL}
    for col in df.columns:
        if col.lower() in ("date", "datetime"):
            excluded.add(col)

    candidate_cols = [c for c in df.columns if c not in excluded]
    candidate_cols = _drop_leakage_columns(candidate_cols)

    numeric_cols = []
    for col in candidate_cols:
        if _is_datetime_like(df[col]):
            continue
        if df[col].dtype == "O":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                numeric_cols.append(col)

    return numeric_cols


def lag_features(df: pd.DataFrame, feature_cols: list[str], decision_time: str) -> pd.DataFrame:
    shifted = df[feature_cols].copy()

    if decision_time not in ("close", "open"):
        raise ValueError("DECISION_TIME must be 'close' or 'open'.")

    for col in feature_cols:
        if decision_time == "close" and col in OHLCV_COLS:
            continue
        shifted[col] = shifted[col].shift(1)

    return shifted


def build_feature_matrix(
    df: pd.DataFrame,
    timestamp_col: str,
    decision_time: str = DECISION_TIME,
    imputer: Optional[SimpleImputer] = None,
    feature_cols: Optional[list[str]] = None,
    micro_df: Optional[pd.DataFrame] = None,
    micro_cols: Optional[list[str]] = None,
) -> FeatureArtifacts:
    if feature_cols is None:
        feature_cols = select_feature_columns(df, timestamp_col)
    base_features = lag_features(df, feature_cols, decision_time)
    for col in feature_cols:
        base_features[col] = pd.to_numeric(base_features[col], errors="coerce")

    if micro_df is not None:
        if micro_cols is None:
            micro_cols = list(micro_df.columns)
        micro_aligned = micro_df.reindex(df.index)
        micro_features = micro_aligned[micro_cols].copy()
        for col in micro_cols:
            micro_features[col] = pd.to_numeric(micro_features[col], errors="coerce")
        features = pd.concat([base_features, micro_features], axis=1)
        feature_cols = feature_cols + micro_cols
    else:
        features = base_features

    features = features.replace([np.inf, -np.inf], np.nan)

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(features)
    else:
        imputed = imputer.transform(features)

    features_imputed = pd.DataFrame(imputed, columns=feature_cols, index=df.index)

    return FeatureArtifacts(features=features_imputed, feature_cols=feature_cols, imputer=imputer)
