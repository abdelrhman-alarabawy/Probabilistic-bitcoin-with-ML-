from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .data import build_base_features
from .features_extended import build_features_extended


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
