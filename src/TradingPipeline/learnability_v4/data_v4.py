from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ..config import DECISION_TIME, LABEL_COL
from ..features import build_feature_matrix
from ..io import load_5m_dataset, load_dataset
from ..microstructure_5m import (
    build_5m_index,
    compute_microstructure_features,
    compute_train_return_cutoff,
)
from ..risk_filters import apply_risk_filters, fit_risk_filter_cutoffs
from .config_v4 import (
    AMBIGUOUS_COLUMN,
    CHOP_PCTL,
    DATASET_PATH,
    FIVE_MIN_PATH,
    LABEL_COLUMN,
    MISSING_FRAC_MAX,
    REGIME_COL_CANDIDATES,
    TRAIN_FRAC,
    VAL_FRAC,
    WICKINESS_PCTL,
)


@dataclass
class SplitIndex:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


@dataclass
class DataBundle:
    df: pd.DataFrame
    splits: SplitIndex
    features: pd.DataFrame
    feature_cols: list[str]
    micro_cols: list[str]
    imputer: SimpleImputer
    risk_df: pd.DataFrame
    direction_train_mask: pd.Series
    direction_quality_mask: pd.Series
    micro_cutoffs: dict
    regime_col: Optional[str]


def _align_labels_for_close(df: pd.DataFrame) -> pd.DataFrame:
    if DECISION_TIME != "close":
        return df
    shifted = df.copy()
    shifted[LABEL_COL] = shifted[LABEL_COL].shift(-1)
    if AMBIGUOUS_COLUMN in shifted.columns:
        shifted[AMBIGUOUS_COLUMN] = shifted[AMBIGUOUS_COLUMN].shift(-1)
    return shifted.iloc[:-1].reset_index(drop=True)


def _chronological_split(df: pd.DataFrame) -> SplitIndex:
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)
    return SplitIndex(
        train_idx=df.index[:train_end],
        val_idx=df.index[train_end:val_end],
        test_idx=df.index[val_end:],
    )


def _detect_regime_column(df: pd.DataFrame) -> Optional[str]:
    for col in REGIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _compute_direction_quality(
    micro_features: pd.DataFrame, splits: SplitIndex
) -> tuple[pd.Series, dict]:
    wickiness = micro_features["ms_wickiness_mean"]
    chop_proxy = 1.0 - micro_features["ms_body_to_range_mean"]
    missing_frac = micro_features["ms_missing_frac"]

    train_idx = splits.train_idx
    wick_cut = float(wickiness.loc[train_idx].quantile(WICKINESS_PCTL))
    chop_cut = float(chop_proxy.loc[train_idx].quantile(CHOP_PCTL))
    if np.isnan(wick_cut):
        wick_cut = float("inf")
    if np.isnan(chop_cut):
        chop_cut = float("inf")

    quality_mask = (wickiness <= wick_cut) & (chop_proxy <= chop_cut) & (
        missing_frac <= MISSING_FRAC_MAX
    )

    return quality_mask, {
        "wickiness_pctl": WICKINESS_PCTL,
        "wickiness_cutoff": wick_cut,
        "chop_pctl": CHOP_PCTL,
        "chop_cutoff": chop_cut,
        "missing_frac_max": MISSING_FRAC_MAX,
    }


def load_data_bundle() -> DataBundle:
    load = load_dataset(DATASET_PATH)
    df = _align_labels_for_close(load.df)

    splits = _chronological_split(df)

    df5 = load_5m_dataset(FIVE_MIN_PATH)
    idx5 = build_5m_index(df5)
    train_end_ts = df.loc[splits.train_idx[-1], "timestamp"]
    ret_cutoff = compute_train_return_cutoff(df5, train_end_ts)

    vol_scale = df["volume"].rolling(window=30 * 24, min_periods=1).median().shift(1)
    micro_features, _ = compute_microstructure_features(
        df["timestamp"], idx5, vol_scale=vol_scale, ret_cutoff=ret_cutoff
    )
    micro_cols = list(micro_features.columns)

    direction_quality_mask, micro_cutoffs = _compute_direction_quality(
        micro_features, splits
    )
    micro_cutoffs["ret_cutoff"] = ret_cutoff

    risk_cutoffs = fit_risk_filter_cutoffs(df.loc[splits.train_idx])
    risk_df = apply_risk_filters(df, risk_cutoffs)

    ambiguous_ok = pd.Series(True, index=df.index)
    if AMBIGUOUS_COLUMN in df.columns:
        ambiguous_ok = ~df[AMBIGUOUS_COLUMN].fillna(False)

    direction_train_mask = (
        df[LABEL_COLUMN].isin(["long", "short"])
        & risk_df["risk_pass"]
        & direction_quality_mask
        & ambiguous_ok
    )

    train_features = build_feature_matrix(
        df.loc[splits.train_idx],
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
        micro_df=micro_features,
        micro_cols=micro_cols,
    )
    base_feature_cols = [col for col in train_features.feature_cols if col not in micro_cols]
    feature_cols = base_feature_cols + micro_cols
    imputer = train_features.imputer

    full_features = build_feature_matrix(
        df,
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
        imputer=imputer,
        feature_cols=base_feature_cols,
        micro_df=micro_features,
        micro_cols=micro_cols,
    )

    regime_col = _detect_regime_column(df)

    return DataBundle(
        df=df,
        splits=splits,
        features=full_features.features,
        feature_cols=feature_cols,
        micro_cols=micro_cols,
        imputer=imputer,
        risk_df=risk_df,
        direction_train_mask=direction_train_mask,
        direction_quality_mask=direction_quality_mask,
        micro_cutoffs=micro_cutoffs,
        regime_col=regime_col,
    )
