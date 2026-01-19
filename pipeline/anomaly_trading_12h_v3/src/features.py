from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    ADD_MISSING_FLAGS,
    INCLUDE_LIQ_DEFAULT,
    LOOKBACK_MISC,
    LOOKBACK_VOL,
)


def build_event_features(
    df: pd.DataFrame, include_liq: bool | None = None
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    include_liq = INCLUDE_LIQ_DEFAULT if include_liq is None else include_liq

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    ret_raw = close.pct_change()

    features = pd.DataFrame(index=df.index)
    features["return_1"] = ret_raw.shift(1)
    features["return_2"] = close.pct_change(2).shift(1)
    features["return_3"] = close.pct_change(3).shift(1)

    range_raw = (high - low) / close.replace(0, np.nan)
    features["range_pct"] = range_raw.shift(1)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr_raw = tr_components.max(axis=1)
    atr = tr_raw.rolling(LOOKBACK_VOL).mean()
    features["atr_14_pct"] = (atr / close).shift(1)

    features["volatility_14"] = ret_raw.rolling(LOOKBACK_VOL).std().shift(1)
    downside_ret = ret_raw.where(ret_raw < 0.0, 0.0)
    features["downside_vol_14"] = downside_ret.rolling(LOOKBACK_VOL).std().shift(1)

    roll_max = close.rolling(LOOKBACK_MISC).max()
    drawdown = close / roll_max - 1.0
    features["drawdown_lookback"] = drawdown.shift(1)

    ret_mean = ret_raw.rolling(LOOKBACK_MISC).mean()
    ret_std = ret_raw.rolling(LOOKBACK_MISC).std()
    zscore_ret = (ret_raw - ret_mean) / ret_std
    features["zscore_return_lookback"] = zscore_ret.shift(1)

    vol_mean = volume.rolling(LOOKBACK_MISC).mean()
    vol_std = volume.rolling(LOOKBACK_MISC).std()
    volume_zscore = (volume - vol_mean) / vol_std
    features["volume_zscore"] = volume_zscore.shift(1)
    if ADD_MISSING_FLAGS:
        features["volume_zscore_missing"] = volume_zscore.shift(1).isna().astype(int)

    liq_cols: List[str] = []
    liq_flag_cols: List[str] = []
    if include_liq:
        liq_cols = [col for col in df.columns if col.startswith("liq_")]
        for col in liq_cols:
            series = df[col].astype(float)
            features[col] = series.shift(1)
            if ADD_MISSING_FLAGS:
                flag_col = f"{col}_missing"
                features[flag_col] = series.shift(1).isna().astype(int)
                liq_flag_cols.append(flag_col)

    base_features = [
        "return_1",
        "return_2",
        "return_3",
        "range_pct",
        "atr_14_pct",
        "volatility_14",
        "downside_vol_14",
        "drawdown_lookback",
        "zscore_return_lookback",
        "volume_zscore",
    ]
    if ADD_MISSING_FLAGS:
        base_features.append("volume_zscore_missing")

    return features, base_features, liq_cols, liq_flag_cols


def compute_liq_missing(features: pd.DataFrame, liq_cols: List[str], liq_flag_cols: List[str]) -> pd.Series:
    if not liq_cols:
        return pd.Series(False, index=features.index)
    if liq_flag_cols:
        return features[liq_flag_cols].fillna(0).sum(axis=1) > 0
    return features[liq_cols].isna().any(axis=1)
