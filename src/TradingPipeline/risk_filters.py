from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import (
    ATR_PCT_Q,
    ATR_WINDOW,
    ENABLE_ATR_FILTER,
    ENABLE_RANGE_FILTER,
    ENABLE_VOLUME_FILTER,
    RANGE_Z_HI,
    RANGE_Z_LO,
    ROLLING_WINDOW,
    VOLUME_RATIO_Q,
)


@dataclass(frozen=True)
class RiskFilterCutoffs:
    vol_ratio_q: float | None
    range_z_lo: float | None
    range_z_hi: float | None
    atr_pct_q: float | None


def _rolling_median(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).median().shift(1)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean().shift(1)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std(ddof=0).shift(1)


def compute_vol_ratio(df: pd.DataFrame) -> pd.Series:
    rolling_med = _rolling_median(df["volume"], ROLLING_WINDOW)
    return df["volume"] / rolling_med.replace(0, np.nan)


def compute_range_z(df: pd.DataFrame) -> pd.Series:
    range_pct = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    mean = _rolling_mean(range_pct, ROLLING_WINDOW)
    std = _rolling_std(range_pct, ROLLING_WINDOW).replace(0, np.nan)
    return (range_pct - mean) / std


def compute_atr_pct(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=ATR_WINDOW, min_periods=1).mean().shift(1)
    return atr / df["close"].replace(0, np.nan)


def fit_risk_filter_cutoffs(df_train: pd.DataFrame) -> RiskFilterCutoffs:
    vol_ratio = compute_vol_ratio(df_train)
    range_z = compute_range_z(df_train)
    atr_pct = compute_atr_pct(df_train)

    vol_ratio_q = float(np.nanquantile(vol_ratio, VOLUME_RATIO_Q)) if ENABLE_VOLUME_FILTER else None
    range_z_lo = float(np.nanquantile(range_z, RANGE_Z_LO)) if ENABLE_RANGE_FILTER else None
    range_z_hi = float(np.nanquantile(range_z, RANGE_Z_HI)) if ENABLE_RANGE_FILTER else None
    atr_pct_q = float(np.nanquantile(atr_pct, ATR_PCT_Q)) if ENABLE_ATR_FILTER else None

    return RiskFilterCutoffs(
        vol_ratio_q=vol_ratio_q,
        range_z_lo=range_z_lo,
        range_z_hi=range_z_hi,
        atr_pct_q=atr_pct_q,
    )


def apply_risk_filters(df: pd.DataFrame, cutoffs: RiskFilterCutoffs) -> pd.DataFrame:
    vol_ratio = compute_vol_ratio(df)
    range_z = compute_range_z(df)
    atr_pct = compute_atr_pct(df)

    vol_pass = (
        vol_ratio >= cutoffs.vol_ratio_q
        if ENABLE_VOLUME_FILTER and cutoffs.vol_ratio_q is not None
        else pd.Series(True, index=df.index)
    )
    range_pass = (
        (range_z >= cutoffs.range_z_lo) & (range_z <= cutoffs.range_z_hi)
        if ENABLE_RANGE_FILTER and cutoffs.range_z_lo is not None and cutoffs.range_z_hi is not None
        else pd.Series(True, index=df.index)
    )
    atr_pass = (
        atr_pct <= cutoffs.atr_pct_q
        if ENABLE_ATR_FILTER and cutoffs.atr_pct_q is not None
        else pd.Series(True, index=df.index)
    )

    combined = vol_pass & range_pass & atr_pass

    return pd.DataFrame(
        {
            "vol_ratio": vol_ratio,
            "vol_ratio_pass": vol_pass,
            "range_z": range_z,
            "range_z_pass": range_pass,
            "atr_pct": atr_pct,
            "atr_pct_pass": atr_pass,
            "risk_pass": combined,
        },
        index=df.index,
    )


def risk_filter_cutoffs_table(cutoffs: RiskFilterCutoffs) -> Dict[str, float | None]:
    return {
        "vol_ratio_q": cutoffs.vol_ratio_q,
        "range_z_lo": cutoffs.range_z_lo,
        "range_z_hi": cutoffs.range_z_hi,
        "atr_pct_q": cutoffs.atr_pct_q,
    }
