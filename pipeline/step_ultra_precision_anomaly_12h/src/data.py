from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(
        drop=True
    )
    return df


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)
    df["forward_return_12h"] = close.shift(-1) / close - 1.0
    df["forward_return_24h"] = close.shift(-2) / close - 1.0
    df["forward_return_72h"] = close.shift(-6) / close - 1.0
    return df


def add_future_path_stats(df: pd.DataFrame, horizon_bars: int = 6) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    n = len(df)
    high_max = np.full(n, np.nan, dtype=float)
    low_min = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        start = idx + 1
        end = idx + horizon_bars + 1
        if end > n:
            continue
        high_max[idx] = float(np.nanmax(high[start:end]))
        low_min[idx] = float(np.nanmin(low[start:end]))

    mfe_long = high_max / close - 1.0
    mae_long = low_min / close - 1.0
    mfe_short = close / low_min - 1.0
    mae_short = close / high_max - 1.0

    df["mfe_long"] = mfe_long
    df["mae_long"] = mae_long
    df["mfe_short"] = mfe_short
    df["mae_short"] = mae_short
    return df


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None

    log_ret = np.log(close / close.shift(1))
    features["log_return_12h"] = log_ret
    features["range_pct"] = (high - low) / close.replace(0.0, np.nan)
    features["realized_vol_rolling_20"] = log_ret.rolling(20, min_periods=20).std()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    features["range_strength"] = (ema20 - ema50).abs() / close.replace(0.0, np.nan)

    if volume is not None:
        features["volume_z_rolling_20"] = _rolling_zscore(volume, window=20)

    spread_source = None
    if "spread_bps_last" in df.columns:
        spread_source = "spread_bps_last"
    elif "spread_mean" in df.columns:
        spread_source = "spread_mean"
    if spread_source:
        spread = df[spread_source].astype(float)
        features["spread_roll_mean"] = spread.rolling(20, min_periods=20).mean()
        features["spread_roll_std"] = spread.rolling(20, min_periods=20).std()

    imbalance_source = None
    if "imbalance_last" in df.columns:
        imbalance_source = "imbalance_last"
    elif "imbalance_mean" in df.columns:
        imbalance_source = "imbalance_mean"
    if imbalance_source:
        imbalance = df[imbalance_source].astype(float)
        features["imbalance_roll_mean"] = imbalance.rolling(20, min_periods=20).mean()
        features["imbalance_roll_std"] = imbalance.rolling(20, min_periods=20).std()

    if {"atm_iv_7d", "atm_iv_1d"}.issubset(df.columns):
        features["iv_slope_atm"] = (
            df["atm_iv_7d"].astype(float) - df["atm_iv_1d"].astype(float)
        )
    if {"rr25_7d", "rr25_1d"}.issubset(df.columns):
        features["rr_slope"] = (
            df["rr25_7d"].astype(float) - df["rr25_1d"].astype(float)
        )
    if {"fly25_7d", "fly25_1d"}.issubset(df.columns):
        features["fly_slope"] = (
            df["fly25_7d"].astype(float) - df["fly25_1d"].astype(float)
        )
    if "term_slope_1d_7d" in df.columns:
        features["term_slope_1d_7d"] = df["term_slope_1d_7d"].astype(float)

    for col in [
        "liq_net_notional_long_minus_short",
        "liq_count",
        "liq_amount",
        "liq_notional",
    ]:
        if col in df.columns:
            features[col] = df[col].astype(float)

    features = features.shift(1)
    features = features.add_prefix("feat_")
    feature_cols = list(features.columns)
    df = pd.concat([df, features], axis=1)
    return df, feature_cols


def prepare_dataset(path: str) -> PreparedData:
    df = load_data(path)
    df = add_forward_returns(df)
    df = add_future_path_stats(df, horizon_bars=6)
    df, feature_cols = build_features(df)

    required = feature_cols + [
        "forward_return_12h",
        "forward_return_24h",
        "forward_return_72h",
        "mfe_long",
        "mae_long",
        "mfe_short",
        "mae_short",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    return PreparedData(df=df, feature_cols=feature_cols)
