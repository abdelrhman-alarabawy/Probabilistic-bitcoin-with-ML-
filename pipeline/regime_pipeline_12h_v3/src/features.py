from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close.shift(1))
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / (atr + 1e-12)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return 100 * dx.ewm(alpha=1 / window, adjust=False).mean()


def build_features(df: pd.DataFrame, feature_shift: int = 1) -> pd.DataFrame:
    open_ = df["open"].shift(feature_shift)
    high = df["high"].shift(feature_shift)
    low = df["low"].shift(feature_shift)
    close = df["close"].shift(feature_shift)
    volume = df["volume"].shift(feature_shift)

    features = pd.DataFrame(index=df.index)
    features["open_s1"] = open_
    features["high_s1"] = high
    features["low_s1"] = low
    features["close_s1"] = close
    features["volume_s1"] = volume

    ret_1 = close.pct_change()
    log_ret = np.log(close + 1e-12) - np.log(close.shift(1) + 1e-12)
    features["ret_1"] = ret_1
    features["log_ret_1"] = log_ret
    features["rv_10"] = log_ret.rolling(10).std() * np.sqrt(10)
    features["rv_20"] = log_ret.rolling(20).std() * np.sqrt(20)
    features["rv_30"] = log_ret.rolling(30).std() * np.sqrt(30)

    tr = _true_range(high, low, close.shift(1))
    features["atr_14"] = tr.rolling(14).mean()

    ma_10 = close.rolling(10).mean()
    ma_30 = close.rolling(30).mean()
    features["ma_slope_10"] = ma_10.diff() / (close + 1e-12)
    features["ma_slope_30"] = ma_30.diff() / (close + 1e-12)
    features["ma_cross"] = ma_10 - ma_30

    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    signal = _ema(macd, 9)
    features["macd_hist"] = macd - signal

    features["range_pct"] = (high - low) / (close + 1e-12)
    body = (close - open_).abs()
    features["body_pct"] = body / (close + 1e-12)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    features["upper_wick_pct"] = upper_wick / (close + 1e-12)
    features["lower_wick_pct"] = lower_wick / (close + 1e-12)
    features["wick_ratio"] = upper_wick / (body + 1e-12)
    features["wick_ratio_low"] = lower_wick / (body + 1e-12)

    rsi = _rsi(close, 14)
    features["rsi_14"] = rsi
    features["rsi_band"] = (rsi - 50) / 50

    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    features["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mean + 1e-12)

    ret_mean = ret_1.rolling(20).mean()
    ret_std = ret_1.rolling(20).std()
    features["ret_zscore"] = (ret_1 - ret_mean) / (ret_std + 1e-12)

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    features["volume_z"] = (volume - vol_mean) / (vol_std + 1e-12)

    obv = (np.sign(ret_1).fillna(0.0) * volume).cumsum()
    features["obv_slope_5"] = obv.diff(5)

    features["adx_14"] = _adx(high, low, close, 14)

    return features
