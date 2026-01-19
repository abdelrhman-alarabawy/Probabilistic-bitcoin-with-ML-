from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    category: str
    formula: str
    lookback: str
    leakage_safe: bool


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close.shift(1))
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / (
        atr + 1e-12
    )
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / (
        atr + 1e-12
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return 100 * dx.ewm(alpha=1 / window, adjust=False).mean()


def _rolling_r2(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def _rolling_downside_std(values: np.ndarray) -> float:
    downside = values[values < 0]
    if len(downside) == 0:
        return 0.0
    return float(np.std(downside, ddof=0))


def _rolling_cvar(values: np.ndarray, q: float = 0.1) -> float:
    if len(values) == 0:
        return 0.0
    k = max(int(np.ceil(len(values) * q)), 1)
    return float(np.mean(np.sort(values)[:k]))


def _rolling_max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / (cummax + 1e-12)
    return float(np.abs(drawdown.min()))


def _rolling_hurst(values: np.ndarray) -> float:
    if len(values) < 10:
        return 0.5
    mean = np.mean(values)
    dev = np.cumsum(values - mean)
    r = dev.max() - dev.min()
    s = np.std(values, ddof=0)
    if s == 0:
        return 0.5
    rs = r / s
    return float(np.log(rs + 1e-12) / np.log(len(values)))


def _rolling_autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _aroon(series: pd.Series, window: int, kind: str) -> pd.Series:
    if kind == "up":
        return series.rolling(window).apply(lambda x: (window - 1 - np.argmax(x)) / window * 100, raw=True)
    return series.rolling(window).apply(lambda x: (window - 1 - np.argmin(x)) / window * 100, raw=True)


def get_core_feature_names() -> List[str]:
    return [
        "log_return_1",
        "realized_vol_5",
        "realized_vol_10",
        "realized_vol_20",
        "atr_14",
        "atr_pct_14",
        "parkinson_vol_20",
        "gk_vol_20",
        "ema_slope_20",
        "ema_slope_50",
        "ma_cross_10_20",
        "ma_cross_20_50",
        "adx_14",
        "aroon_osc_14",
        "r2_trend_20",
        "bb_bandwidth_20",
        "donchian_width_20",
        "keltner_width_20",
        "range_pct",
        "return_z_20",
        "volume_z_20",
        "obv_slope_10",
    ]


def build_features_extended(df: pd.DataFrame, feature_shift: int = 1) -> Tuple[pd.DataFrame, List[FeatureSpec]]:
    specs: List[FeatureSpec] = []
    eps = 1e-12

    open_s = df["open"].shift(feature_shift)
    high_s = df["high"].shift(feature_shift)
    low_s = df["low"].shift(feature_shift)
    close_s = df["close"].shift(feature_shift)
    volume_s = df["volume"].shift(feature_shift)

    features = pd.DataFrame(index=df.index)

    def add_feature(name: str, series: pd.Series, category: str, formula: str, lookback: str) -> None:
        features[name] = series
        specs.append(
            FeatureSpec(
                name=name,
                category=category,
                formula=formula,
                lookback=lookback,
                leakage_safe=True,
            )
        )

    log_return_1 = np.log((close_s + eps) / (close_s.shift(1) + eps))
    add_feature("log_return_1", log_return_1, "return_vol", "log(close/close.shift(1))", "1")

    for w in [5, 10, 20]:
        realized_vol = log_return_1.rolling(w).std()
        add_feature(
            f"realized_vol_{w}",
            realized_vol,
            "return_vol",
            "rolling_std(log_return_1)",
            str(w),
        )

    for w in [5, 10]:
        vol_of_vol = features[f"realized_vol_{w}"].rolling(w).std()
        add_feature(
            f"vol_of_vol_{w}",
            vol_of_vol,
            "return_vol",
            f"rolling_std(realized_vol_{w})",
            str(w),
        )

    for w in [7, 14, 28]:
        tr = _true_range(high_s, low_s, close_s.shift(1))
        atr = tr.rolling(w).mean()
        add_feature(f"atr_{w}", atr, "return_vol", "rolling_mean(true_range)", str(w))
        add_feature(
            f"atr_pct_{w}",
            atr / (close_s + eps),
            "return_vol",
            "atr_w / close",
            str(w),
        )

    for w in [10, 20]:
        hl_log = np.log((high_s + eps) / (low_s + eps))
        parkinson = np.sqrt((hl_log.pow(2)).rolling(w).mean() / (4 * np.log(2)))
        add_feature(
            f"parkinson_vol_{w}",
            parkinson,
            "return_vol",
            "sqrt(mean(log(high/low)^2)/(4 ln2))",
            str(w),
        )
        gk_raw = 0.5 * hl_log.pow(2) - (2 * np.log(2) - 1) * np.log((close_s + eps) / (open_s + eps)).pow(2)
        gk_vol = np.sqrt(gk_raw.rolling(w).mean().clip(lower=0))
        add_feature(
            f"gk_vol_{w}",
            gk_vol,
            "return_vol",
            "sqrt(mean(GarmanKlass))",
            str(w),
        )

    for w in [10, 20, 50]:
        ema = _ema(close_s, w)
        ema_slope = ema.diff() / (close_s + eps)
        add_feature(
            f"ema_slope_{w}",
            ema_slope,
            "trend",
            "diff(EMA)/close",
            str(w),
        )

    for a, b in [(10, 20), (20, 50)]:
        ema_a = _ema(close_s, a)
        ema_b = _ema(close_s, b)
        add_feature(
            f"ma_cross_{a}_{b}",
            ema_a - ema_b,
            "trend",
            "EMA(a) - EMA(b)",
            f"{a},{b}",
        )

    for w in [7, 14, 28]:
        adx = _adx(high_s, low_s, close_s, w)
        add_feature(f"adx_{w}", adx, "trend", "ADX", str(w))

    for w in [14, 28]:
        aroon_up = _aroon(high_s, w, "up")
        aroon_down = _aroon(low_s, w, "down")
        add_feature(f"aroon_up_{w}", aroon_up, "trend", "Aroon up", str(w))
        add_feature(f"aroon_down_{w}", aroon_down, "trend", "Aroon down", str(w))
        add_feature(
            f"aroon_osc_{w}",
            aroon_up - aroon_down,
            "trend",
            "Aroon up - down",
            str(w),
        )

    for w in [20, 50]:
        r2 = close_s.rolling(w).apply(lambda x: _rolling_r2(np.asarray(x)), raw=False)
        add_feature(
            f"r2_trend_{w}",
            r2,
            "trend",
            "rolling_r2(close vs time)",
            str(w),
        )

    for w in [20, 50]:
        ma = close_s.rolling(w).mean()
        std = close_s.rolling(w).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        bandwidth = (upper - lower) / (ma + eps)
        pct_b = (close_s - lower) / ((upper - lower) + eps)
        add_feature(
            f"bb_bandwidth_{w}",
            bandwidth,
            "compression",
            "(upper-lower)/ma",
            f"{w},2",
        )
        add_feature(
            f"bb_pctb_{w}",
            pct_b,
            "compression",
            "(close-lower)/(upper-lower)",
            f"{w},2",
        )

    for w in [20, 50]:
        donchian = (high_s.rolling(w).max() - low_s.rolling(w).min()) / (close_s + eps)
        add_feature(
            f"donchian_width_{w}",
            donchian,
            "compression",
            "(max(high)-min(low))/close",
            str(w),
        )

    for w in [20, 50]:
        ema = _ema(close_s, w)
        atr = _true_range(high_s, low_s, close_s.shift(1)).rolling(w).mean()
        keltner_width = (2 * atr) / (ema + eps)
        add_feature(
            f"keltner_width_{w}",
            keltner_width,
            "compression",
            "(2*ATR)/EMA",
            str(w),
        )
        bb_ma = close_s.rolling(w).mean()
        bb_std = close_s.rolling(w).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        keltner_upper = ema + 1.5 * atr
        keltner_lower = ema - 1.5 * atr
        squeeze = ((bb_upper < keltner_upper) & (bb_lower > keltner_lower)).astype(float)
        add_feature(
            f"squeeze_flag_{w}",
            squeeze,
            "compression",
            "BB inside Keltner",
            str(w),
        )

    range_pct = (high_s - low_s) / (close_s + eps)
    add_feature("range_pct", range_pct, "compression", "(high-low)/close", "1")
    for w in [10, 20]:
        add_feature(
            f"range_pct_mean_{w}",
            range_pct.rolling(w).mean(),
            "compression",
            "rolling_mean(range_pct)",
            str(w),
        )
        add_feature(
            f"range_pct_std_{w}",
            range_pct.rolling(w).std(),
            "compression",
            "rolling_std(range_pct)",
            str(w),
        )

    body_pct = (close_s - open_s).abs() / ((high_s - low_s) + eps)
    upper_wick_pct = (high_s - np.maximum(open_s, close_s)) / ((high_s - low_s) + eps)
    lower_wick_pct = (np.minimum(open_s, close_s) - low_s) / ((high_s - low_s) + eps)
    clv = (close_s - low_s) / ((high_s - low_s) + eps)
    gap_pct = (open_s - close_s.shift(1)) / (close_s.shift(1) + eps)

    add_feature("body_pct", body_pct, "microstructure", "abs(close-open)/(high-low)", "1")
    add_feature("upper_wick_pct", upper_wick_pct, "microstructure", "(high-max(open,close))/(high-low)", "1")
    add_feature("lower_wick_pct", lower_wick_pct, "microstructure", "(min(open,close)-low)/(high-low)", "1")
    add_feature("close_location_value", clv, "microstructure", "(close-low)/(high-low)", "1")
    add_feature("gap_pct", gap_pct, "microstructure", "(open-close.shift(1))/close.shift(1)", "1")

    for w in [5, 10, 20]:
        add_feature(
            f"body_pct_mean_{w}",
            body_pct.rolling(w).mean(),
            "microstructure",
            "rolling_mean(body_pct)",
            str(w),
        )
        add_feature(
            f"body_pct_std_{w}",
            body_pct.rolling(w).std(),
            "microstructure",
            "rolling_std(body_pct)",
            str(w),
        )
        add_feature(
            f"upper_wick_pct_mean_{w}",
            upper_wick_pct.rolling(w).mean(),
            "microstructure",
            "rolling_mean(upper_wick_pct)",
            str(w),
        )
        add_feature(
            f"upper_wick_pct_std_{w}",
            upper_wick_pct.rolling(w).std(),
            "microstructure",
            "rolling_std(upper_wick_pct)",
            str(w),
        )
        add_feature(
            f"lower_wick_pct_mean_{w}",
            lower_wick_pct.rolling(w).mean(),
            "microstructure",
            "rolling_mean(lower_wick_pct)",
            str(w),
        )
        add_feature(
            f"lower_wick_pct_std_{w}",
            lower_wick_pct.rolling(w).std(),
            "microstructure",
            "rolling_std(lower_wick_pct)",
            str(w),
        )

    vol_log = np.log(volume_s + 1)
    add_feature("vol_log", vol_log, "volume", "log(volume+1)", "1")

    for w in [20, 50]:
        vol_mean = volume_s.rolling(w).mean()
        vol_std = volume_s.rolling(w).std()
        add_feature(
            f"volume_z_{w}",
            (volume_s - vol_mean) / (vol_std + eps),
            "volume",
            "(volume-mean)/std",
            str(w),
        )

    obv = (np.sign(log_return_1).fillna(0.0) * volume_s).cumsum()
    add_feature("obv", obv, "volume", "cumsum(sign(ret)*volume)", "1")
    for w in [10, 20]:
        add_feature(
            f"obv_slope_{w}",
            obv.diff(w),
            "volume",
            f"obv.diff({w})",
            str(w),
        )

    mf_multiplier = ((close_s - low_s) - (high_s - close_s)) / ((high_s - low_s) + eps)
    mf_volume = mf_multiplier * volume_s
    cmf_20 = mf_volume.rolling(20).sum() / (volume_s.rolling(20).sum() + eps)
    add_feature("chaikin_mf", mf_multiplier, "volume", "MF multiplier", "1")
    add_feature("cmf_20", cmf_20, "volume", "rolling_sum(MF)/rolling_sum(volume)", "20")

    vpt = (volume_s * log_return_1).cumsum()
    add_feature("vpt", vpt, "volume", "cumsum(volume*return)", "1")
    for w in [10, 20]:
        add_feature(
            f"vpt_slope_{w}",
            vpt.diff(w),
            "volume",
            f"vpt.diff({w})",
            str(w),
        )

    add_feature("dollar_volume", close_s * volume_s, "volume", "close*volume", "1")

    for w in [20, 50]:
        ret_mean = log_return_1.rolling(w).mean()
        ret_std = log_return_1.rolling(w).std()
        return_z = (log_return_1 - ret_mean) / (ret_std + eps)
        add_feature(
            f"return_z_{w}",
            return_z,
            "tail_risk",
            "(return-mean)/std",
            str(w),
        )
        downside = log_return_1.rolling(w).apply(lambda x: _rolling_downside_std(np.asarray(x)), raw=False)
        add_feature(
            f"downside_vol_{w}",
            downside,
            "tail_risk",
            "std(negative_returns)",
            str(w),
        )

    cvar_50 = log_return_1.rolling(50).apply(lambda x: _rolling_cvar(np.asarray(x), q=0.1), raw=False)
    add_feature("cvar_proxy_50", cvar_50, "tail_risk", "mean(worst 10% returns)", "50")

    for w in [50, 100]:
        mdd = close_s.rolling(w).apply(lambda x: _rolling_max_drawdown(np.asarray(x)), raw=False)
        add_feature(
            f"max_drawdown_{w}",
            mdd,
            "tail_risk",
            "max_drawdown(close)",
            str(w),
        )

    jump_flag = (features["return_z_20"].abs() > 3).astype(float)
    add_feature("jump_flag", jump_flag, "tail_risk", "abs(return_z_20) > 3", "20")

    rolling_skew = log_return_1.rolling(50).skew()
    rolling_kurt = log_return_1.rolling(50).kurt()
    add_feature("rolling_skew_50", rolling_skew, "distribution", "rolling_skew(returns)", "50")
    add_feature("rolling_kurt_50", rolling_kurt, "distribution", "rolling_kurtosis(returns)", "50")

    hurst = log_return_1.rolling(100).apply(lambda x: _rolling_hurst(np.asarray(x)), raw=False)
    add_feature("hurst_proxy_100", hurst, "distribution", "rescaled_range_hurst", "100")

    autocorr_1 = log_return_1.rolling(50).apply(lambda x: _rolling_autocorr(np.asarray(x), 1), raw=False)
    autocorr_2 = log_return_1.rolling(50).apply(lambda x: _rolling_autocorr(np.asarray(x), 2), raw=False)
    add_feature("autocorr_return_50_lag1", autocorr_1, "distribution", "autocorr(returns,1)", "50")
    add_feature("autocorr_return_50_lag2", autocorr_2, "distribution", "autocorr(returns,2)", "50")

    return features, specs
