from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import rolling_slope, rolling_zscore, safe_divide, sigmoid


def compute_rvs(
    df: pd.DataFrame, column_map: Dict[str, str], settings: dict
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Create random variables using only data available at or before time t.
    All rolling statistics use shifted inputs to avoid look-ahead.
    """
    rv = pd.DataFrame(index=df.index)
    meta: Dict[str, dict] = {}

    def add_rv(
        name: str,
        series: pd.Series,
        description: str,
        formula: str,
        inputs: list,
        rv_type: str = "continuous",
        shift: int = 0,
        window: int = 0,
    ) -> None:
        rv[name] = series
        meta[name] = {
            "description": description,
            "formula": formula,
            "inputs": inputs,
            "type": rv_type,
            "shift": shift,
            "window": window,
        }

    open_col = column_map["open"]
    high_col = column_map["high"]
    low_col = column_map["low"]
    close_col = column_map["close"]
    volume_col = column_map["volume"]

    open_ = df[open_col]
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    ret_1h = np.log(safe_divide(close, close.shift(settings["return_lag"])))
    add_rv(
        "ret_1h",
        ret_1h,
        "Log return over 1 hour.",
        "ret_t = log(C_t / C_{t-1})",
        [close_col],
        rv_type="continuous",
        shift=settings["return_lag"],
    )

    direction = pd.Series(np.where(ret_1h > 0, 1, np.where(ret_1h < 0, -1, 0)), index=rv.index)
    add_rv(
        "direction_sign",
        direction,
        "Signed return direction (-1, 0, 1).",
        "direction_t = sign(ret_t)",
        ["ret_1h"],
        rv_type="discrete",
        shift=0,
    )

    range_1h = safe_divide(high - low, close)
    add_rv(
        "range_1h",
        range_1h,
        "Normalized high-low range.",
        "range_t = (H_t - L_t) / C_t",
        [high_col, low_col, close_col],
    )

    body_1h = safe_divide(close - open_, close)
    add_rv(
        "body_1h",
        body_1h,
        "Normalized candle body.",
        "body_t = (C_t - O_t) / C_t",
        [open_col, close_col],
    )

    vol_window = settings["realized_vol_window"]
    realized_vol = ret_1h.shift(1).rolling(window=vol_window, min_periods=vol_window).std()
    add_rv(
        "realized_vol_24h",
        realized_vol,
        "Trailing realized volatility (24 bars).",
        "rv_t = std(ret_{t-1..t-24})",
        ["ret_1h"],
        shift=1,
        window=vol_window,
    )

    tail_window = settings["tail_risk_window"]
    tail_mean = ret_1h.shift(1).rolling(window=tail_window, min_periods=tail_window).mean()
    tail_std = ret_1h.shift(1).rolling(window=tail_window, min_periods=tail_window).std()
    tail_risk_z = (ret_1h - tail_mean) / tail_std
    add_rv(
        "tail_risk_z",
        tail_risk_z,
        "Return z-score versus trailing window (tail risk proxy).",
        "tail_z_t = (ret_t - mean(ret_{t-1..t-N})) / std(ret_{t-1..t-N})",
        ["ret_1h"],
        shift=1,
        window=tail_window,
    )

    horizons = settings["momentum_horizons"]
    z_window = settings["momentum_z_window"]
    zscores = []
    for horizon in horizons:
        ret_h = np.log(safe_divide(close, close.shift(horizon)))
        mu = ret_h.shift(1).rolling(window=z_window, min_periods=z_window).mean()
        sigma = ret_h.shift(1).rolling(window=z_window, min_periods=z_window).std()
        zscores.append((ret_h - mu) / sigma)
    momentum_score = pd.concat(zscores, axis=1).mean(axis=1)
    add_rv(
        "momentum_score",
        momentum_score,
        "Average z-scored multi-horizon return momentum.",
        "momentum_t = mean(zscore(log(C_t/C_{t-h})) for h in horizons)",
        [close_col],
        shift=1,
        window=z_window,
    )

    fast = settings["trend_fast_window"]
    slow = settings["trend_slow_window"]
    ma_fast = close.shift(1).rolling(window=fast, min_periods=fast).mean()
    ma_slow = close.shift(1).rolling(window=slow, min_periods=slow).mean()
    trend_score = safe_divide(ma_fast - ma_slow, close)
    add_rv(
        "trend_score",
        trend_score,
        "Distance between fast and slow moving averages.",
        "trend_t = (MA_fast - MA_slow) / C_t",
        [close_col],
        shift=1,
        window=slow,
    )

    mean_reversion = -safe_divide(close - ma_slow, ma_slow)
    add_rv(
        "mean_reversion_score",
        mean_reversion,
        "Negative distance from slow moving average (overbought/oversold).",
        "mr_t = -(C_t - MA_slow) / MA_slow",
        [close_col],
        shift=1,
        window=slow,
    )

    volume = df[volume_col]
    vol_window = settings["volume_window"]
    volume_z = rolling_zscore(volume, vol_window)
    add_rv(
        "volume_z_24h",
        volume_z,
        "Volume z-score versus trailing window.",
        "vol_z_t = (V_t - mean(V_{t-1..t-24})) / std(V_{t-1..t-24})",
        [volume_col],
        shift=1,
        window=vol_window,
    )

    vol_trend_window = settings["volume_trend_window"]
    volume_trend = rolling_slope(volume.shift(1), vol_trend_window)
    add_rv(
        "volume_trend_24h",
        volume_trend,
        "Slope of volume over trailing window.",
        "vol_trend_t = slope(V_{t-1..t-24})",
        [volume_col],
        shift=1,
        window=vol_trend_window,
    )

    imbalance_last = column_map.get("imbalance_last")
    imbalance_mean = column_map.get("imbalance_mean")
    imbalance_window = settings["imbalance_window"]
    imbalance_series = None
    inputs = []
    if imbalance_last:
        inputs.append(imbalance_last)
        imbalance_series = rolling_zscore(df[imbalance_last], imbalance_window)
    if imbalance_mean:
        inputs.append(imbalance_mean)
        imbalance_mean_z = rolling_zscore(df[imbalance_mean], imbalance_window)
        imbalance_series = (
            imbalance_mean_z
            if imbalance_series is None
            else 0.5 * (imbalance_series + imbalance_mean_z)
        )
    if imbalance_series is not None:
        add_rv(
            "order_imbalance_score",
            imbalance_series,
            "Order book imbalance z-score (last and mean).",
            "imbalance_t = mean(zscore(imb_last), zscore(imb_mean))",
            inputs,
            shift=1,
            window=imbalance_window,
        )

    spread_bps = column_map.get("spread_bps_last")
    spread_last = column_map.get("spread_last")
    bid_last = column_map.get("bid_last")
    ask_last = column_map.get("ask_last")
    mid_last = column_map.get("mid_last")
    spread_cost = None
    spread_inputs = []
    if spread_bps:
        spread_cost = df[spread_bps]
        spread_inputs = [spread_bps]
    else:
        if spread_last:
            spread_cost = df[spread_last]
            spread_inputs.append(spread_last)
        elif bid_last and ask_last:
            spread_cost = df[ask_last] - df[bid_last]
            spread_inputs.extend([bid_last, ask_last])
        if spread_cost is not None:
            mid_price = None
            if mid_last:
                mid_price = df[mid_last]
                spread_inputs.append(mid_last)
            elif bid_last and ask_last:
                mid_price = (df[bid_last] + df[ask_last]) / 2.0
            if mid_price is not None:
                spread_cost = safe_divide(spread_cost, mid_price) * 10000.0
    if spread_cost is not None:
        add_rv(
            "spread_cost_bps",
            spread_cost,
            "Estimated spread cost in basis points.",
            "spread_bps_t = (ask_t - bid_t) / mid_t * 1e4",
            spread_inputs,
        )

    liq_notional = column_map.get("liq_notional")
    liq_amount = column_map.get("liq_amount")
    liq_count = column_map.get("liq_count")
    liq_value = None
    liq_inputs = []
    if liq_notional:
        liq_value = df[liq_notional]
        liq_inputs.append(liq_notional)
    elif liq_amount:
        liq_value = df[liq_amount]
        liq_inputs.append(liq_amount)
    elif liq_count:
        liq_value = df[liq_count]
        liq_inputs.append(liq_count)
    if liq_value is not None:
        liq_intensity = np.log1p(liq_value)
        add_rv(
            "liquidity_intensity",
            liq_intensity,
            "Log-scaled liquidation intensity.",
            "liq_intensity_t = log(1 + liq_metric_t)",
            liq_inputs,
        )

    liq_buy = column_map.get("liq_buy_notional")
    liq_sell = column_map.get("liq_sell_notional")
    liq_net = column_map.get("liq_net_notional")
    agg_bias = None
    agg_inputs = []
    if liq_buy and liq_sell:
        agg_inputs = [liq_buy, liq_sell]
        agg_bias = safe_divide(df[liq_buy] - df[liq_sell], df[liq_buy] + df[liq_sell])
    elif liq_net and liq_notional:
        agg_inputs = [liq_net, liq_notional]
        agg_bias = safe_divide(df[liq_net], df[liq_notional])
    if agg_bias is not None:
        add_rv(
            "aggressor_bias",
            agg_bias,
            "Net aggressor bias from liquidation flows.",
            "bias_t = (buy_notional - sell_notional) / (buy_notional + sell_notional)",
            agg_inputs,
        )

    atm_iv_cols = column_map.get("atm_iv_cols") or []
    if not atm_iv_cols and column_map.get("atm_iv"):
        atm_iv_cols = [column_map.get("atm_iv")]
    atm_iv_cols = list(dict.fromkeys([col for col in atm_iv_cols if col]))
    if atm_iv_cols:
        iv_level = df[atm_iv_cols].mean(axis=1)
        add_rv(
            "iv_level",
            iv_level,
            "Average ATM implied volatility level.",
            "iv_level_t = mean(atm_iv_horizons)",
            atm_iv_cols,
        )

    term_slope_col = column_map.get("term_slope")
    term_slope = None
    term_inputs = []
    if term_slope_col:
        term_slope = df[term_slope_col]
        term_inputs.append(term_slope_col)
    else:
        iv_1d = next((col for col in atm_iv_cols if "1d" in col.lower()), None)
        iv_7d = next((col for col in atm_iv_cols if "7d" in col.lower()), None)
        if iv_1d and iv_7d:
            term_slope = df[iv_7d] - df[iv_1d]
            term_inputs.extend([iv_7d, iv_1d])
    if term_slope is not None:
        add_rv(
            "term_structure_slope",
            term_slope,
            "Implied volatility term structure slope.",
            "term_slope_t = iv_7d - iv_1d",
            term_inputs,
        )

    implied_1h = column_map.get("implied_move_1h")
    implied_24h = column_map.get("implied_move_24h")
    if implied_1h and implied_24h:
        implied_ratio = safe_divide(df[implied_1h], df[implied_24h])
        add_rv(
            "implied_move_ratio",
            implied_ratio,
            "Ratio of 1h to 24h implied move.",
            "implied_ratio_t = move_1h / move_24h",
            [implied_1h, implied_24h],
        )

    weight_m = settings["signal_weights"]["momentum"]
    weight_t = settings["signal_weights"]["trend"]
    if "momentum_score" in rv.columns and "trend_score" in rv.columns:
        signal_dir = weight_m * rv["momentum_score"] + weight_t * rv["trend_score"]
        add_rv(
            "signal_direction_score",
            signal_dir,
            "Weighted direction score from trend and momentum.",
            "signal_dir_t = w_m * momentum_t + w_t * trend_t",
            ["momentum_score", "trend_score"],
        )

        alignment = (
            np.sign(rv["momentum_score"]) == np.sign(rv["trend_score"])
        ).astype(float)
        alignment_factor = 0.5 + settings["signal_confidence"]["alignment_boost"] * alignment
        tail_penalty = settings["signal_confidence"]["tail_risk_penalty"]
        if "tail_risk_z" in rv.columns:
            vol_penalty = np.exp(-tail_penalty * rv["tail_risk_z"].abs())
        else:
            vol_penalty = 1.0
        strength = signal_dir.abs()
        base = sigmoid(strength, scale=settings["signal_confidence"]["sigmoid_scale"])
        signal_confidence = (base * alignment_factor * vol_penalty).clip(0.0, 1.0)
        add_rv(
            "signal_confidence",
            signal_confidence,
            "Soft confidence score in [0,1] for downstream evidence.",
            "conf_t = sigmoid(|signal_dir_t|) * alignment * exp(-|tail_z|)",
            ["signal_direction_score", "tail_risk_z"],
        )

    return rv, meta
