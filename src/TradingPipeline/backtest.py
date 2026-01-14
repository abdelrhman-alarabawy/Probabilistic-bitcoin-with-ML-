from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import BAR_MINUTES, FEE_RATE, HORIZON_MINUTES, SLIPPAGE_PCT


@dataclass(frozen=True)
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.DataFrame


def _compute_levels(entry: float, side: str, tp_points: float, sl_points: float) -> tuple[float, float]:
    multiplier = entry / 100000
    if side == "long":
        tp = entry + tp_points * multiplier
        sl = entry - sl_points * multiplier
    else:
        tp = entry - tp_points * multiplier
        sl = entry + sl_points * multiplier
    return tp, sl


def backtest_trades(
    df: pd.DataFrame,
    predictions: pd.Series,
    tp_points: float,
    sl_points: float,
    horizon_minutes: int = HORIZON_MINUTES,
) -> BacktestResult:
    horizon_bars = max(int(horizon_minutes / BAR_MINUTES), 1)
    trades: List[dict] = []

    for idx, signal in predictions.items():
        if signal not in ("long", "short"):
            continue
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue

        entry_price = float(df.loc[entry_idx, "open"])
        entry_ts = df.loc[entry_idx, "timestamp"]
        tp, sl = _compute_levels(entry_price, signal, tp_points, sl_points)

        exit_price = float(df.loc[entry_idx, "close"])
        exit_ts = df.loc[entry_idx, "timestamp"]
        reason = "timeout"

        last_idx = min(entry_idx + horizon_bars - 1, len(df) - 1)
        for bar_idx in range(entry_idx, last_idx + 1):
            high = float(df.loc[bar_idx, "high"])
            low = float(df.loc[bar_idx, "low"])

            if signal == "long":
                hit_tp = high >= tp
                hit_sl = low <= sl
                if hit_tp and hit_sl:
                    exit_price = sl
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "sl"
                    break
                if hit_tp:
                    exit_price = tp
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "tp"
                    break
                if hit_sl:
                    exit_price = sl
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "sl"
                    break
            else:
                hit_tp = low <= tp
                hit_sl = high >= sl
                if hit_tp and hit_sl:
                    exit_price = sl
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "sl"
                    break
                if hit_tp:
                    exit_price = tp
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "tp"
                    break
                if hit_sl:
                    exit_price = sl
                    exit_ts = df.loc[bar_idx, "timestamp"]
                    reason = "sl"
                    break
            exit_price = float(df.loc[bar_idx, "close"])
            exit_ts = df.loc[bar_idx, "timestamp"]

        if signal == "long":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        cost = (FEE_RATE + SLIPPAGE_PCT) * 2
        net_return = gross_return - cost

        trades.append(
            {
                "signal_index": idx,
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "side": signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "reason": reason,
            }
        )

    trade_log = pd.DataFrame(trades)

    if trade_log.empty:
        equity = pd.DataFrame(columns=["timestamp", "equity"])
        return BacktestResult(trade_log=trade_log, equity_curve=equity)

    trade_log = trade_log.sort_values("exit_timestamp").reset_index(drop=True)
    equity = (1 + trade_log["net_return"]).cumprod()
    equity_curve = pd.DataFrame(
        {"timestamp": trade_log["exit_timestamp"], "equity": equity}
    )

    return BacktestResult(trade_log=trade_log, equity_curve=equity_curve)
