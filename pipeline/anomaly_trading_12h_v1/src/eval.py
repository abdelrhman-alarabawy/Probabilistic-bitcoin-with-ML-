from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import HOLD_BARS, STOP_LOSS_PCT, TAKE_PROFIT_PCT


@dataclass(frozen=True)
class TradeResult:
    ret: float
    exit_type: str
    exit_idx: int
    mae: float
    mfe: float
    hold_bars: int


def simulate_trade(
    idx: int,
    direction: int,
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    hold_bars: int = HOLD_BARS,
    tp_pct: float = TAKE_PROFIT_PCT,
    sl_pct: float = STOP_LOSS_PCT,
) -> TradeResult | None:
    if direction == 0:
        return None
    if idx + hold_bars - 1 >= len(open_px):
        return None

    entry = float(open_px[idx])
    if not np.isfinite(entry) or entry == 0.0:
        return None

    tp_price = entry * (1.0 + tp_pct) if direction > 0 else entry * (1.0 - tp_pct)
    sl_price = entry * (1.0 - sl_pct) if direction > 0 else entry * (1.0 + sl_pct)

    exit_price = float(close_px[idx + hold_bars - 1])
    exit_idx = idx + hold_bars - 1
    exit_type = "time"

    for j in range(idx, idx + hold_bars):
        hi = float(high_px[j])
        lo = float(low_px[j])
        hit_tp = hi >= tp_price if direction > 0 else lo <= tp_price
        hit_sl = lo <= sl_price if direction > 0 else hi >= sl_price
        if hit_tp and hit_sl:
            exit_price = sl_price
            exit_idx = j
            exit_type = "sl"
            break
        if hit_tp:
            exit_price = tp_price
            exit_idx = j
            exit_type = "tp"
            break
        if hit_sl:
            exit_price = sl_price
            exit_idx = j
            exit_type = "sl"
            break

    if direction > 0:
        ret = (exit_price - entry) / entry
        high_max = float(np.nanmax(high_px[idx : exit_idx + 1]))
        low_min = float(np.nanmin(low_px[idx : exit_idx + 1]))
        mfe = (high_max - entry) / entry
        mae = (low_min - entry) / entry
    else:
        ret = (entry - exit_price) / entry
        high_max = float(np.nanmax(high_px[idx : exit_idx + 1]))
        low_min = float(np.nanmin(low_px[idx : exit_idx + 1]))
        mfe = (entry - low_min) / entry
        mae = (entry - high_max) / entry

    hold = exit_idx - idx + 1
    return TradeResult(ret=ret, exit_type=exit_type, exit_idx=exit_idx, mae=mae, mfe=mfe, hold_bars=hold)


def equity_curve(returns: np.ndarray) -> np.ndarray:
    if len(returns) == 0:
        return np.array([], dtype=float)
    return np.cumprod(1.0 + returns)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(np.min(drawdowns))


def cvar_95(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return float("nan")
    sorted_returns = np.sort(returns)
    k = max(1, int(np.ceil(0.05 * len(sorted_returns))))
    return float(np.mean(sorted_returns[:k]))


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_return": float("nan"),
            "median_return": float("nan"),
            "mae_mean": float("nan"),
            "mfe_mean": float("nan"),
            "max_drawdown": float("nan"),
            "cvar_95": float("nan"),
        }

    returns = trades["return"].to_numpy(dtype=float)
    equity = equity_curve(returns)
    return {
        "n_trades": int(len(trades)),
        "win_rate": float(np.mean(returns > 0.0)),
        "avg_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "mae_mean": float(np.mean(trades["mae"].to_numpy(dtype=float))),
        "mfe_mean": float(np.mean(trades["mfe"].to_numpy(dtype=float))),
        "max_drawdown": max_drawdown(equity),
        "cvar_95": cvar_95(returns),
    }
