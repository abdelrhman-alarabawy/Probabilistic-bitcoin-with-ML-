from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak)
    return float(drawdown.min())


def simulate_pnl(df: pd.DataFrame, action_col: str, fee: float = 0.0005) -> dict:
    if "close" not in df.columns:
        return {
            "signals_per_month": 0.0,
            "winrate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    ret = close.pct_change().fillna(0.0).to_numpy()

    action = df[action_col].astype(str).to_numpy()
    pos = np.where(action == "long", 1.0, np.where(action == "short", -1.0, 0.0))
    turns = np.abs(np.diff(pos, prepend=0.0)) > 0
    trade_fee = turns.astype(float) * fee
    pnl = pos * ret - trade_fee

    traded = pos != 0
    traded_pnl = pnl[traded]
    wins = np.sum(traded_pnl > 0)
    n_trades = int(np.sum(traded))

    gross_win = float(np.sum(traded_pnl[traded_pnl > 0])) if n_trades else 0.0
    gross_loss = float(np.abs(np.sum(traded_pnl[traded_pnl < 0]))) if n_trades else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf" if gross_win > 0 else 0.0)

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    months = max(1, ts.dt.to_period("M").nunique())

    equity = np.cumsum(pnl)
    return {
        "signals_per_month": float(n_trades / months),
        "winrate": float(wins / n_trades) if n_trades else 0.0,
        "avg_pnl": float(np.mean(traded_pnl)) if n_trades else 0.0,
        "profit_factor": float(profit_factor),
        "max_drawdown": _max_drawdown(equity),
    }


def evaluate_trading(df_test: pd.DataFrame, action_col: str, fee: float = 0.0005) -> dict:
    return simulate_pnl(df_test, action_col=action_col, fee=fee)
