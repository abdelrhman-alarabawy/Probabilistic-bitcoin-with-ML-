from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .rule_search import Rule, apply_rule


@dataclass(frozen=True)
class Split:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_months: List[str]
    test_months: List[str]


def generate_splits(
    df: pd.DataFrame,
    mode: str,
    train_months: int,
    test_months: int,
    step_months: int,
    min_train_months: int = 12,
) -> List[Split]:
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    splits: List[Split] = []

    if mode == "expanding":
        start = min_train_months
        while start + test_months <= len(months):
            train_slice = months[:start]
            test_slice = months[start : start + test_months]
            train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
            test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
            splits.append(
                Split(
                    fold_id=len(splits),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_months=train_slice,
                    test_months=test_slice,
                )
            )
            start += step_months
    elif mode == "rolling":
        start = 0
        while start + train_months + test_months <= len(months):
            train_slice = months[start : start + train_months]
            test_slice = months[start + train_months : start + train_months + test_months]
            train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
            test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
            splits.append(
                Split(
                    fold_id=len(splits),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    train_months=train_slice,
                    test_months=test_slice,
                )
            )
            start += step_months
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return splits


def compute_profit_factor(trade_returns: np.ndarray) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("nan")
    return float(gains / abs(losses))


def compute_drawdown(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return float("nan")
    cumulative = np.cumsum(trade_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(np.min(drawdown))


def compute_cvar95(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return float("nan")
    return float(np.percentile(trade_returns, 5))


def evaluate_rule_on_df(
    rule: Rule, df: pd.DataFrame, fee: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mask = apply_rule(rule, df)
    df_events = df[mask].copy()
    if df_events.empty:
        return df_events, {
            "precision": float("nan"),
            "trade_count": 0,
            "avg_pnl": float("nan"),
            "profit_factor": float("nan"),
            "max_drawdown": float("nan"),
            "cvar95": float("nan"),
        }

    horizon_col = f"forward_return_{rule.horizon}h"
    returns = df_events[horizon_col].to_numpy()
    if rule.direction == "short":
        returns = -returns

    trade_returns = returns - fee
    precision = float((returns >= rule.tp).mean())
    metrics = {
        "precision": precision,
        "trade_count": int(len(returns)),
        "avg_pnl": float(np.mean(trade_returns)),
        "profit_factor": compute_profit_factor(trade_returns),
        "max_drawdown": compute_drawdown(trade_returns),
        "cvar95": compute_cvar95(trade_returns),
    }
    return df_events, metrics


def combine_signals(
    events_by_rule: List[Tuple[Rule, pd.DataFrame]], fee: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not events_by_rule:
        return pd.DataFrame(), {
            "precision_long": float("nan"),
            "precision_short": float("nan"),
            "trade_count": 0,
            "avg_pnl": float("nan"),
            "profit_factor": float("nan"),
            "max_drawdown": float("nan"),
            "cvar95": float("nan"),
        }

    rows = []
    for rule, df_events in events_by_rule:
        if df_events.empty:
            continue
        for _, row in df_events.iterrows():
            horizon_col = f"forward_return_{rule.horizon}h"
            raw_ret = float(row[horizon_col])
            direction = rule.direction
            if direction == "short":
                raw_ret = -raw_ret
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "direction": rule.direction,
                    "rule_id": rule.rule_id,
                    "tp": rule.tp,
                    "horizon": rule.horizon,
                    "raw_return": raw_ret,
                    "return_net": raw_ret - fee,
                }
            )
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {
            "precision_long": float("nan"),
            "precision_short": float("nan"),
            "trade_count": 0,
            "avg_pnl": float("nan"),
            "profit_factor": float("nan"),
            "max_drawdown": float("nan"),
            "cvar95": float("nan"),
        }

    trades = trades.sort_values("timestamp")
    dup_counts = trades.groupby("timestamp")["direction"].nunique()
    conflict_ts = dup_counts[dup_counts > 1].index
    trades = trades[~trades["timestamp"].isin(conflict_ts)].copy()

    long_mask = trades["direction"] == "long"
    short_mask = trades["direction"] == "short"
    precision_long = float((trades.loc[long_mask, "raw_return"] >= trades.loc[long_mask, "tp"]).mean()) if long_mask.any() else float("nan")
    precision_short = float((trades.loc[short_mask, "raw_return"] >= trades.loc[short_mask, "tp"]).mean()) if short_mask.any() else float("nan")

    trade_returns = trades["return_net"].to_numpy(dtype=float)
    metrics = {
        "precision_long": precision_long,
        "precision_short": precision_short,
        "trade_count": int(len(trades)),
        "avg_pnl": float(np.mean(trade_returns)),
        "profit_factor": compute_profit_factor(trade_returns),
        "max_drawdown": compute_drawdown(trade_returns),
        "cvar95": compute_cvar95(trade_returns),
    }
    return trades, metrics
