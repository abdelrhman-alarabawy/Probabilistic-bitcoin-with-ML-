from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class EligibilityResult:
    stats: pd.DataFrame
    eligible_regimes: set[int]


def _average_duration(states: Iterable[int], regime_id: int) -> float:
    durations = []
    run = 0
    in_regime = False
    for state in states:
        if state == regime_id:
            run += 1
            in_regime = True
        else:
            if in_regime:
                durations.append(run)
                run = 0
                in_regime = False
    if in_regime:
        durations.append(run)
    return float(np.mean(durations)) if durations else 0.0


def _transition_risk(states: np.ndarray, regime_id: int, window: int) -> float:
    if window <= 0:
        return 1.0
    leave_flags = []
    n = len(states)
    for i in range(n):
        if states[i] != regime_id:
            continue
        end = min(i + window + 1, n)
        future = states[i + 1 : end]
        leave_flags.append(1.0 if np.any(future != regime_id) else 0.0)
    return float(np.mean(leave_flags)) if leave_flags else 1.0


def compute_regime_stats(
    df: pd.DataFrame,
    state_col: str,
    future_return_col: str,
    past_return_col: str,
    mae_col: str,
    horizon: int,
    window: int,
    min_regime_frac: float,
) -> pd.DataFrame:
    states = df[state_col].values
    total = len(df)
    regime_ids = np.unique(states)
    stats = []
    for rid in regime_ids:
        mask = df[state_col] == rid
        subset = df[mask]
        count = len(subset)
        frac = count / total if total else 0.0
        avg_duration = _average_duration(states, rid)
        transition_risk = _transition_risk(states, rid, window)
        future_returns = subset[future_return_col]
        past_returns = subset[past_return_col]
        proxy_dir = np.sign(past_returns)
        valid = proxy_dir != 0
        proxy_wins = (proxy_dir[valid] * future_returns[valid]) > 0
        win_rate = float(proxy_wins.mean()) if valid.any() else np.nan
        tail_loss = float(np.quantile(future_returns, 0.05)) if count else np.nan
        stats.append(
            {
                "regime_id": int(rid),
                "count": int(count),
                "fraction": frac,
                "avg_duration": avg_duration,
                "transition_risk": transition_risk,
                "win_rate": win_rate,
                "mean_return": float(future_returns.mean()) if count else np.nan,
                "median_return": float(future_returns.median()) if count else np.nan,
                "tail_loss_p05": tail_loss,
                "mae_median": float(subset[mae_col].median()) if count else np.nan,
                "horizon_bars": horizon,
                "below_min_frac": frac < min_regime_frac,
            }
        )
    return pd.DataFrame(stats).sort_values("regime_id").reset_index(drop=True)


def map_eligibility(
    stats: pd.DataFrame,
    min_duration: float,
    tail_loss_max: float,
    win_rate_min: float,
    transition_risk_max: float,
    min_regime_frac: float,
) -> EligibilityResult:
    rules = []
    eligible_regimes = set()
    for _, row in stats.iterrows():
        reasons = []
        if row["avg_duration"] < min_duration:
            reasons.append("duration")
        if row["tail_loss_p05"] < -tail_loss_max:
            reasons.append("tail_loss")
        if np.isnan(row["win_rate"]) or row["win_rate"] < win_rate_min:
            reasons.append("win_rate")
        if row["transition_risk"] > transition_risk_max:
            reasons.append("transition_risk")
        if row["fraction"] < min_regime_frac:
            reasons.append("min_frac")
        eligible = len(reasons) == 0
        if eligible:
            eligible_regimes.add(int(row["regime_id"]))
        rules.append("eligible" if eligible else f"excluded:{','.join(reasons)}")

    stats = stats.copy()
    stats["eligible"] = [r.startswith("eligible") for r in rules]
    stats["rule_reason"] = rules
    return EligibilityResult(stats=stats, eligible_regimes=eligible_regimes)
