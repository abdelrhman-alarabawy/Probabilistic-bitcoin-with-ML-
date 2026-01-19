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


def _leave_prob(states: np.ndarray, regime_id: int) -> float:
    leave_flags = []
    n = len(states)
    for i in range(n - 1):
        if states[i] != regime_id:
            continue
        leave_flags.append(1.0 if states[i + 1] != regime_id else 0.0)
    return float(np.mean(leave_flags)) if leave_flags else 1.0


def compute_regime_stats(
    df: pd.DataFrame,
    state_col: str,
    label_col: str,
    entropy_col: str,
) -> pd.DataFrame:
    states = df[state_col].values
    regimes = np.unique(states)
    stats = []
    for rid in regimes:
        subset = df[df[state_col] == rid]
        counts = subset[label_col].value_counts()
        total = len(subset)
        p_skip = counts.get("skip", 0) / total if total else 0.0
        p_long = counts.get("long", 0) / total if total else 0.0
        p_short = counts.get("short", 0) / total if total else 0.0
        non_skip = p_long + p_short
        if non_skip > 0:
            purity = max(p_long, p_short) / non_skip
            dominant = "long" if p_long >= p_short else "short"
        else:
            purity = 0.0
            dominant = "none"
        stats.append(
            {
                "regime_id": int(rid),
                "count": int(total),
                "action_rate": 1.0 - p_skip,
                "p_long": p_long,
                "p_short": p_short,
                "p_skip": p_skip,
                "direction_purity": purity,
                "dominant_direction": dominant,
                "avg_duration": _average_duration(states, rid),
                "leave_prob": _leave_prob(states, rid),
                "entropy_mean": float(subset[entropy_col].mean()),
                "entropy_p90": float(subset[entropy_col].quantile(0.9)),
            }
        )
    return pd.DataFrame(stats).sort_values("regime_id").reset_index(drop=True)


def map_eligibility(
    stats: pd.DataFrame,
    min_action_rate: float,
    min_purity: float,
    min_duration: float,
    max_leave_prob: float,
) -> EligibilityResult:
    eligible = set()
    rule_notes = []
    for _, row in stats.iterrows():
        reasons = []
        if row["action_rate"] < min_action_rate:
            reasons.append("action_rate")
        if row["direction_purity"] < min_purity:
            reasons.append("purity")
        if row["avg_duration"] < min_duration:
            reasons.append("duration")
        if row["leave_prob"] > max_leave_prob:
            reasons.append("leave_prob")
        ok = len(reasons) == 0
        if ok:
            eligible.add(int(row["regime_id"]))
        rule_notes.append("eligible" if ok else f"excluded:{','.join(reasons)}")

    stats = stats.copy()
    stats["eligible"] = [note == "eligible" for note in rule_notes]
    stats["rule_notes"] = rule_notes
    return EligibilityResult(stats=stats, eligible_regimes=eligible)
