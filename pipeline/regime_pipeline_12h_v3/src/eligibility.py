from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class EligibilityResult:
    stats: pd.DataFrame
    eligible_regimes: set


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
    stats_rows: List[dict] = []
    for rid in regimes:
        subset = df[df[state_col] == rid]
        counts = subset[label_col].value_counts()
        n_total = len(subset)
        n_long = int(counts.get("long", 0))
        n_short = int(counts.get("short", 0))
        n_skip = int(counts.get("skip", 0))
        n_action = n_long + n_short
        action_rate = n_action / n_total if n_total else 0.0
        purity_actionable = max(n_long, n_short) / max(1, n_action)
        directional_margin = abs(n_long - n_short) / max(1, n_action)
        skip_dominance = n_skip / n_total if n_total else 0.0
        dominant_direction = "long" if n_long >= n_short else "short"
        if n_action == 0:
            dominant_direction = "none"

        p_long = n_long / n_total if n_total else 0.0
        p_short = n_short / n_total if n_total else 0.0
        p_skip = n_skip / n_total if n_total else 0.0
        stats_rows.append(
            {
                "regime_id": int(rid),
                "n_total": n_total,
                "n_long": n_long,
                "n_short": n_short,
                "n_skip": n_skip,
                "n_actionable": n_action,
                "p_long": p_long,
                "p_short": p_short,
                "p_skip": p_skip,
                "action_rate": action_rate,
                "directional_purity_actionable": purity_actionable,
                "directional_margin": directional_margin,
                "skip_dominance": skip_dominance,
                "effective_samples": n_action,
                "dominant_direction": dominant_direction,
                "avg_duration": _average_duration(states, rid),
                "leave_prob": _leave_prob(states, rid),
                "entropy_mean": float(subset[entropy_col].mean()),
                "entropy_p90": float(subset[entropy_col].quantile(0.9)),
            }
        )

    return pd.DataFrame(stats_rows).sort_values("regime_id").reset_index(drop=True)


def map_eligibility(
    stats: pd.DataFrame,
    min_action_samples: int,
    min_action_rate: float,
    min_purity_actionable: float,
    min_margin: float,
    min_duration: float,
    max_leave_prob: float,
) -> EligibilityResult:
    eligible_regimes = set()
    rule_notes = []
    for _, row in stats.iterrows():
        reasons = []
        if row["n_actionable"] < min_action_samples:
            reasons.append("min_action_samples")
        if row["action_rate"] < min_action_rate:
            reasons.append("action_rate")
        if (row["directional_purity_actionable"] < min_purity_actionable) and (
            row["directional_margin"] < min_margin
        ):
            reasons.append("purity_or_margin")
        if row["avg_duration"] < min_duration:
            reasons.append("duration")
        if row["leave_prob"] > max_leave_prob:
            reasons.append("leave_prob")
        ok = len(reasons) == 0
        if ok:
            eligible_regimes.add(int(row["regime_id"]))
        rule_notes.append("eligible" if ok else f"excluded:{','.join(reasons)}")

    stats = stats.copy()
    stats["eligible"] = [note == "eligible" for note in rule_notes]
    stats["rule_notes"] = rule_notes
    return EligibilityResult(stats=stats, eligible_regimes=eligible_regimes)
