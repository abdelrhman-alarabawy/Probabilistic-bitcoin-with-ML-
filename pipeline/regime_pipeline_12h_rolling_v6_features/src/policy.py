from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def decide(
    eligible: bool,
    entropy: float,
    entropy_max: float,
    p_trade: Optional[float],
    gate_threshold: float,
    p_long: Optional[float],
    p_short: Optional[float],
    direction_threshold: float,
) -> Tuple[str, str]:
    if not eligible:
        return "skip", "not_eligible"
    if not np.isfinite(entropy) or entropy > entropy_max:
        return "skip", "entropy_gate"
    if p_trade is None or not np.isfinite(p_trade):
        return "skip", "no_gate_model"
    if p_trade < gate_threshold:
        return "skip", "gate_prob"
    if p_long is None or p_short is None:
        return "skip", "no_direction_model"
    if max(p_long, p_short) < direction_threshold:
        return "skip", "direction_prob"
    decision = "long" if p_long >= p_short else "short"
    return decision, "eligible"


def apply_policy_batch(
    eligible_flags: np.ndarray,
    entropies: np.ndarray,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    gate_threshold: float,
    direction_threshold: float,
    entropy_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    decisions = []
    reasons = []
    for i in range(len(eligible_flags)):
        decision, reason = decide(
            eligible=bool(eligible_flags[i]),
            entropy=float(entropies[i]),
            entropy_max=entropy_max,
            p_trade=float(p_trade[i]) if np.isfinite(p_trade[i]) else None,
            gate_threshold=gate_threshold,
            p_long=float(p_long[i]) if np.isfinite(p_long[i]) else None,
            p_short=float(p_short[i]) if np.isfinite(p_short[i]) else None,
            direction_threshold=direction_threshold,
        )
        decisions.append(decision)
        reasons.append(reason)
    return np.array(decisions, dtype=object), np.array(reasons, dtype=object)
