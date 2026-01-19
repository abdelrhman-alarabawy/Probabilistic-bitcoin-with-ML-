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
