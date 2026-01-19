from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def decide(
    eligible: bool,
    entropy: float,
    entropy_max: float,
    p_long: Optional[float],
    p_short: Optional[float],
    decision_threshold: float,
) -> Tuple[str, Optional[float]]:
    if not eligible:
        return "skip", None
    if not np.isfinite(entropy) or entropy > entropy_max:
        return "skip", None
    if p_long is None or p_short is None:
        return "skip", None
    if max(p_long, p_short) < decision_threshold:
        return "skip", max(p_long, p_short)
    decision = "long" if p_long >= p_short else "short"
    return decision, max(p_long, p_short)
