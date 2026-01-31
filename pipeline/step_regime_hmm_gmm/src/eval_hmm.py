from __future__ import annotations

import numpy as np


def occupancy_balance(occupancy: dict) -> float:
    if not occupancy:
        return float("nan")
    probs = np.array(list(occupancy.values()), dtype=float)
    probs = probs / np.clip(probs.sum(), 1.0e-12, None)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)))
    if probs.size <= 1:
        return 0.0
    return float(entropy / np.log(probs.size))
