from __future__ import annotations

import numpy as np


def safe_mean(values):
    if not values:
        return float("nan")
    return float(np.mean(values))
