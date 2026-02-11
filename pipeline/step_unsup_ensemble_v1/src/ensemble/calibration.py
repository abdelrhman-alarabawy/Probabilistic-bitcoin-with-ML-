from __future__ import annotations

import numpy as np

from ..utils import validate_prob_rows


def maybe_calibrate_probs(probs: np.ndarray, calibration_cfg: dict) -> np.ndarray:
    # Placeholder for temperature scaling if enabled later.
    _ = calibration_cfg
    validate_prob_rows(probs)
    return probs
