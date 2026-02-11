from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import LABELS, validate_prob_rows


def build_state_label_table(
    zhard_train: np.ndarray,
    labels_train: pd.Series,
    n_states: int,
    alpha: float,
) -> dict:
    counts = np.zeros((n_states, len(LABELS)), dtype=float)
    label_to_idx = {label: i for i, label in enumerate(LABELS)}

    for z, label in zip(zhard_train.astype(int), labels_train.astype(str)):
        if label in label_to_idx and 0 <= z < n_states:
            counts[z, label_to_idx[label]] += 1.0

    probs = (counts + alpha) / ((counts + alpha).sum(axis=1, keepdims=True))
    validate_prob_rows(probs)
    return {
        "labels": LABELS,
        "counts": counts,
        "probs": probs,
    }


def state_posteriors_to_label_probs(
    zprob_all: np.ndarray,
    state_label_probs: np.ndarray,
) -> np.ndarray:
    out = zprob_all @ state_label_probs
    validate_prob_rows(out)
    return out
