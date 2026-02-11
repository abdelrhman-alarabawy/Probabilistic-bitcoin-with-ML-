from __future__ import annotations

import numpy as np

from ..utils import validate_prob_rows


def combine_weighted_average(model_probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(model_probs) == 0:
        raise ValueError("No model probabilities provided.")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(model_probs):
        raise ValueError("Weights shape mismatch.")
    w = w / np.sum(w)
    out = np.zeros_like(model_probs[0], dtype=float)
    for p, wi in zip(model_probs, w):
        out += wi * p
    validate_prob_rows(out)
    return out


def combine_poe(model_probs: list[np.ndarray], weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(model_probs) == 0:
        raise ValueError("No model probabilities provided.")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(model_probs):
        raise ValueError("Weights shape mismatch.")
    w = w / np.sum(w)
    logp = np.zeros_like(model_probs[0], dtype=float)
    for p, wi in zip(model_probs, w):
        logp += wi * np.log(np.clip(p, eps, 1.0))
    max_log = np.max(logp, axis=1, keepdims=True)
    exp_log = np.exp(logp - max_log)
    out = exp_log / exp_log.sum(axis=1, keepdims=True)
    validate_prob_rows(out)
    return out
