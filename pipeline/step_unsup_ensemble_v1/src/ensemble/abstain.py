from __future__ import annotations

import numpy as np

from ..utils import LABELS, validate_prob_rows


def _entropy(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(prob * np.log(np.clip(prob, eps, 1.0)), axis=1)


def decisions_from_probs(
    probs: np.ndarray,
    tau_trade: float,
    tau_margin: float,
    tau_entropy: float | None,
    abstain_label: str = "skip",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_prob_rows(probs)
    if abstain_label not in LABELS:
        raise ValueError(f"abstain_label must be one of {LABELS}")

    idx_sorted = np.argsort(-probs, axis=1)
    top1_idx = idx_sorted[:, 0]
    top2_idx = idx_sorted[:, 1]
    top1_prob = probs[np.arange(len(probs)), top1_idx]
    top2_prob = probs[np.arange(len(probs)), top2_idx]
    margin = top1_prob - top2_prob
    ent = _entropy(probs)

    ok = (top1_prob >= tau_trade) & (margin >= tau_margin)
    if tau_entropy is not None:
        ok = ok & (ent <= tau_entropy)

    decisions = np.array([abstain_label] * len(probs), dtype=object)
    predicted = np.array([LABELS[i] for i in top1_idx], dtype=object)
    decisions[ok] = predicted[ok]
    meta = {
        "p_max": top1_prob,
        "margin": margin,
        "entropy": ent,
    }
    return decisions, meta
