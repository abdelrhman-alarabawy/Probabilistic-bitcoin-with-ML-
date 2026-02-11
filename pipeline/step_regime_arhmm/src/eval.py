from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


LABELS = ("long", "short", "skip")


def num_params_arhmm(k: int, d: int) -> int:
    trans = k * (k - 1)
    pi = k - 1
    ar = k * (d * d + d)
    cov = k * (d * (d + 1) // 2)
    return int(trans + pi + ar + cov)


def aic_bic(loglik: float, n_params: int, n_obs: int) -> Tuple[float, float]:
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + n_params * np.log(max(n_obs, 1))
    return float(aic), float(bic)


def entropy_stats(gamma: np.ndarray) -> Dict[str, float]:
    if gamma.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    k = gamma.shape[1]
    ent = -np.sum(gamma * np.log(np.clip(gamma, 1e-12, None)), axis=1)
    ent = ent / np.log(k)
    return {
        "mean": float(np.mean(ent)),
        "p50": float(np.quantile(ent, 0.50)),
        "p90": float(np.quantile(ent, 0.90)),
    }


def _map_label(value) -> str:
    if value is None:
        return "skip"
    if isinstance(value, float) and np.isnan(value):
        return "skip"
    if isinstance(value, (int, np.integer)):
        if value > 0:
            return "long"
        if value < 0:
            return "short"
        return "skip"
    if isinstance(value, (float, np.floating)):
        if value > 0:
            return "long"
        if value < 0:
            return "short"
        return "skip"
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("long", "buy", "1", "l"):
            return "long"
        if s in ("short", "sell", "-1", "s"):
            return "short"
        if s in ("skip", "flat", "hold", "0", "neutral", "none"):
            return "skip"
    return "skip"


def _label_counts(labels: Iterable[str], weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    counts = {label: 0.0 for label in LABELS}
    if weights is None:
        for lab in labels:
            counts[lab] += 1.0
    else:
        for lab, w in zip(labels, weights):
            counts[lab] += float(w)
    total = sum(counts.values())
    if total <= 0:
        pct = {label: 0.0 for label in LABELS}
    else:
        pct = {label: counts[label] / total for label in LABELS}
    return {"counts": counts, "pct": pct, "total": float(total)}


def label_composition(labels: Optional[np.ndarray], gamma: np.ndarray) -> Optional[Dict[str, Dict]]:
    if labels is None:
        return None
    mapped = np.array([_map_label(v) for v in labels])
    t, k = gamma.shape

    hard_states = np.argmax(gamma, axis=1)

    hard = {}
    soft = {}
    for state in range(k):
        idx = hard_states == state
        hard_labels = mapped[idx]
        hard[state] = _label_counts(hard_labels)

        soft_weights = gamma[:, state]
        soft[state] = _label_counts(mapped, weights=soft_weights)

    return {"hard": hard, "soft": soft}


def tradeable_states(label_stats: Dict[str, Dict], top_n: int = 2) -> List[Tuple[int, float]]:
    if label_stats is None:
        return []
    soft = label_stats.get("soft") or {}
    scores = []
    for state, stats in soft.items():
        pct = stats.get("pct", {})
        score = float(pct.get("long", 0.0) + pct.get("short", 0.0))
        scores.append((int(state), score))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_n]
