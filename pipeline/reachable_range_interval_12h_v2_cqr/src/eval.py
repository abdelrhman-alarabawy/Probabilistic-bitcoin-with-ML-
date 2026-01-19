from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)


def ensure_order(lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.minimum(lower, upper), np.maximum(lower, upper)


def compute_hits(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (y_true >= lower) & (y_true <= upper)


def compute_tight_confusion(hits: np.ndarray, tight: np.ndarray) -> ConfusionCounts:
    tp = int(np.sum(hits & tight))
    fp = int(np.sum(~hits & tight))
    fn = int(np.sum(hits & ~tight))
    tn = int(np.sum(~hits & ~tight))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def summarize_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    width_thresholds: Iterable[float],
    nominal_coverage: float,
) -> Tuple[Dict[str, float], Dict[str, ConfusionCounts], Dict[str, Dict[str, float]]]:
    lower, upper = ensure_order(lower, upper)
    hits = compute_hits(y_true, lower, upper)
    width_pct = upper - lower

    metrics = {
        "coverage": float(np.mean(hits)) if len(hits) else float("nan"),
        "width_pct_mean": float(np.mean(width_pct)) if len(width_pct) else float("nan"),
        "width_pct_median": float(np.median(width_pct)) if len(width_pct) else float("nan"),
    }
    metrics["coverage_gap"] = metrics["coverage"] - nominal_coverage

    confusions: Dict[str, ConfusionCounts] = {}
    tight_metrics: Dict[str, Dict[str, float]] = {}

    if len(width_pct):
        p10 = float(np.nanpercentile(width_pct, 10))
        p25 = float(np.nanpercentile(width_pct, 25))
    else:
        p10 = float("nan")
        p25 = float("nan")

    thresholds = {f"W{value:.3f}": value for value in width_thresholds}
    thresholds["p10"] = p10
    thresholds["p25"] = p25

    for key, threshold in thresholds.items():
        if not np.isfinite(threshold):
            continue
        tight = width_pct <= threshold
        confusion = compute_tight_confusion(hits, tight)
        confusions[key] = confusion
        coverage_tight = float(np.mean(tight)) if len(tight) else float("nan")
        precision_tight = confusion.precision if np.isfinite(confusion.precision) else float("nan")
        tight_metrics[key] = {
            "precision_tight": precision_tight,
            "coverage_tight": coverage_tight,
        }

    return metrics, confusions, tight_metrics
