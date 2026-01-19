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


def compute_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return upper - lower


def compute_width_pct(width: np.ndarray, close_current: np.ndarray) -> np.ndarray:
    denom = np.where(close_current == 0, np.nan, close_current)
    return width / denom


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
    close_current: np.ndarray,
    width_thresholds: Iterable[float],
    nominal_coverage: float,
) -> Tuple[Dict[str, float], Dict[float, ConfusionCounts], Dict[float, Dict[str, float]]]:
    lower, upper = ensure_order(lower, upper)
    hits = compute_hits(y_true, lower, upper)
    width = compute_width(lower, upper)
    width_pct = compute_width_pct(width, close_current)

    metrics = {
        "coverage": float(np.mean(hits)) if len(hits) else float("nan"),
        "width_mean": float(np.mean(width)) if len(width) else float("nan"),
        "width_median": float(np.median(width)) if len(width) else float("nan"),
        "width_pct_mean": float(np.mean(width_pct)) if len(width_pct) else float("nan"),
        "width_pct_median": float(np.median(width_pct)) if len(width_pct) else float("nan"),
    }
    metrics["coverage_gap"] = metrics["coverage"] - nominal_coverage

    confusions: Dict[float, ConfusionCounts] = {}
    tight_metrics: Dict[float, Dict[str, float]] = {}
    for threshold in width_thresholds:
        tight = width_pct <= threshold
        confusion = compute_tight_confusion(hits, tight)
        confusions[threshold] = confusion
        coverage_tight = float(np.mean(tight)) if len(tight) else float("nan")
        precision_tight = confusion.precision if np.isfinite(confusion.precision) else float("nan")
        tight_metrics[threshold] = {
            "precision_tight": precision_tight,
            "coverage_tight": coverage_tight,
        }

    return metrics, confusions, tight_metrics
