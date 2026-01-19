from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .evaluate import ConfusionStats, compute_confusion


@dataclass
class BaselineSummary:
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    fpr_mean: float
    fpr_std: float


def summarize_baseline(stats: List[ConfusionStats]) -> BaselineSummary:
    precision = [s.precision for s in stats]
    recall = [s.recall for s in stats]
    fpr = [s.fpr for s in stats]
    return BaselineSummary(
        precision_mean=float(np.mean(precision)) if precision else float("nan"),
        precision_std=float(np.std(precision)) if precision else float("nan"),
        recall_mean=float(np.mean(recall)) if recall else float("nan"),
        recall_std=float(np.std(recall)) if recall else float("nan"),
        fpr_mean=float(np.mean(fpr)) if fpr else float("nan"),
        fpr_std=float(np.std(fpr)) if fpr else float("nan"),
    )


def random_k_baseline(
    y_true: np.ndarray,
    k: int,
    reps: int,
    rng: np.random.Generator,
) -> BaselineSummary:
    n = len(y_true)
    k = min(k, n)
    stats: List[ConfusionStats] = []
    for _ in range(reps):
        idx = rng.choice(n, size=k, replace=False)
        y_pred = np.zeros(n, dtype=int)
        y_pred[idx] = 1
        stats.append(compute_confusion(y_true, y_pred))
    return summarize_baseline(stats)


def volatility_topk_baseline(
    y_true: np.ndarray,
    values: Optional[np.ndarray],
    k: int,
) -> Optional[ConfusionStats]:
    if values is None:
        return None
    n = len(values)
    k = min(k, n)
    values_safe = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(values_safe)[::-1]
    idx = order[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[idx] = 1
    return compute_confusion(y_true, y_pred)


def always_trade_baseline(y_true: np.ndarray) -> ConfusionStats:
    y_pred = np.ones_like(y_true)
    return compute_confusion(y_true, y_pred)
