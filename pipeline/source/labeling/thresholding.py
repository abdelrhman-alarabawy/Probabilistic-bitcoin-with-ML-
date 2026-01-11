from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics import precision_score


@dataclass
class ThresholdResult:
    threshold_long: float
    threshold_short: float
    precision_long: float
    precision_short: float
    coverage: float
    score: float


def route_predictions(
    proba: np.ndarray,
    class_order: List[str],
    threshold_long: float,
    threshold_short: float,
) -> List[str]:
    idx_long = class_order.index("long")
    idx_short = class_order.index("short")

    preds: List[str] = []
    for row in proba:
        max_idx = int(np.argmax(row))
        if max_idx == idx_long and row[idx_long] >= threshold_long:
            preds.append("long")
        elif max_idx == idx_short and row[idx_short] >= threshold_short:
            preds.append("short")
        else:
            preds.append("skip")
    return preds


def compute_precision(y_true: List[str], y_pred: List[str], class_order: List[str]) -> Tuple[float, float]:
    precisions = precision_score(y_true, y_pred, labels=class_order, average=None, zero_division=0)
    precision_long = float(precisions[class_order.index("long")])
    precision_short = float(precisions[class_order.index("short")])
    return precision_long, precision_short


def compute_coverage(y_pred: List[str]) -> float:
    if not y_pred:
        return 0.0
    trade_count = sum(1 for label in y_pred if label in ("long", "short"))
    return trade_count / len(y_pred)


def grid_search_thresholds(
    proba: np.ndarray,
    y_true: List[str],
    class_order: List[str],
    threshold_grid: Iterable[float],
    min_trade_coverage: float,
) -> ThresholdResult:
    best: ThresholdResult | None = None

    for th_long in threshold_grid:
        for th_short in threshold_grid:
            preds = route_predictions(proba, class_order, th_long, th_short)
            coverage = compute_coverage(preds)
            if coverage < min_trade_coverage:
                continue
            precision_long, precision_short = compute_precision(y_true, preds, class_order)
            score = precision_long + precision_short
            candidate = ThresholdResult(
                threshold_long=th_long,
                threshold_short=th_short,
                precision_long=precision_long,
                precision_short=precision_short,
                coverage=coverage,
                score=score,
            )
            if best is None:
                best = candidate
                continue
            if candidate.score > best.score:
                best = candidate
            elif candidate.score == best.score and candidate.coverage > best.coverage:
                best = candidate

    if best is None:
        return ThresholdResult(
            threshold_long=max(threshold_grid),
            threshold_short=max(threshold_grid),
            precision_long=0.0,
            precision_short=0.0,
            coverage=0.0,
            score=0.0,
        )
    return best