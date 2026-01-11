from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


@dataclass
class ThresholdBest:
    th_long: float
    th_short: float
    precision_long: float
    precision_short: float
    coverage: float
    trade_precision: float
    score: float


def route_predictions(
    proba: np.ndarray,
    class_order: List[str],
    th_long: float,
    th_short: float,
) -> List[str]:
    idx_long = class_order.index("long")
    idx_short = class_order.index("short")

    preds: List[str] = []
    for row in proba:
        max_idx = int(np.argmax(row))
        if max_idx == idx_long and row[idx_long] >= th_long:
            preds.append("long")
        elif max_idx == idx_short and row[idx_short] >= th_short:
            preds.append("short")
        else:
            preds.append("skip")
    return preds


def _trade_precision(y_true: List[str], y_pred: List[str]) -> float:
    trade_idx = [i for i, pred in enumerate(y_pred) if pred in ("long", "short")]
    if not trade_idx:
        return 0.0
    correct = sum(1 for i in trade_idx if y_pred[i] == y_true[i])
    return correct / len(trade_idx)


def threshold_sweep(
    proba: np.ndarray,
    y_true: List[str],
    class_order: List[str],
    grid: List[float],
    min_coverage: float,
) -> Tuple[pd.DataFrame, ThresholdBest]:
    rows: List[Dict[str, float]] = []
    columns = [
        "th_long",
        "th_short",
        "precision_long",
        "precision_short",
        "coverage",
        "trade_precision",
        "score",
    ]
    best: ThresholdBest = None

    for th_long in grid:
        for th_short in grid:
            preds = route_predictions(proba, class_order, th_long, th_short)
            coverage = sum(1 for p in preds if p in ("long", "short")) / len(preds)
            if coverage < min_coverage:
                continue
            precisions = precision_score(
                y_true,
                preds,
                labels=class_order,
                average=None,
                zero_division=0,
            )
            precision_long = float(precisions[class_order.index("long")])
            precision_short = float(precisions[class_order.index("short")])
            trade_precision = _trade_precision(y_true, preds)
            score = (precision_long + precision_short) / 2
            rows.append(
                {
                    "th_long": th_long,
                    "th_short": th_short,
                    "precision_long": precision_long,
                    "precision_short": precision_short,
                    "coverage": coverage,
                    "trade_precision": trade_precision,
                    "score": score,
                }
            )
            candidate = ThresholdBest(
                th_long,
                th_short,
                precision_long,
                precision_short,
                coverage,
                trade_precision,
                score,
            )
            if best is None:
                best = candidate
            elif candidate.score > best.score:
                best = candidate
            elif candidate.score == best.score and candidate.coverage > best.coverage:
                best = candidate

    if best is None:
        best = ThresholdBest(
            th_long=max(grid),
            th_short=max(grid),
            precision_long=0.0,
            precision_short=0.0,
            coverage=0.0,
            trade_precision=0.0,
            score=0.0,
        )
    if not rows:
        return pd.DataFrame(columns=columns), best
    return pd.DataFrame(rows), best


def build_probability_sample(
    df_test: pd.DataFrame,
    y_true: List[str],
    proba: np.ndarray,
    class_order: List[str],
    th_long: float,
    th_short: float,
    time_col: str,
    sample_size: int = 30,
) -> pd.DataFrame:
    preds = route_predictions(proba, class_order, th_long, th_short)
    argmax_labels = [class_order[int(np.argmax(row))] for row in proba]

    sample_idx = np.linspace(0, len(df_test) - 1, num=min(sample_size, len(df_test)), dtype=int)
    rows = []
    for idx in sample_idx:
        row = {
            time_col: df_test.iloc[idx][time_col],
            "true_label": y_true[idx],
            "argmax_label": argmax_labels[idx],
            "p_long": float(proba[idx, class_order.index("long")]),
            "p_short": float(proba[idx, class_order.index("short")]),
            "p_skip": float(proba[idx, class_order.index("skip")]),
            "final_decision": preds[idx],
        }
        rows.append(row)
    return pd.DataFrame(rows)
