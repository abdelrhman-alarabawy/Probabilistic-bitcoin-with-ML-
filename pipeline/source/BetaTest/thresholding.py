from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


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


def sweep_thresholds_multiclass(
    proba: np.ndarray,
    y_true: List[str],
    class_order: List[str],
    thresholds: List[float],
    min_coverage: float,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
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
    best: Optional[Dict[str, float]] = None

    for th_long in thresholds:
        for th_short in thresholds:
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

            row = {
                "th_long": th_long,
                "th_short": th_short,
                "precision_long": precision_long,
                "precision_short": precision_short,
                "coverage": coverage,
                "trade_precision": trade_precision,
                "score": score,
            }
            rows.append(row)

            if best is None:
                best = row
                continue
            if row["score"] > best["score"]:
                best = row
            elif row["score"] == best["score"]:
                if row["trade_precision"] > best["trade_precision"]:
                    best = row
                elif row["trade_precision"] == best["trade_precision"]:
                    if row["coverage"] > best["coverage"]:
                        best = row

    grid_df = pd.DataFrame(rows, columns=columns)
    if best is None:
        print(
            f"No feasible thresholds found for MIN_COVERAGE={min_coverage}. "
            "Lower MIN_COVERAGE or expand grid."
        )
    return grid_df, best


def multiclass_prob_stats(
    proba: np.ndarray,
    class_order: List[str],
) -> Dict[str, float]:
    idx_long = class_order.index("long")
    idx_short = class_order.index("short")
    idx_skip = class_order.index("skip")

    argmax_idx = np.argmax(proba, axis=1)
    argmax_long_pct = float((argmax_idx == idx_long).mean())
    argmax_short_pct = float((argmax_idx == idx_short).mean())
    argmax_skip_pct = float((argmax_idx == idx_skip).mean())

    mean_p_long = float(proba[:, idx_long].mean())
    mean_p_short = float(proba[:, idx_short].mean())
    mean_p_skip = float(proba[:, idx_skip].mean())

    return {
        "argmax_long_pct": argmax_long_pct,
        "argmax_short_pct": argmax_short_pct,
        "argmax_skip_pct": argmax_skip_pct,
        "mean_p_long": mean_p_long,
        "mean_p_short": mean_p_short,
        "mean_p_skip": mean_p_skip,
    }


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
        rows.append(
            {
                time_col: df_test.iloc[idx][time_col],
                "true_label": y_true[idx],
                "argmax_label": argmax_labels[idx],
                "p_long": float(proba[idx, class_order.index("long")]),
                "p_short": float(proba[idx, class_order.index("short")]),
                "p_skip": float(proba[idx, class_order.index("skip")]),
                "final_decision": preds[idx],
            }
        )
    return pd.DataFrame(rows)
