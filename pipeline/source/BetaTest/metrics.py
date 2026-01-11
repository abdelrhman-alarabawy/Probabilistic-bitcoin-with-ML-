from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def trade_metrics(
    y_true: List[int],
    y_pred: List[int],
    proba: Optional[List[float]] = None,
) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = None
    if proba is not None and len(set(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = None
    return {
        "precision_trade": float(precision),
        "recall_trade": float(recall),
        "f1_trade": float(f1),
        "auc_trade": float(auc) if auc is not None else None,
    }


def direction_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, object]:
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    report = {
        f"precision_{label}": float(precisions[idx])
        for idx, label in enumerate(labels)
    }
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    report["confusion"] = confusion
    return report


def multiclass_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    return {"report": report, "confusion": confusion}