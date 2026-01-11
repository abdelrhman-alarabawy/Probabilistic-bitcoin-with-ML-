from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)


@dataclass
class EvaluationResult:
    precision_long: float
    precision_short: float
    coverage: float
    macro_f1: float
    balanced_accuracy: float
    report: Dict
    confusion: np.ndarray


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    class_order: List[str],
) -> EvaluationResult:
    precisions = precision_score(y_true, y_pred, labels=class_order, average=None, zero_division=0)
    precision_long = float(precisions[class_order.index("long")])
    precision_short = float(precisions[class_order.index("short")])
    coverage = float(sum(label in ("long", "short") for label in y_pred) / len(y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=class_order)

    return EvaluationResult(
        precision_long=precision_long,
        precision_short=precision_short,
        coverage=coverage,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
        report=report,
        confusion=confusion,
    )


def save_confusion_matrix_csv(confusion: np.ndarray, class_order: List[str], path: str) -> None:
    df = pd.DataFrame(confusion, index=class_order, columns=class_order)
    df.index.name = "actual"
    df.to_csv(path)


def save_predictions_csv(
    df: pd.DataFrame,
    proba: np.ndarray,
    class_order: List[str],
    output_path: str,
) -> None:
    output = df.copy()
    for idx, label in enumerate(class_order):
        output[f"proba_{label}"] = proba[:, idx]
    output.to_csv(output_path, index=False)