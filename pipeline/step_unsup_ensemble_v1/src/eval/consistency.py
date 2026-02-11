from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ..utils import LABELS, save_json


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_consistency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_json: Path,
    out_png: Path,
) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    coverage = float(np.mean(y_pred != "skip"))
    precision = float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = {
        "coverage": coverage,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "labels": LABELS,
    }
    save_json(out_json, report)
    _plot_confusion_matrix(cm, out_png, title=out_json.stem)
    return report
