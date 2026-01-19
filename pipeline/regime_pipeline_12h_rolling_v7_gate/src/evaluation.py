from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SEABORN_AVAILABLE = False


@dataclass
class ConfusionStats:
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

    @property
    def fpr(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionStats:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return ConfusionStats(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def confusion_matrix_plot_grid(
    matrices: List[np.ndarray],
    titles: List[str],
    labels: List[str],
    path: str,
) -> None:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, cm, title in zip(axes, matrices, titles):
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_baseline(stats: List[ConfusionStats]) -> dict:
    precision = [s.precision for s in stats]
    recall = [s.recall for s in stats]
    fpr = [s.fpr for s in stats]
    return {
        "precision_mean": float(np.mean(precision)) if precision else float("nan"),
        "precision_std": float(np.std(precision)) if precision else float("nan"),
        "recall_mean": float(np.mean(recall)) if recall else float("nan"),
        "recall_std": float(np.std(recall)) if recall else float("nan"),
        "fpr_mean": float(np.mean(fpr)) if fpr else float("nan"),
        "fpr_std": float(np.std(fpr)) if fpr else float("nan"),
    }


def confusion_to_array(stats: ConfusionStats) -> np.ndarray:
    return np.array([[stats.tn, stats.fp], [stats.fn, stats.tp]], dtype=int)


def median_metric(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return float("nan")
    return float(np.nanmedian(df[col]))
