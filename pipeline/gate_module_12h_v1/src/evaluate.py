from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve

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


def confusion_to_array(stats: ConfusionStats) -> np.ndarray:
    return np.array([[stats.tn, stats.fp], [stats.fn, stats.tp]], dtype=int)


def compute_gate_diagnostics(y_true: np.ndarray, p_trade: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(p_trade)
    if not mask.any():
        return float("nan"), float("nan")
    ap = average_precision_score(y_true[mask], p_trade[mask])
    brier = brier_score_loss(y_true[mask], p_trade[mask])
    return float(ap), float(brier)


def pr_curve_plot(
    curves: List[Tuple[np.ndarray, np.ndarray, float, str]],
    path: str,
    ncols: int = 3,
) -> None:
    n = len(curves)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (precision, recall, ap, title) in zip(axes, curves):
        ax.plot(recall, precision, color="tab:blue")
        ax.set_title(f"{title}\nAP={ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[len(curves) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
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


def precision_vs_k_plot(
    summary: pd.DataFrame,
    path: str,
) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(summary["K"], summary["gate_median_precision"], marker="o", label="Gate median precision")
    ax.plot(summary["K"], summary["random_mean_precision"], marker="o", label="Random-K mean precision")
    ax.plot(summary["K"], summary["vol_mean_precision"], marker="o", label="Volatility top-K mean precision")
    ax.plot(summary["K"], summary["always_precision"], marker="o", label="Always-trade precision")
    ax.set_xlabel("K")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_pr_curve(y_true: np.ndarray, p_trade: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(p_trade)
    if not mask.any():
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    precision, recall, _ = precision_recall_curve(y_true[mask], p_trade[mask])
    return precision, recall


def aggregate_counts(rows: List[Dict]) -> Dict[int, ConfusionStats]:
    totals: Dict[int, ConfusionStats] = {}
    for row in rows:
        k = int(row["K"])
        stats = totals.get(k, ConfusionStats(tp=0, fp=0, tn=0, fn=0))
        totals[k] = ConfusionStats(
            tp=stats.tp + int(row["tp"]),
            fp=stats.fp + int(row["fp"]),
            tn=stats.tn + int(row["tn"]),
            fn=stats.fn + int(row["fn"]),
        )
    return totals
