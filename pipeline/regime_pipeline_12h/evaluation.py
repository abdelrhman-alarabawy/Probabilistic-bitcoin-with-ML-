from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SEABORN_AVAILABLE = False


@dataclass
class TradeMetrics:
    coverage: float
    precision: float
    trade_count: int
    avg_return: float
    tail_loss_p05: float


def direction_label(future_return: float, threshold: float) -> int:
    if future_return > threshold:
        return 1
    if future_return < -threshold:
        return -1
    return 0


def compute_trade_metrics(
    df: pd.DataFrame,
    decision_col: str,
    future_return_col: str,
    threshold: float,
) -> TradeMetrics:
    total = len(df)
    trades = df[df[decision_col] != "skip"].copy()
    trade_count = len(trades)
    coverage = trade_count / total if total else 0.0
    if trade_count == 0:
        return TradeMetrics(coverage, 0.0, 0, float("nan"), float("nan"))

    true_dir = trades[future_return_col].apply(lambda r: direction_label(r, threshold))
    decision_dir = trades[decision_col].map({"long": 1, "short": -1}).fillna(0)
    correct = (true_dir == decision_dir) & (true_dir != 0)
    precision = correct.sum() / max((true_dir != 0).sum(), 1)
    avg_return = float(trades[future_return_col].mean())
    tail_loss = float(np.quantile(trades[future_return_col], 0.05))
    return TradeMetrics(coverage, precision, trade_count, avg_return, tail_loss)


def confusion_matrix_plot(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    path: str,
) -> None:
    plt.figure(figsize=(5, 4))
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def transition_matrix_plot(matrix: np.ndarray, title: str, path: str) -> None:
    plt.figure(figsize=(5, 4))
    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")
    else:
        plt.imshow(matrix, cmap="viridis")
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def regime_timeline_plot(
    timestamps: Iterable,
    regimes: Iterable,
    title: str,
    path: str,
) -> None:
    ts = pd.Series(timestamps)
    rg = pd.Series(regimes)
    plt.figure(figsize=(10, 3))
    plt.scatter(ts, rg, c=rg, cmap="tab20", s=6)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    path: str,
) -> None:
    if len(y_true) == 0:
        plt.figure(figsize=(5, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def returns_distribution_plot(
    df: pd.DataFrame,
    regime_col: str,
    return_col: str,
    title: str,
    path: str,
) -> None:
    plt.figure(figsize=(10, 4))
    if SEABORN_AVAILABLE:
        sns.violinplot(data=df, x=regime_col, y=return_col, inner="quartile")
    else:
        grouped = [group[return_col].values for _, group in df.groupby(regime_col)]
        plt.boxplot(grouped, labels=sorted(df[regime_col].unique()))
    plt.title(title)
    plt.xlabel("Regime")
    plt.ylabel("Future return")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    if len(y_true) == 0:
        return np.zeros((len(labels), len(labels)), dtype=int)
    return confusion_matrix(y_true, y_pred, labels=labels)
