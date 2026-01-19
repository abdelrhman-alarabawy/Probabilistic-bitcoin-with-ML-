from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SEABORN_AVAILABLE = False


@dataclass
class TradeMetrics:
    coverage: float
    precision_dir: float
    trade_count: int
    precision_long: float
    precision_short: float


def compute_trade_metrics(df: pd.DataFrame, decision_col: str, label_col: str) -> TradeMetrics:
    total = len(df)
    trades = df[df[decision_col] != "skip"]
    trade_count = len(trades)
    coverage = trade_count / total if total else 0.0
    if trade_count == 0:
        return TradeMetrics(coverage, 0.0, 0, 0.0, 0.0)
    correct = trades[label_col] == trades[decision_col]
    precision = correct.mean()
    long_trades = trades[trades[decision_col] == "long"]
    short_trades = trades[trades[decision_col] == "short"]
    precision_long = (long_trades[label_col] == "long").mean() if len(long_trades) else 0.0
    precision_short = (short_trades[label_col] == "short").mean() if len(short_trades) else 0.0
    return TradeMetrics(coverage, precision, trade_count, precision_long, precision_short)


def gate_metrics_from_decisions(
    y_true_gate: np.ndarray,
    trade_mask: np.ndarray,
) -> Tuple[float, float, float, int]:
    tp = (trade_mask & (y_true_gate == 1)).sum()
    fp = (trade_mask & (y_true_gate == 0)).sum()
    fn = (~trade_mask & (y_true_gate == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    coverage = trade_mask.mean() if len(trade_mask) else 0.0
    return precision, recall, coverage, int(trade_mask.sum())


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    pareto = []
    for idx, row in df.iterrows():
        dominates = df[(df[x_col] >= row[x_col]) & (df[y_col] >= row[y_col])]
        if len(dominates) == 0:
            pareto.append(True)
            continue
        if len(dominates) == 1 and dominates.index[0] == idx:
            pareto.append(True)
            continue
        is_dominated = ((dominates[x_col] > row[x_col]) | (dominates[y_col] > row[y_col])).any()
        pareto.append(not is_dominated)
    return pd.Series(pareto, index=df.index)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    if len(y_true) == 0:
        return np.zeros((len(labels), len(labels)), dtype=int)
    return confusion_matrix(y_true, y_pred, labels=labels)


def confusion_matrix_plot(cm: np.ndarray, labels: List[str], title: str, path: str) -> None:
    plt.figure(figsize=(5, 4))
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def regime_timeline_plot(timestamps: Iterable, regimes: Iterable, title: str, path: str) -> None:
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


def calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: str) -> None:
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


def pr_curve_plot(y_true: np.ndarray, y_score: np.ndarray, title: str, path: str) -> float:
    if len(y_true) == 0:
        plt.figure(figsize=(5, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "No PR data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return ap


def distribution_plot(values: np.ndarray, title: str, path: str, bins: int = 20) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def frontier_plot(df: pd.DataFrame, title: str, path: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(df["coverage"], df["precision_dir"], alpha=0.6, label="Grid")
    pareto = df[df["pareto"]]
    if len(pareto) > 0:
        plt.scatter(pareto["coverage"], pareto["precision_dir"], color="red", label="Pareto")
    plt.title(title)
    plt.xlabel("Coverage")
    plt.ylabel("Direction Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def heatmap_plot(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    path: str,
) -> None:
    plt.figure(figsize=(10, 4))
    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, annot=False, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels)
    else:
        plt.imshow(matrix, aspect="auto", cmap="viridis")
        plt.yticks(range(len(y_labels)), y_labels)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45)
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def timeline_plot(
    window_starts: Iterable[pd.Timestamp],
    window_labels: Iterable[str],
    title: str,
    path: str,
) -> None:
    starts = pd.Series(window_starts)
    labels = pd.Series(window_labels)
    colors = labels.map({"GOOD": "green", "BAD": "red"}).fillna("gray")
    plt.figure(figsize=(10, 2.5))
    plt.scatter(starts, np.ones(len(starts)), c=colors, s=40)
    plt.yticks([])
    plt.xlabel("Window start")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
