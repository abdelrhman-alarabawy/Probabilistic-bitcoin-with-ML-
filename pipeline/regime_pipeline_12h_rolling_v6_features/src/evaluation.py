from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    correct_count: int


def compute_trade_metrics(df: pd.DataFrame, decision_col: str, label_col: str) -> TradeMetrics:
    total = len(df)
    trades = df[df[decision_col] != "skip"]
    trade_count = len(trades)
    coverage = trade_count / total if total else 0.0
    if trade_count == 0:
        return TradeMetrics(coverage, 0.0, 0, 0.0, 0.0, 0)
    correct = trades[label_col] == trades[decision_col]
    precision = correct.mean()
    long_trades = trades[trades[decision_col] == "long"]
    short_trades = trades[trades[decision_col] == "short"]
    precision_long = (long_trades[label_col] == "long").mean() if len(long_trades) else 0.0
    precision_short = (short_trades[label_col] == "short").mean() if len(short_trades) else 0.0
    return TradeMetrics(coverage, precision, trade_count, precision_long, precision_short, int(correct.sum()))


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


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1 + (z**2) / n
    center = phat + (z**2) / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return max((center - margin) / denom, 0.0)


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
    plt.xlabel("Value")
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


def timeline_plot(
    window_starts: Iterable[pd.Timestamp],
    window_labels: Iterable[str],
    title: str,
    path: str,
) -> None:
    starts = pd.Series(window_starts)
    labels = pd.Series(window_labels)
    colors = labels.map({"GOOD": "green", "BAD": "red", "INSUFFICIENT": "orange"}).fillna("gray")
    plt.figure(figsize=(10, 2.5))
    plt.scatter(starts, np.ones(len(starts)), c=colors, s=40)
    plt.yticks([])
    plt.xlabel("Window start")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def confusion_matrix_plot(cm: np.ndarray, labels: List[str], title: str, path: str) -> None:
    plt.figure(figsize=(4, 3))
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
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def correlation_heatmap(corr: pd.DataFrame, title: str, path: str) -> None:
    plt.figure(figsize=(10, 8))
    if SEABORN_AVAILABLE:
        sns.heatmap(corr, cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
    plt.title(title)
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
    plt.figure(figsize=(4, 3))
    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")
    else:
        plt.imshow(matrix, cmap="viridis")
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def regime_scatter_plot(embedding: np.ndarray, regimes: np.ndarray, title: str, path: str) -> None:
    if embedding.shape[1] < 2:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=regimes, cmap="tab20", s=8, alpha=0.8)
    plt.title(title)
    plt.xlabel("Comp 1")
    plt.ylabel("Comp 2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def feature_shift_plot(
    stats_df: pd.DataFrame,
    feature_cols: List[str],
    title: str,
    path: str,
) -> None:
    plt.figure(figsize=(10, 4))
    for col in feature_cols:
        if col in stats_df.columns:
            plt.plot(stats_df["window_id"], stats_df[col], label=col)
    plt.title(title)
    plt.xlabel("Window")
    plt.ylabel("Median value")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    if len(y_true) == 0:
        return np.zeros((len(labels), len(labels)), dtype=int)
    return confusion_matrix(y_true, y_pred, labels=labels)
