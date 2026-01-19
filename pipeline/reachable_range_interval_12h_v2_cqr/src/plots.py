from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_timeline(df: pd.DataFrame, metric: str, path: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for (target, gate_flag), group in df.groupby(["target", "gate_flag"]):
        label = f"{target}-{'gate' if gate_flag else 'all'}"
        ax.plot(group["window_id"], group[metric], marker="o", label=label)
    ax.set_xlabel("Window ID")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    x_labels: Iterable[str],
    y_labels: Iterable[str],
    title: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    x_labels = list(x_labels)
    y_labels = list(y_labels)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gate_vs_all(df: pd.DataFrame, path: str, title: str) -> None:
    if df.empty:
        return
    targets = df["target"].unique().tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(targets))
    width = 0.35
    gate_cov = []
    all_cov = []
    for target in targets:
        subset = df[df["target"] == target]
        gate_cov.append(subset[subset["gate_flag"]]["coverage"].median())
        all_cov.append(subset[~subset["gate_flag"]]["coverage"].median())
    ax.bar(x - width / 2, all_cov, width, label="all")
    ax.bar(x + width / 2, gate_cov, width, label="gate")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Coverage (median)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
