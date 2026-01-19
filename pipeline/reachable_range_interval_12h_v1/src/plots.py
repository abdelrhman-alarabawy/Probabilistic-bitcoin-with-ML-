from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_frontier(df: pd.DataFrame, path: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for gate_flag, group in df.groupby("gate_flag"):
        label = "gate" if gate_flag else "all"
        ax.plot(
            group["width_pct_mean"],
            group["coverage"],
            marker="o",
            linestyle="-",
            label=label,
        )
    ax.set_xlabel("Mean interval width %")
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_tight(df: pd.DataFrame, path: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for gate_flag, group in df.groupby("gate_flag"):
        label = "gate" if gate_flag else "all"
        ax.plot(
            group["width_threshold"],
            group["precision_tight"],
            marker="o",
            linestyle="-",
            label=label,
        )
    ax.set_xlabel("Width threshold")
    ax.set_ylabel("Precision (tight)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
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


def plot_yearly_summary(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    targets = df["target"].unique().tolist() if "target" in df.columns else ["all"]
    nrows = len(targets)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 3.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        subset = df[df["target"] == target] if "target" in df.columns else df
        ax.plot(subset["year"], subset["coverage"], marker="o", label="Coverage")
        ax.plot(subset["year"], subset["width_pct_mean"], marker="o", label="Width % mean")
        ax.set_ylabel("Metric")
        ax.set_title(f"Coverage and Width by Year ({target})")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Year")
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
