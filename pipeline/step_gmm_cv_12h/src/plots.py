from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_close_by_regime(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    close_col: str,
    regime_col: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = df[timestamp_col] if timestamp_col else np.arange(len(df))
    scatter = ax.scatter(x, df[close_col], c=df[regime_col], cmap="tab10", s=6)
    ax.set_title("Close Price by Regime")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.colorbar(scatter, ax=ax, label="Regime")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_regime_counts(
    counts: pd.Series,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Regime Counts")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_regime_stats(
    stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    stats_df.plot(x="regime_id", y="mean_return", kind="bar", ax=axes[0], legend=False)
    axes[0].set_title("Mean Return by Regime")
    stats_df.plot(x="regime_id", y="volatility", kind="bar", ax=axes[1], legend=False)
    axes[1].set_title("Volatility by Regime")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_label_distribution(
    label_dist: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    label_dist.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Label Distribution by Regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Share")
    ax.legend(title="Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, confusion[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
