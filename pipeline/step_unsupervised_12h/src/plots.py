from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except Exception:  # pragma: no cover
    sns = None
    SEABORN_AVAILABLE = False

import matplotlib.pyplot as plt


def plot_price_regimes(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    close_col: str,
    regime_col: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    if timestamp_col:
        x = df[timestamp_col]
    else:
        x = np.arange(len(df))
    regimes = df[regime_col].astype(str)
    scatter = ax.scatter(x, df[close_col], c=pd.factorize(regimes)[0], cmap="tab10", s=6)
    ax.set_title("Close Price by Final Regime")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.colorbar(scatter, ax=ax, label="Regime")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_state_occupancy(
    hmm_counts: pd.Series,
    final_counts: pd.Series,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hmm_counts.sort_index().plot(kind="bar", ax=axes[0], title="HMM State Occupancy")
    final_counts.sort_index().plot(kind="bar", ax=axes[1], title="Final Regime Occupancy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if SEABORN_AVAILABLE:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    else:
        im = ax.imshow(matrix.values, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_boxplots(
    df: pd.DataFrame,
    regime_col: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["returns", "volatility", "range_pct"]
    for idx, metric in enumerate(metrics):
        if SEABORN_AVAILABLE:
            sns.boxplot(x=regime_col, y=metric, data=df, ax=axes[idx])
        else:
            df.boxplot(column=metric, by=regime_col, ax=axes[idx])
        axes[idx].set_title(metric)
    fig.suptitle("Regime Diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
