from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(conf_matrix: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_matrix.values, cmap="Blues")
    ax.set_xticks(range(len(conf_matrix.columns)))
    ax.set_xticklabels(conf_matrix.columns)
    ax.set_yticks(range(len(conf_matrix.index)))
    ax.set_yticklabels(conf_matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix.iloc[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_coverage(sweep: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sweep["coverage_total"], sweep["min_precision"], alpha=0.6, s=20)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Min Precision (long/short)")
    ax.set_title("Threshold Sweep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_equity_curve(equity_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not equity_curve.empty:
        ax.plot(equity_curve["timestamp"], equity_curve["equity"], color="#1f77b4")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
