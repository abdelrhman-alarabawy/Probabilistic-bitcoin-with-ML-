from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plot_metric_vs_k(ax, df: pd.DataFrame, metric_col: str, y_label: str) -> None:
    plotted = False
    for cov_type, part in df.groupby("covariance_type"):
        series = part.groupby("n_components")[metric_col].mean().sort_index()
        if series.empty:
            continue
        ax.plot(series.index, series.values, marker="o", linewidth=1.8, label=str(cov_type))
        plotted = True
    ax.set_xlabel("K (n_components)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def save_group_plots(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    if summary_df.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.copy()

    for col in ["n_components", "bic_train_mean", "aic_train_mean", "avg_entropy_test_mean", "silhouette_test_mean"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["n_components"])
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _plot_metric_vs_k(axes[0], plot_df, "bic_train_mean", "BIC (train mean)")
    axes[0].set_title("BIC vs K")
    _plot_metric_vs_k(axes[1], plot_df, "aic_train_mean", "AIC (train mean)")
    axes[1].set_title("AIC vs K")
    fig.tight_layout()
    fig.savefig(plots_dir / "bic_aic_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_metric_vs_k(ax, plot_df, "avg_entropy_test_mean", "Avg entropy (test mean)")
    ax.set_title("Responsibility Entropy vs K")
    fig.tight_layout()
    fig.savefig(plots_dir / "entropy_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_metric_vs_k(ax, plot_df, "silhouette_test_mean", "Silhouette (test mean)")
    ax.set_title("Silhouette vs K")
    fig.tight_layout()
    fig.savefig(plots_dir / "silhouette_vs_k.png", dpi=150)
    plt.close(fig)

