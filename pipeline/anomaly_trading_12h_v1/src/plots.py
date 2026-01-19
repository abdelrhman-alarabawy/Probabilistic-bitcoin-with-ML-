from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_score_hist(scores: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(scores["score_robustz"].dropna(), bins=50, color="#4C78A8", alpha=0.8)
    axes[0].set_title("RobustZ score")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("count")

    axes[1].hist(scores["score_iforest"].dropna(), bins=50, color="#F58518", alpha=0.8)
    axes[1].set_title("IsolationForest score")
    axes[1].set_xlabel("score")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_signals_timeline(trades: pd.DataFrame, path: str, percentile: int) -> None:
    if trades.empty:
        return
    df = trades[trades["threshold_pct"] == percentile].copy()
    if df.empty:
        return

    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    grouped = df.groupby(["month", "strategy", "model"]).size().reset_index(name="signals")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, strategy in zip(axes, ["MR", "MOM"]):
        sub = grouped[grouped["strategy"] == strategy]
        for model in ["robustz", "iforest"]:
            series = sub[sub["model"] == model]
            if series.empty:
                continue
            ax.plot(series["month"], series["signals"], label=model)
        ax.set_title(f"Signals per month ({strategy}, p{percentile})")
        ax.set_xlabel("month")
        ax.set_ylabel("signals")
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity_curve(trades: pd.DataFrame, path: str, percentile: int) -> None:
    if trades.empty:
        return
    df = trades[trades["threshold_pct"] == percentile].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, strategy in zip(axes, ["MR", "MOM"]):
        sub = df[df["strategy"] == strategy]
        for model in ["robustz", "iforest"]:
            series = sub[sub["model"] == model]
            if series.empty:
                continue
            series = series.sort_values("timestamp")
            equity = np.cumprod(1.0 + series["return"].to_numpy(dtype=float))
            ax.plot(series["timestamp"], equity, label=model)
        ax.set_title(f"Equity curve ({strategy}, p{percentile})")
        ax.set_xlabel("time")
        ax.set_ylabel("equity")
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
