from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_curve(trades: pd.DataFrame, path: str) -> None:
    if trades.empty:
        return
    df = trades.sort_values("timestamp")
    equity = np.cumprod(1.0 + df["ret_net"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], equity, color="#4C78A8")
    ax.set_title("Equity curve (net)")
    ax.set_xlabel("time")
    ax.set_ylabel("equity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_signals_timeline(trades: pd.DataFrame, path: str) -> None:
    if trades.empty:
        return
    df = trades.copy()
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    grouped = df.groupby("month").size().reset_index(name="signals")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grouped["month"], grouped["signals"], color="#F58518")
    ax.set_title("Signals per month")
    ax.set_xlabel("month")
    ax.set_ylabel("signals")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_return_distribution(trades: pd.DataFrame, path: str) -> None:
    if trades.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["ret_net"].to_numpy(dtype=float), bins=40, color="#54A24B", alpha=0.8)
    ax.set_title("Return distribution (net)")
    ax.set_xlabel("return")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
