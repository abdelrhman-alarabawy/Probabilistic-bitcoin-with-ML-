from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_rules_library(rules: List[Dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def plot_event_timeline(events_df: pd.DataFrame, path: Path) -> None:
    if events_df.empty:
        return
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
    long_mask = df["direction_candidate"] == "long"
    short_mask = df["direction_candidate"] == "short"

    plt.figure(figsize=(10, 3))
    plt.scatter(df.loc[long_mask, "timestamp"], np.ones(long_mask.sum()), color="#2ca02c", s=10, label="long")
    plt.scatter(df.loc[short_mask, "timestamp"], np.zeros(short_mask.sum()), color="#d62728", s=10, label="short")
    plt.yticks([0, 1], ["short", "long"])
    plt.title("Event Timeline")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_equity_curve(trades_df: pd.DataFrame, path: Path) -> None:
    if trades_df.empty:
        return
    equity = np.cumprod(1.0 + trades_df["return_net"].to_numpy(dtype=float))
    plt.figure(figsize=(10, 4))
    plt.plot(trades_df["timestamp"], equity, color="#4c78a8")
    plt.title("Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_precision_vs_coverage(rules_df: pd.DataFrame, path: Path) -> None:
    if rules_df.empty:
        return
    trades_col = "trades_per_month"
    if trades_col not in rules_df.columns and "trades_per_month_mean" in rules_df.columns:
        trades_col = "trades_per_month_mean"
    precision_col = "precision"
    if precision_col not in rules_df.columns and "precision_mean" in rules_df.columns:
        precision_col = "precision_mean"
    plt.figure(figsize=(6, 4))
    plt.scatter(
        rules_df[trades_col],
        rules_df[precision_col],
        color="#ff7f0e",
        s=30,
        alpha=0.8,
    )
    plt.title("Precision vs Coverage")
    plt.xlabel("Trades per Month")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
