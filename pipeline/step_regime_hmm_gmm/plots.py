from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_state_probabilities(timestamps: pd.Series, posteriors: np.ndarray, path: str) -> None:
    plt.figure(figsize=(12, 4))
    for k in range(posteriors.shape[1]):
        plt.plot(timestamps, posteriors[:, k], label=f"state_{k}", alpha=0.8)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_viterbi_states(timestamps: pd.Series, states: np.ndarray, path: str) -> None:
    plt.figure(figsize=(12, 2.5))
    plt.step(timestamps, states, where="post")
    plt.yticks(sorted(set(states)))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_duration_hist(run_lengths: Dict[int, List[int]], path: str) -> None:
    plt.figure(figsize=(8, 4))
    for state, lengths in run_lengths.items():
        if not lengths:
            continue
        plt.hist(lengths, bins=30, alpha=0.6, label=f"state_{state}")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
