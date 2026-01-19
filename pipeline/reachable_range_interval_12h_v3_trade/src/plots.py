from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


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
