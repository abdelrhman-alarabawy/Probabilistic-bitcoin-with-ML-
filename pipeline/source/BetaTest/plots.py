from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_label_distribution(counts: Dict[str, int], output_path: str) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.title("Label Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_label_distribution_per_setting(
    settings: List[str],
    counts_per_setting: List[Dict[str, int]],
    output_path: str,
) -> None:
    labels = ["long", "short", "skip"]
    x = np.arange(len(settings))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for idx, label in enumerate(labels):
        values = [counts.get(label, 0) for counts in counts_per_setting]
        plt.bar(x + idx * width, values, width=width, label=label)
    plt.xticks(x + width, settings, rotation=45, ha="right")
    plt.title("Label Distribution per Setting")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(confusion: np.ndarray, labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_coverage_curve(
    coverage: List[float],
    precision: List[float],
    output_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(coverage, precision, marker="o")
    plt.title("Precision vs Coverage")
    plt.xlabel("Coverage")
    plt.ylabel("Trade Precision")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_gate_precision_coverage_curve(
    coverage: List[float],
    precision: List[float],
    output_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(coverage, precision, marker="o")
    plt.title("Gate Precision vs Coverage")
    plt.xlabel("Coverage (Trade Rate)")
    plt.ylabel("Precision (Trade)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_probability_histograms(
    proba: np.ndarray,
    labels: List[str],
    output_path: str,
    bins: int = 30,
) -> None:
    plt.figure(figsize=(7, 4))
    for idx, label in enumerate(labels):
        plt.hist(
            proba[:, idx],
            bins=bins,
            alpha=0.5,
            label=label,
            density=True,
        )
    plt.title("Probability Histograms")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
