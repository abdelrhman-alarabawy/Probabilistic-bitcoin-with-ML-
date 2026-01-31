from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def map_states_to_labels(states: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for state in np.unique(states):
        mask = states == state
        if mask.sum() == 0:
            continue
        values, counts = np.unique(labels[mask], return_counts=True)
        mapping[int(state)] = int(values[np.argmax(counts)])
    return mapping


def apply_state_label_map(states: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(s), -1) for s in states])


def label_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    return {
        "label_precision": float(precision),
        "label_recall": float(recall),
        "label_f1": float(f1),
    }


def label_report(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[str, np.ndarray]:
    report = classification_report(true_labels, pred_labels, zero_division=0)
    matrix = confusion_matrix(true_labels, pred_labels)
    return report, matrix
