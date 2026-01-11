from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def label_distribution(labels: pd.Series, allowed: List[str]) -> Dict[str, int]:
    counts = labels.value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in allowed}


def skip_run_lengths(labels: pd.Series) -> List[int]:
    lengths: List[int] = []
    current = 0
    for label in labels:
        if label == "skip":
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def run_length_stats(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(lengths)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def rolling_distribution_drift(
    labels: pd.Series,
    allowed: List[str],
    window: int = 500,
) -> Dict[str, float]:
    if len(labels) < window:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}

    global_dist = np.array([labels.value_counts().get(lbl, 0) for lbl in allowed], dtype=float)
    global_dist = global_dist / global_dist.sum() if global_dist.sum() else global_dist

    distances: List[float] = []
    for start in range(0, len(labels) - window + 1, window):
        window_labels = labels.iloc[start : start + window]
        dist = np.array([window_labels.value_counts().get(lbl, 0) for lbl in allowed], dtype=float)
        dist = dist / dist.sum() if dist.sum() else dist
        distances.append(float(np.abs(dist - global_dist).sum()))

    if not distances:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}

    arr = np.array(distances)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def ambiguous_breakdown(
    labels: pd.Series,
    ambiguous: pd.Series,
    allowed: List[str],
) -> Dict[str, Dict[str, int]]:
    result = {"ambiguous_true": {}, "ambiguous_false": {}}
    for flag_value, key in [(True, "ambiguous_true"), (False, "ambiguous_false")]:
        subset = labels[ambiguous == flag_value]
        result[key] = label_distribution(subset, allowed)
    return result


def build_label_stats_table(
    labels: pd.Series,
    allowed: List[str],
    ambiguous: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    total = len(labels)
    counts = label_distribution(labels, allowed)
    pct = {label: (counts[label] / total if total else 0.0) for label in allowed}

    skip_lengths = skip_run_lengths(labels)
    skip_stats = run_length_stats(skip_lengths)
    drift_stats = rolling_distribution_drift(labels, allowed)

    stats = {
        "total_rows": total,
        "skip_run_mean": skip_stats["mean"],
        "skip_run_median": skip_stats["median"],
        "skip_run_p95": skip_stats["p95"],
        "drift_l1_mean": drift_stats["mean"],
        "drift_l1_median": drift_stats["median"],
        "drift_l1_p95": drift_stats["p95"],
    }
    for label in allowed:
        stats[f"count_{label}"] = counts[label]
        stats[f"pct_{label}"] = pct[label]

    if ambiguous is not None:
        ambiguous_true = int(ambiguous.astype(bool).sum())
        stats["ambiguous_true"] = ambiguous_true
        stats["ambiguous_true_pct"] = ambiguous_true / total if total else 0.0
        stats["ambiguous_breakdown"] = ambiguous_breakdown(labels, ambiguous.astype(bool), allowed)

    table = pd.DataFrame([stats])
    return table, stats