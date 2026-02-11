from __future__ import annotations

from typing import Dict, List

import pandas as pd


LABELS = ["long", "short", "skip"]


def majority_vote(row: pd.Series, group_names: List[str]) -> str:
    counts = {label: 0 for label in LABELS}
    for g in group_names:
        label = str(row.get(f"{g}_trade_label", "skip")).strip().lower()
        if label not in LABELS:
            label = "skip"
        counts[label] += 1
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    for label in LABELS:
        if label in candidates:
            return label
    return "skip"


def weighted_vote(row: pd.Series, group_names: List[str], weights: Dict[str, float]) -> str:
    scores = {label: 0.0 for label in LABELS}
    for g in group_names:
        label = str(row.get(f"{g}_trade_label", "skip")).strip().lower()
        if label not in LABELS:
            label = "skip"
        scores[label] += float(weights.get(g, 0.0))
    max_score = max(scores.values())
    candidates = [k for k, v in scores.items() if v == max_score]
    for label in LABELS:
        if label in candidates:
            return label
    return "skip"


def reference_label(row: pd.Series, ref_group: str) -> str:
    label = str(row.get(f"{ref_group}_trade_label", "skip")).strip().lower()
    return label if label in LABELS else "skip"

