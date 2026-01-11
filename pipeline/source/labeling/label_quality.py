from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def label_counts(labels: Sequence[str]) -> Counter:
    return Counter(labels)


def print_label_counts(title: str, labels: Sequence[str], allowed_labels: Iterable[str]) -> Counter:
    counts = label_counts(labels)
    total = len(labels)
    print(f"\n=== {title} ===")
    for candle_type in allowed_labels:
        count = counts.get(candle_type, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{candle_type}:  {count} ({pct:.2f}%)")
    return counts


def counts_from_series(series, allowed_labels: Iterable[str]) -> dict:
    counts = Counter(series)
    return {label: counts.get(label, 0) for label in allowed_labels}