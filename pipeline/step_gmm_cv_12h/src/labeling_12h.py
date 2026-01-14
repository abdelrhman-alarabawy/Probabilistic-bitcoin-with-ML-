from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LabelingParams:
    base_horizon_minutes: int = 60
    tp_points: float = 2000.0
    sl_points: float = 1000.0

    @property
    def horizon_minutes(self) -> int:
        return self.base_horizon_minutes * 12


def label_candles(
    df: pd.DataFrame,
    open_col: str,
    high_col: str,
    low_col: str,
    params: LabelingParams,
) -> list[str]:
    labels: list[str] = []
    for _, row in df.iterrows():
        o = row[open_col]
        h = row[high_col]
        l = row[low_col]
        if pd.isna(o) or pd.isna(h) or pd.isna(l):
            labels.append("skip")
            continue

        multiplier = o / 100000
        long_tp = o + params.tp_points * multiplier
        long_sl = o - params.sl_points * multiplier
        short_tp = o - params.tp_points * multiplier
        short_sl = o + params.sl_points * multiplier

        if h >= long_tp and l >= long_sl:
            labels.append("long")
        elif l <= short_tp and h <= short_sl:
            labels.append("short")
        elif h >= long_tp and l <= short_tp:
            labels.append("skip")
        elif h >= long_tp and l < long_sl:
            labels.append("skip")
        elif l <= short_tp and h > short_sl:
            labels.append("skip")
        else:
            labels.append("skip")
    return labels


def label_distribution(labels: list[str]) -> dict[str, float]:
    series = pd.Series(labels)
    counts = series.value_counts()
    total = len(series)
    return {label: float(count) / total for label, count in counts.items()}
