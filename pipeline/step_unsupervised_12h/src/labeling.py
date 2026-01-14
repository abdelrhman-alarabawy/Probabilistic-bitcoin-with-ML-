from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelingParams:
    base_horizon_minutes: int = 60
    tp_points: float = 2000
    sl_points: float = 1000

    @property
    def horizon_minutes(self) -> int:
        return self.base_horizon_minutes * 12


def label_candles(
    df: pd.DataFrame,
    open_col: str,
    high_col: str,
    low_col: str,
    params: LabelingParams,
) -> tuple[list[str], list[bool]]:
    labels: list[str] = []
    ambiguous_flags: list[bool] = []

    for _, row in df.iterrows():
        o = row[open_col]
        h = row[high_col]
        l = row[low_col]
        if pd.isna(o) or pd.isna(h) or pd.isna(l):
            labels.append("skip")
            ambiguous_flags.append(True)
            continue

        multiplier = o / 100000
        long_tp = o + params.tp_points * multiplier
        long_sl = o - params.sl_points * multiplier
        short_tp = o - params.tp_points * multiplier
        short_sl = o + params.sl_points * multiplier

        if h >= long_tp and l >= long_sl:
            labels.append("long")
            ambiguous_flags.append(False)
        elif l <= short_tp and h <= short_sl:
            labels.append("short")
            ambiguous_flags.append(False)
        elif h >= long_tp and l <= short_tp:
            labels.append("skip")
            ambiguous_flags.append(True)
        elif h >= long_tp and l < long_sl:
            labels.append("skip")
            ambiguous_flags.append(True)
        elif l <= short_tp and h > short_sl:
            labels.append("skip")
            ambiguous_flags.append(True)
        else:
            labels.append("skip")
            ambiguous_flags.append(False)

    return labels, ambiguous_flags


def label_distribution(labels: list[str]) -> dict[str, float]:
    counts = pd.Series(labels).value_counts()
    total = len(labels)
    dist = {label: float(count) / total for label, count in counts.items()}
    return dist


def label_transition_matrix(labels: list[str]) -> pd.DataFrame:
    series = pd.Series(labels, name="label")
    return pd.crosstab(series.shift(1), series, dropna=False)
