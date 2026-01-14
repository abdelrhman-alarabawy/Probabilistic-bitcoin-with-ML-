from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_months: list[str]
    test_month: str


def build_monthly_folds(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    min_train_months: int = 2,
) -> tuple[list[Fold], str]:
    if timestamp_col is None:
        print("WARNING: No timestamp column found; using index-based month buckets.")
        buckets = (df.index.to_series() // 60).astype(int).astype(str)
        df = df.copy()
        df["_month_id"] = buckets
        month_col = "_month_id"
    else:
        df = df.copy()
        df["_month_id"] = df[timestamp_col].dt.to_period("M").astype(str)
        month_col = "_month_id"

    months = sorted(df[month_col].unique())
    folds: list[Fold] = []
    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]
        train_idx = df.index[df[month_col].isin(train_months)].to_numpy()
        test_idx = df.index[df[month_col] == test_month].to_numpy()
        folds.append(
            Fold(
                fold_id=i - min_train_months + 1,
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_months,
                test_month=test_month,
            )
        )
    return folds, month_col
