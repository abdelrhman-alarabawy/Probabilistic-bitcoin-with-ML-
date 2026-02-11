from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_months(
    df: pd.DataFrame,
    timestamp_col: str,
    train_months: int,
    test_months: int,
    step_months: int,
    min_train_rows: int,
) -> List[Fold]:
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months/test_months/step_months must be > 0")

    ts = pd.to_datetime(df[timestamp_col], utc=True)
    min_ts = ts.min()
    max_ts = ts.max()

    folds: list[Fold] = []
    fold_id = 0
    train_start = min_ts

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= test_start) & (ts < test_end)

        if test_mask.sum() == 0:
            break

        if train_mask.sum() >= min_train_rows:
            folds.append(
                Fold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_idx=np.where(train_mask)[0],
                    test_idx=np.where(test_mask)[0],
                )
            )
            fold_id += 1

        train_start = train_start + pd.DateOffset(months=step_months)
        if train_start >= max_ts:
            break

    return folds
