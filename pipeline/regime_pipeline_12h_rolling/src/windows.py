from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from config import WindowConfig


@dataclass(frozen=True)
class WindowSlice:
    window_id: int
    config_name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def generate_rolling_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    config: WindowConfig,
    min_train_rows: int,
    min_test_rows: int,
) -> Iterator[WindowSlice]:
    ts = pd.to_datetime(df[timestamp_col])
    start = ts.min()
    end = ts.max()
    step = pd.DateOffset(months=config.step_months)
    train_offset = pd.DateOffset(months=config.train_months)
    test_offset = pd.DateOffset(months=config.test_months)

    window_id = 0
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + train_offset
        test_start = train_end
        test_end = test_start + test_offset
        if test_end > end:
            break

        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= test_start) & (ts < test_end)
        if train_mask.sum() >= min_train_rows and test_mask.sum() >= min_test_rows:
            yield WindowSlice(
                window_id=window_id,
                config_name=config.name,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            )
            window_id += 1

        cursor = cursor + step
