from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def detect_timestamp_column(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "ts_utc" in df.columns:
        return "ts_utc"
    raise ValueError("No timestamp column found for sorting.")


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = detect_timestamp_column(df)
    return df.sort_values(ts_col).reset_index(drop=True)


def time_holdout_split(n_samples: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_idx = int(n_samples * (1 - test_size))
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, n_samples)
    return train_idx, test_idx


def time_series_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(train_idx, test_idx) for train_idx, test_idx in tscv.split(np.arange(n_samples))]


def train_val_split_indices(
    train_idx: np.ndarray,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    split_idx = int(len(train_idx) * (1 - val_size))
    return train_idx[:split_idx], train_idx[split_idx:]