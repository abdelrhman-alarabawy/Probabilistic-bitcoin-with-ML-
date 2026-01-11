from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def holdout_indices(n_samples: int, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1.")
    split_idx = int(n_samples * (1 - test_frac))
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, n_samples)
    return train_idx, test_idx


def rolling_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(train_idx, test_idx) for train_idx, test_idx in tscv.split(np.arange(n_samples))]


def _label_distribution(series: pd.Series, labels: List[str]) -> Dict[str, int]:
    counts = series.value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in labels}


def build_split_table(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    labels: List[str],
    holdout: Tuple[np.ndarray, np.ndarray],
    rolling: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_row(name: str, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        row = {
            "split": name,
            "train_start": train_df[time_col].min(),
            "train_end": train_df[time_col].max(),
            "test_start": test_df[time_col].min(),
            "test_end": test_df[time_col].max(),
        }
        train_counts = _label_distribution(train_df[label_col], labels)
        test_counts = _label_distribution(test_df[label_col], labels)
        for label in labels:
            row[f"train_{label}"] = train_counts[label]
            row[f"test_{label}"] = test_counts[label]
        rows.append(row)

    add_row("holdout", holdout[0], holdout[1])
    for idx, (train_idx, test_idx) in enumerate(rolling, start=1):
        add_row(f"cv_{idx}", train_idx, test_idx)

    return pd.DataFrame(rows)