from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .config import HOLDOUT_TRAIN_FRAC, TEST_FRAC, TRAIN_FRAC, VAL_FRAC, WALK_FORWARD_SPLITS


@dataclass(frozen=True)
class SplitResult:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


@dataclass(frozen=True)
class HoldoutSplit:
    train_idx: pd.Index
    test_idx: pd.Index


def holdout_split(df: pd.DataFrame) -> HoldoutSplit:
    split_idx = int(len(df) * HOLDOUT_TRAIN_FRAC)
    train_idx = df.index[:split_idx]
    test_idx = df.index[split_idx:]
    return HoldoutSplit(train_idx=train_idx, test_idx=test_idx)


def train_val_test_split(df: pd.DataFrame) -> SplitResult:
    if abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.")

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]
    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def walk_forward_splits(df: pd.DataFrame) -> List[dict]:
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for walk-forward splits.") from exc

    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    splits = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        split = {
            "fold": fold,
            "train_start": int(train_idx[0]),
            "train_end": int(train_idx[-1]),
            "test_start": int(test_idx[0]),
            "test_end": int(test_idx[-1]),
        }
        splits.append(split)
    return splits
