from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustZModel:
    median: pd.Series
    mad: pd.Series


def fit_robust_z(X_train: pd.DataFrame) -> RobustZModel:
    median = X_train.median()
    mad = (X_train - median).abs().median()
    return RobustZModel(median=median, mad=mad)


def score_robust_z(model: RobustZModel, X: pd.DataFrame) -> np.ndarray:
    mad = model.mad.replace(0.0, np.nan)
    z = 0.6745 * (X - model.median) / mad
    score = z.abs().mean(axis=1, skipna=True).fillna(0.0)
    return score.to_numpy()


def compute_thresholds(scores: np.ndarray, percentiles: list[int]) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for pct in percentiles:
        thresholds[pct] = float(np.nanpercentile(scores, pct))
    return thresholds
