from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .config import (
    IFOREST_CONTAMINATION,
    IFOREST_MAX_SAMPLES,
    IFOREST_N_ESTIMATORS,
    IFOREST_N_JOBS,
    IFOREST_RANDOM_STATE,
)


@dataclass(frozen=True)
class RobustZModel:
    median: pd.Series
    mad: pd.Series


@dataclass(frozen=True)
class IsolationForestModel:
    scaler: RobustScaler
    model: IsolationForest


def fit_robust_z(X_train: pd.DataFrame) -> RobustZModel:
    median = X_train.median()
    mad = (X_train - median).abs().median()
    return RobustZModel(median=median, mad=mad)


def score_robust_z(model: RobustZModel, X: pd.DataFrame) -> np.ndarray:
    mad = model.mad.replace(0.0, np.nan)
    z = 0.6745 * (X - model.median) / mad
    score = z.abs().mean(axis=1, skipna=True).fillna(0.0)
    return score.to_numpy()


def fit_isolation_forest(X_train: pd.DataFrame) -> IsolationForestModel:
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        max_samples=IFOREST_MAX_SAMPLES,
        contamination=IFOREST_CONTAMINATION,
        random_state=IFOREST_RANDOM_STATE,
        n_jobs=IFOREST_N_JOBS,
    )
    model.fit(X_scaled)
    return IsolationForestModel(scaler=scaler, model=model)


def score_isolation_forest(model: IsolationForestModel, X: pd.DataFrame) -> np.ndarray:
    X_scaled = model.scaler.transform(X)
    scores = model.model.score_samples(X_scaled)
    return (-scores).astype(float)


def compute_thresholds(scores: np.ndarray, percentiles: list[int]) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for pct in percentiles:
        thresholds[pct] = float(np.nanpercentile(scores, pct))
    return thresholds
