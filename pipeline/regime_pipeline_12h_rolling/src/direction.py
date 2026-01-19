from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import LinearSVC


@dataclass
class DirectionModelResult:
    model: Optional[CalibratedClassifierCV]
    info: dict


def _train_calibrated(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
    method: str,
) -> DirectionModelResult:
    if len(np.unique(y_train)) < 2:
        return DirectionModelResult(None, {"status": "single_class"})
    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = CalibratedClassifierCV(base_model, method=method, cv=tscv)
    clf.fit(X_train, y_train)
    return DirectionModelResult(clf, {"status": "ok"})


def train_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    calibration_method: str,
    n_splits: int = 3,
) -> DirectionModelResult:
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    result = _train_calibrated(base, X_train, y_train, n_splits, calibration_method)
    if result.model is not None:
        return result
    fallback = LinearSVC(class_weight="balanced")
    return _train_calibrated(fallback, X_train, y_train, n_splits, calibration_method)


def predict_direction_prob(
    model: Optional[CalibratedClassifierCV],
    X: np.ndarray,
) -> Optional[np.ndarray]:
    if model is None:
        return None
    return model.predict_proba(X)
