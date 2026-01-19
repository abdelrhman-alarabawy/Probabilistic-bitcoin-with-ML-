from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class DirectionModelResult:
    model: Optional[CalibratedClassifierCV]
    info: dict


def train_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: List[int],
    calibration_method: str,
    solver: str,
    max_iter: int,
    n_splits: int,
    random_state: int,
) -> DirectionModelResult:
    if len(np.unique(y_train)) < 2:
        return DirectionModelResult(None, {"status": "single_class"})
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model = None
    best_brier = float("inf")
    best_info = {"status": "no_model"}

    for c_val in c_grid:
        base = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver=solver,
            C=c_val,
            random_state=random_state,
        )
        try:
            clf = CalibratedClassifierCV(base, method=calibration_method, cv=tscv)
            clf.fit(X_train, y_train)
        except ValueError:
            continue
        probs = clf.predict_proba(X_train)
        classes = list(clf.classes_)
        if 1 not in classes:
            continue
        idx = classes.index(1)
        p_long = probs[:, idx]
        brier = brier_score_loss(y_train, p_long)
        if brier < best_brier:
            best_brier = brier
            best_model = clf
            best_info = {"status": "ok", "C": c_val, "brier_train": brier}

    return DirectionModelResult(best_model, best_info)


def predict_direction_prob(
    model: Optional[CalibratedClassifierCV],
    X: np.ndarray,
) -> Optional[np.ndarray]:
    if model is None:
        return None
    return model.predict_proba(X)
