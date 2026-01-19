from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class GateResult:
    model: Optional[CalibratedClassifierCV]
    status: str


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def train_gate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    solver: str,
    max_iter: int,
    n_jobs: int,
    calibration_method: str,
    calibration_splits: int,
    random_state: int,
) -> GateResult:
    if len(np.unique(y_train)) < 2:
        return GateResult(None, "single_class")
    tscv = TimeSeriesSplit(n_splits=calibration_splits)
    base = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver=solver,
        C=c_value,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    try:
        clf = CalibratedClassifierCV(base, method=calibration_method, cv=tscv)
        clf.fit(X_train, y_train)
    except ValueError:
        return GateResult(None, "fit_failed")
    return GateResult(clf, "ok")


def predict_trade_prob(model: Optional[CalibratedClassifierCV], X: np.ndarray) -> Optional[np.ndarray]:
    if model is None:
        return None
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return None
    idx = classes.index(1)
    return probs[:, idx]


def select_topk(p_trade: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    n = len(p_trade)
    k_eff = min(k, n)
    selected = np.zeros(n, dtype=bool)
    if k_eff <= 0:
        return selected, k_eff
    p_trade_safe = np.where(np.isfinite(p_trade), p_trade, -np.inf)
    order = np.argsort(p_trade_safe)[::-1]
    selected[order[:k_eff]] = True
    return selected, k_eff
