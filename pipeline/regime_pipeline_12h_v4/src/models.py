from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import brier_score_loss

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None
    HGB_AVAILABLE = False


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray, y: Any = None) -> "QuantileClipper":
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("QuantileClipper must be fit before transform.")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def build_preprocessor(
    clip_quantiles: Tuple[float, float],
    use_pca: bool,
    pca_components: float,
) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(lower=clip_quantiles[0], upper=clip_quantiles[1])),
        ("scaler", RobustScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    return Pipeline(steps)


@dataclass
class ModelResult:
    model: Optional[Any]
    info: dict


def _train_calibrated(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
    method: str,
) -> ModelResult:
    if len(np.unique(y_train)) < 2:
        return ModelResult(None, {"status": "single_class"})
    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = CalibratedClassifierCV(base_model, method=method, cv=tscv)
    clf.fit(X_train, y_train)
    return ModelResult(clf, {"status": "ok"})


def train_gate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    method: str = "isotonic",
) -> ModelResult:
    best_model = None
    best_brier = float("inf")
    best_info = {"status": "no_model"}
    for c_val in [1, 10, 50]:
        base = LogisticRegression(max_iter=1000, class_weight="balanced", C=c_val)
        result = _train_calibrated(base, X_train, y_train, n_splits, method)
        if result.model is None:
            continue
        probs = result.model.predict_proba(X_train)
        classes = list(result.model.classes_)
        if 1 not in classes:
            continue
        idx = classes.index(1)
        p_trade = probs[:, idx]
        brier = brier_score_loss(y_train, p_trade)
        if brier < best_brier:
            best_brier = brier
            best_model = result.model
            best_info = {"status": "ok", "C": c_val, "brier_train": brier}
    if best_model is not None:
        return ModelResult(best_model, best_info)
    if HGB_AVAILABLE:
        hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05)
        return _train_calibrated(hgb, X_train, y_train, n_splits, method)
    return ModelResult(None, best_info)


def train_direction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    method: str = "sigmoid",
) -> ModelResult:
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    result = _train_calibrated(base, X_train, y_train, n_splits, method)
    if result.model is not None:
        return result
    fallback = LinearSVC(class_weight="balanced")
    return _train_calibrated(fallback, X_train, y_train, n_splits, method)


def predict_proba_safe(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    if model is None:
        return None
    return model.predict_proba(X)
