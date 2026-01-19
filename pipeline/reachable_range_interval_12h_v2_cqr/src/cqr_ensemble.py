from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import KNN_NEIGHBORS, RANDOM_SEED, TARGET_COVERAGE
from .models import QuantileModel, select_primary_family, train_quantile_model


class KNNQuantileRegressor:
    def __init__(self, n_neighbors: int = KNN_NEIGHBORS) -> None:
        self.n_neighbors = n_neighbors
        self._nn: Optional[NearestNeighbors] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNQuantileRegressor":
        n_neighbors = min(self.n_neighbors, len(X)) if len(X) else 1
        self._nn = NearestNeighbors(n_neighbors=n_neighbors)
        self._nn.fit(X)
        self._y_train = y
        return self

    def predict(self, X: np.ndarray, quantile: float) -> np.ndarray:
        if self._nn is None or self._y_train is None:
            raise ValueError("KNNQuantileRegressor must be fit before predict.")
        distances, indices = self._nn.kneighbors(X, return_distance=True)
        neighbors = self._y_train[indices]
        return np.quantile(neighbors, quantile, axis=1)


@dataclass
class EnsembleState:
    models_low: Dict[str, QuantileModel]
    models_high: Dict[str, QuantileModel]
    knn_model: Optional[KNNQuantileRegressor]
    weights: Dict[str, float]
    qhat: float
    coverage_cal: float
    width_cal: float


def _try_catboost() -> Optional[object]:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError:
        return None
    return CatBoostRegressor


def _catboost_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float,
    random_state: int = RANDOM_SEED,
) -> QuantileModel:
    cb = _try_catboost()
    if cb is None:
        raise ImportError("CatBoost is not available.")
    model = cb(
        loss_function=f"Quantile:alpha={quantile}",
        depth=6,
        learning_rate=0.05,
        iterations=300,
        random_seed=random_state,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return QuantileModel(model=model, family="catboost")


def available_models() -> Dict[str, bool]:
    available = {
        "lightgbm": True,
        "xgboost": True,
        "catboost": _try_catboost() is not None,
        "knn": True,
    }
    try:
        import lightgbm  # type: ignore  # noqa: F401
    except ImportError:
        available["lightgbm"] = False
    try:
        import xgboost  # type: ignore  # noqa: F401
    except ImportError:
        available["xgboost"] = False
    return available


def build_weight_grid(
    model_names: List[str],
    step: float,
    max_knn: float,
) -> List[Dict[str, float]]:
    if not model_names:
        return []
    names = list(model_names)
    if "knn" not in names:
        names.append("knn")

    grid: List[Dict[str, float]] = []

    def recurse(idx: int, remaining: float, current: Dict[str, float]) -> None:
        if idx == len(names) - 1:
            final_name = names[idx]
            weight = max(0.0, remaining)
            if final_name == "knn" and weight > max_knn + 1e-9:
                return
            current[final_name] = weight
            total = sum(current.values())
            if total <= 0:
                return
            grid.append(current.copy())
            return

        name = names[idx]
        max_weight = remaining
        if name == "knn":
            max_weight = min(max_weight, max_knn)
        steps = int(round(max_weight / step))
        for i in range(steps + 1):
            w = i * step
            current[name] = w
            recurse(idx + 1, remaining - w, current)

    recurse(0, 1.0, {})
    return grid


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def _quantile_with_finite_sample(scores: np.ndarray, target_coverage: float) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    alpha = 1.0 - target_coverage
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(1.0, level)
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))


def fit_cqr_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    q_low: float,
    q_high: float,
    weight_grid: List[Dict[str, float]],
    target_coverage: float = TARGET_COVERAGE,
) -> EnsembleState:
    available = available_models()
    primary_family = select_primary_family(("lightgbm", "xgboost", "sklearn"))

    models_low: Dict[str, QuantileModel] = {}
    models_high: Dict[str, QuantileModel] = {}
    preds_low_cal: Dict[str, np.ndarray] = {}
    preds_high_cal: Dict[str, np.ndarray] = {}
    knn_model: Optional[KNNQuantileRegressor] = None

    if available["lightgbm"]:
        try:
            models_low["lightgbm"] = train_quantile_model(X_train, y_train, q_low, "lightgbm")
            models_high["lightgbm"] = train_quantile_model(X_train, y_train, q_high, "lightgbm")
            preds_low_cal["lightgbm"] = models_low["lightgbm"].predict(X_cal)
            preds_high_cal["lightgbm"] = models_high["lightgbm"].predict(X_cal)
        except Exception:
            models_low.pop("lightgbm", None)
            models_high.pop("lightgbm", None)

    if available["xgboost"]:
        try:
            models_low["xgboost"] = train_quantile_model(X_train, y_train, q_low, "xgboost")
            models_high["xgboost"] = train_quantile_model(X_train, y_train, q_high, "xgboost")
            preds_low_cal["xgboost"] = models_low["xgboost"].predict(X_cal)
            preds_high_cal["xgboost"] = models_high["xgboost"].predict(X_cal)
        except Exception:
            models_low.pop("xgboost", None)
            models_high.pop("xgboost", None)

    if available["catboost"]:
        try:
            models_low["catboost"] = _catboost_quantile(X_train, y_train, q_low)
            models_high["catboost"] = _catboost_quantile(X_train, y_train, q_high)
            preds_low_cal["catboost"] = models_low["catboost"].predict(X_cal)
            preds_high_cal["catboost"] = models_high["catboost"].predict(X_cal)
        except Exception:
            models_low.pop("catboost", None)
            models_high.pop("catboost", None)

    if not models_low:
        models_low[primary_family] = train_quantile_model(X_train, y_train, q_low, primary_family)
        models_high[primary_family] = train_quantile_model(X_train, y_train, q_high, primary_family)
        preds_low_cal[primary_family] = models_low[primary_family].predict(X_cal)
        preds_high_cal[primary_family] = models_high[primary_family].predict(X_cal)

    knn_model = KNNQuantileRegressor().fit(X_train, y_train)
    preds_low_cal["knn"] = knn_model.predict(X_cal, q_low)
    preds_high_cal["knn"] = knn_model.predict(X_cal, q_high)

    best_weights: Dict[str, float] = {}
    best_coverage = -1.0
    best_width = np.inf

    available_names = list(preds_low_cal.keys())
    for weight_set in weight_grid:
        weights = _normalize_weights({name: weight_set.get(name, 0.0) for name in available_names})
        low = np.zeros(len(y_cal))
        high = np.zeros(len(y_cal))
        for name, w in weights.items():
            if w <= 0:
                continue
            low += w * preds_low_cal[name]
            high += w * preds_high_cal[name]
        lower = np.minimum(low, high)
        upper = np.maximum(low, high)
        hits = (y_cal >= lower) & (y_cal <= upper)
        coverage = float(np.mean(hits)) if len(hits) else 0.0
        width = float(np.mean(upper - lower)) if len(hits) else np.inf
        if coverage > best_coverage or (np.isclose(coverage, best_coverage) and width < best_width):
            best_coverage = coverage
            best_width = width
            best_weights = weights

    low_cal = np.zeros(len(y_cal))
    high_cal = np.zeros(len(y_cal))
    for name, w in best_weights.items():
        if w <= 0:
            continue
        low_cal += w * preds_low_cal[name]
        high_cal += w * preds_high_cal[name]
    lower_cal = np.minimum(low_cal, high_cal)
    upper_cal = np.maximum(low_cal, high_cal)
    scores = np.maximum.reduce(
        [lower_cal - y_cal, y_cal - upper_cal, np.zeros_like(y_cal)]
    )
    qhat = _quantile_with_finite_sample(scores, target_coverage)

    return EnsembleState(
        models_low=models_low,
        models_high=models_high,
        knn_model=knn_model,
        weights=best_weights,
        qhat=qhat,
        coverage_cal=best_coverage,
        width_cal=best_width,
    )


def predict_cqr(
    state: EnsembleState,
    X: np.ndarray,
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.zeros(len(X))
    high = np.zeros(len(X))
    for name, w in state.weights.items():
        if w <= 0:
            continue
        if name == "knn":
            if state.knn_model is None:
                continue
            low += w * state.knn_model.predict(X, q_low)
            high += w * state.knn_model.predict(X, q_high)
            continue
        low += w * state.models_low[name].predict(X)
        high += w * state.models_high[name].predict(X)

    lower = np.minimum(low, high) - state.qhat
    upper = np.maximum(low, high) + state.qhat
    return lower, upper
