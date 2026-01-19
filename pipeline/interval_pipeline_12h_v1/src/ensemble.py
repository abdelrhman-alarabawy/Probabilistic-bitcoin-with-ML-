from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .models import QuantileModel, available_families, train_quantile_model


@dataclass
class EnsembleBundle:
    models_low: Dict[str, QuantileModel]
    models_high: Dict[str, QuantileModel]
    weights: Dict[str, float]


def _normalize_weights(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n > 0 else {}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


def fit_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    q_low: float,
    q_high: float,
    families: List[str],
) -> Tuple[EnsembleBundle, Dict[str, float]]:
    available = available_families()
    families = [fam for fam in families if available.get(fam, False)]
    if not families:
        families = ["sklearn"]

    models_low: Dict[str, QuantileModel] = {}
    models_high: Dict[str, QuantileModel] = {}
    scores: Dict[str, float] = {}

    for family in families:
        low_model = train_quantile_model(X_train, y_train, q_low, family)
        high_model = train_quantile_model(X_train, y_train, q_high, family)
        pred_low = low_model.predict(X_val)
        pred_high = high_model.predict(X_val)
        lower = np.minimum(pred_low, pred_high)
        upper = np.maximum(pred_low, pred_high)
        hits = (y_val >= lower) & (y_val <= upper)
        coverage = float(np.mean(hits)) if len(hits) else 0.0
        models_low[family] = low_model
        models_high[family] = high_model
        scores[family] = coverage

    weights = _normalize_weights(scores)
    return EnsembleBundle(models_low=models_low, models_high=models_high, weights=weights), scores


def predict_ensemble(
    bundle: EnsembleBundle,
    X: np.ndarray,
    method: str = "weighted",
) -> Tuple[np.ndarray, np.ndarray]:
    if not bundle.models_low:
        return np.array([]), np.array([])

    lows = []
    highs = []
    weights = []
    for family, low_model in bundle.models_low.items():
        high_model = bundle.models_high[family]
        low_pred = low_model.predict(X)
        high_pred = high_model.predict(X)
        lower = np.minimum(low_pred, high_pred)
        upper = np.maximum(low_pred, high_pred)
        lows.append(lower)
        highs.append(upper)
        weights.append(bundle.weights.get(family, 0.0))

    lows_arr = np.vstack(lows)
    highs_arr = np.vstack(highs)

    if method == "conservative":
        return np.min(lows_arr, axis=0), np.max(highs_arr, axis=0)

    weights_arr = np.array(weights, dtype=float)
    if weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr) / len(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    lower = np.average(lows_arr, axis=0, weights=weights_arr)
    upper = np.average(highs_arr, axis=0, weights=weights_arr)
    return lower, upper
