from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from .config import SEED


@dataclass
class ModelArtifacts:
    model: Any
    family: str


def _build_lightgbm() -> Optional[Any]:
    try:
        import lightgbm as lgb
    except Exception:
        return None
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
    }
    return lgb.LGBMClassifier(**params)


def _build_xgboost() -> Optional[Any]:
    try:
        import xgboost as xgb
    except Exception:
        return None
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    return xgb.XGBClassifier(**params)


def _build_hist_gbdt() -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        random_state=SEED,
    )


def build_model() -> ModelArtifacts:
    model = _build_lightgbm()
    if model is not None:
        return ModelArtifacts(model=model, family="lightgbm")
    model = _build_xgboost()
    if model is not None:
        return ModelArtifacts(model=model, family="xgboost")
    return ModelArtifacts(model=_build_hist_gbdt(), family="hist_gbdt")


def train_classifier(X: np.ndarray, y: np.ndarray) -> ModelArtifacts:
    artifacts = build_model()
    artifacts.model.fit(X, y)
    return artifacts


def predict_proba_positive(model: Any, X: np.ndarray, positive_label: int = 1) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label} missing in model classes: {classes}")
    idx = classes.index(positive_label)
    return probs[:, idx]
