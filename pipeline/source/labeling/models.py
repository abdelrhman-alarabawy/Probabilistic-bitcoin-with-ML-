from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class ModelImportError(ImportError):
    pass


@dataclass
class ModelSpec:
    name: str
    model: object
    class_order: list[str]


def _require_package(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ModelImportError(
            f"Missing dependency '{module_name}'. Install it to use this model."
        ) from exc


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    seed: int,
) -> object:
    lgb = _require_package("lightgbm")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    seed: int,
) -> object:
    xgb = _require_package("xgboost")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    seed: int,
) -> object:
    cb = _require_package("catboost")
    model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def build_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_order: list[str],
    seed: int,
) -> ModelSpec:
    num_classes = len(class_order)
    if model_name == "lightgbm":
        model = train_lightgbm(X_train, y_train, num_classes, seed)
    elif model_name == "xgboost":
        model = train_xgboost(X_train, y_train, num_classes, seed)
    elif model_name == "catboost":
        model = train_catboost(X_train, y_train, num_classes, seed)
    else:
        raise ValueError(f"Unknown model '{model_name}'.")
    return ModelSpec(name=model_name, model=model, class_order=class_order)