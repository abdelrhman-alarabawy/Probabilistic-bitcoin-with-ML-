from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


try:
    from lightgbm import LGBMClassifier  # type: ignore

    LGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier  # type: ignore

    CAT_AVAILABLE = True
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CAT_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGB_AVAILABLE = False


@dataclass
class TrainEvalResult:
    model: object
    report: dict
    confusion: np.ndarray
    labels: list[str]


def pick_model(num_classes: int, random_state: int) -> object:
    if LGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            class_weight="balanced",
            verbosity=-1,
        )
    if CAT_AVAILABLE:
        return CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_state=random_state,
            verbose=False,
        )
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=random_state,
        )
    try:
        return HistGradientBoostingClassifier(random_state=random_state)
    except Exception:
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        )


def compute_sample_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights_map = {cls: w for cls, w in zip(classes, class_weights)}
    return np.array([weights_map[label] for label in y_train])


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    random_state: int,
) -> TrainEvalResult:
    model = pick_model(len(labels), random_state)
    sample_weight = compute_sample_weights(y_train)

    fit_kwargs = {}
    if model.__class__.__name__.startswith("LGBM"):
        fit_kwargs["sample_weight"] = sample_weight
    elif model.__class__.__name__.startswith("CatBoost"):
        fit_kwargs["sample_weight"] = sample_weight
    elif model.__class__.__name__.startswith("XGB"):
        fit_kwargs["sample_weight"] = sample_weight
    elif model.__class__.__name__.startswith("HistGradient"):
        fit_kwargs["sample_weight"] = sample_weight
    elif model.__class__.__name__.startswith("RandomForest"):
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = acc
    confusion = confusion_matrix(y_test, preds, labels=list(range(len(labels))))
    return TrainEvalResult(model=model, report=report, confusion=confusion, labels=labels)
