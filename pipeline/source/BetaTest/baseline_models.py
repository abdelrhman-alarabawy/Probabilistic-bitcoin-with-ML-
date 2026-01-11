from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import direction_metrics, multiclass_report, trade_metrics


class ModelImportError(ImportError):
    pass


@dataclass
class ModelOutput:
    name: str
    metrics: Dict[str, object]
    y_pred: List
    proba: Optional[np.ndarray] = None


@dataclass
class MulticlassOutput:
    model_name: str
    metrics: Dict[str, object]
    y_pred: List[str]
    proba: np.ndarray
    class_order: List[str]
    model: object


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _get_tree_model_binary(seed: int):
    lgb = _try_import("lightgbm")
    if lgb:
        return "lightgbm", lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    xgb = _try_import("xgboost")
    if xgb:
        return "xgboost", xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
        )
    return "random_forest", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def _get_tree_model_multiclass(seed: int, num_classes: int):
    lgb = _try_import("lightgbm")
    if lgb:
        return "lightgbm", lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    xgb = _try_import("xgboost")
    if xgb:
        return "xgboost", xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
        )
    return "random_forest", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def _align_proba(
    proba: np.ndarray,
    model_classes: List[object],
    class_order: List[str],
    label_to_int: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    if list(model_classes) == class_order:
        return proba
    if label_to_int:
        idx = [list(model_classes).index(label_to_int[label]) for label in class_order]
        return proba[:, idx]
    idx = [list(model_classes).index(label) for label in class_order]
    return proba[:, idx]


def train_trade_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> List[ModelOutput]:
    outputs: List[ModelOutput] = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )
    logreg.fit(X_train_scaled, y_train)
    proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    preds_lr = (proba_lr >= 0.5).astype(int).tolist()
    metrics_lr = trade_metrics(y_test.tolist(), preds_lr, proba_lr.tolist())
    outputs.append(ModelOutput("logistic", metrics_lr, preds_lr, proba_lr))

    tree_name, tree_model = _get_tree_model_binary(seed)
    tree_model.fit(X_train, y_train)
    proba_tree = tree_model.predict_proba(X_test)[:, 1]
    preds_tree = (proba_tree >= 0.5).astype(int).tolist()
    metrics_tree = trade_metrics(y_test.tolist(), preds_tree, proba_tree.tolist())
    outputs.append(ModelOutput(tree_name, metrics_tree, preds_tree, proba_tree))

    return outputs


def train_direction_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    labels: List[str],
) -> List[ModelOutput]:
    outputs: List[ModelOutput] = []

    label_to_int = {label: idx for idx, label in enumerate(labels)}
    y_train_enc = np.array([label_to_int[label] for label in y_train])
    y_test_enc = np.array([label_to_int[label] for label in y_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )
    logreg.fit(X_train_scaled, y_train_enc)
    preds_lr_enc = logreg.predict(X_test_scaled).tolist()
    preds_lr = [labels[pred] for pred in preds_lr_enc]
    metrics_lr = direction_metrics(y_test.tolist(), preds_lr, labels)
    outputs.append(ModelOutput("logistic", metrics_lr, preds_lr))

    tree_name, tree_model = _get_tree_model_binary(seed)
    tree_model.fit(X_train, y_train_enc)
    preds_tree_enc = tree_model.predict(X_test).tolist()
    preds_tree = [labels[pred] for pred in preds_tree_enc]
    metrics_tree = direction_metrics(y_test.tolist(), preds_tree, labels)
    outputs.append(ModelOutput(tree_name, metrics_tree, preds_tree))

    return outputs


def train_multiclass_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_order: List[str],
    seed: int,
) -> MulticlassOutput:
    name, model = _get_tree_model_multiclass(seed, len(class_order))
    label_to_int = {label: idx for idx, label in enumerate(class_order)}
    y_train_enc = np.array([label_to_int[label] for label in y_train])
    y_test_enc = np.array([label_to_int[label] for label in y_test])

    model.fit(X_train, y_train_enc)
    proba = model.predict_proba(X_test)
    proba_aligned = _align_proba(
        proba,
        list(model.classes_),
        class_order,
        label_to_int=label_to_int,
    )
    preds_enc = model.predict(X_test).tolist()
    preds = [class_order[pred] for pred in preds_enc]
    metrics = multiclass_report(y_test.tolist(), preds, class_order)
    return MulticlassOutput(name, metrics, preds, proba_aligned, class_order, model)
