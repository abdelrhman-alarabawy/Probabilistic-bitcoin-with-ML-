from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .config_v4 import (
    ENABLE_REGIME_MODELS,
    HGB_PARAMS,
    LGBM_PARAMS,
    MIN_CALIBRATION_SAMPLES,
    MIN_REGIME_TRAIN,
    MIN_REGIME_VAL,
    SEED,
    XGB_PARAMS,
)


@dataclass
class ModelBundle:
    model: Any
    calibrated: Any
    family: str
    balance_method: str
    calibration_method: str = "none"


@dataclass
class DirectionModels:
    global_model: ModelBundle
    regime_models: Dict[str, ModelBundle]


def _build_lightgbm(class_weight: Optional[dict]) -> Optional[Any]:
    try:
        import lightgbm as lgb
    except Exception:
        return None
    params = dict(LGBM_PARAMS)
    if class_weight is not None:
        params["class_weight"] = class_weight
    return lgb.LGBMClassifier(**params)


def _build_xgboost(scale_pos_weight: Optional[float]) -> Optional[Any]:
    try:
        import xgboost as xgb
    except Exception:
        return None
    params = dict(XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**params)


def _build_hist_gbdt(class_weight: Optional[dict]) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = dict(HGB_PARAMS)
    return HistGradientBoostingClassifier(**params, class_weight=class_weight)


def _class_weight(y: np.ndarray) -> dict:
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    weight_pos = negatives / positives
    return {0: 1.0, 1: weight_pos}


def train_gate_model(X: np.ndarray, y: np.ndarray) -> ModelBundle:
    model = _build_lightgbm(class_weight=None)
    if model is not None:
        model.fit(X, y)
        return ModelBundle(model=model, calibrated=None, family="lightgbm", balance_method="none")
    model = _build_xgboost(scale_pos_weight=None)
    if model is not None:
        model.fit(X, y)
        return ModelBundle(model=model, calibrated=None, family="xgboost", balance_method="none")
    model = _build_hist_gbdt(class_weight=None)
    model.fit(X, y)
    return ModelBundle(model=model, calibrated=None, family="hist_gbdt", balance_method="none")


def train_direction_model(X: np.ndarray, y: np.ndarray, balance: bool = True) -> ModelBundle:
    class_weight = _class_weight(y) if balance else None
    scale_pos_weight = class_weight[1] if class_weight else None

    model = _build_lightgbm(class_weight=class_weight)
    if model is not None:
        model.fit(X, y)
        return ModelBundle(
            model=model,
            calibrated=None,
            family="lightgbm",
            balance_method="class_weight" if balance else "none",
        )

    model = _build_xgboost(scale_pos_weight=scale_pos_weight)
    if model is not None:
        model.fit(X, y)
        return ModelBundle(
            model=model,
            calibrated=None,
            family="xgboost",
            balance_method="scale_pos_weight" if balance else "none",
        )

    model = _build_hist_gbdt(class_weight=class_weight)
    model.fit(X, y)
    return ModelBundle(
        model=model,
        calibrated=None,
        family="hist_gbdt",
        balance_method="class_weight" if balance else "none",
    )


def _calibration_method(y: np.ndarray) -> str:
    if len(y) >= MIN_CALIBRATION_SAMPLES:
        return "isotonic"
    return "sigmoid"


def calibrate_model_on_val(model_bundle: ModelBundle, X_val: np.ndarray, y_val: np.ndarray) -> ModelBundle:
    if len(y_val) == 0 or len(np.unique(y_val)) < 2:
        model_bundle.calibrated = model_bundle.model
        model_bundle.calibration_method = "none"
        return model_bundle

    method = _calibration_method(y_val)
    calibrator = CalibratedClassifierCV(model_bundle.model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)
    model_bundle.calibrated = calibrator
    model_bundle.calibration_method = method
    return model_bundle


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        idx = classes.index(1)
    else:
        idx = 1
    return probs[:, idx]


def train_direction_models_by_regime(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: pd.Series,
    val_mask: pd.Series,
    regime_series: pd.Series,
) -> DirectionModels:
    global_model = train_direction_model(X.loc[train_mask].to_numpy(), y.loc[train_mask].to_numpy())
    global_model = calibrate_model_on_val(
        global_model,
        X.loc[val_mask].to_numpy(),
        y.loc[val_mask].to_numpy(),
    )

    regime_models: Dict[str, ModelBundle] = {}
    if not ENABLE_REGIME_MODELS:
        return DirectionModels(global_model=global_model, regime_models=regime_models)

    for regime, group_idx in regime_series.groupby(regime_series).groups.items():
        train_idx = train_mask & regime_series.index.isin(group_idx)
        val_idx = val_mask & regime_series.index.isin(group_idx)
        if train_idx.sum() < MIN_REGIME_TRAIN:
            continue
        model = train_direction_model(X.loc[train_idx].to_numpy(), y.loc[train_idx].to_numpy())
        if val_idx.sum() >= MIN_REGIME_VAL:
            model = calibrate_model_on_val(
                model,
                X.loc[val_idx].to_numpy(),
                y.loc[val_idx].to_numpy(),
            )
        else:
            model.calibrated = global_model.calibrated
        regime_models[str(regime)] = model

    return DirectionModels(global_model=global_model, regime_models=regime_models)


def predict_direction_probs(
    direction_models: DirectionModels,
    X: pd.DataFrame,
    regime_series: Optional[pd.Series],
) -> pd.Series:
    if regime_series is None or not direction_models.regime_models:
        probs = predict_proba_positive(direction_models.global_model.calibrated, X.to_numpy())
        return pd.Series(probs, index=X.index)

    p_long = pd.Series(index=X.index, dtype=float)
    for regime, idx in regime_series.groupby(regime_series).groups.items():
        model = direction_models.regime_models.get(str(regime), direction_models.global_model)
        probs = predict_proba_positive(model.calibrated, X.loc[idx].to_numpy())
        p_long.loc[idx] = probs
    return p_long
