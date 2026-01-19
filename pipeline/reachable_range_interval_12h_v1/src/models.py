from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import RANDOM_SEED


@dataclass
class QuantileModel:
    model: object
    family: str
    quantile: Optional[float] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.family == "quantile_forest" and self.quantile is not None:
            preds = self.model.predict(X, quantiles=[self.quantile])
            return np.asarray(preds).reshape(-1)
        return self.model.predict(X)


def _try_lightgbm() -> Optional[object]:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        return None
    return lgb


def _try_xgboost() -> Optional[object]:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        return None
    return xgb


def _try_quantile_forest() -> Optional[object]:
    try:
        from quantile_forest import RandomForestQuantileRegressor  # type: ignore
    except ImportError:
        return None
    return RandomForestQuantileRegressor


def available_families() -> Dict[str, bool]:
    return {
        "lightgbm": _try_lightgbm() is not None,
        "xgboost": _try_xgboost() is not None,
        "quantile_forest": _try_quantile_forest() is not None,
        "sklearn": True,
    }


def train_quantile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float,
    family: str,
    random_state: int = RANDOM_SEED,
) -> QuantileModel:
    if family == "lightgbm":
        lgb = _try_lightgbm()
        if lgb is None:
            family = "sklearn"
        else:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            return QuantileModel(model=model, family="lightgbm")

    if family == "xgboost":
        xgb = _try_xgboost()
        if xgb is None:
            family = "sklearn"
        else:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            return QuantileModel(model=model, family="xgboost")

    if family == "quantile_forest":
        qrf = _try_quantile_forest()
        if qrf is None:
            family = "sklearn"
        else:
            model = qrf(
                n_estimators=300,
                max_depth=12,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            return QuantileModel(model=model, family="quantile_forest", quantile=quantile)

    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return QuantileModel(model=model, family="sklearn")


def select_primary_family(preference: tuple[str, ...]) -> str:
    available = available_families()
    for family in preference:
        if available.get(family, False):
            return family
    return "sklearn"
