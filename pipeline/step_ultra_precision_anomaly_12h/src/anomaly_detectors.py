from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def rolling_robust_zscores(
    df: pd.DataFrame, feature_cols: List[str], window: int = 200
) -> pd.DataFrame:
    zscores = {}
    for col in feature_cols:
        series = df[col].astype(float)
        median = series.rolling(window=window, min_periods=window).median()
        mad = (series - median).abs().rolling(window=window, min_periods=window).median()
        mad = mad.replace(0.0, np.nan)
        z = 0.6745 * (series - median) / mad
        zscores[col] = z
    return pd.DataFrame(zscores, index=df.index)


def robust_zscore_anomaly_score(
    df: pd.DataFrame, feature_cols: List[str], window: int = 200
) -> pd.Series:
    zscores = rolling_robust_zscores(df, feature_cols, window=window)
    score = zscores.abs().mean(axis=1, skipna=True).fillna(0.0)
    return score


@dataclass
class IFDetector:
    model: Pipeline


def fit_isolation_forest(
    X_train: pd.DataFrame, random_seed: int = 42
) -> IFDetector:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.01,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(X_train)
    return IFDetector(model=pipeline)


def score_isolation_forest(detector: IFDetector, X: pd.DataFrame) -> np.ndarray:
    scores = detector.model.decision_function(X)
    return -scores


@dataclass
class GMMDetector:
    model: GaussianMixture
    scaler: RobustScaler
    train_freq: Dict[int, float]


def fit_gmm(
    X_train: pd.DataFrame, k_range: List[int] | None = None, random_seed: int = 42
) -> GMMDetector:
    if k_range is None:
        k_range = [2, 3, 4, 5, 6]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    best_bic = np.inf
    best_model = None
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", max_iter=500, random_state=random_seed
        )
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        if bic < best_bic:
            best_bic = bic
            best_model = gmm

    if best_model is None:
        raise RuntimeError("GMM fitting failed.")

    labels = best_model.predict(X_scaled)
    unique, counts = np.unique(labels, return_counts=True)
    freq = {int(k): float(v) / float(len(labels)) for k, v in zip(unique, counts)}
    return GMMDetector(model=best_model, scaler=scaler, train_freq=freq)


def score_gmm(detector: GMMDetector, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_scaled = detector.scaler.transform(X)
    probs = detector.model.predict_proba(X_scaled)
    labels = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rarity = np.array([1.0 - detector.train_freq.get(int(cid), 0.0) for cid in labels])
    return labels, conf, rarity
