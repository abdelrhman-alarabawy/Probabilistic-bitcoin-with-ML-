from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class GMMSelectionResult:
    best_k: int
    bic_scores: dict[int, float]


def select_k_bic(
    X_train: np.ndarray,
    k_list: Iterable[int],
    random_state: int,
) -> GMMSelectionResult:
    bic_scores: dict[int, float] = {}
    best_k = None
    best_bic = float("inf")
    for k in k_list:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        )
        model.fit(X_train)
        bic = float(model.bic(X_train))
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_k = k
    if best_k is None:
        raise RuntimeError("Failed to select a GMM component count.")
    return GMMSelectionResult(best_k=best_k, bic_scores=bic_scores)


def fit_gmm(
    X_train: np.ndarray,
    n_components: int,
    random_state: int,
) -> GaussianMixture:
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def assign_regimes(model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def compute_regime_stats(
    df: pd.DataFrame,
    regime_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
    label_col: str,
) -> pd.DataFrame:
    returns = df[close_col].pct_change().fillna(0.0)
    vol = returns.rolling(window=10, min_periods=2).std().fillna(0.0)
    range_pct = (df[high_col] - df[low_col]) / df[close_col].replace(0, np.nan)
    range_pct = range_pct.fillna(0.0)

    stats = pd.DataFrame(
        {
            "regime_id": df[regime_col],
            "mean_return": returns,
            "volatility": vol,
            "range_pct": range_pct,
        }
    )
    grouped = stats.groupby("regime_id").mean()
    grouped["occupancy"] = df[regime_col].value_counts(normalize=True).sort_index()
    label_dist = (
        df.groupby([regime_col, label_col])
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .unstack(fill_value=0.0)
    )
    grouped = grouped.join(label_dist, how="left")
    return grouped.reset_index()
