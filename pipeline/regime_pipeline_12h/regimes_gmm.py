from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMSelectionResult:
    model: GaussianMixture
    n_states: int
    seed: int
    metrics: dict[str, Any]


def regime_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(posteriors, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def fit_gmm(
    X: np.ndarray,
    n_states: int,
    seed: int,
    n_iter: int,
    cov_type: str,
) -> GaussianMixture:
    model = GaussianMixture(
        n_components=n_states,
        covariance_type=cov_type,
        max_iter=n_iter,
        random_state=seed,
    )
    model.fit(X)
    return model


def select_gmm_model(
    X_train: np.ndarray,
    ks: list[int],
    seeds: list[int],
    n_iter: int,
    cov_type: str,
) -> GMMSelectionResult:
    candidates = []
    for k in ks:
        for seed in seeds:
            model = fit_gmm(X_train, k, seed, n_iter, cov_type)
            bic = model.bic(X_train)
            aic = model.aic(X_train)
            candidates.append(
                {
                    "n_states": k,
                    "seed": seed,
                    "bic": bic,
                    "aic": aic,
                }
            )

    best_entry = min(candidates, key=lambda r: r["bic"])
    best_model = fit_gmm(X_train, best_entry["n_states"], best_entry["seed"], n_iter, cov_type)
    return GMMSelectionResult(
        model=best_model,
        n_states=best_entry["n_states"],
        seed=best_entry["seed"],
        metrics={"selection": best_entry, "candidates": candidates, "status": "ok"},
    )


def predict_gmm(model: GaussianMixture, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posteriors = model.predict_proba(X)
    states = posteriors.argmax(axis=1)
    ent = regime_entropy(posteriors)
    return states, posteriors, ent
