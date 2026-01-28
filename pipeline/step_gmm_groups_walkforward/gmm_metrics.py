from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Type

import logging

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

EPS = 1e-12


@dataclass
class RunMetrics:
    seed: int
    train_avg_ll: float
    test_avg_ll: float
    train_bic: float
    train_aic: float
    test_bic: float
    test_aic: float
    train_silhouette: float
    train_davies_bouldin: float
    test_silhouette: float
    test_davies_bouldin: float
    train_resp_entropy: float
    test_resp_entropy: float
    weights: np.ndarray
    means: np.ndarray


def compute_avg_loglik(gmm: GaussianMixture, X: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    return float(np.mean(gmm.score_samples(X)))


def _num_params(covariance_type: str, n_components: int, n_features: int) -> int:
    weights_params = n_components - 1
    means_params = n_components * n_features
    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "spherical":
        cov_params = n_components
    else:
        cov_params = n_components * n_features
    return int(weights_params + means_params + cov_params)


def compute_bic_aic(model: object, X: np.ndarray) -> Tuple[float, float]:
    if X.size == 0:
        return float("nan"), float("nan")
    if hasattr(model, "bic") and hasattr(model, "aic"):
        try:
            return float(model.bic(X)), float(model.aic(X))
        except Exception:
            pass

    try:
        n_samples, n_features = X.shape
        covariance_type = getattr(model, "covariance_type", "full")
        n_components = getattr(model, "n_components", None)
        if n_components is None:
            n_components = int(getattr(model, "weights_", np.array([])).shape[0])
        n_params = _num_params(str(covariance_type), int(n_components), int(n_features))
        log_lik = float(np.sum(model.score_samples(X)))
        bic = -2.0 * log_lik + n_params * np.log(n_samples)
        aic = -2.0 * log_lik + 2.0 * n_params
        return float(bic), float(aic)
    except Exception:
        return float("nan"), float("nan")


def _labels_and_entropy(gmm: GaussianMixture, X: np.ndarray) -> Tuple[np.ndarray, float]:
    if X.size == 0:
        return np.array([]), float("nan")
    resp = gmm.predict_proba(X)
    labels = np.argmax(resp, axis=1)
    entropy = -np.sum(resp * np.log(resp + EPS), axis=1)
    return labels, float(np.mean(entropy))


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if X.size == 0 or labels.size == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        logging.warning("Silhouette undefined (single cluster); returning NaN.")
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    if X.size == 0 or labels.size == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        logging.warning("Davies-Bouldin undefined (single cluster); returning NaN.")
        return float("nan")
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def fit_gmm_and_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
    covariance_type: str,
    k: int,
    seed: int,
    max_iter: int = 500,
    tol: float = 1e-4,
    reg_covar: float = 1e-6,
    model_class: Type = GaussianMixture,
    model_kwargs: Optional[dict] = None,
) -> RunMetrics:
    params = {
        "n_components": k,
        "covariance_type": covariance_type,
        "random_state": seed,
        "n_init": 1,
        "max_iter": max_iter,
        "tol": tol,
        "reg_covar": reg_covar,
        "init_params": "kmeans",
    }
    if model_kwargs:
        params.update(model_kwargs)
    gmm = model_class(**params)
    gmm.fit(X_train)

    train_avg_ll = compute_avg_loglik(gmm, X_train)
    test_avg_ll = compute_avg_loglik(gmm, X_test)

    train_bic, train_aic = compute_bic_aic(gmm, X_train)
    test_bic, test_aic = compute_bic_aic(gmm, X_test)

    train_labels, train_entropy = _labels_and_entropy(gmm, X_train)
    test_labels, test_entropy = _labels_and_entropy(gmm, X_test)

    train_sil = safe_silhouette(X_train, train_labels)
    train_db = safe_davies_bouldin(X_train, train_labels)
    test_sil = safe_silhouette(X_test, test_labels)
    test_db = safe_davies_bouldin(X_test, test_labels)

    return RunMetrics(
        seed=seed,
        train_avg_ll=train_avg_ll,
        test_avg_ll=test_avg_ll,
        train_bic=train_bic,
        train_aic=train_aic,
        test_bic=test_bic,
        test_aic=test_aic,
        train_silhouette=train_sil,
        train_davies_bouldin=train_db,
        test_silhouette=test_sil,
        test_davies_bouldin=test_db,
        train_resp_entropy=train_entropy,
        test_resp_entropy=test_entropy,
        weights=gmm.weights_.copy(),
        means=gmm.means_.copy(),
    )


def _hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _align_components(reference_means: np.ndarray, means: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cost = np.linalg.norm(reference_means[:, None, :] - means[None, :, :], axis=2)
    assignment = _hungarian_min_cost(cost)
    return weights[assignment], means[assignment]


def select_best_run(runs: Iterable[RunMetrics]) -> int:
    runs_list = list(runs)
    if not runs_list:
        raise ValueError("No runs provided.")
    test_ll = np.array([r.test_avg_ll for r in runs_list], dtype=float)
    if np.all(np.isnan(test_ll)):
        train_bic = np.array([r.train_bic for r in runs_list], dtype=float)
        return int(np.nanargmin(train_bic))
    best = np.nanargmax(test_ll)
    return int(best)


def compute_stability(runs: List[RunMetrics]) -> Tuple[float, float, float]:
    if not runs:
        return float("nan"), float("nan"), float("nan")
    test_ll = np.array([r.test_avg_ll for r in runs], dtype=float)
    ll_std = float(np.nanstd(test_ll, ddof=1)) if len(runs) > 1 else 0.0

    ref_means = runs[0].means
    aligned_weights = []
    aligned_means = []

    for r in runs:
        w_aligned, m_aligned = _align_components(ref_means, r.means, r.weights)
        aligned_weights.append(w_aligned)
        aligned_means.append(m_aligned)

    aligned_weights = np.array(aligned_weights)
    aligned_means = np.array(aligned_means)

    if len(runs) > 1:
        weight_std = np.std(aligned_weights, axis=0, ddof=1)
        weight_std_mean = float(np.mean(weight_std))
    else:
        weight_std_mean = 0.0

    mean_shift_stds = []
    for comp in range(aligned_means.shape[1]):
        dists = np.linalg.norm(aligned_means[:, comp, :] - ref_means[comp], axis=1)
        if len(dists) > 1:
            mean_shift_stds.append(float(np.std(dists, ddof=1)))
        else:
            mean_shift_stds.append(0.0)
    mean_shift_std_mean = float(np.mean(mean_shift_stds)) if mean_shift_stds else 0.0

    return ll_std, weight_std_mean, mean_shift_std_mean
