from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import json

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def _logsumexp(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    maxv = np.max(arr, axis=axis, keepdims=True)
    return maxv + np.log(np.sum(np.exp(arr - maxv), axis=axis, keepdims=True))


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = mean.shape[0]
    cov = cov + np.eye(d) * 1.0e-9
    try:
        chol = np.linalg.cholesky(cov)
        diff = (X - mean).T
        sol = np.linalg.solve(chol, diff)
        quad = np.sum(sol * sol, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    except Exception:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(cov)) + 1.0e-12)
        diff = X - mean
        quad = np.sum(diff * np.dot(diff, np.linalg.inv(cov)), axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _get_covar(model, state: int, mix: int) -> np.ndarray:
    covars = model.covars_
    if covars.ndim == 4:
        return covars[state, mix]
    if covars.ndim == 3:
        if covars.shape[0] == model.n_components:
            return covars[state]
        if covars.shape[0] == model.n_mix:
            return covars[mix]
        return covars[0]
    if covars.ndim == 2:
        return covars
    raise ValueError("Unsupported covariance shape")


def _component_logpdfs(model, X: np.ndarray, state: int) -> np.ndarray:
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    logpdfs = np.zeros((X.shape[0], n_mix))
    for m in range(n_mix):
        mean = model.means_[state, m] if n_mix > 1 else model.means_[state]
        cov = _get_covar(model, state, m)
        logpdfs[:, m] = _log_gaussian(X, mean, cov)
    return logpdfs


def _responsibilities(model, X: np.ndarray, state: int) -> np.ndarray:
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    if n_mix == 1:
        return np.ones((X.shape[0], 1))
    logpdfs = _component_logpdfs(model, X, state)
    logw = np.log(np.clip(model.weights_[state], 1.0e-12, 1.0))
    log_joint = logpdfs + logw
    log_norm = _logsumexp(log_joint, axis=1)
    resp = np.exp(log_joint - log_norm)
    return resp


def _mixture_loglik(model, X: np.ndarray, state: int) -> np.ndarray:
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    if n_mix == 1:
        return _component_logpdfs(model, X, state)[:, 0]
    logpdfs = _component_logpdfs(model, X, state)
    logw = np.log(np.clip(model.weights_[state], 1.0e-12, 1.0))
    return _logsumexp(logpdfs + logw, axis=1)[:, 0]


def _param_count(d: int, cov_type: str, n_mix: int) -> int:
    weights = n_mix - 1 if n_mix > 1 else 0
    means = n_mix * d
    if cov_type == "full":
        covars = int(n_mix * d * (d + 1) / 2)
    elif cov_type == "tied":
        covars = int(d * (d + 1) / 2)
    else:
        raise ValueError("Unsupported covariance type")
    return int(weights + means + covars)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _weighted_choice(
    rng: np.random.Generator, weights: np.ndarray, size: int
) -> np.ndarray:
    weights = np.clip(weights.astype(float), 0.0, None)
    total = float(np.sum(weights))
    if total <= 0:
        return np.array([], dtype=int)
    probs = weights / total
    return rng.choice(weights.size, size=size, replace=False if size <= weights.size else True, p=probs)


def _safe_cluster_metrics_weighted(
    X: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray],
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[float]]:
    if X.shape[0] < 3 or len(np.unique(labels)) < 2:
        return None, None

    if X.shape[0] > max_samples:
        if weights is None:
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        else:
            idx = _weighted_choice(rng, weights, size=max_samples)
            if idx.size == 0:
                return None, None
        X_eval = X[idx]
        labels_eval = labels[idx]
    else:
        X_eval = X
        labels_eval = labels

    try:
        sil = float(silhouette_score(X_eval, labels_eval))
    except Exception:
        sil = None
    try:
        db = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception:
        db = None
    return sil, db


def state_gmm_metrics(
    model,
    X: np.ndarray,
    viterbi_states: np.ndarray,
    posteriors: Optional[np.ndarray],
    cov_type: str,
    subset: str,
    assign_mode: str,
    max_samples: int,
    rng: np.random.Generator,
    state_stability: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[Dict[str, object]]:
    n_states = model.n_components
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    results: List[Dict[str, object]] = []

    for state in range(n_states):
        if assign_mode == "weighted":
            if posteriors is None:
                raise ValueError("posteriors required for weighted state metrics.")
            weights = posteriors[:, state]
            X_state = X
            n_samples = int(X_state.shape[0])
            n_effective = float(np.sum(weights))
        else:
            idx = np.where(viterbi_states == state)[0]
            X_state = X[idx]
            n_samples = int(X_state.shape[0])
            n_effective = float(n_samples)
            weights = None

        if n_effective <= 0 or n_samples == 0:
            results.append(
                {
                    "state": state,
                    "subset": subset,
                    "assign_mode": assign_mode,
                    "n_samples": 0,
                    "n_effective": float(n_effective),
                    "avg_loglik": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "component_entropy": np.nan,
                    "component_occupancy": json.dumps({}, separators=(",", ":")),
                    "ari_mean": np.nan,
                    "ari_std": np.nan,
                }
            )
            continue

        loglik = _mixture_loglik(model, X_state, state)
        if assign_mode == "weighted":
            avg_loglik = _weighted_mean(loglik, weights)
        else:
            avg_loglik = float(np.mean(loglik))

        n_params = _param_count(X.shape[1] if assign_mode == "weighted" else X_state.shape[1], cov_type, n_mix)
        if assign_mode == "weighted":
            loglik_sum = float(np.sum(loglik * weights))
            aic = 2 * n_params - 2 * loglik_sum
            bic = n_params * np.log(max(n_effective, 1.0)) - 2 * loglik_sum
        else:
            aic = 2 * n_params - 2 * float(np.sum(loglik))
            bic = n_params * np.log(n_samples) - 2 * float(np.sum(loglik))

        resp = _responsibilities(model, X_state, state)
        if resp.shape[1] == 1:
            comp_entropy = 0.0
            comp_occ = {0: 1.0}
        else:
            eps = 1.0e-12
            ent = -np.sum(np.clip(resp, eps, 1.0) * np.log(resp + eps), axis=1)
            if assign_mode == "weighted":
                comp_entropy = _weighted_mean(ent, weights)
                comp_occ = {
                    int(i): float(np.sum(weights * resp[:, i]) / max(n_effective, 1.0))
                    for i in range(resp.shape[1])
                }
            else:
                comp_entropy = float(np.mean(ent))
                comp_occ = {int(i): float(np.mean(resp[:, i])) for i in range(resp.shape[1])}

        hard_labels = np.argmax(resp, axis=1)
        if X_state.shape[0] < 3 or len(np.unique(hard_labels)) < 2:
            sil = np.nan
            db = np.nan
        else:
            sil, db = _safe_cluster_metrics_weighted(
                X_state, hard_labels, weights if assign_mode == "weighted" else None, max_samples, rng
            )
            sil = np.nan if sil is None else sil
            db = np.nan if db is None else db

        ari_mean = np.nan
        ari_std = np.nan
        if state_stability and state in state_stability:
            ari_mean = state_stability[state].get("ari_mean", np.nan)
            ari_std = state_stability[state].get("ari_std", np.nan)

        results.append(
            {
                "state": state,
                "subset": subset,
                "assign_mode": assign_mode,
                "n_samples": n_samples,
                "n_effective": float(n_effective),
                "avg_loglik": avg_loglik,
                "aic": float(aic),
                "bic": float(bic),
                "silhouette": sil,
                "davies_bouldin": db,
                "component_entropy": comp_entropy,
                "component_occupancy": json.dumps(comp_occ, separators=(",", ":")),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
            }
        )

    return results


def state_component_posteriors(
    model,
    X: np.ndarray,
    viterbi_states: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    n_states = model.n_components
    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for state in range(n_states):
        idx = np.where(viterbi_states == state)[0]
        if idx.size == 0:
            continue
        resp = _responsibilities(model, X[idx], state)
        result[state] = (idx, resp)
    return result
