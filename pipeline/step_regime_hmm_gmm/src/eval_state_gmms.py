from __future__ import annotations

from typing import Dict, List, Tuple

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
    d = X.shape[1]
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


def state_gmm_metrics(
    model,
    X: np.ndarray,
    viterbi_states: np.ndarray,
    cov_type: str,
    subset: str,
) -> List[Dict[str, object]]:
    n_states = model.n_components
    n_mix = model.n_mix if hasattr(model, "n_mix") else 1
    results: List[Dict[str, object]] = []

    for state in range(n_states):
        idx = np.where(viterbi_states == state)[0]
        X_state = X[idx]
        n_samples = int(X_state.shape[0])
        if n_samples == 0:
            results.append(
                {
                    "state": state,
                    "subset": subset,
                    "n_samples": 0,
                    "avg_loglik": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "component_entropy": np.nan,
                    "component_occupancy": json.dumps({}, separators=(",", ":")),
                }
            )
            continue

        loglik = _mixture_loglik(model, X_state, state)
        avg_loglik = float(np.mean(loglik))

        n_params = _param_count(X_state.shape[1], cov_type, n_mix)
        aic = 2 * n_params - 2 * float(np.sum(loglik))
        bic = n_params * np.log(n_samples) - 2 * float(np.sum(loglik))

        resp = _responsibilities(model, X_state, state)
        if resp.shape[1] == 1:
            comp_entropy = 0.0
            comp_occ = {0: 1.0}
        else:
            eps = 1.0e-12
            comp_entropy = float(np.mean(-np.sum(np.clip(resp, eps, 1.0) * np.log(resp + eps), axis=1)))
            comp_occ = {int(i): float(np.mean(resp[:, i])) for i in range(resp.shape[1])}

        hard_labels = np.argmax(resp, axis=1)
        if X_state.shape[0] < 3 or len(np.unique(hard_labels)) < 2:
            sil = np.nan
            db = np.nan
        else:
            try:
                sil = float(silhouette_score(X_state, hard_labels))
            except Exception:
                sil = np.nan
            try:
                db = float(davies_bouldin_score(X_state, hard_labels))
            except Exception:
                db = np.nan

        results.append(
            {
                "state": state,
                "subset": subset,
                "n_samples": n_samples,
                "avg_loglik": avg_loglik,
                "aic": float(aic),
                "bic": float(bic),
                "silhouette": sil,
                "davies_bouldin": db,
                "component_entropy": comp_entropy,
                "component_occupancy": json.dumps(comp_occ, separators=(",", ":")),
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
