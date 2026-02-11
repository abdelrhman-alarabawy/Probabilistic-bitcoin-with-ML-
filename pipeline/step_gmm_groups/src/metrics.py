from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

EPS = 1e-12


def gmm_num_params(n_components: int, n_features: int, covariance_type: str) -> int:
    weights_params = n_components - 1
    means_params = n_components * n_features
    cov_type = covariance_type.lower()
    if cov_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif cov_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type '{covariance_type}'.")
    return int(weights_params + means_params + cov_params)


def aic_bic_from_loglik(loglik_total: float, n_samples: int, n_params: int) -> Tuple[float, float]:
    if n_samples <= 0:
        return float("nan"), float("nan")
    aic = 2.0 * n_params - 2.0 * loglik_total
    bic = np.log(float(n_samples)) * n_params - 2.0 * loglik_total
    return float(aic), float(bic)


def responsibility_stats(resp: np.ndarray) -> Dict[str, float]:
    if resp.size == 0:
        return {
            "avg_entropy": float("nan"),
            "avg_probmax": float("nan"),
            "p_prob_ge_0_9": float("nan"),
        }
    probmax = np.max(resp, axis=1)
    entropy = -np.sum(resp * np.log(resp + EPS), axis=1)
    return {
        "avg_entropy": float(np.mean(entropy)),
        "avg_probmax": float(np.mean(probmax)),
        "p_prob_ge_0_9": float(np.mean(probmax >= 0.9)),
    }


def _safe_metric(metric_fn, X: np.ndarray, labels: np.ndarray) -> float:
    if X.size == 0 or labels.size == 0:
        return float("nan")
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2:
        return float("nan")
    if X.shape[0] <= unique_labels.shape[0]:
        return float("nan")
    try:
        return float(metric_fn(X, labels))
    except Exception:
        return float("nan")


def separation_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "silhouette": _safe_metric(silhouette_score, X, labels),
        "davies_bouldin": _safe_metric(davies_bouldin_score, X, labels),
        "calinski_harabasz": _safe_metric(calinski_harabasz_score, X, labels),
    }


def run_length_stats(labels: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {"avg_run_len": float("nan"), "median_run_len": float("nan"), "num_runs": float("nan")}
    run_lengths = []
    current_len = 1
    for i in range(1, labels.size):
        if labels[i] == labels[i - 1]:
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_len = 1
    run_lengths.append(current_len)
    run_lengths_arr = np.array(run_lengths, dtype=float)
    return {
        "avg_run_len": float(np.mean(run_lengths_arr)),
        "median_run_len": float(np.median(run_lengths_arr)),
        "num_runs": float(run_lengths_arr.shape[0]),
    }


def transition_matrix(labels: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, float]:
    counts = np.zeros((n_components, n_components), dtype=int)
    if labels.size >= 2:
        for i in range(labels.size - 1):
            src = int(labels[i])
            dst = int(labels[i + 1])
            counts[src, dst] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )
    total_transitions = int(counts.sum())
    p_stay = float(np.trace(counts) / total_transitions) if total_transitions > 0 else float("nan")
    return counts, probs, p_stay


def component_fractions(labels: np.ndarray, n_components: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=n_components).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(n_components, dtype=float)
    return counts / total


def component_var_summary(gmm: GaussianMixture) -> np.ndarray:
    cov_type = gmm.covariance_type
    if cov_type == "full":
        covs = np.asarray(gmm.covariances_)
        traces = np.trace(covs, axis1=1, axis2=2)
        return traces / covs.shape[1]
    if cov_type == "tied":
        cov = np.asarray(gmm.covariances_)
        per_component = np.repeat(np.trace(cov) / cov.shape[0], gmm.n_components)
        return per_component.astype(float)
    return np.full(gmm.n_components, np.nan, dtype=float)


def compute_run_metrics(
    gmm: GaussianMixture,
    X_train: np.ndarray,
    X_test: np.ndarray,
    runtime_fit_seconds: float,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    train_ll_samples = gmm.score_samples(X_train)
    test_ll_samples = gmm.score_samples(X_test)

    train_loglik_total = float(np.sum(train_ll_samples))
    test_loglik_total = float(np.sum(test_ll_samples))
    train_avg_loglik = float(np.mean(train_ll_samples))
    test_avg_loglik = float(np.mean(test_ll_samples))

    n_params = gmm_num_params(gmm.n_components, X_train.shape[1], gmm.covariance_type)
    aic_test, bic_test = aic_bic_from_loglik(test_loglik_total, X_test.shape[0], n_params)

    resp_train = gmm.predict_proba(X_train)
    resp_test = gmm.predict_proba(X_test)
    labels_train = np.argmax(resp_train, axis=1).astype(int)
    labels_test = np.argmax(resp_test, axis=1).astype(int)

    resp_stats_train = responsibility_stats(resp_train)
    resp_stats_test = responsibility_stats(resp_test)
    sep_train = separation_metrics(X_train, labels_train)
    sep_test = separation_metrics(X_test, labels_test)

    labels_concat = np.concatenate([labels_train, labels_test], axis=0)
    persistence = run_length_stats(labels_concat)
    transition_counts, transition_probs, p_stay = transition_matrix(labels_concat, gmm.n_components)

    component_train = component_fractions(labels_train, gmm.n_components)
    component_test = component_fractions(labels_test, gmm.n_components)
    mean_norms = np.linalg.norm(gmm.means_, axis=1)
    var_summary = component_var_summary(gmm)

    metrics = {
        "train_loglik_total": train_loglik_total,
        "train_avg_loglik": train_avg_loglik,
        "test_loglik_total": test_loglik_total,
        "test_avg_loglik": test_avg_loglik,
        "aic_train": float(gmm.aic(X_train)),
        "bic_train": float(gmm.bic(X_train)),
        "aic_test": aic_test,
        "bic_test": bic_test,
        "n_params": int(n_params),
        "avg_entropy_train": resp_stats_train["avg_entropy"],
        "avg_entropy_test": resp_stats_test["avg_entropy"],
        "avg_probmax_train": resp_stats_train["avg_probmax"],
        "avg_probmax_test": resp_stats_test["avg_probmax"],
        "p_prob_ge_0_9_train": resp_stats_train["p_prob_ge_0_9"],
        "p_prob_ge_0_9_test": resp_stats_test["p_prob_ge_0_9"],
        "silhouette_train": sep_train["silhouette"],
        "silhouette_test": sep_test["silhouette"],
        "davies_bouldin_train": sep_train["davies_bouldin"],
        "davies_bouldin_test": sep_test["davies_bouldin"],
        "calinski_harabasz_train": sep_train["calinski_harabasz"],
        "calinski_harabasz_test": sep_test["calinski_harabasz"],
        "avg_run_len": persistence["avg_run_len"],
        "median_run_len": persistence["median_run_len"],
        "num_runs": persistence["num_runs"],
        "p_stay": p_stay,
        "component_fraction_train": json.dumps(component_train.tolist()),
        "component_fraction_test": json.dumps(component_test.tolist()),
        "component_mean_norm": json.dumps(mean_norms.astype(float).tolist()),
        "component_var_summary": json.dumps(var_summary.astype(float).tolist()),
        "converged": bool(getattr(gmm, "converged_", False)),
        "n_iter": int(getattr(gmm, "n_iter_", -1)),
        "runtime_fit_seconds": float(runtime_fit_seconds),
    }

    diagnostics = {
        "train_labels": labels_train,
        "test_labels": labels_test,
        "train_responsibilities": resp_train,
        "test_responsibilities": resp_test,
        "transition_counts": transition_counts,
        "transition_probs": transition_probs,
    }
    return metrics, diagnostics

