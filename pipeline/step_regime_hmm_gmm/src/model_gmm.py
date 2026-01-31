from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


@dataclass
class GMMRunResult:
    k: int
    cov_type: str
    seed: int
    train_loglik_avg: float
    test_loglik_avg: float
    train_aic: float
    train_bic: float
    silhouette_train: Optional[float]
    silhouette_test: Optional[float]
    db_train: Optional[float]
    db_test: Optional[float]
    resp_entropy_mean: float
    n_runs: int


def _safe_cluster_metrics(
    X: np.ndarray, labels: np.ndarray, max_samples: int, rng: np.random.Generator
) -> Tuple[Optional[float], Optional[float]]:
    if X.shape[0] < 3 or len(np.unique(labels)) < 2:
        return None, None

    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_eval = X[idx]
        labels_eval = labels[idx]
    else:
        X_eval = X
        labels_eval = labels

    try:
        sil = silhouette_score(X_eval, labels_eval)
    except Exception:
        sil = None
    try:
        db = davies_bouldin_score(X_eval, labels_eval)
    except Exception:
        db = None
    return sil, db


def _responsibility_entropy(resp: np.ndarray) -> np.ndarray:
    eps = 1e-12
    resp_safe = np.clip(resp, eps, 1.0)
    return -np.sum(resp_safe * np.log(resp_safe), axis=1)


def run_gmm_grid(
    X_train: np.ndarray,
    X_test: np.ndarray,
    components: List[int],
    cov_types: List[str],
    n_runs: int,
    n_init: int,
    max_iter: int,
    max_samples_metrics: int,
    seed: int,
) -> Tuple[List[GMMRunResult], List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    run_results: List[GMMRunResult] = []
    summary_rows: List[Dict[str, float]] = []

    for k in components:
        for cov_type in cov_types:
            seeds = rng.integers(0, 1_000_000, size=n_runs)
            per_run: List[GMMRunResult] = []
            hard_labels_runs = []

            for run_seed in seeds:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=int(run_seed),
                    n_init=n_init,
                    max_iter=max_iter,
                )
                model.fit(X_train)

                train_ll_avg = float(model.score(X_train))
                test_ll_avg = float(model.score(X_test))
                train_aic = float(model.aic(X_train))
                train_bic = float(model.bic(X_train))

                train_labels = model.predict(X_train)
                test_labels = model.predict(X_test)

                sil_train, db_train = _safe_cluster_metrics(
                    X_train, train_labels, max_samples_metrics, rng
                )
                sil_test, db_test = _safe_cluster_metrics(
                    X_test, test_labels, max_samples_metrics, rng
                )

                resp = model.predict_proba(X_train)
                resp_entropy = _responsibility_entropy(resp).mean()

                result = GMMRunResult(
                    k=k,
                    cov_type=cov_type,
                    seed=int(run_seed),
                    train_loglik_avg=train_ll_avg,
                    test_loglik_avg=test_ll_avg,
                    train_aic=train_aic,
                    train_bic=train_bic,
                    silhouette_train=sil_train,
                    silhouette_test=sil_test,
                    db_train=db_train,
                    db_test=db_test,
                    resp_entropy_mean=float(resp_entropy),
                    n_runs=n_runs,
                )
                per_run.append(result)
                run_results.append(result)
                hard_labels_runs.append(train_labels)

            ari_scores = []
            for i in range(len(hard_labels_runs)):
                for j in range(i + 1, len(hard_labels_runs)):
                    ari_scores.append(
                        adjusted_rand_score(hard_labels_runs[i], hard_labels_runs[j])
                    )
            ari_mean = float(np.mean(ari_scores)) if ari_scores else np.nan
            ari_std = float(np.std(ari_scores)) if ari_scores else np.nan

            summary_rows.append(
                {
                    "k": k,
                    "cov_type": cov_type,
                    "train_loglik_avg_mean": float(np.mean([r.train_loglik_avg for r in per_run])),
                    "train_loglik_avg_std": float(np.std([r.train_loglik_avg for r in per_run])),
                    "test_loglik_avg_mean": float(np.mean([r.test_loglik_avg for r in per_run])),
                    "test_loglik_avg_std": float(np.std([r.test_loglik_avg for r in per_run])),
                    "train_aic_mean": float(np.mean([r.train_aic for r in per_run])),
                    "train_aic_std": float(np.std([r.train_aic for r in per_run])),
                    "train_bic_mean": float(np.mean([r.train_bic for r in per_run])),
                    "train_bic_std": float(np.std([r.train_bic for r in per_run])),
                    "silhouette_train_mean": float(
                        np.nanmean(
                            [
                                r.silhouette_train if r.silhouette_train is not None else np.nan
                                for r in per_run
                            ]
                        )
                    ),
                    "silhouette_test_mean": float(
                        np.nanmean(
                            [
                                r.silhouette_test if r.silhouette_test is not None else np.nan
                                for r in per_run
                            ]
                        )
                    ),
                    "db_train_mean": float(
                        np.nanmean([r.db_train if r.db_train is not None else np.nan for r in per_run])
                    ),
                    "db_test_mean": float(
                        np.nanmean([r.db_test if r.db_test is not None else np.nan for r in per_run])
                    ),
                    "resp_entropy_mean": float(np.mean([r.resp_entropy_mean for r in per_run])),
                    "ari_mean": ari_mean,
                    "ari_std": ari_std,
                    "n_runs": n_runs,
                }
            )

    return run_results, summary_rows
