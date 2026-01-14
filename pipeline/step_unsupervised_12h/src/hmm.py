from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    HMM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None
    HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture

    GMM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None
    GMM_AVAILABLE = False


@dataclass
class CandidateResult:
    n_states: int
    model: object
    val_loglik: float
    train_loglik: float
    aic: float
    bic: float
    occupancy: np.ndarray
    avg_duration: float
    transition_entropy: float
    transition_sparsity: float
    notes: str


def compute_transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    trans = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans / row_sums


def average_run_length(states: np.ndarray) -> float:
    if len(states) == 0:
        return 0.0
    lengths = []
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            lengths.append(length)
            current = s
            length = 1
    lengths.append(length)
    return float(np.mean(lengths)) if lengths else 0.0


def transition_entropy(trans: np.ndarray) -> float:
    entropies = []
    for row in trans:
        row_safe = row[row > 0]
        if len(row_safe) == 0:
            continue
        entropies.append(-np.sum(row_safe * np.log(row_safe)))
    return float(np.mean(entropies)) if entropies else 0.0


def transition_sparsity(trans: np.ndarray, threshold: float = 0.01) -> float:
    if trans.size == 0:
        return 0.0
    return float(np.mean(trans < threshold))


def hmm_param_count(n_states: int, n_features: int) -> int:
    start_prob = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * (n_features * (n_features + 1) / 2)
    return int(start_prob + trans + means + covars)


def fit_candidates(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_states_list: list[int],
    random_state: int,
    max_iter: int = 200,
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    n_features = X_train.shape[1]
    for n_states in n_states_list:
        if HMM_AVAILABLE:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_train)
            train_loglik = float(model.score(X_train))
            val_loglik = float(model.score(X_val))
            params = hmm_param_count(n_states, n_features)
            aic = 2 * params - 2 * train_loglik
            bic = params * math.log(len(X_train)) - 2 * train_loglik
            states_train = model.predict(X_train)
            occupancy = np.bincount(states_train, minlength=n_states) / len(states_train)
            avg_dur = average_run_length(states_train)
            trans = model.transmat_
            entropy = transition_entropy(trans)
            sparsity = transition_sparsity(trans)
            notes = "hmmlearn"
        elif GMM_AVAILABLE:
            model = GaussianMixture(
                n_components=n_states,
                covariance_type="full",
                random_state=random_state,
                max_iter=max_iter,
            )
            model.fit(X_train)
            train_loglik = float(model.score(X_train) * len(X_train))
            val_loglik = float(model.score(X_val) * len(X_val))
            aic = float(model.aic(X_train))
            bic = float(model.bic(X_train))
            states_train = model.predict(X_train)
            occupancy = np.bincount(states_train, minlength=n_states) / len(states_train)
            avg_dur = average_run_length(states_train)
            trans = compute_transition_matrix(states_train, n_states)
            entropy = transition_entropy(trans)
            sparsity = transition_sparsity(trans)
            notes = "gmm_fallback"
        else:
            raise RuntimeError("Neither hmmlearn nor sklearn GaussianMixture is available.")

        results.append(
            CandidateResult(
                n_states=n_states,
                model=model,
                val_loglik=val_loglik,
                train_loglik=train_loglik,
                aic=aic,
                bic=bic,
                occupancy=occupancy,
                avg_duration=avg_dur,
                transition_entropy=entropy,
                transition_sparsity=sparsity,
                notes=notes,
            )
        )
    return results


def select_best_candidate(results: list[CandidateResult]) -> CandidateResult:
    val_scores = np.array([r.val_loglik for r in results], dtype=float)
    bic_scores = np.array([r.bic for r in results], dtype=float)

    val_norm = (val_scores - val_scores.mean()) / (val_scores.std() + 1e-9)
    bic_norm = (bic_scores - bic_scores.mean()) / (bic_scores.std() + 1e-9)

    best_idx = 0
    best_score = -np.inf
    for idx, res in enumerate(results):
        min_state = float(res.occupancy.min()) if len(res.occupancy) else 0.0
        penalty = 0.0
        if min_state < 0.02:
            penalty += 1.0
        if res.avg_duration < 2.0:
            penalty += 0.5
        score = val_norm[idx] - 0.5 * bic_norm[idx] - penalty
        if score > best_score:
            best_score = score
            best_idx = idx
    return results[best_idx]


def decode_states(model: object, X: np.ndarray) -> np.ndarray:
    if HMM_AVAILABLE and isinstance(model, GaussianHMM):
        return model.predict(X)
    if GMM_AVAILABLE and isinstance(model, GaussianMixture):
        return model.predict(X)
    raise RuntimeError("Unsupported model for state decoding.")
