from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GaussianHMM = None
    HMM_AVAILABLE = False


@dataclass
class HMMSelectionResult:
    model: Optional[Any]
    n_states: Optional[int]
    seed: Optional[int]
    metrics: dict[str, Any]


def regime_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(posteriors, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _average_durations(states: np.ndarray, n_states: int) -> np.ndarray:
    durations = {state: [] for state in range(n_states)}
    run = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            run += 1
        else:
            durations[states[i - 1]].append(run)
            run = 1
    if len(states) > 0:
        durations[states[-1]].append(run)
    avg = np.array([np.mean(durations[s]) if durations[s] else 0.0 for s in range(n_states)])
    return avg


def _state_fractions(states: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.bincount(states, minlength=n_states)
    return counts / max(len(states), 1)


def fit_hmm(
    X: np.ndarray,
    n_states: int,
    seed: int,
    n_iter: int,
    cov_type: str,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cov_type,
        n_iter=n_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    return model


def select_hmm_model(
    X_train: np.ndarray,
    ks: List[int],
    seeds: List[int],
    n_iter: int,
    cov_type: str,
    val_ratio: float,
    min_state_frac: float,
    min_avg_duration: float,
    stable_fraction: float,
) -> HMMSelectionResult:
    if not HMM_AVAILABLE:
        return HMMSelectionResult(None, None, None, {"status": "hmmlearn_missing"})

    n_samples = len(X_train)
    split = int(n_samples * (1 - val_ratio))
    if split <= 0 or split >= n_samples:
        raise ValueError("Invalid validation split for HMM selection.")

    X_subtrain = X_train[:split]
    X_val = X_train[split:]

    results = []
    for k in ks:
        for seed in seeds:
            try:
                model = fit_hmm(X_subtrain, k, seed, n_iter, cov_type)
                ll = model.score(X_val)
                post = model.predict_proba(X_val)
                states = post.argmax(axis=1)
                fractions = _state_fractions(states, k)
                avg_durations = _average_durations(states, k)
                stable_count = (avg_durations >= min_avg_duration).sum()
                stable_ratio = stable_count / k if k else 0.0
                degenerate = (fractions.min() < min_state_frac) or (stable_ratio < stable_fraction)
                results.append(
                    {
                        "n_states": k,
                        "seed": seed,
                        "log_likelihood": ll,
                        "min_state_frac": float(fractions.min()),
                        "stable_ratio": float(stable_ratio),
                        "degenerate": degenerate,
                    }
                )
            except Exception as exc:  # pragma: no cover
                results.append(
                    {
                        "n_states": k,
                        "seed": seed,
                        "log_likelihood": -np.inf,
                        "min_state_frac": 0.0,
                        "stable_ratio": 0.0,
                        "degenerate": True,
                        "error": str(exc),
                    }
                )

    viable = [r for r in results if not r["degenerate"]]
    if viable:
        best_entry = max(viable, key=lambda r: r["log_likelihood"])
        status = "ok"
    else:
        best_entry = max(results, key=lambda r: r["log_likelihood"])
        status = "degenerate_fallback"

    best_model = fit_hmm(X_train, best_entry["n_states"], best_entry["seed"], n_iter, cov_type)
    return HMMSelectionResult(
        model=best_model,
        n_states=best_entry["n_states"],
        seed=best_entry["seed"],
        metrics={"selection": best_entry, "candidates": results, "status": status},
    )


def predict_hmm(model: GaussianHMM, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posteriors = model.predict_proba(X)
    states = posteriors.argmax(axis=1)
    ent = regime_entropy(posteriors)
    return states, posteriors, ent
