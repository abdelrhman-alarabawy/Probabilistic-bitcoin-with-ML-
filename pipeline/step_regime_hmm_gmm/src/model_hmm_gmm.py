from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json

import numpy as np

try:
    from hmmlearn.hmm import GMMHMM, GaussianHMM
except Exception:  # pragma: no cover
    GMMHMM = None
    GaussianHMM = None


@dataclass
class HMMRunResult:
    k: int
    cov_type: str
    n_mix: int
    seed: int
    train_loglik: float
    test_loglik: float
    train_loglik_avg: float
    test_loglik_avg: float
    aic: float
    bic: float
    entropy_train: float
    entropy_test: float
    switches_test: int
    mean_self_transition: float
    state_occupancy_train: str
    state_occupancy_test: str
    stationary_dist: str


def _state_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(posteriors, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _count_switches(states: np.ndarray) -> int:
    if states.size <= 1:
        return 0
    return int(np.sum(states[1:] != states[:-1]))


def _expected_durations(transmat: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(transmat), 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 - diag)


def _param_count(k: int, d: int, cov_type: str, n_mix: int) -> int:
    start = k - 1
    trans = k * (k - 1)
    if n_mix == 1:
        means = k * d
        if cov_type == "full":
            covars = int(k * d * (d + 1) / 2)
        elif cov_type == "tied":
            covars = int(d * (d + 1) / 2)
        else:
            raise ValueError("Unsupported covariance type")
        weights = 0
    else:
        weights = k * (n_mix - 1)
        means = k * n_mix * d
        if cov_type == "full":
            covars = int(k * n_mix * d * (d + 1) / 2)
        elif cov_type == "tied":
            covars = int(k * d * (d + 1) / 2)
        else:
            raise ValueError("Unsupported covariance type")
    return int(start + trans + weights + means + covars)


def _stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    try:
        eigvals, eigvecs = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        vec = np.real(eigvecs[:, idx])
        vec = np.where(vec < 0, 0, vec)
        total = vec.sum()
        if total == 0:
            return np.ones(transmat.shape[0]) / transmat.shape[0]
        return vec / total
    except Exception:
        return np.ones(transmat.shape[0]) / transmat.shape[0]


def _occupancy(states: np.ndarray, k: int) -> Dict[int, float]:
    counts = {int(i): 0 for i in range(k)}
    total = max(int(states.size), 1)
    for i in range(k):
        counts[int(i)] = float(np.sum(states == i)) / float(total)
    return counts


def _build_model(k: int, cov_type: str, n_mix: int, seed: int, cfg: dict):
    if n_mix > 1:
        if GMMHMM is None:
            raise RuntimeError("hmmlearn GMMHMM not available.")
        init_params = cfg["init_params"]
        params = cfg["params"]
        return GMMHMM(
            n_components=k,
            n_mix=n_mix,
            covariance_type=cov_type,
            n_iter=cfg["n_iter"],
            tol=cfg["tol"],
            init_params=init_params,
            params=params,
            min_covar=cfg.get("min_covar", 1.0e-6),
            random_state=seed,
        )

    if GaussianHMM is None:
        raise RuntimeError("hmmlearn GaussianHMM not available.")
    init_params = cfg["init_params"].replace("w", "")
    params = cfg["params"].replace("w", "")
    return GaussianHMM(
        n_components=k,
        covariance_type=cov_type,
        n_iter=cfg["n_iter"],
        tol=cfg["tol"],
        init_params=init_params,
        params=params,
        min_covar=cfg.get("min_covar", 1.0e-6),
        random_state=seed,
    )


def run_hmm_grid(
    X_train: np.ndarray,
    X_test: np.ndarray,
    states: List[int],
    cov_types: List[str],
    n_mix_list: List[int],
    n_runs: int,
    seed: int,
    cfg: dict,
) -> Tuple[List[HMMRunResult], List[Dict[str, float]], Dict[str, Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    run_results: List[HMMRunResult] = []
    summary_rows: List[Dict[str, float]] = []
    best_artifacts: Dict[str, Dict[str, object]] = {}

    for k in states:
        for cov_type in cov_types:
            for n_mix in n_mix_list:
                seeds = rng.integers(0, 1_000_000, size=n_runs)
                per_run: List[HMMRunResult] = []
                best_run = None
                best_ll = -np.inf
                best_pack = None

                for run_seed in seeds:
                    try:
                        model = _build_model(k, cov_type, n_mix, int(run_seed), cfg)
                        model.fit(X_train)
                        train_ll = float(model.score(X_train))
                        test_ll = float(model.score(X_test))
                        train_ll_avg = train_ll / X_train.shape[0]
                        test_ll_avg = test_ll / X_test.shape[0]

                        post_train = model.predict_proba(X_train)
                        post_test = model.predict_proba(X_test)

                        entropy_train = float(np.mean(_state_entropy(post_train)))
                        entropy_test = float(np.mean(_state_entropy(post_test)))

                        viterbi_train = model.predict(X_train)
                        viterbi_test = model.predict(X_test)
                        switches_test = _count_switches(viterbi_test)
                        mean_self = float(np.mean(np.diag(model.transmat_)))

                        n_params = _param_count(k, X_train.shape[1], cov_type, n_mix)
                        aic = 2 * n_params - 2 * train_ll
                        bic = n_params * np.log(X_train.shape[0]) - 2 * train_ll

                        occ_train = _occupancy(viterbi_train, k)
                        occ_test = _occupancy(viterbi_test, k)
                        stationary = _stationary_distribution(model.transmat_)

                        result = HMMRunResult(
                            k=k,
                            cov_type=cov_type,
                            n_mix=n_mix,
                            seed=int(run_seed),
                            train_loglik=train_ll,
                            test_loglik=test_ll,
                            train_loglik_avg=train_ll_avg,
                            test_loglik_avg=test_ll_avg,
                            aic=float(aic),
                            bic=float(bic),
                            entropy_train=entropy_train,
                            entropy_test=entropy_test,
                            switches_test=int(switches_test),
                            mean_self_transition=mean_self,
                            state_occupancy_train=json.dumps(occ_train, separators=(",", ":")),
                            state_occupancy_test=json.dumps(occ_test, separators=(",", ":")),
                            stationary_dist=json.dumps(
                                {int(i): float(v) for i, v in enumerate(stationary)},
                                separators=(",", ":"),
                            ),
                        )
                        per_run.append(result)
                        run_results.append(result)

                        if train_ll > best_ll:
                            best_ll = train_ll
                            best_run = result
                            best_pack = {
                                "model": model,
                                "post_train": post_train,
                                "post_test": post_test,
                                "viterbi_train": viterbi_train,
                                "viterbi_test": viterbi_test,
                            }
                    except Exception:
                        continue

                if not per_run:
                    continue

                summary_rows.append(
                    {
                        "k": k,
                        "cov_type": cov_type,
                        "n_mix": n_mix,
                        "train_loglik_avg_mean": float(np.mean([r.train_loglik_avg for r in per_run])),
                        "train_loglik_avg_std": float(np.std([r.train_loglik_avg for r in per_run])),
                        "test_loglik_avg_mean": float(np.mean([r.test_loglik_avg for r in per_run])),
                        "test_loglik_avg_std": float(np.std([r.test_loglik_avg for r in per_run])),
                        "aic_mean": float(np.mean([r.aic for r in per_run])),
                        "bic_mean": float(np.mean([r.bic for r in per_run])),
                        "entropy_train_mean": float(np.mean([r.entropy_train for r in per_run])),
                        "entropy_test_mean": float(np.mean([r.entropy_test for r in per_run])),
                        "switches_test_mean": float(np.mean([r.switches_test for r in per_run])),
                        "mean_self_transition": float(np.mean([r.mean_self_transition for r in per_run])),
                        "n_runs": n_runs,
                    }
                )

                if best_run and best_pack:
                    key = "k%d_cov%s_mix%d" % (best_run.k, best_run.cov_type, best_run.n_mix)
                    best_artifacts[key] = {
                        "run": best_run,
                        "model": best_pack["model"],
                        "post_train": best_pack["post_train"],
                        "post_test": best_pack["post_test"],
                        "viterbi_train": best_pack["viterbi_train"],
                        "viterbi_test": best_pack["viterbi_test"],
                    }

    return run_results, summary_rows, best_artifacts


def hmm_duration_diagnostics(transmat: np.ndarray, viterbi_states: np.ndarray) -> Dict[str, object]:
    expected = _expected_durations(transmat)
    runs: Dict[int, List[int]] = {}
    if viterbi_states.size > 0:
        current = int(viterbi_states[0])
        length = 1
        for s in viterbi_states[1:]:
            s = int(s)
            if s == current:
                length += 1
            else:
                runs.setdefault(current, []).append(length)
                current = s
                length = 1
        runs.setdefault(current, []).append(length)
    return {"expected_durations": expected, "run_lengths": runs}
