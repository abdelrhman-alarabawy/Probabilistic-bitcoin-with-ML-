from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.eval_state_gmms import state_component_posteriors

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
    force_gmmhmm_for_mix1 = bool(cfg.get("force_gmmhmm_for_mix1", False))
    if n_mix > 1 or (n_mix == 1 and force_gmmhmm_for_mix1):
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


def _regularize_covars(model, reg_covar: float) -> None:
    if reg_covar <= 0 or not hasattr(model, "covars_"):
        return
    covars = model.covars_
    dim = covars.shape[-1]
    eye = np.eye(dim) * reg_covar
    if covars.ndim == 4:
        for s in range(covars.shape[0]):
            for m in range(covars.shape[1]):
                covars[s, m] = covars[s, m] + eye
    elif covars.ndim == 3:
        for i in range(covars.shape[0]):
            covars[i] = covars[i] + eye
    elif covars.ndim == 2:
        covars = covars + eye
    model.covars_ = covars


def _covar_for_state_mix(model, state: int, mix: int = 0) -> Optional[np.ndarray]:
    if not hasattr(model, "covars_"):
        return None
    covars = model.covars_
    if covars.ndim == 4:
        return covars[state, mix]
    if covars.ndim == 3:
        if covars.shape[0] == model.n_components:
            return covars[state]
        if hasattr(model, "n_mix") and covars.shape[0] == model.n_mix:
            return covars[mix]
        return covars[0]
    if covars.ndim == 2:
        return covars
    return None


def _print_failure_diagnostics(
    model,
    X_train: np.ndarray,
    err_msg: str,
    reg_covar: float,
) -> None:
    n_eff = None
    min_mass = None
    try:
        post_train = model.predict_proba(X_train)
        n_eff = [float(np.sum(post_train[:, i])) for i in range(post_train.shape[1])]
        min_mass = float(np.min(n_eff)) if n_eff else None
    except Exception:
        pass

    iter_info = None
    if hasattr(model, "monitor_") and hasattr(model.monitor_, "iter"):
        iter_info = int(model.monitor_.iter)

    state_id = None
    mix_id = 0
    match = re.search(r"state #(\d+)(?:, mixture #(\d+))?", err_msg)
    if match:
        state_id = int(match.group(1))
        if match.group(2) is not None:
            mix_id = int(match.group(2))

    min_eig = None
    max_eig = None
    cond = None
    if state_id is not None:
        cov = _covar_for_state_mix(model, state_id, mix_id)
        if cov is not None:
            try:
                eigvals = np.linalg.eigvalsh(cov)
                min_eig = float(np.min(eigvals))
                max_eig = float(np.max(eigvals))
                cond = float(max_eig / max(min_eig, 1.0e-12))
            except Exception:
                pass

    print(
        "HMM_DIAG reg_covar=%s iter=%s n_eff=%s min_mass=%s min_eig=%s max_eig=%s cond=%s"
        % (
            "%0.0e" % reg_covar,
            "NA" if iter_info is None else str(iter_info),
            "NA" if n_eff is None else str([round(x, 3) for x in n_eff]),
            "NA" if min_mass is None else "%0.3f" % min_mass,
            "NA" if min_eig is None else "%0.3e" % min_eig,
            "NA" if max_eig is None else "%0.3e" % max_eig,
            "NA" if cond is None else "%0.3e" % cond,
        )
    )


def _component_labels_by_state(
    model,
    X: np.ndarray,
    viterbi_states: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    labels_by_state: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    posts = state_component_posteriors(model, X, viterbi_states)
    for state, (idx, resp) in posts.items():
        if resp.size == 0:
            continue
        labels = np.argmax(resp, axis=1)
        labels_by_state[int(state)] = (idx, labels)
    return labels_by_state


def _state_component_ari(
    labels_per_run: List[Dict[int, Tuple[np.ndarray, np.ndarray]]],
    n_states: int,
) -> Dict[int, Dict[str, float]]:
    state_stats: Dict[int, Dict[str, float]] = {}
    for state in range(n_states):
        state_runs = [run[state] for run in labels_per_run if state in run]
        if len(state_runs) < 2:
            state_stats[state] = {"ari_mean": np.nan, "ari_std": np.nan, "n_pairs": 0}
            continue
        ari_scores = []
        for i in range(len(state_runs)):
            idx_i, labels_i = state_runs[i]
            map_i = {int(idx): int(lbl) for idx, lbl in zip(idx_i, labels_i)}
            for j in range(i + 1, len(state_runs)):
                idx_j, labels_j = state_runs[j]
                map_j = {int(idx): int(lbl) for idx, lbl in zip(idx_j, labels_j)}
                common = np.intersect1d(idx_i, idx_j, assume_unique=False)
                if common.size < 3:
                    continue
                li = np.array([map_i[int(x)] for x in common])
                lj = np.array([map_j[int(x)] for x in common])
                if len(np.unique(li)) < 2 and len(np.unique(lj)) < 2:
                    continue
                ari_scores.append(adjusted_rand_score(li, lj))
        if ari_scores:
            state_stats[state] = {
                "ari_mean": float(np.mean(ari_scores)),
                "ari_std": float(np.std(ari_scores)),
                "n_pairs": int(len(ari_scores)),
            }
        else:
            state_stats[state] = {"ari_mean": np.nan, "ari_std": np.nan, "n_pairs": 0}
    return state_stats


def run_hmm_grid(
    X_train: np.ndarray,
    X_test: np.ndarray,
    states: List[int],
    cov_types: List[str],
    n_mix_list: List[int],
    n_runs: int,
    seed: int,
    cfg: dict,
) -> Tuple[
    List[HMMRunResult],
    List[Dict[str, float]],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[int, Dict[str, float]]],
]:
    rng = np.random.default_rng(seed)
    reg_list = cfg.get("reg_covar_list") or [cfg.get("reg_covar", 1.0e-6)]
    fixed_seeds = cfg.get("fixed_seeds")
    run_results: List[HMMRunResult] = []
    summary_rows: List[Dict[str, float]] = []
    best_artifacts: Dict[str, Dict[str, object]] = {}
    state_stability: Dict[str, Dict[int, Dict[str, float]]] = {}

    for k in states:
        for cov_type in cov_types:
            for n_mix in n_mix_list:
                if isinstance(fixed_seeds, list) and fixed_seeds:
                    seeds = np.array(fixed_seeds, dtype=int)
                else:
                    seeds = rng.integers(0, 1_000_000, size=n_runs)
                per_run: List[HMMRunResult] = []
                best_run = None
                best_ll = -np.inf
                best_pack = None
                per_run_labels: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = []

                for run_seed in seeds:
                    try:
                        model = _build_model(k, cov_type, n_mix, int(run_seed), cfg)
                        model.fit(X_train)
                        attempt = 0
                        success = False
                        last_exc = None
                        for attempt, reg_covar in enumerate(reg_list, start=1):
                            _regularize_covars(model, float(reg_covar))
                            try:
                                train_ll = float(model.score(X_train))
                                test_ll = float(model.score(X_test))
                                success = True
                                if attempt > 1:
                                    print(
                                        "HMM_RETRY_OK reg_covar=%s after_attempt=%d"
                                        % ("%0.0e" % float(reg_covar), attempt)
                                    )
                                break
                            except ValueError as exc:
                                last_exc = exc
                                if "not positive definite" in str(exc):
                                    _print_failure_diagnostics(
                                        model, X_train, str(exc), float(reg_covar)
                                    )
                                    continue
                                raise
                        if not success:
                            raise last_exc if last_exc is not None else RuntimeError("HMM run failed")
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
                        per_run_labels.append(
                            _component_labels_by_state(model, X_train, viterbi_train)
                        )
                    except Exception as exc:
                        print(
                            "HMM_RUN_FAILED k=%d cov=%s n_mix=%d seed=%d error=%s"
                            % (k, cov_type, n_mix, int(run_seed), repr(exc))
                        )
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
                    state_stability[key] = _state_component_ari(per_run_labels, k)

    return run_results, summary_rows, best_artifacts, state_stability


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
