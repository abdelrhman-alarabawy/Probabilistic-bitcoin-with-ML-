from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from run import _preprocess
from src.data import load_data
from src.features import build_feature_sets
from src.first_results_pack import print_first_results_pack
from src.model_gmm import GMMRunResult
from src.model_hmm_gmm import HMMRunResult, _build_model
from src.splits import build_walkforward_folds, period_info, split_by_years
from src.utils import ensure_dir, set_seed


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _find_earliest_fold(results_dir: Path) -> Optional[str]:
    fold_dirs = [p for p in results_dir.glob("fold_*") if p.is_dir()]
    if not fold_dirs:
        return None
    fold_dirs = sorted(fold_dirs, key=lambda p: p.name)
    name = fold_dirs[0].name
    return name.replace("fold_", "", 1)


def _responsibility_entropy(resp: np.ndarray) -> np.ndarray:
    eps = 1e-12
    resp_safe = np.clip(resp, eps, 1.0)
    return -np.sum(resp_safe * np.log(resp_safe), axis=1)


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
        from sklearn.metrics import silhouette_score

        sil = float(silhouette_score(X_eval, labels_eval))
    except Exception:
        sil = None
    try:
        from sklearn.metrics import davies_bouldin_score

        db = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception:
        db = None
    return sil, db


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


def _state_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probs = np.clip(posteriors, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _count_switches(states: np.ndarray) -> int:
    if states.size <= 1:
        return 0
    return int(np.sum(states[1:] != states[:-1]))


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


def _run_gmm_fixed_seeds(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    cov_types: List[str],
    seeds: List[int],
    n_init: int,
    max_iter: int,
    max_samples_metrics: int,
    metrics_seed: int,
) -> Tuple[List[GMMRunResult], List[Dict[str, float]]]:
    rng = np.random.default_rng(metrics_seed)
    run_results: List[GMMRunResult] = []
    summary_rows: List[Dict[str, float]] = []

    for cov_type in cov_types:
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
                n_runs=len(seeds),
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
                "train_loglik_avg_mean": float(
                    np.mean([r.train_loglik_avg for r in per_run])
                ),
                "train_loglik_avg_std": float(
                    np.std([r.train_loglik_avg for r in per_run])
                ),
                "test_loglik_avg_mean": float(
                    np.mean([r.test_loglik_avg for r in per_run])
                ),
                "test_loglik_avg_std": float(
                    np.std([r.test_loglik_avg for r in per_run])
                ),
                "train_aic_mean": float(np.mean([r.train_aic for r in per_run])),
                "train_aic_std": float(np.std([r.train_aic for r in per_run])),
                "train_bic_mean": float(np.mean([r.train_bic for r in per_run])),
                "train_bic_std": float(np.std([r.train_bic for r in per_run])),
                "silhouette_train_mean": float(
                    np.nanmean(
                        [
                            r.silhouette_train
                            if r.silhouette_train is not None
                            else np.nan
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
                    np.nanmean(
                        [r.db_train if r.db_train is not None else np.nan for r in per_run]
                    )
                ),
                "db_test_mean": float(
                    np.nanmean(
                        [r.db_test if r.db_test is not None else np.nan for r in per_run]
                    )
                ),
                "resp_entropy_mean": float(
                    np.mean([r.resp_entropy_mean for r in per_run])
                ),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "n_runs": len(seeds),
            }
        )

    return run_results, summary_rows


def _classify_warnings(warnings: List[str]) -> str:
    flags = set()
    for warning in warnings:
        if "Ghost state" in warning:
            flags.add("ghost")
        elif "Sticky state" in warning:
            flags.add("sticky")
        elif "Collapse risk" in warning:
            flags.add("collapse")
    if not flags:
        return "OK"
    if len(flags) == 1:
        return next(iter(flags))
    return "mixed"


def main(config_path: Optional[str] = None) -> None:
    if config_path is None:
        default_yaml = Path(__file__).with_name("config.yaml")
        config_path = str(default_yaml) if default_yaml.exists() else None

    cfg = load_config(config_path)

    seeds = [111, 222, 333]
    k = 2
    cov_types = ["full", "tied"]
    n_mix_list = [2, 3, 4]

    base_results_dir = Path(
        cfg.get("output", {}).get("dir", "pipeline/step_regime_hmm_gmm/results")
    )
    detected_fold = _find_earliest_fold(base_results_dir)

    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg.get("data", {})
    df = load_data(
        data_cfg["path"],
        data_cfg["timestamp_col"],
        drop_cols=data_cfg.get("drop_cols"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )

    feature_sets = build_feature_sets(df, cfg)
    folds = build_walkforward_folds(
        df,
        data_cfg["timestamp_col"],
        cfg["walkforward"]["min_train_years"],
        cfg["walkforward"]["test_years"],
        cfg["walkforward"].get("start_year"),
        cfg["walkforward"].get("end_year"),
    )

    if detected_fold:
        folds = [fold for fold in folds if fold.name == detected_fold] or [folds[0]]
    else:
        folds = [folds[0]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("pipeline/step_regime_hmm_gmm/results/grid_k2_m_sweep") / timestamp
    summary_dir = results_dir / "summary"
    ensure_dir(results_dir)
    ensure_dir(summary_dir)

    log_path = results_dir / "console_log.txt"
    summary_path = results_dir / "first_pack_summary.json"

    gmm_cfg = cfg["models"]["gmm"]
    hmm_cfg = cfg["models"]["hmm"]

    records: List[Dict[str, object]] = []
    warning_counts = {"OK": 0, "ghost": 0, "sticky": 0, "collapse": 0, "mixed": 0}

    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = _Tee(sys.stdout, log_handle)
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            for fold in folds:
                print(
                    "Fold %s: train=%d-%d test=%d"
                    % (
                        fold.name,
                        fold.train_years[0],
                        fold.train_years[-1],
                        fold.test_years[0],
                    )
                )
                fold_dir = results_dir / ("fold_%s" % fold.name)
                ensure_dir(fold_dir)

                for feat_name, feat_cols in feature_sets.items():
                    feature_dir = fold_dir / ("feature_%s" % feat_name)
                    ensure_dir(feature_dir)

                    train_df, test_df = split_by_years(
                        df, data_cfg["timestamp_col"], fold.train_years, fold.test_years
                    )
                    train_df, test_df, X_train, X_test = _preprocess(
                        train_df, test_df, feat_cols, cfg
                    )
                    period = period_info(train_df, test_df, data_cfg["timestamp_col"])

                    gmm_runs, gmm_summaries = _run_gmm_fixed_seeds(
                        X_train,
                        X_test,
                        k,
                        cov_types,
                        seeds,
                        gmm_cfg["n_init"],
                        gmm_cfg["max_iter"],
                        cfg["preprocess"]["max_samples_metrics"],
                        int(cfg.get("seed", 42)),
                    )
                    gmm_runs_df = pd.DataFrame([asdict(r) for r in gmm_runs])
                    gmm_summary_df = pd.DataFrame(gmm_summaries)

                    if not gmm_runs_df.empty and "resp_entropy_mean" in gmm_runs_df.columns:
                        gmm_runs_df["resp_entropy"] = gmm_runs_df["resp_entropy_mean"]
                    if not gmm_runs_df.empty:
                        for key, value in period.items():
                            gmm_runs_df[key] = value
                    if not gmm_summary_df.empty:
                        for key, value in period.items():
                            gmm_summary_df[key] = value
                    if not gmm_runs_df.empty and not gmm_summary_df.empty:
                        ari_map = (
                            gmm_summary_df.set_index(["k", "cov_type"])["ari_mean"].to_dict()
                        )
                        gmm_runs_df["stability_ari"] = gmm_runs_df.apply(
                            lambda row: float(
                                ari_map.get((row["k"], row["cov_type"]), np.nan)
                            ),
                            axis=1,
                        )
                    if not gmm_runs_df.empty:
                        gmm_runs_df["fold_id"] = fold.name
                        gmm_runs_df["feature_set"] = feat_name
                    if not gmm_summary_df.empty:
                        gmm_summary_df["fold_id"] = fold.name
                        gmm_summary_df["feature_set"] = feat_name
                    gmm_runs_df.to_csv(feature_dir / "gmm_baseline_metrics.csv", index=False)
                    gmm_summary_df.to_csv(
                        feature_dir / "gmm_baseline_summary.csv", index=False
                    )

                    for cov_type in cov_types:
                        for n_mix in n_mix_list:
                            for run_seed in seeds:
                                try:
                                    model = _build_model(k, cov_type, n_mix, int(run_seed), hmm_cfg)
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

                                    n_params = _param_count(
                                        k, X_train.shape[1], cov_type, n_mix
                                    )
                                    aic = 2 * n_params - 2 * train_ll
                                    bic = n_params * np.log(X_train.shape[0]) - 2 * train_ll

                                    occ_train = _occupancy(viterbi_train, k)
                                    occ_test = _occupancy(viterbi_test, k)
                                    stationary = _stationary_distribution(model.transmat_)

                                    hmm_run = HMMRunResult(
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

                                    run_context = {
                                        "fold_id": fold.name,
                                        "feature_set": feat_name,
                                        "period": period,
                                        "config": {
                                            "k": k,
                                            "n_mix": n_mix,
                                            "cov_type": cov_type,
                                            "seed": int(run_seed),
                                        },
                                        "data": {
                                            "n_train": int(X_train.shape[0]),
                                            "n_test": int(X_test.shape[0]),
                                            "n_features": int(X_train.shape[1]),
                                        },
                                        "gmm_runs_df": gmm_runs_df,
                                        "gmm_summary_df": gmm_summary_df,
                                        "hmm": {
                                            "model": model,
                                            "run": hmm_run,
                                            "post_train": post_train,
                                            "post_test": post_test,
                                            "viterbi_train": viterbi_train,
                                            "viterbi_test": viterbi_test,
                                        },
                                        "arrays": {"X_train": X_train, "X_test": X_test},
                                        "params": {
                                            "tau": 0.4,
                                            "max_samples_metrics": cfg["preprocess"][
                                                "max_samples_metrics"
                                            ],
                                            "seed": int(cfg.get("seed", 42)),
                                        },
                                        "output": {"summary_path": summary_path},
                                    }

                                    summary_record = print_first_results_pack(run_context)
                                    records.append(summary_record)
                                    warning_counts[
                                        _classify_warnings(summary_record.get("warnings", []))
                                    ] += 1
                                except Exception as exc:
                                    print(
                                        "CONFIG_FAILED fold=%s feature=%s k=%d mix=%d cov=%s seed=%d error=%s"
                                        % (
                                            fold.name,
                                            feat_name,
                                            k,
                                            n_mix,
                                            cov_type,
                                            int(run_seed),
                                            repr(exc),
                                        )
                                    )
                                    continue
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    total_configs = len(records)
    print("GRID_RUN_COMPLETE")
    print("configs_run=%d" % total_configs)
    print(
        "warnings_count OK=%d ghost=%d sticky=%d collapse=%d mixed=%d"
        % (
            warning_counts["OK"],
            warning_counts["ghost"],
            warning_counts["sticky"],
            warning_counts["collapse"],
            warning_counts["mixed"],
        )
    )

    def _safe_float(val: object) -> float:
        try:
            return float(val)
        except Exception:
            return float("nan")

    def _config_key(rec: Dict[str, object]) -> Tuple[int, int, str, int]:
        return (
            int(rec.get("k", 0)),
            int(rec.get("n_mix", 0)),
            str(rec.get("cov_type", "")),
            int(rec.get("seed", 0)),
        )

    by_hmm = sorted(
        records,
        key=lambda r: _safe_float(r.get("hmm_summary", {}).get("hmm_test_avg_loglik")),
        reverse=True,
    )
    by_delta = sorted(
        records,
        key=lambda r: _safe_float(r.get("hmm_summary", {}).get("hmm_test_avg_loglik"))
        - _safe_float(r.get("gmm_summary", {}).get("gmm_test_avg_loglik")),
        reverse=True,
    )

    print("TOP_5_HMM_TEST_LOGLIK")
    for rec in by_hmm[:5]:
        k_val, m_val, cov_val, seed_val = _config_key(rec)
        score = _safe_float(rec.get("hmm_summary", {}).get("hmm_test_avg_loglik"))
        print(
            "k=%d m=%d cov=%s seed=%d hmm_test_avg_loglik=%.6f"
            % (k_val, m_val, cov_val, seed_val, score)
        )

    print("TOP_5_DELTA_LOGLIK")
    for rec in by_delta[:5]:
        k_val, m_val, cov_val, seed_val = _config_key(rec)
        hmm_score = _safe_float(rec.get("hmm_summary", {}).get("hmm_test_avg_loglik"))
        gmm_score = _safe_float(rec.get("gmm_summary", {}).get("gmm_test_avg_loglik"))
        delta = hmm_score - gmm_score
        print(
            "k=%d m=%d cov=%s seed=%d delta_loglik=%.6f"
            % (k_val, m_val, cov_val, seed_val, delta)
        )

    print("SUMMARY_PATH %s" % str(summary_path))
    print("CONSOLE_LOG %s" % str(log_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K=2 HMM-GMM sweep.")
    parser.add_argument("--config", default=None, help="Path to config.yaml or config.py")
    args = parser.parse_args()
    main(args.config)
