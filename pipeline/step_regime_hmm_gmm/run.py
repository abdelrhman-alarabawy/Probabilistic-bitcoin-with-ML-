from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from src.data import load_data
from src.eval_hmm import occupancy_balance
from src.eval_state_gmms import state_component_posteriors, state_gmm_metrics
from src.features import build_feature_sets
from src.model_gmm import run_gmm_grid
from src.model_hmm_gmm import hmm_duration_diagnostics, run_hmm_grid
from src.plots import (
    plot_duration_hist,
    plot_state_posterior_heatmap,
    plot_state_probabilities,
    plot_viterbi_states,
)
from src.splits import build_walkforward_folds, period_info, split_by_years
from src.utils import ensure_dir, set_seed


def _preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    prep_cfg = cfg.get("preprocess", {})
    missing = prep_cfg.get("missing", "median")
    scale = prep_cfg.get("scale", "standard")

    train_df = train_df.copy()
    test_df = test_df.copy()

    if missing == "drop":
        train_df = train_df.dropna(subset=feature_cols)
        test_df = test_df.dropna(subset=feature_cols)
    else:
        strategy = "median" if missing not in {"mean", "median"} else missing
        imputer = SimpleImputer(strategy=strategy)
        train_df[feature_cols] = imputer.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = imputer.transform(test_df[feature_cols])

    if scale == "standard":
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty after preprocessing.")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    return train_df, test_df, X_train, X_test


def _write_state_posteriors(
    timestamps: pd.Series, posteriors: np.ndarray, path: Path, period: Dict[str, str]
) -> None:
    cols = {"timestamp": timestamps.values}
    for k in range(posteriors.shape[1]):
        cols["state_%d" % k] = posteriors[:, k]
    df = pd.DataFrame(cols)
    for key, value in period.items():
        df[key] = value
    df.to_csv(path, index=False)


def _write_viterbi_states(
    timestamps: pd.Series, states: np.ndarray, path: Path, period: Dict[str, str]
) -> None:
    df = pd.DataFrame({"timestamp": timestamps.values, "state": states})
    for key, value in period.items():
        df[key] = value
    df.to_csv(path, index=False)


def _write_transition_summary(
    path: Path,
    transmat: np.ndarray,
    expected_durations: np.ndarray,
    mean_self: float,
    stationary: str,
    startprob: np.ndarray,
) -> None:
    payload = {
        "mean_self_transition": mean_self,
        "stationary_dist": json.loads(stationary) if stationary else {},
        "expected_durations": expected_durations.tolist(),
        "startprob": startprob.tolist(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(config_path: Optional[str] = None) -> None:
    if config_path is None:
        default_yaml = Path(__file__).with_name("config.yaml")
        config_path = str(default_yaml) if default_yaml.exists() else None

    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

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

    output_cfg = cfg.get("output", {})
    results_dir = Path(output_cfg.get("dir", "pipeline/step_regime_hmm_gmm/results"))
    summary_dir = Path(output_cfg.get("summary_dir", results_dir / "summary"))
    ensure_dir(results_dir)
    ensure_dir(summary_dir)

    leaderboard_rows: List[Dict[str, object]] = []
    notes: List[str] = []

    for fold in folds:
        print("Fold %s: train=%d-%d test=%d" % (fold.name, fold.train_years[0], fold.train_years[-1], fold.test_years[0]))
        fold_dir = results_dir / ("fold_%s" % fold.name)
        ensure_dir(fold_dir)

        for feat_name, feat_cols in feature_sets.items():
            feature_dir = fold_dir / ("feature_%s" % feat_name)
            plots_dir = feature_dir / "plots"
            ensure_dir(feature_dir)
            ensure_dir(plots_dir)

            train_df, test_df = split_by_years(df, data_cfg["timestamp_col"], fold.train_years, fold.test_years)
            train_df, test_df, X_train, X_test = _preprocess(train_df, test_df, feat_cols, cfg)
            period = period_info(train_df, test_df, data_cfg["timestamp_col"])

            # ---- GMM baseline (IID) ----
            gmm_cfg = cfg["models"]["gmm"]
            gmm_runs, gmm_summaries = run_gmm_grid(
                X_train,
                X_test,
                gmm_cfg["components"],
                gmm_cfg["cov_types"],
                gmm_cfg["n_runs"],
                gmm_cfg["n_init"],
                gmm_cfg["max_iter"],
                cfg["preprocess"]["max_samples_metrics"],
                seed,
            )
            gmm_runs_df = pd.DataFrame([r.__dict__ for r in gmm_runs])
            gmm_summary_df = pd.DataFrame(gmm_summaries)
            if not gmm_runs_df.empty and "resp_entropy_mean" in gmm_runs_df.columns:
                gmm_runs_df["resp_entropy"] = gmm_runs_df["resp_entropy_mean"]
            if not gmm_runs_df.empty:
                for key, value in period.items():
                    gmm_runs_df[key] = value
            if not gmm_summary_df.empty:
                for key, value in period.items():
                    gmm_summary_df[key] = value
            if not gmm_runs_df.empty and not gmm_summary_df.empty and "ari_mean" in gmm_summary_df.columns:
                ari_map = (
                    gmm_summary_df.set_index(["k", "cov_type"])["ari_mean"].to_dict()
                )
                gmm_runs_df["stability_ari"] = gmm_runs_df.apply(
                    lambda row: float(ari_map.get((row["k"], row["cov_type"]), np.nan)), axis=1
                )
            gmm_runs_df.to_csv(feature_dir / "gmm_baseline_metrics.csv", index=False)
            gmm_summary_df.to_csv(feature_dir / "gmm_baseline_summary.csv", index=False)

            if not gmm_runs_df.empty:
                best_gmm = gmm_runs_df.sort_values("test_loglik_avg", ascending=False).iloc[0]
                gmm_model = GaussianMixture(
                    n_components=int(best_gmm["k"]),
                    covariance_type=str(best_gmm["cov_type"]),
                    random_state=int(best_gmm["seed"]),
                    n_init=gmm_cfg["n_init"],
                    max_iter=gmm_cfg["max_iter"],
                )
                gmm_model.fit(X_train)
                gmm_train_states = gmm_model.predict(X_train)
                gmm_test_states = gmm_model.predict(X_test)

                gmm_train_df = pd.DataFrame(
                    {
                        data_cfg["timestamp_col"]: train_df[data_cfg["timestamp_col"]].values,
                        "cluster": gmm_train_states,
                    }
                )
                gmm_test_df = pd.DataFrame(
                    {
                        data_cfg["timestamp_col"]: test_df[data_cfg["timestamp_col"]].values,
                        "cluster": gmm_test_states,
                    }
                )
                for key, value in period.items():
                    gmm_train_df[key] = value
                    gmm_test_df[key] = value

                gmm_train_df.to_csv(feature_dir / "gmm_states_train.csv", index=False)
                gmm_test_df.to_csv(feature_dir / "gmm_states_test.csv", index=False)
            if not gmm_runs_df.empty:
                for _, row in gmm_runs_df.iterrows():
                    leaderboard_rows.append(
                        {
                            "model": "gmm_baseline",
                            "fold": fold.name,
                            "feature_set": feat_name,
                            "k": int(row["k"]),
                            "cov_type": str(row["cov_type"]),
                            "n_mix": 1,
                            "seed": int(row["seed"]),
                            "test_loglik_avg": float(row["test_loglik_avg"]),
                            "bic": float(row["train_bic"]),
                            "occupancy_balance": np.nan,
                            "switches_test": np.nan,
                        }
                    )

            # ---- HMM-GMM ----
            hmm_cfg = cfg["models"]["hmm"]
            hmm_runs, hmm_summaries, hmm_artifacts = run_hmm_grid(
                X_train,
                X_test,
                hmm_cfg["states"],
                hmm_cfg["cov_types"],
                hmm_cfg["n_mix"],
                hmm_cfg["n_runs"],
                seed,
                hmm_cfg,
            )
            hmm_runs_df = pd.DataFrame([r.__dict__ for r in hmm_runs])
            hmm_summary_df = pd.DataFrame(hmm_summaries)
            if not hmm_runs_df.empty:
                for key, value in period.items():
                    hmm_runs_df[key] = value
            if not hmm_summary_df.empty:
                for key, value in period.items():
                    hmm_summary_df[key] = value

            hmm_runs_df.to_csv(feature_dir / "hmm_metrics.csv", index=False)
            hmm_summary_df.to_csv(feature_dir / "hmm_summary.csv", index=False)

            # Save artifacts per config (best seed for each config)
            for key, pack in hmm_artifacts.items():
                model = pack["model"]
                post_train = pack["post_train"]
                post_test = pack["post_test"]
                viterbi_train = pack["viterbi_train"]
                viterbi_test = pack["viterbi_test"]

                transmat = model.transmat_
                startprob = model.startprob_
                expected = hmm_duration_diagnostics(transmat, viterbi_test)["expected_durations"]
                mean_self = float(np.mean(np.diag(transmat)))
                stationary = pack["run"].stationary_dist

                np.save(feature_dir / ("transition_matrix_%s.npy" % key), transmat)
                pd.DataFrame(transmat).to_csv(
                    feature_dir / ("transition_matrix_%s.csv" % key), index=False
                )
                pd.DataFrame(
                    {"state": np.arange(len(expected)), "expected_duration": expected}
                ).to_csv(feature_dir / ("expected_durations_%s.csv" % key), index=False)
                _write_transition_summary(
                    feature_dir / ("transition_summary_%s.json" % key),
                    transmat,
                    expected,
                    mean_self,
                    stationary,
                    startprob,
                )
                pd.DataFrame(
                    {"state": np.arange(len(startprob)), "startprob": startprob}
                ).to_csv(feature_dir / ("startprob_%s.csv" % key), index=False)

                if output_cfg.get("save_state_posteriors", True):
                    _write_state_posteriors(
                        train_df[data_cfg["timestamp_col"]],
                        post_train,
                        feature_dir / ("state_posteriors_train_%s.csv" % key),
                        period,
                    )
                    _write_state_posteriors(
                        test_df[data_cfg["timestamp_col"]],
                        post_test,
                        feature_dir / ("state_posteriors_%s.csv" % key),
                        period,
                    )

                _write_viterbi_states(
                    test_df[data_cfg["timestamp_col"]],
                    viterbi_test,
                    feature_dir / ("viterbi_states_%s.csv" % key),
                    period,
                )

                # GMM-inside-state diagnostics
                state_rows_train = state_gmm_metrics(
                    model, X_train, viterbi_train, model.covariance_type, "train"
                )
                state_rows_test = state_gmm_metrics(
                    model, X_test, viterbi_test, model.covariance_type, "test"
                )
                state_rows = state_rows_train + state_rows_test
                state_df = pd.DataFrame(state_rows)
                state_df.to_csv(feature_dir / ("state_gmm_metrics_%s.csv" % key), index=False)

                for state in range(model.n_components):
                    sub = state_df[state_df["state"] == state]
                    sub.to_csv(
                        feature_dir / ("state_%d_gmm_metrics_%s.csv" % (state, key)), index=False
                    )

                weights = model.weights_ if hasattr(model, "weights_") else np.ones((model.n_components, 1))
                weight_rows = []
                for s in range(weights.shape[0]):
                    for m in range(weights.shape[1]):
                        weight_rows.append({"state": s, "component": m, "weight": float(weights[s, m])})
                pd.DataFrame(weight_rows).to_csv(
                    feature_dir / ("state_mixture_weights_%s.csv" % key), index=False
                )

                if output_cfg.get("save_component_posteriors", False):
                    train_posts = state_component_posteriors(model, X_train, viterbi_train)
                    test_posts = state_component_posteriors(model, X_test, viterbi_test)
                    for state, (idx, resp) in train_posts.items():
                        cols = {"timestamp": train_df.iloc[idx][data_cfg["timestamp_col"]].values}
                        for m in range(resp.shape[1]):
                            cols["component_%d" % m] = resp[:, m]
                        df_resp = pd.DataFrame(cols)
                        df_resp.to_csv(
                            feature_dir / ("state_%d_component_posteriors_train_%s.csv" % (state, key)),
                            index=False,
                        )
                    for state, (idx, resp) in test_posts.items():
                        cols = {"timestamp": test_df.iloc[idx][data_cfg["timestamp_col"]].values}
                        for m in range(resp.shape[1]):
                            cols["component_%d" % m] = resp[:, m]
                        df_resp = pd.DataFrame(cols)
                        df_resp.to_csv(
                            feature_dir / ("state_%d_component_posteriors_test_%s.csv" % (state, key)),
                            index=False,
                        )

                # Only plot for the best-by-test LL config to avoid bloating
                # (recorded below)

            # Pick best config by test log-likelihood for plots + generic files
            if not hmm_runs_df.empty:
                best_run = hmm_runs_df.sort_values("test_loglik_avg", ascending=False).iloc[0]
                best_key = "k%d_cov%s_mix%d" % (int(best_run["k"]), best_run["cov_type"], int(best_run["n_mix"]))
                best_pack = hmm_artifacts.get(best_key)
                if best_pack:
                    model = best_pack["model"]
                    post_test = best_pack["post_test"]
                    viterbi_test = best_pack["viterbi_test"]
                    transmat = model.transmat_
                    startprob = model.startprob_

                    np.save(feature_dir / "transition_matrix.npy", transmat)
                    pd.DataFrame(transmat).to_csv(feature_dir / "transition_matrix.csv", index=False)
                    pd.DataFrame(
                        {"state": np.arange(len(startprob)), "startprob": startprob}
                    ).to_csv(feature_dir / "startprob.csv", index=False)

                    _write_state_posteriors(
                        test_df[data_cfg["timestamp_col"]],
                        post_test,
                        feature_dir / "state_posteriors.csv",
                        period,
                    )
                    _write_viterbi_states(
                        test_df[data_cfg["timestamp_col"]],
                        viterbi_test,
                        feature_dir / "viterbi_states.csv",
                        period,
                    )

                    diagnostics = hmm_duration_diagnostics(transmat, viterbi_test)
                    expected = diagnostics["expected_durations"]
                    run_lengths = diagnostics["run_lengths"]
                    pd.DataFrame(
                        {"state": np.arange(len(expected)), "expected_duration": expected}
                    ).to_csv(feature_dir / "expected_durations.csv", index=False)

                    if output_cfg.get("save_plots", True):
                        plot_state_probabilities(
                            test_df[data_cfg["timestamp_col"]],
                            post_test,
                            str(plots_dir / "state_prob_over_time.png"),
                        )
                        plot_viterbi_states(
                            test_df[data_cfg["timestamp_col"]],
                            viterbi_test,
                            str(plots_dir / "viterbi_states_over_time.png"),
                        )
                        plot_duration_hist(run_lengths, str(plots_dir / "duration_hist_per_state.png"))
                        plot_state_posterior_heatmap(
                            post_test, str(plots_dir / "state_posterior_heatmap.png")
                        )

                for _, row in hmm_runs_df.iterrows():
                    occ = {}
                    if "state_occupancy_test" in row:
                        try:
                            occ = json.loads(row["state_occupancy_test"])
                        except Exception:
                            occ = {}
                    leaderboard_rows.append(
                        {
                            "model": "hmm_gmm",
                            "fold": fold.name,
                            "feature_set": feat_name,
                            "k": int(row["k"]),
                            "cov_type": str(row["cov_type"]),
                            "n_mix": int(row["n_mix"]),
                            "seed": int(row["seed"]),
                            "test_loglik_avg": float(row["test_loglik_avg"]),
                            "bic": float(row["bic"]),
                            "occupancy_balance": occupancy_balance(occ),
                            "switches_test": float(row["switches_test"]),
                        }
                    )

            notes.append(
                "Fold %s feature %s: plots only generated for best test-LL config; per-config plots skipped to limit file volume." %
                (fold.name, feat_name)
            )

    notes.append(
        "Per-state mixture responsibilities are saved as summary stats in state_gmm_metrics; full component posterior matrices are not saved (save_component_posteriors=false)."
    )

    # Leaderboard
    leaderboard = pd.DataFrame(leaderboard_rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["test_loglik_avg", "bic", "occupancy_balance", "switches_test"],
            ascending=[False, True, False, True],
        )
        leaderboard.to_csv(summary_dir / "leaderboard.csv", index=False)

    if notes:
        with (summary_dir / "notes.md").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(notes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM-GMM regime pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml or config.py")
    args = parser.parse_args()
    main(args.config)
