from __future__ import annotations

import itertools
import json
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

from config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DECISION_THRESHOLD,
    ELIGIBILITY_PROB_THRESHOLD,
    ENTROPY_MAX,
    FIGURES_DIR,
    FOLD_STEP_SIZE,
    GMM_COV_TYPE,
    GMM_KS,
    GMM_MAX_ITER,
    GMM_SEEDS,
    HMM_COV_TYPE,
    HMM_KS,
    HMM_MAX_ITER,
    HMM_MIN_AVG_DURATION,
    HMM_MIN_STATE_FRAC,
    HMM_SEEDS,
    HMM_STABLE_FRACTION,
    HMM_VAL_RATIO,
    MAX_LEAVE_PROB,
    MIN_ACTION_RATE,
    MIN_ACTION_SAMPLES,
    MIN_DIR_TRAIN_SAMPLES,
    MIN_DURATION,
    MIN_MARGIN,
    MIN_PURITY_ACTIONABLE,
    MIN_TRAIN_SIZE,
    N_SPLITS,
    OUTPUTS_DIR,
    REPORT_PATH,
    SWEEP_ACTION_RATES,
    SWEEP_DECISION_THRESHOLD,
    SWEEP_ENTROPY_MAX,
    SWEEP_MARGIN,
    SWEEP_PURITY,
    TEST_SIZE,
    USE_ELIGIBILITY_CLASSIFIER,
    CLIP_QUANTILES,
    USE_PCA,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
    FEATURE_SHIFT,
)
from data import build_base_features, drop_constant_columns, load_and_clean
from features import build_features
from regimes_hmm import HMM_AVAILABLE, predict_hmm, select_hmm_model
from regimes_gmm import predict_gmm, select_gmm_model
from eligibility import compute_regime_stats, map_eligibility
from models import (
    build_preprocessor,
    predict_proba_safe,
    train_direction_model,
    train_eligibility_classifier,
)
from policy import decide
from evaluation import (
    calibration_plot,
    compute_confusion_matrix,
    compute_trade_metrics,
    confusion_matrix_plot,
    frontier_plot,
    label_distribution_plot,
    regime_timeline_plot,
    transition_matrix_plot,
)


def _resolve_size(value: float, total: int) -> int:
    if value < 1:
        return max(int(total * value), 1)
    return int(value)


def generate_walk_forward_folds(
    n_samples: int,
    n_splits: int,
    min_train_size: float,
    test_size: float,
    step_size: Optional[float] = None,
):
    min_train = _resolve_size(min_train_size, n_samples)
    test = _resolve_size(test_size, n_samples)
    step = _resolve_size(step_size if step_size is not None else test_size, n_samples)

    train_end = min_train
    fold_id = 0
    while fold_id < n_splits and (train_end + test) <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test)
        yield fold_id, train_idx, test_idx
        train_end += step
        fold_id += 1


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _build_features_df(
    df: pd.DataFrame, timestamp_col: str, label_col: str
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_features, _, excluded_cols = build_base_features(
        df, timestamp_col=timestamp_col, label_col=label_col, feature_shift=FEATURE_SHIFT
    )
    extra_features = build_features(df, feature_shift=FEATURE_SHIFT)
    features = pd.concat([base_features, extra_features], axis=1)
    features = drop_constant_columns(features)
    feature_cols = features.columns.tolist()
    return features, feature_cols, excluded_cols


def _evaluate_fold_direction_model(
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    eligible_set: set,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[object]]:
    df_dir_train = df_train[
        (df_train["regime_id"].isin(eligible_set)) & (df_train["label_norm"] != "skip")
    ]
    if len(df_dir_train) < MIN_DIR_TRAIN_SAMPLES:
        return (
            np.full(len(df_test), np.nan),
            np.full(len(df_test), np.nan),
            "insufficient_actionable_samples",
            None,
        )
    y_dir_train = (df_dir_train["label_norm"] == "long").astype(int).values
    X_dir_train = X_train_t[df_dir_train["train_pos"].values]
    dir_clf = train_direction_model(X_dir_train, y_dir_train)
    if dir_clf.model is None:
        return (
            np.full(len(df_test), np.nan),
            np.full(len(df_test), np.nan),
            "direction_model_unavailable",
            None,
        )
    dir_probs_test = predict_proba_safe(dir_clf.model, X_test_t)
    return dir_probs_test[:, 1], dir_probs_test[:, 0], "ok", dir_clf.model


def _apply_policy_batch(
    eligible_flags: np.ndarray,
    entropies: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    decision_threshold: float,
    entropy_max: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    decisions = []
    p_decisions = []
    reasons = []
    for i in range(len(eligible_flags)):
        pl = float(p_long[i]) if np.isfinite(p_long[i]) else None
        ps = float(p_short[i]) if np.isfinite(p_short[i]) else None
        decision, p_decision, reason = decide(
            eligible=bool(eligible_flags[i]),
            entropy=float(entropies[i]),
            entropy_max=entropy_max,
            p_long=pl,
            p_short=ps,
            decision_threshold=decision_threshold,
        )
        decisions.append(decision)
        p_decisions.append(p_decision if p_decision is not None else np.nan)
        reasons.append(reason)
    return np.array(decisions, dtype=object), np.array(p_decisions, dtype=float), reasons


def _pareto_frontier(df: pd.DataFrame) -> pd.Series:
    pareto = []
    for idx, row in df.iterrows():
        dominates = df[
            (df["precision"] >= row["precision"]) & (df["coverage"] >= row["coverage"])
        ]
        if len(dominates) == 0:
            pareto.append(True)
            continue
        if len(dominates) == 1 and dominates.index[0] == idx:
            pareto.append(True)
            continue
        is_dominated = ((dominates["precision"] > row["precision"]) | (dominates["coverage"] > row["coverage"])).any()
        pareto.append(not is_dominated)
    return pd.Series(pareto, index=df.index)


def _evaluate_frontier(
    fold_cache: List[dict],
    stats_by_fold: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    grid = list(
        itertools.product(
            SWEEP_ACTION_RATES,
            SWEEP_PURITY,
            SWEEP_MARGIN,
            SWEEP_ENTROPY_MAX,
            SWEEP_DECISION_THRESHOLD,
        )
    )
    rows = []
    for action_rate, purity, margin, entropy_max, decision_threshold in grid:
        decisions_all = []
        labels_all = []
        fold_ids = []
        p_long_all = []
        p_short_all = []
        for entry in fold_cache:
            fold_id = entry["fold_id"]
            stats = stats_by_fold[fold_id]
            eligible_regimes = set(
                stats[
                    (stats["n_actionable"] >= MIN_ACTION_SAMPLES)
                    & (stats["action_rate"] >= action_rate)
                    & (
                        (stats["directional_purity_actionable"] >= purity)
                        | (stats["directional_margin"] >= margin)
                    )
                    & (stats["avg_duration"] >= MIN_DURATION)
                    & (stats["leave_prob"] <= MAX_LEAVE_PROB)
                ]["regime_id"]
            )
            eligible_flag = np.array(
                [rid in eligible_regimes for rid in entry["states_test"]]
            )
            p_long, p_short, _, _ = _evaluate_fold_direction_model(
                entry["X_train_t"],
                entry["X_test_t"],
                entry["df_train"],
                entry["df_test"],
                eligible_regimes,
            )
            decisions, _, _ = _apply_policy_batch(
                eligible_flag,
                entry["entropy_test"],
                p_long,
                p_short,
                decision_threshold,
                entropy_max,
            )
            decisions_all.append(decisions)
            labels_all.append(entry["df_test"]["label_norm"].values)
            fold_ids.append(np.full(len(decisions), fold_id))
            p_long_all.append(p_long)
            p_short_all.append(p_short)

        decisions_flat = np.concatenate(decisions_all)
        labels_flat = np.concatenate(labels_all)
        df_eval = pd.DataFrame(
            {
                "decision": decisions_flat,
                "label_norm": labels_flat,
            }
        )
        metrics = compute_trade_metrics(df_eval, "decision", "label_norm")
        rows.append(
            {
                "min_action_rate": action_rate,
                "min_purity_actionable": purity,
                "min_margin": margin,
                "entropy_max": entropy_max,
                "decision_threshold": decision_threshold,
                "coverage": metrics.coverage,
                "precision": metrics.precision,
                "precision_long": metrics.precision_long,
                "precision_short": metrics.precision_short,
                "trade_count": metrics.trade_count,
            }
        )
    frontier = pd.DataFrame(rows)
    frontier["pareto"] = _pareto_frontier(frontier)
    return frontier


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df, timestamp_col, label_col = load_and_clean(DATA_PATH)
    df["label_norm"] = df[label_col].astype(str).str.lower()

    features, feature_cols, excluded_cols = _build_features_df(df, timestamp_col, label_col)
    df = df.loc[features.index].copy()
    valid_mask = features.notna().all(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    features = features[valid_mask].reset_index(drop=True)
    feature_cols = features.columns.tolist()

    refreshed = df[[timestamp_col, label_col]].copy()
    if "label_ambiguous" in df.columns:
        refreshed["label_ambiguous"] = df["label_ambiguous"]
    refreshed = pd.concat([refreshed, features], axis=1)
    refreshed.to_csv(OUTPUTS_DIR / "features_refreshed.csv", index=False)

    with open(ARTIFACTS_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    X_all = features.values
    n_samples = len(df)

    hmm_state_all = np.full(n_samples, np.nan)
    hmm_prob_all = np.full(n_samples, np.nan)
    hmm_entropy_all = np.full(n_samples, np.nan)
    gmm_state_all = np.full(n_samples, np.nan)
    gmm_prob_all = np.full(n_samples, np.nan)
    gmm_entropy_all = np.full(n_samples, np.nan)

    decisions_all = np.array(["skip"] * n_samples, dtype=object)
    p_long_all = np.full(n_samples, np.nan)
    p_short_all = np.full(n_samples, np.nan)
    p_decision_all = np.full(n_samples, np.nan)
    eligible_flag_all = np.full(n_samples, False, dtype=object)
    eligible_reason_all = np.array([""] * n_samples, dtype=object)
    regime_primary_all = np.full(n_samples, np.nan)
    entropy_all = np.full(n_samples, np.nan)
    fold_id_all = np.full(n_samples, -1)

    fold_metrics = []
    label_stats_rows = []
    gmm_label_stats_rows = []
    stats_by_fold: Dict[int, pd.DataFrame] = {}
    fold_cache: List[dict] = []
    label_distribution_summary = []

    for fold_id, train_idx, test_idx in generate_walk_forward_folds(
        n_samples, N_SPLITS, MIN_TRAIN_SIZE, TEST_SIZE, FOLD_STEP_SIZE
    ):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        df_train = df.iloc[train_idx].copy()
        df_train["train_pos"] = np.arange(len(train_idx))
        df_test = df.iloc[test_idx].copy()

        preprocessor = build_preprocessor(CLIP_QUANTILES, USE_PCA, PCA_N_COMPONENTS)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        joblib.dump(
            {"preprocessor": preprocessor, "feature_columns": feature_cols},
            ARTIFACTS_DIR / f"preprocessor_fold_{fold_id}.joblib",
        )

        hmm_result = None
        if HMM_AVAILABLE:
            hmm_result = select_hmm_model(
                X_train_t,
                HMM_KS,
                HMM_SEEDS,
                HMM_MAX_ITER,
                HMM_COV_TYPE,
                HMM_VAL_RATIO,
                HMM_MIN_STATE_FRAC,
                HMM_MIN_AVG_DURATION,
                HMM_STABLE_FRACTION,
            )
            hmm_states_train, hmm_post_train, hmm_ent_train = predict_hmm(
                hmm_result.model, X_train_t
            )
            hmm_states_test, hmm_post_test, hmm_ent_test = predict_hmm(
                hmm_result.model, X_test_t
            )
            hmm_state_all[test_idx] = hmm_states_test
            hmm_prob_all[test_idx] = hmm_post_test.max(axis=1)
            hmm_entropy_all[test_idx] = hmm_ent_test
            joblib.dump(
                hmm_result.model,
                ARTIFACTS_DIR / f"hmm_model_fold_{fold_id}.joblib",
            )
        else:
            warnings.warn("hmmlearn missing: skipping HMM regimes.")

        gmm_result = select_gmm_model(X_train_t, GMM_KS, GMM_SEEDS, GMM_MAX_ITER, GMM_COV_TYPE)
        gmm_states_train, gmm_post_train, gmm_ent_train = predict_gmm(
            gmm_result.model, X_train_t
        )
        gmm_states_test, gmm_post_test, gmm_ent_test = predict_gmm(
            gmm_result.model, X_test_t
        )
        gmm_state_all[test_idx] = gmm_states_test
        gmm_prob_all[test_idx] = gmm_post_test.max(axis=1)
        gmm_entropy_all[test_idx] = gmm_ent_test
        joblib.dump(
            gmm_result.model,
            ARTIFACTS_DIR / f"gmm_model_fold_{fold_id}.joblib",
        )

        if hmm_result is not None:
            primary_states_train = hmm_states_train
            primary_states_test = hmm_states_test
            entropy_train = hmm_ent_train
            entropy_test = hmm_ent_test
        else:
            primary_states_train = gmm_states_train
            primary_states_test = gmm_states_test
            entropy_train = gmm_ent_train
            entropy_test = gmm_ent_test

        df_train["regime_id"] = primary_states_train
        df_train["regime_entropy"] = entropy_train
        stats = compute_regime_stats(
            df_train,
            state_col="regime_id",
            label_col="label_norm",
            entropy_col="regime_entropy",
        )
        eligibility = map_eligibility(
            stats,
            min_action_samples=MIN_ACTION_SAMPLES,
            min_action_rate=MIN_ACTION_RATE,
            min_purity_actionable=MIN_PURITY_ACTIONABLE,
            min_margin=MIN_MARGIN,
            min_duration=MIN_DURATION,
            max_leave_prob=MAX_LEAVE_PROB,
        )
        stats = eligibility.stats
        stats["fold_id"] = fold_id
        stats["model"] = "hmm" if hmm_result is not None else "gmm"
        label_stats_rows.append(stats)
        stats_by_fold[fold_id] = stats
        eligible_set = eligibility.eligible_regimes

        df_train_gmm = df_train.copy()
        df_train_gmm["regime_id"] = gmm_states_train
        df_train_gmm["regime_entropy"] = gmm_ent_train
        gmm_stats = compute_regime_stats(
            df_train_gmm,
            state_col="regime_id",
            label_col="label_norm",
            entropy_col="regime_entropy",
        )
        gmm_stats["fold_id"] = fold_id
        gmm_stats["model"] = "gmm"
        gmm_label_stats_rows.append(gmm_stats)

        if USE_ELIGIBILITY_CLASSIFIER:
            y_elig_train = df_train["regime_id"].apply(lambda r: 1 if r in eligible_set else 0).values
            elig_clf = train_eligibility_classifier(X_train_t, y_elig_train)
            if elig_clf.model is not None:
                joblib.dump(
                    elig_clf.model,
                    ARTIFACTS_DIR / f"eligibility_model_fold_{fold_id}.joblib",
                )
                elig_probs_test = predict_proba_safe(elig_clf.model, X_test_t)
                elig_prob = elig_probs_test[:, 1] if elig_probs_test is not None else None
            else:
                elig_prob = None
        else:
            elig_prob = None

        p_long, p_short, dir_status, dir_model = _evaluate_fold_direction_model(
            X_train_t, X_test_t, df_train, df_test, eligible_set
        )
        joblib.dump(
            {"status": dir_status, "model": dir_model},
            ARTIFACTS_DIR / f"direction_model_fold_{fold_id}.joblib",
        )

        eligible_flag = np.array([rid in eligible_set for rid in primary_states_test])
        if USE_ELIGIBILITY_CLASSIFIER and elig_prob is not None:
            eligible_flag = eligible_flag & (elig_prob >= ELIGIBILITY_PROB_THRESHOLD)

        decisions, p_decisions, reasons = _apply_policy_batch(
            eligible_flag,
            entropy_test,
            p_long,
            p_short,
            DECISION_THRESHOLD,
            ENTROPY_MAX,
        )

        decisions_all[test_idx] = decisions
        p_long_all[test_idx] = p_long
        p_short_all[test_idx] = p_short
        p_decision_all[test_idx] = p_decisions
        eligible_flag_all[test_idx] = eligible_flag
        eligible_reason_all[test_idx] = reasons
        regime_primary_all[test_idx] = primary_states_test
        entropy_all[test_idx] = entropy_test
        fold_id_all[test_idx] = fold_id

        fold_df = df_test.copy()
        fold_df["decision"] = decisions
        metrics = compute_trade_metrics(fold_df.assign(label_norm=df_test["label_norm"]), "decision", "label_norm")
        fold_metrics.append(
            {
                "fold_id": fold_id,
                "coverage": metrics.coverage,
                "precision": metrics.precision,
                "trade_count": metrics.trade_count,
                "precision_long": metrics.precision_long,
                "precision_short": metrics.precision_short,
                "eligible_regimes": len(eligible_set),
                "hmm_k": getattr(hmm_result, "n_states", None),
                "gmm_k": gmm_result.n_states,
            }
        )

        label_distribution_summary.append(
            {
                "fold_id": fold_id,
                "train_counts": df_train["label_norm"].value_counts().to_dict(),
                "test_counts": df_test["label_norm"].value_counts().to_dict(),
            }
        )

        fold_cache.append(
            {
                "fold_id": fold_id,
                "X_train_t": X_train_t,
                "X_test_t": X_test_t,
                "df_train": df_train,
                "df_test": df_test,
                "states_test": primary_states_test,
                "entropy_test": entropy_test,
            }
        )

    eligible_reason_all[fold_id_all < 0] = "in_sample"

    regimes_hmm = pd.DataFrame(
        {
            "timestamp": df[timestamp_col],
            "regime_id": hmm_state_all,
            "p_regime_max": hmm_prob_all,
            "entropy": hmm_entropy_all,
        }
    )
    regimes_gmm = pd.DataFrame(
        {
            "timestamp": df[timestamp_col],
            "regime_id": gmm_state_all,
            "p_regime_max": gmm_prob_all,
            "entropy": gmm_entropy_all,
        }
    )
    regimes_hmm.to_csv(OUTPUTS_DIR / "regimes_hmm.csv", index=False)
    regimes_gmm.to_csv(OUTPUTS_DIR / "regimes_gmm.csv", index=False)

    label_stats_df = pd.concat(label_stats_rows, ignore_index=True)
    label_stats_df.to_csv(OUTPUTS_DIR / "regime_label_stats_train.csv", index=False)

    gmm_label_stats_df = pd.concat(gmm_label_stats_rows, ignore_index=True)

    signals = pd.DataFrame(
        {
            "timestamp": df[timestamp_col],
            "true_label": df[label_col],
            "decision": decisions_all,
            "p_long": p_long_all,
            "p_short": p_short_all,
            "regime_id": regime_primary_all,
            "entropy": entropy_all,
            "eligible_reason": eligible_reason_all,
        }
    )
    signals.to_csv(OUTPUTS_DIR / "final_signals.csv", index=False)

    fold_summary = pd.DataFrame(fold_metrics)
    fold_summary.to_csv(OUTPUTS_DIR / "fold_metrics.csv", index=False)

    if HMM_AVAILABLE and not np.isnan(hmm_state_all).all():
        timeline_df = regimes_hmm.dropna()
        if len(timeline_df) > 0:
            sample_df = timeline_df.tail(min(len(timeline_df), 4000))
            regime_timeline_plot(
                sample_df["timestamp"],
                sample_df["regime_id"],
                "HMM Regime Timeline (OOS)",
                FIGURES_DIR / "regime_timeline.png",
            )
        last_fold = fold_summary["fold_id"].iloc[-1] if len(fold_summary) else 0
        hmm_model_path = ARTIFACTS_DIR / f"hmm_model_fold_{last_fold}.joblib"
        if hmm_model_path.exists():
            hmm_model = joblib.load(hmm_model_path)
            transition_matrix_plot(
                hmm_model.transmat_,
                "HMM Transition Matrix",
                FIGURES_DIR / "hmm_transition_matrix.png",
            )

    if not label_stats_df.empty:
        last_fold_id = label_stats_df["fold_id"].max()
        plot_stats = label_stats_df[label_stats_df["fold_id"] == last_fold_id]
        label_distribution_plot(
            plot_stats,
            "Label Distribution per Regime (Train)",
            FIGURES_DIR / "label_distribution_by_regime.png",
        )

    oos_mask = fold_id_all >= 0
    oos_df = df[oos_mask].copy()
    oos_df["decision"] = decisions_all[oos_mask]
    oos_df["p_long"] = p_long_all[oos_mask]
    oos_df["p_short"] = p_short_all[oos_mask]
    oos_df["eligible_flag"] = eligible_flag_all[oos_mask]

    trade_mask = oos_df["decision"].isin(["long", "short"])
    elig_true = (oos_df["label_norm"] != "skip").astype(int).values
    elig_pred = oos_df["eligible_flag"].astype(int).values
    cm_elig = compute_confusion_matrix(elig_true, elig_pred, labels=[0, 1])
    confusion_matrix_plot(
        cm_elig,
        ["Non-action", "Action"],
        "Eligibility vs Actionable Label",
        FIGURES_DIR / "confusion_eligibility.png",
    )
    if trade_mask.any():
        y_true_dir = oos_df.loc[trade_mask, "label_norm"]
        y_pred_dir = oos_df.loc[trade_mask, "decision"]
        valid = y_true_dir.isin(["long", "short"])
        cm_dir = compute_confusion_matrix(
            y_true_dir[valid].values, y_pred_dir[valid].values, labels=["short", "long"]
        )
    else:
        cm_dir = np.zeros((2, 2), dtype=int)
    confusion_matrix_plot(
        cm_dir,
        ["Short", "Long"],
        "Direction Confusion Matrix",
        FIGURES_DIR / "confusion_direction.png",
    )

    calib_mask = trade_mask & oos_df["p_long"].notna() & oos_df["label_norm"].isin(["long", "short"])
    if calib_mask.any():
        y_true = (oos_df.loc[calib_mask, "label_norm"] == "long").astype(int).values
        y_prob = oos_df.loc[calib_mask, "p_long"].values
        brier = brier_score_loss(y_true, y_prob)
    else:
        y_true = np.array([])
        y_prob = np.array([])
        brier = float("nan")
    calibration_plot(y_true, y_prob, "Calibration Curve (Long)", FIGURES_DIR / "calibration_curve.png")

    frontier = _evaluate_frontier(fold_cache, stats_by_fold)
    frontier.to_csv(OUTPUTS_DIR / "eligibility_frontier.csv", index=False)
    frontier_plot(frontier, "Precision vs Coverage Frontier", FIGURES_DIR / "frontier_plot.png")

    summary_metrics = compute_trade_metrics(oos_df, "decision", "label_norm")

    purity_values = label_stats_df["directional_purity_actionable"].values if len(label_stats_df) else np.array([])
    gmm_purity = gmm_label_stats_df["directional_purity_actionable"].values if len(gmm_label_stats_df) else np.array([])
    hmm_purity_by_fold = (
        label_stats_df.groupby("fold_id")["directional_purity_actionable"].mean().to_dict()
        if len(label_stats_df)
        else {}
    )
    gmm_purity_by_fold = (
        gmm_label_stats_df.groupby("fold_id")["directional_purity_actionable"].mean().to_dict()
        if len(gmm_label_stats_df)
        else {}
    )

    if len(purity_values) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(purity_values, bins=10, alpha=0.7, label="HMM")
        if len(gmm_purity) > 0:
            plt.hist(gmm_purity, bins=10, alpha=0.5, label="GMM")
        plt.title("Purity (Actionable) Distribution")
        plt.xlabel("Purity")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "purity_histogram.png", dpi=150)
        plt.close()

    report_lines = []
    report_lines.append("# Regime-First Pipeline Report (12h, v3)")
    report_lines.append("")
    report_lines.append("## Data")
    report_lines.append(f"- Rows used: {len(df)}")
    report_lines.append(f"- Features used: {len(feature_cols)}")
    report_lines.append(f"- Label column: {label_col}")
    report_lines.append(f"- Excluded columns: {', '.join(excluded_cols)}")
    report_lines.append("")
    report_lines.append("## Fold Metrics (OOS)")
    for fold in fold_metrics:
        report_lines.append(
            f"- Fold {fold['fold_id']}: coverage={fold['coverage']:.3f}, "
            f"precision={fold['precision']:.3f}, trades={fold['trade_count']}, "
            f"eligible_regimes={fold['eligible_regimes']}, "
            f"HMM_K={fold['hmm_k']}, GMM_K={fold['gmm_k']}"
        )
    report_lines.append("")
    report_lines.append("## Label Distribution by Fold")
    for entry in label_distribution_summary:
        report_lines.append(f"- Fold {entry['fold_id']} train: {entry['train_counts']}")
        report_lines.append(f"- Fold {entry['fold_id']} test: {entry['test_counts']}")
    report_lines.append("")
    if len(purity_values) > 0:
        report_lines.append("## Purity Diagnostics (HMM)")
        report_lines.append(
            f"- purity_actionable mean={np.mean(purity_values):.3f}, "
            f"p50={np.percentile(purity_values, 50):.3f}, "
            f"p90={np.percentile(purity_values, 90):.3f}, "
            f"max={np.max(purity_values):.3f}"
        )
    if len(gmm_purity) > 0:
        report_lines.append("## Purity Diagnostics (GMM)")
        report_lines.append(
            f"- purity_actionable mean={np.mean(gmm_purity):.3f}, "
            f"p50={np.percentile(gmm_purity, 50):.3f}, "
            f"p90={np.percentile(gmm_purity, 90):.3f}"
        )
    if hmm_purity_by_fold and gmm_purity_by_fold:
        report_lines.append("## HMM vs GMM Label Skew")
        for fold_id in sorted(hmm_purity_by_fold.keys()):
            hmm_val = hmm_purity_by_fold.get(fold_id, float("nan"))
            gmm_val = gmm_purity_by_fold.get(fold_id, float("nan"))
            better = "HMM" if hmm_val >= gmm_val else "GMM"
            report_lines.append(
                f"- Fold {fold_id}: HMM purity_mean={hmm_val:.3f}, "
                f"GMM purity_mean={gmm_val:.3f} -> {better}"
            )
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(
        f"- Coverage @ {DECISION_THRESHOLD}: {summary_metrics.coverage:.3f}"
    )
    report_lines.append(
        f"- Precision @ {DECISION_THRESHOLD}: {summary_metrics.precision:.3f}"
    )
    report_lines.append(f"- Brier score (long vs short): {brier:.4f}")
    report_lines.append("")
    report_lines.append("## Frontier (Pareto)")
    pareto_rows = frontier[frontier["pareto"]].sort_values(
        ["precision", "coverage"], ascending=False
    )
    for _, row in pareto_rows.head(10).iterrows():
        report_lines.append(
            f"- cov={row['coverage']:.3f}, prec={row['precision']:.3f}, "
            f"action_rate={row['min_action_rate']}, purity={row['min_purity_actionable']}, "
            f"margin={row['min_margin']}, entropy_max={row['entropy_max']}, "
            f"decision_thr={row['decision_threshold']}"
        )
    report_lines.append("")
    report_lines.append("## Interpretation")
    if len(purity_values) > 0 and np.percentile(purity_values, 90) < 0.55:
        report_lines.append(
            "- Regimes appear label-neutral; they likely capture volatility/risk states more than direction."
        )
    report_lines.append("- Labels are used as-is; skip is ignored for direction training.")
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("- Inspect label skew by regime and adjust features for directional separation.")
    report_lines.append("- Consider regime-specific thresholds or alternative state models if purity remains low.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Eligible regimes per fold:", [f["eligible_regimes"] for f in fold_metrics])
    print(
        f"Coverage @ {DECISION_THRESHOLD}: {summary_metrics.coverage:.3f} | "
        f"Precision @ {DECISION_THRESHOLD}: {summary_metrics.precision:.3f}"
    )
    top_regimes = (
        label_stats_df[label_stats_df["eligible"]]
        .sort_values(["directional_purity_actionable", "action_rate"], ascending=False)
        .head(5)
    )
    if len(top_regimes) > 0:
        print("Top eligible regimes (dominant direction + purity):")
        for _, row in top_regimes.iterrows():
            print(
                f"- Regime {int(row['regime_id'])}: {row['dominant_direction']} "
                f"(purity={row['directional_purity_actionable']:.2f}, "
                f"action_rate={row['action_rate']:.2f})"
            )
    print("Outputs saved to:", OUTPUTS_DIR)


if __name__ == "__main__":
    run()
