from __future__ import annotations

import itertools
import json
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DIRECTION_THRESHOLD,
    ENTROPY_MAX,
    FIGURES_DIR,
    FOLD_STEP_SIZE,
    GATE_THRESHOLD,
    GMM_COV_TYPE,
    GMM_KS,
    GMM_MAX_ITER,
    GMM_SEEDS,
    MAX_LEAVE_PROB,
    MIN_ACTION_RATE,
    MIN_DIR_TRAIN_SAMPLES,
    MIN_DURATION,
    MIN_TRAIN_SIZE,
    N_SPLITS,
    OUTPUTS_DIR,
    REPORT_PATH,
    SWEEP_DIRECTION_THRESHOLD,
    SWEEP_ENTROPY_MAX,
    SWEEP_GATE_THRESHOLD,
    SWEEP_TOPK_PERCENT,
    TEST_SIZE,
    CLIP_QUANTILES,
    USE_PCA,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
    FEATURE_SHIFT,
    FIXED_GATE_THRESHOLDS,
)
from data import build_base_features, drop_constant_columns, load_and_clean
from eligibility import compute_regime_stats, map_eligibility
from evaluation import (
    calibration_plot,
    compute_confusion_matrix,
    compute_trade_metrics,
    confusion_matrix_plot,
    distribution_plot,
    frontier_plot,
    pr_curve_plot,
    regime_timeline_plot,
    transition_matrix_plot,
)
from features import build_features
from models import (
    build_preprocessor,
    predict_proba_safe,
    train_direction_model,
    train_gate_model,
)
from policy import decide
from regimes_gmm import predict_gmm, select_gmm_model


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


def _apply_policy_batch(
    eligible_flags: np.ndarray,
    entropies: np.ndarray,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    gate_threshold: float,
    direction_threshold: float,
    entropy_max: float,
) -> Tuple[np.ndarray, List[str]]:
    decisions = []
    reasons = []
    for i in range(len(eligible_flags)):
        decision, reason = decide(
            eligible=bool(eligible_flags[i]),
            entropy=float(entropies[i]),
            entropy_max=entropy_max,
            p_trade=float(p_trade[i]) if np.isfinite(p_trade[i]) else None,
            gate_threshold=gate_threshold,
            p_long=float(p_long[i]) if np.isfinite(p_long[i]) else None,
            p_short=float(p_short[i]) if np.isfinite(p_short[i]) else None,
            direction_threshold=direction_threshold,
        )
        decisions.append(decision)
        reasons.append(reason)
    return np.array(decisions, dtype=object), reasons


def _gate_threshold_grid(p_trade_train: np.ndarray) -> List[float]:
    quantiles = np.percentile(p_trade_train, [80, 85, 90, 93, 95, 97])
    grid = sorted(set(np.concatenate([quantiles, np.array(FIXED_GATE_THRESHOLDS)])))
    return [float(x) for x in grid]


def _pareto_frontier(df: pd.DataFrame) -> pd.Series:
    pareto = []
    for idx, row in df.iterrows():
        dominates = df[
            (df["precision_dir"] >= row["precision_dir"]) & (df["coverage"] >= row["coverage"])
        ]
        if len(dominates) == 0:
            pareto.append(True)
            continue
        if len(dominates) == 1 and dominates.index[0] == idx:
            pareto.append(True)
            continue
        is_dominated = (
            (dominates["precision_dir"] > row["precision_dir"])
            | (dominates["coverage"] > row["coverage"])
        ).any()
        pareto.append(not is_dominated)
    return pd.Series(pareto, index=df.index)


def _transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    mat = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        mat[states[i], states[i + 1]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)
    return mat


def _evaluate_frontier(
    oos_df: pd.DataFrame,
    fold_thresholds: Dict[int, List[float]],
    fold_topk_thresholds: Dict[int, Dict[int, float]],
) -> pd.DataFrame:
    gate_grid = sorted(set(itertools.chain.from_iterable(fold_thresholds.values())))
    grid = list(itertools.product(gate_grid, SWEEP_DIRECTION_THRESHOLD, SWEEP_ENTROPY_MAX))
    rows = []
    for mode in ["gate_only", "gate_entropy", "full"]:
        for gate_thr, dir_thr, entropy_max in grid:
            decisions_all = []
            labels_all = []
            gate_tp = 0
            gate_fp = 0
            for fold_id, fold_df in oos_df.groupby("fold_id"):
                eligible = fold_df["eligible_flag"].values
                entropy = fold_df["entropy"].values
                p_trade = fold_df["p_trade"].values
                p_long = fold_df["p_long"].values
                p_short = fold_df["p_short"].values
                if mode == "gate_only":
                    eligible = np.ones_like(eligible, dtype=bool)
                    entropy = np.zeros_like(entropy)
                elif mode == "gate_entropy":
                    eligible = np.ones_like(eligible, dtype=bool)

                decisions, _ = _apply_policy_batch(
                    eligible_flags=eligible,
                    entropies=entropy,
                    p_trade=p_trade,
                    p_long=p_long,
                    p_short=p_short,
                    gate_threshold=gate_thr,
                    direction_threshold=dir_thr,
                    entropy_max=entropy_max,
                )
                decisions_all.append(decisions)
                labels_all.append(fold_df["label_norm"].values)

                gate_pred = (
                    eligible
                    & (entropy <= entropy_max)
                    & np.isfinite(p_trade)
                    & (p_trade >= gate_thr)
                )
                gate_true = fold_df["label_norm"].isin(["long", "short"]).values
                gate_tp += (gate_pred & gate_true).sum()
                gate_fp += (gate_pred & ~gate_true).sum()

            decisions_flat = np.concatenate(decisions_all)
            labels_flat = np.concatenate(labels_all)
            df_eval = pd.DataFrame({"decision": decisions_flat, "label_norm": labels_flat})
            metrics = compute_trade_metrics(df_eval, "decision", "label_norm")
            precision_gate = gate_tp / max(gate_tp + gate_fp, 1)
            rows.append(
                {
                    "mode": mode,
                    "policy": "threshold",
                    "gate_threshold": gate_thr,
                    "direction_threshold": dir_thr,
                    "entropy_max": entropy_max,
                    "coverage": metrics.coverage,
                    "precision_dir": metrics.precision_dir,
                    "precision_gate": precision_gate,
                    "precision_long": metrics.precision_long,
                    "precision_short": metrics.precision_short,
                    "trade_count": metrics.trade_count,
                }
            )

        for dir_thr, entropy_max in itertools.product(SWEEP_DIRECTION_THRESHOLD, SWEEP_ENTROPY_MAX):
            for topk in SWEEP_TOPK_PERCENT:
                decisions_all = []
                labels_all = []
                gate_tp = 0
                gate_fp = 0
                for fold_id, fold_df in oos_df.groupby("fold_id"):
                    eligible = fold_df["eligible_flag"].values
                    entropy = fold_df["entropy"].values
                    p_trade = fold_df["p_trade"].values
                    p_long = fold_df["p_long"].values
                    p_short = fold_df["p_short"].values
                    if mode == "gate_only":
                        eligible = np.ones_like(eligible, dtype=bool)
                        entropy = np.zeros_like(entropy)
                    elif mode == "gate_entropy":
                        eligible = np.ones_like(eligible, dtype=bool)

                    gate_thr = fold_topk_thresholds[fold_id][topk]
                    decisions, _ = _apply_policy_batch(
                        eligible_flags=eligible,
                        entropies=entropy,
                        p_trade=p_trade,
                        p_long=p_long,
                        p_short=p_short,
                        gate_threshold=gate_thr,
                        direction_threshold=dir_thr,
                        entropy_max=entropy_max,
                    )
                    decisions_all.append(decisions)
                    labels_all.append(fold_df["label_norm"].values)

                    gate_pred = (
                        eligible
                        & (entropy <= entropy_max)
                        & np.isfinite(p_trade)
                        & (p_trade >= gate_thr)
                    )
                    gate_true = fold_df["label_norm"].isin(["long", "short"]).values
                    gate_tp += (gate_pred & gate_true).sum()
                    gate_fp += (gate_pred & ~gate_true).sum()

                decisions_flat = np.concatenate(decisions_all)
                labels_flat = np.concatenate(labels_all)
                df_eval = pd.DataFrame({"decision": decisions_flat, "label_norm": labels_flat})
                metrics = compute_trade_metrics(df_eval, "decision", "label_norm")
                precision_gate = gate_tp / max(gate_tp + gate_fp, 1)
                rows.append(
                    {
                        "mode": mode,
                        "policy": "topk",
                        "top_k_percent": topk,
                        "direction_threshold": dir_thr,
                        "entropy_max": entropy_max,
                        "coverage": metrics.coverage,
                        "precision_dir": metrics.precision_dir,
                        "precision_gate": precision_gate,
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

    gmm_state_all = np.full(n_samples, np.nan)
    gmm_prob_all = np.full(n_samples, np.nan)
    gmm_entropy_all = np.full(n_samples, np.nan)

    decisions_all = np.array(["skip"] * n_samples, dtype=object)
    p_trade_all = np.full(n_samples, np.nan)
    p_long_all = np.full(n_samples, np.nan)
    p_short_all = np.full(n_samples, np.nan)
    eligible_flag_all = np.full(n_samples, False, dtype=object)
    eligible_reason_all = np.array([""] * n_samples, dtype=object)
    entropy_all = np.full(n_samples, np.nan)
    fold_id_all = np.full(n_samples, -1)

    gate_metrics_rows = []
    direction_metrics_rows = []
    fold_thresholds: Dict[int, List[float]] = {}
    fold_topk_thresholds: Dict[int, Dict[int, float]] = {}

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
        entropy_all[test_idx] = gmm_ent_test
        joblib.dump(
            gmm_result.model,
            ARTIFACTS_DIR / f"gmm_model_fold_{fold_id}.joblib",
        )

        df_train["regime_id"] = gmm_states_train
        stats = compute_regime_stats(df_train, state_col="regime_id", label_col="label_norm")
        eligibility = map_eligibility(
            stats,
            min_action_rate=MIN_ACTION_RATE,
            min_duration=MIN_DURATION,
            max_leave_prob=MAX_LEAVE_PROB,
        )
        eligible_set = eligibility.eligible_regimes

        y_gate_train = df_train["label_norm"].isin(["long", "short"]).astype(int).values
        gate_model = train_gate_model(X_train_t, y_gate_train)
        joblib.dump(
            gate_model.model,
            ARTIFACTS_DIR / f"gate_model_fold_{fold_id}.joblib",
        )
        gate_probs_test = predict_proba_safe(gate_model.model, X_test_t)
        gate_probs_train = predict_proba_safe(gate_model.model, X_train_t)
        if gate_probs_test is None:
            p_trade = np.full(len(test_idx), np.nan)
            p_trade_train = np.array([])
        else:
            classes = list(gate_model.model.classes_)
            idx = classes.index(1) if 1 in classes else 1
            p_trade = gate_probs_test[:, idx]
            p_trade_train = gate_probs_train[:, idx] if gate_probs_train is not None else np.array([])

        if len(p_trade_train) > 0:
            fold_thresholds[fold_id] = _gate_threshold_grid(p_trade_train)
            fold_topk_thresholds[fold_id] = {
                k: float(np.percentile(p_trade_train, 100 - k)) for k in SWEEP_TOPK_PERCENT
            }
        else:
            fold_thresholds[fold_id] = list(FIXED_GATE_THRESHOLDS)
            fold_topk_thresholds[fold_id] = {k: max(FIXED_GATE_THRESHOLDS) for k in SWEEP_TOPK_PERCENT}

        df_dir_train = df_train[
            (df_train["label_norm"].isin(["long", "short"]))
            & (df_train["regime_id"].isin(eligible_set))
        ]
        if len(df_dir_train) >= MIN_DIR_TRAIN_SAMPLES:
            y_dir_train = (df_dir_train["label_norm"] == "long").astype(int).values
            X_dir_train = X_train_t[df_dir_train["train_pos"].values]
            dir_model = train_direction_model(X_dir_train, y_dir_train)
            joblib.dump(
                dir_model.model,
                ARTIFACTS_DIR / f"direction_model_fold_{fold_id}.joblib",
            )
            dir_probs_test = predict_proba_safe(dir_model.model, X_test_t)
            if dir_probs_test is None:
                p_long = np.full(len(test_idx), np.nan)
                p_short = np.full(len(test_idx), np.nan)
            else:
                p_long = dir_probs_test[:, 1]
                p_short = dir_probs_test[:, 0]
        else:
            p_long = np.full(len(test_idx), np.nan)
            p_short = np.full(len(test_idx), np.nan)

        eligible_flag = np.array([rid in eligible_set for rid in gmm_states_test])
        decisions, reasons = _apply_policy_batch(
            eligible_flags=eligible_flag,
            entropies=gmm_ent_test,
            p_trade=p_trade,
            p_long=p_long,
            p_short=p_short,
            gate_threshold=GATE_THRESHOLD,
            direction_threshold=DIRECTION_THRESHOLD,
            entropy_max=ENTROPY_MAX,
        )

        decisions_all[test_idx] = decisions
        p_trade_all[test_idx] = p_trade
        p_long_all[test_idx] = p_long
        p_short_all[test_idx] = p_short
        eligible_flag_all[test_idx] = eligible_flag
        eligible_reason_all[test_idx] = reasons
        fold_id_all[test_idx] = fold_id

        gate_pred = (
            eligible_flag
            & (gmm_ent_test <= ENTROPY_MAX)
            & np.isfinite(p_trade)
            & (p_trade >= GATE_THRESHOLD)
        )
        gate_true = df_test["label_norm"].isin(["long", "short"]).values
        gate_tp = (gate_pred & gate_true).sum()
        gate_fp = (gate_pred & ~gate_true).sum()
        gate_fn = (~gate_pred & gate_true).sum()
        gate_precision = gate_tp / max(gate_tp + gate_fp, 1)
        gate_recall = gate_tp / max(gate_tp + gate_fn, 1)
        gate_coverage = gate_pred.mean()
        gate_metrics_rows.append(
            {
                "fold_id": fold_id,
                "precision_gate": gate_precision,
                "recall_gate": gate_recall,
                "coverage_gate": gate_coverage,
                "trades_gate": int(gate_pred.sum()),
            }
        )

        df_test_eval = df_test.copy()
        df_test_eval["decision"] = decisions
        dir_metrics = compute_trade_metrics(df_test_eval, "decision", "label_norm")
        direction_metrics_rows.append(
            {
                "fold_id": fold_id,
                "coverage": dir_metrics.coverage,
                "precision_dir": dir_metrics.precision_dir,
                "precision_long": dir_metrics.precision_long,
                "precision_short": dir_metrics.precision_short,
                "trades": dir_metrics.trade_count,
            }
        )

    eligible_reason_all[fold_id_all < 0] = "in_sample"

    regimes_gmm = pd.DataFrame(
        {
            "timestamp": df[timestamp_col],
            "regime_id": gmm_state_all,
            "p_regime_max": gmm_prob_all,
            "entropy": gmm_entropy_all,
        }
    )
    regimes_gmm.to_csv(OUTPUTS_DIR / "regimes_gmm.csv", index=False)

    signals = pd.DataFrame(
        {
            "timestamp": df[timestamp_col],
            "true_label": df[label_col],
            "decision": decisions_all,
            "p_trade": p_trade_all,
            "p_long": p_long_all,
            "p_short": p_short_all,
            "regime_id": gmm_state_all,
            "entropy": entropy_all,
            "eligible_reason": eligible_reason_all,
        }
    )
    signals.to_csv(OUTPUTS_DIR / "final_signals.csv", index=False)

    gate_metrics_df = pd.DataFrame(gate_metrics_rows)
    gate_metrics_df.to_csv(OUTPUTS_DIR / "gate_metrics.csv", index=False)

    direction_metrics_df = pd.DataFrame(direction_metrics_rows)
    direction_metrics_df.to_csv(OUTPUTS_DIR / "direction_metrics.csv", index=False)

    oos_mask = fold_id_all >= 0
    oos_df = df[oos_mask].copy()
    oos_df["label_norm"] = df.loc[oos_mask, "label_norm"].values
    oos_df["decision"] = decisions_all[oos_mask]
    oos_df["p_trade"] = p_trade_all[oos_mask]
    oos_df["p_long"] = p_long_all[oos_mask]
    oos_df["p_short"] = p_short_all[oos_mask]
    oos_df["entropy"] = entropy_all[oos_mask]
    oos_df["eligible_flag"] = eligible_flag_all[oos_mask]
    oos_df["fold_id"] = fold_id_all[oos_mask]

    frontier = _evaluate_frontier(oos_df, fold_thresholds, fold_topk_thresholds)
    frontier.to_csv(OUTPUTS_DIR / "frontier_full.csv", index=False)
    pareto = frontier[frontier["pareto"]].copy()
    pareto.to_csv(OUTPUTS_DIR / "frontier_pareto.csv", index=False)
    frontier_plot(frontier, "Precision vs Coverage Frontier", FIGURES_DIR / "frontier_plot.png")

    if not np.isnan(gmm_state_all).all():
        timeline_df = regimes_gmm.dropna()
        if len(timeline_df) > 0:
            sample_df = timeline_df.tail(min(len(timeline_df), 4000))
            regime_timeline_plot(
                sample_df["timestamp"],
                sample_df["regime_id"],
                "GMM Regime Timeline (OOS)",
                FIGURES_DIR / "regime_timeline.png",
            )
        valid_states = regimes_gmm["regime_id"].dropna().astype(int).values
        if len(valid_states) > 1:
            n_states = int(valid_states.max()) + 1
            trans = _transition_matrix(valid_states, n_states)
            transition_matrix_plot(
                trans,
                "GMM Transition Matrix (OOS)",
                FIGURES_DIR / "gmm_transition_matrix.png",
            )

    gate_true_all = oos_df["label_norm"].isin(["long", "short"]).astype(int).values
    gate_pred_all = (
        oos_df["eligible_flag"]
        & (oos_df["entropy"] <= ENTROPY_MAX)
        & np.isfinite(oos_df["p_trade"])
        & (oos_df["p_trade"] >= GATE_THRESHOLD)
    ).astype(int)
    cm_gate = compute_confusion_matrix(gate_true_all, gate_pred_all, labels=[0, 1])
    confusion_matrix_plot(
        cm_gate,
        ["Skip", "Trade"],
        "Gate Confusion Matrix",
        FIGURES_DIR / "confusion_gate.png",
    )

    trade_mask = oos_df["decision"].isin(["long", "short"])
    if trade_mask.any():
        y_true_dir = oos_df.loc[trade_mask, "label_norm"]
        y_pred_dir = oos_df.loc[trade_mask, "decision"]
        valid = y_true_dir.isin(["long", "short"])
        y_true_bin = (y_true_dir[valid] == "long").astype(int).values
        y_pred_bin = (y_pred_dir[valid] == "long").astype(int).values
        cm_dir = compute_confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    else:
        cm_dir = np.zeros((2, 2), dtype=int)
    confusion_matrix_plot(
        cm_dir,
        ["Short", "Long"],
        "Direction Confusion Matrix",
        FIGURES_DIR / "confusion_direction.png",
    )

    ap_gate = pr_curve_plot(
        gate_true_all,
        np.nan_to_num(oos_df["p_trade"].values, nan=0.0),
        "Gate PR Curve",
        FIGURES_DIR / "gate_pr_curve.png",
    )

    direction_probs = np.nanmax(np.vstack([oos_df["p_long"].values, oos_df["p_short"].values]), axis=0)
    distribution_plot(
        np.nan_to_num(oos_df["p_trade"].values, nan=0.0),
        "Distribution of P(trade)",
        FIGURES_DIR / "p_trade_distribution.png",
    )
    distribution_plot(
        np.nan_to_num(direction_probs, nan=0.0),
        "Distribution of P(direction)",
        FIGURES_DIR / "p_direction_distribution.png",
    )

    summary_metrics = compute_trade_metrics(oos_df, "decision", "label_norm")
    debug_rows = []
    for mode in ["gate_only", "gate_entropy", "full"]:
        eligible = oos_df["eligible_flag"].values.copy()
        entropy = oos_df["entropy"].values.copy()
        if mode == "gate_only":
            eligible = np.ones_like(eligible, dtype=bool)
            entropy = np.zeros_like(entropy)
        elif mode == "gate_entropy":
            eligible = np.ones_like(eligible, dtype=bool)
        decisions_dbg, _ = _apply_policy_batch(
            eligible_flags=eligible,
            entropies=entropy,
            p_trade=oos_df["p_trade"].values,
            p_long=oos_df["p_long"].values,
            p_short=oos_df["p_short"].values,
            gate_threshold=GATE_THRESHOLD,
            direction_threshold=DIRECTION_THRESHOLD,
            entropy_max=ENTROPY_MAX,
        )
        df_dbg = pd.DataFrame(
            {"decision": decisions_dbg, "label_norm": oos_df["label_norm"].values}
        )
        metrics_dbg = compute_trade_metrics(df_dbg, "decision", "label_norm")
        gate_pred = (
            eligible
            & (entropy <= ENTROPY_MAX)
            & np.isfinite(oos_df["p_trade"].values)
            & (oos_df["p_trade"].values >= GATE_THRESHOLD)
        )
        gate_true = oos_df["label_norm"].isin(["long", "short"]).values
        gate_tp = (gate_pred & gate_true).sum()
        gate_fp = (gate_pred & ~gate_true).sum()
        precision_gate = gate_tp / max(gate_tp + gate_fp, 1)
        debug_rows.append(
            {
                "mode": mode,
                "coverage": metrics_dbg.coverage,
                "precision_dir": metrics_dbg.precision_dir,
                "precision_gate": precision_gate,
                "trade_count": metrics_dbg.trade_count,
            }
        )
    mask_brier = np.isfinite(oos_df["p_trade"].values)
    if mask_brier.any():
        from sklearn.metrics import brier_score_loss

        brier_gate = brier_score_loss(gate_true_all[mask_brier], oos_df["p_trade"].values[mask_brier])
    else:
        brier_gate = float("nan")
    p_trade_vals = oos_df["p_trade"].values if "p_trade" in oos_df else np.array([])
    p_trade_vals = p_trade_vals[np.isfinite(p_trade_vals)]
    p_trade_stats = pd.Series(p_trade_vals).describe() if len(p_trade_vals) else None
    p_dir_stats = pd.Series(direction_probs).describe() if len(direction_probs) else None

    report_lines = []
    report_lines.append("# Regime-First Pipeline Report (12h, v4)")
    report_lines.append("")
    report_lines.append("## Data")
    report_lines.append(f"- Rows used: {len(df)}")
    report_lines.append(f"- Features used: {len(feature_cols)}")
    report_lines.append(f"- Label column: {label_col}")
    report_lines.append(f"- Excluded columns: {', '.join(excluded_cols)}")
    report_lines.append("")
    report_lines.append("## Gate Metrics (OOS)")
    for row in gate_metrics_rows:
        report_lines.append(
            f"- Fold {row['fold_id']}: precision={row['precision_gate']:.3f}, "
            f"recall={row['recall_gate']:.3f}, coverage={row['coverage_gate']:.3f}, "
            f"trades={row['trades_gate']}"
        )
    report_lines.append("")
    report_lines.append("## Direction Metrics (OOS)")
    for row in direction_metrics_rows:
        report_lines.append(
            f"- Fold {row['fold_id']}: precision={row['precision_dir']:.3f}, "
            f"coverage={row['coverage']:.3f}, trades={row['trades']}"
        )
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- Coverage @ gate={GATE_THRESHOLD}, dir={DIRECTION_THRESHOLD}: {summary_metrics.coverage:.3f}")
    report_lines.append(f"- Precision (direction): {summary_metrics.precision_dir:.3f}")
    report_lines.append(f"- Gate AP (PR curve): {ap_gate:.3f}")
    report_lines.append(f"- Gate Brier score: {brier_gate:.4f}")
    if p_trade_stats is not None:
        report_lines.append(
            "- P(trade) stats: "
            f"min={p_trade_stats['min']:.3f}, p50={p_trade_stats['50%']:.3f}, max={p_trade_stats['max']:.3f}"
        )
    if p_dir_stats is not None:
        report_lines.append(
            "- P(direction) stats: "
            f"min={p_dir_stats['min']:.3f}, p50={p_dir_stats['50%']:.3f}, max={p_dir_stats['max']:.3f}"
        )
    report_lines.append("")
    report_lines.append("## Debug Modes (Gate Impact)")
    for row in debug_rows:
        report_lines.append(
            f"- {row['mode']}: coverage={row['coverage']:.3f}, "
            f"precision_dir={row['precision_dir']:.3f}, "
            f"precision_gate={row['precision_gate']:.3f}, "
            f"trades={row['trade_count']}"
        )
    report_lines.append("")
    report_lines.append("## Frontier (Pareto)")
    for mode in ["gate_only", "full"]:
        pareto_rows = frontier[(frontier["pareto"]) & (frontier["mode"] == mode)].sort_values(
            ["precision_dir", "coverage"], ascending=False
        )
        report_lines.append(f"- Mode {mode}:")
        if pareto_rows.empty:
            report_lines.append("  - no non-zero coverage points")
        else:
            for _, row in pareto_rows.head(5).iterrows():
                if row.get("policy") == "topk":
                    report_lines.append(
                        f"  - cov={row['coverage']:.3f}, prec_dir={row['precision_dir']:.3f}, "
                        f"prec_gate={row['precision_gate']:.3f}, top_k={row['top_k_percent']}%, "
                        f"dir_thr={row['direction_threshold']}, entropy_max={row['entropy_max']}"
                    )
                else:
                    report_lines.append(
                        f"  - cov={row['coverage']:.3f}, prec_dir={row['precision_dir']:.3f}, "
                        f"prec_gate={row['precision_gate']:.3f}, gate_thr={row['gate_threshold']:.3f}, "
                        f"dir_thr={row['direction_threshold']}, entropy_max={row['entropy_max']}"
                    )
    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("- Direction thresholds are intentionally lower because regime purity is near coin-flip.")
    report_lines.append("- Gate model drives precision by filtering out skip-heavy periods.")
    report_lines.append(
        "- If max P(trade) is below the gate threshold grid, coverage will be zero."
    )
    report_lines.append("- Dynamic gate thresholds now use per-fold P(trade) quantiles and top-k modes.")
    report_lines.append(
        "- Compressed gate scores came from calibration on imbalanced labels; isotonic expands range but still limited."
    )
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("- Use frontier_full.csv + frontier_pareto.csv to choose gate/dir thresholds.")
    report_lines.append("- Consider regime-specific gate thresholds if trade coverage remains low.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Gate metrics saved to:", OUTPUTS_DIR / "gate_metrics.csv")
    print("Direction metrics saved to:", OUTPUTS_DIR / "direction_metrics.csv")
    print("Frontier saved to:", OUTPUTS_DIR / "frontier_full.csv")
    print("Pareto saved to:", OUTPUTS_DIR / "frontier_pareto.csv")
    print("Outputs saved to:", OUTPUTS_DIR)


if __name__ == "__main__":
    run()
