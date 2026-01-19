from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Union
from sklearn.metrics import adjusted_rand_score

from config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DECISION_THRESHOLD,
    ELIGIBILITY_D_MIN,
    ELIGIBILITY_MIN_REGIME_FRAC,
    ELIGIBILITY_PROB_THRESHOLD,
    ELIGIBILITY_TAIL_LOSS_MAX,
    ELIGIBILITY_TRANSITION_RISK_MAX,
    ELIGIBILITY_WIN_RATE_MIN,
    FIGURES_DIR,
    FOLD_STEP_SIZE,
    GMM_COV_TYPE,
    GMM_ENTROPY_THRESHOLD,
    GMM_KS,
    GMM_MAX_ITER,
    GMM_SEEDS,
    HMM_COV_TYPE,
    HMM_ENTROPY_THRESHOLD,
    HMM_KS,
    HMM_MAX_ITER,
    HMM_MIN_AVG_DURATION,
    HMM_MIN_STATE_FRAC,
    HMM_SEEDS,
    HMM_VAL_RATIO,
    HORIZON_BARS,
    FEATURE_SHIFT,
    MIN_TRAIN_SIZE,
    N_SPLITS,
    OUTPUTS_DIR,
    REPORT_PATH,
    RETURN_THRESHOLD,
    TEST_SIZE,
    TRANSITION_RISK_WINDOW,
    USE_ELIGIBILITY_CLASSIFIER,
    CLIP_QUANTILES,
    USE_PCA,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
)
from data import build_dataset
from eligibility import compute_regime_stats, map_eligibility
from models import build_preprocessor, train_direction_model, train_eligibility_classifier, predict_proba_safe
from regimes_hmm import HMM_AVAILABLE, select_hmm_model, predict_hmm
from regimes_gmm import select_gmm_model, predict_gmm
from evaluation import (
    TradeMetrics,
    calibration_plot,
    compute_confusion_matrix,
    compute_trade_metrics,
    confusion_matrix_plot,
    direction_label,
    regime_timeline_plot,
    returns_distribution_plot,
    transition_matrix_plot,
)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _resolve_size(value: Union[float, int], total: int) -> int:
    if value < 1:
        return max(int(total * value), 1)
    return int(value)


def generate_walk_forward_folds(
    n_samples: int,
    n_splits: int,
    min_train_size: Union[float, int],
    test_size: Union[float, int],
    step_size: Optional[Union[float, int]] = None,
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


def _decision_from_probs(
    eligible: bool,
    entropy: float,
    entropy_threshold: float,
    p_long: Optional[float],
    p_short: Optional[float],
    decision_threshold: float,
) -> tuple[str, Optional[float], Optional[float]]:
    if not eligible:
        return "skip", p_long, p_short
    if not np.isfinite(entropy) or entropy > entropy_threshold:
        return "skip", p_long, p_short
    if p_long is None or p_short is None:
        return "skip", p_long, p_short
    if max(p_long, p_short) < decision_threshold:
        return "skip", p_long, p_short
    return ("long" if p_long >= p_short else "short"), p_long, p_short


def _strict_mode_table(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        decisions = []
        for _, row in df.iterrows():
            decision, _, _ = _decision_from_probs(
                eligible=row["eligible_flag"],
                entropy=row["regime_entropy"],
                entropy_threshold=row["entropy_threshold"],
                p_long=row["p_long"],
                p_short=row["p_short"],
                decision_threshold=thr,
            )
            decisions.append(decision)
        tmp = df.copy()
        tmp["decision_tmp"] = decisions
        metrics = compute_trade_metrics(tmp, "decision_tmp", "future_return", RETURN_THRESHOLD)
        rows.append(
            {
                "decision_threshold": thr,
                "coverage": metrics.coverage,
                "precision": metrics.precision,
                "trade_count": metrics.trade_count,
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data_bundle = build_dataset(
        path=str(DATA_PATH),
        horizon=HORIZON_BARS,
        feature_shift=FEATURE_SHIFT,
        enable_leakage_audit=True,
    )
    df = data_bundle.df
    X_all = data_bundle.features.values

    n_samples = len(df)
    fold_outputs = []
    eligibility_rows = []
    backtest_rows = []
    selection_log = []

    hmm_states_all = np.full(n_samples, np.nan)
    hmm_probs_all = np.full(n_samples, np.nan)
    hmm_entropy_all = np.full(n_samples, np.nan)

    gmm_states_all = np.full(n_samples, np.nan)
    gmm_probs_all = np.full(n_samples, np.nan)
    gmm_entropy_all = np.full(n_samples, np.nan)

    decisions_all = np.array(["skip"] * n_samples, dtype=object)
    p_long_all = np.full(n_samples, np.nan)
    p_short_all = np.full(n_samples, np.nan)
    elig_flag_all = np.full(n_samples, False, dtype=object)
    elig_prob_all = np.full(n_samples, np.nan)
    entropy_threshold_all = np.full(n_samples, np.nan)
    oos_mask = np.zeros(n_samples, dtype=bool)

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
            {
                "preprocessor": preprocessor,
                "feature_columns": data_bundle.feature_columns,
            },
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
            )
            hmm_states_train, hmm_post_train, hmm_ent_train = predict_hmm(
                hmm_result.model, X_train_t
            )
            hmm_states_test, hmm_post_test, hmm_ent_test = predict_hmm(
                hmm_result.model, X_test_t
            )
            hmm_probs_train = hmm_post_train.max(axis=1)
            hmm_probs_test = hmm_post_test.max(axis=1)
            hmm_states_all[test_idx] = hmm_states_test
            hmm_probs_all[test_idx] = hmm_probs_test
            hmm_entropy_all[test_idx] = hmm_ent_test

            joblib.dump(
                hmm_result.model,
                ARTIFACTS_DIR / f"hmm_model_fold_{fold_id}.joblib",
            )
            selection_log.append({"fold_id": fold_id, "hmm": hmm_result.metrics})
        else:
            warnings.warn("hmmlearn missing: skipping HMM regimes.")

        gmm_result = select_gmm_model(
            X_train_t,
            GMM_KS,
            GMM_SEEDS,
            GMM_MAX_ITER,
            GMM_COV_TYPE,
        )
        gmm_states_train, gmm_post_train, gmm_ent_train = predict_gmm(
            gmm_result.model, X_train_t
        )
        gmm_states_test, gmm_post_test, gmm_ent_test = predict_gmm(
            gmm_result.model, X_test_t
        )
        gmm_probs_train = gmm_post_train.max(axis=1)
        gmm_probs_test = gmm_post_test.max(axis=1)
        gmm_states_all[test_idx] = gmm_states_test
        gmm_probs_all[test_idx] = gmm_probs_test
        gmm_entropy_all[test_idx] = gmm_ent_test

        joblib.dump(
            gmm_result.model,
            ARTIFACTS_DIR / f"gmm_model_fold_{fold_id}.joblib",
        )
        selection_log.append({"fold_id": fold_id, "gmm": gmm_result.metrics})

        if hmm_result is not None:
            primary_states_train = hmm_states_train
            primary_states_test = hmm_states_test
            entropy_test = hmm_ent_test
            entropy_threshold = HMM_ENTROPY_THRESHOLD
            ari_test = adjusted_rand_score(hmm_states_test, gmm_states_test)
        else:
            primary_states_train = gmm_states_train
            primary_states_test = gmm_states_test
            entropy_test = gmm_ent_test
            entropy_threshold = GMM_ENTROPY_THRESHOLD
            ari_test = np.nan

        df_train["regime_id"] = primary_states_train
        stats = compute_regime_stats(
            df_train,
            state_col="regime_id",
            future_return_col="future_return",
            past_return_col="past_return",
            mae_col="mae",
            horizon=HORIZON_BARS,
            window=TRANSITION_RISK_WINDOW,
            min_regime_frac=ELIGIBILITY_MIN_REGIME_FRAC,
        )
        eligibility = map_eligibility(
            stats,
            min_duration=ELIGIBILITY_D_MIN,
            tail_loss_max=ELIGIBILITY_TAIL_LOSS_MAX,
            win_rate_min=ELIGIBILITY_WIN_RATE_MIN,
            transition_risk_max=ELIGIBILITY_TRANSITION_RISK_MAX,
            min_regime_frac=ELIGIBILITY_MIN_REGIME_FRAC,
        )

        stats = eligibility.stats.copy()
        stats["fold_id"] = fold_id
        eligibility_rows.append(stats)

        eligible_set = eligibility.eligible_regimes
        y_elig_train = df_train["regime_id"].apply(lambda r: 1 if r in eligible_set else 0).values

        if USE_ELIGIBILITY_CLASSIFIER:
            elig_clf = train_eligibility_classifier(X_train_t, y_elig_train)
            joblib.dump(
                elig_clf.model,
                ARTIFACTS_DIR / f"eligibility_model_fold_{fold_id}.joblib",
            )
            elig_probs_test = predict_proba_safe(elig_clf.model, X_test_t)
            elig_prob = elig_probs_test[:, 1] if elig_probs_test is not None else None
        else:
            elig_prob = None

        eligible_flag = np.array([rid in eligible_set for rid in primary_states_test])

        df_dir_train = df_train.copy()
        df_dir_train["eligible"] = y_elig_train
        df_dir_train = df_dir_train[df_dir_train["eligible"] == 1]
        dir_labels = df_dir_train["future_return"].apply(
            lambda r: direction_label(r, RETURN_THRESHOLD)
        )
        dir_mask = dir_labels != 0
        X_dir_train = X_train_t[df_dir_train["train_pos"].values][dir_mask.values]
        y_dir_train = (dir_labels[dir_mask] == 1).astype(int).values

        dir_clf = train_direction_model(X_dir_train, y_dir_train)
        joblib.dump(
            dir_clf.model,
            ARTIFACTS_DIR / f"direction_model_fold_{fold_id}.joblib",
        )

        dir_probs_test = predict_proba_safe(dir_clf.model, X_test_t)
        if dir_probs_test is None:
            p_long = np.full(len(test_idx), np.nan)
            p_short = np.full(len(test_idx), np.nan)
        else:
            p_long = dir_probs_test[:, 1]
            p_short = dir_probs_test[:, 0]

        decisions = []
        for i, rid in enumerate(primary_states_test):
            elig = bool(eligible_flag[i])
            if USE_ELIGIBILITY_CLASSIFIER and elig_prob is not None:
                elig = elig and bool(elig_prob[i] >= ELIGIBILITY_PROB_THRESHOLD)
            decision, pl, ps = _decision_from_probs(
                eligible=elig,
                entropy=entropy_test[i],
                entropy_threshold=entropy_threshold,
                p_long=float(p_long[i]) if np.isfinite(p_long[i]) else None,
                p_short=float(p_short[i]) if np.isfinite(p_short[i]) else None,
                decision_threshold=DECISION_THRESHOLD,
            )
            decisions.append(decision)

        decisions = np.array(decisions, dtype=object)

        decisions_all[test_idx] = decisions
        p_long_all[test_idx] = p_long
        p_short_all[test_idx] = p_short
        elig_flag_all[test_idx] = eligible_flag
        if elig_prob is not None:
            elig_prob_all[test_idx] = elig_prob
        entropy_threshold_all[test_idx] = entropy_threshold
        oos_mask[test_idx] = True

        fold_df = df_test.copy()
        fold_df["decision"] = decisions
        fold_metrics = compute_trade_metrics(
            fold_df, "decision", "future_return", RETURN_THRESHOLD
        )
        fold_outputs.append(
            {
                "fold_id": fold_id,
                "coverage": fold_metrics.coverage,
                "precision": fold_metrics.precision,
                "trade_count": fold_metrics.trade_count,
                "avg_return": fold_metrics.avg_return,
                "tail_loss_p05": fold_metrics.tail_loss_p05,
                "eligible_regimes": len(eligible_set),
                "hmm_states": getattr(hmm_result, "n_states", None),
                "gmm_states": gmm_result.n_states,
                "ari_hmm_gmm_test": ari_test,
            }
        )

        backtest_df_fold = fold_df.assign(regime_id=primary_states_test)
        backtest = (
            backtest_df_fold.groupby("regime_id")["future_return"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        win_rate = (
            backtest_df_fold.groupby("regime_id")["future_return"]
            .apply(lambda s: float((s > 0).mean()))
            .reset_index(name="win_rate")
        )
        tail_loss = (
            backtest_df_fold.groupby("regime_id")["future_return"]
            .apply(lambda s: float(np.quantile(s, 0.05)))
            .reset_index(name="tail_loss_p05")
        )
        backtest = backtest.merge(win_rate, on="regime_id").merge(tail_loss, on="regime_id")
        backtest["fold_id"] = fold_id
        backtest_rows.append(backtest)

    regimes_hmm = pd.DataFrame(
        {
            "timestamp": df[data_bundle.timestamp_column],
            "regime_id": hmm_states_all,
            "regime_prob": hmm_probs_all,
            "regime_entropy": hmm_entropy_all,
        }
    )
    regimes_gmm = pd.DataFrame(
        {
            "timestamp": df[data_bundle.timestamp_column],
            "regime_id": gmm_states_all,
            "regime_prob": gmm_probs_all,
            "regime_entropy": gmm_entropy_all,
        }
    )

    eligibility_df = pd.concat(eligibility_rows, ignore_index=True)
    backtest_df = pd.concat(backtest_rows, ignore_index=True)

    trades_df = df.copy()
    trades_df["decision"] = decisions_all
    trades_df["p_long"] = p_long_all
    trades_df["p_short"] = p_short_all
    trades_df["eligible_flag"] = elig_flag_all
    trades_df["elig_prob"] = elig_prob_all
    trades_df["regime_entropy"] = hmm_entropy_all if HMM_AVAILABLE else gmm_entropy_all
    trades_df["entropy_threshold"] = entropy_threshold_all
    trades_df["hmm_regime_id"] = hmm_states_all
    trades_df["gmm_regime_id"] = gmm_states_all
    trades_df["oos"] = oos_mask

    regimes_hmm.to_csv(OUTPUTS_DIR / "regimes_hmm.csv", index=False)
    regimes_gmm.to_csv(OUTPUTS_DIR / "regimes_gmm.csv", index=False)
    eligibility_df.to_csv(OUTPUTS_DIR / "eligibility_by_regime.csv", index=False)
    trades_df.to_csv(OUTPUTS_DIR / "trades_signals.csv", index=False)
    backtest_df.to_csv(OUTPUTS_DIR / "backtest_summary_by_regime.csv", index=False)

    fold_summary = pd.DataFrame(fold_outputs)
    fold_summary.to_csv(OUTPUTS_DIR / "fold_metrics.csv", index=False)

    if HMM_AVAILABLE and not np.isnan(hmm_states_all).all():
        timeline_df = regimes_hmm.dropna()
        if len(timeline_df) > 0:
            sample_df = timeline_df.tail(min(len(timeline_df), 4000))
            regime_timeline_plot(
                sample_df["timestamp"],
                sample_df["regime_id"],
                "HMM Regime Timeline (OOS)",
                FIGURES_DIR / "regime_timeline.png",
            )

        hmm_model_path = ARTIFACTS_DIR / f"hmm_model_fold_{fold_outputs[-1]['fold_id']}.joblib"
        if hmm_model_path.exists():
            hmm_model = joblib.load(hmm_model_path)
            transition_matrix_plot(
                hmm_model.transmat_,
                "HMM Transition Matrix",
                FIGURES_DIR / "hmm_transition_matrix.png",
            )

    if not np.isnan(gmm_states_all).all():
        returns_distribution_plot(
            trades_df.dropna(subset=["gmm_regime_id"]),
            "gmm_regime_id",
            "future_return",
            "Future Return Distribution by GMM Regime",
            FIGURES_DIR / "returns_by_regime.png",
        )

    if trades_df["oos"].any():
        y_true = trades_df.loc[trades_df["oos"], "eligible_flag"].astype(int).values
        if USE_ELIGIBILITY_CLASSIFIER and not np.isnan(elig_prob_all).all():
            y_pred = (
                trades_df.loc[trades_df["oos"], "elig_prob"] >= ELIGIBILITY_PROB_THRESHOLD
            ).astype(int)
        else:
            y_pred = trades_df.loc[trades_df["oos"], "eligible_flag"].astype(int)
        cm = compute_confusion_matrix(y_true, y_pred.values, labels=[0, 1])
        confusion_matrix_plot(
            cm,
            ["Non-tradable", "Tradable"],
            "Eligibility Confusion Matrix",
            FIGURES_DIR / "confusion_eligibility.png",
        )

    trade_mask = trades_df["decision"].isin(["long", "short"])
    if trade_mask.any():
        y_true_dir = trades_df.loc[trade_mask, "future_return"].apply(
            lambda r: direction_label(r, RETURN_THRESHOLD)
        )
        y_pred_dir = trades_df.loc[trade_mask, "decision"].map({"long": 1, "short": -1})
        valid_mask = y_true_dir != 0
        cm_dir = compute_confusion_matrix(
            y_true_dir[valid_mask].values, y_pred_dir[valid_mask].values, labels=[-1, 1]
        )
        confusion_matrix_plot(
            cm_dir,
            ["Short", "Long"],
            "Direction Confusion Matrix",
            FIGURES_DIR / "confusion_direction.png",
        )
    else:
        cm_dir = np.zeros((2, 2), dtype=int)
        confusion_matrix_plot(
            cm_dir,
            ["Short", "Long"],
            "Direction Confusion Matrix",
            FIGURES_DIR / "confusion_direction.png",
        )

    calib_mask = trade_mask & trades_df["p_long"].notna()
    if calib_mask.any():
        y_true = trades_df.loc[calib_mask, "future_return"].apply(
            lambda r: direction_label(r, RETURN_THRESHOLD)
        )
        y_true = (y_true == 1).astype(int)
        y_prob = trades_df.loc[calib_mask, "p_long"].values
    else:
        y_true = np.array([])
        y_prob = np.array([])
    calibration_plot(
        y_true,
        y_prob,
        "Calibration Curve (Long)",
        FIGURES_DIR / "calibration_curve.png",
    )

    strict_table = _strict_mode_table(
        trades_df[trades_df["oos"]],
        thresholds=[0.8, 0.9, 0.95, 0.98],
    )
    strict_table.to_csv(OUTPUTS_DIR / "strict_mode_table.csv", index=False)

    report = {
        "data_rows": n_samples,
        "feature_count": len(data_bundle.feature_columns),
        "excluded_columns": data_bundle.excluded_columns,
        "suspicious_columns": data_bundle.suspicious_columns,
        "fold_metrics": fold_outputs,
        "strict_mode_table": strict_table.to_dict(orient="records"),
        "hmm_available": HMM_AVAILABLE,
        "selection_log": selection_log,
    }
    with open(ARTIFACTS_DIR / "selection_log.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)

    report_lines = []
    report_lines.append("# Regime-First, Uncertainty-Aware Pipeline Report (12h)")
    report_lines.append("")
    report_lines.append("## Data Summary")
    report_lines.append(f"- Rows: {n_samples}")
    report_lines.append(f"- Features: {len(data_bundle.feature_columns)}")
    report_lines.append(f"- Excluded columns: {', '.join(data_bundle.excluded_columns)}")
    if data_bundle.suspicious_columns:
        report_lines.append(f"- Suspicious columns: {', '.join(data_bundle.suspicious_columns)}")
    report_lines.append("")
    report_lines.append("## Fold Metrics (OOS)")
    for fold in fold_outputs:
        report_lines.append(
            f"- Fold {fold['fold_id']}: coverage={fold['coverage']:.3f}, "
            f"precision={fold['precision']:.3f}, trades={fold['trade_count']}, "
            f"eligible_regimes={fold['eligible_regimes']}, "
            f"HMM_K={fold['hmm_states']}, GMM_K={fold['gmm_states']}"
        )
    report_lines.append("")
    report_lines.append("## Strict Mode Table")
    for row in report["strict_mode_table"]:
        report_lines.append(
            f"- threshold={row['decision_threshold']}: coverage={row['coverage']:.3f}, "
            f"precision={row['precision']:.3f}, trades={row['trade_count']}"
        )
    report_lines.append("")
    report_lines.append("## Conclusions")
    report_lines.append("- Precision is prioritized via regime gating, entropy filters, and calibrated probabilities.")
    report_lines.append("- Coverage is intentionally low at higher thresholds.")
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("- Consider TP/SL-based labels for more trade-like supervision.")
    report_lines.append("- Try different horizons and eligibility thresholds for higher precision.")
    report_lines.append("- Prune features or add regime-specific thresholds if precision remains low.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    best_hmm_k = None
    if HMM_AVAILABLE and fold_outputs:
        best_hmm_k = fold_outputs[-1]["hmm_states"]
    best_gmm_k = fold_outputs[-1]["gmm_states"] if fold_outputs else None
    eligible_counts = [f["eligible_regimes"] for f in fold_outputs]
    eligible_count = int(np.mean(eligible_counts)) if eligible_counts else 0

    metrics = compute_trade_metrics(
        trades_df[trades_df["oos"]], "decision", "future_return", RETURN_THRESHOLD
    )
    print("Chosen HMM K:", best_hmm_k)
    print("Chosen GMM K:", best_gmm_k)
    print("Eligible regimes (avg):", eligible_count)
    print("Coverage @0.95:", metrics.coverage)
    print("Precision @0.95:", metrics.precision)
    print("Outputs saved to:", OUTPUTS_DIR)


if __name__ == "__main__":
    run()
