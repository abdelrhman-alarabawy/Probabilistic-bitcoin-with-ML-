from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest_trades
from .config import (
    AMBIGUOUS_COL,
    ARTIFACTS_DIR,
    DATA_CSV,
    DECISION_TIME,
    EARLY_LATE_K,
    FIGURES_DIR,
    FIVE_MIN_CSV,
    HORIZON_MINUTES,
    LABEL_COL,
    LOOKBACK_HOURS,
    MICROSTRUCTURE_ENABLED,
    MIN_5M_BARS,
    REPORT_PATH,
    REPORTS_DIR,
    SEED,
    RET_5M_Q,
    ROLLING_WINDOW,
    SL_POINTS,
    TABLES_DIR,
    TP_POINTS,
)
from .features import build_feature_matrix
from .io import load_5m_dataset, load_dataset
from .metrics import classification_metrics, label_distribution, per_month_metrics, trade_metrics
from .microstructure_5m import (
    build_5m_index,
    compute_microstructure_features,
    compute_train_return_cutoff,
)
from .models import predict_proba_positive, train_classifier
from .plots import plot_confusion_matrix, plot_equity_curve, plot_precision_coverage
from .report import write_report
from .risk_filters import apply_risk_filters, fit_risk_filter_cutoffs, risk_filter_cutoffs_table
from .splits import holdout_split, train_val_test_split, walk_forward_splits
from .thresholds import apply_decision, save_thresholds, select_best_thresholds, sweep_thresholds

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def _ensure_dirs() -> None:
    for path in (ARTIFACTS_DIR, REPORTS_DIR, TABLES_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _align_labels_for_close(df: pd.DataFrame) -> pd.DataFrame:
    if DECISION_TIME != "close":
        return df
    shifted = df.copy()
    shifted[LABEL_COL] = shifted[LABEL_COL].shift(-1)
    if AMBIGUOUS_COL in shifted.columns:
        shifted[AMBIGUOUS_COL] = shifted[AMBIGUOUS_COL].shift(-1)
    shifted = shifted.iloc[:-1].reset_index(drop=True)
    return shifted


def _save_label_distribution(
    df: pd.DataFrame,
    split_idx: pd.Index,
    name: str,
    risk_pass: pd.Series,
) -> pd.DataFrame:
    split_labels = df.loc[split_idx, LABEL_COL]
    before = split_labels.value_counts()
    after = split_labels[risk_pass.loc[split_idx]].value_counts()
    total_before = len(split_labels)
    total_after = int(risk_pass.loc[split_idx].sum())
    rows = []
    for label in ("long", "short", "skip"):
        rows.append(
            {
                "split": name,
                "label": label,
                "before_count": int(before.get(label, 0)),
                "before_pct": float(before.get(label, 0) / total_before) if total_before else 0.0,
                "after_count": int(after.get(label, 0)),
                "after_pct": float(after.get(label, 0) / total_after) if total_after else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _safe_gate_auc(y_true: pd.Series, p_trade: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_trade))


def _run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    splits,
    risk_df: pd.DataFrame,
    save_artifacts: bool,
) -> dict:
    y_gate = (y != "skip").astype(int)
    X_train = X.loc[splits.train_idx].to_numpy()
    X_val = X.loc[splits.val_idx].to_numpy()
    X_test = X.loc[splits.test_idx].to_numpy()

    if y_gate.loc[splits.train_idx].nunique() < 2:
        raise ValueError("Gate training requires both trade and skip labels in the train split.")

    gate_model = train_classifier(X_train, y_gate.loc[splits.train_idx].to_numpy())
    p_trade_val = predict_proba_positive(gate_model.model, X_val, positive_label=1)
    p_trade_test = predict_proba_positive(gate_model.model, X_test, positive_label=1)

    gate_auc = _safe_gate_auc(y_gate.loc[splits.test_idx], p_trade_test)

    trade_mask_train = y.loc[splits.train_idx] != "skip"
    if trade_mask_train.sum() == 0:
        raise ValueError("Direction training requires at least one trade sample in the train split.")
    X_train_trade = X.loc[splits.train_idx][trade_mask_train].to_numpy()
    y_dir_train = (y.loc[splits.train_idx][trade_mask_train] == "long").astype(int).to_numpy()

    dir_model = train_classifier(X_train_trade, y_dir_train)
    p_long_val = predict_proba_positive(dir_model.model, X_val, positive_label=1)
    p_long_test = predict_proba_positive(dir_model.model, X_test, positive_label=1)

    sweep = sweep_thresholds(
        y.loc[splits.val_idx],
        p_trade_val,
        p_long_val,
        risk_df.loc[splits.val_idx, "risk_pass"],
    )
    if save_artifacts:
        sweep.to_csv(TABLES_DIR / "threshold_sweep.csv", index=False)
        plot_precision_coverage(sweep, FIGURES_DIR / "precision_coverage.png")

    best = select_best_thresholds(sweep)
    if save_artifacts:
        save_thresholds(best, TABLES_DIR / "best_thresholds.json")

    preds_test = apply_decision(
        p_trade_test,
        p_long_test,
        risk_df.loc[splits.test_idx, "risk_pass"],
        best.t_trade,
        best.t_long,
        best.t_short,
    )

    test_metrics = classification_metrics(y.loc[splits.test_idx], preds_test)
    if save_artifacts:
        test_metrics.confusion_matrix.to_csv(TABLES_DIR / "confusion_matrix.csv")
        plot_confusion_matrix(test_metrics.confusion_matrix, FIGURES_DIR / "confusion_matrix.png")

    per_month = per_month_metrics(
        df.loc[splits.test_idx, "timestamp"], y.loc[splits.test_idx], preds_test
    )
    if save_artifacts:
        per_month.to_csv(TABLES_DIR / "per_month_metrics.csv", index=False)

    test_df = df.loc[splits.test_idx].reset_index(drop=True)
    preds_test_reset = preds_test.reset_index(drop=True)
    backtest = backtest_trades(test_df, preds_test_reset, TP_POINTS, SL_POINTS, HORIZON_MINUTES)
    if save_artifacts:
        backtest.trade_log.to_csv(TABLES_DIR / "test_trade_log.csv", index=False)
        plot_equity_curve(backtest.equity_curve, FIGURES_DIR / "equity_curve.png")
        trades_per_month = (
            backtest.trade_log.assign(
                month=backtest.trade_log["entry_timestamp"].dt.to_period("M").astype(str)
            )
            .groupby("month")
            .size()
            .reset_index(name="trades")
        )
        trades_per_month.to_csv(TABLES_DIR / "trades_per_month.csv", index=False)

    trade_stats = trade_metrics(backtest.trade_log)

    return {
        "thresholds": best,
        "metrics": test_metrics,
        "backtest": trade_stats,
        "per_month": per_month,
        "gate_auc": gate_auc,
    }


def _micro_feature_stats(
    micro_df: pd.DataFrame, split_idx: pd.Index, split_name: str
) -> pd.DataFrame:
    rows = []
    for col in micro_df.columns:
        series = micro_df.loc[split_idx, col]
        rows.append(
            {
                "split": split_name,
                "feature": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "p05": float(series.quantile(0.05)),
                "p50": float(series.quantile(0.50)),
                "p95": float(series.quantile(0.95)),
                "missing_pct": float(series.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def _sample_indices(
    rng: np.random.RandomState, idx: pd.Index, count: int
) -> list[int]:
    if len(idx) == 0 or count <= 0:
        return []
    count = min(count, len(idx))
    return rng.choice(idx.to_numpy(), size=count, replace=False).tolist()


def main() -> None:
    _ensure_dirs()

    load = load_dataset(DATA_CSV)
    df = _align_labels_for_close(load.df)

    holdout = holdout_split(df)
    splits = train_val_test_split(df)
    walk_forward = walk_forward_splits(df)
    pd.DataFrame(walk_forward).to_csv(TABLES_DIR / "walk_forward_splits.csv", index=False)

    cutoffs = fit_risk_filter_cutoffs(df.loc[splits.train_idx])
    risk_df = apply_risk_filters(df, cutoffs)
    risk_df["timestamp"] = df["timestamp"]

    pass_rates = (
        risk_df.assign(month=risk_df["timestamp"].dt.to_period("M").astype(str))
        .groupby("month")[["vol_ratio_pass", "range_z_pass", "atr_pct_pass", "risk_pass"]]
        .mean()
        .reset_index()
    )
    pass_rates.to_csv(TABLES_DIR / "risk_filter_pass_rates.csv", index=False)
    pd.DataFrame([risk_filter_cutoffs_table(cutoffs)]).to_csv(
        TABLES_DIR / "risk_filter_cutoffs.csv", index=False
    )

    label_before_after = pd.concat(
        [
            _save_label_distribution(df, splits.train_idx, "train", risk_df["risk_pass"]),
            _save_label_distribution(df, splits.val_idx, "val", risk_df["risk_pass"]),
            _save_label_distribution(df, splits.test_idx, "test", risk_df["risk_pass"]),
        ],
        ignore_index=True,
    )
    label_before_after.to_csv(TABLES_DIR / "risk_filter_label_distribution.csv", index=False)

    train_features_base = build_feature_matrix(
        df.loc[splits.train_idx],
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
    )
    base_feature_cols = train_features_base.feature_cols
    base_imputer = train_features_base.imputer

    full_features_base = build_feature_matrix(
        df,
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
        imputer=base_imputer,
        feature_cols=base_feature_cols,
    )

    micro_features = None
    micro_diag = None
    micro_cols: list[str] = []
    ret_cutoff = float("nan")
    alignment_pass = True

    if MICROSTRUCTURE_ENABLED:
        df5 = load_5m_dataset(FIVE_MIN_CSV)
        index_5m = build_5m_index(df5)
        train_end_ts = df.loc[splits.train_idx[-1], "timestamp"]
        ret_cutoff = compute_train_return_cutoff(df5, train_end_ts, RET_5M_Q)
        vol_scale = (
            df["volume"].rolling(window=ROLLING_WINDOW, min_periods=1).median().shift(1)
        )
        micro_features, micro_diag = compute_microstructure_features(
            df["timestamp"],
            index_5m,
            vol_scale=vol_scale,
            lookback_hours=LOOKBACK_HOURS,
            min_bars=MIN_5M_BARS,
            early_late_k=EARLY_LATE_K,
            ret_cutoff=ret_cutoff,
        )
        micro_cols = list(micro_features.columns)

        stats_df = pd.concat(
            [
                _micro_feature_stats(micro_features, splits.train_idx, "train"),
                _micro_feature_stats(micro_features, splits.val_idx, "val"),
                _micro_feature_stats(micro_features, splits.test_idx, "test"),
            ],
            ignore_index=True,
        )
        stats_df.to_csv(TABLES_DIR / "microstructure_feature_stats.csv", index=False)

        rng = np.random.RandomState(SEED)
        samples = []
        samples += _sample_indices(rng, splits.train_idx, 20)
        samples += _sample_indices(rng, splits.val_idx, 15)
        samples += _sample_indices(rng, splits.test_idx, 15)
        alignment_checks = micro_diag.loc[samples].copy()
        alignment_checks["split"] = [
            "train" if idx in splits.train_idx else "val" if idx in splits.val_idx else "test"
            for idx in alignment_checks.index
        ]
        alignment_checks = alignment_checks[
            [
                "timestamp_1h",
                "window_start",
                "window_end",
                "first_5m_ts",
                "last_5m_ts",
                "n_5m",
                "expected_5m",
                "missing_frac",
                "split",
            ]
        ]
        alignment_checks.to_csv(
            TABLES_DIR / "microstructure_alignment_checks.csv", index=False
        )

        alignment_mask = micro_diag["last_5m_ts"].isna() | (
            micro_diag["last_5m_ts"] < micro_diag["timestamp_1h"]
        )
        alignment_pass = bool(alignment_mask.all())

        debug_candidates = micro_diag[micro_diag["n_5m"] >= MIN_5M_BARS]
        debug_rows = []
        if len(debug_candidates) > 0:
            debug_samples = debug_candidates.sample(
                n=min(10, len(debug_candidates)), random_state=SEED
            )
            for idx, row in debug_samples.iterrows():
                ms_ret = (
                    micro_features.loc[idx, "ms_ret_1h"]
                    if micro_features is not None
                    else float("nan")
                )
                calc_ret = (row["c_last"] / row["o_first"]) - 1.0 if row["n_5m"] > 0 else float("nan")
                debug_rows.append(
                    {
                        "timestamp_1h": row["timestamp_1h"],
                        "window_start": row["window_start"],
                        "window_end": row["window_end"],
                        "first_5m_ts": row["first_5m_ts"],
                        "last_5m_ts": row["last_5m_ts"],
                        "n_5m": row["n_5m"],
                        "o_first": row["o_first"],
                        "c_last": row["c_last"],
                        "ms_ret_1h_calc": calc_ret,
                        "ms_ret_1h_feature": ms_ret,
                    }
                )
        pd.DataFrame(debug_rows).to_csv(
            TABLES_DIR / "microstructure_debug_samples.csv", index=False
        )

    train_features_micro = build_feature_matrix(
        df.loc[splits.train_idx],
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
        feature_cols=base_feature_cols,
        micro_df=micro_features,
        micro_cols=micro_cols,
    )
    micro_imputer = train_features_micro.imputer

    full_features_micro = build_feature_matrix(
        df,
        timestamp_col="timestamp",
        decision_time=DECISION_TIME,
        imputer=micro_imputer,
        feature_cols=base_feature_cols,
        micro_df=micro_features,
        micro_cols=micro_cols,
    )

    y = df[LABEL_COL]
    baseline_results = _run_experiment(
        full_features_base.features,
        y,
        df,
        splits,
        risk_df,
        save_artifacts=False,
    )
    micro_results = _run_experiment(
        full_features_micro.features,
        y,
        df,
        splits,
        risk_df,
        save_artifacts=True,
    )

    label_dist = {
        "train": label_distribution(y.loc[splits.train_idx]),
        "val": label_distribution(y.loc[splits.val_idx]),
        "test": label_distribution(y.loc[splits.test_idx]),
    }

    report_payload = {
        "dataset": {
            "rows": len(df),
            "start": str(df["timestamp"].iloc[0]),
            "end": str(df["timestamp"].iloc[-1]),
            "dropped_duplicates": load.dropped_duplicates,
            "decision_time": DECISION_TIME,
        },
        "splits": {
            "holdout_train_end": str(df.loc[holdout.train_idx[-1], "timestamp"]),
            "holdout_test_start": str(df.loc[holdout.test_idx[0], "timestamp"]),
            "train_end": str(df.loc[splits.train_idx[-1], "timestamp"]),
            "val_end": str(df.loc[splits.val_idx[-1], "timestamp"]),
            "test_start": str(df.loc[splits.test_idx[0], "timestamp"]),
        },
        "label_dist": label_dist,
        "risk": {
            "cutoffs_path": str(TABLES_DIR / "risk_filter_cutoffs.csv"),
            "pass_rates_path": str(TABLES_DIR / "risk_filter_pass_rates.csv"),
            "filtered_labels_path": str(TABLES_DIR / "risk_filter_label_distribution.csv"),
        },
        "microstructure": {
            "enabled": MICROSTRUCTURE_ENABLED,
            "lookback_hours": LOOKBACK_HOURS,
            "min_5m_bars": MIN_5M_BARS,
            "ret_cutoff": ret_cutoff,
            "feature_count": len(micro_cols),
            "alignment_pass": alignment_pass,
            "alignment_checks_path": str(TABLES_DIR / "microstructure_alignment_checks.csv"),
            "feature_stats_path": str(TABLES_DIR / "microstructure_feature_stats.csv"),
            "debug_samples_path": str(TABLES_DIR / "microstructure_debug_samples.csv"),
        },
        "comparison": {
            "baseline": {
                "precision_long": baseline_results["metrics"].precision_long,
                "precision_short": baseline_results["metrics"].precision_short,
                "precision_trade": baseline_results["metrics"].precision_trade,
                "coverage_total": baseline_results["metrics"].coverage_total,
                "expectancy": baseline_results["backtest"]["expectancy"],
                "profit_factor": baseline_results["backtest"]["profit_factor"],
                "max_drawdown": baseline_results["backtest"]["max_drawdown"],
                "gate_auc": baseline_results["gate_auc"],
            },
            "micro": {
                "precision_long": micro_results["metrics"].precision_long,
                "precision_short": micro_results["metrics"].precision_short,
                "precision_trade": micro_results["metrics"].precision_trade,
                "coverage_total": micro_results["metrics"].coverage_total,
                "expectancy": micro_results["backtest"]["expectancy"],
                "profit_factor": micro_results["backtest"]["profit_factor"],
                "max_drawdown": micro_results["backtest"]["max_drawdown"],
                "gate_auc": micro_results["gate_auc"],
            },
        },
        "thresholds": {
            "t_trade": micro_results["thresholds"].t_trade,
            "t_long": micro_results["thresholds"].t_long,
            "t_short": micro_results["thresholds"].t_short,
            "min_precision": min(
                micro_results["thresholds"].precision_long,
                micro_results["thresholds"].precision_short,
            ),
        },
        "test_metrics": {
            "precision_long": micro_results["metrics"].precision_long,
            "precision_short": micro_results["metrics"].precision_short,
            "precision_trade": micro_results["metrics"].precision_trade,
            "coverage_total": micro_results["metrics"].coverage_total,
            "gate_auc": micro_results["gate_auc"],
        },
        "backtest": micro_results["backtest"],
        "tables": {
            "threshold_sweep": str(TABLES_DIR / "threshold_sweep.csv"),
            "per_month": str(TABLES_DIR / "per_month_metrics.csv"),
            "trade_log": str(TABLES_DIR / "test_trade_log.csv"),
            "trades_per_month": str(TABLES_DIR / "trades_per_month.csv"),
        },
        "figures": {
            "precision_coverage": str(FIGURES_DIR / "precision_coverage.png"),
            "confusion_matrix": str(FIGURES_DIR / "confusion_matrix.png"),
            "equity_curve": str(FIGURES_DIR / "equity_curve.png"),
        },
        "verdict": "If precision remains low or coverage collapses, use this pipeline as a filter rather than a standalone trading system.",
    }

    write_report(REPORT_PATH, report_payload)

    print("Trading pipeline complete.")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Microstructure features added: {len(micro_cols)} columns")
    print(f"No-leakage alignment checks: {'PASS' if alignment_pass else 'FAIL'}")
    print(
        f"Test precision long/short: {micro_results['metrics'].precision_long:.3f} / "
        f"{micro_results['metrics'].precision_short:.3f} | coverage={micro_results['metrics'].coverage_total:.3f}"
    )
    print(
        f"Backtest expectancy={micro_results['backtest']['expectancy']:.4f} "
        f"profit_factor={micro_results['backtest']['profit_factor']:.3f} "
        f"max_drawdown={micro_results['backtest']['max_drawdown']:.3f}"
    )


if __name__ == "__main__":
    main()
