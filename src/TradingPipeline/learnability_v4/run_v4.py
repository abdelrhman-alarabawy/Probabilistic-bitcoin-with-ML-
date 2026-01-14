from __future__ import annotations

import json
from typing import Dict

import pandas as pd

from ..backtest import backtest_trades
from ..metrics import trade_metrics
from ..thresholds import apply_decision as apply_decision_baseline
from ..thresholds import select_best_thresholds as select_best_thresholds_baseline
from ..thresholds import sweep_thresholds as sweep_thresholds_baseline
from ..config import DECISION_TIME
from .config_v4 import REPORT_PATH, TABLES_DIR
from .data_v4 import DataBundle, load_data_bundle
from .models_v4 import (
    DirectionModels,
    calibrate_model_on_val,
    predict_direction_probs,
    predict_proba_positive,
    train_direction_model,
    train_direction_models_by_regime,
    train_gate_model,
)
from .report_v4 import write_report
from .thresholds_v4 import select_direction_thresholds, select_trade_threshold
from .config_v4 import T_TRADE_GRID


def _ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _apply_decision_v4(
    p_trade: pd.Series,
    p_long: pd.Series,
    risk_pass: pd.Series,
    direction_quality: pd.Series,
    t_trade: float,
    t_long: float,
    t_short: float,
) -> pd.Series:
    trade_mask = risk_pass & (p_trade >= t_trade)
    dir_mask = trade_mask & direction_quality

    preds = pd.Series("skip", index=p_trade.index)
    preds.loc[dir_mask & (p_long >= t_long)] = "long"
    preds.loc[dir_mask & ((1.0 - p_long) >= t_short)] = "short"
    return preds


def _evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    long_pred = (y_pred == "long")
    short_pred = (y_pred == "short")
    trade_pred = y_pred != "skip"

    precision_long = float((y_true[long_pred] == "long").mean()) if long_pred.any() else 0.0
    precision_short = float((y_true[short_pred] == "short").mean()) if short_pred.any() else 0.0
    trade_precision = float((y_true[trade_pred] != "skip").mean()) if trade_pred.any() else 0.0
    coverage = float(trade_pred.mean())

    return {
        "precision_long": precision_long,
        "precision_short": precision_short,
        "trade_precision": trade_precision,
        "coverage_total": coverage,
        "long_pred_count": int(long_pred.sum()),
        "short_pred_count": int(short_pred.sum()),
    }


def _confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = ["long", "short", "skip"]
    return (
        pd.crosstab(y_true, y_pred, dropna=False)
        .reindex(index=labels, columns=labels, fill_value=0)
    )


def _per_month_metrics(
    timestamps: pd.Series, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": timestamps, "y_true": y_true, "y_pred": y_pred})
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        metrics = _evaluate_predictions(group["y_true"], group["y_pred"])
        rows.append({"month": month, **metrics})
    return pd.DataFrame(rows)


def _baseline_pipeline(bundle: DataBundle) -> Dict[str, float]:
    df = bundle.df
    splits = bundle.splits
    X = bundle.features
    y = df["candle_type"]

    y_gate = (y != "skip").astype(int)
    gate_model = train_gate_model(X.loc[splits.train_idx].to_numpy(), y_gate.loc[splits.train_idx].to_numpy())
    p_trade_val = predict_proba_positive(gate_model.model, X.loc[splits.val_idx].to_numpy())
    p_trade_test = predict_proba_positive(gate_model.model, X.loc[splits.test_idx].to_numpy())

    trade_mask = y.loc[splits.train_idx] != "skip"
    dir_model = train_direction_model(
        X.loc[splits.train_idx][trade_mask].to_numpy(),
        (y.loc[splits.train_idx][trade_mask] == "long").astype(int).to_numpy(),
        balance=False,
    )
    p_long_val = predict_proba_positive(dir_model.model, X.loc[splits.val_idx].to_numpy())
    p_long_test = predict_proba_positive(dir_model.model, X.loc[splits.test_idx].to_numpy())

    sweep = sweep_thresholds_baseline(
        y.loc[splits.val_idx],
        p_trade_val,
        p_long_val,
        bundle.risk_df.loc[splits.val_idx, "risk_pass"],
    )
    best = select_best_thresholds_baseline(sweep)

    preds_test = apply_decision_baseline(
        p_trade_test,
        p_long_test,
        bundle.risk_df.loc[splits.test_idx, "risk_pass"],
        best.t_trade,
        best.t_long,
        best.t_short,
    )

    metrics = _evaluate_predictions(y.loc[splits.test_idx], preds_test)
    backtest = backtest_trades(
        df.loc[splits.test_idx].reset_index(drop=True),
        preds_test.reset_index(drop=True),
        tp_points=400,
        sl_points=200,
        horizon_minutes=60,
    )
    backtest_stats = trade_metrics(backtest.trade_log)
    return {**metrics, **backtest_stats}


def run_v4() -> None:
    _ensure_dirs()
    bundle = load_data_bundle()

    df = bundle.df
    splits = bundle.splits
    X = bundle.features
    y = df["candle_type"]

    y_gate = (y != "skip").astype(int)
    gate_model = train_gate_model(X.loc[splits.train_idx].to_numpy(), y_gate.loc[splits.train_idx].to_numpy())
    gate_model = calibrate_model_on_val(
        gate_model,
        X.loc[splits.val_idx].to_numpy(),
        y_gate.loc[splits.val_idx].to_numpy(),
    )

    direction_train_mask = bundle.direction_train_mask & bundle.df.index.isin(splits.train_idx)
    if direction_train_mask.sum() == 0:
        raise ValueError("No direction training samples after filters.")

    y_dir_train = (y[direction_train_mask] == "long").astype(int)

    if bundle.regime_col:
        regimes = df[bundle.regime_col]
        y_dir = (y == "long").astype(int)
        direction_models = train_direction_models_by_regime(
            X,
            y_dir,
            direction_train_mask,
            (
                bundle.direction_quality_mask
                & bundle.risk_df["risk_pass"]
                & y.isin(["long", "short"])
                & df.index.isin(splits.val_idx)
            ),
            regimes,
        )
        balance_method = direction_models.global_model.balance_method
        calibration_method = direction_models.global_model.calibration_method
    else:
        dir_model = train_direction_model(X.loc[direction_train_mask].to_numpy(), y_dir_train.to_numpy())
        val_dir_mask = (
            bundle.direction_quality_mask
            & bundle.risk_df["risk_pass"]
            & y.isin(["long", "short"])
            & y.index.isin(splits.val_idx)
        )
        dir_model = calibrate_model_on_val(
            dir_model,
            X.loc[val_dir_mask].to_numpy(),
            (y.loc[val_dir_mask] == "long").astype(int).to_numpy(),
        )
        direction_models = DirectionModels(global_model=dir_model, regime_models={})
        balance_method = dir_model.balance_method
        calibration_method = dir_model.calibration_method

    p_trade_val = pd.Series(
        predict_proba_positive(gate_model.calibrated, X.loc[splits.val_idx].to_numpy()),
        index=splits.val_idx,
    )
    p_trade_test = pd.Series(
        predict_proba_positive(gate_model.calibrated, X.loc[splits.test_idx].to_numpy()),
        index=splits.test_idx,
    )

    p_long_val = predict_direction_probs(
        direction_models,
        X.loc[splits.val_idx],
        df.loc[splits.val_idx, bundle.regime_col] if bundle.regime_col else None,
    )
    p_long_test = predict_direction_probs(
        direction_models,
        X.loc[splits.test_idx],
        df.loc[splits.test_idx, bundle.regime_col] if bundle.regime_col else None,
    )

    t_trade, trade_sweep = select_trade_threshold(
        y.loc[splits.val_idx], p_trade_val, bundle.risk_df.loc[splits.val_idx, "risk_pass"]
    )
    selection, direction_sweep = select_direction_thresholds(
        y.loc[splits.val_idx],
        p_trade_val,
        p_long_val,
        bundle.risk_df.loc[splits.val_idx, "risk_pass"],
        bundle.direction_quality_mask.loc[splits.val_idx],
        t_trade,
    )
    if not selection.feasible:
        for candidate in sorted(T_TRADE_GRID):
            if candidate == t_trade:
                continue
            candidate_sel, candidate_sweep = select_direction_thresholds(
                y.loc[splits.val_idx],
                p_trade_val,
                p_long_val,
                bundle.risk_df.loc[splits.val_idx, "risk_pass"],
                bundle.direction_quality_mask.loc[splits.val_idx],
                candidate,
            )
            if candidate_sel.feasible:
                candidate_sel.relaxation_log.insert(0, f"t_trade_relaxed_to={candidate:.2f}")
                selection = candidate_sel
                direction_sweep = candidate_sweep
                break

    preds_test = _apply_decision_v4(
        p_trade_test,
        p_long_test,
        bundle.risk_df.loc[splits.test_idx, "risk_pass"],
        bundle.direction_quality_mask.loc[splits.test_idx],
        selection.t_trade,
        selection.t_long,
        selection.t_short,
    )

    metrics = _evaluate_predictions(y.loc[splits.test_idx], preds_test)

    trade_log = backtest_trades(
        df.loc[splits.test_idx].reset_index(drop=True),
        preds_test.reset_index(drop=True),
        tp_points=400,
        sl_points=200,
        horizon_minutes=60,
    ).trade_log

    backtest_stats = {
        "expectancy": float(trade_log["net_return"].mean()) if not trade_log.empty else 0.0,
        "profit_factor": float(
            trade_log[trade_log["net_return"] > 0]["net_return"].sum()
            / max(-trade_log[trade_log["net_return"] < 0]["net_return"].sum(), 1e-12)
        )
        if not trade_log.empty
        else 0.0,
        "max_drawdown": float(
            ((1 + trade_log["net_return"]).cumprod().pct_change().min())
            if not trade_log.empty
            else 0.0
        ),
    }
    metrics.update(backtest_stats)

    baseline = _baseline_pipeline(bundle)

    per_month = _per_month_metrics(
        df.loc[splits.test_idx, "timestamp"],
        y.loc[splits.test_idx],
        preds_test,
    )

    conf = _confusion_matrix(y.loc[splits.test_idx], preds_test)

    trade_sweep.to_csv(TABLES_DIR / "v4_trade_threshold_sweep.csv", index=False)
    direction_sweep.to_csv(TABLES_DIR / "v4_direction_threshold_sweep.csv", index=False)
    per_month.to_csv(TABLES_DIR / "v4_per_month_metrics.csv", index=False)
    conf.to_csv(TABLES_DIR / "v4_confusion_matrix.csv")

    counts = pd.DataFrame(
        [
            {"split": "test", **metrics},
        ]
    )
    counts.to_csv(TABLES_DIR / "v4_prediction_counts.csv", index=False)

    threshold_payload = {
        "t_trade": selection.t_trade,
        "t_long": selection.t_long,
        "t_short": selection.t_short,
        "relaxation_log": selection.relaxation_log,
    }
    (TABLES_DIR / "v4_chosen_thresholds.json").write_text(json.dumps(threshold_payload, indent=2))

    report_payload = {
        "dataset": {
            "rows": len(df),
            "start": str(df["timestamp"].iloc[0]),
            "end": str(df["timestamp"].iloc[-1]),
            "decision_time": DECISION_TIME,
        },
        "splits": {
            "train_end": str(df.loc[splits.train_idx[-1], "timestamp"]),
            "val_end": str(df.loc[splits.val_idx[-1], "timestamp"]),
            "test_start": str(df.loc[splits.test_idx[0], "timestamp"]),
        },
        "direction": {
            "wickiness_cutoff": bundle.micro_cutoffs["wickiness_cutoff"],
            "chop_cutoff": bundle.micro_cutoffs["chop_cutoff"],
            "missing_frac_max": bundle.micro_cutoffs["missing_frac_max"],
            "direction_train_count": int(bundle.direction_train_mask.sum()),
            "direction_train_trade_frac": float(bundle.direction_train_mask.mean()),
            "balance_method": balance_method,
            "calibration_method": calibration_method,
        },
        "thresholds": threshold_payload,
        "metrics": metrics,
        "baseline": baseline,
        "tables": {
            "trade_sweep": str(TABLES_DIR / "v4_trade_threshold_sweep.csv"),
            "direction_sweep": str(TABLES_DIR / "v4_direction_threshold_sweep.csv"),
            "chosen_thresholds": str(TABLES_DIR / "v4_chosen_thresholds.json"),
            "prediction_counts": str(TABLES_DIR / "v4_prediction_counts.csv"),
            "per_month": str(TABLES_DIR / "v4_per_month_metrics.csv"),
            "confusion_matrix": str(TABLES_DIR / "v4_confusion_matrix.csv"),
        },
    }

    write_report(REPORT_PATH, report_payload)

    print("Learnability v4 pipeline complete.")
    print(f"Report saved to: {REPORT_PATH}")
    print("Baseline vs v4 (test):")
    print(
        f"- Baseline precision_long={baseline['precision_long']:.3f}, "
        f"precision_short={baseline['precision_short']:.3f}, "
        f"trade_precision={baseline['trade_precision']:.3f}, "
        f"coverage={baseline['coverage_total']:.3f}, "
        f"long_pred={baseline['long_pred_count']}, short_pred={baseline['short_pred_count']}"
    )
    print(
        f"- v4 precision_long={metrics['precision_long']:.3f}, "
        f"precision_short={metrics['precision_short']:.3f}, "
        f"trade_precision={metrics['trade_precision']:.3f}, "
        f"coverage={metrics['coverage_total']:.3f}, "
        f"long_pred={metrics['long_pred_count']}, short_pred={metrics['short_pred_count']}"
    )
    print(
        f"Backtest expectancy={metrics['expectancy']:.4f}, "
        f"profit_factor={metrics['profit_factor']:.3f}, "
        f"max_drawdown={metrics['max_drawdown']:.3f}"
    )


if __name__ == "__main__":
    run_v4()
