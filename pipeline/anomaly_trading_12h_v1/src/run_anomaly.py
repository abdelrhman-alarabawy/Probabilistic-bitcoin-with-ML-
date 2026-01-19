from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    ADD_MISSING_FLAGS,
    ANOMALY_PERCENTILES,
    DATA_PATH,
    FIGURES_DIR,
    HOLD_BARS,
    INCLUDE_LIQ,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MOM_NEAR_HIGH,
    MOM_NEAR_LOW,
    MOM_RANGE_LOOKBACK,
    MOM_RANGE_MULT,
    OUTPUTS_DIR,
    REPORT_PATH,
    SIGNAL_PLOT_PERCENTILE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TEST_MONTHS,
    TIMESTAMP_COL,
    WINDOW_CONFIGS,
)
from .data import load_data
from .eval import summarize_trades, simulate_trade
from .features import build_event_features
from .models import (
    compute_thresholds,
    fit_isolation_forest,
    fit_robust_z,
    score_isolation_forest,
    score_robust_z,
)
from .plots import plot_equity_curve, plot_score_hist, plot_signals_timeline
from .rolling import generate_rolling_windows


def _build_close_loc(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_range = prev_high - prev_low
    close_loc = (prev_close - prev_low) / prev_range.replace(0.0, np.nan)
    return close_loc


def _build_range_expand(df: pd.DataFrame) -> pd.Series:
    range_raw = df["high"] - df["low"]
    range_median = range_raw.rolling(MOM_RANGE_LOOKBACK).median().shift(1)
    prev_range = range_raw.shift(1)
    return prev_range > (range_median * MOM_RANGE_MULT)


def run() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(str(DATA_PATH))
    features, base_features, liq_cols = build_event_features(df)

    feature_cols = features.columns.tolist()

    open_px = df["open"].astype(float).to_numpy()
    high_px = df["high"].astype(float).to_numpy()
    low_px = df["low"].astype(float).to_numpy()
    close_px = df["close"].astype(float).to_numpy()

    ret_prev = features["return_1"].to_numpy(dtype=float)
    close_loc = _build_close_loc(df)
    range_expand = _build_range_expand(df)

    score_rows: List[Dict] = []
    threshold_rows: List[Dict] = []
    test_rows: List[pd.DataFrame] = []
    trade_rows: List[Dict] = []

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df,
                TIMESTAMP_COL,
                config,
                min_train_rows=MIN_TRAIN_ROWS,
                min_test_rows=MIN_TEST_ROWS,
            )
        )
        for window in windows:
            train_idx = window.train_idx
            test_idx = window.test_idx

            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]

            train_median = X_train.median()
            X_train = X_train.fillna(train_median)
            X_test = X_test.fillna(train_median)

            rz_model = fit_robust_z(X_train)
            score_train_rz = score_robust_z(rz_model, X_train)
            score_test_rz = score_robust_z(rz_model, X_test)
            thr_rz = compute_thresholds(score_train_rz, ANOMALY_PERCENTILES)

            if_model = fit_isolation_forest(X_train)
            score_train_if = score_isolation_forest(if_model, X_train)
            score_test_if = score_isolation_forest(if_model, X_test)
            thr_if = compute_thresholds(score_train_if, ANOMALY_PERCENTILES)

            score_rows.append(
                {
                    "window_id": window.window_id,
                    "score_robustz": score_test_rz,
                    "score_iforest": score_test_if,
                }
            )
            for pct in ANOMALY_PERCENTILES:
                threshold_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": "robustz",
                        "percentile": pct,
                        "threshold": thr_rz[pct],
                    }
                )
                threshold_rows.append(
                    {
                        "window_id": window.window_id,
                        "model": "iforest",
                        "percentile": pct,
                        "threshold": thr_if[pct],
                    }
                )

            n_test = len(test_idx)
            max_start = n_test - HOLD_BARS
            valid_mask = np.zeros(n_test, dtype=bool)
            if max_start >= 0:
                valid_mask[: max_start + 1] = True

            ret_prev_test = ret_prev[test_idx]
            close_loc_test = close_loc.iloc[test_idx].to_numpy(dtype=float)
            range_expand_test = range_expand.iloc[test_idx].fillna(False).to_numpy(dtype=bool)

            close_near_high = np.isfinite(close_loc_test) & (close_loc_test >= MOM_NEAR_HIGH)
            close_near_low = np.isfinite(close_loc_test) & (close_loc_test <= MOM_NEAR_LOW)

            test_output = pd.DataFrame(
                {
                    "window_id": window.window_id,
                    "timestamp": df.loc[test_idx, TIMESTAMP_COL].values,
                    "score_robustz": score_test_rz,
                    "score_iforest": score_test_if,
                    "thr_robustz_p98": thr_rz.get(98),
                    "thr_robustz_p99": thr_rz.get(99),
                    "thr_iforest_p98": thr_if.get(98),
                    "thr_iforest_p99": thr_if.get(99),
                }
            )

            model_specs = {
                "robustz": (score_test_rz, thr_rz),
                "iforest": (score_test_if, thr_if),
            }
            strategies = ["MR", "MOM"]

            signal_payload: Dict[str, np.ndarray] = {}
            for model_name in model_specs:
                for pct in ANOMALY_PERCENTILES:
                    for strategy in strategies:
                        prefix = f"{strategy.lower()}_{model_name}_p{pct}"
                        signal_payload[f"{prefix}_signal"] = np.zeros(n_test, dtype=int)
                        signal_payload[f"{prefix}_return"] = np.full(n_test, np.nan)
                        signal_payload[f"{prefix}_exit"] = np.array([None] * n_test, dtype=object)
                        signal_payload[f"{prefix}_mae"] = np.full(n_test, np.nan)
                        signal_payload[f"{prefix}_mfe"] = np.full(n_test, np.nan)

            for model_name, (scores, thresholds) in model_specs.items():
                for pct in ANOMALY_PERCENTILES:
                    flag = scores >= thresholds[pct]
                    test_output[f"flag_{model_name}_p{pct}"] = flag.astype(int)

                    flag_trade = flag & valid_mask

                    direction_mr = np.zeros(n_test, dtype=int)
                    direction_mr[(flag_trade) & (ret_prev_test > 0)] = -1
                    direction_mr[(flag_trade) & (ret_prev_test < 0)] = 1

                    direction_mom = np.zeros(n_test, dtype=int)
                    direction_mom[flag_trade & range_expand_test & close_near_high] = 1
                    direction_mom[flag_trade & range_expand_test & close_near_low] = -1

                    for strategy, direction in [("MR", direction_mr), ("MOM", direction_mom)]:
                        prefix = f"{strategy.lower()}_{model_name}_p{pct}"
                        signal_arr = signal_payload[f"{prefix}_signal"]
                        ret_arr = signal_payload[f"{prefix}_return"]
                        exit_arr = signal_payload[f"{prefix}_exit"]
                        mae_arr = signal_payload[f"{prefix}_mae"]
                        mfe_arr = signal_payload[f"{prefix}_mfe"]

                        for pos in np.where(direction != 0)[0]:
                            idx = test_idx[pos]
                            result = simulate_trade(
                                idx,
                                int(direction[pos]),
                                open_px,
                                high_px,
                                low_px,
                                close_px,
                            )
                            if result is None:
                                continue
                            signal_arr[pos] = int(direction[pos])
                            ret_arr[pos] = result.ret
                            exit_arr[pos] = result.exit_type
                            mae_arr[pos] = result.mae
                            mfe_arr[pos] = result.mfe

                            trade_rows.append(
                                {
                                    "window_id": window.window_id,
                                    "timestamp": df.loc[idx, TIMESTAMP_COL],
                                    "model": model_name,
                                    "threshold_pct": pct,
                                    "strategy": strategy,
                                    "direction": int(direction[pos]),
                                    "return": result.ret,
                                    "exit_type": result.exit_type,
                                    "mae": result.mae,
                                    "mfe": result.mfe,
                                    "hold_bars": result.hold_bars,
                                }
                            )

            for key, arr in signal_payload.items():
                test_output[key] = arr

            test_rows.append(test_output)

    anomalies_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    anomalies_path = OUTPUTS_DIR / "anomalies_test.csv"
    anomalies_df.to_csv(anomalies_path, index=False)

    trade_df = pd.DataFrame(trade_rows)
    per_window_metrics = pd.DataFrame()
    if not trade_df.empty:
        grouped = trade_df.groupby(["window_id", "model", "threshold_pct", "strategy"], as_index=False)
        metrics = []
        for (window_id, model, threshold_pct, strategy), group in grouped:
            summary = summarize_trades(group)
            summary.update(
                {
                    "window_id": window_id,
                    "model": model,
                    "threshold_pct": threshold_pct,
                    "strategy": strategy,
                    "signals_per_month": group.shape[0] / TEST_MONTHS,
                }
            )
            metrics.append(summary)
        per_window_metrics = pd.DataFrame(metrics)

    outputs_metrics_path = OUTPUTS_DIR / "per_window_metrics.csv"
    per_window_metrics.to_csv(outputs_metrics_path, index=False)

    aggregate_metrics = pd.DataFrame()
    if not per_window_metrics.empty:
        aggregate_metrics = (
            per_window_metrics.groupby(["model", "threshold_pct", "strategy"], as_index=False)
            .agg(
                n_trades=("n_trades", "sum"),
                win_rate=("win_rate", "mean"),
                avg_return=("avg_return", "mean"),
                median_return=("median_return", "median"),
                mae_mean=("mae_mean", "mean"),
                mfe_mean=("mfe_mean", "mean"),
                max_drawdown=("max_drawdown", "min"),
                cvar_95=("cvar_95", "mean"),
                signals_per_month=("signals_per_month", "mean"),
                win_rate_std=("win_rate", "std"),
                avg_return_std=("avg_return", "std"),
            )
        )

    aggregate_metrics_path = OUTPUTS_DIR / "aggregate_metrics.csv"
    aggregate_metrics.to_csv(aggregate_metrics_path, index=False)

    thresholds_df = pd.DataFrame(threshold_rows)
    thresholds_path = OUTPUTS_DIR / "thresholds_by_window.csv"
    thresholds_df.to_csv(thresholds_path, index=False)

    if score_rows:
        scores_stack = pd.DataFrame(score_rows)
        scores_long = pd.DataFrame(
            {
                "score_robustz": np.concatenate(scores_stack["score_robustz"].values),
                "score_iforest": np.concatenate(scores_stack["score_iforest"].values),
            }
        )
        plot_score_hist(scores_long, str(FIGURES_DIR / "score_hist.png"))

    if not trade_df.empty:
        plot_signals_timeline(trade_df, str(FIGURES_DIR / "signals_timeline.png"), SIGNAL_PLOT_PERCENTILE)
        plot_equity_curve(trade_df, str(FIGURES_DIR / "equity_curve.png"), SIGNAL_PLOT_PERCENTILE)

    report_lines: List[str] = []
    report_lines.append("# Anomaly Trading Report (12h)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append("- Data: 12h_features_indicators_with_ohlcv.csv")
    report_lines.append("- Features shift: t uses data up to t-1 (shift=1)")
    report_lines.append(
        f"- Windows: train={WINDOW_CONFIGS[0].train_months}m, test={WINDOW_CONFIGS[0].test_months}m, "
        f"step={WINDOW_CONFIGS[0].step_months}m"
    )
    report_lines.append(f"- Hold bars: {HOLD_BARS}, TP={TAKE_PROFIT_PCT:.3f}, SL={STOP_LOSS_PCT:.3f}")
    report_lines.append(
        f"- MOM: near_high>={MOM_NEAR_HIGH:.2f}, near_low<={MOM_NEAR_LOW:.2f}, "
        f"range_expand: prev_range > median*{MOM_RANGE_MULT:.2f} (lookback={MOM_RANGE_LOOKBACK})"
    )
    report_lines.append("")

    report_lines.append("## Feature Set")
    report_lines.append(f"- Base features ({len(base_features)}): {', '.join(base_features)}")
    if INCLUDE_LIQ and liq_cols:
        report_lines.append(f"- Liquidity features ({len(liq_cols)}): {', '.join(liq_cols)}")
    else:
        report_lines.append("- Liquidity features: none")
    report_lines.append(f"- Missingness flags: {'on' if ADD_MISSING_FLAGS else 'off'}")
    report_lines.append("")

    report_lines.append("## Thresholds (Median over windows)")
    if thresholds_df.empty:
        report_lines.append("- No thresholds computed.")
    else:
        med_thr = thresholds_df.groupby(["model", "percentile"])["threshold"].median().reset_index()
        report_lines.append("| Model | Percentile | Median threshold |")
        report_lines.append("| --- | --- | --- |")
        for _, row in med_thr.iterrows():
            report_lines.append(
                f"| {row['model']} | {int(row['percentile'])} | {row['threshold']:.4f} |"
            )
    report_lines.append("")

    report_lines.append("## Strategy Metrics (Aggregate)")
    if aggregate_metrics.empty:
        report_lines.append("- No trades were generated.")
    else:
        report_lines.append(
            "| Model | Pct | Strategy | Trades | Signals/mo | Win rate | Avg ret | Med ret | MAE | MFE | Max DD | CVaR95 |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in aggregate_metrics.sort_values(["strategy", "model", "threshold_pct"]).iterrows():
            report_lines.append(
                "| {model} | {pct} | {strategy} | {trades} | {signals:.2f} | {win:.3f} | {avg:.4f} | "
                "{med:.4f} | {mae:.4f} | {mfe:.4f} | {dd:.4f} | {cvar:.4f} |".format(
                    model=row["model"],
                    pct=int(row["threshold_pct"]),
                    strategy=row["strategy"],
                    trades=int(row["n_trades"]),
                    signals=row["signals_per_month"],
                    win=row["win_rate"],
                    avg=row["avg_return"],
                    med=row["median_return"],
                    mae=row["mae_mean"],
                    mfe=row["mfe_mean"],
                    dd=row["max_drawdown"],
                    cvar=row["cvar_95"],
                )
            )
    report_lines.append("")

    report_lines.append("## Stability Across Windows")
    if aggregate_metrics.empty:
        report_lines.append("- No stability metrics available.")
    else:
        report_lines.append("| Model | Pct | Strategy | Win rate std | Avg return std |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for _, row in aggregate_metrics.sort_values(["strategy", "model", "threshold_pct"]).iterrows():
            report_lines.append(
                f"| {row['model']} | {int(row['threshold_pct'])} | {row['strategy']} | "
                f"{row['win_rate_std']:.3f} | {row['avg_return_std']:.4f} |"
            )
    report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append("- figures/score_hist.png")
    report_lines.append("- figures/signals_timeline.png")
    report_lines.append("- figures/equity_curve.png")
    report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("- No labels changed; candle_type is untouched.")
    report_lines.append("- Test windows overlap (step < test length), so trades may repeat across windows.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
