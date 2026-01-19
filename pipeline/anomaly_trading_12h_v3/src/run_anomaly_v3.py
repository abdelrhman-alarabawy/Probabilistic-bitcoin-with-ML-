from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    ADD_MISSING_FLAGS,
    BUCKET4_MODE,
    BUCKET_PCTS,
    DATA_PATH,
    EMA200_GATE_OPTIONS,
    FEATURE_VARIANTS,
    FEE_PER_TRADE,
    FIGURES_DIR,
    HOLD_BARS_GRID,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    OUTPUTS_DIR,
    PCT_LIST,
    RANGE_BUCKETS,
    REPORT_PATH,
    SL_GRID,
    TP_GRID,
    TIMESTAMP_COL,
    WINDOW_CONFIGS,
)
from .data import load_data
from .eval import summarize_trades, simulate_trade
from .features import build_event_features, compute_liq_missing
from .models import compute_thresholds, fit_robust_z, score_robust_z
from .plots import plot_equity_curve, plot_return_distribution, plot_signals_timeline
from .rolling import generate_rolling_windows


def _build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close_prev = df["close"].shift(1)
    ema20 = close_prev.ewm(span=20, adjust=False).mean()
    ema50 = close_prev.ewm(span=50, adjust=False).mean()
    ema200 = close_prev.ewm(span=200, adjust=False).mean()
    range_strength = (ema20 - ema50).abs() / close_prev.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "close_prev": close_prev,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "range_strength": range_strength,
            "regime_long_ok": close_prev > ema200,
            "regime_short_ok": close_prev < ema200,
        }
    )


def _months_covered(timestamps: pd.Series) -> int:
    if timestamps.empty:
        return 1
    months = timestamps.dt.to_period("M").nunique()
    return max(1, int(months))


def _direction_from_return(ret_prev: pd.Series) -> np.ndarray:
    direction = np.zeros(len(ret_prev), dtype=int)
    direction[ret_prev < 0.0] = 1
    direction[ret_prev > 0.0] = -1
    return direction


def _assign_buckets(range_strength: np.ndarray) -> np.ndarray:
    bucket = np.full(len(range_strength), 4, dtype=int)
    cond1 = range_strength <= RANGE_BUCKETS[0]
    cond2 = (range_strength > RANGE_BUCKETS[0]) & (range_strength <= RANGE_BUCKETS[1])
    cond3 = (range_strength > RANGE_BUCKETS[1]) & (range_strength <= RANGE_BUCKETS[2])
    bucket[cond1] = 1
    bucket[cond2] = 2
    bucket[cond3] = 3
    return bucket


def _adaptive_flags(
    scores: np.ndarray,
    thresholds: Dict[int, float],
    buckets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    adaptive_pct = np.full(len(scores), np.nan)
    for idx, pct in enumerate(BUCKET_PCTS, start=1):
        adaptive_pct[buckets == idx] = pct

    adaptive_flag_raw = np.zeros(len(scores), dtype=bool)
    for pct in PCT_LIST:
        mask = adaptive_pct == pct
        if np.any(mask):
            adaptive_flag_raw[mask] = scores[mask] >= thresholds[pct]

    return adaptive_pct, adaptive_flag_raw


def run() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(str(DATA_PATH))
    trend_df = _build_trend_features(df)

    open_px = df["open"].astype(float).to_numpy()
    high_px = df["high"].astype(float).to_numpy()
    low_px = df["low"].astype(float).to_numpy()
    close_px = df["close"].astype(float).to_numpy()

    dedup_outputs: List[pd.DataFrame] = []
    filter_audit_rows: List[Dict] = []
    grid_rows: List[Dict] = []
    best_configs_rows: List[Dict] = []
    percentile_rows: List[Dict] = []

    for variant in FEATURE_VARIANTS:
        include_liq = variant == "with_liq"
        features, base_features, liq_cols, liq_flag_cols = build_event_features(
            df, include_liq=include_liq
        )
        liq_missing = compute_liq_missing(features, liq_cols, liq_flag_cols)

        ret_prev = features["return_1"].astype(float)
        direction_all = _direction_from_return(ret_prev)

        test_rows: List[pd.DataFrame] = []
        raw_signal_counts: Dict[int, int] = {pct: 0 for pct in PCT_LIST}

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
                X_train = X_train.fillna(train_median).fillna(0.0)
                X_test = X_test.fillna(train_median).fillna(0.0)

                rz_model = fit_robust_z(X_train)
                score_train = score_robust_z(rz_model, X_train)
                score_test = score_robust_z(rz_model, X_test)
                thresholds = compute_thresholds(score_train, PCT_LIST)

                test_output = pd.DataFrame(
                    {
                        "window_id": window.window_id,
                        "timestamp": df.loc[test_idx, TIMESTAMP_COL].values,
                        "row_idx": test_idx,
                        "score_robustz": score_test,
                        "direction": direction_all[test_idx],
                        "range_strength": trend_df.loc[test_idx, "range_strength"].values,
                        "close_prev": trend_df.loc[test_idx, "close_prev"].values,
                        "ema20": trend_df.loc[test_idx, "ema20"].values,
                        "ema50": trend_df.loc[test_idx, "ema50"].values,
                        "ema200": trend_df.loc[test_idx, "ema200"].values,
                        "regime_long_ok": trend_df.loc[test_idx, "regime_long_ok"].values,
                        "regime_short_ok": trend_df.loc[test_idx, "regime_short_ok"].values,
                        "liq_missing": liq_missing.iloc[test_idx].values,
                    }
                )

                for pct in PCT_LIST:
                    flag = score_test >= thresholds[pct]
                    test_output[f"flag_p{pct}"] = flag.astype(int)
                    raw_signal_counts[pct] += int(flag.sum())

                buckets = _assign_buckets(test_output["range_strength"].to_numpy(dtype=float))
                adaptive_pct, adaptive_flag_raw = _adaptive_flags(score_test, thresholds, buckets)
                test_output["trend_bucket"] = buckets
                test_output["adaptive_pct"] = adaptive_pct
                test_output["adaptive_flag_raw"] = adaptive_flag_raw.astype(int)
                test_rows.append(test_output)

        all_test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
        if all_test_df.empty:
            raise RuntimeError("No test windows generated; check data coverage.")

        all_test_df = all_test_df.sort_values(["timestamp", "window_id"]).reset_index(drop=True)
        dedup_df = all_test_df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        dedup_df["feature_variant"] = variant

        bucket_skip = (dedup_df["trend_bucket"] == 4) & (BUCKET4_MODE == "skip")
        dedup_df["bucket_skip"] = bucket_skip.astype(int)
        dedup_df["adaptive_flag"] = dedup_df["adaptive_flag_raw"]
        if BUCKET4_MODE == "skip":
            dedup_df.loc[bucket_skip, "adaptive_flag"] = 0

        dedup_outputs.append(dedup_df)

        dedup_months = _months_covered(dedup_df["timestamp"])
        for pct in PCT_LIST:
            flag_col = f"flag_p{pct}"
            if flag_col in dedup_df:
                signals = int(dedup_df[flag_col].sum())
                percentile_rows.append(
                    {
                        "feature_variant": variant,
                        "percentile": pct,
                        "signals": signals,
                        "signals_per_month": signals / float(dedup_months),
                    }
                )

        for ema_gate in EMA200_GATE_OPTIONS:
            base_signal = dedup_df["adaptive_flag_raw"].to_numpy(dtype=bool)
            liq_ok = (~dedup_df["liq_missing"].to_numpy(dtype=bool)) if include_liq else np.ones(
                len(dedup_df), dtype=bool
            )
            bucket_ok = ~bucket_skip.to_numpy(dtype=bool)
            direction = dedup_df["direction"].to_numpy(dtype=int)
            regime_long = dedup_df["regime_long_ok"].to_numpy(dtype=bool)
            regime_short = dedup_df["regime_short_ok"].to_numpy(dtype=bool)
            ema_ok = np.ones(len(dedup_df), dtype=bool)
            if ema_gate:
                ema_ok = ((direction > 0) & regime_long) | ((direction < 0) & regime_short)

            signals_detected = int(base_signal.sum())
            after_liq = int((base_signal & liq_ok).sum())
            after_bucket = int((base_signal & liq_ok & bucket_ok).sum())
            after_ema = int((base_signal & liq_ok & bucket_ok & ema_ok).sum())
            max_hold = max(HOLD_BARS_GRID)
            valid_entry = (dedup_df["row_idx"].to_numpy(dtype=int) + max_hold - 1) < len(open_px)
            executed = int((base_signal & liq_ok & bucket_ok & ema_ok & (direction != 0) & valid_entry).sum())

            filter_audit_rows.append(
                {
                    "feature_variant": variant,
                    "ema200_gate": ema_gate,
                    "signals_detected_dedup": signals_detected,
                    "after_liq_policy": after_liq if include_liq else signals_detected,
                    "after_trend_bucket_skip": after_bucket,
                    "after_ema200_gate": after_ema if ema_gate else after_bucket,
                    "executed_trades": executed,
                }
            )

            for hold_bars in HOLD_BARS_GRID:
                for tp in TP_GRID:
                    for sl in SL_GRID:
                        trade_rows: List[Dict] = []
                        valid_entry = (
                            dedup_df["row_idx"].to_numpy(dtype=int) + hold_bars - 1
                        ) < len(open_px)
                        signal_mask = (
                            base_signal
                            & liq_ok
                            & bucket_ok
                            & ema_ok
                            & (direction != 0)
                            & valid_entry
                        )
                        for pos in np.where(signal_mask)[0]:
                            idx = int(dedup_df.loc[pos, "row_idx"])
                            result = simulate_trade(
                                idx,
                                int(direction[pos]),
                                open_px,
                                high_px,
                                low_px,
                                close_px,
                                hold_bars=hold_bars,
                                tp_pct=tp,
                                sl_pct=sl,
                                fee_per_trade=FEE_PER_TRADE,
                            )
                            if result is None:
                                continue
                            trade_rows.append(
                                {
                                    "timestamp": dedup_df.loc[pos, "timestamp"],
                                    "feature_variant": variant,
                                    "ema200_gate": ema_gate,
                                    "hold_bars": hold_bars,
                                    "tp": tp,
                                    "sl": sl,
                                    "ret_net": result.ret_net,
                                    "ret_gross": result.ret_gross,
                                    "exit_type": result.exit_type,
                                    "mae": result.mae,
                                    "mfe": result.mfe,
                                    "hold_bars_realized": result.hold_bars,
                                }
                            )

                        trades_df = pd.DataFrame(trade_rows)
                        summary = summarize_trades(trades_df)
                        summary.update(
                            {
                                "feature_variant": variant,
                                "ema200_gate": ema_gate,
                                "hold_bars": hold_bars,
                                "tp": tp,
                                "sl": sl,
                                "signals_per_month": summary["n_trades"] / float(dedup_months),
                            }
                        )
                        grid_rows.append(summary)

    dedup_all = pd.concat(dedup_outputs, ignore_index=True)
    dedup_all.to_csv(OUTPUTS_DIR / "oos_dedup_signals.csv", index=False)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUTPUTS_DIR / "grid_results.csv", index=False)

    filter_audit_df = pd.DataFrame(filter_audit_rows)
    filter_audit_df.to_csv(OUTPUTS_DIR / "filter_audit.csv", index=False)

    percentile_df = pd.DataFrame(percentile_rows)

    eligible = grid_df[
        (grid_df["signals_per_month"] >= 1.0)
        & (grid_df["signals_per_month"] <= 2.5)
        & (grid_df["n_trades"] >= 20)
    ]
    if eligible.empty:
        eligible = grid_df[grid_df["n_trades"] > 0]

    ranked = eligible.sort_values(
        ["win_rate", "cvar_95", "max_drawdown"], ascending=[False, False, False]
    )
    top3 = ranked.head(3).copy()
    if not top3.empty:
        best_configs_rows.extend(top3.to_dict(orient="records"))
    pd.DataFrame(best_configs_rows).to_csv(OUTPUTS_DIR / "best_configs.csv", index=False)

    top1 = top3.iloc[0] if not top3.empty else None
    if top1 is not None:
        trades_mask = (
            (grid_df["feature_variant"] == top1["feature_variant"])
            & (grid_df["ema200_gate"] == top1["ema200_gate"])
            & (grid_df["hold_bars"] == top1["hold_bars"])
            & (grid_df["tp"] == top1["tp"])
            & (grid_df["sl"] == top1["sl"])
        )
        selected = grid_df[trades_mask]

        if not selected.empty:
            variant = top1["feature_variant"]
            ema_gate = top1["ema200_gate"]
            dedup_variant = dedup_all[dedup_all["feature_variant"] == variant]

            direction = dedup_variant["direction"].to_numpy(dtype=int)
            liq_ok = (
                ~dedup_variant["liq_missing"].to_numpy(dtype=bool)
                if variant == "with_liq"
                else np.ones(len(dedup_variant), dtype=bool)
            )
            bucket_ok = ~(dedup_variant["bucket_skip"].to_numpy(dtype=bool))
            ema_ok = np.ones(len(dedup_variant), dtype=bool)
            if ema_gate:
                regime_long = dedup_variant["regime_long_ok"].to_numpy(dtype=bool)
                regime_short = dedup_variant["regime_short_ok"].to_numpy(dtype=bool)
                ema_ok = ((direction > 0) & regime_long) | ((direction < 0) & regime_short)

            base_signal = dedup_variant["adaptive_flag_raw"].to_numpy(dtype=bool)
            valid_entry = (
                dedup_variant["row_idx"].to_numpy(dtype=int) + int(top1["hold_bars"]) - 1
            ) < len(open_px)
            signal_mask = (
                base_signal
                & liq_ok
                & bucket_ok
                & ema_ok
                & (direction != 0)
                & valid_entry
            )

            trade_rows = []
            for pos in np.where(signal_mask)[0]:
                idx = int(dedup_variant.iloc[pos]["row_idx"])
                result = simulate_trade(
                    idx,
                    int(direction[pos]),
                    open_px,
                    high_px,
                    low_px,
                    close_px,
                    hold_bars=int(top1["hold_bars"]),
                    tp_pct=float(top1["tp"]),
                    sl_pct=float(top1["sl"]),
                    fee_per_trade=FEE_PER_TRADE,
                )
                if result is None:
                    continue
                trade_rows.append(
                    {
                        "timestamp": dedup_variant.iloc[pos]["timestamp"],
                        "ret_net": result.ret_net,
                    }
                )
            top1_trades = pd.DataFrame(trade_rows)
            if not top1_trades.empty:
                plot_equity_curve(top1_trades, str(FIGURES_DIR / "equity_top1.png"))
                plot_signals_timeline(top1_trades, str(FIGURES_DIR / "signals_timeline_top1.png"))
                plot_return_distribution(top1_trades, str(FIGURES_DIR / "return_dist_top1.png"))

    report_lines: List[str] = []
    report_lines.append("# Anomaly Trading Report (12h, v3)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append("- Data: 12h_features_indicators_with_ohlcv.csv")
    report_lines.append("- Features shift: t uses data up to t-1 (shift=1)")
    report_lines.append(
        f"- Windows: train={WINDOW_CONFIGS[0].train_months}m, test={WINDOW_CONFIGS[0].test_months}m, "
        f"step={WINDOW_CONFIGS[0].step_months}m"
    )
    report_lines.append(f"- Fee per trade: {FEE_PER_TRADE:.4f}")
    report_lines.append(f"- Bucket4 mode: {BUCKET4_MODE}")
    report_lines.append("")

    report_lines.append("## Percentile Sweep (dedup, pre-filters)")
    if percentile_df.empty:
        report_lines.append("- No percentile sweep data.")
    else:
        report_lines.append("| Variant | Percentile | Signals | Signals/mo |")
        report_lines.append("| --- | --- | --- | --- |")
        for _, row in percentile_df.iterrows():
            report_lines.append(
                f"| {row['feature_variant']} | {int(row['percentile'])} | {int(row['signals'])} | "
                f"{row['signals_per_month']:.2f} |"
            )
    report_lines.append("")

    report_lines.append("## Filter Audit")
    if filter_audit_df.empty:
        report_lines.append("- No audit rows.")
    else:
        report_lines.append(
            "| Variant | EMA200 | Signals | After liq | After bucket | After EMA | Executed |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in filter_audit_df.iterrows():
            report_lines.append(
                f"| {row['feature_variant']} | {row['ema200_gate']} | {int(row['signals_detected_dedup'])} | "
                f"{int(row['after_liq_policy'])} | {int(row['after_trend_bucket_skip'])} | "
                f"{int(row['after_ema200_gate'])} | {int(row['executed_trades'])} |"
            )
    report_lines.append("")

    report_lines.append("## Best Configs (under constraints)")
    if top3.empty:
        report_lines.append("- No configs met the signal/trade constraints; showing best available.")
    if not ranked.empty:
        report_lines.append(
            "| Rank | Variant | EMA200 | Hold | TP | SL | Trades | Signals/mo | Win rate | Avg ret | CVaR95 | Max DD |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
            report_lines.append(
                "| {rank} | {variant} | {ema} | {hold} | {tp:.4f} | {sl:.4f} | {trades} | {signals:.2f} | "
                "{win:.3f} | {avg:.4f} | {cvar:.4f} | {dd:.4f} |".format(
                    rank=i,
                    variant=row["feature_variant"],
                    ema=row["ema200_gate"],
                    hold=int(row["hold_bars"]),
                    tp=row["tp"],
                    sl=row["sl"],
                    trades=int(row["n_trades"]),
                    signals=row["signals_per_month"],
                    win=row["win_rate"],
                    avg=row["avg_return"],
                    cvar=row["cvar_95"],
                    dd=row["max_drawdown"],
                )
            )
    report_lines.append("")

    report_lines.append("## Comparison Summary")
    if ranked.empty:
        report_lines.append("- No configs available for comparison.")
    else:
        report_lines.append("| Variant | EMA200 | Best win rate | Trades | Signals/mo |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for variant in FEATURE_VARIANTS:
            for ema_gate in EMA200_GATE_OPTIONS:
                subset = ranked[
                    (ranked["feature_variant"] == variant)
                    & (ranked["ema200_gate"] == ema_gate)
                ]
                if subset.empty:
                    report_lines.append(f"| {variant} | {ema_gate} | n/a | 0 | 0 |")
                    continue
                best = subset.iloc[0]
                report_lines.append(
                    f"| {variant} | {ema_gate} | {best['win_rate']:.3f} | "
                    f"{int(best['n_trades'])} | {best['signals_per_month']:.2f} |"
                )
    report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append("- figures/equity_top1.png")
    report_lines.append("- figures/signals_timeline_top1.png")
    report_lines.append("- figures/return_dist_top1.png")
    report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("- RobustZ only; MOM strategy removed.")
    report_lines.append("- Adaptive thresholds by range_strength buckets.")
    report_lines.append("- Dedup keeps earliest window decision per timestamp.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
