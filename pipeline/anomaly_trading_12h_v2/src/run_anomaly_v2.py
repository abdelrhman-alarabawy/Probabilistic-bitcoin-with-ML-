from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    ADD_MISSING_FLAGS,
    ANOMALY_PERCENTILES,
    DATA_PATH,
    FEE_PER_TRADE,
    FIGURES_DIR,
    FOCUS_PERCENTILE,
    HOLD_BARS_GRID,
    INCLUDE_LIQ,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    OUTPUTS_DIR,
    RANGE_THRESHOLD_GRID,
    REPORT_PATH,
    SL_GRID,
    TP_GRID,
    TIMESTAMP_COL,
    WINDOW_CONFIGS,
)
from .data import load_data
from .eval import summarize_trades, simulate_trade
from .features import build_event_features
from .models import compute_thresholds, fit_robust_z, score_robust_z
from .plots import plot_equity_curve, plot_return_distribution, plot_signals_timeline
from .rolling import generate_rolling_windows


def _build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close_prev = df["close"].shift(1)
    ema20 = close_prev.ewm(span=20, adjust=False).mean()
    ema50 = close_prev.ewm(span=50, adjust=False).mean()
    ema200 = close_prev.ewm(span=200, adjust=False).mean()
    trend_strength = (ema20 - ema50).abs() / close_prev.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "close_prev": close_prev,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "trend_strength": trend_strength,
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


def run() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(str(DATA_PATH))
    features, base_features, liq_cols = build_event_features(df)

    trend_df = _build_trend_features(df)

    open_px = df["open"].astype(float).to_numpy()
    high_px = df["high"].astype(float).to_numpy()
    low_px = df["low"].astype(float).to_numpy()
    close_px = df["close"].astype(float).to_numpy()

    ret_prev = features["return_1"].astype(float)
    direction_all = _direction_from_return(ret_prev)

    test_rows: List[pd.DataFrame] = []
    raw_signal_counts: Dict[int, int] = {pct: 0 for pct in ANOMALY_PERCENTILES}

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
            thresholds = compute_thresholds(score_train, ANOMALY_PERCENTILES)

            test_output = pd.DataFrame(
                {
                    "window_id": window.window_id,
                    "timestamp": df.loc[test_idx, TIMESTAMP_COL].values,
                    "row_idx": test_idx,
                    "score_robustz": score_test,
                    "thr_p98": thresholds.get(98),
                    "thr_p99": thresholds.get(99),
                    "direction": direction_all[test_idx],
                    "trend_strength": trend_df.loc[test_idx, "trend_strength"].values,
                    "close_prev": trend_df.loc[test_idx, "close_prev"].values,
                    "ema20": trend_df.loc[test_idx, "ema20"].values,
                    "ema50": trend_df.loc[test_idx, "ema50"].values,
                    "ema200": trend_df.loc[test_idx, "ema200"].values,
                    "regime_long_ok": trend_df.loc[test_idx, "regime_long_ok"].values,
                    "regime_short_ok": trend_df.loc[test_idx, "regime_short_ok"].values,
                }
            )

            for pct in ANOMALY_PERCENTILES:
                flag = score_test >= thresholds[pct]
                test_output[f"flag_p{pct}"] = flag.astype(int)
                raw_signal_counts[pct] += int(flag.sum())

            test_rows.append(test_output)

    all_test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    if all_test_df.empty:
        raise RuntimeError("No test windows generated; check data coverage.")

    all_test_df = all_test_df.sort_values(["timestamp", "window_id"]).reset_index(drop=True)
    dedup_df = all_test_df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    for pct in ANOMALY_PERCENTILES:
        flag_col = f"flag_p{pct}"
        if flag_col not in dedup_df.columns:
            dedup_df[flag_col] = 0

    dedup_path = OUTPUTS_DIR / "oos_dedup_signals.csv"
    dedup_df.to_csv(dedup_path, index=False)

    dedup_months = _months_covered(dedup_df["timestamp"])

    config_records: List[Dict] = []
    for pct in ANOMALY_PERCENTILES:
        for hold_bars in HOLD_BARS_GRID:
            for tp in TP_GRID:
                for sl in SL_GRID:
                    for range_thr in RANGE_THRESHOLD_GRID:
                        config_records.append(
                            {
                                "percentile": pct,
                                "hold_bars": hold_bars,
                                "tp": tp,
                                "sl": sl,
                                "range_threshold": range_thr,
                            }
                        )

    trade_rows: List[Dict] = []
    for record in config_records:
        pct = record["percentile"]
        hold_bars = record["hold_bars"]
        tp = record["tp"]
        sl = record["sl"]
        range_thr = record["range_threshold"]

        flag = dedup_df[f"flag_p{pct}"] == 1
        direction = dedup_df["direction"].to_numpy(dtype=int)
        trend_strength = dedup_df["trend_strength"].to_numpy(dtype=float)
        regime_long = dedup_df["regime_long_ok"].to_numpy(dtype=bool)
        regime_short = dedup_df["regime_short_ok"].to_numpy(dtype=bool)
        row_idx = dedup_df["row_idx"].to_numpy(dtype=int)

        range_ok = trend_strength <= range_thr
        regime_ok = (direction > 0) & regime_long | (direction < 0) & regime_short
        signal_mask = flag & range_ok & regime_ok & (direction != 0)

        for pos in np.where(signal_mask)[0]:
            idx = int(row_idx[pos])
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
                    "percentile": pct,
                    "hold_bars": hold_bars,
                    "tp": tp,
                    "sl": sl,
                    "range_threshold": range_thr,
                    "direction": int(direction[pos]),
                    "ret_net": result.ret_net,
                    "ret_gross": result.ret_gross,
                    "exit_type": result.exit_type,
                    "mae": result.mae,
                    "mfe": result.mfe,
                    "hold_bars_realized": result.hold_bars,
                }
            )

    trade_df = pd.DataFrame(trade_rows)

    metrics_map: Dict[Tuple, Dict] = {}
    if not trade_df.empty:
        for key, group in trade_df.groupby(["percentile", "hold_bars", "tp", "sl", "range_threshold"]):
            metrics_map[key] = summarize_trades(group)

    grid_rows: List[Dict] = []
    for record in config_records:
        key = (
            record["percentile"],
            record["hold_bars"],
            record["tp"],
            record["sl"],
            record["range_threshold"],
        )
        summary = metrics_map.get(key)
        if summary is None:
            summary = summarize_trades(pd.DataFrame())
        record = record.copy()
        record.update(summary)
        record["signals_per_month"] = record["n_trades"] / float(dedup_months)
        grid_rows.append(record)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUTPUTS_DIR / "grid_results.csv", index=False)

    best_rows: List[Dict] = []
    if not grid_df.empty:
        candidates = grid_df[grid_df["n_trades"] > 0].copy()
        if not candidates.empty:
            top_win = candidates.sort_values("win_rate", ascending=False).head(10)
            top_win = top_win.assign(rank_group="win_rate")
            best_rows.extend(top_win.to_dict(orient="records"))

            top_ret = candidates.sort_values("avg_return", ascending=False).head(10)
            top_ret = top_ret.assign(rank_group="avg_return")
            best_rows.extend(top_ret.to_dict(orient="records"))

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUTPUTS_DIR / "best_configs.csv", index=False)

    best_high_precision = None
    best_expectancy = None
    best_hp_source = f"p{FOCUS_PERCENTILE}"
    best_exp_source = f"p{FOCUS_PERCENTILE}"
    rare_constraints_met = True
    exp_constraints_met = True
    if not grid_df.empty:
        focus = grid_df[grid_df["percentile"] == FOCUS_PERCENTILE]
        rare = focus[(focus["signals_per_month"] <= 2.0) & (focus["n_trades"] >= 15)]
        if not rare.empty:
            best_high_precision = rare.sort_values("win_rate", ascending=False).iloc[0]
        else:
            rare_constraints_met = False
            fallback = focus[focus["n_trades"] > 0].sort_values("win_rate", ascending=False)
            if not fallback.empty:
                best_high_precision = fallback.iloc[0]
            else:
                alt = grid_df[(grid_df["signals_per_month"] <= 2.0) & (grid_df["n_trades"] >= 15)]
                if not alt.empty:
                    best_high_precision = alt.sort_values("win_rate", ascending=False).iloc[0]
                    best_hp_source = f"p{int(best_high_precision['percentile'])}"
                else:
                    alt = grid_df[grid_df["n_trades"] > 0].sort_values("win_rate", ascending=False)
                    if not alt.empty:
                        best_high_precision = alt.iloc[0]
                        best_hp_source = f"p{int(best_high_precision['percentile'])}"

        expectancy = focus[focus["n_trades"] >= 30]
        if not expectancy.empty:
            best_expectancy = expectancy.sort_values("avg_return", ascending=False).iloc[0]
        else:
            exp_constraints_met = False
            fallback = focus[focus["n_trades"] > 0].sort_values("avg_return", ascending=False)
            if not fallback.empty:
                best_expectancy = fallback.iloc[0]
            else:
                alt = grid_df[grid_df["n_trades"] >= 30]
                if not alt.empty:
                    best_expectancy = alt.sort_values("avg_return", ascending=False).iloc[0]
                    best_exp_source = f"p{int(best_expectancy['percentile'])}"
                else:
                    alt = grid_df[grid_df["n_trades"] > 0].sort_values("avg_return", ascending=False)
                    if not alt.empty:
                        best_expectancy = alt.iloc[0]
                        best_exp_source = f"p{int(best_expectancy['percentile'])}"

    best_trade_df = pd.DataFrame()
    if best_expectancy is not None and not trade_df.empty:
        best_trade_df = trade_df[
            (trade_df["percentile"] == best_expectancy["percentile"])
            & (trade_df["hold_bars"] == best_expectancy["hold_bars"])
            & (trade_df["tp"] == best_expectancy["tp"])
            & (trade_df["sl"] == best_expectancy["sl"])
            & (trade_df["range_threshold"] == best_expectancy["range_threshold"])
        ]

    if not best_trade_df.empty:
        plot_equity_curve(best_trade_df, str(FIGURES_DIR / "equity_best.png"))
        plot_signals_timeline(best_trade_df, str(FIGURES_DIR / "signals_best.png"))

    if best_high_precision is not None and not trade_df.empty:
        hp_trade_df = trade_df[
            (trade_df["percentile"] == best_high_precision["percentile"])
            & (trade_df["hold_bars"] == best_high_precision["hold_bars"])
            & (trade_df["tp"] == best_high_precision["tp"])
            & (trade_df["sl"] == best_high_precision["sl"])
            & (trade_df["range_threshold"] == best_high_precision["range_threshold"])
        ]
        if not hp_trade_df.empty:
            plot_return_distribution(hp_trade_df, str(FIGURES_DIR / "return_distribution.png"))

    dedup_counts = {
        pct: int(dedup_df[f"flag_p{pct}"].sum()) for pct in ANOMALY_PERCENTILES
    }

    report_lines: List[str] = []
    report_lines.append("# Anomaly Trading Report (12h, v2)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append("- Data: 12h_features_indicators_with_ohlcv.csv")
    report_lines.append("- Features shift: t uses data up to t-1 (shift=1)")
    report_lines.append(
        f"- Windows: train={WINDOW_CONFIGS[0].train_months}m, test={WINDOW_CONFIGS[0].test_months}m, "
        f"step={WINDOW_CONFIGS[0].step_months}m"
    )
    report_lines.append(f"- Fee per trade: {FEE_PER_TRADE:.4f}")
    report_lines.append("")

    report_lines.append("## Dedup Summary")
    report_lines.append(f"- Raw test rows: {len(all_test_df)}")
    report_lines.append(f"- Dedup test rows: {len(dedup_df)}")
    for pct in ANOMALY_PERCENTILES:
        raw = raw_signal_counts[pct]
        dedup = dedup_counts[pct]
        report_lines.append(
            f"- p{pct} raw signals: {raw}, dedup signals: {dedup}, duplicates removed: {raw - dedup}"
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

    report_lines.append("## Best Configs (focus p99)")
    p99_trades = int(grid_df[grid_df["percentile"] == FOCUS_PERCENTILE]["n_trades"].sum()) if not grid_df.empty else 0
    report_lines.append(f"- p{FOCUS_PERCENTILE} total trades after filters: {p99_trades}")
    if best_high_precision is None:
        report_lines.append("- No config meets the rare high-precision constraints.")
    else:
        report_lines.append(
            f"- Rare high precision ({best_hp_source}, win rate max, signals/mo<=2, trades>=15): "
            f"hold={int(best_high_precision['hold_bars'])}, tp={best_high_precision['tp']:.4f}, "
            f"sl={best_high_precision['sl']:.4f}, range_thr={best_high_precision['range_threshold']:.4f}, "
            f"win_rate={best_high_precision['win_rate']:.3f}, avg_ret={best_high_precision['avg_return']:.4f}, "
            f"trades={int(best_high_precision['n_trades'])}, signals/mo={best_high_precision['signals_per_month']:.2f}"
        )
        if not rare_constraints_met:
            report_lines.append("- Rare high-precision constraints not met; showing best available config.")
    if best_expectancy is None:
        report_lines.append("- No config meets the best-expectancy constraints.")
    else:
        report_lines.append(
            f"- Best expectancy ({best_exp_source}, avg return max, trades>=30): "
            f"hold={int(best_expectancy['hold_bars'])}, tp={best_expectancy['tp']:.4f}, "
            f"sl={best_expectancy['sl']:.4f}, range_thr={best_expectancy['range_threshold']:.4f}, "
            f"win_rate={best_expectancy['win_rate']:.3f}, avg_ret={best_expectancy['avg_return']:.4f}, "
            f"trades={int(best_expectancy['n_trades'])}, signals/mo={best_expectancy['signals_per_month']:.2f}"
        )
        if not exp_constraints_met:
            report_lines.append("- Best-expectancy constraints not met; showing best available config.")
    report_lines.append("")

    report_lines.append("## Confusion-Style Summary")
    if best_high_precision is not None:
        report_lines.append("- Rare high precision (wins/losses)")
        report_lines.append(
            f"  wins={int(best_high_precision['wins'])}, losses={int(best_high_precision['losses'])}"
        )
    if best_expectancy is not None:
        report_lines.append("- Best expectancy (wins/losses)")
        report_lines.append(
            f"  wins={int(best_expectancy['wins'])}, losses={int(best_expectancy['losses'])}"
        )
    report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append("- figures/equity_best.png")
    report_lines.append("- figures/signals_best.png")
    report_lines.append("- figures/return_distribution.png")
    report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("- RobustZ only; MOM strategy removed.")
    report_lines.append("- Trend filters: range_strength + EMA200 regime gate.")
    report_lines.append("- Dedup keeps earliest window decision per timestamp.")
    if p99_trades == 0:
        report_lines.append("- p99 produced zero trades after filters; best configs fall back to other percentiles.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
