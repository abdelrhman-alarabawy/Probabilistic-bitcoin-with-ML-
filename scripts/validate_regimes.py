from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# === CONFIGURATION ===
INPUT_CSV = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv"
)
PREDS_CSV_CANDIDATES = [
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_per_row.csv"
    ),
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\preds_per_row.csv"
    ),
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_regimes_per_row.csv"
    ),
]

OUT_DIR = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\validation"
)

TP_POINTS = 2000
SL_POINTS = 1000
FEE_PER_TRADE = 0.0005
THRESHOLDS = [0.80, 0.90, 0.95, 0.98, 0.99]


META_COLS = ["timestamp", "open", "high", "low", "close"]


def resolve_input_path(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Preds CSV not found. Tried: {[str(p) for p in candidates]}")


def parse_timestamp(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return parse_numeric_timestamp(series)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().mean() > 0.05:
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().any():
            parsed = parse_numeric_timestamp(as_num)
    return parsed


def parse_numeric_timestamp(series: pd.Series) -> pd.Series:
    max_val = series.max()
    if max_val > 1e12:
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    if max_val > 1e10:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".6f", index: bool = True) -> str:
    def format_value(value: object) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return format(value, floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    if index:
        index_name = df.index.name if df.index.name else "index"
        columns = [index_name] + [str(col) for col in df.columns]
        rows = []
        for idx, row in df.iterrows():
            rows.append([format_value(idx)] + [format_value(v) for v in row.tolist()])
    else:
        columns = [str(col) for col in df.columns]
        rows = [[format_value(v) for v in row] for row in df.values.tolist()]

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compute_run_lengths(regimes: np.ndarray) -> Tuple[List[int], Dict[int, List[int]]]:
    run_lengths: List[int] = []
    per_regime: Dict[int, List[int]] = {}
    if len(regimes) == 0:
        return run_lengths, per_regime
    current = regimes[0]
    length = 1
    for regime in regimes[1:]:
        if regime == current:
            length += 1
        else:
            run_lengths.append(length)
            per_regime.setdefault(current, []).append(length)
            current = regime
            length = 1
    run_lengths.append(length)
    per_regime.setdefault(current, []).append(length)
    return run_lengths, per_regime


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if pnl.empty:
        return np.nan
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return np.nan
    cum = pnl.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    return float(drawdown.min())


def cvar_95(pnl: pd.Series) -> float:
    if pnl.empty:
        return np.nan
    q = pnl.quantile(0.05)
    tail = pnl[pnl <= q]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    preds_path = resolve_input_path(PREDS_CSV_CANDIDATES)
    print(f"Using preds CSV: {preds_path}")

    df_main = pd.read_csv(INPUT_CSV)
    df_main.columns = df_main.columns.str.strip()
    df_preds = pd.read_csv(preds_path)
    df_preds.columns = df_preds.columns.str.strip()

    ts_col_main = "timestamp" if "timestamp" in df_main.columns else None
    if ts_col_main is None:
        raise ValueError("timestamp column missing from main dataset.")

    ts_col_preds = "timestamp" if "timestamp" in df_preds.columns else None
    if ts_col_preds is None:
        raise ValueError("timestamp column missing from preds dataset.")

    df_main["timestamp"] = parse_timestamp(df_main["timestamp"])
    df_preds["timestamp"] = parse_timestamp(df_preds["timestamp"])

    df_main = df_main[df_main["timestamp"].notna()].copy()
    df_preds = df_preds[df_preds["timestamp"].notna()].copy()

    for col in META_COLS[1:]:
        if col not in df_main.columns:
            raise ValueError(f"Missing required column in main dataset: {col}")
        df_main[col] = pd.to_numeric(df_main[col], errors="coerce")

    # Normalize prediction columns
    if "regime" not in df_preds.columns and "regime_id" in df_preds.columns:
        df_preds = df_preds.rename(columns={"regime_id": "regime"})
    if "regime_prob_max" not in df_preds.columns and "prob_max" in df_preds.columns:
        df_preds = df_preds.rename(columns={"prob_max": "regime_prob_max"})

    missing_preds = [col for col in ["regime", "regime_prob_max"] if col not in df_preds.columns]
    if missing_preds:
        raise ValueError(f"Missing required prediction columns: {missing_preds}")

    df = pd.merge(
        df_main,
        df_preds[["timestamp", "regime", "regime_prob_max"] + [c for c in ["entropy", "candle_type"] if c in df_preds.columns]],
        on="timestamp",
        how="inner",
    )

    if df.empty:
        raise ValueError("Merged dataset is empty. Check timestamp alignment.")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df = df[df["regime"].notna()].copy()
    df["regime"] = df["regime"].astype(int)
    df["regime_prob_max"] = pd.to_numeric(df["regime_prob_max"], errors="coerce")
    if "entropy" in df.columns:
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    candle_type_missing = "candle_type" not in df.columns
    if candle_type_missing:
        df["candle_type"] = "skip"

    df["candle_type"] = df["candle_type"].astype(str).str.lower().str.strip()
    df.loc[~df["candle_type"].isin(["long", "short", "skip"]), "candle_type"] = "skip"

    open_px = df["open"].to_numpy()
    high_px = df["high"].to_numpy()
    low_px = df["low"].to_numpy()

    multiplier = open_px / 100000.0
    long_tp = open_px + TP_POINTS * multiplier
    long_sl = open_px - SL_POINTS * multiplier
    short_tp = open_px - TP_POINTS * multiplier
    short_sl = open_px + SL_POINTS * multiplier

    both_touched = (high_px >= long_tp) & (low_px <= short_tp)

    if candle_type_missing:
        long_cond = (high_px >= long_tp) & (low_px > long_sl)
        short_cond = (low_px <= short_tp) & (high_px < short_sl)
        computed = np.where(both_touched, "skip", np.where(long_cond, "long", np.where(short_cond, "short", "skip")))
        df["candle_type"] = computed

    long_tp_hit = (high_px >= long_tp) & (low_px > long_sl)
    long_sl_hit = (low_px <= long_sl) & (high_px < long_tp)
    long_ambig = (high_px >= long_tp) & (low_px <= long_sl)
    short_tp_hit = (low_px <= short_tp) & (high_px < short_sl)
    short_sl_hit = (high_px >= short_sl) & (low_px > short_tp)
    short_ambig = (low_px <= short_tp) & (high_px >= short_sl)

    outcome = np.full(len(df), "SKIP", dtype=object)
    long_outcome = np.where(
        long_tp_hit, "TP", np.where(long_sl_hit, "SL", np.where(long_ambig, "AMBIG", "NONE"))
    )
    short_outcome = np.where(
        short_tp_hit, "TP", np.where(short_sl_hit, "SL", np.where(short_ambig, "AMBIG", "NONE"))
    )
    outcome = np.where(df["candle_type"] == "long", long_outcome, outcome)
    outcome = np.where(df["candle_type"] == "short", short_outcome, outcome)

    if candle_type_missing:
        outcome[both_touched] = "AMBIG_BOTH"

    df["outcome"] = outcome

    pnl = np.full(len(df), np.nan, dtype=float)
    long_tp_pnl = (long_tp / open_px) - 1.0
    long_sl_pnl = (long_sl / open_px) - 1.0
    short_tp_pnl = (open_px / short_tp) - 1.0
    short_sl_pnl = (open_px / short_sl) - 1.0

    long_tp_mask = (df["candle_type"] == "long") & (df["outcome"] == "TP")
    long_sl_mask = (df["candle_type"] == "long") & (df["outcome"] == "SL")
    short_tp_mask = (df["candle_type"] == "short") & (df["outcome"] == "TP")
    short_sl_mask = (df["candle_type"] == "short") & (df["outcome"] == "SL")

    pnl[long_tp_mask] = long_tp_pnl[long_tp_mask] - FEE_PER_TRADE
    pnl[long_sl_mask] = long_sl_pnl[long_sl_mask] - FEE_PER_TRADE
    pnl[short_tp_mask] = short_tp_pnl[short_tp_mask] - FEE_PER_TRADE
    pnl[short_sl_mask] = short_sl_pnl[short_sl_mask] - FEE_PER_TRADE

    df["pnl"] = pnl
    df["r_next"] = (df["close"].shift(-1) / df["close"]) - 1.0

    regime_ids = df["regime"].to_numpy()
    n_regimes = int(regime_ids.max()) + 1

    transition = pd.crosstab(
        regime_ids[:-1], regime_ids[1:], rownames=["from"], colnames=["to"], dropna=False
    ).reindex(index=range(n_regimes), columns=range(n_regimes), fill_value=0)

    run_lengths, per_regime_runs = compute_run_lengths(regime_ids)
    run_lengths = run_lengths if run_lengths else [0]

    summary_rows = []
    for regime in range(n_regimes):
        sub = df[df["regime"] == regime]
        n = len(sub)
        share = n / len(df) if len(df) > 0 else np.nan

        avg_probmax = sub["regime_prob_max"].mean()
        p_prob_ge_0_9 = float((sub["regime_prob_max"] >= 0.9).mean()) if n > 0 else np.nan
        avg_entropy = sub["entropy"].mean() if "entropy" in sub.columns else np.nan

        runs = per_regime_runs.get(regime, [])
        avg_run = float(np.mean(runs)) if runs else np.nan
        median_run = float(np.median(runs)) if runs else np.nan

        trans_row = transition.loc[regime] if regime in transition.index else None
        row_sum = trans_row.sum() if trans_row is not None else 0
        if row_sum > 0:
            p_stay = float(trans_row[regime] / row_sum)
            next_candidates = trans_row.copy()
            next_candidates.loc[regime] = -1
            top_next = int(next_candidates.idxmax())
            p_top_next = float(trans_row[top_next] / row_sum)
        else:
            p_stay = np.nan
            top_next = np.nan
            p_top_next = np.nan

        tradeable_rate = float((sub["candle_type"] != "skip").mean()) if n > 0 else np.nan
        long_rate = float((sub["candle_type"] == "long").mean()) if n > 0 else np.nan
        short_rate = float((sub["candle_type"] == "short").mean()) if n > 0 else np.nan

        r_next = sub["r_next"].dropna()
        mu_next = r_next.mean() if not r_next.empty else np.nan
        median_next = r_next.median() if not r_next.empty else np.nan
        std_next = r_next.std(ddof=0) if not r_next.empty else np.nan
        sharpe_like_next = mu_next / std_next if std_next and std_next > 0 else np.nan
        winrate_next = float((r_next > 0).mean()) if not r_next.empty else np.nan

        n_long = int((sub["candle_type"] == "long").sum())
        n_short = int((sub["candle_type"] == "short").sum())

        long_outcomes = sub[sub["candle_type"] == "long"]["outcome"]
        short_outcomes = sub[sub["candle_type"] == "short"]["outcome"]

        tp_rate_long = float((long_outcomes == "TP").mean()) if n_long > 0 else np.nan
        sl_rate_long = float((long_outcomes == "SL").mean()) if n_long > 0 else np.nan
        ambig_rate_long = float((long_outcomes.str.contains("AMBIG")).mean()) if n_long > 0 else np.nan

        tp_rate_short = float((short_outcomes == "TP").mean()) if n_short > 0 else np.nan
        sl_rate_short = float((short_outcomes == "SL").mean()) if n_short > 0 else np.nan
        ambig_rate_short = float((short_outcomes.str.contains("AMBIG")).mean()) if n_short > 0 else np.nan

        long_trades = sub[(sub["candle_type"] == "long") & (sub["outcome"].isin(["TP", "SL"]))]["pnl"].dropna()
        short_trades = sub[(sub["candle_type"] == "short") & (sub["outcome"].isin(["TP", "SL"]))]["pnl"].dropna()
        all_trades = sub[sub["outcome"].isin(["TP", "SL"])]["pnl"].dropna()

        avg_pnl_long = long_trades.mean() if not long_trades.empty else np.nan
        win_rate_long = float((long_trades > 0).mean()) if not long_trades.empty else np.nan
        profit_factor_long = profit_factor(long_trades)

        avg_pnl_short = short_trades.mean() if not short_trades.empty else np.nan
        win_rate_short = float((short_trades > 0).mean()) if not short_trades.empty else np.nan
        profit_factor_short = profit_factor(short_trades)

        avg_pnl_all = all_trades.mean() if not all_trades.empty else np.nan
        win_rate_all = float((all_trades > 0).mean()) if not all_trades.empty else np.nan
        profit_factor_all = profit_factor(all_trades)

        pnl_series = sub["pnl"].copy()
        pnl_series = pnl_series.fillna(0.0)
        max_dd_all = max_drawdown(pnl_series)
        cvar95_all = cvar_95(all_trades)

        if np.isnan(avg_run) or np.isnan(p_stay) or avg_run < 2 or (p_stay < 0.4):
            decision_regime_type = "transition_noise"
        elif avg_run >= 8 and p_stay >= 0.7:
            decision_regime_type = "stable"
        else:
            decision_regime_type = "semi_stable"

        decision_tradable = (
            n >= 200
            and tradeable_rate is not np.nan
            and avg_run is not np.nan
            and tradeable_rate >= 0.05
            and avg_run >= 3
        )

        decision_side = "none"
        if (
            profit_factor_long is not np.nan
            and profit_factor_long > 1.05
            and n_long >= 50
            and (profit_factor_short <= 1.05 or np.isnan(profit_factor_short))
        ):
            decision_side = "long"
        if (
            profit_factor_short is not np.nan
            and profit_factor_short > 1.05
            and n_short >= 50
            and (profit_factor_long <= 1.05 or np.isnan(profit_factor_long))
        ):
            decision_side = "short"
        if (
            profit_factor_long is not np.nan
            and profit_factor_short is not np.nan
            and profit_factor_long > 1.05
            and profit_factor_short > 1.05
            and n_long >= 50
            and n_short >= 50
        ):
            decision_side = "both"

        summary_rows.append(
            {
                "regime": regime,
                "n": n,
                "share": share,
                "avg_probmax": avg_probmax,
                "p_prob_ge_0_9": p_prob_ge_0_9,
                "avg_entropy": avg_entropy,
                "avg_run_len": avg_run,
                "median_run_len": median_run,
                "p_stay": p_stay,
                "top_next": top_next,
                "p_top_next": p_top_next,
                "tradeable_rate": tradeable_rate,
                "long_rate": long_rate,
                "short_rate": short_rate,
                "mu_next": mu_next,
                "median_next": median_next,
                "std_next": std_next,
                "sharpe_like_next": sharpe_like_next,
                "winrate_next": winrate_next,
                "n_long": n_long,
                "tp_rate_long": tp_rate_long,
                "sl_rate_long": sl_rate_long,
                "ambig_rate_long": ambig_rate_long,
                "n_short": n_short,
                "tp_rate_short": tp_rate_short,
                "sl_rate_short": sl_rate_short,
                "ambig_rate_short": ambig_rate_short,
                "avg_pnl_long": avg_pnl_long,
                "win_rate_long": win_rate_long,
                "profit_factor_long": profit_factor_long,
                "avg_pnl_short": avg_pnl_short,
                "win_rate_short": win_rate_short,
                "profit_factor_short": profit_factor_short,
                "avg_pnl_all": avg_pnl_all,
                "win_rate_all": win_rate_all,
                "profit_factor_all": profit_factor_all,
                "max_drawdown_all": max_dd_all,
                "cvar95_all": cvar95_all,
                "decision_regime_type": decision_regime_type,
                "decision_tradable": bool(decision_tradable),
                "decision_side": decision_side,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "regime_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Threshold sweep
    threshold_rows = []
    for regime in range(n_regimes):
        regime_mask = df["regime"] == regime
        regime_ts = df.loc[regime_mask, "timestamp"]
        if regime_ts.empty:
            months_covered = np.nan
        else:
            delta_days = (regime_ts.max() - regime_ts.min()).total_seconds() / 86400.0
            months_covered = max(delta_days / 30.44, 1.0) if delta_days > 0 else 1.0

        for thr in THRESHOLDS:
            filt = (
                regime_mask
                & (df["regime_prob_max"] >= thr)
                & (df["candle_type"].isin(["long", "short"]))
                & (df["outcome"].isin(["TP", "SL"]))
            )
            trades = df.loc[filt, ["timestamp", "pnl"]].dropna()
            n_trades = len(trades)
            win_rate = float((trades["pnl"] > 0).mean()) if n_trades > 0 else np.nan
            pf = profit_factor(trades["pnl"])
            avg_pnl = trades["pnl"].mean() if n_trades > 0 else np.nan
            median_pnl = trades["pnl"].median() if n_trades > 0 else np.nan
            max_dd = max_drawdown(trades.sort_values("timestamp")["pnl"]) if n_trades > 0 else np.nan
            cvar = cvar_95(trades["pnl"]) if n_trades > 0 else np.nan
            trades_per_month = n_trades / months_covered if months_covered and months_covered > 0 else np.nan

            threshold_rows.append(
                {
                    "regime": regime,
                    "threshold": thr,
                    "n_trades": n_trades,
                    "win_rate": win_rate,
                    "profit_factor": pf,
                    "avg_pnl": avg_pnl,
                    "median_pnl": median_pnl,
                    "max_drawdown": max_dd,
                    "cvar95": cvar,
                    "trades_per_month": trades_per_month,
                }
            )

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = OUT_DIR / "regime_threshold_sweep.csv"
    threshold_df.to_csv(threshold_path, index=False)

    # Side breakdown
    side_rows = []
    for regime in range(n_regimes):
        for side in ["long", "short"]:
            filt = (
                (df["regime"] == regime)
                & (df["candle_type"] == side)
                & (df["outcome"].isin(["TP", "SL"]))
            )
            trades = df.loc[filt, ["timestamp", "pnl"]].dropna()
            n_trades = len(trades)
            win_rate = float((trades["pnl"] > 0).mean()) if n_trades > 0 else np.nan
            pf = profit_factor(trades["pnl"])
            avg_pnl = trades["pnl"].mean() if n_trades > 0 else np.nan
            median_pnl = trades["pnl"].median() if n_trades > 0 else np.nan
            max_dd = max_drawdown(trades.sort_values("timestamp")["pnl"]) if n_trades > 0 else np.nan
            cvar = cvar_95(trades["pnl"]) if n_trades > 0 else np.nan

            side_rows.append(
                {
                    "regime": regime,
                    "side": side,
                    "n_trades": n_trades,
                    "win_rate": win_rate,
                    "profit_factor": pf,
                    "avg_pnl": avg_pnl,
                    "median_pnl": median_pnl,
                    "cvar95": cvar,
                    "max_drawdown": max_dd,
                }
            )

    side_df = pd.DataFrame(side_rows)
    side_path = OUT_DIR / "regime_side_breakdown.csv"
    side_df.to_csv(side_path, index=False)

    # Figures (optional)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(transition.values, cmap="Blues")
    ax.set_title("Regime transition counts")
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_xticks(range(n_regimes))
    ax.set_yticks(range(n_regimes))
    for i in range(n_regimes):
        for j in range(n_regimes):
            ax.text(j, i, str(int(transition.iloc[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_transition_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(run_lengths, bins=30, color="#feb24c", alpha=0.8)
    ax.set_title("Run length distribution")
    ax.set_xlabel("run_length")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "run_length_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["regime_prob_max"].dropna(), bins=30, color="#2c7fb8", alpha=0.8)
    ax.set_title("Max posterior probability")
    ax.set_xlabel("regime_prob_max")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "probmax_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    box_data = []
    labels = []
    for regime in range(n_regimes):
        for side in ["long", "short"]:
            trades = df[
                (df["regime"] == regime)
                & (df["candle_type"] == side)
                & (df["outcome"].isin(["TP", "SL"]))
            ]["pnl"].dropna()
            if trades.empty:
                continue
            box_data.append(trades.values)
            labels.append(f"{regime}-{side[0]}")
    if box_data:
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_title("PNL by regime and side")
        ax.set_xlabel("regime-side")
        ax.set_ylabel("pnl")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "pnl_boxplot_by_regime_and_side.png", dpi=150)
    plt.close(fig)

    # Report
    report_lines: List[str] = []
    report_lines.append("# Regime validation report")
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append(f"- input_csv: {INPUT_CSV}")
    report_lines.append(f"- preds_csv: {preds_path}")
    report_lines.append(f"- TP_POINTS: {TP_POINTS}")
    report_lines.append(f"- SL_POINTS: {SL_POINTS}")
    report_lines.append(f"- fee_per_trade: {FEE_PER_TRADE}")
    report_lines.append(f"- thresholds: {THRESHOLDS}")
    report_lines.append("")

    ranked = summary_df.copy()
    ranked["n_trades"] = df[df["outcome"].isin(["TP", "SL"])].groupby("regime")["pnl"].count()
    ranked["n_trades"] = ranked["n_trades"].fillna(0).astype(int)
    ranked = ranked.sort_values(
        by=["profit_factor_all", "avg_pnl_all", "n_trades"],
        ascending=[False, False, False],
    )

    report_lines.append("## Top tradable regimes (ranked)")
    report_lines.append(
        df_to_markdown(
            ranked[
                [
                    "regime",
                    "n",
                    "n_trades",
                    "profit_factor_all",
                    "avg_pnl_all",
                    "decision_tradable",
                    "decision_side",
                ]
            ].head(10),
            floatfmt=".6f",
            index=False,
        )
    )
    report_lines.append("")

    tradable = summary_df[summary_df["decision_tradable"] == True]
    not_tradable = summary_df[summary_df["decision_tradable"] == False]
    report_lines.append("## Do trade")
    report_lines.append(
        df_to_markdown(tradable[["regime", "decision_side", "profit_factor_all", "avg_pnl_all"]], index=False)
        if not tradable.empty
        else "No regimes meet tradable criteria."
    )
    report_lines.append("")

    report_lines.append("## Don't trade")
    report_lines.append(
        df_to_markdown(not_tradable[["regime", "decision_side", "profit_factor_all", "avg_pnl_all"]], index=False)
        if not not_tradable.empty
        else "All regimes meet tradable criteria."
    )
    report_lines.append("")

    best_threshold_rows = []
    for regime in range(n_regimes):
        sub_thr = threshold_df[threshold_df["regime"] == regime].copy()
        if sub_thr.empty:
            continue
        sub_thr = sub_thr.sort_values(
            by=["profit_factor", "n_trades"], ascending=[False, False]
        )
        best = sub_thr.iloc[0]
        best_threshold_rows.append(
            {
                "regime": regime,
                "best_threshold": best["threshold"],
                "profit_factor": best["profit_factor"],
                "n_trades": best["n_trades"],
            }
        )

    report_lines.append("## Best threshold per regime")
    report_lines.append(
        df_to_markdown(pd.DataFrame(best_threshold_rows), floatfmt=".6f", index=False)
        if best_threshold_rows
        else "No threshold results available."
    )
    report_lines.append("")

    warnings = summary_df[
        (summary_df["n"] < 200) | (summary_df["decision_regime_type"] == "transition_noise")
    ][["regime", "n", "decision_regime_type"]]
    report_lines.append("## Warnings")
    report_lines.append(
        df_to_markdown(warnings, index=False) if not warnings.empty else "No warnings."
    )
    report_lines.append("")

    report_path = OUT_DIR / "regime_validation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Validation complete.")
    print(f"Summary: {summary_path}")
    print(f"Threshold sweep: {threshold_path}")
    print(f"Side breakdown: {side_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
