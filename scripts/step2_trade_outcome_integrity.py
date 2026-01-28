from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# === CONFIGURATION ===
INPUT_CSV = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv"
)
RESULTS_DIR = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\validation"
)
GMM_OUTPUT_CANDIDATES = [
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_regimes_per_row.csv"
    ),
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\gmm_per_row.csv"
    ),
    Path(
        r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results\preds_per_row.csv"
    ),
]

TP_PCT = 0.02
SL_PCT = 0.01
FEE_PER_TRADE = 0.0005
HORIZON_N_CANDLES = 1
USE_LOWER_TF = False
LOWER_TF_CSV: Optional[Path] = None
RANDOM_SEED = 42


META_COLS = ["timestamp", "open", "high", "low", "close"]


def resolve_input_path(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


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


def compute_horizon_extremes(
    high: np.ndarray, low: np.ndarray, horizon_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    horizon_high = np.full(len(high), np.nan, dtype=float)
    horizon_low = np.full(len(low), np.nan, dtype=float)
    for i in range(len(high)):
        end = i + horizon_n
        if end >= len(high):
            break
        window_high = high[i + 1 : end + 1]
        window_low = low[i + 1 : end + 1]
        if np.all(np.isnan(window_high)) or np.all(np.isnan(window_low)):
            continue
        horizon_high[i] = np.nanmax(window_high)
        horizon_low[i] = np.nanmin(window_low)
    return horizon_high, horizon_low


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


class LowerTFIndex:
    def __init__(self, df: pd.DataFrame, ts_col: str, high_col: str, low_col: str) -> None:
        self.index = pd.DatetimeIndex(df[ts_col])
        self.high = df[high_col].to_numpy()
        self.low = df[low_col].to_numpy()

    def iter_slice(self, start: pd.Timestamp, end: pd.Timestamp):
        left = self.index.searchsorted(start, side="left")
        right = self.index.searchsorted(end, side="left")
        for i in range(left, right):
            yield self.high[i], self.low[i]


def load_lower_tf(path: Optional[Path]) -> Optional[LowerTFIndex]:
    if path is None:
        return None
    if not path.exists():
        print(f"Lower TF CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ts_col = None
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        for candidate in ("ts_utc", "time", "open_time", "datetime"):
            if candidate in df.columns:
                ts_col = candidate
                break
    if ts_col is None:
        print("Lower TF timestamp column not found.")
        return None
    if "high" not in df.columns or "low" not in df.columns:
        print("Lower TF missing high/low columns.")
        return None
    df["timestamp"] = parse_timestamp(df[ts_col])
    df = df[df["timestamp"].notna()].sort_values("timestamp")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    return LowerTFIndex(df, "timestamp", "high", "low")


def run_gmm_if_needed() -> Optional[Path]:
    existing = resolve_input_path(GMM_OUTPUT_CANDIDATES)
    if existing is not None:
        return existing
    script_path = Path("scripts/step1_gmm_regimes_open.py")
    if not script_path.exists():
        return None
    print("GMM outputs missing; re-running GMM step.")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        return None
    return resolve_input_path(GMM_OUTPUT_CANDIDATES)


def compute_group_summary(trades: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    n_trades = len(trades)
    n_wins = int((trades["outcome"] == "win").sum())
    n_losses = int((trades["outcome"] == "loss").sum())
    n_both = int((trades["outcome"] == "both_hit_unresolved").sum())
    n_skips = int((trades["outcome"] == "skip").sum())

    denom = n_wins + n_losses
    win_rate = n_wins / denom if denom > 0 else np.nan

    valid_pnl = trades["pnl_after_fee"].dropna()
    avg_pnl = valid_pnl.mean() if not valid_pnl.empty else np.nan
    median_pnl = valid_pnl.median() if not valid_pnl.empty else np.nan
    pnl_std = valid_pnl.std(ddof=0) if not valid_pnl.empty else np.nan
    min_pnl = valid_pnl.min() if not valid_pnl.empty else np.nan
    max_pnl = valid_pnl.max() if not valid_pnl.empty else np.nan
    pf = profit_factor(valid_pnl)

    trades_sorted = trades.sort_values("timestamp")
    max_dd = max_drawdown(trades_sorted["pnl_after_fee"].fillna(0.0))
    cvar = cvar_95(valid_pnl)

    if n_trades > 1:
        ts_span = (trades["timestamp"].max() - trades["timestamp"].min()).total_seconds()
        months = max(ts_span / 86400.0 / 30.44, 1.0)
    else:
        months = 1.0
    trades_per_month = n_trades / months if months > 0 else np.nan

    unique_pnl_count = int(valid_pnl.nunique())
    pct_negative_pnl = float((valid_pnl < 0).mean()) if not valid_pnl.empty else np.nan

    long_mask = trades["side"] == "long"
    short_mask = trades["side"] == "short"
    tp_consistency = np.where(
        long_mask,
        (trades["tp_price"] > trades["entry_open"]) & (trades["sl_price"] < trades["entry_open"]),
        np.where(
            short_mask,
            (trades["tp_price"] < trades["entry_open"]) & (trades["sl_price"] > trades["entry_open"]),
            False,
        ),
    )
    check_tp_sl_consistency = float(np.mean(tp_consistency)) if n_trades > 0 else np.nan

    hits_consistency = (trades["tp_hit"] + trades["sl_hit"]).isin([0, 1, 2])
    check_hits_consistency = float(np.mean(hits_consistency)) if n_trades > 0 else np.nan

    summary = {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_both_hit": n_both,
        "n_skips": n_skips,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "pnl_std": pnl_std,
        "min_pnl": min_pnl,
        "max_pnl": max_pnl,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "cvar95": cvar,
        "trades_per_month": trades_per_month,
        "unique_pnl_count": unique_pnl_count,
        "pct_negative_pnl": pct_negative_pnl,
        "check_tp_sl_consistency": check_tp_sl_consistency,
        "check_hits_consistency": check_hits_consistency,
    }

    aux = {
        "gross_profit": float(valid_pnl[valid_pnl > 0].sum()) if not valid_pnl.empty else np.nan,
        "gross_loss": float(valid_pnl[valid_pnl < 0].sum()) if not valid_pnl.empty else np.nan,
    }
    return summary, aux


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_main = pd.read_csv(INPUT_CSV)
    df_main.columns = df_main.columns.str.strip()
    if "timestamp" not in df_main.columns:
        raise ValueError("timestamp column missing from INPUT_CSV.")
    df_main["timestamp"] = parse_timestamp(df_main["timestamp"])
    df_main = df_main[df_main["timestamp"].notna()].copy()

    for col in META_COLS[1:]:
        if col not in df_main.columns:
            raise ValueError(f"Missing required column in INPUT_CSV: {col}")
        df_main[col] = pd.to_numeric(df_main[col], errors="coerce")

    gmm_path = None
    if "regime_id" not in df_main.columns or "regime_prob_max" not in df_main.columns:
        gmm_path = run_gmm_if_needed()
        if gmm_path is None:
            raise FileNotFoundError("GMM outputs not found and could not re-run GMM step.")

        df_gmm = pd.read_csv(gmm_path)
        df_gmm.columns = df_gmm.columns.str.strip()
        if "timestamp" not in df_gmm.columns:
            raise ValueError("timestamp column missing from GMM outputs.")
        df_gmm["timestamp"] = parse_timestamp(df_gmm["timestamp"])
        df_gmm = df_gmm[df_gmm["timestamp"].notna()].copy()

        if "regime_id" not in df_gmm.columns and "regime" in df_gmm.columns:
            df_gmm = df_gmm.rename(columns={"regime": "regime_id"})
        if "regime_prob_max" not in df_gmm.columns and "prob_max" in df_gmm.columns:
            df_gmm = df_gmm.rename(columns={"prob_max": "regime_prob_max"})

        missing = [c for c in ["regime_id", "regime_prob_max"] if c not in df_gmm.columns]
        if missing:
            raise ValueError(f"Missing required columns in GMM outputs: {missing}")

        merge_cols = ["timestamp", "regime_id", "regime_prob_max"]
        if "entropy" in df_gmm.columns:
            merge_cols.append("entropy")

        df = pd.merge(df_main, df_gmm[merge_cols], on="timestamp", how="inner")
    else:
        df = df_main.copy()

    if df.empty:
        raise ValueError("Merged dataset is empty. Check timestamps and GMM outputs.")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df["regime_id"] = pd.to_numeric(df["regime_id"], errors="coerce")
    df = df[df["regime_id"].notna()].copy()
    df["regime_id"] = df["regime_id"].astype(int)
    df["regime_prob_max"] = pd.to_numeric(df["regime_prob_max"], errors="coerce")
    if "entropy" in df.columns:
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    candle_delta = df["timestamp"].diff().median()
    if pd.isna(candle_delta):
        candle_delta = pd.Timedelta(hours=12)

    horizon_high, horizon_low = compute_horizon_extremes(
        df["high"].to_numpy(), df["low"].to_numpy(), HORIZON_N_CANDLES
    )

    horizon_end_ts = df["timestamp"] + candle_delta * HORIZON_N_CANDLES

    lower_tf_index = load_lower_tf(LOWER_TF_CSV) if USE_LOWER_TF else None

    trades_base = df.copy()
    if "candle_type" in trades_base.columns:
        trades_base["side"] = trades_base["candle_type"].astype(str).str.lower().str.strip()
        trades_base = trades_base[trades_base["side"].isin(["long", "short"])].copy()
        trades_base["side_source"] = "candle_type"
    else:
        long_side = trades_base.copy()
        long_side["side"] = "long"
        long_side["side_source"] = "simulated"
        short_side = trades_base.copy()
        short_side["side"] = "short"
        short_side["side_source"] = "simulated"
        trades_base = pd.concat([long_side, short_side], ignore_index=True)

    if trades_base.empty:
        raise ValueError("No trade rows available after applying side selection.")

    trades_base["entry_open"] = trades_base["open"]
    trades_base["tp_price"] = np.where(
        trades_base["side"] == "long",
        trades_base["entry_open"] * (1.0 + TP_PCT),
        trades_base["entry_open"] * (1.0 - TP_PCT),
    )
    trades_base["sl_price"] = np.where(
        trades_base["side"] == "long",
        trades_base["entry_open"] * (1.0 - SL_PCT),
        trades_base["entry_open"] * (1.0 + SL_PCT),
    )

    trades_base["horizon_high"] = horizon_high[trades_base.index]
    trades_base["horizon_low"] = horizon_low[trades_base.index]
    trades_base["horizon_end_ts"] = horizon_end_ts[trades_base.index]

    missing_data = trades_base["horizon_high"].isna() | trades_base["horizon_low"].isna()

    tp_hit = np.where(
        trades_base["side"] == "long",
        trades_base["horizon_high"] >= trades_base["tp_price"],
        trades_base["horizon_low"] <= trades_base["tp_price"],
    )
    sl_hit = np.where(
        trades_base["side"] == "long",
        trades_base["horizon_low"] <= trades_base["sl_price"],
        trades_base["horizon_high"] >= trades_base["sl_price"],
    )
    tp_hit = np.where(missing_data, False, tp_hit)
    sl_hit = np.where(missing_data, False, sl_hit)

    trades_base["tp_hit"] = tp_hit.astype(int)
    trades_base["sl_hit"] = sl_hit.astype(int)
    trades_base["ambiguous"] = (trades_base["tp_hit"] == 1) & (trades_base["sl_hit"] == 1)
    trades_base["missing_data"] = missing_data
    trades_base["resolved_by_lower_tf"] = False

    outcome = np.full(len(trades_base), "skip", dtype=object)
    win_mask = (trades_base["tp_hit"] == 1) & (trades_base["sl_hit"] == 0)
    loss_mask = (trades_base["tp_hit"] == 0) & (trades_base["sl_hit"] == 1)
    both_mask = (trades_base["tp_hit"] == 1) & (trades_base["sl_hit"] == 1)

    outcome[win_mask] = "win"
    outcome[loss_mask] = "loss"
    outcome[both_mask] = "both_hit_unresolved"

    if lower_tf_index is not None and both_mask.any():
        for idx in trades_base.index[both_mask]:
            row = trades_base.loc[idx]
            start_ts = row["timestamp"]
            end_ts = row["horizon_end_ts"]
            side = row["side"]
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            resolved = None
            for hi, lo in lower_tf_index.iter_slice(start_ts, end_ts):
                if side == "long":
                    hit_tp = hi >= row["tp_price"]
                    hit_sl = lo <= row["sl_price"]
                else:
                    hit_tp = lo <= row["tp_price"]
                    hit_sl = hi >= row["sl_price"]
                if hit_tp and not hit_sl:
                    resolved = "win"
                    break
                if hit_sl and not hit_tp:
                    resolved = "loss"
                    break
                if hit_tp and hit_sl:
                    resolved = "both_hit_unresolved"
                    break
            if resolved in {"win", "loss"}:
                outcome[idx] = resolved
                trades_base.loc[idx, "resolved_by_lower_tf"] = True
            else:
                outcome[idx] = "both_hit_unresolved"

    trades_base["outcome"] = outcome

    pnl_raw = np.full(len(trades_base), np.nan, dtype=float)
    long_win = (trades_base["side"] == "long") & (trades_base["outcome"] == "win")
    long_loss = (trades_base["side"] == "long") & (trades_base["outcome"] == "loss")
    short_win = (trades_base["side"] == "short") & (trades_base["outcome"] == "win")
    short_loss = (trades_base["side"] == "short") & (trades_base["outcome"] == "loss")

    pnl_raw[long_win] = (trades_base.loc[long_win, "tp_price"] / trades_base.loc[long_win, "entry_open"]) - 1.0
    pnl_raw[long_loss] = (trades_base.loc[long_loss, "sl_price"] / trades_base.loc[long_loss, "entry_open"]) - 1.0
    pnl_raw[short_win] = (trades_base.loc[short_win, "entry_open"] / trades_base.loc[short_win, "tp_price"]) - 1.0
    pnl_raw[short_loss] = (trades_base.loc[short_loss, "entry_open"] / trades_base.loc[short_loss, "sl_price"]) - 1.0

    pnl_after_fee = np.where(np.isfinite(pnl_raw), pnl_raw - FEE_PER_TRADE, np.nan)
    trades_base["pnl_raw"] = pnl_raw
    trades_base["pnl_after_fee"] = pnl_after_fee

    trades_out_cols = [
        "timestamp",
        "entry_open",
        "side",
        "tp_price",
        "sl_price",
        "horizon_high",
        "horizon_low",
        "tp_hit",
        "sl_hit",
        "outcome",
        "pnl_raw",
        "pnl_after_fee",
        "regime_id",
        "regime_prob_max",
        "entropy",
        "ambiguous",
        "missing_data",
        "resolved_by_lower_tf",
        "side_source",
    ]
    trades_out_cols = [c for c in trades_out_cols if c in trades_base.columns]
    trades_output_path = RESULTS_DIR / "validation_trades_per_row.csv"
    trades_base[trades_out_cols].to_csv(trades_output_path, index=False)

    summary_rows = []
    summary_aux = {}

    global_summary, global_aux = compute_group_summary(trades_base)
    summary_aux["global"] = global_aux
    summary_rows.append({"regime_id": "ALL", "side": "ALL", **global_summary})

    for regime_id, group in trades_base.groupby("regime_id"):
        summary, aux = compute_group_summary(group)
        summary_aux[f"regime_{regime_id}"] = aux
        summary_rows.append({"regime_id": int(regime_id), "side": "ALL", **summary})

    for side, group in trades_base.groupby("side"):
        summary, aux = compute_group_summary(group)
        summary_aux[f"side_{side}"] = aux
        summary_rows.append({"regime_id": "ALL", "side": side, **summary})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "validation_integrity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Sanity checks
    sanity_lines: List[str] = []
    sanity_lines.append("VALIDATION SANITY CHECKS")
    sanity_lines.append("")

    regime_summaries = summary_df[summary_df["side"] == "ALL"]
    regimes_large = regime_summaries[regime_summaries["n_trades"] >= 5]
    if not regimes_large.empty and (regimes_large["pct_negative_pnl"] == 0).all():
        sanity_lines.append("FAIL: pct_negative_pnl == 0 across all regimes with n_trades >= 5.")
    else:
        sanity_lines.append("PASS: pct_negative_pnl shows negatives in at least one regime.")

    for _, row in regime_summaries.iterrows():
        label = f"regime={row['regime_id']}"
        n_trades = int(row["n_trades"])
        unique_pnl_count = int(row["unique_pnl_count"]) if not pd.isna(row["unique_pnl_count"]) else 0
        pnl_std = row["pnl_std"]
        max_dd = row["max_drawdown"]
        gross_loss = summary_aux.get(f"regime_{row['regime_id']}", {}).get("gross_loss", np.nan)

        if n_trades >= 50 and unique_pnl_count <= 2:
            sanity_lines.append(f"FAIL: {label} unique_pnl_count <= 2 with n_trades >= 50.")
        if n_trades >= 20 and (gross_loss == 0):
            sanity_lines.append(f"FAIL: {label} gross_loss == 0 with n_trades >= 20.")
        if n_trades >= 20 and (max_dd == 0):
            sanity_lines.append(f"FAIL: {label} max_drawdown == 0 with n_trades >= 20.")
        if n_trades >= 20 and (pnl_std == 0):
            sanity_lines.append(f"FAIL: {label} pnl_std == 0 with n_trades >= 20.")

    if len(sanity_lines) == 2:
        sanity_lines.append("PASS: No additional failures detected.")

    # Suspicious rows
    sanity_lines.append("")
    sanity_lines.append("TOP 10 SUSPICIOUS ROWS")
    suspicious = trades_base.copy()
    suspicious_score = np.zeros(len(suspicious), dtype=int)
    suspicious_score += (suspicious["outcome"] == "both_hit_unresolved").astype(int) * 3
    suspicious_score += suspicious["missing_data"].astype(int) * 3
    suspicious_score += (suspicious["outcome"] == "skip").astype(int)
    suspicious_score += (suspicious["tp_hit"] + suspicious["sl_hit"] == 2).astype(int)

    mode_pnl = suspicious["pnl_after_fee"].mode()
    if not mode_pnl.empty:
        mode_value = mode_pnl.iloc[0]
        if suspicious["pnl_after_fee"].value_counts().iloc[0] / max(len(suspicious), 1) > 0.3:
            suspicious_score += (suspicious["pnl_after_fee"] == mode_value).astype(int) * 2

    suspicious["suspicious_score"] = suspicious_score
    suspicious = suspicious.sort_values(["suspicious_score", "timestamp"], ascending=[False, True])
    top_cols = [
        "timestamp",
        "side",
        "entry_open",
        "tp_price",
        "sl_price",
        "horizon_high",
        "horizon_low",
        "tp_hit",
        "sl_hit",
        "outcome",
        "pnl_after_fee",
        "regime_id",
        "suspicious_score",
    ]
    top_cols = [c for c in top_cols if c in suspicious.columns]
    top_rows = suspicious.head(10)[top_cols]
    if top_rows.empty:
        sanity_lines.append("No suspicious rows.")
    else:
        sanity_lines.append(top_rows.to_csv(index=False))

    sanity_path = RESULTS_DIR / "validation_sanity_checks.txt"
    sanity_path.write_text("\n".join(sanity_lines), encoding="utf-8")

    # Spotchecks
    rng = np.random.RandomState(RANDOM_SEED)
    spot_rows = []
    for regime_id, group in trades_base.groupby("regime_id"):
        if group.empty:
            continue
        sample_n = min(10, len(group))
        sample = group.sample(n=sample_n, random_state=rng.randint(0, 2**31 - 1))
        spot_rows.append(sample)
    if spot_rows:
        spot_df = pd.concat(spot_rows, ignore_index=True)
    else:
        spot_df = trades_base.head(0).copy()

    spot_cols = [
        "timestamp",
        "entry_open",
        "side",
        "tp_price",
        "sl_price",
        "horizon_high",
        "horizon_low",
        "tp_hit",
        "sl_hit",
        "pnl_after_fee",
        "outcome",
        "regime_id",
        "regime_prob_max",
    ]
    spot_cols = [c for c in spot_cols if c in spot_df.columns]
    spot_path = RESULTS_DIR / "spotcheck_examples.csv"
    spot_df[spot_cols].to_csv(spot_path, index=False)

    # Figures
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error

        transition = pd.crosstab(
            trades_base["regime_id"].iloc[:-1],
            trades_base["regime_id"].iloc[1:],
            rownames=["from"],
            colnames=["to"],
            dropna=False,
        )
        n_regimes = transition.shape[0]
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

        run_lengths = []
        current = None
        length = 0
        for r in trades_base["regime_id"].to_numpy():
            if current is None:
                current = r
                length = 1
            elif r == current:
                length += 1
            else:
                run_lengths.append(length)
                current = r
                length = 1
        if length:
            run_lengths.append(length)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(run_lengths, bins=30, color="#feb24c", alpha=0.8)
        ax.set_title("Run length distribution")
        ax.set_xlabel("run_length")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "run_length_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(trades_base["regime_prob_max"].dropna(), bins=30, color="#2c7fb8", alpha=0.8)
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
        for regime_id, group in trades_base.groupby("regime_id"):
            for side in ["long", "short"]:
                subset = group[
                    (group["side"] == side) & (group["outcome"].isin(["win", "loss"]))
                ]["pnl_after_fee"].dropna()
                if subset.empty:
                    continue
                box_data.append(subset.values)
                labels.append(f"{regime_id}-{side[0]}")
        if box_data:
            ax.boxplot(box_data, labels=labels, showfliers=False)
            ax.set_title("PNL by regime and side")
            ax.set_xlabel("regime-side")
            ax.set_ylabel("pnl_after_fee")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / "pnl_boxplot_by_regime_and_side.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Figure generation skipped: {exc}")

    total_trades = len(trades_base)
    total_wins = int((trades_base["outcome"] == "win").sum())
    total_losses = int((trades_base["outcome"] == "loss").sum())
    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else np.nan
    avg_pnl = trades_base["pnl_after_fee"].dropna().mean()
    max_dd = max_drawdown(trades_base.sort_values("timestamp")["pnl_after_fee"].fillna(0.0))

    sanity_pass = all("FAIL" not in line for line in sanity_lines)
    print("Validation complete.")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.4f}")
    print(f"Avg pnl: {avg_pnl:.6f}")
    print(f"Max DD: {max_dd:.6f}")
    print(f"Sanity checks: {'PASS' if sanity_pass else 'FAIL'}")
    print(f"Trades per row: {trades_output_path}")
    print(f"Summary: {summary_path}")
    print(f"Sanity checks: {sanity_path}")
    print(f"Spotchecks: {spot_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
