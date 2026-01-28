from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


FEE_PER_TRADE = 0.0005

POLICY_COLUMNS = {
    "baseline_experts": "y_pred",
    "gate_only": "signal_gate_only",
    "agreement": "signal_agreement",
    "low_regime_whitelist_clusters": "signal_low_regime_whitelist_clusters",
    "weighted_vote": "signal_weighted_vote",
}

REQUIRED_TRADE_COLS = {"timestamp", "fold_id", "forward_return_1"}

OUTPUT_DIR = Path(__file__).resolve().parent
STEP_DIR = OUTPUT_DIR.parents[1]


def _read_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def find_trade_rows_file() -> Path:
    candidate = OUTPUT_DIR / "stacking_policy_rows.csv"
    if candidate.exists():
        return candidate

    candidates = list(STEP_DIR.rglob("*.csv"))
    best_match = None
    best_score = -1
    for path in candidates:
        try:
            cols = set(_read_header(path))
        except Exception:
            continue
        if not REQUIRED_TRADE_COLS.issubset(cols):
            continue
        score = len(cols.intersection(REQUIRED_TRADE_COLS))
        if "y_pred" in cols or any(col.startswith("signal_") for col in cols):
            score += 1
        if score > best_score:
            best_score = score
            best_match = path

    if best_match is None:
        raise FileNotFoundError(
            "Could not locate a per-trade rows file with timestamp/fold_id/forward_return_1."
        )
    return best_match


def normalize_label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return "long"
        if value == -1:
            return "short"
        if value == 0:
            return "skip"
    value_str = str(value).strip().lower()
    if "long" in value_str:
        return "long"
    if "short" in value_str:
        return "short"
    if value_str in {"skip", "flat", "none", "neutral"}:
        return "skip"
    return value_str


def compute_profit_factor(trade_returns: np.ndarray) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("nan")
    return float(gains / abs(losses))


def compute_max_drawdown(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + trade_returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min())


def compute_cvar95(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return float("nan")
    threshold = np.quantile(trade_returns, 0.05)
    tail = trade_returns[trade_returns <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(np.mean(tail))


def build_policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if "split_type" in df.columns:
        mode_col = "split_type"
    elif "mode" in df.columns:
        mode_col = "mode"
    else:
        mode_col = "split_type"
        df[mode_col] = "unknown"

    timestamp_col = "timestamp"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

    for policy, col in POLICY_COLUMNS.items():
        if col not in df.columns:
            continue
        for (mode, fold_id, approach), group in df.groupby([mode_col, "fold_id", "approach"]):
            group = group.sort_values(timestamp_col)
            support = len(group)
            signal_raw = group[col].apply(normalize_label)
            y_true = group["y_true"].apply(normalize_label) if "y_true" in group.columns else None

            trade_mask = signal_raw.isin({"long", "short"})
            trade_count = int(trade_mask.sum())
            coverage = float(trade_count / support) if support else float("nan")

            returns = group.loc[trade_mask, "forward_return_1"].astype(float).to_numpy()
            directions = signal_raw[trade_mask].to_numpy()
            signed = np.where(directions == "short", -returns, returns)
            trade_returns = signed - FEE_PER_TRADE

            win_rate = float((trade_returns > 0).mean()) if trade_count else float("nan")
            avg_pnl = float(np.mean(trade_returns)) if trade_count else float("nan")
            profit_factor = compute_profit_factor(trade_returns) if trade_count else float("nan")
            max_drawdown = compute_max_drawdown(trade_returns) if trade_count else float("nan")
            cvar95 = compute_cvar95(trade_returns) if trade_count else float("nan")

            precision_long = float("nan")
            precision_short = float("nan")
            if y_true is not None:
                long_mask = signal_raw == "long"
                short_mask = signal_raw == "short"
                if long_mask.any():
                    precision_long = float((y_true[long_mask] == "long").mean())
                if short_mask.any():
                    precision_short = float((y_true[short_mask] == "short").mean())

            rows.append(
                {
                    "mode": mode,
                    "fold_id": fold_id,
                    "approach": approach,
                    "policy": policy,
                    "support": support,
                    "n_trades": trade_count,
                    "win_rate": win_rate,
                    "avg_pnl_per_trade": avg_pnl,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "cvar95": cvar95,
                    "coverage": coverage,
                    "precision_long": precision_long,
                    "precision_short": precision_short,
                }
            )

    return pd.DataFrame(rows)


def summarize_by_filter(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_trades",
        "win_rate",
        "avg_pnl_per_trade",
        "profit_factor",
        "max_drawdown",
        "cvar95",
        "coverage",
        "precision_long",
        "precision_short",
    ]
    rows = []
    for min_trades in [0, 5, 10]:
        filtered = df[df["n_trades"] >= min_trades]
        for keys, group in filtered.groupby(["mode", "approach", "policy"]):
            mode, approach, policy = keys
            row_mean = {
                "mode": mode,
                "approach": approach,
                "policy": policy,
                "min_trades": min_trades,
                "stat": "mean",
                "n_folds": int(group["fold_id"].nunique()),
            }
            row_std = {
                "mode": mode,
                "approach": approach,
                "policy": policy,
                "min_trades": min_trades,
                "stat": "std",
                "n_folds": int(group["fold_id"].nunique()),
            }
            for metric in metrics:
                row_mean[metric] = float(group[metric].mean())
                row_std[metric] = float(group[metric].std())
            rows.extend([row_mean, row_std])
    return pd.DataFrame(rows)


def print_warnings(df: pd.DataFrame) -> None:
    dd_bad = df[df["max_drawdown"] < -1.0]
    if not dd_bad.empty:
        samples = dd_bad[["mode", "fold_id", "approach", "policy", "max_drawdown"]].head(5)
        print(f"WARNING: max_drawdown < -1.0 in {len(dd_bad)} folds. Examples:")
        print(samples.to_string(index=False))

    pf_bad = df[~np.isfinite(df["profit_factor"])]
    if not pf_bad.empty:
        samples = pf_bad[["mode", "fold_id", "approach", "policy", "profit_factor"]].head(5)
        print(f"WARNING: profit_factor NaN/inf in {len(pf_bad)} folds. Examples:")
        print(samples.to_string(index=False))

    zero_trades = df["n_trades"] == 0
    if zero_trades.any():
        ratio = float(zero_trades.mean())
        if ratio >= 0.5:
            print(f"WARNING: {ratio:.0%} of folds have n_trades == 0.")


def main() -> None:
    trade_rows_path = find_trade_rows_file()
    df = pd.read_csv(trade_rows_path)

    diagnostics = build_policy_metrics(df)
    if diagnostics.empty:
        raise RuntimeError("No diagnostics generated; check policy columns in trade rows.")

    print_warnings(diagnostics)

    cleaned_path = OUTPUT_DIR / "diagnostics_cleaned.csv"
    diagnostics.to_csv(cleaned_path, index=False)

    summary = summarize_by_filter(diagnostics)
    summary_path = OUTPUT_DIR / "diagnostics_summary_filtered.csv"
    summary.to_csv(summary_path, index=False)

    print("Diagnostics written:")
    print(cleaned_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
