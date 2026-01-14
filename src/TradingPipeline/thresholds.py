from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import MIN_COVERAGE, MIN_TRADES, T_LONG_GRID, T_SHORT_GRID, T_TRADE_GRID


@dataclass(frozen=True)
class ThresholdResult:
    t_trade: float
    t_long: float
    t_short: float
    precision_long: float
    precision_short: float
    coverage_total: float
    trades: int


def apply_decision(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    risk_pass: pd.Series,
    t_trade: float,
    t_long: float,
    t_short: float,
) -> pd.Series:
    preds = pd.Series("skip", index=risk_pass.index)

    pass_mask = risk_pass.values.astype(bool)
    trade_mask = pass_mask & (p_trade >= t_trade)

    long_mask = trade_mask & (p_long >= t_long)
    short_mask = trade_mask & (p_long <= (1.0 - t_short))

    preds.loc[long_mask] = "long"
    preds.loc[short_mask] = "short"
    return preds


def _precision(y_true: pd.Series, y_pred: pd.Series, label: str) -> float:
    preds = y_pred == label
    if preds.sum() == 0:
        return 0.0
    return float((y_true[preds] == label).mean())


def sweep_thresholds(
    y_true: pd.Series,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    risk_pass: pd.Series,
) -> pd.DataFrame:
    rows = []
    for t_trade in T_TRADE_GRID:
        for t_long in T_LONG_GRID:
            for t_short in T_SHORT_GRID:
                preds = apply_decision(p_trade, p_long, risk_pass, t_trade, t_long, t_short)
                precision_long = _precision(y_true, preds, "long")
                precision_short = _precision(y_true, preds, "short")
                trades = int((preds != "skip").sum())
                coverage = trades / max(len(preds), 1)
                rows.append(
                    {
                        "t_trade": t_trade,
                        "t_long": t_long,
                        "t_short": t_short,
                        "precision_long": precision_long,
                        "precision_short": precision_short,
                        "min_precision": min(precision_long, precision_short),
                        "coverage_total": coverage,
                        "trades": trades,
                    }
                )
    return pd.DataFrame(rows)


def select_best_thresholds(sweep: pd.DataFrame) -> ThresholdResult:
    filtered = sweep[
        (sweep["coverage_total"] >= MIN_COVERAGE) & (sweep["trades"] >= MIN_TRADES)
    ].copy()
    if filtered.empty:
        filtered = sweep.copy()

    filtered = filtered.sort_values(
        ["min_precision", "coverage_total", "trades"], ascending=False
    ).reset_index(drop=True)
    best = filtered.iloc[0]
    return ThresholdResult(
        t_trade=float(best["t_trade"]),
        t_long=float(best["t_long"]),
        t_short=float(best["t_short"]),
        precision_long=float(best["precision_long"]),
        precision_short=float(best["precision_short"]),
        coverage_total=float(best["coverage_total"]),
        trades=int(best["trades"]),
    )


def save_thresholds(best: ThresholdResult, path: Path) -> None:
    payload = {
        "t_trade": best.t_trade,
        "t_long": best.t_long,
        "t_short": best.t_short,
        "precision_long": best.precision_long,
        "precision_short": best.precision_short,
        "coverage_total": best.coverage_total,
        "trades": best.trades,
    }
    path.write_text(json.dumps(payload, indent=2))
