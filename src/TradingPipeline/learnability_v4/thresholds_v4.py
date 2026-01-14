from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config_v4 import (
    MIN_COVERAGE,
    MIN_LONG_COUNT,
    MIN_SHORT_COUNT,
    RELAX_COVERAGE_FLOOR,
    RELAX_COUNT_FLOOR,
    RELAX_PREC_FLOOR,
    TARGET_LONG_PREC,
    TARGET_SHORT_PREC,
    TARGET_TRADE_PREC,
    T_LONG_GRID,
    T_SHORT_GRID,
    T_TRADE_GRID,
)


@dataclass
class ThresholdSelection:
    t_trade: float
    t_long: float
    t_short: float
    relaxation_log: List[str]
    feasible: bool


def _trade_precision(y_true: pd.Series, trade_mask: pd.Series) -> float:
    if trade_mask.sum() == 0:
        return 0.0
    return float((y_true[trade_mask] != "skip").mean())


def _precision(y_true: pd.Series, y_pred: pd.Series, label: str) -> float:
    pred_mask = y_pred == label
    if pred_mask.sum() == 0:
        return 0.0
    return float((y_true[pred_mask] == label).mean())


def select_trade_threshold(
    y_true: pd.Series,
    p_trade: pd.Series,
    risk_pass: pd.Series,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    best = None
    for t in T_TRADE_GRID:
        trade_mask = risk_pass & (p_trade >= t)
        precision = _trade_precision(y_true, trade_mask)
        trades = int(trade_mask.sum())
        rows.append(
            {
                "t_trade": t,
                "trade_precision": precision,
                "trade_count": trades,
                "coverage": trades / max(len(y_true), 1),
            }
        )
        if precision >= TARGET_TRADE_PREC:
            if best is None or (precision > best["trade_precision"]) or (
                precision == best["trade_precision"] and trades > best["trade_count"]
            ):
                best = rows[-1]

    sweep = pd.DataFrame(rows)
    if best is None:
        best = sweep.sort_values(["trade_precision", "trade_count"], ascending=False).iloc[0].to_dict()

    return float(best["t_trade"]), sweep


def _evaluate_thresholds(
    y_true: pd.Series,
    p_trade: pd.Series,
    p_long: pd.Series,
    risk_pass: pd.Series,
    direction_quality: pd.Series,
    t_trade: float,
    t_long: float,
    t_short: float,
) -> Dict[str, float]:
    trade_mask = risk_pass & (p_trade >= t_trade)
    dir_mask = trade_mask & direction_quality

    preds = pd.Series("skip", index=y_true.index)
    long_mask = dir_mask & (p_long >= t_long)
    short_mask = dir_mask & ((1.0 - p_long) >= t_short)
    preds.loc[long_mask] = "long"
    preds.loc[short_mask] = "short"

    precision_long = _precision(y_true, preds, "long")
    precision_short = _precision(y_true, preds, "short")
    long_count = int((preds == "long").sum())
    short_count = int((preds == "short").sum())
    coverage = float((preds != "skip").mean())
    trade_precision = _trade_precision(y_true, preds != "skip")

    return {
        "precision_long": precision_long,
        "precision_short": precision_short,
        "trade_precision": trade_precision,
        "long_count": long_count,
        "short_count": short_count,
        "coverage": coverage,
    }


def select_direction_thresholds(
    y_true: pd.Series,
    p_trade: pd.Series,
    p_long: pd.Series,
    risk_pass: pd.Series,
    direction_quality: pd.Series,
    t_trade: float,
) -> Tuple[ThresholdSelection, pd.DataFrame]:
    sweep_rows = []
    for t_long in T_LONG_GRID:
        for t_short in T_SHORT_GRID:
            metrics = _evaluate_thresholds(
                y_true,
                p_trade,
                p_long,
                risk_pass,
                direction_quality,
                t_trade,
                t_long,
                t_short,
            )
            sweep_rows.append(
                {
                    "t_trade": t_trade,
                    "t_long": t_long,
                    "t_short": t_short,
                    **metrics,
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)
    relaxation_log: List[str] = []

    precision_targets = []
    step = 0.0
    while True:
        long_target = round(max(TARGET_LONG_PREC - step, RELAX_PREC_FLOOR), 2)
        short_target = round(max(TARGET_SHORT_PREC - step, RELAX_PREC_FLOOR), 2)
        precision_targets.append((long_target, short_target))
        if long_target <= RELAX_PREC_FLOOR and short_target <= RELAX_PREC_FLOOR:
            break
        step += 0.05

    count_targets = list(range(MIN_LONG_COUNT, RELAX_COUNT_FLOOR - 1, -5))
    if RELAX_COUNT_FLOOR not in count_targets:
        count_targets.append(RELAX_COUNT_FLOOR)

    coverage_targets = [MIN_COVERAGE, RELAX_COVERAGE_FLOOR]

    chosen = None

    precision_relaxed_logged = False
    count_relaxed_logged = False
    coverage_relaxed_logged = False

    for prec_target_long, prec_target_short in precision_targets:
        if (
            not precision_relaxed_logged
            and (prec_target_long < TARGET_LONG_PREC or prec_target_short < TARGET_SHORT_PREC)
        ):
            relaxation_log.append(
                f"precision_target_relaxed_to=long:{prec_target_long:.2f},short:{prec_target_short:.2f}"
            )
            precision_relaxed_logged = True
        for count_target in count_targets:
            if not count_relaxed_logged and count_target < MIN_LONG_COUNT:
                relaxation_log.append(f"min_count_relaxed_to={count_target}")
                count_relaxed_logged = True
            for coverage_target in coverage_targets:
                if not coverage_relaxed_logged and coverage_target < MIN_COVERAGE:
                    relaxation_log.append(f"coverage_relaxed_to={coverage_target:.4f}")
                    coverage_relaxed_logged = True

                filtered = sweep_df[
                    (sweep_df["precision_long"] >= prec_target_long)
                    & (sweep_df["precision_short"] >= prec_target_short)
                    & (sweep_df["long_count"] >= count_target)
                    & (sweep_df["short_count"] >= count_target)
                    & (sweep_df["coverage"] >= coverage_target)
                ]
                if not filtered.empty:
                    filtered = filtered.sort_values(
                        ["precision_long", "precision_short", "coverage", "long_count"],
                        ascending=False,
                    )
                    chosen_row = filtered.iloc[0]
                    chosen = (float(chosen_row["t_long"]), float(chosen_row["t_short"]))
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break

    if chosen is None:
        best = sweep_df.sort_values(
            ["precision_long", "precision_short", "coverage"], ascending=False
        ).iloc[0]
        chosen = (float(best["t_long"]), float(best["t_short"]))
        relaxation_log.append("no_feasible_thresholds_used_best_precision")

    selection = ThresholdSelection(
        t_trade=float(t_trade),
        t_long=float(chosen[0]),
        t_short=float(chosen[1]),
        relaxation_log=relaxation_log,
        feasible=chosen is not None and "no_feasible_thresholds_used_best_precision" not in relaxation_log,
    )

    return selection, sweep_df
