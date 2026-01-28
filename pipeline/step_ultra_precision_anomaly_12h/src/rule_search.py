from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TP_LEVELS = [0.005, 0.0075, 0.01, 0.02]
SL_LEVELS = [0.005, 0.0075, 0.01, 0.015]
QUANTILES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
HORIZONS = [12, 24, 72]


@dataclass
class Rule:
    rule_id: str
    detector: str
    condition: str
    features: List[str]
    direction: str
    threshold: float
    horizon: int
    tp: float
    sl: float
    score_col: str | None
    cluster_id: int | None
    event_count_train: int
    trades_per_month_train: float
    precision_train: float
    avg_return_train: float
    median_mae_train: float
    meta: Dict[str, object]


def _trades_per_month(count: int, months: int) -> float:
    return float(count / months) if months else float("nan")


def _precision_long(series: pd.Series, tp: float) -> float:
    if series.empty:
        return float("nan")
    return float((series >= tp).mean())


def _precision_short(series: pd.Series, tp: float) -> float:
    if series.empty:
        return float("nan")
    return float((series <= -tp).mean())


def _avg_return(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else float("nan")


def _select_best_direction(
    df_events: pd.DataFrame,
    horizon: int,
    tp: float,
    sl_candidates: List[float],
) -> Tuple[str | None, float, float]:
    col = f"forward_return_{horizon}h"
    if col not in df_events.columns:
        return None, float("nan"), float("nan")

    long_prec = _precision_long(df_events[col], tp)
    short_prec = _precision_short(df_events[col], tp)
    direction = "long" if long_prec >= short_prec else "short"
    precision = long_prec if direction == "long" else short_prec
    mae_col = "mae_long" if direction == "long" else "mae_short"
    mae = df_events[mae_col].dropna()
    median_mae = float(mae.median()) if not mae.empty else float("nan")

    sl_selected = sl_candidates[0]
    for sl in sl_candidates:
        if np.isfinite(median_mae) and median_mae >= -sl:
            sl_selected = sl
            break
    return direction, precision, sl_selected


def _build_rule(
    detector: str,
    condition: str,
    features: List[str],
    direction: str,
    threshold: float,
    horizon: int,
    tp: float,
    sl: float,
    score_col: str | None,
    cluster_id: int | None,
    event_count: int,
    months_in_train: int,
    precision: float,
    avg_return: float,
    median_mae: float,
    meta: Dict[str, object],
) -> Rule:
    rule_id = f"{detector}|{condition}|{direction}|h{horizon}|tp{tp:.4f}"
    return Rule(
        rule_id=rule_id,
        detector=detector,
        condition=condition,
        features=features,
        direction=direction,
        threshold=threshold,
        horizon=horizon,
        tp=tp,
        sl=sl,
        score_col=score_col,
        cluster_id=cluster_id,
        event_count_train=event_count,
        trades_per_month_train=_trades_per_month(event_count, months_in_train),
        precision_train=precision,
        avg_return_train=avg_return,
        median_mae_train=median_mae,
        meta=meta,
    )


def discover_rules_from_score(
    df_train: pd.DataFrame,
    score_col: str,
    detector: str,
    feature_cols: List[str],
    months_in_train: int,
    min_events: int = 5,
    max_trades_per_month: float = 5.0,
    precision_target: float = 0.90,
) -> List[Rule]:
    rules: List[Rule] = []
    scores = df_train[score_col].dropna()
    if scores.empty:
        return rules

    for q in QUANTILES:
        threshold = float(np.quantile(scores, 1.0 - q / 100.0))
        event_mask = df_train[score_col] >= threshold
        df_events = df_train[event_mask]
        event_count = int(len(df_events))
        if event_count < min_events:
            continue

        for horizon in HORIZONS:
            for tp in TP_LEVELS:
                direction, precision, sl_selected = _select_best_direction(
                    df_events, horizon, tp, SL_LEVELS
                )
                if direction is None or not np.isfinite(precision):
                    continue
                trades_per_month = _trades_per_month(event_count, months_in_train)
                if trades_per_month > max_trades_per_month:
                    continue
                if precision < precision_target:
                    continue

                avg_return = _avg_return(df_events[f"forward_return_{horizon}h"])
                mae_col = "mae_long" if direction == "long" else "mae_short"
                median_mae = float(df_events[mae_col].median())
                condition = f"{score_col} >= {threshold:.4f} (top {q:.2f}%)"
                meta = {
                    "quantile_pct": q,
                    "event_count": event_count,
                    "precision": precision,
                    "avg_return": avg_return,
                    "fragile": event_count < 10,
                }
                rules.append(
                    _build_rule(
                        detector=detector,
                        condition=condition,
                        features=feature_cols,
                        direction=direction,
                        threshold=threshold,
                        horizon=horizon,
                        tp=tp,
                        sl=sl_selected,
                        score_col=score_col,
                        cluster_id=None,
                        event_count=event_count,
                        months_in_train=months_in_train,
                        precision=precision,
                        avg_return=avg_return,
                        median_mae=median_mae,
                        meta=meta,
                    )
                )
    return rules


def discover_rules_from_clusters(
    df_train: pd.DataFrame,
    cluster_col: str,
    freq_map: Dict[int, float],
    detector: str,
    feature_cols: List[str],
    months_in_train: int,
    min_events: int = 5,
    max_trades_per_month: float = 5.0,
    precision_target: float = 0.90,
    freq_thresholds: List[float] | None = None,
) -> List[Rule]:
    rules: List[Rule] = []
    if freq_thresholds is None:
        freq_thresholds = [0.005, 0.01, 0.02]

    for thr in freq_thresholds:
        rare_clusters = [cid for cid, freq in freq_map.items() if freq <= thr]
        for cid in rare_clusters:
            event_mask = df_train[cluster_col] == cid
            df_events = df_train[event_mask]
            event_count = int(len(df_events))
            if event_count < min_events:
                continue

            for horizon in HORIZONS:
                for tp in TP_LEVELS:
                    direction, precision, sl_selected = _select_best_direction(
                        df_events, horizon, tp, SL_LEVELS
                    )
                    if direction is None or not np.isfinite(precision):
                        continue
                    trades_per_month = _trades_per_month(event_count, months_in_train)
                    if trades_per_month > max_trades_per_month:
                        continue
                    if precision < precision_target:
                        continue

                    avg_return = _avg_return(df_events[f"forward_return_{horizon}h"])
                    mae_col = "mae_long" if direction == "long" else "mae_short"
                    median_mae = float(df_events[mae_col].median())
                    condition = f"{cluster_col} == {cid} (freq<= {thr:.3f})"
                    meta = {
                        "cluster_freq": freq_map.get(cid),
                        "event_count": event_count,
                        "precision": precision,
                        "avg_return": avg_return,
                        "fragile": event_count < 10,
                    }
                    rules.append(
                        _build_rule(
                            detector=detector,
                            condition=condition,
                            features=feature_cols,
                            direction=direction,
                            threshold=thr,
                            horizon=horizon,
                            tp=tp,
                            sl=sl_selected,
                            score_col=None,
                            cluster_id=int(cid),
                            event_count=event_count,
                            months_in_train=months_in_train,
                            precision=precision,
                            avg_return=avg_return,
                            median_mae=median_mae,
                            meta=meta,
                        )
                    )
    return rules


def apply_rule(rule: Rule, df: pd.DataFrame) -> pd.Series:
    if rule.cluster_id is not None:
        mask = df["gmm_cluster_id"] == rule.cluster_id
        return mask
    if rule.score_col is None:
        return pd.Series(False, index=df.index)
    return df[rule.score_col] >= rule.threshold


def rule_to_dict(rule: Rule) -> Dict[str, object]:
    return {
        "rule_id": rule.rule_id,
        "detector": rule.detector,
        "condition": rule.condition,
        "features": rule.features,
        "direction": rule.direction,
        "threshold": rule.threshold,
        "horizon": rule.horizon,
        "tp": rule.tp,
        "sl": rule.sl,
        "score_col": rule.score_col,
        "cluster_id": rule.cluster_id,
        "event_count_train": rule.event_count_train,
        "trades_per_month_train": rule.trades_per_month_train,
        "precision_train": rule.precision_train,
        "avg_return_train": rule.avg_return_train,
        "median_mae_train": rule.median_mae_train,
        "meta": json.dumps(rule.meta),
    }
