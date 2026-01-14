from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import ALLOWED_LABELS


@dataclass(frozen=True)
class ClassificationMetrics:
    precision_long: float
    precision_short: float
    precision_trade: float
    coverage_total: float
    confusion_matrix: pd.DataFrame


def _precision(y_true: pd.Series, y_pred: pd.Series, label: str) -> float:
    preds = y_pred == label
    if preds.sum() == 0:
        return 0.0
    return float((y_true[preds] == label).mean())


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> ClassificationMetrics:
    precision_long = _precision(y_true, y_pred, "long")
    precision_short = _precision(y_true, y_pred, "short")

    trade_mask = y_pred != "skip"
    precision_trade = float((y_true[trade_mask] != "skip").mean()) if trade_mask.any() else 0.0
    coverage_total = float(trade_mask.mean())

    conf = pd.crosstab(
        y_true,
        y_pred,
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    ).reindex(index=ALLOWED_LABELS, columns=ALLOWED_LABELS, fill_value=0)

    return ClassificationMetrics(
        precision_long=precision_long,
        precision_short=precision_short,
        precision_trade=precision_trade,
        coverage_total=coverage_total,
        confusion_matrix=conf,
    )


def label_distribution(labels: pd.Series) -> Dict[str, float]:
    counts = labels.value_counts(dropna=False)
    total = len(labels)
    dist = {}
    for label in ALLOWED_LABELS:
        dist[label] = float(counts.get(label, 0) / total) if total else 0.0
    return dist


def per_month_metrics(
    timestamps: pd.Series, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": timestamps, "y_true": y_true, "y_pred": y_pred})
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        trade_mask = group["y_pred"] != "skip"
        precision_long = _precision(group["y_true"], group["y_pred"], "long")
        precision_short = _precision(group["y_true"], group["y_pred"], "short")
        coverage = float(trade_mask.mean()) if len(group) else 0.0
        rows.append(
            {
                "month": month,
                "precision_long": precision_long,
                "precision_short": precision_short,
                "coverage_total": coverage,
                "trades": int(trade_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def trade_metrics(trade_log: pd.DataFrame) -> Dict[str, float]:
    if trade_log.empty:
        return {
            "expectancy": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    returns = trade_log["net_return"].to_numpy()
    expectancy = float(np.mean(returns))
    win_rate = float(np.mean(returns > 0))

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

    equity = (1 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "expectancy": expectancy,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }
