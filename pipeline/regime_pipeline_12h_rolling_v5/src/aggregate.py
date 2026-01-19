from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def aggregate_global_frontier(frontier: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, group in frontier.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        precision_vals = group["precision_dir"].values
        if np.isfinite(precision_vals).any():
            p10_precision = float(np.nanpercentile(precision_vals, 10))
        else:
            p10_precision = float("nan")
        metrics = {
            "mean_precision_dir": float(np.nanmean(precision_vals)),
            "median_precision_dir": float(np.nanmedian(precision_vals)),
            "p10_precision_dir": p10_precision,
            "mean_coverage": float(np.nanmean(group["coverage"])),
            "fraction_windows_trades": float((group["trade_count"] > 0).mean()),
            "mean_trade_count": float(np.nanmean(group["trade_count"])),
            "mean_precision_gate": float(np.nanmean(group["precision_gate"])),
            "window_count": int(len(group)),
        }
        row = dict(zip(group_cols, key))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_global(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    pareto = []
    for idx, row in df.iterrows():
        dominates = df[(df[x_col] >= row[x_col]) & (df[y_col] >= row[y_col])]
        if len(dominates) == 0:
            pareto.append(True)
            continue
        if len(dominates) == 1 and dominates.index[0] == idx:
            pareto.append(True)
            continue
        is_dominated = ((dominates[x_col] > row[x_col]) | (dominates[y_col] > row[y_col])).any()
        pareto.append(not is_dominated)
    return pd.Series(pareto, index=df.index)
