from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def attach_gmm_columns(df: pd.DataFrame, responsibilities: np.ndarray, hard_states: np.ndarray, probmax: np.ndarray, entropy: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    k = responsibilities.shape[1]
    out["gmm_hard_state"] = hard_states.astype(int)
    for i in range(k):
        out[f"gmm_prob_state_{i}"] = responsibilities[:, i]
    out["gmm_probmax"] = probmax
    out["gmm_entropy"] = entropy
    return out


def attach_labels(
    df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame],
    timestamp_col: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    if labels_df is None or timestamp_col is None:
        out["trade_label"] = np.nan
        return out

    left = out.copy()
    if timestamp_col != "timestamp":
        left = left.rename(columns={timestamp_col: "timestamp"})
    left["timestamp"] = pd.to_datetime(left["timestamp"], errors="coerce", utc=True)

    right = labels_df.copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], errors="coerce", utc=True)

    merged = left.merge(right, on="timestamp", how="left")
    merged["trade_label"] = merged["trade_label"].astype(str).str.strip().str.lower()
    merged.loc[~merged["trade_label"].isin(["long", "short", "skip"]), "trade_label"] = "skip"

    if timestamp_col != "timestamp":
        merged = merged.rename(columns={"timestamp": timestamp_col})
    return merged


def select_output_columns(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    include_selected_features: bool,
    selected_features: Sequence[str],
) -> pd.DataFrame:
    cols: List[str] = []
    if timestamp_col and timestamp_col in df.columns:
        cols.append(timestamp_col)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            cols.append(col)
    if include_selected_features:
        for col in selected_features:
            if col in df.columns and col not in cols:
                cols.append(col)
    state_cols = ["gmm_hard_state"] + [c for c in df.columns if c.startswith("gmm_prob_state_")] + ["gmm_probmax", "gmm_entropy", "trade_label"]
    for col in state_cols:
        if col in df.columns and col not in cols:
            cols.append(col)
    return df.loc[:, cols].copy()


def compute_state_label_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    if "gmm_hard_state" not in df.columns:
        return {}

    if "trade_label" not in df.columns:
        grouped = df.groupby("gmm_hard_state").size().rename("count").reset_index()
        return {"state_counts": grouped.to_dict(orient="records")}

    work = df.copy()
    work["trade_label"] = work["trade_label"].fillna("skip").astype(str).str.lower()

    counts = (
        work.groupby(["gmm_hard_state", "trade_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["gmm_hard_state", "trade_label"], kind="mergesort")
    )
    totals = counts.groupby("gmm_hard_state")["count"].transform("sum")
    counts["pct_within_state"] = np.where(totals > 0, counts["count"] / totals, 0.0)
    return {"state_label_distribution": counts.to_dict(orient="records")}

