from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def detect_timestamp_column(columns: Iterable[str], candidates: List[str]) -> str:
    columns_list = list(columns)
    columns_lower = {col.lower(): col for col in columns_list}
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    for col in columns_list:
        if "timestamp" in col.lower():
            return col
    raise ValueError("No timestamp column found in labeled CSV.")


def load_labeled_csv(path: Path, time_candidates: List[str]) -> Tuple[pd.DataFrame, str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    time_col = detect_timestamp_column(df.columns, time_candidates)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=[time_col]).reset_index(drop=True)
    dropped = before - len(df)

    summary = {
        "rows": len(df),
        "time_col": time_col,
        "dropped_duplicates": dropped,
        "start": df[time_col].min(),
        "end": df[time_col].max(),
    }
    return df, time_col, summary


def missingness_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    missing_pct = df.isna().mean().sort_values(ascending=False)
    top = missing_pct.head(top_n)
    return top.reset_index().rename(columns={"index": "column", 0: "missing_pct"})


def select_feature_columns(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    ambig_col: str,
    drop_missing_above: float,
) -> Tuple[List[str], pd.DataFrame]:
    meta_cols = {
        time_col,
        "timestamp",
        "ts_utc",
        "nts-utc",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    drop_cols = set(meta_cols)
    drop_cols.add(label_col)
    if ambig_col in df.columns:
        drop_cols.add(ambig_col)

    for col in df.columns:
        if "label" in col.lower() or "target" in col.lower():
            drop_cols.add(col)

    candidate_cols = [col for col in df.columns if col not in drop_cols]
    if not candidate_cols:
        return [], pd.DataFrame()

    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    missing_pct = numeric_df.isna().mean()
    keep_cols = [
        col for col in candidate_cols if missing_pct.get(col, 1.0) <= drop_missing_above
    ]

    non_empty_cols = [col for col in keep_cols if not numeric_df[col].isna().all()]
    constant_cols = [
        col for col in non_empty_cols if numeric_df[col].nunique(dropna=False) <= 1
    ]
    feature_cols = [col for col in non_empty_cols if col not in constant_cols]

    missing_table = missing_pct.sort_values(ascending=False).reset_index()
    missing_table.columns = ["column", "missing_pct"]
    return feature_cols, missing_table