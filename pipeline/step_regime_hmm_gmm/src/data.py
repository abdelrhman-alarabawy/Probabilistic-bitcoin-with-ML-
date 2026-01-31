from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def load_data(
    path: str,
    timestamp_col: str,
    drop_cols: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError("Timestamp column not found: %s" % timestamp_col)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
    df = df[df[timestamp_col].notna()].copy()
    df = df.sort_values(timestamp_col)

    if drop_cols:
        cols = [c for c in drop_cols if c in df.columns]
        if cols:
            df = df.drop(columns=cols)

    if start_date:
        df = df[df[timestamp_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[timestamp_col] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)
