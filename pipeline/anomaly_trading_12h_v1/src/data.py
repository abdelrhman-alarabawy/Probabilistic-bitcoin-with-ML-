from __future__ import annotations

import pandas as pd

from .config import TIMESTAMP_COL


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Missing {TIMESTAMP_COL} column")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")
    df = df.reset_index(drop=True)
    return df
