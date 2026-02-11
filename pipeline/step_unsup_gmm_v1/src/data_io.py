from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .utils import setup_logging


logger = setup_logging()


def load_dataset(cfg_data: dict, timezone: str = "UTC") -> Tuple[pd.DataFrame, list[str]]:
    csv_path = Path(cfg_data["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    ts_col = cfg_data.get("timestamp_col", "timestamp")
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found in {csv_path}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if timezone and str(timezone).upper() != "UTC":
        df[ts_col] = df[ts_col].dt.tz_convert(timezone)
    df = df.dropna(subset=[ts_col])

    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)

    for col in cfg_data.get("ohlcv_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns detected after loading dataset.")

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    if df[numeric_cols].isna().any().any():
        raise ValueError("NaNs remain after imputation.")

    logger.info("Loaded %d rows, %d numeric columns", len(df), len(numeric_cols))
    return df, numeric_cols
