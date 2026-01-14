from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "date",
    "datetime",
    "Datetime",
    "nts-utc",
    "nts_utc",
]


@dataclass
class PreprocessBundle:
    imputer: SimpleImputer
    scaler: RobustScaler
    feature_cols: list[str]
    use_regime: bool


def detect_timestamp_column(columns: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for col in columns:
        if "timestamp" in col.lower():
            return col
    return None


def detect_ohlcv_columns(columns: Iterable[str]) -> dict[str, str]:
    lower_map = {col.lower(): col for col in columns}
    required = ["open", "high", "low", "close", "volume"]
    mapping: dict[str, str] = {}
    for key in required:
        if key not in lower_map:
            raise ValueError(f"Missing required OHLCV column: {key}")
        mapping[key] = lower_map[key]
    return mapping


def numeric_feature_columns(
    df: pd.DataFrame,
    exclude: Iterable[str],
) -> list[str]:
    exclude_set = {col for col in exclude if col is not None}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_set]
    feature_cols = [col for col in feature_cols if df[col].nunique(dropna=True) > 1]
    return feature_cols


def fit_preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, SimpleImputer, RobustScaler]:
    train_data = df.loc[train_idx, feature_cols]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_data)
    train_imputed = imputer.transform(train_data)
    scaler = RobustScaler()
    scaler.fit(train_imputed)
    full_imputed = imputer.transform(df[feature_cols])
    full_scaled = scaler.transform(full_imputed)
    return full_scaled, imputer, scaler


def data_quality_report(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
) -> dict:
    report: dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }
    if timestamp_col:
        report["time_range"] = {
            "start": df[timestamp_col].min(),
            "end": df[timestamp_col].max(),
        }
        report["timestamp_duplicates"] = int(df[timestamp_col].duplicated().sum())
    else:
        report["time_range"] = None
        report["timestamp_duplicates"] = None
    missing_pct = (df.isna().mean() * 100).to_dict()
    report["missing_pct"] = {k: float(v) for k, v in missing_pct.items()}
    return report
