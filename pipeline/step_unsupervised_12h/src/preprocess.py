from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


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
class PreprocessArtifacts:
    imputer: SimpleImputer
    scaler: object
    feature_columns: list[str]


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


def load_raw_csv(path: Path) -> tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ts_col = detect_timestamp_column(df.columns)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.sort_values(ts_col)
    return df, ts_col


def select_numeric_features(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    exclude_columns: Iterable[str],
) -> list[str]:
    exclude = {col for col in exclude_columns if col is not None}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude]
    non_constant = [
        col for col in feature_cols if df[col].nunique(dropna=True) > 1
    ]
    return non_constant


def fit_transform_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: slice,
    scaler_type: str = "robust",
) -> tuple[np.ndarray, PreprocessArtifacts]:
    train_data = df.iloc[train_idx][feature_cols]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_data)
    train_imputed = imputer.transform(train_data)

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    scaler.fit(train_imputed)

    full_imputed = imputer.transform(df[feature_cols])
    full_scaled = scaler.transform(full_imputed)
    artifacts = PreprocessArtifacts(
        imputer=imputer,
        scaler=scaler,
        feature_columns=feature_cols,
    )
    return full_scaled, artifacts


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            summary[col] = float("nan")
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            summary[col] = 0.0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        pct = ((series < lower) | (series > upper)).mean() * 100
        summary[col] = float(pct)
    return summary


def build_quality_report(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    numeric_cols: list[str],
) -> dict:
    report: dict[str, object] = {}
    report["rows"] = int(len(df))
    report["columns"] = int(df.shape[1])
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
    outlier_cols = numeric_cols[:30]
    report["outlier_pct_iqr"] = outlier_summary(df, outlier_cols)
    return report


def save_quality_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, default=str, indent=2)
