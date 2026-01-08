import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read config.yaml. Install with: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).lower(): col for col in columns}


def resolve_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    required: bool = True,
    label: Optional[str] = None,
) -> Optional[str]:
    column_map = normalize_columns(columns)
    for candidate in candidates or []:
        key = str(candidate).lower()
        if key in column_map:
            return column_map[key]
    if required:
        raise ValueError(
            f"Missing required column for {label or 'field'}. Tried: {list(candidates)}"
        )
    return None


def resolve_optional_columns(
    columns: Iterable[str], optional_map: Dict[str, List[str]]
) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    column_map = normalize_columns(columns)
    for key, candidates in optional_map.items():
        resolved[key] = None
        for candidate in candidates:
            match = column_map.get(str(candidate).lower())
            if match is not None:
                resolved[key] = match
                break
    return resolved


def resolve_all_candidates(
    columns: Iterable[str], candidates: Iterable[str]
) -> List[str]:
    column_map = normalize_columns(columns)
    matches: List[str] = []
    for candidate in candidates or []:
        match = column_map.get(str(candidate).lower())
        if match is not None:
            matches.append(match)
    return matches


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    shifted = series.shift(1)
    mean = shifted.rolling(window=window, min_periods=window).mean()
    std = shifted.rolling(window=window, min_periods=window).std()
    return (series - mean) / std


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(len(values))
        x_mean = x.mean()
        y_mean = values.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return np.nan
        return float(np.sum((x - x_mean) * (values - y_mean)) / denom)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def sigmoid(values: pd.Series, scale: float = 1.0) -> pd.Series:
    scaled = values / scale
    return 1.0 / (1.0 + np.exp(-scaled))


def discretize_quantile(
    series: pd.Series, bins: int, min_unique: int
) -> Tuple[Optional[pd.Series], Optional[List[float]]]:
    cleaned = series.dropna()
    if cleaned.nunique() < min_unique:
        return None, None
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.nanquantile(cleaned.values, quantiles))
    if len(edges) <= 2:
        return None, None
    codes = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    return codes, edges.tolist()
