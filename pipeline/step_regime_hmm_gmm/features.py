from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric


def build_feature_sets(df: pd.DataFrame, cfg: dict) -> Dict[str, List[str]]:
    features_cfg = cfg.get("features", {})
    timestamp_col = cfg.get("data", {}).get("timestamp_col")
    label_col = cfg.get("data", {}).get("label_col")
    exclude_columns = set(features_cfg.get("exclude_columns", []))
    if timestamp_col:
        exclude_columns.add(timestamp_col)
    if label_col:
        exclude_columns.add(label_col)

    price_columns = set(features_cfg.get("price_columns") or [])
    indicator_columns = features_cfg.get("indicator_columns")

    numeric_cols = _numeric_columns(df)

    feature_sets: Dict[str, List[str]] = {}
    for entry in features_cfg.get("feature_sets", []):
        name = entry.get("name")
        ftype = entry.get("type")
        if not name or not ftype:
            continue

        if ftype == "custom":
            cols = [c for c in entry.get("columns", []) if c in df.columns]
        elif ftype == "indicators_only":
            if indicator_columns:
                cols = [c for c in indicator_columns if c in df.columns]
            else:
                cols = [c for c in numeric_cols if c not in price_columns]
        elif ftype == "all_numeric":
            cols = list(numeric_cols)
        else:
            cols = list(numeric_cols)

        cols = [c for c in cols if c not in exclude_columns]
        cols = [c for c in cols if c in df.columns]
        if cols:
            feature_sets[name] = cols

    if not feature_sets:
        cols = [c for c in numeric_cols if c not in exclude_columns]
        if cols:
            feature_sets["all_numeric"] = cols

    return feature_sets
