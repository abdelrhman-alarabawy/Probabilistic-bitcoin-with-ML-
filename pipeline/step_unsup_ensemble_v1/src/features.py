from __future__ import annotations

import re

import numpy as np
import pandas as pd


def select_feature_columns(
    numeric_cols: list[str],
    include_spec: str | list[str],
    exclude_cols: list[str] | None = None,
) -> list[str]:
    exclude = set(exclude_cols or [])
    include_items = include_spec if isinstance(include_spec, list) else [include_spec]

    selected: set[str] = set()
    for item in include_items:
        if isinstance(item, str) and item.upper() == "ALL_NUMERIC":
            selected.update([c for c in numeric_cols if c not in exclude])
        elif isinstance(item, str) and item.startswith("REGEX:"):
            pattern = item.split("REGEX:", 1)[1]
            regex = re.compile(pattern)
            selected.update([c for c in numeric_cols if regex.search(c)])
        elif isinstance(item, str) and item in numeric_cols:
            selected.add(item)
        else:
            if not isinstance(item, str):
                raise ValueError(f"Unsupported include item: {item}")

    output = [c for c in numeric_cols if c in selected and c not in exclude]
    if not output:
        raise ValueError("No feature columns selected.")
    return output


def select_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    featureset_cfg: dict,
) -> tuple[np.ndarray, list[str]]:
    cols = select_feature_columns(
        numeric_cols=numeric_cols,
        include_spec=featureset_cfg.get("include", "ALL_NUMERIC"),
        exclude_cols=featureset_cfg.get("exclude", []),
    )
    return df[cols].to_numpy(dtype=float), cols
