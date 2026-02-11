from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import as_list


def _apply_regex(columns: list[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [c for c in columns if regex.search(c)]


def select_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    featureset_cfg: dict,
) -> Tuple[np.ndarray, list[str]]:
    include = featureset_cfg.get("include", "ALL_NUMERIC")
    exclude = set(as_list(featureset_cfg.get("exclude", [])))

    selected: set[str] = set()

    include_items = as_list(include)
    for item in include_items:
        if isinstance(item, str) and item.upper() == "ALL_NUMERIC":
            selected.update([c for c in numeric_cols if c not in exclude])
        elif isinstance(item, str) and item.startswith("REGEX:"):
            pattern = item.split("REGEX:", 1)[1]
            selected.update(_apply_regex(numeric_cols, pattern))
        elif isinstance(item, str):
            if item in numeric_cols:
                selected.add(item)
        else:
            raise ValueError(f"Unsupported include pattern: {item}")

    selected = [c for c in numeric_cols if c in selected and c not in exclude]
    if not selected:
        raise ValueError(f"No features selected for featureset '{featureset_cfg.get('name', '')}'")

    x = df[selected].to_numpy(dtype=float)
    return x, selected
