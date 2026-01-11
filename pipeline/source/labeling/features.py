from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd

from .config import LABEL_COLUMNS, META_COLUMNS


def select_feature_columns(
    df: pd.DataFrame,
    label_col: str = "candle_type",
    extra_drop: Iterable[str] | None = None,
) -> List[str]:
    drop_cols = set(META_COLUMNS)
    drop_cols.update(LABEL_COLUMNS)
    drop_cols.add(label_col)
    if extra_drop:
        drop_cols.update(extra_drop)

    candidate_cols = [col for col in df.columns if col not in drop_cols]
    if not candidate_cols:
        return []

    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    non_empty_cols = [col for col in candidate_cols if not numeric_df[col].isna().all()]
    constant_cols = [
        col for col in non_empty_cols if numeric_df[col].nunique(dropna=False) <= 1
    ]
    return [col for col in non_empty_cols if col not in constant_cols]


def build_features_and_labels(
    df: pd.DataFrame,
    label_col: str = "candle_type",
    extra_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = select_feature_columns(df, label_col=label_col, extra_drop=extra_drop)
    if not feature_cols:
        return pd.DataFrame(index=df.index), df[label_col].copy(), []

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[label_col].copy()
    return X, y, feature_cols
