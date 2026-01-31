from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd


@dataclass
class Fold:
    name: str
    train_years: List[int]
    test_years: List[int]


def build_walkforward_folds(
    df: pd.DataFrame,
    timestamp_col: str,
    min_train_years: int,
    test_years: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> List[Fold]:
    years = sorted(df[timestamp_col].dt.year.unique().tolist())
    if not years:
        raise ValueError("No years found in dataset.")

    start_year = start_year or years[0]
    end_year = end_year or years[-1]
    all_years = [y for y in years if start_year <= y <= end_year]

    if len(all_years) < min_train_years + test_years:
        raise ValueError("Not enough years to create folds.")

    folds: List[Fold] = []
    for i in range(min_train_years, len(all_years) - test_years + 1):
        train_years = all_years[:i]
        test_years_list = all_years[i : i + test_years]
        name = "%d_%d_test_%d" % (train_years[0], train_years[-1], test_years_list[0])
        folds.append(Fold(name=name, train_years=train_years, test_years=test_years_list))

    return folds


def split_by_years(
    df: pd.DataFrame, timestamp_col: str, train_years: List[int], test_years: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[timestamp_col].dt.year.isin(train_years)].copy()
    test_df = df[df[timestamp_col].dt.year.isin(test_years)].copy()
    return train_df, test_df


def period_info(train_df: pd.DataFrame, test_df: pd.DataFrame, ts_col: str) -> Dict[str, str]:
    def _fmt(value) -> str:
        if pd.isna(value):
            return ""
        return pd.Timestamp(value).isoformat()

    return {
        "train_start": _fmt(train_df[ts_col].min()),
        "train_end": _fmt(train_df[ts_col].max()),
        "test_start": _fmt(test_df[ts_col].min()),
        "test_end": _fmt(test_df[ts_col].max()),
    }
