from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class Fold:
    name: str
    train_years: List[int]
    test_years: List[int]


def load_data(
    path: str,
    timestamp_col: str,
    drop_cols: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False)
    df = df.sort_values(timestamp_col)

    if drop_cols:
        cols = [c for c in drop_cols if c in df.columns]
        if cols:
            df = df.drop(columns=cols)

    if start_date:
        df = df[df[timestamp_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[timestamp_col] <= pd.to_datetime(end_date)]

    df = df.reset_index(drop=True)
    return df


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
        raise ValueError("Not enough years to create at least one fold.")

    folds: List[Fold] = []
    for i in range(min_train_years, len(all_years) - test_years + 1):
        train_years = all_years[:i]
        test_years_list = all_years[i : i + test_years]
        name = f"{train_years[0]}_{train_years[-1]}_test_{test_years_list[0]}"
        folds.append(Fold(name=name, train_years=train_years, test_years=test_years_list))

    return folds


def split_by_years(
    df: pd.DataFrame, timestamp_col: str, train_years: List[int], test_years: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[timestamp_col].dt.year.isin(train_years)].copy()
    test_df = df[df[timestamp_col].dt.year.isin(test_years)].copy()
    return train_df, test_df
