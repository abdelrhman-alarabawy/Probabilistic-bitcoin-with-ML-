from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FoldDefinition:
    fold_id: str
    fold_type: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: Optional[str]
    train_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]

    @property
    def train_size(self) -> int:
        return int(self.train_idx.shape[0])

    @property
    def test_size(self) -> int:
        return int(self.test_idx.shape[0])


def _indices_to_boundaries(
    idx: np.ndarray,
    timestamps: Optional[pd.Series],
) -> tuple[Optional[str], Optional[str]]:
    if idx.size == 0:
        return None, None
    if timestamps is None:
        return str(int(idx.min())), str(int(idx.max()))
    ts_slice = timestamps.iloc[idx]
    if ts_slice.empty:
        return None, None
    return str(ts_slice.iloc[0]), str(ts_slice.iloc[-1])


def _make_holdout_fold(
    n_rows: int,
    train_ratio: float,
    timestamps: Optional[pd.Series],
) -> FoldDefinition:
    train_end = int(round(n_rows * train_ratio))
    train_end = max(1, min(train_end, n_rows - 1))

    train_idx = np.arange(0, train_end, dtype=int)
    test_idx = np.arange(train_end, n_rows, dtype=int)
    train_start, train_end_str = _indices_to_boundaries(train_idx, timestamps)
    test_start, test_end_str = _indices_to_boundaries(test_idx, timestamps)
    return FoldDefinition(
        fold_id="holdout_80_20",
        fold_type="holdout",
        train_idx=train_idx,
        test_idx=test_idx,
        train_start=train_start,
        train_end=train_end_str,
        test_start=test_start,
        test_end=test_end_str,
    )


def _make_walkforward_folds_with_timestamps(
    timestamps: pd.Series,
    train_months: int,
    test_months: int,
    step_months: int,
    min_train_samples: int,
    max_folds: Optional[int],
) -> List[FoldDefinition]:
    folds: List[FoldDefinition] = []
    ts = timestamps.reset_index(drop=True)
    min_ts = ts.min()
    max_ts = ts.max()

    if pd.isna(min_ts) or pd.isna(max_ts):
        return folds

    fold_counter = 0
    start = min_ts
    while True:
        train_end_exclusive = start + pd.DateOffset(months=train_months)
        test_end_exclusive = train_end_exclusive + pd.DateOffset(months=test_months)
        if test_end_exclusive > max_ts + pd.Timedelta(seconds=1):
            break

        train_idx = np.flatnonzero((ts >= start) & (ts < train_end_exclusive))
        test_idx = np.flatnonzero((ts >= train_end_exclusive) & (ts < test_end_exclusive))
        if train_idx.size >= min_train_samples and test_idx.size > 0:
            fold_counter += 1
            train_start, train_end = _indices_to_boundaries(train_idx, ts)
            test_start, test_end = _indices_to_boundaries(test_idx, ts)
            folds.append(
                FoldDefinition(
                    fold_id=f"walkforward_{fold_counter:03d}",
                    fold_type="walkforward",
                    train_idx=train_idx.astype(int),
                    test_idx=test_idx.astype(int),
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            if max_folds is not None and len(folds) >= max_folds:
                break
        start = start + pd.DateOffset(months=step_months)
        if start >= max_ts:
            break
    return folds


def _make_walkforward_folds_row_based(
    n_rows: int,
    train_size: int,
    test_size: int,
    step_size: int,
    max_folds: Optional[int],
) -> List[FoldDefinition]:
    folds: List[FoldDefinition] = []
    start = 0
    fold_counter = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_end = train_end + test_size
        if test_end > n_rows:
            break
        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(train_end, test_end, dtype=int)
        fold_counter += 1
        folds.append(
            FoldDefinition(
                fold_id=f"walkforward_{fold_counter:03d}",
                fold_type="walkforward",
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=str(train_start),
                train_end=str(train_end - 1),
                test_start=str(train_end),
                test_end=str(test_end - 1),
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
        start += step_size
        if start >= n_rows:
            break
    return folds


def _make_year_based_folds(
    timestamps: pd.Series,
    min_train_years: int,
) -> List[FoldDefinition]:
    ts = timestamps.reset_index(drop=True)
    years = sorted(int(y) for y in ts.dropna().dt.year.unique())
    folds: List[FoldDefinition] = []

    if len(years) < (min_train_years + 1):
        return folds

    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_idx = np.flatnonzero(ts.dt.year.isin(train_years).to_numpy())
        test_idx = np.flatnonzero((ts.dt.year == test_year).to_numpy())
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        train_start, train_end = _indices_to_boundaries(train_idx, ts)
        test_start, test_end = _indices_to_boundaries(test_idx, ts)
        folds.append(
            FoldDefinition(
                fold_id=f"year_based_train_{train_years[0]}_{train_years[-1]}_test_{test_year}",
                fold_type="year_based",
                train_idx=train_idx.astype(int),
                test_idx=test_idx.astype(int),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def build_folds(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    fold_cfg: Dict[str, object],
) -> List[FoldDefinition]:
    n_rows = int(df.shape[0])
    if n_rows < 2:
        raise ValueError("Not enough rows to create train/test folds.")

    enabled: Sequence[str] = fold_cfg.get("enabled", ["holdout", "walkforward", "year_based"])
    enabled_set = {name.strip().lower() for name in enabled}

    timestamps: Optional[pd.Series] = None
    if timestamp_col is not None and timestamp_col in df.columns:
        timestamps = df[timestamp_col]

    folds: List[FoldDefinition] = []

    if "holdout" in enabled_set:
        train_ratio = float(fold_cfg.get("holdout_train_ratio", 0.8))
        folds.append(_make_holdout_fold(n_rows, train_ratio, timestamps))

    if "walkforward" in enabled_set:
        wf_cfg = dict(fold_cfg.get("walkforward", {}))
        max_folds = wf_cfg.get("max_folds")
        max_folds = int(max_folds) if max_folds is not None else None
        if timestamps is not None and timestamps.notna().sum() > 0:
            folds.extend(
                _make_walkforward_folds_with_timestamps(
                    timestamps=timestamps,
                    train_months=int(wf_cfg.get("train_months", 18)),
                    test_months=int(wf_cfg.get("test_months", 6)),
                    step_months=int(wf_cfg.get("step_months", 3)),
                    min_train_samples=int(wf_cfg.get("min_train_samples", 100)),
                    max_folds=max_folds,
                )
            )
        else:
            logging.warning("Timestamp column unavailable. Falling back to row-based walk-forward windows.")
            folds.extend(
                _make_walkforward_folds_row_based(
                    n_rows=n_rows,
                    train_size=int(wf_cfg.get("fallback_row_train_size", 540)),
                    test_size=int(wf_cfg.get("fallback_row_test_size", 180)),
                    step_size=int(wf_cfg.get("fallback_row_step_size", 90)),
                    max_folds=max_folds,
                )
            )

    if "year_based" in enabled_set:
        if timestamps is None or timestamps.notna().sum() == 0:
            logging.warning("Skipping year-based folds because timestamp data is unavailable.")
        else:
            y_cfg = dict(fold_cfg.get("year_based", {}))
            folds.extend(
                _make_year_based_folds(
                    timestamps=timestamps,
                    min_train_years=int(y_cfg.get("min_train_years", 1)),
                )
            )

    deduped: List[FoldDefinition] = []
    seen = set()
    for fold in folds:
        key = (fold.fold_id, fold.train_size, fold.test_size)
        if fold.train_size == 0 or fold.test_size == 0:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fold)

    if not deduped:
        raise ValueError("No valid folds were generated with the provided configuration.")

    return deduped

