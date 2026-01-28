#!/usr/bin/env python
"""
Export top_configs with full (non-truncated) feature lists.

This script reconstructs group definitions per split using the same grouping
logic and uses metrics_tied.csv to recover the stack-group ordering.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from gmm_grouping import GroupDefinition, build_corr_groups, build_domain_groups


INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
RESULTS_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_gmm_groups_walkforward\results")

DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

TRAIN_YEARS_LIST = [2, 3, 4]
TEST_YEARS = 1
STEP_YEARS = 1

MISSING_COL_THRESHOLD = 0.20
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 10

TIMESTAMP_CANDIDATES = ["ts", "timestamp", "date", "datetime", "ts_utc"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]


@dataclass
class SplitDef:
    split_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_years: int


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def detect_timestamp_column(columns: Iterable[str]) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for name in TIMESTAMP_CANDIDATES:
        if name in col_map:
            return col_map[name]
    for name in TIMESTAMP_CANDIDATES:
        for c in columns:
            if name in c.lower():
                return c
    return None


def detect_ohlcv_columns(columns: Iterable[str]) -> List[str]:
    col_map = {c.lower(): c for c in columns}
    return [col_map[c] for c in OHLCV_CANDIDATES if c in col_map]


def generate_walkforward_splits(
    start_date: str,
    end_date: str,
    train_years_list: Sequence[int],
    test_years: int,
    step_years: int,
) -> List[SplitDef]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    splits: List[SplitDef] = []

    for train_years in train_years_list:
        split_start = start
        while True:
            train_start = split_start
            train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
            if test_end > end:
                break
            split_id = (
                f"{train_years}y_"
                f"{train_start:%Y%m%d}_{train_end:%Y%m%d}_"
                f"{test_start:%Y%m%d}_{test_end:%Y%m%d}"
            )
            splits.append(
                SplitDef(
                    split_id=split_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_years=train_years,
                )
            )
            split_start = split_start + pd.DateOffset(years=step_years)

    splits.sort(key=lambda s: (s.train_start, s.train_years))
    return splits


def select_numeric_features(df: pd.DataFrame, ts_col: str) -> List[str]:
    ohlcv_cols = set(c.lower() for c in detect_ohlcv_columns(df.columns))
    numeric_cols = []
    for c in df.columns:
        if c == ts_col:
            continue
        if c.lower() in ohlcv_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def preprocess_split(
    df: pd.DataFrame,
    ts_col: str,
    feature_candidates: Sequence[str],
    split: SplitDef,
) -> Optional[Tuple[np.ndarray, List[str], Dict[str, float], np.ndarray]]:
    train_df = df[(df[ts_col] >= split.train_start) & (df[ts_col] <= split.train_end)].copy()
    test_df = df[(df[ts_col] >= split.test_start) & (df[ts_col] <= split.test_end)].copy()

    if train_df.empty or test_df.empty:
        logging.warning("Split %s has empty train/test; skipping.", split.split_id)
        return None

    missing_ratio = train_df[feature_candidates].isna().mean()
    keep_cols = missing_ratio[missing_ratio <= MISSING_COL_THRESHOLD].index.tolist()
    if not keep_cols:
        logging.warning("Split %s dropped all features after missing threshold.", split.split_id)
        return None

    variances_raw = train_df[keep_cols].var(skipna=True)
    keep_cols = [c for c in keep_cols if variances_raw.get(c, 0.0) > 0]
    if not keep_cols:
        logging.warning("Split %s dropped all features after variance filter.", split.split_id)
        return None

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[keep_cols])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    variances = dict(zip(keep_cols, np.var(X_train, axis=0, ddof=0)))
    if len(keep_cols) > 1:
        corr = np.corrcoef(X_train_scaled, rowvar=False)
        abs_corr = np.abs(np.nan_to_num(corr, nan=0.0))
    else:
        abs_corr = np.eye(len(keep_cols))

    return X_train_scaled, keep_cols, variances, abs_corr


def build_group_indices(features: Sequence[str]) -> Dict[str, int]:
    return {f: i for i, f in enumerate(features)}


def add_stack_groups(
    groups: List[GroupDefinition],
    group_best_tied: Dict[str, float],
    max_stack: int = 4,
) -> List[GroupDefinition]:
    ordered = sorted(groups, key=lambda g: group_best_tied.get(g.group_id, -math.inf), reverse=True)
    if not ordered:
        return []

    stacked_groups: List[GroupDefinition] = []
    selected_features: List[str] = []
    for i in range(min(max_stack, len(ordered))):
        g = ordered[i]
        for f in g.features:
            if f not in selected_features:
                selected_features.append(f)
        group_id = f"stack_{i+1}"
        stacked_groups.append(
            GroupDefinition(
                method="stack",
                group_id=group_id,
                group_name=f"stack_{i+1}",
                features=list(selected_features),
            )
        )
    return stacked_groups


def parse_split_id(model_id: str, group_id: str, cov_type: str, k: int) -> Optional[str]:
    prefix = "gmm_wf_"
    if not model_id.startswith(prefix):
        return None
    remainder = model_id[len(prefix):]
    suffix = f"_{group_id}_{cov_type}_K{int(k)}"
    if remainder.endswith(suffix):
        return remainder[: -len(suffix)]
    # Fallback: try splitting by covariance marker
    marker = f"_{cov_type}_K"
    if marker in remainder:
        left = remainder.split(marker)[0]
        if left.endswith(f"_{group_id}"):
            return left[: -len(group_id) - 1]
    return None


def load_csv_strip(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def main() -> int:
    setup_logging()
    top_path = RESULTS_DIR / "top_configs.csv"
    metrics_tied_path = RESULTS_DIR / "metrics_tied.csv"

    if not top_path.exists():
        logging.error("Missing %s", top_path)
        return 1
    if not metrics_tied_path.exists():
        logging.error("Missing %s", metrics_tied_path)
        return 1

    top_df = load_csv_strip(top_path)
    tied_df = load_csv_strip(metrics_tied_path)

    required_cols = {"Model_ID", "Group_ID", "Covariance_Type", "K"}
    if not required_cols.issubset(set(top_df.columns)):
        logging.error("top_configs.csv missing required columns: %s", required_cols)
        return 1

    # Add Split_ID to top_df via Model_ID parsing
    split_ids: List[Optional[str]] = []
    for _, row in top_df.iterrows():
        split_id = parse_split_id(
            str(row["Model_ID"]),
            str(row["Group_ID"]),
            str(row["Covariance_Type"]),
            int(float(row["K"])),
        )
        split_ids.append(split_id)
    top_df["Split_ID"] = split_ids

    missing_split = top_df["Split_ID"].isna().sum()
    if missing_split:
        logging.warning("Could not parse Split_ID for %d rows.", missing_split)

    # Add Split_ID to tied_df via Model_ID parsing
    tied_split_ids: List[Optional[str]] = []
    for _, row in tied_df.iterrows():
        split_id = parse_split_id(
            str(row["Model_ID"]),
            str(row["Group_ID"]),
            str(row["Covariance_Type"]),
            int(float(row["K"])),
        )
        tied_split_ids.append(split_id)
    tied_df["Split_ID"] = tied_split_ids

    df = pd.read_csv(INPUT_CSV)
    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        logging.error("Could not detect timestamp column among %s", TIMESTAMP_CANDIDATES)
        return 1

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)
    df = df.drop_duplicates(subset=[ts_col], keep="last")

    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)
    df = df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()

    numeric_features = select_numeric_features(df, ts_col)
    if not numeric_features:
        logging.error("No numeric indicator columns found after excluding OHLCV.")
        return 1

    splits = generate_walkforward_splits(DATE_START, DATE_END, TRAIN_YEARS_LIST, TEST_YEARS, STEP_YEARS)
    split_map = {s.split_id: s for s in splits}

    group_feature_map: Dict[Tuple[str, str], List[str]] = {}

    needed_splits = sorted(set([s for s in top_df["Split_ID"].dropna()]))
    for split_id in needed_splits:
        split = split_map.get(split_id)
        if split is None:
            logging.warning("Split %s not found in generated splits.", split_id)
            continue

        pre = preprocess_split(df, ts_col, numeric_features, split)
        if pre is None:
            continue
        X_train_scaled, feature_names, variances, abs_corr = pre
        if len(feature_names) < MIN_GROUP_SIZE:
            logging.warning("Split %s has only %d features; skipping.", split_id, len(feature_names))
            continue

        corr_groups = build_corr_groups(
            feature_names,
            abs_corr,
            variances=variances,
            min_size=MIN_GROUP_SIZE,
            max_size=MAX_GROUP_SIZE,
            logger=logging,
        )
        domain_groups = build_domain_groups(
            feature_names,
            variances=variances,
            abs_corr=abs_corr,
            min_size=MIN_GROUP_SIZE,
            max_size=MAX_GROUP_SIZE,
            logger=logging,
        )
        base_groups = corr_groups + domain_groups

        tied_split = tied_df[tied_df["Split_ID"] == split_id]
        group_best_tied: Dict[str, float] = {}
        for group_id, sub in tied_split.groupby("Group_ID"):
            if group_id.startswith("stack_"):
                continue
            try:
                best_ll = float(sub["Test_AvgLogLik"].astype(float).max())
            except Exception:
                best_ll = float("nan")
            group_best_tied[group_id] = best_ll

        stack_groups = add_stack_groups(base_groups, group_best_tied, max_stack=4)
        all_groups = base_groups + stack_groups

        for g in all_groups:
            group_feature_map[(split_id, g.group_id)] = list(g.features)

    feature_full = []
    for _, row in top_df.iterrows():
        split_id = row.get("Split_ID")
        group_id = row.get("Group_ID")
        feats = group_feature_map.get((split_id, group_id), [])
        feature_full.append("|".join(feats))
    top_df["Feature_List_Full"] = feature_full

    out_path = RESULTS_DIR / "top_configs_full.csv"
    top_df.to_csv(out_path, index=False)
    logging.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

