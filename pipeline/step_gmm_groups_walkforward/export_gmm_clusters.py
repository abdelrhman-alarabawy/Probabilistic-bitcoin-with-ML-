#!/usr/bin/env python
"""
Export cluster assignments for specific GMM model IDs using the same preprocessing
as the walk-forward pipeline. Outputs CSV with timestamp, OHLCV, indicators, and cluster.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
RESULTS_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_gmm_groups_walkforward\results")
TOP_CONFIGS_JSON = RESULTS_DIR / "top_configs_full.json"
OUTPUT_DIR = RESULTS_DIR / "clusters"

TIMESTAMP_CANDIDATES = ["ts", "timestamp", "date", "datetime", "ts_utc"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]

SEEDS = [0, 1, 2, 3, 4]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def detect_timestamp_column(columns: List[str]) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for name in TIMESTAMP_CANDIDATES:
        if name in col_map:
            return col_map[name]
    for name in TIMESTAMP_CANDIDATES:
        for c in columns:
            if name in c.lower():
                return c
    return None


def detect_ohlcv_columns(columns: List[str]) -> List[str]:
    col_map = {c.lower(): c for c in columns}
    return [col_map[c] for c in OHLCV_CANDIDATES if c in col_map]


def parse_model_id(model_id: str) -> Tuple[str, str, str, int]:
    prefix = "gmm_wf_"
    if not model_id.startswith(prefix):
        raise ValueError(f"Unexpected model_id format: {model_id}")
    remainder = model_id[len(prefix):]
    cov_type = None
    k = None
    for cov in ("tied", "full"):
        marker = f"_{cov}_K"
        if marker in remainder:
            left, right = remainder.rsplit(marker, 1)
            cov_type = cov
            try:
                k = int(right)
            except ValueError:
                k = None
            remainder = left
            break
    if cov_type is None or k is None:
        raise ValueError(f"Unexpected model_id format: {model_id}")

    # remainder now: {split_id}_{group_id}
    parts = remainder.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected split_id in model_id: {model_id}")
    split_id = "_".join(parts[:5])
    group_id = "_".join(parts[5:])
    return split_id, group_id, cov_type, k


def parse_split_dates(split_id: str) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    # Example: 4y_20210101_20241231_20250101_20251231
    parts = split_id.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected split_id format: {split_id}")
    train_start = pd.Timestamp(parts[1])
    train_end = pd.Timestamp(parts[2])
    test_start = pd.Timestamp(parts[3])
    test_end = pd.Timestamp(parts[4])
    return train_start, train_end, test_start, test_end


def load_top_configs() -> pd.DataFrame:
    df = pd.read_json(TOP_CONFIGS_JSON)
    if "Model_ID" not in df.columns:
        raise RuntimeError("top_configs_full.json missing Model_ID.")
    return df


def select_feature_list(row: pd.Series) -> List[str]:
    if "Feature_List_Full_List" in row and isinstance(row["Feature_List_Full_List"], list):
        return row["Feature_List_Full_List"]
    if "Feature_List_Full" in row and isinstance(row["Feature_List_Full"], str):
        return [p for p in row["Feature_List_Full"].split("|") if p]
    raise RuntimeError("No full feature list available in top_configs_full.json")


def fit_best_seed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    cov_type: str,
) -> GaussianMixture:
    best_model = None
    best_ll = -np.inf
    for seed in SEEDS:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            random_state=seed,
            n_init=1,
            max_iter=500,
            tol=1e-4,
            reg_covar=1e-6,
            init_params="kmeans",
        )
        gmm.fit(X_train)
        ll = float(np.mean(gmm.score_samples(X_test)))
        if ll > best_ll:
            best_ll = ll
            best_model = gmm
    if best_model is None:
        raise RuntimeError("Failed to fit any GMM seed.")
    return best_model


def build_output(
    df: pd.DataFrame,
    ts_col: str,
    ohlcv_cols: List[str],
    features: List[str],
    model_id: str,
    split_id: str,
    cov_type: str,
    k: int,
) -> Path:
    train_start, train_end, test_start, test_end = parse_split_dates(split_id)
    df_split = df[(df[ts_col] >= train_start) & (df[ts_col] <= test_end)].copy()
    if df_split.empty:
        raise RuntimeError(f"No data for split {split_id}")

    train_mask = (df_split[ts_col] >= train_start) & (df_split[ts_col] <= train_end)
    test_mask = (df_split[ts_col] >= test_start) & (df_split[ts_col] <= test_end)

    missing_features = [f for f in features if f not in df_split.columns]
    if missing_features:
        raise RuntimeError(f"Missing features in data: {missing_features}")

    X_train_raw = df_split.loc[train_mask, features].to_numpy()
    X_test_raw = df_split.loc[test_mask, features].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = fit_best_seed(X_train_s, X_test_s, k=k, cov_type=cov_type)

    labels_train = model.predict(X_train_s)
    labels_test = model.predict(X_test_s)

    out_df = df_split[[ts_col] + ohlcv_cols + features].copy()
    out_df["Set"] = np.where(train_mask, "train", np.where(test_mask, "test", ""))
    out_df["Cluster"] = np.nan
    out_df.loc[train_mask, "Cluster"] = labels_train
    out_df.loc[test_mask, "Cluster"] = labels_test
    out_df["Model_ID"] = model_id

    out_df = out_df[out_df["Set"] != ""].copy()
    out_df = out_df.sort_values(ts_col)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{model_id}_clusters.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    setup_logging()
    if not TOP_CONFIGS_JSON.exists():
        raise RuntimeError(f"Missing {TOP_CONFIGS_JSON}")

    model_ids = [
        "gmm_wf_4y_20210101_20241231_20250101_20251231_corr_2_tied_K9",
        "gmm_wf_4y_20200101_20231231_20240101_20241231_stack_3_tied_K3",
    ]

    top_df = load_top_configs()
    df = pd.read_csv(INPUT_CSV)
    ts_col = detect_timestamp_column(list(df.columns))
    if not ts_col:
        raise RuntimeError(f"Could not detect timestamp column among {TIMESTAMP_CANDIDATES}")

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)
    df = df.drop_duplicates(subset=[ts_col], keep="last")

    ohlcv_cols = detect_ohlcv_columns(list(df.columns))

    for model_id in model_ids:
        row = top_df[top_df["Model_ID"] == model_id]
        if row.empty:
            raise RuntimeError(f"Model_ID not found in top_configs_full.json: {model_id}")
        row = row.iloc[0]
        features = select_feature_list(row)
        split_id, _, cov_type, k = parse_model_id(model_id)
        out_path = build_output(df, ts_col, ohlcv_cols, features, model_id, split_id, cov_type, k)
        logging.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
