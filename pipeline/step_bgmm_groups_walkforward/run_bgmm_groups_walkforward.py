#!/usr/bin/env python
"""
Walk-forward BGMM evaluation for BTC daily indicators with group ablations and stability checks.

Notes:
- Indicators in the input CSV are assumed pre-shifted to avoid leakage.
- Grouping, imputation, and scaling are fit on TRAIN only and applied to TEST.
- For each (split, group, K, cov_type) we run multiple seeds and select the best-seed
  metrics by Test_AvgLogLik; stability stats are computed across all seeds.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
GMM_DIR = CURRENT_DIR.parent / "step_gmm_groups_walkforward"
if str(GMM_DIR) not in sys.path:
    sys.path.append(str(GMM_DIR))

from gmm_grouping import GroupDefinition, build_corr_groups, build_domain_groups, feature_list_short
from gmm_metrics import compute_stability, fit_gmm_and_score, select_best_run


INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
OUTPUT_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_bgmm_groups_walkforward\results")
LOG_DIR = OUTPUT_DIR.parent / "logs"

DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

TRAIN_YEARS_LIST = [2, 3, 4]
TEST_YEARS = 1
STEP_YEARS = 1

COVARIANCE_TYPES = ["tied", "full"]
K_RANGE = list(range(2, 11))
SEEDS = [0, 1, 2, 3, 4]

MISSING_COL_THRESHOLD = 0.20
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 10

TIMESTAMP_CANDIDATES = ["ts", "timestamp", "date", "datetime", "ts_utc"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]

METRICS_COLUMNS = [
    "Model_ID",
    "Group_Method",
    "Group_ID",
    "Group_Name",
    "N_Features",
    "Feature_List_Short",
    "Covariance_Type",
    "K",
    "Train_Start",
    "Train_End",
    "Test_Start",
    "Test_End",
    "Train_AvgLogLik",
    "Test_AvgLogLik",
    "Train_BIC",
    "Train_AIC",
    "Test_BIC",
    "Test_AIC",
    "Train_Silhouette",
    "Train_DaviesBouldin",
    "Test_Silhouette",
    "Test_DaviesBouldin",
    "Train_RespEntropy",
    "Test_RespEntropy",
    "MultiRun_LL_Std",
    "MultiRun_Weight_Std_Mean",
    "MultiRun_MeanShift_Std_Mean",
    "PCA_Used",
    "PCA_Components",
    "Rank_Type",
]


@dataclass
class SplitDef:
    split_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_years: int


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "run_bgmm_groups_walkforward.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def detect_timestamp_column(columns: Iterable[str]) -> str | None:
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
) -> None | tuple[np.ndarray, np.ndarray, List[str], Dict[str, float], np.ndarray]:
    train_df = df[(df[ts_col] >= split.train_start) & (df[ts_col] <= split.train_end)].copy()
    test_df = df[(df[ts_col] >= split.test_start) & (df[ts_col] <= split.test_end)].copy()

    if train_df.empty or test_df.empty:
        logging.warning("Split %s has empty train or test; skipping.", split.split_id)
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
    X_test = imputer.transform(test_df[keep_cols])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    variances = dict(zip(keep_cols, np.var(X_train, axis=0, ddof=0)))
    if len(keep_cols) > 1:
        corr = np.corrcoef(X_train_scaled, rowvar=False)
        abs_corr = np.abs(np.nan_to_num(corr, nan=0.0))
    else:
        abs_corr = np.eye(len(keep_cols))

    return X_train_scaled, X_test_scaled, keep_cols, variances, abs_corr


def evaluate_config(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cov_type: str,
    k: int,
    seeds: Sequence[int],
) -> tuple[Dict[str, float], float, float, float]:
    runs = []
    for seed in seeds:
        run = fit_gmm_and_score(
            X_train,
            X_test,
            covariance_type=cov_type,
            k=k,
            seed=seed,
            model_class=BayesianGaussianMixture,
        )
        runs.append(run)

    best_idx = select_best_run(runs)
    best = runs[best_idx]
    ll_std, weight_std_mean, mean_shift_std_mean = compute_stability(runs)

    metrics = {
        "Train_AvgLogLik": best.train_avg_ll,
        "Test_AvgLogLik": best.test_avg_ll,
        "Train_BIC": best.train_bic,
        "Train_AIC": best.train_aic,
        "Test_BIC": best.test_bic,
        "Test_AIC": best.test_aic,
        "Train_Silhouette": best.train_silhouette,
        "Train_DaviesBouldin": best.train_davies_bouldin,
        "Test_Silhouette": best.test_silhouette,
        "Test_DaviesBouldin": best.test_davies_bouldin,
        "Train_RespEntropy": best.train_resp_entropy,
        "Test_RespEntropy": best.test_resp_entropy,
    }

    return metrics, ll_std, weight_std_mean, mean_shift_std_mean


def build_group_indices(features: Sequence[str]) -> Dict[str, int]:
    return {f: i for i, f in enumerate(features)}


def slice_group_data(X: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if X.size == 0:
        return X
    return X[:, indices]


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


def is_extreme_entropy(test_entropy: float, k: int) -> bool:
    if math.isnan(test_entropy):
        return False
    upper = math.log(k) * 0.9
    return test_entropy < 0.02 or test_entropy > upper


def rank_top_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_id, cov_type), sub in metrics_df.groupby(["Split_ID", "Covariance_Type"]):
        sub = sub.copy()
        sub["ExtremeEntropy"] = sub.apply(lambda r: is_extreme_entropy(float(r["Test_RespEntropy"]), int(r["K"])), axis=1)
        filtered = sub[~sub["ExtremeEntropy"]]
        if filtered.empty:
            candidate = sub
            logging.warning("All configs extreme entropy for %s | %s; using best available.", split_id, cov_type)
        else:
            candidate = filtered

        candidate = candidate.sort_values(
            by=["Test_AvgLogLik", "Train_BIC", "MultiRun_LL_Std"],
            ascending=[False, True, True],
        )
        top = candidate.head(1).copy()
        top["Rank_Type"] = "Top_by_Test_AvgLogLik"
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=METRICS_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    logging.info("Loading data from %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        raise ValueError(f"Could not detect timestamp column among {TIMESTAMP_CANDIDATES}")

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)
    df = df.drop_duplicates(subset=[ts_col], keep="last")

    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)
    df = df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()
    logging.info("Data rows after date filter: %d", len(df))

    numeric_features = select_numeric_features(df, ts_col)
    logging.info("Numeric indicator features (excluding OHLCV): %d", len(numeric_features))
    if not numeric_features:
        raise RuntimeError("No numeric indicator columns found after excluding OHLCV.")

    splits = generate_walkforward_splits(DATE_START, DATE_END, TRAIN_YEARS_LIST, TEST_YEARS, STEP_YEARS)
    logging.info("Generated %d walk-forward splits.", len(splits))

    all_rows: List[Dict[str, float]] = []

    for split_idx, split in enumerate(splits, start=1):
        split_rows_start = len(all_rows)
        logging.info(
            "Split %d/%d | %s | Train %s..%s | Test %s..%s",
            split_idx,
            len(splits),
            split.split_id,
            split.train_start.date(),
            split.train_end.date(),
            split.test_start.date(),
            split.test_end.date(),
        )

        pre = preprocess_split(df, ts_col, numeric_features, split)
        if pre is None:
            continue
        X_train, X_test, feature_names, variances, abs_corr = pre
        if len(feature_names) < MIN_GROUP_SIZE:
            logging.warning("Split %s has only %d features after filters; skipping.", split.split_id, len(feature_names))
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
        if not base_groups:
            logging.warning("Split %s has no groups; skipping.", split.split_id)
            continue

        idx_map = build_group_indices(feature_names)

        group_best_tied: Dict[str, float] = {}
        group_rows_cache: Dict[str, List[Dict[str, float]]] = {}

        for g in base_groups:
            indices = [idx_map[f] for f in g.features if f in idx_map]
            if len(indices) < MIN_GROUP_SIZE:
                continue
            X_train_g = slice_group_data(X_train, indices)
            X_test_g = slice_group_data(X_test, indices)
            group_rows_cache[g.group_id] = []

            for cov_type in COVARIANCE_TYPES:
                for k in K_RANGE:
                    metrics, ll_std, weight_std_mean, mean_shift_std_mean = evaluate_config(
                        X_train_g, X_test_g, cov_type, k, SEEDS
                    )
                    model_id = f"bgmm_wf_{split.split_id}_{g.group_id}_{cov_type}_K{k}"
                    row = {
                        "Model_ID": model_id,
                        "Group_Method": g.method,
                        "Group_ID": g.group_id,
                        "Group_Name": g.group_name,
                        "N_Features": len(indices),
                        "Feature_List_Short": feature_list_short(g.features),
                        "Covariance_Type": cov_type,
                        "K": k,
                        "Train_Start": split.train_start.strftime("%Y-%m-%d"),
                        "Train_End": split.train_end.strftime("%Y-%m-%d"),
                        "Test_Start": split.test_start.strftime("%Y-%m-%d"),
                        "Test_End": split.test_end.strftime("%Y-%m-%d"),
                        "Train_AvgLogLik": metrics["Train_AvgLogLik"],
                        "Test_AvgLogLik": metrics["Test_AvgLogLik"],
                        "Train_BIC": metrics["Train_BIC"],
                        "Train_AIC": metrics["Train_AIC"],
                        "Test_BIC": metrics["Test_BIC"],
                        "Test_AIC": metrics["Test_AIC"],
                        "Train_Silhouette": metrics["Train_Silhouette"],
                        "Train_DaviesBouldin": metrics["Train_DaviesBouldin"],
                        "Test_Silhouette": metrics["Test_Silhouette"],
                        "Test_DaviesBouldin": metrics["Test_DaviesBouldin"],
                        "Train_RespEntropy": metrics["Train_RespEntropy"],
                        "Test_RespEntropy": metrics["Test_RespEntropy"],
                        "MultiRun_LL_Std": ll_std,
                        "MultiRun_Weight_Std_Mean": weight_std_mean,
                        "MultiRun_MeanShift_Std_Mean": mean_shift_std_mean,
                        "PCA_Used": 0,
                        "PCA_Components": 0,
                        "Rank_Type": "",
                        "Split_ID": split.split_id,
                    }
                    all_rows.append(row)
                    group_rows_cache[g.group_id].append(row)

            tied_rows = [r for r in group_rows_cache[g.group_id] if r["Covariance_Type"] == "tied"]
            if tied_rows:
                best_tied = max(tied_rows, key=lambda r: r["Test_AvgLogLik"])
                group_best_tied[g.group_id] = float(best_tied["Test_AvgLogLik"])

        stack_groups = add_stack_groups(base_groups, group_best_tied, max_stack=4)
        for g in stack_groups:
            indices = [idx_map[f] for f in g.features if f in idx_map]
            if len(indices) < MIN_GROUP_SIZE:
                continue
            X_train_g = slice_group_data(X_train, indices)
            X_test_g = slice_group_data(X_test, indices)

            for cov_type in COVARIANCE_TYPES:
                for k in K_RANGE:
                    metrics, ll_std, weight_std_mean, mean_shift_std_mean = evaluate_config(
                        X_train_g, X_test_g, cov_type, k, SEEDS
                    )
                    model_id = f"bgmm_wf_{split.split_id}_{g.group_id}_{cov_type}_K{k}"
                    row = {
                        "Model_ID": model_id,
                        "Group_Method": g.method,
                        "Group_ID": g.group_id,
                        "Group_Name": g.group_name,
                        "N_Features": len(indices),
                        "Feature_List_Short": feature_list_short(g.features),
                        "Covariance_Type": cov_type,
                        "K": k,
                        "Train_Start": split.train_start.strftime("%Y-%m-%d"),
                        "Train_End": split.train_end.strftime("%Y-%m-%d"),
                        "Test_Start": split.test_start.strftime("%Y-%m-%d"),
                        "Test_End": split.test_end.strftime("%Y-%m-%d"),
                        "Train_AvgLogLik": metrics["Train_AvgLogLik"],
                        "Test_AvgLogLik": metrics["Test_AvgLogLik"],
                        "Train_BIC": metrics["Train_BIC"],
                        "Train_AIC": metrics["Train_AIC"],
                        "Test_BIC": metrics["Test_BIC"],
                        "Test_AIC": metrics["Test_AIC"],
                        "Train_Silhouette": metrics["Train_Silhouette"],
                        "Train_DaviesBouldin": metrics["Train_DaviesBouldin"],
                        "Test_Silhouette": metrics["Test_Silhouette"],
                        "Test_DaviesBouldin": metrics["Test_DaviesBouldin"],
                        "Train_RespEntropy": metrics["Train_RespEntropy"],
                        "Test_RespEntropy": metrics["Test_RespEntropy"],
                        "MultiRun_LL_Std": ll_std,
                        "MultiRun_Weight_Std_Mean": weight_std_mean,
                        "MultiRun_MeanShift_Std_Mean": mean_shift_std_mean,
                        "PCA_Used": 0,
                        "PCA_Components": 0,
                        "Rank_Type": "",
                        "Split_ID": split.split_id,
                    }
                    all_rows.append(row)

        split_rows = len(all_rows) - split_rows_start
        logging.info("Split %s completed with %d rows.", split.split_id, split_rows)

    if not all_rows:
        logging.warning("No results generated.")
        return 1

    metrics_df = pd.DataFrame(all_rows)
    for col in METRICS_COLUMNS:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_df = metrics_df[METRICS_COLUMNS + ["Split_ID"]]

    metrics_out = metrics_df[METRICS_COLUMNS].copy()
    metrics_out.to_csv(OUTPUT_DIR / "metrics_all.csv", index=False)

    metrics_tied = metrics_df[metrics_df["Covariance_Type"] == "tied"].copy()
    metrics_full = metrics_df[metrics_df["Covariance_Type"] == "full"].copy()

    metrics_tied[METRICS_COLUMNS].to_csv(OUTPUT_DIR / "metrics_tied.csv", index=False)
    metrics_full[METRICS_COLUMNS].to_csv(OUTPUT_DIR / "metrics_full.csv", index=False)

    top_configs = rank_top_configs(metrics_df)
    if not top_configs.empty:
        for _, row in top_configs.iterrows():
            logging.info(
                "Top | %s | cov=%s | group=%s | K=%s | TestLL=%.6f | TrainBIC=%.2f | LL_Std=%.6f",
                row.get("Split_ID", ""),
                row.get("Covariance_Type", ""),
                row.get("Group_ID", ""),
                row.get("K", ""),
                float(row.get("Test_AvgLogLik", float("nan"))),
                float(row.get("Train_BIC", float("nan"))),
                float(row.get("MultiRun_LL_Std", float("nan"))),
            )
    top_configs[METRICS_COLUMNS].to_csv(OUTPUT_DIR / "top_configs.csv", index=False)

    logging.info("Outputs written to %s", OUTPUT_DIR)
    logging.info("metrics_all.csv rows: %d", len(metrics_df))
    logging.info("metrics_tied.csv rows: %d", len(metrics_tied))
    logging.info("metrics_full.csv rows: %d", len(metrics_full))
    logging.info("top_configs.csv rows: %d", len(top_configs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

