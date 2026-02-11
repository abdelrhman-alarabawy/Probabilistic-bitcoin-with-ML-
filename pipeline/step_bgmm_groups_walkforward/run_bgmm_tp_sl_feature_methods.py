#!/usr/bin/env python
"""
Automatic BGMM walk-forward sweep over TP/SL labeling settings and multiple
feature-selection methods.

Outputs are written in the same style as the existing BGMM step:
- metrics_all.csv
- metrics_tied.csv
- metrics_full.csv
- top_configs.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
GMM_DIR = CURRENT_DIR.parent / "step_gmm_groups_walkforward"
if str(GMM_DIR) not in sys.path:
    sys.path.append(str(GMM_DIR))

from gmm_grouping import GroupDefinition, build_corr_groups, build_domain_groups, feature_list_short
from gmm_metrics import compute_stability, fit_gmm_and_score, select_best_run


INPUT_CSV = Path(r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv")
OUTPUT_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_bgmm_groups_walkforward\results\tp_sl_methods")
LOG_DIR = OUTPUT_DIR / "logs"

DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

TRAIN_YEARS_LIST = [2, 3, 4]
TEST_YEARS = 1
STEP_YEARS = 1

COVARIANCE_TYPES = ["tied", "full"]
K_RANGE = [2, 3, 4]
SEEDS = [0, 1]

TP_LIST = [1000.0, 1500.0, 2000.0, 2500.0]
SL_LIST = [500.0, 1000.0, 1500.0]

MISSING_COL_THRESHOLD = 0.20
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 10
VAR_TOP_SIZES = [5, 8, 10]
RANDOM_GROUPS_PER_SIZE = 3
RANDOM_SEED = 42

TIMESTAMP_CANDIDATES = ["ts", "timestamp", "date", "datetime", "ts_utc"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]

METRICS_COLUMNS = [
    "Model_ID",
    "TP",
    "SL",
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
    log_path = LOG_DIR / "run_bgmm_tp_sl_methods.log"
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


def select_numeric_features(df: pd.DataFrame, ts_col: str, label_col: str) -> List[str]:
    ohlcv_cols = set(c.lower() for c in detect_ohlcv_columns(df.columns))
    cols = []
    for c in df.columns:
        if c in (ts_col, label_col):
            continue
        if c.lower() in ohlcv_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_candle_type(df: pd.DataFrame, tp_points: float, sl_points: float) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")

    multiplier = o / 100000.0
    long_tp = o + tp_points * multiplier
    long_sl = o - sl_points * multiplier
    short_tp = o - tp_points * multiplier
    short_sl = o + sl_points * multiplier

    labels = np.full(len(df), "skip", dtype=object)

    # clean long / short
    labels[(h >= long_tp) & (l >= long_sl)] = "long"
    labels[(l <= short_tp) & (h <= short_sl)] = "short"

    # ambiguous cases stay skip (no lower timeframe tie-break here)
    return pd.Series(labels, index=df.index)


def preprocess_split(
    df: pd.DataFrame,
    ts_col: str,
    label_col: str,
    feature_candidates: Sequence[str],
    split: SplitDef,
) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, float], np.ndarray]:
    train_df = df[(df[ts_col] >= split.train_start) & (df[ts_col] <= split.train_end)].copy()
    test_df = df[(df[ts_col] >= split.test_start) & (df[ts_col] <= split.test_end)].copy()

    if train_df.empty or test_df.empty:
        return None

    missing_ratio = train_df[feature_candidates].isna().mean()
    keep_cols = missing_ratio[missing_ratio <= MISSING_COL_THRESHOLD].index.tolist()
    if not keep_cols:
        return None

    variances_raw = train_df[keep_cols].var(skipna=True)
    keep_cols = [c for c in keep_cols if variances_raw.get(c, 0.0) > 0]
    if not keep_cols:
        return None

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[keep_cols])
    X_test = imputer.transform(test_df[keep_cols])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_map = {"skip": 0, "long": 1, "short": 2}
    y_train = train_df[label_col].map(y_map).fillna(0).astype(int).to_numpy()

    variances = dict(zip(keep_cols, np.var(X_train, axis=0, ddof=0)))
    if len(keep_cols) > 1:
        corr = np.corrcoef(X_train_scaled, rowvar=False)
        abs_corr = np.abs(np.nan_to_num(corr, nan=0.0))
    else:
        abs_corr = np.eye(len(keep_cols))

    return X_train_scaled, X_test_scaled, y_train, keep_cols, variances, abs_corr


def build_variance_groups(
    feature_names: Sequence[str], variances: Dict[str, float], sizes: Sequence[int]
) -> List[GroupDefinition]:
    ordered = sorted(feature_names, key=lambda c: variances.get(c, 0.0), reverse=True)
    groups: List[GroupDefinition] = []
    for size in sizes:
        size = min(size, len(ordered))
        if size < MIN_GROUP_SIZE:
            continue
        feats = ordered[:size]
        group_id = f"var_top{size}"
        groups.append(GroupDefinition(method="variance", group_id=group_id, group_name=group_id, features=feats))
    return groups


def build_supervised_groups(
    method: str,
    feature_names: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    sizes: Sequence[int],
) -> List[GroupDefinition]:
    if len(feature_names) == 0:
        return []
    try:
        if method == "mi":
            scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)
        elif method == "f_classif":
            scores, _ = f_classif(X_train, y_train)
        else:
            return []
    except Exception:
        return []

    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    ranked = [feature_names[i] for i in order]
    groups: List[GroupDefinition] = []
    for size in sizes:
        size = min(size, len(ranked))
        if size < MIN_GROUP_SIZE:
            continue
        feats = ranked[:size]
        group_id = f"{method}_top{size}"
        groups.append(GroupDefinition(method=method, group_id=group_id, group_name=group_id, features=feats))
    return groups


def build_random_groups(feature_names: Sequence[str], sizes: Sequence[int], n_per_size: int) -> List[GroupDefinition]:
    rng = np.random.RandomState(RANDOM_SEED)
    groups: List[GroupDefinition] = []
    seen = set()
    for size in sizes:
        size = min(size, len(feature_names))
        if size < MIN_GROUP_SIZE:
            continue
        made = 0
        attempts = 0
        while made < n_per_size and attempts < n_per_size * 20:
            attempts += 1
            idx = rng.choice(len(feature_names), size=size, replace=False)
            idx.sort()
            feats = [feature_names[i] for i in idx]
            key = tuple(feats)
            if key in seen:
                continue
            seen.add(key)
            group_id = f"rand_s{size}_r{made+1}"
            groups.append(GroupDefinition(method="random", group_id=group_id, group_name=group_id, features=feats))
            made += 1
    return groups


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

    best = runs[select_best_run(runs)]
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


def rank_top_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["TP", "SL", "Split_ID", "Covariance_Type"]
    for _, sub in metrics_df.groupby(keys):
        candidate = sub.sort_values(by=["Test_AvgLogLik", "Train_BIC", "MultiRun_LL_Std"], ascending=[False, True, True])
        top = candidate.head(1).copy()
        top["Rank_Type"] = "Top_by_Test_AvgLogLik"
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=METRICS_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default=str(INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--tp-list", type=str, default=",".join(str(x) for x in TP_LIST))
    parser.add_argument("--sl-list", type=str, default=",".join(str(x) for x in SL_LIST))
    parser.add_argument("--k-list", type=str, default=",".join(str(x) for x in K_RANGE))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--cov-types", type=str, default=",".join(COVARIANCE_TYPES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG_DIR
    LOG_DIR = out_dir / "logs"
    setup_logging()

    tp_list = parse_float_list(args.tp_list)
    sl_list = parse_float_list(args.sl_list)
    k_list = [int(float(x)) for x in args.k_list.split(",") if x.strip()]
    seeds = [int(float(x)) for x in args.seeds.split(",") if x.strip()]
    cov_types = [x.strip() for x in args.cov_types.split(",") if x.strip()]

    df = pd.read_csv(input_csv)
    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        raise ValueError("Timestamp column not found.")

    ohlcv_cols = detect_ohlcv_columns(df.columns)
    if not all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
        raise ValueError("Input CSV must contain OHLCV columns to create labels for TP/SL sweep.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last")
    df = df[(df[ts_col] >= pd.Timestamp(DATE_START)) & (df[ts_col] <= pd.Timestamp(DATE_END))].copy()

    splits = generate_walkforward_splits(DATE_START, DATE_END, TRAIN_YEARS_LIST, TEST_YEARS, STEP_YEARS)
    if args.smoke:
        splits = splits[:1]
        tp_list = tp_list[:1]
        sl_list = sl_list[:1]
        k_list = k_list[:2]
        seeds = seeds[:1]

    all_rows: List[Dict[str, float]] = []

    for tp in tp_list:
        for sl in sl_list:
            label_col = "candle_type"
            df_l = df.copy()
            df_l[label_col] = make_candle_type(df_l, tp_points=tp, sl_points=sl)

            numeric_features = select_numeric_features(df_l, ts_col, label_col)
            if not numeric_features:
                logging.warning("No numeric indicators for TP=%s SL=%s", tp, sl)
                continue

            logging.info("TP=%s SL=%s | features=%d", tp, sl, len(numeric_features))

            for split in splits:
                pre = preprocess_split(df_l, ts_col, label_col, numeric_features, split)
                if pre is None:
                    continue
                X_train, X_test, y_train, feature_names, variances, abs_corr = pre
                if len(feature_names) < MIN_GROUP_SIZE:
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
                variance_groups = build_variance_groups(feature_names, variances, VAR_TOP_SIZES)
                mi_groups = build_supervised_groups("mi", feature_names, X_train, y_train, VAR_TOP_SIZES)
                f_groups = build_supervised_groups("f_classif", feature_names, X_train, y_train, VAR_TOP_SIZES)
                random_groups = build_random_groups(feature_names, VAR_TOP_SIZES, RANDOM_GROUPS_PER_SIZE)

                groups = corr_groups + domain_groups + variance_groups + mi_groups + f_groups + random_groups
                if not groups:
                    continue

                idx_map = {f: i for i, f in enumerate(feature_names)}
                for g in groups:
                    idx = [idx_map[f] for f in g.features if f in idx_map]
                    if len(idx) < MIN_GROUP_SIZE:
                        continue
                    X_train_g = X_train[:, idx]
                    X_test_g = X_test[:, idx]

                    for cov_type in cov_types:
                        for k in k_list:
                            metrics, ll_std, weight_std_mean, mean_shift_std_mean = evaluate_config(
                                X_train_g, X_test_g, cov_type=cov_type, k=k, seeds=seeds
                            )
                            model_id = f"bgmm_wf_{split.split_id}_tp{int(tp)}_sl{int(sl)}_{g.group_id}_{cov_type}_K{k}"
                            all_rows.append(
                                {
                                    "Model_ID": model_id,
                                    "TP": tp,
                                    "SL": sl,
                                    "Group_Method": g.method,
                                    "Group_ID": g.group_id,
                                    "Group_Name": g.group_name,
                                    "N_Features": len(idx),
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
                            )

    if not all_rows:
        logging.warning("No rows generated.")
        return 1

    metrics_df = pd.DataFrame(all_rows)
    for col in METRICS_COLUMNS:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_df = metrics_df[METRICS_COLUMNS + ["Split_ID"]]

    metrics_df[METRICS_COLUMNS].to_csv(out_dir / "metrics_all.csv", index=False)
    metrics_df[metrics_df["Covariance_Type"] == "tied"][METRICS_COLUMNS].to_csv(out_dir / "metrics_tied.csv", index=False)
    metrics_df[metrics_df["Covariance_Type"] == "full"][METRICS_COLUMNS].to_csv(out_dir / "metrics_full.csv", index=False)

    top = rank_top_configs(metrics_df)
    top[METRICS_COLUMNS].to_csv(out_dir / "top_configs.csv", index=False)

    logging.info("Outputs written to %s", out_dir)
    logging.info("rows all=%d tied=%d full=%d top=%d", len(metrics_df), (metrics_df["Covariance_Type"] == "tied").sum(), (metrics_df["Covariance_Type"] == "full").sum(), len(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
