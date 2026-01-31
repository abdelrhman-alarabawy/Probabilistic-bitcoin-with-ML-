#!/usr/bin/env python
"""
Walk-forward GMM evaluation for BTC daily indicators with group ablations and stability checks.

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from gmm_grouping import GroupDefinition, build_corr_groups, build_domain_groups, feature_list_short
from gmm_metrics import compute_stability, fit_gmm_and_score, select_best_run


INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
OUTPUT_ROOT = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_gmm_groups_walkforward\results")

DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

TRAIN_YEARS_LIST = [2, 3, 4]
TEST_YEARS = 1
STEP_YEARS = 1

TRAIN_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3
WF_TAG = "wf_6m_train_3m_test_step_3m"
WF_SHORT_TAG = "wf_6m3m"

WINDOW_MODE = "years"  # "years" or "months"
MAX_FOLDS: Optional[int] = None

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
    train_years: Optional[int] = None
    train_months: Optional[int] = None


@dataclass
class RunConfig:
    input_csv: str
    output_root: Path
    date_start: str
    date_end: str
    window_mode: str
    train_years_list: Sequence[int]
    test_years: int
    step_years: int
    train_months: int
    test_months: int
    step_months: int
    wf_tag: str
    wf_short_tag: str
    covariance_types: Sequence[str]
    k_range: Sequence[int]
    seeds: Sequence[int]
    max_folds: Optional[int]


def build_default_config() -> RunConfig:
    return RunConfig(
        input_csv=INPUT_CSV,
        output_root=OUTPUT_ROOT,
        date_start=DATE_START,
        date_end=DATE_END,
        window_mode=WINDOW_MODE,
        train_years_list=TRAIN_YEARS_LIST,
        test_years=TEST_YEARS,
        step_years=STEP_YEARS,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        wf_tag=WF_TAG,
        wf_short_tag=WF_SHORT_TAG,
        covariance_types=COVARIANCE_TYPES,
        k_range=K_RANGE,
        seeds=SEEDS,
        max_folds=MAX_FOLDS,
    )


def build_months_config() -> RunConfig:
    return RunConfig(
        input_csv=INPUT_CSV,
        output_root=OUTPUT_ROOT,
        date_start=DATE_START,
        date_end=DATE_END,
        window_mode="months",
        train_years_list=TRAIN_YEARS_LIST,
        test_years=TEST_YEARS,
        step_years=STEP_YEARS,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        wf_tag=WF_TAG,
        wf_short_tag=WF_SHORT_TAG,
        covariance_types=COVARIANCE_TYPES,
        k_range=K_RANGE,
        seeds=SEEDS,
        max_folds=MAX_FOLDS,
    )


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_gmm_groups_walkforward.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
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


def resolve_output_dir(output_root: Path, window_mode: str, wf_tag: str) -> Path:
    if window_mode == "months":
        base_dir = output_root / wf_tag
    else:
        base_dir = output_root

    if base_dir.exists() and any(base_dir.iterdir()):
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        base_dir = base_dir.parent / f"{base_dir.name}_{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_model_id(
    split: SplitDef,
    group_method: str,
    group_id: str,
    cov_type: str,
    k: int,
    window_mode: str,
    wf_short_tag: str,
) -> str:
    if window_mode == "months":
        return (
            f"gmm_{wf_short_tag}_{group_method}_{group_id}_{cov_type}_K{k}_"
            f"{split.train_start:%Y%m%d}_{split.train_end:%Y%m%d}_"
            f"{split.test_start:%Y%m%d}_{split.test_end:%Y%m%d}"
        )
    return f"gmm_wf_{split.split_id}_{group_id}_{cov_type}_K{k}"


def normalize_timestamp(value: str | pd.Timestamp, tz: Optional[object]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo else ts
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def generate_walkforward_splits_years(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
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


def generate_walkforward_splits_months(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
    wf_short_tag: str,
) -> List[SplitDef]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    splits: List[SplitDef] = []
    split_start = start
    while True:
        train_start = split_start
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end:
            break
        split_id = (
            f"{wf_short_tag}_"
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
                train_months=train_months,
            )
        )
        split_start = split_start + pd.DateOffset(months=step_months)
    splits.sort(key=lambda s: s.train_start)
    return splits


def build_metrics_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    group_cols = [
        "Group_Method",
        "Group_ID",
        "Group_Name",
        "N_Features",
        "Feature_List_Short",
        "Covariance_Type",
        "K",
        "PCA_Used",
        "PCA_Components",
    ]
    metric_cols = [
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
    ]
    agg_map: Dict[str, object] = {"Split_ID": pd.Series.nunique}
    for col in metric_cols:
        agg_map[col] = ["mean", "std"]

    summary = metrics_df.groupby(group_cols).agg(agg_map)
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary = summary.rename(columns={"Split_ID_nunique": "Folds"})
    return summary


def log_summary(metrics_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if metrics_df.empty or summary_df.empty:
        logging.info("No summary generated (empty metrics).")
        return

    def log_rows(title: str, frame: pd.DataFrame, score_col: str, ascending: bool) -> None:
        logging.info(title)
        top = frame.sort_values(score_col, ascending=ascending).head(5)
        for _, row in top.iterrows():
            logging.info(
                "  %s | group=%s(%s) | cov=%s | K=%s | %s=%.6f | LL_std=%.6f | ent=%.4f | folds=%s",
                row.get("Group_ID", ""),
                row.get("Group_Method", ""),
                row.get("Group_Name", ""),
                row.get("Covariance_Type", ""),
                int(row.get("K", 0)),
                score_col,
                float(row.get(score_col, float("nan"))),
                float(row.get("MultiRun_LL_Std_mean", float("nan"))),
                float(row.get("Test_RespEntropy_mean", float("nan"))),
                int(row.get("Folds", 0)),
            )

    log_rows("Summary | Best by Test_AvgLogLik (mean across folds)", summary_df, "Test_AvgLogLik_mean", False)
    log_rows("Summary | Best by Test_BIC (mean across folds)", summary_df, "Test_BIC_mean", True)
    log_rows("Summary | Best by Test_AIC (mean across folds)", summary_df, "Test_AIC_mean", True)

    ll_std_col = "MultiRun_LL_Std_mean"
    if ll_std_col in summary_df.columns and not summary_df[ll_std_col].isna().all():
        thresh = summary_df[ll_std_col].quantile(0.25)
        filtered = summary_df[summary_df[ll_std_col] <= thresh].copy()
        filtered["ExtremeEntropy"] = filtered.apply(
            lambda r: is_extreme_entropy(float(r.get("Test_RespEntropy_mean", float("nan"))), int(r.get("K", 0))),
            axis=1,
        )
        filtered = filtered[~filtered["ExtremeEntropy"]]
        if not filtered.empty:
            log_rows(
                "Summary | Low MultiRun_LL_Std + non-extreme entropy (Top Test_AvgLogLik)",
                filtered,
                "Test_AvgLogLik_mean",
                False,
            )
        else:
            logging.info("Summary | No configs met low-LL-std + non-extreme entropy filter.")


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
    min_train_samples: int,
    min_test_samples: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float], np.ndarray]]:
    train_df = df[(df[ts_col] >= split.train_start) & (df[ts_col] <= split.train_end)].copy()
    test_df = df[(df[ts_col] >= split.test_start) & (df[ts_col] <= split.test_end)].copy()

    if train_df.empty or test_df.empty:
        logging.warning("Split %s has empty train or test; skipping.", split.split_id)
        return None

    if len(train_df) < min_train_samples or len(test_df) < min_test_samples:
        logging.warning(
            "Split %s has insufficient samples (train=%d, test=%d); skipping.",
            split.split_id,
            len(train_df),
            len(test_df),
        )
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
) -> Tuple[Dict[str, float], float, float, float]:
    runs = []
    for seed in seeds:
        run = fit_gmm_and_score(
            X_train,
            X_test,
            covariance_type=cov_type,
            k=k,
            seed=seed,
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


def main(config: Optional[RunConfig] = None) -> int:
    cfg = config or build_default_config()
    output_dir = resolve_output_dir(cfg.output_root, cfg.window_mode, cfg.wf_tag)
    setup_logging(output_dir / "logs")

    logging.info("Loading data from %s", cfg.input_csv)
    logging.info("Window mode: %s", cfg.window_mode)
    if cfg.window_mode == "months":
        logging.info(
            "Months config: train=%dm, test=%dm, step=%dm | tag=%s",
            cfg.train_months,
            cfg.test_months,
            cfg.step_months,
            cfg.wf_tag,
        )
    else:
        logging.info(
            "Years config: train=%s, test=%dy, step=%dy",
            cfg.train_years_list,
            cfg.test_years,
            cfg.step_years,
        )
    df = pd.read_csv(cfg.input_csv)

    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        raise ValueError(f"Could not detect timestamp column among {TIMESTAMP_CANDIDATES}")

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)
    df = df.drop_duplicates(subset=[ts_col], keep="last")

    tz = df[ts_col].dt.tz
    start = normalize_timestamp(cfg.date_start, tz)
    end = normalize_timestamp(cfg.date_end, tz)
    df = df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()
    logging.info("Data rows after date filter: %d", len(df))

    numeric_features = select_numeric_features(df, ts_col)
    logging.info("Numeric indicator features (excluding OHLCV): %d", len(numeric_features))
    if not numeric_features:
        raise RuntimeError("No numeric indicator columns found after excluding OHLCV.")

    if cfg.window_mode == "months":
        splits = generate_walkforward_splits_months(
            start,
            end,
            cfg.train_months,
            cfg.test_months,
            cfg.step_months,
            cfg.wf_short_tag,
        )
    else:
        splits = generate_walkforward_splits_years(
            start,
            end,
            cfg.train_years_list,
            cfg.test_years,
            cfg.step_years,
        )
    logging.info("Generated %d walk-forward splits.", len(splits))

    all_rows: List[Dict[str, float]] = []

    min_train_samples = max(cfg.k_range) + 1
    min_test_samples = max(cfg.k_range)

    for split_idx, split in enumerate(splits, start=1):
        if cfg.max_folds and split_idx > cfg.max_folds:
            logging.info("Reached max_folds=%d; stopping.", cfg.max_folds)
            break
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

        pre = preprocess_split(df, ts_col, numeric_features, split, min_train_samples, min_test_samples)
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

            for cov_type in cfg.covariance_types:
                for k in cfg.k_range:
                    metrics, ll_std, weight_std_mean, mean_shift_std_mean = evaluate_config(
                        X_train_g, X_test_g, cov_type, k, cfg.seeds
                    )
                    model_id = build_model_id(
                        split,
                        g.method,
                        g.group_id,
                        cov_type,
                        k,
                        cfg.window_mode,
                        cfg.wf_short_tag,
                    )
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

            for cov_type in cfg.covariance_types:
                for k in cfg.k_range:
                    metrics, ll_std, weight_std_mean, mean_shift_std_mean = evaluate_config(
                        X_train_g, X_test_g, cov_type, k, cfg.seeds
                    )
                    model_id = build_model_id(
                        split,
                        g.method,
                        g.group_id,
                        cov_type,
                        k,
                        cfg.window_mode,
                        cfg.wf_short_tag,
                    )
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
    metrics_all_path = output_dir / "metrics_all.csv"
    metrics_out.to_csv(metrics_all_path, index=False)

    metrics_tied = metrics_df[metrics_df["Covariance_Type"] == "tied"].copy()
    metrics_full = metrics_df[metrics_df["Covariance_Type"] == "full"].copy()

    metrics_tied[METRICS_COLUMNS].to_csv(output_dir / "metrics_tied.csv", index=False)
    metrics_full[METRICS_COLUMNS].to_csv(output_dir / "metrics_full.csv", index=False)

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
    top_configs[METRICS_COLUMNS].to_csv(output_dir / "top_configs.csv", index=False)

    summary_df = build_metrics_summary(metrics_df)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
        log_summary(metrics_df, summary_df)

    logging.info("Outputs written to %s", output_dir)
    logging.info("metrics_all.csv rows: %d", len(metrics_df))
    logging.info("metrics_tied.csv rows: %d", len(metrics_tied))
    logging.info("metrics_full.csv rows: %d", len(metrics_full))
    logging.info("top_configs.csv rows: %d", len(top_configs))
    if not summary_df.empty:
        logging.info("metrics_summary.csv rows: %d", len(summary_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
