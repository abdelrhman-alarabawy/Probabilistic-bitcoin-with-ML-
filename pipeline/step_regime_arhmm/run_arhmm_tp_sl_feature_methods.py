#!/usr/bin/env python
"""
Automatic AR-HMM walk-forward sweep over TP/SL labeling settings and multiple
feature-selection methods (aligned with BGMM/GMM grouping logic).
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
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
GMM_DIR = CURRENT_DIR.parent / "step_gmm_groups_walkforward"
if str(GMM_DIR) not in sys.path:
    sys.path.append(str(GMM_DIR))

from gmm_grouping import GroupDefinition, build_corr_groups, build_domain_groups, feature_list_short
from src.arhmm import ARHMM
from src.eval import entropy_stats, num_params_arhmm, aic_bic


INPUT_CSV = Path(r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv")
OUTPUT_DIR = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_arhmm\results\tp_sl_methods")
LOG_DIR = OUTPUT_DIR / "logs"

DATE_START = "2020-01-01"
DATE_END = "2025-12-31"

TRAIN_YEARS_LIST = [2, 3, 4]
TEST_YEARS = 1
STEP_YEARS = 1

K_RANGE = [2, 3, 4]
SEEDS = [89250, 773956, 654571]
EPS_LIST = [1e-6, 1e-5, 1e-4]

TP_LIST = [1000.0, 1500.0, 2000.0, 2500.0]
SL_LIST = [500.0, 1000.0, 1500.0]

MISSING_COL_THRESHOLD = 0.20
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 10
TOP_SIZES = [5, 8, 10]
RANDOM_GROUPS_PER_SIZE = 3
RANDOM_SEED = 42

MAX_ITER = 200
TOL = 1e-4

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
    "Eps",
    "Train_Start",
    "Train_End",
    "Test_Start",
    "Test_End",
    "Train_AvgLogLik",
    "Test_AvgLogLik",
    "Train_AIC",
    "Train_BIC",
    "Train_RespEntropy",
    "Test_RespEntropy",
    "MultiRun_LL_Std",
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


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_arhmm_tp_sl_feature_methods.log"
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
    out = []
    for c in df.columns:
        if c in (ts_col, label_col):
            continue
        if c.lower() in ohlcv_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


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
    labels[(h >= long_tp) & (l >= long_sl)] = "long"
    labels[(l <= short_tp) & (h <= short_sl)] = "short"
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


def build_variance_groups(feature_names: Sequence[str], variances: Dict[str, float], sizes: Sequence[int]) -> List[GroupDefinition]:
    ordered = sorted(feature_names, key=lambda c: variances.get(c, 0.0), reverse=True)
    groups: List[GroupDefinition] = []
    for size in sizes:
        size = min(size, len(ordered))
        if size < MIN_GROUP_SIZE:
            continue
        gid = f"var_top{size}"
        groups.append(GroupDefinition(method="variance", group_id=gid, group_name=gid, features=ordered[:size]))
    return groups


def build_supervised_groups(method: str, feature_names: Sequence[str], X_train: np.ndarray, y_train: np.ndarray, sizes: Sequence[int]) -> List[GroupDefinition]:
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
        gid = f"{method}_top{size}"
        groups.append(GroupDefinition(method=method, group_id=gid, group_name=gid, features=ranked[:size]))
    return groups


def build_random_groups(feature_names: Sequence[str], sizes: Sequence[int], n_per_size: int) -> List[GroupDefinition]:
    rng = np.random.RandomState(RANDOM_SEED)
    seen = set()
    groups: List[GroupDefinition] = []
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
            gid = f"rand_s{size}_r{made+1}"
            groups.append(GroupDefinition(method="random", group_id=gid, group_name=gid, features=feats))
            made += 1
    return groups


def evaluate_arhmm_multiseed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    eps: float,
    seeds: Sequence[int],
) -> Tuple[Dict[str, float], float]:
    runs = []
    d = X_train.shape[1]
    n_params = num_params_arhmm(k, d)

    for seed in seeds:
        model = ARHMM(
            n_states=k,
            n_features=d,
            eps=eps,
            max_iter=MAX_ITER,
            tol=TOL,
            seed=seed,
        )
        fit = model.fit(X_train)
        train_ll = model.loglikelihood(X_train)
        test_ll = model.loglikelihood(X_test)
        gamma_train = model.predict_proba(X_train)
        gamma_test = model.predict_proba(X_test)
        ent_train = entropy_stats(gamma_train)
        ent_test = entropy_stats(gamma_test)
        train_aic, train_bic = aic_bic(train_ll, n_params, X_train.shape[0] - 1)

        runs.append(
            {
                "seed": seed,
                "train_ll": float(train_ll),
                "test_ll": float(test_ll),
                "train_avg_ll": float(train_ll / max(1, X_train.shape[0] - 1)),
                "test_avg_ll": float(test_ll / max(1, X_test.shape[0] - 1)),
                "train_aic": float(train_aic),
                "train_bic": float(train_bic),
                "train_entropy": float(ent_train["mean"]),
                "test_entropy": float(ent_test["mean"]),
                "converged": bool(fit.converged),
            }
        )

    best_idx = int(np.argmax([r["test_avg_ll"] for r in runs]))
    best = runs[best_idx]
    ll_std = float(np.std([r["test_avg_ll"] for r in runs]))

    metrics = {
        "Train_AvgLogLik": best["train_avg_ll"],
        "Test_AvgLogLik": best["test_avg_ll"],
        "Train_AIC": best["train_aic"],
        "Train_BIC": best["train_bic"],
        "Train_RespEntropy": best["train_entropy"],
        "Test_RespEntropy": best["test_entropy"],
    }
    return metrics, ll_std


def rank_top_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sub in metrics_df.groupby(["TP", "SL", "Split_ID", "Covariance_Type"]):
        candidate = sub.sort_values(
            by=["Test_AvgLogLik", "Train_BIC", "MultiRun_LL_Std"],
            ascending=[False, True, True],
        )
        top = candidate.head(1).copy()
        top["Rank_Type"] = "Top_by_Test_AvgLogLik"
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=METRICS_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default=str(INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--tp-list", type=str, default=",".join(str(x) for x in TP_LIST))
    parser.add_argument("--sl-list", type=str, default=",".join(str(x) for x in SL_LIST))
    parser.add_argument("--k-list", type=str, default=",".join(str(x) for x in K_RANGE))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))
    parser.add_argument("--eps-list", type=str, default=",".join(str(x) for x in EPS_LIST))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "logs")

    tp_list = parse_float_list(args.tp_list)
    sl_list = parse_float_list(args.sl_list)
    k_list = parse_int_list(args.k_list)
    seeds = parse_int_list(args.seeds)
    eps_list = parse_float_list(args.eps_list)

    df = pd.read_csv(input_csv)
    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        raise ValueError("Timestamp column not found.")

    required_ohlcv = ["open", "high", "low", "close", "volume"]
    missing_ohlcv = [c for c in required_ohlcv if c not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"Missing OHLCV columns for TP/SL labels: {missing_ohlcv}")

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
        eps_list = eps_list[:1]

    all_rows: List[Dict[str, float]] = []

    for tp in tp_list:
        for sl in sl_list:
            df_l = df.copy()
            label_col = "candle_type"
            df_l[label_col] = make_candle_type(df_l, tp_points=tp, sl_points=sl)

            numeric_features = select_numeric_features(df_l, ts_col, label_col)
            if not numeric_features:
                continue

            logging.info("TP=%s SL=%s | candidate features=%d", tp, sl, len(numeric_features))

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
                variance_groups = build_variance_groups(feature_names, variances, TOP_SIZES)
                mi_groups = build_supervised_groups("mi", feature_names, X_train, y_train, TOP_SIZES)
                f_groups = build_supervised_groups("f_classif", feature_names, X_train, y_train, TOP_SIZES)
                random_groups = build_random_groups(feature_names, TOP_SIZES, RANDOM_GROUPS_PER_SIZE)

                groups = corr_groups + domain_groups + variance_groups + mi_groups + f_groups + random_groups
                idx_map = {f: i for i, f in enumerate(feature_names)}

                for g in groups:
                    idx = [idx_map[f] for f in g.features if f in idx_map]
                    if len(idx) < MIN_GROUP_SIZE:
                        continue
                    X_train_g = X_train[:, idx]
                    X_test_g = X_test[:, idx]

                    for k in k_list:
                        for eps in eps_list:
                            try:
                                metrics, ll_std = evaluate_arhmm_multiseed(
                                    X_train_g,
                                    X_test_g,
                                    k=k,
                                    eps=eps,
                                    seeds=seeds,
                                )
                            except Exception as exc:
                                logging.warning(
                                    "Failed TP=%s SL=%s split=%s group=%s K=%s eps=%s: %s",
                                    tp,
                                    sl,
                                    split.split_id,
                                    g.group_id,
                                    k,
                                    eps,
                                    exc,
                                )
                                continue

                            model_id = f"arhmm_wf_{split.split_id}_tp{int(tp)}_sl{int(sl)}_{g.group_id}_full_K{k}_eps{eps}"
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
                                    "Covariance_Type": "full",
                                    "K": k,
                                    "Eps": eps,
                                    "Train_Start": split.train_start.strftime("%Y-%m-%d"),
                                    "Train_End": split.train_end.strftime("%Y-%m-%d"),
                                    "Test_Start": split.test_start.strftime("%Y-%m-%d"),
                                    "Test_End": split.test_end.strftime("%Y-%m-%d"),
                                    "Train_AvgLogLik": metrics["Train_AvgLogLik"],
                                    "Test_AvgLogLik": metrics["Test_AvgLogLik"],
                                    "Train_AIC": metrics["Train_AIC"],
                                    "Train_BIC": metrics["Train_BIC"],
                                    "Train_RespEntropy": metrics["Train_RespEntropy"],
                                    "Test_RespEntropy": metrics["Test_RespEntropy"],
                                    "MultiRun_LL_Std": ll_std,
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
    metrics_df[metrics_df["Covariance_Type"] == "full"][METRICS_COLUMNS].to_csv(out_dir / "metrics_full.csv", index=False)
    metrics_df[metrics_df["Covariance_Type"] == "full"][METRICS_COLUMNS].to_csv(out_dir / "metrics_tied.csv", index=False)

    top = rank_top_configs(metrics_df)
    top[METRICS_COLUMNS].to_csv(out_dir / "top_configs.csv", index=False)

    logging.info("Outputs written to %s", out_dir)
    logging.info("rows all=%d full=%d top=%d", len(metrics_df), len(metrics_df), len(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
