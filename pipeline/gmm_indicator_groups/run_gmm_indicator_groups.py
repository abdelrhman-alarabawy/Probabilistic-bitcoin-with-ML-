#!/usr/bin/env python
"""
GMM Indicator Group Discovery and Evaluation (Daily Bitcoin)

Notes on leakage and shifting:
- Indicators in the input CSV are already shifted and safe to use as-is.
- Do NOT shift any features here; this script assumes the indicators are leakage-free.

Why group synergy matters:
- Many indicators measure similar dynamics (trend, momentum, volatility). A group of
  related indicators can capture richer regime structure than any single indicator.

How to interpret key metrics:
- Avg log-likelihood: higher means the model fits the data distribution better.
- BIC/AIC: penalized likelihood for model selection; lower is better (use TRAIN BIC/AIC).
- Responsibility entropy: higher means more ambiguous cluster assignments.
  Low entropy implies confident regime assignment.

Selecting the final indicator set:
- Prefer groups (or group combinations) with strong TEST Avg log-likelihood and stable
  multi-run variance. Use TRAIN BIC as a sanity check for overfitting.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional
    raise RuntimeError("matplotlib is required for plots") from exc


# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
OUTPUT_DIR = r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\gmm_indicator_groups"

TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

K_LIST = [2, 3, 4, 5, 6, 7, 8]
COV_TYPES_TO_TRY = ["full", "tied", "diag"]

MISSING_COL_THRESHOLD = 0.30
GROUP_MIN_CORR = 0.80
GROUP_MAX_SIZE = 25
GROUP_MIN_SIZE = 2

PCA_GROUP_TOP_N = 5
PCA_GROUP_TOP_FEATURES = 10
PCA_REDUCE_THRESHOLD = 25  # if a group has > this many features, reduce to 95% variance

GREEDY_MAX_GROUPS = 5
GREEDY_CANDIDATE_LIMIT = None  # set to int to limit groups considered in greedy stage

N_RUNS = 10
N_INIT = 5
RANDOM_STATE = 42
EPS = 1e-12

TIMESTAMP_CANDIDATES = ["timestamp", "date", "time", "ts", "ts_utc", "datetime"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]

LOG_PATH = Path(OUTPUT_DIR) / "logs" / "run_gmm_indicator_groups.log"


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: List[str]
    ts_col: str


@dataclass
class GroupCandidate:
    method: str
    group_id: str
    group_name: str
    features: List[str]


# -----------------------------
# Utility functions
# -----------------------------

def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(),
        ],
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
    found = [col_map[c] for c in OHLCV_CANDIDATES if c in col_map]
    return found


def filter_and_split(df: pd.DataFrame) -> SplitData:
    ts_col = detect_timestamp_column(df.columns)
    if not ts_col:
        raise ValueError(
            f"Could not detect timestamp column among {TIMESTAMP_CANDIDATES}"
        )

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    start_all = pd.Timestamp(TRAIN_START)
    end_all = pd.Timestamp(TEST_END)
    df = df[(df[ts_col] >= start_all) & (df[ts_col] <= end_all)].copy()

    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train_df = df[(df[ts_col] >= train_start) & (df[ts_col] <= train_end)].copy()
    test_df = df[(df[ts_col] >= test_start) & (df[ts_col] <= test_end)].copy()

    return SplitData(train_df=train_df, test_df=test_df, feature_cols=list(df.columns), ts_col=ts_col)


def get_numeric_indicator_columns(df: pd.DataFrame, ts_col: str) -> List[str]:
    ohlcv_cols = set(c.lower() for c in detect_ohlcv_columns(df.columns))
    cols = []
    for c in df.columns:
        if c == ts_col:
            continue
        if c.lower() in ohlcv_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def drop_high_missing(train_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    missing_ratio = train_df[feature_cols].isna().mean()
    keep_cols = missing_ratio[missing_ratio <= MISSING_COL_THRESHOLD].index.tolist()
    return keep_cols


def compute_avg_loglik(gmm: GaussianMixture, X: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    return float(np.mean(gmm.score_samples(X)))


def compute_labels_from_responsibilities(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    resp = gmm.predict_proba(X)
    return np.argmax(resp, axis=1)


def compute_entropy(gmm: GaussianMixture, X: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    resp = gmm.predict_proba(X)
    entropy = -np.sum(resp * np.log(resp + EPS), axis=1)
    return float(np.mean(entropy))


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")
    return float(davies_bouldin_score(X, labels))


def bic_aic_safe(gmm: GaussianMixture, X: np.ndarray) -> Tuple[float, float]:
    if X.size == 0:
        return float("nan"), float("nan")
    try:
        return float(gmm.bic(X)), float(gmm.aic(X))
    except Exception:
        return float("nan"), float("nan")


def hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    """Hungarian algorithm for square cost matrix (minimization). Returns assignment array
    where assignment[i] = j (row i assigned to column j).
    """
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def align_components(ref_means: np.ndarray, means: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align components to reference means via Hungarian assignment."""
    cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
    assignment = hungarian_min_cost(cost)
    return weights[assignment], means[assignment]


def ensure_dirs() -> Dict[str, Path]:
    base = Path(OUTPUT_DIR)
    results_dir = base / "results"
    plots_dir = base / "plots"
    logs_dir = base / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "results": results_dir,
        "plots": plots_dir,
        "logs": logs_dir,
    }


def correlation_groups(X_train: np.ndarray, feature_names: List[str]) -> List[GroupCandidate]:
    if X_train.shape[1] < 2:
        return []

    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    distance = 1.0 - abs_corr
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")

    best_thr = None
    best_score = -1
    thresholds = np.arange(0.05, 0.90, 0.05)

    for thr in thresholds:
        clusters = fcluster(Z, t=thr, criterion="distance")
        groups = {}
        for idx, cid in enumerate(clusters):
            groups.setdefault(cid, []).append(idx)

        good_groups = []
        for g in groups.values():
            if len(g) < GROUP_MIN_SIZE or len(g) > GROUP_MAX_SIZE:
                continue
            sub = abs_corr[np.ix_(g, g)]
            if sub.size <= 1:
                continue
            avg_corr = (np.sum(sub) - len(g)) / (len(g) * (len(g) - 1))
            if avg_corr >= GROUP_MIN_CORR:
                good_groups.append(avg_corr)

        if not good_groups:
            continue

        score = len(good_groups) + float(np.mean(good_groups))
        if score > best_score:
            best_score = score
            best_thr = thr

    if best_thr is None:
        best_thr = 0.30

    clusters = fcluster(Z, t=best_thr, criterion="distance")
    groups = {}
    for idx, cid in enumerate(clusters):
        groups.setdefault(cid, []).append(idx)

    candidates = []
    for cid, indices in groups.items():
        if len(indices) < GROUP_MIN_SIZE:
            continue
        names = [feature_names[i] for i in indices]
        group_id = f"corr_{cid}"
        candidates.append(GroupCandidate(method="corr", group_id=group_id, group_name=group_id, features=names))

    return candidates


def name_based_groups(feature_names: List[str]) -> List[GroupCandidate]:
    patterns = [
        ("RSI", ["RSI"]),
        ("MACD", ["MACD"]),
        ("STOCH", ["STOCH", "STO"]),
        ("ATR", ["ATR"]),
        ("ADX", ["ADX", "ADXR"]),
        ("BB", ["BB", "BOLL", "BANDS"]),
        ("EMA", ["EMA"]),
        ("SMA", ["SMA"]),
        ("VWAP", ["VWAP"]),
        ("VOL", ["VOL", "VOLUME"]),
        ("OBV", ["OBV"]),
        ("MFI", ["MFI"]),
        ("CCI", ["CCI"]),
        ("ROC", ["ROC"]),
        ("MOM", ["MOM", "MOMENTUM"]),
        ("WILLIAMS", ["WILL", "WILLIAMS"]),
        ("DONCHIAN", ["DONCHIAN"]),
        ("KELTNER", ["KELTNER"]),
        ("ICHIMOKU", ["ICHIMOKU", "TENKAN", "KIJUN", "SENKOU"]),
    ]

    assigned = set()
    candidates = []

    for group_name, keys in patterns:
        group_feats = []
        for f in feature_names:
            upper = f.upper()
            if f in assigned:
                continue
            if any(k in upper for k in keys):
                group_feats.append(f)
                assigned.add(f)
        if len(group_feats) >= GROUP_MIN_SIZE:
            group_id = f"name_{group_name.lower()}"
            candidates.append(GroupCandidate(method="name", group_id=group_id, group_name=group_name, features=group_feats))

    return candidates


def pca_groups(X_train: np.ndarray, feature_names: List[str]) -> List[GroupCandidate]:
    if X_train.shape[1] < 2:
        return []
    pca = PCA(n_components=min(PCA_GROUP_TOP_N, X_train.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X_train)
    components = pca.components_

    candidates = []
    for i, comp in enumerate(components):
        idx = np.argsort(np.abs(comp))[::-1][:PCA_GROUP_TOP_FEATURES]
        feats = [feature_names[j] for j in idx]
        if len(feats) >= GROUP_MIN_SIZE:
            group_id = f"pca_pc{i+1}"
            candidates.append(GroupCandidate(method="pca", group_id=group_id, group_name=f"PC{i+1}", features=feats))

    return candidates


def save_feature_groups(candidates: List[GroupCandidate], results_dir: Path) -> None:
    groups_dict = {}
    rows = []
    for g in candidates:
        groups_dict[g.group_id] = {
            "method": g.method,
            "group_name": g.group_name,
            "features": g.features,
        }
        rows.append({
            "group_id": g.group_id,
            "method": g.method,
            "group_name": g.group_name,
            "n_features": len(g.features),
            "feature_list": ",".join(g.features),
        })

    json_path = results_dir / "feature_groups.json"
    csv_path = results_dir / "feature_groups.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(groups_dict, f, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def feature_list_short(features: List[str], max_len: int = 120) -> str:
    s = "|".join(features)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def prepare_group_data(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: List[str],
    group_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    idx_map = {name: i for i, name in enumerate(feature_names)}
    indices = [idx_map[f] for f in group_features if f in idx_map]
    X_train_g = X_train_scaled[:, indices]
    X_test_g = X_test_scaled[:, indices] if X_test_scaled.size else X_test_scaled

    pca_meta = {"pca_used": 0, "pca_components": 0}
    if X_train_g.shape[1] > PCA_REDUCE_THRESHOLD:
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train_g = pca.fit_transform(X_train_g)
        X_test_g = pca.transform(X_test_g) if X_test_g.size else X_test_g
        pca_meta = {"pca_used": 1, "pca_components": X_train_g.shape[1]}

    return X_train_g, X_test_g, pca_meta


def evaluate_gmm_for_group(
    X_train_g: np.ndarray,
    X_test_g: np.ndarray,
    cov_type: str,
    k: int,
) -> Dict[str, float]:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    gmm.fit(X_train_g)

    train_avg_ll = compute_avg_loglik(gmm, X_train_g)
    test_avg_ll = compute_avg_loglik(gmm, X_test_g)

    train_bic, train_aic = bic_aic_safe(gmm, X_train_g)
    test_bic, test_aic = bic_aic_safe(gmm, X_test_g)

    train_labels = compute_labels_from_responsibilities(gmm, X_train_g)
    test_labels = compute_labels_from_responsibilities(gmm, X_test_g) if X_test_g.size else np.array([])

    train_sil = safe_silhouette(X_train_g, train_labels) if k > 1 else float("nan")
    train_db = safe_davies_bouldin(X_train_g, train_labels) if k > 1 else float("nan")
    test_sil = safe_silhouette(X_test_g, test_labels) if k > 1 else float("nan")
    test_db = safe_davies_bouldin(X_test_g, test_labels) if k > 1 else float("nan")

    train_entropy = compute_entropy(gmm, X_train_g)
    test_entropy = compute_entropy(gmm, X_test_g)

    return {
        "gmm": gmm,
        "Train_AvgLogLik": train_avg_ll,
        "Test_AvgLogLik": test_avg_ll,
        "Train_BIC": train_bic,
        "Train_AIC": train_aic,
        "Test_BIC": test_bic,
        "Test_AIC": test_aic,
        "Train_Silhouette": train_sil,
        "Train_DaviesBouldin": train_db,
        "Test_Silhouette": test_sil,
        "Test_DaviesBouldin": test_db,
        "Train_RespEntropy": train_entropy,
        "Test_RespEntropy": test_entropy,
    }


def multi_run_stability(
    X_train_g: np.ndarray,
    cov_type: str,
    k: int,
) -> Tuple[float, float, float]:
    run_ll = []
    run_weights = []
    run_means = []

    for i in range(N_RUNS):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            random_state=RANDOM_STATE + i + 1,
            n_init=N_INIT,
        )
        gmm.fit(X_train_g)
        run_ll.append(compute_avg_loglik(gmm, X_train_g))
        run_weights.append(gmm.weights_.copy())
        run_means.append(gmm.means_.copy())

    ref_means = run_means[0]
    aligned_weights = []
    aligned_means = []

    for weights, means in zip(run_weights, run_means):
        w_aligned, m_aligned = align_components(ref_means, means, weights)
        aligned_weights.append(w_aligned)
        aligned_means.append(m_aligned)

    aligned_weights = np.array(aligned_weights)
    aligned_means = np.array(aligned_means)

    ll_std = float(np.std(run_ll, ddof=1)) if len(run_ll) > 1 else 0.0
    weight_std_per_comp = np.std(aligned_weights, axis=0, ddof=1) if len(run_ll) > 1 else np.zeros(k)
    weight_std_mean = float(np.mean(weight_std_per_comp))

    mean_shift_stds = []
    for comp in range(k):
        dists = np.linalg.norm(aligned_means[:, comp, :] - ref_means[comp], axis=1)
        mean_shift_stds.append(np.std(dists, ddof=1) if len(dists) > 1 else 0.0)
    mean_shift_std_mean = float(np.mean(mean_shift_stds))

    return ll_std, weight_std_mean, mean_shift_std_mean


def select_best_row(rows: List[Dict[str, float]]) -> int:
    test_lls = np.array([r["Test_AvgLogLik"] for r in rows], dtype=float)
    if np.all(np.isnan(test_lls)):
        train_bics = np.array([r["Train_BIC"] for r in rows], dtype=float)
        return int(np.nanargmin(train_bics))
    return int(np.nanargmax(test_lls))


def plot_top_groups(metrics_df: pd.DataFrame, plots_dir: Path) -> None:
    top = metrics_df.sort_values("Test_AvgLogLik", ascending=False).head(10)
    if top.empty:
        return

    plt.figure(figsize=(10, 6))
    for (group_id, cov_type), sub in top.groupby(["Group_ID", "Covariance_Type"]):
        sub = sub.sort_values("K")
        label = f"{group_id}-{cov_type}"
        plt.plot(sub["K"], sub["Test_AvgLogLik"], marker="o", label=label)

    plt.title("Top Groups: Test AvgLogLik vs K")
    plt.xlabel("K")
    plt.ylabel("Test AvgLogLik")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "top_groups_test_loglik.png", dpi=150)
    plt.close()


def plot_greedy_curve(greedy_df: pd.DataFrame, plots_dir: Path) -> None:
    if greedy_df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(greedy_df["step_idx"], greedy_df["Test_AvgLogLik"], marker="o")
    plt.title("Greedy Group Combination: Test AvgLogLik")
    plt.xlabel("Step")
    plt.ylabel("Test AvgLogLik")
    plt.tight_layout()
    plt.savefig(plots_dir / "greedy_test_loglik_curve.png", dpi=150)
    plt.close()


def find_ohlcv_baseline(results_dir: Path) -> Optional[Dict[str, float]]:
    baseline_dir = Path(r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\gmm_ohlcv_eval\results")
    if not baseline_dir.exists():
        return None

    candidates = sorted(baseline_dir.glob("metrics_summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    try:
        df = pd.read_csv(candidates[0])
    except Exception:
        return None

    if "Test_AvgLogLik" not in df.columns:
        return None

    best_idx = df["Test_AvgLogLik"].idxmax()
    row = df.loc[best_idx]
    return {
        "file": str(candidates[0]),
        "K": float(row.get("K", float("nan"))),
        "Test_AvgLogLik": float(row.get("Test_AvgLogLik", float("nan"))),
    }


def main() -> int:
    np.random.seed(RANDOM_STATE)
    paths = ensure_dirs()
    setup_logging()

    logging.info("Loading data from %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    split = filter_and_split(df)

    ts_col = split.ts_col

    numeric_cols = get_numeric_indicator_columns(df, ts_col)
    if not numeric_cols:
        raise RuntimeError("No numeric indicator columns found after dropping OHLCV.")

    keep_cols = drop_high_missing(split.train_df, numeric_cols)
    if not keep_cols:
        raise RuntimeError("All indicator columns dropped due to missingness threshold.")

    train_df = split.train_df[[ts_col] + keep_cols].copy()
    test_df = split.test_df[[ts_col] + keep_cols].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df[keep_cols])
    X_test = imputer.transform(test_df[keep_cols]) if not test_df.empty else np.empty((0, len(keep_cols)))

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test.size else X_test

    logging.info("Train rows: %d, Test rows: %d", X_train_scaled.shape[0], X_test_scaled.shape[0])
    logging.info("Indicator features kept: %d", len(keep_cols))

    # Grouping
    logging.info("Building correlation-based groups")
    corr_groups = correlation_groups(X_train_scaled, keep_cols)

    logging.info("Building name-based groups")
    name_groups = name_based_groups(keep_cols)

    logging.info("Building PCA-based groups")
    pca_groups_list = pca_groups(X_train_scaled, keep_cols)

    candidates = corr_groups + name_groups + pca_groups_list
    if not candidates:
        raise RuntimeError("No feature groups were generated.")

    save_feature_groups(candidates, paths["results"])

    # Evaluate each group
    results_rows: List[Dict[str, float]] = []
    group_best_map: Dict[str, Dict[str, float]] = {}

    for g in candidates:
        logging.info("Evaluating group %s (%s) with %d features", g.group_id, g.method, len(g.features))
        X_train_g, X_test_g, pca_meta = prepare_group_data(X_train_scaled, X_test_scaled, keep_cols, g.features)

        group_rows = []

        for cov_type in COV_TYPES_TO_TRY:
            for k in K_LIST:
                metrics = evaluate_gmm_for_group(X_train_g, X_test_g, cov_type, k)

                model_id = f"gmm_indicators_{g.method}_{g.group_id}_{cov_type}_K{k}"
                row = {
                    "Model_ID": model_id,
                    "Group_Method": g.method,
                    "Group_ID": g.group_id,
                    "Group_Name": g.group_name,
                    "N_Features": len(g.features),
                    "Feature_List_Short": feature_list_short(g.features),
                    "Covariance_Type": cov_type,
                    "K": k,
                    "Train_Start": TRAIN_START,
                    "Train_End": TRAIN_END,
                    "Test_Start": TEST_START,
                    "Test_End": TEST_END,
                    "Train_AvgLogLik": metrics["Train_AvgLogLik"],
                    "Test_AvgLogLik": metrics["Test_AvgLogLik"],
                    "Train_BIC": metrics["Train_BIC"],
                    "Train_AIC": metrics["Train_AIC"],
                    "Train_Silhouette": metrics["Train_Silhouette"],
                    "Train_DaviesBouldin": metrics["Train_DaviesBouldin"],
                    "Test_Silhouette": metrics["Test_Silhouette"],
                    "Test_DaviesBouldin": metrics["Test_DaviesBouldin"],
                    "Train_RespEntropy": metrics["Train_RespEntropy"],
                    "Test_RespEntropy": metrics["Test_RespEntropy"],
                    "MultiRun_LL_Std": float("nan"),
                    "MultiRun_Weight_Std_Mean": float("nan"),
                    "MultiRun_MeanShift_Std_Mean": float("nan"),
                    "PCA_Used": pca_meta["pca_used"],
                    "PCA_Components": pca_meta["pca_components"],
                }
                results_rows.append(row)
                group_rows.append(row)

        # Multi-run stability for best row in this group
        best_idx = select_best_row(group_rows)
        best_row = group_rows[best_idx]
        best_cov = best_row["Covariance_Type"]
        best_k = int(best_row["K"])

        ll_std, weight_std_mean, mean_shift_std_mean = multi_run_stability(X_train_g, best_cov, best_k)

        best_row["MultiRun_LL_Std"] = ll_std
        best_row["MultiRun_Weight_Std_Mean"] = weight_std_mean
        best_row["MultiRun_MeanShift_Std_Mean"] = mean_shift_std_mean

        group_best_map[g.group_id] = {
            **best_row,
            "features": g.features,
        }

    metrics_df = pd.DataFrame(results_rows)
    metrics_path = paths["results"] / "group_gmm_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Best groups summary
    top_by_test = metrics_df.sort_values("Test_AvgLogLik", ascending=False).head(20).copy()
    top_by_test["Rank_Type"] = "Top_by_Test_AvgLogLik"

    top_by_bic = metrics_df.sort_values("Train_BIC", ascending=True).head(20).copy()
    top_by_bic["Rank_Type"] = "Top_by_Train_BIC"

    best_summary = pd.concat([top_by_test, top_by_bic], ignore_index=True)
    best_summary_path = paths["results"] / "best_groups_summary.csv"
    best_summary.to_csv(best_summary_path, index=False)

    # Greedy group combination
    greedy_rows = []

    best_single = max(group_best_map.values(), key=lambda r: r["Test_AvgLogLik"])
    selected_groups = [best_single["Group_ID"]]
    selected_features = set(best_single["features"])
    current_best_ll = best_single["Test_AvgLogLik"]

    def group_pool() -> List[str]:
        ids = list(group_best_map.keys())
        if GREEDY_CANDIDATE_LIMIT is not None:
            ids = sorted(ids, key=lambda gid: group_best_map[gid]["Test_AvgLogLik"], reverse=True)[:GREEDY_CANDIDATE_LIMIT]
        return ids

    def best_combo_for_features(feature_list: List[str]) -> Dict[str, float]:
        X_train_g, X_test_g, _ = prepare_group_data(X_train_scaled, X_test_scaled, keep_cols, feature_list)
        best = None
        for cov_type in COV_TYPES_TO_TRY:
            for k in K_LIST:
                metrics = evaluate_gmm_for_group(X_train_g, X_test_g, cov_type, k)
                if best is None or metrics["Test_AvgLogLik"] > best["Test_AvgLogLik"]:
                    best = {
                        **metrics,
                        "Covariance_Type": cov_type,
                        "K": k,
                    }
        return best

    # Step 0
    greedy_rows.append({
        "step_idx": 0,
        "added_group_id": best_single["Group_ID"],
        "added_group_name": best_single["Group_Name"],
        "total_groups_in_combo": 1,
        "total_feature_count": len(selected_features),
        "best_cov_type": best_single["Covariance_Type"],
        "best_K": int(best_single["K"]),
        "Train_AvgLogLik": best_single["Train_AvgLogLik"],
        "Test_AvgLogLik": best_single["Test_AvgLogLik"],
        "Train_BIC": best_single["Train_BIC"],
        "Train_AIC": best_single["Train_AIC"],
        "Train_RespEntropy": best_single["Train_RespEntropy"],
        "Test_RespEntropy": best_single["Test_RespEntropy"],
    })

    for step in range(1, GREEDY_MAX_GROUPS):
        best_candidate = None
        best_candidate_id = None

        for gid in group_pool():
            if gid in selected_groups:
                continue
            candidate_features = list(selected_features.union(group_best_map[gid]["features"]))
            metrics = best_combo_for_features(candidate_features)
            if best_candidate is None or metrics["Test_AvgLogLik"] > best_candidate["Test_AvgLogLik"]:
                best_candidate = metrics
                best_candidate_id = gid

        if best_candidate is None:
            break
        if best_candidate["Test_AvgLogLik"] <= current_best_ll:
            break

        selected_groups.append(best_candidate_id)
        selected_features.update(group_best_map[best_candidate_id]["features"])
        current_best_ll = best_candidate["Test_AvgLogLik"]

        greedy_rows.append({
            "step_idx": step,
            "added_group_id": best_candidate_id,
            "added_group_name": group_best_map[best_candidate_id]["Group_Name"],
            "total_groups_in_combo": len(selected_groups),
            "total_feature_count": len(selected_features),
            "best_cov_type": best_candidate["Covariance_Type"],
            "best_K": int(best_candidate["K"]),
            "Train_AvgLogLik": best_candidate["Train_AvgLogLik"],
            "Test_AvgLogLik": best_candidate["Test_AvgLogLik"],
            "Train_BIC": best_candidate["Train_BIC"],
            "Train_AIC": best_candidate["Train_AIC"],
            "Train_RespEntropy": best_candidate["Train_RespEntropy"],
            "Test_RespEntropy": best_candidate["Test_RespEntropy"],
        })

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_path = paths["results"] / "greedy_group_combo.csv"
    greedy_df.to_csv(greedy_path, index=False)

    # Optional plots
    plot_top_groups(metrics_df, paths["plots"])
    plot_greedy_curve(greedy_df, paths["plots"])

    baseline = find_ohlcv_baseline(paths["results"])

    logging.info("=== Summary ===")
    logging.info("Best single group (by Test AvgLogLik): %s", best_single["Group_ID"])
    logging.info("Best greedy combo steps: %d", len(greedy_df))
    if baseline:
        logging.info("OHLCV baseline from %s | Test AvgLogLik=%.6f", baseline["file"], baseline["Test_AvgLogLik"])
    else:
        logging.info("OHLCV baseline not provided")

    logging.info("Outputs:")
    logging.info("- %s", metrics_path)
    logging.info("- %s", best_summary_path)
    logging.info("- %s", greedy_path)
    logging.info("- %s", paths["results"] / "feature_groups.json")
    logging.info("- %s", paths["results"] / "feature_groups.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
