#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover
    ks_2samp = None
    wasserstein_distance = None

import matplotlib
matplotlib.use("Agg")


# =========================
# CONFIG
# =========================
DATA_DIR = "data/external/12H"
LABEL_SCRIPT = "scripts/Signal_code_hour_version_1_0.py"
OUT_DIR = "reports"
SEED = 42
SHIFTS = [1, 2, 3, 6]
R = 100
TOPK = 20
NEG_POS_RATIO = 5
CORR_THRESHOLD = 0.95
STABILITY_MIN = 0.30
KS_MIN = 0.10
MAX_ITER = 5000

# Performance tuning
FAST_MODE = True  # set False for full run
PERM_REPEATS = 5
RF_ESTIMATORS = 150
C_GRID = [0.01, 0.1, 1.0, 10.0]
REGIME_KS = [3, 4, 5, 6]
REGIME_SHIFTS = "all"  # "all" or "best_only"
TOP_FEATURE_PLOTS = 5

fast_env = os.getenv("FEATURE_DISCOVERY_FAST")
if fast_env is not None:
    FAST_MODE = fast_env.strip().lower() in {"1", "true", "yes", "y"}

if FAST_MODE:
    SHIFTS = [1, 2, 3]
    R = 20
    MAX_ITER = 2000
    PERM_REPEATS = 3
    RF_ESTIMATORS = 80
    C_GRID = [0.1, 1.0]
    REGIME_KS = [3, 4]
    REGIME_SHIFTS = "best_only"
    TOP_FEATURE_PLOTS = 3


TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime", "Date", "Datetime", "ts_utc"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
LABEL_COL = "label"
EXCLUDE_SUBSTRINGS = ["timestamp", "time", "date", "datetime", "local_"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def resolve_data_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path("data/external/12h")
    if alt.exists():
        logging.warning("DATA_DIR %s not found. Falling back to %s", p, alt)
        return alt
    raise FileNotFoundError(f"DATA_DIR not found: {p}")


def resolve_label_script(path_str: str) -> Path:
    candidates = [
        Path(path_str),
        Path("scripts/Signal_code_hour_version_1_0.py"),
        Path("scripts/signals_code_hour_version_1_0.py"),
        Path("scripts/signals_code_hour_v1_0.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Label script not found. Tried: {candidates}")


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c
    for c in cols:
        if "timestamp" in c.lower():
            return c
    return None


def normalize_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    if ts_col != "timestamp":
        out = out.rename(columns={ts_col: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def list_csvs(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.csv"))


def select_by_keywords(files: List[Path]) -> List[Path]:
    keywords = {
        "market_structure": ["market_structure"],
        "order_flow_derivatives": ["orderflow", "order_flow", "derivatives"],
        "oscillator": ["oscillator"],
        "trend": ["trend"],
        "volatility": ["volatility"],
        "volume": ["volume"],
    }
    selected = []
    lower_map = {f: f.name.lower() for f in files}
    for key, pats in keywords.items():
        match = None
        for f, name in lower_map.items():
            if any(pat in name for pat in pats):
                match = f
                break
        if match is not None and match not in selected:
            selected.append(match)
    if len(selected) < 6:
        for f in files:
            if f not in selected:
                selected.append(f)
            if len(selected) >= 6:
                break
    return selected


def load_and_prepare_csv(path: Path) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(path, low_memory=False)
    ts_col = detect_timestamp_column(df)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}")
    dup = int(df[ts_col].duplicated().sum())
    df = normalize_timestamp(df, ts_col)
    return df, dup


def drop_duplicate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in OHLCV_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def merge_dataframes(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    if not dfs:
        raise ValueError("No dataframes to merge.")

    base_idx = next((i for i, df in enumerate(dfs) if all(c in df.columns for c in OHLCV_COLS)), 0)
    base = dfs[base_idx]
    merged_inner = base.copy()
    merged_outer = base.copy()
    for i, df in enumerate(dfs):
        if i == base_idx:
            continue
        df_reduced = drop_duplicate_ohlcv(df)
        merged_inner = merged_inner.merge(df_reduced, on="timestamp", how="inner")
        merged_outer = merged_outer.merge(df_reduced, on="timestamp", how="outer")

    diagnostics = {
        "inner_rows": int(merged_inner.shape[0]),
        "outer_rows": int(merged_outer.shape[0]),
    }
    return merged_inner, merged_outer, diagnostics


def decide_merge(inner_df: pd.DataFrame, outer_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    inner_rows = inner_df.shape[0]
    outer_rows = max(outer_df.shape[0], 1)
    drop_ratio = 1.0 - (inner_rows / outer_rows)
    if drop_ratio > 0.30:
        logging.warning("Inner join dropped %.2f%% rows. Using OUTER join.", drop_ratio * 100)
        return outer_df, "outer"
    return inner_df, "inner"

def label_merged(df: pd.DataFrame, label_script: Path) -> pd.DataFrame:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("label_module", label_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import label script: {label_script}")
    label_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = label_mod
    spec.loader.exec_module(label_mod)  # type: ignore

    labels_df = label_mod.label_dataframe(df, timestamp_col="timestamp")
    labels_df = labels_df.rename(columns={"candle_type": LABEL_COL})
    merged = df.merge(labels_df, on="timestamp", how="left")
    merged[LABEL_COL] = merged[LABEL_COL].astype(str).str.strip().str.lower()
    merged.loc[~merged[LABEL_COL].isin(["long", "short", "skip"]), LABEL_COL] = "skip"
    return merged


def shift_features(df: pd.DataFrame, feature_cols: Sequence[str], shift: int) -> pd.DataFrame:
    out = df.copy()
    out.loc[:, feature_cols] = out.loc[:, feature_cols].shift(shift)
    return out


def remove_constant_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    kept = []
    removed = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if float(np.nanvar(values.to_numpy())) < 1e-12:
            removed.append(col)
        else:
            kept.append(col)
    return kept, removed


def correlation_prune(df: pd.DataFrame, feature_cols: Sequence[str], threshold: float) -> Tuple[List[str], List[str]]:
    if len(feature_cols) <= 1:
        return list(feature_cols), []
    corr = df[feature_cols].corr(method="spearman").abs()
    to_drop = set()
    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if corr.iat[i, j] > threshold:
                to_drop.add(feature_cols[j])
    kept = [c for c in feature_cols if c not in to_drop]
    return kept, sorted(list(to_drop))


def forward_fill_and_median(train_df: pd.DataFrame, full_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = full_df.copy()
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].ffill()
    medians = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    out[feature_cols] = out[feature_cols].fillna(medians)
    return out


def compute_missingness(df: pd.DataFrame, feature_cols: Sequence[str]) -> float:
    if not feature_cols:
        return 0.0
    tmp = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return float(tmp.isna().mean().mean())


def time_split_three(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = df.shape[0]
    train_end = int(round(n * 0.8))
    val_end = int(round(n * 0.9))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def subsample_indices(rng: np.random.RandomState, y: np.ndarray, neg_pos_ratio: int) -> np.ndarray:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size == 0:
        return np.array([], dtype=int)
    target_neg = min(neg_idx.size, pos_idx.size * neg_pos_ratio)
    neg_sample = rng.choice(neg_idx, size=target_neg, replace=False) if target_neg > 0 else np.array([], dtype=int)
    return np.concatenate([pos_idx, neg_sample])


def compute_distribution_stats(x_pos: np.ndarray, x_neg: np.ndarray) -> Tuple[float, float]:
    if x_pos.size == 0 or x_neg.size == 0:
        return 0.0, 0.0
    ks = float(ks_2samp(x_pos, x_neg).statistic) if ks_2samp else 0.0
    wasser = float(wasserstein_distance(x_pos, x_neg)) if wasserstein_distance else 0.0
    return ks, wasser

def fit_logistic_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    rng: np.random.RandomState,
    repeats: int,
    neg_pos_ratio: int,
    fixed_c: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[List[str]], List[List[str]]]:
    coef_scores = {f: [] for f in feature_names}
    perm_scores = {f: [] for f in feature_names}
    topk_logit = []
    topk_perm = []

    Cs = [fixed_c] if fixed_c is not None else C_GRID
    rank_lists = []
    for r in range(repeats):
        idx = subsample_indices(rng, y_train, neg_pos_ratio)
        if idx.size == 0:
            continue
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_val_scaled = scaler.transform(X_val)

        best_model = None
        best_ap = -1.0
        for C in Cs:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=C,
                max_iter=MAX_ITER,
                class_weight="balanced",
                random_state=rng.randint(0, 1_000_000),
            )
            model.fit(X_sub_scaled, y_sub)
            preds = model.predict_proba(X_val_scaled)[:, 1]
            ap = average_precision_score(y_val, preds)
            if ap > best_ap:
                best_ap = ap
                best_model = model

        if best_model is None:
            continue

        coefs = np.abs(best_model.coef_.ravel())
        coef_rank = [feature_names[i] for i in np.argsort(-coefs)]
        rank_lists.append(coef_rank)
        topk_logit.append(coef_rank[:TOPK])
        for feat, score in zip(feature_names, coefs):
            coef_scores[feat].append(float(score))

    # Single permutation importance on full train/val to reduce runtime.
    try:
        scaler_full = StandardScaler()
        X_train_scaled = scaler_full.fit_transform(X_train)
        X_val_scaled = scaler_full.transform(X_val)
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=fixed_c if fixed_c is not None else 1.0,
            max_iter=MAX_ITER,
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(X_train_scaled, y_train)
        perm = permutation_importance(
            model,
            X_val_scaled,
            y_val,
            n_repeats=PERM_REPEATS,
            scoring="average_precision",
            random_state=rng.randint(0, 1_000_000),
        )
        perm_scores_arr = perm.importances_mean
        perm_rank = [feature_names[i] for i in np.argsort(-perm_scores_arr)]
        topk_perm.append(perm_rank[:TOPK])
        for feat, score in zip(feature_names, perm_scores_arr):
            perm_scores[feat].append(float(score))
    except Exception:
        pass

    coef_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in coef_scores.items()}
    perm_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in perm_scores.items()}
    rank_median = {f: float(np.median([r.index(f) + 1 for r in rank_lists])) if rank_lists else float("inf") for f in feature_names}
    return coef_mean, perm_mean, rank_median, topk_logit, topk_perm


def fit_tree_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    rng: np.random.RandomState,
    repeats: int,
    neg_pos_ratio: int,
) -> Tuple[Dict[str, float], List[List[str]]]:
    imp_scores = {f: [] for f in feature_names}
    topk_lists = []

    for r in range(repeats):
        idx = subsample_indices(rng, y_train, neg_pos_ratio)
        if idx.size == 0:
            continue
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            max_depth=None,
            random_state=rng.randint(0, 1_000_000),
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_sub, y_sub)
        imps = model.feature_importances_
        rank = [feature_names[i] for i in np.argsort(-imps)]
        topk_lists.append(rank[:TOPK])
        for feat, score in zip(feature_names, imps):
            imp_scores[feat].append(float(score))

    imp_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in imp_scores.items()}
    return imp_mean, topk_lists


def selection_frequency(topk_lists: List[List[str]], feature_names: List[str]) -> Dict[str, float]:
    counts = {f: 0 for f in feature_names}
    for lst in topk_lists:
        for feat in lst:
            counts[feat] += 1
    total = max(len(topk_lists), 1)
    return {f: counts[f] / total for f in feature_names}


def rank_from_scores(scores: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=higher_is_better)
    return {feat: rank + 1 for rank, (feat, _) in enumerate(items)}


def fused_rank(ranks: List[Dict[str, int]], feature_names: List[str]) -> Dict[str, float]:
    fused = {}
    for feat in feature_names:
        fused[feat] = float(np.mean([r[feat] for r in ranks]))
    return fused


def make_plots(plots_dir: Path, shift: int, class_name: str, ranking_df: pd.DataFrame, train_df: pd.DataFrame) -> List[Path]:
    import matplotlib.pyplot as plt

    plot_paths = []
    top20 = ranking_df.sort_values("fused_rank").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top20["feature"][::-1], (1 / top20["fused_rank"])[::-1])
    ax.set_title(f"{class_name.upper()} Shift {shift} Top 20 (Fused Rank)")
    ax.set_xlabel("1 / fused_rank")
    fig.tight_layout()
    out_path = plots_dir / f"shift{shift}_{class_name}_top20_fused.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plot_paths.append(out_path)

    topn = ranking_df.sort_values("fused_rank").head(TOP_FEATURE_PLOTS)["feature"].tolist()
    fig, axes = plt.subplots(1, len(topn), figsize=(4 * len(topn), 4), sharey=False)
    if len(topn) == 1:
        axes = [axes]
    for ax, feat in zip(axes, topn):
        pos = train_df[train_df[LABEL_COL] == class_name][feat].dropna()
        neg = train_df[train_df[LABEL_COL] != class_name][feat].dropna()
        ax.hist(pos, bins=30, alpha=0.6, label="pos")
        ax.hist(neg, bins=30, alpha=0.6, label="neg")
        ax.set_title(feat)
    axes[0].legend()
    fig.tight_layout()
    out_path = plots_dir / f"shift{shift}_{class_name}_top5_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plot_paths.append(out_path)

    return plot_paths


def plot_stability(plots_dir: Path, shift: int, class_name: str, ranking_df: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    top = ranking_df.sort_values("fused_rank").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["selection_freq_logit"][::-1])
    ax.set_title(f"{class_name.upper()} Shift {shift} Stability (Logit)")
    ax.set_xlabel("Selection Frequency")
    fig.tight_layout()
    out_path = plots_dir / f"shift{shift}_{class_name}_stability.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_auc(plots_dir: Path, shift: int, pr_auc_rows: List[Dict[str, float]]) -> Path:
    import matplotlib.pyplot as plt

    classes = ["long", "short", "skip"]
    valid_scores = [r["pr_auc_valid"] for r in pr_auc_rows]
    test_scores = [r["pr_auc_test"] for r in pr_auc_rows]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.15, valid_scores, width=0.3, label="valid")
    ax.bar(x + 0.15, test_scores, width=0.3, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes])
    ax.set_title(f"Shift {shift} PR-AUC")
    ax.legend()
    fig.tight_layout()
    out_path = plots_dir / f"shift{shift}_pr_auc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def choose_regime_features(feature_cols: List[str]) -> List[str]:
    patterns = ["tr_percentile", "atr_pct", "bb_width", "adx", "volume_roc", "returns"]
    selected = [c for c in feature_cols if any(p in c for p in patterns)]
    if len(selected) > 12:
        selected = selected[:12]
    if len(selected) < 6:
        selected = feature_cols[:12]
    return selected


def fit_regime_model(X_train: np.ndarray) -> Tuple[GaussianMixture, int]:
    best_bic = float("inf")
    best_model = None
    best_k = 3
    for k in REGIME_KS:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=SEED,
            n_init=3,
            max_iter=500,
        )
        model.fit(X_train)
        bic = model.bic(X_train)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = k
    if best_model is None:
        raise RuntimeError("Failed to fit regime GMM.")
    return best_model, best_k

def run_feature_discovery(
    df: pd.DataFrame,
    feature_cols: List[str],
    class_name: str,
    shift: int,
    rng: np.random.RandomState,
    ranks_dir: Path,
    plots_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Path]]:
    y = (df[LABEL_COL] == class_name).astype(int).to_numpy()
    train_df, val_df, test_df = time_split_three(df)

    X_train = train_df[feature_cols].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_train = (train_df[LABEL_COL] == class_name).astype(int).to_numpy()
    y_val = (val_df[LABEL_COL] == class_name).astype(int).to_numpy()
    y_test = (test_df[LABEL_COL] == class_name).astype(int).to_numpy()

    neg_pos_ratio = NEG_POS_RATIO if y_train.sum() >= 300 else 10

    best_c = None
    best_ap = 0.0
    if y_val.sum() > 0 and y_train.sum() > 0:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        for C in [0.01, 0.1, 1.0, 10.0]:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=C,
                max_iter=MAX_ITER,
                class_weight="balanced",
                random_state=SEED,
            )
            model.fit(X_train_scaled, y_train)
            preds = model.predict_proba(X_val_scaled)[:, 1]
            ap = average_precision_score(y_val, preds)
            if ap > best_ap:
                best_ap = ap
                best_c = C

    coef_scores, perm_scores, rank_median_logit, topk_logit, topk_perm = fit_logistic_models(
        X_train, y_train, X_val, y_val, feature_cols, rng, R, neg_pos_ratio, fixed_c=best_c
    )
    tree_scores, topk_tree = fit_tree_models(
        X_train, y_train, feature_cols, rng, R, neg_pos_ratio
    )

    mi_scores = {}
    try:
        mi = mutual_info_classif(train_df[feature_cols], y_train, discrete_features=False, random_state=SEED)
        mi_scores = {f: float(v) for f, v in zip(feature_cols, mi)}
    except Exception:
        mi_scores = {f: 0.0 for f in feature_cols}

    ks_scores = {}
    wasser_scores = {}
    for feat in feature_cols:
        x_pos = train_df.loc[train_df[LABEL_COL] == class_name, feat].to_numpy()
        x_neg = train_df.loc[train_df[LABEL_COL] != class_name, feat].to_numpy()
        ks, wasser = compute_distribution_stats(x_pos, x_neg)
        ks_scores[feat] = ks
        wasser_scores[feat] = wasser

    rank_logit = rank_from_scores(coef_scores, higher_is_better=True)
    rank_logit_median = {}
    for f in feature_cols:
        rm = rank_median_logit.get(f, float("inf"))
        if not np.isfinite(rm):
            rm = float(rank_logit[f])
        rank_logit_median[f] = int(rm)
    rank_perm = rank_from_scores(perm_scores, higher_is_better=True)
    rank_mi = rank_from_scores(mi_scores, higher_is_better=True)
    rank_ks = rank_from_scores(ks_scores, higher_is_better=True)

    fused = fused_rank([rank_logit_median, rank_perm, rank_mi, rank_ks], feature_cols)
    selection_freq_logit = selection_frequency(topk_logit, feature_cols)
    selection_freq_tree = selection_frequency(topk_tree, feature_cols)
    selection_freq_perm = selection_frequency(topk_perm, feature_cols)

    ranking_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "logit_coef": [coef_scores[f] for f in feature_cols],
            "tree_importance": [tree_scores[f] for f in feature_cols],
            "perm_importance": [perm_scores[f] for f in feature_cols],
            "mi": [mi_scores[f] for f in feature_cols],
            "ks": [ks_scores[f] for f in feature_cols],
            "wasserstein": [wasser_scores[f] for f in feature_cols],
            "rank_logit": [rank_logit[f] for f in feature_cols],
            "rank_logit_median": [rank_logit_median[f] for f in feature_cols],
            "rank_perm": [rank_perm[f] for f in feature_cols],
            "rank_mi": [rank_mi[f] for f in feature_cols],
            "rank_ks": [rank_ks[f] for f in feature_cols],
            "fused_rank": [fused[f] for f in feature_cols],
            "selection_freq_logit": [selection_freq_logit[f] for f in feature_cols],
            "selection_freq_tree": [selection_freq_tree[f] for f in feature_cols],
            "selection_freq_perm": [selection_freq_perm[f] for f in feature_cols],
        }
    ).sort_values("fused_rank")

    out_csv = ranks_dir / f"{class_name}_rankings.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(out_csv, index=False)

    fused_top = ranking_df.head(50)
    gated = fused_top[(fused_top["selection_freq_logit"] >= STABILITY_MIN) & (fused_top["ks"] >= KS_MIN)]
    if gated.empty:
        gated = ranking_df.head(TOPK)

    gated_features = gated.head(TOPK)["feature"].tolist()

    if gated_features:
        scaler = StandardScaler()
        X_train_g = scaler.fit_transform(X_train[:, [feature_cols.index(f) for f in gated_features]])
        X_val_g = scaler.transform(X_val[:, [feature_cols.index(f) for f in gated_features]])
        X_test_g = scaler.transform(X_test[:, [feature_cols.index(f) for f in gated_features]])

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=best_c if best_c is not None else 1.0,
            max_iter=MAX_ITER,
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(X_train_g, y_train)
        pr_auc_valid = average_precision_score(y_val, model.predict_proba(X_val_g)[:, 1]) if y_val.sum() > 0 else 0.0
        pr_auc_test = average_precision_score(y_test, model.predict_proba(X_test_g)[:, 1]) if y_test.sum() > 0 else 0.0
    else:
        pr_auc_valid = 0.0
        pr_auc_test = 0.0

    plots = make_plots(plots_dir, shift, class_name, ranking_df, train_df)
    plots.append(plot_stability(plots_dir, shift, class_name, ranking_df))

    stats = {
        "class": class_name,
        "shift": shift,
        "pr_auc_valid": pr_auc_valid,
        "pr_auc_test": pr_auc_test,
        "gated_feature_count": len(gated_features),
        "gated_features": gated_features[:TOPK],
        "ks_mean_top20": float(ranking_df.head(20)["ks"].mean()),
    }

    return ranking_df, stats, plots


def main() -> None:
    setup_logging()
    rng = np.random.RandomState(SEED)

    data_dir = resolve_data_dir(DATA_DIR)
    label_script = resolve_label_script(LABEL_SCRIPT)
    out_dir = Path(OUT_DIR)
    plots_dir = out_dir / "plots_12H"
    ranks_dir = out_dir / "feature_rankings_12H"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    files = list_csvs(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    logging.info("Detected CSVs:")
    for f in files:
        logging.info(" - %s", f)

    if len(files) > 6:
        files = select_by_keywords(files)
        logging.info("Selected 6 files by keyword matching:")
        for f in files:
            logging.info(" - %s", f)

    logging.info("FAST_MODE=%s SHIFTS=%s R=%d MAX_ITER=%d REGIME_SHIFTS=%s",
                 FAST_MODE, SHIFTS, R, MAX_ITER, REGIME_SHIFTS)

    dfs = []
    dup_counts = {}
    for path in files:
        df, dup = load_and_prepare_csv(path)
        dup_counts[path.name] = dup
        logging.info("Loaded %s shape=%s columns=%d", path.name, df.shape, len(df.columns))
        logging.info("Columns: %s", ", ".join(df.columns))
        dfs.append(df)

    inner_df, outer_df, merge_diag = merge_dataframes(dfs)
    logging.info("Merge diagnostics: inner_rows=%d outer_rows=%d", merge_diag["inner_rows"], merge_diag["outer_rows"])
    merged_df, merge_mode = decide_merge(inner_df, outer_df)
    logging.info("Merge mode selected: %s rows=%d", merge_mode, merged_df.shape[0])

    merged_path = Path("data/processed/12H_merged_indicators.csv")
    merged_df.to_csv(merged_path, index=False)
    logging.info("Saved merged dataset: %s", merged_path)

    labeled_df = label_merged(merged_df, label_script)
    labeled_path = Path("data/processed/12H_merged_indicators_labeled.csv")
    labeled_df.to_csv(labeled_path, index=False)
    logging.info("Saved labeled dataset: %s", labeled_path)

    label_counts = labeled_df[LABEL_COL].value_counts().to_dict()
    logging.info("Label counts: %s", label_counts)

    numeric_cols = labeled_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["timestamp"] + OHLCV_COLS + [LABEL_COL]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and not any(s in c.lower() for s in EXCLUDE_SUBSTRINGS)]
    logging.info("Initial feature count: %d", len(feature_cols))

    summary_rows = []
    report_lines = []
    report_lines.append("# 12H Feature Discovery Report")
    report_lines.append("")
    report_lines.append("## Input Files")
    for f in files:
        report_lines.append(f"- {f}")
    report_lines.append("")
    report_lines.append("## Merge Summary")
    report_lines.append(f"- inner_rows: {merge_diag['inner_rows']}")
    report_lines.append(f"- outer_rows: {merge_diag['outer_rows']}")
    report_lines.append(f"- merge_mode: {merge_mode}")
    report_lines.append(f"- duplicate_timestamps_by_file: {dup_counts}")
    report_lines.append("")
    report_lines.append("## Label Summary")
    report_lines.append(f"- counts: {label_counts}")
    report_lines.append("")
    report_lines.append("## Feature Pruning")
    report_lines.append(f"- initial_features: {len(feature_cols)}")
    report_lines.append(f"- corr_threshold: {CORR_THRESHOLD}")
    report_lines.append("")
    report_lines.append("## Shift Sweep")
    report_lines.append(f"- shifts: {SHIFTS}")
    report_lines.append("- feature_shift means features at t use data from <= t-shift")
    report_lines.append("")

    for shift in SHIFTS:
        report_lines.append(f"### Shift {shift}")
        df_shifted = shift_features(labeled_df, feature_cols, shift=shift)
        df_shifted = df_shifted.iloc[shift:].reset_index(drop=True)
        train_df, val_df, test_df = time_split_three(df_shifted)

        missing_before = compute_missingness(train_df, feature_cols)
        df_shifted = forward_fill_and_median(train_df, df_shifted, feature_cols)
        missing_after = compute_missingness(df_shifted, feature_cols)
        # Re-split after imputation to ensure downstream models see clean data.
        train_df, val_df, test_df = time_split_three(df_shifted)

        pruned_features, removed_const = remove_constant_features(df_shifted, feature_cols)
        pruned_features, removed_corr = correlation_prune(train_df, pruned_features, threshold=CORR_THRESHOLD)

        report_lines.append(f"- removed_constants: {len(removed_const)}")
        report_lines.append(f"- removed_corr: {len(removed_corr)}")
        report_lines.append(f"- missingness_before: {missing_before:.4f}")
        report_lines.append(f"- missingness_after: {missing_after:.4f}")

        pr_auc_rows = []
        for class_name in ["long", "short", "skip"]:
            rank_dir = ranks_dir / str(shift) / "global"
            ranking_df, stats, plots = run_feature_discovery(
                df_shifted,
                pruned_features,
                class_name,
                shift,
                rng,
                rank_dir,
                plots_dir,
            )
            pr_auc_rows.append(stats)
            summary_rows.append(stats)
            report_lines.append(f"- {class_name} pr_auc_valid={stats['pr_auc_valid']:.4f} pr_auc_test={stats['pr_auc_test']:.4f}")
            report_lines.append(f"- {class_name} gated_features={stats['gated_features']}")

        plot_pr_auc(plots_dir, shift, pr_auc_rows)

        if REGIME_SHIFTS == "all":
            report_lines.append("#### Regime Conditioning")
            regime_features = choose_regime_features(pruned_features)
            report_lines.append(f"- regime_features_selected: {regime_features}")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[regime_features].to_numpy())
            gmm, best_k = fit_regime_model(X_train)
            report_lines.append(f"- regime_K_selected: {best_k}")
            full_regime = gmm.predict(scaler.transform(df_shifted[regime_features].to_numpy()))
            df_shifted = df_shifted.copy()
            df_shifted["regime_id"] = full_regime

            for regime_id in range(best_k):
                df_reg = df_shifted[df_shifted["regime_id"] == regime_id].copy()
                if df_reg.shape[0] < 200:
                    report_lines.append(f"- regime_{regime_id}: skipped (n={df_reg.shape[0]})")
                    continue
                report_lines.append(f"- regime_{regime_id}: n={df_reg.shape[0]}")
                for class_name in ["long", "short", "skip"]:
                    rank_dir = ranks_dir / str(shift) / f"regime_{regime_id}"
                    ranking_df, stats, plots = run_feature_discovery(
                        df_reg,
                        pruned_features,
                        class_name,
                        shift,
                        rng,
                        rank_dir,
                        plots_dir,
                    )
                    summary_rows.append({**stats, "regime_id": regime_id})
                    report_lines.append(
                        f"  - {class_name} pr_auc_valid={stats['pr_auc_valid']:.4f} pr_auc_test={stats['pr_auc_test']:.4f}"
                    )

        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)
    shift_summary = summary_df[["shift", "class", "pr_auc_valid", "pr_auc_test", "gated_feature_count", "ks_mean_top20"]]
    shift_summary.to_csv(ranks_dir / "shift_summary.csv", index=False)

    if REGIME_SHIFTS == "best_only":
        report_lines.append("## Regime Conditioning (Best Shift Only)")
        best_shift = shift_summary[shift_summary["class"].isin(["long", "short"])].groupby("shift")["pr_auc_valid"].mean().sort_values(ascending=False).index[0]
        report_lines.append(f"- selected_shift_for_regimes: {int(best_shift)}")

        df_shifted = shift_features(labeled_df, feature_cols, shift=int(best_shift))
        df_shifted = df_shifted.iloc[int(best_shift):].reset_index(drop=True)
        train_df, val_df, test_df = time_split_three(df_shifted)
        df_shifted = forward_fill_and_median(train_df, df_shifted, feature_cols)
        train_df, val_df, test_df = time_split_three(df_shifted)
        pruned_features, _ = remove_constant_features(df_shifted, feature_cols)
        pruned_features, _ = correlation_prune(train_df, pruned_features, threshold=CORR_THRESHOLD)

        regime_features = choose_regime_features(pruned_features)
        report_lines.append(f"- regime_features_selected: {regime_features}")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[regime_features].to_numpy())
        gmm, best_k = fit_regime_model(X_train)
        report_lines.append(f"- regime_K_selected: {best_k}")
        full_regime = gmm.predict(scaler.transform(df_shifted[regime_features].to_numpy()))
        df_shifted = df_shifted.copy()
        df_shifted["regime_id"] = full_regime

        for regime_id in range(best_k):
            df_reg = df_shifted[df_shifted["regime_id"] == regime_id].copy()
            if df_reg.shape[0] < 200:
                report_lines.append(f"- regime_{regime_id}: skipped (n={df_reg.shape[0]})")
                continue
            report_lines.append(f"- regime_{regime_id}: n={df_reg.shape[0]}")
            for class_name in ["long", "short", "skip"]:
                rank_dir = ranks_dir / str(int(best_shift)) / f"regime_{regime_id}"
                ranking_df, stats, plots = run_feature_discovery(
                    df_reg,
                    pruned_features,
                    class_name,
                    int(best_shift),
                    rng,
                    rank_dir,
                    plots_dir,
                )
                summary_rows.append({**stats, "regime_id": regime_id})
                report_lines.append(
                    f"  - {class_name} pr_auc_valid={stats['pr_auc_valid']:.4f} pr_auc_test={stats['pr_auc_test']:.4f}"
                )

    report_lines.append("## Recommendations")
    best_long = shift_summary[shift_summary["class"] == "long"].sort_values(["pr_auc_valid", "pr_auc_test"], ascending=False).head(1)
    best_short = shift_summary[shift_summary["class"] == "short"].sort_values(["pr_auc_valid", "pr_auc_test"], ascending=False).head(1)
    if not best_long.empty:
        report_lines.append(f"- best_shift_long: {int(best_long['shift'].iloc[0])}")
    if not best_short.empty:
        report_lines.append(f"- best_shift_short: {int(best_short['shift'].iloc[0])}")
    report_lines.append("- regime_conditioning: review per-regime PR-AUC for gains")
    report_lines.append("- if PR-AUC < 0.30: labels may be noisy, consider horizon adjustment")

    report_path = Path("reports/feature_discovery_12H_report.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Artifacts created:")
    print(f"- {merged_path}")
    print(f"- {labeled_path}")
    print(f"- {report_path}")
    print(f"- {ranks_dir}/shift_summary.csv")
    print(f"- {plots_dir}/*.png")

    if not best_long.empty and not best_short.empty:
        print("\nBest shifts:")
        print(f"LONG: shift {int(best_long['shift'].iloc[0])}")
        print(f"SHORT: shift {int(best_short['shift'].iloc[0])}")


if __name__ == "__main__":
    main()
