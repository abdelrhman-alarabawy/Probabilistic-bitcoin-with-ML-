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
DATA_DIR = "data/external/1D"
LABEL_SCRIPT = "scripts/signals_code_hour_version_1_0.py"
OUT_DIR = "reports"
SEED = 42
R = 50
TOP_K = 20
NEG_POS_RATIO = 5
CORR_THRESHOLD = 0.95


TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime", "Date", "Datetime", "ts_utc"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
LABEL_COL = "trade_label"


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
    alt = Path("data/external/1d")
    if alt.exists():
        logging.warning("DATA_DIR %s not found. Falling back to %s", p, alt)
        return alt
    raise FileNotFoundError(f"DATA_DIR not found: {p}")


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
    before = int(df[ts_col].duplicated().sum()) if ts_col in df.columns else 0
    df = normalize_timestamp(df, ts_col)
    return df, before


def drop_duplicate_ohlcv(df: pd.DataFrame, keep_ohlcv: bool) -> pd.DataFrame:
    if keep_ohlcv:
        return df
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
        df_reduced = drop_duplicate_ohlcv(df, keep_ohlcv=False)
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


def label_merged(df: pd.DataFrame) -> pd.DataFrame:
    import importlib.util
    import sys

    script_path = Path(LABEL_SCRIPT)
    if not script_path.exists():
        raise FileNotFoundError(f"Label script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("label_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import label script: {script_path}")
    label_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = label_mod
    spec.loader.exec_module(label_mod)  # type: ignore

    labels_df = label_mod.label_dataframe(df, timestamp_col="timestamp")
    labels_df = labels_df.rename(columns={"candle_type": LABEL_COL})
    merged = df.merge(labels_df, on="timestamp", how="left")
    merged[LABEL_COL] = merged[LABEL_COL].astype(str).str.strip().str.lower()
    merged.loc[~merged[LABEL_COL].isin(["long", "short", "skip"]), LABEL_COL] = "skip"
    return merged


def shift_features(df: pd.DataFrame, feature_cols: Sequence[str], shift: int = 1) -> pd.DataFrame:
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


def forward_fill_and_median(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = out[feature_cols].ffill()
    medians = out[feature_cols].median(numeric_only=True)
    out[feature_cols] = out[feature_cols].fillna(medians)
    return out


def compute_missingness(df: pd.DataFrame, feature_cols: Sequence[str]) -> float:
    if not feature_cols:
        return 0.0
    return float(df[feature_cols].isna().mean().mean())


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = df.shape[0]
    cut = int(round(n * train_frac))
    cut = max(1, min(cut, n - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def subsample_indices(rng: np.random.RandomState, y: np.ndarray, neg_pos_ratio: int) -> np.ndarray:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size == 0:
        return np.array([], dtype=int)
    target_neg = min(neg_idx.size, pos_idx.size * neg_pos_ratio)
    neg_sample = rng.choice(neg_idx, size=target_neg, replace=False) if target_neg > 0 else np.array([], dtype=int)
    return np.concatenate([pos_idx, neg_sample])


def compute_distribution_stats(x_pos: np.ndarray, x_neg: np.ndarray) -> Tuple[float, float, float]:
    if x_pos.size == 0 or x_neg.size == 0:
        return 0.0, 0.0, 0.0
    ks = float(ks_2samp(x_pos, x_neg).statistic) if ks_2samp else 0.0
    wasser = float(wasserstein_distance(x_pos, x_neg)) if wasserstein_distance else 0.0
    mean_diff = float(np.mean(x_pos) - np.mean(x_neg))
    pooled = float(np.sqrt((np.var(x_pos) + np.var(x_neg)) / 2.0))
    cohen_d = mean_diff / pooled if pooled > 0 else 0.0
    return ks, wasser, cohen_d


def fit_logistic_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    rng: np.random.RandomState,
    repeats: int,
    neg_pos_ratio: int,
    fixed_c: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[List[str]], List[List[str]], List[List[str]]]:
    coef_scores = {f: [] for f in feature_names}
    perm_scores = {f: [] for f in feature_names}
    topk_lists = []
    topk_perm_lists = []
    rank_lists = []

    Cs = [fixed_c] if fixed_c is not None else [0.01, 0.1, 1.0, 10.0]
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
                max_iter=2000,
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
        topk_lists.append(coef_rank[:TOP_K])
        for feat, score in zip(feature_names, coefs):
            coef_scores[feat].append(float(score))

        perm = permutation_importance(
            best_model,
            X_val_scaled,
            y_val,
            n_repeats=5,
            scoring="average_precision",
            random_state=rng.randint(0, 1_000_000),
        )
        perm_scores_arr = perm.importances_mean
        perm_rank = [feature_names[i] for i in np.argsort(-perm_scores_arr)]
        topk_perm_lists.append(perm_rank[:TOP_K])
        for feat, score in zip(feature_names, perm_scores_arr):
            perm_scores[feat].append(float(score))

    coef_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in coef_scores.items()}
    perm_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in perm_scores.items()}
    rank_median = {f: float(np.median([r.index(f) + 1 for r in rank_lists])) if rank_lists else float("inf") for f in feature_names}
    return coef_mean, perm_mean, rank_median, topk_lists, topk_perm_lists, rank_lists


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
            n_estimators=150,
            max_depth=None,
            random_state=rng.randint(0, 1_000_000),
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_sub, y_sub)
        imps = model.feature_importances_
        rank = [feature_names[i] for i in np.argsort(-imps)]
        topk_lists.append(rank[:TOP_K])
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


def make_plots(out_dir: Path, class_name: str, ranking_df: pd.DataFrame, train_df: pd.DataFrame, label_col: str) -> List[Path]:
    import matplotlib.pyplot as plt

    plot_paths = []
    top20 = ranking_df.sort_values("fused_rank").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top20["feature"][::-1], (1 / top20["fused_rank"])[::-1])
    ax.set_title(f"{class_name.upper()} Top 20 (Fused Rank)")
    ax.set_xlabel("1 / fused_rank")
    fig.tight_layout()
    out_path = out_dir / f"{class_name}_top20_fused.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plot_paths.append(out_path)

    top5 = ranking_df.sort_values("fused_rank").head(5)["feature"].tolist()
    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4), sharey=False)
    if len(top5) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top5):
        pos = train_df[train_df[label_col] == class_name][feat].dropna()
        neg = train_df[train_df[label_col] != class_name][feat].dropna()
        ax.hist(pos, bins=30, alpha=0.6, label="pos")
        ax.hist(neg, bins=30, alpha=0.6, label="neg")
        ax.set_title(feat)
    axes[0].legend()
    fig.tight_layout()
    out_path = out_dir / f"{class_name}_top5_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plot_paths.append(out_path)
    return plot_paths


def main() -> None:
    setup_logging()
    rng = np.random.RandomState(SEED)

    data_dir = resolve_data_dir(DATA_DIR)
    out_dir = Path(OUT_DIR)
    plots_dir = out_dir / "plots"
    ranks_dir = out_dir / "feature_rankings"
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

    merged_path = Path("data/processed/1D_merged_indicators.csv")
    merged_df.to_csv(merged_path, index=False)
    logging.info("Saved merged dataset: %s", merged_path)

    labeled_df = label_merged(merged_df)
    labeled_path = Path("data/processed/1D_merged_indicators_labeled.csv")
    labeled_df.to_csv(labeled_path, index=False)
    logging.info("Saved labeled dataset: %s", labeled_path)

    label_counts = labeled_df[LABEL_COL].value_counts().to_dict()
    logging.info("Label counts: %s", label_counts)

    numeric_cols = labeled_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["timestamp"] + OHLCV_COLS + [LABEL_COL]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    logging.info("Initial feature count: %d", len(feature_cols))

    labeled_df = shift_features(labeled_df, feature_cols, shift=1)
    labeled_df = labeled_df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    missing_before = compute_missingness(labeled_df, feature_cols)
    logging.info("Missingness before impute: %.4f", missing_before)
    labeled_df = forward_fill_and_median(labeled_df, feature_cols)
    missing_after = compute_missingness(labeled_df, feature_cols)
    logging.info("Missingness after impute: %.4f", missing_after)

    feature_cols, removed_const = remove_constant_features(labeled_df, feature_cols)
    logging.info("Removed constant features: %d", len(removed_const))

    train_df, val_df = time_split(labeled_df, train_frac=0.8)
    feature_cols, removed_corr = correlation_prune(train_df, feature_cols, threshold=CORR_THRESHOLD)
    logging.info("Removed highly correlated features: %d", len(removed_corr))

    report_lines = []
    report_lines.append("# 1D Feature Discovery Report")
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
    report_lines.append(f"- initial_features: {len(numeric_cols) - len(exclude_cols)}")
    report_lines.append(f"- removed_constants: {len(removed_const)}")
    report_lines.append(f"- removed_corr: {len(removed_corr)}")
    report_lines.append(f"- missingness_before: {missing_before:.4f}")
    report_lines.append(f"- missingness_after: {missing_after:.4f}")
    report_lines.append(f"- feature_shift: 1 (to avoid leakage)")
    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append(f"- merged_raw: {merged_path}")
    report_lines.append(f"- merged_labeled: {labeled_path}")
    report_lines.append(f"- rankings_dir: {ranks_dir}")
    report_lines.append(f"- plots_dir: {plots_dir}")
    report_lines.append("")

    outputs = []

    for class_name in ["long", "short", "skip"]:
        report_lines.append(f"## Class: {class_name.upper()}")
        y_train = (train_df[LABEL_COL] == class_name).astype(int).to_numpy()
        y_val = (val_df[LABEL_COL] == class_name).astype(int).to_numpy()

        pos = int(y_train.sum())
        neg = int((y_train == 0).sum())
        ratio = (neg / max(pos, 1)) if pos else math.inf
        report_lines.append(f"- train_pos: {pos}")
        report_lines.append(f"- train_neg: {neg}")
        report_lines.append(f"- imbalance_ratio (neg/pos): {ratio:.2f}")

        neg_pos_ratio = NEG_POS_RATIO if pos >= 200 else 10
        X_train = train_df[feature_cols].to_numpy()
        X_val = val_df[feature_cols].to_numpy()

        best_c = None
        best_ap = 0.0
        if y_val.sum() > 0 and y_train.sum() > 0:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            best_ap = -1.0
            for C in [0.01, 0.1, 1.0, 10.0]:
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                )
                model.fit(X_train_scaled, y_train)
                preds = model.predict_proba(X_val_scaled)[:, 1]
                ap = average_precision_score(y_val, preds)
                if ap > best_ap:
                    best_ap = ap
                    best_c = C

        coef_scores, perm_scores, rank_median_logit, topk_logit, topk_perm, _rank_lists = fit_logistic_models(
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
        cohen_scores = {}
        for feat in feature_cols:
            x_pos = train_df.loc[train_df[LABEL_COL] == class_name, feat].to_numpy()
            x_neg = train_df.loc[train_df[LABEL_COL] != class_name, feat].to_numpy()
            ks, wasser, cohen = compute_distribution_stats(x_pos, x_neg)
            ks_scores[feat] = ks
            wasser_scores[feat] = wasser
            cohen_scores[feat] = abs(cohen)

        rank_logit = rank_from_scores(coef_scores, higher_is_better=True)
        rank_logit_median = {f: int(rank_median_logit.get(f, rank_logit[f])) for f in feature_cols}
        rank_tree = rank_from_scores(tree_scores, higher_is_better=True)
        rank_perm = rank_from_scores(perm_scores, higher_is_better=True)
        rank_mi = rank_from_scores(mi_scores, higher_is_better=True)
        rank_ks = rank_from_scores(ks_scores, higher_is_better=True)
        rank_wasser = rank_from_scores(wasser_scores, higher_is_better=True)
        rank_cohen = rank_from_scores(cohen_scores, higher_is_better=True)

        fused = fused_rank(
            [rank_logit_median, rank_perm, rank_mi, rank_ks, rank_wasser],
            feature_cols,
        )
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
                "cohen_d": [cohen_scores[f] for f in feature_cols],
                "rank_logit": [rank_logit[f] for f in feature_cols],
                "rank_logit_median": [rank_logit_median[f] for f in feature_cols],
                "rank_tree": [rank_tree[f] for f in feature_cols],
                "rank_perm": [rank_perm[f] for f in feature_cols],
                "rank_mi": [rank_mi[f] for f in feature_cols],
                "rank_ks": [rank_ks[f] for f in feature_cols],
                "rank_wasser": [rank_wasser[f] for f in feature_cols],
                "rank_cohen": [rank_cohen[f] for f in feature_cols],
                "fused_rank": [fused[f] for f in feature_cols],
                "selection_freq_logit": [selection_freq_logit[f] for f in feature_cols],
                "selection_freq_tree": [selection_freq_tree[f] for f in feature_cols],
                "selection_freq_perm": [selection_freq_perm[f] for f in feature_cols],
            }
        ).sort_values("fused_rank")

        out_csv = ranks_dir / f"{class_name}_rankings.csv"
        ranking_df.to_csv(out_csv, index=False)
        outputs.append(out_csv)

        top_by_method = {
            "logit": ranking_df.sort_values("rank_logit_median").head(TOP_K)["feature"].tolist(),
            "tree": ranking_df.sort_values("rank_tree").head(TOP_K)["feature"].tolist(),
            "perm": ranking_df.sort_values("rank_perm").head(TOP_K)["feature"].tolist(),
            "mi": ranking_df.sort_values("rank_mi").head(TOP_K)["feature"].tolist(),
            "ks": ranking_df.sort_values("rank_ks").head(TOP_K)["feature"].tolist(),
            "wasser": ranking_df.sort_values("rank_wasser").head(TOP_K)["feature"].tolist(),
            "cohen": ranking_df.sort_values("rank_cohen").head(TOP_K)["feature"].tolist(),
        }
        fused_top = ranking_df.head(TOP_K)["feature"].tolist()
        report_lines.append(f"- top_{TOP_K}_by_method: {json.dumps(top_by_method)}")
        report_lines.append(f"- fused_top_{TOP_K}: {fused_top}")
        report_lines.append(
            f"- stability_topk_logit_mean: {np.mean(list(selection_freq_logit.values())):.3f}, "
            f"tree_mean: {np.mean(list(selection_freq_tree.values())):.3f}, "
            f"perm_mean: {np.mean(list(selection_freq_perm.values())):.3f}"
        )

        top_fused_df = ranking_df.head(10)
        ks_avg = float(top_fused_df["ks"].mean())
        wasser_avg = float(top_fused_df["wasserstein"].mean())
        separation_flag = "strong" if (ks_avg >= 0.2 and wasser_avg >= 0.2) else "weak"
        report_lines.append(f"- separation_heuristic: {separation_flag} (avg_ks={ks_avg:.3f}, avg_wasser={wasser_avg:.3f})")

        if best_c is not None:
            report_lines.append(f"- PR-AUC (logit tuned, C={best_c}): {best_ap:.4f}")
        else:
            report_lines.append("- PR-AUC: not computable (no positives in validation)")

        plot_paths = make_plots(plots_dir, class_name, ranking_df, train_df, LABEL_COL)
        outputs.extend(plot_paths)

        print(f"\n{class_name.upper()} top features by method:")
        for method, feats in top_by_method.items():
            print(f"  {method}: {feats[:10]}")
        print(f"  fused: {fused_top[:10]}")

        report_lines.append("")

    report_path = Path("reports/feature_discovery_1D_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    outputs.extend([merged_path, labeled_path, report_path])

    print("Artifacts created:")
    for p in outputs:
        print(f"- {p}")

    print("\nSummary:")
    print(f"Label counts: {label_counts}")
    for class_name in ["long", "short", "skip"]:
        ranking_df = pd.read_csv(ranks_dir / f"{class_name}_rankings.csv")
        top10 = ranking_df.head(10)["feature"].tolist()
        print(f"{class_name.upper()} top10: {top10}")


if __name__ == "__main__":
    main()
