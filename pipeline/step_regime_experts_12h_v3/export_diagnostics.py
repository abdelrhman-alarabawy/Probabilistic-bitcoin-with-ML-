from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


os.environ.setdefault("LIGHTGBM_VERBOSE", "-1")


CONFIG = {
    "random_seed": 42,
    "feature_shift": 1,
    "fee_per_trade": 0.0005,
    "data_path": "data/processed/12h_features_indicators_with_ohlcv.csv",
    "timestamp_col": "timestamp",
    "label_candidates": [
        "candle_type",
        "Candle_type",
        "label",
        "target",
        "y",
        "signal",
        "class",
        "bucket4",
        "trade_label",
    ],
    "label_order": ["long", "short", "skip"],
    "gmm_k_range": list(range(2, 9)),
    "gmm_cov_type": "full",
    "gmm_max_iter": 500,
    "hmm_k_range": list(range(2, 9)),
    "hmm_cov_type": "diag",
    "hmm_max_iter": 200,
    "min_cluster_samples": 300,
    "small_cluster_strategy": "global",  # "global" or "merge"
    "mode1_train_months": 18,
    "mode1_test_months": 6,
    "mode1_step_months": 3,
    "expanding_min_train_months": 2,
    "rolling_train_months": [6, 7],
    "rolling_test_months": 2,
    "rolling_step_months": 2,
    "use_robust_scaler": True,
    "max_plot_points": 5000,
    "stack_allow_regimes": ["low", "mid", "high"],
    "stack_conf_threshold": 0.6,
    "stack_weight_anomaly": 1.0,
    "stack_weight_model": 1.0,
    "anomaly_signal_candidates": ["anomaly_signal", "anomaly_sig", "anomaly_pred", "anomaly_decision"],
    "anomaly_score_candidates": ["anomaly_score", "score_robustz", "anomaly_score_raw"],
}


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
DATA_PATH = REPO_ROOT / CONFIG["data_path"]
OUTPUT_DIR = ROOT_DIR
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"


@dataclass
class Fold:
    fold_id: int
    split_type: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_months: list[str]
    test_months: list[str]


@dataclass
class ClusterSelection:
    model: GaussianMixture
    metrics: pd.DataFrame
    best_k: int


@dataclass
class HMMSelection:
    model: GaussianHMM
    metrics: pd.DataFrame
    best_k: int



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_label_column(columns: list[str]) -> str:
    for candidate in CONFIG["label_candidates"]:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"No label column found. Candidates: {CONFIG['label_candidates']}. "
        f"Available columns: {columns}"
    )


def detect_anomaly_columns(columns: list[str], label_col: str) -> tuple[str | None, str | None]:
    signal_col = None
    for candidate in CONFIG["anomaly_signal_candidates"]:
        if candidate in columns and candidate != label_col:
            signal_col = candidate
            break
    score_col = None
    for candidate in CONFIG["anomaly_score_candidates"]:
        if candidate in columns and candidate != label_col:
            score_col = candidate
            break
    return signal_col, score_col


def load_data(path: Path) -> tuple[pd.DataFrame, str, str, str | None, str | None]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if CONFIG["timestamp_col"] not in df.columns:
        raise ValueError(f"Missing timestamp column: {CONFIG['timestamp_col']}")
    df[CONFIG["timestamp_col"]] = pd.to_datetime(
        df[CONFIG["timestamp_col"]], utc=True, errors="coerce"
    )
    df = df.sort_values(CONFIG["timestamp_col"]).reset_index(drop=True)
    label_col = detect_label_column(df.columns.tolist())
    anomaly_signal_col, anomaly_score_col = detect_anomaly_columns(df.columns.tolist(), label_col)
    return df, CONFIG["timestamp_col"], label_col, anomaly_signal_col, anomaly_score_col


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    engineered = []

    if {"close", "open", "high", "low"}.issubset(df.columns):
        df["log_return_12h"] = np.log(df["close"] / df["close"].shift(1))
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["body_pct"] = (df["close"] - df["open"]).abs() / df["close"]
        df["realized_vol_rolling_20"] = (
            df["log_return_12h"].rolling(window=20, min_periods=20).std()
        )
        engineered.extend(
            [
                "log_return_12h",
                "range_pct",
                "body_pct",
                "realized_vol_rolling_20",
            ]
        )

    if "volume" in df.columns:
        df["volume_z_rolling_20"] = rolling_zscore(df["volume"], window=20)
        engineered.append("volume_z_rolling_20")

    spread_source = None
    if "spread_bps_last" in df.columns:
        spread_source = "spread_bps_last"
    elif "spread_mean" in df.columns:
        spread_source = "spread_mean"
    if spread_source:
        df["spread_roll_mean"] = (
            df[spread_source].rolling(window=20, min_periods=20).mean()
        )
        df["spread_roll_std"] = (
            df[spread_source].rolling(window=20, min_periods=20).std()
        )
        engineered.extend(["spread_roll_mean", "spread_roll_std"])

    imbalance_source = None
    if "imbalance_last" in df.columns:
        imbalance_source = "imbalance_last"
    elif "imbalance_mean" in df.columns:
        imbalance_source = "imbalance_mean"
    if imbalance_source:
        df["imbalance_roll_mean"] = (
            df[imbalance_source].rolling(window=20, min_periods=20).mean()
        )
        df["imbalance_roll_std"] = (
            df[imbalance_source].rolling(window=20, min_periods=20).std()
        )
        engineered.extend(["imbalance_roll_mean", "imbalance_roll_std"])

    if {"atm_iv_7d", "atm_iv_1d"}.issubset(df.columns):
        df["iv_slope_atm"] = df["atm_iv_7d"] - df["atm_iv_1d"]
        engineered.append("iv_slope_atm")

    if {"rr25_7d", "rr25_1d"}.issubset(df.columns):
        df["rr_slope"] = df["rr25_7d"] - df["rr25_1d"]
        engineered.append("rr_slope")

    if {"fly25_7d", "fly25_1d"}.issubset(df.columns):
        df["fly_slope"] = df["fly25_7d"] - df["fly25_1d"]
        engineered.append("fly_slope")

    if "close" in df.columns:
        df["forward_return_1"] = df["close"].shift(-1) / df["close"] - 1.0

    return df, engineered


def build_feature_matrices(
    df: pd.DataFrame, label_col: str, timestamp_col: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()
    df["row_idx"] = np.arange(len(df))

    exclude = {label_col, "row_idx", "forward_return_1"}
    for col in df.columns:
        col_lower = col.lower()
        if col == label_col:
            continue
        if "label" in col_lower or "target" in col_lower:
            exclude.add(col)
    exclude.add(timestamp_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    model_features = [c for c in numeric_cols if c not in exclude]

    cluster_candidates = [
        "log_return_12h",
        "range_pct",
        "body_pct",
        "realized_vol_rolling_20",
        "volume_z_rolling_20",
        "spread_roll_mean",
        "spread_roll_std",
        "imbalance_roll_mean",
        "imbalance_roll_std",
        "iv_slope_atm",
        "rr_slope",
        "fly_slope",
    ]
    cluster_features = [c for c in cluster_candidates if c in df.columns]

    if not cluster_features:
        raise ValueError("No cluster features available after engineering.")

    all_features = sorted(set(model_features + cluster_features))
    df[all_features] = df[all_features].shift(CONFIG["feature_shift"])

    required = all_features + [label_col, timestamp_col]
    df = df.dropna(subset=required).reset_index(drop=True)

    return df, cluster_features, model_features


def make_splits_mode1_walkforward(df: pd.DataFrame, timestamp_col: str) -> list[Fold]:
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    folds: list[Fold] = []

    train_m = CONFIG["mode1_train_months"]
    test_m = CONFIG["mode1_test_months"]
    step_m = CONFIG["mode1_step_months"]

    start = 0
    while start + train_m + test_m <= len(months):
        train_slice = months[start : start + train_m]
        test_slice = months[start + train_m : start + train_m + test_m]
        train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
        test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
        folds.append(
            Fold(
                fold_id=len(folds),
                split_type="mode1_walkforward",
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_slice,
                test_months=test_slice,
            )
        )
        start += step_m
    return folds


def make_splits_mode2_expanding(df: pd.DataFrame, timestamp_col: str) -> list[Fold]:
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    folds: list[Fold] = []
    min_train = CONFIG["expanding_min_train_months"]

    for idx in range(min_train, len(months)):
        train_slice = months[:idx]
        test_slice = [months[idx]]
        train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
        test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
        folds.append(
            Fold(
                fold_id=len(folds),
                split_type="mode2_expanding",
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_slice,
                test_months=test_slice,
            )
        )
    return folds


def make_splits_mode3_rolling(
    df: pd.DataFrame, timestamp_col: str, train_months: int
) -> list[Fold]:
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    test_m = CONFIG["rolling_test_months"]
    step_m = CONFIG["rolling_step_months"]

    folds: list[Fold] = []
    start = 0
    while start + train_months + test_m <= len(months):
        train_slice = months[start : start + train_months]
        test_slice = months[start + train_months : start + train_months + test_m]
        train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
        test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
        folds.append(
            Fold(
                fold_id=len(folds),
                split_type=f"mode3_rolling_W{train_months}",
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_slice,
                test_months=test_slice,
            )
        )
        start += step_m
    return folds



def build_preprocessor() -> Pipeline:
    scaler = RobustScaler() if CONFIG["use_robust_scaler"] else StandardScaler()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )


def fit_cluster_model(X_train: np.ndarray) -> ClusterSelection:
    metrics = []
    best_model = None
    best_bic = np.inf
    best_k = None

    for k in CONFIG["gmm_k_range"]:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=CONFIG["gmm_cov_type"],
            max_iter=CONFIG["gmm_max_iter"],
            random_state=CONFIG["random_seed"],
        )
        gmm.fit(X_train)
        bic = gmm.bic(X_train)
        aic = gmm.aic(X_train)
        labels = gmm.predict(X_train)
        silhouette = np.nan
        if len(np.unique(labels)) > 1 and len(X_train) > k:
            silhouette = silhouette_score(X_train, labels)
        metrics.append({"k": k, "bic": bic, "aic": aic, "silhouette": silhouette})
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_k = k

    if best_model is None or best_k is None:
        raise ValueError("GMM training failed to produce a model.")

    metrics_df = pd.DataFrame(metrics)
    return ClusterSelection(model=best_model, metrics=metrics_df, best_k=best_k)


def hmm_param_count(n_states: int, n_features: int) -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * n_features
    return startprob + transmat + means + covars


def fit_hmm_model(X_train: np.ndarray) -> HMMSelection | None:
    if not HMM_AVAILABLE:
        return None

    metrics = []
    best_model = None
    best_bic = np.inf
    best_k = None

    for k in CONFIG["hmm_k_range"]:
        hmm = GaussianHMM(
            n_components=k,
            covariance_type=CONFIG["hmm_cov_type"],
            n_iter=CONFIG["hmm_max_iter"],
            random_state=CONFIG["random_seed"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(X_train)
        loglik = hmm.score(X_train)
        n_params = hmm_param_count(k, X_train.shape[1])
        bic = -2.0 * loglik + n_params * math.log(len(X_train))
        aic = -2.0 * loglik + 2.0 * n_params
        metrics.append({"k": k, "loglik": loglik, "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best_model = hmm
            best_k = k

    if best_model is None or best_k is None:
        return None

    metrics_df = pd.DataFrame(metrics)
    return HMMSelection(model=best_model, metrics=metrics_df, best_k=best_k)


def assign_clusters(model: GaussianMixture, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(X)
    labels = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return labels, confidence


def map_regimes_3(
    df_train: pd.DataFrame,
    cluster_ids: np.ndarray,
    primary_col: str,
    secondary_col: str,
) -> dict[int, str]:
    df_tmp = df_train.copy()
    df_tmp["cluster_id"] = cluster_ids

    stat_col = primary_col if primary_col in df_tmp.columns else secondary_col
    if stat_col not in df_tmp.columns:
        clusters = sorted(df_tmp["cluster_id"].unique())
        return {int(cid): "mid" for cid in clusters}

    stats = df_tmp.groupby("cluster_id")[stat_col].mean().sort_values()
    clusters_sorted = stats.index.tolist()
    n_clusters = len(clusters_sorted)
    mapping: dict[int, str] = {}
    for idx, cid in enumerate(clusters_sorted):
        tercile = min(int(math.floor(3 * idx / max(n_clusters, 1))), 2)
        mapping[int(cid)] = ["low", "mid", "high"][tercile]
    return mapping


def compute_class_weights(y_train: np.ndarray) -> tuple[np.ndarray, dict]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {cls: float(weight) for cls, weight in zip(classes, weights)}
    return classes, weight_dict


def build_classifier(classes: np.ndarray, class_weight: dict) -> tuple[str, object]:
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=300,
            random_state=CONFIG["random_seed"],
            class_weight=class_weight,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )
        return "lightgbm", model
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        class_weights_list = [class_weight[c] for c in classes]
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=CONFIG["random_seed"],
            class_weights=class_weights_list,
            verbose=False,
        )
        return "catboost", model
    except Exception:
        pass

    model = HistGradientBoostingClassifier(random_state=CONFIG["random_seed"])
    if "class_weight" in model.get_params():
        model.set_params(class_weight=class_weight)
        return "hist_gb", model

    model = RandomForestClassifier(random_state=CONFIG["random_seed"], n_estimators=300)
    if "class_weight" in model.get_params():
        model.set_params(class_weight=class_weight)
    return "random_forest", model


def train_global_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[object, str]:
    classes, weight_dict = compute_class_weights(y_train)
    model_name, model = build_classifier(classes, weight_dict)
    model.fit(X_train, y_train)
    return model, model_name


def train_expert_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_train: np.ndarray,
    cluster_features_train: np.ndarray,
) -> tuple[dict[int, object], dict[int, int | None], dict[int, int]]:
    experts: dict[int, object] = {}
    cluster_sizes = {
        int(cid): int((cluster_train == cid).sum()) for cid in np.unique(cluster_train)
    }

    large_clusters = [
        cid for cid, size in cluster_sizes.items() if size >= CONFIG["min_cluster_samples"]
    ]

    centroids = {}
    if CONFIG["small_cluster_strategy"] == "merge" and large_clusters:
        for cid in large_clusters:
            centroids[cid] = cluster_features_train[cluster_train == cid].mean(axis=0)

    cluster_to_model: dict[int, int | None] = {}
    for cid in cluster_sizes.keys():
        if cluster_sizes[cid] < CONFIG["min_cluster_samples"]:
            if CONFIG["small_cluster_strategy"] == "merge" and large_clusters:
                distances = {
                    large_cid: np.linalg.norm(
                        centroids[large_cid]
                        - cluster_features_train[cluster_train == cid].mean(axis=0)
                    )
                    for large_cid in large_clusters
                }
                nearest = min(distances, key=distances.get)
                cluster_to_model[cid] = nearest
            else:
                cluster_to_model[cid] = None
            continue

        y_cluster = y_train[cluster_train == cid]
        if len(np.unique(y_cluster)) < 2:
            cluster_to_model[cid] = None
            continue

        classes, weight_dict = compute_class_weights(y_cluster)
        _, model = build_classifier(classes, weight_dict)
        model.fit(X_train[cluster_train == cid], y_cluster)
        experts[cid] = model
        cluster_to_model[cid] = cid

    return experts, cluster_to_model, cluster_sizes


def predict_proba_aligned(model: object, X: np.ndarray, label_values: list) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.full((len(X), len(label_values)), np.nan)

    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None:
        return np.full((len(X), len(label_values)), np.nan)

    aligned = np.full((len(X), len(label_values)), np.nan)
    for idx, label in enumerate(label_values):
        matches = np.where(classes == label)[0]
        if len(matches) == 0:
            continue
        aligned[:, idx] = proba[:, matches[0]]
    return aligned


def predict_with_routing(
    X_test: np.ndarray,
    cluster_test: np.ndarray,
    global_model: object,
    experts: dict[int, object],
    cluster_to_model: dict[int, int | None],
    label_values: list,
) -> tuple[np.ndarray, np.ndarray]:
    preds = np.empty(len(X_test), dtype=object)
    probas = np.full((len(X_test), len(label_values)), np.nan)
    for cid in np.unique(cluster_test):
        idx = np.where(cluster_test == cid)[0]
        target_cid = cluster_to_model.get(int(cid))
        model = experts.get(target_cid) if target_cid is not None else None
        if model is None:
            model = global_model
        preds[idx] = model.predict(X_test[idx])
        probas[idx] = predict_proba_aligned(model, X_test[idx], label_values)
    return preds, probas



def resolve_label_order(y: np.ndarray) -> list:
    labels = [label for label in CONFIG["label_order"] if label in y]
    if labels:
        return labels
    return sorted(np.unique(y).tolist())


def infer_trade_label_mapping(labels: list) -> tuple[str | None, str | None, str | None]:
    lower = {str(label).lower(): label for label in labels}
    long_label = None
    short_label = None
    skip_label = None

    for key in lower:
        if key == "long":
            long_label = lower[key]
        if key == "short":
            short_label = lower[key]
        if key in {"skip", "flat", "none"}:
            skip_label = lower[key]

    numeric = set()
    try:
        numeric = set([float(label) for label in labels])
    except Exception:
        numeric = set()

    if long_label is None and short_label is None and skip_label is None:
        if numeric.issubset({-1.0, 0.0, 1.0}):
            long_label = 1
            short_label = -1
            skip_label = 0

    return long_label, short_label, skip_label


def proba_column_name(label: object) -> str:
    label_str = str(label).strip().lower().replace(" ", "_")
    if label_str in {"long", "short", "skip"}:
        return f"proba_{label_str}"
    return f"proba_{label_str}"


def compute_pnl_proxy(preds: np.ndarray, forward_returns: np.ndarray, label_values: list) -> np.ndarray:
    long_label, short_label, skip_label = infer_trade_label_mapping(label_values)
    pnl = np.zeros(len(preds))
    for i, label in enumerate(preds):
        if not np.isfinite(forward_returns[i]):
            pnl[i] = np.nan
            continue
        if long_label is not None and label == long_label:
            pnl[i] = forward_returns[i] - CONFIG["fee_per_trade"]
        elif short_label is not None and label == short_label:
            pnl[i] = -forward_returns[i] - CONFIG["fee_per_trade"]
        elif skip_label is not None and label == skip_label:
            pnl[i] = 0.0
        else:
            pnl[i] = 0.0
    return pnl


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_values: list,
) -> tuple[dict, np.ndarray]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_values,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_values,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_values)

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

    for label in label_values:
        label_str = str(label)
        if label_str in report:
            metrics[f"{label_str}_precision"] = report[label_str]["precision"]
            metrics[f"{label_str}_recall"] = report[label_str]["recall"]
            metrics[f"{label_str}_f1"] = report[label_str]["f1-score"]

    return metrics, cm


def compute_drawdown(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return np.nan
    cumulative = np.cumsum(trade_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(drawdown.min())


def compute_profit_factor(trade_returns: np.ndarray) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / abs(losses))


def compute_group_metrics(
    df_group: pd.DataFrame,
    label_values: list,
    months_in_test: int,
) -> dict:
    support = len(df_group)
    y_true = df_group["y_true"].to_numpy()
    y_pred = df_group["y_pred"].to_numpy()

    metrics, _ = evaluate_predictions(y_true, y_pred, label_values)

    long_label, short_label, skip_label = infer_trade_label_mapping(label_values)
    if skip_label is None:
        trade_mask = np.ones(len(df_group), dtype=bool)
    else:
        trade_mask = y_pred != skip_label

    trade_count = int(trade_mask.sum())
    coverage = float(trade_count / support) if support else np.nan

    trade_returns = df_group.loc[trade_mask, "pnl_proxy"].dropna().to_numpy()
    avg_pnl = float(np.mean(trade_returns)) if len(trade_returns) else np.nan
    median_pnl = float(np.median(trade_returns)) if len(trade_returns) else np.nan
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan
    profit_factor = compute_profit_factor(trade_returns) if len(trade_returns) else np.nan
    max_dd = compute_drawdown(trade_returns)
    cvar95 = (
        float(np.percentile(trade_returns, 5)) if len(trade_returns) else np.nan
    )

    prfs = precision_recall_fscore_support(
        y_true, y_pred, labels=label_values, zero_division=0
    )
    label_metrics = {label: idx for idx, label in enumerate(label_values)}

    def get_label_metric(target_label: object, metric_idx: int) -> float:
        idx = label_metrics.get(target_label)
        if idx is None:
            return np.nan
        return float(prfs[metric_idx][idx])

    return {
        "support": support,
        "trade_count": trade_count,
        "coverage": coverage,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision_long": get_label_metric(long_label, 0),
        "recall_long": get_label_metric(long_label, 1),
        "f1_long": get_label_metric(long_label, 2),
        "precision_short": get_label_metric(short_label, 0),
        "recall_short": get_label_metric(short_label, 1),
        "f1_short": get_label_metric(short_label, 2),
        "avg_pnl_per_trade": avg_pnl,
        "median_pnl_per_trade": median_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "cvar95": cvar95,
        "trades_per_month": float(trade_count / months_in_test) if months_in_test else np.nan,
    }


def aggregate_group_summaries(
    df: pd.DataFrame,
    group_col: str,
    split_type: str,
) -> pd.DataFrame:
    metric_cols = [
        "support",
        "trade_count",
        "coverage",
        "accuracy",
        "macro_f1",
        "precision_long",
        "recall_long",
        "f1_long",
        "precision_short",
        "recall_short",
        "f1_short",
        "avg_pnl_per_trade",
        "median_pnl_per_trade",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "cvar95",
        "trades_per_month",
    ]
    rows = []
    group_keys = ["approach", group_col]
    for (approach, group_val), group in df.groupby(group_keys):
        row = {
            "split_type": split_type,
            "approach": approach,
            group_col: group_val,
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)



def tpsl_suggestions(df: pd.DataFrame, label_values: list) -> pd.DataFrame:
    long_label, short_label, _ = infer_trade_label_mapping(label_values)
    rows = []

    for approach, df_app in df.groupby("approach"):
        for regime, df_reg in df_app.groupby("regime3"):
            for direction, label in [("long", long_label), ("short", short_label)]:
                if label is None:
                    continue
                df_trades = df_reg[df_reg["y_pred"] == label]
                if df_trades.empty:
                    continue
                if direction == "long":
                    signed_returns = df_trades["forward_return_1"].dropna().to_numpy()
                else:
                    signed_returns = -df_trades["forward_return_1"].dropna().to_numpy()
                if len(signed_returns) == 0:
                    continue

                percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
                pct_values = np.percentile(signed_returns, percentiles)
                pct_map = {f"p{p:02d}": float(v) for p, v in zip(percentiles, pct_values)}

                vol_mean = (
                    float(df_reg["realized_vol_rolling_20"].mean())
                    if "realized_vol_rolling_20" in df_reg
                    else np.nan
                )
                vol_std = (
                    float(df_reg["realized_vol_rolling_20"].std())
                    if "realized_vol_rolling_20" in df_reg
                    else np.nan
                )

                tp_candidates = [pct_map["p60"], pct_map["p70"], pct_map["p80"]]
                tp_candidates = [v for v in tp_candidates if v > CONFIG["fee_per_trade"]]
                sl_candidates = [abs(pct_map["p20"]), abs(pct_map["p10"])]
                sl_candidates = [v for v in sl_candidates if v > CONFIG["fee_per_trade"]]

                rows.append(
                    {
                        "approach": approach,
                        "regime3": regime,
                        "direction": direction,
                        **pct_map,
                        "realized_vol_mean": vol_mean,
                        "realized_vol_std": vol_std,
                        "tp_candidates": json.dumps(tp_candidates),
                        "sl_candidates": json.dumps(sl_candidates),
                    }
                )
    return pd.DataFrame(rows)


def add_stacking_columns(
    df: pd.DataFrame,
    label_values: list,
    anomaly_signal_col: str | None,
    anomaly_score_col: str | None,
) -> pd.DataFrame:
    df = df.copy()
    long_label, short_label, skip_label = infer_trade_label_mapping(label_values)
    skip_value = skip_label if skip_label is not None else "skip"

    if anomaly_signal_col and anomaly_signal_col in df.columns:
        df["anomaly_signal"] = df[anomaly_signal_col]
    else:
        df["anomaly_signal"] = np.nan

    allow_regimes = set(CONFIG["stack_allow_regimes"])
    conf_thr = CONFIG["stack_conf_threshold"]

    def gate_signal(row: pd.Series) -> object:
        if pd.isna(row["anomaly_signal"]):
            return skip_value
        if row["regime3"] in allow_regimes and row["cluster_confidence"] >= conf_thr:
            return row["anomaly_signal"]
        return skip_value

    df["final_signal_stack1_gate"] = df.apply(gate_signal, axis=1)

    proba_long = df.get("proba_long")
    proba_short = df.get("proba_short")
    if proba_long is None or proba_short is None:
        df["final_signal_stack2_score"] = np.nan
    else:
        model_score = proba_long - proba_short
        if anomaly_score_col and anomaly_score_col in df.columns:
            anomaly_score = df[anomaly_score_col]
            df["final_signal_stack2_score"] = (
                CONFIG["stack_weight_anomaly"] * anomaly_score
                + CONFIG["stack_weight_model"] * model_score
            )
        else:
            df["final_signal_stack2_score"] = np.nan

    return df


def run_fold(
    df: pd.DataFrame,
    fold: Fold,
    cluster_features: list[str],
    model_features: list[str],
    label_col: str,
    label_values: list,
) -> tuple[list[pd.DataFrame], list[dict], list[dict]]:
    df_train = df.iloc[fold.train_idx]
    df_test = df.iloc[fold.test_idx]

    y_train = df_train[label_col].to_numpy()
    y_test = df_test[label_col].to_numpy()
    if len(np.unique(y_train)) < 2:
        warnings.warn(f"Fold {fold.fold_id} skipped: only one class in training.")
        return [], [], []

    cluster_preprocessor = build_preprocessor()
    X_cluster_train = cluster_preprocessor.fit_transform(df_train[cluster_features].values)
    X_cluster_test = cluster_preprocessor.transform(df_test[cluster_features].values)

    cluster_selection = fit_cluster_model(X_cluster_train)
    cluster_train, _ = assign_clusters(cluster_selection.model, X_cluster_train)
    cluster_test, cluster_conf_test = assign_clusters(cluster_selection.model, X_cluster_test)

    regime_map = map_regimes_3(
        df_train,
        cluster_train,
        primary_col="realized_vol_rolling_20",
        secondary_col="range_pct",
    )
    regime_test = pd.Series(cluster_test).map(regime_map).fillna("mid").to_numpy()

    model_preprocessor = build_preprocessor()
    X_train = model_preprocessor.fit_transform(df_train[model_features].values)
    X_test = model_preprocessor.transform(df_test[model_features].values)

    global_model, model_name = train_global_model(X_train, y_train)
    experts, cluster_to_model, cluster_sizes = train_expert_models(
        X_train,
        y_train,
        cluster_train,
        X_cluster_train,
    )

    small_clusters = [cid for cid, size in cluster_sizes.items() if size < CONFIG["min_cluster_samples"]]
    if small_clusters:
        print(
            f"{fold.split_type} fold {fold.fold_id} small clusters {small_clusters} -> "
            f"{CONFIG['small_cluster_strategy']}"
        )

    preds_global = global_model.predict(X_test)
    probas_global = predict_proba_aligned(global_model, X_test, label_values)

    preds_expert, probas_expert = predict_with_routing(
        X_test,
        cluster_test,
        global_model,
        experts,
        cluster_to_model,
        label_values,
    )

    rows = []
    group_cluster_rows = []
    group_regime_rows = []

    for approach, preds, probas in [
        ("global", preds_global, probas_global),
        ("experts", preds_expert, probas_expert),
    ]:
        pnl_proxy = compute_pnl_proxy(
            preds,
            df_test["forward_return_1"].to_numpy(),
            label_values,
        )

        pred_df = pd.DataFrame(
            {
                "timestamp": df_test[CONFIG["timestamp_col"]].to_numpy(),
                "fold_id": fold.fold_id,
                "split_type": fold.split_type,
                "approach": approach,
                "y_true": y_test,
                "y_pred": preds,
                "cluster_id": cluster_test,
                "cluster_confidence": cluster_conf_test,
                "regime3": regime_test,
                "close": df_test["close"].to_numpy(),
                "forward_return_1": df_test["forward_return_1"].to_numpy(),
                "pnl_proxy": pnl_proxy,
                "realized_vol_rolling_20": df_test.get("realized_vol_rolling_20"),
            }
        )

        for idx, label in enumerate(label_values):
            pred_df[proba_column_name(label)] = probas[:, idx]

        rows.append(pred_df)

        months_in_test = max(1, len(set(fold.test_months)))
        for group_col in ["cluster_id", "regime3"]:
            grouped = []
            for group_val, df_group in pred_df.groupby(group_col):
                metrics = compute_group_metrics(df_group, label_values, months_in_test)
                grouped.append(
                    {
                        "fold_id": fold.fold_id,
                        "split_type": fold.split_type,
                        "approach": approach,
                        group_col: group_val,
                        **metrics,
                    }
                )
            if group_col == "cluster_id":
                group_cluster_rows.extend(grouped)
            else:
                group_regime_rows.extend(grouped)

    return rows, group_cluster_rows, group_regime_rows


def main() -> None:
    np.random.seed(CONFIG["random_seed"])
    ensure_dir(REPORTS_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(CONFUSION_DIR)

    df, timestamp_col, label_col, anomaly_signal_col, anomaly_score_col = load_data(DATA_PATH)
    df, _ = engineer_features(df)
    df, cluster_features, model_features = build_feature_matrices(df, label_col, timestamp_col)
    label_values = resolve_label_order(df[label_col].to_numpy())

    folds_mode1 = make_splits_mode1_walkforward(df, timestamp_col)
    folds_mode2 = make_splits_mode2_expanding(df, timestamp_col)
    folds_mode3_w6 = make_splits_mode3_rolling(df, timestamp_col, 6)
    folds_mode3_w7 = make_splits_mode3_rolling(df, timestamp_col, 7)

    split_map = {
        "mode1_walkforward": folds_mode1,
        "mode2_expanding": folds_mode2,
        "mode3_rolling_W6": folds_mode3_w6,
        "mode3_rolling_W7": folds_mode3_w7,
    }

    report_lines = []
    report_lines.append("# Diagnostics Export Report")
    report_lines.append("")
    report_lines.append(f"Label column: {label_col}")
    report_lines.append(f"Anomaly signal column: {anomaly_signal_col}")
    report_lines.append(f"Anomaly score column: {anomaly_score_col}")
    report_lines.append("")

    for split_type, folds in split_map.items():
        pred_rows = []
        cluster_rows = []
        regime_rows = []
        for fold in folds:
            rows, group_cluster_rows, group_regime_rows = run_fold(
                df,
                fold,
                cluster_features,
                model_features,
                label_col,
                label_values,
            )
            pred_rows.extend(rows)
            cluster_rows.extend(group_cluster_rows)
            regime_rows.extend(group_regime_rows)

        if not pred_rows:
            report_lines.append(f"- {split_type}: no folds evaluated")
            continue

        preds_df = pd.concat(pred_rows, ignore_index=True)
        preds_path = RESULTS_DIR / f"preds_per_row_{split_type}.csv"
        preds_df.to_csv(preds_path, index=False)

        cluster_df = pd.DataFrame(cluster_rows)
        regime_df = pd.DataFrame(regime_rows)

        cluster_summary = aggregate_group_summaries(cluster_df, "cluster_id", split_type)
        regime_summary = aggregate_group_summaries(regime_df, "regime3", split_type)

        cluster_path = RESULTS_DIR / f"per_cluster_summary_{split_type}.csv"
        regime_path = RESULTS_DIR / f"per_regime3_summary_{split_type}.csv"
        cluster_summary.to_csv(cluster_path, index=False)
        regime_summary.to_csv(regime_path, index=False)

        tpsl_df = tpsl_suggestions(preds_df, label_values)
        tpsl_path = RESULTS_DIR / f"tpsl_suggestions_by_regime3_{split_type}.csv"
        tpsl_df.to_csv(tpsl_path, index=False)

        stack_df = add_stacking_columns(
            preds_df,
            label_values,
            anomaly_signal_col,
            anomaly_score_col,
        )
        stacking_path = RESULTS_DIR / f"stacking_rows_{split_type}.csv"
        stack_df.to_csv(stacking_path, index=False)

        report_lines.append(f"- {split_type}: rows={len(preds_df)}")
        report_lines.append(f"  - preds: {preds_path}")
        report_lines.append(f"  - per_cluster: {cluster_path}")
        report_lines.append(f"  - per_regime3: {regime_path}")
        report_lines.append(f"  - tpsl: {tpsl_path}")
        report_lines.append(f"  - stacking: {stacking_path}")

    report_path = REPORTS_DIR / "diagnostics_export_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Diagnostics export complete.")
    print("Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
