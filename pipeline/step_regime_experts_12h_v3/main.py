from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


CONFIG = {
    "random_seed": 42,
    "feature_shift": 1,
    "fee_per_trade": 0.0005,
    "gmm_k_range": list(range(2, 9)),
    "gmm_cov_type": "full",
    "gmm_max_iter": 500,
    "hmm_k_range": list(range(2, 9)),
    "hmm_max_iter": 200,
    "hmm_cov_type": "diag",
    "min_cluster_samples": 300,
    "small_cluster_strategy": "global",  # "global" or "merge"
    "expanding_min_train_months": 2,
    "rolling_train_months": [6, 7],
    "rolling_test_months": 2,
    "rolling_step_months": 2,
    "label_candidates": [
        "candle_type",
        "Candle_type",
        "label",
        "target",
        "y",
        "signal",
        "class",
        "bucket4",
        "bucket",
        "trade_label",
    ],
    "timestamp_candidates": ["timestamp", "time", "date", "datetime"],
    "label_order": ["long", "short", "skip"],
    "anomaly_signals_path": "pipeline/anomaly_trading_12h_v3/outputs/oos_dedup_signals.csv",
    "anomaly_variant": "with_liq",
    "max_plot_points": 5000,
}


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"
OUTPUT_DIR = ROOT_DIR
REPORTS_DIR = OUTPUT_DIR / "reports"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"


@dataclass
class Fold:
    fold_id: int
    split_type: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_months: list[str]
    test_months: list[str]
    config: dict


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


def detect_timestamp_column(columns: list[str]) -> str | None:
    for candidate in CONFIG["timestamp_candidates"]:
        if candidate in columns:
            return candidate
    return None


def detect_label_column(columns: list[str]) -> str:
    for candidate in CONFIG["label_candidates"]:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"No label column found. Candidates: {CONFIG['label_candidates']}. "
        f"Available columns: {columns}"
    )


def load_data(path: Path) -> tuple[pd.DataFrame, str | None, str]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    timestamp_col = detect_timestamp_column(df.columns.tolist())
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    label_col = detect_label_column(df.columns.tolist())
    return df, timestamp_col, label_col


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
        df["spread_bps_rolling_mean"] = (
            df[spread_source].rolling(window=20, min_periods=20).mean()
        )
        df["spread_bps_rolling_std"] = (
            df[spread_source].rolling(window=20, min_periods=20).std()
        )
        engineered.extend(["spread_bps_rolling_mean", "spread_bps_rolling_std"])

    imbalance_source = None
    if "imbalance_last" in df.columns:
        imbalance_source = "imbalance_last"
    elif "imbalance_mean" in df.columns:
        imbalance_source = "imbalance_mean"
    if imbalance_source:
        df["imbalance_rolling_mean"] = (
            df[imbalance_source].rolling(window=20, min_periods=20).mean()
        )
        df["imbalance_rolling_std"] = (
            df[imbalance_source].rolling(window=20, min_periods=20).std()
        )
        engineered.extend(["imbalance_rolling_mean", "imbalance_rolling_std"])

    if {"atm_iv_7d", "atm_iv_1d"}.issubset(df.columns):
        df["iv_slope_atm"] = df["atm_iv_7d"] - df["atm_iv_1d"]
        engineered.append("iv_slope_atm")

    if {"rr25_7d", "rr25_1d"}.issubset(df.columns):
        df["rr_change_1d_7d"] = df["rr25_7d"] - df["rr25_1d"]
        engineered.append("rr_change_1d_7d")

    if {"fly25_7d", "fly25_1d"}.issubset(df.columns):
        df["fly_change_1d_7d"] = df["fly25_7d"] - df["fly25_1d"]
        engineered.append("fly_change_1d_7d")

    if "close" in df.columns:
        df["next_return"] = df["close"].shift(-1) / df["close"] - 1.0

    return df, engineered


def build_feature_sets(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()
    df["row_idx"] = np.arange(len(df))

    exclude = {label_col, "next_return", "row_idx"}
    for col in df.columns:
        if col == label_col:
            continue
        if "label" in col.lower() or "target" in col.lower():
            exclude.add(col)
    if timestamp_col:
        exclude.add(timestamp_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    model_features = [c for c in numeric_cols if c not in exclude]

    cluster_feature_candidates = [
        "log_return_12h",
        "range_pct",
        "body_pct",
        "realized_vol_rolling_20",
        "volume_z_rolling_20",
        "spread_bps_rolling_mean",
        "spread_bps_rolling_std",
        "imbalance_rolling_mean",
        "imbalance_rolling_std",
        "iv_slope_atm",
        "rr_change_1d_7d",
        "fly_change_1d_7d",
    ]
    cluster_features = [c for c in cluster_feature_candidates if c in df.columns]

    if not cluster_features:
        raise ValueError("No cluster features available after engineering.")

    all_features = sorted(set(model_features + cluster_features))
    df[all_features] = df[all_features].shift(CONFIG["feature_shift"])

    required = all_features + [label_col]
    if timestamp_col:
        required.append(timestamp_col)
    df = df.dropna(subset=required).reset_index(drop=True)

    return df, cluster_features, model_features



def make_splits_expanding(df: pd.DataFrame, timestamp_col: str) -> list[Fold]:
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    folds: list[Fold] = []
    min_train = CONFIG["expanding_min_train_months"]

    for idx in range(min_train, len(months)):
        train_months = months[:idx]
        test_months = [months[idx]]
        train_idx = df.index[df["month"].isin(train_months)].to_numpy()
        test_idx = df.index[df["month"].isin(test_months)].to_numpy()
        folds.append(
            Fold(
                fold_id=len(folds),
                split_type="expanding",
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_months,
                test_months=test_months,
                config={"min_train_months": min_train},
            )
        )
    return folds


def make_splits_rolling(df: pd.DataFrame, timestamp_col: str, train_months: int) -> list[Fold]:
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    test_months = CONFIG["rolling_test_months"]
    step = CONFIG["rolling_step_months"]

    folds: list[Fold] = []
    start = 0
    while start + train_months + test_months <= len(months):
        train_slice = months[start : start + train_months]
        test_slice = months[start + train_months : start + train_months + test_months]
        train_idx = df.index[df["month"].isin(train_slice)].to_numpy()
        test_idx = df.index[df["month"].isin(test_slice)].to_numpy()
        folds.append(
            Fold(
                fold_id=len(folds),
                split_type=f"rolling_{train_months}m",
                train_idx=train_idx,
                test_idx=test_idx,
                train_months=train_slice,
                test_months=test_slice,
                config={
                    "train_months": train_months,
                    "test_months": test_months,
                    "step_months": step,
                },
            )
        )
        start += step
    return folds


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
        return {cid: "mid" for cid in clusters}

    stats = df_tmp.groupby("cluster_id")[stat_col].mean().sort_values()
    clusters_sorted = stats.index.tolist()
    n_clusters = len(clusters_sorted)
    mapping: dict[int, str] = {}
    for idx, cid in enumerate(clusters_sorted):
        tercile = min(int(math.floor(3 * idx / max(n_clusters, 1))), 2)
        mapping[cid] = ["low", "mid", "high"][tercile]
    return mapping



def resolve_classifier() -> tuple[str, object]:
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=300,
            random_state=CONFIG["random_seed"],
        )
        return "lightgbm", model
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=CONFIG["random_seed"],
            verbose=False,
        )
        return "catboost", model
    except Exception:
        pass

    model = HistGradientBoostingClassifier(random_state=CONFIG["random_seed"])
    return "hist_gb", model


def build_model_pipeline() -> tuple[str, Pipeline]:
    name, clf = resolve_classifier()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    return name, pipeline


def train_models_global_and_experts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_train: np.ndarray,
    cluster_features_train: np.ndarray,
) -> tuple[Pipeline, dict[int, Pipeline], dict[int, int | None], dict[int, int], str]:
    model_name, global_model = build_model_pipeline()
    global_model.fit(X_train, y_train)

    experts: dict[int, Pipeline] = {}
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

        _, expert = build_model_pipeline()
        expert.fit(X_train[cluster_train == cid], y_train[cluster_train == cid])
        experts[cid] = expert
        cluster_to_model[cid] = cid

    return global_model, experts, cluster_to_model, cluster_sizes, model_name


def predict_with_experts(
    X_test: np.ndarray,
    cluster_test: np.ndarray,
    global_model: Pipeline,
    experts: dict[int, Pipeline],
    cluster_to_model: dict[int, int | None],
) -> np.ndarray:
    preds = np.empty(len(X_test), dtype=object)
    for cid in np.unique(cluster_test):
        idx = np.where(cluster_test == cid)[0]
        target_cid = cluster_to_model.get(cid)
        model = experts.get(target_cid) if target_cid is not None else None
        if model is None:
            model = global_model
        preds[idx] = model.predict(X_test[idx])
    return preds


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


def compute_trade_metrics(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    label_values: list,
    fee: float,
) -> dict:
    long_label, short_label, skip_label = infer_trade_label_mapping(label_values)
    if skip_label is None:
        return {
            "coverage": np.nan,
            "avg_trade_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
            "cvar_95": np.nan,
            "trade_count": 0,
        }

    trade_mask = preds != skip_label
    trade_count = int(trade_mask.sum())
    coverage = float(trade_mask.mean())

    if trade_count == 0:
        return {
            "coverage": coverage,
            "avg_trade_return": np.nan,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
            "cvar_95": np.nan,
            "trade_count": trade_count,
        }

    next_returns = df_test.loc[trade_mask, "next_return"].to_numpy()
    pred_trades = preds[trade_mask]
    trade_returns = np.zeros(len(next_returns))

    for i, label in enumerate(pred_trades):
        if label == long_label:
            trade_returns[i] = next_returns[i] - fee
        elif label == short_label:
            trade_returns[i] = -next_returns[i] - fee
        else:
            trade_returns[i] = 0.0

    avg_trade_return = float(np.mean(trade_returns))
    win_rate = float((trade_returns > 0).mean())

    cumulative = np.cumsum(trade_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan

    cvar_95 = float(
        np.mean(np.sort(trade_returns)[: max(1, int(0.05 * len(trade_returns)))])
    )

    return {
        "coverage": coverage,
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "cvar_95": cvar_95,
        "trade_count": trade_count,
    }



def evaluate_fold(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    label_values: list,
) -> tuple[dict, np.ndarray]:
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        labels=label_values,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        preds,
        labels=label_values,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, preds, labels=label_values)

    trade_metrics = compute_trade_metrics(
        df_test,
        preds,
        label_values,
        fee=CONFIG["fee_per_trade"],
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "coverage": trade_metrics["coverage"],
        "avg_trade_return": trade_metrics["avg_trade_return"],
        "win_rate": trade_metrics["win_rate"],
        "max_drawdown": trade_metrics["max_drawdown"],
        "cvar_95": trade_metrics["cvar_95"],
        "trade_count": trade_metrics["trade_count"],
    }

    for label in label_values:
        label_str = str(label)
        if label_str in report:
            metrics[f"{label_str}_precision"] = report[label_str]["precision"]
            metrics[f"{label_str}_recall"] = report[label_str]["recall"]
            metrics[f"{label_str}_f1"] = report[label_str]["f1-score"]

    return metrics, cm


def data_profile(df: pd.DataFrame, label_col: str, timestamp_col: str | None) -> str:
    lines = []
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {df.shape[1]}")
    if timestamp_col:
        lines.append(f"Timestamp column: {timestamp_col}")
        lines.append(f"Time span: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    lines.append(f"Label column: {label_col}")
    lines.append("\nLabel distribution:")
    label_counts = df[label_col].value_counts(dropna=False)
    for label, count in label_counts.items():
        lines.append(f"- {label}: {count}")
    missing = df.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        lines.append("\nMissing rate (non-zero):")
        for col, rate in missing.items():
            lines.append(f"- {col}: {rate:.4f}")
    return "\n".join(lines)


def load_anomaly_predictions(df: pd.DataFrame, timestamp_col: str | None) -> pd.Series | None:
    path = REPO_ROOT / CONFIG["anomaly_signals_path"]
    if not path.exists():
        return None

    usecols = ["timestamp", "row_idx", "direction", "adaptive_flag", "feature_variant"]
    available = pd.read_csv(path, nrows=1).columns.tolist()
    missing = [col for col in usecols if col not in available]
    if missing:
        return None

    signals = pd.read_csv(path, usecols=usecols)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")

    if "feature_variant" in signals.columns:
        signals = signals[signals["feature_variant"] == CONFIG["anomaly_variant"]]

    signals = signals.dropna(subset=["row_idx"]).copy()
    signals["row_idx"] = signals["row_idx"].astype(int)

    pred = pd.Series("skip", index=df.index)
    if "row_idx" in df.columns:
        join = df[["row_idx"]].merge(
            signals[["row_idx", "direction", "adaptive_flag"]],
            on="row_idx",
            how="left",
        )
        direction = join["direction"].fillna(0).astype(int)
        flag = join["adaptive_flag"].fillna(0).astype(int)
        pred.loc[flag == 1] = direction.replace({1: "long", -1: "short"})
        return pred

    if timestamp_col and timestamp_col in df.columns:
        join = df[[timestamp_col]].merge(
            signals[["timestamp", "direction", "adaptive_flag"]],
            left_on=timestamp_col,
            right_on="timestamp",
            how="left",
        )
        direction = join["direction"].fillna(0).astype(int)
        flag = join["adaptive_flag"].fillna(0).astype(int)
        pred.loc[flag == 1] = direction.replace({1: "long", -1: "short"})
        return pred

    return None


def plot_bic(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics["k"], metrics["bic"], marker="o", label="BIC")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC vs K")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_embedding(
    X: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    if len(X) > max_points:
        idx = np.random.choice(len(X), size=max_points, replace=False)
        X_plot = X[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X
        labels_plot = labels

    pca = PCA(n_components=2, random_state=CONFIG["random_seed"])
    coords = pca.fit_transform(X_plot)

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels_plot, s=8, cmap="tab20")
    ax.set_title("Cluster Embedding (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_clusters_over_time(df: pd.DataFrame, timestamp_col: str, output_path: Path) -> None:
    if timestamp_col not in df.columns:
        return
    sample = df
    if len(df) > CONFIG["max_plot_points"]:
        sample = df.sample(CONFIG["max_plot_points"], random_state=CONFIG["random_seed"])
    sample = sample.sort_values(timestamp_col)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(sample[timestamp_col], sample["cluster_id"], s=5, alpha=0.6)
    ax.set_title("Clusters Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cluster_feature_means(df: pd.DataFrame, cluster_features: list[str], output_path: Path) -> None:
    means = df.groupby("cluster_id")[cluster_features].mean().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_title("Cluster Feature Means")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    ax.set_xticks(range(len(cluster_features)))
    ax.set_xticklabels(cluster_features, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)



def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "coverage",
        "avg_trade_return",
        "win_rate",
        "max_drawdown",
        "cvar_95",
        "trade_count",
    ]
    rows = []
    for (split_type, approach), group in df.groupby(["split_type", "approach"]):
        for metric in metric_cols:
            if metric not in group:
                continue
            rows.append(
                {
                    "split_type": split_type,
                    "approach": approach,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std()),
                }
            )
    return pd.DataFrame(rows)


def build_cv_report(df: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}", ""]
    for (split_type, approach), group in df.groupby(["split_type", "approach"]):
        lines.append(f"## {split_type} - {approach}")
        lines.append(f"- folds: {len(group)}")
        lines.append(f"- accuracy: {group['accuracy'].mean():.4f}")
        lines.append(f"- macro_f1: {group['macro_f1'].mean():.4f}")
        lines.append(f"- coverage: {group['coverage'].mean():.4f}")
        lines.append(f"- avg_trade_return: {group['avg_trade_return'].mean():.6f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    np.random.seed(CONFIG["random_seed"])
    ensure_dir(REPORTS_DIR)
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(CONFUSION_DIR)

    df, timestamp_col, label_col = load_data(DATA_PATH)
    df, _ = engineer_features(df)
    df, cluster_features, model_features = build_feature_sets(df, label_col, timestamp_col)

    label_values = [label for label in CONFIG["label_order"] if label in df[label_col].unique()]
    if not label_values:
        label_values = sorted(df[label_col].unique().tolist())

    REPORTS_DIR.joinpath("data_profile.txt").write_text(
        data_profile(df, label_col, timestamp_col), encoding="utf-8"
    )

    cluster_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_cluster_full = cluster_preprocessor.fit_transform(df[cluster_features].values)
    gmm_selection_full = fit_cluster_model(X_cluster_full)
    gmm_model_full = gmm_selection_full.model

    hmm_selection_full = fit_hmm_model(X_cluster_full)

    cluster_labels_full, cluster_conf_full = assign_clusters(gmm_model_full, X_cluster_full)
    df_full_clusters = df.copy()
    df_full_clusters["cluster_id"] = cluster_labels_full
    df_full_clusters["cluster_confidence"] = cluster_conf_full

    regime_mapping_full = map_regimes_3(
        df_full_clusters,
        cluster_labels_full,
        primary_col="realized_vol_rolling_20",
        secondary_col="range_pct",
    )
    df_full_clusters["regime3"] = df_full_clusters["cluster_id"].map(regime_mapping_full)

    cluster_assignments_path = ARTIFACTS_DIR / "cluster_assignments.csv"
    cluster_cols = ["row_idx", "cluster_id", "cluster_confidence", "regime3"]
    if timestamp_col:
        cluster_cols.insert(1, timestamp_col)
    df_full_clusters[cluster_cols].to_csv(cluster_assignments_path, index=False)

    joblib.dump(cluster_preprocessor, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(gmm_model_full, ARTIFACTS_DIR / "cluster_model.joblib")

    plot_bic(gmm_selection_full.metrics, FIGURES_DIR / "bic_vs_k.png")
    plot_embedding(
        X_cluster_full,
        cluster_labels_full,
        FIGURES_DIR / "embedding_2d.png",
        CONFIG["max_plot_points"],
    )
    if timestamp_col:
        plot_clusters_over_time(
            df_full_clusters, timestamp_col, FIGURES_DIR / "clusters_over_time.png"
        )
    plot_cluster_feature_means(
        df_full_clusters,
        cluster_features,
        FIGURES_DIR / "cluster_feature_means.png",
    )

    clustering_report = []
    clustering_report.append("# Clustering Report")
    clustering_report.append("")
    clustering_report.append("## Feature Sets")
    clustering_report.append(f"- Cluster features: {', '.join(cluster_features)}")
    clustering_report.append(f"- Model features count: {len(model_features)}")
    clustering_report.append("")
    clustering_report.append("## GMM Selection (Full Data, no CV)")
    for _, row in gmm_selection_full.metrics.iterrows():
        clustering_report.append(
            f"- K={int(row['k'])}: BIC={row['bic']:.2f}, AIC={row['aic']:.2f}, "
            f"silhouette={row['silhouette']:.4f}"
        )
    clustering_report.append("")
    clustering_report.append(f"Chosen K (BIC): {gmm_selection_full.best_k}")
    if hmm_selection_full is None:
        clustering_report.append("\nHMM: hmmlearn not available, skipped.")
    else:
        clustering_report.append("\n## HMM Selection (Full Data, no CV)")
        for _, row in hmm_selection_full.metrics.iterrows():
            clustering_report.append(
                f"- K={int(row['k'])}: loglik={row['loglik']:.2f}, "
                f"BIC={row['bic']:.2f}, AIC={row['aic']:.2f}"
            )
        clustering_report.append("")
        clustering_report.append(f"Chosen K (BIC): {hmm_selection_full.best_k}")

    REPORTS_DIR.joinpath("clustering_report.md").write_text(
        "\n".join(clustering_report), encoding="utf-8"
    )

    folds_expanding = []
    if timestamp_col:
        folds_expanding = make_splits_expanding(df, timestamp_col)
    if not folds_expanding:
        warnings.warn("No expanding folds generated. Check timestamp coverage.")

    folds_rolling = []
    if timestamp_col:
        for train_months in CONFIG["rolling_train_months"]:
            folds_rolling.extend(make_splits_rolling(df, timestamp_col, train_months))
    if not folds_rolling:
        warnings.warn("No rolling folds generated. Check timestamp coverage.")

    anomaly_preds = None
    if timestamp_col:
        anomaly_preds = load_anomaly_predictions(df, timestamp_col)

    fold_rows = []
    for fold in folds_expanding + folds_rolling:
        df_train = df.iloc[fold.train_idx]
        df_test = df.iloc[fold.test_idx]

        y_train = df_train[label_col].to_numpy()
        y_test = df_test[label_col].to_numpy()
        if len(np.unique(y_train)) < 2:
            warnings.warn(f"Fold {fold.fold_id} skipped: only one class in training.")
            continue

        cluster_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_cluster_train = cluster_pipe.fit_transform(df_train[cluster_features].values)
        X_cluster_test = cluster_pipe.transform(df_test[cluster_features].values)

        gmm_selection = fit_cluster_model(X_cluster_train)
        cluster_train, cluster_conf_train = assign_clusters(
            gmm_selection.model, X_cluster_train
        )
        cluster_test, cluster_conf_test = assign_clusters(
            gmm_selection.model, X_cluster_test
        )

        regime_map = map_regimes_3(
            df_train,
            cluster_train,
            primary_col="realized_vol_rolling_20",
            secondary_col="range_pct",
        )
        regime_test = pd.Series(cluster_test).map(regime_map).fillna("mid").to_numpy()

        X_train = df_train[model_features].values
        X_test = df_test[model_features].values

        global_model, experts, cluster_to_model, cluster_sizes, model_name = (
            train_models_global_and_experts(
                X_train,
                y_train,
                cluster_train,
                X_cluster_train,
            )
        )

        preds_global = global_model.predict(X_test)
        metrics_global, cm_global = evaluate_fold(df_test, y_test, preds_global, label_values)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "split_type": fold.split_type,
                "approach": "global",
                "model": model_name,
                "k": gmm_selection.best_k,
                "train_months": ",".join(fold.train_months),
                "test_months": ",".join(fold.test_months),
                **metrics_global,
            }
        )
        cm_path = CONFUSION_DIR / f"confusion_{fold.split_type}_fold{fold.fold_id}_global.csv"
        pd.DataFrame(cm_global, index=label_values, columns=label_values).to_csv(cm_path)

        preds_expert = predict_with_experts(
            X_test,
            cluster_test,
            global_model,
            experts,
            cluster_to_model,
        )
        metrics_expert, cm_expert = evaluate_fold(df_test, y_test, preds_expert, label_values)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "split_type": fold.split_type,
                "approach": "regime_experts",
                "model": model_name,
                "k": gmm_selection.best_k,
                "train_months": ",".join(fold.train_months),
                "test_months": ",".join(fold.test_months),
                **metrics_expert,
            }
        )
        cm_path = CONFUSION_DIR / f"confusion_{fold.split_type}_fold{fold.fold_id}_experts.csv"
        pd.DataFrame(cm_expert, index=label_values, columns=label_values).to_csv(cm_path)

        if anomaly_preds is not None:
            preds_anomaly = anomaly_preds.iloc[fold.test_idx].to_numpy()
            metrics_anom, cm_anom = evaluate_fold(df_test, y_test, preds_anomaly, label_values)
            fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "split_type": fold.split_type,
                    "approach": "anomaly_baseline",
                    "model": "anomaly",
                    "k": gmm_selection.best_k,
                    "train_months": ",".join(fold.train_months),
                    "test_months": ",".join(fold.test_months),
                    **metrics_anom,
                }
            )
            cm_path = CONFUSION_DIR / f"confusion_{fold.split_type}_fold{fold.fold_id}_anomaly.csv"
            pd.DataFrame(cm_anom, index=label_values, columns=label_values).to_csv(cm_path)

        cluster_stats_path = ARTIFACTS_DIR / f"cluster_sizes_{fold.split_type}_fold{fold.fold_id}.json"
        cluster_stats_path.write_text(json.dumps(cluster_sizes, indent=2), encoding="utf-8")

        if timestamp_col:
            fold_cluster_assignments = df_test[[timestamp_col, "row_idx"]].copy()
        else:
            fold_cluster_assignments = df_test[["row_idx"]].copy()
        fold_cluster_assignments["cluster_id"] = cluster_test
        fold_cluster_assignments["cluster_confidence"] = cluster_conf_test
        fold_cluster_assignments["regime3"] = regime_test
        fold_cluster_assignments.to_csv(
            ARTIFACTS_DIR / f"cluster_assignments_{fold.split_type}_fold{fold.fold_id}.csv",
            index=False,
        )

    if not fold_rows:
        raise RuntimeError("No folds were evaluated. Check data and split configuration.")

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df_expanding = metrics_df[metrics_df["split_type"] == "expanding"]
    metrics_df_rolling = metrics_df[metrics_df["split_type"].str.startswith("rolling")]

    if not metrics_df_expanding.empty:
        metrics_df_expanding.to_csv(RESULTS_DIR / "fold_metrics_expanding.csv", index=False)
        REPORTS_DIR.joinpath("cv_report_expanding.md").write_text(
            build_cv_report(metrics_df_expanding, "Expanding CV Report"),
            encoding="utf-8",
        )
    if not metrics_df_rolling.empty:
        metrics_df_rolling.to_csv(RESULTS_DIR / "fold_metrics_rolling.csv", index=False)
        REPORTS_DIR.joinpath("cv_report_rolling.md").write_text(
            build_cv_report(metrics_df_rolling, "Rolling CV Report"),
            encoding="utf-8",
        )

    summary_df = summarize_cv(metrics_df)
    summary_df.to_csv(RESULTS_DIR / "overall_summary.csv", index=False)

    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
