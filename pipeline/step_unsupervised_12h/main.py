import json
import logging
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, RobustScaler


CONFIG = {
    "DATA_PATH": "data/processed/12h_features_indicators_with_ohlcv.csv",
    "OUTPUT_DIR": "pipeline/step_unsupervised_12h",
    "TIMESTAMP_COL": "timestamp",
    "LABEL_CANDIDATES": [
        "candle_type",
        "Candle_type",
        "label",
        "target",
        "y",
        "signal",
        "class",
    ],
    "SEED": 42,
    "CLUSTER_K_RANGE": (2, 4),
    "GMM_COV_TYPES": ["diag"],
    "CLUSTER_METHOD": "gmm",  # auto, gmm, hmm
    "ROLLING_WINDOW": 20,
    "EXPANDING_MIN_TRAIN": 2,
    "EXPANDING_TEST_WINDOW": 1,
    "ROLLING_TRAIN_WINDOWS": [6],
    "ROLLING_TEST_WINDOW": 2,
    "ROLLING_STEP_MONTHS": 2,
    "IMPUTE_STRATEGY": "median",
    "CLIP_OUTLIERS": True,
    "CLIP_QUANTILES": (0.001, 0.999),
    "CLUSTER_SCALER": "robust",
    "MODEL_SCALER": "robust",
    "MODEL_TYPE": "auto",  # auto, lightgbm, catboost, hist_gb, random_forest
    "MIN_CLUSTER_SAMPLES": 200,
    "MIN_CLUSTER_FRACTION": 0.02,
    "LOW_SAMPLE_STRATEGY": "global",
    "EMBEDDING_METHOD": "pca",
    "MAX_FEATURES_HEATMAP": 30,
    "SILHOUETTE_SAMPLE_SIZE": 500,
    "PLOT_DPI": 120,
    "QUICK_RUN": True,
    "MAX_EXPANDING_FOLDS": 8,
    "MAX_ROLLING_FOLDS": 8,
    "SHIFT_LABEL_AFTER_FEATURES": True,
    "LABEL_SHIFT": -1,
    "SHIFT_TIMESTAMP_WITH_LABEL": True,
}

@dataclass
class ClusterArtifacts:
    model: object
    scaler: object
    imputer: object
    clipper: Optional[object]
    feature_cols: List[str]
    method: str
    selection_report: Dict[str, object]


class QuantileClipper:
    def __init__(self, low_q: float, high_q: float):
        self.low_q = low_q
        self.high_q = high_q
        self.low_: Optional[np.ndarray] = None
        self.high_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "QuantileClipper":
        self.low_ = np.nanquantile(X, self.low_q, axis=0)
        self.high_ = np.nanquantile(X, self.high_q, axis=0)
        self.low_ = np.where(np.isfinite(self.low_), self.low_, -np.inf)
        self.high_ = np.where(np.isfinite(self.high_), self.high_, np.inf)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.low_ is None or self.high_ is None:
            raise ValueError("QuantileClipper not fitted.")
        return np.clip(X, self.low_, self.high_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class ConstantPredictor:
    def __init__(self, constant: int):
        self.constant = constant

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantPredictor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant, dtype=int)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    reports_dir = base_dir / "reports"
    artifacts_dir = base_dir / "artifacts"
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    confusion_dir = results_dir / "confusion_matrices"
    for path in [reports_dir, artifacts_dir, results_dir, figures_dir, confusion_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "reports": reports_dir,
        "artifacts": artifacts_dir,
        "results": results_dir,
        "figures": figures_dir,
        "confusion": confusion_dir,
    }


def detect_label_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    column_set = set(columns)
    for cand in candidates:
        if cand in column_set:
            return cand
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def load_data(config: Dict[str, object]) -> Tuple[pd.DataFrame, str, str]:
    data_path = Path(config["DATA_PATH"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    label_col = detect_label_column(df.columns.tolist(), config["LABEL_CANDIDATES"])
    if label_col is None:
        raise ValueError(
            "Label column not found. Available columns: "
            + ", ".join(df.columns.astype(str).tolist())
        )
    timestamp_col = config["TIMESTAMP_COL"]
    if timestamp_col not in df.columns:
        fallback = [c for c in df.columns if "timestamp" in c.lower()]
        if not fallback:
            raise ValueError(
                "Timestamp column not found. Available columns: "
                + ", ".join(df.columns.astype(str).tolist())
            )
        timestamp_col = fallback[0]
        logging.warning("Using fallback timestamp column: %s", timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, label_col]).copy()
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col])
    return df, label_col, timestamp_col


def coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_engineered_features(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    window = int(config["ROLLING_WINDOW"])
    if {"close", "open", "high", "low"}.issubset(df.columns):
        df["log_return_12h"] = np.log(df["close"] / df["close"].shift(1))
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["body_pct"] = (df["close"] - df["open"]).abs() / df["close"]
    if "volume" in df.columns:
        volume = df["volume"]
        roll_mean = volume.shift(1).rolling(window).mean()
        roll_std = volume.shift(1).rolling(window).std()
        df["volume_zscore_rolling"] = (volume - roll_mean) / roll_std
    if "log_return_12h" in df.columns:
        df["realized_vol_rolling"] = (
            df["log_return_12h"].shift(1).rolling(window).std()
        )
    if "spread_bps_last" in df.columns:
        spread = df["spread_bps_last"]
        df["spread_bps_roll_mean"] = spread.shift(1).rolling(window).mean()
        df["spread_bps_roll_std"] = spread.shift(1).rolling(window).std()
    if "imbalance_last" in df.columns:
        imb = df["imbalance_last"]
        df["imbalance_roll_mean"] = imb.shift(1).rolling(window).mean()
        df["imbalance_roll_std"] = imb.shift(1).rolling(window).std()
    if {"atm_iv_7d", "atm_iv_1d"}.issubset(df.columns):
        df["atm_iv_term_7d_1d"] = df["atm_iv_7d"] - df["atm_iv_1d"]
    if {"call25_iv_7d", "call25_iv_1d"}.issubset(df.columns):
        df["call25_iv_term_7d_1d"] = df["call25_iv_7d"] - df["call25_iv_1d"]
    if {"put25_iv_7d", "put25_iv_1d"}.issubset(df.columns):
        df["put25_iv_term_7d_1d"] = df["put25_iv_7d"] - df["put25_iv_1d"]
    return df


def build_features(
    df: pd.DataFrame, label_col: str, timestamp_col: str, config: Dict[str, object]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = coerce_numeric(df, exclude=[label_col, timestamp_col])
    df = add_engineered_features(df, config)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]
    numeric_cols = [c for c in numeric_cols if "timestamp" not in c.lower()]
    numeric_cols = [c for c in numeric_cols if not df[c].isna().all()]
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    numeric_cols = [c for c in numeric_cols if c.lower() not in ohlcv_cols]
    cluster_features = list(numeric_cols)
    model_features = list(numeric_cols)
    engineered_cols = [
        "log_return_12h",
        "range_pct",
        "body_pct",
        "volume_zscore_rolling",
        "realized_vol_rolling",
        "spread_bps_roll_mean",
        "spread_bps_roll_std",
        "imbalance_roll_mean",
        "imbalance_roll_std",
        "atm_iv_term_7d_1d",
        "call25_iv_term_7d_1d",
        "put25_iv_term_7d_1d",
    ]
    engineered_cols = [c for c in engineered_cols if c in df.columns]
    if engineered_cols:
        df = df.dropna(subset=engineered_cols)
    return df, cluster_features, model_features


def write_data_profile(
    df: pd.DataFrame, timestamp_col: str, label_col: str, path: Path
) -> None:
    missing = df.isna().mean().sort_values(ascending=False)
    numeric_desc = df.select_dtypes(include=[np.number]).describe().T
    time_min = df[timestamp_col].min()
    time_max = df[timestamp_col].max()
    with path.open("w", encoding="utf-8") as f:
        f.write("Data profile\n")
        f.write("=" * 40 + "\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Time coverage: {time_min} -> {time_max}\n")
        f.write(f"Label column: {label_col}\n\n")
        f.write("Missingness (fraction)\n")
        f.write(missing.to_string())
        f.write("\n\nDtypes\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nNumeric summary\n")
        f.write(numeric_desc.to_string())


def make_monthly_splits_expanding(
    df: pd.DataFrame,
    timestamp_col: str,
    min_train_months: int,
    test_window_months: int,
) -> List[Dict[str, object]]:
    months = df[timestamp_col].dt.to_period("M")
    unique_months = sorted(months.unique())
    splits = []
    for i in range(min_train_months, len(unique_months) - test_window_months + 1):
        train_months = unique_months[:i]
        test_months = unique_months[i : i + test_window_months]
        train_idx = df[months.isin(train_months)].index.values
        test_idx = df[months.isin(test_months)].index.values
        splits.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_months": [str(m) for m in train_months],
                "test_months": [str(m) for m in test_months],
            }
        )
    return splits


def make_monthly_splits_rolling(
    df: pd.DataFrame,
    timestamp_col: str,
    train_window_months: int,
    test_window_months: int,
    step_months: int,
) -> List[Dict[str, object]]:
    months = df[timestamp_col].dt.to_period("M")
    unique_months = sorted(months.unique())
    splits = []
    start = 0
    while start + train_window_months + test_window_months <= len(unique_months):
        train_months = unique_months[start : start + train_window_months]
        test_months = unique_months[
            start + train_window_months : start + train_window_months + test_window_months
        ]
        train_idx = df[months.isin(train_months)].index.values
        test_idx = df[months.isin(test_months)].index.values
        splits.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_months": [str(m) for m in train_months],
                "test_months": [str(m) for m in test_months],
            }
        )
        start += step_months
    return splits


def fit_preprocessor(
    X: pd.DataFrame, config: Dict[str, object]
) -> Tuple[Optional[QuantileClipper], SimpleImputer, RobustScaler, np.ndarray]:
    X_values = X.values
    clipper = None
    if config["CLIP_OUTLIERS"]:
        clipper = QuantileClipper(*config["CLIP_QUANTILES"]).fit(X_values)
        X_values = clipper.transform(X_values)
    imputer = SimpleImputer(strategy=config["IMPUTE_STRATEGY"])
    X_values = imputer.fit_transform(X_values)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_values)
    return clipper, imputer, scaler, X_scaled


def transform_with_preprocessor(
    X: pd.DataFrame,
    clipper: Optional[QuantileClipper],
    imputer: SimpleImputer,
    scaler: RobustScaler,
) -> np.ndarray:
    X_values = X.values
    if clipper is not None:
        X_values = clipper.transform(X_values)
    X_values = imputer.transform(X_values)
    X_scaled = scaler.transform(X_values)
    return X_scaled


def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    return -2 * log_likelihood + n_params * math.log(max(n_samples, 1))


def estimate_hmm_params(n_states: int, n_features: int) -> int:
    trans_params = n_states * (n_states - 1)
    start_params = n_states - 1
    means_params = n_states * n_features
    cov_params = n_states * n_features
    return trans_params + start_params + means_params + cov_params


def safe_import_hmm():
    try:
        from hmmlearn.hmm import GaussianHMM

        return GaussianHMM, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def search_gmm_models(
    X_scaled: np.ndarray, config: Dict[str, object]
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    k_min, k_max = config["CLUSTER_K_RANGE"]
    results = []
    best = None
    for cov_type in config["GMM_COV_TYPES"]:
        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(
                n_components=k, covariance_type=cov_type, random_state=config["SEED"]
            )
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            sample_size = min(config["SILHOUETTE_SAMPLE_SIZE"], len(labels))
            if k > 1 and sample_size > 1:
                silhouette = silhouette_score(
                    X_scaled, labels, sample_size=sample_size, random_state=config["SEED"]
                )
            else:
                silhouette = np.nan
            bic = gmm.bic(X_scaled)
            aic = gmm.aic(X_scaled)
            result = {
                "model": gmm,
                "k": k,
                "cov_type": cov_type,
                "bic": bic,
                "aic": aic,
                "silhouette": silhouette,
            }
            results.append(result)
            if best is None or bic < best["bic"]:
                best = result
    return best, results


def search_hmm_models(
    X_scaled: np.ndarray, config: Dict[str, object]
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]], Optional[Exception]]:
    GaussianHMM, err = safe_import_hmm()
    if GaussianHMM is None:
        return None, [], err
    k_min, k_max = config["CLUSTER_K_RANGE"]
    results = []
    best = None
    for k in range(k_min, k_max + 1):
        hmm = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=200,
            random_state=config["SEED"],
        )
        try:
            hmm.fit(X_scaled)
            log_likelihood = hmm.score(X_scaled)
            n_params = estimate_hmm_params(k, X_scaled.shape[1])
            bic = compute_bic(log_likelihood, n_params, X_scaled.shape[0])
            labels = hmm.predict(X_scaled)
            sample_size = min(config["SILHOUETTE_SAMPLE_SIZE"], len(labels))
            if k > 1 and sample_size > 1:
                silhouette = silhouette_score(
                    X_scaled, labels, sample_size=sample_size, random_state=config["SEED"]
                )
            else:
                silhouette = np.nan
            result = {
                "model": hmm,
                "k": k,
                "cov_type": "diag",
                "bic": bic,
                "aic": np.nan,
                "silhouette": silhouette,
                "log_likelihood": log_likelihood,
            }
            results.append(result)
            if best is None or bic < best["bic"]:
                best = result
        except Exception as exc:  # noqa: BLE001
            logging.warning("HMM fit failed for k=%s: %s", k, exc)
    return best, results, None


def choose_cluster_model(
    gmm_best: Dict[str, object],
    hmm_best: Optional[Dict[str, object]],
    config: Dict[str, object],
) -> Tuple[Dict[str, object], str]:
    if config["CLUSTER_METHOD"] == "gmm" or hmm_best is None:
        return gmm_best, "gmm"
    if config["CLUSTER_METHOD"] == "hmm":
        return hmm_best, "hmm"
    gmm_score = gmm_best.get("silhouette", np.nan)
    hmm_score = hmm_best.get("silhouette", np.nan)
    if np.isnan(hmm_score):
        return gmm_best, "gmm"
    if np.isnan(gmm_score):
        return hmm_best, "hmm"
    if hmm_score > gmm_score:
        return hmm_best, "hmm"
    return gmm_best, "gmm"


def fit_cluster_model(
    df: pd.DataFrame, feature_cols: List[str], config: Dict[str, object]
) -> ClusterArtifacts:
    X = df[feature_cols]
    clipper, imputer, scaler, X_scaled = fit_preprocessor(X, config)
    gmm_best, gmm_results = search_gmm_models(X_scaled, config)
    hmm_best, hmm_results, hmm_error = search_hmm_models(X_scaled, config)
    chosen, method = choose_cluster_model(gmm_best, hmm_best, config)
    selection_report = {
        "gmm_results": [
            {
                "k": res["k"],
                "cov_type": res["cov_type"],
                "bic": res["bic"],
                "aic": res["aic"],
                "silhouette": res["silhouette"],
            }
            for res in gmm_results
        ],
        "hmm_results": [
            {
                "k": res["k"],
                "cov_type": res["cov_type"],
                "bic": res["bic"],
                "aic": res["aic"],
                "silhouette": res["silhouette"],
                "log_likelihood": res.get("log_likelihood"),
            }
            for res in hmm_results
        ],
        "hmm_error": None if hmm_error is None else str(hmm_error),
        "chosen_method": method,
        "chosen_k": chosen["k"],
        "chosen_cov_type": chosen["cov_type"],
        "chosen_bic": chosen["bic"],
        "chosen_aic": chosen.get("aic", np.nan),
        "chosen_silhouette": chosen.get("silhouette", np.nan),
    }
    return ClusterArtifacts(
        model=chosen["model"],
        scaler=scaler,
        imputer=imputer,
        clipper=clipper,
        feature_cols=feature_cols,
        method=method,
        selection_report=selection_report,
    )


def assign_clusters(
    df: pd.DataFrame, artifacts: ClusterArtifacts
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[artifacts.feature_cols]
    X_scaled = transform_with_preprocessor(
        X, artifacts.clipper, artifacts.imputer, artifacts.scaler
    )
    if artifacts.method == "gmm":
        model = artifacts.model
        cluster_id = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
        return cluster_id, confidence
    model = artifacts.model
    cluster_id = model.predict(X_scaled)
    try:
        proba = model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
    except Exception:  # noqa: BLE001
        confidence = np.full(len(cluster_id), np.nan)
    return cluster_id, confidence


def map_clusters_to_regimes(
    df: pd.DataFrame, cluster_id: np.ndarray
) -> Tuple[np.ndarray, Dict[int, str]]:
    mapping_features = []
    for col in ["realized_vol_rolling", "range_pct"]:
        if col in df.columns:
            mapping_features.append(col)
    if not mapping_features:
        regime_labels = {cid: f"regime_{cid}" for cid in np.unique(cluster_id)}
        mapped = np.array([regime_labels[cid] for cid in cluster_id])
        return mapped, regime_labels
    temp = df.copy()
    temp["cluster_id"] = cluster_id
    stats = temp.groupby("cluster_id")[mapping_features].mean()
    score = stats.sum(axis=1)
    order = score.sort_values().index.tolist()
    n = len(order)
    regime_labels = {}
    for i, cid in enumerate(order):
        if n > 3:
            if i < n / 3:
                label = "low_vol"
            elif i < 2 * n / 3:
                label = "mid_vol"
            else:
                label = "high_vol"
        else:
            label = f"regime_{cid}"
        regime_labels[int(cid)] = label
    mapped = np.array([regime_labels[int(cid)] for cid in cluster_id])
    return mapped, regime_labels


def get_model_instance(model_type: str, random_state: int):
    if model_type in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                verbosity=-1,
                random_state=random_state,
            )
        except Exception:  # noqa: BLE001
            if model_type == "lightgbm":
                raise
    if model_type in ("auto", "catboost"):
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                verbose=False,
                random_seed=random_state,
            )
        except Exception:  # noqa: BLE001
            if model_type == "catboost":
                raise
    if model_type in ("auto", "hist_gb"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, random_state=random_state
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_supervised_model(
    X: np.ndarray, y: np.ndarray, config: Dict[str, object]
):
    unique_classes = np.unique(y)
    if unique_classes.size == 1:
        return ConstantPredictor(unique_classes[0])
    model = get_model_instance(config["MODEL_TYPE"], config["SEED"])
    model.fit(X, y)
    return model


def train_expert_models(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    config: Dict[str, object],
) -> Tuple[Dict[int, object], List[int]]:
    models = {}
    low_sample_clusters = []
    total_samples = len(y)
    min_samples = max(
        int(config["MIN_CLUSTER_FRACTION"] * total_samples),
        int(config["MIN_CLUSTER_SAMPLES"]),
    )
    for cid in np.unique(clusters):
        idx = clusters == cid
        if idx.sum() < min_samples:
            low_sample_clusters.append(int(cid))
            continue
        model = fit_supervised_model(X[idx], y[idx], config)
        models[int(cid)] = model
    return models, low_sample_clusters


def predict_with_experts(
    X: np.ndarray,
    clusters: np.ndarray,
    expert_models: Dict[int, object],
    fallback_model: object,
) -> np.ndarray:
    preds = np.empty(len(clusters), dtype=int)
    for cid in np.unique(clusters):
        idx = clusters == cid
        model = expert_models.get(int(cid), fallback_model)
        preds[idx] = model.predict(X[idx])
    return preds


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_precision"] = precision
    metrics["macro_recall"] = recall
    metrics["macro_f1"] = f1
    precision_c, recall_c, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(label_names))), zero_division=0
    )
    label_map = {name.lower(): idx for idx, name in enumerate(label_names)}
    for key in ["long", "short", "skip"]:
        if key in label_map:
            idx = label_map[key]
            metrics[f"precision_{key}"] = precision_c[idx]
            metrics[f"recall_{key}"] = recall_c[idx]
    return metrics


def plot_confusion_matrix(
    df_cm: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=CONFIG["PLOT_DPI"])
    im = ax.imshow(df_cm.values, cmap="Blues")
    ax.set_xticks(range(len(df_cm.columns)))
    ax.set_yticks(range(len(df_cm.index)))
    ax.set_xticklabels(df_cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(df_cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(df_cm.values):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    path: Path,
    plot_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    df_cm.to_csv(path, index=True)
    if plot_path is not None:
        plot_confusion_matrix(df_cm, plot_path, title or path.stem)
    return df_cm


def run_fold(
    df: pd.DataFrame,
    feature_cols_cluster: List[str],
    feature_cols_model: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_encoder: LabelEncoder,
    label_names: List[str],
    config: Dict[str, object],
) -> Dict[str, object]:
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    y_train = label_encoder.transform(train_df["_label_str"])
    y_test = label_encoder.transform(test_df["_label_str"])

    cluster_artifacts = fit_cluster_model(train_df, feature_cols_cluster, config)
    train_clusters, _ = assign_clusters(train_df, cluster_artifacts)
    test_clusters, _ = assign_clusters(test_df, cluster_artifacts)

    model_clipper, model_imputer, model_scaler, X_train_scaled = fit_preprocessor(
        train_df[feature_cols_model], config
    )
    X_test_scaled = transform_with_preprocessor(
        test_df[feature_cols_model], model_clipper, model_imputer, model_scaler
    )

    global_model = fit_supervised_model(X_train_scaled, y_train, config)
    baseline_pred = global_model.predict(X_test_scaled)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred, label_names)

    expert_models, low_sample = train_expert_models(
        X_train_scaled, y_train, train_clusters, config
    )
    expert_pred = predict_with_experts(
        X_test_scaled, test_clusters, expert_models, global_model
    )
    expert_metrics = evaluate_predictions(y_test, expert_pred, label_names)

    return {
        "baseline_pred": baseline_pred,
        "expert_pred": expert_pred,
        "y_test": y_test,
        "baseline_metrics": baseline_metrics,
        "expert_metrics": expert_metrics,
        "low_sample_clusters": low_sample,
        "num_clusters": int(np.unique(train_clusters).size),
    }


def summarize_metrics(records: List[Dict[str, object]], cv_type: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [col for col in numeric_cols if col not in ("fold_id",)]
    summary_rows = []
    for approach in df["approach"].unique():
        subset = df[df["approach"] == approach]
        for metric in metric_cols:
            if metric in ("label_dist_train", "label_dist_test"):
                continue
            summary_rows.append(
                {
                    "cv_type": cv_type,
                    "approach": approach,
                    "metric": metric,
                    "mean": subset[metric].mean(),
                    "std": subset[metric].std(),
                    "folds": len(subset),
                }
            )
    return pd.DataFrame(summary_rows)


def evaluate_fold(
    df: pd.DataFrame,
    feature_cols_cluster: List[str],
    feature_cols_model: List[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_encoder: LabelEncoder,
    label_names: List[str],
    config: Dict[str, object],
) -> Dict[str, object]:
    return run_fold(
        df,
        feature_cols_cluster,
        feature_cols_model,
        train_idx,
        test_idx,
        label_encoder,
        label_names,
        config,
    )


def aggregate_results(records: List[Dict[str, object]], cv_type: str) -> pd.DataFrame:
    return summarize_metrics(records, cv_type)


def choose_winner(summary: pd.DataFrame) -> str:
    pivot = summary.pivot_table(
        index=["approach"], columns="metric", values="mean", aggfunc="first"
    )
    if "macro_f1" not in pivot.columns:
        return "winner_not_determined"
    pivot["score"] = pivot["macro_f1"]
    if "recall_long" in pivot.columns:
        pivot["score"] += 0.25 * pivot["recall_long"]
    if "recall_short" in pivot.columns:
        pivot["score"] += 0.25 * pivot["recall_short"]
    winner = pivot["score"].idxmax()
    return str(winner)


def plot_bic_aic(results: List[Dict[str, object]], path: Path, title: str) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=CONFIG["PLOT_DPI"])
    for cov_type in df["cov_type"].unique():
        subset = df[df["cov_type"] == cov_type]
        ax.plot(subset["k"], subset["bic"], label=f"BIC {cov_type}")
        if "aic" in subset and subset["aic"].notna().any():
            ax.plot(subset["k"], subset["aic"], linestyle="--", label=f"AIC {cov_type}")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_embedding(
    X_scaled: np.ndarray,
    clusters: np.ndarray,
    path: Path,
) -> PCA:
    pca = PCA(n_components=2, random_state=CONFIG["SEED"])
    embedding = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=CONFIG["PLOT_DPI"])
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=clusters,
        s=8,
        cmap="tab10",
        alpha=0.7,
    )
    ax.set_title("PCA embedding colored by cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster_id")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return pca


def plot_cluster_ts(
    timestamps: pd.Series, clusters: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=CONFIG["PLOT_DPI"])
    ax.scatter(timestamps, clusters, s=6, alpha=0.7)
    ax.set_title("Cluster assignment over time")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("cluster_id")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_feature_means(
    df: pd.DataFrame,
    clusters: np.ndarray,
    feature_cols: List[str],
    path: Path,
    max_features: int,
) -> None:
    temp = df.copy()
    temp["cluster_id"] = clusters
    variances = temp[feature_cols].var().sort_values(ascending=False)
    selected = variances.head(max_features).index.tolist()
    means = temp.groupby("cluster_id")[selected].mean()
    fig, ax = plt.subplots(figsize=(12, 4), dpi=CONFIG["PLOT_DPI"])
    im = ax.imshow(means.values, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(means.shape[0]))
    ax.set_yticklabels([str(c) for c in means.index])
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90)
    ax.set_title("Cluster feature means (top variance features)")
    fig.colorbar(im, ax=ax, label="mean value")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_label_distribution(
    labels: np.ndarray,
    clusters: np.ndarray,
    label_names: List[str],
    path: Path,
) -> None:
    df = pd.DataFrame({"label": labels, "cluster": clusters})
    counts = (
        df.groupby(["cluster", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(range(len(label_names))))
    )
    fig, ax = plt.subplots(figsize=(8, 4), dpi=CONFIG["PLOT_DPI"])
    bottom = np.zeros(counts.shape[0])
    for idx, label_name in enumerate(label_names):
        values = counts[idx].values
        ax.bar(counts.index, values, bottom=bottom, label=label_name)
        bottom += values
    ax.set_title("Per-cluster label distribution")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_clustering_report(
    path: Path,
    artifacts: ClusterArtifacts,
    feature_cols: List[str],
    n_samples: int,
) -> None:
    report = artifacts.selection_report
    with path.open("w", encoding="utf-8") as f:
        f.write("# Clustering report\n\n")
        f.write(f"- Samples used: {n_samples}\n")
        f.write(f"- Features used: {len(feature_cols)}\n")
        f.write(f"- Scaler: RobustScaler\n")
        f.write(f"- Outlier clipping: {CONFIG['CLIP_OUTLIERS']}\n")
        f.write(f"- Selection method: {report['chosen_method']}\n")
        f.write(f"- Selected K: {report['chosen_k']}\n")
        f.write(f"- Selected cov_type: {report['chosen_cov_type']}\n")
        f.write(f"- BIC: {report['chosen_bic']}\n")
        f.write(f"- Silhouette (sampled): {report['chosen_silhouette']}\n\n")
        f.write("## GMM search\n")
        f.write(json.dumps(report["gmm_results"], indent=2))
        f.write("\n\n## HMM search\n")
        f.write(json.dumps(report["hmm_results"], indent=2))
        if report["hmm_error"]:
            f.write("\n\nHMM error: " + report["hmm_error"])


def write_cv_report(
    path: Path,
    summary: pd.DataFrame,
    cv_type: str,
    winner: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# CV report ({cv_type})\n\n")
        f.write(f"- Winner (Macro F1 + long/short recall): {winner}\n\n")
        f.write("## Summary metrics\n")
        f.write(summary.to_csv(index=False))


def main() -> None:
    setup_logging()
    base_dir = Path(CONFIG["OUTPUT_DIR"])
    paths = ensure_dirs(base_dir)
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    confusion_run_dir = paths["confusion"] / run_id
    confusion_plot_dir = confusion_run_dir / "plots"
    confusion_summary_dir = confusion_run_dir / "summary"
    confusion_run_dir.mkdir(parents=True, exist_ok=True)
    confusion_plot_dir.mkdir(parents=True, exist_ok=True)
    confusion_summary_dir.mkdir(parents=True, exist_ok=True)
    df, label_col, timestamp_col = load_data(CONFIG)
    df, cluster_features, model_features = build_features(
        df, label_col, timestamp_col, CONFIG
    )
    if CONFIG.get("SHIFT_LABEL_AFTER_FEATURES"):
        shift = int(CONFIG.get("LABEL_SHIFT", -1))
        df[label_col] = df[label_col].shift(shift)
        if CONFIG.get("SHIFT_TIMESTAMP_WITH_LABEL"):
            df[timestamp_col] = df[timestamp_col].shift(shift)
        df = df.dropna(subset=[label_col]).copy()
    df["_label_str"] = df[label_col].astype(str)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["_label_str"])
    label_names = label_encoder.classes_.tolist()
    write_data_profile(
        df,
        timestamp_col,
        label_col,
        paths["reports"] / "data_profile.txt",
    )

    logging.info("Fitting full clustering model for artifacts and plots...")
    full_cluster_artifacts = fit_cluster_model(df, cluster_features, CONFIG)
    cluster_id, confidence = assign_clusters(df, full_cluster_artifacts)
    mapped_regime, regime_map = map_clusters_to_regimes(df, cluster_id)

    artifacts_payload = {
        "model": full_cluster_artifacts.model,
        "imputer": full_cluster_artifacts.imputer,
        "scaler": full_cluster_artifacts.scaler,
        "clipper": full_cluster_artifacts.clipper,
        "feature_cols": full_cluster_artifacts.feature_cols,
        "method": full_cluster_artifacts.method,
        "selection_report": full_cluster_artifacts.selection_report,
    }
    joblib.dump(artifacts_payload, paths["artifacts"] / "cluster_model.joblib")
    joblib.dump(full_cluster_artifacts.scaler, paths["artifacts"] / "scaler.joblib")

    assignments = pd.DataFrame(
        {
            timestamp_col: df[timestamp_col].values,
            "cluster_id": cluster_id,
            "confidence": confidence,
            "mapped_regime_3": mapped_regime,
        }
    )
    assignments.to_csv(
        paths["artifacts"] / "cluster_assignments.csv", index=False
    )

    write_clustering_report(
        paths["reports"] / "clustering_report.md",
        full_cluster_artifacts,
        cluster_features,
        len(df),
    )

    logging.info("Creating plots...")
    plot_cluster_ts(
        df[timestamp_col], cluster_id, paths["figures"] / "clusters_ts.png"
    )
    plot_cluster_feature_means(
        df,
        cluster_id,
        cluster_features,
        paths["figures"] / "cluster_feature_means.png",
        CONFIG["MAX_FEATURES_HEATMAP"],
    )
    plot_cluster_label_distribution(
        label_encoder.transform(df["_label_str"]),
        cluster_id,
        label_names,
        paths["figures"] / "per_cluster_label_distribution.png",
    )
    clipper, imputer, scaler, X_scaled = fit_preprocessor(
        df[cluster_features], CONFIG
    )
    pca = plot_embedding(
        X_scaled, cluster_id, paths["figures"] / "embedding_2d.png"
    )
    joblib.dump(pca, paths["artifacts"] / "pca.joblib")

    plot_bic_aic(
        full_cluster_artifacts.selection_report["gmm_results"],
        paths["figures"] / "bic_aic_vs_k.png",
        "GMM model selection (BIC/AIC)",
    )

    logging.info("Running expanding-window CV...")
    expanding_splits = make_monthly_splits_expanding(
        df,
        timestamp_col,
        CONFIG["EXPANDING_MIN_TRAIN"],
        CONFIG["EXPANDING_TEST_WINDOW"],
    )
    if CONFIG.get("QUICK_RUN"):
        max_folds = int(CONFIG.get("MAX_EXPANDING_FOLDS", 0) or 0)
        if max_folds > 0:
            expanding_splits = expanding_splits[:max_folds]
        logging.info("Quick run enabled: expanding folds=%s", len(expanding_splits))
    expanding_records = []
    summary_matrices: Dict[str, np.ndarray] = {}
    for fold_id, split in enumerate(expanding_splits, start=1):
        result = evaluate_fold(
            df,
            cluster_features,
            model_features,
            split["train_idx"],
            split["test_idx"],
            label_encoder,
            label_names,
            CONFIG,
        )
        for approach, metrics in [
            ("baseline", result["baseline_metrics"]),
            ("cluster_experts", result["expert_metrics"]),
        ]:
            record = {
                "fold_id": fold_id,
                "approach": approach,
                "cv_type": "expanding",
                "train_months": ",".join(split["train_months"]),
                "test_months": ",".join(split["test_months"]),
                "num_clusters": result["num_clusters"],
                "low_sample_clusters": ",".join(map(str, result["low_sample_clusters"])),
                "label_dist_train": json.dumps(
                    df.loc[split["train_idx"]]["_label_str"]
                    .value_counts()
                    .to_dict()
                ),
                "label_dist_test": json.dumps(
                    df.loc[split["test_idx"]]["_label_str"]
                    .value_counts()
                    .to_dict()
                ),
            }
            record.update(metrics)
            expanding_records.append(record)
        base_path = confusion_run_dir / f"expanding_fold_{fold_id}_baseline_confusion.csv"
        base_plot = (
            confusion_plot_dir / f"expanding_fold_{fold_id}_baseline_confusion.png"
        )
        df_base = save_confusion_matrix(
            result["y_test"],
            result["baseline_pred"],
            label_names,
            base_path,
            plot_path=base_plot,
            title=f"expanding fold {fold_id} baseline",
        )
        key_base = "expanding_baseline"
        summary_matrices[key_base] = summary_matrices.get(
            key_base, np.zeros_like(df_base.values)
        ) + df_base.values

        cluster_path = confusion_run_dir / f"expanding_fold_{fold_id}_cluster_confusion.csv"
        cluster_plot = (
            confusion_plot_dir / f"expanding_fold_{fold_id}_cluster_confusion.png"
        )
        df_cluster = save_confusion_matrix(
            result["y_test"],
            result["expert_pred"],
            label_names,
            cluster_path,
            plot_path=cluster_plot,
            title=f"expanding fold {fold_id} cluster",
        )
        key_cluster = "expanding_cluster_experts"
        summary_matrices[key_cluster] = summary_matrices.get(
            key_cluster, np.zeros_like(df_cluster.values)
        ) + df_cluster.values

    expanding_df = pd.DataFrame(expanding_records)
    expanding_df.to_csv(paths["results"] / "fold_metrics_expanding.csv", index=False)
    summary_expanding = aggregate_results(expanding_records, "expanding")
    winner_expanding = choose_winner(summary_expanding)
    write_cv_report(
        paths["reports"] / "cv_report_expanding.md",
        summary_expanding,
        "expanding",
        winner_expanding,
    )

    logging.info("Running rolling-window CV...")
    rolling_records = []
    for window in CONFIG["ROLLING_TRAIN_WINDOWS"]:
        rolling_splits = make_monthly_splits_rolling(
            df,
            timestamp_col,
            window,
            CONFIG["ROLLING_TEST_WINDOW"],
            CONFIG["ROLLING_STEP_MONTHS"],
        )
        if CONFIG.get("QUICK_RUN"):
            max_folds = int(CONFIG.get("MAX_ROLLING_FOLDS", 0) or 0)
            if max_folds > 0:
                rolling_splits = rolling_splits[:max_folds]
            logging.info(
                "Quick run enabled: rolling folds=%s for window=%s",
                len(rolling_splits),
                window,
            )
        for fold_id, split in enumerate(rolling_splits, start=1):
            result = evaluate_fold(
                df,
                cluster_features,
                model_features,
                split["train_idx"],
                split["test_idx"],
                label_encoder,
                label_names,
                CONFIG,
            )
            for approach, metrics in [
                ("baseline", result["baseline_metrics"]),
                ("cluster_experts", result["expert_metrics"]),
            ]:
                record = {
                    "fold_id": fold_id,
                    "approach": approach,
                    "cv_type": f"rolling_w{window}",
                    "train_months": ",".join(split["train_months"]),
                    "test_months": ",".join(split["test_months"]),
                    "num_clusters": result["num_clusters"],
                    "low_sample_clusters": ",".join(
                        map(str, result["low_sample_clusters"])
                    ),
                    "label_dist_train": json.dumps(
                        df.loc[split["train_idx"]]["_label_str"]
                        .value_counts()
                        .to_dict()
                    ),
                    "label_dist_test": json.dumps(
                        df.loc[split["test_idx"]]["_label_str"]
                        .value_counts()
                        .to_dict()
                    ),
                }
                record.update(metrics)
                rolling_records.append(record)
            base_path = (
                confusion_run_dir
                / f"rolling_w{window}_fold_{fold_id}_baseline_confusion.csv"
            )
            base_plot = (
                confusion_plot_dir
                / f"rolling_w{window}_fold_{fold_id}_baseline_confusion.png"
            )
            df_base = save_confusion_matrix(
                result["y_test"],
                result["baseline_pred"],
                label_names,
                base_path,
                plot_path=base_plot,
                title=f"rolling w{window} fold {fold_id} baseline",
            )
            key_base = f"rolling_w{window}_baseline"
            summary_matrices[key_base] = summary_matrices.get(
                key_base, np.zeros_like(df_base.values)
            ) + df_base.values

            cluster_path = (
                confusion_run_dir
                / f"rolling_w{window}_fold_{fold_id}_cluster_confusion.csv"
            )
            cluster_plot = (
                confusion_plot_dir
                / f"rolling_w{window}_fold_{fold_id}_cluster_confusion.png"
            )
            df_cluster = save_confusion_matrix(
                result["y_test"],
                result["expert_pred"],
                label_names,
                cluster_path,
                plot_path=cluster_plot,
                title=f"rolling w{window} fold {fold_id} cluster",
            )
            key_cluster = f"rolling_w{window}_cluster_experts"
            summary_matrices[key_cluster] = summary_matrices.get(
                key_cluster, np.zeros_like(df_cluster.values)
            ) + df_cluster.values

    rolling_df = pd.DataFrame(rolling_records)
    rolling_df.to_csv(paths["results"] / "fold_metrics_rolling.csv", index=False)
    summary_rolling = aggregate_results(rolling_records, "rolling")
    winner_rolling = choose_winner(summary_rolling)
    write_cv_report(
        paths["reports"] / "cv_report_rolling.md",
        summary_rolling,
        "rolling",
        winner_rolling,
    )

    overall_summary = pd.concat([summary_expanding, summary_rolling], ignore_index=True)
    overall_summary.to_csv(paths["results"] / "overall_summary.csv", index=False)

    for key, matrix in summary_matrices.items():
        df_sum = pd.DataFrame(matrix, index=label_names, columns=label_names)
        csv_path = confusion_summary_dir / f"{key}_summary_confusion.csv"
        png_path = confusion_summary_dir / f"{key}_summary_confusion.png"
        df_sum.to_csv(csv_path, index=True)
        plot_confusion_matrix(df_sum, png_path, title=f"{key} summary")

    logging.info("Outputs saved to %s", base_dir.resolve())
    logging.info("Expanding winner: %s", winner_expanding)
    logging.info("Rolling winner: %s", winner_rolling)


if __name__ == "__main__":
    main()



