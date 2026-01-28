#!/usr/bin/env python
"""
GMM OHLCV Evaluation Script (Daily)

SHIFT_OHLCV_BY_1
- True: uses OHLCV from t-1 to predict regime at t (real-time safe; avoids look-ahead).
- False: uses same-day OHLCV at t (retrospective regime modeling; not real-time safe if you
  would not have end-of-day OHLCV at decision time).

Metrics (brief):
- Avg log-likelihood: mean of per-sample log-likelihoods; higher is better fit.
- BIC/AIC: penalized likelihood criteria for model selection (lower is better). Use TRAIN
  BIC/AIC for selection; TEST BIC/AIC are printed for reference only.
- Silhouette/Davies-Bouldin: cluster separation/compactness metrics on hard labels.
  Silhouette higher is better; Davies-Bouldin lower is better.
- Responsibility entropy: average uncertainty of soft assignments; higher => more ambiguous
  regime membership.
- Multi-run variance: stability across random initializations for the best K. Lower std of
  log-likelihood / weights / means implies more stable solutions.

Reproducibility:
- Fixed random seeds, fixed chronological split, scaler fit on train only.
"""

from __future__ import annotations

import json
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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional
    raise RuntimeError("matplotlib is required for plots") from exc


# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\data_1d_indicators.csv"
OUTPUT_DIR = r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\gmm_ohlcv_eval"

SHIFT_OHLCV_BY_1 = True
K_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10]
COVARIANCE_TYPES = ["diag", "tied"]  # list of {"full", "diag", "tied", "spherical"}

TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

N_RUNS = 10
N_INIT = 5
RANDOM_STATE = 42
EPS = 1e-12

TIMESTAMP_CANDIDATES = ["timestamp", "date", "time", "ts", "ts_utc", "datetime"]
OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: List[str]
    ts_col: str


# -----------------------------
# Utility functions
# -----------------------------

def detect_timestamp_column(columns: Iterable[str]) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for name in TIMESTAMP_CANDIDATES:
        if name in col_map:
            return col_map[name]
    # Fallback: look for any column containing the candidate name
    for name in TIMESTAMP_CANDIDATES:
        for c in columns:
            if name in c.lower():
                return c
    return None


def detect_ohlcv_columns(columns: Iterable[str]) -> List[str]:
    col_map = {c.lower(): c for c in columns}
    missing = [c for c in OHLCV_CANDIDATES if c not in col_map]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return [col_map[c] for c in OHLCV_CANDIDATES]


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

    feature_cols = detect_ohlcv_columns(df.columns)

    # Filter years 2020-2025 inclusive
    start_all = pd.Timestamp(TRAIN_START)
    end_all = pd.Timestamp(TEST_END)
    df = df[(df[ts_col] >= start_all) & (df[ts_col] <= end_all)].copy()

    if SHIFT_OHLCV_BY_1:
        df[feature_cols] = df[feature_cols].shift(1)

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train_df = df[(df[ts_col] >= train_start) & (df[ts_col] <= train_end)].copy()
    test_df = df[(df[ts_col] >= test_start) & (df[ts_col] <= test_end)].copy()

    return SplitData(train_df=train_df, test_df=test_df, feature_cols=feature_cols, ts_col=ts_col)


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


def align_components(ref_means: np.ndarray, means: np.ndarray, weights: np.ndarray, covs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align components to reference means via Hungarian assignment."""
    cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
    assignment = hungarian_min_cost(cost)
    return weights[assignment], means[assignment], covs[assignment]


def build_full_covariances(covariances: np.ndarray, cov_type: str, n_components: int, n_features: int) -> np.ndarray:
    if cov_type == "full":
        return covariances
    if cov_type == "diag":
        return np.array([np.diag(c) for c in covariances])
    if cov_type == "tied":
        return np.repeat(covariances[None, :, :], n_components, axis=0)
    if cov_type == "spherical":
        return np.array([np.eye(n_features) * c for c in covariances])
    raise ValueError(f"Unknown covariance type: {cov_type}")


def cov_scaled_to_original(full_cov_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scale = scaler.scale_
    scale_outer = np.outer(scale, scale)
    return full_cov_scaled * scale_outer


def means_scaled_to_original(means_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return means_scaled * scaler.scale_ + scaler.mean_


def ensure_dirs() -> Dict[str, Path]:
    base = Path(OUTPUT_DIR)
    results_dir = base / "results"
    models_dir = base / "models"
    plots_dir = base / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "results": results_dir,
        "models": models_dir,
        "plots": plots_dir,
    }


def plot_metrics(
    k_list: List[int],
    train_bic: List[float],
    test_bic: List[float],
    train_ll: List[float],
    test_ll: List[float],
    plots_dir: Path,
    cov_type: str,
) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(k_list, train_bic, marker="o", label="Train BIC")
    plt.plot(k_list, test_bic, marker="o", label="Test BIC")
    plt.title("BIC vs K")
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_list, train_ll, marker="o", label="Train AvgLogLik")
    plt.plot(k_list, test_ll, marker="o", label="Test AvgLogLik")
    plt.title("Avg Log-Likelihood vs K")
    plt.xlabel("K")
    plt.ylabel("Avg Log-Likelihood")
    plt.legend()

    plt.tight_layout()
    out_path = plots_dir / f"gmm_bic_loglik_vs_k_{cov_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, gmm: GaussianMixture, plots_dir: Path) -> None:
    if X.size == 0:
        return
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)
    means2 = pca.transform(gmm.means_)

    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10, alpha=0.6, cmap="tab10")
    plt.title("Train PCA (2D) - Hard Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Optional covariance ellipses
    try:
        covs_full = build_full_covariances(gmm.covariances_, gmm.covariance_type, gmm.n_components, gmm.means_.shape[1])
        for k in range(gmm.n_components):
            cov2 = pca.components_ @ covs_full[k] @ pca.components_.T
            vals, vecs = np.linalg.eigh(cov2)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2.0 * np.sqrt(np.maximum(vals, 0))
            from matplotlib.patches import Ellipse

            ellipse = Ellipse(
                xy=means2[k],
                width=width,
                height=height,
                angle=angle,
                edgecolor="black",
                facecolor="none",
                alpha=0.7,
                linewidth=1.0,
            )
            plt.gca().add_patch(ellipse)
    except Exception:
        pass

    out_path = plots_dir / "gmm_pca_train_scatter.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    np.random.seed(RANDOM_STATE)

    paths = ensure_dirs()

    df = pd.read_csv(INPUT_CSV)
    split = filter_and_split(df)

    feature_cols = split.feature_cols
    ts_col = split.ts_col

    train_df = split.train_df
    test_df = split.test_df

    if train_df.empty:
        raise RuntimeError("Training split is empty after filtering")

    X_train = train_df[feature_cols].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float) if not test_df.empty else np.empty((0, len(feature_cols)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test.size else X_test

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_rows = []
    models = {}

    print("=== GMM OHLCV Evaluation ===")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"SHIFT_OHLCV_BY_1: {SHIFT_OHLCV_BY_1}")
    print(f"Covariance types: {COVARIANCE_TYPES}")
    print(f"Train period: {TRAIN_START} to {TRAIN_END}")
    print(f"Test period:  {TEST_START} to {TEST_END}")
    print("Feature columns:", feature_cols)
    print()

    for cov_type in COVARIANCE_TYPES:
        train_bic_list = []
        test_bic_list = []
        train_ll_list = []
        test_ll_list = []

        print(f"--- Covariance type: {cov_type} ---")

        for k in K_LIST:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=RANDOM_STATE,
                n_init=N_INIT,
            )
            gmm.fit(X_train_scaled)
            models[(cov_type, k)] = gmm

            train_avg_ll = compute_avg_loglik(gmm, X_train_scaled)
            test_avg_ll = compute_avg_loglik(gmm, X_test_scaled)

            train_bic, train_aic = bic_aic_safe(gmm, X_train_scaled)
            test_bic, test_aic = bic_aic_safe(gmm, X_test_scaled)

            train_labels = compute_labels_from_responsibilities(gmm, X_train_scaled)
            test_labels = compute_labels_from_responsibilities(gmm, X_test_scaled) if X_test_scaled.size else np.array([])

            train_sil = safe_silhouette(X_train_scaled, train_labels) if k > 1 else float("nan")
            train_db = safe_davies_bouldin(X_train_scaled, train_labels) if k > 1 else float("nan")
            test_sil = safe_silhouette(X_test_scaled, test_labels) if k > 1 else float("nan")
            test_db = safe_davies_bouldin(X_test_scaled, test_labels) if k > 1 else float("nan")

            train_entropy = compute_entropy(gmm, X_train_scaled)
            test_entropy = compute_entropy(gmm, X_test_scaled)

            model_id = f"{run_id}_K{k}_cov{cov_type}_shift{int(SHIFT_OHLCV_BY_1)}"

            row = {
                "Model_ID": model_id,
                "SHIFT_OHLCV_BY_1": SHIFT_OHLCV_BY_1,
                "Covariance_Type": cov_type,
                "K": k,
                "Train_Start": TRAIN_START,
                "Train_End": TRAIN_END,
                "Test_Start": TEST_START,
                "Test_End": TEST_END,
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

            results_rows.append(row)
            train_bic_list.append(train_bic)
            test_bic_list.append(test_bic)
            train_ll_list.append(train_avg_ll)
            test_ll_list.append(test_avg_ll)

            print(f"K={k} | Train AvgLogLik={train_avg_ll:.6f} | Train BIC={train_bic:.2f} | Train AIC={train_aic:.2f}")
            print(f"     Test  AvgLogLik={test_avg_ll:.6f} | Test  BIC={test_bic:.2f} | Test  AIC={test_aic:.2f}")
            print(f"     Train Silhouette={train_sil:.4f} | Train DB={train_db:.4f} | Train Entropy={train_entropy:.4f}")
            print(f"     Test  Silhouette={test_sil:.4f} | Test  DB={test_db:.4f} | Test  Entropy={test_entropy:.4f}")
            print()

        plot_metrics(K_LIST, train_bic_list, test_bic_list, train_ll_list, test_ll_list, paths["plots"], cov_type)

    # Select best by minimum TRAIN BIC across all covariance types
    train_bics = np.array([row["Train_BIC"] for row in results_rows], dtype=float)
    best_idx = int(np.nanargmin(train_bics))
    best_k = int(results_rows[best_idx]["K"])
    best_cov_type = str(results_rows[best_idx]["Covariance_Type"])
    best_model = models[(best_cov_type, best_k)]
    best_model_id = results_rows[best_idx]["Model_ID"]

    print("=== Best Model (Overall) ===")
    print(f"Best K by Train BIC: {best_k}")
    print(f"Best covariance type: {best_cov_type}")
    print(f"Model ID: {best_model_id}")
    print()

    # Multi-run variance for best K
    run_ll = []
    run_weights = []
    run_means = []
    run_covs = []

    for i in range(N_RUNS):
        gmm = GaussianMixture(
            n_components=best_k,
            covariance_type=best_cov_type,
            random_state=RANDOM_STATE + i + 1,
            n_init=N_INIT,
        )
        gmm.fit(X_train_scaled)
        run_ll.append(compute_avg_loglik(gmm, X_train_scaled))
        run_weights.append(gmm.weights_.copy())
        run_means.append(gmm.means_.copy())
        covs_full = build_full_covariances(
            gmm.covariances_, gmm.covariance_type, gmm.n_components, gmm.means_.shape[1]
        )
        run_covs.append(covs_full)

    ref_means = run_means[0]
    aligned_weights = []
    aligned_means = []
    aligned_covs = []

    for weights, means, covs in zip(run_weights, run_means, run_covs):
        w_aligned, m_aligned, c_aligned = align_components(ref_means, means, weights, covs)
        aligned_weights.append(w_aligned)
        aligned_means.append(m_aligned)
        aligned_covs.append(c_aligned)

    aligned_weights = np.array(aligned_weights)
    aligned_means = np.array(aligned_means)

    ll_std = float(np.std(run_ll, ddof=1)) if len(run_ll) > 1 else 0.0
    weight_std_per_comp = np.std(aligned_weights, axis=0, ddof=1) if len(run_ll) > 1 else np.zeros(best_k)
    weight_std_mean = float(np.mean(weight_std_per_comp))

    mean_shift_stds = []
    for comp in range(best_k):
        dists = np.linalg.norm(aligned_means[:, comp, :] - ref_means[comp], axis=1)
        mean_shift_stds.append(np.std(dists, ddof=1) if len(dists) > 1 else 0.0)
    mean_shift_std_mean = float(np.mean(mean_shift_stds))

    # Append multi-run variance to best row
    results_rows[best_idx]["MultiRun_LL_Std"] = ll_std
    results_rows[best_idx]["MultiRun_Weight_Std_Mean"] = weight_std_mean
    results_rows[best_idx]["MultiRun_MeanShift_Std_Mean"] = mean_shift_std_mean

    print("=== Multi-run variance (Best K) ===")
    print(f"LL std: {ll_std:.6f}")
    print(f"Weight std mean: {weight_std_mean:.6f}")
    print(f"Mean-shift std mean: {mean_shift_std_mean:.6f}")
    print()

    # Fill missing multi-run fields for non-best rows
    for idx in range(len(results_rows)):
        if "MultiRun_LL_Std" not in results_rows[idx]:
            results_rows[idx]["MultiRun_LL_Std"] = float("nan")
        if "MultiRun_Weight_Std_Mean" not in results_rows[idx]:
            results_rows[idx]["MultiRun_Weight_Std_Mean"] = float("nan")
        if "MultiRun_MeanShift_Std_Mean" not in results_rows[idx]:
            results_rows[idx]["MultiRun_MeanShift_Std_Mean"] = float("nan")

    # Save metrics summary
    metrics_df = pd.DataFrame(results_rows)
    metrics_path = paths["results"] / f"metrics_summary_{run_id}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save best model parameters (original scale)
    means_orig = means_scaled_to_original(best_model.means_, scaler)
    covs_full_scaled = build_full_covariances(
        best_model.covariances_, best_model.covariance_type, best_model.n_components, best_model.means_.shape[1]
    )
    covs_full_orig = cov_scaled_to_original(covs_full_scaled, scaler)

    comp_rows = []
    for i in range(best_model.n_components):
        cov = covs_full_orig[i]
        diag = np.diag(cov)
        comp_rows.append({
            "Model_ID": best_model_id,
            "Component_ID": i,
            "Weight": float(best_model.weights_[i]),
            "Mean_open": float(means_orig[i, 0]),
            "Mean_high": float(means_orig[i, 1]),
            "Mean_low": float(means_orig[i, 2]),
            "Mean_close": float(means_orig[i, 3]),
            "Mean_volume": float(means_orig[i, 4]),
            "Var_open": float(diag[0]),
            "Var_high": float(diag[1]),
            "Var_low": float(diag[2]),
            "Var_close": float(diag[3]),
            "Var_volume": float(diag[4]),
            "Cov_open_close": float(cov[0, 3]),
            "Cov_high_low": float(cov[1, 2]),
            "Cov_close_volume": float(cov[3, 4]),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_path = paths["results"] / f"model_parameters_components_{run_id}.csv"
    comp_df.to_csv(comp_path, index=False)

    # Save best model bundle
    model_bundle = {
        "gmm": best_model,
        "scaler": scaler,
        "config": {
            "input_csv": INPUT_CSV,
            "output_dir": OUTPUT_DIR,
            "shift_ohlcv_by_1": SHIFT_OHLCV_BY_1,
            "k_list": K_LIST,
            "covariance_type": best_cov_type,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "random_state": RANDOM_STATE,
            "n_init": N_INIT,
            "n_runs": N_RUNS,
            "feature_cols": feature_cols,
            "timestamp_col": ts_col,
        },
    }

    model_path = paths["models"] / f"best_gmm_{run_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # Plots
    best_labels = compute_labels_from_responsibilities(best_model, X_train_scaled)
    plot_pca_scatter(X_train_scaled, best_labels, best_model, paths["plots"])

    print("=== Outputs ===")
    print(f"Metrics summary: {metrics_path}")
    print(f"Model parameters: {comp_path}")
    print(f"Best model pickle: {model_path}")
    print(f"Plots dir: {paths['plots']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

