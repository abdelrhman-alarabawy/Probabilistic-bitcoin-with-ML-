from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from regimes import predict_gmm, select_gmm_model


@dataclass
class RegimeDiagnostics:
    silhouette_train: float
    silhouette_test: float
    dbi_train: float
    dbi_test: float
    avg_duration_train: float
    avg_duration_test: float
    entropy_mean_train: float
    entropy_mean_test: float


@dataclass
class RegimeOptionResult:
    name: str
    model: object
    scaler: RobustScaler
    transformer: Optional[object]
    base_scaler: Optional[RobustScaler]
    X_train_t: np.ndarray
    X_test_t: np.ndarray
    states_train: np.ndarray
    states_test: np.ndarray
    entropy_train: np.ndarray
    entropy_test: np.ndarray
    diagnostics: RegimeDiagnostics


@dataclass
class RegimeSelectionResult:
    selected_name: str
    selected: RegimeOptionResult
    options: Dict[str, RegimeOptionResult]
    selection_reason: str


def _average_duration(states: np.ndarray) -> float:
    if len(states) == 0:
        return 0.0
    runs = []
    run_len = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return float(np.mean(runs)) if runs else 0.0


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def _dbi_safe(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) < 2:
        return float("nan")
    return float(davies_bouldin_score(X, labels))


def _build_option(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    ks: List[int],
    seeds: List[int],
    n_iter: int,
    cov_type: str,
    scaler: Optional[RobustScaler] = None,
    transformer: Optional[object] = None,
    pre_scaled: bool = False,
) -> RegimeOptionResult:
    if pre_scaled:
        X_train_t = X_train
        X_test_t = X_test
        scaler_out = scaler
    else:
        scaler_out = RobustScaler() if scaler is None else scaler
        X_train_t = scaler_out.fit_transform(X_train)
        X_test_t = scaler_out.transform(X_test)

    gmm_result = select_gmm_model(X_train_t, ks, seeds, n_iter, cov_type)
    states_train, post_train, ent_train = predict_gmm(gmm_result.model, X_train_t)
    states_test, post_test, ent_test = predict_gmm(gmm_result.model, X_test_t)

    diagnostics = RegimeDiagnostics(
        silhouette_train=_silhouette_safe(X_train_t, states_train),
        silhouette_test=_silhouette_safe(X_test_t, states_test),
        dbi_train=_dbi_safe(X_train_t, states_train),
        dbi_test=_dbi_safe(X_test_t, states_test),
        avg_duration_train=_average_duration(states_train),
        avg_duration_test=_average_duration(states_test),
        entropy_mean_train=float(np.nanmean(ent_train)),
        entropy_mean_test=float(np.nanmean(ent_test)),
    )

    return RegimeOptionResult(
        name=name,
        model=gmm_result.model,
        scaler=scaler_out,
        transformer=transformer,
        base_scaler=None,
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        states_train=states_train,
        states_test=states_test,
        entropy_train=ent_train,
        entropy_test=ent_test,
        diagnostics=diagnostics,
    )


def select_regime_model(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    feature_cols: List[str],
    core_features: List[str],
    ks: List[int],
    seeds: List[int],
    n_iter: int,
    cov_type: str,
    pca_components: int,
) -> RegimeSelectionResult:
    core_cols = [col for col in core_features if col in feature_cols]
    core_indices = [feature_cols.index(col) for col in core_cols]
    if len(core_indices) < 3:
        core_indices = list(range(min(len(feature_cols), 10)))

    X_train_core = X_train_full[:, core_indices]
    X_test_core = X_test_full[:, core_indices]
    core_option = _build_option("core", X_train_core, X_test_core, ks, seeds, n_iter, cov_type)

    scaler_full = RobustScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_full)
    X_test_scaled = scaler_full.transform(X_test_full)
    n_components = min(pca_components, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_inner_scaler = RobustScaler()
    X_train_pca_scaled = pca_inner_scaler.fit_transform(X_train_pca)
    X_test_pca_scaled = pca_inner_scaler.transform(X_test_pca)
    pca_option = _build_option(
        "pca",
        X_train_pca_scaled,
        X_test_pca_scaled,
        ks,
        seeds,
        n_iter,
        cov_type,
        scaler=pca_inner_scaler,
        transformer=pca,
        pre_scaled=True,
    )
    pca_option.scaler = pca_inner_scaler
    pca_option.transformer = pca
    pca_option.base_scaler = scaler_full

    options = {"core": core_option, "pca": pca_option}

    sil_core = core_option.diagnostics.silhouette_train
    sil_pca = pca_option.diagnostics.silhouette_train
    dbi_core = core_option.diagnostics.dbi_train
    dbi_pca = pca_option.diagnostics.dbi_train

    selection_reason = "silhouette"
    if np.isnan(sil_core) and np.isnan(sil_pca):
        selected = pca_option
        selection_reason = "fallback_pca"
    elif np.isnan(sil_pca) or (sil_core > sil_pca):
        selected = core_option
    elif np.isnan(sil_core) or (sil_pca > sil_core):
        selected = pca_option
    else:
        selection_reason = "dbi_tiebreak"
        selected = core_option if dbi_core <= dbi_pca else pca_option

    return RegimeSelectionResult(
        selected_name=selected.name,
        selected=selected,
        options=options,
        selection_reason=selection_reason,
    )
