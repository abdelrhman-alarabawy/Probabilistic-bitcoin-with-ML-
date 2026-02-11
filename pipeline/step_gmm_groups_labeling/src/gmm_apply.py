from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler


EPS = 1e-12


@dataclass
class FitResult:
    responsibilities: np.ndarray
    hard_states: np.ndarray
    probmax: np.ndarray
    entropy: np.ndarray
    feature_columns_used: List[str]
    training_rows: int
    prediction_rows: int
    fit_converged: bool
    fit_n_iter: int
    train_start: Optional[str]
    train_end: Optional[str]


def _build_scaler(name: str):
    lower = str(name).lower()
    if lower == "robust":
        return RobustScaler()
    if lower == "standard":
        return StandardScaler()
    if lower in {"none", "identity"}:
        return None
    raise ValueError(f"Unsupported scaler: {name}")


def _coerce_int(value: object, default: int) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: object, default: float) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def infer_indicator_feature_columns(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    ohlcv_columns: Sequence[str],
    label_patterns: Sequence[str],
) -> List[str]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    excluded = {c.lower() for c in ohlcv_columns}
    if timestamp_col:
        excluded.add(timestamp_col.lower())

    features: List[str] = []
    for col in numeric_cols:
        lc = col.lower()
        if lc in excluded:
            continue
        if any(pattern.lower() in lc for pattern in label_patterns):
            continue
        features.append(col)
    return features


def resolve_feature_list_for_config(
    df: pd.DataFrame,
    config_row: pd.Series,
    selected_features_fallback: Optional[List[str]],
    timestamp_col: Optional[str],
    label_patterns: Sequence[str],
) -> List[str]:
    requested = []
    if "selected_features_signature" in config_row.index:
        text = str(config_row.get("selected_features_signature", "")).strip()
        if text and text.lower() != "nan":
            requested = [s for s in text.split("|") if s]
    if not requested and selected_features_fallback:
        requested = list(selected_features_fallback)

    existing = set(df.columns)
    requested = [col for col in requested if col in existing]
    if requested:
        return requested

    return infer_indicator_feature_columns(
        df=df,
        timestamp_col=timestamp_col,
        ohlcv_columns=["open", "high", "low", "close", "volume"],
        label_patterns=label_patterns,
    )


def apply_shift(df: pd.DataFrame, feature_cols: Sequence[str], shift: int) -> pd.DataFrame:
    out = df.copy()
    if shift > 0 and feature_cols:
        out.loc[:, feature_cols] = out.loc[:, feature_cols].shift(shift)
    return out


def _build_training_mask(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    row: pd.Series,
    mode: str,
    holdout_train_ratio: float,
) -> np.ndarray:
    n = df.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    if mode == "fit_all_predict_all":
        return np.ones(n, dtype=bool)

    if timestamp_col is not None and "train_start" in row.index and "train_end" in row.index:
        train_start_raw = row.get("train_start")
        train_end_raw = row.get("train_end")
        if pd.notna(train_start_raw) and pd.notna(train_end_raw):
            ts = pd.to_datetime(df[timestamp_col], errors="coerce")
            start_ts = pd.to_datetime(train_start_raw, errors="coerce")
            end_ts = pd.to_datetime(train_end_raw, errors="coerce")
            if pd.notna(start_ts) and pd.notna(end_ts):
                mask = (ts >= start_ts) & (ts <= end_ts)
                if mask.sum() >= 10:
                    return mask.to_numpy(dtype=bool)

    cut = int(round(n * holdout_train_ratio))
    cut = max(1, min(cut, n - 1))
    mask = np.zeros(n, dtype=bool)
    mask[:cut] = True
    return mask


def fit_predict_gmm(
    df: pd.DataFrame,
    config_row: pd.Series,
    feature_cols: Sequence[str],
    mode: str,
    shift: int,
    scaler_name: str,
    holdout_train_ratio: float,
    timestamp_col: Optional[str],
    gmm_defaults: Dict[str, object],
    random_state_fallback: int,
) -> FitResult:
    if not feature_cols:
        raise ValueError("No feature columns available for GMM.")

    working = apply_shift(df, feature_cols, shift=shift)
    feat_df = working.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    train_mask = _build_training_mask(
        df=working,
        timestamp_col=timestamp_col,
        row=config_row,
        mode=mode,
        holdout_train_ratio=holdout_train_ratio,
    )
    if train_mask.size == 0:
        raise ValueError("Training mask is empty.")

    X_all = feat_df.to_numpy(dtype=float)
    X_train = X_all[train_mask]

    keep_idx = []
    for idx, col in enumerate(feature_cols):
        col_train = X_train[:, idx]
        if np.all(np.isnan(col_train)):
            continue
        keep_idx.append(idx)
    if not keep_idx:
        raise ValueError("All selected features are NaN on training rows after shift.")

    final_features = [feature_cols[i] for i in keep_idx]
    X_all = X_all[:, keep_idx]
    X_train = X_train[:, keep_idx]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_all_imp = imputer.transform(X_all)

    scaler = _build_scaler(scaler_name)
    if scaler is not None:
        X_train_proc = scaler.fit_transform(X_train_imp)
        X_all_proc = scaler.transform(X_all_imp)
    else:
        X_train_proc = X_train_imp
        X_all_proc = X_all_imp

    X_train_proc = np.nan_to_num(X_train_proc, nan=0.0, posinf=0.0, neginf=0.0)
    X_all_proc = np.nan_to_num(X_all_proc, nan=0.0, posinf=0.0, neginf=0.0)

    n_components = _coerce_int(config_row.get("n_components"), int(gmm_defaults.get("n_components_default", 2)))
    covariance_type = str(config_row.get("covariance_type", gmm_defaults.get("covariance_type_default", "tied"))).lower()
    reg_covar = _coerce_float(config_row.get("reg_covar"), float(gmm_defaults.get("reg_covar_default", 1e-6)))
    n_init = _coerce_int(config_row.get("n_init"), int(gmm_defaults.get("n_init", 5)))
    max_iter = _coerce_int(config_row.get("max_iter"), int(gmm_defaults.get("max_iter", 500)))
    tol = _coerce_float(config_row.get("tol"), float(gmm_defaults.get("tol", 1e-3)))
    init_params = str(config_row.get("init_params", gmm_defaults.get("init_params", "kmeans")))
    random_state = _coerce_int(config_row.get("seed"), random_state_fallback)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        init_params=init_params,
        random_state=random_state,
    )
    model.fit(X_train_proc)
    resp = model.predict_proba(X_all_proc)
    hard = np.argmax(resp, axis=1).astype(int)
    probmax = np.max(resp, axis=1)
    entropy = -np.sum(resp * np.log(resp + EPS), axis=1)

    train_idx = np.flatnonzero(train_mask)
    train_start = None
    train_end = None
    if timestamp_col is not None and train_idx.size > 0:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        train_start = str(ts.iloc[train_idx[0]])
        train_end = str(ts.iloc[train_idx[-1]])

    return FitResult(
        responsibilities=resp,
        hard_states=hard,
        probmax=probmax,
        entropy=entropy,
        feature_columns_used=final_features,
        training_rows=int(X_train_proc.shape[0]),
        prediction_rows=int(X_all_proc.shape[0]),
        fit_converged=bool(getattr(model, "converged_", False)),
        fit_n_iter=int(getattr(model, "n_iter_", -1)),
        train_start=train_start,
        train_end=train_end,
    )
