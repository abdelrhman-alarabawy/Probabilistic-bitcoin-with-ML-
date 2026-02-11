from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from .folds import FoldDefinition


@dataclass
class FoldFeatureData:
    train_matrix: np.ndarray
    test_matrix: np.ndarray
    selected_features: List[str]
    candidate_features: List[str]
    dropped_missing_features: List[str]
    dropped_low_variance_features: List[str]


def _matches_pattern(name: str, patterns: Sequence[str]) -> bool:
    lower = name.lower()
    return any(pattern.lower() in lower for pattern in patterns)


def infer_base_feature_columns(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    label_column_patterns: Sequence[str],
    drop_ohlcv_columns: bool,
    ohlcv_columns: Sequence[str],
) -> List[str]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    exclusions = set()
    if timestamp_col:
        exclusions.add(timestamp_col.lower())
    if drop_ohlcv_columns:
        exclusions.update({col.lower() for col in ohlcv_columns})

    feature_cols: List[str] = []
    for col in numeric_cols:
        col_lower = col.lower()
        if col_lower in exclusions:
            continue
        if _matches_pattern(col, label_column_patterns):
            continue
        feature_cols.append(col)
    return feature_cols


def apply_feature_shift(df: pd.DataFrame, feature_cols: Sequence[str], shift: int) -> pd.DataFrame:
    if shift <= 0:
        return df
    df = df.copy()
    df.loc[:, feature_cols] = df.loc[:, feature_cols].shift(shift)
    return df


def _sanitize_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    feat_df = df.loc[:, feature_cols].copy()
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    return feat_df


def _drop_low_quality_features(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    missing_threshold: float,
    near_constant_std_threshold: float,
) -> Tuple[List[str], List[str], List[str]]:
    kept: List[str] = []
    dropped_missing: List[str] = []
    dropped_low_variance: List[str] = []

    for col in feature_cols:
        s = train_df[col]
        if float(s.isna().mean()) > missing_threshold:
            dropped_missing.append(col)
            continue
        unique_non_na = s.dropna().nunique()
        if unique_non_na <= 1:
            dropped_low_variance.append(col)
            continue
        std_value = float(np.nanstd(s.to_numpy(dtype=float)))
        if not np.isfinite(std_value) or std_value <= near_constant_std_threshold:
            dropped_low_variance.append(col)
            continue
        kept.append(col)
    return kept, dropped_missing, dropped_low_variance


def _robust_variance(series: pd.Series) -> float:
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    median = float(np.nanmedian(arr))
    arr = np.where(np.isnan(arr), median, arr)
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return float(iqr * iqr)


def _select_features_variance_prune(
    train_df: pd.DataFrame,
    candidate_features: Sequence[str],
    top_n_features: int,
    variance_top_m: int,
    corr_prune_threshold: float,
) -> List[str]:
    variance_scores = {col: _robust_variance(train_df[col]) for col in candidate_features}
    ordered = sorted(candidate_features, key=lambda c: variance_scores[c], reverse=True)
    top_pool = ordered[: max(top_n_features, min(variance_top_m, len(ordered)))]

    filled = train_df.loc[:, top_pool].copy()
    for col in top_pool:
        median_val = float(filled[col].median(skipna=True))
        if not np.isfinite(median_val):
            median_val = 0.0
        filled[col] = filled[col].fillna(median_val)
    corr = filled.corr(method="spearman").abs()

    selected: List[str] = []
    for col in top_pool:
        if not selected:
            selected.append(col)
        else:
            max_corr = max(float(corr.loc[col, existing]) for existing in selected)
            if max_corr <= corr_prune_threshold:
                selected.append(col)
        if len(selected) >= top_n_features:
            break

    if len(selected) < top_n_features:
        for col in ordered:
            if col not in selected:
                selected.append(col)
            if len(selected) >= top_n_features:
                break

    return selected


def _build_scaler(name: str):
    lower = name.lower()
    if lower == "robust":
        return RobustScaler()
    if lower == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler '{name}'. Use 'robust' or 'standard'.")


def _select_features_pseudo_mi(
    train_df: pd.DataFrame,
    candidate_features: Sequence[str],
    top_n_features: int,
    scaler_name: str,
    pseudo_mi_cfg: Dict[str, object],
) -> List[str]:
    if len(candidate_features) <= top_n_features:
        return list(candidate_features)

    imputer = SimpleImputer(strategy="median")
    scaler = _build_scaler(scaler_name)
    X = train_df.loc[:, candidate_features].to_numpy(dtype=float)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    n_samples = X_scaled.shape[0]
    if n_samples < 3:
        return list(candidate_features)[:top_n_features]

    pseudo_k = int(pseudo_mi_cfg.get("pseudo_mi_k", 3))
    pseudo_k = max(2, min(pseudo_k, max(2, min(8, n_samples - 1))))
    random_state = int(pseudo_mi_cfg.get("pseudo_mi_random_state", 7))
    n_init = int(pseudo_mi_cfg.get("pseudo_mi_n_init", 1))
    max_iter = int(pseudo_mi_cfg.get("pseudo_mi_max_iter", 200))

    try:
        pseudo_model = GaussianMixture(
            n_components=pseudo_k,
            covariance_type="tied",
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=1e-6,
            init_params="kmeans",
        )
        pseudo_labels = pseudo_model.fit_predict(X_scaled)
        if len(np.unique(pseudo_labels)) < 2:
            raise ValueError("Pseudo-labels collapsed into one cluster.")
        mi = mutual_info_classif(X_scaled, pseudo_labels, discrete_features=False, random_state=random_state)
        scores = {candidate_features[i]: float(mi[i]) for i in range(len(candidate_features))}
        ordered = sorted(candidate_features, key=lambda c: scores[c], reverse=True)
        return ordered[:top_n_features]
    except Exception as exc:
        logging.warning("Pseudo-MI feature selection failed (%s). Falling back to variance ordering.", exc)
        fallback = sorted(
            candidate_features,
            key=lambda c: float(np.nanstd(train_df[c].to_numpy(dtype=float))),
            reverse=True,
        )
        return fallback[:top_n_features]


def select_features_for_fold(
    train_feature_df: pd.DataFrame,
    candidate_features: Sequence[str],
    features_cfg: Dict[str, object],
) -> List[str]:
    mode = str(features_cfg.get("mode", "selected_top10")).lower()
    if mode == "all_features":
        return list(candidate_features)

    selector_method = str(features_cfg.get("selector_method", "variance_prune")).lower()
    top_n = int(features_cfg.get("top_n_features", 10))
    if len(candidate_features) <= top_n:
        return list(candidate_features)

    if selector_method == "variance_prune":
        return _select_features_variance_prune(
            train_df=train_feature_df,
            candidate_features=candidate_features,
            top_n_features=top_n,
            variance_top_m=int(features_cfg.get("variance_top_m", 30)),
            corr_prune_threshold=float(features_cfg.get("corr_prune_threshold", 0.9)),
        )
    if selector_method == "pseudo_mi":
        return _select_features_pseudo_mi(
            train_df=train_feature_df,
            candidate_features=candidate_features,
            top_n_features=top_n,
            scaler_name=str(features_cfg.get("scaler", "robust")),
            pseudo_mi_cfg=features_cfg,
        )
    raise ValueError(f"Unsupported selector_method '{selector_method}'.")


def prepare_fold_features(
    df: pd.DataFrame,
    fold: FoldDefinition,
    base_feature_columns: Sequence[str],
    data_cfg: Dict[str, object],
    features_cfg: Dict[str, object],
) -> FoldFeatureData:
    train_df_raw = _sanitize_feature_frame(df.iloc[fold.train_idx], base_feature_columns)
    test_df_raw = _sanitize_feature_frame(df.iloc[fold.test_idx], base_feature_columns)

    candidate_features, dropped_missing, dropped_low_variance = _drop_low_quality_features(
        train_df=train_df_raw,
        feature_cols=base_feature_columns,
        missing_threshold=float(data_cfg.get("missing_col_threshold", 0.5)),
        near_constant_std_threshold=float(data_cfg.get("near_constant_std_threshold", 1e-12)),
    )
    if not candidate_features:
        raise ValueError("No candidate features remained after quality filtering.")

    selected_features = select_features_for_fold(
        train_feature_df=train_df_raw,
        candidate_features=candidate_features,
        features_cfg=features_cfg,
    )
    if not selected_features:
        raise ValueError("Feature selection produced an empty feature list.")

    imputer = SimpleImputer(strategy="median")
    scaler = _build_scaler(str(features_cfg.get("scaler", "robust")))

    X_train = train_df_raw.loc[:, selected_features].to_numpy(dtype=float)
    X_test = test_df_raw.loc[:, selected_features].to_numpy(dtype=float)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Defensive fallback for rare scaler edge cases.
    X_train = np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return FoldFeatureData(
        train_matrix=X_train,
        test_matrix=X_test,
        selected_features=list(selected_features),
        candidate_features=list(candidate_features),
        dropped_missing_features=dropped_missing,
        dropped_low_variance_features=dropped_low_variance,
    )

