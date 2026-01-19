from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from aggregate import aggregate_global_frontier, pareto_global
from config import (
    ADAPTIVE_DIR_THR_BAD,
    ADAPTIVE_DIR_THR_GOOD,
    ADAPTIVE_LOOKBACK,
    ADAPTIVE_TOPK_BAD,
    ADAPTIVE_TOPK_GOOD,
    ARTIFACTS_DIR,
    CLIP_QUANTILES,
    CONFUSION_DIR,
    CORR_THRESHOLD,
    DATA_PATH,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_ENTROPY_MAX,
    DEFAULT_GATE_QUANTILE,
    DEFAULT_POLICY_TYPE,
    DEFAULT_TOPK_PERCENT,
    DIRECTION_THRESHOLDS,
    DIR_CALIBRATION_METHOD,
    DIR_C_GRID,
    DIR_MAX_ITER,
    DIR_N_SPLITS,
    DIR_SOLVER,
    ENTROPY_MAX_GRID,
    FEATURE_SHIFT,
    FIGURES_DIR,
    GATE_CALIBRATION_METHOD,
    GATE_C_GRID,
    GATE_MAX_ITER,
    GATE_N_SPLITS,
    GATE_QUANTILES,
    GATE_SOLVER,
    GLOBAL_COVERAGE_TARGET,
    GMM_COV_TYPE,
    GMM_KS,
    GMM_MAX_ITER,
    GMM_SEEDS,
    MAX_LEAVE_PROB,
    MIN_ACTION_RATE,
    MIN_DIR_TRAIN_SAMPLES,
    MIN_DURATION,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MIN_TRADES,
    MISSINGNESS_MAX,
    OUTPUTS_DIR,
    PCA_N_COMPONENTS,
    QUALITY_COVERAGE_MAX,
    QUALITY_GATE_AP_MIN,
    QUALITY_WILSON_MIN,
    RANDOM_SEED,
    REPORT_PATH,
    SIGNALS_DIR,
    TOPK_PERCENTS,
    USE_CLIPPER,
    USE_PCA,
    WINDOW_CONFIGS,
)
from direction import predict_direction_prob, train_direction_model
from evaluation import (
    compute_confusion_matrix,
    compute_trade_metrics,
    confusion_matrix_plot,
    correlation_heatmap,
    distribution_plot,
    feature_shift_plot,
    frontier_plot,
    gate_metrics_from_decisions,
    pareto_frontier,
    pr_curve_plot,
    regime_scatter_plot,
    regime_timeline_plot,
    timeline_plot,
    transition_matrix_plot,
    wilson_lower_bound,
)
from features_extended import build_features_extended, get_core_feature_names
from gate import (
    compute_gate_diagnostics,
    gate_thresholds_from_train,
    predict_trade_prob,
    topk_thresholds_from_train,
    train_gate_model,
)
from policy import apply_policy_batch
from regimes_core import select_regime_model
from windows import WindowSlice, generate_rolling_windows


TIMESTAMP_PATTERNS = ("timestamp", "nts", "time", "datetime")
LABEL_CANDIDATES = ("candle_type", "label")
EXCLUDE_PATTERNS = ("future", "target", "lead", "ahead", "t+")


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray, y=None) -> "QuantileClipper":
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("QuantileClipper must be fit before transform.")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


@dataclass
class EligibilityResult:
    stats: pd.DataFrame
    eligible_regimes: set


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lower = name.lower()
    return any(pat in lower for pat in patterns)


def detect_timestamp_column(columns: Iterable[str]) -> str:
    candidates = [col for col in columns if _matches_any(col, TIMESTAMP_PATTERNS)]
    if not candidates:
        raise ValueError("Could not detect timestamp column.")
    return candidates[0]


def detect_label_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if lower in LABEL_CANDIDATES or "candle_type" in lower:
            candidates.append(col)
    if not candidates:
        raise ValueError("Could not detect label column.")
    label_col = candidates[0]
    values = df[label_col].dropna().astype(str).str.lower().unique()
    allowed = {"long", "short", "skip"}
    if not set(values).issubset(allowed):
        raise ValueError(f"Label column {label_col} has unexpected values: {values}")
    return label_col


def load_and_clean(path: Path) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    timestamp_col = detect_timestamp_column(df.columns)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")
    df = df.reset_index(drop=True)
    label_col = detect_label_column(df)
    return df, timestamp_col, label_col


def build_base_features(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_shift: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_cols = [timestamp_col, label_col, "open", "high", "low", "close", "volume"]
    for col in df.columns:
        if col.lower() == "label_ambiguous":
            excluded_cols.append(col)
        if _matches_any(col, EXCLUDE_PATTERNS):
            excluded_cols.append(col)

    excluded_cols = list(dict.fromkeys(excluded_cols))
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]

    features = df[feature_cols].copy()
    if feature_shift > 0:
        features = features.shift(feature_shift)

    return features, feature_cols, excluded_cols


def drop_constant_columns(features: pd.DataFrame) -> pd.DataFrame:
    nunique = features.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    return features[keep_cols]


def drop_missing_columns(features: pd.DataFrame, max_missing: float) -> pd.DataFrame:
    missing_frac = features.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing].index.tolist()
    return features[keep_cols]


def drop_high_corr(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if features.empty:
        return features
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return features.drop(columns=drop_cols)


def build_preprocessor(use_pca: bool, pca_components: float) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if USE_CLIPPER:
        steps.append(("clipper", QuantileClipper(lower=CLIP_QUANTILES[0], upper=CLIP_QUANTILES[1])))
    steps.append(("scaler", RobustScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_SEED)))
    return Pipeline(steps)


def _average_duration(states: Iterable[int], regime_id: int) -> float:
    durations = []
    run = 0
    in_regime = False
    for state in states:
        if state == regime_id:
            run += 1
            in_regime = True
        else:
            if in_regime:
                durations.append(run)
                run = 0
                in_regime = False
    if in_regime:
        durations.append(run)
    return float(np.mean(durations)) if durations else 0.0


def _leave_prob(states: np.ndarray, regime_id: int) -> float:
    leave_flags = []
    n = len(states)
    for i in range(n - 1):
        if states[i] != regime_id:
            continue
        leave_flags.append(1.0 if states[i + 1] != regime_id else 0.0)
    return float(np.mean(leave_flags)) if leave_flags else 1.0


def compute_regime_stats(
    df: pd.DataFrame,
    state_col: str,
    label_col: str,
) -> pd.DataFrame:
    states = df[state_col].values
    regimes = np.unique(states)
    stats_rows: List[dict] = []
    for rid in regimes:
        subset = df[df[state_col] == rid]
        counts = subset[label_col].value_counts()
        n_total = len(subset)
        n_long = int(counts.get("long", 0))
        n_short = int(counts.get("short", 0))
        n_skip = int(counts.get("skip", 0))
        n_action = n_long + n_short
        action_rate = n_action / n_total if n_total else 0.0
        stats_rows.append(
            {
                "regime_id": int(rid),
                "n_total": n_total,
                "n_long": n_long,
                "n_short": n_short,
                "n_skip": n_skip,
                "action_rate": action_rate,
                "avg_duration": _average_duration(states, rid),
                "leave_prob": _leave_prob(states, rid),
            }
        )
    return pd.DataFrame(stats_rows).sort_values("regime_id").reset_index(drop=True)


def map_eligibility(
    stats: pd.DataFrame,
    min_action_rate: float,
    min_duration: float,
    max_leave_prob: float,
) -> EligibilityResult:
    eligible_regimes = set()
    rule_notes = []
    for _, row in stats.iterrows():
        reasons = []
        if row["action_rate"] < min_action_rate:
            reasons.append("action_rate")
        if row["avg_duration"] < min_duration:
            reasons.append("duration")
        if row["leave_prob"] > max_leave_prob:
            reasons.append("leave_prob")
        ok = len(reasons) == 0
        if ok:
            eligible_regimes.add(int(row["regime_id"]))
        rule_notes.append("eligible" if ok else f"excluded:{','.join(reasons)}")

    stats = stats.copy()
    stats["eligible"] = [note == "eligible" for note in rule_notes]
    stats["rule_notes"] = rule_notes
    return EligibilityResult(stats=stats, eligible_regimes=eligible_regimes)


def _transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    mat = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        mat[states[i], states[i + 1]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)
    return mat


def _safe_threshold_lookup(thresholds: Dict, key, fallback: float) -> float:
    if key in thresholds:
        return float(thresholds[key])
    return fallback


def _window_label(window: WindowSlice) -> str:
    return f"{window.config_name}_window_{window.window_id}"


def _evaluate_frontier_window(
    window: WindowSlice,
    df_test: pd.DataFrame,
    eligible_flag: np.ndarray,
    entropies: np.ndarray,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    gate_thresholds: Dict[str, float],
    topk_thresholds: Dict[int, float],
) -> pd.DataFrame:
    rows = []
    y_true_gate = df_test["label_norm"].isin(["long", "short"]).astype(int).values

    for gate_label, gate_thr in gate_thresholds.items():
        for dir_thr in DIRECTION_THRESHOLDS:
            for entropy_max in ENTROPY_MAX_GRID:
                decisions, _ = apply_policy_batch(
                    eligible_flags=eligible_flag,
                    entropies=entropies,
                    p_trade=p_trade,
                    p_long=p_long,
                    p_short=p_short,
                    gate_threshold=gate_thr,
                    direction_threshold=dir_thr,
                    entropy_max=entropy_max,
                )
                df_eval = pd.DataFrame(
                    {"decision": decisions, "label_norm": df_test["label_norm"].values}
                )
                metrics = compute_trade_metrics(df_eval, "decision", "label_norm")
                trade_mask = decisions != "skip"
                precision_gate, _, coverage_gate, _ = gate_metrics_from_decisions(
                    y_true_gate, trade_mask
                )
                rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                        "policy": "threshold",
                        "gate_quantile": gate_label,
                        "top_k_percent": -1,
                        "gate_threshold": gate_thr,
                        "direction_threshold": dir_thr,
                        "entropy_max": entropy_max,
                        "coverage": metrics.coverage,
                        "precision_dir": metrics.precision_dir,
                        "precision_gate": precision_gate,
                        "precision_long": metrics.precision_long,
                        "precision_short": metrics.precision_short,
                        "trade_count": metrics.trade_count,
                        "correct_count": metrics.correct_count,
                        "coverage_gate": coverage_gate,
                    }
                )

    for topk, gate_thr in topk_thresholds.items():
        for dir_thr in DIRECTION_THRESHOLDS:
            for entropy_max in ENTROPY_MAX_GRID:
                decisions, _ = apply_policy_batch(
                    eligible_flags=eligible_flag,
                    entropies=entropies,
                    p_trade=p_trade,
                    p_long=p_long,
                    p_short=p_short,
                    gate_threshold=gate_thr,
                    direction_threshold=dir_thr,
                    entropy_max=entropy_max,
                )
                df_eval = pd.DataFrame(
                    {"decision": decisions, "label_norm": df_test["label_norm"].values}
                )
                metrics = compute_trade_metrics(df_eval, "decision", "label_norm")
                trade_mask = decisions != "skip"
                precision_gate, _, coverage_gate, _ = gate_metrics_from_decisions(
                    y_true_gate, trade_mask
                )
                rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                        "policy": "topk",
                        "gate_quantile": f"topk_{topk}",
                        "top_k_percent": topk,
                        "gate_threshold": gate_thr,
                        "direction_threshold": dir_thr,
                        "entropy_max": entropy_max,
                        "coverage": metrics.coverage,
                        "precision_dir": metrics.precision_dir,
                        "precision_gate": precision_gate,
                        "precision_long": metrics.precision_long,
                        "precision_short": metrics.precision_short,
                        "trade_count": metrics.trade_count,
                        "correct_count": metrics.correct_count,
                        "coverage_gate": coverage_gate,
                    }
                )

    return pd.DataFrame(rows)


def _best_precision_at_coverage(
    frontier: pd.DataFrame,
    coverage_max: float,
) -> Tuple[float, float, int, int, float, str]:
    if frontier.empty:
        return 0.0, 0.0, 0, 0, 0.0, "none"
    pareto = frontier[frontier["pareto"]]
    subset = pareto[pareto["coverage"] <= coverage_max]
    if subset.empty:
        return 0.0, 0.0, 0, 0, 0.0, "none"
    best = subset.sort_values(["precision_dir", "coverage"], ascending=False).iloc[0]
    return (
        float(best["precision_dir"]),
        float(best["precision_gate"]),
        int(best["trade_count"]),
        int(best["correct_count"]),
        float(best["coverage"]),
        f"{best['policy']}|{best['gate_quantile']}|dir{best['direction_threshold']}|ent{best['entropy_max']}",
    )


def _adaptive_policy(prior_labels: List[str]) -> Dict[str, object]:
    if len(prior_labels) < ADAPTIVE_LOOKBACK:
        return {"skip": False, "topk_percent": ADAPTIVE_TOPK_BAD, "direction_threshold": ADAPTIVE_DIR_THR_BAD}
    recent = prior_labels[-ADAPTIVE_LOOKBACK:]
    if all(label == "INSUFFICIENT" for label in recent):
        return {"skip": True, "topk_percent": None, "direction_threshold": None}
    good_count = sum(label == "GOOD" for label in recent)
    if good_count >= 2:
        return {"skip": False, "topk_percent": ADAPTIVE_TOPK_GOOD, "direction_threshold": ADAPTIVE_DIR_THR_GOOD}
    return {"skip": False, "topk_percent": ADAPTIVE_TOPK_BAD, "direction_threshold": ADAPTIVE_DIR_THR_BAD}


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, timestamp_col, label_col = load_and_clean(DATA_PATH)
    df_raw["label_norm"] = df_raw[label_col].astype(str).str.lower()

    base_features, _, excluded_cols = build_base_features(
        df_raw, timestamp_col=timestamp_col, label_col=label_col, feature_shift=FEATURE_SHIFT
    )
    extra_features, specs = build_features_extended(df_raw, feature_shift=FEATURE_SHIFT)
    features = pd.concat([base_features, extra_features], axis=1)

    catalog_df = pd.DataFrame([spec.__dict__ for spec in specs])
    catalog_df.to_csv(OUTPUTS_DIR / "features_added_catalog.csv", index=False)

    features = drop_missing_columns(features, MISSINGNESS_MAX)
    features = drop_constant_columns(features)
    features = drop_high_corr(features, CORR_THRESHOLD)
    feature_cols = features.columns.tolist()

    valid_mask = features.notna().all(axis=1)
    df = df_raw.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    feature_cols = features.columns.tolist()

    refreshed = df[[timestamp_col, label_col]].copy()
    if "label_ambiguous" in df.columns:
        refreshed["label_ambiguous"] = df["label_ambiguous"]
    refreshed = pd.concat([refreshed, features], axis=1)
    refreshed.to_csv(OUTPUTS_DIR / "features_refreshed.csv", index=False)

    with open(ARTIFACTS_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    if len(feature_cols) > 2:
        corr = features[feature_cols].corr()
        corr_subset = corr.iloc[: min(len(corr), 40), : min(len(corr), 40)]
        correlation_heatmap(
            corr_subset,
            "Feature Correlation (subset)",
            FIGURES_DIR / "feature_correlation_heatmap.png",
        )

    all_frontier_rows = []
    all_pareto_rows = []
    all_windows_summary = []
    regime_rows = []
    confusion_gate_total = np.zeros((2, 2), dtype=int)
    confusion_dir_total = np.zeros((2, 2), dtype=int)

    core_features = get_core_feature_names()

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df, timestamp_col, config, min_train_rows=MIN_TRAIN_ROWS, min_test_rows=MIN_TEST_ROWS
            )
        )
        if not windows:
            continue

        prior_labels: List[str] = []
        window_labels = []
        window_starts = []
        feature_shift_stats = []

        for window in windows:
            window_tag = _window_label(window)
            df_train = df.iloc[window.train_idx].copy().reset_index(drop=True)
            df_test = df.iloc[window.test_idx].copy().reset_index(drop=True)
            X_train = features.iloc[window.train_idx].values
            X_test = features.iloc[window.test_idx].values

            regime_selection = select_regime_model(
                X_train,
                X_test,
                feature_cols,
                core_features,
                ks=GMM_KS,
                seeds=GMM_SEEDS,
                n_iter=GMM_MAX_ITER,
                cov_type=GMM_COV_TYPE,
                pca_components=8,
            )
            selected = regime_selection.selected

            gmm_states_train = selected.states_train
            gmm_states_test = selected.states_test
            gmm_ent_test = selected.entropy_test

            joblib.dump(
                {
                    "model": selected.model,
                    "scaler": selected.scaler,
                    "base_scaler": selected.base_scaler,
                    "transformer": selected.transformer,
                    "method": selected.name,
                },
                ARTIFACTS_DIR / f"gmm_model_{window_tag}.joblib",
            )

            for name, option in regime_selection.options.items():
                regime_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "method": name,
                        "selected": name == regime_selection.selected_name,
                        "selection_reason": regime_selection.selection_reason,
                        "silhouette_train": option.diagnostics.silhouette_train,
                        "silhouette_test": option.diagnostics.silhouette_test,
                        "dbi_train": option.diagnostics.dbi_train,
                        "dbi_test": option.diagnostics.dbi_test,
                        "avg_duration_train": option.diagnostics.avg_duration_train,
                        "avg_duration_test": option.diagnostics.avg_duration_test,
                        "entropy_mean_train": option.diagnostics.entropy_mean_train,
                        "entropy_mean_test": option.diagnostics.entropy_mean_test,
                    }
                )

            if selected.X_train_t.shape[1] >= 2:
                regime_scatter_plot(
                    selected.X_train_t,
                    selected.states_train,
                    f"Regime PCA ({window_tag}, {selected.name})",
                    FIGURES_DIR / f"regime_pca_{window_tag}.png",
                )
                try:
                    import umap

                    sample_n = min(len(selected.X_train_t), 2000)
                    embed = umap.UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(
                        selected.X_train_t[:sample_n]
                    )
                    regime_scatter_plot(
                        embed,
                        selected.states_train[:sample_n],
                        f"Regime UMAP ({window_tag}, {selected.name})",
                        FIGURES_DIR / f"regime_umap_{window_tag}.png",
                    )
                except Exception:
                    pass

            if len(gmm_states_test) > 0:
                sample_df = df_test.copy()
                sample_df["regime_id"] = gmm_states_test
                if len(sample_df) > 4000:
                    sample_df = sample_df.tail(4000)
                regime_timeline_plot(
                    sample_df[timestamp_col],
                    sample_df["regime_id"],
                    f"Regime Timeline ({window_tag})",
                    FIGURES_DIR / f"regime_timeline_{window_tag}.png",
                )

            if len(gmm_states_test) > 1:
                n_states = int(np.nanmax(gmm_states_test)) + 1
                trans = _transition_matrix(gmm_states_test.astype(int), n_states)
                transition_matrix_plot(
                    trans,
                    f"Regime Transition ({window_tag})",
                    FIGURES_DIR / f"regime_transition_{window_tag}.png",
                )

            df_train["regime_id"] = gmm_states_train
            stats = compute_regime_stats(df_train, state_col="regime_id", label_col="label_norm")
            eligibility = map_eligibility(
                stats,
                min_action_rate=MIN_ACTION_RATE,
                min_duration=MIN_DURATION,
                max_leave_prob=MAX_LEAVE_PROB,
            )
            eligible_set = eligibility.eligible_regimes

            preprocessor = build_preprocessor(USE_PCA, PCA_N_COMPONENTS)
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)
            joblib.dump(
                {"preprocessor": preprocessor, "feature_columns": feature_cols},
                ARTIFACTS_DIR / f"preprocessor_{window_tag}.joblib",
            )

            y_gate_train = df_train["label_norm"].isin(["long", "short"]).astype(int).values
            gate_model = train_gate_model(
                X_train_t,
                y_gate_train,
                c_grid=GATE_C_GRID,
                calibration_method=GATE_CALIBRATION_METHOD,
                solver=GATE_SOLVER,
                max_iter=GATE_MAX_ITER,
                n_splits=GATE_N_SPLITS,
                random_state=RANDOM_SEED,
            )
            joblib.dump(gate_model.model, ARTIFACTS_DIR / f"gate_model_{window_tag}.joblib")
            p_trade_train = predict_trade_prob(gate_model.model, X_train_t)
            p_trade_test = predict_trade_prob(gate_model.model, X_test_t)
            if p_trade_train is None:
                p_trade_train = np.array([])
            if p_trade_test is None:
                p_trade_test = np.full(len(df_test), np.nan)

            gate_thresholds = gate_thresholds_from_train(p_trade_train, GATE_QUANTILES)
            topk_thresholds = topk_thresholds_from_train(p_trade_train, TOPK_PERCENTS)
            if not gate_thresholds:
                gate_thresholds = {f"q{q}": 1.0 for q in GATE_QUANTILES}
            if not topk_thresholds:
                topk_thresholds = {k: 1.0 for k in TOPK_PERCENTS}

            y_gate_test = df_test["label_norm"].isin(["long", "short"]).astype(int).values
            gate_diag = compute_gate_diagnostics(y_gate_test, p_trade_test)

            pr_curve_plot(
                y_gate_test,
                np.nan_to_num(p_trade_test, nan=0.0),
                f"Gate PR Curve ({window_tag})",
                FIGURES_DIR / f"gate_pr_{window_tag}.png",
            )
            distribution_plot(
                np.nan_to_num(p_trade_test, nan=0.0),
                f"P(trade) Distribution ({window_tag})",
                FIGURES_DIR / f"p_trade_dist_{window_tag}.png",
            )

            df_dir_train = df_train[df_train["label_norm"].isin(["long", "short"])].copy()
            df_dir_train = df_dir_train[df_dir_train["regime_id"].isin(eligible_set)]
            if len(df_dir_train) >= MIN_DIR_TRAIN_SAMPLES:
                y_dir_train = (df_dir_train["label_norm"] == "long").astype(int).values
                X_dir_train = X_train_t[df_dir_train.index]
                dir_model = train_direction_model(
                    X_dir_train,
                    y_dir_train,
                    c_grid=DIR_C_GRID,
                    calibration_method=DIR_CALIBRATION_METHOD,
                    solver=DIR_SOLVER,
                    max_iter=DIR_MAX_ITER,
                    n_splits=DIR_N_SPLITS,
                    random_state=RANDOM_SEED,
                )
                joblib.dump(dir_model.model, ARTIFACTS_DIR / f"direction_model_{window_tag}.joblib")
                dir_probs_test = predict_direction_prob(dir_model.model, X_test_t)
                if dir_probs_test is None:
                    p_long = np.full(len(df_test), np.nan)
                    p_short = np.full(len(df_test), np.nan)
                else:
                    classes = list(dir_model.model.classes_)
                    if 1 in classes and 0 in classes:
                        idx_long = classes.index(1)
                        idx_short = classes.index(0)
                        p_long = dir_probs_test[:, idx_long]
                        p_short = dir_probs_test[:, idx_short]
                    else:
                        p_long = np.full(len(df_test), np.nan)
                        p_short = np.full(len(df_test), np.nan)
            else:
                p_long = np.full(len(df_test), np.nan)
                p_short = np.full(len(df_test), np.nan)

            eligible_flag = np.array([rid in eligible_set for rid in gmm_states_test])

            if DEFAULT_POLICY_TYPE == "topk":
                gate_thr_default = _safe_threshold_lookup(
                    topk_thresholds, DEFAULT_TOPK_PERCENT, fallback=1.0
                )
                default_policy_desc = f"topk_{DEFAULT_TOPK_PERCENT}"
            else:
                gate_thr_default = _safe_threshold_lookup(
                    gate_thresholds, f"q{DEFAULT_GATE_QUANTILE}", fallback=1.0
                )
                default_policy_desc = f"q{DEFAULT_GATE_QUANTILE}"

            decisions_default, reasons_default = apply_policy_batch(
                eligible_flags=eligible_flag,
                entropies=gmm_ent_test,
                p_trade=p_trade_test,
                p_long=p_long,
                p_short=p_short,
                gate_threshold=gate_thr_default,
                direction_threshold=DEFAULT_DIRECTION_THRESHOLD,
                entropy_max=DEFAULT_ENTROPY_MAX,
            )

            adaptive_decision = _adaptive_policy(prior_labels)
            adaptive_basis = "|".join(prior_labels[-ADAPTIVE_LOOKBACK:]) if prior_labels else "none"
            if adaptive_decision["skip"]:
                decisions_adaptive = np.array(["skip"] * len(df_test), dtype=object)
                reasons_adaptive = np.array(["adaptive_skip"] * len(df_test), dtype=object)
                adaptive_gate_thr = float("nan")
                adaptive_dir_thr = float("nan")
                adaptive_topk = float("nan")
                adaptive_policy_desc = "skip"
            else:
                adaptive_topk = int(adaptive_decision["topk_percent"])
                adaptive_dir_thr = float(adaptive_decision["direction_threshold"])
                adaptive_gate_thr = _safe_threshold_lookup(topk_thresholds, adaptive_topk, fallback=1.0)
                decisions_adaptive, reasons_adaptive = apply_policy_batch(
                    eligible_flags=eligible_flag,
                    entropies=gmm_ent_test,
                    p_trade=p_trade_test,
                    p_long=p_long,
                    p_short=p_short,
                    gate_threshold=adaptive_gate_thr,
                    direction_threshold=adaptive_dir_thr,
                    entropy_max=DEFAULT_ENTROPY_MAX,
                )
                adaptive_policy_desc = f"topk_{adaptive_topk}"

            signals = pd.DataFrame(
                {
                    "timestamp": df_test[timestamp_col].values,
                    "true_label": df_test[label_col].values,
                    "decision_default": decisions_default,
                    "decision_adaptive": decisions_adaptive,
                    "decision_default_reason": reasons_default,
                    "decision_adaptive_reason": reasons_adaptive,
                    "policy_default": default_policy_desc,
                    "policy_adaptive": adaptive_policy_desc,
                    "gate_threshold_default": gate_thr_default,
                    "gate_threshold_adaptive": adaptive_gate_thr,
                    "direction_threshold_default": DEFAULT_DIRECTION_THRESHOLD,
                    "direction_threshold_adaptive": adaptive_dir_thr,
                    "entropy_max_default": DEFAULT_ENTROPY_MAX,
                    "entropy_max_adaptive": DEFAULT_ENTROPY_MAX,
                    "adaptive_basis": adaptive_basis,
                    "p_trade": p_trade_test,
                    "p_long": p_long,
                    "p_short": p_short,
                    "regime_id": gmm_states_test,
                    "entropy": gmm_ent_test,
                    "eligible_flag": eligible_flag,
                    "config_name": window.config_name,
                    "window_id": window.window_id,
                }
            )
            signals.to_csv(SIGNALS_DIR / f"signals_{window_tag}.csv", index=False)

            window_frontier = _evaluate_frontier_window(
                window,
                df_test,
                eligible_flag,
                gmm_ent_test,
                p_trade_test,
                p_long,
                p_short,
                gate_thresholds,
                topk_thresholds,
            )
            window_frontier["pareto"] = pareto_frontier(window_frontier, "coverage", "precision_dir")
            frontier_plot(
                window_frontier,
                f"Frontier ({window_tag})",
                FIGURES_DIR / f"frontier_{window_tag}.png",
            )

            all_frontier_rows.append(window_frontier)
            all_pareto_rows.append(window_frontier[window_frontier["pareto"]])

            best_precision, best_gate_precision, best_trades, best_correct, best_cov, best_policy = (
                _best_precision_at_coverage(window_frontier, QUALITY_COVERAGE_MAX)
            )
            wilson_lb = wilson_lower_bound(best_correct, best_trades)
            if best_trades < MIN_TRADES:
                quality_label = "INSUFFICIENT"
                quality_reason = "insufficient_trades"
            else:
                reasons = []
                if gate_diag.ap < QUALITY_GATE_AP_MIN:
                    reasons.append("gate_AP")
                if wilson_lb < QUALITY_WILSON_MIN:
                    reasons.append("wilson")
                if len(reasons) == 0:
                    quality_label = "GOOD"
                    quality_reason = "ok"
                else:
                    quality_label = "BAD"
                    quality_reason = f"fail:{','.join(reasons)}"

            prior_labels.append(quality_label)
            window_labels.append(quality_label)
            window_starts.append(window.train_start)

            y_gate_pred_default = (
                eligible_flag
                & (gmm_ent_test <= DEFAULT_ENTROPY_MAX)
                & np.isfinite(p_trade_test)
                & (p_trade_test >= gate_thr_default)
            ).astype(int)
            cm_gate = compute_confusion_matrix(y_gate_test, y_gate_pred_default, labels=[0, 1])
            confusion_gate_total += cm_gate
            confusion_matrix_plot(
                cm_gate,
                ["Skip", "Trade"],
                f"Gate Confusion ({window_tag})",
                FIGURES_DIR / f"gate_confusion_{window_tag}.png",
            )
            pd.DataFrame(cm_gate, index=["true_skip", "true_trade"], columns=["pred_skip", "pred_trade"]).to_csv(
                CONFUSION_DIR / f"gate_cm_{window_tag}.csv"
            )

            trade_mask = decisions_default != "skip"
            if trade_mask.any():
                y_true_dir = df_test.loc[trade_mask, "label_norm"]
                y_pred_dir = pd.Series(decisions_default[trade_mask], index=y_true_dir.index)
                valid = y_true_dir.isin(["long", "short"])
                y_true_bin = (y_true_dir[valid] == "long").astype(int).values
                y_pred_bin = (y_pred_dir[valid] == "long").astype(int).values
                cm_dir = compute_confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            else:
                cm_dir = np.zeros((2, 2), dtype=int)
            confusion_dir_total += cm_dir
            confusion_matrix_plot(
                cm_dir,
                ["Short", "Long"],
                f"Direction Confusion ({window_tag})",
                FIGURES_DIR / f"direction_confusion_{window_tag}.png",
            )
            pd.DataFrame(cm_dir, index=["true_short", "true_long"], columns=["pred_short", "pred_long"]).to_csv(
                CONFUSION_DIR / f"direction_cm_{window_tag}.csv"
            )

            all_windows_summary.append(
                {
                    "config_name": window.config_name,
                    "window_id": window.window_id,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "train_rows": len(df_train),
                    "test_rows": len(df_test),
                    "gate_ap": gate_diag.ap,
                    "gate_brier": gate_diag.brier,
                    "p_trade_min": gate_diag.p_trade_stats["min"],
                    "p_trade_p25": gate_diag.p_trade_stats["p25"],
                    "p_trade_p50": gate_diag.p_trade_stats["p50"],
                    "p_trade_p75": gate_diag.p_trade_stats["p75"],
                    "p_trade_max": gate_diag.p_trade_stats["max"],
                    "best_precision_dir_5pct": best_precision,
                    "best_precision_gate_5pct": best_gate_precision,
                    "best_trade_count_5pct": best_trades,
                    "best_correct_count_5pct": best_correct,
                    "best_coverage_5pct": best_cov,
                    "wilson_lower_95": wilson_lb,
                    "best_policy_5pct": best_policy,
                    "quality_label": quality_label,
                    "quality_reason": quality_reason,
                    "regime_method": selected.name,
                }
            )

            shift_features = [f for f in core_features if f in features.columns][:4]
            shift_stats = {"window_id": window.window_id}
            test_feat_slice = features.iloc[window.test_idx]
            for feat in shift_features:
                shift_stats[feat] = float(test_feat_slice[feat].median())
            feature_shift_stats.append(shift_stats)

        if window_starts:
            timeline_plot(
                window_starts,
                window_labels,
                f"Window Quality Timeline ({config.name})",
                FIGURES_DIR / f"window_quality_{config.name}.png",
            )

        if feature_shift_stats:
            shift_df = pd.DataFrame(feature_shift_stats)
            shift_features = [f for f in core_features if f in shift_df.columns][:4]
            if shift_features:
                feature_shift_plot(
                    shift_df,
                    shift_features,
                    f"Feature Shift ({config.name})",
                    FIGURES_DIR / f"feature_shift_{config.name}.png",
                )

    if all_frontier_rows:
        frontier_all = pd.concat(all_frontier_rows, ignore_index=True)
        frontier_all.to_csv(OUTPUTS_DIR / "frontier_per_window.csv", index=False)
    else:
        frontier_all = pd.DataFrame()

    if all_pareto_rows:
        pareto_all = pd.concat(all_pareto_rows, ignore_index=True)
        pareto_all.to_csv(OUTPUTS_DIR / "frontier_pareto_per_window.csv", index=False)
    else:
        pareto_all = pd.DataFrame()

    windows_summary_df = pd.DataFrame(all_windows_summary)
    if not windows_summary_df.empty:
        windows_summary_df.to_csv(OUTPUTS_DIR / "windows_summary.csv", index=False)

    if regime_rows:
        regime_df = pd.DataFrame(regime_rows)
        regime_df.to_csv(OUTPUTS_DIR / "regime_separability.csv", index=False)

    pd.DataFrame(
        confusion_gate_total, index=["true_skip", "true_trade"], columns=["pred_skip", "pred_trade"]
    ).to_csv(CONFUSION_DIR / "gate_cm_aggregate.csv")
    pd.DataFrame(
        confusion_dir_total, index=["true_short", "true_long"], columns=["pred_short", "pred_long"]
    ).to_csv(CONFUSION_DIR / "direction_cm_aggregate.csv")
    confusion_matrix_plot(
        confusion_gate_total,
        ["Skip", "Trade"],
        "Gate Confusion (Aggregate)",
        FIGURES_DIR / "gate_confusion_aggregate.png",
    )
    confusion_matrix_plot(
        confusion_dir_total,
        ["Short", "Long"],
        "Direction Confusion (Aggregate)",
        FIGURES_DIR / "direction_confusion_aggregate.png",
    )

    global_frontier = None
    if not frontier_all.empty:
        group_cols = [
            "config_name",
            "policy",
            "gate_quantile",
            "top_k_percent",
            "direction_threshold",
            "entropy_max",
        ]
        global_frontier = aggregate_global_frontier(frontier_all, group_cols)
        global_frontier["pareto"] = pareto_global(
            global_frontier, x_col="mean_coverage", y_col="p10_precision_dir"
        )
        global_frontier.to_csv(OUTPUTS_DIR / "frontier_global.csv", index=False)
        global_frontier_pareto = global_frontier[global_frontier["pareto"]].copy()
        global_frontier_pareto.to_csv(OUTPUTS_DIR / "frontier_global_pareto.csv", index=False)

        default_config = WINDOW_CONFIGS[0].name
        global_best = global_frontier[global_frontier["config_name"] == default_config].copy()
        if not global_best.empty:
            global_best["coverage"] = global_best["mean_coverage"]
            global_best["precision_dir"] = global_best["p10_precision_dir"]
            global_best["pareto"] = pareto_global(global_best, x_col="coverage", y_col="precision_dir")
            frontier_plot(
                global_best,
                f"Global Frontier (p10 vs coverage, {default_config})",
                FIGURES_DIR / "global_frontier.png",
            )

    v5_metrics = None
    v5_csv = OUTPUTS_DIR.parents[1].parent / "regime_pipeline_12h_rolling_v5" / "outputs" / "windows_summary.csv"
    if v5_csv.exists():
        v5_df = pd.read_csv(v5_csv)
        v5_metrics = {
            "median_gate_ap": float(np.nanmedian(v5_df.get("gate_ap", np.array([np.nan])))),
            "p10_gate_ap": float(np.nanpercentile(v5_df.get("gate_ap", np.array([np.nan])), 10)),
            "median_precision": float(
                np.nanmedian(v5_df.get("best_precision_dir_5pct", np.array([np.nan])))
            ),
            "pct_windows_trades": float(
                (v5_df.get("best_trade_count_5pct", pd.Series(dtype=float)) >= MIN_TRADES).mean()
            ),
        }

    report_lines = []
    report_lines.append("# Regime-First Rolling Pipeline Report (12h, v6)")
    report_lines.append("")
    report_lines.append("## Data")
    report_lines.append(f"- Rows used: {len(df)}")
    report_lines.append(f"- Features used: {len(feature_cols)}")
    report_lines.append(f"- Label column: {label_col}")
    report_lines.append(f"- Excluded columns: {', '.join(excluded_cols)}")
    report_lines.append("")

    report_lines.append("## Feature Engineering")
    report_lines.append(f"- Added features: {len(catalog_df)}")
    report_lines.append(f"- Final features after pruning: {len(feature_cols)}")
    report_lines.append("- Catalog saved to outputs/features_added_catalog.csv")
    report_lines.append("")

    report_lines.append("## Regime Separability")
    if regime_rows:
        regime_df = pd.DataFrame(regime_rows)
        selected_counts = regime_df[regime_df["selected"]].groupby("method").size().to_dict()
        report_lines.append(
            f"- Selected method counts: core={selected_counts.get('core', 0)}, pca={selected_counts.get('pca', 0)}"
        )
        report_lines.append("- Diagnostics saved to outputs/regime_separability.csv")
    else:
        report_lines.append("- No regime diagnostics produced.")
    report_lines.append("")

    report_lines.append("## Window Summary (by config)")
    if windows_summary_df.empty:
        report_lines.append("- No valid windows generated.")
        summary_df = pd.DataFrame()
    else:
        summary_rows = []
        for config_name, group in windows_summary_df.groupby("config_name"):
            median_gate_ap = float(np.nanmedian(group["gate_ap"]))
            median_best_precision = float(np.nanmedian(group["best_precision_dir_5pct"]))
            std_precision = float(np.nanstd(group["best_precision_dir_5pct"]))
            pct_windows_trades = float((group["best_trade_count_5pct"] >= MIN_TRADES).mean())
            median_trades = float(np.nanmedian(group["best_trade_count_5pct"]))
            summary_rows.append(
                {
                    "config_name": config_name,
                    "median_gate_ap": median_gate_ap,
                    "median_precision_dir_5pct": median_best_precision,
                    "pct_windows_with_trades": pct_windows_trades,
                    "median_trades": median_trades,
                    "stability_std": std_precision,
                }
            )
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["median_trades", "median_precision_dir_5pct"], ascending=[False, False]
        )
        report_lines.append("| Window | Median gate AP | Median precision<=5% | % windows w/ trades | Median trades | Stability (std) |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in summary_df.iterrows():
            report_lines.append(
                f"| {row['config_name']} | {row['median_gate_ap']:.3f} | "
                f"{row['median_precision_dir_5pct']:.3f} | {row['pct_windows_with_trades']:.2f} | "
                f"{row['median_trades']:.1f} | {row['stability_std']:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Gate Calibration Diagnostics")
    if windows_summary_df.empty:
        report_lines.append("- No gate diagnostics available.")
    else:
        report_lines.append(
            f"- Median gate AP: {windows_summary_df['gate_ap'].median():.3f}"
        )
        report_lines.append(
            f"- Median gate Brier: {windows_summary_df['gate_brier'].median():.3f}"
        )
    report_lines.append("")

    report_lines.append("## Comparison vs v5")
    if v5_metrics is None or windows_summary_df.empty:
        report_lines.append("- v5 comparison unavailable.")
    else:
        v6_gate_ap = float(np.nanmedian(windows_summary_df["gate_ap"]))
        v6_gate_p10 = float(np.nanpercentile(windows_summary_df["gate_ap"], 10))
        v6_precision = float(np.nanmedian(windows_summary_df["best_precision_dir_5pct"]))
        v6_pct_trades = float((windows_summary_df["best_trade_count_5pct"] >= MIN_TRADES).mean())
        report_lines.append(
            f"- Gate AP median: v5={v5_metrics['median_gate_ap']:.3f}, v6={v6_gate_ap:.3f} "
            f"(p10 v5={v5_metrics['p10_gate_ap']:.3f}, v6={v6_gate_p10:.3f})"
        )
        report_lines.append(
            f"- Precision<=5% median: v5={v5_metrics['median_precision']:.3f}, v6={v6_precision:.3f}"
        )
        report_lines.append(
            f"- % windows with >= {MIN_TRADES} trades: v5={v5_metrics['pct_windows_trades']:.2f}, "
            f"v6={v6_pct_trades:.2f}"
        )
    report_lines.append("")

    report_lines.append("## Frontier Interpretation")
    if global_frontier is None or global_frontier.empty:
        report_lines.append("- Global frontier unavailable.")
    else:
        default_config = WINDOW_CONFIGS[0].name
        candidate = global_frontier[
            (global_frontier["config_name"] == default_config)
            & (global_frontier["mean_coverage"] <= GLOBAL_COVERAGE_TARGET)
        ]
        if candidate.empty:
            report_lines.append("- No strategy meets the 5% mean coverage target.")
        else:
            best = candidate.sort_values(
                ["p10_precision_dir", "median_precision_dir"], ascending=False
            ).iloc[0]
            report_lines.append(
                f"- Best p10 policy: {best['policy']} {best['gate_quantile']} "
                f"dir_thr={best['direction_threshold']} entropy_max={best['entropy_max']} "
                f"(p10={best['p10_precision_dir']:.3f}, median={best['median_precision_dir']:.3f}, "
                f"coverage={best['mean_coverage']:.3f})."
            )
    report_lines.append("")

    report_lines.append("## Conclusion")
    report_lines.append(
        "- Directional signal at 12h remains weak; improvements mainly target gate ranking and regime separation."
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
