from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from config import (
    CLIP_QUANTILES,
    CORR_THRESHOLD,
    DATA_PATH,
    ENTROPY_MAX_DEFAULT,
    FEATURE_SHIFT,
    FIGURES_DIR,
    GATE_CALIBRATION_METHOD,
    GATE_C_GRID,
    GATE_MAX_ITER,
    GATE_N_SPLITS,
    GATE_SOLVER,
    GMM_COV_TYPE,
    GMM_KS,
    GMM_MAX_ITER,
    GMM_SEEDS,
    MAX_LEAVE_PROB,
    MIN_ACTION_RATE,
    MIN_DURATION,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    MISSINGNESS_MAX,
    OUTPUTS_DIR,
    RANDOM_BASELINE_REPS,
    RANDOM_SEED,
    REPORT_PATH,
    USE_CLIPPER,
    WINDOW_CONFIGS,
    K_LIST,
)
from evaluation import (
    ConfusionStats,
    compute_confusion,
    confusion_matrix_plot_grid,
    confusion_to_array,
    median_metric,
    summarize_baseline,
)
from features_extended import build_features_extended
from gate import compute_gate_diagnostics, predict_trade_prob, train_gate_model
from regimes import predict_gmm, select_gmm_model
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
) -> Tuple[pd.DataFrame, List[str]]:
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
    return features, excluded_cols


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


def build_preprocessor() -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if USE_CLIPPER:
        steps.append(("clipper", QuantileClipper(lower=CLIP_QUANTILES[0], upper=CLIP_QUANTILES[1])))
    steps.append(("scaler", RobustScaler()))
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


def _window_label(window: WindowSlice) -> str:
    return f"{window.config_name}_window_{window.window_id}"


def _select_topk_with_guarantee(
    p_trade: np.ndarray,
    eligible_flags: np.ndarray,
    entropies: np.ndarray,
    k: int,
    entropy_max: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str, int]:
    n = len(p_trade)
    k = min(k, n)
    p_trade_safe = np.where(np.isfinite(p_trade), p_trade, -np.inf)
    stage = "eligible_entropy"
    mask = eligible_flags & (entropies <= entropy_max)
    if mask.sum() < k:
        stage = "eligible_only"
        mask = eligible_flags
    if mask.sum() < k:
        stage = "all_regimes"
        mask = np.ones(n, dtype=bool)
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) < k:
        stage = f"{stage}|fallback_all"
        candidate_idx = np.arange(n)
    if not np.isfinite(p_trade_safe[candidate_idx]).any():
        top_idx = rng.choice(candidate_idx, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        selected = np.zeros(n, dtype=bool)
        selected[top_idx] = True
        return selected, f"{stage}|random", len(candidate_idx)

    order = np.argsort(p_trade_safe[candidate_idx])[::-1]
    top_idx = candidate_idx[order][:k]
    selected = np.zeros(n, dtype=bool)
    selected[top_idx] = True
    return selected, stage, len(candidate_idx)


def _baseline_topk_by_feature(values: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    k = min(k, n)
    values_safe = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(values_safe)[::-1]
    top_idx = order[:k]
    selected = np.zeros(n, dtype=bool)
    selected[top_idx] = True
    return selected


def run() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, timestamp_col, label_col = load_and_clean(DATA_PATH)
    df_raw["label_norm"] = df_raw[label_col].astype(str).str.lower()

    base_features, excluded_cols = build_base_features(
        df_raw, timestamp_col=timestamp_col, label_col=label_col, feature_shift=FEATURE_SHIFT
    )
    extra_features, specs = build_features_extended(df_raw, feature_shift=FEATURE_SHIFT)
    features = pd.concat([base_features, extra_features], axis=1)

    features = drop_missing_columns(features, MISSINGNESS_MAX)
    features = drop_constant_columns(features)
    features = drop_high_corr(features, CORR_THRESHOLD)
    feature_cols = features.columns.tolist()

    valid_mask = features.notna().all(axis=1)
    df = df_raw.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    feature_cols = features.columns.tolist()

    with open(OUTPUTS_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    all_confusion_rows: List[dict] = []
    all_baseline_rows: List[dict] = []
    aggregate_counts: Dict[int, ConfusionStats] = {}
    window_quality_rows: List[dict] = []

    for config in WINDOW_CONFIGS:
        windows = list(
            generate_rolling_windows(
                df, timestamp_col, config, min_train_rows=MIN_TRAIN_ROWS, min_test_rows=MIN_TEST_ROWS
            )
        )
        if not windows:
            continue

        for window in windows:
            window_tag = _window_label(window)
            df_train = df.iloc[window.train_idx].reset_index(drop=True)
            df_test = df.iloc[window.test_idx].reset_index(drop=True)
            X_train = features.iloc[window.train_idx].reset_index(drop=True)
            X_test = features.iloc[window.test_idx].reset_index(drop=True)

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            y_gate_train = df_train["label_norm"].isin(["long", "short"]).astype(int).values
            y_gate_test = df_test["label_norm"].isin(["long", "short"]).astype(int).values

            gate_result = train_gate_model(
                X_train_proc,
                y_gate_train,
                c_grid=GATE_C_GRID,
                calibration_method=GATE_CALIBRATION_METHOD,
                solver=GATE_SOLVER,
                max_iter=GATE_MAX_ITER,
                n_splits=GATE_N_SPLITS,
                random_state=RANDOM_SEED,
            )
            p_trade_train = predict_trade_prob(gate_result.model, X_train_proc)
            p_trade_test = predict_trade_prob(gate_result.model, X_test_proc)

            if p_trade_train is None:
                p_trade_train = np.full(len(df_train), np.nan)
            if p_trade_test is None:
                p_trade_test = np.full(len(df_test), np.nan)

            gate_diag = compute_gate_diagnostics(y_gate_test, p_trade_test)

            gmm_selected = select_gmm_model(
                X_train_proc, ks=GMM_KS, seeds=GMM_SEEDS, n_iter=GMM_MAX_ITER, cov_type=GMM_COV_TYPE
            )
            gmm_states_train, _, _ = predict_gmm(gmm_selected.model, X_train_proc)
            gmm_states_test, _, gmm_entropy_test = predict_gmm(gmm_selected.model, X_test_proc)

            df_train_regime = df_train.copy()
            df_train_regime["regime_id"] = gmm_states_train
            regime_stats = compute_regime_stats(df_train_regime, "regime_id", "label_norm")
            eligibility = map_eligibility(regime_stats, MIN_ACTION_RATE, MIN_DURATION, MAX_LEAVE_PROB)
            eligible_flags = np.array([rid in eligibility.eligible_regimes for rid in gmm_states_test])

            window_quality_rows.append(
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
                    "regime_states": gmm_selected.n_states,
                }
            )

            rng = np.random.default_rng(RANDOM_SEED + window.window_id)
            for k in K_LIST:
                selected, stage, candidate_count = _select_topk_with_guarantee(
                    p_trade=p_trade_test,
                    eligible_flags=eligible_flags,
                    entropies=gmm_entropy_test,
                    k=k,
                    entropy_max=ENTROPY_MAX_DEFAULT,
                    rng=rng,
                )
                y_pred = selected.astype(int)
                stats = compute_confusion(y_gate_test, y_pred)
                aggregate = aggregate_counts.get(k, ConfusionStats(tp=0, fp=0, tn=0, fn=0))
                aggregate_counts[k] = ConfusionStats(
                    tp=aggregate.tp + stats.tp,
                    fp=aggregate.fp + stats.fp,
                    tn=aggregate.tn + stats.tn,
                    fn=aggregate.fn + stats.fn,
                )

                all_confusion_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                        "K": k,
                        "filter_stage": stage,
                        "candidates_after_filters": candidate_count,
                        "trades_selected": int(selected.sum()),
                        "tp": stats.tp,
                        "fp": stats.fp,
                        "tn": stats.tn,
                        "fn": stats.fn,
                        "precision_gate": stats.precision,
                        "recall_gate": stats.recall,
                        "fpr_gate": stats.fpr,
                    }
                )

                n_test = len(df_test)
                k_eff = min(k, n_test)
                random_stats: List[ConfusionStats] = []
                for _ in range(RANDOM_BASELINE_REPS):
                    idx = rng.choice(n_test, size=k_eff, replace=False)
                    y_rand = np.zeros(n_test, dtype=int)
                    y_rand[idx] = 1
                    random_stats.append(compute_confusion(y_gate_test, y_rand))
                baseline_summary = summarize_baseline(random_stats)
                all_baseline_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "K": k,
                        "baseline": "random",
                        **baseline_summary,
                    }
                )

                vol_col = "atr_pct_14" if "atr_pct_14" in X_test.columns else "realized_vol_20"
                if vol_col in X_test.columns:
                    selected_vol = _baseline_topk_by_feature(X_test[vol_col].values, k_eff)
                    stats_vol = compute_confusion(y_gate_test, selected_vol.astype(int))
                    all_baseline_rows.append(
                        {
                            "config_name": window.config_name,
                            "window_id": window.window_id,
                            "K": k,
                            "baseline": f"vol_topk_{vol_col}",
                            "precision_mean": stats_vol.precision,
                            "precision_std": 0.0,
                            "recall_mean": stats_vol.recall,
                            "recall_std": 0.0,
                            "fpr_mean": stats_vol.fpr,
                            "fpr_std": 0.0,
                        }
                    )
                else:
                    all_baseline_rows.append(
                        {
                            "config_name": window.config_name,
                            "window_id": window.window_id,
                            "K": k,
                            "baseline": "vol_topk_missing",
                            "precision_mean": float("nan"),
                            "precision_std": float("nan"),
                            "recall_mean": float("nan"),
                            "recall_std": float("nan"),
                            "fpr_mean": float("nan"),
                            "fpr_std": float("nan"),
                        }
                    )

                y_all = np.ones(n_test, dtype=int)
                stats_all = compute_confusion(y_gate_test, y_all)
                all_baseline_rows.append(
                    {
                        "config_name": window.config_name,
                        "window_id": window.window_id,
                        "K": k,
                        "baseline": "always_trade",
                        "precision_mean": stats_all.precision,
                        "precision_std": 0.0,
                        "recall_mean": stats_all.recall,
                        "recall_std": 0.0,
                        "fpr_mean": stats_all.fpr,
                        "fpr_std": 0.0,
                    }
                )

    confusion_df = pd.DataFrame(all_confusion_rows)
    confusion_df.to_csv(OUTPUTS_DIR / "gate_confusion_by_window.csv", index=False)

    baseline_df = pd.DataFrame(all_baseline_rows)
    baseline_df.to_csv(OUTPUTS_DIR / "baselines_by_window.csv", index=False)

    aggregate_rows = []
    matrices = []
    titles = []
    for k in K_LIST:
        if k not in aggregate_counts:
            continue
        stats = aggregate_counts[k]
        aggregate_rows.append(
            {
                "K": k,
                "tp": stats.tp,
                "fp": stats.fp,
                "tn": stats.tn,
                "fn": stats.fn,
                "precision_gate": stats.precision,
                "recall_gate": stats.recall,
                "fpr_gate": stats.fpr,
            }
        )
        matrices.append(confusion_to_array(stats))
        titles.append(f"K={k}")

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUTS_DIR / "gate_confusion_aggregate.csv", index=False)

    if matrices:
        confusion_matrix_plot_grid(
            matrices=matrices,
            titles=titles,
            labels=["Skip", "Trade"],
            path=str(FIGURES_DIR / "confusion_matrices_gate_K5_K10_K20.png"),
        )

    report_lines = []
    report_lines.append("# Regime-First Rolling Pipeline Report (12h, v7)")
    report_lines.append("")
    report_lines.append("## Data")
    report_lines.append(f"- Rows used: {len(df)}")
    report_lines.append(f"- Features used: {len(feature_cols)}")
    report_lines.append(f"- Label column: {label_col}")
    report_lines.append(f"- Excluded columns: {', '.join(excluded_cols)}")
    report_lines.append("")

    report_lines.append("## Gate Confusion Summary")
    if confusion_df.empty:
        report_lines.append("- No windows produced.")
    else:
        report_lines.append("| K | Median precision | Median recall | Median FPR | Std precision | Aggregate precision |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for k in K_LIST:
            subset = confusion_df[confusion_df["K"] == k]
            if subset.empty:
                continue
            agg = aggregate_df[aggregate_df["K"] == k]
            agg_prec = float(agg["precision_gate"].iloc[0]) if not agg.empty else float("nan")
            report_lines.append(
                f"| {k} | {median_metric(subset, 'precision_gate'):.3f} | "
                f"{median_metric(subset, 'recall_gate'):.3f} | {median_metric(subset, 'fpr_gate'):.3f} | "
                f"{subset['precision_gate'].std():.3f} | {agg_prec:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Baseline Comparison")
    if baseline_df.empty or confusion_df.empty:
        report_lines.append("- Baseline comparison unavailable.")
    else:
        report_lines.append("| K | Gate median precision | Random-K mean precision | Volatility top-K mean precision | Always-trade precision |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for k in K_LIST:
            subset = confusion_df[confusion_df["K"] == k]
            base_k = baseline_df[baseline_df["K"] == k]
            if subset.empty or base_k.empty:
                continue
            random_prec = base_k[base_k["baseline"] == "random"]["precision_mean"].mean()
            vol_rows = base_k[base_k["baseline"].str.startswith("vol_topk")]
            vol_prec = vol_rows["precision_mean"].mean() if not vol_rows.empty else float("nan")
            always_prec = base_k[base_k["baseline"] == "always_trade"]["precision_mean"].mean()
            report_lines.append(
                f"| {k} | {median_metric(subset, 'precision_gate'):.3f} | "
                f"{random_prec:.3f} | {vol_prec:.3f} | {always_prec:.3f} |"
            )
    report_lines.append("")

    report_lines.append("## Stability and Best K")
    if confusion_df.empty or baseline_df.empty:
        report_lines.append("- Not enough data to rank K.")
    else:
        best_k = None
        best_prec = -1.0
        for k in K_LIST:
            subset = confusion_df[confusion_df["K"] == k]
            if subset.empty:
                continue
            med_prec = median_metric(subset, "precision_gate")
            if med_prec > best_prec:
                best_prec = med_prec
                best_k = k
        if best_k is not None:
            report_lines.append(f"- Best K by median precision: {best_k} (median precision={best_prec:.3f}).")

        beat_rows = []
        for k in K_LIST:
            subset = confusion_df[confusion_df["K"] == k].merge(
                baseline_df[baseline_df["K"] == k][["window_id", "baseline", "precision_mean"]],
                on="window_id",
                how="left",
            )
            random_prec = subset[subset["baseline"] == "random"][["window_id", "precision_mean"]].drop_duplicates()
            if random_prec.empty:
                continue
            merged = confusion_df[confusion_df["K"] == k].merge(
                random_prec, on="window_id", how="left", suffixes=("", "_random")
            )
            win_rate = float((merged["precision_gate"] > merged["precision_mean"]).mean())
            beat_rows.append((k, win_rate))
        if beat_rows:
            beat_desc = ", ".join([f"K={k}: {rate:.2f}" for k, rate in beat_rows])
            report_lines.append(f"- Fraction of windows beating random-K precision: {beat_desc}.")
    report_lines.append("")

    report_lines.append("## Conclusion")
    report_lines.append("- Gate evaluation uses top-K trades per window to guarantee trade counts.")
    report_lines.append("- See outputs/gate_confusion_by_window.csv and outputs/baselines_by_window.csv for details.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Outputs saved to:", OUTPUTS_DIR)
    print("Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    run()
