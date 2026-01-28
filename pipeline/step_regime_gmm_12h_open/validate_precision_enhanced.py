from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# === CONFIGURATION ===
INPUT_CSV = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\12h_features_indicators_with_ohlcv.csv"
)
RESULTS_DIR = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\pipeline\step_regime_gmm_12h_open\results"
)
OUTPUT_DIR = RESULTS_DIR / "validation" / "precision_enhanced"

TP_POINTS = 2000
SL_POINTS = 1000
FEE_PER_TRADE = 0.0005
PROB_THRESHOLD = 0.8
HORIZON_N_CANDLES = 1
RANDOM_SEED = 42

TRADABLE_REGIMES = {0, 2, 5, 6}
SKIP_REGIMES = {1, 3, 4, 7}

DEFAULT_5M_PATH = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\BTCUSDT_5m_2026-01-03.csv"
)

POLICY_CONFIG = {
    "gate_only": {
        "regime_whitelist": sorted(TRADABLE_REGIMES),
        "prob_threshold": PROB_THRESHOLD,
        "entropy_max": None,
    },
    "low_regime_whitelist_clusters": {
        "regime_whitelist": [0, 5],
        "prob_threshold": 0.75,
        "entropy_max": None,
    },
    "baseline_experts": {
        "regime_whitelist": sorted(TRADABLE_REGIMES),
        "prob_threshold": 0.90,
        "entropy_max": 0.01,
        "regime_prob_thresholds": {0: 0.90, 2: 0.92, 5: 0.90, 6: 0.88},
    },
    "weighted_vote": {
        "regime_whitelist": sorted(TRADABLE_REGIMES),
        "prob_threshold": 0.85,
        "entropy_max": 0.005,
        "direction_bias": {0: "long", 2: "long", 5: "short", 6: "short"},
    },
    "agreement": {
        "regime_whitelist": [0, 2],
        "prob_threshold": 0.90,
        "entropy_max": 0.02,
    },
}

POLICY_REQUIRED_KEYS = {
    "gate_only": ["regime_whitelist", "prob_threshold", "entropy_max"],
    "low_regime_whitelist_clusters": ["regime_whitelist", "prob_threshold", "entropy_max"],
    "baseline_experts": ["regime_whitelist", "prob_threshold", "entropy_max", "regime_prob_thresholds"],
    "weighted_vote": ["regime_whitelist", "prob_threshold", "entropy_max", "direction_bias"],
    "agreement": ["regime_whitelist", "prob_threshold", "entropy_max"],
}


def parse_timestamp(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return parse_numeric_timestamp(series)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().mean() > 0.05:
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().any():
            parsed = parse_numeric_timestamp(as_num)
    return parsed


def parse_numeric_timestamp(series: pd.Series) -> pd.Series:
    max_val = series.max()
    if max_val > 1e12:
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    if max_val > 1e10:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def resolve_timestamp_col(columns: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in ["timestamp", "datetime", "time", "open_time", "ts_utc"]:
        if candidate in lowered:
            return lowered[candidate]
    return None


def compute_horizon_extremes(
    high: np.ndarray, low: np.ndarray, horizon_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    horizon_high = np.full(len(high), np.nan, dtype=float)
    horizon_low = np.full(len(low), np.nan, dtype=float)
    if horizon_n <= 0:
        return horizon_high, horizon_low
    for i in range(len(high)):
        end = i + horizon_n
        if end >= len(high):
            break
        window_high = high[i + 1 : end + 1]
        window_low = low[i + 1 : end + 1]
        if np.all(np.isnan(window_high)) or np.all(np.isnan(window_low)):
            continue
        horizon_high[i] = np.nanmax(window_high)
        horizon_low[i] = np.nanmin(window_low)
    return horizon_high, horizon_low


def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    ts_col = resolve_timestamp_col(df.columns)
    if ts_col is None:
        raise ValueError("timestamp column not found in input CSV.")
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df = df[df["timestamp"].notna()].copy()
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def find_gmm_outputs(results_dir: Path) -> Optional[Path]:
    candidates = [
        results_dir / "gmm_regimes_per_row.csv",
        results_dir / "gmm_per_row.csv",
        results_dir / "preds_per_row.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_gmm_outputs(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    gmm_path = find_gmm_outputs(results_dir)
    if gmm_path is not None:
        gmm_df = pd.read_csv(gmm_path)
        gmm_df.columns = gmm_df.columns.str.strip()
        ts_col = resolve_timestamp_col(gmm_df.columns)
        if ts_col is None:
            raise ValueError(f"timestamp column not found in {gmm_path}")
        if ts_col != "timestamp":
            gmm_df = gmm_df.rename(columns={ts_col: "timestamp"})
        gmm_df["timestamp"] = parse_timestamp(gmm_df["timestamp"])
        gmm_df = gmm_df[gmm_df["timestamp"].notna()].copy()
        gmm_df = gmm_df.sort_values("timestamp").drop_duplicates(subset="timestamp")

        if "regime_id" not in gmm_df.columns and "regime" in gmm_df.columns:
            gmm_df = gmm_df.rename(columns={"regime": "regime_id"})
        if "regime_prob_max" not in gmm_df.columns and "prob_max" in gmm_df.columns:
            gmm_df = gmm_df.rename(columns={"prob_max": "regime_prob_max"})
        if "regime_prob_max" not in gmm_df.columns:
            if "max_prob" in gmm_df.columns:
                gmm_df = gmm_df.rename(columns={"max_prob": "regime_prob_max"})
            else:
                prob_cols = [
                    c
                    for c in gmm_df.columns
                    if c.lower().startswith("prob_") or c.lower().startswith("regime_prob_")
                ]
                prob_cols = [c for c in prob_cols if c.lower() not in {"prob_max", "regime_prob_max"}]
                if prob_cols:
                    gmm_df["regime_prob_max"] = gmm_df[prob_cols].max(axis=1)
                else:
                    raise ValueError("No probability column found in GMM outputs.")

        required = ["regime_id", "regime_prob_max"]
        missing = [c for c in required if c not in gmm_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in GMM outputs: {missing}")

        merge_cols = ["timestamp", "regime_id", "regime_prob_max"]
        if "entropy" in gmm_df.columns:
            merge_cols.append("entropy")
        merged = pd.merge(df, gmm_df[merge_cols], on="timestamp", how="inner")
        if merged.empty:
            raise ValueError("Merge with GMM outputs produced empty dataframe.")
        return merged

    model_path = results_dir / "gmm_model.joblib"
    scaler_path = results_dir / "scaler.joblib"
    imputer_path = results_dir / "imputer.joblib"
    features_path = results_dir / "selected_features.json"
    if not (model_path.exists() and scaler_path.exists() and imputer_path.exists() and features_path.exists()):
        raise FileNotFoundError(
            "No GMM per-row outputs found and model artifacts missing; cannot compute regimes."
        )

    with features_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    feature_cols = config.get("selected_features", [])
    shift_periods = int(config.get("shift_periods", 1))

    if not feature_cols:
        raise ValueError("selected_features.json is missing selected_features.")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Selected feature missing from data: {col}")

    X_raw = df[feature_cols].copy()
    for col in feature_cols:
        if not np.issubdtype(X_raw[col].dtype, np.number):
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    X_raw = X_raw.shift(shift_periods)
    df_shifted = df.iloc[shift_periods:].copy()
    X_raw = X_raw.iloc[shift_periods:]

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    gmm = joblib.load(model_path)

    X_imputed = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    probs = gmm.predict_proba(X_scaled)
    regime_id = probs.argmax(axis=1)
    regime_prob_max = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    preds = pd.DataFrame(
        {
            "timestamp": df_shifted["timestamp"].values,
            "regime_id": regime_id,
            "regime_prob_max": regime_prob_max,
            "entropy": entropy,
        }
    )
    merged = pd.merge(df, preds, on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("Merge with computed GMM predictions produced empty dataframe.")
    return merged


class LowerTFIndex:
    def __init__(self, df: pd.DataFrame, ts_col: str, high_col: str, low_col: str) -> None:
        self.index = pd.DatetimeIndex(df[ts_col])
        self.high = df[high_col].to_numpy()
        self.low = df[low_col].to_numpy()

    def iter_slice(self, start: pd.Timestamp, end: pd.Timestamp):
        left = self.index.searchsorted(start, side="left")
        right = self.index.searchsorted(end, side="left")
        for i in range(left, right):
            yield self.high[i], self.low[i]


def load_lower_tf(path: Optional[Path]) -> Optional[LowerTFIndex]:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ts_col = resolve_timestamp_col(df.columns)
    if ts_col is None or "high" not in df.columns or "low" not in df.columns:
        return None
    df["timestamp"] = parse_timestamp(df[ts_col])
    df = df[df["timestamp"].notna()].sort_values("timestamp")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    return LowerTFIndex(df, "timestamp", "high", "low")


def resolve_with_5m(
    lower_tf: LowerTFIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    long_tp: float,
    long_sl: float,
    short_tp: float,
    short_sl: float,
    ambiguous_type: str,
) -> str:
    for hi, lo in lower_tf.iter_slice(start, end):
        if ambiguous_type in {"long", "both"}:
            hit_long_tp = hi >= long_tp
            hit_long_sl = lo <= long_sl
            if hit_long_tp and hit_long_sl:
                return "both_hit_unresolved"
            if hit_long_tp:
                return "long"
            if hit_long_sl:
                return "sl_first"
        if ambiguous_type in {"short", "both"}:
            hit_short_tp = lo <= short_tp
            hit_short_sl = hi >= short_sl
            if hit_short_tp and hit_short_sl:
                return "both_hit_unresolved"
            if hit_short_tp:
                return "short"
            if hit_short_sl:
                return "sl_first"
    return "both_hit_unresolved"


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, labels: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics_df = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=labels
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    summary_df = pd.DataFrame(
        {
            "precision": [macro[0], weighted[0]],
            "recall": [macro[1], weighted[1]],
            "f1": [macro[2], weighted[2]],
            "support": [support.sum(), support.sum()],
        },
        index=["macro_avg", "weighted_avg"],
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return cm_df, pd.concat([metrics_df, summary_df])


def trade_only_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mask: pd.Series,
    labels: List[str],
    drop_pred: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]
    valid = y_true_sub.isin(labels) & y_pred_sub.isin(labels)
    dropped = len(y_true_sub) - int(valid.sum())
    if valid.any():
        cm_df, metrics_df = compute_metrics(y_true_sub[valid], y_pred_sub[valid], labels)
    else:
        cm_df = pd.DataFrame(0, index=labels, columns=labels)
        metrics_df = pd.DataFrame(
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}, index=labels
        )
        metrics_df.loc["macro_avg"] = [0.0, 0.0, 0.0, 0]
        metrics_df.loc["weighted_avg"] = [0.0, 0.0, 0.0, 0]
    return cm_df, metrics_df, dropped


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if pnl.empty:
        return np.nan
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def validate_policy_config() -> None:
    missing_policies = [p for p in POLICY_REQUIRED_KEYS if p not in POLICY_CONFIG]
    missing_keys = {
        policy: [key for key in POLICY_REQUIRED_KEYS[policy] if key not in POLICY_CONFIG.get(policy, {})]
        for policy in POLICY_REQUIRED_KEYS
        if policy in POLICY_CONFIG
    }
    missing_keys = {policy: keys for policy, keys in missing_keys.items() if keys}
    if missing_policies or missing_keys:
        details = []
        if missing_policies:
            details.append(f"missing policies: {missing_policies}")
        if missing_keys:
            details.append(f"missing keys: {missing_keys}")
        raise ValueError(f"Policy configuration incomplete: {', '.join(details)}")


def jaccard_similarity(a: pd.Series, b: pd.Series) -> float:
    a_bool = a.astype(bool).to_numpy()
    b_bool = b.astype(bool).to_numpy()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a_bool, b_bool).sum()
    return float(inter / union)


def build_policies(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, object]]]:
    validate_policy_config()
    policies: Dict[str, pd.Series] = {}
    policy_meta: Dict[str, Dict[str, object]] = {}

    for policy_name, cfg in POLICY_CONFIG.items():
        whitelist = set(cfg["regime_whitelist"])
        base_mask = df["regime_id"].isin(whitelist)

        if "regime_prob_thresholds" in cfg:
            per_regime = df["regime_id"].map(cfg["regime_prob_thresholds"]).fillna(
                cfg["prob_threshold"]
            )
            prob_mask = df["regime_prob_max"] >= per_regime
        else:
            prob_mask = df["regime_prob_max"] >= cfg["prob_threshold"]

        entropy_mask = pd.Series(True, index=df.index)
        if cfg.get("entropy_max") is not None:
            if "entropy" not in df.columns:
                raise ValueError(f"Policy '{policy_name}' requires entropy column.")
            entropy_mask = df["entropy"] <= cfg["entropy_max"]

        in_coverage = base_mask & prob_mask & entropy_mask & df["regime_prob_max"].notna()
        policies[policy_name] = in_coverage

        policy_meta[policy_name] = {
            "whitelist": whitelist,
            "prob_threshold": cfg["prob_threshold"],
            "entropy_max": cfg.get("entropy_max"),
            "direction_bias": cfg.get("direction_bias"),
            "regime_prob_thresholds": cfg.get("regime_prob_thresholds"),
            "base_mask": base_mask,
            "prob_mask": prob_mask,
            "entropy_mask": entropy_mask,
        }

    return policies, policy_meta


def classify_dropped_rows(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    in_coverage: pd.Series,
    mask: pd.Series,
) -> pd.Series:
    reasons = pd.Series("dropped_due_to_other", index=df.index)
    missing_mask = df["missing_regime"] | df["regime_prob_max"].isna()
    reasons[missing_mask] = "dropped_due_to_nan_features_or_missing_regime"

    ambiguous_skip = df["ambiguous_original"] & (df["final_label"] == "skip")
    ambiguous_mask = ambiguous_skip & (reasons == "dropped_due_to_other")
    reasons[ambiguous_mask] = "dropped_due_to_ambiguous_to_skip"

    out_cov_mask = (~in_coverage) & (reasons == "dropped_due_to_other")
    reasons[out_cov_mask] = "dropped_due_to_out_of_coverage"

    label_mask = (
        (~y_true.isin(["long", "short"])) | (~y_pred.isin(["long", "short"]))
    ) & (reasons == "dropped_due_to_other")
    reasons[label_mask] = "dropped_due_to_label_not_in_LS"

    reasons = reasons.where(mask, "not_dropped")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced precision/confusion validation.")
    parser.add_argument("--fallback_policy", choices=["skip", "loss"], default="skip")
    parser.add_argument("--use_5m", choices=["true", "false"], default="true")
    parser.add_argument("--min_warn_trades", type=int, default=50)
    args = parser.parse_args()

    use_5m = args.use_5m.lower() == "true"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy_debug_dir = OUTPUT_DIR / "policy_debug"
    policy_debug_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_CSV)
    df = load_gmm_outputs(df, RESULTS_DIR)

    df["regime_id"] = pd.to_numeric(df["regime_id"], errors="coerce")
    df["missing_regime"] = df["regime_id"].isna()
    df["regime_id"] = df["regime_id"].fillna(-1).astype(int)
    df["regime_prob_max"] = pd.to_numeric(df["regime_prob_max"], errors="coerce")
    df["missing_prob"] = df["regime_prob_max"].isna()
    if "entropy" in df.columns:
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    else:
        df["entropy"] = np.nan

    horizon_high, horizon_low = compute_horizon_extremes(
        df["high"].to_numpy(), df["low"].to_numpy(), HORIZON_N_CANDLES
    )
    df["horizon_high"] = horizon_high
    df["horizon_low"] = horizon_low

    entry_open = df["open"].to_numpy()
    long_tp = entry_open + TP_POINTS
    long_sl = entry_open - SL_POINTS
    short_tp = entry_open - TP_POINTS
    short_sl = entry_open + SL_POINTS

    long_tp_hit = horizon_high >= long_tp
    long_sl_hit = horizon_low <= long_sl
    short_tp_hit = horizon_low <= short_tp
    short_sl_hit = horizon_high >= short_sl

    long_clean = long_tp_hit & ~long_sl_hit
    short_clean = short_tp_hit & ~short_sl_hit

    ambiguous_long = long_tp_hit & long_sl_hit
    ambiguous_short = short_tp_hit & short_sl_hit
    ambiguous_any = ambiguous_long | ambiguous_short

    base_label = np.full(len(df), "skip", dtype=object)
    base_label[long_clean] = "long"
    base_label[short_clean] = "short"
    resolved_outcome = np.where(
        base_label == "long",
        "clean_long",
        np.where(base_label == "short", "clean_short", "skip"),
    )

    lower_tf = load_lower_tf(DEFAULT_5M_PATH) if use_5m else None
    candle_delta = df["timestamp"].diff().median()
    if pd.isna(candle_delta):
        candle_delta = pd.Timedelta(hours=12)
    horizon_end = df["timestamp"] + candle_delta * HORIZON_N_CANDLES

    df["ambiguous_original"] = ambiguous_any
    final_label = base_label.copy()
    audit_rows: List[Dict[str, object]] = []
    resolved_5m = 0
    resolved_fallback = 0

    for idx in np.where(ambiguous_any)[0]:
        amb_long = bool(ambiguous_long[idx])
        amb_short = bool(ambiguous_short[idx])
        ambiguous_type = "both" if amb_long and amb_short else "long" if amb_long else "short"

        resolved_outcome_value = "both_hit_unresolved"
        resolved_by = "fallback"
        label = "skip"

        if lower_tf is not None and use_5m:
            resolved = resolve_with_5m(
                lower_tf,
                df.loc[idx, "timestamp"],
                horizon_end.iloc[idx],
                long_tp[idx],
                long_sl[idx],
                short_tp[idx],
                short_sl[idx],
                ambiguous_type,
            )
            resolved_by = "5m"
            if resolved == "long":
                label = "long"
                resolved_outcome_value = "win"
            elif resolved == "short":
                label = "short"
                resolved_outcome_value = "win"
            elif resolved == "sl_first":
                if args.fallback_policy == "loss" and ambiguous_type in {"long", "short"}:
                    label = ambiguous_type
                    resolved_outcome_value = "loss"
                else:
                    label = "skip"
                    resolved_outcome_value = "skip"
            else:
                resolved_outcome_value = "both_hit_unresolved"
        else:
            if args.fallback_policy == "loss" and ambiguous_type in {"long", "short"}:
                label = ambiguous_type
                resolved_outcome_value = "loss"
            else:
                label = "skip"
                resolved_outcome_value = "skip"

        if resolved_by == "5m":
            resolved_5m += 1
        else:
            resolved_fallback += 1

        final_label[idx] = label
        resolved_outcome[idx] = resolved_outcome_value

        tp_price = long_tp[idx] if ambiguous_type in {"long", "both"} else short_tp[idx]
        sl_price = long_sl[idx] if ambiguous_type in {"long", "both"} else short_sl[idx]
        tp_hit = int(long_tp_hit[idx] if ambiguous_type in {"long", "both"} else short_tp_hit[idx])
        sl_hit = int(long_sl_hit[idx] if ambiguous_type in {"long", "both"} else short_sl_hit[idx])

        audit_rows.append(
            {
                "timestamp": df.loc[idx, "timestamp"],
                "entry_open": entry_open[idx],
                "tp": tp_price,
                "sl": sl_price,
                "horizon_high": horizon_high[idx],
                "horizon_low": horizon_low[idx],
                "tp_hit": tp_hit,
                "sl_hit": sl_hit,
                "resolved_outcome": resolved_outcome_value,
                "resolved_by": resolved_by,
                "fallback_policy": args.fallback_policy,
                "ambiguous_type": ambiguous_type,
            }
        )

    df["final_label"] = final_label
    df["resolved_outcome"] = resolved_outcome
    df["trade_flag"] = df["final_label"].isin(["long", "short"]).astype(int)
    df["side"] = df["final_label"].where(df["final_label"].isin(["long", "short"]))

    ambiguous_rate = float(ambiguous_any.mean())
    ambiguous_handled_5m = resolved_5m / ambiguous_any.sum() if ambiguous_any.any() else 0.0
    ambiguous_handled_fallback = resolved_fallback / ambiguous_any.sum() if ambiguous_any.any() else 0.0

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_DIR / "ambiguous_resolution_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    policies, policy_meta = build_policies(df)
    modes = ["oracle", "random"]

    summary_rows = []
    regime_rows = []
    sanity_lines: List[str] = []
    sanity_lines.append("SANITY REPORT")
    sanity_lines.append("")

    rng = np.random.RandomState(RANDOM_SEED)
    y_true = df["final_label"].astype(str)

    policy_predictions: Dict[str, Dict[str, pd.Series]] = {}
    in_coverage_masks: Dict[str, pd.Series] = {}
    trade_pred_masks: Dict[str, pd.Series] = {}

    for policy_name, in_coverage in policies.items():
        policy_seed = rng.randint(0, 2**31 - 1)
        y_pred_oracle = pd.Series(np.where(in_coverage, y_true, "skip"), index=df.index)

        coverage_labels = y_true[in_coverage]
        priors = coverage_labels.value_counts(normalize=True)
        probs = np.array(
            [priors.get("long", 0.0), priors.get("short", 0.0), priors.get("skip", 0.0)]
        )
        if probs.sum() == 0:
            probs = np.array([0.0, 0.0, 1.0])
        probs = probs / probs.sum()
        rng_policy = np.random.RandomState(policy_seed)
        samples = rng_policy.choice(
            ["long", "short", "skip"], size=int(in_coverage.sum()), p=probs
        )
        y_pred_random = pd.Series("skip", index=df.index)
        y_pred_random.loc[in_coverage] = samples

        policy_predictions[policy_name] = {
            "oracle": y_pred_oracle,
            "random": y_pred_random,
            "in_coverage": in_coverage,
            "seed": policy_seed,
        }

        in_coverage_masks[policy_name] = in_coverage
        trade_pred_masks[policy_name] = y_pred_oracle != "skip"

        coverage_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "regime_id": df["regime_id"],
                "prob_max": df["regime_prob_max"],
                "entropy": df["entropy"],
                "in_coverage": in_coverage,
            }
        )
        coverage_df.to_csv(policy_debug_dir / f"in_coverage_{policy_name}.csv", index=False)

        trade_mask_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "y_true": y_true,
                "y_pred_oracle": y_pred_oracle,
                "y_pred_random": y_pred_random,
                "trade_pred": y_pred_oracle != "skip",
                "trade_pred_oracle": y_pred_oracle != "skip",
                "trade_pred_random": y_pred_random != "skip",
                "trade_true": y_true != "skip",
                "in_coverage": in_coverage,
            }
        )
        trade_mask_df.to_csv(policy_debug_dir / f"trade_mask_{policy_name}.csv", index=False)

    policy_names = list(in_coverage_masks.keys())
    overlap_in_cov = pd.DataFrame(index=policy_names, columns=policy_names, dtype=float)
    overlap_trade = pd.DataFrame(index=policy_names, columns=policy_names, dtype=float)
    for i, p1 in enumerate(policy_names):
        for j, p2 in enumerate(policy_names):
            overlap_in_cov.loc[p1, p2] = jaccard_similarity(
                in_coverage_masks[p1], in_coverage_masks[p2]
            )
            overlap_trade.loc[p1, p2] = jaccard_similarity(
                trade_pred_masks[p1], trade_pred_masks[p2]
            )
            if i < j and overlap_in_cov.loc[p1, p2] >= 0.99:
                msg = f"WARNING: policies {p1} and {p2} are effectively identical (in_coverage)."
                print(msg)
                sanity_lines.append(msg)
            if i < j and overlap_trade.loc[p1, p2] >= 0.99:
                msg = f"WARNING: policies {p1} and {p2} are effectively identical (trade_pred oracle)."
                print(msg)
                sanity_lines.append(msg)
    overlap_in_cov.to_csv(policy_debug_dir / "policy_overlap_in_coverage.csv")
    overlap_trade.to_csv(policy_debug_dir / "policy_overlap_trade_mask.csv")

    if "agreement" in policy_meta:
        meta = policy_meta["agreement"]
        sanity_lines.append("")
        sanity_lines.append("Agreement gate diagnostics:")
        sanity_lines.append(f"- total_rows: {len(df)}")
        sanity_lines.append(f"- regime_pass: {int(meta['base_mask'].sum())}")
        sanity_lines.append(f"- prob_pass: {int(meta['prob_mask'].sum())}")
        sanity_lines.append(f"- entropy_pass: {int(meta['entropy_mask'].sum())}")
        sanity_lines.append(f"- combined_pass: {int(policies['agreement'].sum())}")

    for policy_name, policy_data in policy_predictions.items():
        in_coverage = policy_data["in_coverage"]
        for mode in modes:
            y_pred = policy_data[mode]

            cm_full, metrics_full = compute_metrics(y_true, y_pred, ["long", "short", "skip"])
            cm_full.to_csv(OUTPUT_DIR / f"confusion_matrix_full_{mode}_{policy_name}.csv")

            true_trade_mask = y_true != "skip"
            cm_true, metrics_true, dropped_true = trade_only_metrics(
                y_true, y_pred, true_trade_mask, ["long", "short"], drop_pred=True
            )
            cm_true.to_csv(
                OUTPUT_DIR / f"confusion_matrix_tradeonly_trueTrades_{mode}_{policy_name}.csv"
            )

            pred_trade_mask = y_pred != "skip"
            cm_pred, metrics_pred, dropped_pred = trade_only_metrics(
                y_true, y_pred, pred_trade_mask, ["long", "short"], drop_pred=False
            )
            cm_pred.to_csv(
                OUTPUT_DIR / f"confusion_matrix_tradeonly_predTrades_{mode}_{policy_name}.csv"
            )

            coverage = float(in_coverage.mean())
            trade_rate = float((y_pred != "skip").mean())
            macro_f1 = float(metrics_full.loc["macro_avg", "f1"])
            trade_macro_true = float(metrics_true.loc["macro_avg", "f1"])
            trade_macro_pred = float(metrics_pred.loc["macro_avg", "f1"])

            summary_rows.append(
                {
                    "policy": policy_name,
                    "mode": mode,
                    "coverage": coverage,
                    "trade_rate": trade_rate,
                    "macro_f1_3class": macro_f1,
                    "trade_only_macro_f1_trueTrades": trade_macro_true,
                    "trade_only_macro_f1_predTrades": trade_macro_pred,
                    "ambiguous_rate": ambiguous_rate,
                    "ambiguous_handled_5m_rate": ambiguous_handled_5m,
                    "ambiguous_handled_fallback_rate": ambiguous_handled_fallback,
                    "dropped_trueTrades_nonLS": dropped_true,
                    "dropped_predTrades_nonLS": dropped_pred,
                }
            )

            breakdown_rows = []
            dropped_union = pd.Series(False, index=df.index)
            for mask_name, mask in [
                ("true_trades", true_trade_mask),
                ("pred_trades", pred_trade_mask),
            ]:
                dropped_mask = mask & ~(y_true.isin(["long", "short"]) & y_pred.isin(["long", "short"]))
                reasons = classify_dropped_rows(df, y_true, y_pred, in_coverage, dropped_mask)
                counts = reasons[dropped_mask].value_counts().to_dict()
                breakdown_rows.append(
                    {
                        "mask_type": mask_name,
                        "dropped_due_to_out_of_coverage": counts.get(
                            "dropped_due_to_out_of_coverage", 0
                        ),
                        "dropped_due_to_ambiguous_to_skip": counts.get(
                            "dropped_due_to_ambiguous_to_skip", 0
                        ),
                        "dropped_due_to_nan_features_or_missing_regime": counts.get(
                            "dropped_due_to_nan_features_or_missing_regime", 0
                        ),
                        "dropped_due_to_label_not_in_LS": counts.get(
                            "dropped_due_to_label_not_in_LS", 0
                        ),
                        "dropped_due_to_other": counts.get("dropped_due_to_other", 0),
                    }
                )
                dropped_union = dropped_union | dropped_mask

            breakdown_df = pd.DataFrame(breakdown_rows)
            breakdown_df.to_csv(
                policy_debug_dir / f"dropped_trades_breakdown_{policy_name}_{mode}.csv",
                index=False,
            )

            if dropped_union.any():
                reasons_union = classify_dropped_rows(
                    df, y_true, y_pred, in_coverage, dropped_union
                )
                sample_df = pd.DataFrame(
                    {
                        "timestamp": df["timestamp"],
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "regime_id": df["regime_id"],
                        "prob_max": df["regime_prob_max"],
                        "ambiguous_original": df["ambiguous_original"],
                        "resolved_outcome": df["resolved_outcome"],
                        "reason": reasons_union,
                    }
                )
                sample_df = sample_df[dropped_union].copy()
                sample_n = min(20, len(sample_df))
                sample_df = sample_df.sample(n=sample_n, random_state=policy_data["seed"])
                sample_df.to_csv(
                    policy_debug_dir / f"dropped_trades_sample_{policy_name}_{mode}.csv",
                    index=False,
                )

            for regime_id, group in df.groupby("regime_id"):
                idx = group.index
                y_true_r = y_true.loc[idx]
                y_pred_r = y_pred.loc[idx]
                in_cov_r = in_coverage.loc[idx]
                coverage_r = float(in_cov_r.mean()) if len(group) else np.nan
                trade_rate_r = float((y_pred_r != "skip").mean()) if len(group) else np.nan

                cm_r, metrics_r = compute_metrics(y_true_r, y_pred_r, ["long", "short", "skip"])
                true_mask_r = y_true_r != "skip"
                _, metrics_true_r, _ = trade_only_metrics(
                    y_true_r, y_pred_r, true_mask_r, ["long", "short"], drop_pred=True
                )
                pred_mask_r = y_pred_r != "skip"
                _, metrics_pred_r, _ = trade_only_metrics(
                    y_true_r, y_pred_r, pred_mask_r, ["long", "short"], drop_pred=False
                )

                regime_rows.append(
                    {
                        "policy": policy_name,
                        "mode": mode,
                        "regime_id": regime_id,
                        "n_rows": len(group),
                        "n_true_long": int((y_true_r == "long").sum()),
                        "n_true_short": int((y_true_r == "short").sum()),
                        "n_true_skip": int((y_true_r == "skip").sum()),
                        "coverage_rate": coverage_r,
                        "predicted_trade_rate": trade_rate_r,
                        "precision_long": metrics_r.loc["long", "precision"],
                        "recall_long": metrics_r.loc["long", "recall"],
                        "f1_long": metrics_r.loc["long", "f1"],
                        "precision_short": metrics_r.loc["short", "precision"],
                        "recall_short": metrics_r.loc["short", "recall"],
                        "f1_short": metrics_r.loc["short", "f1"],
                        "macro_f1_3class": metrics_r.loc["macro_avg", "f1"],
                        "trade_only_macro_f1_trueTrades": metrics_true_r.loc["macro_avg", "f1"],
                        "trade_only_macro_f1_predTrades": metrics_pred_r.loc["macro_avg", "f1"],
                        "avg_max_posterior_prob": float(group["regime_prob_max"].mean()),
                        "avg_entropy": float(group["entropy"].mean()) if "entropy" in group.columns else np.nan,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUTPUT_DIR / "regime_validation_table.csv", index=False)

    gate_rows = summary_df[summary_df["policy"] == "gate_only"]
    for mode in ["oracle", "random"]:
        row = gate_rows[gate_rows["mode"] == mode]
        if row.empty:
            continue
        row = row.iloc[0]
        print(
            f"Gate_only {mode}: coverage={row['coverage']:.4f}, "
            f"macro_f1={row['macro_f1_3class']:.4f}, "
            f"trade_only_f1_true={row['trade_only_macro_f1_trueTrades']:.4f}, "
            f"ambiguous_rate={row['ambiguous_rate']:.4f}"
        )

    # Sanity checks
    valid_trades = df[df["final_label"].isin(["long", "short"])].copy()
    pnl = np.full(len(valid_trades), np.nan, dtype=float)
    long_mask = valid_trades["final_label"] == "long"
    short_mask = valid_trades["final_label"] == "short"
    pnl[long_mask] = (valid_trades.loc[long_mask, "open"] + TP_POINTS) / valid_trades.loc[
        long_mask, "open"
    ] - 1.0
    pnl[short_mask] = valid_trades.loc[short_mask, "open"] / (
        valid_trades.loc[short_mask, "open"] - TP_POINTS
    ) - 1.0
    pnl = pnl - FEE_PER_TRADE
    valid_trades["pnl"] = pnl

    wins = (valid_trades["pnl"] > 0).sum()
    losses = (valid_trades["pnl"] < 0).sum()
    pf = profit_factor(valid_trades["pnl"].dropna())
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    if np.isinf(pf):
        sanity_lines.append(f"WARNING: profit_factor is inf (n_losses={losses}).")
    if win_rate == 1.0 and len(valid_trades) >= args.min_warn_trades:
        sanity_lines.append(
            f"WARNING: win_rate==1.0 with n_trades={len(valid_trades)}. Sample trades:"
        )
        sample_cols = [
            "timestamp",
            "open",
            "horizon_high",
            "horizon_low",
            "final_label",
        ]
        sanity_lines.append(
            valid_trades.head(10)[sample_cols].to_csv(index=False).strip()
        )
    if (valid_trades["pnl"] < 0).sum() == 0:
        sanity_lines.append(
            "WARNING: No negative pnl trades found after ambiguity resolution."
        )

    sanity_path = OUTPUT_DIR / "sanity_report.txt"
    sanity_path.write_text("\n".join(sanity_lines), encoding="utf-8")

    print("Enhanced precision validation complete.")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Ambiguous rate: {ambiguous_rate:.2%}")
    print(f"Ambiguous handled by 5m: {ambiguous_handled_5m:.2%}")
    print(f"Ambiguous handled by fallback: {ambiguous_handled_fallback:.2%}")


if __name__ == "__main__":
    main()
