from __future__ import annotations

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
OUTPUT_DIR = RESULTS_DIR / "validation" / "frontier"

TP_POINTS = 2000
SL_POINTS = 1000
FEE_PER_TRADE = 0.0005
HORIZON_N_CANDLES = 1
RANDOM_SEED = 42

PROB_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99]
ENTROPY_THRESHOLDS = [None, 0.20, 0.10, 0.05, 0.02]
TOPN_OPTIONS = [None, 1, 2, 3, 4]

TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 3


POLICY_CONFIG = {
    "gate_only": {
        "regime_whitelist": [0, 2, 5, 6],
        "prob_threshold": 0.80,
        "entropy_max": None,
        "direction_rule": "majority",
        "min_trades": 30,
    },
    "low_regime_whitelist_clusters": {
        "regime_whitelist": [0, 5],
        "prob_threshold": 0.75,
        "entropy_max": None,
        "direction_rule": "majority",
        "min_trades": 30,
    },
    "baseline_experts": {
        "regime_whitelist": [0, 2, 5, 6],
        "prob_threshold": 0.90,
        "entropy_max": 0.01,
        "direction_rule": "precision_threshold",
        "min_precision": 0.60,
        "min_trades": 50,
    },
    "weighted_vote": {
        "regime_whitelist": [0, 2, 5, 6],
        "prob_threshold": 0.85,
        "entropy_max": 0.005,
        "direction_rule": "score_threshold",
        "min_abs_score": 0.20,
        "min_trades": 50,
    },
    "agreement": {
        "regime_whitelist": [0, 2],
        "prob_threshold": 0.90,
        "entropy_max": 0.02,
        "direction_rule": "agreement",
        "min_precision": 0.65,
        "min_trade_rate": 0.10,
        "min_trades": 50,
    },
}

POLICY_REQUIRED_KEYS = {
    "gate_only": ["regime_whitelist", "prob_threshold", "entropy_max", "direction_rule", "min_trades"],
    "low_regime_whitelist_clusters": [
        "regime_whitelist",
        "prob_threshold",
        "entropy_max",
        "direction_rule",
        "min_trades",
    ],
    "baseline_experts": [
        "regime_whitelist",
        "prob_threshold",
        "entropy_max",
        "direction_rule",
        "min_precision",
        "min_trades",
    ],
    "weighted_vote": [
        "regime_whitelist",
        "prob_threshold",
        "entropy_max",
        "direction_rule",
        "min_abs_score",
        "min_trades",
    ],
    "agreement": [
        "regime_whitelist",
        "prob_threshold",
        "entropy_max",
        "direction_rule",
        "min_precision",
        "min_trade_rate",
        "min_trades",
    ],
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


def build_walkforward_splits(
    timestamps: pd.Series, train_months: int, test_months: int, step_months: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    splits = []
    if timestamps.empty:
        return splits
    start = timestamps.min()
    end = timestamps.max()
    current_start = start
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        splits.append((current_start, train_end, train_end, test_end))
        current_start = current_start + pd.DateOffset(months=step_months)
    return splits


def compute_regime_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = train_df.groupby("regime_id")
    stats = grouped["y_true"].value_counts().unstack(fill_value=0)
    for col in ["long", "short", "skip"]:
        if col not in stats.columns:
            stats[col] = 0
    stats["n_total"] = stats[["long", "short", "skip"]].sum(axis=1)
    stats["n_trade"] = stats[["long", "short"]].sum(axis=1)
    stats["trade_rate"] = stats["n_trade"] / stats["n_total"].replace(0, np.nan)
    stats["majority_dir"] = np.where(stats["long"] >= stats["short"], "long", "short")
    stats["majority_precision"] = stats[["long", "short"]].max(axis=1) / stats["n_trade"].replace(0, np.nan)
    stats = stats.fillna(0.0)
    return stats


def select_top_regimes(stats: pd.DataFrame, whitelist: List[int], top_n: Optional[int]) -> List[int]:
    if top_n is None:
        return whitelist
    filtered = stats.loc[stats.index.intersection(whitelist)].copy()
    if filtered.empty:
        return []
    filtered = filtered.sort_values("majority_precision", ascending=False)
    return filtered.head(top_n).index.astype(int).tolist()


def build_direction_map(
    stats: pd.DataFrame,
    whitelist: List[int],
    policy_cfg: Dict[str, object],
) -> Dict[int, str]:
    direction_map: Dict[int, str] = {}
    for regime in whitelist:
        if regime not in stats.index:
            direction_map[regime] = "skip"
            continue
        row = stats.loc[regime]
        n_trade = int(row["n_trade"])
        if n_trade < policy_cfg.get("min_trades", 0):
            direction_map[regime] = "skip"
            continue
        rule = policy_cfg["direction_rule"]
        if rule == "majority":
            direction_map[regime] = "long" if row["long"] >= row["short"] else "short"
        elif rule == "precision_threshold":
            precision = row["majority_precision"]
            if precision >= policy_cfg["min_precision"]:
                direction_map[regime] = row["majority_dir"]
            else:
                direction_map[regime] = "skip"
        elif rule == "score_threshold":
            score = (row["long"] - row["short"]) / max(n_trade, 1)
            if abs(score) >= policy_cfg["min_abs_score"]:
                direction_map[regime] = "long" if score > 0 else "short"
            else:
                direction_map[regime] = "skip"
        elif rule == "agreement":
            precision = row["majority_precision"]
            trade_rate = row["trade_rate"]
            if precision >= policy_cfg["min_precision"] and trade_rate >= policy_cfg["min_trade_rate"]:
                direction_map[regime] = row["majority_dir"]
            else:
                direction_map[regime] = "skip"
        else:
            raise ValueError(f"Unknown direction_rule '{rule}'.")
    return direction_map


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = ["long", "short", "skip"]
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


def trade_only_metrics(y_true: pd.Series, y_pred: pd.Series, mask: pd.Series) -> float:
    labels = ["long", "short"]
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]
    valid = y_true_sub.isin(labels) & y_pred_sub.isin(labels)
    if valid.any():
        _, metrics_df = compute_metrics(y_true_sub[valid], y_pred_sub[valid])
        return float(metrics_df.loc["macro_avg", "f1"])
    return 0.0


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if pnl.empty:
        return np.nan
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / abs(losses)


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return np.nan
    cum = pnl.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    return float(drawdown.min())


def cvar_95(pnl: pd.Series) -> float:
    if pnl.empty:
        return np.nan
    q = pnl.quantile(0.05)
    tail = pnl[pnl <= q]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def compute_trade_pnl(df: pd.DataFrame, y_pred: pd.Series, y_true: pd.Series) -> pd.Series:
    open_px = df["open"]
    multiplier = open_px / 100000.0
    long_tp = open_px + TP_POINTS * multiplier
    long_sl = open_px - SL_POINTS * multiplier
    short_tp = open_px - TP_POINTS * multiplier
    short_sl = open_px + SL_POINTS * multiplier

    pnl = np.full(len(df), np.nan, dtype=float)
    pred_long = y_pred == "long"
    pred_short = y_pred == "short"

    pnl[pred_long & (y_true == "long")] = (long_tp[pred_long & (y_true == "long")] / open_px[pred_long & (y_true == "long")]) - 1.0
    pnl[pred_long & (y_true == "short")] = (long_sl[pred_long & (y_true == "short")] / open_px[pred_long & (y_true == "short")]) - 1.0
    pnl[pred_short & (y_true == "short")] = (open_px[pred_short & (y_true == "short")] / short_tp[pred_short & (y_true == "short")]) - 1.0
    pnl[pred_short & (y_true == "long")] = (open_px[pred_short & (y_true == "long")] / short_sl[pred_short & (y_true == "long")]) - 1.0

    pnl[(pred_long | pred_short) & (y_true == "skip")] = 0.0
    pnl = np.where(np.isfinite(pnl), pnl - FEE_PER_TRADE, np.nan)
    return pd.Series(pnl, index=df.index)


def compute_frontier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min_precision"] = df[["precision_long", "precision_short"]].min(axis=1)
    df = df.sort_values(["min_precision", "profit_factor", "max_drawdown"], ascending=[False, False, True])
    return df


def main() -> None:
    validate_policy_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = OUTPUT_DIR / "full_sweep_partial.csv"

    df = load_data(INPUT_CSV)
    df = load_gmm_outputs(df, RESULTS_DIR)

    if "candle_type" not in df.columns:
        raise ValueError("candle_type column missing from input data.")
    df["candle_type"] = df["candle_type"].astype(str).str.lower().str.strip()
    df.loc[~df["candle_type"].isin(["long", "short", "skip"]), "candle_type"] = "skip"

    if "label_ambiguous" in df.columns:
        df["label_ambiguous"] = df["label_ambiguous"].astype(bool)
        ambiguous_original = df["label_ambiguous"]
    else:
        ambiguous_original = pd.Series(False, index=df.index)

    df["y_true"] = df["candle_type"].where(~ambiguous_original, "skip")
    df["ambiguous_original"] = ambiguous_original

    df["regime_id"] = pd.to_numeric(df["regime_id"], errors="coerce").fillna(-1).astype(int)
    df["regime_prob_max"] = pd.to_numeric(df["regime_prob_max"], errors="coerce")
    if "entropy" in df.columns:
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    else:
        df["entropy"] = np.nan

    splits = build_walkforward_splits(
        df["timestamp"], TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS
    )
    if not splits:
        raise ValueError("Not enough data to create walk-forward splits.")

    results: List[Dict[str, object]] = []
    rng = np.random.RandomState(RANDOM_SEED)

    for policy_name, policy_cfg in POLICY_CONFIG.items():
        for prob_threshold in PROB_THRESHOLDS:
            for entropy_threshold in ENTROPY_THRESHOLDS:
                for top_n in TOPN_OPTIONS:
                    y_true_all = []
                    y_pred_all = []
                    prob_all = []
                    entropy_all = []
                    ts_all = []
                    amb_all = []
                    pnl_all = []
                    coverage_all = []
                    trade_all = []

                    for train_start, train_end, test_start, test_end in splits:
                        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
                        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)

                        train_df = df.loc[train_mask].copy()
                        test_df = df.loc[test_mask].copy()
                        if train_df.empty or test_df.empty:
                            continue

                        stats = compute_regime_stats(train_df)
                        base_whitelist = policy_cfg["regime_whitelist"]
                        whitelist = select_top_regimes(stats, base_whitelist, top_n)

                        direction_map = build_direction_map(stats, whitelist, policy_cfg)

                        final_prob_threshold = max(policy_cfg["prob_threshold"], prob_threshold)
                        if policy_cfg["entropy_max"] is None and entropy_threshold is None:
                            final_entropy = None
                        elif policy_cfg["entropy_max"] is None:
                            final_entropy = entropy_threshold
                        elif entropy_threshold is None:
                            final_entropy = policy_cfg["entropy_max"]
                        else:
                            final_entropy = min(policy_cfg["entropy_max"], entropy_threshold)

                        in_whitelist = test_df["regime_id"].isin(whitelist)
                        prob_mask = test_df["regime_prob_max"] >= final_prob_threshold
                        entropy_mask = (
                            (test_df["entropy"] <= final_entropy)
                            if final_entropy is not None
                            else pd.Series(True, index=test_df.index)
                        )
                        in_coverage = in_whitelist & prob_mask & entropy_mask & test_df["regime_prob_max"].notna()

                        direction = test_df["regime_id"].map(direction_map).fillna("skip")
                        y_pred = pd.Series("skip", index=test_df.index)
                        y_pred.loc[in_coverage] = direction.loc[in_coverage]

                        y_true = test_df["y_true"].astype(str)
                        pnl = compute_trade_pnl(test_df, y_pred, y_true)

                        y_true_all.append(y_true)
                        y_pred_all.append(y_pred)
                        prob_all.append(test_df["regime_prob_max"])
                        entropy_all.append(test_df["entropy"])
                        ts_all.append(test_df["timestamp"])
                        amb_all.append(test_df["ambiguous_original"])
                        pnl_all.append(pnl)
                        coverage_all.append(in_coverage)
                        trade_all.append(y_pred != "skip")

                    if not y_true_all:
                        continue

                    y_true_all = pd.concat(y_true_all, axis=0)
                    y_pred_all = pd.concat(y_pred_all, axis=0)
                    ts_all = pd.concat(ts_all, axis=0)
                    amb_all = pd.concat(amb_all, axis=0)
                    pnl_all = pd.concat(pnl_all, axis=0)
                    coverage_all = pd.concat(coverage_all, axis=0)
                    trade_all = pd.concat(trade_all, axis=0)

                    cm_df, metrics_df = compute_metrics(y_true_all, y_pred_all)
                    precision_long = float(metrics_df.loc["long", "precision"])
                    recall_long = float(metrics_df.loc["long", "recall"])
                    f1_long = float(metrics_df.loc["long", "f1"])
                    precision_short = float(metrics_df.loc["short", "precision"])
                    recall_short = float(metrics_df.loc["short", "recall"])
                    f1_short = float(metrics_df.loc["short", "f1"])
                    macro_f1 = float(metrics_df.loc["macro_avg", "f1"])

                    trade_only_true = trade_only_metrics(y_true_all, y_pred_all, y_true_all != "skip")
                    trade_only_pred = trade_only_metrics(y_true_all, y_pred_all, y_pred_all != "skip")

                    coverage = float(coverage_all.mean())
                    trade_rate = float(trade_all.mean())

                    trades = pnl_all[trade_all]
                    n_trades = int(trade_all.sum())
                    trades_per_month = np.nan
                    if not ts_all.empty:
                        span_days = (ts_all.max() - ts_all.min()).total_seconds() / 86400.0
                        months = max(span_days / 30.44, 1.0) if span_days > 0 else 1.0
                        trades_per_month = n_trades / months

                    win_rate = float((trades > 0).mean()) if n_trades > 0 else np.nan
                    avg_pnl = float(trades.mean()) if n_trades > 0 else np.nan
                    median_pnl = float(trades.median()) if n_trades > 0 else np.nan
                    pf = profit_factor(trades.dropna())
                    max_dd = max_drawdown(trades.dropna())
                    cvar = cvar_95(trades.dropna())

                    ambiguous_rate = float(amb_all[trade_all].mean()) if trade_all.any() else 0.0
                    both_hit_unresolved = float(
                        ((amb_all) & (y_true_all == "skip") & trade_all).mean()
                    ) if trade_all.any() else 0.0

                    cm_flat = cm_df.values.flatten().tolist()

                    results.append(
                        {
                            "policy": policy_name,
                            "prob_threshold": prob_threshold,
                            "entropy_threshold": entropy_threshold,
                            "topN": top_n,
                            "coverage": coverage,
                            "trade_rate": trade_rate,
                            "precision_long": precision_long,
                            "recall_long": recall_long,
                            "f1_long": f1_long,
                            "precision_short": precision_short,
                            "recall_short": recall_short,
                            "f1_short": f1_short,
                            "macro_f1_3class": macro_f1,
                            "trade_only_macro_f1_trueTrades": trade_only_true,
                            "trade_only_macro_f1_predTrades": trade_only_pred,
                            "n_trades": n_trades,
                            "trades_per_month": trades_per_month,
                            "win_rate": win_rate,
                            "avg_pnl_per_trade": avg_pnl,
                            "median_pnl_per_trade": median_pnl,
                            "profit_factor": pf,
                            "max_drawdown": max_dd,
                            "cvar95": cvar,
                            "ambiguous_rate": ambiguous_rate,
                            "both_hit_unresolved_rate": both_hit_unresolved,
                            "cm_ll": cm_flat[0],
                            "cm_ls": cm_flat[1],
                            "cm_lskip": cm_flat[2],
                            "cm_sl": cm_flat[3],
                            "cm_ss": cm_flat[4],
                            "cm_sskip": cm_flat[5],
                            "cm_kl": cm_flat[6],
                            "cm_ks": cm_flat[7],
                            "cm_kskip": cm_flat[8],
                        }
                    )

        if results:
            partial_df = pd.DataFrame(results)
            partial_df.to_csv(partial_path, index=False)
            print(f"Wrote partial results for policy '{policy_name}' to {partial_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "full_sweep.csv", index=False)

    constraints = [0.25, 0.5, 1.0, 2.0]
    best_config = {}
    for constraint in constraints:
        filtered = results_df[results_df["trades_per_month"] >= constraint].copy()
        if filtered.empty:
            continue
        frontier = compute_frontier(filtered)
        out_path = OUTPUT_DIR / f"frontier_tpmo_{str(constraint).replace('.', '_')}.csv"
        frontier.to_csv(out_path, index=False)
        best = frontier.iloc[0].to_dict()
        best_config[str(constraint)] = {
            "policy": best["policy"],
            "prob_threshold": best["prob_threshold"],
            "entropy_threshold": best["entropy_threshold"],
            "topN": best["topN"],
            "min_precision": best["min_precision"],
            "profit_factor": best["profit_factor"],
            "trades_per_month": best["trades_per_month"],
        }

        print(f"\nTop 10 configs for trades_per_month >= {constraint}:")
        cols = [
            "policy",
            "prob_threshold",
            "entropy_threshold",
            "topN",
            "min_precision",
            "profit_factor",
            "max_drawdown",
            "trades_per_month",
        ]
        print(frontier[cols].head(10).to_string(index=False))

    with (OUTPUT_DIR / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("\nPrecision frontier sweep complete.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
