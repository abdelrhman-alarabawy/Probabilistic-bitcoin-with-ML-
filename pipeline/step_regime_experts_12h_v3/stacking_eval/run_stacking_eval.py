from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "pipeline" / "step_regime_experts_12h_v3" / "stacking_eval"
PREDS_PATH = (
    REPO_ROOT
    / "pipeline"
    / "step_regime_experts_12h_v3"
    / "results"
    / "preds_per_row_mode1_walkforward.csv"
)
DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"
ANOMALY_PIPELINE_DIR = REPO_ROOT / "pipeline" / "anomaly_trading_12h_v3"

FEE_PER_TRADE = 0.0005
LOW_REGIME = "low"
CONF_THRESHOLD = 0.60
WHITELIST_CLUSTERS = {2, 5, 7}
BAD_CLUSTERS = {0, 1, 4}

TP_LONG = 0.00666
SL_LONG = 0.00695
TP_SHORT = 0.00697
SL_SHORT = 0.00628
HOLD_BARS_TPSL = 1


def ensure_datetime_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def normalize_label(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return str(value).strip().lower()


def resolve_label_order(labels: Iterable[object]) -> List[object]:
    labels = [label for label in labels if not pd.isna(label)]
    label_set = {label for label in labels}
    ordered = []
    for label in ["long", "short", "skip"]:
        if label in label_set:
            ordered.append(label)
    if ordered:
        return ordered
    return sorted(label_set)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[object]
) -> Tuple[Dict[str, float], Dict[object, Dict[str, float]]]:
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}, {}

    accuracy = float(np.mean(y_true == y_pred))
    per_label: Dict[object, Dict[str, float]] = {}
    f1s = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return {"accuracy": accuracy, "macro_f1": macro_f1}, per_label


def compute_trade_returns(
    signals: np.ndarray, forward_returns: np.ndarray, fee: float
) -> np.ndarray:
    returns = np.zeros(len(signals), dtype=float)
    for idx, signal in enumerate(signals):
        if not np.isfinite(forward_returns[idx]):
            returns[idx] = np.nan
            continue
        if signal == "long":
            returns[idx] = forward_returns[idx] - fee
        elif signal == "short":
            returns[idx] = -forward_returns[idx] - fee
        else:
            returns[idx] = 0.0
    return returns


def compute_profit_factor(trade_returns: np.ndarray) -> float:
    gains = np.nansum(trade_returns[trade_returns > 0])
    losses = np.nansum(trade_returns[trade_returns < 0])
    if losses == 0:
        return float("nan")
    return float(gains / abs(losses))


def compute_drawdown(trade_returns: np.ndarray) -> float:
    if len(trade_returns) == 0:
        return float("nan")
    cumulative = np.cumsum(trade_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(np.min(drawdown))


def months_covered(timestamps: pd.Series) -> int:
    if timestamps.empty:
        return 1
    months = timestamps.dt.to_period("M").nunique()
    return max(1, int(months))


def compute_group_metrics(
    df_group: pd.DataFrame,
    signal_col: str,
    labels: List[object],
    months_in_test: int,
) -> Dict[str, float]:
    support = len(df_group)
    y_true = df_group["y_true_norm"].to_numpy()
    y_pred = df_group[signal_col].to_numpy()

    metrics, per_label = compute_classification_metrics(y_true, y_pred, labels)

    skip_label = "skip" if "skip" in labels else None
    df_sorted = df_group.sort_values("timestamp")
    sorted_signals = df_sorted[signal_col].to_numpy()
    trade_mask = (
        sorted_signals != skip_label
        if skip_label is not None
        else np.ones(len(sorted_signals), dtype=bool)
    )
    trade_count = int(np.sum(trade_mask))
    coverage = float(trade_count / support) if support else float("nan")
    trade_returns_all = compute_trade_returns(
        sorted_signals,
        df_sorted["forward_return_1"].to_numpy(),
        FEE_PER_TRADE,
    )
    trade_returns = trade_returns_all[trade_mask]
    avg_pnl = float(np.nanmean(trade_returns)) if len(trade_returns) else float("nan")
    median_pnl = float(np.nanmedian(trade_returns)) if len(trade_returns) else float("nan")
    win_rate = float(np.nanmean(trade_returns > 0)) if len(trade_returns) else float("nan")
    profit_factor = compute_profit_factor(trade_returns) if len(trade_returns) else float("nan")
    max_dd = compute_drawdown(trade_returns)
    cvar95 = (
        float(np.nanpercentile(trade_returns, 5))
        if len(trade_returns)
        else float("nan")
    )

    f1_long = per_label.get("long", {}).get("f1", float("nan"))
    f1_short = per_label.get("short", {}).get("f1", float("nan"))
    prec_long = per_label.get("long", {}).get("precision", float("nan"))
    rec_long = per_label.get("long", {}).get("recall", float("nan"))
    prec_short = per_label.get("short", {}).get("precision", float("nan"))
    rec_short = per_label.get("short", {}).get("recall", float("nan"))

    return {
        "support": support,
        "trade_count": trade_count,
        "coverage": coverage,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision_long": prec_long,
        "recall_long": rec_long,
        "f1_long": f1_long,
        "precision_short": prec_short,
        "recall_short": rec_short,
        "f1_short": f1_short,
        "avg_pnl_per_trade": avg_pnl,
        "median_pnl_per_trade": median_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "cvar95": cvar95,
        "trades_per_month": float(trade_count / months_in_test) if months_in_test else float("nan"),
    }


def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "support",
        "trade_count",
        "coverage",
        "accuracy",
        "macro_f1",
        "precision_long",
        "recall_long",
        "f1_long",
        "precision_short",
        "recall_short",
        "f1_short",
        "avg_pnl_per_trade",
        "median_pnl_per_trade",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "cvar95",
        "trades_per_month",
    ]
    rows: List[Dict[str, float]] = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def load_anomaly_modules(anomaly_dir: Path):
    if str(anomaly_dir) not in sys.path:
        sys.path.insert(0, str(anomaly_dir))
    from src import config as aconfig
    from src import data as adata
    from src import eval as aeval
    from src import features as afeatures
    from src import models as amodels
    from src import rolling as arolling
    from src import run_anomaly_v3 as arun

    return aconfig, adata, aeval, afeatures, amodels, arolling, arun


def compute_anomaly_signals(
    data_path: Path, include_liq: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    aconfig, adata, _, afeatures, amodels, arolling, arun = load_anomaly_modules(
        ANOMALY_PIPELINE_DIR
    )

    df = adata.load_data(str(data_path))
    trend_df = arun._build_trend_features(df)

    features, _, liq_cols, liq_flag_cols = afeatures.build_event_features(
        df, include_liq=include_liq
    )
    _ = afeatures.compute_liq_missing(features, liq_cols, liq_flag_cols)

    ret_prev = features["return_1"].astype(float)
    direction_all = arun._direction_from_return(ret_prev)

    test_rows: List[pd.DataFrame] = []
    for config in aconfig.WINDOW_CONFIGS:
        windows = list(
            arolling.generate_rolling_windows(
                df,
                aconfig.TIMESTAMP_COL,
                config,
                min_train_rows=aconfig.MIN_TRAIN_ROWS,
                min_test_rows=aconfig.MIN_TEST_ROWS,
            )
        )
        for window in windows:
            train_idx = window.train_idx
            test_idx = window.test_idx

            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]

            train_median = X_train.median()
            X_train = X_train.fillna(train_median).fillna(0.0)
            X_test = X_test.fillna(train_median).fillna(0.0)

            rz_model = amodels.fit_robust_z(X_train)
            score_train = amodels.score_robust_z(rz_model, X_train)
            score_test = amodels.score_robust_z(rz_model, X_test)
            thresholds = amodels.compute_thresholds(score_train, aconfig.PCT_LIST)

            test_output = pd.DataFrame(
                {
                    "window_id": window.window_id,
                    "timestamp": df.loc[test_idx, aconfig.TIMESTAMP_COL].values,
                    "row_idx": test_idx,
                    "score_robustz": score_test,
                    "direction": direction_all[test_idx],
                    "range_strength": trend_df.loc[test_idx, "range_strength"].values,
                }
            )

            buckets = arun._assign_buckets(test_output["range_strength"].to_numpy(dtype=float))
            adaptive_pct, adaptive_flag_raw = arun._adaptive_flags(score_test, thresholds, buckets)
            test_output["trend_bucket"] = buckets
            test_output["adaptive_pct"] = adaptive_pct
            test_output["adaptive_flag_raw"] = adaptive_flag_raw.astype(int)
            adaptive_threshold = np.full(len(score_test), np.nan)
            for pct in aconfig.PCT_LIST:
                mask = adaptive_pct == pct
                if np.any(mask):
                    adaptive_threshold[mask] = thresholds.get(pct, np.nan)
            test_output["adaptive_threshold"] = adaptive_threshold
            test_rows.append(test_output)

    all_test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    if all_test_df.empty:
        raise RuntimeError("No anomaly test windows generated; check data coverage.")

    all_test_df = all_test_df.sort_values(["timestamp", "window_id"]).reset_index(drop=True)
    dedup_df = all_test_df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(
        drop=True
    )

    bucket_skip = (dedup_df["trend_bucket"] == 4) & (aconfig.BUCKET4_MODE == "skip")
    dedup_df["adaptive_flag"] = dedup_df["adaptive_flag_raw"]
    if aconfig.BUCKET4_MODE == "skip":
        dedup_df.loc[bucket_skip, "adaptive_flag"] = 0

    signals = np.full(len(dedup_df), "skip", dtype=object)
    is_trade = (dedup_df["adaptive_flag"].to_numpy(dtype=int) == 1) & (
        dedup_df["direction"].to_numpy(dtype=int) != 0
    )
    signals[is_trade & (dedup_df["direction"] > 0)] = "long"
    signals[is_trade & (dedup_df["direction"] < 0)] = "short"

    anomaly_conf = np.full(len(dedup_df), np.nan)
    thresholds = dedup_df["adaptive_threshold"].to_numpy(dtype=float)
    scores = dedup_df["score_robustz"].to_numpy(dtype=float)
    valid_thr = np.isfinite(thresholds) & (thresholds > 0)
    anomaly_conf[valid_thr] = scores[valid_thr] / thresholds[valid_thr]

    anomaly_df = pd.DataFrame(
        {
            "timestamp": ensure_datetime_utc(dedup_df["timestamp"]),
            "anomaly_signal": signals,
            "anomaly_score": dedup_df["score_robustz"].astype(float),
            "anomaly_confidence": anomaly_conf,
        }
    )

    return anomaly_df, df, aconfig


def select_weighted_vote_thresholds(
    train_df: pd.DataFrame, min_trades: int = 5
) -> Tuple[float, float]:
    subset = train_df[
        (train_df["regime3"] == LOW_REGIME)
        & (train_df["cluster_id"].isin(WHITELIST_CLUSTERS))
    ].copy()
    scores = subset["weighted_score"].dropna().to_numpy()
    if len(scores) < 20:
        return float("nan"), float("nan")

    high_q = [0.6, 0.7, 0.8, 0.9]
    low_q = [0.1, 0.2, 0.3, 0.4]
    thr_long_candidates = np.unique(np.nanquantile(scores, high_q))
    thr_short_candidates = np.unique(np.nanquantile(scores, low_q))

    best = (float("nan"), float("nan"))
    best_score = -np.inf

    def score_pair(thr_long: float, thr_short: float, min_trades_req: int) -> float:
        signals = np.full(len(subset), "skip", dtype=object)
        signals[subset["weighted_score"] > thr_long] = "long"
        signals[subset["weighted_score"] < thr_short] = "short"
        trade_mask = signals != "skip"
        trade_count = int(np.sum(trade_mask))
        if trade_count < min_trades_req:
            return -np.inf
        returns = compute_trade_returns(
            signals, subset["forward_return_1"].to_numpy(), FEE_PER_TRADE
        )
        return float(np.nansum(returns[trade_mask]))

    for thr_long in thr_long_candidates:
        for thr_short in thr_short_candidates:
            if thr_long <= thr_short:
                continue
            score = score_pair(float(thr_long), float(thr_short), min_trades)
            if score > best_score:
                best_score = score
                best = (float(thr_long), float(thr_short))

    if math.isinf(best_score):
        for thr_long in thr_long_candidates:
            for thr_short in thr_short_candidates:
                if thr_long <= thr_short:
                    continue
                score = score_pair(float(thr_long), float(thr_short), 1)
                if score > best_score:
                    best_score = score
                    best = (float(thr_long), float(thr_short))

    return best


def add_weighted_vote_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_score"] = df["proba_long"] - df["proba_short"]
    df["weighted_score"] = 0.6 * df["anomaly_score"] + 0.4 * df["model_score"]

    thresholds = []
    for approach in sorted(df["approach"].dropna().unique()):
        subset = df[df["approach"] == approach]
        fold_ids = sorted(subset["fold_id"].dropna().unique())
        for fold_id in fold_ids:
            train_df = subset[subset["fold_id"] < fold_id]
            thr_long, thr_short = select_weighted_vote_thresholds(train_df)
            thresholds.append(
                {
                    "approach": approach,
                    "fold_id": fold_id,
                    "thr_long": thr_long,
                    "thr_short": thr_short,
                }
            )

    thr_df = pd.DataFrame(thresholds)
    df = df.merge(thr_df, on=["approach", "fold_id"], how="left")

    df["signal_weighted_vote"] = "skip"
    valid = (
        (df["regime3"] == LOW_REGIME)
        & (df["cluster_id"].isin(WHITELIST_CLUSTERS))
        & df["weighted_score"].notna()
        & df["thr_long"].notna()
        & df["thr_short"].notna()
    )
    df.loc[valid & (df["weighted_score"] > df["thr_long"]), "signal_weighted_vote"] = "long"
    df.loc[valid & (df["weighted_score"] < df["thr_short"]), "signal_weighted_vote"] = "short"
    return df


def run_tpsl_low_regime(
    df: pd.DataFrame,
    policy_name: str,
    signal_col: str,
    data_df: pd.DataFrame,
    aeval,
) -> List[Dict[str, float]]:
    data_df = data_df.copy()
    data_df["timestamp"] = ensure_datetime_utc(data_df["timestamp"])
    idx_map = {ts: int(idx) for idx, ts in enumerate(data_df["timestamp"])}

    open_px = data_df["open"].astype(float).to_numpy()
    high_px = data_df["high"].astype(float).to_numpy()
    low_px = data_df["low"].astype(float).to_numpy()
    close_px = data_df["close"].astype(float).to_numpy()

    rows: List[Dict[str, float]] = []
    for approach in sorted(df["approach"].dropna().unique()):
        df_app = df[df["approach"] == approach]
        for fold_id in sorted(df_app["fold_id"].dropna().unique()):
            df_fold = df_app[df_app["fold_id"] == fold_id]
            df_fold = df_fold[df_fold["regime3"] == LOW_REGIME].copy()
            if df_fold.empty:
                continue

            trade_rows = []
            for _, row in df_fold.iterrows():
                signal = row[signal_col]
                if signal not in {"long", "short"}:
                    continue
                idx = idx_map.get(row["timestamp"])
                if idx is None:
                    continue
                direction = 1 if signal == "long" else -1
                tp = TP_LONG if signal == "long" else TP_SHORT
                sl = SL_LONG if signal == "long" else SL_SHORT
                result = aeval.simulate_trade(
                    idx,
                    direction,
                    open_px,
                    high_px,
                    low_px,
                    close_px,
                    hold_bars=HOLD_BARS_TPSL,
                    tp_pct=tp,
                    sl_pct=sl,
                    fee_per_trade=FEE_PER_TRADE,
                )
                if result is None:
                    continue
                trade_rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "ret_net": result.ret_net,
                        "exit_type": result.exit_type,
                        "mae": result.mae,
                        "mfe": result.mfe,
                    }
                )

            trades_df = pd.DataFrame(trade_rows)
            summary = aeval.summarize_trades(trades_df)
            returns = trades_df["ret_net"].to_numpy(dtype=float) if not trades_df.empty else np.array([])
            profit_factor = compute_profit_factor(returns) if len(returns) else float("nan")
            months_in_test = months_covered(df_fold["timestamp"])

            rows.append(
                {
                    "approach": approach,
                    "policy": policy_name,
                    "fold_id": int(fold_id),
                    "stat": "fold",
                    "n_trades": summary["n_trades"],
                    "win_rate": summary["win_rate"],
                    "avg_return": summary["avg_return"],
                    "median_return": summary["median_return"],
                    "max_drawdown": summary["max_drawdown"],
                    "cvar_95": summary["cvar_95"],
                    "profit_factor": profit_factor,
                    "trades_per_month": float(summary["n_trades"] / months_in_test)
                    if months_in_test
                    else float("nan"),
                }
            )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return rows

    metric_cols = [
        "n_trades",
        "win_rate",
        "avg_return",
        "median_return",
        "max_drawdown",
        "cvar_95",
        "profit_factor",
        "trades_per_month",
    ]
    for (approach, policy), group in results_df.groupby(["approach", "policy"]):
        mean_row = {
            "approach": approach,
            "policy": policy,
            "fold_id": np.nan,
            "stat": "mean",
        }
        std_row = {
            "approach": approach,
            "policy": policy,
            "fold_id": np.nan,
            "stat": "std",
        }
        for metric in metric_cols:
            mean_row[metric] = float(group[metric].mean())
            std_row[metric] = float(group[metric].std())
        rows.extend([mean_row, std_row])

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anomaly_df, data_df, _ = compute_anomaly_signals(DATA_PATH, include_liq=True)
    anomaly_path = OUTPUT_DIR / "anomaly_signals_12h_v3.csv"
    anomaly_df.to_csv(anomaly_path, index=False)

    preds_df = pd.read_csv(PREDS_PATH)
    preds_df["timestamp"] = ensure_datetime_utc(preds_df["timestamp"])

    merged_full = preds_df.merge(
        anomaly_df[["timestamp", "anomaly_signal", "anomaly_score"]], on="timestamp", how="left"
    )
    merged_full["anomaly_signal"] = merged_full["anomaly_signal"].fillna("skip")

    proba_cols = [col for col in merged_full.columns if col.startswith("proba_")]
    merged_cols = [
        "timestamp",
        "fold_id",
        "approach",
        "y_true",
        "y_pred",
        *proba_cols,
        "cluster_id",
        "cluster_confidence",
        "regime3",
        "anomaly_signal",
        "anomaly_score",
        "close",
        "forward_return_1",
        "pnl_proxy",
    ]
    merged_out = merged_full[merged_cols]
    merged_path = OUTPUT_DIR / "merged_rows_mode1_walkforward.csv"
    merged_out.to_csv(merged_path, index=False)

    stack_df = merged_full.copy()
    stack_df["y_true_norm"] = stack_df["y_true"].map(normalize_label)
    stack_df["y_pred_norm"] = stack_df["y_pred"].map(normalize_label)
    stack_df["anomaly_signal_norm"] = stack_df["anomaly_signal"].map(normalize_label).fillna(
        "skip"
    )
    stack_df["regime3"] = stack_df["regime3"].astype(str).str.lower()
    stack_df.loc[stack_df["regime3"] == "nan", "regime3"] = np.nan
    stack_df["cluster_id"] = pd.to_numeric(stack_df["cluster_id"], errors="coerce").astype(
        "Int64"
    )
    stack_df["cluster_confidence"] = pd.to_numeric(
        stack_df["cluster_confidence"], errors="coerce"
    )

    is_low = stack_df["regime3"] == LOW_REGIME
    is_whitelist = stack_df["cluster_id"].isin(WHITELIST_CLUSTERS)
    anomaly_ok = stack_df["anomaly_signal_norm"].isin(["long", "short"])

    stack_df["signal_gate_only"] = np.where(
        is_low & (stack_df["cluster_confidence"] >= CONF_THRESHOLD) & anomaly_ok,
        stack_df["anomaly_signal_norm"],
        "skip",
    )

    stack_df["signal_agreement"] = np.where(
        is_low
        & is_whitelist
        & (stack_df["cluster_confidence"] >= CONF_THRESHOLD)
        & anomaly_ok
        & (stack_df["anomaly_signal_norm"] == stack_df["y_pred_norm"]),
        stack_df["anomaly_signal_norm"],
        "skip",
    )

    stack_df["signal_low_regime_whitelist_clusters"] = np.where(
        is_low & is_whitelist,
        stack_df["y_pred_norm"],
        "skip",
    )

    has_weighted_vote = (
        "anomaly_score" in stack_df.columns
        and "proba_long" in stack_df.columns
        and "proba_short" in stack_df.columns
        and stack_df["anomaly_score"].notna().any()
    )
    if has_weighted_vote:
        stack_df = add_weighted_vote_signal(stack_df)

    stacking_cols = [
        "signal_gate_only",
        "signal_agreement",
        "signal_low_regime_whitelist_clusters",
    ]
    if has_weighted_vote:
        stacking_cols.append("signal_weighted_vote")

    stack_path = OUTPUT_DIR / "stacking_policy_rows.csv"
    stack_df[
        [
            "timestamp",
            "fold_id",
            "split_type",
            "approach",
            "y_true",
            "y_pred",
            *proba_cols,
            "cluster_id",
            "cluster_confidence",
            "regime3",
            "anomaly_signal",
            "anomaly_score",
            "close",
            "forward_return_1",
            "pnl_proxy",
            *stacking_cols,
        ]
    ].to_csv(stack_path, index=False)

    label_values = resolve_label_order(stack_df["y_true_norm"].dropna().unique())
    policy_map = {
        "baseline_experts": "y_pred_norm",
        "gate_only": "signal_gate_only",
        "agreement": "signal_agreement",
        "low_regime_whitelist_clusters": "signal_low_regime_whitelist_clusters",
    }
    if has_weighted_vote:
        policy_map["weighted_vote"] = "signal_weighted_vote"

    fold_rows = []
    fold_cluster_rows = []
    fold_regime_rows = []
    for approach in sorted(stack_df["approach"].dropna().unique()):
        df_app = stack_df[stack_df["approach"] == approach]
        for fold_id in sorted(df_app["fold_id"].dropna().unique()):
            df_fold = df_app[df_app["fold_id"] == fold_id].copy()
            if df_fold.empty:
                continue
            months_in_test = months_covered(df_fold["timestamp"])
            for policy, signal_col in policy_map.items():
                metrics = compute_group_metrics(
                    df_fold, signal_col, label_values, months_in_test
                )
                fold_rows.append(
                    {
                        "split_type": "mode1_walkforward",
                        "approach": approach,
                        "policy": policy,
                        "fold_id": int(fold_id),
                        **metrics,
                    }
                )

                for regime_val, df_regime in df_fold.groupby("regime3"):
                    reg_metrics = compute_group_metrics(
                        df_regime, signal_col, label_values, months_in_test
                    )
                    fold_regime_rows.append(
                        {
                            "split_type": "mode1_walkforward",
                            "approach": approach,
                            "policy": policy,
                            "fold_id": int(fold_id),
                            "regime3": regime_val,
                            **reg_metrics,
                        }
                    )

                for cluster_val, df_cluster in df_fold.groupby("cluster_id"):
                    cl_metrics = compute_group_metrics(
                        df_cluster, signal_col, label_values, months_in_test
                    )
                    fold_cluster_rows.append(
                        {
                            "split_type": "mode1_walkforward",
                            "approach": approach,
                            "policy": policy,
                            "fold_id": int(fold_id),
                            "cluster_id": cluster_val,
                            **cl_metrics,
                        }
                    )

    fold_df = pd.DataFrame(fold_rows)
    fold_regime_df = pd.DataFrame(fold_regime_rows)
    fold_cluster_df = pd.DataFrame(fold_cluster_rows)

    summary_df = aggregate_metrics(
        fold_df, ["split_type", "approach", "policy"]
    )
    regime_summary_df = aggregate_metrics(
        fold_regime_df, ["split_type", "approach", "policy", "regime3"]
    )
    cluster_summary_df = aggregate_metrics(
        fold_cluster_df, ["split_type", "approach", "policy", "cluster_id"]
    )

    summary_path = OUTPUT_DIR / "stacking_summary_mode1_walkforward.csv"
    summary_df.to_csv(summary_path, index=False)
    regime_summary_path = OUTPUT_DIR / "stacking_by_regime3.csv"
    regime_summary_df.to_csv(regime_summary_path, index=False)
    cluster_summary_path = OUTPUT_DIR / "stacking_by_cluster.csv"
    cluster_summary_df.to_csv(cluster_summary_path, index=False)

    _, _, aeval, _, _, _, _ = load_anomaly_modules(ANOMALY_PIPELINE_DIR)
    tpsl_rows = []
    for policy, signal_col in policy_map.items():
        tpsl_rows.extend(
            run_tpsl_low_regime(stack_df, policy, signal_col, data_df, aeval)
        )
    tpsl_df = pd.DataFrame(tpsl_rows)
    tpsl_path = OUTPUT_DIR / "tpsl_results_low_regime.csv"
    tpsl_df.to_csv(tpsl_path, index=False)

    output_paths = [
        anomaly_path,
        merged_path,
        stack_path,
        summary_path,
        regime_summary_path,
        cluster_summary_path,
        tpsl_path,
    ]
    print("Outputs:")
    for path in output_paths:
        print(path.resolve())
    print("")
    print("stacking_summary_mode1_walkforward.csv")
    print(summary_df.to_csv(index=False))


if __name__ == "__main__":
    main()
