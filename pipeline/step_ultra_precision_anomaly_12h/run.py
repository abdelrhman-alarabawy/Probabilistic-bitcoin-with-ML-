from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.anomaly_detectors import (
    fit_gmm,
    fit_isolation_forest,
    rolling_robust_zscores,
    score_gmm,
    score_isolation_forest,
)
from src.backtest import combine_signals, evaluate_rule_on_df, generate_splits
from src.data import prepare_dataset
from src.reporting import plot_equity_curve, plot_event_timeline, plot_precision_vs_coverage, save_rules_library
from src.rule_search import (
    Rule,
    discover_rules_from_clusters,
    discover_rules_from_score,
    rule_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"
OUTPUT_DIR = REPO_ROOT / "pipeline" / "step_ultra_precision_anomaly_12h"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

FEE_PER_TRADE = 0.0005
MAX_TRADES_PER_MONTH = 2.0
PRECISION_TARGET = 0.90
MIN_EVENTS = 3


def months_covered(timestamps: pd.Series) -> int:
    if timestamps.empty:
        return 1
    return max(1, int(timestamps.dt.to_period("M").nunique()))


def _rank_rules(rule_summary: pd.DataFrame) -> pd.DataFrame:
    df = rule_summary.copy()
    df["fold_penalty"] = np.where(df["fold_count"] <= 1, 0.05, 0.0)
    df["rank_score"] = df["precision_mean"] - df["fold_penalty"]
    df = df.sort_values(
        ["rank_score", "precision_mean", "trades_per_month_mean"],
        ascending=[False, False, True],
    )
    return df


def evaluate_mode(
    df: pd.DataFrame,
    feature_cols: List[str],
    rz_score_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    if mode == "expanding":
        splits = generate_splits(
            df,
            mode="expanding",
            train_months=12,
            test_months=2,
            step_months=2,
            min_train_months=12,
        )
    else:
        splits = generate_splits(
            df,
            mode="rolling",
            train_months=12,
            test_months=2,
            step_months=2,
            min_train_months=12,
        )

    all_events: List[Dict[str, object]] = []
    all_trades: List[pd.DataFrame] = []
    rule_perf_rows: List[Dict[str, object]] = []
    combined_rows: List[Dict[str, object]] = []
    rule_meta: Dict[str, Dict[str, object]] = {}

    for split in splits:
        train_df = df.loc[split.train_idx].copy()
        test_df = df.loc[split.test_idx].copy()
        months_in_train = len(split.train_months)
        months_in_test = len(split.test_months)

        iforest = fit_isolation_forest(train_df[feature_cols])
        train_df["score_iforest"] = score_isolation_forest(iforest, train_df[feature_cols])
        test_df["score_iforest"] = score_isolation_forest(iforest, test_df[feature_cols])

        gmm = fit_gmm(train_df[feature_cols])
        train_labels, train_conf, train_rarity = score_gmm(gmm, train_df[feature_cols])
        test_labels, test_conf, test_rarity = score_gmm(gmm, test_df[feature_cols])
        train_df["gmm_cluster_id"] = train_labels
        train_df["gmm_rarity"] = train_rarity
        test_df["gmm_cluster_id"] = test_labels
        test_df["gmm_rarity"] = test_rarity
        train_df["gmm_conf"] = train_conf
        test_df["gmm_conf"] = test_conf

        rules: List[Rule] = []
        for score_col in rz_score_cols:
            rules.extend(
                discover_rules_from_score(
                    train_df,
                    score_col,
                    "robust_z",
                    feature_cols,
                    months_in_train,
                    min_events=MIN_EVENTS,
                    max_trades_per_month=MAX_TRADES_PER_MONTH,
                    precision_target=PRECISION_TARGET,
                )
            )
        rules.extend(
            discover_rules_from_score(
                train_df,
                "score_iforest",
                "iforest",
                feature_cols,
                months_in_train,
                min_events=MIN_EVENTS,
                max_trades_per_month=MAX_TRADES_PER_MONTH,
                precision_target=PRECISION_TARGET,
            )
        )
        rules.extend(
            discover_rules_from_clusters(
                train_df,
                "gmm_cluster_id",
                gmm.train_freq,
                "gmm",
                feature_cols,
                months_in_train,
                min_events=MIN_EVENTS,
                max_trades_per_month=MAX_TRADES_PER_MONTH,
                precision_target=PRECISION_TARGET,
            )
        )

        rules = sorted(
            rules,
            key=lambda r: (-r.precision_train, r.trades_per_month_train),
        )[:25]

        print(f"{mode} fold {split.fold_id}: {len(rules)} candidate rules")
        if not rules:
            print(
                "  No rules met precision/frequency thresholds on train; "
                "combined strategy will have 0 trades."
            )

        events_by_rule: List[Tuple[Rule, pd.DataFrame]] = []
        for rule in rules:
            if rule.rule_id not in rule_meta:
                rule_meta[rule.rule_id] = rule_to_dict(rule)
            df_events, metrics = evaluate_rule_on_df(rule, test_df, FEE_PER_TRADE)
            print(
                f"  {rule.rule_id} -> test trades: {metrics['trade_count']}, "
                f"test precision: {metrics['precision']:.3f}"
            )
            if df_events.empty:
                continue

            events_by_rule.append((rule, df_events))
            rule_perf_rows.append(
                {
                    "mode": mode,
                    "fold_id": split.fold_id,
                    "rule_id": rule.rule_id,
                    "precision": metrics["precision"],
                    "trade_count": metrics["trade_count"],
                    "trades_per_month": float(metrics["trade_count"] / months_in_test),
                    "avg_pnl": metrics["avg_pnl"],
                    "profit_factor": metrics["profit_factor"],
                    "max_drawdown": metrics["max_drawdown"],
                    "cvar95": metrics["cvar95"],
                    "event_count_train": rule.event_count_train,
                    "event_count_test": metrics["trade_count"],
                }
            )

            for _, row in df_events.iterrows():
                score = row[rule.score_col] if rule.score_col else row.get("gmm_rarity", np.nan)
                cluster_id = int(row["gmm_cluster_id"]) if "gmm_cluster_id" in row else None
                direction = rule.direction
                if direction == "long":
                    mfe = row["mfe_long"]
                    mae = row["mae_long"]
                else:
                    mfe = row["mfe_short"]
                    mae = row["mae_short"]
                all_events.append(
                    {
                        "timestamp": row["timestamp"],
                        "event_type": rule.rule_id,
                        "score": float(score) if np.isfinite(score) else np.nan,
                        "cluster_id": cluster_id,
                        "regime": None,
                        "direction_candidate": direction,
                        "forward_return_12h": row["forward_return_12h"],
                        "forward_return_24h": row["forward_return_24h"],
                        "forward_return_72h": row["forward_return_72h"],
                        "max_favorable_excursion": mfe,
                        "max_adverse_excursion": mae,
                        "meta_json": json.dumps(
                            {
                                "mode": mode,
                                "fold_id": split.fold_id,
                                "horizon": rule.horizon,
                                "tp": rule.tp,
                                "sl": rule.sl,
                            }
                        ),
                    }
                )

        trades, combo_metrics = combine_signals(events_by_rule, FEE_PER_TRADE)
        if trades.empty:
            if rules and not events_by_rule:
                print("  Combined strategy: 0 trades (no rule triggers in test window).")
            elif rules and events_by_rule:
                print("  Combined strategy: 0 trades (conflicting signals removed).")
            elif not rules:
                print("  Combined strategy: 0 trades (no rules selected on train).")
            combined_rows.append(
                {
                    "mode": mode,
                    "fold_id": split.fold_id,
                    "stat": "fold",
                    "precision_long": float("nan"),
                    "precision_short": float("nan"),
                    "trades_per_month": 0.0,
                    "coverage": 0.0,
                    "avg_pnl_per_trade": float("nan"),
                    "profit_factor": float("nan"),
                    "max_drawdown": float("nan"),
                    "cvar95": float("nan"),
                    "worst_fold_precision": float("nan"),
                }
            )
        else:
            trades = trades.copy()
            trades["mode"] = mode
            trades["fold_id"] = split.fold_id
            all_trades.append(trades)
            combined_rows.append(
                {
                    "mode": mode,
                    "fold_id": split.fold_id,
                    "stat": "fold",
                    "precision_long": combo_metrics["precision_long"],
                    "precision_short": combo_metrics["precision_short"],
                    "trades_per_month": float(len(trades) / months_in_test),
                    "coverage": float(len(trades) / len(test_df)),
                    "avg_pnl_per_trade": combo_metrics["avg_pnl"],
                    "profit_factor": combo_metrics["profit_factor"],
                    "max_drawdown": combo_metrics["max_drawdown"],
                    "cvar95": combo_metrics["cvar95"],
                    "worst_fold_precision": float(
                        np.nanmin([combo_metrics["precision_long"], combo_metrics["precision_short"]])
                    ),
                }
            )

    rule_perf_df = pd.DataFrame(rule_perf_rows)
    combined_df = pd.DataFrame(combined_rows)
    events_df = pd.DataFrame(all_events)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    if rule_perf_df.empty:
        rule_summary = pd.DataFrame(
            columns=[
                "rule_id",
                "precision_mean",
                "precision_std",
                "trades_per_month_mean",
                "avg_pnl_mean",
                "max_drawdown_mean",
                "cvar95_mean",
                "fold_count",
                "worst_fold_precision",
                "event_count_train",
                "event_count_test",
            ]
        )
    else:
        rule_summary = (
            rule_perf_df.groupby("rule_id")
            .agg(
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                trades_per_month_mean=("trades_per_month", "mean"),
                avg_pnl_mean=("avg_pnl", "mean"),
                max_drawdown_mean=("max_drawdown", "mean"),
                cvar95_mean=("cvar95", "mean"),
                fold_count=("fold_id", "nunique"),
                worst_fold_precision=("precision", "min"),
                event_count_train=("event_count_train", "mean"),
                event_count_test=("event_count_test", "mean"),
            )
            .reset_index()
        )

    return rule_perf_df, rule_summary, combined_df, events_df, trades_df, list(rule_meta.values())


def _add_backtest_overall(combined: pd.DataFrame) -> pd.DataFrame:
    rows = [combined]
    for mode, group in combined[combined["stat"] == "fold"].groupby("mode"):
        mean_row = {
            "mode": mode,
            "fold_id": None,
            "stat": "mean",
            "precision_long": group["precision_long"].mean(),
            "precision_short": group["precision_short"].mean(),
            "trades_per_month": group["trades_per_month"].mean(),
            "coverage": group["coverage"].mean(),
            "avg_pnl_per_trade": group["avg_pnl_per_trade"].mean(),
            "profit_factor": group["profit_factor"].mean(),
            "max_drawdown": group["max_drawdown"].mean(),
            "cvar95": group["cvar95"].mean(),
            "worst_fold_precision": group["worst_fold_precision"].min(),
        }
        std_row = {
            "mode": mode,
            "fold_id": None,
            "stat": "std",
            "precision_long": group["precision_long"].std(),
            "precision_short": group["precision_short"].std(),
            "trades_per_month": group["trades_per_month"].std(),
            "coverage": group["coverage"].std(),
            "avg_pnl_per_trade": group["avg_pnl_per_trade"].std(),
            "profit_factor": group["profit_factor"].std(),
            "max_drawdown": group["max_drawdown"].std(),
            "cvar95": group["cvar95"].std(),
            "worst_fold_precision": group["worst_fold_precision"].min(),
        }
        rows.extend([pd.DataFrame([mean_row]), pd.DataFrame([std_row])])

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataset(str(DATA_PATH))
    df = prepared.df.copy()
    feature_cols = prepared.feature_cols

    zscores = rolling_robust_zscores(df, feature_cols, window=200).abs()
    rz_score_cols = []
    for col in zscores.columns:
        score_col = f"score_rz_{col}"
        df[score_col] = zscores[col].fillna(0.0)
        rz_score_cols.append(score_col)
    df["score_robustz"] = zscores.mean(axis=1, skipna=True).fillna(0.0)
    df["score_robustz_max"] = zscores.max(axis=1, skipna=True).fillna(0.0)
    rz_score_cols.extend(["score_robustz", "score_robustz_max"])

    print(f"Loaded {len(df)} rows with {len(feature_cols)} features.")

    rule_perf_exp, rule_summary_exp, combined_exp, events_exp, trades_exp, meta_exp = evaluate_mode(
        df, feature_cols, rz_score_cols, mode="expanding"
    )
    rule_perf_roll, rule_summary_roll, combined_roll, events_roll, trades_roll, meta_roll = evaluate_mode(
        df, feature_cols, rz_score_cols, mode="rolling"
    )

    rule_perf = pd.concat([rule_perf_exp, rule_perf_roll], ignore_index=True)
    rule_summary = pd.concat([rule_summary_exp, rule_summary_roll], ignore_index=True)
    combined = pd.concat([combined_exp, combined_roll], ignore_index=True)
    events_df = pd.concat([events_exp, events_roll], ignore_index=True)
    trades_df = pd.concat([trades_exp, trades_roll], ignore_index=True) if not trades_exp.empty or not trades_roll.empty else pd.DataFrame()

    combined = _add_backtest_overall(combined)
    rule_summary_ranked = _rank_rules(rule_summary)
    rule_summary_filtered = rule_summary_ranked[
        (rule_summary_ranked["precision_mean"] >= PRECISION_TARGET)
        & (rule_summary_ranked["worst_fold_precision"] >= PRECISION_TARGET)
    ].copy()

    event_columns = [
        "timestamp",
        "event_type",
        "score",
        "cluster_id",
        "regime",
        "direction_candidate",
        "forward_return_12h",
        "forward_return_24h",
        "forward_return_72h",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "meta_json",
    ]
    events_df = events_df.reindex(columns=event_columns)

    meta_map = {row["rule_id"]: row for row in meta_exp + meta_roll}
    rule_library = []
    for _, row in rule_summary_filtered.iterrows():
        meta = meta_map.get(row["rule_id"], {})
        rule_library.append(
            {
                "rule_id": row["rule_id"],
                "condition": meta.get("condition", ""),
                "features": meta.get("features", []),
                "direction": meta.get("direction", ""),
                "threshold": meta.get("threshold", ""),
                "expected_hold_horizon": meta.get("horizon", ""),
                "tp": meta.get("tp", ""),
                "sl": meta.get("sl", ""),
                "crossval_metrics": {
                    "precision": row["precision_mean"],
                    "trades_per_month": row["trades_per_month_mean"],
                    "avg_pnl": row["avg_pnl_mean"],
                    "max_drawdown": row["max_drawdown_mean"],
                    "cvar95": row["cvar95_mean"],
                    "worst_fold_precision": row["worst_fold_precision"],
                    "fold_count": row["fold_count"],
                },
                "event_count_train": row["event_count_train"],
                "event_count_test": row["event_count_test"],
                "time_discovered": datetime.utcnow().isoformat() + "Z",
                "folds_used": "expanding+rolling",
            }
        )

    events_df.to_csv(RESULTS_DIR / "anomaly_events.csv", index=False)
    rule_perf.to_csv(RESULTS_DIR / "rule_performance.csv", index=False)
    save_rules_library(rule_library, RESULTS_DIR / "rules_library.json")
    combined.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)

    top_rules = rule_summary_ranked.head(10)
    print("Top 10 rules (all):")
    if top_rules.empty:
        print("  No rules found.")
    for _, row in top_rules.iterrows():
        print(
            f"{row['rule_id']} | precision={row['precision_mean']:.3f} "
            f"| trades/mo={row['trades_per_month_mean']:.2f} "
            f"| avg_pnl={row['avg_pnl_mean']:.4f} "
            f"| worst_fold_precision={row['worst_fold_precision']:.3f} "
            f"| train_events={row['event_count_train']:.1f} "
            f"| test_events={row['event_count_test']:.1f}"
        )

    print("Top 10 rules (filtered by precision target):")
    if rule_summary_filtered.empty:
        print("  No rules met the precision target on test folds.")
    for _, row in rule_summary_filtered.head(10).iterrows():
        print(
            f"{row['rule_id']} | precision={row['precision_mean']:.3f} "
            f"| trades/mo={row['trades_per_month_mean']:.2f} "
            f"| avg_pnl={row['avg_pnl_mean']:.4f} "
            f"| worst_fold_precision={row['worst_fold_precision']:.3f} "
            f"| train_events={row['event_count_train']:.1f} "
            f"| test_events={row['event_count_test']:.1f}"
        )

    if not trades_df.empty:
        plot_equity_curve(trades_df, FIGURES_DIR / "equity_curve.png")
    if not events_df.empty:
        plot_event_timeline(events_df, FIGURES_DIR / "event_timeline.png")
    if not rule_summary.empty:
        plot_precision_vs_coverage(rule_summary, FIGURES_DIR / "precision_vs_coverage.png")

    print("Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
