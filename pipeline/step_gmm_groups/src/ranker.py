from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def rank_top_configs(summary_df: pd.DataFrame, ranking_cfg: Dict[str, object]) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    df = summary_df.copy()
    if "successful_runs" in df.columns:
        df = df[df["successful_runs"] > 0].copy()
    if df.empty:
        return df

    for col in ["test_avg_loglik_mean", "bic_train_mean", "avg_entropy_test_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    weights = dict(ranking_cfg.get("weights", {}))
    w_test = float(weights.get("test_avg_loglik_mean", 1.0))
    w_bic = float(weights.get("bic_train_mean", 1.0))
    w_entropy = float(weights.get("avg_entropy_test_mean", 1.0))

    df["rank_test_avg_loglik"] = df["test_avg_loglik_mean"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    df["rank_bic_train"] = df["bic_train_mean"].rank(ascending=True, method="min", na_option="bottom")
    df["rank_avg_entropy_test"] = df["avg_entropy_test_mean"].rank(
        ascending=True, method="min", na_option="bottom"
    )
    df["rank_score"] = (
        w_test * df["rank_test_avg_loglik"]
        + w_bic * df["rank_bic_train"]
        + w_entropy * df["rank_avg_entropy_test"]
    )

    df = df.sort_values(
        by=["rank_score", "test_avg_loglik_mean", "bic_train_mean", "avg_entropy_test_mean"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["overall_rank"] = np.arange(1, len(df) + 1, dtype=int)

    top_n = int(ranking_cfg.get("top_n", 10))
    return df.head(top_n).copy()

