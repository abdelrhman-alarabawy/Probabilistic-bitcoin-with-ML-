from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


try:
    from sklearn.cluster import KMeans

    KMEANS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    KMeans = None
    KMEANS_AVAILABLE = False

try:
    from sklearn.metrics import silhouette_score

    SIL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    silhouette_score = None
    SIL_AVAILABLE = False


@dataclass
class RegimeMappingResult:
    mapping: dict[int, str]
    regime_stats: pd.DataFrame
    k: int
    method: str


def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    def slope_func(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coef = np.polyfit(x, values, 1)
        return float(coef[0])

    return series.rolling(window=window, min_periods=2).apply(slope_func, raw=True)


def compute_state_stats(
    df: pd.DataFrame,
    state_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
    window: int = 10,
) -> pd.DataFrame:
    returns = df[close_col].pct_change().fillna(0.0)
    vol = returns.rolling(window=window, min_periods=2).std().fillna(0.0)
    range_pct = (df[high_col] - df[low_col]) / df[close_col].replace(0, np.nan)
    range_pct = range_pct.fillna(0.0)
    slope = rolling_slope(df[close_col], window=window).fillna(0.0)

    stats = pd.DataFrame(
        {
            "mean_return": returns,
            "volatility": vol,
            "range_pct": range_pct,
            "trend_slope": slope,
        }
    )
    stats[state_col] = df[state_col].values
    grouped = stats.groupby(state_col).mean()
    grouped["occupancy"] = df[state_col].value_counts(normalize=True).sort_index()
    return grouped.reset_index()


def _kmeans_cluster(stats_df: pd.DataFrame, k: int, random_state: int) -> np.ndarray:
    features = stats_df[["mean_return", "volatility", "range_pct", "trend_slope"]].values
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return model.fit_predict(features)


def choose_k(
    stats_df: pd.DataFrame,
    target_k: int,
    random_state: int,
) -> tuple[int, str]:
    if not KMEANS_AVAILABLE or len(stats_df) < 3:
        return target_k, "rule_based"

    candidate_ks = [k for k in range(2, min(5, len(stats_df)) + 1)]
    if target_k in candidate_ks:
        candidate_ks.insert(0, candidate_ks.pop(candidate_ks.index(target_k)))

    best_k = candidate_ks[0]
    best_score = -np.inf
    for k in candidate_ks:
        labels = _kmeans_cluster(stats_df, k, random_state)
        counts = pd.Series(labels).value_counts()
        if counts.min() < 1:
            continue
        score = 0.0
        if SIL_AVAILABLE and len(stats_df) > k:
            score = float(silhouette_score(stats_df[["mean_return", "volatility", "range_pct", "trend_slope"]], labels))
        score -= float((counts.min() == 1) * 0.5)
        if score > best_score:
            best_score = score
            best_k = k
    method = "kmeans" if best_k == target_k else "kmeans_adjusted"
    return best_k, method


def map_states_to_regimes(
    stats_df: pd.DataFrame,
    target_k: int = 3,
    random_state: int = 42,
) -> RegimeMappingResult:
    k, method = choose_k(stats_df, target_k, random_state)
    if KMEANS_AVAILABLE and k >= 2:
        labels = _kmeans_cluster(stats_df, k, random_state)
    else:
        labels = np.zeros(len(stats_df), dtype=int)
        k = 1
        method = "rule_based"

    stats_df = stats_df.copy()
    stats_df["cluster"] = labels

    cluster_vol = stats_df.groupby("cluster")["volatility"].mean().sort_values()
    cluster_order = list(cluster_vol.index)

    mapping: dict[int, str] = {}
    if k == 3:
        name_map = {
            cluster_order[0]: "stable",
            cluster_order[1]: "transition",
            cluster_order[2]: "extreme",
        }
    elif k == 2:
        name_map = {
            cluster_order[0]: "stable",
            cluster_order[1]: "extreme",
        }
    else:
        name_map = {cluster: f"regime_{idx}" for idx, cluster in enumerate(cluster_order)}

    for _, row in stats_df.iterrows():
        mapping[int(row["HMM_state"])] = name_map[int(row["cluster"])]

    return RegimeMappingResult(
        mapping=mapping,
        regime_stats=stats_df,
        k=k,
        method=method,
    )
