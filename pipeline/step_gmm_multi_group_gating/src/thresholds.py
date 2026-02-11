from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class GroupThresholds:
    probmax_thr: float
    entropy_thr: float
    rarity_thr: float
    state_freq: Dict[int, float]
    train_rows: int


def compute_state_freq(train_states: pd.Series) -> Dict[int, float]:
    counts = train_states.value_counts().to_dict()
    total = float(train_states.shape[0])
    if total <= 0:
        return {}
    return {int(k): float(v) / total for k, v in counts.items()}


def compute_thresholds(
    df: pd.DataFrame,
    prob_col: str,
    ent_col: str,
    state_col: str,
    train_fraction: float,
    prob_q: float,
    ent_q: float,
    rare_q: float,
) -> GroupThresholds:
    n = df.shape[0]
    cut = int(round(n * train_fraction))
    cut = max(1, min(cut, n))
    train = df.iloc[:cut]

    if prob_col in train.columns:
        prob_thr = float(train[prob_col].quantile(prob_q))
    else:
        prob_thr = 1.0

    if ent_col in train.columns:
        ent_thr = float(train[ent_col].quantile(ent_q))
    else:
        ent_thr = 0.0

    state_freq = compute_state_freq(train[state_col])
    if state_freq:
        rarity_thr = float(pd.Series(list(state_freq.values())).quantile(rare_q))
    else:
        rarity_thr = 0.0

    return GroupThresholds(
        probmax_thr=prob_thr,
        entropy_thr=ent_thr,
        rarity_thr=rarity_thr,
        state_freq=state_freq,
        train_rows=train.shape[0],
    )


def rarity_score(freq: float, rarity_thr: float) -> float:
    if rarity_thr <= 0:
        return 0.5
    return float(np.clip((rarity_thr - freq) / rarity_thr, 0.0, 1.0))

