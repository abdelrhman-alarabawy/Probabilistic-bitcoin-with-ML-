from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .rolling import WindowSlice


def build_gate_scores(
    window: WindowSlice,
    timestamps: np.ndarray,
    p_trade: np.ndarray,
    k_default: int,
) -> pd.DataFrame:
    n = len(p_trade)
    p_trade_safe = np.where(np.isfinite(p_trade), p_trade, -np.inf)
    ranks = pd.Series(p_trade_safe).rank(method="first", ascending=False).astype(int).values
    topk = min(k_default, n)
    order = np.argsort(p_trade_safe)[::-1]
    selected = np.zeros(n, dtype=int)
    if topk > 0:
        selected[order[:topk]] = 1
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "p_trade": p_trade,
            "gate_rank": ranks,
            "selected_topk_flag": selected,
            "window_id": window.window_id,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "test_start": window.test_start,
            "test_end": window.test_end,
        }
    )


def concat_scores(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
