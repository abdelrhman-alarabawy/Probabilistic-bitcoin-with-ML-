from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import resolve_metric, softmax


def compute_model_scores(metrics_df: pd.DataFrame, score_components: dict[str, float]) -> np.ndarray:
    scores = []
    for _, row in metrics_df.iterrows():
        row_dict = row.to_dict()
        score = 0.0
        for key, coef in score_components.items():
            val = resolve_metric(row_dict, key)
            if np.isnan(val):
                continue
            score += float(coef) * float(val)
        scores.append(score)
    return np.array(scores, dtype=float)


def compute_model_weights(metrics_df: pd.DataFrame, weights_cfg: dict) -> pd.DataFrame:
    alpha = float(weights_cfg.get("alpha", 1.0))
    components = weights_cfg.get("score_components", {})
    scores = compute_model_scores(metrics_df, components)
    weights = softmax(alpha * scores)
    out = metrics_df.copy()
    out["ensemble_score"] = scores
    out["ensemble_weight"] = weights
    return out
