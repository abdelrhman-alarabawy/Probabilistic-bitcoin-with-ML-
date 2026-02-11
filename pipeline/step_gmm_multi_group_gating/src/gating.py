from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .thresholds import GroupThresholds, rarity_score


def apply_group_gate(
    df: pd.DataFrame,
    group_name: str,
    thresholds: GroupThresholds,
) -> Tuple[pd.Series, pd.Series]:
    state_col = f"{group_name}_hard_state"
    prob_col = f"{group_name}_probmax"
    ent_col = f"{group_name}_entropy"

    states = df[state_col].astype(float)
    state_freq = thresholds.state_freq

    if prob_col in df.columns:
        prob = pd.to_numeric(df[prob_col], errors="coerce")
        pass_prob = prob >= thresholds.probmax_thr
        denom = max(1.0 - thresholds.probmax_thr, 1e-9)
        s_prob = ((prob - thresholds.probmax_thr) / denom).clip(0.0, 1.0)
    else:
        pass_prob = pd.Series(True, index=df.index)
        s_prob = pd.Series(0.5, index=df.index)

    if ent_col in df.columns:
        ent = pd.to_numeric(df[ent_col], errors="coerce")
        pass_ent = ent <= thresholds.entropy_thr
        denom = max(thresholds.entropy_thr, 1e-9)
        s_ent = ((thresholds.entropy_thr - ent) / denom).clip(0.0, 1.0)
    else:
        pass_ent = pd.Series(True, index=df.index)
        s_ent = pd.Series(0.5, index=df.index)

    freq_values = states.map(lambda x: state_freq.get(int(x), 0.0) if pd.notna(x) else 0.0)
    pass_rare = freq_values <= thresholds.rarity_thr
    s_rare = freq_values.map(lambda f: rarity_score(float(f), thresholds.rarity_thr))

    gate_pass = pass_prob & pass_ent & pass_rare
    gate_score = (s_prob + s_ent + s_rare) / 3.0
    return gate_pass.astype(int), gate_score.astype(float)


def combine_gates(
    df: pd.DataFrame,
    group_names: list[str],
    mode: str,
    k_required: int,
    weights: Dict[str, float],
    score_threshold: float,
) -> Tuple[pd.Series, pd.Series]:
    mode = mode.lower()
    gate_cols = [f"{g}_gate_pass" for g in group_names]
    score_cols = [f"{g}_gate_score" for g in group_names]

    if mode == "strict_and":
        final_pass = (df[gate_cols].sum(axis=1) == len(gate_cols)).astype(int)
        final_score = pd.Series(np.nan, index=df.index)
        return final_pass, final_score

    if mode == "k_of_n":
        final_pass = (df[gate_cols].sum(axis=1) >= int(k_required)).astype(int)
        final_score = pd.Series(np.nan, index=df.index)
        return final_pass, final_score

    if mode == "weighted_score":
        w = {g: float(weights.get(g, 0.0)) for g in group_names}
        total = sum(w.values())
        if total <= 0:
            w = {g: 1.0 / len(group_names) for g in group_names}
        else:
            w = {g: w[g] / total for g in group_names}
        score = pd.Series(0.0, index=df.index)
        for g, col in zip(group_names, score_cols):
            score += w[g] * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        final_pass = (score >= float(score_threshold)).astype(int)
        return final_pass, score

    raise ValueError(f"Unsupported mode '{mode}'.")

