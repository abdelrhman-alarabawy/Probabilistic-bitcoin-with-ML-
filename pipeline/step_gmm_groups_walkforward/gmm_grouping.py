from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class GroupDefinition:
    method: str
    group_id: str
    group_name: str
    features: List[str]


def feature_list_short(features: Sequence[str], max_len: int = 120) -> str:
    joined = "|".join(features)
    if len(joined) <= max_len:
        return joined
    return joined[: max_len - 3] + "..."


def _variance_order(feature_names: Sequence[str], variances: Optional[Dict[str, float]]) -> List[str]:
    if not variances:
        return list(feature_names)
    return sorted(feature_names, key=lambda f: variances.get(f, 0.0), reverse=True)


def build_corr_groups(
    feature_names: Sequence[str],
    abs_corr: np.ndarray,
    variances: Optional[Dict[str, float]] = None,
    min_size: int = 5,
    max_size: int = 10,
    logger: Optional[object] = None,
) -> List[GroupDefinition]:
    if len(feature_names) == 0:
        return []
    if abs_corr.shape[0] != len(feature_names):
        raise ValueError("abs_corr size does not match feature_names length.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    remaining = set(feature_names)
    order = _variance_order(feature_names, variances)

    groups: List[List[str]] = []
    used = set()

    for seed in order:
        if seed not in remaining:
            continue
        remaining.remove(seed)
        if seed in used:
            continue

        seed_idx = name_to_idx[seed]
        candidates = list(remaining)
        candidates.sort(key=lambda f: abs_corr[seed_idx, name_to_idx[f]], reverse=True)

        target_size = min(max_size, 1 + len(candidates))
        if target_size < min_size:
            remaining.add(seed)
            break

        group_feats = [seed] + candidates[: target_size - 1]
        for f in group_feats:
            if f in remaining:
                remaining.remove(f)
            used.add(f)

        groups.append(group_feats)

    leftovers = list(remaining)
    if leftovers and groups:
        leftovers = _variance_order(leftovers, variances)
        for f in leftovers:
            best_group_idx = None
            best_score = -1.0
            f_idx = name_to_idx[f]
            for i, grp in enumerate(groups):
                if len(grp) >= max_size:
                    continue
                grp_idx = [name_to_idx[g] for g in grp]
                score = float(np.mean(abs_corr[f_idx, grp_idx]))
                if score > best_score:
                    best_score = score
                    best_group_idx = i
            if best_group_idx is None:
                if logger is not None:
                    logger.warning("Dropping leftover corr feature %s (no group capacity).", f)
            else:
                groups[best_group_idx].append(f)

    corr_groups: List[GroupDefinition] = []
    for i, grp in enumerate(groups, start=1):
        if len(grp) < min_size:
            if logger is not None:
                logger.warning("Dropping corr group %d with size %d (<%d).", i, len(grp), min_size)
            continue
        group_id = f"corr_{i}"
        corr_groups.append(
            GroupDefinition(method="corr", group_id=group_id, group_name=group_id, features=list(grp))
        )

    return corr_groups


def build_domain_groups(
    feature_names: Sequence[str],
    variances: Optional[Dict[str, float]],
    abs_corr: Optional[np.ndarray] = None,
    min_size: int = 5,
    max_size: int = 10,
    logger: Optional[object] = None,
) -> List[GroupDefinition]:
    patterns = [
        ("options_iv", ["iv", "implied", "skew"]),
        ("momentum", ["rsi", "macd", "roc", "mom"]),
        ("volatility", ["atr", "vol", "bb", "band"]),
        ("liquidity_flow", ["liq", "volume", "flow"]),
        ("trend_ma", ["sma", "ema", "ma_"]),
    ]

    assigned = set()
    groups: Dict[str, List[str]] = {}

    for group_name, keys in patterns:
        matched = []
        for f in feature_names:
            if f in assigned:
                continue
            lower = f.lower()
            if any(k in lower for k in keys):
                matched.append(f)
                assigned.add(f)
        if matched:
            groups[group_name] = matched

    leftovers = [f for f in feature_names if f not in assigned]
    if leftovers:
        groups["other"] = leftovers

    for name, feats in list(groups.items()):
        if len(feats) > max_size:
            ordered = _variance_order(feats, variances)
            groups[name] = ordered[:max_size]

    if abs_corr is not None and groups:
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

        def avg_corr(a: List[str], b: List[str]) -> float:
            if not a or not b:
                return -1.0
            idx_a = [name_to_idx[x] for x in a]
            idx_b = [name_to_idx[x] for x in b]
            return float(np.mean(abs_corr[np.ix_(idx_a, idx_b)]))

        while True:
            small = [k for k, v in groups.items() if len(v) < min_size]
            if not small or len(groups) == 1:
                break
            small.sort(key=lambda k: len(groups[k]))
            gname = small[0]
            best_other = None
            best_score = -1.0
            for other in groups.keys():
                if other == gname:
                    continue
                score = avg_corr(groups[gname], groups[other])
                if score > best_score:
                    best_score = score
                    best_other = other
            if best_other is None:
                break
            merged = groups[best_other] + groups[gname]
            ordered = _variance_order(merged, variances)
            groups[best_other] = ordered[:max_size]
            del groups[gname]

    domain_groups: List[GroupDefinition] = []
    for name, feats in groups.items():
        if len(feats) < min_size:
            if logger is not None:
                logger.warning("Dropping domain group %s with size %d (<%d).", name, len(feats), min_size)
            continue
        group_id = f"domain_{name}"
        group_name = name
        domain_groups.append(
            GroupDefinition(method="domain", group_id=group_id, group_name=group_name, features=list(feats))
        )

    return domain_groups
