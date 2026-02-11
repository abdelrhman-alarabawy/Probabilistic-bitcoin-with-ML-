from __future__ import annotations

from typing import Dict, List

import pandas as pd


def compute_pass_rates(df: pd.DataFrame, group_names: List[str]) -> pd.DataFrame:
    rows = []
    for g in group_names:
        col = f"{g}_gate_pass"
        if col not in df.columns:
            continue
        rate = float(df[col].mean()) if df.shape[0] > 0 else 0.0
        rows.append({"group_name": g, "pass_rate": rate})
    return pd.DataFrame(rows)


def compute_label_agreement(df: pd.DataFrame, group_names: List[str]) -> pd.DataFrame:
    records = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = group_names[i]
            g2 = group_names[j]
            c1 = f"{g1}_trade_label"
            c2 = f"{g2}_trade_label"
            if c1 not in df.columns or c2 not in df.columns:
                continue
            valid = df[[c1, c2]].dropna()
            if valid.empty:
                agreement = 0.0
                n = 0
            else:
                agreement = float((valid[c1] == valid[c2]).mean())
                n = int(valid.shape[0])
            records.append(
                {
                    "group_a": g1,
                    "group_b": g2,
                    "agreement_rate": agreement,
                    "n_samples": n,
                }
            )
    return pd.DataFrame(records)


def summarize_final_labels(df: pd.DataFrame) -> Dict[str, float]:
    total = int(df.shape[0])
    counts = df["final_label"].value_counts().to_dict() if total > 0 else {}
    long_c = int(counts.get("long", 0))
    short_c = int(counts.get("short", 0))
    skip_c = int(counts.get("skip", 0))
    return {
        "final_long_pct": (long_c / total) if total else 0.0,
        "final_short_pct": (short_c / total) if total else 0.0,
        "final_skip_pct": (skip_c / total) if total else 0.0,
        "final_long_count": long_c,
        "final_short_count": short_c,
        "final_skip_count": skip_c,
        "final_total": total,
    }
