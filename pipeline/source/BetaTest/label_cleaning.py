from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

ALLOWED_LABELS = ("long", "short", "skip")


def _label_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in ALLOWED_LABELS}


def apply_label_cleaning(
    df: pd.DataFrame,
    label_col: str,
    ambig_col: str,
    force_ambiguous_to_skip: bool,
    min_range_filter: bool,
    min_range_pct: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' for cleaning.")

    cleaned = df.copy()
    original_labels = df[label_col].copy()
    before_counts = _label_counts(original_labels)

    step1_labels = original_labels.copy()
    ambiguous_true = 0
    changed_by_ambiguous = 0
    if force_ambiguous_to_skip and ambig_col in df.columns:
        ambiguous_mask = df[ambig_col].astype(bool)
        ambiguous_true = int(ambiguous_mask.sum())
        step1_labels = step1_labels.where(~ambiguous_mask, "skip")
        changed_by_ambiguous = int((step1_labels != original_labels).sum())

    range_pct_min = None
    range_pct_median = None
    range_pct_max = None
    range_filtered = 0
    changed_by_min_range = 0
    if min_range_filter:
        for col in ("open", "high", "low"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for range filter.")
        denom = df["open"].replace(0, pd.NA)
        range_pct = (df["high"] - df["low"]) / denom
        range_pct_min = float(range_pct.min(skipna=True)) if range_pct.notna().any() else None
        range_pct_median = float(range_pct.median(skipna=True)) if range_pct.notna().any() else None
        range_pct_max = float(range_pct.max(skipna=True)) if range_pct.notna().any() else None
        range_mask = (range_pct < min_range_pct).fillna(False)
        range_filtered = int(range_mask.sum())
        step2_labels = step1_labels.where(~range_mask, "skip")
        changed_by_min_range = int((step2_labels != step1_labels).sum())
    else:
        step2_labels = step1_labels

    cleaned[label_col] = step2_labels
    after_counts = _label_counts(cleaned[label_col])
    total_rows_changed = int((step2_labels != original_labels).sum())

    stats = {
        "min_range_pct": min_range_pct,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "ambiguous_true": ambiguous_true,
        "num_changed_by_ambiguous": changed_by_ambiguous,
        "num_changed_by_min_range": changed_by_min_range,
        "total_rows_changed": total_rows_changed,
        "range_pct_min": range_pct_min,
        "range_pct_median": range_pct_median,
        "range_pct_max": range_pct_max,
        "range_filtered": range_filtered,
    }
    return cleaned, stats
