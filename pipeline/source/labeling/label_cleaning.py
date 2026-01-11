from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import ALLOWED_LABELS
from .label_quality import counts_from_series


class LabelCleaningError(ValueError):
    pass


def apply_high_precision_cleaning(
    df: pd.DataFrame,
    label_col: str = "candle_type",
    ambiguous_col: str = "ambiguous_flag",
    min_range_pct: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    if label_col not in df.columns:
        raise LabelCleaningError(f"Missing label column '{label_col}'.")
    if ambiguous_col not in df.columns:
        raise LabelCleaningError(f"Missing ambiguous flag column '{ambiguous_col}'.")

    cleaned = df.copy()
    before_counts = counts_from_series(cleaned[label_col], ALLOWED_LABELS)

    ambiguous_mask = cleaned[ambiguous_col].astype(bool)
    cleaned.loc[ambiguous_mask, label_col] = "skip"

    range_mask = None
    if min_range_pct is not None:
        for col in ("open", "high", "low"):
            if col not in cleaned.columns:
                raise LabelCleaningError(f"Missing required column '{col}' for range filter.")
        denom = cleaned["open"].replace(0, pd.NA)
        range_pct = (cleaned["high"] - cleaned["low"]) / denom
        range_mask = range_pct < min_range_pct
        range_mask = range_mask.fillna(False)
        cleaned.loc[range_mask, label_col] = "skip"

    after_counts = counts_from_series(cleaned[label_col], ALLOWED_LABELS)

    changed_count = int((cleaned[label_col] != df[label_col]).sum())
    stats = {
        "before_counts": before_counts,
        "after_counts": after_counts,
        "changed_rows": changed_count,
        "min_range_pct": min_range_pct,
        "ambiguous_skipped": int(ambiguous_mask.sum()),
        "range_filtered": int(range_mask.sum()) if range_mask is not None else 0,
    }
    return cleaned, stats