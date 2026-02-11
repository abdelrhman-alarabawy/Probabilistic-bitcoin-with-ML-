from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def discover_groups(labeled_root: Path) -> List[Path]:
    return sorted([p for p in labeled_root.iterdir() if p.is_dir() and p.name.lower() != "all"])


def _detect_timestamp_column(columns: List[str]) -> Optional[str]:
    if "timestamp" in columns:
        return "timestamp"
    for candidate in ["datetime", "time", "date", "ts_utc"]:
        if candidate in columns:
            return candidate
    for col in columns:
        if "timestamp" in col.lower():
            return col
    return None


def load_group_labeled(
    group_dir: Path,
    model_rank: int,
) -> Tuple[str, pd.DataFrame]:
    group_name = group_dir.name
    model_dir = group_dir / f"model_rank{model_rank:02d}"
    labeled_path = model_dir / "labeled.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled.csv for {group_name} rank {model_rank}: {labeled_path}")

    df = pd.read_csv(labeled_path)
    ts_col = _detect_timestamp_column(list(df.columns))
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {labeled_path}")
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    for required in ["gmm_hard_state", "trade_label"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column '{required}' in {labeled_path}")
    if "gmm_probmax" not in df.columns:
        logging.warning("%s missing gmm_probmax; gate_prob will be neutral.", labeled_path)
    if "gmm_entropy" not in df.columns:
        logging.warning("%s missing gmm_entropy; gate_entropy will be neutral.", labeled_path)

    df["trade_label"] = df["trade_label"].astype(str).str.strip().str.lower()
    df.loc[~df["trade_label"].isin(["long", "short", "skip"]), "trade_label"] = "skip"

    return group_name, df


def align_groups(
    group_frames: Dict[str, pd.DataFrame],
    align_mode: str = "intersection",
) -> pd.DataFrame:
    align_mode = align_mode.lower()
    if align_mode not in {"intersection", "union"}:
        raise ValueError(f"Unsupported align_mode '{align_mode}'.")

    merged: Optional[pd.DataFrame] = None
    for group_name, df in group_frames.items():
        cols = ["timestamp", "gmm_hard_state", "gmm_probmax", "gmm_entropy", "trade_label"]
        cols_present = [c for c in cols if c in df.columns]
        subset = df.loc[:, cols_present].copy()
        subset = subset.rename(
            columns={
                "gmm_hard_state": f"{group_name}_hard_state",
                "gmm_probmax": f"{group_name}_probmax",
                "gmm_entropy": f"{group_name}_entropy",
                "trade_label": f"{group_name}_trade_label",
            }
        )

        if merged is None:
            merged = subset
        else:
            how = "inner" if align_mode == "intersection" else "outer"
            merged = merged.merge(subset, on="timestamp", how=how)

    if merged is None:
        raise ValueError("No group frames to align.")

    merged = merged.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return merged

