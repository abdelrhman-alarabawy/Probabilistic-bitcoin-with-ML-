from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class GroupInput:
    group_name: str
    csv_path: Path


def sanitize_group_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return sanitized or "group"


def discover_group_csvs(root: Path, expected_num_groups: Optional[int] = 6) -> List[GroupInput]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Group root folder does not exist or is not a directory: {root}")

    direct_csvs = sorted(root.glob("*.csv"))
    groups: List[GroupInput] = []

    if direct_csvs:
        for csv_path in direct_csvs:
            groups.append(GroupInput(group_name=sanitize_group_name(csv_path.stem), csv_path=csv_path))
    else:
        subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
        for folder in subfolders:
            csv_candidates = sorted(folder.glob("*.csv"))
            if not csv_candidates:
                csv_candidates = sorted(folder.rglob("*.csv"))
            if not csv_candidates:
                continue
            if len(csv_candidates) > 1:
                chosen = max(csv_candidates, key=lambda p: p.stat().st_size)
                logging.warning(
                    "Found multiple CSV files in %s; selecting largest file: %s",
                    folder,
                    chosen.name,
                )
            else:
                chosen = csv_candidates[0]
            groups.append(GroupInput(group_name=sanitize_group_name(folder.name), csv_path=chosen))

    if not groups:
        raise FileNotFoundError(
            f"No CSV groups found in {root}. Expected layout A (files) or layout B (subfolders)."
        )

    groups = sorted(groups, key=lambda item: item.group_name)
    if expected_num_groups is not None and len(groups) != expected_num_groups:
        logging.warning(
            "Expected %d groups but found %d in %s.",
            expected_num_groups,
            len(groups),
            root,
        )
    return groups


def detect_timestamp_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    original_cols = list(columns)
    lower_map = {col.lower(): col for col in original_cols}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in original_cols:
            if candidate_lower in col.lower():
                return col
    return None


def load_group_dataframe(
    csv_path: Path,
    timestamp_candidates: Sequence[str],
    sort_by_timestamp: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    timestamp_col = detect_timestamp_column(df.columns, timestamp_candidates)

    if timestamp_col is not None:
        parsed = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        df[timestamp_col] = parsed.dt.tz_localize(None)
        invalid = int(parsed.isna().sum())
        if invalid > 0:
            logging.warning(
                "%s has %d rows with invalid timestamps in column '%s'.",
                csv_path.name,
                invalid,
                timestamp_col,
            )
        if sort_by_timestamp:
            df = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    return df, timestamp_col


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

