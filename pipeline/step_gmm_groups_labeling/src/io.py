from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class GroupFile:
    group_name: str
    csv_path: Path


def sanitize_group_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return clean or "group"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_timestamp_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    col_map = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    for candidate in candidates:
        cl = candidate.lower()
        for col in cols:
            if cl in col.lower():
                return col
    return None


def discover_group_csvs(root: Path, expected_groups: Optional[int] = 6) -> List[GroupFile]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Group root folder not found: {root}")

    groups: List[GroupFile] = []
    direct_csvs = sorted(root.glob("*.csv"))
    if direct_csvs:
        for path in direct_csvs:
            groups.append(GroupFile(group_name=sanitize_group_name(path.stem), csv_path=path))
    else:
        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        for sub in subdirs:
            csvs = sorted(sub.glob("*.csv"))
            if not csvs:
                csvs = sorted(sub.rglob("*.csv"))
            if not csvs:
                continue
            if len(csvs) > 1:
                chosen = max(csvs, key=lambda p: p.stat().st_size)
                logging.warning("Multiple CSV files in %s, selecting %s.", sub, chosen.name)
            else:
                chosen = csvs[0]
            groups.append(GroupFile(group_name=sanitize_group_name(sub.name), csv_path=chosen))

    groups = sorted(groups, key=lambda g: g.group_name)
    if expected_groups is not None and len(groups) != expected_groups:
        logging.warning("Expected %d groups but discovered %d in %s.", expected_groups, len(groups), root)
    if not groups:
        raise FileNotFoundError(f"No group CSVs found in {root}.")
    return groups


def load_group_dataframe(
    csv_path: Path,
    timestamp_candidates: Sequence[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    ts_col = detect_timestamp_column(df.columns, timestamp_candidates)
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df[ts_col] = ts.dt.tz_localize(None)
        bad = int(ts.isna().sum())
        if bad > 0:
            logging.warning("%s has %d invalid timestamps in '%s'.", csv_path.name, bad, ts_col)
        df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    return df, ts_col


def discover_prev_groups(prev_results_root: Path) -> List[str]:
    if not prev_results_root.exists():
        raise FileNotFoundError(f"Previous results root not found: {prev_results_root}")
    groups = sorted([p.name for p in prev_results_root.iterdir() if p.is_dir() and (p / "top10.csv").exists()])
    return groups


def load_top10(top10_path: Path, top_n: int = 10) -> pd.DataFrame:
    if not top10_path.exists():
        raise FileNotFoundError(f"top10.csv not found: {top10_path}")
    df = pd.read_csv(top10_path)
    if df.empty:
        return df
    if "overall_rank" in df.columns:
        df = df.sort_values("overall_rank", kind="mergesort")
    df = df.head(top_n).copy()
    df = df.reset_index(drop=True)
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    col_map = {c.lower(): c for c in df.columns}
    out = df.copy()
    mapping: Dict[str, Optional[str]] = {}
    for canonical in ["open", "high", "low", "close", "volume"]:
        source = col_map.get(canonical)
        mapping[canonical] = source
        if source is not None and source != canonical:
            out = out.rename(columns={source: canonical})
    return out, mapping


def parse_feature_signature(signature: object) -> List[str]:
    if signature is None:
        return []
    text = str(signature).strip()
    if not text:
        return []
    if text.lower() == "nan":
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def load_selected_features_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

