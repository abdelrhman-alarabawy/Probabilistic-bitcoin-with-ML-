from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ALLOWED_LABELS, AMBIGUOUS_COL, LABEL_COL, OHLCV_COLS, TIMESTAMP_CANDIDATES


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    timestamp_col: str
    dropped_duplicates: int


def _normalize(col: str) -> str:
    return "".join(ch for ch in col.lower().strip() if ch.isalnum() or ch == "_")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    normalized = {_normalize(col): col for col in columns}
    for cand in candidates:
        key = _normalize(cand)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing required column among candidates: {list(candidates)}")


def load_dataset(path: Path) -> LoadResult:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    ts_col = _find_column(df.columns, TIMESTAMP_CANDIDATES)
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
        ts_col = "timestamp"

    for col in OHLCV_COLS:
        if col not in df.columns:
            alt = _find_column(df.columns, [col])
            if alt != col:
                df = df.rename(columns={alt: col})

    if LABEL_COL not in df.columns:
        alt = _find_column(df.columns, [LABEL_COL])
        if alt != LABEL_COL:
            df = df.rename(columns={alt: LABEL_COL})

    if AMBIGUOUS_COL in df.columns:
        df[AMBIGUOUS_COL] = df[AMBIGUOUS_COL].astype(bool)
    else:
        for col in df.columns:
            if _normalize(col) == AMBIGUOUS_COL:
                df = df.rename(columns={col: AMBIGUOUS_COL})
                df[AMBIGUOUS_COL] = df[AMBIGUOUS_COL].astype(bool)
                break

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()
    invalid = sorted(set(df[LABEL_COL]) - set(ALLOWED_LABELS))
    if invalid:
        raise ValueError(f"Unexpected labels found: {invalid}")

    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    dropped = before - len(df)

    return LoadResult(df=df, timestamp_col="timestamp", dropped_duplicates=dropped)


def load_5m_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"5m CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    ts_col = _find_column(df.columns, TIMESTAMP_CANDIDATES)
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    for col in OHLCV_COLS:
        if col not in df.columns:
            alt = _find_column(df.columns, [col])
            if alt != col:
                df = df.rename(columns={alt: col})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df
