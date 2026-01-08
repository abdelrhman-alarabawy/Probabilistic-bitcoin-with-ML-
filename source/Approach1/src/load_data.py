from typing import Dict, Tuple

import pandas as pd

from .utils import resolve_all_candidates, resolve_column, resolve_optional_columns


REQUIRED_COLUMN_KEYS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_data(config: dict) -> Tuple[pd.DataFrame, Dict[str, str]]:
    paths = config["paths"]
    columns_cfg = config["columns"]

    df = pd.read_csv(paths["input_csv"], low_memory=False)
    resolved: Dict[str, str] = {}

    resolved["timestamp"] = resolve_column(
        df.columns,
        columns_cfg["timestamp_candidates"],
        required=True,
        label="timestamp",
    )
    resolved["open"] = resolve_column(
        df.columns,
        columns_cfg["open_candidates"],
        required=True,
        label="open",
    )
    resolved["high"] = resolve_column(
        df.columns,
        columns_cfg["high_candidates"],
        required=True,
        label="high",
    )
    resolved["low"] = resolve_column(
        df.columns,
        columns_cfg["low_candidates"],
        required=True,
        label="low",
    )
    resolved["close"] = resolve_column(
        df.columns,
        columns_cfg["close_candidates"],
        required=True,
        label="close",
    )
    resolved["volume"] = resolve_column(
        df.columns,
        columns_cfg["volume_candidates"],
        required=True,
        label="volume",
    )
    label_col = resolve_column(
        df.columns,
        columns_cfg.get("label_candidates", []),
        required=False,
        label="label",
    )
    if label_col:
        resolved["label"] = label_col

    optional_cfg = columns_cfg.get("optional", {})
    resolved_optional = resolve_optional_columns(df.columns, optional_cfg)
    if "atm_iv" in optional_cfg:
        resolved_optional["atm_iv_cols"] = resolve_all_candidates(
            df.columns, optional_cfg["atm_iv"]
        )

    ts_col = resolved["timestamp"]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    before_drop = len(df)
    df = df.dropna(subset=[ts_col])
    if len(df) < before_drop:
        print(f"Dropped {before_drop - len(df)} rows with invalid timestamps.")

    df = df.sort_values(ts_col)
    before_dupe = len(df)
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    if len(df) < before_dupe:
        print(f"Dropped {before_dupe - len(df)} duplicate timestamps.")

    non_numeric = {ts_col}
    if label_col:
        non_numeric.add(label_col)

    for col in df.columns:
        if col in non_numeric:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    resolved.update(resolved_optional)
    return df, resolved
