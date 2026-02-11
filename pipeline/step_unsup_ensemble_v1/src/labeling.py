from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_label_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Labeling script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("signals_code_hour_version_1_0", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "label_dataframe"):
        raise AttributeError("Labeling script must expose label_dataframe(df, ...).")
    return module


def apply_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    label_cfg = cfg["labeling"]
    data_cfg = cfg["data"]
    module = _load_label_script(Path(label_cfg["script_path"]))

    ts_col = data_cfg["timestamp_col"]
    label_col = data_cfg["label_col"]

    labeled = module.label_dataframe(
        df.copy(),
        timestamp_col=ts_col,
        horizon_minutes=label_cfg.get("horizon_minutes", 720),
        tp_points=label_cfg.get("tp_points", 2000),
        sl_points=label_cfg.get("sl_points", 1000),
        five_min_csv=label_cfg.get("five_min_csv"),
    )

    if ts_col not in labeled.columns and "timestamp" in labeled.columns:
        labeled = labeled.rename(columns={"timestamp": ts_col})

    labeled[ts_col] = pd.to_datetime(labeled[ts_col], utc=True, errors="coerce")
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    if label_col in out.columns:
        out = out.drop(columns=[label_col])
    out = out.merge(labeled[[ts_col, label_col]], on=ts_col, how="left")
    if label_col not in out.columns or out[label_col].isna().any():
        raise ValueError("Labeling failed: candle_type missing or contains NaN.")

    values = set(out[label_col].dropna().astype(str).unique().tolist())
    required = {"long", "short", "skip"}
    if not required.issubset(values):
        raise ValueError(f"Labeling output missing required values. Found={values}")

    return out
