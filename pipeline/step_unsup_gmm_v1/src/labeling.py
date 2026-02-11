from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import ensure_dir, setup_logging


logger = setup_logging()


def _load_labeler(script_path: Path):
    spec = importlib.util.spec_from_file_location("signals_code_hour_version_1_0", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load labeling script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "label_dataframe"):
        raise AttributeError("Labeling script must expose label_dataframe().")
    return module


def label_top_models(config_path: str | Path) -> dict:
    cfg = load_config(config_path)

    output_root = Path(cfg["output"]["root_dir"])
    top_dir = output_root / "top_models"
    labeled_dir = ensure_dir(output_root / "labeled")

    if not top_dir.exists():
        raise FileNotFoundError(f"Top models directory not found: {top_dir}. Run export-top first.")

    script_path = Path("scripts/signals_code_hour_version_1_0.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Labeling script not found: {script_path}")

    labeler = _load_labeler(script_path)

    horizon = cfg["labeling"].get("horizon_minutes", 720)
    tp_points = cfg["labeling"].get("tp_points", 2000)
    sl_points = cfg["labeling"].get("sl_points", 1000)
    five_min_csv = cfg["labeling"].get("five_min_csv")

    labeled_paths = []

    for model_dir in sorted(top_dir.glob("fold_*")):
        source_path = model_dir / "candles_with_gmm_probs.csv"
        if not source_path.exists():
            continue

        df = pd.read_csv(source_path)
        df.columns = df.columns.str.strip()
        ts_col = cfg["data"]["timestamp_col"]
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        label_df = labeler.label_dataframe(
            df,
            timestamp_col=ts_col,
            horizon_minutes=horizon,
            tp_points=tp_points,
            sl_points=sl_points,
            five_min_csv=five_min_csv,
        )

        if ts_col not in label_df.columns and "timestamp" in label_df.columns:
            label_df = label_df.rename(columns={"timestamp": ts_col})

        if ts_col in label_df.columns:
            label_df[ts_col] = pd.to_datetime(label_df[ts_col], utc=True, errors="coerce")

        merged = df.merge(
            label_df[[ts_col, cfg["data"]["label_col"]]],
            on=ts_col,
            how="left",
        )

        if merged[cfg["data"]["label_col"]].isna().any():
            raise ValueError(f"Labeling failed for {source_path}. candle_type has NaNs.")

        model_id = model_dir.name
        out_path = labeled_dir / f"{model_id}__labeled.csv"
        merged.to_csv(out_path, index=False)
        labeled_paths.append(out_path)
        logger.info("Labeled %s", out_path)

    return {"labeled_dir": labeled_dir, "labeled_files": labeled_paths}
