from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be .yaml, .yml, or .json")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    base_dir = path.parent

    data.setdefault("data", {})
    data.setdefault("featuresets", [])
    data.setdefault("splits", {})
    data.setdefault("gmm_sweep", {})
    data.setdefault("selection", {})
    data.setdefault("output", {})
    data.setdefault("labeling", {})

    data["data"].setdefault("timestamp_col", "timestamp")
    data["data"].setdefault("ohlcv_cols", ["open", "high", "low", "close", "volume"])
    data["data"].setdefault("label_col", "candle_type")

    if "csv_path" in data["data"]:
        data["data"]["csv_path"] = _resolve_path(data["data"]["csv_path"], base_dir)

    data["output"].setdefault("root_dir", "pipeline/step_unsup_gmm_v1/results")
    data["output"]["root_dir"] = _resolve_path(data["output"]["root_dir"], base_dir)

    if data["labeling"].get("five_min_csv"):
        data["labeling"]["five_min_csv"] = _resolve_path(
            data["labeling"]["five_min_csv"], base_dir
        )

    data["labeling"].setdefault("horizon_minutes", 720)
    data["labeling"].setdefault("tp_points", 2000)
    data["labeling"].setdefault("sl_points", 1000)

    return data
