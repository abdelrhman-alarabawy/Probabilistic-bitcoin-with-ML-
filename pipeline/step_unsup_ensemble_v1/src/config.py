from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be YAML or JSON.")

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be mapping.")

    base_dir = path.parent

    cfg.setdefault("data", {})
    cfg.setdefault("featuresets", [])
    cfg.setdefault("splits", {})
    cfg.setdefault("labeling", {})
    cfg.setdefault("regime_models", [])
    cfg.setdefault("state_to_label", {})
    cfg.setdefault("ensemble", {})
    cfg.setdefault("decision_policy", {})
    cfg.setdefault("anomaly_gate", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("output", {})

    cfg["data"].setdefault("timestamp_col", "timestamp")
    cfg["data"].setdefault("ohlcv_cols", ["open", "high", "low", "close", "volume"])
    cfg["data"].setdefault("label_col", "candle_type")
    cfg["data"].setdefault("timezone", "UTC")
    cfg["data"]["csv_path"] = _resolve_path(cfg["data"]["csv_path"], base_dir)

    cfg["labeling"].setdefault("merge_on", ["timestamp"])
    cfg["labeling"]["script_path"] = _resolve_path(cfg["labeling"]["script_path"], base_dir)
    if cfg["labeling"].get("five_min_csv"):
        cfg["labeling"]["five_min_csv"] = _resolve_path(cfg["labeling"]["five_min_csv"], base_dir)

    cfg["output"].setdefault("root_dir", "../../../pipeline/step_unsup_ensemble_v1/results")
    cfg["output"]["root_dir"] = _resolve_path(cfg["output"]["root_dir"], base_dir)
    cfg["output"].setdefault("float_format", "%.8f")

    cfg["state_to_label"].setdefault("smoothing_alpha", 1.0)

    cfg["decision_policy"].setdefault("tau_trade", 0.7)
    cfg["decision_policy"].setdefault("tau_margin", 0.15)
    cfg["decision_policy"].setdefault("tau_entropy", None)
    cfg["decision_policy"].setdefault("abstain_label", "skip")

    cfg["anomaly_gate"].setdefault("enabled", False)
    cfg["anomaly_gate"].setdefault("model", "mahalanobis")
    cfg["anomaly_gate"].setdefault("mode", "force_skip")
    cfg["anomaly_gate"].setdefault("raised_tau_trade", 0.8)
    cfg["anomaly_gate"].setdefault("raised_tau_margin", 0.2)
    cfg["anomaly_gate"].setdefault("contamination", 0.01)
    cfg["anomaly_gate"].setdefault("mahalanobis_robust", True)
    cfg["anomaly_gate"].setdefault("anomaly_threshold_quantile", 0.99)

    return cfg
