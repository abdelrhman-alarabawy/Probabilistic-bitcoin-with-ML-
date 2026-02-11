from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER_NAME = "unsup_ensemble_v1"
LABELS = ["long", "short", "skip"]


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    hashed = int(digest[:12], 16) % (2**31 - 1)
    return (base_seed + hashed) % (2**31 - 1)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_jsonable)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def validate_prob_rows(prob: np.ndarray, atol: float = 1e-4) -> None:
    if prob.ndim != 2:
        raise ValueError("Probability array must be 2D.")
    row_sum = prob.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=atol):
        max_diff = float(np.max(np.abs(row_sum - 1.0)))
        raise ValueError(f"Probability rows do not sum to 1. max_diff={max_diff:.6f}")


def resolve_metric(metrics: dict[str, Any], key: str) -> float:
    if key in metrics:
        return float(metrics[key])
    if key.startswith("repeats_"):
        stripped = key.replace("repeats_", "", 1)
        if stripped in metrics:
            return float(metrics[stripped])
    return float("nan")
