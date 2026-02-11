from __future__ import annotations

import json
import logging
import hashlib
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER_NAME = "unsup_gmm_v1"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_jsonable)


def validate_no_nans(x: np.ndarray, context: str) -> None:
    if np.isnan(x).any():
        raise ValueError(f"NaNs detected in {context}.")


def validate_responsibilities(resp: np.ndarray, tol: float = 1e-3) -> None:
    row_sums = resp.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        max_diff = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"Responsibilities do not sum to 1. Max diff={max_diff:.6f}")


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def format_model_id(
    fold_id: int,
    featureset: str,
    k: int,
    covariance_type: str,
    reg_covar: float,
) -> str:
    reg_str = f"{reg_covar:.0e}".replace("+", "")
    return (
        f"fold_{fold_id}__feat_{featureset}__k_{k}__cov_{covariance_type}__reg_{reg_str}"
    )


def parse_model_id(model_id: str) -> dict:
    parts = model_id.split("__")
    parsed = {}
    for part in parts:
        if part.startswith("fold_"):
            parsed["fold_id"] = int(part.replace("fold_", ""))
        elif part.startswith("feat_"):
            parsed["featureset"] = part.replace("feat_", "")
        elif part.startswith("k_"):
            parsed["K"] = int(part.replace("k_", ""))
        elif part.startswith("cov_"):
            parsed["covariance_type"] = part.replace("cov_", "")
        elif part.startswith("reg_"):
            parsed["reg_covar"] = part.replace("reg_", "")
    return parsed


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(np.mean(values))


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=0))


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    hashed = int(digest[:12], 16) % (2**31 - 1)
    return (base_seed + hashed) % (2**31 - 1)
