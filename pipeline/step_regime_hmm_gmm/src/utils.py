from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def json_loads(value: str, default: Any = None) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return default
