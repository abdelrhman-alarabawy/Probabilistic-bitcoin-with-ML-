from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]

DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"

OUTPUTS_DIR = PIPELINE_DIR / "outputs"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "report.md"

RANDOM_SEED = 42
FEATURE_SHIFT = 1

USE_CLIPPER = True
CLIP_QUANTILES = (0.01, 0.99)

GMM_KS = [2, 3, 4, 5, 6]
GMM_SEEDS = [0, 1, 2]
GMM_MAX_ITER = 200
GMM_COV_TYPE = "full"

MIN_ACTION_RATE = 0.05
MIN_DURATION = 3
MAX_LEAVE_PROB = 0.55

GATE_C_GRID = [1, 10, 50]
GATE_SOLVER = "saga"
GATE_MAX_ITER = 10000
GATE_CALIBRATION_METHOD = "isotonic"
GATE_N_SPLITS = 3

ENTROPY_MAX_DEFAULT = 0.7

K_LIST = [5, 10, 20]
RANDOM_BASELINE_REPS = 200

WINDOW_TRAIN_MONTHS = 18
WINDOW_TEST_MONTHS = 6
STEP_MONTHS = 3
MIN_TRAIN_ROWS = 600
MIN_TEST_ROWS = 250

CORR_THRESHOLD = 0.98
MISSINGNESS_MAX = 0.25


@dataclass(frozen=True)
class WindowConfig:
    name: str
    train_months: int
    test_months: int
    step_months: int


WINDOW_CONFIGS: List[WindowConfig] = [
    WindowConfig(name="train18_test6_step3", train_months=18, test_months=6, step_months=3)
]
