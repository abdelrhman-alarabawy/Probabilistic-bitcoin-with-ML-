from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]

DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"

OUTPUTS_DIR = PIPELINE_DIR / "outputs"
FIGURES_DIR = PIPELINE_DIR / "figures"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
REPORT_PATH = PIPELINE_DIR / "report.md"
FINAL_SIGNALS_DIR = OUTPUTS_DIR / "final_signals_by_window"

RANDOM_SEED = 42
FEATURE_SHIFT = 1

CLIP_QUANTILES = (0.01, 0.99)
USE_PCA = False
PCA_N_COMPONENTS = 0.95

GMM_KS = [2, 3, 4, 5, 6]
GMM_SEEDS = [0, 1, 2]
GMM_MAX_ITER = 200
GMM_COV_TYPE = "full"

MIN_ACTION_RATE = 0.05
MIN_DURATION = 3
MAX_LEAVE_PROB = 0.55

MIN_DIR_TRAIN_SAMPLES = 200

GATE_C_GRID = [1, 10, 50]
GATE_CALIBRATION_METHOD = "isotonic"
DIR_CALIBRATION_METHOD = "sigmoid"

GATE_QUANTILES = [80, 85, 90, 93, 95, 97]
TOPK_PERCENTS = [1, 2, 5, 10]
DIRECTION_THRESHOLDS = [0.52, 0.55, 0.58, 0.60]
ENTROPY_MAX_GRID = [float("inf"), 0.8, 0.7, 0.6]

DEFAULT_POLICY_TYPE = "topk"
DEFAULT_TOPK_PERCENT = 2
DEFAULT_DIRECTION_THRESHOLD = 0.55
DEFAULT_ENTROPY_MAX = 0.7
DEFAULT_GATE_QUANTILE = 90

QUALITY_GATE_AP_MIN = 0.40
QUALITY_DIR_PRECISION_MIN = 0.55
QUALITY_GATE_PRECISION_MIN = 0.70
QUALITY_MIN_TRADES = 5
QUALITY_COVERAGE_MAX = 0.05

GLOBAL_COVERAGE_TARGET = 0.05

WINDOW_TRAIN_MONTHS = 12
WINDOW_TEST_MONTHS = 3
STEP_MONTHS = WINDOW_TEST_MONTHS
MIN_TRAIN_ROWS = 400
MIN_TEST_ROWS = 100

WINDOW_TRAIN_OPTIONS = [6, 12, 18]
WINDOW_TEST_OPTIONS = [1, 3, 6]


@dataclass(frozen=True)
class WindowConfig:
    name: str
    train_months: int
    test_months: int
    step_months: int


WINDOW_CONFIGS: List[WindowConfig] = [
    WindowConfig(name="train6_test3", train_months=6, test_months=3, step_months=3),
    WindowConfig(name="train12_test3", train_months=12, test_months=3, step_months=3),
    WindowConfig(name="train18_test3", train_months=18, test_months=3, step_months=3),
]
