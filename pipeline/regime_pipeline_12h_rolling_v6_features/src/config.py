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
SIGNALS_DIR = OUTPUTS_DIR / "signals_by_window"
CONFUSION_DIR = OUTPUTS_DIR / "confusion_matrices"

RANDOM_SEED = 42
FEATURE_SHIFT = 1

USE_PCA = False
PCA_N_COMPONENTS = 0.95

USE_CLIPPER = True
CLIP_QUANTILES = (0.01, 0.99)

GMM_KS = [2, 3, 4, 5, 6]
GMM_SEEDS = [0, 1, 2]
GMM_MAX_ITER = 200
GMM_COV_TYPE = "full"

MIN_ACTION_RATE = 0.05
MIN_DURATION = 3
MAX_LEAVE_PROB = 0.55

MIN_DIR_TRAIN_SAMPLES = 200

GATE_C_GRID = [1, 10, 50]
GATE_SOLVER = "saga"
GATE_MAX_ITER = 5000
GATE_CALIBRATION_METHOD = "isotonic"
GATE_N_SPLITS = 3

DIR_C_GRID = [1, 10, 50]
DIR_SOLVER = "saga"
DIR_MAX_ITER = 5000
DIR_CALIBRATION_METHOD = "sigmoid"
DIR_N_SPLITS = 3

GATE_QUANTILES = [80, 85, 90, 93, 95, 97]
TOPK_PERCENTS = [1, 2, 5]
DIRECTION_THRESHOLDS = [0.52, 0.55, 0.58, 0.60]
ENTROPY_MAX_GRID = [float("inf"), 0.8, 0.7, 0.6]

DEFAULT_POLICY_TYPE = "topk"
DEFAULT_TOPK_PERCENT = 2
DEFAULT_DIRECTION_THRESHOLD = 0.55
DEFAULT_ENTROPY_MAX = 0.7
DEFAULT_GATE_QUANTILE = 90

MIN_TRADES = 20
QUALITY_GATE_AP_MIN = 0.40
QUALITY_WILSON_MIN = 0.55
QUALITY_COVERAGE_MAX = 0.05

GLOBAL_COVERAGE_TARGET = 0.05

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
    WindowConfig(name="train18_test6_step3", train_months=18, test_months=6, step_months=3),
    WindowConfig(name="train18_test3_step3", train_months=18, test_months=3, step_months=3),
]

ADAPTIVE_LOOKBACK = 2
ADAPTIVE_TOPK_GOOD = 5
ADAPTIVE_TOPK_BAD = 2
ADAPTIVE_DIR_THR_GOOD = 0.55
ADAPTIVE_DIR_THR_BAD = 0.60
