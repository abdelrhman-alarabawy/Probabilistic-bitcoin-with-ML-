from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]

DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "report.md"

GATE_FEATURES_PATH = REPO_ROOT / "pipeline" / "gate_module_12h_v1" / "outputs" / "features_used.json"

RANDOM_SEED = 42
REACH_HORIZON = 3  # Supported values: 2, 3, 4, 6
FEATURE_SHIFT = 1
MISSINGNESS_MAX = 0.25
CORR_THRESHOLD = 0.98

QUANTILE_PAIRS: List[Tuple[float, float]] = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
WIDTH_THRESHOLDS = [0.005, 0.01, 0.02]
SUMMARY_QUANTILE_PAIR = (0.2, 0.8)
MIN_COVERAGE_TIGHT = 0.05

USE_GATE = True
GATE_TOPK = 20

TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 3
MIN_TRAIN_ROWS = 600
MIN_TEST_ROWS = 250

GATE_C = 10
GATE_SOLVER = "saga"
GATE_MAX_ITER = 10000
GATE_N_JOBS = -1
GATE_CALIBRATION_METHOD = "isotonic"
GATE_CALIBRATION_SPLITS = 3


@dataclass(frozen=True)
class WindowConfig:
    name: str
    train_months: int
    test_months: int
    step_months: int


WINDOW_CONFIGS: List[WindowConfig] = [
    WindowConfig(
        name="train18_test6_step3",
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )
]
