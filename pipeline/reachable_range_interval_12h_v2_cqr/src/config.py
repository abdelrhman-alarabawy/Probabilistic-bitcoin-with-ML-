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

TARGET_COVERAGE = 0.70
ALPHA_LOW = 0.15
ALPHA_HIGH = 0.85
KNN_NEIGHBORS = 50

WEIGHT_GRID_STEP = 0.1
MAX_KNN_WEIGHT = 0.5

CAL_FRACTION_MIN = 0.15
CAL_FRACTION_MAX = 0.25

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

WIDTH_THRESHOLDS = [0.005, 0.01, 0.02]


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


def weight_grid(step: float = WEIGHT_GRID_STEP, max_knn: float = MAX_KNN_WEIGHT) -> List[Tuple[float, float, float, float]]:
    grid = []
    steps = int(round(1 / step))
    for i in range(steps + 1):
        w_lgb = i * step
        for j in range(steps + 1):
            w_xgb = j * step
            for k in range(steps + 1):
                w_cat = k * step
                w_knn = 1.0 - (w_lgb + w_xgb + w_cat)
                if w_knn < -1e-9:
                    continue
                if w_knn > max_knn + 1e-9:
                    continue
                if w_knn < 0:
                    w_knn = 0.0
                total = w_lgb + w_xgb + w_cat + w_knn
                if total <= 0:
                    continue
                grid.append((w_lgb, w_xgb, w_cat, w_knn))
    return grid
