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

RANDOM_SEED = 42
FEATURE_SHIFT = 1
MISSINGNESS_MAX = 0.25
CORR_THRESHOLD = 0.98

K_DEFAULT = 5
K_SWEEP = [5, 10, 20]
RANDOM_BASELINE_REPS = 200
MAX_MEDIAN_FPR = 0.06

TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 3
MIN_TRAIN_ROWS = 600
MIN_TEST_ROWS = 250

GATE_C = 10
GATE_SOLVER = "saga"
GATE_MAX_ITER = 10000
GATE_N_JOBS = -1
CALIBRATION_METHOD = "isotonic"
CALIBRATION_SPLITS = 3

# Confusion report defaults (trade vs notrade)
CONFUSION_K = 20
CONFUSION_BY_WINDOW_PATH = OUTPUTS_DIR / "gate_confusion_by_window.csv"
CONFUSION_AGGREGATE_PATH = OUTPUTS_DIR / "gate_confusion_aggregate.csv"
CONFUSION_FIGURES_DIR = FIGURES_DIR / "trade_confusion_by_window"
CONFUSION_AGG_FIG_PATH = FIGURES_DIR / "confusion_trade_notrade_aggregate.png"
CONFUSION_SUMMARY_FIG_PATH = FIGURES_DIR / "precision_and_counts_by_window.png"


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
