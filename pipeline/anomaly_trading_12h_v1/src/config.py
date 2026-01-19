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

TIMESTAMP_COL = "timestamp"

FEATURE_SHIFT = 1
LOOKBACK_VOL = 14
LOOKBACK_MISC = 50

INCLUDE_LIQ = True
ADD_MISSING_FLAGS = True

MOM_NEAR_HIGH = 0.8
MOM_NEAR_LOW = 0.2
MOM_RANGE_LOOKBACK = 50
MOM_RANGE_MULT = 1.0

HOLD_BARS = 3
TAKE_PROFIT_PCT = 0.01
STOP_LOSS_PCT = 0.01

IFOREST_N_ESTIMATORS = 200
IFOREST_MAX_SAMPLES = "auto"
IFOREST_CONTAMINATION = "auto"
IFOREST_N_JOBS = -1
IFOREST_RANDOM_STATE = 42

ANOMALY_PERCENTILES = [98, 99]

TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 3
MIN_TRAIN_ROWS = 600
MIN_TEST_ROWS = 250

SIGNAL_PLOT_PERCENTILE = 99


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
