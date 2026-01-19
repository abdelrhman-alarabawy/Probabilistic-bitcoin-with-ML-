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

ANOMALY_PERCENTILES = [98, 99]
FOCUS_PERCENTILE = 99

HOLD_BARS_GRID = [1, 2, 3, 4, 6]
TP_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015]
SL_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015]

RANGE_THRESHOLD_GRID = [0.003, 0.005, 0.008]

FEE_PER_TRADE = 0.0005

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
