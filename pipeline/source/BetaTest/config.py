from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

LABELED_CSV_PATH = Path("pipeline/source/labeling/output_1h_labels_baseline_no5m.csv")
UNLABELED_CSV_PATH = Path("data/processed/features_1h_ALL-2025_merged_prev_indicators.csv")
FIVE_MIN_CSV_PATH = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\BTCUSDT_5m_2026-01-03.csv"
)

TIME_COL_CANDIDATES = [
    "nts-utc",
    "timestamp",
    "ts_utc",
    "time",
    "datetime",
    "open_time",
]

LABEL_COL = "candle_type"
AMBIG_COL = "ambiguous_flag"

HOLDOUT_TEST_FRAC = 0.2
CV_SPLITS = 5
DROP_MISSING_ABOVE = 0.30
MIN_COVERAGE = 0.01
MIN_COVERAGE_TRADE = 0.01
MIN_COVERAGE_TOTAL = 0.01
SEED = 42

THRESH_GRID = [round(x, 2) for x in [i / 20 for i in range(1, 20)]]
GATE_THRESH_GRID = list(THRESH_GRID)
DIR_THRESH_GRID = list(THRESH_GRID)
MIN_DIR_SAMPLES = 50
DIR_LONG_HIGH = 0.70
DIR_SHORT_LOW = 0.30
DIR_LONG_HIGH_LIST = list(DIR_THRESH_GRID)
DIR_SHORT_LOW_LIST = list(DIR_THRESH_GRID)

USE_LABEL_CLEANING = True
FORCE_AMBIGUOUS_TO_SKIP = True
MIN_RANGE_FILTER = True
MIN_RANGE_PCT_LIST = [0.0, 0.005, 0.01, 0.015, 0.02]


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path = Path("pipeline/artifacts/BetaTest")
    reports_dir: Path = base_dir / "reports"
    figures_dir: Path = base_dir / "figures"
    tables_dir: Path = base_dir / "tables"
    predictions_dir: Path = base_dir / "predictions"


@dataclass(frozen=True)
class BetaTestConfig:
    labeled_csv_path: Path = LABELED_CSV_PATH
    time_col_candidates: List[str] = None
    label_col: str = LABEL_COL
    ambig_col: str = AMBIG_COL
    holdout_test_frac: float = HOLDOUT_TEST_FRAC
    cv_splits: int = CV_SPLITS
    drop_missing_above: float = DROP_MISSING_ABOVE
    min_coverage: float = MIN_COVERAGE
    min_coverage_trade: float = MIN_COVERAGE_TRADE
    min_coverage_total: float = MIN_COVERAGE_TOTAL
    thresh_grid: List[float] = None
    gate_thresh_grid: List[float] = None
    dir_thresh_grid: List[float] = None
    min_dir_samples: int = MIN_DIR_SAMPLES
    dir_long_high: float = DIR_LONG_HIGH
    dir_short_low: float = DIR_SHORT_LOW
    dir_long_high_list: List[float] = None
    dir_short_low_list: List[float] = None
    seed: int = SEED
    use_label_cleaning: bool = USE_LABEL_CLEANING
    force_ambiguous_to_skip: bool = FORCE_AMBIGUOUS_TO_SKIP
    min_range_filter: bool = MIN_RANGE_FILTER
    min_range_pct_list: List[float] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "time_col_candidates",
            self.time_col_candidates or list(TIME_COL_CANDIDATES),
        )
        object.__setattr__(self, "thresh_grid", self.thresh_grid or list(THRESH_GRID))
        object.__setattr__(
            self, "gate_thresh_grid", self.gate_thresh_grid or list(GATE_THRESH_GRID)
        )
        object.__setattr__(
            self, "dir_thresh_grid", self.dir_thresh_grid or list(DIR_THRESH_GRID)
        )
        object.__setattr__(
            self, "dir_long_high_list", self.dir_long_high_list or list(DIR_LONG_HIGH_LIST)
        )
        object.__setattr__(
            self, "dir_short_low_list", self.dir_short_low_list or list(DIR_SHORT_LOW_LIST)
        )
        object.__setattr__(
            self, "min_range_pct_list", self.min_range_pct_list or list(MIN_RANGE_PCT_LIST)
        )
