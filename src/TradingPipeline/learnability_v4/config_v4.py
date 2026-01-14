from __future__ import annotations

from pathlib import Path

from ..config import (
    AMBIGUOUS_COL,
    DATA_CSV,
    DECISION_TIME,
    FIVE_MIN_CSV,
    LABEL_COL,
    REPO_ROOT,
    SEED,
)


ARTIFACTS_DIR = REPO_ROOT / "pipeline" / "artifacts" / "TradingPipeline"
TABLES_DIR = ARTIFACTS_DIR / "tables"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORT_PATH = ARTIFACTS_DIR / "TradingPipeline_Report_v4.md"

DATASET_PATH = DATA_CSV
FIVE_MIN_PATH = FIVE_MIN_CSV

TIMESTAMP_COL = "timestamp"
LABEL_COLUMN = LABEL_COL
AMBIGUOUS_COLUMN = AMBIGUOUS_COL

# Chronological split fractions.
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Microstructure quality filters.
WICKINESS_PCTL = 0.80
CHOP_PCTL = 0.80
MISSING_FRAC_MAX = 0.20

# Threshold search grids.
T_TRADE_GRID = [round(x, 2) for x in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]
T_LONG_GRID = [round(x, 2) for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]]
T_SHORT_GRID = [round(x, 2) for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]]

TARGET_TRADE_PREC = 0.55
TARGET_LONG_PREC = 0.35
TARGET_SHORT_PREC = 0.35
MIN_LONG_COUNT = 30
MIN_SHORT_COUNT = 30
MIN_COVERAGE = 0.01

RELAX_PREC_FLOOR = 0.20
RELAX_COUNT_FLOOR = 10
RELAX_COVERAGE_FLOOR = 0.005

MIN_CALIBRATION_SAMPLES = 1000

# Regime-specific models.
REGIME_COL_CANDIDATES = ("regime", "market_regime", "regime_label")
ENABLE_REGIME_MODELS = True
MIN_REGIME_TRAIN = 500
MIN_REGIME_VAL = 200

# Model parameters.
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 400,
    "random_state": SEED,
}
