from __future__ import annotations

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

DATA_PATH = REPO_ROOT / "data" / "processed" / "12h_features_indicators_with_ohlcv.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
REPORT_PATH = BASE_DIR / "regime_pipeline_12h_report.md"
README_PATH = BASE_DIR / "README.md"

# Reproducibility
RANDOM_SEED = 42

# Time horizon (in bars) for forward returns and evaluation
HORIZON_BARS = 2

# Features: shift by 1 to avoid leakage
FEATURE_SHIFT = 1

# Walk-forward folds
N_SPLITS = 4
MIN_TRAIN_SIZE = 0.5  # fraction if < 1.0, otherwise absolute count
TEST_SIZE = 0.1  # fraction if < 1.0, otherwise absolute count
FOLD_STEP_SIZE = None  # defaults to TEST_SIZE when None

# Preprocessing
CLIP_QUANTILES = (0.01, 0.99)
USE_PCA = False
PCA_N_COMPONENTS = 0.95

# HMM settings
HMM_KS = [2, 3, 4, 5, 6, 8]
HMM_SEEDS = [0, 1, 2]
HMM_MAX_ITER = 200
HMM_COV_TYPE = "full"
HMM_VAL_RATIO = 0.2
HMM_MIN_STATE_FRAC = 0.05
HMM_MIN_AVG_DURATION = 3
HMM_ENTROPY_THRESHOLD = 1.0
AUTO_INSTALL_HMMLEARN = False

# GMM settings
GMM_KS = [2, 3, 4, 5, 6, 8]
GMM_SEEDS = [0, 1, 2]
GMM_MAX_ITER = 200
GMM_COV_TYPE = "full"
GMM_ENTROPY_THRESHOLD = 1.0

# Eligibility mapping thresholds
ELIGIBILITY_D_MIN = 3
ELIGIBILITY_TAIL_LOSS_MAX = 0.05
ELIGIBILITY_WIN_RATE_MIN = 0.55
ELIGIBILITY_TRANSITION_RISK_MAX = 0.6
ELIGIBILITY_MIN_REGIME_FRAC = 0.05
TRANSITION_RISK_WINDOW = 3

# Eligibility classifier
USE_ELIGIBILITY_CLASSIFIER = True
ELIGIBILITY_PROB_THRESHOLD = 0.8

# Direction model thresholds
RETURN_THRESHOLD = 0.002
DECISION_THRESHOLD = 0.95

# Plotting
MAX_TIMELINE_POINTS = 4000
