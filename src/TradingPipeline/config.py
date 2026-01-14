from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

DATA_CSV = REPO_ROOT / "data" / "processed" / "features_1h_ALL-2025_merged_prev_indicators_labeled.csv"

ARTIFACTS_DIR = REPO_ROOT / "pipeline" / "artifacts" / "TradingPipeline"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TABLES_DIR = ARTIFACTS_DIR / "tables"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

REPORT_PATH = REPORTS_DIR / "TradingPipeline_Report.md"

TIMESTAMP_CANDIDATES = ("ts_utc", "timestamp", "time", "open_time", "datetime")
LABEL_COL = "candle_type"
AMBIGUOUS_COL = "ambiguous_flag"
ALLOWED_LABELS = ("long", "short", "skip")
OHLCV_COLS = ("open", "high", "low", "close", "volume")

# Feature timing: "open" uses information strictly before the 1h bar opens.
# "close" uses the hour ending at t and shifts labels forward by one bar.
DECISION_TIME = "open"

# 5m microstructure source.
FIVE_MIN_CSV = REPO_ROOT / "data" / "processed" / "BTCUSDT_5m_2026-01-03.csv"

# Microstructure configuration (no-leakage, time-aligned).
MICROSTRUCTURE_ENABLED = True
LOOKBACK_HOURS = 1
MIN_5M_BARS = 8
EARLY_LATE_K = 3
RET_5M_Q = 0.95

# Splits (chronological).
HOLDOUT_TRAIN_FRAC = 0.8
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
WALK_FORWARD_SPLITS = 5

# Risk filter windows.
ROLLING_WINDOW = 30 * 24  # 30 days of hourly bars.
ATR_WINDOW = 14

# Risk filter percentiles (train-only thresholds).
VOLUME_RATIO_Q = 0.2
RANGE_Z_LO = 0.05
RANGE_Z_HI = 0.95
ATR_PCT_Q = 0.95

ENABLE_VOLUME_FILTER = True
ENABLE_RANGE_FILTER = True
ENABLE_ATR_FILTER = True

# Threshold sweep.
T_TRADE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
T_LONG_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]
T_SHORT_GRID = [0.6, 0.7, 0.8, 0.9, 0.95]

MIN_COVERAGE = 0.01
MIN_TRADES = 200

# Backtest settings.
BAR_MINUTES = 60
HORIZON_MINUTES = 60
TP_POINTS = 400
SL_POINTS = 200
FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0002
