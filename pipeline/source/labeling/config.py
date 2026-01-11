from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

SEED = 42

# === CONFIGURATION ===
# Which timeframes to run in this execution. Default to ["1h"] for local run.
TIMEFRAMES_TO_RUN: Iterable[str] = ["1h"]

# Forward horizon (minutes) and TP/SL distances (in price points) per timeframe.
TIMEFRAME_CONFIG = {
    "4h": {
        "input_csv": Path(r"D:\GitHub\Technical-Analysis-BTC\data\hourly\4h\BTCUSDT_4h_2025-08-31 (1).csv"),
        "output_csv": Path("pipeline/source/labeling/output_4h_labels.csv"),
        "horizon_minutes": 240,
        "tp_points": 1200,
        "sl_points": 650,
    },
    "1h": {
        "input_csv": Path("data/processed/features_1h_ALL-2025_merged_prev_indicators.csv"),
        "output_csv": Path("data/processed/features_1h_ALL-2025_merged_prev_indicators_labeled.csv"),
        "horizon_minutes": 60,
        "tp_points": 400,
        "sl_points": 200,
    },
    "1d": {
        "input_csv": Path("data/raw/BTCUSDT_1d_test.csv"),
        "output_csv": Path("pipeline/source/labeling/output_1d_labels.csv"),
        # Use full 24h horizon per updated instructions.
        "horizon_minutes": 1440,
        "tp_points": 2000,
        "sl_points": 1000,
    },
}

# Granular candles used to resolve which side hit first when both TP/SL are touched.
# Set to None to skip tie-break by lower timeframe.
FIVE_MIN_CSV: Optional[Path] = Path(
    r"D:\GitHub\bitcoin-probabilistic-learning\data\processed\BTCUSDT_5m_2026-01-03.csv"
)

# Run both modes in a single execution.
RUN_MODES: Sequence[str] = ("no5m", "with5m")

# Output locations for 1h relabeling runs.
OUTPUT_1H_BASELINE = Path("pipeline/source/labeling/output_1h_labels_baseline_no5m.csv")
OUTPUT_1H_WITH5M = Path("pipeline/source/labeling/output_1h_labels_with5m.csv")
TRANSITION_OUTPUT = Path("pipeline/source/labeling/label_transition_1h_no5m_vs_with5m.csv")

# Date range filter (UTC dates). If None, will use full range of each file.
START_DATE: Optional[pd.Timestamp] = None
END_DATE: Optional[pd.Timestamp] = None

ALLOWED_LABELS = ("long", "short", "skip")

META_COLUMNS = {
    "timestamp",
    "ts_utc",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

LABEL_COLUMNS = {"candle_type", "ambiguous_flag"}


@dataclass(frozen=True)
class LabelCleaningConfig:
    enabled: bool = False
    min_range_pct: Optional[float] = None


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = SEED
    test_size: float = 0.2
    cv_splits: int = 5
    min_trade_coverage: float = 0.05
    threshold_grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)
    model_names: Sequence[str] = ("lightgbm", "xgboost")
    imputer_strategy: str = "median"
    val_size: float = 0.1


@dataclass(frozen=True)
class PathsConfig:
    artifacts_labeling: Path = Path("pipeline/artifacts/labeling")
    artifacts_models: Path = Path("pipeline/artifacts/models")
    artifacts_reports: Path = Path("pipeline/artifacts/reports")