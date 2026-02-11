from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


# === CONFIGURATION ===
# Which timeframes to run in this execution. Default to ["12h"] for local run.
TIMEFRAMES_TO_RUN: Iterable[str] = ["12h"]

# Forward horizon (minutes) and TP/SL distances (in price points) per timeframe.
TIMEFRAME_CONFIG = {
    "4h": {
        "input_csv": Path(r"D:\GitHub\Technical-Analysis-BTC\data\hourly\4h\BTCUSDT_4h_2025-08-31 (1).csv"),
        "output_baseline": Path("pipeline/source/labeling/output_4h_labels_baseline_no5m.csv"),
        "output_with5m": Path("pipeline/source/labeling/output_4h_labels_with5m.csv"),
        "transition_output": Path("pipeline/source/labeling/label_transition_4h_no5m_vs_with5m.csv"),
        "horizon_minutes": 240,
        "tp_points": 1200,
        "sl_points": 650,
    },
    "1h": {
        "input_csv": Path("data/processed/features_1h_ALL-2025_merged_prev_indicators.csv"),
        "output_baseline": Path("pipeline/source/labeling/output_1h_labels_baseline_no5m.csv"),
        "output_with5m": Path("pipeline/source/labeling/output_1h_labels_with5m.csv"),
        "transition_output": Path("pipeline/source/labeling/label_transition_1h_no5m_vs_with5m.csv"),
        "horizon_minutes": 60,
        "tp_points": 400,
        "sl_points": 200,
    },
    "12h": {
        "input_csv": Path("data_12h_indicators.csv"),
        "output_baseline": Path("pipeline/source/labeling/output_12h_labels_baseline_no5m.csv"),
        "output_with5m": Path("pipeline/source/labeling/output_12h_labels_with5m.csv"),
        "transition_output": Path("pipeline/source/labeling/label_transition_12h_no5m_vs_with5m.csv"),
        "horizon_minutes": 720,
        "tp_points": 2000,
        "sl_points": 1000,
    },
    "1d": {
        "input_csv": Path("data/raw/BTCUSDT_1d_test.csv"),
        "output_baseline": Path("pipeline/source/labeling/output_1d_labels_baseline_no5m.csv"),
        "output_with5m": Path("pipeline/source/labeling/output_1d_labels_with5m.csv"),
        "transition_output": Path("pipeline/source/labeling/label_transition_1d_no5m_vs_with5m.csv"),
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

# Date range filter (UTC dates). If None, will use full range of each file.
START_DATE: Optional[pd.Timestamp] = None
END_DATE: Optional[pd.Timestamp] = None

ALLOWED_LABELS = ("long", "short", "skip")


def detect_timestamp_column(columns: Sequence[str]) -> str:
    if "timestamp" in columns:
        return "timestamp"
    for candidate in ("ts_utc", "time", "open_time", "datetime"):
        if candidate in columns:
            return candidate
    for col in columns:
        if "timestamp" in col.lower():
            return col
    raise ValueError("No timestamp column found in 5-minute CSV.")


def load_five_min(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"5-minute CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ts_col = detect_timestamp_column(df.columns)
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["high", "low"]:
        if col not in df.columns:
            raise ValueError(f"Missing required OHLCV column '{col}' in {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class FiveMinIndex:
    def __init__(self, df: pd.DataFrame) -> None:
        self.index = pd.DatetimeIndex(df["timestamp"])
        self.high = df["high"].to_numpy()
        self.low = df["low"].to_numpy()

    def slice_high_low(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
        left = self.index.searchsorted(start, side="left")
        right = self.index.searchsorted(end, side="left")
        return self.high[left:right], self.low[left:right]


def label_candles(
    df_primary_filtered: pd.DataFrame,
    horizon_delta: pd.Timedelta,
    tp_points: float,
    sl_points: float,
    five_min_index: Optional[FiveMinIndex],
) -> Tuple[Sequence[str], Sequence[bool]]:
    labels: list[str] = []
    ambiguous_flags: list[bool] = []

    for _, row in df_primary_filtered.iterrows():
        ts = row["timestamp"]
        o = row["open"]
        h = row["high"]
        l = row["low"]
        multiplier = o / 100000
        Long_TP = o + tp_points * multiplier
        Long_SL = o - sl_points * multiplier

        Short_TP = o - tp_points * multiplier
        Short_SL = o + sl_points * multiplier

        def get_slice() -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
            if five_min_index is None:
                return None, None
            return five_min_index.slice_high_low(ts, ts + horizon_delta)

        # Case 1: Long candle (clean long)
        if h >= Long_TP and l >= Long_SL:
            labels.append("long")
            ambiguous_flags.append(False)

        # Case 2: Short candle (clean short)
        elif l <= Short_TP and h <= Short_SL:
            labels.append("short")
            ambiguous_flags.append(False)

        # Case 3: Both thresholds touched -- check which one came first using 5-min candles
        elif h >= Long_TP and l <= Short_TP:
            signal_type = "skip"
            slice_high, slice_low = get_slice()
            if slice_high is not None:
                for high_5, low_5 in zip(slice_high, slice_low):
                    if high_5 >= Long_TP:
                        signal_type = "long"
                        break
                    if low_5 <= Short_TP:
                        signal_type = "short"
                        break
            labels.append(signal_type)
            ambiguous_flags.append(True)

        # Case 4.1: High hit, low partially hit (TP hit first, SL not yet)
        elif h >= Long_TP:
            if l < Long_SL:
                signal_type = "skip"  # Default is skip
                slice_high, slice_low = get_slice()
                if slice_high is not None:
                    for high_5, low_5 in zip(slice_high, slice_low):
                        if high_5 >= Long_TP:  # TP hit first
                            signal_type = "long"  # Long if TP reached before SL
                            break
                        if low_5 <= Long_SL:  # SL hit before TP
                            signal_type = "skip"
                            break
                labels.append(signal_type)
                ambiguous_flags.append(True)
            else:
                labels.append("long")
                ambiguous_flags.append(False)

        # Case 4.2: Low hit first (SL hit first, TP not yet)
        elif l <= Short_TP:
            if h > Short_SL:
                signal_type = "skip"  # Default is skip
                slice_high, slice_low = get_slice()
                if slice_high is not None:
                    for high_5, low_5 in zip(slice_high, slice_low):
                        if low_5 <= Short_TP:  # SL hit first
                            signal_type = "short"  # Short if SL reached before TP
                            break
                        if high_5 >= Short_SL:  # TP hit before SL
                            signal_type = "skip"
                            break
                labels.append(signal_type)
                ambiguous_flags.append(True)
            else:
                labels.append("short")
                ambiguous_flags.append(False)
        else:
            labels.append("skip")
            ambiguous_flags.append(False)

    return labels, ambiguous_flags


def print_counts(title: str, labels: Sequence[str]) -> Counter:
    counts = Counter(labels)
    total = len(labels)
    print(f"\n=== {title} ===")
    for candle_type in ALLOWED_LABELS:
        count = counts.get(candle_type, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{candle_type}:  {count} ({pct:.2f}%)")
    return counts


for timeframe in TIMEFRAMES_TO_RUN:
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Expected one of {list(TIMEFRAME_CONFIG)}")

    cfg = TIMEFRAME_CONFIG[timeframe]
    input_csv = cfg["input_csv"]
    horizon_minutes = cfg["horizon_minutes"]
    tp_points = cfg["tp_points"]
    sl_points = cfg["sl_points"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # === LOAD DATA ===
    df_primary = pd.read_csv(input_csv)
    df_primary.columns = df_primary.columns.str.strip()
    timestamp_col = "timestamp"
    if "timestamp" not in df_primary.columns and "ts_utc" in df_primary.columns:
        df_primary = df_primary.rename(columns={"ts_utc": "timestamp"})
        timestamp_col = "ts_utc"

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df_primary.columns:
            raise ValueError(f"Missing required OHLCV column '{col}' in {input_csv}")
        df_primary[col] = pd.to_numeric(df_primary[col], errors="coerce")

    df_primary["timestamp"] = pd.to_datetime(df_primary["timestamp"], utc=True)

    # Parse timestamp to datetime with UTC and add simple date column for filtering.
    df_primary["date"] = df_primary["timestamp"].dt.date

    # === FILTER DATE RANGE ONLY (full-file by default) ===
    effective_start = START_DATE.date() if START_DATE is not None else df_primary["date"].min()
    effective_end = END_DATE.date() if END_DATE is not None else df_primary["date"].max()
    df_primary_filtered = df_primary[
        (df_primary["date"] >= effective_start) & (df_primary["date"] <= effective_end)
    ]

    horizon_delta = pd.Timedelta(minutes=horizon_minutes)

    five_min_index = None
    if "with5m" in RUN_MODES:
        df_5 = load_five_min(FIVE_MIN_CSV)
        if df_5 is not None:
            five_min_index = FiveMinIndex(df_5)

    def run_mode(mode_name: str, five_min_index_for_run: Optional[FiveMinIndex], output_path: Path):
        labels, ambiguous_flags = label_candles(
            df_primary_filtered,
            horizon_delta,
            tp_points,
            sl_points,
            five_min_index_for_run,
        )
        if len(labels) != len(df_primary_filtered):
            raise ValueError("Label count does not match number of 1h candles.")
        if not set(labels).issubset(ALLOWED_LABELS):
            raise ValueError(f"Unexpected labels detected: {set(labels) - set(ALLOWED_LABELS)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = df_primary_filtered.copy()
        output_df["candle_type"] = labels
        if timestamp_col == "ts_utc":
            output_df = output_df.rename(columns={"timestamp": "ts_utc"})
        output_df.to_csv(output_path, index=False)
        return labels, ambiguous_flags

    output_baseline = cfg["output_baseline"]
    output_with5m = cfg["output_with5m"]
    transition_output = cfg["transition_output"]

    labels_baseline, ambiguous_baseline = run_mode("no5m", None, output_baseline)
    labels_with5m, ambiguous_with5m = run_mode("with5m", five_min_index, output_with5m)

    if len(labels_baseline) != len(labels_with5m):
        raise ValueError("Baseline and with-5m outputs have different row counts.")

    counts_baseline = print_counts("Baseline (NO 5m)", labels_baseline)
    counts_with5m = print_counts("With 5m tie-break", labels_with5m)

    print("\n=== Delta (with5m - baseline) ===")
    for candle_type in ALLOWED_LABELS:
        delta = counts_with5m.get(candle_type, 0) - counts_baseline.get(candle_type, 0)
        print(f"{candle_type}:  {delta:+d}")

    transition_df = pd.crosstab(
        pd.Series(labels_baseline, name="from"),
        pd.Series(labels_with5m, name="to"),
        dropna=False,
    ).reindex(index=ALLOWED_LABELS, columns=ALLOWED_LABELS, fill_value=0)
    transition_df.index.name = "from\\to"

    transition_output.parent.mkdir(parents=True, exist_ok=True)
    transition_df.to_csv(transition_output)

    print("\n=== Label transition matrix ===")
    print("from\\to, long, short, skip")
    for row_label in ALLOWED_LABELS:
        row = transition_df.loc[row_label]
        print(f"{row_label},   {row['long']},    {row['short']},     {row['skip']}")

    changed_indices = [
        i for i, (base_label, new_label) in enumerate(zip(labels_baseline, labels_with5m))
        if base_label != new_label
    ]
    non_ambiguous_changes = [
        i for i in changed_indices
        if not (ambiguous_baseline[i] or ambiguous_with5m[i])
    ]
    if non_ambiguous_changes:
        print(f"\nWARNING: {len(non_ambiguous_changes)} label changes occurred outside ambiguous cases.")

    print(
        f"\nTimeframe {timeframe}: Done. Outputs: {output_baseline} and {output_with5m}"
    )
