from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .config import (
    ALLOWED_LABELS,
    END_DATE,
    FIVE_MIN_CSV,
    OUTPUT_1H_BASELINE,
    OUTPUT_1H_WITH5M,
    RUN_MODES,
    START_DATE,
    TIMEFRAME_CONFIG,
    TIMEFRAMES_TO_RUN,
    TRANSITION_OUTPUT,
)


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


def load_primary_df(input_csv: Path) -> Tuple[pd.DataFrame, str]:
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
    df_primary["date"] = df_primary["timestamp"].dt.date
    return df_primary, timestamp_col


def filter_by_date(
    df_primary: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    effective_start = start_date.date() if start_date is not None else df_primary["date"].min()
    effective_end = end_date.date() if end_date is not None else df_primary["date"].max()
    return df_primary[
        (df_primary["date"] >= effective_start) & (df_primary["date"] <= effective_end)
    ]


def run_mode(
    df_primary_filtered: pd.DataFrame,
    horizon_delta: pd.Timedelta,
    tp_points: float,
    sl_points: float,
    five_min_index_for_run: Optional[FiveMinIndex],
    output_path: Path,
    timestamp_col: str,
) -> Tuple[Sequence[str], Sequence[bool], pd.DataFrame]:
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
    output_df["ambiguous_flag"] = ambiguous_flags
    if timestamp_col == "ts_utc":
        output_df = output_df.rename(columns={"timestamp": "ts_utc"})
    output_df.to_csv(output_path, index=False)
    return labels, ambiguous_flags, output_df


def run_labeling_pipeline(
    timeframes: Iterable[str] = TIMEFRAMES_TO_RUN,
    timeframe_config: Optional[dict] = None,
    five_min_csv: Optional[Path] = FIVE_MIN_CSV,
    run_modes: Sequence[str] = RUN_MODES,
    start_date: Optional[pd.Timestamp] = START_DATE,
    end_date: Optional[pd.Timestamp] = END_DATE,
    output_baseline: Path = OUTPUT_1H_BASELINE,
    output_with5m: Path = OUTPUT_1H_WITH5M,
    transition_output: Path = TRANSITION_OUTPUT,
) -> None:
    config = timeframe_config or TIMEFRAME_CONFIG

    for timeframe in timeframes:
        if timeframe not in config:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Expected one of {list(config)}")

        cfg = config[timeframe]
        input_csv = cfg["input_csv"]
        horizon_minutes = cfg["horizon_minutes"]
        tp_points = cfg["tp_points"]
        sl_points = cfg["sl_points"]

        if timeframe != "1h":
            raise ValueError("This script execution is configured to run 1h only.")
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        df_primary, timestamp_col = load_primary_df(input_csv)
        df_primary_filtered = filter_by_date(df_primary, start_date, end_date)
        horizon_delta = pd.Timedelta(minutes=horizon_minutes)

        five_min_index = None
        if "with5m" in run_modes:
            df_5 = load_five_min(five_min_csv)
            if df_5 is not None:
                five_min_index = FiveMinIndex(df_5)

        labels_baseline, ambiguous_baseline, _ = run_mode(
            df_primary_filtered,
            horizon_delta,
            tp_points,
            sl_points,
            None,
            output_baseline,
            timestamp_col,
        )
        labels_with5m, ambiguous_with5m, _ = run_mode(
            df_primary_filtered,
            horizon_delta,
            tp_points,
            sl_points,
            five_min_index,
            output_with5m,
            timestamp_col,
        )

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
            print(
                f"\nWARNING: {len(non_ambiguous_changes)} label changes occurred outside ambiguous cases."
            )

        print(
            f"\nTimeframe {timeframe}: Done. Outputs: {output_baseline} and {output_with5m}"
        )
