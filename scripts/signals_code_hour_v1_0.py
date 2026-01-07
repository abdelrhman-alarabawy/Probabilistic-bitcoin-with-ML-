import pandas as pd
from pathlib import Path
from typing import Iterable, Optional


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
FIVE_MIN_CSV: Optional[Path] = None

# Date range filter (UTC dates). If None, will use full range of each file.
START_DATE: Optional[pd.Timestamp] = None
END_DATE: Optional[pd.Timestamp] = None


def load_five_min(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"5-minute CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


df_5 = load_five_min(FIVE_MIN_CSV)

for timeframe in TIMEFRAMES_TO_RUN:
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Expected one of {list(TIMEFRAME_CONFIG)}")

    cfg = TIMEFRAME_CONFIG[timeframe]
    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
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
    effective_start = (START_DATE.date() if START_DATE is not None else df_primary["date"].min())
    effective_end = (END_DATE.date() if END_DATE is not None else df_primary["date"].max())
    df_primary_filtered = df_primary[(df_primary["date"] >= effective_start) & (df_primary["date"] <= effective_end)]

    horizon_delta = pd.Timedelta(minutes=horizon_minutes)

    # === PROCESS ===
    results = []

    for _, row in df_primary_filtered.iterrows():
        ts = row["timestamp"]
        o = row["open"]
        h = row["high"]
        l = row["low"]
        c = row["close"]
        v = row["volume"]
        multiplier = o / 100000
        Long_TP = o + tp_points * multiplier
        Long_SL = o - sl_points * multiplier

        Short_TP = o - tp_points * multiplier
        Short_SL = o + sl_points * multiplier

        if df_5 is not None:
            df_5_slice = df_5[(df_5["timestamp"] >= ts) & (df_5["timestamp"] < ts + horizon_delta)]
        else:
            df_5_slice = pd.DataFrame(columns=["high", "low"])

        # Case 1: Long candle (clean long)
        if h >= Long_TP and l >= Long_SL:
            results.append((ts, o, h, l, c, v, "long"))

        # Case 2: Short candle (clean short)
        elif l <= Short_TP and h <= Short_SL:
            results.append((ts, o, h, l, c, v, "short"))

        # Case 3: Both thresholds touched -- check which one came first using 5-min candles
        elif h >= Long_TP and l <= Short_TP:
            signal_type = "skip"
            for _, r5 in df_5_slice.iterrows():
                if r5["high"] >= Long_TP:
                    signal_type = "long"
                    break
                elif r5["low"] <= Short_TP:
                    signal_type = "short"
                    break
            results.append((ts, o, h, l, c, v, signal_type))

        # Case 4.1: High hit, low partially hit (TP hit first, SL not yet)
        elif h >= Long_TP:
            if l < Long_SL:
                signal_type = "skip"  # Default is skip
                for _, r5 in df_5_slice.iterrows():
                    if r5["high"] >= Long_TP:  # TP hit first
                        signal_type = "long"  # Long if TP reached before SL
                        break  # Break immediately when TP is hit
                    elif r5["low"] <= Long_SL:  # SL hit before TP
                        signal_type = "skip"  # Skip if SL hits before TP
                        break  # Break immediately when SL is hit
                results.append((ts, o, h, l, c, v, signal_type))
            else:
                results.append((ts, o, h, l, c, v, "long"))

        # Case 4.2: Low hit first (SL hit first, TP not yet)
        elif l <= Short_TP:
            if h > Short_SL:
                signal_type = "skip"  # Default is skip
                for _, r5 in df_5_slice.iterrows():
                    if r5["low"] <= Short_TP:  # SL hit first
                        signal_type = "short"  # Short if SL reached before TP
                        break  # Break immediately when SL is hit
                    elif r5["high"] >= Short_SL:  # TP hit before SL
                        signal_type = "skip"  # Skip if TP hits before SL
                        break  # Break immediately when TP is hit
                results.append((ts, o, h, l, c, v, signal_type))
            else:
                results.append((ts, o, h, l, c, v, "short"))
        else:
            results.append((ts, o, h, l, c, v, "skip"))

    # === OUTPUT TO CSV ===
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df = df_primary_filtered.copy()
    output_df["candle_type"] = [row[-1] for row in results]
    if timestamp_col == "ts_utc":
        output_df = output_df.rename(columns={"timestamp": "ts_utc"})
    output_df.to_csv(output_csv, index=False)

    print(f"Timeframe {timeframe}: Done. Output written to {output_csv}")
    print(f"Timeframe: {timeframe}, horizon: {horizon_minutes} minutes, TP/SL points: {tp_points}/{sl_points}")

    from collections import Counter

    # Count the types
    counts = Counter([row[-1] for row in results])
    total = sum(counts.values())

    # Print header
    print("\n=== Candle Signal Summary ===")

    # Loop over types and print percentage
    for candle_type in ["long", "short", "skip"]:
        count = counts.get(candle_type, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{candle_type.capitalize()} candles: {count} ({pct:.2f}%)")
