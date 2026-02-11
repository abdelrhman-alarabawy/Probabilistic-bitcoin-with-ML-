from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

ALLOWED_LABELS = ("long", "short", "skip")


@dataclass
class LabelingConfig:
    horizon_minutes: int = 720
    tp_points: float = 2000
    sl_points: float = 1000
    five_min_csv: Optional[Path] = None


def _detect_timestamp_column(columns: Sequence[str]) -> str:
    if "timestamp" in columns:
        return "timestamp"
    for candidate in ("ts_utc", "time", "open_time", "datetime"):
        if candidate in columns:
            return candidate
    for col in columns:
        if "timestamp" in col.lower():
            return col
    raise ValueError("No timestamp column found.")


def _load_five_min(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"5-minute CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    ts_col = _detect_timestamp_column(df.columns)
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


def _label_candles(
    df_primary: pd.DataFrame,
    horizon_delta: pd.Timedelta,
    tp_points: float,
    sl_points: float,
    five_min_index: Optional[FiveMinIndex],
) -> list[str]:
    labels: list[str] = []

    for _, row in df_primary.iterrows():
        ts = row["timestamp"]
        o = row["open"]
        h = row["high"]
        l = row["low"]
        multiplier = o / 100000
        long_tp = o + tp_points * multiplier
        long_sl = o - sl_points * multiplier

        short_tp = o - tp_points * multiplier
        short_sl = o + sl_points * multiplier

        def get_slice() -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
            if five_min_index is None:
                return None, None
            return five_min_index.slice_high_low(ts, ts + horizon_delta)

        if h >= long_tp and l >= long_sl:
            labels.append("long")
        elif l <= short_tp and h <= short_sl:
            labels.append("short")
        elif h >= long_tp and l <= short_tp:
            signal_type = "skip"
            slice_high, slice_low = get_slice()
            if slice_high is not None:
                for high_5, low_5 in zip(slice_high, slice_low):
                    if high_5 >= long_tp:
                        signal_type = "long"
                        break
                    if low_5 <= short_tp:
                        signal_type = "short"
                        break
            labels.append(signal_type)
        elif h >= long_tp:
            if l < long_sl:
                signal_type = "skip"
                slice_high, slice_low = get_slice()
                if slice_high is not None:
                    for high_5, low_5 in zip(slice_high, slice_low):
                        if high_5 >= long_tp:
                            signal_type = "long"
                            break
                        if low_5 <= long_sl:
                            signal_type = "skip"
                            break
                labels.append(signal_type)
            else:
                labels.append("long")
        elif l <= short_tp:
            if h > short_sl:
                signal_type = "skip"
                slice_high, slice_low = get_slice()
                if slice_high is not None:
                    for high_5, low_5 in zip(slice_high, slice_low):
                        if low_5 <= short_tp:
                            signal_type = "short"
                            break
                        if high_5 >= short_sl:
                            signal_type = "skip"
                            break
                labels.append(signal_type)
            else:
                labels.append("short")
        else:
            labels.append("skip")

    return labels


def label_dataframe(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon_minutes: int = 720,
    tp_points: float = 2000,
    sl_points: float = 1000,
    five_min_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if timestamp_col not in df.columns:
        timestamp_col = _detect_timestamp_column(df.columns)

    df = df.rename(columns={timestamp_col: "timestamp"})

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required OHLCV column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    horizon_delta = pd.Timedelta(minutes=horizon_minutes)

    five_min_index = None
    if five_min_csv:
        df_5 = _load_five_min(Path(five_min_csv))
        if df_5 is not None:
            five_min_index = FiveMinIndex(df_5)

    labels = _label_candles(df, horizon_delta, tp_points, sl_points, five_min_index)
    if len(labels) != len(df):
        raise ValueError("Label count does not match number of candles.")
    if not set(labels).issubset(ALLOWED_LABELS):
        raise ValueError(f"Unexpected labels detected: {set(labels) - set(ALLOWED_LABELS)}")

    out = pd.DataFrame({"timestamp": df["timestamp"], "candle_type": labels})
    return out


def label_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    timestamp_col: str = "timestamp",
    horizon_minutes: int = 720,
    tp_points: float = 2000,
    sl_points: float = 1000,
    five_min_csv: Optional[str | Path] = None,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    df = pd.read_csv(input_csv)
    labels = label_dataframe(
        df,
        timestamp_col=timestamp_col,
        horizon_minutes=horizon_minutes,
        tp_points=tp_points,
        sl_points=sl_points,
        five_min_csv=five_min_csv,
    )
    merged = df.merge(labels, left_on=timestamp_col, right_on="timestamp", how="left")
    if "timestamp" != timestamp_col:
        merged = merged.drop(columns=["timestamp"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Label candles with TP/SL rules")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--horizon-minutes", type=int, default=720)
    parser.add_argument("--tp-points", type=float, default=2000)
    parser.add_argument("--sl-points", type=float, default=1000)
    parser.add_argument("--five-min-csv", default=None)

    args = parser.parse_args()

    label_csv(
        input_csv=args.input,
        output_csv=args.output,
        timestamp_col=args.timestamp_col,
        horizon_minutes=args.horizon_minutes,
        tp_points=args.tp_points,
        sl_points=args.sl_points,
        five_min_csv=args.five_min_csv,
    )


if __name__ == "__main__":
    main()
