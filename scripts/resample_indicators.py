from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def detect_timestamp_column(columns: pd.Index) -> str:
    if "timestamp" in columns:
        return "timestamp"
    for candidate in ("ts_utc", "time", "open_time", "datetime"):
        if candidate in columns:
            return candidate
    for col in columns:
        if "timestamp" in col.lower():
            return col
    raise ValueError("No timestamp column found in input CSV.")


def read_hourly_data(filepath: Path) -> pd.DataFrame:
    """Read the 1-hour candle CSV file."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    ts_col = detect_timestamp_column(df.columns)
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def get_aggregation_rules(df: pd.DataFrame) -> dict[str, str]:
    """Define how each column should be aggregated."""
    agg_dict: dict[str, str] = {}

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in {"open", "high", "low", "close"}:
            if col_lower == "open":
                agg_dict[col] = "first"
            elif col_lower == "high":
                agg_dict[col] = "max"
            elif col_lower == "low":
                agg_dict[col] = "min"
            else:
                agg_dict[col] = "last"
        elif col_lower == "volume":
            agg_dict[col] = "sum"
        elif col in ["quote_updates"]:
            agg_dict[col] = "sum"
        elif col.endswith("_last") or col == "local_timestamp_last":
            agg_dict[col] = "last"
        elif col.endswith("_std"):
            agg_dict[col] = "mean"
        elif col.endswith("_mean"):
            agg_dict[col] = "mean"
        elif col.endswith("_amt_last"):
            agg_dict[col] = "last"
        else:
            agg_dict[col] = "mean"

    return agg_dict


def resample_data(df: pd.DataFrame, frequency: str, agg_dict: dict[str, str]) -> pd.DataFrame:
    """Resample data to specified frequency."""
    return df.resample(frequency).agg(agg_dict)


def process_candles(
    input_file: Path,
    output_1d: Path,
    output_4h: Path,
    output_12h: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Reading data from {input_file}...")
    df_1h = read_hourly_data(input_file)

    print(f"Original data shape: {df_1h.shape}")
    print(f"Date range: {df_1h.index.min()} to {df_1h.index.max()}")

    agg_dict = get_aggregation_rules(df_1h)

    print("\nCreating 1-day candles...")
    df_1d = resample_data(df_1h, "1D", agg_dict).dropna(how="all")
    print(f"1-day candles shape: {df_1d.shape}")

    print("\nCreating 4-hour candles...")
    df_4h = resample_data(df_1h, "4H", agg_dict).dropna(how="all")
    print(f"4-hour candles shape: {df_4h.shape}")

    print("\nCreating 12-hour candles...")
    df_12h = resample_data(df_1h, "12H", agg_dict).dropna(how="all")
    print(f"12-hour candles shape: {df_12h.shape}")

    print(f"\nSaving 1-day candles to {output_1d}...")
    df_1d.to_csv(output_1d)

    print(f"Saving 4-hour candles to {output_4h}...")
    df_4h.to_csv(output_4h)

    print(f"Saving 12-hour candles to {output_12h}...")
    df_12h.to_csv(output_12h)

    print("\nDone! Files created successfully.")
    return df_1d, df_4h, df_12h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample 1h indicators into 1D, 4H, and 12H aggregates.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the 1h indicators CSV (must include 'timestamp' column).",
    )
    parser.add_argument(
        "--output-1d",
        type=Path,
        default=Path("data/processed/features_1d_ALL.csv"),
        help="Output CSV for 1D aggregates.",
    )
    parser.add_argument(
        "--output-4h",
        type=Path,
        default=Path("data/processed/features_4h_ALL.csv"),
        help="Output CSV for 4H aggregates.",
    )
    parser.add_argument(
        "--output-12h",
        type=Path,
        default=Path("data/processed/features_12h_ALL.csv"),
        help="Output CSV for 12H aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_candles(
        input_file=args.input_file,
        output_1d=args.output_1d,
        output_4h=args.output_4h,
        output_12h=args.output_12h,
    )


if __name__ == "__main__":
    main()
