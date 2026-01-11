from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    END_DATE,
    FIVE_MIN_CSV,
    OUTPUT_1H_BASELINE,
    OUTPUT_1H_WITH5M,
    START_DATE,
    TIMEFRAME_CONFIG,
    TRANSITION_OUTPUT,
)
from .label_cleaning import apply_high_precision_cleaning
from .label_quality import print_label_counts
from .labeling_core import run_labeling_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeling for BTCUSDT 1h.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Override 1h input CSV path")
    parser.add_argument("--five-min-csv", type=Path, default=FIVE_MIN_CSV, help="5m CSV path")
    parser.add_argument("--output-baseline", type=Path, default=OUTPUT_1H_BASELINE)
    parser.add_argument("--output-with5m", type=Path, default=OUTPUT_1H_WITH5M)
    parser.add_argument("--transition-output", type=Path, default=TRANSITION_OUTPUT)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clean", action="store_true", help="Enable high-precision label cleaning")
    parser.add_argument("--min-range-pct", type=float, default=None)
    parser.add_argument(
        "--cleaned-output-dir",
        type=Path,
        default=Path("pipeline/artifacts/labeling"),
        help="Where to store cleaned outputs",
    )
    return parser.parse_args()


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def main() -> None:
    args = parse_args()

    cfg = dict(TIMEFRAME_CONFIG)
    if args.input_csv is not None:
        cfg["1h"] = dict(cfg["1h"])
        cfg["1h"]["input_csv"] = args.input_csv

    start_date = _parse_date(args.start_date) if args.start_date else START_DATE
    end_date = _parse_date(args.end_date) if args.end_date else END_DATE

    run_labeling_pipeline(
        timeframe_config=cfg,
        five_min_csv=args.five_min_csv,
        start_date=start_date,
        end_date=end_date,
        output_baseline=args.output_baseline,
        output_with5m=args.output_with5m,
        transition_output=args.transition_output,
    )

    if args.clean:
        args.cleaned_output_dir.mkdir(parents=True, exist_ok=True)
        for output in (args.output_baseline, args.output_with5m):
            df = pd.read_csv(output)
            print_label_counts(f"Before cleaning: {output.name}", df["candle_type"], ["long", "short", "skip"])
            cleaned, stats = apply_high_precision_cleaning(
                df,
                min_range_pct=args.min_range_pct,
            )
            print_label_counts(f"After cleaning: {output.name}", cleaned["candle_type"], ["long", "short", "skip"])
            print(
                f"Cleaned {output.name}: {stats['changed_rows']} rows changed, "
                f"min_range_pct={stats['min_range_pct']}"
            )
            cleaned_path = args.cleaned_output_dir / f"{output.stem}_cleaned.csv"
            cleaned.to_csv(cleaned_path, index=False)


if __name__ == "__main__":
    main()