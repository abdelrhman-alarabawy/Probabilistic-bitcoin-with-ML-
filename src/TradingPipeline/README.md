# TradingPipeline

This module builds a reproducible, strictly chronological BTC 1H trading pipeline focused on high-precision trade selection.

## Quickstart
- Ensure the labeled CSV exists at:
  - `data/processed/features_1h_ALL-2025_merged_prev_indicators_labeled.csv`
- Ensure the 5m CSV exists at:
  - `data/processed/BTCUSDT_5m_2026-01-03.csv`
- Run:
  - `python -m TradingPipeline.run_pipeline`

## Outputs
Artifacts are written to:
- `pipeline/artifacts/TradingPipeline/reports/TradingPipeline_Report.md`
- `pipeline/artifacts/TradingPipeline/tables/`
- `pipeline/artifacts/TradingPipeline/figures/`

## Key Behavior
- Chronological splits only (no random sampling).
- All non-OHLCV indicators are lagged by one bar to prevent leakage.
- 5m microstructure features use the time window `[t - 1h, t)` (end exclusive).
- Risk filters use rolling, scale-invariant statistics with train-only percentiles.
- Decision rule uses a gate + direction model with explicit thresholds and rejection.

## Decision Timing
- `DECISION_TIME="open"` (default): decisions are made at the 1h open; features use data strictly before t.
- `DECISION_TIME="close"`: features are built for the hour ending at t and labels shift forward by one bar.

## Interpretation
- High precision with low coverage is expected; focus on whether precision stabilizes across months.
- If coverage collapses or precision is unstable, treat the signals as a filter rather than a standalone system.
