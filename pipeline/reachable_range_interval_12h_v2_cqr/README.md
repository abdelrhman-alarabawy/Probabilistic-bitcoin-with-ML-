# Reachable Range Interval Pipeline (12h, v2 CQR)

This pipeline trains a conformalized quantile regression (CQR) ensemble for reachable
high/low extremes over the next N candles and reports coverage, width, and tight precision.

## Quick Start

From `pipeline/reachable_range_interval_12h_v2_cqr`:

```bash
python -m src.run
```

## Inputs

- `data/processed/12h_features_indicators_with_ohlcv.csv`
- Uses `pipeline/gate_module_12h_v1/outputs/features_used.json` for feature selection.

## Outputs

- `outputs/per_window.csv`
- `outputs/aggregate.csv`
- `outputs/confusion_high_reach.csv`
- `outputs/confusion_low_reach.csv`
- `figures/coverage_timeline.png`
- `figures/width_timeline.png`
- `figures/confusion_matrix_high_reach.png`
- `figures/confusion_matrix_low_reach.png`
- `figures/gate_vs_all.png`
- `report.md`

## Notes

- Targets are reachable high/low over the next N candles (default N=3).
- Labels are unchanged; `candle_type` is only used for optional gating.
