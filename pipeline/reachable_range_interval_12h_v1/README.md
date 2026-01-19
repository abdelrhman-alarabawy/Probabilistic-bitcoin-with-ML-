# Reachable Range Interval Pipeline (12h, v1)

This pipeline predicts intervals for reachable high/low extremes over the next N candles
and reports coverage, interval width, tight precision, and gate-vs-all comparisons.

## Quick Start

From `pipeline/reachable_range_interval_12h_v1`:

```bash
python -m src.run
```

## Inputs

- `data/processed/12h_features_indicators_with_ohlcv.csv`
- Uses `pipeline/gate_module_12h_v1/outputs/features_used.json` for feature selection.

## Outputs

- `outputs/per_window_metrics.csv`
- `outputs/aggregate_metrics.csv`
- `outputs/confusion_high_reach_W*.csv`
- `outputs/confusion_low_reach_W*.csv`
- `outputs/frontier_high_reach.csv`
- `outputs/frontier_low_reach.csv`
- `figures/coverage_vs_width_high_reach.png`
- `figures/coverage_vs_width_low_reach.png`
- `figures/tight_precision_vs_W.png`
- `figures/confusion_matrix_high_reach_W*.png`
- `figures/confusion_matrix_low_reach_W*.png`
- `figures/yearly_summary.png`
- `figures/gate_vs_all_comparison.png`
- `report.md`

## Notes

- Targets are reachable high/low over the next N candles (default N=3).
- Labels are unchanged; `candle_type` is only used for optional gating.
