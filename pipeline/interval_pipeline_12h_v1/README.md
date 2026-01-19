# Interval Prediction Pipeline (12h, v1)

This pipeline builds quantile-based prediction intervals for next-candle HIGH/LOW
and reports coverage, interval width, tight-precision metrics, and trade-off plots.

## Quick Start

From `pipeline/interval_pipeline_12h_v1`:

```bash
python -m src.run
```

## Inputs

- `data/processed/12h_features_indicators_with_ohlcv.csv`
- Uses `pipeline/gate_module_12h_v1/outputs/features_used.json` for feature selection.

## Outputs

- `outputs/per_window_metrics.csv`
- `outputs/aggregate_metrics.csv`
- `outputs/frontier_high.csv`
- `outputs/frontier_low.csv`
- `outputs/confusion_high_W*.csv`
- `outputs/confusion_low_W*.csv`
- `figures/coverage_vs_width_high.png`
- `figures/coverage_vs_width_low.png`
- `figures/precision_tight_vs_width_threshold_high.png`
- `figures/precision_tight_vs_width_threshold_low.png`
- `figures/confusion_matrix_high_W*_all.png`
- `figures/confusion_matrix_low_W*_all.png`
- `figures/yearly_summary_width_and_coverage.png`
- `report.md`

## Notes

- Targets are next-candle HIGH/LOW (H=1 by default).
- Labels are unchanged; `candle_type` is only used for optional gating.
