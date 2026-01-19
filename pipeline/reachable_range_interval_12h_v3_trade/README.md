# Reachable Range Trade Pipeline (12h, v3)

This pipeline converts reachable range intervals into trade/no‑trade rules and evaluates
confusion matrices against the existing candle_type labels.

## Quick Start

From `pipeline/reachable_range_interval_12h_v3_trade`:

```bash
python -m src.run
```

## Inputs

- `data/processed/12h_features_indicators_with_ohlcv.csv`
- Uses `pipeline/gate_module_12h_v1/outputs/features_used.json` for feature selection.

## Outputs

- `outputs/trade_confusion_by_window.csv`
- `outputs/trade_confusion_aggregate.csv`
- `outputs/interval_metrics_by_window.csv`
- `outputs/tightness_thresholds.csv`
- `figures/confusion_trade_*.png`
- `report.md`

## Notes

- Targets are reachable high/low over the next N candles (default N=3).
- Labels are unchanged; `candle_type` defines actual trade vs no‑trade.
- `LIQ_POLICY` controls whether liq_* features are dropped or imputed with missingness flags.
