# Gate Module 12h v1

This module trains a calibrated Trade-vs-Skip gate model and exports a TOP-K
ranking signal for downstream strategies. It does not predict direction.

## Quick Start

From `pipeline/gate_module_12h_v1`:

```bash
python -m src.run
```

## Inputs

- `data/processed/12h_features_indicators_with_ohlcv.csv`
- Auto-detects timestamp column and `candle_type` label.

## Outputs

- `outputs/gate_scores.csv`: per-test-window ranking signal (TOP-K=5)
- `outputs/gate_confusion_by_window.csv`
- `outputs/gate_confusion_aggregate.csv`
- `outputs/baselines_by_window.csv`
- `outputs/features_used.json`
- `figures/pr_curves_by_window.png`
- `figures/confusion_matrices_K5.png`
- `figures/precision_vs_K.png`
- `artifacts/scaler.joblib`
- `artifacts/gate_model.joblib`
- `artifacts/gate_calibrator.joblib`
- `artifacts/metadata.json`

## How to Use the Gate Score

- Use `selected_topk_flag` to allow trades only at the top-5 ranked timestamps.
- Use `p_trade` to scale exposure (higher `p_trade` = more tradeable).

## Notes

- Labels are unchanged: `{long, short, skip}`.
- Features are leakage-safe and scaled with `RobustScaler`.
