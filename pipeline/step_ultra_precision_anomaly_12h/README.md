# Ultra Precision Anomaly Pipeline (12h BTC)

Goal: find very rare but very high-precision anomaly-like trading triggers using 12h BTC data from 2020-2025.

## What it does
- Builds leak-free features (shift=1) and forward returns (12h, 24h, 72h).
- Computes future path stats (MFE/MAE) using the next 6 candles.
- Fits multiple anomaly detectors on each train fold:
  - Robust Z-score anomalies (rolling median/MAD, aggregated)
  - Isolation Forest anomalies
  - GMM rare clusters
- Searches rule thresholds on train only (precision-first, low trade frequency).
- Evaluates rules on test in strict walk-forward splits:
  - Expanding: start -> t, test next 2 months
  - Rolling: last 12 months, test next 2 months
- Builds a combined strategy (union of rules, conflict timestamps skipped).

## Outputs
`results/anomaly_events.csv`
One row per detected test event with:
timestamp, event_type, score, cluster_id, direction_candidate,
forward_return_12h/24h/72h, MFE/MAE, meta_json

`results/rules_library.json`
Top rules with thresholds, direction, horizon, TP/SL suggestions and cross-val metrics.

`results/backtest_summary.csv`
Per-fold and overall metrics for the combined strategy.

`figures/`
event_timeline.png, equity_curve.png, precision_vs_coverage.png

## How to run
```bash
python pipeline/step_ultra_precision_anomaly_12h/run.py
```

## Notes
- No leakage: all features shifted by 1 and models fit on train only.
- Precision-first filtering: default target precision is 0.90 with trades/month <= 2 (set in `run.py`).
- If no rules are found, loosen thresholds in `run.py`.
