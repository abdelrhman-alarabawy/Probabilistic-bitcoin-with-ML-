Regime-First, Uncertainty-Aware Pipeline (12h, v3)
==================================================

This pipeline discovers latent market regimes first, maps regimes to tradable vs non-tradable
using the existing candle_type labels (long/short/skip), then predicts direction only inside
eligible regimes with calibrated probabilities. Labels are used as-is and never redefined.

How to run
----------
From repo root:
```
python pipeline/regime_pipeline_12h_v3/src/run.py
```

Eligibility rules (default)
---------------------------
Regime is eligible when all are satisfied:
- n_actionable >= MIN_ACTION_SAMPLES
- action_rate >= MIN_ACTION_RATE
- directional_purity_actionable >= MIN_PURITY_ACTIONABLE OR directional_margin >= MIN_MARGIN
- avg_duration >= MIN_DURATION
- leave_prob <= MAX_LEAVE_PROB

What this version adds
----------------------
- Correct purity computed on actionable labels only.
- Additional purity diagnostics and label distributions per regime.
- Broader precision-vs-coverage frontier sweep with Pareto points.
- Expanded regime features for separation without leakage.

Outputs
-------
- `pipeline/regime_pipeline_12h_v3/outputs/` (regimes, stats, frontier, signals)
- `pipeline/regime_pipeline_12h_v3/figures/` (timelines, transitions, label bars, frontier)
- `pipeline/regime_pipeline_12h_v3/artifacts/` (models, scalers, features)
- `pipeline/regime_pipeline_12h_v3/report.md`
