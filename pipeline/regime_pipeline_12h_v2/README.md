Regime-First, Uncertainty-Aware Pipeline (12h, v2)
==================================================

This pipeline discovers latent market regimes first, maps regimes to tradable vs non-tradable
using the existing label column (long/short/skip), and only predicts direction inside eligible
regimes with calibrated probabilities. Labels are used as-is and never redefined.

How to run
----------
From repo root:
```
python pipeline/regime_pipeline_12h_v2/src/run.py
```

What it does
------------
- Adds leakage-safe features (volatility, trend, range, RSI/Bollinger, volume state).
- Fits HMM and GMM regimes per fold (walk-forward, no shuffling).
- Maps regimes to tradable/non-tradable using label distribution and stability rules.
- Trains direction model only on eligible regimes and non-skip labels.
- Applies strict entropy and probability gates for trading decisions.

Eligibility rules
-----------------
Regime is eligible when all are satisfied:
- action_rate >= MIN_ACTION_RATE
- direction_purity >= MIN_PURITY (among non-skip samples)
- avg_duration >= MIN_DURATION
- leave_prob <= MAX_LEAVE_PROB

Results and outputs
-------------------
Outputs are written to:
- `pipeline/regime_pipeline_12h_v2/outputs/`
- `pipeline/regime_pipeline_12h_v2/figures/`
- `pipeline/regime_pipeline_12h_v2/artifacts/`
- `pipeline/regime_pipeline_12h_v2/report.md`
