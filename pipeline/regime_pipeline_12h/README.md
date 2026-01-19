Regime-First, Uncertainty-Aware Pipeline (12h)
==============================================

This pipeline discovers latent market regimes first, then gates trading decisions
based on regime eligibility and uncertainty. Direction modeling only happens inside
eligible regimes and uses calibrated probabilities with high thresholds.

How regimes are selected
------------------------
- HMM (GaussianHMM) is the primary regime model.
- Candidate K values are evaluated on a time-ordered validation split of the train window.
- Selection criteria: highest validation log-likelihood, while avoiding degenerate regimes
  (minimum state fraction and average duration constraints).
- If hmmlearn is missing, the pipeline falls back to GMM-only regimes and logs the limitation.

Eligibility rules
-----------------
Each regime is mapped to Tradable vs Non-tradable using offline statistics on the train window:
- Minimum average duration.
- Tail loss constraint (5th percentile of forward returns).
- Minimum proxy win-rate (trend-following proxy).
- Transition risk (probability of leaving within a few bars).
- Minimum regime fraction.

Regime uncertainty gating
-------------------------
Every prediction includes a regime entropy score. If entropy exceeds the threshold,
the pipeline skips trading regardless of eligibility.

How to run
----------
From repo root:
```
python pipeline/regime_pipeline_12h/run_pipeline.py
```

Outputs are written to:
- `pipeline/regime_pipeline_12h/outputs/`
- `pipeline/regime_pipeline_12h/artifacts/`
- `pipeline/regime_pipeline_12h/figures/`
- `pipeline/regime_pipeline_12h/regime_pipeline_12h_report.md`
