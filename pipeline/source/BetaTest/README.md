# BetaTest Diagnostics

Run
- `python -m pipeline.source.BetaTest.run_betatests`

Artifacts
- Reports: `pipeline/artifacts/BetaTest/reports/`
- Figures: `pipeline/artifacts/BetaTest/figures/`
- Tables: `pipeline/artifacts/BetaTest/tables/`
- Predictions: `pipeline/artifacts/BetaTest/predictions/`

Key outputs to share
1) Label cleaning effects (`label_cleaning_effects.csv`)
2) Gate metrics (`gate_threshold_grid_*.csv` + best setting in report)
3) Direction metrics (`direction_threshold_grid_*.csv` + `direction_band_grid_*.csv`)
4) Best thresholds + coverage (`best_thresholds_*.json`)
5) Probability sanity sample (`probability_sanity_sample_*.csv`)
6) Direction debug sample (`direction_debug_sample.csv`)
7) Multiclass prob stats (`multiclass_prob_stats.csv`)
8) Final report (`BetaTest_Report_v3.md`)
