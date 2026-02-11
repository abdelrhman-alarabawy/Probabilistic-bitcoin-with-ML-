# Step: GMM Groups Labeling

This step reads each group `top10.csv` from `pipeline/step_gmm_groups/results`, refits those GMM configs, computes per-candle regime states/probabilities, and attaches `long/short/skip` labels from your script:

- `scripts/signals_code_hour_v1_0.py`

## Run

From repo root:

```bash
python pipeline/step_gmm_groups_labeling/run_label_top10.py ^
  --root "data/external/1d" ^
  --prev_results "pipeline/step_gmm_groups/results" ^
  --label_script "D:\GitHub\bitcoin-probabilistic-learning\scripts\signals_code_hour_v1_0.py" ^
  --mode "fit_train_predict_all" ^
  --shift 1
```

Or use defaults:

```bash
python pipeline/step_gmm_groups_labeling/run_label_top10.py
```

## Label script invocation

- `signals_code_hour_v1_0.py` is script-style (top-level execution, no CLI args for custom input path).
- This step runs it **as-is** in an isolated temporary working directory via subprocess:
  - writes the current group OHLCV data to `data_12h_indicators.csv` inside temp dir
  - executes `python <label_script_path>`
  - reads generated output from:
    - `pipeline/source/labeling/output_12h_labels_baseline_no5m.csv` (default)
    - or `output_12h_labels_with5m.csv` when `label_output_mode: with5m`
- No label rules are reimplemented or modified.

## Leakage control

- Modeling features are shifted by `shift` (default `1`): row `t` uses indicator values from `t-1`.
- OHLCV used by labeling remains unshifted.
- Rows are sorted chronologically by timestamp before fitting/prediction.

## Fit mode

- `fit_train_predict_all` (default):
  - if `train_start/train_end` exist in the top10 row, those timestamps define train set
  - otherwise fallback to chronological holdout train (first 80%)
  - predict regime probabilities for all rows
- `fit_all_predict_all`:
  - fit and predict on all rows (offline analysis mode)

## Output structure

`pipeline/step_gmm_groups_labeling/results`

- `<group_name>/model_rank01/labeled.csv`
- `<group_name>/model_rank01/gmm_config.json`
- `<group_name>/model_rank01/diagnostics.json`
- ...
- `<group_name>/model_rank10/...`
- `ALL/combined_index.csv` (paths + metadata for all exports)

Each `labeled.csv` includes:

- timestamp
- OHLCV (if present)
- selected modeling features (configurable)
- `gmm_hard_state`
- `gmm_prob_state_0..K-1`
- `gmm_probmax`
- `gmm_entropy`
- `trade_label` in `{long, short, skip}` when labeling succeeded

`diagnostics.json` includes:

- fit metadata (converged, n_iter, train rows, features used, shift)
- label-script execution metadata (return code and output tail)
- regime-to-label distribution (`P(label | state)` counts and percentages)

## Assumptions

- Group files are the same six datasets used in previous step.
- `top10.csv` exists per group in previous results root.
- If OHLCV is missing, labeling is skipped with warning; GMM states are still exported.

