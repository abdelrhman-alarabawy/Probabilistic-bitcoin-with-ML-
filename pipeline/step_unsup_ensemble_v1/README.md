# Unsupervised Regime -> Action Ensemble v1

This project builds a purely unsupervised regime system, converts regime posteriors to label probabilities, and combines multiple regime models into final actions with abstention and anomaly gating.

## Key Idea

1. Train unsupervised regime models on `X` only (no supervised classifier).
2. Use rule labels (`candle_type`) only to estimate `p(label | state)` on train data.
3. Convert each model's state posterior to label posterior:
   - `p_m(label | x_t) = sum_k p(label | z_t=k) p_m(z_t=k | x_t)`
4. Combine models:
   - Option A (Weighted Average): `pA = sum_m w_m p_m`
   - Option B (PoE): `pB(label) proportional to product_m p_m(label) ^ w_m`
5. Apply abstain policy and optional anomaly gate.
6. Evaluate walk-forward out-of-sample.

## Install

```bash
pip install -r pipeline/step_unsup_ensemble_v1/requirements.txt
```

## Run

```bash
python -m pipeline.step_unsup_ensemble_v1.src.cli run --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
```

Subcommands:

```bash
python -m pipeline.step_unsup_ensemble_v1.src.cli sweep --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
python -m pipeline.step_unsup_ensemble_v1.src.cli build-state-tables --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
python -m pipeline.step_unsup_ensemble_v1.src.cli ensemble --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
python -m pipeline.step_unsup_ensemble_v1.src.cli eval --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
python -m pipeline.step_unsup_ensemble_v1.src.cli excel --config pipeline/step_unsup_ensemble_v1/configs/ensemble_v1.yaml
```

## Abstain Policy

Given label probabilities `p(label | x_t)`:

- `p_max = max_label p(label | x_t)`
- `margin = p_top1 - p_top2`
- `entropy = -sum_label p log(p)`

Trade if:

- `p_max >= tau_trade`
- `margin >= tau_margin`
- and if configured: `entropy <= tau_entropy`

Otherwise action is `abstain_label` (default: `skip`).

## Anomaly Gate

Optional anomaly model (`mahalanobis` or `isolation_forest`) is trained on train features.

Modes:

- `force_skip`: anomalous rows are forced to `skip`
- `raise_thresholds`: anomalous rows use stricter thresholds

## Output Layout

Default root: `pipeline/step_unsup_ensemble_v1/results`

- `folds/fold_<id>__feat_<name>/ensemble_candles.csv`
- `models/fold_<id>__feat_<name>/<model_id>/...`
- `ensemble/fold_<id>__feat_<name>/...`
- `reports/fold_<id>__feat_<name>/...`
- `ledger/gmm_sweep_all.csv`
- `ledger/gmm_top_per_fold.csv`
- `ledger/ensemble_weights.csv`
- `ledger/ensemble_summary.csv`
- `reports/ensemble_summary.xlsx`
