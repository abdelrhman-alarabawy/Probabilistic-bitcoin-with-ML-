# Step: Multi-Group Gating

This step builds a systematic gating layer over the six labeled GMM group outputs and produces final trade-label percentages after gate filtering.

## What gating means

Each group is treated as an independent view. For each group and timestamp, we compute a per-group gate based on confidence proxies (probmax, entropy, state rarity). The six per-group gates are then combined (strict AND, K-of-N, or weighted score) to decide which candles pass.

## How thresholds are set (no look-ahead)

Thresholds are computed using the **early portion only** (default first 70% of time):

- `probmax_threshold`: 90th percentile of `gmm_probmax`
- `entropy_threshold`: 10th percentile of `gmm_entropy`
- `rarity_threshold`: 15th percentile of state frequencies (computed on hard states)

Per-group gate rule:

- pass if `probmax >= probmax_threshold` (if present)
- and `entropy <= entropy_threshold` (if present)
- and state frequency <= `rarity_threshold`

Gate score averages normalized components:

- `s_prob = clip((probmax - thr)/(1-thr), 0,1)` or `0.5` if probmax missing
- `s_ent = clip((thr - entropy)/thr, 0,1)` or `0.5` if entropy missing
- `s_rare = clip((rarity_thr - freq)/rarity_thr, 0,1)`

## Combining six groups

Supported modes:

- `strict_and`: all six gates must pass
- `k_of_n`: at least K groups pass (default K=4)
- `weighted_score`: weighted sum of gate scores >= threshold

## Final trade labels

Supported voting:

- `majority` (default): majority vote across groups, ties -> `skip`
- `reference`: use a single reference group’s labels
- `weighted`: weighted vote using the same weights as the weighted-score gate

## Output files

`pipeline/step_gmm_multi_group_gating/results`

- `combined_time_aligned.csv`
  - timestamp
  - `{group}_hard_state`, `{group}_probmax`, `{group}_entropy`, `{group}_trade_label`
  - `{group}_gate_pass`, `{group}_gate_score`
  - `final_pass`, `final_score`, `final_label`
- `final_pass_candles.csv`
  - only rows where `final_pass = 1`
- `summary.json`
  - coverage and final label percentages
- `per_group_thresholds.json`
- `per_group_pass_rates.csv`
- `label_vote_agreement.csv`

## CLI

```bash
python pipeline/step_gmm_multi_group_gating/run_gating.py ^
  --labeled_root "pipeline/step_gmm_groups_labeling/results" ^
  --model_rank 1 ^
  --mode "k_of_n" ^
  --k 4 ^
  --align "intersection" ^
  --prob_q 0.90 ^
  --ent_q 0.10 ^
  --rare_q 0.15 ^
  --vote "majority" ^
  --out_dir "pipeline/step_gmm_multi_group_gating/results"
```

## Tuning for rare but high precision

- Increase `prob_q` (e.g., 0.95)
- Decrease `ent_q` (e.g., 0.05)
- Decrease `rare_q` (e.g., 0.10)
- Increase K for `k_of_n`
- Use `weighted_score` with higher threshold
