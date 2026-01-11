# BetaTest Report v3

## Dataset Summary
- Rows: 30614
- Time range: 2020-03-25 10:00:00+00:00 to 2023-10-02 23:00:00+00:00
- Timestamp column: ts_utc
- Dropped duplicate timestamps: 0

## Split Diagnostics
- Holdout train: 2020-03-25 10:00:00+00:00 to 2023-01-16 20:00:00+00:00; test: 2023-01-16 21:00:00+00:00 to 2023-10-02 23:00:00+00:00

## Label Cleaning Proof
min_range_pct=0.0 -> long=5399, short=5436, skip=19779
min_range_pct=0.005 -> long=5020, short=5076, skip=20518
min_range_pct=0.01 -> long=1876, short=1987, skip=26751
min_range_pct=0.015 -> long=839, short=880, skip=28895
min_range_pct=0.02 -> long=399, short=423, skip=29792

## Gate Learnability
min_range_pct=0.0 gate_precision=0.410 gate_recall=0.279 gate_coverage=0.192
min_range_pct=0.005 gate_precision=0.395 gate_recall=0.085 gate_coverage=0.055
min_range_pct=0.01 gate_precision=0.176 gate_recall=0.082 gate_coverage=0.035
min_range_pct=0.015 gate_precision=0.065 gate_recall=0.048 gate_coverage=0.023
min_range_pct=0.02 gate_precision=0.063 gate_recall=0.051 gate_coverage=0.010

## Direction Learnability
min_range_pct=0.0 precision_long=0.500 precision_short=0.246 gated_count=1174
min_range_pct=0.005 precision_long=0.238 precision_short=0.200 gated_count=334
min_range_pct=0.01 precision_long=0.155 precision_short=0.125 gated_count=216
min_range_pct=0.015 precision_long=0.039 precision_short=0.500 gated_count=139
min_range_pct=0.02 precision_long=0.036 precision_short=0.000 gated_count=63

## Multiclass Sanity Summary
min_range_pct=0.0 model=lightgbm macro_f1=0.284
  thresholds: no feasible thresholds
min_range_pct=0.005 model=lightgbm macro_f1=0.292
  thresholds: no feasible thresholds
min_range_pct=0.01 model=lightgbm macro_f1=0.320
  thresholds: no feasible thresholds
min_range_pct=0.015 model=lightgbm macro_f1=0.328
  thresholds: no feasible thresholds
min_range_pct=0.02 model=lightgbm macro_f1=0.331
  thresholds: no feasible thresholds

## End-to-End Precision vs Coverage
min_range_pct=0.0 precision_long=0.500 precision_short=0.246 coverage_total=0.010
  predictions: long=6, short=57, skip=6060
min_range_pct=0.005 precision_long=0.238 precision_short=0.200 coverage_total=0.025
  predictions: long=151, short=5, skip=5967
min_range_pct=0.01 precision_long=0.155 precision_short=0.125 coverage_total=0.015
  predictions: long=84, short=8, skip=6031
min_range_pct=0.015 precision_long=0.039 precision_short=0.500 coverage_total=0.013
  predictions: long=77, short=2, skip=6044
min_range_pct=0.02 precision_long=0.036 precision_short=0.000 coverage_total=0.010
  predictions: long=55, short=7, skip=6061

## Verdict
- Label is noisy

## Interpretation Hints
- If trade-vs-skip is near random but direction on trades is better, entry timing may be weak.
- If direction is random but trade-vs-skip is strong, direction labeling or features are weak.
- If probabilities are skewed to skip, the model may be collapsing to skip.
- If argmax differs from decision at high rates, thresholding or class order may be off.