# BetaTest Report

## Dataset Summary
- Rows: 30614
- Time range: 2020-03-25 10:00:00+00:00 to 2023-10-02 23:00:00+00:00
- Timestamp column: ts_utc
- Dropped duplicate timestamps: 0

## Label Diagnostics
- Counts: long=5399, short=5436, skip=19779
- Percentages: long=0.176, short=0.178, skip=0.646
- Skip run length mean/median/p95: 3.05 / 2.00 / 9.00
- Rolling drift L1 mean/median/p95: 0.147 / 0.140 / 0.328
- Ambiguous true count: 8633 (0.282)

## Split Diagnostics
See split table in artifacts for full details.
- Holdout train: 2020-03-25 10:00:00+00:00 to 2023-01-16 20:00:00+00:00; test: 2023-01-16 21:00:00+00:00 to 2023-10-02 23:00:00+00:00

## Trade vs Skip (Binary) Results
- always_skip: precision=0.000, recall=0.000, f1=0.000, auc=None
- majority: precision=0.000, recall=0.000, f1=0.000, auc=None
- random: precision=0.286, recall=0.505, f1=0.365, auc=None
- logistic: precision=0.379, recall=0.053, f1=0.094, auc=0.5812299673676099
- lightgbm: precision=0.407, recall=0.035, f1=0.065, auc=0.6551192951994468

## Direction on Trades (Binary) Results
- logistic: precision_long=0.523, precision_short=0.546
- lightgbm: precision_long=0.504, precision_short=0.530

## Multiclass Results
- Model: lightgbm, macro_f1=0.284

## Threshold Diagnostics
- Best thresholds: long=0.95, short=0.95, coverage=0.000
- precision_long=0.000, precision_short=0.000, trade_precision=0.000

## Interpretation Hints
- If trade-vs-skip is near random but direction on trades is better, entry timing may be weak.
- If direction is random but trade-vs-skip is strong, direction labeling or features are weak.
- If probabilities are skewed to skip, the model may be collapsing to skip.
- If argmax differs from decision at high rates, thresholding or class order may be off.