# BetaTest Report v2

## Dataset Summary
- Rows: 30614
- Time range: 2020-03-25 10:00:00+00:00 to 2023-10-02 23:00:00+00:00
- Timestamp column: ts_utc
- Dropped duplicate timestamps: 0

## Split Diagnostics
- Holdout train: 2020-03-25 10:00:00+00:00 to 2023-01-16 20:00:00+00:00; test: 2023-01-16 21:00:00+00:00 to 2023-10-02 23:00:00+00:00

## Label Cleaning Experiments
### min_range_pct=0.0
- Counts: long=5399, short=5436, skip=19779
- Ambiguous true count: 8633
- Gate: precision_trade=0.410, recall_trade=0.279, coverage_trade=0.192
- Direction: precision_long=0.500, precision_short=0.208, gated_count=1174
- End-to-end: precision_long=0.500, precision_short=0.208, coverage_total=0.192
- Predictions: long=6, short=1168, skip=4949
- Multiclass: model=lightgbm, macro_f1=0.284
- Multiclass thresholds: no feasible thresholds

### min_range_pct=0.001
- Counts: long=5399, short=5436, skip=19779
- Ambiguous true count: 8633
- Gate: precision_trade=0.410, recall_trade=0.279, coverage_trade=0.192
- Direction: precision_long=0.500, precision_short=0.208, gated_count=1174
- End-to-end: precision_long=0.500, precision_short=0.208, coverage_total=0.192
- Predictions: long=6, short=1168, skip=4949
- Multiclass: model=lightgbm, macro_f1=0.284
- Multiclass thresholds: no feasible thresholds

### min_range_pct=0.002
- Counts: long=5399, short=5436, skip=19779
- Ambiguous true count: 8633
- Gate: precision_trade=0.410, recall_trade=0.279, coverage_trade=0.192
- Direction: precision_long=0.500, precision_short=0.208, gated_count=1174
- End-to-end: precision_long=0.500, precision_short=0.208, coverage_total=0.192
- Predictions: long=6, short=1168, skip=4949
- Multiclass: model=lightgbm, macro_f1=0.284
- Multiclass thresholds: no feasible thresholds

### min_range_pct=0.003
- Counts: long=5399, short=5436, skip=19779
- Ambiguous true count: 8633
- Gate: precision_trade=0.410, recall_trade=0.279, coverage_trade=0.192
- Direction: precision_long=0.500, precision_short=0.208, gated_count=1174
- End-to-end: precision_long=0.500, precision_short=0.208, coverage_total=0.192
- Predictions: long=6, short=1168, skip=4949
- Multiclass: model=lightgbm, macro_f1=0.284
- Multiclass thresholds: no feasible thresholds

### min_range_pct=0.004
- Counts: long=5399, short=5436, skip=19779
- Ambiguous true count: 8633
- Gate: precision_trade=0.410, recall_trade=0.279, coverage_trade=0.192
- Direction: precision_long=0.500, precision_short=0.208, gated_count=1174
- End-to-end: precision_long=0.500, precision_short=0.208, coverage_total=0.192
- Predictions: long=6, short=1168, skip=4949
- Multiclass: model=lightgbm, macro_f1=0.284
- Multiclass thresholds: no feasible thresholds

## Interpretation Hints
- If trade-vs-skip is near random but direction on trades is better, entry timing may be weak.
- If direction is random but trade-vs-skip is strong, direction labeling or features are weak.
- If probabilities are skewed to skip, the model may be collapsing to skip.
- If argmax differs from decision at high rates, thresholding or class order may be off.
