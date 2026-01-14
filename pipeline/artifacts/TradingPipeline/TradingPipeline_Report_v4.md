# TradingPipeline Report v4

## Dataset Summary
- Rows: 30614
- Time range: 2020-03-25 10:00:00+00:00 to 2023-10-02 23:00:00+00:00
- Decision time: open

## Split Diagnostics
- Train end: 2022-09-05 06:00:00+00:00
- Val end: 2023-03-22 14:00:00+00:00
- Test start: 2023-03-22 15:00:00+00:00

## Direction Training Filters
- wickiness_p80: 0.513452
- chop_p80: 0.513741
- ms_missing_frac_max: 0.20
- direction_train_count: 6834
- direction_train_trade_frac: 22.32%
- balance_method: class_weight
- calibration_method: sigmoid

## Threshold Selection
- T_trade: 0.3
- T_long: 0.5
- T_short: 0.5
- relaxation_log: ['t_trade_relaxed_to=0.30', 'coverage_relaxed_to=0.0050', 'min_count_relaxed_to=25', 'precision_target_relaxed_to=long:0.30,short:0.30']

## Baseline vs v4 (Test)
- Baseline precision_long=0.0000, precision_short=0.0000, trade_precision=0.3333, coverage=0.0007, long_pred_count=0, short_pred_count=3, expectancy=-0.0000, profit_factor=0.9634, max_drawdown=-0.0032
- v4 precision_long=0.2000, precision_short=0.1982, trade_precision=0.4019, coverage=0.1642, long_pred_count=300, short_pred_count=454, expectancy=-0.0014, profit_factor=0.3090, max_drawdown=-0.0032

## Artifacts
- threshold_trade_sweep: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_trade_threshold_sweep.csv
- threshold_direction_sweep: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_direction_threshold_sweep.csv
- chosen_thresholds: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_chosen_thresholds.json
- prediction_counts: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_prediction_counts.csv
- per_month_metrics: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_per_month_metrics.csv
- confusion_matrix: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\v4_confusion_matrix.csv
