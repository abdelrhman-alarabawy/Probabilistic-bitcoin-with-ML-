# TradingPipeline Report

## Dataset Summary
- Rows: 30614
- Time range: 2020-03-25 10:00:00+00:00 to 2023-10-02 23:00:00+00:00
- Dropped duplicate timestamps: 0

## Split Diagnostics
- Holdout train end: 2023-01-16 20:00:00+00:00
- Holdout test start: 2023-01-16 21:00:00+00:00
- Train end: 2022-09-05 06:00:00+00:00
- Val end: 2023-03-22 14:00:00+00:00
- Test start: 2023-03-22 15:00:00+00:00

## Label Distribution
- train: long=19.05%, short=19.27%, skip=61.68%
- val: long=15.77%, short=15.74%, skip=68.49%
- test: long=12.91%, short=12.72%, skip=74.37%

## Risk Filters
- Volume ratio filter uses rolling median volume and train percentile cutoffs.
- Range z-score filter uses rolling mean/std of range percent and train percentiles.
- ATR percent filter uses rolling ATR and train percentile cutoffs.
- Cutoffs: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\risk_filter_cutoffs.csv
- Pass rates: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\risk_filter_pass_rates.csv
- Label distribution (filtered): D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\risk_filter_label_distribution.csv

## Microstructure (5m) Features
- Decision time: open (window end exclusive).
- No leakage: 5m candles are aligned by time window [t - lookback, t).
- Lookback hours: 1
- Min 5m bars: 8
- Ret cutoff (train P95): 0.004560424598481204
- Feature count: 21
- Alignment checks: PASS
- Alignment samples: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\microstructure_alignment_checks.csv
- Feature stats: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\microstructure_feature_stats.csv
- Debug samples: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\microstructure_debug_samples.csv

## Before vs After Microstructure
- Baseline: precision_long=0.1250, precision_short=0.2000, precision_trade=0.4301, coverage=0.0202, expectancy=-0.0015, profit_factor=0.2741, max_drawdown=-0.1352, gate_auc=0.6205
- Micro: precision_long=0.0000, precision_short=0.1935, precision_trade=0.4706, coverage=0.0074, expectancy=-0.0011, profit_factor=0.3830, max_drawdown=-0.0383, gate_auc=0.6499

## Threshold Sweep
- Sweep table: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\threshold_sweep.csv
- Precision vs coverage plot: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\figures\precision_coverage.png

## Selected Thresholds
- T_trade: 0.5
- T_long: 0.7
- T_short: 0.6
- Validation min precision: 0.2857142857142857

## Test Metrics
- precision_long: 0.0000
- precision_short: 0.1935
- precision_trade: 0.4706
- coverage_total: 0.0074
- gate_auc: 0.6499
- confusion matrix: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\figures\confusion_matrix.png
- per-month metrics: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\per_month_metrics.csv

## Backtest Summary
- expectancy: -0.0011
- win_rate: 0.3235
- profit_factor: 0.3830
- max_drawdown: -0.0383
- trades per month: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\trades_per_month.csv
- trade log: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\tables\test_trade_log.csv
- equity curve: D:\GitHub\bitcoin-probabilistic-learning\pipeline\artifacts\TradingPipeline\figures\equity_curve.png

## Verdict
If precision remains low or coverage collapses, use this pipeline as a filter rather than a standalone trading system.
