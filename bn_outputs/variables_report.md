# Random Variables Report

## Target RV
- candle_type: categorical label for candle type.

## Market State RVs
- VolState: discretized from _volatility using 3 quantile bins.
- TrendState: discretized from _trend using 3 quantile bins.
- RangeState: discretized from _range using 3 quantile bins.
- VolumeState: discretized from _volume_state using 3 quantile bins.

## Indicator RVs
- 43 indicators discretized into 3 quantile bins (train-only).