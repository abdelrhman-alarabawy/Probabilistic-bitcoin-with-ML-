# 1D Feature Discovery Report

## Input Files
- data\external\1D\ohlcv_with_market_structure_indicators.csv
- data\external\1D\ohlcv_with_orderflow_derivatives_indicators.csv
- data\external\1D\ohlcv_with_oscillator_indicators.csv
- data\external\1D\ohlcv_with_trend_indicators.csv
- data\external\1D\ohlcv_with_volatility_indicators.csv
- data\external\1D\ohlcv_with_volume_indicators.csv

## Merge Summary
- inner_rows: 2106
- outer_rows: 2110
- merge_mode: inner
- duplicate_timestamps_by_file: {'ohlcv_with_market_structure_indicators.csv': 0, 'ohlcv_with_orderflow_derivatives_indicators.csv': 0, 'ohlcv_with_oscillator_indicators.csv': 0, 'ohlcv_with_trend_indicators.csv': 0, 'ohlcv_with_volatility_indicators.csv': 0, 'ohlcv_with_volume_indicators.csv': 0}

## Label Summary
- counts: {'skip': 1260, 'long': 442, 'short': 404}

## Feature Pruning
- initial_features: 341
- removed_constants: 2
- removed_corr: 61
- missingness_before: 0.0215
- missingness_after: 0.0000
- feature_shift: 1 (to avoid leakage)

## Outputs
- merged_raw: data\processed\1D_merged_indicators.csv
- merged_labeled: data\processed\1D_merged_indicators_labeled.csv
- rankings_dir: reports\feature_rankings
- plots_dir: reports\plots

## Class: LONG
- train_pos: 355
- train_neg: 1330
- imbalance_ratio (neg/pos): 3.75
- top_20_by_method: {"logit": ["peak_count_20d", "imbalance_ma_20", "lower_shadow", "var_99_60d", "iv_volatility_30d", "roc_12", "roc_9", "minus_di_14", "plus_di_14", "rmi_21_10", "roc_25", "psychological_line_12", "mass_index_x", "stoch_rsi_d_21", "stoch_rsi_21", "stoch_rsi_k_14", "adx_14_x", "momentum_14", "aroon_down_14", "stoch_rsi_14"], "tree": ["imbalance_ma_20", "volume_roc_12", "imbalance_last", "fly25_2d", "buy_volume", "fly25_momentum", "imbalance_momentum", "ask_amt_last", "rr25_1d", "vpci_10", "volume_roc_25", "returns_2d", "term_slope_1d_7d", "buy_pressure", "volume_ratio_50", "volume_trend_14", "lower_shadow", "hurst_exponent_60d", "fly25_change_1d", "vsa_signal"], "perm": ["peak_count_20d", "var_99_60d", "imbalance_ma_20", "returns_x", "returns_2d", "returns_5d", "returns_10d", "returns_20d", "returns_50d", "overnight_return", "high_low_range_pct", "cumulative_return", "cumulative_return_252d", "drawdown", "max_drawdown_20d", "max_drawdown_60d", "max_drawdown_252d", "is_underwater", "days_since_high", "returns_skewness_20d"], "mi": ["eom_14", "fly25_2d", "tr_percentile_50d", "volume_trend_14", "intraday_intensity_sum_30", "body_pct", "fly25_1d", "up_capture_60d", "returns_10d", "days_since_high", "price_volume_corr_20", "roc_9", "tr", "volume_sma_5", "klinger_signal", "bb_width_10", "price_range_pct_21", "gap_filled", "win_rate_20d", "returns_median_252d"], "ks": ["volume_roc_12", "volume_trend_14", "peak_count_20d", "imbalance_ma_20", "fly25_2d", "volume_ratio_50", "dpo_20", "var_99_60d", "force_index_ema_13", "returns_x", "negative_return", "roc_25", "tr_percentile_50d", "chaikin_volatility_10", "awesome_oscillator", "high_low_range_pct", "mfi_14_x", "eom_14", "rsi_21_x", "volume_roc_25"], "wasser": ["local_timestamp_last", "eom_14", "money_flow", "eom_20", "money_flow_sma_10", "force_index", "force_index_ema_13", "klinger_oscillator", "klinger_signal", "ad_line", "obv_x", "quote_updates", "intraday_intensity_20", "intraday_intensity_sum_21", "quote_intensity_ma_20", "intraday_intensity_sum_30", "quote_intensity_std_20", "bid_amt_last", "volume_momentum_20_y", "volume_momentum_10"], "cohen": ["peak_count_20d", "var_99_60d", "lower_shadow", "imbalance_ma_20", "iv_volatility_30d", "negative_return", "spread_pct_change", "ema_cross_12_26", "peak_count_60d", "consecutive_up_days", "returns_skewness_60d", "avg_loss_60d", "force_index", "high_low_range_pct", "distance_from_resistance_50d", "spread_last", "cci_30", "returns_2d", "atm_iv_1d", "tr"]}
- fused_top_20: ['peak_count_20d', 'imbalance_ma_20', 'tr_percentile_50d', 'awesome_oscillator', 'tr', 'dpo_20', 'returns_x', 'eom_14', 'days_since_high', 'roc_25', 'var_99_60d', 'tii_30', 'local_timestamp_last', 'rsi_7', 'body_pct', 'consecutive_up_days', 'buy_volume', 'volume_roc_25', 'chaikin_oscillator_12_26', 'returns_2d']
- stability_topk_logit_mean: 0.071, tree_mean: 0.071, perm_mean: 0.071
- separation_heuristic: weak (avg_ks=0.075, avg_wasser=19842058.060)
- PR-AUC (logit tuned, C=0.01): 0.2423

## Class: SHORT
- train_pos: 325
- train_neg: 1360
- imbalance_ratio (neg/pos): 4.18
- top_20_by_method: {"logit": ["vsa_signal", "volume_roc_12", "efficiency_ratio_20d", "updown_volume_ratio_20", "volume_momentum_20_x", "volume_oscillator_5_10", "stoch_rsi_21", "volume_delta", "plus_di_14", "adx_14_x", "rmi_21_10", "stoch_rsi_d_21", "mass_index_x", "minus_di_14", "stoch_rsi_k_14", "stoch_rsi_14", "aroon_oscillator_25_x", "aroon_oscillator_14_x", "psychological_line_12", "roc_9"], "tree": ["volume_roc_12", "vsa_signal", "rr25_momentum_1d", "volume_trend_21", "adx_14_x", "quote_intensity_std_20", "atr_pct_14", "volume_delta", "volume_oscillator_5_10", "imbalance_mean", "hurst_exponent_60d", "fly25_1d", "price_range_pct_7", "efficiency_ratio_20d", "rr25_1d", "lower_shadow", "volume_momentum_20_x", "imbalance_momentum", "imbalance_last", "fly25_momentum"], "perm": ["updown_volume_ratio_20", "volume_roc_12", "stoch_rsi_21", "returns_x", "returns_2d", "returns_5d", "returns_10d", "returns_20d", "returns_50d", "overnight_return", "high_low_range_pct", "cumulative_return", "cumulative_return_252d", "drawdown", "max_drawdown_20d", "max_drawdown_60d", "max_drawdown_252d", "is_underwater", "days_since_high", "returns_skewness_20d"], "mi": ["eom_20", "tii_30", "distance_from_support_20d", "up_capture_20d", "eom_14", "stoch_rsi_21", "supertrend_direction_14_2", "days_since_high", "elder_bear_power_13", "stoch_rsi_d_21", "ema_cross_12_26", "momentum_14", "price_range_pct_21", "chaikin_volatility_20", "returns_75th_252d", "donchian_width_20", "psychological_line_12", "negative_return", "vwbb_width_30", "supertrend_direction_10_3"], "ks": ["vsa_signal", "volume_roc_12", "efficiency_ratio_20d", "tr_percentile_14d", "volume_momentum_20_x", "stoch_rsi_21", "distance_from_resistance_20d", "rvi_14", "stoch_rsi_d_21", "volume_ratio_10", "rr25_1d", "position_in_range_20d", "mfi_20", "returns_20d", "price_volume_corr_20", "volume_roc_25", "up_capture_60d", "ulcer_index_14", "quote_intensity_std_20", "buy_volume"], "wasser": ["local_timestamp_last", "eom_14", "eom_20", "money_flow", "money_flow_sma_10", "force_index", "force_index_ema_13", "klinger_oscillator", "klinger_signal", "obv_x", "ad_line", "quote_intensity_ma_20", "quote_updates", "intraday_intensity_sum_30", "intraday_intensity_20", "intraday_intensity_sum_21", "quote_intensity_std_20", "bid_amt_last", "volume_momentum_20_y", "ask_amt_last"], "cohen": ["vsa_signal", "volume_roc_12", "volume_momentum_20_x", "volume_ratio_10", "tr_percentile_14d", "efficiency_ratio_20d", "volume_ratio_50", "updown_volume_ratio_20", "volume_oscillator_5_10", "positive_return", "volume_delta", "high_low_range_pct", "stoch_rsi_21", "buy_volume", "body_pct", "plus_di_14", "tmf_21", "ultimate_oscillator", "consecutive_up_days", "tr_percentile_50d"]}
- fused_top_20: ['volume_roc_12', 'stoch_rsi_21', 'quote_intensity_std_20', 'tr_percentile_14d', 'elder_bear_power_13', 'mfi_20', 'distance_from_support_20d', 'momentum_14', 'distance_from_resistance_20d', 'eom_20', 'days_since_high', 'stoch_rsi_d_21', 'klinger_signal', 'position_in_range_20d', 'cci_30', 'eom_14', 'vsa_signal', 'upper_shadow_pct', 'coppock_curve', 'returns_x']
- stability_topk_logit_mean: 0.071, tree_mean: 0.071, perm_mean: 0.071
- separation_heuristic: weak (avg_ks=0.086, avg_wasser=26931389.616)
- PR-AUC (logit tuned, C=0.01): 0.2113

## Class: SKIP
- train_pos: 1005
- train_neg: 680
- imbalance_ratio (neg/pos): 0.68
- top_20_by_method: {"logit": ["updown_volume_ratio_20", "vsa_signal", "volume_ratio_50", "iv_volatility_30d", "adx_14_x", "vwmacd_histogram", "volume_roc_12", "fly25_1d", "fly25_2d", "supertrend_direction_10_3", "lower_shadow", "upper_shadow", "body", "ema_cross_12_26", "r_squared_30d", "rr25_7d", "aroon_down_14", "peak_count_60d", "volume_trend_14", "aroon_oscillator_14_x"], "tree": ["volume_roc_12", "imbalance_last", "volume_roc_25", "vsa_signal", "volume_momentum_10", "volume_ratio_10", "lower_shadow", "imbalance_momentum", "ask_amt_last", "rr25_momentum_1d", "rr25_1d", "volume_ratio_50", "buy_pressure", "volume_momentum_20_x", "rr25_2d", "fly25_2d", "volume_oscillator_5_10", "fly25_momentum", "atm_iv_momentum_1d", "imbalance_mean"], "perm": ["volume_roc_12", "updown_volume_ratio_20", "vwmacd_histogram", "vsa_signal", "body", "peak_count_60d", "adx_14_x", "peak_count_20d", "volume_oscillator_5_10", "supertrend_direction_10_3", "ema_cross_12_26", "fly25_2d", "volume_ratio_50", "chaikin_oscillator_3_10", "fractal_dimension_120d", "r_squared_30d", "volume_roc_25", "ask_amt_last", "max_consecutive_down_60d", "klinger_signal"], "mi": ["coppock_curve", "tii_30", "days_since_high", "intraday_intensity_20", "intraday_intensity_sum_21", "volume_momentum_10", "roc_9", "efficiency_ratio_10d", "fly25_2d", "intraday_intensity_sum_30", "returns_2d", "returns_50d", "avg_loss_20d", "max_consecutive_up_60d", "fly25_1d", "iv_term_contango", "returns_median_252d", "vfi_260", "cumulative_return_252d", "price_range_pct_21"], "ks": ["volume_roc_12", "volume_roc_25", "volume_ratio_10", "tr_percentile_50d", "volume_ratio_50", "vsa_signal", "cmf_20_x", "tr_percentile_14d", "efficiency_ratio_20d", "volume_momentum_20_x", "bb_width_30", "price_volume_corr_20", "volume_trend_14", "fly25_2d", "volume_momentum_20_y", "tmf_21", "body_pct", "bb_width_20", "avg_loss_20d", "term_slope_1d_7d"], "wasser": ["local_timestamp_last", "eom_20", "eom_14", "money_flow", "money_flow_sma_10", "force_index", "force_index_ema_13", "klinger_oscillator", "klinger_signal", "ad_line", "obv_x", "quote_intensity_ma_20", "intraday_intensity_20", "quote_updates", "intraday_intensity_sum_21", "intraday_intensity_sum_30", "quote_intensity_std_20", "bid_amt_last", "volume_momentum_20_y", "volume_momentum_10"], "cohen": ["volume_roc_12", "tr_percentile_14d", "volume_ratio_50", "high_low_range_pct", "lower_shadow", "tr_percentile_50d", "volume_ratio_10", "volume_trend_14", "updown_volume_ratio_20", "cmf_20_x", "body_pct", "volume_momentum_20_x", "tr", "iv_volatility_30d", "body", "spread_last", "efficiency_ratio_20d", "bb_width_20", "spread_pct_change", "var_99_60d"]}
- fused_top_20: ['volume_roc_12', 'volume_roc_25', 'klinger_signal', 'vwmacd_histogram', 'vsa_signal', 'chaikin_oscillator_3_10', 'quote_intensity_std_20', 'body', 'adx_14_x', 'fly25_2d', 'volume_momentum_20_y', 'tr_percentile_14d', 'bb_width_20', 'upper_shadow_pct', 'volume_momentum_10', 'vwap_x', 'bb_width_30', 'bid_amt_last', 'intraday_intensity_sum_30', 'updown_volume_ratio_20']
- stability_topk_logit_mean: 0.071, tree_mean: 0.071, perm_mean: 0.071
- separation_heuristic: weak (avg_ks=0.070, avg_wasser=149037.834)
- PR-AUC (logit tuned, C=0.1): 0.6438
