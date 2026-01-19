# Regime-First Rolling Pipeline Report (12h)

## Data
- Rows used: 2404
- Features used: 83
- Label column: candle_type
- Excluded columns: timestamp, candle_type, open, high, low, close, volume, label_ambiguous

## Window Length Selection
| Window | Median gate AP | Median precision<=5% | % windows w/ trades | Stability (std) |
| --- | --- | --- | --- | --- |
| train18_test3 | 0.400 | 0.400 | 0.71 | 0.388 |
| train12_test3 | 0.382 | 0.000 | 0.44 | 0.335 |

- Selected window length: train18_test3

## GOOD vs BAD Windows (offline, test-based)
- train12_test3 window 0: BAD | gate_AP=0.517 | best_prec=0.000 | trades=0 | policy=none | reason=fail:precision,trades
- train12_test3 window 1: BAD | gate_AP=0.553 | best_prec=0.000 | trades=0 | policy=none | reason=fail:precision,trades
- train12_test3 window 2: BAD | gate_AP=0.466 | best_prec=0.500 | trades=6 | policy=threshold|q95|dir0.55|entinf | reason=fail:precision
- train12_test3 window 3: BAD | gate_AP=0.352 | best_prec=0.000 | trades=8 | policy=threshold|q80|dir0.52|entinf | reason=fail:gate_AP,precision
- train12_test3 window 4: BAD | gate_AP=0.382 | best_prec=0.000 | trades=0 | policy=none | reason=fail:gate_AP,precision,trades
- train12_test3 window 5: BAD | gate_AP=0.474 | best_prec=1.000 | trades=1 | policy=threshold|q93|dir0.6|entinf | reason=fail:trades
- train12_test3 window 6: BAD | gate_AP=0.225 | best_prec=0.000 | trades=0 | policy=none | reason=fail:gate_AP,precision,trades
- train12_test3 window 7: BAD | gate_AP=0.304 | best_prec=0.000 | trades=0 | policy=none | reason=fail:gate_AP,precision,trades
- train12_test3 window 8: BAD | gate_AP=0.351 | best_prec=0.400 | trades=5 | policy=threshold|q80|dir0.58|entinf | reason=fail:gate_AP,precision
- train18_test3 window 0: GOOD | gate_AP=0.492 | best_prec=0.400 | trades=5 | policy=topk|topk_1|dir0.52|entinf | reason=ok
- train18_test3 window 1: BAD | gate_AP=0.363 | best_prec=1.000 | trades=2 | policy=topk|topk_1|dir0.52|entinf | reason=fail:gate_AP,trades
- train18_test3 window 2: BAD | gate_AP=0.400 | best_prec=0.000 | trades=0 | policy=none | reason=fail:gate_AP,precision,trades
- train18_test3 window 3: BAD | gate_AP=0.490 | best_prec=0.000 | trades=0 | policy=none | reason=fail:precision,trades
- train18_test3 window 4: BAD | gate_AP=0.224 | best_prec=0.250 | trades=4 | policy=topk|topk_1|dir0.52|entinf | reason=fail:gate_AP,precision,trades
- train18_test3 window 5: BAD | gate_AP=0.304 | best_prec=0.500 | trades=2 | policy=threshold|q80|dir0.58|entinf | reason=fail:gate_AP,precision,trades
- train18_test3 window 6: BAD | gate_AP=0.439 | best_prec=1.000 | trades=2 | policy=threshold|q93|dir0.52|entinf | reason=fail:trades

## Recommended Policy (Robust)
- Default policy: topk 2 dir_thr=0.55 entropy_max=0.7
- Best p10 precision policy: topk topk_1 dir_thr=0.52 entropy_max=0.6 (p10=0.000, median=0.200, coverage=0.021)

## Adaptive Policy Notes
- Adaptive signals use prior-window GOOD/BAD labels only (no peeking into the current test window).
- GOOD -> top_k up to 5%, direction_thr >= 0.55; BAD -> top_k 1%, direction_thr 0.60.

## Conclusion
- Direction remains near coin-flip at 12h; the label likely encodes weak directional signal.
- Gate ranking remains useful for selecting the most tradeable slices under drift.