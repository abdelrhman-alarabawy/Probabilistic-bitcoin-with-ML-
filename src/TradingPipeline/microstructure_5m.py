from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import EARLY_LATE_K, LOOKBACK_HOURS, MIN_5M_BARS, RET_5M_Q


@dataclass(frozen=True)
class FiveMinIndex:
    timestamps: pd.DatetimeIndex
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray

    def slice_indices(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[int, int]:
        left = self.timestamps.searchsorted(start, side="left")
        right = self.timestamps.searchsorted(end, side="left")
        return int(left), int(right)


def build_5m_index(df5: pd.DataFrame) -> FiveMinIndex:
    return FiveMinIndex(
        timestamps=pd.DatetimeIndex(df5["timestamp"]),
        opens=df5["open"].to_numpy(),
        highs=df5["high"].to_numpy(),
        lows=df5["low"].to_numpy(),
        closes=df5["close"].to_numpy(),
        volumes=df5["volume"].to_numpy(),
    )


def compute_train_return_cutoff(
    df5: pd.DataFrame,
    train_end: pd.Timestamp,
    quantile: float = RET_5M_Q,
) -> float:
    df_train = df5[df5["timestamp"] < train_end]
    if df_train.empty:
        return float("nan")
    ret = (df_train["close"] / df_train["open"]) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return float("nan")
    return float(np.nanquantile(np.abs(ret), quantile))


def _safe_std(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.std(values, ddof=0))


def _safe_skew(values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def compute_microstructure_features(
    timestamps_1h: pd.Series,
    index_5m: FiveMinIndex,
    vol_scale: Optional[pd.Series] = None,
    lookback_hours: int = LOOKBACK_HOURS,
    min_bars: int = MIN_5M_BARS,
    early_late_k: int = EARLY_LATE_K,
    ret_cutoff: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expected_bars = int((lookback_hours * 60) / 5)
    eps = 1e-12

    feature_rows = []
    diag_rows = []

    for idx, ts in timestamps_1h.items():
        end = ts
        start = ts - pd.Timedelta(hours=lookback_hours)
        # Window end is exclusive: only 5m candles strictly before the 1h timestamp.
        left, right = index_5m.slice_indices(start, end)
        n = right - left
        missing_frac = 1.0 - min(n / expected_bars, 1.0) if expected_bars > 0 else 1.0

        if n > 0:
            first_ts = index_5m.timestamps[left]
            last_ts = index_5m.timestamps[right - 1]
            o_first = float(index_5m.opens[left])
            c_last = float(index_5m.closes[right - 1])
        else:
            first_ts = pd.NaT
            last_ts = pd.NaT
            o_first = float("nan")
            c_last = float("nan")

        diag_rows.append(
            {
                "timestamp_1h": ts,
                "window_start": start,
                "window_end": end,
                "first_5m_ts": first_ts,
                "last_5m_ts": last_ts,
                "n_5m": n,
                "expected_5m": expected_bars,
                "missing_frac": missing_frac,
                "o_first": o_first,
                "c_last": c_last,
            }
        )

        if n < min_bars:
            feature_rows.append(
                {
                    "ms_missing_frac": missing_frac,
                }
            )
            continue

        o = index_5m.opens[left:right]
        h = index_5m.highs[left:right]
        l = index_5m.lows[left:right]
        c = index_5m.closes[left:right]
        v = index_5m.volumes[left:right]

        ret_5m = (c / o) - 1.0
        log_ret = np.log(c[1:] / c[:-1]) if len(c) > 1 else np.array([])

        range_pct = (h - l) / np.maximum(o, eps)
        body = np.abs(c - o) / np.maximum(o, eps)
        wick = (h - np.maximum(o, c)) + (np.minimum(o, c) - l)
        wickiness = wick / np.maximum(h - l, eps)
        body_to_range = np.abs(c - o) / np.maximum(h - l, eps)

        k = min(early_late_k, len(c))
        ret_first_k = (c[k - 1] / o[0]) - 1.0
        ret_last_k = (c[-1] / o[-k]) - 1.0
        range_first_k = float(np.mean(range_pct[:k]))
        range_last_k = float(np.mean(range_pct[-k:]))

        if ret_cutoff is None or np.isnan(ret_cutoff):
            big_move_frac = float("nan")
        else:
            big_move_frac = float(np.mean(np.abs(ret_5m) > ret_cutoff))

        vol_sum = float(np.sum(v))
        vol_mean = float(np.mean(v))
        if vol_scale is None:
            vol_ratio = float("nan")
            vol_sum_ratio = float("nan")
            vol_mean_ratio = float("nan")
        else:
            denom = float(vol_scale.loc[idx]) if idx in vol_scale.index else float("nan")
            if denom == 0.0 or np.isnan(denom):
                vol_ratio = float("nan")
                vol_sum_ratio = float("nan")
                vol_mean_ratio = float("nan")
            else:
                vol_ratio = float(vol_sum / denom)
                vol_sum_ratio = vol_ratio
                vol_mean_ratio = float(vol_mean / (denom / max(expected_bars, 1)))

        signed_range = float(np.sum(np.sign(c - o) * range_pct))

        feature_rows.append(
            {
                "ms_ret_1h": float((c[-1] / o[0]) - 1.0),
                "ms_ret_sum": float(np.sum(ret_5m)),
                "ms_ret_std": _safe_std(log_ret),
                "ms_ret_skew": _safe_skew(log_ret),
                "ms_body_mean": float(np.mean(body)),
                "ms_range_mean": float(np.mean(range_pct)),
                "ms_wickiness_mean": float(np.mean(wickiness)),
                "ms_body_to_range_mean": float(np.mean(body_to_range)),
                "ms_up_candle_frac": float(np.mean(c > o)),
                "ms_down_candle_frac": float(np.mean(c < o)),
                "ms_range_max": float(np.max(range_pct)),
                "ms_big_move_frac": big_move_frac,
                "ms_ret_first_k": float(ret_first_k),
                "ms_ret_last_k": float(ret_last_k),
                "ms_range_first_k_mean": float(range_first_k),
                "ms_range_last_k_mean": float(range_last_k),
                "ms_vol_sum": float(vol_sum_ratio),
                "ms_vol_mean": float(vol_mean_ratio),
                "ms_vol_ratio": float(vol_ratio),
                "ms_signed_range": signed_range,
                "ms_missing_frac": missing_frac,
            }
        )

    features = pd.DataFrame(feature_rows, index=timestamps_1h.index)
    diag = pd.DataFrame(diag_rows, index=timestamps_1h.index)
    return features, diag
