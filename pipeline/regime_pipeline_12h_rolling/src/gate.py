from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class GateModelResult:
    model: Optional[CalibratedClassifierCV]
    info: dict


@dataclass
class GateDiagnostics:
    ap: float
    brier: float
    p_trade_stats: dict


def train_gate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: List[int],
    calibration_method: str,
    n_splits: int = 3,
) -> GateModelResult:
    if len(np.unique(y_train)) < 2:
        return GateModelResult(None, {"status": "single_class"})
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model = None
    best_brier = float("inf")
    best_info = {"status": "no_model"}

    for c_val in c_grid:
        base = LogisticRegression(max_iter=1000, class_weight="balanced", C=c_val)
        clf = CalibratedClassifierCV(base, method=calibration_method, cv=tscv)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_train)
        classes = list(clf.classes_)
        if 1 not in classes:
            continue
        idx = classes.index(1)
        p_trade = probs[:, idx]
        brier = brier_score_loss(y_train, p_trade)
        if brier < best_brier:
            best_brier = brier
            best_model = clf
            best_info = {"status": "ok", "C": c_val, "brier_train": brier}

    return GateModelResult(best_model, best_info)


def predict_trade_prob(model: Optional[CalibratedClassifierCV], X: np.ndarray) -> Optional[np.ndarray]:
    if model is None:
        return None
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return None
    idx = classes.index(1)
    return probs[:, idx]


def gate_thresholds_from_train(
    p_trade_train: np.ndarray,
    quantiles: List[int],
) -> Dict[str, float]:
    p_vals = p_trade_train[np.isfinite(p_trade_train)]
    thresholds: Dict[str, float] = {}
    if len(p_vals) == 0:
        return thresholds
    for q in quantiles:
        thresholds[f"q{int(q)}"] = float(np.percentile(p_vals, q))
    return thresholds


def topk_thresholds_from_train(
    p_trade_train: np.ndarray,
    topk_percents: List[int],
) -> Dict[int, float]:
    p_vals = p_trade_train[np.isfinite(p_trade_train)]
    thresholds: Dict[int, float] = {}
    if len(p_vals) == 0:
        return thresholds
    for k in topk_percents:
        thresholds[k] = float(np.percentile(p_vals, 100 - k))
    return thresholds


def compute_gate_diagnostics(
    y_true_gate: np.ndarray,
    p_trade: np.ndarray,
) -> GateDiagnostics:
    mask = np.isfinite(p_trade)
    if mask.any():
        ap = average_precision_score(y_true_gate[mask], p_trade[mask])
        brier = brier_score_loss(y_true_gate[mask], p_trade[mask])
        stats = {
            "min": float(np.nanmin(p_trade)),
            "p50": float(np.nanmedian(p_trade)),
            "max": float(np.nanmax(p_trade)),
        }
    else:
        ap = float("nan")
        brier = float("nan")
        stats = {"min": float("nan"), "p50": float("nan"), "max": float("nan")}
    return GateDiagnostics(ap=ap, brier=brier, p_trade_stats=stats)
