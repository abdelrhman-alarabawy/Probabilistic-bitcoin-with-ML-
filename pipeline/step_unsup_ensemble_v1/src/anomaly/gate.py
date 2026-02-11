from __future__ import annotations

import numpy as np

from ..ensemble.abstain import decisions_from_probs
from .isolation_forest import IsolationForestAnomaly
from .mahalanobis import MahalanobisAnomaly


def fit_anomaly_model(X_train: np.ndarray, gate_cfg: dict, seed: int) -> tuple[object, float]:
    model_name = gate_cfg.get("model", "mahalanobis")
    q = float(gate_cfg.get("anomaly_threshold_quantile", 0.99))
    if model_name == "mahalanobis":
        model = MahalanobisAnomaly(robust=bool(gate_cfg.get("mahalanobis_robust", True))).fit(X_train)
    elif model_name == "isolation_forest":
        model = IsolationForestAnomaly(
            contamination=float(gate_cfg.get("contamination", 0.01)),
            seed=seed,
        ).fit(X_train)
    else:
        raise ValueError(f"Unsupported anomaly model: {model_name}")
    train_scores = model.score(X_train)
    threshold = float(np.quantile(train_scores, q))
    return model, threshold


def apply_anomaly_gate(
    probs: np.ndarray,
    decisions: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    gate_cfg: dict,
    decision_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    mode = gate_cfg.get("mode", "force_skip")
    abstain_label = decision_cfg.get("abstain_label", "skip")
    anomaly_mask = anomaly_scores >= threshold
    out = decisions.copy()

    if mode == "force_skip":
        out[anomaly_mask] = abstain_label
        return out, anomaly_mask

    if mode == "raise_thresholds":
        raised_trade = float(gate_cfg.get("raised_tau_trade", decision_cfg["tau_trade"]))
        raised_margin = float(gate_cfg.get("raised_tau_margin", decision_cfg["tau_margin"]))
        raised_entropy = decision_cfg.get("tau_entropy")
        stricter, _ = decisions_from_probs(
            probs=probs[anomaly_mask],
            tau_trade=raised_trade,
            tau_margin=raised_margin,
            tau_entropy=raised_entropy,
            abstain_label=abstain_label,
        )
        out[anomaly_mask] = stricter
        return out, anomaly_mask

    raise ValueError(f"Unsupported anomaly gate mode: {mode}")
