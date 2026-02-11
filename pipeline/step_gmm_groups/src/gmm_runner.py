from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from .metrics import compute_run_metrics


@dataclass
class RunOutcome:
    ledger_row: Dict[str, object]
    diagnostics: Optional[Dict[str, np.ndarray]]


def _base_ledger_row(
    group_name: str,
    fold_meta: Dict[str, object],
    feature_meta: Dict[str, object],
    model_meta: Dict[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "group_name": group_name,
        **fold_meta,
        **feature_meta,
        **model_meta,
    }
    return row


def run_single_gmm(
    group_name: str,
    fold_meta: Dict[str, object],
    feature_meta: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    covariance_type: str,
    seed: int,
    reg_covar: float,
    gmm_cfg: Dict[str, object],
) -> RunOutcome:
    model_meta = {
        "n_components": int(k),
        "covariance_type": str(covariance_type),
        "seed": int(seed),
        "reg_covar": float(reg_covar),
        "init_params": str(gmm_cfg.get("init_params", "kmeans")),
        "n_init": int(gmm_cfg.get("n_init", 5)),
        "max_iter": int(gmm_cfg.get("max_iter", 500)),
        "tol": float(gmm_cfg.get("tol", 1e-3)),
    }
    row = _base_ledger_row(group_name, fold_meta, feature_meta, model_meta)

    try:
        np.random.seed(seed)
        model = GaussianMixture(
            n_components=int(k),
            covariance_type=str(covariance_type),
            random_state=int(seed),
            n_init=int(gmm_cfg.get("n_init", 5)),
            reg_covar=float(reg_covar),
            max_iter=int(gmm_cfg.get("max_iter", 500)),
            tol=float(gmm_cfg.get("tol", 1e-3)),
            init_params=str(gmm_cfg.get("init_params", "kmeans")),
        )
        t0 = time.perf_counter()
        model.fit(X_train)
        runtime = time.perf_counter() - t0

        metrics, diagnostics = compute_run_metrics(
            gmm=model,
            X_train=X_train,
            X_test=X_test,
            runtime_fit_seconds=runtime,
        )
        row.update(metrics)
        row["success"] = True
        row["error_message"] = ""
        return RunOutcome(ledger_row=row, diagnostics=diagnostics)
    except Exception as exc:
        row.update(
            {
                "success": False,
                "error_message": str(exc),
                "runtime_fit_seconds": float("nan"),
            }
        )
        return RunOutcome(ledger_row=row, diagnostics=None)

