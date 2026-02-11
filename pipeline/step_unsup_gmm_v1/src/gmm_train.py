from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import load_config
from .data_io import load_dataset
from .features import select_features
from .metrics import aggregate_runs
from .splits import walk_forward_months
from .export_top_models import select_top_models
from .utils import (
    ensure_dir,
    save_json,
    seed_everything,
    setup_logging,
    stable_seed,
    validate_no_nans,
    validate_responsibilities,
)


logger = setup_logging()


def _grid(cfg_gmm: dict) -> list[dict]:
    keys = ["Ks", "covariance_types", "reg_covar", "max_iter", "n_init", "init_params"]
    values = [cfg_gmm.get(k, []) for k in keys]
    grid = []
    for K, cov, reg, max_iter, n_init, init_params in product(*values):
        grid.append(
            {
                "K": int(K),
                "covariance_type": cov,
                "reg_covar": float(reg),
                "max_iter": int(max_iter),
                "n_init": int(n_init),
                "init_params": init_params,
            }
        )
    return grid


def run_sweep(config_path: str | Path) -> dict:
    cfg = load_config(config_path)

    output_root = Path(cfg["output"]["root_dir"])
    runs_dir = ensure_dir(output_root / "runs")
    ledger_dir = ensure_dir(output_root / "ledger")
    reports_dir = ensure_dir(output_root / "reports")

    df, numeric_cols = load_dataset(cfg["data"], timezone=cfg["splits"].get("timezone", "UTC"))

    folds = walk_forward_months(
        df=df,
        timestamp_col=cfg["data"]["timestamp_col"],
        train_months=cfg["splits"]["train_months"],
        test_months=cfg["splits"]["test_months"],
        step_months=cfg["splits"]["step_months"],
        min_train_rows=cfg["splits"]["min_train_rows"],
    )

    if not folds:
        raise ValueError("No folds generated. Check split configuration or dataset size.")

    grid = _grid(cfg["gmm_sweep"])
    repeats = int(cfg["gmm_sweep"].get("repeats_per_config", 1))
    base_seed = int(cfg["gmm_sweep"].get("random_seed", 42))
    tol = float(cfg["gmm_sweep"].get("convergence_tol", 1e-3))

    run_records: list[dict] = []
    ledger_rows: list[dict] = []

    total_runs = len(cfg["featuresets"]) * len(folds) * len(grid) * repeats
    logger.info("Total runs: %d", total_runs)

    for featureset_cfg in cfg["featuresets"]:
        featureset_name = featureset_cfg.get("name", "featureset")
        x_all, feature_cols = select_features(df, numeric_cols, featureset_cfg)

        feature_path = ledger_dir / f"features_{featureset_name}.json"
        save_json(feature_path, {"featureset": featureset_name, "features": feature_cols})

        for fold in folds:
            x_train = x_all[fold.train_idx]
            x_test = x_all[fold.test_idx]

            validate_no_nans(x_train, f"X_train fold {fold.fold_id} {featureset_name}")
            validate_no_nans(x_test, f"X_test fold {fold.fold_id} {featureset_name}")
            if len(fold.train_idx) and len(fold.test_idx):
                if fold.train_idx.max() >= fold.test_idx.min():
                    raise ValueError(
                        f"Train/Test overlap detected for fold {fold.fold_id} ({featureset_name})."
                    )

            for cfg_item in grid:
                for repeat in range(repeats):
                    run_seed = stable_seed(
                        base_seed,
                        featureset_name,
                        fold.fold_id,
                        cfg_item["K"],
                        cfg_item["covariance_type"],
                        cfg_item["reg_covar"],
                        repeat,
                    )

                    seed_everything(run_seed)

                    gmm = GaussianMixture(
                        n_components=cfg_item["K"],
                        covariance_type=cfg_item["covariance_type"],
                        reg_covar=cfg_item["reg_covar"],
                        max_iter=cfg_item["max_iter"],
                        n_init=cfg_item["n_init"],
                        init_params=cfg_item["init_params"],
                        tol=tol,
                        random_state=run_seed,
                    )

                    status = "ok"
                    error = ""

                    try:
                        gmm.fit(x_train)

                        train_avg_loglik = float(gmm.score(x_train))
                        test_avg_loglik = float(gmm.score(x_test))

                        train_total_loglik = train_avg_loglik * len(x_train)
                        test_total_loglik = test_avg_loglik * len(x_test)

                        train_aic = float(gmm.aic(x_train))
                        train_bic = float(gmm.bic(x_train))
                        test_aic = float(gmm.aic(x_test))
                        test_bic = float(gmm.bic(x_test))

                        resp_test = gmm.predict_proba(x_test)
                        validate_responsibilities(resp_test)

                        eps = 1e-12
                        entropy = float(
                            np.mean(-np.sum(resp_test * np.log(resp_test + eps), axis=1))
                        )

                        weights = gmm.weights_.astype(float)
                        means_flat = gmm.means_.astype(float).reshape(-1)

                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                        train_avg_loglik = math.nan
                        test_avg_loglik = math.nan
                        train_total_loglik = math.nan
                        test_total_loglik = math.nan
                        train_aic = math.nan
                        train_bic = math.nan
                        test_aic = math.nan
                        test_bic = math.nan
                        entropy = math.nan
                        weights = np.array([])
                        means_flat = np.array([])

                    run_id = (
                        f"fold{fold.fold_id}_feat{featureset_name}_k{cfg_item['K']}_"
                        f"cov{cfg_item['covariance_type']}_reg{cfg_item['reg_covar']}_"
                        f"rep{repeat}"
                    )

                    run_payload = {
                        "run_id": run_id,
                        "status": status,
                        "error": error,
                        "featureset": featureset_name,
                        "fold_id": fold.fold_id,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "n_train": len(x_train),
                        "n_test": len(x_test),
                        "K": cfg_item["K"],
                        "covariance_type": cfg_item["covariance_type"],
                        "reg_covar": cfg_item["reg_covar"],
                        "max_iter": cfg_item["max_iter"],
                        "n_init": cfg_item["n_init"],
                        "init_params": cfg_item["init_params"],
                        "repeat": repeat,
                        "seed": run_seed,
                        "train_avg_loglik": train_avg_loglik,
                        "train_total_loglik": train_total_loglik,
                        "test_avg_loglik": test_avg_loglik,
                        "test_total_loglik": test_total_loglik,
                        "train_aic": train_aic,
                        "train_bic": train_bic,
                        "test_aic": test_aic,
                        "test_bic": test_bic,
                        "entropy": entropy,
                        "weights": weights,
                        "means_flat": means_flat,
                    }

                    run_records.append(run_payload)

                    ledger_rows.append({k: v for k, v in run_payload.items() if k not in {"weights", "means_flat"}})

                    run_path = runs_dir / run_id / "run.json"
                    save_json(run_path, run_payload)

                    if status == "ok":
                        logger.info(
                            "OK fold=%s feat=%s k=%s cov=%s reg=%s rep=%s test_ll=%.6f",
                            fold.fold_id,
                            featureset_name,
                            cfg_item["K"],
                            cfg_item["covariance_type"],
                            cfg_item["reg_covar"],
                            repeat,
                            test_avg_loglik,
                        )
                    else:
                        logger.warning(
                            "FAIL fold=%s feat=%s k=%s cov=%s reg=%s rep=%s error=%s",
                            fold.fold_id,
                            featureset_name,
                            cfg_item["K"],
                            cfg_item["covariance_type"],
                            cfg_item["reg_covar"],
                            repeat,
                            error,
                        )

    ledger_all_path = ledger_dir / "gmm_ledger_all.csv"
    pd.DataFrame(ledger_rows).to_csv(ledger_all_path, index=False)

    group_cols = [
        "featureset",
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "K",
        "covariance_type",
        "reg_covar",
        "max_iter",
        "n_init",
        "init_params",
    ]

    agg_df = aggregate_runs(run_records, group_cols)
    ledger_agg_path = ledger_dir / "gmm_ledger_aggregated.csv"
    agg_df.to_csv(ledger_agg_path, index=False)

    top_df = pd.DataFrame()
    if not agg_df.empty:
        top_df = select_top_models(agg_df, cfg)
        top_path = ledger_dir / "top_10_models.csv"
        top_df.to_csv(top_path, index=False)

    report_path = reports_dir / "run_report.md"
    _write_report(
        report_path,
        cfg,
        df,
        folds,
        grid,
        agg_df,
        top_df,
        ledger_all_path,
        ledger_agg_path,
    )

    return {
        "ledger_all": ledger_all_path,
        "ledger_agg": ledger_agg_path,
        "report": report_path,
    }


def _write_report(
    report_path: Path,
    cfg: dict,
    df: pd.DataFrame,
    folds: list,
    grid: list,
    agg_df: pd.DataFrame,
    top_df: pd.DataFrame,
    ledger_all_path: Path,
    ledger_agg_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Unsupervised GMM Sweep Report\n")
    lines.append(f"Dataset: `{cfg['data']['csv_path']}`")
    lines.append(f"Rows: {len(df)}")
    lines.append(
        f"Date range: {df[cfg['data']['timestamp_col']].min()} -> {df[cfg['data']['timestamp_col']].max()}"
    )
    lines.append(f"Numeric columns: {len(df.select_dtypes(include=[float, int]).columns)}\n")
    lines.append(f"Folds: {len(folds)}")
    lines.append(f"Grid size: {len(grid)}")
    lines.append(
        f"Total configs (featuresets x folds x grid): {len(cfg['featuresets']) * len(folds) * len(grid)}\n"
    )
    lines.append(f"Ledger all: `{ledger_all_path}`")
    lines.append(f"Ledger aggregated: `{ledger_agg_path}`\n")

    if not top_df.empty:
        lines.append("## Top 10 Preview")
        lines.append(_df_to_markdown(top_df))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    df = df.head(max_rows)
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns.tolist()]
    rows = df.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
