from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly.gate import apply_anomaly_gate, fit_anomaly_model
from .config import load_config
from .data_io import load_dataset
from .ensemble.abstain import decisions_from_probs
from .ensemble.calibration import maybe_calibrate_probs
from .ensemble.combine import combine_poe, combine_weighted_average
from .ensemble.weights import compute_model_weights
from .eval.consistency import evaluate_consistency
from .eval.reports import aggregate_fold_reports
from .eval.trading import evaluate_trading
from .export.artifacts import save_fold_candles, save_ledgers
from .export.excel import build_excel_summary
from .features import select_features
from .labeling import apply_labels
from .models.gmm import GMMRegimeModel, run_gmm_sweep_fold, select_top_gmm_per_fold
from .splits import Fold, walk_forward_months
from .state_tables import build_state_label_table, state_posteriors_to_label_probs
from .utils import LABELS, ensure_dir, save_json, setup_logging, stable_seed

logger = setup_logging()

STAGES = ["sweep", "build-state-tables", "ensemble", "eval", "excel"]


def _stage_index(stage: str) -> int:
    if stage == "run":
        return len(STAGES) - 1
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return STAGES.index(stage)


def _fold_key(fold: Fold, featureset: str) -> str:
    return f"fold_{fold.fold_id}__feat_{featureset}"


def _model_id(row: pd.Series) -> str:
    reg = f"{float(row['reg_covar']):.0e}".replace("+", "")
    return f"gmm__k_{int(row['K'])}__cov_{row['covariance_type']}__reg_{reg}"


def _prob_col(prefix: str, label: str) -> str:
    return f"{prefix}__p_{label}"


def run_pipeline(config_path: str | Path, stage: str = "run") -> dict:
    stage_idx = _stage_index(stage)
    cfg = load_config(config_path)

    output_root = Path(cfg["output"]["root_dir"])
    folds_dir = ensure_dir(output_root / "folds")
    models_dir = ensure_dir(output_root / "models")
    ensemble_dir = ensure_dir(output_root / "ensemble")
    reports_dir = ensure_dir(output_root / "reports")
    ledger_dir = ensure_dir(output_root / "ledger")

    df_raw, numeric_cols = load_dataset(cfg["data"])
    df = apply_labels(df_raw, cfg)

    ts_col = cfg["data"]["timestamp_col"]
    label_col = cfg["data"]["label_col"]

    folds = walk_forward_months(
        df=df,
        timestamp_col=ts_col,
        train_months=int(cfg["splits"]["train_months"]),
        test_months=int(cfg["splits"]["test_months"]),
        step_months=int(cfg["splits"]["step_months"]),
        min_train_rows=int(cfg["splits"]["min_train_rows"]),
    )
    if not folds:
        raise ValueError("No folds generated.")

    feature_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    for featureset_cfg in cfg["featuresets"]:
        f_name = str(featureset_cfg["name"])
        feature_cache[f_name] = select_features(df, numeric_cols, featureset_cfg)

    gmm_sweep_frames: list[pd.DataFrame] = []
    gmm_top_frames: list[pd.DataFrame] = []
    weights_rows: list[dict] = []
    summary_rows: list[dict] = []
    distribution_rows: list[dict] = []

    for featureset_cfg in cfg["featuresets"]:
        featureset = str(featureset_cfg["name"])
        X_all, feat_cols = feature_cache[featureset]
        logger.info("Featureset %s with %d features", featureset, len(feat_cols))

        for fold in folds:
            fold_key = _fold_key(fold, featureset)
            logger.info("Processing %s", fold_key)
            X_train = X_all[fold.train_idx]
            X_test = X_all[fold.test_idx]

            for model_cfg in cfg["regime_models"]:
                if model_cfg.get("name") != "gmm" or not model_cfg.get("enabled", True):
                    continue
                all_df, agg_df = run_gmm_sweep_fold(
                    X_train=X_train,
                    X_test=X_test,
                    sweep_cfg=model_cfg["sweep"],
                    fold_id=fold.fold_id,
                    featureset=featureset,
                    tol=float(model_cfg["sweep"].get("tol", 1e-3)),
                )
                drop_cols = [c for c in ["weights_arr", "means_arr"] if c in all_df.columns]
                gmm_sweep_frames.append(all_df.drop(columns=drop_cols, errors="ignore"))
                if not agg_df.empty:
                    agg_df["train_start"] = fold.train_start
                    agg_df["train_end"] = fold.train_end
                    agg_df["test_start"] = fold.test_start
                    agg_df["test_end"] = fold.test_end
                    top_df = select_top_gmm_per_fold(agg_df, model_cfg["selection"])
                    gmm_top_frames.append(top_df)

            if stage_idx == _stage_index("sweep"):
                continue

            if not gmm_top_frames:
                continue
            gmm_top_all = pd.concat(gmm_top_frames, ignore_index=True)
            fold_top = gmm_top_all[
                (gmm_top_all["fold_id"] == fold.fold_id) & (gmm_top_all["featureset"] == featureset)
            ].copy()
            if fold_top.empty:
                logger.warning("No top models for %s", fold_key)
                continue

            ts = pd.to_datetime(df[ts_col], utc=True)
            period_mask = (ts >= fold.train_start) & (ts < fold.test_end)
            period_idx = np.where(period_mask)[0]
            df_period = df.iloc[period_idx].copy()
            X_period = X_all[period_idx]
            train_pos = np.where(np.isin(period_idx, fold.train_idx))[0]
            test_pos = np.where(np.isin(period_idx, fold.test_idx))[0]

            model_prob_list: list[np.ndarray] = []
            model_meta_rows: list[dict] = []

            for _, row in fold_top.iterrows():
                mid = _model_id(row)
                model_seed = stable_seed(
                    int(cfg["regime_models"][0]["sweep"]["seed"]),
                    fold.fold_id,
                    featureset,
                    mid,
                )
                model = GMMRegimeModel(
                    n_components=int(row["K"]),
                    covariance_type=str(row["covariance_type"]),
                    reg_covar=float(row["reg_covar"]),
                    max_iter=int(row["max_iter"]),
                    n_init=int(row["n_init"]),
                    init_params=str(row["init_params"]),
                    seed=model_seed,
                )
                try:
                    model.fit(X_train)
                except Exception as exc:
                    logger.exception("Model fit failed for %s: %s", mid, exc)
                    continue

                zprob = model.predict_state_proba(X_period)
                zhard = zprob.argmax(axis=1).astype(int)
                n_states = zprob.shape[1]

                labels_train = df_period.iloc[train_pos][label_col]
                table = build_state_label_table(
                    zhard_train=zhard[train_pos],
                    labels_train=labels_train,
                    n_states=n_states,
                    alpha=float(cfg["state_to_label"]["smoothing_alpha"]),
                )
                label_prob = state_posteriors_to_label_probs(zprob, table["probs"])
                model_prob_list.append(label_prob)

                prefix = f"m_{mid}"
                for k in range(n_states):
                    df_period[f"{prefix}__zprob_{k}"] = zprob[:, k]
                df_period[f"{prefix}__zhard"] = zhard
                for j, lbl in enumerate(LABELS):
                    df_period[_prob_col(prefix, lbl)] = label_prob[:, j]

                # Distribution of each model's implied action over full fold period.
                implied = np.array([LABELS[i] for i in label_prob.argmax(axis=1)], dtype=object)
                distribution_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "featureset": featureset,
                        "source": "model",
                        "id": mid,
                        "n_total": int(len(implied)),
                        "n_long": int(np.sum(implied == "long")),
                        "n_short": int(np.sum(implied == "short")),
                        "n_skip": int(np.sum(implied == "skip")),
                    }
                )

                model_out_dir = ensure_dir(models_dir / fold_key / mid)
                model.save(model_out_dir / "model.pkl")
                save_json(model_out_dir / "state_label_table.json", table)
                save_json(model_out_dir / "selection_metrics.json", row.to_dict())

                model_df = pd.DataFrame({ts_col: df_period[ts_col].values})
                for k in range(n_states):
                    model_df[f"zprob_{k}"] = zprob[:, k]
                model_df["zhard"] = zhard
                for j, lbl in enumerate(LABELS):
                    model_df[f"p_{lbl}"] = label_prob[:, j]
                model_df.to_csv(model_out_dir / "candles_model_probs.csv", index=False)

                meta = row.to_dict()
                meta["model_id"] = mid
                model_meta_rows.append(meta)

            if not model_prob_list:
                continue

            if stage_idx == _stage_index("build-state-tables"):
                fold_stage_dir = ensure_dir(folds_dir / fold_key)
                save_fold_candles(df_period, fold_stage_dir / "ensemble_candles.csv", cfg["output"]["float_format"])
                continue

            metrics_df = pd.DataFrame(model_meta_rows)
            metrics_df = compute_model_weights(metrics_df, cfg["ensemble"]["weights"])
            weights = metrics_df["ensemble_weight"].to_numpy(dtype=float)
            for _, wr in metrics_df.iterrows():
                weights_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "featureset": featureset,
                        "model_id": wr["model_id"],
                        "ensemble_weight": wr["ensemble_weight"],
                        "ensemble_score": wr["ensemble_score"],
                    }
                )

            pA = combine_weighted_average(model_prob_list, weights)
            pB = combine_poe(model_prob_list, weights)
            pA = maybe_calibrate_probs(pA, cfg["ensemble"].get("calibration", {}))
            pB = maybe_calibrate_probs(pB, cfg["ensemble"].get("calibration", {}))

            dcfg = cfg["decision_policy"]
            actA, metaA = decisions_from_probs(
                pA,
                tau_trade=float(dcfg["tau_trade"]),
                tau_margin=float(dcfg["tau_margin"]),
                tau_entropy=dcfg.get("tau_entropy"),
                abstain_label=str(dcfg.get("abstain_label", "skip")),
            )
            actB, metaB = decisions_from_probs(
                pB,
                tau_trade=float(dcfg["tau_trade"]),
                tau_margin=float(dcfg["tau_margin"]),
                tau_entropy=dcfg.get("tau_entropy"),
                abstain_label=str(dcfg.get("abstain_label", "skip")),
            )

            anomaly_scores = np.zeros(len(df_period), dtype=float)
            anomaly_flags = np.zeros(len(df_period), dtype=bool)
            anomaly_threshold = float("nan")
            if cfg["anomaly_gate"].get("enabled", False):
                anom_seed = stable_seed(int(cfg["regime_models"][0]["sweep"]["seed"]), fold.fold_id, featureset, "anom")
                anomaly_model, anomaly_threshold = fit_anomaly_model(X_train, cfg["anomaly_gate"], anom_seed)
                anomaly_scores = anomaly_model.score(X_period)
                actA, anomaly_flags = apply_anomaly_gate(
                    probs=pA,
                    decisions=actA,
                    anomaly_scores=anomaly_scores,
                    threshold=anomaly_threshold,
                    gate_cfg=cfg["anomaly_gate"],
                    decision_cfg=dcfg,
                )
                actB, _ = apply_anomaly_gate(
                    probs=pB,
                    decisions=actB,
                    anomaly_scores=anomaly_scores,
                    threshold=anomaly_threshold,
                    gate_cfg=cfg["anomaly_gate"],
                    decision_cfg=dcfg,
                )

            for j, lbl in enumerate(LABELS):
                df_period[f"ensA_p_{lbl}"] = pA[:, j]
                df_period[f"ensB_p_{lbl}"] = pB[:, j]
            df_period["ensA_action"] = actA
            df_period["ensB_action"] = actB
            df_period["ensA_p_max"] = metaA["p_max"]
            df_period["ensA_margin"] = metaA["margin"]
            df_period["ensA_entropy"] = metaA["entropy"]
            df_period["ensB_p_max"] = metaB["p_max"]
            df_period["ensB_margin"] = metaB["margin"]
            df_period["ensB_entropy"] = metaB["entropy"]
            df_period["anomaly_score"] = anomaly_scores
            df_period["anomaly_flag"] = anomaly_flags

            fold_stage_dir = ensure_dir(folds_dir / fold_key)
            save_fold_candles(df_period, fold_stage_dir / "ensemble_candles.csv", cfg["output"]["float_format"])
            save_json(
                ensure_dir(ensemble_dir / fold_key) / "metadata.json",
                {
                    "fold_id": fold.fold_id,
                    "featureset": featureset,
                    "anomaly_threshold": anomaly_threshold,
                    "n_models": len(model_prob_list),
                    "model_ids": metrics_df["model_id"].tolist(),
                },
            )

            distribution_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "featureset": featureset,
                    "source": "ensemble",
                    "id": "optionA",
                    "n_total": int(len(df_period)),
                    "n_long": int(np.sum(df_period["ensA_action"] == "long")),
                    "n_short": int(np.sum(df_period["ensA_action"] == "short")),
                    "n_skip": int(np.sum(df_period["ensA_action"] == "skip")),
                }
            )
            distribution_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "featureset": featureset,
                    "source": "ensemble",
                    "id": "optionB",
                    "n_total": int(len(df_period)),
                    "n_long": int(np.sum(df_period["ensB_action"] == "long")),
                    "n_short": int(np.sum(df_period["ensB_action"] == "short")),
                    "n_skip": int(np.sum(df_period["ensB_action"] == "skip")),
                }
            )

            if stage_idx == _stage_index("ensemble"):
                continue

            df_test = df_period.iloc[test_pos].copy()
            y_true = df_test[label_col].astype(str).to_numpy()
            yA = df_test["ensA_action"].astype(str).to_numpy()
            yB = df_test["ensB_action"].astype(str).to_numpy()

            fold_report_dir = ensure_dir(reports_dir / fold_key)
            cA = evaluate_consistency(
                y_true=y_true,
                y_pred=yA,
                out_json=fold_report_dir / "consistency_optionA.json",
                out_png=fold_report_dir / "consistency_optionA.png",
            )
            cB = evaluate_consistency(
                y_true=y_true,
                y_pred=yB,
                out_json=fold_report_dir / "consistency_optionB.json",
                out_png=fold_report_dir / "consistency_optionB.png",
            )
            fee = float(cfg["evaluation"].get("fee", 0.0005))
            tA = evaluate_trading(df_test, action_col="ensA_action", fee=fee)
            tB = evaluate_trading(df_test, action_col="ensB_action", fee=fee)
            save_json(fold_report_dir / "trading_optionA.json", tA)
            save_json(fold_report_dir / "trading_optionB.json", tB)

            summary_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "featureset": featureset,
                    "coverage_A": cA["coverage"],
                    "precision_A": cA["precision_macro"],
                    "recall_A": cA["recall_macro"],
                    "f1_A": cA["f1_macro"],
                    "signals_per_month_A": tA["signals_per_month"],
                    "winrate_A": tA["winrate"],
                    "avg_pnl_A": tA["avg_pnl"],
                    "profit_factor_A": tA["profit_factor"],
                    "max_drawdown_A": tA["max_drawdown"],
                    "coverage_B": cB["coverage"],
                    "precision_B": cB["precision_macro"],
                    "recall_B": cB["recall_macro"],
                    "f1_B": cB["f1_macro"],
                    "signals_per_month_B": tB["signals_per_month"],
                    "winrate_B": tB["winrate"],
                    "avg_pnl_B": tB["avg_pnl"],
                    "profit_factor_B": tB["profit_factor"],
                    "max_drawdown_B": tB["max_drawdown"],
                }
            )

    gmm_sweep_all = pd.concat(gmm_sweep_frames, ignore_index=True) if gmm_sweep_frames else pd.DataFrame()
    gmm_top_per_fold = pd.concat(gmm_top_frames, ignore_index=True) if gmm_top_frames else pd.DataFrame()
    ensemble_weights = pd.DataFrame(weights_rows)
    ensemble_summary = pd.DataFrame(summary_rows)
    distribution_df = pd.DataFrame(distribution_rows)

    save_ledgers(
        ledger_dir=ledger_dir,
        gmm_sweep_all=gmm_sweep_all,
        gmm_top_per_fold=gmm_top_per_fold,
        ensemble_weights=ensemble_weights,
        ensemble_summary=ensemble_summary,
    )
    if not distribution_df.empty:
        distribution_df.to_csv(ledger_dir / "ensemble_distribution.csv", index=False)

    if stage_idx >= _stage_index("eval") and not ensemble_summary.empty:
        aggregate_fold_reports(summary_rows, reports_dir)

    if stage_idx >= _stage_index("excel"):
        xlsx_path = build_excel_summary(output_root)
        logger.info("Excel summary written: %s", xlsx_path)

    return {
        "output_root": output_root,
        "n_folds": len(folds),
        "gmm_sweep_rows": int(len(gmm_sweep_all)),
        "gmm_top_rows": int(len(gmm_top_per_fold)),
        "ensemble_rows": int(len(ensemble_summary)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised Regime->Action Ensemble v1")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ["run", "sweep", "build-state-tables", "ensemble", "eval", "excel"]:
        p = sub.add_parser(cmd)
        p.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, stage=args.command)


if __name__ == "__main__":
    main()
