#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from src.features import apply_feature_shift, infer_base_feature_columns, prepare_fold_features
from src.folds import FoldDefinition, build_folds
from src.gmm_runner import run_single_gmm
from src.io import GroupInput, discover_group_csvs, ensure_dir, load_group_dataframe, write_json
from src.plots import save_group_plots
from src.ranker import rank_top_configs


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config YAML must contain a dictionary at the top level.")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GMM regime discovery/evaluation on grouped CSV files.")
    parser.add_argument("--config", type=str, default=str(CURRENT_DIR / "config.yaml"))
    parser.add_argument("--root", type=str, help="Root folder containing 6 group CSVs or 6 subfolders.")
    parser.add_argument("--out", type=str, help="Output folder for results.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override seed list.")
    parser.add_argument("--feature-mode", choices=["selected_top10", "all_features"], help="Feature mode override.")
    parser.add_argument("--selector-method", choices=["variance_prune", "pseudo_mi"], help="Selector method override.")
    parser.add_argument("--shift", type=int, help="Feature shift override.")
    parser.add_argument("--save-diagnostics", action="store_true", help="Save hard labels/responsibilities artifacts.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    parser.add_argument("--expected-groups", type=int, help="Expected number of groups (default from config).")
    parser.add_argument("--max-walkforward-folds", type=int, help="Optional cap for walk-forward folds.")
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = deepcopy(config)
    if args.root:
        updated["root_group_folder"] = args.root
    if args.out:
        updated["output_dir"] = args.out
    if args.seeds:
        updated.setdefault("run", {})
        updated["run"]["seeds"] = list(args.seeds)
    if args.feature_mode:
        updated.setdefault("features", {})
        updated["features"]["mode"] = args.feature_mode
    if args.selector_method:
        updated.setdefault("features", {})
        updated["features"]["selector_method"] = args.selector_method
    if args.shift is not None:
        updated.setdefault("data", {})
        updated["data"]["shift"] = int(args.shift)
    if args.save_diagnostics:
        updated.setdefault("run", {})
        updated["run"]["save_diagnostics"] = True
    if args.no_plots:
        updated.setdefault("run", {})
        updated["run"]["save_plots"] = False
    if args.expected_groups is not None:
        updated["expected_num_groups"] = int(args.expected_groups)
    if args.max_walkforward_folds is not None:
        updated.setdefault("folds", {}).setdefault("walkforward", {})
        updated["folds"]["walkforward"]["max_folds"] = int(args.max_walkforward_folds)
    return updated


def setup_logging(output_dir: Path, verbose: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_gmm_groups.log"
    handlers: List[logging.Handler] = [logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _normalize_for_filename(value: float) -> str:
    text = f"{value:.0e}" if value < 1e-3 else f"{value:g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def save_run_diagnostics(
    group_dir: Path,
    fold: FoldDefinition,
    row: Dict[str, Any],
    diagnostics: Dict[str, np.ndarray],
) -> None:
    diagnostics_dir = ensure_dir(group_dir / "diagnostics")
    prefix = (
        f"{fold.fold_id}_k{row['n_components']}_{row['covariance_type']}"
        f"_seed{row['seed']}_reg{_normalize_for_filename(float(row['reg_covar']))}"
    )

    labels_df = pd.DataFrame(
        {
            "segment": ["train"] * fold.train_size + ["test"] * fold.test_size,
            "row_index": np.concatenate([fold.train_idx, fold.test_idx]),
            "label": np.concatenate([diagnostics["train_labels"], diagnostics["test_labels"]]),
        }
    )
    labels_df.to_csv(diagnostics_dir / f"hard_labels_{prefix}.csv", index=False)

    resp_train = pd.DataFrame(diagnostics["train_responsibilities"])
    resp_train.columns = [f"r_{i}" for i in range(resp_train.shape[1])]
    resp_train.insert(0, "row_index", fold.train_idx)
    resp_train.insert(0, "segment", "train")

    resp_test = pd.DataFrame(diagnostics["test_responsibilities"])
    resp_test.columns = [f"r_{i}" for i in range(resp_test.shape[1])]
    resp_test.insert(0, "row_index", fold.test_idx)
    resp_test.insert(0, "segment", "test")
    pd.concat([resp_train, resp_test], ignore_index=True).to_csv(
        diagnostics_dir / f"responsibilities_{prefix}.csv",
        index=False,
    )

    trans = pd.DataFrame(diagnostics["transition_probs"])
    trans.columns = [f"to_{i}" for i in range(trans.shape[1])]
    trans.index = [f"from_{i}" for i in range(trans.shape[0])]
    trans.to_csv(diagnostics_dir / f"transition_matrix_{prefix}.csv", index=True)


def aggregate_summary(ledger_df: pd.DataFrame) -> pd.DataFrame:
    if ledger_df.empty:
        return pd.DataFrame()

    group_cols = [
        "group_name",
        "fold_id",
        "fold_type",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "feature_mode",
        "selector_method",
        "selected_feature_count",
        "selected_features_signature",
        "candidate_feature_count",
        "n_components",
        "covariance_type",
        "reg_covar",
        "init_params",
        "n_init",
        "max_iter",
        "tol",
    ]

    metrics_for_agg = [
        "train_avg_loglik",
        "test_avg_loglik",
        "aic_train",
        "bic_train",
        "aic_test",
        "bic_test",
        "avg_entropy_train",
        "avg_entropy_test",
        "silhouette_train",
        "silhouette_test",
        "davies_bouldin_train",
        "davies_bouldin_test",
        "avg_probmax_train",
        "avg_probmax_test",
        "p_prob_ge_0_9_train",
        "p_prob_ge_0_9_test",
        "avg_run_len",
        "median_run_len",
        "p_stay",
        "runtime_fit_seconds",
    ]

    work = ledger_df.copy()
    work["success"] = work["success"].astype(bool)
    for col in metrics_for_agg:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    success_df = work[work["success"]].copy()
    if success_df.empty:
        counts = work.groupby(group_cols, dropna=False).agg(
            total_runs=("success", "size"),
            successful_runs=("success", "sum"),
        )
        counts = counts.reset_index()
        counts["failed_runs"] = counts["total_runs"] - counts["successful_runs"]
        return counts

    agg_df = success_df.groupby(group_cols, dropna=False).agg({m: ["mean", "std"] for m in metrics_for_agg})
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    counts = work.groupby(group_cols, dropna=False).agg(
        total_runs=("success", "size"),
        successful_runs=("success", "sum"),
    )
    counts = counts.reset_index()
    counts["failed_runs"] = counts["total_runs"] - counts["successful_runs"]

    merged = agg_df.merge(counts, on=group_cols, how="left")
    merged = merged.sort_values(
        by=["fold_type", "fold_id", "covariance_type", "n_components", "reg_covar"],
        kind="mergesort",
    ).reset_index(drop=True)
    return merged


def run_group(
    group_input: GroupInput,
    config: Dict[str, Any],
    output_root: Path,
) -> pd.DataFrame:
    data_cfg: Dict[str, Any] = dict(config.get("data", {}))
    features_cfg: Dict[str, Any] = dict(config.get("features", {}))
    folds_cfg: Dict[str, Any] = dict(config.get("folds", {}))
    gmm_cfg: Dict[str, Any] = dict(config.get("gmm", {}))
    run_cfg: Dict[str, Any] = dict(config.get("run", {}))

    group_dir = ensure_dir(output_root / group_input.group_name)
    logging.info("Running group '%s' from %s", group_input.group_name, group_input.csv_path)

    df, timestamp_col = load_group_dataframe(
        csv_path=group_input.csv_path,
        timestamp_candidates=list(data_cfg.get("timestamp_candidates", ["timestamp", "time", "date", "datetime"])),
        sort_by_timestamp=bool(data_cfg.get("sort_by_timestamp", True)),
    )
    if df.empty:
        raise ValueError(f"Input CSV is empty: {group_input.csv_path}")

    base_feature_columns = infer_base_feature_columns(
        df=df,
        timestamp_col=timestamp_col,
        label_column_patterns=list(
            data_cfg.get("label_column_patterns", ["label", "target", "long", "short", "skip", "signal"])
        ),
        drop_ohlcv_columns=bool(data_cfg.get("drop_ohlcv_columns", True)),
        ohlcv_columns=list(data_cfg.get("ohlcv_columns", ["open", "high", "low", "close", "volume"])),
    )
    if not base_feature_columns:
        raise ValueError(f"No usable numeric indicator features found in {group_input.csv_path.name}.")

    shift = int(data_cfg.get("shift", 0))
    df = apply_feature_shift(df, base_feature_columns, shift)

    # Remove rows where all candidate features are missing after optional shift.
    feature_frame = df.loc[:, base_feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = ~feature_frame.isna().all(axis=1)
    dropped_rows = int((~valid_mask).sum())
    if dropped_rows > 0:
        logging.info("Dropped %d rows with all-NaN features after shift for group '%s'.", dropped_rows, group_input.group_name)
        df = df.loc[valid_mask].reset_index(drop=True)

    min_samples = int(data_cfg.get("min_samples_after_cleaning", 50))
    if df.shape[0] < min_samples:
        raise ValueError(f"Too few samples after cleaning ({df.shape[0]}). Minimum required: {min_samples}.")

    folds = build_folds(df=df, timestamp_col=timestamp_col, fold_cfg=folds_cfg)
    logging.info("Generated %d folds for group '%s'.", len(folds), group_input.group_name)

    selected_features_by_fold: Dict[str, List[str]] = {}
    ledger_rows: List[Dict[str, Any]] = []

    seeds: Sequence[int] = [int(s) for s in run_cfg.get("seeds", [1, 2, 3, 4, 5])]
    k_values: Sequence[int] = [int(k) for k in gmm_cfg.get("n_components", [2, 3, 4, 5, 6, 7, 8])]
    cov_types: Sequence[str] = [str(c).lower() for c in gmm_cfg.get("covariance_types", ["tied", "full"])]
    reg_covars: Sequence[float] = [float(v) for v in gmm_cfg.get("reg_covar", [1e-6, 1e-5])]

    for fold in folds:
        fold_meta = {
            "fold_id": fold.fold_id,
            "fold_type": fold.fold_type,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_size": fold.train_size,
            "test_size": fold.test_size,
        }
        try:
            fold_data = prepare_fold_features(
                df=df,
                fold=fold,
                base_feature_columns=base_feature_columns,
                data_cfg=data_cfg,
                features_cfg=features_cfg,
            )
        except Exception as exc:
            logging.exception(
                "Feature preparation failed for group '%s', fold '%s': %s",
                group_input.group_name,
                fold.fold_id,
                exc,
            )
            continue

        selected_features_by_fold[fold.fold_id] = fold_data.selected_features
        feature_meta = {
            "feature_mode": str(features_cfg.get("mode", "selected_top10")),
            "selector_method": str(features_cfg.get("selector_method", "variance_prune")),
            "selected_feature_count": len(fold_data.selected_features),
            "selected_features": json.dumps(fold_data.selected_features),
            "selected_features_signature": "|".join(fold_data.selected_features),
            "candidate_feature_count": len(fold_data.candidate_features),
            "dropped_missing_features_count": len(fold_data.dropped_missing_features),
            "dropped_low_variance_features_count": len(fold_data.dropped_low_variance_features),
        }

        for k in k_values:
            for cov_type in cov_types:
                if cov_type not in {"tied", "full"}:
                    logging.warning("Skipping unsupported covariance_type '%s'.", cov_type)
                    continue
                for reg_covar in reg_covars:
                    for seed in seeds:
                        outcome = run_single_gmm(
                            group_name=group_input.group_name,
                            fold_meta=fold_meta,
                            feature_meta=feature_meta,
                            X_train=fold_data.train_matrix,
                            X_test=fold_data.test_matrix,
                            k=int(k),
                            covariance_type=cov_type,
                            seed=int(seed),
                            reg_covar=float(reg_covar),
                            gmm_cfg=gmm_cfg,
                        )
                        ledger_rows.append(outcome.ledger_row)
                        if bool(run_cfg.get("save_diagnostics", False)) and outcome.diagnostics is not None:
                            save_run_diagnostics(
                                group_dir=group_dir,
                                fold=fold,
                                row=outcome.ledger_row,
                                diagnostics=outcome.diagnostics,
                            )

    selected_payload = {
        "group_name": group_input.group_name,
        "source_csv": str(group_input.csv_path),
        "timestamp_column": timestamp_col,
        "shift": shift,
        "feature_mode": str(features_cfg.get("mode", "selected_top10")),
        "selector_method": str(features_cfg.get("selector_method", "variance_prune")),
        "base_feature_columns": list(base_feature_columns),
        "selected_features_by_fold": selected_features_by_fold,
    }
    write_json(group_dir / "selected_features.json", selected_payload)

    ledger_df = pd.DataFrame(ledger_rows)
    ledger_path = group_dir / "ledger.csv"
    ledger_df.to_csv(ledger_path, index=False)

    summary_df = aggregate_summary(ledger_df)
    summary_path = group_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ranking_cfg: Dict[str, Any] = dict(config.get("ranking", {}))
    top10_df = rank_top_configs(summary_df, ranking_cfg=ranking_cfg)
    top10_path = group_dir / "top10.csv"
    top10_df.to_csv(top10_path, index=False)

    if bool(run_cfg.get("save_plots", True)):
        save_group_plots(summary_df, group_dir / "plots")

    logging.info(
        "Finished group '%s': ledger=%s rows, summary=%s rows, top10=%s rows",
        group_input.group_name,
        len(ledger_df),
        len(summary_df),
        len(top10_df),
    )
    return top10_df


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_config = load_yaml_config(config_path)
    config = apply_cli_overrides(base_config, args)

    root_folder = resolve_repo_path(str(config.get("root_group_folder", "data/external/1d")))
    output_root = ensure_dir(resolve_repo_path(str(config.get("output_dir", "pipeline/step_gmm_groups/results"))))
    setup_logging(output_root, verbose=bool(config.get("run", {}).get("verbose", True)))

    logging.info("Using root group folder: %s", root_folder)
    logging.info("Writing outputs to: %s", output_root)

    groups = discover_group_csvs(
        root=root_folder,
        expected_num_groups=config.get("expected_num_groups", 6),
    )
    logging.info("Discovered %d groups.", len(groups))

    all_top10: List[pd.DataFrame] = []
    for group in groups:
        try:
            top_df = run_group(group, config=config, output_root=output_root)
            if not top_df.empty:
                top_df = top_df.copy()
                if "group_name" in top_df.columns:
                    top_df["group_name"] = group.group_name
                else:
                    top_df.insert(0, "group_name", group.group_name)
                all_top10.append(top_df)
        except Exception:
            logging.exception("Group '%s' failed. Continuing with remaining groups.", group.group_name)

    if all_top10:
        all_top10_df = pd.concat(all_top10, axis=0, ignore_index=True)
    else:
        all_top10_df = pd.DataFrame()
    all_top10_df.to_csv(output_root / "ALL_groups_top10.csv", index=False)

    logging.info("Pipeline complete. ALL_groups_top10.csv rows: %d", len(all_top10_df))


if __name__ == "__main__":
    main()
