#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from src.gmm_apply import fit_predict_gmm, resolve_feature_list_for_config
from src.io import (
    discover_group_csvs,
    ensure_dir,
    load_group_dataframe,
    load_selected_features_json,
    load_top10,
    normalize_ohlcv_columns,
    write_json,
)
from src.label_integration import build_group_labels
from src.merge import attach_gmm_columns, attach_labels, compute_state_label_diagnostics, select_output_columns


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary.")
    return data


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = deepcopy(config)
    if args.root:
        merged["root_group_folder"] = args.root
    if args.prev_results:
        merged["prev_results_root"] = args.prev_results
    if args.label_script:
        merged["label_script_path"] = args.label_script
    if args.out:
        merged["output_dir"] = args.out
    if args.mode:
        merged.setdefault("run", {})
        merged["run"]["mode"] = args.mode
    if args.shift is not None:
        merged.setdefault("run", {})
        merged["run"]["shift"] = int(args.shift)
    if args.top_n is not None:
        merged.setdefault("run", {})
        merged["run"]["top_n"] = int(args.top_n)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit top-10 GMM configs per group and attach trade labels.")
    parser.add_argument("--config", type=str, default=str(CURRENT_DIR / "config.yaml"))
    parser.add_argument("--root", type=str, help="Root folder containing group CSVs.")
    parser.add_argument("--prev_results", type=str, help="Previous GMM results root (contains per-group top10.csv).")
    parser.add_argument("--label_script", type=str, help="Path to signals_code_hour_v1_0.py")
    parser.add_argument("--out", type=str, help="Output root for labeling step results.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fit_train_predict_all", "fit_all_predict_all"],
        help="Model fit mode.",
    )
    parser.add_argument("--shift", type=int, help="Feature shift for GMM inputs.")
    parser.add_argument("--top_n", type=int, help="Number of top ranked configs per group.")
    return parser.parse_args()


def setup_logging(output_dir: Path, verbose: bool = True) -> None:
    ensure_dir(output_dir)
    handlers: List[logging.Handler] = [logging.FileHandler(output_dir / "run_label_top10.log", mode="w", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _safe_int(value: object, default: int) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, float) and pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, float) and pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _extract_prev_defaults(prev_step_cfg_path: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    if not prev_step_cfg_path.exists():
        return defaults
    prev_cfg = load_yaml(prev_step_cfg_path)
    defaults["scaler"] = prev_cfg.get("features", {}).get("scaler", "robust")
    gmm_cfg = prev_cfg.get("gmm", {})
    defaults["gmm"] = {
        "init_params": gmm_cfg.get("init_params", "kmeans"),
        "n_init": int(gmm_cfg.get("n_init", 5)),
        "max_iter": int(gmm_cfg.get("max_iter", 500)),
        "tol": float(gmm_cfg.get("tol", 1e-3)),
        "n_components_default": int(gmm_cfg.get("n_components", [2])[0]),
        "covariance_type_default": str(gmm_cfg.get("covariance_types", ["tied"])[0]),
        "reg_covar_default": float(gmm_cfg.get("reg_covar", [1e-6])[0]),
    }
    defaults["label_patterns"] = prev_cfg.get("data", {}).get(
        "label_column_patterns",
        ["label", "target", "long", "short", "skip", "signal"],
    )
    defaults["timestamp_candidates"] = prev_cfg.get("data", {}).get(
        "timestamp_candidates",
        ["timestamp", "time", "date", "datetime"],
    )
    return defaults


def _resolve_selected_fallback(
    selected_json: Dict[str, Any],
    fold_id: Optional[str],
) -> Optional[List[str]]:
    if not selected_json:
        return None
    by_fold = selected_json.get("selected_features_by_fold")
    if not isinstance(by_fold, dict):
        return None
    if fold_id and fold_id in by_fold and isinstance(by_fold[fold_id], list):
        return [str(x) for x in by_fold[fold_id]]
    if "holdout_80_20" in by_fold and isinstance(by_fold["holdout_80_20"], list):
        return [str(x) for x in by_fold["holdout_80_20"]]
    return None


def _save_artifacts(
    out_dir: Path,
    labeled_df: pd.DataFrame,
    gmm_config: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    labeled_df.to_csv(out_dir / "labeled.csv", index=False)
    write_json(out_dir / "gmm_config.json", gmm_config)
    write_json(out_dir / "diagnostics.json", diagnostics)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(Path(args.config).resolve()), args)

    root_group_folder = resolve_path(str(config.get("root_group_folder", "data/external/1d")))
    prev_results_root = resolve_path(str(config.get("prev_results_root", "pipeline/step_gmm_groups/results")))
    label_script_path = resolve_path(str(config.get("label_script_path", "scripts/signals_code_hour_v1_0.py")))
    output_root = ensure_dir(resolve_path(str(config.get("output_dir", "pipeline/step_gmm_groups_labeling/results"))))

    run_cfg = dict(config.get("run", {}))
    setup_logging(output_root, verbose=bool(run_cfg.get("verbose", True)))

    prev_defaults = _extract_prev_defaults(resolve_path(str(config.get("prev_step_config", "pipeline/step_gmm_groups/config.yaml"))))
    timestamp_candidates = config.get("timestamp_candidates", prev_defaults.get("timestamp_candidates", ["timestamp", "time", "date", "datetime"]))
    label_patterns = config.get("label_column_patterns", prev_defaults.get("label_patterns", ["label", "target", "long", "short", "skip", "signal"]))
    gmm_defaults = dict(prev_defaults.get("gmm", {}))
    gmm_defaults.update(dict(config.get("gmm_defaults", {})))

    model_mode = str(run_cfg.get("mode", "fit_train_predict_all"))
    shift = int(run_cfg.get("shift", 1))
    top_n = int(run_cfg.get("top_n", 10))
    holdout_train_ratio = float(run_cfg.get("holdout_train_ratio", 0.8))
    include_selected_features = bool(run_cfg.get("include_selected_features", True))
    scaler_name = str(run_cfg.get("scaler", prev_defaults.get("scaler", "robust")))
    label_mode = str(run_cfg.get("label_output_mode", "no5m")).lower()

    logging.info("Root groups: %s", root_group_folder)
    logging.info("Prev results: %s", prev_results_root)
    logging.info("Label script: %s", label_script_path)
    logging.info("Output root: %s", output_root)
    logging.info("Mode=%s shift=%d top_n=%d scaler=%s label_mode=%s", model_mode, shift, top_n, scaler_name, label_mode)

    groups = discover_group_csvs(root_group_folder, expected_groups=config.get("expected_num_groups", 6))
    group_map = {g.group_name: g for g in groups}

    combined_records: List[Dict[str, Any]] = []
    for group_name, group_file in sorted(group_map.items()):
        top10_path = prev_results_root / group_name / "top10.csv"
        if not top10_path.exists():
            logging.warning("Skipping group '%s': missing %s", group_name, top10_path)
            continue

        top10_df = load_top10(top10_path, top_n=top_n)
        if top10_df.empty:
            logging.warning("Skipping group '%s': empty top10.csv", group_name)
            continue

        group_df_raw, ts_col = load_group_dataframe(group_file.csv_path, timestamp_candidates=timestamp_candidates)
        group_df, ohlcv_map = normalize_ohlcv_columns(group_df_raw)

        try:
            labels_df, label_diag = build_group_labels(
                df=group_df,
                timestamp_col=ts_col,
                script_path=label_script_path,
                label_mode=label_mode,
            )
        except Exception as exc:
            labels_df = None
            label_diag = {
                "labeling_enabled": False,
                "label_warning": f"Label script failed: {exc}",
            }
            logging.warning("Group '%s' labeling failed, continuing without labels: %s", group_name, exc)
        if not bool(label_diag.get("labeling_enabled", False)):
            logging.warning("Group '%s': %s", group_name, label_diag.get("label_warning", "labeling skipped"))

        selected_json = load_selected_features_json(prev_results_root / group_name / "selected_features.json")

        group_out = ensure_dir(output_root / group_name)
        for i, (_, row) in enumerate(top10_df.iterrows(), start=1):
            rank_name = f"model_rank{i:02d}"
            rank_out = group_out / rank_name

            fallback_features = _resolve_selected_fallback(selected_json, fold_id=str(row.get("fold_id", "")))
            feature_cols = resolve_feature_list_for_config(
                df=group_df,
                config_row=row,
                selected_features_fallback=fallback_features,
                timestamp_col=ts_col,
                label_patterns=label_patterns,
            )
            if not feature_cols:
                logging.warning("Group '%s' rank %02d has no valid features. Skipping.", group_name, i)
                continue

            try:
                fit = fit_predict_gmm(
                    df=group_df,
                    config_row=row,
                    feature_cols=feature_cols,
                    mode=model_mode,
                    shift=shift,
                    scaler_name=scaler_name,
                    holdout_train_ratio=holdout_train_ratio,
                    timestamp_col=ts_col,
                    gmm_defaults=gmm_defaults,
                    random_state_fallback=42 + i,
                )
                merged = attach_gmm_columns(
                    df=group_df,
                    responsibilities=fit.responsibilities,
                    hard_states=fit.hard_states,
                    probmax=fit.probmax,
                    entropy=fit.entropy,
                )
                merged = attach_labels(merged, labels_df=labels_df, timestamp_col=ts_col)
                output_df = select_output_columns(
                    df=merged,
                    timestamp_col=ts_col,
                    include_selected_features=include_selected_features,
                    selected_features=fit.feature_columns_used,
                )

                diag = compute_state_label_diagnostics(output_df)
                diag.update(
                    {
                        "group_name": group_name,
                        "model_rank": i,
                        "labeling_enabled": bool(label_diag.get("labeling_enabled", False)),
                        "label_warning": label_diag.get("label_warning", ""),
                        "training_rows": fit.training_rows,
                        "prediction_rows": fit.prediction_rows,
                        "fit_converged": fit.fit_converged,
                        "fit_n_iter": fit.fit_n_iter,
                        "train_start_effective": fit.train_start,
                        "train_end_effective": fit.train_end,
                        "shift_features": shift,
                        "scaler": scaler_name,
                        "feature_columns_used": fit.feature_columns_used,
                    }
                )
                for key in ["label_script_returncode", "label_script_stdout_tail", "label_script_stderr_tail", "label_mode_selected"]:
                    if key in label_diag:
                        diag[key] = label_diag[key]

                gmm_config = {
                    "group_name": group_name,
                    "source_csv": str(group_file.csv_path),
                    "model_rank": i,
                    "fit_mode": model_mode,
                    "shift_features": shift,
                    "n_components": _safe_int(row.get("n_components"), int(gmm_defaults.get("n_components_default", 2))),
                    "covariance_type": str(row.get("covariance_type", gmm_defaults.get("covariance_type_default", "tied"))),
                    "reg_covar": _safe_float(row.get("reg_covar"), float(gmm_defaults.get("reg_covar_default", 1e-6))),
                    "init_params": str(row.get("init_params", gmm_defaults.get("init_params", "kmeans"))),
                    "n_init": _safe_int(row.get("n_init"), int(gmm_defaults.get("n_init", 5))),
                    "max_iter": _safe_int(row.get("max_iter"), int(gmm_defaults.get("max_iter", 500))),
                    "tol": _safe_float(row.get("tol"), float(gmm_defaults.get("tol", 1e-3))),
                    "seed_used": _safe_int(row.get("seed"), 42 + i),
                    "fold_id_from_top10": str(row.get("fold_id", "")),
                    "train_start_from_top10": str(row.get("train_start", "")),
                    "train_end_from_top10": str(row.get("train_end", "")),
                    "selected_features_used": fit.feature_columns_used,
                }

                _save_artifacts(rank_out, output_df, gmm_config=gmm_config, diagnostics=diag)
                combined_records.append(
                    {
                        "group_name": group_name,
                        "model_rank": i,
                        "fold_id": str(row.get("fold_id", "")),
                        "n_components": gmm_config["n_components"],
                        "covariance_type": gmm_config["covariance_type"],
                        "reg_covar": gmm_config["reg_covar"],
                        "fit_mode": model_mode,
                        "labeled_csv_path": str((rank_out / "labeled.csv").resolve()),
                        "gmm_config_path": str((rank_out / "gmm_config.json").resolve()),
                        "diagnostics_path": str((rank_out / "diagnostics.json").resolve()),
                    }
                )
                logging.info("Saved %s %s", group_name, rank_name)
            except Exception as exc:
                logging.exception("Failed %s rank %02d: %s", group_name, i, exc)

    all_dir = ensure_dir(output_root / "ALL")
    combined_df = pd.DataFrame(combined_records)
    combined_df.to_csv(all_dir / "combined_index.csv", index=False)
    logging.info("Done. Combined outputs: %d", len(combined_df))


if __name__ == "__main__":
    main()
