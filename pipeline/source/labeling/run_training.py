from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import (
    ALLOWED_LABELS,
    OUTPUT_1H_BASELINE,
    OUTPUT_1H_WITH5M,
    PathsConfig,
    TrainingConfig,
)
from .evaluate import evaluate_predictions, save_confusion_matrix_csv, save_predictions_csv
from .features import build_features_and_labels
from .label_cleaning import apply_high_precision_cleaning
from .label_quality import print_label_counts
from .models import ModelImportError, build_model
from .splits import sort_by_time, time_holdout_split, time_series_splits, train_val_split_indices
from .thresholding import grid_search_thresholds, route_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate models on labeled BTCUSDT data.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=OUTPUT_1H_WITH5M,
        help="Labeled CSV to train on (baseline or with5m).",
    )
    parser.add_argument(
        "--label-source",
        choices=["baseline", "with5m"],
        default=None,
        help="Shortcut to select baseline or with5m CSV.",
    )
    parser.add_argument("--clean", action="store_true", help="Enable high-precision label cleaning")
    parser.add_argument("--min-range-pct", type=float, default=None)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    parser.add_argument("--test-size", type=float, default=TrainingConfig().test_size)
    parser.add_argument("--cv-splits", type=int, default=TrainingConfig().cv_splits)
    parser.add_argument("--min-trade-coverage", type=float, default=TrainingConfig().min_trade_coverage)
    parser.add_argument("--threshold-grid", type=str, default=None)
    parser.add_argument("--val-size", type=float, default=TrainingConfig().val_size)
    return parser.parse_args()


def _resolve_input_csv(args: argparse.Namespace) -> Path:
    if args.label_source == "baseline":
        return OUTPUT_1H_BASELINE
    if args.label_source == "with5m":
        return OUTPUT_1H_WITH5M
    return args.input_csv


def _parse_threshold_grid(raw: str | None, default: List[float]) -> List[float]:
    if raw is None:
        return default
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        return default
    return values


def _encode_labels(y: pd.Series, class_order: List[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(class_order)}
    if not set(y.unique()).issubset(mapping):
        missing = set(y.unique()) - set(mapping)
        raise ValueError(f"Unexpected labels found: {missing}")
    return y.map(mapping).astype(int).to_numpy()


def _make_artifacts_dirs(paths: PathsConfig) -> None:
    paths.artifacts_labeling.mkdir(parents=True, exist_ok=True)
    paths.artifacts_models.mkdir(parents=True, exist_ok=True)
    paths.artifacts_reports.mkdir(parents=True, exist_ok=True)


def _save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_metrics_csv(rows: List[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        min_trade_coverage=args.min_trade_coverage,
        threshold_grid=_parse_threshold_grid(args.threshold_grid, list(TrainingConfig().threshold_grid)),
        val_size=args.val_size,
    )
    paths = PathsConfig()
    _make_artifacts_dirs(paths)

    input_csv = _resolve_input_csv(args)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    np.random.seed(config.seed)

    df = pd.read_csv(input_csv, low_memory=False)
    df = sort_by_time(df)

    if "candle_type" not in df.columns:
        raise ValueError("Input CSV missing candle_type column.")

    if args.clean:
        print_label_counts("Before cleaning", df["candle_type"], ALLOWED_LABELS)
        df, stats = apply_high_precision_cleaning(df, min_range_pct=args.min_range_pct)
        print_label_counts("After cleaning", df["candle_type"], ALLOWED_LABELS)
        print(f"Cleaning changed {stats['changed_rows']} rows.")

    X_df, y_series, feature_cols = build_features_and_labels(df)
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns detected after filtering. "
            "Check input data types or provide numeric feature columns."
        )
    class_order = list(ALLOWED_LABELS)
    y_encoded = _encode_labels(y_series, class_order)

    model_names = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else list(config.model_names)
    )

    artifacts_snapshot = {
        "input_csv": str(input_csv),
        "model_names": model_names,
        "class_order": class_order,
        "clean": args.clean,
        "min_range_pct": args.min_range_pct,
        "config": config.__dict__,
        "feature_count": len(feature_cols),
    }
    _save_json(artifacts_snapshot, paths.artifacts_reports / "config_snapshot.json")
    _save_json({"features": feature_cols}, paths.artifacts_reports / "features.json")

    metrics_rows: List[dict] = []

    n_samples = len(df)
    train_idx, test_idx = time_holdout_split(n_samples, config.test_size)
    train_idx_inner, val_idx_inner = train_val_split_indices(train_idx, config.val_size)

    X_train = X_df.iloc[train_idx_inner].to_numpy()
    y_train = y_encoded[train_idx_inner]
    X_val = X_df.iloc[val_idx_inner].to_numpy()
    y_val = y_series.iloc[val_idx_inner].tolist()
    X_test = X_df.iloc[test_idx].to_numpy()
    y_test = y_series.iloc[test_idx].tolist()

    imputer = SimpleImputer(strategy=config.imputer_strategy)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    for model_name in model_names:
        try:
            spec = build_model(model_name, X_train, y_train, class_order, config.seed)
        except ModelImportError as exc:
            print(str(exc))
            continue

        proba_val = spec.model.predict_proba(X_val)
        best_thresholds = grid_search_thresholds(
            proba_val,
            y_val,
            class_order,
            config.threshold_grid,
            config.min_trade_coverage,
        )

        proba_test = spec.model.predict_proba(X_test)
        preds_test = route_predictions(
            proba_test,
            class_order,
            best_thresholds.threshold_long,
            best_thresholds.threshold_short,
        )
        eval_result = evaluate_predictions(y_test, preds_test, class_order)

        metrics_rows.append(
            {
                "model": model_name,
                "split": "holdout",
                "precision_long": eval_result.precision_long,
                "precision_short": eval_result.precision_short,
                "coverage": eval_result.coverage,
                "macro_f1": eval_result.macro_f1,
                "balanced_accuracy": eval_result.balanced_accuracy,
                "threshold_long": best_thresholds.threshold_long,
                "threshold_short": best_thresholds.threshold_short,
            }
        )

        model_path = paths.artifacts_models / f"{model_name}_holdout.joblib"
        joblib.dump(spec.model, model_path)

        thresholds_path = paths.artifacts_reports / f"{model_name}_thresholds.json"
        _save_json(best_thresholds.__dict__, thresholds_path)

        confusion_path = paths.artifacts_reports / f"{model_name}_confusion_holdout.csv"
        save_confusion_matrix_csv(eval_result.confusion, class_order, str(confusion_path))

        report_path = paths.artifacts_reports / f"{model_name}_report_holdout.json"
        _save_json(eval_result.report, report_path)

        predictions_df = df.iloc[test_idx].copy()
        predictions_df["predicted_label"] = preds_test
        predictions_path = paths.artifacts_reports / f"{model_name}_predictions_holdout.csv"
        save_predictions_csv(predictions_df, proba_test, class_order, str(predictions_path))

        print(
            f"{model_name} holdout precision_long={eval_result.precision_long:.3f}, "
            f"precision_short={eval_result.precision_short:.3f}, coverage={eval_result.coverage:.3f}"
        )

    splits = time_series_splits(n_samples, config.cv_splits)
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx_inner, val_idx_inner = train_val_split_indices(train_idx, config.val_size)

        X_train = X_df.iloc[train_idx_inner].to_numpy()
        y_train = y_encoded[train_idx_inner]
        X_val = X_df.iloc[val_idx_inner].to_numpy()
        y_val = y_series.iloc[val_idx_inner].tolist()
        X_test = X_df.iloc[test_idx].to_numpy()
        y_test = y_series.iloc[test_idx].tolist()

        imputer = SimpleImputer(strategy=config.imputer_strategy)
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        for model_name in model_names:
            try:
                spec = build_model(model_name, X_train, y_train, class_order, config.seed)
            except ModelImportError as exc:
                print(str(exc))
                continue

            proba_val = spec.model.predict_proba(X_val)
            best_thresholds = grid_search_thresholds(
                proba_val,
                y_val,
                class_order,
                config.threshold_grid,
                config.min_trade_coverage,
            )

            proba_test = spec.model.predict_proba(X_test)
            preds_test = route_predictions(
                proba_test,
                class_order,
                best_thresholds.threshold_long,
                best_thresholds.threshold_short,
            )
            eval_result = evaluate_predictions(y_test, preds_test, class_order)

            metrics_rows.append(
                {
                    "model": model_name,
                    "split": f"cv_{split_id}",
                    "precision_long": eval_result.precision_long,
                    "precision_short": eval_result.precision_short,
                    "coverage": eval_result.coverage,
                    "macro_f1": eval_result.macro_f1,
                    "balanced_accuracy": eval_result.balanced_accuracy,
                    "threshold_long": best_thresholds.threshold_long,
                    "threshold_short": best_thresholds.threshold_short,
                }
            )

    metrics_path = paths.artifacts_reports / "metrics_summary.csv"
    _save_metrics_csv(metrics_rows, metrics_path)


if __name__ == "__main__":
    main()
