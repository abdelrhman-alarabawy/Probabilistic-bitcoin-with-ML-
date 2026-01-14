from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import (  # noqa: E402
    build_quality_report,
    detect_ohlcv_columns,
    fit_transform_features,
    load_raw_csv,
    save_quality_report,
    select_numeric_features,
)
from hmm import compute_transition_matrix, decode_states, fit_candidates, select_best_candidate  # noqa: E402
from labeling import LabelingParams, label_candles, label_distribution, label_transition_matrix  # noqa: E402
from mapping import compute_state_stats, map_states_to_regimes  # noqa: E402
from plots import plot_boxplots, plot_heatmap, plot_price_regimes, plot_state_occupancy  # noqa: E402
from train_models import save_confusion_plot, save_model, train_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="12h regime discovery pipeline.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=str(BASE_DIR),
        help="Output root directory.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--hmm-states",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8],
        help="Candidate HMM state counts.",
    )
    parser.add_argument(
        "--target-regimes",
        type=int,
        default=3,
        help="Target final regime count.",
    )
    parser.add_argument(
        "--tp-points",
        type=float,
        default=2000.0,
        help="TP points (1h baseline, will scale to 12h).",
    )
    parser.add_argument(
        "--sl-points",
        type=float,
        default=1000.0,
        help="SL points (1h baseline, will scale to 12h).",
    )
    parser.add_argument(
        "--base-horizon-minutes",
        type=int,
        default=60,
        help="Base horizon in minutes for 1h; scaled by 12 for 12h labeling.",
    )
    parser.add_argument(
        "--train-global",
        action="store_true",
        help="Train a global model across all regimes.",
    )
    return parser.parse_args()


def chronological_splits(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "output": output_root / "output",
        "models": output_root / "models",
        "plots": output_root / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    dirs = ensure_dirs(output_root)

    print(f"Loading data: {input_path}")
    df, timestamp_col = load_raw_csv(input_path)
    ohlcv_cols = detect_ohlcv_columns(df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    quality_report = build_quality_report(df, timestamp_col, numeric_cols)
    save_quality_report(quality_report, dirs["output"] / "data_quality.json")

    exclude_for_features = list(ohlcv_cols.values())
    if timestamp_col:
        exclude_for_features.append(timestamp_col)
    feature_cols = select_numeric_features(df, timestamp_col, exclude_for_features)

    if len(feature_cols) == 0:
        raise ValueError("No numeric indicator columns available for modeling.")

    train_idx, val_idx, test_idx = chronological_splits(len(df))
    X_full, artifacts = fit_transform_features(df, feature_cols, train_idx)

    print("Fitting HMM candidates...")
    candidates = fit_candidates(
        X_full[train_idx],
        X_full[val_idx],
        n_states_list=args.hmm_states,
        random_state=args.random_state,
    )
    best = select_best_candidate(candidates)

    candidates_rows = []
    for res in candidates:
        candidates_rows.append(
            {
                "n_states": res.n_states,
                "val_loglik": res.val_loglik,
                "train_loglik": res.train_loglik,
                "aic": res.aic,
                "bic": res.bic,
                "min_state_occupancy": float(res.occupancy.min()) if len(res.occupancy) else 0.0,
                "avg_duration": res.avg_duration,
                "transition_entropy": res.transition_entropy,
                "transition_sparsity": res.transition_sparsity,
                "notes": res.notes,
            }
        )
    pd.DataFrame(candidates_rows).to_csv(dirs["output"] / "hmm_candidates.csv", index=False)

    print(f"Selected HMM states: {best.n_states} ({best.notes})")
    states = decode_states(best.model, X_full)
    df["HMM_state"] = states

    trans = compute_transition_matrix(states, best.n_states)
    trans_df = pd.DataFrame(trans, index=[f"s{i}" for i in range(best.n_states)])
    trans_df.to_csv(dirs["output"] / "hmm_transition_matrix.csv")

    stats_df = compute_state_stats(
        df,
        state_col="HMM_state",
        close_col=ohlcv_cols["close"],
        high_col=ohlcv_cols["high"],
        low_col=ohlcv_cols["low"],
    )
    mapping_result = map_states_to_regimes(
        stats_df,
        target_k=args.target_regimes,
        random_state=args.random_state,
    )

    df["final_regime"] = df["HMM_state"].map(mapping_result.mapping)

    regime_transition = pd.crosstab(
        df["final_regime"].shift(1),
        df["final_regime"],
        dropna=False,
        normalize="index",
    )
    regime_transition.to_csv(dirs["output"] / "final_regime_transition_matrix.csv")

    combined_path = dirs["output"] / "combined_with_states.csv"
    df.to_csv(combined_path, index=False)

    regime_stats = mapping_result.regime_stats
    regime_stats.to_csv(dirs["output"] / "hmm_state_stats.csv", index=False)

    regime_names = (
        df.groupby("final_regime")[ohlcv_cols["close"]]
        .apply(lambda s: s.pct_change().std())
        .sort_values()
        .index.tolist()
    )

    label_params = LabelingParams(
        base_horizon_minutes=args.base_horizon_minutes,
        tp_points=args.tp_points,
        sl_points=args.sl_points,
    )

    label_summary = {}
    regime_files = []
    for idx, regime_name in enumerate(regime_names):
        regime_df = df[df["final_regime"] == regime_name].copy()
        regime_label = f"regime_{idx}_{regime_name}"
        out_path = dirs["output"] / f"{regime_label}.csv"
        regime_df.to_csv(out_path, index=False)
        regime_files.append(out_path)

        labels, ambiguous = label_candles(
            regime_df,
            open_col=ohlcv_cols["open"],
            high_col=ohlcv_cols["high"],
            low_col=ohlcv_cols["low"],
            params=label_params,
        )
        regime_df["candle_type"] = labels
        regime_df["label_ambiguous"] = ambiguous
        df.loc[regime_df.index, "candle_type"] = labels
        df.loc[regime_df.index, "label_ambiguous"] = ambiguous

        labeled_path = dirs["output"] / f"{regime_label}_labeled.csv"
        regime_df.to_csv(labeled_path, index=False)

        label_dist = label_distribution(labels)
        label_trans = label_transition_matrix(labels)
        label_trans.to_csv(dirs["output"] / f"{regime_label}_label_transition.csv")
        label_summary[regime_label] = {
            "distribution": label_dist,
            "rows": len(regime_df),
        }

        model_dir = dirs["models"] / regime_label
        exclude_cols = list(ohlcv_cols.values()) + [
            "HMM_state",
            "final_regime",
            "candle_type",
            "label_ambiguous",
        ]
        if timestamp_col:
            exclude_cols.append(timestamp_col)
        model_result = train_classifier(
            regime_df,
            label_col="candle_type",
            exclude_cols=exclude_cols,
            output_dir=model_dir,
            random_state=args.random_state,
        )
        if model_result:
            save_model(model_result, model_dir)
            cm_path = dirs["plots"] / f"{regime_label}_confusion.png"
            save_confusion_plot(model_result.confusion, model_result.label_encoder.classes_.tolist(), cm_path)

    combined_labeled_path = dirs["output"] / "combined_with_states_labeled.csv"
    df.to_csv(combined_labeled_path, index=False)

    if args.train_global:
        global_label = "global_model"
        model_dir = dirs["models"] / global_label
        exclude_cols = list(ohlcv_cols.values()) + [
            "HMM_state",
            "final_regime",
            "candle_type",
            "label_ambiguous",
        ]
        if timestamp_col:
            exclude_cols.append(timestamp_col)
        model_result = train_classifier(
            df,
            label_col="candle_type",
            exclude_cols=exclude_cols,
            output_dir=model_dir,
            random_state=args.random_state,
        )
        if model_result:
            save_model(model_result, model_dir)
            cm_path = dirs["plots"] / f"{global_label}_confusion.png"
            save_confusion_plot(model_result.confusion, model_result.label_encoder.classes_.tolist(), cm_path)

    print("Generating plots...")
    plot_price_regimes(
        df,
        timestamp_col=timestamp_col,
        close_col=ohlcv_cols["close"],
        regime_col="final_regime",
        output_path=dirs["plots"] / "price_by_regime.png",
    )
    plot_state_occupancy(
        df["HMM_state"].value_counts(normalize=True),
        df["final_regime"].value_counts(normalize=True),
        dirs["plots"] / "state_occupancy.png",
    )
    plot_heatmap(
        trans_df,
        "HMM Transition Matrix",
        dirs["plots"] / "hmm_transition_heatmap.png",
    )
    plot_heatmap(
        regime_transition,
        "Final Regime Transition Matrix",
        dirs["plots"] / "final_regime_transition_heatmap.png",
    )

    returns = df[ohlcv_cols["close"]].pct_change().fillna(0.0)
    vol = returns.rolling(window=10, min_periods=2).std().fillna(0.0)
    range_pct = (df[ohlcv_cols["high"]] - df[ohlcv_cols["low"]]) / df[ohlcv_cols["close"]].replace(0, np.nan)
    range_pct = range_pct.fillna(0.0)
    plot_df = df[["final_regime"]].copy()
    plot_df["returns"] = returns
    plot_df["volatility"] = vol
    plot_df["range_pct"] = range_pct
    plot_boxplots(plot_df, "final_regime", dirs["plots"] / "regime_boxplots.png")

    report_path = output_root / "REPORT.md"
    report = {
        "input": str(input_path),
        "rows": len(df),
        "columns": df.shape[1],
        "timestamp_col": timestamp_col,
        "hmm_selected": {
            "n_states": best.n_states,
            "method": best.notes,
        },
        "final_regime_count": len(regime_names),
        "label_summary": label_summary,
        "outputs": {
            "combined": str(combined_path),
            "combined_labeled": str(combined_labeled_path),
            "regime_files": [str(p) for p in regime_files],
        },
        "plots": {
            "price_by_regime": str(dirs["plots"] / "price_by_regime.png"),
            "state_occupancy": str(dirs["plots"] / "state_occupancy.png"),
            "hmm_transition": str(dirs["plots"] / "hmm_transition_heatmap.png"),
            "final_transition": str(dirs["plots"] / "final_regime_transition_heatmap.png"),
            "boxplots": str(dirs["plots"] / "regime_boxplots.png"),
        },
    }

    report_lines = [
        "# 12h Regime Discovery Report",
        "",
        "## Data Summary",
        f"- Input: `{input_path}`",
        f"- Rows: {len(df)}",
        f"- Columns: {df.shape[1]}",
        f"- Timestamp column: {timestamp_col}",
        "",
        "## HMM Selection",
        f"- Selected n_states: {best.n_states} ({best.notes})",
        "- Candidate metrics: `output/hmm_candidates.csv`",
        "",
        "## Regime Mapping",
        f"- Final regimes: {len(regime_names)}",
        "- Mapping method: k-means on state stats (volatility-ordered labels when possible).",
        "- HMM state stats: `output/hmm_state_stats.csv`",
        f"- Target regimes: {args.target_regimes}; selected: {mapping_result.k}",
        "",
        "## Labeling (12h adaptation)",
        f"- Base horizon: {label_params.base_horizon_minutes} minutes",
        f"- 12h horizon: {label_params.horizon_minutes} minutes",
        f"- TP points: {label_params.tp_points}",
        f"- SL points: {label_params.sl_points}",
        "",
        "## Per-Regime Label Distribution",
    ]
    for regime_label, summary in label_summary.items():
        report_lines.append(f"- {regime_label}: {summary['distribution']}")

    report_lines += [
        "",
        "## Outputs",
        "- Combined dataset with states: `output/combined_with_states.csv`",
        "- Combined dataset with labels: `output/combined_with_states_labeled.csv`",
        "- Regime splits: `output/regime_*.csv`",
        "- Labeled regime files: `output/regime_*_labeled.csv`",
        "",
        "## Plots",
        "- `plots/price_by_regime.png`",
        "- `plots/state_occupancy.png`",
        "- `plots/hmm_transition_heatmap.png`",
        "- `plots/final_regime_transition_heatmap.png`",
        "- `plots/regime_boxplots.png`",
        "",
        "## Notes",
        "- Chronological splits used for HMM training/validation and per-regime modeling.",
        "- If optional ML libraries are missing, fallback models are used.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    json.dump(report, (output_root / "report_summary.json").open("w", encoding="utf-8"), indent=2)

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
