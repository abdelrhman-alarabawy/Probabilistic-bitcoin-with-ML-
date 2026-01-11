from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def write_report(
    output_path: Path,
    dataset_summary: Dict[str, object],
    label_stats: Dict[str, object],
    split_table: pd.DataFrame,
    trade_results: List[Dict[str, object]],
    direction_results: List[Dict[str, object]],
    multiclass_summary: Dict[str, object],
    threshold_summary: Dict[str, object],
    notes: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# BetaTest Report")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append(f"- Rows: {dataset_summary['rows']}")
    lines.append(f"- Time range: {dataset_summary['start']} to {dataset_summary['end']}")
    lines.append(f"- Timestamp column: {dataset_summary['time_col']}")
    lines.append(f"- Dropped duplicate timestamps: {dataset_summary['dropped_duplicates']}")
    lines.append("")

    lines.append("## Label Diagnostics")
    lines.append(
        f"- Counts: long={label_stats.get('count_long', 0)}, "
        f"short={label_stats.get('count_short', 0)}, skip={label_stats.get('count_skip', 0)}"
    )
    lines.append(
        f"- Percentages: long={label_stats.get('pct_long', 0):.3f}, "
        f"short={label_stats.get('pct_short', 0):.3f}, skip={label_stats.get('pct_skip', 0):.3f}"
    )
    lines.append(
        "- Skip run length mean/median/p95: "
        f"{label_stats.get('skip_run_mean', 0):.2f} / "
        f"{label_stats.get('skip_run_median', 0):.2f} / "
        f"{label_stats.get('skip_run_p95', 0):.2f}"
    )
    lines.append(
        "- Rolling drift L1 mean/median/p95: "
        f"{label_stats.get('drift_l1_mean', 0):.3f} / "
        f"{label_stats.get('drift_l1_median', 0):.3f} / "
        f"{label_stats.get('drift_l1_p95', 0):.3f}"
    )
    if "ambiguous_true" in label_stats:
        lines.append(
            f"- Ambiguous true count: {label_stats.get('ambiguous_true')} "
            f"({label_stats.get('ambiguous_true_pct', 0):.3f})"
        )
    lines.append("")

    lines.append("## Split Diagnostics")
    lines.append("See split table in artifacts for full details.")
    if not split_table.empty:
        row = split_table.iloc[0]
        lines.append(
            f"- Holdout train: {row['train_start']} to {row['train_end']}; "
            f"test: {row['test_start']} to {row['test_end']}"
        )
    lines.append("")

    lines.append("## Trade vs Skip (Binary) Results")
    for result in trade_results:
        lines.append(
            f"- {result['model']}: precision={result['precision_trade']:.3f}, "
            f"recall={result['recall_trade']:.3f}, f1={result['f1_trade']:.3f}, "
            f"auc={result.get('auc_trade')}"
        )
    lines.append("")

    lines.append("## Direction on Trades (Binary) Results")
    for result in direction_results:
        lines.append(
            f"- {result['model']}: precision_long={result['precision_long']:.3f}, "
            f"precision_short={result['precision_short']:.3f}"
        )
    lines.append("")

    lines.append("## Multiclass Results")
    lines.append(
        f"- Model: {multiclass_summary.get('model')}, "
        f"macro_f1={multiclass_summary.get('macro_f1', 0):.3f}"
    )
    lines.append("")

    lines.append("## Threshold Diagnostics")
    lines.append(
        f"- Best thresholds: long={threshold_summary.get('th_long')}, "
        f"short={threshold_summary.get('th_short')}, coverage={threshold_summary.get('coverage'):.3f}"
    )
    lines.append(
        f"- precision_long={threshold_summary.get('precision_long'):.3f}, "
        f"precision_short={threshold_summary.get('precision_short'):.3f}, "
        f"trade_precision={threshold_summary.get('trade_precision'):.3f}"
    )
    lines.append("")

    lines.append("## Interpretation Hints")
    for note in notes:
        lines.append(f"- {note}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_v2(
    output_path: Path,
    dataset_summary: Dict[str, object],
    split_table: pd.DataFrame,
    settings_results: List[Dict[str, object]],
    notes: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# BetaTest Report v2")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append(f"- Rows: {dataset_summary['rows']}")
    lines.append(f"- Time range: {dataset_summary['start']} to {dataset_summary['end']}")
    lines.append(f"- Timestamp column: {dataset_summary['time_col']}")
    lines.append(f"- Dropped duplicate timestamps: {dataset_summary['dropped_duplicates']}")
    lines.append("")

    lines.append("## Split Diagnostics")
    if not split_table.empty:
        row = split_table.iloc[0]
        lines.append(
            f"- Holdout train: {row['train_start']} to {row['train_end']}; "
            f"test: {row['test_start']} to {row['test_end']}"
        )
    lines.append("")

    lines.append("## Label Cleaning Experiments")
    for setting in settings_results:
        label_counts = setting.get("label_counts", {})
        lines.append(f"### min_range_pct={setting.get('min_range_pct')}")
        lines.append(
            f"- Counts: long={label_counts.get('long', 0)}, "
            f"short={label_counts.get('short', 0)}, skip={label_counts.get('skip', 0)}"
        )
        lines.append(f"- Ambiguous true count: {setting.get('ambiguous_true', 0)}")

        gate = setting.get("gate")
        if gate and gate.get("best"):
            lines.append(
                "- Gate: precision_trade={:.3f}, recall_trade={:.3f}, coverage_trade={:.3f}".format(
                    gate["best"]["precision_trade"],
                    gate["best"]["recall_trade"],
                    gate["best"]["coverage_trade"],
                )
            )
        else:
            lines.append("- Gate: no feasible thresholds")

        direction = setting.get("direction")
        if direction and direction.get("best"):
            lines.append(
                "- Direction: precision_long={:.3f}, precision_short={:.3f}, gated_count={}".format(
                    direction["best"]["precision_long"],
                    direction["best"]["precision_short"],
                    direction.get("gated_count", 0),
                )
            )
        else:
            lines.append("- Direction: no feasible thresholds")

        end_to_end = setting.get("end_to_end")
        if end_to_end:
            lines.append(
                "- End-to-end: precision_long={:.3f}, precision_short={:.3f}, coverage_total={:.3f}".format(
                    end_to_end.get("precision_long", 0.0),
                    end_to_end.get("precision_short", 0.0),
                    end_to_end.get("coverage_total", 0.0),
                )
            )
            lines.append(
                "- Predictions: long={}, short={}, skip={}".format(
                    end_to_end.get("pred_long", 0),
                    end_to_end.get("pred_short", 0),
                    end_to_end.get("pred_skip", 0),
                )
            )
        else:
            lines.append("- End-to-end: no feasible result")

        multiclass = setting.get("multiclass")
        if multiclass:
            lines.append(
                "- Multiclass: model={}, macro_f1={:.3f}".format(
                    multiclass.get("model"),
                    multiclass.get("macro_f1", 0.0),
                )
            )
            if multiclass.get("thresholds"):
                thresholds = multiclass["thresholds"]
                lines.append(
                    "- Multiclass thresholds: long={}, short={}, coverage={:.3f}".format(
                        thresholds.get("th_long"),
                        thresholds.get("th_short"),
                        thresholds.get("coverage", 0.0),
                    )
                )
            else:
                lines.append("- Multiclass thresholds: no feasible thresholds")

        lines.append("")

    lines.append("## Interpretation Hints")
    for note in notes:
        lines.append(f"- {note}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_v3(
    output_path: Path,
    dataset_summary: Dict[str, object],
    split_table: pd.DataFrame,
    settings_results: List[Dict[str, object]],
    notes: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# BetaTest Report v3")
    lines.append("")

    lines.append("## Dataset Summary")
    lines.append(f"- Rows: {dataset_summary['rows']}")
    lines.append(f"- Time range: {dataset_summary['start']} to {dataset_summary['end']}")
    lines.append(f"- Timestamp column: {dataset_summary['time_col']}")
    lines.append(f"- Dropped duplicate timestamps: {dataset_summary['dropped_duplicates']}")
    lines.append("")

    lines.append("## Split Diagnostics")
    if not split_table.empty:
        row = split_table.iloc[0]
        lines.append(
            f"- Holdout train: {row['train_start']} to {row['train_end']}; "
            f"test: {row['test_start']} to {row['test_end']}"
        )
    lines.append("")

    lines.append("## Label Cleaning Proof")
    counts_seen = set()
    for setting in settings_results:
        counts = setting.get("label_counts", {})
        counts_tuple = (counts.get("long", 0), counts.get("short", 0), counts.get("skip", 0))
        counts_seen.add(counts_tuple)
        lines.append(
            "min_range_pct={} -> long={}, short={}, skip={}".format(
                setting.get("min_range_pct"),
                counts.get("long", 0),
                counts.get("short", 0),
                counts.get("skip", 0),
            )
        )
    if len(counts_seen) == 1 and settings_results:
        lines.append("WARNING: Label counts identical across settings. Cleaning may be ineffective.")
    lines.append("")

    lines.append("## Gate Learnability")
    for setting in settings_results:
        gate = setting.get("gate", {}).get("best")
        if gate:
            lines.append(
                "min_range_pct={} gate_precision={:.3f} gate_recall={:.3f} gate_coverage={:.3f}".format(
                    setting.get("min_range_pct"),
                    gate.get("precision_trade", 0.0),
                    gate.get("recall_trade", 0.0),
                    gate.get("coverage_trade", 0.0),
                )
            )
        else:
            lines.append(f"min_range_pct={setting.get('min_range_pct')} gate: no feasible thresholds")
    lines.append("")

    lines.append("## Direction Learnability")
    for setting in settings_results:
        direction = setting.get("direction", {}).get("best")
        gated_count = setting.get("direction", {}).get("gated_count", 0)
        if direction:
            lines.append(
                "min_range_pct={} precision_long={:.3f} precision_short={:.3f} gated_count={}".format(
                    setting.get("min_range_pct"),
                    direction.get("precision_long", 0.0),
                    direction.get("precision_short", 0.0),
                    gated_count,
                )
            )
        else:
            lines.append(
                f"min_range_pct={setting.get('min_range_pct')} direction: no feasible thresholds"
            )
    lines.append("")

    lines.append("## Multiclass Sanity Summary")
    for setting in settings_results:
        multiclass = setting.get("multiclass", {})
        lines.append(
            "min_range_pct={} model={} macro_f1={:.3f}".format(
                setting.get("min_range_pct"),
                multiclass.get("model"),
                multiclass.get("macro_f1", 0.0),
            )
        )
        if multiclass.get("thresholds"):
            thresholds = multiclass["thresholds"]
            lines.append(
                "  thresholds: long={}, short={}, coverage={:.3f}".format(
                    thresholds.get("th_long"),
                    thresholds.get("th_short"),
                    thresholds.get("coverage", 0.0),
                )
            )
        else:
            lines.append("  thresholds: no feasible thresholds")
    lines.append("")

    lines.append("## End-to-End Precision vs Coverage")
    best_setting = None
    best_score = -1.0
    for setting in settings_results:
        end_to_end = setting.get("end_to_end")
        if not end_to_end:
            continue
        score = (end_to_end.get("precision_long", 0.0) + end_to_end.get("precision_short", 0.0)) / 2
        lines.append(
            "min_range_pct={} precision_long={:.3f} precision_short={:.3f} coverage_total={:.3f}".format(
                setting.get("min_range_pct"),
                end_to_end.get("precision_long", 0.0),
                end_to_end.get("precision_short", 0.0),
                end_to_end.get("coverage_total", 0.0),
            )
        )
        lines.append(
            "  predictions: long={}, short={}, skip={}".format(
                end_to_end.get("pred_long", 0),
                end_to_end.get("pred_short", 0),
                end_to_end.get("pred_skip", 0),
            )
        )
        if score > best_score:
            best_score = score
            best_setting = setting
    lines.append("")

    verdict = "Model misconfigured"
    if best_setting and best_setting.get("end_to_end"):
        end_to_end = best_setting["end_to_end"]
        gate_best = best_setting.get("gate", {}).get("best", {})
        direction_best = best_setting.get("direction", {}).get("best", {})
        precision_long = end_to_end.get("precision_long", 0.0)
        precision_short = end_to_end.get("precision_short", 0.0)
        if precision_long >= 0.7 and precision_short >= 0.7 and end_to_end.get("coverage_total", 0.0) > 0:
            verdict = "High-precision trading feasible with constraints"
        else:
            gate_precision = gate_best.get("precision_trade", 0.0)
            dir_precision = (direction_best.get("precision_long", 0.0) + direction_best.get("precision_short", 0.0)) / 2
            if gate_precision < 0.55 and dir_precision < 0.55:
                verdict = "Label is noisy"
            elif gate_precision >= 0.6 and dir_precision < 0.55:
                verdict = "Features insufficient"
            else:
                verdict = "Model misconfigured"

    lines.append("## Verdict")
    lines.append(f"- {verdict}")
    lines.append("")

    lines.append("## Interpretation Hints")
    for note in notes:
        lines.append(f"- {note}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
