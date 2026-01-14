from __future__ import annotations

from pathlib import Path
from typing import Any


def build_report(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    lines: list[str] = [
        "# 12h GMM + Rolling CV Report",
        "",
        "## Data Summary",
        f"- Input: `{summary['input_path']}`",
        f"- Rows: {summary['rows']}",
        f"- Columns: {summary['columns']}",
        f"- Timestamp column: {summary['timestamp_col']}",
        "",
        "## Labeling",
        f"- Base horizon: {summary['labeling']['base_horizon']} minutes",
        f"- 12h horizon: {summary['labeling']['scaled_horizon']} minutes",
        f"- TP points: {summary['labeling']['tp_points']}",
        f"- SL points: {summary['labeling']['sl_points']}",
        "",
        "## GMM Regimes",
        f"- k selection strategy: {summary['gmm']['k_strategy']}",
        f"- Selected k: {summary['gmm']['selected_k']}",
        "",
        "## Rolling Monthly CV",
        f"- Folds: {summary['folds_count']}",
        f"- Best setting: {summary['best_setting']}",
        "- Fold table: `output/folds_table.csv`",
        "",
        "## Averaged Metrics",
    ]
    for key, metrics in summary["avg_metrics"].items():
        lines.append(f"- {key}: {metrics}")

    lines += [
        "",
        "## Outputs",
        "- Labeled full dataset: `output/labeled_full.csv`",
        "- Regimes full dataset: `output/regimes_full.csv`",
        "- Fold metrics: `output/folds_metrics.csv`",
        "",
        "## Plots",
        "- `plots/close_by_regime.png`",
        "- `plots/regime_counts.png`",
        "- `plots/gmm_cluster_stats.png`",
        "- `plots/label_distribution_by_regime.png`",
        "- `plots/confusion_matrix_fold_*.png`",
        "",
        "## Notes",
        "- Rolling monthly CV uses strict chronological splits.",
        "- Preprocessing (imputer/scaler) is fit on each training fold only.",
        "- GMM fit is per fold; k is selected using BIC on the earliest training window.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
