from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(report_path: Path, payload: Dict) -> None:
    dataset = payload["dataset"]
    splits = payload["splits"]
    metrics = payload["metrics"]
    baseline = payload["baseline"]
    thresholds = payload["thresholds"]
    direction = payload["direction"]
    tables = payload["tables"]

    lines = [
        "# TradingPipeline Report v4",
        "",
        "## Dataset Summary",
        f"- Rows: {dataset['rows']}",
        f"- Time range: {dataset['start']} to {dataset['end']}",
        f"- Decision time: {dataset['decision_time']}",
        "",
        "## Split Diagnostics",
        f"- Train end: {splits['train_end']}",
        f"- Val end: {splits['val_end']}",
        f"- Test start: {splits['test_start']}",
        "",
        "## Direction Training Filters",
        f"- wickiness_p80: {direction['wickiness_cutoff']:.6f}",
        f"- chop_p80: {direction['chop_cutoff']:.6f}",
        f"- ms_missing_frac_max: {direction['missing_frac_max']:.2f}",
        f"- direction_train_count: {direction['direction_train_count']}",
        f"- direction_train_trade_frac: {_format_pct(direction['direction_train_trade_frac'])}",
        f"- balance_method: {direction['balance_method']}",
        f"- calibration_method: {direction['calibration_method']}",
        "",
        "## Threshold Selection",
        f"- T_trade: {thresholds['t_trade']}",
        f"- T_long: {thresholds['t_long']}",
        f"- T_short: {thresholds['t_short']}",
        f"- relaxation_log: {thresholds['relaxation_log']}",
        "",
        "## Baseline vs v4 (Test)",
        f"- Baseline precision_long={baseline['precision_long']:.4f}, "
        f"precision_short={baseline['precision_short']:.4f}, "
        f"trade_precision={baseline['trade_precision']:.4f}, "
        f"coverage={baseline['coverage_total']:.4f}, "
        f"long_pred_count={baseline['long_pred_count']}, "
        f"short_pred_count={baseline['short_pred_count']}, "
        f"expectancy={baseline['expectancy']:.4f}, "
        f"profit_factor={baseline['profit_factor']:.4f}, "
        f"max_drawdown={baseline['max_drawdown']:.4f}",
        f"- v4 precision_long={metrics['precision_long']:.4f}, "
        f"precision_short={metrics['precision_short']:.4f}, "
        f"trade_precision={metrics['trade_precision']:.4f}, "
        f"coverage={metrics['coverage_total']:.4f}, "
        f"long_pred_count={metrics['long_pred_count']}, "
        f"short_pred_count={metrics['short_pred_count']}, "
        f"expectancy={metrics['expectancy']:.4f}, "
        f"profit_factor={metrics['profit_factor']:.4f}, "
        f"max_drawdown={metrics['max_drawdown']:.4f}",
        "",
        "## Artifacts",
        f"- threshold_trade_sweep: {tables['trade_sweep']}",
        f"- threshold_direction_sweep: {tables['direction_sweep']}",
        f"- chosen_thresholds: {tables['chosen_thresholds']}",
        f"- prediction_counts: {tables['prediction_counts']}",
        f"- per_month_metrics: {tables['per_month']}",
        f"- confusion_matrix: {tables['confusion_matrix']}",
        "",
    ]

    report_path.write_text("\n".join(lines))
