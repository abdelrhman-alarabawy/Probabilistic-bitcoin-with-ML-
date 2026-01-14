from __future__ import annotations

from pathlib import Path
from typing import Dict


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(report_path: Path, payload: Dict) -> None:
    lines = ["# TradingPipeline Report", ""]

    dataset = payload["dataset"]
    splits = payload["splits"]
    label_dist = payload["label_dist"]
    risk = payload["risk"]
    micro = payload.get("microstructure")
    comparison = payload.get("comparison")
    thresholds = payload["thresholds"]
    test_metrics = payload["test_metrics"]
    backtest = payload["backtest"]
    tables = payload["tables"]
    figures = payload["figures"]

    lines.extend(
        [
            "## Dataset Summary",
            f"- Rows: {dataset['rows']}",
            f"- Time range: {dataset['start']} to {dataset['end']}",
            f"- Dropped duplicate timestamps: {dataset['dropped_duplicates']}",
            "",
            "## Split Diagnostics",
            f"- Holdout train end: {splits['holdout_train_end']}",
            f"- Holdout test start: {splits['holdout_test_start']}",
            f"- Train end: {splits['train_end']}",
            f"- Val end: {splits['val_end']}",
            f"- Test start: {splits['test_start']}",
            "",
            "## Label Distribution",
        ]
    )
    for split_name, dist in label_dist.items():
        lines.append(f"- {split_name}: long={_format_pct(dist['long'])}, "
                     f"short={_format_pct(dist['short'])}, skip={_format_pct(dist['skip'])}")

    lines.extend(
        [
            "",
            "## Risk Filters",
            "- Volume ratio filter uses rolling median volume and train percentile cutoffs.",
            "- Range z-score filter uses rolling mean/std of range percent and train percentiles.",
            "- ATR percent filter uses rolling ATR and train percentile cutoffs.",
            f"- Cutoffs: {risk['cutoffs_path']}",
            f"- Pass rates: {risk['pass_rates_path']}",
            f"- Label distribution (filtered): {risk['filtered_labels_path']}",
            "",
        ]
    )

    if micro:
        alignment_status = "PASS" if micro.get("alignment_pass") else "FAIL"
        lines.extend(
            [
                "## Microstructure (5m) Features",
                f"- Decision time: {dataset.get('decision_time', 'open')} (window end exclusive).",
                "- No leakage: 5m candles are aligned by time window [t - lookback, t).",
                f"- Lookback hours: {micro.get('lookback_hours')}",
                f"- Min 5m bars: {micro.get('min_5m_bars')}",
                f"- Ret cutoff (train P95): {micro.get('ret_cutoff')}",
                f"- Feature count: {micro.get('feature_count')}",
                f"- Alignment checks: {alignment_status}",
                f"- Alignment samples: {micro.get('alignment_checks_path')}",
                f"- Feature stats: {micro.get('feature_stats_path')}",
                f"- Debug samples: {micro.get('debug_samples_path')}",
                "",
            ]
        )

    if comparison:
        base = comparison["baseline"]
        micro_comp = comparison["micro"]
        lines.extend(
            [
                "## Before vs After Microstructure",
                f"- Baseline: precision_long={base['precision_long']:.4f}, "
                f"precision_short={base['precision_short']:.4f}, "
                f"precision_trade={base['precision_trade']:.4f}, "
                f"coverage={base['coverage_total']:.4f}, "
                f"expectancy={base['expectancy']:.4f}, "
                f"profit_factor={base['profit_factor']:.4f}, "
                f"max_drawdown={base['max_drawdown']:.4f}, "
                f"gate_auc={base['gate_auc']:.4f}",
                f"- Micro: precision_long={micro_comp['precision_long']:.4f}, "
                f"precision_short={micro_comp['precision_short']:.4f}, "
                f"precision_trade={micro_comp['precision_trade']:.4f}, "
                f"coverage={micro_comp['coverage_total']:.4f}, "
                f"expectancy={micro_comp['expectancy']:.4f}, "
                f"profit_factor={micro_comp['profit_factor']:.4f}, "
                f"max_drawdown={micro_comp['max_drawdown']:.4f}, "
                f"gate_auc={micro_comp['gate_auc']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Threshold Sweep",
            f"- Sweep table: {tables['threshold_sweep']}",
            f"- Precision vs coverage plot: {figures['precision_coverage']}",
            "",
            "## Selected Thresholds",
            f"- T_trade: {thresholds['t_trade']}",
            f"- T_long: {thresholds['t_long']}",
            f"- T_short: {thresholds['t_short']}",
            f"- Validation min precision: {thresholds['min_precision']}",
            "",
            "## Test Metrics",
            f"- precision_long: {test_metrics['precision_long']:.4f}",
            f"- precision_short: {test_metrics['precision_short']:.4f}",
            f"- precision_trade: {test_metrics['precision_trade']:.4f}",
            f"- coverage_total: {test_metrics['coverage_total']:.4f}",
            f"- gate_auc: {test_metrics.get('gate_auc', float('nan')):.4f}",
            f"- confusion matrix: {figures['confusion_matrix']}",
            f"- per-month metrics: {tables['per_month']}",
            "",
            "## Backtest Summary",
            f"- expectancy: {backtest['expectancy']:.4f}",
            f"- win_rate: {backtest['win_rate']:.4f}",
            f"- profit_factor: {backtest['profit_factor']:.4f}",
            f"- max_drawdown: {backtest['max_drawdown']:.4f}",
            f"- trades per month: {tables['trades_per_month']}",
            f"- trade log: {tables['trade_log']}",
            f"- equity curve: {figures['equity_curve']}",
            "",
            "## Verdict",
            payload["verdict"],
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
