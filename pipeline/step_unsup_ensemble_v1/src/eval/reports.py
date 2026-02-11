from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import save_json


def aggregate_fold_reports(fold_rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not fold_rows:
        summary = {"n_folds": 0}
        save_json(out_dir / "summary.json", summary)
        (out_dir / "summary.md").write_text("# Ensemble Summary\n\nNo fold rows.\n", encoding="utf-8")
        return summary

    df = pd.DataFrame(fold_rows)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    mean_row = {f"{c}_mean": float(df[c].mean()) for c in numeric}
    std_row = {f"{c}_std": float(df[c].std()) for c in numeric}
    summary = {"n_folds": int(df["fold_id"].nunique()), **mean_row, **std_row}
    save_json(out_dir / "summary.json", summary)
    df.to_csv(out_dir / "per_fold_summary.csv", index=False)

    lines = ["# Ensemble Summary", "", f"Folds: {summary['n_folds']}", ""]
    lines.append("## Mean Metrics")
    for k, v in mean_row.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Std Metrics")
    for k, v in std_row.items():
        lines.append(f"- {k}: {v:.6f}")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
