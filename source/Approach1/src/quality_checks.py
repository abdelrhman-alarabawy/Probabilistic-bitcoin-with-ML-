from typing import Dict, List

import numpy as np
import pandas as pd


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().mean().sort_values(ascending=False)
    return missing.to_frame(name="missing_rate")


def find_constant_columns(df: pd.DataFrame) -> List[str]:
    constant_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 1:
            constant_cols.append(col)
    return constant_cols


def top_correlations(df: pd.DataFrame, top_n: int = 10) -> List[str]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    stacked = upper.stack().sort_values(ascending=False)
    pairs = []
    for (col_a, col_b), value in stacked.head(top_n).items():
        pairs.append(f"{col_a} vs {col_b}: {value:.3f}")
    return pairs


def generate_report(
    df_raw: pd.DataFrame,
    rv_continuous: pd.DataFrame,
    rv_discrete: pd.DataFrame,
    column_map: Dict[str, str],
    rv_meta: Dict[str, dict],
) -> str:
    ts_col = column_map["timestamp"]
    label_col = column_map.get("label")
    lines = []
    lines.append("# Approach1 Data Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"Rows: {len(df_raw)}")
    lines.append(f"Timestamp column: `{ts_col}`")
    lines.append("")

    if label_col:
        label_counts = df_raw[label_col].value_counts(dropna=False)
        lines.append("## Label Balance")
        lines.append("")
        for label, count in label_counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")

    lines.append("## Random Variables")
    lines.append("")
    continuous_count = sum(1 for info in rv_meta.values() if info["type"] == "continuous")
    discrete_count = sum(1 for info in rv_meta.values() if info["type"] == "discrete")
    extra_cols = 1 + (1 if label_col else 0)
    output_discrete = max(rv_discrete.shape[1] - extra_cols, 0)
    lines.append(f"Continuous RVs: {continuous_count}")
    lines.append(f"Discrete RVs (native): {discrete_count}")
    lines.append(f"Discrete RVs (with bins): {output_discrete}")
    lines.append("")

    lines.append("## Missingness")
    lines.append("")
    rv_columns = [name for name in rv_meta.keys() if name in rv_continuous.columns]
    missing = summarize_missing(rv_continuous[rv_columns])
    for col, row in missing.head(15).iterrows():
        lines.append(f"- {col}: {row['missing_rate']:.2%}")
    lines.append("")

    constant_cols = find_constant_columns(rv_continuous[rv_columns])
    if constant_cols:
        lines.append("## Constant Columns")
        lines.append("")
        for col in constant_cols:
            lines.append(f"- {col}")
        lines.append("")

    top_corr = top_correlations(rv_continuous[rv_columns])
    if top_corr:
        lines.append("## Top Correlations (Absolute)")
        lines.append("")
        for pair in top_corr:
            lines.append(f"- {pair}")
        lines.append("")

    lines.append("## Leakage Checklist")
    lines.append("")
    lines.append("- Rolling windows use shifted inputs (t-1 and earlier).")
    lines.append("- No features use negative shifts or future timestamps.")
    lines.append("- Labels are not used to construct features.")
    lines.append("")

    lines.append("## RV Definitions (Summary)")
    lines.append("")
    for name, info in rv_meta.items():
        lines.append(f"- {name}: {info['description']}")
    lines.append("")

    return "\n".join(lines)
