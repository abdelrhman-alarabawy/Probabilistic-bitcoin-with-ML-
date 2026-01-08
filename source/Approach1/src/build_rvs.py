import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .load_data import load_data
from .quality_checks import generate_report
from .rv_definitions import compute_rvs
from .utils import discretize_quantile, ensure_dir, load_yaml, write_json


def build_datasets(config_path: str) -> None:
    config = load_yaml(config_path)
    np.random.seed(42)

    df, column_map = load_data(config)
    settings = config["rv_settings"]

    rv_continuous, rv_meta = compute_rvs(df, column_map, settings)

    ts_col = column_map["timestamp"]
    label_col = column_map.get("label")

    rv_continuous.insert(0, ts_col, df[ts_col])
    if label_col:
        rv_continuous[label_col] = df[label_col]

    disc_cfg = config["discretization"]
    if disc_cfg["method"] != "quantile":
        raise ValueError(f"Unsupported discretization method: {disc_cfg['method']}")
    missing_value = disc_cfg["missing_value"]
    suffix = disc_cfg["label_suffix"]

    rv_discrete = pd.DataFrame(index=rv_continuous.index)
    rv_discrete[ts_col] = rv_continuous[ts_col]
    if label_col:
        rv_discrete[label_col] = rv_continuous[label_col]
    if "direction_sign" in rv_continuous.columns:
        rv_discrete["direction_sign"] = rv_continuous["direction_sign"]

    bin_edges = {}
    bin_counts = {}

    for name, info in rv_meta.items():
        if info["type"] != "continuous":
            continue
        series = rv_continuous[name]
        codes, edges = discretize_quantile(
            series,
            bins=disc_cfg["bins"],
            min_unique=disc_cfg["min_unique"],
        )
        if codes is None or edges is None:
            continue
        codes = codes.fillna(missing_value).astype(int)
        rv_discrete[f"{name}{suffix}"] = codes
        bin_edges[name] = edges
        bin_counts[name] = len(edges) - 1

    output_dir = Path(config["paths"]["output_dir"])
    ensure_dir(str(output_dir))

    cont_path = output_dir / config["paths"]["continuous_output"]
    disc_path = output_dir / config["paths"]["discrete_output"]
    meta_path = output_dir / config["paths"]["metadata_output"]
    report_path = output_dir / config["paths"]["report_output"]

    rv_continuous.to_csv(cont_path, index=False)
    rv_discrete.to_csv(disc_path, index=False)

    rv_only = rv_continuous[[name for name in rv_meta.keys() if name in rv_continuous.columns]]
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_csv": config["paths"]["input_csv"],
        "resolved_columns": column_map,
        "rv_definitions": rv_meta,
        "discretization": {
            "method": disc_cfg["method"],
            "bins": disc_cfg["bins"],
            "label_suffix": suffix,
            "missing_value": missing_value,
            "bin_edges": bin_edges,
            "bin_counts": bin_counts,
        },
        "stats": {
            "rows": len(df),
            "rv_missingness": rv_only.isna().mean().to_dict(),
        },
    }
    write_json(str(meta_path), metadata)

    report = generate_report(df, rv_continuous, rv_discrete, column_map, rv_meta)
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved continuous RVs to {cont_path}")
    print(f"Saved discrete RVs to {disc_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Saved report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Approach1 random variables")
    parser.add_argument(
        "--config",
        default="source/Approach1/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    build_datasets(args.config)


if __name__ == "__main__":
    main()
