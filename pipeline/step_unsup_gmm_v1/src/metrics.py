from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .utils import safe_mean, safe_std


def aggregate_runs(run_records: list[dict], group_cols: Iterable[str]) -> pd.DataFrame:
    ok_runs = [r for r in run_records if r.get("status") == "ok"]
    if not ok_runs:
        return pd.DataFrame()

    df = pd.DataFrame(ok_runs)

    metrics = [
        "train_avg_loglik",
        "test_avg_loglik",
        "train_aic",
        "train_bic",
        "test_aic",
        "test_bic",
        "entropy",
    ]

    group_cols = list(group_cols)
    rows = []

    for _, group in df.groupby(group_cols, dropna=False):
        record = {col: group.iloc[0][col] for col in group_cols}
        record["n_repeats"] = len(group)

        for metric in metrics:
            record[f"{metric}_mean"] = safe_mean(group[metric].tolist())
            record[f"{metric}_std"] = safe_std(group[metric].tolist())

        weights = np.stack(group["weights"].to_list())
        means = np.stack(group["means_flat"].to_list())

        weights_std = weights.std(axis=0)
        means_std = means.std(axis=0)

        record["weights_l1_std"] = float(np.sum(np.abs(weights_std)))
        record["means_l2_std"] = float(np.linalg.norm(means_std))

        rows.append(record)

    return pd.DataFrame(rows)
