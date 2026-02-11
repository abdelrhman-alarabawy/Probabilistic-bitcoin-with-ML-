from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import load_config
from .data_io import load_dataset
from .features import select_features
from .splits import walk_forward_months
from .utils import (
    ensure_dir,
    format_model_id,
    save_json,
    setup_logging,
    validate_no_nans,
    validate_responsibilities,
)


logger = setup_logging()


def select_top_models(agg_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    primary = cfg["selection"]["primary_metric"]
    tie_breakers = cfg["selection"].get("tie_breakers", [])
    dedup_cols = cfg["selection"].get("deduplicate_on", [])
    top_k = int(cfg["selection"].get("top_k", 10))

    sort_cols = [primary] + tie_breakers
    ascending = [False] + [True] * len(tie_breakers)

    ranked = agg_df.dropna(subset=[primary]).sort_values(by=sort_cols, ascending=ascending)

    if dedup_cols:
        ranked = ranked.drop_duplicates(subset=dedup_cols, keep="first")

    return ranked.head(top_k).reset_index(drop=True)


def export_top_models(config_path: str | Path) -> dict:
    cfg = load_config(config_path)

    output_root = Path(cfg["output"]["root_dir"])
    ledger_dir = ensure_dir(output_root / "ledger")
    top_dir = ensure_dir(output_root / "top_models")

    agg_path = ledger_dir / "gmm_ledger_aggregated.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregated ledger not found: {agg_path}. Run sweep first.")

    agg_df = pd.read_csv(agg_path)
    top_df = select_top_models(agg_df, cfg)

    top_path = ledger_dir / "top_10_models.csv"
    top_df.to_csv(top_path, index=False)

    df, numeric_cols = load_dataset(cfg["data"], timezone=cfg["splits"].get("timezone", "UTC"))

    folds = walk_forward_months(
        df=df,
        timestamp_col=cfg["data"]["timestamp_col"],
        train_months=cfg["splits"]["train_months"],
        test_months=cfg["splits"]["test_months"],
        step_months=cfg["splits"]["step_months"],
        min_train_rows=cfg["splits"]["min_train_rows"],
    )

    features_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    for featureset_cfg in cfg["featuresets"]:
        name = featureset_cfg.get("name", "featureset")
        features_cache[name] = select_features(df, numeric_cols, featureset_cfg)

    for _, row in top_df.iterrows():
        fold_id = int(row["fold_id"])
        featureset_name = row["featureset"]

        if fold_id >= len(folds):
            raise ValueError(f"Fold id {fold_id} not available.")

        fold = folds[fold_id]
        x_all, feature_cols = features_cache[featureset_name]

        x_train = x_all[fold.train_idx]
        validate_no_nans(x_train, f"X_train fold {fold_id} {featureset_name}")

        model = GaussianMixture(
            n_components=int(row["K"]),
            covariance_type=row["covariance_type"],
            reg_covar=float(row["reg_covar"]),
            max_iter=int(row["max_iter"]),
            n_init=int(row["n_init"]),
            init_params=row["init_params"],
            random_state=int(cfg["gmm_sweep"].get("random_seed", 42)),
            tol=float(cfg["gmm_sweep"].get("convergence_tol", 1e-3)),
        )

        model.fit(x_train)

        ts = pd.to_datetime(df[cfg["data"]["timestamp_col"]], utc=True)
        period_mask = (ts >= fold.train_start) & (ts < fold.test_end)
        period_idx = np.where(period_mask)[0]
        x_period = x_all[period_idx]

        validate_no_nans(x_period, f"X_period fold {fold_id} {featureset_name}")

        resp = model.predict_proba(x_period)
        validate_responsibilities(resp)

        hard_cluster = resp.argmax(axis=1)

        base_cols = [cfg["data"]["timestamp_col"]]
        base_cols.extend([c for c in cfg["data"]["ohlcv_cols"] if c in df.columns])

        feature_cols_unique = [c for c in feature_cols if c not in base_cols]
        out_cols = base_cols + feature_cols_unique

        out_df = df.iloc[period_idx][out_cols].copy()

        if cfg["output"].get("save_responsibilities", True):
            for k in range(resp.shape[1]):
                out_df[f"prob_{k}"] = resp[:, k]

        if cfg["output"].get("save_hard_assignments", True):
            out_df["hard_cluster"] = hard_cluster

        model_id = format_model_id(
            fold_id=fold_id,
            featureset=featureset_name,
            k=int(row["K"]),
            covariance_type=row["covariance_type"],
            reg_covar=float(row["reg_covar"]),
        )

        model_dir = ensure_dir(top_dir / model_id)
        out_path = model_dir / "candles_with_gmm_probs.csv"
        out_df.to_csv(out_path, index=False, float_format=cfg["output"].get("float_format"))

        meta = {
            "model_id": model_id,
            "featureset": featureset_name,
            "feature_cols": feature_cols,
            "fold_id": fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "config": {
                "K": int(row["K"]),
                "covariance_type": row["covariance_type"],
                "reg_covar": float(row["reg_covar"]),
                "max_iter": int(row["max_iter"]),
                "n_init": int(row["n_init"]),
                "init_params": row["init_params"],
            },
        }

        save_json(model_dir / "metadata.json", meta)
        logger.info("Exported %s", out_path)

    return {"top_csv": top_path, "top_models_dir": top_dir}
