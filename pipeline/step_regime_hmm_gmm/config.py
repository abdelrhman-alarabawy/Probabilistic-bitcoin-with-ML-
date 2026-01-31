from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "path": "data/processed/data_1d_indicators.csv",
        "timestamp_col": "timestamp",
        "label_col": None,
        "drop_cols": [],
        "start_date": None,
        "end_date": None,
    },
    "features": {
        "price_columns": ["open", "high", "low", "close", "volume"],
        "indicator_columns": None,
        "exclude_columns": ["local_timestamp_last"],
        "feature_sets": [
            {"name": "indicators_only", "type": "indicators_only"},
            {"name": "ohlcv_only", "type": "ohlcv_only"},
            {"name": "all_numeric", "type": "all_numeric"},
            {
                "name": "custom_example",
                "type": "custom",
                "columns": ["atm_iv_1d", "rr25_1d", "fly25_1d"],
            },
        ],
    },
    "preprocess": {
        "missing": "median",  # median, mean, drop
        "scale": "standard",  # standard, none
        "max_samples_metrics": 10000,
    },
    "walkforward": {
        "min_train_years": 2,
        "test_years": 1,
        "start_year": None,
        "end_year": None,
    },
    "models": {
        "gmm": {
            "components": [2, 3, 4, 5, 6],
            "cov_types": ["full", "tied"],
            "n_runs": 5,
            "n_init": 1,
            "max_iter": 500,
        },
        "hmm": {
            "states": [2, 3, 4, 5, 6],
            "cov_types": ["full", "tied"],
            "n_mix": [1, 2, 3],
            "n_runs": 3,
            "n_iter": 200,
            "tol": 1.0e-3,
            "init_params": "stmcw",
            "params": "stmcw",
            "min_covar": 1.0e-6,
        },
    },
    "output": {
        "dir": "pipeline/step_regime_hmm_gmm/results",
        "summary_dir": "pipeline/step_regime_hmm_gmm/results/summary",
        "save_plots": True,
        "save_state_posteriors": True,
        "save_component_posteriors": False,
        "verbose": True,
    },
}


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG.copy()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. "
                "Install it or pass a python config."
            ) from exc
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    if config_path.suffix.lower() == ".py":
        # Load python config that defines CONFIG dict
        namespace: Dict[str, Any] = {}
        exec(config_path.read_text(encoding="utf-8"), namespace)
        if "CONFIG" not in namespace:
            raise RuntimeError("Python config must define CONFIG dict.")
        return namespace["CONFIG"]

    raise RuntimeError("Unsupported config format. Use .yaml/.yml or .py")
