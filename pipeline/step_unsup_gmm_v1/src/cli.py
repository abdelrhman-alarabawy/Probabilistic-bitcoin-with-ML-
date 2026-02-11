from __future__ import annotations

import argparse
from pathlib import Path

from .excel_summary import build_excel_summary
from .export_top_models import export_top_models
from .gmm_train import run_sweep
from .labeling import label_top_models
from .utils import setup_logging


logger = setup_logging()


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised GMM Pipeline v1")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep_p = sub.add_parser("sweep", help="Run GMM sweep")
    _add_config_arg(sweep_p)

    export_p = sub.add_parser("export-top", help="Export per-candle probabilities for top models")
    _add_config_arg(export_p)

    label_p = sub.add_parser("label-top", help="Label per-candle files")
    _add_config_arg(label_p)

    excel_p = sub.add_parser("excel-summary", help="Build Excel summary")
    _add_config_arg(excel_p)

    all_p = sub.add_parser("all", help="Run full pipeline")
    _add_config_arg(all_p)

    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "sweep":
        run_sweep(config_path)
    elif args.command == "export-top":
        export_top_models(config_path)
    elif args.command == "label-top":
        label_top_models(config_path)
    elif args.command == "excel-summary":
        build_excel_summary(config_path)
    elif args.command == "all":
        run_sweep(config_path)
        export_top_models(config_path)
        label_top_models(config_path)
        build_excel_summary(config_path)


if __name__ == "__main__":
    main()
