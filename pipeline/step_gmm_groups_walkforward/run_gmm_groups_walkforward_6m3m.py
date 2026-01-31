#!/usr/bin/env python
"""
Entry point for 6m train / 3m test / 3m step walk-forward GMM groups.
"""

from __future__ import annotations

from run_gmm_groups_walkforward import build_months_config, main


def run() -> int:
    cfg = build_months_config()
    cfg.covariance_types = ["tied", "full"]
    cfg.k_range = list(range(2, 11))
    return main(cfg)


if __name__ == "__main__":
    raise SystemExit(run())
