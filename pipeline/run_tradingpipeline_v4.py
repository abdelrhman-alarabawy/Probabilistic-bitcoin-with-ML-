from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from TradingPipeline.learnability_v4.run_v4 import run_v4


if __name__ == "__main__":
    run_v4()
