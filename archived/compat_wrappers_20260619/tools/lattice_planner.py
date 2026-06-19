"""Compatibility wrapper for the moved lattice planner entrypoint."""
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sociality_estimation.planning.lattice_planner import *  # noqa: F401,F403
