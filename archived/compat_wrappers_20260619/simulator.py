"""Compatibility wrapper for the moved simulation entrypoint.

New code should import from ``pipelines.simulation.simulator``.
"""
from pathlib import Path
import runpy
import sys

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

if __name__ == "__main__":
    runpy.run_module("pipelines.simulation.simulator", run_name="__main__")
else:
    from pipelines.simulation.simulator import *  # noqa: F401,F403
