"""Compatibility wrapper for the moved core agent module.

New code should import from ``sociality_estimation.core.agent``.
"""
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sociality_estimation.core.agent import *  # noqa: F401,F403
