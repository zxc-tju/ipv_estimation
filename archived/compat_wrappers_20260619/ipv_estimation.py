"""Compatibility wrapper for the moved IPV estimation module.

New code should import from ``sociality_estimation.core.ipv_estimation``.
"""
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sociality_estimation.core.ipv_estimation import *  # noqa: F401,F403
