"""Compatibility CLI for the moved InterHub processing pipeline.

New code should import from ``pipelines.interhub.process_interhub``.
"""
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pipelines.interhub.process_interhub import *  # noqa: F401,F403
from pipelines.interhub.process_interhub import main


if __name__ == "__main__":
    main()
