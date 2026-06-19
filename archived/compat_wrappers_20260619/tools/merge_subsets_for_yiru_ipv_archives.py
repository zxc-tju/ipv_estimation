"""Compatibility CLI for the moved InterHub archive-merge helper."""
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pipelines.interhub.tools.merge_subsets_for_yiru_ipv_archives import *  # noqa: F401,F403
from pipelines.interhub.tools.merge_subsets_for_yiru_ipv_archives import main


if __name__ == "__main__":
    main()
