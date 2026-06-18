"""Backward-compatible entrypoint for the InterHub IPV pipeline.

Use ``process_interhub.py`` for new commands. This module remains so older
scripts and notebooks that import or execute the subset-specific name continue
to work.
"""
from __future__ import annotations

import process_interhub as _process_interhub

globals().update(
    {
        name: getattr(_process_interhub, name)
        for name in dir(_process_interhub)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _process_interhub.main()
