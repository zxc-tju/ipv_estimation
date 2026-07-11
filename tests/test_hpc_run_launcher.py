from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hpc.prepare_research_run import load_spec, validate_spec


ROOT = Path(__file__).resolve().parents[1]


def write_spec(path: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": 1,
        "rq_id": "RQ014",
        "run_id": "blocked-smoke",
        "operation": "scientific_compute",
        "git_commit": "0" * 40,
        "data_manifest_path": "/missing",
        "data_manifest_sha256": "0" * 64,
        "csv_path": "/missing.csv",
        "pkl_root": "/missing-pkl",
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rq014_is_rejected_before_filesystem_or_git_checks(tmp_path) -> None:
    spec = load_spec(write_spec(tmp_path / "run.yaml"))
    with pytest.raises(ValueError, match="not authorized"):
        validate_spec(spec, base=tmp_path, repo=ROOT)


def test_run_identifiers_are_fail_closed(tmp_path) -> None:
    path = write_spec(tmp_path / "run.yaml", run_id="../escape")
    with pytest.raises(ValueError, match="Unsafe run_id"):
        load_spec(path)
