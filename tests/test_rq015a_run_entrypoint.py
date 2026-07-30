"""Fixtures for the guarded RQ015A entrypoint."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

import build_ledger  # noqa: E402
import run_rq015a  # noqa: E402


SCHEMA = ROOT / "reports" / "plans" / "RQ015A_ledger_schema_v2.json"
RUN_SPEC_V2 = ROOT / "reports" / "plans" / "RQ015A_run_spec_v2_20260727.json"
RUN_SPEC_V3 = ROOT / "reports" / "plans" / "RQ015A_run_spec_v3_20260730.json"
AUTH = ROOT / "configs" / "research_authorization.json"
EXPECTED_RQ015A_FIXTURES = [
    "tests/test_rq015a_contracts.py",
    "tests/test_rq015a_build_ledger.py",
    "tests/test_rq015a_validate_receipt.py",
    "tests/test_rq015a_factor_analysis.py",
    "tests/test_rq015a_run_entrypoint.py",
]


def _run_spec(tmp_path: Path, execution_authorized=False) -> Path:
    payload = {
        "operation_id": "rq015a_concentration_audit",
        "execution_authorized": execution_authorized,
        "environment": {
            "min_python": "3.9",
            "required_modules": ["json", "math", "dataclasses", "pathlib", "csv"],
            "validate_phase_only": ["pytest"],
            "parquet_engine": {
                "accept": ["pyarrow", "fastparquet"],
                "record_actual_in_receipt": True,
            },
        },
        "input_roots": [],
        "output_root": str(tmp_path / "out_<UTC>_<planSHA8>"),
        "bound_artifacts": {
            "plan": "reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md",
            "ledger_schema": "reports/plans/RQ015A_ledger_schema_v2.json",
        },
    }
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _auth(tmp_path: Path, allowed=False) -> Path:
    operations = ["rq015a_concentration_audit"] if allowed else []
    payload = {
        "schema_version": 1,
        "authorizations": {
            "rq015a_concentration_audit": {
                "allowed_operations": operations,
                "execution_authorized": allowed,
            }
        },
    }
    path = tmp_path / "research_authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_execute_denies_false_authorization_before_measurement_reader(tmp_path, monkeypatch):
    opened = []

    def fail_if_opened(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("MeasurementReader must not be constructed")

    monkeypatch.setattr(build_ledger, "open_measurement_reader", fail_if_opened)
    receipt_path = tmp_path / "run_receipt.json"

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(_run_spec(tmp_path, execution_authorized=False)),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--receipt", str(receipt_path),
    ])

    assert code != 0
    assert opened == []
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["receipt_kind"] == "run_receipt"
    assert data["machine_verdict"] == "FAIL"
    assert data["reads_measurement_fields"] is False
    assert data["metadata"]["measurement_reader_constructed"] is False


def test_validate_only_does_not_call_execute_permit_loader(tmp_path, monkeypatch):
    def forbidden_permit(*args, **kwargs):
        raise AssertionError("ExecutePermit path must not be touched by validate-only")

    monkeypatch.setattr(build_ledger, "load_execute_permit", forbidden_permit)
    monkeypatch.setattr(
        run_rq015a.validate_only,
        "run_contract_fixtures",
        lambda repo_root: {"command": ["pytest"], "passed": 1, "output_last_line": "1 passed"},
    )
    receipt_path = tmp_path / "validate_receipt.json"

    code = run_rq015a.main([
        "--validate-only",
        "--run-spec", str(_run_spec(tmp_path)),
        "--schema", str(SCHEMA),
        "--receipt", str(receipt_path),
    ])

    assert code == 0
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["receipt_kind"] == "validate_receipt"
    assert data["metadata"]["runtime_environment"]["executable"]


def test_execute_denies_missing_authorization_object_before_reader(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(
        build_ledger,
        "open_measurement_reader",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(_run_spec(tmp_path, execution_authorized=True)),
        "--authorization", str(_auth(tmp_path, allowed=False)),
        "--schema", str(SCHEMA),
    ])

    assert code != 0
    assert opened == []


@pytest.mark.parametrize("argv", [[], ["--validate-only", "--execute"]])
def test_entrypoint_requires_exactly_one_mode(argv):
    with pytest.raises(SystemExit) as exc:
        run_rq015a.main(argv)
    assert exc.value.code != 0


def test_existing_receipt_path_is_rejected_before_validate_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_rq015a.validate_only,
        "run_contract_fixtures",
        lambda repo_root: {"command": ["pytest"], "passed": 1, "output_last_line": "1 passed"},
    )
    receipt_path = tmp_path / "validate_receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")

    code = run_rq015a.main([
        "--validate-only",
        "--run-spec", str(_run_spec(tmp_path)),
        "--schema", str(SCHEMA),
        "--receipt", str(receipt_path),
    ])

    assert code != 0


def test_run_spec_v2_loads_and_binds_ledger_schema_v2_without_fixture_count():
    data = json.loads(RUN_SPEC_V2.read_text(encoding="utf-8"))
    text = RUN_SPEC_V2.read_text(encoding="utf-8")

    assert data["operation_id"] == "rq015a_concentration_audit"
    assert data["bound_artifacts"]["ledger_schema"].endswith("RQ015A_ledger_schema_v2.json")
    assert "fixtures 16/16" not in text
    assert re.search(r"fixtures\s+\d+/\d+", text) is None


def test_run_spec_v3_loads_binds_v5_plan_and_remains_denied():
    data = json.loads(RUN_SPEC_V3.read_text(encoding="utf-8"))
    text = RUN_SPEC_V3.read_text(encoding="utf-8")

    assert data["operation_id"] == "rq015a_concentration_audit"
    assert data["execution_authorized"] is False
    assert data["bound_artifacts"]["plan"] == (
        "reports/plans/RQ015A_plan_v5_concentration_audit_20260730.md"
    )
    assert data["bound_artifacts"]["plan_superseded"] == [
        "reports/plans/RQ015A_plan_v4_concentration_audit_20260727.md",
        "reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md",
    ]
    assert data["bound_artifacts"]["ledger_schema"].endswith(
        "RQ015A_ledger_schema_v2.json"
    )
    assert data["bound_artifacts"]["fixtures"] == EXPECTED_RQ015A_FIXTURES
    assert data["bound_artifacts"]["checksum_manifest"] == (
        "reports/plans/RQ015A_plan_v5_checksums_20260730.sha256"
    )
    assert "T10" in data["bound_artifacts"]["checksum_manifest_pending"]
    assert "228 passed" not in text
    assert re.search(r"\b\d+\s+passed\b", text) is None
    assert re.search(r"fixtures\s+\d+/\d+", text) is None
    digest_policy = data["environment"]["input_digest_policy"]
    assert digest_policy["large_file_policy"] == "sha256_full_file"
    assert digest_policy["directory_manifest_sha256_includes"] == [
        "path",
        "bytes",
        "sha256",
    ]
    assert "mtime_ns" in digest_policy["directory_manifest_entry_metadata_excluded"]
    assert data["input_root_constraints"]["symlink_policy"].startswith(
        "Reject every symlink"
    )
    assert data["aggregation_key_encoding"]["version"] == "rq015a-aggregation-key-v2"
    assert data["aggregation_key_encoding"]["product_row_key_escaped_chars"] == [
        "\\",
        "|",
        "=",
    ]
    assert data["trust_boundary"] == [
        "`scripts/rq015a/run_rq015a.py` 是唯一受信入口。",
        "直接调用内部函数，或用 object.__new__ / pickle / 元类伪造对象，不在信任模型内。",
        "边界内必须成立：(a) 从公开 CLI 出发的任何可达路径都不得绕过 permit 校验、allowlist 回源核对、或结构列 denylist；(b) 外部对象不得替换掉已经过校验的对象。",
    ]

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V3),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
    ])
    assert code != 0


def test_research_authorization_contains_denied_rq015a_operation():
    data = json.loads(AUTH.read_text(encoding="utf-8"))
    entry = data["authorizations"]["rq015a_concentration_audit"]

    assert entry["execution_authorized"] is False
    assert entry["allowed_operations"] == []
    assert entry["run_spec_path"] == "reports/plans/RQ015A_run_spec_v3_20260730.json"
    assert entry["execution_contract_path"] == (
        "reports/plans/RQ015A_run_spec_v3_20260730.json"
    )
    assert entry["decision_path"] == (
        "reports/plans/RQ015A_plan_v5_concentration_audit_20260730.md"
    )
