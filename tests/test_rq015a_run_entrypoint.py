"""Fixtures for the guarded RQ015A entrypoint."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

import build_ledger  # noqa: E402
import receipt  # noqa: E402
import run_rq015a  # noqa: E402


SCHEMA = ROOT / "reports" / "plans" / "RQ015A_ledger_schema_v2.json"
RUN_SPEC_V2 = ROOT / "reports" / "plans" / "RQ015A_run_spec_v2_20260727.json"
RUN_SPEC_V5 = ROOT / "reports" / "plans" / "RQ015A_run_spec_v5_20260730.json"
AUTH = ROOT / "configs" / "research_authorization.json"
CURRENT_HEAD = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    text=True,
).strip()
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
            "plan": "reports/plans/RQ015A_plan_v7_concentration_audit_20260730.md",
            "ledger_schema": "reports/plans/RQ015A_ledger_schema_v2.json",
            "fixtures": EXPECTED_RQ015A_FIXTURES,
            "checksum_manifest": "reports/plans/RQ015A_plan_v7_checksums_20260730.sha256",
        },
    }
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_DEFAULT_AUTHORIZED_PACKAGE_COMMIT = object()


def _auth(
    tmp_path: Path,
    allowed=False,
    run_spec_path: Path = None,
    authorized_package_commit=_DEFAULT_AUTHORIZED_PACKAGE_COMMIT,
) -> Path:
    operations = ["rq015a_concentration_audit"] if allowed else []
    bound_run_spec = str((run_spec_path or (tmp_path / "run_spec.json")).resolve())
    if authorized_package_commit is _DEFAULT_AUTHORIZED_PACKAGE_COMMIT:
        package_commit = CURRENT_HEAD if allowed else None
    else:
        package_commit = authorized_package_commit
    payload = {
        "schema_version": 1,
        "authorizations": {
            "rq015a_concentration_audit": {
                "allowed_operations": operations,
                "execution_authorized": allowed,
                "run_spec_path": bound_run_spec,
                "execution_contract_path": bound_run_spec,
                "authorized_package_commit": package_commit,
            }
        },
    }
    path = tmp_path / "research_authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validate_receipt(
    tmp_path: Path,
    run_spec_path: Path = None,
    input_sha256=None,
    machine_verdict="PASS",
) -> Path:
    path = tmp_path / "validate_receipt.json"
    if run_spec_path is not None:
        run_spec = json.loads(Path(run_spec_path).read_text(encoding="utf-8"))
        manifest_path = run_spec["bound_artifacts"]["checksum_manifest"]
        recorded_spec = str(Path(run_spec_path).resolve())
    else:
        manifest_path = "manifest.sha256"
        recorded_spec = "run_spec.json"
    payload = {
        "receipt_kind": "validate_receipt",
        "machine_verdict": machine_verdict,
        "per_artifact_conservation": {},
        "held_out_parsed_rows": 0,
        "held_out_conclusion_rows": 0,
        "duplicate_primary_keys": 0,
        "unmapped_measurement_roles": 0,
        "k_unknown_rows": 0,
        "bins_stability_verdict": "BINS_REPORTABLE",
        "c0_routing_stability": {},
        "input_sha256": dict(input_sha256 or {}),
        "parquet_engine": {"name": "pyarrow", "version": "21.0.0"},
        "schema_version": receipt.SCHEMA_VERSION,
        "m4_only_channel_excluded": True,
        "artifacts_absent_locally": [],
        "ledger_bearing_artifacts": [],
        "aggregation_key_derivation": "v1: test fixture receipt.",
        "reads_measurement_fields": False,
        "failure_reasons": [],
        "metadata": {
            "run_spec_path": recorded_spec,
            "checksum_manifest": {"path": manifest_path},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_execute_denies_false_authorization_before_measurement_reader(tmp_path, monkeypatch):
    opened = []

    def fail_if_opened(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("MeasurementReader must not be constructed")

    monkeypatch.setattr(build_ledger, "open_measurement_reader", fail_if_opened)
    receipt_path = tmp_path / "run_receipt.json"
    run_spec_path = _run_spec(tmp_path, execution_authorized=False)

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(run_spec_path),
        "--authorization", str(_auth(tmp_path, allowed=False, run_spec_path=run_spec_path)),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(_validate_receipt(tmp_path, run_spec_path)),
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
        lambda *args, **kwargs: {
            "command": ["pytest"],
            "commands": [["pytest"]],
            "expected_files": EXPECTED_RQ015A_FIXTURES,
            "actual_files": EXPECTED_RQ015A_FIXTURES,
            "per_file": {},
            "passed": 1,
            "total_passed": 1,
            "has_failures": False,
            "failures": [],
            "output_last_line": "1 passed",
        },
    )
    monkeypatch.setattr(
        run_rq015a.validate_only,
        "verify_run_spec_checksum_manifest",
        lambda *args, **kwargs: {
            "path": "manifest.sha256",
            "manifest_sha256": "0" * 64,
            "line_count": 1,
            "checked_count": 1,
            "expected_paths": [],
            "actual_paths": [],
            "failure_count": 0,
            "failures": [],
            "entries": {},
        },
    )
    run_spec_path = _run_spec(tmp_path)
    receipt_path = tmp_path / "validate_receipt.json"

    code = run_rq015a.main([
        "--validate-only",
        "--run-spec", str(run_spec_path),
        "--authorization", str(_auth(tmp_path, run_spec_path=run_spec_path)),
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


def test_run_spec_v5_loads_binds_v7_plan_and_remains_denied():
    data = json.loads(RUN_SPEC_V5.read_text(encoding="utf-8"))
    text = RUN_SPEC_V5.read_text(encoding="utf-8")

    assert data["operation_id"] == "rq015a_concentration_audit"
    assert data["execution_authorized"] is False
    assert data["bound_artifacts"]["plan"] == (
        "reports/plans/RQ015A_plan_v7_concentration_audit_20260730.md"
    )
    assert data["bound_artifacts"]["plan_superseded"] == [
        "reports/plans/RQ015A_plan_v6_concentration_audit_20260730.md",
        "reports/plans/RQ015A_plan_v5_concentration_audit_20260730.md",
        "reports/plans/RQ015A_plan_v4_concentration_audit_20260727.md",
        "reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md",
    ]
    assert data["bound_artifacts"]["ledger_schema"].endswith(
        "RQ015A_ledger_schema_v2.json"
    )
    assert data["bound_artifacts"]["fixtures"] == EXPECTED_RQ015A_FIXTURES
    assert data["bound_artifacts"]["checksum_manifest"] == (
        "reports/plans/RQ015A_plan_v7_checksums_20260730.sha256"
    )
    # v5 binds the v7 manifest filename; this worker does not re-sign the manifest.
    assert "checksum_manifest_pending" not in data["bound_artifacts"]
    manifest_rel = data["bound_artifacts"]["checksum_manifest"]
    assert manifest_rel.endswith("RQ015A_plan_v7_checksums_20260730.sha256")
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
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
    ])
    assert code != 0


def test_research_authorization_contains_denied_rq015a_operation():
    data = json.loads(AUTH.read_text(encoding="utf-8"))
    entry = data["authorizations"]["rq015a_concentration_audit"]

    assert entry["execution_authorized"] is False
    assert entry["allowed_operations"] == []
    assert entry["run_spec_path"] == "reports/plans/RQ015A_run_spec_v5_20260730.json"
    assert entry["execution_contract_path"] == (
        "reports/plans/RQ015A_run_spec_v5_20260730.json"
    )
    assert entry["authorized_package_commit"] is None
    assert entry["decision_path"] == (
        "reports/plans/RQ015A_plan_v7_concentration_audit_20260730.md"
    )


def test_forged_tmp_run_spec_is_rejected_by_cli_binding_check(tmp_path, capsys):
    forged = _run_spec(tmp_path, execution_authorized=True)

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(forged),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "run_spec_path mismatch" in captured.err


def test_forged_tmp_run_spec_is_rejected_by_validate_only_binding_check(tmp_path, capsys):
    forged = _run_spec(tmp_path, execution_authorized=True)

    code = run_rq015a.main([
        "--validate-only",
        "--run-spec", str(forged),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "run_spec_path mismatch" in captured.err


def test_canonical_v5_run_spec_binding_passes_before_authorization_denial():
    entry = receipt.assert_run_spec_authorization_binding(
        ROOT,
        RUN_SPEC_V5,
        AUTH,
        "rq015a_concentration_audit",
    )

    assert entry["run_spec_path"] == "reports/plans/RQ015A_run_spec_v5_20260730.json"
    with pytest.raises(build_ledger.ContractViolation, match="execution_authorized is not true"):
        build_ledger.load_execute_permit(RUN_SPEC_V5, AUTH, repo_root=ROOT)


def test_authorized_package_commit_null_is_rejected_when_authorization_true(tmp_path):
    run_spec_path = _run_spec(tmp_path, execution_authorized=True)
    auth_path = _auth(
        tmp_path,
        allowed=True,
        run_spec_path=run_spec_path,
        authorized_package_commit=None,
    )

    with pytest.raises(build_ledger.ContractViolation, match="authorized_package_commit is required"):
        build_ledger.load_execute_permit(run_spec_path, auth_path, repo_root=ROOT)


def test_authorized_package_commit_mismatch_is_rejected_when_authorization_true(tmp_path):
    run_spec_path = _run_spec(tmp_path, execution_authorized=True)
    auth_path = _auth(
        tmp_path,
        allowed=True,
        run_spec_path=run_spec_path,
        authorized_package_commit="0" * 40,
    )

    with pytest.raises(build_ledger.ContractViolation, match="authorized_package_commit mismatch"):
        build_ledger.load_execute_permit(run_spec_path, auth_path, repo_root=ROOT)


def test_canonical_v5_cli_checks_validate_receipt_before_authorization(tmp_path, capsys):
    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(_validate_receipt(tmp_path, RUN_SPEC_V5)),
    ])

    assert code != 0
    captured = capsys.readouterr()
    # execute 必须因 input digest 与 validate receipt 不一致而拒绝。
    # 不绑定具体文案：实现给出的是更具体的 "input SHA differs from validate
    # receipt: missing_in_receipt:<roots>"，点名了缺失的输入根。
    assert "RQ015A_EXECUTE_DENIED" in captured.err
    assert "input SHA differs from validate receipt" in captured.err


def test_execute_rejects_minimal_validate_receipt_before_authorization(tmp_path, capsys):
    minimal = tmp_path / "minimal_validate_receipt.json"
    minimal.write_text(
        json.dumps({
            "receipt_kind": "validate_receipt",
            "machine_verdict": "PASS",
            "input_sha256": {},
        }),
        encoding="utf-8",
    )

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(minimal),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "validate receipt missing required fields" in captured.err


def test_execute_rejects_validate_receipt_missing_required_field(tmp_path, capsys):
    path = _validate_receipt(tmp_path, RUN_SPEC_V5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    path.write_text(json.dumps(payload), encoding="utf-8")

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(path),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "validate receipt missing required fields: schema_version" in captured.err


def test_execute_rejects_validate_receipt_fail_verdict(tmp_path, capsys):
    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(_validate_receipt(
            tmp_path,
            RUN_SPEC_V5,
            machine_verdict="FAIL",
        )),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "validate receipt machine_verdict is not PASS" in captured.err


@pytest.mark.parametrize("case, expected", [
    ("run_spec_path", "validate receipt run_spec_path mismatch"),
    ("schema_version", "validate receipt schema_version mismatch"),
    ("checksum_manifest", "validate receipt checksum_manifest path mismatch"),
])
def test_execute_rejects_validate_receipt_binding_mismatch(tmp_path, capsys, case, expected):
    path = _validate_receipt(tmp_path, RUN_SPEC_V5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if case == "run_spec_path":
        payload["metadata"]["run_spec_path"] = "reports/plans/not_this_run_spec.json"
    elif case == "schema_version":
        payload["schema_version"] = "rq015a-wrong-schema"
    else:
        payload["metadata"]["checksum_manifest"]["path"] = "reports/plans/not_this_manifest.sha256"
    path.write_text(json.dumps(payload), encoding="utf-8")

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(RUN_SPEC_V5),
        "--authorization", str(AUTH),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(path),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert expected in captured.err


def test_execute_requires_validate_receipt_before_permit(tmp_path, capsys):
    run_spec_path = _run_spec(tmp_path, execution_authorized=True)

    code = run_rq015a.main([
        "--execute",
        "--run-spec", str(run_spec_path),
        "--authorization", str(_auth(tmp_path, allowed=True, run_spec_path=run_spec_path)),
        "--schema", str(SCHEMA),
    ])

    assert code != 0
    captured = capsys.readouterr()
    assert "validate receipt is required before execute" in captured.err


def test_execute_rejects_validate_receipt_input_digest_mismatch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_root = repo / "data"
    data_root.mkdir(parents=True)
    target = data_root / "input.txt"
    target.write_text("after", encoding="utf-8")
    run_spec_path = repo / "run_spec.json"
    payload = json.loads(_run_spec(tmp_path, execution_authorized=True).read_text())
    payload["input_roots"] = ["data"]
    payload["output_root"] = str(repo / "out_<UTC>_<planSHA8>")
    run_spec_path.write_text(json.dumps(payload), encoding="utf-8")
    auth_path = _auth(tmp_path, allowed=True, run_spec_path=run_spec_path)
    validate_receipt = _validate_receipt(repo, run_spec_path, {"data": "0" * 64})
    run_receipt = tmp_path / "run_receipt.json"
    opened = []
    monkeypatch.setattr(
        build_ledger,
        "open_measurement_reader",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )

    code = run_rq015a.main([
        "--execute",
        "--repo-root", str(repo),
        "--run-spec", str(run_spec_path),
        "--authorization", str(auth_path),
        "--schema", str(SCHEMA),
        "--validate-receipt", str(validate_receipt),
        "--receipt", str(run_receipt),
    ])

    assert code != 0
    assert opened == []
    data = json.loads(run_receipt.read_text(encoding="utf-8"))
    assert data["machine_verdict"] == "FAIL"
    assert "input SHA differs from validate receipt" in ";".join(data["failure_reasons"])


def test_run_spec_declared_manifest_members_match_derived_members():
    """规格里声明的 checksum_manifest_members 必须与代码推导的成员集合一致。

    代码从 bound_artifacts 的各具名字段推导校验清单成员；规格另外声明了一份
    可读数组供外部消费者使用。两份列表若漂移，清单闸门保护的范围就与规格
    宣称的范围不符——本项目已多次因"同一事实存两份"而出问题，故在此断言。
    """
    import json
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts" / "rq015a"))
    import validate_only as vo

    spec_path = repo_root / "reports/plans/RQ015A_run_spec_v5_20260730.json"
    spec = json.loads(spec_path.read_text())
    declared = spec.get("bound_artifacts", {}).get("checksum_manifest_members")
    assert declared, "run spec must declare checksum_manifest_members"

    derived = vo.expected_checksum_manifest_paths(repo_root, spec_path, spec)
    assert set(declared) == set(derived), (
        "declared manifest members drifted from derived members; "
        "only-declared=%s only-derived=%s"
        % (sorted(set(declared) - set(derived)), sorted(set(derived) - set(declared)))
    )
