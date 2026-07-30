"""Fixtures for RQ015A validate-only and receipt contracts."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

import receipt  # noqa: E402
import validate_only  # noqa: E402
from rq015a_contracts import (  # noqa: E402
    ATTEMPTED,
    ConservationReport,
    ContractViolation,
    check_conservation,
    load_schema,
)
from rq015a_types import (  # noqa: E402
    StructuralColumnSet as LedgerStructuralColumnSet,
    is_measurement_like_field,
)

SCHEMA = ROOT / "reports" / "plans" / "RQ015A_ledger_schema_v2.json"
VALID_SHA256 = "0" * 64
EXPECTED_RQ015A_FIXTURES = [
    "tests/test_rq015a_contracts.py",
    "tests/test_rq015a_build_ledger.py",
    "tests/test_rq015a_validate_receipt.py",
    "tests/test_rq015a_factor_analysis.py",
    "tests/test_rq015a_run_entrypoint.py",
]
MANIFEST_PACKAGE_PATHS = [
    "configs/research_authorization.json",
    "reports/knowledge/RQ015A_ipv_estimability_labelling/known_issues_and_audit_boundary_20260730.md",
    "reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_20260726.md",
    "reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md",
    "reports/plans/RQ015A_ledger_schema_v2.json",
    "reports/plans/RQ015A_plan_v7_concentration_audit_20260730.md",
    "reports/plans/RQ015A_run_spec_v5_20260730.json",
    "reports/plans/RQ015A_wod_retrieval_spec_v1.json",
    "scripts/rq015a/build_ledger.py",
    "scripts/rq015a/factor_analysis.py",
    "scripts/rq015a/receipt.py",
    "scripts/rq015a/rq015a_contracts.py",
    "scripts/rq015a/rq015a_types.py",
    "scripts/rq015a/run_rq015a.py",
    "scripts/rq015a/validate_only.py",
    "tests/test_rq015a_build_ledger.py",
    "tests/test_rq015a_contracts.py",
    "tests/test_rq015a_factor_analysis.py",
    "tests/test_rq015a_run_entrypoint.py",
    "tests/test_rq015a_validate_receipt.py",
]


class _Completed:
    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout


def _base_fields(schema):
    return dict(
        per_artifact_conservation={
            "fixture_artifact": {
                "identity_1": True,
                "identity_2": True,
                "identity_3": True,
                "measurement_rows": 1,
            }
        },
        held_out_parsed_rows=0,
        held_out_conclusion_rows=0,
        duplicate_primary_keys=0,
        unmapped_measurement_roles=0,
        k_unknown_rows=7,
        bins_stability_verdict="BINS_REPORTABLE",
        c0_routing_stability={"RQ014": True},
        input_sha256={"fixture.csv": VALID_SHA256},
        parquet_engine={"name": "pyarrow", "version": "21.0.0"},
        m4_only_channel_excluded=True,
        aggregation_key_derivation="v1: perspective/configuration derive from W1 adapter fields.",
        reads_measurement_fields=False,
        failure_reasons=(),
    )


def _checks(schema=None, **updates):
    schema = schema or load_schema(SCHEMA)
    fields = _base_fields(schema)
    fields.update(updates)
    return receipt.build_receipt_checks_from_schema(schema, **fields)


def _conservation_report(artifact_id, measurement_rows=1):
    status_counts = {ATTEMPTED: measurement_rows} if measurement_rows else {}
    recoverability_counts = {"L1_DIRECT": measurement_rows} if measurement_rows else {}
    return check_conservation(
        str(artifact_id),
        measurement_rows,
        1,
        1,
        status_counts,
        recoverability_counts,
    )


def _passing_conservation_for_schema(schema):
    return {
        artifact_id: _conservation_report(artifact_id)
        for artifact_id in schema["ledger_bearing_artifact_ids"]
    }


def _boolean_conservation_for_schema(schema):
    return {
        artifact_id: {
            "identity_1": True,
            "identity_2": True,
            "identity_3": True,
            "measurement_rows": 1,
        }
        for artifact_id in schema["ledger_bearing_artifact_ids"]
    }


def _run_checks(**updates):
    schema = load_schema(SCHEMA)
    fields = _base_fields(schema)
    fields["per_artifact_conservation"] = _passing_conservation_for_schema(schema)
    fields["input_sha256"] = {"fixture.csv": VALID_SHA256}
    fields.update(updates)
    return receipt.build_receipt_checks_from_schema(schema, **fields)


def _empty_manifest_result():
    return {
        "path": "manifest.sha256",
        "manifest_sha256": VALID_SHA256,
        "line_count": 20,
        "checked_count": 20,
        "expected_paths": MANIFEST_PACKAGE_PATHS,
        "actual_paths": MANIFEST_PACKAGE_PATHS,
        "failure_count": 0,
        "failures": [],
        "entries": {},
    }


def _machine_verdict_for_fixture_result(result):
    reasons = validate_only._validate_only_failure_reasons(
        (),
        _empty_manifest_result(),
        result,
    )
    return receipt.build_validate_receipt(_checks(failure_reasons=reasons)).machine_verdict


def _write_manifest_fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    run_spec_rel = "reports/plans/RQ015A_run_spec_v5_20260730.json"
    manifest_rel = "reports/plans/RQ015A_plan_v7_checksums_20260730.sha256"
    payload = {
        "operation_id": "rq015a_concentration_audit",
        "entrypoint": "scripts/rq015a/run_rq015a.py",
        "authorization_object": "configs/research_authorization.json#authorizations.rq015a_concentration_audit",
        "input_roots": [],
        "bound_artifacts": {
            "plan": "reports/plans/RQ015A_plan_v7_concentration_audit_20260730.md",
            "ledger_schema": "reports/plans/RQ015A_ledger_schema_v2.json",
            "contracts_impl": "scripts/rq015a/rq015a_contracts.py",
            "ledger_builder": "scripts/rq015a/build_ledger.py",
            "validate_only_impl": "scripts/rq015a/validate_only.py",
            "receipt_impl": "scripts/rq015a/receipt.py",
            "factor_analysis_impl": "scripts/rq015a/factor_analysis.py",
            "fixtures": EXPECTED_RQ015A_FIXTURES,
            "checksum_manifest": manifest_rel,
            "sealed_exposure_disclosure": "reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md",
        },
    }
    for rel in MANIFEST_PACKAGE_PATHS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == run_spec_rel:
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        else:
            path.write_text("content for %s\n" % rel, encoding="utf-8")
    manifest_path = repo / manifest_rel
    lines = []
    for rel in sorted(MANIFEST_PACKAGE_PATHS):
        lines.append("%s  %s" % (validate_only.sha256_file(repo / rel), rel))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return repo, repo / run_spec_rel, payload, manifest_path


def test_validate_only_ast_has_no_reader_or_permit_names():
    tree = ast.parse((ROOT / "scripts" / "rq015a" / "validate_only.py").read_text())
    forbidden = {"MeasurementReader", "ExecutePermit", "open_measurement_reader"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                seen.add(alias.name.rsplit(".", 1)[-1])
                if alias.asname:
                    seen.add(alias.asname)
    assert not (seen & forbidden)


def test_structural_scan_plan_rejects_ipv_measurement_column():
    with pytest.raises(ContractViolation):
        validate_only.StructuralColumnSet(("counterpart_ipv_error_current",))


@pytest.mark.parametrize("column", [
    "driver_rating",
    "preference_score",
    "human_comment",
    "quality_label",
])
def test_structural_scan_plan_rejects_score_bearing_columns(column):
    with pytest.raises(ContractViolation):
        validate_only.StructuralColumnSet((column,))


@pytest.mark.parametrize("column", [
    "IPV_error",
    " target_ipv_error",
    "Counterpart_IPV_error",
    " M4_ONLY_channel ",
    "ｉｐｖ",
    "driver_RATING",
    "preferenceScore",
    "human-score",
    "quality_LABEL",
    "risk_score",
])
def test_structural_denylist_normalizes_columns_in_both_entrypoints(column):
    assert is_measurement_like_field(column) is True
    with pytest.raises(ContractViolation):
        validate_only.StructuralColumnSet((column,))
    with pytest.raises(ContractViolation):
        LedgerStructuralColumnSet((column,), True)


@pytest.mark.parametrize("column", [
    "ra\u200bting",
    "ipv_\u200cerror",
    "sco\ufeffre",
])
def test_structural_denylist_strips_unicode_format_characters(column):
    assert is_measurement_like_field(column) is True
    with pytest.raises(ContractViolation):
        validate_only.StructuralColumnSet((column,))
    with pytest.raises(ContractViolation):
        LedgerStructuralColumnSet((column,), True)


def test_structural_denylist_allows_legal_structural_columns_after_unicode_fix():
    columns = (
        "case_key",
        "frame_index",
        "timestamp_ms",
        "anchor_frame_index",
        "source_dataset",
        "underscore_count",
    )
    assert all(is_measurement_like_field(column) is False for column in columns)
    assert validate_only.StructuralColumnSet(columns).columns == columns
    assert LedgerStructuralColumnSet(columns, True).columns == columns


def test_structural_denylist_allows_score_inside_unrelated_token():
    column = "underscore_count"
    assert "score" in column
    assert is_measurement_like_field(column) is False
    assert validate_only.StructuralColumnSet((column,)).columns == (column,)
    assert LedgerStructuralColumnSet((column,), True).columns == (column,)


def test_fold_name_cannot_be_used_as_split_filter():
    with pytest.raises(ContractViolation):
        validate_only.StructuralScanPlan(
            requested_columns={"a": validate_only.StructuralColumnSet(("case_id",))},
            split_filters={"rq009_feature_matrix": ("train",)},
            input_roots=(),
        )


def test_validate_receipt_json_records_no_measurement_reads(tmp_path):
    checks = _checks()
    out = receipt.build_validate_receipt(checks)
    path = receipt.write_receipt(tmp_path / "validate_receipt.json", out)
    data = json.loads(path.read_text())
    assert data["reads_measurement_fields"] is False
    assert data["machine_verdict"] == "PASS"


def test_validate_receipt_does_not_require_run_conservation_coverage():
    checks = _checks(per_artifact_conservation={})
    out = receipt.build_validate_receipt(checks)

    assert out.machine_verdict == "PASS"


def test_machine_verdict_cannot_be_supplied_by_caller():
    schema = load_schema(SCHEMA)
    fields = _base_fields(schema)
    fields["machine_verdict"] = "PASS"
    with pytest.raises(ContractViolation):
        receipt.build_receipt_checks_from_schema(schema, **fields)
    with pytest.raises(ContractViolation):
        receipt.build_run_receipt(machine_verdict="PASS")


def test_held_out_parsed_rows_nonzero_makes_machine_verdict_fail():
    checks = _checks(held_out_parsed_rows=1)
    assert receipt.compute_machine_verdict(checks) == "FAIL"


def test_missing_required_receipt_field_raises_contract_violation():
    schema = load_schema(SCHEMA)
    fields = _base_fields(schema)
    fields.pop("input_sha256")
    with pytest.raises(ContractViolation):
        receipt.build_receipt_checks_from_schema(schema, **fields)


def test_artifacts_absent_locally_are_derived_from_schema(tmp_path):
    schema = json.loads(SCHEMA.read_text())
    loaded = load_schema(SCHEMA)
    assert receipt.artifacts_absent_locally_from_schema(loaded) == (
        "wod_rq010b_full479_audited",
        "wod_phase1_phase1b_10hz_schemeb",
        "rq014_g2r_anchor_scores",
    )
    schema["artifacts"].append({
        "artifact_id": "new_absent_fixture",
        "status": "ARTIFACT_NOT_PRESENT_LOCALLY",
    })
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(schema))
    changed = load_schema(p)
    assert "new_absent_fixture" in receipt.artifacts_absent_locally_from_schema(changed)


def test_receipt_path_is_no_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    with pytest.raises(ContractViolation):
        receipt.require_new_json_path(path)
    with pytest.raises(ContractViolation):
        receipt.write_receipt(path, receipt.build_validate_receipt(_checks()))


def test_schema_version_mismatch_makes_machine_verdict_fail():
    checks = _checks(schema_version="rq015a-concentration-ledger-v1")
    assert receipt.compute_machine_verdict(checks) == "FAIL"


def test_directory_file_manifest_detects_equal_size_content_change(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    target = root / "same_size.txt"
    target.write_text("AAAA", encoding="utf-8")
    before = validate_only.structural_path_record(root)
    target.write_text("BBBB", encoding="utf-8")
    after = validate_only.structural_path_record(root)

    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert before["digest_policy"]["small_file_policy"] == "sha256_full_file"
    assert before["digest_policy"]["large_file_policy"] == "sha256_full_file"
    assert before["digest_policy"]["symlink_policy"] == "reject_all_symlinks"
    assert before["files"][0]["hash_policy"] == "sha256_full_file"
    assert before["files"][0]["bytes"] == after["files"][0]["bytes"] == 4


def test_directory_file_manifest_digest_ignores_mtime_metadata(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    target = root / "same_content.txt"
    target.write_text("AAAA", encoding="utf-8")
    before = validate_only.structural_path_record(root)
    st = target.stat()

    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    after = validate_only.structural_path_record(root)

    assert before["manifest_sha256"] == after["manifest_sha256"]
    assert before["files"][0]["mtime_ns"] != after["files"][0]["mtime_ns"]
    assert before["digest_policy"]["directory_manifest_sha256_includes"] == [
        "path", "bytes", "sha256",
    ]


def test_directory_file_manifest_rejects_directory_symlink(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    target = tmp_path / "target_dir"
    target.mkdir()
    (target / "data.txt").write_text("AAAA", encoding="utf-8")
    link = root / "linkdir"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("symlink unavailable: %s" % exc)

    with pytest.raises(ContractViolation, match="linkdir"):
        validate_only.structural_path_record(root)


def test_file_manifest_rejects_file_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("AAAA", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("symlink unavailable: %s" % exc)

    with pytest.raises(ContractViolation, match="link.txt"):
        validate_only.file_manifest_entry(link)


def test_record_input_roots_rejects_parent_traversal(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()

    with pytest.raises(ContractViolation, match="escapes repository root"):
        validate_only.record_input_roots(repo, ["../outside"])


def test_record_input_roots_rejects_absolute_outside_path(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()

    with pytest.raises(ContractViolation, match="escapes repository root"):
        validate_only.record_input_roots(repo, [str(outside)])


def test_record_input_roots_accepts_repo_internal_path(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "data"
    data.mkdir(parents=True)
    (data / "fixture.txt").write_text("AAAA", encoding="utf-8")

    records, failures = validate_only.record_input_roots(repo, ["data"])

    assert failures == ()
    assert records["data"] == validate_only.structural_path_record(data)["manifest_sha256"]


def test_file_manifest_detects_middle_byte_change_after_mtime_restore(tmp_path):
    target = tmp_path / "large_input.bin"
    payload = bytearray(b"A" * (1024 * 1024 + 3))
    payload[len(payload) // 2] = ord("B")
    target.write_bytes(payload)
    before = validate_only.file_manifest_entry(target)
    st = target.stat()

    with target.open("r+b") as handle:
        handle.seek(len(payload) // 2)
        handle.write(b"C")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = validate_only.file_manifest_entry(target)

    assert before["bytes"] == after["bytes"]
    assert before["mtime_ns"] == after["mtime_ns"]
    assert before["hash_policy"] == after["hash_policy"] == "sha256_full_file"
    assert before["sha256"] != after["sha256"]


def test_reads_measurement_fields_true_makes_validate_verdict_fail():
    checks = _checks(reads_measurement_fields=True)
    out = receipt.build_validate_receipt(checks)
    assert out.machine_verdict == "FAIL"


def test_run_receipt_empty_conservation_fails():
    out = receipt.build_run_receipt(_run_checks(per_artifact_conservation={}))
    assert out.machine_verdict == "FAIL"


def test_run_receipt_missing_audited_artifact_fails():
    schema = load_schema(SCHEMA)
    partial = _passing_conservation_for_schema(schema)
    partial.pop(schema["ledger_bearing_artifact_ids"][0])

    out = receipt.build_run_receipt(_run_checks(per_artifact_conservation=partial))

    assert out.machine_verdict == "FAIL"


def test_run_receipt_manual_boolean_conservation_payload_fails():
    schema = load_schema(SCHEMA)
    out = receipt.build_run_receipt(
        _run_checks(per_artifact_conservation=_boolean_conservation_for_schema(schema))
    )

    assert out.machine_verdict == "FAIL"


def test_run_receipt_real_conservation_reports_pass():
    out = receipt.build_run_receipt(_run_checks())

    assert out.machine_verdict == "PASS"


def test_run_receipt_all_zero_measurement_rows_fail():
    schema = load_schema(SCHEMA)
    conservation = {
        artifact_id: _conservation_report(artifact_id, 0)
        for artifact_id in schema["ledger_bearing_artifact_ids"]
    }

    out = receipt.build_run_receipt(
        _run_checks(per_artifact_conservation=conservation)
    )

    assert out.machine_verdict == "FAIL"


def test_run_receipt_partial_zero_measurement_rows_fail():
    schema = load_schema(SCHEMA)
    conservation = _passing_conservation_for_schema(schema)
    artifact_id = schema["ledger_bearing_artifact_ids"][0]
    conservation[artifact_id] = _conservation_report(artifact_id, 0)

    out = receipt.build_run_receipt(
        _run_checks(per_artifact_conservation=conservation)
    )

    assert out.machine_verdict == "FAIL"


def test_run_receipt_empty_input_sha256_fails():
    out = receipt.build_run_receipt(_run_checks(input_sha256={}))
    assert out.machine_verdict == "FAIL"


@pytest.mark.parametrize("digest", ["", "   ", "sha256:abc", "0" * 63, "g" * 64, 123, True])
def test_receipt_input_sha256_rejects_malformed_digest_values(digest):
    with pytest.raises(ContractViolation, match="input_sha256"):
        _run_checks(input_sha256={"fixture.csv": digest})


@pytest.mark.parametrize("path_key", ["", "   ", 123, True])
def test_receipt_input_sha256_rejects_empty_or_non_string_path_keys(path_key):
    with pytest.raises(ContractViolation, match="input_sha256 key"):
        _run_checks(input_sha256={path_key: VALID_SHA256})


@pytest.mark.parametrize("updates", [
    {"parquet_engine": {"name": "   ", "version": "21.0.0"}},
    {"parquet_engine": {"name": "pyarrow", "version": ""}},
    {"schema_version": "   "},
    {"c0_routing_stability": {"": True}},
    {"c0_routing_stability": {"RQ014": "true"}},
    {"failure_reasons": ("",)},
])
def test_receipt_rejects_present_but_meaningless_scalar_values(updates):
    with pytest.raises(ContractViolation):
        _checks(**updates)


def test_run_receipt_failed_conservation_report_fails():
    schema = load_schema(SCHEMA)
    conservation = _passing_conservation_for_schema(schema)
    artifact_id = schema["ledger_bearing_artifact_ids"][0]
    conservation[artifact_id] = ConservationReport(
        artifact_id=str(artifact_id),
        physical_rows=1,
        expansion_factor=1,
        collapse_factor=1,
        measurement_rows_expected=1,
        measurement_rows_observed=1,
        status_counts={ATTEMPTED: 1},
        recoverability_counts={"L1_DIRECT": 0},
    )

    out = receipt.build_run_receipt(_run_checks(per_artifact_conservation=conservation))

    assert out.machine_verdict == "FAIL"


def test_run_receipt_c0_routing_false_is_recorded_not_machine_failure():
    out = receipt.build_run_receipt(_run_checks(c0_routing_stability={"RQ014": False}))
    assert out.machine_verdict == "PASS"


def test_artifacts_absent_locally_override_must_match_schema():
    schema = load_schema(SCHEMA)
    with pytest.raises(ContractViolation, match="schema-derived"):
        receipt.build_receipt_checks_from_schema(
            schema,
            **_base_fields(schema),
            artifacts_absent_locally=(),
        )


def test_literal_backed_runtime_invariants_are_checked(monkeypatch):
    with pytest.raises(ContractViolation):
        validate_only.StructuralScanPlan(
            requested_columns={"a": validate_only.StructuralColumnSet(("case_id",))},
            split_filters={},
            input_roots=(),
            reads_measurement_fields=True,
        )
    with pytest.raises(ContractViolation):
        validate_only.StructuralColumnSet(("case_id",), forbids_measurement_fields=False)
    schema = load_schema(SCHEMA)
    fields = _base_fields(schema)
    fields["m4_only_channel_excluded"] = False
    with pytest.raises(ContractViolation):
        receipt.build_receipt_checks_from_schema(schema, **fields)
    monkeypatch.setattr(validate_only, "READS_MEASUREMENT_FIELDS", True)
    with pytest.raises(ContractViolation):
        validate_only.assert_validate_only_runtime_contract()


def test_pytest_pass_count_is_parsed_not_hardcoded():
    assert validate_only.parse_pytest_pass_count("29 passed in 0.44s") == 29
    assert validate_only.parse_pytest_pass_count("16 passed in 0.10s") == 16


def test_contract_fixture_gate_fails_when_only_one_bound_file_runs(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _Completed(0, "1 passed in 0.01s\n")

    monkeypatch.setattr(validate_only.subprocess, "run", fake_run)
    result = validate_only.run_contract_fixtures(
        ROOT,
        EXPECTED_RQ015A_FIXTURES,
        actual_fixture_paths=EXPECTED_RQ015A_FIXTURES[:1],
    )

    assert result["total_passed"] == 1
    assert result["has_failures"] is True
    assert any("fixture_set_mismatch" in item for item in result["failures"])
    assert _machine_verdict_for_fixture_result(result) == "FAIL"


def test_contract_fixture_gate_fails_on_file_set_mismatch(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _Completed(0, "2 passed in 0.01s\n")

    monkeypatch.setattr(validate_only.subprocess, "run", fake_run)
    result = validate_only.run_contract_fixtures(
        ROOT,
        EXPECTED_RQ015A_FIXTURES,
        actual_fixture_paths=[EXPECTED_RQ015A_FIXTURES[0], "tests/not_bound.py"],
    )

    assert result["has_failures"] is True
    assert any("missing=" in item and "extra=tests/not_bound.py" in item
               for item in result["failures"])
    assert _machine_verdict_for_fixture_result(result) == "FAIL"


def test_contract_fixture_gate_passes_when_all_bound_files_pass(monkeypatch):
    pass_counts = {
        rel: index + 1
        for index, rel in enumerate(EXPECTED_RQ015A_FIXTURES)
    }

    def fake_run(cmd, **kwargs):
        rel = Path(cmd[-2]).relative_to(ROOT).as_posix()
        return _Completed(0, "%d passed in 0.01s\n" % pass_counts[rel])

    monkeypatch.setattr(validate_only.subprocess, "run", fake_run)
    result = validate_only.run_contract_fixtures(ROOT, EXPECTED_RQ015A_FIXTURES)

    assert result["has_failures"] is False
    assert result["total_passed"] == sum(pass_counts.values())
    assert set(result["per_file"]) == set(EXPECTED_RQ015A_FIXTURES)
    assert _machine_verdict_for_fixture_result(result) == "PASS"


def test_contract_fixture_gate_fails_when_any_file_fails(monkeypatch):
    def fake_run(cmd, **kwargs):
        rel = Path(cmd[-2]).relative_to(ROOT).as_posix()
        if rel == EXPECTED_RQ015A_FIXTURES[1]:
            return _Completed(1, "1 failed, 1 passed in 0.01s\n")
        return _Completed(0, "2 passed in 0.01s\n")

    monkeypatch.setattr(validate_only.subprocess, "run", fake_run)
    result = validate_only.run_contract_fixtures(ROOT, EXPECTED_RQ015A_FIXTURES)

    assert result["has_failures"] is True
    assert result["per_file"][EXPECTED_RQ015A_FIXTURES[1]]["failed"] == 1
    assert _machine_verdict_for_fixture_result(result) == "FAIL"


def test_checksum_manifest_verifies_all_twenty_package_files(tmp_path):
    repo, run_spec_path, run_spec, _manifest_path = _write_manifest_fixture_repo(tmp_path)

    result = validate_only.verify_run_spec_checksum_manifest(repo, run_spec_path, run_spec)

    assert result["failure_count"] == 0
    assert result["line_count"] == 20
    assert result["checked_count"] == 20


def test_checksum_manifest_fails_when_registered_file_drifts(tmp_path):
    repo, run_spec_path, run_spec, _manifest_path = _write_manifest_fixture_repo(tmp_path)
    (repo / "scripts/rq015a/receipt.py").write_text("changed\n", encoding="utf-8")

    result = validate_only.verify_run_spec_checksum_manifest(repo, run_spec_path, run_spec)

    assert result["failure_count"] > 0
    assert any("sha256_mismatch:scripts/rq015a/receipt.py" == item
               for item in result["failures"])


def test_checksum_manifest_fails_when_line_is_missing(tmp_path):
    repo, run_spec_path, run_spec, manifest_path = _write_manifest_fixture_repo(tmp_path)
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    result = validate_only.verify_run_spec_checksum_manifest(repo, run_spec_path, run_spec)

    assert result["failure_count"] > 0
    assert any(item.startswith("missing_entries:") for item in result["failures"])


def test_checksum_manifest_fails_when_extra_line_is_present(tmp_path):
    repo, run_spec_path, run_spec, manifest_path = _write_manifest_fixture_repo(tmp_path)
    extra = repo / "extra.txt"
    extra.write_text("extra\n", encoding="utf-8")
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write("%s  extra.txt\n" % validate_only.sha256_file(extra))

    result = validate_only.verify_run_spec_checksum_manifest(repo, run_spec_path, run_spec)

    assert result["failure_count"] > 0
    assert any(item == "extra_entries:extra.txt" for item in result["failures"])
