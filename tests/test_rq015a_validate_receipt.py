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
