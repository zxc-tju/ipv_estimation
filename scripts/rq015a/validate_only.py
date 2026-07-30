#!/usr/bin/env python3
"""RQ015A validate-only structural gate.

This module records pre-execution structural checks. It intentionally has no
measurement-opening interface and never requests score-bearing or IPV columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from typing import Literal

from receipt import (
    ValidationReceipt,
    assert_run_spec_authorization_binding,
    build_receipt_checks_from_schema,
    build_validate_receipt,
    canonical_repo_path,
    detect_parquet_engine,
    load_json_object,
    require_new_json_path,
    strip_json_fragment,
    write_receipt,
)
from rq015a_contracts import ContractViolation, SCHEMA_VERSION, load_schema
from rq015a_types import assert_structural_columns_are_safe

READS_MEASUREMENT_FIELDS: Literal[False] = False
MUST_PRECEDE_EXECUTE: Literal[True] = True

RQ009_FOLD_NAMES = ("train", "guard_tune", "calibration", "test")
RQ007_VALIDATE_SPLITS = ("development", "guard")
MANIFEST_HASH_CHUNK_BYTES = 1024 * 1024
OPERATION_ID = "rq015a_concentration_audit"
DEFAULT_RUN_SPEC = "reports/plans/RQ015A_run_spec_v3_20260730.json"
DEFAULT_SCHEMA = "reports/plans/RQ015A_ledger_schema_v2.json"
DEFAULT_AUTHORIZATION = "configs/research_authorization.json"
V3_MANIFEST_SUPPLEMENTAL_PATHS = (
    "scripts/rq015a/rq015a_types.py",
    "reports/plans/RQ015A_wod_retrieval_spec_v1.json",
    "reports/knowledge/RQ015A_ipv_estimability_labelling/known_issues_and_audit_boundary_20260730.md",
    "reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_20260726.md",
)


@dataclass(frozen=True)
class StructuralColumnSet:
    columns: Tuple[str, ...]
    forbids_measurement_fields: Literal[True] = True

    def __post_init__(self) -> None:
        if self.forbids_measurement_fields is not True:
            raise ContractViolation("forbids_measurement_fields must be true")
        _assert_structural_columns(self.columns)


@dataclass(frozen=True)
class StructuralScanPlan:
    requested_columns: Mapping[str, StructuralColumnSet]
    split_filters: Mapping[str, Tuple[str, ...]]
    input_roots: Tuple[str, ...]
    held_out_parsed_rows: int = 0
    reads_measurement_fields: Literal[False] = READS_MEASUREMENT_FIELDS
    must_precede_execute: Literal[True] = MUST_PRECEDE_EXECUTE

    def __post_init__(self) -> None:
        if self.reads_measurement_fields is not False:
            raise ContractViolation("READS_MEASUREMENT_FIELDS must remain false")
        if self.must_precede_execute is not True:
            raise ContractViolation("must_precede_execute must remain true")
        if self.held_out_parsed_rows != 0:
            raise ContractViolation("held_out_parsed_rows must be 0")
        for column_set in self.requested_columns.values():
            _assert_structural_columns(column_set.columns)
        for artifact_id, splits in self.split_filters.items():
            _assert_validate_splits(artifact_id, splits)


def assert_validate_only_runtime_contract() -> None:
    if READS_MEASUREMENT_FIELDS is not False:
        raise ContractViolation("READS_MEASUREMENT_FIELDS must remain false")
    if MUST_PRECEDE_EXECUTE is not True:
        raise ContractViolation("must_precede_execute must remain true")


def build_structural_scan_plan(schema: Mapping[str, Any],
                               allowlist_splits: Optional[Sequence[str]] = None,
                               input_roots: Optional[Sequence[str]] = None) -> StructuralScanPlan:
    assert_validate_only_runtime_contract()
    splits = tuple(allowlist_splits or RQ007_VALIDATE_SPLITS)
    _assert_validate_splits("rq007_allowlist", splits)

    requested = {}
    split_filters = {}
    for artifact in schema.get("artifacts", []):
        artifact_id = str(artifact.get("artifact_id"))
        status = artifact.get("status")
        cols = _structural_columns_for_artifact(artifact)
        requested[artifact_id] = StructuralColumnSet(tuple(sorted(cols)))
        if status is None and artifact.get("rq007_split_applicable") is True:
            split_filters[artifact_id] = splits
    return StructuralScanPlan(
        requested_columns=requested,
        split_filters=split_filters,
        input_roots=tuple(input_roots or ()),
    )


def validate_structural_inputs(plan: StructuralScanPlan,
                               readers: Mapping[str, Any]) -> Mapping[str, Any]:
    assert_validate_only_runtime_contract()
    failures = []
    input_sha256 = {}
    row_counts = {}
    for artifact_id, column_set in sorted(plan.requested_columns.items()):
        reader = readers.get(artifact_id)
        if reader is None:
            continue
        available = tuple(reader.schema_columns())
        _assert_structural_columns(available)
        missing = [c for c in column_set.columns if c not in available]
        if missing:
            failures.append("%s_missing_columns:%s" % (artifact_id, ",".join(missing)))
            continue
        row_counts[artifact_id] = int(reader.count_rows(column_set))
        input_sha256[artifact_id] = str(reader.sha256())
    return {"failure_reasons": tuple(failures),
            "input_sha256": input_sha256,
            "row_counts": row_counts}


def run_validate_only(repo_root: Path, schema_path: Path, run_spec_path: Path,
                      receipt_path: Path,
                      authorization_path: Optional[Path] = None) -> ValidationReceipt:
    assert_validate_only_runtime_contract()
    new_path = require_new_json_path(Path(receipt_path))
    repo = Path(repo_root).resolve()
    spec_path = Path(run_spec_path).resolve()
    if authorization_path is not None:
        assert_run_spec_authorization_binding(
            repo,
            spec_path,
            Path(authorization_path).resolve(),
            OPERATION_ID,
        )
    schema = load_schema(schema_path)
    run_spec = load_json_object(spec_path, "run spec")
    input_roots = tuple(run_spec.get("input_roots", ()))
    plan = build_structural_scan_plan(schema, input_roots=input_roots)

    manifest_result = verify_run_spec_checksum_manifest(repo, spec_path, run_spec)
    fixture_result = run_contract_fixtures(repo, fixture_paths_from_run_spec(run_spec))
    root_records, root_failures = record_input_roots(repo, input_roots)
    failure_reasons = _validate_only_failure_reasons(
        root_failures,
        manifest_result,
        fixture_result,
    )
    checks = build_receipt_checks_from_schema(
        schema,
        per_artifact_conservation={},
        held_out_parsed_rows=plan.held_out_parsed_rows,
        held_out_conclusion_rows=0,
        duplicate_primary_keys=0,
        unmapped_measurement_roles=0,
        k_unknown_rows=0,
        bins_stability_verdict="BINS_REPORTABLE",
        c0_routing_stability={},
        input_sha256=root_records,
        parquet_engine=detect_parquet_engine(),
        m4_only_channel_excluded=True,
        aggregation_key_derivation="v1: perspective/configuration are supplied by W1 adapter; validate-only only asserts this description is present.",
        reads_measurement_fields=READS_MEASUREMENT_FIELDS,
        failure_reasons=failure_reasons,
    )
    metadata = {
        "schema_load_self_check": schema.get("schema_id") == SCHEMA_VERSION,
        "must_precede_execute": MUST_PRECEDE_EXECUTE,
        "run_spec_execution_authorized": run_spec.get("execution_authorized"),
        "contract_fixtures": fixture_result,
        "checksum_manifest": manifest_result,
        "structural_requested_columns": {
            k: list(v.columns) for k, v in sorted(plan.requested_columns.items())
        },
    }
    receipt = build_validate_receipt(checks, metadata=metadata)
    write_receipt(new_path, receipt)
    return receipt


def _validate_only_failure_reasons(
    root_failures: Sequence[str],
    manifest_result: Mapping[str, Any],
    fixture_result: Mapping[str, Any],
) -> Tuple[str, ...]:
    reasons = []
    reasons.extend(root_failures)
    for failure in manifest_result.get("failures", ()):
        reasons.append("checksum_manifest:%s" % failure)
    for failure in fixture_result.get("failures", ()):
        reasons.append("fixtures:%s" % failure)
    return tuple(reasons)


def fixture_paths_from_run_spec(run_spec: Mapping[str, Any]) -> Tuple[str, ...]:
    bound = run_spec.get("bound_artifacts")
    if not isinstance(bound, Mapping):
        raise ContractViolation("run spec missing bound_artifacts object")
    fixtures = bound.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ContractViolation("run spec bound_artifacts.fixtures must be a non-empty list")
    out = []
    for index, value in enumerate(fixtures):
        if not isinstance(value, str) or not value.strip():
            raise ContractViolation("fixtures[%d] must be a non-empty string" % index)
        out.append(_normalize_repo_relative_string(value))
    if len(set(out)) != len(out):
        raise ContractViolation("run spec fixtures must not contain duplicates")
    return tuple(out)


def expected_checksum_manifest_paths(
    repo_root: Path,
    run_spec_path: Path,
    run_spec: Mapping[str, Any],
) -> Tuple[str, ...]:
    bound = run_spec.get("bound_artifacts")
    if not isinstance(bound, Mapping):
        raise ContractViolation("run spec missing bound_artifacts object")
    paths = []
    for key in (
        "plan",
        "ledger_schema",
        "contracts_impl",
        "ledger_builder",
        "validate_only_impl",
        "receipt_impl",
        "factor_analysis_impl",
        "sealed_exposure_disclosure",
    ):
        value = bound.get(key)
        if value:
            paths.append(_normalize_repo_relative_string(str(value)))
    paths.extend(fixture_paths_from_run_spec(run_spec))
    entrypoint = run_spec.get("entrypoint")
    if entrypoint:
        paths.append(_normalize_repo_relative_string(str(entrypoint)))
    paths.append(_repo_relative_existing_or_contract(repo_root, run_spec_path))
    authorization_object = run_spec.get("authorization_object")
    if authorization_object:
        paths.append(_normalize_repo_relative_string(strip_json_fragment(authorization_object)))
    paths.extend(V3_MANIFEST_SUPPLEMENTAL_PATHS)
    deduped = tuple(sorted(set(paths)))
    return deduped


def verify_run_spec_checksum_manifest(
    repo_root: Path,
    run_spec_path: Path,
    run_spec: Mapping[str, Any],
) -> Mapping[str, Any]:
    bound = run_spec.get("bound_artifacts")
    failures = []
    if not isinstance(bound, Mapping):
        return _manifest_result("", "", 0, 0, (), (), ("missing_bound_artifacts",), {})
    manifest_raw = bound.get("checksum_manifest")
    if not isinstance(manifest_raw, str) or not manifest_raw.strip():
        return _manifest_result("", "", 0, 0, (), (), ("missing_checksum_manifest_path",), {})

    repo = Path(repo_root).resolve()
    manifest_rel = _normalize_repo_relative_string(manifest_raw)
    manifest_path = canonical_repo_path(repo, manifest_rel)
    expected_paths = expected_checksum_manifest_paths(repo, Path(run_spec_path).resolve(), run_spec)
    if not manifest_path.exists():
        return _manifest_result(
            manifest_rel,
            "",
            0,
            0,
            expected_paths,
            (),
            ("checksum_manifest_missing:%s" % manifest_rel,),
            {},
        )

    entries, parse_failures, line_count = _parse_checksum_manifest(manifest_path)
    failures.extend(parse_failures)
    actual_paths = tuple(sorted(entries.keys()))
    expected_set = set(expected_paths)
    actual_set = set(actual_paths)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        failures.append("missing_entries:%s" % ",".join(missing))
    if extra:
        failures.append("extra_entries:%s" % ",".join(extra))

    checked_count = 0
    details = {}
    for rel_path, expected_sha in sorted(entries.items()):
        path = canonical_repo_path(repo, rel_path)
        try:
            path.relative_to(repo)
        except ValueError:
            failures.append("entry_escapes_repo:%s" % rel_path)
            details[rel_path] = {"expected": expected_sha, "actual": None, "ok": False}
            continue
        if not path.exists():
            failures.append("entry_missing_file:%s" % rel_path)
            details[rel_path] = {"expected": expected_sha, "actual": None, "ok": False}
            continue
        actual_sha = sha256_file(path)
        ok = actual_sha.lower() == expected_sha.lower()
        checked_count += 1
        if not ok:
            failures.append("sha256_mismatch:%s" % rel_path)
        details[rel_path] = {"expected": expected_sha.lower(), "actual": actual_sha, "ok": ok}

    return _manifest_result(
        manifest_rel,
        sha256_file(manifest_path),
        line_count,
        checked_count,
        expected_paths,
        actual_paths,
        tuple(failures),
        details,
    )


def _manifest_result(
    manifest_path: str,
    manifest_sha256: str,
    line_count: int,
    checked_count: int,
    expected_paths: Sequence[str],
    actual_paths: Sequence[str],
    failures: Sequence[str],
    details: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "line_count": int(line_count),
        "checked_count": int(checked_count),
        "expected_paths": list(expected_paths),
        "actual_paths": list(actual_paths),
        "failure_count": len(tuple(failures)),
        "failures": list(failures),
        "entries": details,
    }


def _parse_checksum_manifest(path: Path) -> Tuple[Mapping[str, str], Tuple[str, ...], int]:
    entries: Dict[str, str] = {}
    failures: List[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            failures.append("blank_line:%d" % index)
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            failures.append("malformed_line:%d" % index)
            continue
        rel_path = parts[1].strip()
        if rel_path.startswith("*"):
            rel_path = rel_path[1:]
        rel_path = _normalize_repo_relative_string(rel_path)
        if rel_path in entries:
            failures.append("duplicate_entry:%s" % rel_path)
            continue
        entries[rel_path] = parts[0].lower()
    return entries, tuple(failures), len(lines)


def _normalize_repo_relative_string(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation("repository path must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        raise ContractViolation("repository path must be relative: %s" % value)
    normalized = path.as_posix()
    if normalized.startswith("../") or normalized == ".." or "/../" in normalized:
        raise ContractViolation("repository path must not traverse upward: %s" % value)
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _repo_relative_existing_or_contract(repo_root: Path, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        raise ContractViolation("run spec path escapes repository root: %s" % path)


def assert_validate_receipt_inputs_current(
    repo_root: Path,
    run_spec: Mapping[str, Any],
    validate_receipt_path: Path,
) -> Mapping[str, Any]:
    data = load_json_object(Path(validate_receipt_path), "validate receipt")
    if data.get("receipt_kind") != "validate_receipt":
        raise ContractViolation("validate receipt has wrong receipt_kind")
    if data.get("machine_verdict") != "PASS":
        raise ContractViolation("validate receipt machine_verdict is not PASS")
    recorded = data.get("input_sha256")
    if not isinstance(recorded, Mapping):
        raise ContractViolation("validate receipt missing input_sha256")
    current, failures = record_input_roots(
        Path(repo_root).resolve(),
        tuple(run_spec.get("input_roots", ())),
    )
    if failures:
        raise ContractViolation("current input digest unavailable: " + ",".join(failures))
    recorded_normalized = {str(k): str(v) for k, v in recorded.items()}
    current_normalized = {str(k): str(v) for k, v in current.items()}
    if recorded_normalized != current_normalized:
        missing = sorted(set(current_normalized) - set(recorded_normalized))
        extra = sorted(set(recorded_normalized) - set(current_normalized))
        changed = sorted(
            key for key in set(recorded_normalized) & set(current_normalized)
            if recorded_normalized[key] != current_normalized[key]
        )
        parts = []
        if missing:
            parts.append("missing_in_receipt:%s" % ",".join(missing))
        if extra:
            parts.append("extra_in_receipt:%s" % ",".join(extra))
        if changed:
            parts.append("sha256_changed:%s" % ",".join(changed))
        raise ContractViolation("input SHA differs from validate receipt: " + ";".join(parts))
    return {
        "input_sha256": current_normalized,
        "validated_receipt": str(Path(validate_receipt_path).resolve()),
    }


def record_input_roots(repo_root: Path, input_roots: Sequence[str]) -> Tuple[Mapping[str, Any], Tuple[str, ...]]:
    records = {}
    failures = []
    repo = Path(repo_root).resolve()
    for rel in input_roots:
        raw = str(rel)
        path = _resolve_repo_input_root(repo, raw)
        if not path.exists():
            failures.append("missing_input_root:%s" % raw)
            continue
        records[raw] = _input_record_digest(structural_path_record(path))
    return records, tuple(failures)


def _resolve_repo_input_root(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        raise ContractViolation("input_root escapes repository root: %s" % raw)
    return resolved


def structural_path_record(path: Path) -> Mapping[str, Any]:
    _reject_symlink(path)
    if path.is_file():
        return file_manifest_entry(path)
    if path.is_dir():
        files = []
        directories = []
        for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
            _reject_symlink(child)
            rel = child.relative_to(path).as_posix()
            if child.is_dir():
                directories.append(rel)
            elif child.is_file():
                entry = dict(file_manifest_entry(child))
                entry["path"] = rel
                files.append(entry)
        payload = {
            "manifest_version": "rq015a-input-file-manifest-v1",
            "digest_policy": input_digest_policy(),
            "directories": directories,
            "files": files,
        }
        hash_payload = _directory_manifest_hash_payload(files)
        digest = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "kind": "directory_file_manifest",
            "manifest_version": payload["manifest_version"],
            "digest_policy": payload["digest_policy"],
            "entries": len(files),
            "directories": len(directories),
            "manifest_sha256": digest,
            "files": files,
        }
    raise ContractViolation("unsupported input path kind: %s" % path)


def input_digest_policy() -> Mapping[str, Any]:
    return {
        "file_policy": "sha256_full_file",
        "small_file_policy": "sha256_full_file",
        "large_file_policy": "sha256_full_file",
        "directory_manifest_sha256_policy": "content_only_file_path_size_sha256",
        "directory_manifest_sha256_includes": ["path", "bytes", "sha256"],
        "directory_manifest_entry_metadata_excluded": ["kind", "hash_policy", "mtime_ns"],
        "directory_manifest_non_file_metadata_excluded": [
            "manifest_version", "directories", "digest_policy",
        ],
        "symlink_policy": "reject_all_symlinks",
        "directory_symlink_policy": "reject",
        "file_symlink_policy": "reject",
        "chunk_bytes": MANIFEST_HASH_CHUNK_BYTES,
    }


def _directory_manifest_hash_payload(files: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "files": [
            {
                "path": entry["path"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            for entry in files
        ],
    }


def _input_record_digest(record: Mapping[str, Any]) -> str:
    if record.get("kind") == "directory_file_manifest":
        digest = record.get("manifest_sha256")
    elif record.get("kind") == "file":
        digest = record.get("sha256")
    else:
        raise ContractViolation("unsupported input record kind: %r" % record.get("kind"))
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ContractViolation("input record missing SHA-256 digest")
    return digest


def file_manifest_entry(path: Path) -> Mapping[str, Any]:
    _reject_symlink(path)
    st = path.stat()
    out = {
        "kind": "file",
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "hash_policy": "sha256_full_file",
        "sha256": sha256_file(path),
    }
    return out


def _reject_symlink(path: Path) -> None:
    if Path(path).is_symlink():
        raise ContractViolation("input symlink is forbidden: %s" % path)


def sha256_file(path: Path) -> str:
    _reject_symlink(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MANIFEST_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def run_contract_fixtures(
    repo_root: Path,
    fixture_paths: Optional[Sequence[str]] = None,
    actual_fixture_paths: Optional[Sequence[str]] = None,
) -> Mapping[str, Any]:
    if fixture_paths is None:
        raise ContractViolation("fixture paths must be read from run spec")
    expected = tuple(_normalize_repo_relative_string(path) for path in fixture_paths)
    actual = (
        tuple(_normalize_repo_relative_string(path) for path in actual_fixture_paths)
        if actual_fixture_paths is not None
        else expected
    )
    failures: List[str] = []
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing=%s" % ",".join(missing))
        if extra:
            parts.append("extra=%s" % ",".join(extra))
        failures.append("fixture_set_mismatch:" + ";".join(parts))

    per_file = {}
    commands = []
    output_last_line = ""
    total_passed = 0
    for rel_path in actual:
        abs_path = canonical_repo_path(repo_root, rel_path)
        try:
            abs_path.relative_to(Path(repo_root).resolve())
        except ValueError:
            failures.append("fixture_escapes_repo:%s" % rel_path)
            continue
        cmd = [sys.executable, "-m", "pytest", str(abs_path), "-q"]
        commands.append(cmd)
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        summary = parse_pytest_summary_counts(proc.stdout)
        passed = int(summary.get("passed", 0))
        total_passed += passed
        last_line = _last_line(proc.stdout)
        output_last_line = last_line or output_last_line
        file_result = {
            "command": cmd,
            "returncode": proc.returncode,
            "passed": passed,
            "failed": int(summary.get("failed", 0)),
            "errors": int(summary.get("errors", 0)),
            "summary_found": bool(summary.get("summary_found", False)),
            "output_last_line": last_line,
        }
        per_file[rel_path] = file_result
        if proc.returncode != 0:
            failures.append("fixture_failed:%s:%s" % (rel_path, last_line))
        if not summary.get("summary_found", False):
            failures.append("fixture_summary_missing:%s" % rel_path)

    aggregate_command = [sys.executable, "-m", "pytest"] + [
        str(canonical_repo_path(repo_root, path)) for path in actual
    ] + ["-q"]
    has_failures = bool(failures) or any(
        result["returncode"] != 0 for result in per_file.values()
    )
    return {
        "command": aggregate_command,
        "commands": commands,
        "expected_files": list(expected),
        "actual_files": list(actual),
        "per_file": per_file,
        "total_passed": total_passed,
        "passed": total_passed,
        "has_failures": has_failures,
        "failures": failures,
        "output_last_line": output_last_line,
    }


def parse_pytest_summary_counts(output: str) -> Mapping[str, Any]:
    summary = {"passed": 0, "failed": 0, "errors": 0, "summary_found": False}
    for line in reversed(output.splitlines()):
        matches = re.findall(
            r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)\b",
            line,
        )
        if not matches:
            continue
        summary["summary_found"] = True
        for count, label in matches:
            key = "errors" if label in ("error", "errors") else label
            summary[key] = int(summary.get(key, 0)) + int(count)
        return summary
    return summary


def parse_pytest_pass_count(output: str) -> int:
    for line in reversed(output.splitlines()):
        match = re.search(r"(\d+)\s+passed\b", line)
        if match:
            return int(match.group(1))
    raise ContractViolation("could not parse pytest passed count")


def _last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _structural_columns_for_artifact(artifact: Mapping[str, Any]) -> Tuple[str, ...]:
    cols = set(artifact.get("row_key_fields", ()))
    if artifact.get("rq007_split_applicable") is True:
        split_column = artifact.get("split_filter_column")
        if split_column:
            cols.add(str(split_column))
    if artifact.get("artifact_id") == "interhub_sigma01_hw4_timeseries":
        cols.add("scene_unique_id")
        cols.add("frame_index")
    if artifact.get("artifact_id") == "rq009_feature_matrix":
        cols.add("case_key")
        cols.add("fold")
        cols.add("source_dataset")
    if artifact.get("artifact_id") == "onsite_dense_timeseries":
        cols.add("case_key")
        cols.add("frame_index")
        cols.add("timestamp_ms")
    return tuple(cols)


def _assert_structural_columns(columns: Sequence[str]) -> None:
    assert_structural_columns_are_safe(columns)


def _assert_validate_splits(artifact_id: str, splits: Sequence[str]) -> None:
    bad_fold = [split for split in splits if split in RQ009_FOLD_NAMES]
    if bad_fold:
        raise ContractViolation("%s used RQ009 fold as split: %s" %
                                (artifact_id, ", ".join(bad_fold)))
    bad = [split for split in splits if split not in RQ007_VALIDATE_SPLITS]
    if bad:
        raise ContractViolation("%s split allowlist must be development/guard only: %s" %
                                (artifact_id, ", ".join(bad)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--run-spec", default=DEFAULT_RUN_SPEC)
    parser.add_argument("--authorization", default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = run_validate_only(
        Path(args.repo_root),
        Path(args.schema),
        Path(args.run_spec),
        Path(args.receipt),
        Path(args.authorization),
    )
    print(
        "RQ015A_VALIDATE_ONLY: machine_verdict=%s fixture_total_passed=%s"
        % (result.machine_verdict, result.metadata["contract_fixtures"]["total_passed"])
    )
    return 0 if result.machine_verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
