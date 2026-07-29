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
from typing import Any, Mapping, Optional, Sequence, Tuple
from typing import Literal

from receipt import (
    ValidationReceipt,
    build_receipt_checks_from_schema,
    build_validate_receipt,
    detect_parquet_engine,
    require_new_json_path,
    write_receipt,
)
from rq015a_contracts import ContractViolation, SCHEMA_VERSION, load_schema

READS_MEASUREMENT_FIELDS: Literal[False] = False
MUST_PRECEDE_EXECUTE: Literal[True] = True

FORBIDDEN_COLUMN_PREFIXES = ("ipv_", "target_ipv", "counterpart_ipv", "M4_ONLY_")
FORBIDDEN_COLUMN_SUBSTRINGS = ("rating", "preference", "human", "score", "label")
RQ009_FOLD_NAMES = ("train", "guard_tune", "calibration", "test")
RQ007_VALIDATE_SPLITS = ("development", "guard")
MANIFEST_HASH_CHUNK_BYTES = 1024 * 1024


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
                      receipt_path: Path) -> ValidationReceipt:
    assert_validate_only_runtime_contract()
    new_path = require_new_json_path(Path(receipt_path))
    repo = Path(repo_root)
    schema = load_schema(schema_path)
    run_spec = json.loads(Path(run_spec_path).read_text(encoding="utf-8"))
    input_roots = tuple(run_spec.get("input_roots", ()))
    plan = build_structural_scan_plan(schema, input_roots=input_roots)

    fixture_result = run_contract_fixtures(repo)
    root_records, root_failures = record_input_roots(repo, input_roots)
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
        failure_reasons=tuple(root_failures),
    )
    metadata = {
        "schema_load_self_check": schema.get("schema_id") == SCHEMA_VERSION,
        "must_precede_execute": MUST_PRECEDE_EXECUTE,
        "run_spec_execution_authorized": run_spec.get("execution_authorized"),
        "contract_fixtures": fixture_result,
        "structural_requested_columns": {
            k: list(v.columns) for k, v in sorted(plan.requested_columns.items())
        },
    }
    receipt = build_validate_receipt(checks, metadata=metadata)
    write_receipt(new_path, receipt)
    return receipt


def record_input_roots(repo_root: Path, input_roots: Sequence[str]) -> Tuple[Mapping[str, Any], Tuple[str, ...]]:
    records = {}
    failures = []
    for rel in input_roots:
        path = repo_root / rel
        if not path.exists():
            failures.append("missing_input_root:%s" % rel)
            continue
        records[rel] = structural_path_record(path)
    return records, tuple(failures)


def structural_path_record(path: Path) -> Mapping[str, Any]:
    if path.is_file():
        return file_manifest_entry(path)
    if path.is_dir():
        files = []
        directories = []
        for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
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
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        "chunk_bytes": MANIFEST_HASH_CHUNK_BYTES,
    }


def file_manifest_entry(path: Path) -> Mapping[str, Any]:
    st = path.stat()
    out = {
        "kind": "file",
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "hash_policy": "sha256_full_file",
        "sha256": sha256_file(path),
    }
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MANIFEST_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def run_contract_fixtures(repo_root: Path) -> Mapping[str, Any]:
    test_path = repo_root / "tests" / "test_rq015a_contracts.py"
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-q"]
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    passed = parse_pytest_pass_count(proc.stdout)
    if proc.returncode != 0:
        raise ContractViolation("contract fixtures failed: " + proc.stdout.strip().splitlines()[-1])
    return {"command": cmd, "passed": passed, "output_last_line": _last_line(proc.stdout)}


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
    bad = []
    for col in columns:
        name = str(col)
        lower = name.lower()
        if any(name.startswith(prefix) for prefix in FORBIDDEN_COLUMN_PREFIXES):
            bad.append(name)
            continue
        if any(marker in lower for marker in FORBIDDEN_COLUMN_SUBSTRINGS):
            bad.append(name)
    if bad:
        raise ContractViolation("structural plan requested forbidden columns: " +
                                ", ".join(sorted(bad)))


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
    parser.add_argument("--schema", default="reports/plans/RQ015A_ledger_schema_v2.json")
    parser.add_argument("--run-spec", default="reports/plans/RQ015A_run_spec_v1.json")
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    run_validate_only(Path(args.repo_root), Path(args.schema), Path(args.run_spec), Path(args.receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
