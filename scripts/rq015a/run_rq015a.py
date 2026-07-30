#!/usr/bin/env python3
"""Unique RQ015A entrypoint for validate-only and execute modes.

The current repository state is BUILD_WHILE_DENY.  Execute mode therefore
fails closed at the authorization gate before any MeasurementReader can be
constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import receipt
import validate_only
from rq015a_contracts import ContractViolation, SCHEMA_VERSION, load_schema


DEFAULT_RUN_SPEC = "reports/plans/RQ015A_run_spec_v4_20260730.json"
DEFAULT_SCHEMA = "reports/plans/RQ015A_ledger_schema_v2.json"
DEFAULT_AUTHORIZATION = "configs/research_authorization.json"
OPERATION_ID = "rq015a_concentration_audit"

OPTIONAL_ENV_MODULES = ("scipy", "pytest", "pyarrow", "fastparquet")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ015A guarded entrypoint")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="run structural validation and write validate receipt")
    mode.add_argument("--execute", action="store_true",
                      help="attempt authorized execution; denied while authorization is false")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()))
    parser.add_argument("--run-spec", default=DEFAULT_RUN_SPEC)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--authorization", default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--receipt", default=None,
                        help="explicit receipt path; must not already exist")
    parser.add_argument("--validate-receipt", default=None,
                        help="validate_receipt.json required before --execute")
    parser.add_argument("--output-root", default=None,
                        help="explicit output root for generated receipt paths; must not already exist")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    try:
        if args.validate_only:
            result = _run_validate_only(args, repo_root)
            return 0 if result.machine_verdict == "PASS" else 1
        return _run_execute(args, repo_root)
    except ContractViolation as exc:
        print("RQ015A_ENTRYPOINT_FAIL: %s" % exc, file=sys.stderr)
        return 1


def _run_validate_only(args: argparse.Namespace, repo_root: Path) -> receipt.ValidationReceipt:
    run_spec_path = _resolve(repo_root, args.run_spec)
    authorization_path = _resolve(repo_root, args.authorization)
    schema_path = _resolve(repo_root, args.schema)
    run_spec = _load_json(run_spec_path)
    receipt.assert_run_spec_authorization_binding(
        repo_root,
        run_spec_path,
        authorization_path,
        OPERATION_ID,
    )
    receipt_path = _receipt_path(args, repo_root, run_spec, "validate_receipt.json")

    validate_only.assert_validate_only_runtime_contract()
    environment = _runtime_environment(run_spec, validate_phase=True)
    schema = load_schema(schema_path)
    input_roots = tuple(run_spec.get("input_roots", ()))
    plan = validate_only.build_structural_scan_plan(schema, input_roots=input_roots)

    manifest_result = validate_only.verify_run_spec_checksum_manifest(
        repo_root,
        run_spec_path,
        run_spec,
    )
    fixture_result = validate_only.run_contract_fixtures(
        repo_root,
        validate_only.fixture_paths_from_run_spec(run_spec),
    )
    root_records, root_failures = validate_only.record_input_roots(repo_root, input_roots)
    failure_reasons = validate_only._validate_only_failure_reasons(
        root_failures,
        manifest_result,
        fixture_result,
    )
    checks = receipt.build_receipt_checks_from_schema(
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
        parquet_engine=environment["parquet_engine"],
        m4_only_channel_excluded=True,
        aggregation_key_derivation=(
            "v1: perspective/configuration are supplied by W1 adapter; "
            "validate-only only asserts this description is present."
        ),
        reads_measurement_fields=validate_only.READS_MEASUREMENT_FIELDS,
        failure_reasons=failure_reasons,
    )
    metadata = {
        "schema_load_self_check": schema.get("schema_id") == SCHEMA_VERSION,
        "must_precede_execute": validate_only.MUST_PRECEDE_EXECUTE,
        "run_spec_execution_authorized": run_spec.get("execution_authorized"),
        "contract_fixtures": fixture_result,
        "checksum_manifest": manifest_result,
        "structural_requested_columns": {
            k: list(v.columns) for k, v in sorted(plan.requested_columns.items())
        },
        "runtime_environment": environment,
        "factor_analysis": {"bootstrap_counts_recordable": True},
    }
    result = receipt.build_validate_receipt(checks, metadata=metadata)
    receipt.write_receipt(receipt_path, result)
    print(
        "RQ015A_VALIDATE_ONLY: machine_verdict=%s fixture_total_passed=%s"
        % (result.machine_verdict, fixture_result["total_passed"])
    )
    return result


def _run_execute(args: argparse.Namespace, repo_root: Path) -> int:
    run_spec_path = _resolve(repo_root, args.run_spec)
    authorization_path = _resolve(repo_root, args.authorization)
    run_spec = _load_json(run_spec_path)
    ledger = importlib.import_module("build_ledger")
    try:
        receipt.assert_run_spec_authorization_binding(
            repo_root,
            run_spec_path,
            authorization_path,
            OPERATION_ID,
        )
        if not args.validate_receipt:
            raise ContractViolation("validate receipt is required before execute")
        validate_receipt_path = _resolve(repo_root, args.validate_receipt)
        ledger.load_execute_permit(run_spec_path, authorization_path, repo_root=repo_root)
        validate_only.assert_validate_receipt_inputs_current(
            repo_root,
            run_spec,
            validate_receipt_path,
        )
    except ContractViolation as exc:
        if args.receipt:
            _write_execute_fail_receipt(args, repo_root, str(exc))
        print("RQ015A_EXECUTE_DENIED: %s" % exc, file=sys.stderr)
        return 1

    raise ContractViolation(
        "execution permit unexpectedly succeeded in BUILD_WHILE_DENY; "
        "refusing to run audit without PI-reviewed post-authorization handoff"
    )


def _write_execute_fail_receipt(args: argparse.Namespace, repo_root: Path, reason: str) -> None:
    run_spec = _load_json(_resolve(repo_root, args.run_spec))
    schema = load_schema(_resolve(repo_root, args.schema))
    environment = _runtime_environment(run_spec, validate_phase=False)
    checks = receipt.build_receipt_checks_from_schema(
        schema,
        per_artifact_conservation={},
        held_out_parsed_rows=0,
        held_out_conclusion_rows=0,
        duplicate_primary_keys=0,
        unmapped_measurement_roles=0,
        k_unknown_rows=0,
        bins_stability_verdict="BINS_WITHHELD_UNSTABLE",
        c0_routing_stability={},
        input_sha256={},
        parquet_engine=environment["parquet_engine"],
        m4_only_channel_excluded=True,
        aggregation_key_derivation="not_run: execute permit denied before any reader construction.",
        reads_measurement_fields=False,
        failure_reasons=("execute_permit_denied:%s" % reason,),
    )
    metadata = {
        "runtime_environment": environment,
        "execute_permit_denied": True,
        "measurement_reader_constructed": False,
        "factor_analysis": {
            "results": [],
            "bootstrap_counts_recordable": True,
        },
    }
    receipt.write_receipt(
        _receipt_path(args, repo_root, run_spec, "run_receipt.json"),
        receipt.build_run_receipt(checks, metadata=metadata),
    )


def _runtime_environment(run_spec: Mapping[str, Any], validate_phase: bool) -> Mapping[str, Any]:
    env = run_spec.get("environment") or {}
    required = list(env.get("required_modules") or ())
    if validate_phase:
        required.extend(env.get("validate_phase_only") or ())
    missing = []
    versions: Dict[str, str] = {}
    for name in sorted(set(required).union(OPTIONAL_ENV_MODULES)):
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = "NOT_INSTALLED"
            if name in required:
                missing.append(name)
            continue
        versions[name] = str(getattr(module, "__version__", "stdlib"))
    if missing:
        raise ContractViolation("missing required modules: " + ",".join(sorted(missing)))

    min_python = str(env.get("min_python") or "3.9")
    if _version_tuple(platform.python_version()) < _version_tuple(min_python):
        raise ContractViolation("python %s is below required %s" %
                                (platform.python_version(), min_python))
    parquet_engine = receipt.detect_parquet_engine()
    return {
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "module_versions": versions,
        "parquet_engine": parquet_engine,
    }


def _version_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def _receipt_path(
    args: argparse.Namespace,
    repo_root: Path,
    run_spec: Mapping[str, Any],
    filename: str,
) -> Path:
    if args.receipt:
        return receipt.require_new_json_path(_resolve(repo_root, args.receipt)).path
    output_root = _output_root(args, repo_root, run_spec)
    if output_root.exists():
        raise ContractViolation("output root already exists: %s" % output_root)
    return receipt.require_new_json_path(output_root / filename).path


def _output_root(args: argparse.Namespace, repo_root: Path, run_spec: Mapping[str, Any]) -> Path:
    if args.output_root:
        return _resolve(repo_root, args.output_root)
    template = str(run_spec.get("output_root") or "")
    if not template:
        raise ContractViolation("run spec missing output_root")
    rendered = template.replace("<UTC>", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    rendered = rendered.replace("<planSHA8>", _plan_sha8(repo_root, run_spec))
    return _resolve(repo_root, rendered)


def _plan_sha8(repo_root: Path, run_spec: Mapping[str, Any]) -> str:
    plan_rel = ((run_spec.get("bound_artifacts") or {}).get("plan"))
    if not plan_rel:
        return "unknown0"
    plan_path = _resolve(repo_root, str(plan_rel))
    h = hashlib.sha256()
    with plan_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _load_json(path: Path) -> Mapping[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ContractViolation("expected JSON object: %s" % path)
    return payload


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
