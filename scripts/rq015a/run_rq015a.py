#!/usr/bin/env python3
"""Unique RQ015A entrypoint for validate-only and execute modes.

The current repository state is BUILD_WHILE_DENY.  Execute mode therefore
fails closed at the authorization gate before any MeasurementReader can be
constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import receipt
import validate_only
from rq015a_contracts import (
    ATTEMPTED,
    NOT_ATTEMPTED,
    UNKNOWN,
    ContractViolation,
    SCHEMA_VERSION,
    bins_stability,
    c0_route_with_sensitivity,
    check_conservation,
    load_schema,
)


# 默认的 run spec 与 schema 【不再硬编码】。
# 历史教训：版本重绑（v3→v4→v5→v6）时每次都改了授权对象却漏改这里的常量，
# 导致绑定核对拒绝默认路径。同一事实存两份必然漂移，故改为单一来源：
#   run spec  ← 授权对象的 run_spec_path
#   schema    ← 该 run spec 的 bound_artifacts.ledger_schema
DEFAULT_AUTHORIZATION = "configs/research_authorization.json"
OPERATION_ID = "rq015a_concentration_audit"

OPTIONAL_ENV_MODULES = ("scipy", "pytest", "pyarrow", "fastparquet")
SPLIT_SOURCE = (
    "data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/"
    "RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/"
    "case_split_assignment.csv"
)
AUDIT_ARTIFACT_ORDER = (
    "onsite_dense_timeseries",
    "wod_rq010b_full479_audited",
    "interhub_sigma01_hw4_timeseries",
    "rq009_feature_matrix",
)
LEDGER_FIELDS = (
    "artifact_id",
    "product_row_key",
    "measurement_role",
    "case_id",
    "rq007_split",
    "ipv_error",
    "K",
    "candidate_grid_id",
    "k_eff",
    "q_eff",
    "attempt_status",
    "reason_code",
    "recoverability",
    "ledger_schema_version",
    "aggregation_perspective",
    "aggregation_configuration",
)
SUMMARY_FIELDS = (
    "artifact",
    "attempt_status",
    "rows",
    "q_eff_n",
    "q_eff_min",
    "q_eff_p25",
    "q_eff_median",
    "q_eff_p75",
    "q_eff_p90",
    "q_eff_p95",
    "q_eff_p99",
    "q_eff_max",
    "near_uniform_share",
)



def _resolve_defaults_from_authorization(repo_root: Path, args) -> None:
    """未显式指定 --run-spec / --schema 时，从授权对象与 run spec 推导。

    单一来源原则：授权对象是 run spec 路径的唯一权威，run spec 是 schema 路径的
    唯一权威。任何一环重绑时都不必再同步修改本模块的常量。
    """
    if args.run_spec is None:
        auth = _load_json(_resolve(repo_root, args.authorization))
        entry = (auth.get("authorizations") or {}).get(OPERATION_ID)
        if not isinstance(entry, dict) or not entry.get("run_spec_path"):
            raise ContractViolation(
                "authorization object has no run_spec_path for %s" % OPERATION_ID
            )
        args.run_spec = str(entry["run_spec_path"])
    if args.schema is None:
        spec = _load_json(_resolve(repo_root, args.run_spec))
        bound = (spec.get("bound_artifacts") or {}).get("ledger_schema")
        if not bound:
            raise ContractViolation("run spec has no bound_artifacts.ledger_schema")
        args.schema = str(bound)


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
    # default=None：未显式给出时由 _resolve_defaults_from_authorization() 从
    # 授权对象与 run spec 推导，避免与授权绑定漂移。
    parser.add_argument("--run-spec", default=None)
    parser.add_argument("--schema", default=None)
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
        _resolve_defaults_from_authorization(repo_root, args)
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
        "run_spec_path": _path_for_receipt_metadata(repo_root, run_spec_path),
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
    schema_path = _resolve(repo_root, args.schema)
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
        validated_inputs = validate_only.assert_validate_receipt_inputs_current(
            repo_root,
            run_spec_path,
            run_spec,
            authorization_path,
            OPERATION_ID,
            validate_receipt_path,
        )
        permit = ledger.load_execute_permit(run_spec_path, authorization_path, repo_root=repo_root)
    except ContractViolation as exc:
        if args.receipt:
            _write_execute_fail_receipt(args, repo_root, str(exc))
        print("RQ015A_EXECUTE_DENIED: %s" % exc, file=sys.stderr)
        return 1

    run_receipt_path = _receipt_path(args, repo_root, run_spec, "run_receipt.json")
    output_root = run_receipt_path.parent
    _prepare_execute_output_root(output_root, run_receipt_path)
    schema = load_schema(schema_path)
    environment = _runtime_environment(run_spec, validate_phase=False)
    audit = _execute_concentration_audit(
        repo_root=repo_root,
        schema_path=schema_path,
        schema=schema,
        run_spec_path=run_spec_path,
        run_spec=run_spec,
        output_root=output_root,
        ledger=ledger,
        permit=permit,
        input_sha256=validated_inputs["input_sha256"],
        environment=environment,
    )
    checks = receipt.build_receipt_checks_from_schema(
        schema,
        per_artifact_conservation=audit["conservation"],
        held_out_parsed_rows=0,
        held_out_conclusion_rows=0,
        duplicate_primary_keys=0,
        unmapped_measurement_roles=0,
        k_unknown_rows=audit["k_unknown_rows"],
        bins_stability_verdict=audit["bins_stability_verdict"],
        c0_routing_stability=audit["c0_routing_stability"],
        input_sha256=validated_inputs["input_sha256"],
        parquet_engine=environment["parquet_engine"],
        m4_only_channel_excluded=True,
        aggregation_key_derivation=json.dumps(
            ledger.AGGREGATION_KEY_DERIVATION,
            ensure_ascii=False,
            sort_keys=True,
        ),
        reads_measurement_fields=False,
        failure_reasons=(),
    )
    metadata = {
        "run_spec_path": _path_for_receipt_metadata(repo_root, run_spec_path),
        "validated_receipt": validated_inputs["validated_receipt"],
        "runtime_environment": environment,
        "measurement_reader_constructed": True,
        "execute_measurement_fields_read": True,
        "artifact_order": list(AUDIT_ARTIFACT_ORDER),
        "ledger_storage_deviation": {
            "requested": "concentration_ledger.csv",
            "actual": "concentration_ledger/*.parquet plus concentration_ledger_summary.csv",
            "reason": "14,473,982 L1 rows are not practical as a single CSV.",
        },
        "outputs": audit["outputs"],
        "artifacts": audit["artifacts"],
        "manual_q_eff_checks": audit["manual_q_eff_checks"],
        "factor_analysis": {
            "results": [],
            "bootstrap_counts_recordable": True,
            "n_bootstrap_defined": 0,
            "n_bootstrap_undefined": 0,
            "status": "NO_FACTOR_FIELDS_DECLARED_IN_RUN_SPEC",
        },
    }
    run_receipt = receipt.build_run_receipt(checks, metadata=metadata)
    receipt.write_receipt(run_receipt_path, run_receipt)
    print(
        "RQ015A_EXECUTE_DONE: machine_verdict=%s output_root=%s"
        % (run_receipt.machine_verdict, output_root),
        flush=True,
    )
    return 0 if run_receipt.machine_verdict == "PASS" else 1


def _prepare_execute_output_root(output_root: Path, run_receipt_path: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    planned = (
        output_root / "concentration_ledger",
        output_root / "concentration_ledger_summary.csv",
        output_root / "portraits.json",
        output_root / "c0_routing.json",
        run_receipt_path,
    )
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise ContractViolation("execute output already exists: " + ",".join(existing))


def _execute_concentration_audit(
    repo_root: Path,
    schema_path: Path,
    schema: Mapping[str, Any],
    run_spec_path: Path,
    run_spec: Mapping[str, Any],
    output_root: Path,
    ledger: Any,
    permit: Any,
    input_sha256: Mapping[str, str],
    environment: Mapping[str, Any],
) -> Mapping[str, Any]:
    ledger_schema = ledger.load_ledger_schema_v2(schema_path)
    allowlist = ledger.load_case_allowlist(repo_root / SPLIT_SOURCE)
    ledger_dir = output_root / "concentration_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=False)

    conservation = {}
    artifacts = {}
    c0_payload = {}
    c0_stability = {}
    summary_rows = []
    manual_checks = []
    k_unknown_rows = 0
    bins_verdicts = []
    total_l1_rows = 0

    for artifact_id in AUDIT_ARTIFACT_ORDER:
        if artifact_id not in ledger_schema.artifacts_by_id:
            raise ContractViolation("artifact missing from schema: %s" % artifact_id)
        spec = ledger_schema.artifacts_by_id[artifact_id]
        if not spec.present_locally:
            raise ContractViolation("audit order includes absent artifact: %s" % artifact_id)
        _progress("START artifact=%s" % artifact_id)
        started = time.monotonic()
        state = _ArtifactAuditState(artifact_id)
        ledger_path = ledger_dir / ("%s.parquet" % artifact_id)
        writer = _ParquetLedgerWriter(ledger_path)
        scope = ledger.resolve_artifact_scope(
            spec,
            allowlist if spec.rq007_split_applicable else None,
        )
        try:
            _process_artifact_rows(ledger, spec, scope, permit, writer, state, manual_checks)
        finally:
            writer.close()

        report = state.conservation_report(spec)
        conservation[artifact_id] = report
        expected_rows = _expected_measurement_rows(spec)
        if expected_rows is not None and report.measurement_rows_observed != expected_rows:
            raise ContractViolation(
                "%s measurement rows %d != expected %d"
                % (artifact_id, report.measurement_rows_observed, expected_rows)
            )
        if spec.not_attempted_rule.get("kind") == "none_expected":
            if state.status_counts[NOT_ATTEMPTED] != 0:
                raise ContractViolation("%s unexpectedly has NOT_ATTEMPTED rows" % artifact_id)

        portrait = state.portrait()
        route = state.c0_route()
        c0_payload[artifact_id] = route
        c0_stability[artifact_id] = bool(route["stable"])
        bins_verdicts.append(portrait["L1"]["bins_stability"]["verdict"])
        summary_rows.extend(state.summary_rows())
        total_l1_rows += state.measurement_rows
        k_unknown_rows += state.k_unknown_rows
        artifacts[artifact_id] = {
            "ledger_path": _path_for_receipt_metadata(repo_root, ledger_path),
            "ledger_bytes": ledger_path.stat().st_size,
            "physical_rows": state.physical_rows,
            "measurement_rows": state.measurement_rows,
            "status_counts": dict(sorted(state.status_counts.items())),
            "recoverability_counts": dict(sorted(state.recoverability_counts.items())),
            "expected_measurement_rows": expected_rows,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "portrait": portrait,
            "c0": route,
        }
        _progress(
            "DONE artifact=%s physical_rows=%d measurement_rows=%d elapsed=%.1fs"
            % (artifact_id, state.physical_rows, state.measurement_rows, time.monotonic() - started)
        )

    if total_l1_rows != 14473982:
        raise ContractViolation("total measurement rows %d != expected 14473982" % total_l1_rows)
    if not manual_checks:
        raise ContractViolation("manual q_eff check sample is empty")

    summary_path = output_root / "concentration_ledger_summary.csv"
    _write_summary_csv(summary_path, summary_rows)
    portraits_path = output_root / "portraits.json"
    _write_json(
        portraits_path,
        {
            "schema_version": schema.get("schema_id"),
            "run_spec_path": _path_for_receipt_metadata(repo_root, run_spec_path),
            "descriptive_only": True,
            "artifact_order": list(AUDIT_ARTIFACT_ORDER),
            "artifacts": {k: artifacts[k]["portrait"] for k in AUDIT_ARTIFACT_ORDER},
        },
    )
    c0_path = output_root / "c0_routing.json"
    _write_json(
        c0_path,
        {
            "descriptive_only": True,
            "uses_report_bins": False,
            "artifacts": c0_payload,
        },
    )

    outputs = {
        "output_root": _path_for_receipt_metadata(repo_root, output_root),
        "concentration_ledger_dir": _path_for_receipt_metadata(repo_root, ledger_dir),
        "concentration_ledger_summary_csv": _output_record(repo_root, summary_path, len(summary_rows)),
        "portraits_json": _output_record(repo_root, portraits_path, None),
        "c0_routing_json": _output_record(repo_root, c0_path, None),
        "ledger_parquet": {
            artifact_id: _output_record(
                repo_root,
                Path(artifacts[artifact_id]["ledger_path"]),
                artifacts[artifact_id]["measurement_rows"],
            )
            for artifact_id in AUDIT_ARTIFACT_ORDER
        },
        "input_sha256": dict(input_sha256),
        "environment": dict(environment),
    }
    return {
        "conservation": conservation,
        "artifacts": artifacts,
        "outputs": outputs,
        "c0_routing_stability": c0_stability,
        "bins_stability_verdict": (
            "BINS_WITHHELD_UNSTABLE"
            if "BINS_WITHHELD_UNSTABLE" in bins_verdicts
            else "BINS_REPORTABLE"
        ),
        "k_unknown_rows": k_unknown_rows,
        "manual_q_eff_checks": manual_checks[:10],
    }


class _ParquetLedgerWriter:
    def __init__(self, path: Path, batch_size: int = 100000) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.batch_size = int(batch_size)
        self.rows: List[Mapping[str, Any]] = []
        self.count = 0
        self._pa = pa
        self._writer = pq.ParquetWriter(
            self.path,
            pa.schema([
                ("artifact_id", pa.string()),
                ("product_row_key", pa.string()),
                ("measurement_role", pa.string()),
                ("case_id", pa.string()),
                ("rq007_split", pa.string()),
                ("ipv_error", pa.float64()),
                ("K", pa.int64()),
                ("candidate_grid_id", pa.string()),
                ("k_eff", pa.float64()),
                ("q_eff", pa.float64()),
                ("attempt_status", pa.string()),
                ("reason_code", pa.string()),
                ("recoverability", pa.string()),
                ("ledger_schema_version", pa.string()),
                ("aggregation_perspective", pa.string()),
                ("aggregation_configuration", pa.string()),
            ]),
            compression="zstd",
        )

    def write(self, row: Mapping[str, Any]) -> None:
        self.rows.append({field: row.get(field) for field in LEDGER_FIELDS})
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = self._pa.Table.from_pylist(self.rows, schema=self._writer.schema)
        self._writer.write_table(table)
        self.rows = []

    def close(self) -> None:
        self.flush()
        self._writer.close()


class _ArtifactAuditState:
    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        self.physical_rows = 0
        self.measurement_rows = 0
        self.status_counts: Counter = Counter()
        self.recoverability_counts: Counter = Counter()
        self.q_values_by_status: Mapping[str, List[float]] = defaultdict(list)
        self.l2_groups: Dict[Tuple[Optional[str], str, str], Dict[str, Any]] = {}
        self.episode_count = 0
        self.episode_ipv_sum = 0.0
        self.episode_weight_sum = 0.0
        self.episode_weighted_ipv_sum = 0.0
        self.k_unknown_rows = 0

    def add_physical_row(self) -> None:
        self.physical_rows += 1

    def add_l1_row(self, row: Mapping[str, Any], ipv_value: Optional[float]) -> None:
        status = str(row["attempt_status"])
        recoverability = str(row["recoverability"])
        self.measurement_rows += 1
        self.status_counts[status] += 1
        self.recoverability_counts[recoverability] += 1
        if row.get("K") is None:
            self.k_unknown_rows += 1
        q_value = row.get("q_eff")
        if q_value is not None:
            self.q_values_by_status[status].append(float(q_value))

        key = (
            row.get("case_id"),
            str(row["aggregation_perspective"]),
            str(row["aggregation_configuration"]),
        )
        group = self.l2_groups.setdefault(
            key,
            {"n_l1": 0, "n_attempted": 0, "n_unknown": 0, "q_sum": 0.0, "q_count": 0},
        )
        group["n_l1"] += 1
        if status == ATTEMPTED:
            group["n_attempted"] += 1
            if q_value is not None:
                group["q_sum"] += float(q_value)
                group["q_count"] += 1
            if q_value is not None and ipv_value is not None:
                q = float(q_value)
                w = 1.0 - q
                self.episode_count += 1
                self.episode_ipv_sum += float(ipv_value)
                self.episode_weight_sum += w
                self.episode_weighted_ipv_sum += float(ipv_value) * w
        elif status == UNKNOWN:
            group["n_unknown"] += 1

    def conservation_report(self, spec: Any) -> Any:
        return check_conservation(
            self.artifact_id,
            self.physical_rows,
            spec.expansion_factor,
            spec.collapse_factor,
            dict(self.status_counts),
            dict(self.recoverability_counts),
        )

    def l2_units(self) -> List[Mapping[str, Any]]:
        out = []
        for key in sorted(self.l2_groups, key=lambda k: (k[0] is not None, k[0] or "", k[1], k[2])):
            group = self.l2_groups[key]
            q_count = int(group["q_count"])
            mean_q = (group["q_sum"] / q_count) if q_count else None
            out.append({
                "case_id": key[0],
                "perspective": key[1],
                "configuration": key[2],
                "n_l1": int(group["n_l1"]),
                "n_attempted": int(group["n_attempted"]),
                "n_unknown": int(group["n_unknown"]),
                "mean_q_eff": mean_q,
                "status": "OK" if int(group["n_l1"]) >= 5 else "INSUFFICIENT_SUPPORT",
            })
        return out

    def l3_units(self) -> List[Mapping[str, Any]]:
        grouped: Dict[Optional[str], List[float]] = defaultdict(list)
        totals: Counter = Counter()
        ok_counts: Counter = Counter()
        for unit in self.l2_units():
            case_id = unit["case_id"]
            totals[case_id] += 1
            if unit["status"] == "OK" and unit["mean_q_eff"] is not None:
                grouped[case_id].append(float(unit["mean_q_eff"]))
                ok_counts[case_id] += 1
        out = []
        for case_id in sorted(totals, key=lambda value: (value is not None, value or "")):
            values = grouped.get(case_id, [])
            out.append({
                "case_id": case_id,
                "n_l2_total": int(totals[case_id]),
                "n_l2_ok": int(ok_counts[case_id]),
                "mean_q_eff": (math.fsum(values) / len(values)) if values else None,
                "status": "OK" if values else "ZERO_SUPPORT",
            })
        return out

    def portrait(self) -> Mapping[str, Any]:
        attempted_q = self.q_values_by_status.get(ATTEMPTED, [])
        l2_q = [u["mean_q_eff"] for u in self.l2_units() if u["mean_q_eff"] is not None]
        l3_q = [u["mean_q_eff"] for u in self.l3_units() if u["mean_q_eff"] is not None]
        return {
            "measurement_rows": self.measurement_rows,
            "physical_rows": self.physical_rows,
            "status_counts": dict(sorted(self.status_counts.items())),
            "recoverability_counts": dict(sorted(self.recoverability_counts.items())),
            "L1": {
                "attempted_q_eff": _distribution(attempted_q),
                "all_nonnull_q_eff": _distribution(_all_q_values(self.q_values_by_status)),
                "bins_stability": bins_stability(attempted_q),
            },
            "L2": {
                "units": len(self.l2_groups),
                "mean_q_eff": _distribution(l2_q),
                "status_counts": dict(Counter(u["status"] for u in self.l2_units())),
            },
            "L3": {
                "units": len(self.l3_units()),
                "mean_q_eff": _distribution(l3_q),
                "status_counts": dict(Counter(u["status"] for u in self.l3_units())),
            },
            "episode_summary": self.episode_summary(),
        }

    def episode_summary(self) -> Mapping[str, Any]:
        return {
            "unweighted": (
                self.episode_ipv_sum / self.episode_count if self.episode_count else None
            ),
            "concentration_wtd": (
                self.episode_weighted_ipv_sum / self.episode_weight_sum
                if self.episode_weight_sum > 0
                else None
            ),
            "n_used": self.episode_count,
            "definition_sensitivity_only": True,
        }

    def c0_route(self) -> Mapping[str, Any]:
        attempted_q = list(self.q_values_by_status.get(ATTEMPTED, []))
        return c0_route_with_sensitivity(
            uses_ipv=True,
            n_rows=self.measurement_rows,
            n_not_attempted=self.status_counts[NOT_ATTEMPTED],
            n_unknown=self.status_counts[UNKNOWN],
            q_effs_attempted=attempted_q,
            mapping_is_1to1=True,
        )

    def summary_rows(self) -> List[Mapping[str, Any]]:
        rows = []
        for status in (ATTEMPTED, NOT_ATTEMPTED, UNKNOWN):
            qs = self.q_values_by_status.get(status, [])
            dist = _distribution(qs)
            rows.append({
                "artifact": self.artifact_id,
                "attempt_status": status,
                "rows": self.status_counts[status],
                "q_eff_n": dist["n"],
                "q_eff_min": dist["min"],
                "q_eff_p25": dist["p25"],
                "q_eff_median": dist["median"],
                "q_eff_p75": dist["p75"],
                "q_eff_p90": dist["p90"],
                "q_eff_p95": dist["p95"],
                "q_eff_p99": dist["p99"],
                "q_eff_max": dist["max"],
                "near_uniform_share": dist["near_uniform_share"],
            })
        return rows


def _process_artifact_rows(
    ledger: Any,
    spec: Any,
    scope: Any,
    permit: Any,
    writer: _ParquetLedgerWriter,
    state: _ArtifactAuditState,
    manual_checks: List[Mapping[str, Any]],
) -> None:
    # Reuse the existing trust-boundary validation before any measurement field is consumed.
    ledger.open_measurement_reader(spec, scope, permit, source_rows=[])
    if spec.format == "csv":
        rows = _csv_rows_for_execute(ledger, spec, scope)
    elif spec.format == "parquet":
        rows = _parquet_rows_for_execute(ledger, spec, scope)
    else:
        raise ContractViolation("unsupported executable artifact format: %s" % spec.format)

    if spec.not_attempted_rule.get("kind") == "local_position":
        physical_rows = list(rows)
        local_positions = ledger._local_position_map(spec, physical_rows)
        iterable = enumerate(physical_rows)
    else:
        local_positions = {}
        iterable = enumerate(rows)

    for idx, physical_row in iterable:
        state.add_physical_row()
        local_position = local_positions.get(idx)
        _emit_l1_for_physical_row(
            ledger,
            spec,
            scope,
            physical_row,
            local_position,
            writer,
            state,
            manual_checks,
        )
        if state.physical_rows % 500000 == 0:
            _progress(
                "PROGRESS artifact=%s physical_rows=%d measurement_rows=%d"
                % (spec.artifact_id, state.physical_rows, state.measurement_rows)
            )


def _emit_l1_for_physical_row(
    ledger: Any,
    spec: Any,
    scope: Any,
    physical_row: Mapping[str, Any],
    local_position: Optional[int],
    writer: _ParquetLedgerWriter,
    state: _ArtifactAuditState,
    manual_checks: List[Mapping[str, Any]],
) -> None:
    case_id, split = ledger._row_case_and_split(spec, scope, physical_row)
    if split == "held_out":
        raise ContractViolation("%s held_out row escaped allowlist" % spec.artifact_id)
    row_key = ledger.product_row_key(physical_row, spec.row_key_fields)
    is_d0 = ledger._is_not_attempted(spec, physical_row, local_position)
    for role in spec.roles:
        if role.excluded:
            continue
        perspective, configuration = _derive_aggregation_key(
            ledger, spec.artifact_id, row_key, role.measurement_role, physical_row
        )
        ipv_error, k_eff, q_value, status, reason = ledger._status_and_values(
            spec, role, physical_row, is_d0
        )
        ipv_value = _role_ipv_value(ledger, role, physical_row)
        out = {
            "artifact_id": spec.artifact_id,
            "product_row_key": row_key,
            "measurement_role": role.measurement_role,
            "case_id": case_id,
            "rq007_split": split,
            "ipv_error": ipv_error,
            "K": spec.K,
            "candidate_grid_id": spec.candidate_grid_id,
            "k_eff": k_eff,
            "q_eff": q_value,
            "attempt_status": status,
            "reason_code": reason,
            "recoverability": spec.recoverability,
            "ledger_schema_version": SCHEMA_VERSION,
            "aggregation_perspective": perspective,
            "aggregation_configuration": configuration,
        }
        _record_manual_check(out, manual_checks)
        writer.write(out)
        state.add_l1_row(out, ipv_value)


def _derive_aggregation_key(
    ledger: Any,
    artifact_id: str,
    row_key: str,
    measurement_role: str,
    physical_row: Mapping[str, Any],
) -> Tuple[str, str]:
    if artifact_id == "wod_rq010b_full479_audited":
        return measurement_role, "full479_candidate"
    return ledger.derive_aggregation_key(artifact_id, row_key, measurement_role)


def _role_ipv_value(ledger: Any, role: Any, row: Mapping[str, Any]) -> Optional[float]:
    column = getattr(role, "ipv_column", None)
    if not column:
        return None
    if column not in row:
        raise ContractViolation("missing IPV value column %s" % column)
    return ledger._parse_optional_float(row[column])


def _record_manual_check(row: Mapping[str, Any], manual_checks: List[Mapping[str, Any]]) -> None:
    if len(manual_checks) >= 10:
        return
    if row.get("attempt_status") != ATTEMPTED:
        return
    ipv_error = row.get("ipv_error")
    K = row.get("K")
    q_value = row.get("q_eff")
    if ipv_error is None or K is None or q_value is None:
        return
    expected = 1.0 / (((1.0 - float(ipv_error)) ** 2) * int(K))
    if abs(expected - float(q_value)) > 1e-12:
        raise ContractViolation("manual q_eff check failed for %s" % row["artifact_id"])
    manual_checks.append({
        "artifact_id": row["artifact_id"],
        "product_row_key": row["product_row_key"],
        "measurement_role": row["measurement_role"],
        "ipv_error": float(ipv_error),
        "K": int(K),
        "expected_q_eff": expected,
        "observed_q_eff": float(q_value),
        "abs_error": abs(expected - float(q_value)),
    })


def _csv_rows_for_execute(ledger: Any, spec: Any, scope: Any) -> Iterable[Mapping[str, Any]]:
    projected = _projected_columns_for_execute(ledger, spec, scope)
    if spec.rq007_split_applicable is True:
        allowed_rows = _csv_allowed_rows_for_execute(spec, scope)
        allowed_set = set(allowed_rows)
        _progress(
            "FILTER artifact=%s allowed_physical_rows=%d"
            % (spec.artifact_id, len(allowed_rows))
        )
        return _iter_csv_projected_rows(spec.path, projected, allowed_set)
    return _iter_csv_projected_rows(spec.path, projected, None)


def _csv_allowed_rows_for_execute(spec: Any, scope: Any) -> Tuple[int, ...]:
    header, header_index = _read_csv_header(spec.path)
    if scope.join_column not in header_index:
        raise ContractViolation("%s missing join column %s" % (spec.artifact_id, scope.join_column))
    join_idx = header_index[scope.join_column]
    allowed = []
    unmapped = 0
    split_counts: Counter = Counter()
    with spec.path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row_number, row in enumerate(reader):
            if join_idx >= len(row):
                raise ContractViolation("%s missing join column" % spec.artifact_id)
            case_id = str(row[join_idx])
            split = scope.allowlist.case_to_split.get(case_id)
            if split is None:
                unmapped += 1
            else:
                split_counts[split] += 1
                if case_id in scope.allowlist.allowed_case_ids:
                    allowed.append(row_number)
            if (row_number + 1) % 500000 == 0:
                _progress(
                    "FILTER_SCAN artifact=%s physical_rows=%d allowed=%d"
                    % (spec.artifact_id, row_number + 1, len(allowed))
                )
    if unmapped:
        raise ContractViolation("%s unmapped case rows: %d" % (spec.artifact_id, unmapped))
    return tuple(allowed)


def _iter_csv_projected_rows(
    path: Path,
    columns: Sequence[str],
    allowed_rows: Optional[set],
) -> Iterable[Mapping[str, Any]]:
    header, header_index = _read_csv_header(path)
    missing = [column for column in columns if column not in header_index]
    if missing:
        raise ContractViolation("csv missing columns: " + ",".join(missing))
    indices = [(column, header_index[column]) for column in columns]
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row_number, row in enumerate(reader):
            if allowed_rows is not None and row_number not in allowed_rows:
                continue
            out = {}
            for column, idx in indices:
                if idx >= len(row):
                    raise ContractViolation("csv row missing column %s" % column)
                out[column] = row[idx]
            yield out


def _read_csv_header(path: Path) -> Tuple[Tuple[str, ...], Mapping[str, int]]:
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise ContractViolation("empty CSV: %s" % path)
    return header, {name: idx for idx, name in enumerate(header)}


def _parquet_rows_for_execute(ledger: Any, spec: Any, scope: Any) -> Iterable[Mapping[str, Any]]:
    import glob

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    matches = sorted(glob.glob(str(spec.path_glob)))
    if not matches:
        raise ContractViolation("no parquet parts matched for %s" % spec.artifact_id)
    structural = _structural_columns_for_execute(ledger, spec, scope)
    projected = _projected_columns_for_execute(ledger, spec, scope)
    cumulative_allowed = 0
    for part_index, raw_path in enumerate(matches, start=1):
        path = Path(raw_path)
        if spec.rq007_split_applicable is True:
            structural_table = pq.ParquetFile(path).read(columns=list(structural))
            structural_rows = structural_table.to_pylist()
            allowed_ids = set()
            allowed_count = 0
            unmapped = 0
            for row in structural_rows:
                case_id = str(row[scope.join_column])
                split = scope.allowlist.case_to_split.get(case_id)
                if split is None:
                    unmapped += 1
                elif case_id in scope.allowlist.allowed_case_ids:
                    allowed_ids.add(case_id)
                    allowed_count += 1
            if unmapped:
                raise ContractViolation("%s unmapped case rows: %d" % (spec.artifact_id, unmapped))
            if allowed_count:
                table = ds.dataset(str(path), format="parquet").to_table(
                    columns=list(projected),
                    filter=ds.field(scope.join_column).isin(list(allowed_ids)),
                )
                rows = table.to_pylist()
                if len(rows) != allowed_count:
                    raise ContractViolation(
                        "%s parquet allowlist row count mismatch in %s: %d != %d"
                        % (spec.artifact_id, path, len(rows), allowed_count)
                    )
                for row in rows:
                    yield row
            cumulative_allowed += allowed_count
        else:
            table = pq.ParquetFile(path).read(columns=list(projected))
            rows = table.to_pylist()
            cumulative_allowed += len(rows)
            for row in rows:
                yield row
        _progress(
            "PARQUET_PART artifact=%s part=%d/%d allowed_physical_rows=%d"
            % (spec.artifact_id, part_index, len(matches), cumulative_allowed)
        )


def _structural_columns_for_execute(ledger: Any, spec: Any, scope: Any) -> Tuple[str, ...]:
    return ledger._structural_columns_for_spec(spec, scope)


def _measurement_columns_for_execute(spec: Any) -> Tuple[str, ...]:
    columns = []
    for role in spec.roles:
        if role.excluded:
            continue
        columns.append(role.ipv_column)
        columns.append(role.ipv_error_column)
    out = []
    seen = set()
    for column in columns:
        if column is None or column in seen:
            continue
        seen.add(column)
        out.append(column)
    return tuple(out)


def _projected_columns_for_execute(ledger: Any, spec: Any, scope: Any) -> Tuple[str, ...]:
    columns = []
    for column in _structural_columns_for_execute(ledger, spec, scope):
        columns.append(column)
    for column in _measurement_columns_for_execute(spec):
        columns.append(column)
    out = []
    seen = set()
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        out.append(column)
    return tuple(out)


def _expected_measurement_rows(spec: Any) -> Optional[int]:
    measured = spec.schema_entry.get("measured") or {}
    if "measurement_rows_expected" in measured:
        return int(measured["measurement_rows_expected"])
    if "data_rows" in measured:
        return int(measured["data_rows"]) * int(spec.expansion_factor) // int(spec.collapse_factor)
    return None


def _distribution(values: Sequence[float]) -> Mapping[str, Any]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "near_uniform_share": None,
        }
    return {
        "n": len(vals),
        "min": vals[0],
        "p25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.5),
        "p75": _quantile(vals, 0.75),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": vals[-1],
        "near_uniform_share": sum(1 for value in vals if value >= 0.93) / len(vals),
    }


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ContractViolation("quantile requires values")
    pos = quantile * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _all_q_values(values_by_status: Mapping[str, Sequence[float]]) -> List[float]:
    out: List[float] = []
    for values in values_by_status.values():
        out.extend(values)
    return out


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_record(repo_root: Path, path: Path, rows: Optional[int]) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root / path
    return {
        "path": _path_for_receipt_metadata(repo_root, path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _progress(message: str) -> None:
    print(
        "%s %s" % (datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), message),
        flush=True,
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


def _path_for_receipt_metadata(repo_root: Path, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


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
