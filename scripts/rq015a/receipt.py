#!/usr/bin/env python3
"""RQ015A receipt construction and append-only JSON writing."""

from __future__ import annotations

import dataclasses
import json
import numbers
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rq015a_contracts import ConservationReport, ContractViolation, SCHEMA_VERSION

BINS_VERDICTS = ("BINS_REPORTABLE", "BINS_WITHHELD_UNSTABLE")
REQUIRED_RECEIPT_FIELDS = (
    "per_artifact_conservation",
    "held_out_parsed_rows",
    "held_out_conclusion_rows",
    "duplicate_primary_keys",
    "unmapped_measurement_roles",
    "k_unknown_rows",
    "bins_stability_verdict",
    "c0_routing_stability",
    "input_sha256",
    "parquet_engine",
    "schema_version",
    "m4_only_channel_excluded",
    "artifacts_absent_locally",
    "ledger_bearing_artifacts",
    "aggregation_key_derivation",
    "reads_measurement_fields",
    "failure_reasons",
)
ZERO_REQUIRED_COUNTERS = (
    "held_out_parsed_rows",
    "held_out_conclusion_rows",
    "duplicate_primary_keys",
    "unmapped_measurement_roles",
)
SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
RUN_SPEC_BINDING_FIELDS = ("run_spec_path", "execution_contract_path")


def canonical_repo_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation("path value must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def strip_json_fragment(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation("path value must be a non-empty string")
    return value.split("#", 1)[0]


def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ContractViolation("%s must be a JSON object: %s" % (label, path))
    return payload


def authorization_entry_for_operation(
    authorization_path: Path,
    operation_id: str,
) -> Mapping[str, Any]:
    auth = load_json_object(Path(authorization_path), "authorization file")
    authorizations = auth.get("authorizations")
    if not isinstance(authorizations, Mapping):
        raise ContractViolation("authorization file missing authorizations object")
    entry = authorizations.get(operation_id)
    if not isinstance(entry, Mapping):
        raise ContractViolation("authorization entry missing for %s" % operation_id)
    return entry


def assert_run_spec_authorization_binding(
    repo_root: Path,
    run_spec_path: Path,
    authorization_path: Path,
    operation_id: str,
) -> Mapping[str, Any]:
    provided = Path(run_spec_path).resolve()
    entry = authorization_entry_for_operation(Path(authorization_path), operation_id)
    for field_name in RUN_SPEC_BINDING_FIELDS:
        bound_raw = strip_json_fragment(entry.get(field_name))
        bound = canonical_repo_path(Path(repo_root), bound_raw)
        if bound != provided:
            raise ContractViolation(
                "execution_authorized gate rejected unbound run spec; "
                "authorization %s mismatch for %s: caller run spec %s != bound %s"
                % (field_name, operation_id, provided, bound)
            )
    return entry


@dataclass(frozen=True)
class NewJsonPath:
    path: Path


def require_new_json_path(path: Path) -> NewJsonPath:
    p = Path(path)
    if p.exists():
        raise ContractViolation(f"receipt path already exists: {p}")
    if p.suffix.lower() != ".json":
        raise ContractViolation(f"receipt path must end with .json: {p}")
    return NewJsonPath(p)


@dataclass(frozen=True, init=False)
class ReceiptChecks:
    per_artifact_conservation: Mapping[str, Any]
    held_out_parsed_rows: int
    held_out_conclusion_rows: int
    duplicate_primary_keys: int
    unmapped_measurement_roles: int
    k_unknown_rows: int
    bins_stability_verdict: str
    c0_routing_stability: Mapping[str, bool]
    input_sha256: Mapping[str, Any]
    parquet_engine: Mapping[str, str]
    schema_version: str
    m4_only_channel_excluded: bool
    artifacts_absent_locally: Tuple[str, ...]
    ledger_bearing_artifacts: Tuple[str, ...]
    aggregation_key_derivation: str
    reads_measurement_fields: bool
    failure_reasons: Tuple[str, ...]

    def __init__(self, **kwargs: Any) -> None:
        if "machine_verdict" in kwargs:
            raise ContractViolation("machine_verdict is computed, not caller-supplied")
        missing = [name for name in REQUIRED_RECEIPT_FIELDS if name not in kwargs]
        if missing:
            raise ContractViolation("missing required receipt fields: " + ", ".join(missing))
        unexpected = sorted(set(kwargs) - set(REQUIRED_RECEIPT_FIELDS))
        if unexpected:
            raise ContractViolation("unexpected receipt fields: " + ", ".join(unexpected))

        normalized = dict(kwargs)
        normalized["artifacts_absent_locally"] = tuple(normalized["artifacts_absent_locally"])
        normalized["ledger_bearing_artifacts"] = tuple(normalized["ledger_bearing_artifacts"])
        normalized["failure_reasons"] = tuple(normalized["failure_reasons"])

        for name in ZERO_REQUIRED_COUNTERS + ("k_unknown_rows",):
            _require_nonnegative_int(name, normalized[name])
        if normalized["bins_stability_verdict"] not in BINS_VERDICTS:
            raise ContractViolation("invalid bins_stability_verdict: %r" %
                                    (normalized["bins_stability_verdict"],))
        if not isinstance(normalized["per_artifact_conservation"], Mapping):
            raise ContractViolation("per_artifact_conservation must be a mapping")
        if not isinstance(normalized["c0_routing_stability"], Mapping):
            raise ContractViolation("c0_routing_stability must be a mapping")
        if not isinstance(normalized["input_sha256"], Mapping):
            raise ContractViolation("input_sha256 must be a mapping")
        if not isinstance(normalized["parquet_engine"], Mapping):
            raise ContractViolation("parquet_engine must record name/version")
        _validate_input_sha256(normalized["input_sha256"])
        _validate_c0_routing_stability(normalized["c0_routing_stability"])
        _require_nonempty_string(
            "parquet_engine.name", normalized["parquet_engine"].get("name")
        )
        _require_nonempty_string(
            "parquet_engine.version", normalized["parquet_engine"].get("version")
        )
        if normalized["m4_only_channel_excluded"] is not True:
            raise ContractViolation("m4_only_channel_excluded must be true")
        if not isinstance(normalized["aggregation_key_derivation"], str):
            raise ContractViolation("aggregation_key_derivation must be a string")
        if not normalized["aggregation_key_derivation"].strip():
            raise ContractViolation("aggregation_key_derivation must be non-empty")
        _require_nonempty_string("schema_version", normalized["schema_version"])
        if not isinstance(normalized["reads_measurement_fields"], bool):
            raise ContractViolation("reads_measurement_fields must be boolean")
        _validate_nonempty_string_sequence(
            "artifacts_absent_locally", normalized["artifacts_absent_locally"]
        )
        _validate_nonempty_string_sequence(
            "ledger_bearing_artifacts", normalized["ledger_bearing_artifacts"]
        )
        _validate_nonempty_string_sequence(
            "failure_reasons", normalized["failure_reasons"]
        )

        for name in REQUIRED_RECEIPT_FIELDS:
            object.__setattr__(self, name, normalized[name])


@dataclass(frozen=True)
class ValidationReceipt:
    machine_verdict: str
    checks: ReceiptChecks
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunReceipt:
    machine_verdict: str
    checks: ReceiptChecks
    metadata: Mapping[str, Any]


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation("%s must be a non-negative integer" % name)


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ContractViolation("%s must be a string" % name)
    if not value.strip():
        raise ContractViolation("%s must be non-empty" % name)
    return value


def _validate_nonempty_string_sequence(name: str, values: Any) -> None:
    if not isinstance(values, tuple):
        raise ContractViolation("%s must be a tuple" % name)
    for index, value in enumerate(values):
        _require_nonempty_string("%s[%d]" % (name, index), value)


def _validate_c0_routing_stability(value: Mapping[str, Any]) -> None:
    for key, is_stable in value.items():
        _require_nonempty_string("c0_routing_stability key", key)
        if not isinstance(is_stable, bool):
            raise ContractViolation("c0_routing_stability values must be boolean")


def _validate_input_sha256(value: Mapping[str, Any]) -> None:
    for key, digest in value.items():
        _require_nonempty_string("input_sha256 key", key)
        if not isinstance(digest, str) or not SHA256_HEX_RE.fullmatch(digest):
            raise ContractViolation(
                "input_sha256[%r] must be a 64-character SHA-256 hex digest" % key
            )


def artifacts_absent_locally_from_schema(schema: Mapping[str, Any]) -> Tuple[str, ...]:
    if "non_ledger_artifacts" not in schema:
        raise ContractViolation("schema must be loaded by load_schema() before deriving absent artifacts")
    absent = []
    for artifact in schema["non_ledger_artifacts"]:
        if artifact.get("status") == "ARTIFACT_NOT_PRESENT_LOCALLY":
            absent.append(str(artifact.get("artifact_id")))
    return tuple(absent)


def detect_parquet_engine() -> Mapping[str, str]:
    for name in ("pyarrow", "fastparquet"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", "UNKNOWN")
        return {"name": name, "version": str(version)}
    raise ContractViolation("no parquet engine available; run spec v2 must require pyarrow or fastparquet")


def build_receipt_checks_from_schema(schema: Mapping[str, Any], **fields_: Any) -> ReceiptChecks:
    derived_absent = artifacts_absent_locally_from_schema(schema)
    derived_ledger_bearing = tuple(
        str(artifact_id) for artifact_id in schema.get("ledger_bearing_artifact_ids", ())
    )
    fields_.setdefault("schema_version", schema.get("schema_id"))
    if "artifacts_absent_locally" in fields_:
        supplied_absent = tuple(fields_["artifacts_absent_locally"])
        if supplied_absent != derived_absent:
            raise ContractViolation(
                "artifacts_absent_locally must be schema-derived"
            )
    fields_["artifacts_absent_locally"] = derived_absent
    if "ledger_bearing_artifacts" in fields_:
        supplied_ledger_bearing = tuple(fields_["ledger_bearing_artifacts"])
        if supplied_ledger_bearing != derived_ledger_bearing:
            raise ContractViolation(
                "ledger_bearing_artifacts must be schema-derived"
            )
    fields_["ledger_bearing_artifacts"] = derived_ledger_bearing
    fields_.setdefault("parquet_engine", detect_parquet_engine())
    return ReceiptChecks(**fields_)


def _base_machine_verdict_reasons(checks: ReceiptChecks) -> Sequence[str]:
    reasons = list(checks.failure_reasons)
    for name in ZERO_REQUIRED_COUNTERS:
        if getattr(checks, name) != 0:
            reasons.append("%s_nonzero" % name)
    if checks.schema_version != SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if checks.m4_only_channel_excluded is not True:
        reasons.append("m4_only_channel_not_excluded")
    if checks.reads_measurement_fields is not False:
        reasons.append("validate_reads_measurement_fields")
    return tuple(reasons)


def compute_machine_verdict(checks: ReceiptChecks) -> str:
    reasons = _base_machine_verdict_reasons(checks)
    return "FAIL" if reasons else "PASS"


def compute_run_machine_verdict(checks: ReceiptChecks) -> str:
    reasons = list(_base_machine_verdict_reasons(checks))
    reasons.extend(_run_completeness_failure_reasons(checks))
    reasons.extend(_conservation_failure_reasons(checks.per_artifact_conservation))
    return "FAIL" if reasons else "PASS"


def _run_completeness_failure_reasons(checks: ReceiptChecks) -> Tuple[str, ...]:
    reasons = []
    expected = set(checks.ledger_bearing_artifacts)
    provided = set(str(k) for k in checks.per_artifact_conservation.keys())
    if not checks.per_artifact_conservation:
        reasons.append("per_artifact_conservation_empty")
    missing = sorted(expected - provided)
    if missing:
        reasons.append("per_artifact_conservation_missing:%s" % ",".join(missing))
    for artifact_id in sorted(expected & provided):
        measurement_rows = _measurement_rows_for_report(
            checks.per_artifact_conservation[artifact_id]
        )
        if measurement_rows is None:
            reasons.append("%s_measurement_rows_missing" % artifact_id)
        elif (
            isinstance(measurement_rows, bool)
            or not isinstance(measurement_rows, int)
            or measurement_rows <= 0
        ):
            reasons.append("%s_measurement_rows_not_positive" % artifact_id)
    if not checks.input_sha256:
        reasons.append("input_sha256_empty")
    return tuple(reasons)


def build_validate_receipt(checks: ReceiptChecks,
                           metadata: Optional[Mapping[str, Any]] = None) -> ValidationReceipt:
    return ValidationReceipt(
        machine_verdict=compute_machine_verdict(checks),
        checks=checks,
        metadata=dict(metadata or {}),
    )


def build_run_receipt(checks: Optional[ReceiptChecks] = None,
                      metadata: Optional[Mapping[str, Any]] = None,
                      **fields_: Any) -> RunReceipt:
    if "machine_verdict" in fields_:
        raise ContractViolation("machine_verdict is computed, not caller-supplied")
    if checks is not None and fields_:
        raise ContractViolation("pass either checks or receipt fields, not both")
    if checks is None:
        checks = ReceiptChecks(**fields_)
    return RunReceipt(
        machine_verdict=compute_run_machine_verdict(checks),
        checks=checks,
        metadata=dict(metadata or {}),
    )


def write_receipt(path: Any, receipt: Any) -> Path:
    new_path = path if isinstance(path, NewJsonPath) else require_new_json_path(Path(path))
    payload = _receipt_to_dict(receipt)
    new_path.path.parent.mkdir(parents=True, exist_ok=True)
    new_path.path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return new_path.path


def _conservation_failure_reasons(per_artifact: Mapping[str, Any]) -> Tuple[str, ...]:
    reasons = []
    for artifact_id, report in sorted(per_artifact.items()):
        try:
            _verify_conservation_report(str(artifact_id), report)
        except ContractViolation as exc:
            reasons.append("%s_conservation_failed:%s" % (artifact_id, exc))
    return tuple(reasons)


def _require_report_int(artifact_id: str, name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ContractViolation("%s: %s must be an integer" % (artifact_id, name))
    value = int(value)
    if value < minimum:
        raise ContractViolation("%s: %s must be >= %d" % (artifact_id, name, minimum))
    return value


def _sum_report_counts(artifact_id: str, name: str, counts: Any) -> int:
    if not isinstance(counts, Mapping):
        raise ContractViolation("%s: %s must be a mapping" % (artifact_id, name))
    total = 0
    for label, value in counts.items():
        total += _require_report_int(
            artifact_id, "%s[%r]" % (name, label), value, 0
        )
    return total


def _verify_conservation_report(artifact_id: str, report: Any) -> None:
    if not isinstance(report, ConservationReport):
        raise ContractViolation("expected ConservationReport from check_conservation()")
    if str(report.artifact_id) != artifact_id:
        raise ContractViolation(
            "artifact_id mismatch %s != %s" % (report.artifact_id, artifact_id)
        )
    physical_rows = _require_report_int(artifact_id, "physical_rows", report.physical_rows, 0)
    expansion_factor = _require_report_int(
        artifact_id, "expansion_factor", report.expansion_factor, 1
    )
    collapse_factor = _require_report_int(
        artifact_id, "collapse_factor", report.collapse_factor, 1
    )
    observed = _require_report_int(
        artifact_id, "measurement_rows_observed", report.measurement_rows_observed, 0
    )
    expected = _require_report_int(
        artifact_id, "measurement_rows_expected", report.measurement_rows_expected, 0
    )
    numerator = physical_rows * expansion_factor
    if numerator % collapse_factor != 0:
        raise ContractViolation(
            "%s: %d not divisible by collapse_factor %d"
            % (artifact_id, numerator, collapse_factor)
        )
    recomputed_expected = numerator // collapse_factor
    if expected != recomputed_expected or observed != expected:
        raise ContractViolation(
            "%s: identity_1 failed %d != %d"
            % (artifact_id, observed, recomputed_expected)
        )
    status_total = _sum_report_counts(artifact_id, "status_counts", report.status_counts)
    if status_total != observed:
        raise ContractViolation(
            "%s: identity_2 failed %d != %d" % (artifact_id, status_total, observed)
        )
    recoverability_total = _sum_report_counts(
        artifact_id, "recoverability_counts", report.recoverability_counts
    )
    if recoverability_total != observed:
        raise ContractViolation(
            "%s: identity_3 failed %d != %d"
            % (artifact_id, recoverability_total, observed)
        )
    report.assert_ok()


def _measurement_rows_for_report(report: Any) -> Optional[Any]:
    if isinstance(report, Mapping):
        if "measurement_rows" in report:
            return report.get("measurement_rows")
        if "measurement_rows_observed" in report:
            return report.get("measurement_rows_observed")
        return None
    if hasattr(report, "measurement_rows"):
        return getattr(report, "measurement_rows")
    if hasattr(report, "measurement_rows_observed"):
        return getattr(report, "measurement_rows_observed")
    return None


def _receipt_to_dict(receipt: Any) -> Dict[str, Any]:
    if isinstance(receipt, ValidationReceipt):
        kind = "validate_receipt"
    elif isinstance(receipt, RunReceipt):
        kind = "run_receipt"
    else:
        raise ContractViolation("unsupported receipt object")
    out = _checks_to_dict(receipt.checks)
    out["receipt_kind"] = kind
    out["machine_verdict"] = receipt.machine_verdict
    out["metadata"] = _to_jsonable(receipt.metadata)
    return out


def _checks_to_dict(checks: ReceiptChecks) -> Dict[str, Any]:
    out = {}
    for field in fields(checks):
        out[field.name] = _to_jsonable(getattr(checks, field.name))
    return out


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    return value
