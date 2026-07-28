#!/usr/bin/env python3
"""RQ015A receipt construction and append-only JSON writing."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rq015a_contracts import ContractViolation, SCHEMA_VERSION

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
        if not normalized["parquet_engine"].get("name"):
            raise ContractViolation("parquet_engine.name is required")
        if not normalized["parquet_engine"].get("version"):
            raise ContractViolation("parquet_engine.version is required")
        if normalized["m4_only_channel_excluded"] is not True:
            raise ContractViolation("m4_only_channel_excluded must be true")
        if not isinstance(normalized["aggregation_key_derivation"], str):
            raise ContractViolation("aggregation_key_derivation must be a string")
        if not normalized["aggregation_key_derivation"].strip():
            raise ContractViolation("aggregation_key_derivation must be non-empty")
        if not isinstance(normalized["schema_version"], str):
            raise ContractViolation("schema_version must be a string")
        if not isinstance(normalized["reads_measurement_fields"], bool):
            raise ContractViolation("reads_measurement_fields must be boolean")

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
    fields_.setdefault("schema_version", schema.get("schema_id"))
    if "artifacts_absent_locally" in fields_:
        supplied_absent = tuple(fields_["artifacts_absent_locally"])
        if supplied_absent != derived_absent:
            raise ContractViolation(
                "artifacts_absent_locally must be schema-derived"
            )
    fields_["artifacts_absent_locally"] = derived_absent
    fields_.setdefault("parquet_engine", detect_parquet_engine())
    return ReceiptChecks(**fields_)


def compute_machine_verdict(checks: ReceiptChecks) -> str:
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
    reasons.extend(_conservation_failure_reasons(checks.per_artifact_conservation))
    return "FAIL" if reasons else "PASS"


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
        machine_verdict=compute_machine_verdict(checks),
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
        assert_ok = getattr(report, "assert_ok", None)
        if callable(assert_ok):
            try:
                assert_ok()
            except ContractViolation as exc:
                reasons.append("%s_conservation_failed:%s" % (artifact_id, exc))
            continue
        if isinstance(report, Mapping):
            if report.get("passed") is False:
                reasons.append("%s_conservation_failed" % artifact_id)
                continue
            identities = (report.get("identity_1"), report.get("identity_2"), report.get("identity_3"))
            if any(value is False for value in identities):
                reasons.append("%s_conservation_failed" % artifact_id)
                continue
            if any(value is None for value in identities):
                reasons.append("%s_conservation_incomplete" % artifact_id)
                continue
        else:
            reasons.append("%s_conservation_unreadable" % artifact_id)
    return tuple(reasons)


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
