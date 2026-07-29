#!/usr/bin/env python3
"""Shared runtime types for the RQ015A concentration ledger.

The type aliases intentionally use Python 3.9-compatible ``typing.Union`` /
``typing.Optional`` forms.  Literal annotations are backed by construction-time
checks for the invariants that matter at runtime.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

try:  # pragma: no cover - import style depends on caller path setup
    from rq015a_contracts import ContractViolation, SCHEMA_VERSION
except ImportError:  # pragma: no cover
    from .rq015a_contracts import ContractViolation, SCHEMA_VERSION


SchemaVersion = Literal["rq015a-concentration-ledger-v2"]
AttemptStatus = Literal["ATTEMPTED", "NOT_ATTEMPTED", "UNKNOWN"]
MachineVerdict = Literal["PASS", "FAIL"]
PresentArtifactId = Literal[
    "interhub_sigma01_hw4_timeseries",
    "rq009_feature_matrix",
    "onsite_dense_timeseries",
]
AbsentArtifactId = Literal[
    "wod_rq010b_full479_audited",
    "wod_phase1_phase1b_10hz_schemeb",
    "rq014_g2r_anchor_scores",
]
ArtifactId = Union[PresentArtifactId, AbsentArtifactId, str]
SplitApplicableArtifactId = Literal[
    "interhub_sigma01_hw4_timeseries",
    "rq009_feature_matrix",
]
SplitNotApplicableArtifactId = Literal[
    "onsite_dense_timeseries",
    "wod_rq010b_full479_audited",
    "wod_phase1_phase1b_10hz_schemeb",
    "rq014_g2r_anchor_scores",
]
RQ007IncludedSplit = Literal["development", "guard"]
LedgerSplitValue = Literal[
    "development",
    "guard",
    "held_out",
    "RQ007_SPLIT_NOT_APPLICABLE",
    "unknown",
]
CandidateGridId = Literal["legacy7_pi_over_8", "realtime5_pi_over_8", "UNKNOWN"]
Recoverability = Literal[
    "L1_DIRECT",
    "L2_PROVENANCE",
    "L3_PENDING_RQ015B",
    "L4_UNRECOVERABLE",
    "RECOVERABLE_BY_REPLAY_OUT_OF_SCOPE",
    "ARTIFACT_NOT_PRESENT_LOCALLY",
]


ATTEMPTED = "ATTEMPTED"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
UNKNOWN = "UNKNOWN"
RQ007_SPLIT_NOT_APPLICABLE = "RQ007_SPLIT_NOT_APPLICABLE"
ARTIFACT_NOT_PRESENT_LOCALLY = "ARTIFACT_NOT_PRESENT_LOCALLY"

_ATTEMPT_STATUSES = frozenset((ATTEMPTED, NOT_ATTEMPTED, UNKNOWN))
_LEDGER_SPLITS = frozenset(
    ("development", "guard", "held_out", RQ007_SPLIT_NOT_APPLICABLE, "unknown")
)
_RECOVERABILITY = frozenset(
    (
        "L1_DIRECT",
        "L2_PROVENANCE",
        "L3_PENDING_RQ015B",
        "L4_UNRECOVERABLE",
        "RECOVERABLE_BY_REPLAY_OUT_OF_SCOPE",
        ARTIFACT_NOT_PRESENT_LOCALLY,
    )
)
_FORMATS = frozenset(("csv", "parquet", "absent"))
_JOIN_COLUMNS = frozenset(("scene_unique_id", "case_key"))
_INCLUDED_SPLITS = frozenset(("development", "guard"))
_SPLIT_NOT_APPLICABLE_REASONS = frozenset(
    ("non_rq007_artifact", "artifact_absent_locally")
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def _require_in(name: str, value: object, allowed: FrozenSet[str]) -> None:
    if value not in allowed:
        raise ContractViolation("%s %r not in %s" % (name, value, sorted(allowed)))


def is_forbidden_human_field(column: str) -> bool:
    name = str(column).lower()
    return any(token in name for token in ("rating", "preference", "human", "score"))


def is_measurement_like_field(column: str) -> bool:
    name = str(column)
    low = name.lower()
    return (
        name.startswith("ipv_")
        or name.startswith("target_ipv")
        or name.startswith("counterpart_ipv")
        or name.startswith("M4_ONLY_")
        or "label" in low
        or is_forbidden_human_field(name)
    )


@dataclass(frozen=True, init=False)
class CaseAllowlist:
    source_path: Path
    included_splits: Tuple[str, ...]
    allowed_case_ids: FrozenSet[str]
    split_counts: Mapping[str, int]
    source_sha256: str
    case_to_split: Mapping[str, str]

    @classmethod
    def _from_loaded_split(
        cls,
        source_path: Path,
        included_splits: Sequence[str],
        allowed_case_ids: FrozenSet[str],
        split_counts: Mapping[str, int],
        source_sha256: str,
        case_to_split: Mapping[str, str],
    ) -> "CaseAllowlist":
        obj = cls()
        object.__setattr__(obj, "source_path", Path(source_path))
        object.__setattr__(obj, "included_splits", tuple(included_splits))
        object.__setattr__(obj, "allowed_case_ids", frozenset(allowed_case_ids))
        object.__setattr__(obj, "split_counts", dict(split_counts))
        object.__setattr__(obj, "source_sha256", str(source_sha256))
        object.__setattr__(obj, "case_to_split", dict(case_to_split))
        obj._validate()
        return obj

    def _validate(self) -> None:
        _require(self.included_splits, "CaseAllowlist included_splits is empty")
        for split in self.included_splits:
            _require_in("included split", split, _INCLUDED_SPLITS)
        for case_id in self.allowed_case_ids:
            _require(case_id in self.case_to_split, "allowed case missing split label")
            _require(
                self.case_to_split[case_id] in self.included_splits,
                "allowed case has non-included split",
            )
        _require(len(self.source_sha256) == 64, "source_sha256 must be sha256 hex")


def validate_case_allowlist_token(allowlist: object) -> "CaseAllowlist":
    _require(
        type(allowlist) is CaseAllowlist,
        "allowlist token must be exact CaseAllowlist",
    )
    allowlist._validate()
    return allowlist


def validate_case_allowlist_source(allowlist: object) -> "CaseAllowlist":
    token = validate_case_allowlist_token(allowlist)
    source_path = Path(token.source_path)
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != token.source_sha256:
        raise ContractViolation(
            "CaseAllowlist source_sha256 mismatch: %s" % source_path
        )
    allowed, split_counts, case_to_split = _derive_allowlist_from_split_source(
        source_path,
        token.included_splits,
    )
    if frozenset(token.allowed_case_ids) != allowed:
        raise ContractViolation(
            "CaseAllowlist allowed_case_ids mismatch with source: %s" % source_path
        )
    if dict(token.split_counts) != split_counts:
        raise ContractViolation(
            "CaseAllowlist split_counts mismatch with source: %s" % source_path
        )
    if dict(token.case_to_split) != case_to_split:
        raise ContractViolation(
            "CaseAllowlist case_to_split mismatch with source: %s" % source_path
        )
    return token


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        raise ContractViolation("CaseAllowlist source unreadable: %s" % path)
    return h.hexdigest()


def _derive_allowlist_from_split_source(
    source_path: Path,
    included_splits: Sequence[str],
) -> Tuple[FrozenSet[str], Dict[str, int], Dict[str, str]]:
    include_tuple = tuple(included_splits)
    split_counts = Counter()
    case_to_split = {}
    allowed = set()
    try:
        with Path(source_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "case_id" not in (reader.fieldnames or ()) or "split" not in (
                reader.fieldnames or ()
            ):
                raise ContractViolation("split CSV must contain case_id,split")
            for row in reader:
                case_id = str(row.get("case_id") or "")
                split = str(row.get("split") or "")
                if not case_id or not split:
                    raise ContractViolation("empty case_id/split in allowlist source")
                if case_id in case_to_split:
                    raise ContractViolation(
                        "duplicate case_id in split source: %s" % case_id
                    )
                case_to_split[case_id] = split
                split_counts[split] += 1
                if split in include_tuple:
                    allowed.add(case_id)
    except OSError:
        raise ContractViolation("CaseAllowlist source unreadable: %s" % source_path)
    return frozenset(allowed), dict(split_counts), dict(case_to_split)


@dataclass(frozen=True)
class RequiresRQ007Allowlist:
    join_column: str
    allowed_splits: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_in("join_column", self.join_column, _JOIN_COLUMNS)
        for split in self.allowed_splits:
            _require_in("allowed split", split, _INCLUDED_SPLITS)


@dataclass(frozen=True)
class RQ007SplitNotApplicable:
    value: str
    reason: str

    def __post_init__(self) -> None:
        _require(
            self.value == RQ007_SPLIT_NOT_APPLICABLE,
            "split-not-applicable scope must use RQ007_SPLIT_NOT_APPLICABLE",
        )
        _require_in("split-not-applicable reason", self.reason, _SPLIT_NOT_APPLICABLE_REASONS)


SplitPolicy = Union[RequiresRQ007Allowlist, RQ007SplitNotApplicable]


@dataclass(frozen=True, init=False)
class AllowlistedArtifactScope:
    artifact_id: str
    join_column: str
    allowlist: CaseAllowlist
    held_out_parsed_rows: int
    unmapped_rows: int

    @classmethod
    def _from_schema(
        cls,
        spec: object,
        allowlist: CaseAllowlist,
        join_column: str,
        held_out_parsed_rows: int = 0,
        unmapped_rows: int = 0,
    ) -> "AllowlistedArtifactScope":
        _require(
            getattr(spec, "rq007_split_applicable", None) is True,
            "%s is not schema-declared split-applicable" % getattr(spec, "artifact_id", "<unknown>"),
        )
        obj = cls()
        object.__setattr__(obj, "artifact_id", getattr(spec, "artifact_id"))
        object.__setattr__(obj, "join_column", join_column)
        object.__setattr__(obj, "allowlist", allowlist)
        object.__setattr__(obj, "held_out_parsed_rows", held_out_parsed_rows)
        object.__setattr__(obj, "unmapped_rows", unmapped_rows)
        obj._validate()
        return obj

    def _validate(self) -> None:
        _require_in("join_column", self.join_column, _JOIN_COLUMNS)
        validate_case_allowlist_token(self.allowlist)
        _require(self.held_out_parsed_rows == 0, "held_out_parsed_rows must be 0")
        _require(self.unmapped_rows == 0, "unmapped_rows must be 0")


@dataclass(frozen=True, init=False)
class SplitNotApplicableArtifactScope:
    artifact_id: str
    rq007_split_value: str
    reason: str

    @classmethod
    def _from_schema(cls, spec: object, reason: str) -> "SplitNotApplicableArtifactScope":
        _require(
            getattr(spec, "rq007_split_applicable", None) is False,
            "%s is not schema-declared split-not-applicable"
            % getattr(spec, "artifact_id", "<unknown>"),
        )
        _require(
            getattr(spec, "rq007_split_value", None) == RQ007_SPLIT_NOT_APPLICABLE,
            "%s missing RQ007_SPLIT_NOT_APPLICABLE schema value"
            % getattr(spec, "artifact_id", "<unknown>"),
        )
        obj = cls()
        object.__setattr__(obj, "artifact_id", getattr(spec, "artifact_id"))
        object.__setattr__(obj, "rq007_split_value", RQ007_SPLIT_NOT_APPLICABLE)
        object.__setattr__(obj, "reason", reason)
        obj._validate()
        return obj

    def _validate(self) -> None:
        _require(self.rq007_split_value == RQ007_SPLIT_NOT_APPLICABLE, "bad split value")
        _require_in("split-not-applicable reason", self.reason, _SPLIT_NOT_APPLICABLE_REASONS)


FilteredArtifactScope = Union[AllowlistedArtifactScope, SplitNotApplicableArtifactScope]


@dataclass(frozen=True)
class RoleSpec:
    measurement_role: str
    ipv_column: Optional[str]
    ipv_error_column: Optional[str]
    excluded: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.measurement_role), "measurement_role is required")
        _require(isinstance(self.excluded, bool), "excluded must be bool")
        if not self.excluded:
            _require(self.ipv_error_column is not None, "active role missing error column")
        if self.measurement_role.startswith("M4_ONLY_") and not self.excluded:
            raise ContractViolation("M4_ONLY role may not be active")


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    present_locally: bool
    format: str
    path: Optional[Path]
    path_glob: Optional[str]
    row_key_fields: Tuple[str, ...]
    roles: Tuple[RoleSpec, ...]
    expansion_factor: int
    collapse_factor: int
    split_policy: Optional[SplitPolicy]
    candidate_grid_id: str
    K: Optional[int]
    recoverability: str
    rq007_split_applicable: Optional[bool]
    rq007_split_value: Optional[str]
    not_attempted_rule: Mapping[str, Any]
    status: Optional[str]
    schema_entry: Mapping[str, Any]
    split_filter_column: Optional[str] = None

    def __post_init__(self) -> None:
        _require(bool(self.artifact_id), "artifact_id required")
        _require(isinstance(self.present_locally, bool), "present_locally must be bool")
        _require_in("format", self.format, _FORMATS)
        _require(isinstance(self.expansion_factor, int) and self.expansion_factor >= 1,
                 "bad expansion_factor")
        _require(isinstance(self.collapse_factor, int) and self.collapse_factor >= 1,
                 "bad collapse_factor")
        _require_in("recoverability", self.recoverability, _RECOVERABILITY)
        if self.present_locally:
            _require(self.format in ("csv", "parquet"), "present artifact must be csv/parquet")
            _require(self.rq007_split_applicable in (True, False),
                     "%s missing rq007_split_applicable" % self.artifact_id)
        if self.rq007_split_applicable is False:
            _require(
                self.rq007_split_value == RQ007_SPLIT_NOT_APPLICABLE,
                "%s missing rq007_split_value" % self.artifact_id,
            )
        if self.K is not None:
            _require(isinstance(self.K, int) and not isinstance(self.K, bool) and self.K >= 1,
                     "%s has invalid K" % self.artifact_id)
        _require(tuple(self.row_key_fields) == self.row_key_fields, "row_key_fields must be tuple")


@dataclass(frozen=True)
class L1LedgerRow:
    artifact_id: str
    product_row_key: str
    measurement_role: str
    case_id: Optional[str]
    rq007_split: str
    ipv_error: Optional[float]
    K: Optional[int]
    candidate_grid_id: str
    k_eff: Optional[float]
    q_eff: Optional[float]
    attempt_status: str
    reason_code: Optional[str]
    recoverability: str
    ledger_schema_version: str
    aggregation_perspective: str
    aggregation_configuration: str

    def __post_init__(self) -> None:
        _require(bool(self.artifact_id), "L1 row missing artifact_id")
        _require(bool(self.product_row_key), "L1 row missing product_row_key")
        _require(bool(self.measurement_role), "L1 row missing measurement_role")
        _require_in("rq007_split", self.rq007_split, _LEDGER_SPLITS)
        _require_in("attempt_status", self.attempt_status, _ATTEMPT_STATUSES)
        _require_in("recoverability", self.recoverability, _RECOVERABILITY)
        _require(
            self.ledger_schema_version == SCHEMA_VERSION,
            "ledger_schema_version must be %s" % SCHEMA_VERSION,
        )
        _require(bool(self.aggregation_perspective), "aggregation_perspective required")
        _require(bool(self.aggregation_configuration), "aggregation_configuration required")
        if self.case_id is not None:
            _require(isinstance(self.case_id, str), "case_id must be str or None")

    def to_contract_dict(self) -> Dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "case_id": self.case_id,
            "perspective": self.aggregation_perspective,
            "configuration": self.aggregation_configuration,
            "attempt_status": self.attempt_status,
            "q_eff": self.q_eff,
        }


def l1_sort_key(row: L1LedgerRow) -> Tuple[object, ...]:
    return (
        row.artifact_id,
        row.case_id is not None,
        row.case_id or "",
        row.product_row_key,
        row.measurement_role,
    )


@dataclass(frozen=True)
class SortedL1LedgerRows:
    artifact_id: str
    rows: Tuple[L1LedgerRow, ...]
    sort_key: str

    def __post_init__(self) -> None:
        _require(
            self.sort_key == "artifact_id,case_id,product_row_key,measurement_role",
            "bad L1 sort key",
        )
        for row in self.rows:
            _require(row.artifact_id == self.artifact_id, "mixed artifact in SortedL1LedgerRows")
        _require(tuple(sorted(self.rows, key=l1_sort_key)) == self.rows, "L1 rows are not sorted")


@dataclass(frozen=True)
class SortedL2Units:
    artifact_id: str
    units: Tuple[object, ...]
    sort_key: str

    def __post_init__(self) -> None:
        _require(
            self.sort_key == "artifact_id,case_id,perspective,configuration",
            "bad L2 sort key",
        )


@dataclass(frozen=True)
class SortedL3Units:
    artifact_id: str
    units: Tuple[object, ...]
    sort_key: str


@dataclass(frozen=True)
class AbsentArtifactCoverage:
    artifact_id: str
    attempt_status: str
    recoverability: str
    reason_code: str
    schema_status: str

    def __post_init__(self) -> None:
        _require(self.attempt_status == UNKNOWN, "absent artifact attempt_status must be UNKNOWN")
        _require(
            self.recoverability == ARTIFACT_NOT_PRESENT_LOCALLY,
            "absent artifact recoverability must be ARTIFACT_NOT_PRESENT_LOCALLY",
        )
        _require(
            self.schema_status == ARTIFACT_NOT_PRESENT_LOCALLY,
            "absent artifact must be schema-derived from non_ledger_artifacts",
        )


@dataclass(frozen=True)
class LedgerSchema:
    schema_version: str
    artifacts: Tuple[ArtifactSpec, ...]
    artifacts_by_id: Mapping[str, ArtifactSpec]
    ledger_bearing_artifact_ids: Tuple[str, ...]
    non_ledger_artifacts: Tuple[Mapping[str, str], ...]
    artifacts_absent_locally: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require(self.schema_version == SCHEMA_VERSION, "schema version mismatch")
        for artifact_id in self.artifacts_absent_locally:
            matching = [
                a for a in self.non_ledger_artifacts
                if a.get("artifact_id") == artifact_id
                and a.get("status") == ARTIFACT_NOT_PRESENT_LOCALLY
            ]
            _require(bool(matching), "absent artifact not derived from non_ledger_artifacts")


@dataclass(frozen=True)
class LedgerBuildResult:
    l1_by_artifact: Mapping[str, SortedL1LedgerRows]
    absent_artifacts: Tuple[AbsentArtifactCoverage, ...]
    conservation: Mapping[str, object]
    aggregation_key_derivation: Mapping[str, object]
    held_out_parsed_rows: int

    def __post_init__(self) -> None:
        _require(self.held_out_parsed_rows == 0, "held_out_parsed_rows must be 0")
        _require("version" in self.aggregation_key_derivation, "missing aggregation derivation")


@dataclass(frozen=True)
class StructuralColumnSet:
    columns: Tuple[str, ...]
    forbids_measurement_fields: bool

    def __post_init__(self) -> None:
        _require(
            self.forbids_measurement_fields is True,
            "forbids_measurement_fields must be True",
        )
        for column in self.columns:
            if is_measurement_like_field(column):
                raise ContractViolation("structural scan requested measurement field %s" % column)


class StructuralReader(Protocol):
    def schema_columns(self) -> Tuple[str, ...]:
        ...

    def count_rows(self, columns: StructuralColumnSet) -> int:
        ...

    def sha256(self) -> str:
        ...


class MeasurementReader(Protocol):
    artifact_id: str
    scope: FilteredArtifactScope

    def iter_measurement_rows(self) -> Iterator[Mapping[str, object]]:
        ...


@dataclass(frozen=True, init=False)
class ExecutePermit:
    operation_id: str
    execution_authorized: bool
    authorization_path: Path
    authorization_sha256: str

    @classmethod
    def _from_authorization(
        cls,
        operation_id: str,
        execution_authorized: bool,
        authorization_path: Path,
        authorization_sha256: str,
    ) -> "ExecutePermit":
        obj = cls()
        object.__setattr__(obj, "operation_id", operation_id)
        object.__setattr__(obj, "execution_authorized", execution_authorized)
        object.__setattr__(obj, "authorization_path", Path(authorization_path))
        object.__setattr__(obj, "authorization_sha256", str(authorization_sha256))
        obj._validate()
        return obj

    def _validate(self) -> None:
        _require(
            self.operation_id == "rq015a_concentration_audit",
            "ExecutePermit operation_id mismatch",
        )
        _require(self.execution_authorized is True, "ExecutePermit requires authorization True")
        _require(len(self.authorization_sha256) == 64, "authorization_sha256 must be sha256 hex")
