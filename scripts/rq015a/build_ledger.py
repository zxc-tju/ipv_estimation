#!/usr/bin/env python3
"""Build RQ015A L1 concentration-ledger rows under BUILD_WHILE_DENY.

This module does not authorize or launch a full audit.  Measurement readers are
structurally gated by a schema-derived filtered scope plus an ExecutePermit; the
current production authorization file cannot create that permit.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import numbers
import weakref
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import style depends on caller path setup
    from rq015a_contracts import (
        ContractViolation,
        SCHEMA_VERSION,
        aggregate_l2,
        aggregate_l3,
        assert_single_artifact,
        check_conservation,
        k_eff_from_error,
        load_schema as _load_schema_contract,
        local_positions,
        q_eff as _q_eff,
        validate_artifact_id,
    )
    from rq015a_types import (
        ARTIFACT_NOT_PRESENT_LOCALLY,
        ATTEMPTED,
        NOT_ATTEMPTED,
        RQ007_SPLIT_NOT_APPLICABLE,
        UNKNOWN,
        AbsentArtifactCoverage,
        AllowlistedArtifactScope,
        ArtifactSpec,
        CaseAllowlist,
        ExecutePermit,
        FilteredArtifactScope,
        L1LedgerRow,
        LedgerBuildResult,
        LedgerSchema,
        RequiresRQ007Allowlist,
        RoleSpec,
        RQ007SplitNotApplicable,
        SortedL1LedgerRows,
        SortedL2Units,
        SortedL3Units,
        SplitNotApplicableArtifactScope,
        is_forbidden_human_field,
        l1_sort_key,
        validate_case_allowlist_source,
        validate_case_allowlist_token,
    )
except ImportError:  # pragma: no cover
    from .rq015a_contracts import (
        ContractViolation,
        SCHEMA_VERSION,
        aggregate_l2,
        aggregate_l3,
        assert_single_artifact,
        check_conservation,
        k_eff_from_error,
        load_schema as _load_schema_contract,
        local_positions,
        q_eff as _q_eff,
        validate_artifact_id,
    )
    from .rq015a_types import (
        ARTIFACT_NOT_PRESENT_LOCALLY,
        ATTEMPTED,
        NOT_ATTEMPTED,
        RQ007_SPLIT_NOT_APPLICABLE,
        UNKNOWN,
        AbsentArtifactCoverage,
        AllowlistedArtifactScope,
        ArtifactSpec,
        CaseAllowlist,
        ExecutePermit,
        FilteredArtifactScope,
        L1LedgerRow,
        LedgerBuildResult,
        LedgerSchema,
        RequiresRQ007Allowlist,
        RoleSpec,
        RQ007SplitNotApplicable,
        SortedL1LedgerRows,
        SortedL2Units,
        SortedL3Units,
        SplitNotApplicableArtifactScope,
        is_forbidden_human_field,
        l1_sort_key,
        validate_case_allowlist_source,
        validate_case_allowlist_token,
    )


L1_SORT_KEY = "artifact_id,case_id,product_row_key,measurement_role"
L2_SORT_KEY = "artifact_id,case_id,perspective,configuration"
AGGREGATION_KEY_DERIVATION = {
    "version": "rq015a-aggregation-key-v2",
    "rules": {
        "interhub_sigma01_hw4_timeseries": (
            "perspective=measurement_role; configuration=sigma01_hw4"
        ),
        "rq009_feature_matrix": (
            "perspective=product_row_key.perspective; "
            "configuration=measurement_role|source_dataset with product-row-key escaping"
        ),
        "onsite_dense_timeseries": (
            "perspective=measurement_role; configuration=product_row_key.case_key"
        ),
    },
}
CSV_REQUIRED_ROW_LIMIT_MESSAGE = "BUILD_WHILE_DENY requires explicit csv_row_limit"
PRODUCT_ROW_KEY_ESCAPE = "\\"
PRODUCT_ROW_KEY_SEPARATOR = "|"
PRODUCT_ROW_KEY_ASSIGN = "="
PRODUCT_ROW_KEY_ESCAPED_CHARS = (
    PRODUCT_ROW_KEY_ESCAPE,
    PRODUCT_ROW_KEY_SEPARATOR,
    PRODUCT_ROW_KEY_ASSIGN,
)
_OPENED_MEASUREMENT_READERS = weakref.WeakSet()


def _dedupe_columns(columns: Iterable[Optional[str]]) -> Tuple[str, ...]:
    out = []
    seen = set()
    for column in columns:
        if column is None:
            continue
        name = str(column)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root_for_schema(schema_path: Path) -> Path:
    resolved = Path(schema_path).resolve()
    if len(resolved.parents) >= 3 and resolved.parents[1].name == "reports":
        return resolved.parents[2]
    return Path.cwd()


def _as_tuple(value: object) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ContractViolation("expected list, got %r" % (value,))
    return tuple(str(v) for v in value)


def _row_key_fields(entry: Mapping[str, object]) -> Tuple[str, ...]:
    fields = entry.get("row_key_fields")
    if fields is None:
        fields = entry.get("row_key_fields_actual")
    return _as_tuple(fields)


def _split_join_column(entry: Mapping[str, object]) -> str:
    direct = entry.get("split_filter_column")
    if direct:
        return str(direct)
    for field in _row_key_fields(entry):
        if field in ("scene_unique_id", "case_key"):
            return field
    raise ContractViolation("%s: no schema-derived split join column" % entry.get("artifact_id"))


def _role_specs(entry: Mapping[str, object], allow_missing_columns: bool = False) -> Tuple[RoleSpec, ...]:
    role_source = entry.get("role_source_columns") or {}
    if not isinstance(role_source, Mapping):
        raise ContractViolation("%s: role_source_columns must be mapping" % entry.get("artifact_id"))
    out = []
    for role in _as_tuple(entry.get("measurement_roles")):
        if role.startswith("M4_ONLY_"):
            raise ContractViolation("M4_ONLY role cannot be ledger-bearing")
        columns = role_source.get(role)
        if columns is None:
            out.append(RoleSpec(role, None, None, allow_missing_columns))
            continue
        if not isinstance(columns, list) or len(columns) != 2:
            raise ContractViolation("%s: bad role columns for %s" % (entry.get("artifact_id"), role))
        out.append(RoleSpec(role, str(columns[0]), str(columns[1]), False))
    excluded = entry.get("excluded_roles") or {}
    if not isinstance(excluded, Mapping):
        raise ContractViolation("%s: excluded_roles must be mapping" % entry.get("artifact_id"))
    for role, meta in excluded.items():
        if not isinstance(meta, Mapping) or meta.get("default") != "EXCLUDED":
            raise ContractViolation("%s: M4_ONLY not explicitly excluded" % entry.get("artifact_id"))
        columns = meta.get("columns") or []
        if not isinstance(columns, list) or len(columns) != 2:
            raise ContractViolation("%s: bad excluded columns for %s" % (entry.get("artifact_id"), role))
        out.append(RoleSpec(str(role), str(columns[0]), str(columns[1]), True))
    return tuple(out)


def _artifact_spec(
    entry: Mapping[str, object],
    repo_root: Path,
    non_ledger_status: Optional[str] = None,
) -> ArtifactSpec:
    status = non_ledger_status or entry.get("status")
    present = status != ARTIFACT_NOT_PRESENT_LOCALLY
    fmt = str(entry.get("format") or ("absent" if not present else ""))
    if not fmt:
        raise ContractViolation("%s: missing format" % entry.get("artifact_id"))
    k_source = entry.get("K_source") or {}
    if not isinstance(k_source, Mapping):
        raise ContractViolation("%s: bad K_source" % entry.get("artifact_id"))
    k_value = k_source.get("value") if k_source.get("kind") == "constant" else None
    grid_id = str(k_source.get("grid_id") or "UNKNOWN")
    rq007_split_applicable = entry.get("rq007_split_applicable")
    split_policy = None
    split_filter_column = None
    if rq007_split_applicable is True:
        split_filter_column = _split_join_column(entry)
        split_policy = RequiresRQ007Allowlist(split_filter_column, ("development", "guard"))
    elif rq007_split_applicable is False:
        reason = "artifact_absent_locally" if not present else "non_rq007_artifact"
        split_policy = RQ007SplitNotApplicable(
            str(entry.get("rq007_split_value")), reason
        )
    elif present:
        raise ContractViolation("%s: missing rq007_split_applicable" % entry.get("artifact_id"))
    path = Path(str(entry["path"])) if entry.get("path") else None
    if path is not None and not path.is_absolute():
        path = repo_root / path
    path_glob = str(entry["path_glob"]) if entry.get("path_glob") else None
    if path_glob is not None and not Path(path_glob).is_absolute():
        path_glob = str(repo_root / path_glob)
    recoverability = str(entry.get("recoverability") or ARTIFACT_NOT_PRESENT_LOCALLY)
    if status == ARTIFACT_NOT_PRESENT_LOCALLY:
        recoverability = ARTIFACT_NOT_PRESENT_LOCALLY
    return ArtifactSpec(
        artifact_id=str(entry["artifact_id"]),
        present_locally=present,
        format=fmt,
        path=path,
        path_glob=path_glob,
        row_key_fields=_row_key_fields(entry),
        roles=_role_specs(entry, status == ARTIFACT_NOT_PRESENT_LOCALLY),
        expansion_factor=int(entry.get("expansion_factor") or 1),
        collapse_factor=int(entry.get("collapse_factor") or 1),
        split_policy=split_policy,
        candidate_grid_id=grid_id,
        K=int(k_value) if k_value is not None else None,
        recoverability=recoverability,
        rq007_split_applicable=(
            rq007_split_applicable if rq007_split_applicable in (True, False) else None
        ),
        rq007_split_value=(
            str(entry.get("rq007_split_value")) if entry.get("rq007_split_value") else None
        ),
        not_attempted_rule=entry.get("not_attempted_rule") or {},
        status=str(status) if status else None,
        schema_entry=dict(entry),
        split_filter_column=split_filter_column,
    )


def load_ledger_schema_v2(schema_path: Path) -> LedgerSchema:
    raw = _load_schema_contract(schema_path)
    repo_root = _repo_root_for_schema(Path(schema_path))
    entries_by_id = {a["artifact_id"]: a for a in raw["artifacts"]}
    non_ledger = tuple(dict(a) for a in raw["non_ledger_artifacts"])
    absent_ids = tuple(
        str(a["artifact_id"])
        for a in non_ledger
        if a.get("status") == ARTIFACT_NOT_PRESENT_LOCALLY
    )
    specs = []
    for artifact_id in raw["ledger_bearing_artifact_ids"]:
        specs.append(_artifact_spec(entries_by_id[artifact_id], repo_root))
    for artifact_id in absent_ids:
        specs.append(
            _artifact_spec(entries_by_id[artifact_id], repo_root, ARTIFACT_NOT_PRESENT_LOCALLY)
        )
    by_id = {spec.artifact_id: spec for spec in specs}
    return LedgerSchema(
        schema_version=str(raw["schema_id"]),
        artifacts=tuple(specs),
        artifacts_by_id=by_id,
        ledger_bearing_artifact_ids=tuple(str(v) for v in raw["ledger_bearing_artifact_ids"]),
        non_ledger_artifacts=non_ledger,
        artifacts_absent_locally=absent_ids,
    )


def load_case_allowlist(
    split_csv: Path,
    include: Iterable[str] = ("development", "guard"),
    event_log: Optional[List[str]] = None,
) -> CaseAllowlist:
    include_tuple = tuple(include)
    for split in include_tuple:
        if split not in ("development", "guard"):
            raise ContractViolation("allowlist may include only development/guard")
    split_counts = Counter()
    case_to_split = {}
    allowed = set()
    with Path(split_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        if "case_id" not in (reader.fieldnames or ()) or "split" not in (reader.fieldnames or ()):
            raise ContractViolation("split CSV must contain case_id,split")
        for row in reader:
            case_id = str(row.get("case_id") or "")
            split = str(row.get("split") or "")
            if not case_id or not split:
                raise ContractViolation("empty case_id/split in allowlist source")
            if case_id in case_to_split:
                raise ContractViolation("duplicate case_id in split source: %s" % case_id)
            case_to_split[case_id] = split
            split_counts[split] += 1
            if split in include_tuple:
                allowed.add(case_id)
    if event_log is not None:
        event_log.append("allowlist.loaded")
    return CaseAllowlist._from_loaded_split(
        Path(split_csv),
        include_tuple,
        frozenset(allowed),
        dict(split_counts),
        _sha256(Path(split_csv)),
        case_to_split,
    )


def resolve_artifact_scope(
    spec: ArtifactSpec,
    allowlist: Optional[CaseAllowlist] = None,
    event_log: Optional[List[str]] = None,
) -> FilteredArtifactScope:
    if spec.rq007_split_applicable is True:
        if allowlist is None:
            raise ContractViolation("%s requires CaseAllowlist" % spec.artifact_id)
        validate_case_allowlist_token(allowlist)
        if not isinstance(spec.split_policy, RequiresRQ007Allowlist):
            raise ContractViolation("%s split policy mismatch" % spec.artifact_id)
        scope = AllowlistedArtifactScope._from_schema(
            spec, allowlist, spec.split_policy.join_column, 0, 0
        )
    elif spec.rq007_split_applicable is False:
        reason = "artifact_absent_locally" if not spec.present_locally else "non_rq007_artifact"
        scope = SplitNotApplicableArtifactScope._from_schema(spec, reason)
    else:
        raise ContractViolation("%s missing rq007_split_applicable" % spec.artifact_id)
    if event_log is not None:
        event_log.append("scope.resolved:%s" % spec.artifact_id)
    return scope


def load_execute_permit(run_spec_path: Path, authorization_path: Path) -> ExecutePermit:
    operation_id = "rq015a_concentration_audit"
    run_spec = json.loads(Path(run_spec_path).read_text())
    if run_spec.get("operation_id") != operation_id:
        raise ContractViolation("run spec operation_id mismatch")
    if run_spec.get("execution_authorized") is not True:
        raise ContractViolation("execution_authorized is not true")
    auth = json.loads(Path(authorization_path).read_text())
    authorizations = auth.get("authorizations")
    if not isinstance(authorizations, Mapping):
        raise ContractViolation("authorization file missing authorizations object")
    entry = authorizations.get(operation_id)
    if not isinstance(entry, Mapping):
        raise ContractViolation("authorization entry missing for %s" % operation_id)
    if entry.get("execution_authorized") is not True:
        raise ContractViolation("authorization entry execution_authorized is not true")
    allowed = entry.get("allowed_operations")
    if not isinstance(allowed, list):
        raise ContractViolation("authorization entry allowed_operations must be a list")
    if operation_id not in [str(value) for value in allowed]:
        raise ContractViolation("authorization object does not allow rq015a_concentration_audit")
    return ExecutePermit._from_authorization(
        operation_id,
        True,
        Path(authorization_path),
        _sha256(Path(authorization_path)),
    )


def _make_test_permit_UNSAFE() -> ExecutePermit:
    """Explicit test-only path; production construction must use load_execute_permit()."""
    return ExecutePermit._from_authorization(
        "rq015a_concentration_audit",
        True,
        Path("UNSAFE_TEST_ONLY_DO_NOT_USE_IN_PRODUCTION"),
        "0" * 64,
    )


def _scope_matches_schema(spec: ArtifactSpec, scope: FilteredArtifactScope) -> None:
    if getattr(scope, "artifact_id", None) != spec.artifact_id:
        raise ContractViolation("scope/spec artifact mismatch")
    if spec.rq007_split_applicable is True and not isinstance(scope, AllowlistedArtifactScope):
        raise ContractViolation("%s requires allowlisted scope" % spec.artifact_id)
    if spec.rq007_split_applicable is True:
        scope._validate()
        if isinstance(spec.split_policy, RequiresRQ007Allowlist):
            if tuple(scope.allowlist.included_splits) != tuple(
                spec.split_policy.allowed_splits
            ):
                raise ContractViolation(
                    "%s allowlist included_splits mismatch" % spec.artifact_id
                )
    if spec.rq007_split_applicable is False and not isinstance(
        scope, SplitNotApplicableArtifactScope
    ):
        raise ContractViolation("%s requires split-not-applicable scope" % spec.artifact_id)
    if spec.rq007_split_applicable is False:
        scope._validate()
    if spec.rq007_split_applicable not in (True, False):
        raise ContractViolation("%s missing rq007_split_applicable" % spec.artifact_id)


def _validate_measurement_columns(spec: ArtifactSpec) -> None:
    for role in spec.roles:
        for column in (role.ipv_column, role.ipv_error_column):
            if column and is_forbidden_human_field(column):
                raise ContractViolation("forbidden human/rating column requested: %s" % column)
        if role.measurement_role.startswith("M4_ONLY_") and not role.excluded:
            raise ContractViolation("M4_ONLY channel is not excluded")


def _copy_allowlist_token(allowlist: CaseAllowlist) -> CaseAllowlist:
    token = validate_case_allowlist_token(allowlist)
    return CaseAllowlist._from_loaded_split(
        Path(token.source_path),
        tuple(token.included_splits),
        frozenset(token.allowed_case_ids),
        dict(token.split_counts),
        str(token.source_sha256),
        dict(token.case_to_split),
    )


def _snapshot_allowlisted_scope(
    spec: ArtifactSpec,
    scope: AllowlistedArtifactScope,
) -> AllowlistedArtifactScope:
    token = validate_case_allowlist_source(scope.allowlist)
    return AllowlistedArtifactScope._from_schema(
        spec,
        _copy_allowlist_token(token),
        scope.join_column,
        scope.held_out_parsed_rows,
        scope.unmapped_rows,
    )


class _InMemoryMeasurementReader:
    def __init__(
        self,
        spec: ArtifactSpec,
        scope: FilteredArtifactScope,
        rows: Sequence[Mapping[str, object]],
        event_log: Optional[List[str]] = None,
    ) -> None:
        self.artifact_id = spec.artifact_id
        self.scope = scope
        self._spec = spec
        self._event_log = event_log
        self._rows = self._filter_rows(rows)

    def _log(self, message: str) -> None:
        if self._event_log is not None:
            self._event_log.append(message)

    def _verify_allowlist_source_unchanged(self, stage: str) -> None:
        if isinstance(self.scope, AllowlistedArtifactScope):
            validate_case_allowlist_source(self.scope.allowlist)
            self._log("reader.allowlist_source_verified:%s" % stage)

    def _filter_rows(self, rows: Sequence[Mapping[str, object]]) -> Tuple[Mapping[str, object], ...]:
        self._log("reader.open:%s" % self.artifact_id)
        if isinstance(self.scope, AllowlistedArtifactScope):
            self._log("reader.structural_columns:%s" % self.scope.join_column)
            kept = []
            unmapped = 0
            for row in rows:
                if self.scope.join_column not in row:
                    raise ContractViolation("%s missing join column" % self.artifact_id)
                case_id = _require_allowlist_case_id(
                    self._spec, self.scope.join_column, row[self.scope.join_column]
                )
                split = self.scope.allowlist.case_to_split.get(case_id)
                if split is None:
                    unmapped += 1
                elif case_id in self.scope.allowlist.allowed_case_ids:
                    kept.append(row)
            if unmapped:
                raise ContractViolation("%s unmapped case rows: %d" % (self.artifact_id, unmapped))
            self._log("reader.allowlist_applied:%d" % len(kept))
            return tuple(kept)
        if isinstance(self.scope, SplitNotApplicableArtifactScope):
            self._log("reader.split_not_applicable:%d" % len(rows))
            return tuple(rows)
        raise ContractViolation("unknown scope type")

    def iter_measurement_rows(self) -> Iterator[Mapping[str, object]]:
        self._log("reader.measurement_iter_started:%s" % self.artifact_id)
        self._verify_allowlist_source_unchanged("before")
        try:
            for row in self._rows:
                yield row
        finally:
            self._verify_allowlist_source_unchanged("after")


def _load_limited_rows(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    parquet_part_limit: Optional[int],
    csv_row_limit: Optional[int],
    event_log: Optional[List[str]] = None,
) -> Sequence[Mapping[str, object]]:
    if spec.format == "parquet":
        if parquet_part_limit != 1:
            raise ContractViolation("BUILD_WHILE_DENY allows at most one parquet part")
        matches = sorted(glob.glob(str(spec.path_glob)))
        if not matches:
            raise ContractViolation("no parquet parts matched")
        return _load_parquet_projected_rows(spec, scope, Path(matches[0]), event_log)
    if spec.format == "csv":
        if spec.path is None:
            raise ContractViolation("csv artifact missing path")
        return _load_csv_projected_rows(spec, scope, csv_row_limit, event_log)
    raise ContractViolation("cannot open absent artifact")


def _structural_columns_for_spec(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
) -> Tuple[str, ...]:
    columns = list(spec.row_key_fields)
    if isinstance(scope, AllowlistedArtifactScope):
        columns.append(scope.join_column)
    if spec.split_filter_column:
        columns.append(spec.split_filter_column)
    kind = spec.not_attempted_rule.get("kind")
    if kind == "global_frame_index":
        columns.append("frame_index")
    elif kind == "local_position":
        columns.extend(("case_key", "timestamp_ms", "frame_index"))
    return _dedupe_columns(columns)


def _measurement_columns_for_spec(spec: ArtifactSpec) -> Tuple[str, ...]:
    columns = []
    for role in spec.roles:
        if role.excluded:
            continue
        columns.append(role.ipv_error_column)
    return _dedupe_columns(columns)


def _projected_columns_for_spec(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    structural = _structural_columns_for_spec(spec, scope)
    measurement = _measurement_columns_for_spec(spec)
    return structural, measurement, _dedupe_columns(tuple(structural) + tuple(measurement))


def _log_real_path_event(
    event_log: Optional[List[str]],
    label: str,
    columns: Sequence[str],
) -> None:
    if event_log is not None:
        event_log.append("%s:%s" % (label, ",".join(columns)))


def _require_allowlist_case_id(
    spec: ArtifactSpec,
    join_column: str,
    value: object,
) -> str:
    if value is None:
        raise ContractViolation("%s null join key in %s" % (spec.artifact_id, join_column))
    if not isinstance(value, str):
        raise ContractViolation(
            "%s non-string join key in %s: %s"
            % (spec.artifact_id, join_column, type(value).__name__)
        )
    return value


def _allowed_case_ids_and_row_count_from_structural_rows(
    spec: ArtifactSpec,
    scope: AllowlistedArtifactScope,
    rows: Sequence[Mapping[str, object]],
) -> Tuple[Tuple[str, ...], int]:
    validate_case_allowlist_token(scope.allowlist)
    allowed = set()
    allowed_rows = 0
    unmapped = 0
    for row in rows:
        if scope.join_column not in row:
            raise ContractViolation("%s missing join column" % spec.artifact_id)
        case_id = _require_allowlist_case_id(spec, scope.join_column, row[scope.join_column])
        split = scope.allowlist.case_to_split.get(case_id)
        if split is None:
            unmapped += 1
        elif case_id in scope.allowlist.allowed_case_ids:
            allowed.add(case_id)
            allowed_rows += 1
    if unmapped:
        raise ContractViolation("%s unmapped case rows: %d" % (spec.artifact_id, unmapped))
    return tuple(sorted(allowed)), allowed_rows


def _assert_two_stage_allowed_row_count(
    spec: ArtifactSpec,
    expected_rows: int,
    actual_rows: int,
) -> None:
    if expected_rows != actual_rows:
        raise ContractViolation(
            "%s allowlist row count mismatch: structural=%d measurement=%d"
            % (spec.artifact_id, expected_rows, actual_rows)
        )


def _load_parquet_projected_rows(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    path: Path,
    event_log: Optional[List[str]],
) -> Sequence[Mapping[str, object]]:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    structural, measurement, projected = _projected_columns_for_spec(spec, scope)
    if isinstance(scope, AllowlistedArtifactScope):
        _log_real_path_event(event_log, "reader.real_path.structural_columns", structural)
        structural_table = pq.ParquetFile(path).read(columns=list(structural))
        allowed_ids, allowed_row_count = _allowed_case_ids_and_row_count_from_structural_rows(
            spec, scope, structural_table.to_pylist()
        )
        if event_log is not None:
            event_log.append("reader.real_path.allowlist_applied:%d" % allowed_row_count)
        if allowed_row_count == 0:
            return tuple()
        _log_real_path_event(event_log, "reader.real_path.measurement_columns", measurement)
        table = ds.dataset(str(path), format="parquet").to_table(
            columns=list(projected),
            filter=ds.field(scope.join_column).isin(list(allowed_ids)),
        )
        out = table.to_pylist()
        _assert_two_stage_allowed_row_count(spec, allowed_row_count, len(out))
        return out

    _log_real_path_event(event_log, "reader.real_path.columns", projected)
    return pq.ParquetFile(path).read(columns=list(projected)).to_pylist()


def _load_csv_projected_rows(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    csv_row_limit: Optional[int],
    event_log: Optional[List[str]],
) -> Sequence[Mapping[str, object]]:
    if csv_row_limit is None or csv_row_limit < 1:
        raise ContractViolation(CSV_REQUIRED_ROW_LIMIT_MESSAGE)
    structural, measurement, projected = _projected_columns_for_spec(spec, scope)
    header, header_index = _read_csv_header(spec.path)
    _require_csv_columns(spec, header_index, projected)

    if isinstance(scope, AllowlistedArtifactScope):
        _log_real_path_event(event_log, "reader.real_path.structural_columns", structural)
        allowed_rows = _csv_allowed_row_numbers(spec, scope, csv_row_limit, header_index)
        if event_log is not None:
            event_log.append("reader.real_path.allowlist_applied:%d" % len(allowed_rows))
        if not allowed_rows:
            return tuple()
        _log_real_path_event(event_log, "reader.real_path.measurement_columns", measurement)
        out = _csv_project_allowed_rows(
            spec.path, header, header_index, projected, csv_row_limit, allowed_rows
        )
        _assert_two_stage_allowed_row_count(spec, len(allowed_rows), len(out))
        return out

    _log_real_path_event(event_log, "reader.real_path.columns", projected)
    return _csv_project_allowed_rows(
        spec.path,
        header,
        header_index,
        projected,
        csv_row_limit,
        None,
    )


def _read_csv_header(path: Optional[Path]) -> Tuple[Tuple[str, ...], Mapping[str, int]]:
    if path is None:
        raise ContractViolation("csv artifact missing path")
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise ContractViolation("csv artifact is empty")
    return header, {name: idx for idx, name in enumerate(header)}


def _require_csv_columns(
    spec: ArtifactSpec,
    header_index: Mapping[str, int],
    columns: Sequence[str],
) -> None:
    missing = [column for column in columns if column not in header_index]
    if missing:
        raise ContractViolation(
            "%s missing csv columns: %s" % (spec.artifact_id, ",".join(missing))
        )


def _csv_allowed_row_numbers(
    spec: ArtifactSpec,
    scope: AllowlistedArtifactScope,
    csv_row_limit: int,
    header_index: Mapping[str, int],
) -> Tuple[int, ...]:
    validate_case_allowlist_token(scope.allowlist)
    allowed = []
    unmapped = 0
    join_idx = header_index[scope.join_column]
    with spec.path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row_number, row in enumerate(reader):
            if row_number >= csv_row_limit:
                break
            if join_idx >= len(row):
                raise ContractViolation("%s missing join column" % spec.artifact_id)
            case_id = str(row[join_idx])
            split = scope.allowlist.case_to_split.get(case_id)
            if split is None:
                unmapped += 1
            elif case_id in scope.allowlist.allowed_case_ids:
                allowed.append(row_number)
    if unmapped:
        raise ContractViolation("%s unmapped case rows: %d" % (spec.artifact_id, unmapped))
    return tuple(allowed)


def _csv_project_allowed_rows(
    path: Optional[Path],
    header: Sequence[str],
    header_index: Mapping[str, int],
    columns: Sequence[str],
    csv_row_limit: int,
    allowed_rows: Optional[Sequence[int]],
) -> Sequence[Mapping[str, object]]:
    if path is None:
        raise ContractViolation("csv artifact missing path")
    allowed_set = None if allowed_rows is None else set(allowed_rows)
    out = []
    column_indices = [(column, header_index[column]) for column in columns]
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row_number, row in enumerate(reader):
            if row_number >= csv_row_limit:
                break
            if allowed_set is not None and row_number not in allowed_set:
                continue
            projected = {}
            for column, idx in column_indices:
                if idx >= len(row):
                    raise ContractViolation("csv row missing column %s" % column)
                projected[column] = row[idx]
            out.append(projected)
    return tuple(out)


def open_measurement_reader(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    permit: ExecutePermit,
    source_rows: Optional[Sequence[Mapping[str, object]]] = None,
    event_log: Optional[List[str]] = None,
    parquet_part_limit: Optional[int] = None,
    csv_row_limit: Optional[int] = None,
) -> _InMemoryMeasurementReader:
    if not isinstance(permit, ExecutePermit):
        raise ContractViolation("ExecutePermit instance required")
    if permit.execution_authorized is not True:
        raise ContractViolation("execution permit is not authorized")
    if not spec.present_locally:
        raise ContractViolation("absent artifact cannot open MeasurementReader")
    _scope_matches_schema(spec, scope)
    reader_scope = scope
    if isinstance(scope, AllowlistedArtifactScope):
        reader_scope = _snapshot_allowlisted_scope(spec, scope)
    _validate_measurement_columns(spec)
    rows = source_rows
    if rows is None:
        rows = _load_limited_rows(spec, reader_scope, parquet_part_limit, csv_row_limit, event_log)
    reader = _InMemoryMeasurementReader(spec, reader_scope, rows, event_log)
    _OPENED_MEASUREMENT_READERS.add(reader)
    return reader


def _stringify_key_value(value: object) -> str:
    if value is None:
        raise ContractViolation("product row key contains None")
    return str(value)


def _escape_product_row_key_component(value: object) -> str:
    text = _stringify_key_value(value)
    out = []
    for ch in text:
        if ch in PRODUCT_ROW_KEY_ESCAPED_CHARS:
            out.append(PRODUCT_ROW_KEY_ESCAPE)
        out.append(ch)
    return "".join(out)


def _unescape_product_row_key_component(value: str) -> str:
    out = []
    escaped = False
    for ch in value:
        if escaped:
            if ch not in PRODUCT_ROW_KEY_ESCAPED_CHARS:
                raise ContractViolation("bad product_row_key escape: \\%s" % ch)
            out.append(ch)
            escaped = False
            continue
        if ch == PRODUCT_ROW_KEY_ESCAPE:
            escaped = True
            continue
        out.append(ch)
    if escaped:
        raise ContractViolation("dangling product_row_key escape")
    return "".join(out)


def _split_product_row_key_escaped(value: str, separator: str) -> Tuple[str, ...]:
    parts = []
    start = 0
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == PRODUCT_ROW_KEY_ESCAPE:
            escaped = True
            continue
        if ch == separator:
            parts.append(value[start:idx])
            start = idx + 1
    if escaped:
        raise ContractViolation("dangling product_row_key escape")
    parts.append(value[start:])
    return tuple(parts)


def _split_product_row_key_assignment(part: str) -> Tuple[str, str]:
    escaped = False
    for idx, ch in enumerate(part):
        if escaped:
            escaped = False
            continue
        if ch == PRODUCT_ROW_KEY_ESCAPE:
            escaped = True
            continue
        if ch == PRODUCT_ROW_KEY_ASSIGN:
            return part[:idx], part[idx + 1:]
    if escaped:
        raise ContractViolation("dangling product_row_key escape")
    raise ContractViolation("bad product_row_key part: %s" % part)


def _join_escaped_components(values: Sequence[object]) -> str:
    return PRODUCT_ROW_KEY_SEPARATOR.join(
        _escape_product_row_key_component(value) for value in values
    )


def product_row_key(row: Mapping[str, object], fields: Sequence[str]) -> str:
    parts = []
    for field in fields:
        if field not in row:
            raise ContractViolation("row missing key field %s" % field)
        parts.append(
            "%s=%s"
            % (
                _escape_product_row_key_component(field),
                _escape_product_row_key_component(row[field]),
            )
        )
    return PRODUCT_ROW_KEY_SEPARATOR.join(parts)


def _parse_product_row_key(key: str) -> MutableMapping[str, str]:
    out = {}
    for part in _split_product_row_key_escaped(key, PRODUCT_ROW_KEY_SEPARATOR):
        raw_k, raw_v = _split_product_row_key_assignment(part)
        k = _unescape_product_row_key_component(raw_k)
        v = _unescape_product_row_key_component(raw_v)
        if k in out:
            raise ContractViolation("duplicate product_row_key field: %s" % k)
        out[k] = v
    return out


def derive_aggregation_key(
    artifact_id: str,
    row_key: str,
    measurement_role: str,
) -> Tuple[str, str]:
    parsed = _parse_product_row_key(row_key)
    if artifact_id == "interhub_sigma01_hw4_timeseries":
        return measurement_role, "sigma01_hw4"
    if artifact_id == "rq009_feature_matrix":
        if "perspective" not in parsed or "source_dataset" not in parsed:
            raise ContractViolation("feature matrix row key missing perspective/source_dataset")
        return parsed["perspective"], _join_escaped_components(
            (measurement_role, parsed["source_dataset"])
        )
    if artifact_id == "onsite_dense_timeseries":
        if "case_key" not in parsed:
            raise ContractViolation("OnSite row key missing case_key")
        return measurement_role, parsed["case_key"]
    raise ContractViolation("no aggregation key derivation for %s" % artifact_id)


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractViolation("bool is not numeric")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        value = stripped
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ContractViolation("cannot parse float from %r" % (value,))
    if not math.isfinite(parsed):
        raise ContractViolation("non-finite float %r" % (value,))
    return parsed


def _parse_integral_field(
    value: object,
    field_name: str,
    artifact_id: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise ContractViolation(
            "%s %s must be an integer, got %r" % (artifact_id, field_name, value)
        )
    if isinstance(value, numbers.Integral):
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = int(stripped)
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "%s %s parse failed: %s" % (artifact_id, field_name, exc)
            )
    else:
        raise ContractViolation(
            "%s %s must be an integer, got %r" % (artifact_id, field_name, value)
        )
    if parsed < minimum:
        raise ContractViolation(
            "%s %s must be >= %s" % (artifact_id, field_name, minimum)
        )
    return parsed


def _is_not_attempted(
    spec: ArtifactSpec,
    row: Mapping[str, object],
    local_position: Optional[int],
) -> bool:
    kind = spec.not_attempted_rule.get("kind")
    if kind == "global_frame_index":
        if "frame_index" not in row:
            raise ContractViolation("%s missing frame_index" % spec.artifact_id)
        frame_index = _parse_integral_field(
            row["frame_index"], "frame_index", spec.artifact_id, 0
        )
        min_observation = _parse_integral_field(
            spec.not_attempted_rule.get("min_observation", 4),
            "min_observation",
            spec.artifact_id,
            0,
        )
        return frame_index < min_observation
    if kind == "local_position":
        if local_position is None:
            raise ContractViolation("%s missing local_position" % spec.artifact_id)
        return local_position < 4
    if kind == "none_expected":
        return False
    raise ContractViolation("%s unsupported not_attempted_rule %r" % (spec.artifact_id, kind))


def _local_position_map(
    spec: ArtifactSpec,
    rows: Sequence[Mapping[str, object]],
) -> Mapping[int, int]:
    if spec.not_attempted_rule.get("kind") != "local_position":
        return {}
    if not rows:
        raise ContractViolation("%s local_position requires non-empty rows" % spec.artifact_id)
    by_case = defaultdict(list)
    for idx, row in enumerate(rows):
        if "case_key" not in row or "timestamp_ms" not in row or "frame_index" not in row:
            raise ContractViolation("OnSite local_position requires case_key,timestamp_ms,frame_index")
        by_case[str(row["case_key"])].append(
            (idx, (row["timestamp_ms"], row["frame_index"]))
        )
    positions = {}
    for _, items in by_case.items():
        pos = local_positions([pair for _, pair in items])
        for (idx, _), local_pos in zip(items, pos):
            positions[idx] = local_pos
    return positions


def _require_internal_measurement_reader(reader: object) -> _InMemoryMeasurementReader:
    if type(reader) is not _InMemoryMeasurementReader or reader not in _OPENED_MEASUREMENT_READERS:
        raise ContractViolation("internal MeasurementReader from open_measurement_reader required")
    return reader


def _row_case_and_split(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    row: Mapping[str, object],
) -> Tuple[Optional[str], str]:
    if isinstance(scope, AllowlistedArtifactScope):
        if scope.join_column not in row:
            raise ContractViolation("%s missing join column" % spec.artifact_id)
        case_id = _require_allowlist_case_id(spec, scope.join_column, row[scope.join_column])
        if case_id not in scope.allowlist.allowed_case_ids:
            raise ContractViolation("%s row escaped allowlist" % spec.artifact_id)
        split = scope.allowlist.case_to_split.get(case_id)
        if split not in scope.allowlist.included_splits:
            raise ContractViolation("%s row has non-included split" % spec.artifact_id)
        return case_id, str(split)
    if isinstance(scope, SplitNotApplicableArtifactScope):
        return None, RQ007_SPLIT_NOT_APPLICABLE
    raise ContractViolation("unknown scope")


def _status_and_values(
    spec: ArtifactSpec,
    role: RoleSpec,
    row: Mapping[str, object],
    is_d0: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[str]]:
    if role.ipv_error_column is None:
        ipv_error = None
    else:
        if role.ipv_error_column not in row:
            raise ContractViolation("%s missing %s" % (spec.artifact_id, role.ipv_error_column))
        ipv_error = _parse_optional_float(row[role.ipv_error_column])
    q_value = _q_eff(ipv_error, spec.K) if ipv_error is not None else None
    k_eff = k_eff_from_error(ipv_error) if ipv_error is not None else None
    if is_d0:
        return ipv_error, k_eff, q_value, NOT_ATTEMPTED, "D0_WARMUP"
    if ipv_error is None:
        return ipv_error, k_eff, q_value, UNKNOWN, "EMPTY_CELL_UNEXPLAINED"
    if q_value is None:
        return ipv_error, k_eff, q_value, UNKNOWN, "DEGENERATE_IPV_ERROR"
    return ipv_error, k_eff, q_value, ATTEMPTED, None


def sort_l1_rows(artifact_id: str, rows: Iterable[L1LedgerRow]) -> SortedL1LedgerRows:
    return SortedL1LedgerRows(
        artifact_id=artifact_id,
        rows=tuple(sorted(tuple(rows), key=l1_sort_key)),
        sort_key=L1_SORT_KEY,
    )


def assert_l1_conservation(
    spec: ArtifactSpec,
    physical_rows: int,
    rows: Sequence[L1LedgerRow],
) -> object:
    status_counts = Counter(row.attempt_status for row in rows)
    recoverability_counts = Counter(row.recoverability for row in rows)
    return check_l1_conservation_counts(
        spec.artifact_id,
        physical_rows,
        spec.expansion_factor,
        spec.collapse_factor,
        len(rows),
        dict(status_counts),
        dict(recoverability_counts),
    )


def check_l1_conservation_counts(
    artifact_id: str,
    physical_rows: int,
    expansion_factor: int,
    collapse_factor: int,
    measurement_rows_observed: int,
    status_counts: Mapping[str, int],
    recoverability_counts: Mapping[str, int],
) -> object:
    if physical_rows < 0 or expansion_factor < 1 or collapse_factor < 1:
        raise ContractViolation("%s: invalid factors" % artifact_id)
    numerator = physical_rows * expansion_factor
    if numerator % collapse_factor != 0:
        raise ContractViolation(
            "%s: %d not divisible by collapse_factor %d"
            % (artifact_id, numerator, collapse_factor)
        )
    expected = numerator // collapse_factor
    if measurement_rows_observed != expected:
        raise ContractViolation(
            "%s: identity_1 failed %d != %d"
            % (artifact_id, measurement_rows_observed, expected)
        )
    status_total = sum(status_counts.values())
    if status_total != measurement_rows_observed:
        raise ContractViolation(
            "%s: identity_2 failed %d != %d"
            % (artifact_id, status_total, measurement_rows_observed)
        )
    recoverability_total = sum(recoverability_counts.values())
    if recoverability_total != measurement_rows_observed:
        raise ContractViolation(
            "%s: identity_3 failed %d != %d"
            % (artifact_id, recoverability_total, measurement_rows_observed)
        )
    return check_conservation(
        artifact_id,
        physical_rows,
        expansion_factor,
        collapse_factor,
        dict(status_counts),
        dict(recoverability_counts),
    )


def build_l1_for_artifact(
    spec: ArtifactSpec,
    scope: FilteredArtifactScope,
    reader: _InMemoryMeasurementReader,
) -> SortedL1LedgerRows:
    reader = _require_internal_measurement_reader(reader)
    if reader.artifact_id != spec.artifact_id:
        raise ContractViolation("reader/spec artifact mismatch")
    _scope_matches_schema(spec, scope)
    _scope_matches_schema(spec, reader.scope)
    physical_rows = list(reader.iter_measurement_rows())
    local_pos = _local_position_map(spec, physical_rows)
    rows = []
    for idx, physical_row in enumerate(physical_rows):
        case_id, split = _row_case_and_split(spec, reader.scope, physical_row)
        row_key = product_row_key(physical_row, spec.row_key_fields)
        is_d0 = _is_not_attempted(spec, physical_row, local_pos.get(idx))
        for role in spec.roles:
            if role.excluded:
                continue
            perspective, configuration = derive_aggregation_key(
                spec.artifact_id, row_key, role.measurement_role
            )
            ipv_error, k_eff, q_value, status, reason = _status_and_values(
                spec, role, physical_row, is_d0
            )
            rows.append(
                L1LedgerRow(
                    artifact_id=spec.artifact_id,
                    product_row_key=row_key,
                    measurement_role=role.measurement_role,
                    case_id=case_id,
                    rq007_split=split,
                    ipv_error=ipv_error,
                    K=spec.K,
                    candidate_grid_id=spec.candidate_grid_id,
                    k_eff=k_eff,
                    q_eff=q_value,
                    attempt_status=status,
                    reason_code=reason,
                    recoverability=spec.recoverability,
                    ledger_schema_version=SCHEMA_VERSION,
                    aggregation_perspective=perspective,
                    aggregation_configuration=configuration,
                )
            )
    report = assert_l1_conservation(spec, len(physical_rows), rows)
    if spec.not_attempted_rule.get("kind") == "none_expected":
        if any(row.attempt_status == NOT_ATTEMPTED for row in rows):
            raise ContractViolation("%s unexpectedly has NOT_ATTEMPTED rows" % spec.artifact_id)
    sorted_rows = sort_l1_rows(spec.artifact_id, rows)
    # Keep the conservation report reachable to callers that use build_ledger().
    object.__setattr__(sorted_rows, "_conservation_report", report)
    return sorted_rows


def build_absent_artifact_coverage(spec: ArtifactSpec) -> AbsentArtifactCoverage:
    if spec.status != ARTIFACT_NOT_PRESENT_LOCALLY:
        raise ContractViolation("%s is not schema-derived absent-local" % spec.artifact_id)
    return AbsentArtifactCoverage(
        artifact_id=spec.artifact_id,
        attempt_status=UNKNOWN,
        recoverability=ARTIFACT_NOT_PRESENT_LOCALLY,
        reason_code=ARTIFACT_NOT_PRESENT_LOCALLY,
        schema_status=ARTIFACT_NOT_PRESENT_LOCALLY,
    )


def aggregate_l1_to_l2(rows: SortedL1LedgerRows) -> SortedL2Units:
    if not rows.rows:
        raise ContractViolation("SortedL1LedgerRows cannot be empty")
    dicts = [row.to_contract_dict() for row in rows.rows]
    actual_artifact_id = assert_single_artifact(dicts)
    if actual_artifact_id != rows.artifact_id:
        raise ContractViolation(
            "SortedL1LedgerRows artifact_id mismatch: container=%s units=%s"
            % (rows.artifact_id, actual_artifact_id)
        )
    units = tuple(
        sorted(
            aggregate_l2(dicts),
            key=lambda u: (
                getattr(u, "case_id") is not None,
                getattr(u, "case_id") or "",
                getattr(u, "perspective"),
                getattr(u, "configuration"),
            ),
        )
    )
    return SortedL2Units(rows.artifact_id, units, L2_SORT_KEY)


def _l2_unit_artifact_id(unit: object) -> object:
    if isinstance(unit, Mapping):
        if "artifact_id" not in unit:
            raise ContractViolation("L2 unit missing artifact_id")
        return unit["artifact_id"]
    try:
        return getattr(unit, "artifact_id")
    except AttributeError:
        raise ContractViolation("L2 unit missing artifact_id")


def _assert_l2_units_match_container(units: SortedL2Units) -> None:
    container_artifact_id = validate_artifact_id(units.artifact_id)
    artifact_ids = []
    for unit in units.units:
        artifact_ids.append(validate_artifact_id(_l2_unit_artifact_id(unit)))
    distinct = set(artifact_ids)
    if len(distinct) > 1:
        raise ContractViolation(
            "cross-artifact pooling forbidden; got %s"
            % sorted(map(str, distinct))
        )
    if distinct and next(iter(distinct)) != container_artifact_id:
        raise ContractViolation(
            "SortedL2Units artifact_id mismatch: container=%s units=%s"
            % (container_artifact_id, next(iter(distinct)))
        )


def aggregate_l2_to_l3(units: SortedL2Units) -> SortedL3Units:
    if not units.units:
        raise ContractViolation("SortedL2Units cannot be empty")
    _assert_l2_units_match_container(units)
    out = tuple(sorted(aggregate_l3(units.units), key=lambda u: getattr(u, "case_id") or ""))
    return SortedL3Units(units.artifact_id, out, "artifact_id,case_id")


def build_ledger(
    schema: LedgerSchema,
    allowlist: CaseAllowlist,
    permit: ExecutePermit,
    source_rows_by_artifact: Mapping[str, Sequence[Mapping[str, object]]],
) -> LedgerBuildResult:
    l1_by_artifact = {}
    conservation = {}
    for artifact_id in schema.ledger_bearing_artifact_ids:
        spec = schema.artifacts_by_id[artifact_id]
        scope = resolve_artifact_scope(spec, allowlist if spec.rq007_split_applicable else None)
        reader = open_measurement_reader(
            spec, scope, permit, source_rows=source_rows_by_artifact.get(artifact_id)
        )
        l1 = build_l1_for_artifact(spec, scope, reader)
        l1_by_artifact[artifact_id] = l1
        conservation[artifact_id] = getattr(l1, "_conservation_report")
    absent = tuple(
        build_absent_artifact_coverage(schema.artifacts_by_id[artifact_id])
        for artifact_id in schema.artifacts_absent_locally
    )
    return LedgerBuildResult(
        l1_by_artifact=l1_by_artifact,
        absent_artifacts=absent,
        conservation=conservation,
        aggregation_key_derivation=AGGREGATION_KEY_DERIVATION,
        held_out_parsed_rows=0,
    )
