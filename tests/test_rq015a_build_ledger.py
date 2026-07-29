"""RQ015A T1 ledger-builder fixtures.

All tests use synthetic rows only; no production audit data is scanned.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import pickle
import sys
from pathlib import Path
from typing import Iterator, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

from build_ledger import (  # noqa: E402
    aggregate_l1_to_l2,
    aggregate_l2_to_l3,
    build_absent_artifact_coverage,
    build_l1_for_artifact,
    check_l1_conservation_counts,
    derive_aggregation_key,
    load_case_allowlist,
    load_execute_permit,
    load_ledger_schema_v2,
    open_measurement_reader,
    product_row_key,
    resolve_artifact_scope,
    sort_l1_rows,
    _make_test_permit_UNSAFE,
)
from rq015a_contracts import ATTEMPTED, NOT_ATTEMPTED, UNKNOWN, ContractViolation, L2Unit  # noqa: E402
from rq015a_types import (  # noqa: E402
    ARTIFACT_NOT_PRESENT_LOCALLY,
    AllowlistedArtifactScope,
    CaseAllowlist,
    L1LedgerRow,
    RQ007_SPLIT_NOT_APPLICABLE,
    SortedL1LedgerRows,
    SortedL2Units,
    SplitNotApplicableArtifactScope,
    StructuralColumnSet,
)


SCHEMA = ROOT / "reports" / "plans" / "RQ015A_ledger_schema_v2.json"
RUN_SPEC = ROOT / "reports" / "plans" / "RQ015A_run_spec_v1.json"
AUTH = ROOT / "configs" / "research_authorization.json"


def _schema():
    return load_ledger_schema_v2(SCHEMA)


def _split_csv(tmp_path: Path) -> Path:
    path = tmp_path / "case_split_assignment.csv"
    path.write_text(
        "case_id,split\n"
        "case_dev,development\n"
        "case_guard,guard\n"
        "case_hold,held_out\n",
        encoding="utf-8",
    )
    return path


def _allowlist(tmp_path: Path, event_log=None):
    return load_case_allowlist(_split_csv(tmp_path), event_log=event_log)


def _permit():
    return _make_test_permit_UNSAFE()


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SpyRow(Mapping[str, object]):
    def __init__(self, data, event_log, measurement_columns):
        self._data = dict(data)
        self._event_log = event_log
        self._measurement_columns = set(measurement_columns)

    def __getitem__(self, key: str) -> object:
        if key in self._measurement_columns and not any(
            item.startswith("reader.allowlist_applied") for item in self._event_log
        ):
            raise AssertionError("measurement column read before allowlist filtering")
        self._event_log.append("field:%s" % key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _feature_row(case_key="case_dev", err="0.5"):
    return {
        "case_key": case_key,
        "anchor_frame_index": "7",
        "perspective": "key_agent_1",
        "source_dataset": "interhub",
        "counterpart_ipv_current": "0.1",
        "counterpart_ipv_error_current": err,
        "target_ipv_future": "0.2",
        "target_ipv_error_future": err,
        "fold": "test",
    }


def _build(spec, scope, rows, event_log=None):
    reader = open_measurement_reader(
        spec, scope, _permit(), source_rows=rows, event_log=event_log
    )
    return build_l1_for_artifact(spec, scope, reader)


def test_r1_allowlist_filter_happens_before_measurement_column_access(tmp_path):
    schema = _schema()
    event_log = []
    allowlist = _allowlist(tmp_path, event_log)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    scope = resolve_artifact_scope(spec, allowlist, event_log)
    measurement_columns = {
        "counterpart_ipv_error_current",
        "target_ipv_error_future",
        "counterpart_ipv_current",
        "target_ipv_future",
    }
    rows = [
        SpyRow(_feature_row("case_dev", "0.5"), event_log, measurement_columns),
        SpyRow(_feature_row("case_hold", "0.5"), event_log, measurement_columns),
    ]

    out = _build(spec, scope, rows, event_log)

    assert len(out.rows) == 2
    applied = next(i for i, item in enumerate(event_log) if item.startswith("reader.allowlist_applied"))
    first_measure = next(i for i, item in enumerate(event_log) if item in {
        "field:counterpart_ipv_error_current", "field:target_ipv_error_future"
    })
    assert event_log.index("allowlist.loaded") < event_log.index("scope.resolved:rq009_feature_matrix")
    assert applied < first_measure
    assert {row.case_id for row in out.rows} == {"case_dev"}


def test_m1_split_not_applicable_impersonation_is_rejected(tmp_path):
    schema = _schema()
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    bad_scope = object.__new__(SplitNotApplicableArtifactScope)
    object.__setattr__(bad_scope, "artifact_id", "rq009_feature_matrix")
    object.__setattr__(bad_scope, "rq007_split_value", RQ007_SPLIT_NOT_APPLICABLE)
    object.__setattr__(bad_scope, "reason", "non_rq007_artifact")
    with pytest.raises(ContractViolation, match="requires allowlisted scope"):
        open_measurement_reader(spec, bad_scope, _permit(), source_rows=[_feature_row()])


def test_sigma01_d0_global_frame_index_and_warmup_priority(tmp_path):
    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["interhub_sigma01_hw4_timeseries"]
    scope = resolve_artifact_scope(spec, allowlist)
    rows = [{
        "scene_unique_id": "case_dev",
        "frame_index": "3",
        "ipv_key_agent_1": "0.0",
        "ipv_key_agent_1_error": "1.0",
        "ipv_key_agent_2": "0.0",
        "ipv_key_agent_2_error": "1.0",
    }]

    out = _build(spec, scope, rows)

    assert [row.attempt_status for row in out.rows] == [NOT_ATTEMPTED, NOT_ATTEMPTED]
    assert {row.reason_code for row in out.rows} == {"D0_WARMUP"}
    assert all(row.q_eff is None for row in out.rows)


def test_feature_matrix_has_no_d0_and_excludes_m4_only(tmp_path):
    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    scope = resolve_artifact_scope(spec, allowlist)

    out = _build(spec, scope, [_feature_row()])

    assert [row.measurement_role for row in out.rows] == [
        "counterpart_current", "target_future"
    ]
    assert all(row.attempt_status == ATTEMPTED for row in out.rows)


def test_onsite_d0_uses_local_position_with_nonzero_discontinuous_frames():
    schema = _schema()
    spec = schema.artifacts_by_id["onsite_dense_timeseries"]
    scope = resolve_artifact_scope(spec)
    rows = []
    for timestamp, frame in [(1000, 101), (1040, 103), (1020, 102), (1100, 120), (1300, 140)]:
        rows.append({
            "case_key": "onsite_case",
            "frame_index": str(frame),
            "timestamp_ms": str(timestamp),
            "ipv_ego_hw4_error": "" if timestamp < 1300 else "0.5",
            "ipv_ego_hw10_error": "" if timestamp < 1300 else "0.5",
            "ipv_counterpart_hw4_error": "" if timestamp < 1300 else "0.5",
            "ipv_counterpart_hw10_error": "" if timestamp < 1300 else "0.5",
        })

    out = _build(spec, scope, rows)

    assert sum(1 for row in out.rows if row.attempt_status == NOT_ATTEMPTED) == 16
    assert sum(1 for row in out.rows if row.attempt_status == ATTEMPTED) == 4
    assert all(int(row.product_row_key.split("frame_index=", 1)[1].split("|", 1)[0]) >= 101
               for row in out.rows)


def test_conservation_identity_failures_are_independent():
    with pytest.raises(ContractViolation, match="identity_1"):
        check_l1_conservation_counts("x", 10, 2, 1, 19, {ATTEMPTED: 19}, {"L1_DIRECT": 19})
    with pytest.raises(ContractViolation, match="identity_2"):
        check_l1_conservation_counts("x", 10, 2, 1, 20, {ATTEMPTED: 19}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation, match="identity_3"):
        check_l1_conservation_counts("x", 10, 2, 1, 20, {ATTEMPTED: 20}, {"L1_DIRECT": 19})


def test_unmapped_case_fails_closed_before_measurement(tmp_path):
    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    scope = resolve_artifact_scope(spec, allowlist)
    with pytest.raises(ContractViolation, match="unmapped case rows"):
        open_measurement_reader(spec, scope, _permit(), source_rows=[_feature_row("missing_case")])


@pytest.mark.parametrize("bad_case_id", [123, 1.25, True])
def test_non_string_allowlisted_join_key_fails_closed(tmp_path, bad_case_id):
    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    scope = resolve_artifact_scope(spec, allowlist)

    with pytest.raises(ContractViolation, match="non-string join key"):
        open_measurement_reader(spec, scope, _permit(), source_rows=[_feature_row(bad_case_id)])


def test_real_parquet_null_join_key_fails_closed_before_measurement(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    schema = _schema()
    event_log = []
    allowlist = _allowlist(tmp_path, event_log)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    parquet_path = tmp_path / "feature_null_case.parquet"
    pq.write_table(pa.Table.from_pylist([_feature_row(None, "0.5")]), parquet_path)
    spec = dataclasses.replace(spec, path_glob=str(parquet_path))
    scope = resolve_artifact_scope(spec, allowlist, event_log)

    with pytest.raises(ContractViolation, match="null join key"):
        open_measurement_reader(
            spec,
            scope,
            _permit(),
            event_log=event_log,
            parquet_part_limit=1,
        )
    assert not any(item.startswith("reader.real_path.measurement_columns") for item in event_log)


def test_real_parquet_two_stage_allowed_row_count_mismatch_fails(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ds = pytest.importorskip("pyarrow.dataset")

    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    parquet_path = tmp_path / "feature_part.parquet"
    pq.write_table(pa.Table.from_pylist([_feature_row("case_dev", "0.5")]), parquet_path)
    spec = dataclasses.replace(spec, path_glob=str(parquet_path))
    scope = resolve_artifact_scope(spec, allowlist)

    class EmptyDataset:
        def to_table(self, columns, filter):
            return pa.Table.from_pydict({column: [] for column in columns})

    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: EmptyDataset())

    with pytest.raises(ContractViolation, match="allowlist row count mismatch"):
        open_measurement_reader(spec, scope, _permit(), parquet_part_limit=1)


def test_onsite_empty_string_is_unknown_not_zero_after_d0():
    schema = _schema()
    spec = schema.artifacts_by_id["onsite_dense_timeseries"]
    scope = resolve_artifact_scope(spec)
    rows = []
    for i in range(5):
        rows.append({
            "case_key": "onsite_empty",
            "frame_index": str(101 + i * 2),
            "timestamp_ms": str(1000 + i * 20),
            "ipv_ego_hw4_error": "",
            "ipv_ego_hw10_error": "",
            "ipv_counterpart_hw4_error": "",
            "ipv_counterpart_hw10_error": "",
        })

    out = _build(spec, scope, rows)
    last_rows = [row for row in out.rows if "frame_index=109" in row.product_row_key]

    assert len(last_rows) == 4
    assert all(row.attempt_status == UNKNOWN for row in last_rows)
    assert all(row.q_eff is None and row.k_eff is None for row in last_rows)


def test_absent_artifacts_are_schema_derived_and_not_silently_skipped():
    schema = _schema()

    assert schema.artifacts_absent_locally == (
        "wod_rq010b_full479_audited",
        "wod_phase1_phase1b_10hz_schemeb",
        "rq014_g2r_anchor_scores",
    )
    coverages = [
        build_absent_artifact_coverage(schema.artifacts_by_id[artifact_id])
        for artifact_id in schema.artifacts_absent_locally
    ]
    assert [coverage.recoverability for coverage in coverages] == [
        ARTIFACT_NOT_PRESENT_LOCALLY,
        ARTIFACT_NOT_PRESENT_LOCALLY,
        ARTIFACT_NOT_PRESENT_LOCALLY,
    ]
    assert all(coverage.attempt_status == UNKNOWN for coverage in coverages)


def test_l1_rows_require_artifact_and_cross_artifact_aggregation_fails():
    with pytest.raises(ContractViolation, match="artifact_id"):
        L1LedgerRow(
            artifact_id="",
            product_row_key="k=1",
            measurement_role="agent_1",
            case_id="case_dev",
            rq007_split="development",
            ipv_error=0.5,
            K=7,
            candidate_grid_id="legacy7_pi_over_8",
            k_eff=4.0,
            q_eff=4.0 / 7.0,
            attempt_status=ATTEMPTED,
            reason_code=None,
            recoverability="L1_DIRECT",
            ledger_schema_version="rq015a-concentration-ledger-v2",
            aggregation_perspective="agent_1",
            aggregation_configuration="sigma01_hw4",
        )
    rows = (
        _l1("interhub_sigma01_hw4_timeseries", "case_dev", "k=1"),
        _l1("rq009_feature_matrix", "case_dev", "k=2"),
    )
    with pytest.raises(ContractViolation, match="mixed artifact"):
        SortedL1LedgerRows("interhub_sigma01_hw4_timeseries", rows, "artifact_id,case_id,product_row_key,measurement_role")


def test_l1_wrapper_artifact_label_cannot_hide_rows_from_another_artifact():
    rows = (_l1("rq009_feature_matrix", "case_dev", "k=1"),)
    forged = object.__new__(SortedL1LedgerRows)
    object.__setattr__(forged, "artifact_id", "interhub_sigma01_hw4_timeseries")
    object.__setattr__(forged, "rows", rows)
    object.__setattr__(forged, "sort_key", "artifact_id,case_id,product_row_key,measurement_role")

    with pytest.raises(ContractViolation, match="artifact_id mismatch"):
        aggregate_l1_to_l2(forged)


def test_l2_wrapper_artifact_label_cannot_hide_units_from_another_artifact():
    forged = SortedL2Units(
        "interhub_sigma01_hw4_timeseries",
        (
            L2Unit(
                case_id="case_dev",
                perspective="agent_1",
                configuration="cfg",
                n_l1=5,
                n_attempted=5,
                n_unknown=0,
                mean_q_eff=0.5,
                status="OK",
                artifact_id="rq009_feature_matrix",
            ),
        ),
        "artifact_id,case_id,perspective,configuration",
    )

    with pytest.raises(ContractViolation, match="artifact_id mismatch"):
        aggregate_l2_to_l3(forged)


def test_aggregate_l1_to_l2_rejects_empty_container():
    rows = SortedL1LedgerRows(
        "rq009_feature_matrix",
        (),
        "artifact_id,case_id,product_row_key,measurement_role",
    )

    with pytest.raises(ContractViolation, match="empty"):
        aggregate_l1_to_l2(rows)


def test_aggregate_l2_to_l3_rejects_empty_container():
    units = SortedL2Units(
        "rq009_feature_matrix",
        (),
        "artifact_id,case_id,perspective,configuration",
    )

    with pytest.raises(ContractViolation, match="empty"):
        aggregate_l2_to_l3(units)


def _l1(artifact_id, case_id, key, role="agent_1"):
    return L1LedgerRow(
        artifact_id=artifact_id,
        product_row_key=key,
        measurement_role=role,
        case_id=case_id,
        rq007_split="development" if case_id is not None else RQ007_SPLIT_NOT_APPLICABLE,
        ipv_error=0.5,
        K=7,
        candidate_grid_id="legacy7_pi_over_8",
        k_eff=4.0,
        q_eff=4.0 / 7.0,
        attempt_status=ATTEMPTED,
        reason_code=None,
        recoverability="L1_DIRECT",
        ledger_schema_version="rq015a-concentration-ledger-v2",
        aggregation_perspective=role,
        aggregation_configuration="cfg",
    )


def test_m5_l1_sort_handles_mixed_none_and_string_case_ids_stably():
    rows = [
        _l1("onsite_dense_timeseries", "case_b", "k=3", "ego_hw4"),
        _l1("onsite_dense_timeseries", None, "k=2", "ego_hw4"),
        _l1("onsite_dense_timeseries", "case_a", "k=1", "ego_hw4"),
    ]

    out = sort_l1_rows("onsite_dense_timeseries", rows)

    assert [row.case_id for row in out.rows] == [None, "case_a", "case_b"]


def test_m6_aggregation_key_derivation_is_fixed_for_three_artifacts():
    assert derive_aggregation_key(
        "interhub_sigma01_hw4_timeseries",
        "scene_unique_id=case_dev|frame_index=5",
        "agent_2",
    ) == ("agent_2", "sigma01_hw4")
    assert derive_aggregation_key(
        "rq009_feature_matrix",
        "case_key=case_dev|anchor_frame_index=7|perspective=key_agent_2|source_dataset=interhub",
        "target_future",
    ) == ("key_agent_2", "target_future|interhub")
    assert derive_aggregation_key(
        "onsite_dense_timeseries",
        "case_key=onsite_case|frame_index=101|timestamp_ms=1000",
        "ego_hw10",
    ) == ("ego_hw10", "onsite_case")


def test_product_row_key_escapes_separator_escape_and_prevents_collisions():
    fields = ("case_key", "anchor_frame_index", "perspective", "source_dataset")
    left = _feature_row("case_dev")
    left["perspective"] = "p"
    left["source_dataset"] = "a|source_dataset=b"
    right = _feature_row("case_dev")
    right["perspective"] = "p"
    right["source_dataset"] = "b"
    escaped = _feature_row("case_dev")
    escaped["perspective"] = "p"
    escaped["source_dataset"] = r"a\b|c"

    left_key = product_row_key(left, fields)
    right_key = product_row_key(right, fields)
    escaped_key = product_row_key(escaped, fields)

    assert r"\|" in left_key and r"\=" in left_key
    assert r"\\" in escaped_key and r"\|" in escaped_key
    assert left_key != right_key
    assert derive_aggregation_key("rq009_feature_matrix", left_key, "target_future") != (
        derive_aggregation_key("rq009_feature_matrix", right_key, "target_future")
    )
    with pytest.raises(ContractViolation, match="duplicate product_row_key field"):
        derive_aggregation_key(
            "rq009_feature_matrix",
            "case_key=case_dev|anchor_frame_index=7|perspective=p|source_dataset=a|source_dataset=b",
            "target_future",
        )


def test_literal_runtime_invariants_are_enforced(tmp_path):
    schema = _schema()
    allowlist = _allowlist(tmp_path)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    with pytest.raises(ContractViolation, match="held_out_parsed_rows"):
        from rq015a_types import AllowlistedArtifactScope

        AllowlistedArtifactScope._from_schema(spec, allowlist, "case_key", held_out_parsed_rows=1)
    with pytest.raises(ContractViolation, match="forbids_measurement_fields"):
        StructuralColumnSet(("case_key",), False)
    with pytest.raises(ContractViolation, match="measurement field"):
        StructuralColumnSet(("case_key", "counterpart_ipv_error_current"), True)


def test_execute_false_cannot_construct_measurement_reader_and_prod_loader_not_unsafe():
    with pytest.raises(ContractViolation, match="execution_authorized"):
        load_execute_permit(RUN_SPEC, AUTH)
    assert "_make_test_permit_UNSAFE" not in inspect.getsource(load_execute_permit)


def test_execute_permit_rejects_false_target_authorization_entry(tmp_path):
    run_spec = _write_json(
        tmp_path / "run_spec.json",
        {
            "operation_id": "rq015a_concentration_audit",
            "execution_authorized": True,
        },
    )
    auth = _write_json(
        tmp_path / "auth.json",
        {
            "authorizations": {
                "rq015a_concentration_audit": {
                    "execution_authorized": False,
                    "allowed_operations": ["rq015a_concentration_audit"],
                }
            }
        },
    )

    with pytest.raises(ContractViolation, match="authorization entry execution_authorized"):
        load_execute_permit(run_spec, auth)


def test_execute_permit_rejects_irrelevant_flattened_allowed_operation(tmp_path):
    run_spec = _write_json(
        tmp_path / "run_spec.json",
        {
            "operation_id": "rq015a_concentration_audit",
            "execution_authorized": True,
        },
    )
    auth = _write_json(
        tmp_path / "auth.json",
        {
            "authorizations": {
                "rq015a_concentration_audit": {
                    "execution_authorized": True,
                    "allowed_operations": [],
                },
                "unrelated_operation": {
                    "execution_authorized": True,
                    "allowed_operations": ["rq015a_concentration_audit"],
                },
            }
        },
    )

    with pytest.raises(ContractViolation, match="does not allow"):
        load_execute_permit(run_spec, auth)


def test_real_parquet_path_projects_measurements_after_allowlist(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    schema = _schema()
    event_log = []
    allowlist = _allowlist(tmp_path, event_log)
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    parquet_path = tmp_path / "feature_part.parquet"
    pq.write_table(
        pa.Table.from_pylist([
            _feature_row("case_dev", "0.5"),
            _feature_row("case_hold", "0.5"),
        ]),
        parquet_path,
    )
    spec = dataclasses.replace(spec, path_glob=str(parquet_path))
    scope = resolve_artifact_scope(spec, allowlist, event_log)

    reader = open_measurement_reader(
        spec,
        scope,
        _permit(),
        event_log=event_log,
        parquet_part_limit=1,
    )
    out = build_l1_for_artifact(spec, scope, reader)

    assert {row.case_id for row in out.rows} == {"case_dev"}
    structural_idx = next(
        i for i, item in enumerate(event_log)
        if item.startswith("reader.real_path.structural_columns:")
    )
    applied_idx = next(
        i for i, item in enumerate(event_log)
        if item.startswith("reader.real_path.allowlist_applied:")
    )
    measurement_idx = next(
        i for i, item in enumerate(event_log)
        if item.startswith("reader.real_path.measurement_columns:")
    )
    assert structural_idx < applied_idx < measurement_idx
    assert "counterpart_ipv_error_current" not in event_log[structural_idx]
    assert "target_ipv_error_future" not in event_log[structural_idx]
    assert "counterpart_ipv_error_current" in event_log[measurement_idx]


def test_evil_allowlist_subclass_is_rejected_before_reader_load(tmp_path, monkeypatch):
    class EvilAllowlist(CaseAllowlist):
        pass

    evil = object.__new__(EvilAllowlist)
    object.__setattr__(evil, "source_path", tmp_path / "evil.csv")
    object.__setattr__(evil, "included_splits", ("development", "guard", "held_out"))
    object.__setattr__(evil, "allowed_case_ids", frozenset(("case_hold",)))
    object.__setattr__(evil, "split_counts", {"held_out": 1})
    object.__setattr__(evil, "source_sha256", "0" * 64)
    object.__setattr__(evil, "case_to_split", {"case_hold": "held_out"})

    schema = _schema()
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    bad_scope = object.__new__(AllowlistedArtifactScope)
    object.__setattr__(bad_scope, "artifact_id", spec.artifact_id)
    object.__setattr__(bad_scope, "join_column", "case_key")
    object.__setattr__(bad_scope, "allowlist", evil)
    object.__setattr__(bad_scope, "held_out_parsed_rows", 0)
    object.__setattr__(bad_scope, "unmapped_rows", 0)
    loaded = []

    def fail_if_loaded(*args, **kwargs):
        loaded.append((args, kwargs))
        raise AssertionError("real/source rows must not be loaded")

    monkeypatch.setattr("build_ledger._load_limited_rows", fail_if_loaded)

    with pytest.raises(ContractViolation, match="exact CaseAllowlist"):
        open_measurement_reader(spec, bad_scope, _permit(), source_rows=None)
    assert loaded == []


def test_pickled_forged_allowlist_mapping_is_rejected_at_reader_open(tmp_path):
    schema = _schema()
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    good = _allowlist(tmp_path)
    forged = object.__new__(CaseAllowlist)
    case_to_split = dict(good.case_to_split)
    case_to_split["case_hold"] = "development"
    object.__setattr__(forged, "source_path", good.source_path)
    object.__setattr__(forged, "included_splits", good.included_splits)
    object.__setattr__(
        forged,
        "allowed_case_ids",
        frozenset(
            case_id
            for case_id, split in case_to_split.items()
            if split in good.included_splits
        ),
    )
    object.__setattr__(forged, "split_counts", dict(good.split_counts))
    object.__setattr__(forged, "source_sha256", good.source_sha256)
    object.__setattr__(forged, "case_to_split", case_to_split)
    forged = pickle.loads(pickle.dumps(forged))
    scope = AllowlistedArtifactScope._from_schema(spec, forged, "case_key")

    with pytest.raises(ContractViolation, match="allowed_case_ids"):
        open_measurement_reader(spec, scope, _permit(), source_rows=[_feature_row()])


def test_allowlist_source_recheck_accepts_legitimate_token(tmp_path):
    schema = _schema()
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    allowlist = _allowlist(tmp_path)
    scope = resolve_artifact_scope(spec, allowlist)

    reader = open_measurement_reader(
        spec,
        scope,
        _permit(),
        source_rows=[_feature_row("case_dev"), _feature_row("case_hold")],
    )

    assert [row["case_key"] for row in reader.iter_measurement_rows()] == ["case_dev"]


def test_allowlist_source_replacement_after_reader_open_fails_closed(tmp_path):
    schema = _schema()
    spec = schema.artifacts_by_id["rq009_feature_matrix"]
    allowlist = _allowlist(tmp_path)
    scope = resolve_artifact_scope(spec, allowlist)
    reader = open_measurement_reader(
        spec,
        scope,
        _permit(),
        source_rows=[_feature_row("case_dev")],
    )
    allowlist.source_path.write_text(
        "case_id,split\n"
        "case_dev,held_out\n"
        "case_guard,guard\n"
        "case_hold,held_out\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="source_sha256 mismatch"):
        build_l1_for_artifact(spec, scope, reader)
