"""RQ015A T1 ledger-builder fixtures.

All tests use synthetic rows only; no production audit data is scanned.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Iterator, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

from build_ledger import (  # noqa: E402
    build_absent_artifact_coverage,
    build_l1_for_artifact,
    check_l1_conservation_counts,
    derive_aggregation_key,
    load_case_allowlist,
    load_execute_permit,
    load_ledger_schema_v2,
    open_measurement_reader,
    resolve_artifact_scope,
    sort_l1_rows,
    _make_test_permit_UNSAFE,
)
from rq015a_contracts import ATTEMPTED, NOT_ATTEMPTED, UNKNOWN, ContractViolation  # noqa: E402
from rq015a_types import (  # noqa: E402
    ARTIFACT_NOT_PRESENT_LOCALLY,
    L1LedgerRow,
    RQ007_SPLIT_NOT_APPLICABLE,
    SortedL1LedgerRows,
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
