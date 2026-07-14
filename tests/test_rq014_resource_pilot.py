from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.hpc.prepare_research_run as launcher
import scripts.rq014.run_managed_g2 as managed
from scripts.rq014.run_resource_pilot import (
    FAILURE_CODES,
    PilotError,
    _assemble_windows,
    _derive_state,
    canonical_json_bytes,
    run_resource_pilot,
    select_resource_pilot_cells,
    sha256_file,
)
from scripts.rq014.wod_ipv_preprocessing import derive_window_kinematics


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "run_specs" / "RQ014_g2_resource_pilot.template.json"
GOLDEN = ROOT / "tests" / "fixtures" / "rq014_resource_pilot_cells_v1.json"
LANE = ROOT / "reports" / "plans" / "RQ014_recovery_lane_v3.json"


def _resolved_template() -> dict[str, object]:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload["run_id"] = "RQ014_resource_pilot_fixture"
    payload["git_commit"] = "1" * 40
    payload["declassification_export_commit"] = "2" * 40
    for value in payload.values():
        if isinstance(value, dict) and set(value) == {"path", "sha256"}:
            value["path"] = "/managed/fixture"
            value["sha256"] = "3" * 64
    return payload


def _write_spec(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pilot_template_and_manual_schema_accept_exact_resolved_keys(tmp_path: Path) -> None:
    payload = _resolved_template()
    loaded = launcher.load_spec(_write_spec(tmp_path / "pilot.json", payload))
    assert loaded["resource_profile_id"] == "rq014-g2-resource-pilot-cpu-v1"
    assert loaded["pilot_scope"] == {
        "cell_selection_rule_id": "LANE_V3_NON_M3_COST_EXTREMES_V1",
        "non_m3_stages": ["source_load", "window_assembly", "feature_prep"],
        "m3_stage_enabled": False,
        "env_v4_required": True,
        "m3_cost_estimate": "EXPLICITLY_UNMEASURED",
    }
    schema = json.loads(
        (ROOT / "configs" / "run_specs" / "research_run_spec_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["$defs"]["pilotScope"]["additionalProperties"] is False
    assert schema["oneOf"][2]["properties"]["resource_profile_id"]["const"] == (
        "rq014-g2-resource-pilot-cpu-v1"
    )


@pytest.mark.parametrize(
    "field",
    [
        "contract_preflight_receipt",
        "contract_preflight_done",
        "wod_path_type_mapping_manifest",
        "pilot_scope",
    ],
)
def test_pilot_new_top_level_fields_are_required_exactly(tmp_path: Path, field: str) -> None:
    payload = _resolved_template()
    del payload[field]
    with pytest.raises(ValueError, match=f"missing.*{field}"):
        launcher.load_spec(_write_spec(tmp_path / f"missing-{field}.json", payload))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cell_selection_rule_id", "OTHER"),
        ("non_m3_stages", ["source_load"]),
        ("m3_stage_enabled", True),
        ("env_v4_required", False),
        ("m3_cost_estimate", 0),
    ],
)
def test_pilot_scope_rejects_value_drift_and_extra_keys(
    tmp_path: Path, field: str, replacement: object
) -> None:
    payload = _resolved_template()
    payload["pilot_scope"][field] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match="pilot_scope"):
        launcher.load_spec(_write_spec(tmp_path / f"bad-{field}.json", payload))
    payload = _resolved_template()
    payload["pilot_scope"]["extra"] = "forbidden"  # type: ignore[index]
    with pytest.raises(ValueError, match="pilot_scope"):
        launcher.load_spec(_write_spec(tmp_path / f"extra-{field}.json", payload))


def test_pilot_rejects_unexpected_top_level_field(tmp_path: Path) -> None:
    payload = _resolved_template()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        launcher.load_spec(_write_spec(tmp_path / "unexpected.json", payload))


def test_lane_v3_cell_selection_matches_golden_fixture() -> None:
    selected = select_resource_pilot_cells(json.loads(LANE.read_text(encoding="utf-8")))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert selected == {key: golden[key] for key in selected}


@pytest.mark.parametrize(
    "axis_name", ["sampling_axis", "temporal_axis", "horizon_axis", "readout_axis"]
)
@pytest.mark.parametrize("mutation", ["value", "order"])
def test_lane_v3_cell_selection_rejects_any_axis_value_or_order_drift(
    axis_name: str,
    mutation: str,
) -> None:
    lane = copy.deepcopy(json.loads(LANE.read_text(encoding="utf-8")))
    bank = lane["rating_blind_feature_bank"]
    axis = bank[axis_name]
    if axis_name == "temporal_axis":
        axis = axis["recipes"]
    if mutation == "order":
        axis[0], axis[1] = axis[1], axis[0]
    elif axis_name == "sampling_axis":
        axis[0]["rate_hz"] = 5
    elif axis_name == "temporal_axis":
        axis[0]["window_s"] = 1.25
    elif axis_name == "horizon_axis":
        axis[0]["maximum_tau_tick"] = "DRIFT"
    else:
        axis[0] = "OTHER"
    with pytest.raises(PilotError, match="ordered axis contract drifted"):
        select_resource_pilot_cells(lane)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.write_text(
        ",".join(header) + "\n" + "".join(",".join(map(str, row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_runtime_fixture(tmp_path: Path) -> dict[str, object]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    times_history = [round(-1.0 + index * 0.25, 2) for index in range(5)]
    times_future = [round(index * 0.25, 2) for index in range(1, 13)]
    _write_csv(
        bundle / "ego_history_states.csv",
        ["segment_id", "time_s", "pos_x_m", "pos_y_m"],
        [["S1", value, value, 0.0] for value in times_history],
    )
    _write_csv(
        bundle / "candidate_states.csv",
        [
            "segment_id",
            "candidate_id",
            "effective_time_s",
            "pos_x_m",
            "pos_y_m",
            "included_in_effective_future",
        ],
        [
            ["S1", candidate_id, value, value, 0.1 * value, "true"]
            for candidate_id in ("C1", "C2", "C3")
            for value in times_future
        ],
    )
    counterpart_times = [round(-1.0 + index * 0.25, 2) for index in range(17)]
    _write_csv(
        bundle / "counterpart_tracks.csv",
        ["segment_id", "time_s", "x_m", "y_m"],
        [["S1", value, value + 2.0, 1.0] for value in counterpart_times],
    )
    bundle_files = []
    for name in (
        "candidate_states.csv",
        "ego_history_states.csv",
        "counterpart_tracks.csv",
    ):
        path = bundle / name
        bundle_files.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "contains_rating": False,
            }
        )
    (bundle / "file_manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-score-stripped-file-manifest-v1",
                "files": bundle_files,
            }
        )
    )
    mapping = tmp_path / "wod_path_type_mapping.csv"
    _write_csv(
        mapping,
        ["segment_id", "tstar_context_step", "path_type"],
        [["S1", 10, "CP"]],
    )
    mapping_manifest = tmp_path / "mapping_manifest.json"
    mapping_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-wod-path-type-mapping-manifest-v1",
                "row_count": 1,
                "contains_rating": False,
                "allowed_values": ["CP", "HO", "MP", "F"],
                "key_columns": ["segment_id", "tstar_context_step"],
                "value_column": "path_type",
                "mapping": {
                    "format": "RFC4180_CSV",
                    "path": str(mapping),
                    "size_bytes": mapping.stat().st_size,
                    "sha256": sha256_file(mapping),
                },
            }
        )
    )
    inputs = {}
    for name in ("input_manifest", "sanitization", "ledger", "export_receipt", "export_done"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(canonical_json_bytes({"fixture": name}))
        inputs[name] = path
    m3 = tmp_path / "m3_scorer.joblib"
    m3.write_bytes(b"verification-only-m3")
    m3_sha = sha256_file(m3)
    preflight_receipt = tmp_path / "preflight_receipt.json"
    preflight_receipt.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-g2-contract-preflight-receipt-v1",
                "operation": "rq014_g2_contract_preflight",
                "status": "PASS",
                "input_manifest_sha256": sha256_file(inputs["input_manifest"]),
                "materialization_ledger_sha256": sha256_file(inputs["ledger"]),
                "declassification_export_receipt_sha256": sha256_file(inputs["export_receipt"]),
                "declassification_export_done_sha256": sha256_file(inputs["export_done"]),
                "wod_path_type_mapping": {
                    "manifest_sha256": sha256_file(mapping_manifest)
                },
                "m3_artifact_input_receipt": {
                    "size_bytes": m3.stat().st_size,
                    "sha256": m3_sha,
                    "deserialized": False,
                },
            }
        )
    )
    preflight_done = tmp_path / "preflight_DONE.json"
    preflight_done.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "rq014-managed-operation-done-v1",
                "operation": "rq014_g2_contract_preflight",
                "receipt_sha256": sha256_file(preflight_receipt),
                "status": "PASS",
            }
        )
    )
    return {
        "run_id": "fixture",
        "lane_path": LANE,
        "bundle_root": bundle,
        "input_manifest_path": inputs["input_manifest"],
        "sanitization_receipt_path": inputs["sanitization"],
        "materialization_ledger_path": inputs["ledger"],
        "mapping_manifest_path": mapping_manifest,
        "m3_artifact_path": m3,
        "m3_artifact_size_bytes": m3.stat().st_size,
        "m3_artifact_sha256": m3_sha,
        "export_receipt_path": inputs["export_receipt"],
        "export_done_path": inputs["export_done"],
        "preflight_receipt_path": preflight_receipt,
        "preflight_done_path": preflight_done,
    }


def _anchor_sources(
    *,
    candidate_end_s: tuple[float, float, float] = (3.0, 3.0, 3.0),
    counterpart_end_s: float = 3.0,
) -> dict[str, object]:
    history = [
        (tick / 4.0, tick / 4.0, 0.0)
        for tick in range(-4, 1)
    ]
    candidates = {
        ("S1", candidate_id): [
            (tick / 4.0, tick / 4.0, 0.1 * tick / 4.0)
            for tick in range(1, round(end_s * 4) + 1)
        ]
        for candidate_id, end_s in zip(
            ("C1", "C2", "C3"), candidate_end_s
        )
    }
    counterpart = [
        (tick / 4.0, tick / 4.0 + 2.0, 1.0)
        for tick in range(-4, round(counterpart_end_s * 4) + 1)
    ]
    return {
        "candidate": candidates,
        "history": {("S1",): history},
        "counterpart": {("S1",): counterpart},
        "mapped_segment_count": 1,
        "source_row_count": sum(len(rows) for rows in candidates.values())
        + len(history)
        + len(counterpart),
    }


def test_anchor_domain_uses_joint_three_candidate_h_common() -> None:
    sources = _anchor_sources(candidate_end_s=(3.0, 2.5, 2.0))
    windows = _assemble_windows(sources, "RR3-R10L-TF-HFEAS-NEX_MEAN")
    assert len(windows) == 3 * 11
    assert all(window[-1][0] == 2.0 for pair in windows for window in pair)


def test_anchor_domain_h20_is_all_or_none_and_gates_hfeas() -> None:
    assert len(
        _assemble_windows(_anchor_sources(), "RR3-R04N-CH-W10-H20-NEX_MEAN")
    ) == 3 * 5
    sources = _anchor_sources(candidate_end_s=(3.0, 3.0, 1.75))
    with pytest.raises(PilotError, match="No complete rating-blind windows"):
        _assemble_windows(sources, "RR3-R04N-CH-W10-H20-NEX_MEAN")
    with pytest.raises(PilotError, match="No complete rating-blind windows"):
        _assemble_windows(sources, "RR3-R10L-TF-HFEAS-NEX_MEAN")


@pytest.mark.parametrize("stream", ["C1", "C2", "C3", "counterpart"])
def test_anchor_domain_requires_every_exact_four_way_grid_tick(stream: str) -> None:
    sources = _anchor_sources()
    key = ("S1", stream)
    if stream == "counterpart":
        rows = sources["counterpart"][("S1",)]
    else:
        rows = sources["candidate"][key]
    rows[:] = [row for row in rows if row[0] != 1.5]
    with pytest.raises(PilotError, match="No complete rating-blind windows"):
        _assemble_windows(sources, "RR3-R04N-CH-W10-H20-NEX_MEAN")


def test_stdlib_window_kinematics_matches_frozen_numpy_implementation() -> None:
    points = [
        (0.0, 0.0, 0.0),
        (0.25, 0.2, 0.0),
        (0.5, 0.5, 0.1),
        (0.75, 0.9, 0.3),
    ]
    observed = np.asarray(_derive_state(points))
    expected = derive_window_kinematics(
        np.asarray([(point[1], point[2]) for point in points]), 0.25
    )
    np.testing.assert_allclose(observed[:, 1:3], expected["position"], rtol=0, atol=0)
    np.testing.assert_allclose(observed[:, 3:5], expected["velocity"], rtol=0, atol=0)
    np.testing.assert_allclose(observed[:, 5:7], expected["acceleration"], rtol=0, atol=0)
    np.testing.assert_allclose(observed[:, 7], expected["heading"], rtol=0, atol=0)


def test_heading_stationary_propagation_threshold_and_all_stationary_rejection() -> None:
    states = _derive_state(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (3.0, 1.0, 0.0)]
    )
    assert [state[7] for state in states] == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(PilotError, match="All-stationary"):
        _derive_state([(0.0, 0.0, 0.0), (1.0, 1e-9, 0.0)])
    with pytest.raises(PilotError, match="All-stationary"):
        _derive_state([(0.0, 2.0, 3.0), (1.0, 2.0, 3.0), (2.0, 2.0, 3.0)])

    sources = _anchor_sources()
    sources["counterpart"][("S1",)] = [
        (point[0], 2.0, 3.0) for point in sources["counterpart"][("S1",)]
    ]
    with pytest.raises(PilotError, match="No complete rating-blind windows"):
        _assemble_windows(sources, "RR3-R04N-CH-W10-H20-NEX_MEAN")


def test_heading_exact_negative_pi_normalizes_to_positive_pi() -> None:
    states = _derive_state(
        [(0.0, 0.0, 0.0), (1.0, -1.0, -0.0), (2.0, -2.0, -0.0)]
    )
    assert [state[7] for state in states] == [math.pi, math.pi, math.pi]


def test_resource_pilot_receipt_measures_only_non_m3_stages(tmp_path: Path) -> None:
    receipt = run_resource_pilot(**_build_runtime_fixture(tmp_path))
    assert receipt["status"] == "PASS"
    assert receipt["pilot_scope"] == {
        "non_m3_stages": ["source_load", "window_assembly", "feature_prep"],
        "m3_stage_enabled": False,
        "env_v4_required": True,
        "m3_cost_estimate": "EXPLICITLY_UNMEASURED",
    }
    assert len(receipt["measurements"]) == 5
    assert {row["stage_id"] for row in receipt["measurements"]} == {
        "source_load",
        "window_assembly",
        "feature_prep",
    }
    assert all(row["status"] == "PASS" and row["failure_code"] == "NONE" for row in receipt["measurements"])
    source_rows = [
        row for row in receipt["measurements"] if row["stage_id"] == "source_load"
    ]
    assert len(source_rows) == 1
    assert source_rows[0]["cell_id"] == "SHARED"
    assert set(receipt["failure_taxonomy"]) == set(FAILURE_CODES) - {"NONE"}
    assert receipt["failure_rate"] == 0.0
    assert receipt["projection"]["m3_cost_estimate"] == "EXPLICITLY_UNMEASURED"
    assert receipt["projection"]["combined_g2r_cost_estimate"] == "EXPLICITLY_UNMEASURED"
    assert receipt["projection"]["projected_non_m3_cpu_hours"] >= 0.0
    maximum_cell_wall = max(
        timing["total_serial_walltime_seconds"]
        for timing in receipt["per_cell_serial_timings"].values()
    )
    maximum_cell_cpu = max(
        timing["total_serial_cpu_seconds"]
        for timing in receipt["per_cell_serial_timings"].values()
    )
    assert receipt["projection"]["formula"] == (
        "shared_source_load_once + 320 * "
        "max_endpoint_window_assembly_plus_feature_prep"
    )
    assert math.isclose(
        receipt["projection"]["projected_non_m3_serial_walltime_hours"],
        (source_rows[0]["walltime_seconds"] + 320 * maximum_cell_wall) / 3600.0,
    )
    assert math.isclose(
        receipt["projection"]["projected_non_m3_cpu_hours"],
        (source_rows[0]["cpu_seconds"] + 320 * maximum_cell_cpu) / 3600.0,
    )
    assert math.isclose(
        receipt["projection"]["projected_non_m3_parallel_walltime_hours"],
        (source_rows[0]["walltime_seconds"] + 20 * maximum_cell_wall) / 3600.0,
    )
    assert receipt["projection"]["projected_non_m3_parallel_walltime_hours"] <= (
        receipt["projection"]["projected_non_m3_serial_walltime_hours"]
    )
    assert set(receipt["per_cell_serial_timings"]) == {
        "RR3-R04N-CH-W10-H20-NEX_MEAN",
        "RR3-R10L-TF-HFEAS-NEX_MEAN",
    }
    for timing in receipt["per_cell_serial_timings"].values():
        assert set(timing) == {
            "stages",
            "measured_stage_count",
            "total_serial_walltime_seconds",
            "total_serial_cpu_seconds",
        }
        assert set(timing["stages"]) == {
            "window_assembly",
            "feature_prep",
        }
        assert all(
            set(stage_timing) == {"walltime_seconds", "cpu_seconds"}
            for stage_timing in timing["stages"].values()
        )
        assert timing["measured_stage_count"] == 2
        assert timing["total_serial_walltime_seconds"] >= 0.0
        assert timing["total_serial_cpu_seconds"] >= 0.0
    assert receipt["parallel_execution"] == {
        "configured_max_workers": 16,
        "actual_worker_count": 2,
        "selected_cell_count": 2,
        "worker_model": "PROCESS_POOL",
        "worker_thread_limits": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
        "shared_source_load_walltime_seconds": receipt["parallel_execution"][
            "shared_source_load_walltime_seconds"
        ],
        "worker_pool_walltime_seconds": receipt["parallel_execution"][
            "worker_pool_walltime_seconds"
        ],
        "aggregate_walltime_seconds": receipt["parallel_execution"][
            "aggregate_walltime_seconds"
        ],
    }
    assert receipt["parallel_execution"]["shared_source_load_walltime_seconds"] == (
        source_rows[0]["walltime_seconds"]
    )
    assert receipt["parallel_execution"]["worker_pool_walltime_seconds"] >= 0.0
    assert receipt["parallel_execution"]["aggregate_walltime_seconds"] >= 0.0


def test_resource_pilot_records_input_failure_without_done_eligible_status(tmp_path: Path) -> None:
    fixture = _build_runtime_fixture(tmp_path)
    Path(fixture["preflight_done_path"]).write_bytes(b"{}\n")
    receipt = run_resource_pilot(**fixture)
    assert receipt["status"] == "FAIL"
    assert receipt["failure_taxonomy"]["INPUT_CONTRACT_FAILURE"] == 1
    assert receipt["failure_rate"] == 1.0
    assert receipt["projection"]["projected_non_m3_cpu_hours"] == (
        "EXPLICITLY_UNMEASURED"
    )
    assert receipt["projection"]["projected_non_m3_parallel_walltime_hours"] == (
        "EXPLICITLY_UNMEASURED"
    )
    assert receipt["per_cell_serial_timings"] == {}
    assert receipt["parallel_execution"]["actual_worker_count"] == 0
    assert receipt["parallel_execution"]["aggregate_walltime_seconds"] == (
        "EXPLICITLY_UNMEASURED"
    )


def test_managed_resource_pilot_writes_hash_chained_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outputs"
    monkeypatch.setattr(
        managed,
        "run_resource_pilot",
        lambda **kwargs: {
            "schema_version": "rq014-g2-resource-pilot-receipt-v1",
            "operation": "rq014_g2_resource_pilot",
            "status": "PASS",
        },
    )
    fixture = tmp_path / "fixture"
    fixture.write_text("x", encoding="utf-8")
    arguments = [
        "run_managed_g2.py",
        "resource-pilot",
        "--run-id",
        "fixture",
        "--lane-v3",
        str(fixture),
        "--bundle-root",
        str(tmp_path),
        "--wod-path-type-mapping-manifest",
        str(fixture),
        "--contract-preflight-receipt",
        str(fixture),
        "--contract-preflight-done",
        str(fixture),
        "--m3-artifact",
        str(fixture),
        "--m3-artifact-size-bytes",
        "1",
        "--m3-artifact-sha256",
        "a" * 64,
        "--input-manifest",
        str(fixture),
        "--sanitization-receipt",
        str(fixture),
        "--materialization-ledger",
        str(fixture),
        "--declassification-export-receipt",
        str(fixture),
        "--declassification-export-done",
        str(fixture),
        "--output-root",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert managed.main() == 0
    receipt_path = output / "rq014_g2_resource_pilot_receipt.json"
    done = json.loads((output / "DONE.json").read_text(encoding="utf-8"))
    assert set(done) == {"schema_version", "operation", "receipt_sha256", "status"}
    assert done["operation"] == "rq014_g2_resource_pilot"
    assert done["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def test_pilot_validate_only_table_and_sbatch_wiring(tmp_path: Path) -> None:
    base = tmp_path / "managed"
    run_root = base / "work_dirs" / "RQ014" / "pilot"
    common = {
        "commit": "a" * 40,
        "code_snapshot_files": {},
        "run_id": "pilot",
        "job_name": "zxc-rq014-pilot-123456789abc",
        "resource_profile_id": "rq014-g2-resource-pilot-cpu-v1",
        "slurm_profile": {
            "partition": "amd",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 16,
            "memory": "32G",
            "time": "04:00:00",
        },
        "thread_limits": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
        "pilot_scope": {
            "cell_selection_rule_id": "LANE_V3_NON_M3_COST_EXTREMES_V1",
            "non_m3_stages": ["source_load", "window_assembly", "feature_prep"],
            "m3_stage_enabled": False,
            "env_v4_required": True,
            "m3_cost_estimate": "EXPLICITLY_UNMEASURED",
        },
        "environment_manifest_path": str(base / "environment.json"),
        "environment_manifest_sha256": "1" * 64,
        "python_executable_path": str(base / "python3.9"),
        "python_executable_sha256": "2" * 64,
        "python_version": "3.9.24",
        "isolated_sys_path": ["/python39.zip", "/python3.9", "/lib-dynload"],
        "stdlib_checksum_manifest_sha256": "3" * 64,
        "native_library_manifest_sha256": "4" * 64,
        "m3_artifact_verification": {"deserialized": False},
    }
    planned = launcher._with_rq014_validate_only_plan(common)
    assert planned["submission_plan"]["pilot_stage_plan"] == common["pilot_scope"]
    assert planned["runtime_metadata"]["m3_execution"] == {
        "enabled": False,
        "env_v4_required": True,
        "cost_estimate": "EXPLICITLY_UNMEASURED",
    }

    validated = {
        **common,
        "fixed_subcommand": "resource-pilot",
        "entrypoint": "scripts/rq014/run_managed_g2.py",
        "commit": "a" * 40,
        "declassification_export_commit": "b" * 40,
        "run_spec_sha256": "5" * 64,
        "authorization_sha256": "6" * 64,
        "execution_contract_sha256": "7" * 64,
        "formal_g1_relative_path": "formal.json",
        "formal_g1_sha256": "8" * 64,
        "stdlib_root": str(base / "stdlib"),
        "lib_dynload_root": str(base / "stdlib" / "lib-dynload"),
        "python_zip_path": str(base / "python39.zip"),
        "stdlib_regular_file_count": 10,
        "stdlib_regular_file_total_size_bytes": 1000,
        "stdlib_checksum_manifest_path": str(base / "stdlib.sha256"),
        "native_library_manifest_path": str(base / "native.tsv"),
        "native_library_row_count": 20,
        "native_library_total_size_bytes": 14656296,
        "native_library_symlink_row_count": 16,
        "m3_artifact_path": str(base / "m3.joblib"),
        "m3_artifact_size_bytes": 88306301,
        "m3_artifact_sha256": "9" * 64,
        "input_manifest_path": str(base / "input.json"),
        "input_manifest_sha256": "a" * 64,
        "sanitization_receipt_path": str(base / "sanitization.json"),
        "sanitization_receipt_sha256": "b" * 64,
        "materialization_ledger_path": str(base / "ledger.json"),
        "materialization_ledger_sha256": "c" * 64,
        "wod_path_type_mapping_manifest_path": str(base / "mapping.json"),
        "wod_path_type_mapping_manifest_sha256": "d" * 64,
        "declassification_export_receipt_path": str(base / "export-receipt.json"),
        "declassification_export_receipt_sha256": "e" * 64,
        "declassification_export_done_path": str(base / "export-DONE.json"),
        "declassification_export_done_sha256": "f" * 64,
        "contract_preflight_receipt_path": str(base / "preflight-receipt.json"),
        "contract_preflight_receipt_sha256": "0" * 64,
        "contract_preflight_done_path": str(base / "preflight-DONE.json"),
        "contract_preflight_done_sha256": "1" * 64,
        "score_stripped_bundle_root": str(base / "bundle"),
        "contract_bundle_relative_path": "reports/plans/bundle.sha256",
        "contract_bundle_sha256": "2" * 64,
        "code_snapshot_receipt_sha256": "3" * 64,
    }
    script = launcher.render_rq014_sbatch(
        validated=validated,
        base=base,
        repo=ROOT,
        run_root=run_root,
        code=run_root / "code",
        sealed_spec_path=run_root / "manifests" / "run_spec.json",
    )
    assert "#SBATCH --job-name=zxc-rq014-pilot-123456789abc" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "resource-pilot" in script
    assert "--contract-preflight-receipt" in script
    assert "--wod-path-type-mapping-manifest" in script
    assert "run_resource_pilot.py" in script
    assert "m3.joblib" in script
