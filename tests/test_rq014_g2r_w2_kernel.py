from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.rq014 import build_wod_m3_anchors as M3
from scripts.rq014 import build_wod_scene_anchor_domain as DOMAIN


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "rq014_g2r_v1"
W2B_FIXTURES = ROOT / "tests" / "fixtures" / "rq014_g2r_w2b"
NC_IDS = (
    "NC_HISTORY_BRANCH_R04N_W10",
    "NC_HISTORY_BRANCH_R04N_W25",
    "NC_HISTORY_BRANCH_R10L_W10",
    "NC_HISTORY_BRANCH_R10L_W25",
    "NC_HISTORY_FUTURE_PERTURBATION",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nc_input(fixture_id: str) -> dict:
    return _load(FIXTURES / f"{fixture_id.lower()}_input.json")


def _nc_expected(fixture_id: str) -> dict:
    return _load(FIXTURES / f"{fixture_id.lower()}_expected.json")


def _assert_nc_receipt_schema(receipt: dict) -> None:
    assert set(receipt) == {
        "schema_version", "control_id", "status", "implementation_path",
        "implementation_size_bytes", "implementation_sha256", "estimator_sha256",
        "python_executable_sha256", "environment_manifest_sha256", "fixtures",
        "failure_or_NA", "created_at_utc",
    }
    assert receipt["schema_version"] == "rq014-nc-pretstar-history-only-receipt-v1"
    assert receipt["control_id"] == "NC_PRETSTAR_HISTORY_ONLY"
    assert [item["fixture_id"] for item in receipt["fixtures"]] == list(NC_IDS)
    for item in receipt["fixtures"]:
        assert set(item) == {
            "fixture_id", "input_path", "input_size_bytes", "input_sha256",
            "expected_path", "expected_size_bytes", "expected_sha256",
            "observed_state_bytes_sha256", "observed_m3_context_bytes_sha256",
            "observed_focal_reference_bytes_sha256",
            "observed_counterpart_reference_bytes_sha256", "observed_ipv_bytes_sha256",
            "observed_payload_sha256", "status",
        }


def test_w2_feature_kernel_reproduces_w1_32_column_golden_bytes_and_hash() -> None:
    fixture = _load(FIXTURES / "wod_m3_feature_construction_golden.json")
    expected_path = FIXTURES / "m3_input_row_expected.json"
    expected = _load(expected_path)
    observed = M3.build_m3_input_row_from_history(
        fixture["history_rows"],
        fixture["categories"],
        case_start_timestamp_s=fixture["case_start_timestamp_s"],
    )
    observed_bytes, observed_sha256 = M3.m3_input_row_bytes_and_sha256(observed)
    assert tuple(observed["columns"]) == M3.M3_INPUT_COLUMNS
    assert len(observed["values"]) == 32
    assert observed == expected
    assert observed_bytes == expected_path.read_bytes()
    assert observed_sha256 == "9ecc870d825b39239be06ea142dca1691b9834f882cf3ccd662f87eafde33762"
    assert M3.build_m3_input_row_from_history(
        fixture["history_rows"],
        fixture["categories"],
        case_start_timestamp_s=fixture["case_start_timestamp_s"],
    ) == observed


def test_w2_all_eight_temporal_bounds_and_explicit_context_alignment_are_frozen() -> None:
    expected_at_rate_4 = {
        "CH-W10": (4, 8),
        "CH-W25": (-2, 8),
        "LF-W10": (8, 12),
        "LF-W25": (8, 18),
        "HF-W10": (4, 12),
        "HF-W25": (-2, 18),
        "TP": (0, 8),
        "TF": (0, 20),
    }
    assert M3.TEMPORAL_FAMILY_IDS == tuple(expected_at_rate_4)
    assert {
        temporal_id: M3.temporal_window_bounds(temporal_id, 8, 4, 20)
        for temporal_id in M3.TEMPORAL_FAMILY_IDS
    } == expected_at_rate_4
    assert {
        temporal_id: M3.temporal_window_bounds(temporal_id, 20, 10, 50)
        for temporal_id in M3.TEMPORAL_FAMILY_IDS
    } == {
        "CH-W10": (10, 20),
        "CH-W25": (-5, 20),
        "LF-W10": (20, 30),
        "LF-W25": (20, 45),
        "HF-W10": (10, 30),
        "HF-W25": (-5, 45),
        "TP": (0, 20),
        "TF": (0, 50),
    }
    positions = {tick: (float(tick), 0.0) for tick in range(-10, 51)}
    sensitivity_contexts = {
        temporal_id: M3.temporal_window_bounds(temporal_id, 8, 4, 20)[1] - 6
        for temporal_id in M3.TEMPORAL_FAMILY_IDS
    }
    for temporal_id in M3.TEMPORAL_FAMILY_IDS:
        assert M3.resolve_m3_context_tick(
            alignment=M3.ALIGNMENT_PRIMARY,
            tau_tick=8,
            temporal_id=temporal_id,
            rate_hz=4,
            h_common_tick=20,
        ) == 8
        ego, counterpart, context_tick = M3.select_feature_family_context(
            positions,
            positions,
            temporal_id=temporal_id,
            tau_tick=8,
            rate_hz=4,
            h_common_tick=20,
            case_start_tick=-10,
            alignment=M3.ALIGNMENT_PRIMARY,
        )
        assert context_tick == 8
        assert len(ego) == len(counterpart) == 19
        assert M3.resolve_m3_context_tick(
            alignment=M3.ALIGNMENT_SENSITIVITY,
            tau_tick=8,
            temporal_id=temporal_id,
            rate_hz=4,
            h_common_tick=20,
        ) == sensitivity_contexts[temporal_id]
        sensitivity_ego, sensitivity_counterpart, sensitivity_tick = (
            M3.select_feature_family_context(
                positions,
                positions,
                temporal_id=temporal_id,
                tau_tick=8,
                rate_hz=4,
                h_common_tick=20,
                case_start_tick=-10,
                alignment=M3.ALIGNMENT_SENSITIVITY,
            )
        )
        assert sensitivity_tick == sensitivity_contexts[temporal_id]
        assert len(sensitivity_ego) == len(sensitivity_counterpart) == (
            sensitivity_tick - (-10) + 1
        )
    assert M3.resolve_m3_context_tick(
        alignment=M3.ALIGNMENT_SENSITIVITY,
        tau_tick=4,
        temporal_id="CH-W10",
        rate_hz=4,
        h_common_tick=20,
    ) == -2
    _, _, negative_context_tick = M3.select_feature_family_context(
        positions,
        positions,
        temporal_id="CH-W10",
        tau_tick=4,
        rate_hz=4,
        h_common_tick=20,
        case_start_tick=-10,
        alignment=M3.ALIGNMENT_SENSITIVITY,
    )
    assert negative_context_tick == -2
    with pytest.raises(M3.WodM3KernelError, match="precedes the explicit case-start"):
        M3.select_feature_family_context(
            positions,
            positions,
            temporal_id="CH-W10",
            tau_tick=4,
            rate_hz=4,
            h_common_tick=20,
            case_start_tick=-1,
            alignment=M3.ALIGNMENT_SENSITIVITY,
        )


def test_w2_raw_position_port_is_deterministic_and_uses_kinematic_hv_hv_tokens() -> None:
    times = np.arange(14, dtype=float) * 0.25
    ego = np.column_stack([times, np.zeros(len(times))])
    counterpart = np.column_stack([10.0 - times, np.full(len(times), 0.5)])
    focal_reference = M3.build_scene_focal_reference(ego, route_intent="GO_STRAIGHT")
    first = M3.build_wod_m3_input_row(
        ego,
        counterpart,
        sample_dt_s=0.25,
        case_start_timestamp_s=0.0,
        context_end_timestamp_s=2.5,
        scene_focal_reference=focal_reference,
        counterpart_is_vehicle=True,
    )
    second = M3.build_wod_m3_input_row(
        ego,
        counterpart,
        sample_dt_s=0.25,
        case_start_timestamp_s=0.0,
        context_end_timestamp_s=2.5,
        scene_focal_reference=focal_reference,
        counterpart_is_vehicle=True,
    )
    assert M3.m3_input_row_bytes_and_sha256(first) == M3.m3_input_row_bytes_and_sha256(second)
    by_name = dict(zip(first["columns"], first["values"]))
    assert by_name["agent_type_pair"] == "HV;HV"
    assert by_name["vehicle_type_list"] == "['HV', 'HV']"
    assert by_name["av_included"] == "all_HV"
    assert by_name["history_row_count"] == 10
    assert all(
        np.isfinite(float(by_name[name]))
        for name in (
            "counterpart_ipv_current",
            "counterpart_ipv_error_current",
            "counterpart_ipv_slope_pre_anchor",
        )
    )


def test_w2_scene_focal_reference_is_one_shared_byte_identity_for_c1_c2_c3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(14, dtype=float) * 0.25
    shared_history = np.column_stack([times, np.zeros(len(times))])
    focal_reference = M3.build_scene_focal_reference(
        shared_history, route_intent="GO_STRAIGHT"
    )
    observed_reference_bytes: list[bytes] = []

    def capture_reference(focal, counterpart, **kwargs):
        observed_reference_bytes.append(np.asarray(focal.reference, dtype=float).tobytes())
        row_count = len(np.asarray(focal.data))
        return np.full((row_count, 2), 0.25), np.full((row_count, 2), 0.125)

    monkeypatch.setattr(M3, "estimate_ipv_pair", capture_reference)
    for offset in (0.0, 1.0, 2.0):
        candidate = np.column_stack([times, np.full(len(times), offset)])
        counterpart = np.column_stack([10.0 - times, np.full(len(times), 0.5)])
        M3.build_wod_m3_input_row(
            candidate,
            counterpart,
            sample_dt_s=0.25,
            case_start_timestamp_s=0.0,
            context_end_timestamp_s=2.5,
            scene_focal_reference=focal_reference,
            counterpart_is_vehicle=True,
        )
    assert len(observed_reference_bytes) == 30
    assert set(observed_reference_bytes) == {np.asarray(focal_reference).tobytes()}
    with pytest.raises(TypeError):
        M3.build_wod_m3_input_row(
            shared_history,
            shared_history,
            sample_dt_s=0.25,
            case_start_timestamp_s=0.0,
            context_end_timestamp_s=2.5,
            counterpart_is_vehicle=True,
        )


@pytest.mark.parametrize("bad_index", [4, 8, 13])
def test_w2_nonfinite_ipv_tail10_fails_closed_without_row_deletion(
    bad_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = np.arange(14, dtype=float) * 0.25
    ego = np.column_stack([times, np.zeros(len(times))])
    counterpart = np.column_stack([10.0 - times, np.full(len(times), 0.5)])
    focal_reference = M3.build_scene_focal_reference(ego, route_intent="GO_STRAIGHT")
    ipv = np.full(len(times), 0.25)
    error = np.full(len(times), 0.125)
    ipv[bad_index] = np.nan if bad_index != 8 else np.inf
    monkeypatch.setattr(
        M3,
        "_estimate_counterpart_ipv_series",
        lambda *args, **kwargs: (ipv.copy(), error.copy()),
    )
    assembler_called = False

    def forbidden_assembler(*args, **kwargs):
        nonlocal assembler_called
        assembler_called = True
        raise AssertionError("assembler must not receive a shortened tail")

    monkeypatch.setattr(M3, "build_m3_anchor_features", forbidden_assembler)
    with pytest.raises(M3.M3ScoringNumericalFailure) as exc_info:
        M3.build_wod_m3_input_row(
            ego,
            counterpart,
            sample_dt_s=0.25,
            case_start_timestamp_s=0.0,
            context_end_timestamp_s=2.5,
            scene_focal_reference=focal_reference,
            counterpart_is_vehicle=True,
        )
    assert exc_info.value.status == "M3_SCORING_NUMERICAL_FAILURE"
    assert exc_info.value.reason_code == "F_M3_SCORING_NUMERICAL_FAILURE"
    assert not assembler_called


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("geometry_path_category", "HO"),
        ("geometry_path_category", None),
        ("geometry_path_relation", "ARBITRARY"),
        ("turn_pair_label", "U-S"),
        ("agent_type_pair", "AV;HV"),
        ("vehicle_type_list", "['AV', 'HV']"),
        ("av_included", "mixed"),
        ("priority_role", "unknown"),
    ],
)
def test_w2_categories_reject_every_noncontract_token(field: str, value: object) -> None:
    fixture = _load(FIXTURES / "wod_m3_feature_construction_golden.json")
    categories = dict(fixture["categories"])
    categories[field] = value
    with pytest.raises(M3.WodM3KernelError, match="invalid frozen category token"):
        M3.build_m3_input_row_from_history(
            fixture["history_rows"],
            categories,
            case_start_timestamp_s=fixture["case_start_timestamp_s"],
        )
    categories = dict(fixture["categories"])
    categories["extra"] = "F"
    with pytest.raises(M3.WodM3KernelError, match="exact seven frozen keys"):
        M3.build_m3_input_row_from_history(
            fixture["history_rows"],
            categories,
            case_start_timestamp_s=fixture["case_start_timestamp_s"],
        )


def test_w2_nc_kernel_reproduces_all_five_w1_pairs_and_future_invariance(
    tmp_path: Path,
) -> None:
    assert tuple(inspect.signature(M3.build_nc_history_only_payload).parameters) == (
        "fixture_id",
        "sample_dt_s",
        "ego_history_xy",
        "counterpart_history_xy",
        "route_intent",
    )
    forbidden = {
        "candidate_id", "candidate_geometry", "candidate_future", "ego_future",
        "H_common", "M3_result", "envelope_result", "rating", "rating_derived_input",
    }
    assert forbidden.isdisjoint(inspect.signature(M3.build_nc_history_only_payload).parameters)
    environment = tmp_path / "environment.json"
    environment.write_bytes(b"{}\n")
    python = Path(sys.executable).resolve(strict=True)
    receipt = M3.run_nc_pretstar_history_only_gate(
        repo_root=ROOT,
        python_executable_path=python,
        python_executable_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        environment_manifest_path=environment,
        environment_manifest_sha256=hashlib.sha256(environment.read_bytes()).hexdigest(),
        created_at_utc="2026-07-16T00:00:00Z",
    )
    _assert_nc_receipt_schema(receipt)
    assert receipt["status"] == "PASS" and receipt["failure_or_NA"] == "NA"
    assert [item["fixture_id"] for item in receipt["fixtures"]] == list(NC_IDS)
    for fixture_id, observation in zip(NC_IDS, receipt["fixtures"]):
        expected_path = FIXTURES / f"{fixture_id.lower()}_expected.json"
        expected = _nc_expected(fixture_id)
        assert observation["status"] == "PASS"
        assert observation["observed_m3_context_bytes_sha256"] == expected[
            "m3_context_bytes_sha256"
        ]
        assert observation["observed_payload_sha256"] == next(
            iter(expected["candidate_payload_sha256_by_ordinal"].values())
        )

    future_input = _nc_input("NC_HISTORY_FUTURE_PERTURBATION")
    baseline = M3.build_nc_history_only_payload(
        fixture_id=future_input["fixture_id"],
        sample_dt_s=future_input["sample_dt_s"],
        ego_history_xy=future_input["ego_history_xy"],
        counterpart_history_xy=future_input["counterpart_history_xy"],
        route_intent=future_input["route_intent"],
    )
    perturbed = copy.deepcopy(future_input)
    perturbed["base_candidate_futures"] = perturbed["perturbed_candidate_futures"]
    perturbed["base_ego_future"] = perturbed["perturbed_ego_future"]
    repeated = M3.build_nc_history_only_payload(
        fixture_id=perturbed["fixture_id"],
        sample_dt_s=perturbed["sample_dt_s"],
        ego_history_xy=perturbed["ego_history_xy"],
        counterpart_history_xy=perturbed["counterpart_history_xy"],
        route_intent=perturbed["route_intent"],
    )
    assert M3.canonical_json_bytes(repeated) == M3.canonical_json_bytes(baseline)
    at_tau = copy.deepcopy(future_input)
    at_tau["ego_history_xy"][-1][0] = (
        float.fromhex(at_tau["ego_history_xy"][-1][0]) + 0.125
    ).hex()
    changed = M3.build_nc_history_only_payload(
        fixture_id=at_tau["fixture_id"],
        sample_dt_s=at_tau["sample_dt_s"],
        ego_history_xy=at_tau["ego_history_xy"],
        counterpart_history_xy=at_tau["counterpart_history_xy"],
        route_intent=at_tau["route_intent"],
    )
    assert changed["m3_context_bytes_sha256"] != baseline["m3_context_bytes_sha256"]
    assert changed["candidate_payload_sha256_by_ordinal"] != baseline[
        "candidate_payload_sha256_by_ordinal"
    ]

    copied_root = tmp_path / "mutated-repo"
    for relative in (
        "reports/plans/RQ014_g2r_output_contract_v1.json",
        "src/sociality_estimation/core/ipv_estimation.py",
        "scripts/rq014/build_wod_m3_anchors.py",
    ):
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copytree(FIXTURES, copied_root / FIXTURES.relative_to(ROOT))
    mutated_input_path = (
        copied_root / FIXTURES.relative_to(ROOT) / "nc_history_future_perturbation_input.json"
    )
    mutated_expected_path = (
        copied_root / FIXTURES.relative_to(ROOT) / "nc_history_future_perturbation_expected.json"
    )
    mutated_input = _load(mutated_input_path)
    mutated_input["ego_history_xy"][-1][0] = (
        float.fromhex(mutated_input["ego_history_xy"][-1][0]) + 0.125
    ).hex()
    mutated_input_path.write_bytes(M3.canonical_json_bytes(mutated_input))
    mutated_expected = M3.build_nc_history_only_payload(
        fixture_id=mutated_input["fixture_id"],
        sample_dt_s=mutated_input["sample_dt_s"],
        ego_history_xy=mutated_input["ego_history_xy"],
        counterpart_history_xy=mutated_input["counterpart_history_xy"],
        route_intent=mutated_input["route_intent"],
    )
    mutated_expected_path.write_bytes(M3.canonical_json_bytes(mutated_expected))
    failed = M3.run_nc_pretstar_history_only_gate(
        repo_root=copied_root,
        python_executable_path=python,
        python_executable_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        environment_manifest_path=environment,
        environment_manifest_sha256=hashlib.sha256(environment.read_bytes()).hexdigest(),
        created_at_utc="2026-07-16T00:00:00Z",
    )
    _assert_nc_receipt_schema(failed)
    assert failed["status"] == "FAIL"
    assert failed["failure_or_NA"]["failure_class"] == "FIXTURE_HASH_MISMATCH"


def _moving_ticks(start: int, stop: int, *, y: float = 0.0) -> dict[int, tuple[float, float]]:
    return {tick: (float(tick), y) for tick in range(start, stop + 1)}


def _available_scene(segment_id: str = "scene-a") -> DOMAIN.SceneDomainInput:
    sampling = {}
    for sampling_id, rate_hz in DOMAIN.SAMPLING_AXIS:
        stop = 5 * rate_hz
        candidates = {
            "C1": _moving_ticks(-3 * rate_hz, stop, y=0.0),
            "C2": _moving_ticks(-3 * rate_hz, stop, y=1.0),
            "C3": _moving_ticks(-3 * rate_hz, stop, y=2.0),
        }
        sampling[sampling_id] = DOMAIN.SamplingTimelines(
            candidates=candidates,
            counterpart=_moving_ticks(-3 * rate_hz, stop, y=3.0),
        )
    return DOMAIN.SceneDomainInput(
        segment_id=segment_id, path_type_or_NA="CP", sampling=sampling
    )


def test_w2_anchor_domain_all_families_exact_csv_bytes_and_horizon_rules() -> None:
    rows = DOMAIN.build_anchor_domain_rows([_available_scene()])
    assert len({(row["segment_id"], row["feature_id"], row["horizon_id"]) for row in rows}) == 32
    assert {row["feature_id"] for row in rows} == set(DOMAIN.FEATURE_FAMILIES)
    assert all(row["membership_status"] == "AVAILABLE" for row in rows)
    for feature_id in DOMAIN.FEATURE_FAMILIES:
        h20 = [
            row
            for row in rows
            if row["feature_id"] == feature_id and row["horizon_id"] == "H20"
        ]
        sampling_id = feature_id.split("-")[1]
        rate_hz = dict(DOMAIN.SAMPLING_AXIS)[sampling_id]
        assert [int(row["tau_tick_or_NA"]) for row in h20] == list(
            range(rate_hz, 2 * rate_hz + 1)
        )
        h_values = {row["h_common_tick_or_NA"] for row in h20}
        assert ("-TF" in feature_id and h_values == {str(5 * rate_hz)}) or (
            "-TF" not in feature_id and h_values == {"NA"}
        )
    encoded = DOMAIN.encode_anchor_domain_csv(rows)
    assert encoded.startswith((",".join(DOMAIN.DOMAIN_COLUMNS) + "\n").encode())
    assert encoded.endswith(b"\n") and b"\r" not in encoded
    assert encoded == DOMAIN.encode_anchor_domain_csv(
        DOMAIN.build_anchor_domain_rows([_available_scene()])
    )

    broken = _available_scene("scene-b")
    broken_candidates = {
        key: dict(value) for key, value in broken.sampling["R04N"].candidates.items()
    }
    del broken_candidates["C2"][4]
    broken_sampling = dict(broken.sampling)
    broken_sampling["R04N"] = DOMAIN.SamplingTimelines(
        candidates=broken_candidates,
        counterpart=broken.sampling["R04N"].counterpart,
    )
    broken_rows = DOMAIN.build_anchor_domain_rows(
        [
            DOMAIN.SceneDomainInput(
                segment_id="scene-b", path_type_or_NA="CP", sampling=broken_sampling
            )
        ]
    )
    for horizon_id in DOMAIN.HORIZON_AXIS:
        group = [
            row
            for row in broken_rows
            if row["feature_id"] == "F-R04N-CH-W10"
            and row["horizon_id"] == horizon_id
        ]
        assert len(group) == 1
        assert group[0]["membership_status"] == "INELIGIBLE_TIMELINE_SUPPORT"


def test_w2_h_common_is_largest_joint_tick_and_endpoint_hole_preserves_tf() -> None:
    scene = _available_scene("joint-endpoint-hole")
    sampling = dict(scene.sampling)
    r04 = sampling["R04N"]
    candidates = {key: dict(value) for key, value in r04.candidates.items()}
    del candidates["C2"][20]
    candidates["C2"][21] = (21.0, 1.0)
    sampling["R04N"] = DOMAIN.SamplingTimelines(
        candidates=candidates, counterpart=r04.counterpart
    )
    rows = DOMAIN.build_anchor_domain_rows(
        [DOMAIN.SceneDomainInput(scene.segment_id, scene.path_type_or_NA, sampling)]
    )
    tf_h20 = [
        row for row in rows
        if row["feature_id"] == "F-R04N-TF" and row["horizon_id"] == "H20"
    ]
    tf_hfeas = [
        row for row in rows
        if row["feature_id"] == "F-R04N-TF" and row["horizon_id"] == "HFEAS"
    ]
    assert {row["h_common_tick_or_NA"] for row in tf_h20 + tf_hfeas} == {"19"}
    assert [int(row["tau_tick_or_NA"]) for row in tf_h20] == list(range(4, 9))
    assert [int(row["tau_tick_or_NA"]) for row in tf_hfeas] == list(range(4, 20))


def test_w2_derived_state_overflow_is_fail_closed_nonfinite() -> None:
    scene = _available_scene("overflow")
    sampling = dict(scene.sampling)
    r04 = sampling["R04N"]
    candidates = {key: dict(value) for key, value in r04.candidates.items()}
    candidates["C1"].update(
        {0: (1.7e308, 0.0), 1: (-1.7e308, 0.0), 2: (1.7e308, 0.0)}
    )
    sampling["R04N"] = DOMAIN.SamplingTimelines(
        candidates=candidates, counterpart=r04.counterpart
    )
    rows = DOMAIN.build_anchor_domain_rows(
        [DOMAIN.SceneDomainInput(scene.segment_id, scene.path_type_or_NA, sampling)]
    )
    group = [
        row for row in rows
        if row["feature_id"] == "F-R04N-CH-W10" and row["horizon_id"] == "H20"
    ]
    assert len(group) == 1
    assert group[0]["membership_status"] == "INELIGIBLE_STATE_NONFINITE"
    assert group[0]["reason_code"] == "F_STATE_NONFINITE"


def test_w2_anchor_domain_authority_example_has_exact_lf_csv_bytes() -> None:
    rows = [
        {
            "segment_id": "scene-a",
            "feature_id": "F-R04N-CH-W10",
            "horizon_id": "H20",
            "path_type_or_NA": "CP",
            "h_common_tick_or_NA": "NA",
            "tau_tick_or_NA": "4",
            "membership_status": "AVAILABLE",
            "reason_code": "F_AVAILABLE_CONTINUE",
        },
        {
            "segment_id": "scene-b",
            "feature_id": "F-R04N-CH-W10",
            "horizon_id": "H20",
            "path_type_or_NA": "NA",
            "h_common_tick_or_NA": "NA",
            "tau_tick_or_NA": "NA",
            "membership_status": "MISSING_WOD_PATH_TYPE",
            "reason_code": "F_MISSING_WOD_PATH_TYPE",
        },
    ]
    assert DOMAIN.encode_anchor_domain_csv(rows) == (
        b"segment_id,feature_id,horizon_id,path_type_or_NA,h_common_tick_or_NA,"
        b"tau_tick_or_NA,membership_status,reason_code\n"
        b"scene-a,F-R04N-CH-W10,H20,CP,NA,4,AVAILABLE,F_AVAILABLE_CONTINUE\n"
        b"scene-b,F-R04N-CH-W10,H20,NA,NA,NA,MISSING_WOD_PATH_TYPE,"
        b"F_MISSING_WOD_PATH_TYPE\n"
    )


def test_w2_anchor_domain_validator_rejects_cross_group_and_membership_drift() -> None:
    rows = DOMAIN.build_anchor_domain_rows([_available_scene()])
    path_drift = copy.deepcopy(rows)
    next(
        row for row in path_drift if row["feature_id"] == "F-R04N-LF-W10"
    )["path_type_or_NA"] = "HO"
    with pytest.raises(DOMAIN.AnchorDomainError, match="path-type drift|cross-group"):
        DOMAIN.validate_anchor_domain_rows(path_drift, expected_scene_count=1)

    h20_drift = copy.deepcopy(rows)
    h20_drift.remove(
        next(
            row
            for row in h20_drift
            if row["feature_id"] == "F-R04N-CH-W10"
            and row["horizon_id"] == "H20"
            and row["tau_tick_or_NA"] == "4"
        )
    )
    with pytest.raises(DOMAIN.AnchorDomainError, match="H20 membership drift"):
        DOMAIN.validate_anchor_domain_rows(h20_drift, expected_scene_count=1)

    hfeas_drift = copy.deepcopy(rows)
    first_hfeas_index = next(
        index
        for index, row in enumerate(hfeas_drift)
        if row["feature_id"] == "F-R04N-CH-W10"
        and row["horizon_id"] == "HFEAS"
    )
    below_range = copy.deepcopy(hfeas_drift[first_hfeas_index])
    below_range["tau_tick_or_NA"] = "-1"
    hfeas_drift.insert(first_hfeas_index, below_range)
    with pytest.raises(DOMAIN.AnchorDomainError, match="HFEAS lower-bound drift"):
        DOMAIN.validate_anchor_domain_rows(hfeas_drift, expected_scene_count=1)

    negative_ticks = {tick: (float(tick), 0.0) for tick in range(-5, 0)}
    negative_scene = DOMAIN.SceneDomainInput(
        segment_id="negative-h-common",
        path_type_or_NA="CP",
        sampling={
            sampling_id: DOMAIN.SamplingTimelines(
                candidates={candidate_id: negative_ticks for candidate_id in ("C1", "C2", "C3")},
                counterpart=negative_ticks,
            )
            for sampling_id, _ in DOMAIN.SAMPLING_AXIS
        },
    )
    negative_rows = DOMAIN.build_anchor_domain_rows([negative_scene])
    assert all(row["membership_status"] == "INELIGIBLE_TIMELINE_SUPPORT" for row in negative_rows)
    assert all(row["h_common_tick_or_NA"] == "NA" for row in negative_rows)
    terminal_h_common_drift = copy.deepcopy(negative_rows)
    next(
        row for row in terminal_h_common_drift if row["feature_id"] == "F-R04N-CH-W10"
    )["h_common_tick_or_NA"] = "4"
    with pytest.raises(DOMAIN.AnchorDomainError, match="non-TF terminal h_common drift"):
        DOMAIN.validate_anchor_domain_rows(
            terminal_h_common_drift, expected_scene_count=1
        )


def test_w2_anchor_domain_cli_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"scenes":[],"scenes":[]}\n', encoding="utf-8")
    with pytest.raises(DOMAIN.AnchorDomainError, match="duplicate JSON key"):
        DOMAIN._strict_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(DOMAIN.AnchorDomainError, match="nonfinite JSON token"):
        DOMAIN._strict_load(nonfinite)


def test_w2_scene_path_and_status_are_derived_only_after_full_upstream_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    mapping_root = tmp_path / "mapping"
    bundle.mkdir()
    mapping_root.mkdir()
    scene_manifest = bundle / "blind_scene_manifest.csv"
    with scene_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("segment_id", "candidate_geometry_available", "tstar_context_step"))
        writer.writerow(("scene-000", "true", "10"))
        for index in range(1, 479):
            writer.writerow((f"scene-{index:03d}", "false", "NA"))
    mapping_csv = mapping_root / "wod_path_type_mapping.csv"
    mapping_csv.write_text(
        "segment_id,tstar_context_step,path_type\nscene-000,10,CP\n", encoding="utf-8"
    )
    source_manifest = tmp_path / "file_manifest.json"
    mapping_manifest = tmp_path / "mapping_manifest.json"
    for path in (source_manifest, mapping_manifest):
        path.write_bytes(b"{}\n")
    calls: list[str] = []
    monkeypatch.setattr(
        DOMAIN, "validate_score_stripped_bundle", lambda **kwargs: calls.append("source")
    )
    monkeypatch.setattr(
        DOMAIN,
        "validate_wod_path_type_mapping_manifest",
        lambda *args, **kwargs: calls.append("mapping"),
    )
    registry = DOMAIN.load_verified_scene_registry(
        bundle_root=bundle,
        score_schema_path=tmp_path / "schema.json",
        source_manifest_path=source_manifest,
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        export_receipt_path=tmp_path / "receipt.json",
        path_mapping_manifest_path=mapping_manifest,
        path_mapping_manifest_sha256=hashlib.sha256(mapping_manifest.read_bytes()).hexdigest(),
        mapping_root=mapping_root,
    )
    assert calls == ["source", "mapping"]
    assert registry.by_segment["scene-000"] == ("CP", None)
    assert registry.by_segment["scene-001"] == ("NA", "INELIGIBLE_BLIND")
    with pytest.raises(DOMAIN.AnchorDomainError, match="only segment_id and resampled sampling"):
        DOMAIN._parse_scene(
            {"segment_id": "scene-000", "sampling": {}, "path_type_or_NA": "HO"},
            registry,
        )


def test_w2_anchor_domain_479_scene_count_manifest_and_determinism() -> None:
    scenes = [
        DOMAIN.SceneDomainInput(
            segment_id=f"scene-{index:03d}", path_type_or_NA="NA", sampling={}
        )
        for index in range(479)
    ]
    rows = DOMAIN.build_anchor_domain_rows(scenes)
    DOMAIN.validate_anchor_domain_rows(rows)
    assert len(rows) == 15328
    assert len({(row["segment_id"], row["feature_id"], row["horizon_id"]) for row in rows}) == 15328
    assert all(row["membership_status"] == "MISSING_WOD_PATH_TYPE" for row in rows)
    artifact = DOMAIN.encode_anchor_domain_csv(rows)
    golden = W2B_FIXTURES / "anchor_domain_15328_golden.csv"
    binding = _load(W2B_FIXTURES / "anchor_domain_15328_golden.binding.json")
    assert artifact == golden.read_bytes()
    assert len(artifact) == binding["artifact_size_bytes"] == 1272338
    assert hashlib.sha256(artifact).hexdigest() == binding["artifact_sha256"] == (
        "019f523406b0db89fffd5d24e503e04bee34abb625449879c88db07994ac2e84"
    )
    mutated = bytearray(artifact)
    mutated[-2] ^= 1
    assert hashlib.sha256(mutated).hexdigest() != binding["artifact_sha256"]


def test_w2_anchor_manifest_rehashes_every_bound_input(tmp_path: Path) -> None:
    scenes = [
        DOMAIN.SceneDomainInput(
            segment_id=f"scene-{index:03d}", path_type_or_NA="NA", sampling={}
        )
        for index in range(479)
    ]
    rows = DOMAIN.build_anchor_domain_rows(scenes)
    artifact = DOMAIN.encode_anchor_domain_csv(rows)
    source_manifest = tmp_path / "file_manifest.json"
    mapping_manifest = tmp_path / "mapping_manifest.json"
    environment_manifest = tmp_path / "environment.json"
    python = Path(sys.executable).resolve(strict=True)
    for path, payload in (
        (source_manifest, b"source\n"),
        (mapping_manifest, b"mapping\n"),
        (environment_manifest, b"environment\n"),
    ):
        path.write_bytes(payload)
    registry = DOMAIN.VerifiedSceneRegistry(
        by_segment={scene.segment_id: ("NA", "MISSING_WOD_PATH_TYPE") for scene in scenes},
        source_manifest_path=source_manifest,
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        path_mapping_manifest_path=mapping_manifest,
        path_mapping_manifest_sha256=hashlib.sha256(mapping_manifest.read_bytes()).hexdigest(),
    )
    kwargs = {
        "artifact_bytes": artifact,
        "rows": rows,
        "generator_path": ROOT / "scripts/rq014/build_wod_scene_anchor_domain.py",
        "repo_root": ROOT,
        "verified_scene_registry": registry,
        "python_executable_path": python,
        "python_executable_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "environment_manifest_path": environment_manifest,
        "environment_manifest_sha256": hashlib.sha256(
            environment_manifest.read_bytes()
        ).hexdigest(),
        "created_at_utc": "2026-07-16T00:00:00Z",
    }
    first = DOMAIN.build_manifest(**kwargs)
    second = DOMAIN.build_manifest(**kwargs)
    assert first == second
    assert first["group_count"] == first["terminal_group_count"] == 15328
    assert first["available_group_count"] == 0
    assert first["artifact_sha256"] == hashlib.sha256(artifact).hexdigest()
    assert M3.canonical_json_bytes(first) == M3.canonical_json_bytes(second)
    source_manifest.write_bytes(b"mutated\n")
    with pytest.raises(DOMAIN.AnchorDomainError, match="source manifest SHA-256 mismatch"):
        DOMAIN.build_manifest(**kwargs)


def test_w2_does_not_expose_scorer_or_managed_g2r_operation() -> None:
    source = (ROOT / "scripts/rq014/build_wod_m3_anchors.py").read_text(encoding="utf-8")
    assert "m3_scorer.joblib" not in source
    contract = _load(ROOT / "reports/plans/RQ014_g2r_output_contract_v1.json")
    assert contract["future_operation_binding"]["central_authorization"] == "DENY"
    authorization = _load(ROOT / "configs/research_authorization.json")
    assert "rq014_r2_blind_feature_build" not in authorization["authorizations"][
        "RQ014"
    ]["allowed_operations"]
    assert not (ROOT / "scripts/rq014/run_g2r.py").exists()
