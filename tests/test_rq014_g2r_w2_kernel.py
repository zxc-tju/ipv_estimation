from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rq014 import build_wod_m3_anchors as M3
from scripts.rq014 import build_wod_scene_anchor_domain as DOMAIN


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "rq014_g2r_v1"
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
    first = M3.build_wod_m3_input_row(
        ego,
        counterpart,
        sample_dt_s=0.25,
        case_start_timestamp_s=0.0,
        context_end_timestamp_s=2.5,
        route_intent="GO_STRAIGHT",
        counterpart_is_vehicle=True,
    )
    second = M3.build_wod_m3_input_row(
        ego,
        counterpart,
        sample_dt_s=0.25,
        case_start_timestamp_s=0.0,
        context_end_timestamp_s=2.5,
        route_intent="GO_STRAIGHT",
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


def test_w2_nc_kernel_reproduces_all_five_w1_pairs_and_future_invariance() -> None:
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
    inputs = [_nc_input(fixture_id) for fixture_id in NC_IDS]
    observed = M3.run_nc_pretstar_history_only_gate(inputs)
    for fixture_id, payload in zip(NC_IDS, observed):
        expected_path = FIXTURES / f"{fixture_id.lower()}_expected.json"
        assert payload == _nc_expected(fixture_id)
        assert M3.canonical_json_bytes(payload) == expected_path.read_bytes()

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
    with pytest.raises(M3.WodM3KernelError, match="five unique"):
        M3.run_nc_pretstar_history_only_gate([*inputs, inputs[0]])


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
    kwargs = {
        "artifact_bytes": artifact,
        "rows": rows,
        "generator_path": ROOT / "scripts/rq014/build_wod_scene_anchor_domain.py",
        "recovery_contract_sha256": "a" * 64,
        "envelope_contract_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "path_mapping_sha256": "d" * 64,
        "python_executable_sha256": "e" * 64,
        "environment_manifest_sha256": "f" * 64,
        "created_at_utc": "2026-07-16T00:00:00Z",
    }
    first = DOMAIN.build_manifest(**kwargs)
    second = DOMAIN.build_manifest(**kwargs)
    assert first == second
    assert first["group_count"] == first["terminal_group_count"] == 15328
    assert first["available_group_count"] == 0
    assert first["artifact_sha256"] == hashlib.sha256(artifact).hexdigest()
    assert M3.canonical_json_bytes(first) == M3.canonical_json_bytes(second)


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
