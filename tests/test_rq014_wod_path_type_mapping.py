from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.rq014.derive_wod_path_type_mapping import (
    BUNDLE_HEADERS,
    DIRECTION_SPEED_TOLERANCE_MPS,
    OLS_DENOMINATOR_TOLERANCE_S2,
    TSTAR_TIME_TOLERANCE_S,
    canonical_json_bytes,
    classify_rating_blind_primitives,
    derive_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/rq014_wod_path_type_mapping_golden_v1.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
BOUNDARY_CASE_IDS = {
    "BOUNDARY_ANGLE_45_CP",
    "BOUNDARY_ANGLE_45_JUST_ABOVE_CP",
    "BOUNDARY_ANGLE_45_JUST_BELOW_MP",
    "BOUNDARY_ANGLE_135_CP",
    "BOUNDARY_LATERAL_4_MP",
    "BOUNDARY_LATERAL_5_HO",
    "BOUNDARY_LONGITUDINAL_MINUS_8_MP",
    "UNMAPPED_MISSING_COUNTERPART",
    "UNMAPPED_LOW_MOTION",
    "UNMAPPED_OPPOSING_NEARBY",
}
EXACT_BOUNDARY_EXPECTED_MEASUREMENTS = {
    "BOUNDARY_ANGLE_45_CP": ("angle_deg", 45.0),
    "BOUNDARY_ANGLE_135_CP": ("angle_deg", 135.0),
    "BOUNDARY_LATERAL_4_MP": ("lateral_m", 4.0),
    "BOUNDARY_LATERAL_5_HO": ("lateral_m", 5.0),
    "BOUNDARY_LONGITUDINAL_MINUS_8_MP": ("longitudinal_m", -8.0),
}
PUBLISHED_MAPPING = (
    "/share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/"
    "wod_path_type_mapping/v1/wod_path_type_mapping.csv"
)


def _csv_bytes(name: str, rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=BUNDLE_HEADERS[name],
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _state_row(
    segment_id: str,
    sample_index: int,
    time_s: float,
    x: float,
    y: float,
) -> dict[str, str]:
    return {
        "segment_id": segment_id,
        "tstar_context_step": "149",
        "sample_index": str(sample_index),
        "time_s": str(time_s),
        "pos_x_m": str(x),
        "pos_y_m": str(y),
        "pos_z_m": "NA",
        "vel_x_mps": "NA",
        "vel_y_mps": "NA",
        "accel_x_mps2": "NA",
        "accel_y_mps2": "NA",
    }


def _build_synthetic_bundle(root: Path) -> tuple[str, str]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case_by_id = {case["case_id"]: case for case in fixture["cases"]}
    chosen = [
        case_by_id["CP_CROSSING"],
        case_by_id["HO_OPPOSING"],
        case_by_id["MP_LEADING_OR_MERGING"],
        case_by_id["F_SAME_LANE_OR_FOLLOWING"],
        case_by_id["UNMAPPED_PARALLEL_NEARBY"],
    ]
    root.mkdir()
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in BUNDLE_HEADERS}
    for index in range(479):
        segment_id = f"scene{index:04d}"
        if index >= len(chosen):
            rows["blind_scene_manifest.csv"].append(
                {
                    "segment_id": segment_id,
                    "tstar_context_step": "NA",
                    "source_shard_id": "NA",
                    "scenario_cluster": "NA",
                    "path_type": "NA",
                    "route_intent_code": "NA",
                    "route_intent_name": "NA",
                    "coordinate_frame": "ego_at_tstar",
                    "native_frame_rate_hz": "10.0",
                    "state_rate_hz": "NA",
                    "candidate_rate_hz": "NA",
                    "candidate_count": "0",
                    "candidate_geometry_available": "false",
                    "ego_future_state_count": "0",
                    "tstar_ego_pose_element_count": "0",
                    "structural_status": "MISSING_DECLASSIFIED_PHASE1_SCENE",
                    "candidate_set_sha256": hashlib.sha256(b"").hexdigest(),
                }
            )
            rows["structural_attrition.csv"].append(
                {
                    "segment_id": segment_id,
                    "stage": "DECLASSIFICATION",
                    "reason_code": "MISSING_DECLASSIFIED_PHASE1_SCENE",
                    "source_receipt_id": "golden_fixture",
                }
            )
            continue

        case = chosen[index]
        rows["blind_scene_manifest.csv"].append(
            {
                "segment_id": segment_id,
                "tstar_context_step": "149",
                "source_shard_id": "fixture",
                "scenario_cluster": "NA",
                "path_type": "UNMAPPED",
                "route_intent_code": str(case["route_intent_code"]),
                "route_intent_name": case["route_intent_name"],
                "coordinate_frame": "ego_at_tstar",
                "native_frame_rate_hz": "10.0",
                "state_rate_hz": "4.0",
                "candidate_rate_hz": "4.0",
                "candidate_count": "3",
                "candidate_geometry_available": "true",
                "ego_future_state_count": "20",
                "tstar_ego_pose_element_count": "16",
                "structural_status": "GEOMETRY_AVAILABLE",
                "candidate_set_sha256": hashlib.sha256(segment_id.encode()).hexdigest(),
            }
        )
        for sample_index in range(16):
            time_s = -3.75 + 0.25 * sample_index
            rows["ego_history_states.csv"].append(
                _state_row(segment_id, sample_index, time_s, 4.0 * time_s, 0.0)
            )
        for sample_index in range(20):
            time_s = 0.25 * (sample_index + 1)
            rows["ego_future_states.csv"].append(
                _state_row(segment_id, sample_index, time_s, 4.0 * time_s, 0.0)
            )
        for matrix_row in range(4):
            for matrix_column in range(4):
                rows["tstar_ego_pose.csv"].append(
                    {
                        "segment_id": segment_id,
                        "tstar_context_step": "149",
                        "matrix_row": str(matrix_row),
                        "matrix_column": str(matrix_column),
                        "value": "1.0" if matrix_row == matrix_column else "0.0",
                    }
                )
        observed = case["counterpart_observed"]
        for context_step, (time_s, x, y, vx, vy) in enumerate(observed):
            rows["counterpart_tracks.csv"].append(
                {
                    "segment_id": segment_id,
                    "tstar_context_step": "149",
                    "counterpart_track_id": "fixture-track",
                    "context_step": str(context_step),
                    "time_s": str(time_s),
                    "x_m": str(x),
                    "y_m": str(y),
                    "vx_mps": str(vx),
                    "vy_mps": str(vy),
                    "class_name": "Vehicle",
                    "detector_confidence": "1.0",
                }
            )

    payloads = {name: _csv_bytes(name, value) for name, value in rows.items()}
    manifest_rows = []
    for name in sorted(payloads):
        payload = payloads[name]
        (root / name).write_bytes(payload)
        manifest_rows.append(
            {
                "contains_rating": False,
                "primary_key": ["fixture"],
                "relative_path": name,
                "row_count": len(rows[name]),
                "schema_id": f"rq014-score-stripped-schema-v1#{name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    file_manifest = canonical_json_bytes(
        {
            "files": manifest_rows,
            "schema_version": "rq014-score-stripped-file-manifest-v1",
        }
    )
    (root / "file_manifest.json").write_bytes(file_manifest)
    receipt = canonical_json_bytes(
        {
            "candidate_count_distribution": {"0": 474, "3": 5},
            "geometry_available_scene_count": 5,
            "output_file_hashes": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(payloads.items())
            },
            "schema_version": "rq014-score-stripped-sanitization-v1",
            "universe_segment_count": 479,
        }
    )
    (root / "sanitization_receipt.json").write_bytes(receipt)
    return hashlib.sha256(file_manifest).hexdigest(), hashlib.sha256(receipt).hexdigest()


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda case: case["case_id"])
def test_golden_path_type_case(case: dict[str, object]) -> None:
    assert GOLDEN["contains_rating"] is False
    result = classify_rating_blind_primitives(
        route_intent_code=case["route_intent_code"],
        route_intent_name=case["route_intent_name"],
        tstar_pose_row_major=case["tstar_pose_row_major"],
        ego_history=case["ego_history"],
        ego_future=case["ego_future"],
        counterpart_observed=case["counterpart_observed"],
    )
    assert result["path_type"] == case["expected_path_type"]
    assert result["status"] == case["expected_status"]
    if case["case_id"] in EXACT_BOUNDARY_EXPECTED_MEASUREMENTS:
        field, value = EXACT_BOUNDARY_EXPECTED_MEASUREMENTS[case["case_id"]]
        assert result[field] == value
    if case["case_id"] == "BOUNDARY_ANGLE_45_JUST_BELOW_MP":
        assert result["angle_deg"] < 45.0
    if case["case_id"] == "BOUNDARY_ANGLE_45_JUST_ABOVE_CP":
        assert result["angle_deg"] > 45.0


def test_golden_fixture_covers_every_reviewed_boundary() -> None:
    assert len(GOLDEN["cases"]) == 15
    assert BOUNDARY_CASE_IDS <= {case["case_id"] for case in GOLDEN["cases"]}


def test_addendum_normatively_defines_float_comparison_tolerances() -> None:
    text = (
        ROOT / "reports/plans/RQ014_plan_v1p7_addendum_pathtype_20260713.md"
    ).read_text(encoding="utf-8")
    for clause in ("1e-18 s^2", "1e-12 s", "binary64", "OLS conditioning"):
        assert clause in text
    assert OLS_DENOMINATOR_TOLERANCE_S2 == 1e-18
    assert TSTAR_TIME_TOLERANCE_S == 1e-12
    assert DIRECTION_SPEED_TOLERANCE_MPS == 1e-9


def test_full_bundle_derivation_is_deterministic_and_excludes_undecidable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    file_manifest_sha, receipt_sha = _build_synthetic_bundle(bundle)
    first = derive_mapping(
        bundle_root=bundle,
        output_dir=tmp_path / "first",
        expected_file_manifest_sha256=file_manifest_sha,
        expected_sanitization_receipt_sha256=receipt_sha,
        published_mapping_path=PUBLISHED_MAPPING,
    )
    second = derive_mapping(
        bundle_root=bundle,
        output_dir=tmp_path / "second",
        expected_file_manifest_sha256=file_manifest_sha,
        expected_sanitization_receipt_sha256=receipt_sha,
        published_mapping_path=PUBLISHED_MAPPING,
    )
    for name in ("wod_path_type_mapping.csv", "manifest.json", "distribution_summary.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    mapping_rows = list(
        csv.DictReader((tmp_path / "first/wod_path_type_mapping.csv").open(newline="", encoding="utf-8"))
    )
    assert [row["path_type"] for row in mapping_rows] == ["CP", "HO", "MP", "F"]
    assert first["path_type_counts"] == {"CP": 1, "HO": 1, "MP": 1, "F": 1}
    assert first["status_counts"] == {
        "K_EXCLUDED_STRUCTURAL_NO_GEOMETRY": 474,
        "MAPPED_CROSSING": 1,
        "MAPPED_LEADING_OR_MERGING": 1,
        "MAPPED_OPPOSING": 1,
        "MAPPED_SAME_LANE_OR_FOLLOWING": 1,
        "UNMAPPED_EXCLUDED_PARALLEL_NEARBY": 1,
    }
    assert first["attrition_stage_counts"] == {
        "F_MISSING_WOD_PATH_TYPE": 1,
        "K_STRUCTURAL_NO_GEOMETRY": 474,
        "MAPPED_PATH_TYPE": 4,
    }
    assert first["float_comparison_contract"] == {
        "direction_speed_tolerance_mps": 1e-9,
        "ols_denominator_tolerance_s2": 1e-18,
        "path_type_boundaries": "direct_binary64_comparisons_without_epsilon",
        "tstar_time_tolerance_s": 1e-12,
    }
    manifest = json.loads((tmp_path / "first/manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 4
    assert manifest["mapping"]["path"] == PUBLISHED_MAPPING
    assert manifest["contains_rating"] is False
    assert first["contains_rating"] is False
