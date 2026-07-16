from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

from sociality_estimation.verifier.anchors import (
    CATEGORY_COLUMNS,
    HISTORY_COLUMNS,
    build_m3_anchor_features,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "reports" / "plans" / "RQ014_g2r_output_contract_v1.json"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "rq014_g2r_v1"
SHAPES_PATH = FIXTURE_ROOT / "schema_shape_goldens.json"
M3_INPUT_PATH = FIXTURE_ROOT / "m3_input_row_expected.json"
FEATURE_FIXTURE_PATH = FIXTURE_ROOT / "wod_m3_feature_construction_golden.json"
GEOMETRY_FIXTURE_PATH = FIXTURE_ROOT / "geometry_path_crosstab_expected.json"
CELL_ORDER_FIXTURE_PATH = FIXTURE_ROOT / "canonical_cell_order_golden.json"
BLIND_FIXTURE_PATH = FIXTURE_ROOT / "blind_scene_predicate_golden.json"
FIXTURE_MANIFEST_PATH = FIXTURE_ROOT / "fixture_manifest.json"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"nonfinite JSON token: {token}")


def _strict_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )


def _strict_load(path: Path) -> Any:
    return _strict_loads(path.read_text(encoding="utf-8"))


def _normalize_signed_zero(value: Any) -> Any:
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_normalize_signed_zero(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_signed_zero(item) for key, item in value.items()}
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize_signed_zero(value)
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _assert_exact_keys(payload: dict[str, Any], exact_keys: list[str]) -> None:
    missing = sorted(set(exact_keys) - set(payload))
    extra = sorted(set(payload) - set(exact_keys))
    if missing or extra:
        raise ValueError(f"exact-key mismatch missing={missing} extra={extra}")


def _complete_object_contracts(node: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" in node:
            yield path, node
        for key, value in node.items():
            yield from _complete_object_contracts(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _complete_object_contracts(value, f"{path}/{index}")


def _path_category(absolute_heading_difference_degrees: float) -> str:
    if absolute_heading_difference_degrees < 25.0:
        return "F"
    if absolute_heading_difference_degrees > 135.0:
        return "CP"
    return "MP"


def _path_relation(category: str, longitudinal_m: float, lateral_m: float) -> str:
    if category == "F":
        return "F"
    if category == "CP":
        return "O-C" if abs(lateral_m) < 6.0 else "C-C"
    return "P-M" if abs(longitudinal_m) >= abs(lateral_m) else "P-P"


def _priority_role(longitudinal_m: float) -> str:
    if abs(longitudinal_m) < 2.0:
        return "equal"
    return "yield" if longitudinal_m > 0.0 else "priority"


def _turn_label(unwrapped_delta_degrees: float) -> str:
    if unwrapped_delta_degrees >= 12.0:
        return "L"
    if unwrapped_delta_degrees <= -12.0:
        return "R"
    return "S"


def _deviations(value: float, lower: float, median: float, upper: float) -> tuple[float, float, float]:
    if not all(math.isfinite(item) for item in (value, lower, median, upper)):
        raise ValueError("nonfinite deviation input")
    if not lower < median < upper:
        raise ValueError("invalid strict interval")
    if value < lower:
        nex = (lower - value) / (median - lower)
    elif value > upper:
        nex = (value - upper) / (upper - median)
    else:
        nex = 0.0
    if value < median:
        nmd = (median - value) / (median - lower)
    elif value > median:
        nmd = (value - median) / (upper - median)
    else:
        nmd = 0.0
    return nex, nmd, abs(value - median)


def _blind_scene_eligible(example: dict[str, Any], cell_count: int) -> bool:
    overrides = example["overrides_by_cell_index"]
    for index in range(cell_count):
        cell = overrides.get(str(index), example["default_cell"])
        if cell["available"] != [True, True, True]:
            return False
        values = cell["predictor_values_or_NA"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return False
        if not all(math.isfinite(float(value)) for value in values):
            return False
        normalized = [0.0 if float(value) == 0.0 else float(value) for value in values]
        if normalized[0] == normalized[1] == normalized[2]:
            return False
    return True


def test_all_fixture_and_contract_json_is_strict_and_canonical() -> None:
    paths = [CONTRACT_PATH, *sorted(FIXTURE_ROOT.glob("*.json"))]
    for path in paths:
        payload = _strict_load(path)
        assert path.read_bytes() == _canonical_json_bytes(payload), path

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _strict_loads('{"a":1,"a":2}')
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="nonfinite JSON token"):
            _strict_loads(f'{{"value":{token}}}')


@pytest.mark.parametrize(
    "shape_id",
    [
        "anchor_score_row",
        "availability_mask_row",
        "blind_feature_row",
        "nc_gate_receipt_shape_only",
        "operation_receipt",
        "output_manifest",
        "predictor_manifest_row",
    ],
)
def test_every_g2r_schema_rejects_missing_and_extra_root_keys(shape_id: str) -> None:
    shape = _strict_load(SHAPES_PATH)["schemas"][shape_id]
    schema = _strict_load(ROOT / shape["schema_path"])
    exact_keys = shape["exact_keys"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == exact_keys
    assert list(schema["properties"]) == exact_keys

    key_only_payload = {key: None for key in exact_keys}
    _assert_exact_keys(key_only_payload, exact_keys)
    missing = copy.deepcopy(key_only_payload)
    missing.pop(exact_keys[0])
    with pytest.raises(ValueError, match="missing="):
        _assert_exact_keys(missing, exact_keys)
    extra = copy.deepcopy(key_only_payload)
    extra["unexpected"] = None
    with pytest.raises(ValueError, match="extra="):
        _assert_exact_keys(extra, exact_keys)

    for object_path, object_schema in _complete_object_contracts(schema):
        assert object_schema["additionalProperties"] is False, object_path
        assert set(object_schema["required"]) == set(object_schema["properties"]), object_path


def test_wod_m3_feature_construction_matches_all_32_columns_and_d8_bytes() -> None:
    contract = _strict_load(CONTRACT_PATH)
    fixture = _strict_load(FEATURE_FIXTURE_PATH)
    expected = _strict_load(M3_INPUT_PATH)
    assert fixture["history_columns"] == list(HISTORY_COLUMNS)
    assert list(fixture["categories"]) == sorted(CATEGORY_COLUMNS)

    history = pd.DataFrame(fixture["history_rows"], columns=fixture["history_columns"])
    observed = build_m3_anchor_features(
        history,
        fixture["categories"],
        case_start_timestamp_s=fixture["case_start_timestamp_s"],
    ).iloc[0].to_dict()
    columns = contract["wod_to_m3_feature_contract"]["m3_input_row_preimage"]["columns"]
    assert columns == expected["columns"]
    assert len(columns) == 32
    assert [item["name"] for item in contract["wod_to_m3_feature_contract"]["numeric_feature_formulas"]] == columns[:25]
    assert list(CATEGORY_COLUMNS) == columns[25:]

    observed_row = {
        "schema_version": "rq014-g2r-m3-input-row-v1",
        "columns": columns,
        "values": [observed[column] for column in columns],
    }
    assert observed_row == expected
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in expected["values"][:25]
    )
    assert all(isinstance(value, str) for value in expected["values"][25:])
    expected_bytes = M3_INPUT_PATH.read_bytes()
    assert _canonical_json_bytes(observed_row) == expected_bytes
    binding = fixture["expected_m3_input_row"]
    assert len(expected_bytes) == binding["size_bytes"]
    assert hashlib.sha256(expected_bytes).hexdigest() == binding["sha256"]


def test_d5_d6_d7_category_boundaries_and_wod_feature_categories_are_exact() -> None:
    fixture = _strict_load(GEOMETRY_FIXTURE_PATH)
    for case in fixture["path_category_cases"]:
        assert _path_category(case["absolute_heading_difference_degrees"]) == case["expected"]
    for case in fixture["path_relation_cases"]:
        assert _path_relation(
            case["geometry_path_category"],
            case["absolute_ego_longitudinal_m"],
            case["absolute_ego_lateral_m"],
        ) == case["expected"]
    for case in fixture["priority_role_cases"]:
        assert _priority_role(case["counterpart_ego_longitudinal_m"]) == case["expected"]
    for case in fixture["turn_pair_cases"]:
        observed = (
            _turn_label(case["ego_unwrapped_delta_degrees"])
            + "-"
            + _turn_label(case["counterpart_unwrapped_delta_degrees"])
        )
        assert observed == case["expected"]

    construction = _strict_load(FEATURE_FIXTURE_PATH)
    last = dict(zip(construction["history_columns"], construction["history_rows"][-1]))
    heading_delta = abs(math.degrees(last["counterpart_heading"] - last["ego_heading"]))
    category = _path_category(heading_delta)
    longitudinal = last["counterpart_x"] - last["ego_x"]
    lateral = last["counterpart_y"] - last["ego_y"]
    assert construction["categories"] == {
        "agent_type_pair": fixture["agent_constants"]["agent_type_pair"],
        "av_included": fixture["agent_constants"]["av_included"],
        "geometry_path_category": category,
        "geometry_path_relation": _path_relation(category, longitudinal, lateral),
        "priority_role": _priority_role(longitudinal),
        "turn_pair_label": "S-S",
        "vehicle_type_list": fixture["agent_constants"]["vehicle_type_list"],
    }


def test_canonical_320_cell_order_and_digest_are_reconstructed_from_axes() -> None:
    contract = _strict_load(CONTRACT_PATH)
    grid = contract["grid_contract"]
    golden = _strict_load(CELL_ORDER_FIXTURE_PATH)
    cell_ids = [
        f"RR3-{sampling['sampling_id']}-{temporal['temporal_id']}-{horizon['horizon_id']}-{readout}"
        for sampling in grid["sampling_axis"]
        for temporal in grid["temporal_families"]
        for horizon in grid["horizon_axis"]
        for readout in grid["readout_axis"]
    ]
    preimage = "".join(f"{cell_id}\n" for cell_id in cell_ids).encode("utf-8")
    assert grid["enumeration_order"] == golden["enumeration_order"]
    assert len(cell_ids) == grid["cell_count"] == golden["cell_count"] == 320
    assert cell_ids[0] == grid["first_cell_id"] == golden["first_cell_id"]
    assert cell_ids[-1] == grid["last_cell_id"] == golden["last_cell_id"]
    assert len(preimage) == grid["canonical_cell_id_payload_size_bytes"] == golden["canonical_cell_id_payload_size_bytes"]
    assert hashlib.sha256(preimage).hexdigest() == grid["canonical_cell_ids_sha256"] == golden["canonical_cell_ids_sha256"]


def test_blind_scene_predicate_requires_every_one_of_320_cells_without_ratings() -> None:
    contract = _strict_load(CONTRACT_PATH)
    golden = _strict_load(BLIND_FIXTURE_PATH)
    assert contract["blind_scene_predicate"]["rating_predicates"] == golden["rating_predicates"] == "NONE"
    for example in golden["examples"]:
        assert _blind_scene_eligible(example, golden["cell_count"]) is example["expected"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, (0.5, 1.5, 3.0)),
        (2.0, (0.0, 0.5, 1.0)),
        (3.0, (0.0, 0.0, 0.0)),
        (5.0, (0.0, 0.5, 2.0)),
        (9.0, (0.5, 1.5, 6.0)),
    ],
)
def test_nex_nmd_amd_ordinary_finite_interval(value: float, expected: tuple[float, float, float]) -> None:
    assert _deviations(value, 1.0, 3.0, 7.0) == expected


def test_d8_canonical_json_is_order_invariant_and_normalizes_signed_zero() -> None:
    expected = _strict_load(M3_INPUT_PATH)
    reordered = {
        "values": expected["values"],
        "schema_version": expected["schema_version"],
        "columns": expected["columns"],
    }
    assert _canonical_json_bytes(reordered) == M3_INPUT_PATH.read_bytes()
    assert _canonical_json_bytes({"z": -0.0, "a": 1}) == b'{"a":1,"z":0.0}\n'
    with pytest.raises(ValueError, match="Out of range float values"):
        _canonical_json_bytes({"value": math.nan})


def test_g2r_artifact_schemas_reject_rating_leaderboard_and_recovery_ledger_fields() -> None:
    shapes = _strict_load(SHAPES_PATH)["schemas"]
    data_shapes = (
        "anchor_score_row",
        "availability_mask_row",
        "blind_feature_row",
        "predictor_manifest_row",
    )
    for shape_id in data_shapes:
        exact_keys = shapes[shape_id]["exact_keys"]
        base = {key: None for key in exact_keys}
        for forbidden in ("rating", "leaderboard_rank", "recovery_ledger_sha256"):
            mutated = {**base, forbidden: "FORBIDDEN"}
            with pytest.raises(ValueError, match="extra="):
                _assert_exact_keys(mutated, exact_keys)

    output_schema = _strict_load(ROOT / shapes["output_manifest"]["schema_path"])
    artifact_properties = output_schema["properties"]["artifacts"]["properties"]
    assert "leaderboard" not in artifact_properties
    assert "recovery_ledger" not in artifact_properties
    forbidden_scan = output_schema["properties"]["forbidden_output_scan"]["properties"]
    assert {key: value["const"] for key, value in forbidden_scan.items()} == {
        "rating_field_count": 0,
        "leaderboard_file_count": 0,
        "recovery_ledger_file_count": 0,
    }
    operation_schema = _strict_load(ROOT / shapes["operation_receipt"]["schema_path"])
    assert operation_schema["properties"]["rating_access"]["const"] == "FORBIDDEN"
    assert operation_schema["properties"]["rating_join"]["const"] == "FORBIDDEN"
    assert operation_schema["properties"]["observed_rating_statistics"]["const"] == "FORBIDDEN"
    assert operation_schema["properties"]["rating_value_read_count"]["const"] == 0
    assert operation_schema["properties"]["leaderboard_row_count"]["const"] == 0
    assert operation_schema["properties"]["recovery_ledger_written"]["const"] is False


def test_g2r_operation_remains_centrally_denied_and_has_no_runnable_surface() -> None:
    contract = _strict_load(CONTRACT_PATH)
    assert contract["authority_status"] == "W1_OUTPUT_SCHEMA_FROZEN_OPERATION_DENIED"
    assert contract["future_operation_binding"] == {
        "batching": "OUT_OF_SCOPE_LATER_WAVE",
        "central_authorization": "DENY",
        "launcher": "OUT_OF_SCOPE_LATER_WAVE",
        "resource_profile": "OUT_OF_SCOPE_LATER_WAVE",
        "retry": "OUT_OF_SCOPE_LATER_WAVE",
        "run_spec": "OUT_OF_SCOPE_LATER_WAVE",
    }
    authorization = _strict_load(ROOT / "configs" / "research_authorization.json")
    assert "rq014_r2_blind_feature_build" not in authorization["authorizations"]["RQ014"]["allowed_operations"]
    assert not (ROOT / "scripts" / "rq014" / "run_g2r.py").exists()


def test_fixture_manifest_binds_construction_bytes_and_defers_scorer_goldens() -> None:
    manifest = _strict_load(FIXTURE_MANIFEST_PATH)
    for entry in manifest["construction_goldens"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    for entry in manifest["helper_bindings"].values():
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    deferred = manifest["scorer_dependent_bindings"]
    assert deferred["status"] == "PENDING_W3_SCORER"
    assert deferred["generated_scorer_output_file_count"] == 0
    assert deferred["local_builder_assessment"] == {
        "artifact_hash_and_size_match": True,
        "reviewed_managed_v4_closure": False,
        "version_compatible_python_stack": True,
    }
    assert deferred["required_environment"]["scikit_learn"] == "1.6.1"
    assert deferred["required_environment"]["joblib"] == "1.5.3"
    contract = _strict_load(CONTRACT_PATH)
    bindings = contract["fixture_bindings"]
    assert bindings["binding_status"] == "CONSTRUCTION_GOLDENS_BOUND_SCORER_GOLDENS_PENDING_W3"
    assert bindings["m3_pre_mask_golden"]["binding_status"] == "PENDING_W3_SCORER"
    assert {item["binding_status"] for item in bindings["nc_fixture_pairs"]} == {"PENDING_W3_SCORER"}
    bound_files = {
        "blind_scene_predicate_golden": BLIND_FIXTURE_PATH,
        "canonical_cell_order_golden": CELL_ORDER_FIXTURE_PATH,
        "geometry_path_crosstab_golden": GEOMETRY_FIXTURE_PATH,
        "m3_input_row_golden": M3_INPUT_PATH,
        "schema_shape_goldens": SHAPES_PATH,
        "wod_m3_feature_construction_golden": FEATURE_FIXTURE_PATH,
        "wod_to_m3_port_fixture_manifest": FIXTURE_MANIFEST_PATH,
    }
    for binding_id, path in bound_files.items():
        binding = bindings[binding_id]
        assert ROOT / binding["path"] == path
        assert binding["size_bytes"] == path.stat().st_size
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "PENDING_W1B" not in CONTRACT_PATH.read_text(encoding="utf-8")
    assert not (FIXTURE_ROOT / "m3_pre_mask_expected.json").exists()
    assert not any(FIXTURE_ROOT.glob("nc_*payload*.json"))


@pytest.mark.skip(
    reason=(
        "PENDING_W3_SCORER: A08/A15 pre-mask M3 outputs and NC payload hashes require "
        "the reviewed managed-v4 scikit-learn 1.6.1/joblib 1.5.3 scorer run"
    )
)
def test_m3_pre_mask_and_nc_payload_goldens_under_reviewed_v4_scorer() -> None:
    """W3 must replace this skip only after binding real scorer-produced bytes."""
