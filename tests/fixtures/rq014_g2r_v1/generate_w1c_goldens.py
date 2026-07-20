#!/usr/bin/env python3
"""Regenerate scorer-independent RQ014 G2R W1c golden fixtures."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.rq014.wod_ipv_adapter import configure_ipv_estimator_timing  # noqa: E402
from scripts.rq014.wod_ipv_preprocessing import state_sequence_from_window_xy  # noqa: E402
from scripts.rq014.wod_reference_builder import build_ego_route_reference  # noqa: E402
from sociality_estimation.core.ipv_estimation import (  # noqa: E402
    MotionSequence,
    estimate_ipv_pair,
)


NC_CASES = (
    ("NC_HISTORY_BRANCH_R04N_W10", "R04N", 1.0, 0.25),
    ("NC_HISTORY_BRANCH_R04N_W25", "R04N", 2.5, 0.25),
    ("NC_HISTORY_BRANCH_R10L_W10", "R10L", 1.0, 0.1),
    ("NC_HISTORY_BRANCH_R10L_W25", "R10L", 2.5, 0.1),
    ("NC_HISTORY_FUTURE_PERTURBATION", "R10L", 2.5, 0.1),
)
NC_IPV_FLOAT_KEYS = (
    "counterpart_ipv_error",
    "counterpart_ipv",
    "ego_ipv_error",
    "ego_ipv",
)
NC_COMPONENT_HASH_KEYS = (
    "counterpart_reference_bytes_sha256",
    "focal_reference_bytes_sha256",
    "m3_context_bytes_sha256",
    "state_bytes_sha256",
    "ipv_bytes_sha256",
)


def normalize_signed_zero(value: Any) -> Any:
    """Apply the frozen D8 signed-zero rule recursively before serialization."""

    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [normalize_signed_zero(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_signed_zero(item) for item in value)
    if isinstance(value, dict):
        return {key: normalize_signed_zero(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the W1 D8 canonical JSON representation."""

    return (
        json.dumps(
            normalize_signed_zero(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _float_hex_rows(rows: np.ndarray) -> list[list[str]]:
    return [[float(value).hex() for value in row] for row in np.asarray(rows)]


def _decode_xy(rows: list[list[str]]) -> np.ndarray:
    return np.asarray([[float.fromhex(value) for value in row] for row in rows], dtype=float)


def _trajectory_rows(window_s: float, sample_dt_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interval_count = int(round(window_s / sample_dt_s))
    times = np.asarray(
        [(index - interval_count) * sample_dt_s for index in range(interval_count + 1)],
        dtype=float,
    )
    ego = np.column_stack([4.0 + times, np.zeros(len(times), dtype=float)])
    counterpart = np.column_stack([6.0 - times, np.zeros(len(times), dtype=float)])
    return times, ego, counterpart


def _future_rows(sample_dt_s: float, *, offset: float, lateral_scale: float) -> list[list[str]]:
    rows = []
    for index in range(1, 5):
        timestamp = index * sample_dt_s
        rows.append(
            [
                timestamp.hex(),
                (4.0 + timestamp + offset).hex(),
                (lateral_scale * index).hex(),
            ]
        )
    return rows


def nc_fixture_input(
    fixture_id: str,
    sampling_id: str,
    window_s: float,
    sample_dt_s: float,
) -> dict[str, Any]:
    _, ego, counterpart = _trajectory_rows(window_s, sample_dt_s)
    return {
        "base_candidate_futures": {
            candidate_id: _future_rows(
                sample_dt_s,
                offset=0.1 * ordinal,
                lateral_scale=0.01 * ordinal,
            )
            for ordinal, candidate_id in enumerate(("C1", "C2", "C3"), start=1)
        },
        "base_ego_future": _future_rows(sample_dt_s, offset=0.0, lateral_scale=0.0),
        "counterpart_history_xy": _float_hex_rows(counterpart),
        "ego_history_xy": _float_hex_rows(ego),
        "fixture_id": fixture_id,
        "perturbed_candidate_futures": {
            "C1": _future_rows(sample_dt_s, offset=25.0, lateral_scale=10.0),
            "C2": _future_rows(sample_dt_s, offset=-40.0, lateral_scale=-20.0),
            "C3": _future_rows(sample_dt_s, offset=75.0, lateral_scale=30.0),
        },
        "perturbed_ego_future": _future_rows(
            sample_dt_s,
            offset=-100.0,
            lateral_scale=50.0,
        ),
        "route_intent": "GO_STRAIGHT",
        "sample_dt_s": sample_dt_s,
        "sampling_id": sampling_id,
        "schema_version": "rq014-g2r-nc-fixture-input-v1",
        "window_s": window_s,
    }


def _component_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _nc_ipv_payload(ipv_float_values: dict[str, float]) -> dict[str, str]:
    return {
        "counterpart_ipv_error_hex": ipv_float_values["counterpart_ipv_error"].hex(),
        "counterpart_ipv_hex": ipv_float_values["counterpart_ipv"].hex(),
        "ego_ipv_error_hex": ipv_float_values["ego_ipv_error"].hex(),
        "ego_ipv_hex": ipv_float_values["ego_ipv"].hex(),
        "schema_version": "rq014-g2r-nc-ipv-v1",
    }


def _nc_candidate_payload_sha256(payload: dict[str, Any]) -> str:
    candidate_payload = {
        **{key: payload[key] for key in NC_COMPONENT_HASH_KEYS},
        "control_id": "NC_PRETSTAR_HISTORY_ONLY",
        "schema_version": "rq014-g2r-nc-candidate-payload-v1",
        "terminal_status": "AVAILABLE",
    }
    return _component_hash(candidate_payload)


def build_nc_history_only_payload(
    *,
    fixture_id: str,
    sample_dt_s: float,
    ego_history_xy: list[list[str]],
    counterpart_history_xy: list[list[str]],
    route_intent: str,
) -> dict[str, Any]:
    """Build only from the registered pre-tau arguments; futures are inaccessible."""

    sample_dt_s = float(sample_dt_s)
    ego_xy = _decode_xy(ego_history_xy)
    counterpart_xy = _decode_xy(counterpart_history_xy)
    if len(ego_xy) != len(counterpart_xy):
        raise ValueError("NC histories must have equal row counts")
    interval_count = len(ego_xy) - 1
    times = np.asarray(
        [(index - interval_count) * sample_dt_s for index in range(len(ego_xy))],
        dtype=float,
    )
    ego_state = state_sequence_from_window_xy(ego_xy, sample_dt_s)
    counterpart_state = state_sequence_from_window_xy(counterpart_xy, sample_dt_s)
    state_payload = {
        "counterpart_state_hex": _float_hex_rows(
            np.column_stack([times, counterpart_state])
        ),
        "ego_state_hex": _float_hex_rows(np.column_stack([times, ego_state])),
        "schema_version": "rq014-g2r-nc-state-v1",
    }
    context_payload = {
        "context_end_timestamp_s_hex": times[-1].hex(),
        "counterpart_state_hex": state_payload["counterpart_state_hex"][-10:],
        "ego_state_hex": state_payload["ego_state_hex"][-10:],
        "schema_version": "rq014-g2r-nc-m3-context-v1",
    }

    scene = {
        "intent_name": route_intent,
        "past_states": {
            "pos_x": ego_xy[:, 0].tolist(),
            "pos_y": ego_xy[:, 1].tolist(),
        },
    }
    focal_reference = build_ego_route_reference(scene)
    focal_reference_payload = {
        "reference_xy_hex": _float_hex_rows(focal_reference),
        "schema_version": "rq014-g2r-nc-focal-reference-v1",
    }
    counterpart_reference_payload = {
        "reference_xy_hex": _float_hex_rows(counterpart_state[:, :2]),
        "schema_version": "rq014-g2r-nc-counterpart-reference-v1",
    }

    configure_ipv_estimator_timing(sample_dt_s)
    ipv_values, ipv_errors = estimate_ipv_pair(
        MotionSequence(ego_state, target="rq014_nc_ego", reference=focal_reference),
        MotionSequence(
            counterpart_state,
            target="rq014_nc_counterpart",
            reference=counterpart_state[:, :2],
        ),
        history_window=len(ego_state) - 1,
        min_observation=len(ego_state) - 1,
        solver_mode="exact",
        solver_options=None,
        max_workers=1,
    )
    terminal_values = np.concatenate([ipv_values[-1], ipv_errors[-1]])
    if not np.all(np.isfinite(terminal_values)):
        raise ValueError("NC exact estimator did not produce a finite terminal pair")
    ipv_float_values = {
        "counterpart_ipv_error": float(ipv_errors[-1, 1]),
        "counterpart_ipv": float(ipv_values[-1, 1]),
        "ego_ipv_error": float(ipv_errors[-1, 0]),
        "ego_ipv": float(ipv_values[-1, 0]),
    }
    ipv_payload = _nc_ipv_payload(ipv_float_values)

    component_hashes = {
        "counterpart_reference_bytes_sha256": _component_hash(
            counterpart_reference_payload
        ),
        "focal_reference_bytes_sha256": _component_hash(focal_reference_payload),
        "ipv_bytes_sha256": _component_hash(ipv_payload),
        "m3_context_bytes_sha256": _component_hash(context_payload),
        "state_bytes_sha256": _component_hash(state_payload),
    }
    candidate_payload_sha256 = _nc_candidate_payload_sha256(component_hashes)
    return {
        **component_hashes,
        "candidate_payload_sha256_by_ordinal": {
            str(ordinal): candidate_payload_sha256 for ordinal in (1, 2, 3)
        },
        "fixture_id": fixture_id,
        "ipv_float_values": ipv_float_values,
        "schema_version": "rq014-g2r-nc-fixture-expected-v2",
        "terminal_status": "AVAILABLE",
    }


class M3ScoringNumericalFailure(ValueError):
    pass


def pointwise_deviations(
    value: float,
    lower: float,
    median: float,
    upper: float,
) -> tuple[float, float, float]:
    inputs = (value, lower, median, upper)
    if not all(math.isfinite(item) for item in inputs) or not lower < median < upper:
        raise M3ScoringNumericalFailure("invalid M3 interval input")
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
    result = (nex, nmd, abs(value - median))
    if not all(math.isfinite(item) for item in result):
        raise M3ScoringNumericalFailure("nonfinite derived deviation")
    return result


def trapezoid_readouts(
    times_s: Iterable[float],
    nex: Iterable[float],
    nmd: Iterable[float],
    amd: Iterable[float],
) -> dict[str, float]:
    times = tuple(float(value) for value in times_s)
    vectors = {
        "NEX": tuple(float(value) for value in nex),
        "NMD": tuple(float(value) for value in nmd),
        "AMD": tuple(float(value) for value in amd),
    }
    if len(times) < 2 or any(len(values) != len(times) for values in vectors.values()):
        raise M3ScoringNumericalFailure("readout needs at least two aligned anchors")
    if not all(math.isfinite(value) for value in times) or any(
        not all(math.isfinite(value) for value in values) for values in vectors.values()
    ):
        raise M3ScoringNumericalFailure("readout input is nonfinite")
    deltas = tuple(right - left for left, right in zip(times, times[1:]))
    if any(delta <= 0.0 for delta in deltas):
        raise M3ScoringNumericalFailure("readout times are not strictly increasing")
    duration = times[-1] - times[0]

    def trapezoid_mean(values: tuple[float, ...]) -> float:
        area = math.fsum(
            delta * (left + right) / 2.0
            for delta, left, right in zip(deltas, values, values[1:])
        )
        return area / duration

    result: dict[str, float] = {}
    for source_id, values in vectors.items():
        result[f"{source_id}_MEAN"] = trapezoid_mean(values)
        result[f"{source_id}_MAX"] = max(values)
        result[f"{source_id}_LAST"] = values[-1]
    outside = tuple(1.0 if value > 0.0 else 0.0 for value in vectors["NEX"])
    result["NEX_FOUT"] = trapezoid_mean(outside)
    return result


def _decode_token(token: str) -> float:
    special = {
        "NONFINITE_NAN": math.nan,
        "NONFINITE_POSITIVE_INFINITY": math.inf,
        "NONFINITE_NEGATIVE_INFINITY": -math.inf,
    }
    return special[token] if token in special else float.fromhex(token)


def deviations_readouts_fixture() -> dict[str, Any]:
    anchors = (
        (4.0, 0.0, 1.0, 3.0, 7.0),
        (5.0, 2.0, 1.0, 3.0, 7.0),
        (7.0, 5.0, 1.0, 3.0, 7.0),
        (8.0, 9.0, 1.0, 3.0, 7.0),
    )
    pointwise = [pointwise_deviations(*row[1:]) for row in anchors]
    readouts = trapezoid_readouts(
        [row[0] for row in anchors],
        [row[0] for row in pointwise],
        [row[1] for row in pointwise],
        [row[2] for row in pointwise],
    )
    invalid_inputs = (
        ("NONFINITE_VALUE", ("NONFINITE_NAN", 1.0.hex(), 3.0.hex(), 7.0.hex())),
        ("NONFINITE_LOWER", (2.0.hex(), "NONFINITE_NEGATIVE_INFINITY", 3.0.hex(), 7.0.hex())),
        ("LOWER_EQUALS_MEDIAN", (2.0.hex(), 3.0.hex(), 3.0.hex(), 7.0.hex())),
        ("LOWER_GREATER_THAN_MEDIAN", (2.0.hex(), 4.0.hex(), 3.0.hex(), 7.0.hex())),
        ("MEDIAN_EQUALS_UPPER", (5.0.hex(), 1.0.hex(), 7.0.hex(), 7.0.hex())),
        ("MEDIAN_GREATER_THAN_UPPER", (5.0.hex(), 1.0.hex(), 8.0.hex(), 7.0.hex())),
        (
            "STRICT_TINY_HALF_WIDTH_DERIVED_OVERFLOW",
            (1.0.hex(), (-math.ulp(0.0)).hex(), 0.0.hex(), math.ulp(0.0).hex()),
        ),
    )
    direct_pointwise_inputs = (
        ("VALUE_EQUALS_LOWER", (1.0, 1.0, 3.0, 7.0)),
        ("VALUE_EQUALS_MEDIAN", (3.0, 1.0, 3.0, 7.0)),
        ("VALUE_EQUALS_UPPER", (7.0, 1.0, 3.0, 7.0)),
        ("NONFINITE_MEDIAN", (2.0, 1.0, math.nan, 7.0)),
        ("NONFINITE_UPPER", (2.0, 1.0, 3.0, math.inf)),
    )

    direct_pointwise_cases = []
    for case_id, values in direct_pointwise_inputs:
        encoded_inputs = {
            key: (
                "NONFINITE_NAN"
                if math.isnan(value)
                else "NONFINITE_POSITIVE_INFINITY"
                if value == math.inf
                else value.hex()
            )
            for key, value in zip(("value", "lower", "median", "upper"), values)
        }
        try:
            deviations = pointwise_deviations(*values)
        except M3ScoringNumericalFailure:
            expected_deviations = "NA"
            expected_reason_code = "F_M3_SCORING_NUMERICAL_FAILURE"
            expected_status = "M3_SCORING_NUMERICAL_FAILURE"
        else:
            expected_deviations = {
                key: value.hex()
                for key, value in zip(("nex", "nmd", "amd"), deviations)
            }
            expected_reason_code = "F_AVAILABLE_CONTINUE"
            expected_status = "AVAILABLE"
        direct_pointwise_cases.append(
            {
                "case_id": case_id,
                "expected_deviations_hex_or_NA": expected_deviations,
                "expected_reason_code": expected_reason_code,
                "expected_status": expected_status,
                "inputs_hex_or_token": encoded_inputs,
            }
        )
    return {
        "direct_pointwise_cases": direct_pointwise_cases,
        "degenerate_readout_cases": [
            {
                "case_id": "SINGLE_ANCHOR_ZERO_DURATION",
                "expected_reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "expected_status": "M3_SCORING_NUMERICAL_FAILURE",
                "times_s_hex": [0.0.hex()],
            },
            {
                "case_id": "DUPLICATE_ANCHOR_TIME",
                "expected_reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "expected_status": "M3_SCORING_NUMERICAL_FAILURE",
                "times_s_hex": [0.0.hex(), 0.0.hex()],
            },
        ],
        "invalid_interval_cases": [
            {
                "case_id": case_id,
                "expected_reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "expected_status": "M3_SCORING_NUMERICAL_FAILURE",
                "inputs_hex_or_token": {
                    "lower": values[1],
                    "median": values[2],
                    "upper": values[3],
                    "value": values[0],
                },
            }
            for case_id, values in invalid_inputs
        ],
        "ordinary_case": {
            "anchors": [
                {
                    "amd_hex": pointwise_value[2].hex(),
                    "lower_hex": row[2].hex(),
                    "median_hex": row[3].hex(),
                    "nex_hex": pointwise_value[0].hex(),
                    "nmd_hex": pointwise_value[1].hex(),
                    "tau_s_hex": row[0].hex(),
                    "upper_hex": row[4].hex(),
                    "value_hex": row[1].hex(),
                }
                for row, pointwise_value in zip(anchors, pointwise)
            ],
            "expected_readouts_hex": {
                key: value.hex() for key, value in sorted(readouts.items())
            },
        },
        "readout_order": [
            "NEX_MEAN",
            "NEX_MAX",
            "NEX_LAST",
            "NEX_FOUT",
            "NMD_MEAN",
            "NMD_MAX",
            "NMD_LAST",
            "AMD_MEAN",
            "AMD_MAX",
            "AMD_LAST",
        ],
        "schema_version": "rq014-g2r-deviations-readouts-golden-v1",
    }


def availability_statuses_fixture(contract: dict[str, Any]) -> dict[str, Any]:
    statuses = contract["status_contract"]
    return {
        "candidate_precedence_case": {
            "candidate_rows": [
                {
                    "candidate_ordinal": 1,
                    "reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                    "reason_priority": 61,
                    "status": "M3_SCORING_NUMERICAL_FAILURE",
                },
                {
                    "candidate_ordinal": 2,
                    "reason_code": "F_TIMELINE_SOURCE_GAP",
                    "reason_priority": 41,
                    "status": "INELIGIBLE_TIMELINE_SOURCE_GAP",
                },
                {
                    "candidate_ordinal": 3,
                    "reason_code": "F_TIMELINE_SUPPORT",
                    "reason_priority": 40,
                    "status": "INELIGIBLE_TIMELINE_SUPPORT",
                },
            ],
            "expected_reason_code": "F_TIMELINE_SUPPORT",
            "expected_status": "INELIGIBLE_TIMELINE_SUPPORT",
        },
        "candidate_status_rows": [
            {
                **row,
                "available": row["status"] == "AVAILABLE",
                "predictor_finite": row["status"] == "AVAILABLE",
            }
            for row in statuses["candidate_upstream_statuses"]
        ],
        "global_fatal_rows": [
            {
                **row,
                "cell_fatal_upstream_status_or_NA": row["status"],
                "cell_terminal_status": "CELL_FATAL",
                "global_abort": True,
            }
            for row in statuses["global_fatal_status_reason_mappings"]
        ],
        "m3_numerical_failure_propagation": {
            "candidate_rows": [
                {
                    "available": True,
                    "candidate_id": "C1",
                    "candidate_ordinal": 1,
                    "predictor_finite": True,
                    "reason_code": "F_AVAILABLE_CONTINUE",
                    "status": "AVAILABLE",
                },
                {
                    "available": False,
                    "candidate_id": "C2",
                    "candidate_ordinal": 2,
                    "predictor_finite": False,
                    "reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                    "status": "M3_SCORING_NUMERICAL_FAILURE",
                },
                {
                    "available": True,
                    "candidate_id": "C3",
                    "candidate_ordinal": 3,
                    "predictor_finite": True,
                    "reason_code": "F_AVAILABLE_CONTINUE",
                    "status": "AVAILABLE",
                },
            ],
            "expected_cell": {
                "cell_fatal_upstream_status_or_NA": "NA",
                "cell_terminal_status": "TERMINAL",
                "global_abort": False,
                "reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
            },
            "expected_scene_cell": {
                "all_three_available": False,
                "all_three_deviations_finite": False,
                "blind_cell_scene_eligible": False,
                "deviation_vector_nonconstant": False,
                "reason_code": "F_M3_SCORING_NUMERICAL_FAILURE",
                "scene_cell_status": "M3_SCORING_NUMERICAL_FAILURE",
            },
        },
        "schema_version": "rq014-g2r-availability-statuses-golden-v1",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _binding(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    data = path.read_bytes()
    return {
        "path": relative,
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def main() -> None:
    contract_path = ROOT / "reports/plans/RQ014_g2r_output_contract_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    generated: list[Path] = []
    for fixture_id, sampling_id, window_s, sample_dt_s in NC_CASES:
        stem = fixture_id.lower()
        input_path = FIXTURE_ROOT / f"{stem}_input.json"
        expected_path = FIXTURE_ROOT / f"{stem}_expected.json"
        payload = nc_fixture_input(fixture_id, sampling_id, window_s, sample_dt_s)
        _write_json(input_path, payload)
        _write_json(
            expected_path,
            build_nc_history_only_payload(
                fixture_id=payload["fixture_id"],
                sample_dt_s=payload["sample_dt_s"],
                ego_history_xy=payload["ego_history_xy"],
                counterpart_history_xy=payload["counterpart_history_xy"],
                route_intent=payload["route_intent"],
            ),
        )
        generated.extend((input_path, expected_path))

    readout_path = FIXTURE_ROOT / "deviations_readouts_v1.json"
    status_path = FIXTURE_ROOT / "availability_statuses_v1.json"
    _write_json(readout_path, deviations_readouts_fixture())
    _write_json(status_path, availability_statuses_fixture(contract))
    generated.extend((readout_path, status_path))

    manifest_path = FIXTURE_ROOT / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonicalization"] = (
        "recursively normalize signed zero to 0.0, then CPython 3.9 "
        "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,"
        "allow_nan=False), UTF-8, exactly one terminal LF"
    )
    existing = {
        Path(entry["path"]).name: entry
        for entry in manifest["construction_goldens"]
        if Path(entry["path"]).name not in {path.name for path in generated}
    }
    for path in generated:
        existing[path.name] = _binding(path)
    manifest["construction_goldens"] = [existing[name] for name in sorted(existing)]
    manifest["helper_bindings"]["w1c_golden_generator"] = _binding(Path(__file__))
    scorer = manifest["scorer_dependent_bindings"]
    scorer["deferred_artifacts"] = ["m3_pre_mask_expected.json (A08/A15)"]
    scorer["reason"] = (
        "only A08/A15 pre-mask q_0p5/lo_90/hi_90 bytes remain deferred to the "
        "reviewed managed-v4 scorer run; NC, deviation/readout, and A09/A10 "
        "fixtures are scorer-independent and bound in W1c"
    )
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
