#!/usr/bin/env python3
"""Rating-blind WOD-to-RQ009-M3 feature and leakage-control kernel.

This module implements only the G2R W2 construction kernel.  It does not load
the M3 scorer, enumerate the 320 predictors, or expose a managed operation.
The formulas and row bytes are frozen by
``RQ014_g2r_output_contract_v1.json``; the trajectory-to-feature port follows
the recovered RQ012B OnSite builder while retaining the WOD window-local state
and exact-pair-estimator rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.rq014.wod_ipv_adapter import configure_ipv_estimator_timing
from scripts.rq014.wod_ipv_preprocessing import state_sequence_from_window_xy
from scripts.rq014.wod_reference_builder import build_ego_route_reference
from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_pair
from sociality_estimation.verifier.features import (
    apet_constant_velocity_proxy,
    closing_ttc,
    relative_state,
    theil_sen_slope,
    wrap_angle,
)


M3_INPUT_COLUMNS = (
    "elapsed_time_s",
    "history_row_count",
    "ego_vx_anchor",
    "ego_vy_anchor",
    "ego_heading_anchor",
    "counterpart_vx_anchor",
    "counterpart_vy_anchor",
    "counterpart_heading_anchor",
    "relative_dx_anchor",
    "relative_dy_anchor",
    "relative_distance_anchor",
    "relative_dvx_anchor",
    "relative_dvy_anchor",
    "relative_speed_anchor",
    "closing_rate_anchor",
    "heading_difference_anchor",
    "relative_distance_mean_wx",
    "relative_distance_std_wx",
    "relative_speed_mean_wx",
    "closing_rate_mean_wx",
    "closing_ttc_anchor",
    "apet_online_proxy",
    "counterpart_ipv_current",
    "counterpart_ipv_error_current",
    "counterpart_ipv_slope_pre_anchor",
    "geometry_path_category",
    "geometry_path_relation",
    "turn_pair_label",
    "agent_type_pair",
    "vehicle_type_list",
    "av_included",
    "priority_role",
)
HISTORY_COLUMNS = (
    "timestamp_s",
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "ego_heading",
    "counterpart_x",
    "counterpart_y",
    "counterpart_vx",
    "counterpart_vy",
    "counterpart_heading",
    "counterpart_ipv",
    "counterpart_ipv_error",
)
CATEGORY_COLUMNS = M3_INPUT_COLUMNS[25:]
TEMPORAL_FAMILY_IDS = (
    "CH-W10",
    "CH-W25",
    "LF-W10",
    "LF-W25",
    "HF-W10",
    "HF-W25",
    "TP",
    "TF",
)
ALIGNMENT_PRIMARY = "anchor_at_tau_history_only"
ALIGNMENT_SENSITIVITY = "terminal_minus_6_rows"


class WodM3KernelError(ValueError):
    """Fail-closed construction error in the rating-blind W2 kernel."""


def normalize_signed_zero(value: Any) -> Any:
    """Apply the D8 signed-zero rule recursively."""

    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [normalize_signed_zero(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_signed_zero(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_signed_zero(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one D8 canonical JSON object including its terminal LF."""

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


def temporal_window_bounds(
    temporal_id: str,
    tau_tick: int,
    rate_hz: int,
    h_common_tick: int,
) -> tuple[int, int]:
    """Return the exact closed tick bounds for all eight lane-v3 families."""

    if temporal_id == "CH-W10":
        return tau_tick - rate_hz, tau_tick
    if temporal_id == "CH-W25":
        return tau_tick - 5 * rate_hz // 2, tau_tick
    if temporal_id == "LF-W10":
        return tau_tick, tau_tick + rate_hz
    if temporal_id == "LF-W25":
        return tau_tick, tau_tick + 5 * rate_hz // 2
    if temporal_id == "HF-W10":
        return tau_tick - rate_hz, tau_tick + rate_hz
    if temporal_id == "HF-W25":
        return tau_tick - 5 * rate_hz // 2, tau_tick + 5 * rate_hz // 2
    if temporal_id == "TP":
        return 0, tau_tick
    if temporal_id == "TF":
        return 0, h_common_tick
    raise WodM3KernelError(f"unknown temporal family: {temporal_id}")


def resolve_m3_context_tick(
    *,
    alignment: str,
    tau_tick: int,
    temporal_id: str,
    rate_hz: int,
    h_common_tick: int,
) -> int:
    """Resolve the frozen family-specific history-only M3 context anchor."""

    if alignment == ALIGNMENT_PRIMARY:
        return tau_tick
    if alignment == ALIGNMENT_SENSITIVITY:
        _, terminal_tick = temporal_window_bounds(
            temporal_id, tau_tick, rate_hz, h_common_tick
        )
        return terminal_tick - 6
    raise WodM3KernelError(f"unknown m3_context_alignment: {alignment}")


def select_feature_family_context(
    ego_positions_by_tick: Mapping[int, Sequence[float]],
    counterpart_positions_by_tick: Mapping[int, Sequence[float]],
    *,
    temporal_id: str,
    tau_tick: int,
    rate_hz: int,
    h_common_tick: int,
    case_start_tick: int,
    alignment: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate one family window and select its separate causal M3 history.

    The lane-family window is checked independently from the M3 history.  The
    primary context always ends at tau; the sensitivity ends six rows before
    the selected temporal family's frozen terminal physical tick.
    """

    lower_tick, upper_tick = temporal_window_bounds(
        temporal_id, tau_tick, rate_hz, h_common_tick
    )
    family_ticks = range(lower_tick, upper_tick + 1)
    if any(
        tick not in ego_positions_by_tick or tick not in counterpart_positions_by_tick
        for tick in family_ticks
    ):
        raise WodM3KernelError("family position window leaves observed support")
    context_tick = resolve_m3_context_tick(
        alignment=alignment,
        tau_tick=tau_tick,
        temporal_id=temporal_id,
        rate_hz=rate_hz,
        h_common_tick=h_common_tick,
    )
    if context_tick < case_start_tick:
        raise WodM3KernelError("M3 context anchor precedes the explicit case-start tick")
    context_ticks = range(case_start_tick, context_tick + 1)
    if any(
        tick not in ego_positions_by_tick or tick not in counterpart_positions_by_tick
        for tick in context_ticks
    ):
        raise WodM3KernelError("causal M3 context leaves observed support")
    ego = np.asarray([ego_positions_by_tick[tick] for tick in context_ticks], dtype=float)
    counterpart = np.asarray(
        [counterpart_positions_by_tick[tick] for tick in context_ticks], dtype=float
    )
    return ego, counterpart, context_tick


def _typed_na(reason_code: str) -> dict[str, str]:
    return {"kind": "NA", "reason_code": reason_code}


def _history_matrix(history_rows: Sequence[Sequence[Any] | Mapping[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for item in history_rows:
        values = (
            [item[column] for column in HISTORY_COLUMNS]
            if isinstance(item, Mapping)
            else list(item)
        )
        if len(values) != len(HISTORY_COLUMNS):
            raise WodM3KernelError("standardized history must have exactly 13 columns")
        rows.append([float(value) for value in values])
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(HISTORY_COLUMNS) or len(matrix) < 4:
        raise WodM3KernelError("at least four standardized history rows are required")
    if not np.all(np.isfinite(matrix)):
        raise WodM3KernelError("standardized history contains nonfinite values")
    order = np.argsort(matrix[:, 0], kind="stable")
    matrix = matrix[order]
    if np.any(np.diff(matrix[:, 0]) <= 0.0):
        raise WodM3KernelError("history timestamps must be strictly increasing")
    return matrix[-10:]


def derive_categories(
    ego_state: np.ndarray,
    counterpart_state: np.ndarray,
    *,
    counterpart_is_vehicle: bool,
) -> dict[str, str]:
    """Derive the seven D5-D7 M3 categorical tokens from causal state."""

    if not counterpart_is_vehicle:
        raise WodM3KernelError("selected counterpart is not vehicle-class eligible")
    ego = np.asarray(ego_state, dtype=float)[-10:]
    counterpart = np.asarray(counterpart_state, dtype=float)[-10:]
    if ego.shape != counterpart.shape or ego.ndim != 2 or ego.shape[1] < 5:
        raise WodM3KernelError("category state histories must be aligned x,y,vx,vy,heading rows")
    heading_delta = wrap_angle(float(counterpart[-1, 4] - ego[-1, 4]))
    absolute_degrees = abs(math.degrees(heading_delta))
    if absolute_degrees < 25.0:
        category = "F"
    elif absolute_degrees > 135.0:
        category = "CP"
    else:
        category = "MP"

    dx = float(counterpart[-1, 0] - ego[-1, 0])
    dy = float(counterpart[-1, 1] - ego[-1, 1])
    heading = float(ego[-1, 4])
    longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    if category == "F":
        relation = "F"
    elif category == "CP":
        relation = "O-C" if abs(lateral) < 6.0 else "C-C"
    else:
        relation = "P-M" if abs(longitudinal) >= abs(lateral) else "P-P"

    def turn_label(headings: np.ndarray) -> str:
        delta_degrees = math.degrees(float(np.unwrap(headings)[-1] - np.unwrap(headings)[0]))
        if delta_degrees >= 12.0:
            return "L"
        if delta_degrees <= -12.0:
            return "R"
        return "S"

    if abs(longitudinal) < 2.0:
        priority = "equal"
    else:
        priority = "yield" if longitudinal > 0.0 else "priority"
    return {
        "geometry_path_category": category,
        "geometry_path_relation": relation,
        "turn_pair_label": f"{turn_label(ego[:, 4])}-{turn_label(counterpart[:, 4])}",
        "agent_type_pair": "HV;HV",
        "vehicle_type_list": "['HV', 'HV']",
        "av_included": "all_HV",
        "priority_role": priority,
    }


def build_m3_input_row_from_history(
    history_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    categories: Mapping[str, Any],
    *,
    case_start_timestamp_s: float,
) -> dict[str, Any]:
    """Build the exact 32-field D8 row from standardized causal history."""

    missing = [name for name in CATEGORY_COLUMNS if name not in categories]
    if missing:
        raise WodM3KernelError(f"missing category fields: {missing}")
    history = _history_matrix(history_rows)
    anchor = history[-1]
    relative = relative_state(
        history[:, 1], history[:, 2], history[:, 3], history[:, 4],
        history[:, 6], history[:, 7], history[:, 8], history[:, 9],
    )
    ego_position = anchor[[1, 2]]
    ego_velocity = anchor[[3, 4]]
    counterpart_position = anchor[[6, 7]]
    counterpart_velocity = anchor[[8, 9]]
    ttc = closing_ttc(float(relative["distance"][-1]), float(relative["closing_rate"][-1]))
    apet = apet_constant_velocity_proxy(
        ego_position, ego_velocity, counterpart_position, counterpart_velocity
    )
    values: list[Any] = [
        float(anchor[0] - float(case_start_timestamp_s)),
        int(len(history)),
        float(anchor[3]), float(anchor[4]), float(anchor[5]),
        float(anchor[8]), float(anchor[9]), float(anchor[10]),
        float(relative["dx"][-1]), float(relative["dy"][-1]),
        float(relative["distance"][-1]), float(relative["dvx"][-1]),
        float(relative["dvy"][-1]), float(relative["rel_speed"][-1]),
        float(relative["closing_rate"][-1]),
        wrap_angle(float(anchor[10] - anchor[5])),
        float(np.mean(relative["distance"])),
        float(np.std(relative["distance"], ddof=0)),
        float(np.mean(relative["rel_speed"])),
        float(np.mean(relative["closing_rate"])),
        float(ttc) if math.isfinite(float(ttc)) else _typed_na("F_M3_INPUT_TTC_UNDEFINED"),
        float(apet) if math.isfinite(float(apet)) else _typed_na("F_M3_INPUT_APET_UNDEFINED"),
        float(anchor[11]),
        float(anchor[12]),
        float(theil_sen_slope(history[-5:, 0], history[-5:, 11])),
        *[categories[name] for name in CATEGORY_COLUMNS],
    ]
    for index, value in enumerate(values[:25]):
        if isinstance(value, dict):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise WodM3KernelError(f"nonfinite numeric M3 input: {M3_INPUT_COLUMNS[index]}")
    payload = {
        "schema_version": "rq014-g2r-m3-input-row-v1",
        "columns": list(M3_INPUT_COLUMNS),
        "values": normalize_signed_zero(values),
    }
    canonical_json_bytes(payload)
    return payload


def m3_input_row_bytes_and_sha256(payload: Mapping[str, Any]) -> tuple[bytes, str]:
    """Return the frozen row preimage and its complete-byte SHA-256."""

    if tuple(payload) != ("schema_version", "columns", "values"):
        if set(payload) != {"schema_version", "columns", "values"}:
            raise WodM3KernelError("M3 row has an exact-key mismatch")
    if tuple(payload["columns"]) != M3_INPUT_COLUMNS:
        raise WodM3KernelError("M3 input column order drift")
    data = canonical_json_bytes(dict(payload))
    return data, hashlib.sha256(data).hexdigest()


def _estimate_counterpart_ipv_series(
    ego_xy: np.ndarray,
    counterpart_xy: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    sample_dt_s: float,
    route_intent: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the recovered H10/min-4 exact causal schedule for needed rows."""

    configure_ipv_estimator_timing(sample_dt_s)
    values = np.full(len(ego_xy), np.nan, dtype=float)
    errors = np.full(len(ego_xy), np.nan, dtype=float)
    first_needed = 4
    scene = {
        "intent_name": route_intent,
        "past_states": {"pos_x": ego_xy[:, 0].tolist(), "pos_y": ego_xy[:, 1].tolist()},
    }
    ego_reference = build_ego_route_reference(scene)
    for position in range(first_needed, len(ego_xy)):
        start = max(0, position - 10)
        ego_state = state_sequence_from_window_xy(ego_xy[start : position + 1], sample_dt_s)
        counterpart_state = state_sequence_from_window_xy(
            counterpart_xy[start : position + 1], sample_dt_s
        )
        ipv, error = estimate_ipv_pair(
            MotionSequence(ego_state, target="rq014_g2r_ego", reference=ego_reference),
            MotionSequence(
                counterpart_state,
                target="rq014_g2r_counterpart",
                reference=counterpart_state[:, :2],
            ),
            history_window=10,
            min_observation=4,
            solver_mode="exact",
            solver_options=None,
            max_workers=1,
        )
        values[position] = float(ipv[-1, 1])
        errors[position] = float(error[-1, 1])
    if not math.isfinite(values[-1]) or not math.isfinite(errors[-1]):
        raise WodM3KernelError("counterpart exact IPV estimator was nonfinite at context anchor")
    return values, errors


def build_wod_m3_input_row(
    ego_history_xy: Sequence[Sequence[float]],
    counterpart_history_xy: Sequence[Sequence[float]],
    *,
    sample_dt_s: float,
    case_start_timestamp_s: float,
    context_end_timestamp_s: float,
    route_intent: str,
    counterpart_is_vehicle: bool,
) -> dict[str, Any]:
    """Construct a WOD M3 row from one already-resampled causal position history."""

    ego_xy = np.asarray(ego_history_xy, dtype=float)
    counterpart_xy = np.asarray(counterpart_history_xy, dtype=float)
    if ego_xy.shape != counterpart_xy.shape or ego_xy.ndim != 2 or ego_xy.shape[1] != 2:
        raise WodM3KernelError("ego/counterpart position histories must be aligned n-by-2 arrays")
    if (
        len(ego_xy) < 5
        or not np.all(np.isfinite(ego_xy))
        or not np.all(np.isfinite(counterpart_xy))
    ):
        raise WodM3KernelError("WOD causal position histories need at least five finite rows")
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise WodM3KernelError("sample_dt_s must be positive and finite")
    timestamps = context_end_timestamp_s - np.arange(len(ego_xy) - 1, -1, -1) * sample_dt_s
    ipv, ipv_error = _estimate_counterpart_ipv_series(
        ego_xy, counterpart_xy, timestamps, sample_dt_s=sample_dt_s, route_intent=route_intent
    )
    context_start = max(0, len(ego_xy) - 10)
    context_ego = ego_xy[context_start:]
    context_counterpart = counterpart_xy[context_start:]
    ego_state = state_sequence_from_window_xy(context_ego, sample_dt_s)
    counterpart_state = state_sequence_from_window_xy(context_counterpart, sample_dt_s)
    categories = derive_categories(
        ego_state, counterpart_state, counterpart_is_vehicle=counterpart_is_vehicle
    )
    standardized = np.column_stack(
        [
            timestamps[context_start:],
            ego_state,
            counterpart_state,
            ipv[context_start:],
            ipv_error[context_start:],
        ]
    )
    finite_ipv = np.isfinite(standardized[:, 11]) & np.isfinite(standardized[:, 12])
    standardized = standardized[finite_ipv]
    if len(standardized) < 4:
        raise WodM3KernelError("fewer than four estimator-eligible M3 context rows")
    return build_m3_input_row_from_history(
        standardized, categories, case_start_timestamp_s=case_start_timestamp_s
    )


def build_feature_family_m3_input_row(
    ego_positions_by_tick: Mapping[int, Sequence[float]],
    counterpart_positions_by_tick: Mapping[int, Sequence[float]],
    *,
    temporal_id: str,
    tau_tick: int,
    rate_hz: int,
    h_common_tick: int,
    case_start_tick: int,
    route_intent: str,
    counterpart_is_vehicle: bool,
    m3_context_alignment: str = ALIGNMENT_PRIMARY,
) -> dict[str, Any]:
    """Build one candidate's M3 row for any of the eight temporal families."""

    ego, counterpart, context_tick = select_feature_family_context(
        ego_positions_by_tick,
        counterpart_positions_by_tick,
        temporal_id=temporal_id,
        tau_tick=tau_tick,
        rate_hz=rate_hz,
        h_common_tick=h_common_tick,
        case_start_tick=case_start_tick,
        alignment=m3_context_alignment,
    )
    return build_wod_m3_input_row(
        ego,
        counterpart,
        sample_dt_s=1.0 / rate_hz,
        case_start_timestamp_s=case_start_tick / rate_hz,
        context_end_timestamp_s=context_tick / rate_hz,
        route_intent=route_intent,
        counterpart_is_vehicle=counterpart_is_vehicle,
    )


def _float_hex_rows(rows: np.ndarray) -> list[list[str]]:
    return [[float(value).hex() for value in row] for row in np.asarray(rows)]


def _decode_xy(rows: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray([[float.fromhex(value) for value in row] for row in rows], dtype=float)


def _component_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_nc_history_only_payload(
    *,
    fixture_id: str,
    sample_dt_s: float,
    ego_history_xy: list[list[str]],
    counterpart_history_xy: list[list[str]],
    route_intent: str,
) -> dict[str, Any]:
    """Build one NC payload from pre-tau history only; futures are inaccessible."""

    sample_dt_s = float(sample_dt_s)
    ego_xy = _decode_xy(ego_history_xy)
    counterpart_xy = _decode_xy(counterpart_history_xy)
    if len(ego_xy) != len(counterpart_xy):
        raise WodM3KernelError("NC histories must have equal row counts")
    interval_count = len(ego_xy) - 1
    times = np.asarray(
        [(index - interval_count) * sample_dt_s for index in range(len(ego_xy))], dtype=float
    )
    ego_state = state_sequence_from_window_xy(ego_xy, sample_dt_s)
    counterpart_state = state_sequence_from_window_xy(counterpart_xy, sample_dt_s)
    state_payload = {
        "counterpart_state_hex": _float_hex_rows(np.column_stack([times, counterpart_state])),
        "ego_state_hex": _float_hex_rows(np.column_stack([times, ego_state])),
        "schema_version": "rq014-g2r-nc-state-v1",
    }
    context_payload = {
        "context_end_timestamp_s_hex": times[-1].hex(),
        "counterpart_state_hex": state_payload["counterpart_state_hex"][-10:],
        "ego_state_hex": state_payload["ego_state_hex"][-10:],
        "schema_version": "rq014-g2r-nc-m3-context-v1",
    }
    focal_reference = build_ego_route_reference(
        {
            "intent_name": route_intent,
            "past_states": {"pos_x": ego_xy[:, 0].tolist(), "pos_y": ego_xy[:, 1].tolist()},
        }
    )
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
    terminal = np.concatenate([ipv_values[-1], ipv_errors[-1]])
    if not np.all(np.isfinite(terminal)):
        raise WodM3KernelError("NC exact estimator did not produce a finite terminal pair")
    ipv_payload = {
        "counterpart_ipv_error_hex": float(ipv_errors[-1, 1]).hex(),
        "counterpart_ipv_hex": float(ipv_values[-1, 1]).hex(),
        "ego_ipv_error_hex": float(ipv_errors[-1, 0]).hex(),
        "ego_ipv_hex": float(ipv_values[-1, 0]).hex(),
        "schema_version": "rq014-g2r-nc-ipv-v1",
    }
    component_hashes = {
        "counterpart_reference_bytes_sha256": _component_hash(counterpart_reference_payload),
        "focal_reference_bytes_sha256": _component_hash(focal_reference_payload),
        "ipv_bytes_sha256": _component_hash(ipv_payload),
        "m3_context_bytes_sha256": _component_hash(context_payload),
        "state_bytes_sha256": _component_hash(state_payload),
    }
    candidate_payload = {
        **component_hashes,
        "control_id": "NC_PRETSTAR_HISTORY_ONLY",
        "schema_version": "rq014-g2r-nc-candidate-payload-v1",
        "terminal_status": "AVAILABLE",
    }
    candidate_hash = _component_hash(candidate_payload)
    return {
        **component_hashes,
        "candidate_payload_sha256_by_ordinal": {
            str(ordinal): candidate_hash for ordinal in (1, 2, 3)
        },
        "fixture_id": fixture_id,
        "schema_version": "rq014-g2r-nc-fixture-expected-v1",
        "terminal_status": "AVAILABLE",
    }


def run_nc_pretstar_history_only_gate(
    fixtures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute the four registered branches plus adversarial future fixture."""

    expected_ids = (
        "NC_HISTORY_BRANCH_R04N_W10",
        "NC_HISTORY_BRANCH_R04N_W25",
        "NC_HISTORY_BRANCH_R10L_W10",
        "NC_HISTORY_BRANCH_R10L_W25",
        "NC_HISTORY_FUTURE_PERTURBATION",
    )
    fixture_ids = [str(fixture["fixture_id"]) for fixture in fixtures]
    if len(fixtures) != len(expected_ids) or len(fixture_ids) != len(set(fixture_ids)):
        raise WodM3KernelError("NC gate requires five unique registered fixture IDs")
    by_id = {fixture_id: fixture for fixture_id, fixture in zip(fixture_ids, fixtures)}
    if tuple(sorted(by_id)) != tuple(sorted(expected_ids)):
        raise WodM3KernelError("NC gate requires the exact five registered fixture IDs")
    results = [
        build_nc_history_only_payload(
            fixture_id=fixture_id,
            sample_dt_s=float(by_id[fixture_id]["sample_dt_s"]),
            ego_history_xy=by_id[fixture_id]["ego_history_xy"],
            counterpart_history_xy=by_id[fixture_id]["counterpart_history_xy"],
            route_intent=str(by_id[fixture_id]["route_intent"]),
        )
        for fixture_id in expected_ids
    ]
    branch = results[3].copy()
    future = results[4].copy()
    branch.pop("fixture_id")
    future.pop("fixture_id")
    if branch != future:
        raise WodM3KernelError("FATAL_CANDIDATE_ID_OR_FUTURE_LEAKAGE")
    return results


def _strict_load(path: Path) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WodM3KernelError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            WodM3KernelError(f"nonfinite JSON token: {token}")
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    request = _strict_load(args.request_json)
    mode = request.get("mode")
    if mode == "standardized_history":
        payload = build_m3_input_row_from_history(
            request["history_rows"],
            request["categories"],
            case_start_timestamp_s=float(request["case_start_timestamp_s"]),
        )
    elif mode == "nc_history_only":
        payload = build_nc_history_only_payload(
            fixture_id=request["fixture_id"],
            sample_dt_s=request["sample_dt_s"],
            ego_history_xy=request["ego_history_xy"],
            counterpart_history_xy=request["counterpart_history_xy"],
            route_intent=request["route_intent"],
        )
    else:
        raise WodM3KernelError(f"unsupported request mode: {mode!r}")
    args.output_json.write_bytes(canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
