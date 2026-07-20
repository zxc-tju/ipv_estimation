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
import pandas as pd

from scripts.rq014.wod_ipv_adapter import configure_ipv_estimator_timing
from scripts.rq014.wod_ipv_preprocessing import state_sequence_from_window_xy
from scripts.rq014.wod_reference_builder import build_ego_route_reference
from sociality_estimation.core.agent import resolve_ipv_candidate_values
from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_pair
from sociality_estimation.verifier.anchors import build_m3_anchor_features
from sociality_estimation.verifier.features import wrap_angle


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
G2R_OUTPUT_CONTRACT_PATH = Path("reports/plans/RQ014_g2r_output_contract_v1.json")
G2R_OUTPUT_CONTRACT_SHA256 = (
    "397585550d6a7f977cee0d5e8b82ba0c8756e5cc66b657bd03c93a1a49b4593d"
)
NC_FIXTURE_IDS = (
    "NC_HISTORY_BRANCH_R04N_W10",
    "NC_HISTORY_BRANCH_R04N_W25",
    "NC_HISTORY_BRANCH_R10L_W10",
    "NC_HISTORY_BRANCH_R10L_W25",
    "NC_HISTORY_FUTURE_PERTURBATION",
)
NC_IPV_FLOAT_KEYS = (
    "counterpart_ipv_error",
    "counterpart_ipv",
    "ego_ipv_error",
    "ego_ipv",
)
NC_IPV_ERROR_KEYS = ("counterpart_ipv_error", "ego_ipv_error")
NC_IPV_VALUE_KEYS = ("counterpart_ipv", "ego_ipv")
NC_EXACT_COMPONENT_HASH_KEYS = (
    "counterpart_reference_bytes_sha256",
    "focal_reference_bytes_sha256",
    "m3_context_bytes_sha256",
    "state_bytes_sha256",
)
NC_COMPONENT_HASH_KEYS = (
    *NC_EXACT_COMPONENT_HASH_KEYS,
    "ipv_bytes_sha256",
)
NC_PAYLOAD_TOPLEVEL_KEYS = frozenset(
    {
        *NC_COMPONENT_HASH_KEYS,
        "candidate_payload_sha256_by_ordinal",
        "fixture_id",
        "ipv_float_values",
        "schema_version",
        "terminal_status",
    }
)
NC_IPV_ERROR_ATOL = 1e-5
NC_IPV_ERROR_RTOL = 0.0
_NC_EXACT_IPV_CANDIDATES = np.asarray(
    resolve_ipv_candidate_values("exact"), dtype=float
)
NC_IPV_VALUE_MIN = float(np.min(_NC_EXACT_IPV_CANDIDATES))
NC_IPV_VALUE_MAX = float(np.max(_NC_EXACT_IPV_CANDIDATES))
_CATEGORY_ALLOWED_VALUES = {
    "geometry_path_category": frozenset({"F", "CP", "MP"}),
    "geometry_path_relation": frozenset({"F", "O-C", "C-C", "P-M", "P-P"}),
    "turn_pair_label": frozenset(
        f"{ego}-{counterpart}" for ego in ("L", "S", "R") for counterpart in ("L", "S", "R")
    ),
    "agent_type_pair": frozenset({"HV;HV"}),
    "vehicle_type_list": frozenset({"['HV', 'HV']"}),
    "av_included": frozenset({"all_HV"}),
    "priority_role": frozenset({"equal", "yield", "priority"}),
}


class WodM3KernelError(ValueError):
    """Fail-closed construction error in the rating-blind W2 kernel."""


class NCIPVNonfiniteError(WodM3KernelError):
    """A committed or observed NC IPV payload contains a non-finite float."""


class M3ScoringNumericalFailure(WodM3KernelError):
    """Typed A09/A10 terminal status for nonfinite M3 construction inputs."""

    status = "M3_SCORING_NUMERICAL_FAILURE"
    reason_code = "F_M3_SCORING_NUMERICAL_FAILURE"


def sha256_file(path: Path) -> str:
    """Hash one regular file after rejecting symlink substitution."""

    if path.is_symlink() or not path.is_file():
        raise WodM3KernelError(f"regular file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    finite = np.isfinite(matrix)
    if not np.all(finite[:, 11:13]):
        raise M3ScoringNumericalFailure(
            "nonfinite counterpart IPV/error in frozen trailing M3 context"
        )
    if not np.all(finite):
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


def build_scene_focal_reference(
    scene_history_xy: Sequence[Sequence[float]], *, route_intent: str
) -> np.ndarray:
    """Build the sole scene-level focal reference shared byte-for-byte by C1-C3."""

    xy = np.asarray(scene_history_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.all(np.isfinite(xy)):
        raise WodM3KernelError("scene focal history must be finite n-by-2")
    return build_ego_route_reference(
        {
            "intent_name": route_intent,
            "past_states": {"pos_x": xy[:, 0].tolist(), "pos_y": xy[:, 1].tolist()},
        }
    )


def build_m3_input_row_from_history(
    history_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    categories: Mapping[str, Any],
    *,
    case_start_timestamp_s: float,
) -> dict[str, Any]:
    """Build the exact 32-field D8 row from standardized causal history."""

    if set(categories) != set(CATEGORY_COLUMNS):
        raise WodM3KernelError("M3 categories require the exact seven frozen keys")
    for name in CATEGORY_COLUMNS:
        value = categories[name]
        if not isinstance(value, str) or value not in _CATEGORY_ALLOWED_VALUES[name]:
            raise WodM3KernelError(f"invalid frozen category token for {name}: {value!r}")
    history = _history_matrix(history_rows)
    assembled = build_m3_anchor_features(
        pd.DataFrame(history, columns=HISTORY_COLUMNS),
        categories,
        case_start_timestamp_s=float(case_start_timestamp_s),
    ).iloc[0]
    values: list[Any] = []
    for name in M3_INPUT_COLUMNS:
        value = assembled[name]
        if name in CATEGORY_COLUMNS:
            values.append(value)
        elif name == "history_row_count":
            values.append(int(value))
        elif math.isfinite(float(value)):
            values.append(float(value))
        elif name == "closing_ttc_anchor":
            values.append(_typed_na("F_M3_INPUT_TTC_UNDEFINED"))
        elif name == "apet_online_proxy":
            values.append(_typed_na("F_M3_INPUT_APET_UNDEFINED"))
        else:
            raise M3ScoringNumericalFailure(f"nonfinite numeric M3 input: {name}")
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
    focal_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the recovered H10/min-4 exact causal schedule for needed rows."""

    configure_ipv_estimator_timing(sample_dt_s)
    values = np.full(len(ego_xy), np.nan, dtype=float)
    errors = np.full(len(ego_xy), np.nan, dtype=float)
    first_needed = 4
    ego_reference = np.asarray(focal_reference, dtype=float)
    if (
        ego_reference.ndim != 2
        or ego_reference.shape[1] != 2
        or len(ego_reference) < 2
        or not np.all(np.isfinite(ego_reference))
    ):
        raise WodM3KernelError("scene-level focal reference must be finite n-by-2")
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
    scene_focal_reference: Sequence[Sequence[float]],
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
        ego_xy,
        counterpart_xy,
        timestamps,
        sample_dt_s=sample_dt_s,
        focal_reference=np.asarray(scene_focal_reference, dtype=float),
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
    if not np.all(np.isfinite(standardized[:, 11:13])):
        raise M3ScoringNumericalFailure(
            "nonfinite counterpart IPV/error in exact frozen tail-10 context"
        )
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
    scene_focal_reference: Sequence[Sequence[float]],
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
        scene_focal_reference=scene_focal_reference,
        counterpart_is_vehicle=counterpart_is_vehicle,
    )


def _float_hex_rows(rows: np.ndarray) -> list[list[str]]:
    return [[float(value).hex() for value in row] for row in np.asarray(rows)]


def _decode_xy(rows: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray([[float.fromhex(value) for value in row] for row in rows], dtype=float)


def _component_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _nc_ipv_payload(ipv_float_values: Mapping[str, Any]) -> dict[str, str]:
    """Return the byte-hash preimage for four finite binary64 IPV values."""

    if set(ipv_float_values) != set(NC_IPV_FLOAT_KEYS):
        raise WodM3KernelError("NC IPV float values exact-key drift")
    values: dict[str, float] = {}
    for key in NC_IPV_FLOAT_KEYS:
        value = ipv_float_values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WodM3KernelError(f"NC IPV float value is not numeric: {key}")
        values[key] = float(value)
        if not math.isfinite(values[key]):
            raise NCIPVNonfiniteError(f"NC IPV float value is nonfinite: {key}")
    return {
        "counterpart_ipv_error_hex": values["counterpart_ipv_error"].hex(),
        "counterpart_ipv_hex": values["counterpart_ipv"].hex(),
        "ego_ipv_error_hex": values["ego_ipv_error"].hex(),
        "ego_ipv_hex": values["ego_ipv"].hex(),
        "schema_version": "rq014-g2r-nc-ipv-v1",
    }


def _nc_candidate_payload_sha256(payload: Mapping[str, Any]) -> str:
    candidate_payload = {
        **{key: payload[key] for key in NC_COMPONENT_HASH_KEYS},
        "control_id": "NC_PRETSTAR_HISTORY_ONLY",
        "schema_version": "rq014-g2r-nc-candidate-payload-v1",
        "terminal_status": "AVAILABLE",
    }
    return _component_hash(candidate_payload)


def _nc_ipv_values_and_validate_hash(payload: Mapping[str, Any]) -> dict[str, float]:
    ipv_float_values = payload.get("ipv_float_values")
    if not isinstance(ipv_float_values, Mapping):
        raise WodM3KernelError("NC IPV float values must be an object")
    ipv_payload = _nc_ipv_payload(ipv_float_values)
    if payload.get("ipv_bytes_sha256") != _component_hash(ipv_payload):
        raise WodM3KernelError("NC IPV bytes hash is inconsistent with its float values")
    return {key: float(ipv_float_values[key]) for key in NC_IPV_FLOAT_KEYS}


def _nc_ipv_out_of_bounds_keys(ipv_float_values: Mapping[str, float]) -> list[str]:
    """Return IPV point estimates outside the exact solver candidate hull."""

    return [
        key
        for key in NC_IPV_VALUE_KEYS
        if not NC_IPV_VALUE_MIN <= ipv_float_values[key] <= NC_IPV_VALUE_MAX
    ]


def _validate_nc_candidate_ordinal_hashes(payload: Mapping[str, Any]) -> None:
    candidate_hashes = payload.get("candidate_payload_sha256_by_ordinal")
    if not isinstance(candidate_hashes, Mapping) or set(candidate_hashes) != {
        "1",
        "2",
        "3",
    }:
        raise WodM3KernelError("NC candidate ordinal hash keys drift")
    candidate_payload_sha256 = _nc_candidate_payload_sha256(payload)
    if any(value != candidate_payload_sha256 for value in candidate_hashes.values()):
        raise WodM3KernelError(
            "NC candidate ordinal hashes do not copy the in-process payload hash"
        )


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
        raise NCIPVNonfiniteError(
            "NC exact estimator did not produce a finite terminal pair"
        )
    ipv_float_values = {
        "counterpart_ipv_error": float(ipv_errors[-1, 1]),
        "counterpart_ipv": float(ipv_values[-1, 1]),
        "ego_ipv_error": float(ipv_errors[-1, 0]),
        "ego_ipv": float(ipv_values[-1, 0]),
    }
    ipv_payload = _nc_ipv_payload(ipv_float_values)
    component_hashes = {
        "counterpart_reference_bytes_sha256": _component_hash(counterpart_reference_payload),
        "focal_reference_bytes_sha256": _component_hash(focal_reference_payload),
        "ipv_bytes_sha256": _component_hash(ipv_payload),
        "m3_context_bytes_sha256": _component_hash(context_payload),
        "state_bytes_sha256": _component_hash(state_payload),
    }
    candidate_hash = _nc_candidate_payload_sha256(component_hashes)
    return {
        **component_hashes,
        "candidate_payload_sha256_by_ordinal": {
            str(ordinal): candidate_hash for ordinal in (1, 2, 3)
        },
        "fixture_id": fixture_id,
        "ipv_float_values": ipv_float_values,
        "schema_version": "rq014-g2r-nc-fixture-expected-v2",
        "terminal_status": "AVAILABLE",
    }


def _nc_observation(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": binding["fixture_id"],
        "input_path": binding["input_path"],
        "input_size_bytes": binding["input_size_bytes"],
        "input_sha256": binding["input_sha256"],
        "expected_path": binding["expected_path"],
        "expected_size_bytes": binding["expected_size_bytes"],
        "expected_sha256": binding["expected_sha256"],
        "observed_state_bytes_sha256": "NA",
        "observed_m3_context_bytes_sha256": "NA",
        "observed_focal_reference_bytes_sha256": "NA",
        "observed_counterpart_reference_bytes_sha256": "NA",
        "observed_ipv_bytes_sha256": "NA",
        "observed_payload_sha256": "NA",
        "status": "FAIL",
    }


def run_nc_pretstar_history_only_gate(
    *,
    repo_root: Path,
    python_executable_path: Path,
    python_executable_sha256: str,
    environment_manifest_path: Path,
    environment_manifest_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Verify committed fixtures, execute all five branches, and emit the W1 receipt."""

    repo_root = repo_root.resolve(strict=True)
    implementation = repo_root / "scripts/rq014/build_wod_m3_anchors.py"
    contract_path = repo_root / G2R_OUTPUT_CONTRACT_PATH
    estimator = repo_root / "src/sociality_estimation/core/ipv_estimation.py"
    bindings: list[Mapping[str, Any]] = []
    observations: list[dict[str, Any]] = []
    failure_class = "INPUT_CONTRACT_FAILURE"
    failure_fixture = "NA"
    failure_message = "NC gate did not initialize"
    results: list[dict[str, Any]] = []
    try:
        if sha256_file(contract_path) != G2R_OUTPUT_CONTRACT_SHA256:
            raise WodM3KernelError("G2R output contract SHA-256 drift")
        contract = _strict_load(contract_path)
        bindings = contract["fixture_bindings"]["nc_fixture_pairs"]
        if tuple(item.get("fixture_id") for item in bindings) != NC_FIXTURE_IDS:
            raise WodM3KernelError("NC fixture binding order or identity drift")
        observations = [_nc_observation(binding) for binding in bindings]
        estimator_binding = contract["source_bindings"]["ipv_estimation_core"]
        if (
            estimator_binding.get("path") != "src/sociality_estimation/core/ipv_estimation.py"
            or sha256_file(estimator) != estimator_binding.get("sha256")
        ):
            failure_class = "IMPLEMENTATION_HASH_MISMATCH"
            raise WodM3KernelError("frozen exact IPV estimator SHA-256 drift")
        for path, expected, label in (
            (python_executable_path, python_executable_sha256, "python executable"),
            (environment_manifest_path, environment_manifest_sha256, "environment manifest"),
        ):
            if sha256_file(path) != expected:
                failure_class = "IMPLEMENTATION_HASH_MISMATCH"
                raise WodM3KernelError(f"{label} SHA-256 drift")

        input_keys = {
            "base_candidate_futures", "base_ego_future", "counterpart_history_xy",
            "ego_history_xy", "fixture_id", "perturbed_candidate_futures",
            "perturbed_ego_future", "route_intent", "sample_dt_s", "sampling_id",
            "schema_version", "window_s",
        }
        for index, binding in enumerate(bindings):
            failure_fixture = binding["fixture_id"]
            if binding.get("binding_status") != "BOUND_SCORER_INDEPENDENT":
                raise WodM3KernelError("NC fixture is not a committed scorer-independent binding")
            input_path = repo_root / binding["input_path"]
            expected_path = repo_root / binding["expected_path"]
            for path, size_key, sha_key, label in (
                (input_path, "input_size_bytes", "input_sha256", "input"),
                (expected_path, "expected_size_bytes", "expected_sha256", "expected"),
            ):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != binding[size_key]
                    or sha256_file(path) != binding[sha_key]
                ):
                    failure_class = "FIXTURE_HASH_MISMATCH"
                    raise WodM3KernelError(f"committed NC {label} bytes drift")
            fixture = _strict_load(input_path)
            expected = _strict_load(expected_path)
            if set(fixture) != input_keys:
                failure_class = "INPUT_CONTRACT_FAILURE"
                raise WodM3KernelError("NC input fixture exact-key drift")
            expected_payload_keys = set(expected)
            if expected_payload_keys != NC_PAYLOAD_TOPLEVEL_KEYS:
                failure_class = "PAYLOAD_MISMATCH"
                missing = sorted(NC_PAYLOAD_TOPLEVEL_KEYS - expected_payload_keys)
                extra = sorted(expected_payload_keys - NC_PAYLOAD_TOPLEVEL_KEYS)
                raise WodM3KernelError(
                    "SCHEMA_KEY_DRIFT: golden "
                    f"missing={missing or ['NA']} extra={extra or ['NA']}"
                )
            if fixture["fixture_id"] != failure_fixture or expected["fixture_id"] != failure_fixture:
                raise WodM3KernelError("NC fixture identity differs from committed binding")
            if expected["schema_version"] != "rq014-g2r-nc-fixture-expected-v2":
                raise WodM3KernelError("NC expected fixture schema-version drift")
            try:
                expected_ipv_values = _nc_ipv_values_and_validate_hash(expected)
                _validate_nc_candidate_ordinal_hashes(expected)
            except NCIPVNonfiniteError as exc:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    f"IPV_NONFINITE: golden NC IPV payload is non-finite: {exc}"
                ) from exc
            except Exception as exc:
                failure_class = "INPUT_CONTRACT_FAILURE"
                raise WodM3KernelError(f"NC expected fixture is inconsistent: {exc}") from exc
            expected_out_of_bounds = _nc_ipv_out_of_bounds_keys(expected_ipv_values)
            if expected_out_of_bounds:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    "IPV_OUT_OF_BOUNDS: golden NC IPV value outside exact-solver "
                    "candidate hull: " + ",".join(expected_out_of_bounds)
                )
            try:
                observed = build_nc_history_only_payload(
                    fixture_id=failure_fixture,
                    sample_dt_s=float(fixture["sample_dt_s"]),
                    ego_history_xy=fixture["ego_history_xy"],
                    counterpart_history_xy=fixture["counterpart_history_xy"],
                    route_intent=str(fixture["route_intent"]),
                )
            except NCIPVNonfiniteError as exc:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    f"IPV_NONFINITE: observed NC exact estimator is non-finite: {exc}"
                ) from exc
            except Exception as exc:
                failure_class = "HISTORY_BRANCH_FAILURE"
                raise WodM3KernelError(f"NC history branch failed: {exc}") from exc
            schema_key_drift: list[str] = []
            for label, payload in (("observed", observed), ("golden", expected)):
                payload_keys = set(payload)
                if payload_keys != NC_PAYLOAD_TOPLEVEL_KEYS:
                    missing = sorted(NC_PAYLOAD_TOPLEVEL_KEYS - payload_keys)
                    extra = sorted(payload_keys - NC_PAYLOAD_TOPLEVEL_KEYS)
                    schema_key_drift.append(
                        f"{label} missing={missing or ['NA']} extra={extra or ['NA']}"
                    )
            if schema_key_drift:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    "SCHEMA_KEY_DRIFT: " + "; ".join(schema_key_drift)
                )
            try:
                observed_ipv_values = _nc_ipv_values_and_validate_hash(observed)
            except NCIPVNonfiniteError as exc:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    f"IPV_NONFINITE: observed NC IPV payload is non-finite: {exc}"
                ) from exc
            except Exception as exc:
                failure_class = "HISTORY_BRANCH_FAILURE"
                raise WodM3KernelError(f"NC observed IPV payload is inconsistent: {exc}") from exc
            observed_out_of_bounds = _nc_ipv_out_of_bounds_keys(observed_ipv_values)
            if observed_out_of_bounds:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    "IPV_OUT_OF_BOUNDS: observed NC IPV value outside exact-solver "
                    "candidate hull: " + ",".join(observed_out_of_bounds)
                )
            try:
                _validate_nc_candidate_ordinal_hashes(observed)
            except Exception as exc:
                failure_class = "CANDIDATE_OR_FUTURE_LEAKAGE"
                raise WodM3KernelError(f"NC ordinal-copy invariant failed: {exc}") from exc
            exact_keys = (
                *NC_EXACT_COMPONENT_HASH_KEYS,
                "fixture_id",
                "schema_version",
                "terminal_status",
            )
            exact_mismatches = [
                key for key in exact_keys if observed.get(key) != expected.get(key)
            ]
            if exact_mismatches:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    "BYTE_EXACT_COMPONENT_MISMATCH: " + ",".join(exact_mismatches)
                )
            # Point-estimate IPV values are non-anchored provenance: the exact
            # solver may choose different members of an equal-error solution set
            # across platforms.  Only finite, in-domain membership is required.
            try:
                np.testing.assert_allclose(
                    np.asarray(
                        [observed_ipv_values[key] for key in NC_IPV_ERROR_KEYS],
                        dtype=float,
                    ),
                    np.asarray(
                        [expected_ipv_values[key] for key in NC_IPV_ERROR_KEYS],
                        dtype=float,
                    ),
                    rtol=NC_IPV_ERROR_RTOL,
                    atol=NC_IPV_ERROR_ATOL,
                    equal_nan=False,
                )
            except AssertionError as exc:
                failure_class = "PAYLOAD_MISMATCH"
                raise WodM3KernelError(
                    "IPV_PORTABLE_PARITY_MISMATCH: observed IPV errors exceed "
                    "rtol=0.0, atol=1e-5"
                ) from exc
            candidate_hashes = observed["candidate_payload_sha256_by_ordinal"]
            observation = observations[index]
            observation.update(
                {
                    "observed_state_bytes_sha256": observed["state_bytes_sha256"],
                    "observed_m3_context_bytes_sha256": observed["m3_context_bytes_sha256"],
                    "observed_focal_reference_bytes_sha256": observed[
                        "focal_reference_bytes_sha256"
                    ],
                    "observed_counterpart_reference_bytes_sha256": observed[
                        "counterpart_reference_bytes_sha256"
                    ],
                    "observed_ipv_bytes_sha256": observed["ipv_bytes_sha256"],
                    "observed_payload_sha256": next(iter(candidate_hashes.values())),
                    "status": "PASS",
                }
            )
            results.append(observed)
        baseline = dict(results[3])
        future = dict(results[4])
        baseline.pop("fixture_id")
        future.pop("fixture_id")
        if baseline != future:
            observations[4]["status"] = "FAIL"
            failure_class = "CANDIDATE_OR_FUTURE_LEAKAGE"
            failure_fixture = NC_FIXTURE_IDS[4]
            raise WodM3KernelError("strictly post-tau perturbation changed NC bytes")
        failure = "NA"
        status = "PASS"
    except Exception as exc:
        if not observations:
            observations = [
                _nc_observation(
                    {
                        "fixture_id": fixture_id,
                        "input_path": "NA",
                        "input_size_bytes": 1,
                        "input_sha256": "0" * 64,
                        "expected_path": "NA",
                        "expected_size_bytes": 1,
                        "expected_sha256": "0" * 64,
                    }
                )
                for fixture_id in NC_FIXTURE_IDS
            ]
        failure_message = str(exc)
        failure = {
            "kind": "RUNTIME_FAILURE",
            "fixture_id_or_NA": failure_fixture,
            "failure_class": failure_class,
            "message": failure_message,
        }
        status = "FAIL"
    return {
        "schema_version": "rq014-nc-pretstar-history-only-receipt-v1",
        "control_id": "NC_PRETSTAR_HISTORY_ONLY",
        "status": status,
        "implementation_path": "scripts/rq014/build_wod_m3_anchors.py",
        "implementation_size_bytes": implementation.stat().st_size,
        "implementation_sha256": sha256_file(implementation),
        "estimator_sha256": sha256_file(estimator),
        "python_executable_sha256": python_executable_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "fixtures": observations,
        "failure_or_NA": failure,
        "created_at_utc": created_at_utc,
    }


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
