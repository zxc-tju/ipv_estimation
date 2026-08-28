"""Independent trajectory generator for the RQ027 feasibility pilot.

This module deliberately depends only on the standard library and NumPy.  It
does not import the production IPV estimator, planner, costs, likelihood, or
simulation state machine.  The only shared contract is the estimator-facing
motion schema ``[x, y, vx, vy, heading]`` and a 0.1 s sampling interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np


DT = 0.1
N_STEPS = 16
ORACLE_DELTA = math.pi / 16.0
ORACLE_STEPS = 10
OPPORTUNITY_HORIZON_S = 3.0
OPPORTUNITY_DISTANCE_M = 5.0
COLLISION_DISTANCE_M = 1.4
SAFETY_SHIELD_DISTANCE_M = 2.5

TARGET_IPV_LEVELS = tuple(
    multiplier * math.pi / 8.0 for multiplier in (-2.0, -1.5, 0.0, 1.5, 2.0)
)
COUNTERPART_IPV_LEVELS = tuple(
    multiplier * math.pi / 8.0 for multiplier in (-2.0, 2.0)
)
TEMPLATES = (
    "clear_priority_crossing",
    "ambiguous_priority_crossing",
    "merge",
    "same_direction_negotiation",
)
INTENSITIES = ("weak", "strong")
NEGATIVE_CONTROL_TYPES = (
    "no_conflict_neighbour",
    "time_shifted_counterpart",
    "wrong_run_pseudo_pair",
    "post_resolution_window",
)


@dataclass(frozen=True)
class GeneratedRun:
    """One estimator-compatible synthetic run and generator-side truth."""

    target_motion: np.ndarray
    counterpart_motion: np.ndarray
    target_reference: np.ndarray
    counterpart_reference: np.ndarray
    opportunity_mask: np.ndarray
    oracle_informativeness: np.ndarray
    collision_mask: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _Path:
    points: np.ndarray
    cumulative: np.ndarray

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])


@dataclass
class _ActorState:
    s: float
    speed: float
    acceleration: float = 0.0

    def copy(self) -> "_ActorState":
        return _ActorState(self.s, self.speed, self.acceleration)


def _dense_polyline(points: Sequence[Sequence[float]], samples: int = 161) -> np.ndarray:
    control = np.asarray(points, dtype=float)
    if control.ndim != 2 or control.shape[1] != 2 or len(control) < 2:
        raise ValueError("A path needs at least two two-dimensional control points")
    segment_lengths = np.linalg.norm(np.diff(control, axis=0), axis=1)
    if np.any(segment_lengths <= 0):
        raise ValueError("Path control points must be distinct")
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_s = np.linspace(0.0, cumulative[-1], samples)
    x = np.interp(sample_s, cumulative, control[:, 0])
    y = np.interp(sample_s, cumulative, control[:, 1])
    return np.column_stack([x, y])


def _make_path(points: Sequence[Sequence[float]]) -> _Path:
    dense = _dense_polyline(points)
    distance = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    return _Path(dense, np.concatenate([[0.0], np.cumsum(distance)]))


def _path_pose(path: _Path, s_value: float) -> tuple[np.ndarray, np.ndarray]:
    s_clipped = float(np.clip(s_value, 0.0, path.length))
    x = float(np.interp(s_clipped, path.cumulative, path.points[:, 0]))
    y = float(np.interp(s_clipped, path.cumulative, path.points[:, 1]))
    right = int(np.searchsorted(path.cumulative, s_clipped, side="right"))
    left = max(0, min(right - 1, len(path.points) - 2))
    tangent = path.points[left + 1] - path.points[left]
    norm = float(np.linalg.norm(tangent))
    if norm == 0:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / norm
    return np.array([x, y], dtype=float), tangent


def _motion_row(path: _Path, state: _ActorState) -> np.ndarray:
    position, tangent = _path_pose(path, state.s)
    velocity = tangent * state.speed
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    return np.array(
        [position[0], position[1], velocity[0], velocity[1], heading], dtype=float
    )


def _advance(state: _ActorState, acceleration: float, path: _Path) -> _ActorState:
    new_speed = float(np.clip(state.speed + acceleration * DT, 0.0, 7.0))
    new_s = min(path.length, state.s + 0.5 * (state.speed + new_speed) * DT)
    return _ActorState(new_s, new_speed, float(acceleration))


def _predict_positions(
    state: _ActorState,
    path: _Path,
    acceleration: float,
    horizon_steps: int,
) -> np.ndarray:
    predicted = []
    cursor = state.copy()
    for _ in range(horizon_steps):
        cursor = _advance(cursor, acceleration, path)
        predicted.append(_path_pose(path, cursor.s)[0])
    return np.asarray(predicted, dtype=float)


def _choose_acceleration(
    state: _ActorState,
    path: _Path,
    counterpart_state: _ActorState,
    counterpart_path: _Path,
    ipv_rad: float,
    intensity: str,
) -> float:
    """Choose from an independent discrete action bank.

    The objective has independently implemented progress, comfort and proximity
    terms.  It shares only the sign convention that positive IPV places more
    weight on the other road user.  No production cost or optimiser is called.
    """

    actions = (-2.5, -1.25, 0.0, 1.25)
    horizon = 25
    desired_speed = 4.2 if intensity == "strong" else 3.8
    other_positions = _predict_positions(
        counterpart_state,
        counterpart_path,
        counterpart_state.acceleration,
        horizon,
    )
    social_weight = math.sin(float(ipv_rad))
    own_weight = max(math.cos(float(ipv_rad)), 0.2)
    interaction_scale = 30.0 if intensity == "strong" else 15.0

    scored: list[tuple[float, float]] = []
    for acceleration in actions:
        own_positions = _predict_positions(state, path, acceleration, horizon)
        distances = np.linalg.norm(own_positions - other_positions, axis=1)
        min_distance = float(np.min(distances))
        proximity = float(np.mean(np.exp(-0.5 * (distances / 3.0) ** 2)))
        end_speed = float(
            np.clip(state.speed + acceleration * DT * horizon, 0.0, 7.0)
        )
        progress_cost = (desired_speed - end_speed) ** 2
        comfort_cost = 0.10 * acceleration**2 + 0.18 * (
            acceleration - state.acceleration
        ) ** 2
        objective = own_weight * (progress_cost + comfort_cost)
        # The independent generator plants the IPV trade-off through a
        # proximity-activated yielding action: positive IPV favours braking,
        # negative IPV favours maintaining/raising progress.  This is not the
        # production group-cost implementation.
        objective += social_weight * interaction_scale * proximity * acceleration
        objective += 2.0 * proximity
        if min_distance < SAFETY_SHIELD_DISTANCE_M:
            objective += 40.0 * (SAFETY_SHIELD_DISTANCE_M - min_distance) ** 2
        scored.append((objective, acceleration))
    costs = np.asarray([item[0] for item in scored], dtype=float)
    accelerations = np.asarray([item[1] for item in scored], dtype=float)
    temperature = 0.45
    logits = -(costs - float(np.min(costs))) / temperature
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    weights /= float(np.sum(weights))
    return float(np.sum(weights * accelerations))


def _constant_velocity_opportunity(target_row: np.ndarray, counterpart_row: np.ndarray) -> bool:
    steps = int(round(OPPORTUNITY_HORIZON_S / DT)) + 1
    times = np.arange(steps, dtype=float)[:, None] * DT
    target_positions = target_row[:2] + times * target_row[2:4]
    counterpart_positions = counterpart_row[:2] + times * counterpart_row[2:4]
    minimum = float(
        np.min(np.linalg.norm(target_positions - counterpart_positions, axis=1))
    )
    return minimum < OPPORTUNITY_DISTANCE_M


def _counterfactual_track(
    target_state: _ActorState,
    target_path: _Path,
    counterpart_state: _ActorState,
    counterpart_path: _Path,
    ipv_rad: float,
    intensity: str,
) -> np.ndarray:
    target_cursor = target_state.copy()
    counterpart_cursor = counterpart_state.copy()
    positions = []
    acceleration = _choose_acceleration(
        target_cursor,
        target_path,
        counterpart_cursor,
        counterpart_path,
        ipv_rad,
        intensity,
    )
    for _ in range(ORACLE_STEPS):
        target_cursor = _advance(target_cursor, acceleration, target_path)
        counterpart_cursor = _advance(
            counterpart_cursor, counterpart_cursor.acceleration, counterpart_path
        )
        positions.append(_path_pose(target_path, target_cursor.s)[0])
    return np.asarray(positions, dtype=float)


def _oracle_value(
    target_state: _ActorState,
    target_path: _Path,
    counterpart_state: _ActorState,
    counterpart_path: _Path,
    true_ipv_rad: float,
    intensity: str,
) -> float:
    lower = _counterfactual_track(
        target_state,
        target_path,
        counterpart_state,
        counterpart_path,
        true_ipv_rad - ORACLE_DELTA,
        intensity,
    )
    upper = _counterfactual_track(
        target_state,
        target_path,
        counterpart_state,
        counterpart_path,
        true_ipv_rad + ORACLE_DELTA,
        intensity,
    )
    rms = float(np.sqrt(np.mean(np.sum((upper - lower) ** 2, axis=1))))
    return rms / (2.0 * ORACLE_DELTA)


def _scenario_paths(template: str, intensity: str) -> tuple[_Path, _Path, float, float]:
    if template == "clear_priority_crossing":
        target = _make_path(((-6.0, 0.0), (14.0, 0.0)))
        counterpart_start = -8.5 if intensity == "weak" else -7.5
        counterpart = _make_path(((0.0, counterpart_start), (0.0, 14.0)))
        return target, counterpart, 4.2, 3.8
    if template == "ambiguous_priority_crossing":
        target = _make_path(((-6.5, 0.0), (14.0, 0.0)))
        counterpart_start = -7.2 if intensity == "weak" else -6.5
        counterpart = _make_path(((0.0, counterpart_start), (0.0, 14.0)))
        return target, counterpart, 4.0, 4.0
    if template == "merge":
        target = _make_path(((-6.0, 2.3), (-1.5, 0.5), (1.0, 0.0), (14.0, 0.0)))
        counterpart_start = -8.0 if intensity == "weak" else -6.8
        counterpart = _make_path(((counterpart_start, 0.0), (14.0, 0.0)))
        return target, counterpart, 4.0, 4.0
    if template == "same_direction_negotiation":
        target = _make_path(((-8.0, 0.0), (16.0, 0.0)))
        lead_start = -1.5 if intensity == "weak" else -3.5
        counterpart = _make_path(((lead_start, 0.0), (18.0, 0.0)))
        return target, counterpart, 4.6, 3.5
    raise ValueError(f"Unknown scenario template: {template}")


def _simulate(
    case: Dict[str, Any],
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, _Path, _Path]:
    template = str(case["template"])
    intensity = str(case["intensity"])
    target_path, counterpart_path, target_speed, counterpart_speed = _scenario_paths(
        template, intensity
    )
    rng = np.random.default_rng(int(case["seed"]))
    target_state = _ActorState(0.0, max(0.5, target_speed + rng.normal(0.0, 0.08)))
    counterpart_state = _ActorState(
        0.0, max(0.5, counterpart_speed + rng.normal(0.0, 0.08))
    )
    target_rows = []
    counterpart_rows = []
    oracle = []
    states: list[tuple[_ActorState, _ActorState]] = []

    for _ in range(steps):
        target_rows.append(_motion_row(target_path, target_state))
        counterpart_rows.append(_motion_row(counterpart_path, counterpart_state))
        states.append((target_state.copy(), counterpart_state.copy()))
        oracle.append(
            _oracle_value(
                target_state,
                target_path,
                counterpart_state,
                counterpart_path,
                float(case["true_ipv_rad"]),
                intensity,
            )
        )
        target_acceleration = _choose_acceleration(
            target_state,
            target_path,
            counterpart_state,
            counterpart_path,
            float(case["true_ipv_rad"]),
            intensity,
        )
        predicted_target = _advance(target_state, target_acceleration, target_path)
        counterpart_acceleration = _choose_acceleration(
            counterpart_state,
            counterpart_path,
            predicted_target,
            target_path,
            float(case["counterpart_ipv_rad"]),
            intensity,
        )

        predicted_counterpart = _advance(
            counterpart_state, counterpart_acceleration, counterpart_path
        )
        shield_target = _predict_positions(
            target_state, target_path, target_acceleration, 12
        )
        shield_counterpart = _predict_positions(
            counterpart_state, counterpart_path, counterpart_acceleration, 12
        )
        predicted_min_distance = float(
            np.min(np.linalg.norm(shield_target - shield_counterpart, axis=1))
        )
        if predicted_min_distance < SAFETY_SHIELD_DISTANCE_M:
            if float(case["true_ipv_rad"]) >= float(case["counterpart_ipv_rad"]):
                target_acceleration = -4.0
                predicted_target = _advance(target_state, target_acceleration, target_path)
            else:
                counterpart_acceleration = -4.0
                predicted_counterpart = _advance(
                    counterpart_state, counterpart_acceleration, counterpart_path
                )
            target_position = _path_pose(target_path, predicted_target.s)[0]
            counterpart_position = _path_pose(counterpart_path, predicted_counterpart.s)[0]
            if np.linalg.norm(target_position - counterpart_position) < COLLISION_DISTANCE_M:
                if float(case["true_ipv_rad"]) >= float(case["counterpart_ipv_rad"]):
                    predicted_target = _ActorState(
                        target_state.s,
                        max(0.0, target_state.speed - 4.0 * DT),
                        -4.0,
                    )
                else:
                    predicted_counterpart = _ActorState(
                        counterpart_state.s,
                        max(0.0, counterpart_state.speed - 4.0 * DT),
                        -4.0,
                    )
        target_state = predicted_target
        counterpart_state = predicted_counterpart

    return (
        np.asarray(target_rows, dtype=float),
        np.asarray(counterpart_rows, dtype=float),
        np.asarray(oracle, dtype=float),
        np.asarray(states, dtype=object),
        target_path,
        counterpart_path,
    )


def _recompute_masks(target_motion: np.ndarray, counterpart_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    opportunity = np.asarray(
        [
            _constant_velocity_opportunity(target_row, counterpart_row)
            for target_row, counterpart_row in zip(target_motion, counterpart_motion)
        ],
        dtype=bool,
    )
    collision = np.linalg.norm(target_motion[:, :2] - counterpart_motion[:, :2], axis=1) < COLLISION_DISTANCE_M
    return opportunity, collision


def _recompute_motion_kinematics(motion: np.ndarray) -> np.ndarray:
    updated = np.asarray(motion, dtype=float).copy()
    positions = updated[:, :2]
    velocity = np.gradient(positions, DT, axis=0, edge_order=1)
    updated[:, 2:4] = velocity
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    if len(heading) > 1:
        for index in range(1, len(heading)):
            if speed[index] < 1e-12:
                heading[index] = heading[index - 1]
    updated[:, 4] = heading
    return updated


def build_interactive_cases() -> list[Dict[str, Any]]:
    cases: list[Dict[str, Any]] = []
    for template_index, template in enumerate(TEMPLATES):
        for target_index, true_ipv_rad in enumerate(TARGET_IPV_LEVELS):
            for counterpart_index, counterpart_ipv_rad in enumerate(COUNTERPART_IPV_LEVELS):
                for intensity in INTENSITIES:
                    for seed_index in range(3):
                        run_id = (
                            f"interactive|{template}|t{target_index}|c{counterpart_index}|"
                            f"{intensity}|s{seed_index}"
                        )
                        cases.append(
                            {
                                "run_id": run_id,
                                "run_kind": "interactive",
                                "negative_control_type": "",
                                "template": template,
                                "true_ipv_rad": float(true_ipv_rad),
                                "true_ipv_source": (
                                    "on_grid" if target_index in (0, 2, 4) else "off_grid"
                                ),
                                "counterpart_ipv_rad": float(counterpart_ipv_rad),
                                "intensity": intensity,
                                "seed_index": seed_index,
                                "seed": (
                                    2026082800
                                    + template_index * 100
                                    + counterpart_index * 20
                                    + INTENSITIES.index(intensity) * 10
                                    + seed_index
                                ),
                            }
                        )
    if len(cases) != 240:
        raise AssertionError(f"Interactive matrix drifted: {len(cases)} != 240")
    return cases


def build_negative_control_cases() -> list[Dict[str, Any]]:
    cases: list[Dict[str, Any]] = []
    target_levels = (-2.0 * math.pi / 8.0, 0.0, 2.0 * math.pi / 8.0)
    for control_index, control_type in enumerate(NEGATIVE_CONTROL_TYPES):
        local_index = 0
        for true_ipv_rad in target_levels:
            for counterpart_ipv_rad in COUNTERPART_IPV_LEVELS:
                for seed_index in range(2):
                    template = TEMPLATES[(control_index + local_index) % len(TEMPLATES)]
                    cases.append(
                        {
                            "run_id": f"negative|{control_type}|{local_index:02d}",
                            "run_kind": "negative_control",
                            "negative_control_type": control_type,
                            "template": template,
                            "true_ipv_rad": float(true_ipv_rad),
                            "true_ipv_source": "control_truth_not_scored",
                            "counterpart_ipv_rad": float(counterpart_ipv_rad),
                            "intensity": "strong" if local_index % 2 else "weak",
                            "seed_index": seed_index,
                            "seed": 2026083800 + len(cases),
                        }
                    )
                    local_index += 1
    if len(cases) != 48:
        raise AssertionError(f"Negative-control matrix drifted: {len(cases)} != 48")
    return cases


def generate_run(case: Dict[str, Any]) -> GeneratedRun:
    """Generate one deterministic run without using estimator-side code."""

    control_type = str(case.get("negative_control_type", ""))
    if control_type == "post_resolution_window":
        target_all, counterpart_all, oracle_all, _, target_path, counterpart_path = _simulate(
            case, steps=N_STEPS + 28
        )
        target_motion = target_all[-N_STEPS:]
        counterpart_motion = counterpart_all[-N_STEPS:]
        oracle = np.zeros(N_STEPS, dtype=float)
    else:
        target_motion, counterpart_motion, oracle, _, target_path, counterpart_path = _simulate(
            case, steps=N_STEPS
        )

    counterpart_reference = counterpart_path.points.copy()
    if control_type == "no_conflict_neighbour":
        offset = np.array([9.0, 9.0])
        counterpart_motion = counterpart_motion.copy()
        counterpart_motion[:, :2] += offset
        counterpart_reference += offset
        oracle = np.zeros(N_STEPS, dtype=float)
    elif control_type == "time_shifted_counterpart":
        shift = 6
        counterpart_motion = np.vstack(
            [
                np.repeat(counterpart_motion[:1, :], shift, axis=0),
                counterpart_motion[:-shift],
            ]
        )
        counterpart_motion = _recompute_motion_kinematics(counterpart_motion)
        oracle = np.zeros(N_STEPS, dtype=float)
    elif control_type == "wrong_run_pseudo_pair":
        offset = np.array([-11.0, 8.0])
        counterpart_motion = counterpart_motion.copy()
        counterpart_motion[:, :2] += offset
        counterpart_reference += offset
        oracle = np.zeros(N_STEPS, dtype=float)
    elif control_type == "post_resolution_window":
        velocity = counterpart_motion[0, 2:4]
        norm = float(np.linalg.norm(velocity))
        direction = velocity / norm if norm > 1e-12 else np.array([1.0, 0.0])
        offset = direction * 10.0
        counterpart_motion = counterpart_motion.copy()
        counterpart_motion[:, :2] += offset
        counterpart_reference += offset
        oracle = np.zeros(N_STEPS, dtype=float)

    opportunity, collision = _recompute_masks(target_motion, counterpart_motion)
    metadata = dict(case)
    metadata.update(
        {
            "dt_s": DT,
            "n_steps": N_STEPS,
            "generator": "rq027_independent_discrete_rollout_v1",
            "generator_import_contract": "stdlib+numpy_only",
        }
    )
    return GeneratedRun(
        target_motion=np.asarray(target_motion, dtype=float),
        counterpart_motion=np.asarray(counterpart_motion, dtype=float),
        target_reference=target_path.points.copy(),
        counterpart_reference=counterpart_reference,
        opportunity_mask=opportunity,
        oracle_informativeness=np.asarray(oracle, dtype=float),
        collision_mask=collision,
        metadata=metadata,
    )


def all_cases() -> Iterable[Dict[str, Any]]:
    yield from build_interactive_cases()
    yield from build_negative_control_cases()
