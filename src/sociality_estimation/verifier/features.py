"""Shared causal feature formulas for the RQ009 M3 verifier contract."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


TTC_CAP_S = 20.0
APET_MAX_S = 20.0
APET_CONFLICT_RADIUS_M = 5.0


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def relative_state(
    ego_px: np.ndarray,
    ego_py: np.ndarray,
    ego_vx: np.ndarray,
    ego_vy: np.ndarray,
    counterpart_px: np.ndarray,
    counterpart_py: np.ndarray,
    counterpart_vx: np.ndarray,
    counterpart_vy: np.ndarray,
) -> Dict[str, np.ndarray]:
    dx = counterpart_px - ego_px
    dy = counterpart_py - ego_py
    dvx = counterpart_vx - ego_vx
    dvy = counterpart_vy - ego_vy
    distance = np.sqrt(dx * dx + dy * dy)
    relative_speed = np.sqrt(dvx * dvx + dvy * dvy)
    distance_dot = np.divide(
        dx * dvx + dy * dvy,
        distance,
        out=np.zeros_like(distance),
        where=distance > 1e-9,
    )
    return {
        "dx": dx,
        "dy": dy,
        "dvx": dvx,
        "dvy": dvy,
        "distance": distance,
        "rel_speed": relative_speed,
        "closing_rate": -distance_dot,
    }


def closing_ttc(distance: float, closing_rate: float) -> float:
    if (
        not np.isfinite(distance)
        or not np.isfinite(closing_rate)
        or closing_rate <= 1e-9
    ):
        return float("nan")
    return float(min(distance / closing_rate, TTC_CAP_S))


def apet_constant_velocity_proxy(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    counterpart_pos: np.ndarray,
    counterpart_vel: np.ndarray,
) -> float:
    matrix = np.column_stack([ego_vel, -counterpart_vel])
    offset = counterpart_pos - ego_pos
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        return float("nan")
    if np.linalg.norm(ego_vel) < 1e-9 or np.linalg.norm(counterpart_vel) < 1e-9:
        return float("nan")
    try:
        times, _, _, _ = np.linalg.lstsq(matrix, offset, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    ego_time = float(times[0])
    counterpart_time = float(times[1])
    if (
        ego_time < 0
        or counterpart_time < 0
        or ego_time > APET_MAX_S
        or counterpart_time > APET_MAX_S
    ):
        return float("nan")
    ego_conflict = ego_pos + ego_vel * ego_time
    counterpart_conflict = counterpart_pos + counterpart_vel * counterpart_time
    if np.linalg.norm(ego_conflict - counterpart_conflict) > APET_CONFLICT_RADIUS_M:
        return float("nan")
    return float(abs(ego_time - counterpart_time))


def theil_sen_slope(times_s: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(times_s) & np.isfinite(values)
    x = times_s[finite]
    y = values[finite]
    if len(x) < 2:
        return float("nan")
    slopes: List[float] = []
    for index in range(len(x) - 1):
        delta_x = x[index + 1 :] - x[index]
        delta_y = y[index + 1 :] - y[index]
        valid = np.abs(delta_x) > 1e-12
        slopes.extend((delta_y[valid] / delta_x[valid]).tolist())
    return float(np.median(np.asarray(slopes, dtype=float))) if slopes else float("nan")
