#!/usr/bin/env python3
"""Managed scene-level WOD reference builder for RQ014.

Byte provenance: constants and numerical statements are extracted from
``analyze_wod_e2e_ipv_rating_pilot.py`` (42,665 bytes, SHA-256
``7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2``),
source lines 35-37 and 82-142.  The managed preprocessing dependency replaces
the same-file ``traj_xy`` helper without changing its bytes-to-array behavior.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from scripts.rq014.wod_ipv_preprocessing import traj_xy


ROUTE_TURN_RADIUS_M = 12.0
ROUTE_EXTENSION_M = 80.0
ROUTE_STEP_M = 1.0


def infer_heading_direction(past_xy: np.ndarray) -> np.ndarray:
    """Infer the historical terminal heading direction from up to six points."""

    if len(past_xy) >= 2:
        for back_idx in range(max(0, len(past_xy) - 6), len(past_xy) - 1):
            delta = past_xy[-1] - past_xy[back_idx]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm
    return np.array([1.0, 0.0], dtype=float)


def build_ego_route_reference(scene: Dict[str, Any]) -> np.ndarray:
    """Construct the RQ010B section-6 map-free ego reference line."""

    past_xy = traj_xy(scene["past_states"])
    if len(past_xy) == 0:
        past_xy = np.zeros((1, 2), dtype=float)
    p0 = past_xy[-1]
    direction = infer_heading_direction(past_xy)
    heading = math.atan2(direction[1], direction[0])
    keep_every = max(1, len(past_xy) // 8)
    past_ref = past_xy[::keep_every]
    if np.linalg.norm(past_ref[-1] - p0) > 1e-9:
        past_ref = np.vstack([past_ref, p0])

    intent = str(scene.get("intent_name", "GO_STRAIGHT")).upper()
    if "LEFT" in intent:
        turn_sign = 1.0
    elif "RIGHT" in intent:
        turn_sign = -1.0
    else:
        turn_sign = 0.0

    if turn_sign == 0.0:
        s_values = np.arange(ROUTE_STEP_M, ROUTE_EXTENSION_M + ROUTE_STEP_M, ROUTE_STEP_M)
        forward = p0 + s_values[:, None] * direction[None, :]
    else:
        left_normal = np.array([-direction[1], direction[0]], dtype=float)
        center = p0 + turn_sign * ROUTE_TURN_RADIUS_M * left_normal
        theta0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
        arc_len = ROUTE_TURN_RADIUS_M * math.pi / 2.0
        arc_steps = max(8, int(math.ceil(arc_len / ROUTE_STEP_M)))
        phi = np.linspace(ROUTE_STEP_M / ROUTE_TURN_RADIUS_M, math.pi / 2.0, arc_steps)
        theta = theta0 + turn_sign * phi
        arc = center + ROUTE_TURN_RADIUS_M * np.column_stack([np.cos(theta), np.sin(theta)])
        end_heading = heading + turn_sign * math.pi / 2.0
        end_direction = np.array([math.cos(end_heading), math.sin(end_heading)], dtype=float)
        tail_len = max(0.0, ROUTE_EXTENSION_M - arc_len)
        if tail_len > 0.0:
            s_values = np.arange(ROUTE_STEP_M, tail_len + ROUTE_STEP_M, ROUTE_STEP_M)
            tail = arc[-1] + s_values[:, None] * end_direction[None, :]
            forward = np.vstack([arc, tail])
        else:
            forward = arc
    return np.vstack([past_ref, forward])
