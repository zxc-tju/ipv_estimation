#!/usr/bin/env python3
"""Managed WOD position-to-state preprocessing for RQ014.

Byte provenance: input extraction is taken from
``analyze_wod_e2e_ipv_rating_pilot.py`` (42,665 bytes, SHA-256
``7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2``),
source lines 40-52.  Its source-velocity shortcut, whole-trajectory gradient,
constant-velocity counterpart extrapolation, and pre-slice seam derivation are
not copied because the reviewed exact-window contract supersedes them.  The
finite-difference implementation below directly encodes
``RQ014_envelope_builder_contract_v2.json#/timeline_and_state_contract``.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def finite_float(value: Any, default: float = float("nan")) -> float:
    """Convert a finite scalar to float, otherwise return ``default``."""

    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def traj_xy(traj: Dict[str, Any]) -> np.ndarray:
    """Extract the common-length XY prefix from a historical trajectory."""

    x = np.asarray(traj.get("pos_x", []), dtype=float)
    y = np.asarray(traj.get("pos_y", []), dtype=float)
    n = min(len(x), len(y))
    return np.column_stack([x[:n], y[:n]]) if n else np.zeros((0, 2), dtype=float)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Apply the registered one-sided/centered operator within one window."""

    result = np.empty_like(values, dtype=float)
    result[0] = (values[1] - values[0]) / dt
    result[-1] = (values[-1] - values[-2]) / dt
    if len(values) > 2:
        result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return result


def derive_window_kinematics(xy: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Derive velocity, acceleration, and heading inside one closed XY window."""

    position = np.asarray(xy, dtype=float)
    if position.ndim != 2 or position.shape[1] != 2 or len(position) < 2:
        raise ValueError("window XY must have shape (n, 2) with n >= 2")
    if not math.isfinite(dt) or dt <= 0.0 or not np.all(np.isfinite(position)):
        raise ValueError("window XY and dt must be finite, with dt > 0")
    velocity = _finite_difference(position, dt)
    acceleration = _finite_difference(velocity, dt)
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.full(len(position), np.nan, dtype=float)
    moving = speed > 1.0e-9
    heading[moving] = np.arctan2(velocity[moving, 1], velocity[moving, 0])
    heading[heading == -math.pi] = math.pi
    if not np.any(moving):
        raise ValueError("all-stationary window has undefined heading")
    first = int(np.flatnonzero(moving)[0])
    heading[:first] = heading[first]
    for index in range(first + 1, len(heading)):
        if not math.isfinite(float(heading[index])):
            heading[index] = heading[index - 1]
    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "heading": heading,
    }


def state_sequence_from_window_xy(xy: np.ndarray, dt: float) -> np.ndarray:
    """Return estimator ``x,y,vx,vy,heading`` rows for one exact window."""

    derived = derive_window_kinematics(xy, dt)
    return np.column_stack(
        [derived["position"], derived["velocity"], derived["heading"]]
    )
