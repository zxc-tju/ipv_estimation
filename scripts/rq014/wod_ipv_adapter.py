#!/usr/bin/env python3
"""Managed WOD-to-IPV estimator adapter for the RQ014 recovery lane.

Byte provenance: this module derives the estimator-facing behavior from
``analyze_wod_e2e_ipv_rating_pilot.py`` (42,665 bytes, SHA-256
``7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2``):
imports at source lines 29-30 and functions at source lines 288-399.  Analysis,
rating, plotting, and report-writing code from that historical file is
intentionally excluded.  The historical short-window fallback is superseded:
the managed call below fixes exact mode and uses ``point_count-1`` for both
window arguments as required by the reviewed estimator-call contract.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from sociality_estimation.core import agent as agent_module
from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_pair


ORIGINAL_TRAJ_RELIABILITY = agent_module.cal_traj_reliability


def last_finite(values: np.ndarray) -> float:
    """Return the last finite array value, or NaN when none is finite."""

    arr = np.asarray(values, dtype=float)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return float("nan")
    return float(arr[np.where(ok)[0][-1]])


def configure_ipv_estimator_timing(sample_dt: float) -> None:
    """Align legacy IPV dynamics and likelihood numerics with WOD-E2E."""

    if not math.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError(f"sample_dt must be positive and finite, got {sample_dt!r}")
    agent_module.dt = float(sample_dt)
    agent_module.cal_traj_reliability = stable_traj_reliability


def stable_traj_reliability(
    inter_track: np.ndarray,
    act_track: np.ndarray,
    vir_track_coll: Sequence[np.ndarray],
    target: str,
) -> np.ndarray:
    """Compute the historical trajectory reliability in stable log space."""

    if np.size(inter_track) != 0:
        return ORIGINAL_TRAJ_RELIABILITY(inter_track, act_track, vir_track_coll, target)

    candidates_num = len(vir_track_coll)
    if candidates_num == 0:
        return np.zeros(0, dtype=float)

    act = np.asarray(act_track, dtype=float)
    sigma_value = float(agent_module.sigma)
    if sigma_value <= 0.0 or not math.isfinite(sigma_value):
        return np.ones(candidates_num, dtype=float) / candidates_num

    mean_loglike = np.full(candidates_num, -np.inf, dtype=float)
    for idx, virtual_track in enumerate(vir_track_coll):
        virtual = np.asarray(virtual_track, dtype=float)
        if virtual.shape != act.shape:
            n = min(len(virtual), len(act))
            if n == 0:
                continue
            virtual = virtual[:n]
            actual = act[:n]
        else:
            actual = act
        rel_dis = np.linalg.norm(virtual - actual, axis=1)
        if len(rel_dis) == 0 or not np.all(np.isfinite(rel_dis)):
            continue
        mean_loglike[idx] = float(np.mean(-(rel_dis**2) / (2.0 * sigma_value**2)))

    finite = np.isfinite(mean_loglike)
    if not np.any(finite):
        return np.ones(candidates_num, dtype=float) / candidates_num

    shifted = np.full_like(mean_loglike, -np.inf)
    shifted[finite] = mean_loglike[finite] - np.max(mean_loglike[finite])
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return np.ones(candidates_num, dtype=float) / candidates_num
    return weights / total


def estimate_ego_ipv(
    ego_state: np.ndarray,
    counterpart_state: np.ndarray,
    *,
    ego_reference: np.ndarray,
) -> Tuple[float, float]:
    """Adapt two WOD state sequences to the frozen pair estimator."""

    if len(ego_state) != len(counterpart_state):
        raise ValueError("focal and counterpart states must use one exact common window")
    steps = len(ego_state)
    if steps < 5:
        return float("nan"), float("nan")
    reference = np.asarray(ego_reference, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("explicit scene-level ego route reference is required")
    counterpart_reference = counterpart_state[:, 0:2]
    ipv_values, ipv_errors = estimate_ipv_pair(
        MotionSequence(ego_state, target="wod_e2e_ego", reference=reference),
        MotionSequence(
            counterpart_state,
            target="wod_e2e_counterpart",
            reference=counterpart_reference,
        ),
        history_window=steps - 1,
        min_observation=steps - 1,
        solver_mode="exact",
        solver_options=None,
        max_workers=1,
    )
    return last_finite(ipv_values[:, 0]), last_finite(ipv_errors[:, 0])
