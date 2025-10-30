"""
Utility helpers for estimating Interaction Preference Value (IPV) from paired
trajectory data.

The existing implementation embeds IPV inference in ``Agent.estimate_self_ipv``.
This module provides higher level wrappers so IPV estimation can be re-used
outside of the original notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from agent import Agent


@dataclass
class MotionSequence:
    """
    Container for a single agent's motion features.

    Attributes:
        data: Array with columns ``[x, y, vx, vy, heading]`` for each timestep.
        target: Target identifier expected by ``Agent`` (e.g., ``'lt_argo'``).
        reference: Optional reference polyline as ``(N, 2)`` array.
    """

    data: np.ndarray
    target: str
    reference: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.shape[1] < 5:
            raise ValueError(
                "MotionSequence.data must be a 2D array with columns [x, y, vx, vy, heading]."
            )


def estimate_ipv_pair(
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    history_window: int = 10,
    min_observation: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate IPV for two interacting agents across a trajectory window.

    The function mirrors the iterative logic previously embedded inside
    ``argoverse_process.ipynb`` while remaining agnostic to the data source.
    A fresh ``Agent`` instance is created at each timestep to avoid state
    leakage between windows.

    Args:
        primary: Motion (positions, velocities, headings) for the agent whose IPV
            is reported in the first column of the output.
        counterpart: Motion data for the interacting agent (second output column).
        history_window: Number of historical steps (exclusive of the current)
            considered when fitting each local IPV model.
        min_observation: Earliest timestep (0-indexed) at which estimation
            starts. Earlier rows are filled with ``np.nan``.

    Returns:
        ipv_values: Array ``(T, 2)`` holding IPV for ``primary`` and ``counterpart``.
        ipv_errors: Array ``(T, 2)`` with reliability metrics emitted by
            ``Agent.estimate_self_ipv``.
    """

    steps = min(len(primary.data), len(counterpart.data))
    if steps == 0:
        raise ValueError("Empty trajectories provided; cannot estimate IPV.")

    ipv_values = np.zeros((steps, 2), dtype=float)
    ipv_errors = np.ones((steps, 2), dtype=float)

    for t in range(min_observation, steps):
        start = max(0, t - history_window)

        # Primary agent perspective
        primary_agent = Agent(
            primary.data[start, 0:2],
            primary.data[start, 2:4],
            primary.data[start, 4],
            primary.target,
        )
        primary_agent.reference = primary.reference

        primary_track = primary.data[start : t + 1, 0:2]
        counterpart_track = counterpart.data[start : t + 1, 0:2]
        primary_agent.estimate_self_ipv(primary_track, counterpart_track)
        ipv_values[t, 0] = primary_agent.ipv
        ipv_errors[t, 0] = primary_agent.ipv_error

        # Counterpart perspective
        counterpart_agent = Agent(
            counterpart.data[start, 0:2],
            counterpart.data[start, 2:4],
            counterpart.data[start, 4],
            counterpart.target,
        )
        counterpart_agent.reference = counterpart.reference

        counterpart_agent.estimate_self_ipv(counterpart_track, primary_track)
        ipv_values[t, 1] = counterpart_agent.ipv
        ipv_errors[t, 1] = counterpart_agent.ipv_error

    return ipv_values, ipv_errors


def concat_motion(
    positions: np.ndarray,
    *,
    sample_time: float,
    heading_smoothing: bool = False,
) -> np.ndarray:
    """
    Construct ``[x, y, vx, vy, heading]`` arrays from position-only samples.

    This helper mirrors the velocity and heading reconstruction used in the
    original notebook for Argoverse 1 data.

    Args:
        positions: ``(T, 2)`` array of xy coordinates.
        sample_time: Sampling interval used to derive velocities.
        heading_smoothing: When ``True`` apply ``np.unwrap`` to headings to
            mitigate sudden jumps. Disabled by default to preserve parity with
            the legacy pipeline.

    Returns:
        ``(T-1, 5)`` array with reconstructed motion features.
    """

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be a (T, 2) array of xy coordinates.")
    if positions.shape[0] < 2:
        raise ValueError("At least two points are required to derive velocities.")
    if sample_time <= 0:
        raise ValueError("sample_time must be positive.")

    displacements = np.diff(positions, axis=0)
    velocities = displacements / sample_time
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    if heading_smoothing:
        headings = np.unwrap(headings)

    motion = np.column_stack((positions[:-1], velocities, headings))
    return motion
