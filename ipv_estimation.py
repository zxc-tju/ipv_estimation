"""
Utility helpers for estimating Interaction Preference Value (IPV) from paired
trajectory data.

The existing implementation embeds IPV inference in ``Agent.estimate_self_ipv``.
This module provides higher level wrappers so IPV estimation can be re-used
outside of the original notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
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
    return_diagnostics: bool = False,
    diagnostic_steps: Optional[Sequence[int]] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
]:
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
        return_diagnostics: When ``True`` the function also returns the virtual
            trajectories and weights generated at selected timesteps.
        diagnostic_steps: Optional iterable of timestep indices to capture when
            ``return_diagnostics`` is enabled. If ``None`` every step starting
            from ``min_observation`` is recorded.

    Returns:
        ipv_values: Array ``(T, 2)`` holding IPV for ``primary`` and ``counterpart``.
        ipv_errors: Array ``(T, 2)`` with reliability metrics emitted by
            ``Agent.estimate_self_ipv``.
        diagnostics (optional): Dictionary containing per-step details suitable
            for debugging/visualisation. Present only when ``return_diagnostics``
            is ``True``.
    """

    steps = min(len(primary.data), len(counterpart.data))
    if steps == 0:
        raise ValueError("Empty trajectories provided; cannot estimate IPV.")

    ipv_values = np.zeros((steps, 2), dtype=float)
    ipv_errors = np.ones((steps, 2), dtype=float)

    diagnostics: Dict[str, List[Dict[str, np.ndarray]]]
    diagnostics = {"primary": [], "counterpart": []}
    diag_steps = set(diagnostic_steps) if diagnostic_steps is not None else None

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

        collect_primary = return_diagnostics and (diag_steps is None or t in diag_steps)
        if collect_primary:
            primary_details = primary_agent.estimate_self_ipv(
                primary_track, counterpart_track, return_details=True
            )
        else:
            primary_details = None
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

        collect_counter = return_diagnostics and (diag_steps is None or t in diag_steps)
        if collect_counter:
            counter_details = counterpart_agent.estimate_self_ipv(
                counterpart_track, primary_track, return_details=True
            )
        else:
            counter_details = None
            counterpart_agent.estimate_self_ipv(counterpart_track, primary_track)

        ipv_values[t, 1] = counterpart_agent.ipv
        ipv_errors[t, 1] = counterpart_agent.ipv_error

        if collect_primary and primary_details is not None:
            diagnostics["primary"].append(
                {
                    "step": t,
                    "start_index": start,
                    "observed": primary_track.copy(),
                    "interacting": counterpart_track.copy(),
                    "virtual_tracks": [track.copy() for track in primary_details["virtual_tracks"]],
                    "weights": primary_details["weights"].copy(),
                    "ipv_range": primary_details["ipv_range"].copy(),
                    "ipv": float(primary_agent.ipv),
                    "ipv_error": float(primary_agent.ipv_error),
                }
            )

        if collect_counter and counter_details is not None:
            diagnostics["counterpart"].append(
                {
                    "step": t,
                    "start_index": start,
                    "observed": counterpart_track.copy(),
                    "interacting": primary_track.copy(),
                    "virtual_tracks": [track.copy() for track in counter_details["virtual_tracks"]],
                    "weights": counter_details["weights"].copy(),
                    "ipv_range": counter_details["ipv_range"].copy(),
                    "ipv": float(counterpart_agent.ipv),
                    "ipv_error": float(counterpart_agent.ipv_error),
                }
            )

    if return_diagnostics:
        return ipv_values, ipv_errors, diagnostics
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


def plot_virtual_vs_observed(
    observed_track: np.ndarray,
    virtual_tracks: Sequence[np.ndarray],
    *,
    interacting_track: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    show: bool = True,
) -> plt.Axes:
    """
    Visualise the observed trajectory alongside the virtual trajectories used for IPV estimation.

    Args:
        observed_track: ``(N, 2)`` array containing the observed xy positions.
        virtual_tracks: Sequence of arrays with the same shape as ``observed_track``.
        interacting_track: Optional ``(N, 2)`` array for the interacting agent trajectory.
        weights: Optional reliability weights corresponding to ``virtual_tracks``; appended to legend.
        title: Optional plot title.
        ax: Optional Matplotlib axes to draw on. If ``None``, a new figure is created.
        cmap: Matplotlib colormap name for differentiating virtual tracks.
        show: Whether to call ``plt.show()`` when a new figure is created.

    Returns:
        The axes containing the plot (useful for further customisation).
    """

    if observed_track.ndim != 2 or observed_track.shape[1] != 2:
        raise ValueError("observed_track must have shape (N, 2).")
    for track in virtual_tracks:
        if track.ndim != 2 or track.shape[1] != 2:
            raise ValueError("Each virtual track must have shape (N, 2).")
        if track.shape[0] != observed_track.shape[0]:
            raise ValueError("Virtual and observed tracks must share the same number of points.")
    if interacting_track is not None:
        if interacting_track.ndim != 2 or interacting_track.shape[1] != 2:
            raise ValueError("interacting_track must have shape (N, 2).")
        if interacting_track.shape[0] != observed_track.shape[0]:
            raise ValueError("Interacting track must match observed track length.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        created_fig = True

    ax.plot(
        observed_track[:, 0],
        observed_track[:, 1],
        marker="o",
        color="#d62728",
        linewidth=2.0,
        label="observed",
    )

    if interacting_track is not None:
        ax.plot(
            interacting_track[:, 0],
            interacting_track[:, 1],
            marker="o",
            linestyle="--",
            color="#1f77b4",
            linewidth=1.5,
            label="interacting",
        )

    cmap_obj = plt.get_cmap(cmap)
    num_virtual = len(virtual_tracks)
    colors = [cmap_obj(i / max(num_virtual - 1, 1)) for i in range(num_virtual)]

    for idx, (track, color) in enumerate(zip(virtual_tracks, colors)):
        label = f"virtual #{idx}"
        if weights is not None and idx < len(weights):
            label += f" (w={weights[idx]:.3f})"
        ax.plot(
            track[:, 0],
            track[:, 1],
            linestyle="--",
            color=color,
            linewidth=1.0,
            alpha=0.8,
            label=label,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    if created_fig and show:
        plt.show()

    return ax
