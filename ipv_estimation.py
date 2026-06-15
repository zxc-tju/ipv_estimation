"""
Utility helpers for estimating Interaction Preference Value (IPV) from paired
trajectory data.

The existing implementation embeds IPV inference in ``Agent.estimate_self_ipv``.
This module provides higher level wrappers so IPV estimation can be re-used
outside of the original notebooks.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import agent as agent_module
from agent import Agent

SOLVER_PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    "accurate": None,
    "parallel_accurate": None,
    "balanced": {"maxiter": 20, "ftol": 1e-3},
    "realtime": {"maxiter": 8, "ftol": 1e-2},
}

SIGN_REALTIME_CANDIDATE_IPV_VALUES = np.array(
    [-3, -1, 0, 1, 3],
    dtype=float,
) * np.pi / 8


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


def _prepare_reference_for_repeated_use(reference):
    if reference is None:
        return None
    if isinstance(reference, tuple) and len(reference) == 2:
        return reference
    return agent_module.smooth_ployline(reference)


def _resolve_solver_options(
    solver_preset: str,
    solver_options: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if solver_preset not in SOLVER_PRESETS:
        allowed = ", ".join(sorted(SOLVER_PRESETS))
        raise ValueError(
            f"Unknown solver_preset {solver_preset!r}; expected one of: {allowed}."
        )

    preset_options = SOLVER_PRESETS[solver_preset]
    resolved = {} if preset_options is None else dict(preset_options)
    if solver_options:
        resolved.update(solver_options)
    return resolved or None


def _estimate_agent_ipv(
    agent: Agent,
    self_track: np.ndarray,
    inter_track: np.ndarray,
    *,
    return_details: bool,
    solver_options: Optional[Dict[str, float]],
    candidate_executor=None,
    candidate_ipv_values=None,
):
    kwargs = {
        "return_details": return_details,
    }
    if solver_options is None:
        pass
    else:
        kwargs["solver_options"] = solver_options
    if candidate_executor is not None:
        kwargs["candidate_executor"] = candidate_executor
    if candidate_ipv_values is not None:
        kwargs["candidate_ipv_values"] = candidate_ipv_values
    return agent.estimate_self_ipv(
        self_track,
        inter_track,
        **kwargs,
    )


def _estimate_agent_pair_ipv_parallel(
    primary_agent: Agent,
    counterpart_agent: Agent,
    primary_track: np.ndarray,
    counterpart_track: np.ndarray,
    *,
    return_primary_details: bool,
    return_counterpart_details: bool,
    solver_options: Optional[Dict[str, float]],
    candidate_executor,
    candidate_ipv_values=None,
):
    primary_tasks = agent_module._build_candidate_ipv_tasks(
        primary_agent,
        counterpart_track,
        solver_options,
        candidate_ipv_values,
    )
    counterpart_tasks = agent_module._build_candidate_ipv_tasks(
        counterpart_agent,
        primary_track,
        solver_options,
        candidate_ipv_values,
    )
    split_index = len(primary_tasks)
    virtual_tracks = list(
        candidate_executor.map(
            agent_module._solve_candidate_ipv_track,
            primary_tasks + counterpart_tasks,
        )
    )
    primary_details = agent_module._apply_candidate_ipv_tracks(
        primary_agent,
        primary_track,
        virtual_tracks[:split_index],
        return_details=return_primary_details,
        candidate_ipv_values=candidate_ipv_values,
    )
    counterpart_details = agent_module._apply_candidate_ipv_tracks(
        counterpart_agent,
        counterpart_track,
        virtual_tracks[split_index:],
        return_details=return_counterpart_details,
        candidate_ipv_values=candidate_ipv_values,
    )
    return primary_details, counterpart_details


def estimate_ipv_pair(
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    history_window: int = 10,
    min_observation: int = 4,
    return_diagnostics: bool = False,
    diagnostic_steps: Optional[Sequence[int]] = None,
    solver_preset: str = "accurate",
    solver_options: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
    candidate_executor=None,
    candidate_ipv_values: Optional[Sequence[float]] = None,
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
        solver_preset: Optimizer preset. ``"accurate"`` preserves the legacy
            SLSQP defaults; ``"balanced"`` is a conservative bounded-iteration
            mode; ``"realtime"`` prioritises online latency; and
            ``"parallel_accurate"`` keeps the accurate solver settings while
            evaluating IPV candidates with a process pool.
        solver_options: Optional SciPy SLSQP options overriding the selected
            preset, e.g. ``{"maxiter": 20, "ftol": 1e-3}``.
        max_workers: Optional process-pool worker count used by
            ``"parallel_accurate"`` when no ``candidate_executor`` is supplied.
        candidate_executor: Optional executor implementing ``map`` for callers
            that want to reuse a persistent worker pool across online frames.
        candidate_ipv_values: Optional explicit IPV candidate grid. Defaults to
            the legacy seven-candidate grid in ``agent.virtual_agent_IPV_range``.

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
    primary_reference = _prepare_reference_for_repeated_use(primary.reference)
    counterpart_reference = _prepare_reference_for_repeated_use(counterpart.reference)
    resolved_solver_options = _resolve_solver_options(solver_preset, solver_options)
    executor_context = nullcontext(candidate_executor)
    if solver_preset == "parallel_accurate" and candidate_executor is None:
        executor_context = ProcessPoolExecutor(max_workers=max_workers)

    with executor_context as active_candidate_executor:
        for t in range(min_observation, steps):
            start = max(0, t - history_window)

            # Primary agent perspective
            primary_agent = Agent(
                primary.data[start, 0:2],
                primary.data[start, 2:4],
                primary.data[start, 4],
                primary.target,
            )
            primary_agent.reference = primary_reference

            primary_track = primary.data[start : t + 1, 0:2]
            counterpart_track = counterpart.data[start : t + 1, 0:2]

            collect_primary = return_diagnostics and (diag_steps is None or t in diag_steps)
            # Counterpart perspective
            counterpart_agent = Agent(
                counterpart.data[start, 0:2],
                counterpart.data[start, 2:4],
                counterpart.data[start, 4],
                counterpart.target,
            )
            counterpart_agent.reference = counterpart_reference

            collect_counter = return_diagnostics and (diag_steps is None or t in diag_steps)
            if active_candidate_executor is None:
                primary_details = _estimate_agent_ipv(
                    primary_agent,
                    primary_track,
                    counterpart_track,
                    return_details=collect_primary,
                    solver_options=resolved_solver_options,
                    candidate_ipv_values=candidate_ipv_values,
                )
                counter_details = _estimate_agent_ipv(
                    counterpart_agent,
                    counterpart_track,
                    primary_track,
                    return_details=collect_counter,
                    solver_options=resolved_solver_options,
                    candidate_ipv_values=candidate_ipv_values,
                )
            else:
                primary_details, counter_details = _estimate_agent_pair_ipv_parallel(
                    primary_agent,
                    counterpart_agent,
                    primary_track,
                    counterpart_track,
                    return_primary_details=collect_primary,
                    return_counterpart_details=collect_counter,
                    solver_options=resolved_solver_options,
                    candidate_executor=active_candidate_executor,
                    candidate_ipv_values=candidate_ipv_values,
                )
            if not collect_primary:
                primary_details = None
            if not collect_counter:
                counter_details = None

            ipv_values[t, 0] = primary_agent.ipv
            ipv_errors[t, 0] = primary_agent.ipv_error

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


def estimate_ipv_current(
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    history_window: int = 10,
    return_diagnostics: bool = False,
    solver_preset: str = "parallel_accurate",
    solver_options: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
    candidate_executor=None,
    candidate_ipv_values: Optional[Sequence[float]] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
]:
    """
    Estimate only the latest IPV pair from the available history.

    This is the online/receding-horizon counterpart to ``estimate_ipv_pair``:
    callers can append the newest sample to each ``MotionSequence`` and invoke
    this function without recomputing all earlier timesteps. For sustained
    realtime loops, prefer ``RealtimeIPVEstimator`` so the worker pool is reused
    across frames.
    """

    steps = min(len(primary.data), len(counterpart.data))
    if steps < 2:
        raise ValueError(
            "At least two trajectory samples are required for current IPV estimation."
        )

    result = estimate_ipv_pair(
        primary,
        counterpart,
        history_window=history_window,
        min_observation=steps - 1,
        return_diagnostics=return_diagnostics,
        diagnostic_steps=[steps - 1] if return_diagnostics else None,
        solver_preset=solver_preset,
        solver_options=solver_options,
        max_workers=max_workers,
        candidate_executor=candidate_executor,
        candidate_ipv_values=candidate_ipv_values,
    )

    if return_diagnostics:
        ipv_values, ipv_errors, diagnostics = result
        return ipv_values[-1].copy(), ipv_errors[-1].copy(), diagnostics

    ipv_values, ipv_errors = result
    return ipv_values[-1].copy(), ipv_errors[-1].copy()


def sign_ipv_values(ipv_values: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Convert IPV values to {-1, 0, 1} with a symmetric neutral band."""

    values = np.asarray(ipv_values, dtype=float)
    return np.where(
        values > threshold,
        1,
        np.where(values < -threshold, -1, 0),
    ).astype(int)


class RealtimeIPVEstimator:
    """
    Stateful online IPV estimator that can reuse workers across frames.

    Use this class for realtime loops. Calling ``estimate_ipv_current`` with
    ``solver_preset="parallel_accurate"`` creates a temporary process pool for
    that call, while this wrapper keeps the pool alive until ``close`` or the
    context manager exits.
    """

    def __init__(
        self,
        *,
        history_window: int = 10,
        solver_preset: str = "parallel_accurate",
        solver_options: Optional[Dict[str, float]] = None,
        max_workers: Optional[int] = None,
        candidate_executor=None,
        candidate_ipv_values: Optional[Sequence[float]] = None,
    ):
        self.history_window = history_window
        self.solver_preset = solver_preset
        self.solver_options = solver_options
        self.max_workers = max_workers
        self.candidate_ipv_values = candidate_ipv_values
        self._external_candidate_executor = candidate_executor
        self._candidate_executor = None

    @classmethod
    def for_realtime_sign(cls, **kwargs):
        kwargs.setdefault("solver_preset", "parallel_accurate")
        if "candidate_ipv_values" not in kwargs:
            kwargs["candidate_ipv_values"] = SIGN_REALTIME_CANDIDATE_IPV_VALUES.copy()
        return cls(**kwargs)

    def __enter__(self):
        self._ensure_candidate_executor()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _ensure_candidate_executor(self):
        if self._external_candidate_executor is not None:
            return self._external_candidate_executor
        if self.solver_preset != "parallel_accurate":
            return None
        if self._candidate_executor is None:
            self._candidate_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._candidate_executor

    def close(self) -> None:
        if self._candidate_executor is not None:
            self._candidate_executor.shutdown()
            self._candidate_executor = None

    def estimate_current(
        self,
        primary: MotionSequence,
        counterpart: MotionSequence,
        *,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
    ]:
        return estimate_ipv_current(
            primary,
            counterpart,
            history_window=self.history_window,
            return_diagnostics=return_diagnostics,
            solver_preset=self.solver_preset,
            solver_options=self.solver_options,
            candidate_executor=self._ensure_candidate_executor(),
            candidate_ipv_values=self.candidate_ipv_values,
        )

    def estimate_sign_current(
        self,
        primary: MotionSequence,
        counterpart: MotionSequence,
        *,
        threshold: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ipv_values, ipv_errors = self.estimate_current(
            primary,
            counterpart,
            return_diagnostics=False,
        )
        return sign_ipv_values(ipv_values, threshold=threshold), ipv_values, ipv_errors


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
