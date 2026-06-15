from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import agent
import ipv_estimation
import process_interhub
from ipv_estimation import MotionSequence, estimate_ipv_pair


def _legacy_group_cost(track_self: np.ndarray, track_inter: np.ndarray) -> float:
    pos_rel = track_inter - track_self
    dis_rel = np.linalg.norm(pos_rel, axis=1)
    vel_self = (track_self[1:, :] - track_self[0:-1, :]) / agent.dt
    vel_inter = (track_inter[1:, :] - track_inter[0:-1, :]) / agent.dt
    vel_rel = vel_self - vel_inter

    vel_rel_along_sum = 0.0
    for i in range(np.size(vel_rel, 0)):
        if dis_rel[i + 1] > 3:
            collision_factor = 0.5
        else:
            collision_factor = 1.5
        nearness_temp = collision_factor * pos_rel[i + 1, :].dot(vel_rel[i, :]) / dis_rel[i + 1]
        vel_rel_along_sum = vel_rel_along_sum + (nearness_temp + np.abs(nearness_temp)) * 0.5
    return vel_rel_along_sum / agent.TRACK_LEN * agent.WEIGHT_GRP


def _legacy_individual_cost(track: np.ndarray, ref) -> float:
    cv, _s = ref
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))

    travel_distance = np.linalg.norm(track[-1, 0:2] - track[0, 0:2]) / np.size(track, 0)
    cost_travel_distance = -travel_distance
    cost_mean_deviation = max(0.2, dis2cv.mean())
    cost_metric = np.array([cost_travel_distance, cost_mean_deviation])
    return agent.weight_metric.dot(cost_metric.T) * agent.WEIGHT_INT


def test_utility_fun_resolves_default_reference_once(monkeypatch):
    calls = []
    prepared_ref = (
        np.array([[0.0, -2.0], [5.0, -2.0], [10.0, -2.0]]),
        np.array([0.0, 5.0, 10.0]),
    )

    def fake_get_central_vertices(target, origin_point=None):
        calls.append((target, None if origin_point is None else tuple(origin_point)))
        return prepared_ref

    monkeypatch.setattr(agent, "get_central_vertices", fake_get_central_vertices)

    self_info = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        0.0,
        0.0,
        "gs",
        None,
    ]
    inter_track = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]])
    fun = agent.utility_fun(self_info, inter_track)
    action = np.zeros(2 * (len(inter_track) - 1))

    fun(action)
    fun(action)

    assert calls == [("gs", None)]


def test_estimate_ipv_pair_prepares_raw_references_once(monkeypatch):
    references_seen = []

    def fake_smooth_ployline(ref):
        cv = np.asarray(ref, dtype=float)
        return cv, np.arange(len(cv), dtype=float)

    def fake_estimate_self_ipv(self, self_actual_track, inter_track, *, return_details=False):
        references_seen.append(self.reference)
        self.ipv = 0.25
        self.ipv_error = 0.1
        details = {
            "virtual_tracks": [np.asarray(self_actual_track, dtype=float)],
            "weights": np.ones(1),
            "ipv_range": np.array([0.25]),
        }
        if return_details:
            return details
        return None

    monkeypatch.setattr(agent, "smooth_ployline", fake_smooth_ployline)
    monkeypatch.setattr(agent.Agent, "estimate_self_ipv", fake_estimate_self_ipv)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    ref_primary = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ref_counter = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])

    estimate_ipv_pair(
        MotionSequence(data=data, target="gs", reference=ref_primary),
        MotionSequence(data=data.copy(), target="lt", reference=ref_counter),
        min_observation=1,
        history_window=2,
    )

    assert len(references_seen) == 4
    assert all(isinstance(ref, tuple) and len(ref) == 2 for ref in references_seen)
    assert references_seen[0] is references_seen[2]
    assert references_seen[1] is references_seen[3]


def test_estimate_self_ipv_does_not_deepcopy_candidate_agents(monkeypatch):
    def fail_deepcopy(_value):
        raise AssertionError("candidate IPV estimation should not deepcopy Agent state")

    def fake_solve_optimization(self, inter_track):
        offsets = np.arange(len(inter_track), dtype=float)[:, None]
        return np.column_stack(
            [
                self.position[0] + offsets[:, 0],
                np.full(len(inter_track), self.position[1] + float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(agent.copy, "deepcopy", fail_deepcopy)
    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    subject = agent.Agent(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, "gs")
    actual = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    other = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

    details = subject.estimate_self_ipv(actual, other, return_details=True)

    assert len(details["virtual_tracks"]) == len(agent.virtual_agent_IPV_range)
    assert math.isfinite(subject.ipv)
    assert math.isfinite(subject.ipv_error)


def test_vectorized_costs_match_legacy_loop_calculations():
    ref = (
        np.array(
            [
                [-1.0, -0.1],
                [0.0, 0.0],
                [1.0, 0.1],
                [2.0, 0.2],
                [3.0, 0.25],
            ]
        ),
        np.arange(5, dtype=float),
    )
    track_self = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.04],
            [0.9, 0.12],
            [1.5, 0.18],
        ]
    )
    track_inter = np.array(
        [
            [4.0, 1.0],
            [3.6, 0.9],
            [3.1, 0.8],
            [2.5, 0.7],
        ]
    )

    assert agent.cal_individual_cost(track_self, target="gs", ref=ref) == pytest.approx(
        _legacy_individual_cost(track_self, ref)
    )
    assert agent.cal_group_cost([track_self, track_inter]) == pytest.approx(
        _legacy_group_cost(track_self, track_inter)
    )


def test_solve_optimization_forwards_solver_options(monkeypatch):
    captured = {}

    def fake_minimize(fun, u0, bounds=None, method=None, options=None):
        captured["options"] = options
        captured["method"] = method
        captured["bounds_count"] = len(bounds)
        return SimpleNamespace(x=np.zeros_like(u0))

    monkeypatch.setattr(agent, "minimize", fake_minimize)

    subject = agent.Agent(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, "gs")
    inter_track = np.array([[5.0, 0.0], [5.2, 0.0], [5.4, 0.0]])

    subject.solve_optimization(inter_track, solver_options={"maxiter": 5, "ftol": 1e-2})

    assert captured == {
        "options": {"maxiter": 5, "ftol": 1e-2},
        "method": "SLSQP",
        "bounds_count": 4,
    }


def test_estimate_ipv_current_can_use_explicit_realtime_preset(monkeypatch):
    calls = []

    def fake_estimate_self_ipv(self, self_actual_track, inter_track, *, return_details=False, solver_options=None):
        calls.append(
            {
                "target": self.target,
                "self_track": np.asarray(self_actual_track).copy(),
                "inter_track": np.asarray(inter_track).copy(),
                "solver_options": solver_options,
            }
        )
        self.ipv = 1.0 if self.target == "gs" else -1.0
        self.ipv_error = 0.2 if self.target == "gs" else 0.3
        if return_details:
            return {
                "virtual_tracks": [np.asarray(self_actual_track, dtype=float)],
                "weights": np.ones(1),
                "ipv_range": np.array([self.ipv]),
            }
        return None

    monkeypatch.setattr(agent.Agent, "estimate_self_ipv", fake_estimate_self_ipv)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    ipv, err = ipv_estimation.estimate_ipv_current(
        MotionSequence(data=data, target="gs"),
        MotionSequence(data=data.copy(), target="lt"),
        history_window=2,
        solver_preset="realtime",
    )

    assert ipv.tolist() == [1.0, -1.0]
    assert err.tolist() == [0.2, 0.3]
    assert [call["target"] for call in calls] == ["gs", "lt"]
    assert all(call["solver_options"] == {"maxiter": 8, "ftol": 1e-2} for call in calls)
    assert calls[0]["self_track"][:, 0].tolist() == [0.1, 0.2, 0.3]
    assert calls[1]["inter_track"][:, 0].tolist() == [0.1, 0.2, 0.3]


def test_estimate_ipv_current_defaults_to_parallel_accurate(monkeypatch):
    executors = []

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.task_batches = []
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, tasks):
            batch = list(tasks)
            self.task_batches.append(batch)
            return [fn(task) for task in batch]

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(ipv_estimation, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    ipv_estimation.estimate_ipv_current(
        MotionSequence(data=data, target="gs"),
        MotionSequence(data=data.copy(), target="lt"),
        history_window=2,
    )

    assert len(executors) == 1
    assert [len(batch) for batch in executors[0].task_batches] == [
        2 * len(agent.virtual_agent_IPV_range)
    ]


def test_sign_ipv_values_uses_symmetric_neutral_threshold():
    signs = ipv_estimation.sign_ipv_values(np.array([-0.2, -0.05, 0.0, 0.05, 0.2]))

    assert signs.tolist() == [-1, 0, 0, 0, 1]


def test_realtime_ipv_estimator_can_return_current_sign(monkeypatch):
    def fake_estimate_ipv_current(primary, counterpart, **kwargs):
        return np.array([0.2, -0.01]), np.array([0.1, 0.2])

    monkeypatch.setattr(ipv_estimation, "estimate_ipv_current", fake_estimate_ipv_current)

    estimator = ipv_estimation.RealtimeIPVEstimator(history_window=2)
    signs, ipv, err = estimator.estimate_sign_current(
        MotionSequence(data=np.zeros((2, 5)), target="gs"),
        MotionSequence(data=np.zeros((2, 5)), target="lt"),
        threshold=0.05,
    )

    assert signs.tolist() == [1, 0]
    assert ipv.tolist() == [0.2, -0.01]
    assert err.tolist() == [0.1, 0.2]


def test_sign_realtime_candidate_grid_is_named():
    expected = np.array([-3, -1, 0, 1, 3], dtype=float) * np.pi / 8

    assert np.allclose(ipv_estimation.SIGN_REALTIME_CANDIDATE_IPV_VALUES, expected)


def test_realtime_ipv_estimator_sign_optimized_factory_uses_named_grid():
    estimator = ipv_estimation.RealtimeIPVEstimator.for_realtime_sign(
        history_window=2,
        max_workers=3,
    )

    assert estimator.history_window == 2
    assert estimator.solver_preset == "parallel_accurate"
    assert estimator.max_workers == 3
    assert np.allclose(
        estimator.candidate_ipv_values,
        ipv_estimation.SIGN_REALTIME_CANDIDATE_IPV_VALUES,
    )


def test_process_interhub_cli_accepts_solver_preset():
    args = process_interhub.build_arg_parser().parse_args(["--solver-preset", "realtime"])

    assert args.solver_preset == "realtime"


def test_process_interhub_cli_accepts_parallel_accurate_solver_preset():
    args = process_interhub.build_arg_parser().parse_args(
        ["--solver-preset", "parallel_accurate"]
    )

    assert args.solver_preset == "parallel_accurate"


def test_estimate_self_ipv_can_use_candidate_executor(monkeypatch):
    class RecordingExecutor:
        def __init__(self):
            self.tasks = []

        def map(self, fn, tasks):
            self.tasks = list(tasks)
            return [fn(task) for task in self.tasks]

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    executor = RecordingExecutor()
    subject = agent.Agent(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, "gs")
    actual = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    other = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

    details = subject.estimate_self_ipv(
        actual,
        other,
        return_details=True,
        candidate_executor=executor,
    )

    assert len(executor.tasks) == len(agent.virtual_agent_IPV_range)
    assert len(details["virtual_tracks"]) == len(agent.virtual_agent_IPV_range)
    assert math.isfinite(subject.ipv)
    assert math.isfinite(subject.ipv_error)


def test_estimate_self_ipv_accepts_candidate_ipv_values(monkeypatch):
    custom_ipv_values = np.array([-1.0, 0.0, 1.0])

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    subject = agent.Agent(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, "gs")
    actual = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    other = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

    details = subject.estimate_self_ipv(
        actual,
        other,
        return_details=True,
        candidate_ipv_values=custom_ipv_values,
    )

    assert len(details["virtual_tracks"]) == len(custom_ipv_values)
    assert details["ipv_range"].tolist() == custom_ipv_values.tolist()


def test_estimate_ipv_current_parallel_accurate_creates_candidate_executor(monkeypatch):
    executors = []

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.shutdown_called = False
            self.task_batches = []
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.shutdown_called = True
            return False

        def map(self, fn, tasks):
            batch = list(tasks)
            self.task_batches.append(batch)
            return [fn(task) for task in batch]

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(ipv_estimation, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    ipv, err = ipv_estimation.estimate_ipv_current(
        MotionSequence(data=data, target="gs"),
        MotionSequence(data=data.copy(), target="lt"),
        history_window=2,
        solver_preset="parallel_accurate",
        max_workers=3,
    )

    assert ipv.shape == (2,)
    assert err.shape == (2,)
    assert np.all(np.isfinite(ipv))
    assert np.all(np.isfinite(err))
    assert len(executors) == 1
    assert executors[0].max_workers == 3
    assert executors[0].shutdown_called is True
    assert [len(batch) for batch in executors[0].task_batches] == [
        2 * len(agent.virtual_agent_IPV_range)
    ]


def test_parallel_accurate_batches_both_agents_in_one_executor_map(monkeypatch):
    class RecordingExecutor:
        def __init__(self):
            self.task_batches = []

        def map(self, fn, tasks):
            batch = list(tasks)
            self.task_batches.append(batch)
            return [fn(task) for task in batch]

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    executor = RecordingExecutor()

    ipv_estimation.estimate_ipv_current(
        MotionSequence(data=data, target="gs"),
        MotionSequence(data=data.copy(), target="lt"),
        history_window=2,
        solver_preset="parallel_accurate",
        candidate_executor=executor,
    )

    assert [len(batch) for batch in executor.task_batches] == [
        2 * len(agent.virtual_agent_IPV_range)
    ]


def test_parallel_accurate_batches_custom_candidate_ipv_values(monkeypatch):
    class RecordingExecutor:
        def __init__(self):
            self.task_batches = []

        def map(self, fn, tasks):
            batch = list(tasks)
            self.task_batches.append(batch)
            return [fn(task) for task in batch]

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    executor = RecordingExecutor()
    custom_ipv_values = np.array([-1.0, 0.0, 1.0])

    ipv_estimation.estimate_ipv_current(
        MotionSequence(data=data, target="gs"),
        MotionSequence(data=data.copy(), target="lt"),
        history_window=2,
        solver_preset="parallel_accurate",
        candidate_executor=executor,
        candidate_ipv_values=custom_ipv_values,
    )

    assert [len(batch) for batch in executor.task_batches] == [
        2 * len(custom_ipv_values)
    ]


def test_realtime_ipv_estimator_reuses_parallel_executor(monkeypatch):
    executors = []

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.shutdown_calls = 0
            self.task_batches = []
            executors.append(self)

        def map(self, fn, tasks):
            batch = list(tasks)
            self.task_batches.append(batch)
            return [fn(task) for task in batch]

        def shutdown(self):
            self.shutdown_calls += 1

    def fake_solve_optimization(self, inter_track, *, solver_options=None):
        offsets = np.arange(len(inter_track), dtype=float)
        return np.column_stack(
            [
                offsets,
                np.full(len(inter_track), float(self.ipv)),
                np.ones(len(inter_track)),
                np.zeros(len(inter_track)),
                np.zeros(len(inter_track)),
            ]
        )

    monkeypatch.setattr(ipv_estimation, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(agent.Agent, "solve_optimization", fake_solve_optimization)

    data = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    with ipv_estimation.RealtimeIPVEstimator(history_window=2, max_workers=4) as estimator:
        first_ipv, _ = estimator.estimate_current(
            MotionSequence(data=data[:3], target="gs"),
            MotionSequence(data[:3].copy(), target="lt"),
        )
        second_ipv, _ = estimator.estimate_current(
            MotionSequence(data=data, target="gs"),
            MotionSequence(data.copy(), target="lt"),
        )

    assert np.all(np.isfinite(first_ipv))
    assert np.all(np.isfinite(second_ipv))
    assert len(executors) == 1
    assert executors[0].max_workers == 4
    assert executors[0].shutdown_calls == 1
    assert [len(batch) for batch in executors[0].task_batches] == [
        2 * len(agent.virtual_agent_IPV_range),
        2 * len(agent.virtual_agent_IPV_range),
    ]
