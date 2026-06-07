from __future__ import annotations

import numpy as np

import agent


def test_utility_function_smooths_static_reference_once(monkeypatch):
    calls = []

    def fake_smooth_ployline(ref):
        calls.append(np.asarray(ref).shape)
        cv = np.asarray(ref, dtype=float)
        return cv, np.arange(len(cv), dtype=float)

    monkeypatch.setattr(agent, "smooth_ployline", fake_smooth_ployline)

    self_info = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        0.0,
        0.0,
        "gs",
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
    ]
    inter_track = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]])
    fun = agent.utility_fun(self_info, inter_track)
    action = np.zeros(2 * (len(inter_track) - 1))

    fun(action)
    fun(action)

    assert calls == [(4, 2)]


def test_utility_function_accepts_prepared_reference_without_resmoothing(monkeypatch):
    def fail_smooth_ployline(_ref):
        raise AssertionError("prepared references should not be smoothed again")

    monkeypatch.setattr(agent, "smooth_ployline", fail_smooth_ployline)

    prepared_reference = (
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        np.arange(4, dtype=float),
    )
    self_info = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        0.0,
        0.0,
        "gs",
        prepared_reference,
    ]
    inter_track = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]])
    fun = agent.utility_fun(self_info, inter_track)
    action = np.zeros(2 * (len(inter_track) - 1))

    value = fun(action)

    assert np.isfinite(value)
