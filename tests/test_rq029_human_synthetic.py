from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.simulation.generate_rq029_human_synthetic import (
    _bounded_allocate,
    _calibrated_quantile_values,
    _semantic_background_payload,
    _warp_ego_rows,
)
from pipelines.simulation.validate_rq029_human_synthetic import (
    _background_hash_and_lines,
    _max_polyline_distance_m,
    _recompute_bootstrap_intervals,
)


def test_bounded_allocate_obeys_total_minimum_and_capacity() -> None:
    got = _bounded_allocate(
        total=19,
        weights=np.array([1.0, 2.0, 3.0]),
        capacities=np.array([4, 7, 12]),
        minimums=np.array([1, 2, 3]),
    )
    assert int(got.sum()) == 19
    assert np.all(got >= np.array([1, 2, 3]))
    assert np.all(got <= np.array([4, 7, 12]))
    balanced = _bounded_allocate(
        total=17,
        weights=np.ones(5),
        capacities=np.full(5, 10),
    )
    assert int(balanced.max() - balanced.min()) <= 1


def test_calibrated_quantiles_and_threshold_count_are_exact() -> None:
    values = _calibrated_quantile_values(
        n=101,
        q25=3.9,
        q50=6.94,
        q75=13.54,
        below_threshold_count=13,
        threshold=2.0,
    )
    assert int(np.sum(values < 2.0)) == 13
    assert np.allclose(np.quantile(values, [0.25, 0.5, 0.75]), [3.9, 6.94, 13.54])
    assert np.all(np.diff(values) >= 0)


def test_warp_is_deterministic_same_path_and_small_speed_change() -> None:
    rows = [
        {
            "latitude": 31.0,
            "longitude": 121.0 + index * 1e-5,
            "speed": 30.0,
            "courseAngle": 90.0,
            "globalTimeStamp": str(1_000 + index * 100),
            "name": "AV",
            "acc": 0.0,
        }
        for index in range(80)
    ]
    first, metrics_first = _warp_ego_rows(rows, seed=22, driver_id="D03", scenario_id="B2")
    second, metrics_second = _warp_ego_rows(rows, seed=22, driver_id="D03", scenario_id="B2")
    assert first == second
    assert metrics_first == metrics_second
    assert first[0]["latitude"] == rows[0]["latitude"]
    assert first[0]["longitude"] == rows[0]["longitude"]
    assert first[-1]["latitude"] == rows[-1]["latitude"]
    assert first[-1]["longitude"] == rows[-1]["longitude"]
    assert metrics_first["max_cross_track_m"] <= 1e-6
    assert 0 < metrics_first["speed_relative_abs_p95"] <= 0.12


def test_background_payload_ignores_only_the_ego_object() -> None:
    record = {
        "caseId": 2325,
        "participantTrajectories": [
            {"role": "av", "value": [{"id": "ego", "isPerception": 0, "speed": 1.0}]},
            {"role": "mvSimulation", "value": [{"id": 7, "speed": 2.0}]},
        ],
    }
    changed = copy.deepcopy(record)
    changed["participantTrajectories"][0]["value"][0]["speed"] = 1.1
    assert _semantic_background_payload(record) == _semantic_background_payload(changed)
    changed["participantTrajectories"][1]["value"][0]["speed"] = 2.1
    assert _semantic_background_payload(record) != _semantic_background_payload(changed)


def test_background_audit_exposes_ego_even_when_background_is_unchanged(
    tmp_path: Path,
) -> None:
    record = {
        "caseId": 2325,
        "participantTrajectories": [
            {
                "role": "av",
                "value": [
                    {
                        "isPerception": 0,
                        "globalTimeStamp": "1",
                        "latitude": 31.0,
                        "longitude": 121.0,
                        "speed": 4.0,
                        "courseAngle": 10.0,
                    }
                ],
            },
            {"role": "mvSimulation", "value": [{"id": 7, "speed": 3.0}]},
        ],
    }
    changed = copy.deepcopy(record)
    changed["participantTrajectories"][0]["value"][0]["speed"] = 999.0
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(json.dumps(record) + "\n", encoding="utf-8")
    second.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    first_background, _, _, first_ego = _background_hash_and_lines(first)
    second_background, _, _, second_ego = _background_hash_and_lines(second)
    assert first_background == second_background
    assert first_ego.loc[0, "ego_speed_kmh"] == 4.0
    assert second_ego.loc[0, "ego_speed_kmh"] == 999.0


def test_bootstrap_intervals_are_recomputed_from_detail_tables() -> None:
    candidates = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "band_90": band,
                "future_min_ttc_s_calibrated": value,
            }
            for run_id in ("r1", "r2", "r3", "r4")
            for band, value in (("lower", 1.5), ("inside", 3.5))
        ]
    )
    counterpart = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "band_90": band,
                "anchor_speed_drop_kmh_calibrated": drop,
                "window_speed_range_kmh_calibrated": speed_range,
                "n_frames_lt_m3_calibrated": brake,
                "n_frames_total_calibrated": 20,
            }
            for run_id in ("r1", "r2", "r3", "r4")
            for band, drop, speed_range, brake in (
                ("lower", 3.0, 4.0, 0),
                ("inside", 1.5, 2.0, 2),
            )
        ]
    )
    unit = pd.DataFrame(
        {
            "n_both": [10, 10, 10, 10],
            "n_below_90": [1, 1, 1, 1],
            "n_above_90": [0, 0, 0, 0],
        }
    )
    first = _recompute_bootstrap_intervals(candidates, counterpart, unit, 17)
    second = _recompute_bootstrap_intervals(candidates, counterpart, unit, 17)
    assert first == second
    assert first["counterpart_speed_drop_ratio"] == [2.0, 2.0]
    assert first["counterpart_brake_difference"][1] < 0.0


def test_cross_track_recomputation_detects_an_off_path_point() -> None:
    source_latitude = np.array([31.0, 31.0, 31.0])
    source_longitude = np.array([121.0, 121.0001, 121.0002])
    on_path = _max_polyline_distance_m(
        source_latitude,
        source_longitude,
        source_latitude,
        source_longitude,
    )
    off_path = _max_polyline_distance_m(
        source_latitude + np.array([0.0, 0.0001, 0.0]),
        source_longitude,
        source_latitude,
        source_longitude,
    )
    assert on_path < 1e-9
    assert off_path > 10.0
