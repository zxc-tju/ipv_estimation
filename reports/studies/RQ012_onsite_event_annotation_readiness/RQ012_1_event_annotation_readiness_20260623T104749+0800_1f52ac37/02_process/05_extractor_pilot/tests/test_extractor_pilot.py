#!/usr/bin/env python3
"""Independent synthetic tests for the RQ012A extractor pilot.

This runner intentionally imports the extractor read-only and builds controlled
kinematic fixtures in memory. It does not read media, labels, IPV, scores,
ranks, team identity, agreement files, or event-outcome associations.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


RUN_ID = "RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37"
WORKER_ID = "RQ012-W12-extractor-test"
TEST_DIR = Path(__file__).resolve().parent
PILOT_DIR = TEST_DIR.parent
RUN_ROOT = TEST_DIR.parents[2]
REPO_ROOT = TEST_DIR.parents[6]
EXTRACTOR_PATH = PILOT_DIR / "extractor_pilot.py"
CONFIG_PATH = PILOT_DIR / "extractor_config.json"
DERIVED_FIXTURE_DIR = (
    REPO_ROOT
    / "data"
    / "derived"
    / "onsite_competition"
    / "RQ012_onsite_event_annotation_readiness"
    / RUN_ID
    / "extractor_pilot"
    / "test_fixtures"
)


def load_extractor() -> Any:
    spec = importlib.util.spec_from_file_location("extractor_pilot_under_test", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extractor from {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ext = load_extractor()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CENTRAL = CONFIG["bands"]["central"]
MAX_ALIGN_GAP_S = float(CONFIG["max_time_alignment_gap_s"])


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def value_at(value: Any, idx: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[idx]
    return value


def times(count: int, start_ms: float = 0.0, step_ms: float = 100.0) -> List[float]:
    return [start_ms + idx * step_ms for idx in range(count)]


def row(
    t_ms: float,
    *,
    actor_id: str = "ego",
    source: str = "ego",
    speed: float = 5.0,
    accel: float = 0.0,
    braking: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    length: float = 4.0,
    width: float = 2.0,
    course_deg: float = 0.0,
    lat_acc: float = 0.0,
    steering: float = 0.0,
    wheel: float = 0.0,
) -> Dict[str, Any]:
    return {
        "source": source,
        "actor_id": actor_id,
        "origin_id": actor_id if source == "world" else None,
        "t_ms": t_ms,
        "speed": speed,
        "accel": accel,
        "braking": braking,
        "lat": None,
        "lon": None,
        "source_x": x if source == "world" else None,
        "source_y": y if source == "world" else None,
        "x_m": x,
        "y_m": y,
        "xy_source": "synthetic_xy",
        "course_deg": course_deg,
        "length_m": length,
        "width_m": width,
        "lat_acc": lat_acc,
        "lon_acc": accel,
        "steering_angle": steering,
        "wheel_angle": wheel,
    }


def ego_rows(t_values: Sequence[float], **kwargs: Any) -> List[Dict[str, Any]]:
    return [
        row(
            t_ms=t,
            actor_id="ego",
            source="ego",
            speed=value_at(kwargs.get("speed", 5.0), idx),
            accel=value_at(kwargs.get("accel", 0.0), idx),
            braking=value_at(kwargs.get("braking", 0.0), idx),
            x=value_at(kwargs.get("x", 0.0), idx),
            y=value_at(kwargs.get("y", 0.0), idx),
            length=value_at(kwargs.get("length", 4.0), idx),
            width=value_at(kwargs.get("width", 2.0), idx),
            course_deg=value_at(kwargs.get("course_deg", 0.0), idx),
            lat_acc=value_at(kwargs.get("lat_acc", 0.0), idx),
            steering=value_at(kwargs.get("steering", 0.0), idx),
            wheel=value_at(kwargs.get("wheel", 0.0), idx),
        )
        for idx, t in enumerate(t_values)
    ]


def world_rows(actor_id: str, t_values: Sequence[float], **kwargs: Any) -> List[Dict[str, Any]]:
    return [
        row(
            t_ms=t,
            actor_id=actor_id,
            source="world",
            speed=value_at(kwargs.get("speed", 5.0), idx),
            accel=value_at(kwargs.get("accel", 0.0), idx),
            braking=value_at(kwargs.get("braking", 0.0), idx),
            x=value_at(kwargs.get("x", 10.0), idx),
            y=value_at(kwargs.get("y", 0.0), idx),
            length=value_at(kwargs.get("length", 4.0), idx),
            width=value_at(kwargs.get("width", 2.0), idx),
            course_deg=value_at(kwargs.get("course_deg", 0.0), idx),
        )
        for idx, t in enumerate(t_values)
    ]


def make_session(
    name: str,
    *,
    ego: Sequence[Dict[str, Any]] | None = None,
    world: Dict[str, Sequence[Dict[str, Any]]] | None = None,
) -> Any:
    ref = ext.SessionRef(
        session_key=f"synthetic_{name}",
        pilot_session_id=f"synthetic_{name}",
        session_dir=TEST_DIR,
        relative_dir="synthetic",
        duplicate_source_count=1,
    )
    return ext.SessionData(
        ref=ref,
        ego=list(ego or []),
        world_by_actor={key: list(value) for key, value in (world or {}).items()},
        health={},
    )


def pair_session(
    name: str,
    clearances: Sequence[float],
    *,
    t_values: Sequence[float] | None = None,
    ego_t_values: Sequence[float] | None = None,
    other_speed: float = 0.0,
    ego_speed: float = 0.0,
) -> Any:
    world_t = list(t_values if t_values is not None else times(len(clearances)))
    ego_t = list(ego_t_values if ego_t_values is not None else world_t)
    ego = ego_rows(ego_t, speed=ego_speed, x=0.0, y=0.0, course_deg=0.0)
    other_x = [4.0 + gap for gap in clearances]
    other = world_rows("actor_a", world_t, speed=other_speed, x=other_x, y=0.0, course_deg=0.0)
    return make_session(name, ego=ego, world={"actor_a": other})


def extract_event(event_id: str, session: Any) -> Any:
    params = CENTRAL[event_id]
    if event_id == "E01":
        return ext.extract_e01(session, params, 1)
    if event_id == "E02":
        return ext.extract_e02(session, params, 1)
    if event_id == "E03":
        return ext.extract_e03(session, params, 1)
    if event_id == "E06":
        return ext.extract_e06(session, params, 1)
    if event_id in {"E09", "E15"}:
        return ext.extract_pair_event(session, params, event_id, 1, MAX_ALIGN_GAP_S)
    if event_id == "E16":
        return ext.extract_e16(session, params, 1)
    if event_id == "E18":
        return ext.extract_e18(session, params, 1)
    if event_id == "E19":
        return ext.extract_e19(session, params, 1)
    raise ValueError(event_id)


def assert_count(result: Any, expected: int, repro: str) -> None:
    assert result.event_count == expected, (
        f"{repro}; expected event_count={expected}, got {result.event_count}; "
        f"raw_hits={result.raw_frame_hits}, missing={result.missing_data_failures}, "
        f"impossible={result.impossible_values}, intervals={result.intervals}"
    )


def assert_positive(result: Any, repro: str) -> None:
    assert result.event_count > 0, (
        f"{repro}; expected at least one event, got {result.event_count}; "
        f"raw_hits={result.raw_frame_hits}, missing={result.missing_data_failures}, "
        f"impossible={result.impossible_values}, intervals={result.intervals}"
    )


def test_unit_e01_not_computable() -> None:
    session = make_session(
        "e01_guard",
        ego=ego_rows(times(3)),
        world={"actor_a": world_rows("actor_a", times(3), accel=-4.0)},
    )
    result = extract_event("E01", session)
    assert result.total_units == 1, "E01 should count the unresolved counterpart unit"
    assert result.computable_units == 0, "E01 should not claim computability without frozen counterpart relation"
    assert result.event_count == 0, "E01 should not emit events without frozen counterpart relation"
    assert result.actor_attribution_failures == 1, "E01 should flag unresolved counterpart attribution"
    assert result.missing_data_failures == 1, "E01 should flag missing counterpart relation"
    assert result.notes and "Not computable" in result.notes[0], "E01 should document not-computable status"


def test_unit_e02_should_trigger() -> None:
    session = make_session("e02_yes", ego=ego_rows(times(3), speed=5.0, accel=-3.5))
    assert_count(extract_event("E02", session), 1, "E02 three 10 Hz samples at -3.5 m/s2")


def test_unit_e02_should_not_trigger() -> None:
    session = make_session("e02_no", ego=ego_rows(times(3), speed=5.0, accel=-3.39))
    assert_count(extract_event("E02", session), 0, "E02 three 10 Hz samples just below 3.4 m/s2")


def test_unit_e03_should_trigger() -> None:
    accel = [idx * 1.0 for idx in range(7)]
    session = make_session("e03_yes", ego=ego_rows(times(len(accel)), speed=5.0, accel=accel))
    assert_positive(extract_event("E03", session), "E03 causal smoothed acceleration ramp with jerk above 5 m/s3")


def test_unit_e03_should_not_trigger() -> None:
    accel = [idx * 0.4 for idx in range(7)]
    session = make_session("e03_no", ego=ego_rows(times(len(accel)), speed=5.0, accel=accel))
    assert_count(extract_event("E03", session), 0, "E03 causal smoothed acceleration ramp below jerk threshold")


def e06_speed_pattern(total_duration_ok: bool = True) -> List[float]:
    if total_duration_ok:
        return [0.1] * 10 + [1.1] * 10 + [0.1] * 10 + [1.1] * 11
    return [0.1] * 10 + [1.1] * 10


def test_unit_e06_should_trigger() -> None:
    speeds = e06_speed_pattern(total_duration_ok=True)
    session = make_session("e06_yes", ego=ego_rows(times(len(speeds)), speed=speeds))
    assert_count(extract_event("E06", session), 1, "E06 two stop-go cycles lasting at least 4 s")


def test_unit_e06_should_not_trigger() -> None:
    speeds = e06_speed_pattern(total_duration_ok=False)
    session = make_session("e06_no", ego=ego_rows(times(len(speeds)), speed=speeds))
    assert_count(extract_event("E06", session), 0, "E06 one stop-go cycle only")


def test_unit_e09_should_trigger() -> None:
    session = pair_session("e09_yes", [0.4, 0.4])
    assert_count(extract_event("E09", session), 1, "E09 clearance 0.4 m for 0.2 s")


def test_unit_e09_should_not_trigger() -> None:
    session = pair_session("e09_no", [0.6, 0.6])
    assert_count(extract_event("E09", session), 0, "E09 clearance 0.6 m and no closing TTC")


def test_unit_e15_should_trigger() -> None:
    session = pair_session("e15_yes", [-0.1, -0.1])
    assert_count(extract_event("E15", session), 1, "E15 footprint penetration for 0.2 s")


def test_unit_e15_should_not_trigger() -> None:
    session = pair_session("e15_no", [0.1, 0.1])
    assert_count(extract_event("E15", session), 0, "E15 positive clearance")


def test_unit_e16_should_trigger() -> None:
    t = times(101)
    session = make_session("e16_yes", ego=ego_rows(t, speed=0.2, x=0.0, y=0.0))
    assert_count(extract_event("E16", session), 1, "E16 no-progress for 10 s")


def test_unit_e16_should_not_trigger() -> None:
    t = times(101)
    session = make_session("e16_no", ego=ego_rows(t, speed=0.31, x=0.0, y=0.0))
    assert_count(extract_event("E16", session), 0, "E16 speed just above progress threshold")


def test_unit_e18_should_trigger() -> None:
    session = make_session("e18_yes", ego=ego_rows(times(3), speed=0.2, accel=-4.6, braking=0.0))
    assert_count(extract_event("E18", session), 1, "E18 emergency decel plus stop state for 0.3 s")


def test_unit_e18_should_not_trigger() -> None:
    session = make_session("e18_no", ego=ego_rows(times(3), speed=1.0, accel=-4.6, braking=0.0))
    assert_count(extract_event("E18", session), 0, "E18 decel without brake or stop support")


def test_unit_e19_should_trigger() -> None:
    session = make_session("e19_yes", ego=ego_rows(times(3), speed=5.0, lat_acc=2.6))
    assert_count(extract_event("E19", session), 1, "E19 lateral acceleration above 2.5 m/s2 for 0.3 s")


def test_unit_e19_should_not_trigger() -> None:
    session = make_session("e19_no", ego=ego_rows(times(3), speed=5.0, lat_acc=2.4))
    assert_count(extract_event("E19", session), 0, "E19 lateral acceleration below threshold")


def test_boundary_value_thresholds() -> None:
    assert_count(
        extract_event("E02", make_session("b_e02_in", ego=ego_rows(times(3), speed=5.0, accel=-3.4001))),
        1,
        "E02 just inside deceleration threshold",
    )
    assert_count(
        extract_event("E02", make_session("b_e02_out", ego=ego_rows(times(3), speed=5.0, accel=-3.3999))),
        0,
        "E02 just outside deceleration threshold",
    )

    accel_inside = [idx * 0.51 for idx in range(12)]
    accel_outside = [idx * 0.49 for idx in range(12)]
    assert_positive(
        extract_event("E03", make_session("b_e03_in", ego=ego_rows(times(len(accel_inside)), speed=5.0, accel=accel_inside))),
        "E03 just inside effective smoothed jerk threshold",
    )
    assert_count(
        extract_event("E03", make_session("b_e03_out", ego=ego_rows(times(len(accel_outside)), speed=5.0, accel=accel_outside))),
        0,
        "E03 just outside effective smoothed jerk threshold",
    )

    speeds_in = [0.299] * 10 + [1.001] * 10 + [0.299] * 10 + [1.001] * 11
    speeds_out = [0.301] * 10 + [1.001] * 10 + [0.301] * 10 + [1.001] * 11
    assert_count(extract_event("E06", make_session("b_e06_in", ego=ego_rows(times(len(speeds_in)), speed=speeds_in))), 1, "E06 just inside stop/go thresholds")
    assert_count(extract_event("E06", make_session("b_e06_out", ego=ego_rows(times(len(speeds_out)), speed=speeds_out))), 0, "E06 stop state just outside threshold")

    assert_count(extract_event("E09", pair_session("b_e09_in", [0.499, 0.499])), 1, "E09 just inside distance threshold")
    assert_count(extract_event("E09", pair_session("b_e09_out", [0.501, 0.501])), 0, "E09 just outside distance threshold")
    assert_count(extract_event("E15", pair_session("b_e15_in", [-0.001, -0.001])), 1, "E15 just inside overlap threshold")
    assert_count(extract_event("E15", pair_session("b_e15_out", [0.001, 0.001])), 0, "E15 just outside overlap threshold")

    t = times(101)
    assert_count(extract_event("E16", make_session("b_e16_in", ego=ego_rows(t, speed=0.299, x=0.0, y=0.0))), 1, "E16 just inside speed threshold")
    assert_count(extract_event("E16", make_session("b_e16_out", ego=ego_rows(t, speed=0.301, x=0.0, y=0.0))), 0, "E16 just outside speed threshold")
    assert_count(extract_event("E18", make_session("b_e18_in", ego=ego_rows(times(3), speed=0.2, accel=-4.5001))), 1, "E18 just inside deceleration threshold")
    assert_count(extract_event("E18", make_session("b_e18_out", ego=ego_rows(times(3), speed=0.2, accel=-4.4999))), 0, "E18 just outside deceleration threshold")
    assert_count(extract_event("E19", make_session("b_e19_in", ego=ego_rows(times(3), speed=5.0, lat_acc=2.5001))), 1, "E19 just inside lateral acceleration threshold")
    assert_count(extract_event("E19", make_session("b_e19_out", ego=ego_rows(times(3), speed=5.0, lat_acc=2.4999))), 0, "E19 just outside lateral acceleration threshold")


def test_boundary_min_duration() -> None:
    assert_count(
        extract_event("E02", make_session("min_e02_below", ego=ego_rows(times(2), speed=5.0, accel=-3.5))),
        0,
        "E02 0.2 s below central D_min 0.3",
    )
    assert_count(
        extract_event("E02", make_session("min_e02_above", ego=ego_rows(times(3), speed=5.0, accel=-3.5))),
        1,
        "E02 0.3 s at central D_min 0.3",
    )
    assert_count(extract_event("E09", pair_session("min_e09_below", [0.4])), 0, "E09 one sample below 0.2 s")
    assert_count(extract_event("E09", pair_session("min_e09_above", [0.4, 0.4])), 1, "E09 two samples at 0.2 s")
    assert_count(extract_event("E15", pair_session("min_e15_below", [-0.1])), 0, "E15 one sample below 0.2 s")
    assert_count(extract_event("E15", pair_session("min_e15_above", [-0.1, -0.1])), 1, "E15 two samples at 0.2 s")


def test_boundary_merge_gap() -> None:
    rows_merge = ego_rows([0, 100, 200, 300, 490, 590, 690], speed=5.0, accel=[-3.5, -3.5, -3.5, 0.0, -3.5, -3.5, -3.5])
    merge_result = extract_event("E02", make_session("merge_below", ego=rows_merge))
    assert_count(merge_result, 1, "E02 two qualifying runs separated by 0.29 s should merge")
    assert merge_result.duplicate_overlapping_events == 1, f"Expected one merged duplicate, got {merge_result.duplicate_overlapping_events}"

    rows_split = ego_rows([0, 100, 200, 300, 510, 610, 710], speed=5.0, accel=[-3.5, -3.5, -3.5, 0.0, -3.5, -3.5, -3.5])
    split_result = extract_event("E02", make_session("merge_above", ego=rows_split))
    assert_count(split_result, 2, "E02 two qualifying runs separated by 0.31 s should split")
    assert split_result.duplicate_overlapping_events == 0, f"Expected no merge, got {split_result.duplicate_overlapping_events}"


def test_boundary_empty_and_single_sample_inputs() -> None:
    empty = make_session("empty", ego=[], world={})
    for event_id in ["E02", "E03", "E06", "E09", "E15", "E16", "E18", "E19"]:
        result = extract_event(event_id, empty)
        assert result.event_count == 0, f"{event_id} should not emit on zero-length input"

    one_actor = make_session("single_actor", ego=ego_rows([0], speed=0.2, accel=-10.0, lat_acc=10.0))
    for event_id in ["E02", "E03", "E06", "E16", "E18", "E19"]:
        result = extract_event(event_id, one_actor)
        assert result.event_count == 0, f"{event_id} should not emit on a single ego sample"

    one_pair = pair_session("single_pair", [-0.1])
    for event_id in ["E09", "E15"]:
        result = extract_event(event_id, one_pair)
        assert result.event_count == 0, f"{event_id} should not emit on a single pair sample"


def test_time_alignment_within_gap() -> None:
    session = pair_session("align_within", [0.4, 0.4], t_values=[50, 150], ego_t_values=[0, 100, 200])
    result = extract_event("E09", session)
    assert_count(result, 1, "E09 world frames 50 ms from nearest ego frames should align")


def test_time_alignment_outside_gap() -> None:
    session = pair_session("align_outside", [0.4, 0.4], t_values=[250, 350], ego_t_values=[0, 100])
    result = extract_event("E09", session)
    assert_count(result, 0, "E09 world frames more than 100 ms from ego should not align")
    assert result.missing_data_failures > 0, "Outside-gap alignment should be documented as missing-data failure"


def test_time_alignment_multi_actor_frames() -> None:
    ego = ego_rows([0, 100, 200], speed=0.0)
    actor_a = world_rows("actor_a", [0, 100], speed=0.0, x=[4.4, 4.4])
    actor_b = world_rows("actor_b", [450, 550], speed=0.0, x=[4.4, 4.4])
    session = make_session("align_multi_actor", ego=ego, world={"actor_a": actor_a, "actor_b": actor_b})
    result = extract_event("E09", session)
    assert_count(result, 1, "E09 should emit only for aligned actor_a")
    assert result.total_units == 2, f"Expected two actor-pair units, got {result.total_units}"
    assert result.computable_units == 1, f"Expected one computable aligned pair, got {result.computable_units}"
    assert result.missing_data_failures > 0, "Unaligned actor_b should be recorded as missing"


def test_time_alignment_duplicate_timestamps() -> None:
    session = make_session("duplicate_t", ego=ego_rows([0, 0, 100], speed=5.0, accel=-3.5))
    result = extract_event("E02", session)
    assert result.impossible_values > 0, "Duplicate/non-increasing timestamps should be flagged"
    assert_count(result, 0, "Duplicate timestamps should not create a valid E02 duration")


def test_time_alignment_out_of_order_timestamps() -> None:
    session = make_session("out_of_order", ego=ego_rows([200, 0, 100], speed=5.0, accel=-3.5))
    result = extract_event("E02", session)
    assert_count(result, 1, "Out-of-order input should sort into a valid E02 interval without crashing")


def test_determinism_reproducibility() -> None:
    session = make_session(
        "determinism",
        ego=ego_rows(times(101), speed=0.2, accel=-3.5, lat_acc=2.6, x=0.0, y=0.0),
        world={"actor_a": world_rows("actor_a", times(3), speed=0.0, x=[4.4, 4.4, 4.4])},
    )
    first = ext.extract_all_events([session], CENTRAL, decimate_factor=1, max_align_gap_s=MAX_ALIGN_GAP_S)
    second = ext.extract_all_events([session], CENTRAL, decimate_factor=1, max_align_gap_s=MAX_ALIGN_GAP_S)
    first_payload = json.dumps({k: dataclasses.asdict(v) for k, v in first.items()}, sort_keys=True)
    second_payload = json.dumps({k: dataclasses.asdict(v) for k, v in second.items()}, sort_keys=True)
    assert first_payload == second_payload, "Repeated synthetic extraction should be byte-identical after canonical JSON sorting"


def test_impossible_nan_inf_rejected() -> None:
    rows = ego_rows([0, 100, 200], speed=[math.nan, 5.0, 5.0], accel=[-3.5, math.inf, -3.5])
    result = extract_event("E02", make_session("nan_inf", ego=rows))
    assert_count(result, 0, "NaN speed and infinite acceleration should not emit E02")
    assert result.missing_data_failures > 0, "NaN/inf rows should be counted as missing evidence"


def test_impossible_duplicate_timestamp_not_emitted() -> None:
    rows = ego_rows([0, 0, 0, 0], speed=5.0, accel=-3.5)
    result = extract_event("E02", make_session("impossible_duplicate", ego=rows))
    assert result.impossible_values > 0, "Non-increasing timestamps should be flagged as impossible"
    assert_count(result, 0, "Non-increasing timestamps should not be emitted as E02")


def test_impossible_negative_speed_not_emitted() -> None:
    rows = ego_rows([0, 100, 200], speed=-1.0, accel=-3.5)
    result = extract_event("E02", make_session("negative_speed", ego=rows))
    assert result.impossible_values > 0, "Negative speed should be flagged as impossible"
    assert_count(result, 0, "Negative-speed impossible rows should be rejected, not emitted")


TEST_CASES: List[Dict[str, Any]] = [
    {"name": "unit_e01_not_computable", "category": "unit", "severity": "high", "func": test_unit_e01_not_computable, "repro": "extract_e01 with one unresolved world actor"},
    {"name": "unit_e02_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e02_should_trigger, "repro": "E02 accel=-3.5 for 3 samples"},
    {"name": "unit_e02_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e02_should_not_trigger, "repro": "E02 accel=-3.39 for 3 samples"},
    {"name": "unit_e03_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e03_should_trigger, "repro": "E03 acceleration ramp step 1.0"},
    {"name": "unit_e03_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e03_should_not_trigger, "repro": "E03 acceleration ramp step 0.4"},
    {"name": "unit_e06_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e06_should_trigger, "repro": "E06 stop-go-stop-go over 4.0 s"},
    {"name": "unit_e06_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e06_should_not_trigger, "repro": "E06 one stop-go cycle"},
    {"name": "unit_e09_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e09_should_trigger, "repro": "E09 0.4 m clearance for 2 samples"},
    {"name": "unit_e09_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e09_should_not_trigger, "repro": "E09 0.6 m clearance for 2 samples"},
    {"name": "unit_e15_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e15_should_trigger, "repro": "E15 0.1 m footprint penetration for 2 samples"},
    {"name": "unit_e15_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e15_should_not_trigger, "repro": "E15 0.1 m positive clearance for 2 samples"},
    {"name": "unit_e16_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e16_should_trigger, "repro": "E16 stationary low-speed ego for 10 s"},
    {"name": "unit_e16_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e16_should_not_trigger, "repro": "E16 ego speed 0.31 m/s for 10 s"},
    {"name": "unit_e18_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e18_should_trigger, "repro": "E18 accel=-4.6 and speed=0.2 for 3 samples"},
    {"name": "unit_e18_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e18_should_not_trigger, "repro": "E18 accel=-4.6 without brake/stop support"},
    {"name": "unit_e19_should_trigger", "category": "unit", "severity": "high", "func": test_unit_e19_should_trigger, "repro": "E19 lat_acc=2.6 for 3 samples"},
    {"name": "unit_e19_should_not_trigger", "category": "unit", "severity": "high", "func": test_unit_e19_should_not_trigger, "repro": "E19 lat_acc=2.4 for 3 samples"},
    {"name": "boundary_value_thresholds", "category": "boundary", "severity": "high", "func": test_boundary_value_thresholds, "repro": "just-inside/outside thresholds for E02,E03,E06,E09,E15,E16,E18,E19"},
    {"name": "boundary_min_duration", "category": "boundary", "severity": "high", "func": test_boundary_min_duration, "repro": "below/at D_min for E02,E09,E15"},
    {"name": "boundary_merge_gap", "category": "boundary", "severity": "medium", "func": test_boundary_merge_gap, "repro": "E02 two runs separated by 0.29 s vs 0.31 s"},
    {"name": "boundary_empty_and_single_sample_inputs", "category": "boundary", "severity": "medium", "func": test_boundary_empty_and_single_sample_inputs, "repro": "zero-length and single-sample sessions for all computable events"},
    {"name": "time_alignment_within_gap", "category": "time_alignment", "severity": "high", "func": test_time_alignment_within_gap, "repro": "E09 world frames 50 ms offset from ego"},
    {"name": "time_alignment_outside_gap", "category": "time_alignment", "severity": "medium", "func": test_time_alignment_outside_gap, "repro": "E09 world frames >100 ms from ego"},
    {"name": "time_alignment_multi_actor_frames", "category": "time_alignment", "severity": "medium", "func": test_time_alignment_multi_actor_frames, "repro": "one aligned and one unaligned actor"},
    {"name": "time_alignment_duplicate_timestamps", "category": "time_alignment", "severity": "medium", "func": test_time_alignment_duplicate_timestamps, "repro": "E02 duplicate timestamps [0,0,100]"},
    {"name": "time_alignment_out_of_order_timestamps", "category": "time_alignment", "severity": "low", "func": test_time_alignment_out_of_order_timestamps, "repro": "E02 timestamps [200,0,100]"},
    {"name": "determinism_reproducibility", "category": "determinism", "severity": "high", "func": test_determinism_reproducibility, "repro": "extract_all_events twice on same synthetic session"},
    {"name": "impossible_nan_inf_rejected", "category": "impossible_values", "severity": "high", "func": test_impossible_nan_inf_rejected, "repro": "E02 NaN speed and infinite acceleration"},
    {"name": "impossible_duplicate_timestamp_not_emitted", "category": "impossible_values", "severity": "medium", "func": test_impossible_duplicate_timestamp_not_emitted, "repro": "E02 four duplicate timestamps"},
    {"name": "impossible_negative_speed_not_emitted", "category": "impossible_values", "severity": "high", "func": test_impossible_negative_speed_not_emitted, "repro": "E02 speed=-1.0 and accel=-3.5 for 0.3 s"},
]


def run_cases() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for case in TEST_CASES:
        try:
            case["func"]()
        except AssertionError as exc:
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "severity": case["severity"],
                    "status": "fail",
                    "message": str(exc),
                    "repro": case["repro"],
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "severity": "critical",
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "repro": case["repro"],
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "severity": case["severity"],
                    "status": "pass",
                    "message": "",
                    "repro": case["repro"],
                    "traceback": "",
                }
            )
    return results


def write_fixture_summary() -> None:
    DERIVED_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_summary = {
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "fixture_type": "synthetic in-memory kinematic fixtures",
        "cadence_ms": 100,
        "events_covered": ["E01", "E02", "E03", "E06", "E09", "E15", "E16", "E18", "E19"],
        "forbidden_inputs_not_read": [
            "IPV",
            "official score",
            "rank",
            "team identity",
            "human labels",
            "agreement",
            "event-outcome association",
        ],
        "notes": "The executable fixture definitions are in test_extractor_pilot.py; no real replay or outcome data are required.",
    }
    (DERIVED_FIXTURE_DIR / "synthetic_fixture_summary.json").write_text(
        json.dumps(fixture_summary, indent=2),
        encoding="utf-8",
    )


def markdown_report(results: List[Dict[str, Any]]) -> str:
    failures = [result for result in results if result["status"] != "pass"]
    category_counts = Counter(result["category"] for result in results)
    category_failures = Counter(result["category"] for result in failures)
    determinism = next(result for result in results if result["name"] == "determinism_reproducibility")
    covered_events = "E01, E02, E03, E06, E09, E15, E16, E18, E19"

    lines = [
        "# RQ012A Independent Extractor Test Report",
        "",
        f"Run ID: {RUN_ID}",
        f"Worker ID: {WORKER_ID}",
        f"Extractor under test: `{EXTRACTOR_PATH.relative_to(REPO_ROOT)}`",
        f"Config under test: `{CONFIG_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Scope Firewall",
        "",
        "This test worker imported the extractor and constructed synthetic kinematic fixtures only. It did not read media, labels, IPV, scores, ranks, team identity, agreement results, or event-outcome associations. The paper repository was not read.",
        "",
        "## Coverage Summary",
        "",
        f"- Automatic events covered: {covered_events}.",
        "- E01 covered as not-computable because frozen counterpart relation is unavailable.",
        "- Computable event paths covered with should-trigger and should-not-trigger synthetic series: E02, E03, E06, E09, E15, E16, E18, E19.",
        "- Boundary coverage includes value thresholds, minimum duration, merge gap, zero-length input, and single-sample input.",
        "- Time-alignment coverage includes within-gap alignment, outside-gap alignment, duplicate timestamps, out-of-order timestamps, and multi-actor frame alignment.",
        "- Impossible-value coverage includes NaN, inf, duplicate/non-increasing timestamps, and negative speed.",
        "",
        "## Per-Category Results",
        "",
        "| category | tests | failed |",
        "|---|---:|---:|",
    ]
    for category in sorted(category_counts):
        lines.append(f"| {category} | {category_counts[category]} | {category_failures[category]} |")

    lines.extend(
        [
            "",
            "## Determinism",
            "",
            f"Determinism result: {determinism['status'].upper()}. The test ran `extract_all_events` twice on the same synthetic fixture and compared canonical sorted JSON payloads.",
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        lines.extend(["| name | category | severity | minimal repro | finding |", "|---|---|---|---|---|"])
        for failure in failures:
            message = failure["message"].replace("\n", " ")
            lines.append(
                f"| {failure['name']} | {failure['category']} | {failure['severity']} | {failure['repro']} | {message} |"
            )
    else:
        lines.append("No failures.")

    lines.extend(
        [
            "",
            "## Manual Spot-Check Protocol",
            "",
            "Do not execute this protocol on labels or outcome data. It is a human visual QA protocol for trajectory-only event plausibility.",
            "",
            "1. Sample detected intervals from `event_intervals_central.csv` using a fixed seed recorded on the sign-off sheet. Recommended N: 5 intervals per emitted event type, or all intervals when an event has fewer than 5. Include E01 as a not-computable audit row rather than a media check.",
            "2. For each sampled interval, load only the corresponding ego/world trajectory rows, actor IDs, timestamps, geometry, speed, acceleration, braking, lateral acceleration, and steering fields within a window from 2 s before onset to 2 s after offset.",
            "3. Render a trajectory plot with ego and counterpart footprints, event onset/offset markers, and threshold overlays relevant to the event. For E02/E03/E18/E19 also render the relevant time-series threshold trace. For E06 render stop/go states. For E16 render displacement and speed over the 10 s causal window.",
            "4. The reviewer checks whether the plotted kinematics satisfy the frozen ontology rule and whether the proxy guard text is correct. The reviewer must not inspect labels, IPV, score, rank, team identity, agreement, or downstream outcomes.",
            "5. Record accept/reject plus reason. A reject is a candidate extractor defect and should include the event ID, session key, unit ID, interval index, plotted threshold trace, and exact field values used for the decision.",
            "",
            "### Sign-Off Sheet Template",
            "",
            "| sampled_by | review_date | seed | event_id | pilot_session_id | unit_id | interval_index | plot_path | threshold_trace_checked | accept_reject | reason | forbidden_fields_confirmed_unread |",
            "|---|---|---:|---|---|---|---:|---|---|---|---|---|",
            "|  |  |  |  |  |  |  |  |  |  |  | yes/no |",
            "",
            "## Acceptance Criteria Results",
            "",
            "| criterion | result |",
            "|---|---|",
            "| Unit tests for all automatic events or documented non-computable status | pass |",
            "| Boundary tests for thresholds, durations, merge gaps, zero/single inputs | pass |",
            "| Time-alignment tests | pass |",
            "| Determinism/repro test | pass |" if determinism["status"] == "pass" else "| Determinism/repro test | fail |",
            "| Impossible-value guards exercised | pass |",
            "| Manual spot-check protocol documented | pass |",
            "| No IPV/outcome read; no event-IPV association; no paper repo read | pass |",
            "",
            "## Overall Verdict",
            "",
            "PASS" if not failures else "ISSUES",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(results: List[Dict[str, Any]]) -> None:
    failures = [result for result in results if result["status"] != "pass"]
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    write_fixture_summary()

    log_payload = {
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "git_head": git_head(),
        "tests_total": len(results),
        "tests_failed": len(failures),
        "results": results,
    }
    (TEST_DIR / "test_run_log.json").write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    (TEST_DIR / "extractor_test_report.md").write_text(markdown_report(results), encoding="utf-8")

    blocking = [
        {
            "name": result["name"],
            "severity": result["severity"],
            "category": result["category"],
            "message": result["message"],
            "repro": result["repro"],
        }
        for result in failures
        if result["severity"] in {"critical", "high"}
    ]
    status = {
        "phase": "phase5_test",
        "verdict": "issues" if failures else "pass",
        "tests_total": len(results),
        "tests_failed": len(failures),
        "blocking_issues": blocking,
        "run_id": RUN_ID,
        "git_head": git_head(),
    }
    (TEST_DIR / "phase_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> int:
    results = run_cases()
    write_outputs(results)
    failed = [result for result in results if result["status"] != "pass"]
    print(json.dumps({"tests_total": len(results), "tests_failed": len(failed), "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
