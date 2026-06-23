#!/usr/bin/env python3
"""Regression tests for RQ012A extractor robustness fixes.

The fixtures are synthetic trajectory rows only. They do not read labels, IPV,
scores, ranks, team identity, agreement files, outcomes, or the paper repo.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence


RUN_ID = "RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37"
WORKER_ID = "RQ012-W17a-extractor-robustness"
TEST_DIR = Path(__file__).resolve().parent
PILOT_DIR = TEST_DIR.parent
RUN_ROOT = TEST_DIR.parents[2]
REPO_ROOT = TEST_DIR.parents[6]
EXTRACTOR_PATH = PILOT_DIR / "extractor_pilot.py"
CONFIG_PATH = PILOT_DIR / "extractor_config.json"


def load_extractor() -> Any:
    spec = importlib.util.spec_from_file_location("extractor_pilot_robustness_under_test", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extractor from {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ext = load_extractor()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CENTRAL = CONFIG["bands"]["central"]
HIGH = CONFIG["bands"]["high"]
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


def times(count: int, start_ms: float = 0.0, step_ms: float = 100.0) -> List[float]:
    return [start_ms + idx * step_ms for idx in range(count)]


def value_at(value: Any, idx: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[idx]
    return value


def row(
    t_ms: float,
    *,
    actor_id: str = "ego",
    source: str = "ego",
    origin_id: str | None = None,
    name: str | None = None,
    speed: float = 0.0,
    accel: float = 0.0,
    braking: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    length: float = 4.0,
    width: float = 2.0,
    course_deg: float = 0.0,
) -> Dict[str, Any]:
    return {
        "source": source,
        "actor_id": actor_id,
        "origin_id": origin_id if source == "world" else None,
        "name": name if source == "world" else None,
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
        "lat_acc": 0.0,
        "lon_acc": accel,
        "steering_angle": 0.0,
        "wheel_angle": 0.0,
    }


def ego_rows(t_values: Sequence[float], **kwargs: Any) -> List[Dict[str, Any]]:
    return [
        row(
            t,
            source="ego",
            actor_id="ego",
            speed=value_at(kwargs.get("speed", 0.0), idx),
            accel=value_at(kwargs.get("accel", 0.0), idx),
            braking=value_at(kwargs.get("braking", 0.0), idx),
            x=value_at(kwargs.get("x", 0.0), idx),
            y=value_at(kwargs.get("y", 0.0), idx),
            course_deg=value_at(kwargs.get("course_deg", 0.0), idx),
        )
        for idx, t in enumerate(t_values)
    ]


def world_rows(actor_id: str, t_values: Sequence[float], **kwargs: Any) -> List[Dict[str, Any]]:
    return [
        row(
            t,
            source="world",
            actor_id=actor_id,
            origin_id=value_at(kwargs.get("origin_id", actor_id), idx),
            name=value_at(kwargs.get("name", "vehicle"), idx),
            speed=value_at(kwargs.get("speed", 0.0), idx),
            accel=value_at(kwargs.get("accel", 0.0), idx),
            x=value_at(kwargs.get("x", 4.4), idx),
            y=value_at(kwargs.get("y", 0.0), idx),
            course_deg=value_at(kwargs.get("course_deg", 0.0), idx),
        )
        for idx, t in enumerate(t_values)
    ]


def make_session(name: str, ego: Sequence[Dict[str, Any]], world: Dict[str, Sequence[Dict[str, Any]]]) -> Any:
    ref = ext.SessionRef(
        session_key=f"synthetic_{name}",
        pilot_session_id=f"synthetic_{name}",
        session_dir=TEST_DIR,
        relative_dir="synthetic",
        duplicate_source_count=1,
    )
    return ext.SessionData(
        ref=ref,
        ego=list(ego),
        world_by_actor={key: list(value) for key, value in world.items()},
        health={},
    )


def pair_session(name: str, clearances: Sequence[float], *, band_t_ms: Sequence[float] | None = None) -> Any:
    t_values = list(band_t_ms if band_t_ms is not None else times(len(clearances)))
    ego = ego_rows(t_values, speed=0.0, x=0.0, y=0.0)
    other_x = [4.0 + gap for gap in clearances]
    other = world_rows("actor_a", t_values, x=other_x, speed=0.0)
    return make_session(name, ego=ego, world={"actor_a": other})


def primary_count(result: Any) -> int:
    return int(getattr(result, "primary_event_count", result.event_count))


def test_e09_e15_duplicate_ego_timestamp_rejected() -> None:
    ego = ego_rows([0, 0, 100], speed=0.0, x=[0.0, 100.0, 0.0], y=0.0)
    other = world_rows("actor_a", [0, 100], x=[4.4, 4.4], speed=0.0)
    session = make_session("duplicate_ego_pair_time", ego=ego, world={"actor_a": other})
    result = ext.extract_pair_event(session, CENTRAL["E09"], "E09", 1, MAX_ALIGN_GAP_S)
    assert primary_count(result) == 0, (
        "Duplicate ego timestamps in pair extraction must not emit a primary E09 interval; "
        f"got event_count={result.event_count}, intervals={result.intervals}"
    )
    assert result.impossible_values > 0, "Pair-event duplicate ego timestamps must increment impossible_values"


def test_tied_nearest_neighbor_uses_lower_timestamp_deterministically() -> None:
    ego = ego_rows([0, 100], speed=0.0, x=[0.0, 100.0], y=0.0)
    ego_times, ego_sorted = ext.nearest_ego_rows(ego)
    picks = [ext.nearest_by_time(ego_times, ego_sorted, 0.05, MAX_ALIGN_GAP_S) for _ in range(5)]
    picked_times = [pick["t_ms"] if pick else None for pick in picks]
    assert picked_times == [0, 0, 0, 0, 0], f"Tied nearest-neighbor selection should choose the lower timestamp deterministically, got {picked_times}"


def test_actor_identity_change_splits_and_blocks_primary_pair_interval() -> None:
    ego = ego_rows([0, 100, 200, 300], speed=0.0, x=0.0, y=0.0)
    other = world_rows(
        "actor_a",
        [0, 100, 200, 300],
        x=[4.4, 4.4, 4.4, 4.4],
        speed=0.0,
        origin_id=["origin_1", "origin_1", "origin_2", "origin_2"],
        name=["car_1", "car_1", "car_2", "car_2"],
    )
    session = make_session("identity_change_pair", ego=ego, world={"actor_a": other})
    result = ext.extract_pair_event(session, CENTRAL["E09"], "E09", 1, MAX_ALIGN_GAP_S)
    assert primary_count(result) == 0, (
        "Pair intervals from a same-id actor with origin/name identity changes must be non-primary; "
        f"got event_count={result.event_count}, intervals={result.intervals}"
    )
    assert result.actor_attribution_failures >= 2, "Identity-change windows should be counted as actor-attribution failures"


def test_high_band_e15_suppresses_same_interval_e09_primary_count() -> None:
    session = pair_session("high_contact_vs_near_miss", [0.05, 0.05, 0.05])
    results = ext.extract_all_events([session], HIGH, decimate_factor=1, max_align_gap_s=MAX_ALIGN_GAP_S)
    assert primary_count(results["E15"]) == 1, f"Expected one high-band E15 contact candidate, got {results['E15']}"
    assert primary_count(results["E09"]) == 0, (
        "High-band contact within E15 tolerance must suppress same-pair/time E09 as a primary endpoint; "
        f"E09 intervals={results['E09'].intervals}"
    )


def test_e02_e18_ego_hard_stop_precedence_deoverlaps_primary_counts() -> None:
    ego = ego_rows([0, 100, 200], speed=0.2, accel=-5.0, braking=1.0, x=0.0, y=0.0)
    session = make_session("ego_hard_stop_precedence", ego=ego, world={})
    results = ext.extract_all_events([session], CENTRAL, decimate_factor=1, max_align_gap_s=MAX_ALIGN_GAP_S)
    assert results["E02"].raw_event_count == 1, f"E02 raw diagnostic count should remain 1, got {results['E02'].raw_event_count}"
    assert primary_count(results["E18"]) == 1, f"E18 should keep the ego hard-stop primary endpoint, got {results['E18']}"
    assert primary_count(results["E02"]) == 0, (
        "E02 ego deceleration overlapping E18 hard-stop must be de-overlapped from primary endpoint counts; "
        f"E02 intervals={results['E02'].intervals}"
    )


TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "e09_e15_duplicate_ego_timestamp_rejected",
        "finding": "V03",
        "func": test_e09_e15_duplicate_ego_timestamp_rejected,
    },
    {
        "name": "tied_nearest_neighbor_uses_lower_timestamp_deterministically",
        "finding": "V03",
        "func": test_tied_nearest_neighbor_uses_lower_timestamp_deterministically,
    },
    {
        "name": "actor_identity_change_splits_and_blocks_primary_pair_interval",
        "finding": "V04",
        "func": test_actor_identity_change_splits_and_blocks_primary_pair_interval,
    },
    {
        "name": "high_band_e15_suppresses_same_interval_e09_primary_count",
        "finding": "V05",
        "func": test_high_band_e15_suppresses_same_interval_e09_primary_count,
    },
    {
        "name": "e02_e18_ego_hard_stop_precedence_deoverlaps_primary_counts",
        "finding": "V01",
        "func": test_e02_e18_ego_hard_stop_precedence_deoverlaps_primary_counts,
    },
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
                    "finding": case["finding"],
                    "status": "fail",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case["name"],
                    "finding": case["finding"],
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            results.append(
                {
                    "name": case["name"],
                    "finding": case["finding"],
                    "status": "pass",
                    "message": "",
                    "traceback": "",
                }
            )
    return results


def write_outputs(results: List[Dict[str, Any]]) -> None:
    failed = [result for result in results if result["status"] != "pass"]
    payload = {
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "git_head": git_head(),
        "tests_total": len(results),
        "tests_failed": len(failed),
        "scope_firewall": "Synthetic trajectory-only fixtures; no labels/IPV/outcomes/team identity/agreement/paper repo read.",
        "results": results,
    }
    (TEST_DIR / "robustness_test_run_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# RQ012A Extractor Robustness Regression Report",
        "",
        f"Run ID: {RUN_ID}",
        f"Worker ID: {WORKER_ID}",
        f"Git HEAD: {payload['git_head']}",
        "",
        "## Scope Firewall",
        "",
        "Synthetic trajectory-only fixtures; no labels, IPV, scores, ranks, team identity, agreement files, outcomes, or paper repo content were read.",
        "",
        "## Results",
        "",
        "| finding | test | status | message |",
        "|---|---|---|---|",
    ]
    for result in results:
        message = result["message"].replace("\n", " ")
        lines.append(f"| {result['finding']} | {result['name']} | {result['status']} | {message} |")
    lines.extend(["", "Overall: " + ("PASS" if not failed else "FAIL"), ""])
    (TEST_DIR / "extractor_robustness_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results = run_cases()
    write_outputs(results)
    failed = [result for result in results if result["status"] != "pass"]
    print(json.dumps({"tests_total": len(results), "tests_failed": len(failed), "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
