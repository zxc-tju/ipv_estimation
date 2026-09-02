#!/usr/bin/env python3
"""Generate an explicitly synthetic 20 x 15 Shanghai human-driving proxy dataset.

The raw replay layout mirrors the OnSite AV package. Background traffic is copied
unchanged from one Shanghai AV session, while the ego is re-timed along the same
polyline. Aggregate-constrained annotation columns reproduce the published RQ022
margins but are not outputs of the frozen IPV estimator and are never evidence of
observed human behaviour.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SESSION = (
    REPO_ROOT
    / "data/onsite_competition/all_teams_dataset/teams/shanghai/01_T11_wsd/"
    "sessions/6923-1766197775"
)
DEFAULT_TARGETS = REPO_ROOT / "reports/plans/RQ029_human_statistical_targets_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/derived/rq029_human_synthetic_microdata/v1"
DATA_STATUS = "SYNTHETIC_NOT_OBSERVED"
SCHEMA_VERSION = "RQ029-human-synthetic-microdata-v1"

CASE_TO_SCENARIO_SHANGHAI = {
    2325: "A1",
    2328: "A2",
    2323: "A3",
    2322: "A4",
    2321: "A5",
    2327: "A6",
    2319: "A7",
    2318: "B1",
    2317: "B2",
    2316: "B3",
    2315: "B4",
    2314: "C1",
    2313: "C2",
    2346: "C3",
    2311: "C4",
}
SCENARIOS = tuple(CASE_TO_SCENARIO_SHANGHAI.values())
REQUIRED_LOGS = (
    "monitor.log",
    "simulation_trajectory.log",
    "vehicle_perception_simulation_trajectory.log",
    "vehicle_trajectory.log",
)
OPTIONAL_LOG = "vehicle_perception_trajectory.log"


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_allocate(
    total: int,
    weights: np.ndarray,
    capacities: np.ndarray,
    minimums: np.ndarray | None = None,
) -> np.ndarray:
    """Largest-remainder-like allocation with integer lower and upper bounds."""

    weights = np.asarray(weights, dtype=float)
    capacities = np.asarray(capacities, dtype=int)
    minimums = (
        np.zeros(len(capacities), dtype=int)
        if minimums is None
        else np.asarray(minimums, dtype=int)
    )
    if not (len(weights) == len(capacities) == len(minimums)):
        raise ValueError("allocation vectors must have equal length")
    if np.any(minimums < 0) or np.any(capacities < minimums):
        raise ValueError("invalid allocation bounds")
    if not int(minimums.sum()) <= total <= int(capacities.sum()):
        raise ValueError("requested total is outside allocation bounds")
    out = minimums.copy()
    remaining = int(total - out.sum())
    while remaining:
        room = capacities - out
        available = room > 0
        effective = np.where(available, np.maximum(weights, 0.0), 0.0)
        if effective.sum() == 0:
            effective = available.astype(float)
        quota = remaining * effective / effective.sum()
        increment = np.minimum(np.floor(quota).astype(int), room)
        if increment.sum() == 0:
            allocated_extra = out - minimums
            priority = np.where(
                available,
                effective / (allocated_extra + 1.0),
                -1.0,
            )
            best = int(np.argmax(priority))
            increment[best] = 1
        out += increment
        remaining = int(total - out.sum())
    return out


def _calibrated_quantile_values(
    n: int,
    q25: float,
    q50: float,
    q75: float,
    below_threshold_count: int = 0,
    threshold: float = 0.0,
) -> np.ndarray:
    """Create monotone values whose linear-interpolated quartiles are exact."""

    if n < 5 or not 0 <= below_threshold_count < n:
        raise ValueError("invalid calibrated sample size or threshold count")
    if not q25 <= q50 <= q75:
        raise ValueError("quartiles must be ordered")
    ranks = np.arange(n, dtype=float) / (n - 1)
    knots_x = [0.0]
    low = min(q25 * 0.45, threshold * 0.45) if below_threshold_count else q25 * 0.45
    knots_y = [max(0.0, low)]
    if below_threshold_count:
        left = (below_threshold_count - 1) / (n - 1)
        right = below_threshold_count / (n - 1)
        if right >= 0.25:
            raise ValueError("threshold count must lie below the first quartile")
        knots_x.extend([left, right])
        knots_y.extend([threshold - 0.1, threshold + 0.05])
    knots_x.extend([0.25, 0.5, 0.75, 1.0])
    knots_y.extend([q25, q50, q75, max(q75 * 1.8, q75 + 1.0)])
    values = np.interp(ranks, np.asarray(knots_x), np.asarray(knots_y))
    # A requested quantile can fall between two ranks that straddle a knot with
    # different left/right slopes. Pin both bracketing order statistics so
    # NumPy/Pandas linear interpolation returns the requested value exactly.
    for quantile, target in ((0.25, q25), (0.50, q50), (0.75, q75)):
        position = (n - 1) * quantile
        values[int(math.floor(position))] = target
        values[int(math.ceil(position))] = target
    return values


def _ego_location(record: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    found: list[tuple[int, int, dict[str, Any]]] = []
    for group_index, group in enumerate(record.get("participantTrajectories", [])):
        for value_index, actor in enumerate(group.get("value", [])):
            if actor.get("isPerception") == 0:
                found.append((group_index, value_index, actor))
    if len(found) != 1:
        raise ValueError(f"expected exactly one ego object, found {len(found)}")
    return found[0]


def _semantic_background_payload(record: dict[str, Any]) -> str:
    """Canonical JSON with only the unique ego object removed."""

    background = copy.deepcopy(record)
    for group in background.get("participantTrajectories", []):
        group["value"] = [
            actor for actor in group.get("value", []) if actor.get("isPerception") != 0
        ]
    return json.dumps(background, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _local_xy(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(latitude[0])
    lon0 = float(longitude[0])
    y = (latitude - lat0) * 111_320.0
    x = (longitude - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return x, y


def _warp_ego_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    driver_id: str,
    scenario_id: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Re-time an ego along its source polyline without lateral displacement."""

    if len(rows) < 3:
        raise ValueError("a scenario needs at least three ego frames")
    latitude = np.asarray([float(row["latitude"]) for row in rows])
    longitude = np.asarray([float(row["longitude"]) for row in rows])
    base_speed = np.asarray([float(row.get("speed", 0.0)) for row in rows])
    timestamps = np.asarray([int(row["globalTimeStamp"]) for row in rows], dtype=float)
    x, y = _local_xy(latitude, longitude)
    step_distance = np.hypot(np.diff(x), np.diff(y))
    cumulative = np.concatenate([[0.0], np.cumsum(step_distance)])
    total_distance = float(cumulative[-1])

    rng = np.random.default_rng(_stable_seed(seed, driver_id, scenario_id))
    u = np.linspace(0.0, 1.0, len(step_distance), endpoint=True)
    phase1, phase2 = rng.uniform(0.0, 2 * np.pi, size=2)
    amplitude1 = rng.uniform(0.025, 0.045)
    amplitude2 = rng.uniform(0.010, 0.022)
    pace = 1.0 + amplitude1 * np.sin(2 * np.pi * u + phase1)
    pace += amplitude2 * np.sin(4 * np.pi * u + phase2)
    pace = np.clip(pace, 0.90, 1.10)
    if total_distance > 1e-6:
        weighted = step_distance * pace
        if weighted.sum() > 0:
            pace *= total_distance / float(weighted.sum())
        warped_cumulative = np.concatenate([[0.0], np.cumsum(step_distance * pace)])
        warped_cumulative[-1] = total_distance
    else:
        warped_cumulative = cumulative.copy()

    unique = np.concatenate([[True], np.diff(cumulative) > 1e-9])
    unique[-1] = True
    base_s = cumulative[unique]
    base_lat = latitude[unique]
    base_lon = longitude[unique]
    if len(base_s) < 2:
        warped_lat = latitude.copy()
        warped_lon = longitude.copy()
    else:
        warped_lat = np.interp(warped_cumulative, base_s, base_lat)
        warped_lon = np.interp(warped_cumulative, base_s, base_lon)
    warped_lat[0], warped_lat[-1] = latitude[0], latitude[-1]
    warped_lon[0], warped_lon[-1] = longitude[0], longitude[-1]

    frame_pace = np.empty(len(rows), dtype=float)
    frame_pace[0] = pace[0]
    frame_pace[-1] = pace[-1]
    if len(rows) > 2:
        frame_pace[1:-1] = 0.5 * (pace[:-1] + pace[1:])
    warped_speed = np.maximum(0.0, base_speed * frame_pace)

    source_heading = np.asarray([float(row.get("courseAngle", 0.0)) for row in rows])
    unwrapped_heading = np.unwrap(np.radians(source_heading))
    heading_s = cumulative[unique]
    heading_values = unwrapped_heading[unique]
    if len(heading_s) >= 2:
        warped_heading = np.degrees(np.interp(warped_cumulative, heading_s, heading_values)) % 360
    else:
        warped_heading = source_heading.copy()

    time_s = (timestamps - timestamps[0]) / 1000.0
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"non-monotone timestamps in {scenario_id}")
    acceleration = np.gradient(warped_speed / 3.6, time_s)

    warped: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = copy.deepcopy(source)
        row["latitude"] = float(warped_lat[index])
        row["longitude"] = float(warped_lon[index])
        row["speed"] = float(warped_speed[index])
        row["courseAngle"] = float(warped_heading[index])
        if "name" in row:
            row["name"] = f"SYNTHETIC_HUMAN_{driver_id}"
        for key in ("acc", "acceleration", "acceleration ", "lonAcc"):
            if key in row:
                row[key] = float(acceleration[index])
        warped.append(row)

    valid_speed = base_speed > 1.0
    relative = (
        np.abs(warped_speed[valid_speed] / base_speed[valid_speed] - 1.0)
        if valid_speed.any()
        else np.asarray([0.0])
    )
    metrics = {
        "distance_m": total_distance,
        "max_cross_track_m": 0.0,
        "speed_relative_abs_mean": float(np.mean(relative)),
        "speed_relative_abs_p95": float(np.quantile(relative, 0.95)),
        "speed_relative_abs_max": float(np.max(relative)),
    }
    return warped, metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _actor_value(actor: dict[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = actor.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _nearest_background(record: dict[str, Any], ego: dict[str, Any]) -> dict[str, Any] | None:
    ego_lat = _actor_value(ego, ("latitude", "lat"))
    ego_lon = _actor_value(ego, ("longitude", "lng", "lon"))
    if ego_lat is None or ego_lon is None:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for group in record.get("participantTrajectories", []):
        if group.get("role") != "mvSimulation":
            continue
        for actor in group.get("value", []):
            lat = _actor_value(actor, ("latitude", "lat"))
            lon = _actor_value(actor, ("longitude", "lng", "lon"))
            if lat is None or lon is None:
                continue
            dy = (lat - ego_lat) * 111_320.0
            dx = (lon - ego_lon) * 111_320.0 * math.cos(math.radians(ego_lat))
            distance = math.hypot(dx, dy)
            if best is None or distance < best[0]:
                best = (distance, actor)
    if best is None:
        return None
    actor = best[1]
    return {
        "id": str(actor.get("id", "")),
        "latitude": _actor_value(actor, ("latitude", "lat")),
        "longitude": _actor_value(actor, ("longitude", "lng", "lon")),
        "speed_kmh": _actor_value(actor, ("speed",)),
        "acceleration_mps2": _actor_value(actor, ("acceleration", "acceleration ", "acc")),
        "distance_m": float(best[0]),
    }


def _velocity(speed_kmh: float | None, course_deg: float | None) -> tuple[float, float]:
    if speed_kmh is None or course_deg is None:
        return float("nan"), float("nan")
    speed_mps = speed_kmh / 3.6
    angle = math.radians(course_deg)
    return speed_mps * math.sin(angle), speed_mps * math.cos(angle)


def _candidate_positions(n_rows: int, n_select: int) -> np.ndarray:
    if not 0 <= n_select <= n_rows:
        raise ValueError("candidate selection exceeds available frames")
    if n_select == 0:
        return np.asarray([], dtype=int)
    positions = np.floor((np.arange(n_select) + 0.5) * n_rows / n_select).astype(int)
    if len(np.unique(positions)) != n_select:
        raise AssertionError("uniform candidate selector produced duplicates")
    return positions


def _build_annotations(
    trajectories: pd.DataFrame,
    targets: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unit_keys = [
        (driver, scenario)
        for scenario in SCENARIOS
        for driver in sorted(trajectories["driver_id"].unique())
    ]
    raw_counts = (
        trajectories.groupby(["driver_id", "scenario_id"]).size().to_dict()
    )
    scenario_cap = np.asarray(
        [sum(raw_counts[(driver, scenario)] for driver in sorted(trajectories.driver_id.unique()))
         for scenario in SCENARIOS],
        dtype=int,
    )
    scenario_min = np.asarray(
        [targets["per_scenario_alpha90"][scenario]["n_both"] for scenario in SCENARIOS],
        dtype=int,
    )
    scenario_candidates = _bounded_allocate(
        targets["gates"]["n_candidate_moments"],
        scenario_cap.astype(float),
        scenario_cap,
        scenario_min,
    )

    candidate_quota: dict[tuple[str, str], int] = {}
    both_quota: dict[tuple[str, str], int] = {}
    drivers = sorted(trajectories.driver_id.unique())
    for scenario_index, scenario in enumerate(SCENARIOS):
        capacities = np.asarray([raw_counts[(driver, scenario)] for driver in drivers])
        order = np.roll(np.arange(len(drivers)), -scenario_index)
        candidate_counts_ordered = _bounded_allocate(
            int(scenario_candidates[scenario_index]),
            np.ones(len(drivers)),
            capacities[order],
        )
        candidate_counts = np.empty_like(candidate_counts_ordered)
        candidate_counts[order] = candidate_counts_ordered
        both_counts_ordered = _bounded_allocate(
            int(targets["per_scenario_alpha90"][scenario]["n_both"]),
            np.ones(len(drivers)),
            candidate_counts[order],
        )
        both_counts = np.empty_like(both_counts_ordered)
        both_counts[order] = both_counts_ordered
        for driver_index, driver in enumerate(drivers):
            candidate_quota[(driver, scenario)] = int(candidate_counts[driver_index])
            both_quota[(driver, scenario)] = int(both_counts[driver_index])

    selected_parts: list[pd.DataFrame] = []
    for driver, scenario in unit_keys:
        group = trajectories[
            (trajectories.driver_id == driver) & (trajectories.scenario_id == scenario)
        ].sort_values("scenario_frame_index")
        positions = _candidate_positions(len(group), candidate_quota[(driver, scenario)])
        selected_parts.append(group.iloc[positions].copy())
    candidates = pd.concat(selected_parts, ignore_index=True)
    candidates["candidate_key"] = candidates.apply(
        lambda row: (
            f"synthetic:shanghai:{row.driver_id}:{row.scenario_id}:"
            f"frame:{int(row.scenario_frame_index)}"
        ),
        axis=1,
    )
    candidates["status"] = "ABSTAIN"
    candidates["reason_code"] = "NEAR_UNIFORM"
    candidates["mechanism2_gate_ok"] = False

    both_counts_array = np.asarray([both_quota[key] for key in unit_keys], dtype=int)
    candidate_counts_array = np.asarray([candidate_quota[key] for key in unit_keys], dtype=int)
    gate_order = np.random.default_rng(_stable_seed(seed, "gate1_allocation")).permutation(
        len(unit_keys)
    )
    gate1_extra_ordered = _bounded_allocate(
        targets["gates"]["n_gate1_pass"] - targets["gates"]["n_both_gates"],
        (candidate_counts_array - both_counts_array)[gate_order].astype(float),
        (candidate_counts_array - both_counts_array)[gate_order],
    )
    gate1_extra = np.empty_like(gate1_extra_ordered)
    gate1_extra[gate_order] = gate1_extra_ordered
    for unit_index, (driver, scenario) in enumerate(unit_keys):
        mask = (candidates.driver_id == driver) & (candidates.scenario_id == scenario)
        indices = candidates.index[mask].to_numpy()
        rng = np.random.default_rng(_stable_seed(seed, driver, scenario, "gates"))
        indices = indices[rng.permutation(len(indices))]
        n_both = both_quota[(driver, scenario)]
        n_gate1 = n_both + int(gate1_extra[unit_index])
        candidates.loc[indices[:n_gate1], ["status", "reason_code"]] = ["OK", None]
        candidates.loc[indices[:n_both], "mechanism2_gate_ok"] = True

    both_mask = (candidates.status == "OK") & candidates.mechanism2_gate_ok
    both_indices = candidates.index[both_mask].to_numpy()
    candidates["band_80"] = pd.NA
    candidates["band_90"] = pd.NA
    candidates["band_95"] = pd.NA
    candidates.loc[both_indices, ["band_80", "band_90", "band_95"]] = "inside"

    scenario_flagged = np.asarray(
        [targets["per_scenario_alpha90"][scenario]["n_flagged"] for scenario in SCENARIOS],
        dtype=int,
    )
    scenario_lower = _bounded_allocate(
        targets["flag_counts"]["90"]["n_below"],
        scenario_flagged.astype(float),
        scenario_flagged,
    )
    for scenario_index, scenario in enumerate(SCENARIOS):
        eligible = candidates.index[both_mask & (candidates.scenario_id == scenario)].to_numpy()
        rng = np.random.default_rng(_stable_seed(seed, scenario, "alpha90"))
        # Aggregate statistics do not identify driver random effects. Use the
        # maximum-entropy choice here: exchangeable rows within each scenario.
        score = rng.random(len(eligible))
        flagged = eligible[np.argsort(score)[::-1][: int(scenario_flagged[scenario_index])]]
        flagged = flagged[rng.permutation(len(flagged))]
        n_lower = int(scenario_lower[scenario_index])
        candidates.loc[flagged[:n_lower], "band_90"] = "lower"
        candidates.loc[flagged[n_lower:], "band_90"] = "upper"

    lower90 = candidates.index[candidates.band_90 == "lower"].to_numpy()
    upper90 = candidates.index[candidates.band_90 == "upper"].to_numpy()
    inside90 = candidates.index[candidates.band_90 == "inside"].to_numpy()
    rng = np.random.default_rng(_stable_seed(seed, "nested_bands"))
    lower95 = rng.permutation(lower90)[: targets["flag_counts"]["95"]["n_below"]]
    upper95 = rng.permutation(upper90)[: targets["flag_counts"]["95"]["n_above"]]
    candidates.loc[lower90, "band_80"] = "lower"
    candidates.loc[upper90, "band_80"] = "upper"
    candidates.loc[lower95, "band_95"] = "lower"
    candidates.loc[upper95, "band_95"] = "upper"
    inside_shuffle = rng.permutation(inside90)
    extra_lower = targets["flag_counts"]["80"]["n_below"] - len(lower90)
    extra_upper = targets["flag_counts"]["80"]["n_above"] - len(upper90)
    candidates.loc[inside_shuffle[:extra_lower], "band_80"] = "lower"
    candidates.loc[inside_shuffle[extra_lower: extra_lower + extra_upper], "band_80"] = "upper"

    candidates["lo_80"] = -0.4
    candidates["hi_80"] = 0.4
    candidates["lo_90"] = -0.6
    candidates["hi_90"] = 0.6
    candidates["lo_95"] = -0.8
    candidates["hi_95"] = 0.8
    candidates["ipv_log"] = np.nan
    candidates.loc[candidates.status == "OK", "ipv_log"] = 0.0
    candidates.loc[candidates.band_80 == "lower", "ipv_log"] = -0.5
    candidates.loc[candidates.band_80 == "upper", "ipv_log"] = 0.5
    candidates.loc[candidates.band_90 == "lower", "ipv_log"] = -0.7
    candidates.loc[candidates.band_90 == "upper", "ipv_log"] = 0.7
    candidates.loc[candidates.band_95 == "lower", "ipv_log"] = -0.9
    candidates.loc[candidates.band_95 == "upper", "ipv_log"] = 0.9
    candidates["future_min_ttc_s_calibrated"] = np.nan

    ttc_target = targets["signature"]["ego_ttc"]
    for band in ("lower", "inside"):
        spec = ttc_target[band]
        eligible = candidates.index[candidates.band_90 == band].to_numpy()
        rng_band = np.random.default_rng(_stable_seed(seed, band, "ttc"))
        finite = eligible[
            np.isfinite(candidates.loc[eligible, "current_ttc_raw_s"].to_numpy(float))
        ]
        missing = np.setdiff1d(eligible, finite, assume_unique=False)
        ordered = np.concatenate([rng_band.permutation(finite), rng_band.permutation(missing)])
        selected = ordered[: spec["n"]]
        values = _calibrated_quantile_values(
            spec["n"], spec["q25"], spec["q50"], spec["q75"],
            spec["lt2_num"], 2.0,
        )
        raw = candidates.loc[selected, "current_ttc_raw_s"].to_numpy(float)
        rank_key = np.where(np.isfinite(raw), raw, np.inf)
        candidates.loc[selected[np.argsort(rank_key)], "future_min_ttc_s_calibrated"] = values

    cp_target = targets["signature"]["counterpart"]
    selected_by_band: dict[str, np.ndarray] = {}
    for band, n_select in (("lower", cp_target["n_lower"]), ("inside", cp_target["n_inside"])):
        eligible = candidates.index[candidates.band_90 == band].to_numpy()
        rng_band = np.random.default_rng(_stable_seed(seed, band, "counterpart"))
        available = eligible[
            np.isfinite(candidates.loc[eligible, "counterpart_speed_kmh"].to_numpy(float))
        ]
        unavailable = np.setdiff1d(eligible, available, assume_unique=False)
        selected_by_band[band] = np.concatenate(
            [rng_band.permutation(available), rng_band.permutation(unavailable)]
        )[:n_select]
    cp_indices = np.concatenate([selected_by_band["lower"], selected_by_band["inside"]])
    counterpart = candidates.loc[cp_indices, [
        "candidate_key", "driver_id", "scenario_id", "run_id", "scenario_frame_index",
        "band_90", "counterpart_id", "counterpart_speed_kmh",
        "counterpart_acceleration_mps2",
    ]].copy()
    counterpart.rename(
        columns={
            "counterpart_speed_kmh": "anchor_speed_kmh_raw",
            "counterpart_acceleration_mps2": "anchor_acceleration_mps2_raw",
        },
        inplace=True,
    )
    trajectory_by_run = {
        run_id: group.sort_values("scenario_frame_index").reset_index(drop=True)
        for run_id, group in trajectories.groupby("run_id")
    }
    raw_drop: list[float] = []
    raw_range: list[float] = []
    raw_frame_total: list[int] = []
    raw_brake_total: list[int] = []
    for row in counterpart.itertuples(index=False):
        group = trajectory_by_run[row.run_id]
        start = int(row.scenario_frame_index)
        window = group[
            group.scenario_frame_index.between(start, start + 30, inclusive="both")
        ]
        if pd.notna(row.counterpart_id):
            window = window[window.counterpart_id.astype(str) == str(row.counterpart_id)]
        else:
            window = window.iloc[0:0]
        speeds = window.counterpart_speed_kmh.dropna().to_numpy(float)
        accelerations = window.counterpart_acceleration_mps2.dropna().to_numpy(float)
        if len(speeds) and pd.notna(row.anchor_speed_kmh_raw):
            raw_drop.append(max(0.0, float(row.anchor_speed_kmh_raw) - float(speeds.min())))
            raw_range.append(float(speeds.max() - speeds.min()))
        else:
            raw_drop.append(float("nan"))
            raw_range.append(float("nan"))
        raw_frame_total.append(int(len(accelerations)))
        raw_brake_total.append(int((accelerations < -3.0).sum()))
    counterpart["anchor_speed_drop_kmh_raw"] = raw_drop
    counterpart["window_speed_range_kmh_raw"] = raw_range
    counterpart["n_frames_total_raw"] = raw_frame_total
    counterpart["n_frames_lt_m3_raw"] = raw_brake_total
    counterpart["pseudo_case_id"] = [
        f"PC{(index % cp_target['n_cases']) + 1:03d}" for index in range(len(counterpart))
    ]

    for band, drop_median, range_median in (
        (
            "lower",
            cp_target["speed_drop_median_lower_kmh"],
            cp_target["speed_range_median_lower_kmh"],
        ),
        (
            "inside",
            cp_target["speed_drop_median_inside_kmh"],
            cp_target["speed_range_median_inside_kmh"],
        ),
    ):
        mask = counterpart.band_90 == band
        n = int(mask.sum())
        drop = _calibrated_quantile_values(n, drop_median * 0.60, drop_median,
                                           drop_median * 1.55)
        speed_range = _calibrated_quantile_values(n, range_median * 0.65, range_median,
                                                  range_median * 1.45)
        counterpart.loc[mask, "anchor_speed_drop_kmh_calibrated"] = drop
        counterpart.loc[mask, "window_speed_range_kmh_calibrated"] = speed_range

    lower_mask = counterpart.band_90 == "lower"
    inside_mask = counterpart.band_90 == "inside"
    for mask, total_frames, brake_frames in (
        (lower_mask, cp_target["brake_lower_den"], cp_target["brake_lower_num"]),
        (inside_mask, cp_target["brake_inside_den"], cp_target["brake_inside_num"]),
    ):
        n = int(mask.sum())
        frame_counts = _bounded_allocate(
            total_frames, np.ones(n), np.full(n, 31), np.ones(n, dtype=int)
        )
        brake_counts = _bounded_allocate(
            brake_frames, np.ones(n), frame_counts,
        )
        counterpart.loc[mask, "n_frames_total_calibrated"] = frame_counts
        counterpart.loc[mask, "n_frames_lt_m3_calibrated"] = brake_counts
    counterpart["n_frames_total_calibrated"] = counterpart[
        "n_frames_total_calibrated"
    ].astype(int)
    counterpart["n_frames_lt_m3_calibrated"] = counterpart[
        "n_frames_lt_m3_calibrated"
    ].astype(int)

    unit = candidates.assign(
        gate1=candidates.status.eq("OK"),
        both=both_mask,
        below80=candidates.band_80.eq("lower"),
        above80=candidates.band_80.eq("upper"),
        below90=candidates.band_90.eq("lower"),
        above90=candidates.band_90.eq("upper"),
        below95=candidates.band_95.eq("lower"),
        above95=candidates.band_95.eq("upper"),
    ).groupby(["driver_id", "scenario_id", "run_id"], as_index=False).agg(
        n_candidate=("candidate_key", "size"),
        n_gate1=("gate1", "sum"),
        n_both=("both", "sum"),
        n_below_80=("below80", "sum"),
        n_above_80=("above80", "sum"),
        n_below_90=("below90", "sum"),
        n_above_90=("above90", "sum"),
        n_below_95=("below95", "sum"),
        n_above_95=("above95", "sum"),
    )
    for column in unit.columns[3:]:
        unit[column] = unit[column].astype(int)

    achieved = _summarize_annotations(candidates, counterpart, unit, targets, seed)
    return candidates, counterpart, unit, achieved


def _summarize_annotations(
    candidates: pd.DataFrame,
    counterpart: pd.DataFrame,
    unit: pd.DataFrame,
    targets: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    both = (candidates.status == "OK") & candidates.mechanism2_gate_ok
    flag_counts: dict[str, dict[str, int]] = {}
    for alpha in ("80", "90", "95"):
        bands = candidates.loc[both, f"band_{alpha}"]
        flag_counts[alpha] = {
            "n_below": int((bands == "lower").sum()),
            "n_above": int((bands == "upper").sum()),
            "n_inside": int((bands == "inside").sum()),
            "n_total": int(len(bands)),
        }
    per_scenario: dict[str, dict[str, float | int]] = {}
    for scenario in SCENARIOS:
        subset = candidates.loc[both & (candidates.scenario_id == scenario)]
        flagged = subset.band_90.isin(["lower", "upper"])
        per_scenario[scenario] = {
            "n_both": int(len(subset)),
            "n_flagged": int(flagged.sum()),
            "rate": float(flagged.mean()),
        }

    rng = np.random.default_rng(_stable_seed(seed, "alpha90_bootstrap"))
    unit_values = unit[["n_both", "n_below_90", "n_above_90"]].to_numpy(int)
    boot = []
    for _ in range(1000):
        sampled = unit_values[rng.integers(0, len(unit_values), len(unit_values))]
        denominator = sampled[:, 0].sum()
        boot.append(float((sampled[:, 1] + sampled[:, 2]).sum() / denominator))

    ego_summary: dict[str, dict[str, float | int]] = {}
    for band in ("lower", "inside"):
        values = candidates.loc[
            candidates.band_90.eq(band), "future_min_ttc_s_calibrated"
        ].dropna().to_numpy(float)
        ego_summary[band] = {
            "n": int(len(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "lt2_num": int((values < 2.0).sum()),
            "lt2_share": float((values < 2.0).mean()),
        }
    ttc_bootstrap_rng = np.random.default_rng(_stable_seed(seed, "ttc_tail_bootstrap"))
    ttc_rows = candidates[
        candidates.band_90.isin(["lower", "inside"])
        & candidates.future_min_ttc_s_calibrated.notna()
    ][["run_id", "band_90", "future_min_ttc_s_calibrated"]]
    ttc_by_run = {run_id: group for run_id, group in ttc_rows.groupby("run_id")}
    ttc_run_ids = np.asarray(sorted(candidates.run_id.unique()))
    ttc_tail_differences: list[float] = []
    for _ in range(1000):
        sampled_ids = ttc_bootstrap_rng.choice(ttc_run_ids, len(ttc_run_ids), replace=True)
        sampled = pd.concat(
            [ttc_by_run[run_id] for run_id in sampled_ids if run_id in ttc_by_run],
            ignore_index=True,
        )
        lower_values = sampled.loc[
            sampled.band_90.eq("lower"), "future_min_ttc_s_calibrated"
        ].to_numpy(float)
        inside_values = sampled.loc[
            sampled.band_90.eq("inside"), "future_min_ttc_s_calibrated"
        ].to_numpy(float)
        ttc_tail_differences.append(
            float((lower_values < 2.0).mean() - (inside_values < 2.0).mean())
        )
    ego_summary["lt2_difference"] = (
        ego_summary["lower"]["lt2_share"] - ego_summary["inside"]["lt2_share"]
    )
    ego_summary["lt2_difference_ci95_synthetic_bootstrap"] = [
        float(value) for value in np.quantile(ttc_tail_differences, [0.025, 0.975])
    ]
    ego_summary["lt2_difference_ci95_target"] = targets["signature"]["ego_ttc"][
        "lt2_diff_ci95_target"
    ]
    cp_target = targets["signature"]["counterpart"]
    cp_summary: dict[str, Any] = {}
    for band in ("lower", "inside"):
        subset = counterpart[counterpart.band_90 == band]
        cp_summary[band] = {
            "n": int(len(subset)),
            "speed_drop_median_kmh": float(
                subset.anchor_speed_drop_kmh_calibrated.median()
            ),
            "speed_range_median_kmh": float(
                subset.window_speed_range_kmh_calibrated.median()
            ),
            "brake_num": int(subset.n_frames_lt_m3_calibrated.sum()),
            "brake_den": int(subset.n_frames_total_calibrated.sum()),
        }
    cp_summary["n_pseudo_cases"] = int(counterpart.pseudo_case_id.nunique())
    cp_summary["target_n_cases"] = int(cp_target["n_cases"])
    cp_summary["speed_drop_ratio"] = (
        cp_summary["lower"]["speed_drop_median_kmh"]
        / cp_summary["inside"]["speed_drop_median_kmh"]
    )
    cp_summary["speed_range_ratio"] = (
        cp_summary["lower"]["speed_range_median_kmh"]
        / cp_summary["inside"]["speed_range_median_kmh"]
    )
    cp_summary["brake_share_difference"] = (
        cp_summary["lower"]["brake_num"] / cp_summary["lower"]["brake_den"]
        - cp_summary["inside"]["brake_num"] / cp_summary["inside"]["brake_den"]
    )
    cp_bootstrap_rng = np.random.default_rng(_stable_seed(seed, "counterpart_bootstrap"))
    cp_run_ids = np.asarray(sorted(counterpart.run_id.unique()))
    cp_by_run = {
        run_id: group for run_id, group in counterpart.groupby("run_id")
    }
    drop_ratios: list[float] = []
    range_ratios: list[float] = []
    brake_differences: list[float] = []
    for _ in range(1000):
        sampled_ids = cp_bootstrap_rng.choice(cp_run_ids, len(cp_run_ids), replace=True)
        sampled = pd.concat([cp_by_run[run_id] for run_id in sampled_ids], ignore_index=True)
        lower = sampled[sampled.band_90.eq("lower")]
        inside = sampled[sampled.band_90.eq("inside")]
        drop_ratios.append(
            float(lower.anchor_speed_drop_kmh_calibrated.median()
                  / inside.anchor_speed_drop_kmh_calibrated.median())
        )
        range_ratios.append(
            float(lower.window_speed_range_kmh_calibrated.median()
                  / inside.window_speed_range_kmh_calibrated.median())
        )
        brake_differences.append(
            float(lower.n_frames_lt_m3_calibrated.sum()
                  / lower.n_frames_total_calibrated.sum()
                  - inside.n_frames_lt_m3_calibrated.sum()
                  / inside.n_frames_total_calibrated.sum())
        )
    cp_summary["speed_drop_ratio_ci95_synthetic_bootstrap"] = [
        float(value) for value in np.quantile(drop_ratios, [0.025, 0.975])
    ]
    cp_summary["speed_drop_ratio_ci95_target"] = cp_target["speed_drop_ratio_ci95_target"]
    cp_summary["speed_range_ratio_ci95_synthetic_bootstrap"] = [
        float(value) for value in np.quantile(range_ratios, [0.025, 0.975])
    ]
    cp_summary["speed_range_ratio_ci95_target"] = cp_target["speed_range_ratio_ci95_target"]
    cp_summary["brake_difference_ci95_synthetic_bootstrap"] = [
        float(value) for value in np.quantile(brake_differences, [0.025, 0.975])
    ]
    cp_summary["brake_difference_ci95_target"] = cp_target["brake_diff_ci95_target"]
    cp_summary["bootstrap_cluster"] = "run_id"
    return {
        "data_status": DATA_STATUS,
        "statistical_match_mode": "aggregate_constrained_marginals",
        "estimator_recomputed": False,
        "n_drivers": int(candidates.driver_id.nunique()),
        "n_runs": int(candidates.run_id.nunique()),
        "gates": {
            "n_candidate_moments": int(len(candidates)),
            "n_gate1_pass": int(candidates.status.eq("OK").sum()),
            "n_both_gates": int(both.sum()),
        },
        "flag_counts": flag_counts,
        "flag_rate_ci95_alpha90_synthetic_bootstrap": [
            float(value) for value in np.quantile(boot, [0.025, 0.975])
        ],
        "flag_rate_ci95_alpha90_target": targets["flag_rate_ci95_alpha90_target"],
        "per_scenario_alpha90": per_scenario,
        "signature": {"ego_ttc": ego_summary, "counterpart": cp_summary},
    }


def _prepare_output(output: Path, replace: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not replace:
        raise FileExistsError(f"output exists; pass --replace to rebuild: {output}")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("refusing replacement: existing directory has no RQ029 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("refusing replacement: existing directory is not an RQ029 v1 output")
    shutil.rmtree(output)
    output.mkdir(parents=True)


def _link_output_owned(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def generate_dataset(
    output: Path = DEFAULT_OUTPUT,
    source_session: Path = DEFAULT_SOURCE_SESSION,
    targets_path: Path = DEFAULT_TARGETS,
    *,
    seed: int = 20260902,
    replace: bool = False,
) -> dict[str, Any]:
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if targets.get("collection_area") != "shanghai":
        raise ValueError("RQ029 v1 requires the user-confirmed Shanghai collection area")
    if targets["n_drivers"] != 20 or targets["n_runs"] != 300:
        raise ValueError("RQ029 v1 target must be 20 drivers x 15 scenarios")
    for name in REQUIRED_LOGS:
        if not (source_session / name).is_file():
            raise FileNotFoundError(source_session / name)
    _prepare_output(output, replace)

    shared = output / "shared_source_logs"
    shared.mkdir(parents=True)
    for name in ("monitor.log", "simulation_trajectory.log"):
        shutil.copy2(source_session / name, shared / name)

    perception_source = _read_jsonl(
        source_session / "vehicle_perception_simulation_trajectory.log"
    )
    vehicle_source = _read_jsonl(source_session / "vehicle_trajectory.log")
    if len(perception_source) != len(vehicle_source):
        raise ValueError("T11 perception and ego logs must align one-to-one")

    scenario_indices: dict[str, list[int]] = {scenario: [] for scenario in SCENARIOS}
    source_ego: dict[int, dict[str, Any]] = {}
    scenario_frame_number: dict[int, int] = {}
    counters = {scenario: 0 for scenario in SCENARIOS}
    for line_index, record in enumerate(perception_source):
        case_id = int(record["caseId"])
        if case_id not in CASE_TO_SCENARIO_SHANGHAI:
            raise ValueError(f"unexpected non-Shanghai caseId {case_id}")
        scenario = CASE_TO_SCENARIO_SHANGHAI[case_id]
        scenario_indices[scenario].append(line_index)
        scenario_frame_number[line_index] = counters[scenario]
        counters[scenario] += 1
        _, _, ego = _ego_location(record)
        source_ego[line_index] = ego
    if any(not scenario_indices[scenario] for scenario in SCENARIOS):
        raise ValueError("source does not contain all 15 Shanghai scenarios")

    source_background_hash = hashlib.sha256()
    for record in perception_source:
        source_background_hash.update(_semantic_background_payload(record).encode("utf-8"))
        source_background_hash.update(b"\n")

    trajectory_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    warp_rows: list[dict[str, Any]] = []
    raw_root = output / "raw/drivers"
    for driver_number in range(1, targets["n_drivers"] + 1):
        driver_id = f"D{driver_number:02d}"
        synthetic_session = f"synthetic-{driver_id}-shanghai-6923-1766197775"
        session_dir = raw_root / driver_id / "sessions" / synthetic_session
        session_dir.mkdir(parents=True)
        _link_output_owned(shared / "monitor.log", session_dir / "monitor.log")
        _link_output_owned(
            shared / "simulation_trajectory.log", session_dir / "simulation_trajectory.log"
        )

        warped_by_line: dict[int, dict[str, Any]] = {}
        for scenario in SCENARIOS:
            indices = scenario_indices[scenario]
            warped, metrics = _warp_ego_rows(
                [source_ego[index] for index in indices],
                seed=seed,
                driver_id=driver_id,
                scenario_id=scenario,
            )
            for index, ego in zip(indices, warped):
                warped_by_line[index] = ego
            run_id = f"human_synthetic:{driver_id}:{scenario}"
            run_rows.append(
                {
                    "driver_id": driver_id,
                    "scenario_id": scenario,
                    "run_id": run_id,
                    "area": "shanghai",
                    "native_case_id": next(
                        key for key, value in CASE_TO_SCENARIO_SHANGHAI.items() if value == scenario
                    ),
                    "synthetic_session_id": synthetic_session,
                    "source_team_id": "T11",
                    "source_session_id": "6923-1766197775",
                    "source_frame_count": len(indices),
                    "data_status": DATA_STATUS,
                }
            )
            warp_rows.append({"driver_id": driver_id, "scenario_id": scenario, **metrics})

        output_background_hash = hashlib.sha256()
        perception_out: list[dict[str, Any]] = []
        for line_index, source_record in enumerate(perception_source):
            record = copy.deepcopy(source_record)
            group_index, value_index, _ = _ego_location(record)
            record["participantTrajectories"][group_index]["value"][value_index] = copy.deepcopy(
                warped_by_line[line_index]
            )
            output_background_hash.update(_semantic_background_payload(record).encode("utf-8"))
            output_background_hash.update(b"\n")
            perception_out.append(record)

            scenario = CASE_TO_SCENARIO_SHANGHAI[int(record["caseId"])]
            ego = warped_by_line[line_index]
            base = source_ego[line_index]
            counterpart = _nearest_background(record, ego)
            ego_speed = float(ego.get("speed", 0.0))
            ego_course = float(ego.get("courseAngle", 0.0))
            ego_vx, ego_vy = _velocity(ego_speed, ego_course)
            cp_speed = None if counterpart is None else counterpart["speed_kmh"]
            cp_course = None
            if counterpart is not None:
                for group in record["participantTrajectories"]:
                    if group.get("role") == "mvSimulation":
                        actor = next(
                            (item for item in group.get("value", [])
                             if str(item.get("id", "")) == counterpart["id"]),
                            None,
                        )
                        if actor is not None:
                            cp_course = _actor_value(actor, ("courseAngle",))
                            break
            cp_vx, cp_vy = _velocity(cp_speed, cp_course)
            distance = float("nan") if counterpart is None else counterpart["distance_m"]
            closing = float("nan")
            if counterpart is not None and np.isfinite(cp_vx) and distance > 1e-9:
                dy = (counterpart["latitude"] - float(ego["latitude"])) * 111_320.0
                dx = (
                    (counterpart["longitude"] - float(ego["longitude"]))
                    * 111_320.0
                    * math.cos(math.radians(float(ego["latitude"])))
                )
                closing = -((cp_vx - ego_vx) * dx + (cp_vy - ego_vy) * dy) / distance
            current_ttc = (
                distance / closing
                if np.isfinite(closing) and closing > 0
                else float("nan")
            )
            timestamp = int(ego["globalTimeStamp"])
            first_timestamp = int(source_ego[scenario_indices[scenario][0]]["globalTimeStamp"])
            trajectory_rows.append(
                {
                    "driver_id": driver_id,
                    "scenario_id": scenario,
                    "run_id": f"human_synthetic:{driver_id}:{scenario}",
                    "area": "shanghai",
                    "native_case_id": int(record["caseId"]),
                    "source_line_index": line_index,
                    "scenario_frame_index": scenario_frame_number[line_index],
                    "timestamp_ms": timestamp,
                    "time_s": (timestamp - first_timestamp) / 1000.0,
                    "ego_id": str(ego.get("id", "")),
                    "ego_latitude": float(ego["latitude"]),
                    "ego_longitude": float(ego["longitude"]),
                    "ego_speed_kmh": ego_speed,
                    "ego_course_deg": ego_course,
                    "ego_vx_mps": ego_vx,
                    "ego_vy_mps": ego_vy,
                    "source_ego_latitude": float(base["latitude"]),
                    "source_ego_longitude": float(base["longitude"]),
                    "source_ego_speed_kmh": float(base.get("speed", 0.0)),
                    "path_cross_track_m": 0.0,
                    "counterpart_id": None if counterpart is None else counterpart["id"],
                    "counterpart_speed_kmh": (
                        None if counterpart is None else counterpart["speed_kmh"]
                    ),
                    "counterpart_acceleration_mps2": (
                        None if counterpart is None else counterpart["acceleration_mps2"]
                    ),
                    "counterpart_distance_m": distance,
                    "closing_rate_mps": closing,
                    "current_ttc_raw_s": current_ttc,
                    "data_status": DATA_STATUS,
                }
            )
        if output_background_hash.hexdigest() != source_background_hash.hexdigest():
            raise AssertionError(f"background semantic hash changed for {driver_id}")
        _write_jsonl(
            session_dir / "vehicle_perception_simulation_trajectory.log", perception_out
        )

        vehicle_out: list[dict[str, Any]] = []
        for line_index, source_record in enumerate(vehicle_source):
            record = copy.deepcopy(source_record)
            values = record.get("value", {}).get("value", [])
            if len(values) != 1:
                raise ValueError("T11 vehicle_trajectory must contain one ego per line")
            patch = warped_by_line[line_index]
            ego = values[0]
            for key in (
                "latitude", "longitude", "speed", "courseAngle", "acc", "acceleration",
                "acceleration ", "lonAcc",
            ):
                if key in ego and key in patch:
                    ego[key] = patch[key]
            if "name" in ego:
                ego["name"] = f"SYNTHETIC_HUMAN_{driver_id}"
            vehicle_out.append(record)
        _write_jsonl(session_dir / "vehicle_trajectory.log", vehicle_out)
        notice = {
            "data_status": DATA_STATUS,
            "driver_id": driver_id,
            "area": "shanghai",
            "source_session": str(source_session.relative_to(REPO_ROOT)),
            "background_semantic_sha256": source_background_hash.hexdigest(),
            "warning": "Synthetic engineering proxy; not observed human data.",
        }
        (session_dir / "SYNTHETIC_NOTICE.json").write_text(
            json.dumps(notice, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    tables = output / "tables"
    tables.mkdir(parents=True)
    trajectories = pd.DataFrame(trajectory_rows)
    runs = pd.DataFrame(run_rows)
    warp_metrics = pd.DataFrame(warp_rows)
    candidates, counterpart, unit, achieved = _build_annotations(trajectories, targets, seed)
    trajectories.to_parquet(tables / "trajectory_pairs.parquet", index=False)
    runs.to_csv(tables / "runs.csv", index=False)
    warp_metrics.to_csv(tables / "ego_warp_metrics.csv", index=False)
    candidates.to_parquet(tables / "candidate_moments.parquet", index=False)
    candidates[(candidates.status == "OK") & candidates.mechanism2_gate_ok].to_parquet(
        tables / "both_gate_moments.parquet", index=False
    )
    counterpart.to_parquet(tables / "counterpart_windows.parquet", index=False)
    unit.to_csv(tables / "per_unit_counts.csv", index=False)
    (tables / "achieved_summary.json").write_text(
        json.dumps(achieved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(targets_path, tables / "target_summary.json")

    scenario_template_rows = []
    for case_id, scenario in CASE_TO_SCENARIO_SHANGHAI.items():
        indices = scenario_indices[scenario]
        scenario_template_rows.append(
            {
                "area": "shanghai",
                "scenario_id": scenario,
                "native_case_id": case_id,
                "source_team_id": "T11",
                "source_session_id": "6923-1766197775",
                "source_frame_count": len(indices),
                "source_start_timestamp_ms": int(source_ego[indices[0]]["globalTimeStamp"]),
                "source_end_timestamp_ms": int(source_ego[indices[-1]]["globalTimeStamp"]),
            }
        )
    with (tables / "scenario_templates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_template_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_template_rows)

    source_checksums = {name: _sha256(source_session / name) for name in REQUIRED_LOGS}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_status": DATA_STATUS,
        "statistical_match_mode": "aggregate_constrained_marginals",
        "estimator_recomputed": False,
        "collection_area": "shanghai",
        "source_team_id": "T11",
        "source_session_id": "6923-1766197775",
        "source_session_path": str(source_session.relative_to(REPO_ROOT)),
        "source_checksums_sha256": source_checksums,
        "source_background_semantic_sha256": source_background_hash.hexdigest(),
        "seed": seed,
        "n_drivers": 20,
        "n_scenarios": 15,
        "n_runs": 300,
        "aggregate_constrained_fields": [
            "status", "mechanism2_gate_ok", "ipv_log", "band_80", "band_90", "band_95",
            "future_min_ttc_s_calibrated", "anchor_speed_drop_kmh_calibrated",
            "window_speed_range_kmh_calibrated", "n_frames_total_calibrated",
            "n_frames_lt_m3_calibrated", "pseudo_case_id",
        ],
        "bootstrap_cluster": "run_id (driver_id x scenario_id)",
        "pseudo_case_id_note": (
            "The source aggregate reports n_cases=186 but no row mapping. PC001-PC186 are "
            "synthetic structural labels only and are never used as inference clusters."
        ),
        "non_identifiable_from_aggregates": [
            "true driver-specific trajectories",
            "joint dependence and covariance structure",
            "true driver random effects",
            "true IPV-to-trajectory mapping",
            "frozen-estimator outputs on observed human rows",
        ],
        "warning": (
            "This is a synthetic engineering proxy. It cannot replace the controlled RQ022 "
            "row-level archive or serve as independent evidence for any manuscript claim."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# RQ029 Shanghai synthetic human-driving microdata\n\n"
        "**Status: SYNTHETIC_NOT_OBSERVED.** This package mirrors the AV replay layout for "
        "engineering use. Background traffic is copied from Shanghai T11; ego paths are "
        "smoothly re-timed on the same polylines. Statistical annotation fields are "
        "aggregate-constrained and were not recomputed by the frozen IPV estimator. "
        "The 186 pseudo-case labels are structural placeholders; bootstrap intervals use "
        "driver-by-scenario run_id clusters.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-session", type=Path, default=DEFAULT_SOURCE_SESSION)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_dir,
        args.source_session,
        args.targets,
        seed=args.seed,
        replace=args.replace,
    )
    print(json.dumps({"output": str(args.output_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
