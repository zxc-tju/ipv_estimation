#!/usr/bin/env python3
"""Outcome-blind pilot extractor for RQ012A automatic events.

The pilot intentionally reads only trajectory/geometry/time/actor-id data plus
RQ003 annotation item IDs for exclusion. It does not read IPV, score, rank, team
identity, human labels, or event-outcome associations.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RUN_ID = "RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37"
WORKER_ID = "RQ012-W17a-extractor-robustness"
EVENTS = {
    "E01": "counterpart hard braking",
    "E02": "high deceleration",
    "E03": "high jerk",
    "E06": "repeated stop-go",
    "E09": "near miss",
    "E15": "collision/contact geometric candidate",
    "E16": "no-progress timeout",
    "E18": "kinematic emergency-stop candidate",
    "E19": "abrupt lateral comfort events",
}
PAIR_EVENTS = {"E09", "E15"}
ACTOR_EVENTS = {"E01", "E02", "E03", "E06"}
EGO_EVENTS = {"E16", "E18", "E19"}
R_EARTH_M = 6371000.0
PRE_FIX_CENTRAL_AUDIT = {
    "E01": {"event_count": 0, "raw_frame_hits": 0, "impossible_values": 0, "missing_data_failures": 1703},
    "E02": {"event_count": 1770, "raw_frame_hits": 108552, "impossible_values": 2, "missing_data_failures": 5},
    "E03": {"event_count": 1615, "raw_frame_hits": 30099, "impossible_values": 2, "missing_data_failures": 5},
    "E06": {"event_count": 186, "raw_frame_hits": 108439, "impossible_values": 2, "missing_data_failures": 4},
    "E09": {"event_count": 950, "raw_frame_hits": 28026, "impossible_values": 0, "missing_data_failures": 974},
    "E15": {"event_count": 152, "raw_frame_hits": 4562, "impossible_values": 0, "missing_data_failures": 974},
    "E16": {"event_count": 48, "raw_frame_hits": 2194, "impossible_values": 2, "missing_data_failures": 4},
    "E18": {"event_count": 0, "raw_frame_hits": 7, "impossible_values": 2, "missing_data_failures": 4},
    "E19": {"event_count": 2, "raw_frame_hits": 67, "impossible_values": 2, "missing_data_failures": 4},
}


@dataclass
class SessionRef:
    session_key: str
    pilot_session_id: str
    session_dir: Path
    relative_dir: str
    duplicate_source_count: int
    rq003_public_replay_id_matches: List[str] = field(default_factory=list)


@dataclass
class SessionData:
    ref: SessionRef
    ego: List[Dict[str, Any]]
    world_by_actor: Dict[str, List[Dict[str, Any]]]
    health: Dict[str, int]


@dataclass
class EventResult:
    event_id: str
    event_name: str
    total_units: int = 0
    computable_units: int = 0
    event_count: int = 0
    raw_event_count: int = 0
    primary_event_count: int = 0
    suppressed_by_precedence: int = 0
    raw_frame_hits: int = 0
    runs_before_duration: int = 0
    events_before_merge: int = 0
    events_after_merge: int = 0
    duplicate_overlapping_events: int = 0
    overlaps_after_merge: int = 0
    impossible_values: int = 0
    actor_attribution_failures: int = 0
    missing_data_failures: int = 0
    intervals: List[Dict[str, Any]] = field(default_factory=list)
    cross_event_audit: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def computable_fraction(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return self.computable_units / self.total_units

    @property
    def event_rate_per_unit(self) -> float:
        if self.computable_units <= 0:
            return 0.0
        return self.event_count / self.computable_units


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[6]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_value(row: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def parse_timestamp_ms(container: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> Optional[float]:
    value = container.get("value", {}) if isinstance(container.get("value"), dict) else {}
    candidates = [value.get("timestamp")]
    if row:
        candidates.extend([row.get("globalTimeStamp"), row.get("timestamp")])
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            return safe_float(text)
    return None


def scaled_dimension_m(raw_value: Any) -> Optional[float]:
    value = safe_float(raw_value)
    if value is None:
        return None
    if value > 30.0:
        value = value / 100.0
    return value


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def event_unit_key(row: Dict[str, Any]) -> str:
    return str(row.get("actor_id", "unknown"))


def pilot_id_for_session(session_key: str) -> str:
    digest = hashlib.sha256(f"{RUN_ID}|{session_key}".encode("utf-8")).hexdigest()[:12]
    return f"pilot_{digest}"


def discover_sessions(onsite_root: Path) -> List[SessionRef]:
    dirs: Dict[str, List[Path]] = defaultdict(list)
    for sim_path in onsite_root.rglob("simulation_trajectory.log"):
        session_dir = sim_path.parent
        required = [
            session_dir / "simulation_trajectory.log",
            session_dir / "vehicle_trajectory.log",
            session_dir / "vehicle_perception_simulation_trajectory.log",
        ]
        if all(path.exists() for path in required):
            session_key = session_dir.name
            dirs[session_key].append(session_dir)

    refs: List[SessionRef] = []
    for session_key, candidates in dirs.items():
        # Deduplicate hardlinked/copied replay logs by session key. Prefer raw
        # material when present, then the shortest path for reproducibility.
        candidates = sorted(
            candidates,
            key=lambda p: ("top5_research_subset" in p.parts, len(p.parts), str(p)),
        )
        chosen = candidates[0]
        refs.append(
            SessionRef(
                session_key=session_key,
                pilot_session_id=pilot_id_for_session(session_key),
                session_dir=chosen,
                relative_dir=str(chosen.relative_to(onsite_root.parent.parent if False else onsite_root.parent)),
                duplicate_source_count=len(candidates),
            )
        )
    refs.sort(key=lambda ref: hashlib.sha256(ref.session_key.encode("utf-8")).hexdigest())
    return refs


def load_rq003_public_replay_ids(annotation_dir: Path) -> List[str]:
    ids: List[str] = []
    for name in ("mechanism_sample_manifest.csv", "validation_sample_manifest.csv"):
        path = annotation_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ("blind_case_id", "public_replay_case_id"):
                    value = safe_str(row.get(col))
                    if value:
                        ids.append(value)
    return sorted(set(ids))


def match_rq003_session_item_ids(ref: SessionRef, candidate_ids: Sequence[str]) -> List[str]:
    """Return RQ003 item IDs that exactly identify this replay session.

    RQ003 public replay IDs can also appear inside non-ego actor display names
    in OnSite trajectory rows. Those actor-name prefixes are not session IDs, so
    they are deliberately not used for session exclusion; otherwise the pilot
    would exclude essentially the full available OnSite sample based on traffic
    participant labels rather than annotation-session identity.
    """

    path_parts = set(ref.session_dir.parts)
    matches = set()
    for item_id in candidate_ids:
        if item_id == ref.session_key or item_id in path_parts:
            matches.add(item_id)
    return sorted(matches)


def select_pilot_sample(
    onsite_root: Path,
    rq003_annotation_dir: Path,
    seed: int,
    sample_n: int,
    derived_dir: Path,
) -> Tuple[List[SessionRef], List[SessionRef], List[str]]:
    refs = discover_sessions(onsite_root)
    rq003_ids = load_rq003_public_replay_ids(rq003_annotation_dir)
    eligible: List[SessionRef] = []
    excluded: List[SessionRef] = []
    for ref in refs:
        matches = match_rq003_session_item_ids(ref, rq003_ids)
        ref.rq003_public_replay_id_matches = matches
        if matches:
            excluded.append(ref)
        else:
            eligible.append(ref)

    rng = random.Random(seed)
    stable_pool = sorted(eligible, key=lambda r: r.pilot_session_id)
    selected = rng.sample(stable_pool, min(sample_n, len(stable_pool)))
    selected.sort(key=lambda r: r.pilot_session_id)

    derived_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = derived_dir / "pilot_sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pilot_session_id",
                "session_key",
                "source_relative_dir",
                "duplicate_source_count",
                "selected",
                "rq003_excluded",
                "rq003_public_replay_id_matches",
            ],
        )
        writer.writeheader()
        selected_ids = {ref.pilot_session_id for ref in selected}
        excluded_ids = {ref.pilot_session_id for ref in excluded}
        for ref in sorted(refs, key=lambda r: r.pilot_session_id):
            writer.writerow(
                {
                    "pilot_session_id": ref.pilot_session_id,
                    "session_key": ref.session_key,
                    "source_relative_dir": ref.relative_dir,
                    "duplicate_source_count": ref.duplicate_source_count,
                    "selected": ref.pilot_session_id in selected_ids,
                    "rq003_excluded": ref.pilot_session_id in excluded_ids,
                    "rq003_public_replay_id_matches": ";".join(ref.rq003_public_replay_id_matches),
                }
            )

    seed_payload = {
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "seed": seed,
        "sample_n_requested": sample_n,
        "available_unique_sessions": len(refs),
        "rq003_item_ids_loaded": len(rq003_ids),
        "rq003_excluded_sessions": len(excluded),
        "eligible_sessions": len(eligible),
        "selected_sessions": len(selected),
        "selection_rule": "Python random.Random(seed).sample over sorted eligible pilot_session_id values after RQ003 item-ID exclusion.",
        "rq003_exclusion_match_rule": "Exact match against OnSite session_key or source path segment only; actor display-name prefixes are not treated as session IDs.",
        "no_outcome_fields_used": True,
    }
    (derived_dir / "pilot_seed.json").write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
    return selected, excluded, rq003_ids


def read_ego_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows_out: List[Dict[str, Any]] = []
    health = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            health["ego_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                health["ego_json_bad"] += 1
                continue
            t_ms = parse_timestamp_ms(record)
            values = record.get("value", {}).get("value", [])
            for row in values:
                actor_id = safe_str(row.get("id")) or "ego"
                row_t = parse_timestamp_ms(record, row) or t_ms
                out = {
                    "source": "ego",
                    "actor_id": actor_id,
                    "t_ms": row_t,
                    "speed": safe_float(row.get("speed")),
                    "accel": safe_float(first_value(row, ["acceleration", "acceleration ", "lonAcc"])),
                    "braking": safe_float(row.get("braking")),
                    "lat": safe_float(first_value(row, ["latitude", "lat"])),
                    "lon": safe_float(first_value(row, ["longitude", "lng"])),
                    "course_deg": safe_float(row.get("courseAngle")),
                    "length_m": scaled_dimension_m(row.get("length")),
                    "width_m": scaled_dimension_m(row.get("width")),
                    "lat_acc": safe_float(row.get("latAcc")),
                    "lon_acc": safe_float(row.get("lonAcc")),
                    "steering_angle": safe_float(row.get("steeringWheelAngle")),
                    "wheel_angle": safe_float(row.get("wheelAngle")),
                }
                rows_out.append(out)
    rows_out.sort(key=lambda r: (r["t_ms"] if r["t_ms"] is not None else -1))
    return rows_out, dict(health)


def read_world_rows(path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    by_actor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    health = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            health["world_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                health["world_json_bad"] += 1
                continue
            t_ms = parse_timestamp_ms(record)
            values = record.get("value", {}).get("value", [])
            for row in values:
                actor_id = safe_str(row.get("id")) or safe_str(row.get("originId"))
                if not actor_id:
                    health["world_missing_actor_id"] += 1
                    actor_id = "missing_actor_id"
                row_t = parse_timestamp_ms(record, row) or t_ms
                out = {
                    "source": "world",
                    "actor_id": actor_id,
                    "origin_id": safe_str(row.get("originId")),
                    "name": safe_str(row.get("name")),
                    "t_ms": row_t,
                    "speed": safe_float(row.get("speed")),
                    "accel": safe_float(first_value(row, ["acceleration", "acceleration "])),
                    "braking": safe_float(row.get("braking")),
                    "lat": safe_float(first_value(row, ["latitude", "lat"])),
                    "lon": safe_float(first_value(row, ["longitude", "lng"])),
                    "source_x": safe_float(row.get("x")),
                    "source_y": safe_float(row.get("y")),
                    "course_deg": safe_float(row.get("courseAngle")),
                    "length_m": scaled_dimension_m(row.get("length")),
                    "width_m": scaled_dimension_m(row.get("width")),
                }
                by_actor[actor_id].append(out)
    for actor_rows in by_actor.values():
        actor_rows.sort(key=lambda r: (r["t_ms"] if r["t_ms"] is not None else -1))
    return dict(by_actor), dict(health)


def add_metric_xy(session: SessionData) -> None:
    lat0 = None
    lon0 = None
    for row in session.ego:
        if is_finite_number(row.get("lat")) and is_finite_number(row.get("lon")):
            lat0 = float(row["lat"])
            lon0 = float(row["lon"])
            break
    if lat0 is None or lon0 is None:
        session.health["coordinate_anchor_missing"] = 1
        return

    lat0_rad = math.radians(lat0)

    def convert(row: Dict[str, Any]) -> None:
        lat = row.get("lat")
        lon = row.get("lon")
        if is_finite_number(lat) and is_finite_number(lon):
            row["x_m"] = math.radians(float(lon) - lon0) * math.cos(lat0_rad) * R_EARTH_M
            row["y_m"] = math.radians(float(lat) - lat0) * R_EARTH_M
            row["xy_source"] = "latlon_local_m"
        elif is_finite_number(row.get("source_x")) and is_finite_number(row.get("source_y")):
            row["x_m"] = float(row["source_x"])
            row["y_m"] = float(row["source_y"])
            row["xy_source"] = "source_xy"
        else:
            row["x_m"] = None
            row["y_m"] = None
            row["xy_source"] = "missing"

    for row in session.ego:
        convert(row)
    for actor_rows in session.world_by_actor.values():
        for row in actor_rows:
            convert(row)


def load_session(ref: SessionRef) -> SessionData:
    ego, ego_health = read_ego_rows(ref.session_dir / "vehicle_trajectory.log")
    world, world_health = read_world_rows(ref.session_dir / "simulation_trajectory.log")
    data = SessionData(ref=ref, ego=ego, world_by_actor=world, health={**ego_health, **world_health})
    add_metric_xy(data)
    return data


def decimate_rows(rows: Sequence[Dict[str, Any]], factor: int) -> List[Dict[str, Any]]:
    if factor <= 1:
        return list(rows)
    return [row for idx, row in enumerate(rows) if idx % factor == 0]


def positive_median_dt_s(rows: Sequence[Dict[str, Any]]) -> float:
    times = [float(r["t_ms"]) / 1000.0 for r in rows if is_finite_number(r.get("t_ms"))]
    dts = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not dts:
        return 0.1
    return max(0.001, float(median(dts)))


def row_quality_for_emission(
    row: Dict[str, Any],
    required_fields: Sequence[str],
    previous_t_s: Optional[float],
    require_dimensions: bool = False,
) -> Tuple[bool, bool, Optional[float]]:
    """Return whether a row is missing, impossible, and its timestamp in seconds."""

    missing = False
    impossible = False
    t_s: Optional[float] = None
    if not is_finite_number(row.get("t_ms")):
        missing = True
    else:
        t_s = float(row["t_ms"]) / 1000.0
        if previous_t_s is not None and t_s <= previous_t_s:
            impossible = True

    for field_name in required_fields:
        if not is_finite_number(row.get(field_name)):
            missing = True

    speed = row.get("speed")
    if is_finite_number(speed) and float(speed) < 0.0:
        impossible = True

    finite_if_present = (
        "accel",
        "braking",
        "lat_acc",
        "lon_acc",
        "steering_angle",
        "wheel_angle",
        "course_deg",
        "x_m",
        "y_m",
        "signed_distance",
        "ttc",
    )
    for field_name in finite_if_present:
        if row.get(field_name) is not None and not is_finite_number(row.get(field_name)):
            missing = True

    length_m = row.get("length_m")
    width_m = row.get("width_m")
    if require_dimensions:
        if not is_finite_number(length_m) or not is_finite_number(width_m):
            missing = True
        elif not (0.1 <= float(length_m) <= 30.0) or not (0.1 <= float(width_m) <= 10.0):
            impossible = True
    else:
        if is_finite_number(length_m) and not (0.1 <= float(length_m) <= 30.0):
            impossible = True
        if is_finite_number(width_m) and not (0.1 <= float(width_m) <= 10.0):
            impossible = True

    return missing, impossible, t_s


def split_valid_segments(
    rows: Sequence[Dict[str, Any]],
    required_fields: Sequence[str],
    missing_gap_s: float,
    require_dimensions: bool = False,
) -> Tuple[List[List[Dict[str, Any]]], int, int, int]:
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    missing_rows = 0
    impossible = 0
    gap_failures = 0
    last_t: Optional[float] = None
    prev_valid_t: Optional[float] = None

    for row in sorted(rows, key=lambda r: (r["t_ms"] if r.get("t_ms") is not None else -1)):
        missing, row_impossible, t_s = row_quality_for_emission(row, required_fields, last_t, require_dimensions)
        if t_s is not None:
            last_t = t_s
        if missing or row_impossible:
            missing_rows += int(missing)
            impossible += int(row_impossible)
            if current:
                segments.append(current)
                current = []
            prev_valid_t = None
            continue
        if t_s is None:
            continue
        if prev_valid_t is not None and t_s - prev_valid_t > missing_gap_s:
            gap_failures += 1
            if current:
                segments.append(current)
            current = []
        current.append(row)
        prev_valid_t = t_s

    if current:
        segments.append(current)
    return segments, missing_rows, gap_failures, impossible


def causal_rolling_median(values: Sequence[Optional[float]], times_s: Sequence[float], window_s: float) -> List[Optional[float]]:
    smoothed: List[Optional[float]] = []
    left = 0
    for i, t in enumerate(times_s):
        while left < i and t - times_s[left] > window_s:
            left += 1
        window = [v for v in values[left : i + 1] if v is not None and math.isfinite(v)]
        smoothed.append(float(median(window)) if window else None)
    return smoothed


def derivative(values: Sequence[Optional[float]], times_s: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for prev_v, value, prev_t, t in zip(values, values[1:], times_s, times_s[1:]):
        if value is None or prev_v is None or t <= prev_t:
            out.append(None)
        else:
            out.append((value - prev_v) / (t - prev_t))
    return out


def interval_stats_and_records(
    rows: Sequence[Dict[str, Any]],
    flags: Sequence[bool],
    min_duration_s: float,
    merge_gap_s: float,
    event_id: str,
    session_ref: SessionRef,
    actor_id: str = "",
    counterpart_id: str = "",
    unit_id: str = "",
    note: str = "",
    primary_endpoint: bool = True,
    identity_status: str = "stable",
    precedence_status: str = "primary",
    precedence_reason: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not rows or not flags:
        return [], {
            "raw_frame_hits": 0,
            "runs_before_duration": 0,
            "events_before_merge": 0,
            "events_after_merge": 0,
            "duplicate_overlapping_events": 0,
            "overlaps_after_merge": 0,
            "impossible_values": 0,
        }
    dt_est = positive_median_dt_s(rows)
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        if (not flag or idx == len(flags) - 1) and start is not None:
            end = idx if flag and idx == len(flags) - 1 else idx - 1
            runs.append((start, end))
            start = None
    before_duration = len(runs)
    accepted: List[Dict[str, Any]] = []
    impossible = 0
    for start_idx, end_idx in runs:
        start_s = float(rows[start_idx]["t_ms"]) / 1000.0
        end_s = float(rows[end_idx]["t_ms"]) / 1000.0
        duration_s = end_s - start_s + dt_est
        if duration_s < -1e-9:
            impossible += 1
            continue
        if duration_s + 1e-9 >= min_duration_s:
            accepted.append({"start_s": start_s, "end_s": end_s, "duration_s": max(duration_s, 0.0)})

    merged: List[Dict[str, Any]] = []
    for item in accepted:
        if merged and item["start_s"] - merged[-1]["end_s"] <= merge_gap_s:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], item["end_s"])
            merged[-1]["duration_s"] = merged[-1]["end_s"] - merged[-1]["start_s"] + dt_est
        else:
            merged.append(dict(item))

    overlaps_after = 0
    prev_end = None
    for item in merged:
        if prev_end is not None and item["start_s"] <= prev_end:
            overlaps_after += 1
        prev_end = item["end_s"]

    records = []
    for index, item in enumerate(merged, start=1):
        records.append(
            {
                "event_id": event_id,
                "event_name": EVENTS[event_id],
                "pilot_session_id": session_ref.pilot_session_id,
                "session_key": session_ref.session_key,
                "unit_id": unit_id or actor_id or counterpart_id or "ego",
                "actor_id": actor_id,
                "counterpart_id": counterpart_id,
                "interval_index": index,
                "start_s": f"{item['start_s']:.3f}",
                "end_s": f"{item['end_s']:.3f}",
                "duration_s": f"{item['duration_s']:.3f}",
                "candidate_or_proxy_guard": proxy_guard(event_id),
                "note": note,
                "primary_endpoint": "true" if primary_endpoint else "false",
                "identity_status": identity_status,
                "precedence_status": precedence_status,
                "precedence_reason": precedence_reason,
            }
        )
    stats = {
        "raw_frame_hits": int(sum(1 for flag in flags if flag)),
        "runs_before_duration": before_duration,
        "events_before_merge": len(accepted),
        "events_after_merge": len(merged),
        "duplicate_overlapping_events": max(0, len(accepted) - len(merged)),
        "overlaps_after_merge": overlaps_after,
        "impossible_values": impossible,
    }
    return records, stats


def merge_result(base: EventResult, records: List[Dict[str, Any]], stats: Dict[str, int]) -> None:
    base.intervals.extend(records)
    base.raw_frame_hits += stats.get("raw_frame_hits", 0)
    base.runs_before_duration += stats.get("runs_before_duration", 0)
    base.events_before_merge += stats.get("events_before_merge", 0)
    base.events_after_merge += stats.get("events_after_merge", 0)
    base.duplicate_overlapping_events += stats.get("duplicate_overlapping_events", 0)
    base.overlaps_after_merge += stats.get("overlaps_after_merge", 0)
    base.impossible_values += stats.get("impossible_values", 0)
    finalize_event_primary_counts(base)


def interval_is_primary(row: Dict[str, Any]) -> bool:
    return str(row.get("primary_endpoint", "true")).lower() == "true"


def finalize_event_primary_counts(result: EventResult) -> None:
    result.raw_event_count = len(result.intervals)
    result.primary_event_count = sum(1 for row in result.intervals if interval_is_primary(row))
    result.suppressed_by_precedence = result.raw_event_count - result.primary_event_count
    result.event_count = result.primary_event_count


def proxy_guard(event_id: str) -> str:
    if event_id == "E01":
        return "not emitted; frozen counterpart relation unavailable"
    if event_id == "E15":
        return "geometric contact candidate only; not sensor-confirmed collision"
    if event_id == "E16":
        return "no-progress only; off-route guarded off because route/lane/goal geometry is unavailable"
    if event_id == "E18":
        return "kinematic emergency-stop candidate only; no explicit e-stop command flag"
    return "automatic kinematic/geometric pilot output"


def add_actor_series_event_units(result: EventResult, rows_by_actor: Dict[str, List[Dict[str, Any]]], include_ego: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    series: Dict[str, List[Dict[str, Any]]] = {}
    if include_ego:
        actor_id = safe_str(include_ego[0].get("actor_id")) if include_ego else "ego"
        series[f"ego:{actor_id or 'ego'}"] = include_ego
    for actor_id, rows in rows_by_actor.items():
        if actor_id == "missing_actor_id":
            result.actor_attribution_failures += 1
            continue
        series[f"world:{actor_id}"] = rows
    result.total_units += len(series)
    return series


def extract_e01(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E01", EVENTS["E01"])
    world_units = {actor_id: decimate_rows(rows, decimate_factor) for actor_id, rows in session.world_by_actor.items() if actor_id != "missing_actor_id"}
    result.total_units += len(world_units)
    result.actor_attribution_failures += len(world_units)
    result.missing_data_failures += len(world_units)
    result.notes.append("Not computable in this pilot: no frozen counterpart relation is available; all non-ego actors are treated as unresolved for E01.")
    return result


def extract_e02(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E02", EVENTS["E02"])
    series = add_actor_series_event_units(
        result,
        {actor_id: decimate_rows(rows, decimate_factor) for actor_id, rows in session.world_by_actor.items()},
        decimate_rows(session.ego, decimate_factor),
    )
    for unit_id, rows in series.items():
        segments, missing, gaps, impossible = split_valid_segments(rows, ["t_ms", "speed", "accel"], params["missing_gap_s"])
        result.missing_data_failures += missing + gaps
        result.impossible_values += impossible
        if not segments:
            continue
        result.computable_units += 1
        for segment in segments:
            flags = [float(row["accel"]) <= -params["decel_mps2"] for row in segment]
            records, stats = interval_stats_and_records(
                segment,
                flags,
                params["d_min_s"],
                params["merge_gap_s"],
                "E02",
                session.ref,
                actor_id=unit_id,
                unit_id=unit_id,
            )
            merge_result(result, records, stats)
    return result


def extract_e03(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E03", EVENTS["E03"])
    series = add_actor_series_event_units(
        result,
        {actor_id: decimate_rows(rows, decimate_factor) for actor_id, rows in session.world_by_actor.items()},
        decimate_rows(session.ego, decimate_factor),
    )
    for unit_id, rows in series.items():
        segments, missing, gaps, impossible = split_valid_segments(rows, ["t_ms", "accel"], params["missing_gap_s"])
        result.missing_data_failures += missing + gaps
        result.impossible_values += impossible
        if not segments:
            continue
        result.computable_units += 1
        for segment in segments:
            times = [float(row["t_ms"]) / 1000.0 for row in segment]
            accels = [float(row["accel"]) for row in segment]
            smoothed = causal_rolling_median(accels, times, params["smooth_s"])
            jerks = derivative(smoothed, times)
            flags = [j is not None and abs(j) >= params["jerk_mps3"] for j in jerks]
            records, stats = interval_stats_and_records(
                segment,
                flags,
                params["d_min_s"],
                params["merge_gap_s"],
                "E03",
                session.ref,
                actor_id=unit_id,
                unit_id=unit_id,
            )
            merge_result(result, records, stats)
    return result


def qualifying_state_runs(segment: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not segment:
        return []
    dt_est = positive_median_dt_s(segment)
    states: List[Optional[str]] = []
    for row in segment:
        speed = float(row["speed"])
        if speed <= params["stop_speed_mps"]:
            states.append("stop")
        elif speed >= params["go_speed_mps"]:
            states.append("go")
        else:
            states.append(None)
    raw_runs: List[Tuple[str, int, int]] = []
    start = None
    current_state = None
    for idx, state in enumerate(states):
        if state is None:
            if current_state is not None and start is not None:
                raw_runs.append((current_state, start, idx - 1))
            current_state = None
            start = None
            continue
        if current_state is None:
            current_state = state
            start = idx
        elif state != current_state:
            raw_runs.append((current_state, start if start is not None else idx, idx - 1))
            current_state = state
            start = idx
    if current_state is not None and start is not None:
        raw_runs.append((current_state, start, len(states) - 1))

    qualifying = []
    for state, start_idx, end_idx in raw_runs:
        start_s = float(segment[start_idx]["t_ms"]) / 1000.0
        end_s = float(segment[end_idx]["t_ms"]) / 1000.0
        duration = end_s - start_s + dt_est
        min_duration = params["stop_min_s"] if state == "stop" else params["go_min_s"]
        if duration + 1e-9 >= min_duration:
            qualifying.append({"state": state, "start_idx": start_idx, "end_idx": end_idx, "start_s": start_s, "end_s": end_s})
    return qualifying


def extract_e06(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E06", EVENTS["E06"])
    series = add_actor_series_event_units(
        result,
        {actor_id: decimate_rows(rows, decimate_factor) for actor_id, rows in session.world_by_actor.items()},
        decimate_rows(session.ego, decimate_factor),
    )
    required_runs = int(params["n_cycles"]) * 2
    for unit_id, rows in series.items():
        segments, missing, gaps, impossible = split_valid_segments(rows, ["t_ms", "speed"], params["missing_gap_s"])
        result.missing_data_failures += missing + gaps
        result.impossible_values += impossible
        if not segments:
            continue
        result.computable_units += 1
        for segment in segments:
            q_runs = qualifying_state_runs(segment, params)
            flags = [False] * len(segment)
            result.runs_before_duration += len(q_runs)
            idx = 0
            accepted = []
            while idx <= len(q_runs) - required_runs:
                window = q_runs[idx : idx + required_runs]
                alternating = all(window[i]["state"] != window[i - 1]["state"] for i in range(1, len(window)))
                duration = window[-1]["end_s"] - window[0]["start_s"]
                if alternating and duration + 1e-9 >= params["d_min_s"]:
                    accepted.append((window[0]["start_idx"], window[-1]["end_idx"]))
                    for j in range(window[0]["start_idx"], window[-1]["end_idx"] + 1):
                        flags[j] = True
                    idx += required_runs
                else:
                    idx += 1
            records, stats = interval_stats_and_records(
                segment,
                flags,
                0.001,
                params["merge_gap_s"],
                "E06",
                session.ref,
                actor_id=unit_id,
                unit_id=unit_id,
            )
            stats["runs_before_duration"] += len(accepted)
            merge_result(result, records, stats)
    return result


def rect_corners(x: float, y: float, length: float, width: float, heading_deg: float) -> List[Tuple[float, float]]:
    theta = math.radians(heading_deg)
    ux = math.cos(theta)
    uy = math.sin(theta)
    vx = -uy
    vy = ux
    hl = length / 2.0
    hw = width / 2.0
    return [
        (x + ux * hl + vx * hw, y + uy * hl + vy * hw),
        (x + ux * hl - vx * hw, y + uy * hl - vy * hw),
        (x - ux * hl - vx * hw, y - uy * hl - vy * hw),
        (x - ux * hl + vx * hw, y - uy * hl + vy * hw),
    ]


def polygon_axes(poly: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    axes = []
    for (x1, y1), (x2, y2) in zip(poly, list(poly[1:]) + [poly[0]]):
        ex = x2 - x1
        ey = y2 - y1
        nx, ny = -ey, ex
        norm = math.hypot(nx, ny)
        if norm > 1e-12:
            axes.append((nx / norm, ny / norm))
    return axes


def project(poly: Sequence[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    dots = [x * axis[0] + y * axis[1] for x, y in poly]
    return min(dots), max(dots)


def point_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def edge_distance(poly_a: Sequence[Tuple[float, float]], poly_b: Sequence[Tuple[float, float]]) -> float:
    dists = []
    edges_a = list(zip(poly_a, list(poly_a[1:]) + [poly_a[0]]))
    edges_b = list(zip(poly_b, list(poly_b[1:]) + [poly_b[0]]))
    for point in poly_a:
        for a, b in edges_b:
            dists.append(point_segment_distance(point, a, b))
    for point in poly_b:
        for a, b in edges_a:
            dists.append(point_segment_distance(point, a, b))
    return min(dists) if dists else math.inf


def signed_rect_distance(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Optional[float]:
    needed = ["x_m", "y_m", "length_m", "width_m", "course_deg"]
    if any(not is_finite_number(row_a.get(k)) for k in needed) or any(not is_finite_number(row_b.get(k)) for k in needed):
        return None
    poly_a = rect_corners(float(row_a["x_m"]), float(row_a["y_m"]), float(row_a["length_m"]), float(row_a["width_m"]), float(row_a["course_deg"]))
    poly_b = rect_corners(float(row_b["x_m"]), float(row_b["y_m"]), float(row_b["length_m"]), float(row_b["width_m"]), float(row_b["course_deg"]))
    min_overlap = math.inf
    separated = False
    for axis in polygon_axes(poly_a) + polygon_axes(poly_b):
        a_min, a_max = project(poly_a, axis)
        b_min, b_max = project(poly_b, axis)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < 0:
            separated = True
            break
        min_overlap = min(min_overlap, overlap)
    if separated:
        return edge_distance(poly_a, poly_b)
    return -min_overlap if math.isfinite(min_overlap) else 0.0


def velocity(row: Dict[str, Any]) -> Tuple[float, float]:
    speed = float(row.get("speed") or 0.0)
    heading = math.radians(float(row.get("course_deg") or 0.0))
    return speed * math.cos(heading), speed * math.sin(heading)


def time_to_conflict_s(ego: Dict[str, Any], other: Dict[str, Any], signed_distance: float) -> Optional[float]:
    if not all(is_finite_number(ego.get(k)) for k in ("x_m", "y_m", "speed", "course_deg")):
        return None
    if not all(is_finite_number(other.get(k)) for k in ("x_m", "y_m", "speed", "course_deg")):
        return None
    rx = float(other["x_m"]) - float(ego["x_m"])
    ry = float(other["y_m"]) - float(ego["y_m"])
    dist = math.hypot(rx, ry)
    if dist <= 1e-9:
        return 0.0
    evx, evy = velocity(ego)
    ovx, ovy = velocity(other)
    rvx = ovx - evx
    rvy = ovy - evy
    closing = -((rx * rvx + ry * rvy) / dist)
    if closing <= 1e-6:
        return None
    clearance = max(signed_distance, 0.0)
    return clearance / closing


def nearest_ego_rows(ego_rows: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[Dict[str, Any]]]:
    valid = [row for row in ego_rows if is_finite_number(row.get("t_ms"))]
    valid.sort(key=lambda r: float(r["t_ms"]))
    return [float(row["t_ms"]) / 1000.0 for row in valid], valid


def nearest_by_time(times: Sequence[float], rows: Sequence[Dict[str, Any]], t_s: float, max_gap_s: float) -> Optional[Dict[str, Any]]:
    idx = bisect.bisect_left(times, t_s)
    best: Optional[Tuple[float, float, int, Dict[str, Any]]] = None
    for j in (idx - 1, idx):
        if 0 <= j < len(rows):
            gap = abs(times[j] - t_s)
            # Tie-break by lower timestamp, then stable row index, so equal
            # nearest-neighbor gaps are reproducible across Python versions.
            candidate = (gap, times[j], j, rows[j])
            if gap <= max_gap_s and (best is None or candidate[:3] < best[:3]):
                best = candidate
    return best[3] if best else None


def validate_pair_time_series(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """Return rows safe for pair alignment plus missing/impossible diagnostics.

    Pair events align two independent time bases, so timestamp ambiguity must be
    removed before nearest-neighbor matching. Missing timestamps are dropped.
    Non-monotonic rows in source order are dropped. Duplicate timestamps are
    treated as ambiguous and all rows at that timestamp are dropped rather than
    allowing a silent geometry choice.
    """

    missing = 0
    impossible = 0
    finite_rows: List[Dict[str, Any]] = []
    timestamp_counts: Dict[float, int] = defaultdict(int)
    for row in rows:
        if not is_finite_number(row.get("t_ms")):
            missing += 1
            continue
        timestamp_counts[float(row["t_ms"])] += 1
        finite_rows.append(row)

    duplicate_times = {t_ms for t_ms, count in timestamp_counts.items() if count > 1}
    impossible += sum(1 for row in finite_rows if float(row["t_ms"]) in duplicate_times)

    monotonic_rows: List[Dict[str, Any]] = []
    prev_t: Optional[float] = None
    for row in finite_rows:
        t_ms = float(row["t_ms"])
        if t_ms in duplicate_times:
            continue
        if prev_t is not None and t_ms <= prev_t:
            impossible += 1
            continue
        monotonic_rows.append(row)
        prev_t = t_ms

    monotonic_rows.sort(key=lambda r: float(r["t_ms"]))
    return monotonic_rows, missing, impossible


def actor_identity_signature(row: Dict[str, Any]) -> Tuple[str, str, str]:
    actor_id = safe_str(row.get("actor_id")) or "missing_actor_id"
    origin_id = safe_str(row.get("origin_id")) or ""
    name = safe_str(row.get("name")) or ""
    return actor_id, origin_id, name


def split_actor_identity_windows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Tuple[List[Dict[str, Any]], bool, str]], int]:
    """Split world rows when same actor_id carries changing identity evidence."""

    if not rows:
        return [], 0
    ordered = sorted(rows, key=lambda r: (float(r["t_ms"]) if is_finite_number(r.get("t_ms")) else -1.0))
    windows: List[Tuple[List[Dict[str, Any]], bool, str]] = []
    current: List[Dict[str, Any]] = []
    current_sig: Optional[Tuple[str, str, str]] = None
    signatures = {actor_identity_signature(row) for row in ordered}
    identity_unresolved = len(signatures) > 1

    for row in ordered:
        sig = actor_identity_signature(row)
        if current and current_sig is not None and sig != current_sig:
            windows.append((current, identity_unresolved, "identity changed within actor_id; window is diagnostic-only"))
            current = []
        current.append(row)
        current_sig = sig
    if current:
        reason = "identity changed within actor_id; window is diagnostic-only" if identity_unresolved else ""
        windows.append((current, identity_unresolved, reason))

    failures = len(windows) if identity_unresolved else 0
    return windows, failures


def extract_pair_event(
    session: SessionData,
    params: Dict[str, Any],
    event_id: str,
    decimate_factor: int,
    max_align_gap_s: float,
    contact_tolerance_m: Optional[float] = None,
) -> EventResult:
    result = EventResult(event_id, EVENTS[event_id])
    ego_rows, ego_missing, ego_impossible = validate_pair_time_series(decimate_rows(session.ego, decimate_factor))
    result.missing_data_failures += ego_missing
    result.impossible_values += ego_impossible
    if ego_impossible > 0:
        result.total_units += len(session.world_by_actor)
        result.missing_data_failures += len(session.world_by_actor)
        result.notes.append("Pair event not emitted: ambiguous ego timestamp series was rejected before alignment.")
        finalize_event_primary_counts(result)
        return result
    ego_times, ego_sorted = nearest_ego_rows(ego_rows)
    if contact_tolerance_m is None:
        active_contact_tolerance = 0.0
        include_contact_candidates_for_audit = False
    else:
        active_contact_tolerance = float(contact_tolerance_m)
        include_contact_candidates_for_audit = True
    if not ego_sorted:
        result.total_units += len(session.world_by_actor)
        result.missing_data_failures += len(session.world_by_actor)
        return result
    for actor_id, raw_rows in session.world_by_actor.items():
        if actor_id == "missing_actor_id":
            result.actor_attribution_failures += 1
            continue
        rows, world_missing, world_impossible = validate_pair_time_series(decimate_rows(raw_rows, decimate_factor))
        result.missing_data_failures += world_missing
        result.impossible_values += world_impossible
        result.total_units += 1
        if world_impossible > 0:
            result.missing_data_failures += 1
            result.notes.append(f"Pair actor world:{actor_id} not emitted: ambiguous world timestamp series was rejected before alignment.")
            continue
        identity_windows, attribution_failures = split_actor_identity_windows(rows)
        result.actor_attribution_failures += attribution_failures
        actor_computable = False
        for window_rows, identity_unresolved, identity_reason in identity_windows:
            pair_rows: List[Dict[str, Any]] = []
            missing = 0
            impossible = 0
            for row in window_rows:
                ego = nearest_by_time(ego_times, ego_sorted, float(row["t_ms"]) / 1000.0, max_align_gap_s)
                if ego is None:
                    missing += 1
                    continue
                ego_missing_row, ego_impossible_row, _ = row_quality_for_emission(
                    ego,
                    ["t_ms", "x_m", "y_m", "course_deg"],
                    None,
                    require_dimensions=True,
                )
                row_missing, row_impossible, _ = row_quality_for_emission(
                    row,
                    ["t_ms", "x_m", "y_m", "course_deg"],
                    None,
                    require_dimensions=True,
                )
                if ego_missing_row or row_missing or ego_impossible_row or row_impossible:
                    missing += int(ego_missing_row) + int(row_missing)
                    impossible += int(ego_impossible_row) + int(row_impossible)
                    continue
                signed_distance = signed_rect_distance(ego, row)
                if signed_distance is None:
                    missing += 1
                    continue
                if signed_distance < -30.0:
                    impossible += 1
                ttc = time_to_conflict_s(ego, row, signed_distance)
                pair_rows.append(
                    {
                        "t_ms": row["t_ms"],
                        "actor_id": actor_id,
                        "signed_distance": signed_distance,
                        "ttc": ttc,
                        "speed": row.get("speed"),
                        "length_m": row.get("length_m"),
                        "width_m": row.get("width_m"),
                    }
                )
            segments, missing_rows, gaps, seg_impossible = split_valid_segments(
                pair_rows,
                ["t_ms", "signed_distance"],
                params["missing_gap_s"],
                require_dimensions=True,
            )
            result.missing_data_failures += missing + missing_rows + gaps
            result.impossible_values += impossible + seg_impossible
            if not segments:
                continue
            actor_computable = True
            for segment in segments:
                if event_id == "E15":
                    flags = [float(row["signed_distance"]) <= params["overlap_tolerance_m"] for row in segment]
                else:
                    flags = []
                    for row in segment:
                        signed_distance = float(row["signed_distance"])
                        ttc = row.get("ttc")
                        contact = signed_distance <= active_contact_tolerance
                        near_distance = signed_distance <= params["distance_m"] and (
                            include_contact_candidates_for_audit or not contact
                        )
                        near_ttc = (
                            ttc is not None
                            and ttc <= params["time_to_conflict_s"]
                            and (include_contact_candidates_for_audit or not contact)
                        )
                        flags.append(near_distance or near_ttc)
                records, stats = interval_stats_and_records(
                    segment,
                    flags,
                    params["d_min_s"],
                    params["merge_gap_s"],
                    event_id,
                    session.ref,
                    actor_id="ego",
                    counterpart_id=f"world:{actor_id}",
                    unit_id=f"ego-world:{actor_id}",
                    note=identity_reason,
                    primary_endpoint=not identity_unresolved,
                    identity_status="identity_unresolved" if identity_unresolved else "stable",
                    precedence_status="identity_unresolved" if identity_unresolved else "primary",
                    precedence_reason=identity_reason,
                )
                merge_result(result, records, stats)
        if actor_computable:
            result.computable_units += 1
    finalize_event_primary_counts(result)
    return result


def extract_e16(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E16", EVENTS["E16"], total_units=1)
    rows = decimate_rows(session.ego, decimate_factor)
    segments, missing, gaps, impossible = split_valid_segments(rows, ["t_ms", "speed", "x_m", "y_m"], params["missing_gap_s"])
    result.missing_data_failures += missing + gaps
    result.impossible_values += impossible
    if not segments:
        return result
    result.computable_units = 1
    for segment in segments:
        times = [float(row["t_ms"]) / 1000.0 for row in segment]
        flags = [False] * len(segment)
        left = 0
        for i, t in enumerate(times):
            while left < i and t - times[left] > params["no_progress_s"]:
                left += 1
            if t - times[left] + 1e-9 < params["no_progress_s"]:
                continue
            start = segment[left]
            end = segment[i]
            displacement = math.hypot(float(end["x_m"]) - float(start["x_m"]), float(end["y_m"]) - float(start["y_m"]))
            speeds = [float(row["speed"]) for row in segment[left : i + 1] if is_finite_number(row.get("speed"))]
            max_speed = max(speeds) if speeds else math.inf
            flags[i] = displacement <= params["progress_displacement_m"] and max_speed <= params["progress_speed_mps"]
        records, stats = interval_stats_and_records(
            segment,
            flags,
            0.001,
            params["merge_gap_s"],
            "E16",
            session.ref,
            actor_id="ego",
            unit_id="ego",
            note="off-route subcase guarded off",
        )
        merge_result(result, records, stats)
    return result


def extract_e18(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E18", EVENTS["E18"], total_units=1)
    rows = decimate_rows(session.ego, decimate_factor)
    segments, missing, gaps, impossible = split_valid_segments(rows, ["t_ms", "speed", "accel"], params["missing_gap_s"])
    result.missing_data_failures += missing + gaps
    result.impossible_values += impossible
    if not segments:
        return result
    result.computable_units = 1
    for segment in segments:
        flags: List[bool] = []
        for row in segment:
            decel = float(row["accel"]) <= -params["emergency_decel_mps2"]
            stop = float(row["speed"]) <= params["stop_speed_mps"]
            brake = is_finite_number(row.get("braking")) and float(row["braking"]) > 0
            mode = params.get("brake_mode", "decel_plus_brake_or_stop")
            if mode == "strict_decel_to_stop":
                flags.append(decel and stop)
            elif mode == "broad_decel_or_brake":
                flags.append(decel or brake)
            else:
                flags.append(decel and (brake or stop))
        records, stats = interval_stats_and_records(
            segment,
            flags,
            params["d_min_s"],
            params["merge_gap_s"],
            "E18",
            session.ref,
            actor_id="ego",
            unit_id="ego",
        )
        merge_result(result, records, stats)
    return result


def extract_e19(session: SessionData, params: Dict[str, Any], decimate_factor: int) -> EventResult:
    result = EventResult("E19", EVENTS["E19"], total_units=1)
    rows = decimate_rows(session.ego, decimate_factor)
    required = ["t_ms", "speed", "lat_acc"]
    segments, missing, gaps, impossible = split_valid_segments(rows, required, params["missing_gap_s"])
    result.missing_data_failures += missing + gaps
    result.impossible_values += impossible
    if not segments:
        return result
    result.computable_units = 1
    for segment in segments:
        times = [float(row["t_ms"]) / 1000.0 for row in segment]
        lat_acc = [float(row["lat_acc"]) for row in segment]
        steering = [
            safe_float(row.get("steering_angle")) if safe_float(row.get("steering_angle")) is not None else safe_float(row.get("wheel_angle"))
            for row in segment
        ]
        smooth_lat = causal_rolling_median(lat_acc, times, params["smooth_s"])
        smooth_steering = causal_rolling_median(steering, times, params["smooth_s"])
        lat_jerk = derivative(smooth_lat, times)
        steer_rate = derivative(smooth_steering, times)
        flags = []
        for a_lat, j_lat, s_rate in zip(smooth_lat, lat_jerk, steer_rate):
            flags.append(
                (a_lat is not None and abs(a_lat) >= params["lateral_acc_mps2"])
                or (j_lat is not None and abs(j_lat) >= params["lateral_jerk_mps3"])
                or (s_rate is not None and abs(s_rate) >= params["steering_rate_dps"])
            )
        records, stats = interval_stats_and_records(
            segment,
            flags,
            params["d_min_s"],
            params["merge_gap_s"],
            "E19",
            session.ref,
            actor_id="ego",
            unit_id="ego",
        )
        merge_result(result, records, stats)
    return result


def interval_seconds(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def overlap_bounds(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    start = max(interval_seconds(a, "start_s"), interval_seconds(b, "start_s"))
    end = min(interval_seconds(a, "end_s"), interval_seconds(b, "end_s"))
    if start <= end + 1e-9:
        return start, end
    return None


def is_ego_interval(row: Dict[str, Any]) -> bool:
    values = {safe_str(row.get("unit_id")), safe_str(row.get("actor_id")), safe_str(row.get("counterpart_id"))}
    return "ego" in values or "ego:ego" in values


def actor_tokens(row: Dict[str, Any]) -> set:
    tokens = set()
    for key in ("unit_id", "actor_id", "counterpart_id"):
        value = safe_str(row.get(key))
        if value:
            tokens.add(value)
            if value.startswith("ego-world:"):
                tokens.add("world:" + value.split("ego-world:", 1)[1])
    return tokens


def same_pair_unit(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        a.get("pilot_session_id") == b.get("pilot_session_id")
        and safe_str(a.get("unit_id")) == safe_str(b.get("unit_id"))
        and safe_str(a.get("counterpart_id")) == safe_str(b.get("counterpart_id"))
    )


def same_ego_unit(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.get("pilot_session_id") == b.get("pilot_session_id") and is_ego_interval(a) and is_ego_interval(b)


def same_actor_window(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.get("pilot_session_id") == b.get("pilot_session_id") and bool(actor_tokens(a) & actor_tokens(b))


def reset_primary_flags(results: Dict[str, EventResult]) -> None:
    for result in results.values():
        for row in result.intervals:
            if row.get("identity_status") == "identity_unresolved":
                row["primary_endpoint"] = "false"
                row["precedence_status"] = "identity_unresolved"
                row["precedence_reason"] = row.get("precedence_reason") or "actor identity unresolved"
            else:
                row["primary_endpoint"] = "true"
                row["precedence_status"] = "primary"
                row["precedence_reason"] = ""


def suppress_interval(row: Dict[str, Any], status: str, reason: str) -> bool:
    was_primary = interval_is_primary(row)
    if was_primary:
        row["primary_endpoint"] = "false"
        row["precedence_status"] = status
        row["precedence_reason"] = reason
    return was_primary


def audit_overlap_row(
    relation: str,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    overlap: Tuple[float, float],
    suppressed_event: str,
    kept_event: str,
    rule: str,
) -> Dict[str, Any]:
    start, end = overlap
    return {
        "relation": relation,
        "event_a": row_a.get("event_id", ""),
        "event_b": row_b.get("event_id", ""),
        "pilot_session_id": row_a.get("pilot_session_id", ""),
        "session_key": row_a.get("session_key", ""),
        "unit_id_a": row_a.get("unit_id", ""),
        "unit_id_b": row_b.get("unit_id", ""),
        "actor_id_a": row_a.get("actor_id", ""),
        "actor_id_b": row_b.get("actor_id", ""),
        "counterpart_id_a": row_a.get("counterpart_id", ""),
        "counterpart_id_b": row_b.get("counterpart_id", ""),
        "interval_index_a": row_a.get("interval_index", ""),
        "interval_index_b": row_b.get("interval_index", ""),
        "start_s_a": row_a.get("start_s", ""),
        "end_s_a": row_a.get("end_s", ""),
        "start_s_b": row_b.get("start_s", ""),
        "end_s_b": row_b.get("end_s", ""),
        "overlap_start_s": f"{start:.3f}",
        "overlap_end_s": f"{end:.3f}",
        "overlap_duration_s": f"{max(0.0, end - start):.3f}",
        "suppressed_event": suppressed_event,
        "kept_event": kept_event,
        "precedence_rule": rule,
    }


def interval_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("event_id", "")),
        str(row.get("pilot_session_id", "")),
        str(row.get("unit_id", "")),
        str(row.get("interval_index", "")),
        str(row.get("start_s", "")),
    )


def cross_event_precedence(results: Dict[str, EventResult]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply subset/precedence rules and return detail plus summary audit rows."""

    reset_primary_flags(results)
    audit_rows: List[Dict[str, Any]] = []
    suppressed_by_relation: Dict[str, set] = defaultdict(set)

    e01_rows = results.get("E01", EventResult("E01", EVENTS["E01"])).intervals
    e02_rows = results.get("E02", EventResult("E02", EVENTS["E02"])).intervals
    e09_rows = results.get("E09", EventResult("E09", EVENTS["E09"])).intervals
    e15_rows = results.get("E15", EventResult("E15", EVENTS["E15"])).intervals
    e18_rows = results.get("E18", EventResult("E18", EVENTS["E18"])).intervals

    for e01 in e01_rows:
        for e02 in e02_rows:
            overlap = overlap_bounds(e01, e02)
            if overlap is None or not same_actor_window(e01, e02):
                continue
            reason = "E01 is a counterpart-attributed subset of E02 and is deferred until a frozen counterpart relation exists"
            if suppress_interval(e01, "suppressed_subset_of_E02", reason):
                suppressed_by_relation["E01/E02"].add(interval_key(e01))
            audit_rows.append(audit_overlap_row("E01/E02", e01, e02, overlap, "E01", "E02", reason))

    for e02 in e02_rows:
        for e18 in e18_rows:
            overlap = overlap_bounds(e02, e18)
            if overlap is None or not same_ego_unit(e02, e18):
                continue
            reason = "E18 ego hard-stop precedence over overlapping ego E02 deceleration"
            if suppress_interval(e02, "suppressed_by_E18", reason):
                suppressed_by_relation["E02/E18"].add(interval_key(e02))
            audit_rows.append(audit_overlap_row("E02/E18", e02, e18, overlap, "E02", "E18", reason))

    for e09 in e09_rows:
        for e15 in e15_rows:
            overlap = overlap_bounds(e09, e15)
            if overlap is None or not same_pair_unit(e09, e15):
                continue
            reason = "E15 contact precedence over same-pair/time E09 near-miss candidate under the active band"
            if suppress_interval(e09, "suppressed_by_E15", reason):
                suppressed_by_relation["E09/E15"].add(interval_key(e09))
            audit_rows.append(audit_overlap_row("E09/E15", e09, e15, overlap, "E09", "E15", reason))

    for result in results.values():
        finalize_event_primary_counts(result)

    summary_rows = []
    relation_specs = [
        ("E01/E02", "E01", "E02", "E01 subset of E02; E01 remains deferred without frozen counterpart relation"),
        ("E02/E18", "E02", "E18", "E18 takes primary precedence for overlapping ego hard-stop windows"),
        ("E09/E15", "E09", "E15", "E15 takes primary precedence for same-pair/time contact windows"),
    ]
    for relation, event_a, event_b, rule in relation_specs:
        relation_rows = [row for row in audit_rows if row["relation"] == relation]
        a_result = results.get(event_a, EventResult(event_a, EVENTS[event_a]))
        b_result = results.get(event_b, EventResult(event_b, EVENTS[event_b]))
        summary_rows.append(
            {
                "relation": relation,
                "event_a": event_a,
                "event_b": event_b,
                "overlap_rows": len(relation_rows),
                "suppressed_intervals": len(suppressed_by_relation.get(relation, set())),
                "event_a_raw_count": a_result.raw_event_count,
                "event_a_primary_count": a_result.primary_event_count,
                "event_b_raw_count": b_result.raw_event_count,
                "event_b_primary_count": b_result.primary_event_count,
                "precedence_rule": rule,
            }
        )
    return audit_rows, summary_rows


def extract_all_events(
    sessions: Sequence[SessionData],
    params_by_event: Dict[str, Dict[str, Any]],
    decimate_factor: int,
    max_align_gap_s: float,
) -> Dict[str, EventResult]:
    results = {event_id: EventResult(event_id, EVENTS[event_id]) for event_id in EVENTS}
    for session in sessions:
        session_results = {
            "E01": extract_e01(session, params_by_event["E01"], decimate_factor),
            "E02": extract_e02(session, params_by_event["E02"], decimate_factor),
            "E03": extract_e03(session, params_by_event["E03"], decimate_factor),
            "E06": extract_e06(session, params_by_event["E06"], decimate_factor),
            "E15": extract_pair_event(session, params_by_event["E15"], "E15", decimate_factor, max_align_gap_s),
            "E09": extract_pair_event(
                session,
                params_by_event["E09"],
                "E09",
                decimate_factor,
                max_align_gap_s,
                contact_tolerance_m=float(params_by_event["E15"]["overlap_tolerance_m"]),
            ),
            "E16": extract_e16(session, params_by_event["E16"], decimate_factor),
            "E18": extract_e18(session, params_by_event["E18"], decimate_factor),
            "E19": extract_e19(session, params_by_event["E19"], decimate_factor),
        }
        for event_id, partial in session_results.items():
            target = results[event_id]
            target.total_units += partial.total_units
            target.computable_units += partial.computable_units
            target.event_count += partial.event_count
            target.raw_event_count += partial.raw_event_count
            target.primary_event_count += partial.primary_event_count
            target.suppressed_by_precedence += partial.suppressed_by_precedence
            target.raw_frame_hits += partial.raw_frame_hits
            target.runs_before_duration += partial.runs_before_duration
            target.events_before_merge += partial.events_before_merge
            target.events_after_merge += partial.events_after_merge
            target.duplicate_overlapping_events += partial.duplicate_overlapping_events
            target.overlaps_after_merge += partial.overlaps_after_merge
            target.impossible_values += partial.impossible_values
            target.actor_attribution_failures += partial.actor_attribution_failures
            target.missing_data_failures += partial.missing_data_failures
            target.intervals.extend(partial.intervals)
            target.notes.extend(partial.notes)
            target.cross_event_audit.extend(partial.cross_event_audit)
    audit_rows, _ = cross_event_precedence(results)
    for result in results.values():
        result.cross_event_audit = [row for row in audit_rows if row["event_a"] == result.event_id or row["event_b"] == result.event_id]
    return results


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def result_metric_rows(
    central: Dict[str, EventResult],
    threshold_counts: Dict[str, Dict[str, int]],
    sampling_counts: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:
    rows = []
    for event_id in EVENTS:
        result = central[event_id]
        low = threshold_counts.get("low", {}).get(event_id, 0)
        center = threshold_counts.get("central", {}).get(event_id, 0)
        high = threshold_counts.get("high", {}).get(event_id, 0)
        original = sampling_counts.get("original", {}).get(event_id, 0)
        decimated = sampling_counts.get("decimate2", {}).get(event_id, 0)
        delta = decimated - original
        rel_delta = "" if original == 0 else f"{delta / original:.6f}"
        dup_rate = 0.0 if result.events_before_merge == 0 else result.duplicate_overlapping_events / result.events_before_merge
        reason = ""
        if result.computable_units == 0:
            reason = "; ".join(sorted(set(result.notes))) or "No computable units in pilot sample."
        rows.append(
            {
                "event_id": event_id,
                "event_name": result.event_name,
                "computable_units": result.computable_units,
                "total_units": result.total_units,
                "computable_fraction": f"{result.computable_fraction:.6f}",
                "event_count": result.event_count,
                "raw_event_count": result.raw_event_count,
                "primary_event_count": result.primary_event_count,
                "suppressed_by_precedence": result.suppressed_by_precedence,
                "event_rate_per_computable_unit": f"{result.event_rate_per_unit:.6f}",
                "event_frequency": f"{result.event_count} events / {result.computable_units} computable units",
                "raw_frame_hits": result.raw_frame_hits,
                "runs_before_duration": result.runs_before_duration,
                "events_before_merge": result.events_before_merge,
                "events_after_merge": result.events_after_merge,
                "duplicate_overlapping_events": result.duplicate_overlapping_events,
                "duplicate_rate_before_after": f"before={result.events_before_merge}; after={result.events_after_merge}; merged_duplicate_rate={dup_rate:.6f}; post_merge_overlaps={result.overlaps_after_merge}",
                "impossible_values": result.impossible_values,
                "sampling_rate_sensitivity": f"original={original}; decimate2={decimated}; delta={delta}; relative_delta={rel_delta}",
                "threshold_sensitivity_low_central_high": f"low={low}; central={center}; high={high}",
                "actor_attribution_failures": result.actor_attribution_failures,
                "missing_data_failures": result.missing_data_failures,
                "not_computable_reason": reason,
                "candidate_or_proxy_guard": proxy_guard(event_id),
            }
        )
    return rows


def write_spec(path: Path) -> None:
    text = f"""# Automatic Event Extractor Spec

Run ID: {RUN_ID}
Worker: {WORKER_ID}
Scope: small outcome-blind extractor pilot for the nine retained automatic events.

## Firewall

This extractor reads only OnSite trajectory, geometry, time, and actor-id fields,
plus RQ003 annotation item IDs used solely to exclude already-used annotation
items from the pilot sample. It does not read IPV, deviation, official
coordination scores, ranks, team identities, human labels, agreement files, or
event-outcome associations. No event-IPV/outcome association is computed.

## Shared Implementation Rules

- Parameters are loaded from `extractor_config.json`, copied from the frozen
  confirmatory thresholds and sensitivity bands in `event_threshold_rationale.md`.
- Frame-level detections are converted to runs, filtered by the frozen
  minimum-duration parameter, then merged by the frozen same-actor or same-pair
  merge-gap parameter.
- Sampling-rate sensitivity is measured by rerunning central thresholds after
  deterministic per-actor decimation by a factor of two.
- Threshold sensitivity is measured by rerunning low, central, and high bands.
- Dimensions above 30 source units are treated as centimeters and divided by 100
  before geometric footprint calculations.
- Before any event flags or intervals are emitted, each actor or pair row passes
  a shared emission-quality guard. Rows with missing required fields, NaN/inf
  kinematics, negative speed magnitudes, duplicate/non-increasing timestamps
  after per-series timestamp ordering, or impossible geometry dimensions when
  geometry is required are excluded from emission and split the current segment.
  Excluded rows still increment the missing-data or impossible-value diagnostics.
- Geometry uses current-frame oriented rectangle footprints and nearest ego/world
  alignment within 100 ms. Before pair alignment, ego and world time series are
  prevalidated: missing timestamps are dropped, non-monotonic rows are rejected,
  duplicate timestamps are treated as ambiguous and removed, impossible-value
  diagnostics are recorded, and tied nearest-neighbor matches choose the lower
  timestamp deterministically.
- Pair events require stable world actor identity over the emitted interval.
  Rows are split when the same actor ID carries changing originId/name evidence;
  affected windows increment actor-attribution failures and any resulting pair
  intervals are diagnostic-only rather than primary endpoints.
- Cross-event primary endpoint precedence is applied after raw interval
  extraction: E01 is a counterpart-attributed subset of E02 and remains deferred,
  E18 takes precedence over overlapping ego E02 hard-stop windows, and E15
  contact takes precedence over same-pair/time E09 near-miss candidates under
  the active band. The audit table preserves the overlapping raw intervals.

## Event Rules

| event_id | computed rule | unit of analysis | mode | limitations |
|---|---|---|---|---|
| E01 | Counterpart hard braking would require a frozen counterpart relation plus deceleration persistence. | counterpart actor-window | online | Not computable in this pilot because no frozen counterpart identity relation is available. Non-ego actors are not promoted to counterpart status. |
| E02 | Actor acceleration <= -T_decel for D_min, merged by G_merge. | ego/world actor-window | online | Uses direct acceleration field only; no noncausal interpolation. |
| E03 | Causal rolling-median acceleration, backward-difference jerk, abs(jerk) >= T_jerk for D_min. | ego/world actor-window | online | Derivative is sensitive to gaps and timestamp jitter. |
| E06 | Stop/go states from speed thresholds; alternating qualifying stop/go runs meet N_cycles and D_min. | ego/world actor-window | online | Does not infer traffic-control or route context. |
| E09 | Ego-other oriented-footprint clearance <= T_distance or constant-velocity TTC <= T_time_to_conflict; same-pair/time E15 contact windows suppress E09 only in primary endpoint counts and are retained in the cross-event audit. | ego-other pair-window | online | Near-miss proxy only; no future trajectory or outcome is used. |
| E15 | Ego-other oriented-footprint signed distance <= T_overlap_tolerance for D_min. | ego-other pair-window | online | Geometric contact candidate only; not sensor-confirmed collision. |
| E16 | Ego no-progress if rolling causal window displacement and speed remain below frozen thresholds for D_no_progress. | ego session-window | online | Off-route subcase is guarded off until route/lane/goal geometry is frozen. |
| E18 | Ego hard deceleration plus braking/stop-state rule per band, D_min, and merge-gap. | ego session-window | online | Kinematic emergency-stop candidate only; no explicit e-stop flag/dictionary. |
| E19 | Ego lateral acceleration, lateral jerk, or steering-rate exceeds frozen comfort thresholds after causal smoothing. | ego session-window | online | Does not infer lane-change intent or route curvature. |

## Known Parameters Needing RQ011/Future Freezes

- E01 remains blocked by missing frozen counterpart relation.
- E16 off-route remains blocked by missing route/lane/goal geometry.
- E15 and E09 are geometric proxies; sensor contact flags or authoritative
  collision dictionaries would be a future signal freeze, not a pilot tuning item.
- E18 remains a kinematic candidate until an explicit emergency-stop command flag
  or status dictionary exists.
"""
    path.write_text(text, encoding="utf-8")


def write_report(
    path: Path,
    selected: Sequence[SessionRef],
    excluded: Sequence[SessionRef],
    seed: int,
    metrics: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
    sampling_rows: Sequence[Dict[str, Any]],
    audit_summary_rows: Sequence[Dict[str, Any]],
) -> None:
    computable_summary = ", ".join(f"{row['event_id']}={row['computable_fraction']}" for row in metrics)
    total_before = sum(int(row["events_before_merge"]) for row in metrics)
    total_after = sum(int(row["events_after_merge"]) for row in metrics)
    total_dups = sum(int(row["duplicate_overlapping_events"]) for row in metrics)
    text_lines = [
        "# Automatic Event Extractor Pilot Report",
        "",
        f"Run ID: {RUN_ID}",
        f"Worker: {WORKER_ID}",
        "",
        "## Firewall Statement",
        "",
        "No IPV, deviation, official coordination score, rank, team identity, human label, agreement result, or event-IPV/outcome association was read or computed. RQ003 manifests were used only for item-ID exclusion, and OnSite trajectory logs were used only for kinematics, geometry, timestamps, and actor IDs.",
        "",
        "## Sample",
        "",
        f"Recorded seed: {seed}",
        f"Selected pilot sessions: {len(selected)}",
        f"RQ003-excluded sessions: {len(excluded)}",
        "",
        "| pilot_session_id | session_key | rq003_exclusion_matches |",
        "|---|---|---|",
    ]
    for ref in selected:
        text_lines.append(f"| {ref.pilot_session_id} | {ref.session_key} | {';'.join(ref.rq003_public_replay_id_matches)} |")
    text_lines.extend(
        [
            "",
            "## Health Metrics",
            "",
            f"Computable fraction summary: {computable_summary}",
            f"Duplicate merge summary: before={total_before}; after={total_after}; merged_duplicates={total_dups}; post-merge overlaps are reported per event in the CSV.",
            "",
            "| event_id | computable_fraction | raw_count | primary_count | suppressed_by_precedence | duplicate_rate_before_after | impossible_values | actor_attribution_failures | missing_data_failures | guard |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in metrics:
        text_lines.append(
            f"| {row['event_id']} | {row['computable_fraction']} | {row['raw_event_count']} | {row['primary_event_count']} | {row['suppressed_by_precedence']} | {row['duplicate_rate_before_after']} | {row['impossible_values']} | {row['actor_attribution_failures']} | {row['missing_data_failures']} | {row['candidate_or_proxy_guard']} |"
        )
    text_lines.extend(
        [
            "",
            "## Cross-Event Duplicate And Precedence Audit",
            "",
            "Primary endpoint counts below apply the frozen hierarchy without deleting raw diagnostic intervals. Full overlap details are written to `cross_event_audit.csv`.",
            "",
            "| relation | overlap_rows | suppressed_intervals | event_a_raw | event_a_primary | event_b_raw | event_b_primary | precedence_rule |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in audit_summary_rows:
        text_lines.append(
            f"| {row['relation']} | {row['overlap_rows']} | {row['suppressed_intervals']} | {row['event_a_raw_count']} | {row['event_a_primary_count']} | {row['event_b_raw_count']} | {row['event_b_primary_count']} | {row['precedence_rule']} |"
        )
    text_lines.extend(
        [
            "",
            "## Phase 5 Fix Audit",
            "",
            "W12 found that rows already counted as impossible could still enter event flag construction. The extractor now applies a shared emission-quality guard before event flags or intervals are built; invalid rows split segments and remain counted in health metrics.",
            "",
            "W17a adds pair-event timestamp prevalidation, actor identity stability guards, and cross-event primary-endpoint precedence for E01/E02, E02/E18, and E09/E15. Raw counts remain available for diagnostics; primary counts are de-overlapped.",
            "",
            "W12 E02 repro closure: pre-fix speed=-1.0 and accel=-3.5 for 0.3 s emitted 1 interval with raw_hits=3 and impossible_values=3; post-fix it emits 0 intervals with raw_hits=0 and impossible_values=3.",
            "",
            "Same-seed pilot comparison against the pre-fix central rerun. `post_primary_count` is the de-overlapped endpoint count used for primary reporting.",
            "",
            "| event_id | event_count_pre | post_raw_count | post_primary_count | primary_delta_vs_pre | raw_hits_pre | raw_hits_post | raw_hits_delta | impossible_pre | impossible_post | missing_pre | missing_post |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        pre = PRE_FIX_CENTRAL_AUDIT[row["event_id"]]
        raw_count_post = int(row["raw_event_count"])
        primary_count_post = int(row["primary_event_count"])
        raw_hits_post = int(row["raw_frame_hits"])
        impossible_post = int(row["impossible_values"])
        missing_post = int(row["missing_data_failures"])
        text_lines.append(
            f"| {row['event_id']} | {pre['event_count']} | {raw_count_post} | {primary_count_post} | {primary_count_post - pre['event_count']} | {pre['raw_frame_hits']} | {raw_hits_post} | {raw_hits_post - pre['raw_frame_hits']} | {pre['impossible_values']} | {impossible_post} | {pre['missing_data_failures']} | {missing_post} |"
        )
    text_lines.extend(
        [
            "",
            "## Threshold Sensitivity",
            "",
            "Band labels follow the recorded sensitivity bands exactly. Counts are not necessarily monotone where a band changes categorical logic, such as E18 brake-mode support.",
            "",
            "| event_id | low_count | central_count | high_count |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        text_lines.append(f"| {row['event_id']} | {row['low_count']} | {row['central_count']} | {row['high_count']} |")
    text_lines.extend(
        [
            "",
            "## Sampling-Rate Sensitivity",
            "",
            "| event_id | original_count | decimate2_count | delta | relative_delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sampling_rows:
        text_lines.append(
            f"| {row['event_id']} | {row['original_count']} | {row['decimate2_count']} | {row['delta']} | {row['relative_delta']} |"
        )
    text_lines.extend(
        [
            "",
            "## Proxy And Candidate Guards",
            "",
            "- E15 is emitted only as a geometric contact candidate, not a sensor-confirmed collision.",
            "- E16 emits only no-progress timeout; off-route extraction is guarded off.",
            "- E18 is emitted only as a kinematic emergency-stop candidate, not an explicit e-stop command.",
            "- E01 is attempted but not computable because the frozen counterpart relation is unavailable in the pilot inputs.",
            "",
            "## Interpretation Boundary",
            "",
            "These are data-health and extractor-stability metrics only. Counts are not interpreted as outcomes, labels, agreement, or IPV-related evidence, and no threshold was tuned to the pilot results.",
        ]
    )
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def test_logs_pass(pilot_dir: Path) -> bool:
    expected_logs = [
        pilot_dir / "tests" / "test_run_log.json",
        pilot_dir / "tests" / "robustness_test_run_log.json",
    ]
    for path in expected_logs:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if int(payload.get("tests_failed", 1)) != 0:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("extractor_config.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = repo_root_from_script()
    run_root = repo_root / "reports" / "studies" / "RQ012_onsite_event_annotation_readiness" / RUN_ID
    results_dir = run_root / "01_results"
    pilot_dir = run_root / "02_process" / "05_extractor_pilot"
    derived_dir = repo_root / "data" / "derived" / "onsite_competition" / "RQ012_onsite_event_annotation_readiness" / RUN_ID / "extractor_pilot"
    onsite_root = repo_root / "data" / "onsite_competition"
    rq003_annotation_dir = repo_root / "reports" / "studies" / "RQ003_nsfc_external_evidence" / "RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424" / "01_results" / "annotations"

    selected, excluded, rq003_ids = select_pilot_sample(
        onsite_root=onsite_root,
        rq003_annotation_dir=rq003_annotation_dir,
        seed=int(config["pilot_seed"]),
        sample_n=int(config["sample_n_sessions"]),
        derived_dir=derived_dir,
    )
    sessions = [load_session(ref) for ref in selected]

    central = extract_all_events(
        sessions,
        config["bands"]["central"],
        decimate_factor=1,
        max_align_gap_s=float(config["max_time_alignment_gap_s"]),
    )
    cross_event_audit_rows, cross_event_audit_summary = cross_event_precedence(central)
    threshold_results = {
        band: extract_all_events(
            sessions,
            config["bands"][band],
            decimate_factor=1,
            max_align_gap_s=float(config["max_time_alignment_gap_s"]),
        )
        for band in ("low", "central", "high")
    }
    sampling_results = {
        "original": central,
        "decimate2": extract_all_events(
            sessions,
            config["bands"]["central"],
            decimate_factor=2,
            max_align_gap_s=float(config["max_time_alignment_gap_s"]),
        ),
    }
    threshold_counts = {band: {event_id: res.event_count for event_id, res in results.items()} for band, results in threshold_results.items()}
    sampling_counts = {name: {event_id: res.event_count for event_id, res in results.items()} for name, results in sampling_results.items()}
    metrics = result_metric_rows(central, threshold_counts, sampling_counts)

    interval_rows: List[Dict[str, Any]] = []
    for event_id in EVENTS:
        interval_rows.extend(central[event_id].intervals)
    write_csv(
        derived_dir / "event_intervals_central.csv",
        interval_rows,
        [
            "event_id",
            "event_name",
            "pilot_session_id",
            "session_key",
            "unit_id",
            "actor_id",
            "counterpart_id",
            "interval_index",
            "start_s",
            "end_s",
            "duration_s",
            "candidate_or_proxy_guard",
            "note",
            "primary_endpoint",
            "identity_status",
            "precedence_status",
            "precedence_reason",
        ],
    )
    cross_event_audit_fields = [
        "relation",
        "event_a",
        "event_b",
        "pilot_session_id",
        "session_key",
        "unit_id_a",
        "unit_id_b",
        "actor_id_a",
        "actor_id_b",
        "counterpart_id_a",
        "counterpart_id_b",
        "interval_index_a",
        "interval_index_b",
        "start_s_a",
        "end_s_a",
        "start_s_b",
        "end_s_b",
        "overlap_start_s",
        "overlap_end_s",
        "overlap_duration_s",
        "suppressed_event",
        "kept_event",
        "precedence_rule",
    ]
    write_csv(derived_dir / "cross_event_audit.csv", cross_event_audit_rows, cross_event_audit_fields)
    write_csv(
        derived_dir / "cross_event_audit_summary.csv",
        cross_event_audit_summary,
        [
            "relation",
            "event_a",
            "event_b",
            "overlap_rows",
            "suppressed_intervals",
            "event_a_raw_count",
            "event_a_primary_count",
            "event_b_raw_count",
            "event_b_primary_count",
            "precedence_rule",
        ],
    )

    metric_fields = [
        "event_id",
        "event_name",
        "computable_units",
        "total_units",
        "computable_fraction",
        "event_count",
        "raw_event_count",
        "primary_event_count",
        "suppressed_by_precedence",
        "event_rate_per_computable_unit",
        "event_frequency",
        "raw_frame_hits",
        "runs_before_duration",
        "events_before_merge",
        "events_after_merge",
        "duplicate_overlapping_events",
        "duplicate_rate_before_after",
        "impossible_values",
        "sampling_rate_sensitivity",
        "threshold_sensitivity_low_central_high",
        "actor_attribution_failures",
        "missing_data_failures",
        "not_computable_reason",
        "candidate_or_proxy_guard",
    ]
    write_csv(results_dir / "automatic_event_pilot.csv", metrics, metric_fields)
    write_csv(derived_dir / "automatic_event_pilot_metrics.csv", metrics, metric_fields)

    threshold_rows = []
    for event_id in EVENTS:
        threshold_rows.append(
            {
                "event_id": event_id,
                "event_name": EVENTS[event_id],
                "low_count": threshold_counts["low"][event_id],
                "central_count": threshold_counts["central"][event_id],
                "high_count": threshold_counts["high"][event_id],
            }
        )
    write_csv(derived_dir / "threshold_sensitivity.csv", threshold_rows, ["event_id", "event_name", "low_count", "central_count", "high_count"])

    sampling_rows = []
    for event_id in EVENTS:
        original = sampling_counts["original"][event_id]
        decimated = sampling_counts["decimate2"][event_id]
        delta = decimated - original
        sampling_rows.append(
            {
                "event_id": event_id,
                "event_name": EVENTS[event_id],
                "original_count": original,
                "decimate2_count": decimated,
                "delta": delta,
                "relative_delta": "" if original == 0 else f"{delta / original:.6f}",
            }
        )
    write_csv(derived_dir / "sampling_rate_sensitivity.csv", sampling_rows, ["event_id", "event_name", "original_count", "decimate2_count", "delta", "relative_delta"])

    session_health_rows = []
    for session in sessions:
        row = {"pilot_session_id": session.ref.pilot_session_id, "session_key": session.ref.session_key}
        row.update(session.health)
        session_health_rows.append(row)
    health_fields = sorted({key for row in session_health_rows for key in row.keys()})
    write_csv(derived_dir / "session_load_health.csv", session_health_rows, health_fields)

    read_audit = {
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "rq003_item_id_count": len(rq003_ids),
        "read_scope_statement": "Read OnSite trajectory logs and RQ003 item IDs for exclusion only.",
        "denied_data_statement": "No IPV, official score, rank, team identity, human label, agreement, or event-outcome association was read or computed by this extractor.",
        "selected_pilot_session_ids": [ref.pilot_session_id for ref in selected],
        "rq003_excluded_pilot_session_ids": [ref.pilot_session_id for ref in excluded],
    }
    (derived_dir / "read_scope_audit.json").write_text(json.dumps(read_audit, indent=2), encoding="utf-8")

    write_spec(results_dir / "automatic_event_extractor_spec.md")
    write_report(
        results_dir / "automatic_event_pilot_report.md",
        selected,
        excluded,
        int(config["pilot_seed"]),
        metrics,
        threshold_rows,
        sampling_rows,
        cross_event_audit_summary,
    )

    tests_pass = test_logs_pass(pilot_dir)
    status = {
        "phase": "phase5_robustness",
        "deliverable": "extractor_pilot",
        "verdict": "complete" if len(metrics) == len(EVENTS) and selected else "partial",
        "w12_high_finding": "addressed",
        "red_team_fixed": ["V01", "V03", "V04", "V05"],
        "tests_pass": tests_pass,
        "sample_n_sessions": len(selected),
        "run_id": RUN_ID,
        "git_head": git_head(repo_root),
        "worker_id": "RQ012-W17a-extractor-robustness",
        "no_ipv_outcome_association_computed": True,
        "rq003_excluded_sessions": len(excluded),
        "results": {
            "automatic_event_extractor_spec": str((results_dir / "automatic_event_extractor_spec.md").relative_to(repo_root)),
            "automatic_event_pilot_csv": str((results_dir / "automatic_event_pilot.csv").relative_to(repo_root)),
            "automatic_event_pilot_report": str((results_dir / "automatic_event_pilot_report.md").relative_to(repo_root)),
            "cross_event_audit_csv": str((derived_dir / "cross_event_audit.csv").relative_to(repo_root)),
        },
    }
    (pilot_dir / "phase_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["verdict"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
