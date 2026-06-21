#!/usr/bin/env python3
"""Build outcome-blind capacity-matched baseline features for RQ003 Phase 4B.

The script intentionally reads only the hardened routing key table and the
trajectory logs referenced by that table. It does not read official outcome
tables, score/rank columns, or Phase 4A predictor outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(
    "/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation"
)
RUN_ID = "RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424"
WORKER_ID = "rq003_p4b2_causal_baseline_rerun"
RUN_ROOT = REPO_ROOT / "reports/studies/RQ003_nsfc_external_evidence" / RUN_ID
DERIVED_ROOT = (
    REPO_ROOT
    / "data/derived/onsite_competition/RQ003_nsfc_external_evidence"
    / RUN_ID
)
ROUTING_PATH = DERIVED_ROOT / "intermediate/routing_keys_only.csv"
RESULT_TABLE_DIR = RUN_ROOT / "01_results/tables"
PROCESS_DIR = RUN_ROOT / "02_process/09_baseline_models"
MANIFEST_DIR = DERIVED_ROOT / "manifests"

MIN_OBSERVATIONS = 4
ACTIVE_DISTANCE_M = 15.0
ACTIVE_CLOSING_DISTANCE_M = 30.0
NOMINAL_DT_SEC = 0.1

ROUTING_COLUMNS = [
    "area",
    "team_code",
    "scenario",
    "replay_log_path",
    "replay_case_id",
    "replay_task_id",
    "replay_case_name",
]

FORBIDDEN_FEATURE_TOKENS = (
    "observed_pet",
    "realized_passing_order",
    "post_hoc_phase",
    "full_window",
)


@dataclass
class FrameSnapshot:
    case_id: str
    task_id: str
    frame_id: int
    timestamp_ms: int
    rel_time_sec: float
    av: dict[str, Any]
    objects: list[dict[str, Any]]


@dataclass
class AccessRecord:
    path: str
    purpose: str
    columns: str = "N/A"
    outcome_access: str = "NONE"


class AccessLog:
    def __init__(self) -> None:
        self.records: list[AccessRecord] = []

    def add(self, path: Path, purpose: str, columns: Iterable[str] | None = None) -> None:
        col_text = "N/A" if columns is None else ",".join(columns)
        self.records.append(
            AccessRecord(
                path=str(path.relative_to(REPO_ROOT) if path.is_absolute() else path),
                purpose=purpose,
                columns=col_text,
            )
        )

    def write(self, path: Path) -> None:
        lines = [
            f"worker_id: {WORKER_ID}",
            f"run_id: {RUN_ID}",
            f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
            "outcome_access: NONE",
            "routing_ingestion: DERIVED_ROOT/intermediate/routing_keys_only.csv only; original mapping was not read by this builder.",
            "",
            "accessed_files:",
        ]
        for rec in self.records:
            lines.append(f"- path: {rec.path}")
            lines.append(f"  purpose: {rec.purpose}")
            lines.append(f"  columns: {rec.columns}")
            lines.append(f"  outcome_access: {rec.outcome_access}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ACCESS = AccessLog()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def finite(value: float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isfinite(value_f):
        return value_f
    return math.nan


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def int_or_default(value: Any, default: int = 0) -> int:
    value_f = finite(value)
    if math.isfinite(value_f):
        return int(value_f)
    return default


def mean_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(statistics.fmean(vals)) if vals else math.nan


def min_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(min(vals)) if vals else math.nan


def max_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(max(vals)) if vals else math.nan


def last_or_nan(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(vals[-1]) if vals else math.nan


def angle_deg_to_heading(angle_deg: float) -> np.ndarray:
    if not math.isfinite(angle_deg):
        return np.array([math.nan, math.nan], dtype=float)
    theta = math.radians(angle_deg)
    return np.array([math.sin(theta), math.cos(theta)], dtype=float)


def signed_angle_diff_deg(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return float((a - b + 180.0) % 360.0 - 180.0)


def dim_m(value: Any) -> float:
    v = finite(value)
    if not math.isfinite(v):
        return math.nan
    return v / 100.0 if abs(v) > 20.0 else v


def speed_mps(record: dict[str, Any]) -> float:
    return finite(record.get("speed"))


def accel_mps2(record: dict[str, Any]) -> float:
    for key in ("acceleration ", "acceleration", "lonAcc"):
        v = finite(record.get(key))
        if math.isfinite(v):
            return v
    return math.nan


def timestamp_ms(record: dict[str, Any], fallback: Any) -> int:
    for key in ("globalTimeStamp", "timestamp"):
        value = record.get(key)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    try:
        return int(float(fallback))
    except (TypeError, ValueError):
        return 0


def local_xy(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    if not all(math.isfinite(v) for v in (lat, lon, lat0, lon0)):
        return np.array([math.nan, math.nan], dtype=float)
    radius = 6_371_000.0
    x = math.radians(lon - lon0) * radius * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * radius
    return np.array([x, y], dtype=float)


def velocity_vector(record: dict[str, Any]) -> np.ndarray:
    speed = speed_mps(record)
    heading = angle_deg_to_heading(finite(record.get("courseAngle")))
    if not math.isfinite(speed) or not np.all(np.isfinite(heading)):
        return np.array([math.nan, math.nan], dtype=float)
    return speed * heading


def parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def extract_frame(raw: dict[str, Any], case_id: str, lat0: float | None, lon0: float | None) -> tuple[FrameSnapshot | None, float | None, float | None]:
    if str(raw.get("caseId", "")) != case_id:
        return None, lat0, lon0
    participants = raw.get("participantTrajectories") or []
    if not participants:
        return None, lat0, lon0
    participant = participants[0]
    values = participant.get("value") or []
    if not values:
        return None, lat0, lon0

    av = None
    objects: list[dict[str, Any]] = []
    for record in values:
        if int_or_default(record.get("isPerception"), 0) == 0 and av is None:
            av = dict(record)
        else:
            objects.append(dict(record))
    if av is None:
        return None, lat0, lon0

    if lat0 is None or lon0 is None:
        lat0 = finite(av.get("latitude"))
        lon0 = finite(av.get("longitude"))
    t_ms = timestamp_ms(av, participant.get("timestamp") or raw.get("timestamp"))
    frame_id = int_or_default(av.get("frameId"), len(values))
    snapshot = FrameSnapshot(
        case_id=case_id,
        task_id=str(raw.get("taskId", "")),
        frame_id=frame_id,
        timestamp_ms=t_ms,
        rel_time_sec=0.0,
        av=av,
        objects=objects,
    )
    return snapshot, lat0, lon0


def load_case_frames(log_path: Path, case_id: str) -> list[FrameSnapshot]:
    ACCESS.add(log_path, "trajectory_log_for_online_baseline_features")
    frames: list[FrameSnapshot] = []
    lat0: float | None = None
    lon0: float | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = parse_json_line(line)
            if raw is None:
                continue
            frame, lat0, lon0 = extract_frame(raw, case_id, lat0, lon0)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda f: (f.timestamp_ms, f.frame_id))
    if frames:
        t0 = frames[0].timestamp_ms
        for frame in frames:
            frame.rel_time_sec = max(0.0, (frame.timestamp_ms - t0) / 1000.0)
    return frames


def object_metrics(frame: FrameSnapshot, lat0: float, lon0: float) -> list[dict[str, float]]:
    av = frame.av
    av_pos = local_xy(finite(av.get("latitude")), finite(av.get("longitude")), lat0, lon0)
    av_heading = angle_deg_to_heading(finite(av.get("courseAngle")))
    av_vel = velocity_vector(av)
    av_len = dim_m(av.get("length"))
    av_wid = dim_m(av.get("width"))
    metrics: list[dict[str, float]] = []
    for obj in frame.objects:
        obj_pos = local_xy(finite(obj.get("latitude")), finite(obj.get("longitude")), lat0, lon0)
        rel_vec = obj_pos - av_pos
        if not np.all(np.isfinite(rel_vec)):
            continue
        distance = float(np.linalg.norm(rel_vec))
        if distance <= 0.0:
            unit = np.array([math.nan, math.nan], dtype=float)
        else:
            unit = rel_vec / distance
        obj_vel = velocity_vector(obj)
        rel_vel = obj_vel - av_vel
        longitudinal = float(np.dot(rel_vec, av_heading)) if np.all(np.isfinite(av_heading)) else math.nan
        lateral = (
            float(-av_heading[1] * rel_vec[0] + av_heading[0] * rel_vec[1])
            if np.all(np.isfinite(av_heading))
            else math.nan
        )
        obj_long_speed = float(np.dot(obj_vel, av_heading)) if np.all(np.isfinite(obj_vel)) and np.all(np.isfinite(av_heading)) else math.nan
        av_long_speed = float(np.dot(av_vel, av_heading)) if np.all(np.isfinite(av_vel)) and np.all(np.isfinite(av_heading)) else math.nan
        closing_long = av_long_speed - obj_long_speed if math.isfinite(av_long_speed) and math.isfinite(obj_long_speed) else math.nan
        obj_len = dim_m(obj.get("length"))
        obj_wid = dim_m(obj.get("width"))
        half_len = 0.5 * sum(v for v in (av_len, obj_len) if math.isfinite(v))
        half_wid = 0.5 * sum(v for v in (av_wid, obj_wid) if math.isfinite(v))
        longitudinal_gap = max(0.0, longitudinal - half_len) if math.isfinite(longitudinal) else math.nan
        lateral_gap = max(0.0, abs(lateral) - half_wid) if math.isfinite(lateral) else math.nan
        range_closing = float(-np.dot(rel_vec, rel_vel) / distance) if distance > 0.0 and np.all(np.isfinite(rel_vel)) else math.nan
        ttc_long = longitudinal_gap / closing_long if math.isfinite(longitudinal_gap) and closing_long > 1e-6 else math.inf
        ttc_range = distance / range_closing if math.isfinite(range_closing) and range_closing > 1e-6 else math.inf
        ttc = min(ttc_long, ttc_range)
        heading_delta = abs(signed_angle_diff_deg(finite(av.get("courseAngle")), finite(obj.get("courseAngle"))))
        metrics.append(
            {
                "distance_m": distance,
                "longitudinal_m": longitudinal,
                "lateral_m": lateral,
                "longitudinal_gap_m": longitudinal_gap,
                "lateral_gap_m": lateral_gap,
                "closing_speed_mps": max(0.0, closing_long if math.isfinite(closing_long) else range_closing),
                "range_closing_mps": max(0.0, range_closing) if math.isfinite(range_closing) else math.nan,
                "ttc_sec": ttc,
                "speed_mps": speed_mps(obj),
                "heading_delta_abs_deg": heading_delta,
                "vehicle_type": finite(obj.get("vehicleType")),
                "is_ahead": 1.0 if math.isfinite(longitudinal) and longitudinal > 0 else 0.0,
                "is_crossing": 1.0 if math.isfinite(heading_delta) and 45.0 <= heading_delta <= 135.0 else 0.0,
            }
        )
    return metrics


def first_lat_lon(frames: list[FrameSnapshot]) -> tuple[float, float]:
    for frame in frames:
        lat0 = finite(frame.av.get("latitude"))
        lon0 = finite(frame.av.get("longitude"))
        if math.isfinite(lat0) and math.isfinite(lon0):
            return lat0, lon0
    return math.nan, math.nan


def is_active_conflict(metrics: list[dict[str, float]]) -> bool:
    for m in metrics:
        distance = m["distance_m"]
        closing = m["closing_speed_mps"]
        if math.isfinite(distance) and distance <= ACTIVE_DISTANCE_M:
            return True
        if math.isfinite(distance) and math.isfinite(closing) and distance <= ACTIVE_CLOSING_DISTANCE_M and closing > 0.0:
            return True
    return False


def intent_active(frame: FrameSnapshot) -> bool:
    av = frame.av
    return any(
        [
            speed_mps(av) >= 0.5,
            abs(accel_mps2(av)) >= 0.2 if math.isfinite(accel_mps2(av)) else False,
            abs(finite(av.get("wheelAngle"))) >= 0.5 if math.isfinite(finite(av.get("wheelAngle"))) else False,
            finite(av.get("braking")) > 0.0 if math.isfinite(finite(av.get("braking"))) else False,
            finite(av.get("acceleratorPedal")) > 0.0 if math.isfinite(finite(av.get("acceleratorPedal"))) else False,
        ]
    )


def choose_decision_index(frames: list[FrameSnapshot], per_frame_metrics: list[list[dict[str, float]]]) -> tuple[int, str]:
    min_idx = min(max(MIN_OBSERVATIONS - 1, 0), len(frames) - 1)
    for idx in range(min_idx, len(frames)):
        if is_active_conflict(per_frame_metrics[idx]):
            return idx, "first_active_conflict_frame"
    for idx in range(min_idx, len(frames)):
        if intent_active(frames[idx]):
            return idx, "first_intent_frame"
    return min_idx, "minimum_observation_frame"


def nearest_metric(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    finite_metrics = [m for m in metrics if math.isfinite(m["distance_m"])]
    if not finite_metrics:
        return None
    return min(finite_metrics, key=lambda m: m["distance_m"])


def ahead_metric(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    ahead = [m for m in metrics if m["is_ahead"] == 1.0 and math.isfinite(m["longitudinal_gap_m"])]
    if not ahead:
        return None
    return min(ahead, key=lambda m: m["longitudinal_gap_m"])


def current_value(record: dict[str, Any], key: str) -> float:
    return finite(record.get(key))


def compute_cell_features(row: pd.Series, frames: list[FrameSnapshot]) -> dict[str, Any]:
    if not frames:
        raise ValueError("no frames for routed replay case")

    lat0, lon0 = first_lat_lon(frames)
    per_metrics = [object_metrics(frame, lat0, lon0) for frame in frames]
    decision_idx, decision_rule = choose_decision_index(frames, per_metrics)
    decision_frame = frames[decision_idx]
    prefix = frames[: decision_idx + 1]
    prefix_metrics = per_metrics[: decision_idx + 1]
    current_metrics = per_metrics[decision_idx]
    nearest = nearest_metric(current_metrics) or {}
    ahead = ahead_metric(current_metrics) or {}

    speeds = [speed_mps(frame.av) for frame in prefix]
    accels = [accel_mps2(frame.av) for frame in prefix]
    lon_accels = [current_value(frame.av, "lonAcc") for frame in prefix]
    lat_accels = [current_value(frame.av, "latAcc") for frame in prefix]
    headings = [finite(frame.av.get("courseAngle")) for frame in prefix]
    wheel_angles = [finite(frame.av.get("wheelAngle")) for frame in prefix]
    steering_angles = [finite(frame.av.get("steeringWheelAngle")) for frame in prefix]
    brake_flags = [1.0 if finite(frame.av.get("braking")) > 0.0 else 0.0 for frame in prefix]
    accelerator_vals = [finite(frame.av.get("acceleratorPedal")) for frame in prefix]
    auto_status = [finite(frame.av.get("autoStatus")) for frame in prefix]
    object_counts = [len(metrics) for metrics in prefix_metrics]
    crossing_counts = [sum(1 for metric in metrics if metric["is_crossing"] == 1.0) for metrics in prefix_metrics]
    nearest_distances = [nearest_metric(metrics)["distance_m"] for metrics in prefix_metrics if nearest_metric(metrics)]
    nearest_ttc = [nearest_metric(metrics)["ttc_sec"] for metrics in prefix_metrics if nearest_metric(metrics)]
    nearest_lat_gaps = [nearest_metric(metrics)["lateral_gap_m"] for metrics in prefix_metrics if nearest_metric(metrics)]
    ahead_gaps = [ahead_metric(metrics)["longitudinal_gap_m"] for metrics in prefix_metrics if ahead_metric(metrics)]
    closing_values = [m["closing_speed_mps"] for metrics in prefix_metrics for m in metrics]
    ttc_values = [m["ttc_sec"] for metrics in prefix_metrics for m in metrics]
    lateral_gap_values = [m["lateral_gap_m"] for metrics in prefix_metrics for m in metrics]

    av_start = prefix[0].av
    av_now = decision_frame.av
    start_xy = local_xy(finite(av_start.get("latitude")), finite(av_start.get("longitude")), lat0, lon0)
    now_xy = local_xy(finite(av_now.get("latitude")), finite(av_now.get("longitude")), lat0, lon0)
    travel_m = float(np.linalg.norm(now_xy - start_xy)) if np.all(np.isfinite(now_xy - start_xy)) else math.nan
    start_heading = finite(av_start.get("courseAngle"))
    now_heading = finite(av_now.get("courseAngle"))
    heading_change = abs(signed_angle_diff_deg(now_heading, start_heading))
    current_ttc = finite(nearest.get("ttc_sec"))
    if math.isinf(current_ttc):
        current_ttc = 999.0
    min_ttc = min_or_nan([v if math.isfinite(v) else 999.0 for v in ttc_values])
    if math.isinf(min_ttc):
        min_ttc = 999.0

    current_lateral_gap = finite(nearest.get("lateral_gap_m"))
    min_lateral_gap = min_or_nan(lateral_gap_values)
    current_distance = finite(nearest.get("distance_m"))
    current_closing = finite(nearest.get("closing_speed_mps"))
    current_headway = (
        finite(ahead.get("longitudinal_gap_m")) / max(speed_mps(av_now), 1e-6)
        if math.isfinite(finite(ahead.get("longitudinal_gap_m"))) and speed_mps(av_now) > 0.1
        else math.nan
    )
    gap_delta = (
        finite_values(ahead_gaps)[-1] - finite_values(ahead_gaps)[0]
        if len(finite_values(ahead_gaps)) >= 2
        else math.nan
    )
    prefix_duration = decision_frame.rel_time_sec - prefix[0].rel_time_sec
    decision_time = decision_frame.rel_time_sec
    active_prefix_frames = sum(1 for metrics in prefix_metrics if is_active_conflict(metrics))
    lateral_offsets = []
    start_heading_vec = angle_deg_to_heading(start_heading)
    for frame in prefix:
        xy = local_xy(finite(frame.av.get("latitude")), finite(frame.av.get("longitude")), lat0, lon0)
        if np.all(np.isfinite(xy - start_xy)) and np.all(np.isfinite(start_heading_vec)):
            rel_vec = xy - start_xy
            lateral_offsets.append(float(-start_heading_vec[1] * rel_vec[0] + start_heading_vec[0] * rel_vec[1]))
    max_abs_lateral_offset = max_or_nan([abs(v) for v in lateral_offsets])

    features: dict[str, Any] = {
        "area": row["area"],
        "team_code": row["team_code"],
        "scenario": row["scenario"],
        "cell_key": f"{row['area']}|{row['team_code']}|{row['scenario']}",
        "replay_case_id": str(row["replay_case_id"]),
        "replay_task_id": str(row.get("replay_task_id", "")),
        "decision_frame_id": decision_frame.frame_id,
        "decision_time_sec": decision_time,
        "decision_rule": decision_rule,
        "causal_prefix_frames": len(prefix),
        "causal_prefix_duration_sec": prefix_duration,
        "state_av_x_m": float(now_xy[0]) if np.all(np.isfinite(now_xy)) else math.nan,
        "state_av_y_m": float(now_xy[1]) if np.all(np.isfinite(now_xy)) else math.nan,
        "state_av_heading_sin": float(angle_deg_to_heading(now_heading)[0]) if math.isfinite(now_heading) else math.nan,
        "state_av_heading_cos": float(angle_deg_to_heading(now_heading)[1]) if math.isfinite(now_heading) else math.nan,
        "state_av_length_m": dim_m(av_now.get("length")),
        "state_av_width_m": dim_m(av_now.get("width")),
        "state_object_count_current": float(len(current_metrics)),
        "state_object_count_prefix_mean": mean_or_nan(object_counts),
        "state_crossing_object_count_current": float(sum(1 for m in current_metrics if m["is_crossing"] == 1.0)),
        "state_crossing_object_count_prefix_max": max_or_nan(crossing_counts),
        "state_nearest_object_distance_m": current_distance,
        "state_nearest_object_longitudinal_m": finite(nearest.get("longitudinal_m")),
        "state_nearest_object_lateral_m": finite(nearest.get("lateral_m")),
        "state_nearest_object_speed_mps": finite(nearest.get("speed_mps")),
        "state_nearest_object_heading_delta_abs_deg": finite(nearest.get("heading_delta_abs_deg")),
        "kin_av_speed_start_mps": speed_mps(av_start),
        "kin_av_speed_current_mps": speed_mps(av_now),
        "kin_av_speed_prefix_mean_mps": mean_or_nan(speeds),
        "kin_av_speed_prefix_max_mps": max_or_nan(speeds),
        "kin_av_speed_delta_mps": speed_mps(av_now) - speed_mps(av_start),
        "kin_av_accel_current_mps2": accel_mps2(av_now),
        "kin_av_accel_prefix_mean_mps2": mean_or_nan(accels),
        "kin_av_lon_accel_prefix_mean_mps2": mean_or_nan(lon_accels),
        "kin_av_lat_accel_abs_prefix_mean_mps2": mean_or_nan([abs(v) for v in lat_accels]),
        "kin_travel_distance_prefix_m": travel_m,
        "kin_heading_change_abs_prefix_deg": heading_change,
        "kin_wheel_angle_abs_current_deg": abs(finite(av_now.get("wheelAngle"))) if math.isfinite(finite(av_now.get("wheelAngle"))) else math.nan,
        "kin_wheel_angle_abs_prefix_max_deg": max_or_nan([abs(v) for v in wheel_angles]),
        "kin_steering_angle_abs_prefix_max_deg": max_or_nan([abs(v) for v in steering_angles]),
        "kin_brake_flag_prefix": max_or_nan(brake_flags),
        "kin_accelerator_prefix_mean": mean_or_nan(accelerator_vals),
        "kin_gap_ahead_current_m": finite(ahead.get("longitudinal_gap_m")),
        "kin_gap_ahead_delta_prefix_m": gap_delta,
        "kin_headway_current_sec": current_headway,
        "kin_closing_speed_current_mps": current_closing,
        "kin_closing_speed_prefix_max_mps": max_or_nan(closing_values),
        "kin_initiation_delay_sec": decision_time,
        "safety_ttc_current_sec": current_ttc,
        "safety_ttc_min_prefix_sec": min_ttc,
        "safety_lateral_gap_current_m": current_lateral_gap,
        "safety_lateral_gap_min_prefix_m": min_lateral_gap,
        "safety_inverse_distance_current": 1.0 / current_distance if math.isfinite(current_distance) and current_distance > 0.0 else math.nan,
        "safety_active_conflict_frame_count_prefix": float(active_prefix_frames),
        "safety_ttc_below_2s_prefix_flag": 1.0 if any(math.isfinite(v) and v < 2.0 for v in ttc_values) else 0.0,
        "safety_lateral_gap_below_1p5m_prefix_flag": 1.0 if any(math.isfinite(v) and v < 1.5 for v in lateral_gap_values) else 0.0,
        "safety_collision_risk_proxy_prefix": 1.0
        if any(
            (
                math.isfinite(m["distance_m"])
                and m["distance_m"] <= 5.0
                and math.isfinite(m["closing_speed_mps"])
                and m["closing_speed_mps"] > 0.0
            )
            or (math.isfinite(m["ttc_sec"]) and m["ttc_sec"] < 1.0)
            for metrics in prefix_metrics
            for m in metrics
        )
        else 0.0,
        "safety_takeover_proxy_prefix_flag": 1.0
        if any(math.isfinite(v) and v <= 0.0 for v in auto_status)
        else 0.0,
        "safety_line_crossing_proxy_prefix_flag": 1.0
        if math.isfinite(max_abs_lateral_offset) and max_abs_lateral_offset >= 1.75
        else 0.0,
    }
    return features


def build_feature_manifest(feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    definitions: dict[str, tuple[str, str, str]] = {
        "decision_frame_id": ("state", "Frame id at online decision frame.", "available at t_decision"),
        "decision_time_sec": ("state", "Elapsed seconds from first routed case frame to online decision frame.", "available at t_decision"),
        "causal_prefix_frames": ("state", "Number of frames in prefix ending at decision frame.", "available at t_decision"),
        "causal_prefix_duration_sec": ("state", "Duration of prefix ending at decision frame.", "available at t_decision"),
        "state_av_x_m": ("state", "AV local east-coordinate at decision frame relative to case origin.", "current frame"),
        "state_av_y_m": ("state", "AV local north-coordinate at decision frame relative to case origin.", "current frame"),
        "state_av_heading_sin": ("state", "Sine component of AV heading at decision frame.", "current frame"),
        "state_av_heading_cos": ("state", "Cosine component of AV heading at decision frame.", "current frame"),
        "state_av_length_m": ("state", "AV length converted to meters from trajectory field.", "current frame"),
        "state_av_width_m": ("state", "AV width converted to meters from trajectory field.", "current frame"),
        "state_object_count_current": ("state", "Number of perceived objects in the decision frame.", "current frame"),
        "state_object_count_prefix_mean": ("state", "Mean perceived object count over causal prefix.", "available at t_decision"),
        "state_crossing_object_count_current": ("state", "Objects with approximately crossing heading at decision frame.", "current frame"),
        "state_crossing_object_count_prefix_max": ("state", "Maximum crossing-object count over causal prefix.", "available at t_decision"),
        "state_nearest_object_distance_m": ("state", "Center distance to nearest perceived object at decision frame.", "current frame"),
        "state_nearest_object_longitudinal_m": ("state", "Nearest object position projected onto AV heading.", "current frame"),
        "state_nearest_object_lateral_m": ("state", "Nearest object lateral offset from AV heading axis.", "current frame"),
        "state_nearest_object_speed_mps": ("state", "Nearest object speed at decision frame.", "current frame"),
        "state_nearest_object_heading_delta_abs_deg": ("state", "Absolute heading difference to nearest object.", "current frame"),
        "kin_av_speed_start_mps": ("causal_kinematics", "AV speed at first routed case frame.", "first frame"),
        "kin_av_speed_current_mps": ("causal_kinematics", "AV speed at decision frame.", "current frame"),
        "kin_av_speed_prefix_mean_mps": ("causal_kinematics", "Mean AV speed over causal prefix.", "available at t_decision"),
        "kin_av_speed_prefix_max_mps": ("causal_kinematics", "Maximum AV speed over causal prefix.", "available at t_decision"),
        "kin_av_speed_delta_mps": ("causal_kinematics", "Decision-frame speed minus first-frame speed.", "available at t_decision"),
        "kin_av_accel_current_mps2": ("causal_kinematics", "AV acceleration at decision frame.", "current frame"),
        "kin_av_accel_prefix_mean_mps2": ("causal_kinematics", "Mean AV acceleration over causal prefix.", "available at t_decision"),
        "kin_av_lon_accel_prefix_mean_mps2": ("causal_kinematics", "Mean longitudinal acceleration over causal prefix.", "available at t_decision"),
        "kin_av_lat_accel_abs_prefix_mean_mps2": ("causal_kinematics", "Mean absolute lateral acceleration over causal prefix.", "available at t_decision"),
        "kin_travel_distance_prefix_m": ("causal_kinematics", "AV displacement over causal prefix.", "available at t_decision"),
        "kin_heading_change_abs_prefix_deg": ("causal_kinematics", "Absolute AV heading change over causal prefix.", "available at t_decision"),
        "kin_wheel_angle_abs_current_deg": ("causal_kinematics", "Absolute wheel angle at decision frame.", "current frame"),
        "kin_wheel_angle_abs_prefix_max_deg": ("causal_kinematics", "Maximum absolute wheel angle over causal prefix.", "available at t_decision"),
        "kin_steering_angle_abs_prefix_max_deg": ("causal_kinematics", "Maximum absolute steering wheel angle over causal prefix.", "available at t_decision"),
        "kin_brake_flag_prefix": ("causal_kinematics", "Whether brake was active in causal prefix.", "available at t_decision"),
        "kin_accelerator_prefix_mean": ("causal_kinematics", "Mean accelerator pedal value over causal prefix.", "available at t_decision"),
        "kin_gap_ahead_current_m": ("causal_kinematics", "Current longitudinal edge gap to nearest ahead object.", "current frame"),
        "kin_gap_ahead_delta_prefix_m": ("causal_kinematics", "Change in nearest-ahead longitudinal gap over causal prefix.", "available at t_decision"),
        "kin_headway_current_sec": ("causal_kinematics", "Current ahead gap divided by AV speed.", "current frame"),
        "kin_closing_speed_current_mps": ("causal_kinematics", "Current positive closing speed toward nearest object.", "current frame"),
        "kin_closing_speed_prefix_max_mps": ("causal_kinematics", "Maximum positive closing speed over causal prefix.", "available at t_decision"),
        "kin_initiation_delay_sec": ("causal_kinematics", "Elapsed time until selected online decision frame.", "available at t_decision"),
        "safety_ttc_current_sec": ("safety", "Current online TTC proxy to nearest object; 999 for no closing conflict.", "current frame"),
        "safety_ttc_min_prefix_sec": ("safety", "Minimum online TTC proxy over causal prefix only.", "available at t_decision"),
        "safety_lateral_gap_current_m": ("safety", "Current lateral edge gap to nearest object.", "current frame"),
        "safety_lateral_gap_min_prefix_m": ("safety", "Minimum lateral edge gap over causal prefix only.", "available at t_decision"),
        "safety_inverse_distance_current": ("safety", "Inverse nearest-object distance at decision frame.", "current frame"),
        "safety_active_conflict_frame_count_prefix": ("safety", "Count of prefix frames satisfying frozen active-conflict rule.", "available at t_decision"),
        "safety_ttc_below_2s_prefix_flag": ("safety", "Flag for online TTC proxy below 2s in causal prefix.", "available at t_decision"),
        "safety_lateral_gap_below_1p5m_prefix_flag": ("safety", "Flag for lateral gap below 1.5m in causal prefix.", "available at t_decision"),
        "safety_collision_risk_proxy_prefix": ("safety", "Prefix-only near/closing or TTC<1 collision-risk proxy.", "available at t_decision"),
        "safety_takeover_proxy_prefix_flag": ("safety", "Prefix-only proxy from AV auto-status field.", "available at t_decision"),
        "safety_line_crossing_proxy_prefix_flag": ("safety", "Prefix-only proxy from lateral deviation relative to initial heading.", "available at t_decision"),
    }
    for feature in feature_columns:
        if feature in {"area", "team_code", "scenario", "cell_key", "replay_case_id", "replay_task_id", "decision_rule"}:
            continue
        group, definition, first_time = definitions[feature]
        rows.append(
            {
                "feature_name": feature,
                "feature_group": group,
                "definition": definition,
                "source": "routed vehicle_perception_simulation_trajectory.log; causal prefix only",
                "first_available_time": first_time,
                "online_availability_proof": "computed from current frame or frames with timestamp <= t_decision",
                "uses_future_information": "NO",
                "uses_forbidden_full_window_or_posthoc": "NO",
            }
        )
    return pd.DataFrame(rows)


def load_routing() -> pd.DataFrame:
    ACCESS.add(ROUTING_PATH, "hardened_cell_to_replay_routing", ROUTING_COLUMNS)
    df = pd.read_csv(ROUTING_PATH, usecols=ROUTING_COLUMNS, dtype=str, keep_default_na=False)
    if len(df) != 150:
        raise ValueError(f"expected 150 routed cells, found {len(df)}")
    if df[["area", "team_code", "scenario"]].drop_duplicates().shape[0] != 150:
        raise ValueError("routing keys are not unique at area|team_code|scenario")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_artifact_manifest(artifacts: list[Path], path: Path) -> None:
    rows = []
    for artifact in artifacts:
        rows.append(
            {
                "artifact": artifact.name,
                "path": rel(artifact),
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
                "outcome_access": "NONE",
            }
        )
    write_csv(pd.DataFrame(rows), path)


def write_report(
    feature_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    output_paths: dict[str, Path],
    spec_deviation: str,
) -> None:
    groups = manifest_df["feature_group"].value_counts().to_dict()
    decision_counts = feature_df["decision_rule"].value_counts().to_dict()
    missing_cells = 150 - len(feature_df)
    report = f"""# Phase 4B Hardened Baseline Feature Build

Worker: `{WORKER_ID}`

Run: `{RUN_ID}`

Status: PASS

## Outcome-Blind Controls

- The builder read `DERIVED_ROOT/intermediate/routing_keys_only.csv` and routed trajectory logs only.
- `routing_keys_only.csv` was created from the original mapping by explicit key-column allowlist before this build.
- No official outcome table, score/rank column, Phase 4A predictor table, or model target was read by this builder.
- No model was fit and no predictor-outcome association was computed.

## Feature Set

- Cells produced: {len(feature_df)} / 150.
- Missing routed cells: {missing_cells}.
- Feature groups: {json.dumps(groups, sort_keys=True)}.
- Decision-frame rules: {json.dumps(decision_counts, sort_keys=True)}.
- Online proof: every manifest row is current-frame or causal-prefix-only with timestamp `<= t_decision`.

## Capacity Contract

Baseline contains only frozen groups: `state`, `causal_kinematics`, and `safety`.
The corresponding full model remains baseline plus the two frozen Phase 4A IPV predictors.
Identifiers (`area`, `team_code`, `scenario`, `cell_key`) are retained as keys/metadata, not model predictors.

## Routing

- Routing rows: {len(routing_df)}.
- Distinct cells: {routing_df[['area', 'team_code', 'scenario']].drop_duplicates().shape[0]}.
- Materialized routing columns: {', '.join(routing_df.columns)}.
- Routing manifest: `{rel(DERIVED_ROOT / 'manifests/routing_keys_only_manifest.json')}`.

## Spec Deviations

{spec_deviation}

## Outputs

- `{rel(output_paths['features'])}`
- `{rel(output_paths['manifest'])}`
- `{rel(output_paths['routing'])}`
- `{rel(output_paths['file_access'])}`
- `{rel(output_paths['artifact_manifest'])}`
- `{rel(output_paths['worker_report'])}`
"""
    output_paths["report"].write_text(report, encoding="utf-8")


def main() -> None:
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    routing_df = load_routing()
    cell_rows: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, str], list[FrameSnapshot]] = {}
    errors: list[dict[str, str]] = []

    for _, row in routing_df.iterrows():
        log_path = REPO_ROOT / row["replay_log_path"]
        case_id = str(row["replay_case_id"])
        cache_key = (str(log_path), case_id)
        try:
            if cache_key not in frame_cache:
                frame_cache[cache_key] = load_case_frames(log_path, case_id)
            features = compute_cell_features(row, frame_cache[cache_key])
            cell_rows.append(features)
        except Exception as exc:  # noqa: BLE001 - report all per-cell blockers.
            errors.append(
                {
                    "area": row["area"],
                    "team_code": row["team_code"],
                    "scenario": row["scenario"],
                    "replay_case_id": case_id,
                    "error": str(exc),
                }
            )

    feature_df = pd.DataFrame(cell_rows)
    if errors:
        error_path = PROCESS_DIR / "baseline_feature_errors.csv"
        write_csv(pd.DataFrame(errors), error_path)
        raise RuntimeError(f"feature build failed for {len(errors)} cells; see {error_path}")
    if len(feature_df) != 150:
        raise RuntimeError(f"feature build produced {len(feature_df)} rows instead of 150")

    feature_df = feature_df.sort_values(["area", "team_code", "scenario"]).reset_index(drop=True)
    feature_columns = list(feature_df.columns)
    forbidden_hits = [
        col
        for col in feature_columns
        if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_hits:
        raise RuntimeError(f"forbidden feature names generated: {forbidden_hits}")

    manifest_df = build_feature_manifest(feature_columns)
    if manifest_df.empty:
        raise RuntimeError("empty feature manifest")
    if set(manifest_df["feature_group"]) != {"state", "causal_kinematics", "safety"}:
        raise RuntimeError("feature manifest groups do not match frozen baseline groups")
    if not manifest_df["uses_future_information"].eq("NO").all():
        raise RuntimeError("future-information check failed")

    output_paths = {
        "features": RESULT_TABLE_DIR / "cell_level_baseline_features.csv",
        "manifest": RESULT_TABLE_DIR / "baseline_feature_manifest.csv",
        "routing": ROUTING_PATH,
        "report": PROCESS_DIR / "baseline_report.md",
        "file_access": PROCESS_DIR / "file_access_manifest.txt",
        "artifact_manifest": PROCESS_DIR / "artifact_manifest.csv",
        "worker_report": PROCESS_DIR / "worker_report.json",
    }

    write_csv(feature_df, output_paths["features"])
    write_csv(manifest_df, output_paths["manifest"])
    ACCESS.write(output_paths["file_access"])

    spec_deviation = (
        "- Routed trajectory logs are physically under `data/onsite_competition/top5_research_subset/teams/...` "
        "because the hardened Gate0 routing keys point there; only those trajectory log paths were opened. "
        "Outcome tables in that subset were not read.\n"
        "- True lane-boundary line crossing and official takeover labels were not present in the routed trajectory logs; "
        "prefix-only proxy flags were emitted and explicitly named as proxies."
    )
    write_report(feature_df, manifest_df, routing_df, output_paths, spec_deviation)

    worker_report = {
        "status": "PASS",
        "worker_id": WORKER_ID,
        "role": "causal baseline worker (rerun, hardened)",
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(feature_df)),
        "distinct_cells": int(feature_df[["area", "team_code", "scenario"]].drop_duplicates().shape[0]),
        "feature_count_total": int(len(manifest_df)),
        "feature_count_by_group": {
            str(k): int(v) for k, v in manifest_df["feature_group"].value_counts().sort_index().items()
        },
        "decision_rule_counts": {
            str(k): int(v) for k, v in feature_df["decision_rule"].value_counts().sort_index().items()
        },
        "routing_columns": list(routing_df.columns),
        "routing_rows": int(len(routing_df)),
        "outcome_access": "NONE",
        "capacity_conformance": "baseline=state+causal_kinematics+safety; full adds only D_comp_auc and D_yield_auc",
        "spec_deviation": spec_deviation,
        "outputs": {key: rel(path) for key, path in output_paths.items()},
    }
    output_paths["worker_report"].write_text(
        json.dumps(worker_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    artifacts = [
        output_paths["features"],
        output_paths["manifest"],
        output_paths["routing"],
        MANIFEST_DIR / "routing_keys_only_manifest.json",
        output_paths["report"],
        output_paths["file_access"],
        output_paths["worker_report"],
    ]
    write_artifact_manifest(artifacts, output_paths["artifact_manifest"])

    print(json.dumps(worker_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
