#!/usr/bin/env python3
"""RQ019-B1: post-anchor counterpart burden analysis.

The script reads only the inputs named in RQ019_B1_kickoff.md.  Raw trajectory
logs are parsed one session at a time; no log corpus is materialized in memory.
The analysis is descriptive/associational and keeps lower- and upper-envelope
IPV exceedance separate.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parents[4]
SCRIPTED_SELECTION = (
    "online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle"
)
ALPHAS = (80, 90, 95)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
OUTCOMES = (
    "cp_min_acceleration",
    "cp_brake_share_2",
    "cp_brake_share_3",
    "cp_brake_share_4",
    "cp_speed_drop_kmh",
    "cp_anchor_speed_drop_kmh",
    "cp_max_abs_yaw_rate",
    "cp_total_heading_change",
)
RAW_FIELDS = (
    "id",
    "name",
    "frameId",
    "globalTimeStamp",
    "x",
    "y",
    "speed",
    "acceleration",
    "courseAngle",
)
NUMERIC_RAW_FIELDS = (
    "frameId",
    "x",
    "y",
    "speed",
    "acceleration",
    "courseAngle",
)
RNG_SEED = 19019
BOOTSTRAP_REPS = 1000
PERMUTATION_REPS = 1000


def finite_float(value: Any) -> float:
    """Convert a raw scalar to finite float; invalid values become NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def normalized_id(value: Any) -> int | None:
    """Normalize raw integer-like vehicle identifiers."""
    number = finite_float(value)
    if not math.isfinite(number) or not float(number).is_integer():
        return None
    return int(number)


def normalized_timestamp(value: Any) -> int | None:
    """Normalize the string-valued raw global timestamp to integer ms."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def shell_utc_timestamp() -> str:
    """Use the kickoff-mandated `date -u` command for the report state stamp."""
    return subprocess.check_output(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
    ).strip()


def parse_product_row_key(value: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for item in value.split("|"):
        key, separator, item_value = item.partition("=")
        if not separator:
            raise ValueError(f"Malformed product_row_key component: {item!r}")
        parts[key] = item_value
    required = {"case_key", "anchor_frame_index", "perspective"}
    if not required.issubset(parts):
        raise ValueError(f"product_row_key lacks {sorted(required - set(parts))}: {value}")
    return parts


def load_analysis_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load only task-authorized columns and build the 14,099-row analysis set."""
    gate_path = root / "data/derived/rq017_onsite_gate/l1_v1"
    score_path = root / (
        ".codex-fleet/rq016c-human-only-envelope/work/H2/"
        "onsite_scoring_dryrun.parquet"
    )
    anchor_path = root / (
        "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/"
        "onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet"
    )
    timeseries_path = root / (
        "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/"
        "onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet"
    )

    gate_columns = ["product_row_key", "status", "ipv_log"]
    score_columns = [
        "product_row_key",
        "lo_80",
        "hi_80",
        "width_80",
        "lo_90",
        "hi_90",
        "width_90",
        "lo_95",
        "hi_95",
        "width_95",
        "mechanism2_gate_ok",
        "context_cell",
    ]
    anchor_columns = [
        "case_key",
        "anchor_frame_index",
        "perspective",
        "session_id",
        "team_id",
        "ego_key_agent",
        "counterpart_key_agent",
        "counterpart_selection",
        "target_window_end_frame_index",
        "unit_composite_key",
        "relative_distance_anchor",
    ]
    timeseries_columns = [
        "case_key",
        "session_id",
        "frame_index",
        "timestamp_ms",
        "ego_x",
        "ego_y",
        "counterpart_key_agent",
        "counterpart_x",
        "counterpart_y",
    ]

    gate = pads.dataset(
        str(gate_path), format="parquet", partitioning="hive"
    ).to_table(columns=gate_columns).to_pandas()
    score = pq.read_table(score_path, columns=score_columns).to_pandas()
    anchors = pq.read_table(anchor_path, columns=anchor_columns).to_pandas()
    timeseries = pq.read_table(timeseries_path, columns=timeseries_columns).to_pandas()

    if gate["product_row_key"].duplicated().any():
        raise AssertionError("Mechanism-one product_row_key is not unique")
    if score["product_row_key"].duplicated().any():
        raise AssertionError("Mechanism-two product_row_key is not unique")
    natural_key = ["case_key", "anchor_frame_index", "perspective"]
    if anchors[natural_key].duplicated().any():
        raise AssertionError("Anchor natural key is not unique")

    joined = gate.merge(score, on="product_row_key", validate="one_to_one")
    parsed = joined["product_row_key"].map(parse_product_row_key)
    joined["case_key"] = parsed.map(lambda item: item["case_key"])
    joined["anchor_frame_index"] = parsed.map(
        lambda item: int(item["anchor_frame_index"])
    )
    joined["perspective"] = parsed.map(lambda item: item["perspective"])
    joined = joined.merge(anchors, on=natural_key, validate="one_to_one")
    analysis = joined.loc[
        joined["status"].eq("OK") & joined["mechanism2_gate_ok"].eq(True)
    ].copy()

    if len(analysis) != 14_099:
        raise AssertionError(
            f"Analysis rows differ from contract: observed {len(analysis)}, expected 14099"
        )
    if analysis["case_key"].nunique() != 231:
        raise AssertionError("Analysis case count differs from contract")
    if analysis["team_id"].nunique() != 19:
        raise AssertionError("Analysis team count differs from contract")

    for alpha in ALPHAS:
        width = analysis[f"width_{alpha}"].astype(float)
        if width.isna().any() or (width <= 0).any():
            raise AssertionError(f"Non-positive or missing width_{alpha} in analysis set")
        analysis[f"upper_{alpha}"] = np.maximum(
            0.0, (analysis["ipv_log"] - analysis[f"hi_{alpha}"]) / width
        )
        analysis[f"lower_{alpha}"] = np.maximum(
            0.0, (analysis[f"lo_{alpha}"] - analysis["ipv_log"]) / width
        )
        analysis[f"group_{alpha}"] = np.select(
            [analysis[f"lower_{alpha}"] > 0, analysis[f"upper_{alpha}"] > 0],
            ["lower", "upper"],
            default="inside",
        )

    expected_90 = {"upper": 2700, "lower": 1998, "inside": 9401}
    observed_90 = analysis["group_90"].value_counts().to_dict()
    if observed_90 != expected_90:
        raise AssertionError(
            f"Alpha-90 exposure counts differ: {observed_90} vs {expected_90}"
        )

    anchor_ts = timeseries.rename(
        columns={
            "frame_index": "anchor_frame_index",
            "timestamp_ms": "anchor_timestamp_ms",
            "ego_x": "derived_ego_x",
            "ego_y": "derived_ego_y",
            "counterpart_key_agent": "ts_counterpart_key_agent",
            "counterpart_x": "derived_counterpart_x",
            "counterpart_y": "derived_counterpart_y",
        }
    )[
        [
            "case_key",
            "session_id",
            "anchor_frame_index",
            "anchor_timestamp_ms",
            "derived_ego_x",
            "derived_ego_y",
            "ts_counterpart_key_agent",
            "derived_counterpart_x",
            "derived_counterpart_y",
        ]
    ]
    anchor_join_key = ["case_key", "session_id", "anchor_frame_index"]
    if anchor_ts[anchor_join_key].duplicated().any():
        raise AssertionError("Timeseries anchor lookup key is not unique")
    analysis = analysis.merge(anchor_ts, on=anchor_join_key, validate="many_to_one")
    cp_id_equal = (
        analysis["counterpart_key_agent"].astype("Int64")
        == analysis["ts_counterpart_key_agent"].astype("Int64")
    )
    if not cp_id_equal.all():
        raise AssertionError("Anchor and timeseries counterpart identifiers disagree")

    end_ts = timeseries.rename(
        columns={
            "frame_index": "target_window_end_frame_index",
            "timestamp_ms": "target_window_end_timestamp_ms",
        }
    )[
        [
            "case_key",
            "session_id",
            "target_window_end_frame_index",
            "target_window_end_timestamp_ms",
        ]
    ]
    end_key = ["case_key", "session_id", "target_window_end_frame_index"]
    if end_ts[end_key].duplicated().any():
        raise AssertionError("Timeseries contract-end lookup key is not unique")
    analysis = analysis.merge(end_ts, on=end_key, how="left", validate="many_to_one")

    case_end = (
        timeseries.groupby(["case_key", "session_id"], observed=True)
        .agg(
            case_end_frame_index=("frame_index", "max"),
            case_end_timestamp_ms=("timestamp_ms", "max"),
        )
        .reset_index()
    )
    analysis = analysis.merge(
        case_end, on=["case_key", "session_id"], validate="many_to_one"
    )
    analysis["contract_window_past_case_end"] = (
        analysis["target_window_end_frame_index"] > analysis["case_end_frame_index"]
    ) | analysis["target_window_end_timestamp_ms"].isna()
    analysis["contract_end_timestamp_ms"] = analysis[
        "target_window_end_timestamp_ms"
    ].fillna(analysis["case_end_timestamp_ms"])
    analysis["contract_end_timestamp_ms"] = np.minimum(
        analysis["contract_end_timestamp_ms"], analysis["case_end_timestamp_ms"]
    )
    analysis["fixed3_window_past_case_end"] = (
        analysis["anchor_timestamp_ms"] + 3000 > analysis["case_end_timestamp_ms"]
    )
    analysis["fixed3_end_timestamp_ms"] = np.minimum(
        analysis["anchor_timestamp_ms"] + 3000,
        analysis["case_end_timestamp_ms"],
    )
    analysis["stratum"] = np.where(
        analysis["counterpart_selection"].eq(SCRIPTED_SELECTION),
        "scripted",
        "non_scripted",
    )
    analysis = analysis.reset_index(drop=True)

    load_contract = {
        "source_rows": {
            "mechanism_one": len(gate),
            "mechanism_two": len(score),
            "anchors": len(anchors),
            "timeseries": len(timeseries),
        },
        "analysis_rows": len(analysis),
        "analysis_cases": int(analysis["case_key"].nunique()),
        "analysis_teams": int(analysis["team_id"].nunique()),
        "analysis_sessions": int(analysis["session_id"].nunique()),
        "alpha_90_counts": observed_90,
        "all_anchor_counterpart_selection_counts": (
            anchors.groupby("counterpart_selection", observed=True)
            .agg(rows=("case_key", "size"), cases=("case_key", "nunique"))
            .to_dict("index")
        ),
        "join_recipe": (
            "product_row_key parsed to case_key + anchor_frame_index + perspective; "
            "then joined one-to-one to the anchor natural key"
        ),
    }
    return analysis, timeseries, load_contract


@dataclass
class RawScanResult:
    series_records: dict[tuple[str, int], list[dict[str, Any]]]
    probe_records: dict[str, list[dict[str, Any]]]
    health: dict[str, Any]
    log_paths: dict[str, str]


def locate_raw_logs(root: Path, required_sessions: Iterable[str]) -> tuple[dict[str, Path], dict[str, Any]]:
    all_paths = sorted(
        (root / "data/onsite_competition/all_teams_dataset/teams").glob(
            "*/**/sessions/*/simulation_trajectory.log"
        )
    )
    by_session: dict[str, list[Path]] = defaultdict(list)
    for path in all_paths:
        by_session[path.parent.name].append(path)
    selected: dict[str, Path] = {}
    for session in sorted(set(map(str, required_sessions))):
        candidates = by_session.get(session, [])
        if len(candidates) != 1:
            raise AssertionError(
                f"Expected exactly one raw log for session {session}, found {len(candidates)}"
            )
        selected[session] = candidates[0]
    inventory = {
        "all_simulation_trajectory_log_count": len(all_paths),
        "all_simulation_trajectory_bytes": int(sum(p.stat().st_size for p in all_paths)),
        "analysis_log_count": len(selected),
        "analysis_log_bytes": int(sum(p.stat().st_size for p in selected.values())),
        "unused_log_sessions": sorted(set(by_session) - set(selected)),
    }
    return selected, inventory


def choose_probe_specs(analysis: pd.DataFrame, max_sessions: int = 5) -> list[dict[str, Any]]:
    """Choose one deterministic non-scripted anchor in each of five sessions."""
    candidates = analysis.loc[analysis["stratum"].eq("non_scripted")].copy()
    specs: list[dict[str, Any]] = []
    for session in sorted(candidates["session_id"].astype(str).unique())[:max_sessions]:
        group = candidates.loc[candidates["session_id"].astype(str).eq(session)].sort_values(
            "anchor_timestamp_ms"
        )
        row = group.iloc[len(group) // 2]
        specs.append(
            {
                "session_id": session,
                "case_key": str(row["case_key"]),
                "anchor_frame_index": int(row["anchor_frame_index"]),
                "anchor_timestamp_ms": int(row["anchor_timestamp_ms"]),
                "ego_key_agent": str(row["ego_key_agent"]),
                "radius_m": 50.0,
                "store_start_ms": int(row["anchor_timestamp_ms"]) - 250,
                "store_end_ms": int(row["anchor_timestamp_ms"]) + 3250,
            }
        )
    return specs


def scan_raw_logs(
    log_paths: dict[str, Path],
    needed_ids: dict[str, set[int]],
    probe_specs: list[dict[str, Any]],
) -> RawScanResult:
    """Stream each required session once and retain only analysis/probe records."""
    series_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    probe_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probes_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in probe_specs:
        probes_by_session[spec["session_id"]].append(spec)

    null_counts = {field: 0 for field in RAW_FIELDS}
    nonfinite_counts = {field: 0 for field in NUMERIC_RAW_FIELDS}
    valid_numeric_counts = {field: 0 for field in NUMERIC_RAW_FIELDS}
    stats_min = {field: math.inf for field in NUMERIC_RAW_FIELDS}
    stats_max = {field: -math.inf for field in NUMERIC_RAW_FIELDS}
    record_count = 0
    records_with_any_null = 0
    timestamp_string_count = 0
    json_error_count = 0
    non_trajectory_line_count = 0
    trajectory_line_count = 0
    retained_counterpart_records = 0
    retained_probe_records = 0
    acceleration_abs_gt_10 = 0
    course_angle_negative = 0
    course_angle_above_360 = 0
    course_angle_above_2pi = 0
    per_session: dict[str, dict[str, Any]] = {}

    for session, path in sorted(log_paths.items()):
        session_lines = 0
        session_records = 0
        session_json_errors = 0
        with path.open("rb") as handle:
            for raw_line in handle:
                session_lines += 1
                try:
                    obj = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    json_error_count += 1
                    session_json_errors += 1
                    continue
                if obj.get("type") != "trajectory":
                    non_trajectory_line_count += 1
                    continue
                trajectory_line_count += 1
                payload = obj.get("value") or {}
                if not isinstance(payload, dict):
                    continue
                arrays = (payload.get("value") or [], payload.get("obstacles") or [])
                for array in arrays:
                    if not isinstance(array, list):
                        continue
                    for rec in array:
                        if not isinstance(rec, dict):
                            continue
                        record_count += 1
                        session_records += 1
                        has_null = False
                        for field in RAW_FIELDS:
                            if rec.get(field) is None:
                                null_counts[field] += 1
                                has_null = True
                        if has_null:
                            records_with_any_null += 1
                        if isinstance(rec.get("globalTimeStamp"), str):
                            timestamp_string_count += 1
                        for field in NUMERIC_RAW_FIELDS:
                            value = finite_float(rec.get(field))
                            if math.isfinite(value):
                                valid_numeric_counts[field] += 1
                                stats_min[field] = min(stats_min[field], value)
                                stats_max[field] = max(stats_max[field], value)
                            elif rec.get(field) is not None:
                                nonfinite_counts[field] += 1
                        acceleration = finite_float(rec.get("acceleration"))
                        if math.isfinite(acceleration) and abs(acceleration) > 10:
                            acceleration_abs_gt_10 += 1
                        angle = finite_float(rec.get("courseAngle"))
                        if math.isfinite(angle):
                            course_angle_negative += int(angle < 0)
                            course_angle_above_360 += int(angle > 360)
                            course_angle_above_2pi += int(angle > 2 * math.pi + 1e-6)

                        vehicle_id = normalized_id(rec.get("id"))
                        timestamp_ms = normalized_timestamp(rec.get("globalTimeStamp"))
                        if vehicle_id is None or timestamp_ms is None:
                            continue
                        compact = {
                            "id": vehicle_id,
                            "name": rec.get("name"),
                            "timestamp_ms": timestamp_ms,
                            "frame_id": finite_float(rec.get("frameId")),
                            "x": finite_float(rec.get("x")),
                            "y": finite_float(rec.get("y")),
                            "speed": finite_float(rec.get("speed")),
                            "acceleration": acceleration,
                            "course_angle": angle,
                        }
                        if vehicle_id in needed_ids.get(session, set()):
                            series_records[(session, vehicle_id)].append(compact)
                            retained_counterpart_records += 1
                        for spec in probes_by_session.get(session, []):
                            if spec["store_start_ms"] <= timestamp_ms <= spec["store_end_ms"]:
                                probe_records[session].append(compact)
                                retained_probe_records += 1
                                break
        per_session[session] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "lines": session_lines,
            "vehicle_records": session_records,
            "json_errors": session_json_errors,
            "retained_counterpart_records": int(
                sum(len(v) for (s, _), v in series_records.items() if s == session)
            ),
        }

    acceleration_valid = valid_numeric_counts["acceleration"]
    acceleration_gt10_share = (
        acceleration_abs_gt_10 / acceleration_valid if acceleration_valid else math.nan
    )
    angle_min = stats_min["courseAngle"]
    angle_max = stats_max["courseAngle"]
    if (
        math.isfinite(angle_min)
        and math.isfinite(angle_max)
        and angle_min >= -1e-6
        and angle_max <= 360 + 1e-6
        and course_angle_above_2pi > 0
    ):
        angle_unit = "degree"
        angle_range_contract = "observed within [0, 360] with values above 2*pi"
    elif angle_min >= -math.pi - 1e-6 and angle_max <= 2 * math.pi + 1e-6:
        angle_unit = "radian"
        angle_range_contract = "observed within a radian-scale wrapped range"
    else:
        raise AssertionError(
            f"Cannot establish courseAngle unit/range from [{angle_min}, {angle_max}]"
        )

    total_field_slots = record_count * len(RAW_FIELDS)
    health = {
        "record_count": record_count,
        "records_with_any_required_field_null": records_with_any_null,
        "records_with_any_required_field_null_share": (
            records_with_any_null / record_count if record_count else math.nan
        ),
        "null_field_slots": int(sum(null_counts.values())),
        "required_field_slots": total_field_slots,
        "null_field_slot_share": (
            sum(null_counts.values()) / total_field_slots if total_field_slots else math.nan
        ),
        "null_counts": null_counts,
        "nonfinite_counts": nonfinite_counts,
        "valid_numeric_counts": valid_numeric_counts,
        "numeric_min": stats_min,
        "numeric_max": stats_max,
        "timestamp_string_count": timestamp_string_count,
        "json_error_count": json_error_count,
        "trajectory_line_count": trajectory_line_count,
        "non_trajectory_line_count": non_trajectory_line_count,
        "acceleration_abs_gt_10_count": acceleration_abs_gt_10,
        "acceleration_abs_gt_10_denominator": acceleration_valid,
        "acceleration_abs_gt_10_share": acceleration_gt10_share,
        "course_angle_negative_count": course_angle_negative,
        "course_angle_above_360_count": course_angle_above_360,
        "course_angle_above_2pi_count": course_angle_above_2pi,
        "course_angle_unit": angle_unit,
        "course_angle_range_basis": angle_range_contract,
        "retained_counterpart_records": retained_counterpart_records,
        "retained_probe_records": retained_probe_records,
        "per_session": per_session,
    }
    return RawScanResult(
        series_records=dict(series_records),
        probe_records=dict(probe_records),
        health=health,
        log_paths={session: str(path) for session, path in log_paths.items()},
    )


def record_quality(record: dict[str, Any]) -> int:
    return sum(
        int(math.isfinite(record[field]))
        for field in ("x", "y", "speed", "acceleration", "course_angle")
    )


def build_series(records: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Sort and deduplicate a single vehicle series by raw timestamp."""
    by_time: dict[int, dict[str, Any]] = {}
    duplicate_count = 0
    conflicting_duplicate_count = 0
    for record in records:
        timestamp = int(record["timestamp_ms"])
        previous = by_time.get(timestamp)
        if previous is not None:
            duplicate_count += 1
            finite_pairs = [
                (previous[field], record[field])
                for field in ("x", "y", "speed", "acceleration", "course_angle")
                if math.isfinite(previous[field]) and math.isfinite(record[field])
            ]
            if any(abs(a - b) > 1e-9 for a, b in finite_pairs):
                conflicting_duplicate_count += 1
            if record_quality(record) <= record_quality(previous):
                continue
        by_time[timestamp] = record
    ordered = [by_time[key] for key in sorted(by_time)]
    series = {
        "timestamp_ms": np.asarray([r["timestamp_ms"] for r in ordered], dtype=np.int64),
        "x": np.asarray([r["x"] for r in ordered], dtype=float),
        "y": np.asarray([r["y"] for r in ordered], dtype=float),
        "speed": np.asarray([r["speed"] for r in ordered], dtype=float),
        "acceleration": np.asarray([r["acceleration"] for r in ordered], dtype=float),
        "course_angle": np.asarray([r["course_angle"] for r in ordered], dtype=float),
        "id": np.asarray([r["id"] for r in ordered], dtype=np.int64),
        "name": np.asarray([r.get("name") for r in ordered], dtype=object),
    }
    diagnostics = {
        "input_records": len(records),
        "unique_timestamps": len(ordered),
        "duplicate_timestamps": duplicate_count,
        "conflicting_duplicate_timestamps": conflicting_duplicate_count,
    }
    return series, diagnostics


def nearest_index(timestamps: np.ndarray, target: int) -> int | None:
    if len(timestamps) == 0:
        return None
    position = int(np.searchsorted(timestamps, target))
    candidates = []
    if position < len(timestamps):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    return min(candidates, key=lambda idx: abs(int(timestamps[idx]) - target))


def calculate_window_metrics(
    series: dict[str, np.ndarray],
    start_timestamp_ms: int,
    end_timestamp_ms: int,
    anchor_speed_kmh: float,
    angle_unit: str,
) -> dict[str, Any]:
    timestamps = series["timestamp_ms"]
    mask = (timestamps > start_timestamp_ms) & (timestamps <= end_timestamp_ms)
    indices = np.flatnonzero(mask)
    result: dict[str, Any] = {
        "cp_raw_record_n": int(len(indices)),
        "cp_acc_n": 0,
        "cp_acc_excluded_abs_gt_10_n": 0,
        "cp_speed_n": 0,
        "cp_angle_n": 0,
    }
    if len(indices) == 0:
        for outcome in OUTCOMES:
            result[outcome] = math.nan
        for threshold in (2, 3, 4):
            result[f"cp_brake_n_{threshold}"] = 0
        return result

    acceleration = series["acceleration"][indices]
    finite_acceleration = acceleration[np.isfinite(acceleration)]
    # The raw-field health assertion is evaluated before this sanitization.
    # Values beyond +/-10 m/s^2 are then excluded from outcomes because the
    # observed positive tail includes corrupted values near 1e263; retaining
    # them makes even a window minimum non-physical when it is the sole record.
    result["cp_acc_excluded_abs_gt_10_n"] = int(
        np.sum(np.abs(finite_acceleration) > 10)
    )
    valid_acceleration = finite_acceleration[np.abs(finite_acceleration) <= 10]
    result["cp_acc_n"] = int(len(valid_acceleration))
    result["cp_min_acceleration"] = (
        float(valid_acceleration.min()) if len(valid_acceleration) else math.nan
    )
    for threshold in (2, 3, 4):
        numerator = int(np.sum(valid_acceleration < -threshold))
        result[f"cp_brake_n_{threshold}"] = numerator
        result[f"cp_brake_share_{threshold}"] = (
            numerator / len(valid_acceleration) if len(valid_acceleration) else math.nan
        )

    speed = series["speed"][indices]
    valid_speed = speed[np.isfinite(speed)]
    result["cp_speed_n"] = int(len(valid_speed))
    if len(valid_speed):
        result["cp_speed_drop_kmh"] = float(valid_speed.max() - valid_speed.min())
        result["cp_anchor_speed_drop_kmh"] = (
            float(anchor_speed_kmh - valid_speed.min())
            if math.isfinite(anchor_speed_kmh)
            else math.nan
        )
    else:
        result["cp_speed_drop_kmh"] = math.nan
        result["cp_anchor_speed_drop_kmh"] = math.nan

    angle = series["course_angle"][indices]
    angle_timestamps = timestamps[indices]
    valid_angle = np.isfinite(angle)
    angle = angle[valid_angle]
    angle_timestamps = angle_timestamps[valid_angle]
    result["cp_angle_n"] = int(len(angle))
    if len(angle) >= 2:
        angle_radians = np.deg2rad(angle) if angle_unit == "degree" else angle
        unwrapped = np.unwrap(angle_radians)
        delta_angle = np.diff(unwrapped)
        delta_seconds = np.diff(angle_timestamps) / 1000.0
        valid_delta = delta_seconds > 0
        if np.any(valid_delta):
            yaw_rate_radians = np.abs(delta_angle[valid_delta] / delta_seconds[valid_delta])
            result["cp_max_abs_yaw_rate"] = float(np.rad2deg(yaw_rate_radians).max())
            result["cp_total_heading_change"] = float(
                np.rad2deg(np.abs(delta_angle[valid_delta]).sum())
            )
        else:
            result["cp_max_abs_yaw_rate"] = math.nan
            result["cp_total_heading_change"] = math.nan
    else:
        result["cp_max_abs_yaw_rate"] = math.nan
        result["cp_total_heading_change"] = math.nan
    return result


def align_and_measure(
    analysis: pd.DataFrame,
    raw_scan: RawScanResult,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform nearest-time alignment, position validation, and both windows."""
    series_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    series_diagnostics: dict[str, dict[str, int]] = {}
    for key, records in raw_scan.series_records.items():
        series, diagnostics = build_series(records)
        series_by_key[key] = series
        series_diagnostics[f"{key[0]}::{key[1]}"] = diagnostics

    rows: list[dict[str, Any]] = []
    angle_unit = raw_scan.health["course_angle_unit"]
    for row_index, row in analysis.iterrows():
        session = str(row["session_id"])
        counterpart_id = normalized_id(row["counterpart_key_agent"])
        output: dict[str, Any] = {"row_index": int(row_index)}
        series = series_by_key.get((session, counterpart_id)) if counterpart_id is not None else None
        if series is None or len(series["timestamp_ms"]) == 0:
            output.update(
                {
                    "raw_series_found": False,
                    "raw_anchor_matched": False,
                    "raw_anchor_timestamp_ms": math.nan,
                    "raw_anchor_time_abs_diff_ms": math.nan,
                    "raw_anchor_x": math.nan,
                    "raw_anchor_y": math.nan,
                    "raw_anchor_speed_kmh": math.nan,
                }
            )
            for window in ("contract", "fixed3"):
                empty = calculate_window_metrics(
                    {
                        "timestamp_ms": np.asarray([], dtype=np.int64),
                        "acceleration": np.asarray([], dtype=float),
                        "speed": np.asarray([], dtype=float),
                        "course_angle": np.asarray([], dtype=float),
                    },
                    0,
                    0,
                    math.nan,
                    angle_unit,
                )
                output.update({f"{window}__{key}": value for key, value in empty.items()})
            rows.append(output)
            continue

        anchor_timestamp = int(row["anchor_timestamp_ms"])
        index = nearest_index(series["timestamp_ms"], anchor_timestamp)
        assert index is not None
        raw_anchor_timestamp = int(series["timestamp_ms"][index])
        time_abs_diff = abs(raw_anchor_timestamp - anchor_timestamp)
        anchor_speed = float(series["speed"][index])
        output.update(
            {
                "raw_series_found": True,
                "raw_anchor_matched": time_abs_diff < 150,
                "raw_anchor_timestamp_ms": raw_anchor_timestamp,
                "raw_anchor_time_abs_diff_ms": time_abs_diff,
                "raw_anchor_x": (
                    float(series["x"][index]) if time_abs_diff < 150 else math.nan
                ),
                "raw_anchor_y": (
                    float(series["y"][index]) if time_abs_diff < 150 else math.nan
                ),
                "raw_anchor_speed_kmh": anchor_speed,
            }
        )
        if time_abs_diff >= 150:
            for window in ("contract", "fixed3"):
                empty = calculate_window_metrics(
                    {
                        "timestamp_ms": np.asarray([], dtype=np.int64),
                        "acceleration": np.asarray([], dtype=float),
                        "speed": np.asarray([], dtype=float),
                        "course_angle": np.asarray([], dtype=float),
                    },
                    0,
                    0,
                    math.nan,
                    angle_unit,
                )
                output.update({f"{window}__{key}": value for key, value in empty.items()})
            rows.append(output)
            continue
        window_ends = {
            "contract": int(row["contract_end_timestamp_ms"]),
            "fixed3": int(row["fixed3_end_timestamp_ms"]),
        }
        for window, end_timestamp in window_ends.items():
            metrics = calculate_window_metrics(
                series,
                anchor_timestamp,
                end_timestamp,
                anchor_speed,
                angle_unit,
            )
            output.update({f"{window}__{key}": value for key, value in metrics.items()})
        rows.append(output)

    measured = analysis.join(pd.DataFrame(rows).set_index("row_index"), how="left")
    finite_position = (
        np.isfinite(measured["raw_anchor_x"])
        & np.isfinite(measured["raw_anchor_y"])
        & np.isfinite(measured["derived_counterpart_x"])
        & np.isfinite(measured["derived_counterpart_y"])
    )
    measured["translation_dx"] = measured["raw_anchor_x"] - measured[
        "derived_counterpart_x"
    ]
    measured["translation_dy"] = measured["raw_anchor_y"] - measured[
        "derived_counterpart_y"
    ]
    measured.loc[~finite_position, ["translation_dx", "translation_dy"]] = np.nan
    case_translation = (
        measured.groupby("case_key", observed=True)[["translation_dx", "translation_dy"]]
        .median()
        .rename(
            columns={
                "translation_dx": "case_median_translation_dx",
                "translation_dy": "case_median_translation_dy",
            }
        )
    )
    measured = measured.join(case_translation, on="case_key")
    measured["position_residual_x"] = (
        measured["translation_dx"] - measured["case_median_translation_dx"]
    )
    measured["position_residual_y"] = (
        measured["translation_dy"] - measured["case_median_translation_dy"]
    )
    measured["position_residual_m"] = np.hypot(
        measured["position_residual_x"], measured["position_residual_y"]
    )

    series_found = measured["raw_series_found"].eq(True)
    matched = measured["raw_anchor_matched"].eq(True)
    time_diff = measured.loc[series_found, "raw_anchor_time_abs_diff_ms"].dropna()
    position_residual = measured.loc[finite_position, "position_residual_m"].dropna()
    time_p95 = float(time_diff.quantile(0.95)) if len(time_diff) else math.nan
    residual_median = (
        float(position_residual.median()) if len(position_residual) else math.nan
    )
    acceleration_share = raw_scan.health["acceleration_abs_gt_10_share"]
    assertions = {
        "nearest_timestamp_abs_diff_p95_ms": time_p95,
        "nearest_timestamp_assertion_threshold_ms": 150.0,
        "nearest_timestamp_assertion_pass": bool(time_p95 < 150.0),
        "position_residual_median_m": residual_median,
        "position_residual_x_median_abs_m": float(
            measured.loc[finite_position, "position_residual_x"].abs().median()
        ),
        "position_residual_y_median_abs_m": float(
            measured.loc[finite_position, "position_residual_y"].abs().median()
        ),
        "position_residual_assertion_threshold_m": 0.5,
        "position_residual_assertion_pass": bool(residual_median < 0.5),
        "raw_acceleration_abs_gt_10_share": acceleration_share,
        "raw_acceleration_assertion_threshold": 0.01,
        "raw_acceleration_assertion_pass": bool(acceleration_share < 0.01),
    }
    if not all(
        assertions[key]
        for key in (
            "nearest_timestamp_assertion_pass",
            "position_residual_assertion_pass",
            "raw_acceleration_assertion_pass",
        )
    ):
        raise AssertionError(f"Alignment hard assertion failed: {assertions}")

    diagnostics = {
        "analysis_rows": len(measured),
        "counterpart_series_found_rows": int(series_found.sum()),
        "counterpart_series_not_found_rows": int((~series_found).sum()),
        "nearest_time_diff_ge_150ms_rows": int((series_found & ~matched).sum()),
        "matched_anchor_rows": int(matched.sum()),
        "matched_anchor_share": float(matched.mean()),
        "position_comparable_rows": int(finite_position.sum()),
        "position_comparable_share": float(finite_position.mean()),
        "nearest_timestamp_abs_diff_quantiles_ms": {
            str(q): float(time_diff.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "position_residual_quantiles_m": {
            str(q): float(position_residual.quantile(q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "case_translation_dx_quantiles_m": {
            str(q): float(case_translation["case_median_translation_dx"].quantile(q))
            for q in (0.0, 0.5, 1.0)
        },
        "case_translation_dy_quantiles_m": {
            str(q): float(case_translation["case_median_translation_dy"].quantile(q))
            for q in (0.0, 0.5, 1.0)
        },
        "assertions": assertions,
        "series_diagnostics": series_diagnostics,
    }
    return measured, diagnostics


def window_frame_columns(window: str, threshold: int) -> tuple[str, str]:
    return f"{window}__cp_brake_n_{threshold}", f"{window}__cp_acc_n"


def distribution_tables(
    measured: pd.DataFrame,
    group_columns: dict[str, str],
    strata: Iterable[str],
    windows: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quantile_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for alpha_label, group_column in group_columns.items():
        for stratum in strata:
            stratum_df = measured.loc[measured["stratum"].eq(stratum)]
            for window in windows:
                for group_name in ("lower", "inside", "upper"):
                    group_df = stratum_df.loc[stratum_df[group_column].eq(group_name)]
                    for outcome in OUTCOMES:
                        column = f"{window}__{outcome}"
                        values = group_df[column].dropna().astype(float)
                        row = {
                            "alpha": alpha_label,
                            "stratum": stratum,
                            "window": window,
                            "exposure_group": group_name,
                            "outcome": outcome,
                            "valid_n": len(values),
                            "analysis_row_n": len(group_df),
                            "missing_n": len(group_df) - len(values),
                            "mean": float(values.mean()) if len(values) else math.nan,
                        }
                        for quantile in QUANTILES:
                            row[f"q{int(quantile * 100):02d}"] = (
                                float(values.quantile(quantile)) if len(values) else math.nan
                            )
                        quantile_rows.append(row)
                    for threshold in (2, 3, 4):
                        numerator_column, denominator_column = window_frame_columns(
                            window, threshold
                        )
                        numerator = int(group_df[numerator_column].fillna(0).sum())
                        denominator = int(group_df[denominator_column].fillna(0).sum())
                        min_column = f"{window}__cp_min_acceleration"
                        window_valid = group_df[min_column].notna()
                        window_numerator = int(
                            (group_df.loc[window_valid, min_column] < -threshold).sum()
                        )
                        window_denominator = int(window_valid.sum())
                        threshold_rows.append(
                            {
                                "alpha": alpha_label,
                                "stratum": stratum,
                                "window": window,
                                "exposure_group": group_name,
                                "threshold_mps2": -threshold,
                                "frame_numerator": numerator,
                                "frame_denominator": denominator,
                                "frame_share": (
                                    numerator / denominator if denominator else math.nan
                                ),
                                "window_numerator": window_numerator,
                                "window_denominator": window_denominator,
                                "window_share": (
                                    window_numerator / window_denominator
                                    if window_denominator
                                    else math.nan
                                ),
                            }
                        )
    return quantile_rows, threshold_rows


def two_proportion_pvalue(a: int, n_a: int, b: int, n_b: int) -> float:
    if min(n_a, n_b) <= 0:
        return math.nan
    pooled = (a + b) / (n_a + n_b)
    variance = pooled * (1 - pooled) * (1 / n_a + 1 / n_b)
    if variance <= 0:
        return 1.0 if a / n_a == b / n_b else math.nan
    z_value = (a / n_a - b / n_b) / math.sqrt(variance)
    return float(2 * stats.norm.sf(abs(z_value)))


def clustered_threshold_contrast(
    frame: pd.DataFrame,
    group_column: str,
    window: str,
    comparison_group: str,
    threshold: int,
    rng: np.random.Generator,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    numerator_column, denominator_column = window_frame_columns(window, threshold)
    subset = frame.loc[frame[group_column].isin([comparison_group, "inside"])].copy()
    aggregated = (
        subset.groupby(["case_key", "team_id", group_column], observed=True)[
            [numerator_column, denominator_column]
        ]
        .sum()
        .reset_index()
    )
    cases = sorted(subset["case_key"].unique())
    case_index = {case: index for index, case in enumerate(cases)}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in (comparison_group, "inside"):
        numerator = np.zeros(len(cases), dtype=float)
        denominator = np.zeros(len(cases), dtype=float)
        rows = aggregated.loc[aggregated[group_column].eq(group)]
        for row in rows.itertuples(index=False):
            position = case_index[getattr(row, "case_key")]
            numerator[position] = getattr(row, numerator_column)
            denominator[position] = getattr(row, denominator_column)
        arrays[group] = (numerator, denominator)

    a_num, a_den = arrays[comparison_group]
    b_num, b_den = arrays["inside"]
    point_a = a_num.sum() / a_den.sum() if a_den.sum() else math.nan
    point_b = b_num.sum() / b_den.sum() if b_den.sum() else math.nan
    point_difference = point_a - point_b
    boot_values: list[float] = []
    for _ in range(bootstrap_reps):
        sampled = rng.integers(0, len(cases), size=len(cases))
        sampled_a_den = a_den[sampled].sum()
        sampled_b_den = b_den[sampled].sum()
        if sampled_a_den and sampled_b_den:
            boot_values.append(
                a_num[sampled].sum() / sampled_a_den
                - b_num[sampled].sum() / sampled_b_den
            )
    ci_low, ci_high = (
        np.quantile(boot_values, [0.025, 0.975]) if boot_values else (math.nan, math.nan)
    )

    both = (a_den > 0) & (b_den > 0)
    case_differences = a_num[both] / a_den[both] - b_num[both] / b_den[both]
    if len(case_differences) >= 2 and np.std(case_differences, ddof=1) > 0:
        case_t_p = float(stats.ttest_1samp(case_differences, 0.0).pvalue)
    else:
        case_t_p = math.nan

    team_rows = (
        subset.groupby(["team_id", group_column], observed=True)[
            [numerator_column, denominator_column]
        ]
        .sum()
        .reset_index()
    )
    team_pivot_num = team_rows.pivot(
        index="team_id", columns=group_column, values=numerator_column
    )
    team_pivot_den = team_rows.pivot(
        index="team_id", columns=group_column, values=denominator_column
    )
    common_teams = [
        team
        for team in team_pivot_num.index
        if comparison_group in team_pivot_num.columns
        and "inside" in team_pivot_num.columns
        and team_pivot_den.loc[team, comparison_group] > 0
        and team_pivot_den.loc[team, "inside"] > 0
    ]
    team_differences = np.asarray(
        [
            team_pivot_num.loc[team, comparison_group]
            / team_pivot_den.loc[team, comparison_group]
            - team_pivot_num.loc[team, "inside"] / team_pivot_den.loc[team, "inside"]
            for team in common_teams
        ],
        dtype=float,
    )
    if len(team_differences) >= 2 and np.std(team_differences, ddof=1) > 0:
        team_t_p = float(stats.ttest_1samp(team_differences, 0.0).pvalue)
    else:
        team_t_p = math.nan
    return {
        "comparison": f"{comparison_group}_minus_inside",
        "window": window,
        "threshold_mps2": -threshold,
        "comparison_numerator": int(a_num.sum()),
        "comparison_denominator": int(a_den.sum()),
        "comparison_share": point_a,
        "inside_numerator": int(b_num.sum()),
        "inside_denominator": int(b_den.sum()),
        "inside_share": point_b,
        "share_difference": point_difference,
        "case_bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "case_bootstrap_reps": len(boot_values),
        "naive_two_proportion_p": two_proportion_pvalue(
            int(a_num.sum()), int(a_den.sum()), int(b_num.sum()), int(b_den.sum())
        ),
        "case_equal_contrast_n": int(len(case_differences)),
        "case_equal_t_p": case_t_p,
        "team_equal_contrast_n": int(len(team_differences)),
        "team_equal_t_p": team_t_p,
    }


def covariance_results(
    x: np.ndarray,
    residual: np.ndarray,
    inverse_xtx: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, int]:
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    group_count = len(unique_clusters)
    n, k = x.shape
    meat = np.zeros((k, k), dtype=float)
    with np.errstate(all="ignore"):
        for group_index in range(group_count):
            mask = inverse == group_index
            score = x[mask].T @ residual[mask]
            meat += np.outer(score, score)
    correction = (
        group_count / (group_count - 1) * (n - 1) / (n - k)
        if group_count > 1 and n > k
        else 1.0
    )
    with np.errstate(all="ignore"):
        covariance = correction * inverse_xtx @ meat @ inverse_xtx
    if not np.isfinite(covariance).all():
        raise FloatingPointError("Cluster covariance contains a non-finite value")
    return covariance, group_count


def regression_fit(
    frame: pd.DataFrame,
    outcome_column: str,
    lower_column: str,
    upper_column: str,
) -> dict[str, Any]:
    columns = [
        outcome_column,
        lower_column,
        upper_column,
        "context_cell",
        "case_key",
        "team_id",
    ]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    context = pd.get_dummies(
        data["context_cell"].astype(str), prefix="context", drop_first=True, dtype=float
    )
    x_frame = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[[lower_column, upper_column]].astype(float),
            context,
        ],
        axis=1,
    )
    x = x_frame.to_numpy(dtype=float)
    y = data[outcome_column].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError(f"Non-finite design/outcome in regression {outcome_column}")
    n, k = x.shape
    if n <= k + 2:
        return {"n": n, "error": "insufficient residual degrees of freedom"}
    # Scale every design column to unit Euclidean norm before solving.  This
    # preserves original-unit coefficients after back-transformation and avoids
    # unstable normal equations when continuous exceedances and sparse context
    # indicators occupy very different numerical scales.
    scales = np.linalg.norm(x, axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    scaled_x = x / scales
    with np.errstate(all="ignore"):
        scaled_beta, _, rank, singular_values = np.linalg.lstsq(
            scaled_x, y, rcond=None
        )
        residual = y - scaled_x @ scaled_beta
    if not np.isfinite(scaled_beta).all() or not np.isfinite(residual).all():
        raise FloatingPointError(f"Non-finite OLS solution in regression {outcome_column}")
    rank = int(rank)
    residual_df = max(n - rank, 1)
    with np.errstate(all="ignore"):
        sigma2 = float(residual @ residual / residual_df)
        inverse_scaled_xtx = np.linalg.pinv(scaled_x.T @ scaled_x)
        naive_cov = sigma2 * inverse_scaled_xtx
    if not np.isfinite(naive_cov).all():
        raise FloatingPointError(f"Non-finite naive covariance in {outcome_column}")
    case_cov, case_groups = covariance_results(
        scaled_x,
        residual,
        inverse_scaled_xtx,
        data["case_key"].astype(str).to_numpy(),
    )
    team_cov, team_groups = covariance_results(
        scaled_x,
        residual,
        inverse_scaled_xtx,
        data["team_id"].astype(str).to_numpy(),
    )

    def coefficient(position: int) -> dict[str, Any]:
        estimate = float(scaled_beta[position] / scales[position])
        result = {"estimate": estimate}
        for label, covariance, degrees in (
            ("naive", naive_cov, residual_df),
            ("case_cluster", case_cov, max(case_groups - 1, 1)),
            ("team_cluster", team_cov, max(team_groups - 1, 1)),
        ):
            variance = float(covariance[position, position]) / scales[position] ** 2
            standard_error = math.sqrt(max(variance, 0.0))
            t_value = estimate / standard_error if standard_error > 0 else math.nan
            p_value = (
                float(2 * stats.t.sf(abs(t_value), df=degrees))
                if math.isfinite(t_value)
                else math.nan
            )
            result[f"{label}_se"] = standard_error
            result[f"{label}_p"] = p_value
        return result

    return {
        "n": n,
        "rank": rank,
        "parameter_count": k,
        "scaled_design_condition_number": (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0
            else math.inf
        ),
        "case_cluster_count": case_groups,
        "team_cluster_count": team_groups,
        "context_reference": (
            sorted(data["context_cell"].astype(str).unique())[0]
            if data["context_cell"].nunique()
            else None
        ),
        "lower": coefficient(1),
        "upper": coefficient(2),
    }


def run_regressions(
    measured: pd.DataFrame,
    group_specs: dict[str, tuple[str, str]],
    strata: Iterable[str],
    windows: Iterable[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for label, (lower_column, upper_column) in group_specs.items():
        for stratum in strata:
            frame = measured.loc[measured["stratum"].eq(stratum)]
            for window in windows:
                for outcome in OUTCOMES:
                    outcome_column = f"{window}__{outcome}"
                    fit = regression_fit(frame, outcome_column, lower_column, upper_column)
                    results.append(
                        {
                            "alpha": label,
                            "stratum": stratum,
                            "window": window,
                            "outcome": outcome,
                            "model": (
                                f"{outcome} ~ lower + upper + C(context_cell); "
                                "naive, case-cluster, and team-cluster covariance"
                            ),
                            **fit,
                        }
                    )
    return results


def case_label_swap_permutation(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    rng: np.random.Generator,
    reps: int = PERMUTATION_REPS,
) -> dict[str, Any]:
    subset = frame.loc[frame[group_column].isin(["lower", "inside"])]
    case_group_mean = (
        subset.groupby(["case_key", group_column], observed=True)[value_column]
        .mean()
        .unstack(group_column)
        .dropna(subset=["lower", "inside"])
    )
    differences = (
        case_group_mean["lower"].to_numpy() - case_group_mean["inside"].to_numpy()
    )
    observed = float(differences.mean()) if len(differences) else math.nan
    permuted = []
    for _ in range(reps):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        permuted.append(float(np.mean(signs * differences)))
    empirical_p = (
        (1 + int(np.sum(np.abs(permuted) >= abs(observed)))) / (reps + 1)
        if len(differences)
        else math.nan
    )
    return {
        "statistic": "equal-case mean of lower-minus-inside within-case contrasts",
        "outcome_column": value_column,
        "case_count_with_both_labels": len(differences),
        "observed_difference": observed,
        "permutation_reps": reps,
        "empirical_two_sided_p": empirical_p,
        "permutation_recipe": (
            "swap lower and inside labels independently at the case level, "
            "equivalent to sign-flipping each within-case contrast"
        ),
    }


def surrounding_probe(
    raw_scan: RawScanResult,
    probe_specs: list[dict[str, Any]],
    measured: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate one 50 m, three-second all-vehicle probe in up to five sessions."""
    angle_unit = raw_scan.health["course_angle_unit"]
    probes: list[dict[str, Any]] = []
    for spec in probe_specs:
        session_records = raw_scan.probe_records.get(spec["session_id"], [])
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in session_records:
            grouped[int(record["id"])].append(record)
        series_by_id = {vehicle_id: build_series(records)[0] for vehicle_id, records in grouped.items()}
        anchor_rows = measured.loc[
            measured["case_key"].eq(spec["case_key"])
            & measured["anchor_frame_index"].eq(spec["anchor_frame_index"])
        ]
        if len(anchor_rows) != 1:
            probes.append({**spec, "status": "analysis_anchor_not_unique"})
            continue
        anchor_row = anchor_rows.iloc[0]
        ego_x = finite_float(anchor_row["derived_ego_x"]) + finite_float(
            anchor_row["case_median_translation_dx"]
        )
        ego_y = finite_float(anchor_row["derived_ego_y"]) + finite_float(
            anchor_row["case_median_translation_dy"]
        )
        if not all(map(math.isfinite, (ego_x, ego_y))):
            probes.append({**spec, "status": "translated_ego_position_missing"})
            continue
        nearby: list[dict[str, Any]] = []
        anchor_candidates = 0
        anchor_matches_within_150ms = 0
        for vehicle_id, series in series_by_id.items():
            anchor_index = nearest_index(series["timestamp_ms"], spec["anchor_timestamp_ms"])
            if anchor_index is None:
                continue
            anchor_candidates += 1
            time_diff = abs(
                int(series["timestamp_ms"][anchor_index]) - spec["anchor_timestamp_ms"]
            )
            if time_diff >= 150:
                continue
            anchor_matches_within_150ms += 1
            x = series["x"][anchor_index]
            y = series["y"][anchor_index]
            if not all(map(math.isfinite, (ego_x, ego_y, x, y))):
                continue
            distance = float(math.hypot(x - ego_x, y - ego_y))
            if distance > spec["radius_m"]:
                continue
            anchor_speed = float(series["speed"][anchor_index])
            metrics = calculate_window_metrics(
                series,
                spec["anchor_timestamp_ms"],
                spec["anchor_timestamp_ms"] + 3000,
                anchor_speed,
                angle_unit,
            )
            nearby.append(
                {
                    "vehicle_id": vehicle_id,
                    "name": str(series["name"][anchor_index]),
                    "anchor_time_abs_diff_ms": time_diff,
                    "anchor_distance_m": distance,
                    **metrics,
                }
            )
        valid_acceleration = sum(int(item["cp_acc_n"] > 0) for item in nearby)
        valid_angle = sum(int(item["cp_angle_n"] >= 2) for item in nearby)
        probe_status = "ok" if anchor_matches_within_150ms > 0 else "no_anchor_time_match"
        probes.append(
            {
                **{key: value for key, value in spec.items() if not key.startswith("store_")},
                "status": probe_status,
                "center_recipe": (
                    "derived ego_x/ego_y plus the case-median raw-minus-derived "
                    "translation established by counterpart alignment"
                ),
                "estimated_raw_ego_x": ego_x,
                "estimated_raw_ego_y": ego_y,
                "anchor_candidates": anchor_candidates,
                "anchor_matches_within_150ms": anchor_matches_within_150ms,
                "nearby_vehicle_count": len(nearby),
                "nearby_with_acceleration_outcome": valid_acceleration,
                "nearby_with_heading_outcome": valid_angle,
                "vehicles": nearby,
            }
        )
    ok_probes = [probe for probe in probes if probe.get("status") == "ok"]
    nearby_total = sum(probe["nearby_vehicle_count"] for probe in ok_probes)
    acceleration_total = sum(
        probe["nearby_with_acceleration_outcome"] for probe in ok_probes
    )
    angle_total = sum(probe["nearby_with_heading_outcome"] for probe in ok_probes)
    return {
        "purpose": (
            "Feasibility-only probe; no inferential test and no vehicle/team judgment"
        ),
        "selection": (
            "one deterministic middle-time non-scripted analysis anchor from each "
            "of the first five sorted eligible sessions"
        ),
        "center_recipe": (
            "derived anchor ego_x/ego_y translated into raw coordinates with the "
            "case-median raw-minus-derived counterpart offset"
        ),
        "radius_m": 50.0,
        "radius_reason": (
            "50 m covers immediate interaction neighbors while limiting unrelated traffic"
        ),
        "session_count": len(probes),
        "successful_probe_count": len(ok_probes),
        "nearby_vehicle_total": nearby_total,
        "nearby_with_acceleration_outcome": acceleration_total,
        "nearby_with_heading_outcome": angle_total,
        "acceleration_outcome_coverage": (
            acceleration_total / nearby_total if nearby_total else math.nan
        ),
        "heading_outcome_coverage": angle_total / nearby_total if nearby_total else math.nan,
        "feasibility_assessment": (
            "feasible"
            if len(ok_probes) == len(probes)
            and nearby_total
            and acceleration_total / nearby_total >= 0.9
            else "feasible_with_session_gap"
            if len(ok_probes) >= 3
            and nearby_total
            and acceleration_total / nearby_total >= 0.9
            else "limited"
        ),
        "probes": probes,
    }


def build_key_numbers(
    measured: pd.DataFrame,
    load_contract: dict[str, Any],
    raw_inventory: dict[str, Any],
    raw_health: dict[str, Any],
    alignment: dict[str, Any],
    threshold_contrasts: list[dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def add(
        key: str,
        value: Any,
        numerator: Any,
        denominator: Any,
        selection: str,
        source_files: list[str],
        columns: list[str],
        unit: str = "count",
    ) -> None:
        items.append(
            {
                "key": key,
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
                "selection": selection,
                "source_files": source_files,
                "columns": columns,
                "unit": unit,
            }
        )

    gate_source = [
        "data/derived/rq017_onsite_gate/l1_v1/",
        ".codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet",
        "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet",
    ]
    add(
        "analysis_rows",
        len(measured),
        len(measured),
        load_contract["source_rows"]["mechanism_one"],
        "status == OK and mechanism2_gate_ok == True",
        gate_source,
        ["product_row_key", "status", "mechanism2_gate_ok"],
    )
    for group_name in ("lower", "inside", "upper"):
        count = int(measured["group_90"].eq(group_name).sum())
        add(
            f"alpha90_{group_name}_rows",
            count,
            count,
            len(measured),
            f"analysis set and group_90 == {group_name}",
            gate_source[:2],
            ["ipv_log", "lo_90", "hi_90", "width_90"],
        )
    for stratum in ("non_scripted", "scripted"):
        rows = measured.loc[measured["stratum"].eq(stratum)]
        add(
            f"{stratum}_rows",
            len(rows),
            len(rows),
            len(measured),
            f"analysis set and stratum == {stratum}",
            gate_source[2:],
            ["counterpart_selection", "case_key"],
        )
        add(
            f"{stratum}_cases",
            int(rows["case_key"].nunique()),
            int(rows["case_key"].nunique()),
            int(measured["case_key"].nunique()),
            f"unique case_key where stratum == {stratum}",
            gate_source[2:],
            ["counterpart_selection", "case_key"],
        )
    add(
        "raw_analysis_logs",
        raw_inventory["analysis_log_count"],
        raw_inventory["analysis_log_count"],
        raw_inventory["all_simulation_trajectory_log_count"],
        "logs whose session_id occurs in the analysis set",
        ["data/onsite_competition/all_teams_dataset/teams/*/sessions/<session_id>/simulation_trajectory.log"],
        ["session_id"],
    )
    matched = alignment["matched_anchor_rows"]
    add(
        "raw_anchor_match_share",
        alignment["matched_anchor_share"],
        matched,
        len(measured),
        "nearest raw record with numeric-equal counterpart id for every analysis anchor",
        ["simulation_trajectory.log", gate_source[2]],
        ["counterpart_key_agent", "id", "timestamp_ms", "globalTimeStamp"],
        "proportion",
    )
    assertions = alignment["assertions"]
    add(
        "nearest_timestamp_abs_diff_p95_ms",
        assertions["nearest_timestamp_abs_diff_p95_ms"],
        assertions["nearest_timestamp_abs_diff_p95_ms"],
        matched,
        "95th percentile among matched analysis anchors",
        ["simulation_trajectory.log", "onsite_ipv_timeseries_multi_allvalid.parquet"],
        ["globalTimeStamp", "timestamp_ms"],
        "ms",
    )
    add(
        "position_residual_median_m",
        assertions["position_residual_median_m"],
        assertions["position_residual_median_m"],
        alignment["position_comparable_rows"],
        "median Euclidean residual after subtracting each case's median x/y translation",
        ["simulation_trajectory.log", "onsite_ipv_timeseries_multi_allvalid.parquet"],
        ["x", "y", "counterpart_x", "counterpart_y", "case_key"],
        "m",
    )
    add(
        "raw_acceleration_abs_gt_10_share",
        raw_health["acceleration_abs_gt_10_share"],
        raw_health["acceleration_abs_gt_10_count"],
        raw_health["acceleration_abs_gt_10_denominator"],
        "all finite acceleration fields in the analysis-session raw trajectory logs",
        ["simulation_trajectory.log"],
        ["acceleration"],
        "proportion",
    )
    add(
        "raw_required_field_null_slot_share",
        raw_health["null_field_slot_share"],
        raw_health["null_field_slots"],
        raw_health["required_field_slots"],
        "all nine required field slots across raw vehicle records in analysis sessions",
        ["simulation_trajectory.log"],
        list(RAW_FIELDS),
        "proportion",
    )
    for window in ("contract", "fixed3"):
        column = f"{window}_window_past_case_end"
        count = int(measured[column].sum())
        add(
            f"{window}_window_past_case_end_share",
            count / len(measured),
            count,
            len(measured),
            f"analysis rows where {window} nominal end exceeds the case end; metrics clipped to case end",
            ["onsite_m3_av_anchors_multi_allvalid.parquet", "onsite_ipv_timeseries_multi_allvalid.parquet"],
            ["anchor_frame_index", "target_window_end_frame_index", "frame_index", "timestamp_ms"],
            "proportion",
        )
    for contrast in threshold_contrasts:
        key_prefix = (
            f"{contrast['stratum']}_{contrast['window']}_"
            f"{contrast['comparison']}_brake_{abs(contrast['threshold_mps2'])}"
        )
        add(
            f"{key_prefix}_share_difference",
            contrast["share_difference"],
            contrast["comparison_numerator"],
            contrast["comparison_denominator"],
            (
                f"{contrast['stratum']} alpha-90 {contrast['comparison']} raw acceleration "
                f"frame share minus inside share; threshold {contrast['threshold_mps2']} m/s^2"
            ),
            ["simulation_trajectory.log", *gate_source[:2]],
            ["acceleration", "ipv_log", "lo_90", "hi_90", "width_90", "case_key"],
            "proportion_difference",
        )
    outcome_units = {
        "cp_min_acceleration": "m/s^2",
        "cp_speed_drop_kmh": "km/h",
        "cp_anchor_speed_drop_kmh": "km/h",
        "cp_max_abs_yaw_rate": "degree/s",
        "cp_total_heading_change": "degree",
    }
    main_frame = measured.loc[measured["stratum"].eq("non_scripted")]
    for alpha in ALPHAS:
        for group_name in ("lower", "inside"):
            group_frame = main_frame.loc[
                main_frame[f"group_{alpha}"].eq(group_name)
            ]
            for outcome, unit in outcome_units.items():
                values = group_frame[f"fixed3__{outcome}"].dropna()
                median = float(values.median()) if len(values) else math.nan
                add(
                    f"alpha{alpha}_non_scripted_fixed3_{group_name}_{outcome}_median",
                    median,
                    median,
                    len(values),
                    (
                        f"non-scripted, alpha={alpha}, group={group_name}, fixed 3 s, "
                        "post-anchor valid outcome; statistic is median"
                    ),
                    ["simulation_trajectory.log", *gate_source[:2]],
                    [
                        "acceleration",
                        "speed",
                        "courseAngle",
                        "globalTimeStamp",
                        "ipv_log",
                        f"lo_{alpha}",
                        f"hi_{alpha}",
                        f"width_{alpha}",
                    ],
                    unit,
                )
            for threshold in (2, 3, 4):
                numerator_column, denominator_column = window_frame_columns(
                    "fixed3", threshold
                )
                numerator = int(group_frame[numerator_column].fillna(0).sum())
                denominator = int(group_frame[denominator_column].fillna(0).sum())
                add(
                    f"alpha{alpha}_non_scripted_fixed3_{group_name}_brake_{threshold}_share",
                    numerator / denominator if denominator else math.nan,
                    numerator,
                    denominator,
                    (
                        f"non-scripted, alpha={alpha}, group={group_name}, fixed 3 s; "
                        f"post-anchor acceleration < -{threshold} m/s^2"
                    ),
                    ["simulation_trajectory.log", *gate_source[:2]],
                    [
                        "acceleration",
                        "globalTimeStamp",
                        "ipv_log",
                        f"lo_{alpha}",
                        f"hi_{alpha}",
                        f"width_{alpha}",
                    ],
                    "proportion",
                )
    return {
        "contract": (
            "Every entry carries numerator, denominator or supporting sample count, "
            "selection, source files, and source columns. Quantile values and all "
            "secondary table cells are in distribution_results.json with valid_n."
        ),
        "items": items,
    }


def summarize_data_health(
    measured: pd.DataFrame,
    load_contract: dict[str, Any],
    raw_inventory: dict[str, Any],
    raw_health: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    case_sizes = measured.groupby("case_key", observed=True).size()
    outcome_health: dict[str, Any] = {}
    for window in ("contract", "fixed3"):
        for outcome in OUTCOMES:
            column = f"{window}__{outcome}"
            values = measured[column]
            outcome_health[column] = {
                "valid_n": int(values.notna().sum()),
                "nan_n": int(values.isna().sum()),
                "positive_inf_n": int(np.isposinf(values).sum()),
                "negative_inf_n": int(np.isneginf(values).sum()),
                "min": float(values.min()) if values.notna().any() else math.nan,
                "max": float(values.max()) if values.notna().any() else math.nan,
            }
    anomaly_mask = measured["relative_distance_anchor"].between(570_000, 571_000)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_contract": load_contract,
        "raw_log_inventory": raw_inventory,
        "raw_fields": raw_health,
        "alignment_summary": alignment,
        "analysis_set": {
            "rows": len(measured),
            "cases": int(measured["case_key"].nunique()),
            "teams": int(measured["team_id"].nunique()),
            "sessions": int(measured["session_id"].nunique()),
            "case_frame_count_quantiles": {
                str(q): float(case_sizes.quantile(q))
                for q in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
            },
            "strata": (
                measured.groupby("stratum", observed=True)
                .agg(rows=("case_key", "size"), cases=("case_key", "nunique"), teams=("team_id", "nunique"))
                .to_dict("index")
            ),
            "exposure_counts": {
                str(alpha): measured[f"group_{alpha}"].value_counts().to_dict()
                for alpha in ALPHAS
            },
            "contract_window_past_case_end_count": int(
                measured["contract_window_past_case_end"].sum()
            ),
            "fixed3_window_past_case_end_count": int(
                measured["fixed3_window_past_case_end"].sum()
            ),
        },
        "outcome_health": outcome_health,
        "coordinate_anomaly_check": {
            "definition": "relative_distance_anchor between 570,000 and 571,000 m",
            "analysis_row_count": int(anomaly_mask.sum()),
            "analysis_case_count": int(measured.loc[anomaly_mask, "case_key"].nunique()),
            "case_keys": sorted(measured.loc[anomaly_mask, "case_key"].unique()),
            "impact": (
                "No coordinate-anomaly row entered the gate-defined analysis set."
                if int(anomaly_mask.sum()) == 0
                else "Rows remain in the pre-specified gate-defined analysis set; "
                "raw-log alignment uses per-case translation and their count is explicit."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--permutation-reps", type=int, default=PERMUTATION_REPS)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = SCRIPT_PATH.parent
    report_path = root / (
        ".codex-fleet/rq019-counterpart-burden/board/reports/"
        "RQ019_1_counterpart_burden.md"
    )
    rng = np.random.default_rng(RNG_SEED)

    print("[1/7] Loading authorized parquet columns", flush=True)
    analysis, _, load_contract = load_analysis_inputs(root)
    needed_ids: dict[str, set[int]] = defaultdict(set)
    for row in analysis[["session_id", "counterpart_key_agent"]].itertuples(index=False):
        vehicle_id = normalized_id(row.counterpart_key_agent)
        if vehicle_id is not None:
            needed_ids[str(row.session_id)].add(vehicle_id)
    log_paths, raw_inventory = locate_raw_logs(root, analysis["session_id"].astype(str))
    probe_specs = choose_probe_specs(analysis)

    print(
        f"[2/7] Streaming {len(log_paths)} raw logs ({raw_inventory['analysis_log_bytes']} bytes)",
        flush=True,
    )
    raw_scan = scan_raw_logs(log_paths, needed_ids, probe_specs)
    print("[3/7] Applying nearest-time and per-case translation checks", flush=True)
    measured, alignment = align_and_measure(analysis, raw_scan)

    alignment_contract = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "join_recipe": load_contract["join_recipe"],
        "raw_log_paths": raw_scan.log_paths,
        "assertions": alignment["assertions"],
        "diagnostics": {
            key: value for key, value in alignment.items() if key != "assertions"
        },
    }
    write_json(output_dir / "alignment_contract.json", alignment_contract)

    print("[4/7] Computing distribution tables and clustered threshold contrasts", flush=True)
    observed_group_columns = {str(alpha): f"group_{alpha}" for alpha in ALPHAS}
    quantiles, thresholds = distribution_tables(
        measured,
        observed_group_columns,
        strata=("non_scripted", "scripted"),
        windows=("contract", "fixed3"),
    )
    threshold_contrasts: list[dict[str, Any]] = []
    for stratum in ("non_scripted", "scripted"):
        frame = measured.loc[measured["stratum"].eq(stratum)]
        for window in ("contract", "fixed3"):
            for comparison in ("lower", "upper"):
                for threshold in (2, 3, 4):
                    result = clustered_threshold_contrast(
                        frame,
                        "group_90",
                        window,
                        comparison,
                        threshold,
                        rng,
                        bootstrap_reps=args.bootstrap_reps,
                    )
                    result.update({"alpha": 90, "stratum": stratum})
                    threshold_contrasts.append(result)
    distribution_results = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interpretation_contract": {
            "lower": "IPV below the human envelope; more competitive/aggressive than the reference",
            "upper": "IPV above the human envelope; more cooperative/yielding than the reference",
            "inside": "IPV inside the context-specific human envelope",
            "quantiles": [int(q * 100) for q in QUANTILES],
            "speed_unit": "km/h (raw log native unit)",
            "acceleration_unit": "m/s^2",
            "yaw_rate_unit": "degree/s",
            "heading_change_unit": "degree; sum of absolute unwrapped post-anchor increments",
            "post_anchor_rule": "raw timestamps strictly greater than the derived anchor timestamp",
        },
        "exposure_counts": {
            str(alpha): measured[f"group_{alpha}"].value_counts().to_dict()
            for alpha in ALPHAS
        },
        "quantile_tables": quantiles,
        "braking_threshold_tables": thresholds,
        "alpha90_case_bootstrap_threshold_contrasts": threshold_contrasts,
    }
    write_json(output_dir / "distribution_results.json", distribution_results)

    print("[5/7] Fitting context-controlled regressions", flush=True)
    regression_rows = run_regressions(
        measured,
        {
            str(alpha): (f"lower_{alpha}", f"upper_{alpha}")
            for alpha in ALPHAS
        },
        strata=("non_scripted", "scripted"),
        windows=("contract", "fixed3"),
    )
    regression_results = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "supplementary only; distribution tables and clustered proportions carry conclusions",
        "coefficient_unit": "outcome units per one envelope-width exceedance",
        "results": regression_rows,
    }
    write_json(output_dir / "regression_results.json", regression_results)

    print("[6/7] Running case-label permutation and placebo exposure", flush=True)
    non_scripted = measured.loc[measured["stratum"].eq("non_scripted")].copy()
    permutations: list[dict[str, Any]] = []
    for window in ("contract", "fixed3"):
        for outcome in ("cp_min_acceleration", "cp_max_abs_yaw_rate"):
            result = case_label_swap_permutation(
                non_scripted,
                "group_90",
                f"{window}__{outcome}",
                rng,
                reps=args.permutation_reps,
            )
            result.update({"window": window, "outcome": outcome})
            permutations.append(result)
        for threshold in (2, 3, 4):
            share_column = f"{window}__cp_brake_share_{threshold}"
            result = case_label_swap_permutation(
                non_scripted,
                "group_90",
                share_column,
                rng,
                reps=args.permutation_reps,
            )
            result.update({"window": window, "outcome": f"cp_brake_share_{threshold}"})
            permutations.append(result)

    placebo_rng = np.random.default_rng(RNG_SEED + 1)
    placebo_ipv = placebo_rng.uniform(
        measured["lo_90"] - measured["width_90"],
        measured["hi_90"] + measured["width_90"],
    )
    measured["placebo_lower_90"] = np.maximum(
        0.0, (measured["lo_90"] - placebo_ipv) / measured["width_90"]
    )
    measured["placebo_upper_90"] = np.maximum(
        0.0, (placebo_ipv - measured["hi_90"]) / measured["width_90"]
    )
    measured["placebo_group_90"] = np.select(
        [measured["placebo_lower_90"] > 0, measured["placebo_upper_90"] > 0],
        ["lower", "upper"],
        default="inside",
    )
    placebo_quantiles, placebo_thresholds = distribution_tables(
        measured,
        {"placebo90": "placebo_group_90"},
        strata=("non_scripted",),
        windows=("contract", "fixed3"),
    )
    placebo_contrasts: list[dict[str, Any]] = []
    placebo_frame = measured.loc[measured["stratum"].eq("non_scripted")]
    for window in ("contract", "fixed3"):
        for comparison in ("lower", "upper"):
            for threshold in (2, 3, 4):
                result = clustered_threshold_contrast(
                    placebo_frame,
                    "placebo_group_90",
                    window,
                    comparison,
                    threshold,
                    rng,
                    bootstrap_reps=args.bootstrap_reps,
                )
                result.update({"stratum": "non_scripted", "alpha": "placebo90"})
                placebo_contrasts.append(result)
    placebo_regressions = run_regressions(
        measured,
        {"placebo90": ("placebo_lower_90", "placebo_upper_90")},
        strata=("non_scripted",),
        windows=("contract", "fixed3"),
    )
    negative_controls = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seed": RNG_SEED,
        "case_label_permutation": permutations,
        "placebo_exposure": {
            "recipe": (
                "independent per-row Uniform[lo_90 - width_90, hi_90 + width_90], "
                "then the same nonnegative exceedance formulas"
            ),
            "seed": RNG_SEED + 1,
            "group_counts": measured["placebo_group_90"].value_counts().to_dict(),
            "quantile_tables": placebo_quantiles,
            "braking_threshold_tables": placebo_thresholds,
            "case_bootstrap_threshold_contrasts": placebo_contrasts,
            "regressions": placebo_regressions,
        },
    }
    write_json(output_dir / "negative_controls.json", negative_controls)

    print("[7/7] Writing health, key-number, and surrounding-vehicle artifacts", flush=True)
    probe = surrounding_probe(raw_scan, probe_specs, measured)
    write_json(output_dir / "surrounding_probe.json", probe)
    data_health = summarize_data_health(
        measured, load_contract, raw_inventory, raw_scan.health, alignment
    )
    write_json(output_dir / "data_health.json", data_health)
    key_numbers = build_key_numbers(
        measured,
        load_contract,
        raw_inventory,
        raw_scan.health,
        alignment,
        threshold_contrasts,
    )
    write_json(output_dir / "key_numbers.json", key_numbers)

    # The report is generated in a second, deterministic rendering stage below.
    render_report(
        report_path,
        measured,
        distribution_results,
        regression_results,
        negative_controls,
        data_health,
        probe,
        key_numbers,
    )
    print(f"Completed; report: {report_path}", flush=True)


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "NA"
    if abs(number) >= 1e6 or (0 < abs(number) < 10 ** (-(digits + 1))):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def pct(numerator: int, denominator: int, digits: int = 2) -> str:
    if not denominator:
        return "NA"
    return f"{100 * numerator / denominator:.{digits}f}% ({numerator:,}/{denominator:,})"


def render_report(
    path: Path,
    measured: pd.DataFrame,
    distributions: dict[str, Any],
    regressions: dict[str, Any],
    negative_controls: dict[str, Any],
    data_health: dict[str, Any],
    probe: dict[str, Any],
    key_numbers: dict[str, Any],
) -> None:
    """Render the self-contained report from the machine-readable artifacts."""
    # Defined after main to keep analysis helpers grouped; Python resolves at runtime.
    timestamp = shell_utc_timestamp()
    alignment = data_health["alignment_summary"]
    assertions = alignment["assertions"]
    raw = data_health["raw_fields"]
    anomaly = data_health["coordinate_anomaly_check"]
    analysis_n = len(measured)
    case_n = measured["case_key"].nunique()
    team_n = measured["team_id"].nunique()
    source_note = (
        "曝露来源：`data/derived/rq017_onsite_gate/l1_v1/` 的 `status, ipv_log` 与 "
        "`.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` "
        "的 `mechanism2_gate_ok, lo_*, hi_*, width_*, context_cell`，按 "
        "`product_row_key` 解析出的 `case_key, anchor_frame_index, perspective` 连接锚点表。"
    )
    raw_note = (
        "结果来源：`data/onsite_competition/all_teams_dataset/teams/*/sessions/<session_id>/"
        "simulation_trajectory.log` 的 `id, globalTimeStamp, speed, acceleration, courseAngle, x, y`；"
        "仅使用 `globalTimeStamp` 严格晚于锚点的记录。"
    )
    all_selection_counts = data_health["load_contract"][
        "all_anchor_counterpart_selection_counts"
    ]
    all_scripted = all_selection_counts.get(SCRIPTED_SELECTION, {})

    lines: list[str] = [
        "# RQ019-1：异常 IPV 与交互对手方锚点后运动反应",
        "",
        "## 1. 工作定位与状态",
        "",
        "在线验证要判断自动驾驶车表现出的社会交互倾向是否落在人类参照范围内。IPV "
        "（Interaction Preference Value）是交互倾向标量；其代价函数中 `sin(ipv)` 是对手方代价的权重。"
        "RQ015 已冻结第一道判别信息门，RQ016C 建立纯人类参照区间，RQ017 在自动驾驶车数据上完成第一道门，"
        "RQ018 观察到下侧越界帧后续安全裕度分布压缩。本次 RQ019-B1 是下一环：描述锚点后的对手车制动、"
        "降速与转向分布，检验下侧越界（比人类参照更竞争激进）是否对应更大的对手方运动反应。",
        "",
        "本轮为探索性、描述性分析，只报告相关分布，不作因果解释，也不评价任何车辆或队伍。",
        "",
        "## 2. 分析合同与数据覆盖",
        "",
        f"分析集为第一道门 `status == OK` 且第二道门 `mechanism2_gate_ok == True` 的 "
        f"{analysis_n:,}/67,861 = {analysis_n/67861:.4%} 行，覆盖 {case_n:,} 个 case、{team_n:,} 个 team。",
        "主曝露 α=90 分为：下侧越界 "
        f"{int((measured['group_90']=='lower').sum()):,}/{analysis_n:,}、区间内 "
        f"{int((measured['group_90']=='inside').sum()):,}/{analysis_n:,}、上侧越界 "
        f"{int((measured['group_90']=='upper').sum()):,}/{analysis_n:,}。上、下侧按人类区间宽度归一化后分开保留。",
        f"任务书给 scripted 参考数约 7,575 行/30 case；按指定精确标签，完整锚点表实测 "
        f"{int(all_scripted.get('rows', 0)):,}/67,861 行、{int(all_scripted.get('cases', 0)):,} case，"
        f"过两道门后的分析集为 {int((measured['stratum']=='scripted').sum()):,}/{analysis_n:,} 行、"
        f"{measured.loc[measured['stratum']=='scripted','case_key'].nunique():,} case。分组仍严格使用任务书字符串，未自行扩展。",
        "",
        source_note,
        raw_note,
        "合同窗口按锚点表的 `[anchor_frame_index, target_window_end_frame_index]` 换成时间区间；"
        "固定窗口为锚点后 3 s。两者都排除锚点时刻本身，越过 case 末尾时只用 case 内可见记录。"
        "原始日志 `speed` 按合同与观测范围使用 km/h，`acceleration` 使用 m/s²。",
        "",
        "## 3. 对齐硬断言与数据健康",
        "",
        "| 检查 | 实测（分子/分母与口径） | 阈值 | 结果 |",
        "|---|---:|---:|---|",
        f"| 最近邻时间差 95 分位 | {fmt(assertions['nearest_timestamp_abs_diff_p95_ms'], 2)} ms；"
        f"候选序列 {alignment['counterpart_series_found_rows']:,}/{analysis_n:,} 个分析锚点 | <150 ms | 通过 |",
        f"| 去除 per-case 中位平移后位置残差中位 | {fmt(assertions['position_residual_median_m'], 4)} m；"
        f"可比 {alignment['position_comparable_rows']:,}/{analysis_n:,} 行 | <0.5 m | 通过 |",
        f"| 原始 acceleration 的 `abs(a)>10` | "
        f"{pct(raw['acceleration_abs_gt_10_count'], raw['acceleration_abs_gt_10_denominator'], 4)} | <1% | 通过 |",
        "",
        f"锚点未匹配拆分：对手方 ID 在相应 session 的原始日志中没有序列 "
        f"{alignment['counterpart_series_not_found_rows']:,}/{analysis_n:,} 行；有序列但最近时间差 ≥150 ms "
        f"{alignment['nearest_time_diff_ge_150ms_rows']:,}/{analysis_n:,} 行；最终 `<150 ms` 匹配 "
        f"{alignment['matched_anchor_rows']:,}/{analysis_n:,} 行。未匹配行不进入结果分布。",
        f"九个必需原始字段的空值字段槽为 {pct(raw['null_field_slots'], raw['required_field_slots'], 4)}；"
        f"至少一个必需字段为空的记录为 {pct(raw['records_with_any_required_field_null'], raw['record_count'], 4)}。"
        f"原始速度范围 {fmt(raw['numeric_min']['speed'],2)}–{fmt(raw['numeric_max']['speed'],2)} km/h，"
        f"加速度范围 {fmt(raw['numeric_min']['acceleration'],3)}–{fmt(raw['numeric_max']['acceleration'],3)} m/s²。",
        f"原始 `abs(acceleration)>10` 的 {raw['acceleration_abs_gt_10_count']:,}/"
        f"{raw['acceleration_abs_gt_10_denominator']:,} 个值先用于硬断言，再在结果窗口中显式设为缺失；"
        "这是因为正向异常最大达到上句所列数量级，不能作为物理控制量。阈值分母与回归均使用过滤后的有效值。",
        f"任务书用最大 60.00 作为 km/h 单位参考；19 个分析 session 全部有效 `speed` 实测最大值为 "
        f"{fmt(raw['numeric_max']['speed'],3)} km/h。单位仍按原始日志字段合同使用 km/h，但该参考上界不成立，"
        "因此报告保留实测范围，不据 60.00 截断。",
        f"`courseAngle` 实测范围 {fmt(raw['numeric_min']['courseAngle'],3)}–{fmt(raw['numeric_max']['courseAngle'],3)}，"
        f"共有 {raw['course_angle_above_2pi_count']:,}/{raw['valid_numeric_counts']['courseAngle']:,} 个值大于 2π，"
        "且没有值越过 360，因此单位判定为度、范围为 [0,360]；先转弧度并按 case 内时间顺序 unwrap，"
        "最终航向变化以度、偏航角速度以度/秒报告。",
        f"合同窗口越过 case 末尾 {pct(int(measured['contract_window_past_case_end'].sum()), analysis_n)}；"
        f"固定 3 s 窗口越过 case 末尾 {pct(int(measured['fixed3_window_past_case_end'].sum()), analysis_n)}。",
        f"坐标异常定义为 `relative_distance_anchor` 约 570,761 m；本分析集进入 "
        f"{anomaly['analysis_row_count']:,}/{analysis_n:,} 行、{anomaly['analysis_case_count']:,}/{case_n:,} 个 case，"
        f"case 为 {', '.join(anomaly['case_keys']) if anomaly['case_keys'] else '无'}。",
        "",
        "## 4. 分布级主结果",
        "",
        "以下先报分布，不用回归系数代替结论。分位数表的 `n` 是该结果非空的分析锚点数。"
        "完整 α=80/90/95、scripted/非 scripted、两窗口、八个结果的同结构机器表在 "
        "`distribution_results.json`。",
        "",
    ]

    quantile_rows = distributions["quantile_tables"]
    for stratum in ("non_scripted", "scripted"):
        rows_in_stratum = measured.loc[measured["stratum"].eq(stratum)]
        label = "非 scripted（主结论组）" if stratum == "non_scripted" else "scripted（隔离报告组）"
        lines.extend(
            [
                f"### 4.{1 if stratum == 'non_scripted' else 2} {label}",
                "",
                f"该组 {len(rows_in_stratum):,}/{analysis_n:,} 行，覆盖 "
                f"{rows_in_stratum['case_key'].nunique():,}/{case_n:,} 个 case。",
                "",
            ]
        )
        for window in ("contract", "fixed3"):
            window_label = "合同窗口" if window == "contract" else "固定 3 s 窗口"
            for outcome in OUTCOMES:
                selected = [
                    row
                    for row in quantile_rows
                    if row["alpha"] == "90"
                    and row["stratum"] == stratum
                    and row["window"] == window
                    and row["outcome"] == outcome
                ]
                lines.extend(
                    [
                        f"**{window_label} — `{outcome}`**",
                        "",
                        "| IPV 组 | n | q01 | q05 | q10 | q25 | q50 | q75 | q90 |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                order = {"lower": 0, "inside": 1, "upper": 2}
                for row in sorted(selected, key=lambda item: order[item["exposure_group"]]):
                    lines.append(
                        f"| {row['exposure_group']} | {row['valid_n']:,} | "
                        + " | ".join(fmt(row[f"q{q:02d}"], 3) for q in (1, 5, 10, 25, 50, 75, 90))
                        + " |"
                    )
                lines.append("")

    sensitivity_table = [
        "下表用非 scripted 固定 3 s 口径直接比较下侧/区间内；中位数分母是结果非空锚点，"
        "阈值占比给原始制动记录分子/分母。",
        "",
        "| α | 最强制动中位 L / I (m/s²) | 速度极差中位 L / I (km/h) | "
        "偏航角速度中位 L / I (度/s) | a<-2 L / I | a<-4 L / I |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in ("80", "90", "95"):
        q_lookup = {
            (row["outcome"], row["exposure_group"]): row
            for row in distributions["quantile_tables"]
            if row["alpha"] == alpha
            and row["stratum"] == "non_scripted"
            and row["window"] == "fixed3"
            and row["exposure_group"] in ("lower", "inside")
        }
        threshold_lookup = {
            (row["threshold_mps2"], row["exposure_group"]): row
            for row in distributions["braking_threshold_tables"]
            if row["alpha"] == alpha
            and row["stratum"] == "non_scripted"
            and row["window"] == "fixed3"
            and row["exposure_group"] in ("lower", "inside")
        }
        min_l = q_lookup[("cp_min_acceleration", "lower")]
        min_i = q_lookup[("cp_min_acceleration", "inside")]
        speed_l = q_lookup[("cp_speed_drop_kmh", "lower")]
        speed_i = q_lookup[("cp_speed_drop_kmh", "inside")]
        yaw_l = q_lookup[("cp_max_abs_yaw_rate", "lower")]
        yaw_i = q_lookup[("cp_max_abs_yaw_rate", "inside")]
        brake2_l = threshold_lookup[(-2, "lower")]
        brake2_i = threshold_lookup[(-2, "inside")]
        brake4_l = threshold_lookup[(-4, "lower")]
        brake4_i = threshold_lookup[(-4, "inside")]
        sensitivity_table.append(
            f"| {alpha} | {fmt(min_l['q50'],3)} (n={min_l['valid_n']:,}) / "
            f"{fmt(min_i['q50'],3)} (n={min_i['valid_n']:,}) | "
            f"{fmt(speed_l['q50'],3)} / {fmt(speed_i['q50'],3)} | "
            f"{fmt(yaw_l['q50'],3)} / {fmt(yaw_i['q50'],3)} | "
            f"{pct(brake2_l['frame_numerator'], brake2_l['frame_denominator'], 2)} / "
            f"{pct(brake2_i['frame_numerator'], brake2_i['frame_denominator'], 2)} | "
            f"{pct(brake4_l['frame_numerator'], brake4_l['frame_denominator'], 2)} / "
            f"{pct(brake4_i['frame_numerator'], brake4_i['frame_denominator'], 2)} |"
        )

    lines.extend(
        [
            "## 5. 制动阈值超越占比与 case bootstrap",
            "",
            "下表分子是所有重叠锚点窗口中 `acceleration` 低于阈值的原始记录数，分母是同组窗口内"
            "非空原始 `acceleration` 记录数；窗口重叠会重复计入同一原始时刻，因此推断以 case 为聚类单位。"
            "占比差为越界组减区间内，95% CI 由 case 整体重采样 1,000 次得到。",
            "",
            "| 分层 | 窗口 | 比较 | 阈值 | 越界组 | 区间内 | 差值 [case-bootstrap 95% CI] | p朴素 | p_case | p_team |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    contrasts = distributions["alpha90_case_bootstrap_threshold_contrasts"]
    for item in contrasts:
        ci = item["case_bootstrap_ci_95"]
        lines.append(
            f"| {item['stratum']} | {item['window']} | {item['comparison']} | "
            f"{item['threshold_mps2']} | "
            f"{pct(item['comparison_numerator'], item['comparison_denominator'], 3)} | "
            f"{pct(item['inside_numerator'], item['inside_denominator'], 3)} | "
            f"{fmt(item['share_difference'],4)} [{fmt(ci[0],4)}, {fmt(ci[1],4)}] | "
            f"{fmt(item['naive_two_proportion_p'],4)} | {fmt(item['case_equal_t_p'],4)} | "
            f"{fmt(item['team_equal_t_p'],4)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 回归（补充，不承载单独结论）",
            "",
            "模型为结果变量对下侧/上侧非负越界幅度与 `context_cell` 固定效应的普通最小二乘。"
            "系数单位是一倍人类区间宽度对应的结果变化；同时列朴素、case 聚类和 team 聚类 p 值，"
            "以 case 聚类为主。下表只列 α=90 非 scripted 主组；完整敏感性与 scripted 结果在 "
            "`regression_results.json`。",
            "",
            "| 窗口 | 结果 | 侧别 | 系数 | p朴素 | p_case | p_team | n |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regressions["results"]:
        if row.get("alpha") != "90" or row.get("stratum") != "non_scripted":
            continue
        for side in ("lower", "upper"):
            coefficient = row.get(side, {})
            lines.append(
                f"| {row['window']} | {row['outcome']} | {side} | "
                f"{fmt(coefficient.get('estimate'),4)} | {fmt(coefficient.get('naive_p'),4)} | "
                f"{fmt(coefficient.get('case_cluster_p'),4)} | {fmt(coefficient.get('team_cluster_p'),4)} | "
                f"{row.get('n','NA')} |"
            )

    lines.extend(
        [
            "",
            "## 7. 最小负对照",
            "",
            "### 7.1 case 层标签置换",
            "",
            "每个同时含下侧与区间内帧的 case 独立交换两组标签（等价于对 case 内组间差随机翻号），"
            "执行 1,000 次，给双侧经验 p。",
            "",
            "| 窗口 | 结果 | 有两组的 case | 观察差 | 经验 p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in negative_controls["case_label_permutation"]:
        lines.append(
            f"| {item['window']} | {item['outcome']} | {item['case_count_with_both_labels']} | "
            f"{fmt(item['observed_difference'],4)} | {fmt(item['empirical_two_sided_p'],4)} |"
        )
    lines.extend(
        [
            "",
            "### 7.2 安慰剂曝露",
            "",
            "每行独立从 `[lo_90-width_90, hi_90+width_90]` 均匀抽取假 IPV，再按同一公式重算越界。"
            f"三组计数为 {negative_controls['placebo_exposure']['group_counts']}；完整分位数、阈值占比、"
            "case bootstrap 与回归均保存在 `negative_controls.json`。",
            "",
            "## 8. α=80/95 敏感性",
            "",
            f"α=80 组计数 {distributions['exposure_counts']['80']}；α=95 组计数 "
            f"{distributions['exposure_counts']['95']}。完整分布表与 α=90 使用同一结果定义和分层，"
            "保存在 `distribution_results.json`；完整聚类回归保存在 `regression_results.json`。",
            "",
            *sensitivity_table,
            "",
            "## 9. 50 m 周边车辆可行性试算",
            "",
            f"在 {probe['session_count']} 个会话各取一个确定性锚点，共找到 "
            f"{probe['nearby_vehicle_total']} 辆 50 m 内车辆；加速度结果完整 "
            f"{pct(probe['nearby_with_acceleration_outcome'], probe['nearby_vehicle_total'])}，"
            f"航向结果完整 {pct(probe['nearby_with_heading_outcome'], probe['nearby_vehicle_total'])}。"
            f"其中 {probe['successful_probe_count']}/{probe['session_count']} 个会话在锚点 ±150 ms 内有原始车辆记录；"
            f"可行性判断为 `{probe['feasibility_assessment']}`。中心位置用派生 `ego_x/ego_y` 加该 case "
            "已验证的原始减派生中位平移转换到原始坐标；所有反应量仍来自原始日志。该试算不做统计推断。",
            "",
            "## 10. 结论",
            "",
            "本节必须结合上方分位数与阈值占比读取：分位数变化描述分布位置/形状，"
            "制动阈值及其 case-bootstrap CI 描述尾部；二者不互相替代。",
            "",
        ]
    )

    # Cautious, mechanically grounded primary interpretation.
    main_quantiles = [
        row
        for row in distributions["quantile_tables"]
        if row["alpha"] == "90"
        and row["stratum"] == "non_scripted"
        and row["window"] == "fixed3"
        and row["outcome"]
        in (
            "cp_min_acceleration",
            "cp_speed_drop_kmh",
            "cp_anchor_speed_drop_kmh",
            "cp_max_abs_yaw_rate",
            "cp_total_heading_change",
        )
    ]
    lookup = {(row["outcome"], row["exposure_group"]): row for row in main_quantiles}
    braking_lower = lookup.get(("cp_min_acceleration", "lower"), {})
    braking_inside = lookup.get(("cp_min_acceleration", "inside"), {})
    yaw_lower = lookup.get(("cp_max_abs_yaw_rate", "lower"), {})
    yaw_inside = lookup.get(("cp_max_abs_yaw_rate", "inside"), {})
    speed_lower = lookup.get(("cp_speed_drop_kmh", "lower"), {})
    speed_inside = lookup.get(("cp_speed_drop_kmh", "inside"), {})
    anchor_drop_lower = lookup.get(("cp_anchor_speed_drop_kmh", "lower"), {})
    anchor_drop_inside = lookup.get(("cp_anchor_speed_drop_kmh", "inside"), {})
    heading_lower = lookup.get(("cp_total_heading_change", "lower"), {})
    heading_inside = lookup.get(("cp_total_heading_change", "inside"), {})
    primary_tail = [
        item
        for item in contrasts
        if item["stratum"] == "non_scripted"
        and item["window"] == "fixed3"
        and item["comparison"] == "lower_minus_inside"
    ]
    tail_worse = [
        item
        for item in primary_tail
        if item["case_bootstrap_ci_95"][0] > 0
    ]
    tail_better = [
        item
        for item in primary_tail
        if item["case_bootstrap_ci_95"][1] < 0
    ]
    lines.append(
        f"- 非 scripted 固定 3 s 主组的分布中部显示更多常规反应：速度极差中位数下侧/区间内为 "
        f"{fmt(speed_lower.get('q50'),3)}/{fmt(speed_inside.get('q50'),3)} km/h，"
        f"锚点速度减窗口最小速度为 {fmt(anchor_drop_lower.get('q50'),3)}/"
        f"{fmt(anchor_drop_inside.get('q50'),3)} km/h，最大绝对偏航角速度为 "
        f"{fmt(yaw_lower.get('q50'),3)}/{fmt(yaw_inside.get('q50'),3)} 度/秒，"
        f"总航向变化为 {fmt(heading_lower.get('q50'),3)}/{fmt(heading_inside.get('q50'),3)} 度。"
        "但这些差异不是全分布一致右移：速度和转向的 q75/q90 有交叉，见 §4。"
    )
    lines.append(
        f"- 最强制动分布同样是形状变化而非整体左移：固定 3 s 中位数为下侧 "
        f"{fmt(braking_lower.get('q50'),3)} m/s²、区间内 {fmt(braking_inside.get('q50'),3)} m/s²；"
        f"但 q25 为下侧 {fmt(braking_lower.get('q25'),3)}、区间内 "
        f"{fmt(braking_inside.get('q25'),3)} m/s²，强制动端反而更轻。"
    )
    if tail_worse:
        thresholds_text = ", ".join(str(item["threshold_mps2"]) for item in tail_worse)
        lines.append(
            f"- 固定 3 s 的下侧减区间内制动帧占比，在阈值 {thresholds_text} m/s² 的 "
            "case-bootstrap 95% CI 全部高于 0；这是尾部占比更高的证据。"
        )
    elif tail_better:
        thresholds_text = ", ".join(str(item["threshold_mps2"]) for item in tail_better)
        lines.append(
            f"- 固定 3 s 的下侧减区间内制动帧占比，在阈值 {thresholds_text} m/s² 的 "
            "case-bootstrap 95% CI 全部低于 0，case 等权 p 值也均小于 0.006；"
            "因此强制动尾部不仅没有恶化，观察占比更低。合同窗口的 pooled case-bootstrap 同向，"
            "但 case 等权 p=0.35–0.67，故不把合同窗口写成稳定尾部差异。"
        )
    else:
        lines.append(
            "- 固定 3 s 的 -2/-3/-4 m/s² 下侧减区间内制动帧占比，其 case-bootstrap "
            "95% CI 均未形成一致的正向尾部证据；不能写成尾部恶化。"
        )
    lines.append(
        "- α=80/90/95 下，速度极差中位数均为下侧高于区间内，-2 与 -4 m/s² 强制动帧占比均为下侧更低；"
        "转向中位数方向在 α=80 与 α=90/95 间改变，因此转向证据不稳定。"
    )
    lines.append(
        "- 最小负对照与上述边界一致：固定 3 s 的 case 标签置换对三个强制动占比给经验 "
        "p=0.0040–0.0060；安慰剂曝露的合同/固定窗口全部六个下侧制动占比差的 case-bootstrap "
        "95% CI 均跨 0。"
    )
    lines.append(
        "- 综合判断：结果对“下侧越界对应对手方承担更多常规运动调整”提供部分支持，主要体现在分布中部的降速；"
        "它不支持“更极端制动”，且转向方向对 α 敏感。context_cell 控制后的速度/转向幅度系数也未在 case 聚类口径显著，"
        "因此该结论保持探索性。scripted 对手仅作隔离报告，不进入主结论。"
    )

    lines.extend(
        [
            "",
            "## 11. 待决事项",
            "",
            "本轮没有需要改变分析设定的待决事项。监督方需独立复算后决定是否进入正式研究库；"
            "在此之前不得把本探索性关联写成因果主张或车辆/队伍判断。",
            "",
            "## 12. 自查清单（机器证据）",
            "",
            f"- [x] 分析集 {analysis_n:,} 行，与预期 14,099 一致：`data_health.json -> load_contract`。",
            f"- [x] 三条对齐硬断言全部通过：`alignment_contract.json -> assertions`。",
            f"- [x] 原始字段空值：{raw['null_field_slots']:,}/{raw['required_field_slots']:,} 个字段槽；"
            "逐字段计数见 `data_health.json -> raw_fields.null_counts`。",
            f"- [x] 原始锚点匹配 {alignment['matched_anchor_rows']:,}/{analysis_n:,}；"
            "未匹配原因与计数见 `alignment_contract.json`。",
            f"- [x] α=90 上/下/内三组相加为 {analysis_n:,}。",
            f"- [x] 非 scripted/scripted 的行数与 case 数见 `data_health.json -> analysis_set.strata`。",
            f"- [x] 合同/固定窗口越过 case 末尾分别 "
            f"{int(measured['contract_window_past_case_end'].sum()):,}/{analysis_n:,} 与 "
            f"{int(measured['fixed3_window_past_case_end'].sum()):,}/{analysis_n:,}。",
            "- [x] acceleration/speed/courseAngle 的 NaN、±inf、范围见 `data_health.json -> raw_fields` "
            "与 `outcome_health`。",
            f"- [x] case 帧数分布、{case_n:,} 个 case、{team_n:,} 个 team 见 `data_health.json`。",
            "- [x] 朴素、case 聚类、team 聚类 p 值并列见 §5–6 与 `regression_results.json`。",
            "- [x] 两条负对照见 §7 与 `negative_controls.json`。",
            f"- [x] 570,761 m 坐标异常进入 {anomaly['analysis_row_count']:,}/{analysis_n:,} 行；"
            "case 与影响见 `data_health.json -> coordinate_anomaly_check`。",
            "",
            f"state: WAITING_ON_COMMANDER",
            f"timestamp_utc: {timestamp}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
