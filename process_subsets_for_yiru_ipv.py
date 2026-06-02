"""Key-agent IPV estimation for ``subsets_for_yiru`` pkl data."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
RAW_ROOT = THIS_DIR / "interhub_traj_lane" / "0_raw_data" / "subsets_for_yiru"
DEFAULT_CSV_PATH = RAW_ROOT / "selected_interactive_segments_equalized.csv"
DEFAULT_PKL_ROOT = RAW_ROOT / "pkl"
DEFAULT_OUTPUT_ROOT = (
    THIS_DIR / "interhub_traj_lane" / "1_ipv_estimation_results" / "subsets_for_yiru"
)

HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
HEADING_THRESHOLD_DEG = 12.0
WORKER_CANDIDATES = [1, 2, 4, 8, 12, 16]
BENCHMARK_SAMPLE_SIZE = 48
CSV_OUTPUT_COLUMNS = [
    "ipv_key_agent_1_mean",
    "ipv_key_agent_1_error_mean",
    "ipv_key_agent_2_mean",
    "ipv_key_agent_2_error_mean",
    "ipv_result_status",
    "ipv_result_case_dir",
    "ipv_result_error",
    "ipv_pkl_file",
    "ipv_segment_id",
    "ipv_reference_source_1",
    "ipv_reference_source_2",
]

_PKL_CACHE: Dict[Path, Mapping[str, object]] = {}


@dataclass(frozen=True)
class EventRef:
    """Location of one interaction event inside a pkl file."""

    pkl_path: Path
    segment_id: str


@dataclass(frozen=True)
class AlignedMotion:
    """Common-timestamp motion arrays for the two key agents."""

    primary_motion: np.ndarray
    secondary_motion: np.ndarray
    timestamps: List[object]


@dataclass(frozen=True)
class CaseTask:
    """Serializable case-processing task."""

    row_index: int
    row: Dict[str, object]
    event_ref: EventRef
    output_root: Path
    csv_path: Path
    pkl_root: Path
    history_window: int
    min_observation: int


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_name(value: object, max_len: int = 140) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (safe or "case")[:max_len]


def _dedupe_preserve_order(values: Iterable[object]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value)
        if text and text != "nan" and text not in result:
            result.append(text)
    return result


def _drop_consecutive_duplicate_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [True]
    diffs = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    keep.extend(bool(delta > 1e-9) for delta in diffs)
    cleaned = points[np.asarray(keep, dtype=bool), :2]
    return cleaned if len(cleaned) >= 2 else points[:, :2]


def _as_xy_array(value: object) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return _drop_consecutive_duplicate_points(arr[:, :2])


def _load_pickle(path: Path) -> Mapping[str, object]:
    path = Path(path)
    cached = _PKL_CACHE.get(path)
    if cached is not None:
        return cached
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected dict-like pkl content in {path}")
    _PKL_CACHE[path] = data
    return data


def parse_key_agents(value: object) -> List[str]:
    agents = [part.strip() for part in str(value).split(";") if part.strip()]
    if len(agents) != 2:
        raise ValueError(f"Expected exactly two key_agents, got {value!r}")
    return agents


def event_key_from_metadata(metadata: Mapping[str, object]) -> Tuple[str, str, str, str]:
    folder = str(metadata.get("folder") or metadata.get("subdata"))
    scenario_idx = str(metadata.get("scenario_idx"))
    key_agents = str(metadata.get("key_agents"))
    track_ids = metadata.get("track_ids") or []
    track_id = ";".join(str(item) for item in track_ids)
    return folder, scenario_idx, key_agents, track_id


def csv_key(row: Mapping[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row["folder"]),
        str(row["scenario_idx"]),
        str(row["key_agents"]),
        str(row["track_id"]),
    )


def build_event_index(pkl_root: Path) -> Dict[Tuple[str, str, str, str], EventRef]:
    """Index pkl events by ``folder, scenario_idx, key_agents, track_id``."""
    index: Dict[Tuple[str, str, str, str], EventRef] = {}
    duplicates: List[Tuple[str, str, str, str]] = []
    for pkl_path in sorted(Path(pkl_root).rglob("*.pkl")):
        data = _load_pickle(pkl_path)
        for segment_id, event in data.items():
            if not isinstance(event, Mapping):
                continue
            metadata = event.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            key = event_key_from_metadata(metadata)
            if key in index:
                duplicates.append(key)
            index[key] = EventRef(pkl_path=Path(pkl_path), segment_id=str(segment_id))
    if duplicates:
        raise ValueError(f"Duplicate event index keys detected: {duplicates[:5]}")
    return index


def load_event(event_ref: EventRef) -> Mapping[str, object]:
    data = _load_pickle(event_ref.pkl_path)
    event = data[event_ref.segment_id]
    if not isinstance(event, Mapping):
        raise TypeError(f"Event {event_ref.segment_id} is not dict-like")
    return event


def build_vehicle_reference(event: Mapping[str, object], vehicle_id: str) -> Tuple[np.ndarray, str]:
    """Build vehicle reference from lane centerlines, with observed trajectory fallback."""
    vehicles = event.get("vehicles", {})
    if not isinstance(vehicles, Mapping) or vehicle_id not in vehicles:
        raise KeyError(f"Vehicle {vehicle_id!r} not found")
    vehicle = vehicles[vehicle_id]
    if not isinstance(vehicle, Mapping):
        raise TypeError(f"Vehicle {vehicle_id!r} is not dict-like")

    road_info = event.get("road_info", {})
    lane_map = {}
    if isinstance(road_info, Mapping):
        raw_lane_map = road_info.get("all_lane_centerlines", {})
        if isinstance(raw_lane_map, Mapping):
            lane_map = {str(key): value for key, value in raw_lane_map.items()}

    for source_name in ("lane_ids", "frame_lane_ids"):
        lane_ids = _dedupe_preserve_order(vehicle.get(source_name, []))
        pieces: List[np.ndarray] = []
        for lane_id in lane_ids:
            lane_ref = _as_xy_array(lane_map.get(lane_id))
            if lane_ref is not None:
                pieces.append(lane_ref)
        if pieces:
            return _drop_consecutive_duplicate_points(np.vstack(pieces)), source_name

    observed = _as_xy_array(vehicle.get("positions"))
    if observed is None:
        raise ValueError(f"No usable reference or observed trajectory for {vehicle_id!r}")
    return observed, "observed_trajectory_fallback"


def _motion_arrays(vehicle: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[object]]:
    positions = np.asarray(vehicle.get("positions"), dtype=float)
    velocities = np.asarray(vehicle.get("velocities"), dtype=float)
    headings = np.asarray(vehicle.get("headings"), dtype=float)
    timestamps = list(vehicle.get("timestamps") or [])
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must be a 2D array with at least two columns")
    if velocities.ndim != 2 or velocities.shape[1] < 2:
        raise ValueError("velocities must be a 2D array with at least two columns")
    min_len = min(len(positions), len(velocities), len(headings), len(timestamps))
    if min_len == 0:
        raise ValueError("empty motion arrays")
    return positions[:min_len, :2], velocities[:min_len, :2], headings[:min_len], timestamps[:min_len]


def _motion_for_timestamps(vehicle: Mapping[str, object], timestamps: Sequence[object]) -> np.ndarray:
    positions, velocities, headings, vehicle_timestamps = _motion_arrays(vehicle)
    timestamp_to_index = {timestamp: idx for idx, timestamp in enumerate(vehicle_timestamps)}
    indices = [timestamp_to_index[timestamp] for timestamp in timestamps]
    return np.column_stack(
        (
            positions[indices, :],
            velocities[indices, :],
            headings[indices],
        )
    )


def align_key_agent_motion(
    event: Mapping[str, object],
    key_agents: Sequence[str],
    *,
    min_steps: int = MIN_OBSERVATION + 1,
) -> AlignedMotion:
    """Align two key-agent motion sequences on common timestamps."""
    vehicles = event.get("vehicles", {})
    if not isinstance(vehicles, Mapping):
        raise TypeError("event['vehicles'] must be dict-like")
    first_id, second_id = key_agents
    if first_id not in vehicles or second_id not in vehicles:
        raise KeyError(f"Missing one or more key agents: {key_agents}")

    _, _, _, first_timestamps = _motion_arrays(vehicles[first_id])
    _, _, _, second_timestamps = _motion_arrays(vehicles[second_id])
    common = sorted(set(first_timestamps).intersection(second_timestamps))
    if len(common) < min_steps:
        raise ValueError(f"Only {len(common)} common timestamps; need at least {min_steps}")

    return AlignedMotion(
        primary_motion=_motion_for_timestamps(vehicles[first_id], common),
        secondary_motion=_motion_for_timestamps(vehicles[second_id], common),
        timestamps=list(common),
    )


def classify_heading(headings: np.ndarray, threshold_deg: float = HEADING_THRESHOLD_DEG) -> str:
    unwrapped = np.unwrap(np.asarray(headings, dtype=float))
    delta_deg = float(np.degrees(unwrapped[-1] - unwrapped[0]))
    if delta_deg >= threshold_deg:
        return "lt"
    if delta_deg <= -threshold_deg:
        return "rt"
    return "gs"


def ensure_unique_labels(labels: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    result: List[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label}{counts[label]}")
    return result


def compute_valid_ipv_summary(
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
    *,
    min_observation: int = MIN_OBSERVATION,
) -> Dict[str, float]:
    valid_ipv = np.asarray(ipv_values, dtype=float)[min_observation:, :]
    valid_err = np.asarray(ipv_errors, dtype=float)[min_observation:, :]
    if valid_ipv.size == 0:
        raise ValueError("No valid IPV rows available for summary")
    return {
        "ipv_key_agent_1_mean": round(float(np.nanmean(valid_ipv[:, 0])), 12),
        "ipv_key_agent_1_error_mean": round(float(np.nanmean(valid_err[:, 0])), 12),
        "ipv_key_agent_2_mean": round(float(np.nanmean(valid_ipv[:, 1])), 12),
        "ipv_key_agent_2_error_mean": round(float(np.nanmean(valid_err[:, 1])), 12),
    }


def build_csv_copy(source_df: pd.DataFrame, results: Mapping[int, Mapping[str, object]]) -> pd.DataFrame:
    """Return a CSV copy with IPV result columns and no duplicated key-agent id columns."""
    output = source_df.copy()
    for column in CSV_OUTPUT_COLUMNS:
        output[column] = ""
    for row_index, result in results.items():
        for column in CSV_OUTPUT_COLUMNS:
            if column in result:
                output.loc[row_index, column] = result[column]
    return output


def choose_recommended_workers(
    records: Sequence[Mapping[str, object]],
    *,
    min_gain: float = 0.10,
) -> int:
    """Choose the fastest stable worker count, preferring lower counts on tiny gains."""
    successful = sorted(
        (
            record
            for record in records
            if int(record.get("failed", 0)) == 0
            and int(record.get("processed", 1)) > 0
            and float(record.get("cases_per_minute", 0.0)) > 0
        ),
        key=lambda item: int(item["workers"]),
    )
    if not successful:
        return 1
    chosen = successful[0]
    for candidate in successful[1:]:
        chosen_rate = float(chosen["cases_per_minute"])
        candidate_rate = float(candidate["cases_per_minute"])
        if candidate_rate <= chosen_rate:
            continue
        gain = math.inf if chosen_rate == 0 else (candidate_rate - chosen_rate) / chosen_rate
        if gain >= min_gain:
            chosen = candidate
    return int(chosen["workers"])


def _save_ipv_table(
    output_path: Path,
    timestamps: Sequence[object],
    key_agents: Sequence[str],
    primary_motion: np.ndarray,
    secondary_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "timestamp": list(timestamps),
            "key_agent_1": key_agents[0],
            "key_agent_2": key_agents[1],
            "ipv_key_agent_1": ipv_values[:, 0],
            "ipv_key_agent_1_error": ipv_errors[:, 0],
            "key_agent_1_px": primary_motion[:, 0],
            "key_agent_1_py": primary_motion[:, 1],
            "key_agent_1_vx": primary_motion[:, 2],
            "key_agent_1_vy": primary_motion[:, 3],
            "key_agent_1_heading": primary_motion[:, 4],
            "ipv_key_agent_2": ipv_values[:, 1],
            "ipv_key_agent_2_error": ipv_errors[:, 1],
            "key_agent_2_px": secondary_motion[:, 0],
            "key_agent_2_py": secondary_motion[:, 1],
            "key_agent_2_vx": secondary_motion[:, 2],
            "key_agent_2_vy": secondary_motion[:, 3],
            "key_agent_2_heading": secondary_motion[:, 4],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def _plot_case_summary(
    output_path: Path,
    key_agents: Sequence[str],
    primary_motion: np.ndarray,
    secondary_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
    primary_reference: np.ndarray,
    secondary_reference: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_ipv, ax_traj) = plt.subplots(1, 2, figsize=(16, 8))
    time_idx = np.arange(len(ipv_values))
    ax_ipv.set_ylim([-2, 2])
    ax_ipv.plot(time_idx, ipv_values[:, 0], color="#df7565", label=f"{key_agents[0]} IPV")
    ax_ipv.fill_between(
        time_idx,
        ipv_values[:, 0] - ipv_errors[:, 0],
        ipv_values[:, 0] + ipv_errors[:, 0],
        color="#df7565",
        alpha=0.25,
    )
    ax_ipv.plot(time_idx, ipv_values[:, 1], color="#2FAECE", label=f"{key_agents[1]} IPV")
    ax_ipv.fill_between(
        time_idx,
        ipv_values[:, 1] - ipv_errors[:, 1],
        ipv_values[:, 1] + ipv_errors[:, 1],
        color="#2FAECE",
        alpha=0.25,
    )
    ax_ipv.set_xlabel("time index")
    ax_ipv.set_ylabel("IPV")
    ax_ipv.legend()

    ax_traj.plot(primary_reference[:, 0], primary_reference[:, 1], "--", color="#df7565", alpha=0.35)
    ax_traj.plot(secondary_reference[:, 0], secondary_reference[:, 1], "--", color="#2FAECE", alpha=0.35)
    ax_traj.plot(primary_motion[:, 0], primary_motion[:, 1], color="#df7565", label=key_agents[0])
    ax_traj.plot(secondary_motion[:, 0], secondary_motion[:, 1], color="#2FAECE", label=key_agents[1])
    ax_traj.axis("equal")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def case_output_dir(output_root: Path, row_index: int, row: Mapping[str, object], segment_id: str) -> Path:
    dataset = _safe_name(row.get("dataset", "dataset"))
    folder = _safe_name(row.get("folder", "folder"))
    scenario = _safe_name(row.get("scenario_idx", "scenario"))
    segment_hash = hashlib.sha1(str(segment_id).encode("utf-8")).hexdigest()[:12]
    case = f"row_{row_index:05d}_{segment_hash}"
    return Path(output_root) / "cases" / dataset / folder / f"scenario_{scenario}" / case


def process_case(task: CaseTask) -> Dict[str, object]:
    from ipv_estimation import MotionSequence, estimate_ipv_pair

    row = task.row
    case_dir = case_output_dir(task.output_root, task.row_index, row, task.event_ref.segment_id)
    base_result: Dict[str, object] = {
        "ipv_result_status": "failed",
        "ipv_result_case_dir": str(case_dir),
        "ipv_result_error": "",
        "ipv_pkl_file": task.event_ref.pkl_path.relative_to(task.pkl_root).as_posix(),
        "ipv_segment_id": task.event_ref.segment_id,
        "ipv_reference_source_1": "",
        "ipv_reference_source_2": "",
    }
    try:
        event = load_event(task.event_ref)
        key_agents = parse_key_agents(row["key_agents"])
        aligned = align_key_agent_motion(
            event,
            key_agents,
            min_steps=task.min_observation + 1,
        )
        primary_ref, primary_ref_source = build_vehicle_reference(event, key_agents[0])
        secondary_ref, secondary_ref_source = build_vehicle_reference(event, key_agents[1])
        base_result["ipv_reference_source_1"] = primary_ref_source
        base_result["ipv_reference_source_2"] = secondary_ref_source

        labels = ensure_unique_labels(
            [
                classify_heading(aligned.primary_motion[:, 4]),
                classify_heading(aligned.secondary_motion[:, 4]),
            ]
        )
        seq_primary = MotionSequence(aligned.primary_motion, target=labels[0], reference=primary_ref)
        seq_secondary = MotionSequence(aligned.secondary_motion, target=labels[1], reference=secondary_ref)
        ipv_values, ipv_errors = estimate_ipv_pair(
            seq_primary,
            seq_secondary,
            history_window=task.history_window,
            min_observation=task.min_observation,
        )
        summary = compute_valid_ipv_summary(
            ipv_values,
            ipv_errors,
            min_observation=task.min_observation,
        )

        data_dir = case_dir / "data"
        fig_dir = case_dir / "fig"
        _save_ipv_table(
            data_dir / "ipv_results.xlsx",
            aligned.timestamps,
            key_agents,
            aligned.primary_motion,
            aligned.secondary_motion,
            ipv_values,
            ipv_errors,
        )
        _plot_case_summary(
            fig_dir / "ipv_curve.png",
            key_agents,
            aligned.primary_motion,
            aligned.secondary_motion,
            ipv_values,
            ipv_errors,
            primary_ref,
            secondary_ref,
        )
        metadata = {
            "row_index": task.row_index,
            "csv_path": str(task.csv_path),
            "pkl_file": base_result["ipv_pkl_file"],
            "segment_id": task.event_ref.segment_id,
            "dataset": row.get("dataset"),
            "folder": row.get("folder"),
            "scenario_idx": row.get("scenario_idx"),
            "track_id": row.get("track_id"),
            "key_agents": row.get("key_agents"),
            "two_multi": row.get("two/multi"),
            "reference_source_1": primary_ref_source,
            "reference_source_2": secondary_ref_source,
            "history_window": task.history_window,
            "min_observation": task.min_observation,
            "timestamps": aligned.timestamps,
            "summary": summary,
            "status": "ok",
        }
        metadata_path = data_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        base_result.update(summary)
        base_result["ipv_result_status"] = "ok"
        return base_result
    except Exception as exc:  # pylint: disable=broad-except
        case_dir.mkdir(parents=True, exist_ok=True)
        base_result["ipv_result_error"] = f"{type(exc).__name__}: {exc}"
        failure_metadata = {
            "row_index": task.row_index,
            "row": row,
            "pkl_file": base_result["ipv_pkl_file"],
            "segment_id": task.event_ref.segment_id,
            "status": "failed",
            "error": base_result["ipv_result_error"],
        }
        (case_dir / "failure_metadata.json").write_text(
            json.dumps(failure_metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return base_result


def _build_tasks(
    df: pd.DataFrame,
    event_index: Mapping[Tuple[str, str, str, str], EventRef],
    *,
    csv_path: Path,
    pkl_root: Path,
    output_root: Path,
    history_window: int,
    min_observation: int,
    limit: Optional[int] = None,
) -> Tuple[List[CaseTask], Dict[int, Dict[str, object]]]:
    tasks: List[CaseTask] = []
    initial_results: Dict[int, Dict[str, object]] = {}
    rows = df if limit is None else df.head(limit)
    for row_index, row in rows.iterrows():
        row_dict = row.to_dict()
        key = csv_key(row_dict)
        event_ref = event_index.get(key)
        if event_ref is None:
            initial_results[row_index] = {
                "ipv_result_status": "missing_pkl_event",
                "ipv_result_error": f"No pkl event for key={key}",
            }
            continue
        tasks.append(
            CaseTask(
                row_index=int(row_index),
                row=row_dict,
                event_ref=event_ref,
                output_root=Path(output_root),
                csv_path=Path(csv_path),
                pkl_root=Path(pkl_root),
                history_window=history_window,
                min_observation=min_observation,
            )
        )
    return tasks, initial_results


def _run_tasks(tasks: Sequence[CaseTask], workers: int) -> Dict[int, Dict[str, object]]:
    results: Dict[int, Dict[str, object]] = {}
    if not tasks:
        return results
    workers = max(1, min(int(workers), len(tasks)))
    if workers == 1:
        for task in tqdm(tasks, desc="IPV cases", unit="case"):
            results[task.row_index] = process_case(task)
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(process_case, task): task.row_index for task in tasks}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="IPV cases", unit="case"):
            row_index = future_to_index[future]
            try:
                results[row_index] = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                results[row_index] = {
                    "ipv_result_status": "failed",
                    "ipv_result_error": f"{type(exc).__name__}: {exc}",
                }
    return results


def _summarize_reference_coverage(
    df: pd.DataFrame,
    event_index: Mapping[Tuple[str, str, str, str], EventRef],
) -> Dict[str, object]:
    status_counts: Dict[str, int] = {}
    by_dataset: Dict[str, int] = {}
    min_ref_len: Optional[int] = None
    max_ref_len: Optional[int] = None
    checked_agents = 0
    examples: List[Dict[str, object]] = []
    for row_index, row in df.iterrows():
        event_ref = event_index.get(csv_key(row.to_dict()))
        if event_ref is None:
            continue
        event = load_event(event_ref)
        dataset = str(row.get("dataset"))
        for vehicle_id in parse_key_agents(row["key_agents"]):
            try:
                ref, source = build_vehicle_reference(event, vehicle_id)
            except Exception:  # pylint: disable=broad-except
                source = "missing"
                ref = np.empty((0, 2))
            checked_agents += 1
            status_counts[source] = status_counts.get(source, 0) + 1
            by_dataset[f"{dataset}:{source}"] = by_dataset.get(f"{dataset}:{source}", 0) + 1
            if len(ref):
                min_ref_len = len(ref) if min_ref_len is None else min(min_ref_len, len(ref))
                max_ref_len = len(ref) if max_ref_len is None else max(max_ref_len, len(ref))
            if len(examples) < 5:
                examples.append(
                    {
                        "row_index": int(row_index),
                        "vehicle_id": vehicle_id,
                        "source": source,
                        "reference_len": int(len(ref)),
                    }
                )
    return {
        "checked_key_agents": checked_agents,
        "reference_source_counts": status_counts,
        "reference_source_by_dataset": by_dataset,
        "reference_len_min": min_ref_len,
        "reference_len_max": max_ref_len,
        "examples": examples,
    }


def run_preflight(csv_path: Path, pkl_root: Path, output_root: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    event_index = build_event_index(pkl_root)
    matched = 0
    unmatched_examples: List[Dict[str, object]] = []
    for row_index, row in df.iterrows():
        key = csv_key(row.to_dict())
        if key in event_index:
            matched += 1
        elif len(unmatched_examples) < 10:
            unmatched_examples.append({"row_index": int(row_index), "key": key})
    reference_summary = _summarize_reference_coverage(df, event_index)
    summary = {
        "csv_path": str(csv_path),
        "pkl_root": str(pkl_root),
        "output_root": str(output_root),
        "csv_rows": int(len(df)),
        "pkl_files": [path.relative_to(pkl_root).as_posix() for path in sorted(Path(pkl_root).rglob("*.pkl"))],
        "pkl_events": int(len(event_index)),
        "matched_rows": int(matched),
        "unmatched_rows": int(len(df) - matched),
        "unmatched_examples": unmatched_examples,
        "dataset_counts": {str(k): int(v) for k, v in df["dataset"].value_counts(dropna=False).items()},
        "two_multi_counts": {str(k): int(v) for k, v in df["two/multi"].value_counts(dropna=False).items()},
        "reference": reference_summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _benchmark_sample_indices(df: pd.DataFrame, sample_size: int = BENCHMARK_SAMPLE_SIZE) -> List[int]:
    per_kind = sample_size // 2
    selected: List[int] = []
    for kind in ("multi", "two"):
        subset = df[df["two/multi"].astype(str) == kind]
        per_dataset = max(1, per_kind // max(1, df["dataset"].nunique()))
        kind_indices: List[int] = []
        for _, dataset_rows in subset.groupby("dataset", sort=True):
            kind_indices.extend(dataset_rows.head(per_dataset).index.tolist())
        if len(kind_indices) < per_kind:
            for idx in subset.index:
                if idx not in kind_indices:
                    kind_indices.append(idx)
                if len(kind_indices) >= per_kind:
                    break
        selected.extend(kind_indices[:per_kind])
    if len(selected) < sample_size:
        for idx in df.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= sample_size:
                break
    return selected[:sample_size]


def _worker_candidates(max_workers: int) -> List[int]:
    return [candidate for candidate in WORKER_CANDIDATES if candidate <= max_workers]


def run_worker_benchmark(
    csv_path: Path,
    pkl_root: Path,
    output_root: Path,
    *,
    history_window: int,
    min_observation: int,
    max_workers: int,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    event_index = build_event_index(pkl_root)
    sample_indices = _benchmark_sample_indices(df)
    sample_df = df.loc[sample_indices]
    records: List[Dict[str, object]] = []
    benchmark_root = output_root / "_benchmark_workers"
    for workers in _worker_candidates(max_workers):
        worker_root = benchmark_root / f"worker_{workers}"
        tasks, initial_results = _build_tasks(
            sample_df,
            event_index,
            csv_path=csv_path,
            pkl_root=pkl_root,
            output_root=worker_root,
            history_window=history_window,
            min_observation=min_observation,
        )
        start = time.perf_counter()
        results = dict(initial_results)
        results.update(_run_tasks(tasks, workers))
        elapsed = time.perf_counter() - start
        processed = sum(1 for result in results.values() if result.get("ipv_result_status") == "ok")
        failed = len(results) - processed
        records.append(
            {
                "workers": workers,
                "processed": processed,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 3),
                "cases_per_minute": round((processed / elapsed * 60) if elapsed else 0.0, 6),
                "failure_examples": [
                    result.get("ipv_result_error", "")
                    for result in results.values()
                    if result.get("ipv_result_status") != "ok"
                ][:5],
            }
        )
    recommended = choose_recommended_workers(records)
    benchmark = {
        "sample_indices": [int(idx) for idx in sample_indices],
        "candidate_workers": _worker_candidates(max_workers),
        "records": records,
        "recommended_workers": recommended,
    }
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "worker_benchmark.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(records).to_csv(benchmark_root / "worker_benchmark.csv", index=False)
    return benchmark


def _load_recommended_workers(output_root: Path) -> Optional[int]:
    benchmark_path = output_root / "_benchmark_workers" / "worker_benchmark.json"
    if not benchmark_path.exists():
        return None
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    workers = benchmark.get("recommended_workers")
    return int(workers) if workers else None


def _resolve_workers(
    workers_arg: str,
    *,
    csv_path: Path,
    pkl_root: Path,
    output_root: Path,
    history_window: int,
    min_observation: int,
    max_workers: int,
) -> int:
    if workers_arg != "auto":
        return max(1, min(int(workers_arg), max_workers))
    recommended = _load_recommended_workers(output_root)
    if recommended is not None:
        return max(1, min(recommended, max_workers))
    benchmark = run_worker_benchmark(
        csv_path,
        pkl_root,
        output_root,
        history_window=history_window,
        min_observation=min_observation,
        max_workers=max_workers,
    )
    return max(1, min(int(benchmark["recommended_workers"]), max_workers))


def run_processing(
    csv_path: Path,
    pkl_root: Path,
    output_root: Path,
    *,
    workers: int,
    history_window: int,
    min_observation: int,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    event_index = build_event_index(pkl_root)
    tasks, initial_results = _build_tasks(
        df,
        event_index,
        csv_path=csv_path,
        pkl_root=pkl_root,
        output_root=output_root,
        history_window=history_window,
        min_observation=min_observation,
        limit=limit,
    )
    results = dict(initial_results)
    results.update(_run_tasks(tasks, workers))
    csv_copy = build_csv_copy(df, results)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = "_limit" if limit is not None else ""
    csv_output = output_root / f"selected_interactive_segments_equalized_with_ipv{suffix}.csv"
    csv_copy.to_csv(csv_output, index=False, encoding="utf-8-sig")
    status_counts: Dict[str, int] = {}
    for result in results.values():
        status = str(result.get("ipv_result_status", "not_processed"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "csv_path": str(csv_path),
        "pkl_root": str(pkl_root),
        "output_root": str(output_root),
        "workers": workers,
        "limit": limit,
        "processed_task_count": len(tasks),
        "result_count": len(results),
        "status_counts": status_counts,
        "csv_output": str(csv_output),
    }
    (output_root / f"processing_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _append_workflow_log(summary: Mapping[str, object], *, task_name: str) -> None:
    log_path = THIS_DIR / "main_workflow.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status_counts = summary.get("status_counts", {})
    entry = (
        f"\n[{timestamp}] STATUS: COMPLETED | TASK: {task_name}.\n"
        "Outcome:\n"
        f"- Output root: {summary.get('output_root')}.\n"
        f"- Workers: {summary.get('workers')}.\n"
        f"- Limit: {summary.get('limit')}.\n"
        f"- Status counts: {status_counts}.\n"
        "Artifacts:\n"
        f"- {summary.get('csv_output', summary.get('output_root'))}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--pkl-root", type=Path, default=DEFAULT_PKL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-workers", type=int, default=max(1, math.floor((os.cpu_count() or 1) * 0.7)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--history-window", type=int, default=HISTORY_WINDOW)
    parser.add_argument("--min-observation", type=int, default=MIN_OBSERVATION)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--benchmark-workers", action="store_true")
    parser.add_argument("--log-workflow", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(args.csv, args.pkl_root, args.output_root)
    LOGGER.info(
        "Preflight: csv_rows=%s pkl_events=%s matched=%s unmatched=%s",
        preflight["csv_rows"],
        preflight["pkl_events"],
        preflight["matched_rows"],
        preflight["unmatched_rows"],
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False, default=_json_default))
        return

    if args.benchmark_workers:
        benchmark = run_worker_benchmark(
            args.csv,
            args.pkl_root,
            args.output_root,
            history_window=args.history_window,
            min_observation=args.min_observation,
            max_workers=args.max_workers,
        )
        print(json.dumps(benchmark, indent=2, ensure_ascii=False, default=_json_default))
        return

    workers = _resolve_workers(
        args.workers,
        csv_path=args.csv,
        pkl_root=args.pkl_root,
        output_root=args.output_root,
        history_window=args.history_window,
        min_observation=args.min_observation,
        max_workers=args.max_workers,
    )
    summary = run_processing(
        args.csv,
        args.pkl_root,
        args.output_root,
        workers=workers,
        history_window=args.history_window,
        min_observation=args.min_observation,
        limit=args.limit,
    )
    if args.log_workflow:
        _append_workflow_log(summary, task_name="Run subsets_for_yiru key-agent IPV processing")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
