"""
IPV estimation pipeline for the interhub trajectory datasets.

This module mirrors the Argoverse processing flow while accommodating the
structure of the custom JSON files found under ``interhub_traj_lane/``.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import Agent  # noqa: F401  # ensure agent modifications are applied
from ipv_estimation import (
    MotionSequence,
    estimate_ipv_pair,
    plot_virtual_vs_observed,
)
from argoverse.argoverse_process import plot_ipv_summary, save_ipv_table

LOGGER = logging.getLogger(__name__)

INTERHUB_ROOT = Path("interhub_traj_lane")
OUTPUT_ROOT = INTERHUB_ROOT / "ipv_estimation"

LANE_DISTANCE_THRESHOLD = 2.0
HEADING_THRESHOLD_DEG = 12.0
HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
DIAGNOSTIC_STEPS = (5, 6)


def _load_dataset(
    json_path: Path,
    lane_distance_threshold: float = LANE_DISTANCE_THRESHOLD,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Load the raw JSON file and regroup entries by scenario/vehicle.

    Returns:
        {"scenario_id": {"vehicle_id": normalized_vehicle_entry, ...}, ...}
    """
    LOGGER.info("Loading dataset %s", json_path.name)
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    skipped = 0

    for value in raw.values():
        reference = value.get("filtered_centerline_path") or value.get("centerline_path")
        if not reference:
            skipped += 1
            continue

        if lane_distance_threshold is not None:
            dist = value.get("trajectory_centerline_distance")
            if dist is not None and float(dist) > lane_distance_threshold:
                skipped += 1
                continue

        prepared = _prepare_vehicle_entry(value, reference)
        if prepared is None:
            skipped += 1
            continue

        scenario_id = prepared["scenario_id"]
        vehicle_id = prepared["vehicle_id"]
        scenarios.setdefault(scenario_id, {})[vehicle_id] = prepared

    LOGGER.info(
        "Loaded %d scenarios from %s (skipped %d entries)",
        len(scenarios),
        json_path.name,
        skipped,
    )
    return scenarios


def _prepare_vehicle_entry(
    entry: Dict[str, object],
    reference: Iterable[Iterable[float]],
) -> Optional[Dict[str, object]]:
    positions = np.asarray(entry.get("positions"), dtype=float)
    if positions.size == 0:
        return None
    if positions.ndim != 2 or positions.shape[0] < 2:
        return None
    if positions.shape[1] >= 2:
        positions = positions[:, :2]

    velocities = np.asarray(entry.get("velocities"), dtype=float)
    headings = np.asarray(entry.get("headings"), dtype=float)
    timestamps = np.asarray(entry.get("timestamps"), dtype=float)

    min_len = min(len(positions), len(velocities), len(headings), len(timestamps))
    if min_len < 2:
        return None

    positions = positions[:min_len]
    velocities = velocities[:min_len, :2]
    headings = headings[:min_len]
    timestamps = timestamps[:min_len]

    ref_array = np.asarray(reference, dtype=float)
    if ref_array.ndim != 2 or ref_array.shape[0] < 2:
        return None
    if ref_array.shape[1] >= 2:
        ref_array = ref_array[:, :2]

    scenario_id = str(entry.get("scenario_idx"))
    vehicle_id = str(entry.get("vehicle_id"))

    return {
        "scenario_id": scenario_id,
        "vehicle_id": vehicle_id,
        "is_av": vehicle_id.lower() == "ego",
        "timestamps": timestamps,
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "reference": ref_array,
        "dt": float(entry.get("dt", 0.1)),
        "start_time": float(entry.get("start_time", timestamps[0])),
        "end_time": float(entry.get("end_time", timestamps[-1])),
    }


def _select_vehicle_pair(
    vehicles: Dict[str, Dict[str, object]]
) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    avs = [v for v in vehicles.values() if v["is_av"]]
    hvs = [v for v in vehicles.values() if not v["is_av"]]

    if avs and hvs:
        return avs[0], hvs[0]
    if len(hvs) >= 2:
        return hvs[0], hvs[1]
    return None


def _classify_heading(headings: np.ndarray, threshold_deg: float = HEADING_THRESHOLD_DEG) -> str:
    headings = np.unwrap(headings.astype(float))
    delta = headings[-1] - headings[0]
    delta_deg = float(np.degrees(delta))
    if delta_deg >= threshold_deg:
        return "lt"
    if delta_deg <= -threshold_deg:
        return "rt"
    return "gs"


def _build_motion_sequence(vehicle: Dict[str, object], label: str) -> MotionSequence:
    positions = vehicle["positions"]
    velocities = vehicle["velocities"]
    headings = vehicle["headings"]

    data = np.column_stack((positions, velocities, headings))
    return MotionSequence(
        data=data,
        target=label,
        reference=vehicle["reference"],
    )


def _ensure_unique_labels(labels: List[str]) -> List[str]:
    counter: Dict[str, count] = {}
    result: List[str] = []
    for label in labels:
        counter.setdefault(label, count(1))
        idx = next(counter[label])
        if idx > 1:
            result.append(f"{label}{idx}")
        else:
            result.append(label)
    return result


def process_dataset(
    json_path: Path,
    output_root: Path = OUTPUT_ROOT,
    lane_distance_threshold: float = LANE_DISTANCE_THRESHOLD,
    heading_threshold_deg: float = HEADING_THRESHOLD_DEG,
    max_workers: Optional[int] = None,
) -> None:
    scenarios = _load_dataset(json_path, lane_distance_threshold)
    dataset_name = json_path.stem
    tasks: List[Tuple[str, str, Dict[str, object], Dict[str, object], List[str], Path]] = []
    skipped = 0

    for scenario_id, vehicles in scenarios.items():
        pair = _select_vehicle_pair(vehicles)
        if not pair:
            skipped += 1
            continue

        primary, secondary = pair
        labels = [
            _classify_heading(primary["headings"], heading_threshold_deg),
            _classify_heading(secondary["headings"], heading_threshold_deg),
        ]
        labels = _ensure_unique_labels(labels)
        tasks.append((dataset_name, scenario_id, primary, secondary, labels, output_root))

    if not tasks:
        LOGGER.info("Dataset %s completed: 0 processed, %d skipped", dataset_name, skipped)
        return

    workers = max_workers if max_workers not in (None, 0) else os.cpu_count()
    if workers is None or workers <= 0:
        workers = 1
    workers = min(workers, len(tasks))

    LOGGER.info(
        "Processing %d scenarios from %s with %d worker(s)",
        len(tasks),
        dataset_name,
        workers,
    )

    processed = 0
    failed = 0

    if workers == 1:
        iterator = tqdm(tasks, desc=f"{dataset_name}", unit="scenario")
        for task in iterator:
            try:
                _process_pair(
                    dataset_name=task[0],
                    scenario_id=task[1],
                    primary=task[2],
                    secondary=task[3],
                    labels=task[4],
                    output_root=task[5],
                )
                processed += 1
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception(
                    "Failed to process dataset=%s scenario=%s (%s)",
                    task[0],
                    task[1],
                    exc,
                )
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_task, task): task[1] for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_name}", unit="scenario"):
                scenario_id = futures[future]
                try:
                    future.result()
                    processed += 1
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.error(
                        "Failed to process dataset=%s scenario=%s (%s)",
                        dataset_name,
                        scenario_id,
                        exc,
                    )
                    failed += 1

    LOGGER.info(
        "Dataset %s completed: %d processed, %d failed, %d skipped",
        dataset_name,
        processed,
        failed,
        skipped,
    )


def _process_task(task: Tuple[str, str, Dict[str, object], Dict[str, object], List[str], Path]) -> None:
    dataset_name, scenario_id, primary, secondary, labels, output_root = task
    _process_pair(
        dataset_name=dataset_name,
        scenario_id=scenario_id,
        primary=primary,
        secondary=secondary,
        labels=labels,
        output_root=output_root,
    )


def _process_pair(
    *,
    dataset_name: str,
    scenario_id: str,
    primary: Dict[str, object],
    secondary: Dict[str, object],
    labels: List[str],
    output_root: Path,
) -> None:
    seq_primary = _build_motion_sequence(primary, labels[0])
    seq_secondary = _build_motion_sequence(secondary, labels[1])

    ipv_values, ipv_errors, diagnostics = estimate_ipv_pair(
        seq_primary,
        seq_secondary,
        history_window=HISTORY_WINDOW,
        min_observation=MIN_OBSERVATION,
        return_diagnostics=True,
        diagnostic_steps=DIAGNOSTIC_STEPS,
    )

    steps = ipv_values.shape[0]
    lt_motion = seq_primary.data[:steps]
    gs_motion = seq_secondary.data[:steps]

    case_name = f"{scenario_id}_{primary['vehicle_id']}_{secondary['vehicle_id']}"

    base_dir = output_root / dataset_name / f"scenario_{scenario_id}"
    data_dir = base_dir / "data"
    fig_dir = base_dir / "fig"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_ipv_table(
        data_dir / f"{case_name}_ipv_results.xlsx",
        lt_motion,
        gs_motion,
        ipv_values,
        ipv_errors,
    )

    plot_ipv_summary(
        fig_dir / f"{case_name}_ipv_curve.png",
        lt_motion,
        gs_motion,
        ipv_values,
        ipv_errors,
    )

    diag_dir = fig_dir / "virtual_tracks" / case_name
    if diagnostics:
        diag_dir.mkdir(parents=True, exist_ok=True)
        for role, entries in diagnostics.items():
            for entry in entries:
                ax = plot_virtual_vs_observed(
                    entry["observed"],
                    entry["virtual_tracks"],
                    interacting_track=entry["interacting"],
                    weights=entry["weights"],
                    title=(
                        f"{role} step {entry['step']} "
                        f"(ipv={entry['ipv']:.3f}, err={entry['ipv_error']:.3f})"
                    ),
                    show=False,
                )
                fig = ax.figure
                fig.savefig(diag_dir / f"{role}_step_{entry['step']}.png", dpi=300)
                plt.close(fig)

    metadata = {
        "dataset": dataset_name,
        "scenario_id": scenario_id,
        "pair": {
            "primary": {
                "vehicle_id": primary["vehicle_id"],
                "is_av": primary["is_av"],
                "label": labels[0],
                "start_time": primary["start_time"],
                "end_time": primary["end_time"],
            },
            "secondary": {
                "vehicle_id": secondary["vehicle_id"],
                "is_av": secondary["is_av"],
                "label": labels[1],
                "start_time": secondary["start_time"],
                "end_time": secondary["end_time"],
            },
        },
        "settings": {
            "heading_threshold_deg": HEADING_THRESHOLD_DEG,
            "history_window": HISTORY_WINDOW,
            "min_observation": MIN_OBSERVATION,
            "diagnostic_steps": list(DIAGNOSTIC_STEPS),
        },
    }
    metadata_path = data_dir / f"{case_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run IPV estimation on interhub trajectory datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Specific dataset filenames (e.g. trajectory_data_interaction_single.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--lane-threshold",
        type=float,
        default=LANE_DISTANCE_THRESHOLD,
        help="Lane distance threshold in meters",
    )
    parser.add_argument(
        "--heading-threshold",
        type=float,
        default=HEADING_THRESHOLD_DEG,
        help="Heading change threshold in degrees",
    )
    args = parser.parse_args()

    if args.datasets:
        json_files = [INTERHUB_ROOT / name for name in args.datasets]
    else:
        json_files = sorted(INTERHUB_ROOT.glob("trajectory_data_*.json"))

    json_files = [path for path in json_files if path.exists()]
    if not json_files:
        LOGGER.warning("No matching trajectory_data files found under %s", INTERHUB_ROOT)
        return

    for json_path in json_files:
        process_dataset(
            json_path,
            lane_distance_threshold=args.lane_threshold,
            heading_threshold_deg=args.heading_threshold,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
