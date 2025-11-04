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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent import Agent  # noqa: F401  # ensure agent modifications are applied
from ipv_estimation import (
    MotionSequence,
    estimate_ipv_pair,
    plot_virtual_vs_observed,
)
from tools.utility import smooth_ployline

LOGGER = logging.getLogger(__name__)

INTERHUB_ROOT = THIS_DIR / "interhub_traj_lane"
OUTPUT_ROOT = INTERHUB_ROOT / "ipv_estimation"
SELECTION_DIAG_ROOT = INTERHUB_ROOT / "diagnostics_selection_skipped"

LANE_DISTANCE_THRESHOLD = 2.0
HEADING_THRESHOLD_DEG = 12.0
HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
DIAGNOSTIC_STEPS = (5, 6)
DEFAULT_ENABLE_DIAGNOSTICS = False


def _save_ipv_table_labeled(
    output_path: Path,
    primary_motion: np.ndarray,
    secondary_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
    primary_label: str,
    secondary_label: str,
) -> None:
    """Save IPV results to Excel with dynamic labels based on heading classification."""
    columns = {
        f"ipv_{primary_label}": ipv_values[:, 0],
        f"ipv_{primary_label}_error": ipv_errors[:, 0],
        f"{primary_label}_px": primary_motion[:, 0],
        f"{primary_label}_py": primary_motion[:, 1],
        f"{primary_label}_vx": primary_motion[:, 2],
        f"{primary_label}_vy": primary_motion[:, 3],
        f"{primary_label}_heading": primary_motion[:, 4],
        f"ipv_{secondary_label}": ipv_values[:, 1],
        f"ipv_{secondary_label}_error": ipv_errors[:, 1],
        f"{secondary_label}_px": secondary_motion[:, 0],
        f"{secondary_label}_py": secondary_motion[:, 1],
        f"{secondary_label}_vx": secondary_motion[:, 2],
        f"{secondary_label}_vy": secondary_motion[:, 3],
        f"{secondary_label}_heading": secondary_motion[:, 4],
    }
    df = pd.DataFrame(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def _plot_ipv_summary_labeled(
    fig_path: Path,
    primary_motion: np.ndarray,
    secondary_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
    primary_label: str,
    secondary_label: str,
    primary_reference: Optional[np.ndarray] = None,
    secondary_reference: Optional[np.ndarray] = None,
) -> None:
    """Plot IPV summary with dynamic labels based on heading classification."""
    # Define colors for different maneuvers
    label_colors = {
        "lt": "#df7565",  # red-ish for left turn
        "rt": "#65df75",  # green-ish for right turn
        "gs": "#2FAECE",  # blue for go-straight
    }
    # Alternative colors for when both agents have the same label
    label_colors_alt = {
        "lt": "#c94f3d",  # darker red for left turn
        "rt": "#4dcc5d",  # darker green for right turn
        "gs": "#1a7a94",  # darker blue for go-straight
    }
    label_names = {
        "lt": "left-turn",
        "rt": "right-turn",
        "gs": "go-straight",
    }
    
    # Extract base label (remove numeric suffix if present, e.g., "gs2" -> "gs")
    def get_base_label(label: str) -> str:
        for base in ["lt", "rt", "gs"]:
            if label.startswith(base):
                return base
        return label
    
    primary_base = get_base_label(primary_label)
    secondary_base = get_base_label(secondary_label)
    
    # Assign colors: if both base labels are the same, use alternative color for secondary
    primary_color = label_colors.get(primary_base, "#df7565")
    if primary_base == secondary_base:
        secondary_color = label_colors_alt.get(secondary_base, "#1a7a94")
    else:
        secondary_color = label_colors.get(secondary_base, "#2FAECE")
    
    # Get display names (add suffix if label has one)
    primary_name = label_names.get(primary_base, primary_base)
    secondary_name = label_names.get(secondary_base, secondary_base)
    if primary_label != primary_base:  # has numeric suffix
        primary_name = f"{primary_name} (1)"
    if secondary_label != secondary_base:  # has numeric suffix
        suffix_num = secondary_label[len(secondary_base):]
        secondary_name = f"{secondary_name} ({suffix_num})"
    
    fig, (ax_ipv, ax_traj) = plt.subplots(1, 2, figsize=(16, 8))
    time_idx = np.arange(len(ipv_values))

    ax_ipv.set_ylim([-2, 2])
    if len(time_idx) >= 2:
        primary_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_values[:, 0])))
        primary_err_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_errors[:, 0])))
        secondary_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_values[:, 1])))
        secondary_err_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_errors[:, 1])))

        ax_ipv.plot(primary_curve[:, 0], primary_curve[:, 1], color=primary_color, 
                    label=f"estimated {primary_label} IPV")
        ax_ipv.fill_between(
            primary_curve[:, 0],
            primary_curve[:, 1] - primary_err_curve[:, 1],
            primary_curve[:, 1] + primary_err_curve[:, 1],
            alpha=0.3,
            color=primary_color,
        )
        ax_ipv.plot(secondary_curve[:, 0], secondary_curve[:, 1], color=secondary_color,
                    label=f"estimated {secondary_label} IPV")
        ax_ipv.fill_between(
            secondary_curve[:, 0],
            secondary_curve[:, 1] - secondary_err_curve[:, 1],
            secondary_curve[:, 1] + secondary_err_curve[:, 1],
            alpha=0.3,
            color=secondary_color,
        )
    else:
        ax_ipv.plot(time_idx, ipv_values[:, 0], color=primary_color, 
                    label=f"estimated {primary_label} IPV")
        ax_ipv.plot(time_idx, ipv_values[:, 1], color=secondary_color,
                    label=f"estimated {secondary_label} IPV")

    ax_ipv.set_xlabel("time index")
    ax_ipv.set_ylabel("IPV")
    ax_ipv.legend()

    # Plot trajectories
    ax_traj.plot(primary_motion[:, 0], primary_motion[:, 1], color=primary_color, 
                 linewidth=2.0, label=primary_name, zorder=3)
    ax_traj.plot(secondary_motion[:, 0], secondary_motion[:, 1], color=secondary_color, 
                 linewidth=2.0, label=secondary_name, zorder=3)
    
    # Plot reference paths if provided
    if primary_reference is not None:
        ax_traj.plot(primary_reference[:, 0], primary_reference[:, 1], 
                     color=primary_color, linestyle='--', linewidth=1.5, 
                     alpha=0.6, label=f"{primary_name} reference", zorder=2)
    if secondary_reference is not None:
        ax_traj.plot(secondary_reference[:, 0], secondary_reference[:, 1], 
                     color=secondary_color, linestyle='--', linewidth=1.5, 
                     alpha=0.6, label=f"{secondary_name} reference", zorder=2)
    
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _load_dataset(
    json_path: Path,
    lane_distance_threshold: float = LANE_DISTANCE_THRESHOLD,
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, List[str]]]:
    """
    Load the raw JSON file and regroup entries by scenario/vehicle.

    Returns:
        (
            {"scenario_id": {"vehicle_id": normalized_vehicle_entry, ...}, ...},
            {"reason": [scenario_ids], ...}
        )
    """
    LOGGER.info("Loading dataset %s", json_path.name)
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scenarios: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    skip_reasons: Dict[str, List[str]] = {
        "missing_reference": [],
        "lane_distance_exceeded": [],
        "invalid_entry": [],
    }

    for value in raw.values():
        scenario_id = str(value.get("scenario_idx"))
        reference = value.get("filtered_centerline_path") or value.get("centerline_path")
        if not reference:
            skip_reasons["missing_reference"].append(scenario_id)
            continue

        if lane_distance_threshold is not None:
            dist = value.get("trajectory_centerline_distance")
            if dist is not None and float(dist) > lane_distance_threshold:
                skip_reasons["lane_distance_exceeded"].append(scenario_id)
                continue

        prepared = _prepare_vehicle_entry(value, reference)
        if prepared is None:
            skip_reasons["invalid_entry"].append(scenario_id)
            continue

        scenario_id = prepared["scenario_id"]
        vehicle_id = prepared["vehicle_id"]
        scenarios.setdefault(scenario_id, {})[vehicle_id] = prepared

    LOGGER.info(
        "Loaded %d scenarios from %s (skipped %d entries)",
        len(scenarios),
        json_path.name,
        sum(len(v) for v in skip_reasons.values()),
    )
    return scenarios, skip_reasons


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
) -> Tuple[Optional[Tuple[Dict[str, object], Dict[str, object]]], Optional[str]]:
    avs = [v for v in vehicles.values() if v["is_av"]]
    hvs = [v for v in vehicles.values() if not v["is_av"]]

    if avs and hvs:
        return (avs[0], hvs[0]), None
    if len(hvs) >= 2:
        return (hvs[0], hvs[1]), None
    if avs and not hvs:
        reason = "only autonomous vehicles" if len(avs) > 1 else "single autonomous vehicle"
        return None, reason
    if not avs and len(hvs) == 1:
        return None, "only one human-driven vehicle"
    if len(vehicles) < 2:
        return None, "fewer than two vehicles after filtering"
    return None, "no valid pairing combination"


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
    enable_diagnostics: bool = DEFAULT_ENABLE_DIAGNOSTICS,
    diagnostic_steps: Optional[Sequence[int]] = DIAGNOSTIC_STEPS,
) -> None:
    scenarios, load_skips = _load_dataset(json_path, lane_distance_threshold)
    dataset_name = json_path.stem
    tasks: [
        Tuple[
            str,
            str,
            Dict[str, object],
            Dict[str, object],
            List[str],
            Path,
            bool,
            Optional[Sequence[int]],
        ]
    ] = []
    selection_skipped: List[Tuple[str, Dict[str, Dict[str, object]], str]] = []

    for scenario_id, vehicles in scenarios.items():
        pair, reason = _select_vehicle_pair(vehicles)
        if not pair:
            selection_skipped.append((scenario_id, vehicles, reason or "no valid pair"))
            continue

        primary, secondary = pair
        labels = [
            _classify_heading(primary["headings"], heading_threshold_deg),
            _classify_heading(secondary["headings"], heading_threshold_deg),
        ]
        labels = _ensure_unique_labels(labels)
        steps = diagnostic_steps if enable_diagnostics else None
        tasks.append(
            (
                dataset_name,
                scenario_id,
                primary,
                secondary,
                labels,
                output_root,
                enable_diagnostics,
                steps,
            )
        )

    if not tasks:
        _log_load_skips(load_skips)
        if selection_skipped:
            _log_selection_skips(selection_skipped, dataset_name)
            LOGGER.info(
                "Selection-stage skips (no valid pair): %d scenarios (examples: %s)",
                len(selection_skipped),
                [sid for sid, _, _ in selection_skipped[:5]],
            )
        LOGGER.info(
            "Dataset %s completed: 0 processed, 0 failed, %d skipped",
            dataset_name,
            len(selection_skipped),
        )
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
                    enable_diagnostics=task[6],
                    diagnostic_steps=task[7],
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
        len(selection_skipped),
    )
    _log_load_skips(load_skips)
    if selection_skipped:
        _log_selection_skips(selection_skipped, dataset_name)
        LOGGER.info(
            "Selection-stage skips (no valid pair): %d scenarios (examples: %s)",
            len(selection_skipped),
            [sid for sid, _, _ in selection_skipped[:5]],
        )


def _log_load_skips(skip_reasons: Dict[str, List[str]]) -> None:
    for reason, ids in skip_reasons.items():
        if not ids:
            continue
        LOGGER.info("Load-stage skips (%s): %d entries (examples: %s)",
                    reason, len(ids), ids[:5])


def _log_selection_skips(
    skipped: List[Tuple[str, Dict[str, Dict[str, object]], str]],
    dataset_name: str,
) -> None:
    reason_counts: Dict[str, int] = {}
    for _, _, reason in skipped:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in reason_counts.items():
        LOGGER.info("Selection skip reason '%s': %d scenarios", reason, count)

    for scenario_id, vehicles, reason in skipped:
        if not vehicles:
            LOGGER.info(
                "Skipping plotting for scenario %s (%s) because no vehicles remain after filtering",
                scenario_id,
                reason,
            )
            continue
        _plot_selection_skip(dataset_name, scenario_id, vehicles, reason)


def _process_task(
    task: Tuple[
        str,
        str,
        Dict[str, object],
        Dict[str, object],
        List[str],
        Path,
        bool,
        Optional[Sequence[int]],
    ]
) -> None:
    dataset_name, scenario_id, primary, secondary, labels, output_root, enable_diagnostics, diagnostic_steps = task
    _process_pair(
        dataset_name=dataset_name,
        scenario_id=scenario_id,
        primary=primary,
        secondary=secondary,
        labels=labels,
        output_root=output_root,
        enable_diagnostics=enable_diagnostics,
        diagnostic_steps=diagnostic_steps,
    )


def _process_pair(
    *,
    dataset_name: str,
    scenario_id: str,
    primary: Dict[str, object],
    secondary: Dict[str, object],
    labels: List[str],
    output_root: Path,
    enable_diagnostics: bool,
    diagnostic_steps: Optional[Sequence[int]],
) -> None:
    seq_primary = _build_motion_sequence(primary, labels[0])
    seq_secondary = _build_motion_sequence(secondary, labels[1])

    diag_steps = diagnostic_steps if enable_diagnostics else None
    result = estimate_ipv_pair(
        seq_primary,
        seq_secondary,
        history_window=HISTORY_WINDOW,
        min_observation=MIN_OBSERVATION,
        return_diagnostics=enable_diagnostics,
        diagnostic_steps=diag_steps,
    )
    if enable_diagnostics:
        ipv_values, ipv_errors, diagnostics = result  # type: ignore[misc]
    else:
        ipv_values, ipv_errors = result  # type: ignore[misc]
        diagnostics = None

    steps = ipv_values.shape[0]
    primary_motion = seq_primary.data[:steps]
    secondary_motion = seq_secondary.data[:steps]

    case_name = f"{scenario_id}_{primary['vehicle_id']}_{secondary['vehicle_id']}"

    base_dir = output_root / dataset_name / f"scenario_{scenario_id}"
    data_dir = base_dir / "data"
    fig_dir = base_dir / "fig"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _save_ipv_table_labeled(
        data_dir / f"{case_name}_ipv_results.xlsx",
        primary_motion,
        secondary_motion,
        ipv_values,
        ipv_errors,
        labels[0],
        labels[1],
    )

    _plot_ipv_summary_labeled(
        fig_dir / f"{case_name}_ipv_curve.png",
        primary_motion,
        secondary_motion,
        ipv_values,
        ipv_errors,
        labels[0],
        labels[1],
        primary_reference=primary["reference"],
        secondary_reference=secondary["reference"],
    )

    diag_dir = fig_dir / "virtual_tracks" / case_name
    if enable_diagnostics and diagnostics:
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


def _plot_selection_skip(
    dataset_name: str,
    scenario_id: str,
    vehicles: Dict[str, Dict[str, object]],
    reason: str,
) -> None:
    diag_dir = SELECTION_DIAG_ROOT / dataset_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for vehicle_id, vehicle in vehicles.items():
        track = vehicle["positions"]
        ax.plot(
            track[:, 0],
            track[:, 1],
            marker="o",
            linewidth=1.5,
            label=f"{vehicle_id} ({'AV' if vehicle['is_av'] else 'HV'})",
        )
        ref = vehicle["reference"]
        ax.plot(
            ref[:, 0],
            ref[:, 1],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_title(f"Scenario {scenario_id} skipped: {reason}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(diag_dir / f"scenario_{scenario_id}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run IPV estimation on interhub trajectory datasets."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable virtual-track diagnostics and plotting",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Specific dataset filenames (e.g. trajectory_data_interaction_single.json)",
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
        # Use iterdir instead of glob to avoid path issues on some systems
        json_files = sorted([
            f for f in INTERHUB_ROOT.iterdir() 
            if f.is_file() and f.name.startswith("trajectory_data_") and f.name.endswith(".json")
        ])
    
    if not json_files:
        LOGGER.warning("No matching trajectory_data files found under %s", INTERHUB_ROOT)
        return
    
    LOGGER.info("Found %d trajectory data file(s) to process", len(json_files))

    for json_path in json_files:
        process_dataset(
            json_path,
            lane_distance_threshold=args.lane_threshold,
            heading_threshold_deg=args.heading_threshold,
            max_workers=args.workers,
            enable_diagnostics=args.diagnostics,
            diagnostic_steps=DIAGNOSTIC_STEPS if args.diagnostics else None,
        )


if __name__ == "__main__":
    main()

# inserted comment
