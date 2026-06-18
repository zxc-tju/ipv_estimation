"""
Batch IPV estimation workflow for Argoverse interaction datasets.

This script rewrites the legacy Jupyter notebook into a reusable Python module.
It relies on :mod:`ipv_estimation` for the core IPV inference and mirrors the
original directory structure for inputs and outputs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ipv_estimation import (
    MotionSequence,
    concat_motion,
    estimate_ipv_pair,
    plot_virtual_vs_observed,
)
from tools.utility import smooth_ployline

LOGGER = logging.getLogger(__name__)

DATA_ROOT = THIS_DIR / "0_souce_data"
OUTPUT_ROOT = THIS_DIR / "1_experiment_result" / "ipv_estimation"

# Toggle diagnostic plotting for virtual trajectories.
# Set these to True / specific indices before running main(), or override when
# calling process_dataset directly.
DEBUG_VIRTUAL_TRACKS = False
DEBUG_STEPS: Optional[Sequence[int]] = None
MAX_WORKERS: Optional[int] = 18


ARGO_CONFIG: Dict[str, Dict[str, Dict[str, Path]]] = {
    "argo1": {
        "rush": {
            "hv_hv": DATA_ROOT / "argo1" / "HV_HV_rush",
            "hv_av": DATA_ROOT / "argo1" / "HV_AV_rush",
            "av_hv": DATA_ROOT / "argo1" / "AV_HV_rush",
        },
        "yield": {
            "hv_hv": DATA_ROOT / "argo1" / "HV_HV_yield",
            "hv_av": DATA_ROOT / "argo1" / "HV_AV_yield",
            "av_hv": DATA_ROOT / "argo1" / "AV_HV_yield",
        },
    },
    "argo2": {
        "rush": {
            "hv_hv": DATA_ROOT / "argo2" / "interaction_hv" / "left_turn_rush",
            "hv_av": DATA_ROOT / "argo2" / "interaction_av_lt" / "left_turn_rush",
            "av_hv": DATA_ROOT / "argo2" / "interaction_av_gs" / "left_turn_rush",
        },
        "yield": {
            "hv_hv": DATA_ROOT / "argo2" / "interaction_hv" / "left_turn_yield",
            "hv_av": DATA_ROOT / "argo2" / "interaction_av_lt" / "left_turn_yield",
            "av_hv": DATA_ROOT / "argo2" / "interaction_av_gs" / "left_turn_yield",
        },
    },
}


@dataclass(frozen=True)
class CaseArtifacts:
    motion_lt: MotionSequence
    motion_gs: MotionSequence


def collect_case_ids(source_dir: Path) -> List[int]:
    ids = set()
    for csv_path in source_dir.glob("*.csv"):
        stem = csv_path.stem
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            ids.add(int(prefix))
    return sorted(ids)


def resolve_single_csv(source_dir: Path, pattern: str) -> Path:
    matches = sorted(source_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' in {source_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous matches for pattern '{pattern}' in {source_dir}: {matches}")
    return matches[0]


def load_case_artifacts(
    data_version: str,
    source_dir: Path,
    case_id: int,
    *,
    sample_time_argo1: float = 0.1,
) -> CaseArtifacts:
    if data_version == "argo1":
        lt_pos = pd.read_csv(source_dir / f"{case_id}_lt.csv")[["x", "y"]].values
        gs_pos = pd.read_csv(source_dir / f"{case_id}_gs.csv")[["x", "y"]].values

        lt_motion = concat_motion(lt_pos, sample_time=sample_time_argo1)
        gs_motion = concat_motion(gs_pos, sample_time=sample_time_argo1)

        lt_ref = pd.read_csv(source_dir / f"{case_id}_reflinelt.csv")[["x", "y"]].values
        gs_ref = pd.read_csv(source_dir / f"{case_id}_reflinegs.csv")[["x", "y"]].values
    elif data_version == "argo2":
        lt_motion = pd.read_csv(
            resolve_single_csv(source_dir, f"{case_id}_ego*.csv")
        )[["x", "y", "vx", "vy", "heading"]].values
        gs_motion = pd.read_csv(
            resolve_single_csv(source_dir, f"{case_id}_agent*.csv")
        )[["x", "y", "vx", "vy", "heading"]].values

        lt_ref = pd.read_csv(
            resolve_single_csv(source_dir, f"{case_id}_refline*_lt*.csv")
        )[["x", "y"]].values
        gs_ref = pd.read_csv(
            resolve_single_csv(source_dir, f"{case_id}_refline*_gs*.csv")
        )[["x", "y"]].values
    else:
        raise ValueError(f"Unsupported data_version '{data_version}'")

    lt_sequence = MotionSequence(data=lt_motion, target="lt_argo", reference=lt_ref)
    gs_sequence = MotionSequence(data=gs_motion, target="gs_argo", reference=gs_ref)
    return CaseArtifacts(motion_lt=lt_sequence, motion_gs=gs_sequence)


def save_ipv_table(
    output_path: Path,
    lt_motion: np.ndarray,
    gs_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
) -> None:
    columns = {
        "ipv_lt": ipv_values[:, 0],
        "ipv_lt_error": ipv_errors[:, 0],
        "lt_px": lt_motion[:, 0],
        "lt_py": lt_motion[:, 1],
        "lt_vx": lt_motion[:, 2],
        "lt_vy": lt_motion[:, 3],
        "lt_heading": lt_motion[:, 4],
        "ipv_gs": ipv_values[:, 1],
        "ipv_gs_error": ipv_errors[:, 1],
        "gs_px": gs_motion[:, 0],
        "gs_py": gs_motion[:, 1],
        "gs_vx": gs_motion[:, 2],
        "gs_vy": gs_motion[:, 3],
        "gs_heading": gs_motion[:, 4],
    }
    df = pd.DataFrame(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def _process_single_case(args) -> Tuple[int, bool, List[str]]:
    (
        case_id,
        data_version,
        scenario,
        source_dir,
        sample_time_argo1,
        debug_virtual_tracks,
        debug_steps,
        data_dir,
        fig_dir,
    ) = args

    source_dir = Path(source_dir)
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)

    errors: List[str] = []

    try:
        artifacts = load_case_artifacts(
            data_version,
            source_dir,
            case_id,
            sample_time_argo1=sample_time_argo1,
        )
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"load_case: {exc}")
        return case_id, False, errors

    try:
        if debug_virtual_tracks:
            ipv_values, ipv_errors, diagnostics = estimate_ipv_pair(
                artifacts.motion_lt,
                artifacts.motion_gs,
                return_diagnostics=True,
                diagnostic_steps=debug_steps,
            )
        else:
            ipv_values, ipv_errors = estimate_ipv_pair(
                artifacts.motion_lt,
                artifacts.motion_gs,
            )
            diagnostics = None
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"estimate_ipv: {exc}")
        return case_id, False, errors

    steps = ipv_values.shape[0]
    lt_motion = artifacts.motion_lt.data[:steps]
    gs_motion = artifacts.motion_gs.data[:steps]

    try:
        save_ipv_table(
            data_dir / f"{case_id}_ipv_results.xlsx",
            lt_motion,
            gs_motion,
            ipv_values,
            ipv_errors,
        )
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"save_table: {exc}")

    try:
        plot_ipv_summary(
            fig_dir / f"{case_id}_ipv_curve.png",
            lt_motion,
            gs_motion,
            ipv_values,
            ipv_errors,
        )
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"plot_summary: {exc}")

    if debug_virtual_tracks and diagnostics:
        debug_dir = fig_dir / "virtual_tracks" / f"{case_id}"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"create_debug_dir: {exc}")
        else:
            for role, entries in diagnostics.items():
                for entry in entries:
                    try:
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
                        fig.savefig(
                            debug_dir / f"{role}_step_{entry['step']}.png",
                            dpi=300,
                        )
                        plt.close(fig)
                    except Exception as exc:  # pylint: disable=broad-except
                        errors.append(f"plot_virtual[{role} step {entry['step']}]: {exc}")

    success = not errors
    return case_id, success, errors


def plot_ipv_summary(
    fig_path: Path,
    lt_motion: np.ndarray,
    gs_motion: np.ndarray,
    ipv_values: np.ndarray,
    ipv_errors: np.ndarray,
) -> None:
    fig, (ax_ipv, ax_traj) = plt.subplots(1, 2, figsize=(16, 8))
    time_idx = np.arange(len(ipv_values))

    ax_ipv.set_ylim([-2, 2])
    if len(time_idx) >= 2:
        lt_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_values[:, 0])))
        lt_err_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_errors[:, 0])))
        gs_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_values[:, 1])))
        gs_err_curve, _ = smooth_ployline(np.column_stack((time_idx, ipv_errors[:, 1])))

        ax_ipv.plot(lt_curve[:, 0], lt_curve[:, 1], color="blue", label="estimated lt IPV")
        ax_ipv.fill_between(
            lt_curve[:, 0],
            lt_curve[:, 1] - lt_err_curve[:, 1],
            lt_curve[:, 1] + lt_err_curve[:, 1],
            alpha=0.3,
            color="blue",
        )
        ax_ipv.plot(gs_curve[:, 0], gs_curve[:, 1], color="red", label="estimated gs IPV")
        ax_ipv.fill_between(
            gs_curve[:, 0],
            gs_curve[:, 1] - gs_err_curve[:, 1],
            gs_curve[:, 1] + gs_err_curve[:, 1],
            alpha=0.3,
            color="red",
        )
    else:
        ax_ipv.plot(time_idx, ipv_values[:, 0], color="blue", label="estimated lt IPV")
        ax_ipv.plot(time_idx, ipv_values[:, 1], color="red", label="estimated gs IPV")

    ax_ipv.set_xlabel("time index")
    ax_ipv.set_ylabel("IPV")
    ax_ipv.legend()

    ax_traj.plot(lt_motion[:, 0], lt_motion[:, 1], color="#df7565", label="left-turn")
    ax_traj.plot(gs_motion[:, 0], gs_motion[:, 1], color="#2FAECE", label="go-straight")
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def process_dataset(
    data_version: str,
    scenario: str,
    source_map: Dict[str, Path],
    *,
    sample_time_argo1: float = 0.1,
    debug_virtual_tracks: bool = False,
    debug_steps: Optional[Sequence[int]] = None,
    max_workers: Optional[int] = None,
) -> None:
    LOGGER.info("Processing %s %s cases", data_version, scenario)
    for path_name, source_dir in source_map.items():
        case_ids = collect_case_ids(source_dir)
        if not case_ids:
            LOGGER.warning("No cases found in %s", source_dir)
            continue

        data_dir = OUTPUT_ROOT / data_version / f"{path_name}_left_turn_{scenario}" / "data"
        fig_dir = OUTPUT_ROOT / data_version / f"{path_name}_left_turn_{scenario}" / "fig"
        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        debug_steps_payload = tuple(debug_steps) if debug_steps is not None else None
        tasks = [
            (
                case_id,
                data_version,
                scenario,
                str(source_dir),
                sample_time_argo1,
                debug_virtual_tracks,
                debug_steps_payload,
                str(data_dir),
                str(fig_dir),
            )
            for case_id in case_ids
        ]

        workers = max_workers if max_workers not in (None, 0) else MAX_WORKERS
        if workers in (None, 0):
            workers = os.cpu_count() or 1
        workers = max(1, min(workers, len(tasks)))

        if workers == 1:
            iterator = tqdm(tasks, desc=f"{data_version}-{path_name}-{scenario}", unit="case")
            for task in iterator:
                case_id, _, errors = _process_single_case(task)
                for msg in errors:
                    LOGGER.error(
                        "Case %s (%s %s %s): %s",
                        case_id,
                        data_version,
                        scenario,
                        path_name,
                        msg,
                    )
        else:
            desc = f"{data_version}-{path_name}-{scenario}"
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_single_case, task): task[0]
                    for task in tasks
                }
                with tqdm(total=len(futures), desc=desc, unit="case") as progress:
                    for future in as_completed(futures):
                        case_id = futures[future]
                        try:
                            _, _, errors = future.result()
                        except Exception as exc:  # pylint: disable=broad-except
                            LOGGER.error(
                                "Case %s (%s %s %s) raised an unhandled exception: %s",
                                case_id,
                                data_version,
                                scenario,
                                path_name,
                                exc,
                            )
                        else:
                            for msg in errors:
                                LOGGER.error(
                                    "Case %s (%s %s %s): %s",
                                    case_id,
                                    data_version,
                                    scenario,
                                    path_name,
                                    msg,
                                )
                        progress.update(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for data_version, scenarios in ARGO_CONFIG.items():
        for scenario, source_map in scenarios.items():
            process_dataset(
                data_version,
                scenario,
                source_map,
                debug_virtual_tracks=DEBUG_VIRTUAL_TRACKS,
                debug_steps=DEBUG_STEPS,
                max_workers=MAX_WORKERS,
            )


if __name__ == "__main__":
    main()
