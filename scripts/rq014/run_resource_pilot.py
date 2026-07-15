#!/usr/bin/env python3
"""Measure the rating-blind RQ014 lane-v3 pilot, including pinned M3 scoring."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import time
import types
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable


CELL_SELECTION_RULE_ID = "LANE_V3_NON_M3_COST_EXTREMES_V1"
NON_M3_STAGES = ("source_load", "window_assembly", "feature_prep")
M3_STAGE = "m3_scoring"
M3_COST_SENTINEL = "EXPLICITLY_UNMEASURED"
FAILURE_CODES = (
    "NONE",
    "INPUT_CONTRACT_FAILURE",
    "SOURCE_LOAD_FAILURE",
    "WINDOW_ASSEMBLY_FAILURE",
    "FEATURE_PREP_FAILURE",
    "M3_STAGE_FAILURE",
)
LIGHTEST_CELL_ID = "RR3-R04N-CH-W10-H20-NEX_MEAN"
HEAVIEST_CELL_ID = "RR3-R10L-TF-HFEAS-NEX_MEAN"
LANE_V3_AXIS_SHA256 = "72216349fe299a31c7f00d534e129b19e9a7c0cf8ac1ec3fb0876d71e09413a1"
LANE_V3_CELL_IDS_SHA256 = "db280b77a5fba7e7bb8546da9d2d22337e66c1b1d267d8f8acd281326eaaadee"
MAX_PARALLEL_WORKERS = 16
WORKER_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_SHARED_SOURCES: dict[str, Any] | None = None
_SHARED_M3: dict[str, Any] | None = None


class PilotError(ValueError):
    """Raised when a pilot contract or rating-blind input fails closed."""


class SourceGapError(PilotError):
    """Raised when interpolation would cross a frozen-ineligible source gap."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotError(f"Expected a JSON object: {path}")
    return value


def enumerate_lane_v3_cells(lane: dict[str, Any]) -> list[str]:
    bank = lane.get("rating_blind_feature_bank")
    if not isinstance(bank, dict):
        raise PilotError("Lane v3 has no rating_blind_feature_bank")
    temporal = bank.get("temporal_axis")
    if not isinstance(temporal, dict):
        raise PilotError("Lane v3 temporal axis is missing")
    sampling = bank.get("sampling_axis")
    recipes = temporal.get("recipes")
    horizons = bank.get("horizon_axis")
    readouts = bank.get("readout_axis")
    if not all(isinstance(axis, list) and axis for axis in (sampling, recipes, horizons, readouts)):
        raise PilotError("Lane v3 pilot axes are missing or empty")
    axis_contract = {
        "sampling_axis": sampling,
        "temporal_recipes": recipes,
        "horizon_axis": horizons,
        "readout_axis": readouts,
    }
    if hashlib.sha256(canonical_json_bytes(axis_contract)).hexdigest() != LANE_V3_AXIS_SHA256:
        raise PilotError("Lane v3 ordered axis contract drifted")
    cells = [
        f"RR3-{sample['sampling_id']}-{recipe['temporal_id']}-{horizon['horizon_id']}-{readout}"
        for sample in sampling
        for recipe in recipes
        for horizon in horizons
        for readout in readouts
    ]
    expected = bank.get("predictor_cell_enumeration", {}).get(
        "registered_predictor_cell_count"
    )
    cell_bytes = ("\n".join(cells) + "\n").encode("utf-8")
    if (
        expected != 320
        or len(cells) != expected
        or len(set(cells)) != expected
        or hashlib.sha256(cell_bytes).hexdigest() != LANE_V3_CELL_IDS_SHA256
    ):
        raise PilotError("Lane v3 predictor-cell enumeration is not the frozen 320-cell grid")
    return cells


def select_resource_pilot_cells(lane: dict[str, Any]) -> dict[str, Any]:
    cells = enumerate_lane_v3_cells(lane)
    if LIGHTEST_CELL_ID not in cells or HEAVIEST_CELL_ID not in cells:
        raise PilotError("Frozen resource-pilot endpoint cell is absent from lane v3")
    return {
        "rule_id": CELL_SELECTION_RULE_ID,
        "registered_cell_count": len(cells),
        "lightest_cell_id": LIGHTEST_CELL_ID,
        "heaviest_cell_id": HEAVIEST_CELL_ID,
    }


def _io_snapshot() -> dict[str, int | str]:
    proc_io = Path("/proc/self/io")
    if not proc_io.is_file():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "source": "getrusage_blocks",
            "read_bytes": int(usage.ru_inblock) * 512,
            "write_bytes": int(usage.ru_oublock) * 512,
            "read_chars": 0,
            "write_chars": 0,
        }
    values: dict[str, int] = {}
    for line in proc_io.read_text(encoding="ascii").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip())
    return {
        "source": "proc_self_io",
        "read_bytes": values.get("read_bytes", 0),
        "write_bytes": values.get("write_bytes", 0),
        "read_chars": values.get("rchar", 0),
        "write_chars": values.get("wchar", 0),
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _measure(stage_id: str, cell_id: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    before_io = _io_snapshot()
    before_wall = time.monotonic()
    before_cpu = time.process_time()
    failure: Exception | None = None
    try:
        result = function()
    except Exception as exc:  # stage failures are evidence, not silent row loss
        result = None
        failure = exc
    cpu_seconds = time.process_time() - before_cpu
    wall_seconds = time.monotonic() - before_wall
    after_io = _io_snapshot()
    measurement = {
        "cell_id": cell_id,
        "stage_id": stage_id,
        "status": "PASS" if failure is None else "FAIL",
        "failure_code": (
            "NONE"
            if failure is None
            else {
                "source_load": "SOURCE_LOAD_FAILURE",
                "window_assembly": "WINDOW_ASSEMBLY_FAILURE",
                "feature_prep": "FEATURE_PREP_FAILURE",
                "m3_scoring": "M3_STAGE_FAILURE",
                "m3_model_load": "M3_STAGE_FAILURE",
            }[stage_id]
        ),
        "walltime_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "io": {
            "source": after_io["source"],
            "read_bytes": int(after_io["read_bytes"]) - int(before_io["read_bytes"]),
            "write_bytes": int(after_io["write_bytes"]) - int(before_io["write_bytes"]),
            "read_chars": int(after_io["read_chars"]) - int(before_io["read_chars"]),
            "write_chars": int(after_io["write_chars"]) - int(before_io["write_chars"]),
        },
    }
    if failure is not None:
        measurement["failure_detail"] = {
            "exception_type": type(failure).__name__,
            "message": str(failure),
        }
    return result, measurement


def _normalise_header(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _read_positions(
    path: Path,
    *,
    key_columns: tuple[str, ...],
    time_column: str,
    x_column: str,
    y_column: str,
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[dict[tuple[str, ...], list[tuple[float, float, float]]], int]:
    groups: dict[tuple[str, ...], list[tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PilotError(f"CSV has no header: {path}")
        forbidden = {"preferencescore", "humanrating", "rating", "score", "observedrho"}
        if forbidden & {_normalise_header(name) for name in reader.fieldnames}:
            raise PilotError(f"Rating-bearing column reached the pilot: {path}")
        required = {*key_columns, time_column, x_column, y_column}
        if not required <= set(reader.fieldnames):
            raise PilotError(f"CSV lacks required pilot columns: {path}")
        row_count = 0
        for row in reader:
            if row_filter is not None and not row_filter(row):
                continue
            key = tuple(row[column] for column in key_columns)
            point = (float(row[time_column]), float(row[x_column]), float(row[y_column]))
            if not all(math.isfinite(value) for value in point):
                raise PilotError(f"Nonfinite pilot position row: {path}")
            groups.setdefault(key, []).append(point)
            row_count += 1
    for points in groups.values():
        points.sort()
    return groups, row_count


def _load_sources(bundle_root: Path, mapping_manifest_path: Path) -> dict[str, Any]:
    bundle_manifest = _load_json(bundle_root / "file_manifest.json")
    if bundle_manifest.get("schema_version") != "rq014-score-stripped-file-manifest-v1":
        raise PilotError("Score-stripped file-manifest schema drift")
    rows = bundle_manifest.get("files")
    if not isinstance(rows, list):
        raise PilotError("Score-stripped file-manifest rows are missing")
    registered: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("contains_rating") is not False:
            raise PilotError("Score-stripped file-manifest row is malformed")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or Path(relative).name != relative or relative in registered:
            raise PilotError("Score-stripped file-manifest path is unsafe or duplicated")
        path = bundle_root / relative
        if path.is_symlink() or not path.is_file():
            raise PilotError("Score-stripped registered source is missing or symlinked")
        if path.stat().st_size != row.get("size_bytes") or sha256_file(path) != row.get("sha256"):
            raise PilotError("Score-stripped registered source bytes drifted")
        registered.add(relative)
    required_sources = {
        "candidate_states.csv",
        "ego_history_states.csv",
        "counterpart_tracks.csv",
    }
    if not required_sources <= registered:
        raise PilotError("Score-stripped file-manifest omits a pilot source")

    manifest = _load_json(mapping_manifest_path)
    if manifest.get("schema_version") != "rq014-wod-path-type-mapping-manifest-v1":
        raise PilotError("WOD path-type mapping manifest schema drift")
    mapping_ref = manifest.get("mapping")
    if not isinstance(mapping_ref, dict) or set(mapping_ref) != {
        "format",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise PilotError("WOD path-type mapping reference is malformed")
    mapping_path = Path(mapping_ref["path"])
    if not mapping_path.is_file() or mapping_path.stat().st_size != mapping_ref["size_bytes"]:
        raise PilotError("WOD path-type mapping file is missing or size-drifted")
    if sha256_file(mapping_path) != mapping_ref["sha256"]:
        raise PilotError("WOD path-type mapping SHA-256 drift")
    mapped_segments: set[str] = set()
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["segment_id", "tstar_context_step", "path_type"]:
            raise PilotError("WOD path-type mapping columns drifted")
        for row in reader:
            mapped_segments.add(row["segment_id"])

    candidate, candidate_rows = _read_positions(
        bundle_root / "candidate_states.csv",
        key_columns=("segment_id", "candidate_id"),
        time_column="effective_time_s",
        x_column="pos_x_m",
        y_column="pos_y_m",
        row_filter=lambda row: row["included_in_effective_future"] == "true",
    )
    history, history_rows = _read_positions(
        bundle_root / "ego_history_states.csv",
        key_columns=("segment_id",),
        time_column="time_s",
        x_column="pos_x_m",
        y_column="pos_y_m",
    )
    counterpart, counterpart_rows = _read_positions(
        bundle_root / "counterpart_tracks.csv",
        key_columns=("segment_id",),
        time_column="time_s",
        x_column="x_m",
        y_column="y_m",
    )
    candidate = {key: value for key, value in candidate.items() if key[0] in mapped_segments}
    history = {key: value for key, value in history.items() if key[0] in mapped_segments}
    counterpart = {key: value for key, value in counterpart.items() if key[0] in mapped_segments}
    return {
        "candidate": candidate,
        "history": history,
        "counterpart": counterpart,
        "mapped_segment_count": len(mapped_segments),
        "source_row_count": candidate_rows + history_rows + counterpart_rows,
    }


def _interpolate(
    points: list[tuple[float, float, float]],
    target: float,
    *,
    maximum_source_gap_s: float | None = None,
) -> tuple[float, float, float]:
    times = [point[0] for point in points]
    index = bisect_right(times, target)
    tolerance = 1e-12
    if index == 0:
        if math.isclose(target, points[0][0], rel_tol=0.0, abs_tol=tolerance):
            return (target, points[0][1], points[0][2])
        raise PilotError("Interpolation target leaves observed support")
    if index <= len(points) - 1 and points[index - 1][0] == target:
        return (target, points[index - 1][1], points[index - 1][2])
    if index == len(points):
        if math.isclose(target, points[-1][0], rel_tol=0.0, abs_tol=tolerance):
            return (target, points[-1][1], points[-1][2])
        raise PilotError("Interpolation target leaves observed support")
    left, right = points[index - 1], points[index]
    source_gap_s = right[0] - left[0]
    if maximum_source_gap_s is not None and source_gap_s > maximum_source_gap_s:
        raise SourceGapError("Interpolation source gap exceeds the frozen 2*dt limit")
    fraction = (target - left[0]) / source_gap_s
    return (
        target,
        left[1] + fraction * (right[1] - left[1]),
        left[2] + fraction * (right[2] - left[2]),
    )


def _resample(
    points: list[tuple[float, float, float]],
    rate_hz: int,
    *,
    interpolate_to_grid: bool = False,
    maximum_source_gap_s: float | None = None,
) -> list[tuple[float, float, float]]:
    if rate_hz == 4 and not interpolate_to_grid:
        return points
    first_tick = math.ceil((points[0][0] * rate_hz) - 1e-9)
    last_tick = math.floor((points[-1][0] * rate_hz) + 1e-9)
    return [
        _interpolate(
            points,
            tick / rate_hz,
            maximum_source_gap_s=maximum_source_gap_s,
        )
        for tick in range(first_tick, last_tick + 1)
    ]


def _grid_tick_points(
    points: list[tuple[float, float, float]],
    rate_hz: int,
    *,
    interpolate_to_grid: bool = False,
    maximum_source_gap_s: float | None = None,
) -> dict[int, tuple[float, float, float]]:
    tick_points: dict[int, tuple[float, float, float]] = {}
    for point in _resample(
        points,
        rate_hz,
        interpolate_to_grid=interpolate_to_grid,
        maximum_source_gap_s=maximum_source_gap_s,
    ):
        raw_tick = point[0] * rate_hz
        tick = round(raw_tick)
        if not math.isclose(raw_tick, tick, abs_tol=1e-9):
            raise PilotError("Position timestamp is off the registered grid phase")
        if tick in tick_points:
            raise PilotError("Position timeline has a duplicate registered grid tick")
        tick_points[tick] = point
    return tick_points


def _window_tick_bounds(
    temporal_id: str,
    tau_tick: int,
    h_common_tick: int,
    rate_hz: int,
) -> tuple[int, int]:
    if temporal_id == "CH-W10":
        return tau_tick - rate_hz, tau_tick
    if temporal_id == "TF":
        return 0, h_common_tick
    raise PilotError(f"Pilot selected an unsupported temporal endpoint: {temporal_id}")


def _exact_tick_window(
    tick_points: dict[int, tuple[float, float, float]],
    lower_tick: int,
    upper_tick: int,
) -> list[tuple[float, float, float]] | None:
    required = range(lower_tick, upper_tick + 1)
    if not all(tick in tick_points for tick in required):
        return None
    return [tick_points[tick] for tick in required]


def _assemble_windows(sources: dict[str, Any], cell_id: str) -> list[tuple[list[Any], list[Any]]]:
    parts = cell_id.split("-")
    sampling_id = parts[1]
    temporal_id = "-".join(parts[2:-2])
    horizon_id = parts[-2]
    rate_hz = 4 if sampling_id == "R04N" else 10
    windows: list[tuple[list[Any], list[Any]]] = []
    candidates_by_segment: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    for (segment_id, candidate_id), future in sources["candidate"].items():
        candidates_by_segment.setdefault(segment_id, {})[candidate_id] = future
    for segment_id in sorted(candidates_by_segment):
        futures = candidates_by_segment[segment_id]
        history = sources["history"].get((segment_id,))
        counterpart = sources["counterpart"].get((segment_id,))
        if not history or not counterpart or set(futures) != {"C1", "C2", "C3"}:
            continue
        branch_ticks = {
            candidate_id: _grid_tick_points(history + futures[candidate_id], rate_hz)
            for candidate_id in ("C1", "C2", "C3")
        }
        try:
            counterpart_ticks = _grid_tick_points(
                counterpart,
                rate_hz,
                interpolate_to_grid=True,
                maximum_source_gap_s=2.0 / rate_hz,
            )
        except SourceGapError:
            continue
        four_way_ticks = [*branch_ticks.values(), counterpart_ticks]
        if any(not tick_points for tick_points in four_way_ticks):
            continue
        h_common_tick = min(max(tick_points) for tick_points in four_way_ticks)
        h20_ticks = tuple(range(rate_hz, 2 * rate_hz + 1))

        def exact_windows(
            tau_tick: int,
        ) -> tuple[dict[str, list[tuple[float, float, float]]], list[tuple[float, float, float]]] | None:
            if tau_tick > h_common_tick:
                return None
            lower_tick, upper_tick = _window_tick_bounds(
                temporal_id, tau_tick, h_common_tick, rate_hz
            )
            candidate_windows = {
                candidate_id: _exact_tick_window(
                    branch_ticks[candidate_id], lower_tick, upper_tick
                )
                for candidate_id in ("C1", "C2", "C3")
            }
            counterpart_window = _exact_tick_window(
                counterpart_ticks, lower_tick, upper_tick
            )
            if counterpart_window is None or any(
                candidate_window is None
                for candidate_window in candidate_windows.values()
            ):
                return None
            resolved_candidates = {
                candidate_id: candidate_window
                for candidate_id, candidate_window in candidate_windows.items()
                if candidate_window is not None
            }
            try:
                for window in [*resolved_candidates.values(), counterpart_window]:
                    _derive_state(window)
            except PilotError:
                return None
            return resolved_candidates, counterpart_window

        resolved_windows = {
            tick: exact_windows(tick)
            for tick in range(rate_hz, h_common_tick + 1)
        }
        h20_windows = {tick: resolved_windows.get(tick) for tick in h20_ticks}
        if any(value is None for value in h20_windows.values()):
            continue
        anchor_ticks = (
            h20_ticks
            if horizon_id == "H20"
            else tuple(
                tick
                for tick, resolved in resolved_windows.items()
                if resolved is not None
            )
        )
        for tick in anchor_ticks:
            resolved = resolved_windows[tick]
            if resolved is None:
                continue
            candidate_windows, counterpart_window = resolved
            for candidate_id in ("C1", "C2", "C3"):
                windows.append((candidate_windows[candidate_id], counterpart_window))
    if not windows:
        raise PilotError(f"No complete rating-blind windows for {cell_id}")
    return windows


def _derive_state(points: list[tuple[float, float, float]]) -> list[tuple[float, ...]]:
    if len(points) < 2 or not all(
        math.isfinite(value) for point in points for value in point
    ):
        raise PilotError("Window-local state requires at least two finite positions")
    dt = points[1][0] - points[0][0]
    if dt <= 0.0 or any(
        not math.isclose(points[index][0] - points[index - 1][0], dt, abs_tol=1e-9)
        for index in range(2, len(points))
    ):
        raise PilotError("Window-local positions are not on one uniform grid")

    def finite_difference(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
        derived = [((values[1][0] - values[0][0]) / dt, (values[1][1] - values[0][1]) / dt)]
        for index in range(1, len(values) - 1):
            derived.append(
                (
                    (values[index + 1][0] - values[index - 1][0]) / (2.0 * dt),
                    (values[index + 1][1] - values[index - 1][1]) / (2.0 * dt),
                )
            )
        derived.append(
            (
                (values[-1][0] - values[-2][0]) / dt,
                (values[-1][1] - values[-2][1]) / dt,
            )
        )
        return derived

    position_xy = [(point[1], point[2]) for point in points]
    velocity = finite_difference(position_xy)
    acceleration = finite_difference(velocity)
    heading: list[float | None] = []
    for vx, vy in velocity:
        if math.hypot(vx, vy) <= 1e-9:
            heading.append(None)
            continue
        value = math.atan2(vy, vx)
        heading.append(math.pi if value == -math.pi else value)
    if all(value is None for value in heading):
        raise PilotError("All-stationary window has undefined heading")
    first_defined = next(index for index, value in enumerate(heading) if value is not None)
    heading[:first_defined] = [heading[first_defined]] * first_defined
    for index in range(first_defined + 1, len(heading)):
        if heading[index] is None:
            heading[index] = heading[index - 1]
    states: list[tuple[float, ...]] = []
    for index, point in enumerate(points):
        states.append(
            (
                point[0],
                point[1],
                point[2],
                velocity[index][0],
                velocity[index][1],
                acceleration[index][0],
                acceleration[index][1],
                float(heading[index]),
            )
        )
    return states


def _prepare_features(windows: list[tuple[list[Any], list[Any]]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    state_rows = 0
    for branch, other in windows:
        for states in (_derive_state(branch), _derive_state(other)):
            state_rows += len(states)
            for state in states:
                digest.update(("|".join(format(value, ".17g") for value in state) + "\n").encode("ascii"))
    return {"window_count": len(windows), "state_row_count": state_rows, "feature_prep_sha256": digest.hexdigest()}


def _load_m3_model(artifact_path: Path) -> dict[str, Any]:
    if "sociality_estimation.verifier.model" not in sys.modules:
        sociality_package = types.ModuleType("sociality_estimation")
        sociality_package.__path__ = ()
        verifier_package = types.ModuleType("sociality_estimation.verifier")
        verifier_package.__path__ = ()
        sociality_package.verifier = verifier_package
        sys.modules["sociality_estimation"] = sociality_package
        sys.modules["sociality_estimation.verifier"] = verifier_package
        model_path = Path(__file__).resolve().parents[2] / "src" / "sociality_estimation" / "verifier" / "model.py"
        model_spec = importlib.util.spec_from_file_location(
            "sociality_estimation.verifier.model", model_path
        )
        if model_spec is None or model_spec.loader is None:
            raise PilotError("Cannot construct exact-path portable M3 model loader")
        model_module = importlib.util.module_from_spec(model_spec)
        verifier_package.model = model_module
        sys.modules["sociality_estimation.verifier.model"] = model_module
        model_spec.loader.exec_module(model_module)
    import joblib

    scorer = joblib.load(artifact_path, mmap_mode=None)
    required = {"tier_model", "gate_model", "radii", "feature_contract"}
    if not isinstance(scorer, dict) or not required <= set(scorer):
        raise PilotError("Portable M3 scorer key contract drift")
    return scorer


def _score_m3_stage(window_count: int, scorer: dict[str, Any]) -> dict[str, Any]:
    """Run M3 on a deterministic rating-free, row-count-matched cost substrate."""

    import numpy as np
    import pandas as pd
    from sociality_estimation.verifier import model

    if window_count <= 0:
        raise PilotError("M3 cost substrate row count is empty")
    gate = scorer["gate_model"]
    base_row = {
        column: 0.0 for column in scorer["feature_contract"]["required_input_columns"]
    }
    joint = str(sorted(gate.supported_joint_cells)[0]).split("|")
    if len(joint) != len(gate.joint_cell_columns):
        raise PilotError("M3 frozen joint support cell is malformed")
    base_row.update(dict(zip(gate.joint_cell_columns, joint)))
    for column in gate.support_columns:
        if column not in gate.joint_cell_columns:
            base_row[column] = sorted(gate.support_levels[column])[0]
    for tier in (scorer["tier_model"],):
        for index, column in enumerate(tier.spec.categorical):
            if column not in gate.support_columns:
                base_row[column] = str(tier.preprocessor.encoder.categories_[index][0])
    frame = pd.DataFrame([base_row] * window_count)
    quantiles, _, _ = model.predict_tier_quantiles(scorer["tier_model"], frame)
    category_ok = model.category_support_mask(frame, gate)
    distances = np.full(len(frame), np.nan, dtype=np.float32)
    if category_ok.any():
        matrix = model.transform_gate_matrix(frame, gate)
        values, _ = gate.tree.query(matrix[category_ok], k=model.GATE_K, workers=1)
        distances[category_ok] = values.mean(axis=1).astype(np.float32)
    gate_ok = category_ok & (distances <= gate.threshold)
    calibrated: list[np.ndarray] = []
    for alpha in model.ALPHAS:
        lower_level, upper_level = model.QUANTILE_BY_ALPHA[alpha]
        lower, upper = model.calibrated_bounds(
            quantiles[:, model.Q_INDEX[lower_level]],
            quantiles[:, model.Q_INDEX[upper_level]],
            float(scorer["radii"][model.ALPHA_LABEL[alpha]]["c_alpha"]),
        )
        lower[~gate_ok] = np.nan
        upper[~gate_ok] = np.nan
        calibrated.extend((lower, upper))
    digest = hashlib.sha256()
    for index in range(len(frame)):
        digest.update(
            ("|".join(format(float(value), ".17g") for value in quantiles[index])
             + "|" + "|".join(format(float(values[index]), ".17g") for values in calibrated)
             + f"|{int(gate_ok[index])}\n").encode("ascii")
        )
    return {
        "scored_row_count": len(frame),
        "support_gate_pass_count": int(gate_ok.sum()),
        "m3_output_sha256": digest.hexdigest(),
        "knn_workers": 1,
        "workload_kind": "DETERMINISTIC_RATING_FREE_FROZEN_SUPPORT_VECTOR_COST_ONLY",
    }


def _configure_worker_thread_limits() -> None:
    for name, value in WORKER_THREAD_LIMITS.items():
        os.environ[name] = value


def _run_pilot_cell(
    cell_id: str,
) -> dict[str, Any]:
    if _SHARED_SOURCES is None or _SHARED_M3 is None:
        raise PilotError("Shared pilot sources were not installed before worker start")
    sources = _SHARED_SOURCES
    measurements: list[dict[str, Any]] = []
    windows, window_measurement = _measure(
        "window_assembly", cell_id, lambda: _assemble_windows(sources, cell_id)
    )
    measurements.append(window_measurement)
    detail: dict[str, Any] | None = None
    projection_basis: dict[str, float] | None = None
    if windows is not None:
        feature_detail, feature_measurement = _measure(
            "feature_prep", cell_id, lambda: _prepare_features(windows)
        )
        measurements.append(feature_measurement)
        if feature_detail is not None:
            m3_detail, m3_measurement = _measure(
                M3_STAGE,
                cell_id,
                lambda: _score_m3_stage(
                    feature_detail["window_count"],
                    _SHARED_M3["scorer"],
                ),
            )
            measurements.append(m3_measurement)
            detail = {
                "mapped_segment_count": sources["mapped_segment_count"],
                "source_row_count": sources["source_row_count"],
                **feature_detail,
            }
            if m3_detail is not None:
                detail.update(m3_detail)
            if m3_detail is not None:
                projection_basis = {
                "per_cell_cpu_seconds": (
                    window_measurement["cpu_seconds"]
                    + feature_measurement["cpu_seconds"]
                ),
                "per_cell_walltime_seconds": (
                    window_measurement["walltime_seconds"]
                    + feature_measurement["walltime_seconds"]
                ),
                "per_cell_m3_cpu_seconds": m3_measurement["cpu_seconds"],
                "per_cell_m3_walltime_seconds": m3_measurement["walltime_seconds"],
                }
    stage_timings = {
        measurement["stage_id"]: {
            "walltime_seconds": measurement["walltime_seconds"],
            "cpu_seconds": measurement["cpu_seconds"],
        }
        for measurement in measurements
    }
    return {
        "cell_id": cell_id,
        "measurements": measurements,
        "detail": detail,
        "projection_basis": projection_basis,
        "serial_timing": {
            "stages": stage_timings,
            "measured_stage_count": len(measurements),
            "total_serial_walltime_seconds": sum(
                measurement["walltime_seconds"] for measurement in measurements
            ),
            "total_serial_cpu_seconds": sum(
                measurement["cpu_seconds"] for measurement in measurements
            ),
        },
    }


def _validate_preflight_chain(
    *,
    preflight_receipt_path: Path,
    preflight_done_path: Path,
    input_manifest_path: Path,
    materialization_ledger_path: Path,
    mapping_manifest_path: Path,
    m3_artifact_path: Path,
    m3_artifact_size_bytes: int,
    m3_artifact_sha256: str,
    export_receipt_path: Path,
    export_done_path: Path,
) -> dict[str, str]:
    receipt = _load_json(preflight_receipt_path)
    done = _load_json(preflight_done_path)
    if receipt.get("schema_version") != "rq014-g2-contract-preflight-receipt-v1" or receipt.get("status") != "PASS":
        raise PilotError("Prior contract-preflight receipt is not PASS")
    if set(done) != {"schema_version", "operation", "receipt_sha256", "status"}:
        raise PilotError("Prior contract-preflight DONE exact keys drifted")
    if (
        done.get("schema_version") != "rq014-managed-operation-done-v1"
        or done.get("operation") != "rq014_g2_contract_preflight"
        or done.get("status") != "PASS"
        or done.get("receipt_sha256") != sha256_file(preflight_receipt_path)
    ):
        raise PilotError("Prior contract-preflight DONE chain is invalid")
    m3_receipt = receipt.get("m3_artifact_input_receipt")
    if (
        not isinstance(m3_receipt, dict)
        or m3_receipt.get("sha256") != m3_artifact_sha256
        or m3_receipt.get("size_bytes") != m3_artifact_size_bytes
        or m3_receipt.get("deserialized") is not False
        or m3_artifact_path.stat().st_size != m3_artifact_size_bytes
        or sha256_file(m3_artifact_path) != m3_artifact_sha256
    ):
        raise PilotError("Frozen M3 verification-only lineage differs")
    expected = {
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "materialization_ledger_sha256": sha256_file(materialization_ledger_path),
        "declassification_export_receipt_sha256": sha256_file(export_receipt_path),
        "declassification_export_done_sha256": sha256_file(export_done_path),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise PilotError("Prior contract-preflight lineage differs from the pilot inputs")
    mapping = receipt.get("wod_path_type_mapping")
    if not isinstance(mapping, dict) or mapping.get("manifest_sha256") != sha256_file(mapping_manifest_path):
        raise PilotError("Prior contract-preflight mapping lineage differs")
    return {
        "contract_preflight_receipt_sha256": sha256_file(preflight_receipt_path),
        "contract_preflight_done_sha256": sha256_file(preflight_done_path),
        **expected,
        "wod_path_type_mapping_manifest_sha256": sha256_file(mapping_manifest_path),
        "m3_artifact_sha256": m3_artifact_sha256,
    }


def run_resource_pilot(
    *,
    run_id: str,
    lane_path: Path,
    bundle_root: Path,
    input_manifest_path: Path,
    sanitization_receipt_path: Path,
    materialization_ledger_path: Path,
    mapping_manifest_path: Path,
    m3_artifact_path: Path,
    m3_artifact_size_bytes: int,
    m3_artifact_sha256: str,
    m3_parity_fixture_path: Path,
    export_receipt_path: Path,
    export_done_path: Path,
    preflight_receipt_path: Path,
    preflight_done_path: Path,
) -> dict[str, Any]:
    global _SHARED_SOURCES, _SHARED_M3
    try:
        lane = _load_json(lane_path)
        selection = select_resource_pilot_cells(lane)
        lineage = _validate_preflight_chain(
            preflight_receipt_path=preflight_receipt_path,
            preflight_done_path=preflight_done_path,
            input_manifest_path=input_manifest_path,
            materialization_ledger_path=materialization_ledger_path,
            mapping_manifest_path=mapping_manifest_path,
            m3_artifact_path=m3_artifact_path,
            m3_artifact_size_bytes=m3_artifact_size_bytes,
            m3_artifact_sha256=m3_artifact_sha256,
            export_receipt_path=export_receipt_path,
            export_done_path=export_done_path,
        )
        if sha256_file(m3_parity_fixture_path) != "ae62b9fddba53308d319ccef5a70d56a9f0ae243fe009aa3f85e36cb20fcee37":
            raise PilotError("M3 parity fixture differs from the reviewed v4 standard")
    except Exception as exc:
        taxonomy = {code: 0 for code in FAILURE_CODES if code != "NONE"}
        taxonomy["INPUT_CONTRACT_FAILURE"] = 1
        return {
            "schema_version": "rq014-g2-resource-pilot-receipt-v1",
            "operation": "rq014_g2_resource_pilot",
            "run_id": run_id,
            "status": "FAIL",
            "rating_access": "NONE",
            "rating_join": "NONE",
            "observed_rating_statistics": "NONE",
            "pilot_scope": {
                "non_m3_stages": list(NON_M3_STAGES),
                "m3_stage_enabled": True,
                "env_v4_required": True,
                "m3_cost_estimate": "MEASURED",
            },
            "cell_selection": None,
            "measurements": [],
            "cell_details": {},
            "per_cell_serial_timings": {},
            "parallel_execution": {
                "configured_max_workers": MAX_PARALLEL_WORKERS,
                "actual_worker_count": 0,
                "selected_cell_count": 0,
                "worker_model": "PROCESS_POOL",
                "worker_thread_limits": WORKER_THREAD_LIMITS,
                "shared_source_load_walltime_seconds": M3_COST_SENTINEL,
                "shared_m3_model_load_walltime_seconds": M3_COST_SENTINEL,
                "worker_pool_walltime_seconds": M3_COST_SENTINEL,
                "aggregate_walltime_seconds": M3_COST_SENTINEL,
            },
            "failure_taxonomy": taxonomy,
            "failed_stage_count": 1,
            "measured_stage_count": 0,
            "failure_rate": 1.0,
            "failure_detail": {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            "projection": {
                "projected_non_m3_cpu_hours": M3_COST_SENTINEL,
                "projected_non_m3_serial_walltime_hours": M3_COST_SENTINEL,
                "projected_non_m3_parallel_walltime_hours": M3_COST_SENTINEL,
                "m3_cost_estimate": M3_COST_SENTINEL,
                "combined_g2r_cost_estimate": M3_COST_SENTINEL,
                "env_v4_required": True,
            },
            "lineage": {},
        }
    lineage["sanitization_receipt_sha256"] = sha256_file(sanitization_receipt_path)
    lineage["lane_v3_sha256"] = sha256_file(lane_path)
    lineage["m3_parity_fixture_sha256"] = sha256_file(m3_parity_fixture_path)
    measurements: list[dict[str, Any]] = []
    cell_totals: dict[str, dict[str, float]] = {}
    cell_details: dict[str, dict[str, Any]] = {}
    per_cell_serial_timings: dict[str, dict[str, Any]] = {}
    selected_cell_ids = (
        selection["lightest_cell_id"],
        selection["heaviest_cell_id"],
    )
    execution_started = time.monotonic()
    sources, source_measurement = _measure(
        "source_load",
        "SHARED",
        lambda: _load_sources(bundle_root, mapping_manifest_path),
    )
    measurements.append(source_measurement)
    scorer, m3_load_measurement = _measure(
        "m3_model_load", "SHARED", lambda: _load_m3_model(m3_artifact_path)
    )
    measurements.append(m3_load_measurement)
    actual_worker_count = 0
    worker_pool_walltime: float | str = M3_COST_SENTINEL
    cell_results: list[dict[str, Any]] = []
    if sources is not None and scorer is not None:
        actual_worker_count = min(MAX_PARALLEL_WORKERS, len(selected_cell_ids))
        worker_pool_started = time.monotonic()
        _SHARED_SOURCES = sources
        _SHARED_M3 = {"scorer": scorer}
        try:
            with ProcessPoolExecutor(
                max_workers=actual_worker_count,
                mp_context=get_context("fork"),
                initializer=_configure_worker_thread_limits,
            ) as executor:
                cell_results = list(
                    executor.map(
                        _run_pilot_cell,
                        selected_cell_ids,
                    )
                )
        finally:
            _SHARED_SOURCES = None
            _SHARED_M3 = None
        worker_pool_walltime = time.monotonic() - worker_pool_started
    aggregate_walltime_seconds = time.monotonic() - execution_started
    for result in cell_results:
        cell_id = result["cell_id"]
        measurements.extend(result["measurements"])
        per_cell_serial_timings[cell_id] = result["serial_timing"]
        if result["detail"] is not None:
            cell_details[cell_id] = result["detail"]
        if result["projection_basis"] is not None:
            cell_totals[cell_id] = result["projection_basis"]
    failures = [row for row in measurements if row["status"] == "FAIL"]
    taxonomy = {code: 0 for code in FAILURE_CODES if code != "NONE"}
    for row in failures:
        taxonomy[row["failure_code"]] += 1
    if not failures and len(cell_totals) == 2:
        source_cpu = source_measurement["cpu_seconds"]
        source_wall = source_measurement["walltime_seconds"]
        per_cell_cpu = max(value["per_cell_cpu_seconds"] for value in cell_totals.values())
        per_cell_wall = max(value["per_cell_walltime_seconds"] for value in cell_totals.values())
        non_m3_cpu: float | str = (source_cpu + 320 * per_cell_cpu) / 3600.0
        non_m3_wall: float | str = (source_wall + 320 * per_cell_wall) / 3600.0
        non_m3_parallel_wall: float | str = (
            source_wall + math.ceil(320 / MAX_PARALLEL_WORKERS) * per_cell_wall
        ) / 3600.0
        per_cell_m3_cpu = max(value["per_cell_m3_cpu_seconds"] for value in cell_totals.values())
        per_cell_m3_wall = max(value["per_cell_m3_walltime_seconds"] for value in cell_totals.values())
        m3_cpu: float | str = (m3_load_measurement["cpu_seconds"] + 320 * per_cell_m3_cpu) / 3600.0
        m3_wall: float | str = (m3_load_measurement["walltime_seconds"] + 320 * per_cell_m3_wall) / 3600.0
        m3_parallel_wall: float | str = (m3_load_measurement["walltime_seconds"] + math.ceil(320 / MAX_PARALLEL_WORKERS) * per_cell_m3_wall) / 3600.0
        combined_cpu: float | str = non_m3_cpu + m3_cpu
        combined_wall: float | str = non_m3_wall + m3_wall
        combined_parallel_wall: float | str = non_m3_parallel_wall + m3_parallel_wall
    else:
        non_m3_cpu = M3_COST_SENTINEL
        non_m3_wall = M3_COST_SENTINEL
        non_m3_parallel_wall = M3_COST_SENTINEL
        m3_cpu = m3_wall = m3_parallel_wall = M3_COST_SENTINEL
        combined_cpu = combined_wall = combined_parallel_wall = M3_COST_SENTINEL
    projection = {
        "formula": "shared_source_load_once + 320 * max_endpoint_window_assembly_plus_feature_prep",
        "parallel_formula": "shared_source_load_once + ceil(320 / 16) * max_endpoint_window_assembly_plus_feature_prep",
        "m3_formula": "shared_m3_model_load_once + 320 * max_endpoint_m3_scoring",
        "m3_parallel_formula": "shared_m3_model_load_once + ceil(320 / 16) * max_endpoint_m3_scoring",
        "combined_formula": "non_m3_projection + m3_projection",
        "parallel_worker_count": MAX_PARALLEL_WORKERS,
        "projected_non_m3_cpu_hours": non_m3_cpu,
        "projected_non_m3_serial_walltime_hours": non_m3_wall,
        "projected_non_m3_parallel_walltime_hours": non_m3_parallel_wall,
        "projected_m3_cpu_hours": m3_cpu,
        "projected_m3_serial_walltime_hours": m3_wall,
        "projected_m3_parallel_walltime_hours": m3_parallel_wall,
        "m3_cost_estimate": "MEASURED" if not failures and len(cell_totals) == 2 else M3_COST_SENTINEL,
        "projected_combined_cpu_hours": combined_cpu,
        "projected_combined_serial_walltime_hours": combined_wall,
        "projected_combined_parallel_walltime_hours": combined_parallel_wall,
        "combined_g2r_cost_estimate": "MEASURED" if not failures and len(cell_totals) == 2 else M3_COST_SENTINEL,
        "env_v4_required": True,
    }
    return {
        "schema_version": "rq014-g2-resource-pilot-receipt-v1",
        "operation": "rq014_g2_resource_pilot",
        "run_id": run_id,
        "status": "PASS" if not failures and len(cell_totals) == 2 else "FAIL",
        "rating_access": "NONE",
        "rating_join": "NONE",
        "observed_rating_statistics": "NONE",
        "pilot_scope": {
            "non_m3_stages": list(NON_M3_STAGES),
            "m3_stage_enabled": True,
            "env_v4_required": True,
            "m3_cost_estimate": "MEASURED",
        },
        "cell_selection": selection,
        "measurements": measurements,
        "cell_details": cell_details,
        "per_cell_serial_timings": per_cell_serial_timings,
        "parallel_execution": {
            "configured_max_workers": MAX_PARALLEL_WORKERS,
            "actual_worker_count": actual_worker_count,
            "selected_cell_count": len(selected_cell_ids),
            "worker_model": "PROCESS_POOL",
            "worker_thread_limits": WORKER_THREAD_LIMITS,
            "shared_source_load_walltime_seconds": source_measurement[
                "walltime_seconds"
            ],
            "shared_m3_model_load_walltime_seconds": m3_load_measurement[
                "walltime_seconds"
            ],
            "worker_pool_walltime_seconds": worker_pool_walltime,
            "aggregate_walltime_seconds": aggregate_walltime_seconds,
        },
        "failure_taxonomy": taxonomy,
        "failed_stage_count": len(failures),
        "measured_stage_count": len(measurements),
        "failure_rate": len(failures) / len(measurements),
        "projection": projection,
        "lineage": lineage,
    }
