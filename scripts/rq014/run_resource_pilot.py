#!/usr/bin/env python3
"""Measure the rating-blind, non-M3 portion of the RQ014 lane-v3 grid."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable


CELL_SELECTION_RULE_ID = "LANE_V3_NON_M3_COST_EXTREMES_V1"
NON_M3_STAGES = ("source_load", "window_assembly", "feature_prep")
M3_COST_SENTINEL = "EXPLICITLY_UNMEASURED"
FAILURE_CODES = (
    "NONE",
    "INPUT_CONTRACT_FAILURE",
    "SOURCE_LOAD_FAILURE",
    "WINDOW_ASSEMBLY_FAILURE",
    "FEATURE_PREP_FAILURE",
)
LIGHTEST_CELL_ID = "RR3-R04N-CH-W10-H20-NEX_MEAN"
HEAVIEST_CELL_ID = "RR3-R10L-TF-HFEAS-NEX_MEAN"


class PilotError(ValueError):
    """Raised when a pilot contract or rating-blind input fails closed."""


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
    sampling = bank.get("sampling_axis")
    recipes = bank.get("temporal_axis", {}).get("recipes")
    horizons = bank.get("horizon_axis")
    readouts = bank.get("readout_axis")
    if not all(isinstance(axis, list) and axis for axis in (sampling, recipes, horizons, readouts)):
        raise PilotError("Lane v3 pilot axes are missing or empty")
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
    if expected != 320 or len(cells) != expected or len(set(cells)) != expected:
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


def _interpolate(points: list[tuple[float, float, float]], target: float) -> tuple[float, float, float]:
    times = [point[0] for point in points]
    index = bisect_right(times, target)
    if index == 0 or index == len(points) + 1:
        raise PilotError("Interpolation target leaves observed support")
    if index <= len(points) - 1 and math.isclose(points[index - 1][0], target, abs_tol=1e-9):
        return (target, points[index - 1][1], points[index - 1][2])
    if index == len(points):
        if math.isclose(points[-1][0], target, abs_tol=1e-9):
            return (target, points[-1][1], points[-1][2])
        raise PilotError("Interpolation target leaves observed support")
    left, right = points[index - 1], points[index]
    fraction = (target - left[0]) / (right[0] - left[0])
    return (
        target,
        left[1] + fraction * (right[1] - left[1]),
        left[2] + fraction * (right[2] - left[2]),
    )


def _resample(points: list[tuple[float, float, float]], rate_hz: int) -> list[tuple[float, float, float]]:
    if rate_hz == 4:
        return points
    first_tick = math.ceil((points[0][0] * rate_hz) - 1e-9)
    last_tick = math.floor((points[-1][0] * rate_hz) + 1e-9)
    return [_interpolate(points, tick / rate_hz) for tick in range(first_tick, last_tick + 1)]


def _assemble_windows(sources: dict[str, Any], cell_id: str) -> list[tuple[list[Any], list[Any]]]:
    parts = cell_id.split("-")
    sampling_id = parts[1]
    temporal_id = "-".join(parts[2:-2])
    horizon_id = parts[-2]
    rate_hz = 4 if sampling_id == "R04N" else 10
    windows: list[tuple[list[Any], list[Any]]] = []
    for (segment_id, _candidate_id), future in sources["candidate"].items():
        history = sources["history"].get((segment_id,))
        counterpart = sources["counterpart"].get((segment_id,))
        if not history or not counterpart or not future:
            continue
        branch = _resample(history + future, rate_hz)
        other = _resample(counterpart, rate_hz)
        h_common_tick = min(
            math.floor(branch[-1][0] * rate_hz + 1e-9),
            math.floor(other[-1][0] * rate_hz + 1e-9),
        )
        maximum_tick = min(h_common_tick, 2 * rate_hz) if horizon_id == "H20" else h_common_tick
        if maximum_tick < rate_hz:
            continue
        for tick in range(rate_hz, maximum_tick + 1):
            tau = tick / rate_hz
            if temporal_id == "CH-W10":
                lower, upper = tau - 1.0, tau
            elif temporal_id == "TF":
                lower, upper = 0.0, h_common_tick / rate_hz
            else:
                raise PilotError(f"Pilot selected an unsupported temporal endpoint: {temporal_id}")
            branch_window = [point for point in branch if lower - 1e-9 <= point[0] <= upper + 1e-9]
            other_window = [point for point in other if lower - 1e-9 <= point[0] <= upper + 1e-9]
            if len(branch_window) >= 3 and len(other_window) >= 3:
                windows.append((branch_window, other_window))
    if not windows:
        raise PilotError(f"No complete rating-blind windows for {cell_id}")
    return windows


def _derive_state(points: list[tuple[float, float, float]]) -> list[tuple[float, ...]]:
    velocity: list[tuple[float, float]] = []
    for index in range(len(points)):
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        if left == right:
            raise PilotError("Window-local derivative has fewer than two points")
        dt = points[right][0] - points[left][0]
        velocity.append(((points[right][1] - points[left][1]) / dt, (points[right][2] - points[left][2]) / dt))
    states: list[tuple[float, ...]] = []
    for index, point in enumerate(points):
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        dt = points[right][0] - points[left][0]
        ax = (velocity[right][0] - velocity[left][0]) / dt
        ay = (velocity[right][1] - velocity[left][1]) / dt
        heading = math.atan2(velocity[index][1], velocity[index][0])
        states.append((point[0], point[1], point[2], velocity[index][0], velocity[index][1], ax, ay, heading))
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
    export_receipt_path: Path,
    export_done_path: Path,
    preflight_receipt_path: Path,
    preflight_done_path: Path,
) -> dict[str, Any]:
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
                "m3_stage_enabled": False,
                "env_v4_required": True,
                "m3_cost_estimate": M3_COST_SENTINEL,
            },
            "cell_selection": None,
            "measurements": [],
            "cell_details": {},
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
                "m3_cost_estimate": M3_COST_SENTINEL,
                "combined_g2r_cost_estimate": M3_COST_SENTINEL,
                "env_v4_required": True,
            },
            "lineage": {},
        }
    lineage["sanitization_receipt_sha256"] = sha256_file(sanitization_receipt_path)
    lineage["lane_v3_sha256"] = sha256_file(lane_path)
    measurements: list[dict[str, Any]] = []
    cell_totals: dict[str, dict[str, float]] = {}
    cell_details: dict[str, dict[str, Any]] = {}
    for cell_id in (selection["lightest_cell_id"], selection["heaviest_cell_id"]):
        sources, source_measurement = _measure(
            "source_load", cell_id, lambda: _load_sources(bundle_root, mapping_manifest_path)
        )
        measurements.append(source_measurement)
        if sources is None:
            continue
        windows, window_measurement = _measure(
            "window_assembly", cell_id, lambda: _assemble_windows(sources, cell_id)
        )
        measurements.append(window_measurement)
        if windows is None:
            continue
        detail, feature_measurement = _measure(
            "feature_prep", cell_id, lambda: _prepare_features(windows)
        )
        measurements.append(feature_measurement)
        if detail is None:
            continue
        cell_details[cell_id] = {
            "mapped_segment_count": sources["mapped_segment_count"],
            "source_row_count": sources["source_row_count"],
            **detail,
        }
        cell_totals[cell_id] = {
            "source_cpu_seconds": source_measurement["cpu_seconds"],
            "source_walltime_seconds": source_measurement["walltime_seconds"],
            "per_cell_cpu_seconds": window_measurement["cpu_seconds"] + feature_measurement["cpu_seconds"],
            "per_cell_walltime_seconds": window_measurement["walltime_seconds"] + feature_measurement["walltime_seconds"],
        }
    failures = [row for row in measurements if row["status"] == "FAIL"]
    taxonomy = {code: 0 for code in FAILURE_CODES if code != "NONE"}
    for row in failures:
        taxonomy[row["failure_code"]] += 1
    if not failures and len(cell_totals) == 2:
        source_cpu = max(value["source_cpu_seconds"] for value in cell_totals.values())
        source_wall = max(value["source_walltime_seconds"] for value in cell_totals.values())
        per_cell_cpu = max(value["per_cell_cpu_seconds"] for value in cell_totals.values())
        per_cell_wall = max(value["per_cell_walltime_seconds"] for value in cell_totals.values())
        non_m3_cpu: float | str = (source_cpu + 320 * per_cell_cpu) / 3600.0
        non_m3_wall: float | str = (source_wall + 320 * per_cell_wall) / 3600.0
    else:
        non_m3_cpu = M3_COST_SENTINEL
        non_m3_wall = M3_COST_SENTINEL
    projection = {
        "formula": "max_source_load_once + 320 * max_endpoint_window_assembly_plus_feature_prep",
        "projected_non_m3_cpu_hours": non_m3_cpu,
        "projected_non_m3_serial_walltime_hours": non_m3_wall,
        "m3_cost_estimate": M3_COST_SENTINEL,
        "combined_g2r_cost_estimate": M3_COST_SENTINEL,
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
            "m3_stage_enabled": False,
            "env_v4_required": True,
            "m3_cost_estimate": M3_COST_SENTINEL,
        },
        "cell_selection": selection,
        "measurements": measurements,
        "cell_details": cell_details,
        "failure_taxonomy": taxonomy,
        "failed_stage_count": len(failures),
        "measured_stage_count": len(measurements),
        "failure_rate": len(failures) / len(measurements),
        "projection": projection,
        "lineage": lineage,
    }
