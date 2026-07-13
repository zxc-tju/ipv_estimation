#!/usr/bin/env python3
"""Derive the frozen rating-blind WOD CP/HO/MP/F scene mapping."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


HEX64 = frozenset("0123456789abcdef")
PATH_TYPES = ("CP", "HO", "MP", "F")
INTENTS = {1: "GO_STRAIGHT", 2: "GO_LEFT", 3: "GO_RIGHT"}
MAPPING_FILENAME = "wod_path_type_mapping.csv"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "distribution_summary.json"
EXPECTED_FILE_MANIFEST_SHA256 = (
    "4c41172fd16f00d48187df711ad6435063d80c4bbeb81c5e6e07025d63c78ef9"
)
EXPECTED_SANITIZATION_RECEIPT_SHA256 = (
    "4dfc81056fe97db66d0ba0df04955a9c7c3a5464237010f298f3fa64363911a3"
)

BUNDLE_HEADERS = {
    "blind_scene_manifest.csv": [
        "segment_id",
        "tstar_context_step",
        "source_shard_id",
        "scenario_cluster",
        "path_type",
        "route_intent_code",
        "route_intent_name",
        "coordinate_frame",
        "native_frame_rate_hz",
        "state_rate_hz",
        "candidate_rate_hz",
        "candidate_count",
        "candidate_geometry_available",
        "ego_future_state_count",
        "tstar_ego_pose_element_count",
        "structural_status",
        "candidate_set_sha256",
    ],
    "candidate_states.csv": [
        "segment_id",
        "tstar_context_step",
        "candidate_id",
        "candidate_ordinal",
        "geometry_sha256",
        "raw_sample_index",
        "raw_time_s",
        "dropped_as_tstar_duplicate",
        "included_in_effective_future",
        "effective_sample_index",
        "effective_time_s",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "vel_x_mps",
        "vel_y_mps",
        "accel_x_mps2",
        "accel_y_mps2",
    ],
    "counterpart_tracks.csv": [
        "segment_id",
        "tstar_context_step",
        "counterpart_track_id",
        "context_step",
        "time_s",
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "class_name",
        "detector_confidence",
    ],
    "ego_future_states.csv": [
        "segment_id",
        "tstar_context_step",
        "sample_index",
        "time_s",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "vel_x_mps",
        "vel_y_mps",
        "accel_x_mps2",
        "accel_y_mps2",
    ],
    "ego_history_states.csv": [
        "segment_id",
        "tstar_context_step",
        "sample_index",
        "time_s",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "vel_x_mps",
        "vel_y_mps",
        "accel_x_mps2",
        "accel_y_mps2",
    ],
    "structural_attrition.csv": [
        "segment_id",
        "stage",
        "reason_code",
        "source_receipt_id",
    ],
    "tstar_ego_pose.csv": [
        "segment_id",
        "tstar_context_step",
        "matrix_row",
        "matrix_column",
        "value",
    ],
}
EXPECTED_BUNDLE_FILES = frozenset(
    (*BUNDLE_HEADERS, "file_manifest.json", "sanitization_receipt.json")
)


class FreezeError(ValueError):
    """The rating-blind source or requested output violates the frozen contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX64 for character in value)
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FreezeError(f"Non-finite JSON token: {value}")


def _load_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"Malformed JSON: {label}") from exc
    if not isinstance(value, dict):
        raise FreezeError(f"JSON root is not an object: {label}")
    if canonical_json_bytes(value) != payload:
        raise FreezeError(f"JSON is not canonical sorted compact UTF-8 plus LF: {label}")
    return value


def _read_regular_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise FreezeError(f"Cannot open regular no-follow input: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FreezeError(f"Input is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise FreezeError(f"Input changed during retained-descriptor read: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise FreezeError(f"Short read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _read_csv(payload: bytes, name: str) -> list[dict[str, str]]:
    if b"\r" in payload or b"\x00" in payload or not payload.endswith(b"\n"):
        raise FreezeError(f"CSV is not UTF-8 LF text: {name}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeError(f"CSV is not UTF-8: {name}") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != BUNDLE_HEADERS[name]:
            raise FreezeError(f"Unexpected CSV header: {name}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise FreezeError(f"Malformed CSV row {name}:{row_number}")
            rows.append(dict(row))
    return rows


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"Non-numeric {label}") from exc
    if not math.isfinite(result):
        raise FreezeError(f"Non-finite {label}")
    return result


def _canonical_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise FreezeError(f"Non-canonical integer {label}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FreezeError(f"Non-integer {label}") from exc
    if str(result) != str(value) or result < 0:
        raise FreezeError(f"Non-canonical integer {label}")
    return result


def _ordered_xy(
    rows: Sequence[Sequence[object]],
    label: str,
) -> list[tuple[float, float, float]]:
    converted = []
    for index, row in enumerate(rows):
        if len(row) < 3:
            raise FreezeError(f"Malformed {label} row {index}")
        converted.append(
            (
                _finite_float(row[0], f"{label} time"),
                _finite_float(row[1], f"{label} x"),
                _finite_float(row[2], f"{label} y"),
            )
        )
    if any(right[0] <= left[0] for left, right in zip(converted, converted[1:])):
        raise FreezeError(f"Non-increasing {label} time")
    return converted


def _effective_velocity(
    observations: Sequence[tuple[float, float, float, float, float]],
) -> tuple[float, float]:
    last_time = observations[-1][0]
    fit = [row for row in observations if row[0] >= last_time - 1.5]
    if len(fit) < 3:
        fit = list(observations)
    mean_time = math.fsum(row[0] for row in fit) / len(fit)
    mean_x = math.fsum(row[1] for row in fit) / len(fit)
    mean_y = math.fsum(row[2] for row in fit) / len(fit)
    denominator = math.fsum((row[0] - mean_time) ** 2 for row in fit)
    if denominator <= 1e-18:
        return 0.0, 0.0
    vx = math.fsum((row[0] - mean_time) * (row[1] - mean_x) for row in fit) / denominator
    vy = math.fsum((row[0] - mean_time) * (row[2] - mean_y) for row in fit) / denominator
    return vx, vy


def _gradient_direction(
    points: Sequence[tuple[float, float, float]],
    index: int,
) -> tuple[float, float]:
    if index == 0:
        left, right = points[0], points[1]
    elif index == len(points) - 1:
        left, right = points[-2], points[-1]
    else:
        left, right = points[index - 1], points[index + 1]
    delta_time = right[0] - left[0]
    return (right[1] - left[1]) / delta_time, (right[2] - left[2]) / delta_time


def classify_rating_blind_primitives(
    *,
    route_intent_code: object,
    route_intent_name: object,
    tstar_pose_row_major: Sequence[object],
    ego_history: Sequence[Sequence[object]],
    ego_future: Sequence[Sequence[object]],
    counterpart_observed: Sequence[Sequence[object]],
) -> dict[str, Any]:
    """Apply the frozen scene-level historical conflict-geometry rule."""

    if isinstance(route_intent_code, bool):
        raise FreezeError("Route intent code is not an integer")
    try:
        intent_code = int(route_intent_code)
    except (TypeError, ValueError) as exc:
        raise FreezeError("Route intent code is not an integer") from exc
    if INTENTS.get(intent_code) != route_intent_name:
        raise FreezeError("Route intent code/name mismatch")
    if len(tstar_pose_row_major) != 16:
        raise FreezeError("tstar pose must contain exactly 16 elements")
    for index, value in enumerate(tstar_pose_row_major):
        _finite_float(value, f"tstar pose element {index}")

    history = _ordered_xy(ego_history, "ego history")
    future = _ordered_xy(ego_future, "ego future")
    if len(history) < 2 or abs(history[-1][0]) > 1e-12:
        raise FreezeError("Ego history must contain tstar=0 and at least two points")
    if len(future) < 2 or future[0][0] <= 0.0:
        raise FreezeError("Ego future must contain at least two positive-time points")

    observations = []
    for index, row in enumerate(counterpart_observed):
        if len(row) != 5:
            raise FreezeError(f"Malformed counterpart observation {index}")
        converted = tuple(
            _finite_float(value, f"counterpart observation {index}") for value in row
        )
        if converted[0] <= 1e-12:
            observations.append(converted)
    observations.sort(key=lambda row: row[0])
    if len(observations) < 2:
        return {
            "path_type": None,
            "status": "UNMAPPED_EXCLUDED_MISSING_COUNTERPART",
        }
    if any(right[0] <= left[0] for left, right in zip(observations, observations[1:])):
        raise FreezeError("Counterpart pre/tstar times are not strictly increasing")

    cp_vx, cp_vy = _effective_velocity(observations)
    cp_speed = math.hypot(cp_vx, cp_vy)
    last = observations[-1]
    if abs(last[0]) <= 1e-12:
        cp_x0, cp_y0 = last[1], last[2]
    else:
        cp_x0 = last[1] - last[0] * last[3]
        cp_y0 = last[2] - last[0] * last[4]
    counterpart_future = [
        (time_s, cp_x0 + cp_vx * time_s, cp_y0 + cp_vy * time_s)
        for time_s, _, _ in future
    ]
    squared_distances = [
        (ego[1] - counterpart[1]) ** 2 + (ego[2] - counterpart[2]) ** 2
        for ego, counterpart in zip(future, counterpart_future)
    ]
    closest_index = min(range(len(squared_distances)), key=squared_distances.__getitem__)
    ego_vx, ego_vy = _gradient_direction(future, closest_index)
    ego_speed = math.hypot(ego_vx, ego_vy)
    if ego_speed <= 1e-9 or cp_speed <= 1e-9:
        return {
            "path_type": None,
            "status": "UNMAPPED_EXCLUDED_LOW_MOTION",
        }

    ego_dx, ego_dy = ego_vx / ego_speed, ego_vy / ego_speed
    cp_dx, cp_dy = cp_vx / cp_speed, cp_vy / cp_speed
    dot = max(-1.0, min(1.0, ego_dx * cp_dx + ego_dy * cp_dy))
    angle_deg = math.degrees(math.acos(dot))
    ego_closest = future[closest_index]
    cp_closest = counterpart_future[closest_index]
    rel_x = cp_closest[1] - ego_closest[1]
    rel_y = cp_closest[2] - ego_closest[2]
    lateral_m = abs(ego_dx * rel_y - ego_dy * rel_x)
    longitudinal_m = ego_dx * rel_x + ego_dy * rel_y

    if 45.0 <= angle_deg <= 135.0:
        path_type, status = "CP", "MAPPED_CROSSING"
    elif angle_deg > 135.0:
        if lateral_m <= 5.0:
            path_type, status = "HO", "MAPPED_OPPOSING"
        else:
            path_type, status = None, "UNMAPPED_EXCLUDED_OPPOSING_NEARBY"
    elif lateral_m <= 4.0 and longitudinal_m >= -8.0:
        path_type, status = "MP", "MAPPED_LEADING_OR_MERGING"
    elif lateral_m <= 4.0:
        path_type, status = "F", "MAPPED_SAME_LANE_OR_FOLLOWING"
    else:
        path_type, status = None, "UNMAPPED_EXCLUDED_PARALLEL_NEARBY"
    return {
        "angle_deg": angle_deg,
        "closest_future_time_s": future[closest_index][0],
        "lateral_m": lateral_m,
        "longitudinal_m": longitudinal_m,
        "path_type": path_type,
        "status": status,
    }


def _verify_bundle(
    bundle_root: Path,
    expected_file_manifest_sha256: str,
    expected_sanitization_receipt_sha256: str,
) -> tuple[dict[str, bytes], dict[str, list[dict[str, str]]]]:
    if not _is_sha256(expected_file_manifest_sha256):
        raise FreezeError("Expected file-manifest SHA-256 is malformed")
    if not _is_sha256(expected_sanitization_receipt_sha256):
        raise FreezeError("Expected sanitization-receipt SHA-256 is malformed")
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise FreezeError("Bundle root must be a regular directory, not a symlink")
    names = {path.name for path in bundle_root.iterdir()}
    if names != EXPECTED_BUNDLE_FILES:
        raise FreezeError(f"Bundle file set drift: {sorted(names ^ EXPECTED_BUNDLE_FILES)}")
    payloads = {name: _read_regular_nofollow(bundle_root / name) for name in sorted(names)}
    if sha256_bytes(payloads["file_manifest.json"]) != expected_file_manifest_sha256:
        raise FreezeError("Published file-manifest SHA-256 mismatch")
    if sha256_bytes(payloads["sanitization_receipt.json"]) != expected_sanitization_receipt_sha256:
        raise FreezeError("Published sanitization-receipt SHA-256 mismatch")
    manifest = _load_canonical_json(payloads["file_manifest.json"], "file_manifest.json")
    if set(manifest) != {"files", "schema_version"} or manifest["schema_version"] != (
        "rq014-score-stripped-file-manifest-v1"
    ):
        raise FreezeError("Score-stripped file-manifest identity drift")
    if not isinstance(manifest["files"], list):
        raise FreezeError("Score-stripped file-manifest rows are malformed")
    registered = {}
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {
            "contains_rating",
            "primary_key",
            "relative_path",
            "row_count",
            "schema_id",
            "sha256",
            "size_bytes",
        }:
            raise FreezeError("Score-stripped file-manifest row schema drift")
        name = row["relative_path"]
        if name in registered or name not in BUNDLE_HEADERS or row["contains_rating"] is not False:
            raise FreezeError("Score-stripped file-manifest path or rating flag drift")
        if not _is_sha256(row["sha256"]):
            raise FreezeError(f"Malformed registered SHA-256: {name}")
        registered[name] = row
    if set(registered) != set(BUNDLE_HEADERS):
        raise FreezeError("Score-stripped file-manifest coverage drift")

    csv_rows = {}
    for name in BUNDLE_HEADERS:
        row = registered[name]
        payload = payloads[name]
        if len(payload) != row["size_bytes"] or sha256_bytes(payload) != row["sha256"]:
            raise FreezeError(f"Published CSV byte mismatch: {name}")
        csv_rows[name] = _read_csv(payload, name)
        if len(csv_rows[name]) != row["row_count"]:
            raise FreezeError(f"Published CSV row-count mismatch: {name}")

    receipt = _load_canonical_json(
        payloads["sanitization_receipt.json"], "sanitization_receipt.json"
    )
    if receipt.get("schema_version") != "rq014-score-stripped-sanitization-v1":
        raise FreezeError("Sanitization receipt identity drift")
    if receipt.get("universe_segment_count") != 479:
        raise FreezeError("Sanitization receipt universe drift")
    expected_hashes = {name: registered[name]["sha256"] for name in sorted(BUNDLE_HEADERS)}
    if receipt.get("output_file_hashes") != expected_hashes:
        raise FreezeError("Sanitization receipt output hash drift")
    return payloads, csv_rows


def _group_rows(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["segment_id"], row["tstar_context_step"])].append(row)
    return grouped


def _mapping_csv_bytes(rows: Sequence[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["segment_id", "tstar_context_step", "path_type"],
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(str(path), flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def derive_mapping(
    *,
    bundle_root: Path,
    output_dir: Path,
    expected_file_manifest_sha256: str,
    expected_sanitization_receipt_sha256: str,
    published_mapping_path: str,
) -> dict[str, Any]:
    """Verify the nine-file bundle and emit a no-clobber deterministic freeze candidate."""

    if not published_mapping_path.startswith("/") or not published_mapping_path.endswith(
        "/wod_path_type_mapping.csv"
    ):
        raise FreezeError("Published mapping path must be the fixed absolute mapping filename")
    payloads, csv_rows = _verify_bundle(
        bundle_root.resolve(),
        expected_file_manifest_sha256,
        expected_sanitization_receipt_sha256,
    )
    blind = csv_rows["blind_scene_manifest.csv"]
    if len(blind) != 479 or [row["segment_id"] for row in blind] != sorted(
        row["segment_id"] for row in blind
    ):
        raise FreezeError("Blind scene universe is not the exact sorted 479-row set")
    if len({row["segment_id"] for row in blind}) != 479:
        raise FreezeError("Blind scene universe contains duplicate segment IDs")
    history = _group_rows(csv_rows["ego_history_states.csv"])
    future = _group_rows(csv_rows["ego_future_states.csv"])
    pose = _group_rows(csv_rows["tstar_ego_pose.csv"])
    counterpart = _group_rows(csv_rows["counterpart_tracks.csv"])

    mapping_rows = []
    statuses: Counter[str] = Counter()
    path_types: Counter[str] = Counter()
    for row in blind:
        if row["candidate_geometry_available"] == "false":
            if row["path_type"] != "NA" or row["structural_status"] != (
                "MISSING_DECLASSIFIED_PHASE1_SCENE"
            ):
                raise FreezeError("Malformed structural-exclusion row")
            statuses["UNMAPPED_EXCLUDED_STRUCTURAL"] += 1
            continue
        if (
            row["candidate_geometry_available"] != "true"
            or row["path_type"] != "UNMAPPED"
            or row["coordinate_frame"] != "ego_at_tstar"
            or row["structural_status"] != "GEOMETRY_AVAILABLE"
        ):
            raise FreezeError("Malformed geometry-available blind row")
        tstar = _canonical_nonnegative_int(row["tstar_context_step"], "tstar_context_step")
        key = (row["segment_id"], str(tstar))
        history_rows = sorted(
            history.get(key, []), key=lambda item: _canonical_nonnegative_int(item["sample_index"], "history index")
        )
        future_rows = sorted(
            future.get(key, []), key=lambda item: _canonical_nonnegative_int(item["sample_index"], "future index")
        )
        pose_rows = sorted(
            pose.get(key, []),
            key=lambda item: (
                _canonical_nonnegative_int(item["matrix_row"], "pose row"),
                _canonical_nonnegative_int(item["matrix_column"], "pose column"),
            ),
        )
        if len(history_rows) != 16 or len(future_rows) != 20 or len(pose_rows) != 16:
            raise FreezeError(f"Safe primitive cardinality drift: {key}")
        if [int(item["sample_index"]) for item in history_rows] != list(range(16)):
            raise FreezeError(f"Ego history index drift: {key}")
        if [int(item["sample_index"]) for item in future_rows] != list(range(20)):
            raise FreezeError(f"Ego future index drift: {key}")
        if [
            (int(item["matrix_row"]), int(item["matrix_column"])) for item in pose_rows
        ] != [(matrix_row, matrix_column) for matrix_row in range(4) for matrix_column in range(4)]:
            raise FreezeError(f"tstar pose index drift: {key}")
        observed_rows = counterpart.get(key, [])
        if observed_rows and len({item["counterpart_track_id"] for item in observed_rows}) != 1:
            raise FreezeError(f"Multiple selected counterpart identities: {key}")
        result = classify_rating_blind_primitives(
            route_intent_code=row["route_intent_code"],
            route_intent_name=row["route_intent_name"],
            tstar_pose_row_major=[item["value"] for item in pose_rows],
            ego_history=[
                (item["time_s"], item["pos_x_m"], item["pos_y_m"]) for item in history_rows
            ],
            ego_future=[
                (item["time_s"], item["pos_x_m"], item["pos_y_m"]) for item in future_rows
            ],
            counterpart_observed=[
                (
                    item["time_s"],
                    item["x_m"],
                    item["y_m"],
                    item["vx_mps"],
                    item["vy_mps"],
                )
                for item in observed_rows
            ],
        )
        statuses[result["status"]] += 1
        if result["path_type"] is not None:
            path_types[result["path_type"]] += 1
            mapping_rows.append(
                {
                    "segment_id": row["segment_id"],
                    "tstar_context_step": str(tstar),
                    "path_type": result["path_type"],
                }
            )
    if sum(statuses.values()) != 479:
        raise FreezeError("Derivation did not terminate every frozen scene")
    mapping_rows.sort(key=lambda item: (item["segment_id"], int(item["tstar_context_step"])))
    mapping_payload = _mapping_csv_bytes(mapping_rows)
    mapping_sha256 = sha256_bytes(mapping_payload)
    mapping_manifest = {
        "allowed_values": list(PATH_TYPES),
        "contains_rating": False,
        "key_columns": ["segment_id", "tstar_context_step"],
        "mapping": {
            "format": "RFC4180_CSV",
            "path": published_mapping_path,
            "sha256": mapping_sha256,
            "size_bytes": len(mapping_payload),
        },
        "row_count": len(mapping_rows),
        "schema_version": "rq014-wod-path-type-mapping-manifest-v1",
        "value_column": "path_type",
    }
    manifest_payload = canonical_json_bytes(mapping_manifest)
    manifest_sha256 = sha256_bytes(manifest_payload)
    summary = {
        "algorithm_id": "rq014-wod-historical-conflict-geometry-scene-v1",
        "comparison_interpretation": (
            "The historical reference counts candidate trajectories; this freeze classifies each "
            "scene once from its actual driven ego future. Distinct units and future paths make "
            "class-count equality inappropriate, including the historical F alternatives versus "
            "the single observed scene-level F mapping."
        ),
        "contains_rating": False,
        "historical_candidate_level_reference": {
            "counts": {"CP": 90, "F": 14, "HO": 88, "MP": 36},
            "scope": "228 candidate-level rows; not an expected scene-level equality",
        },
        "mapping": {
            "path": published_mapping_path,
            "sha256": mapping_sha256,
            "size_bytes": len(mapping_payload),
        },
        "mapping_manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "size_bytes": len(manifest_payload),
        },
        "path_type_counts": {path_type: path_types[path_type] for path_type in PATH_TYPES},
        "schema_version": "rq014-wod-path-type-mapping-derivation-summary-v1",
        "source_bundle": {
            "file_manifest_sha256": expected_file_manifest_sha256,
            "files": {
                name: {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}
                for name, payload in sorted(payloads.items())
            },
            "sanitization_receipt_sha256": expected_sanitization_receipt_sha256,
        },
        "status_counts": dict(sorted(statuses.items())),
        "universe_row_count": 479,
    }
    summary_payload = canonical_json_bytes(summary)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FreezeError(f"Output directory already exists: {output_dir}")
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise FreezeError(f"Output parent must be an existing non-symlink directory: {output_dir.parent}")
    staging = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FreezeError(f"Staging directory already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        _write_exclusive(staging / MAPPING_FILENAME, mapping_payload)
        _write_exclusive(staging / MANIFEST_FILENAME, manifest_payload)
        _write_exclusive(staging / SUMMARY_FILENAME, summary_payload)
        staging.rename(output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-file-manifest-sha256",
        default=EXPECTED_FILE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-sanitization-receipt-sha256",
        default=EXPECTED_SANITIZATION_RECEIPT_SHA256,
    )
    parser.add_argument("--published-mapping-path", required=True)
    args = parser.parse_args()
    summary = derive_mapping(
        bundle_root=args.bundle_root,
        output_dir=args.output_dir,
        expected_file_manifest_sha256=args.expected_file_manifest_sha256,
        expected_sanitization_receipt_sha256=args.expected_sanitization_receipt_sha256,
        published_mapping_path=args.published_mapping_path,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
