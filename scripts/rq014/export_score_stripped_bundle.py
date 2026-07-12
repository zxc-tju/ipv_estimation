#!/usr/bin/env python3
"""Export the audited canonical RQ014 score-stripped trajectory bundle.

This is a declassification utility, not a G2 operation. It may read only the
legacy score-omitting Phase-1 scene bundles, the structural readiness table,
and selected counterpart tracks. Its canonical outputs are the only WOD-E2E
trajectory inputs that G2 may mount.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import stat
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from scripts.rq014.preflight import validate_score_stripped_bundle


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_RATING_RELATED_SOURCE_KEYS = {"ratings_blind", "preference_trajectories"}
FORBIDDEN_SOURCE_KEY_TOKENS = {
    "candidate_score",
    "candidate_scores",
    "human_rating",
    "preference_score",
    "rating_order",
    "rating_rank",
    "score_order",
    "score_rank",
    "trajectory_rating",
}
STATE_FIELDS = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y")
STATE_OUTPUT_FIELDS = {
    "pos_x": "pos_x_m",
    "pos_y": "pos_y_m",
    "pos_z": "pos_z_m",
    "vel_x": "vel_x_mps",
    "vel_y": "vel_y_mps",
    "accel_x": "accel_x_mps2",
    "accel_y": "accel_y_mps2",
}
ROUTE_INTENTS = {
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}
TOP_LEVEL_KEYS = {"metadata", "scenes", "prep_abstentions"}
METADATA_KEYS = {
    "n_shards",
    "native_dt_s",
    "notes",
    "phase0_provenance_json",
    "post_frames",
    "ratings_blind",
    "segment_root",
    "shard_index",
    "target_csv",
    "version",
}
SCENE_KEYS = {
    "scene_id",
    "segment_key",
    "context_name",
    "context_name_tstar",
    "source_shard",
    "record_index",
    "tstar_context_step",
    "timestamp_micros",
    "tstar_ego_pose",
    "intent",
    "intent_name",
    "past_states",
    "future_states",
    "preference_trajectories",
    "forward_cameras",
    "post_window",
}
READINESS_SEGMENT_FIELD = "segment_key"
READINESS_CLUSTER_FIELD = "scenario_cluster"
READINESS_NATIVE_HZ_FIELD = "native_cadence_hz"
READINESS_SOURCE_HEADER = [
    "segment_key",
    "scenario_cluster",
    "index_source",
    "frame_count",
    "unique_context_steps",
    "duplicate_context_step_rows",
    "min_context_step",
    "max_context_step",
    "context_step_span",
    "context_step_delta_min",
    "context_step_delta_mode",
    "context_step_delta_max",
    "native_cadence_s",
    "native_cadence_hz",
    "max_contiguous_observed_run_frames",
    "max_contiguous_observed_run_start_step",
    "max_contiguous_observed_run_end_step",
    "max_contiguous_observed_run_span_s",
    "best_tstar_step_for_pre_run",
    "max_contiguous_pre_tstar_frames",
    "max_contiguous_pre_tstar_start_step",
    "max_contiguous_pre_tstar_end_step",
    "max_contiguous_pre_tstar_frame_to_frame_span_s",
    "pre_history_duration_to_tstar_s",
    "meets_ge10_contiguous_pre_tstar_frames",
    "forward_arc_expected_cameras",
    "forward_images_per_frame_min",
    "forward_images_per_frame_mode",
    "forward_images_per_frame_max",
    "forward_calibrations_per_frame_min",
    "forward_calibrations_per_frame_mode",
    "forward_calibrations_per_frame_max",
    "rows_with_5_forward_images",
    "rows_with_5_forward_calibrations",
    "forward_arc_5cam_complete_all_rows",
    "past_states_len_min",
    "past_states_len_mode",
    "past_states_len_max",
    "future_states_len_min",
    "future_states_len_mode",
    "future_states_len_max",
    "preference_trajectory_count_min",
    "preference_trajectory_count_mode",
    "preference_trajectory_count_max",
    "final_tfrecord_bytes",
    "final_index_bytes",
    "shard_tfrecord_count",
    "shard_index_count",
    "shard_tfrecord_bytes",
    "segment_dir_bytes",
]
COUNTERPART_SOURCE_HEADER = [
    "segment_key",
    "tstar_context_step",
    "counterpart_track_id",
    "context_step",
    "t_rel_s",
    "x",
    "y",
    "vx",
    "vy",
    "class_name",
    "score",
]


class ExportError(ValueError):
    """Raised when the score-stripping boundary fails closed."""


class RestrictedUnpickler(pickle.Unpickler):
    """Reject all pickle global/class construction opcodes."""

    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - exercised by malicious fixture
        raise ExportError(f"Pickle global is forbidden: {module}.{name}")


def restricted_load(handle: BinaryIO) -> Any:
    return RestrictedUnpickler(handle).load()


def _normalized_semantic(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def _reject_rating_semantic_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExportError(f"Non-string mapping key in {label}")
            normalized = _normalized_semantic(key)
            forbidden = (
                normalized in FORBIDDEN_SOURCE_KEY_TOKENS
                or "human_rating" in normalized
                or "preference_score" in normalized
                or normalized.startswith("rating_")
                or normalized.endswith("_rating")
            )
            if forbidden and normalized not in ALLOWED_RATING_RELATED_SOURCE_KEYS:
                raise ExportError(f"Rating-semantic source key is forbidden: {label}.{key}")
            _reject_rating_semantic_keys(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_rating_semantic_keys(item, f"{label}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_reviewed_source_bytes(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, str]:
    """Read and bind one reviewed source through exactly one no-follow descriptor."""

    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ExportError(f"Invalid expected size for {label}")
    if not isinstance(expected_sha256, str) or HEX64.fullmatch(expected_sha256) is None:
        raise ExportError(f"Invalid expected SHA-256 for {label}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ExportError("O_NOFOLLOW is required at the declassification boundary")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ExportError(f"Cannot open reviewed source without following links: {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExportError(f"Reviewed source is not a regular file: {label}")
        if before.st_size != expected_size:
            raise ExportError(f"Reviewed source size mismatch: {label}")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ExportError(f"Reviewed source ended before expected size: {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ExportError(f"Reviewed source exceeds expected size: {label}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    continuity_fields = ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in continuity_fields):
        raise ExportError(f"Reviewed source descriptor identity drift: {label}")
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or digest != expected_sha256:
        raise ExportError(f"Reviewed source SHA-256 mismatch: {label}")
    return payload, digest


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ExportError(f"{label} is non-finite")
    return parsed


def canonical_float(value: Any, label: str) -> str:
    return repr(finite_float(value, label))


def optional_float(values: list[Any], index: int, label: str) -> str:
    if not values:
        return "NA"
    if len(values) <= index:
        raise ExportError(f"{label} length mismatch")
    return canonical_float(values[index], label)


def state_output_values(
    state: dict[str, list[Any]], index: int, label: str
) -> dict[str, str]:
    return {
        output_field: optional_float(state[source_field], index, f"{label}.{source_field}")
        for source_field, output_field in STATE_OUTPUT_FIELDS.items()
    }


def trajectory_geometry_sha256(trajectory: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in STATE_FIELDS:
        values = trajectory[field]
        if not isinstance(values, list):
            raise ExportError(f"Candidate {field} must be a list")
        digest.update(field.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(len(values)).encode("ascii"))
        digest.update(b":")
        for value in values:
            digest.update(format(finite_float(value, field), ".17g").encode("ascii"))
            digest.update(b",")
        digest.update(b";")
    return digest.hexdigest()


def candidate_set_sha256(segment_id: str, tstar: int, hashes: list[str]) -> str:
    payload = {
        "candidate_geometry_sha256": hashes,
        "segment_id": segment_id,
        "tstar_context_step": tstar,
    }
    return hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()


def _validate_state_object(
    value: Any,
    label: str,
    *,
    expected_length: int | None = None,
    min_length: int = 1,
    max_length: int | None = None,
) -> dict[str, list[Any]]:
    if not isinstance(value, dict) or set(value) != set(STATE_FIELDS):
        raise ExportError(f"Unexpected {label} state keys")
    for field in STATE_FIELDS:
        if not isinstance(value[field], list):
            raise ExportError(f"{label}.{field} must be a list")
        for item in value[field]:
            finite_float(item, f"{label}.{field}")
    length = len(value["pos_x"])
    if len(value["pos_y"]) != length or length < min_length:
        raise ExportError(f"{label} requires equal nonempty pos_x/pos_y")
    if expected_length is not None and length != expected_length:
        raise ExportError(f"{label} must contain exactly {expected_length} states")
    if max_length is not None and length > max_length:
        raise ExportError(f"{label} contains more than {max_length} states")
    for field in STATE_FIELDS[2:]:
        if len(value[field]) not in {0, length}:
            raise ExportError(f"{label}.{field} must be empty or match pos_x length")
    return value


def _validate_pose(value: Any, label: str) -> list[list[Any]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ExportError(f"{label} must be an exact 4x4 list")
    for row_index, row in enumerate(value):
        for column_index, item in enumerate(row):
            finite_float(item, f"{label}[{row_index}][{column_index}]")
    return value


def _load_readiness(payload: bytes) -> tuple[list[str], dict[str, dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExportError("Readiness TSV must be UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != READINESS_SOURCE_HEADER:
            raise ExportError(f"Unexpected readiness TSV header: {reader.fieldnames}")
        rows: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ExportError(f"Malformed readiness TSV row {row_number}")
            segment = row[READINESS_SEGMENT_FIELD]
            if not SAFE_SEGMENT.fullmatch(segment) or segment in rows:
                raise ExportError("Unsafe or duplicate readiness segment ID")
            rows[segment] = {key: str(value) for key, value in row.items()}
    if len(rows) != 479:
        raise ExportError(f"Expected 479 readiness segments, found {len(rows)}")
    return sorted(rows), rows


def _load_scenes(
    bundle_sources: list[tuple[Path, bytes, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    scenes: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for index, (path, payload, digest) in enumerate(bundle_sources):
        source_hashes[f"phase1_scene_bundle_{index:02d}"] = digest
        bundle = restricted_load(io.BytesIO(payload))
        if not isinstance(bundle, dict) or set(bundle) != TOP_LEVEL_KEYS:
            raise ExportError(f"Unexpected Phase-1 bundle shape: {path}")
        _reject_rating_semantic_keys(bundle, f"phase1_bundle_{index:02d}")
        if (
            not isinstance(bundle["metadata"], dict)
            or set(bundle["metadata"]) != METADATA_KEYS
            or not isinstance(bundle["scenes"], list)
            or not isinstance(bundle["prep_abstentions"], list)
        ):
            raise ExportError(f"Malformed Phase-1 bundle: {path}")
        if bundle["metadata"]["ratings_blind"] is not True:
            raise ExportError(f"Phase-1 bundle is not attested rating-blind: {path}")
        if bundle["prep_abstentions"]:
            raise ExportError(f"Audited Phase-1 bundle must have zero prep abstentions: {path}")
        if finite_float(bundle["metadata"]["native_dt_s"], "bundle metadata native_dt_s") != 0.1:
            raise ExportError("Phase-1 bundle metadata native_dt_s must equal 0.1")
        for scene in bundle["scenes"]:
            if not isinstance(scene, dict) or set(scene) != SCENE_KEYS:
                raise ExportError("Unexpected Phase-1 scene keys")
            segment = scene["segment_key"]
            if not isinstance(segment, str) or not SAFE_SEGMENT.fullmatch(segment) or segment in scenes:
                raise ExportError("Unsafe or duplicate Phase-1 segment")
            if not isinstance(scene["source_shard"], str) or not SAFE_SEGMENT.fullmatch(
                scene["source_shard"]
            ):
                raise ExportError(f"Unsafe or empty source shard ID: {segment}")
            candidates = scene["preference_trajectories"]
            if not isinstance(candidates, list) or len(candidates) != 3:
                raise ExportError(f"Phase-1 scene does not contain exactly three candidates: {segment}")
            for ordinal, candidate in enumerate(candidates, start=1):
                _validate_state_object(
                    candidate,
                    f"{segment}.candidate{ordinal}",
                    min_length=2,
                    max_length=21,
                )
            _validate_state_object(
                scene["past_states"],
                f"{segment}.past_states",
                expected_length=16,
            )
            _validate_state_object(
                scene["future_states"],
                f"{segment}.future_states",
                expected_length=20,
            )
            _validate_pose(scene["tstar_ego_pose"], f"{segment}.tstar_ego_pose")
            intent = scene["intent"]
            if isinstance(intent, bool) or not isinstance(intent, int) or intent not in ROUTE_INTENTS:
                raise ExportError(f"Unexpected route intent code: {segment}")
            if scene["intent_name"] != ROUTE_INTENTS[intent]:
                raise ExportError(f"Route intent code/name mismatch: {segment}")
            if not isinstance(scene["post_window"], dict):
                raise ExportError(f"Malformed post_window: {segment}")
            scenes[segment] = scene
    return scenes, source_hashes


def _load_counterparts(payload: bytes, universe: set[str]) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExportError("Counterpart CSV must be UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COUNTERPART_SOURCE_HEADER:
            raise ExportError(f"Unexpected counterpart source header: {reader.fieldnames}")
        rows: list[dict[str, str]] = []
        keys: set[tuple[str, str, str]] = set()
        for row_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ExportError(f"Malformed counterpart CSV row {row_number}")
            segment = raw["segment_key"]
            if segment not in universe:
                raise ExportError("Counterpart row references unknown segment")
            key = (segment, raw["counterpart_track_id"], raw["context_step"])
            if key in keys:
                raise ExportError(f"Duplicate counterpart key: {key}")
            keys.add(key)
            rows.append(
                {
                    "segment_id": segment,
                    "tstar_context_step": str(int(raw["tstar_context_step"])),
                    "counterpart_track_id": raw["counterpart_track_id"],
                    "context_step": str(int(raw["context_step"])),
                    "time_s": canonical_float(raw["t_rel_s"], "counterpart time"),
                    "x_m": canonical_float(raw["x"], "counterpart x"),
                    "y_m": canonical_float(raw["y"], "counterpart y"),
                    "vx_mps": "NA" if raw["vx"] in {"", "NA"} else canonical_float(raw["vx"], "counterpart vx"),
                    "vy_mps": "NA" if raw["vy"] in {"", "NA"} else canonical_float(raw["vy"], "counterpart vy"),
                    "class_name": raw["class_name"],
                    "detector_confidence": canonical_float(raw["score"], "detector confidence"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["segment_id"], row["counterpart_track_id"], int(row["context_step"])),
    )


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ExportError(f"Output row schema mismatch for {path.name}")
            writer.writerow(row)
            count += 1
    return count


def export_bundle(
    *,
    bundle_paths: list[Path],
    readiness_path: Path,
    counterpart_path: Path,
    schema_path: Path,
    output_root: Path,
    exporter_git_commit: str,
    exporter_environment_sha256: str,
    created_at_utc: str,
    source_expectations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not HEX40.fullmatch(exporter_git_commit):
        raise ExportError("exporter_git_commit must be a lowercase 40-hex commit")
    if not HEX64.fullmatch(exporter_environment_sha256):
        raise ExportError("exporter_environment_sha256 must be a lowercase SHA-256")
    if len(bundle_paths) != 8:
        raise ExportError("Exactly eight Phase-1 scene bundles are required")
    expected_roles = {
        *(f"phase1_scene_bundle_{index:02d}" for index in range(8)),
        "rated479_structural_readiness",
        "selected_counterpart_tracks",
    }
    if not isinstance(source_expectations, dict) or set(source_expectations) != expected_roles:
        raise ExportError("Source expectations must cover the exact reviewed source roles")
    for role, expectation in source_expectations.items():
        if not isinstance(expectation, dict) or set(expectation) != {"size_bytes", "sha256"}:
            raise ExportError(f"Malformed source expectation: {role}")
    if schema_path.is_symlink() or not schema_path.is_file():
        raise ExportError(f"Regular non-symlink source required: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_version") != "rq014-score-stripped-schema-v1":
        raise ExportError("Wrong score-stripped schema")
    ordered_bundle_paths = sorted(Path(os.path.abspath(path)) for path in bundle_paths)
    bundle_sources: list[tuple[Path, bytes, str]] = []
    for index, path in enumerate(ordered_bundle_paths):
        role = f"phase1_scene_bundle_{index:02d}"
        expectation = source_expectations[role]
        payload, digest = _read_reviewed_source_bytes(
            path,
            expected_size=expectation["size_bytes"],
            expected_sha256=expectation["sha256"],
            label=role,
        )
        bundle_sources.append((path, payload, digest))
    readiness_expectation = source_expectations["rated479_structural_readiness"]
    readiness_payload, readiness_digest = _read_reviewed_source_bytes(
        Path(os.path.abspath(readiness_path)),
        expected_size=readiness_expectation["size_bytes"],
        expected_sha256=readiness_expectation["sha256"],
        label="rated479_structural_readiness",
    )
    counterpart_expectation = source_expectations["selected_counterpart_tracks"]
    counterpart_payload, counterpart_digest = _read_reviewed_source_bytes(
        Path(os.path.abspath(counterpart_path)),
        expected_size=counterpart_expectation["size_bytes"],
        expected_sha256=counterpart_expectation["sha256"],
        label="selected_counterpart_tracks",
    )
    universe, readiness = _load_readiness(readiness_payload)
    scenes, source_hashes = _load_scenes(bundle_sources)
    unknown_scenes = set(scenes) - set(universe)
    if unknown_scenes:
        raise ExportError(f"Phase-1 bundle contains unknown segments: {sorted(unknown_scenes)[:3]}")
    counterparts = _load_counterparts(counterpart_payload, set(universe))
    source_hashes["rated479_structural_readiness"] = readiness_digest
    source_hashes["selected_counterpart_tracks"] = counterpart_digest

    if output_root.is_symlink():
        raise ExportError(f"Output root may not be a symlink: {output_root}")
    if output_root.exists():
        raise ExportError(f"Output root already exists: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.partial-{os.getpid()}")
    if staging.exists():
        raise ExportError(f"Staging root already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        blind_rows: list[dict[str, str]] = []
        candidate_rows: list[dict[str, str]] = []
        history_rows: list[dict[str, str]] = []
        future_rows: list[dict[str, str]] = []
        pose_rows: list[dict[str, str]] = []
        attrition_rows: list[dict[str, str]] = []
        geometry_count = 0
        for segment in universe:
            scene = scenes.get(segment)
            readiness_row = readiness[segment]
            if scene is None:
                blind_rows.append(
                    {
                        "segment_id": segment,
                        "tstar_context_step": "NA",
                        "source_shard_id": "NA",
                        "scenario_cluster": readiness_row.get(READINESS_CLUSTER_FIELD) or "NA",
                        "path_type": "NA",
                        "route_intent_code": "NA",
                        "route_intent_name": "NA",
                        "coordinate_frame": "ego_at_tstar",
                        "native_frame_rate_hz": readiness_row.get(READINESS_NATIVE_HZ_FIELD) or "NA",
                        "state_rate_hz": "NA",
                        "candidate_rate_hz": "NA",
                        "candidate_count": "0",
                        "candidate_geometry_available": "false",
                        "ego_future_state_count": "0",
                        "tstar_ego_pose_element_count": "0",
                        "structural_status": "MISSING_DECLASSIFIED_PHASE1_SCENE",
                        "candidate_set_sha256": hashlib.sha256(b"").hexdigest(),
                    }
                )
                attrition_rows.append(
                    {
                        "segment_id": segment,
                        "stage": "DECLASSIFICATION",
                        "reason_code": "MISSING_DECLASSIFIED_PHASE1_SCENE",
                        "source_receipt_id": "phase1_bundle_declassification_v1",
                    }
                )
                continue

            geometry_count += 1
            tstar = int(scene["tstar_context_step"])
            native_dt = finite_float(scene["post_window"].get("native_dt_s"), "native_dt_s")
            if native_dt != 0.1:
                raise ExportError("Phase-1 bundle native_dt_s must be exactly 0.1")
            candidate_hashes = [trajectory_geometry_sha256(item) for item in scene["preference_trajectories"]]
            blind_rows.append(
                {
                    "segment_id": segment,
                    "tstar_context_step": str(tstar),
                    "source_shard_id": scene["source_shard"],
                    "scenario_cluster": readiness_row.get(READINESS_CLUSTER_FIELD) or "NA",
                    "path_type": "UNMAPPED",
                    "route_intent_code": str(scene["intent"]),
                    "route_intent_name": scene["intent_name"],
                    "coordinate_frame": "ego_at_tstar",
                    "native_frame_rate_hz": canonical_float(1.0 / native_dt, "native rate"),
                    "state_rate_hz": "4.0",
                    "candidate_rate_hz": "4.0",
                    "candidate_count": "3",
                    "candidate_geometry_available": "true",
                    "ego_future_state_count": str(len(scene["future_states"]["pos_x"])),
                    "tstar_ego_pose_element_count": "16",
                    "structural_status": "GEOMETRY_AVAILABLE",
                    "candidate_set_sha256": candidate_set_sha256(segment, tstar, candidate_hashes),
                }
            )
            for ordinal, (candidate, geometry_sha) in enumerate(
                zip(scene["preference_trajectories"], candidate_hashes), start=1
            ):
                last_past_x = finite_float(scene["past_states"]["pos_x"][-1], "last past x")
                last_past_y = finite_float(scene["past_states"]["pos_y"][-1], "last past y")
                first_x = finite_float(candidate["pos_x"][0], "candidate first x")
                first_y = finite_float(candidate["pos_y"][0], "candidate first y")
                drop_first = math.hypot(last_past_x - first_x, last_past_y - first_y) < 0.75
                effective_raw_indices = list(range(1 if drop_first else 0, len(candidate["pos_x"])))[:20]
                effective_lookup = {
                    raw_index: effective_index
                    for effective_index, raw_index in enumerate(effective_raw_indices, start=1)
                }
                for raw_sample_index in range(len(candidate["pos_x"])):
                    effective_index = effective_lookup.get(raw_sample_index)
                    raw_time = (
                        raw_sample_index * 0.25
                        if drop_first
                        else (raw_sample_index + 1) * 0.25
                    )
                    candidate_rows.append(
                        {
                            "segment_id": segment,
                            "tstar_context_step": str(tstar),
                            "candidate_id": f"C{ordinal}",
                            "candidate_ordinal": str(ordinal),
                            "geometry_sha256": geometry_sha,
                            "raw_sample_index": str(raw_sample_index),
                            "raw_time_s": canonical_float(raw_time, "candidate raw time"),
                            "dropped_as_tstar_duplicate": (
                                "true" if drop_first and raw_sample_index == 0 else "false"
                            ),
                            "included_in_effective_future": "true" if effective_index is not None else "false",
                            "effective_sample_index": str(effective_index) if effective_index is not None else "NA",
                            "effective_time_s": (
                                canonical_float(raw_time, "candidate effective time")
                                if effective_index is not None
                                else "NA"
                            ),
                            **state_output_values(
                                candidate,
                                raw_sample_index,
                                f"{segment}.candidate{ordinal}",
                            ),
                        }
                    )
            past = scene["past_states"]
            past_count = len(past["pos_x"])
            for sample_index in range(past_count):
                history_rows.append(
                    {
                        "segment_id": segment,
                        "tstar_context_step": str(tstar),
                        "sample_index": str(sample_index),
                        "time_s": canonical_float((sample_index - (past_count - 1)) * 0.25, "history time"),
                        **state_output_values(past, sample_index, f"{segment}.past_states"),
                    }
                )
            future = scene["future_states"]
            for sample_index in range(len(future["pos_x"])):
                future_rows.append(
                    {
                        "segment_id": segment,
                        "tstar_context_step": str(tstar),
                        "sample_index": str(sample_index),
                        "time_s": canonical_float((sample_index + 1) * 0.25, "future time"),
                        **state_output_values(future, sample_index, f"{segment}.future_states"),
                    }
                )
            for matrix_row, pose_row in enumerate(scene["tstar_ego_pose"]):
                for matrix_column, value in enumerate(pose_row):
                    pose_rows.append(
                        {
                            "segment_id": segment,
                            "tstar_context_step": str(tstar),
                            "matrix_row": str(matrix_row),
                            "matrix_column": str(matrix_column),
                            "value": canonical_float(value, "tstar ego pose"),
                        }
                    )

        output_rows = {
            "blind_scene_manifest.csv": blind_rows,
            "candidate_states.csv": candidate_rows,
            "ego_history_states.csv": history_rows,
            "ego_future_states.csv": future_rows,
            "tstar_ego_pose.csv": pose_rows,
            "counterpart_tracks.csv": counterparts,
            "structural_attrition.csv": attrition_rows,
        }
        row_counts: dict[str, int] = {}
        for name, rows in output_rows.items():
            row_counts[name] = _write_csv(staging / name, schema["files"][name]["columns"], rows)

        manifest_rows: list[dict[str, Any]] = []
        output_hashes: dict[str, str] = {}
        for name in sorted(output_rows):
            path = staging / name
            digest = sha256_file(path)
            output_hashes[name] = digest
            manifest_rows.append(
                {
                    "relative_path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "schema_id": f"{schema['schema_version']}#{name}",
                    "row_count": row_counts[name],
                    "primary_key": schema["files"][name]["primary_key"],
                    "contains_rating": False,
                }
            )
        (staging / "file_manifest.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "rq014-score-stripped-file-manifest-v1",
                    "files": manifest_rows,
                }
            )
        )
        distribution: dict[str, int] = {}
        for row in blind_rows:
            distribution[row["candidate_count"]] = distribution.get(row["candidate_count"], 0) + 1
        receipt = {
            "schema_version": "rq014-score-stripped-sanitization-v1",
            "exporter_code_sha256": sha256_file(Path(__file__).resolve()),
            "exporter_git_commit": exporter_git_commit,
            "exporter_environment_sha256": exporter_environment_sha256,
            "source_artifact_ids_and_sha256": dict(sorted(source_hashes.items())),
            "source_rating_bearing_classification": "parents originate from a rating-bearing pipeline; only audited score-omitting Phase-1 objects and structural/track tables were read",
            "source_mounts_not_exposed_to_g2": True,
            "output_file_hashes": output_hashes,
            "output_schema_hashes": {schema_path.name: sha256_file(schema_path)},
            "universe_segment_count": len(universe),
            "geometry_available_scene_count": geometry_count,
            "candidate_count_distribution": dict(sorted(distribution.items())),
            "forbidden_field_scan": 0,
            "unexpected_field_scan": 0,
            "duplicate_key_scan": 0,
            "nonfinite_value_scan": 0,
            "created_at_utc": created_at_utc,
        }
        (staging / "sanitization_receipt.json").write_bytes(canonical_json_bytes(receipt))
        validate_score_stripped_bundle(
            bundle_root=staging,
            schema_path=schema_path,
            file_manifest_path=staging / "file_manifest.json",
            receipt_path=staging / "sanitization_receipt.json",
            full_hash=True,
        )
        staging.replace(output_root)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-bundle", action="append", type=Path, required=True)
    parser.add_argument(
        "--source-expectation",
        action="append",
        nargs=3,
        metavar=("ROLE", "SIZE_BYTES", "SHA256"),
        required=True,
    )
    parser.add_argument("--readiness-tsv", type=Path, required=True)
    parser.add_argument("--counterpart-tracks", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("reports/plans/RQ014_score_stripped_schema_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-receipt-root", type=Path, required=True)
    parser.add_argument("--exporter-git-commit", required=True)
    parser.add_argument("--exporter-environment-sha256", required=True)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()
    source_expectations: dict[str, dict[str, Any]] = {}
    for role, raw_size, digest in args.source_expectation:
        if role in source_expectations or not raw_size.isdecimal():
            raise ExportError(f"Duplicate or malformed source expectation: {role}")
        source_expectations[role] = {"size_bytes": int(raw_size), "sha256": digest}
    run_receipt_root = args.run_receipt_root.resolve()
    if (
        args.run_receipt_root.is_symlink()
        or not run_receipt_root.is_dir()
        or any(run_receipt_root.iterdir())
    ):
        raise ExportError(f"Run receipt root must be an existing empty directory: {run_receipt_root}")
    output_root = args.output_root.resolve()
    receipt = export_bundle(
        bundle_paths=args.scene_bundle,
        readiness_path=Path(os.path.abspath(args.readiness_tsv)),
        counterpart_path=Path(os.path.abspath(args.counterpart_tracks)),
        schema_path=args.schema.resolve(),
        output_root=output_root,
        exporter_git_commit=args.exporter_git_commit,
        exporter_environment_sha256=args.exporter_environment_sha256,
        created_at_utc=args.created_at_utc,
        source_expectations=source_expectations,
    )
    try:
        sanitization_path = output_root / "sanitization_receipt.json"
        file_manifest_path = output_root / "file_manifest.json"
        run_receipt = {
            "schema_version": "rq014-g2-declassification-export-receipt-v1",
            "status": "PASS",
            "operation": "rq014_g2_declassification_export",
            "rating_access": "NONE",
            "score_stripped_bundle_root": str(output_root),
            "sanitization_receipt_sha256": sha256_file(sanitization_path),
            "file_manifest_sha256": sha256_file(file_manifest_path),
            "universe_segment_count": receipt["universe_segment_count"],
            "geometry_available_scene_count": receipt["geometry_available_scene_count"],
        }
        run_receipt_bytes = canonical_json_bytes(run_receipt)
        (run_receipt_root / "rq014_g2_declassification_export_receipt.json").write_bytes(
            run_receipt_bytes
        )
        (run_receipt_root / "DONE.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "rq014-managed-operation-done-v1",
                    "operation": "rq014_g2_declassification_export",
                    "receipt_sha256": hashlib.sha256(run_receipt_bytes).hexdigest(),
                    "status": "PASS",
                }
            )
        )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        for name in ("rq014_g2_declassification_export_receipt.json", "DONE.json"):
            (run_receipt_root / name).unlink(missing_ok=True)
        raise
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
