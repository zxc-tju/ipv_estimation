#!/usr/bin/env python3
"""Fail-closed validators for the rating-blind RQ014 G2 contract preflight."""
from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
PLACEHOLDER = "TO_FREEZE_AT_G2"
DENIED_PATH_PARTS = (
    "/rated479_segments",
    "/full479_targets",
    "/phase3_preference_test",
    "ratings_extracted.csv",
    "joined_candidates.csv",
    "/rq010b_wod_e2e/code",
    "/zxc/ipv_estimation",
    "/zxc/rq014_recovery",
)
ROUTE_INTENTS = {
    "1": "GO_STRAIGHT",
    "2": "GO_LEFT",
    "3": "GO_RIGHT",
}
STATE_OUTPUT_COLUMNS = (
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "vel_x_mps",
    "vel_y_mps",
    "accel_x_mps2",
    "accel_y_mps2",
)
SCIENTIFIC_TIME_SERIES_CONTRACT_SHA256 = (
    "382304b384f3e6b0642059ba768cc192174c4e92ae2bd16c1b8788274868a214"
)
BLIND_ANCHOR_RUNTIME_ROOT = Path("inputs/RQ014/blind_anchor/v1")
BLIND_ANCHOR_RUNTIME_PATH = (
    BLIND_ANCHOR_RUNTIME_ROOT / "RQ014_blind_anchor_receipt_v1p5.json"
)
BLIND_ANCHOR_SIZE_BYTES = 1752
BLIND_ANCHOR_SHA256 = (
    "80e393f73e353e19da4d280ca946a6e7dcee3197824f723155944829f295496a"
)


class ContractError(ValueError):
    """Raised when a preflight predicate fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"Non-finite JSON token: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Top-level JSON object required: {path}")
    return value


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


def _candidate_set_sha256(segment_id: str, tstar: int, hashes: list[str]) -> str:
    payload = {
        "candidate_geometry_sha256": hashes,
        "segment_id": segment_id,
        "tstar_context_step": tstar,
    }
    return hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()


def _trajectory_geometry_sha256_from_rows(rows: list[dict[str, str]]) -> str:
    source_to_output = {
        "pos_x": "pos_x_m",
        "pos_y": "pos_y_m",
        "pos_z": "pos_z_m",
        "vel_x": "vel_x_mps",
        "vel_y": "vel_y_mps",
        "accel_x": "accel_x_mps2",
        "accel_y": "accel_y_mps2",
    }
    digest = hashlib.sha256()
    for source_field, output_field in source_to_output.items():
        tokens = [row[output_field] for row in rows]
        if all(token == "NA" for token in tokens):
            values: list[str] = []
        elif any(token == "NA" for token in tokens):
            raise ContractError(f"Candidate {output_field} mixes NA and finite values")
        else:
            values = tokens
        digest.update(source_field.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(len(values)).encode("ascii"))
        digest.update(b":")
        for value in values:
            digest.update(format(float(value), ".17g").encode("ascii"))
            digest.update(b",")
        digest.update(b";")
    return digest.hexdigest()


def tstar_anchored_linear_resample(
    times: list[float],
    values: list[float],
    *,
    step_s: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Reference R10L operator: t*-anchored, linear, observed support only."""
    if len(times) != len(values) or len(times) < 2:
        raise ContractError("Linear resampling needs at least two paired samples")
    if not math.isfinite(step_s) or step_s <= 0:
        raise ContractError("Linear resampling step must be positive and finite")
    if any(not math.isfinite(item) for item in (*times, *values)):
        raise ContractError("Linear resampling inputs must be finite")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ContractError("Linear resampling times must be strictly increasing")
    tolerance = 1e-12
    first_k = math.ceil(times[0] / step_s - tolerance)
    last_k = math.floor(times[-1] / step_s + tolerance)
    grid = [round(index * step_s, 12) for index in range(first_k, last_k + 1)]
    output: list[float] = []
    left_index = 0
    for target in grid:
        while left_index + 1 < len(times) and times[left_index + 1] < target - tolerance:
            left_index += 1
        if math.isclose(target, times[left_index], rel_tol=0.0, abs_tol=tolerance):
            output.append(values[left_index])
            continue
        if left_index + 1 >= len(times):
            if math.isclose(target, times[-1], rel_tol=0.0, abs_tol=tolerance):
                output.append(values[-1])
                continue
            raise ContractError("R10L target exceeds observed support")
        left_time = times[left_index]
        right_time = times[left_index + 1]
        if target < left_time - tolerance or target > right_time + tolerance:
            raise ContractError("R10L target is not bracketed by observed support")
        weight = (target - left_time) / (right_time - left_time)
        output.append(values[left_index] + weight * (values[left_index + 1] - values[left_index]))
    return grid, output


def secant_kinematics(
    times: list[float], positions: list[float]
) -> tuple[list[float], list[float]]:
    """Frozen derivative operator with one-sided endpoints and centered interiors."""
    if len(times) != len(positions) or len(times) < 2:
        raise ContractError("Derivative operator needs at least two paired samples")
    if any(not math.isfinite(item) for item in (*times, *positions)):
        raise ContractError("Derivative inputs must be finite")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ContractError("Derivative times must be strictly increasing")

    def differentiate(values: list[float]) -> list[float]:
        result = [(values[1] - values[0]) / (times[1] - times[0])]
        result.extend(
            (values[index + 1] - values[index - 1])
            / (times[index + 1] - times[index - 1])
            for index in range(1, len(values) - 1)
        )
        result.append((values[-1] - values[-2]) / (times[-1] - times[-2]))
        return result

    velocity = differentiate(positions)
    return velocity, differentiate(velocity)


def require_exact_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_file_hash(path: Path, expected: str, label: str) -> None:
    require_sha256(expected, f"{label}.sha256")
    if not path.is_file():
        raise ContractError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(f"{label} SHA-256 mismatch: {path}")


def _normalized_path(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def reject_denied_path(path: Path) -> None:
    text = _normalized_path(path)
    for part in DENIED_PATH_PARTS:
        if part in text:
            raise ContractError(f"Denied RQ014 G2 path: {path}")


def require_contained_regular_file(path: Path, roots: Iterable[Path]) -> Path:
    if path.is_symlink():
        raise ContractError(f"Symlink input is forbidden: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = absolute.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"Regular file required: {resolved}")
    for root in roots:
        root_absolute = Path(os.path.abspath(root))
        if root_absolute.is_symlink():
            continue
        try:
            lexical_relative = absolute.relative_to(root_absolute)
        except ValueError:
            continue
        current = root_absolute
        symlink_found = False
        for part in lexical_relative.parts:
            current = current / part
            if current.is_symlink():
                symlink_found = True
                break
        if symlink_found:
            continue
        root_resolved = root_absolute.resolve(strict=True)
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        reject_denied_path(resolved)
        return resolved
    raise ContractError(f"Input escapes allowed roots or crosses a symlink: {path}")


def validate_m3_artifact_ref(
    ref: dict[str, Any],
    *,
    base: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify preflight-required, export-prohibited M3 bytes through one retained descriptor."""

    def mismatch(message: str, exc: OSError | None = None) -> None:
        error = ContractError(f"M3_ARTIFACT_MISMATCH: {message}")
        if exc is None:
            raise error
        raise error from exc

    delivery = contract.get("m3_artifact_delivery_contract")
    delivery_keys = {
        "spec_ref_field",
        "required_for_operation",
        "prohibited_for_operation",
        "path",
        "allowed_root",
        "size_bytes",
        "sha256",
        "open_policy",
        "verification_order",
        "deserialization_in_contract_preflight",
        "verification_only_for_operation",
        "deserialization_in_resource_pilot",
        "immutable_receipt_schema",
        "job_start_reverification",
    }
    if not isinstance(delivery, dict) or set(delivery) != delivery_keys:
        mismatch("delivery contract is missing or malformed")
    expected_constants = {
        "spec_ref_field": "m3_artifact",
        "required_for_operation": "rq014_g2_contract_preflight",
        "prohibited_for_operation": "rq014_g2_declassification_export",
        "open_policy": "SINGLE_RETAINED_FD_O_RDONLY_O_NOFOLLOW_O_CLOEXEC_O_NONBLOCK",
        "verification_order": (
            "BEFORE_INPUT_MANIFEST_MATERIALIZATION_LEDGER_CELL_RATING_AND_DESERIALIZATION"
        ),
        "deserialization_in_contract_preflight": "FORBIDDEN",
        "verification_only_for_operation": "rq014_g2_resource_pilot",
        "deserialization_in_resource_pilot": (
            "REQUIRED_AFTER_OPERATION_BOUND_V4_CLOSURE_GATE; "
            "FAILURE_IS_GLOBAL_ABORT_WITH_NO_DONE"
        ),
        "immutable_receipt_schema": "rq014-m3-artifact-input-receipt-v1",
        "job_start_reverification": True,
    }
    if any(delivery.get(key) != value for key, value in expected_constants.items()):
        mismatch("delivery contract must require preflight and prohibit export")
    if not isinstance(ref, dict) or set(ref) != {"path", "size_bytes", "sha256"}:
        mismatch("spec reference keys differ")
    if (
        not isinstance(delivery.get("path"), str)
        or not isinstance(delivery.get("allowed_root"), str)
        or not isinstance(delivery.get("size_bytes"), int)
        or isinstance(delivery.get("size_bytes"), bool)
        or delivery["size_bytes"] <= 0
        or not isinstance(delivery.get("sha256"), str)
        or HEX64.fullmatch(delivery["sha256"]) is None
    ):
        mismatch("delivery path, size, or digest is malformed")
    if ref != {
        "path": delivery["path"],
        "size_bytes": delivery["size_bytes"],
        "sha256": delivery["sha256"],
    }:
        mismatch("spec reference differs from the reviewed delivery contract")

    base_absolute = Path(os.path.abspath(base))
    expected_root = base_absolute / "checkpoints" / "rq009_m3"
    root = Path(os.path.abspath(Path(delivery["allowed_root"])))
    path = Path(os.path.abspath(Path(delivery["path"])))
    if root != expected_root or path != expected_root / "m3_scorer.joblib":
        mismatch("path or managed-root binding drifted")
    try:
        relative = path.relative_to(base_absolute)
    except ValueError:
        mismatch("path escapes the managed base")
    current = base_absolute
    try:
        if current.is_symlink() or not current.is_dir():
            mismatch("managed base is not a regular directory root")
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                mismatch(f"path crosses a symlink: {current}")
    except OSError as exc:
        mismatch("path-component inspection failed", exc)

    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    descriptor: int | None = None
    directory_descriptors: list[int] = []
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        directory_descriptors.append(os.open(base_absolute, directory_flags))
        for component in ("checkpoints", "rq009_m3"):
            directory_descriptors.append(
                os.open(component, directory_flags, dir_fd=directory_descriptors[-1])
            )
        descriptor = os.open(
            "m3_scorer.joblib",
            flags,
            dir_fd=directory_descriptors[-1],
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            mismatch("artifact is not a regular file")
        if before.st_size != delivery["size_bytes"]:
            mismatch("artifact size differs from the reviewed binding")
        digest = hashlib.sha256()
        remaining = delivery["size_bytes"]
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                mismatch("artifact ended before the reviewed size")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            mismatch("artifact exceeds the reviewed size")
        after = os.fstat(descriptor)
    except OSError as exc:
        mismatch("retained no-follow open/read failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
    continuity_fields = (
        "st_mode",
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in continuity_fields):
        mismatch("descriptor identity changed during verification")
    if digest.hexdigest() != delivery["sha256"]:
        mismatch("artifact SHA-256 differs from the reviewed binding")
    return {
        "schema_version": "rq014-m3-artifact-input-receipt-v1",
        "role": "frozen_m3_scorer",
        "path": delivery["path"],
        "size_bytes": delivery["size_bytes"],
        "sha256": delivery["sha256"],
        "verification_mode": "SINGLE_RETAINED_FD_O_NOFOLLOW_PRE_DESERIALIZATION",
        "deserialized": False,
    }


def _normalize_semantic(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def _forbidden_semantics(schema: dict[str, Any]) -> set[str]:
    values = schema.get("forbidden_semantic_fields_normalized")
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ContractError("Malformed score-stripped semantic denylist")
    return {_normalize_semantic(item) for item in values}


def _finite(value: str, label: str, *, allow_na: bool = False) -> float | None:
    if allow_na and value == "NA":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ContractError(f"{label} is non-finite")
    return parsed


def _read_csv(
    path: Path,
    *,
    expected_columns: list[str],
    forbidden: set[str],
) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError(f"CSV must use UTF-8 LF with a trailing newline: {path.name}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"CSV is not UTF-8: {path.name}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != expected_columns:
        raise ContractError(
            f"CSV schema mismatch for {path.name}: {reader.fieldnames} != {expected_columns}"
        )
    normalized = {_normalize_semantic(name) for name in reader.fieldnames}
    overlap = normalized & forbidden
    if overlap:
        raise ContractError(f"Forbidden columns in {path.name}: {sorted(overlap)}")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=2):
        if None in row:
            raise ContractError(f"Extra CSV fields at {path.name}:{index}")
        if any(value is None for value in row.values()):
            raise ContractError(f"Missing CSV field at {path.name}:{index}")
        normalized_row = {key: str(value) for key, value in row.items()}
        for column, value in normalized_row.items():
            semantic = _normalize_semantic(value)
            if (
                any(token and token in semantic for token in forbidden)
                or "human_rating" in semantic
                or "preference_score" in semantic
                or semantic == "rating"
                or semantic.startswith("rating_")
                or semantic.endswith("_rating")
            ):
                raise ContractError(
                    f"Forbidden rating semantic in {path.name}:{index}:{column}"
                )
        rows.append(normalized_row)
    canonical = io.StringIO(newline="")
    writer = csv.DictWriter(
        canonical,
        fieldnames=expected_columns,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    if canonical.getvalue().encode("utf-8") != raw:
        raise ContractError(f"CSV is not in canonical RFC4180/LF form: {path.name}")
    return rows


def _assert_unique(rows: list[dict[str, str]], keys: list[str], label: str) -> None:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in seen:
            raise ContractError(f"Duplicate {label} primary key: {key}")
        seen.add(key)


def _assert_contiguous(rows: list[dict[str, str]], group: list[str], index: str, label: str) -> None:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row in rows:
        try:
            grouped[tuple(row[key] for key in group)].append(int(row[index]))
        except ValueError as exc:
            raise ContractError(f"Non-integer {label} sample index") from exc
    for key, values in grouped.items():
        ordered = sorted(values)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            raise ContractError(f"Non-contiguous {label} indices for {key}")


def validate_score_stripped_bundle(
    *,
    bundle_root: Path,
    schema_path: Path,
    file_manifest_path: Path,
    receipt_path: Path,
    full_hash: bool = True,
    expected_exporter_git_commit: str | None = None,
    expected_exporter_environment_sha256: str | None = None,
    expected_exporter_code_sha256: str | None = None,
    expected_source_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    if bundle_root.is_symlink():
        raise ContractError("Score-stripped bundle root may not be a symlink")
    bundle_root = bundle_root.resolve(strict=True)
    if not bundle_root.is_dir():
        raise ContractError("Score-stripped bundle root must be a directory")
    reject_denied_path(bundle_root)

    schema = load_json(schema_path)
    if schema.get("schema_version") != "rq014-score-stripped-schema-v1":
        raise ContractError("Wrong score-stripped schema")
    time_series_contract = schema.get("scientific_time_series_contract")
    if not isinstance(time_series_contract, dict) or hashlib.sha256(
        canonical_json_bytes(time_series_contract)
    ).hexdigest() != SCIENTIFIC_TIME_SERIES_CONTRACT_SHA256:
        raise ContractError("Scientific time-series contract drifted")
    expected_csv_files = {
        "blind_scene_manifest.csv",
        "candidate_states.csv",
        "ego_history_states.csv",
        "ego_future_states.csv",
        "tstar_ego_pose.csv",
        "counterpart_tracks.csv",
        "structural_attrition.csv",
    }
    if set(schema.get("files", {})) != expected_csv_files:
        raise ContractError("Score-stripped CSV allowlist drifted")
    forbidden = _forbidden_semantics(schema)
    required_files = set(schema["files"]) | set(schema["required_receipts"])
    actual_files = {path.name for path in bundle_root.iterdir() if path.is_file()}
    non_files = [path for path in bundle_root.iterdir() if not path.is_file()]
    if non_files:
        raise ContractError(f"Unexpected directory/symlink in score-stripped bundle: {non_files[0]}")
    if actual_files != required_files:
        raise ContractError(
            f"Score-stripped bundle files differ; missing={sorted(required_files - actual_files)}, "
            f"unexpected={sorted(actual_files - required_files)}"
        )

    loaded_rows: dict[str, list[dict[str, str]]] = {}
    for name, file_schema in schema["files"].items():
        path = bundle_root / name
        if path.suffix != ".csv" or path.is_symlink():
            raise ContractError(f"Canonical CSV regular file required: {path}")
        rows = _read_csv(
            path,
            expected_columns=file_schema["columns"],
            forbidden=forbidden,
        )
        _assert_unique(rows, file_schema["primary_key"], name)
        loaded_rows[name] = rows

    scenes = loaded_rows["blind_scene_manifest.csv"]
    if len(scenes) != int(schema["frozen_universe_segment_count"]):
        raise ContractError("Blind scene universe is not exactly 479 rows")
    scene_ids = {row["segment_id"] for row in scenes}
    if len(scene_ids) != len(scenes) or not all(SAFE_ID.fullmatch(item) for item in scene_ids):
        raise ContractError("Blind scene IDs are duplicate or unsafe")
    scene_by_id = {row["segment_id"]: row for row in scenes}
    geometry_scenes: set[str] = set()
    empty_candidate_set_sha256 = hashlib.sha256(b"").hexdigest()
    for row in scenes:
        if row["candidate_geometry_available"] not in {"true", "false"}:
            raise ContractError("candidate_geometry_available must be true/false")
        if row["candidate_geometry_available"] == "true":
            geometry_scenes.add(row["segment_id"])
            if row["candidate_count"] != "3":
                raise ContractError("Geometry-available scene must contain three candidates")
            try:
                int(row["tstar_context_step"])
            except ValueError as exc:
                raise ContractError("Geometry-available scene needs an integer tstar") from exc
            if row["coordinate_frame"] != "ego_at_tstar":
                raise ContractError("Geometry-available scene has the wrong coordinate frame")
            if row["source_shard_id"] == "NA" or not SAFE_ID.fullmatch(row["source_shard_id"]):
                raise ContractError("Geometry-available scene needs a nonempty safe source shard ID")
            if row["path_type"] != "UNMAPPED":
                raise ContractError("Path type must remain UNMAPPED before the frozen route mapping")
            if ROUTE_INTENTS.get(row["route_intent_code"]) != row["route_intent_name"]:
                raise ContractError("Route intent code/name differs from the audited source mapping")
            if _finite(row["native_frame_rate_hz"], "native frame rate") != 10.0:
                raise ContractError("Geometry-available scene must retain the 10 Hz native rate")
            if _finite(row["state_rate_hz"], "state rate") != 4.0:
                raise ContractError("Geometry-available scene must retain the 4 Hz state rate")
            if _finite(row["candidate_rate_hz"], "candidate rate") != 4.0:
                raise ContractError("Geometry-available scene must retain the 4 Hz candidate rate")
            if row["ego_future_state_count"] != "20":
                raise ContractError("Geometry-available scene must preserve 20 ego-future states")
            if row["tstar_ego_pose_element_count"] != "16":
                raise ContractError("Geometry-available scene must preserve a 4x4 tstar pose")
            if row["structural_status"] != "GEOMETRY_AVAILABLE":
                raise ContractError("Geometry-available scene has the wrong structural status")
        else:
            if (
                row["tstar_context_step"] != "NA"
                or row["source_shard_id"] != "NA"
                or row["path_type"] != "NA"
                or row["route_intent_code"] != "NA"
                or row["route_intent_name"] != "NA"
                or row["state_rate_hz"] != "NA"
                or row["candidate_rate_hz"] != "NA"
                or row["candidate_count"] != "0"
                or row["ego_future_state_count"] != "0"
                or row["tstar_ego_pose_element_count"] != "0"
                or row["structural_status"] != "MISSING_DECLASSIFIED_PHASE1_SCENE"
                or row["candidate_set_sha256"] != empty_candidate_set_sha256
            ):
                raise ContractError("Geometry-unavailable scene does not use the frozen empty-scene encoding")
        require_sha256(row["candidate_set_sha256"], "candidate_set_sha256")

    candidates = loaded_rows["candidate_states.csv"]
    _assert_contiguous(
        candidates,
        ["segment_id", "candidate_ordinal"],
        "raw_sample_index",
        "candidate",
    )
    candidate_ids: dict[str, set[tuple[str, str]]] = defaultdict(set)
    candidate_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        if row["segment_id"] not in scene_ids:
            raise ContractError("Candidate row references unknown segment")
        scene = scene_by_id[row["segment_id"]]
        if row["segment_id"] not in geometry_scenes:
            raise ContractError("Candidate row references a geometry-unavailable segment")
        if row["tstar_context_step"] != scene["tstar_context_step"]:
            raise ContractError("Candidate row tstar differs from the blind scene manifest")
        if row["candidate_id"] not in {"C1", "C2", "C3"}:
            raise ContractError("candidate_id must be C1/C2/C3")
        if row["candidate_ordinal"] not in {"1", "2", "3"}:
            raise ContractError("candidate_ordinal must be 1/2/3")
        if row["candidate_id"] != f"C{row['candidate_ordinal']}":
            raise ContractError("candidate_id and ordinal disagree")
        require_sha256(row["geometry_sha256"], "geometry_sha256")
        _finite(row["raw_time_s"], "candidate raw time")
        for flag in ("dropped_as_tstar_duplicate", "included_in_effective_future"):
            if row[flag] not in {"true", "false"}:
                raise ContractError(f"{flag} must be true/false")
        if row["included_in_effective_future"] == "true":
            try:
                effective_index = int(row["effective_sample_index"])
            except ValueError as exc:
                raise ContractError("Included candidate row needs an integer effective index") from exc
            effective_time = _finite(row["effective_time_s"], "candidate effective time")
            if effective_index < 1 or effective_index > 20 or effective_time != effective_index * 0.25:
                raise ContractError("Candidate effective time/index contract failed")
            if effective_time <= 0 or effective_time != float(row["raw_time_s"]):
                raise ContractError("Effective candidate must retain its positive raw time")
        elif row["effective_sample_index"] != "NA" or row["effective_time_s"] != "NA":
            raise ContractError("Excluded candidate row must use NA effective index/time")
        for state_field in STATE_OUTPUT_COLUMNS:
            _finite(
                row[state_field],
                f"candidate {state_field}",
                allow_na=state_field not in {"pos_x_m", "pos_y_m"},
            )
        candidate_ids[row["segment_id"]].add((row["candidate_id"], row["geometry_sha256"]))
        candidate_groups[(row["segment_id"], row["candidate_ordinal"])].append(row)
    for key, group_rows in candidate_groups.items():
        group_rows.sort(key=lambda row: int(row["raw_sample_index"]))
        if int(group_rows[0]["raw_sample_index"]) != 0:
            raise ContractError(f"Candidate raw indices must start at zero for {key}")
        if len(group_rows) < 2 or len(group_rows) > 21:
            raise ContractError(f"Candidate source length must be within 2..21 for {key}")
        if len({row["geometry_sha256"] for row in group_rows}) != 1:
            raise ContractError(f"Candidate geometry hash drifts within {key}")
        for state_field in STATE_OUTPUT_COLUMNS[2:]:
            values = [row[state_field] for row in group_rows]
            if any(value == "NA" for value in values) and not all(
                value == "NA" for value in values
            ):
                raise ContractError(f"Candidate {state_field} mixes empty and populated encoding for {key}")
        if group_rows[0]["geometry_sha256"] != _trajectory_geometry_sha256_from_rows(group_rows):
            raise ContractError(f"Candidate legacy seven-field geometry SHA-256 mismatch for {key}")
        included = [row for row in group_rows if row["included_in_effective_future"] == "true"]
        if [int(row["effective_sample_index"]) for row in included] != list(range(1, len(included) + 1)):
            raise ContractError(f"Candidate effective indices are not contiguous for {key}")
        dropped = [row for row in group_rows if row["dropped_as_tstar_duplicate"] == "true"]
        if len(dropped) > 1 or (dropped and dropped[0]["raw_sample_index"] != "0"):
            raise ContractError(f"Candidate duplicate-drop flag is invalid for {key}")
    for segment_id in geometry_scenes:
        id_hash_pairs = candidate_ids.get(segment_id, set())
        ids = {item[0] for item in id_hash_pairs}
        if ids != {"C1", "C2", "C3"}:
            raise ContractError(f"Geometry scene lacks exact C1/C2/C3: {segment_id}")
        hashes_by_id = {candidate_id: digest for candidate_id, digest in id_hash_pairs}
        expected_set_hash = _candidate_set_sha256(
            segment_id,
            int(scene_by_id[segment_id]["tstar_context_step"]),
            [hashes_by_id[f"C{ordinal}"] for ordinal in range(1, 4)],
        )
        if scene_by_id[segment_id]["candidate_set_sha256"] != expected_set_hash:
            raise ContractError(f"Candidate-set SHA-256 mismatch: {segment_id}")

    histories = loaded_rows["ego_history_states.csv"]
    _assert_contiguous(histories, ["segment_id"], "sample_index", "ego history")
    for row in histories:
        if row["segment_id"] not in scene_ids:
            raise ContractError("History row references unknown segment")
        scene = scene_by_id[row["segment_id"]]
        if row["segment_id"] not in geometry_scenes:
            raise ContractError("History row references a geometry-unavailable segment")
        if row["tstar_context_step"] != scene["tstar_context_step"]:
            raise ContractError("History row tstar differs from the blind scene manifest")
        _finite(row["time_s"], "history time")
        for state_field in STATE_OUTPUT_COLUMNS:
            _finite(
                row[state_field],
                f"history {state_field}",
                allow_na=state_field not in {"pos_x_m", "pos_y_m"},
            )
    history_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in histories:
        history_groups[row["segment_id"]].append(row)
    for segment_id, group_rows in history_groups.items():
        ordered = sorted(group_rows, key=lambda row: int(row["sample_index"]))
        if int(ordered[0]["sample_index"]) != 0:
            raise ContractError(f"Ego-history indices must start at zero: {segment_id}")
        if len(ordered) != 16:
            raise ContractError(f"Ego history must preserve exactly 16 source states: {segment_id}")
        expected_times = [(index - (len(ordered) - 1)) * 0.25 for index in range(len(ordered))]
        actual_times = [float(row["time_s"]) for row in ordered]
        if actual_times != expected_times:
            raise ContractError(f"Ego-history 0.25 s time axis mismatch: {segment_id}")
        for state_field in STATE_OUTPUT_COLUMNS[2:]:
            values = [row[state_field] for row in ordered]
            if any(value == "NA" for value in values) and not all(
                value == "NA" for value in values
            ):
                raise ContractError(f"History {state_field} mixes empty and populated encoding: {segment_id}")
        r10_times, _ = tstar_anchored_linear_resample(
            actual_times,
            [float(row["pos_x_m"]) for row in ordered],
        )
        if not r10_times or r10_times[-1] != 0.0 or any(time > 0 for time in r10_times):
            raise ContractError(f"History R10L grid is not tstar-anchored at zero: {segment_id}")
    if set(history_groups) != geometry_scenes:
        raise ContractError("Ego-history rows do not exactly cover geometry-available scenes")

    futures = loaded_rows["ego_future_states.csv"]
    _assert_contiguous(futures, ["segment_id"], "sample_index", "ego future")
    future_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in futures:
        if row["segment_id"] not in geometry_scenes:
            raise ContractError("Ego-future row references a geometry-unavailable segment")
        scene = scene_by_id[row["segment_id"]]
        if row["tstar_context_step"] != scene["tstar_context_step"]:
            raise ContractError("Ego-future row tstar differs from the blind scene manifest")
        _finite(row["time_s"], "ego-future time")
        for state_field in STATE_OUTPUT_COLUMNS:
            _finite(
                row[state_field],
                f"ego-future {state_field}",
                allow_na=state_field not in {"pos_x_m", "pos_y_m"},
            )
        future_groups[row["segment_id"]].append(row)
    for segment_id, group_rows in future_groups.items():
        ordered = sorted(group_rows, key=lambda row: int(row["sample_index"]))
        if len(ordered) != 20 or ordered[0]["sample_index"] != "0":
            raise ContractError(f"Ego future must preserve exactly 20 source states: {segment_id}")
        expected_times = [(index + 1) * 0.25 for index in range(20)]
        if [float(row["time_s"]) for row in ordered] != expected_times:
            raise ContractError(f"Ego-future positive 0.25 s time axis mismatch: {segment_id}")
        for state_field in STATE_OUTPUT_COLUMNS[2:]:
            values = [row[state_field] for row in ordered]
            if any(value == "NA" for value in values) and not all(
                value == "NA" for value in values
            ):
                raise ContractError(f"Ego-future {state_field} mixes empty and populated encoding: {segment_id}")
    if set(future_groups) != geometry_scenes:
        raise ContractError("Ego-future rows do not exactly cover geometry-available scenes")

    poses = loaded_rows["tstar_ego_pose.csv"]
    pose_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in poses:
        if row["segment_id"] not in geometry_scenes:
            raise ContractError("Pose row references a geometry-unavailable segment")
        if row["tstar_context_step"] != scene_by_id[row["segment_id"]]["tstar_context_step"]:
            raise ContractError("Pose row tstar differs from the blind scene manifest")
        if row["matrix_row"] not in {"0", "1", "2", "3"} or row["matrix_column"] not in {
            "0",
            "1",
            "2",
            "3",
        }:
            raise ContractError("Pose indices must describe an exact 4x4 matrix")
        _finite(row["value"], "tstar ego pose value")
        pose_groups[row["segment_id"]].append(row)
    expected_pose_indices = {(str(row), str(column)) for row in range(4) for column in range(4)}
    for segment_id, group_rows in pose_groups.items():
        indices = {(row["matrix_row"], row["matrix_column"]) for row in group_rows}
        if len(group_rows) != 16 or indices != expected_pose_indices:
            raise ContractError(f"Tstar ego pose is not an exact 4x4 matrix: {segment_id}")
    if set(pose_groups) != geometry_scenes:
        raise ContractError("Tstar ego-pose rows do not exactly cover geometry-available scenes")
    for key, group_rows in candidate_groups.items():
        segment_id = key[0]
        history = sorted(history_groups.get(segment_id, []), key=lambda row: int(row["sample_index"]))
        if not history:
            raise ContractError(f"Candidate scene lacks ego history: {segment_id}")
        first = min(group_rows, key=lambda row: int(row["raw_sample_index"]))
        last_past = history[-1]
        distance = math.hypot(
            float(last_past["pos_x_m"]) - float(first["pos_x_m"]),
            float(last_past["pos_y_m"]) - float(first["pos_y_m"]),
        )
        expected_drop = distance < 0.75
        actual_drop = first["dropped_as_tstar_duplicate"] == "true"
        if actual_drop != expected_drop:
            raise ContractError(f"Candidate duplicate-drop decision mismatch for {key}")
        ordered_raw = sorted(group_rows, key=lambda row: int(row["raw_sample_index"]))
        expected_raw_times = [
            (int(row["raw_sample_index"]) if expected_drop else int(row["raw_sample_index"]) + 1)
            * 0.25
            for row in ordered_raw
        ]
        if [float(row["raw_time_s"]) for row in ordered_raw] != expected_raw_times:
            raise ContractError(f"Candidate raw-time interpretation mismatch for {key}")
        if expected_drop and float(first["raw_time_s"]) != 0.0:
            raise ContractError(f"Candidate duplicate raw0 must be the audit-only tstar sample for {key}")
        if not expected_drop and float(first["raw_time_s"]) != 0.25:
            raise ContractError(f"Candidate nonduplicate raw0 must be the first positive sample for {key}")
        expected_raw = [
            int(row["raw_sample_index"])
            for row in ordered_raw
            if int(row["raw_sample_index"]) >= (1 if expected_drop else 0)
        ][:20]
        actual_raw = [
            int(row["raw_sample_index"])
            for row in sorted(
                (item for item in group_rows if item["included_in_effective_future"] == "true"),
                key=lambda item: int(item["effective_sample_index"]),
            )
        ]
        if actual_raw != expected_raw:
            raise ContractError(f"Candidate effective-future membership mismatch for {key}")
        effective = sorted(
            (row for row in group_rows if row["included_in_effective_future"] == "true"),
            key=lambda row: int(row["effective_sample_index"]),
        )
        if not effective or any(float(row["effective_time_s"]) <= 0 for row in effective):
            raise ContractError(f"Candidate effective series must contain positive-time support for {key}")
        history_times = [float(row["time_s"]) for row in history]
        effective_times = [float(row["effective_time_s"]) for row in effective]
        r04_times = history_times + effective_times
        for state_field in ("pos_x_m", "pos_y_m"):
            r04_values = [float(row[state_field]) for row in history] + [
                float(row[state_field]) for row in effective
            ]
            secant_kinematics(r04_times, r04_values)
            r10_times, r10_values = tstar_anchored_linear_resample(
                r04_times,
                r04_values,
            )
            future_r10_times = [time for time in r10_times if time > 0]
            if not future_r10_times or future_r10_times[0] != 0.1:
                raise ContractError(f"Candidate R10L future grid is not tstar-anchored for {key}")
            secant_kinematics(r10_times, r10_values)

    tracks = loaded_rows["counterpart_tracks.csv"]
    for row in tracks:
        if row["segment_id"] not in scene_ids:
            raise ContractError("Counterpart row references unknown segment")
        try:
            row_tstar = int(row["tstar_context_step"])
            context_step = int(row["context_step"])
        except ValueError as exc:
            raise ContractError("Counterpart tstar/context step must be integers") from exc
        scene_tstar = scene_by_id[row["segment_id"]]["tstar_context_step"]
        if scene_tstar != "NA" and row["tstar_context_step"] != scene_tstar:
            raise ContractError("Counterpart row tstar differs from the blind scene manifest")
        time_s = _finite(row["time_s"], "counterpart time_s")
        if not math.isclose(
            time_s,
            (context_step - row_tstar) * 0.1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContractError("Counterpart time axis differs from the 10 Hz context-step contract")
        for field in ("time_s", "x_m", "y_m", "vx_mps", "vy_mps", "detector_confidence"):
            _finite(row[field], f"counterpart {field}", allow_na=field in {"vx_mps", "vy_mps"})

    attrition = loaded_rows["structural_attrition.csv"]
    attrition_segments: set[str] = set()
    for row in attrition:
        if row["segment_id"] not in scene_ids:
            raise ContractError("Attrition row references unknown segment")
        if (
            row["stage"] != "DECLASSIFICATION"
            or row["reason_code"] != "MISSING_DECLASSIFIED_PHASE1_SCENE"
            or row["source_receipt_id"] != "phase1_bundle_declassification_v1"
        ):
            raise ContractError("Structural attrition row differs from the frozen declassification encoding")
        attrition_segments.add(row["segment_id"])
        normalized_reason = _normalize_semantic(row["reason_code"])
        if normalized_reason in forbidden or any(token in normalized_reason for token in ("rating", "preference")):
            raise ContractError("Rating-derived attrition reason is forbidden in G2")
    if attrition_segments != scene_ids - geometry_scenes:
        raise ContractError("Structural attrition does not exactly cover geometry-unavailable scenes")

    file_manifest = load_json(file_manifest_path)
    if file_manifest_path.read_bytes() != canonical_json_bytes(file_manifest):
        raise ContractError("File manifest is not canonical JSON")
    require_exact_keys(file_manifest, {"schema_version", "files"}, "file_manifest")
    if file_manifest["schema_version"] != "rq014-score-stripped-file-manifest-v1":
        raise ContractError("Wrong score-stripped file manifest schema")
    rows = file_manifest["files"]
    if not isinstance(rows, list):
        raise ContractError("file_manifest.files must be a list")
    manifest_by_name: dict[str, dict[str, Any]] = {}
    required_manifest_fields = {
        "relative_path",
        "size_bytes",
        "sha256",
        "schema_id",
        "row_count",
        "primary_key",
        "contains_rating",
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("file manifest rows must be objects")
        require_exact_keys(row, required_manifest_fields, "file manifest row")
        relative = row["relative_path"]
        if relative not in schema["files"] or relative in manifest_by_name:
            raise ContractError(f"Unexpected or duplicate file-manifest row: {relative}")
        if row["contains_rating"] is not False:
            raise ContractError(f"Rating-bearing file registered in G2: {relative}")
        if row["schema_id"] != f"{schema['schema_version']}#{relative}":
            raise ContractError(f"Schema ID drift: {relative}")
        require_sha256(row["sha256"], f"{relative}.sha256")
        path = bundle_root / relative
        if path.stat().st_size != row["size_bytes"]:
            raise ContractError(f"Size mismatch: {relative}")
        if full_hash and sha256_file(path) != row["sha256"]:
            raise ContractError(f"Hash mismatch: {relative}")
        if row["row_count"] != len(loaded_rows[relative]):
            raise ContractError(f"Row-count mismatch: {relative}")
        if row["primary_key"] != schema["files"][relative]["primary_key"]:
            raise ContractError(f"Primary-key contract mismatch: {relative}")
        manifest_by_name[relative] = row
    if set(manifest_by_name) != set(schema["files"]):
        raise ContractError("File manifest does not cover every canonical CSV")

    receipt = load_json(receipt_path)
    if receipt_path.read_bytes() != canonical_json_bytes(receipt):
        raise ContractError("Sanitization receipt is not canonical JSON")
    expected_receipt_keys = {
        "schema_version",
        "exporter_code_sha256",
        "exporter_git_commit",
        "exporter_environment_sha256",
        "source_artifact_ids_and_sha256",
        "source_rating_bearing_classification",
        "source_mounts_not_exposed_to_g2",
        "output_file_hashes",
        "output_schema_hashes",
        "universe_segment_count",
        "geometry_available_scene_count",
        "candidate_count_distribution",
        "forbidden_field_scan",
        "unexpected_field_scan",
        "duplicate_key_scan",
        "nonfinite_value_scan",
        "created_at_utc",
    }
    require_exact_keys(receipt, expected_receipt_keys, "sanitization_receipt")
    if receipt["schema_version"] != "rq014-score-stripped-sanitization-v1":
        raise ContractError("Wrong sanitization receipt schema")
    for key in ("exporter_code_sha256", "exporter_environment_sha256"):
        require_sha256(receipt[key], key)
    if not isinstance(receipt["exporter_git_commit"], str) or not HEX40.fullmatch(
        receipt["exporter_git_commit"]
    ):
        raise ContractError("Exporter git commit must be lowercase 40-hex")
    if not isinstance(receipt["created_at_utc"], str) or not UTC_SECONDS.fullmatch(
        receipt["created_at_utc"]
    ):
        raise ContractError("Sanitization creation time must be exact UTC seconds")
    if not isinstance(receipt["source_artifact_ids_and_sha256"], dict):
        raise ContractError("source artifact receipt must be an ID-to-SHA object")
    expected_source_ids = {
        *(f"phase1_scene_bundle_{index:02d}" for index in range(8)),
        "rated479_structural_readiness",
        "selected_counterpart_tracks",
    }
    if set(receipt["source_artifact_ids_and_sha256"]) != expected_source_ids:
        raise ContractError("Source artifact receipt does not contain the exact declassification inputs")
    for artifact_id, digest in receipt["source_artifact_ids_and_sha256"].items():
        if not SAFE_ID.fullmatch(artifact_id):
            raise ContractError("Unsafe source artifact ID")
        require_sha256(digest, f"source artifact {artifact_id}")
    if receipt["source_mounts_not_exposed_to_g2"] is not True:
        raise ContractError("Source mounts must be absent from G2")
    expected_classification = (
        "parents originate from a rating-bearing pipeline; only audited score-omitting "
        "Phase-1 objects and structural/track tables were read"
    )
    if receipt["source_rating_bearing_classification"] != expected_classification:
        raise ContractError("Source rating-bearing classification drifted")
    if receipt["universe_segment_count"] != len(scenes):
        raise ContractError("Receipt universe count mismatch")
    if receipt["geometry_available_scene_count"] != len(geometry_scenes):
        raise ContractError("Receipt geometry count mismatch")
    expected_output_hashes = {name: manifest_by_name[name]["sha256"] for name in sorted(manifest_by_name)}
    if receipt["output_file_hashes"] != expected_output_hashes:
        raise ContractError("Receipt output hashes differ from file manifest")
    if receipt["output_schema_hashes"] != {schema_path.name: sha256_file(schema_path)}:
        raise ContractError("Receipt schema hash mismatch")
    expected_distribution = dict(sorted(Counter(row["candidate_count"] for row in scenes).items()))
    if receipt["candidate_count_distribution"] != expected_distribution:
        raise ContractError("Candidate-count distribution mismatch")
    for scan in ("forbidden_field_scan", "unexpected_field_scan", "duplicate_key_scan", "nonfinite_value_scan"):
        if receipt[scan] != 0:
            raise ContractError(f"Sanitization scan did not pass: {scan}")
    expected_values = {
        "exporter_git_commit": expected_exporter_git_commit,
        "exporter_environment_sha256": expected_exporter_environment_sha256,
        "exporter_code_sha256": expected_exporter_code_sha256,
        "source_artifact_ids_and_sha256": expected_source_artifacts,
    }
    for key, expected_value in expected_values.items():
        if expected_value is not None and receipt[key] != expected_value:
            raise ContractError(f"Sanitization provenance mismatch: {key}")

    return {
        "bundle_root": str(bundle_root),
        "schema_sha256": sha256_file(schema_path),
        "file_manifest_sha256": sha256_file(file_manifest_path),
        "sanitization_receipt_sha256": sha256_file(receipt_path),
        "scene_count": len(scenes),
        "geometry_available_scene_count": len(geometry_scenes),
        "canonical_csv_count": len(loaded_rows),
    }


def g2_input_allowed_roots(base: Path) -> list[Path]:
    """Return the identical managed roots used before submit and inside the job."""

    return [base / "inputs" / "RQ014", base / "manifests" / "RQ014"]


def validate_input_manifest_g2(
    *,
    manifest_path: Path,
    contract: dict[str, Any],
    allowed_roots: list[Path],
) -> dict[str, Path]:
    manifest = load_json(manifest_path)
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise ContractError("G2 input manifest is not canonical JSON")
    require_exact_keys(
        manifest,
        {"schema_version", "stage", "parent_manifest_sha256", "entries"},
        "input_manifest.g2",
    )
    g2_contract = contract["staged_input_manifests"]["G2"]
    if manifest["schema_version"] != g2_contract["schema_version"] or manifest["stage"] != "G2":
        raise ContractError("Wrong G2 input-manifest schema or stage")
    if manifest["parent_manifest_sha256"] is not None:
        raise ContractError("G2 is the root input manifest and has no parent")
    if not isinstance(manifest["entries"], list):
        raise ContractError("input_manifest.g2 entries must be a list")
    roles: dict[str, Path] = {}
    input_ids: set[str] = set()
    input_paths: set[Path] = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise ContractError("G2 input entry must be an object")
        require_exact_keys(
            entry,
            {"input_id", "role", "absolute_path", "sha256", "contains_rating"},
            "G2 input entry",
        )
        if not SAFE_ID.fullmatch(entry["input_id"]):
            raise ContractError("Unsafe G2 input ID")
        if entry["input_id"] in input_ids:
            raise ContractError(f"Duplicate G2 input ID: {entry['input_id']}")
        input_ids.add(entry["input_id"])
        role = entry["role"]
        if role in roles:
            raise ContractError(f"Duplicate G2 input role: {role}")
        if entry["contains_rating"] is not False:
            raise ContractError(f"Rating-bearing G2 input: {role}")
        path = require_contained_regular_file(Path(entry["absolute_path"]), allowed_roots)
        if path in input_paths:
            raise ContractError(f"Aliased G2 input path: {path}")
        input_paths.add(path)
        if path.suffix.lower() != ".json":
            raise ContractError(f"G2 input role must resolve to a JSON manifest/receipt: {role}")
        require_file_hash(path, entry["sha256"], role)
        roles[role] = path
    required = set(g2_contract["required_roles"])
    if set(roles) != required:
        raise ContractError(
            f"G2 roles differ; missing={sorted(required - set(roles))}, "
            f"unexpected={sorted(set(roles) - required)}"
        )
    if set(roles) & set(g2_contract["forbidden_roles"]):
        raise ContractError("Forbidden role appears in G2")
    return roles


def validate_anchor_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_date",
            "source_contract",
            "source_kind",
            "contains_row_level_ratings",
            "permits_new_rating_derived_statistics",
            "g2_verification_mode",
            "g2_rho_recomputation",
            "required_g2_outcome_labels",
            "run_scoped_predictor_and_key_hashes",
            "rating_artifact_hashes",
            "anchors",
            "deferred_recomputation_rule",
        },
        "blind-anchor receipt",
    )
    if receipt.get("schema_version") != "rq014-blind-anchor-receipt-v1p5":
        raise ContractError("Wrong blind-anchor receipt schema")
    expected_constants = {
        "receipt_date": "2026-07-12",
        "source_contract": (
            "reports/plans/RQ014_plan_v1_wod_e2e_rating_ipv_deviation_recovery_20260710.md#13.3"
        ),
        "source_kind": "previously_published_aggregate_constants_only",
        "permits_new_rating_derived_statistics": False,
        "g2_verification_mode": "receipt_integrity_and_rating_free_implementation_fixtures_only",
        "required_g2_outcome_labels": [
            "ATTESTED_RECEIPT_MATCH",
            "RATING_FREE_PREDICTOR_PARITY_PASS",
            "RHO_NOT_RECOMPUTED_BY_DESIGN",
        ],
        "run_scoped_predictor_and_key_hashes": "TO_BIND_IN_INPUT_MANIFEST_G2",
        "rating_artifact_hashes": "DEFERRED_TO_AUTHORIZED_INPUT_MANIFEST_G3",
        "deferred_recomputation_rule": (
            "Exact N/key/rho recomputation is deferred to a future checksum-bound rating-join "
            "operation; it must occur before new discovery statistics and requires separate PI "
            "and central authorization."
        ),
    }
    for key, expected_value in expected_constants.items():
        if receipt[key] != expected_value:
            raise ContractError(f"Blind-anchor constant drift: {key}")
    if receipt.get("contains_row_level_ratings") is not False:
        raise ContractError("Blind-anchor receipt contains row-level ratings")
    if receipt.get("g2_rho_recomputation") != "FORBIDDEN":
        raise ContractError("G2 rho recomputation must remain forbidden")
    anchors = receipt.get("anchors")
    if not isinstance(anchors, list) or [row.get("anchor_id") for row in anchors] != ["A1", "A2", "A3", "A4"]:
        raise ContractError("Blind-anchor receipt must contain A1-A4 in order")
    expected = {
        "A1": (4, 1, 75, 0.14767623020869206),
        "A2": (4, 2, 98, 0.03129843185743807),
        "A3": (10, 1, 75, 0.16481712868606582),
        "A4": (10, 2, 47, 0.12753396435827072),
    }
    for row in anchors:
        if not isinstance(row, dict):
            raise ContractError("Blind-anchor row must be an object")
        require_exact_keys(
            row,
            {
                "anchor_id",
                "rate_hz",
                "scheme_id",
                "public_scene_n",
                "expected_rho",
                "rho_tolerance",
            },
            "blind-anchor row",
        )
        if (
            row["rate_hz"],
            row["scheme_id"],
            row["public_scene_n"],
            row["expected_rho"],
        ) != expected[row["anchor_id"]]:
            raise ContractError(f"Public anchor constant drift: {row['anchor_id']}")
        if row["rho_tolerance"] != 0.000001:
            raise ContractError(f"Public anchor tolerance drift: {row['anchor_id']}")
    return receipt


def validate_blind_anchor_role(
    path: Path,
    *,
    base: Path,
    contract: dict[str, Any],
) -> Path:
    """Bind the G2 role to one installed, reviewed blind-anchor receipt."""

    policy = contract.get("blind_anchor_contract")
    if not isinstance(policy, dict):
        raise ContractError("Missing blind-anchor contract")
    expected_policy = {
        "runtime_install_root": str(BLIND_ANCHOR_RUNTIME_ROOT),
        "runtime_receipt_path": str(BLIND_ANCHOR_RUNTIME_PATH),
        "runtime_receipt_size_bytes": BLIND_ANCHOR_SIZE_BYTES,
        "runtime_receipt_sha256": BLIND_ANCHOR_SHA256,
        "snapshot_receipt_usage": "REVIEW_PROVENANCE_ONLY_NOT_RUNTIME_INPUT",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ContractError(f"Blind-anchor runtime contract drift: {key}")

    root = base / BLIND_ANCHOR_RUNTIME_ROOT
    expected_path = Path(os.path.abspath(base / BLIND_ANCHOR_RUNTIME_PATH))
    resolved = require_contained_regular_file(path, [root])
    if resolved != expected_path:
        raise ContractError("G2 blind-anchor receipt is outside its fixed install path")
    if resolved.stat().st_size != BLIND_ANCHOR_SIZE_BYTES:
        raise ContractError("G2 blind-anchor receipt size drift")
    if sha256_file(resolved) != BLIND_ANCHOR_SHA256:
        raise ContractError("G2 blind-anchor receipt SHA-256 drift")
    validate_anchor_receipt(resolved)
    return resolved


def validate_g2_input_roles(
    *,
    manifest_path: Path,
    contract: dict[str, Any],
    base: Path,
) -> dict[str, Path]:
    """Validate G2 roles through the shared launcher/job root and anchor policy."""

    roles = validate_input_manifest_g2(
        manifest_path=manifest_path,
        contract=contract,
        allowed_roots=g2_input_allowed_roots(base),
    )
    roles["blind_anchor_receipt"] = validate_blind_anchor_role(
        roles["blind_anchor_receipt"],
        base=base,
        contract=contract,
    )
    return roles


def validate_declassification_export_receipts(
    *,
    export_receipt_path: Path,
    done_receipt_path: Path,
    sanitization_receipt_path: Path,
    file_manifest_path: Path,
    expected_bundle_root: Path,
) -> dict[str, Any]:
    export_receipt = load_json(export_receipt_path)
    if export_receipt_path.read_bytes() != canonical_json_bytes(export_receipt):
        raise ContractError("Declassification export receipt is not canonical JSON")
    require_exact_keys(
        export_receipt,
        {
            "schema_version",
            "status",
            "operation",
            "rating_access",
            "score_stripped_bundle_root",
            "sanitization_receipt_sha256",
            "file_manifest_sha256",
            "universe_segment_count",
            "geometry_available_scene_count",
        },
        "declassification export receipt",
    )
    if (
        export_receipt["schema_version"] != "rq014-g2-declassification-export-receipt-v1"
        or export_receipt["status"] != "PASS"
        or export_receipt["operation"] != "rq014_g2_declassification_export"
        or export_receipt["rating_access"] != "NONE"
    ):
        raise ContractError("Declassification export receipt did not PASS the fixed operation")
    if Path(export_receipt["score_stripped_bundle_root"]).resolve() != expected_bundle_root.resolve():
        raise ContractError("Declassification export receipt binds the wrong bundle root")
    if export_receipt["sanitization_receipt_sha256"] != sha256_file(
        sanitization_receipt_path
    ):
        raise ContractError("Export receipt sanitization hash mismatch")
    if export_receipt["file_manifest_sha256"] != sha256_file(file_manifest_path):
        raise ContractError("Export receipt file-manifest hash mismatch")
    if (
        export_receipt["universe_segment_count"] != 479
        or not isinstance(export_receipt["geometry_available_scene_count"], int)
        or not 0 <= export_receipt["geometry_available_scene_count"] <= 479
    ):
        raise ContractError("Export receipt scene counts are invalid")

    done = load_json(done_receipt_path)
    if done_receipt_path.read_bytes() != canonical_json_bytes(done):
        raise ContractError("Declassification DONE receipt is not canonical JSON")
    require_exact_keys(
        done,
        {"schema_version", "operation", "receipt_sha256", "status"},
        "declassification DONE receipt",
    )
    if (
        done["schema_version"] != "rq014-managed-operation-done-v1"
        or done["operation"] != "rq014_g2_declassification_export"
        or done["status"] != "PASS"
        or done["receipt_sha256"] != sha256_file(export_receipt_path)
    ):
        raise ContractError("Declassification DONE receipt does not bind the PASS receipt")
    return export_receipt


def _reject_wod_rating_semantics(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            semantic = _normalize_semantic(str(key))
            if (
                "human_rating" in semantic
                or "preference_score" in semantic
                or semantic.startswith("rating_")
                or semantic.endswith("_rating")
                or semantic in {"candidate_scores", "rating", "ratings"}
            ):
                raise ContractError(f"WOD rating semantic is forbidden in {label}: {key}")
            _reject_wod_rating_semantics(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_wod_rating_semantics(item, f"{label}[{index}]")
    elif isinstance(value, str):
        semantic = _normalize_semantic(value)
        if "human_rating" in semantic or "preference_score" in semantic:
            raise ContractError(f"WOD rating semantic is forbidden in {label}")


def validate_interhub_source_manifest(
    path: Path,
    *,
    snapshot_root: Path,
) -> dict[str, Any]:
    manifest = load_json(path)
    if path.read_bytes() != canonical_json_bytes(manifest):
        raise ContractError("InterHub source manifest is not canonical JSON")
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "dataset_id",
            "cohort",
            "contains_wod_e2e_ratings",
            "producer",
            "files",
        },
        "InterHub source manifest",
    )
    if (
        manifest["schema_version"] != "rq014-interhub-human-ipv-source-manifest-v1"
        or manifest["dataset_id"] != "InterHub"
        or manifest["cohort"] != "pure_hv_hv"
        or manifest["contains_wod_e2e_ratings"] is not False
    ):
        raise ContractError("InterHub source-manifest identity or rating boundary drift")
    producer = manifest["producer"]
    if not isinstance(producer, dict):
        raise ContractError("InterHub producer must be an object")
    require_exact_keys(
        producer,
        {"git_commit", "code_sha256", "environment_sha256"},
        "InterHub producer",
    )
    if not HEX40.fullmatch(str(producer["git_commit"])):
        raise ContractError("InterHub producer commit is malformed")
    for key in ("code_sha256", "environment_sha256"):
        require_sha256(producer[key], f"InterHub producer {key}")
    rows = manifest["files"]
    if not isinstance(rows, list) or not rows:
        raise ContractError("InterHub source manifest must contain files")
    file_ids: set[str] = set()
    paths: set[Path] = set()
    roles: set[str] = set()
    allowed_roles = {"human_ipv_trajectory_source", "path_type_mapping", "provenance"}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("InterHub source row must be an object")
        require_exact_keys(
            row,
            {
                "file_id",
                "role",
                "absolute_path",
                "sha256",
                "size_bytes",
                "format",
                "contains_wod_e2e_ratings",
            },
            "InterHub source row",
        )
        if not SAFE_ID.fullmatch(str(row["file_id"])) or row["file_id"] in file_ids:
            raise ContractError("Unsafe or duplicate InterHub file ID")
        file_ids.add(row["file_id"])
        if row["role"] not in allowed_roles:
            raise ContractError(f"Unknown InterHub source role: {row['role']}")
        roles.add(row["role"])
        if row["contains_wod_e2e_ratings"] is not False:
            raise ContractError("InterHub source row contains WOD-E2E ratings")
        source = require_contained_regular_file(Path(row["absolute_path"]), [snapshot_root])
        if source in paths:
            raise ContractError("Aliased InterHub source path")
        paths.add(source)
        expected_format = {".csv": "RFC4180_CSV", ".json": "canonical_JSON"}.get(
            source.suffix.lower()
        )
        if expected_format is None or row["format"] != expected_format:
            raise ContractError("InterHub source format is not allowlisted")
        if source.stat().st_size != row["size_bytes"]:
            raise ContractError("InterHub source size mismatch")
        require_file_hash(source, row["sha256"], f"InterHub source {row['file_id']}")
        if source.suffix.lower() == ".json":
            _reject_wod_rating_semantics(load_json(source), f"InterHub source {row['file_id']}")
        else:
            with source.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            for column in header:
                _reject_wod_rating_semantics(column, f"InterHub header {row['file_id']}")
    if not {"human_ipv_trajectory_source", "path_type_mapping"} <= roles:
        raise ContractError("InterHub source manifest lacks trajectories or path-type mapping")
    return manifest


def validate_wod_path_type_mapping_manifest(
    path: Path,
    *,
    mapping_root: Path,
) -> dict[str, Any]:
    """Validate the separately frozen WOD path-type mapping input."""

    manifest = load_json(path)
    if path.read_bytes() != canonical_json_bytes(manifest):
        raise ContractError("WOD path-type mapping manifest is not canonical JSON")
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "contains_rating",
            "mapping",
            "row_count",
            "key_columns",
            "value_column",
            "allowed_values",
        },
        "WOD path-type mapping manifest",
    )
    if (
        manifest["schema_version"] != "rq014-wod-path-type-mapping-manifest-v1"
        or manifest["contains_rating"] is not False
        or manifest["key_columns"] != ["segment_id", "tstar_context_step"]
        or manifest["value_column"] != "path_type"
        or manifest["allowed_values"] != ["CP", "HO", "MP", "F"]
    ):
        raise ContractError("WOD path-type mapping manifest identity drift")
    reference = manifest["mapping"]
    if not isinstance(reference, dict):
        raise ContractError("WOD path-type mapping reference is not an object")
    require_exact_keys(
        reference,
        {"path", "size_bytes", "sha256", "format"},
        "WOD path-type mapping reference",
    )
    source = require_contained_regular_file(Path(reference["path"]), [mapping_root])
    if source != (mapping_root / "wod_path_type_mapping.csv").resolve():
        raise ContractError("WOD path-type mapping filename or root drift")
    if (
        reference["format"] != "RFC4180_CSV"
        or not isinstance(reference["size_bytes"], int)
        or isinstance(reference["size_bytes"], bool)
        or source.stat().st_size != reference["size_bytes"]
    ):
        raise ContractError("WOD path-type mapping format or size mismatch")
    require_file_hash(source, reference["sha256"], "WOD path-type mapping")
    rows = _read_csv(
        source,
        expected_columns=["segment_id", "tstar_context_step", "path_type"],
        forbidden=set(),
    )
    for row in rows:
        if not SAFE_ID.fullmatch(row["segment_id"]):
            raise ContractError("Unsafe WOD path-type mapping segment ID")
        try:
            step = int(row["tstar_context_step"])
        except ValueError as exc:
            raise ContractError("WOD path-type mapping step is not an integer") from exc
        if str(step) != row["tstar_context_step"] or step < 0:
            raise ContractError("WOD path-type mapping step is not canonical nonnegative integer")
        if row["path_type"] not in {"CP", "HO", "MP", "F"}:
            raise ContractError("Unknown WOD path type")
    _assert_unique(
        rows,
        ["segment_id", "tstar_context_step"],
        "WOD path-type mapping",
    )
    if (
        not isinstance(manifest["row_count"], int)
        or isinstance(manifest["row_count"], bool)
        or manifest["row_count"] != len(rows)
    ):
        raise ContractError("WOD path-type mapping row count mismatch")
    return {
        "manifest_sha256": sha256_file(path),
        "mapping_sha256": reference["sha256"],
        "row_count": len(rows),
    }


def validate_wod_mapping_registry_binding(
    mapping_receipt: dict[str, Any],
    materialization_ledger: dict[str, Any],
) -> None:
    """Require the installed WOD mapping table to equal its reviewed registry binding."""

    binding_id = "valid.envelope.wod_path_type_mapping.mapping_table_sha256"
    expected = materialization_ledger.get("bindings", {}).get(binding_id)
    if expected != mapping_receipt.get("mapping_sha256"):
        raise ContractError("WOD path-type mapping table differs from reviewed registry binding")


def validate_materialization_ledger(
    *,
    ledger_path: Path,
    repo_root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    from scripts.rq014.materialize_registry import (
        canonical_bytes,
        count_placeholder,
        get_pointer,
        set_pointer,
    )

    ledger = load_json(ledger_path)
    if ledger_path.read_bytes() != canonical_json_bytes(ledger):
        raise ContractError("Materialization ledger is not canonical JSON")
    require_exact_keys(
        ledger,
        {
            "schema_version",
            "stage",
            "execution_contract",
            "freeze_values",
            "materializer_sha256",
            "source_registries",
            "bindings",
            "outputs",
        },
        "materialization ledger",
    )
    if ledger.get("schema_version") != "rq014-registry-materialization-ledger-g2-v1":
        raise ContractError("Wrong materialization ledger schema")
    if ledger.get("stage") != "G2":
        raise ContractError("Wrong registry materialization stage")
    policy = contract["registry_binding_contract"]
    if set(ledger.get("bindings", {})) != set(policy["required_binding_ids"]):
        raise ContractError("Materialization ledger bindings are incomplete")
    for value in ledger["bindings"].values():
        require_sha256(value, "materialization binding")
    for left, right in policy["cross_registry_equalities"]:
        if ledger["bindings"][left] != ledger["bindings"][right]:
            raise ContractError(f"Materialization cross-registry binding mismatch: {left} != {right}")
    execution_contract_path = repo_root / "reports" / "plans" / "RQ014_execution_contract_v1p5.json"
    require_exact_keys(ledger["execution_contract"], {"path", "sha256"}, "ledger execution contract")
    if ledger["execution_contract"]["path"] != str(execution_contract_path.relative_to(repo_root)):
        raise ContractError("Ledger execution-contract path drift")
    require_file_hash(
        execution_contract_path,
        ledger["execution_contract"]["sha256"],
        "ledger execution contract",
    )
    materializer_path = repo_root / "scripts" / "rq014" / "materialize_registry.py"
    require_file_hash(materializer_path, ledger["materializer_sha256"], "ledger materializer")

    output_root = ledger_path.resolve().parent
    require_exact_keys(ledger["freeze_values"], {"path", "sha256"}, "ledger registry bindings")
    if ledger["freeze_values"]["path"] != "registry_bindings.g2.json":
        raise ContractError("Ledger registry-binding filename drift")
    freeze_path = require_contained_regular_file(
        output_root / ledger["freeze_values"]["path"],
        [output_root],
    )
    require_file_hash(freeze_path, ledger["freeze_values"]["sha256"], "registry bindings")
    freeze_values = load_json(freeze_path)
    if freeze_path.read_bytes() != canonical_json_bytes(freeze_values):
        raise ContractError("Registry bindings are not canonical JSON")
    require_exact_keys(freeze_values, {"schema_version", "stage", "bindings"}, "registry bindings")
    if (
        freeze_values["schema_version"] != "rq014-registry-bindings-g2-v1"
        or freeze_values["stage"] != "G2"
        or freeze_values["bindings"] != ledger["bindings"]
    ):
        raise ContractError("Registry bindings and materialization ledger disagree")

    active_registries = contract["active_registries"]
    if set(ledger["source_registries"]) != set(active_registries):
        raise ContractError("Materialization ledger source registries are incomplete")
    sources: dict[str, dict[str, Any]] = {}
    for name, relative in active_registries.items():
        source = ledger["source_registries"][name]
        require_exact_keys(source, {"path", "sha256"}, f"source registry {name}")
        if source["path"] != relative:
            raise ContractError(f"Source-registry path drift: {name}")
        path = require_contained_regular_file(repo_root / relative, [repo_root])
        require_file_hash(path, source["sha256"], f"source registry {name}")
        sources[name] = load_json(path)

    expected_documents = copy.deepcopy(sources)
    binding_mode = policy.get("source_binding_mode", "MATERIALIZE_PLACEHOLDERS")
    if binding_mode == "VERIFY_PREFILLED_EXACT":
        for binding_id in policy["required_binding_ids"]:
            target = policy["binding_targets"][binding_id]
            actual = get_pointer(expected_documents[target["registry"]], target["pointer"])
            if actual != ledger["bindings"][binding_id]:
                raise ContractError(f"Prefilled source binding mismatch: {binding_id}")
    elif binding_mode == "MATERIALIZE_PLACEHOLDERS":
        for binding_id in policy["required_binding_ids"]:
            target = policy["binding_targets"][binding_id]
            set_pointer(
                expected_documents[target["registry"]],
                target["pointer"],
                ledger["bindings"][binding_id],
            )
    else:
        raise ContractError(f"Unknown registry source-binding mode: {binding_mode}")
    if any(count_placeholder(document) for document in expected_documents.values()):
        raise ContractError("Expected materialized registry retains a placeholder")

    if set(ledger["outputs"]) != set(active_registries):
        raise ContractError("Materialization ledger outputs are incomplete")
    for name, output in ledger["outputs"].items():
        require_exact_keys(output, {"path", "sha256"}, f"materialized registry {name}")
        expected_name = f"{name}.materialized.json"
        if output["path"] != expected_name:
            raise ContractError(f"Materialized-registry filename drift: {name}")
        path = require_contained_regular_file(output_root / output["path"], [output_root])
        require_file_hash(path, output["sha256"], f"materialized registry {name}")
        if path.read_bytes() != canonical_bytes(expected_documents[name]):
            raise ContractError(f"Materialized registry content mismatch: {name}")
    return ledger


def run_preflight(
    *,
    base: Path,
    repo_root: Path,
    execution_contract_path: Path,
    m3_artifact_ref: dict[str, Any],
    input_manifest_path: Path,
    sanitization_receipt_path: Path,
    materialization_ledger_path: Path,
    declassification_export_receipt_path: Path,
    declassification_export_done_path: Path,
    expected_exporter_git_commit: str,
    expected_exporter_environment_sha256: str,
) -> dict[str, Any]:
    contract = load_json(execution_contract_path)
    if contract.get("schema_version") != "rq014-execution-contract-v1p5":
        raise ContractError("Wrong RQ014 execution contract")
    m3_receipt = validate_m3_artifact_ref(m3_artifact_ref, base=base, contract=contract)
    roles = validate_g2_input_roles(
        manifest_path=input_manifest_path,
        contract=contract,
        base=base,
    )
    if roles["wod_score_stripped_sanitization_receipt"] != sanitization_receipt_path.resolve():
        raise ContractError("Run spec and G2 manifest name different sanitization receipts")
    bundle_manifest = roles["wod_score_stripped_bundle_manifest"]
    bundle_root = bundle_manifest.parent
    expected_bundle_root = base / "inputs" / "RQ014" / "wod_rated479_score_stripped" / "v1"
    if bundle_root != expected_bundle_root.resolve():
        raise ContractError("G2 WOD bundle is not at the fixed managed input root")
    if sanitization_receipt_path.resolve() != bundle_root / "sanitization_receipt.json":
        raise ContractError("G2 sanitization receipt is outside the canonical bundle")
    path_mapping = validate_wod_path_type_mapping_manifest(
        roles["wod_path_type_mapping_manifest"],
        mapping_root=base / "inputs" / "RQ014" / "wod_path_type_mapping" / "v1",
    )
    export_receipt = validate_declassification_export_receipts(
        export_receipt_path=declassification_export_receipt_path,
        done_receipt_path=declassification_export_done_path,
        sanitization_receipt_path=sanitization_receipt_path,
        file_manifest_path=bundle_manifest,
        expected_bundle_root=expected_bundle_root,
    )
    inventory_path = (
        repo_root
        / "reports"
        / "studies"
        / "RQ014_wod_e2e_rating_recovery"
        / "02_g2_preflight"
        / "RQ014_declassification_source_inventory_20260712.json"
    )
    inventory = load_json(inventory_path)
    expected_sources = {row["role"]: row["sha256"] for row in inventory["files"]}
    schema_path = repo_root / "reports" / "plans" / "RQ014_score_stripped_schema_v1.json"
    bundle = validate_score_stripped_bundle(
        bundle_root=bundle_root,
        schema_path=schema_path,
        file_manifest_path=bundle_manifest,
        receipt_path=sanitization_receipt_path,
        full_hash=True,
        expected_exporter_git_commit=expected_exporter_git_commit,
        expected_exporter_environment_sha256=expected_exporter_environment_sha256,
        expected_exporter_code_sha256=sha256_file(
            repo_root / "scripts" / "rq014" / "export_score_stripped_bundle.py"
        ),
        expected_source_artifacts=expected_sources,
    )
    if export_receipt["geometry_available_scene_count"] != bundle["geometry_available_scene_count"]:
        raise ContractError("Export receipt and canonical bundle geometry counts differ")
    ledger = validate_materialization_ledger(
        ledger_path=materialization_ledger_path,
        repo_root=repo_root,
        contract=contract,
    )
    validate_wod_mapping_registry_binding(path_mapping, ledger)
    return {
        "schema_version": "rq014-g2-contract-preflight-receipt-v1",
        "status": "PASS",
        "operation": "rq014_g2_contract_preflight",
        "rating_access": "NONE",
        "rating_join": "NONE",
        "observed_statistics": "NONE",
        "anchor_statuses": [
            "ATTESTED_RECEIPT_MATCH",
            "RATING_FREE_PREDICTOR_PARITY_NOT_EXECUTED_BY_CONTRACT_PREFLIGHT",
            "RHO_NOT_RECOMPUTED_BY_DESIGN",
        ],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "execution_contract_sha256": sha256_file(execution_contract_path),
        "materialization_ledger_sha256": sha256_file(materialization_ledger_path),
        "m3_artifact_input_receipt": m3_receipt,
        "wod_path_type_mapping": path_mapping,
        "declassification_export_receipt_sha256": sha256_file(
            declassification_export_receipt_path
        ),
        "declassification_export_done_sha256": sha256_file(
            declassification_export_done_path
        ),
        "bundle": bundle,
        "materialized_registry_outputs": ledger["outputs"],
        "next_operation": "rq014_g2_resource_pilot",
        "next_operation_authorized": False,
    }
