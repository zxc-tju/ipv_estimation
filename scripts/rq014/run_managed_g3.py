#!/usr/bin/env python3
"""Run the single managed RQ014 G3R rating join and terminal screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


OPERATION = "rq014_r3_full_rating_join_and_rank"
PRIOR_OPERATION = "rq014_r2_blind_feature_build"
RATING_COLUMNS = [
    "segment_id",
    "tstar_context_step",
    "candidate_ordinal",
    "candidate_id",
    "geometry_sha256",
    "preference_score",
]
ASSOCIATIONS = ("RWS", "PSP", "PPR")
BOOTSTRAP_REPLICATES = 2000
LEDGER_DOMAIN = b"RQ014-RECOVERY-LEDGER-v2\0"
EXPECTED_BANK_RUN_ID = "RQ014_2_blind_feature_build_20260722T210000Z_e41c8792"
EXPECTED_BANK_RECEIPT_SHA256_PREFIX = "b74bb0e2"
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HEX64 = set("0123456789abcdef")


class G3RFailure(RuntimeError):
    """Fail-closed error whose details must not cross the disclosure boundary."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _verified_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
    label: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise G3RFailure(f"{label} is not a regular non-symlink file")
    stat = path.stat()
    if expected_size_bytes is not None and stat.st_size != expected_size_bytes:
        raise G3RFailure(f"{label} size binding mismatch")
    observed = _sha256_file(path)
    if not _is_hex64(expected_sha256) or observed != expected_sha256:
        raise G3RFailure(f"{label} SHA-256 binding mismatch")
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": observed}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G3RFailure("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise G3RFailure("nonfinite JSON number")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise G3RFailure("JSON root is not an object")
    return value


def _read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or b"\r" in payload or b"\n\n" in payload:
        raise G3RFailure("noncanonical JSONL framing")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        row = json.loads(line)
        if not isinstance(row, dict) or canonical_json_bytes(row) != line:
            raise G3RFailure("noncanonical JSONL row")
        rows.append(row)
    return rows


def _normalize_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite")
    return 0.0 if result == 0.0 else result


def _finite_float(value: float) -> dict[str, Any]:
    return {"kind": "FINITE_FLOAT", "value": _normalize_float(value)}


def _finite_int(value: int) -> dict[str, Any]:
    return {"kind": "FINITE_INT", "value": int(value)}


def _na(reason: str) -> dict[str, str]:
    return {"kind": "NA", "reason_code": reason}


def _support_id(segment_ids: Iterable[str]) -> str:
    ordered = sorted(segment_ids, key=lambda value: value.encode("utf-8"))
    return _sha256_bytes(b"".join(value.encode("utf-8") + b"\n" for value in ordered))


def average_midranks(values: Sequence[float]) -> list[float]:
    numeric = [_normalize_float(value) for value in values]
    order = sorted(range(len(numeric)), key=lambda index: (numeric[index], index))
    ranks = [0.0] * len(numeric)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and numeric[order[stop]] == numeric[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def weighted_pearson(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float]
) -> tuple[str, float | None]:
    if len(x) == 0 or len(x) != len(y) or len(x) != len(weights):
        return "ASSOCIATION_NUMERICAL_FAILURE", None
    try:
        nx = [_normalize_float(value) for value in x]
        ny = [_normalize_float(value) for value in y]
        nw = [_normalize_float(value) for value in weights]
        sw = math.fsum(nw)
        if not math.isfinite(sw) or sw <= 0.0:
            return "ASSOCIATION_NUMERICAL_FAILURE", None
        mx = math.fsum(w * value for w, value in zip(nw, nx)) / sw
        my = math.fsum(w * value for w, value in zip(nw, ny)) / sw
        sxy = math.fsum(w * (vx - mx) * (vy - my) for w, vx, vy in zip(nw, nx, ny))
        sxx = math.fsum(w * (vx - mx) ** 2 for w, vx in zip(nw, nx))
        syy = math.fsum(w * (vy - my) ** 2 for w, vy in zip(nw, ny))
        if sxx <= 0.0 or syy <= 0.0:
            return "ASSOCIATION_CONSTANT", None
        result = sxy / math.sqrt(sxx * syy)
        if not math.isfinite(result):
            return "ASSOCIATION_NUMERICAL_FAILURE", None
        if -1.0 - 1e-15 <= result < -1.0:
            result = -1.0
        elif 1.0 < result <= 1.0 + 1e-15:
            result = 1.0
        elif result < -1.0 - 1e-15 or result > 1.0 + 1e-15:
            return "ASSOCIATION_NUMERICAL_FAILURE", None
        return "ASSOCIATION_AVAILABLE", 0.0 if result == 0.0 else result
    except (TypeError, ValueError, OverflowError):
        return "ASSOCIATION_NUMERICAL_FAILURE", None


SceneRow = tuple[str, tuple[float, float, float], tuple[float, float, float], str, str]


def association(method: str, scenes: Sequence[SceneRow]) -> tuple[str, float | None]:
    if not scenes:
        return "ASSOCIATION_CONSTANT", None
    if method == "RWS":
        values: list[float] = []
        for _segment, ratings, deviations, _cluster, _shard in scenes:
            status, value = weighted_pearson(
                average_midranks(ratings),
                average_midranks(deviations),
                (1.0 / 3.0,) * 3,
            )
            if status != "ASSOCIATION_AVAILABLE" or value is None:
                return status, None
            values.append(value)
        result = math.fsum(values) / len(values)
        return ("ASSOCIATION_AVAILABLE", 0.0 if result == 0.0 else result)
    ratings = [value for scene in scenes for value in scene[1]]
    deviations = [value for scene in scenes for value in scene[2]]
    weights = [1.0 / len(ratings)] * len(ratings)
    if method == "PSP":
        return weighted_pearson(
            average_midranks(ratings), average_midranks(deviations), weights
        )
    if method == "PPR":
        return weighted_pearson(ratings, deviations, weights)
    raise G3RFailure("unregistered association method")


def _quantile_type7(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    j = math.floor(h)
    g = h - j
    if j + 1 == len(ordered):
        return ordered[j]
    return (1.0 - g) * ordered[j] + g * ordered[j + 1]


def _bootstrap_interval(
    method: str,
    cell_id: str,
    support_id: str,
    scenes: Sequence[SceneRow],
    replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np

        leaderboard_id = f"{cell_id}-{method}"
        seed_object = {
            "association_id": method,
            "cell_id": cell_id,
            "leaderboard_id": leaderboard_id,
            "replicates": replicates,
            "support_id": support_id,
        }
        seed = int.from_bytes(
            hashlib.sha256(b"RQ014-ASSOCBOOT|" + canonical_json_bytes(seed_object)).digest()[:8],
            "big",
            signed=False,
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        estimates: list[float] = []
        for _replicate_index in range(replicates):
            draw = rng.integers(0, len(scenes), size=len(scenes), dtype=np.int64, endpoint=False)
            sampled = [scenes[int(index)] for index in draw]
            status, value = association(method, sampled)
            if status != "ASSOCIATION_AVAILABLE" or value is None:
                raise ValueError("bootstrap association unavailable")
            estimates.append(value)
        return (
            _finite_float(_quantile_type7(estimates, 0.025)),
            _finite_float(_quantile_type7(estimates, 0.975)),
        )
    except Exception:
        return _na("UNCERTAINTY_UNAVAILABLE"), _na("UNCERTAINTY_UNAVAILABLE")


def _stability_metrics(
    method: str,
    cell_id: str,
    scenes: Sequence[SceneRow],
    support_id: str,
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    fold_values: list[float] = []
    fold_reason = "INSUFFICIENT_FOLD_SUPPORT"
    for fold in range(5):
        subset = [
            scene
            for scene in scenes
            if int.from_bytes(
                hashlib.sha256(b"RQ014-R2-FOLD|" + scene[0].encode("utf-8")).digest()[:8],
                "big",
            )
            % 5
            == fold
        ]
        if len(subset) < 5:
            fold_values = []
            break
        status, value = association(method, subset)
        if status != "ASSOCIATION_AVAILABLE" or value is None:
            fold_reason = "NONFINITE_FOLD_RECOMPUTATION"
            fold_values = []
            break
        fold_values.append(value)
    if len(fold_values) == 5:
        fold_associations: Any = [_finite_float(value) for value in fold_values]
        fold_negative_count: Any = _finite_int(sum(value < 0.0 for value in fold_values))
        median_fold: Any = _finite_float(sorted(fold_values)[2])
    else:
        fold_associations = _na(fold_reason)
        fold_negative_count = _na(fold_reason)
        median_fold = _na(fold_reason)

    loo_values: list[float] = []
    if len(scenes) >= 2:
        for index in range(len(scenes)):
            status, value = association(method, [*scenes[:index], *scenes[index + 1 :]])
            if status != "ASSOCIATION_AVAILABLE" or value is None:
                loo_values = []
                break
            loo_values.append(value)
    leave_one_scene_out_max = (
        _finite_float(max(loo_values))
        if len(loo_values) == len(scenes) and loo_values
        else _na(
            "INSUFFICIENT_LOO_SUPPORT"
            if len(scenes) < 2
            else "NONFINITE_LOO_RECOMPUTATION"
        )
    )

    def omission_metrics(field_index: int, kind: str) -> tuple[Any, Any, Any]:
        counts = Counter(scene[field_index] for scene in scenes if scene[field_index] != "NA")
        eligible = sorted(
            (
                key
                for key, count in counts.items()
                if count >= 5 and len(scenes) - count >= 5
            ),
            key=lambda value: value.encode("utf-8"),
        )
        values: list[float] = []
        for key in eligible:
            status, value = association(method, [scene for scene in scenes if scene[field_index] != key])
            if status != "ASSOCIATION_AVAILABLE" or value is None:
                reason = f"NONFINITE_{kind}_RECOMPUTATION"
                return _finite_int(len(eligible)), _na(reason), _na(reason)
            values.append(value)
        negative = _finite_int(sum(value < 0.0 for value in values))
        if len(eligible) < 2:
            return _finite_int(len(eligible)), negative, _na(f"INSUFFICIENT_{kind}_SUPPORT")
        return _finite_int(len(eligible)), negative, _finite_float(negative["value"] / len(eligible))

    cluster_count, negative_cluster, cluster_fraction = omission_metrics(3, "CLUSTER")
    shard_count, negative_shard, shard_fraction = omission_metrics(4, "SHARD")
    ci_lower, ci_upper = _bootstrap_interval(
        method, cell_id, support_id, scenes, bootstrap_replicates
    )
    return {
        "fold_associations": fold_associations,
        "fold_negative_count": fold_negative_count,
        "median_fold_association": median_fold,
        "leave_one_scene_out_max": leave_one_scene_out_max,
        "scenario_cluster_na_scene_count_metric": _finite_int(
            sum(scene[3] == "NA" for scene in scenes)
        ),
        "eligible_cluster_count_metric": cluster_count,
        "negative_cluster_count_metric": negative_cluster,
        "leave_one_cluster_out_negative_fraction": cluster_fraction,
        "eligible_shard_count_metric": shard_count,
        "negative_shard_count_metric": negative_shard,
        "leave_one_shard_out_negative_fraction": shard_fraction,
        "association_ci_lower": ci_lower,
        "association_ci_upper": ci_upper,
    }


def _recovery_compatible(raw: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    def finite(kind: str, key: str) -> bool:
        value = metrics[key]
        return isinstance(value, dict) and value.get("kind") == kind

    folds = metrics["fold_associations"]
    return bool(
        raw["n_informative_scenes"] >= 40
        and finite("FINITE_FLOAT", "association_value")
        and metrics["association_value"]["value"] <= -0.30
        and isinstance(folds, list)
        and len(folds) == 5
        and sum(row["value"] < 0.0 for row in folds) >= 4
        and finite("FINITE_FLOAT", "median_fold_association")
        and metrics["median_fold_association"]["value"] <= -0.20
        and finite("FINITE_FLOAT", "leave_one_scene_out_max")
        and metrics["leave_one_scene_out_max"]["value"] < 0.0
        and finite("FINITE_INT", "eligible_cluster_count_metric")
        and metrics["eligible_cluster_count_metric"]["value"] >= 2
        and finite("FINITE_INT", "negative_cluster_count_metric")
        and finite("FINITE_FLOAT", "leave_one_cluster_out_negative_fraction")
        and metrics["leave_one_cluster_out_negative_fraction"]["value"] >= 0.80
    )


def _rank_sort_tuple(
    row: Mapping[str, Any], status_ranks: Mapping[str, int]
) -> list[Any]:
    metrics = row["metrics"]

    def tagged(metric: Any, kind: str, negate: bool, zero: float | int) -> tuple[int, Any]:
        if isinstance(metric, dict) and metric.get("kind") == kind:
            value = metric["value"]
            return 0, -value if negate else value
        return 1, zero

    assoc = tagged(metrics["association_value"], "FINITE_FLOAT", False, 0.0)
    folds = tagged(metrics["fold_negative_count"], "FINITE_INT", True, 0)
    clusters = tagged(
        metrics["leave_one_cluster_out_negative_fraction"], "FINITE_FLOAT", True, 0.0
    )
    n_scenes = tagged(metrics["n_informative_scenes_metric"], "FINITE_INT", True, 0)
    return [
        status_ranks[row["ledger_status"]],
        0 if row["recovery_compatible"] else 1,
        assoc[0],
        assoc[1],
        folds[0],
        folds[1],
        clusters[0],
        clusters[1],
        n_scenes[0],
        n_scenes[1],
        row["leaderboard_id"],
    ]


def _verify_bank(
    *,
    bank_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    done_path: Path,
    done_sha256: str,
    expected_scene_count: int,
    expected_cell_count: int,
    expected_run_id: str | None,
    expected_receipt_sha256_prefix: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    _verified_file(manifest_path, manifest_sha256, label="G2R umbrella manifest")
    _verified_file(receipt_path, receipt_sha256, label="G2R receipt")
    _verified_file(done_path, done_sha256, label="G2R DONE")
    manifest = _strict_json(manifest_path)
    receipt = _strict_json(receipt_path)
    done = _strict_json(done_path)
    if expected_receipt_sha256_prefix and not receipt_sha256.startswith(
        expected_receipt_sha256_prefix
    ):
        raise G3RFailure("G2R receipt is not the frozen BANK_VERIFY=PASS receipt")
    if manifest_path.parent.resolve() != bank_root.resolve():
        raise G3RFailure("G2R bank root and umbrella location differ")
    if manifest.get("operation") != PRIOR_OPERATION or manifest.get("status") != "COMPLETE":
        raise G3RFailure("G2R umbrella is not COMPLETE")
    if expected_run_id is not None and (
        manifest.get("run_id") != expected_run_id or receipt.get("run_id") != expected_run_id
    ):
        raise G3RFailure("G2R bank run identity differs")
    counts = manifest.get("counts", {})
    if (
        counts.get("registered_scene_count") != expected_scene_count
        or counts.get("registered_cell_count") != expected_cell_count
    ):
        raise G3RFailure("G2R bank registered counts differ")
    output_ref = receipt.get("output_manifest")
    if (
        receipt.get("schema_version") != "rq014-r2-blind-feature-build-receipt-v1"
        or receipt.get("operation") != PRIOR_OPERATION
        or receipt.get("status") != "PASS"
        or receipt.get("rating_value_read_count") != 0
        or receipt.get("registered_cell_count") != expected_cell_count
        or receipt.get("terminal_cell_count") != expected_cell_count
        or receipt.get("leaderboard_row_count") != 0
        or receipt.get("recovery_ledger_written") is not False
        or not isinstance(output_ref, dict)
        or output_ref.get("sha256") != manifest_sha256
    ):
        raise G3RFailure("G2R PASS receipt contract drift")
    if done != {
        "schema_version": "rq014-managed-operation-done-v1",
        "operation": PRIOR_OPERATION,
        "receipt_sha256": receipt_sha256,
        "status": "PASS",
    }:
        raise G3RFailure("G2R DONE chain mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 8:
        raise G3RFailure("G2R umbrella artifact set differs")
    paths: dict[str, Path] = {}
    for role, reference in artifacts.items():
        if not isinstance(reference, dict):
            raise G3RFailure("malformed G2R artifact reference")
        relative = reference.get("relative_path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise G3RFailure("unsafe G2R artifact path")
        path = bank_root / relative
        _verified_file(
            path,
            reference.get("sha256"),
            expected_size_bytes=reference.get("size_bytes"),
            label=f"G2R artifact {role}",
        )
        paths[role] = path
    return manifest, paths


def _load_rating_free_keys(
    bank_manifest: Mapping[str, Any], expected_scene_count: int
) -> tuple[list[str], dict[str, dict[str, str]], dict[tuple[str, int], tuple[int, str]]]:
    lineage = bank_manifest.get("lineage", {})
    input_ref = lineage.get("input_manifest")
    if not isinstance(input_ref, dict):
        raise G3RFailure("G2R umbrella omits prior input manifest")
    input_path = Path(input_ref["path"])
    _verified_file(
        input_path,
        input_ref["sha256"],
        expected_size_bytes=input_ref["size_bytes"],
        label="prior G2 input manifest",
    )
    input_manifest = _strict_json(input_path)
    entries = input_manifest.get("entries")
    if not isinstance(entries, list):
        raise G3RFailure("prior input manifest entries are malformed")
    file_rows = [row for row in entries if row.get("role") == "wod_score_stripped_bundle_manifest"]
    if len(file_rows) != 1 or file_rows[0].get("contains_rating") is not False:
        raise G3RFailure("prior score-stripped manifest role is not unique and rating-free")
    file_manifest_path = Path(file_rows[0]["absolute_path"])
    _verified_file(
        file_manifest_path,
        file_rows[0]["sha256"],
        label="score-stripped file manifest",
    )
    file_manifest = _strict_json(file_manifest_path)
    by_name = {row["relative_path"]: row for row in file_manifest.get("files", [])}
    required = {"blind_scene_manifest.csv", "candidate_states.csv"}
    if not required <= set(by_name):
        raise G3RFailure("score-stripped join-key artifacts are missing")
    paths: dict[str, Path] = {}
    for name in required:
        row = by_name[name]
        if row.get("contains_rating") is not False:
            raise G3RFailure("rating-free join-key artifact is marked rating-bearing")
        path = file_manifest_path.parent / name
        _verified_file(
            path,
            row["sha256"],
            expected_size_bytes=row["size_bytes"],
            label=name,
        )
        paths[name] = path

    scene_meta: dict[str, dict[str, str]] = {}
    with paths["blind_scene_manifest.csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required_columns = {
            "segment_id",
            "tstar_context_step",
            "source_shard_id",
            "scenario_cluster",
        }
        if reader.fieldnames is None or not required_columns <= set(reader.fieldnames):
            raise G3RFailure("blind-scene join metadata columns differ")
        for row in reader:
            segment = row["segment_id"]
            if segment in scene_meta:
                raise G3RFailure("duplicate blind-scene segment")
            scene_meta[segment] = row
    if len(scene_meta) != expected_scene_count:
        raise G3RFailure("blind-scene universe count differs")
    segments = sorted(scene_meta, key=lambda value: value.encode("utf-8"))

    geometry: dict[tuple[str, int], tuple[int, str]] = {}
    with paths["candidate_states.csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        needed = {
            "segment_id",
            "tstar_context_step",
            "candidate_ordinal",
            "candidate_id",
            "geometry_sha256",
        }
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise G3RFailure("candidate-state join metadata columns differ")
        for row in reader:
            ordinal = int(row["candidate_ordinal"])
            if ordinal not in (1, 2, 3) or row["candidate_id"] != f"C{ordinal}":
                raise G3RFailure("candidate-state identity drift")
            key = (row["segment_id"], ordinal)
            value = (int(row["tstar_context_step"]), row["geometry_sha256"])
            if key in geometry and geometry[key] != value:
                raise G3RFailure("candidate-state geometry is not invariant")
            geometry[key] = value
    return segments, scene_meta, geometry


def _read_ratings(
    path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[dict[tuple[str, int, int, str], list[float | None]], dict[str, Any]]:
    source_ref = _verified_file(
        path,
        expected_sha256,
        expected_size_bytes=expected_size_bytes,
        label="rating source",
    )
    values: dict[tuple[str, int, int, str], list[float | None]] = defaultdict(list)
    row_count = 0
    nonfinite_count = 0
    missing_count = 0
    duplicate_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != RATING_COLUMNS:
            raise G3RFailure("rating source columns differ from the frozen interface")
        for row in reader:
            row_count += 1
            segment = row["segment_id"]
            ordinal = int(row["candidate_ordinal"])
            if ordinal not in (1, 2, 3) or row["candidate_id"] != f"C{ordinal}":
                raise G3RFailure("rating source candidate identity drift")
            geometry = row["geometry_sha256"]
            if not _is_hex64(geometry):
                raise G3RFailure("rating source geometry digest is malformed")
            key = (segment, int(row["tstar_context_step"]), ordinal, geometry)
            if key in values:
                duplicate_count += 1
            raw = row["preference_score"]
            if raw == "":
                missing_count += 1
                value = None
            else:
                try:
                    value = _normalize_float(raw)
                except ValueError:
                    nonfinite_count += 1
                    value = float("nan")
            values[key].append(value)
    key_payload = b"".join(
        (f"{segment}\t{step}\t{ordinal}\t{geometry}\n").encode("utf-8")
        for segment, step, ordinal, geometry in sorted(
            values, key=lambda item: (item[0].encode("utf-8"), item[1], item[2], item[3])
        )
    )
    return values, {
        "source_ref": source_ref,
        "rating_value_read_count": row_count,
        "rating_row_count": row_count,
        "rating_source_key_count": len(values),
        "rating_source_keyset_sha256": _sha256_bytes(key_payload),
        "duplicate_key_count": duplicate_count,
        "missing_value_count": missing_count,
        "nonfinite_value_count": nonfinite_count,
    }


def _rollup_contract(lane: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    screen = lane["full_data_recovery_screen"]
    rollup = screen["upstream_terminal_rollup"]
    rows = rollup["rows"]
    by_status = {row["upstream_status"]: row for row in rows}
    if len(by_status) != len(rows):
        raise G3RFailure("terminal rollup contains duplicate statuses")
    return by_status, rollup["ledger_status_rank"]


JoinedRating = tuple[
    str,
    Optional[tuple[float, float, float]],
    tuple[tuple[str, int, int, str], ...],
]


def _join_ratings_once(
    *,
    segments: Sequence[str],
    geometry: Mapping[tuple[str, int], tuple[int, str]],
    ratings: Mapping[tuple[str, int, int, str], Sequence[float | None]],
) -> tuple[dict[str, JoinedRating], int]:
    """Materialize the one geometry-keyed rating join before any cell statistics."""
    joined: dict[str, JoinedRating] = {}
    joined_key_count = 0
    for segment in segments:
        expected_keys: list[tuple[str, int, int, str]] = []
        for ordinal in (1, 2, 3):
            key_geometry = geometry.get((segment, ordinal))
            if key_geometry is None:
                joined[segment] = ("RATING_JOIN_KEY_MISSING", None, tuple(expected_keys))
                break
            step, digest = key_geometry
            expected_keys.append((segment, step, ordinal, digest))
        else:
            slot_counts = [len(ratings.get(key, ())) for key in expected_keys]
            if any(count > 1 for count in slot_counts):
                joined[segment] = (
                    "RATING_JOIN_KEY_AMBIGUOUS",
                    None,
                    tuple(expected_keys),
                )
                continue
            if any(count == 0 for count in slot_counts):
                joined[segment] = ("RATING_JOIN_KEY_MISSING", None, tuple(expected_keys))
                continue
            joined_key_count += 3
            values = tuple(ratings[key][0] for key in expected_keys)
            if any(value is None for value in values):
                joined[segment] = ("RATING_VALUE_MISSING", None, tuple(expected_keys))
                continue
            numeric = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in numeric):
                joined[segment] = ("RATING_VALUE_NONFINITE", None, tuple(expected_keys))
                continue
            if len(set(numeric)) < 2:
                joined[segment] = ("RATING_VECTOR_CONSTANT", None, tuple(expected_keys))
                continue
            joined[segment] = ("AVAILABLE", numeric, tuple(expected_keys))
    if set(joined) != set(segments):
        raise G3RFailure("single rating join did not terminate every base scene")
    return joined, joined_key_count


def _first_failure(
    failures: Sequence[tuple[str, str]], rollup: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    stage_rank = {"K": 0, "F": 1, "R": 2, "D": 3, "I": 4}
    candidates = []
    for stage, status in failures:
        row = rollup.get(status)
        if row is None or row["stage"] != stage:
            raise G3RFailure("unmapped or stage-inconsistent terminal status")
        candidates.append((stage_rank[stage], row["reason_priority"], row["reason_code"], row))
    if not candidates:
        raise G3RFailure("missing terminal failure for empty support")
    return min(candidates, key=lambda item: (item[0], item[1], item[2].encode("utf-8")))[3]


def _build_cell_rows(
    *,
    cell: Mapping[str, Any],
    segments: Sequence[str],
    scene_meta: Mapping[str, Mapping[str, str]],
    geometry: Mapping[tuple[str, int], tuple[int, str]],
    joined_ratings: Mapping[str, JoinedRating],
    masks: Mapping[tuple[str, str], Mapping[str, Any]],
    features: Mapping[tuple[str, str, int], Mapping[str, Any]],
    rollup: Mapping[str, Mapping[str, Any]],
    status_ranks: Mapping[str, int],
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell_id = str(cell["cell_id"])
    counts = {name: 0 for name in ("N_B", "N_K", "N_F", "N_R", "N_D", "N_I", "X_K", "X_F", "X_R", "X_D", "X_I")}
    counts["N_B"] = len(segments)
    reason_counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    informative: list[SceneRow] = []
    for segment in segments:
        joined_status, joined_values, _expected_keys = joined_ratings[segment]
        if joined_status in {"RATING_JOIN_KEY_MISSING", "RATING_JOIN_KEY_AMBIGUOUS"}:
            counts["X_K"] += 1
            failures.append(("K", joined_status))
            reason_counts[rollup[joined_status]["reason_code"]] += 1
            continue
        counts["N_K"] += 1

        mask = masks.get((cell_id, segment))
        feature_rows = [features.get((cell_id, segment, ordinal)) for ordinal in (1, 2, 3)]
        feature_failure: str | None = None
        deviations: list[float] = []
        if mask is None or any(row is None for row in feature_rows):
            raise G3RFailure("bank feature/mask row is missing")
        if not mask.get("all_three_available") or not mask.get("all_three_deviations_finite"):
            feature_failure = str(mask.get("scene_cell_status"))
            if feature_failure == "AVAILABLE":
                feature_failure = next(
                    (str(row.get("upstream_status")) for row in feature_rows if row.get("upstream_status") != "AVAILABLE"),
                    "INELIGIBLE_STATE_NONFINITE",
                )
        else:
            try:
                for row in feature_rows:
                    value = row["predictor_value"]
                    if value.get("kind") != "FINITE_FLOAT":
                        raise ValueError("typed NA")
                    deviations.append(_normalize_float(value["value"]))
            except (KeyError, TypeError, ValueError):
                feature_failure = "INELIGIBLE_STATE_NONFINITE"
        if feature_failure is not None:
            if feature_failure not in rollup:
                raise G3RFailure("bank feature status is absent from terminal rollup")
            counts["X_F"] += 1
            failures.append(("F", feature_failure))
            reason_counts[rollup[feature_failure]["reason_code"]] += 1
            continue
        counts["N_F"] += 1

        if joined_status in {"RATING_VALUE_MISSING", "RATING_VALUE_NONFINITE"}:
            status = joined_status
            counts["X_R"] += 1
            failures.append(("R", status))
            reason_counts[rollup[status]["reason_code"]] += 1
            continue
        counts["N_R"] += 1

        if joined_status == "RATING_VECTOR_CONSTANT":
            status = joined_status
            counts["X_D"] += 1
            failures.append(("D", status))
            reason_counts[rollup[status]["reason_code"]] += 1
            continue
        if joined_status != "AVAILABLE" or joined_values is None:
            raise G3RFailure("single rating join returned an unregistered state")
        rating_values = joined_values
        deviation_values = tuple(deviations)
        if len(set(deviation_values)) < 2:
            status = "DEVIATION_VECTOR_CONSTANT"
            counts["X_D"] += 1
            failures.append(("D", status))
            reason_counts[rollup[status]["reason_code"]] += 1
            continue
        counts["N_D"] += 1
        counts["N_I"] += 1
        informative.append(
            (
                segment,
                rating_values,
                deviation_values,
                str(scene_meta[segment]["scenario_cluster"]),
                str(scene_meta[segment]["source_shard_id"]),
            )
        )

    counts["N_F"] = counts["N_K"] - counts["X_F"]
    counts["N_R"] = counts["N_F"] - counts["X_R"]
    counts["N_D"] = counts["N_R"] - counts["X_D"]
    counts["N_I"] = counts["N_D"] - counts["X_I"]
    if (
        counts["N_B"] != counts["N_K"] + counts["X_K"]
        or counts["N_K"] != counts["N_F"] + counts["X_F"]
        or counts["N_F"] != counts["N_R"] + counts["X_R"]
        or counts["N_R"] != counts["N_D"] + counts["X_D"]
        or counts["N_D"] != counts["N_I"] + counts["X_I"]
        or len(informative) != counts["N_I"]
    ):
        raise G3RFailure("attrition accounting identity failed")
    support = _support_id(scene[0] for scene in informative)
    pooled_ratings = [value for scene in informative for value in scene[1]]
    pooled_deviations = [value for scene in informative for value in scene[2]]
    n_i = len(informative)
    raw = {
        **counts,
        "n_scene_universe": counts["N_B"],
        "n_blind_available_scenes": counts["N_F"],
        "n_rating_key_complete_scenes": counts["N_K"],
        "n_excluded_nonfinite_scenes": sum(
            count
            for reason, count in reason_counts.items()
            if reason in {"F_STATE_NONFINITE", "F_IPV_NUMERICAL", "R_RATING_VALUE_NONFINITE"}
        ),
        "n_excluded_constant_rating_scenes": reason_counts["D_RATING_VECTOR_CONSTANT"],
        "n_excluded_constant_deviation_scenes": reason_counts["D_DEVIATION_VECTOR_CONSTANT"],
        "n_informative_scenes": n_i,
        "n_informative_candidates": 3 * n_i,
        "rating_distinct_count_pooled": len(set(pooled_ratings)),
        "deviation_distinct_count_pooled": len(set(pooled_deviations)),
        "scene_weight_sum": math.fsum([1.0 / n_i] * n_i) if n_i else 0.0,
        "candidate_weight_sum": math.fsum([1.0 / (3 * n_i)] * (3 * n_i)) if n_i else 0.0,
        "support_id": support,
    }
    rows: list[dict[str, Any]] = []
    for method in ASSOCIATIONS:
        leaderboard_id = f"{cell_id}-{method}"
        if n_i == 0:
            outcome = _first_failure(failures, rollup)
            metrics = {
                "association_value": _na("ROW_NOT_OBSERVED"),
                "n_informative_scenes_metric": _na("ROW_NOT_OBSERVED"),
                "fold_associations": _na("ROW_NOT_OBSERVED"),
                "fold_negative_count": _na("ROW_NOT_OBSERVED"),
                "median_fold_association": _na("ROW_NOT_OBSERVED"),
                "leave_one_scene_out_max": _na("ROW_NOT_OBSERVED"),
                "scenario_cluster_na_scene_count_metric": _na("ROW_NOT_OBSERVED"),
                "eligible_cluster_count_metric": _na("ROW_NOT_OBSERVED"),
                "negative_cluster_count_metric": _na("ROW_NOT_OBSERVED"),
                "leave_one_cluster_out_negative_fraction": _na("ROW_NOT_OBSERVED"),
                "eligible_shard_count_metric": _na("ROW_NOT_OBSERVED"),
                "negative_shard_count_metric": _na("ROW_NOT_OBSERVED"),
                "leave_one_shard_out_negative_fraction": _na("ROW_NOT_OBSERVED"),
                "association_ci_lower": _na("ROW_NOT_OBSERVED"),
                "association_ci_upper": _na("ROW_NOT_OBSERVED"),
            }
        else:
            status, value = association(method, informative)
            outcome = rollup[status]
            if status == "ASSOCIATION_AVAILABLE" and value is not None:
                metrics = {
                    "association_value": _finite_float(value),
                    "n_informative_scenes_metric": _finite_int(n_i),
                    **_stability_metrics(
                        method,
                        cell_id,
                        informative,
                        support,
                        bootstrap_replicates=bootstrap_replicates,
                    ),
                }
            else:
                metrics = {
                    key: _na("ROW_NOT_OBSERVED")
                    for key in (
                        "association_value",
                        "n_informative_scenes_metric",
                        "fold_associations",
                        "fold_negative_count",
                        "median_fold_association",
                        "leave_one_scene_out_max",
                        "scenario_cluster_na_scene_count_metric",
                        "eligible_cluster_count_metric",
                        "negative_cluster_count_metric",
                        "leave_one_cluster_out_negative_fraction",
                        "eligible_shard_count_metric",
                        "negative_shard_count_metric",
                        "leave_one_shard_out_negative_fraction",
                        "association_ci_lower",
                        "association_ci_upper",
                    )
                }
        row = {
            "schema_version": "rq014-recovery-ledger-row-v2",
            "row_index": -1,
            "leaderboard_id": leaderboard_id,
            "cell_id": cell_id,
            "association_id": method,
            "ledger_status": outcome["ledger_status"],
            "reason_code": outcome["reason_code"],
            "upstream_status": outcome["upstream_status"],
            "first_failure_stage": outcome["stage"],
            "reason_priority": outcome["reason_priority"],
            "raw_denominators": dict(raw),
            "metrics": metrics,
            "recovery_compatible": False,
            "rank_sort_tuple": [],
            "prev_record_sha256": ZERO_SHA256,
            "record_sha256": ZERO_SHA256,
        }
        row["recovery_compatible"] = (
            row["ledger_status"] == "OBSERVED" and _recovery_compatible(raw, metrics)
        )
        row["rank_sort_tuple"] = _rank_sort_tuple(row, status_ranks)
        rows.append(row)
    attrition = {"cell_id": cell_id, **counts, "support_id": support, "identity_check": "PASS"}
    for reason in sorted({row["reason_code"] for row in rollup.values()}):
        attrition[f"reason_count_{reason}"] = reason_counts[reason]
    return rows, attrition


def _encode_attrition(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise G3RFailure("empty attrition table")
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _read_common_support(path: Path, segments: Sequence[str]) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["segment_id"]:
            raise G3RFailure("common-support blind manifest columns differ")
        rows = [row["segment_id"] for row in reader]
    canonical = sorted(rows, key=lambda value: value.encode("utf-8"))
    if rows != canonical or len(set(rows)) != len(rows) or not set(rows) <= set(segments):
        raise G3RFailure("common-support blind manifest identity/order differs")
    return rows


def _common_support_sensitivity(
    *,
    blind_segments: Sequence[str],
    selected_cell_id: str,
    scene_meta: Mapping[str, Mapping[str, str]],
    joined_ratings: Mapping[str, JoinedRating],
    features: Mapping[tuple[str, str, int], Mapping[str, Any]],
    blind_cell_count: int,
) -> bytes:
    scenes: list[SceneRow] = []
    for segment in blind_segments:
        joined_status, rating_values, _keys = joined_ratings[segment]
        if joined_status != "AVAILABLE" or rating_values is None:
            continue
        deviations: list[float] = []
        for ordinal in (1, 2, 3):
            row = features.get((selected_cell_id, segment, ordinal))
            if row is None or row.get("predictor_value", {}).get("kind") != "FINITE_FLOAT":
                raise G3RFailure("common-support selected-cell feature is unavailable")
            deviations.append(_normalize_float(row["predictor_value"]["value"]))
        if len(set(deviations)) < 2:
            raise G3RFailure("common-support blind manifest admits a constant deviation vector")
        scenes.append(
            (
                segment,
                rating_values,
                tuple(deviations),
                str(scene_meta[segment]["scenario_cluster"]),
                str(scene_meta[segment]["source_shard_id"]),
            )
        )
    if not blind_segments:
        empty_status = "COMMON_SUPPORT_EMPTY_BLIND"
    elif not scenes:
        empty_status = "COMMON_SUPPORT_EMPTY_AFTER_RATING"
    else:
        empty_status = ""
    rows: list[dict[str, Any]] = []
    top_status = "COMMON_SUPPORT_AVAILABLE"
    for method in ASSOCIATIONS:
        if empty_status:
            row_status = empty_status
            value = None
        else:
            status, value = association(method, scenes)
            row_status = {
                "ASSOCIATION_AVAILABLE": "COMMON_SUPPORT_AVAILABLE",
                "ASSOCIATION_CONSTANT": "COMMON_SUPPORT_ASSOCIATION_CONSTANT",
                "ASSOCIATION_NUMERICAL_FAILURE": "COMMON_SUPPORT_NUMERICAL_FAILURE",
            }[status]
        if top_status == "COMMON_SUPPORT_AVAILABLE" and row_status != top_status:
            top_status = row_status
        rows.append(
            {
                "association_id": method,
                "status": row_status,
                "association_value": (
                    _finite_float(value)
                    if row_status == "COMMON_SUPPORT_AVAILABLE" and value is not None
                    else _na("ROW_NOT_OBSERVED")
                ),
                "n_informative_scenes_metric": (
                    _finite_int(len(scenes)) if scenes else _na("ROW_NOT_OBSERVED")
                ),
            }
        )
    return canonical_json_bytes(
        {
            "schema_version": "rq014-common-support-sensitivity-v1",
            "status": top_status,
            "blind_cell_count": blind_cell_count,
            "blind_scene_count": len(blind_segments),
            "rated_informative_scene_count": len(scenes),
            "support_id": _support_id(scene[0] for scene in scenes),
            "rows": rows,
        }
    )


def _runtime_gate_manifests(bindings: Mapping[str, Any]) -> dict[str, bytes]:
    common = {
        "implementation": bindings["implementation"],
        "fixture": bindings["fixture"],
        "fixture_output_sha256": bindings["fixture_output_sha256"],
        "python_executable_sha256": bindings["python_executable_sha256"],
        "python_version": bindings["python_version"],
        "environment_manifest_sha256": bindings["environment_manifest_sha256"],
    }
    return {
        "association_kernel_manifest.json": canonical_json_bytes(
            {
                "schema_version": "rq014-association-kernel-manifest-v1",
                "status": "STATISTIC_KERNEL_SATISFIED",
                **common,
                "versions": bindings["versions"],
                "fixture_ids": [
                    "STAT_MIDRANK_TIES",
                    "STAT_PEARSON_SIGNED_ZERO",
                    "STAT_RWS_REDUCTION",
                    "STAT_PSP_REDUCTION",
                    "STAT_PPR_REDUCTION",
                    "STAT_SUBSET_RECOMPUTE",
                ],
            }
        ),
        "association_attrition_manifest.json": canonical_json_bytes(
            {
                "schema_version": "rq014-association-attrition-manifest-v1",
                "status": "ATTRITION_KERNEL_SATISFIED",
                **common,
                "terminal_rollup_sha256": bindings["terminal_rollup_sha256"],
                "fixture_ids": [
                    "ATTRITION_ALL_PASS",
                    "ATTRITION_ONE_FAILURE_EACH_STAGE",
                    "ATTRITION_MULTIPLE_REASON_PRECEDENCE",
                    "ATTRITION_MISSING_GEOMETRY_RETAINED_AT_B",
                ],
            }
        ),
    }


def build_g3r_artifacts(
    *,
    bank_manifest: Mapping[str, Any],
    bank_paths: Mapping[str, Path],
    ratings: Mapping[tuple[str, int, int, str], Sequence[float | None]],
    expected_scene_count: int,
    expected_cell_count: int,
    bootstrap_replicates: int,
    recovery_lane: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any] | None = None,
    join_audit: dict[str, int] | None = None,
) -> dict[str, bytes]:
    segments, scene_meta, geometry = _load_rating_free_keys(
        bank_manifest, expected_scene_count
    )
    predictor_rows = _read_canonical_jsonl(bank_paths["g2r_predictor_manifest"])
    feature_rows = _read_canonical_jsonl(bank_paths["g2r_blind_feature_bank"])
    mask_rows = _read_canonical_jsonl(bank_paths["g2r_availability_masks"])
    if len(predictor_rows) != expected_cell_count:
        raise G3RFailure("predictor-manifest count differs")
    predictor_rows.sort(key=lambda row: row["cell_index"])
    if [row["cell_index"] for row in predictor_rows] != list(range(expected_cell_count)):
        raise G3RFailure("predictor-manifest canonical index order differs")
    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in feature_rows:
        key = (row["cell_id"], row["segment_id"], row["candidate_ordinal"])
        if key in features:
            raise G3RFailure("duplicate feature-bank row")
        features[key] = row
    masks: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mask_rows:
        key = (row["cell_id"], row["segment_id"])
        if key in masks:
            raise G3RFailure("duplicate availability-mask row")
        masks[key] = row
    if len(features) != expected_cell_count * expected_scene_count * 3:
        raise G3RFailure("feature-bank Cartesian row count differs")
    if len(masks) != expected_cell_count * expected_scene_count:
        raise G3RFailure("availability-mask Cartesian row count differs")

    joined_ratings, joined_key_count = _join_ratings_once(
        segments=segments,
        geometry=geometry,
        ratings=ratings,
    )
    if join_audit is not None:
        join_audit["joined_key_count"] = joined_key_count
    rollup, status_ranks = _rollup_contract(recovery_lane)
    all_rows: list[dict[str, Any]] = []
    attrition_rows: list[dict[str, Any]] = []
    for cell in predictor_rows:
        rows, attrition = _build_cell_rows(
            cell=cell,
            segments=segments,
            scene_meta=scene_meta,
            geometry=geometry,
            joined_ratings=joined_ratings,
            masks=masks,
            features=features,
            rollup=rollup,
            status_ranks=status_ranks,
            bootstrap_replicates=bootstrap_replicates,
        )
        all_rows.extend(rows)
        attrition_rows.append(attrition)
    if len(all_rows) != expected_cell_count * 3:
        raise G3RFailure("terminal leaderboard row count differs")
    sorted_rows = sorted(all_rows, key=lambda row: tuple(row["rank_sort_tuple"]))
    if len({row["leaderboard_id"] for row in sorted_rows}) != len(sorted_rows):
        raise G3RFailure("leaderboard IDs are not unique")
    rank_index = {
        row["leaderboard_id"]: rank for rank, row in enumerate(sorted_rows, start=1)
    }
    if sorted(rank_index.values()) != list(range(1, len(all_rows) + 1)):
        raise G3RFailure("leaderboard ranks are not total")
    blind_segments = _read_common_support(
        bank_paths["common_support_blind_manifest"], segments
    )
    common_support = _common_support_sensitivity(
        blind_segments=blind_segments,
        selected_cell_id=sorted_rows[0]["cell_id"],
        scene_meta=scene_meta,
        joined_ratings=joined_ratings,
        features=features,
        blind_cell_count=expected_cell_count,
    )

    previous = ZERO_SHA256
    ledger_parts: list[bytes] = []
    for index, row in enumerate(all_rows):
        row["row_index"] = index
        row["prev_record_sha256"] = previous
        preimage_row = dict(row)
        del preimage_row["record_sha256"]
        record = _sha256_bytes(LEDGER_DOMAIN + canonical_json_bytes(preimage_row))
        row["record_sha256"] = record
        previous = record
        ledger_parts.append(canonical_json_bytes(row))
    ledger = b"".join(ledger_parts)
    terminal = canonical_json_bytes(
        {
            "schema_version": "rq014-recovery-ledger-terminal-v1",
            "row_count": len(all_rows),
            "terminal_record_sha256": previous,
            "ledger_size_bytes": len(ledger),
            "ledger_sha256": _sha256_bytes(ledger),
        }
    )
    ranking = canonical_json_bytes(
        {
            "schema_version": "rq014-g3r-rank-index-v1",
            "row_count": len(all_rows),
            "ranked_leaderboard_ids": [row["leaderboard_id"] for row in sorted_rows],
        }
    )
    artifacts = {
        "recovery_ledger.jsonl": ledger,
        "recovery_ledger_terminal_digest.json": terminal,
        "association_attrition.csv": _encode_attrition(attrition_rows),
        "rank_index.json": ranking,
        "common_support_sensitivity.json": common_support,
    }
    if runtime_bindings is not None:
        artifacts.update(_runtime_gate_manifests(runtime_bindings))
    return artifacts


def _artifact_ref(path: Path, schema_version: str, row_count: int) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "schema_version": schema_version,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "row_count": row_count,
    }


def _rating_access_receipt(
    *,
    run_id: str,
    created_at_utc: str,
    status: str,
    audit: Mapping[str, Any],
    joined_key_count: int,
) -> dict[str, Any]:
    source = audit.get("source_ref", {})
    return {
        "schema_version": "rq014-g3r-rating-access-receipt-v1",
        "operation": OPERATION,
        "run_id": run_id,
        "status": status,
        "source_size_bytes": int(source.get("size_bytes", 0)),
        "source_sha256": str(source.get("sha256", EMPTY_SHA256)),
        "rating_value_read_count": int(audit.get("rating_value_read_count", 0)),
        "rating_row_count": int(audit.get("rating_row_count", 0)),
        "rating_source_key_count": int(audit.get("rating_source_key_count", 0)),
        "rating_source_keyset_sha256": str(
            audit.get("rating_source_keyset_sha256", EMPTY_SHA256)
        ),
        "joined_key_count": int(joined_key_count),
        "duplicate_key_count": int(audit.get("duplicate_key_count", 0)),
        "missing_value_count": int(audit.get("missing_value_count", 0)),
        "nonfinite_value_count": int(audit.get("nonfinite_value_count", 0)),
        "created_at_utc": created_at_utc,
    }


def _failure_receipt(
    *,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    stage: str,
    failure_class: str,
    rating_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "rq014-r3-full-rating-join-and-rank-receipt-v1",
        "operation": OPERATION,
        "run_id": run_id,
        "git_commit": git_commit,
        "status": "FAIL",
        "rating_access": "CONTROLLED_SINGLE_SOURCE",
        "rating_join": "AT_MOST_ONCE",
        "rating_value_read_count": int(rating_audit.get("rating_value_read_count", 0)),
        "registered_cell_count": 320,
        "terminal_leaderboard_row_count": 0,
        "result_artifacts": _na("OUTPUTS_NOT_PUBLISHED"),
        "rating_access_receipt": _na("OUTPUTS_NOT_PUBLISHED"),
        "failure": {
            "kind": "RUNTIME_FAILURE",
            "stage": stage,
            "failure_class": failure_class,
        },
        "created_at_utc": created_at_utc,
    }


def _validate_kernel_fixture(
    *, fixture_path: Path, fixture_sha256: str, recovery_lane: Mapping[str, Any]
) -> str:
    _verified_file(fixture_path, fixture_sha256, label="G3R kernel fixture")
    fixture = _strict_json(fixture_path)
    if set(fixture) != {"schema_version", "statistics", "attrition"} or fixture.get(
        "schema_version"
    ) != "rq014-g3r-kernel-fixtures-v1":
        raise G3RFailure("G3R kernel fixture shape differs")
    statistics = fixture["statistics"]
    if average_midranks(statistics["STAT_MIDRANK_TIES"]["values"]) != statistics[
        "STAT_MIDRANK_TIES"
    ]["expected"]:
        raise G3RFailure("midrank golden fixture failed")
    pearson = statistics["STAT_PEARSON_SIGNED_ZERO"]
    pearson_result = weighted_pearson(pearson["x"], pearson["y"], pearson["weights"])
    if list(pearson_result) != [pearson["expected_status"], pearson["expected_value"]]:
        raise G3RFailure("Pearson golden fixture failed")
    scenes: list[SceneRow] = [
        ("synthetic-a", (1.0, 2.0, 3.0), (3.0, 2.0, 1.0), "A", "S1"),
        ("synthetic-b", (2.0, 3.0, 4.0), (4.0, 3.0, 2.0), "B", "S2"),
    ]
    observed: dict[str, Any] = {
        "STAT_MIDRANK_TIES": average_midranks(
            statistics["STAT_MIDRANK_TIES"]["values"]
        ),
        "STAT_PEARSON_SIGNED_ZERO": list(pearson_result),
    }
    for method in ASSOCIATIONS:
        fixture_id = f"STAT_{method}_REDUCTION"
        result = association(method, scenes)
        expected = statistics[fixture_id]
        if list(result) != [expected["expected_status"], expected["expected_value"]]:
            raise G3RFailure(f"{method} golden fixture failed")
        observed[fixture_id] = list(result)
    subset = association("PSP", scenes[1:])
    expected_subset = statistics["STAT_SUBSET_RECOMPUTE"]
    if list(subset) != [
        expected_subset["expected_status"],
        expected_subset["expected_value"],
    ]:
        raise G3RFailure("subset recomputation golden fixture failed")
    observed["STAT_SUBSET_RECOMPUTE"] = list(subset)

    rollup, _status_ranks = _rollup_contract(recovery_lane)
    attrition = fixture["attrition"]
    observed_attrition: dict[str, str] = {}
    for fixture_id, row in attrition.items():
        statuses = row["statuses"]
        if statuses == ["ASSOCIATION_AVAILABLE"]:
            selected = "ASSOCIATION_AVAILABLE"
        else:
            selected = _first_failure(
                [(rollup[status]["stage"], status) for status in statuses], rollup
            )["upstream_status"]
        if selected != row["expected"]:
            raise G3RFailure(f"{fixture_id} attrition golden fixture failed")
        observed_attrition[fixture_id] = selected
    return _sha256_bytes(
        canonical_json_bytes({"statistics": observed, "attrition": observed_attrition})
    )


def _runtime_bindings(
    *, args: argparse.Namespace, recovery_lane: Mapping[str, Any]
) -> dict[str, Any]:
    snapshot = _strict_json(args.code_snapshot)
    if (
        snapshot.get("schema_version") != "rq014-code-snapshot-v2"
        or snapshot.get("git_commit") != args.git_commit
        or not isinstance(snapshot.get("files"), list)
    ):
        raise G3RFailure("code-snapshot receipt shape differs")
    snapshot_rows = {row.get("path"): row for row in snapshot["files"]}
    implementation_path = "scripts/rq014/run_managed_g3.py"
    fixture_relative = "tests/fixtures/rq014_g3r_v1/statistics_and_attrition_goldens.json"
    for relative, expected_sha256 in (
        (implementation_path, snapshot_rows.get(implementation_path, {}).get("sha256")),
        (fixture_relative, args.kernel_fixture_sha256),
    ):
        row = snapshot_rows.get(relative)
        if (
            not isinstance(row, dict)
            or row.get("sha256") != expected_sha256
            or row.get("size_bytes") != (args.repo_root / relative).stat().st_size
        ):
            raise G3RFailure(f"code-snapshot binding is missing or drifted: {relative}")
        _verified_file(
            args.repo_root / relative,
            expected_sha256,
            expected_size_bytes=row["size_bytes"],
            label=f"code-snapshot file {relative}",
        )
    fixture_output_sha256 = _validate_kernel_fixture(
        fixture_path=args.kernel_fixture,
        fixture_sha256=args.kernel_fixture_sha256,
        recovery_lane=recovery_lane,
    )
    environment = _strict_json(args.environment_manifest)
    python_ref = environment.get("python_executable")
    versions = environment.get("package_versions")
    if not isinstance(python_ref, dict) or not isinstance(versions, dict):
        raise G3RFailure("environment runtime bindings are malformed")
    if not _is_hex64(python_ref.get("sha256")) or "numpy" not in versions:
        raise G3RFailure("environment runtime bindings are incomplete")
    rollup = recovery_lane["full_data_recovery_screen"]["upstream_terminal_rollup"]
    implementation_row = snapshot_rows[implementation_path]
    fixture_row = snapshot_rows[fixture_relative]
    return {
        "implementation": {
            "path": implementation_path,
            "size_bytes": implementation_row["size_bytes"],
            "sha256": implementation_row["sha256"],
        },
        "fixture": {
            "path": fixture_relative,
            "size_bytes": fixture_row["size_bytes"],
            "sha256": fixture_row["sha256"],
        },
        "fixture_output_sha256": fixture_output_sha256,
        "python_executable_sha256": python_ref["sha256"],
        "python_version": str(python_ref.get("version", platform.python_version())),
        "environment_manifest_sha256": args.environment_manifest_sha256,
        "versions": {"math": "stdlib", "numpy": str(versions["numpy"])},
        "terminal_rollup_sha256": _sha256_bytes(canonical_json_bytes(rollup)),
    }


def run_g3r_managed(
    args: argparse.Namespace,
    *,
    expected_scene_count: int = 479,
    expected_cell_count: int = 320,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    expected_bank_run_id: str | None = EXPECTED_BANK_RUN_ID,
    expected_bank_receipt_sha256_prefix: str = EXPECTED_BANK_RECEIPT_SHA256_PREFIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = "INPUT_CONTRACT"
    failure_class = "INPUT_CONTRACT_FAILURE"
    rating_audit: dict[str, Any] = {}
    output_root = args.output_root.resolve()
    staging = output_root / ".g3r.private.partial"
    final = output_root / "g3r"
    published = False
    try:
        if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
            raise G3RFailure("G3R output namespace already exists")
        lane_ref = _verified_file(
            args.recovery_contract,
            args.recovery_contract_sha256,
            label="recovery-lane v3 contract",
        )
        recovery_lane = _strict_json(args.recovery_contract)
        if recovery_lane.get("schema_version") != "rq014-historical-recovery-lane-v3":
            raise G3RFailure("recovery-lane schema version differs")
        _verified_file(
            args.environment_manifest,
            args.environment_manifest_sha256,
            label="environment manifest",
        )
        _verified_file(args.code_snapshot, args.code_snapshot_sha256, label="code snapshot")
        runtime_bindings = _runtime_bindings(args=args, recovery_lane=recovery_lane)
        bank_manifest, bank_paths = _verify_bank(
            bank_root=args.bank_root.resolve(),
            manifest_path=args.bank_manifest,
            manifest_sha256=args.bank_manifest_sha256,
            receipt_path=args.bank_receipt,
            receipt_sha256=args.bank_receipt_sha256,
            done_path=args.bank_done,
            done_sha256=args.bank_done_sha256,
            expected_scene_count=expected_scene_count,
            expected_cell_count=expected_cell_count,
            expected_run_id=expected_bank_run_id,
            expected_receipt_sha256_prefix=expected_bank_receipt_sha256_prefix,
        )

        stage = "RATING_ACCESS"
        failure_class = "RATING_ACCESS_FAILURE"
        ratings, rating_audit = _read_ratings(
            args.ratings_source,
            args.ratings_source_sha256,
            args.ratings_source_size_bytes,
        )
        rating_audit["recovery_contract"] = lane_ref

        stage = "SINGLE_JOIN_AND_TERMINAL_SCREEN"
        failure_class = "TERMINAL_SCREEN_FAILURE"
        staging.mkdir(parents=True, mode=0o700)
        os.chmod(staging, 0o700)
        join_audit: dict[str, int] = {}
        artifacts = build_g3r_artifacts(
            bank_manifest=bank_manifest,
            bank_paths=bank_paths,
            ratings=ratings,
            expected_scene_count=expected_scene_count,
            expected_cell_count=expected_cell_count,
            bootstrap_replicates=bootstrap_replicates,
            recovery_lane=recovery_lane,
            runtime_bindings=runtime_bindings,
            join_audit=join_audit,
        )
        for name, payload in artifacts.items():
            path = staging / name
            _write_once(path, payload)
        access_receipt = _rating_access_receipt(
            run_id=args.run_id,
            created_at_utc=args.created_at_utc,
            status="PASS",
            audit=rating_audit,
            joined_key_count=join_audit["joined_key_count"],
        )
        access_path = staging / "rating_access_receipt.json"
        _write_once(access_path, canonical_json_bytes(access_receipt))

        stage = "OUTPUT_VALIDATION"
        failure_class = "OUTPUT_INTEGRITY_FAILURE"
        ledger_rows = _read_canonical_jsonl(staging / "recovery_ledger.jsonl")
        if len(ledger_rows) != expected_cell_count * 3:
            raise G3RFailure("staged ledger row count differs")
        for index, row in enumerate(ledger_rows):
            if row["row_index"] != index:
                raise G3RFailure("staged ledger row index differs")
            preimage = dict(row)
            record = preimage.pop("record_sha256")
            if record != _sha256_bytes(LEDGER_DOMAIN + canonical_json_bytes(preimage)):
                raise G3RFailure("staged ledger record digest differs")

        stage = "ATOMIC_PUBLICATION"
        failure_class = "ATOMIC_PUBLICATION_FAILURE"
        _fsync_directory(staging)
        staging.rename(final)
        published = True
        _fsync_directory(output_root)
        result_refs = {
            "recovery_ledger": _artifact_ref(
                final / "recovery_ledger.jsonl",
                "rq014-recovery-ledger-row-v2",
                expected_cell_count * 3,
            ),
            "recovery_ledger_terminal_digest": _artifact_ref(
                final / "recovery_ledger_terminal_digest.json",
                "rq014-recovery-ledger-terminal-v1",
                1,
            ),
            "association_attrition": _artifact_ref(
                final / "association_attrition.csv",
                "rq014-association-attrition-v1",
                expected_cell_count,
            ),
            "rank_index": _artifact_ref(
                final / "rank_index.json", "rq014-g3r-rank-index-v1", 1
            ),
            "common_support_sensitivity": _artifact_ref(
                final / "common_support_sensitivity.json",
                "rq014-common-support-sensitivity-v1",
                3,
            ),
            "association_kernel_manifest": _artifact_ref(
                final / "association_kernel_manifest.json",
                "rq014-association-kernel-manifest-v1",
                1,
            ),
            "association_attrition_manifest": _artifact_ref(
                final / "association_attrition_manifest.json",
                "rq014-association-attrition-manifest-v1",
                1,
            ),
        }
        access_ref = _artifact_ref(
            final / "rating_access_receipt.json",
            "rq014-g3r-rating-access-receipt-v1",
            1,
        )
        receipt = {
            "schema_version": "rq014-r3-full-rating-join-and-rank-receipt-v1",
            "operation": OPERATION,
            "run_id": args.run_id,
            "git_commit": args.git_commit,
            "status": "PASS",
            "rating_access": "CONTROLLED_SINGLE_SOURCE",
            "rating_join": "EXACTLY_ONCE",
            "rating_value_read_count": rating_audit["rating_value_read_count"],
            "registered_cell_count": expected_cell_count,
            "terminal_leaderboard_row_count": expected_cell_count * 3,
            "result_artifacts": result_refs,
            "rating_access_receipt": access_ref,
            "failure": {"kind": "NONE"},
            "created_at_utc": args.created_at_utc,
        }
        return receipt, access_receipt
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if published and final.exists() and not final.is_symlink():
            shutil.rmtree(final)
            _fsync_directory(output_root)
        return (
            _failure_receipt(
                run_id=args.run_id,
                git_commit=args.git_commit,
                created_at_utc=args.created_at_utc,
                stage=stage,
                failure_class=failure_class,
                rating_audit=rating_audit,
            ),
            _rating_access_receipt(
                run_id=args.run_id,
                created_at_utc=args.created_at_utc,
                status="FAIL",
                audit=rating_audit,
                joined_key_count=0,
            ),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--bank-manifest-sha256", required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--bank-receipt-sha256", required=True)
    parser.add_argument("--bank-done", type=Path, required=True)
    parser.add_argument("--bank-done-sha256", required=True)
    parser.add_argument("--ratings-source", type=Path, required=True)
    parser.add_argument("--ratings-source-size-bytes", type=int, required=True)
    parser.add_argument("--ratings-source-sha256", required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--recovery-contract-sha256", required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest-sha256", required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--kernel-fixture", type=Path, required=True)
    parser.add_argument("--kernel-fixture-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    receipt, rating_receipt = run_g3r_managed(args)
    if receipt["status"] == "FAIL":
        _write_once(
            output / "rating_access_receipt.json", canonical_json_bytes(rating_receipt)
        )
    receipt_payload = canonical_json_bytes(receipt)
    _write_once(output / "rq014_r3_full_rating_join_and_rank_receipt.json", receipt_payload)
    if receipt["status"] == "PASS":
        _write_once(
            output / "DONE.json",
            canonical_json_bytes(
                {
                    "schema_version": "rq014-managed-operation-done-v1",
                    "operation": OPERATION,
                    "receipt_sha256": _sha256_bytes(receipt_payload),
                    "status": "PASS",
                }
            ),
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
