#!/usr/bin/env python3
"""Run a fixed, managed RQ014 G2 operation from the tracked checkout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from scripts.rq014.preflight import canonical_json_bytes, run_preflight
from scripts.rq014.run_resource_pilot import run_resource_pilot


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing non-regular immutable output: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite non-identical receipt: {path}")
        path.chmod(0o444)
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"G2R lineage input is not a regular non-symlink file: {path}")
    observed = _sha256_file(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"G2R lineage input SHA-256 mismatch: {path}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def _read_route_intents(bundle_root: Path, score_schema_path: Path) -> dict[str, str]:
    schema = json.loads(score_schema_path.read_text(encoding="utf-8"))
    expected_columns = schema["files"]["blind_scene_manifest.csv"]["columns"]
    path = bundle_root / "blind_scene_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError("blind-scene manifest columns differ from the frozen schema")
        rows = list(reader)
    route_intents = {row["segment_id"]: row["route_intent_name"] for row in rows}
    if len(rows) != 479 or len(route_intents) != 479:
        raise ValueError("blind-scene manifest does not contain exactly 479 unique scenes")
    return route_intents


def _read_counterpart_vehicle_flags(
    bundle_root: Path,
    score_schema_path: Path,
    required_segments: set[str],
) -> dict[str, bool]:
    schema = json.loads(score_schema_path.read_text(encoding="utf-8"))
    expected_columns = schema["files"]["counterpart_tracks.csv"]["columns"]
    identities: dict[str, set[str]] = {}
    class_confidence: dict[str, dict[str, float]] = {}
    path = bundle_root / "counterpart_tracks.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError("counterpart-track columns differ from the frozen schema")
        for row in reader:
            segment_id = row["segment_id"]
            identities.setdefault(segment_id, set()).add(row["counterpart_track_id"])
            per_class = class_confidence.setdefault(segment_id, {})
            per_class[row["class_name"]] = per_class.get(row["class_name"], 0.0) + float(
                row["detector_confidence"]
            )
    flags: dict[str, bool] = {}
    for segment_id in sorted(required_segments, key=lambda value: value.encode("utf-8")):
        segment_ids = identities.get(segment_id, set())
        if len(segment_ids) != 1 or "" in segment_ids:
            raise ValueError(
                f"G2R requires exactly one frozen counterpart identity for {segment_id}"
            )
        # Detector class labels can flip within one track (low-confidence noise); resolve to the
        # confidence-weighted majority class. Require at least one non-empty class.
        per_class = {
            name: total
            for name, total in class_confidence.get(segment_id, {}).items()
            if name != ""
        }
        if not per_class:
            raise ValueError(
                f"G2R requires at least one non-empty frozen counterpart class for {segment_id}"
            )
        resolved_class = max(per_class, key=lambda name: (per_class[name], name))
        flags[segment_id] = resolved_class.upper() == "VEHICLE"
    return flags


def _terminal_sampling(domain: Any, status: str) -> Any:
    return domain.SamplingTimelines(candidates={}, counterpart={}, terminal_status=status)


def _sampling_status_from_error(error: Exception) -> str:
    from scripts.rq014.run_resource_pilot import SourceGapError

    if isinstance(error, SourceGapError):
        return "INELIGIBLE_TIMELINE_SOURCE_GAP"
    message = str(error).lower()
    if "grid phase" in message:
        return "INELIGIBLE_TIMELINE_GRID_PHASE"
    if "duplicate" in message or "seam" in message:
        return "INELIGIBLE_TIMELINE_SEAM"
    if "nonfinite" in message:
        return "INELIGIBLE_STATE_NONFINITE"
    return "INELIGIBLE_TIMELINE_SUPPORT"


def _build_g2r_scene_inputs(
    *,
    bundle_root: Path,
    mapping_manifest_path: Path,
    score_schema_path: Path,
    registry: Any,
    route_intents: dict[str, str],
    domain: Any,
    orchestrator: Any,
) -> list[Any]:
    from scripts.rq014.run_resource_pilot import (
        PilotError,
        SourceGapError,
        _grid_tick_points,
        _load_sources,
    )

    sources = _load_sources(bundle_root, mapping_manifest_path)
    geometry_segments = {
        segment_id
        for segment_id, (_, terminal_status) in registry.by_segment.items()
        if terminal_status is None
    }
    counterpart_vehicle = _read_counterpart_vehicle_flags(
        bundle_root, score_schema_path, geometry_segments
    )
    scenes: list[Any] = []
    for segment_id in sorted(registry.by_segment, key=lambda value: value.encode("utf-8")):
        path_type, terminal_status = registry.by_segment[segment_id]
        sampling: dict[str, Any] = {}
        if terminal_status is None:
            history = sources["history"].get((segment_id,))
            counterpart = sources["counterpart"].get((segment_id,))
            futures = {
                candidate_id: sources["candidate"].get((segment_id, candidate_id))
                for candidate_id in ("C1", "C2", "C3")
            }
            if not history or not counterpart or any(value is None for value in futures.values()):
                sampling = {
                    sampling_id: _terminal_sampling(domain, "INELIGIBLE_TIMELINE_SUPPORT")
                    for sampling_id, _ in domain.SAMPLING_AXIS
                }
            else:
                for sampling_id, rate_hz in domain.SAMPLING_AXIS:
                    try:
                        candidate_ticks = {
                            candidate_id: {
                                tick: (point[1], point[2])
                                for tick, point in _grid_tick_points(
                                    history + futures[candidate_id], rate_hz
                                ).items()
                            }
                            for candidate_id in ("C1", "C2", "C3")
                        }
                        counterpart_ticks = {
                            tick: (point[1], point[2])
                            for tick, point in _grid_tick_points(
                                counterpart,
                                rate_hz,
                                interpolate_to_grid=True,
                                maximum_source_gap_s=2.0 / rate_hz,
                            ).items()
                        }
                        sampling[sampling_id] = domain.SamplingTimelines(
                            candidates=candidate_ticks,
                            counterpart=counterpart_ticks,
                        )
                    except (PilotError, SourceGapError) as exc:
                        sampling[sampling_id] = _terminal_sampling(
                            domain, _sampling_status_from_error(exc)
                        )
        scene_domain = domain.SceneDomainInput(
            segment_id=segment_id,
            path_type_or_NA=path_type,
            sampling=sampling,
            terminal_status=terminal_status,
        )
        scenes.append(
            orchestrator.SceneBlindInput(
                domain=scene_domain,
                route_intent=route_intents[segment_id],
                counterpart_is_vehicle=counterpart_vehicle.get(segment_id, False),
            )
        )
    if len(scenes) != 479:
        raise ValueError("G2R source adapter did not build exactly 479 scene inputs")
    return scenes


def _artifact_ref(module: Any, path: Path, schema_version: str, row_count: int) -> Any:
    return module.ArtifactReference(
        relative_path=path.name,
        schema_version=schema_version,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        row_count=row_count,
    )


def _runtime_failure_receipt(
    *,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    stage: str,
    failure_class: str,
    error: Exception,
    nc_gate_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "rq014-r2-blind-feature-build-receipt-v1",
        "operation": "rq014_r2_blind_feature_build",
        "run_id": run_id,
        "git_commit": git_commit,
        "status": "FAIL",
        "rating_access": "FORBIDDEN",
        "rating_join": "FORBIDDEN",
        "observed_rating_statistics": "FORBIDDEN",
        "rating_value_read_count": 0,
        "registered_cell_count": 320,
        "terminal_cell_count": 0,
        "leaderboard_row_count": 0,
        "recovery_ledger_written": False,
        "nc_gate_status": nc_gate_status,
        "output_manifest": {"kind": "NA", "reason_code": "OUTPUTS_NOT_PUBLISHED"},
        "failure": {
            "kind": "RUNTIME_FAILURE",
            "stage": stage,
            "failure_class": failure_class,
            "message": str(error) or type(error).__name__,
        },
        "created_at_utc": created_at_utc,
    }


def _g2r_pass_receipt(
    *,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    output_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "rq014-r2-blind-feature-build-receipt-v1",
        "operation": "rq014_r2_blind_feature_build",
        "run_id": run_id,
        "git_commit": git_commit,
        "status": "PASS",
        "rating_access": "FORBIDDEN",
        "rating_join": "FORBIDDEN",
        "observed_rating_statistics": "FORBIDDEN",
        "rating_value_read_count": 0,
        "registered_cell_count": 320,
        "terminal_cell_count": 320,
        "leaderboard_row_count": 0,
        "recovery_ledger_written": False,
        "nc_gate_status": "PASS",
        "output_manifest": {
            "relative_path": "g2r_output_manifest.json",
            "schema_version": "rq014-g2r-output-manifest-v1",
            "size_bytes": output_manifest_path.stat().st_size,
            "sha256": _sha256_file(output_manifest_path),
            "row_count": 1,
        },
        "failure": {"kind": "NONE"},
        "created_at_utc": created_at_utc,
    }


def run_g2r_managed(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.rq014 import build_g2r_blind_outputs as orchestrator
    from scripts.rq014 import build_wod_m3_anchors as anchors
    from scripts.rq014 import build_wod_scene_anchor_domain as domain

    stage = "INPUT_CONTRACT"
    failure_class = "INPUT_CONTRACT_FAILURE"
    nc_status = "FAIL"
    output_root = args.output_root.resolve()
    staging = output_root / ".g2r.partial"
    final = output_root / "g2r"
    published = False
    try:
        if args.g2r_output_contract.resolve() != (
            args.repo_root.resolve() / "reports/plans/RQ014_g2r_output_contract_v1.json"
        ):
            raise ValueError("G2R output-contract path is not the reviewed closed-snapshot path")
        _file_ref(
            args.g2r_output_contract,
            args.g2r_output_contract_sha256,
        )
        if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
            raise ValueError("G2R final or staging output path already exists")
        expected_refs = {
            "input_manifest": _file_ref(args.input_manifest),
            "sanitization_receipt": _file_ref(args.sanitization_receipt),
            "materialization_ledger": _file_ref(args.materialization_ledger),
            "wod_path_type_mapping_manifest": _file_ref(
                args.wod_path_type_mapping_manifest
            ),
            "m3_artifact": _file_ref(args.m3_artifact, args.m3_artifact_sha256),
            "environment_manifest": _file_ref(args.environment_manifest),
            "python_executable": _file_ref(args.python_executable),
            "contract_preflight_receipt": _file_ref(args.contract_preflight_receipt),
            "contract_preflight_done": _file_ref(args.contract_preflight_done),
            "resource_pilot_receipt": _file_ref(args.resource_pilot_receipt),
            "resource_pilot_done": _file_ref(args.resource_pilot_done),
            "code_snapshot": _file_ref(args.code_snapshot),
        }
        if expected_refs["m3_artifact"]["size_bytes"] != args.m3_artifact_size_bytes:
            raise ValueError("G2R M3 artifact size differs from the frozen binding")
        for receipt_key, done_key, operation in (
            (
                "contract_preflight_receipt",
                "contract_preflight_done",
                "rq014_g2_contract_preflight",
            ),
            ("resource_pilot_receipt", "resource_pilot_done", "rq014_g2_resource_pilot"),
        ):
            receipt = json.loads(Path(expected_refs[receipt_key]["path"]).read_text("utf-8"))
            done = json.loads(Path(expected_refs[done_key]["path"]).read_text("utf-8"))
            if receipt.get("status") != "PASS" or done != {
                "schema_version": "rq014-managed-operation-done-v1",
                "operation": operation,
                "receipt_sha256": expected_refs[receipt_key]["sha256"],
                "status": "PASS",
            }:
                raise ValueError(f"G2R prior {operation} lineage is not an exact PASS chain")

        model_manifest_path = args.repo_root / "models/rq009_m3/manifest.json"
        model_manifest_ref = _file_ref(model_manifest_path, args.m3_manifest_sha256)
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        artifact_binding = model_manifest.get("artifact")
        if artifact_binding != {
            "compression": "joblib-lzma-3",
            "legacy_sha256": "bf9a0c7ae41ba9efcb2ad997aaac1b7881d7788cf8dadd01252c17ed7a6b0ba5",
            "path": "m3_scorer.joblib",
            "sha256": args.m3_artifact_sha256,
            "size_bytes": args.m3_artifact_size_bytes,
        }:
            raise ValueError("G2R closed-snapshot M3 manifest artifact binding drift")
        feature_binding = model_manifest.get("feature_contract")
        if (
            not isinstance(feature_binding, dict)
            or set(feature_binding)
            != {"legacy_path", "legacy_sha256", "path", "sha256"}
            or feature_binding.get("path") != "feature_spec_contract.json"
            or feature_binding.get("sha256") != args.m3_feature_contract_sha256
        ):
            raise ValueError("G2R closed-snapshot M3 feature-contract binding drift")
        feature_contract_path = model_manifest_path.parent / feature_binding["path"]
        feature_contract_ref = _file_ref(
            feature_contract_path, feature_binding["sha256"]
        )
        lineage = {
            **expected_refs,
            "m3_manifest": model_manifest_ref,
            "m3_feature_spec_contract": feature_contract_ref,
        }

        stage = "SOURCE_LOAD"
        failure_class = "SOURCE_LOAD_FAILURE"
        registry = domain.load_verified_scene_registry(
            bundle_root=args.bundle_root,
            score_schema_path=args.score_schema,
            source_manifest_path=args.source_manifest,
            source_manifest_sha256=args.source_manifest_sha256,
            export_receipt_path=args.sanitization_receipt,
            path_mapping_manifest_path=args.wod_path_type_mapping_manifest,
            path_mapping_manifest_sha256=expected_refs[
                "wod_path_type_mapping_manifest"
            ]["sha256"],
            mapping_root=args.wod_path_type_mapping_root,
        )
        routes = _read_route_intents(args.bundle_root, args.score_schema)
        scenes = _build_g2r_scene_inputs(
            bundle_root=args.bundle_root,
            mapping_manifest_path=args.wod_path_type_mapping_manifest,
            score_schema_path=args.score_schema,
            registry=registry,
            route_intents=routes,
            domain=domain,
            orchestrator=orchestrator,
        )

        stage = "ANCHOR_DOMAIN"
        failure_class = "ANCHOR_DOMAIN_FAILURE"
        domain_rows = domain.build_anchor_domain_rows([scene.domain for scene in scenes])
        domain_bytes = domain.encode_anchor_domain_csv(domain_rows)
        domain_manifest = domain.build_manifest(
            artifact_bytes=domain_bytes,
            rows=domain_rows,
            generator_path=args.repo_root / "scripts/rq014/build_wod_scene_anchor_domain.py",
            repo_root=args.repo_root,
            verified_scene_registry=registry,
            python_executable_path=args.python_executable,
            python_executable_sha256=expected_refs["python_executable"]["sha256"],
            environment_manifest_path=args.environment_manifest,
            environment_manifest_sha256=expected_refs["environment_manifest"]["sha256"],
            created_at_utc=args.created_at_utc,
        )

        stage = "NC_GATE"
        failure_class = "NC_GATE_FAILURE"
        nc_receipt = anchors.run_nc_pretstar_history_only_gate(
            repo_root=args.repo_root,
            python_executable_path=args.python_executable,
            python_executable_sha256=expected_refs["python_executable"]["sha256"],
            environment_manifest_path=args.environment_manifest,
            environment_manifest_sha256=expected_refs["environment_manifest"]["sha256"],
            created_at_utc=args.created_at_utc,
        )
        nc_status = str(nc_receipt.get("status"))
        if nc_status != "PASS":
            raise ValueError("NC_PRETSTAR_HISTORY_ONLY did not PASS")

        staging.mkdir(parents=False, exist_ok=False)
        domain_path = staging / "wod_scene_anchor_domain.csv"
        domain_manifest_path = staging / "wod_scene_anchor_domain_manifest.json"
        nc_path = staging / "nc_pretstar_history_only_receipt.json"
        domain_path.write_bytes(domain_bytes)
        domain_manifest_path.write_bytes(canonical_json_bytes(domain_manifest))
        nc_path.write_bytes(canonical_json_bytes(nc_receipt))
        prerequisite = {
            "wod_scene_anchor_domain": _artifact_ref(
                orchestrator,
                domain_path,
                "rq014-wod-scene-anchor-domain-v1",
                len(domain_rows),
            ),
            "wod_scene_anchor_domain_manifest": _artifact_ref(
                orchestrator,
                domain_manifest_path,
                "rq014-wod-scene-anchor-domain-manifest-v1",
                1,
            ),
            "nc_pretstar_history_only_receipt": _artifact_ref(
                orchestrator,
                nc_path,
                "rq014-nc-pretstar-history-only-receipt-v1",
                1,
            ),
        }

        stage = "M3_SCORING"
        failure_class = "M3_SCORING_FAILURE"
        build = orchestrator.build_blind_output_artifacts(
            scenes,
            domain_rows,
            run_id=args.run_id,
            git_commit=args.git_commit,
            created_at_utc=args.created_at_utc,
            lineage=lineage,
            prerequisite_artifacts=prerequisite,
            scorer_path=args.m3_artifact,
            repo_root=args.repo_root,
        )
        stage = "OUTPUT_VALIDATION"
        failure_class = "OUTPUT_INTEGRITY_FAILURE"
        for name, payload in build.artifacts.items():
            path = staging / name
            if path.exists():
                raise ValueError(f"G2R staged artifact collides with prerequisite: {name}")
            path.write_bytes(payload)
        expected_names = {
            *orchestrator.DIRECT_ARTIFACT_NAMES,
            "wod_scene_anchor_domain.csv",
            "wod_scene_anchor_domain_manifest.json",
            "nc_pretstar_history_only_receipt.json",
        }
        if {path.name for path in staging.iterdir()} != expected_names:
            raise ValueError("G2R staged output universe differs from the frozen contract")
        for path in staging.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
                raise ValueError(f"G2R staged output is malformed: {path.name}")
            path.chmod(0o444)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        output_manifest_path = staging / "g2r_output_manifest.json"
        pass_receipt = _g2r_pass_receipt(
            run_id=args.run_id,
            git_commit=args.git_commit,
            created_at_utc=args.created_at_utc,
            output_manifest_path=output_manifest_path,
        )
        canonical_json_bytes(pass_receipt)
        stage = "ATOMIC_PUBLICATION"
        failure_class = "ATOMIC_PUBLICATION_FAILURE"
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staging.rename(final)
        published = True
        _fsync_directory(output_root)
        return pass_receipt
    except Exception as exc:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if published and final.exists() and not final.is_symlink():
            shutil.rmtree(final)
            _fsync_directory(output_root)
        return _runtime_failure_receipt(
            run_id=args.run_id,
            git_commit=args.git_commit,
            created_at_utc=args.created_at_utc,
            stage=stage,
            failure_class=failure_class,
            error=exc,
            nc_gate_status=nc_status,
        )


def _write_managed_outcome(
    *,
    output: Path,
    receipt: dict[str, Any],
    receipt_name: str,
    operation_name: str,
) -> bytes:
    payload = canonical_json_bytes(receipt)
    _write_once(output / receipt_name, payload)
    if receipt.get("status") == "PASS":
        _write_once(
            output / "DONE.json",
            canonical_json_bytes(
                {
                    "schema_version": "rq014-managed-operation-done-v1",
                    "operation": operation_name,
                    "receipt_sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "PASS",
                }
            ),
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    preflight = subparsers.add_parser("contract-preflight")
    preflight.add_argument("--base", type=Path, required=True)
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--execution-contract", type=Path, required=True)
    preflight.add_argument("--expected-exporter-git-commit", required=True)
    preflight.add_argument("--expected-exporter-environment-sha256", required=True)
    pilot = subparsers.add_parser("resource-pilot")
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--lane-v3", type=Path, required=True)
    pilot.add_argument("--bundle-root", type=Path, required=True)
    pilot.add_argument("--wod-path-type-mapping-manifest", type=Path, required=True)
    pilot.add_argument("--contract-preflight-receipt", type=Path, required=True)
    pilot.add_argument("--contract-preflight-done", type=Path, required=True)
    pilot.add_argument("--m3-parity-fixture", type=Path, required=True)
    g2r = subparsers.add_parser("blind-feature-build")
    g2r.add_argument("--run-id", required=True)
    g2r.add_argument("--git-commit", required=True)
    g2r.add_argument("--created-at-utc", required=True)
    g2r.add_argument("--repo-root", type=Path, required=True)
    g2r.add_argument("--bundle-root", type=Path, required=True)
    g2r.add_argument("--score-schema", type=Path, required=True)
    g2r.add_argument("--source-manifest", type=Path, required=True)
    g2r.add_argument("--source-manifest-sha256", required=True)
    g2r.add_argument("--wod-path-type-mapping-manifest", type=Path, required=True)
    g2r.add_argument("--wod-path-type-mapping-root", type=Path, required=True)
    g2r.add_argument("--g2r-output-contract", type=Path, required=True)
    g2r.add_argument("--g2r-output-contract-sha256", required=True)
    g2r.add_argument("--m3-manifest-sha256", required=True)
    g2r.add_argument("--m3-feature-contract-sha256", required=True)
    g2r.add_argument("--contract-preflight-receipt", type=Path, required=True)
    g2r.add_argument("--contract-preflight-done", type=Path, required=True)
    g2r.add_argument("--resource-pilot-receipt", type=Path, required=True)
    g2r.add_argument("--resource-pilot-done", type=Path, required=True)
    g2r.add_argument("--environment-manifest", type=Path, required=True)
    g2r.add_argument("--python-executable", type=Path, required=True)
    g2r.add_argument("--code-snapshot", type=Path, required=True)
    for operation_parser in (preflight, pilot, g2r):
        operation_parser.add_argument("--m3-artifact", type=Path, required=True)
        operation_parser.add_argument("--m3-artifact-size-bytes", type=int, required=True)
        operation_parser.add_argument("--m3-artifact-sha256", required=True)
        operation_parser.add_argument("--input-manifest", type=Path, required=True)
        operation_parser.add_argument("--sanitization-receipt", type=Path, required=True)
        operation_parser.add_argument("--materialization-ledger", type=Path, required=True)
        operation_parser.add_argument("--declassification-export-receipt", type=Path, required=True)
        operation_parser.add_argument("--declassification-export-done", type=Path, required=True)
        operation_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.operation == "contract-preflight":
        receipt = run_preflight(
            base=args.base.resolve(),
            repo_root=args.repo_root.resolve(),
            execution_contract_path=args.execution_contract.resolve(),
            m3_artifact_ref={
                "path": str(args.m3_artifact),
                "size_bytes": args.m3_artifact_size_bytes,
                "sha256": args.m3_artifact_sha256,
            },
            input_manifest_path=args.input_manifest.resolve(),
            sanitization_receipt_path=args.sanitization_receipt.resolve(),
            materialization_ledger_path=args.materialization_ledger.resolve(),
            declassification_export_receipt_path=args.declassification_export_receipt.resolve(),
            declassification_export_done_path=args.declassification_export_done.resolve(),
            expected_exporter_git_commit=args.expected_exporter_git_commit,
            expected_exporter_environment_sha256=args.expected_exporter_environment_sha256,
        )
        receipt_name = "rq014_g2_contract_preflight_receipt.json"
        operation_name = "rq014_g2_contract_preflight"
    elif args.operation == "resource-pilot":
        receipt = run_resource_pilot(
            run_id=args.run_id,
            lane_path=args.lane_v3.resolve(),
            bundle_root=args.bundle_root.resolve(),
            input_manifest_path=args.input_manifest.resolve(),
            sanitization_receipt_path=args.sanitization_receipt.resolve(),
            materialization_ledger_path=args.materialization_ledger.resolve(),
            mapping_manifest_path=args.wod_path_type_mapping_manifest.resolve(),
            m3_artifact_path=args.m3_artifact.resolve(),
            m3_artifact_size_bytes=args.m3_artifact_size_bytes,
            m3_artifact_sha256=args.m3_artifact_sha256,
            m3_parity_fixture_path=args.m3_parity_fixture.resolve(),
            export_receipt_path=args.declassification_export_receipt.resolve(),
            export_done_path=args.declassification_export_done.resolve(),
            preflight_receipt_path=args.contract_preflight_receipt.resolve(),
            preflight_done_path=args.contract_preflight_done.resolve(),
        )
        receipt_name = "rq014_g2_resource_pilot_receipt.json"
        operation_name = "rq014_g2_resource_pilot"
    else:
        receipt = run_g2r_managed(args)
        receipt_name = "rq014_r2_blind_feature_build_receipt.json"
        operation_name = "rq014_r2_blind_feature_build"
    output = args.output_root.resolve()
    _write_managed_outcome(
        output=output,
        receipt=receipt,
        receipt_name=receipt_name,
        operation_name=operation_name,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
