#!/usr/bin/env python3
"""Build the immutable RQ014 WOD scene-anchor domain and manifest.

The input is a rating-blind, already-resampled position-tick description.  This
W2 kernel performs the frozen 16-feature-family/H20/HFEAS membership logic and
canonical artifact encoding only; source-bundle installation and managed-job
integration remain later waves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.rq014.build_wod_m3_anchors import canonical_json_bytes, temporal_window_bounds
from scripts.rq014.wod_ipv_preprocessing import state_sequence_from_window_xy


DOMAIN_COLUMNS = (
    "segment_id",
    "feature_id",
    "horizon_id",
    "path_type_or_NA",
    "h_common_tick_or_NA",
    "tau_tick_or_NA",
    "membership_status",
    "reason_code",
)
SAMPLING_AXIS = (("R04N", 4), ("R10L", 10))
TEMPORAL_AXIS = (
    "CH-W10",
    "CH-W25",
    "LF-W10",
    "LF-W25",
    "HF-W10",
    "HF-W25",
    "TP",
    "TF",
)
FEATURE_FAMILIES = tuple(
    f"F-{sampling_id}-{temporal_id}"
    for sampling_id, _ in SAMPLING_AXIS
    for temporal_id in TEMPORAL_AXIS
)
HORIZON_AXIS = ("H20", "HFEAS")
PATH_TYPES = frozenset({"CP", "HO", "MP", "F"})
STATUS_TO_REASON = {
    "AVAILABLE": "F_AVAILABLE_CONTINUE",
    "INELIGIBLE_BLIND": "F_SAFE_PRIMITIVE_OR_FIXTURE_GATE",
    "MISSING_WOD_PATH_TYPE": "F_MISSING_WOD_PATH_TYPE",
    "AMBIGUOUS_WOD_PATH_TYPE": "F_AMBIGUOUS_WOD_PATH_TYPE",
    "UNRECOGNIZED_PATH_TYPE": "F_UNRECOGNIZED_PATH_TYPE",
    "INELIGIBLE_TIMELINE_SUPPORT": "F_TIMELINE_SUPPORT",
    "INELIGIBLE_TIMELINE_SOURCE_GAP": "F_TIMELINE_SOURCE_GAP",
    "INELIGIBLE_TIMELINE_GRID_PHASE": "F_TIMELINE_GRID_PHASE",
    "INELIGIBLE_TIMELINE_SEAM": "F_TIMELINE_SEAM",
    "INELIGIBLE_STATE_NONFINITE": "F_STATE_NONFINITE",
    "INELIGIBLE_UNDEFINED_HEADING": "F_UNDEFINED_HEADING",
}
STATUS_PRIORITY = {status: index for index, status in enumerate(STATUS_TO_REASON)}


class AnchorDomainError(ValueError):
    """Fail-closed anchor-domain construction or integrity error."""


@dataclass(frozen=True)
class SamplingTimelines:
    """Four WOD position branches on one exact integer-tick sampling grid."""

    candidates: Mapping[str, Mapping[int, tuple[float, float]]]
    counterpart: Mapping[int, tuple[float, float]]
    terminal_status: str | None = None


@dataclass(frozen=True)
class SceneDomainInput:
    segment_id: str
    path_type_or_NA: str
    sampling: Mapping[str, SamplingTimelines]
    terminal_status: str | None = None


def _terminal_row(
    *,
    segment_id: str,
    feature_id: str,
    horizon_id: str,
    path_type_or_NA: str,
    h_common_tick_or_NA: str,
    status: str,
) -> dict[str, str]:
    if status == "AVAILABLE" or status not in STATUS_TO_REASON:
        raise AnchorDomainError(f"invalid terminal membership status: {status}")
    return {
        "segment_id": segment_id,
        "feature_id": feature_id,
        "horizon_id": horizon_id,
        "path_type_or_NA": path_type_or_NA,
        "h_common_tick_or_NA": h_common_tick_or_NA,
        "tau_tick_or_NA": "NA",
        "membership_status": status,
        "reason_code": STATUS_TO_REASON[status],
    }


def _available_row(
    *,
    segment_id: str,
    feature_id: str,
    horizon_id: str,
    path_type: str,
    h_common_tick_or_NA: str,
    tau_tick: int,
) -> dict[str, str]:
    return {
        "segment_id": segment_id,
        "feature_id": feature_id,
        "horizon_id": horizon_id,
        "path_type_or_NA": path_type,
        "h_common_tick_or_NA": h_common_tick_or_NA,
        "tau_tick_or_NA": str(tau_tick),
        "membership_status": "AVAILABLE",
        "reason_code": "F_AVAILABLE_CONTINUE",
    }


def _window_status(
    timelines: SamplingTimelines,
    *,
    temporal_id: str,
    tau_tick: int,
    rate_hz: int,
    h_common_tick: int,
) -> str:
    lower, upper = temporal_window_bounds(temporal_id, tau_tick, rate_hz, h_common_tick)
    required = tuple(range(lower, upper + 1))
    branches = [timelines.candidates[candidate_id] for candidate_id in ("C1", "C2", "C3")]
    branches.append(timelines.counterpart)
    if any(any(tick not in branch for tick in required) for branch in branches):
        return "INELIGIBLE_TIMELINE_SUPPORT"
    try:
        for branch in branches:
            xy = np.asarray([branch[tick] for tick in required], dtype=float)
            if not np.all(np.isfinite(xy)):
                return "INELIGIBLE_STATE_NONFINITE"
            state_sequence_from_window_xy(xy, 1.0 / rate_hz)
    except ValueError as exc:
        if "all-stationary" in str(exc):
            return "INELIGIBLE_UNDEFINED_HEADING"
        return "INELIGIBLE_STATE_NONFINITE"
    return "AVAILABLE"


def _resolved_path_status(scene: SceneDomainInput) -> str | None:
    if scene.terminal_status is not None:
        return scene.terminal_status
    if scene.path_type_or_NA == "NA":
        return "MISSING_WOD_PATH_TYPE"
    if scene.path_type_or_NA not in PATH_TYPES:
        return "UNRECOGNIZED_PATH_TYPE"
    return None


def build_scene_anchor_rows(scene: SceneDomainInput) -> list[dict[str, str]]:
    """Build all 32 group results for one scene in frozen stored order."""

    if not scene.segment_id:
        raise AnchorDomainError("segment_id must be nonempty")
    scene_terminal = _resolved_path_status(scene)
    rows: list[dict[str, str]] = []
    for sampling_id, rate_hz in SAMPLING_AXIS:
        timelines = scene.sampling.get(sampling_id)
        sampling_terminal = scene_terminal or (
            timelines.terminal_status if timelines is not None else "INELIGIBLE_TIMELINE_SUPPORT"
        )
        h_common_tick: int | None = None
        if timelines is not None and sampling_terminal is None:
            if set(timelines.candidates) != {"C1", "C2", "C3"}:
                sampling_terminal = "INELIGIBLE_TIMELINE_SUPPORT"
            else:
                branches = [*timelines.candidates.values(), timelines.counterpart]
                if any(not branch for branch in branches):
                    sampling_terminal = "INELIGIBLE_TIMELINE_SUPPORT"
                else:
                    h_common_tick = min(max(branch) for branch in branches)
                    if h_common_tick < 0:
                        h_common_tick = None
                        sampling_terminal = "INELIGIBLE_TIMELINE_SUPPORT"
        for temporal_id in TEMPORAL_AXIS:
            feature_id = f"F-{sampling_id}-{temporal_id}"
            h_common_field = (
                str(h_common_tick) if temporal_id == "TF" and h_common_tick is not None else "NA"
            )
            if sampling_terminal is not None or timelines is None or h_common_tick is None:
                status = sampling_terminal or "INELIGIBLE_TIMELINE_SUPPORT"
                for horizon_id in HORIZON_AXIS:
                    rows.append(
                        _terminal_row(
                            segment_id=scene.segment_id,
                            feature_id=feature_id,
                            horizon_id=horizon_id,
                            path_type_or_NA=scene.path_type_or_NA,
                            h_common_tick_or_NA=h_common_field,
                            status=status,
                        )
                    )
                continue

            status_by_tick = {
                tick: _window_status(
                    timelines,
                    temporal_id=temporal_id,
                    tau_tick=tick,
                    rate_hz=rate_hz,
                    h_common_tick=h_common_tick,
                )
                for tick in range(rate_hz, h_common_tick + 1)
            }
            h20_ticks = tuple(range(rate_hz, 2 * rate_hz + 1))
            h20_complete = (
                2 * rate_hz <= h_common_tick
                and all(status_by_tick.get(tick) == "AVAILABLE" for tick in h20_ticks)
            )
            if h20_complete:
                rows.extend(
                    _available_row(
                        segment_id=scene.segment_id,
                        feature_id=feature_id,
                        horizon_id="H20",
                        path_type=scene.path_type_or_NA,
                        h_common_tick_or_NA=h_common_field,
                        tau_tick=tick,
                    )
                    for tick in h20_ticks
                )
                hfeas_ticks = tuple(
                    tick
                    for tick in range(rate_hz, h_common_tick + 1)
                    if status_by_tick[tick] == "AVAILABLE"
                )
                if not hfeas_ticks:
                    raise AnchorDomainError("complete H20 unexpectedly yielded empty HFEAS")
                rows.extend(
                    _available_row(
                        segment_id=scene.segment_id,
                        feature_id=feature_id,
                        horizon_id="HFEAS",
                        path_type=scene.path_type_or_NA,
                        h_common_tick_or_NA=h_common_field,
                        tau_tick=tick,
                    )
                    for tick in hfeas_ticks
                )
            else:
                failure_statuses = [
                    status_by_tick.get(tick, "INELIGIBLE_TIMELINE_SUPPORT") for tick in h20_ticks
                    if status_by_tick.get(tick, "INELIGIBLE_TIMELINE_SUPPORT") != "AVAILABLE"
                ]
                if not failure_statuses:
                    failure_statuses = ["INELIGIBLE_TIMELINE_SUPPORT"]
                failure_status = min(failure_statuses, key=lambda status: STATUS_PRIORITY[status])
                for horizon_id in HORIZON_AXIS:
                    rows.append(
                        _terminal_row(
                            segment_id=scene.segment_id,
                            feature_id=feature_id,
                            horizon_id=horizon_id,
                            path_type_or_NA=scene.path_type_or_NA,
                            h_common_tick_or_NA=h_common_field,
                            status=failure_status,
                        )
                    )
    return rows


def encode_anchor_domain_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    """Encode exact RFC4180/CPython csv.writer bytes after integrity checks."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=DOMAIN_COLUMNS,
        dialect="excel",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(DOMAIN_COLUMNS):
            raise AnchorDomainError("anchor-domain row exact-key mismatch")
        writer.writerow(row)
    data = stream.getvalue().encode("utf-8")
    if b"\r" in data or not data.endswith(b"\n"):
        raise AnchorDomainError("anchor-domain encoding drift")
    return data


def build_anchor_domain_rows(scenes: Sequence[SceneDomainInput]) -> list[dict[str, str]]:
    """Build rows for raw-UTF8 sorted scenes and validate every group."""

    segment_ids = [scene.segment_id for scene in scenes]
    if len(segment_ids) != len(set(segment_ids)):
        raise AnchorDomainError("duplicate segment_id")
    ordered = sorted(scenes, key=lambda scene: scene.segment_id.encode("utf-8"))
    rows = [row for scene in ordered for row in build_scene_anchor_rows(scene)]
    validate_anchor_domain_rows(rows, expected_scene_count=len(scenes))
    return rows


def validate_anchor_domain_rows(
    rows: Sequence[Mapping[str, str]], *, expected_scene_count: int = 479
) -> None:
    """Validate group count, order, and AVAILABLE/terminal exclusivity."""

    feature_order = {feature_id: index for index, feature_id in enumerate(FEATURE_FAMILIES)}
    horizon_order = {horizon_id: index for index, horizon_id in enumerate(HORIZON_AXIS)}
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        if set(row) != set(DOMAIN_COLUMNS):
            raise AnchorDomainError("anchor-domain row exact-key mismatch")
        key = (row["segment_id"], row["feature_id"], row["horizon_id"])
        groups.setdefault(key, []).append(row)
    expected_groups = expected_scene_count * len(FEATURE_FAMILIES) * len(HORIZON_AXIS)
    if len(groups) != expected_groups:
        raise AnchorDomainError(f"anchor-domain group count {len(groups)} != {expected_groups}")
    expected_order = sorted(
        groups,
        key=lambda key: (
            key[0].encode("utf-8"), feature_order[key[1]], horizon_order[key[2]]
        ),
    )
    if list(groups) != expected_order:
        raise AnchorDomainError("anchor-domain group order drift")
    available_paths_by_scene: dict[str, set[str]] = {}
    ticks_by_group: dict[tuple[str, str, str], tuple[int, ...]] = {}
    for key, group in groups.items():
        temporal_id = key[1].split("-", 2)[2]
        sampling_id = key[1].split("-")[1]
        rate_hz = dict(SAMPLING_AXIS)[sampling_id]
        statuses = {row["membership_status"] for row in group}
        if statuses == {"AVAILABLE"}:
            ticks = [int(row["tau_tick_or_NA"]) for row in group]
            if ticks != sorted(set(ticks)) or any(
                row["reason_code"] != "F_AVAILABLE_CONTINUE" for row in group
            ):
                raise AnchorDomainError(f"invalid AVAILABLE group: {key}")
            if len({row["path_type_or_NA"] for row in group}) != 1:
                raise AnchorDomainError(f"path-type drift in AVAILABLE group: {key}")
            path_type = group[0]["path_type_or_NA"]
            if path_type not in PATH_TYPES:
                raise AnchorDomainError(f"invalid AVAILABLE path type: {key}")
            available_paths_by_scene.setdefault(key[0], set()).add(path_type)
            if key[2] == "H20" and ticks != list(range(rate_hz, 2 * rate_hz + 1)):
                raise AnchorDomainError(f"H20 membership drift: {key}")
            if key[2] == "HFEAS" and any(tick < rate_hz for tick in ticks):
                raise AnchorDomainError(f"HFEAS lower-bound drift: {key}")
            ticks_by_group[key] = tuple(ticks)
            h_values = {row["h_common_tick_or_NA"] for row in group}
            if (temporal_id == "TF" and (len(h_values) != 1 or "NA" in h_values)) or (
                temporal_id != "TF" and h_values != {"NA"}
            ):
                raise AnchorDomainError(f"h_common binding drift: {key}")
            if temporal_id == "TF":
                h_common_tick = int(next(iter(h_values)))
                if h_common_tick < 0 or ticks[-1] > h_common_tick:
                    raise AnchorDomainError(f"TF h_common domain drift: {key}")
        elif len(group) != 1 or "AVAILABLE" in statuses or group[0]["tau_tick_or_NA"] != "NA":
            raise AnchorDomainError(f"mixed or malformed terminal group: {key}")
        elif (
            group[0]["membership_status"] not in STATUS_TO_REASON
            or group[0]["reason_code"]
            != STATUS_TO_REASON[group[0]["membership_status"]]
        ):
            raise AnchorDomainError(f"terminal reason-code drift: {key}")
        else:
            h_common = group[0]["h_common_tick_or_NA"]
            if temporal_id != "TF" and h_common != "NA":
                raise AnchorDomainError(f"non-TF terminal h_common drift: {key}")
            if h_common != "NA" and int(h_common) < 0:
                raise AnchorDomainError(f"negative terminal h_common: {key}")
    if any(len(path_types) != 1 for path_types in available_paths_by_scene.values()):
        raise AnchorDomainError("cross-group scene path-type drift")
    for segment_id in {key[0] for key in groups}:
        for feature_id in FEATURE_FAMILIES:
            h20_key = (segment_id, feature_id, "H20")
            hfeas_key = (segment_id, feature_id, "HFEAS")
            if h20_key in ticks_by_group:
                h20_ticks = set(ticks_by_group[h20_key])
                if hfeas_key not in ticks_by_group or not h20_ticks.issubset(
                    ticks_by_group[hfeas_key]
                ):
                    raise AnchorDomainError(f"HFEAS does not retain complete H20: {hfeas_key}")
            elif groups[h20_key][0]["membership_status"] != groups[hfeas_key][0][
                "membership_status"
            ]:
                raise AnchorDomainError(f"terminal horizon status drift: {hfeas_key}")


def build_manifest(
    *,
    artifact_bytes: bytes,
    rows: Sequence[Mapping[str, str]],
    generator_path: Path,
    recovery_contract_sha256: str,
    envelope_contract_sha256: str,
    source_manifest_sha256: str,
    path_mapping_sha256: str,
    python_executable_sha256: str,
    environment_manifest_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the exact 18-key canonical anchor-domain manifest payload."""

    groups = {(row["segment_id"], row["feature_id"], row["horizon_id"]) for row in rows}
    available_groups = {
        (row["segment_id"], row["feature_id"], row["horizon_id"])
        for row in rows
        if row["membership_status"] == "AVAILABLE"
    }
    terminal_groups = groups - available_groups
    generator_bytes = generator_path.read_bytes()
    return {
        "schema_version": "rq014-wod-scene-anchor-domain-manifest-v1",
        "artifact_relative_path": "wod_scene_anchor_domain.csv",
        "artifact_size_bytes": len(artifact_bytes),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "row_count": len(rows),
        "group_count": len(groups),
        "available_group_count": len(available_groups),
        "terminal_group_count": len(terminal_groups),
        "generator_path": "scripts/rq014/build_wod_scene_anchor_domain.py",
        "generator_size_bytes": len(generator_bytes),
        "generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
        "recovery_contract_sha256": recovery_contract_sha256,
        "envelope_contract_sha256": envelope_contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "path_mapping_sha256": path_mapping_sha256,
        "python_executable_sha256": python_executable_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "created_at_utc": created_at_utc,
    }


def _parse_tick_positions(value: Mapping[str, Sequence[float]]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for tick_text, xy in value.items():
        tick = int(tick_text)
        if str(tick) != tick_text or len(xy) != 2:
            raise AnchorDomainError("tick positions require canonical integer keys and XY pairs")
        result[tick] = (float(xy[0]), float(xy[1]))
    return result


def _parse_scene(value: Mapping[str, Any]) -> SceneDomainInput:
    sampling: dict[str, SamplingTimelines] = {}
    for sampling_id, payload in value.get("sampling", {}).items():
        sampling[sampling_id] = SamplingTimelines(
            candidates={
                candidate_id: _parse_tick_positions(rows)
                for candidate_id, rows in payload["candidates"].items()
            },
            counterpart=_parse_tick_positions(payload["counterpart"]),
            terminal_status=payload.get("terminal_status"),
        )
    return SceneDomainInput(
        segment_id=str(value["segment_id"]),
        path_type_or_NA=str(value["path_type_or_NA"]),
        sampling=sampling,
        terminal_status=value.get("terminal_status"),
    )


def _strict_load(path: Path) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AnchorDomainError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> Any:
        raise AnchorDomainError(f"nonfinite JSON token: {token}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_nonfinite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    request = _strict_load(args.input_json)
    scenes = [_parse_scene(item) for item in request["scenes"]]
    if len(scenes) != 479:
        raise AnchorDomainError("production anchor domain requires exactly 479 scenes")
    rows = build_anchor_domain_rows(scenes)
    artifact = encode_anchor_domain_csv(rows)
    manifest = build_manifest(
        artifact_bytes=artifact,
        rows=rows,
        generator_path=Path(__file__),
        recovery_contract_sha256=request["recovery_contract_sha256"],
        envelope_contract_sha256=request["envelope_contract_sha256"],
        source_manifest_sha256=request["source_manifest_sha256"],
        path_mapping_sha256=request["path_mapping_sha256"],
        python_executable_sha256=request["python_executable_sha256"],
        environment_manifest_sha256=request["environment_manifest_sha256"],
        created_at_utc=request["created_at_utc"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wod_scene_anchor_domain.csv").write_bytes(artifact)
    (args.output_dir / "wod_scene_anchor_domain_manifest.json").write_bytes(
        canonical_json_bytes(manifest)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
