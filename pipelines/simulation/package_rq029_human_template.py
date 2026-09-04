#!/usr/bin/env python3
"""Build a fail-closed RQ029 human-driving template release.

The scientific RQ029 v1/v2 packages remain unchanged.  This adapter rewrites
only template-facing identifiers and time axes, preserves the generated motion
and calibrated analysis values, and records every field that must be replaced
or confirmed before any real-collection import.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data/derived/rq029_human_synthetic_microdata/v2_paper_aligned"
DEFAULT_OUTPUT = REPO_ROOT / "data/derived/rq029_human_template/v1"
SCHEMA_VERSION = "RQ029-human-template-v1"
DATA_ORIGIN = "GENERATED_NOT_OBSERVED"
REQUIRED_MODE = "TEMPLATE_MODE"
SOURCE_RECORD_ID = 1766197775
SOURCE_FIRST_TIMESTAMP_MS = 1766197789851
SESSION_OFFSET_SECONDS = 86_400

REQUIRED_LOGS = (
    "monitor.log",
    "simulation_trajectory.log",
    "vehicle_perception_simulation_trajectory.log",
    "vehicle_trajectory.log",
)

RAW_TABLES = {
    "runs.csv": "runs.csv",
    "scenario_templates.csv": "scenario_templates.csv",
    "ego_warp_metrics.csv": "ego_warp_metrics.csv",
    "trajectory_pairs.parquet": "trajectory_pairs.parquet",
    "designated_counterpart_template.parquet": "designated_counterpart_template.parquet",
}

ANALYSIS_TABLES = {
    "candidate_moments.parquet": "candidate_moments.parquet",
    "both_gate_moments.parquet": "both_gate_moments.parquet",
    "counterpart_windows.parquet": "counterpart_windows.parquet",
    "per_unit_counts.csv": "per_unit_counts.csv",
    "synthetic_run_calibration.csv": "run_calibration.csv",
    "target_summary.json": "target_summary.json",
    "achieved_summary.json": "achieved_summary.json",
}

CASE_TO_SCENARIO = {
    2325: "A1",
    2328: "A2",
    2323: "A3",
    2322: "A4",
    2321: "A5",
    2327: "A6",
    2319: "A7",
    2318: "B1",
    2317: "B2",
    2316: "B3",
    2315: "B4",
    2314: "C1",
    2313: "C2",
    2346: "C3",
    2311: "C4",
}
SCENARIO_TO_CASE = {value: key for key, value in CASE_TO_SCENARIO.items()}

SPECIAL_INVENTORY_FILES = {
    "release_manifest.json",
    "MANIFEST.sha256",
    "00_control/file_inventory.csv",
}

FORBIDDEN_PACKAGE_PATTERN = re.compile(r"synthetic", re.IGNORECASE)
SESSION_PATTERN = re.compile(r"^\d{4}-\d{10}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_placeholder_registry() -> list[dict[str, Any]]:
    """Return deterministic, format-compatible, non-observed session values."""

    rows: list[dict[str, Any]] = []
    for number in range(1, 21):
        driver_id = f"D{number:02d}"
        task_id = str(7900 + number)
        offset_seconds = number * SESSION_OFFSET_SECONDS
        record_id = SOURCE_RECORD_ID + offset_seconds
        session_id = f"{task_id}-{record_id}"
        rows.append(
            {
                "driver_id": driver_id,
                "task_id": task_id,
                "record_id": record_id,
                "session_id": session_id,
                "vehicle_id": str(8000 + number),
                "vehicle_name": f"VEHICLE_{driver_id}",
                "timestamp_offset_ms": offset_seconds * 1000,
                "run_id_prefix": f"onsite:shanghai:{driver_id}",
                "replacement_status": "NEEDS_REAL_INPUT",
            }
        )
    return rows


def _registry_by_driver() -> dict[str, dict[str, Any]]:
    return {row["driver_id"]: row for row in build_placeholder_registry()}


def _required_real_fields() -> list[dict[str, str]]:
    return [
        {
            "field_group": "identity",
            "field": "driver_id",
            "grain": "driver",
            "template_characteristic": "D01-D20 reserved pseudonyms",
            "replacement_scope": "IDENTIFIER_OR_CONFIRM",
            "real_source_needed": "Anonymized participant register",
            "required_action": "Provide real pseudonyms or formally adopt D01-D20",
        },
        {
            "field_group": "identity",
            "field": "taskId",
            "grain": "session",
            "template_characteristic": "Reserved four-digit values 7901-7920",
            "replacement_scope": "IDENTIFIER",
            "real_source_needed": "Collection-platform session ledger",
            "required_action": "Replace with the platform-generated taskId",
        },
        {
            "field_group": "identity",
            "field": "recordId",
            "grain": "session",
            "template_characteristic": "Derived ten-digit values with one-day spacing",
            "replacement_scope": "IDENTIFIER",
            "real_source_needed": "Collection-platform session ledger",
            "required_action": "Replace with the platform-generated recordId",
        },
        {
            "field_group": "identity",
            "field": "session_id",
            "grain": "session",
            "template_characteristic": "Composed from reserved taskId-recordId",
            "replacement_scope": "DERIVED_AFTER_IDENTIFIERS",
            "real_source_needed": "taskId and recordId",
            "required_action": "Regenerate as <taskId>-<recordId>",
        },
        {
            "field_group": "identity",
            "field": "vehicle id and name",
            "grain": "vehicle or session",
            "template_characteristic": "Reserved 8001-8020 and VEHICLE_Dxx values",
            "replacement_scope": "IDENTIFIER",
            "real_source_needed": "Instrumented-vehicle or injection-platform register",
            "required_action": "Replace in both ego log streams",
        },
        {
            "field_group": "identity",
            "field": "run_id and candidate_key",
            "grain": "run or frame",
            "template_characteristic": "Deterministically composed from reserved identifiers",
            "replacement_scope": "DERIVED_AFTER_IDENTIFIERS",
            "real_source_needed": "Final driver scenario and case mapping",
            "required_action": "Rebuild after identity and scenario confirmation",
        },
        {
            "field_group": "time",
            "field": "session and frame timestamps",
            "grain": "session and frame",
            "template_characteristic": "Original replay timeline shifted by whole-day offsets",
            "replacement_scope": "FULL_STREAM",
            "real_source_needed": "Real collection logs",
            "required_action": "Replace every timestamp and globalTimeStamp; do not hand-edit",
        },
        {
            "field_group": "monitoring",
            "field": "monitor.log",
            "grain": "monitor sample",
            "template_characteristic": "Identifiers and time are rewritten; system telemetry is reference-shaped",
            "replacement_scope": "FULL_STREAM",
            "real_source_needed": "Real monitor export",
            "required_action": "Replace the complete file, including CPU memory network and AV monitor values",
        },
        {
            "field_group": "trajectory",
            "field": "vehicle_trajectory.log",
            "grain": "frame",
            "template_characteristic": "Generated ego path timing kinematics and control-shaped fields",
            "replacement_scope": "FULL_STREAM",
            "real_source_needed": "Real ego trajectory export",
            "required_action": "Replace pose speed acceleration steering pedal braking and gear fields",
        },
        {
            "field_group": "trajectory",
            "field": "vehicle_perception_simulation_trajectory.log ego stream",
            "grain": "frame",
            "template_characteristic": "Generated ego motion embedded in a fixed replay background",
            "replacement_scope": "FULL_STREAM",
            "real_source_needed": "Real replay/perception export",
            "required_action": "Replace ego frames and verify their alignment with the background",
        },
        {
            "field_group": "trajectory",
            "field": "simulation_trajectory.log and background actors",
            "grain": "frame and actor",
            "template_characteristic": "Same reference background motion with session time shifts",
            "replacement_scope": "CONFIRM_OR_FULL_STREAM",
            "real_source_needed": "Replay protocol and actual background export",
            "required_action": "Confirm the fixed replay was truly reused; otherwise replace the complete background stream",
        },
        {
            "field_group": "vehicle",
            "field": "vehicleType driveType length width height licence and colours",
            "grain": "vehicle",
            "template_characteristic": "Reference-platform values retained for schema compatibility",
            "replacement_scope": "CONFIRM_OR_REPLACE",
            "real_source_needed": "Actual vehicle/platform configuration",
            "required_action": "Confirm shared vehicle use or replace all static vehicle fields",
        },
        {
            "field_group": "collection",
            "field": "site route device and software version",
            "grain": "batch or session",
            "template_characteristic": "Only Shanghai is confirmed; detailed collection context is absent",
            "replacement_scope": "METADATA",
            "real_source_needed": "Study collection ledger",
            "required_action": "Provide site route version device build and software version",
        },
        {
            "field_group": "mapping",
            "field": "scenario_id caseId caseName mapping",
            "grain": "scenario",
            "template_characteristic": "Current project mapping has 11 of 15 semantic conflicts with raw caseName",
            "replacement_scope": "CONFIRMATION",
            "real_source_needed": "Authoritative human protocol and platform task dictionary",
            "required_action": "Choose the authoritative naming convention before deriving run_id",
        },
        {
            "field_group": "mapping",
            "field": "real 186-case row mapping",
            "grain": "analysis case",
            "template_characteristic": "Generated case groups reproduce aggregate structure only",
            "replacement_scope": "FULL_MAPPING",
            "real_source_needed": "Controlled human row-level archive",
            "required_action": "Provide the real row-to-case mapping if case-clustered analysis is required",
        },
        {
            "field_group": "analysis",
            "field": "per-unit estimator outputs",
            "grain": "driver x scenario and candidate frame",
            "template_characteristic": "Aggregate-constrained values rather than frozen-estimator outputs on observed rows",
            "replacement_scope": "RECOMPUTE",
            "real_source_needed": "Real row-level logs plus frozen estimator",
            "required_action": "Recompute all candidate gate band TTC and counterpart metrics after real-data fill",
        },
    ]


def _prepare_output(output: Path, replace: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not replace:
        raise FileExistsError(f"output exists; pass --replace: {output}")
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("replacement is restricted to the canonical RQ029 template root")
    manifest_path = output / "release_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("refusing replacement: release manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("data_origin") != DATA_ORIGIN
        or manifest.get("required_mode") != REQUIRED_MODE
    ):
        raise ValueError("refusing replacement: template release identity mismatch")
    inventory_path = output / "00_control/file_inventory.csv"
    checksum_path = output / "MANIFEST.sha256"
    if not inventory_path.is_file() or not checksum_path.is_file():
        raise ValueError("refusing replacement: managed inventory or checksum manifest is missing")
    inventory = pd.read_csv(inventory_path)
    if inventory.relative_path.duplicated().any():
        raise ValueError("refusing replacement: managed inventory contains duplicate paths")
    inventory_entries = dict(zip(inventory.relative_path, inventory.sha256))
    manifest_entries, malformed = _manifest_entries(checksum_path)
    if malformed or manifest_entries != inventory_entries:
        raise ValueError("refusing replacement: managed inventory and checksum manifest disagree")
    expected_files = set(inventory_entries) | SPECIAL_INVENTORY_FILES
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if unexpected or missing:
        raise ValueError(
            "refusing replacement: unmanaged or missing files detected "
            f"unexpected={unexpected!r} missing={missing!r}"
        )
    modified = sorted(
        relative
        for relative, digest in inventory_entries.items()
        if _sha256(output / relative) != digest
    )
    if modified:
        raise ValueError(
            "refusing replacement: managed files changed since inventory "
            f"{modified!r}"
        )
    shutil.rmtree(output)
    output.mkdir(parents=True)


def _source_session(driver_id: str) -> Path:
    candidates = sorted((SOURCE_ROOT / "raw/drivers" / driver_id / "sessions").glob("*"))
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise ValueError(f"expected one source session for {driver_id}")
    return candidates[0]


def _shift_numeric_timestamp(value: Any, offset_ms: int) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value + offset_ms
    if isinstance(value, float):
        return value + offset_ms
    if isinstance(value, str):
        try:
            number = Decimal(value)
        except InvalidOperation:
            return value
        shifted = number + Decimal(offset_ms)
        return format(shifted, "f")
    return value


def _rewrite_tree(value: Any, entry: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_rewrite_tree(item, entry) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten: dict[str, Any] = {}
    for key, item in value.items():
        if key == "taskId":
            rewritten[key] = entry["task_id"]
        elif key == "recordId":
            rewritten[key] = int(entry["record_id"])
        elif "timestamp" in key.lower():
            rewritten[key] = _shift_numeric_timestamp(item, int(entry["timestamp_offset_ms"]))
        else:
            rewritten[key] = _rewrite_tree(item, entry)
    if rewritten.get("isPerception") == 0 and "id" in rewritten:
        rewritten["id"] = entry["vehicle_id"]
        if "name" in rewritten:
            rewritten["name"] = entry["vehicle_name"]
    return rewritten


def _read_ego_timeline(path: Path) -> tuple[list[int], list[dict[str, Any]]]:
    timestamps: list[int] = []
    actors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            values = record.get("value", {}).get("value", [])
            if len(values) != 1:
                raise ValueError(f"expected one ego value per line: {path}")
            actor = values[0]
            timestamps.append(int(float(actor["globalTimeStamp"])))
            actors.append(actor)
    return timestamps, actors


def _nearest_actor(
    timestamps: list[int], actors: list[dict[str, Any]], timestamp_ms: int
) -> dict[str, Any]:
    index = bisect.bisect_left(timestamps, timestamp_ms)
    candidates = [max(0, min(len(timestamps) - 1, index))]
    if index:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(timestamps[item] - timestamp_ms))
    return actors[best]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _rewrite_log(
    source: Path,
    target: Path,
    entry: dict[str, Any],
    *,
    monitor_timeline: tuple[list[int], list[dict[str, Any]]] | None = None,
) -> None:
    def rows() -> Iterable[dict[str, Any]]:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if monitor_timeline is not None:
                    av_monitor = record.get("avMonitor")
                    if isinstance(av_monitor, dict) and av_monitor.get("timestamp") is not None:
                        actor = _nearest_actor(
                            monitor_timeline[0],
                            monitor_timeline[1],
                            int(float(av_monitor["timestamp"])),
                        )
                        for source_key, target_key in (
                            ("longitude", "longitude"),
                            ("latitude", "latitude"),
                            ("speed", "speed"),
                        ):
                            if source_key in actor:
                                av_monitor[target_key] = actor[source_key]
                yield _rewrite_tree(record, entry)

    _write_jsonl(target, rows())


def _run_id(driver_id: str, scenario_id: str, case_id: int) -> str:
    return f"onsite:shanghai:{driver_id}:{scenario_id}:native_case:{case_id}"


def _sanitize_text(value: str) -> str:
    value = value.replace("SYNTHETIC_NOT_OBSERVED", DATA_ORIGIN)
    value = value.replace("synthetic_not_observed", DATA_ORIGIN.lower())
    return FORBIDDEN_PACKAGE_PATTERN.sub("generated", value)


def _transform_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    registry = _registry_by_driver()
    out = frame.copy()
    rename = {
        column: _sanitize_text(column)
        for column in out.columns
        if FORBIDDEN_PACKAGE_PATTERN.search(column)
    }
    out = out.rename(columns=rename)
    if "generated_session_id" in out.columns:
        out = out.rename(columns={"generated_session_id": "session_id"})
    if "generated_case_id" in out.columns:
        out = out.rename(columns={"generated_case_id": "case_group_id"})
    out = out.drop(
        columns=[
            column
            for column in ("data_status", "source_team_id", "source_session_id")
            if column in out.columns
        ]
    )
    if "driver_id" in out.columns:
        missing = sorted(set(out.driver_id.astype(str)) - set(registry))
        if missing:
            raise ValueError(f"unregistered drivers: {missing}")
        entries = out.driver_id.astype(str).map(registry)
        offsets = entries.map(lambda item: int(item["timestamp_offset_ms"]))
        for column in out.columns:
            if "timestamp" in column.lower() and pd.api.types.is_numeric_dtype(out[column]):
                out[column] = out[column] + offsets
        out["task_id"] = entries.map(lambda item: item["task_id"])
        out["record_id"] = entries.map(lambda item: item["record_id"])
        out["session_id"] = entries.map(lambda item: item["session_id"])
        out["vehicle_id"] = entries.map(lambda item: item["vehicle_id"])
        out["vehicle_name"] = entries.map(lambda item: item["vehicle_name"])
        if "ego_id" in out.columns:
            out["ego_id"] = out["vehicle_id"]
        if "scenario_id" in out.columns:
            cases = (
                out["native_case_id"].astype(int)
                if "native_case_id" in out.columns
                else out["scenario_id"].map(SCENARIO_TO_CASE).astype(int)
            )
            out["run_id"] = [
                _run_id(str(driver), str(scenario), int(case))
                for driver, scenario, case in zip(out.driver_id, out.scenario_id, cases)
            ]
        if "candidate_key" in out.columns:
            frame_ids = out.candidate_key.astype(str).str.rsplit(":", n=1).str[-1]
            out["candidate_key"] = [
                f"onsite:shanghai:{driver}:{scenario}:frame:{frame_id}"
                for driver, scenario, frame_id in zip(
                    out.driver_id, out.scenario_id, frame_ids
                )
            ]
    else:
        # Scenario-level reference tables have no driver key.  Move their
        # absolute timestamps onto the first reserved template timeline so no
        # T11 collection timestamp is exposed as a human-session value.
        reference_offset = int(build_placeholder_registry()[0]["timestamp_offset_ms"])
        for column in out.columns:
            if "timestamp" in column.lower() and pd.api.types.is_numeric_dtype(out[column]):
                out[column] = out[column] + reference_offset
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].map(
            lambda item: _sanitize_text(item) if isinstance(item, str) else item
        )
    return out


def _transform_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {_sanitize_text(str(key)): _transform_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_transform_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _copy_transformed_table(source_name: str, target: Path) -> None:
    source = SOURCE_ROOT / "tables" / source_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet":
        _transform_dataframe(pd.read_parquet(source)).to_parquet(target, index=False)
    elif source.suffix == ".csv":
        _transform_dataframe(pd.read_csv(source)).to_csv(target, index=False)
    elif source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(
            json.dumps(_transform_json(data), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unsupported table: {source}")


def _write_control_files(output: Path) -> None:
    control = output / "00_control"
    control.mkdir(parents=True, exist_ok=True)
    _write_csv(control / "placeholder_registry.csv", build_placeholder_registry())
    _write_csv(control / "required_real_collection_fields.csv", _required_real_fields())
    contract = {
        "schema_version": SCHEMA_VERSION,
        "required_mode": REQUIRED_MODE,
        "production_import_allowed": False,
        "contains_observed_human_trajectories": False,
        "identity_fill_is_sufficient_for_real_status": False,
        "required_before_real_import": [
            "replace or confirm every row in required_real_collection_fields.csv",
            "replace all full-stream fields with collection-system exports",
            "recompute analysis outputs from the real row-level logs",
            "regenerate file_inventory.csv and MANIFEST.sha256",
            "run the production system's independent authenticity checks",
        ],
    }
    (control / "import_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_root_metadata(output: Path) -> None:
    source_manifest = SOURCE_ROOT / "manifest.json"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "data_origin": DATA_ORIGIN,
        "contains_observed_human_trajectories": False,
        "contains_real_human_identity_or_session_metadata": False,
        "intended_use": REQUIRED_MODE,
        "production_import_allowed": False,
        "collection_area_constraint": "shanghai",
        "reference_release_id": "RQ029_PAPER_ALIGNED_V2",
        "reference_release_manifest_sha256": _sha256(source_manifest),
        "boundary": (
            "The package is a generated parser and analysis template. Replacing identifiers "
            "alone does not convert its trajectories or telemetry into observed records."
        ),
    }
    (output / "DATA_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_origin": DATA_ORIGIN,
        "required_mode": REQUIRED_MODE,
        "production_import_allowed": False,
        "contains_observed_human_trajectories": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layers": {
            "01_collection_raw": "AV-compatible raw-log template and normalized raw tables",
            "02_analysis_support": "Generated analysis-rehearsal tables; not empirical paper evidence",
        },
        "n_drivers": 20,
        "n_scenarios": 15,
        "n_runs": 300,
        "n_frames": 81880,
        "manifest_scope": (
            "All files except release_manifest.json, MANIFEST.sha256, and "
            "00_control/file_inventory.csv"
        ),
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_root_readme(output: Path) -> None:
    directories = sorted(
        path.relative_to(output).as_posix() + "/"
        for path in output.rglob("*")
        if path.is_dir()
    )
    directory_lines = "\n".join(f"- `{path}`" for path in directories)
    text = (
        "# RQ029 human-driving template database\n\n"
        "**本数据库根目录下的原始轨迹、派生表和分析表全部为合成数据，"
        "不包含真实人类逐帧采集记录。**\n\n"
        "**All trajectory and analysis payloads in this database are synthetic and not "
        "observed human records.**\n\n"
        "该包只能在 `TEMPLATE_MODE` 下用于解析器、数据库结构、回放和分析流程联调。"
        "真实采集导入前必须按 `00_control/required_real_collection_fields.csv` 替换或确认"
        "所有字段；只替换 ID、文件夹名或时间戳，不会把生成轨迹变成实测轨迹。\n\n"
        "## 数据层\n\n"
        "- `00_control/`：占位符注册表、补录字段合同、验证结果与文件清单。\n"
        "- `01_collection_raw/`：20 个驾驶人会话的四类原始日志及规范化原始表。\n"
        "- `02_analysis_support/`：分析流程演练表；在真实数据重算前不能作为论文实证材料。\n\n"
        "## 全部子目录\n\n"
        f"{directory_lines}\n"
    )
    (output / "README.md").write_text(text, encoding="utf-8")


def _managed_files(output: Path) -> list[Path]:
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in SPECIAL_INVENTORY_FILES
    )


def _finalize_inventory(output: Path) -> None:
    rows: list[dict[str, Any]] = []
    manifest_lines: list[str] = []
    logical_bytes = 0
    for path in _managed_files(output):
        relative = path.relative_to(output).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        logical_bytes += size
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
                "layer": relative.split("/", 1)[0],
            }
        )
        manifest_lines.append(f"{digest}  {relative}")
    _write_csv(output / "00_control/file_inventory.csv", rows)
    (output / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    manifest_path = output / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "n_inventory_files": len(rows),
            "inventory_logical_bytes": logical_bytes,
            "inventory_sha256": _sha256(output / "00_control/file_inventory.csv"),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def package_release(output: Path = DEFAULT_OUTPUT, *, replace: bool = False) -> dict[str, Any]:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    _prepare_output(output, replace)
    raw_root = output / "01_collection_raw/raw/drivers"
    raw_tables = output / "01_collection_raw/tables"
    analysis_tables = output / "02_analysis_support/tables"
    raw_root.mkdir(parents=True)
    raw_tables.mkdir(parents=True)
    analysis_tables.mkdir(parents=True)
    registry = _registry_by_driver()
    for driver_id, entry in registry.items():
        source_session = _source_session(driver_id)
        target_session = raw_root / driver_id / "sessions" / entry["session_id"]
        target_session.mkdir(parents=True)
        timeline = _read_ego_timeline(source_session / "vehicle_trajectory.log")
        for name in REQUIRED_LOGS:
            _rewrite_log(
                source_session / name,
                target_session / name,
                entry,
                monitor_timeline=timeline if name == "monitor.log" else None,
            )
    for source_name, target_name in RAW_TABLES.items():
        _copy_transformed_table(source_name, raw_tables / target_name)
    for source_name, target_name in ANALYSIS_TABLES.items():
        _copy_transformed_table(source_name, analysis_tables / target_name)
    _write_control_files(output)
    _write_root_metadata(output)
    _write_root_readme(output)
    _finalize_inventory(output)
    return json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))


def _normalized_tree(value: Any) -> Any:
    if isinstance(value, list):
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, dict) and item.get("isPerception") == 0:
                continue
            normalized.append(_normalized_tree(item))
        return normalized
    if not isinstance(value, dict):
        return value
    return {
        key: _normalized_tree(item)
        for key, item in value.items()
        if "timestamp" not in key.lower() and key not in {"taskId", "recordId"}
    }


def _normalized_jsonl_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            normalized = _normalized_tree(json.loads(line))
            digest.update(
                json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _scan_forbidden_payload(output: Path) -> list[str]:
    hits: list[str] = []
    for path in output.rglob("*"):
        if not path.is_file() or path == output / "README.md":
            continue
        relative = path.relative_to(output).as_posix()
        if FORBIDDEN_PACKAGE_PATTERN.search(relative):
            hits.append(f"path:{relative}")
            continue
        with path.open("rb") as handle:
            tail = b""
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                payload = (tail + chunk).lower()
                if b"synthetic" in payload:
                    hits.append(f"content:{relative}")
                    break
                tail = payload[-16:]
    return hits


def _manifest_entries(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    malformed: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            malformed.append(line)
            continue
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    return entries, malformed


def _write_validation(output: Path, result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    control = output / "00_control"
    (control / "validation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(checks).to_csv(control / "validation_checks.csv", index=False)
    (control / "VALIDATION_REPORT.md").write_text(
        "# RQ029 template release validation\n\n"
        f"- Status: **{result['validation_status']}**\n"
        f"- Checks: {result['checks_passed']}/{result['checks_total']} passed\n"
        f"- Sessions / runs / frames: {result['n_sessions']} / {result['n_runs']} / {result['n_frames']}\n"
        f"- Mode: `{result['mode']}`\n"
        f"- Deep verification: `{result['deep_hash']}`\n\n"
        "Production import remains disabled until all required real fields and full-stream records "
        "have been replaced or confirmed and analysis outputs have been recomputed.\n",
        encoding="utf-8",
    )


def validate_release(
    output: Path = DEFAULT_OUTPUT,
    *,
    mode: str = "production",
    deep_hash: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    if mode != "template":
        return {
            "validation_status": "TEMPLATE_MODE_REQUIRED",
            "checks_passed": 0,
            "checks_total": 1,
            "failed_checks": ["template_mode_gate"],
            "mode": mode,
            "deep_hash": deep_hash,
            "n_sessions": 0,
            "n_runs": 0,
            "n_frames": 0,
        }
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "DATA_PROVENANCE.json").read_text(encoding="utf-8"))
    check("schema", manifest.get("schema_version") == SCHEMA_VERSION, manifest.get("schema_version"), SCHEMA_VERSION)
    check("data_origin", manifest.get("data_origin") == DATA_ORIGIN, manifest.get("data_origin"), DATA_ORIGIN)
    check("required_mode", manifest.get("required_mode") == REQUIRED_MODE, manifest.get("required_mode"), REQUIRED_MODE)
    check(
        "production_disabled",
        manifest.get("production_import_allowed") is False
        and provenance.get("production_import_allowed") is False,
        [manifest.get("production_import_allowed"), provenance.get("production_import_allowed")],
        [False, False],
    )
    check(
        "not_observed_boundary",
        provenance.get("contains_observed_human_trajectories") is False,
        provenance.get("contains_observed_human_trajectories"),
        False,
    )
    readme = (output / "README.md").read_text(encoding="utf-8")
    disclosure_ok = "全部为合成数据" in readme and "synthetic" in readme.lower()
    check(
        "root_readme_disclosure",
        disclosure_ok,
        {
            "chinese_disclosure": "全部为合成数据" in readme,
            "english_disclosure": "synthetic" in readme.lower(),
        },
        {"chinese_disclosure": True, "english_disclosure": True},
    )
    forbidden_hits = _scan_forbidden_payload(output)
    check("readme_only_forbidden_word", not forbidden_hits, forbidden_hits, [])
    prohibited_files = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and (
            path.name == ".DS_Store"
            or path.suffix.lower() == ".zip"
            or "__MACOSX" in path.parts
            or "NOTICE" in path.name.upper()
        )
    ]
    check("no_prohibited_files", not prohibited_files, prohibited_files, [])

    registry = pd.read_csv(output / "00_control/placeholder_registry.csv", dtype=str)
    check("placeholder_count", len(registry) == 20, len(registry), 20)
    for field in ("task_id", "record_id", "session_id", "vehicle_id", "vehicle_name"):
        check(f"unique_{field}", registry[field].nunique() == 20, int(registry[field].nunique()), 20)
    check(
        "session_format",
        registry.session_id.map(lambda value: bool(SESSION_PATTERN.fullmatch(value))).all(),
        bool(registry.session_id.map(lambda value: bool(SESSION_PATTERN.fullmatch(value))).all()),
        True,
    )

    sessions = sorted((output / "01_collection_raw/raw/drivers").glob("*/sessions/*"))
    check("session_count", len(sessions) == 20, len(sessions), 20)
    observed_tasks: set[str] = set()
    observed_records: set[str] = set()
    observed_vehicle_ids: set[str] = set()
    log_hashes: dict[str, set[str]] = {name: set() for name in REQUIRED_LOGS}
    for session in sessions:
        files = sorted(path.name for path in session.iterdir() if path.is_file())
        driver_name = session.parent.parent.name
        check(f"session_files_{driver_name}", files == sorted(REQUIRED_LOGS), files, sorted(REQUIRED_LOGS))
        with (session / "monitor.log").open(encoding="utf-8") as handle:
            monitor = json.loads(handle.readline())
        observed_tasks.add(str(monitor.get("taskId")))
        observed_records.add(str(monitor.get("recordId")))
        session_task, session_record = session.name.split("-", 1)
        check(
            f"session_identity_{driver_name}",
            str(monitor.get("taskId")) == session_task
            and str(monitor.get("recordId")) == session_record,
            [monitor.get("taskId"), monitor.get("recordId")],
            [session_task, session_record],
        )
        with (session / "vehicle_trajectory.log").open(encoding="utf-8") as handle:
            vehicle = json.loads(handle.readline())
        actor = vehicle["value"]["value"][0]
        observed_vehicle_ids.add(str(actor["id"]))
        for name in REQUIRED_LOGS:
            log_hashes[name].add(_sha256(session / name))
    check("unique_tasks_in_logs", len(observed_tasks) == 20, len(observed_tasks), 20)
    check("unique_records_in_logs", len(observed_records) == 20, len(observed_records), 20)
    check("unique_vehicle_ids_in_logs", len(observed_vehicle_ids) == 20, len(observed_vehicle_ids), 20)
    for name, values in log_hashes.items():
        check(f"unique_hash_{name}", len(values) == 20, len(values), 20)

    runs = pd.read_csv(output / "01_collection_raw/tables/runs.csv")
    trajectory = pd.read_parquet(output / "01_collection_raw/tables/trajectory_pairs.parquet")
    check("run_rows", len(runs) == 300 and runs.run_id.nunique() == 300, [len(runs), int(runs.run_id.nunique())], [300, 300])
    check("trajectory_rows", len(trajectory) == 81880, len(trajectory), 81880)
    check("raw_identity_columns_clean", "data_status" not in runs.columns and "data_status" not in trajectory.columns, list(runs.columns), "no data_status")

    if deep_hash:
        perception_digests = {
            _normalized_jsonl_digest(session / "vehicle_perception_simulation_trajectory.log")
            for session in sessions
        }
        source_digest = _normalized_jsonl_digest(
            _source_session("D01") / "vehicle_perception_simulation_trajectory.log"
        )
        check("background_motion_same_across_sessions", len(perception_digests) == 1, len(perception_digests), 1)
        check("background_motion_matches_reference", perception_digests == {source_digest}, list(perception_digests), [source_digest])
        simulation_digests = {
            _normalized_jsonl_digest(session / "simulation_trajectory.log") for session in sessions
        }
        source_simulation = _normalized_jsonl_digest(_source_session("D01") / "simulation_trajectory.log")
        check("simulation_motion_same_across_sessions", simulation_digests == {source_simulation}, list(simulation_digests), [source_simulation])
        source_trajectory = pd.read_parquet(SOURCE_ROOT / "tables/trajectory_pairs.parquet")
        motion_columns = [
            "driver_id",
            "scenario_id",
            "scenario_frame_index",
            "ego_latitude",
            "ego_longitude",
            "ego_speed_kmh",
            "ego_course_deg",
            "ego_vx_mps",
            "ego_vy_mps",
            "counterpart_latitude",
            "counterpart_longitude",
            "counterpart_speed_kmh",
        ]
        motion_equal = source_trajectory[motion_columns].equals(trajectory[motion_columns])
        check("trajectory_motion_unchanged", motion_equal, motion_equal, True)

    inventory = pd.read_csv(output / "00_control/file_inventory.csv")
    inventory_entries = dict(zip(inventory.relative_path, inventory.sha256))
    manifest_entries, malformed = _manifest_entries(output / "MANIFEST.sha256")
    actual_paths = {path.relative_to(output).as_posix() for path in _managed_files(output)}
    check("inventory_unique", not inventory.relative_path.duplicated().any(), int(inventory.relative_path.duplicated().sum()), 0)
    check("inventory_matches_actual", set(inventory_entries) == actual_paths, sorted(actual_paths - set(inventory_entries)), [])
    check("manifest_matches_inventory", not malformed and manifest_entries == inventory_entries, {"malformed": malformed, "n": len(manifest_entries)}, {"malformed": [], "n": len(inventory_entries)})
    if deep_hash:
        mismatches = [
            relative
            for relative, digest in inventory_entries.items()
            if _sha256(output / relative) != digest
        ]
        check("inventory_hashes", not mismatches, mismatches, [])

    failed = [row for row in checks if row["status"] == "FAIL"]
    result = {
        "validation_status": "PASS" if not failed else "FAIL",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": [row["check"] for row in failed],
        "mode": mode,
        "deep_hash": deep_hash,
        "n_sessions": len(sessions),
        "n_runs": int(runs.run_id.nunique()),
        "n_frames": len(trajectory),
        "n_inventory_files": len(inventory),
    }
    if write_report:
        _write_validation(output, result, checks)
    return result


def build_and_validate(
    output: Path = DEFAULT_OUTPUT,
    *,
    replace: bool = False,
    deep_hash: bool = True,
) -> dict[str, Any]:
    package_release(output, replace=replace)
    for _ in range(2):
        validate_release(output, mode="template", deep_hash=deep_hash, write_report=True)
        _finalize_inventory(output)
    return validate_release(output, mode="template", deep_hash=deep_hash, write_report=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mode", choices=("template", "production"), default="production")
    parser.add_argument("--skip-deep-hash", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        package_release(args.output_dir, replace=args.replace)
        for _ in range(2):
            validate_release(
                args.output_dir,
                mode=args.mode,
                deep_hash=not args.skip_deep_hash,
                write_report=True,
            )
            _finalize_inventory(args.output_dir)
    result = validate_release(
        args.output_dir,
        mode=args.mode,
        deep_hash=not args.skip_deep_hash,
        write_report=False,
    )
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["validation_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
