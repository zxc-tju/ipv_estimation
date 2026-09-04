from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import pipelines.simulation.package_rq029_human_template as template_module

from pipelines.simulation.package_rq029_human_template import (
    DATA_ORIGIN,
    DEFAULT_OUTPUT,
    REQUIRED_MODE,
    SCHEMA_VERSION,
    _prepare_output,
    _finalize_inventory,
    _required_real_fields,
    _rewrite_tree,
    _transform_dataframe,
    build_placeholder_registry,
    validate_release,
)


def test_placeholder_registry_is_unique_and_real_format_compatible() -> None:
    rows = build_placeholder_registry()
    assert len(rows) == 20
    for field in ("driver_id", "task_id", "record_id", "session_id", "vehicle_id", "vehicle_name"):
        assert len({str(row[field]) for row in rows}) == 20
    assert all(re.fullmatch(r"\d{4}-\d{10}", row["session_id"]) for row in rows)
    assert all("synthetic" not in json.dumps(row).lower() for row in rows)


def test_required_real_fields_include_identifiers_and_full_streams() -> None:
    rows = _required_real_fields()
    fields = {row["field"] for row in rows}
    assert {"driver_id", "taskId", "recordId", "session_id"} <= fields
    assert {"monitor.log", "vehicle_trajectory.log"} <= fields
    assert any(row["replacement_scope"] == "FULL_STREAM" for row in rows)
    assert any(row["replacement_scope"] == "RECOMPUTE" for row in rows)


def test_rewrite_tree_shifts_time_and_replaces_identity() -> None:
    entry = build_placeholder_registry()[0]
    source = {
        "taskId": "6923",
        "recordId": 1766197775,
        "timestamp": "1766197789851",
        "value": [
            {
                "id": "2490",
                "name": "SYNTHETIC_HUMAN_D01",
                "isPerception": 0,
                "globalTimeStamp": "1766197789851",
            },
            {
                "id": "1200002",
                "name": "background",
                "isPerception": 1,
                "globalTimeStamp": "1766197789851",
            },
        ],
    }
    result = _rewrite_tree(source, entry)
    assert result["taskId"] == entry["task_id"]
    assert result["recordId"] == entry["record_id"]
    assert int(result["timestamp"]) - int(source["timestamp"]) == entry["timestamp_offset_ms"]
    assert result["value"][0]["id"] == entry["vehicle_id"]
    assert result["value"][0]["name"] == entry["vehicle_name"]
    assert result["value"][1]["id"] == "1200002"
    assert result["value"][1]["name"] == "background"


def test_dataframe_transform_removes_old_markers_and_rebuilds_keys() -> None:
    source = pd.DataFrame(
        [
            {
                "driver_id": "D01",
                "scenario_id": "A1",
                "native_case_id": 2325,
                "run_id": "human_synthetic:D01:A1",
                "candidate_key": "synthetic:shanghai:D01:A1:frame:7",
                "synthetic_session_id": "synthetic-D01-source",
                "synthetic_case_id": "SC001",
                "timestamp_ms": 1766197789851,
                "ego_id": "2490",
                "data_status": "SYNTHETIC_NOT_OBSERVED",
            }
        ]
    )
    result = _transform_dataframe(source)
    assert "data_status" not in result.columns
    assert "session_id" in result.columns
    assert "case_group_id" in result.columns
    assert result.loc[0, "run_id"] == "onsite:shanghai:D01:A1:native_case:2325"
    assert result.loc[0, "candidate_key"] == "onsite:shanghai:D01:A1:frame:7"
    assert result.loc[0, "ego_id"] == result.loc[0, "vehicle_id"]
    assert "synthetic" not in " ".join(map(str, result.columns)).lower()
    assert "synthetic" not in result.astype(str).to_csv(index=False).lower()


def test_replace_rejects_noncanonical_directory(tmp_path) -> None:
    output = tmp_path / "lookalike"
    output.mkdir()
    (output / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "data_origin": DATA_ORIGIN,
                "required_mode": REQUIRED_MODE,
            }
        ),
        encoding="utf-8",
    )
    marker = output / "must_survive.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _prepare_output(output, replace=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_replace_rejects_unmanaged_or_modified_files(tmp_path, monkeypatch) -> None:
    output = tmp_path / "canonical"
    control = output / "00_control"
    control.mkdir(parents=True)
    (output / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "data_origin": DATA_ORIGIN,
                "required_mode": REQUIRED_MODE,
            }
        ),
        encoding="utf-8",
    )
    managed = output / "payload.txt"
    managed.write_text("managed", encoding="utf-8")
    _finalize_inventory(output)
    monkeypatch.setattr(template_module, "DEFAULT_OUTPUT", output)

    extra = output / "manual_note.txt"
    extra.write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="unmanaged"):
        _prepare_output(output, replace=True)
    assert extra.read_text(encoding="utf-8") == "preserve me"

    extra.unlink()
    managed.write_text("user changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since inventory"):
        _prepare_output(output, replace=True)
    assert managed.read_text(encoding="utf-8") == "user changed"


def test_fresh_minimal_package_rewrites_logs_and_builds_inventory(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_session = source_root / "raw/drivers/D01/sessions/source-session"
    tables = source_root / "tables"
    source_session.mkdir(parents=True)
    tables.mkdir()
    (source_root / "manifest.json").write_text(
        json.dumps({"schema_version": "source-v1"}), encoding="utf-8"
    )
    ego = {
        "timestamp": "1766197789851",
        "globalTimeStamp": "1766197789851",
        "frameId": 1,
        "id": "2490",
        "name": "SYNTHETIC_HUMAN_D01",
        "longitude": 121.2,
        "latitude": 31.2,
        "speed": 1.0,
        "isPerception": 0,
    }
    (source_session / "vehicle_trajectory.log").write_text(
        json.dumps(
            {
                "type": "start",
                "value": {
                    "timestamp": "1766197789851",
                    "value": [ego],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_session / "monitor.log").write_text(
        json.dumps(
            {
                "timestamp": 1766197789851,
                "frameId": 1,
                "taskId": "6923",
                "recordId": 1766197775,
                "avMonitor": {
                    "timestamp": 1766197789851,
                    "longitude": 0,
                    "latitude": 0,
                    "speed": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_session / "simulation_trajectory.log").write_text(
        json.dumps(
            {
                "type": "trajectory",
                "value": {
                    "timestamp": "1766197789851",
                    "value": [{"id": "1200002", "speed": 3.0}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_session / "vehicle_perception_simulation_trajectory.log").write_text(
        json.dumps(
            {
                "caseId": 2325,
                "taskId": "6923",
                "participantTrajectories": [
                    {"role": "av", "value": [ego]},
                    {
                        "role": "mvSimulation",
                        "value": [
                            {
                                "id": "1200002",
                                "name": "background",
                                "speed": 3.0,
                                "isPerception": 1,
                            }
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    one_entry = build_placeholder_registry()[0]
    monkeypatch.setattr(template_module, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(template_module, "RAW_TABLES", {})
    monkeypatch.setattr(template_module, "ANALYSIS_TABLES", {})
    monkeypatch.setattr(template_module, "build_placeholder_registry", lambda: [one_entry])
    output = tmp_path / "release"
    template_module.package_release(output)

    session = next((output / "01_collection_raw/raw/drivers/D01/sessions").iterdir())
    assert session.name == one_entry["session_id"]
    assert sorted(path.name for path in session.iterdir()) == sorted(
        template_module.REQUIRED_LOGS
    )
    with (session / "vehicle_trajectory.log").open(encoding="utf-8") as handle:
        rewritten = json.loads(handle.readline())["value"]["value"][0]
    assert rewritten["id"] == one_entry["vehicle_id"]
    assert rewritten["name"] == one_entry["vehicle_name"]
    assert template_module._scan_forbidden_payload(output) == []
    inventory = pd.read_csv(output / "00_control/file_inventory.csv")
    actual = {
        path.relative_to(output).as_posix()
        for path in template_module._managed_files(output)
    }
    assert set(inventory.relative_path) == actual


def test_production_mode_fails_closed_without_reading_payload(tmp_path) -> None:
    result = validate_release(tmp_path, mode="production", deep_hash=False, write_report=False)
    assert result["validation_status"] == "TEMPLATE_MODE_REQUIRED"
    assert result["failed_checks"] == ["template_mode_gate"]


@pytest.mark.skipif(
    not (DEFAULT_OUTPUT / "release_manifest.json").is_file(),
    reason="local RQ029 template release unavailable",
)
def test_current_template_release_passes_light_validation() -> None:
    result = validate_release(
        DEFAULT_OUTPUT,
        mode="template",
        deep_hash=False,
        write_report=False,
    )
    assert result["validation_status"] == "PASS"
    assert result["n_sessions"] == 20
    assert result["n_runs"] == 300
    assert result["n_frames"] == 81_880
