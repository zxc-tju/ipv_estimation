#!/usr/bin/env python3
"""RQ012A merge-validation gate.

This module validates two independently submitted annotation files before any
merge, agreement statistic, adjudication, or event-IPV analysis can run. It is
intentionally a guardrail only: accepted files are preserved and described in a
handoff manifest, but this script never computes agreement and never joins any
IPV, score, rank, team, area, or raw-media data.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RUN_ID = "RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37"
HUMAN_LABEL_STATUS = "BLOCKED_FOR_HUMAN_LABELS"

BEHAVIOR_COLUMNS = [
    "aggressive_intrusion",
    "appropriate_assertiveness",
    "over_yielding_freeze",
    "oscillation",
    "deadlock",
    "smooth_reciprocal_negotiation",
    "unrelated_failure",
]
TIMING_COLUMNS = ["event_start_sec", "event_end_sec"]
CONFIDENCE_COLUMN = "confidence_1_to_5"
NOTES_COLUMN = "free_text_notes"
LABEL_SIMILARITY_COLUMNS = [*BEHAVIOR_COLUMNS, CONFIDENCE_COLUMN]
REQUIRED_COLUMNS = [
    "neutral_item_id",
    *BEHAVIOR_COLUMNS,
    *TIMING_COLUMNS,
    CONFIDENCE_COLUMN,
    NOTES_COLUMN,
]
ALLOWED_COLUMNS = set(REQUIRED_COLUMNS)


@dataclass(frozen=True)
class NearDuplicateConfig:
    label_cell_similarity_threshold: float = 0.95
    minimum_comparable_label_cells: int = 20
    note_similarity_threshold: float = 0.95
    minimum_note_characters: int = 40
    minimum_completion_seconds: float = 300.0
    identical_timing_tolerance_seconds: float = 1.0
    order_similarity_threshold: float = 1.0


DEFAULT_NEAR_DUPLICATE_CONFIG = NearDuplicateConfig()

COMPLETION_DURATION_FIELDS = [
    "completion_duration_seconds",
    "annotation_duration_seconds",
    "elapsed_seconds",
    "annotation_elapsed_seconds",
]
START_TIME_FIELDS = [
    "annotation_started_at_utc",
    "work_started_at_utc",
    "started_at_utc",
    "start_time_utc",
]
END_TIME_FIELDS = [
    "annotation_completed_at_utc",
    "work_completed_at_utc",
    "completed_at_utc",
    "completion_time_utc",
]
CHANNEL_SUBMITTER_FIELDS = [
    "controlled_channel_submitter_id",
    "channel_bound_submitter_id",
    "submission_channel_submitter_id",
    "controlled_channel_verified_submitter_id",
]
CHANNEL_RECEIPT_FIELDS = [
    "controlled_channel_receipt_id",
    "controlled_submission_receipt_id",
    "submission_channel_receipt_id",
    "coordinator_receipt_id",
]

REAL_PROVENANCE_TRUE_FIELDS = [
    "human_attestation",
    "independent_work_attested",
    "authorized_blind_materials_only_attested",
    "no_ai_or_model_labels_attested",
    "no_copy_or_borrow_attested",
    "controlled_submission_channel_verified",
    "file_provenance_checked",
    "coordinator_custody_logged",
]
TEST_FIXTURE_PROVENANCE_TRUE_FIELDS = [
    "adversarial_test_fixture",
    "not_real_human_labels",
    "structural_gate_only",
]

LEAKAGE_HEADER_PATTERNS = [
    "team",
    "team_id",
    "official_score",
    "coordination_score",
    "score",
    "rank",
    "area_rank",
    "ipv",
    "area",
    "area_id",
    "scenario_id",
    "run_id",
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "path",
    "viewing_order",
    "item_order",
    "thumbnail",
    "manifest_stratum",
    "manifest_strata",
    "stratum",
    "strata",
    "prior_annotation",
    "borrowed_annotation",
    "source_id",
    "identity_map",
]
LEAKAGE_TEXT_RE = re.compile(
    r"\b(team|official[_ -]?score|coordination[_ -]?score|score|rank|area[_ -]?rank|"
    r"ipv|area[_ -]?id|scenario[_ -]?id|run[_ -]?id|filename|file[_ -]?name|"
    r"file[_ -]?path|thumbnail|manifest[_ -]?strat|prior[_ -]?annotation|"
    r"borrowed[_ -]?annotation|identity[_ -]?map)\b",
    re.IGNORECASE,
)


class ValidationError(ValueError):
    """Structured validation failure for an unsafe annotation pair."""

    def __init__(self, code: str, message: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class QuarantineError(ValueError):
    """Structured quarantine for suspicious non-identical annotation pairs."""

    def __init__(self, code: str, message: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class CsvData:
    path: Path
    fieldnames: List[str]
    rows: List[Dict[str, str]]


@dataclass(frozen=True)
class RawArchiveRecord:
    role: str
    source_path: str
    sha256: str
    byte_size: int
    archive_path: str
    already_present: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def read_expected_items(path: Path) -> List[str]:
    if not path.exists():
        raise ValidationError("expected_item_manifest_missing", f"Expected-item manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "neutral_item_id" not in reader.fieldnames:
            raise ValidationError(
                "expected_item_manifest_invalid",
                f"Expected-item manifest must contain neutral_item_id: {path}",
            )
        items = [row["neutral_item_id"].strip() for row in reader if row.get("neutral_item_id", "").strip()]
    if not items:
        raise ValidationError("expected_item_manifest_empty", f"Expected-item manifest has no neutral items: {path}")
    if len(items) != len(set(items)):
        raise ValidationError("expected_item_manifest_duplicate_ids", "Expected-item manifest has duplicate IDs")
    return items


def archive_raw_input(path: Path, role: str, archive_dir: Path) -> RawArchiveRecord:
    if not path.exists():
        raise ValidationError("input_file_missing", f"{role} input file missing: {path}", {"role": role})
    digest = sha256_file(path)
    byte_size = path.stat().st_size
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{role}_{digest[:16]}_{path.name}"
    archive_path = archive_dir / archive_name
    already_present = archive_path.exists()
    if already_present:
        if sha256_file(archive_path) != digest:
            raise ValidationError(
                "raw_archive_collision",
                f"Archive path exists with different hash: {archive_path}",
                {"role": role, "archive_path": str(archive_path)},
            )
    else:
        shutil.copy2(path, archive_path)
    return RawArchiveRecord(
        role=role,
        source_path=str(path),
        sha256=digest,
        byte_size=byte_size,
        archive_path=str(archive_path),
        already_present=already_present,
    )


def read_csv_data(path: Path) -> CsvData:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: (value if value is not None else "") for key, value in row.items()} for row in reader]
    if not fieldnames:
        raise ValidationError("schema_missing_header", f"{path} has no CSV header")
    return CsvData(path=path, fieldnames=fieldnames, rows=rows)


def scan_for_leakage(data: CsvData, role: str) -> None:
    for column in data.fieldnames:
        normalized = normalize_name(column)
        if normalized in LEAKAGE_HEADER_PATTERNS:
            raise ValidationError(
                "protected_identity_or_proxy_leakage",
                f"{role} file contains prohibited leakage/proxy column {column!r}",
                {"role": role, "column": column},
            )
    for row_index, row in enumerate(data.rows, start=2):
        notes = row.get(NOTES_COLUMN, "")
        match = LEAKAGE_TEXT_RE.search(notes)
        if match:
            raise ValidationError(
                "protected_identity_or_proxy_leakage",
                f"{role}:{row_index} notes contain prohibited leakage/proxy token {match.group(0)!r}",
                {"role": role, "row": row_index, "token": match.group(0)},
            )


def validate_schema(data: CsvData, role: str) -> None:
    present = set(data.fieldnames)
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise ValidationError(
            "schema_missing_required_columns",
            f"{role} file is missing required columns: {missing}",
            {"role": role, "missing_columns": missing},
        )
    unexpected = [column for column in data.fieldnames if column not in ALLOWED_COLUMNS]
    if unexpected:
        raise ValidationError(
            "schema_unexpected_columns",
            f"{role} file contains unexpected non-template columns: {unexpected}",
            {"role": role, "unexpected_columns": unexpected},
        )


def validate_item_ids(data: CsvData, expected_items: Sequence[str], role: str) -> None:
    if not data.rows:
        raise ValidationError("empty_submission", f"{role} file contains no annotation rows", {"role": role})
    seen: Dict[str, int] = {}
    for row_index, row in enumerate(data.rows, start=2):
        item_id = row.get("neutral_item_id", "").strip()
        if not item_id:
            raise ValidationError("blank_neutral_item_id", f"{role}:{row_index} has blank neutral_item_id")
        if item_id in seen:
            raise ValidationError(
                "duplicate_neutral_item_id",
                f"{role}:{row_index} duplicates neutral_item_id {item_id}",
                {"role": role, "neutral_item_id": item_id, "first_row": seen[item_id], "duplicate_row": row_index},
            )
        seen[item_id] = row_index
    expected_set = set(expected_items)
    actual_set = set(seen)
    if actual_set != expected_set:
        raise ValidationError(
            "item_id_set_mismatch",
            f"{role} neutral_item_id set does not match issued manifest",
            {
                "role": role,
                "unexpected_ids": sorted(actual_set - expected_set),
                "missing_ids": sorted(expected_set - actual_set),
            },
        )


def validate_value_shape(value: str, allowed: Iterable[str], code: str, message: str, details: Mapping[str, Any]) -> None:
    if value not in set(allowed):
        raise ValidationError(code, message, details)


def validate_rows(data: CsvData, role: str, structural_only: bool) -> None:
    for row_index, row in enumerate(data.rows, start=2):
        item_id = row.get("neutral_item_id", "").strip()
        for column in BEHAVIOR_COLUMNS:
            value = row.get(column, "").strip()
            if structural_only and value == "":
                continue
            if value == "":
                raise ValidationError(
                    "incomplete_required_fields",
                    f"{role}:{row_index} {column} is blank; completed submissions require 0/1 labels",
                    {"role": role, "row": row_index, "neutral_item_id": item_id, "column": column},
                )
            validate_value_shape(
                value,
                {"0", "1"},
                "invalid_label_value",
                f"{role}:{row_index} {column} must be 0 or 1, got {value!r}",
                {"role": role, "row": row_index, "neutral_item_id": item_id, "column": column, "value": value},
            )

        confidence = row.get(CONFIDENCE_COLUMN, "").strip()
        if structural_only and confidence == "":
            pass
        elif confidence == "":
            raise ValidationError(
                "incomplete_required_fields",
                f"{role}:{row_index} {CONFIDENCE_COLUMN} is blank; completed submissions require confidence 1..5",
                {"role": role, "row": row_index, "neutral_item_id": item_id, "column": CONFIDENCE_COLUMN},
            )
        else:
            validate_value_shape(
                confidence,
                {"1", "2", "3", "4", "5"},
                "invalid_confidence_value",
                f"{role}:{row_index} {CONFIDENCE_COLUMN} must be 1..5, got {confidence!r}",
                {
                    "role": role,
                    "row": row_index,
                    "neutral_item_id": item_id,
                    "column": CONFIDENCE_COLUMN,
                    "value": confidence,
                },
            )

        start = row.get("event_start_sec", "").strip()
        end = row.get("event_end_sec", "").strip()
        if start or end:
            if not start or not end:
                raise ValidationError(
                    "incomplete_event_time_pair",
                    f"{role}:{row_index} event_start_sec and event_end_sec must be supplied together",
                    {"role": role, "row": row_index, "neutral_item_id": item_id},
                )
            try:
                start_value = float(start)
                end_value = float(end)
            except ValueError as exc:
                raise ValidationError(
                    "invalid_event_time_value",
                    f"{role}:{row_index} event times must be numeric",
                    {"role": role, "row": row_index, "neutral_item_id": item_id},
                ) from exc
            if start_value < 0 or end_value < 0 or end_value < start_value:
                raise ValidationError(
                    "invalid_event_time_order",
                    f"{role}:{row_index} event times must be non-negative and end >= start",
                    {"role": role, "row": row_index, "neutral_item_id": item_id},
                )


def load_provenance(path: Optional[Path], role: str) -> Dict[str, Any]:
    if path is None:
        raise ValidationError("provenance_attestation_missing", f"{role} provenance sidecar was not supplied")
    if not path.exists():
        raise ValidationError("provenance_attestation_missing", f"{role} provenance sidecar missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError("provenance_sidecar_invalid_json", f"{role} provenance sidecar is not valid JSON") from exc


def require_true(provenance: Mapping[str, Any], field: str, role: str, test_fixture: bool) -> None:
    if provenance.get(field) is not True:
        code = "test_fixture_provenance_missing" if test_fixture else "provenance_attestation_missing"
        raise ValidationError(
            code,
            f"{role} provenance does not affirm required field {field!r}",
            {"role": role, "field": field},
        )


def validate_provenance(
    provenance_01_path: Optional[Path],
    provenance_02_path: Optional[Path],
    *,
    structural_only: bool,
    allow_test_fixtures: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    prov_01 = load_provenance(provenance_01_path, "annotator_01")
    prov_02 = load_provenance(provenance_02_path, "annotator_02")

    for role, provenance in [("annotator_01", prov_01), ("annotator_02", prov_02)]:
        if provenance.get("annotator_role") != role:
            raise ValidationError(
                "provenance_role_mismatch",
                f"{role} provenance has annotator_role={provenance.get('annotator_role')!r}",
                {"role": role, "annotator_role": provenance.get("annotator_role")},
            )

    test_fixture_mode = False
    if allow_test_fixtures and structural_only:
        fixture_flags = [
            bool(prov_01.get("adversarial_test_fixture")),
            bool(prov_02.get("adversarial_test_fixture")),
        ]
        if any(fixture_flags):
            test_fixture_mode = True
            for role, provenance in [("annotator_01", prov_01), ("annotator_02", prov_02)]:
                for field in TEST_FIXTURE_PROVENANCE_TRUE_FIELDS:
                    require_true(provenance, field, role, test_fixture=True)
                if provenance.get("human_attestation") is True:
                    raise ValidationError(
                        "test_fixture_must_not_claim_human_attestation",
                        f"{role} test fixture provenance must not claim real human attestation",
                        {"role": role},
                    )
    if not test_fixture_mode:
        for role, provenance in [("annotator_01", prov_01), ("annotator_02", prov_02)]:
            for field in REAL_PROVENANCE_TRUE_FIELDS:
                require_true(provenance, field, role, test_fixture=False)

    identity_01 = str(prov_01.get("coordinator_verified_submitter_id", "")).strip()
    identity_02 = str(prov_02.get("coordinator_verified_submitter_id", "")).strip()
    if not identity_01 or not identity_02:
        raise ValidationError(
            "provenance_identity_missing",
            "Both provenance sidecars must include coordinator_verified_submitter_id",
        )
    if identity_01 == identity_02:
        raise ValidationError(
            "duplicate_annotator_identity",
            "annotator_01 and annotator_02 provenance records identify the same submitter",
            {"coordinator_verified_submitter_id": identity_01},
        )
    return prov_01, prov_02, test_fixture_mode


def validate_no_duplicate_files(raw_01: RawArchiveRecord, raw_02: RawArchiveRecord) -> None:
    if raw_01.sha256 == raw_02.sha256:
        raise ValidationError(
            "copied_duplicate_submission",
            "annotator_02 file is byte/hash-identical to annotator_01 file",
            {"sha256": raw_01.sha256},
        )


def first_present(provenance: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = str(provenance.get(field, "")).strip()
        if value:
            return value
    return ""


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def completion_timing(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    start = parse_timestamp(first_present(provenance, START_TIME_FIELDS))
    end = parse_timestamp(first_present(provenance, END_TIME_FIELDS))
    duration = None
    for field in COMPLETION_DURATION_FIELDS:
        duration = parse_float(provenance.get(field))
        if duration is not None:
            break
    if duration is None and start is not None and end is not None:
        duration = (end - start).total_seconds()
    return {
        "start_utc": start.isoformat() if start else "",
        "end_utc": end.isoformat() if end else "",
        "duration_seconds": duration,
    }


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def label_cell_similarity(data_01: CsvData, data_02: CsvData) -> Dict[str, Any]:
    rows_02 = {row["neutral_item_id"]: row for row in data_02.rows}
    identical = 0
    comparable = 0
    for row_01 in data_01.rows:
        row_02 = rows_02.get(row_01["neutral_item_id"])
        if row_02 is None:
            continue
        for column in LABEL_SIMILARITY_COLUMNS:
            value_01 = row_01.get(column, "").strip()
            value_02 = row_02.get(column, "").strip()
            if not value_01 and not value_02:
                continue
            comparable += 1
            if value_01 == value_02:
                identical += 1
    ratio = identical / comparable if comparable else None
    return {
        "identical_label_confidence_cells": identical,
        "comparable_label_confidence_cells": comparable,
        "similarity": ratio,
    }


def note_text_similarity(data_01: CsvData, data_02: CsvData, config: NearDuplicateConfig) -> Dict[str, Any]:
    rows_02 = {row["neutral_item_id"]: row for row in data_02.rows}
    notes_01: List[str] = []
    notes_02: List[str] = []
    for row_01 in data_01.rows:
        row_02 = rows_02.get(row_01["neutral_item_id"])
        if row_02 is None:
            continue
        note_01 = normalized_text(row_01.get(NOTES_COLUMN, ""))
        note_02 = normalized_text(row_02.get(NOTES_COLUMN, ""))
        if note_01 and note_02:
            notes_01.append(note_01)
            notes_02.append(note_02)
    text_01 = "\n".join(notes_01)
    text_02 = "\n".join(notes_02)
    min_characters = min(len(text_01), len(text_02))
    ratio = None
    if min_characters >= config.minimum_note_characters:
        ratio = difflib.SequenceMatcher(None, text_01, text_02).ratio()
    return {
        "paired_nonblank_note_rows": len(notes_01),
        "minimum_note_characters": min_characters,
        "similarity": ratio,
    }


def item_order_similarity(data_01: CsvData, data_02: CsvData) -> Dict[str, Any]:
    ids_01 = [row["neutral_item_id"] for row in data_01.rows]
    ids_02 = [row["neutral_item_id"] for row in data_02.rows]
    comparable = min(len(ids_01), len(ids_02))
    matches = sum(1 for left, right in zip(ids_01, ids_02) if left == right)
    ratio = matches / comparable if comparable else None
    return {
        "matching_positions": matches,
        "comparable_positions": comparable,
        "similarity": ratio,
        "exact_same_order": ids_01 == ids_02 and comparable > 1,
    }


def timing_similarity(
    provenance_01: Mapping[str, Any],
    provenance_02: Mapping[str, Any],
    config: NearDuplicateConfig,
) -> Dict[str, Any]:
    timing_01 = completion_timing(provenance_01)
    timing_02 = completion_timing(provenance_02)
    durations = [timing_01.get("duration_seconds"), timing_02.get("duration_seconds")]
    short_durations = [duration for duration in durations if isinstance(duration, (int, float)) and duration < config.minimum_completion_seconds]
    start_01 = parse_timestamp(first_present(provenance_01, START_TIME_FIELDS))
    start_02 = parse_timestamp(first_present(provenance_02, START_TIME_FIELDS))
    end_01 = parse_timestamp(first_present(provenance_01, END_TIME_FIELDS))
    end_02 = parse_timestamp(first_present(provenance_02, END_TIME_FIELDS))
    identical_window = False
    if start_01 and start_02 and end_01 and end_02:
        start_delta = abs((start_01 - start_02).total_seconds())
        end_delta = abs((end_01 - end_02).total_seconds())
        identical_window = (
            start_delta <= config.identical_timing_tolerance_seconds
            and end_delta <= config.identical_timing_tolerance_seconds
        )
    return {
        "annotator_01": timing_01,
        "annotator_02": timing_02,
        "short_duration_count": len(short_durations),
        "identical_timing_window": identical_window,
    }


def channel_evidence_summary(
    provenance_01: Mapping[str, Any],
    provenance_02: Mapping[str, Any],
    *,
    test_fixture_mode: bool,
) -> Dict[str, Any]:
    coordinator_01 = str(provenance_01.get("coordinator_verified_submitter_id", "")).strip()
    coordinator_02 = str(provenance_02.get("coordinator_verified_submitter_id", "")).strip()
    channel_01 = first_present(provenance_01, CHANNEL_SUBMITTER_FIELDS)
    channel_02 = first_present(provenance_02, CHANNEL_SUBMITTER_FIELDS)
    receipt_01 = first_present(provenance_01, CHANNEL_RECEIPT_FIELDS)
    receipt_02 = first_present(provenance_02, CHANNEL_RECEIPT_FIELDS)
    issues: List[str] = []

    for role, coordinator_id, channel_id in [
        ("annotator_01", coordinator_01, channel_01),
        ("annotator_02", coordinator_02, channel_02),
    ]:
        if not channel_id:
            if not test_fixture_mode:
                issues.append(f"{role}:missing_channel_submitter_id")
            continue
        if channel_id != coordinator_id:
            issues.append(f"{role}:channel_submitter_mismatch")

    if channel_01 and channel_02 and channel_01 == channel_02:
        issues.append("duplicate_channel_submitter_id")
    if receipt_01 and receipt_02 and receipt_01 == receipt_02:
        issues.append("duplicate_channel_receipt_id")

    return {
        "annotator_01_channel_submitter_id_present": bool(channel_01),
        "annotator_02_channel_submitter_id_present": bool(channel_02),
        "annotator_01_channel_receipt_id_present": bool(receipt_01),
        "annotator_02_channel_receipt_id_present": bool(receipt_02),
        "issues": issues,
    }


def run_near_duplicate_screen(
    data_01: CsvData,
    data_02: CsvData,
    provenance_01: Mapping[str, Any],
    provenance_02: Mapping[str, Any],
    *,
    test_fixture_mode: bool,
    config: NearDuplicateConfig,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "config": asdict(config),
        "label_cell_similarity": label_cell_similarity(data_01, data_02),
        "note_text_similarity": note_text_similarity(data_01, data_02, config),
        "completion_timing": timing_similarity(provenance_01, provenance_02, config),
        "item_order_similarity": item_order_similarity(data_01, data_02),
        "controlled_channel_evidence": channel_evidence_summary(
            provenance_01,
            provenance_02,
            test_fixture_mode=test_fixture_mode,
        ),
    }

    label_similarity = summary["label_cell_similarity"]["similarity"]
    comparable_cells = summary["label_cell_similarity"]["comparable_label_confidence_cells"]
    if (
        label_similarity is not None
        and comparable_cells >= config.minimum_comparable_label_cells
        and label_similarity >= config.label_cell_similarity_threshold
    ):
        raise QuarantineError(
            "near_duplicate_label_pattern",
            "High label+confidence cell similarity requires coordinator review before agreement handoff",
            summary,
        )

    note_similarity = summary["note_text_similarity"]["similarity"]
    if note_similarity is not None and note_similarity >= config.note_similarity_threshold:
        raise QuarantineError(
            "near_duplicate_notes",
            "High free-text note similarity requires coordinator review before agreement handoff",
            summary,
        )

    timing = summary["completion_timing"]
    if timing["short_duration_count"] or timing["identical_timing_window"]:
        raise QuarantineError(
            "implausible_completion_timing",
            "Implausibly fast or identical completion timing requires coordinator review before agreement handoff",
            summary,
        )

    order_similarity = summary["item_order_similarity"]["similarity"]
    if (
        order_similarity is not None
        and summary["item_order_similarity"]["exact_same_order"]
        and order_similarity >= config.order_similarity_threshold
    ):
        raise QuarantineError(
            "template_order_clone",
            "Annotator files use the same item order where assigned templates should differ",
            summary,
        )

    channel_issues = summary["controlled_channel_evidence"]["issues"]
    if channel_issues:
        raise QuarantineError(
            "controlled_channel_evidence_anomaly",
            "Controlled-channel evidence does not independently support two distinct submitters",
            summary,
        )

    summary["verdict"] = "PASS_NO_NEAR_DUPLICATE_OR_COLLUSION_FLAGS"
    return summary


def validate_pair(
    annotator_01: Path,
    annotator_02: Path,
    *,
    expected_items: Path,
    provenance_01: Optional[Path] = None,
    provenance_02: Optional[Path] = None,
    raw_archive_dir: Optional[Path] = None,
    report_json: Optional[Path] = None,
    structural_only: bool = False,
    allow_test_fixtures: bool = False,
    near_duplicate_config: Optional[NearDuplicateConfig] = None,
) -> Dict[str, Any]:
    archive_dir = raw_archive_dir or (Path(__file__).resolve().parent / "raw_input_archive")
    duplicate_config = near_duplicate_config or DEFAULT_NEAR_DUPLICATE_CONFIG
    status: Dict[str, Any] = {
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "structural_only": structural_only,
        "allow_test_fixtures": allow_test_fixtures,
        "near_duplicate_config": asdict(duplicate_config),
        "status": "ERROR",
        "accepted_as_real_human_labels": False,
        "agreement_computed": False,
        "event_ipv_association_computed": False,
        "merge_or_agreement_output_created": False,
        "human_label_status": HUMAN_LABEL_STATUS,
    }

    raw_records: List[RawArchiveRecord] = []
    try:
        raw_01 = archive_raw_input(annotator_01, "annotator_01", archive_dir)
        raw_02 = archive_raw_input(annotator_02, "annotator_02", archive_dir)
        raw_records.extend([raw_01, raw_02])
        status["raw_input_archive"] = [record.__dict__ for record in raw_records]

        validate_no_duplicate_files(raw_01, raw_02)

        expected = read_expected_items(expected_items)
        data_01 = read_csv_data(annotator_01)
        data_02 = read_csv_data(annotator_02)

        for role, data in [("annotator_01", data_01), ("annotator_02", data_02)]:
            scan_for_leakage(data, role)
            validate_schema(data, role)
            validate_item_ids(data, expected, role)
            validate_rows(data, role, structural_only=structural_only)

        prov_01, prov_02, test_fixture_mode = validate_provenance(
            provenance_01,
            provenance_02,
            structural_only=structural_only,
            allow_test_fixtures=allow_test_fixtures,
        )

        status["near_duplicate_screen"] = run_near_duplicate_screen(
            data_01,
            data_02,
            prov_01,
            prov_02,
            test_fixture_mode=test_fixture_mode,
            config=duplicate_config,
        )

        handoff = {
            "frozen_agreement_protocol": "01_results/agreement_analysis_protocol.yaml",
            "raw_input_archive": [record.__dict__ for record in raw_records],
            "annotator_01_provenance_digest": hashlib.sha256(
                json.dumps(prov_01, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "annotator_02_provenance_digest": hashlib.sha256(
                json.dumps(prov_02, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "neutral_item_count": len(expected),
        }
        status.update(
            {
                "status": "PASS_STRUCTURAL_ONLY" if structural_only else "PASS_READY_FOR_AGREEMENT_HANDOFF",
                "accepted_as_real_human_labels": not test_fixture_mode and not structural_only,
                "test_fixture_mode": test_fixture_mode,
                "neutral_item_count": len(expected),
                "handoff_manifest": handoff,
            }
        )
    except QuarantineError as exc:
        status.update(
            {
                "status": "QUARANTINED",
                "quarantine_code": exc.code,
                "quarantine_reason": str(exc),
                "quarantine_details": exc.details,
            }
        )
    except ValidationError as exc:
        status.update(
            {
                "status": "REJECTED",
                "rejection_code": exc.code,
                "rejection_reason": str(exc),
                "rejection_details": exc.details,
            }
        )
    finally:
        if report_json is not None:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            report_json.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return status


def default_run_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    run_root = default_run_root()
    merge_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate two RQ012A annotation submissions before any merge, agreement, "
            "adjudication, or event-IPV association."
        )
    )
    parser.add_argument("--annotator-01", required=True, type=Path)
    parser.add_argument("--annotator-02", required=True, type=Path)
    parser.add_argument("--provenance-01", type=Path)
    parser.add_argument("--provenance-02", type=Path)
    parser.add_argument(
        "--expected-items",
        default=run_root / "01_results" / "annotations" / "neutral_item_manifest.csv",
        type=Path,
    )
    parser.add_argument("--raw-archive-dir", default=merge_dir / "raw_input_archive", type=Path)
    parser.add_argument("--report-json", default=merge_dir / "latest_validation_status.json", type=Path)
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument(
        "--allow-test-fixtures",
        action="store_true",
        help="Allow explicitly marked structural-only test fixtures; never accepts them as human labels.",
    )
    parser.add_argument(
        "--label-cell-similarity-threshold",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.label_cell_similarity_threshold,
        type=float,
        help="Quarantine threshold for identical label+confidence cells across aligned neutral items.",
    )
    parser.add_argument(
        "--minimum-comparable-label-cells",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.minimum_comparable_label_cells,
        type=int,
        help="Minimum nonblank comparable label+confidence cells before label similarity can quarantine.",
    )
    parser.add_argument(
        "--note-similarity-threshold",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.note_similarity_threshold,
        type=float,
        help="Quarantine threshold for paired nonblank free-text note similarity.",
    )
    parser.add_argument(
        "--minimum-note-characters",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.minimum_note_characters,
        type=int,
        help="Minimum paired note text length before note similarity can quarantine.",
    )
    parser.add_argument(
        "--minimum-completion-seconds",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.minimum_completion_seconds,
        type=float,
        help="Quarantine any provenance completion duration shorter than this many seconds.",
    )
    parser.add_argument(
        "--identical-timing-tolerance-seconds",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.identical_timing_tolerance_seconds,
        type=float,
        help="Tolerance for treating two start/end completion windows as identical.",
    )
    parser.add_argument(
        "--order-similarity-threshold",
        default=DEFAULT_NEAR_DUPLICATE_CONFIG.order_similarity_threshold,
        type=float,
        help="Quarantine threshold for same-position neutral-item order similarity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    duplicate_config = NearDuplicateConfig(
        label_cell_similarity_threshold=args.label_cell_similarity_threshold,
        minimum_comparable_label_cells=args.minimum_comparable_label_cells,
        note_similarity_threshold=args.note_similarity_threshold,
        minimum_note_characters=args.minimum_note_characters,
        minimum_completion_seconds=args.minimum_completion_seconds,
        identical_timing_tolerance_seconds=args.identical_timing_tolerance_seconds,
        order_similarity_threshold=args.order_similarity_threshold,
    )
    result = validate_pair(
        args.annotator_01,
        args.annotator_02,
        expected_items=args.expected_items,
        provenance_01=args.provenance_01,
        provenance_02=args.provenance_02,
        raw_archive_dir=args.raw_archive_dir,
        report_json=args.report_json,
        structural_only=args.structural_only,
        allow_test_fixtures=args.allow_test_fixtures,
        near_duplicate_config=duplicate_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
