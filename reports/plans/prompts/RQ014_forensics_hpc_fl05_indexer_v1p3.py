#!/usr/bin/env python3
"""Fail-closed, atomic index of recorded RQ010B correlation statistics.

This script performs read-only discovery.  It does not classify any row as the
lost historical result; variable and scope fields are copied only when they are
explicitly recorded in the source.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import uuid


SCHEMA_VERSION = "rq014-fl05-index-v1p3"
LOGIN_NODE_BYTE_BUDGET = 200 * 1024 * 1024
COMPUTE_NODE_BYTE_BUDGET = 4 * 1024 * 1024 * 1024
SUPPORTED_SUFFIXES = {".csv", ".json", ".md"}
OUTPUT_NAME = "historical_stats_index.csv"
AUDIT_JSON_NAME = "fl05_file_audit.json"
AUDIT_CSV_NAME = "fl05_file_audit.csv"
MANIFEST_NAME = "fl05_run_manifest.json"
DONE_NAME = "DONE"
CURRENT_NAME = "CURRENT"
SLURM_JOB_NAME = "zxc-rq014-fl05"
SLURM_VERIFICATION_ENV = "RQ014_FL05_VERIFIED_SLURM_WRAPPER"
SLURM_PYTHON_REALPATH_ENV = "RQ014_FL05_PYTHON_REALPATH"
SLURM_PYTHON_SHA256_ENV = "RQ014_FL05_PYTHON_SHA256"
SAFE_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

STAT_PATTERN = re.compile(
    r"(?:spearman|pearson|kendall|rho|corr(?:elation)?)", re.IGNORECASE
)
STAT_KEY_PATTERN = (
    r"[A-Za-z0-9_.-]*(?:spearman|pearson|kendall|rho|corr(?:elation)?)"
    r"[A-Za-z0-9_.-]*"
)
MARKDOWN_ASSIGNMENT = re.compile(
    rf"(?P<name>{STAT_KEY_PATTERN})\s*(?:=|:)\s*(?P<value>[^\s,;|)\]]+)",
    re.IGNORECASE,
)

NAME_ALIASES = {
    "statistic_name",
    "statistic",
    "metric_name",
    "metric",
    "measure_name",
    "measure",
    "signal_name",
    "signal",
    "correlation_name",
}
VALUE_ALIASES = (
    "value",
    "statistic_value",
    "stat_value",
    "estimate",
    "effect",
    "coefficient",
    "coef",
    "correlation_value",
)
FIELD_ALIASES = {
    "n": ("n", "count", "n_scenes", "n_segments", "n_rows", "n_candidates"),
    "unit": ("unit", "analysis_unit", "aggregation_unit", "level"),
    "variable_x": ("variable_x", "var_x", "x_variable", "predictor", "exposure"),
    "variable_y": ("variable_y", "var_y", "y_variable", "outcome", "response"),
    "direction_as_recorded": ("direction", "direction_as_recorded", "sign", "relationship"),
    "dataset_scope": ("dataset", "dataset_name", "corpus", "data_scope"),
    "candidate_scope": ("candidate_scope", "scope", "population", "subset", "split"),
}

OUTPUT_FIELDS = [
    "statistic_name",
    "value",
    "n",
    "unit",
    "variable_x",
    "variable_y",
    "direction_as_recorded",
    "dataset_scope",
    "candidate_scope",
    "fingerprint_disposition",
    "config_fields_as_recorded",
    "source_file",
    "source_file_sha256",
    "mtime",
    "parse_status",
    "raw_locator",
    "raw_text_as_recorded",
]
AUDIT_FIELDS = [
    "source_root",
    "source_file",
    "bytes",
    "source_file_sha256",
    "mtime",
    "parser",
    "parser_status",
    "row_count",
    "parsed_row_count",
    "unparsed_candidate_count",
    "error",
]


class FatalError(RuntimeError):
    """An error that prevents a complete, trustworthy index."""


def normalize_key(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def recorded_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def normalized_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in context.items():
        if is_scalar(value):
            result.setdefault(normalize_key(key), value)
    return result


def first_recorded(context: Mapping[str, Any], aliases: Iterable[str]) -> str:
    normalized = normalized_context(context)
    for alias in aliases:
        value = normalized.get(alias)
        if value is not None and recorded_text(value):
            return recorded_text(value)
    return ""


def metadata_from_context(context: Mapping[str, Any]) -> Dict[str, str]:
    return {
        field: first_recorded(context, aliases)
        for field, aliases in FIELD_ALIASES.items()
    }


def config_json(context: Mapping[str, Any]) -> str:
    serializable = {
        str(key): value for key, value in context.items() if is_scalar(value)
    }
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def validate_numeric(value: Any, statistic_name: str) -> Tuple[Optional[str], str]:
    if isinstance(value, bool) or value is None:
        return None, "value is not a finite number"
    raw = recorded_text(value)
    if not raw:
        return None, "value is empty"
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None, "value is not numeric"
    if not math.isfinite(number):
        return None, "value is not finite"

    normalized_name = normalize_key(statistic_name)
    is_probability = bool(
        re.search(r"(?:^|_)(?:p|pvalue|p_value|probability)(?:$|_)", normalized_name)
    )
    if is_probability:
        if not 0.0 <= number <= 1.0:
            return None, "probability attached to correlation is outside [0, 1]"
    elif not -1.0 <= number <= 1.0:
        return None, "correlation coefficient is outside [-1, 1]"
    return raw, ""


class RowSink:
    """Write rows to an unpublished spool while tracking per-file counts."""

    def __init__(self, writer: csv.DictWriter[str]) -> None:
        self.writer = writer
        self.total = 0
        self.parsed = 0
        self.unparsed_count = 0

    def emit(
        self,
        *,
        statistic_name: str,
        value: str,
        context: Mapping[str, Any],
        source: Mapping[str, Any],
        parse_status: str,
        raw_locator: str,
        raw_text: str,
    ) -> None:
        metadata = metadata_from_context(context)
        parsed = parse_status.startswith("PARSED_")
        row = {
            "statistic_name": statistic_name,
            "value": value,
            **metadata,
            "fingerprint_disposition": "UNADJUDICATED" if parsed else "NOT_EVALUABLE",
            "config_fields_as_recorded": config_json(context),
            "source_file": source["source_file"],
            "source_file_sha256": source["source_file_sha256"],
            "mtime": source["mtime"],
            "parse_status": parse_status,
            "raw_locator": raw_locator,
            "raw_text_as_recorded": raw_text,
        }
        self.writer.writerow(row)
        self.total += 1
        if parsed:
            self.parsed += 1
        else:
            self.unparsed_count += 1

    def candidate(
        self,
        *,
        statistic_name: str,
        candidate_value: Any,
        context: Mapping[str, Any],
        source: Mapping[str, Any],
        parsed_status: str,
        raw_locator: str,
        raw_text: str,
    ) -> None:
        value, reason = validate_numeric(candidate_value, statistic_name)
        if value is not None:
            self.emit(
                statistic_name=statistic_name,
                value=value,
                context=context,
                source=source,
                parse_status=parsed_status,
                raw_locator=raw_locator,
                raw_text=raw_text,
            )
            return
        failure_context = dict(context)
        failure_context["unparsed_reason"] = reason
        failure_context["candidate_value_as_recorded"] = recorded_text(candidate_value)
        self.emit(
            statistic_name=statistic_name,
            value="",
            context=failure_context,
            source=source,
            parse_status="UNPARSED_CANDIDATE",
            raw_locator=raw_locator,
            raw_text=raw_text,
        )

    def unparsed(
        self,
        *,
        statistic_name: str,
        context: Mapping[str, Any],
        source: Mapping[str, Any],
        raw_locator: str,
        raw_text: str,
        reason: str,
    ) -> None:
        failure_context = dict(context)
        failure_context["unparsed_reason"] = reason
        self.emit(
            statistic_name=statistic_name,
            value="",
            context=failure_context,
            source=source,
            parse_status="UNPARSED_CANDIDATE",
            raw_locator=raw_locator,
            raw_text=raw_text,
        )


def parse_csv(text: str, source: Mapping[str, Any], sink: RowSink) -> None:
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader, None)
    except csv.Error as exc:
        raise FatalError(f"CSV parser failed for {source['source_file']}: {exc}") from exc
    if not header or not any(cell.strip() for cell in header):
        raise FatalError(f"CSV has no usable header: {source['source_file']}")

    normalized_header = [normalize_key(cell) for cell in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise FatalError(f"CSV has duplicate normalized headers: {source['source_file']}")
    name_columns = [
        index for index, key in enumerate(normalized_header) if key in NAME_ALIASES
    ]
    value_columns = [
        normalized_header.index(alias)
        for alias in VALUE_ALIASES
        if alias in normalized_header
    ]
    metadata_aliases = set(NAME_ALIASES) | set(VALUE_ALIASES)
    for aliases in FIELD_ALIASES.values():
        metadata_aliases.update(aliases)
    wide_columns = [
        index
        for index, cell in enumerate(header)
        if STAT_PATTERN.search(cell) and normalized_header[index] not in metadata_aliases
    ]

    try:
        for row_number, row in enumerate(reader, 2):
            if not any(cell.strip() for cell in row):
                continue
            raw_text = json.dumps(row, ensure_ascii=False)
            if len(row) != len(header):
                if wide_columns or STAT_PATTERN.search(" ".join(row)):
                    sink.unparsed(
                        statistic_name="csv_malformed_candidate_row",
                        context={"raw_row": raw_text},
                        source=source,
                        raw_locator=f"row:{row_number}",
                        raw_text=raw_text,
                        reason=f"row has {len(row)} fields; expected {len(header)}",
                    )
                continue

            context = dict(zip(header, row))
            emitted = False
            for name_index in name_columns:
                statistic_name = row[name_index].strip()
                if not STAT_PATTERN.search(statistic_name):
                    continue
                emitted = True
                if not value_columns:
                    sink.unparsed(
                        statistic_name=statistic_name,
                        context=context,
                        source=source,
                        raw_locator=f"row:{row_number},column:{header[name_index]}",
                        raw_text=raw_text,
                        reason="long-form statistic record has no recognized value column",
                    )
                    continue
                value_index = value_columns[0]
                sink.candidate(
                    statistic_name=statistic_name,
                    candidate_value=row[value_index],
                    context=context,
                    source=source,
                    parsed_status="PARSED_CSV_LONG",
                    raw_locator=f"row:{row_number},column:{header[value_index]}",
                    raw_text=raw_text,
                )

            for statistic_index in wide_columns:
                emitted = True
                sink.candidate(
                    statistic_name=header[statistic_index].strip(),
                    candidate_value=row[statistic_index],
                    context=context,
                    source=source,
                    parsed_status="PARSED_CSV_WIDE",
                    raw_locator=f"row:{row_number},column:{header[statistic_index]}",
                    raw_text=raw_text,
                )

            if not emitted and STAT_PATTERN.search(" ".join(row)):
                match = STAT_PATTERN.search(" ".join(row))
                sink.unparsed(
                    statistic_name=match.group(0) if match else "csv_candidate",
                    context=context,
                    source=source,
                    raw_locator=f"row:{row_number}",
                    raw_text=raw_text,
                    reason="statistic token is not in a recognized wide or long schema",
                )
    except csv.Error as exc:
        raise FatalError(f"CSV parser failed for {source['source_file']}: {exc}") from exc


def select_json_value(record: Mapping[str, Any]) -> Tuple[bool, Any, str]:
    normalized = {normalize_key(key): key for key in record}
    for alias in VALUE_ALIASES:
        if alias in normalized:
            original_key = normalized[alias]
            return True, record[original_key], str(original_key)
    return False, None, ""


def json_path_component(key: Any) -> str:
    return json.dumps(str(key), ensure_ascii=False)


def walk_json(
    value: Any,
    path: str,
    inherited_context: Mapping[str, Any],
    source: Mapping[str, Any],
    sink: RowSink,
) -> None:
    if isinstance(value, dict):
        scalars = {str(key): item for key, item in value.items() if is_scalar(item)}
        context = dict(inherited_context)
        context.update(scalars)
        normalized_keys = {normalize_key(key): key for key in value}
        emitted_name_keys = set()

        for alias in NAME_ALIASES:
            original_name_key = normalized_keys.get(alias)
            if original_name_key is None:
                continue
            statistic_name = recorded_text(value[original_name_key])
            if not STAT_PATTERN.search(statistic_name):
                continue
            emitted_name_keys.add(original_name_key)
            found_value, candidate_value, value_key = select_json_value(value)
            raw_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if found_value:
                sink.candidate(
                    statistic_name=statistic_name,
                    candidate_value=candidate_value,
                    context=context,
                    source=source,
                    parsed_status="PARSED_JSON_LONG",
                    raw_locator=f"{path}/{json_path_component(value_key)}",
                    raw_text=raw_text,
                )
            else:
                sink.unparsed(
                    statistic_name=statistic_name,
                    context=context,
                    source=source,
                    raw_locator=f"{path}/{json_path_component(original_name_key)}",
                    raw_text=raw_text,
                    reason="long-form statistic record has no recognized value field",
                )

        for key, item in value.items():
            key_text = str(key)
            normalized = normalize_key(key)
            child_path = f"{path}/{json_path_component(key)}"
            if STAT_PATTERN.search(key_text) and normalized not in NAME_ALIASES:
                sink.candidate(
                    statistic_name=key_text,
                    candidate_value=item if is_scalar(item) else None,
                    context=context,
                    source=source,
                    parsed_status="PARSED_JSON_WIDE",
                    raw_locator=child_path,
                    raw_text=json.dumps({key_text: item}, ensure_ascii=False, sort_keys=True),
                )
            elif (
                key not in emitted_name_keys
                and is_scalar(item)
                and isinstance(item, str)
                and STAT_PATTERN.search(item)
                and normalized not in NAME_ALIASES
            ):
                sink.unparsed(
                    statistic_name=STAT_PATTERN.search(item).group(0),
                    context=context,
                    source=source,
                    raw_locator=child_path,
                    raw_text=json.dumps({key_text: item}, ensure_ascii=False),
                    reason="statistic token is not in a recognized wide or long schema",
                )
            if isinstance(item, (dict, list)):
                walk_json(item, child_path, context, source, sink)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(item, (dict, list)):
                walk_json(item, child_path, inherited_context, source, sink)
            elif isinstance(item, str) and STAT_PATTERN.search(item):
                sink.unparsed(
                    statistic_name=STAT_PATTERN.search(item).group(0),
                    context=inherited_context,
                    source=source,
                    raw_locator=child_path,
                    raw_text=item,
                    reason="statistic token appears in an unstructured JSON list",
                )
        return

    if isinstance(value, str) and STAT_PATTERN.search(value):
        sink.unparsed(
            statistic_name=STAT_PATTERN.search(value).group(0),
            context=inherited_context,
            source=source,
            raw_locator=path,
            raw_text=value,
            reason="statistic token appears in an unstructured JSON scalar",
        )


def parse_json(text: str, source: Mapping[str, Any], sink: RowSink) -> None:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FatalError(
            f"JSON parser failed for {source['source_file']} at line {exc.lineno}: {exc.msg}"
        ) from exc
    walk_json(document, "$", {}, source, sink)


def markdown_metadata(line: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    all_aliases = set()
    for aliases in FIELD_ALIASES.values():
        all_aliases.update(aliases)
    alias_pattern = "|".join(sorted((re.escape(alias) for alias in all_aliases), key=len, reverse=True))
    pattern = re.compile(
        rf"\b(?P<key>{alias_pattern})\s*(?:=|:)\s*(?P<value>[^\s,;|)\]]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(line):
        result[match.group("key")] = match.group("value").rstrip(".")
    return result


def parse_markdown(text: str, source: Mapping[str, Any], sink: RowSink) -> None:
    for line_number, line in enumerate(text.splitlines(), 1):
        if not STAT_PATTERN.search(line):
            continue
        assignments = list(MARKDOWN_ASSIGNMENT.finditer(line))
        context = markdown_metadata(line)
        context["markdown_line"] = line
        if not assignments:
            match = STAT_PATTERN.search(line)
            sink.unparsed(
                statistic_name=match.group(0) if match else "markdown_candidate",
                context=context,
                source=source,
                raw_locator=f"line:{line_number}",
                raw_text=line,
                reason="statistic token has no statistic-local key=value expression",
            )
            continue
        for match in assignments:
            sink.candidate(
                statistic_name=match.group("name"),
                candidate_value=match.group("value").rstrip("."),
                context=context,
                source=source,
                parsed_status="PARSED_MD_KEY_VALUE",
                raw_locator=f"line:{line_number},columns:{match.start() + 1}-{match.end()}",
                raw_text=line,
            )


def parse_file(text: str, suffix: str, source: Mapping[str, Any], sink: RowSink) -> str:
    if suffix == ".csv":
        parse_csv(text, source, sink)
        return "csv"
    if suffix == ".json":
        parse_json(text, source, sink)
        return "json"
    if suffix == ".md":
        parse_markdown(text, source, sink)
        return "markdown"
    raise FatalError(f"unsupported input suffix: {suffix}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_mtime(stat_result: os.stat_result) -> str:
    return datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolved_even_if_missing(path: Path) -> Path:
    """Resolve every existing prefix, retaining any not-yet-created suffix."""
    absolute = Path(os.path.abspath(os.path.normpath(os.fspath(path))))
    suffix: List[str] = []
    cursor = absolute
    while not cursor.exists() and not cursor.is_symlink():
        if cursor == cursor.parent:
            break
        suffix.append(cursor.name)
        cursor = cursor.parent
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise FatalError(f"cannot resolve path prefix {cursor}: {exc}") from exc
    for component in reversed(suffix):
        resolved /= component
    return resolved


def reject_output_input_overlap(bundle_root: Path, roots: Sequence[Path]) -> None:
    if not bundle_root.is_absolute():
        raise FatalError("--bundle-root must be an absolute path")
    resolved_bundle = resolved_even_if_missing(bundle_root)
    for root in roots:
        resolved_root = resolved_even_if_missing(root)
        if is_within(resolved_bundle, resolved_root) or is_within(
            resolved_root, resolved_bundle
        ):
            raise FatalError(
                "bundle root and required input root must be disjoint: "
                f"bundle={resolved_bundle}, input={resolved_root}"
            )


def discover_files(
    roots: Sequence[Path],
) -> Tuple[List[Tuple[Path, Path, os.stat_result]], int, Dict[str, int]]:
    files_by_path: Dict[Path, Tuple[Path, Path, os.stat_result]] = {}
    counts: Dict[str, int] = {}
    total_bytes = 0

    for root in roots:
        try:
            root_lstat = root.lstat()
        except OSError as exc:
            raise FatalError(f"required input root cannot be inspected: {root}: {exc}") from exc
        if stat.S_ISLNK(root_lstat.st_mode):
            raise FatalError(f"required input root must not be a symlink: {root}")
        if not stat.S_ISDIR(root_lstat.st_mode):
            raise FatalError(f"required input root is missing or not a directory: {root}")
        resolved_root = root.resolve(strict=True)
        root_count = 0
        root_nonempty_count = 0

        def walk_error(exc: OSError) -> None:
            raise FatalError(f"cannot traverse required input root {root}: {exc}")

        for directory, directory_names, file_names in os.walk(
            resolved_root, followlinks=False, onerror=walk_error
        ):
            directory_path = Path(directory)
            for entry_name in list(directory_names) + list(file_names):
                entry = directory_path / entry_name
                try:
                    entry_stat = entry.lstat()
                except OSError as exc:
                    raise FatalError(f"cannot lstat input entry {entry}: {exc}") from exc
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise FatalError(f"symlink input entry is forbidden: {entry}")

            for file_name in sorted(file_names):
                candidate = directory_path / file_name
                if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                try:
                    candidate_stat = candidate.lstat()
                    resolved_candidate = candidate.resolve(strict=True)
                except OSError as exc:
                    raise FatalError(f"cannot inspect candidate file {candidate}: {exc}") from exc
                if not stat.S_ISREG(candidate_stat.st_mode):
                    raise FatalError(f"supported input path is not a regular file: {candidate}")
                if not is_within(resolved_candidate, resolved_root):
                    raise FatalError(
                        f"candidate resolves outside required input root: {candidate}"
                    )
                root_count += 1
                if candidate_stat.st_size > 0:
                    root_nonempty_count += 1
                if resolved_candidate not in files_by_path:
                    files_by_path[resolved_candidate] = (
                        resolved_root,
                        resolved_candidate,
                        candidate_stat,
                    )
                    total_bytes += candidate_stat.st_size
        if root_count == 0:
            raise FatalError(f"required input root has no supported files: {root}")
        if root_nonempty_count == 0:
            raise FatalError(
                f"required input root has no non-empty supported files: {root}"
            )
        counts[str(resolved_root)] = root_count

    files = [files_by_path[path] for path in sorted(files_by_path, key=str)]
    if not files:
        raise FatalError("no supported input files were discovered")
    return files, total_bytes, counts


def read_input_file(path: Path, source_root: Path, expected: os.stat_result) -> bytes:
    if not is_within(path, source_root):
        raise FatalError(f"input path escaped its required root: {path}")
    current = source_root
    for component in path.relative_to(source_root).parts:
        current = current / component
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise FatalError(f"cannot lstat input path component {current}: {exc}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise FatalError(f"symlink appeared while indexing input: {current}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise FatalError(f"input open failed for {path}: {exc}") from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise FatalError(f"opened input is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read()
    except OSError as exc:
        raise FatalError(f"input read failed for {path}: {exc}") from exc
    finally:
        os.close(descriptor)

    try:
        final_stat = path.lstat()
    except OSError as exc:
        raise FatalError(f"input final lstat failed for {path}: {exc}") from exc
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(expected, field) != getattr(opened_stat, field) for field in identity):
        raise FatalError(f"input changed between discovery and open: {path}")
    if any(getattr(opened_stat, field) != getattr(final_stat, field) for field in identity):
        raise FatalError(f"input changed while being indexed: {path}")
    return data


def make_temp(destination: Path) -> Path:
    parent = destination.parent
    if not parent.is_dir():
        raise FatalError(f"artifact parent directory does not exist: {parent}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        suffix=destination.suffix,
        dir=str(parent),
    )
    os.close(descriptor)
    return Path(name)


def flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def write_audit(
    path: Path, audit_records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> None:
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "summary": summary,
                    "files": list(audit_records),
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            flush_and_sync(handle)
        return
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit_records)
            flush_and_sync(handle)
        return
    raise FatalError("--audit must end in .csv or .json")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_regular_file(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise FatalError(f"required generation artifact is missing: {path}: {exc}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise FatalError(f"generation artifact must be a non-symlink regular file: {path}")


def validate_generation(directory: Path, generation_id: str) -> None:
    if not SAFE_GENERATION_ID.fullmatch(generation_id):
        raise FatalError(f"unsafe generation id: {generation_id}")
    try:
        directory_stat = directory.lstat()
    except OSError as exc:
        raise FatalError(f"generation directory is missing: {directory}: {exc}") from exc
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise FatalError(f"generation must be a non-symlink directory: {directory}")

    manifest_path = directory / MANIFEST_NAME
    done_path = directory / DONE_NAME
    require_regular_file(manifest_path)
    require_regular_file(done_path)
    try:
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
        done_document = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FatalError(f"generation metadata is unreadable: {directory}: {exc}") from exc
    if manifest_document.get("schema_version") != SCHEMA_VERSION:
        raise FatalError(f"generation schema mismatch: {directory}")
    if manifest_document.get("generation_id") != generation_id:
        raise FatalError(f"generation id mismatch: {directory}")
    if manifest_document.get("status") not in {
        "COMPLETE",
        "COMPLETE_ZERO_CANDIDATES",
        "COMPLETE_WITH_UNPARSED_CANDIDATES",
    }:
        raise FatalError(f"generation status is not publishable: {directory}")

    artifacts = manifest_document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FatalError(f"generation artifact manifest is malformed: {directory}")
    output_record = artifacts.get("output")
    audit_record = artifacts.get("audit")
    if not isinstance(output_record, dict) or not isinstance(audit_record, dict):
        raise FatalError(f"generation output/audit records are malformed: {directory}")
    if output_record.get("name") != OUTPUT_NAME:
        raise FatalError(f"generation output name is not canonical: {directory}")
    audit_name = audit_record.get("name")
    if audit_name not in {AUDIT_JSON_NAME, AUDIT_CSV_NAME}:
        raise FatalError(f"generation audit name is not canonical: {directory}")

    output_path = directory / OUTPUT_NAME
    audit_path = directory / str(audit_name)
    require_regular_file(output_path)
    require_regular_file(audit_path)
    if sha256_file(output_path) != output_record.get("sha256"):
        raise FatalError(f"generation output checksum mismatch: {directory}")
    if sha256_file(audit_path) != audit_record.get("sha256"):
        raise FatalError(f"generation audit checksum mismatch: {directory}")
    if sha256_file(manifest_path) != done_document.get("manifest_sha256"):
        raise FatalError(f"generation DONE manifest checksum mismatch: {directory}")
    if done_document.get("generation_id") != generation_id:
        raise FatalError(f"generation DONE id mismatch: {directory}")
    if done_document.get("status") != "READY":
        raise FatalError(f"generation DONE status mismatch: {directory}")

    try:
        with output_path.open(newline="", encoding="utf-8") as handle:
            output_reader = csv.DictReader(handle)
            if output_reader.fieldnames != OUTPUT_FIELDS:
                raise FatalError(f"generation output schema mismatch: {directory}")
            output_row_count = sum(1 for _ in output_reader)
        if audit_path.suffix == ".json":
            audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_records = audit_document.get("files")
            if not isinstance(audit_records, list):
                raise FatalError(f"generation JSON audit is malformed: {directory}")
            audit_field_names = (
                set.intersection(*(set(record) for record in audit_records))
                if audit_records
                and all(isinstance(record, dict) for record in audit_records)
                else set()
            )
        else:
            with audit_path.open(newline="", encoding="utf-8") as handle:
                audit_reader = csv.DictReader(handle)
                audit_field_names = set(audit_reader.fieldnames or [])
                audit_records = list(audit_reader)
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise FatalError(f"generation content validation failed: {directory}: {exc}") from exc
    if not set(AUDIT_FIELDS).issubset(audit_field_names):
        raise FatalError(f"generation audit schema mismatch: {directory}")
    if output_row_count != manifest_document.get("row_count"):
        raise FatalError(f"generation output row count mismatch: {directory}")
    if len(audit_records) != manifest_document.get("files_scanned"):
        raise FatalError(f"generation audit completeness mismatch: {directory}")

    expected_entries = {OUTPUT_NAME, str(audit_name), MANIFEST_NAME, DONE_NAME}
    actual_entries = {entry.name for entry in directory.iterdir()}
    if actual_entries != expected_entries:
        raise FatalError(f"generation contains unexpected or missing artifacts: {directory}")


def prepare_bundle_root(bundle_root: Path) -> Path:
    if not bundle_root.is_absolute():
        raise FatalError("--bundle-root must be an absolute path")
    if not bundle_root.parent.is_dir():
        raise FatalError(f"bundle parent directory does not exist: {bundle_root.parent}")
    if bundle_root.is_symlink():
        raise FatalError(f"bundle root must not be a symlink: {bundle_root}")
    try:
        bundle_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise FatalError(f"cannot create bundle root {bundle_root}: {exc}") from exc
    bundle_stat = bundle_root.lstat()
    if stat.S_ISLNK(bundle_stat.st_mode) or not stat.S_ISDIR(bundle_stat.st_mode):
        raise FatalError(f"bundle root is not a non-symlink directory: {bundle_root}")

    generations = bundle_root / "generations"
    if generations.is_symlink():
        raise FatalError(f"bundle generations path must not be a symlink: {generations}")
    generations.mkdir(mode=0o700, exist_ok=True)
    generations_stat = generations.lstat()
    if stat.S_ISLNK(generations_stat.st_mode) or not stat.S_ISDIR(generations_stat.st_mode):
        raise FatalError(f"bundle generations path is invalid: {generations}")

    for entry in generations.iterdir():
        entry_stat = entry.lstat()
        if stat.S_ISLNK(entry_stat.st_mode):
            raise FatalError(f"symlink generation entry is forbidden: {entry}")
        if entry.name.startswith(".staging-"):
            raise FatalError(f"incomplete staging generation requires adjudication: {entry}")
        if not stat.S_ISDIR(entry_stat.st_mode) or not SAFE_GENERATION_ID.fullmatch(entry.name):
            raise FatalError(f"invalid generation entry: {entry}")
        validate_generation(entry, entry.name)

    current = bundle_root / CURRENT_NAME
    if current.exists() or current.is_symlink():
        require_regular_file(current)
        try:
            current_id = current.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise FatalError(f"CURRENT pointer is unreadable: {current}: {exc}") from exc
        if not SAFE_GENERATION_ID.fullmatch(current_id):
            raise FatalError(f"CURRENT pointer contains an unsafe generation id: {current_id}")
        validate_generation(generations / current_id, current_id)
    return generations


def publish_current_pointer(bundle_root: Path, generation_id: str) -> None:
    if not SAFE_GENERATION_ID.fullmatch(generation_id):
        raise FatalError(f"unsafe generation id: {generation_id}")
    current = bundle_root / CURRENT_NAME
    temporary = make_temp(current)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(generation_id + "\n")
            flush_and_sync(handle)
        # This replacement is deliberately the final failable publication step.
        # A failed replace leaves only an unreferenced, fully validated generation.
        os.replace(temporary, current)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def slurm_execution_context(args: argparse.Namespace) -> Tuple[str, str, int]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "")
    verification = os.environ.get(SLURM_VERIFICATION_ENV, "")
    if slurm_job_id:
        expected = f"v1p3:{slurm_job_id}:{SLURM_JOB_NAME}"
        if slurm_job_name != SLURM_JOB_NAME or verification != expected:
            raise FatalError(
                "Slurm compute budget requires the checksum-verified v1.3 wrapper, "
                f"job name {SLURM_JOB_NAME}, and matching verification marker"
            )
        try:
            python_realpath = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise FatalError(f"cannot resolve running Python interpreter: {exc}") from exc
        expected_python_path = os.environ.get(SLURM_PYTHON_REALPATH_ENV, "")
        expected_python_sha = os.environ.get(SLURM_PYTHON_SHA256_ENV, "")
        if str(python_realpath) != expected_python_path:
            raise FatalError(
                "Slurm compute budget requires the wrapper-fixed Python interpreter; "
                f"running={python_realpath}, expected={expected_python_path or 'MISSING'}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_python_sha):
            raise FatalError("Slurm wrapper Python SHA-256 marker is missing or malformed")
        if sha256_file(python_realpath) != expected_python_sha:
            raise FatalError("running Python interpreter checksum does not match wrapper receipt")
        return "slurm_compute_node", slurm_job_id, args.compute_byte_budget
    if verification:
        raise FatalError("Slurm verification marker is forbidden outside a Slurm job")
    return "login_node", "", args.login_byte_budget


def run(args: argparse.Namespace) -> None:
    roots = [Path(root) for root in args.root]
    bundle_root = Path(args.bundle_root)
    execution_mode, slurm_job_id, byte_budget = slurm_execution_context(args)
    if args.login_byte_budget <= 0 or args.compute_byte_budget <= 0:
        raise FatalError("byte budgets must be positive")

    reject_output_input_overlap(bundle_root, roots)
    files, total_bytes, root_counts = discover_files(roots)
    if total_bytes > byte_budget:
        recommendation = (
            "increase the registered compute budget"
            if execution_mode == "slurm_compute_node"
            else "submit the checksum-bound v1.3 zxc- CPU sbatch wrapper"
        )
        raise FatalError(
            f"input tree is {total_bytes} bytes, above {execution_mode} budget "
            f"{byte_budget}; {recommendation}"
        )

    generations = prepare_bundle_root(bundle_root)
    generation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
        + uuid.uuid4().hex
    )
    staging: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".staging-{generation_id}-", dir=str(generations))
    )
    final_generation = generations / generation_id
    audit_name = AUDIT_JSON_NAME if args.audit_format == "json" else AUDIT_CSV_NAME
    assert staging is not None
    output_path = staging / OUTPUT_NAME
    audit_path = staging / audit_name
    manifest_path = staging / MANIFEST_NAME
    done_path = staging / DONE_NAME

    try:
        audit_records: List[Dict[str, Any]] = []
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            sink = RowSink(writer)

            for source_root, path, initial_stat in files:
                data = read_input_file(path, source_root, initial_stat)
                try:
                    text = data.decode("utf-8-sig", errors="strict")
                except UnicodeDecodeError as exc:
                    raise FatalError(f"UTF-8 decode failed for {path}: {exc}") from exc

                source = {
                    "source_file": str(path),
                    "source_file_sha256": sha256_bytes(data),
                    "mtime": iso_mtime(initial_stat),
                }
                before_total = sink.total
                before_parsed = sink.parsed
                before_unparsed = sink.unparsed_count
                parser = parse_file(text, path.suffix.lower(), source, sink)
                row_count = sink.total - before_total
                parsed_count = sink.parsed - before_parsed
                unparsed_count = sink.unparsed_count - before_unparsed
                if row_count == 0:
                    parser_status = "PARSED_ZERO_HIT"
                elif unparsed_count:
                    parser_status = "PARSED_WITH_UNPARSED_CANDIDATES"
                else:
                    parser_status = "PARSED"
                audit_records.append(
                    {
                        "source_root": str(source_root),
                        "source_file": str(path),
                        "bytes": initial_stat.st_size,
                        "source_file_sha256": source["source_file_sha256"],
                        "mtime": source["mtime"],
                        "parser": parser,
                        "parser_status": parser_status,
                        "row_count": row_count,
                        "parsed_row_count": parsed_count,
                        "unparsed_candidate_count": unparsed_count,
                        "error": (
                            f"{unparsed_count} unparsed candidate(s); adjudication required"
                            if unparsed_count
                            else ""
                        ),
                    }
                )

            flush_and_sync(output_handle)

        status = (
            "COMPLETE_WITH_UNPARSED_CANDIDATES"
            if sink.unparsed_count
            else "COMPLETE_ZERO_CANDIDATES"
            if sink.total == 0
            else "COMPLETE"
        )
        summary = {
            "status": status,
            "files_scanned": len(files),
            "total_bytes": total_bytes,
            "row_count": sink.total,
            "parsed_row_count": sink.parsed,
            "unparsed_candidate_count": sink.unparsed_count,
        }
        write_audit(audit_path, audit_records, summary)
        output_hash = sha256_file(output_path)
        audit_hash = sha256_file(audit_path)
        manifest_document = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "status": status,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_context": {
                "mode": execution_mode,
                "slurm_job_id": slurm_job_id,
                "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
                "checksum_verified_wrapper": execution_mode == "slurm_compute_node",
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": sys.version,
                "python_sha256": sha256_file(Path(sys.executable).resolve()),
            },
            "byte_budget": byte_budget,
            "login_byte_budget": args.login_byte_budget,
            "compute_byte_budget": args.compute_byte_budget,
            "input_roots": [str(root.resolve(strict=True)) for root in roots],
            "root_supported_file_counts": root_counts,
            **summary,
            "output_schema": OUTPUT_FIELDS,
            "audit_schema": AUDIT_FIELDS,
            "audit_format": args.audit_format,
            "fingerprint_policy": "RECORDED_FIELDS_ONLY_UNADJUDICATED",
            "artifacts": {
                "output": {"name": OUTPUT_NAME, "sha256": output_hash},
                "audit": {"name": audit_name, "sha256": audit_hash},
                "manifest": {"name": MANIFEST_NAME},
                "done": {"name": DONE_NAME},
            },
        }
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(manifest_document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            flush_and_sync(handle)
        manifest_hash = sha256_file(manifest_path)
        with done_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": generation_id,
                    "manifest_sha256": manifest_hash,
                    "status": "READY",
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            flush_and_sync(handle)

        fsync_directory(staging)
        validate_generation(staging, generation_id)
        os.rename(staging, final_generation)
        fsync_directory(generations)
        staging = None
        publish_current_pointer(bundle_root, generation_id)
        try:
            sys.stderr.write(
                "FL05_OK "
                f"generation={generation_id} status={status} files={len(files)} "
                f"rows={sink.total} unparsed={sink.unparsed_count} "
                f"output_sha256={output_hash}\n"
            )
        except OSError:
            # CURRENT is already atomically committed; logging cannot roll it back.
            pass
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Required input root; repeat for every registered surface.",
    )
    parser.add_argument(
        "--bundle-root",
        required=True,
        help="Absolute generation-bundle root; only CURRENT is atomically replaced.",
    )
    parser.add_argument(
        "--audit-format",
        choices=("json", "csv"),
        default="json",
        help="Per-file audit representation within each immutable generation.",
    )
    parser.add_argument(
        "--login-byte-budget",
        type=int,
        default=LOGIN_NODE_BYTE_BUDGET,
        help="Maximum total bytes outside Slurm (default: 200 MiB).",
    )
    parser.add_argument(
        "--compute-byte-budget",
        type=int,
        default=COMPUTE_NODE_BYTE_BUDGET,
        help="Maximum total bytes under the verified Slurm wrapper (default: 4 GiB).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run(args)
    except (FatalError, OSError, UnicodeError) as exc:
        sys.stderr.write(f"FL05_FATAL: {exc}\n")
        return 2
    except Exception as exc:  # Unexpected parser defects must fail closed too.
        sys.stderr.write(f"FL05_FATAL: unexpected indexer error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
