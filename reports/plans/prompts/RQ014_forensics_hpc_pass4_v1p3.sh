#!/usr/bin/env bash
# RQ014 v1.3 G0 F05 closure scan. Inputs are opened read-only and never modified.
#
# Every invocation publishes one immutable bundle at:
#   OUTPUT/F05/generations/RUN_ID/{manifest.csv,evidence.txt,status.json,DONE}
# The bundle is renamed into place before OUTPUT/F05/CURRENT is replaced atomically.
set -euo pipefail
export LC_ALL=C
FIXED_SLURM_PYTHON="/share/home/u25310231/.conda/envs/ipv/bin/python"
if [[ -n "${RQ014_G0_F05_VERIFIED_SLURM_WRAPPER:-}" ]]; then
  if [[ "${PYTHON_BIN:-}" != "${FIXED_SLURM_PYTHON}" ]]; then
    echo "configuration error: verified Slurm execution requires fixed PYTHON_BIN=${FIXED_SLURM_PYTHON}" >&2
    exit 2
  fi
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "rq014-g0-closure-v1p3"
SURFACE_ID = "F05"
LOGIN_NODE_BYTE_BUDGET = 200 * 1024 * 1024
COMPUTE_NODE_BYTE_BUDGET = 4 * 1024 * 1024 * 1024
SLURM_JOB_NAME = "zxc-rq014-g0-f05"
SLURM_VERIFICATION_ENV = "RQ014_G0_F05_VERIFIED_SLURM_WRAPPER"
SLURM_PYTHON_REALPATH_ENV = "RQ014_G0_PYTHON_REALPATH"
SLURM_PYTHON_SHA256_ENV = "RQ014_G0_PYTHON_SHA256"
SLURM_PYTHON_VERSION_ENV = "RQ014_G0_PYTHON_VERSION"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PATTERN_TEXT = (
    r"spearman|pearson|correlation|(?:^|[^A-Za-z0-9_])rho(?:[^A-Za-z0-9_]|$)|"
    r"preference[_ -]?score|ipv[ _-]?envelope|rating.{0,80}deviat|deviat.{0,80}rating"
)
PATTERN = re.compile(PATTERN_TEXT.encode("ascii"), re.IGNORECASE)
MANIFEST_FIELDS = [
    "surface_id",
    "root_label",
    "source_path",
    "entry_type",
    "size_bytes",
    "sha256",
    "mtime_utc",
    "read_status",
    "match_count",
]


class PublicationError(RuntimeError):
    """The immutable generation exists, but CURRENT could not be advanced."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_mtime(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{os.getpid()}"


def absolute_normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def is_within_even_if_missing(child: Path, parent: Path) -> bool:
    child_abs = absolute_normalized(child)
    parent_abs = absolute_normalized(parent)
    try:
        return os.path.commonpath((str(child_abs), str(parent_abs))) == str(parent_abs)
    except ValueError:
        return False


def resolved_even_if_missing(path: Path) -> Path:
    absolute = absolute_normalized(path)
    suffix: List[str] = []
    cursor = absolute
    while not cursor.exists() and not cursor.is_symlink():
        if cursor == cursor.parent:
            break
        suffix.append(cursor.name)
        cursor = cursor.parent
    resolved = cursor.resolve(strict=True)
    for component in reversed(suffix):
        resolved /= component
    return resolved


def is_resolved_within_even_if_missing(child: Path, parent: Path) -> bool:
    try:
        resolved_even_if_missing(child).relative_to(resolved_even_if_missing(parent))
        return True
    except (OSError, ValueError):
        return False


def reject_symlink_components(path: Path, role: str) -> None:
    absolute = absolute_normalized(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"{role} contains a symlink path component: {current}")


def is_resolved_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=True).relative_to(parent.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def lstat_or_none(path: Path) -> Optional[os.stat_result]:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def reject_symlink(path: Path, role: str) -> None:
    metadata = lstat_or_none(path)
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise OSError(f"{role} is a symlink and is rejected fail-closed: {path}")


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return digest.hexdigest()


def required_row(label: str, path: Path, status_name: str) -> Dict[str, object]:
    return {
        "surface_id": SURFACE_ID,
        "root_label": label,
        "source_path": str(path),
        "entry_type": "required_input",
        "size_bytes": "",
        "sha256": "",
        "mtime_utc": "",
        "read_status": status_name,
        "match_count": 0,
    }


def scan_file(
    label: str,
    path: Path,
    entry_type: str = "regular_file",
) -> Tuple[Dict[str, object], List[Dict[str, object]], Optional[str]]:
    matches: List[Dict[str, object]] = []
    digest = hashlib.sha256()
    fd: Optional[int] = None
    try:
        reject_symlink_components(path, f"{label} source")
        before_path = os.lstat(path)
        if stat.S_ISLNK(before_path.st_mode):
            raise OSError("symlink entry rejected; O_NOFOLLOW policy")
        if not stat.S_ISREG(before_path.st_mode):
            raise OSError("not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        before_fd = os.fstat(fd)
        if not stat.S_ISREG(before_fd.st_mode):
            raise OSError("opened object is not a regular file")
        if (before_path.st_dev, before_path.st_ino) != (before_fd.st_dev, before_fd.st_ino):
            raise OSError("path identity changed before read")
        byte_offset = 0
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = None
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if PATTERN.search(line):
                    matches.append(
                        {
                            "source_path": str(path),
                            "root_label": label,
                            "line_number": line_number,
                            "byte_offset": byte_offset,
                            "line_bytes": len(line),
                            "content_base64": base64.b64encode(line).decode("ascii"),
                            "content_utf8": line.decode("utf-8", errors="replace"),
                        }
                    )
                byte_offset += len(line)
            after_fd = os.fstat(handle.fileno())
        after_path = os.lstat(path)
        identity = (before_fd.st_dev, before_fd.st_ino)
        if identity != (after_fd.st_dev, after_fd.st_ino):
            raise OSError("open file identity changed while being read")
        if identity != (after_path.st_dev, after_path.st_ino):
            raise OSError("path identity changed while being read")
        if (
            before_fd.st_size != after_fd.st_size
            or before_fd.st_mtime_ns != after_fd.st_mtime_ns
            or byte_offset != before_fd.st_size
        ):
            raise OSError("file changed or was not read fully")
        return (
            {
                "surface_id": SURFACE_ID,
                "root_label": label,
                "source_path": str(path),
                "entry_type": entry_type,
                "size_bytes": before_fd.st_size,
                "sha256": digest.hexdigest(),
                "mtime_utc": utc_mtime(before_fd.st_mtime),
                "read_status": "FULL_READ_OK",
                "match_count": len(matches),
            },
            matches,
            None,
        )
    except (OSError, PermissionError) as exc:
        return (
            required_row(label, path, "READ_ERROR") | {"entry_type": entry_type},
            [],
            f"{label}: {path}: {type(exc).__name__}: {exc}",
        )
    finally:
        if fd is not None:
            os.close(fd)


def walk_required_dir(
    label: str,
    root: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    failures: List[str] = []
    root_before = lstat_or_none(root)
    if root_before is None:
        return [required_row(label, root, "MISSING")], [], [
            f"{label}: required directory is missing: {root}"
        ]
    if stat.S_ISLNK(root_before.st_mode):
        return [required_row(label, root, "SYMLINK_ROOT_REJECTED")], [], [
            f"{label}: required root is a symlink: {root}"
        ]
    if not stat.S_ISDIR(root_before.st_mode):
        return [required_row(label, root, "WRONG_TYPE")], [], [
            f"{label}: required input is not a directory: {root}"
        ]

    def on_walk_error(exc: OSError) -> None:
        failures.append(f"{label}: directory traversal error: {exc}")

    nonempty_full_read_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(
            root, topdown=True, onerror=on_walk_error, followlinks=False
        ):
            current = Path(dirpath)
            current_meta = os.lstat(current)
            if stat.S_ISLNK(current_meta.st_mode) or not stat.S_ISDIR(current_meta.st_mode):
                failures.append(f"{label}: traversed directory identity/type invalid: {current}")
                dirnames[:] = []
                continue
            if not is_within_even_if_missing(current, root):
                failures.append(f"{label}: traversal escaped required root: {current}")
                dirnames[:] = []
                continue
            if not is_resolved_within(current, root):
                failures.append(
                    f"{label}: resolved traversal escaped required root: {current}"
                )
                dirnames[:] = []
                continue
            kept_dirs: List[str] = []
            for name in sorted(dirnames):
                candidate = current / name
                try:
                    metadata = os.lstat(candidate)
                    if stat.S_ISLNK(metadata.st_mode):
                        rows.append(required_row(label, candidate, "SYMLINK_DIRECTORY_REJECTED"))
                        failures.append(f"{label}: symlink directory entry rejected: {candidate}")
                    elif not stat.S_ISDIR(metadata.st_mode):
                        rows.append(required_row(label, candidate, "DIRECTORY_ENTRY_WRONG_TYPE"))
                        failures.append(f"{label}: directory entry changed type: {candidate}")
                    elif not is_resolved_within(candidate, root):
                        rows.append(required_row(label, candidate, "CONTAINMENT_FAILURE"))
                        failures.append(
                            f"{label}: resolved directory entry escaped root: {candidate}"
                        )
                    else:
                        kept_dirs.append(name)
                except OSError as exc:
                    rows.append(required_row(label, candidate, "LSTAT_ERROR"))
                    failures.append(f"{label}: cannot inspect directory entry {candidate}: {exc}")
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                path = current / name
                if not is_resolved_within(path, root):
                    rows.append(required_row(label, path, "CONTAINMENT_FAILURE"))
                    failures.append(f"{label}: resolved file entry escaped root: {path}")
                    continue
                row, file_matches, failure = scan_file(label, path)
                rows.append(row)
                matches.extend(file_matches)
                if failure:
                    failures.append(failure)
                elif int(row["size_bytes"]) > 0:
                    nonempty_full_read_count += 1
    except (OSError, PermissionError) as exc:
        failures.append(f"{label}: traversal aborted: {type(exc).__name__}: {exc}")
    root_after = lstat_or_none(root)
    if (
        root_after is None
        or stat.S_ISLNK(root_after.st_mode)
        or (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino)
    ):
        failures.append(f"{label}: required root identity changed during scan: {root}")
    if nonempty_full_read_count == 0:
        failures.append(
            f"{label}: closure requires at least one non-empty regular file read fully"
        )
    return rows, matches, failures


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    default_results = Path("/share/home/u25310231/ZXC/RQ010B_wod_e2e/results")
    parser = argparse.ArgumentParser(
        description="Fail-closed, complete F05 forensic closure scan (read-only)."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument("--pilot-dir", action="append", default=[], metavar="ID=DIR")
    parser.add_argument(
        "--phase3-root",
        type=Path,
        default=Path(
            "/share/home/u25310231/ZXC/RQ010B_wod_e2e/"
            "reframed_pref_analysis/phase3_preference_test"
        ),
    )
    parser.add_argument(
        "--job-log",
        type=Path,
        default=Path(
            "/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/"
            "zxc-rq010b-ipv-rating_1746009.out"
        ),
    )
    return parser.parse_args(argv)


def pilot_inputs(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    names = [
        ("FL01", "rq010b_wod_e2e_ipv_rating_pilot_20260629"),
        ("FL02", "rq010b_wod_e2e_ipv_rating_pilot_dtfix_20260629T123954"),
        ("FL03", "rq010b_wod_e2e_ipv_rating_pilot_fixed_20260629T124417"),
        ("FL04", "rq010b_wod_e2e_ipv_rating_pilot_routefix_20260629T124941"),
    ]
    if not args.pilot_dir:
        return [(item_id, args.results_root / name) for item_id, name in names]
    if len(args.pilot_dir) != 4:
        raise ValueError("--pilot-dir must be repeated exactly four times")
    parsed: List[Tuple[str, Path]] = []
    seen = set()
    for value in args.pilot_dir:
        if "=" not in value:
            raise ValueError(f"invalid --pilot-dir {value!r}; expected ID=DIR")
        item_id, path_text = value.split("=", 1)
        item_id = item_id.strip()
        if not item_id or not path_text or item_id in seen:
            raise ValueError(f"invalid or duplicate --pilot-dir {value!r}")
        seen.add(item_id)
        parsed.append((item_id, Path(path_text)))
    if seen != {"FL01", "FL02", "FL03", "FL04"}:
        raise ValueError("--pilot-dir IDs must be exactly FL01, FL02, FL03, and FL04")
    return parsed


def execution_context() -> Tuple[str, int, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    marker = os.environ.get(SLURM_VERIFICATION_ENV, "")
    if job_id:
        expected = f"v1p3:{job_id}:{SLURM_JOB_NAME}"
        if job_name != SLURM_JOB_NAME or marker != expected:
            raise OSError(
                "compute-node byte budget requires the checksum-bound zxc-rq014-g0-f05 wrapper"
            )
        try:
            python_realpath = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise OSError(f"cannot resolve running Python interpreter: {exc}") from exc
        python_sha256 = sha256_file(python_realpath)
        python_version = ".".join(str(value) for value in sys.version_info[:3])
        expected_realpath = os.environ.get(SLURM_PYTHON_REALPATH_ENV, "")
        expected_sha256 = os.environ.get(SLURM_PYTHON_SHA256_ENV, "")
        expected_version = os.environ.get(SLURM_PYTHON_VERSION_ENV, "")
        if str(python_realpath) != expected_realpath:
            raise OSError(
                "running Python realpath differs from the checksum-bound wrapper receipt"
            )
        if python_sha256 != expected_sha256:
            raise OSError(
                "running Python SHA-256 differs from the checksum-bound wrapper receipt"
            )
        if python_version != expected_version:
            raise OSError(
                "running Python version differs from the checksum-bound wrapper receipt"
            )
        return "slurm_compute_node", COMPUTE_NODE_BYTE_BUDGET, job_id
    if marker:
        raise OSError("Slurm wrapper marker is forbidden outside a Slurm allocation")
    return "login_node", LOGIN_NODE_BYTE_BUDGET, ""


def preflight_total_bytes(inputs: Sequence[Tuple[str, Path, str]]) -> int:
    total = 0
    for _label, path, kind in inputs:
        try:
            metadata = os.lstat(path)
        except OSError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            continue
        if kind == "file":
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(
                path, topdown=True, followlinks=False
            ):
                current = Path(dirpath)
                for name in filenames:
                    try:
                        entry_metadata = os.lstat(current / name)
                    except OSError:
                        continue
                    if stat.S_ISREG(entry_metadata.st_mode):
                        total += entry_metadata.st_size
        except OSError:
            continue
    return total


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    def writer(handle) -> None:
        output = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        output.writeheader()
        output.writerows(rows)

    atomic_write(path, writer)


def write_evidence(path: Path, matches: Sequence[Dict[str, object]]) -> None:
    def writer(handle) -> None:
        handle.write(f"schema_version: {SCHEMA_VERSION}\n")
        handle.write(f"surface_id: {SURFACE_ID}\n")
        handle.write("search_mode: FULL_BYTE_STREAM_NO_SAMPLING\n")
        handle.write("scan_transport: STREAMING_FILE_DESCRIPTOR_READ_TO_EOF\n")
        handle.write(f"pattern: {PATTERN_TEXT}\n")
        handle.write(f"match_count: {len(matches)}\n")
        handle.write(
            "result: CANDIDATE_MATCHES_FOUND\n"
            if matches
            else "result: NO_MATCHES_AFTER_FULL_BYTE_STREAM\n"
        )
        for index, match in enumerate(matches, start=1):
            handle.write(f"\n--- match {index} ---\n")
            for key in (
                "root_label",
                "source_path",
                "line_number",
                "byte_offset",
                "line_bytes",
                "content_base64",
            ):
                handle.write(f"{key}: {match[key]}\n")
            handle.write("content_utf8_begin\n")
            handle.write(str(match["content_utf8"]))
            if not str(match["content_utf8"]).endswith("\n"):
                handle.write("\n")
            handle.write("content_utf8_end\n")

    atomic_write(path, writer)


def validate_generation(directory: Path, run_id: str) -> None:
    expected_names = {"manifest.csv", "evidence.txt", "status.json", "DONE"}
    actual_names = {entry.name for entry in directory.iterdir()}
    if actual_names != expected_names:
        raise OSError(
            f"generation file set mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    for name in expected_names:
        metadata = os.lstat(directory / name)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"generation artifact must be a regular non-symlink file: {name}")

    manifest_path = directory / "manifest.csv"
    evidence_path = directory / "evidence.txt"
    status_path = directory / "status.json"
    done_path = directory / "DONE"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise OSError("generation manifest schema mismatch")
        list(reader)
    evidence = evidence_path.read_text(encoding="utf-8")
    if not evidence.startswith(f"schema_version: {SCHEMA_VERSION}\n"):
        raise OSError("generation evidence schema header mismatch")
    status_doc = json.loads(status_path.read_text(encoding="utf-8"))
    done_doc = json.loads(done_path.read_text(encoding="utf-8"))
    if status_doc.get("schema_version") != SCHEMA_VERSION:
        raise OSError("generation status schema mismatch")
    if status_doc.get("surface_id") != SURFACE_ID or status_doc.get("generation_id") != run_id:
        raise OSError("generation status identity mismatch")
    if status_doc.get("manifest_sha256") != sha256_file(manifest_path):
        raise OSError("generation manifest checksum mismatch")
    if status_doc.get("evidence_sha256") != sha256_file(evidence_path):
        raise OSError("generation evidence checksum mismatch")
    if done_doc.get("schema_version") != SCHEMA_VERSION:
        raise OSError("generation DONE schema mismatch")
    if done_doc.get("surface_id") != SURFACE_ID or done_doc.get("generation_id") != run_id:
        raise OSError("generation DONE identity mismatch")
    if done_doc.get("manifest_sha256") != sha256_file(manifest_path):
        raise OSError("generation DONE manifest checksum mismatch")
    if done_doc.get("evidence_sha256") != sha256_file(evidence_path):
        raise OSError("generation DONE evidence checksum mismatch")
    if done_doc.get("status_sha256") != sha256_file(status_path):
        raise OSError("generation DONE status checksum mismatch")
    if done_doc.get("bundle_complete") is not True:
        raise OSError("generation DONE completeness flag mismatch")


def update_current(surface_root: Path, run_id: str) -> None:
    current = surface_root / "CURRENT"
    reject_symlink(current, "CURRENT pointer")
    fd, tmp_name = tempfile.mkstemp(prefix=".CURRENT.", dir=str(surface_root))
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            handle.write(run_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.environ.get("RQ014_TEST_FAIL_CURRENT_SURFACE") == SURFACE_ID:
            raise OSError("injected CURRENT replacement failure")
        # CURRENT replacement is the commit point and the final failable step.
        os.replace(tmp_name, current)
        try:
            fsync_directory(surface_root)
        except OSError:
            # The pointer is already committed; a post-commit durability hint
            # cannot truthfully restore the previous pointer or fail the run.
            pass
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def publish_bundle(
    output_root: Path,
    run_id: str,
    inputs: Sequence[Tuple[str, Path, str]],
    rows: Sequence[Dict[str, object]],
    matches: Sequence[Dict[str, object]],
    failures: Sequence[str],
    execution_metadata: Dict[str, object],
) -> Dict[str, object]:
    reject_symlink(output_root, "output root")
    output_root.mkdir(parents=True, exist_ok=True)
    surface_root = output_root / SURFACE_ID
    reject_symlink(surface_root, "surface output root")
    surface_root.mkdir(exist_ok=True)
    generations = surface_root / "generations"
    reject_symlink(generations, "generations directory")
    generations.mkdir(exist_ok=True)
    final_generation = generations / run_id
    staging = generations / f".staging-{run_id}-{os.getpid()}"
    if final_generation.exists() or staging.exists():
        raise OSError(f"immutable generation already exists: {final_generation}")
    staging.mkdir()
    try:
        manifest_path = staging / "manifest.csv"
        evidence_path = staging / "evidence.txt"
        status_path = staging / "status.json"
        done_path = staging / "DONE"
        write_manifest(manifest_path, rows)
        write_evidence(evidence_path, matches)
        state = (
            "INACCESSIBLE"
            if failures
            else "FOUND"
            if matches
            else "NOT_FOUND_ON_SCANNED_SURFACES"
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "surface_id": SURFACE_ID,
            "generation_id": run_id,
            "state": state,
            "complete_scan": not failures,
            "search_mode": "FULL_BYTE_STREAM_NO_SAMPLING",
            "scan_transport": "STREAMING_FILE_DESCRIPTOR_READ_TO_EOF",
            "sampling_used": False,
            "required_inputs": [
                {"label": label, "path": str(path), "kind": kind}
                for label, path, kind in inputs
            ],
            "read_failures": list(failures),
            "error": "; ".join(failures) if failures else None,
            "residual_risk_statement": (
                "F05 closure is not established because one or more required inputs "
                "were missing, empty, symlinked, changed, or not read fully."
                if failures
                else "No known residual input-access risk remains on the declared F05 surfaces."
            ),
            "scanned_file_count": sum(row["read_status"] == "FULL_READ_OK" for row in rows),
            "scanned_bytes": sum(
                int(row["size_bytes"])
                for row in rows
                if row["read_status"] == "FULL_READ_OK"
            ),
            "matched_record_count": len(matches),
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "evidence_file": evidence_path.name,
            "evidence_sha256": sha256_file(evidence_path),
            "generated_at_utc": utc_now(),
            **execution_metadata,
        }
        atomic_write(
            status_path,
            lambda handle: json.dump(record, handle, indent=2, sort_keys=True)
            or handle.write("\n"),
        )
        done_record = {
            "schema_version": SCHEMA_VERSION,
            "surface_id": SURFACE_ID,
            "generation_id": run_id,
            "manifest_sha256": record["manifest_sha256"],
            "evidence_sha256": record["evidence_sha256"],
            "status_sha256": sha256_file(status_path),
            "bundle_complete": True,
        }
        atomic_write(
            done_path,
            lambda handle: json.dump(done_record, handle, indent=2, sort_keys=True)
            or handle.write("\n"),
        )
        validate_generation(staging, run_id)
        fsync_directory(staging)
        os.rename(staging, final_generation)
        fsync_directory(generations)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    try:
        update_current(surface_root, run_id)
    except OSError as exc:
        raise PublicationError(
            f"generation {final_generation} is durable but CURRENT was not changed: {exc}"
        ) from exc
    return record


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if not RUN_ID_RE.fullmatch(args.run_id):
        print("configuration error: invalid --run-id", file=sys.stderr)
        return 2
    try:
        pilots = pilot_inputs(args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    inputs: List[Tuple[str, Path, str]] = [
        (item_id, path, "directory") for item_id, path in pilots
    ]
    inputs.extend(
        [("PHASE3", args.phase3_root, "directory"), ("JOB_LOG", args.job_log, "file")]
    )
    if any(
        is_within_even_if_missing(args.output_dir, path)
        or is_resolved_within_even_if_missing(args.output_dir, path)
        for _, path, _ in inputs
    ):
        print(
            "configuration error: --output-dir must be outside every scanned input, "
            "including a declared input that does not yet exist",
            file=sys.stderr,
        )
        return 2
    try:
        mode, byte_budget, slurm_job_id = execution_context()
    except OSError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    preflight_bytes = preflight_total_bytes(inputs)
    if preflight_bytes > byte_budget:
        recommendation = (
            "use the checksum-bound zxc-rq014-g0-f05 sbatch wrapper"
            if mode == "login_node"
            else "registered compute-node byte budget is insufficient"
        )
        print(
            f"configuration error: F05 inputs total {preflight_bytes} bytes, above "
            f"{mode} budget {byte_budget}; {recommendation}",
            file=sys.stderr,
        )
        return 2

    rows: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    failures: List[str] = []
    for label, path, kind in inputs:
        if kind == "directory":
            new_rows, new_matches, new_failures = walk_required_dir(label, path)
            rows.extend(new_rows)
            matches.extend(new_matches)
            failures.extend(new_failures)
            continue
        metadata = lstat_or_none(path)
        if metadata is None:
            rows.append(required_row(label, path, "MISSING"))
            failures.append(f"{label}: required file is missing: {path}")
        elif stat.S_ISLNK(metadata.st_mode):
            rows.append(required_row(label, path, "SYMLINK_ROOT_REJECTED"))
            failures.append(f"{label}: required file is a symlink: {path}")
        elif not stat.S_ISREG(metadata.st_mode):
            rows.append(required_row(label, path, "WRONG_TYPE"))
            failures.append(f"{label}: required input is not a regular file: {path}")
        else:
            row, file_matches, failure = scan_file(label, path)
            rows.append(row)
            matches.extend(file_matches)
            if failure:
                failures.append(failure)
            elif int(row["size_bytes"]) == 0:
                failures.append("JOB_LOG: required job log is empty")
    try:
        record = publish_bundle(
            args.output_dir,
            args.run_id,
            inputs,
            rows,
            matches,
            failures,
            {
                "execution_mode": mode,
                "byte_budget": byte_budget,
                "preflight_total_bytes": preflight_bytes,
                "slurm_job_id": slurm_job_id,
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": sys.version,
                "python_semantic_version": ".".join(
                    str(value) for value in sys.version_info[:3]
                ),
                "python_sha256": sha256_file(Path(sys.executable).resolve()),
                "wrapper_python_realpath_receipt": os.environ.get(
                    SLURM_PYTHON_REALPATH_ENV, ""
                ),
                "wrapper_python_sha256_receipt": os.environ.get(
                    SLURM_PYTHON_SHA256_ENV, ""
                ),
                "wrapper_python_version_receipt": os.environ.get(
                    SLURM_PYTHON_VERSION_ENV, ""
                ),
            },
        )
    except PublicationError as exc:
        try:
            print(f"ERROR: {exc}", file=sys.stderr)
        except OSError:
            pass
        return 4
    except OSError as exc:
        try:
            print(f"ERROR: bundle publication failed: {exc}", file=sys.stderr)
        except OSError:
            pass
        return 4
    try:
        print(json.dumps(record, sort_keys=True))
    except OSError:
        pass
    if failures:
        for failure in failures:
            try:
                print(f"ERROR: {failure}", file=sys.stderr)
            except OSError:
                pass
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
