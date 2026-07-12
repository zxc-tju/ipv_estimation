#!/usr/bin/env bash
# RQ014 v1.3 G0 F06/F07/F08 closure scan for macOS. Inputs are read-only.
#
# F07 and F08 accept only pre-study frozen scope manifests. Each surface is
# published as generations/RUN_ID/{manifest.csv,evidence.txt,status.json,DONE},
# then made current by one atomic CURRENT text-pointer replacement.
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RQ014_PASS4_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "rq014-g0-closure-v1p3"
SCOPE_SCHEMA_VERSION = "rq014-frozen-scope-v1p3"
SCOPE_CUTOFF = datetime(2026, 7, 10, tzinfo=timezone.utc)
SCOPE_CUTOFF_TEXT = "2026-07-10T00:00:00Z"
# No production RFC 3161 trust profile or genuine pre-cutoff whole-inventory
# receipt is registered in v1.3. Git metadata can bind content bytes but its
# timestamps are user-controlled. F07/F08 therefore remain deliberately
# INACCESSIBLE until a later checksum-bound amendment registers and validates
# a real receipt that predates the fixed cutoff.
SCOPE_TIME_WITNESS_STATE = "INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT"
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
    """A generation is durable, but its CURRENT pointer was not changed."""


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


def is_resolved_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=True).relative_to(parent.resolve(strict=True))
        return True
    except (OSError, ValueError):
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


def required_row(
    surface_id: str, label: str, path: str, status_name: str
) -> Dict[str, object]:
    return {
        "surface_id": surface_id,
        "root_label": label,
        "source_path": path,
        "entry_type": "required_input",
        "size_bytes": "",
        "sha256": "",
        "mtime_utc": "",
        "read_status": status_name,
        "match_count": 0,
    }


def scan_file(
    surface_id: str,
    label: str,
    path: Path,
    entry_type: str = "regular_file",
) -> Tuple[Dict[str, object], List[Dict[str, object]], Optional[str]]:
    digest = hashlib.sha256()
    matches: List[Dict[str, object]] = []
    fd: Optional[int] = None
    try:
        reject_symlink_components(path, f"{surface_id} source")
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
            raise OSError("open file identity changed during read")
        if identity != (after_path.st_dev, after_path.st_ino):
            raise OSError("path identity changed during read")
        if (
            before_fd.st_size != after_fd.st_size
            or before_fd.st_mtime_ns != after_fd.st_mtime_ns
            or byte_offset != before_fd.st_size
        ):
            raise OSError("file changed or was not read fully")
        return (
            {
                "surface_id": surface_id,
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
            required_row(surface_id, label, str(path), "READ_ERROR")
            | {"entry_type": entry_type},
            [],
            f"{label}: {path}: {type(exc).__name__}: {exc}",
        )
    finally:
        if fd is not None:
            os.close(fd)


def walk_required_dir(
    surface_id: str,
    label: str,
    root: Path,
    excluded_dir_names: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    failures: List[str] = []
    excluded = excluded_dir_names or set()
    root_before = lstat_or_none(root)
    if root_before is None:
        return [required_row(surface_id, label, str(root), "MISSING")], [], [
            f"{label}: required directory is missing: {root}"
        ]
    if stat.S_ISLNK(root_before.st_mode):
        return [required_row(surface_id, label, str(root), "SYMLINK_ROOT_REJECTED")], [], [
            f"{label}: required root is a symlink: {root}"
        ]
    if not stat.S_ISDIR(root_before.st_mode):
        return [required_row(surface_id, label, str(root), "WRONG_TYPE")], [], [
            f"{label}: required input is not a directory: {root}"
        ]

    def on_walk_error(exc: OSError) -> None:
        failures.append(f"{label}: directory traversal error: {exc}")

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
                        rows.append(
                            required_row(
                                surface_id, label, str(candidate), "SYMLINK_DIRECTORY_REJECTED"
                            )
                        )
                        failures.append(f"{label}: symlink directory entry rejected: {candidate}")
                    elif not stat.S_ISDIR(metadata.st_mode):
                        rows.append(
                            required_row(
                                surface_id, label, str(candidate), "DIRECTORY_ENTRY_WRONG_TYPE"
                            )
                        )
                        failures.append(f"{label}: directory entry changed type: {candidate}")
                    elif not is_resolved_within(candidate, root):
                        rows.append(
                            required_row(
                                surface_id, label, str(candidate), "CONTAINMENT_FAILURE"
                            )
                        )
                        failures.append(
                            f"{label}: resolved directory entry escaped root: {candidate}"
                        )
                    elif name not in excluded:
                        kept_dirs.append(name)
                except OSError as exc:
                    rows.append(required_row(surface_id, label, str(candidate), "LSTAT_ERROR"))
                    failures.append(f"{label}: cannot inspect directory entry {candidate}: {exc}")
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                path = current / name
                if not is_resolved_within(path, root):
                    rows.append(
                        required_row(surface_id, label, str(path), "CONTAINMENT_FAILURE")
                    )
                    failures.append(f"{label}: resolved file entry escaped root: {path}")
                    continue
                row, file_matches, failure = scan_file(surface_id, label, path)
                rows.append(row)
                matches.extend(file_matches)
                if failure:
                    failures.append(failure)
    except (OSError, PermissionError) as exc:
        failures.append(f"{label}: traversal aborted: {type(exc).__name__}: {exc}")
    root_after = lstat_or_none(root)
    if (
        root_after is None
        or stat.S_ISLNK(root_after.st_mode)
        or (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino)
    ):
        failures.append(f"{label}: required root identity changed during scan: {root}")
    return rows, matches, failures


def git_run(repo: Path, args: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"git {' '.join(args)} failed ({completed.returncode}): {message}")
    return completed.stdout


def stream_git_blob(
    repo: Path, oid: str, label: str
) -> Tuple[int, str, List[Dict[str, object]], bool]:
    declared_size = int(git_run(repo, ["cat-file", "-s", oid]).strip())
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "blob", oid],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise OSError(f"could not open streaming stdout for Git blob {oid}")
    digest = hashlib.sha256()
    matches: List[Dict[str, object]] = []
    byte_offset = 0
    first_line: Optional[bytes] = None
    for line_number, line in enumerate(process.stdout, start=1):
        if first_line is None:
            first_line = line
        digest.update(line)
        if PATTERN.search(line):
            matches.append(
                {
                    "source_path": oid,
                    "root_label": label,
                    "line_number": line_number,
                    "byte_offset": byte_offset,
                    "line_bytes": len(line),
                    "content_base64": base64.b64encode(line).decode("ascii"),
                    "content_utf8": line.decode("utf-8", errors="replace"),
                }
            )
        byte_offset += len(line)
    stderr = process.stderr.read() if process.stderr is not None else b""
    returncode = process.wait()
    if returncode != 0:
        raise OSError(
            f"streaming Git blob {oid} failed ({returncode}): "
            f"{stderr.decode('utf-8', errors='replace')}"
        )
    if byte_offset != declared_size:
        raise OSError(f"Git blob {oid} was not read fully")
    unresolved_lfs = first_line == b"version https://git-lfs.github.com/spec/v1\n"
    return declared_size, digest.hexdigest(), matches, unresolved_lfs


def scan_git_history(
    surface_id: str, label: str, repo: Path
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    failures: List[str] = []
    repo_meta = lstat_or_none(repo)
    if repo_meta is None or not stat.S_ISDIR(repo_meta.st_mode):
        return [required_row(surface_id, label, str(repo), "MISSING_OR_WRONG_TYPE")], [], [
            f"{label}: required paper repository is unavailable: {repo}"
        ]
    if stat.S_ISLNK(repo_meta.st_mode):
        return [required_row(surface_id, label, str(repo), "SYMLINK_ROOT_REJECTED")], [], [
            f"{label}: required paper repository root is a symlink: {repo}"
        ]
    try:
        if git_run(repo, ["rev-parse", "--is-inside-work-tree"]).strip() != b"true":
            raise OSError("path is not a Git work tree")
        raw_commits = git_run(repo, ["log", "--all", "--format=%H%x09%cI"])
        commit_rows = [line.split(b"\t", 1) for line in raw_commits.splitlines() if line]
        if not commit_rows:
            raise OSError("repository has no reachable commits")
        blob_cache: Dict[str, Tuple[int, str, List[Dict[str, object]], bool]] = {}
        for commit_parts in commit_rows:
            if len(commit_parts) != 2:
                raise OSError("could not parse commit timestamp manifest")
            commit = commit_parts[0].decode("ascii")
            commit_time = commit_parts[1].decode("ascii")
            tree = git_run(repo, ["ls-tree", "-r", "-z", "-l", commit])
            for item in tree.split(b"\0"):
                if not item:
                    continue
                metadata, separator, raw_path = item.partition(b"\t")
                fields = metadata.split()
                if not separator or len(fields) != 4:
                    raise OSError(f"could not parse ls-tree record in {commit}")
                mode, object_type, raw_oid, _raw_size = fields
                source_file = raw_path.decode("utf-8", errors="replace")
                source_path = f"git:{repo}@{commit}:{source_file}"
                if mode == b"120000":
                    rows.append(
                        required_row(surface_id, label, source_path, "GIT_SYMLINK_REJECTED")
                    )
                    failures.append(f"{label}: Git symlink entry rejected: {source_path}")
                    continue
                if object_type != b"blob":
                    rows.append(
                        required_row(
                            surface_id, label, source_path, "UNSCANNED_NON_BLOB_GIT_ENTRY"
                        )
                    )
                    failures.append(f"{label}: non-blob Git entry is inaccessible: {source_path}")
                    continue
                oid = raw_oid.decode("ascii")
                if oid not in blob_cache:
                    blob_cache[oid] = stream_git_blob(repo, oid, label)
                size_bytes, digest, blob_matches, unresolved_lfs = blob_cache[oid]
                occurrence_matches: List[Dict[str, object]] = []
                for blob_match in blob_matches:
                    occurrence = dict(blob_match)
                    occurrence["source_path"] = source_path
                    occurrence_matches.append(occurrence)
                if unresolved_lfs:
                    failures.append(
                        f"{label}: Git LFS pointer is not archived object content: {source_path}"
                    )
                rows.append(
                    {
                        "surface_id": surface_id,
                        "root_label": label,
                        "source_path": source_path,
                        "entry_type": "git_blob_snapshot",
                        "size_bytes": size_bytes,
                        "sha256": digest,
                        "mtime_utc": commit_time,
                        "read_status": (
                            "GIT_LFS_POINTER_UNRESOLVED" if unresolved_lfs else "FULL_READ_OK"
                        ),
                        "match_count": len(occurrence_matches),
                    }
                )
                matches.extend(occurrence_matches)
    except (OSError, ValueError) as exc:
        rows.append(required_row(surface_id, label, str(repo), "GIT_READ_ERROR"))
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return rows, matches, failures


def parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("snapshot_at_utc must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot_at_utc must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def read_regular_bytes(path: Path, maximum_bytes: int) -> Tuple[bytes, str]:
    reject_symlink_components(path, "scope manifest")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OSError("scope manifest must be a non-symlink regular file")
    if metadata.st_size > maximum_bytes:
        raise OSError(f"scope manifest exceeds {maximum_bytes} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("scope manifest identity changed before read")
        digest = hashlib.sha256()
        chunks: List[bytes] = []
        total = 0
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise OSError(f"scope manifest exceeds {maximum_bytes} bytes")
                digest.update(chunk)
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
        if (
            (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
            or opened.st_size != after.st_size
            or opened.st_mtime_ns != after.st_mtime_ns
            or total != opened.st_size
        ):
            raise OSError("scope manifest changed or was not read fully")
        return b"".join(chunks), digest.hexdigest()
    finally:
        if fd >= 0:
            os.close(fd)


def load_scope_manifest(
    path: Path, expected_surface: str
) -> Tuple[List[Dict[str, str]], str, List[str]]:
    failures: List[str] = []
    try:
        raw, manifest_sha256 = read_regular_bytes(path, 10 * 1024 * 1024)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("scope manifest root must be an object")
        if payload.get("schema_version") != SCOPE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCOPE_SCHEMA_VERSION}")
        if payload.get("surface_id") != expected_surface:
            raise ValueError(f"surface_id must be {expected_surface}")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("entries must be a non-empty list")
        normalized: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"entry {index} must be an object")
            source_path = entry.get("source_path")
            source_sha256 = entry.get("source_sha256")
            snapshot_at_utc = entry.get("snapshot_at_utc")
            snapshot_evidence_kind = entry.get("snapshot_evidence_kind")
            snapshot_git_repo = entry.get("snapshot_git_repo")
            snapshot_git_commit = entry.get("snapshot_git_commit")
            snapshot_git_path = entry.get("snapshot_git_path")
            if not isinstance(source_path, str) or not Path(source_path).is_absolute():
                raise ValueError(f"entry {index} source_path must be absolute")
            canonical_path = str(absolute_normalized(Path(source_path)))
            if canonical_path in seen:
                raise ValueError(f"duplicate source_path: {canonical_path}")
            seen.add(canonical_path)
            if not isinstance(source_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", source_sha256
            ):
                raise ValueError(f"entry {index} source_sha256 must be lowercase SHA-256")
            if snapshot_evidence_kind != "git_blob_v1_content_integrity_only":
                raise ValueError(
                    "entry {index} snapshot_evidence_kind must be "
                    "git_blob_v1_content_integrity_only".format(index=index)
                )
            if not isinstance(snapshot_git_repo, str) or not Path(
                snapshot_git_repo
            ).is_absolute():
                raise ValueError(f"entry {index} snapshot_git_repo must be absolute")
            if not isinstance(snapshot_git_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", snapshot_git_commit
            ):
                raise ValueError(f"entry {index} snapshot_git_commit must be a full object id")
            if not isinstance(snapshot_git_path, str):
                raise ValueError(f"entry {index} snapshot_git_path must be a string")
            git_path = PurePosixPath(snapshot_git_path)
            if (
                git_path.is_absolute()
                or not git_path.parts
                or ".." in git_path.parts
                or ":" in snapshot_git_path
                or "\x00" in snapshot_git_path
            ):
                raise ValueError(f"entry {index} snapshot_git_path is unsafe")
            snapshot = parse_timestamp(snapshot_at_utc)
            if not snapshot < SCOPE_CUTOFF:
                raise ValueError(
                    f"entry {index} descriptive Git timestamp must be strictly before "
                    f"{SCOPE_CUTOFF_TEXT}; it is not trusted cutoff evidence"
                )
            normalized.append(
                {
                    "source_path": canonical_path,
                    "source_sha256": source_sha256,
                    "snapshot_at_utc": str(snapshot_at_utc),
                    "snapshot_evidence_kind": snapshot_evidence_kind,
                    "snapshot_git_repo": str(absolute_normalized(Path(snapshot_git_repo))),
                    "snapshot_git_commit": snapshot_git_commit,
                    "snapshot_git_path": snapshot_git_path,
                }
            )
        return normalized, manifest_sha256, []
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        failures.append(
            f"{expected_surface}: frozen scope manifest invalid: {path}: "
            f"{type(exc).__name__}: {exc}"
        )
        return [], "", failures


def verify_scope_git_anchor(entry: Dict[str, str]) -> None:
    repo = Path(entry["snapshot_git_repo"])
    reject_symlink_components(repo, "snapshot Git repository")
    metadata = os.lstat(repo)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"snapshot Git repository is not a directory: {repo}")
    commit = entry["snapshot_git_commit"]
    resolved_commit = git_run(repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"]).strip()
    if resolved_commit.decode("ascii") != commit:
        raise OSError(f"snapshot Git commit does not resolve exactly: {commit}")
    commit_time_text = git_run(repo, ["show", "-s", "--format=%cI", commit]).decode(
        "ascii"
    ).strip()
    commit_time = parse_timestamp(commit_time_text)
    declared_time = parse_timestamp(entry["snapshot_at_utc"])
    if commit_time != declared_time:
        raise OSError(
            "snapshot_at_utc does not equal the anchored Git committer timestamp: "
            f"declared={entry['snapshot_at_utc']}, git={commit_time_text}"
        )
    if not commit_time < SCOPE_CUTOFF:
        raise OSError(f"anchored Git commit is not strictly before {SCOPE_CUTOFF_TEXT}")
    object_spec = f"{commit}:{entry['snapshot_git_path']}"
    object_type = git_run(repo, ["cat-file", "-t", object_spec]).strip()
    if object_type != b"blob":
        raise OSError(f"anchored Git object is not a blob: {object_spec}")
    blob = git_run(repo, ["cat-file", "blob", object_spec])
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if blob_sha256 != entry["source_sha256"]:
        raise OSError(
            "anchored Git blob SHA-256 differs from scope source SHA-256: "
            f"{object_spec}"
        )


def scan_frozen_scope(
    surface_id: str,
    scope_manifest: Path,
    entries: Sequence[Dict[str, str]],
    initial_failures: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    matches: List[Dict[str, object]] = []
    failures = list(initial_failures)
    if failures:
        rows.append(
            required_row(surface_id, "FROZEN_SCOPE_MANIFEST", str(scope_manifest), "INVALID")
        )
        return rows, matches, failures
    failures.append(
        f"{surface_id}: {SCOPE_TIME_WITNESS_STATE}: v1.3 has no checksum-bound "
        "production TSA profile and no genuine RFC3161 receipt over the exact "
        "whole scope-manifest bytes issued before the cutoff; Git timestamps are "
        "content provenance only and cannot establish scope freeze time"
    )
    for index, entry in enumerate(entries, start=1):
        label = f"FROZEN_SCOPE_ENTRY_{index:04d}"
        path = Path(entry["source_path"])
        try:
            verify_scope_git_anchor(entry)
        except (OSError, ValueError, UnicodeError) as exc:
            rows.append(required_row(surface_id, label, str(path), "SNAPSHOT_ANCHOR_INVALID"))
            failures.append(
                f"{surface_id}: external Git snapshot anchor invalid for {path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        row, file_matches, failure = scan_file(surface_id, label, path, "frozen_scope_file")
        if failure:
            rows.append(row)
            failures.append(failure)
            continue
        if row["sha256"] != entry["source_sha256"]:
            row["read_status"] = "HASH_MISMATCH"
            rows.append(row)
            failures.append(
                f"{surface_id}: source hash differs from frozen manifest: {path}; "
                f"expected {entry['source_sha256']}, observed {row['sha256']}"
            )
            continue
        rows.append(row)
        matches.extend(file_matches)
    return rows, matches, failures


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    def writer(handle) -> None:
        output = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        output.writeheader()
        output.writerows(rows)

    atomic_write(path, writer)


def write_evidence(
    path: Path, surface_id: str, matches: Sequence[Dict[str, object]]
) -> None:
    def writer(handle) -> None:
        handle.write(f"schema_version: {SCHEMA_VERSION}\n")
        handle.write(f"surface_id: {surface_id}\n")
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


def validate_generation(directory: Path, surface_id: str, run_id: str) -> None:
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
    if status_doc.get("surface_id") != surface_id or status_doc.get("generation_id") != run_id:
        raise OSError("generation status identity mismatch")
    if status_doc.get("manifest_sha256") != sha256_file(manifest_path):
        raise OSError("generation manifest checksum mismatch")
    if status_doc.get("evidence_sha256") != sha256_file(evidence_path):
        raise OSError("generation evidence checksum mismatch")
    if done_doc.get("schema_version") != SCHEMA_VERSION:
        raise OSError("generation DONE schema mismatch")
    if done_doc.get("surface_id") != surface_id or done_doc.get("generation_id") != run_id:
        raise OSError("generation DONE identity mismatch")
    if done_doc.get("manifest_sha256") != sha256_file(manifest_path):
        raise OSError("generation DONE manifest checksum mismatch")
    if done_doc.get("evidence_sha256") != sha256_file(evidence_path):
        raise OSError("generation DONE evidence checksum mismatch")
    if done_doc.get("status_sha256") != sha256_file(status_path):
        raise OSError("generation DONE status checksum mismatch")
    if done_doc.get("bundle_complete") is not True:
        raise OSError("generation DONE completeness flag mismatch")


def update_current(surface_root: Path, surface_id: str, run_id: str) -> None:
    current = surface_root / "CURRENT"
    reject_symlink(current, "CURRENT pointer")
    fd, tmp_name = tempfile.mkstemp(prefix=".CURRENT.", dir=str(surface_root))
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
            handle.write(run_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.environ.get("RQ014_TEST_FAIL_CURRENT_SURFACE") == surface_id:
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


def publish_surface(
    output_root: Path,
    run_id: str,
    surface_id: str,
    required_inputs: Sequence[Dict[str, str]],
    rows: Sequence[Dict[str, object]],
    matches: Sequence[Dict[str, object]],
    failures: Sequence[str],
    exclusions: Sequence[str] = (),
    extra_status: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    reject_symlink(output_root, "output root")
    output_root.mkdir(parents=True, exist_ok=True)
    surface_root = output_root / surface_id
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
        write_evidence(evidence_path, surface_id, matches)
        state = (
            "INACCESSIBLE"
            if failures
            else "FOUND"
            if matches
            else "NOT_FOUND_ON_SCANNED_SURFACES"
        )
        record: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "surface_id": surface_id,
            "generation_id": run_id,
            "state": state,
            "complete_scan": not failures,
            "search_mode": "FULL_BYTE_STREAM_NO_SAMPLING",
            "scan_transport": "STREAMING_FILE_DESCRIPTOR_READ_TO_EOF",
            "sampling_used": False,
            "required_inputs": list(required_inputs),
            "declared_exclusions": list(exclusions),
            "read_failures": list(failures),
            "error": "; ".join(failures) if failures else None,
            "residual_risk_statement": (
                f"{surface_id} closure is not established because at least one required "
                "root, frozen-scope item, hash, or full-byte read failed."
                if failures
                else f"No known residual input-access risk remains on the declared {surface_id} surface."
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
        }
        if extra_status:
            record.update(extra_status)
        atomic_write(
            status_path,
            lambda handle: json.dump(record, handle, indent=2, sort_keys=True)
            or handle.write("\n"),
        )
        done_record = {
            "schema_version": SCHEMA_VERSION,
            "surface_id": surface_id,
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
        validate_generation(staging, surface_id, run_id)
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
        update_current(surface_root, surface_id, run_id)
    except OSError as exc:
        raise PublicationError(
            f"{surface_id} generation {final_generation} is durable but CURRENT "
            f"was not changed: {exc}"
        ) from exc
    return record


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    codes_root = Path(
        "/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes"
    )
    parser = argparse.ArgumentParser(
        description="Fail-closed F06/F07/F08 forensic closure scan (read-only)."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--archived-root", type=Path, default=codes_root / "archived")
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=codes_root / "9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle",
    )
    parser.add_argument("--f07-scope-manifest", type=Path, required=True)
    parser.add_argument("--f08-scope-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if not RUN_ID_RE.fullmatch(args.run_id):
        print("configuration error: invalid --run-id", file=sys.stderr)
        return 2
    declared_inputs = [
        args.archived_root,
        args.paper_repo,
        args.f07_scope_manifest,
        args.f08_scope_manifest,
    ]
    if any(
        is_within_even_if_missing(args.output_dir, path)
        or is_resolved_within_even_if_missing(args.output_dir, path)
        for path in declared_inputs
    ):
        print(
            "configuration error: --output-dir must be outside all declared inputs, "
            "including inputs that do not yet exist",
            file=sys.stderr,
        )
        return 2

    f07_entries, f07_scope_sha, f07_scope_failures = load_scope_manifest(
        args.f07_scope_manifest, "F07"
    )
    f08_entries, f08_scope_sha, f08_scope_failures = load_scope_manifest(
        args.f08_scope_manifest, "F08"
    )
    all_scope_paths = {
        resolved_even_if_missing(args.f07_scope_manifest),
        resolved_even_if_missing(args.f08_scope_manifest),
    }
    output_resolved = resolved_even_if_missing(args.output_dir)
    repo_root_text = os.environ.get("RQ014_PASS4_REPO_ROOT", "")
    current_repo_root = (
        resolved_even_if_missing(Path(repo_root_text)) if repo_root_text else None
    )
    for surface_id, scope_path, entries, scope_failures in (
        ("F07", args.f07_scope_manifest, f07_entries, f07_scope_failures),
        ("F08", args.f08_scope_manifest, f08_entries, f08_scope_failures),
    ):
        for entry in entries:
            source = Path(entry["source_path"])
            try:
                source_resolved = resolved_even_if_missing(source)
            except OSError as exc:
                scope_failures.append(
                    f"{surface_id}: cannot resolve frozen source path: {source}: {exc}"
                )
                continue
            if (
                is_within_even_if_missing(source, args.output_dir)
                or is_resolved_within_even_if_missing(source, args.output_dir)
                or is_within_even_if_missing(args.output_dir, source)
                or is_resolved_within_even_if_missing(args.output_dir, source)
                or source_resolved in all_scope_paths
                or (
                    current_repo_root is not None
                    and is_within_even_if_missing(source_resolved, current_repo_root)
                )
            ):
                scope_failures.append(
                    f"{surface_id}: frozen scope source overlaps current output, either "
                    f"scope manifest, or the current RQ014 repository and is rejected "
                    f"against self-contamination: {source}"
                )

    f06_required = [
        {"label": "SIBLING_ARCHIVED", "path": str(args.archived_root), "kind": "directory"},
        {"label": "PAPER_REPOSITORY", "path": str(args.paper_repo), "kind": "git_repository"},
    ]
    f06_rows, f06_matches, f06_failures = walk_required_dir(
        "F06", "SIBLING_ARCHIVED", args.archived_root
    )
    work_rows, work_matches, work_failures = walk_required_dir(
        "F06", "PAPER_WORKTREE", args.paper_repo, excluded_dir_names={".git"}
    )
    git_rows, git_matches, git_failures = scan_git_history(
        "F06", "PAPER_GIT_ALL_REACHABLE_COMMITS", args.paper_repo
    )
    f06_rows.extend(work_rows + git_rows)
    f06_matches.extend(work_matches + git_matches)
    f06_failures.extend(work_failures + git_failures)

    f07_rows, f07_matches, f07_failures = scan_frozen_scope(
        "F07", args.f07_scope_manifest, f07_entries, f07_scope_failures
    )
    f08_rows, f08_matches, f08_failures = scan_frozen_scope(
        "F08", args.f08_scope_manifest, f08_entries, f08_scope_failures
    )

    specs = [
        (
            "F06",
            f06_required,
            f06_rows,
            f06_matches,
            f06_failures,
            ["paper .git administrative directory; reachable Git blobs scanned instead"],
            {},
        ),
        (
            "F07",
            [
                {
                    "label": "FROZEN_SCOPE_MANIFEST",
                    "path": str(args.f07_scope_manifest),
                    "kind": "json_scope_manifest",
                }
            ],
            f07_rows,
            f07_matches,
            f07_failures,
            [],
            {
                "scope_manifest_file": str(args.f07_scope_manifest),
                "scope_manifest_sha256": f07_scope_sha,
                "scope_cutoff_utc": SCOPE_CUTOFF_TEXT,
                "scope_entry_count": len(f07_entries),
                "scope_anchor_kind": "git_blob_v1_content_integrity_only",
                "scope_anchor_count": len(f07_entries),
                "scope_time_witness_state": SCOPE_TIME_WITNESS_STATE,
            },
        ),
        (
            "F08",
            [
                {
                    "label": "FROZEN_SCOPE_MANIFEST",
                    "path": str(args.f08_scope_manifest),
                    "kind": "json_scope_manifest",
                }
            ],
            f08_rows,
            f08_matches,
            f08_failures,
            [],
            {
                "scope_manifest_file": str(args.f08_scope_manifest),
                "scope_manifest_sha256": f08_scope_sha,
                "scope_cutoff_utc": SCOPE_CUTOFF_TEXT,
                "scope_entry_count": len(f08_entries),
                "scope_anchor_kind": "git_blob_v1_content_integrity_only",
                "scope_anchor_count": len(f08_entries),
                "scope_time_witness_state": SCOPE_TIME_WITNESS_STATE,
            },
        ),
    ]
    records: List[Dict[str, object]] = []
    publication_errors: List[str] = []
    for surface_id, required, rows, matches, failures, exclusions, extra in specs:
        try:
            records.append(
                publish_surface(
                    args.output_dir,
                    args.run_id,
                    surface_id,
                    required,
                    rows,
                    matches,
                    failures,
                    exclusions,
                    extra,
                )
            )
        except (PublicationError, OSError) as exc:
            publication_errors.append(str(exc))
    if publication_errors:
        for error in publication_errors:
            try:
                print(f"ERROR: {error}", file=sys.stderr)
            except OSError:
                pass
        return 4
    try:
        print(json.dumps(records, sort_keys=True))
    except OSError:
        pass
    inaccessible = [record for record in records if record["state"] == "INACCESSIBLE"]
    for record in inaccessible:
        for failure in record["read_failures"]:
            try:
                print(f"ERROR {record['surface_id']}: {failure}", file=sys.stderr)
            except OSError:
                pass
    return 3 if inaccessible else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
