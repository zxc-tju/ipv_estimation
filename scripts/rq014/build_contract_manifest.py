#!/usr/bin/env python3
"""Build deterministic RQ014 review/final SHA-256 manifests."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed manifest line {path}:{line_number}") from exc
        if relative in rows:
            raise ValueError(f"Duplicate manifest path: {relative}")
        rows[relative] = digest
    return rows


def review_paths(repo_root: Path) -> set[str]:
    repo_root_string = str(repo_root)
    if repo_root_string not in sys.path:
        sys.path.insert(0, repo_root_string)
    from scripts.hpc.prepare_research_run import RQ014_REVIEW_REQUIRED_PATHS

    paths = set(RQ014_REVIEW_REQUIRED_PATHS)
    legacy = repo_root / "reports" / "plans" / "RQ014_plan_v1p3_checksums_20260711.sha256"
    paths.update(parse_manifest(legacy))
    return paths


def build_manifest(repo_root: Path, paths: set[str]) -> bytes:
    rows: list[str] = []
    for relative in sorted(paths):
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes repository: {relative}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Manifest path is not a regular non-symlink file: {relative}")
        rows.append(f"{sha256_file(path)}  {relative}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"Temporary manifest path already exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=["review", "final"], required=True)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.mode == "review":
        paths = review_paths(repo_root)
    else:
        if args.review_manifest is None:
            raise ValueError("Final mode requires --review-manifest")
        review_manifest = args.review_manifest.resolve()
        paths = set(parse_manifest(review_manifest))
        paths.add(str(review_manifest.relative_to(repo_root)))
    paths.update(args.extra)
    payload = build_manifest(repo_root, paths)
    output = args.output if args.output.is_absolute() else repo_root / args.output
    atomic_write(output.resolve(), payload)
    print(f"{sha256_file(output.resolve())}  {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
