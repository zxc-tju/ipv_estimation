#!/usr/bin/env python3
"""Lint reports/knowledge/ against CONVENTIONS.md.

Checks (see ../CONVENTIONS.md):
  1. No .DS_Store anywhere under knowledge/.
  2. No loose .md/.json at the knowledge root except the allowlist.
  3. Every reviews/*.md matches the role-naming regex.
  4. Every RQ/paper folder has its required standard files (per registry status).
  5. Registry id <-> folder consistency (incl. registered B-series and papers).

Exit code 0 = clean, 1 = violations found. Run: python3 lint_knowledge.py
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

KNOWLEDGE = Path(__file__).resolve().parent.parent
REGISTRY = KNOWLEDGE / "rq_progress_registry.csv"

ROOT_ALLOW = {
    "README.md",
    "CONVENTIONS.md",
    "RQ_PROGRESS_DASHBOARD.md",
    "rq_progress_registry.csv",
}
ROOT_ALLOW_GLOBS = ("KNOWLEDGE_GOVERNANCE_PLAN_", )  # prefix allowlist
REVIEW_RE = re.compile(r"^(chatgpt|claude|codex)_(review|response)(_[a-z][a-z0-9]*)*\.md$")
STANDARD = ("README.md", "report_index.md", "synthesis.md", "decision.md")
# Folders exempt from the full standard-file set (README still required).
LIGHT_FOLDERS = {"PAPER001"}  # imported manuscript context only

violations: list[str] = []


def v(msg: str) -> None:
    violations.append(msg)


def load_registry() -> dict[str, str]:
    """program_id -> program_status (lowercased)."""
    ids: dict[str, str] = {}
    with REGISTRY.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("program_id") or "").strip()
            if pid:
                ids[pid] = (row.get("program_status") or "").strip().lower()
    return ids


def folder_for(pid: str, folders: list[str]) -> str | None:
    for f in folders:
        if f == pid or f.startswith(pid + "_"):
            return f
    return None


def main() -> int:
    if not REGISTRY.exists():
        print(f"FATAL: registry not found at {REGISTRY}")
        return 2
    registry = load_registry()

    # Check 1: no .DS_Store
    for p in KNOWLEDGE.rglob(".DS_Store"):
        v(f"[1] stray .DS_Store: {p.relative_to(KNOWLEDGE)}")

    # Check 2: no loose md/json at root
    for p in KNOWLEDGE.iterdir():
        if p.is_file() and p.suffix in (".md", ".json"):
            if p.name in ROOT_ALLOW:
                continue
            if any(p.name.startswith(g) for g in ROOT_ALLOW_GLOBS):
                continue
            v(f"[2] loose file at knowledge root: {p.name}")

    # Check 3: review filenames
    for rev in KNOWLEDGE.rglob("reviews"):
        if not rev.is_dir():
            continue
        for f in rev.iterdir():
            if f.is_file() and f.suffix == ".md" and not REVIEW_RE.match(f.name):
                v(f"[3] non-conforming review name: {f.relative_to(KNOWLEDGE)}")

    # Folder set (non-infrastructure)
    folders = [
        p.name for p in KNOWLEDGE.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    ]

    # Check 5a: every registry id has a folder
    matched: set[str] = set()
    for pid in registry:
        f = folder_for(pid, folders)
        if f is None:
            v(f"[5] registry id has no knowledge folder: {pid}")
        else:
            matched.add(f)

    # Check 5b: every folder maps to a registry id
    for f in folders:
        if f not in matched:
            v(f"[5] folder maps to no registry id: {f}")

    # Check 4: standard files per folder
    for pid, status in registry.items():
        f = folder_for(pid, folders)
        if f is None:
            continue
        fp = KNOWLEDGE / f
        # README always required
        if not (fp / "README.md").exists():
            v(f"[4] {f}: missing README.md")
        if pid in LIGHT_FOLDERS or status == "planning":
            continue
        for std in STANDARD[1:]:  # report_index/synthesis/decision
            if not (fp / std).exists():
                v(f"[4] {f}: missing {std} (status={status or 'n/a'})")

    if violations:
        print(f"FAIL: {len(violations)} knowledge-governance violation(s):")
        for msg in violations:
            print("  " + msg)
        return 1
    print("OK: knowledge/ conforms to CONVENTIONS.md "
          f"({len(registry)} registry ids, {len(folders)} folders).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
