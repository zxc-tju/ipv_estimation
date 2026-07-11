#!/usr/bin/env python3
"""Record the durable gate proving a managed preflight passed after data switch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path("/share/home/u25310231/ZXC/sociality_estimation")
INVENTORY_ID = "pre_migration_20260711_v1"
SNAPSHOT_ID = "interhub_legacy_20260711_v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise RuntimeError(f"Malformed marker line in {path}: {line!r}")
        values[key] = value
    return values


def require_within(path: Path, parent: Path) -> None:
    if os.path.commonpath([str(path.resolve()), str(parent.resolve())]) != str(parent.resolve()):
        raise RuntimeError(f"Path escapes expected snapshot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if (
        not SAFE_ID.fullmatch(args.run_id)
        or not args.job_id.isdigit()
        or not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit)
    ):
        raise ValueError("Unsafe run, job, or commit id")

    inventory = BASE / "manifests" / "legacy_migration" / INVENTORY_ID
    switch_path = inventory / "SWITCH_COMPLETE"
    switch = parse_key_values(switch_path)
    raw_snapshot = BASE / "data" / "interhub" / "snapshots" / SNAPSHOT_ID
    results_snapshot = BASE / "archives" / "historical-results" / SNAPSHOT_ID
    expected_switch = {
        "snapshot_id": SNAPSHOT_ID,
        "raw_snapshot": str(raw_snapshot),
        "results_snapshot": str(results_snapshot),
    }
    for key, expected in expected_switch.items():
        if switch.get(key) != expected:
            raise RuntimeError(f"Switch marker mismatch for {key}")
    verification_dir = inventory / f"snapshot_reverification_{SNAPSHOT_ID}"
    attestation_path = verification_dir / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    verification_logs = {
        "raw": verification_dir / "raw_snapshot_verify.txt",
        "results": verification_dir / "results_snapshot_verify.txt",
    }
    source_manifests = {
        "raw": inventory / "raw_sha256.txt",
        "results": inventory / "results_sha256.txt",
    }
    if attestation.get("switch_marker_sha256") != sha256_file(switch_path):
        raise RuntimeError("Migration attestation does not bind the switch marker")
    if attestation.get("source_manifest_sha256") != {
        name: sha256_file(path) for name, path in source_manifests.items()
    }:
        raise RuntimeError("Migration attestation does not bind the source manifests")
    if attestation.get("snapshot_verification_sha256") != {
        name: sha256_file(path) for name, path in verification_logs.items()
    }:
        raise RuntimeError("Migration attestation does not bind the snapshot verification logs")
    if attestation.get("verification_counts") != {"raw": 51, "results": 173034}:
        raise RuntimeError("Migration attestation has unexpected verification counts")

    accounting_output = subprocess.check_output(
        [
            "sacct", "-P", "-n", "-X", "-j", args.job_id,
            "--format=JobIDRaw,State,ExitCode,Start,WorkDir,JobName",
        ],
        text=True,
    ).strip().splitlines()
    if not accounting_output:
        raise RuntimeError("No Slurm accounting record for preflight job")
    fields = accounting_output[0].split("|")
    if len(fields) != 6:
        raise RuntimeError(f"Malformed Slurm accounting record: {accounting_output[0]}")
    job_id_raw, state, exit_code, start_text, work_dir, job_name = fields
    if job_id_raw != args.job_id or state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(f"Preflight job did not complete successfully: {accounting_output[0]}")
    switch_time = datetime.fromisoformat(switch["switched_at"])
    start_time = datetime.fromisoformat(start_text)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone(timedelta(hours=8)))
    if switch_time.tzinfo is None or start_time < switch_time:
        raise RuntimeError("Preflight job did not start after the data switch")

    run_root = BASE / "work_dirs" / "INFRA" / args.run_id
    run_manifest_path = run_root / "manifests" / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    spec = run_manifest["spec"]
    validated = run_manifest["validated"]
    if spec["rq_id"] != "INFRA" or spec["operation"] != "interhub_preflight":
        raise RuntimeError("Run is not an authorized INFRA InterHub preflight")
    if spec["run_id"] != args.run_id:
        raise RuntimeError("Run id differs from run manifest")
    receipt = (run_root / "manifests" / "submission_receipt.txt").read_text(encoding="utf-8")
    if re.fullmatch(rf"Submitted batch job {re.escape(args.job_id)}\s*", receipt) is None:
        raise RuntimeError("Job id differs from submission receipt")
    if job_name != f"zxc-infra-{args.run_id}"[:80]:
        raise RuntimeError("Slurm job name differs from the managed run")
    if Path(work_dir).resolve() != (BASE / "code" / "repo").resolve():
        raise RuntimeError("Slurm job was submitted from an unexpected checkout")
    if validated["commit"] != args.expected_commit or spec["git_commit"] != args.expected_commit:
        raise RuntimeError("Run did not use the explicitly approved commit")
    run_code = run_root / "code"
    if subprocess.check_output(
        ["git", "-C", str(run_code), "rev-parse", "HEAD"], text=True
    ).strip() != args.expected_commit:
        raise RuntimeError("Detached run checkout differs from approved commit")
    if subprocess.check_output(
        ["git", "-C", str(run_code), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Detached run checkout is dirty")
    published = subprocess.run(
        [
            "git", "-C", str(BASE / "code" / "repo"), "merge-base", "--is-ancestor",
            args.expected_commit, "refs/remotes/origin/main",
        ]
    )
    if published.returncode != 0:
        raise RuntimeError("Approved preflight commit is not published on origin/main")

    raw_manifest = inventory / "raw_sha256.txt"
    if validated["data_manifest_path"] != str(raw_manifest):
        raise RuntimeError("Preflight did not use the migrated raw-data manifest")
    if validated["data_manifest_sha256"] != sha256_file(raw_manifest):
        raise RuntimeError("Preflight raw-data manifest SHA-256 mismatch")
    require_within(Path(validated["csv_path"]), raw_snapshot)
    require_within(Path(validated["pkl_root"]), raw_snapshot)
    if not Path(validated["csv_path"]).is_file() or not Path(validated["pkl_root"]).is_dir():
        raise RuntimeError("Preflight inputs are missing after switch")

    for snapshot in (raw_snapshot, results_snapshot):
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise RuntimeError(f"Snapshot is not a real directory: {snapshot}")
        writable = subprocess.check_output(
            ["find", str(snapshot), "-perm", "/222", "-print", "-quit"], text=True
        ).strip()
        if writable:
            raise RuntimeError(f"Writable path found in immutable snapshot: {writable}")

    checks = {
        name: (verification_logs[name], source_manifests[name])
        for name in ("raw", "results")
    }
    verification_counts: dict[str, int] = {}
    for name, (verification, manifest) in checks.items():
        ok_count = sum(line.endswith(": OK\n") for line in verification.open(encoding="utf-8"))
        manifest_count = sum(1 for _ in manifest.open(encoding="utf-8"))
        if ok_count != manifest_count:
            raise RuntimeError(f"Incomplete {name} copy verification")
        verification_counts[name] = ok_count

    marker = {
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "run_id": args.run_id,
        "job_id": args.job_id,
        "job_state": "COMPLETED",
        "job_exit_code": "0:0",
        "job_started_at": start_time.isoformat(),
        "job_name": job_name,
        "job_work_dir": work_dir,
        "git_commit": validated["commit"],
        "data_manifest_path": str(raw_manifest),
        "data_manifest_sha256": validated["data_manifest_sha256"],
        "raw_snapshot": str(raw_snapshot),
        "results_snapshot": str(results_snapshot),
        "verification_counts": verification_counts,
        "migration_attestation_path": str(attestation_path),
        "migration_attestation_sha256": sha256_file(attestation_path),
        "snapshot_verification_sha256": attestation["snapshot_verification_sha256"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = inventory / "POST_SWITCH_PREFLIGHT_COMPLETE.json"
    if marker_path.exists():
        raise RuntimeError(f"Post-switch preflight marker already exists: {marker_path}")
    with tempfile.NamedTemporaryFile("w", dir=inventory, delete=False, encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(marker_path)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
