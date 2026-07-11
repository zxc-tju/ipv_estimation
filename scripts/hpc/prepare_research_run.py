#!/usr/bin/env python3
"""Validate and submit one immutable Tongji HPC research run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

DEFAULT_BASE = Path("/share/home/u25310231/ZXC/sociality_estimation")
MODEL_SHA256 = "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "rq_id",
        "run_id",
        "operation",
        "git_commit",
        "data_manifest_path",
        "data_manifest_sha256",
        "csv_path",
        "pkl_root",
    }
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Run spec missing keys: {sorted(missing)}")
    if int(spec["schema_version"]) != 1:
        raise ValueError("Unsupported run spec schema")
    for key in ("rq_id", "run_id", "operation"):
        if not SAFE_ID.fullmatch(str(spec[key])):
            raise ValueError(f"Unsafe {key}: {spec[key]}")
    return spec


def validate_spec(spec: dict, *, base: Path, repo: Path) -> dict:
    authorization_path = repo / "configs" / "research_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    rq = authorization["authorizations"].get(spec["rq_id"])
    if rq is None or spec["operation"] not in rq["allowed_operations"]:
        raise ValueError(f"Operation is not authorized: {spec['rq_id']} / {spec['operation']}")

    commit = str(spec["git_commit"])
    subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True,
    )
    published = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"]
    )
    if published.returncode != 0:
        raise ValueError("Run commit is not published on origin/main")
    if run_git(repo, "status", "--porcelain"):
        raise ValueError(f"Managed checkout is dirty: {repo}")

    manifest = Path(spec["data_manifest_path"]).resolve()
    if not manifest.is_file() or sha256_file(manifest) != spec["data_manifest_sha256"]:
        raise ValueError("Data manifest is missing or has the wrong SHA-256")
    csv_path = Path(spec["csv_path"]).resolve()
    pkl_root = Path(spec["pkl_root"]).resolve()
    if not csv_path.is_file() or not pkl_root.is_dir():
        raise ValueError("CSV or PKL root is missing")

    model = base / "checkpoints" / "rq009_m3" / "m3_scorer.joblib"
    if sha256_file(model) != MODEL_SHA256:
        raise ValueError("Portable M3 scorer SHA-256 mismatch")
    model_manifest_path = model.with_name("manifest.json")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest["artifact"]["sha256"] != MODEL_SHA256:
        raise ValueError("Portable M3 manifest does not register the production scorer")
    contract = model.with_name(model_manifest["feature_contract"]["path"])
    contract_sha = sha256_file(contract)
    if contract_sha != model_manifest["feature_contract"]["sha256"]:
        raise ValueError("Portable M3 feature contract SHA-256 mismatch")

    run_root = base / "work_dirs" / spec["rq_id"] / spec["run_id"]
    return {
        "authorization_sha256": sha256_file(authorization_path),
        "model_path": str(model),
        "model_sha256": MODEL_SHA256,
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "model_contract_path": str(contract),
        "model_contract_sha256": contract_sha,
        "run_root": str(run_root),
        "commit": commit,
        "csv_path": str(csv_path),
        "pkl_root": str(pkl_root),
        "data_manifest_path": str(manifest),
        "data_manifest_sha256": spec["data_manifest_sha256"],
    }


def prepare_and_submit(spec: dict, validated: dict, *, base: Path, repo: Path) -> str:
    run_root = Path(validated["run_root"])
    if run_root.exists():
        raise ValueError(f"Run root already exists: {run_root}")
    for name in ("outputs", "logs", "manifests"):
        (run_root / name).mkdir(parents=True, exist_ok=False)
    code = run_root / "code"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(code), validated["commit"]],
        check=True,
    )

    manifest = {"schema_version": 1, "spec": spec, "validated": validated}
    manifest_path = run_root / "manifests" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    python = base / "envs" / "ipv-exact-sigma01" / "bin" / "python"
    command = [
        str(python),
        str(code / "pipelines" / "interhub" / "process_interhub.py"),
        "--csv", validated["csv_path"],
        "--pkl-root", validated["pkl_root"],
        "--output-root", str(run_root / "outputs"),
        "--log-workflow",
        "--workflow-log-path", str(run_root / "logs" / "workflow.log"),
    ]
    if spec["operation"] == "interhub_preflight":
        command.append("--preflight-only")
    elif spec["operation"] == "interhub_parity_fixture":
        command.extend(["--limit", str(int(spec.get("limit", 1))), "--no-plots", "--workers", "1"])
    else:
        raise ValueError("Unsupported managed operation")

    job_name = f"zxc-{spec['rq_id'].lower()}-{spec['run_id']}"[:80]
    script = run_root / "manifests" / "run.sbatch"
    quoted = " ".join(shlex.quote(item) for item in command)
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --partition=amd\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=2\n#SBATCH --time=01:00:00\n"
        f"#SBATCH --output={run_root}/logs/slurm_%j.out\n"
        f"#SBATCH --error={run_root}/logs/slurm_%j.err\n\n"
        "set -euo pipefail\n"
        f"test \"$(git -C {shlex.quote(str(code))} rev-parse HEAD)\" = {shlex.quote(validated['commit'])}\n"
        f"test -z \"$(git -C {shlex.quote(str(code))} status --porcelain)\"\n"
        f"test \"$(sha256sum {shlex.quote(validated['data_manifest_path'])} | awk '{{print $1}}')\" = {validated['data_manifest_sha256']}\n"
        f"test \"$(sha256sum {shlex.quote(validated['model_path'])} | awk '{{print $1}}')\" = {MODEL_SHA256}\n"
        f"test \"$(sha256sum {shlex.quote(validated['model_manifest_path'])} | awk '{{print $1}}')\" = {validated['model_manifest_sha256']}\n"
        f"test \"$(sha256sum {shlex.quote(validated['model_contract_path'])} | awk '{{print $1}}')\" = {validated['model_contract_sha256']}\n"
        f"exec 8>{shlex.quote(str(base / 'manifests' / 'runtime_maintenance.lock'))}\n"
        "flock -s 8\n"
        f"export SOCIALITY_PRODUCTION_RUN_ROOT={shlex.quote(str(run_root))}\n"
        f"export SOCIALITY_M3_SCORER={shlex.quote(validated['model_path'])}\n"
        f"export PYTHONPATH={shlex.quote(str(code / 'src'))}\n"
        f"{quoted}\n",
        encoding="utf-8",
    )
    submitted = subprocess.check_output(["sbatch", str(script)], text=True).strip()
    (run_root / "manifests" / "submission_receipt.txt").write_text(submitted + "\n", encoding="utf-8")
    return submitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    repo = base / "code" / "repo"
    spec = load_spec(args.spec)
    validated = validate_spec(spec, base=base, repo=repo)
    if args.submit:
        print(prepare_and_submit(spec, validated, base=base, repo=repo))
    else:
        print(json.dumps(validated, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
