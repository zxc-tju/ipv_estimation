#!/usr/bin/env python3
"""Validate and submit one immutable Tongji HPC research run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

RQ014_MANAGED_BASE = Path("/share/home/u25310231/ZXC/sociality_estimation")
RQ014_RUNTIME_LOCK_PATH = RQ014_MANAGED_BASE / "manifests" / "runtime_maintenance.lock"
RQ014_WRAPPER_PATH = (
    RQ014_MANAGED_BASE / "code" / "repo" / "scripts" / "hpc" / "submit_research_run.sh"
)
RQ014_PROC_FD_ROOT = Path("/proc/self/fd")
_RQ014_VERIFIED_WRAPPER_CAPABILITY = object()


def _verify_rq014_wrapper_capability(
    *,
    runtime_lock_path: Path = RQ014_RUNTIME_LOCK_PATH,
    wrapper_path: Path = RQ014_WRAPPER_PATH,
    proc_fd_root: Path = RQ014_PROC_FD_ROOT,
    runtime_lock_fd: int = 8,
    wrapper_fd: int = 9,
) -> object:
    """Verify the local wrapper provenance descriptors before RQ014 imports."""

    expected = (
        (runtime_lock_fd, Path(runtime_lock_path), "runtime lock"),
        (wrapper_fd, Path(wrapper_path), "managed wrapper"),
    )
    identity_fields = ("st_dev", "st_ino", "st_mode")
    for descriptor, path, label in expected:
        try:
            before = os.lstat(path)
            opened = os.fstat(descriptor)
            target = os.readlink(proc_fd_root / str(descriptor))
            after = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(f"RQ014 wrapper capability is missing or invalid: {label}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"RQ014 wrapper capability path is not regular: {label}")
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"RQ014 wrapper capability descriptor is not regular: {label}")
        if target != str(path):
            raise RuntimeError(f"RQ014 wrapper capability target mismatch: {label}")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise RuntimeError(f"RQ014 wrapper capability path identity drift: {label}")
        if any(getattr(opened, field) != getattr(after, field) for field in identity_fields):
            raise RuntimeError(f"RQ014 wrapper capability descriptor identity mismatch: {label}")
    return _RQ014_VERIFIED_WRAPPER_CAPABILITY


_RQ014_WRAPPER_CAPABILITY_VERIFIED: object | None = None
_RQ014_DIRECT_INTERNAL_MODE_REQUESTED = "--rq014-only" in sys.argv[1:]
if _RQ014_DIRECT_INTERNAL_MODE_REQUESTED:
    _RQ014_WRAPPER_CAPABILITY_VERIFIED = _verify_rq014_wrapper_capability()

REPO_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _exact_rq014_package() -> Any:
    """Create the closed package surface used by direct isolated launcher startup."""

    direct_startup = not __package__
    scripts_package = sys.modules.get("scripts")
    if scripts_package is None:
        scripts_package = types.ModuleType("scripts")
        scripts_package.__path__ = ()
        scripts_package.__package__ = "scripts"
        sys.modules["scripts"] = scripts_package
    rq014_package = sys.modules.get("scripts.rq014")
    if rq014_package is None:
        rq014_package = types.ModuleType("scripts.rq014")
        rq014_package.__path__ = (
            ()
            if direct_startup
            else (str(REPO_SOURCE_ROOT / "scripts" / "rq014"),)
        )
        rq014_package.__package__ = "scripts.rq014"
        sys.modules["scripts.rq014"] = rq014_package
    scripts_package.rq014 = rq014_package
    return rq014_package


def _load_exact_preflight_module() -> Any:
    """Load exact reviewed validator dependencies without a path-based import."""

    rq014_package = _exact_rq014_package()
    materializer_path = REPO_SOURCE_ROOT / "scripts" / "rq014" / "materialize_registry.py"
    materializer_spec = importlib.util.spec_from_file_location(
        "scripts.rq014.materialize_registry", materializer_path
    )
    if materializer_spec is None or materializer_spec.loader is None:
        raise RuntimeError(
            f"Cannot construct exact-path registry materializer loader: {materializer_path}"
        )
    materializer_module = importlib.util.module_from_spec(materializer_spec)
    rq014_package.materialize_registry = materializer_module
    sys.modules["scripts.rq014.materialize_registry"] = materializer_module
    materializer_spec.loader.exec_module(materializer_module)

    path = REPO_SOURCE_ROOT / "scripts" / "rq014" / "preflight.py"
    spec = importlib.util.spec_from_file_location("_rq014_reviewed_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot construct exact-path preflight loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RQ014_PREFLIGHT = None
if __name__ != "__main__" or _RQ014_DIRECT_INTERNAL_MODE_REQUESTED:
    _RQ014_PREFLIGHT = _load_exact_preflight_module()
    RQ014ContractError = _RQ014_PREFLIGHT.ContractError
    load_rq014_json = _RQ014_PREFLIGHT.load_json
    require_contained_regular_file = _RQ014_PREFLIGHT.require_contained_regular_file
    validate_declassification_export_receipts = (
        _RQ014_PREFLIGHT.validate_declassification_export_receipts
    )
    validate_g2_input_roles = _RQ014_PREFLIGHT.validate_g2_input_roles
    validate_m3_artifact_ref = _RQ014_PREFLIGHT.validate_m3_artifact_ref
    validate_materialization_ledger = _RQ014_PREFLIGHT.validate_materialization_ledger
    validate_score_stripped_bundle = _RQ014_PREFLIGHT.validate_score_stripped_bundle
    validate_wod_path_type_mapping_manifest = (
        _RQ014_PREFLIGHT.validate_wod_path_type_mapping_manifest
    )

DEFAULT_BASE = RQ014_MANAGED_BASE
SYSTEM_GIT = "/usr/bin/git"
SYSTEM_SBATCH = "/usr/bin/sbatch"
SYSTEM_SHA256SUM = "/usr/bin/sha256sum"
SYSTEM_AWK = "/usr/bin/awk"
SYSTEM_FLOCK = "/usr/bin/flock"
SYSTEM_ENV = "/usr/bin/env"
SYSTEM_FIND = "/usr/bin/find"
SYSTEM_STAT = "/usr/bin/stat"
SYSTEM_READLINK = "/usr/bin/readlink"
MINIMAL_PATH = "/usr/bin:/bin"
MINIMAL_COMMAND_ENV = {
    "PATH": MINIMAL_PATH,
    "LANG": "C",
    "LC_ALL": "C",
}
GIT_COMMAND_ENV = {
    **MINIMAL_COMMAND_ENV,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
}
MODEL_SHA256 = "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253"
M3_ARTIFACT_PATH = str(RQ014_MANAGED_BASE / "checkpoints" / "rq009_m3" / "m3_scorer.joblib")
M3_ARTIFACT_SIZE_BYTES = 88306301
SAFE_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
RQ014_EXPORT_OPERATION = "rq014_g2_declassification_export"
RQ014_PREFLIGHT_OPERATION = "rq014_g2_contract_preflight"
RQ014_EXPORT_RESOURCE_PROFILE = "rq014-g2-declassify-cpu-v1"
RQ014_PREFLIGHT_RESOURCE_PROFILE = "rq014-g2-preflight-cpu-v1"
RQ014_WRAPPER_CAPABILITY_CONTRACT = (
    "local machine provenance gate: the clean operator bootstrap opens and locks fd8 at "
    "/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock, "
    "opens fd9 at the exact managed submit_research_run.sh, hashes and executes that retained "
    "fd9; the wrapper must inherit both descriptors and may not create them; before any RQ014 "
    "registry-materializer or preflight preload, the launcher requires exact /proc/self/fd "
    "targets, regular non-symlink fixed paths, and fstat st_dev/st_ino/st_mode equality; this "
    "is not a cryptographic secret against deliberate same-account descriptor emulation"
)
RQ014_PYTHON_IMPORT_SURFACE = (
    "before the first managed-Python process, the wrapper uses absolute system tools to "
    "verify the pinned launcher, preflight and registry-materializer bytes, Python executable, "
    "v3 environment "
    "manifest, exact stdlib checksum manifest, zip absence, zero stdlib symlinks, complete "
    "stdlib regular-file set/count/size/hash, pinned consumer-derived native dependency set, "
    "exact loader links/final bytes and exact isolated sys.path; /lib64 is the explicit "
    "operating-system ABI trust boundary; the closed-snapshot registry materializer is "
    "explicitly preloaded by spec_from_file_location as scripts.rq014.materialize_registry, "
    "then preflight and the fixed scientific entrypoint are loaded only by exact reviewed "
    "closed-snapshot paths; scripts and scripts.rq014 keep empty __path__, so ordinary path "
    "import, checkout/root/src/sitecustomize/usercustomize/PYTHONPATH/local shadows remain "
    "unavailable"
)
RQ014_REVIEW_MANIFEST = (
    "reports/plans/RQ014_plan_v1p6_preflight_review_manifest_20260713.sha256"
)
RQ014_STATISTICS_REVIEW = (
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p6_preflight_statistics_review_20260713.json"
)
RQ014_EXECUTION_REVIEW = (
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p6_preflight_execution_governance_review_20260713.json"
)
RQ014_FORMAL_G1 = (
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_formal_G1_v1p6_preflight_20260713.yaml"
)
RQ014_FINAL_BUNDLE = "reports/plans/RQ014_plan_v1p6_checksums_20260713.sha256"
RQ014_REVIEW_REQUIRED_PATHS = {
    "START_HERE.md",
    "configs/research_authorization.json",
    "configs/run_specs/README.md",
    "configs/run_specs/RQ014_g2_contract_preflight.template.json",
    "configs/run_specs/RQ014_g2_declassification_export.template.json",
    "configs/run_specs/research_run_spec_v2.schema.json",
    "configs/run_specs/rq014_managed_python_environment_v3.schema.json",
    "reports/plans/RQ014_PI_decision_G0_waiver_launch_20260711.md",
    "reports/plans/RQ014_PI_decision_G2_start_v1p5_20260712.md",
    "reports/plans/RQ014_PI_decision_D1_preflight_v1p6_20260713.md",
    "reports/plans/RQ014_PI_decision_envelope_identification_20260713.md",
    "reports/plans/RQ014_blind_anchor_receipt_v1p5.json",
    "reports/plans/RQ014_config_space_v1p5.yaml",
    "reports/plans/RQ014_config_space_v1p6.yaml",
    "reports/plans/RQ014_envelope_builder_contract_v2.json",
    "reports/plans/RQ014_execution_contract_v1p5.json",
    "reports/plans/RQ014_estimator_core_tree_v1p7.json",
    "reports/plans/RQ014_forensic_registry_v1p5.yaml",
    "reports/plans/RQ014_plan_v1p3_checksums_20260711.sha256",
    "reports/plans/RQ014_plan_v1p5_amendment_20260712.md",
    "reports/plans/RQ014_plan_v1p7_amendment_20260713.md",
    "reports/plans/RQ014_plan_v1p7_addendum_pathtype_20260713.md",
    "reports/plans/RQ014_plan_v1p5_review_manifest_round1_blocked_20260712.sha256",
    "reports/plans/RQ014_plan_v1p5_review_manifest_round2_blocked_20260712.sha256",
    "reports/plans/RQ014_plan_v1p5_review_manifest_round3_blocked_20260712.sha256",
    "reports/plans/RQ014_plan_v1p5_review_manifest_round4_blocked_20260712.sha256",
    "reports/plans/RQ014_plan_v1p5_review_manifest_round5_blocked_20260712.sha256",
    "reports/plans/RQ014_recovery_extension_registry_v1p5.yaml",
    "reports/plans/RQ014_recovery_extension_registry_v1p6.yaml",
    "reports/plans/RQ014_recovery_lane_v2.json",
    "reports/plans/RQ014_recovery_lane_v3.json",
    "reports/plans/RQ014_score_stripped_schema_v1.json",
    "reports/plans/prompts/RQ014_G2_kickoff_prompt_v1p5_20260712.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/"
    "RQ014_declassification_source_inventory_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/"
    "RQ014_hpc_runtime_compatibility_probe_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/"
    "wod_path_type_mapping_v1/wod_path_type_mapping.csv",
    "reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/"
    "wod_path_type_mapping_v1/manifest.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/"
    "wod_path_type_mapping_v1/distribution_summary.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/forensics_report.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_statistics_review_round1_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_execution_governance_review_round1_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_round1_remediation_20260712.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_statistics_review_round2_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_execution_governance_review_round2_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_round2_remediation_20260712.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_statistics_review_round3_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_execution_governance_review_round3_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_round3_remediation_20260712.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_statistics_review_round4_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_execution_governance_review_round4_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_round4_remediation_20260712.md",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_statistics_review_round5_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_execution_governance_review_round5_blocked_20260712.json",
    "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
    "RQ014_v1p5_round5_remediation_20260712.md",
    "scripts/hpc/prepare_research_run.py",
    "scripts/hpc/submit_research_run.sh",
    "scripts/rq014/export_score_stripped_bundle.py",
    "scripts/rq014/build_contract_manifest.py",
    "scripts/rq014/materialize_registry.py",
    "scripts/rq014/preflight.py",
    "scripts/rq014/run_managed_g2.py",
    "scripts/rq014/derive_wod_path_type_mapping.py",
    "scripts/rq014/spearman_average_midranks.py",
    "scripts/rq014/spearman_version_manifest_v1.json",
    "scripts/rq014/wod_ipv_adapter.py",
    "scripts/rq014/wod_ipv_preprocessing.py",
    "scripts/rq014/wod_reference_builder.py",
    "scripts/rq014/wod_path_type_mapping_version_manifest_v1.json",
    "src/sociality_estimation/__init__.py",
    "src/sociality_estimation/core/__init__.py",
    "src/sociality_estimation/core/agent.py",
    "src/sociality_estimation/core/ipv_estimation.py",
    "src/sociality_estimation/planning/Lattice.py",
    "src/sociality_estimation/planning/__init__.py",
    "src/sociality_estimation/planning/utility.py",
    "tests/test_hpc_run_launcher.py",
    "tests/test_rq014_fl05_indexer.py",
    "tests/test_rq014_g0_closure_scripts.py",
    "tests/test_rq014_managed_hpc_contract.py",
    "tests/test_rq014_recovery_lane_v2_contract.py",
    "tests/test_rq014_recovery_lane_v3_contract.py",
    "tests/test_rq014_science_freeze_v1p7.py",
    "tests/test_rq014_score_stripped_export.py",
    "tests/test_rq014_v1p3_registry_contract.py",
    "tests/test_rq014_wod_path_type_mapping.py",
    "tests/test_rq014_v1p5_contract.py",
    "tests/fixtures/rq014_wod_path_type_mapping_golden_v1.json",
    "models/rq009_m3/README.md",
    "models/rq009_m3/feature_spec_contract.json",
    "models/rq009_m3/manifest.json",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON token: {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON run spec: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Run spec must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_command(repo: Path, *args: str) -> list[str]:
    return [
        SYSTEM_GIT,
        "--no-replace-objects",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.attributesFile=/dev/null",
        "-C",
        str(repo),
        *args,
    ]


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        _git_command(repo, *args),
        text=True,
        env=GIT_COMMAND_ENV,
    ).strip()


def _require_published_commit(repo: Path, commit: str, *, label: str) -> None:
    subprocess.run(
        _git_command(repo, "cat-file", "-e", f"{commit}^{{commit}}"),
        check=True,
        env=GIT_COMMAND_ENV,
    )
    published = subprocess.run(
        _git_command(
            repo,
            "merge-base",
            "--is-ancestor",
            commit,
            "refs/remotes/origin/main",
        ),
        env=GIT_COMMAND_ENV,
    )
    if published.returncode != 0:
        raise ValueError(f"{label} is not published on origin/main")


def _run_git_bytes(repo: Path, *args: str) -> bytes:
    """Read Git object bytes without filters, hooks, or worktree participation."""

    process = subprocess.Popen(
        _git_command(repo, *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=GIT_COMMAND_ENV,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git object lookup failed: {detail or args}")
    return stdout


def _read_git_commit_regular_blob(repo: Path, commit: str, relative: str) -> tuple[bytes, str]:
    """Return one exact regular-file blob and its Git mode from a commit tree."""

    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != relative
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(character in relative for character in ("\0", "\n", "\r", "\t"))
    ):
        raise ValueError(f"Unsafe RQ014 commit-tree path: {relative!r}")
    record = _run_git_bytes(
        repo,
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        relative,
    )
    if not record.endswith(b"\0") or record.count(b"\0") != 1:
        raise ValueError(f"RQ014 commit does not contain exactly one registered path: {relative}")
    metadata, raw_path = record[:-1].split(b"\t", 1)
    try:
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        committed_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed Git tree record for RQ014 path: {relative}") from exc
    if (
        committed_path != relative
        or object_type != "blob"
        or mode not in {"100644", "100755"}
        or HEX40.fullmatch(object_id) is None
    ):
        raise ValueError(f"RQ014 commit path is not a regular file: {relative}")
    return _run_git_bytes(repo, "cat-file", "blob", object_id), mode


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components_for_new_path(path: Path, *, root: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    root_absolute = Path(os.path.abspath(root))
    try:
        relative = absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its managed root") from exc
    if root_absolute.is_symlink():
        raise ValueError(f"{label} managed root is a symlink")
    current = root_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} crosses a symlink: {current}")


def load_spec(path: Path) -> dict:
    return _validate_loaded_spec(_strict_json_load(path))


def _validate_loaded_spec(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        version = int(spec["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Run spec has no valid schema_version") from exc
    if version == 1:
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
    elif version == 2:
        common = {
            "schema_version",
            "rq_id",
            "run_id",
            "operation",
            "git_commit",
            "formal_g1",
            "contract_bundle",
            "environment_manifest",
            "resource_profile_id",
        }
        common_ref_keys = ("formal_g1", "contract_bundle", "environment_manifest")
        if spec.get("operation") == RQ014_EXPORT_OPERATION:
            required = common | {
                "scene_bundles",
                "readiness_table",
                "counterpart_tracks",
                "environment_manifest",
                "created_at_utc",
            }
            ref_keys = common_ref_keys + ("readiness_table", "counterpart_tracks")
            allowed = required
            if not isinstance(spec.get("scene_bundles"), list) or len(spec["scene_bundles"]) != 8:
                raise ValueError("RQ014 declassification export requires exactly eight scene bundles")
            for index, ref in enumerate(spec["scene_bundles"]):
                if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                    raise ValueError(f"Run spec v2 scene_bundles[{index}] must contain path and sha256")
                if not isinstance(ref["sha256"], str) or not HEX64.fullmatch(ref["sha256"]):
                    raise ValueError(f"Run spec v2 scene_bundles[{index}].sha256 is malformed")
                if ref["sha256"] == "0" * 64:
                    raise ValueError(f"Run spec v2 scene_bundles[{index}].sha256 is a placeholder")
            if spec.get("resource_profile_id") != RQ014_EXPORT_RESOURCE_PROFILE:
                raise ValueError("Unknown RQ014 declassification resource profile")
            if not isinstance(spec.get("created_at_utc"), str) or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", spec["created_at_utc"]
            ) is None:
                raise ValueError("RQ014 export created_at_utc must be exact UTC seconds")
        elif spec.get("operation") == RQ014_PREFLIGHT_OPERATION:
            required = common | {
                "declassification_export_commit",
                "m3_artifact",
                "input_manifest",
                "sanitization_receipt",
                "materialization_ledger",
                "declassification_export_receipt",
                "declassification_export_done",
            }
            ref_keys = common_ref_keys + (
                "input_manifest",
                "sanitization_receipt",
                "materialization_ledger",
                "declassification_export_receipt",
                "declassification_export_done",
            )
            if spec.get("resource_profile_id") != RQ014_PREFLIGHT_RESOURCE_PROFILE:
                raise ValueError("Unknown RQ014 preflight resource profile")
            allowed = required
        else:
            raise ValueError("Unsupported RQ014 v2 operation")
        if not required <= set(spec) or not set(spec) <= allowed:
            raise ValueError(
                f"Run spec v2 keys differ; missing={sorted(required - set(spec))}, "
                f"unexpected={sorted(set(spec) - allowed)}"
            )
        for key in ref_keys:
            if not isinstance(spec[key], dict) or set(spec[key]) != {"path", "sha256"}:
                raise ValueError(f"Run spec v2 {key} must contain exactly path and sha256")
            if not isinstance(spec[key]["sha256"], str) or not HEX64.fullmatch(spec[key]["sha256"]):
                raise ValueError(f"Run spec v2 {key}.sha256 is malformed")
            if spec[key]["sha256"] == "0" * 64:
                raise ValueError(f"Run spec v2 {key}.sha256 is a placeholder")
        if "m3_artifact" in spec:
            m3_ref = spec["m3_artifact"]
            if (
                not isinstance(m3_ref, dict)
                or set(m3_ref) != {"path", "size_bytes", "sha256"}
                or m3_ref.get("path") != M3_ARTIFACT_PATH
                or m3_ref.get("size_bytes") != M3_ARTIFACT_SIZE_BYTES
                or m3_ref.get("sha256") != MODEL_SHA256
            ):
                raise ValueError("Run spec v2 m3_artifact differs from the frozen delivery binding")
        if spec["rq_id"] != "RQ014":
            raise ValueError("Run spec v2 supports only RQ014")
        if not isinstance(spec["git_commit"], str) or not HEX40.fullmatch(spec["git_commit"]):
            raise ValueError("Run spec v2 git_commit must be lowercase 40-hex")
        if spec["git_commit"] == "0" * 40:
            raise ValueError("Run spec v2 git_commit is a placeholder")
        if spec["operation"] == RQ014_PREFLIGHT_OPERATION:
            export_commit = spec["declassification_export_commit"]
            if not isinstance(export_commit, str) or not HEX40.fullmatch(export_commit):
                raise ValueError(
                    "Run spec v2 declassification_export_commit must be lowercase 40-hex"
                )
            if export_commit == "0" * 40:
                raise ValueError(
                    "Run spec v2 declassification_export_commit is a placeholder"
                )
    else:
        raise ValueError("Unsupported run spec schema")
    for key in ("rq_id", "run_id", "operation"):
        if str(spec[key]) in {".", ".."} or not SAFE_ID.fullmatch(str(spec[key])):
            raise ValueError(f"Unsafe {key}: {spec[key]}")
    return spec


def _parse_canonical_run_spec_bytes(payload: bytes) -> dict[str, Any]:
    try:
        text_payload = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Run spec must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text_payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON token: {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON run spec: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Run spec must be a JSON object")
    spec = load_spec_from_value(value)
    if payload != _canonical_spec_bytes(spec):
        raise ValueError("Production run spec must be canonical JSON")
    return spec


def _load_canonical_run_spec(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    return _parse_canonical_run_spec_bytes(payload), payload


def _load_managed_canonical_run_spec(
    path: Path,
    *,
    base: Path,
) -> tuple[dict[str, Any], bytes]:
    """Read one immutable managed production spec through one retained descriptor."""

    root = base / "manifests" / "RQ014" / "run_specs"
    lexical_path = Path(os.path.abspath(path))
    lexical_root = Path(os.path.abspath(root))
    if lexical_path.parent != lexical_root:
        raise ValueError("Production run spec must be directly inside the managed run_specs root")
    _reject_symlink_components_for_new_path(
        lexical_path,
        root=base,
        label="RQ014 production run spec",
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("O_NOFOLLOW is required for production run specs")
    try:
        descriptor = os.open(
            lexical_path,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ValueError("Cannot open production run spec without following links") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Production run spec must be a regular file")
        if stat.S_IMODE(before.st_mode) & 0o222:
            raise ValueError("Production run spec must be read-only")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Production run spec ended before its opened size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("Production run spec grew while being read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    continuity_fields = ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in continuity_fields):
        raise ValueError("Production run spec descriptor identity drift")
    payload = b"".join(chunks)
    return _parse_canonical_run_spec_bytes(payload), payload


def load_spec_from_value(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate an already parsed run-spec object without rereading a path."""

    return _validate_loaded_spec(spec)


def _canonical_spec_bytes(spec: dict[str, Any]) -> bytes:
    return (
        json.dumps(spec, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _resolve_ref(
    ref: dict[str, str],
    *,
    roots: list[Path],
    label: str,
) -> Path:
    try:
        path = require_contained_regular_file(Path(ref["path"]), roots)
    except (KeyError, OSError, RQ014ContractError) as exc:
        raise ValueError(f"Invalid {label} path: {exc}") from exc
    if sha256_file(path) != ref["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _verify_checksum_manifest(path: Path, *, repo: Path) -> dict[str, str]:
    registered: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", raw)
        if match is None:
            raise ValueError(f"Malformed checksum manifest line {line_number}: {path}")
        digest, relative = match.groups()
        if relative in registered:
            raise ValueError(f"Duplicate checksum-manifest path: {relative}")
        lexical_target = repo / relative
        _reject_symlink_components_for_new_path(
            lexical_target,
            root=repo,
            label=f"Checksum-manifest path {relative}",
        )
        target = lexical_target.resolve()
        try:
            target.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError(f"Checksum path escapes repository: {relative}") from exc
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Checksum manifest mismatch: {relative}")
        registered[relative] = digest
    if not registered:
        raise ValueError("Checksum manifest is empty")
    return registered


def _validate_formal_review(
    path: Path,
    *,
    repo: Path,
    role: str,
    reviewed_manifest_path: str,
    reviewed_manifest_sha256: str,
) -> dict[str, Any]:
    review = load_rq014_json(path)
    required = {
        "schema_version",
        "review_role",
        "reviewed_manifest_path",
        "reviewed_manifest_sha256",
        "verdict",
        "unresolved_blockers",
        "unresolved_majors",
        "findings",
        "reviewer_agent",
        "fresh_reviewer_attested",
        "rating_values_accessed",
        "hpc_jobs_submitted",
        "reviewed_at_utc",
    }
    if set(review) != required:
        raise ValueError(f"Formal review has missing or unexpected keys: {role}")
    if review["schema_version"] != "rq014-formal-review-v1p5":
        raise ValueError(f"Wrong formal-review schema: {role}")
    if review["review_role"] != role:
        raise ValueError(f"Formal-review role drift: {role}")
    if (
        review["reviewed_manifest_path"] != reviewed_manifest_path
        or review["reviewed_manifest_sha256"] != reviewed_manifest_sha256
    ):
        raise ValueError(f"Formal review binds a different candidate manifest: {role}")
    if (
        review["verdict"] != "NO_BLOCKER"
        or review["unresolved_blockers"] != 0
        or review["unresolved_majors"] != 0
    ):
        raise ValueError(f"Formal review retains a blocker or major: {role}")
    if review["fresh_reviewer_attested"] is not True:
        raise ValueError(f"Formal reviewer independence is not attested: {role}")
    if review["rating_values_accessed"] is not False or review["hpc_jobs_submitted"] is not False:
        raise ValueError(f"Formal reviewer crossed the execution boundary: {role}")
    if not isinstance(review["reviewer_agent"], str) or not review["reviewer_agent"].strip():
        raise ValueError(f"Formal reviewer identity is missing: {role}")
    if not isinstance(review["reviewed_at_utc"], str) or not UTC_SECONDS.fullmatch(
        review["reviewed_at_utc"]
    ):
        raise ValueError(f"Formal review timestamp is malformed: {role}")
    if not isinstance(review["findings"], list):
        raise ValueError(f"Formal review findings must be a list: {role}")
    for finding in review["findings"]:
        if not isinstance(finding, dict) or set(finding) != {
            "finding_id",
            "severity",
            "status",
            "summary",
            "evidence",
        }:
            raise ValueError(f"Malformed formal-review finding: {role}")
        if finding["severity"] not in {"BLOCKER", "MAJOR", "MINOR", "NOTE"}:
            raise ValueError(f"Unknown formal-review severity: {role}")
        if finding["status"] not in {"RESOLVED", "ACCEPTED_RESIDUAL", "NO_FINDING"}:
            raise ValueError(f"Unknown formal-review finding status: {role}")
        if finding["severity"] in {"BLOCKER", "MAJOR"} and finding["status"] != "RESOLVED":
            raise ValueError(f"Unresolved formal-review blocker/major: {role}")
    return review


def _validate_g0_registry_evidence(
    *,
    repo: Path,
    reviewed: dict[str, str],
) -> None:
    registry_relative = "reports/plans/RQ014_forensic_registry_v1p5.yaml"
    evidence_relative = (
        "reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/forensics_report.md"
    )
    evidence_sha256 = "e6f1ce6e136cdf7355e18e38b5853b934658d462f72e61acb3bddf282ca16c88"
    if reviewed.get(evidence_relative) != evidence_sha256:
        raise ValueError("Formal G1 review set does not bind the exact F01-F04 evidence byte")
    evidence_path = repo / evidence_relative
    _reject_symlink_components_for_new_path(
        evidence_path,
        root=repo,
        label="RQ014 F01-F04 closure evidence",
    )
    if (
        evidence_path.is_symlink()
        or not evidence_path.is_file()
        or sha256_file(evidence_path) != evidence_sha256
    ):
        raise ValueError("RQ014 F01-F04 closure evidence hash mismatch")
    registry = load_rq014_json(repo / registry_relative)
    if (
        registry.get("schema_version") != "rq014-forensic-v1p5"
        or registry.get("registry_status")
        != "G0_CLOSED_FORMAL_G1_RESOLVED_BY_EXTERNAL_ARTIFACT"
        or not isinstance(registry.get("forensic_surfaces"), list)
    ):
        raise ValueError("Malformed RQ014 forensic registry")
    surface_rows = registry["forensic_surfaces"]
    surfaces = {
        row.get("surface_id"): row
        for row in surface_rows
        if isinstance(row, dict) and isinstance(row.get("surface_id"), str)
    }
    for surface_id in ("F01", "F02", "F03", "F04"):
        if sum(
            isinstance(row, dict) and row.get("surface_id") == surface_id
            for row in surface_rows
        ) != 1:
            raise ValueError(f"RQ014 forensic registry duplicates or omits {surface_id}")
        row = surfaces.get(surface_id)
        if not isinstance(row, dict):
            raise ValueError(f"RQ014 forensic registry omits {surface_id}")
        expected_evidence = {"path": evidence_relative, "sha256": evidence_sha256}
        if (
            row.get("status") != "NOT_FOUND_ON_SCANNED_SURFACES"
            or row.get("complete_scan") is not True
            or row.get("all_required_reads_ok") is not True
            or row.get("zero_matches") is not True
            or row.get("closure_basis") != "COMPLETED_SCAN"
            or row.get("closure_evidence") != expected_evidence
            or row.get("negative_finding_claim_allowed") is not True
        ):
            raise ValueError(f"RQ014 forensic registry closure evidence drift: {surface_id}")


def _validate_formal_g1(
    path: Path,
    *,
    repo: Path,
    expected_review_manifest: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    artifact = load_rq014_json(path)
    required = {
        "schema_version",
        "status",
        "reviewed_manifest_path",
        "reviewed_manifest_sha256",
        "g0_closure_status",
        "reviewer_artifacts",
        "unresolved_blockers",
        "unresolved_majors",
        "no_rating_access_attested",
        "no_hpc_submission_attested",
        "adjudicated_at_utc",
    }
    if set(artifact) != required:
        raise ValueError("Formal G1 artifact has missing or unexpected keys")
    if artifact["schema_version"] != "rq014-formal-g1-v1p5":
        raise ValueError("Wrong formal G1 schema")
    if artifact["status"] != "FORMAL_G1_PASS":
        raise ValueError("RQ014 formal G1 is not PASS")
    if artifact["g0_closure_status"] != "CLOSED_WITH_INACCESSIBLE_SURFACES":
        raise ValueError("RQ014 G0 closure state drift")
    if artifact["unresolved_blockers"] != 0 or artifact["unresolved_majors"] != 0:
        raise ValueError("Formal G1 retains unresolved findings")
    if artifact["no_rating_access_attested"] is not True or artifact["no_hpc_submission_attested"] is not True:
        raise ValueError("Formal G1 execution-boundary attestations failed")
    if not isinstance(artifact["adjudicated_at_utc"], str) or not UTC_SECONDS.fullmatch(
        artifact["adjudicated_at_utc"]
    ):
        raise ValueError("Formal G1 adjudication timestamp is malformed")
    if artifact["reviewed_manifest_path"] != expected_review_manifest:
        raise ValueError("Formal G1 does not bind the contract-declared review manifest")
    if not isinstance(artifact["reviewed_manifest_sha256"], str) or not HEX64.fullmatch(
        artifact["reviewed_manifest_sha256"]
    ):
        raise ValueError("Formal G1 review-manifest SHA-256 is malformed")
    reviewers = artifact["reviewer_artifacts"]
    if not isinstance(reviewers, dict) or set(reviewers) != {"statistics", "execution_governance"}:
        raise ValueError("Formal G1 must bind statistics and execution/governance reviewers")
    expected_review_paths = {
        "statistics": RQ014_STATISTICS_REVIEW,
        "execution_governance": RQ014_EXECUTION_REVIEW,
    }
    for role, review in reviewers.items():
        if not isinstance(review, dict) or set(review) != {"path", "sha256", "verdict"}:
            raise ValueError(f"Malformed formal G1 reviewer artifact: {role}")
        if review["verdict"] != "NO_BLOCKER" or not HEX64.fullmatch(str(review["sha256"])):
            raise ValueError(f"Formal G1 reviewer did not return NO_BLOCKER: {role}")
        review_path = (repo / review["path"]).resolve()
        if review["path"] != expected_review_paths[role]:
            raise ValueError(f"Formal G1 reviewer path drift: {role}")
        try:
            review_path.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError(f"Reviewer artifact escapes repository: {role}") from exc
        if not review_path.is_file() or sha256_file(review_path) != review["sha256"]:
            raise ValueError(f"Reviewer artifact hash mismatch: {role}")
    reviewed_manifest = (repo / artifact["reviewed_manifest_path"]).resolve()
    try:
        reviewed_manifest.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError("Formal G1 review manifest escapes repository") from exc
    if not reviewed_manifest.is_file() or sha256_file(reviewed_manifest) != artifact["reviewed_manifest_sha256"]:
        raise ValueError("Formal G1 review manifest hash mismatch")
    reviewed = _verify_checksum_manifest(reviewed_manifest, repo=repo)
    if not RQ014_REVIEW_REQUIRED_PATHS <= set(reviewed):
        raise ValueError(
            "Formal G1 review manifest omits required bytes: "
            f"{sorted(RQ014_REVIEW_REQUIRED_PATHS - set(reviewed))}"
        )
    _validate_g0_registry_evidence(repo=repo, reviewed=reviewed)
    legacy_manifest = repo / "reports" / "plans" / "RQ014_plan_v1p3_checksums_20260711.sha256"
    legacy_registered = _verify_checksum_manifest(legacy_manifest, repo=repo)
    for relative, digest in legacy_registered.items():
        if reviewed.get(relative) != digest:
            raise ValueError(f"Formal G1 did not review inherited v1.3 byte: {relative}")
    validated_reviews: dict[str, dict[str, Any]] = {}
    for role, reference in reviewers.items():
        validated_reviews[role] = _validate_formal_review(
            repo / reference["path"],
            repo=repo,
            role=role,
            reviewed_manifest_path=artifact["reviewed_manifest_path"],
            reviewed_manifest_sha256=artifact["reviewed_manifest_sha256"],
        )
    if (
        validated_reviews["statistics"]["reviewer_agent"].strip()
        == validated_reviews["execution_governance"]["reviewer_agent"].strip()
    ):
        raise ValueError("Formal G1 statistics and execution reviewer identities must differ")
    return artifact, reviewed


def _load_rq014_source_inventory(repo: Path) -> dict[str, Any]:
    path = (
        repo
        / "reports"
        / "studies"
        / "RQ014_wod_e2e_rating_recovery"
        / "02_g2_preflight"
        / "RQ014_declassification_source_inventory_20260712.json"
    )
    inventory = load_rq014_json(path)
    required = {
        "schema_version",
        "captured_date",
        "capture_mode",
        "rating_values_read",
        "raw_rated479_tfrecord_opened",
        "execution_binding_contract",
        "files",
        "managed_interpreter",
        "managed_environment_manifest",
        "score_omitting_bundle_structural_audit",
        "provenance_code_audit",
        "next_required_artifacts",
    }
    if set(inventory) != required:
        raise ValueError("RQ014 source inventory has missing or unexpected keys")
    if inventory["schema_version"] != "rq014-declassification-source-inventory-v1":
        raise ValueError("Wrong RQ014 source-inventory schema")
    if inventory["rating_values_read"] is not False or inventory["raw_rated479_tfrecord_opened"] is not False:
        raise ValueError("RQ014 source inventory crossed the rating boundary")
    if inventory["execution_binding_contract"] != {
        "expected_metadata_source": "files[].role/path/size_bytes/sha256",
        "open_policy": "SINGLE_FD_O_RDONLY_O_NOFOLLOW_O_CLOEXEC_O_NONBLOCK",
        "continuity_fields": [
            "st_mode",
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ],
        "parse_policy": "VERIFY_EXPECTED_SIZE_AND_SHA256_THEN_PARSE_ONLY_RETAINED_BYTES",
        "receipt_policy": "RECORD_THE_SAME_RETAINED_BYTES_SHA256",
    }:
        raise ValueError("RQ014 source inventory single-descriptor binding contract drift")
    rows = inventory["files"]
    if not isinstance(rows, list):
        raise ValueError("RQ014 source-inventory files must be a list")
    expected_roles = {
        *(f"phase1_scene_bundle_{index:02d}" for index in range(8)),
        "rated479_structural_readiness",
        "selected_counterpart_tracks",
    }
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "size_bytes", "sha256"}:
            raise ValueError("Malformed RQ014 source-inventory row")
        role = row["role"]
        if role in by_role or role not in expected_roles:
            raise ValueError(f"Unexpected or duplicate RQ014 source role: {role}")
        if (
            not isinstance(row["size_bytes"], int)
            or row["size_bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or not HEX64.fullmatch(row["sha256"])
        ):
            raise ValueError(f"Malformed RQ014 source metadata: {role}")
        by_role[role] = row
    if set(by_role) != expected_roles:
        raise ValueError("RQ014 source inventory is incomplete")
    interpreter = inventory["managed_interpreter"]
    if not isinstance(interpreter, dict) or set(interpreter) != {
        "path",
        "version",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("Malformed RQ014 managed-interpreter inventory")
    return inventory


def _validate_stdlib_checksum_manifest(
    *,
    manifest_path: Path,
    stdlib_root: Path,
    expected_count: int,
    expected_total_size: int,
) -> None:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_count * 2:
        raise ValueError("RQ014 stdlib checksum-manifest row count drift")
    registered: list[Path] = []
    total_size = 0
    previous = ""
    for index in range(expected_count):
        if index and index % 500 == 0:
            print(f"rq014-stdlib-validate {index}/{expected_count}", file=sys.stderr, flush=True)
        size_line = lines[index * 2]
        checksum_line = lines[index * 2 + 1]
        size_match = re.fullmatch(r"# size_bytes=(\d+)", size_line)
        checksum_match = re.fullmatch(r"([0-9a-f]{64})  (/[^\0]+)", checksum_line)
        if size_match is None or checksum_match is None:
            raise ValueError(f"Malformed RQ014 stdlib checksum row: {index}")
        size_bytes = int(size_match.group(1))
        digest, raw_path = checksum_match.groups()
        path = Path(raw_path)
        if raw_path <= previous:
            raise ValueError("RQ014 stdlib checksum paths are not strictly sorted")
        previous = raw_path
        _reject_symlink_components_for_new_path(
            path,
            root=stdlib_root,
            label=f"RQ014 stdlib checksum path {index}",
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"RQ014 stdlib checksum path is not a regular file: {path}")
        if path.stat().st_size != size_bytes or sha256_file(path) != digest:
            raise ValueError(f"RQ014 stdlib checksum mismatch: {path}")
        total_size += size_bytes
        registered.append(path)
    actual: list[Path] = []
    for directory, dirnames, filenames in os.walk(stdlib_root, followlinks=False):
        current = Path(directory)
        for name in list(dirnames):
            candidate = current / name
            if candidate.is_symlink():
                raise ValueError(f"RQ014 stdlib tree contains a symlink: {candidate}")
        for name in filenames:
            candidate = current / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"RQ014 stdlib tree contains a non-regular entry: {candidate}")
            actual.append(candidate)
    if sorted(registered) != sorted(actual):
        raise ValueError("RQ014 stdlib checksum manifest is not the exact regular-file set")
    if total_size != expected_total_size:
        raise ValueError("RQ014 stdlib checksum total size drift")


def _validate_native_library_manifest(
    *,
    manifest_path: Path,
    environment_root: Path,
    expected_count: int,
    expected_total_size: int,
    expected_symlink_count: int,
) -> None:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    expected_headers = [
        "# rq014-managed-python-native-libs-v1",
        (
            "# columns=soname<TAB>loader_path<TAB>link_target_or_dash<TAB>"
            "resolved_path<TAB>size_bytes<TAB>sha256"
        ),
        "# discovery=ldd_python3.9_plus_every_regular_lib-dynload_so",
        f"# managed_environment_root={environment_root}",
        f"# row_count={expected_count}",
    ]
    if lines[:5] != expected_headers:
        raise ValueError("RQ014 native-library manifest header drift")
    data_lines = lines[5:]
    if len(data_lines) != expected_count:
        raise ValueError("RQ014 native-library manifest row count drift")
    previous_soname = ""
    loader_paths: set[Path] = set()
    resolved_paths: set[Path] = set()
    allowed_roots = [environment_root, Path("/lib64").resolve(strict=False)]
    total_size = 0
    symlink_count = 0
    for index, line in enumerate(data_lines):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"Malformed RQ014 native-library row: {index}")
        soname, raw_loader, link_target, raw_resolved, raw_size, digest = fields
        if (
            re.fullmatch(r"[A-Za-z0-9_.+-]+", soname) is None
            or soname <= previous_soname
            or re.fullmatch(r"[0-9]+", raw_size) is None
            or HEX64.fullmatch(digest) is None
        ):
            raise ValueError(f"Malformed or unsorted RQ014 native-library row: {index}")
        previous_soname = soname
        if os.path.normpath(raw_loader) != raw_loader or os.path.normpath(raw_resolved) != raw_resolved:
            raise ValueError(f"Non-normalized RQ014 native-library path: {index}")
        loader_path = Path(raw_loader)
        resolved_path = Path(raw_resolved)
        for label, candidate in (("loader", loader_path), ("resolved", resolved_path)):
            if not candidate.is_absolute():
                raise ValueError(f"RQ014 native {label} path is not absolute: {candidate}")
            if not any(
                _is_path_within(candidate.resolve(strict=False), root.resolve(strict=False))
                for root in allowed_roots
            ):
                raise ValueError(
                    f"RQ014 native {label} path escapes the managed/runtime trust roots: "
                    f"{candidate}"
                )
        if loader_path in loader_paths or resolved_path in resolved_paths:
            raise ValueError("Duplicate RQ014 native loader or resolved path")
        loader_paths.add(loader_path)
        resolved_paths.add(resolved_path)
        if link_target == "-":
            if loader_path.is_symlink() or not loader_path.is_file():
                raise ValueError(f"RQ014 native loader must be a regular file: {loader_path}")
        else:
            if not loader_path.is_symlink() or os.readlink(loader_path) != link_target:
                raise ValueError(f"RQ014 native loader symlink drift: {loader_path}")
            symlink_count += 1
            lexical_target = Path(
                os.path.normpath(
                    link_target
                    if os.path.isabs(link_target)
                    else str(loader_path.parent / link_target)
                )
            )
            if lexical_target.is_symlink() or not lexical_target.is_file():
                raise ValueError(
                    f"RQ014 native loader has an unbound multi-hop link: {loader_path}"
                )
            if lexical_target.resolve(strict=True) != resolved_path.resolve(strict=True):
                raise ValueError(f"RQ014 native loader has an unbound multi-hop link: {loader_path}")
        try:
            actual_resolved = loader_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"RQ014 native loader cannot resolve: {loader_path}") from exc
        if actual_resolved != resolved_path:
            raise ValueError(f"RQ014 native loader resolved target drift: {loader_path}")
        resolved = require_contained_regular_file(resolved_path, allowed_roots)
        size_bytes = int(raw_size)
        if resolved.stat().st_size != size_bytes or sha256_file(resolved) != digest:
            raise ValueError(f"RQ014 native resolved-library bytes drift: {resolved}")
        total_size += size_bytes
    if total_size != expected_total_size:
        raise ValueError("RQ014 native-library total size drift")
    if symlink_count != expected_symlink_count:
        raise ValueError("RQ014 native-library symlink count drift")


def _validate_rq014_environment_manifest(
    ref: dict[str, str],
    *,
    base: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    path = _resolve_ref(
        ref,
        roots=[base / "manifests" / "RQ014"],
        label="RQ014 managed Python environment manifest",
    )
    if path.name != "managed_python_environment_v3.json":
        raise ValueError("RQ014 environment-manifest filename drift")
    reviewed_manifest = inventory["managed_environment_manifest"]
    if (
        not isinstance(reviewed_manifest, dict)
        or set(reviewed_manifest) != {"path", "size_bytes", "sha256"}
        or path != Path(reviewed_manifest["path"]).resolve()
        or ref["sha256"] != reviewed_manifest["sha256"]
        or path.stat().st_size != reviewed_manifest["size_bytes"]
    ):
        raise ValueError("RQ014 environment manifest differs from the reviewed inventory")
    manifest = load_rq014_json(path)
    if path.read_bytes() != _canonical_spec_bytes(manifest):
        raise ValueError("RQ014 environment manifest is not canonical JSON")
    required = {
        "schema_version",
        "environment_id",
        "python_executable",
        "execution_dependencies",
        "site_packages_imported",
        "isolated_python_flags",
        "isolated_sys_path",
        "stdlib_integrity",
        "native_library_integrity",
        "captured_at_utc",
    }
    if set(manifest) != required:
        raise ValueError("RQ014 environment manifest has missing or unexpected keys")
    if (
        manifest["schema_version"] != "rq014-managed-python-environment-v3"
        or manifest["environment_id"] != "ipv-exact-sigma01"
        or manifest["execution_dependencies"]
        != "PYTHON_STANDARD_LIBRARY_PLUS_PINNED_NATIVE_CLOSURE"
        or manifest["site_packages_imported"] != []
        or manifest["isolated_python_flags"] != ["-I", "-S", "-B", "-X", "utf8"]
    ):
        raise ValueError("RQ014 environment-manifest contract drift")
    if not isinstance(manifest["captured_at_utc"], str) or not UTC_SECONDS.fullmatch(
        manifest["captured_at_utc"]
    ):
        raise ValueError("RQ014 environment-manifest timestamp is malformed")
    expected = inventory["managed_interpreter"]
    if manifest["python_executable"] != expected:
        raise ValueError("RQ014 environment manifest differs from the reviewed interpreter inventory")
    expected_path = base / "envs" / "ipv-exact-sigma01" / "bin" / "python3.9"
    python = require_contained_regular_file(
        Path(expected["path"]),
        [base / "envs" / "ipv-exact-sigma01" / "bin"],
    )
    if python != expected_path.resolve():
        raise ValueError("RQ014 managed Python executable path drift")
    if python.stat().st_size != expected["size_bytes"] or sha256_file(python) != expected["sha256"]:
        raise ValueError("RQ014 managed Python executable bytes drifted")
    version = subprocess.check_output(
        [str(python), "-I", "-S", "-B", "--version"],
        text=True,
        stderr=subprocess.STDOUT,
        env=MINIMAL_COMMAND_ENV,
    ).strip()
    if version != expected["version"]:
        raise ValueError("RQ014 managed Python version drifted")
    stdlib_root = base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9"
    zip_path = base / "envs" / "ipv-exact-sigma01" / "lib" / "python39.zip"
    lib_dynload = stdlib_root / "lib-dynload"
    expected_sys_path = [str(zip_path), str(stdlib_root), str(lib_dynload)]
    if manifest["isolated_sys_path"] != expected_sys_path:
        raise ValueError("RQ014 isolated sys.path contract drift")
    integrity = manifest["stdlib_integrity"]
    integrity_keys = {
        "stdlib_root",
        "lib_dynload_root",
        "zip_path",
        "zip_path_status",
        "symlink_count",
        "regular_file_count",
        "regular_file_total_size_bytes",
        "checksum_manifest_path",
        "checksum_manifest_size_bytes",
        "checksum_manifest_sha256",
    }
    if not isinstance(integrity, dict) or set(integrity) != integrity_keys:
        raise ValueError("Malformed RQ014 stdlib-integrity contract")
    if (
        integrity["stdlib_root"] != str(stdlib_root)
        or integrity["lib_dynload_root"] != str(lib_dynload)
        or integrity["zip_path"] != str(zip_path)
        or integrity["zip_path_status"] != "ABSENT"
        or integrity["symlink_count"] != 0
        or zip_path.exists()
        or not stdlib_root.is_dir()
        or not lib_dynload.is_dir()
    ):
        raise ValueError("RQ014 stdlib roots or zip/symlink contract drift")
    count = integrity["regular_file_count"]
    total_size = integrity["regular_file_total_size_bytes"]
    checksum_size = integrity["checksum_manifest_size_bytes"]
    checksum_sha256 = integrity["checksum_manifest_sha256"]
    checksum_raw_path = integrity["checksum_manifest_path"]
    if (
        not isinstance(count, int)
        or count <= 0
        or not isinstance(total_size, int)
        or total_size <= 0
        or not isinstance(checksum_size, int)
        or checksum_size <= 0
        or not isinstance(checksum_sha256, str)
        or HEX64.fullmatch(checksum_sha256) is None
        or not isinstance(checksum_raw_path, str)
    ):
        raise ValueError("Malformed RQ014 stdlib file count or total size")
    checksum_path = require_contained_regular_file(
        Path(checksum_raw_path),
        [base / "manifests" / "RQ014"],
    )
    if (
        checksum_path.name != "managed_python_stdlib_v1.sha256"
        or checksum_path.stat().st_size != checksum_size
        or sha256_file(checksum_path) != checksum_sha256
    ):
        raise ValueError("RQ014 stdlib checksum manifest differs from its v3 lock")
    _validate_stdlib_checksum_manifest(
        manifest_path=checksum_path,
        stdlib_root=stdlib_root,
        expected_count=count,
        expected_total_size=total_size,
    )
    native = manifest["native_library_integrity"]
    native_keys = {
        "discovery_scope",
        "columns",
        "manifest_path",
        "manifest_size_bytes",
        "manifest_sha256",
        "row_count",
        "resolved_regular_file_total_size_bytes",
        "symlink_row_count",
        "multi_hop_count",
        "system_library_trust_roots",
        "symlink_chain_policy",
    }
    expected_columns = [
        "soname",
        "loader_path",
        "link_target_or_dash",
        "resolved_path",
        "size_bytes",
        "sha256",
    ]
    if (
        not isinstance(native, dict)
        or set(native) != native_keys
        or native["discovery_scope"]
        != "ldd python3.9 plus every regular lib-dynload/*.so; complete recursive resolved closure"
        or native["columns"] != expected_columns
        or native["system_library_trust_roots"] != ["/lib64"]
        or native["symlink_chain_policy"]
        != "exact loader link target and final resolved regular-file bytes"
        or native["multi_hop_count"] != 0
    ):
        raise ValueError("Malformed RQ014 native-library integrity contract")
    native_count = native["row_count"]
    native_total_size = native["resolved_regular_file_total_size_bytes"]
    native_symlink_count = native["symlink_row_count"]
    native_manifest_size = native["manifest_size_bytes"]
    native_manifest_sha256 = native["manifest_sha256"]
    native_manifest_raw_path = native["manifest_path"]
    if (
        not isinstance(native_count, int)
        or native_count <= 0
        or not isinstance(native_total_size, int)
        or native_total_size <= 0
        or not isinstance(native_symlink_count, int)
        or native_symlink_count < 0
        or native_symlink_count > native_count
        or not isinstance(native_manifest_size, int)
        or native_manifest_size <= 0
        or not isinstance(native_manifest_sha256, str)
        or HEX64.fullmatch(native_manifest_sha256) is None
        or not isinstance(native_manifest_raw_path, str)
    ):
        raise ValueError("Malformed RQ014 native-library manifest metadata")
    native_manifest_path = require_contained_regular_file(
        Path(native_manifest_raw_path),
        [base / "manifests" / "RQ014"],
    )
    if (
        native_manifest_path.name != "managed_python_native_libs_v1.tsv"
        or native_manifest_path.stat().st_size != native_manifest_size
        or sha256_file(native_manifest_path) != native_manifest_sha256
    ):
        raise ValueError("RQ014 native-library manifest differs from its v3 lock")
    environment_root = base / "envs" / "ipv-exact-sigma01"
    _validate_native_library_manifest(
        manifest_path=native_manifest_path,
        environment_root=environment_root,
        expected_count=native_count,
        expected_total_size=native_total_size,
        expected_symlink_count=native_symlink_count,
    )
    return {
        "environment_manifest_path": str(path),
        "environment_manifest_sha256": ref["sha256"],
        "python_executable_path": str(python),
        "python_executable_sha256": expected["sha256"],
        "python_version": expected["version"],
        "isolated_sys_path": expected_sys_path,
        "stdlib_root": str(stdlib_root),
        "lib_dynload_root": str(lib_dynload),
        "python_zip_path": str(zip_path),
        "stdlib_regular_file_count": count,
        "stdlib_regular_file_total_size_bytes": total_size,
        "stdlib_checksum_manifest_path": str(checksum_path),
        "stdlib_checksum_manifest_size_bytes": checksum_size,
        "stdlib_checksum_manifest_sha256": checksum_sha256,
        "native_library_manifest_path": str(native_manifest_path),
        "native_library_manifest_size_bytes": native_manifest_size,
        "native_library_manifest_sha256": native_manifest_sha256,
        "native_library_row_count": native_count,
        "native_library_total_size_bytes": native_total_size,
        "native_library_symlink_row_count": native_symlink_count,
    }


def _validate_rq014_export_inputs(
    spec: dict[str, Any],
    *,
    base: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    legacy_root = base.parent / "RQ010B_wod_e2e"
    shard_root = (
        legacy_root
        / "reframed_pref_analysis"
        / "phase1_ipv_build"
        / "shards"
    )
    inventory_by_role = {row["role"]: row for row in inventory["files"]}
    scene_paths: list[Path] = []
    scene_metadata: dict[Path, tuple[int, str]] = {}
    for index, ref in enumerate(spec["scene_bundles"]):
        path = _resolve_ref(ref, roots=[shard_root], label="Phase-1 scene bundle")
        audited = inventory_by_role[f"phase1_scene_bundle_{index:02d}"]
        if (
            path != Path(audited["path"]).resolve()
            or ref["sha256"] != audited["sha256"]
            or path.stat().st_size != audited["size_bytes"]
        ):
            raise ValueError(f"Phase-1 scene bundle differs from reviewed inventory: shard {index}")
        scene_paths.append(path)
        scene_metadata[path] = (audited["size_bytes"], audited["sha256"])
    expected = {
        (shard_root / f"shard_{index}" / "phase1_post_scene_bundle.pkl").resolve()
        for index in range(8)
    }
    if set(scene_paths) != expected:
        raise ValueError("Scene-bundle paths must be the exact shard_0..7 Phase-1 bundle set")

    readiness_expected = (legacy_root / "manifests" / "rated479_segment_readiness.tsv").resolve()
    readiness = _resolve_ref(
        spec["readiness_table"],
        roots=[legacy_root / "manifests"],
        label="rated479 structural readiness",
    )
    if readiness != readiness_expected:
        raise ValueError("Unexpected RQ014 readiness-table path")
    audited_readiness = inventory_by_role["rated479_structural_readiness"]
    if (
        readiness != Path(audited_readiness["path"]).resolve()
        or spec["readiness_table"]["sha256"] != audited_readiness["sha256"]
        or readiness.stat().st_size != audited_readiness["size_bytes"]
    ):
        raise ValueError("RQ014 readiness table differs from reviewed inventory")

    counterpart_expected = (
        legacy_root
        / "reframed_pref_analysis"
        / "phase1_ipv_build"
        / "selected_counterpart_tracks.csv"
    ).resolve()
    counterpart = _resolve_ref(
        spec["counterpart_tracks"],
        roots=[legacy_root / "reframed_pref_analysis" / "phase1_ipv_build"],
        label="selected counterpart tracks",
    )
    if counterpart != counterpart_expected:
        raise ValueError("Unexpected RQ014 counterpart-table path")
    audited_counterpart = inventory_by_role["selected_counterpart_tracks"]
    if (
        counterpart != Path(audited_counterpart["path"]).resolve()
        or spec["counterpart_tracks"]["sha256"] != audited_counterpart["sha256"]
        or counterpart.stat().st_size != audited_counterpart["size_bytes"]
    ):
        raise ValueError("RQ014 counterpart table differs from reviewed inventory")
    output_root = base / "inputs" / "RQ014" / "wod_rated479_score_stripped" / "v1"
    _reject_symlink_components_for_new_path(
        output_root,
        root=base,
        label="RQ014 score-stripped output",
    )
    if output_root.exists():
        raise ValueError(f"Score-stripped output root already exists: {output_root}")
    ordered_scene_paths = sorted(scene_paths)
    return {
        "scene_bundle_paths": [str(path) for path in ordered_scene_paths],
        "scene_bundle_size_bytes": [scene_metadata[path][0] for path in ordered_scene_paths],
        "scene_bundle_sha256": [scene_metadata[path][1] for path in ordered_scene_paths],
        "readiness_table_path": str(readiness),
        "readiness_table_size_bytes": audited_readiness["size_bytes"],
        "readiness_table_sha256": spec["readiness_table"]["sha256"],
        "counterpart_tracks_path": str(counterpart),
        "counterpart_tracks_size_bytes": audited_counterpart["size_bytes"],
        "counterpart_tracks_sha256": spec["counterpart_tracks"]["sha256"],
        "score_stripped_output_root": str(output_root),
    }


def _with_rq014_validate_only_plan(validated: dict[str, Any]) -> dict[str, Any]:
    """Expose deterministic planning evidence without creating submit-only artifacts."""

    runtime_metadata = {
        "environment_manifest_path": validated["environment_manifest_path"],
        "environment_manifest_sha256": validated["environment_manifest_sha256"],
        "python_executable_path": validated["python_executable_path"],
        "python_executable_sha256": validated["python_executable_sha256"],
        "python_version": validated["python_version"],
        "isolated_sys_path": validated["isolated_sys_path"],
        "stdlib_checksum_manifest_sha256": validated["stdlib_checksum_manifest_sha256"],
        "native_library_manifest_sha256": validated["native_library_manifest_sha256"],
    }
    if "m3_artifact_verification" in validated:
        runtime_metadata["m3_artifact_verification"] = validated[
            "m3_artifact_verification"
        ]
    return {
        **validated,
        "code_snapshot_plan": {
            "git_commit": validated["commit"],
            "materialization_source": "EXACT_GIT_COMMIT_TREE_BLOBS",
            "file_count": len(validated["code_snapshot_files"]),
            "files": validated["code_snapshot_files"],
            "receipt_state": "SUBMIT_ONLY_NOT_CREATED_BY_VALIDATE",
        },
        "submission_plan": {
            "job_name": validated["job_name"],
            "resource_profile_id": validated["resource_profile_id"],
            "slurm_profile": validated["slurm_profile"],
            "thread_limits": validated["thread_limits"],
            "environment_export_policy": {
                "sbatch_directive": "#SBATCH --export=NIL",
                "sbatch_command_flag": "--export=NIL",
                "rendered_script_state": "SUBMIT_ONLY_NOT_CREATED_BY_VALIDATE",
            },
        },
        "runtime_metadata": runtime_metadata,
    }


def _validate_rq014_operation_contract(
    operation: Any,
    *,
    operation_name: str,
    resource_profile_id: str,
) -> None:
    if (
        not isinstance(operation, dict)
        or operation.get("status") != "CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1"
    ):
        raise ValueError("RQ014 execution contract does not conditionally authorize operation")
    if operation.get("rating_access") != "FORBIDDEN" or operation.get("rating_join") != "FORBIDDEN":
        raise ValueError("RQ014 operation rating boundary drift")
    if operation.get("required_gates") != {
        "g0": "CLOSED_WITH_INACCESSIBLE_SURFACES",
        "formal_g1": "FORMAL_G1_PASS",
    }:
        raise ValueError("RQ014 operation gate contract drift")
    if operation_name == RQ014_EXPORT_OPERATION:
        if operation.get("required_prior_receipts") != []:
            raise ValueError("RQ014 export prior-receipt contract drift")
        if operation.get("source_byte_binding") != (
            "launcher passes each reviewed role/path/size/SHA-256; exporter opens each source "
            "once with O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK, checks regular-file fstat "
            "continuity, reads exact bytes once, verifies expected size/SHA-256, parses only "
            "retained bytes, and records that same digest"
        ):
            raise ValueError("RQ014 export source-byte binding contract drift")
    elif (
        operation.get("required_prior_receipts")
        != [
            "rq014-g2-declassification-export-receipt-v1",
            "rq014-managed-operation-done-v1",
        ]
        or operation.get("required_run_spec_refs")
        != [
            "declassification_export_receipt",
            "declassification_export_done",
            "environment_manifest",
            "m3_artifact",
        ]
    ):
        raise ValueError("RQ014 preflight prior-receipt predicate drift")
    if resource_profile_id != operation.get("resource_profile_id"):
        raise ValueError("Run spec resource profile differs from operation contract")


def _validate_rq014_spec(
    spec: dict[str, Any],
    *,
    base: Path,
    repo: Path,
    authorization_path: Path,
    authorization: dict[str, Any],
    spec_path: Path | None,
    spec_bytes: bytes | None,
) -> dict[str, Any]:
    rq = authorization["authorizations"].get("RQ014")
    if not isinstance(rq, dict) or spec["operation"] not in rq.get("allowed_operations", []):
        raise ValueError(f"Operation is not authorized: {spec['rq_id']} / {spec['operation']}")
    required_authority_fields = {
        "allowed_operations",
        "decision_path",
        "preflight_decision_path",
        "formal_g1_path",
        "execution_contract_path",
    }
    if set(rq) != required_authority_fields:
        raise ValueError("RQ014 central authorization is malformed")

    commit = str(spec["git_commit"])
    if commit != run_git(repo, "rev-parse", "HEAD"):
        raise ValueError("RQ014 run commit must equal the managed-checkout HEAD")
    _require_published_commit(repo, commit, label="Run commit")
    if spec["operation"] == RQ014_PREFLIGHT_OPERATION:
        _require_published_commit(
            repo,
            spec["declassification_export_commit"],
            label="Declassification export commit",
        )

    decision_key = (
        "preflight_decision_path"
        if spec["operation"] == RQ014_PREFLIGHT_OPERATION
        else "decision_path"
    )
    decision_relative_path = rq[decision_key]
    decision_path = (repo / decision_relative_path).resolve()
    execution_contract_path = (repo / rq["execution_contract_path"]).resolve()
    if rq["formal_g1_path"] != RQ014_FORMAL_G1:
        raise ValueError("RQ014 central authorization formal-G1 path drift")
    formal_authority_path = (repo / rq["formal_g1_path"]).resolve()
    for label, path in (
        ("scoped decision", decision_path),
        ("execution contract", execution_contract_path),
        ("formal G1", formal_authority_path),
    ):
        try:
            path.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError(f"RQ014 {label} escapes repository") from exc
        if not path.is_file():
            raise ValueError(f"RQ014 {label} is missing")

    repo_roots = [repo]
    contract = load_rq014_json(execution_contract_path)
    gate_contract = contract.get("gate_contract", {})
    if gate_contract.get("formal_g1_source") != RQ014_FORMAL_G1:
        raise ValueError("RQ014 execution contract formal-G1 path drift")
    expected_review_manifest = gate_contract.get("formal_g1_review_manifest")
    if expected_review_manifest != RQ014_REVIEW_MANIFEST:
        raise ValueError("RQ014 execution contract review-manifest path drift")
    formal_path = _resolve_ref(spec["formal_g1"], roots=repo_roots, label="formal G1")
    if formal_path != formal_authority_path:
        raise ValueError("Run spec formal G1 differs from central authorization")
    formal_g1, reviewed = _validate_formal_g1(
        formal_path,
        repo=repo,
        expected_review_manifest=expected_review_manifest,
    )
    if decision_relative_path not in reviewed:
        raise ValueError("RQ014 scoped decision is absent from the formally reviewed bytes")

    bundle_path = _resolve_ref(spec["contract_bundle"], roots=repo_roots, label="contract bundle")
    bundle_relative = str(bundle_path.relative_to(repo))
    if bundle_relative != RQ014_FINAL_BUNDLE:
        raise ValueError("RQ014 final contract-bundle path drift")
    registered = _verify_checksum_manifest(bundle_path, repo=repo)
    minimum_bundle_paths = set(reviewed) | {
        rq["formal_g1_path"],
        expected_review_manifest,
        RQ014_STATISTICS_REVIEW,
        RQ014_EXECUTION_REVIEW,
    }
    if not minimum_bundle_paths <= set(registered):
        raise ValueError(f"Contract bundle omits required paths: {sorted(minimum_bundle_paths - set(registered))}")
    for relative, digest in reviewed.items():
        if registered.get(relative) != digest:
            raise ValueError(f"Executed contract byte differs from the formally reviewed byte: {relative}")
    if bundle_relative in registered:
        raise ValueError("Contract bundle must not recursively register itself")
    code_snapshot_files = {**registered, bundle_relative: spec["contract_bundle"]["sha256"]}

    operation = contract["authorization"]["registered_operations"].get(spec["operation"])
    _validate_rq014_operation_contract(
        operation,
        operation_name=spec["operation"],
        resource_profile_id=spec["resource_profile_id"],
    )
    m3_verification = None
    if spec["operation"] == RQ014_PREFLIGHT_OPERATION:
        try:
            m3_verification = validate_m3_artifact_ref(
                spec["m3_artifact"],
                base=base,
                contract=contract,
            )
        except RQ014ContractError as exc:
            raise ValueError(str(exc)) from exc
    inventory = _load_rq014_source_inventory(repo)
    managed_contract = contract["managed_hpc_contract"]
    if (
        managed_contract.get("base") != str(base)
        or managed_contract.get("environment_manifest_schema")
        != "configs/run_specs/rq014_managed_python_environment_v3.schema.json"
        or managed_contract.get("environment_manifest_path")
        != inventory["managed_environment_manifest"]["path"]
        or managed_contract.get("environment_manifest_size_bytes")
        != inventory["managed_environment_manifest"]["size_bytes"]
        or managed_contract.get("environment_manifest_sha256")
        != inventory["managed_environment_manifest"]["sha256"]
        or managed_contract.get("python_executable_sha256")
        != inventory["managed_interpreter"]["sha256"]
    ):
        raise ValueError("RQ014 managed-HPC contract differs from reviewed environment inventory")
    environment = _validate_rq014_environment_manifest(
        spec["environment_manifest"],
        base=base,
        inventory=inventory,
    )
    stdlib_contract = managed_contract.get("stdlib_integrity")
    expected_stdlib_contract = {
        "root": environment["stdlib_root"],
        "zip_path_status": "python39.zip ABSENT",
        "symlink_count": 0,
        "regular_file_count": environment["stdlib_regular_file_count"],
        "regular_file_total_size_bytes": environment[
            "stdlib_regular_file_total_size_bytes"
        ],
        "checksum_manifest_path": environment["stdlib_checksum_manifest_path"],
        "checksum_manifest_size_bytes": environment[
            "stdlib_checksum_manifest_size_bytes"
        ],
        "checksum_manifest_sha256": environment["stdlib_checksum_manifest_sha256"],
    }
    if stdlib_contract != expected_stdlib_contract:
        raise ValueError("RQ014 execution contract differs from the verified stdlib closure")
    native_contract = managed_contract.get("native_library_integrity")
    expected_native_contract = {
        "manifest_path": environment["native_library_manifest_path"],
        "manifest_size_bytes": environment["native_library_manifest_size_bytes"],
        "manifest_sha256": environment["native_library_manifest_sha256"],
        "row_count": environment["native_library_row_count"],
        "resolved_regular_file_total_size_bytes": environment[
            "native_library_total_size_bytes"
        ],
        "symlink_row_count": environment["native_library_symlink_row_count"],
        "multi_hop_count": 0,
        "system_library_trust_roots": ["/lib64"],
    }
    if native_contract != expected_native_contract:
        raise ValueError(
            "RQ014 execution contract differs from the verified native-library closure"
        )
    if (
        managed_contract.get("slurm_environment_export")
        != "NIL_ON_DIRECTIVE_AND_SBATCH_COMMAND; NONE_IS_FORBIDDEN_BECAUSE_IT_CAN_RECONSTRUCT_THE_LOGIN_ENVIRONMENT"
        or managed_contract.get("python_startup_flags") != ["-I", "-S", "-B", "-X", "utf8"]
        or managed_contract.get("wrapper_capability_contract")
        != RQ014_WRAPPER_CAPABILITY_CONTRACT
        or managed_contract.get("python_import_surface") != RQ014_PYTHON_IMPORT_SURFACE
        or managed_contract.get("production_spec_root")
        != str(base / "manifests" / "RQ014" / "run_specs")
        or managed_contract.get("production_spec_contract")
        != (
            "direct-child regular non-symlink read-only canonical JSON; one O_NOFOLLOW "
            "descriptor read with retained exact bytes passed through validation, sealing and "
            "submission; no path reopen"
        )
        or managed_contract.get("validate_only_contract")
        != (
            "side-effect-free output contains rq/run/operation, exact code_snapshot_files and "
            "commit-blob materialization plan, M3 artifact path/size/SHA-256 and retained-"
            "descriptor verification evidence, job/resource/thread plan and pinned runtime "
            "metadata; code_snapshot receipt, immutable M3 input receipt and rendered NIL "
            "sbatch script are submit-only artifacts"
        )
        or "Git worktree checkout" not in str(managed_contract.get("code_execution_surface"))
        or not str(managed_contract.get("code_execution_surface")).endswith("are forbidden")
    ):
        raise ValueError("RQ014 managed execution boundary contract drift")

    if spec_bytes is not None:
        spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    elif spec_path is not None:
        spec_sha256 = sha256_file(spec_path)
    else:
        spec_sha256 = hashlib.sha256(_canonical_spec_bytes(spec)).hexdigest()
    run_namespace = (base / "work_dirs" / "RQ014").resolve()
    run_root = (run_namespace / spec["run_id"]).resolve()
    if run_root.parent != run_namespace:
        raise ValueError("RQ014 run root escapes its managed namespace")
    _reject_symlink_components_for_new_path(
        base / "work_dirs" / "RQ014" / spec["run_id"],
        root=base,
        label="RQ014 run root",
    )
    common_validated = {
        **environment,
        "rq_id": spec["rq_id"],
        "run_id": spec["run_id"],
        "operation": spec["operation"],
        "authorization_sha256": sha256_file(authorization_path),
        "run_spec_sha256": spec_sha256,
        "run_spec_semantic_sha256": hashlib.sha256(_canonical_spec_bytes(spec)).hexdigest(),
        "run_root": str(run_root),
        "commit": commit,
        "formal_g1_path": str(formal_path),
        "formal_g1_relative_path": rq["formal_g1_path"],
        "formal_g1_sha256": spec["formal_g1"]["sha256"],
        "formal_g1_review_manifest_sha256": formal_g1["reviewed_manifest_sha256"],
        "contract_bundle_path": str(bundle_path),
        "contract_bundle_relative_path": bundle_relative,
        "contract_bundle_sha256": spec["contract_bundle"]["sha256"],
        "code_snapshot_files": code_snapshot_files,
        "execution_contract_path": str(execution_contract_path),
        "execution_contract_sha256": sha256_file(execution_contract_path),
        "rating_access": "FORBIDDEN",
    }
    if m3_verification is not None:
        common_validated.update(
            {
                "m3_artifact_path": m3_verification["path"],
                "m3_artifact_size_bytes": m3_verification["size_bytes"],
                "m3_artifact_sha256": m3_verification["sha256"],
                "m3_artifact_verification": m3_verification,
            }
        )
    if spec["operation"] == RQ014_EXPORT_OPERATION:
        export_inputs = _validate_rq014_export_inputs(
            spec,
            base=base,
            inventory=inventory,
        )
        return _with_rq014_validate_only_plan({
            **common_validated,
            **export_inputs,
            "resource_profile_id": RQ014_EXPORT_RESOURCE_PROFILE,
            "job_name": f"zxc-rq014-export-{spec_sha256[:12]}",
            "entrypoint": "scripts/rq014/export_score_stripped_bundle.py",
            "fixed_subcommand": None,
            "created_at_utc": spec["created_at_utc"],
            "slurm_profile": {
                "partition": "amd",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 1,
                "memory": "8G",
                "time": "02:00:00",
            },
            "thread_limits": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
            },
        })

    input_path = _resolve_ref(
        spec["input_manifest"],
        roots=[base / "manifests" / "RQ014"],
        label="G2 input manifest",
    )
    sanitization_path = _resolve_ref(
        spec["sanitization_receipt"],
        roots=[base / "inputs" / "RQ014"],
        label="score-stripped sanitization receipt",
    )
    ledger_path = _resolve_ref(
        spec["materialization_ledger"],
        roots=[base / "manifests" / "RQ014"],
        label="registry materialization ledger",
    )
    export_receipt_path = _resolve_ref(
        spec["declassification_export_receipt"],
        roots=[base / "work_dirs" / "RQ014"],
        label="declassification export receipt",
    )
    export_done_path = _resolve_ref(
        spec["declassification_export_done"],
        roots=[base / "work_dirs" / "RQ014"],
        label="declassification export DONE receipt",
    )
    roles = validate_g2_input_roles(
        manifest_path=input_path,
        contract=contract,
        base=base,
    )
    if roles["wod_score_stripped_sanitization_receipt"] != sanitization_path:
        raise ValueError("Run spec and G2 manifest bind different sanitization receipts")
    path_mapping = validate_wod_path_type_mapping_manifest(
        roles["wod_path_type_mapping_manifest"],
        mapping_root=base / "inputs" / "RQ014" / "wod_path_type_mapping" / "v1",
    )
    bundle_manifest = roles["wod_score_stripped_bundle_manifest"]
    bundle_root = base / "inputs" / "RQ014" / "wod_rated479_score_stripped" / "v1"
    export_receipt = validate_declassification_export_receipts(
        export_receipt_path=export_receipt_path,
        done_receipt_path=export_done_path,
        sanitization_receipt_path=sanitization_path,
        file_manifest_path=bundle_manifest,
        expected_bundle_root=bundle_root,
    )
    expected_sources = {row["role"]: row["sha256"] for row in inventory["files"]}
    bundle = validate_score_stripped_bundle(
        bundle_root=bundle_root,
        schema_path=repo / "reports" / "plans" / "RQ014_score_stripped_schema_v1.json",
        file_manifest_path=bundle_manifest,
        receipt_path=sanitization_path,
        full_hash=True,
        expected_exporter_git_commit=spec["declassification_export_commit"],
        expected_exporter_environment_sha256=environment["environment_manifest_sha256"],
        expected_exporter_code_sha256=sha256_file(
            repo / "scripts" / "rq014" / "export_score_stripped_bundle.py"
        ),
        expected_source_artifacts=expected_sources,
    )
    if export_receipt["geometry_available_scene_count"] != bundle["geometry_available_scene_count"]:
        raise ValueError("Export receipt and score-stripped bundle geometry counts differ")
    validate_materialization_ledger(ledger_path=ledger_path, repo_root=repo, contract=contract)

    job_name = f"zxc-rq014-pre-{spec_sha256[:12]}"
    return _with_rq014_validate_only_plan({
        **common_validated,
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": spec["input_manifest"]["sha256"],
        "sanitization_receipt_path": str(sanitization_path),
        "sanitization_receipt_sha256": spec["sanitization_receipt"]["sha256"],
        "materialization_ledger_path": str(ledger_path),
        "materialization_ledger_sha256": spec["materialization_ledger"]["sha256"],
        "declassification_export_receipt_path": str(export_receipt_path),
        "declassification_export_receipt_sha256": spec["declassification_export_receipt"][
            "sha256"
        ],
        "declassification_export_done_path": str(export_done_path),
        "declassification_export_done_sha256": spec["declassification_export_done"]["sha256"],
        "declassification_export_commit": spec["declassification_export_commit"],
        "wod_path_type_mapping": path_mapping,
        "resource_profile_id": RQ014_PREFLIGHT_RESOURCE_PROFILE,
        "job_name": job_name,
        "entrypoint": "scripts/rq014/run_managed_g2.py",
        "fixed_subcommand": "contract-preflight",
        "slurm_profile": {
            "partition": "amd",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 2,
            "memory": "4G",
            "time": "01:00:00",
        },
        "thread_limits": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
    })


def validate_spec(
    spec: dict,
    *,
    base: Path,
    repo: Path,
    spec_path: Path | None = None,
    spec_bytes: bytes | None = None,
) -> dict:
    version = int(spec["schema_version"])
    if spec["rq_id"] == "RQ014" and version != 2:
        raise ValueError("RQ014 runs require the fail-closed schema v2 launcher path")
    if version == 2 and base.resolve() != DEFAULT_BASE.resolve():
        raise ValueError("RQ014 v2 runs require the fixed managed HPC base")
    authorization_path = repo / "configs" / "research_authorization.json"
    authorization = _strict_json_load(authorization_path)
    rq = authorization["authorizations"].get(spec["rq_id"])
    if rq is None or spec["operation"] not in rq["allowed_operations"]:
        raise ValueError(f"Operation is not authorized: {spec['rq_id']} / {spec['operation']}")

    if version == 2:
        return _validate_rq014_spec(
            spec,
            base=base,
            repo=repo,
            authorization_path=authorization_path,
            authorization=authorization,
            spec_path=spec_path,
            spec_bytes=spec_bytes,
        )

    commit = str(spec["git_commit"])
    subprocess.run(
        _git_command(repo, "cat-file", "-e", f"{commit}^{{commit}}"),
        check=True,
        env=GIT_COMMAND_ENV,
    )
    published = subprocess.run(
        _git_command(
            repo,
            "merge-base",
            "--is-ancestor",
            commit,
            "refs/remotes/origin/main",
        ),
        env=GIT_COMMAND_ENV,
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


_RQ014_EXACT_PATH_BOOTSTRAP = """\
import importlib.util
import json
import sys
import types

expected_sys_path_json, materializer_path, preflight_path, entrypoint, *arguments = sys.argv[1:]
if sys.path != json.loads(expected_sys_path_json):
    raise RuntimeError("Managed isolated sys.path differs from the reviewed runtime closure")
scripts_package = types.ModuleType("scripts")
scripts_package.__path__ = ()
scripts_package.__package__ = "scripts"
rq014_package = types.ModuleType("scripts.rq014")
rq014_package.__path__ = ()
rq014_package.__package__ = "scripts.rq014"
scripts_package.rq014 = rq014_package
sys.modules["scripts"] = scripts_package
sys.modules["scripts.rq014"] = rq014_package

materializer_spec = importlib.util.spec_from_file_location(
    "scripts.rq014.materialize_registry", materializer_path
)
if materializer_spec is None or materializer_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path registry materializer loader")
materializer_module = importlib.util.module_from_spec(materializer_spec)
rq014_package.materialize_registry = materializer_module
sys.modules["scripts.rq014.materialize_registry"] = materializer_module
materializer_spec.loader.exec_module(materializer_module)

preflight_spec = importlib.util.spec_from_file_location(
    "scripts.rq014.preflight", preflight_path
)
if preflight_spec is None or preflight_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path preflight loader")
preflight_module = importlib.util.module_from_spec(preflight_spec)
rq014_package.preflight = preflight_module
sys.modules["scripts.rq014.preflight"] = preflight_module
preflight_spec.loader.exec_module(preflight_module)

sys.argv = [entrypoint, *arguments]
entrypoint_spec = importlib.util.spec_from_file_location("__main__", entrypoint)
if entrypoint_spec is None or entrypoint_spec.loader is None:
    raise RuntimeError("Cannot construct exact-path entrypoint loader")
entrypoint_module = importlib.util.module_from_spec(entrypoint_spec)
sys.modules["__main__"] = entrypoint_module
entrypoint_spec.loader.exec_module(entrypoint_module)
"""


def _rq014_isolated_python_command(
    *,
    python: Path,
    code: Path,
    entrypoint: str,
    arguments: list[str],
    isolated_sys_path: list[str],
) -> list[str]:
    allowed = {
        "scripts/rq014/export_score_stripped_bundle.py",
        "scripts/rq014/run_managed_g2.py",
    }
    if entrypoint not in allowed:
        raise ValueError(f"Unreviewed RQ014 entrypoint: {entrypoint}")
    return [
        str(python),
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        "-c",
        _RQ014_EXACT_PATH_BOOTSTRAP,
        json.dumps(isolated_sys_path, separators=(",", ":")),
        str(code / "scripts" / "rq014" / "materialize_registry.py"),
        str(code / "scripts" / "rq014" / "preflight.py"),
        str(code / entrypoint),
        *arguments,
    ]


def _shell_digest_check(path: str | Path, digest: str) -> str:
    quoted_path = shlex.quote(str(path))
    awk_program = shlex.quote("{print $1}")
    return f'test "$({SYSTEM_SHA256SUM} {quoted_path} | {SYSTEM_AWK} {awk_program})" = {shlex.quote(digest)}\n'


def _stdlib_shell_checks(validated: dict[str, Any]) -> str:
    root = shlex.quote(validated["stdlib_root"])
    lib_dynload = shlex.quote(validated["lib_dynload_root"])
    zip_path = shlex.quote(validated["python_zip_path"])
    checksum_manifest = shlex.quote(validated["stdlib_checksum_manifest_path"])
    count_program = shlex.quote("END { print NR + 0 }")
    size_program = shlex.quote('{ total += $1 } END { printf "%.0f", total + 0 }')
    progress_program = shlex.quote(
        '/FAILED/ { print; fflush() } NR % 500 == 0 { print "stdlib-check", NR; fflush() }'
    )
    return (
        _shell_digest_check(
            validated["stdlib_checksum_manifest_path"],
            validated["stdlib_checksum_manifest_sha256"],
        )
        + f"test -d {root}\n"
        + f"test -d {lib_dynload}\n"
        + f"test ! -e {zip_path}\n"
        + f"test -z \"$({SYSTEM_FIND} {root} -type l -print -quit)\"\n"
        + f"test -z \"$({SYSTEM_FIND} {root} ! -type d ! -type f -print -quit)\"\n"
        + f"test \"$({SYSTEM_FIND} {root} -type f -printf 'x\\n' | {SYSTEM_AWK} {count_program})\" = {validated['stdlib_regular_file_count']}\n"
        + f"test \"$({SYSTEM_FIND} {root} -type f -printf '%s\\n' | {SYSTEM_AWK} {size_program})\" = {validated['stdlib_regular_file_total_size_bytes']}\n"
        + f"(cd / && {SYSTEM_SHA256SUM} --check --strict {checksum_manifest} | {SYSTEM_AWK} {progress_program})\n"
    )


def _native_library_shell_checks(validated: dict[str, Any]) -> str:
    manifest = shlex.quote(validated["native_library_manifest_path"])
    environment_root_header = shlex.quote(
        "# managed_environment_root="
        + str(Path(validated["python_executable_path"]).parents[1])
    )
    return (
        _shell_digest_check(
            validated["native_library_manifest_path"],
            validated["native_library_manifest_sha256"],
        )
        + "native_rows=0\n"
        + "native_symlinks=0\n"
        + "native_total=0\n"
        + f"native_tab=$({SYSTEM_ENV} -i PATH={MINIMAL_PATH} {SYSTEM_AWK} 'BEGIN {{ printf \"\\t\" }}')\n"
        + "{\n"
        + "  IFS= read -r native_header\n"
        + "  test \"$native_header\" = '# rq014-managed-python-native-libs-v1'\n"
        + "  IFS= read -r native_header\n"
        + "  test \"$native_header\" = '# columns=soname<TAB>loader_path<TAB>link_target_or_dash<TAB>resolved_path<TAB>size_bytes<TAB>sha256'\n"
        + "  IFS= read -r native_header\n"
        + "  test \"$native_header\" = '# discovery=ldd_python3.9_plus_every_regular_lib-dynload_so'\n"
        + "  IFS= read -r native_header\n"
        + f"  test \"$native_header\" = {environment_root_header}\n"
        + "  IFS= read -r native_header\n"
        + f"  test \"$native_header\" = '# row_count={validated['native_library_row_count']}'\n"
        + "  while IFS=\"$native_tab\" read -r soname loader_path link_target resolved_path size_bytes digest extra; do\n"
        + "  test -n \"$soname\"\n"
        + "  test -z \"${extra-}\"\n"
        + "  if test \"$link_target\" = -; then\n"
        + "    test ! -L \"$loader_path\"\n"
        + "    test -f \"$loader_path\"\n"
        + "  else\n"
        + "    test -L \"$loader_path\"\n"
        + f"    test \"$({SYSTEM_READLINK} \"$loader_path\")\" = \"$link_target\"\n"
        + "    case \"$link_target\" in /*) lexical_target=$link_target ;; *) lexical_target=${loader_path%/*}/$link_target ;; esac\n"
        + "    test ! -L \"$lexical_target\"\n"
        + "    test -f \"$lexical_target\"\n"
        + f"    test \"$({SYSTEM_READLINK} -f \"$lexical_target\")\" = \"$({SYSTEM_READLINK} -f \"$resolved_path\")\"\n"
        + "    native_symlinks=$((native_symlinks + 1))\n"
        + "  fi\n"
        + f"  test \"$({SYSTEM_READLINK} -f \"$loader_path\")\" = \"$resolved_path\"\n"
        + "  test ! -L \"$resolved_path\"\n"
        + "  test -f \"$resolved_path\"\n"
        + f"  test \"$({SYSTEM_STAT} -c %s \"$resolved_path\")\" = \"$size_bytes\"\n"
        + f"  test \"$({SYSTEM_SHA256SUM} \"$resolved_path\" | {SYSTEM_AWK} '{{print $1}}')\" = \"$digest\"\n"
        + "  native_rows=$((native_rows + 1))\n"
        + "  native_total=$((native_total + size_bytes))\n"
        + "  done\n"
        + f"}} < {manifest}\n"
        + f"test \"$native_rows\" = {validated['native_library_row_count']}\n"
        + f"test \"$native_symlinks\" = {validated['native_library_symlink_row_count']}\n"
        + f"test \"$native_total\" = {validated['native_library_total_size_bytes']}\n"
    )


def render_rq014_sbatch(
    *,
    validated: dict[str, Any],
    base: Path,
    repo: Path,
    run_root: Path,
    code: Path,
    sealed_spec_path: Path,
) -> str:
    python = Path(validated["python_executable_path"])
    if validated["entrypoint"] == "scripts/rq014/export_score_stripped_bundle.py":
        arguments: list[str] = []
        for path in validated["scene_bundle_paths"]:
            arguments.extend(["--scene-bundle", path])
        for index, (size_bytes, digest) in enumerate(
            zip(validated["scene_bundle_size_bytes"], validated["scene_bundle_sha256"])
        ):
            arguments.extend(
                [
                    "--source-expectation",
                    f"phase1_scene_bundle_{index:02d}",
                    str(size_bytes),
                    digest,
                ]
            )
        arguments.extend(
            [
                "--source-expectation",
                "rated479_structural_readiness",
                str(validated["readiness_table_size_bytes"]),
                validated["readiness_table_sha256"],
                "--source-expectation",
                "selected_counterpart_tracks",
                str(validated["counterpart_tracks_size_bytes"]),
                validated["counterpart_tracks_sha256"],
            ]
        )
        arguments.extend(
            [
                "--readiness-tsv",
                validated["readiness_table_path"],
                "--counterpart-tracks",
                validated["counterpart_tracks_path"],
                "--schema",
                str(code / "reports" / "plans" / "RQ014_score_stripped_schema_v1.json"),
                "--output-root",
                validated["score_stripped_output_root"],
                "--run-receipt-root",
                str(run_root / "outputs"),
                "--exporter-git-commit",
                validated["commit"],
                "--exporter-environment-sha256",
                validated["environment_manifest_sha256"],
                "--created-at-utc",
                validated["created_at_utc"],
            ]
        )
        source_checks = "".join(
            _shell_digest_check(path, digest)
            for path, digest in zip(
                validated["scene_bundle_paths"], validated["scene_bundle_sha256"]
            )
        )
        source_checks += (
            _shell_digest_check(
                validated["readiness_table_path"], validated["readiness_table_sha256"]
            )
            + _shell_digest_check(
                validated["counterpart_tracks_path"],
                validated["counterpart_tracks_sha256"],
            )
            + f"test ! -e {shlex.quote(validated['score_stripped_output_root'])}\n"
        )
    else:
        execution_contract = code / "reports" / "plans" / "RQ014_execution_contract_v1p5.json"
        arguments = [
            "contract-preflight",
            "--base",
            str(base),
            "--repo-root",
            str(code),
            "--execution-contract",
            str(execution_contract),
            "--m3-artifact",
            validated["m3_artifact_path"],
            "--m3-artifact-size-bytes",
            str(validated["m3_artifact_size_bytes"]),
            "--m3-artifact-sha256",
            validated["m3_artifact_sha256"],
            "--input-manifest",
            validated["input_manifest_path"],
            "--sanitization-receipt",
            validated["sanitization_receipt_path"],
            "--materialization-ledger",
            validated["materialization_ledger_path"],
            "--declassification-export-receipt",
            validated["declassification_export_receipt_path"],
            "--declassification-export-done",
            validated["declassification_export_done_path"],
            "--expected-exporter-git-commit",
            validated["declassification_export_commit"],
            "--expected-exporter-environment-sha256",
            validated["environment_manifest_sha256"],
            "--output-root",
            str(run_root / "outputs"),
        ]
        m3_path = shlex.quote(validated["m3_artifact_path"])
        source_checks = (
            f"test ! -L {m3_path}\n"
            f"test -f {m3_path}\n"
            f'test "$({SYSTEM_READLINK} -f {m3_path})" = {m3_path}\n'
            f'test "$({SYSTEM_STAT} -c %s {m3_path})" = '
            f"{validated['m3_artifact_size_bytes']}\n"
            + _shell_digest_check(
                validated["m3_artifact_path"], validated["m3_artifact_sha256"]
            )
            + _shell_digest_check(
                validated["input_manifest_path"], validated["input_manifest_sha256"]
            )
            + _shell_digest_check(
                validated["sanitization_receipt_path"],
                validated["sanitization_receipt_sha256"],
            )
            + _shell_digest_check(
                validated["materialization_ledger_path"],
                validated["materialization_ledger_sha256"],
            )
            + _shell_digest_check(
                validated["declassification_export_receipt_path"],
                validated["declassification_export_receipt_sha256"],
            )
            + _shell_digest_check(
                validated["declassification_export_done_path"],
                validated["declassification_export_done_sha256"],
            )
        )
    source_checks += (
        _shell_digest_check(
            validated["environment_manifest_path"],
            validated["environment_manifest_sha256"],
        )
        + _shell_digest_check(
            validated["python_executable_path"], validated["python_executable_sha256"]
        )
    )
    python_command = _rq014_isolated_python_command(
        python=python,
        code=code,
        entrypoint=validated["entrypoint"],
        arguments=arguments,
        isolated_sys_path=validated["isolated_sys_path"],
    )
    runtime_environment = [
        SYSTEM_ENV,
        "-i",
        f"PATH={MINIMAL_PATH}",
        "LANG=C",
        "LC_ALL=C",
        f"SOCIALITY_PRODUCTION_RUN_ROOT={run_root}",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "BLIS_NUM_THREADS=1",
    ]
    quoted = " ".join(shlex.quote(item) for item in [*runtime_environment, *python_command])
    contract_bundle_relative = validated["contract_bundle_relative_path"]
    profile = validated["slurm_profile"]
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={validated['job_name']}\n"
        f"#SBATCH --partition={profile['partition']}\n"
        f"#SBATCH --nodes={profile['nodes']}\n"
        f"#SBATCH --ntasks={profile['ntasks']}\n"
        f"#SBATCH --cpus-per-task={profile['cpus_per_task']}\n"
        f"#SBATCH --mem={profile['memory']}\n"
        f"#SBATCH --time={profile['time']}\n"
        f"#SBATCH --output={run_root}/logs/slurm_%j.out\n"
        f"#SBATCH --error={run_root}/logs/slurm_%j.err\n"
        "#SBATCH --export=NIL\n"
        "#SBATCH --chdir=/\n\n"
        "set -euo pipefail\n"
        "umask 077\n"
        "test -z \"${BASH_ENV-}${ENV-}${LD_PRELOAD-}${PYTHONHOME-}${PYTHONPATH-}\"\n"
        f"test ! -L {shlex.quote(str(base / 'manifests' / 'runtime_maintenance.lock'))}\n"
        f"exec 8>{shlex.quote(str(base / 'manifests' / 'runtime_maintenance.lock'))}\n"
        f"{SYSTEM_FLOCK} -s 8\n"
        f"{_shell_digest_check(sealed_spec_path, validated['run_spec_sha256'])}"
        f"{_shell_digest_check(run_root / 'manifests' / 'code_snapshot.json', validated['code_snapshot_receipt_sha256'])}"
        f"{_shell_digest_check(code / 'configs' / 'research_authorization.json', validated['authorization_sha256'])}"
        f"{_shell_digest_check(code / 'reports' / 'plans' / 'RQ014_execution_contract_v1p5.json', validated['execution_contract_sha256'])}"
        f"{_shell_digest_check(code / validated['formal_g1_relative_path'], validated['formal_g1_sha256'])}"
        f"{source_checks}"
        f"{_stdlib_shell_checks(validated)}"
        f"{_native_library_shell_checks(validated)}"
        f"{_shell_digest_check(code / contract_bundle_relative, validated['contract_bundle_sha256'])}"
        f"(cd {shlex.quote(str(code))} && {SYSTEM_SHA256SUM} -c {shlex.quote(contract_bundle_relative)})\n"
        f"exec {quoted}\n"
    )


def _write_once_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different receipt: {path}")
    path.chmod(0o444)


def _materialize_rq014_code_snapshot(
    *,
    repo: Path,
    code: Path,
    commit: str,
    registered: dict[str, str],
) -> bytes:
    if code.exists():
        raise ValueError(f"Code snapshot path already exists: {code}")
    code.mkdir(parents=False, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for relative, expected_digest in sorted(registered.items()):
        payload, git_mode = _read_git_commit_regular_blob(repo, commit, relative)
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"RQ014 commit-tree blob digest drift: {relative}")
        destination = code / relative
        _reject_symlink_components_for_new_path(
            destination,
            root=code,
            label=f"RQ014 code-snapshot destination {relative}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
        destination.chmod(0o555 if git_mode == "100755" else 0o444)
        rows.append(
            {
                "git_mode": git_mode,
                "path": relative,
                "size_bytes": len(payload),
                "sha256": actual_digest,
            }
        )
    return _canonical_spec_bytes(
        {
            "schema_version": "rq014-code-snapshot-v2",
            "materialization_source": "EXACT_GIT_COMMIT_TREE_BLOBS",
            "git_commit": commit,
            "files": rows,
        }
    )


def _rollback_unsubmitted_rq014_run(
    *,
    run_root: Path,
) -> bool:
    try:
        if run_root.exists():
            shutil.rmtree(run_root)
        return not run_root.exists()
    except OSError:
        return False


def _record_rq014_submission_failure(
    *,
    spec: dict[str, Any],
    validated: dict[str, Any],
    base: Path,
    run_root: Path,
    phase: str,
    error: Exception,
    submission_started: bool,
    cleanup_complete: bool,
    known_job_id: str | None,
    sbatch_response: str | None,
) -> None:
    retry_authorized = not submission_started and cleanup_complete
    payload = {
        "schema_version": "rq014-submission-failure-v1",
        "status": "FAILED",
        "submission_state": (
            "FAILED_BEFORE_SUBMISSION"
            if not submission_started
            else "SUBMISSION_STATE_UNKNOWN"
        ),
        "phase": phase,
        "run_id": spec["run_id"],
        "operation": spec["operation"],
        "run_spec_sha256": validated["run_spec_sha256"],
        "git_commit": validated["commit"],
        "run_root": str(run_root),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "cleanup_complete": cleanup_complete,
        "retry_authorized": retry_authorized,
        "known_job_id": known_job_id,
        "sbatch_response": sbatch_response,
    }
    encoded = _canonical_spec_bytes(payload)
    receipt_digest = hashlib.sha256(encoded).hexdigest()
    failure_root = base / "manifests" / "RQ014" / "submission_failures"
    failure_path = failure_root / (
        f"{spec['run_id']}.{validated['run_spec_sha256'][:12]}.{receipt_digest[:12]}.json"
    )
    _reject_symlink_components_for_new_path(
        failure_path,
        root=base,
        label="RQ014 submission-failure receipt",
    )
    _write_once_bytes(failure_path, encoded)
    if run_root.is_dir():
        _write_once_bytes(run_root / "manifests" / "submission_failure.json", encoded)


def _extract_sbatch_job_id(response: str) -> str | None:
    matches = {
        match.group(1)
        for line in response.splitlines()
        if (
            match := re.fullmatch(
                r"\s*(\d+)(?:;[A-Za-z0-9._-]+)?\s*",
                line,
            )
        )
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def _prepare_and_submit_rq014(
    spec: dict[str, Any],
    validated: dict[str, Any],
    *,
    base: Path,
    repo: Path,
    spec_bytes: bytes,
) -> str:
    run_root = Path(validated["run_root"])
    if run_root.exists():
        raise ValueError(f"Run root already exists: {run_root}")
    code = run_root / "code"
    phase = "create_run_root"
    submission_started = False
    submitted: str | None = None
    sbatch_response_raw: str | None = None
    known_job_id: str | None = None
    try:
        for name in ("outputs", "logs", "manifests"):
            (run_root / name).mkdir(parents=True, exist_ok=False)
        phase = "seal_run_spec"
        if hashlib.sha256(spec_bytes).hexdigest() != validated["run_spec_sha256"]:
            raise ValueError("Retained run spec bytes differ from validation")
        if (
            hashlib.sha256(_canonical_spec_bytes(spec)).hexdigest()
            != validated["run_spec_semantic_sha256"]
        ):
            raise ValueError("Run spec semantics changed after validation")
        sealed_spec_path = run_root / "manifests" / "run_spec.json"
        sealed_spec_path.write_bytes(spec_bytes)

        phase = "materialize_closed_code_snapshot"
        if run_git(repo, "rev-parse", "HEAD") != validated["commit"]:
            raise ValueError("Managed checkout HEAD changed before code snapshot")
        snapshot_bytes = _materialize_rq014_code_snapshot(
            repo=repo,
            code=code,
            commit=validated["commit"],
            registered=validated["code_snapshot_files"],
        )
        if run_git(repo, "rev-parse", "HEAD") != validated["commit"]:
            raise ValueError("Managed checkout HEAD changed during code snapshot")
        snapshot_path = run_root / "manifests" / "code_snapshot.json"
        snapshot_path.write_bytes(snapshot_bytes)
        runtime_validated = {
            **validated,
            "code_snapshot_receipt_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        }

        phase = "write_run_manifest"
        manifest = {"schema_version": 2, "spec": spec, "validated": runtime_validated}
        manifest_path = run_root / "manifests" / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        phase = "render_sbatch"
        python = Path(validated["python_executable_path"])
        if not python.is_file():
            raise ValueError(f"Managed RQ014 interpreter is missing: {python}")
        script = run_root / "manifests" / "run.sbatch"
        script.write_text(
            render_rq014_sbatch(
                validated=runtime_validated,
                base=base,
                repo=repo,
                run_root=run_root,
                code=code,
                sealed_spec_path=sealed_spec_path,
            ),
            encoding="utf-8",
        )
        for path in (sealed_spec_path, snapshot_path, manifest_path, script):
            path.chmod(0o444)

        phase = "submit_sbatch"
        submission_started = True
        try:
            sbatch_response_raw = subprocess.check_output(
                [SYSTEM_SBATCH, "--parsable", "--export=NIL", str(script)],
                text=True,
                stderr=subprocess.STDOUT,
                env=MINIMAL_COMMAND_ENV,
            )
            submitted = sbatch_response_raw.strip()
        except subprocess.CalledProcessError as submit_error:
            raw_response = submit_error.output
            if raw_response in (None, "", b""):
                raw_response = submit_error.stderr
            if isinstance(raw_response, bytes):
                raw_response = raw_response.decode("utf-8", errors="replace")
            if isinstance(raw_response, str):
                sbatch_response_raw = raw_response
                submitted = raw_response.strip()
                known_job_id = _extract_sbatch_job_id(submitted)
            raise
        known_job_id = _extract_sbatch_job_id(submitted)
        if known_job_id is None:
            raise ValueError(f"Could not parse Slurm job ID: {submitted}")

        phase = "write_submission_receipt"
        receipt = {
            "schema_version": "rq014-submission-receipt-v1",
            "job_id": known_job_id,
            "job_name": validated["job_name"],
            "run_id": spec["run_id"],
            "operation": spec["operation"],
            "run_spec_sha256": validated["run_spec_sha256"],
            "git_commit": validated["commit"],
            "sbatch_response": sbatch_response_raw,
        }
        receipt_path = run_root / "manifests" / "submission_receipt.json"
        _write_once_bytes(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return submitted
    except Exception as exc:
        cleanup_complete = False
        if not submission_started:
            cleanup_complete = _rollback_unsubmitted_rq014_run(
                run_root=run_root,
            )
        try:
            _record_rq014_submission_failure(
                spec=spec,
                validated=validated,
                base=base,
                run_root=run_root,
                phase=phase,
                error=exc,
                submission_started=submission_started,
                cleanup_complete=cleanup_complete,
                known_job_id=known_job_id,
                sbatch_response=sbatch_response_raw,
            )
        except Exception as receipt_error:
            raise RuntimeError(
                f"RQ014 {phase} failed and its FAILED receipt could not be written: "
                f"{receipt_error}"
            ) from exc
        raise


def prepare_and_submit(
    spec: dict,
    validated: dict,
    *,
    base: Path,
    repo: Path,
    spec_bytes: bytes | None = None,
    wrapper_capability_verified: object | None = None,
) -> str:
    if int(spec["schema_version"]) != 2 or spec["rq_id"] != "RQ014":
        raise ValueError("Legacy/generic submission is disabled; use the RQ014 schema v2 path")
    if spec_bytes is None:
        raise ValueError("RQ014 submission requires retained canonical run-spec bytes")
    if wrapper_capability_verified is not _RQ014_VERIFIED_WRAPPER_CAPABILITY:
        raise ValueError("RQ014 submission requires a verified managed-wrapper capability")
    return _prepare_and_submit_rq014(
        spec,
        validated,
        base=base,
        repo=repo,
        spec_bytes=spec_bytes,
    )


def _validate_cli_entry_mode(
    spec: dict[str, Any],
    *,
    rq014_only: bool,
    submit: bool,
    wrapper_capability_verified: object | None,
) -> None:
    is_rq014_v2 = int(spec["schema_version"]) == 2 and spec["rq_id"] == "RQ014"
    if rq014_only and not is_rq014_v2:
        raise ValueError("The managed RQ014 wrapper accepts only schema v2 / RQ014 specs")
    if is_rq014_v2:
        if not rq014_only:
            raise ValueError("RQ014 schema v2 requires the managed RQ014-only wrapper")
        if wrapper_capability_verified is not _RQ014_VERIFIED_WRAPPER_CAPABILITY:
            raise ValueError("The RQ014-only flag requires a verified managed-wrapper capability")
        return
    if submit:
        raise ValueError("Submission is disabled outside the managed RQ014-only wrapper")


def _validate_rq014_launcher_sys_path(base: Path) -> None:
    expected = [
        str(base / "envs" / "ipv-exact-sigma01" / "lib" / "python39.zip"),
        str(base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9"),
        str(base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9" / "lib-dynload"),
    ]
    if sys.path != expected:
        raise ValueError("RQ014 launcher did not start with the exact isolated sys.path")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--rq014-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if (
        args.rq014_only
        and _RQ014_WRAPPER_CAPABILITY_VERIFIED is not _RQ014_VERIFIED_WRAPPER_CAPABILITY
    ):
        raise ValueError("The RQ014-only flag requires a verified managed-wrapper capability")
    base = Path(os.path.abspath(args.base))
    if args.rq014_only and base != Path(os.path.abspath(DEFAULT_BASE)):
        raise ValueError("RQ014 v2 runs require the fixed managed HPC base")
    repo = base / "code" / "repo"
    spec_path = Path(os.path.abspath(args.spec))
    if args.rq014_only:
        _validate_rq014_launcher_sys_path(base)
    spec, spec_bytes = _load_managed_canonical_run_spec(spec_path, base=base)
    _validate_cli_entry_mode(
        spec,
        rq014_only=args.rq014_only,
        submit=args.submit,
        wrapper_capability_verified=_RQ014_WRAPPER_CAPABILITY_VERIFIED,
    )
    validated = validate_spec(
        spec,
        base=base,
        repo=repo,
        spec_path=spec_path,
        spec_bytes=spec_bytes,
    )
    if args.submit:
        print(
            prepare_and_submit(
                spec,
                validated,
                base=base,
                repo=repo,
                spec_bytes=spec_bytes,
                wrapper_capability_verified=_RQ014_WRAPPER_CAPABILITY_VERIFIED,
            )
        )
    else:
        print(json.dumps(validated, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
