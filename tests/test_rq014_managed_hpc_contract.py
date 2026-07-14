from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.hpc.prepare_research_run as launcher
from scripts.rq014.materialize_registry import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_V3_PATH = (
    "/share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/"
    "managed_python_environment_v3.json"
)
ENVIRONMENT_V3_SHA256 = "30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a"


def _copy(repo: Path, relative: str) -> Path:
    destination = repo / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)
    return destination


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _checksum_manifest(path: Path, repo: Path, relatives: list[str]) -> Path:
    rows = [f"{sha256_file(repo / relative)}  {relative}" for relative in sorted(relatives)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _init_fixture_git_repo(repo: Path) -> str:
    subprocess.run(["/usr/bin/git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "config", "user.name", "RQ014 fixture"],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "rq014-fixture@example.invalid",
        ],
        check=True,
    )
    subprocess.run(["/usr/bin/git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "commit", "-q", "-m", "RQ014 fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repo),
            "update-ref",
            "refs/remotes/origin/main",
            commit,
        ],
        check=True,
    )
    return commit


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    base = tmp_path / "sociality_estimation"
    repo = base / "code" / "repo"
    legacy_manifest = ROOT / "reports" / "plans" / "RQ014_plan_v1p3_checksums_20260711.sha256"
    inherited = {line.split("  ", 1)[1] for line in legacy_manifest.read_text().splitlines()}
    copied = sorted(set(launcher.RQ014_REVIEW_REQUIRED_PATHS) | inherited)
    for relative in copied:
        _copy(repo, relative)

    legacy = base.parent / "RQ010B_wod_e2e"
    scene_refs = []
    for index in range(8):
        path = (
            legacy
            / "reframed_pref_analysis"
            / "phase1_ipv_build"
            / "shards"
            / f"shard_{index}"
            / "phase1_post_scene_bundle.pkl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture bundle {index}".encode("utf-8"))
        scene_refs.append({"path": str(path), "sha256": sha256_file(path)})
    readiness = legacy / "manifests" / "rated479_segment_readiness.tsv"
    readiness.parent.mkdir(parents=True, exist_ok=True)
    readiness.write_text("fixture readiness\n", encoding="utf-8")
    counterpart = (
        legacy
        / "reframed_pref_analysis"
        / "phase1_ipv_build"
        / "selected_counterpart_tracks.csv"
    )
    counterpart.parent.mkdir(parents=True, exist_ok=True)
    counterpart.write_text("fixture counterpart\n", encoding="utf-8")

    python = base / "envs" / "ipv-exact-sigma01" / "bin" / "python3.9"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nprintf 'Python 3.9.24\\n'\n", encoding="utf-8")
    python.chmod(0o755)
    stdlib_root = base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9"
    lib_dynload = stdlib_root / "lib-dynload"
    lib_dynload.mkdir(parents=True)
    (stdlib_root / "json.py").write_text("VALUE = 'stdlib fixture'\n", encoding="utf-8")
    (lib_dynload / "_fixture.so").write_bytes(b"fixture extension bytes")
    stdlib_files = sorted(path for path in stdlib_root.rglob("*") if path.is_file())
    stdlib_checksum = base / "manifests" / "RQ014" / "managed_python_stdlib_v1.sha256"
    stdlib_checksum.parent.mkdir(parents=True, exist_ok=True)
    stdlib_checksum.write_text(
        "".join(
            f"# size_bytes={path.stat().st_size}\n{sha256_file(path)}  {path}\n"
            for path in stdlib_files
        ),
        encoding="utf-8",
    )
    environment_lib = base / "envs" / "ipv-exact-sigma01" / "lib"
    crypto_resolved = environment_lib / "libcrypto.so.3.0"
    crypto_resolved.write_bytes(b"fixture crypto library")
    crypto_loader = environment_lib / "libcrypto.so.3"
    crypto_loader.symlink_to(crypto_resolved.name)
    zlib_loader = environment_lib / "libz.so.1"
    zlib_loader.write_bytes(b"fixture zlib library")
    native_rows = [
        (
            "libcrypto.so.3",
            crypto_loader,
            crypto_resolved.name,
            crypto_resolved,
        ),
        ("libz.so.1", zlib_loader, "-", zlib_loader),
    ]
    native_manifest = base / "manifests" / "RQ014" / "managed_python_native_libs_v1.tsv"
    native_manifest.write_text(
        "\n".join(
            [
                "# rq014-managed-python-native-libs-v1",
                (
                    "# columns=soname<TAB>loader_path<TAB>link_target_or_dash<TAB>"
                    "resolved_path<TAB>size_bytes<TAB>sha256"
                ),
                "# discovery=ldd_python3.9_plus_every_regular_lib-dynload_so",
                f"# managed_environment_root={base / 'envs' / 'ipv-exact-sigma01'}",
                f"# row_count={len(native_rows)}",
            ]
        )
        + "\n"
        + "".join(
            "\t".join(
                [
                    soname,
                    str(loader),
                    link_target,
                    str(resolved),
                    str(resolved.stat().st_size),
                    sha256_file(resolved),
                ]
            )
            + "\n"
            for soname, loader, link_target, resolved in native_rows
        ),
        encoding="utf-8",
    )

    inventory_path = (
        repo
        / "reports"
        / "studies"
        / "RQ014_wod_e2e_rating_recovery"
        / "02_g2_preflight"
        / "RQ014_declassification_source_inventory_20260712.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sources = {
        **{f"phase1_scene_bundle_{index:02d}": Path(ref["path"]) for index, ref in enumerate(scene_refs)},
        "rated479_structural_readiness": readiness,
        "selected_counterpart_tracks": counterpart,
    }
    for row in inventory["files"]:
        source = sources[row["role"]]
        row.update(path=str(source), size_bytes=source.stat().st_size, sha256=sha256_file(source))
    inventory["managed_interpreter"] = {
        "path": str(python),
        "version": "Python 3.9.24",
        "size_bytes": python.stat().st_size,
        "sha256": sha256_file(python),
    }
    _write_json(inventory_path, inventory)

    environment_payload = {
            "schema_version": "rq014-managed-python-environment-v3",
            "environment_id": "ipv-exact-sigma01",
            "python_executable": inventory["managed_interpreter"],
            "execution_dependencies": "PYTHON_STANDARD_LIBRARY_PLUS_PINNED_NATIVE_CLOSURE",
            "site_packages_imported": [],
            "isolated_python_flags": ["-I", "-S", "-B", "-X", "utf8"],
            "isolated_sys_path": [
                str(base / "envs" / "ipv-exact-sigma01" / "lib" / "python39.zip"),
                str(stdlib_root),
                str(lib_dynload),
            ],
            "stdlib_integrity": {
                "stdlib_root": str(stdlib_root),
                "lib_dynload_root": str(lib_dynload),
                "zip_path": str(
                    base / "envs" / "ipv-exact-sigma01" / "lib" / "python39.zip"
                ),
                "zip_path_status": "ABSENT",
                "symlink_count": 0,
                "regular_file_count": len(stdlib_files),
                "regular_file_total_size_bytes": sum(
                    path.stat().st_size for path in stdlib_files
                ),
                "checksum_manifest_path": str(stdlib_checksum),
                "checksum_manifest_size_bytes": stdlib_checksum.stat().st_size,
                "checksum_manifest_sha256": sha256_file(stdlib_checksum),
            },
            "native_library_integrity": {
                "discovery_scope": (
                    "ldd python3.9 plus every regular lib-dynload/*.so; "
                    "complete recursive resolved closure"
                ),
                "columns": [
                    "soname",
                    "loader_path",
                    "link_target_or_dash",
                    "resolved_path",
                    "size_bytes",
                    "sha256",
                ],
                "manifest_path": str(native_manifest),
                "manifest_size_bytes": native_manifest.stat().st_size,
                "manifest_sha256": sha256_file(native_manifest),
                "row_count": len(native_rows),
                "resolved_regular_file_total_size_bytes": sum(
                    resolved.stat().st_size for _, _, _, resolved in native_rows
                ),
                "symlink_row_count": 1,
                "multi_hop_count": 0,
                "system_library_trust_roots": ["/lib64"],
                "symlink_chain_policy": (
                    "exact loader link target and final resolved regular-file bytes"
                ),
            },
            "captured_at_utc": "2026-07-12T00:00:00Z",
    }
    environment = base / "manifests" / "RQ014" / "managed_python_environment_v3.json"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_bytes(launcher._canonical_spec_bytes(environment_payload))
    inventory["managed_environment_manifest"] = {
        "path": str(environment),
        "size_bytes": environment.stat().st_size,
        "sha256": sha256_file(environment),
    }
    _write_json(inventory_path, inventory)
    execution_contract_path = repo / "reports" / "plans" / "RQ014_execution_contract_v1p5.json"
    execution_contract = json.loads(execution_contract_path.read_text(encoding="utf-8"))
    execution_contract["managed_hpc_contract"].update(
        base=str(base),
        repo=str(repo),
        run_root_pattern=str(base / "work_dirs" / "RQ014" / "<RUN_ID>"),
        production_spec_root=str(base / "manifests" / "RQ014" / "run_specs"),
        environment_manifest_schema=(
            "configs/run_specs/rq014_managed_python_environment_v3.schema.json"
        ),
        environment_manifest_path=str(environment),
        environment_manifest_size_bytes=environment.stat().st_size,
        environment_manifest_sha256=sha256_file(environment),
        python_executable_sha256=sha256_file(python),
    )
    execution_contract["managed_hpc_contract"]["stdlib_integrity"].update(
        root=str(stdlib_root),
        regular_file_count=len(stdlib_files),
        regular_file_total_size_bytes=sum(path.stat().st_size for path in stdlib_files),
        checksum_manifest_path=str(stdlib_checksum),
        checksum_manifest_size_bytes=stdlib_checksum.stat().st_size,
        checksum_manifest_sha256=sha256_file(stdlib_checksum),
    )
    execution_contract["managed_hpc_contract"]["native_library_integrity"] = {
        "manifest_path": str(native_manifest),
        "manifest_size_bytes": native_manifest.stat().st_size,
        "manifest_sha256": sha256_file(native_manifest),
        "row_count": len(native_rows),
        "resolved_regular_file_total_size_bytes": sum(
            resolved.stat().st_size for _, _, _, resolved in native_rows
        ),
        "symlink_row_count": 1,
        "multi_hop_count": 0,
        "system_library_trust_roots": ["/lib64"],
    }
    _write_json(execution_contract_path, execution_contract)

    review_manifest_relative = launcher.RQ014_REVIEW_MANIFEST
    review_manifest = _checksum_manifest(repo / review_manifest_relative, repo, copied)
    review_common = {
        "schema_version": "rq014-formal-review-v1p5",
        "reviewed_manifest_path": review_manifest_relative,
        "reviewed_manifest_sha256": sha256_file(review_manifest),
        "verdict": "NO_BLOCKER",
        "unresolved_blockers": 0,
        "unresolved_majors": 0,
        "findings": [],
        "fresh_reviewer_attested": True,
        "rating_values_accessed": False,
        "hpc_jobs_submitted": False,
        "reviewed_at_utc": "2026-07-12T00:00:00Z",
    }
    stats_relative = launcher.RQ014_STATISTICS_REVIEW
    execution_relative = launcher.RQ014_EXECUTION_REVIEW
    stats = _write_json(
        repo / stats_relative,
        {
            **review_common,
            "review_role": "statistics",
            "reviewer_agent": "fixture-statistics-agent",
        },
    )
    execution = _write_json(
        repo / execution_relative,
        {
            **review_common,
            "review_role": "execution_governance",
            "reviewer_agent": "fixture-execution-governance-agent",
        },
    )
    formal_relative = launcher.RQ014_FORMAL_G1
    formal = _write_json(
        repo / formal_relative,
        {
            "schema_version": "rq014-formal-g1-v1p5",
            "status": "FORMAL_G1_PASS",
            "reviewed_manifest_path": review_manifest_relative,
            "reviewed_manifest_sha256": sha256_file(review_manifest),
            "g0_closure_status": "CLOSED_WITH_INACCESSIBLE_SURFACES",
            "reviewer_artifacts": {
                "statistics": {
                    "path": stats_relative,
                    "sha256": sha256_file(stats),
                    "verdict": "NO_BLOCKER",
                },
                "execution_governance": {
                    "path": execution_relative,
                    "sha256": sha256_file(execution),
                    "verdict": "NO_BLOCKER",
                },
            },
            "unresolved_blockers": 0,
            "unresolved_majors": 0,
            "no_rating_access_attested": True,
            "no_hpc_submission_attested": True,
            "adjudicated_at_utc": "2026-07-12T00:00:00Z",
        },
    )

    final_bundle_relative = launcher.RQ014_FINAL_BUNDLE
    final_paths = copied + [formal_relative, stats_relative, execution_relative, review_manifest_relative]
    final_bundle = _checksum_manifest(repo / final_bundle_relative, repo, final_paths)
    commit = _init_fixture_git_repo(repo)
    spec = _write_json(
        tmp_path / "run_spec.json",
        {
            "schema_version": 2,
            "rq_id": "RQ014",
            "run_id": "RQ014_fixture_export",
            "operation": "rq014_g2_declassification_export",
            "git_commit": commit,
            "formal_g1": {"path": str(formal), "sha256": sha256_file(formal)},
            "contract_bundle": {"path": str(final_bundle), "sha256": sha256_file(final_bundle)},
            "scene_bundles": scene_refs,
            "readiness_table": {"path": str(readiness), "sha256": sha256_file(readiness)},
            "counterpart_tracks": {"path": str(counterpart), "sha256": sha256_file(counterpart)},
            "environment_manifest": {"path": str(environment), "sha256": sha256_file(environment)},
            "created_at_utc": "2026-07-12T00:00:00Z",
            "resource_profile_id": "rq014-g2-declassify-cpu-v1",
        },
    )
    return base, spec, commit


def test_closed_snapshot_reads_only_exact_commit_blobs_not_dirty_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "payload.py"
    committed = b"COMMITTED = 1\n"
    source.write_bytes(committed)
    commit = _init_fixture_git_repo(repo)
    source.write_bytes(b"DIRTY = 2\n")

    code = tmp_path / "sealed"
    receipt = launcher._materialize_rq014_code_snapshot(
        repo=repo,
        code=code,
        commit=commit,
        registered={"payload.py": hashlib.sha256(committed).hexdigest()},
    )
    assert (code / "payload.py").read_bytes() == committed
    payload = json.loads(receipt)
    assert payload["schema_version"] == "rq014-code-snapshot-v2"
    assert payload["materialization_source"] == "EXACT_GIT_COMMIT_TREE_BLOBS"
    assert payload["git_commit"] == commit

    with pytest.raises(ValueError, match="commit-tree blob digest drift"):
        launcher._materialize_rq014_code_snapshot(
            repo=repo,
            code=tmp_path / "rejected",
            commit=commit,
            registered={"payload.py": sha256_file(source)},
        )
    untracked = repo / "self_consistent_bundle.sha256"
    untracked.write_text(f"{sha256_file(source)}  payload.py\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain exactly one registered path"):
        launcher._materialize_rq014_code_snapshot(
            repo=repo,
            code=tmp_path / "untracked-rejected",
            commit=commit,
            registered={untracked.name: sha256_file(untracked)},
        )


@pytest.mark.parametrize("tamper", ["registry_path", "evidence_bytes"])
def test_formal_g1_machine_checks_f01_f04_closure_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    base, _, _ = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    review_manifest = repo / launcher.RQ014_REVIEW_MANIFEST
    reviewed = launcher._verify_checksum_manifest(review_manifest, repo=repo)
    launcher._validate_g0_registry_evidence(repo=repo, reviewed=reviewed)
    if tamper == "registry_path":
        registry_path = repo / "reports" / "plans" / "RQ014_forensic_registry_v1p5.yaml"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["forensic_surfaces"][1]["closure_evidence"]["path"] = (
            "reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/other.md"
        )
        _write_json(registry_path, registry)
        message = "registry closure evidence drift"
    else:
        evidence = (
            repo
            / "reports"
            / "studies"
            / "RQ014_wod_e2e_rating_recovery"
            / "00_forensics"
            / "forensics_report.md"
        )
        evidence.write_bytes(evidence.read_bytes() + b"\nTAMPERED\n")
        message = "closure evidence hash mismatch"
    with pytest.raises(ValueError, match=message):
        launcher._validate_g0_registry_evidence(repo=repo, reviewed=reviewed)


def test_round2_through_round5_and_g0_provenance_are_required_review_bytes() -> None:
    assert {
        "reports/plans/RQ014_plan_v1p5_review_manifest_round2_blocked_20260712.sha256",
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_statistics_review_round2_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_execution_governance_review_round2_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_round2_remediation_20260712.md"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/"
            "forensics_report.md"
        ),
        "reports/plans/RQ014_plan_v1p5_review_manifest_round3_blocked_20260712.sha256",
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_statistics_review_round3_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_execution_governance_review_round3_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_round3_remediation_20260712.md"
        ),
        "reports/plans/RQ014_plan_v1p5_review_manifest_round4_blocked_20260712.sha256",
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_statistics_review_round4_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_execution_governance_review_round4_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_round4_remediation_20260712.md"
        ),
        "reports/plans/RQ014_plan_v1p5_review_manifest_round5_blocked_20260712.sha256",
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_statistics_review_round5_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_execution_governance_review_round5_blocked_20260712.json"
        ),
        (
            "reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/"
            "RQ014_v1p5_round5_remediation_20260712.md"
        ),
    } <= launcher.RQ014_REVIEW_REQUIRED_PATHS


def test_formal_g1_rejects_same_reviewer_identity_for_both_roles(
    tmp_path: Path,
) -> None:
    base, spec_path, _ = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    formal_path = Path(spec["formal_g1"]["path"])
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    for reference in formal["reviewer_artifacts"].values():
        review_path = repo / reference["path"]
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["reviewer_agent"] = "fixture-shared-reviewer-agent"
        _write_json(review_path, review)
        reference["sha256"] = sha256_file(review_path)
    _write_json(formal_path, formal)

    with pytest.raises(ValueError, match="reviewer identities must differ"):
        launcher._validate_formal_g1(
            formal_path,
            repo=repo,
            expected_review_manifest=launcher.RQ014_REVIEW_MANIFEST,
        )


def test_reviewed_runtime_references_use_the_same_v3_environment_lock() -> None:
    for relative in (
        "configs/run_specs/RQ014_g2_declassification_export.template.json",
        "configs/run_specs/RQ014_g2_contract_preflight.template.json",
    ):
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert payload["environment_manifest"] == {
            "path": ENVIRONMENT_V3_PATH,
            "sha256": ENVIRONMENT_V3_SHA256,
        }
    inventory = json.loads(
        (
            ROOT
            / "reports"
            / "studies"
            / "RQ014_wod_e2e_rating_recovery"
            / "02_g2_preflight"
            / "RQ014_declassification_source_inventory_20260712.json"
        ).read_text(encoding="utf-8")
    )
    assert inventory["managed_environment_manifest"] == {
        "path": ENVIRONMENT_V3_PATH,
        "size_bytes": 2229,
        "sha256": ENVIRONMENT_V3_SHA256,
    }
    assert inventory["execution_binding_contract"]["open_policy"] == (
        "SINGLE_FD_O_RDONLY_O_NOFOLLOW_O_CLOEXEC_O_NONBLOCK"
    )
    assert inventory["execution_binding_contract"]["parse_policy"] == (
        "VERIFY_EXPECTED_SIZE_AND_SHA256_THEN_PARSE_ONLY_RETAINED_BYTES"
    )
    schema = json.loads(
        (
            ROOT
            / "configs"
            / "run_specs"
            / "rq014_managed_python_environment_v3.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"].endswith("environment-v3")
    assert schema["properties"]["execution_dependencies"]["const"] == (
        "PYTHON_STANDARD_LIBRARY_PLUS_PINNED_NATIVE_CLOSURE"
    )
    assert schema["properties"]["stdlib_integrity"]["properties"][
        "checksum_manifest_sha256"
    ]["pattern"] == "^[0-9a-f]{64}$"
    execution_contract = json.loads(
        (ROOT / "reports" / "plans" / "RQ014_execution_contract_v1p5.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution_contract["gate_contract"]["formal_g1_requires"][
        "reviewer_agent_identities_must_differ"
    ] is True
    contract = execution_contract["managed_hpc_contract"]
    wrapper = ROOT / "scripts" / "hpc" / "submit_research_run.sh"
    assert contract["operator_wrapper_size_bytes"] == wrapper.stat().st_size
    assert contract["operator_wrapper_sha256"] == sha256_file(wrapper)
    assert contract["operator_wrapper_sha256"] in contract["operator_entrypoint"]
    assert contract["operator_entrypoint"].startswith(
        "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c"
    )
    assert "exec 8>\"$lock\" && /usr/bin/flock -s 8" in contract[
        "operator_entrypoint"
    ]
    assert 'exec 9<"$wrapper"' in contract["operator_entrypoint"]
    assert "/proc/$$/fd/9" in contract["operator_entrypoint"]
    assert 'exec /bin/sh /proc/$$/fd/9 "$@"' in contract["operator_entrypoint"]
    assert contract["environment_manifest_path"] == ENVIRONMENT_V3_PATH
    assert contract["environment_manifest_sha256"] == ENVIRONMENT_V3_SHA256
    assert contract["production_spec_root"].endswith("/manifests/RQ014/run_specs")
    assert "no path reopen" in contract["production_spec_contract"]
    assert "submit-only artifacts" in contract["validate_only_contract"]
    assert contract["wrapper_capability_contract"] == (
        launcher.RQ014_WRAPPER_CAPABILITY_CONTRACT
    )
    assert contract["python_import_surface"] == launcher.RQ014_PYTHON_IMPORT_SURFACE
    assert "scripts.rq014.materialize_registry" in contract["python_import_surface"]
    assert "empty __path__" in contract["python_import_surface"]
    assert contract["slurm_environment_export"].startswith("NIL_ON_DIRECTIVE")
    assert contract["stdlib_integrity"]["checksum_manifest_sha256"] == (
        "0a9944e1de0cf2b4168097b3afe82132333189127976c0a60c0891933853f0d5"
    )
    assert contract["native_library_integrity"] == {
        "manifest_path": (
            "/share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/"
            "managed_python_native_libs_v1.tsv"
        ),
        "manifest_size_bytes": 5858,
        "manifest_sha256": (
            "46004531edef17588151114e6e024a85cac7aad063a887ec5d600903b1dfaa9d"
        ),
        "row_count": 20,
        "resolved_regular_file_total_size_bytes": 14656296,
        "symlink_row_count": 16,
        "multi_hop_count": 0,
        "system_library_trust_roots": ["/lib64"],
    }
    assert "Git worktree checkout" in contract["code_execution_surface"]
    assert contract["code_execution_surface"].endswith("are forbidden")


def test_authoritative_rq014_operator_docs_require_the_same_clean_bootstrap() -> None:
    wrapper_sha256 = sha256_file(ROOT / "scripts" / "hpc" / "submit_research_run.sh")
    historical_wrapper_sha256 = (
        "d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d"
    )
    readme = (ROOT / "configs" / "run_specs" / "README.md").read_text(encoding="utf-8")
    assert wrapper_sha256 in readme
    for relative in (
        "reports/plans/RQ014_plan_v1p5_amendment_20260712.md",
        "reports/plans/prompts/RQ014_G2_kickoff_prompt_v1p5_20260712.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert historical_wrapper_sha256 in text
        assert "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c" in text
        assert 'exec 8>"$lock" && /usr/bin/flock -s 8' in text
        assert 'exec 9<"$wrapper"' in text
        assert "/proc/$$/fd/9" in text
        assert 'exec /bin/sh /proc/$$/fd/9 "$@"' in text
        assert "not a cryptographic secret" in text
        assert "managed_python_environment_v3.json" in text
        assert "managed-native" in text
        assert "materialize_registry.py" in text
        assert "reviewer_agent" in text
        assert "managed_python_environment_v2.json" not in text
        assert "manifests/RQ014/run_specs/" in text
    kickoff = (
        ROOT
        / "reports"
        / "plans"
        / "prompts"
        / "RQ014_G2_kickoff_prompt_v1p5_20260712.md"
    ).read_text(encoding="utf-8")
    assert "validate-only 不创建 run root、snapshot receipt 或 rendered sbatch" in kickoff
    assert "managed checkout clean" not in kickoff
    decision = (
        ROOT / "reports" / "plans" / "RQ014_PI_decision_G2_start_v1p5_20260712.md"
    ).read_text(encoding="utf-8")
    assert "checksum-bound clean-environment bootstrap" in decision
    assert "fd8" in decision and "fd9" in decision
    assert "submit_research_run.sh --spec" not in decision


def test_preflight_and_resource_pilot_are_conditionally_registered_for_review() -> None:
    execution = json.loads(
        (ROOT / "reports" / "plans" / "RQ014_execution_contract_v1p5.json").read_text(
            encoding="utf-8"
        )
    )
    operation = execution["authorization"]["registered_operations"][
        "rq014_g2_contract_preflight"
    ]
    assert operation["status"] == "CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1"
    launcher._validate_rq014_operation_contract(
        operation,
        operation_name="rq014_g2_contract_preflight",
        resource_profile_id="rq014-g2-preflight-cpu-v1",
    )
    authorization = json.loads(
        (ROOT / "configs" / "research_authorization.json").read_text(encoding="utf-8")
    )
    assert authorization["authorizations"]["RQ014"]["allowed_operations"] == [
        "rq014_g2_declassification_export",
        "rq014_g2_contract_preflight",
        "rq014_g2_resource_pilot",
    ]
    assert authorization["authorizations"]["RQ014"]["preflight_decision_path"] == (
        "reports/plans/RQ014_PI_decision_D1_preflight_v1p6_20260713.md"
    )
    assert authorization["authorizations"]["RQ014"]["pilot_decision_path"] == (
        "reports/plans/RQ014_PI_decision_D2_resource_pilot_20260714.md"
    )
    pilot = execution["authorization"]["registered_operations"][
        "rq014_g2_resource_pilot"
    ]
    assert pilot["status"] == "CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1"
    assert pilot["required_prior_receipts"] == [
        "rq014-g2-contract-preflight-receipt-v1",
        "rq014-managed-operation-done-v1",
    ]
    assert authorization["authorizations"]["RQ014"]["formal_g1_path"] == (
        launcher.RQ014_FORMAL_G1
    )
    assert execution["gate_contract"]["formal_g1_source"] == launcher.RQ014_FORMAL_G1
    assert execution["gate_contract"]["formal_g1_review_manifest"] == (
        launcher.RQ014_REVIEW_MANIFEST
    )
    template_path = (
        ROOT / "configs" / "run_specs" / "RQ014_g2_contract_preflight.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["formal_g1"]["path"].endswith(launcher.RQ014_FORMAL_G1)
    assert template["formal_g1"]["sha256"] == "0" * 64
    assert template["declassification_export_commit"] == "0" * 40
    assert template["contract_bundle"]["path"].endswith(launcher.RQ014_FINAL_BUNDLE)
    assert template["contract_bundle"]["sha256"] == "0" * 64
    assert template["m3_artifact"] == {
        "path": (
            "/share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/"
            "m3_scorer.joblib"
        ),
        "size_bytes": 88306301,
        "sha256": "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253",
    }
    schema = json.loads(
        (ROOT / "configs" / "run_specs" / "research_run_spec_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["m3_artifact"] == {"$ref": "#/$defs/m3ArtifactRef"}
    preflight_branch = schema["oneOf"][1]
    assert "m3_artifact" in preflight_branch["required"]


def test_allowlist_change_invalidates_old_formal_g1_before_preflight_submit() -> None:
    formal_path = (
        ROOT
        / "reports"
        / "studies"
        / "RQ014_wod_e2e_rating_recovery"
        / "01_plan_review"
        / "RQ014_formal_G1_v1p5_20260712.yaml"
    )
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    old_review_manifest = ROOT / formal["reviewed_manifest_path"]
    with pytest.raises(
        ValueError,
        match="Checksum manifest mismatch: configs/research_authorization.json",
    ):
        launcher._verify_checksum_manifest(old_review_manifest, repo=ROOT)
    with pytest.raises(ValueError, match="Formal G1 reviewer path drift"):
        launcher._validate_formal_g1(
            formal_path,
            repo=ROOT,
            expected_review_manifest=formal["reviewed_manifest_path"],
        )


@pytest.mark.parametrize("drift", ["status", "receipts", "refs"])
def test_preflight_contract_predicate_drift_is_fail_closed(drift: str) -> None:
    operation = json.loads(
        (ROOT / "reports" / "plans" / "RQ014_execution_contract_v1p5.json").read_text(
            encoding="utf-8"
        )
    )["authorization"]["registered_operations"]["rq014_g2_contract_preflight"]
    if drift == "status":
        operation["status"] = "DENY_PENDING_DECLASSIFICATION_RECEIPT"
        message = "does not conditionally authorize"
    elif drift == "receipts":
        operation["required_prior_receipts"] = []
        message = "prior-receipt predicate drift"
    else:
        operation["required_run_spec_refs"] = ["environment_manifest"]
        message = "prior-receipt predicate drift"
    with pytest.raises(ValueError, match=message):
        launcher._validate_rq014_operation_contract(
            operation,
            operation_name="rq014_g2_contract_preflight",
            resource_profile_id="rq014-g2-preflight-cpu-v1",
        )


def test_v2_managed_declassification_validates_all_contract_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)

    def fake_run_git(repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(launcher, "run_git", fake_run_git)
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(
        spec,
        base=base,
        repo=base / "code" / "repo",
        spec_path=spec_path,
    )
    assert validated["job_name"].startswith("zxc-rq014-export-")
    assert validated["rating_access"] == "FORBIDDEN"
    assert validated["entrypoint"] == "scripts/rq014/export_score_stripped_bundle.py"
    assert validated["run_spec_sha256"] == sha256_file(spec_path)
    assert validated["code_snapshot_plan"] == {
        "git_commit": commit,
        "materialization_source": "EXACT_GIT_COMMIT_TREE_BLOBS",
        "file_count": len(validated["code_snapshot_files"]),
        "files": validated["code_snapshot_files"],
        "receipt_state": "SUBMIT_ONLY_NOT_CREATED_BY_VALIDATE",
    }
    assert validated["submission_plan"]["job_name"] == validated["job_name"]
    assert validated["submission_plan"]["slurm_profile"] == validated["slurm_profile"]
    assert validated["submission_plan"]["environment_export_policy"] == {
        "sbatch_directive": "#SBATCH --export=NIL",
        "sbatch_command_flag": "--export=NIL",
        "rendered_script_state": "SUBMIT_ONLY_NOT_CREATED_BY_VALIDATE",
    }
    assert validated["runtime_metadata"]["python_executable_sha256"] == validated[
        "python_executable_sha256"
    ]
    assert not Path(validated["run_root"]).exists()
    script = launcher.render_rq014_sbatch(
        validated={**validated, "code_snapshot_receipt_sha256": "d" * 64},
        base=base,
        repo=base / "code" / "repo",
        run_root=Path(validated["run_root"]),
        code=Path(validated["run_root"]) / "code",
        sealed_spec_path=Path(validated["run_root"]) / "manifests" / "run_spec.json",
    )
    assert "#SBATCH --job-name=zxc-rq014-export-" in script
    assert "export_score_stripped_bundle.py" in script
    assert "scripts/rq014/materialize_registry.py" in script
    assert "--run-receipt-root" in script
    assert "rated479_segments" not in script
    assert "full479_targets" not in script
    assert "selected_counterpart_tracks.csv" in script
    assert script.count("--source-expectation") == 10
    assert "phase1_scene_bundle_00" in script
    assert "rated479_structural_readiness" in script
    assert "selected_counterpart_tracks" in script
    assert (
        f"# managed_environment_root={base / 'envs' / 'ipv-exact-sigma01'}"
        in script
    )


def test_v2_formal_g1_blocked_denies_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    formal = Path(spec_payload["formal_g1"]["path"])
    payload = json.loads(formal.read_text(encoding="utf-8"))
    payload["status"] = "FORMAL_G1_BLOCKED"
    _write_json(formal, payload)
    spec_payload["formal_g1"]["sha256"] = sha256_file(formal)
    _write_json(spec_path, spec_payload)

    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda repo, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    with pytest.raises(ValueError, match="formal G1 is not PASS"):
        launcher.validate_spec(
            launcher.load_spec(spec_path),
            base=base,
            repo=base / "code" / "repo",
            spec_path=spec_path,
        )


def test_v2_rejects_source_bytes_not_bound_by_reviewed_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    source = Path(spec_payload["scene_bundles"][0]["path"])
    source.write_bytes(b"different audited-source candidate order")
    spec_payload["scene_bundles"][0]["sha256"] = sha256_file(source)
    _write_json(spec_path, spec_payload)
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda repo, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    with pytest.raises(ValueError, match="differs from reviewed inventory"):
        launcher.validate_spec(
            launcher.load_spec(spec_path),
            base=base,
            repo=base / "code" / "repo",
            spec_path=spec_path,
        )


def test_v2_rejects_unstructured_environment_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    environment = Path(spec_payload["environment_manifest"]["path"])
    _write_json(environment, {"schema_version": "fixture-environment-v1"})
    spec_payload["environment_manifest"]["sha256"] = sha256_file(environment)
    _write_json(spec_path, spec_payload)
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda repo, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    with pytest.raises(ValueError, match="environment manifest"):
        launcher.validate_spec(
            launcher.load_spec(spec_path),
            base=base,
            repo=base / "code" / "repo",
            spec_path=spec_path,
        )


@pytest.mark.parametrize("tamper", ["modify", "extra", "symlink", "fifo"])
def test_v2_rejects_any_stdlib_tree_drift_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    stdlib = base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9"
    if tamper == "modify":
        (stdlib / "json.py").write_text("tampered stdlib\n", encoding="utf-8")
    elif tamper == "extra":
        (stdlib / "sitecustomize.py").write_text("INJECTED = True\n", encoding="utf-8")
    elif tamper == "symlink":
        (stdlib / "shadow.py").symlink_to(stdlib / "json.py")
    else:
        os.mkfifo(stdlib / "shadow.py")
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda repo, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    with pytest.raises(ValueError, match="stdlib"):
        launcher.validate_spec(
            launcher.load_spec(spec_path),
            base=base,
            repo=base / "code" / "repo",
            spec_path=spec_path,
        )


def test_v2_submission_uses_absolute_sbatch_and_empty_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(spec, base=base, repo=repo, spec_path=spec_path)

    observed: dict[str, object] = {}

    def fake_check_output(args: list[str], **kwargs: object) -> str:
        observed["args"] = args
        observed["env"] = kwargs.get("env")
        return "1729001\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_check_output)
    submitted = launcher.prepare_and_submit(
        spec,
        validated,
        base=base,
        repo=repo,
        spec_bytes=spec_path.read_bytes(),
        wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
    )
    assert submitted == "1729001"
    assert observed["args"] == [
        "/usr/bin/sbatch",
        "--parsable",
        "--export=NIL",
        str(Path(validated["run_root"]) / "manifests" / "run.sbatch"),
    ]
    assert observed["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    assert not any(key.startswith("SBATCH_") for key in observed["env"])
    receipt = json.loads(
        (Path(validated["run_root"]) / "manifests" / "submission_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["job_id"] == "1729001"
    assert receipt["sbatch_response"] == "1729001\n"


def test_v2_submission_seals_retained_spec_bytes_without_reopening_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    retained_spec_bytes = spec_path.read_bytes()
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(
        spec,
        base=base,
        repo=repo,
        spec_path=spec_path,
        spec_bytes=retained_spec_bytes,
    )
    spec_path.write_bytes(spec_path.read_bytes() + b" ")
    original_read_bytes = Path.read_bytes

    def reject_second_spec_read(path: Path) -> bytes:
        if path == spec_path:
            raise AssertionError("production spec path was reopened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_second_spec_read)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: "1729004\n",
    )
    assert launcher.prepare_and_submit(
        spec,
        validated,
        base=base,
        repo=repo,
        spec_bytes=retained_spec_bytes,
        wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
    ) == "1729004"
    sealed = Path(validated["run_root"]) / "manifests" / "run_spec.json"
    assert original_read_bytes(sealed) == retained_spec_bytes


def test_v2_submission_rejects_mutated_spec_semantics_with_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    retained = spec_path.read_bytes()
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(
        spec,
        base=base,
        repo=repo,
        spec_path=spec_path,
        spec_bytes=retained,
    )
    spec["created_at_utc"] = "2026-07-12T00:00:01Z"
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("sbatch must not run after semantic drift"),
    )
    with pytest.raises(ValueError, match="semantics changed"):
        launcher.prepare_and_submit(
            spec,
            validated,
            base=base,
            repo=repo,
            spec_bytes=retained,
            wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
        )
    assert not Path(validated["run_root"]).exists()


def test_v2_partial_snapshot_failure_always_removes_run_root_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(spec, base=base, repo=repo, spec_path=spec_path)

    def fail_after_partial_snapshot(**kwargs: object) -> bytes:
        code = Path(kwargs["code"])
        code.mkdir()
        (code / "partial.py").write_text("partial\n", encoding="utf-8")
        raise OSError("injected snapshot copy failure")

    monkeypatch.setattr(
        launcher,
        "_materialize_rq014_code_snapshot",
        fail_after_partial_snapshot,
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("sbatch must not run after snapshot failure"),
    )
    with pytest.raises(OSError, match="injected snapshot copy failure"):
        launcher.prepare_and_submit(
            spec,
            validated,
            base=base,
            repo=repo,
            spec_bytes=spec_path.read_bytes(),
            wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
        )
    assert not Path(validated["run_root"]).exists()
    failures = list((base / "manifests" / "RQ014" / "submission_failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["phase"] == "materialize_closed_code_snapshot"
    assert failure["cleanup_complete"] is True
    assert failure["retry_authorized"] is True


def test_v2_ambiguous_sbatch_failure_retains_run_and_records_nonretryable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(spec, base=base, repo=repo, spec_path=spec_path)

    def fail_sbatch(args: list[str], **kwargs: object) -> str:
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(launcher.subprocess, "check_output", fail_sbatch)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.prepare_and_submit(
            spec,
            validated,
            base=base,
            repo=repo,
            spec_bytes=spec_path.read_bytes(),
            wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
        )
    run_root = Path(validated["run_root"])
    assert run_root.is_dir()
    failure = json.loads(
        (run_root / "manifests" / "submission_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "FAILED"
    assert failure["submission_state"] == "SUBMISSION_STATE_UNKNOWN"
    assert failure["phase"] == "submit_sbatch"
    assert failure["cleanup_complete"] is False
    assert failure["retry_authorized"] is False
    assert failure["known_job_id"] is None
    assert failure["sbatch_response"] is None


def test_v2_failed_sbatch_preserves_raw_response_and_parseable_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(spec, base=base, repo=repo, spec_path=spec_path)

    def fail_after_scheduler_response(args: list[str], **kwargs: object) -> str:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args,
            output=" scheduler warning\n1729003;cluster\n",
        )

    monkeypatch.setattr(launcher.subprocess, "check_output", fail_after_scheduler_response)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.prepare_and_submit(
            spec,
            validated,
            base=base,
            repo=repo,
            spec_bytes=spec_path.read_bytes(),
            wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
        )
    failure = json.loads(
        (
            Path(validated["run_root"])
            / "manifests"
            / "submission_failure.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["phase"] == "submit_sbatch"
    assert failure["known_job_id"] == "1729003"
    assert failure["sbatch_response"] == " scheduler warning\n1729003;cluster\n"
    assert failure["retry_authorized"] is False


def test_v2_receipt_write_failure_preserves_known_job_id_for_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, spec_path, commit = _build_fixture(tmp_path)
    repo = base / "code" / "repo"
    monkeypatch.setattr(launcher, "DEFAULT_BASE", base)
    monkeypatch.setattr(
        launcher,
        "run_git",
        lambda candidate, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Python 3.9.24\n" if "--version" in args else "",
        ),
    )
    spec = launcher.load_spec(spec_path)
    validated = launcher.validate_spec(spec, base=base, repo=repo, spec_path=spec_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda args, **kwargs: "1729002;cluster\n",
    )
    original_write_once = launcher._write_once_bytes

    def fail_only_success_receipt(path: Path, payload: bytes) -> None:
        if path.name == "submission_receipt.json":
            raise OSError("injected receipt write failure")
        original_write_once(path, payload)

    monkeypatch.setattr(launcher, "_write_once_bytes", fail_only_success_receipt)
    with pytest.raises(OSError, match="injected receipt write failure"):
        launcher.prepare_and_submit(
            spec,
            validated,
            base=base,
            repo=repo,
            spec_bytes=spec_path.read_bytes(),
            wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
        )
    failure = json.loads(
        (
            Path(validated["run_root"])
            / "manifests"
            / "submission_failure.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["phase"] == "write_submission_receipt"
    assert failure["submission_state"] == "SUBMISSION_STATE_UNKNOWN"
    assert failure["known_job_id"] == "1729002"
    assert failure["sbatch_response"] == "1729002;cluster\n"
    assert failure["retry_authorized"] is False
