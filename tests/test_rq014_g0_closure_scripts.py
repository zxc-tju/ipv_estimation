from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_SCRIPT = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_hpc_pass4_v1p3.sh"
HPC_SBATCH = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_hpc_pass4_v1p3.sbatch"
MAC_SCRIPT = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_mac_pass4_v1p3.sh"
STATUS_SCHEMA_KEYS = {
    "schema_version",
    "surface_id",
    "generation_id",
    "state",
    "complete_scan",
    "search_mode",
    "scan_transport",
    "sampling_used",
    "required_inputs",
    "read_failures",
    "error",
    "residual_risk_statement",
    "scanned_file_count",
    "scanned_bytes",
    "matched_record_count",
    "manifest_file",
    "manifest_sha256",
    "evidence_file",
    "evidence_sha256",
    "generated_at_utc",
}


def _run(
    script: Path,
    *args: str,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=run_env,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation(output_dir: Path, surface_id: str, run_id: Optional[str] = None) -> Path:
    surface_root = output_dir / surface_id
    if run_id is None:
        current = surface_root / "CURRENT"
        assert current.is_file() and not current.is_symlink()
        run_id = current.read_text(encoding="ascii").strip()
    return surface_root / "generations" / run_id


def _load_and_check_status(
    output_dir: Path, surface_id: str, run_id: Optional[str] = None
) -> Dict[str, object]:
    generation = _generation(output_dir, surface_id, run_id)
    assert generation.is_dir()
    assert sorted(path.name for path in generation.iterdir()) == [
        "DONE",
        "evidence.txt",
        "manifest.csv",
        "status.json",
    ]
    record = json.loads((generation / "status.json").read_text(encoding="utf-8"))
    assert STATUS_SCHEMA_KEYS <= record.keys()
    assert record["schema_version"] == "rq014-g0-closure-v1p3"
    assert record["surface_id"] == surface_id
    assert record["generation_id"] == generation.name
    assert record["state"] in {"FOUND", "NOT_FOUND_ON_SCANNED_SURFACES", "INACCESSIBLE"}
    assert record["search_mode"] == "FULL_BYTE_STREAM_NO_SAMPLING"
    assert record["scan_transport"] == "STREAMING_FILE_DESCRIPTOR_READ_TO_EOF"
    assert record["sampling_used"] is False
    assert isinstance(record["residual_risk_statement"], str)
    assert record["residual_risk_statement"]
    manifest = generation / str(record["manifest_file"])
    evidence = generation / str(record["evidence_file"])
    assert _sha256(manifest) == record["manifest_sha256"]
    assert _sha256(evidence) == record["evidence_sha256"]
    with manifest.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == [
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
    done = json.loads((generation / "DONE").read_text(encoding="utf-8"))
    assert done["bundle_complete"] is True
    assert done["generation_id"] == generation.name
    assert done["manifest_sha256"] == _sha256(manifest)
    assert done["evidence_sha256"] == _sha256(evidence)
    assert done["status_sha256"] == _sha256(generation / "status.json")
    if record["state"] == "INACCESSIBLE":
        assert record["error"]
        assert record["read_failures"]
    else:
        assert record["error"] is None
    return record


def _hpc_args(root: Path, output_dir: Path, run_id: str = "fixture-run") -> List[str]:
    return [
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--pilot-dir",
        f"FL01={root / 'pilot_1'}",
        "--pilot-dir",
        f"FL02={root / 'pilot_2'}",
        "--pilot-dir",
        f"FL03={root / 'pilot_3'}",
        "--pilot-dir",
        f"FL04={root / 'pilot_4'}",
        "--phase3-root",
        str(root / "phase3"),
        "--job-log",
        str(root / "job.out"),
    ]


def _make_hpc_fixture(root: Path, content: str = "neutral fixture\n") -> None:
    for index in range(1, 5):
        pilot = root / f"pilot_{index}"
        pilot.mkdir(parents=True)
        (pilot / "artifact.txt").write_text(content, encoding="utf-8")
    phase3 = root / "phase3"
    phase3.mkdir()
    (phase3 / "summary.txt").write_text(content, encoding="utf-8")
    (root / "job.out").write_text(content, encoding="utf-8")


def _git(repo: Path, *args: str, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit_snapshot_anchor(
    repo: Path,
    note: Path,
    session: Path,
    snapshot_at_utc: str = "2026-07-09T23:59:59Z",
) -> str:
    if not (repo / ".git").is_dir():
        repo.mkdir()
        _git(repo, "init", "-q")
    f07_blob = repo / "F07/WODE2E.md"
    f08_blob = repo / "F08/session.jsonl"
    f07_blob.parent.mkdir(exist_ok=True)
    f08_blob.parent.mkdir(exist_ok=True)
    f07_blob.write_bytes(note.read_bytes())
    f08_blob.write_bytes(session.read_bytes())
    _git(repo, "add", "F07/WODE2E.md", "F08/session.jsonl")
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "RQ014 Snapshot Fixture",
            "GIT_AUTHOR_EMAIL": "rq014-snapshot@example.invalid",
            "GIT_COMMITTER_NAME": "RQ014 Snapshot Fixture",
            "GIT_COMMITTER_EMAIL": "rq014-snapshot@example.invalid",
            "GIT_AUTHOR_DATE": snapshot_at_utc,
            "GIT_COMMITTER_DATE": snapshot_at_utc,
        }
    )
    _git(repo, "commit", "-q", "--allow-empty", "-m", "anchored fixture snapshot", env=commit_env)
    return _git_output(repo, "rev-parse", "HEAD")


def _write_scope_manifest(
    path: Path,
    surface_id: str,
    sources: List[Path],
    snapshot_repo: Path,
    snapshot_commit: str,
    snapshot_git_paths: List[str],
    snapshot_at_utc: str = "2026-07-09T23:59:59Z",
) -> None:
    assert len(sources) == len(snapshot_git_paths)
    payload = {
        "schema_version": "rq014-frozen-scope-v1p3",
        "surface_id": surface_id,
        "entries": [
            {
                "source_path": str(source.resolve()),
                "source_sha256": _sha256(source),
                "snapshot_at_utc": snapshot_at_utc,
                "snapshot_evidence_kind": "git_blob_v1_content_integrity_only",
                "snapshot_git_repo": str(snapshot_repo.resolve()),
                "snapshot_git_commit": snapshot_commit,
                "snapshot_git_path": snapshot_git_path,
            }
            for source, snapshot_git_path in zip(sources, snapshot_git_paths)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_mac_fixture(root: Path, content: str = "neutral fixture\n") -> Dict[str, Path]:
    archived = root / "archived"
    archived.mkdir(parents=True)
    (archived / "old_notes.txt").write_text(content, encoding="utf-8")

    paper = root / "paper"
    paper.mkdir()
    _git(paper, "init", "-q")
    (paper / "manuscript.txt").write_text(content, encoding="utf-8")
    _git(paper, "add", "manuscript.txt")
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "RQ014 Fixture",
            "GIT_AUTHOR_EMAIL": "rq014@example.invalid",
            "GIT_COMMITTER_NAME": "RQ014 Fixture",
            "GIT_COMMITTER_EMAIL": "rq014@example.invalid",
        }
    )
    _git(paper, "commit", "-q", "-m", "fixture snapshot", env=commit_env)

    note = root / "WODE2E.md"
    note.write_text(content, encoding="utf-8")
    cowork = root / "cowork"
    cowork.mkdir()
    session = cowork / "session.jsonl"
    session.write_text(content, encoding="utf-8")
    snapshot_repo = root / "snapshot_anchor"
    snapshot_commit = _commit_snapshot_anchor(snapshot_repo, note, session)
    scope_dir = root / "frozen_scopes"
    f07_scope = scope_dir / "F07_scope.json"
    f08_scope = scope_dir / "F08_scope.json"
    _write_scope_manifest(
        f07_scope, "F07", [note], snapshot_repo, snapshot_commit, ["F07/WODE2E.md"]
    )
    _write_scope_manifest(
        f08_scope, "F08", [session], snapshot_repo, snapshot_commit, ["F08/session.jsonl"]
    )
    return {
        "archived": archived,
        "paper": paper,
        "note": note,
        "cowork": cowork,
        "session": session,
        "snapshot_repo": snapshot_repo,
        "f07_scope": f07_scope,
        "f08_scope": f08_scope,
    }


def _refresh_mac_scopes(paths: Dict[str, Path]) -> None:
    snapshot_commit = _commit_snapshot_anchor(
        paths["snapshot_repo"], paths["note"], paths["session"]
    )
    _write_scope_manifest(
        paths["f07_scope"],
        "F07",
        [paths["note"]],
        paths["snapshot_repo"],
        snapshot_commit,
        ["F07/WODE2E.md"],
    )
    _write_scope_manifest(
        paths["f08_scope"],
        "F08",
        [paths["session"]],
        paths["snapshot_repo"],
        snapshot_commit,
        ["F08/session.jsonl"],
    )


def _mac_args(
    paths: Dict[str, Path], output_dir: Path, run_id: str = "fixture-run"
) -> List[str]:
    return [
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--archived-root",
        str(paths["archived"]),
        "--paper-repo",
        str(paths["paper"]),
        "--f07-scope-manifest",
        str(paths["f07_scope"]),
        "--f08-scope-manifest",
        str(paths["f08_scope"]),
    ]


@pytest.mark.parametrize("script", [HPC_SCRIPT, HPC_SBATCH, MAC_SCRIPT])
def test_closure_scripts_have_valid_shell_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_hpc_sbatch_is_checksum_bound_and_fixed_scope() -> None:
    text = HPC_SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=zxc-rq014-g0-f05" in text
    assert "#SBATCH --output=/share/home/u25310231/ZXC/RQ014_recovery/" in text
    assert 'durable_root="/share/home/u25310231/ZXC/RQ014_recovery"' in text
    assert "RQ014_plan_v1p3_checksums_20260711.sha256" in text
    assert "RQ014_forensics_hpc_pass4_v1p3.sh" in text
    assert "RQ014_forensics_hpc_pass4_v1p3.sbatch" in text
    assert "/share/home/u25310231/.conda/envs/ipv/bin/python" in text
    assert "RQ014_G0_F05_VERIFIED_SLURM_WRAPPER" in text
    assert 'readlink -f -- "${python_bin}"' in text
    assert 'export RQ014_G0_PYTHON_REALPATH="${python_realpath}"' in text
    assert 'export RQ014_G0_PYTHON_SHA256="${python_sha256}"' in text
    assert 'export RQ014_G0_PYTHON_VERSION="${python_version}"' in text
    assert '|| -L "${python_bin}"' not in text

    script_text = HPC_SCRIPT.read_text(encoding="utf-8")
    assert "SLURM_PYTHON_REALPATH_ENV" in script_text
    assert "running Python SHA-256 differs" in script_text
    assert "running Python version differs" in script_text

    result = _run(HPC_SBATCH, "safe-run", "unexpected")
    assert result.returncode == 64


def test_hpc_missing_required_roots_is_nonzero_and_inaccessible(tmp_path: Path) -> None:
    output = tmp_path / "output"
    result = _run(HPC_SCRIPT, *_hpc_args(tmp_path / "missing", output))
    assert result.returncode != 0
    record = _load_and_check_status(output, "F05")
    assert record["state"] == "INACCESSIBLE"
    assert record["complete_scan"] is False


def test_hpc_clean_complete_scan_is_not_found(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    _make_hpc_fixture(fixture)
    output = tmp_path / "output"
    result = _run(HPC_SCRIPT, *_hpc_args(fixture, output))
    assert result.returncode == 0, result.stderr
    record = _load_and_check_status(output, "F05")
    assert record["state"] == "NOT_FOUND_ON_SCANNED_SURFACES"
    assert record["complete_scan"] is True
    assert record["scanned_file_count"] == 6
    assert record["matched_record_count"] == 0


def test_hpc_hit_is_found_and_evidence_line_is_untruncated(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    long_line = "x" * 7000 + " Spearman rho=-0.42" + "y" * 7000 + "\n"
    _make_hpc_fixture(fixture)
    (fixture / "pilot_3" / "artifact.txt").write_text(long_line, encoding="utf-8")
    output = tmp_path / "output"
    result = _run(HPC_SCRIPT, *_hpc_args(fixture, output))
    assert result.returncode == 0, result.stderr
    record = _load_and_check_status(output, "F05")
    assert record["state"] == "FOUND"
    evidence = (_generation(output, "F05") / "evidence.txt").read_text(encoding="utf-8")
    encoded = next(
        line.removeprefix("content_base64: ")
        for line in evidence.splitlines()
        if line.startswith("content_base64: ")
    )
    assert base64.b64decode(encoded) == long_line.encode("utf-8")


@pytest.mark.parametrize("empty_target", ["pilot_2", "phase3", "job.out"])
def test_hpc_empty_required_surface_fails_closed(
    tmp_path: Path, empty_target: str
) -> None:
    fixture = tmp_path / "fixture"
    _make_hpc_fixture(fixture)
    target = fixture / empty_target
    if target.is_dir():
        for child in target.iterdir():
            child.unlink()
    else:
        target.write_bytes(b"")
    output = tmp_path / "output"
    result = _run(HPC_SCRIPT, *_hpc_args(fixture, output))
    assert result.returncode != 0
    record = _load_and_check_status(output, "F05")
    assert record["state"] == "INACCESSIBLE"
    assert "empty" in str(record["error"]).lower() or "non-empty" in str(record["error"]).lower()


@pytest.mark.parametrize("symlink_kind", ["root", "entry", "ancestor"])
def test_hpc_symlink_root_or_entry_fails_closed(tmp_path: Path, symlink_kind: str) -> None:
    fixture = tmp_path / "fixture"
    _make_hpc_fixture(fixture)
    if symlink_kind == "root":
        real = fixture / "pilot_1"
        moved = fixture / "pilot_1_real"
        real.rename(moved)
        real.symlink_to(moved, target_is_directory=True)
    else:
        if symlink_kind == "entry":
            (fixture / "pilot_1" / "linked.txt").symlink_to(
                fixture / "pilot_2" / "artifact.txt"
            )
        else:
            real_parent = tmp_path / "real-parent"
            real_parent.mkdir()
            moved_fixture = real_parent / "fixture"
            fixture.rename(moved_fixture)
            alias_parent = tmp_path / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            fixture = alias_parent / "fixture"
    output = tmp_path / "output"
    result = _run(HPC_SCRIPT, *_hpc_args(fixture, output))
    assert result.returncode != 0
    record = _load_and_check_status(output, "F05")
    assert record["state"] == "INACCESSIBLE"
    assert "symlink" in str(record["error"]).lower()


def test_hpc_rejects_output_inside_missing_future_input(tmp_path: Path) -> None:
    missing = tmp_path / "future_root"
    output = missing / "pilot_1" / "rq014_output"
    result = _run(HPC_SCRIPT, *_hpc_args(missing, output))
    assert result.returncode == 2
    assert "including a declared input that does not yet exist" in result.stderr
    assert not output.exists()


def test_hpc_login_budget_rejects_heavy_scan_before_output(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    _make_hpc_fixture(fixture)
    heavy = fixture / "pilot_1/heavy.bin"
    with heavy.open("wb") as handle:
        handle.seek(200 * 1024 * 1024)
        handle.write(b"x")
    output = tmp_path / "output"

    result = _run(HPC_SCRIPT, *_hpc_args(fixture, output))

    assert result.returncode == 2
    assert "above login_node budget" in result.stderr
    assert not output.exists()


def test_hpc_current_failure_preserves_old_pointer_and_generations(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    _make_hpc_fixture(fixture)
    output = tmp_path / "output"
    first = _run(HPC_SCRIPT, *_hpc_args(fixture, output, "old-generation"))
    assert first.returncode == 0, first.stderr
    old_generation = _generation(output, "F05")
    old_hashes = {path.name: _sha256(path) for path in old_generation.iterdir()}

    second = _run(
        HPC_SCRIPT,
        *_hpc_args(fixture, output, "new-generation"),
        env={"RQ014_TEST_FAIL_CURRENT_SURFACE": "F05"},
    )
    assert second.returncode == 4
    assert (output / "F05" / "CURRENT").read_text(encoding="ascii").strip() == "old-generation"
    assert _generation(output, "F05", "new-generation").is_dir()
    _load_and_check_status(output, "F05", "new-generation")
    assert old_hashes == {path.name: _sha256(path) for path in old_generation.iterdir()}

    retry = _run(HPC_SCRIPT, *_hpc_args(fixture, output, "old-generation"))
    assert retry.returncode == 4
    assert old_hashes == {path.name: _sha256(path) for path in old_generation.iterdir()}


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_missing_required_inputs_is_nonzero_and_all_surfaces_inaccessible(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    paths = {
        "archived": missing / "archived",
        "paper": missing / "paper",
        "f07_scope": missing / "F07.json",
        "f08_scope": missing / "F08.json",
    }
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode != 0
    for surface_id in ("F06", "F07", "F08"):
        record = _load_and_check_status(output, surface_id)
        assert record["state"] == "INACCESSIBLE"
        assert record["complete_scan"] is False


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_backdated_git_scope_is_inaccessible_and_scope_hashes_are_bound(
    tmp_path: Path,
) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode == 3, result.stderr
    f06 = _load_and_check_status(output, "F06")
    assert f06["state"] == "NOT_FOUND_ON_SCANNED_SURFACES"
    assert f06["complete_scan"] is True
    f07 = _load_and_check_status(output, "F07")
    f08 = _load_and_check_status(output, "F08")
    for record in (f07, f08):
        assert record["state"] == "INACCESSIBLE"
        assert record["complete_scan"] is False
        assert record["scope_time_witness_state"] == (
            "INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT"
        )
        assert "Git timestamps are content provenance only" in str(record["error"])
    assert f07["scope_manifest_sha256"] == _sha256(paths["f07_scope"])
    assert f08["scope_manifest_sha256"] == _sha256(paths["f08_scope"])
    assert f07["scope_cutoff_utc"] == "2026-07-10T00:00:00Z"
    assert f07["scope_anchor_kind"] == "git_blob_v1_content_integrity_only"
    assert f08["scope_anchor_count"] == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_hits_remain_visible_but_untrusted_scopes_stay_inaccessible(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    long_line = "a" * 5000 + " preference_score and IPV envelope" + "b" * 5000 + "\n"
    (paths["archived"] / "old_notes.txt").write_text(long_line, encoding="utf-8")
    paths["note"].write_text("rating strongly tracks deviation\n", encoding="utf-8")
    paths["session"].write_text('{"message":"Spearman rho=-0.44"}\n', encoding="utf-8")
    _refresh_mac_scopes(paths)
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode == 3, result.stderr
    f06 = _load_and_check_status(output, "F06")
    assert f06["state"] == "FOUND"
    assert f06["matched_record_count"] >= 1
    for surface_id in ("F07", "F08"):
        record = _load_and_check_status(output, surface_id)
        assert record["state"] == "INACCESSIBLE"
        assert record["complete_scan"] is False
        assert record["matched_record_count"] >= 1
    f06_evidence = (_generation(output, "F06") / "evidence.txt").read_text(
        encoding="utf-8"
    )
    encoded = next(
        line.removeprefix("content_base64: ")
        for line in f06_evidence.splitlines()
        if line.startswith("content_base64: ")
    )
    assert base64.b64decode(encoded) == long_line.encode("utf-8")


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_scope_cutoff_is_strict_and_fails_closed(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    _write_scope_manifest(
        paths["f07_scope"],
        "F07",
        [paths["note"]],
        paths["snapshot_repo"],
        _git_output(paths["snapshot_repo"], "rev-parse", "HEAD"),
        ["F07/WODE2E.md"],
        snapshot_at_utc="2026-07-10T00:00:00Z",
    )
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode != 0
    f07 = _load_and_check_status(output, "F07")
    assert f07["state"] == "INACCESSIBLE"
    assert "strictly before" in str(f07["error"])


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F07 fixtures")
def test_mac_self_reported_scope_without_external_git_anchor_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    payload = json.loads(paths["f07_scope"].read_text(encoding="utf-8"))
    for key in (
        "snapshot_evidence_kind",
        "snapshot_git_repo",
        "snapshot_git_commit",
        "snapshot_git_path",
    ):
        payload["entries"][0].pop(key)
    paths["f07_scope"].write_text(json.dumps(payload) + "\n", encoding="utf-8")

    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))

    assert result.returncode != 0
    f07 = _load_and_check_status(output, "F07")
    assert f07["state"] == "INACCESSIBLE"
    assert "snapshot_evidence_kind" in str(f07["error"])


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F07 fixtures")
def test_mac_scope_cannot_scan_the_other_scope_manifest(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    payload = json.loads(paths["f07_scope"].read_text(encoding="utf-8"))
    payload["entries"][0]["source_path"] = str(paths["f08_scope"].resolve())
    payload["entries"][0]["source_sha256"] = _sha256(paths["f08_scope"])
    paths["f07_scope"].write_text(json.dumps(payload) + "\n", encoding="utf-8")

    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))

    assert result.returncode != 0
    f07 = _load_and_check_status(output, "F07")
    assert f07["state"] == "INACCESSIBLE"
    assert "either scope manifest" in str(f07["error"])


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_scope_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    paths["session"].write_text("changed after frozen scope\n", encoding="utf-8")
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode != 0
    f08 = _load_and_check_status(output, "F08")
    assert f08["state"] == "INACCESSIBLE"
    assert "hash differs" in str(f08["error"])


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_frozen_scope_symlink_source_fails_closed(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    target = paths["note"]
    real = target.with_name("WODE2E-real.md")
    target.rename(real)
    target.symlink_to(real)
    # Deliberately preserve the originally frozen source path and expected hash.
    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))
    assert result.returncode != 0
    f07 = _load_and_check_status(output, "F07")
    assert f07["state"] == "INACCESSIBLE"
    assert "symlink" in str(f07["error"]).lower()


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F07 fixtures")
def test_mac_frozen_scope_ancestor_symlink_fails_closed(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    real_dir = tmp_path / "real-source"
    real_dir.mkdir()
    real_note = real_dir / "WODE2E.md"
    real_note.write_bytes(paths["note"].read_bytes())
    alias_dir = tmp_path / "alias-source"
    alias_dir.symlink_to(real_dir, target_is_directory=True)
    payload = json.loads(paths["f07_scope"].read_text(encoding="utf-8"))
    payload["entries"][0]["source_path"] = str(alias_dir / "WODE2E.md")
    paths["f07_scope"].write_text(json.dumps(payload) + "\n", encoding="utf-8")

    output = tmp_path / "output"
    result = _run(MAC_SCRIPT, *_mac_args(paths, output))

    assert result.returncode != 0
    f07 = _load_and_check_status(output, "F07")
    assert f07["state"] == "INACCESSIBLE"
    assert "symlink path component" in str(f07["error"])


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for F06 fixtures")
def test_mac_surface_current_failure_preserves_old_f07_pointer(tmp_path: Path) -> None:
    paths = _make_mac_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    first = _run(MAC_SCRIPT, *_mac_args(paths, output, "old-generation"))
    assert first.returncode == 3, first.stderr
    old_f07 = _generation(output, "F07")
    old_hashes = {path.name: _sha256(path) for path in old_f07.iterdir()}
    second = _run(
        MAC_SCRIPT,
        *_mac_args(paths, output, "new-generation"),
        env={"RQ014_TEST_FAIL_CURRENT_SURFACE": "F07"},
    )
    assert second.returncode == 4
    assert (output / "F07" / "CURRENT").read_text(encoding="ascii").strip() == "old-generation"
    _load_and_check_status(output, "F07", "new-generation")
    assert old_hashes == {path.name: _sha256(path) for path in old_f07.iterdir()}
