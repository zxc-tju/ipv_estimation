from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.hpc.prepare_research_run as launcher
from scripts.hpc.prepare_research_run import load_spec, render_rq014_sbatch, validate_spec


ROOT = Path(__file__).resolve().parents[1]


def write_spec(path: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": 1,
        "rq_id": "RQ014",
        "run_id": "blocked-smoke",
        "operation": "scientific_compute",
        "git_commit": "0" * 40,
        "data_manifest_path": "/missing",
        "data_manifest_sha256": "0" * 64,
        "csv_path": "/missing.csv",
        "pkl_root": "/missing-pkl",
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rq014_legacy_schema_is_rejected_before_filesystem_or_git_checks(tmp_path) -> None:
    spec = load_spec(write_spec(tmp_path / "run.yaml"))
    with pytest.raises(ValueError, match="require.*schema v2"):
        validate_spec(spec, base=tmp_path, repo=ROOT)


def test_run_identifiers_are_fail_closed(tmp_path) -> None:
    path = write_spec(tmp_path / "run.yaml", run_id="../escape")
    with pytest.raises(ValueError, match="Unsafe run_id"):
        load_spec(path)


@pytest.mark.parametrize("run_id", [".", "..", ".hidden", "trailing."])
def test_v2_run_id_cannot_escape_or_use_dot_segments(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe run_id"):
        load_spec(write_v2_spec(tmp_path / f"{run_id.replace('.', 'dot')}.json", run_id=run_id))


def write_v2_spec(path: Path, **overrides: object) -> Path:
    ref = {"path": "/managed/missing", "sha256": "1" * 64}
    payload = {
        "schema_version": 2,
        "rq_id": "RQ014",
        "run_id": "RQ014_preflight_fixture",
        "operation": "rq014_g2_contract_preflight",
        "git_commit": "1" * 40,
        "formal_g1": dict(ref),
        "contract_bundle": dict(ref),
        "environment_manifest": dict(ref),
        "input_manifest": dict(ref),
        "sanitization_receipt": dict(ref),
        "materialization_ledger": dict(ref),
        "declassification_export_receipt": dict(ref),
        "declassification_export_done": dict(ref),
        "resource_profile_id": "rq014-g2-preflight-cpu-v1",
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_managed_v2_spec(base: Path, **overrides: object) -> Path:
    path = base / "manifests" / "RQ014" / "run_specs" / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_v2_spec(path, **overrides)
    spec = load_spec(path)
    path.write_bytes(launcher._canonical_spec_bytes(spec))
    path.chmod(0o444)
    return path


def test_v2_rejects_placeholder_hashes_and_unknown_fields(tmp_path: Path) -> None:
    zero_ref = {"path": "/managed/missing", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="placeholder"):
        load_spec(write_v2_spec(tmp_path / "zero.json", formal_g1=zero_ref))
    path = write_v2_spec(tmp_path / "unknown.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command"] = "arbitrary"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_spec(path)


def test_v2_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":2,"schema_version":2,"rq_id":"RQ014"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_spec(path)


def test_production_spec_is_loaded_once_as_canonical_bytes(tmp_path: Path) -> None:
    path = write_v2_spec(tmp_path / "run.json")
    with pytest.raises(ValueError, match="canonical JSON"):
        launcher._load_canonical_run_spec(path)
    parsed = load_spec(path)
    canonical = launcher._canonical_spec_bytes(parsed)
    path.write_bytes(canonical)
    loaded, loaded_bytes = launcher._load_canonical_run_spec(path)
    assert loaded == parsed
    assert loaded_bytes == canonical


def test_managed_production_spec_uses_one_no_follow_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    path = write_managed_v2_spec(base)
    expected = path.read_bytes()
    original_open = os.open
    opened: list[Path] = []

    def counted_open(candidate: object, flags: int, *args: object, **kwargs: object) -> int:
        opened.append(Path(candidate))
        return original_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "open", counted_open)
    spec, payload = launcher._load_managed_canonical_run_spec(path, base=base)
    assert spec["rq_id"] == "RQ014"
    assert payload == expected
    assert opened == [path]


def test_managed_production_spec_rejects_descriptor_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    path = write_managed_v2_spec(base)
    original_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        result = original_fstat(descriptor)
        if calls == 1:
            return result
        values = {
            field: getattr(result, field)
            for field in (
                "st_mode",
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        values["st_mtime_ns"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(launcher.os, "fstat", drifting_fstat)
    with pytest.raises(ValueError, match="descriptor identity drift"):
        launcher._load_managed_canonical_run_spec(path, base=base)


@pytest.mark.parametrize(
    "invalid_kind",
    ["outside", "writable", "symlink", "parent_symlink", "fifo"],
)
def test_managed_production_spec_rejects_unsafe_location_or_file_type(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    base = tmp_path / "base"
    path = write_managed_v2_spec(base)
    candidate = path
    if invalid_kind == "outside":
        candidate = tmp_path / "outside.json"
        candidate.write_bytes(path.read_bytes())
        candidate.chmod(0o444)
    elif invalid_kind == "writable":
        path.chmod(0o644)
    elif invalid_kind == "symlink":
        target = path.with_name("real.json")
        path.rename(target)
        candidate.symlink_to(target.name)
    elif invalid_kind == "parent_symlink":
        run_specs = path.parent
        real_root = run_specs.with_name("run_specs_real")
        run_specs.rename(real_root)
        run_specs.symlink_to(real_root.name, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path, 0o444)
    with pytest.raises(ValueError, match="managed run_specs|read-only|symlink|regular file"):
        launcher._load_managed_canonical_run_spec(candidate, base=base)


def test_cli_rejects_base_override_before_managed_spec_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_research_run.py",
            "--rq014-only",
            "--base",
            str(tmp_path / "attacker"),
            "--spec",
            str(tmp_path / "attacker" / "manifests" / "RQ014" / "run_specs" / "run.json"),
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_load_managed_canonical_run_spec",
        lambda *args, **kwargs: pytest.fail("attacker-base spec must not be accessed"),
    )
    monkeypatch.setattr(
        launcher,
        "_RQ014_WRAPPER_CAPABILITY_VERIFIED",
        launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
    )
    with pytest.raises(ValueError, match="fixed managed HPC base"):
        launcher.main()


def test_v2_base_override_is_rejected_before_alternate_checkout_access(tmp_path: Path) -> None:
    spec = load_spec(write_v2_spec(tmp_path / "run.json"))
    with pytest.raises(ValueError, match="fixed managed HPC base"):
        validate_spec(
            spec,
            base=tmp_path / "attacker_base",
            repo=tmp_path / "attacker_base" / "code" / "repo",
        )


def test_wrapper_capability_accepts_inherited_fd8_and_fd9(tmp_path: Path) -> None:
    runtime_lock = tmp_path / "runtime_maintenance.lock"
    wrapper = tmp_path / "submit_research_run.sh"
    runtime_lock.write_text("", encoding="utf-8")
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    proc_root = tmp_path / "proc_self_fd"
    proc_root.mkdir()
    (proc_root / "8").symlink_to(runtime_lock)
    (proc_root / "9").symlink_to(wrapper)
    probe = """
import importlib.util
import sys
from pathlib import Path

launcher_path, runtime_lock, wrapper, proc_root = sys.argv[1:]
spec = importlib.util.spec_from_file_location("_rq014_capability_probe", launcher_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot construct launcher probe")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
capability = launcher._verify_rq014_wrapper_capability(
    runtime_lock_path=Path(runtime_lock),
    wrapper_path=Path(wrapper),
    proc_fd_root=Path(proc_root),
)
if capability is not launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY:
    raise RuntimeError("wrong capability identity")
"""
    shell = 'exec 8<"$1"; exec 9<"$2"; exec "$3" -I -S -B -X utf8 -c "$4" "$5" "$1" "$2" "$6"'
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            shell,
            "rq014-capability",
            str(runtime_lock),
            str(wrapper),
            sys.executable,
            probe,
            str(ROOT / "scripts" / "hpc" / "prepare_research_run.py"),
            str(proc_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "failure", ["missing", "wrong_target", "wrong_inode", "symlink", "directory"]
)
def test_wrapper_capability_rejects_invalid_descriptors(
    tmp_path: Path,
    failure: str,
) -> None:
    runtime_lock = tmp_path / "runtime_maintenance.lock"
    wrapper = tmp_path / "submit_research_run.sh"
    other = tmp_path / "other_wrapper.sh"
    runtime_lock.write_text("", encoding="utf-8")
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    other.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    if failure == "symlink":
        wrapper.unlink()
        wrapper.symlink_to(other.name)
    elif failure == "directory":
        wrapper.unlink()
        wrapper.mkdir()
    lock_fd = os.open(runtime_lock, os.O_RDONLY)
    wrapper_fd = os.open(other if failure == "wrong_inode" else wrapper, os.O_RDONLY)
    proc_root = tmp_path / "proc_self_fd"
    proc_root.mkdir()
    (proc_root / str(lock_fd)).symlink_to(runtime_lock)
    target = other if failure == "wrong_target" else wrapper
    (proc_root / str(wrapper_fd)).symlink_to(target)
    if failure == "missing":
        os.close(wrapper_fd)
    try:
        with pytest.raises(RuntimeError, match="wrapper capability"):
            launcher._verify_rq014_wrapper_capability(
                runtime_lock_path=runtime_lock,
                wrapper_path=wrapper,
                proc_fd_root=proc_root,
                runtime_lock_fd=lock_fd,
                wrapper_fd=wrapper_fd,
            )
    finally:
        os.close(lock_fd)
        if failure != "missing":
            os.close(wrapper_fd)


def test_direct_rq014_flag_is_rejected_before_dependency_preload(tmp_path: Path) -> None:
    copied_root = tmp_path / "copied_repo"
    launcher_path = copied_root / "scripts" / "hpc" / "prepare_research_run.py"
    rq014 = copied_root / "scripts" / "rq014"
    launcher_path.parent.mkdir(parents=True)
    rq014.mkdir(parents=True)
    launcher_path.write_bytes(
        (ROOT / "scripts" / "hpc" / "prepare_research_run.py").read_bytes()
    )
    marker = tmp_path / "dependency_preload_executed"
    dependency_payload = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "raise RuntimeError('dependency preload must not execute')\n"
    )
    (rq014 / "materialize_registry.py").write_text(dependency_payload, encoding="utf-8")
    (rq014 / "preflight.py").write_text(dependency_payload, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            str(launcher_path),
            "--rq014-only",
            "--spec",
            str(tmp_path / "missing.json"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "wrapper capability" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "spec, rq014_only, submit, message",
    [
        (
            {"schema_version": 1, "rq_id": "INFRA"},
            True,
            True,
            "only schema v2 / RQ014",
        ),
        (
            {"schema_version": 1, "rq_id": "RQ014"},
            True,
            True,
            "only schema v2 / RQ014",
        ),
        (
            {"schema_version": 1, "rq_id": "INFRA"},
            False,
            True,
            "disabled outside",
        ),
    ],
)
def test_cli_entry_mode_blocks_cross_label_and_generic_submission(
    spec: dict[str, object],
    rq014_only: bool,
    submit: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        launcher._validate_cli_entry_mode(
            spec,
            rq014_only=rq014_only,
            submit=submit,
            wrapper_capability_verified=(
                launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY if rq014_only else None
            ),
        )


def test_cli_entry_mode_accepts_only_rq014_v2_submission() -> None:
    launcher._validate_cli_entry_mode(
        {"schema_version": 2, "rq_id": "RQ014"},
        rq014_only=True,
        submit=True,
        wrapper_capability_verified=launcher._RQ014_VERIFIED_WRAPPER_CAPABILITY,
    )


def test_cli_entry_mode_rejects_rq014_v2_validate_without_internal_wrapper_mode() -> None:
    with pytest.raises(ValueError, match="requires the managed RQ014-only wrapper"):
        launcher._validate_cli_entry_mode(
            {"schema_version": 2, "rq_id": "RQ014"},
            rq014_only=False,
            submit=False,
            wrapper_capability_verified=None,
        )


@pytest.mark.parametrize("caller_value", [False, True, object()])
def test_cli_entry_mode_rejects_caller_supplied_capability_values(
    caller_value: object,
) -> None:
    with pytest.raises(ValueError, match="verified managed-wrapper capability"):
        launcher._validate_cli_entry_mode(
            {"schema_version": 2, "rq_id": "RQ014"},
            rq014_only=True,
            submit=True,
            wrapper_capability_verified=caller_value,
        )


def test_imported_submission_api_also_rejects_generic_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Legacy/generic submission is disabled"):
        launcher.prepare_and_submit(
            {"schema_version": 1, "rq_id": "INFRA"},
            {},
            base=tmp_path,
            repo=tmp_path / "repo",
        )


def test_imported_rq014_submission_api_rejects_caller_boolean_capability(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="verified managed-wrapper capability"):
        launcher.prepare_and_submit(
            {"schema_version": 2, "rq_id": "RQ014"},
            {},
            base=tmp_path,
            repo=tmp_path / "repo",
            spec_bytes=b"{}\n",
            wrapper_capability_verified=True,
        )


def test_submission_wrapper_rejects_checkout_override_before_shell_hooks(
    tmp_path: Path,
) -> None:
    wrapper = ROOT / "scripts" / "hpc" / "submit_research_run.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "${HPC_SOCIALITY_ROOT:-" not in text
    assert "Unsafe launcher environment" in text
    assert "exec python3" not in text
    assert "/share/home/u25310231/ZXC/sociality_estimation/code/repo" not in text
    assert "BASE=/share/home/u25310231/ZXC/sociality_estimation" in text
    assert "WRAPPER=$REPO/scripts/hpc/submit_research_run.sh" in text
    assert "/usr/bin/env -i" in text
    assert '"$PYTHON" -I -S -B -X utf8 "$LAUNCHER" --rq014-only' in text
    assert "managed_python_environment_v3.json" in text
    assert "30de86f702101fbfc8065f6a0d7fd4378daf526d0e55c1197a6a0a147752877a" in text
    assert "0a9944e1de0cf2b4168097b3afe82132333189127976c0a60c0891933853f0d5" in text
    assert "46004531edef17588151114e6e024a85cac7aad063a887ec5d600903b1dfaa9d" in text
    launcher_path = ROOT / "scripts" / "hpc" / "prepare_research_run.py"
    preflight_path = ROOT / "scripts" / "rq014" / "preflight.py"
    materializer_path = ROOT / "scripts" / "rq014" / "materialize_registry.py"
    assert f"LAUNCHER_SHA256={launcher.sha256_file(launcher_path)}" in text
    assert f"PREFLIGHT_SHA256={launcher.sha256_file(preflight_path)}" in text
    assert f"MATERIALIZER_SHA256={launcher.sha256_file(materializer_path)}" in text
    assert f'test "$(/usr/bin/stat -c %s "$LAUNCHER")" = {launcher_path.stat().st_size}' in text
    assert f'test "$(/usr/bin/stat -c %s "$PREFLIGHT")" = {preflight_path.stat().st_size}' in text
    assert (
        f'test "$(/usr/bin/stat -c %s "$MATERIALIZER")" = '
        f"{materializer_path.stat().st_size}"
    ) in text
    assert "registered_count\" = 14326" in text
    assert 'readlink "/proc/$$/fd/8"' in text
    assert 'readlink "/proc/$$/fd/9"' in text
    assert "stat -Lc '%d:%i:%f'" in text
    assert "exec 8>" not in text
    assert "exec 9<" not in text
    assert "Missing or invalid RQ014 wrapper capability descriptors" in text
    assert "registered_total\" = 307357072" in text
    assert '! -type d ! -type f -print -quit' in text
    assert "sha256sum --check --strict" in text
    assert "# row_count=20" in text
    assert 'test "$native_rows" = 20' in text
    assert 'test "$native_symlinks" = 16' in text
    assert text.index("sha256sum --check --strict") < text.index(
        '"$LAUNCHER" --rq014-only'
    )
    assert "--export=NONE" not in text

    marker = tmp_path / "shell_hook_executed"
    hook = tmp_path / "hostile_shell_hook.sh"
    hook.write_text(f"/usr/bin/touch {marker}\n", encoding="utf-8")
    result = subprocess.run(
        ["/bin/sh", str(wrapper), "--help"],
        text=True,
        capture_output=True,
        env={
            "HPC_SOCIALITY_ROOT": str(tmp_path / "attacker_checkout"),
            "BASH_ENV": str(hook),
            "ENV": str(hook),
        },
        check=False,
    )
    assert result.returncode == 64
    assert "env -i operator command" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "HPC_SOCIALITY_ROOT",
        "BASH_ENV",
        "ENV",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "SBATCH_EXPORT",
    ],
)
def test_submission_wrapper_rejects_each_inherited_control_variable(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    wrapper = ROOT / "scripts" / "hpc" / "submit_research_run.sh"
    harmless = tmp_path / "harmless"
    harmless.write_text("", encoding="utf-8")
    result = subprocess.run(
        ["/bin/sh", str(wrapper), "--help"],
        text=True,
        capture_output=True,
        env={unsafe_name: str(harmless)},
        check=False,
    )
    assert result.returncode == 64


def test_rq014_readme_registers_the_clean_operator_boundary() -> None:
    readme = (ROOT / "configs" / "run_specs" / "README.md").read_text(encoding="utf-8")
    assert "/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c" in readme
    assert 'exec 9<"$wrapper"' in readme
    assert "/proc/$$/fd/9" in readme
    assert 'exec /bin/sh /proc/$$/fd/9 "$@"' in readme
    wrapper = ROOT / "scripts" / "hpc" / "submit_research_run.sh"
    assert launcher.sha256_file(wrapper) in readme
    assert "/usr/bin/sha256sum --check --strict -" in readme
    assert 'exec 8>"$lock" && /usr/bin/flock -s 8 &&' in readme
    assert readme.index("/usr/bin/env -i") < readme.index("/usr/bin/sha256sum --check")
    assert "/manifests/RQ014/run_specs/REPLACE_RUN_ID.json" in readme
    assert "never reopens the spec path" in readme
    assert "scripts.rq014.materialize_registry" in readme
    assert "empty\n`__path__`" in readme
    assert "cannot undo a dynamic-loader hook" in readme


def test_exact_path_python_bootstrap_blocks_site_and_shadow_injection(
    tmp_path: Path,
) -> None:
    code = tmp_path / "reviewed_code"
    rq014 = code / "scripts" / "rq014"
    rq014.mkdir(parents=True)
    preflight_path = rq014 / "preflight.py"
    materializer_path = rq014 / "materialize_registry.py"
    preflight_path.write_bytes((ROOT / "scripts" / "rq014" / "preflight.py").read_bytes())
    materializer_path.write_bytes(
        (ROOT / "scripts" / "rq014" / "materialize_registry.py").read_bytes()
    )
    (rq014 / "run_managed_g2.py").write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import scripts\n"
        "import scripts.rq014\n"
        "from scripts.rq014 import preflight\n"
        "try:\n"
        "    preflight.validate_materialization_ledger(\n"
        "        ledger_path=Path(sys.argv[2]), repo_root=Path.cwd(), contract={}\n"
        "    )\n"
        "except preflight.ContractError:\n"
        "    pass\n"
        "else:\n"
        "    raise RuntimeError('missing ledger unexpectedly validated')\n"
        "materializer = sys.modules.get('scripts.rq014.materialize_registry')\n"
        "if materializer is None:\n"
        "    raise RuntimeError('late-import materializer was not preloaded')\n"
        "actual = str(Path(materializer.__file__).resolve())\n"
        "expected = str(Path(sys.argv[3]).resolve())\n"
        "if actual != expected:\n"
        "    raise RuntimeError('materializer did not come from the exact reviewed path')\n"
        "if tuple(scripts.__path__) or tuple(scripts.rq014.__path__):\n"
        "    raise RuntimeError('ordinary package-path import was enabled')\n"
        "Path(sys.argv[1]).write_text(\n"
        "    json.dumps({'materializer_path': actual, 'package_paths_empty': True}, sort_keys=True),\n"
        "    encoding='utf-8',\n"
        ")\n",
        encoding="utf-8",
    )
    hostile = tmp_path / "hostile_imports"
    hostile.mkdir()
    marker = tmp_path / "python_hook_executed"
    hostile_payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n"
    (hostile / "sitecustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "usercustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "json.py").write_text(
        hostile_payload + "raise RuntimeError('shadow json imported')\n",
        encoding="utf-8",
    )
    hostile_rq014 = hostile / "scripts" / "rq014"
    hostile_rq014.mkdir(parents=True)
    (hostile / "scripts" / "__init__.py").write_text(hostile_payload, encoding="utf-8")
    (hostile_rq014 / "__init__.py").write_text(hostile_payload, encoding="utf-8")
    (hostile_rq014 / "materialize_registry.py").write_text(
        hostile_payload + "raise RuntimeError('shadow materializer imported')\n",
        encoding="utf-8",
    )
    output = tmp_path / "bootstrap_output.json"
    missing_ledger = tmp_path / "missing_ledger.json"
    isolated_sys_path = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-X",
                "utf8",
                "-c",
                "import json,sys; print(json.dumps(sys.path))",
            ],
            text=True,
        )
    )
    command = launcher._rq014_isolated_python_command(
        python=Path(sys.executable),
        code=code,
        entrypoint="scripts/rq014/run_managed_g2.py",
        arguments=[str(output), str(missing_ledger), str(materializer_path)],
        isolated_sys_path=isolated_sys_path,
    )
    assert command[1:6] == ["-I", "-S", "-B", "-X", "utf8"]
    assert str(materializer_path) in command
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(hostile),
        PYTHONUSERBASE=str(hostile),
        PYTHONHOME=str(hostile),
    )
    subprocess.run(command, cwd=hostile, env=environment, check=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "materializer_path": str(materializer_path.resolve()),
        "package_paths_empty": True,
    }
    assert not marker.exists()


@pytest.mark.parametrize("materializer_state", ["missing", "wrong_api"])
def test_exact_path_python_bootstrap_fails_closed_for_bad_materializer(
    tmp_path: Path,
    materializer_state: str,
) -> None:
    code = tmp_path / "reviewed_code"
    rq014 = code / "scripts" / "rq014"
    rq014.mkdir(parents=True)
    (rq014 / "preflight.py").write_bytes(
        (ROOT / "scripts" / "rq014" / "preflight.py").read_bytes()
    )
    materializer_path = rq014 / "materialize_registry.py"
    if materializer_state == "wrong_api":
        materializer_path.write_text("WRONG_API = True\n", encoding="utf-8")
    output = tmp_path / "entrypoint_executed"
    (rq014 / "run_managed_g2.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from scripts.rq014 import preflight\n"
        "preflight.validate_materialization_ledger(\n"
        "    ledger_path=Path(sys.argv[2]), repo_root=Path.cwd(), contract={}\n"
        ")\n"
        "Path(sys.argv[1]).write_text('unexpected entrypoint success', encoding='utf-8')\n",
        encoding="utf-8",
    )
    isolated_sys_path = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-X",
                "utf8",
                "-c",
                "import json,sys; print(json.dumps(sys.path))",
            ],
            text=True,
        )
    )
    command = launcher._rq014_isolated_python_command(
        python=Path(sys.executable),
        code=code,
        entrypoint="scripts/rq014/run_managed_g2.py",
        arguments=[str(output), str(tmp_path / "missing_ledger.json")],
        isolated_sys_path=isolated_sys_path,
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "hostile")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_isolated_launcher_validation_preloads_exact_materializer(
    tmp_path: Path,
) -> None:
    hostile = tmp_path / "hostile_launcher_late_import"
    hostile_rq014 = hostile / "scripts" / "rq014"
    hostile_rq014.mkdir(parents=True)
    marker = tmp_path / "isolated_launcher_shadow_executed"
    hostile_payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n"
    (hostile / "sitecustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "usercustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "json.py").write_text(
        hostile_payload + "raise RuntimeError('shadow json imported')\n",
        encoding="utf-8",
    )
    (hostile / "scripts" / "__init__.py").write_text(hostile_payload, encoding="utf-8")
    (hostile_rq014 / "__init__.py").write_text(hostile_payload, encoding="utf-8")
    (hostile_rq014 / "materialize_registry.py").write_text(
        hostile_payload + "raise RuntimeError('shadow materializer imported')\n",
        encoding="utf-8",
    )
    launcher_path = ROOT / "scripts" / "hpc" / "prepare_research_run.py"
    materializer_path = ROOT / "scripts" / "rq014" / "materialize_registry.py"
    probe = """
import importlib.util
import json
import sys
from pathlib import Path

launcher_path, expected_materializer, missing_ledger = sys.argv[1:]
spec = importlib.util.spec_from_file_location("_rq014_isolated_launcher_probe", launcher_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot construct launcher probe loader")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
try:
    launcher.validate_materialization_ledger(
        ledger_path=Path(missing_ledger), repo_root=Path.cwd(), contract={}
    )
except launcher.RQ014ContractError:
    pass
else:
    raise RuntimeError("missing ledger unexpectedly validated")
materializer = sys.modules.get("scripts.rq014.materialize_registry")
if materializer is None:
    raise RuntimeError("launcher did not preload the registry materializer")
actual = str(Path(materializer.__file__).resolve())
if actual != str(Path(expected_materializer).resolve()):
    raise RuntimeError("launcher loaded the registry materializer from the wrong path")
if tuple(sys.modules["scripts"].__path__) or tuple(sys.modules["scripts.rq014"].__path__):
    raise RuntimeError("launcher enabled ordinary package-path imports")
print(json.dumps({"materializer_path": actual}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(hostile),
        PYTHONUSERBASE=str(hostile),
        PYTHONHOME=str(hostile),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            "-c",
            probe,
            str(launcher_path),
            str(materializer_path),
            str(tmp_path / "missing_ledger.json"),
        ],
        cwd=hostile,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "materializer_path": str(materializer_path.resolve())
    }
    assert not marker.exists()


def test_launcher_startup_blocks_site_and_shadow_injection(tmp_path: Path) -> None:
    hostile = tmp_path / "hostile_launcher_imports"
    hostile.mkdir()
    marker = tmp_path / "launcher_hook_executed"
    hostile_payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n"
    (hostile / "sitecustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "usercustomize.py").write_text(hostile_payload, encoding="utf-8")
    (hostile / "json.py").write_text(
        hostile_payload + "raise RuntimeError('shadow json imported')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(hostile),
        PYTHONUSERBASE=str(hostile),
        PYTHONHOME=str(hostile),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            str(ROOT / "scripts" / "hpc" / "prepare_research_run.py"),
            "--help",
        ],
        cwd=hostile,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Validate and submit one immutable" in result.stdout
    assert not marker.exists()


def test_git_config_hooks_fsmonitor_and_checkout_filters_cannot_enter_snapshot(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    subprocess.run([launcher.SYSTEM_GIT, "init", str(repo)], check=True, capture_output=True)
    source = repo / "scripts" / "rq014" / "preflight.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("*.py filter=hostile\n", encoding="utf-8")
    subprocess.run([launcher.SYSTEM_GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            launcher.SYSTEM_GIT,
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )

    marker = tmp_path / "unreviewed_git_extension_executed"
    hostile = tmp_path / "hostile_git_extension.sh"
    hostile.write_text(f"#!/bin/sh\n/usr/bin/touch {marker}\n", encoding="utf-8")
    hostile.chmod(0o755)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    post_checkout = hooks / "post-checkout"
    post_checkout.write_bytes(hostile.read_bytes())
    post_checkout.chmod(0o755)
    for key, value in (
        ("core.hooksPath", str(hooks)),
        ("core.fsmonitor", str(hostile)),
        ("filter.hostile.smudge", str(hostile)),
        ("filter.hostile.clean", str(hostile)),
        ("filter.hostile.required", "true"),
    ):
        subprocess.run(
            [launcher.SYSTEM_GIT, "-C", str(repo), "config", key, value],
            check=True,
        )

    command = launcher._git_command(repo, "rev-parse", "HEAD")
    assert "core.hooksPath=/dev/null" in command
    assert "core.fsmonitor=false" in command
    assert launcher.GIT_COMMAND_ENV["GIT_CONFIG_NOSYSTEM"] == "1"
    assert launcher.GIT_COMMAND_ENV["GIT_CONFIG_GLOBAL"] == "/dev/null"
    commit = launcher.run_git(repo, "rev-parse", "HEAD")

    snapshot_parent = tmp_path / "run"
    snapshot_parent.mkdir()
    code = snapshot_parent / "code"
    receipt = launcher._materialize_rq014_code_snapshot(
        repo=repo,
        code=code,
        commit=commit,
        registered={"scripts/rq014/preflight.py": launcher.sha256_file(source)},
    )
    assert json.loads(receipt)["files"][0]["path"] == "scripts/rq014/preflight.py"
    assert (code / "scripts" / "rq014" / "preflight.py").read_bytes() == source.read_bytes()
    assert not (code / ".git").exists()
    assert not marker.exists()


def test_rq014_sbatch_is_fixed_rating_blind_and_digest_named(tmp_path: Path) -> None:
    base = tmp_path / "sociality_estimation"
    repo = ROOT
    run_root = base / "work_dirs" / "RQ014" / "run1"
    code = run_root / "code"
    validated = {
        "job_name": "zxc-rq014-pre-123456789abc",
        "commit": "a" * 40,
        "run_spec_sha256": "1" * 64,
        "authorization_sha256": "2" * 64,
        "execution_contract_sha256": "3" * 64,
        "formal_g1_path": str(ROOT / "formal.json"),
        "formal_g1_relative_path": "formal.json",
        "formal_g1_sha256": "4" * 64,
        "environment_manifest_path": str(base / "manifests" / "RQ014" / "environment.json"),
        "environment_manifest_sha256": "8" * 64,
        "python_executable_path": str(base / "envs" / "ipv-exact-sigma01" / "bin" / "python3.9"),
        "python_executable_sha256": "9" * 64,
        "isolated_sys_path": ["/managed/python39.zip", "/managed/python3.9", "/managed/lib-dynload"],
        "stdlib_root": str(base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9"),
        "lib_dynload_root": str(
            base / "envs" / "ipv-exact-sigma01" / "lib" / "python3.9" / "lib-dynload"
        ),
        "python_zip_path": str(
            base / "envs" / "ipv-exact-sigma01" / "lib" / "python39.zip"
        ),
        "stdlib_regular_file_count": 10,
        "stdlib_regular_file_total_size_bytes": 1000,
        "stdlib_checksum_manifest_path": str(
            base / "manifests" / "RQ014" / "managed_python_stdlib_v1.sha256"
        ),
        "stdlib_checksum_manifest_sha256": "f" * 64,
        "native_library_manifest_path": str(
            base / "manifests" / "RQ014" / "managed_python_native_libs_v1.tsv"
        ),
        "native_library_manifest_sha256": "0" * 64,
        "native_library_row_count": 20,
        "native_library_total_size_bytes": 14656296,
        "native_library_symlink_row_count": 16,
        "input_manifest_path": str(base / "manifests" / "RQ014" / "input.json"),
        "input_manifest_sha256": "5" * 64,
        "sanitization_receipt_path": str(base / "inputs" / "RQ014" / "receipt.json"),
        "sanitization_receipt_sha256": "6" * 64,
        "materialization_ledger_path": str(base / "manifests" / "RQ014" / "ledger.json"),
        "materialization_ledger_sha256": "7" * 64,
        "declassification_export_receipt_path": str(
            base / "work_dirs" / "RQ014" / "export" / "outputs" / "receipt.json"
        ),
        "declassification_export_receipt_sha256": "a" * 64,
        "declassification_export_done_path": str(
            base / "work_dirs" / "RQ014" / "export" / "outputs" / "DONE.json"
        ),
        "declassification_export_done_sha256": "b" * 64,
        "contract_bundle_path": str(ROOT / "reports" / "plans" / "RQ014_plan_v1p3_checksums_20260711.sha256"),
        "contract_bundle_relative_path": "reports/plans/RQ014_plan_v1p3_checksums_20260711.sha256",
        "contract_bundle_sha256": "e" * 64,
        "code_snapshot_receipt_sha256": "c" * 64,
        "entrypoint": "scripts/rq014/run_managed_g2.py",
        "slurm_profile": {
            "partition": "amd",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 2,
            "memory": "4G",
            "time": "01:00:00",
        },
    }
    script = render_rq014_sbatch(
        validated=validated,
        base=base,
        repo=repo,
        run_root=run_root,
        code=code,
        sealed_spec_path=run_root / "manifests" / "run_spec.json",
    )
    assert "#SBATCH --job-name=zxc-rq014-pre-123456789abc" in script
    assert "scripts/rq014/run_managed_g2.py" in script
    assert "contract-preflight" in script
    assert "--input-manifest" in script
    assert "--sanitization-receipt" in script
    assert "--materialization-ledger" in script
    assert "--declassification-export-receipt" in script
    assert "--declassification-export-done" in script
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --export=NIL" in script
    assert "#SBATCH --chdir=/" in script
    assert "/usr/bin/sha256sum" in script
    assert "/usr/bin/awk" in script
    assert "/usr/bin/find" in script
    assert "managed_python_stdlib_v1.sha256" in script
    assert "/usr/bin/sha256sum --check --strict" in script
    assert "\nNone" not in script
    assert "# rq014-managed-python-native-libs-v1" in script
    assert 'test "$native_rows" = 20' in script
    assert "/usr/bin/flock -s 8" in script
    assert "exec /usr/bin/env -i" in script
    assert " -I -S -B -X utf8 -c " in script
    assert "export PYTHONPATH=" not in script
    assert "${BASH_ENV-}${ENV-}${LD_PRELOAD-}${PYTHONHOME-}${PYTHONPATH-}" in script
    assert "git -C" not in script
    bundle_check = (
        "RQ014_plan_v1p3_checksums_20260711.sha256 | /usr/bin/awk"
    )
    assert bundle_check in script
    assert script.index(bundle_check) < script.index("/usr/bin/sha256sum -c")
    assert script.index("/usr/bin/sha256sum --check --strict") < script.index(
        " -I -S -B -X utf8 -c "
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert f"{name}=1" in script
    assert "/ZXC/ipv_estimation" not in script
    assert "/ZXC/RQ014_recovery" not in script

    rendered = tmp_path / "run.sbatch"
    rendered.write_text(script, encoding="utf-8")
    subprocess.run(["/bin/bash", "-n", str(rendered)], check=True)
