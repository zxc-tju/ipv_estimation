"""Regression tests for the fail-closed RQ014 FL05 forensic indexer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEXER = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.py"
SHELL_WRAPPER = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.sh"
SBATCH_WRAPPER = REPO_ROOT / "reports/plans/prompts/RQ014_forensics_hpc_fl05_v1p3.sbatch"


def run_indexer(
    roots: List[Path],
    artifact_dir: Path,
    *,
    audit_format: str = "json",
    env: Optional[Dict[str, str]] = None,
    login_budget: int = 209_715_200,
    compute_budget: int = 4_294_967_296,
) -> Tuple[subprocess.CompletedProcess[str], Path]:
    bundle_root = artifact_dir / "fl05_bundle"
    command = [
        sys.executable,
        str(INDEXER),
        "--bundle-root",
        str(bundle_root),
        "--audit-format",
        audit_format,
        "--login-byte-budget",
        str(login_budget),
        "--compute-byte-budget",
        str(compute_budget),
    ]
    for root in roots:
        command.extend(["--root", str(root)])
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    return completed, bundle_root


def current_generation(bundle_root: Path) -> Tuple[str, Path]:
    generation_id = (bundle_root / "CURRENT").read_text(encoding="utf-8").strip()
    generation = bundle_root / "generations" / generation_id
    assert generation.is_dir()
    assert (generation / "DONE").is_file()
    return generation_id, generation


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mixed_fixture(root: Path) -> None:
    root.mkdir()
    (root / "01_wide.csv").write_text(
        "scene_id,spearman_rho,n,variable_x,variable_y,dataset,candidate_scope\n"
        "s1,-0.42,75,ipv_deviation,human_rating,WOD-E2E,all_candidates\n"
        "s2,NA,20,ipv_deviation,human_rating,WOD-E2E,all_candidates\n"
        "malformed,row\n",
        encoding="utf-8",
    )
    (root / "02_long.csv").write_text(
        "statistic_name,value,n,unit,predictor,outcome,dataset_name,scope\n"
        "pearson_corr,-0.31,71,scene,ipv_deviation,human_rating,WOD-E2E,valid\n"
        "spearman_rho,not-a-number,70,scene,ipv_deviation,human_rating,WOD-E2E,valid\n",
        encoding="utf-8",
    )
    (root / "03_stats.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "statistic_name": "spearman_rho",
                        "value": -0.5,
                        "n": 68,
                        "unit": "scene",
                        "variable_x": "ipv_deviation",
                        "variable_y": "human_rating",
                        "dataset": "WOD-E2E",
                        "candidate_scope": "complete_candidates",
                    }
                ],
                "wide": {"kendall_corr": 0.2, "n_scenes": 50},
                "bad": {"rho": "NA", "n": 40},
            }
        ),
        encoding="utf-8",
    )
    (root / "04_notes.md").write_text(
        "At 4 Hz, Spearman rho=-0.42 (n=75).\n"
        "pearson_corr=NA for the incomplete subset.\n"
        "A Kendall correlation was discussed without an attached value.\n",
        encoding="utf-8",
    )
    (root / "05_zero_hit.md").write_text(
        "This file intentionally contains no recorded association statistic.\n",
        encoding="utf-8",
    )


def test_generation_indexes_all_layouts_and_audits_zero_hit(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    build_mixed_fixture(root)

    result, bundle = run_indexer([root], artifacts)

    assert result.returncode == 0, result.stderr
    generation_id, generation = current_generation(bundle)
    assert set(path.name for path in generation.iterdir()) == {
        "historical_stats_index.csv",
        "fl05_file_audit.json",
        "fl05_run_manifest.json",
        "DONE",
    }
    rows = read_csv(generation / "historical_stats_index.csv")
    parsed = [row for row in rows if row["parse_status"].startswith("PARSED_")]
    unparsed = [row for row in rows if row["parse_status"] == "UNPARSED_CANDIDATE"]
    assert {float(row["value"]) for row in parsed} >= {-0.5, -0.42, -0.31, 0.2}
    assert len(unparsed) >= 5
    assert all(row["fingerprint_disposition"] == "UNADJUDICATED" for row in parsed)
    assert all(row["fingerprint_disposition"] == "NOT_EVALUABLE" for row in unparsed)

    markdown_rows = [row for row in rows if row["source_file"].endswith("04_notes.md")]
    local_rho = [row for row in markdown_rows if row["parse_status"] == "PARSED_MD_KEY_VALUE"]
    assert [float(row["value"]) for row in local_rho] == [-0.42]
    assert local_rho[0]["value"] != "4"
    long_row = next(row for row in rows if row["statistic_name"] == "pearson_corr")
    assert long_row["variable_x"] == "ipv_deviation"
    assert long_row["variable_y"] == "human_rating"
    assert long_row["dataset_scope"] == "WOD-E2E"
    assert long_row["candidate_scope"] == "valid"

    audit_doc = json.loads((generation / "fl05_file_audit.json").read_text(encoding="utf-8"))
    assert len(audit_doc["files"]) == 5
    assert all("error" in record for record in audit_doc["files"])
    assert any(record["error"] for record in audit_doc["files"])
    zero_hit = next(
        item for item in audit_doc["files"] if item["source_file"].endswith("05_zero_hit.md")
    )
    assert zero_hit["parser_status"] == "PARSED_ZERO_HIT"
    assert zero_hit["row_count"] == 0
    assert zero_hit["error"] == ""
    assert zero_hit["bytes"] > 0
    assert len(zero_hit["source_file_sha256"]) == 64

    manifest_doc = json.loads(
        (generation / "fl05_run_manifest.json").read_text(encoding="utf-8")
    )
    done_doc = json.loads((generation / "DONE").read_text(encoding="utf-8"))
    assert manifest_doc["generation_id"] == generation_id
    assert manifest_doc["status"] == "COMPLETE_WITH_UNPARSED_CANDIDATES"
    assert manifest_doc["fingerprint_policy"] == "RECORDED_FIELDS_ONLY_UNADJUDICATED"
    assert manifest_doc["artifacts"]["output"]["sha256"]
    assert manifest_doc["artifacts"]["audit"]["sha256"]
    assert done_doc["generation_id"] == generation_id
    assert done_doc["status"] == "READY"


def test_generation_supports_csv_audit(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    build_mixed_fixture(root)

    result, bundle = run_indexer([root], artifacts, audit_format="csv")

    assert result.returncode == 0, result.stderr
    _, generation = current_generation(bundle)
    audit_rows = read_csv(generation / "fl05_file_audit.csv")
    assert len(audit_rows) == 5
    assert {
        "source_file",
        "bytes",
        "source_file_sha256",
        "mtime",
        "parser_status",
        "row_count",
        "error",
    }.issubset(audit_rows[0])


def test_nonempty_fully_audited_zero_candidate_tree_is_publishable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "notes.md").write_text(
        "This complete source contains no association statistic.\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    result, bundle = run_indexer([root], artifacts)

    assert result.returncode == 0, result.stderr
    _, generation = current_generation(bundle)
    assert read_csv(generation / "historical_stats_index.csv") == []
    manifest = json.loads(
        (generation / "fl05_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "COMPLETE_ZERO_CANDIDATES"
    assert manifest["row_count"] == 0
    audit = json.loads(
        (generation / "fl05_file_audit.json").read_text(encoding="utf-8")
    )
    assert audit["files"][0]["parser_status"] == "PARSED_ZERO_HIT"
    assert audit["files"][0]["error"] == ""


def test_missing_or_empty_required_root_fails_without_current(tmp_path: Path) -> None:
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    (valid_root / "stats.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    for index, bad_root in enumerate((tmp_path / "missing", empty_root)):
        artifact_dir = tmp_path / f"artifacts-{index}"
        artifact_dir.mkdir()
        result, bundle = run_indexer([valid_root, bad_root], artifact_dir)
        assert result.returncode != 0
        assert "FL05_FATAL" in result.stderr
        assert not (bundle / "CURRENT").exists()


def test_zero_byte_supported_tree_fails_without_current(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "empty.md").write_bytes(b"")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    result, bundle = run_indexer([root], artifacts)

    assert result.returncode != 0
    assert "no non-empty supported files" in result.stderr
    assert not (bundle / "CURRENT").exists()


def test_bundle_root_overlapping_input_fails_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "stats.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    artifact_dir = root / "nested-output"
    artifact_dir.mkdir()

    result, bundle = run_indexer([root], artifact_dir)

    assert result.returncode != 0
    assert "must be disjoint" in result.stderr
    assert not (bundle / "CURRENT").exists()


def test_parse_failure_preserves_current_generation(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "01_valid.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    first_result, bundle = run_indexer([root], artifacts)
    assert first_result.returncode == 0, first_result.stderr
    first_id, first_generation = current_generation(bundle)
    prior_output = (first_generation / "historical_stats_index.csv").read_bytes()
    prior_generations = set((bundle / "generations").iterdir())

    (root / "99_broken.json").write_text('{"rho": -0.4', encoding="utf-8")
    failed_result, _ = run_indexer([root], artifacts)

    assert failed_result.returncode != 0
    current_id, current_path = current_generation(bundle)
    assert current_id == first_id
    assert (current_path / "historical_stats_index.csv").read_bytes() == prior_output
    assert set((bundle / "generations").iterdir()) == prior_generations
    assert not list((bundle / "generations").glob(".staging-*"))


def test_pointer_publication_failure_leaves_only_invisible_orphan(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "stats.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    first_result, bundle = run_indexer([root], artifacts)
    assert first_result.returncode == 0, first_result.stderr
    first_id, first_generation = current_generation(bundle)
    prior_output = (first_generation / "historical_stats_index.csv").read_bytes()
    prior_generation_names = {path.name for path in (bundle / "generations").iterdir()}

    spec = importlib.util.spec_from_file_location("rq014_fl05_indexer_v1p3", INDEXER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    real_replace = module.os.replace

    def fail_current_replace(source: Path, destination: Path) -> None:
        if Path(destination).name == "CURRENT":
            raise OSError("injected CURRENT publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_current_replace)
    return_code = module.main(
        ["--root", str(root), "--bundle-root", str(bundle), "--audit-format", "json"]
    )

    assert return_code != 0
    current_id, current_path = current_generation(bundle)
    assert current_id == first_id
    assert (current_path / "historical_stats_index.csv").read_bytes() == prior_output
    new_generation_names = {path.name for path in (bundle / "generations").iterdir()}
    orphan_names = new_generation_names - prior_generation_names
    assert len(orphan_names) == 1
    orphan_id = orphan_names.pop()
    orphan = bundle / "generations" / orphan_id
    assert (orphan / "DONE").is_file()
    module.validate_generation(orphan, orphan_id)


def test_root_or_any_entry_symlink_fails_closed(tmp_path: Path) -> None:
    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / "stats.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(target_root, target_is_directory=True)
    first_artifacts = tmp_path / "artifacts-root-link"
    first_artifacts.mkdir()
    result, bundle = run_indexer([linked_root], first_artifacts)
    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert not (bundle / "CURRENT").exists()

    clean_root = tmp_path / "root-with-link"
    clean_root.mkdir()
    (clean_root / "stats.csv").write_text("rho\n-0.3\n", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    (clean_root / "unsupported-but-forbidden-link.txt").symlink_to(external)
    second_artifacts = tmp_path / "artifacts-entry-link"
    second_artifacts.mkdir()
    result, bundle = run_indexer([clean_root], second_artifacts)
    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert not (bundle / "CURRENT").exists()


def test_compute_budget_requires_verified_slurm_wrapper_context(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "stats.csv").write_text(
        "statistic_name,value,notes\n"
        "spearman_rho,-0.3," + ("x" * 512) + "\n",
        encoding="utf-8",
    )
    login_artifacts = tmp_path / "login-artifacts"
    login_artifacts.mkdir()
    login_result, login_bundle = run_indexer(
        [root], login_artifacts, login_budget=64, compute_budget=4096
    )
    assert login_result.returncode != 0
    assert not (login_bundle / "CURRENT").exists()

    unverified_artifacts = tmp_path / "unverified-artifacts"
    unverified_artifacts.mkdir()
    unverified_env = os.environ.copy()
    unverified_env.update({"SLURM_JOB_ID": "12345", "SLURM_JOB_NAME": "zxc-rq014-fl05"})
    result, bundle = run_indexer(
        [root], unverified_artifacts, env=unverified_env, login_budget=64, compute_budget=4096
    )
    assert result.returncode != 0
    assert not (bundle / "CURRENT").exists()

    slurm_artifacts = tmp_path / "slurm-artifacts"
    slurm_artifacts.mkdir()
    slurm_env = unverified_env.copy()
    slurm_env["RQ014_FL05_VERIFIED_SLURM_WRAPPER"] = (
        "v1p3:12345:zxc-rq014-fl05"
    )
    python_realpath = Path(sys.executable).resolve()
    slurm_env["RQ014_FL05_PYTHON_REALPATH"] = str(python_realpath)
    slurm_env["RQ014_FL05_PYTHON_SHA256"] = sha256_file(python_realpath)
    slurm_result, bundle = run_indexer(
        [root], slurm_artifacts, env=slurm_env, login_budget=64, compute_budget=4096
    )
    assert slurm_result.returncode == 0, slurm_result.stderr
    _, generation = current_generation(bundle)
    manifest_doc = json.loads(
        (generation / "fl05_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_doc["execution_context"]["mode"] == "slurm_compute_node"
    assert manifest_doc["execution_context"]["checksum_verified_wrapper"] is True
    assert manifest_doc["byte_budget"] == 4096


def test_wrappers_are_syntax_valid_and_sbatch_is_fixed_scope() -> None:
    for wrapper in (SHELL_WRAPPER, SBATCH_WRAPPER):
        result = subprocess.run(
            ["bash", "-n", str(wrapper)], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    sbatch_text = SBATCH_WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=zxc-rq014-fl05" in sbatch_text
    assert "#SBATCH --output=" in sbatch_text
    assert "#SBATCH --error=" in sbatch_text
    assert "[[ $# -ne 1 ]]" in sbatch_text
    assert 'durable_root="/share/home/u25310231/ZXC/RQ014_recovery"' in sbatch_text
    assert 'run_root="${durable_root}/${run_id}"' in sbatch_text
    assert "/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis" in sbatch_text
    assert "/share/home/u25310231/ZXC/RQ010B_wod_e2e/results" in sbatch_text
    assert "RQ014_plan_v1p3_checksums_20260711.sha256" in sbatch_text
    assert "RQ014_forensics_hpc_fl05_indexer_v1p3.py" in sbatch_text
    assert "RQ014_forensics_hpc_fl05_indexer_v1p3.sh" in sbatch_text
    assert "RQ014_forensics_hpc_fl05_v1p3.sbatch" in sbatch_text
    assert "RQ014_FL05_VERIFIED_SLURM_WRAPPER" in sbatch_text
    assert "/share/home/u25310231/.conda/envs/ipv/bin/python" in sbatch_text
    assert "RQ014_FL05_PYTHON_SHA256" in sbatch_text
    assert 'readlink -f -- "${python_bin}"' in sbatch_text
    assert '|| -L "${python_bin}"' not in sbatch_text
    assert "zxc-rq014-fl05-${SLURM_JOB_ID}.out" in sbatch_text
    assert "#SBATCH --output=/share/home/u25310231/ZXC/RQ014_recovery/" in sbatch_text
    assert '"$@"' not in sbatch_text

    extra_argument_result = subprocess.run(
        ["bash", str(SBATCH_WRAPPER), "safe-run", "unexpected"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert extra_argument_result.returncode == 64
    assert "RUN_ID" in extra_argument_result.stderr
