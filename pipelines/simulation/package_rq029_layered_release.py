#!/usr/bin/env python3
"""Package RQ029 synthetic human trajectories into two governed data layers."""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data/derived/rq029_human_synthetic_microdata/v2_paper_aligned"
DEFAULT_OUTPUT = REPO_ROOT / "data/derived/rq029_human_synthetic_release/v1"
SCHEMA_VERSION = "RQ029-layered-synthetic-release-v1"
DATA_STATUS = "SYNTHETIC_NOT_OBSERVED"

LAYER_1_TABLES = (
    "runs.csv",
    "scenario_templates.csv",
    "ego_warp_metrics.csv",
    "trajectory_pairs.parquet",
    "designated_counterpart_template.parquet",
)
LAYER_2_TABLES = (
    "candidate_moments.parquet",
    "both_gate_moments.parquet",
    "counterpart_windows.parquet",
    "per_unit_counts.csv",
    "synthetic_run_calibration.csv",
    "target_summary.json",
    "achieved_summary.json",
)

AUDIT_DIRS = (
    "RQ029_2_paper_aligned_20260902",
    "RQ029_3_trajectory_continuity_audit_20260904",
    "RQ029_4_paper_claim_parity_audit_20260904",
)

CORE_METRICS = (
    "drivers_scenarios_runs_missing",
    "candidate_gate1_both",
    "alpha90_below_above_inside_total",
    "alpha90_rate",
    "alpha90_scenario_range",
    "alpha90_run_bootstrap_ci95",
    "av_to_human_ratio",
    "av_to_human_scenario_bootstrap_ci95",
    "scenarios_av_higher",
    "q75_lower_to_inside_ratio",
    "ttc_lt2_difference_ci95",
    "speed_drop_ratio",
    "speed_drop_ratio_ci95",
    "speed_range_ratio",
    "speed_range_ratio_ci95",
    "brake_share_difference",
    "brake_difference_ci95",
    "nominal_sampling_interval_s",
)

EVIDENCE_ROWS = (
    {
        "evidence_id": "L2-E01",
        "claim_family": "C7 real-human aggregate record",
        "artifact": "01_authoritative_real_aggregate/human_arm_data.json",
        "authority": "DIRECT_SUPPORT",
        "paper_use": "Allowed with RQ022 decision wording and boundaries",
        "synthetic": False,
    },
    {
        "evidence_id": "L2-E02",
        "claim_family": "C7 acceptance and wording constraints",
        "artifact": "01_authoritative_real_aggregate/RQ022_decision.md",
        "authority": "DIRECT_SUPPORT",
        "paper_use": "Allowed; decision controls interpretation",
        "synthetic": False,
    },
    {
        "evidence_id": "L2-E03",
        "claim_family": "AV reference values",
        "artifact": "01_authoritative_real_aggregate/av_reference_values.json",
        "authority": "DIRECT_SUPPORT",
        "paper_use": "Allowed as frozen AV aggregate reference",
        "synthetic": False,
    },
    {
        "evidence_id": "L2-E04",
        "claim_family": "Paper display/calibration target contract",
        "artifact": "02_synthetic_analysis_rehearsal/contracts/paper_record_targets_v2.json",
        "authority": "CALIBRATION_CONTRACT_NOT_EVIDENCE",
        "paper_use": "Defines comparison targets; cannot independently support claims",
        "synthetic": True,
    },
    {
        "evidence_id": "L2-E05",
        "claim_family": "Paper-aligned counts and point estimates",
        "artifact": "02_synthetic_analysis_rehearsal/tables/achieved_summary.json",
        "authority": "METHOD_REHEARSAL_ONLY",
        "paper_use": "Cannot serve as independent empirical evidence",
        "synthetic": True,
    },
    {
        "evidence_id": "L2-E06",
        "claim_family": "Paper claim parity audit",
        "artifact": "03_evidence_and_audits/RQ029_4_paper_claim_parity_audit_20260904/REPORT.md",
        "authority": "BOUNDARY_AND_QA",
        "paper_use": "Use to prevent overclaim; not a real-human result",
        "synthetic": True,
    },
    {
        "evidence_id": "L2-E07",
        "claim_family": "Trajectory continuity",
        "artifact": "03_evidence_and_audits/RQ029_3_trajectory_continuity_audit_20260904/REPORT.md",
        "authority": "ENGINEERING_QA",
        "paper_use": "Supports synthetic file usability only",
        "synthetic": True,
    },
    {
        "evidence_id": "L2-E08",
        "claim_family": "Raw collection-shaped logs",
        "artifact": "../01_collection_shaped_raw/raw/",
        "authority": "CANNOT_PROVE_REAL_HUMAN",
        "paper_use": "Never cite as observed participant data",
        "synthetic": True,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tree(source: Path, target: Path) -> str:
    if target.exists():
        raise FileExistsError(target)
    copied = False
    if os.uname().sysname == "Darwin":
        result = subprocess.run(
            ["cp", "-cR", str(source), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        copied = result.returncode == 0
    if not copied:
        shutil.copytree(source, target)
        return "copytree"
    return "apfs_clone"


def _prepare_output(output: Path, replace: bool) -> None:
    if output.exists():
        if not replace:
            raise FileExistsError(f"output exists; pass --replace: {output}")
        if output.resolve() != DEFAULT_OUTPUT.resolve():
            raise ValueError("replacement is restricted to the canonical RQ029 release root")
        manifest_path = output / "release_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("refusing replacement: output has no release manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema_version": SCHEMA_VERSION,
            "data_status": DATA_STATUS,
            "source_path": str(SOURCE_ROOT.relative_to(REPO_ROOT)),
            "declared_layers": 2,
            "third_layer_status": "NOT_SPECIFIED_BY_USER",
        }
        observed_identity = {
            key: manifest.get(key) for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise ValueError(
                "refusing replacement: release identity mismatch "
                f"{observed_identity!r}"
            )
        required_markers = (
            output / "README.md",
            output / "00_metadata",
            output / "01_collection_shaped_raw",
            output / "02_analysis_and_paper_support",
        )
        if not all(path.exists() for path in required_markers):
            raise ValueError("refusing replacement: release marker structure is incomplete")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _copy_table_set(names: tuple[str, ...], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(SOURCE_ROOT / "tables" / name, target / name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_core_results(target: Path) -> int:
    comparison_path = REPO_ROOT / (
        "reports/studies/RQ029_human_synthetic_microdata/"
        "RQ029_4_paper_claim_parity_audit_20260904/claim_metric_comparison.csv"
    )
    comparison = pd.read_csv(comparison_path).fillna("")
    core = comparison[comparison.metric.isin(CORE_METRICS)].copy()
    core.insert(
        len(core.columns),
        "paper_support_authority",
        "SYNTHETIC_REHEARSAL_NOT_EMPIRICAL_SUPPORT",
    )
    core.to_csv(target, index=False)
    return len(core)


def _write_readmes(output: Path) -> None:
    (output / "README.md").write_text(
        "# RQ029 layered synthetic data release\n\n"
        "**SYNTHETIC_NOT_OBSERVED.** This release implements only the two layers "
        "explicitly defined by the user. No third layer was inferred.\n\n"
        "- `01_collection_shaped_raw/`: collection-shaped synthetic logs and trajectory tables.\n"
        "- `02_analysis_and_paper_support/`: derived analysis, real aggregate authorities, "
        "and claim-evidence boundaries.\n\n"
        "The first layer is not an actual participant collection. In the second layer, only "
        "files under `01_authoritative_real_aggregate/` may directly support the accepted paper "
        "claim; synthetic analyses are rehearsal, QA, or boundary evidence.\n",
        encoding="utf-8",
    )
    layer1 = output / "01_collection_shaped_raw"
    (layer1 / "README.md").write_text(
        "# Layer 1: collection-shaped raw synthetic data\n\n"
        "**SYNTHETIC_NOT_OBSERVED.** These files mimic the AV collection structure for 20 "
        "synthetic drivers x 15 Shanghai scenarios. They are suitable for parsers, replay, "
        "trajectory QA, and analysis-pipeline rehearsal. They are not observed participant data.\n\n"
        "`raw/` contains the 20 driver/session payloads. `tables/trajectory_pairs.parquet` is "
        "the normalized ego/background trajectory table. No candidate bands, calibrated paper "
        "outcomes, or manuscript labels are stored in this layer.\n",
        encoding="utf-8",
    )
    layer2 = output / "02_analysis_and_paper_support"
    (layer2 / "README.md").write_text(
        "# Layer 2: analysis and paper-support package\n\n"
        "This layer intentionally separates evidence authority:\n\n"
        "1. `01_authoritative_real_aggregate/` contains the verified aggregate human record, "
        "frozen AV reference, and accepted RQ022 decision. These are the only direct paper-support "
        "authorities in this package.\n"
        "2. `02_synthetic_analysis_rehearsal/` contains derived synthetic tables and validation. "
        "They support method, code, figure, and interface rehearsal only.\n"
        "3. `03_evidence_and_audits/` contains continuity and claim-parity audits.\n"
        "4. `04_core_results/` provides a compact result index and claim-evidence map.\n\n"
        "Do not cite synthetic calibrated values as independent empirical confirmation. The current "
        "status is POINT_ESTIMATES_EXACT / INTERVALS_PARTIAL / RAW_EMERGENT_MISMATCH / "
        "FULL_DISTRIBUTION_NOT_IDENTIFIABLE.\n",
        encoding="utf-8",
    )
    metadata = output / "00_metadata"
    (metadata / "ACCESS_AND_USE.md").write_text(
        "# Access and use boundary\n\n"
        "- No public repository, DOI, accession, or licence is asserted by this local package.\n"
        "- Human participant driving records are not included; only the accepted aggregate record "
        "is copied as a reference.\n"
        "- Synthetic raw and derived files may be used for internal engineering and manuscript "
        "workflow rehearsal, not as observed-human evidence.\n"
        "- A formal external deposit requires author/institution confirmation of repository, "
        "licence, version, and access conditions.\n",
        encoding="utf-8",
    )
    (metadata / "DATA_DICTIONARY.md").write_text(
        "# Compact data dictionary\n\n"
        "| File | Grain | Main contents | Evidence role |\n"
        "|---|---|---|---|\n"
        "| Layer 1 `raw/drivers/` | session log record | AV-compatible synthetic replay payload | engineering only |\n"
        "| Layer 1 `trajectory_pairs.parquet` | driver x scenario x frame | ego/source/background motion, timestamps, raw kinematics | engineering QA |\n"
        "| Layer 2 `candidate_moments.parquet` | candidate moment | status, gates, bands, calibrated TTC and raw diagnostics | synthetic analysis rehearsal |\n"
        "| Layer 2 `counterpart_windows.parquet` | selected anchor | synthetic case, raw/calibrated counterpart outcomes | synthetic analysis rehearsal |\n"
        "| Layer 2 `per_unit_counts.csv` | driver x scenario run | gate and band counts | bootstrap rehearsal |\n"
        "| Layer 2 authoritative `human_arm_data.json` | aggregate | accepted real-human counts and signatures | direct paper support |\n"
        "| `core_results.csv` | paper metric | target, calibrated, raw, status, denominator | evidence routing |\n\n"
        "Units are encoded in column names where applicable: `_kmh`, `_mps2`, `_s`, `_ms`, and `_m`. "
        "Missing values in parquet are null/NaN; see the source manifest and audit reports for field-specific meaning.\n",
        encoding="utf-8",
    )
    (metadata / "DATA_AVAILABILITY_LOCAL_DRAFT.md").write_text(
        "# Data Availability (local packaging draft; not submission-ready)\n\n"
        "The verified aggregate results for the human reference arm are preserved in "
        "`02_analysis_and_paper_support/01_authoritative_real_aggregate/human_arm_data.json` "
        "together with the accepted RQ022 decision. Individual participant driving records are "
        "not included in this release. The collection-shaped logs and derived analysis tables in "
        "this package are synthetic (`SYNTHETIC_NOT_OBSERVED`) and are provided only for internal "
        "pipeline, figure, and analysis rehearsal; they are not empirical Source Data supporting "
        "the manuscript claims. No public repository identifier, licence, or controlled-access "
        "route is assigned by this local package.\n\n"
        "## Repository and citation actions\n\n"
        "- AUTHOR_INPUT_NEEDED: select the institutional/public/controlled repository route.\n"
        "- AUTHOR_INPUT_NEEDED: assign version, licence/rights, persistent identifier, and citation.\n"
        "- AUTHOR_INPUT_NEEDED: confirm the institutional access process for restricted human records.\n\n"
        "## 中文核对\n\n"
        "- 这个本地包只完成数据分层和证据权限标注，并不等于已公开存储。\n"
        "- 真实人类逐帧记录未包含；合成 raw/派生表不得写成论文实证源数据。\n"
        "- 正式投稿前还需确认仓库、永久标识符、许可/权利和受限访问流程。\n",
        encoding="utf-8",
    )


def package_release(output: Path = DEFAULT_OUTPUT, replace: bool = False) -> dict[str, Any]:
    _prepare_output(output, replace)
    metadata = output / "00_metadata"
    layer1 = output / "01_collection_shaped_raw"
    layer2 = output / "02_analysis_and_paper_support"
    metadata.mkdir()
    layer1.mkdir()
    layer2.mkdir()

    raw_copy_method = _clone_tree(SOURCE_ROOT / "raw", layer1 / "raw")
    shared_log_copy_method = _clone_tree(
        SOURCE_ROOT / "shared_source_logs", layer1 / "shared_source_logs"
    )
    _copy_table_set(LAYER_1_TABLES, layer1 / "tables")
    provenance = layer1 / "provenance"
    provenance.mkdir()
    shutil.copy2(SOURCE_ROOT / "manifest.json", provenance / "source_manifest.json")
    shutil.copy2(SOURCE_ROOT / "README.md", provenance / "source_README.md")

    rehearsal = layer2 / "02_synthetic_analysis_rehearsal"
    _copy_table_set(LAYER_2_TABLES, rehearsal / "tables")
    shutil.copytree(SOURCE_ROOT / "validation", rehearsal / "validation")
    contracts = rehearsal / "contracts"
    contracts.mkdir()
    shutil.copy2(
        REPO_ROOT / "reports/plans/RQ029_paper_record_targets_v2.json",
        contracts / "paper_record_targets_v2.json",
    )

    authoritative = layer2 / "01_authoritative_real_aggregate"
    authoritative.mkdir()
    reference_root = REPO_ROOT / ".codex-fleet/rq022-matched-scenario/work/T1_target_figure"
    shutil.copy2(reference_root / "human_arm_data.json", authoritative / "human_arm_data.json")
    shutil.copy2(reference_root / "av_reference_values.json", authoritative / "av_reference_values.json")
    shutil.copy2(
        REPO_ROOT / "reports/knowledge/RQ022_matched_scenario_human_arm/decision.md",
        authoritative / "RQ022_decision.md",
    )

    audits_target = layer2 / "03_evidence_and_audits"
    audits_source = REPO_ROOT / "reports/studies/RQ029_human_synthetic_microdata"
    audits_target.mkdir()
    for name in AUDIT_DIRS:
        shutil.copytree(audits_source / name, audits_target / name)

    core_target = layer2 / "04_core_results"
    core_target.mkdir()
    core_count = _build_core_results(core_target / "core_results.csv")
    _write_csv(core_target / "claim_evidence_map.csv", list(EVIDENCE_ROWS))
    shutil.copy2(
        audits_source
        / "RQ029_4_paper_claim_parity_audit_20260904/claim_metric_comparison.csv",
        core_target / "claim_metric_comparison.csv",
    )
    shutil.copy2(
        audits_source / "RQ029_4_paper_claim_parity_audit_20260904/summary.json",
        core_target / "claim_parity_summary.json",
    )

    _write_readmes(output)

    excluded = {
        output / "release_manifest.json",
        metadata / "file_inventory.csv",
        output / "MANIFEST.sha256",
        metadata / "validation_summary.json",
        metadata / "validation_checks.csv",
        metadata / "VALIDATION_REPORT.md",
    }
    files = sorted(path for path in output.rglob("*") if path.is_file() and path not in excluded)
    inventory_rows: list[dict[str, Any]] = []
    manifest_lines: list[str] = []
    logical_bytes = 0
    for path in files:
        relative = path.relative_to(output).as_posix()
        size = path.stat().st_size
        digest = _sha256(path)
        logical_bytes += size
        inventory_rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
                "layer": relative.split("/", 1)[0],
            }
        )
        manifest_lines.append(f"{digest}  {relative}")
    _write_csv(metadata / "file_inventory.csv", inventory_rows)
    (output / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_status": DATA_STATUS,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_path": str(SOURCE_ROOT.relative_to(REPO_ROOT)),
        "source_schema_version": json.loads(
            (SOURCE_ROOT / "manifest.json").read_text(encoding="utf-8")
        )["schema_version"],
        "declared_layers": 2,
        "third_layer_status": "NOT_SPECIFIED_BY_USER",
        "layers": {
            "01_collection_shaped_raw": {
                "role": "collection-shaped synthetic source layer",
                "paper_evidence": False,
            },
            "02_analysis_and_paper_support": {
                "role": "derived analysis plus separated authoritative aggregate references",
                "paper_evidence": "authoritative subdirectory only",
            },
        },
        "n_drivers": 20,
        "n_scenarios": 15,
        "n_runs": 300,
        "n_frames": 81880,
        "copy_methods": {
            "raw": raw_copy_method,
            "shared_source_logs": shared_log_copy_method,
            "other_files": "copy2_or_copytree",
        },
        "n_raw_files": sum(1 for path in (layer1 / "raw").rglob("*") if path.is_file()),
        "n_inventory_files": len(inventory_rows),
        "inventory_logical_bytes": logical_bytes,
        "inventory_sha256": _sha256(metadata / "file_inventory.csv"),
        "core_result_rows": core_count,
        "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
        "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
        "full_distribution_match_claimed": False,
        "manifest_scope": "MANIFEST.sha256 excludes itself, release_manifest.json, and validation outputs",
        "boundary": (
            "Collection-shaped raw files are synthetic. Only copied REAL_VERIFIED aggregate "
            "references and the accepted RQ022 decision are direct paper-support authorities."
        ),
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_release(
    output: Path = DEFAULT_OUTPUT,
    *,
    deep_hash: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
    check("schema", manifest.get("schema_version") == SCHEMA_VERSION, manifest.get("schema_version"), SCHEMA_VERSION)
    check("synthetic_status", manifest.get("data_status") == DATA_STATUS, manifest.get("data_status"), DATA_STATUS)
    check("two_declared_layers", manifest.get("declared_layers") == 2, manifest.get("declared_layers"), 2)
    check(
        "third_layer_not_invented",
        manifest.get("third_layer_status") == "NOT_SPECIFIED_BY_USER",
        manifest.get("third_layer_status"),
        "NOT_SPECIFIED_BY_USER",
    )
    check(
        "full_match_not_claimed",
        manifest.get("raw_emergent_match_status") == "NOT_MATCHED_AND_NOT_CLAIMED"
        and manifest.get("full_distribution_match_claimed") is False,
        [manifest.get("raw_emergent_match_status"), manifest.get("full_distribution_match_claimed")],
        ["NOT_MATCHED_AND_NOT_CLAIMED", False],
    )
    copy_methods = manifest.get("copy_methods", {})
    check(
        "copy_methods_recorded",
        copy_methods.get("raw") in {"apfs_clone", "copytree"}
        and copy_methods.get("shared_source_logs") in {"apfs_clone", "copytree"},
        copy_methods,
        "explicit raw/shared copy methods",
    )

    layer1 = output / "01_collection_shaped_raw"
    layer2 = output / "02_analysis_and_paper_support"
    release_raw = sorted(path.relative_to(layer1 / "raw") for path in (layer1 / "raw").rglob("*") if path.is_file())
    source_raw = sorted(path.relative_to(SOURCE_ROOT / "raw") for path in (SOURCE_ROOT / "raw").rglob("*") if path.is_file())
    check("raw_file_set", release_raw == source_raw, len(release_raw), len(source_raw))
    if deep_hash:
        mismatches = [
            str(relative)
            for relative in source_raw
            if not filecmp.cmp(
                SOURCE_ROOT / "raw" / relative,
                layer1 / "raw" / relative,
                shallow=False,
            )
        ]
        check("raw_bytes", not mismatches, mismatches, [])
    else:
        size_match = all(
            (SOURCE_ROOT / "raw" / relative).stat().st_size
            == (layer1 / "raw" / relative).stat().st_size
            for relative in source_raw
        )
        check("raw_sizes", size_match, size_match, True)
    hardlinked = [
        str(relative)
        for relative in source_raw
        if (SOURCE_ROOT / "raw" / relative).stat().st_ino
        == (layer1 / "raw" / relative).stat().st_ino
    ]
    check("raw_not_hardlinked", not hardlinked, hardlinked, [])

    for name in LAYER_1_TABLES:
        source = SOURCE_ROOT / "tables" / name
        target = layer1 / "tables" / name
        check(f"layer1_{name}", target.is_file() and _sha256(source) == _sha256(target), target.is_file(), True)
    for name in LAYER_2_TABLES:
        source = SOURCE_ROOT / "tables" / name
        target = layer2 / "02_synthetic_analysis_rehearsal/tables" / name
        check(f"layer2_{name}", target.is_file() and _sha256(source) == _sha256(target), target.is_file(), True)

    prohibited_layer1 = [
        layer1 / "tables/candidate_moments.parquet",
        layer1 / "tables/counterpart_windows.parquet",
        layer1 / "tables/achieved_summary.json",
    ]
    check(
        "no_analysis_tables_in_layer1",
        not any(path.exists() for path in prohibited_layer1),
        [path.name for path in prohibited_layer1 if path.exists()],
        [],
    )
    authoritative = layer2 / "01_authoritative_real_aggregate"
    human = json.loads((authoritative / "human_arm_data.json").read_text(encoding="utf-8"))
    check("real_aggregate_status", human.get("data_status") == "REAL_VERIFIED", human.get("data_status"), "REAL_VERIFIED")
    evidence = pd.read_csv(layer2 / "04_core_results/claim_evidence_map.csv")
    synthetic_mask = evidence.synthetic.astype(str).str.lower().eq("true")
    invalid_synthetic = evidence[
        synthetic_mask & evidence.authority.eq("DIRECT_SUPPORT")
    ]
    check("synthetic_not_direct_support", invalid_synthetic.empty, len(invalid_synthetic), 0)
    authoritative_files = {
        path.name for path in authoritative.iterdir() if path.is_file()
    }
    classified_authoritative = {
        Path(value).name
        for value in evidence.artifact.astype(str)
        if value.startswith("01_authoritative_real_aggregate/")
    }
    check(
        "authoritative_files_classified",
        authoritative_files == classified_authoritative,
        sorted(classified_authoritative),
        sorted(authoritative_files),
    )
    check(
        "raw_readme_boundary",
        "SYNTHETIC_NOT_OBSERVED" in (layer1 / "README.md").read_text(encoding="utf-8"),
        True,
        True,
    )
    runs = pd.read_csv(layer1 / "tables/runs.csv")
    trajectory_rows = len(pd.read_parquet(layer1 / "tables/trajectory_pairs.parquet", columns=["run_id"]))
    check("run_count", runs.run_id.nunique() == 300, int(runs.run_id.nunique()), 300)
    check("trajectory_rows", trajectory_rows == 81880, trajectory_rows, 81880)

    inventory = pd.read_csv(output / "00_metadata/file_inventory.csv")
    duplicate_paths = int(inventory.relative_path.duplicated().sum())
    check("inventory_unique", duplicate_paths == 0, duplicate_paths, 0)
    missing_inventory = [
        relative
        for relative in inventory.relative_path
        if not (output / relative).is_file()
    ]
    check("inventory_files_present", not missing_inventory, missing_inventory, [])
    if deep_hash:
        hash_mismatches = [
            row.relative_path
            for row in inventory.itertuples(index=False)
            if _sha256(output / row.relative_path) != row.sha256
        ]
        check("inventory_hashes", not hash_mismatches, hash_mismatches, [])
    manifest_entries: dict[str, str] = {}
    malformed_manifest_lines: list[str] = []
    for line in (output / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            malformed_manifest_lines.append(line)
            continue
        digest, relative = line.split("  ", 1)
        manifest_entries[relative] = digest
    inventory_entries = dict(zip(inventory.relative_path, inventory.sha256))
    check(
        "manifest_paths_match_inventory",
        not malformed_manifest_lines
        and set(manifest_entries) == set(inventory_entries),
        {
            "malformed": malformed_manifest_lines,
            "manifest_paths": len(manifest_entries),
        },
        {"malformed": [], "manifest_paths": len(inventory_entries)},
    )
    manifest_hash_mismatches = sorted(
        relative
        for relative in set(manifest_entries) & set(inventory_entries)
        if manifest_entries[relative] != inventory_entries[relative]
    )
    check(
        "manifest_hashes_match_inventory",
        not manifest_hash_mismatches,
        manifest_hash_mismatches,
        [],
    )

    failed = [row for row in checks if row["status"] == "FAIL"]
    result = {
        "validation_status": "PASS" if not failed else "FAIL",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": [row["check"] for row in failed],
        "deep_hash": deep_hash,
        "n_release_files": sum(1 for path in output.rglob("*") if path.is_file()),
        "n_raw_files": len(release_raw),
        "n_runs": int(runs.run_id.nunique()),
        "n_frames": trajectory_rows,
        "boundary": manifest["boundary"],
    }
    if write_report:
        metadata = output / "00_metadata"
        (metadata / "validation_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pd.DataFrame(checks).to_csv(metadata / "validation_checks.csv", index=False)
        (metadata / "VALIDATION_REPORT.md").write_text(
            "# RQ029 layered release validation\n\n"
            f"- Status: **{result['validation_status']}**\n"
            f"- Checks: {result['checks_passed']}/{result['checks_total']} passed\n"
            f"- Raw files: {result['n_raw_files']}\n"
            f"- Runs / frames: {result['n_runs']} / {result['n_frames']}\n"
            f"- Deep byte/hash verification: {deep_hash}\n\n"
            "## Boundary\n\n"
            f"{result['boundary']}\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-deep-hash", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        package_release(args.output_dir, replace=args.replace)
    result = validate_release(
        args.output_dir,
        deep_hash=not args.skip_deep_hash,
        write_report=True,
    )
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["validation_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
