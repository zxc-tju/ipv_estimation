#!/usr/bin/env python3
"""Build the Phase 10 reader-facing RQ003 report package.

This script creates Nature-skill-aligned static figures and an offline HTML
reader without modifying source result tables or analysis code.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import unquote
from urllib.request import urlopen


SCRIPT_PATH = Path(__file__).resolve()
MPLCONFIGDIR = SCRIPT_PATH.parent / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Nature skill mandatory editable text rules.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 450,
    }
)


RUN_ID = "RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0"
WORKER_ID = "RQ003_phase10_report_001"
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"
GIT_HEAD = "c23074a091f9ff57b1034144571f68f771db9d8d"

RUN_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[6]
META = RUN_ROOT / "02_process" / "00_meta"
DIRT = RUN_ROOT / "01_results" / "tables"
FIGS = RUN_ROOT / "01_results" / "figures"
ENTRY = RUN_ROOT / "00_entry"
REP90 = RUN_ROOT / "90_report"
RB = RUN_ROOT / "02_process" / "19_report_build"
TIER = RUN_ROOT / "02_process" / "18_tier_review"
INTERP = RUN_ROOT / "02_process" / "16_red_team_fixes" / "interp_fix"
NATURE_SKILL_DIR = Path("/Users/xiaocong/.claude/skills/nature-figure")

SETTLED_CONCLUSION = (
    "No robust incremental predictive utility relative to the prespecified "
    "kinematic+safety baseline was demonstrated (power-limited, top-five "
    "cohort, N=53; apparent favorable direction not IPV-specific)."
)

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_base": "#E4CCD8",
    "ours_large": "#F0C0CC",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
    "teal": "#42949E",
    "gold": "#C69C32",
    "ink": "#272727",
}

COMMANDS_RUN = [
    "sed/read Nature skill router, manifest, core contract, stance, Python backend fragment, and relevant references",
    "python identity/schema/status inventory over corrected RQ003 package",
    "python reports/studies/RQ003_nsfc_external_evidence/"
    f"{RUN_ID}/02_process/19_report_build/build_reader_package.py",
]

SOURCE_READS: list[Path] = []
WRITES: list[Path] = []
FIGURE_ROWS: list[dict[str, str]] = []
ARTIFACT_ROWS: list[dict[str, str]] = []


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


BUILD_TIME_UTC = now_utc()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RUN_ROOT.resolve()))
    except ValueError:
        try:
            return str(path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(path)


def entry_link(path: Path) -> str:
    return Path(os.path.relpath(path, ENTRY)).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    SOURCE_READS.append(path)
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict:
    SOURCE_READS.append(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    SOURCE_READS.append(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    WRITES.append(path)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    WRITES.append(path)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    WRITES.append(path)


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    WRITES.append(path)


def fnum(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_label(text: str, width: int = 26) -> str:
    words = str(text).replace("_", " ").split()
    lines: list[str] = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= width:
            cur = f"{cur} {word}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["ink"],
    )


def add_zero_line(ax, axis: str = "x") -> None:
    if axis == "x":
        ax.axvline(0, color=PALETTE["neutral_mid"], lw=0.8, ls="--", zorder=0)
    else:
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.8, ls="--", zorder=0)


def set_clean(ax) -> None:
    ax.grid(axis="y", color="#EBEBEB", lw=0.5)
    ax.tick_params(axis="both", length=2, width=0.6)


def save_figure(
    fig,
    fig_id: str,
    title: str,
    claim: str,
    source_rows: list[dict[str, object]],
    source_fields: list[str],
    primary_sources: list[str],
    caption: str,
    panel_map: dict[str, str],
    reviewer_risk: str,
) -> None:
    base = FIGS / fig_id
    source_csv = base.with_name(f"{fig_id}_source.csv")
    metadata_json = base.with_name(f"{fig_id}_metadata.json")
    write_csv(source_csv, source_rows, source_fields)
    fig.tight_layout(pad=1.1)
    for ext in ("svg", "pdf", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=450)
        WRITES.append(out)
    plt.close(fig)
    metadata = {
        "figure_id": fig_id,
        "title": title,
        "created_utc": BUILD_TIME_UTC,
        "worker_id": WORKER_ID,
        "run_id": RUN_ID,
        "nature_skill": {
            "name": "nature-figure",
            "version": "2.0.0",
            "backend": "python",
            "contract": "quantitative grid; SVG/PDF/PNG; editable SVG/PDF text; source CSV traceability",
        },
        "core_conclusion": claim,
        "figure_archetype": "quantitative grid",
        "target_output": "offline reader report and publication-editable static figure exports",
        "final_size_inches": [7.2, 4.8],
        "panel_map": panel_map,
        "evidence_hierarchy": {
            "hero_evidence": next(iter(panel_map.values())) if panel_map else "",
            "controls_or_boundary": reviewer_risk,
        },
        "statistics_notes": caption,
        "source_data": [str(src) for src in primary_sources],
        "image_integrity_notes": "No raster scientific image manipulation; all charts are generated from corrected tabular sources.",
        "reviewer_risk": reviewer_risk,
        "exports": {
            "svg": rel(base.with_suffix(".svg")),
            "pdf": rel(base.with_suffix(".pdf")),
            "png": rel(base.with_suffix(".png")),
            "source_csv": rel(source_csv),
        },
    }
    write_json(metadata_json, metadata)
    FIGURE_ROWS.append(
        {
            "figure_id": fig_id,
            "title": title,
            "svg": rel(base.with_suffix(".svg")),
            "pdf": rel(base.with_suffix(".pdf")),
            "png": rel(base.with_suffix(".png")),
            "source_csv": rel(source_csv),
            "metadata_json": rel(metadata_json),
            "primary_source_tables": ";".join(primary_sources),
            "claim": claim,
            "figure_role": "reader-facing evidence figure",
            "caption": caption,
            "skill_name": "nature-figure",
            "skill_version": "2.0.0",
            "backend": "python",
            "created_utc": BUILD_TIME_UTC,
        }
    )


def verify_identity() -> dict[str, object]:
    checks: dict[str, object] = {
        "run_root_exists": RUN_ROOT.exists(),
        "figures_dir_exists": FIGS.exists(),
        "entry_dir_exists": ENTRY.exists(),
        "report_dir_exists": REP90.exists(),
        "report_build_dir_exists": RB.exists(),
    }
    run_manifest = read_json(META / "run_manifest.json")
    tier_decision = read_json(TIER / "tier_decision.json")
    plan_sha = read_text(META / "plan_sha256.txt").strip()
    checks["run_manifest_RUN_ID"] = run_manifest.get("RUN_ID")
    checks["run_manifest_RUN_ID_matches"] = run_manifest.get("RUN_ID") == RUN_ID
    checks["plan_sha256_matches"] = plan_sha == PLAN_SHA256
    checks["tier_decision_tier"] = tier_decision.get("tier")
    checks["tier_decision_tier_is_B"] = tier_decision.get("tier") == "B"
    checks["matplotlib_version"] = mpl.__version__
    checks["matplotlib_imported"] = True
    checks["python_executable"] = sys.executable
    required = [
        "run_root_exists",
        "figures_dir_exists",
        "entry_dir_exists",
        "report_dir_exists",
        "report_build_dir_exists",
        "run_manifest_RUN_ID_matches",
        "plan_sha256_matches",
        "tier_decision_tier_is_B",
        "matplotlib_imported",
    ]
    checks["status"] = "PASS" if all(checks.get(k) is True for k in required) else "BLOCKED"
    checks["required"] = required
    return checks


def load_sources() -> dict[str, object]:
    return {
        "run_manifest": read_json(META / "run_manifest.json"),
        "tier_decision": read_json(TIER / "tier_decision.json"),
        "paper_handoff": read_text(TIER / "paper_handoff.md"),
        "interpretation_correction": read_text(INTERP / "interpretation_correction.md"),
        "claim_boundary": read_csv(TIER / "claim_boundary_matrix.csv"),
        "confirmatory": read_csv(DIRT / "confirmatory_results.csv"),
        "confirmatory_interp": read_csv(DIRT / "confirmatory_results_interpretation.csv"),
        "confirmatory_before": read_csv(DIRT / "confirmatory_results__before_scenario_fix.csv"),
        "cv_predictions": read_csv(DIRT / "cv_predictions.csv"),
        "negative_controls": read_csv(DIRT / "negative_controls.csv"),
        "state_dependence": read_csv(DIRT / "state_dependence_results.csv"),
        "cell_directional": read_csv(DIRT / "cell_level_directional_ipv.csv"),
        "support": read_csv(DIRT / "support_coverage.csv"),
        "crosswalk": read_csv(DIRT / "scenario_crosswalk_corrected.csv"),
        "coverage_matrix": read_csv(DIRT / "coverage_matrix.csv"),
        "implementation_v2": read_csv(DIRT / "implementation_comparison_v2.csv"),
        "missingness": read_csv(DIRT / "missingness_audit.csv"),
        "gate_minus1": read_json(RUN_ROOT / "02_process" / "02_gate_minus1" / "gate_minus1_status.json"),
        "gate_minus1_review": read_json(
            RUN_ROOT / "02_process" / "03_gate_minus1_review" / "gate_minus1_review_status.json"
        ),
        "gate0": read_json(RUN_ROOT / "02_process" / "04_gate0_measurement" / "gate0_status.json"),
        "gate0_review": read_json(RUN_ROOT / "02_process" / "05_gate0_review" / "gate0_review_status.json"),
        "g0r_cond": read_json(RUN_ROOT / "02_process" / "08_directional_ipv" / "g0r_cond_001_status.json"),
        "freeze_review": read_json(RUN_ROOT / "02_process" / "07_freeze_review" / "freeze_review_status.json"),
        "annotation_status": read_json(RUN_ROOT / "02_process" / "12_blind_annotation" / "annotation_status.json"),
        "red_team1": read_json(RUN_ROOT / "02_process" / "15_red_team" / "red_team_status.json"),
        "scenario_fix": read_json(RUN_ROOT / "02_process" / "16_red_team_fixes" / "fix_status.json"),
        "red_team2": read_json(
            RUN_ROOT / "02_process" / "16_red_team_fixes" / "red_team2" / "red_team2_status.json"
        ),
        "rt2_resolution": read_json(INTERP / "rt2_blockers_resolution.json"),
        "red_team3": read_json(
            RUN_ROOT / "02_process" / "16_red_team_fixes" / "red_team3" / "red_team3_status.json"
        ),
        "replication1": read_json(
            RUN_ROOT / "02_process" / "17_independent_replication" / "replication_status.json"
        ),
        "replication2": read_json(
            RUN_ROOT
            / "02_process"
            / "17_independent_replication"
            / "replication2"
            / "replication2_status.json"
        ),
        "stats_rereview2": read_json(
            RUN_ROOT
            / "02_process"
            / "16_red_team_fixes"
            / "stats_rereview2"
            / "stats_rereview2_status.json"
        ),
        "npc_boundary": read_text(RUN_ROOT / "02_process" / "14_npc_analysis" / "npc_feasibility_and_boundary.md"),
    }


def primary_row(rows: list[dict[str, str]], analysis_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("analysis_id") == analysis_id:
            return row
    raise KeyError(analysis_id)


def fig01_provenance(data: dict[str, object]) -> None:
    gate = data["gate_minus1"]
    crosswalk = data["crosswalk"]
    changed = sum(1 for r in crosswalk if str(r.get("changed_flag")) == "True")
    unchanged = len(crosswalk) - changed
    rows = [
        {
            "panel": "a",
            "metric": "top_five_clean_mapping",
            "value": gate["plan_top5_unique_usable_cells"],
            "denominator": gate["plan_top5_cells"],
            "status": "covered_for_plan",
            "source_table": "gate_minus1_status.json",
            "note": "approved top-five cohort mapping",
        },
        {
            "panel": "a",
            "metric": "full_universe_clean_mapping",
            "value": gate["full_universe_clean_usable_cells"],
            "denominator": gate["full_universe_cells"],
            "status": "not_reader_ready_for_full_universe_claim",
            "source_table": "gate_minus1_status.json",
            "note": "full universe remains outside analysis-ready scope",
        },
        {
            "panel": "b",
            "metric": "scenario_labels_relabelled",
            "value": changed,
            "denominator": len(crosswalk),
            "status": "scenario_fix_required",
            "source_table": "scenario_crosswalk_corrected.csv",
            "note": "official-label correction",
        },
        {
            "panel": "b",
            "metric": "scenario_labels_unchanged",
            "value": unchanged,
            "denominator": len(crosswalk),
            "status": "unchanged",
            "source_table": "scenario_crosswalk_corrected.csv",
            "note": "official-label correction",
        },
        {
            "panel": "c",
            "metric": "top_five_missing_or_not_clean",
            "value": 0,
            "denominator": gate["plan_top5_cells"],
            "status": "zero_missing_for_plan",
            "source_table": "missingness_audit.csv",
            "note": "diagnostic only",
        },
        {
            "panel": "c",
            "metric": "full_universe_missing_or_not_clean",
            "value": gate["full_universe_cells"] - gate["full_universe_clean_usable_cells"],
            "denominator": gate["full_universe_cells"],
            "status": "selection_boundary",
            "source_table": "gate_minus1_status.json",
            "note": "not a full-universe validity basis",
        },
    ]
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], wspace=0.42, hspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["Top-five\napproved", "Full scored\nuniverse"]
    values = [100 * fnum(rows[0]["value"]) / fnum(rows[0]["denominator"]), 100 * fnum(rows[1]["value"]) / fnum(rows[1]["denominator"])]
    bars = ax.bar(labels, values, color=[PALETTE["baseline_dark"], PALETTE["neutral_light"]], edgecolor="black", linewidth=0.8)
    bars[1].set_hatch("//")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Clean mapped cells (%)")
    ax.set_title("Coverage scope")
    for bar, row in zip(bars, rows[:2]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{row['value']}/{row['denominator']}", ha="center", fontsize=7)
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    vals = [changed, unchanged]
    bars = ax.bar(["Relabelled", "Unchanged"], vals, color=[PALETTE["delta_down"], PALETTE["baseline_soft"]], edgecolor="black", linewidth=0.8)
    bars[0].set_hatch("xx")
    ax.set_ylabel("Cells")
    ax.set_title("Scenario-fix footprint")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{val}/150", ha="center", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.22)
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 0])
    vals = [0, gate["full_universe_cells"] - gate["full_universe_clean_usable_cells"]]
    bars = ax.bar(["Top-five\ncohort", "Full\nuniverse"], vals, color=[PALETTE["baseline_dark"], PALETTE["neutral_mid"]], edgecolor="black", linewidth=0.8)
    bars[1].set_hatch("..")
    ax.set_ylabel("Missing or not-clean cells")
    ax.set_title("Selection boundary")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val), ha="center", fontsize=7)
    add_panel_label(ax, "c")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    add_panel_label(ax, "d")
    text = (
        "Reader scope\n"
        "Top-five cohort: 150/150 mapped\n"
        "Primary model sample: N=53 cells\n"
        "Full 20-team universe: not analysis-ready\n"
        "Outcome: official/generated coordination score"
    )
    ax.text(0.0, 0.92, text, va="top", ha="left", fontsize=8, linespacing=1.45)

    save_figure(
        fig,
        "fig01_provenance_coverage",
        "Provenance and coverage boundary",
        "The reader-facing package is scoped to the approved top-five cohort; full-universe inference is not supported.",
        rows,
        ["panel", "metric", "value", "denominator", "status", "source_table", "note"],
        ["coverage_matrix.csv", "scenario_crosswalk_corrected.csv", "missingness_audit.csv", "gate_minus1_status.json"],
        "Bars show cell counts and denominators; hatches mark boundary or correction categories.",
        {
            "a": "Top-five mapping completeness vs full-universe clean mapping.",
            "b": "Scenario labels affected by the official-label correction.",
            "c": "Top-five zero missingness vs full-universe not-clean boundary.",
            "d": "Scope summary for reader interpretation.",
        },
        "The full universe remains a selection boundary and is not a validity claim basis.",
    )


def fig02_missingness(data: dict[str, object]) -> None:
    missing = data["missingness"]
    selected = [
        r
        for r in missing
        if r["scope"] == "full_scored_universe_clean_mapping"
        and r["stratum_type"] in {"overall", "area", "scenario_family", "mapping_status"}
    ]
    source_rows: list[dict[str, object]] = []
    for r in selected:
        source_rows.append(
            {
                "panel": "all",
                "scope": r["scope"],
                "stratum_type": r["stratum_type"],
                "stratum_value": r["stratum_value"],
                "observed_cells": r["observed_cells"],
                "missing_or_not_clean_cells": r["missing_or_not_clean_cells"],
                "observed_coordination_mean": r["observed_coordination_mean"],
                "missing_coordination_mean": r["missing_coordination_mean"],
                "coordination_diff_missing_minus_observed": r["coordination_diff_missing_minus_observed"],
                "diagnostic_only_not_criterion_validity": r["diagnostic_only_not_criterion_validity"],
            }
        )
    top5 = [
        r
        for r in missing
        if r["scope"] == "plan_top5_cohort" and r["stratum_type"] in {"overall", "area", "scenario_family"}
    ]
    for r in top5:
        source_rows.append(
            {
                "panel": "c",
                "scope": r["scope"],
                "stratum_type": r["stratum_type"],
                "stratum_value": r["stratum_value"],
                "observed_cells": r["observed_cells"],
                "missing_or_not_clean_cells": r["missing_or_not_clean_cells"],
                "observed_coordination_mean": r["observed_coordination_mean"],
                "missing_coordination_mean": r["missing_coordination_mean"],
                "coordination_diff_missing_minus_observed": r["coordination_diff_missing_minus_observed"],
                "diagnostic_only_not_criterion_validity": r["diagnostic_only_not_criterion_validity"],
            }
        )

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.56)
    ax = fig.add_subplot(gs[0, 0])
    area_family = [r for r in selected if r["stratum_type"] in {"overall", "area", "scenario_family"}]
    labels = [r["stratum_value"].replace("all_full_cells", "all") for r in area_family]
    observed = [inum(r["observed_cells"]) for r in area_family]
    missing_vals = [inum(r["missing_or_not_clean_cells"]) for r in area_family]
    x = np.arange(len(labels))
    ax.bar(x, observed, color=PALETTE["baseline_mid"], edgecolor="black", linewidth=0.6, label="clean mapped")
    ax.bar(x, missing_vals, bottom=observed, color=PALETTE["neutral_light"], edgecolor="black", linewidth=0.6, hatch="//", label="missing/not clean")
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(v, 12) for v in labels], rotation=0)
    ax.set_ylabel("Cells")
    ax.set_title("Full-universe mapping strata")
    ax.legend(fontsize=6, loc="upper right")
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    diff_rows = [r for r in area_family if r["coordination_diff_missing_minus_observed"] != ""]
    y = np.arange(len(diff_rows))
    diffs = [fnum(r["coordination_diff_missing_minus_observed"]) for r in diff_rows]
    colors = [PALETTE["delta_down"] if v < 0 else PALETTE["baseline_mid"] for v in diffs]
    ax.barh(y, diffs, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(r["stratum_value"].replace("all_full_cells", "all"), 18) for r in diff_rows])
    ax.set_xlabel("Mean score difference\nmissing/not-clean minus observed")
    ax.set_title("Diagnostic selection-bias signal")
    add_zero_line(ax)
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 0])
    labels = [r["stratum_value"].replace("all_plan_cells", "all") for r in top5]
    vals = [inum(r["missing_or_not_clean_cells"]) for r in top5]
    ax.bar(np.arange(len(labels)), vals, color=PALETTE["baseline_dark"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([clean_label(v, 12) for v in labels])
    ax.set_ylim(0, 3)
    ax.set_ylabel("Missing/not-clean cells")
    ax.set_title("Approved top-five cohort")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=7)
    add_panel_label(ax, "c")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    mapping = [r for r in selected if r["stratum_type"] == "mapping_status"]
    labels = [r["stratum_value"] for r in mapping]
    vals = [inum(r["missing_or_not_clean_cells"]) for r in mapping]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=PALETTE["neutral_mid"], edgecolor="black", linewidth=0.6, hatch="..")
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(v, 24) for v in labels])
    ax.set_xlabel("Missing/not-clean cells")
    ax.set_title("Mapping-status boundary")
    add_panel_label(ax, "d")
    set_clean(ax)

    save_figure(
        fig,
        "fig02_missingness_selection_bias",
        "Missingness and selection-bias diagnostic",
        "The approved top-five cohort has zero missing cells, while full-universe not-clean cells show a diagnostic selection boundary.",
        source_rows,
        [
            "panel",
            "scope",
            "stratum_type",
            "stratum_value",
            "observed_cells",
            "missing_or_not_clean_cells",
            "observed_coordination_mean",
            "missing_coordination_mean",
            "coordination_diff_missing_minus_observed",
            "diagnostic_only_not_criterion_validity",
        ],
        ["missingness_audit.csv", "coverage_matrix.csv"],
        "Panel b shows descriptive mean-score differences only; no IPV-outcome association is computed.",
        {
            "a": "Observed and not-clean full-universe cells by stratum.",
            "b": "Diagnostic score difference for not-clean cells where both means exist.",
            "c": "Top-five missingness remains zero.",
            "d": "Mapping-status classes contributing to the boundary.",
        },
        "Selection diagnostics cannot be promoted into criterion-validity evidence.",
    )


def fig03_support(data: dict[str, object]) -> None:
    support = data["support"]
    rows: list[dict[str, object]] = []
    rates = [fnum(r["coverage_rate_ego"]) for r in support if not math.isnan(fnum(r["coverage_rate_ego"]))]
    bins = [("0-0.10", 0.0, 0.10), ("0.10-0.30", 0.10, 0.30), (">0.30", 0.30, 1.000001)]
    for label, lo, hi in bins:
        rows.append(
            {
                "panel": "a",
                "metric": "coverage_rate_bin",
                "category": label,
                "value": sum(1 for v in rates if lo <= v < hi),
                "denominator": len(rates),
                "unit": "cells",
                "source_table": "support_coverage.csv",
            }
        )
    frame_metrics = [
        "estimated_conflict_frames",
        "high_support_frames_ego",
        "abstention_frames_ego",
        "ood_frames_ego",
        "optimizer_error_frames",
    ]
    for metric in frame_metrics:
        rows.append(
            {
                "panel": "b",
                "metric": metric,
                "category": "all_cells",
                "value": sum(inum(r.get(metric)) for r in support),
                "denominator": sum(inum(r.get("estimated_conflict_frames")) for r in support),
                "unit": "frames",
                "source_table": "support_coverage.csv",
            }
        )
    for r in support:
        rows.append(
            {
                "panel": "c",
                "metric": "cell_coverage_rate",
                "category": r["family"],
                "value": r["coverage_rate_ego"],
                "denominator": "1",
                "unit": "rate",
                "source_table": "support_coverage.csv",
                "cell_id": r["cell_id"],
                "team": r["team"],
                "scenario": r["scenario"],
            }
        )

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(rates, bins=np.linspace(0, 1, 11), color=PALETTE["baseline_mid"], edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Coverage rate per cell")
    ax.set_ylabel("Cells")
    ax.set_title("Support coverage distribution")
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    bin_rows = [r for r in rows if r["metric"] == "coverage_rate_bin"]
    labels = [r["category"] for r in bin_rows]
    vals = [inum(r["value"]) for r in bin_rows]
    bars = ax.bar(labels, vals, color=[PALETTE["neutral_mid"], PALETTE["baseline_soft"], PALETTE["baseline_dark"]], edgecolor="black", linewidth=0.6)
    bars[0].set_hatch("..")
    ax.set_ylabel("Cells")
    ax.set_title("Abstention-relevant bins")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, str(val), ha="center", fontsize=7)
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 0])
    frame_rows = [r for r in rows if r["panel"] == "b"]
    labels = [r["metric"].replace("_frames_ego", "").replace("_frames", "").replace("estimated_conflict", "estimated conflict") for r in frame_rows[1:]]
    vals = [inum(r["value"]) for r in frame_rows[1:]]
    ax.bar(np.arange(len(vals)), vals, color=[PALETTE["baseline_dark"], PALETTE["neutral_light"], PALETTE["delta_down"], PALETTE["gold"]], edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([clean_label(v, 12) for v in labels])
    ax.set_ylabel("Frames")
    ax.set_title("Frame-level support and abstention")
    add_panel_label(ax, "c")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    families = sorted({r["family"] for r in support})
    data_by_family = [[fnum(r["coverage_rate_ego"]) for r in support if r["family"] == fam] for fam in families]
    ax.boxplot(
        data_by_family,
        tick_labels=families,
        patch_artist=True,
        medianprops={"color": "black"},
        boxprops={"facecolor": PALETTE["baseline_soft"], "edgecolor": "black"},
    )
    ax.set_ylabel("Coverage rate")
    ax.set_title("Coverage by scenario family")
    add_panel_label(ax, "d")
    set_clean(ax)

    save_figure(
        fig,
        "fig03_support_ood_abstention",
        "Support, OOD and abstention coverage",
        "Support coverage is limited and heterogeneous; the conformal element is an InterHub-calibrated support boundary, not an NSFC coverage guarantee.",
        rows,
        ["panel", "metric", "category", "value", "denominator", "unit", "source_table", "cell_id", "team", "scenario"],
        ["support_coverage.csv", "state_dependence_results.csv"],
        "Rates are cell-level support fractions; frame counts aggregate corrected support-coverage rows.",
        {
            "a": "Distribution of support coverage rates.",
            "b": "Cell counts in abstention-relevant coverage bins.",
            "c": "Frame-level support, abstention, OOD and optimizer-error counts.",
            "d": "Coverage distributions by scenario family.",
        },
        "Do not present NSFC support boundaries as nominal conformal coverage.",
    )


def fig04_directional_ipv(data: dict[str, object]) -> None:
    cells = data["cell_directional"]
    rows = [
        {
            "cell_id": r["cell_id"],
            "team": r["team"],
            "scenario": r["scenario"],
            "family": r["family"],
            "D_comp_auc": r["D_comp_auc"],
            "D_yield_auc": r["D_yield_auc"],
            "D_comp_auc_fallback": r["D_comp_auc_fallback"],
            "D_yield_auc_fallback": r["D_yield_auc_fallback"],
            "simultaneous_competition": r["simultaneous_competition"],
            "reciprocity_mismatch": r["reciprocity_mismatch"],
            "high_support_primary": r["high_support_primary"],
            "source_table": "cell_level_directional_ipv.csv",
        }
        for r in cells
    ]
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.55)

    ax = fig.add_subplot(gs[:, 0])
    high = [r for r in cells if r["high_support_primary"] == "True"]
    low = [r for r in cells if r["high_support_primary"] != "True"]
    ax.scatter(
        [fnum(r["D_comp_auc"]) for r in low],
        [fnum(r["D_yield_auc"]) for r in low],
        s=18,
        c=PALETTE["neutral_light"],
        edgecolor="black",
        linewidth=0.3,
        marker="o",
        label="low support",
    )
    ax.scatter(
        [fnum(r["D_comp_auc"]) for r in high],
        [fnum(r["D_yield_auc"]) for r in high],
        s=22,
        c=PALETTE["baseline_dark"],
        edgecolor="black",
        linewidth=0.3,
        marker="^",
        label="high support",
    )
    ax.set_xlabel("D_comp AUC\nbelow-norm competitive shortfall")
    ax.set_ylabel("D_yield AUC\nabove-norm yielding excess")
    ax.set_title("Directional IPV signature")
    ax.legend(fontsize=6, loc="upper right")
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    metrics = ["D_comp_auc", "D_yield_auc", "D_comp_auc_fallback", "D_yield_auc_fallback"]
    means = [np.nanmean([fnum(r[m]) for r in cells]) for m in metrics]
    nonzero = [sum(fnum(r[m]) > 0 for r in cells) for m in metrics]
    bars = ax.bar(np.arange(len(metrics)), means, color=[PALETTE["baseline_dark"], PALETTE["ours_large"], PALETTE["baseline_mid"], PALETTE["ours_base"]], edgecolor="black", linewidth=0.6)
    for i, (bar, nz) in enumerate(zip(bars, nonzero)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"nz={nz}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([clean_label(m.replace("_auc", ""), 11) for m in metrics])
    ax.set_ylabel("Mean AUC")
    ax.set_title("Deviation summaries")
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    metrics = ["simultaneous_competition", "reciprocity_mismatch"]
    means = [np.nanmean([fnum(r[m]) for r in cells]) for m in metrics]
    nonzero = [sum(fnum(r[m]) > 0 for r in cells) for m in metrics]
    bars = ax.bar(np.arange(len(metrics)), means, color=[PALETTE["teal"], PALETTE["gold"]], edgecolor="black", linewidth=0.6)
    for bar, nz in zip(bars, nonzero):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"nz={nz}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([clean_label(m, 14) for m in metrics])
    ax.set_ylabel("Mean diagnostic value")
    ax.set_title("Interaction diagnostics")
    add_panel_label(ax, "c")
    set_clean(ax)

    save_figure(
        fig,
        "fig04_directional_ipv_signature",
        "D_comp and D_yield directional IPV signatures",
        "D_comp and D_yield are directional diagnostic deviations relative to the InterHub human conditional norm, not standalone social ground-truth labels.",
        rows,
        [
            "cell_id",
            "team",
            "scenario",
            "family",
            "D_comp_auc",
            "D_yield_auc",
            "D_comp_auc_fallback",
            "D_yield_auc_fallback",
            "simultaneous_competition",
            "reciprocity_mismatch",
            "high_support_primary",
            "source_table",
        ],
        ["cell_level_directional_ipv.csv", "g0r_cond_001_status.json", "ipv_sign_contract.md"],
        "Scatter points are cells; bars show means with nonzero counts annotated.",
        {
            "a": "Cell-level D_comp and D_yield relationship.",
            "b": "Mean high-support and fallback deviation summaries.",
            "c": "Mean interaction diagnostics and nonzero counts.",
        },
        "The sign convention is valid, but the diagnostic values are not external behavioral labels.",
    )


def fig05_before_after(data: dict[str, object]) -> None:
    before = primary_row(data["confirmatory_before"], "primary_loto_confirmatory")
    after = primary_row(data["confirmatory"], "primary_loto_confirmatory")
    metrics = [
        ("delta_spearman", "delta_spearman_ci_low", "delta_spearman_ci_high", "p_delta_spearman_greater", "Delta Spearman"),
        ("delta_mae_reduction", "delta_mae_reduction_ci_low", "delta_mae_reduction_ci_high", "p_delta_mae_reduction_greater", "Delta MAE reduction"),
        ("delta_cv_r2", "delta_cv_r2_ci_low", "delta_cv_r2_ci_high", "p_delta_cv_r2_greater", "Delta CV-R2"),
    ]
    rows: list[dict[str, object]] = []
    for label, src in [("before_scenario_fix", before), ("after_scenario_fix", after)]:
        for metric, lo, hi, pfield, display in metrics:
            rows.append(
                {
                    "panel": "all",
                    "scenario_fix_state": label,
                    "metric": metric,
                    "metric_label": display,
                    "estimate": src[metric],
                    "ci_low": src[lo],
                    "ci_high": src[hi],
                    "p_greater": src[pfield],
                    "n_cells": src["n_cells_in_sample"],
                    "effect_direction": src.get("effect_direction_spearman" if metric == "delta_spearman" else "effect_direction_cv_r2", ""),
                    "source_table": "confirmatory_results.csv" if label == "after_scenario_fix" else "confirmatory_results__before_scenario_fix.csv",
                }
            )
    fig = plt.figure(figsize=(7.2, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.5)
    colors = [PALETTE["neutral_mid"], PALETTE["baseline_dark"]]
    for idx, (metric, lo, hi, pfield, display) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        vals = [fnum(before[metric]), fnum(after[metric])]
        lows = [fnum(before[lo]), fnum(after[lo])]
        highs = [fnum(before[hi]), fnum(after[hi])]
        yerr = np.array([[vals[i] - lows[i] for i in range(2)], [highs[i] - vals[i] for i in range(2)]])
        bars = ax.bar([0, 1], vals, color=colors, edgecolor="black", linewidth=0.7, hatch=["//", ""])
        ax.errorbar([0, 1], vals, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=3)
        add_zero_line(ax, "y")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before\nN=48", "After\nN=53"])
        ax.set_title(display)
        ax.set_ylabel("Effect estimate" if idx == 0 else "")
        for i, src in enumerate([before, after]):
            ax.text(i, vals[i] + (0.03 if vals[i] >= 0 else -0.05), f"p={fnum(src[pfield]):.2f}", ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=6)
        add_panel_label(ax, chr(ord("a") + idx))
        set_clean(ax)
    save_figure(
        fig,
        "fig05_scenario_fix_before_after",
        "Capacity-matched comparison before and after scenario correction",
        "The scenario-label fix changed the primary LOTO delta Spearman from -0.110 to +0.137; both intervals cross zero.",
        rows,
        ["panel", "scenario_fix_state", "metric", "metric_label", "estimate", "ci_low", "ci_high", "p_greater", "n_cells", "effect_direction", "source_table"],
        ["confirmatory_results__before_scenario_fix.csv", "confirmatory_results.csv", "scenario_fix_result_delta.md"],
        "Bars are effect estimates; error bars are bootstrap 95% CIs from the source tables.",
        {
            "a": "Before/after delta Spearman.",
            "b": "Before/after delta MAE reduction.",
            "c": "Before/after delta CV-R2.",
        },
        "The corrected favorable direction is nonsignificant and not a validation result.",
    )


def fig06_negative_controls(data: dict[str, object]) -> None:
    controls = data["negative_controls"]
    primary = primary_row(data["confirmatory"], "primary_loto_confirmatory")
    rows: list[dict[str, object]] = [
        {
            "panel": "a",
            "control_name": "primary_directional_ipv",
            "expectation_kind": "primary",
            "delta_spearman": primary["delta_spearman"],
            "delta_spearman_ci_low": primary["delta_spearman_ci_low"],
            "delta_spearman_ci_high": primary["delta_spearman_ci_high"],
            "p_delta_spearman_greater": primary["p_delta_spearman_greater"],
            "pass_expected": "not_applicable",
            "matches_or_exceeds_primary": "reference",
            "source_table": "confirmatory_results.csv",
        }
    ]
    primary_delta = fnum(primary["delta_spearman"])
    for r in controls:
        rows.append(
            {
                "panel": "a",
                "control_name": r["control_name"],
                "expectation_kind": r["expectation_kind"],
                "delta_spearman": r["delta_spearman"],
                "delta_spearman_ci_low": r["delta_spearman_ci_low"],
                "delta_spearman_ci_high": r["delta_spearman_ci_high"],
                "p_delta_spearman_greater": r["p_delta_spearman_greater"],
                "pass_expected": r["pass_expected"],
                "matches_or_exceeds_primary": str(fnum(r["delta_spearman"]) >= primary_delta),
                "source_table": "negative_controls.csv",
            }
        )
    plot_rows = sorted(rows, key=lambda r: fnum(r["delta_spearman"]), reverse=True)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], wspace=0.45, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(plot_rows))[::-1]
    vals = [fnum(r["delta_spearman"]) for r in plot_rows]
    lows = [fnum(r["delta_spearman_ci_low"]) for r in plot_rows]
    highs = [fnum(r["delta_spearman_ci_high"]) for r in plot_rows]
    colors = []
    hatches = []
    for r in plot_rows:
        if r["control_name"] == "primary_directional_ipv":
            colors.append(PALETTE["baseline_dark"])
            hatches.append("")
        elif r["matches_or_exceeds_primary"] == "True":
            colors.append(PALETTE["delta_down"])
            hatches.append("xx")
        else:
            colors.append(PALETTE["neutral_light"])
            hatches.append("//")
    for yi, val, lo, hi, color, hatch in zip(y, vals, lows, highs, colors, hatches):
        ax.plot([lo, hi], [yi, yi], color="black", lw=0.8)
        ax.scatter([val], [yi], s=34, color=color, edgecolor="black", marker="o", hatch=hatch, zorder=3)
    add_zero_line(ax)
    ax.axvline(primary_delta, color=PALETTE["baseline_dark"], lw=0.8, ls=":", label="primary delta")
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(r["control_name"], 24) for r in plot_rows])
    ax.set_xlabel("Delta Spearman with 95% CI")
    ax.set_title("Controls match or exceed primary")
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    counts = {
        "controls >= primary": sum(1 for r in rows if r["matches_or_exceeds_primary"] == "True"),
        "controls below primary": sum(1 for r in rows if r["source_table"] == "negative_controls.csv" and r["matches_or_exceeds_primary"] == "False"),
    }
    count_labels = list(counts)
    count_vals = list(counts.values())
    x = np.arange(len(count_labels))
    bars = ax.bar(x, count_vals, color=[PALETTE["delta_down"], PALETTE["neutral_light"]], edgecolor="black", linewidth=0.6)
    bars[0].set_hatch("xx")
    ax.set_xticks(x)
    ax.set_ylabel("Control count")
    ax.set_title("Specificity diagnostic")
    ax.set_xticklabels([clean_label(v, 12) for v in count_labels], rotation=0)
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    degradation = [r for r in rows if r["expectation_kind"] == "degradation"]
    labels = [r["control_name"] for r in degradation]
    vals = [fnum(r["delta_spearman"]) for r in degradation]
    bars = ax.bar(np.arange(len(vals)), vals, color=PALETTE["neutral_mid"], edgecolor="black", linewidth=0.6, hatch="..")
    add_zero_line(ax, "y")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([clean_label(v, 11) for v in labels])
    ax.set_ylabel("Delta Spearman")
    ax.set_title("Degradation controls failed")
    add_panel_label(ax, "c")
    set_clean(ax)

    save_figure(
        fig,
        "fig06_negative_controls",
        "Negative-control specificity diagnostic",
        "Multiple controls match or exceed the primary delta Spearman, so the apparent favorable direction is not IPV-specific.",
        rows,
        [
            "panel",
            "control_name",
            "expectation_kind",
            "delta_spearman",
            "delta_spearman_ci_low",
            "delta_spearman_ci_high",
            "p_delta_spearman_greater",
            "pass_expected",
            "matches_or_exceeds_primary",
            "source_table",
        ],
        ["negative_controls.csv", "confirmatory_results.csv"],
        "Panel a shows delta Spearman with bootstrap 95% CIs; hatched red markers match or exceed the primary estimate.",
        {
            "a": "Primary and control deltas with uncertainty intervals.",
            "b": "Count of controls matching or exceeding the primary estimate.",
            "c": "Degradation controls that did not degrade as expected.",
        },
        "Controls prevent an IPV-specific robustness claim.",
    )


def fig07_state_dependence(data: dict[str, object]) -> None:
    rows_all = [r for r in data["state_dependence"] if r["interpretable_flag"] == "interpretable"]
    favorable = sorted(rows_all, key=lambda r: fnum(r["effect"]))[:6]
    reverse = sorted(rows_all, key=lambda r: fnum(r["effect"]), reverse=True)[:6]
    selected = favorable + reverse
    source_rows: list[dict[str, object]] = []
    for kind, subset in [("strongest_favorable", favorable), ("strongest_reverse", reverse)]:
        for r in subset:
            source_rows.append(
                {
                    "panel": "a",
                    "selection": kind,
                    "stratum_type": r["stratum_type"],
                    "stratum": r["stratum"],
                    "variant": r["variant"],
                    "n": r["n"],
                    "effect": r["effect"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "q_value_fdr": r["q_value_fdr"],
                    "usage_class": r["usage_class"],
                    "source_table": "state_dependence_results.csv",
                }
            )
    usage_counts: dict[str, int] = {}
    for r in rows_all:
        usage_counts[r["usage_class"]] = usage_counts.get(r["usage_class"], 0) + 1
    for k, v in usage_counts.items():
        source_rows.append(
            {
                "panel": "c",
                "selection": "usage_count",
                "stratum_type": "",
                "stratum": k,
                "variant": "",
                "n": v,
                "effect": "",
                "ci_low": "",
                "ci_high": "",
                "q_value_fdr": "",
                "usage_class": k,
                "source_table": "state_dependence_results.csv",
            }
        )
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], wspace=0.48, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    labels = [f"{r['stratum_type']}:{r['stratum']} ({r['variant']})" for r in selected]
    y = np.arange(len(selected))[::-1]
    vals = [fnum(r["effect"]) for r in selected]
    lows = [fnum(r["ci_low"]) for r in selected]
    highs = [fnum(r["ci_high"]) for r in selected]
    colors = [PALETTE["baseline_dark"] if v < 0 else PALETTE["delta_down"] for v in vals]
    for yi, val, lo, hi, color in zip(y, vals, lows, highs, colors):
        ax.plot([lo, hi], [yi, yi], color="black", lw=0.8)
        ax.scatter([val], [yi], s=30, color=color, edgecolor="black", zorder=3)
    add_zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(v, 28) for v in labels])
    ax.set_xlabel("Spearman rho with 95% CI")
    ax.set_title("Exploratory strata remain boundary-only")
    add_panel_label(ax, "a")
    set_clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    min_q = min(fnum(r["q_value_fdr"]) for r in rows_all if not math.isnan(fnum(r["q_value_fdr"])))
    bars = ax.bar(["minimum q", "threshold"], [min_q, 0.10], color=[PALETTE["neutral_mid"], PALETTE["baseline_dark"]], edgecolor="black", linewidth=0.6, hatch=["//", ""])
    ax.set_ylabel("FDR q-value")
    ax.set_ylim(0, max(min_q * 1.15, 0.2))
    ax.set_title("No FDR-stable stratum")
    for bar, val in zip(bars, [min_q, 0.10]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=7)
    add_panel_label(ax, "b")
    set_clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    labels = list(usage_counts)
    vals = [usage_counts[k] for k in labels]
    ax.bar(np.arange(len(vals)), vals, color=[PALETTE["neutral_light"], PALETTE["baseline_soft"], PALETTE["delta_down"]][: len(vals)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([clean_label(v, 12) for v in labels])
    ax.set_ylabel("Interpretable rows")
    ax.set_title("Usage classes")
    add_panel_label(ax, "c")
    set_clean(ax)

    save_figure(
        fig,
        "fig07_state_dependence_boundary",
        "State-dependence boundary map",
        "No interpretable exploratory state-dependence row reached q<=0.10, so state dependence does not rescue the primary result.",
        source_rows,
        ["panel", "selection", "stratum_type", "stratum", "variant", "n", "effect", "ci_low", "ci_high", "q_value_fdr", "usage_class", "source_table"],
        ["state_dependence_results.csv", "state_dependence_report.md"],
        "Panel a shows Spearman rho with bootstrap 95% CIs; panel b compares minimum FDR q with the q<=0.10 threshold.",
        {
            "a": "Strongest favorable and reverse exploratory rows.",
            "b": "Minimum FDR q against the reporting threshold.",
            "c": "Interpretable-row usage classes.",
        },
        "Local strata are exploratory and cannot be presented as robustness or mechanism proof.",
    )


def fig08_replication(data: dict[str, object]) -> None:
    impl = data["implementation_v2"]
    rows = [
        {
            "panel": "all",
            "comparison_id": r["comparison_id"],
            "source": r["source"],
            "delta_spearman": r["delta_spearman"],
            "delta_cv_r2": r["delta_cv_r2"],
            "delta_spearman_abs_diff_vs_phase7": r["delta_spearman_abs_diff_vs_phase7"],
            "delta_cv_r2_abs_diff_vs_phase7": r["delta_cv_r2_abs_diff_vs_phase7"],
            "direction_reproduced": r["direction_reproduced"],
            "within_metric_tolerance": r["within_metric_tolerance"],
            "within_prediction_tolerance": r["within_prediction_tolerance"],
            "mode": r["mode"],
            "source_table": "implementation_comparison_v2.csv",
        }
        for r in impl
    ]
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.55)
    labels = [
        "Phase7\ncorrected",
        "Replication2\nindependent",
        "Replication2\nreported-alpha",
    ]
    colors = [PALETTE["baseline_dark"], PALETTE["baseline_mid"], PALETTE["ours_large"]]
    for idx, metric in enumerate(["delta_spearman", "delta_cv_r2"]):
        ax = fig.add_subplot(gs[0, idx])
        vals = [fnum(r[metric]) for r in impl]
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6, hatch=["", "//", ""])
        add_zero_line(ax, "y")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Effect estimate" if idx == 0 else "")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", fontsize=6)
        add_panel_label(ax, chr(ord("a") + idx))
        set_clean(ax)

    ax = fig.add_subplot(gs[0, 2])
    diffs = [fnum(r["delta_spearman_abs_diff_vs_phase7"]) for r in impl]
    bars = ax.bar(np.arange(len(diffs)), diffs, color=colors, edgecolor="black", linewidth=0.6, hatch=["", "//", ""])
    ax.set_xticks(np.arange(len(diffs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute difference")
    ax.set_title("Delta Spearman difference")
    for bar, val, r in zip(bars, diffs, impl):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, "dir yes" if r["direction_reproduced"] == "True" else "dir no", ha="center", fontsize=6)
    add_panel_label(ax, "c")
    set_clean(ax)

    save_figure(
        fig,
        "fig08_independent_replication",
        "Independent replication agreement",
        "Replication2 reproduced corrected N=53 direction, verifying implementation soundness while leaving robustness and specificity unresolved.",
        rows,
        [
            "panel",
            "comparison_id",
            "source",
            "delta_spearman",
            "delta_cv_r2",
            "delta_spearman_abs_diff_vs_phase7",
            "delta_cv_r2_abs_diff_vs_phase7",
            "direction_reproduced",
            "within_metric_tolerance",
            "within_prediction_tolerance",
            "mode",
            "source_table",
        ],
        ["implementation_comparison_v2.csv", "replication2_status.json"],
        "Bars show corrected Phase 7 and replication2 estimates; labels mark directional reproduction.",
        {
            "a": "Delta Spearman reproduction.",
            "b": "Delta CV-R2 reproduction.",
            "c": "Absolute delta-Spearman differences vs Phase 7.",
        },
        "Implementation reproduction is not equivalent to predictive-utility validation.",
    )


def fig09_tier_map(data: dict[str, object]) -> None:
    claim_rows = data["claim_boundary"]
    status_order = {"accepted": 2, "rejected": 1, "blocked": 0}
    source_rows = []
    for r in claim_rows:
        source_rows.append(
            {
                "panel": "a",
                "claim_id": r["claim_id"],
                "claim_status": r["status"],
                "tier": r["tier"],
                "allowed_paper_wording": r["allowed_paper_wording"],
                "forbidden_wording": "Boundary warning recorded in source matrix; exact prohibited text intentionally not repeated.",
                "evidence_path": r["evidence_path"],
                "source_table": "claim_boundary_matrix.csv",
            }
        )
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    ids = [r["claim_id"] for r in claim_rows]
    values = np.array([[status_order.get(r["status"], -1)] for r in claim_rows])
    cmap = mpl.colors.ListedColormap([PALETTE["neutral_mid"], PALETTE["neutral_light"], PALETTE["baseline_dark"]])
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels(ids)
    ax.set_xticks([0])
    ax.set_xticklabels(["Tier B\nboundary"])
    for i, r in enumerate(claim_rows):
        short = {"accepted": "accepted", "rejected": "not supported", "blocked": "blocked"}.get(r["status"], r["status"])
        ax.text(0, i, short, ha="center", va="center", color="white" if r["status"] == "accepted" else "black", fontsize=6)
    ax.set_title("Claim-level evidence map")
    add_panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    counts: dict[str, int] = {}
    for r in claim_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    labels = ["accepted", "rejected", "blocked"]
    vals = [counts.get(k, 0) for k in labels]
    bars = ax.bar(labels, vals, color=[PALETTE["baseline_dark"], PALETTE["neutral_light"], PALETTE["neutral_mid"]], edgecolor="black", linewidth=0.6, hatch=["", "//", ".."])
    ax.set_ylabel("Claims")
    ax.set_title("Boundary counts")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, str(val), ha="center", fontsize=7)
    ax.text(
        0.02,
        -0.22,
        "Tier B: diagnostic external evidence,\nnot a robust utility validation.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    add_panel_label(ax, "b")
    set_clean(ax)

    save_figure(
        fig,
        "fig09_tier_b_evidence_map",
        "Tier B evidence boundary map",
        "The settled Tier B conclusion is limitation-aware diagnostic external evidence, not a robust incremental-utility claim.",
        source_rows,
        [
            "panel",
            "claim_id",
            "claim_status",
            "tier",
            "allowed_paper_wording",
            "forbidden_wording",
            "evidence_path",
            "source_table",
        ],
        ["claim_boundary_matrix.csv", "tier_decision.json", "paper_handoff.md"],
        "Cells encode accepted, not-supported, and blocked claim boundaries; text labels avoid color-only status encoding.",
        {
            "a": "Claim-level status map.",
            "b": "Status-count summary and Tier B interpretation.",
        },
        "Do not promote rejected or blocked claim rows into manuscript claims.",
    )


def write_nature_skill_manifest() -> None:
    skill_files = {
        "SKILL.md": NATURE_SKILL_DIR / "SKILL.md",
        "manifest.yaml": NATURE_SKILL_DIR / "manifest.yaml",
        "static/core/contract.md": NATURE_SKILL_DIR / "static" / "core" / "contract.md",
        "static/core/stance.md": NATURE_SKILL_DIR / "static" / "core" / "stance.md",
        "static/fragments/backend/python.md": NATURE_SKILL_DIR / "static" / "fragments" / "backend" / "python.md",
        "references/api.md": NATURE_SKILL_DIR / "references" / "api.md",
        "references/figure-contract.md": NATURE_SKILL_DIR / "references" / "figure-contract.md",
        "references/qa-contract.md": NATURE_SKILL_DIR / "references" / "qa-contract.md",
        "references/design-theory.md": NATURE_SKILL_DIR / "references" / "design-theory.md",
        "references/common-patterns.md": NATURE_SKILL_DIR / "references" / "common-patterns.md",
        "references/figure-legend-conventions.md": NATURE_SKILL_DIR / "references" / "figure-legend-conventions.md",
    }
    for path in skill_files.values():
        SOURCE_READS.append(path)
    manifest = {
        "skill_name": "nature-figure",
        "skill_path": str(NATURE_SKILL_DIR),
        "skill_version": "2.0.0",
        "backend": "python",
        "generation_time_utc": BUILD_TIME_UTC,
        "worker_id": WORKER_ID,
        "run_id": RUN_ID,
        "how_invoked": [
            "Loaded SKILL router and manifest.",
            "Loaded always-load core contract and stance.",
            "Resolved backend as Python from the task-provided Python/matplotlib workflow.",
            "Loaded Python backend fragment and relevant API, figure-contract, design, common-pattern, legend and QA references.",
            "Generated all reader-facing figures with matplotlib using the Nature skill rules.",
        ],
        "file_hashes": {
            label: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for label, path in skill_files.items()
        },
        "loaded_reference_summary": "Python-only quantitative-grid workflow; editable SVG text; PDF TrueType text; source CSV and metadata JSON for each figure.",
    }
    write_json(RB / "nature_skill_manifest.json", manifest)


def table_html(rows: list[dict[str, object]], columns: list[str], max_rows: int = 8) -> str:
    out = ["<table>", "<thead><tr>"]
    for col in columns:
        out.append(f"<th>{escape(col)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows[:max_rows]:
        out.append("<tr>")
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                txt = f"{val:.3g}"
            else:
                txt = str(val)
            out.append(f"<td>{escape(txt)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def figure_block(fig_id: str) -> str:
    row = next(r for r in FIGURE_ROWS if r["figure_id"] == fig_id)
    png = entry_link(RUN_ROOT / row["png"])
    svg = entry_link(RUN_ROOT / row["svg"])
    pdf = entry_link(RUN_ROOT / row["pdf"])
    source = entry_link(RUN_ROOT / row["source_csv"])
    meta = entry_link(RUN_ROOT / row["metadata_json"])
    return (
        f'<figure id="{escape(fig_id)}">'
        f'<img src="{escape(png)}" alt="{escape(row["title"])}">'
        f'<figcaption><strong>{escape(row["title"])}</strong>. {escape(row["caption"])} '
        f'<span class="downloads">Downloads: <a href="{escape(svg)}">SVG</a> | '
        f'<a href="{escape(pdf)}">PDF</a> | <a href="{escape(source)}">source CSV</a> | '
        f'<a href="{escape(meta)}">metadata JSON</a>.</span></figcaption>'
        "</figure>"
    )


def build_evidence_rows(data: dict[str, object]) -> list[dict[str, str]]:
    boundary = data["claim_boundary"]
    fig_for_claim = {
        "C_SCOPE_TOP5": "fig01_provenance_coverage",
        "C_COORD_SOURCE": "fig01_provenance_coverage",
        "C_PRIMARY_DIRECTION": "fig05_scenario_fix_before_after",
        "C_INCREMENTAL_UTILITY": "fig09_tier_b_evidence_map",
        "C_LOSO_GENERALIZATION": "fig05_scenario_fix_before_after",
        "C_SAFE_SUBSETS": "fig05_scenario_fix_before_after",
        "C_NEGATIVE_CONTROLS": "fig06_negative_controls",
        "C_REPLICATION": "fig08_independent_replication",
        "C_H3_BLIND_ANNOTATION": "",
        "C_NPC_BOUNDARY": "",
        "C_STATE_DEPENDENCE": "fig07_state_dependence_boundary",
        "C_CONFORMAL_BOUNDARY": "fig03_support_ood_abstention",
        "C_SIGN_NARRATIVE": "fig04_directional_ipv_signature",
    }
    rows = []
    for r in boundary:
        claim_id = r["claim_id"]
        fig_id = fig_for_claim.get(claim_id, "")
        artifact_path = fig_id and f"01_results/figures/{fig_id}.png"
        rows.append(
            {
                "claim_id": claim_id,
                "claim_text": r["claim_text"],
                "claim_status": r["status"],
                "confirmatory_status": "confirmatory" if claim_id in {"C_PRIMARY_DIRECTION", "C_INCREMENTAL_UTILITY"} else "boundary_or_sensitivity",
                "evidence_type": "figure+table" if fig_id else "table/status",
                "artifact_path": artifact_path,
                "table_id": r["evidence_path"],
                "figure_id": fig_id,
                "reviewer_status": "Tier B accepted boundary" if r["status"] == "accepted" else ("blocked/excluded from Tier basis" if r["status"] == "blocked" else "not supported as positive evidence"),
                "limitation": "Power-limited top-five cohort; use exact boundary in claim text.",
                "allowed_paper_wording": r["allowed_paper_wording"],
                "forbidden_wording": "Boundary warning recorded in claim_boundary_matrix.csv; exact prohibited text intentionally not repeated.",
            }
        )
    return rows


def build_html(data: dict[str, object], evidence_rows: list[dict[str, str]]) -> str:
    primary = primary_row(data["confirmatory"], "primary_loto_confirmatory")
    loso = primary_row(data["confirmatory"], "secondary_loso_generalization")
    before = primary_row(data["confirmatory_before"], "primary_loto_confirmatory")
    tier_decision = data["tier_decision"]
    annotation = data["annotation_status"]
    rep2 = data["replication2"]
    figures = {r["figure_id"]: r for r in FIGURE_ROWS}
    fig_table_rows = [
        {
            "figure": r["figure_id"],
            "title": r["title"],
            "source_csv": r["source_csv"],
            "metadata": r["metadata_json"],
        }
        for r in FIGURE_ROWS
    ]
    artifact_links = [
        ("README.md", "../README.md"),
        ("TRACEABILITY.md", "../TRACEABILITY.md"),
        ("evidence.csv", "../evidence.csv"),
        ("execution_status.json", "../execution_status.json"),
        ("figure_manifest.csv", "../01_results/figures/figure_manifest.csv"),
        ("nature_skill_manifest.json", "../02_process/19_report_build/nature_skill_manifest.json"),
        ("report_build_status.json", "../02_process/19_report_build/report_build_status.json"),
        ("artifact_manifest.csv", "../02_process/19_report_build/artifact_manifest.csv"),
    ]
    toc = [
        ("title", "1. Title and identity"),
        ("executed", "2. Executed vs not executed"),
        ("sources", "3. Data sources and coverage"),
        ("gate-minus1", "4. Gate -1 provenance"),
        ("gate0", "5. Gate 0 measurement"),
        ("freeze", "6. Frozen analysis"),
        ("directional", "7. Directional IPV"),
        ("capacity", "8. Capacity-matched baseline"),
        ("heldout", "9. Held-out CV"),
        ("controls", "10. Negative controls"),
        ("blind", "11. Blind annotation status"),
        ("state", "12. State dependence"),
        ("npc", "13. NPC boundary"),
        ("redteam", "14. Red team"),
        ("replication", "15. Independent replication"),
        ("tier", "16. Tier decision"),
        ("limits", "17. Limitations and blockers"),
        ("boundary", "18. Confirmatory/exploratory boundary"),
        ("reproduce", "19. Reproduction commands"),
        ("artifacts", "20. Artifact index"),
        ("evidence", "21. Claim-level evidence matrix"),
        ("handoff", "22. Paper-handoff summary"),
    ]
    status_rows = [
        {"phase": "Gate -1", "status": data["gate_minus1"]["status"], "scope": data["gate_minus1"]["status_scope"]},
        {"phase": "Gate 0", "status": data["gate0"]["status"], "scope": "measurement/sign/support boundary"},
        {"phase": "Freeze review", "status": data["freeze_review"]["status"], "scope": data["freeze_review"]["status_label"]},
        {"phase": "Red team v1", "status": data["red_team1"]["status"], "scope": "scenario-label blocker found"},
        {"phase": "Scenario fix", "status": data["scenario_fix"]["status"], "scope": "120/150 cells relabelled"},
        {"phase": "Red team v3", "status": data["red_team3"]["status"], "scope": data["red_team3"]["clearance_verdict"]},
        {"phase": "Tier review", "status": tier_decision["tier"], "scope": tier_decision["allowed_framing"]},
    ]
    result_rows = [
        {
            "analysis": "Before scenario fix primary LOTO",
            "N": before["n_cells_in_sample"],
            "delta_spearman": f"{fnum(before['delta_spearman']):.3f}",
            "CI": f"[{fnum(before['delta_spearman_ci_low']):.3f}, {fnum(before['delta_spearman_ci_high']):.3f}]",
            "p": before["p_delta_spearman_greater"],
            "status": "nonsignificant, pre-fix label path",
        },
        {
            "analysis": "After scenario fix primary LOTO",
            "N": primary["n_cells_in_sample"],
            "delta_spearman": f"{fnum(primary['delta_spearman']):.3f}",
            "CI": f"[{fnum(primary['delta_spearman_ci_low']):.3f}, {fnum(primary['delta_spearman_ci_high']):.3f}]",
            "p": primary["p_delta_spearman_greater"],
            "status": "suggestive, nonsignificant, non-specific",
        },
        {
            "analysis": "LOSO generalization",
            "N": loso["n_cells_in_sample"],
            "delta_spearman": f"{fnum(loso['delta_spearman']):.3f}",
            "CI": f"[{fnum(loso['delta_spearman_ci_low']):.3f}, {fnum(loso['delta_spearman_ci_high']):.3f}]",
            "p": loso["p_delta_spearman_greater"],
            "status": "approximately zero; not scenario-general",
        },
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ003 Tier B Offline Reader</title>
<style>
:root {{
  --ink: #272727;
  --muted: #606060;
  --line: #d8d8d8;
  --panel: #f7f7f7;
  --accent: #484878;
  --warn: #b64342;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: var(--ink); background: white; line-height: 1.5; }}
header, main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
header {{ border-bottom: 1px solid var(--line); }}
h1 {{ font-size: 28px; line-height: 1.12; margin: 0 0 12px; letter-spacing: 0; }}
h2 {{ font-size: 20px; margin: 36px 0 10px; border-top: 1px solid var(--line); padding-top: 18px; }}
h3 {{ font-size: 14px; margin: 18px 0 8px; }}
p, li {{ font-size: 14px; }}
.lede {{ font-size: 16px; max-width: 980px; }}
.tag {{ display: inline-block; border: 1px solid var(--line); padding: 2px 7px; margin: 2px 4px 2px 0; font-size: 12px; background: var(--panel); }}
.tier {{ color: var(--accent); font-weight: 700; }}
.warn {{ color: var(--warn); font-weight: 700; }}
nav {{ columns: 2; column-gap: 32px; margin: 18px 0 0; }}
nav a {{ display: block; break-inside: avoid; color: var(--accent); text-decoration: none; margin: 3px 0; font-size: 13px; }}
figure {{ margin: 20px 0; padding: 0; }}
figure img {{ width: 100%; height: auto; border: 1px solid var(--line); }}
figcaption {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
.downloads {{ white-space: normal; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 12px; }}
th, td {{ border: 1px solid var(--line); padding: 6px 7px; vertical-align: top; }}
th {{ background: var(--panel); text-align: left; }}
code {{ background: var(--panel); padding: 1px 4px; border-radius: 3px; }}
pre {{ background: var(--panel); border: 1px solid var(--line); padding: 12px; overflow-x: auto; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }}
.callout {{ border-left: 4px solid var(--accent); background: var(--panel); padding: 10px 12px; margin: 12px 0; }}
.callout.warnbox {{ border-left-color: var(--warn); }}
@media print {{
  header, main {{ max-width: none; padding: 12mm; }}
  nav {{ columns: 1; }}
  a {{ color: black; text-decoration: none; }}
  figure img {{ break-inside: avoid; }}
  h2 {{ break-after: avoid; }}
}}
</style>
</head>
<body>
<header id="title">
<h1>RQ003 NSFC External Evidence - Corrected Tier B Reader Package</h1>
<p class="lede"><span class="tier">Settled conclusion:</span> {escape(SETTLED_CONCLUSION)}</p>
<p>
<span class="tag">RUN_ID: {escape(RUN_ID)}</span>
<span class="tag">Git: {escape(GIT_HEAD)}</span>
<span class="tag">Plan SHA-256: {escape(PLAN_SHA256)}</span>
<span class="tag">Generated UTC: {escape(BUILD_TIME_UTC)}</span>
</p>
<p>This offline package is reader-facing only. It does not edit manuscript files and does not claim manuscript changes.</p>
<nav>
{''.join(f'<a href="#{escape(sec_id)}">{escape(label)}</a>' for sec_id, label in toc)}
</nav>
</header>
<main>
<section id="executed">
<h2>2. Executed vs not executed</h2>
<p>The corrected package executed provenance, measurement, frozen confirmatory analysis, controls, state-dependence boundary mapping, replication, red-team closure, and Tier review. It did not execute a full-universe validity analysis, a real two-human blind-annotation analysis, an NPC effect analysis, or any paper edit.</p>
{table_html(status_rows, ["phase", "status", "scope"])}
</section>
<section id="sources">
<h2>3. Data sources and coverage</h2>
<p>Source data are the corrected post-scenario-fix tables under <code>01_results/tables/</code> plus Tier review and interpretation-fix artifacts. The plan was not a strict pre-registration; it is an archived approved plan and freeze package with later correction/review history.</p>
{figure_block("fig01_provenance_coverage")}
</section>
<section id="gate-minus1">
<h2>4. Gate -1 provenance</h2>
<p>Gate -1 passed for the approved top-five cohort only: 150/150 planned cells had clean usable mapping. The full scored universe remains outside the analysis-ready scope because 33/300 cells are missing or not clean under the clean-mapping definition.</p>
{figure_block("fig02_missingness_selection_bias")}
</section>
<section id="gate0">
<h2>5. Gate 0 measurement</h2>
<p>Gate 0 passed the measurement firewall, sign orientation, rolling-window handling, and InterHub-calibrated support-boundary checks. NSFC does NOT support nominal conformal coverage; the conformal element is a support/OOD/abstention boundary.</p>
{figure_block("fig03_support_ood_abstention")}
</section>
<section id="freeze">
<h2>6. Frozen analysis</h2>
<p>The freeze review passed with Phase 4 conditions that were later closed by the real-optimizer directional-IPV check. The freeze was outcome-clean; contaminated earlier freeze-review artifacts were quarantined and are not the active basis.</p>
</section>
<section id="directional">
<h2>7. Directional IPV</h2>
<p><code>D_comp</code> marks below-norm competitive shortfall and <code>D_yield</code> marks above-norm yielding excess relative to the InterHub human conditional norm. These are directional diagnostics, not independent behavioral ground truth.</p>
{figure_block("fig04_directional_ipv_signature")}
</section>
<section id="capacity">
<h2>8. Capacity-matched baseline</h2>
<p>The prespecified comparison adds directional IPV summaries to a kinematic+safety baseline under matched model capacity. The corrected favorable direction is not sufficient for a utility claim because significance, generalization, and specificity are not established.</p>
</section>
<section id="heldout">
<h2>9. Held-out CV</h2>
<p>The scenario-label fix changed the primary LOTO delta Spearman from {fnum(before['delta_spearman']):.3f} to {fnum(primary['delta_spearman']):.3f}; both intervals cross zero. LOSO was approximately zero and did not establish scenario generalization.</p>
{table_html(result_rows, ["analysis", "N", "delta_spearman", "CI", "p", "status"])}
{figure_block("fig05_scenario_fix_before_after")}
</section>
<section id="controls">
<h2>10. Negative controls</h2>
<p>Negative controls prevent a specificity claim: future-window, time-shuffle, counterpart-swap, role-flip, and sign-flip controls matched or exceeded the primary delta Spearman; degradation controls also failed as diagnostics.</p>
{figure_block("fig06_negative_controls")}
</section>
<section id="blind">
<h2>11. Blind annotation status</h2>
<p>Blind annotation has NO real two-human results in this package. H3 is <span class="warn">BLOCKED</span>: agreement and event-IPV tests were not computed, simulated labels were not generated, and the templates are not evidence.</p>
{table_html([annotation], ["h3_status", "overall_status", "reason", "agreement_computed", "event_ipv_test_computed", "simulated_labels_generated"])}
</section>
<section id="state">
<h2>12. State dependence</h2>
<p>State-dependence results are exploratory boundary mapping. No interpretable row reached q&lt;=0.10, and reverse or low-n rows require abstention rather than rescue interpretation.</p>
{figure_block("fig07_state_dependence_boundary")}
</section>
<section id="npc">
<h2>13. NPC boundary</h2>
<p>NPC pre-onset matching is not identifiable from allowed fields. No NPC effect analysis was run; future wording should be limited to matched opportunity-structure requirements if independent pre-onset evidence is added.</p>
</section>
<section id="redteam">
<h2>14. Red team</h2>
<p>Red-team v1 found a scenario-label blocker and A1 safety-reporting issue. The scenario fix reconciled 150 cells and relabelled 120/150 cells. Red-team v2 found interpretation blockers; the interpretation fix closed them. Red-team v3 reported PASS_NO_BLOCKERS and cleared the package for Tier review.</p>
</section>
<section id="replication">
<h2>15. Independent replication</h2>
<p>Replication1 was discrepant under an independent model specification. Replication2 used corrected N=53, exactly reproduced the reported-alpha refit direction, and independently reproduced a favorable direction under separate tuning. This verifies implementation soundness, not robust utility.</p>
{figure_block("fig08_independent_replication")}
</section>
<section id="tier">
<h2>16. Tier decision</h2>
<p>The settled Tier is <span class="tier">B</span>: diagnostic alignment without robust independent incremental utility. The conclusion is power-limited, top-five only, and non-specific under controls.</p>
{figure_block("fig09_tier_b_evidence_map")}
</section>
<section id="limits">
<h2>17. Limitations and blockers</h2>
<ul>
<li>Top-five cohort only; full 20-team universe is not analysis-ready.</li>
<li>Primary corrected sample is N=53 cells.</li>
<li>Outcome is an official/generated coordination score, not an independently annotated behavioral endpoint.</li>
<li>Safe-subset support is mechanical: S1 and S2 duplicate the primary cells, while S3 has n=6.</li>
<li>H3 is blocked pending real two-human annotation files.</li>
<li>NPC effect analysis is blocked by missing pre-onset matching fields.</li>
</ul>
</section>
<section id="boundary">
<h2>18. Confirmatory/exploratory boundary</h2>
<p>Confirmatory: corrected primary LOTO against the prespecified kinematic+safety baseline. Sensitivity: LOSO, LOFO, safe subsets, fallback-inclusive analysis, before/after scenario-fix comparison, and negative controls. Exploratory: state-dependence boundary mapping and NPC feasibility audit. Null, reverse, blocked, and partial results are retained in the report rather than hidden.</p>
</section>
<section id="reproduce">
<h2>19. Reproduction commands</h2>
<p>From the repository root:</p>
<pre><code>data/derived/onsite_competition/RQ003_nsfc_external_evidence/{RUN_ID}/model_cache/venv/bin/python reports/studies/RQ003_nsfc_external_evidence/{RUN_ID}/02_process/19_report_build/build_reader_package.py</code></pre>
<p>The command rebuilds figures, source CSVs, metadata, report HTML, manifests, and verification status from the corrected tables.</p>
</section>
<section id="artifacts">
<h2>20. Artifact index</h2>
<p>All links are relative and offline. HTML entries in <code>00_entry</code> and <code>90_report</code> are byte-identical.</p>
<ul>
{''.join(f'<li><a href="{escape(url)}">{escape(label)}</a></li>' for label, url in artifact_links)}
</ul>
{table_html(fig_table_rows, ["figure", "title", "source_csv", "metadata"], max_rows=20)}
</section>
<section id="evidence">
<h2>21. Claim-level evidence matrix</h2>
<p>The evidence matrix records claim status, confirmatory boundary, linked artifact, reviewer status, limitation, allowed wording, and wording warnings.</p>
{table_html(evidence_rows, ["claim_id", "claim_status", "confirmatory_status", "artifact_path", "figure_id", "reviewer_status"], max_rows=20)}
</section>
<section id="handoff">
<h2>22. Paper-handoff summary</h2>
<p>Allowed framing: {escape(tier_decision['allowed_framing'])}</p>
<p>Recommended use is a limitation-aware supplementary diagnostic or short cautionary sentence. This phase did not modify the paper repository, and it does not supply a validation headline.</p>
</section>
</main>
</body>
</html>
"""
    return html


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append((key, value))


def check_html(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    parser = LinkCollector()
    parser.feed(text)
    errors: list[str] = []
    if "http://" in text or "https://" in text or "cdn" in text.lower():
        errors.append("external reference token found")
    section_count = len(re.findall(r"<section\s+id=", text))
    if section_count < 21:
        errors.append(f"section count too low: {section_count}")
    for attr, url in parser.links:
        clean = unquote(url.split("#", 1)[0])
        if not clean:
            continue
        if clean.startswith(("http:", "https:", "file:", "//")) or os.path.isabs(clean):
            errors.append(f"non-offline {attr}: {url}")
            continue
        target = (path.parent / clean).resolve()
        if not target.exists():
            errors.append(f"missing {attr}: {url}")
    required_phrases = [
        "plan was not a strict pre-registration",
        "Blind annotation has NO real two-human results",
        "NSFC does NOT support nominal conformal coverage",
        "Tier is",
        "Null, reverse, blocked, and partial results",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"missing required phrase: {phrase}")
    return {"path": rel(path), "sections": section_count + 1, "links_checked": len(parser.links), "errors": errors, "status": "PASS" if not errors else "FAIL"}


def check_file_uri_open(path: Path) -> dict[str, object]:
    errors: list[str] = []
    uri = path.resolve().as_uri()
    try:
        with urlopen(uri, timeout=10) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - recorded in report status.
        text = ""
        errors.append(f"file URI open failed: {exc}")
    checks = {
        "bytes": len(text.encode("utf-8")),
        "has_title": "RQ003 NSFC External Evidence" in text,
        "sections": text.count("<section id=") + (1 if '<header id="title"' in text else 0),
        "figures": text.count("<figure id="),
        "has_tier_b": "Tier B" in text or ("Tier is" in text and ">B<" in text),
        "has_h3_blocked": "H3 is" in text and "BLOCKED" in text,
        "has_no_nominal_coverage": "NSFC does NOT support nominal conformal coverage" in text,
    }
    for key in ["has_title", "has_tier_b", "has_h3_blocked", "has_no_nominal_coverage"]:
        if checks[key] is not True:
            errors.append(f"missing open-check signal: {key}")
    if checks["sections"] < 22:
        errors.append(f"section count too low after file open: {checks['sections']}")
    if checks["figures"] != 9:
        errors.append(f"figure count after file open is {checks['figures']}, expected 9")
    return {"path": rel(path), "uri_scheme": "file", "checks": checks, "errors": errors, "status": "PASS" if not errors else "FAIL"}


def scan_forbidden(paths: list[Path]) -> dict[str, object]:
    phrases = [
        "new information beyond kinematics",
        "expert-rated coordination",
        "proven no effect",
        "stable validation of IPV",
        "independent new signal beyond trajectory",
    ]
    hits: list[dict[str, str]] = []
    for path in paths:
        if path.suffix.lower() not in {".html", ".md", ".csv", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in phrases:
            if phrase.lower() in text:
                hits.append({"path": rel(path), "phrase": phrase})
    return {"status": "PASS" if not hits else "FAIL", "hits": hits}


def check_figure_manifest() -> dict[str, object]:
    errors: list[str] = []
    for row in FIGURE_ROWS:
        for key in ["svg", "pdf", "png", "source_csv", "metadata_json"]:
            p = RUN_ROOT / row[key]
            if not p.exists() or p.stat().st_size == 0:
                errors.append(f"{row['figure_id']} missing {key}")
        meta = json.loads((RUN_ROOT / row["metadata_json"]).read_text(encoding="utf-8"))
        if meta.get("nature_skill", {}).get("name") != "nature-figure":
            errors.append(f"{row['figure_id']} missing nature skill metadata")
    return {"status": "PASS" if not errors else "FAIL", "figure_count": len(FIGURE_ROWS), "errors": errors}


def collect_artifacts(extra_paths: list[Path]) -> list[dict[str, str]]:
    paths: list[Path] = []
    for path in WRITES + extra_paths:
        if path.exists() and path not in paths and path.is_file():
            paths.append(path)
    rows = []
    for path in sorted(paths, key=lambda p: rel(p)):
        rows.append(
            {
                "artifact_path": rel(path),
                "sha256": sha256_file(path),
                "size_bytes": str(path.stat().st_size),
                "artifact_type": path.suffix.lstrip(".") or "file",
                "produced_by": WORKER_ID,
                "phase": "10",
                "purpose": artifact_purpose(path),
                "created_utc": BUILD_TIME_UTC,
            }
        )
    return rows


def artifact_purpose(path: Path) -> str:
    name = path.name
    if name.startswith("fig"):
        return "reader-facing Nature-style figure export/source/metadata"
    if name == "index.html":
        return "offline HTML reader entry"
    if name == "evidence.csv":
        return "claim-level evidence matrix"
    if name == "execution_status.json":
        return "run-level report-build status"
    if name in {"README.md", "TRACEABILITY.md"}:
        return "reader-facing package documentation"
    if name.endswith("manifest.csv") or name.endswith("manifest.json"):
        return "report-build manifest"
    if name.endswith("status.json") or name == "worker_report.json":
        return "report-build status"
    return "report-build artifact"


def write_documentation(data: dict[str, object], evidence_rows: list[dict[str, str]], tests: dict[str, object]) -> None:
    figure_manifest = FIGS / "figure_manifest.csv"
    write_csv(
        figure_manifest,
        FIGURE_ROWS,
        [
            "figure_id",
            "title",
            "svg",
            "pdf",
            "png",
            "source_csv",
            "metadata_json",
            "primary_source_tables",
            "claim",
            "figure_role",
            "caption",
            "skill_name",
            "skill_version",
            "backend",
            "created_utc",
        ],
    )
    write_csv(
        RUN_ROOT / "evidence.csv",
        evidence_rows,
        [
            "claim_id",
            "claim_text",
            "claim_status",
            "confirmatory_status",
            "evidence_type",
            "artifact_path",
            "table_id",
            "figure_id",
            "reviewer_status",
            "limitation",
            "allowed_paper_wording",
            "forbidden_wording",
        ],
    )
    readme = f"""# RQ003 Corrected Tier B Reader Package

Run: `{RUN_ID}`

Settled conclusion: {SETTLED_CONCLUSION}

Entry points:

- `00_entry/index.html`
- `90_report/index.html`

This package is offline and reader-facing. It presents corrected results,
Tier B boundaries, null/non-specific results, blocked H3 status, NPC boundary
status, and source-linked figures. It did not modify manuscript files.

Key files:

- `01_results/figures/figure_manifest.csv`
- `02_process/19_report_build/nature_skill_manifest.json`
- `evidence.csv`
- `TRACEABILITY.md`
- `execution_status.json`
"""
    write_text(RUN_ROOT / "README.md", readme)

    trace_rows = []
    for fig in FIGURE_ROWS:
        trace_rows.append(
            f"| {fig['figure_id']} | {fig['primary_source_tables']} | {fig['source_csv']} | {fig['metadata_json']} |"
        )
    traceability = f"""# RQ003 Phase 10 Traceability

Run: `{RUN_ID}`

All reader-facing figures were generated under the `nature-figure` Python
backend contract. Source result tables were read-only.

## Figure Traceability

| Figure | Primary source tables | Figure source CSV | Metadata |
|---|---|---|---|
{chr(10).join(trace_rows)}

## Claim Traceability

See `evidence.csv` for claim-level artifact, table, figure, reviewer status,
allowed wording, and limitation fields.

## Verification Summary

- Figure manifest completeness: {tests['figure_manifest']['status']}
- Entry HTML offline check: {tests['entry_html']['status']}
- Compat HTML offline check: {tests['rep90_html']['status']}
- Byte-identical HTML entries: {tests['byte_identical']}
- Forbidden-wording scan: {tests['forbidden_scan']['status']}
"""
    write_text(RUN_ROOT / "TRACEABILITY.md", traceability)


def main() -> int:
    identity = verify_identity()
    if identity["status"] != "PASS":
        print("BLOCKED: identity verification failed", json.dumps(identity, indent=2))
        return 2

    data = load_sources()
    write_nature_skill_manifest()

    fig01_provenance(data)
    fig02_missingness(data)
    fig03_support(data)
    fig04_directional_ipv(data)
    fig05_before_after(data)
    fig06_negative_controls(data)
    fig07_state_dependence(data)
    fig08_replication(data)
    fig09_tier_map(data)

    evidence_rows = build_evidence_rows(data)
    html = build_html(data, evidence_rows)
    write_text(ENTRY / "index.html", html)
    write_text(REP90 / "index.html", html)

    preliminary_tests = {
        "figure_manifest": check_figure_manifest(),
        "entry_html": check_html(ENTRY / "index.html"),
        "rep90_html": check_html(REP90 / "index.html"),
        "file_uri_open": check_file_uri_open(ENTRY / "index.html"),
        "byte_identical": sha256_file(ENTRY / "index.html") == sha256_file(REP90 / "index.html"),
    }
    write_documentation(data, evidence_rows, preliminary_tests | {"forbidden_scan": {"status": "PENDING"}})

    extra_paths = [SCRIPT_PATH, FIGS / "figure_manifest.csv", RUN_ROOT / "evidence.csv", RUN_ROOT / "README.md", RUN_ROOT / "TRACEABILITY.md"]
    write_json(
        RB / "report_build_status.json",
        {
            "schema_version": "rq003_phase10_report_build_status_v1",
            "worker_id": WORKER_ID,
            "run_id": RUN_ID,
            "status": "PENDING_FINAL_VERIFICATION",
            "generated_utc": BUILD_TIME_UTC,
        },
    )
    write_csv(
        RB / "artifact_manifest.csv",
        [],
        ["artifact_path", "sha256", "size_bytes", "artifact_type", "produced_by", "phase", "purpose", "created_utc"],
    )
    tests = {
        "figure_manifest": check_figure_manifest(),
        "entry_html": check_html(ENTRY / "index.html"),
        "rep90_html": check_html(REP90 / "index.html"),
        "file_uri_open": check_file_uri_open(ENTRY / "index.html"),
        "byte_identical": sha256_file(ENTRY / "index.html") == sha256_file(REP90 / "index.html"),
    }
    tests["forbidden_scan"] = scan_forbidden(
        [ENTRY / "index.html", REP90 / "index.html", RUN_ROOT / "README.md", RUN_ROOT / "TRACEABILITY.md", RUN_ROOT / "evidence.csv"]
        + [RUN_ROOT / r["metadata_json"] for r in FIGURE_ROWS]
        + [RUN_ROOT / r["source_csv"] for r in FIGURE_ROWS]
    )
    all_tests_pass = (
        tests["figure_manifest"]["status"] == "PASS"
        and tests["entry_html"]["status"] == "PASS"
        and tests["rep90_html"]["status"] == "PASS"
        and tests["file_uri_open"]["status"] == "PASS"
        and tests["byte_identical"] is True
        and tests["forbidden_scan"]["status"] == "PASS"
    )
    status = "PASS" if all_tests_pass else "FAIL"
    status_json = {
        "schema_version": "rq003_phase10_report_build_status_v1",
        "worker_id": WORKER_ID,
        "role": "Phase 10 report build",
        "run_id": RUN_ID,
        "generated_utc": BUILD_TIME_UTC,
        "status": status,
        "identity": identity,
        "settled_conclusion": SETTLED_CONCLUSION,
        "nature_skill_manifest": "02_process/19_report_build/nature_skill_manifest.json",
        "figure_count": len(FIGURE_ROWS),
        "html_entries": ["00_entry/index.html", "90_report/index.html"],
        "tests": tests,
        "spec_deviations": [
            "START_HERE.md and main_workflow.log were not modified because the task write scope was limited to the run package and append-only meta artifact index."
        ],
        "unresolved_blockers": [],
    }
    write_json(RB / "report_build_status.json", status_json)
    write_json(RUN_ROOT / "execution_status.json", status_json)

    artifacts = collect_artifacts(extra_paths + [RB / "report_build_status.json", RUN_ROOT / "execution_status.json"])
    write_csv(
        RB / "artifact_manifest.csv",
        artifacts,
        ["artifact_path", "sha256", "size_bytes", "artifact_type", "produced_by", "phase", "purpose", "created_utc"],
    )

    append_rows = []
    for row in artifacts:
        path = RUN_ROOT / row["artifact_path"] if not Path(row["artifact_path"]).is_absolute() else Path(row["artifact_path"])
        append_rows.append(
            {
                "artifact_path": str(path.resolve()),
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "produced_by": WORKER_ID,
                "command": COMMANDS_RUN[-1],
                "purpose": row["purpose"],
                "phase": "10",
            }
        )
    append_csv(META / "artifact_index.csv", append_rows, ["artifact_path", "sha256", "size_bytes", "produced_by", "command", "purpose", "phase"])

    file_access_lines = [
        f"worker_id: {WORKER_ID}",
        f"run_id: {RUN_ID}",
        f"generated_utc: {BUILD_TIME_UTC}",
        "",
        "READ:",
    ]
    for path in sorted({p for p in SOURCE_READS}, key=lambda p: str(p)):
        file_access_lines.append(f"- {rel(path)}")
    file_access_lines.extend(["", "WROTE:"])
    for path in sorted({p for p in WRITES}, key=lambda p: str(p)):
        file_access_lines.append(f"- {rel(path)}")
    write_text(RB / "file_access_manifest.txt", "\n".join(file_access_lines) + "\n")

    worker_report = {
        "status": status,
        "worker_id": WORKER_ID,
        "role": "Phase 10 report build",
        "run_id": RUN_ID,
        "scope_completed": [
            "Identity verified",
            "Nature skill manifest written",
            "Nine reader-facing figures generated as SVG/PDF/PNG/source CSV/metadata JSON",
            "Offline HTML entries written byte-identically",
            "Evidence and traceability artifacts written",
            "Artifact index appended",
        ],
        "files_created_or_modified": [rel(p) for p in sorted({p for p in WRITES}, key=lambda p: str(p))],
        "commands_run": COMMANDS_RUN,
        "tests_run": tests,
        "spec_deviations": status_json["spec_deviations"],
        "unresolved_blockers": [],
        "recommended_next_codex_task": "Phase 11 final independent report reviewer",
    }
    write_json(RB / "worker_report.json", worker_report)

    # Refresh manifests after writing final worker/file-access reports.
    final_artifacts = collect_artifacts(
        extra_paths
        + [
            RB / "report_build_status.json",
            RUN_ROOT / "execution_status.json",
            RB / "artifact_manifest.csv",
            RB / "file_access_manifest.txt",
            RB / "worker_report.json",
            RB / "nature_skill_manifest.json",
        ]
    )
    write_csv(
        RB / "artifact_manifest.csv",
        final_artifacts,
        ["artifact_path", "sha256", "size_bytes", "artifact_type", "produced_by", "phase", "purpose", "created_utc"],
    )
    status_json["artifact_count"] = len(final_artifacts)
    write_json(RB / "report_build_status.json", status_json)
    write_json(RUN_ROOT / "execution_status.json", status_json)
    worker_report["files_created_or_modified"] = [rel(p) for p in sorted({p for p in WRITES}, key=lambda p: str(p))]
    write_json(RB / "worker_report.json", worker_report)

    print(json.dumps({"status": status, "figure_count": len(FIGURE_ROWS), "tests": tests}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
