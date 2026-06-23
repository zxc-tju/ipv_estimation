#!/usr/bin/env python3
"""Build the onsite competition all-team analysis package.

The package mirrors the top-five subset layout, but includes every official
scored team. Large videos and replay archives are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ONSITE_ROOT = REPO_ROOT / "data" / "onsite_competition"
MANIFEST_ROOT = ONSITE_ROOT / "00_manifest"
LEGACY_SCORE_ROOT = REPO_ROOT / "archived" / "onsite_competition_results_legacy"
DEFAULT_OUTPUT_ROOT = ONSITE_ROOT / "all_teams_dataset"
ARCHIVED_SOURCE_ROOT = (
    REPO_ROOT / "archived" / "onsite_competition_raw_and_top5_subset_20260623"
)
RAW_ROOT_CANDIDATES = [
    ONSITE_ROOT / "raw",
    ARCHIVED_SOURCE_ROOT / "raw",
]

AREA_DIR_TO_SLUG = {
    "北京赛区": "beijing",
    "上海赛区": "shanghai",
}
AREA_SLUG_TO_DIR = {value: key for key, value in AREA_DIR_TO_SLUG.items()}

REQUIRED_LOGS = [
    "monitor.log",
    "simulation_trajectory.log",
    "vehicle_perception_simulation_trajectory.log",
    "vehicle_trajectory.log",
]
OPTIONAL_LOGS = [
    "vehicle_perception_trajectory.log",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def score_float(row: dict[str, str]) -> float:
    value = row.get("score_mean_comprehensive", "").strip()
    return float(value) if value else float("-inf")


def existing_raw_root() -> Path:
    for candidate in RAW_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    choices = ", ".join(str(path) for path in RAW_ROOT_CANDIDATES)
    raise FileNotFoundError(f"onsite raw root not found; checked: {choices}")


def raw_area_path(raw_root: Path, area_dir_or_slug: str) -> Path:
    return raw_root / AREA_DIR_TO_SLUG.get(area_dir_or_slug, area_dir_or_slug)


def source_relative_path(area: str, team_dir: str, *parts: str) -> str:
    return str(Path(AREA_SLUG_TO_DIR.get(area, area)) / team_dir / Path(*parts))


def repo_relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def hardlink_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"destination already exists: {dest}")
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def support_role(path: Path) -> str | None:
    suffix = path.suffix.lower()
    name = path.name
    if suffix == ".pdf":
        return "diagnosis_pdf"
    if suffix == ".txt":
        return "team_info_txt"
    if suffix == ".log" and name.startswith("onsite_"):
        return "onsite_runtime_log"
    if suffix == ".png":
        return "appeal_or_evidence_png"
    if suffix == ".docx":
        return "appeal_docx"
    return None


def folder_name(area_rank: int, team_code: str, official_name: str) -> str:
    safe_name = official_name.replace("/", "_").replace(" ", "_")
    return f"{area_rank:02d}_{team_code}_{safe_name}"


def load_team_ranks(
    score_rows: list[dict[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    area_rank: dict[str, int] = {}
    global_rank: dict[str, int] = {}
    for area in sorted({row["score_area"] for row in score_rows}):
        area_rows = [row for row in score_rows if row["score_area"] == area]
        for index, row in enumerate(
            sorted(area_rows, key=score_float, reverse=True), start=1
        ):
            area_rank[row["team_code"]] = index
    for index, row in enumerate(sorted(score_rows, key=score_float, reverse=True), start=1):
        global_rank[row["team_code"]] = index
    return area_rank, global_rank


def load_score_name_to_code() -> dict[str, str]:
    mapping = {}
    for row in read_csv(LEGACY_SCORE_ROOT / "anno_trans.csv"):
        mapping[row["original_name"]] = row["team_code"]
    return mapping


def load_scenario_scores(
    score_name_to_code: dict[str, str],
    coverage_by_code: dict[str, dict[str, str]],
    area_rank_by_code: dict[str, int],
    global_rank_by_code: dict[str, int],
) -> list[dict[str, object]]:
    source_files = {
        "beijing": LEGACY_SCORE_ROOT / "score_beijing_abilities.csv",
        "shanghai": LEGACY_SCORE_ROOT / "score_shanghai_abilities.csv",
    }
    rows: list[dict[str, object]] = []
    for area, path in source_files.items():
        for row in read_csv(path):
            team_code = score_name_to_code[row["team"]]
            coverage = coverage_by_code[team_code]
            rows.append(
                {
                    "area": area,
                    "area_rank": area_rank_by_code[team_code],
                    "global_rank": global_rank_by_code[team_code],
                    "team_code": team_code,
                    "official_name": coverage["official_name"],
                    "scenario": row["scenario"],
                    "safety": row["safety"],
                    "efficiency": row["efficiency"],
                    "comfort": row["comfort"],
                    "compliance": row["compliance"],
                    "coordination": row["coordination"],
                    "comprehensive": row["comprehensive"].strip(),
                }
            )
    return sorted(rows, key=lambda row: (row["area"], row["area_rank"], row["scenario"]))


def build_package(output_root: Path, replace: bool) -> dict[str, object]:
    raw_root = existing_raw_root()

    if output_root.exists():
        if not replace:
            raise FileExistsError(
                f"{output_root} already exists; rerun with --replace to rebuild it"
            )
        shutil.rmtree(output_root)

    teams_root = output_root / "teams"
    tables_root = output_root / "tables"
    teams_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    score_rows = read_csv(MANIFEST_ROOT / "score_team_coverage.csv")
    score_rows = sorted(score_rows, key=lambda row: (row["score_area"], row["team_code"]))
    official_codes = {row["team_code"] for row in score_rows}
    coverage_by_code = {row["team_code"]: row for row in score_rows}
    area_rank_by_code, global_rank_by_code = load_team_ranks(score_rows)

    team_manifest_rows = {
        row["team_code"]: row
        for row in read_csv(MANIFEST_ROOT / "team_manifest.csv")
        if row.get("team_code") in official_codes
    }
    session_manifest_rows = [
        row
        for row in read_csv(MANIFEST_ROOT / "session_manifest.csv")
        if row.get("team_code") in official_codes
    ]
    sessions_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in session_manifest_rows:
        sessions_by_code[row["team_code"]].append(row)

    materialized_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    all_team_rows: list[dict[str, object]] = []
    all_session_rows: list[dict[str, object]] = []

    for coverage in sorted(
        score_rows, key=lambda row: (row["score_area"], area_rank_by_code[row["team_code"]])
    ):
        team_code = coverage["team_code"]
        area = coverage["score_area"]
        area_rank = area_rank_by_code[team_code]
        global_rank = global_rank_by_code[team_code]
        official_name = coverage["official_name"]
        team_manifest = team_manifest_rows.get(team_code)
        if not team_manifest:
            raise KeyError(f"no team manifest row for {team_code}")

        team_dir = team_manifest["team_dir"]
        src_team_dir = raw_area_path(raw_root, area) / team_dir
        dataset_team_dir = teams_root / area / folder_name(
            area_rank, team_code, official_name
        )
        support_dir = dataset_team_dir / "support_materials"
        sessions_dir = dataset_team_dir / "sessions"

        package_file_count = 0
        package_bytes = 0
        support_file_count = 0

        if src_team_dir.exists():
            for src in sorted(path for path in src_team_dir.iterdir() if path.is_file()):
                role = support_role(src)
                if role is None:
                    continue
                dest = support_dir / src.name
                storage_mode = hardlink_or_copy(src, dest)
                package_file_count += 1
                support_file_count += 1
                package_bytes += src.stat().st_size
                materialized_rows.append(
                    {
                        "area": area,
                        "area_rank": area_rank,
                        "global_rank": global_rank,
                        "team_code": team_code,
                        "official_name": official_name,
                        "session_id": "",
                        "file_role": role,
                        "bytes": src.stat().st_size,
                        "source_relative_path": source_relative_path(area, team_dir, src.name),
                        "current_source_relative_path": repo_relative_path(src),
                        "dataset_relative_path": str(dest.relative_to(output_root)),
                        "storage_mode": storage_mode,
                    }
                )

        materialized_session_count = 0
        for session in sorted(sessions_by_code.get(team_code, []), key=lambda row: row["session_id"]):
            session_id = session["session_id"]
            src_session_dir = raw_area_path(raw_root, area) / team_dir / session_id
            dest_session_dir = sessions_dir / session_id
            present_logs: list[str] = []
            missing_logs: list[str] = []

            for log_name in REQUIRED_LOGS + OPTIONAL_LOGS:
                src = src_session_dir / log_name
                if not src.exists():
                    if log_name in REQUIRED_LOGS:
                        missing_logs.append(log_name)
                    continue
                dest = dest_session_dir / log_name
                storage_mode = hardlink_or_copy(src, dest)
                present_logs.append(log_name)
                package_file_count += 1
                package_bytes += src.stat().st_size
                materialized_rows.append(
                    {
                        "area": area,
                        "area_rank": area_rank,
                        "global_rank": global_rank,
                        "team_code": team_code,
                        "official_name": official_name,
                        "session_id": session_id,
                        "file_role": "replay_log",
                        "bytes": src.stat().st_size,
                        "source_relative_path": source_relative_path(
                            area, team_dir, session_id, log_name
                        ),
                        "current_source_relative_path": repo_relative_path(src),
                        "dataset_relative_path": str(dest.relative_to(output_root)),
                        "storage_mode": storage_mode,
                    }
                )

            required_present = not missing_logs and src_session_dir.exists()
            if src_session_dir.exists():
                materialized_session_count += 1
            dataset_session_path = (
                str(dest_session_dir.relative_to(output_root)) if src_session_dir.exists() else ""
            )

            validation_rows.append(
                {
                    "area": area,
                    "area_rank": area_rank,
                    "global_rank": global_rank,
                    "team_code": team_code,
                    "official_name": official_name,
                    "session_id": session_id,
                    "dataset_session_path": dataset_session_path,
                    "required_logs_present": required_present,
                    "missing_required_logs": ";".join(missing_logs),
                    "present_logs": ";".join(present_logs),
                    "materialized_file_count": len(present_logs),
                    "notes": "" if required_present else "missing required replay log",
                }
            )
            all_session_row = dict(session)
            all_session_row.update(
                {
                    "area_rank": area_rank,
                    "global_rank": global_rank,
                    "dataset_session_path": dataset_session_path,
                    "present_logs": ";".join(present_logs),
                    "missing_required_logs_current": ";".join(missing_logs),
                    "required_logs_present_current": required_present,
                }
            )
            all_session_rows.append(all_session_row)

        if not sessions_by_code.get(team_code):
            validation_rows.append(
                {
                    "area": area,
                    "area_rank": area_rank,
                    "global_rank": global_rank,
                    "team_code": team_code,
                    "official_name": official_name,
                    "session_id": "",
                    "dataset_session_path": "",
                    "required_logs_present": False,
                    "missing_required_logs": ";".join(REQUIRED_LOGS),
                    "present_logs": "",
                    "materialized_file_count": 0,
                    "notes": "no materialized replay session directory",
                }
            )

        all_team_rows.append(
            {
                "area": area,
                "area_rank": area_rank,
                "global_rank": global_rank,
                "team_code": team_code,
                "official_name": official_name,
                "score_rows": coverage["score_rows"],
                "score_mean_comprehensive": coverage["score_mean_comprehensive"],
                "matched_team_dirs": coverage["matched_team_dirs"],
                "matched_session_ids": coverage["matched_session_ids"],
                "has_materialized_replay": as_bool(coverage["has_materialized_replay"]),
                "team_dir": team_dir,
                "dataset_team_path": str(dataset_team_dir.relative_to(output_root)),
                "materialized_session_count": materialized_session_count,
                "materialized_support_file_count": support_file_count,
                "materialized_file_count": package_file_count,
                "materialized_bytes": package_bytes,
                "notes": coverage.get("notes", "") or team_manifest.get("notes", ""),
            }
        )

        readme = dataset_team_dir / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        session_text = coverage["matched_session_ids"] or "none"
        readme.write_text(
            "\n".join(
                [
                    f"# {team_code} / {official_name}",
                    "",
                    f"- Area: {area}",
                    f"- Area rank: {area_rank}",
                    f"- Global rank: {global_rank}",
                    f"- Source team folder: `{repo_relative_path(src_team_dir)}`",
                    f"- Score rows: {coverage['score_rows']}",
                    f"- Mean comprehensive score: {coverage['score_mean_comprehensive']}",
                    f"- Materialized sessions: {session_text}",
                    "",
                    "Videos and replay archive zip files are intentionally excluded.",
                    "Replay logs, diagnosis/team/support files are hardlinked from raw data when present.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    scenario_rows = load_scenario_scores(
        load_score_name_to_code(),
        coverage_by_code,
        area_rank_by_code,
        global_rank_by_code,
    )

    write_csv(
        tables_root / "all_team_summary.csv",
        all_team_rows,
        [
            "area",
            "area_rank",
            "global_rank",
            "team_code",
            "official_name",
            "score_rows",
            "score_mean_comprehensive",
            "matched_team_dirs",
            "matched_session_ids",
            "has_materialized_replay",
            "team_dir",
            "dataset_team_path",
            "materialized_session_count",
            "materialized_support_file_count",
            "materialized_file_count",
            "materialized_bytes",
            "notes",
        ],
    )
    write_csv(
        tables_root / "all_session_manifest.csv",
        all_session_rows,
        [
            "area",
            "area_dir",
            "area_rank",
            "global_rank",
            "team_dir",
            "team_code",
            "official_name",
            "session_id",
            "session_relative_path",
            "dataset_session_path",
            "required_logs_present",
            "required_logs_present_current",
            "missing_required_logs",
            "missing_required_logs_current",
            "optional_logs_present",
            "present_logs",
            "json_first_line_bad_logs",
            "monitor_bytes",
            "simulation_bytes",
            "vehicle_perception_simulation_bytes",
            "vehicle_perception_bytes",
            "vehicle_trajectory_bytes",
        ],
    )
    write_csv(
        tables_root / "all_scenario_scores.csv",
        scenario_rows,
        [
            "area",
            "area_rank",
            "global_rank",
            "team_code",
            "official_name",
            "scenario",
            "safety",
            "efficiency",
            "comfort",
            "compliance",
            "coordination",
            "comprehensive",
        ],
    )
    write_csv(
        tables_root / "materialized_analysis_files.csv",
        materialized_rows,
        [
            "area",
            "area_rank",
            "global_rank",
            "team_code",
            "official_name",
            "session_id",
            "file_role",
            "bytes",
            "source_relative_path",
            "current_source_relative_path",
            "dataset_relative_path",
            "storage_mode",
        ],
    )
    write_csv(
        tables_root / "validation_summary.csv",
        validation_rows,
        [
            "area",
            "area_rank",
            "global_rank",
            "team_code",
            "official_name",
            "session_id",
            "dataset_session_path",
            "required_logs_present",
            "missing_required_logs",
            "present_logs",
            "materialized_file_count",
            "notes",
        ],
    )

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Onsite Competition All-Team Dataset",
                "",
                "Generated by `scripts/build_onsite_all_teams_dataset.py`.",
                "",
                "This package mirrors `top5_research_subset`, but includes every official scored team instead of only the top five per area.",
                "",
                "## Scope",
                "",
                f"- Official scored teams: {len(score_rows)}",
                f"- Materialized replay sessions: {len(all_session_rows)}",
                f"- Scenario score rows: {len(scenario_rows)}",
                f"- Materialized analysis files: {len(materialized_rows)}",
                f"- Source raw root: `{repo_relative_path(raw_root)}`",
                "- Videos and replay archive zip files are intentionally excluded.",
                "- Non-scored or platform folders such as `UE`, `8-高中部`, `长沙理工大学_csust`, `app.zip`, and `tjjhs_db.sql` are not part of the official all-team dataset.",
                "",
                "## Layout",
                "",
                "- `teams/<area>/<rank>_<team_code>_<official_name>/sessions/<session_id>/`: replay logs for trajectory analysis.",
                "- `teams/<area>/<rank>_<team_code>_<official_name>/support_materials/`: diagnosis PDFs, team info, onsite runtime logs, and appeal/evidence files when present.",
                "- `tables/all_team_summary.csv`: one row per official scored team.",
                "- `tables/all_session_manifest.csv`: one row per materialized replay session.",
                "- `tables/all_scenario_scores.csv`: official scenario-level score rows for all official teams.",
                "- `tables/materialized_analysis_files.csv`: source-to-dataset file map plus storage mode.",
                "- `tables/validation_summary.csv`: replay-log completeness check; teams without replay sessions are retained and marked.",
                "",
                "## Storage Note",
                "",
                "Files under `teams/` are hardlinked from the source raw root shown above when possible. Treat them as read-only analysis inputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "teams": len(score_rows),
        "sessions": len(all_session_rows),
        "scenario_rows": len(scenario_rows),
        "materialized_files": len(materialized_rows),
        "validation_rows": len(validation_rows),
        "output_root": output_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = build_package(args.output_root, args.replace)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
