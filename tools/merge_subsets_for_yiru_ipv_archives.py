from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = (
    ROOT
    / "interhub_traj_lane"
    / "0_raw_data"
    / "subsets_for_yiru"
    / "selected_interactive_segments_equalized.csv"
)
RESULT_ROOT = (
    ROOT
    / "interhub_traj_lane"
    / "1_ipv_estimation_results"
    / "subsets_for_yiru"
)
DEFAULT_CASES_ZIP = RESULT_ROOT / "cases.zip"
DEFAULT_NUPLAN_ZIP = RESULT_ROOT / "nuplan_train.zip"
DEFAULT_ANALYSIS_DIR = RESULT_ROOT / "_combined_ipv_analysis"
DEFAULT_OUTPUT_CSV = RESULT_ROOT / "selected_interactive_segments_equalized_with_ipv_combined.csv"

IPV_COLUMNS = [
    "ipv_key_agent_1_mean",
    "ipv_key_agent_1_error_mean",
    "ipv_key_agent_2_mean",
    "ipv_key_agent_2_error_mean",
    "ipv_result_status",
    "ipv_result_case_dir",
    "ipv_result_error",
    "ipv_pkl_file",
    "ipv_segment_id",
    "ipv_reference_source_1",
    "ipv_reference_source_2",
    "ipv_result_source_zip",
    "ipv_result_zip_member",
    "ipv_downsample_factor",
    "ipv_has_metadata",
    "ipv_has_excel",
]


@dataclass(frozen=True)
class ArchiveSpec:
    path: Path
    name: str
    priority: int
    dataset_filter: str | None = None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def metadata_case_dir(member: str) -> str:
    suffix = "/data/metadata.json"
    return member[: -len(suffix)] if member.endswith(suffix) else str(Path(member).parent.parent)


def xlsx_member_for_metadata(member: str) -> str:
    return member[: -len("metadata.json")] + "ipv_results.xlsx"


ROW_NAME_RE = re.compile(r"/row_(\d{5})_[^/]+/data/metadata\.json$")


def row_index_from_member(member: str) -> int | None:
    match = ROW_NAME_RE.search("/" + member.replace("\\", "/"))
    return int(match.group(1)) if match else None


def read_archive_cases(spec: ArchiveSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(spec.path) as zf:
        members = set(zf.namelist())
        metadata_members = sorted(member for member in members if member.endswith("/data/metadata.json"))
        for member in metadata_members:
            has_excel = xlsx_member_for_metadata(member) in members
            try:
                metadata = json.loads(zf.read(member).decode("utf-8"))
                metadata_error = ""
            except Exception as exc:  # pylint: disable=broad-except
                metadata = {}
                metadata_error = f"{type(exc).__name__}: {exc}"

            dataset = normalize(metadata.get("dataset"))
            if spec.dataset_filter and dataset != spec.dataset_filter:
                continue

            summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
            row_index = metadata.get("row_index", row_index_from_member(member))
            row = {
                "row_index": int(row_index) if row_index is not None and str(row_index).isdigit() else row_index,
                "dataset": dataset,
                "folder": normalize(metadata.get("folder")),
                "scenario_idx": normalize(metadata.get("scenario_idx")),
                "track_id": normalize(metadata.get("track_id")),
                "key_agents": normalize(metadata.get("key_agents")),
                "metadata_status": normalize(metadata.get("status")),
                "metadata_error": metadata_error,
                "has_metadata": not bool(metadata_error),
                "has_excel": has_excel,
                "source_zip": spec.name,
                "source_zip_path": str(spec.path),
                "zip_member": member,
                "case_dir": metadata_case_dir(member),
                "priority": spec.priority,
                "pkl_file": normalize(metadata.get("pkl_file")),
                "segment_id": normalize(metadata.get("segment_id")),
                "reference_source_1": normalize(metadata.get("reference_source_1")),
                "reference_source_2": normalize(metadata.get("reference_source_2")),
                "downsample_factor": metadata.get("downsample_factor"),
                "ipv_key_agent_1_mean": summary.get("ipv_key_agent_1_mean"),
                "ipv_key_agent_1_error_mean": summary.get("ipv_key_agent_1_error_mean"),
                "ipv_key_agent_2_mean": summary.get("ipv_key_agent_2_mean"),
                "ipv_key_agent_2_error_mean": summary.get("ipv_key_agent_2_error_mean"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def row_matches_source(source_row: pd.Series, artifact: pd.Series) -> tuple[bool, str]:
    fields = ["dataset", "folder", "scenario_idx", "key_agents", "track_id"]
    mismatches = []
    for field in fields:
        left = normalize(source_row.get(field))
        right = normalize(artifact.get(field))
        if left != right:
            mismatches.append(f"{field}:csv={left}|metadata={right}")
    return not mismatches, "; ".join(mismatches)


def choose_artifacts(source: pd.DataFrame, artifacts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if artifacts.empty:
        empty = pd.DataFrame(columns=["row_index"])
        return empty, empty

    usable = artifacts.copy()
    usable["row_index"] = pd.to_numeric(usable["row_index"], errors="coerce").astype("Int64")
    usable = usable.dropna(subset=["row_index"]).copy()
    usable["row_index"] = usable["row_index"].astype(int)
    usable["complete_artifact"] = (
        usable["has_metadata"].eq(True)
        & usable["has_excel"].eq(True)
        & usable["metadata_status"].eq("ok")
    )

    matched_rows = []
    for _, artifact in usable.iterrows():
        row_index = int(artifact["row_index"])
        if 0 <= row_index < len(source):
            ok, mismatch = row_matches_source(source.iloc[row_index], artifact)
        else:
            ok, mismatch = False, "row_index_out_of_range"
        record = artifact.to_dict()
        record["matches_source_csv"] = ok
        record["source_mismatch"] = mismatch
        record["eligible"] = bool(record["complete_artifact"] and ok)
        matched_rows.append(record)
    matched = pd.DataFrame(matched_rows)

    eligible = matched[matched["eligible"]].copy()
    eligible = eligible.sort_values(
        ["row_index", "priority", "source_zip"],
        ascending=[True, False, True],
    )
    selected = eligible.drop_duplicates("row_index", keep="first").copy()
    return selected, matched


def build_output_csv(
    source: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    output_csv: Path,
    completed_only_csv: Path,
) -> pd.DataFrame:
    output = source.copy()
    for col in IPV_COLUMNS:
        output[col] = ""

    selected_by_row = selected.set_index("row_index", drop=False) if not selected.empty else pd.DataFrame()
    for row_index in output.index:
        if not selected.empty and row_index in selected_by_row.index:
            artifact = selected_by_row.loc[row_index]
            output.at[row_index, "ipv_key_agent_1_mean"] = artifact["ipv_key_agent_1_mean"]
            output.at[row_index, "ipv_key_agent_1_error_mean"] = artifact["ipv_key_agent_1_error_mean"]
            output.at[row_index, "ipv_key_agent_2_mean"] = artifact["ipv_key_agent_2_mean"]
            output.at[row_index, "ipv_key_agent_2_error_mean"] = artifact["ipv_key_agent_2_error_mean"]
            output.at[row_index, "ipv_result_status"] = "ok"
            output.at[row_index, "ipv_result_case_dir"] = artifact["case_dir"]
            output.at[row_index, "ipv_result_error"] = ""
            output.at[row_index, "ipv_pkl_file"] = artifact["pkl_file"]
            output.at[row_index, "ipv_segment_id"] = artifact["segment_id"]
            output.at[row_index, "ipv_reference_source_1"] = artifact["reference_source_1"]
            output.at[row_index, "ipv_reference_source_2"] = artifact["reference_source_2"]
            output.at[row_index, "ipv_result_source_zip"] = artifact["source_zip"]
            output.at[row_index, "ipv_result_zip_member"] = artifact["zip_member"]
            output.at[row_index, "ipv_downsample_factor"] = artifact["downsample_factor"]
            output.at[row_index, "ipv_has_metadata"] = artifact["has_metadata"]
            output.at[row_index, "ipv_has_excel"] = artifact["has_excel"]
        else:
            output.at[row_index, "ipv_result_status"] = "missing_case"
            output.at[row_index, "ipv_result_error"] = "No complete matching metadata/xlsx artifact in merged archives"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output[output["ipv_result_status"].eq("ok")].to_csv(
        completed_only_csv, index=False, encoding="utf-8-sig"
    )
    return output


def summarize(
    source: pd.DataFrame,
    artifacts: pd.DataFrame,
    selected: pd.DataFrame,
    matched: pd.DataFrame,
    output: pd.DataFrame,
    *,
    csv_path: Path,
    output_csv: Path,
    completed_only_csv: Path,
    analysis_dir: Path,
    archive_specs: Iterable[ArchiveSpec],
) -> dict[str, object]:
    complete = output[output["ipv_result_status"].eq("ok")]
    missing = output[~output["ipv_result_status"].eq("ok")]
    summary = {
        "csv_path": rel(csv_path),
        "output_csv": rel(output_csv),
        "completed_only_csv": rel(completed_only_csv),
        "source_rows": int(len(source)),
        "artifact_metadata_cases": int(len(artifacts)),
        "eligible_artifacts": int(matched["eligible"].sum()) if not matched.empty else 0,
        "selected_rows": int(len(selected)),
        "complete_rows": int(len(complete)),
        "missing_rows": int(len(missing)),
        "complete_by_dataset": complete["dataset"].value_counts().sort_index().to_dict(),
        "missing_by_dataset": missing["dataset"].value_counts().sort_index().to_dict(),
        "selected_by_source_zip": complete["ipv_result_source_zip"].value_counts().sort_index().to_dict(),
        "selected_by_dataset_source_zip": {
            f"{dataset}|{source_zip}": int(count)
            for (dataset, source_zip), count in complete.groupby(["dataset", "ipv_result_source_zip"]).size().items()
        },
        "artifact_by_source_zip": artifacts["source_zip"].value_counts().sort_index().to_dict()
        if not artifacts.empty
        else {},
        "artifact_by_dataset_source_zip": {
            f"{dataset}|{source_zip}": int(count)
            for (dataset, source_zip), count in artifacts.groupby(["dataset", "source_zip"]).size().items()
        }
        if not artifacts.empty
        else {},
        "duplicate_eligible_artifacts": int(matched.loc[matched["eligible"], "row_index"].duplicated().sum())
        if not matched.empty
        else 0,
        "nonmatching_artifacts": int((~matched["matches_source_csv"]).sum()) if not matched.empty else 0,
        "archives": [
            {
                "name": spec.name,
                "path": rel(spec.path),
                "priority": spec.priority,
                "dataset_filter": spec.dataset_filter,
            }
            for spec in archive_specs
        ],
        "reports": {
            "analysis_dir": rel(analysis_dir),
            "case_artifact_report_csv": rel(analysis_dir / "case_artifact_report.csv"),
            "row_match_report_csv": rel(analysis_dir / "row_match_report.csv"),
            "missing_or_incomplete_rows_csv": rel(analysis_dir / "missing_or_incomplete_rows.csv"),
            "summary_json": rel(analysis_dir / "summary.json"),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge subsets_for_yiru IPV artifacts from cases.zip plus nuplan_train.zip into one CSV copy."
    )
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--cases-zip", type=Path, default=DEFAULT_CASES_ZIP)
    parser.add_argument("--nuplan-zip", type=Path, default=DEFAULT_NUPLAN_ZIP)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.csv)
    archive_specs = [
        ArchiveSpec(args.cases_zip, "cases.zip", priority=10),
        ArchiveSpec(args.nuplan_zip, "nuplan_train.zip", priority=20, dataset_filter="nuplan_train"),
    ]
    artifacts = pd.concat([read_archive_cases(spec) for spec in archive_specs], ignore_index=True)
    selected, matched = choose_artifacts(source, artifacts)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv
    completed_only_csv = output_csv.with_name(output_csv.stem + "_completed_only.csv")
    output = build_output_csv(
        source,
        selected,
        output_csv=output_csv,
        completed_only_csv=completed_only_csv,
    )

    artifacts.to_csv(args.analysis_dir / "case_artifact_report.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(args.analysis_dir / "row_match_report.csv", index=False, encoding="utf-8-sig")
    output[~output["ipv_result_status"].eq("ok")].to_csv(
        args.analysis_dir / "missing_or_incomplete_rows.csv", index=False, encoding="utf-8-sig"
    )
    summary = summarize(
        source,
        artifacts,
        selected,
        matched,
        output,
        csv_path=args.csv,
        output_csv=output_csv,
        completed_only_csv=completed_only_csv,
        analysis_dir=args.analysis_dir,
        archive_specs=archive_specs,
    )
    (args.analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
