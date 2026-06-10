"""Build a clean rerun CSV for full-dataset cases whose IPV is still missing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "interhub_traj_lane"
    / "1_ipv_estimation_results"
    / "full_datasets"
    / "curated_valid_ipv_cases"
    / "valid_cases_manifest.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "interhub_traj_lane"
    / "0_raw_data"
    / "full_datasets"
    / "missing_ipv_rerun_input.csv"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_CSV.with_suffix(".summary.json")

OUTPUT_COLUMNS = [
    "scene_unique_id",
    "index_scene_unique_id",
    "case_uid",
    "canonical_case_key",
    "dataset",
    "folder",
    "scenario_idx",
    "track_id",
    "start",
    "end",
    "intensity",
    "PET",
    "two/multi",
    "vehicle_type",
    "AV_included",
    "key_agents",
    "key_agents_type",
    "path_category",
    "path_relation",
    "turn_label",
    "priority_label",
    "index_row_index",
    "expected_pkl_file",
    "expected_pkl_path",
    "raw_usable_for_ipv_input",
]


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([""] * len(df), index=df.index)


def build_rerun_input(
    source_csv: Path,
    output_csv: Path,
    summary_json: Path,
    *,
    include_unusable: bool = False,
) -> Dict[str, object]:
    df = pd.read_csv(source_csv, low_memory=False)
    if "curation_status" in df.columns:
        df = df[df["curation_status"].astype(str).eq("missing_ipv_from_index")].copy()
    if "raw_usable_for_ipv_input" in df.columns and not include_unusable:
        df = df[_bool_series(df["raw_usable_for_ipv_input"])].copy()

    output = pd.DataFrame(index=df.index)
    output["scene_unique_id"] = _first_existing_column(df, ["index_scene_unique_id", "scene_unique_id"])
    output["index_scene_unique_id"] = output["scene_unique_id"]
    output["case_uid"] = df.get("case_uid", "")
    output["canonical_case_key"] = df.get("canonical_case_key", "")
    output["dataset"] = df["dataset"]
    output["folder"] = df["folder"]
    output["scenario_idx"] = df["scenario_idx"]
    output["track_id"] = df["track_id"]
    output["start"] = df["start"]
    output["end"] = df["end"]
    output["intensity"] = df.get("intensity", "")
    output["PET"] = df.get("PET", "")
    output["two/multi"] = _first_existing_column(df, ["two/multi", "two_multi"])
    output["vehicle_type"] = df.get("vehicle_type", "")
    output["AV_included"] = df.get("AV_included", "")
    output["key_agents"] = df["key_agents"]
    output["key_agents_type"] = df.get("key_agents_type", "")
    output["path_category"] = df.get("path_category", "")
    output["path_relation"] = df.get("path_relation", "")
    output["turn_label"] = df.get("turn_label", "")
    output["priority_label"] = df.get("priority_label", "")
    output["index_row_index"] = df.get("index_row_index", "")
    output["expected_pkl_file"] = _first_existing_column(
        df,
        ["expected_pkl_file", "expected_pkl_file_y", "expected_pkl_file_x"],
    )
    output["expected_pkl_path"] = df.get("expected_pkl_path", "")
    output["raw_usable_for_ipv_input"] = df.get("raw_usable_for_ipv_input", "")
    output = output[OUTPUT_COLUMNS].copy()

    duplicate_keys = int(output[["folder", "scenario_idx", "key_agents", "track_id"]].duplicated().sum())
    if duplicate_keys:
        raise ValueError(f"Rerun input has duplicate case keys: {duplicate_keys}")
    missing_scene_id = int(output["scene_unique_id"].isna().sum() + output["scene_unique_id"].astype(str).eq("").sum())
    if missing_scene_id:
        raise ValueError(f"Rerun input has rows without scene_unique_id: {missing_scene_id}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "rows": int(len(output)),
        "duplicate_keys": duplicate_keys,
        "missing_scene_id": missing_scene_id,
        "include_unusable": include_unusable,
        "dataset_counts": {
            str(key): int(value)
            for key, value in output["dataset"].value_counts(dropna=False).sort_index().items()
        },
        "folder_counts": {
            str(key): int(value)
            for key, value in output["folder"].value_counts(dropna=False).sort_index().items()
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        "--missing-csv",
        dest="source_csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=(
            "Source CSV. Defaults to the curated all-case manifest and filters "
            "curation_status=missing_ipv_from_index. The legacy --missing-csv "
            "alias is still accepted."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--include-unusable", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_rerun_input(
        args.source_csv,
        args.output_csv,
        args.summary_json,
        include_unusable=args.include_unusable,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
