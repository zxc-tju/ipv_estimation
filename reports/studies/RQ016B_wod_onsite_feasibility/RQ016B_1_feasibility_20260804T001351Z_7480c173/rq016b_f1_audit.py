#!/usr/bin/env python3
"""RQ016B-F1 read-only feasibility audit.

The script inspects only existing local artifacts and writes evidence under the
task work directory. It does not run any estimator or create derived data.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / ".codex-fleet/rq016b-wod-onsite-feasibility/work/F1"

FORBIDDEN_WOD_FIELD_RE = re.compile(r"(rating|preference|human|score)", re.IGNORECASE)

BASE_NUMERIC_CONTEXT = [
    "elapsed_time_s",
    "history_row_count",
    "ego_vx_anchor",
    "ego_vy_anchor",
    "ego_heading_anchor",
    "counterpart_vx_anchor",
    "counterpart_vy_anchor",
    "counterpart_heading_anchor",
    "relative_dx_anchor",
    "relative_dy_anchor",
    "relative_distance_anchor",
    "relative_dvx_anchor",
    "relative_dvy_anchor",
    "relative_speed_anchor",
    "closing_rate_anchor",
    "heading_difference_anchor",
    "relative_distance_mean_wx",
    "relative_distance_std_wx",
    "relative_speed_mean_wx",
    "closing_rate_mean_wx",
    "closing_ttc_anchor",
    "apet_online_proxy",
]

BASE_CATEGORICAL_CONTEXT = [
    "geometry_path_category",
    "geometry_path_relation",
    "turn_pair_label",
    "agent_type_pair",
    "vehicle_type_list",
    "av_included",
    "priority_role",
]

GATE_DISTANCE_NUMERIC_12 = [
    "elapsed_time_s",
    "history_row_count",
    "relative_distance_anchor",
    "relative_speed_anchor",
    "closing_rate_anchor",
    "heading_difference_anchor",
    "relative_distance_mean_wx",
    "relative_distance_std_wx",
    "relative_speed_mean_wx",
    "closing_rate_mean_wx",
    "closing_ttc_anchor",
    "apet_online_proxy",
]

FEATURES_29 = BASE_NUMERIC_CONTEXT + BASE_CATEGORICAL_CONTEXT


PATHS = {
    "wod_projection": ROOT
    / "data/derived/wod_e2e/rq015a_full479_projected/rq010b_wod_full479_audited_candidate_ipv_projected.csv",
    "wod_receipt": ROOT / "data/derived/wod_e2e/rq015a_full479_projected/sanitization_receipt.json",
    "wod_placeholder_readme": ROOT
    / "data/derived/wod_e2e/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/README.md",
    "wod_k2": ROOT / "data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=wod_rq010b_full479_audited",
    "onsite_k2": ROOT / "data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=onsite_dense_timeseries",
    "onsite_dense": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors/onsite_ipv_timeseries.parquet",
    "onsite_anchor_single": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors/onsite_m3_av_anchors.parquet",
    "onsite_anchor_allvalid": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet",
    "onsite_timeseries_allvalid": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet",
    "onsite_ood_single": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate/onsite_m3_av_anchors_scored_ood_gate.parquet",
    "onsite_ood_allvalid": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate_multi/onsite_m3_av_anchors_multi_allvalid_scored_ood_gate.parquet",
    "onsite_mapping_doc": ROOT
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_to_m3_categorical_mapping.md",
    "onsite_channel_target": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_ipv_channel_target.csv",
    "onsite_channel_exact_hw10": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_ipv_channel_target_exact_hw10.csv",
    "onsite_channel_refref": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_ipv_channel_target_refref.csv",
    "onsite_channel_summary": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_ipv_channel_target_summary.json",
    "onsite_frame_strata": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_frame_ipv_strata.csv",
    "onsite_anchor_primitives": ROOT
    / "data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/cleanroom/anchor_primitives.parquet",
    "onsite_coverage_single": ROOT
    / "reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/onsite_ipv/coverage.json",
    "onsite_coverage_allvalid": ROOT
    / "reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/process_allvalid_processpool_amd/coverage.json",
    "onsite_provenance_allvalid": ROOT
    / "reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/process_allvalid_processpool_amd/provenance.json",
    "rq009_lodo": ROOT
    / "reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/05_evaluation/lodo_results.csv",
}


ON_SITE_RAW_TO_FEATURE = {
    "elapsed_time_s": "timestamp_ms grouped by case_key; subtract first timestamp and divide by 1000",
    "history_row_count": "count rows in the Wx window ending at anchor with frame_index >= 4",
    "ego_vx_anchor": "ego_vx",
    "ego_vy_anchor": "ego_vy",
    "ego_heading_anchor": "ego_heading",
    "counterpart_vx_anchor": "counterpart_vx",
    "counterpart_vy_anchor": "counterpart_vy",
    "counterpart_heading_anchor": "counterpart_heading",
    "relative_dx_anchor": "counterpart_x - ego_x",
    "relative_dy_anchor": "counterpart_y - ego_y",
    "relative_distance_anchor": "distance_m or sqrt(relative_dx_anchor^2 + relative_dy_anchor^2)",
    "relative_dvx_anchor": "counterpart_vx - ego_vx",
    "relative_dvy_anchor": "counterpart_vy - ego_vy",
    "relative_speed_anchor": "relative_speed_mps or hypot(relative_dvx_anchor, relative_dvy_anchor)",
    "closing_rate_anchor": "closing_rate_mps or negative distance derivative from relative pose and velocity",
    "heading_difference_anchor": "wrapped counterpart_heading - ego_heading",
    "relative_distance_mean_wx": "mean relative distance over Wx rows",
    "relative_distance_std_wx": "standard deviation of relative distance over Wx rows",
    "relative_speed_mean_wx": "mean relative speed over Wx rows",
    "closing_rate_mean_wx": "mean closing rate over Wx rows",
    "closing_ttc_anchor": "relative_distance_anchor / closing_rate_anchor when closing; otherwise capped/missing per RQ009 helper",
    "apet_online_proxy": "constant-velocity arrival-time difference from anchor pose and velocity",
    "geometry_path_category": "deterministic heading-difference heuristic documented in onsite_to_m3_categorical_mapping.md",
    "geometry_path_relation": "deterministic heading/lateral-longitudinal heuristic documented in onsite_to_m3_categorical_mapping.md",
    "turn_pair_label": "latest Wx heading delta per agent with 12 degree threshold",
    "agent_type_pair": "AV;HV from AV-perspective extractor",
    "vehicle_type_list": "['AV', 'HV'] from AV-perspective extractor",
    "av_included": "AV constant from AV-perspective extractor",
    "priority_role": "kinematic lead/yield/equal heuristic documented in onsite_to_m3_categorical_mapping.md",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def table_schema(path: Path) -> list[str]:
    if path.is_dir():
        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        return list(dataset.schema.names)
    if path.suffix == ".parquet":
        return list(pq.read_schema(path).names)
    if path.suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported table path: {path}")


def wod_guard(path: Path, columns: list[str]) -> dict[str, Any]:
    matches = [c for c in columns if FORBIDDEN_WOD_FIELD_RE.search(c)]
    return {
        "path": rel(path),
        "matched_columns": matches,
        "match_count": len(matches),
        "safe_to_read_content": len(matches) == 0,
    }


def load_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.is_dir():
        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        return dataset.to_table(columns=columns).to_pandas()
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported table path: {path}")


def inspect_table(path: Path, relevant_columns: list[str] | None = None, wod: bool = False) -> dict[str, Any]:
    columns = table_schema(path)
    guard = wod_guard(path, columns) if wod else None
    if guard and not guard["safe_to_read_content"]:
        return {
            "path": rel(path),
            "exists": path.exists(),
            "columns": columns,
            "wod_forbidden_guard": guard,
            "content_read": False,
            "row_count": None,
            "nonnull": {},
        }
    selected = list(dict.fromkeys(relevant_columns or columns))
    selected = [c for c in selected if c in columns]
    df = load_table(path, selected if selected else None)
    row_count = int(len(df))
    nonnull = {c: int(df[c].notna().sum()) for c in selected}
    unique_counts: dict[str, int] = {}
    small_values: dict[str, dict[str, int]] = {}
    for c in selected:
        if df[c].dtype == object or pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            vc = df[c].astype("string").value_counts(dropna=False)
            if len(vc) <= 20:
                small_values[c] = {str(k): int(v) for k, v in vc.items()}
        unique_counts[c] = int(df[c].nunique(dropna=True))
    return {
        "path": rel(path),
        "exists": path.exists(),
        "columns": columns,
        "wod_forbidden_guard": guard,
        "content_read": True,
        "row_count": row_count,
        "nonnull": nonnull,
        "unique_counts": unique_counts,
        "small_value_counts": small_values,
    }


def nonnull_entry(inspected: dict[str, Any], column: str) -> dict[str, Any] | None:
    if column not in inspected.get("columns", []):
        return None
    if not inspected.get("content_read"):
        return None
    return {
        "file": inspected["path"],
        "column": column,
        "nonnull": int(inspected["nonnull"].get(column, 0)),
        "total": int(inspected["row_count"]),
    }


def first_available(feature: str, inspected_tables: dict[str, dict[str, Any]], order: list[str]) -> dict[str, Any] | None:
    for name in order:
        entry = nonnull_entry(inspected_tables[name], feature)
        if entry:
            return entry
    return None


def build_feature_matrix(inspected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    wod_order = ["wod_projection", "wod_k2"]
    onsite_order = [
        "onsite_anchor_allvalid",
        "onsite_ood_allvalid",
        "onsite_anchor_single",
        "onsite_ood_single",
        "onsite_dense",
        "onsite_channel_target",
        "onsite_channel_exact_hw10",
        "onsite_channel_refref",
        "onsite_anchor_primitives",
    ]
    matrix: dict[str, list[dict[str, Any]]] = {"WOD": [], "OnSite": []}
    for feature in FEATURES_29:
        available = first_available(feature, inspected, wod_order)
        if available:
            wod_row = {
                "feature": feature,
                "kind": "numeric" if feature in BASE_NUMERIC_CONTEXT else "categorical",
                "status": "AVAILABLE",
                "evidence": available,
                "algorithm": None,
                "missing_reason": None,
            }
        else:
            wod_row = {
                "feature": feature,
                "kind": "numeric" if feature in BASE_NUMERIC_CONTEXT else "categorical",
                "status": "MISSING",
                "evidence": None,
                "algorithm": None,
                "missing_reason": "Local WOD package has only segment_key, candidate_index, ego_ipv, ego_ipv_error plus K2 bookkeeping; no paired trajectory, timing window, map/reference-line, or RQ009 context columns.",
            }
        matrix["WOD"].append(wod_row)

        onsite_available = first_available(feature, inspected, onsite_order)
        if onsite_available:
            note = None
            if feature in {
                "geometry_path_category",
                "geometry_path_relation",
                "turn_pair_label",
                "priority_role",
            }:
                note = "AVAILABLE as OnSite deterministic heuristic mapping, not the InterHub audited label source."
            onsite_row = {
                "feature": feature,
                "kind": "numeric" if feature in BASE_NUMERIC_CONTEXT else "categorical",
                "status": "AVAILABLE",
                "evidence": onsite_available,
                "algorithm": ON_SITE_RAW_TO_FEATURE.get(feature),
                "missing_reason": None,
                "note": note,
            }
        else:
            onsite_row = {
                "feature": feature,
                "kind": "numeric" if feature in BASE_NUMERIC_CONTEXT else "categorical",
                "status": "DERIVABLE",
                "evidence": None,
                "algorithm": ON_SITE_RAW_TO_FEATURE.get(feature),
                "missing_reason": "No exact precomputed column found in inspected artifacts, but raw OnSite dense fields are present for derivation."
                if feature in ON_SITE_RAW_TO_FEATURE
                else "No inspected OnSite input contract for this field.",
            }
        matrix["OnSite"].append(onsite_row)
    return matrix


def summarize_lodo() -> dict[str, Any]:
    df = pd.read_csv(PATHS["rq009_lodo"])
    heldout_sources = sorted(df["heldout_source"].dropna().unique().tolist())
    m2_90 = df[(df["tier"] == "M2") & (df["alpha_label"].astype(str) == "90")]
    return {
        "path": rel(PATHS["rq009_lodo"]),
        "heldout_sources": heldout_sources,
        "m2_90_coverage_min": float(m2_90["coverage"].min()),
        "m2_90_coverage_max": float(m2_90["coverage"].max()),
        "m2_90_rows": m2_90[
            [
                "heldout_source",
                "coverage",
                "n",
                "total_n",
                "abstained_n",
                "evaluation_rows",
                "fit_sources",
            ]
        ].to_dict(orient="records"),
        "contains_wod_artifact": "wod_rq010b_full479_audited" in heldout_sources,
        "contains_waymo_train": "waymo_train" in heldout_sources,
        "contains_onsite": any("onsite" in str(v).lower() for v in heldout_sources),
    }


def summarize_scope(inspected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    single_cov = read_json(PATHS["onsite_coverage_single"])
    allvalid_cov = read_json(PATHS["onsite_coverage_allvalid"])
    wod_projection = inspected["wod_projection"]
    wod_k2 = inspected["wod_k2"]
    onsite_dense = inspected["onsite_dense"]
    onsite_allvalid = inspected["onsite_anchor_allvalid"]
    onsite_single = inspected["onsite_anchor_single"]

    dense_df = load_table(
        PATHS["onsite_dense"],
        ["case_key", "frame_index", "timestamp_ms"],
    )
    dense_counts = {
        "physical_rows": int(len(dense_df)),
        "case_key_nunique": int(dense_df["case_key"].nunique()),
        "frame_index_nunique": int(dense_df["frame_index"].nunique()),
    }
    deltas = []
    for _, group in dense_df.sort_values(["case_key", "timestamp_ms", "frame_index"]).groupby("case_key"):
        vals = group["timestamp_ms"].dropna().astype("int64").to_numpy()
        if len(vals) > 1:
            deltas.extend([int(v) for v in pd.Series(vals).diff().dropna().tolist()])
    dense_counts["timestamp_ms_delta_counts_top10"] = dict(Counter(deltas).most_common(10))

    return {
        "wod": {
            "projection_rows": wod_projection["row_count"],
            "projection_columns": wod_projection["columns"],
            "k2_rows": wod_k2["row_count"],
            "k2_columns": wod_k2["columns"],
            "case_like_columns_present": [c for c in wod_projection["columns"] + wod_k2["columns"] if "case" in c.lower() or "scene" in c.lower() or "segment" in c.lower()],
            "candidate_index_nonnull": wod_projection["nonnull"].get("candidate_index"),
            "segment_key_nonnull": wod_projection["nonnull"].get("segment_key"),
        },
        "onsite": {
            "dense": dense_counts,
            "single_coverage": {
                "path": rel(PATHS["onsite_coverage_single"]),
                "units_requested": single_cov.get("units_requested"),
                "units_with_anchors": single_cov.get("units_with_anchors"),
                "total_av_anchors": single_cov.get("total_av_anchors"),
                "max_anchors_per_unit": single_cov.get("max_anchors_per_unit"),
                "anchor_extraction_mode": single_cov.get("anchor_extraction_mode"),
                "ipv_frame_rows": single_cov.get("ipv_frame_rows"),
                "ipv_cases_ok": single_cov.get("ipv_cases_ok"),
                "ipv_cases_failed": single_cov.get("ipv_cases_failed"),
            },
            "allvalid_coverage": {
                "path": rel(PATHS["onsite_coverage_allvalid"]),
                "units_requested": allvalid_cov.get("units_requested"),
                "units_with_anchors": allvalid_cov.get("units_with_anchors"),
                "total_av_anchors": allvalid_cov.get("total_av_anchors"),
                "valid_anchor_candidate_units": allvalid_cov.get("valid_anchor_candidate_units"),
                "valid_anchor_candidate_total": allvalid_cov.get("valid_anchor_candidate_total"),
                "valid_anchor_candidates_after_cap": allvalid_cov.get("valid_anchor_candidates_after_cap"),
                "anchor_cap_applied": allvalid_cov.get("anchor_cap_applied"),
                "anchor_cap_per_unit": allvalid_cov.get("anchor_cap_per_unit"),
                "anchors_excluded_by_cap": allvalid_cov.get("anchors_excluded_by_cap"),
                "ipv_frame_rows": allvalid_cov.get("ipv_frame_rows"),
            },
            "parquet_row_counts": {
                "single_anchor_rows": onsite_single["row_count"],
                "allvalid_anchor_rows": onsite_allvalid["row_count"],
                "anchors_excluded_by_one_per_unit_cap_reconstructed": int(onsite_allvalid["row_count"] - onsite_single["row_count"]),
            },
            "range_options": {
                "A_all_aligned_frames": {
                    "unit": "physical frame",
                    "rows": onsite_dense["row_count"],
                    "source_file": inspected["onsite_dense"]["path"],
                    "filter": "all rows",
                },
                "B_all_rq009_timing_valid_anchor_frames_current_materialized": {
                    "unit": "anchor frame",
                    "rows": onsite_allvalid["row_count"],
                    "source_file": inspected["onsite_anchor_allvalid"]["path"],
                    "filter": "all-valid materialized anchor table rows",
                },
                "B_all_rq009_timing_valid_candidates_before_failed_units": {
                    "unit": "candidate anchor",
                    "rows": allvalid_cov.get("valid_anchor_candidate_total"),
                    "source_file": rel(PATHS["onsite_coverage_allvalid"]),
                    "filter": "coverage.json valid_anchor_candidate_total",
                },
                "C_one_anchor_per_unit": {
                    "unit": "anchor frame",
                    "rows": onsite_single["row_count"],
                    "source_file": inspected["onsite_anchor_single"]["path"],
                    "filter": "default bounded closest valid anchor table rows",
                },
            },
        },
    }


def summarize_onsite_inputs(inspected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense = inspected["onsite_dense"]
    raw_fields = [
        "case_key",
        "frame_index",
        "timestamp_ms",
        "ego_key_agent",
        "counterpart_key_agent",
        "ego_x",
        "ego_y",
        "ego_vx",
        "ego_vy",
        "ego_heading",
        "counterpart_x",
        "counterpart_y",
        "counterpart_vx",
        "counterpart_vy",
        "counterpart_heading",
        "distance_m",
        "closing_rate_mps",
        "relative_speed_mps",
    ]
    true_map_terms = [
        "map",
        "lane",
        "route",
        "reference_line",
        "refline",
        "centerline",
        "roadgraph",
    ]
    true_map_cols = [c for c in dense["columns"] if any(term in c.lower() for term in true_map_terms)]
    anchor_cols = inspected["onsite_anchor_allvalid"]["columns"]
    return {
        "dense_raw_evidence": {c: nonnull_entry(dense, c) for c in raw_fields if c in dense["columns"]},
        "true_map_or_reference_columns": true_map_cols,
        "anchor_contract_columns": {
            c: nonnull_entry(inspected["onsite_anchor_allvalid"], c)
            for c in [
                "current_history_window",
                "target_history_window",
                "min_observation",
                "target_offset_rows",
                "reference_basis",
                "solver_preset",
                "candidate_grid",
                "sigma",
                "geometry_mapping_status",
            ]
            if c in anchor_cols
        },
    }


def summarize_channel_values() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key in ["onsite_channel_target", "onsite_channel_exact_hw10", "onsite_channel_refref"]:
        path = PATHS[key]
        df = pd.read_csv(path)
        selected: dict[str, Any] = {"path": rel(path), "row_count": int(len(df)), "columns": list(df.columns)}
        for c in [
            "current_history_window",
            "target_history_window",
            "min_observation",
            "target_offset_rows",
            "target_guard_rows_skipped",
            "reference_basis",
            "solver_preset",
            "candidate_grid",
        ]:
            if c in df.columns:
                selected[c] = {
                    "nonnull": int(df[c].notna().sum()),
                    "unique_values": sorted([str(v) for v in df[c].dropna().unique().tolist()]),
                }
        results[key] = selected
    return results


def summarize_gate_facts(inspected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    k2_cols = [
        "canonical_key",
        "product_row_key",
        "measurement_role",
        "case_id",
        "frame_id",
        "gate_applicable",
        "source_attempt_status",
        "out_of_scope_reason",
        "source_reason_code",
        "ipv_error",
        "q_eff",
        "candidate_grid_id",
        "K",
        "context_cell_key",
    ]
    facts: dict[str, Any] = {}
    for key in ["wod_k2", "onsite_k2"]:
        df = load_table(PATHS[key], [c for c in k2_cols if c in inspected[key]["columns"]])
        info = {"path": inspected[key]["path"], "row_count": int(len(df)), "nonnull": {}}
        for c in df.columns:
            info["nonnull"][c] = int(df[c].notna().sum())
            vc = df[c].astype("string").value_counts(dropna=False)
            if len(vc) <= 30:
                info[f"{c}_value_counts"] = {str(k): int(v) for k, v in vc.items()}
        facts[key] = info
    return facts


def negative_control(inspected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    column = "geometry_path_category"
    source = inspected["onsite_anchor_allvalid"]
    expected = int(source["row_count"])
    actual = int(source["nonnull"].get(column, 0))
    pass_check = actual == expected
    disturbed_expected = expected + 1
    disturbed_pass = actual == disturbed_expected
    return {
        "rule": f"{column} nonnull must equal all-valid anchor row count",
        "source_file": source["path"],
        "column": column,
        "actual_nonnull": actual,
        "expected_total": expected,
        "normal_result": "PASS" if pass_check else "FAIL",
        "disturbed_expected_total": disturbed_expected,
        "disturbed_result": "PASS" if disturbed_pass else "FAIL",
        "disturbed_output": f"FAIL: {column} nonnull {actual} != disturbed expected {disturbed_expected}",
    }


def gather() -> dict[str, Any]:
    relevant = list(
        dict.fromkeys(
            FEATURES_29
            + GATE_DISTANCE_NUMERIC_12
            + [
                "segment_key",
                "candidate_index",
                "ego_ipv",
                "ego_ipv_error",
                "artifact_id",
                "product_row_key",
                "measurement_role",
                "source_attempt_status",
                "out_of_scope_reason",
                "source_reason_code",
                "gate_applicable",
                "canonical_key",
                "product_row_key",
                "measurement_role",
                "case_id",
                "frame_id",
                "ipv_error",
                "q_eff",
                "candidate_grid_id",
                "K",
                "context_cell_key",
                "case_key",
                "frame_index",
                "timestamp_ms",
                "ego_key_agent",
                "counterpart_key_agent",
                "ego_x",
                "ego_y",
                "ego_vx",
                "ego_vy",
                "ego_heading",
                "counterpart_x",
                "counterpart_y",
                "counterpart_vx",
                "counterpart_vy",
                "counterpart_heading",
                "distance_m",
                "closing_rate_mps",
                "relative_speed_mps",
                "current_history_window",
                "target_history_window",
                "min_observation",
                "target_offset_rows",
                "target_guard_rows_skipped",
                "reference_basis",
                "solver_preset",
                "candidate_grid",
                "sigma",
                "geometry_mapping_status",
            ]
        )
    )
    inspect_keys = [
        "wod_projection",
        "wod_k2",
        "onsite_k2",
        "onsite_dense",
        "onsite_anchor_single",
        "onsite_anchor_allvalid",
        "onsite_timeseries_allvalid",
        "onsite_ood_single",
        "onsite_ood_allvalid",
        "onsite_channel_target",
        "onsite_channel_exact_hw10",
        "onsite_channel_refref",
        "onsite_frame_strata",
        "onsite_anchor_primitives",
    ]
    inspected = {
        key: inspect_table(PATHS[key], relevant_columns=relevant, wod=key.startswith("wod"))
        for key in inspect_keys
    }
    matrix = build_feature_matrix(inspected)
    evidence = {
        "generated_at_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "paths": {k: rel(v) for k, v in PATHS.items()},
        "inspected_tables": inspected,
        "wod_receipt": read_json(PATHS["wod_receipt"]),
        "feature_matrix": matrix,
        "gate_distance_numeric_12": {
            dataset: [row for row in rows if row["feature"] in GATE_DISTANCE_NUMERIC_12]
            for dataset, rows in matrix.items()
        },
        "scope": summarize_scope(inspected),
        "onsite_inputs": summarize_onsite_inputs(inspected),
        "onsite_channel_values": summarize_channel_values(),
        "gate_facts": summarize_gate_facts(inspected),
        "lodo": summarize_lodo(),
        "negative_control": negative_control(inspected),
    }
    return evidence


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    evidence = gather()
    (WORK / "audit_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    matrix_payload = {
        "generated_at_utc": evidence["generated_at_utc"],
        "feature_matrix": evidence["feature_matrix"],
        "gate_distance_numeric_12": evidence["gate_distance_numeric_12"],
        "negative_control": evidence["negative_control"],
        "lodo": evidence["lodo"],
        "scope": evidence["scope"],
    }
    (WORK / "feasibility_matrix.json").write_text(
        json.dumps(matrix_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
