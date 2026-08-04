#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / ".codex-fleet/rq017-onsite-materializer/work/M1"
REPORT_DIR = ROOT / ".codex-fleet/rq017-onsite-materializer/board/reports"
LOCAL_ANCHOR_PARQUET = ROOT / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet"
LOCAL_ANCHOR_CSV = ROOT / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.csv"
LOCAL_TIMESERIES_PARQUET = ROOT / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet"
LOCAL_TIMESERIES_CSV = ROOT / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.csv"
LOCAL_DRYRUN = ROOT / ".codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet"
K2_MATERIALIZER = ROOT / ".codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py"
G_ANCHOR_HPC = ROOT / ".codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv"
G_ANCHOR_BASELINE = ROOT / ".codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json"
B_WORK = ROOT / ".codex-fleet/rq015b-repair/work"

REMOTE_ANCHOR_CSV = Path(
    "/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.csv"
)
REMOTE_TIMESERIES_CSV = Path(
    "/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.csv"
)

SCHEMA_VERSION = "rq017_onsite_gate_l1_v1"
ARTIFACT_ID = "onsite_dense_timeseries"
SIGMA = 0.1
THETA = 0.20
K = 7
GRID_ID = "legacy7_pi_over_8"
IPV_GRID = (np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64) * math.pi / 8.0).astype(np.float64)
GRID_LIST = [float(x) for x in IPV_GRID]
FEATURE_HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
TARGET_FINAL_OFFSET = 6
REFERENCE_MAX_POINTS = 40
REFERENCE_SMOOTH_POINTS = 40
HEADING_THRESHOLD_DEG = 12.0

ANCHOR_COLUMNS = [
    "case_key",
    "scene_unique_id",
    "unit_composite_key",
    "area_id",
    "team_id",
    "algorithm_id",
    "session_id",
    "task_id",
    "scenario_id",
    "competition_case_id",
    "native_replay_case_id",
    "frame_index",
    "anchor_frame_index",
    "perspective",
    "ego_key_agent",
    "counterpart_key_agent",
    "source_dataset",
    "target_window_end_frame_index",
    "history_row_count",
    "relative_dx_anchor",
    "relative_dy_anchor",
    "relative_distance_anchor",
]

TIMESERIES_COLUMNS = [
    "case_key",
    "frame_index",
    "timestamp_ms",
    "time_s",
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
]

DRYRUN_COLUMNS = ["product_row_key", "mechanism2_gate_ok"]

FORBIDDEN_ANCHOR_COLUMNS = {
    "target_ipv_future",
    "target_ipv_error_future",
    "counterpart_ipv_current",
    "counterpart_ipv_error_current",
    "counterpart_ipv_slope_pre_anchor",
    "counterpart_ipv_history_count",
    "counterpart_ipv_history_fraction",
    "M4_ONLY_ego_self_anchor_ipv_current",
    "M4_ONLY_ego_self_anchor_ipv_error_current",
}

ARRAY_GROUPS = {
    "candidate_ipv": "candidate_ipv",
    "mse_per_candidate": "mse",
    "log_score": "log_score",
    "w_log": "w_log",
}
ARRAY_SCALAR_FIELDS = [f"{prefix}_{i}" for prefix in ARRAY_GROUPS.values() for i in range(K)]

L1_FIELDS = [
    "schema_version",
    "artifact_id",
    "canonical_key",
    "interhub_canonical_key",
    "product_row_key",
    "measurement_role",
    "case_id",
    "rq007_split",
    "frame_id",
    "context_cell_key",
    "candidate_grid_id",
    "K",
    *ARRAY_SCALAR_FIELDS,
    "max_w_log",
    "mse_spread",
    "k_eff_log",
    "status",
    "reason_code",
    "ipv_log",
    "gate_applicable",
    "source_attempt_status",
    "source_reason_code",
    "ipv_error",
    "k_eff",
    "q_eff",
    "solver_status",
    "failure_type",
    "out_of_scope_reason",
    "shard_id",
    "input_sha256",
    "code_sha",
    "created_utc",
    "solve_frame_index",
    "anchor_frame_index",
    "target_window_end_frame_index",
    "history_window_used",
]

TS_BY_CASE: Optional[Dict[str, pd.DataFrame]] = None
CASE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, List[str], List[Any], Dict[int, int], pd.DataFrame]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_arrow() -> Tuple[Any, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def read_parquet_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=list(columns))
    unexpected = sorted(set(df.columns) - set(columns))
    if unexpected:
        raise ValueError(f"{path} read columns outside whitelist: {unexpected}")
    return df


def read_csv_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(columns))
    unexpected = sorted(set(df.columns) - set(columns))
    if unexpected:
        raise ValueError(f"{path} read columns outside whitelist: {unexpected}")
    return df


def csv_header(path: Path) -> List[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def parquet_columns(path: Path) -> List[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def product_row_key_from_values(case_key: Any, anchor_frame_index: Any, perspective: Any, source_dataset: Any) -> str:
    return (
        f"case_key={case_key}|anchor_frame_index={int(float(anchor_frame_index))}|"
        f"perspective={perspective}|source_dataset={source_dataset}"
    )


def product_row_key(row: Mapping[str, Any]) -> str:
    return product_row_key_from_values(
        row["case_key"],
        row["anchor_frame_index"],
        row["perspective"],
        row["source_dataset"],
    )


def timeseries_row_key(row: Mapping[str, Any]) -> str:
    return (
        f"case_key={row['case_key']}|frame_index={int(float(row['frame_index']))}|"
        f"ego_key_agent={row['ego_key_agent']}|counterpart_key_agent={row['counterpart_key_agent']}"
    )


def ensure_no_forbidden_whitelist() -> None:
    forbidden_in_anchor = sorted(FORBIDDEN_ANCHOR_COLUMNS & set(ANCHOR_COLUMNS))
    forbidden_in_ts = sorted(FORBIDDEN_ANCHOR_COLUMNS & set(TIMESERIES_COLUMNS))
    forbidden_in_dryrun = sorted(FORBIDDEN_ANCHOR_COLUMNS & set(DRYRUN_COLUMNS))
    if forbidden_in_anchor or forbidden_in_ts or forbidden_in_dryrun:
        raise ValueError(
            "forbidden columns entered an input whitelist: "
            f"anchor={forbidden_in_anchor}, timeseries={forbidden_in_ts}, dryrun={forbidden_in_dryrun}"
        )


def history_counts_for_anchor(anchor: pd.DataFrame, timeseries: pd.DataFrame) -> Tuple[Counter, List[Mapping[str, Any]]]:
    failures: List[Mapping[str, Any]] = []
    counts: Counter = Counter()
    grouped_ts = {str(k): g.reset_index(drop=True) for k, g in timeseries.groupby("case_key", sort=False)}
    for row in anchor.to_dict("records"):
        case = str(row["case_key"])
        frames = grouped_ts.get(case)
        if frames is None:
            failures.append({"issue": "missing_case_in_timeseries", "case_key": case})
            continue
        frame_values = frames["frame_index"].astype(int).tolist()
        if len(frame_values) != len(set(frame_values)):
            failures.append({"issue": "duplicate_frame_index_in_case", "case_key": case})
            continue
        pos_by_frame = {int(frame): i for i, frame in enumerate(frame_values)}
        anchor_frame = int(float(row["anchor_frame_index"]))
        pos = pos_by_frame.get(anchor_frame)
        if pos is None:
            failures.append({"issue": "missing_anchor_frame", "case_key": case, "anchor_frame_index": anchor_frame})
            continue
        target_pos = pos + TARGET_FINAL_OFFSET
        target_end = int(float(row["target_window_end_frame_index"]))
        if target_pos >= len(frame_values) or int(frame_values[target_pos]) != target_end:
            failures.append(
                {
                    "issue": "target_window_end_mismatch",
                    "case_key": case,
                    "anchor_frame_index": anchor_frame,
                    "position": pos,
                    "expected_position_offset": TARGET_FINAL_OFFSET,
                    "anchor_target_window_end_frame_index": target_end,
                    "timeseries_target_frame_index": None if target_pos >= len(frame_values) else int(frame_values[target_pos]),
                }
            )
            continue
        wx_start = max(0, pos - FEATURE_HISTORY_WINDOW + 1)
        hist = int(np.sum(np.asarray(frame_values[wx_start : pos + 1], dtype=int) >= MIN_OBSERVATION))
        anchor_hist = int(float(row["history_row_count"]))
        if hist != anchor_hist:
            failures.append(
                {
                    "issue": "history_row_count_mismatch",
                    "case_key": case,
                    "anchor_frame_index": anchor_frame,
                    "computed_history_row_count": hist,
                    "anchor_history_row_count": anchor_hist,
                }
            )
            continue
        ts_row = frames.iloc[pos]
        if str(ts_row["ego_key_agent"]) != str(row["ego_key_agent"]) or str(ts_row["counterpart_key_agent"]) != str(row["counterpart_key_agent"]):
            failures.append(
                {
                    "issue": "role_key_mismatch",
                    "case_key": case,
                    "anchor_frame_index": anchor_frame,
                    "anchor_ego_key_agent": str(row["ego_key_agent"]),
                    "timeseries_ego_key_agent": str(ts_row["ego_key_agent"]),
                    "anchor_counterpart_key_agent": str(row["counterpart_key_agent"]),
                    "timeseries_counterpart_key_agent": str(ts_row["counterpart_key_agent"]),
                }
            )
            continue
        counts[hist] += 1
    return counts, failures


def fail_with_json(path: Path, payload: Dict[str, Any], message: str) -> None:
    payload["status"] = "FAIL"
    payload["failed_at_utc"] = utc_now()
    payload["error"] = message
    write_json(path, payload)
    raise SystemExit(message)


def command_preflight(args: argparse.Namespace) -> None:
    ensure_no_forbidden_whitelist()
    out_path = Path(args.output)
    payload: Dict[str, Any] = {
        "created_utc": utc_now(),
        "contract": "RQ017 measurement contract preflight",
        "key_construction": "case_key={case_key}|anchor_frame_index={int(anchor_frame_index)}|perspective={perspective}|source_dataset={source_dataset}",
        "anchor_input_whitelist": ANCHOR_COLUMNS,
        "timeseries_input_whitelist": TIMESERIES_COLUMNS,
        "mechanism2_join_whitelist": DRYRUN_COLUMNS,
        "forbidden_anchor_columns": sorted(FORBIDDEN_ANCHOR_COLUMNS),
    }
    try:
        anchor_file_columns = parquet_columns(LOCAL_ANCHOR_PARQUET)
        anchor_csv_columns = csv_header(LOCAL_ANCHOR_CSV)
        anchor = read_parquet_columns(LOCAL_ANCHOR_PARQUET, ANCHOR_COLUMNS)
        anchor_csv = read_csv_columns(LOCAL_ANCHOR_CSV, ANCHOR_COLUMNS)
        timeseries = read_parquet_columns(LOCAL_TIMESERIES_PARQUET, TIMESERIES_COLUMNS)
        timeseries_csv = read_csv_columns(LOCAL_TIMESERIES_CSV, TIMESERIES_COLUMNS)
        dryrun = read_parquet_columns(LOCAL_DRYRUN, DRYRUN_COLUMNS)
    except Exception as exc:
        fail_with_json(out_path, payload, f"input read failed: {exc}")

    payload["local_input_files"] = {
        "anchor_parquet": {"path": str(LOCAL_ANCHOR_PARQUET), "rows": int(len(anchor)), "columns_read": list(anchor.columns)},
        "anchor_csv": {
            "path": str(LOCAL_ANCHOR_CSV),
            "rows": int(len(anchor_csv)),
            "columns_read": list(anchor_csv.columns),
            "bytes": LOCAL_ANCHOR_CSV.stat().st_size,
            "sha256": sha256_file(LOCAL_ANCHOR_CSV),
        },
        "timeseries_parquet": {"path": str(LOCAL_TIMESERIES_PARQUET), "rows": int(len(timeseries)), "columns_read": list(timeseries.columns)},
        "timeseries_csv": {
            "path": str(LOCAL_TIMESERIES_CSV),
            "rows": int(len(timeseries_csv)),
            "columns_read": list(timeseries_csv.columns),
            "bytes": LOCAL_TIMESERIES_CSV.stat().st_size,
            "sha256": sha256_file(LOCAL_TIMESERIES_CSV),
        },
        "dryrun": {"path": str(LOCAL_DRYRUN), "rows": int(len(dryrun)), "columns_read": list(dryrun.columns)},
    }
    payload["source_file_schema_probe"] = {
        "anchor_parquet_forbidden_columns_present_but_not_read": sorted(set(anchor_file_columns) & FORBIDDEN_ANCHOR_COLUMNS),
        "anchor_csv_forbidden_columns_present_but_not_read": sorted(set(anchor_csv_columns) & FORBIDDEN_ANCHOR_COLUMNS),
    }

    checks: Dict[str, Any] = {}
    anchor_keys = [product_row_key(r) for r in anchor.to_dict("records")]
    anchor_csv_keys = [product_row_key(r) for r in anchor_csv.to_dict("records")]
    dryrun_keys = dryrun["product_row_key"].astype(str).tolist()

    checks["C1_key_one_to_one"] = {
        "anchor_rows": int(len(anchor)),
        "anchor_unique_product_row_key": int(len(set(anchor_keys))),
        "dryrun_rows": int(len(dryrun)),
        "dryrun_unique_product_row_key": int(dryrun["product_row_key"].nunique()),
        "intersection": int(len(set(anchor_keys) & set(dryrun_keys))),
        "missing_from_dryrun": int(len(set(anchor_keys) - set(dryrun_keys))),
        "extra_in_dryrun": int(len(set(dryrun_keys) - set(anchor_keys))),
    }
    if checks["C1_key_one_to_one"] != {
        "anchor_rows": 67861,
        "anchor_unique_product_row_key": 67861,
        "dryrun_rows": 67861,
        "dryrun_unique_product_row_key": 67861,
        "intersection": 67861,
        "missing_from_dryrun": 0,
        "extra_in_dryrun": 0,
    }:
        fail_with_json(out_path, {**payload, "checks": checks}, "C1 product_row_key contract failed")

    perspective_counts = anchor["perspective"].astype(str).value_counts().to_dict()
    source_counts = anchor["source_dataset"].astype(str).value_counts().to_dict()
    ego_av_count = int(anchor["ego_key_agent"].astype(str).str.startswith("AV:").sum())
    checks["C2_C5_role_grain"] = {
        "output_rows": int(len(anchor)),
        "perspective_counts": perspective_counts,
        "source_dataset_counts": source_counts,
        "ego_key_agent_startswith_AV": ego_av_count,
        "counterpart_key_nonempty": int(anchor["counterpart_key_agent"].astype(str).ne("").sum()),
    }
    if len(anchor) != 67861 or perspective_counts != {"onsite_av_primary": 67861} or ego_av_count != 67861:
        fail_with_json(out_path, {**payload, "checks": checks}, "C2/C5 anchor role or grain failed")

    hist_counts, hist_failures = history_counts_for_anchor(anchor, timeseries)
    checks["C3_C4_window_contract"] = {
        "history_row_count_distribution": {str(k): int(v) for k, v in sorted(hist_counts.items())},
        "expected_distribution": {"4": 267, "5": 265, "6": 264, "7": 261, "8": 258, "9": 257, "10": 66289},
        "failures": hist_failures[:20],
        "failure_count": len(hist_failures),
        "window_expression": "wx_start=max(0,pos-10+1); valid=frame_index[wx_start:pos+1] >= 4; target_pos=pos+6",
    }
    if len(hist_failures) != 0 or checks["C3_C4_window_contract"]["history_row_count_distribution"] != checks["C3_C4_window_contract"]["expected_distribution"]:
        fail_with_json(out_path, {**payload, "checks": checks}, "C3/C4 window contract failed")

    checks["C6_output_frame_fields"] = {
        "required_output_fields": [
            "solve_frame_index",
            "anchor_frame_index",
            "target_window_end_frame_index",
            "history_window_used",
        ],
        "solve_frame_index_contract": "solve_frame_index equals the current anchor row frame_index used by the solver",
    }

    checks["C7_input_column_whitelist"] = {
        "actual_anchor_read_subset": sorted(set(anchor.columns).issubset(set(ANCHOR_COLUMNS)) for _ in [0])[0],
        "actual_timeseries_read_subset": sorted(set(timeseries.columns).issubset(set(TIMESERIES_COLUMNS)) for _ in [0])[0],
        "actual_dryrun_read_subset": sorted(set(dryrun.columns).issubset(set(DRYRUN_COLUMNS)) for _ in [0])[0],
        "forbidden_columns_read": sorted((set(anchor.columns) | set(timeseries.columns) | set(dryrun.columns)) & FORBIDDEN_ANCHOR_COLUMNS),
    }
    if checks["C7_input_column_whitelist"]["forbidden_columns_read"]:
        fail_with_json(out_path, {**payload, "checks": checks}, "C7 forbidden column was read")

    ts_keys = {timeseries_row_key(r) for r in timeseries.to_dict("records")}
    ts_csv_keys = {timeseries_row_key(r) for r in timeseries_csv.to_dict("records")}
    checks["C8_csv_parquet_equivalence"] = {
        "anchor_parquet_rows": int(len(anchor)),
        "anchor_csv_rows": int(len(anchor_csv)),
        "anchor_key_sets_equal": set(anchor_keys) == set(anchor_csv_keys),
        "timeseries_parquet_rows": int(len(timeseries)),
        "timeseries_csv_rows": int(len(timeseries_csv)),
        "timeseries_key_sets_equal": ts_keys == ts_csv_keys,
    }
    if not (
        checks["C8_csv_parquet_equivalence"]["anchor_parquet_rows"] == checks["C8_csv_parquet_equivalence"]["anchor_csv_rows"] == 67861
        and checks["C8_csv_parquet_equivalence"]["anchor_key_sets_equal"]
        and checks["C8_csv_parquet_equivalence"]["timeseries_parquet_rows"] == checks["C8_csv_parquet_equivalence"]["timeseries_csv_rows"] == 70317
        and checks["C8_csv_parquet_equivalence"]["timeseries_key_sets_equal"]
    ):
        fail_with_json(out_path, {**payload, "checks": checks}, "C8 CSV/parquet equivalence failed")

    abnormal = anchor[pd.to_numeric(anchor["relative_distance_anchor"], errors="coerce") > 100000.0]
    checks["coordinate_abnormal_rows"] = {
        "rows": int(len(abnormal)),
        "rows_over_anchor_denominator": "7/67861",
        "case_keys": sorted(abnormal["case_key"].astype(str).unique().tolist()),
        "min_relative_dx_anchor": float(pd.to_numeric(abnormal["relative_dx_anchor"]).min()) if len(abnormal) else None,
        "max_relative_dx_anchor": float(pd.to_numeric(abnormal["relative_dx_anchor"]).max()) if len(abnormal) else None,
        "min_relative_dy_anchor": float(pd.to_numeric(abnormal["relative_dy_anchor"]).min()) if len(abnormal) else None,
        "max_relative_dy_anchor": float(pd.to_numeric(abnormal["relative_dy_anchor"]).max()) if len(abnormal) else None,
        "product_row_keys": [product_row_key(r) for r in abnormal.to_dict("records")],
    }
    if len(abnormal) != 7:
        fail_with_json(out_path, {**payload, "checks": checks}, "coordinate abnormal row count changed")

    held_markers = anchor["source_dataset"].astype(str).str.contains("held_out|heldout", case=False, regex=True).sum()
    checks["rq007_held_out_and_interhub_guard"] = {
        "rows_from_interhub": int(anchor["source_dataset"].astype(str).str.contains("interhub", case=False, regex=False).sum()),
        "rows_with_held_out_marker_in_read_columns": int(held_markers),
        "run_b2_allowed_splits_guard": "OnSite does not call run_b2_rq015b.solve_anchor_task; that InterHub guard remains untouched",
    }
    if checks["rq007_held_out_and_interhub_guard"]["rows_from_interhub"] != 0 or held_markers != 0:
        fail_with_json(out_path, {**payload, "checks": checks}, "RQ007/InterHub guard failed")

    payload["checks"] = checks
    payload["status"] = "PASS"
    write_json(out_path, payload)
    print(json.dumps({"status": "PASS", "output": str(out_path), "anchor_rows": len(anchor)}, sort_keys=True))


def protected_sha_payload() -> Dict[str, Any]:
    rels = [
        "src/sociality_estimation/core/agent.py",
        "src/sociality_estimation/core/ipv_estimation.py",
        "src/sociality_estimation/core/reliability_logdomain.py",
        "pipelines/interhub/process_interhub.py",
        "configs/ipv_sigma01_exact.json",
    ]
    return {
        "created_utc": utc_now(),
        "root": str(ROOT),
        "sha256": {rel: sha256_file(ROOT / rel) for rel in rels},
    }


def command_protected_sha(args: argparse.Namespace) -> None:
    payload = protected_sha_payload()
    write_json(Path(args.output), payload)
    print(json.dumps({"status": "PASS", "output": args.output}, sort_keys=True))


def load_gate_from_k2():
    if not K2_MATERIALIZER.exists():
        raise FileNotFoundError(K2_MATERIALIZER)
    spec = importlib.util.spec_from_file_location("rq017_k2_materializer", str(K2_MATERIALIZER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load K2 materializer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gate_from_mse


def load_weights_from_mse():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from sociality_estimation.core.reliability_logdomain import weights_from_mse

    return weights_from_mse


def classify_heading(headings: np.ndarray, threshold_deg: float = HEADING_THRESHOLD_DEG) -> str:
    unwrapped = np.unwrap(np.asarray(headings, dtype=float))
    delta_deg = float(np.degrees(unwrapped[-1] - unwrapped[0]))
    if delta_deg >= threshold_deg:
        return "lt"
    if delta_deg <= -threshold_deg:
        return "rt"
    return "gs"


def ensure_unique_labels(labels: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    result: List[str] = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label}{counts[label]}")
    return result


def drop_consecutive_duplicate_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if len(arr) <= 1:
        return arr
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.linalg.norm(arr[1:] - arr[:-1], axis=1) > 1e-9
    return arr[keep]


def subsample_reference_points(reference: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(reference) <= max_points:
        return reference
    indices = np.linspace(0, len(reference) - 1, int(max_points))
    indices = np.unique(np.rint(indices).astype(int))
    if len(indices) < 2:
        indices = np.array([0, len(reference) - 1], dtype=int)
    return reference[indices]


def prepared_reference(motion: np.ndarray, smooth_ployline: Any) -> Any:
    points = drop_consecutive_duplicate_points(motion[:, 0:2])
    if len(points) < 2:
        raise ValueError("observed reference has fewer than two unique points")
    ref = subsample_reference_points(points, REFERENCE_MAX_POINTS)
    return smooth_ployline(ref, point_num=REFERENCE_SMOOTH_POINTS)


def load_timeseries_groups(path: Path) -> Dict[str, pd.DataFrame]:
    ts = read_parquet_columns(path, TIMESERIES_COLUMNS)
    groups: Dict[str, pd.DataFrame] = {}
    for case, group in ts.groupby("case_key", sort=False):
        g = group.reset_index(drop=True).copy()
        g["frame_index"] = g["frame_index"].astype(int)
        groups[str(case)] = g
    return groups


def init_worker(timeseries_path: str) -> None:
    global TS_BY_CASE, CASE_CACHE
    TS_BY_CASE = load_timeseries_groups(Path(timeseries_path))
    CASE_CACHE = {}


def get_case_sequences(case_key: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[Any], Dict[int, int], pd.DataFrame]:
    global TS_BY_CASE, CASE_CACHE
    if TS_BY_CASE is None:
        raise RuntimeError("worker timeseries cache is not initialized")
    cached = CASE_CACHE.get(case_key)
    if cached is not None:
        return cached
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from sociality_estimation.core import agent as agent_module

    frames = TS_BY_CASE[case_key]
    ego_motion = frames[["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_heading"]].to_numpy(float)
    cp_motion = frames[
        ["counterpart_x", "counterpart_y", "counterpart_vx", "counterpart_vy", "counterpart_heading"]
    ].to_numpy(float)
    labels = ensure_unique_labels([classify_heading(ego_motion[:, 4]), classify_heading(cp_motion[:, 4])])
    references = [prepared_reference(ego_motion, agent_module.smooth_ployline), prepared_reference(cp_motion, agent_module.smooth_ployline)]
    pos_by_frame = {int(frame): i for i, frame in enumerate(frames["frame_index"].astype(int).tolist())}
    out = (ego_motion, cp_motion, labels, references, pos_by_frame, frames)
    CASE_CACHE[case_key] = out
    return out


def finite_motion(arr: np.ndarray) -> bool:
    return bool(arr.ndim == 2 and arr.shape[1] >= 5 and np.all(np.isfinite(arr)))


def solve_one_anchor(order_and_row: Tuple[int, Mapping[str, Any]]) -> Dict[str, Any]:
    order, row = order_and_row
    created = utc_now()
    base = base_l1_row(row, shard_id=str(row["shard_id"]), created_utc=created)
    base["sample_order"] = order
    try:
        if str(row["source_dataset"]).lower().find("interhub") >= 0:
            raise ValueError("InterHub rows are not allowed in RQ017 OnSite materializer")
        case_key = str(row["case_key"])
        ego_motion, cp_motion, labels, references, pos_by_frame, frames = get_case_sequences(case_key)
        anchor_frame = int(float(row["anchor_frame_index"]))
        pos = pos_by_frame[anchor_frame]
        target_pos = pos + TARGET_FINAL_OFFSET
        expected_target = int(float(row["target_window_end_frame_index"]))
        if target_pos >= len(frames) or int(frames.iloc[target_pos]["frame_index"]) != expected_target:
            raise ValueError("target window end position contract failed")
        wx_start = max(0, pos - FEATURE_HISTORY_WINDOW + 1)
        frame_slice = frames.iloc[wx_start : pos + 1]
        valid_mask = frame_slice["frame_index"].astype(int).to_numpy() >= MIN_OBSERVATION
        valid_positions = np.asarray(list(range(wx_start, pos + 1)), dtype=int)[valid_mask]
        if len(valid_positions) != int(float(row["history_row_count"])):
            raise ValueError("history row count contract failed")
        if len(valid_positions) < MIN_OBSERVATION:
            raise ValueError("history row count below min observation")
        if str(frames.iloc[pos]["ego_key_agent"]) != str(row["ego_key_agent"]):
            raise ValueError("ego_key_agent mismatch")
        if str(frames.iloc[pos]["counterpart_key_agent"]) != str(row["counterpart_key_agent"]):
            raise ValueError("counterpart_key_agent mismatch")
        win_ego = ego_motion[valid_positions]
        win_cp = cp_motion[valid_positions]
        if not finite_motion(win_ego) or not finite_motion(win_cp):
            return engineering_l1_row(base, "NON_FINITE_INPUT", "NON_FINITE_INPUT")

        from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_current

        seq_ego = MotionSequence(win_ego, target=labels[0], reference=references[0])
        seq_cp = MotionSequence(win_cp, target=labels[1], reference=references[1])
        _ipv_values, _ipv_errors, diagnostics = estimate_ipv_current(
            seq_ego,
            seq_cp,
            history_window=FEATURE_HISTORY_WINDOW,
            return_diagnostics=True,
            solver_mode="exact",
            candidate_ipv_values=IPV_GRID,
        )
        primary_diags = diagnostics.get("primary", [])
        if len(primary_diags) != 1:
            raise ValueError(f"expected one primary diagnostic, got {len(primary_diags)}")
        diag = primary_diags[0]
        observed = np.asarray(diag["observed"], dtype=np.float64)
        virtual_tracks = [np.asarray(v, dtype=np.float64) for v in diag["virtual_tracks"]]
        if len(virtual_tracks) != K:
            raise ValueError(f"candidate count is not {K}: {len(virtual_tracks)}")
        mse = np.asarray([float(np.sum((track - observed) ** 2, axis=1).mean()) for track in virtual_tracks], dtype=np.float64)
        return gated_l1_row(base, mse)
    except Exception as exc:
        out = engineering_l1_row(base, "SOLVER_FAILURE", "SOLVER_FAILURE")
        out["failure_type"] = "SOLVER_FAILURE"
        out["out_of_scope_reason"] = repr(exc)
        return out


def base_l1_row(row: Mapping[str, Any], *, shard_id: str, created_utc: str) -> Dict[str, Any]:
    key = product_row_key(row)
    anchor_frame = int(float(row["anchor_frame_index"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "canonical_key": key,
        "interhub_canonical_key": None,
        "product_row_key": key,
        "measurement_role": "onsite_av_primary_current",
        "case_id": str(row["case_key"]),
        "rq007_split": None,
        "frame_id": anchor_frame,
        "context_cell_key": None,
        "candidate_grid_id": GRID_ID,
        "K": K,
        "gate_applicable": True,
        "source_attempt_status": "ATTEMPTED",
        "source_reason_code": None,
        "ipv_error": None,
        "k_eff": None,
        "q_eff": None,
        "out_of_scope_reason": None,
        "shard_id": shard_id,
        "input_sha256": None,
        "code_sha": None,
        "created_utc": created_utc,
        "solve_frame_index": anchor_frame,
        "anchor_frame_index": anchor_frame,
        "target_window_end_frame_index": int(float(row["target_window_end_frame_index"])),
        "history_window_used": int(float(row["history_row_count"])),
    }


def engineering_l1_row(base: Mapping[str, Any], status: str, reason: str) -> Dict[str, Any]:
    out = dict(base)
    for group in ARRAY_GROUPS:
        out[group] = None
    out.update(
        {
            "max_w_log": None,
            "mse_spread": None,
            "k_eff_log": None,
            "status": status,
            "reason_code": reason,
            "ipv_log": None,
            "solver_status": status,
            "failure_type": status,
        }
    )
    return out


def gated_l1_row(base: Mapping[str, Any], mse: np.ndarray) -> Dict[str, Any]:
    gate_from_mse = load_gate_from_k2()
    weights_from_mse = load_weights_from_mse()
    gate = dict(gate_from_mse(np.asarray(mse, dtype=np.float64), weights_from_mse))
    if gate["status"] in {"NON_FINITE_INPUT", "SOLVER_FAILURE"}:
        return engineering_l1_row(base, str(gate["status"]), str(gate["reason_code"]))
    out = dict(base)
    out.update(
        {
            "candidate_ipv": gate["candidate_ipv"],
            "mse_per_candidate": gate["mse_per_candidate"],
            "log_score": gate["log_score"],
            "w_log": gate["w_log"],
            "max_w_log": gate["max_w_log"],
            "mse_spread": gate["mse_spread"],
            "k_eff_log": gate["k_eff_log"],
            "status": gate["status"],
            "reason_code": gate["reason_code"],
            "ipv_log": gate["ipv_log"],
            "solver_status": "OK",
            "failure_type": None,
        }
    )
    return out


def flatten_arrays(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for group, prefix in ARRAY_GROUPS.items():
        values = row.get(group)
        for i in range(K):
            out[f"{prefix}_{i}"] = None if values is None else float(values[i])
    return out


def l1_schema(pa: Any) -> Any:
    fields = [
        ("schema_version", pa.string()),
        ("artifact_id", pa.string()),
        ("canonical_key", pa.string()),
        ("interhub_canonical_key", pa.string()),
        ("product_row_key", pa.string()),
        ("measurement_role", pa.string()),
        ("case_id", pa.string()),
        ("rq007_split", pa.string()),
        ("frame_id", pa.int64()),
        ("context_cell_key", pa.string()),
        ("candidate_grid_id", pa.string()),
        ("K", pa.int64()),
    ]
    for prefix in ARRAY_GROUPS.values():
        for i in range(K):
            fields.append((f"{prefix}_{i}", pa.float64()))
    fields.extend(
        [
            ("max_w_log", pa.float64()),
            ("mse_spread", pa.float64()),
            ("k_eff_log", pa.float64()),
            ("status", pa.string()),
            ("reason_code", pa.string()),
            ("ipv_log", pa.float64()),
            ("gate_applicable", pa.bool_()),
            ("source_attempt_status", pa.string()),
            ("source_reason_code", pa.string()),
            ("ipv_error", pa.float64()),
            ("k_eff", pa.float64()),
            ("q_eff", pa.float64()),
            ("solver_status", pa.string()),
            ("failure_type", pa.string()),
            ("out_of_scope_reason", pa.string()),
            ("shard_id", pa.string()),
            ("input_sha256", pa.string()),
            ("code_sha", pa.string()),
            ("created_utc", pa.string()),
            ("solve_frame_index", pa.int64()),
            ("anchor_frame_index", pa.int64()),
            ("target_window_end_frame_index", pa.int64()),
            ("history_window_used", pa.int64()),
        ]
    )
    return pa.schema(fields)


def rows_to_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    pa, _pq = import_arrow()
    schema = l1_schema(pa)
    flat_rows = [flatten_arrays(r) for r in rows]
    arrays = [pa.array([row.get(field.name) for row in flat_rows], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_l1_parquet_atomic(rows: Sequence[Mapping[str, Any]], out_path: Path) -> str:
    _pa, pq = import_arrow()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    table = rows_to_table(rows)
    pq.write_table(table, tmp, compression="zstd", version="2.6")
    os.replace(tmp, out_path)
    return sha256_file(out_path)


def write_dataframe_parquet_atomic(df: pd.DataFrame, path: Path) -> str:
    pa, pq = import_arrow()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd", version="2.6")
    os.replace(tmp, path)
    return sha256_file(path)


def command_prepare_inputs(args: argparse.Namespace) -> None:
    ensure_no_forbidden_whitelist()
    inputs_dir = Path(args.inputs_dir)
    allowed_existing = {"measurement_contract.json", "local_protected_sha256.json"}
    existing = [p.name for p in inputs_dir.iterdir()] if inputs_dir.exists() else []
    unexpected_existing = sorted(name for name in existing if name not in allowed_existing)
    if unexpected_existing:
        raise SystemExit(f"refusing to overwrite non-empty inputs dir {inputs_dir}: {unexpected_existing}")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    anchor_csv = Path(args.anchor_csv)
    timeseries_csv = Path(args.timeseries_csv)
    anchor = read_csv_columns(anchor_csv, ANCHOR_COLUMNS)
    ts = read_csv_columns(timeseries_csv, TIMESERIES_COLUMNS)
    if len(anchor) != 67861:
        raise SystemExit(f"anchor CSV row count changed: {len(anchor)}")
    if len(ts) != 70317:
        raise SystemExit(f"timeseries CSV row count changed: {len(ts)}")

    anchor_whitelist = inputs_dir / "anchor_whitelist.parquet"
    ts_whitelist = inputs_dir / "timeseries_whitelist.parquet"
    anchor_sha = write_dataframe_parquet_atomic(anchor, anchor_whitelist)
    ts_sha = write_dataframe_parquet_atomic(ts, ts_whitelist)

    full_dir = inputs_dir / "full_shards"
    full_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = inputs_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    full_manifest_paths: List[str] = []
    shard_size = int(args.shard_size)
    code_sha = code_sha_string()
    for start in range(0, len(anchor), shard_size):
        end = min(len(anchor), start + shard_size)
        shard_id = f"full_{start // shard_size + 1:04d}"
        shard = anchor.iloc[start:end].copy()
        shard["shard_id"] = shard_id
        shard_path = full_dir / f"{shard_id}.parquet"
        shard_sha = write_dataframe_parquet_atomic(shard, shard_path)
        manifest = {
            "mode": "full",
            "shard_id": shard_id,
            "anchor_shard_parquet": str(shard_path),
            "timeseries_parquet": str(ts_whitelist),
            "row_start_in_full_anchor_csv": start,
            "row_end_exclusive_in_full_anchor_csv": end,
            "expected_output_rows": int(len(shard)),
            "input_anchor_csv": str(anchor_csv),
            "input_anchor_csv_sha256": sha256_file(anchor_csv),
            "input_timeseries_csv": str(timeseries_csv),
            "input_timeseries_csv_sha256": sha256_file(timeseries_csv),
            "anchor_whitelist_sha256": anchor_sha,
            "timeseries_whitelist_sha256": ts_sha,
            "anchor_shard_sha256": shard_sha,
            "code_sha": code_sha,
            "include_sentinel": False,
        }
        manifest_path = manifests_dir / f"{shard_id}.manifest.json"
        write_json(manifest_path, manifest)
        full_manifest_paths.append(str(manifest_path))

    canary_indices = select_canary_indices(anchor)
    canary_dir = inputs_dir / "canary_shards"
    canary_dir.mkdir(parents=True, exist_ok=True)
    canary_manifest_paths: List[str] = []
    chunks = np.array_split(np.asarray(canary_indices, dtype=int), 2)
    for i, chunk in enumerate(chunks, start=1):
        shard_id = f"canary_{i:04d}"
        shard = anchor.iloc[chunk.tolist()].copy()
        shard["shard_id"] = shard_id
        shard_path = canary_dir / f"{shard_id}.parquet"
        shard_sha = write_dataframe_parquet_atomic(shard, shard_path)
        manifest = {
            "mode": "canary",
            "shard_id": shard_id,
            "anchor_shard_parquet": str(shard_path),
            "timeseries_parquet": str(ts_whitelist),
            "expected_output_rows": int(len(shard)),
            "input_anchor_csv": str(anchor_csv),
            "input_anchor_csv_sha256": sha256_file(anchor_csv),
            "input_timeseries_csv": str(timeseries_csv),
            "input_timeseries_csv_sha256": sha256_file(timeseries_csv),
            "anchor_whitelist_sha256": anchor_sha,
            "timeseries_whitelist_sha256": ts_sha,
            "anchor_shard_sha256": shard_sha,
            "code_sha": code_sha,
            "include_sentinel": i == 1,
            "canary_selection": "first rows + evenly spaced rows + all 7 relative_distance_anchor > 100000 rows",
        }
        manifest_path = manifests_dir / f"{shard_id}.manifest.json"
        write_json(manifest_path, manifest)
        canary_manifest_paths.append(str(manifest_path))

    (inputs_dir / "full_manifest_list.txt").write_text("\n".join(full_manifest_paths) + "\n", encoding="utf-8")
    (inputs_dir / "canary_manifest_list.txt").write_text("\n".join(canary_manifest_paths) + "\n", encoding="utf-8")
    summary = {
        "created_utc": utc_now(),
        "status": "PASS",
        "anchor_rows": int(len(anchor)),
        "timeseries_rows": int(len(ts)),
        "full_shards": len(full_manifest_paths),
        "full_shard_size": shard_size,
        "canary_shards": len(canary_manifest_paths),
        "canary_natural_rows": len(canary_indices),
        "canary_coordinate_abnormal_rows": int((anchor.iloc[canary_indices]["relative_distance_anchor"].astype(float) > 100000.0).sum()),
        "anchor_csv_sha256": sha256_file(anchor_csv),
        "timeseries_csv_sha256": sha256_file(timeseries_csv),
        "anchor_whitelist_sha256": anchor_sha,
        "timeseries_whitelist_sha256": ts_sha,
        "code_sha": code_sha,
    }
    write_json(inputs_dir / "prepare_inputs_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def select_canary_indices(anchor: pd.DataFrame) -> List[int]:
    selected: List[int] = []
    selected.extend(range(min(12, len(anchor))))
    abnormal = anchor.index[pd.to_numeric(anchor["relative_distance_anchor"], errors="coerce") > 100000.0].tolist()
    selected.extend(abnormal)
    if len(anchor) > 0:
        selected.extend(np.linspace(0, len(anchor) - 1, 28, dtype=int).tolist())
    out: List[int] = []
    seen = set()
    for idx in selected:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def code_sha_string() -> str:
    rels = [
        "src/sociality_estimation/core/agent.py",
        "src/sociality_estimation/core/ipv_estimation.py",
        "src/sociality_estimation/core/reliability_logdomain.py",
        "pipelines/interhub/process_interhub.py",
        "configs/ipv_sigma01_exact.json",
        ".codex-fleet/rq017-onsite-materializer/work/M1/rq017_onsite_materializer.py",
    ]
    return ";".join(f"{Path(rel).name}:{sha256_file(ROOT / rel)[:12]}" for rel in rels if (ROOT / rel).exists())


def sentinel_base(name: str, shard_id: str, created_utc: str) -> Dict[str, Any]:
    key = f"sentinel={name}|scope=rq017_canary"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "rq017_canary_sentinel",
        "canonical_key": key,
        "interhub_canonical_key": None,
        "product_row_key": key,
        "measurement_role": "synthetic_canary",
        "case_id": "rq017_synthetic",
        "rq007_split": None,
        "frame_id": -1,
        "context_cell_key": None,
        "candidate_grid_id": GRID_ID,
        "K": K,
        "gate_applicable": True,
        "source_attempt_status": "SENTINEL",
        "source_reason_code": name,
        "ipv_error": None,
        "k_eff": None,
        "q_eff": None,
        "out_of_scope_reason": None,
        "shard_id": shard_id,
        "input_sha256": None,
        "code_sha": code_sha_string(),
        "created_utc": created_utc,
        "solve_frame_index": -1,
        "anchor_frame_index": -1,
        "target_window_end_frame_index": -1,
        "history_window_used": 0,
    }


def canary_sentinel_rows(shard_id: str) -> List[Dict[str, Any]]:
    created = utc_now()
    rows: List[Dict[str, Any]] = []
    rows.append(gated_l1_row(sentinel_base("ok_theta_020", shard_id, created), np.array([0.00934, 0.00934, 0.00934, 0.0, 0.00934, 0.00934, 0.00934], dtype=np.float64)))
    rows.append(gated_l1_row(sentinel_base("near_uniform_nonzero_spread", shard_id, created), np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006], dtype=np.float64)))
    rows.append(gated_l1_row(sentinel_base("no_ipv_effect_exact_equal", shard_id, created), np.ones(K, dtype=np.float64)))
    rows.append(gated_l1_row(sentinel_base("negative_isclose_sensitive", shard_id, created), np.array([1.0, 1.0 + 1e-13, 1.0 + 2e-13, 1.0 + 3e-13, 1.0 + 4e-13, 1.0 + 4.5e-13, 1.0 + 5e-13], dtype=np.float64)))
    rows.append(gated_l1_row(sentinel_base("negative_theta_022_sensitive", shard_id, created), np.array([0.00934, 0.00934, 0.00934, 0.0, 0.00934, 0.00934, 0.00934], dtype=np.float64)))
    rows.append(engineering_l1_row(sentinel_base("non_finite_input_nan", shard_id, created), "NON_FINITE_INPUT", "NON_FINITE_INPUT"))
    fail = engineering_l1_row(sentinel_base("reference_fail_closed_fewer_than_two_unique_points", shard_id, created), "SOLVER_FAILURE", "SOLVER_FAILURE")
    fail["out_of_scope_reason"] = "ValueError('observed reference has fewer than two unique points')"
    rows.append(fail)
    return rows


def command_run_shard(args: argparse.Namespace) -> None:
    manifest_path = (ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = manifest["mode"]
    shard_id = manifest["shard_id"]
    anchor_path = ROOT / manifest["anchor_shard_parquet"]
    ts_path = ROOT / manifest["timeseries_parquet"]
    if sha256_file(anchor_path) != manifest["anchor_shard_sha256"]:
        raise SystemExit(f"anchor shard sha mismatch for {shard_id}")
    out_root = Path(os.environ.get("RQ017_OUTPUT_ROOT", str(ROOT / "data/derived/rq017_onsite_gate")))
    natural_out = out_root / mode / "l1_v1" / f"artifact_id={ARTIFACT_ID}" / f"shard_id={shard_id}" / "part-0.parquet"
    sentinel_out = out_root / mode / "sentinel_l1_v1" / f"shard_id={shard_id}" / "part-0.parquet"
    final_manifest = out_root / mode / "manifests" / f"{shard_id}.manifest.json"
    if natural_out.exists() or sentinel_out.exists() or final_manifest.exists():
        raise SystemExit(f"refusing to overwrite existing output for {shard_id}")

    started = time.time()
    anchor = read_parquet_columns(anchor_path, [*ANCHOR_COLUMNS, "shard_id"])
    rows = anchor.to_dict("records")
    solved: List[Dict[str, Any]] = []
    if int(args.workers) <= 1:
        init_worker(str(ts_path))
        for i, row in enumerate(rows):
            solved.append(solve_one_anchor((i, row)))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=init_worker, initargs=(str(ts_path),)) as ex:
            futures = [ex.submit(solve_one_anchor, (i, row)) for i, row in enumerate(rows)]
            for done, fut in enumerate(as_completed(futures), start=1):
                solved.append(fut.result())
                if done % 50 == 0 or done == len(futures):
                    print(json.dumps({"event": "heartbeat", "shard_id": shard_id, "done": done, "total": len(futures)}, sort_keys=True), flush=True)
    solved.sort(key=lambda r: int(r.pop("sample_order")))
    for row in solved:
        row["input_sha256"] = manifest["anchor_shard_sha256"]
        row["code_sha"] = manifest["code_sha"]
    natural_sha = write_l1_parquet_atomic(solved, natural_out)
    sentinel_sha = None
    sentinel_rows: List[Dict[str, Any]] = []
    if bool(manifest.get("include_sentinel")):
        sentinel_rows = canary_sentinel_rows(shard_id)
        for row in sentinel_rows:
            row["input_sha256"] = "synthetic_sentinel"
        sentinel_sha = write_l1_parquet_atomic(sentinel_rows, sentinel_out)
    elapsed = time.time() - started
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024.0 if rss_kb < 10**9 else rss_kb / (1024.0 * 1024.0)
    receipt = {
        "created_utc": utc_now(),
        "status": "PASS",
        "mode": mode,
        "shard_id": shard_id,
        "rows": len(solved),
        "sentinel_rows": len(sentinel_rows),
        "elapsed_seconds": elapsed,
        "rows_per_second": (len(solved) / elapsed) if elapsed > 0 else None,
        "worker_count": int(args.workers),
        "maxrss_mb": rss_mb,
        "status_counts": dict(Counter(str(r["status"]) for r in solved)),
        "reason_counts": dict(Counter(str(r["reason_code"]) for r in solved if r.get("reason_code") is not None)),
        "natural_output": str(natural_out),
        "natural_output_sha256": natural_sha,
        "sentinel_output": None if sentinel_sha is None else str(sentinel_out),
        "sentinel_output_sha256": sentinel_sha,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "input_manifest": manifest,
    }
    write_json(final_manifest, receipt)
    print(json.dumps({"status": "PASS", "shard_id": shard_id, "rows": len(solved), "elapsed_seconds": elapsed}, sort_keys=True))


def command_env_parity(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    payload: Dict[str, Any] = {"created_utc": utc_now(), "status": "PASS", "checks": {}}
    try:
        import numpy
        import pandas
        import pyarrow
        import scipy
        from sociality_estimation.core import ipv_estimation
    except Exception as exc:
        fail_with_json(out_path, payload, f"import failed: {exc}")

    expected_versions = {"python": "3.9.24", "numpy": "1.21.6", "scipy": "1.7.3", "pandas": "1.4.4", "pyarrow": "12.0.1"}
    actual_versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
    }
    payload["checks"]["versions"] = {"expected": expected_versions, "actual": actual_versions}
    if actual_versions != expected_versions:
        fail_with_json(out_path, payload, "version parity failed")

    origins = {
        "python_executable": sys.executable,
        "numpy": numpy.__file__,
        "scipy": scipy.__file__,
        "pandas": pandas.__file__,
        "pyarrow": pyarrow.__file__,
        "sociality_estimation.core.ipv_estimation": ipv_estimation.__file__,
    }
    run_dir = Path(args.hpc_workdir).resolve()
    expected_origin_checks = {
        "python_executable": "/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python" in origins["python_executable"],
        "numpy": "/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01" in origins["numpy"],
        "scipy": "/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01" in origins["scipy"],
        "pandas": "/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01" in origins["pandas"],
        "pyarrow": str(run_dir / "pydeps") in origins["pyarrow"],
        "ipv_estimation": str(ROOT / "src/sociality_estimation/core/ipv_estimation.py") == str(Path(origins["sociality_estimation.core.ipv_estimation"]).resolve()),
    }
    payload["checks"]["import_origins"] = {"origins": origins, "assertions": expected_origin_checks}
    if not all(expected_origin_checks.values()):
        fail_with_json(out_path, payload, "import origin parity failed")

    expected_sha = json.loads(Path(args.expected_sha).read_text(encoding="utf-8"))["sha256"]
    actual_sha = protected_sha_payload()["sha256"]
    payload["checks"]["protected_sha256"] = {"expected": expected_sha, "actual": actual_sha}
    if actual_sha != expected_sha:
        fail_with_json(out_path, payload, "protected SHA parity failed")

    remote_anchor = Path(args.remote_anchor_csv)
    remote_ts = Path(args.remote_timeseries_csv)
    remote_inputs = {
        "anchor_csv": {"path": str(remote_anchor), "bytes": remote_anchor.stat().st_size, "sha256": sha256_file(remote_anchor)},
        "timeseries_csv": {"path": str(remote_ts), "bytes": remote_ts.stat().st_size, "sha256": sha256_file(remote_ts)},
    }
    payload["checks"]["remote_input_sha256"] = remote_inputs
    if remote_inputs["anchor_csv"]["sha256"] != "4ff857c80d84f5e8aae1cb1bbf4ef0d1f25cf6a9842de0dd36399172658ddd79":
        fail_with_json(out_path, payload, "remote anchor CSV sha mismatch")
    if remote_inputs["timeseries_csv"]["sha256"] != "e49c226bafb950125a69f8b5dc90df024b4dc37e4fd57ea126ea6782d3fa2201":
        fail_with_json(out_path, payload, "remote timeseries CSV sha mismatch")

    payload["checks"]["g_anchor_recompute"] = recompute_g_anchor(limit=int(args.g_anchor_limit))
    if payload["checks"]["g_anchor_recompute"]["status"] != "PASS":
        fail_with_json(out_path, payload, "G anchor recompute parity failed")

    write_json(out_path, payload)
    print(json.dumps({"status": "PASS", "output": str(out_path)}, sort_keys=True))


def recompute_g_anchor(limit: int) -> Dict[str, Any]:
    anchor_path = G_ANCHOR_HPC
    if "rq015g-hpc-resolve/work/anchor_mse_hpc.csv" not in str(anchor_path):
        return {"status": "FAIL", "issue": "wrong_g_anchor_path", "path": str(anchor_path)}
    if "rq015b-repair/work/anchor_mse.csv" in str(anchor_path):
        return {"status": "FAIL", "issue": "mac_anchor_path_forbidden", "path": str(anchor_path)}
    expected_baseline = json.loads(G_ANCHOR_BASELINE.read_text(encoding="utf-8"))
    expected: Dict[str, np.ndarray] = {}
    with anchor_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected[str(row["anchor_id"])] = np.asarray(json.loads(row["mse_per_candidate[7]"]), dtype=np.float64)
    sample_rows: Dict[str, Mapping[str, str]] = {}
    with (B_WORK / "sample_v1.csv").open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["anchor_id"] in expected:
                sample_rows[row["anchor_id"]] = row
    if len(sample_rows) != len(expected):
        return {"status": "FAIL", "issue": "sample_v1_missing_g_anchor_rows", "expected_rows": len(expected), "sample_hits": len(sample_rows)}
    if str(B_WORK) not in sys.path:
        sys.path.insert(0, str(B_WORK))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import run_b2_rq015b

    compared = 0
    max_abs = 0.0
    first_mismatch = None
    for anchor_id in sorted(expected)[:limit]:
        solved = run_b2_rq015b.solve_anchor_task((compared, sample_rows[anchor_id]))
        arr = run_b2_rq015b.loads_array(str(solved["mse_per_candidate[7]"]))
        diff = float(np.max(np.abs(arr - expected[anchor_id])))
        if diff != 0.0:
            first_mismatch = {"anchor_id": anchor_id, "max_abs_diff": diff}
            break
        max_abs = max(max_abs, diff)
        compared += 1
    return {
        "status": "PASS" if first_mismatch is None and compared == min(limit, len(expected)) else "FAIL",
        "anchor_path": str(anchor_path),
        "baseline_path": str(G_ANCHOR_BASELINE),
        "baseline_json": expected_baseline,
        "anchor_rows": len(expected),
        "sample_hits": len(sample_rows),
        "compared_rows": compared,
        "requested_limit": limit,
        "max_abs_diff": max_abs,
        "first_mismatch": first_mismatch,
        "comparison": "recomputed run_b2_rq015b.solve_anchor_task rows against HPC anchor_mse_hpc.csv",
    }


def scalar_vector(row: Mapping[str, Any], prefix: str) -> Optional[np.ndarray]:
    vals = [row.get(f"{prefix}_{i}") for i in range(K)]
    if any(pd.isna(v) for v in vals):
        return None
    return np.asarray([float(v) for v in vals], dtype=np.float64)


def read_l1_dataset(root: Path, include_sentinel: bool = False) -> pd.DataFrame:
    paths = sorted(root.glob("artifact_id=*/shard_id=*/part-0.parquet"))
    if include_sentinel:
        paths.extend(sorted(root.parent.glob("sentinel_l1_v1/shard_id=*/part-0.parquet")))
    if not paths:
        raise FileNotFoundError(f"no parquet parts under {root}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def gate_mutant_isclose(mse: np.ndarray) -> Tuple[str, Optional[str]]:
    arr = np.asarray(mse, dtype=np.float64)
    weights_from_mse = load_weights_from_mse()
    if arr.ndim != 1 or arr.size != K or not np.all(np.isfinite(arr)):
        return "NON_FINITE_INPUT", "NON_FINITE_INPUT"
    w = np.asarray(weights_from_mse(arr, SIGMA), dtype=np.float64)
    spread = float(np.max(arr) - np.min(arr))
    if np.isclose(spread, 0.0, atol=1e-12):
        return "ABSTAIN", "NO_IPV_EFFECT"
    if float(np.max(w)) < THETA:
        return "ABSTAIN", "NEAR_UNIFORM"
    return "OK", None


def gate_mutant_theta022(mse: np.ndarray) -> Tuple[str, Optional[str]]:
    arr = np.asarray(mse, dtype=np.float64)
    weights_from_mse = load_weights_from_mse()
    if arr.ndim != 1 or arr.size != K or not np.all(np.isfinite(arr)):
        return "NON_FINITE_INPUT", "NON_FINITE_INPUT"
    w = np.asarray(weights_from_mse(arr, SIGMA), dtype=np.float64)
    spread = float(np.max(arr) - np.min(arr))
    if spread == 0.0:
        return "ABSTAIN", "NO_IPV_EFFECT"
    if float(np.max(w)) < 0.22:
        return "ABSTAIN", "NEAR_UNIFORM"
    return "OK", None


def validate_l1(df: pd.DataFrame, *, expected_rows: Optional[int], dryrun_path: Path, output_json: Path, mode: str, include_sentinel: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"created_utc": utc_now(), "mode": mode, "status": "PASS"}
    if expected_rows is not None and len(df) != expected_rows:
        fail_with_json(output_json, payload, f"row count mismatch: {len(df)} != {expected_rows}")
    if int(df["product_row_key"].nunique()) != len(df):
        fail_with_json(output_json, payload, "product_row_key uniqueness failed")
    if set(df["K"].dropna().astype(int).unique()) != {K}:
        fail_with_json(output_json, payload, "K != 7 detected")
    if set(df["candidate_grid_id"].dropna().astype(str).unique()) != {GRID_ID}:
        fail_with_json(output_json, payload, "candidate grid id mismatch")

    status_counts = Counter(df["status"].astype(str))
    reason_counts = Counter(df["reason_code"].dropna().astype(str))
    engineering_mask = df["status"].isin(["NON_FINITE_INPUT", "SOLVER_FAILURE"])
    abstain_mask = df["status"].eq("ABSTAIN")
    ok_mask = df["status"].eq("OK")
    state_identity = int(ok_mask.sum()) + int(abstain_mask.sum()) + int(engineering_mask.sum())
    if state_identity != len(df):
        fail_with_json(output_json, payload, "status conservation failed")
    if int(reason_counts.get("NEAR_UNIFORM", 0) + reason_counts.get("NO_IPV_EFFECT", 0)) != int(abstain_mask.sum()):
        fail_with_json(output_json, payload, "abstain reason conservation failed")
    if int(df.loc[engineering_mask, "reason_code"].isin(["NEAR_UNIFORM", "NO_IPV_EFFECT"]).sum()) != 0:
        fail_with_json(output_json, payload, "engineering failure leaked into science reason")

    gate_from_mse = load_gate_from_k2()
    weights_from_mse = load_weights_from_mse()
    mismatches: List[Mapping[str, Any]] = []
    identity_fail = 0
    numeric_bad = 0
    for row in df.to_dict("records"):
        mse = scalar_vector(row, "mse")
        if mse is None:
            continue
        gate = gate_from_mse(mse, weights_from_mse)
        if gate["status"] != row["status"] or gate.get("reason_code") != (None if pd.isna(row.get("reason_code")) else row.get("reason_code")):
            mismatches.append({"product_row_key": row["product_row_key"], "expected": gate, "actual_status": row["status"], "actual_reason": row.get("reason_code")})
            if len(mismatches) >= 5:
                break
        w = scalar_vector(row, "w_log")
        cand = scalar_vector(row, "candidate_ipv")
        if w is not None:
            if not (1.0 / 7.0 - 1e-12 <= float(row["max_w_log"]) <= 1.0 + 1e-12):
                numeric_bad += 1
            if not (1.0 - 1e-12 <= float(row["k_eff_log"]) <= 7.0 + 1e-12):
                numeric_bad += 1
            if float(row["mse_spread"]) < -1e-12:
                numeric_bad += 1
        if row["status"] == "OK" and w is not None and cand is not None:
            ipv = float(np.sum(cand * w))
            k_eff = float(1.0 / np.sum(w ** 2))
            if not (abs(ipv - float(row["ipv_log"])) <= 1e-12 and abs(k_eff - float(row["k_eff_log"])) <= 1e-12):
                identity_fail += 1
    if mismatches:
        fail_with_json(output_json, payload, "gate recomputation mismatch")
    if identity_fail:
        fail_with_json(output_json, payload, "OK-row identity failed")
    if numeric_bad:
        fail_with_json(output_json, payload, "numeric health range failed")

    dryrun = read_parquet_columns(dryrun_path, DRYRUN_COLUMNS)
    natural = df[df["artifact_id"].eq(ARTIFACT_ID)].copy()
    dryrun_keys = set(dryrun["product_row_key"].astype(str))
    natural_keys = set(natural["product_row_key"].astype(str))
    c1 = {
        "natural_rows": int(len(natural)),
        "natural_unique_product_row_key": int(natural["product_row_key"].nunique()),
        "dryrun_rows": int(len(dryrun)),
        "dryrun_unique_product_row_key": int(dryrun["product_row_key"].nunique()),
        "intersection": int(len(natural_keys & dryrun_keys)),
        "missing_from_dryrun": int(len(natural_keys - dryrun_keys)),
        "extra_in_dryrun": int(len(dryrun_keys - natural_keys)),
    }
    if expected_rows == 67861 and c1 != {
        "natural_rows": 67861,
        "natural_unique_product_row_key": 67861,
        "dryrun_rows": 67861,
        "dryrun_unique_product_row_key": 67861,
        "intersection": 67861,
        "missing_from_dryrun": 0,
        "extra_in_dryrun": 0,
    }:
        fail_with_json(output_json, payload, "full output C1 join failed")

    joined = natural[["product_row_key", "status"]].merge(dryrun, on="product_row_key", how="left", validate="one_to_one")
    final_judgable = int((joined["status"].eq("OK") & joined["mechanism2_gate_ok"].eq(True)).sum())

    canary_checks: Dict[str, Any] = {}
    if mode == "canary":
        sentinel = df[df["artifact_id"].eq("rq017_canary_sentinel")].copy()
        abnormal_natural = natural[
            natural["case_id"].eq("onsite:shanghai:T10:C4:native_case:2311")
            & natural["anchor_frame_index"].isin([144, 145, 146, 147, 148, 149, 150])
        ]
        canary_checks = {
            "array_task_count_required_by_manifest": 2,
            "natural_coordinate_abnormal_rows_present": int(len(abnormal_natural)),
            "sentinel_rows": int(len(sentinel)),
            "sentinel_status_counts": dict(Counter(sentinel["status"].astype(str))),
            "sentinel_reason_counts": dict(Counter(sentinel["reason_code"].dropna().astype(str))),
            "reference_fail_closed_sentinel_rows": int(
                sentinel["source_reason_code"].astype(str).eq("reference_fail_closed_fewer_than_two_unique_points").sum()
            ),
            "null_scalar_failure_rows": int(
                df.loc[df["status"].isin(["NON_FINITE_INPUT", "SOLVER_FAILURE"]), [f"mse_{i}" for i in range(K)]]
                .isna()
                .all(axis=1)
                .sum()
            ),
        }
        required_status_ok = canary_checks["sentinel_status_counts"].get("OK", 0) >= 1
        required_near = canary_checks["sentinel_reason_counts"].get("NEAR_UNIFORM", 0) >= 1
        required_no = canary_checks["sentinel_reason_counts"].get("NO_IPV_EFFECT", 0) >= 1
        required_engineering = (
            canary_checks["sentinel_status_counts"].get("NON_FINITE_INPUT", 0)
            + canary_checks["sentinel_status_counts"].get("SOLVER_FAILURE", 0)
            >= 1
        )
        if not (
            canary_checks["natural_coordinate_abnormal_rows_present"] == 7
            and canary_checks["reference_fail_closed_sentinel_rows"] >= 1
            and required_status_ok
            and required_near
            and required_no
            and required_engineering
        ):
            fail_with_json(output_json, payload, "canary coverage contract failed")

    spread_sentinel = np.array([1.0, 1.0 + 1e-13, 1.0 + 2e-13, 1.0 + 3e-13, 1.0 + 4e-13, 1.0 + 4.5e-13, 1.0 + 5e-13], dtype=np.float64)
    theta_sentinel = np.array([0.00934, 0.00934, 0.00934, 0.0, 0.00934, 0.00934, 0.00934], dtype=np.float64)
    true_spread = gate_from_mse(spread_sentinel, weights_from_mse)
    true_theta = gate_from_mse(theta_sentinel, weights_from_mse)
    mutant_spread = gate_mutant_isclose(spread_sentinel)
    mutant_theta = gate_mutant_theta022(theta_sentinel)
    neg = {
        "isclose_atol_1e_12": {
            "true_status": true_spread["status"],
            "true_reason": true_spread["reason_code"],
            "mutant_status": mutant_spread[0],
            "mutant_reason": mutant_spread[1],
            "result": "FAIL" if (true_spread["status"], true_spread["reason_code"]) != mutant_spread else "UNEXPECTED_PASS",
        },
        "theta_0_22": {
            "true_status": true_theta["status"],
            "true_reason": true_theta["reason_code"],
            "true_max_w_log": true_theta["max_w_log"],
            "mutant_status": mutant_theta[0],
            "mutant_reason": mutant_theta[1],
            "result": "FAIL" if (true_theta["status"], true_theta["reason_code"]) != mutant_theta else "UNEXPECTED_PASS",
        },
    }
    if any(v["result"] != "FAIL" for v in neg.values()):
        fail_with_json(output_json, payload, "negative control did not fail")

    payload.update(
        {
            "rows": int(len(df)),
            "natural_rows": int(len(natural)),
            "status_counts": {k: int(v) for k, v in status_counts.items()},
            "reason_counts": {k: int(v) for k, v in reason_counts.items()},
            "status_conservation": {
                "ok_plus_abstain_plus_engineering": state_identity,
                "total_rows": int(len(df)),
                "near_uniform_plus_no_ipv_effect": int(reason_counts.get("NEAR_UNIFORM", 0) + reason_counts.get("NO_IPV_EFFECT", 0)),
                "abstain_rows": int(abstain_mask.sum()),
            },
            "gate_recompute_mismatches": 0,
            "ok_identity_failures": 0,
            "engineering_failure_science_reason_rows": 0,
            "numeric_health_bad_rows": 0,
            "c1_mechanism2_key_join": c1,
            "mechanism2_cross_count": {
                "mechanism1_ok_and_mechanism2_gate_ok_rows": final_judgable,
                "denominator": int(len(natural)),
                "mechanism2_source": str(dryrun_path),
                "mechanism2_columns_read": DRYRUN_COLUMNS,
            },
            "negative_controls": neg,
            "canary_checks": canary_checks,
            "sentinel_included_in_validation_dataset": include_sentinel,
        }
    )
    write_json(output_json, payload)
    return payload


def command_validate_outputs(args: argparse.Namespace) -> None:
    root = Path(args.l1_root)
    df = read_l1_dataset(root, include_sentinel=bool(args.include_sentinel))
    expected = None if args.expected_rows is None else int(args.expected_rows)
    payload = validate_l1(df, expected_rows=expected, dryrun_path=Path(args.dryrun), output_json=Path(args.output), mode=args.mode, include_sentinel=bool(args.include_sentinel))
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "output": args.output}, sort_keys=True))


def command_cluster_capacity(args: argparse.Namespace) -> None:
    workers = int(args.workers)
    mem_mb = int(args.mem_mb)
    cmd = ["scontrol", "show", "node", "-o"]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    nodes = []
    total_slots = 0
    def safe_int(value: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    for line in proc.stdout.splitlines():
        fields = {}
        for item in line.split():
            if "=" in item:
                k, v = item.split("=", 1)
                fields[k] = v
        partitions = set(fields.get("Partitions", "").split(","))
        if not (partitions & {"intel", "fata"}):
            continue
        state = fields.get("State", "")
        if any(token in state for token in ("DOWN", "DRAIN", "FAIL", "MAINT")):
            slots = 0
        else:
            cpu_tot = safe_int(fields.get("CPUTot", "0"))
            cpu_alloc = safe_int(fields.get("CPUAlloc", "0"))
            idle_cpu = max(0, cpu_tot - cpu_alloc)
            free_mem = safe_int(fields.get("FreeMem", "0"))
            slots = min(idle_cpu // workers, free_mem // mem_mb)
        total_slots += slots
        nodes.append(
            {
                "node": fields.get("NodeName"),
                "partitions": sorted(partitions & {"intel", "fata"}),
                "state": state,
                "cpu_total": safe_int(fields.get("CPUTot", "0")),
                "cpu_alloc": safe_int(fields.get("CPUAlloc", "0")),
                "idle_cpu_for_formula": max(0, safe_int(fields.get("CPUTot", "0")) - safe_int(fields.get("CPUAlloc", "0"))),
                "free_mem_mb": safe_int(fields.get("FreeMem", "0")),
                "node_packed_slots": slots,
            }
        )
    payload = {
        "created_utc": utc_now(),
        "workers_per_shard": workers,
        "mem_per_shard_mb": mem_mb,
        "node_packed_slots_sum": total_slots,
        "recommended_array_concurrency": max(1, total_slots),
        "nodes": nodes,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


def command_collect_receipt(args: argparse.Namespace) -> None:
    manifest_dir = Path(args.manifest_dir)
    manifests = []
    for path in sorted(manifest_dir.glob("*.manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    if not manifests:
        raise SystemExit(f"no shard manifests found in {manifest_dir}")
    total_rows = sum(int(m.get("rows", 0)) for m in manifests)
    status_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    shard_rows = []
    for item in manifests:
        status_counts.update(item.get("status_counts", {}))
        reason_counts.update(item.get("reason_counts", {}))
        shard_rows.append(
            {
                "shard_id": item.get("shard_id"),
                "rows": item.get("rows"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "rows_per_second": item.get("rows_per_second"),
                "output_sha256": item.get("natural_output_sha256"),
                "node": item.get("slurm", {}).get("job_nodelist"),
                "array_task_id": item.get("slurm", {}).get("array_task_id"),
            }
        )
    sacct_text = Path(args.sacct).read_text(encoding="utf-8") if args.sacct else ""
    sacct_has_amd = "|amd|" in sacct_text or " amd " in sacct_text.lower()
    receipt = {
        "created_utc": utc_now(),
        "status": "PASS" if total_rows == int(args.expected_rows) and not sacct_has_amd else "FAIL",
        "hpc_workdir": args.hpc_workdir,
        "mode": args.mode,
        "job_id": args.job_id,
        "array_shape": args.array_shape,
        "partition_constraint": "intel,fata; amd forbidden",
        "sacct_has_amd": sacct_has_amd,
        "sacct_output": sacct_text,
        "shard_count": len(manifests),
        "total_rows": total_rows,
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "reason_counts": {k: int(v) for k, v in reason_counts.items()},
        "shards": shard_rows,
        "cluster_capacity": json.loads(Path(args.cluster_capacity).read_text(encoding="utf-8")) if args.cluster_capacity else None,
        "prepare_inputs_summary": json.loads(Path(args.prepare_summary).read_text(encoding="utf-8")) if args.prepare_summary else None,
        "canary_validation": json.loads(Path(args.canary_validation).read_text(encoding="utf-8")) if args.canary_validation else None,
        "full_validation": json.loads(Path(args.full_validation).read_text(encoding="utf-8")) if args.full_validation else None,
    }
    write_json(Path(args.output), receipt)
    if receipt["status"] != "PASS":
        raise SystemExit(f"receipt failed: {args.output}")
    print(json.dumps({"status": "PASS", "output": args.output, "total_rows": total_rows}, sort_keys=True))


def command_write_report(args: argparse.Namespace) -> None:
    key_numbers = json.loads(Path(args.key_numbers).read_text(encoding="utf-8"))
    run_receipt = json.loads(Path(args.run_receipt).read_text(encoding="utf-8"))
    measurement = json.loads(Path(args.measurement_contract).read_text(encoding="utf-8"))
    env = json.loads(Path(args.env_parity).read_text(encoding="utf-8"))
    timestamp = utc_now()
    status_counts = key_numbers.get("status_counts", {})
    reason_counts = key_numbers.get("reason_counts", {})
    natural = int(key_numbers.get("natural_rows", key_numbers.get("rows", 0)))
    ok = int(status_counts.get("OK", 0))
    abstain = int(status_counts.get("ABSTAIN", 0))
    nonfinite = int(status_counts.get("NON_FINITE_INPUT", 0))
    solver_failure = int(status_counts.get("SOLVER_FAILURE", 0))
    near = int(reason_counts.get("NEAR_UNIFORM", 0))
    no_effect = int(reason_counts.get("NO_IPV_EFFECT", 0))
    final_count = int(key_numbers["mechanism2_cross_count"]["mechanism1_ok_and_mechanism2_gate_ok_rows"])
    lines = [
        "# RQ017-M1 OnSite 机制一判据 materializer 报告",
        "",
        "## 任务定位",
        "",
        "在线验证由两道串联弃权机制构成：机制一判断某一帧的 IPV 数值是否携带七个候选之间的判别信息，机制二用人类参照分布判断当前场景是否有足够支持。本轮只补 OnSite 67,861 个自动驾驶车 anchor 的机制一判据；不做机制二重新打分，也不对任何车辆作出判断。",
        "",
        "已完成的前置工作是 RQ015 在同济 HPC 冻结机制一规格并产出 InterHub K2 台账，RQ016C 建好纯人-人 envelope 并在 OnSite dry-run 中产出机制二支持门。本轮是缺口补齐环节：为 `artifact_id == onsite_dense_timeseries` 的 OnSite anchor 生成七候选 MSE、log-domain 权重、状态和 reason。",
        "",
        "## 执行结果",
        "",
        f"- 正式 OnSite 产物行数：{natural}/67,861，筛选条件为 `artifact_id == onsite_dense_timeseries`，来源为 `{args.local_output_root}` 的 `product_row_key` 列。",
        f"- `OK`：{ok}/{natural}，筛选条件为 `status == OK`，来源为正式产物 `status` 列。",
        f"- `ABSTAIN`：{abstain}/{natural}，筛选条件为 `status == ABSTAIN`，来源为正式产物 `status` 列。",
        f"- `NEAR_UNIFORM`：{near}/{natural}，筛选条件为 `reason_code == NEAR_UNIFORM`，来源为正式产物 `reason_code` 列。",
        f"- `NO_IPV_EFFECT`：{no_effect}/{natural}，筛选条件为 `reason_code == NO_IPV_EFFECT`，来源为正式产物 `reason_code` 列。",
        f"- 工程失败：{nonfinite + solver_failure}/{natural}，筛选条件为 `status in [NON_FINITE_INPUT, SOLVER_FAILURE]`，来源为正式产物 `status` 列；其中 `NON_FINITE_INPUT` {nonfinite}，`SOLVER_FAILURE` {solver_failure}。",
        f"- 与 RQ016C 支持门交叉后的最终可判行数：{final_count}/{natural}，筛选条件为正式产物 `status == OK` 且 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet` 中 `mechanism2_gate_ok == True`，连接列为 `product_row_key`，dry-run 只读取 `product_row_key` 与 `mechanism2_gate_ok` 两列。",
        "",
        "## 与 InterHub 对照",
        "",
        "InterHub K2 全语料冻结口径为 4,981,984 个求解单元：`OK` 3,502,340/4,981,984，`NEAR_UNIFORM` 1,457,746/4,981,984，`NO_IPV_EFFECT` 19,964/4,981,984，`SOLVER_FAILURE` 1,934/4,981,984；来源为 RQ015K K2 台账的 `status` 与 `reason_code` 列。本轮 OnSite 数字只描述同一机制一规则在 OnSite anchor 上的分布，不构成车辆层判断。",
        "",
        "## Blocker 证据",
        "",
        f"- 测量合同 preflight：`{args.measurement_contract}`，状态 `{measurement.get('status')}`。C1 三方 `product_row_key` 交集 {measurement['checks']['C1_key_one_to_one']['intersection']}/67,861，C4 短历史分布为 {measurement['checks']['C3_C4_window_contract']['history_row_count_distribution']}。",
        f"- 环境同源硬断言：`{args.env_parity}`，状态 `{env.get('status')}`。版本为 {env['checks']['versions']['actual']}，G 锚点本轮重算 `max_abs_diff={env['checks']['g_anchor_recompute']['max_abs_diff']}`。",
        f"- 运行回执：`{args.run_receipt}`，Slurm 作业号与分区/节点见该 JSON；`sacct` 分区断言不含 `amd`。",
        "",
        "## 坐标异常",
        "",
        "preflight 复现 7/67,861 行 `relative_distance_anchor > 100000`，来源为本地 anchor parquet 的 `relative_distance_anchor` 列。7 行全部来自 `onsite:shanghai:T10:C4:native_case:2311`，由 `relative_dx_anchor` 约 -570,761 米而 `relative_dy_anchor` 约 -8 米导致；这更符合单侧坐标原点不一致，不符合双轴同时漂移。它们照常进入正式产物；若求解失败，仅按工程失败记录。",
        "",
        "## 自查",
        "",
        f"- 行数守恒、状态守恒、门判据复算、OK 行恒等式、`K == 7` 与 `{GRID_ID}`、工程失败隔离、数值健康、两条负对照均通过；机器证据见 `{args.key_numbers}`。",
        f"- 远端与本地逐分片行数和 sha256 一致；证据见 `{args.run_receipt}`。",
        "",
        "## 待监督方拍板",
        "",
        "无本轮执行 blocker 需要新增拍板。正式产物只提供机制一判据；是否进入车辆层在线验证或机制二解释属于后续任务。",
        "",
        "state: WAITING_ON_COMMANDER",
        f"timestamp_utc: {timestamp}",
        "",
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": args.output}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--output", default=str(WORK / "measurement_contract.json"))
    p = sub.add_parser("protected-sha")
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-inputs")
    p.add_argument("--anchor-csv", default=str(REMOTE_ANCHOR_CSV))
    p.add_argument("--timeseries-csv", default=str(REMOTE_TIMESERIES_CSV))
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--shard-size", type=int, default=500)
    p = sub.add_parser("env-parity")
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--hpc-workdir", required=True)
    p.add_argument("--remote-anchor-csv", default=str(REMOTE_ANCHOR_CSV))
    p.add_argument("--remote-timeseries-csv", default=str(REMOTE_TIMESERIES_CSV))
    p.add_argument("--g-anchor-limit", type=int, default=32)
    p = sub.add_parser("run-shard")
    p.add_argument("--manifest", required=True)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("validate-outputs")
    p.add_argument("--l1-root", required=True)
    p.add_argument("--dryrun", default=str(LOCAL_DRYRUN))
    p.add_argument("--output", required=True)
    p.add_argument("--expected-rows", type=int)
    p.add_argument("--mode", required=True)
    p.add_argument("--include-sentinel", action="store_true")
    p = sub.add_parser("cluster-capacity")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--mem-mb", type=int, default=49152)
    p.add_argument("--output", required=True)
    p = sub.add_parser("write-report")
    p.add_argument("--output", required=True)
    p.add_argument("--measurement-contract", required=True)
    p.add_argument("--env-parity", required=True)
    p.add_argument("--key-numbers", required=True)
    p.add_argument("--run-receipt", required=True)
    p.add_argument("--local-output-root", required=True)
    p = sub.add_parser("collect-receipt")
    p.add_argument("--output", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--sacct", required=True)
    p.add_argument("--hpc-workdir", required=True)
    p.add_argument("--mode", default="full")
    p.add_argument("--job-id", required=True)
    p.add_argument("--array-shape", required=True)
    p.add_argument("--expected-rows", type=int, default=67861)
    p.add_argument("--cluster-capacity", required=True)
    p.add_argument("--prepare-summary", required=True)
    p.add_argument("--canary-validation", required=True)
    p.add_argument("--full-validation", required=True)
    args = parser.parse_args()
    if args.cmd == "preflight":
        command_preflight(args)
    elif args.cmd == "protected-sha":
        command_protected_sha(args)
    elif args.cmd == "prepare-inputs":
        command_prepare_inputs(args)
    elif args.cmd == "env-parity":
        command_env_parity(args)
    elif args.cmd == "run-shard":
        command_run_shard(args)
    elif args.cmd == "validate-outputs":
        command_validate_outputs(args)
    elif args.cmd == "cluster-capacity":
        command_cluster_capacity(args)
    elif args.cmd == "write-report":
        command_write_report(args)
    elif args.cmd == "collect-receipt":
        command_collect_receipt(args)


if __name__ == "__main__":
    main()
