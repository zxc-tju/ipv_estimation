#!/usr/bin/env python3
"""Phase 4A outcome-blind directional IPV predictor computation.

This worker reads only frozen configs, routing columns, trajectory logs, and
outcome-free InterHub calibration data. It writes frame-level and cell-level
directional IPV predictor artifacts for RQ003 without joining to outcomes.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(
    "/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation"
).resolve()
RUN_ID = "RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424"
WORKER_ID = "rq003_p4a_directional_ipv"
RUN_ROOT = (
    REPO_ROOT / "reports/studies/RQ003_nsfc_external_evidence" / RUN_ID
).resolve()
DERIVED_ROOT = (
    REPO_ROOT
    / "data/derived/onsite_competition/RQ003_nsfc_external_evidence"
    / RUN_ID
).resolve()
PLAN_PATH = (
    REPO_ROOT
    / "reports/studies/RQ003_nsfc_external_evidence/plans/"
    / "RQ003_plan_v2_nsfc_ipv_validation_20260620.md"
).resolve()
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"

CODE_DIR = RUN_ROOT / "code"
OUT_DIR = RUN_ROOT / "02_process/08_directional_ipv"
TABLE_DIR = RUN_ROOT / "01_results/tables"
TRACE_DIR = RUN_ROOT / "01_results/traces"
META_DIR = RUN_ROOT / "02_process/00_meta"
FRAME_DIR = DERIVED_ROOT / "frame_level"
INTERMEDIATE_DIR = DERIVED_ROOT / "intermediate"
DERIVED_MANIFEST_DIR = DERIVED_ROOT / "manifests"
MODEL_CACHE = DERIVED_ROOT / "model_cache"

RUN_MANIFEST = RUN_ROOT / "02_process/00_meta/run_manifest.json"
GATE0_STATUS = RUN_ROOT / "02_process/04_gate0_measurement/gate0_status.json"
GATE0_REVIEW_STATUS = RUN_ROOT / "02_process/05_gate0_review/gate0_review_status.json"
FREEZE_REVIEW_STATUS = RUN_ROOT / "02_process/07_freeze_review/freeze_review_status.json"
OPERATIONAL_PARAMS = RUN_ROOT / "02_process/04_gate0_measurement/operational_parameters.yaml"
ANALYSIS_FREEZE = RUN_ROOT / "02_process/06_analysis_freeze/analysis_freeze.yaml"
SUPPORT_DEFINITION = RUN_ROOT / "02_process/04_gate0_measurement/support_definition.md"
IPV_SIGN_CONTRACT = RUN_ROOT / "02_process/04_gate0_measurement/ipv_sign_contract.md"
REPLAY_MAPPING = TABLE_DIR / "replay_score_mapping.csv"
INTERHUB_TIMESERIES = (
    REPO_ROOT
    / "data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv"
)
NORM_TABLE = MODEL_CACHE / "human_conditional_norm_table.csv"
CONFORMAL_SUMMARY = MODEL_CACHE / "conformal_calibration_summary.json"

OUTCOME_DENY_EXACT = {
    "area_rank",
    "overall_rank",
    "rank",
    "safety",
    "efficiency",
    "comfort",
    "compliance",
    "coordination",
    "comprehensive",
}

MAPPING_COLUMNS = [
    "area",
    "team_code",
    "scenario",
    "scenario_family",
    "session_id",
    "replay_log_path",
    "replay_case_id",
    "replay_task_id",
    "replay_case_name",
    "replay_frames",
    "ego_frames",
    "counterpart_frames",
    "ego_counterpart_overlap_frames",
    "max_counterpart_id_frames",
    "counterpart_unique_ids",
    "replay_roles_seen",
    "join_status",
    "computable_ipv",
    "computability_rule",
    "computability_reason",
    "notes",
]

INTERHUB_COLUMNS = [
    "scene_unique_id",
    "dataset",
    "frame_index",
    "timestamp",
    "fps",
    "path_category",
    "path_relation",
    "turn_label",
    "priority_label",
    "ipv_key_agent_1",
    "ipv_key_agent_1_error",
    "key_agent_1_px",
    "key_agent_1_py",
    "key_agent_1_vx",
    "key_agent_1_vy",
    "key_agent_1_heading",
    "ipv_key_agent_2",
    "ipv_key_agent_2_error",
    "key_agent_2_px",
    "key_agent_2_py",
    "key_agent_2_vx",
    "key_agent_2_vy",
    "key_agent_2_heading",
]

NOMINAL_DT_SEC = 0.1
MAX_COUNTERPART_AGE_MS = 250
HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
MAX_WORKERS = 10
W_MIN_RAD = math.pi / 16.0
ROBUST_Z_THRESHOLD = 3.0

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str((REPO_ROOT / "src").resolve()))

from run_gate0_measurement_audit import (  # noqa: E402
    as_float,
    causal_initial_heading_reference,
    direction_costs,
    heading_and_velocity,
    lonlat_to_xy,
    progress_bin,
    quantile,
    role_context as interhub_role_context,
    stable_int,
    state_bin,
    theta_npc_bin,
)
from sociality_estimation.core.ipv_estimation import (  # noqa: E402
    MotionSequence,
    SIGN_REALTIME_CANDIDATE_IPV_VALUES,
    estimate_ipv_pair,
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:120]


def frame_path_for_route(route: Dict[str, str]) -> Path:
    return FRAME_DIR / f"{safe_slug(route['cell_key'])}__directional_ipv_frames.csv"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return quantile(vals, 0.5)


def robust_z(value: float, center: Optional[float], mad: Optional[float]) -> Optional[float]:
    if center is None or mad is None:
        return None
    if not math.isfinite(value):
        return None
    if mad <= 1e-12:
        return 0.0 if abs(value - center) <= 1e-12 else float("inf")
    return abs(value - center) / (1.4826 * mad)


class FileAccess:
    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []
        self.outcome_hits: List[str] = []

    def record(
        self,
        path: Path,
        mode: str,
        purpose: str,
        *,
        selected_columns: Optional[Sequence[str]] = None,
        outcome_hit: bool = False,
    ) -> None:
        path = path.resolve()
        if outcome_hit:
            self.outcome_hits.append(str(path))
        self.rows.append(
            {
                "path": str(path),
                "mode": mode,
                "purpose": purpose,
                "selected_columns": ",".join(selected_columns or []),
                "sha256": sha256_file(path) if path.is_file() else "",
                "outcome_hit": "YES" if outcome_hit else "NO",
            }
        )

    def read_text(self, path: Path, purpose: str) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        self.record(path, "read_text", purpose)
        return text

    def write_manifest(self, path: Path) -> None:
        lines = [
            "# Phase 4A file access manifest",
            f"generated_at: {now_iso()}",
            f"worker_id: {WORKER_ID}",
            "outcome_access: NONE" if not self.outcome_hits else "outcome_access: HIT",
            "",
            "path\tmode\toutcome_hit\tsha256\tselected_columns\tpurpose",
        ]
        for row in self.rows:
            lines.append(
                "\t".join(
                    [
                        str(row["path"]),
                        str(row["mode"]),
                        str(row["outcome_hit"]),
                        str(row["sha256"]),
                        str(row["selected_columns"]),
                        str(row["purpose"]).replace("\t", " "),
                    ]
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_csv_rows(
    path: Path,
    selected_columns: Sequence[str],
    access: FileAccess,
    purpose: str,
    limit: Optional[int] = None,
) -> Iterable[Dict[str, str]]:
    blocked = [c for c in selected_columns if c in OUTCOME_DENY_EXACT]
    if blocked:
        raise RuntimeError(f"Refusing to read outcome/rank columns from {path}: {blocked}")
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        missing = [name for name in selected_columns if name not in idx]
        if missing:
            raise KeyError(f"{path} missing selected columns: {missing}")
        kept = [(name, idx[name]) for name in selected_columns]
        access.record(path, "read_selected_columns", purpose, selected_columns=selected_columns)
        for n, row in enumerate(reader, start=1):
            if limit is not None and n > limit:
                break
            yield {name: (row[i] if i < len(row) else "") for name, i in kept}


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def load_frame_rows(path: Path, access: FileAccess) -> List[Dict[str, object]]:
    access.record(path, "read_frame_level_for_aggregation", "recompute cell table from frame trace")
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["conflict_active"] = parse_bool(row.get("conflict_active"))
            row["main_abstain"] = parse_bool(row.get("main_abstain"))
            row["main_ood_flag"] = parse_bool(row.get("main_ood_flag"))
            row["npc_main_abstain"] = parse_bool(row.get("npc_main_abstain"))
            row["npc_main_ood_flag"] = parse_bool(row.get("npc_main_ood_flag"))
            rows.append(row)
    return rows


def verify_identity(access: FileAccess) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    manifest = json.loads(access.read_text(RUN_MANIFEST, "run identity manifest"))
    if manifest.get("run_id") != RUN_ID:
        errors.append("run_manifest run_id mismatch")
    if Path(manifest.get("run_root", "")).resolve() != RUN_ROOT:
        errors.append("run_manifest run_root mismatch")
    if Path(manifest.get("derived_root", "")).resolve() != DERIVED_ROOT:
        errors.append("run_manifest derived_root mismatch")
    if manifest.get("plan_sha256") != PLAN_SHA256:
        errors.append("run_manifest plan_sha256 mismatch")
    if sha256_file(PLAN_PATH) != PLAN_SHA256:
        errors.append("plan SHA-256 mismatch")
    access.record(PLAN_PATH, "hash_only", "verify frozen plan hash")

    for required in [OPERATIONAL_PARAMS, ANALYSIS_FREEZE, SUPPORT_DEFINITION, IPV_SIGN_CONTRACT]:
        if not required.exists():
            errors.append(f"missing frozen file: {required}")
        else:
            access.record(required, "read_frozen_contract", "required frozen config present")

    gate0 = json.loads(access.read_text(GATE0_STATUS, "Gate 0 status"))
    gate0_review = json.loads(access.read_text(GATE0_REVIEW_STATUS, "Gate 0 review status"))
    freeze_review = json.loads(access.read_text(FREEZE_REVIEW_STATUS, "freeze review status"))
    if gate0.get("status") != "PASS":
        errors.append(f"Gate 0 status is {gate0.get('status')!r}")
    if gate0_review.get("review_status") != "PASS":
        errors.append(f"Gate 0 review status is {gate0_review.get('review_status')!r}")
    if freeze_review.get("review_status") != "PASS":
        errors.append(f"freeze review status is {freeze_review.get('review_status')!r}")
    if gate0.get("outcome_access") != "NONE":
        errors.append("Gate 0 did not report outcome_access NONE")
    if freeze_review.get("outcome_access") != "NONE":
        errors.append("freeze review did not report outcome_access NONE")
    return not errors, errors


def load_routing(access: FileAccess, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in selected_csv_rows(
        REPLAY_MAPPING,
        MAPPING_COLUMNS,
        access,
        "replay routing only; score/rank columns ignored",
        limit=limit,
    ):
        row["cell_key"] = f"{row['area']}|{row['team_code']}|{row['scenario']}"
        rows.append(row)
    return rows


def load_exact_norm_table(access: FileAccess) -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    access.record(NORM_TABLE, "read_frozen_norm_table", "Gate 0 InterHub exact conditional norm")
    lookup: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    with NORM_TABLE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["theta_npc_bin"],
                row["state_bin"],
                row["progress_bin"],
                row["role_context"],
            )
            lookup[key] = {
                "lookup_level": "exact_key",
                "theta_npc_bin": row["theta_npc_bin"],
                "state_bin": row["state_bin"],
                "progress_bin": row["progress_bin"],
                "role_context": row["role_context"],
                "n_train_frames": int(row["n_train_frames"]),
                "n_train_scenes": int(row["n_train_scenes"]),
                "q_low": float(row["q_low"]),
                "q50": float(row["q50"]),
                "q_high": float(row["q_high"]),
                "support_level": row["support_level"],
            }
    return lookup


def add_norm_sample(
    groups: Dict[Tuple[str, Tuple[str, ...]], List[float]],
    scenes: Dict[Tuple[str, Tuple[str, ...]], set],
    features: Dict[Tuple[str, Tuple[str, ...]], Dict[str, List[float]]],
    level: str,
    key: Tuple[str, ...],
    scene: str,
    theta_ego: float,
    distance_m: float,
    closing_speed_mps: float,
    progress_sec: float,
) -> None:
    gkey = (level, key)
    groups[gkey].append(theta_ego)
    scenes[gkey].add(scene)
    features[gkey]["distance_m"].append(distance_m)
    features[gkey]["closing_speed_mps"].append(closing_speed_mps)
    features[gkey]["progress_sec"].append(progress_sec)


def build_fallback_norms(access: FileAccess) -> Dict[str, Dict[Tuple[str, ...], Dict[str, object]]]:
    groups: Dict[Tuple[str, Tuple[str, ...]], List[float]] = defaultdict(list)
    scenes: Dict[Tuple[str, Tuple[str, ...]], set] = defaultdict(set)
    features: Dict[Tuple[str, Tuple[str, ...]], Dict[str, List[float]]] = defaultdict(
        lambda: {"distance_m": [], "closing_speed_mps": [], "progress_sec": []}
    )
    used_rows = 0
    skipped = 0

    for row in selected_csv_rows(
        INTERHUB_TIMESERIES,
        INTERHUB_COLUMNS,
        access,
        "InterHub outcome-free calibration for fallback support; PET/actual_order not read",
        limit=240000,
    ):
        scene = row["scene_unique_id"]
        if stable_int(scene) % 10 >= 8:
            continue
        fps = as_float(row["fps"])
        frame_index = as_float(row["frame_index"])
        progress = frame_index / fps if math.isfinite(frame_index) and math.isfinite(fps) and fps > 0 else 0.0
        x1, y1 = as_float(row["key_agent_1_px"]), as_float(row["key_agent_1_py"])
        x2, y2 = as_float(row["key_agent_2_px"]), as_float(row["key_agent_2_py"])
        vx1, vy1 = as_float(row["key_agent_1_vx"]), as_float(row["key_agent_1_vy"])
        vx2, vy2 = as_float(row["key_agent_2_vx"]), as_float(row["key_agent_2_vy"])
        theta1, theta2 = as_float(row["ipv_key_agent_1"]), as_float(row["ipv_key_agent_2"])
        if not all(math.isfinite(v) for v in [x1, y1, x2, y2, vx1, vy1, vx2, vy2, theta1, theta2]):
            skipped += 1
            continue
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            skipped += 1
            continue
        closing = -((dx * (vx2 - vx1) + dy * (vy2 - vy1)) / dist)
        st_bin = state_bin(dist, closing)
        pr_bin = progress_bin(progress)
        role = interhub_role_context(row)
        for theta_ego, theta_counterpart in [(theta1, theta2), (theta2, theta1)]:
            npc_bin = theta_npc_bin(theta_counterpart)
            add_norm_sample(
                groups,
                scenes,
                features,
                "exact_key_recomputed",
                (npc_bin, st_bin, pr_bin, role),
                scene,
                theta_ego,
                dist,
                closing,
                progress,
            )
            add_norm_sample(
                groups,
                scenes,
                features,
                "drop_progress",
                (npc_bin, st_bin, role),
                scene,
                theta_ego,
                dist,
                closing,
                progress,
            )
            add_norm_sample(
                groups,
                scenes,
                features,
                "drop_theta_npc",
                (st_bin, pr_bin, role),
                scene,
                theta_ego,
                dist,
                closing,
                progress,
            )
            add_norm_sample(
                groups,
                scenes,
                features,
                "state_only_marginal",
                (st_bin,),
                scene,
                theta_ego,
                dist,
                closing,
                progress,
            )
        used_rows += 1

    lookups: Dict[str, Dict[Tuple[str, ...], Dict[str, object]]] = defaultdict(dict)
    output_rows: List[Dict[str, object]] = []
    for (level, key), values in groups.items():
        scene_set = scenes[(level, key)]
        feat = features[(level, key)]
        med_dist = median(feat["distance_m"])
        med_closing = median(feat["closing_speed_mps"])
        med_progress = median(feat["progress_sec"])
        mad_dist = median([abs(v - med_dist) for v in feat["distance_m"]]) if med_dist is not None else None
        mad_closing = (
            median([abs(v - med_closing) for v in feat["closing_speed_mps"]])
            if med_closing is not None
            else None
        )
        mad_progress = (
            median([abs(v - med_progress) for v in feat["progress_sec"]])
            if med_progress is not None
            else None
        )
        entry = {
            "lookup_level": level,
            "key": key,
            "n_train_frames": len(values),
            "n_train_scenes": len(scene_set),
            "q_low": quantile(values, 0.05),
            "q50": quantile(values, 0.50),
            "q_high": quantile(values, 0.95),
            "support_level": "high" if len(values) >= 30 and len(scene_set) >= 3 else "low",
            "distance_m_median": med_dist,
            "distance_m_mad": mad_dist,
            "closing_speed_mps_median": med_closing,
            "closing_speed_mps_mad": mad_closing,
            "progress_sec_median": med_progress,
            "progress_sec_mad": mad_progress,
        }
        lookups[level][key] = entry
        output_rows.append(
            {
                "lookup_level": level,
                "key": "|".join(key),
                **{k: v for k, v in entry.items() if k != "key"},
            }
        )

    norm_summary_path = INTERMEDIATE_DIR / "fallback_norm_tables.csv"
    write_csv(
        norm_summary_path,
        sorted(output_rows, key=lambda r: (r["lookup_level"], r["key"])),
        [
            "lookup_level",
            "key",
            "n_train_frames",
            "n_train_scenes",
            "q_low",
            "q50",
            "q_high",
            "support_level",
            "distance_m_median",
            "distance_m_mad",
            "closing_speed_mps_median",
            "closing_speed_mps_mad",
            "progress_sec_median",
            "progress_sec_mad",
        ],
    )
    access.record(norm_summary_path, "write", "fallback norm table derived from InterHub calibration")

    summary_path = INTERMEDIATE_DIR / "fallback_norm_summary.json"
    summary = {
        "generated_at": now_iso(),
        "rows_used_training_split": used_rows,
        "rows_skipped": skipped,
        "levels": {
            level: {
                "groups": len(table),
                "high_support_groups": sum(1 for entry in table.values() if entry["support_level"] == "high"),
            }
            for level, table in lookups.items()
        },
        "source": str(INTERHUB_TIMESERIES),
        "split_rule": "stable_int(scene_unique_id) % 10 < 8, matching Gate 0 calibration",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    access.record(summary_path, "write", "fallback norm summary")
    return lookups


def parse_replay_log(log_path: Path, access: FileAccess) -> Dict[str, Dict[str, object]]:
    access.record(log_path, "read_trajectory_log", "NSFC replay trajectory only")
    cases: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"egos": [], "counters": defaultdict(list), "roles": Counter()}
    )
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("participantTrajectories"):
                case_id = str(obj.get("caseId", "unknown"))
                for participant in obj.get("participantTrajectories") or []:
                    role = str(participant.get("role") or "")
                    timestamp = int(float(participant.get("timestamp") or 0))
                    cases[case_id]["roles"][role] += 1
                    for item in participant.get("value") or []:
                        if item.get("longitude") is None or item.get("latitude") is None:
                            continue
                        is_perception = int(item.get("isPerception") or 0)
                        if role == "av" and is_perception == 0:
                            cases[case_id]["egos"].append((timestamp, item))
                        elif is_perception == 1 or role != "av":
                            cid = str(item.get("id") or item.get("name") or "counter")
                            cases[case_id]["counters"][cid].append((timestamp, item))
            elif obj.get("type") == "trajectory":
                value = obj.get("value") or {}
                case_id = f"raw_stream_{log_path.parent.name}"
                timestamp = int(float(value.get("timestamp") or 0))
                for item in value.get("value") or []:
                    if item.get("longitude") is None or item.get("latitude") is None:
                        continue
                    is_perception = int(item.get("isPerception") or 0)
                    if is_perception == 0:
                        cases[case_id]["egos"].append((timestamp, item))
                    else:
                        cid = str(item.get("id") or item.get("name") or "counter")
                        cases[case_id]["counters"][cid].append((timestamp, item))
    for case in cases.values():
        case["egos"].sort(key=lambda x: x[0])
        for cid in list(case["counters"].keys()):
            case["counters"][cid].sort(key=lambda x: x[0])
    return cases


def aligned_count(
    ego_records: Sequence[Tuple[int, Dict[str, object]]],
    counter_records: Sequence[Tuple[int, Dict[str, object]]],
    max_age_ms: int,
) -> Tuple[int, Optional[float], Optional[float]]:
    counter_ts = [ts for ts, _ in counter_records]
    ages: List[float] = []
    for ego_ts, _ in ego_records:
        idx = bisect.bisect_right(counter_ts, ego_ts) - 1
        if idx >= 0:
            age = ego_ts - counter_ts[idx]
            if 0 <= age <= max_age_ms:
                ages.append(float(age))
    if not ages:
        return 0, None, None
    return len(ages), mean(ages), max(ages)


def distance_between_items(ego: Dict[str, object], counter: Dict[str, object]) -> Optional[float]:
    try:
        lon0 = float(ego["longitude"])
        lat0 = float(ego["latitude"])
        ex, ey = lonlat_to_xy(float(ego["longitude"]), float(ego["latitude"]), lon0, lat0)
        cx, cy = lonlat_to_xy(float(counter["longitude"]), float(counter["latitude"]), lon0, lat0)
    except (KeyError, TypeError, ValueError):
        return None
    return math.hypot(cx - ex, cy - ey)


def choose_counterpart(case: Dict[str, object]) -> Tuple[Optional[str], Dict[str, object]]:
    ego_records = case["egos"]
    candidates = case["counters"]
    ranked: List[Tuple[int, float, str, Optional[float], Optional[float]]] = []
    for cid, records in candidates.items():
        count, mean_age, max_age = aligned_count(ego_records, records, MAX_COUNTERPART_AGE_MS)
        if count <= 0:
            continue
        counter_ts = [ts for ts, _ in records]
        distances: List[float] = []
        for ego_ts, ego in ego_records:
            idx = bisect.bisect_right(counter_ts, ego_ts) - 1
            if idx >= 0 and 0 <= ego_ts - counter_ts[idx] <= MAX_COUNTERPART_AGE_MS:
                d = distance_between_items(ego, records[idx][1])
                if d is not None:
                    distances.append(d)
        mean_dist = mean(distances)
        ranked.append((count, -(mean_dist if mean_dist is not None else 1e12), cid, mean_age, max_age))
    if not ranked:
        return None, {"reason": "no_counterpart_with_causal_overlap"}
    ranked.sort(reverse=True)
    count, neg_mean_dist, cid, mean_age, max_age = ranked[0]
    return cid, {
        "selected_counterpart_id": cid,
        "selected_counterpart_aligned_frames": count,
        "selected_counterpart_mean_age_ms": mean_age,
        "selected_counterpart_max_age_ms": max_age,
        "selected_counterpart_mean_distance_m": -neg_mean_dist if neg_mean_dist > -1e11 else None,
        "candidate_counterpart_count": len(candidates),
    }


def build_aligned_motion(
    route: Dict[str, str],
    case: Dict[str, object],
    counterpart_id: str,
) -> Tuple[List[Dict[str, object]], List[List[float]], List[List[float]]]:
    ego_records = case["egos"]
    counter_records = case["counters"][counterpart_id]
    counter_ts = [ts for ts, _ in counter_records]
    aligned: List[Tuple[int, Dict[str, object], int, Dict[str, object]]] = []
    for ego_ts, ego in ego_records:
        idx = bisect.bisect_right(counter_ts, ego_ts) - 1
        if idx >= 0:
            cts, counter = counter_records[idx]
            if 0 <= ego_ts - cts <= MAX_COUNTERPART_AGE_MS:
                aligned.append((ego_ts, ego, cts, counter))
    if not aligned:
        return [], [], []

    lon0 = float(aligned[0][1]["longitude"])
    lat0 = float(aligned[0][1]["latitude"])
    rows: List[Dict[str, object]] = []
    primary_motion: List[List[float]] = []
    counterpart_motion: List[List[float]] = []
    prev_ts: Optional[int] = None
    for local_idx, (ego_ts, ego, counter_ts_value, counter) in enumerate(aligned):
        ex, ey = lonlat_to_xy(float(ego["longitude"]), float(ego["latitude"]), lon0, lat0)
        cx, cy = lonlat_to_xy(float(counter["longitude"]), float(counter["latitude"]), lon0, lat0)
        evx, evy, eh = heading_and_velocity(ego)
        cvx, cvy, ch = heading_and_velocity(counter)
        dx, dy = cx - ex, cy - ey
        distance_m = math.hypot(dx, dy)
        closing = -((dx * (cvx - evx) + dy * (cvy - evy)) / distance_m) if distance_m > 1e-9 else 0.0
        progress_sec = local_idx * NOMINAL_DT_SEC
        st_bin = state_bin(distance_m, closing)
        pr_bin = progress_bin(progress_sec)
        active = distance_m <= 15.0 or (distance_m <= 30.0 and closing > 0.0)
        dt_from_prev = (ego_ts - prev_ts) / 1000.0 if prev_ts is not None else ""
        prev_ts = ego_ts
        primary_motion.append([ex, ey, evx, evy, eh])
        counterpart_motion.append([cx, cy, cvx, cvy, ch])
        rows.append(
            {
                "area": route["area"],
                "team_code": route["team_code"],
                "scenario": route["scenario"],
                "scenario_family": route["scenario_family"],
                "cell_key": route["cell_key"],
                "replay_case_id": route["replay_case_id"],
                "replay_case_name": route["replay_case_name"],
                "frame_index": local_idx,
                "timestamp_ms": ego_ts,
                "counterpart_timestamp_ms": counter_ts_value,
                "counterpart_age_ms": ego_ts - counter_ts_value,
                "dt_from_previous_sec": dt_from_prev,
                "ego_x": ex,
                "ego_y": ey,
                "ego_vx": evx,
                "ego_vy": evy,
                "ego_heading": eh,
                "counterpart_id": counterpart_id,
                "counterpart_x": cx,
                "counterpart_y": cy,
                "counterpart_vx": cvx,
                "counterpart_vy": cvy,
                "counterpart_heading": ch,
                "distance_m": distance_m,
                "closing_speed_mps": closing,
                "state_bin": st_bin,
                "progress_sec": progress_sec,
                "progress_bin": pr_bin,
                "role_context": route["scenario_family"],
                "role_context_source": "frozen_nsfc_scenario_family",
                "conflict_active": active,
            }
        )
    return rows, primary_motion, counterpart_motion


def exact_robust_z(frame: Dict[str, object], norm: Dict[str, object]) -> Optional[float]:
    values = [
        robust_z(
            float(frame["distance_m"]),
            finite_float(norm.get("distance_m_median")),
            finite_float(norm.get("distance_m_mad")),
        ),
        robust_z(
            float(frame["closing_speed_mps"]),
            finite_float(norm.get("closing_speed_mps_median")),
            finite_float(norm.get("closing_speed_mps_mad")),
        ),
        robust_z(
            float(frame["progress_sec"]),
            finite_float(norm.get("progress_sec_median")),
            finite_float(norm.get("progress_sec_mad")),
        ),
    ]
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if any(v == float("inf") for v in values if v is not None):
        return float("inf")
    return max(finite) if finite else None


def lookup_costs(
    theta_subject: float,
    theta_counterpart: float,
    frame: Dict[str, object],
    exact_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]],
    fallback_lookup: Dict[str, Dict[Tuple[str, ...], Dict[str, object]]],
    conformal_threshold: Optional[float],
) -> Dict[str, object]:
    npc_bin = theta_npc_bin(theta_counterpart)
    st_bin = str(frame["state_bin"])
    pr_bin = str(frame["progress_bin"])
    role = str(frame["role_context"])
    exact_key = (npc_bin, st_bin, pr_bin, role)
    exact_norm = exact_lookup.get(exact_key)

    out: Dict[str, object] = {
        "theta_npc_bin": npc_bin,
        "norm_key_exact": "|".join(exact_key),
        "main_support_level": "abstain_no_exact_interhub_support",
        "main_abstain": True,
        "main_abstain_reason": "missing_exact_key",
        "main_ood_flag": True,
        "main_robust_z": "",
        "main_q_low": "",
        "main_q50": "",
        "main_q_high": "",
        "main_norm_width": "",
        "main_d_comp": "",
        "main_d_yield": "",
        "fallback_level": "abstain",
        "fallback_support_level": "abstain",
        "fallback_q_low": "",
        "fallback_q50": "",
        "fallback_q_high": "",
        "fallback_norm_width": "",
        "fallback_d_comp": "",
        "fallback_d_yield": "",
        "fallback_conformal_nonconformity": "",
        "fallback_conformal_exceeds_90pct": "",
    }

    if exact_norm is not None:
        out["main_support_level"] = exact_norm["support_level"]
        out["main_q_low"] = exact_norm["q_low"]
        out["main_q50"] = exact_norm["q50"]
        out["main_q_high"] = exact_norm["q_high"]
        if exact_norm["support_level"] == "high":
            rz = exact_robust_z(frame, exact_norm)
            out["main_robust_z"] = rz if rz is not None else ""
            if rz is not None and math.isfinite(rz) and rz > ROBUST_Z_THRESHOLD:
                out["main_abstain_reason"] = "robust_z_ood"
            else:
                d_comp, d_yield, width = direction_costs(
                    theta_subject,
                    float(exact_norm["q_low"]),
                    float(exact_norm["q_high"]),
                    W_MIN_RAD,
                )
                out.update(
                    {
                        "main_abstain": False,
                        "main_abstain_reason": "",
                        "main_ood_flag": False,
                        "main_norm_width": width,
                        "main_d_comp": d_comp,
                        "main_d_yield": d_yield,
                    }
                )
        else:
            out["main_abstain_reason"] = "low_exact_support"

    fallback_candidates: List[Tuple[str, Tuple[str, ...], Optional[Dict[str, object]]]] = [
        ("exact_key", exact_key, exact_norm),
        ("drop_progress", (npc_bin, st_bin, role), fallback_lookup["drop_progress"].get((npc_bin, st_bin, role))),
        ("drop_theta_npc", (st_bin, pr_bin, role), fallback_lookup["drop_theta_npc"].get((st_bin, pr_bin, role))),
        ("state_only_marginal", (st_bin,), fallback_lookup["state_only_marginal"].get((st_bin,))),
    ]
    for level, key, norm in fallback_candidates:
        if norm is None or norm.get("support_level") != "high":
            continue
        d_comp, d_yield, width = direction_costs(
            theta_subject,
            float(norm["q_low"]),
            float(norm["q_high"]),
            W_MIN_RAD,
        )
        nonconf = max(d_comp, d_yield)
        out.update(
            {
                "fallback_level": level,
                "fallback_support_level": "high",
                "fallback_q_low": norm["q_low"],
                "fallback_q50": norm["q50"],
                "fallback_q_high": norm["q_high"],
                "fallback_norm_width": width,
                "fallback_d_comp": d_comp,
                "fallback_d_yield": d_yield,
                "fallback_conformal_nonconformity": nonconf,
                "fallback_conformal_exceeds_90pct": (
                    bool(nonconf > conformal_threshold)
                    if conformal_threshold is not None and math.isfinite(conformal_threshold)
                    else ""
                ),
            }
        )
        break
    return out


def attach_ipv_and_costs(
    rows: List[Dict[str, object]],
    primary_motion: Sequence[Sequence[float]],
    counterpart_motion: Sequence[Sequence[float]],
    exact_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]],
    fallback_lookup: Dict[str, Dict[Tuple[str, ...], Dict[str, object]]],
    conformal_threshold: Optional[float],
) -> None:
    primary = MotionSequence(
        np.asarray(primary_motion, dtype=float),
        target="gs",
        reference=causal_initial_heading_reference(primary_motion),
    )
    counterpart = MotionSequence(
        np.asarray(counterpart_motion, dtype=float),
        target="gs",
        reference=causal_initial_heading_reference(counterpart_motion),
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        ipv_values, ipv_errors = estimate_ipv_pair(
            primary,
            counterpart,
            history_window=HISTORY_WINDOW,
            min_observation=MIN_OBSERVATION,
            solver_preset="parallel_accurate",
            candidate_executor=executor,
            candidate_ipv_values=SIGN_REALTIME_CANDIDATE_IPV_VALUES,
        )

    for idx, row in enumerate(rows):
        row.update(
            {
                "theta_ego": "",
                "theta_npc": "",
                "ipv_ego_error": "",
                "ipv_npc_error": "",
                "uncertainty_max": "",
                "estimator_history_window": HISTORY_WINDOW,
                "estimator_min_observation": MIN_OBSERVATION,
                "estimator_solver_preset": "parallel_accurate",
                "estimator_candidate_grid": "[-3,-1,0,1,3]*pi/8",
                "estimator_executor": f"ThreadPoolExecutor(max_workers={MAX_WORKERS})",
                "reference_source": "causal_initial_heading_straight_reference",
                "main_support_level": "pre_min_observation",
                "main_abstain": True,
                "main_abstain_reason": "pre_min_observation",
                "main_ood_flag": True,
                "fallback_level": "abstain",
                "fallback_support_level": "pre_min_observation",
            }
        )
        if idx < MIN_OBSERVATION or idx >= len(ipv_values):
            continue
        theta_ego = float(ipv_values[idx, 0])
        theta_npc = float(ipv_values[idx, 1])
        if not (math.isfinite(theta_ego) and math.isfinite(theta_npc)):
            row["main_abstain_reason"] = "nonfinite_ipv"
            row["fallback_support_level"] = "nonfinite_ipv"
            continue
        row.update(
            {
                "theta_ego": theta_ego,
                "theta_npc": theta_npc,
                "ipv_ego_error": float(ipv_errors[idx, 0]),
                "ipv_npc_error": float(ipv_errors[idx, 1]),
                "uncertainty_max": max(float(ipv_errors[idx, 0]), float(ipv_errors[idx, 1])),
            }
        )
        ego_costs = lookup_costs(theta_ego, theta_npc, row, exact_lookup, fallback_lookup, conformal_threshold)
        npc_frame = dict(row)
        npc_costs = lookup_costs(theta_npc, theta_ego, npc_frame, exact_lookup, fallback_lookup, conformal_threshold)
        for key, value in ego_costs.items():
            row[key] = value
        for key, value in npc_costs.items():
            row[f"npc_{key}"] = value


def numeric(row: Dict[str, object], key: str) -> Optional[float]:
    value = row.get(key)
    if value in ("", None):
        return None
    return finite_float(value)


def aggregate_rows(route: Dict[str, str], rows: List[Dict[str, object]], status: str, failure_reason: str = "") -> Dict[str, object]:
    active = [r for r in rows if bool(r.get("conflict_active"))]
    estimable_active = [r for r in active if r.get("theta_ego") not in ("", None)]
    main_rows = [r for r in estimable_active if r.get("main_abstain") is False and numeric(r, "main_d_comp") is not None]
    fallback_rows = [
        r
        for r in estimable_active
        if r.get("fallback_support_level") == "high" and numeric(r, "fallback_d_comp") is not None
    ]
    main_both_rows = [
        r
        for r in main_rows
        if r.get("npc_main_abstain") is False and numeric(r, "npc_main_d_comp") is not None
    ]
    fallback_both_rows = [
        r
        for r in fallback_rows
        if r.get("npc_fallback_support_level") == "high" and numeric(r, "npc_fallback_d_comp") is not None
    ]

    def auc(source_rows: Sequence[Dict[str, object]], col: str) -> Optional[float]:
        vals = [numeric(r, col) for r in source_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals) * NOMINAL_DT_SEC / (len(vals) * NOMINAL_DT_SEC)

    def simultaneous(source_rows: Sequence[Dict[str, object]], prefix: str) -> Optional[float]:
        if not source_rows:
            return None
        vals = []
        for r in source_rows:
            d1 = numeric(r, f"{prefix}d_comp")
            d2 = numeric(r, f"npc_{prefix}d_comp")
            if d1 is not None and d2 is not None:
                vals.append(1.0 if d1 > 0 and d2 > 0 else 0.0)
        return mean(vals) if vals else None

    def mismatch(source_rows: Sequence[Dict[str, object]], prefix: str) -> Optional[float]:
        vals = []
        for r in source_rows:
            dc1 = numeric(r, f"{prefix}d_comp")
            dy1 = numeric(r, f"{prefix}d_yield")
            dc2 = numeric(r, f"npc_{prefix}d_comp")
            dy2 = numeric(r, f"npc_{prefix}d_yield")
            if None not in (dc1, dy1, dc2, dy2):
                vals.append(abs((dy1 - dc1) - (dy2 - dc2)))
        return mean(vals) if vals else None

    def onset_persistence(source_rows: Sequence[Dict[str, object]], prefix: str) -> Tuple[Optional[float], Optional[float], int, Optional[float]]:
        if not source_rows:
            return None, None, 0, None
        flags: List[bool] = []
        for r in source_rows:
            dc = numeric(r, f"{prefix}d_comp")
            dy = numeric(r, f"{prefix}d_yield")
            flags.append(bool((dc is not None and dc > 0) or (dy is not None and dy > 0)))
        onset_idx = next((i for i, flag in enumerate(flags) if flag), None)
        best = 0
        cur = 0
        for flag in flags:
            if flag:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        onset_norm = None
        onset_sec = None
        if onset_idx is not None:
            onset_norm = onset_idx / max(1, len(flags) - 1)
            onset_sec = onset_idx * NOMINAL_DT_SEC
        persistence_sec = best * NOMINAL_DT_SEC
        persistence_rate = best / len(flags) if flags else None
        return onset_norm, onset_sec, best, persistence_rate

    main_onset, main_onset_sec, main_persist_frames, main_persist_rate = onset_persistence(main_rows, "main_")
    fb_onset, fb_onset_sec, fb_persist_frames, fb_persist_rate = onset_persistence(fallback_rows, "fallback_")

    fallback_levels = Counter(str(r.get("fallback_level", "missing")) for r in estimable_active)
    main_abstain_reasons = Counter(str(r.get("main_abstain_reason", "")) for r in estimable_active if r.get("main_abstain"))
    frame_count = len(rows)
    active_count = len(active)
    main_high = len(main_rows)
    fallback_high = len(fallback_rows)
    cell_main_status = "HIGH_SUPPORT" if main_high > 0 else "ABSTAIN"
    if status != "processed":
        cell_main_status = "FAILED"
    elif active_count == 0:
        cell_main_status = "ABSTAIN_NO_CONFLICT_WINDOW"

    return {
        "area": route["area"],
        "team_code": route["team_code"],
        "scenario": route["scenario"],
        "scenario_family": route["scenario_family"],
        "cell_key": route["cell_key"],
        "replay_case_id": route["replay_case_id"],
        "replay_case_name": route["replay_case_name"],
        "status": status,
        "failure_reason": failure_reason,
        "D_comp_auc": auc(main_rows, "main_d_comp"),
        "D_yield_auc": auc(main_rows, "main_d_yield"),
        "simultaneous_competition": simultaneous(main_both_rows, "main_"),
        "reciprocity_mismatch": mismatch(main_both_rows, "main_"),
        "onset": main_onset,
        "onset_sec": main_onset_sec,
        "persistence": main_persist_rate,
        "persistence_frames": main_persist_frames,
        "D_comp_auc_fallback_inclusive": auc(fallback_rows, "fallback_d_comp"),
        "D_yield_auc_fallback_inclusive": auc(fallback_rows, "fallback_d_yield"),
        "simultaneous_competition_fallback_inclusive": simultaneous(fallback_both_rows, "fallback_"),
        "reciprocity_mismatch_fallback_inclusive": mismatch(fallback_both_rows, "fallback_"),
        "onset_fallback_inclusive": fb_onset,
        "onset_sec_fallback_inclusive": fb_onset_sec,
        "persistence_fallback_inclusive": fb_persist_rate,
        "persistence_frames_fallback_inclusive": fb_persist_frames,
        "main_cell_status": cell_main_status,
        "main_high_support_active_frames": main_high,
        "fallback_high_support_active_frames": fallback_high,
        "conflict_active_frames": active_count,
        "estimable_conflict_active_frames": len(estimable_active),
        "total_aligned_frames": frame_count,
        "conflict_duration_sec": active_count * NOMINAL_DT_SEC,
        "main_support_coverage_active": main_high / len(estimable_active) if estimable_active else 0.0,
        "fallback_support_coverage_active": fallback_high / len(estimable_active) if estimable_active else 0.0,
        "main_abstain_active_frames": len(estimable_active) - main_high,
        "fallback_abstain_active_frames": len(estimable_active) - fallback_high,
        "fallback_level_counts_active": json.dumps(dict(fallback_levels), sort_keys=True),
        "main_abstain_reason_counts_active": json.dumps(dict(main_abstain_reasons), sort_keys=True),
    }


FRAME_FIELDS = [
    "area",
    "team_code",
    "scenario",
    "scenario_family",
    "cell_key",
    "replay_case_id",
    "replay_case_name",
    "frame_index",
    "timestamp_ms",
    "counterpart_timestamp_ms",
    "counterpart_age_ms",
    "dt_from_previous_sec",
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "ego_heading",
    "counterpart_id",
    "counterpart_x",
    "counterpart_y",
    "counterpart_vx",
    "counterpart_vy",
    "counterpart_heading",
    "distance_m",
    "closing_speed_mps",
    "state_bin",
    "progress_sec",
    "progress_bin",
    "role_context",
    "role_context_source",
    "conflict_active",
    "theta_ego",
    "theta_npc",
    "ipv_ego_error",
    "ipv_npc_error",
    "uncertainty_max",
    "theta_npc_bin",
    "norm_key_exact",
    "main_support_level",
    "main_abstain",
    "main_abstain_reason",
    "main_ood_flag",
    "main_robust_z",
    "main_q_low",
    "main_q50",
    "main_q_high",
    "main_norm_width",
    "main_d_comp",
    "main_d_yield",
    "fallback_level",
    "fallback_support_level",
    "fallback_q_low",
    "fallback_q50",
    "fallback_q_high",
    "fallback_norm_width",
    "fallback_d_comp",
    "fallback_d_yield",
    "fallback_conformal_nonconformity",
    "fallback_conformal_exceeds_90pct",
    "npc_theta_npc_bin",
    "npc_norm_key_exact",
    "npc_main_support_level",
    "npc_main_abstain",
    "npc_main_abstain_reason",
    "npc_main_ood_flag",
    "npc_main_q_low",
    "npc_main_q50",
    "npc_main_q_high",
    "npc_main_norm_width",
    "npc_main_d_comp",
    "npc_main_d_yield",
    "npc_fallback_level",
    "npc_fallback_support_level",
    "npc_fallback_q_low",
    "npc_fallback_q50",
    "npc_fallback_q_high",
    "npc_fallback_norm_width",
    "npc_fallback_d_comp",
    "npc_fallback_d_yield",
    "estimator_history_window",
    "estimator_min_observation",
    "estimator_solver_preset",
    "estimator_candidate_grid",
    "estimator_executor",
    "reference_source",
]

CELL_FIELDS = [
    "area",
    "team_code",
    "scenario",
    "scenario_family",
    "cell_key",
    "replay_case_id",
    "replay_case_name",
    "status",
    "failure_reason",
    "D_comp_auc",
    "D_yield_auc",
    "simultaneous_competition",
    "reciprocity_mismatch",
    "onset",
    "onset_sec",
    "persistence",
    "persistence_frames",
    "D_comp_auc_fallback_inclusive",
    "D_yield_auc_fallback_inclusive",
    "simultaneous_competition_fallback_inclusive",
    "reciprocity_mismatch_fallback_inclusive",
    "onset_fallback_inclusive",
    "onset_sec_fallback_inclusive",
    "persistence_fallback_inclusive",
    "persistence_frames_fallback_inclusive",
    "main_cell_status",
    "main_high_support_active_frames",
    "fallback_high_support_active_frames",
    "conflict_active_frames",
    "estimable_conflict_active_frames",
    "total_aligned_frames",
    "conflict_duration_sec",
    "main_support_coverage_active",
    "fallback_support_coverage_active",
    "main_abstain_active_frames",
    "fallback_abstain_active_frames",
    "fallback_level_counts_active",
    "main_abstain_reason_counts_active",
]


def support_coverage_row(cell: Dict[str, object]) -> Dict[str, object]:
    return {
        "area": cell["area"],
        "team_code": cell["team_code"],
        "scenario": cell["scenario"],
        "scenario_family": cell["scenario_family"],
        "cell_key": cell["cell_key"],
        "status": cell["status"],
        "main_cell_status": cell["main_cell_status"],
        "total_aligned_frames": cell["total_aligned_frames"],
        "conflict_active_frames": cell["conflict_active_frames"],
        "estimable_conflict_active_frames": cell["estimable_conflict_active_frames"],
        "main_high_support_active_frames": cell["main_high_support_active_frames"],
        "main_abstain_active_frames": cell["main_abstain_active_frames"],
        "fallback_high_support_active_frames": cell["fallback_high_support_active_frames"],
        "fallback_abstain_active_frames": cell["fallback_abstain_active_frames"],
        "main_support_coverage_active": cell["main_support_coverage_active"],
        "fallback_support_coverage_active": cell["fallback_support_coverage_active"],
        "fallback_level_counts_active": cell["fallback_level_counts_active"],
        "main_abstain_reason_counts_active": cell["main_abstain_reason_counts_active"],
        "failure_reason": cell["failure_reason"],
    }


def process_cell(
    route: Dict[str, str],
    parsed_logs: Dict[Path, Dict[str, Dict[str, object]]],
    access: FileAccess,
    exact_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]],
    fallback_lookup: Dict[str, Dict[Tuple[str, ...], Dict[str, object]]],
    conformal_threshold: Optional[float],
) -> Tuple[Dict[str, object], Optional[Path], Dict[str, object]]:
    log_path = (REPO_ROOT / route["replay_log_path"]).resolve()
    if log_path not in parsed_logs:
        parsed_logs[log_path] = parse_replay_log(log_path, access)
    case_id = str(route["replay_case_id"])
    case = parsed_logs[log_path].get(case_id)
    if case is None:
        cell = aggregate_rows(route, [], "failed", "replay_case_id_not_found")
        return cell, None, {"log_path": str(log_path)}
    counterpart_id, selection = choose_counterpart(case)
    if counterpart_id is None:
        cell = aggregate_rows(route, [], "failed", str(selection.get("reason", "no_counterpart")))
        return cell, None, selection
    rows, primary_motion, counterpart_motion = build_aligned_motion(route, case, counterpart_id)
    if len(rows) < MIN_OBSERVATION + 2:
        cell = aggregate_rows(route, rows, "failed", "insufficient_aligned_frames")
        return cell, None, selection
    attach_ipv_and_costs(rows, primary_motion, counterpart_motion, exact_lookup, fallback_lookup, conformal_threshold)
    frame_path = frame_path_for_route(route)
    write_csv(frame_path, rows, FRAME_FIELDS)
    access.record(frame_path, "write", "frame-level directional IPV trace")
    cell = aggregate_rows(route, rows, "processed")
    cell.update(selection)
    return cell, frame_path, selection


def write_report(
    path: Path,
    cell_rows: Sequence[Dict[str, object]],
    support_rows: Sequence[Dict[str, object]],
    frame_manifest_rows: Sequence[Dict[str, object]],
    commands_run: Sequence[str],
    identity_errors: Sequence[str],
    conformal_threshold: Optional[float],
    fallback_summary: Dict[str, object],
) -> None:
    processed = sum(1 for r in cell_rows if r["status"] == "processed")
    failed = sum(1 for r in cell_rows if r["status"] == "failed")
    main_evaluable = sum(1 for r in cell_rows if r["main_cell_status"] == "HIGH_SUPPORT")
    main_abstained = sum(1 for r in cell_rows if str(r["main_cell_status"]).startswith("ABSTAIN"))
    fallback_evaluable = sum(1 for r in cell_rows if int(r.get("fallback_high_support_active_frames") or 0) > 0)
    dcomp_fb = [float(r["D_comp_auc_fallback_inclusive"]) for r in cell_rows if r.get("D_comp_auc_fallback_inclusive") not in ("", None)]
    dyield_fb = [float(r["D_yield_auc_fallback_inclusive"]) for r in cell_rows if r.get("D_yield_auc_fallback_inclusive") not in ("", None)]
    lines = [
        "# Phase 4A Directional IPV Predictor Report",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Worker: `{WORKER_ID}`",
        f"Generated: {now_iso()}",
        "",
        "## Outcome Firewall",
        "",
        "- OUTCOME_ACCESS: NONE.",
        "- `replay_score_mapping.csv` was read through an allowlist of routing columns only.",
        "- No coordination, efficiency, safety, comprehensive, comfort, compliance, score, or rank values were selected, printed, joined, modeled, or associated with IPV.",
        "- No IPV-outcome association, model fit, cross-validation result, scatter plot, or outcome-connected table was computed.",
        "",
        "## Frozen Sources",
        "",
        f"- Operational parameters: `{rel(OPERATIONAL_PARAMS)}`",
        f"- Analysis freeze: `{rel(ANALYSIS_FREEZE)}`",
        f"- Support definition: `{rel(SUPPORT_DEFINITION)}`",
        f"- IPV sign contract: `{rel(IPV_SIGN_CONTRACT)}`",
        f"- Gate 0 norm table: `{rel(NORM_TABLE)}`",
        f"- InterHub conformal threshold: `{conformal_threshold}` from `{rel(CONFORMAL_SUMMARY)}`",
        "",
        "## Method",
        "",
        "- Estimator: `estimate_ipv_pair` from `src/sociality_estimation/core/ipv_estimation.py`.",
        f"- Runtime: history window `{HISTORY_WINDOW}`, min observation `{MIN_OBSERVATION}`, solver preset `parallel_accurate`, frozen five-candidate grid `[-3,-1,0,1,3]*pi/8`, `ThreadPoolExecutor(max_workers=10)`.",
        f"- Sampling: raw replay order at nominal `{NOMINAL_DT_SEC}` seconds per frame; AV/NPC role streams aligned by causal previous/current counterpart observation with max age `{MAX_COUNTERPART_AGE_MS}` ms and no future interpolation.",
        "- NSFC role context uses frozen `scenario_family` until a richer role map is frozen.",
        "- Main support requires exact `(theta_npc_bin, state_bin, progress_bin, role_context)` InterHub high support and no robust-z OOD flag.",
        "- Fallback-inclusive sensitivity follows the frozen hierarchy: exact key, drop progress, drop theta_npc, state-only marginal, abstain.",
        "- Conflict window: `distance_m <= 15 OR (distance_m <= 30 AND closing_speed_mps > 0)`.",
        "- AUC values are time-normalized means over active conflict frames with available support at the declared level.",
        "- Secondary diagnostics: simultaneous competition is the active-frame fraction with both ego and NPC `D_comp > 0`; reciprocity mismatch is the mean absolute difference between ego and NPC directional balances `(D_yield - D_comp)`; onset is normalized first violation position inside the supported active conflict sequence; persistence is the longest consecutive violation run divided by supported active frames.",
        "",
        "## Coverage",
        "",
        f"- Mapped cells requested: {len(cell_rows)}.",
        f"- Processed: {processed}; failed: {failed}; main high-support cells: {main_evaluable}; main abstained cells: {main_abstained}; fallback-inclusive evaluable cells: {fallback_evaluable}.",
        f"- Frame-level artifacts registered: {len(frame_manifest_rows)}.",
        "- The main high-support count is expected to be limited because the frozen NSFC role context (`scenario_family`) has no exact InterHub role-context key in the Gate 0 norm table; fallback sensitivity therefore mainly resolves through state-only marginal human norms.",
        "",
        "## Predictor Distributions Without Outcomes",
        "",
        f"- Fallback `D_comp_auc`: n={len(dcomp_fb)}, min={min(dcomp_fb) if dcomp_fb else ''}, median={quantile(dcomp_fb, 0.5) if dcomp_fb else ''}, max={max(dcomp_fb) if dcomp_fb else ''}.",
        f"- Fallback `D_yield_auc`: n={len(dyield_fb)}, min={min(dyield_fb) if dyield_fb else ''}, median={quantile(dyield_fb, 0.5) if dyield_fb else ''}, max={max(dyield_fb) if dyield_fb else ''}.",
        "",
        "## Leakage Re-Check",
        "",
        "- Frame-level costs use only current/past trajectory observations at each frame, rolling IPV, rolling counterpart IPV, causal state bins, and elapsed progress.",
        "- No observed PET, realized passing order, post-hoc phase, full-window envelope, NSFC outcome, or predictor-outcome result was used.",
        "- Cell-level tables are recomputable from `DERIVED_ROOT/frame_level/*__directional_ipv_frames.csv`; manifests include path, sha256, size, and generating command.",
        "",
        "## Anomalies And Limitations",
        "",
        f"- Identity errors: {list(identity_errors) if identity_errors else 'none'}.",
        "- Main confirmatory IPV columns are blank for abstained cells; fallback-inclusive columns are sensitivity only.",
        "- Causal AV/NPC timestamp alignment is necessary because replay roles are logged on slightly different millisecond timestamps.",
        f"- Fallback norm summary: `{rel(INTERMEDIATE_DIR / 'fallback_norm_summary.json')}`.",
        "",
        "## Commands",
        "",
        *[f"- `{cmd}`" for cmd in commands_run],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_rows(paths: Sequence[Path], generating_command: str) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": rel(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "generating_command": generating_command,
                }
            )
    return rows


def route_subset(
    routes: Sequence[Dict[str, str]],
    *,
    limit: Optional[int],
    offset: int,
    shard_count: Optional[int],
    shard_index: Optional[int],
) -> List[Dict[str, str]]:
    selected = list(routes)
    if shard_count is not None:
        if shard_count <= 0:
            raise ValueError("--shard-count must be positive")
        if shard_index is None or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("--shard-index must be in [0, shard_count)")
        selected = [route for idx, route in enumerate(selected) if idx % shard_count == shard_index]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def shard_label(args: argparse.Namespace) -> str:
    if args.shard_count is not None:
        return f"shard_{args.shard_index}_of_{args.shard_count}"
    if args.offset or args.limit is not None:
        return f"offset_{args.offset}_limit_{args.limit if args.limit is not None else 'all'}"
    return "all"


def append_meta_logs(status: str, artifacts: Sequence[Path]) -> None:
    log = META_DIR / "execution_log.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            f"## {now_iso()} - {WORKER_ID}\n"
            f"- Status: {status}\n"
            "- Phase 4A directional IPV predictors generated outcome-blind under "
            "`02_process/08_directional_ipv/`, `01_results/tables/`, `01_results/traces/`, "
            "and the run derived root.\n"
        )
    index = META_DIR / "artifact_index.csv"
    exists = index.exists()
    with index.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["created_at", "worker_id", "artifact_path", "sha256", "status"])
        if not exists:
            writer.writeheader()
        for path in artifacts:
            if path.exists():
                writer.writerow(
                    {
                        "created_at": now_iso(),
                        "worker_id": WORKER_ID,
                        "artifact_path": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "status": status,
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="debug limit on mapped cells")
    parser.add_argument("--offset", type=int, default=0, help="skip this many selected cells after sharding")
    parser.add_argument("--shard-count", type=int, default=None, help="number of deterministic compute shards")
    parser.add_argument("--shard-index", type=int, default=None, help="zero-based shard index for compute")
    parser.add_argument("--compute-only", action="store_true", help="write frame-level artifacts for selected cells only")
    parser.add_argument("--aggregate-existing", action="store_true", help="rebuild final tables from existing frame-level artifacts")
    parser.add_argument("--skip-existing", action="store_true", help="reuse existing frame-level files when present")
    args = parser.parse_args()
    if args.compute_only and args.aggregate_existing:
        raise SystemExit("--compute-only and --aggregate-existing are mutually exclusive")

    for directory in [OUT_DIR, TABLE_DIR, TRACE_DIR, FRAME_DIR, INTERMEDIATE_DIR, DERIVED_MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(DERIVED_ROOT / "mplconfig"))

    command = " ".join([Path(sys.executable).name, *sys.argv])
    commands_run = [command]
    shard_output_dir = OUT_DIR / "shards"
    report_output_dir = shard_output_dir if args.compute_only else OUT_DIR
    report_output_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = f"{shard_label(args)}_" if args.compute_only else ""
    access = FileAccess()
    status = "PASS"
    identity_ok = False
    identity_errors: List[str] = []
    cell_rows: List[Dict[str, object]] = []
    support_rows: List[Dict[str, object]] = []
    frame_manifest_rows: List[Dict[str, object]] = []
    created_paths: List[Path] = []
    parsed_logs: Dict[Path, Dict[str, Dict[str, object]]] = {}
    fallback_summary: Dict[str, object] = {}

    try:
        identity_ok, identity_errors = verify_identity(access)
        if not identity_ok:
            status = "BLOCKED"
            raise RuntimeError("identity/gate verification failed: " + "; ".join(identity_errors))

        access.read_text(OPERATIONAL_PARAMS, "frozen operational parameters")
        access.read_text(ANALYSIS_FREEZE, "frozen analysis freeze")
        access.read_text(SUPPORT_DEFINITION, "frozen support definition")
        access.read_text(IPV_SIGN_CONTRACT, "frozen IPV sign contract")
        conformal = json.loads(access.read_text(CONFORMAL_SUMMARY, "Gate 0 conformal threshold summary"))
        conformal_threshold = finite_float(conformal.get("conformal_threshold_90pct"))
        fallback_summary_path = INTERMEDIATE_DIR / "fallback_norm_summary.json"
        if args.aggregate_existing and fallback_summary_path.exists():
            fallback_summary_doc = json.loads(
                access.read_text(fallback_summary_path, "fallback norm summary from compute shards")
            )
            fallback_summary = fallback_summary_doc.get("levels", {})

        all_routes = load_routing(access, limit=None)
        routes = route_subset(
            all_routes,
            limit=args.limit,
            offset=args.offset,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )

        if args.aggregate_existing:
            for route_index, route in enumerate(routes, start=1):
                frame_path = frame_path_for_route(route)
                if frame_path.exists():
                    rows = load_frame_rows(frame_path, access)
                    cell = aggregate_rows(route, rows, "processed")
                    row = {
                        "cell_key": route["cell_key"],
                        "area": route["area"],
                        "team_code": route["team_code"],
                        "scenario": route["scenario"],
                        "path": str(frame_path.resolve()),
                        "relative_path": rel(frame_path),
                        "sha256": sha256_file(frame_path),
                        "bytes": frame_path.stat().st_size,
                        "generating_command": command,
                    }
                    frame_manifest_rows.append(row)
                else:
                    cell = aggregate_rows(route, [], "failed", "missing_frame_level_artifact")
                cell_rows.append(cell)
                support_rows.append(support_coverage_row(cell))
        else:
            exact_lookup = load_exact_norm_table(access)
            fallback_lookup = build_fallback_norms(access)
            fallback_summary = {
                level: {
                    "groups": len(table),
                    "high_support_groups": sum(1 for entry in table.values() if entry["support_level"] == "high"),
                }
                for level, table in fallback_lookup.items()
            }

            for route_index, route in enumerate(routes, start=1):
                print(f"[{route_index}/{len(routes)}] {route['cell_key']}", flush=True)
                frame_path = frame_path_for_route(route)
                try:
                    if args.skip_existing and frame_path.exists():
                        rows = load_frame_rows(frame_path, access)
                        cell = aggregate_rows(route, rows, "processed")
                    else:
                        cell, frame_path, _selection = process_cell(
                            route,
                            parsed_logs,
                            access,
                            exact_lookup,
                            fallback_lookup,
                            conformal_threshold,
                        )
                except Exception as exc:  # keep processing independent cells
                    cell = aggregate_rows(route, [], "failed", f"{type(exc).__name__}: {exc}")
                    cell["traceback"] = traceback.format_exc(limit=3)
                    frame_path = None
                cell_rows.append(cell)
                support_rows.append(support_coverage_row(cell))
                if frame_path is not None and frame_path.exists():
                    row = {
                        "cell_key": route["cell_key"],
                        "area": route["area"],
                        "team_code": route["team_code"],
                        "scenario": route["scenario"],
                        "path": str(frame_path.resolve()),
                        "relative_path": rel(frame_path),
                        "sha256": sha256_file(frame_path),
                        "bytes": frame_path.stat().st_size,
                        "generating_command": command,
                    }
                    frame_manifest_rows.append(row)
                    created_paths.append(frame_path)

        failed = sum(1 for r in cell_rows if r["status"] == "failed")
        if failed:
            status = "PARTIAL"
        if access.outcome_hits:
            status = "FAIL"

        if args.compute_only:
            shard_dir = OUT_DIR / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_manifest = shard_dir / f"{shard_label(args)}_frame_manifest.csv"
            write_csv(
                shard_manifest,
                frame_manifest_rows,
                [
                    "cell_key",
                    "area",
                    "team_code",
                    "scenario",
                    "path",
                    "relative_path",
                    "sha256",
                    "bytes",
                    "generating_command",
                ],
            )
            access.record(shard_manifest, "write", "compute-only shard frame manifest")
            created_paths.append(shard_manifest)
            # Final deliverables are written by --aggregate-existing after all shards finish.
        else:
            if args.aggregate_existing:
                for shard_report in sorted((OUT_DIR / "shards").glob("shard_*_worker_report.json")):
                    try:
                        shard_doc = json.loads(
                            access.read_text(shard_report, "aggregate shard command provenance")
                        )
                    except json.JSONDecodeError:
                        continue
                    for shard_command in shard_doc.get("commands_run", []):
                        if shard_command not in commands_run:
                            commands_run.append(shard_command)

            cell_table = TABLE_DIR / "cell_level_directional_ipv.csv"
            support_table = TABLE_DIR / "support_coverage.csv"
            write_csv(cell_table, cell_rows, CELL_FIELDS)
            write_csv(
                support_table,
                support_rows,
                [
                    "area",
                    "team_code",
                    "scenario",
                    "scenario_family",
                    "cell_key",
                    "status",
                    "main_cell_status",
                    "total_aligned_frames",
                    "conflict_active_frames",
                    "estimable_conflict_active_frames",
                    "main_high_support_active_frames",
                    "main_abstain_active_frames",
                    "fallback_high_support_active_frames",
                    "fallback_abstain_active_frames",
                    "main_support_coverage_active",
                    "fallback_support_coverage_active",
                    "fallback_level_counts_active",
                    "main_abstain_reason_counts_active",
                    "failure_reason",
                ],
            )
            access.record(cell_table, "write", "cell-level directional IPV predictors")
            access.record(support_table, "write", "support coverage table")
            created_paths.extend([cell_table, support_table])

            trace_paths: List[Path] = []
            for frame_row in frame_manifest_rows[:3]:
                src = Path(str(frame_row["path"]))
                trace_path = TRACE_DIR / f"directional_ipv_trace__{safe_slug(str(frame_row['cell_key']))}.csv"
                trace_path.write_bytes(src.read_bytes())
                access.record(trace_path, "write", "audit example full frame-level trace")
                trace_paths.append(trace_path)
            created_paths.extend(trace_paths)

            frame_manifest_path = DERIVED_MANIFEST_DIR / "frame_level_manifest.csv"
            run_frame_manifest = OUT_DIR / "frame_level_manifest.csv"
            frame_fields = [
                "cell_key",
                "area",
                "team_code",
                "scenario",
                "path",
                "relative_path",
                "sha256",
                "bytes",
                "generating_command",
            ]
            write_csv(frame_manifest_path, frame_manifest_rows, frame_fields)
            write_csv(run_frame_manifest, frame_manifest_rows, frame_fields)
            access.record(frame_manifest_path, "write", "derived frame-level artifact manifest")
            access.record(run_frame_manifest, "write", "run frame-level artifact manifest")
            created_paths.extend([frame_manifest_path, run_frame_manifest])

            report_path = OUT_DIR / "directional_ipv_report.md"
            write_report(
                report_path,
                cell_rows,
                support_rows,
                frame_manifest_rows,
                commands_run,
                identity_errors,
                conformal_threshold,
                fallback_summary,
            )
            access.record(report_path, "write", "directional IPV method and coverage report")
            created_paths.append(report_path)

    except Exception as exc:
        if status == "PASS":
            status = "FAIL"
        failure_path = report_output_dir / f"{report_prefix}failure_traceback.txt"
        failure_path.write_text(traceback.format_exc() + "\n", encoding="utf-8")
        access.record(failure_path, "write", f"failure traceback: {type(exc).__name__}")
        created_paths.append(failure_path)

    worker_report = {
        "status": status,
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "generated_at": now_iso(),
        "identity_ok": identity_ok,
        "identity_errors": identity_errors,
        "outcome_access": "NONE" if not access.outcome_hits else access.outcome_hits,
        "cells_requested": len(cell_rows),
        "cells_processed": sum(1 for r in cell_rows if r.get("status") == "processed"),
        "cells_failed": sum(1 for r in cell_rows if r.get("status") == "failed"),
        "cells_main_high_support": sum(1 for r in cell_rows if r.get("main_cell_status") == "HIGH_SUPPORT"),
        "cells_main_abstained": sum(1 for r in cell_rows if str(r.get("main_cell_status", "")).startswith("ABSTAIN")),
        "cells_fallback_evaluable": sum(
            1 for r in cell_rows if int(r.get("fallback_high_support_active_frames") or 0) > 0
        ),
        "frame_artifacts": len(frame_manifest_rows),
        "commands_run": commands_run,
        "fallback_norm_summary": fallback_summary,
        "spec_deviations": [
            "Replay AV and simulation role streams are timestamp-aligned by causal previous/current counterpart observation within 250 ms because exact millisecond timestamps differ across roles.",
            "Root START_HERE.md and main_workflow.log were not modified because this Phase 4A worker write scope is restricted to the run/derived roots and append-only run meta files.",
        ],
    }
    worker_report_path = report_output_dir / f"{report_prefix}worker_report.json"
    worker_report_path.write_text(json.dumps(worker_report, indent=2, ensure_ascii=False), encoding="utf-8")
    access.record(worker_report_path, "write", "worker report")
    created_paths.append(worker_report_path)

    access_manifest_path = report_output_dir / f"{report_prefix}file_access_manifest.txt"
    access.write_manifest(access_manifest_path)
    created_paths.append(access_manifest_path)

    artifact_manifest_path = report_output_dir / f"{report_prefix}artifact_manifest.csv"
    manifest_rows = artifact_rows(created_paths, command)
    write_csv(
        artifact_manifest_path,
        manifest_rows,
        ["path", "relative_path", "sha256", "bytes", "generating_command"],
    )
    created_paths.append(artifact_manifest_path)

    if not args.compute_only:
        append_meta_logs(status, created_paths)

    print(json.dumps(worker_report, indent=2, ensure_ascii=False), flush=True)
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
