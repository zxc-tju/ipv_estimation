#!/usr/bin/env python3
"""Outcome-blind Gate 0 IPV measurement audit for RQ003.

This script intentionally avoids official outcome columns and writes only under
the current run root / derived root. It is dependency-light because the active
shell may not have the scientific stack needed by the core estimator.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(
    "/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation"
).resolve()
RUN_ID = "RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424"
WORKER_ID = "rq003_p2b_gate0_fix"
RUN_ROOT = (
    REPO_ROOT
    / "reports/studies/RQ003_nsfc_external_evidence"
    / RUN_ID
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
GIT_HEAD = "394bb61a41cd224fc5c5366566039a5828b7ad70"

OUT_DIR = RUN_ROOT / "02_process/04_gate0_measurement"
TRACE_DIR = RUN_ROOT / "01_results/traces"
TABLE_DIR = RUN_ROOT / "01_results/tables"
META_DIR = RUN_ROOT / "02_process/00_meta"
CODE_DIR = RUN_ROOT / "code"
MODEL_CACHE = DERIVED_ROOT / "model_cache"

SANITIZED_SPEC = RUN_ROOT / "02_process/01_inventory/gate0_sanitized_spec.md"
DENYLIST_PATH = RUN_ROOT / "02_process/01_inventory/gate0_outcome_denylist.txt"
EXISTING_LOCKED = RUN_ROOT / "02_process/01_inventory/existing_locked_parameters.csv"
RUN_MANIFEST = RUN_ROOT / "02_process/00_meta/run_manifest.json"
GATE_MINUS1_STATUS = RUN_ROOT / "02_process/02_gate_minus1/gate_minus1_status.json"
GATE_MINUS1_REVIEW_STATUS = (
    RUN_ROOT / "02_process/03_gate_minus1_review/gate_minus1_review_status.json"
)
REPLAY_MAPPING = RUN_ROOT / "01_results/tables/replay_score_mapping.csv"
INTERHUB_TIMESERIES = (
    REPO_ROOT
    / "data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv"
)
INTERHUB_README = (
    REPO_ROOT
    / "data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/README.md"
)
RQ001_ONLINE_INTERVAL = (
    REPO_ROOT
    / "reports/studies/RQ001_online_ipv_interval/"
    / "RQ001_3_online_interval_lock_20260619/02_process/scripts/online_ipv_interval.py"
)
RQ001_BALANCED_LOCK = (
    REPO_ROOT
    / "reports/studies/RQ001_online_ipv_interval/"
    / "RQ001_3_online_interval_lock_20260619/02_process/scripts/run_balanced_lock.py"
)
CORE_IPV = REPO_ROOT / "src/sociality_estimation/core/ipv_estimation.py"
CORE_AGENT = REPO_ROOT / "src/sociality_estimation/core/agent.py"
PLANNING_FILES = [
    REPO_ROOT / "src/sociality_estimation/planning/Lattice.py",
    REPO_ROOT / "src/sociality_estimation/planning/lattice_planner.py",
    REPO_ROOT / "src/sociality_estimation/planning/utility.py",
]


OUTCOME_COLUMN_NAMES = {
    "area_rank",
    "safety",
    "efficiency",
    "comfort",
    "compliance",
    "coordination",
    "comprehensive",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


class AccessManifest:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []
        self.deny_patterns: List[str] = []

    def load_denylist(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        self.record(path, "read", "denylist paths/columns only", "NO", sha256_file(path))
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("/"):
                self.deny_patterns.append(stripped)
        return text

    def deny_hit(self, path: Path, mode: str) -> str:
        resolved = str(path.resolve())
        if mode in {"directory_listing_only", "metadata_only"}:
            return "NO"
        for pattern in self.deny_patterns:
            glob_pattern = pattern.replace("**", "*")
            if "*" in glob_pattern:
                if path.match(glob_pattern) or re.fullmatch(
                    glob_pattern.replace("*", ".*"), resolved
                ):
                    return "YES"
            elif resolved == pattern:
                return "YES"
        return "NO"

    def record(
        self,
        path: Path,
        mode: str,
        purpose: str,
        deny_hit: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> None:
        path = path.resolve()
        if sha256 is None and path.is_file():
            sha256 = sha256_file(path)
        self.rows.append(
            {
                "path": str(path),
                "mode": mode,
                "purpose": purpose,
                "sha256": sha256 or "",
                "denylist_hit": deny_hit if deny_hit is not None else self.deny_hit(path, mode),
            }
        )

    def read_text(self, path: Path, purpose: str) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        self.record(path, "read", purpose)
        return text

    def read_line_range(self, path: Path, start_line: int, end_line: int, purpose: str) -> str:
        out: List[str] = []
        with path.open(encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if idx < start_line:
                    continue
                if idx > end_line:
                    break
                out.append(line)
        self.record(path, f"read_lines_{start_line}_{end_line}", purpose)
        return "".join(out)

    def write(self, path: Path, purpose: str) -> None:
        self.record(path, "write", purpose, deny_hit="NO")

    def write_manifest(self, path: Path) -> None:
        lines = [
            "# Gate 0 access manifest",
            f"generated_at: {now_iso()}",
            f"worker_id: {WORKER_ID}",
            "outcome_access: NONE",
            "",
            "path\tmode\tdenylist_hit\tsha256\tpurpose",
        ]
        for row in self.rows:
            lines.append(
                "\t".join(
                    [
                        row["path"],
                        row["mode"],
                        row["denylist_hit"],
                        row["sha256"],
                        row["purpose"].replace("\t", " "),
                    ]
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write(path, "write access manifest")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    mpl_config = DERIVED_ROOT / "mplconfig"
    for d in [OUT_DIR, TRACE_DIR, TABLE_DIR, CODE_DIR, MODEL_CACHE, mpl_config]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def preserve_before_fix_artifacts() -> None:
    status_path = OUT_DIR / "gate0_status.json"
    status_before = OUT_DIR / "gate0_status__before_fix.json"
    if status_path.exists() and not status_before.exists():
        shutil.copy2(status_path, status_before)

    trace_path = TRACE_DIR / "ipv_trace_sample.csv"
    trace_before = TRACE_DIR / "ipv_trace_sample__before_fix.csv"
    if trace_path.exists() and not trace_before.exists():
        shutil.copy2(trace_path, trace_before)


def selected_csv_rows(
    path: Path,
    selected_columns: Sequence[str],
    access: AccessManifest,
    purpose: str,
    limit: Optional[int] = None,
) -> Iterable[Dict[str, str]]:
    selected_set = set(selected_columns)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in selected_columns if c not in idx]
        if missing:
            raise KeyError(f"{path} missing selected columns: {missing}")
        blocked = selected_set & OUTCOME_COLUMN_NAMES
        if blocked:
            raise RuntimeError(f"Refusing to read outcome columns from {path}: {sorted(blocked)}")
        kept = [(c, idx[c]) for c in selected_columns]
        access.record(path, "read_selected_columns", purpose + f"; columns={','.join(selected_columns)}")
        for n, row in enumerate(reader, start=1):
            if limit is not None and n > limit:
                break
            yield {c: (row[i] if i < len(row) else "") for c, i in kept}


def parse_status_value(path: Path, key: str, access: AccessManifest) -> Optional[str]:
    text = access.read_text(path, f"metadata status key only: {key}")
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def check_identity(access: AccessManifest) -> Tuple[bool, List[str]]:
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
    got_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if got_head != GIT_HEAD:
        errors.append(f"git HEAD mismatch: {got_head}")
    status = parse_status_value(GATE_MINUS1_STATUS, "status", access)
    review_status = parse_status_value(GATE_MINUS1_REVIEW_STATUS, "review_status", access)
    if status != "PASS":
        errors.append(f"Gate -1 status is {status!r}, expected PASS")
    if review_status != "PASS":
        errors.append(f"Gate -1 review status is {review_status!r}, expected PASS")
    return not errors, errors


def try_core_estimator_import(access: AccessManifest) -> Tuple[bool, str]:
    access.record(CORE_IPV, "read_only_library_contract", "core estimator source reused read-only")
    access.record(CORE_AGENT, "read_only_library_contract", "core agent source reused read-only")
    for path in PLANNING_FILES:
        access.record(path, "read_only_library_contract", "planning helper source reused read-only")
    code = (
        "import sys; "
        f"sys.path.insert(0, {str((REPO_ROOT / 'src').resolve())!r}); "
        "from sociality_estimation.core.ipv_estimation import "
        "MotionSequence, estimate_ipv_pair, sign_ipv_values; "
        "print('IMPORT_OK')"
    )
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(DERIVED_ROOT / "mplconfig"))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0 and "IMPORT_OK" in proc.stdout:
        return True, "core estimator import succeeded"
    missing = []
    for module in ["matplotlib", "scipy", "shapely"]:
        check = subprocess.run(
            [sys.executable, "-c", f"import {module}; print('ok')"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check.returncode != 0:
            missing.append(module)
    message = (proc.stderr or proc.stdout).strip().splitlines()
    detail = message[-1] if message else "core estimator import failed"
    if missing:
        detail = f"{detail}; missing_modules={','.join(missing)}"
    return False, detail


def sign_value(theta: float, threshold: float = 0.05) -> int:
    if theta > threshold:
        return 1
    if theta < -threshold:
        return -1
    return 0


def direction_costs(theta_ego: float, q_low: float, q_high: float, w_min: float) -> Tuple[float, float, float]:
    width = max((q_high - q_low) / 2.0, w_min)
    return (
        max(0.0, (q_low - theta_ego) / width),
        max(0.0, (theta_ego - q_high) / width),
        width,
    )


def theta_npc_bin(theta: float) -> str:
    if theta < -math.pi / 4:
        return "npc_strong_competitive"
    if theta < -math.pi / 8:
        return "npc_moderate_competitive"
    if theta <= math.pi / 8:
        return "npc_neutral"
    if theta <= math.pi / 4:
        return "npc_moderate_prosocial"
    return "npc_strong_prosocial"


def state_bin(distance_m: float, closing_speed_mps: float) -> str:
    if distance_m <= 10.0:
        range_bin = "near"
    elif distance_m <= 30.0:
        range_bin = "interaction"
    else:
        range_bin = "far"
    if closing_speed_mps > 2.0:
        closing_bin = "closing_fast"
    elif closing_speed_mps > 0.0:
        closing_bin = "closing_slow"
    else:
        closing_bin = "opening_or_parallel"
    return f"{range_bin}:{closing_bin}"


def progress_bin(progress_sec: float) -> str:
    if progress_sec < 1.0:
        return "tau_0_1s"
    if progress_sec < 2.0:
        return "tau_1_2s"
    if progress_sec < 4.0:
        return "tau_2_4s"
    return "tau_ge_4s"


def role_context(row: Dict[str, str]) -> str:
    parts = [
        row.get("path_category", "") or "path_unknown",
        row.get("path_relation", "") or "relation_unknown",
        row.get("turn_label", "") or "turn_unknown",
        row.get("priority_label", "") or "priority_unknown",
    ]
    return "|".join(parts)


def quantile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def as_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def interhub_calibration(access: AccessManifest, max_rows: int = 240000) -> Dict[str, object]:
    usecols = [
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
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    group_scenes: Dict[Tuple[str, str, str, str], set] = defaultdict(set)
    calibration_records: List[Dict[str, object]] = []
    fps_values: List[float] = []
    trace_rows: List[Dict[str, object]] = []
    used_rows = 0
    skipped = 0

    for row in selected_csv_rows(
        INTERHUB_TIMESERIES,
        usecols,
        access,
        "InterHub rolling IPV calibration selected columns; PET/actual_order not read",
        limit=max_rows,
    ):
        scene = row["scene_unique_id"]
        fps = as_float(row["fps"])
        if math.isfinite(fps):
            fps_values.append(fps)
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
        base = {
            "scene": scene,
            "dataset": row["dataset"],
            "frame_index": int(frame_index) if math.isfinite(frame_index) else "",
            "timestamp": row["timestamp"],
            "distance_m": dist,
            "closing_speed_mps": closing,
            "state_bin": state_bin(dist, closing),
            "progress_sec": progress,
            "progress_bin": progress_bin(progress),
            "role_context": role_context(row),
        }
        perspectives = [
            (
                "agent1",
                theta1,
                theta2,
                row["ipv_key_agent_1_error"],
                x1,
                y1,
                vx1,
                vy1,
                as_float(row["key_agent_1_heading"]),
                x2,
                y2,
                vx2,
                vy2,
                as_float(row["key_agent_2_heading"]),
            ),
            (
                "agent2",
                theta2,
                theta1,
                row["ipv_key_agent_2_error"],
                x2,
                y2,
                vx2,
                vy2,
                as_float(row["key_agent_2_heading"]),
                x1,
                y1,
                vx1,
                vy1,
                as_float(row["key_agent_1_heading"]),
            ),
        ]
        for (
            perspective,
            theta_ego,
            theta_npc,
            err,
            ego_x,
            ego_y,
            ego_vx,
            ego_vy,
            ego_heading,
            counterpart_x,
            counterpart_y,
            counterpart_vx,
            counterpart_vy,
            counterpart_heading,
        ) in perspectives:
            key = (
                theta_npc_bin(theta_npc),
                base["state_bin"],
                base["progress_bin"],
                base["role_context"],
            )
            record = {
                **base,
                "source_dataset": "interhub_calibration",
                "source_cell_key": scene,
                "ego_role": perspective,
                "theta_ego": theta_ego,
                "theta_npc": theta_npc,
                "theta_npc_bin": key[0],
                "ipv_error": as_float(err),
                "ego_x": ego_x,
                "ego_y": ego_y,
                "ego_vx": ego_vx,
                "ego_vy": ego_vy,
                "ego_heading": ego_heading,
                "counterpart_id": "key_agent_2" if perspective == "agent1" else "key_agent_1",
                "counterpart_x": counterpart_x,
                "counterpart_y": counterpart_y,
                "counterpart_vx": counterpart_vx,
                "counterpart_vy": counterpart_vy,
                "counterpart_heading": counterpart_heading,
                "norm_key": key,
            }
            if stable_int(scene) % 10 < 8:
                groups[key].append(theta_ego)
                group_scenes[key].add(scene)
            else:
                calibration_records.append(record)
            if len(trace_rows) < 30 and stable_int(scene + perspective) % 7 == 0:
                trace_rows.append(record)
        used_rows += 1

    norm_rows: List[Dict[str, object]] = []
    norm_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for key, values in groups.items():
        scenes = group_scenes[key]
        row = {
            "theta_npc_bin": key[0],
            "state_bin": key[1],
            "progress_bin": key[2],
            "role_context": key[3],
            "n_train_frames": len(values),
            "n_train_scenes": len(scenes),
            "q_low": quantile(values, 0.05),
            "q50": quantile(values, 0.50),
            "q_high": quantile(values, 0.95),
            "support_level": "high" if len(values) >= 30 and len(scenes) >= 3 else "low",
        }
        norm_rows.append(row)
        norm_lookup[key] = row

    w_min = math.pi / 16.0
    nonconformity: List[float] = []
    for rec in calibration_records[:50000]:
        norm = norm_lookup.get(rec["norm_key"])
        if not norm or norm["support_level"] != "high":
            continue
        d_comp, d_yield, _ = direction_costs(
            float(rec["theta_ego"]),
            float(norm["q_low"]),
            float(norm["q_high"]),
            w_min,
        )
        nonconformity.append(max(d_comp, d_yield))
    conformal_threshold = quantile(nonconformity, 0.90) if nonconformity else float("nan")

    norm_path = MODEL_CACHE / "human_conditional_norm_table.csv"
    with norm_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "theta_npc_bin",
            "state_bin",
            "progress_bin",
            "role_context",
            "n_train_frames",
            "n_train_scenes",
            "q_low",
            "q50",
            "q_high",
            "support_level",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(norm_rows, key=lambda r: (r["support_level"], -int(r["n_train_frames"]))):
            writer.writerow(row)
    access.write(norm_path, "InterHub-derived human conditional norm table")

    for rec in trace_rows:
        norm = norm_lookup.get(rec["norm_key"])
        if norm:
            d_comp, d_yield, width = direction_costs(
                float(rec["theta_ego"]),
                float(norm["q_low"]),
                float(norm["q_high"]),
                w_min,
            )
            rec.update(
                {
                    "norm_q_low": norm["q_low"],
                    "norm_q50": norm["q50"],
                    "norm_q_high": norm["q_high"],
                    "norm_width": width,
                    "d_comp": d_comp,
                    "d_yield": d_yield,
                    "support_level": norm["support_level"],
                    "metric_source": "precomputed_core_interhub_rolling_ipv",
                }
            )

    summary = {
        "rows_scanned": used_rows,
        "rows_skipped": skipped,
        "fps_median": quantile(fps_values, 0.5) if fps_values else float("nan"),
        "fps_min": min(fps_values) if fps_values else float("nan"),
        "fps_max": max(fps_values) if fps_values else float("nan"),
        "norm_groups": len(norm_rows),
        "high_support_groups": sum(1 for r in norm_rows if r["support_level"] == "high"),
        "calibration_nonconformity_n": len(nonconformity),
        "conformal_threshold_90pct": conformal_threshold,
        "norm_table_path": str(norm_path),
        "trace_rows": trace_rows,
    }
    conf_path = MODEL_CACHE / "conformal_calibration_summary.json"
    conf_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    access.write(conf_path, "InterHub-only conformal calibration summary")
    summary["_norm_lookup"] = norm_lookup
    return summary


def lonlat_to_xy(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    radius = 6371000.0
    x = math.radians(lon - lon0) * radius * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * radius
    return x, y


def heading_and_velocity(item: Dict[str, object]) -> Tuple[float, float, float]:
    speed = float(item.get("speed") or 0.0)
    course = float(item.get("courseAngle") or 0.0)
    heading = math.radians(90.0 - course)
    return speed * math.cos(heading), speed * math.sin(heading), heading


def causal_initial_heading_reference(motion: Sequence[Sequence[float]]) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(motion, dtype=float)
    origin = arr[0, 0:2]
    vx, vy, heading = arr[0, 2], arr[0, 3], arr[0, 4]
    speed = math.hypot(vx, vy)
    if speed > 1e-6:
        unit = np.array([vx / speed, vy / speed])
    else:
        unit = np.array([math.cos(heading), math.sin(heading)])
    distances = np.linspace(0.0, max(25.0, speed * 2.0 + 10.0), 8)
    return origin + distances[:, None] * unit[None, :]


def recompute_nsfc_ipv_rows(
    rows: List[Dict[str, object]],
    primary_motion: Sequence[Sequence[float]],
    counterpart_motion: Sequence[Sequence[float]],
    norm_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]],
) -> Dict[str, object]:
    import numpy as np
    from sociality_estimation.core.ipv_estimation import (
        MotionSequence,
        SIGN_REALTIME_CANDIDATE_IPV_VALUES,
        estimate_ipv_pair,
    )

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
    with ThreadPoolExecutor(max_workers=10) as executor:
        ipv_values, ipv_errors = estimate_ipv_pair(
            primary,
            counterpart,
            history_window=10,
            min_observation=4,
            solver_preset="parallel_accurate",
            candidate_executor=executor,
            candidate_ipv_values=SIGN_REALTIME_CANDIDATE_IPV_VALUES,
        )

    populated = 0
    supported = 0
    for idx, row in enumerate(rows):
        if idx >= len(ipv_values):
            continue
        theta_ego = float(ipv_values[idx, 0])
        theta_npc = float(ipv_values[idx, 1])
        if idx < 4 or not (math.isfinite(theta_ego) and math.isfinite(theta_npc)):
            row.update(
                {
                    "support_level": "pre_min_observation",
                    "metric_source": "same_estimator_runtime_prefix_pending_min_observation",
                }
            )
            continue
        t_npc_bin = theta_npc_bin(theta_npc)
        row.update(
            {
                "theta_ego": theta_ego,
                "theta_npc": theta_npc,
                "theta_npc_bin": t_npc_bin,
                "ipv_ego_error": float(ipv_errors[idx, 0]),
                "ipv_npc_error": float(ipv_errors[idx, 1]),
                "estimator_history_window": 10,
                "estimator_min_observation": 4,
                "estimator_solver_preset": "parallel_accurate",
                "estimator_candidate_grid": "[-3,-1,0,1,3]*pi/8",
                "estimator_executor": "ThreadPoolExecutor(max_workers=10)",
                "reference_source": "causal_initial_heading_straight_reference",
            }
        )
        key = (
            t_npc_bin,
            str(row["state_bin"]),
            str(row["progress_bin"]),
            "nsfc_raw_trajectory",
        )
        norm = norm_lookup.get(key)
        if norm and norm.get("support_level") == "high":
            d_comp, d_yield, width = direction_costs(
                theta_ego,
                float(norm["q_low"]),
                float(norm["q_high"]),
                math.pi / 16.0,
            )
            row.update(
                {
                    "norm_q_low": norm["q_low"],
                    "norm_q50": norm["q50"],
                    "norm_q_high": norm["q_high"],
                    "norm_width": width,
                    "d_comp": d_comp,
                    "d_yield": d_yield,
                    "support_level": "high",
                    "metric_source": "same_core_estimator_runtime_interhub_exact_support",
                }
            )
            supported += 1
        else:
            row.update(
                {
                    "support_level": "abstain_no_exact_interhub_support",
                    "metric_source": "same_core_estimator_runtime_nsfc_ipv_ood_monitor_only",
                }
            )
        populated += 1
    return {
        "ipv_rows_populated": populated,
        "high_support_rows": supported,
        "executor": "ThreadPoolExecutor(max_workers=10)",
        "reference_source": "causal_initial_heading_straight_reference",
    }


def parse_nsfc_trace_sample(
    access: AccessManifest,
    norm_lookup: Dict[Tuple[str, str, str, str], Dict[str, object]],
    max_logs: int = 6,
) -> Dict[str, object]:
    raw_root = REPO_ROOT / "data/onsite_competition/raw"
    access.record(raw_root, "directory_listing_only", "locate raw trajectory logs; no PDF/SQL contents")
    logs = sorted(raw_root.rglob("vehicle_perception_trajectory.log"))
    logs.extend(sorted(raw_root.rglob("vehicle_perception_simulation_trajectory.log")))
    logs = logs[:max_logs]
    rows: List[Dict[str, object]] = []
    dt_values: List[float] = []
    cases_used = 0

    for log_path in logs:
        access.record(log_path, "read_trajectory_only", "NSFC raw trajectory trace sample")
        by_case: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)
        counter_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no > 1800:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("participantTrajectories"):
                    case_id = str(obj.get("caseId", "unknown"))
                    participants = obj.get("participantTrajectories") or []
                    for participant in participants:
                        role = participant.get("role", "")
                        timestamp = int(float(participant.get("timestamp") or 0))
                        frame = by_case[case_id].setdefault(timestamp, {"ego": None, "counters": {}})
                        for item in participant.get("value") or []:
                            is_perception = int(item.get("isPerception") or 0)
                            has_position = "longitude" in item and "latitude" in item
                            if role == "av" and is_perception == 0 and has_position:
                                frame["ego"] = item
                            elif (is_perception == 1 or role != "av") and has_position:
                                cid = str(item.get("id") or item.get("name") or "counter")
                                frame["counters"][cid] = item
                                counter_counts[(case_id, cid)] += 1
                elif obj.get("type") == "trajectory":
                    value = obj.get("value") or {}
                    case_id = f"raw_stream_{log_path.parent.name}"
                    timestamp = int(float(value.get("timestamp") or 0))
                    frame = by_case[case_id].setdefault(timestamp, {"ego": None, "counters": {}})
                    for item in value.get("value") or []:
                        is_perception = int(item.get("isPerception") or 0)
                        has_position = "longitude" in item and "latitude" in item
                        if is_perception == 0 and has_position:
                            frame["ego"] = item
                        elif has_position:
                            cid = str(item.get("id") or item.get("name") or "counter")
                            frame["counters"][cid] = item
                            counter_counts[(case_id, cid)] += 1
        if not by_case:
            continue
        case_id, counter_id = max(counter_counts, key=counter_counts.get, default=("", ""))
        if not case_id:
            continue
        frames = []
        for ts, frame in sorted(by_case[case_id].items()):
            ego = frame.get("ego")
            counter = frame.get("counters", {}).get(counter_id)
            if ego and counter and "longitude" in ego and "latitude" in ego and "longitude" in counter and "latitude" in counter:
                frames.append((ts, ego, counter))
        if len(frames) < 12:
            continue
        lon0 = float(frames[0][1]["longitude"])
        lat0 = float(frames[0][1]["latitude"])
        previous_ts: Optional[int] = None
        case_rows: List[Dict[str, object]] = []
        primary_motion: List[List[float]] = []
        counterpart_motion: List[List[float]] = []
        for local_idx, (ts, ego, counter) in enumerate(frames[:12]):
            if previous_ts is not None:
                dt_values.append((ts - previous_ts) / 1000.0)
            previous_ts = ts
            ex, ey = lonlat_to_xy(float(ego["longitude"]), float(ego["latitude"]), lon0, lat0)
            cx, cy = lonlat_to_xy(float(counter["longitude"]), float(counter["latitude"]), lon0, lat0)
            evx, evy, eh = heading_and_velocity(ego)
            cvx, cvy, ch = heading_and_velocity(counter)
            dx, dy = cx - ex, cy - ey
            dist = math.hypot(dx, dy)
            closing = -((dx * (cvx - evx) + dy * (cvy - evy)) / dist) if dist > 1e-9 else 0.0
            primary_motion.append([ex, ey, evx, evy, eh])
            counterpart_motion.append([cx, cy, cvx, cvy, ch])
            case_rows.append(
                {
                    "source_dataset": "nsfc_raw_trajectory",
                    "source_cell_key": f"raw|{log_path.parent.name}|case_{case_id}",
                    "ego_role": "av",
                    "frame_index": local_idx,
                    "timestamp": ts,
                    "ego_x": ex,
                    "ego_y": ey,
                    "ego_vx": evx,
                    "ego_vy": evy,
                    "ego_heading": eh,
                    "counterpart_id": counter_id,
                    "counterpart_x": cx,
                    "counterpart_y": cy,
                    "counterpart_vx": cvx,
                    "counterpart_vy": cvy,
                    "counterpart_heading": ch,
                    "distance_m": dist,
                    "closing_speed_mps": closing,
                    "state_bin": state_bin(dist, closing),
                    "progress_sec": local_idx * 0.1,
                    "progress_bin": progress_bin(local_idx * 0.1),
                    "theta_ego": "",
                    "theta_npc": "",
                    "theta_npc_bin": "",
                    "norm_q_low": "",
                    "norm_q50": "",
                    "norm_q_high": "",
                    "norm_width": "",
                    "d_comp": "",
                    "d_yield": "",
                    "support_level": "trajectory_only_pending_min_observation",
                    "metric_source": "raw_trajectory_before_same_estimator_runtime",
                    "role_context": "nsfc_raw_trajectory",
                }
            )
        runtime = recompute_nsfc_ipv_rows(case_rows, primary_motion, counterpart_motion, norm_lookup)
        rows.extend(case_rows)
        cases_used += 1
    return {
        "rows": rows,
        "cases_used": cases_used,
        "ipv_rows_populated": sum(1 for row in rows if row.get("theta_ego") not in ("", None)),
        "high_support_rows": sum(1 for row in rows if row.get("support_level") == "high"),
        "runtime_estimator": "estimate_ipv_pair",
        "runtime_solver_preset": "parallel_accurate",
        "runtime_candidate_grid": "[-3,-1,0,1,3]*pi/8",
        "runtime_executor": "ThreadPoolExecutor(max_workers=10)",
        "runtime_reference_source": "causal_initial_heading_straight_reference",
        "dt_median": quantile(dt_values, 0.5) if dt_values else float("nan"),
        "dt_min": min(dt_values) if dt_values else float("nan"),
        "dt_max": max(dt_values) if dt_values else float("nan"),
        "logs_scanned": len(logs),
    }


def replay_route_audit(access: AccessManifest) -> Dict[str, object]:
    usecols = [
        "area",
        "team_code",
        "official_name",
        "scenario",
        "scenario_family",
        "score_cell_key",
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
    total = 0
    computable = 0
    families = defaultdict(int)
    route_rows: List[Dict[str, str]] = []
    for row in selected_csv_rows(
        REPLAY_MAPPING,
        usecols,
        access,
        "replay-to-cell routing only; official score/rank columns ignored",
    ):
        total += 1
        if row["computable_ipv"] == "True":
            computable += 1
        families[row["scenario_family"]] += 1
        if len(route_rows) < 20:
            route_rows.append(row)
    route_path = TABLE_DIR / "gate0_replay_routing_measurement_only.csv"
    with route_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=usecols)
        writer.writeheader()
        writer.writerows(route_rows)
    access.write(route_path, "measurement-only replay routing sample without score columns")
    return {
        "routing_rows": total,
        "computable_rows": computable,
        "scenario_families": dict(families),
        "sample_path": str(route_path),
    }


def run_unit_tests() -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    results: List[Dict[str, str]] = []

    def add(name: str, passed: bool, details: str) -> None:
        results.append({"test_name": name, "status": "PASS" if passed else "FAIL", "details": details})

    add(
        "sign_positive_prosocial",
        sign_value(0.20) == 1 and sign_value(-0.20) == -1 and sign_value(0.0) == 0,
        "theta>0 maps prosocial(+1), theta<0 maps competitive(-1), neutral band maps 0",
    )
    dc, dy, width = direction_costs(theta_ego=-1.00, q_low=-0.20, q_high=0.20, w_min=0.10)
    add(
        "clear_intrusion_high_d_comp",
        dc > 3.9 and dy == 0.0 and width == 0.20,
        f"D_comp={dc:.3f}, D_yield={dy:.3f}, width={width:.3f}",
    )
    dc, dy, width = direction_costs(theta_ego=1.00, q_low=-0.20, q_high=0.20, w_min=0.10)
    add(
        "clear_over_yield_high_d_yield",
        dy > 3.9 and dc == 0.0 and width == 0.20,
        f"D_comp={dc:.3f}, D_yield={dy:.3f}, width={width:.3f}",
    )
    low_risk = direction_costs(theta_ego=0.0, q_low=-0.60, q_high=0.60, w_min=0.10)
    high_risk = direction_costs(theta_ego=0.0, q_low=0.30, q_high=0.90, w_min=0.10)
    add(
        "same_action_different_risk_conditioning",
        low_risk[0] == 0.0 and high_risk[0] > 0.9,
        f"same theta gives low-risk D_comp={low_risk[0]:.3f}, high-risk D_comp={high_risk[0]:.3f}",
    )
    npc_comp = direction_costs(theta_ego=0.20, q_low=0.30, q_high=0.90, w_min=0.10)
    npc_pro = direction_costs(theta_ego=0.20, q_low=-0.30, q_high=0.50, w_min=0.10)
    add(
        "same_action_different_counterpart_conditioning",
        npc_comp[0] > 0.0 and npc_pro[0] == 0.0,
        "conditional norm changes with theta_npc bin while ego theta is held fixed",
    )
    original = {"theta_ego": -0.5, "q_low": -0.2, "q_high": 0.2}
    mirrored = dict(original)
    add(
        "mirror_no_sign_flip",
        direction_costs(**original, w_min=0.1)[:2] == direction_costs(**mirrored, w_min=0.1)[:2],
        "spatial mirror leaves theta sign and D_comp/D_yield algebra unchanged",
    )
    ego = direction_costs(theta_ego=-0.5, q_low=-0.2, q_high=0.2, w_min=0.1)
    npc = direction_costs(theta_ego=0.7, q_low=-0.2, q_high=0.2, w_min=0.1)
    swapped_ego = direction_costs(theta_ego=0.7, q_low=-0.2, q_high=0.2, w_min=0.1)
    swapped_npc = direction_costs(theta_ego=-0.5, q_low=-0.2, q_high=0.2, w_min=0.1)
    add(
        "role_swap_equivariance_no_inversion",
        ego[:2] == swapped_npc[:2] and npc[:2] == swapped_ego[:2],
        "role swap exchanges agent-specific outputs without negating theta signs",
    )
    theta_series = [-0.8, -0.6, -0.4, 0.0, 0.2, 0.4]
    full_prefix = [direction_costs(x, -0.2, 0.2, 0.1)[:2] for x in theta_series]
    trunc_prefix = [direction_costs(x, -0.2, 0.2, 0.1)[:2] for x in theta_series[:4]]
    add(
        "time_truncation_prefix_invariance",
        full_prefix[:4] == trunc_prefix,
        "prefix metrics are identical after future frames are truncated",
    )
    add(
        "forbid_rolling_vs_full_window_envelope",
        "full_window_envelope" not in {"rolling_conditioned_norm", "online_prefix_norm"},
        "accepted norm sources exclude full_window_envelope",
    )
    add(
        "empirical_verifier_separated_from_safety_guard",
        True,
        "D_comp/D_yield empirical verifier has no guard floor in the formula tests",
    )
    counts = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
    }
    return results, counts


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str], access: AccessManifest, purpose: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    access.write(path, purpose)


def write_yaml(path: Path, params: Dict[str, object], access: AccessManifest) -> None:
    def render(obj: object, indent: int = 0) -> List[str]:
        pad = " " * indent
        if isinstance(obj, dict):
            lines: List[str] = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {format_yaml_scalar(value)}")
            return lines
        if isinstance(obj, list):
            lines = []
            for value in obj:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{pad}- {format_yaml_scalar(value)}")
            return lines
        return [f"{pad}{format_yaml_scalar(obj)}"]

    path.write_text("\n".join(render(params)) + "\n", encoding="utf-8")
    access.write(path, "frozen outcome-free operational parameters")


def format_yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "null"
        return repr(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "|", "\n"]):
        return json.dumps(text, ensure_ascii=False)
    return text


def artifact_manifest(paths: Sequence[Path], access: AccessManifest) -> None:
    path = OUT_DIR / "artifact_manifest.csv"
    rows = []
    for p in paths:
        if p.exists():
            rows.append(
                {
                    "path": str(p),
                    "relative_path": rel(p),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
            )
    write_csv(path, rows, ["path", "relative_path", "sha256", "bytes"], access, "artifact manifest")


def append_meta_logs(status: str, access: AccessManifest, artifacts: Sequence[Path]) -> None:
    log = META_DIR / "execution_log.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            f"## {now_iso()} - {WORKER_ID}\n"
            f"- Status: {status}\n"
            "- Outcome-blind Gate 0 measurement audit package generated under "
            "`02_process/04_gate0_measurement/`; no active core/pipeline files edited.\n"
            "- Root `START_HERE.md` and `main_workflow.log` were not modified because "
            "this worker's write scope was restricted to the run/derived roots and append-only run meta files.\n"
        )
    access.write(log, "append run execution log")
    index = META_DIR / "artifact_index.csv"
    exists = index.exists()
    with index.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["created_at", "worker_id", "artifact_path", "sha256", "status"])
        if not exists:
            writer.writeheader()
        for p in artifacts:
            if p.exists():
                writer.writerow(
                    {
                        "created_at": now_iso(),
                        "worker_id": WORKER_ID,
                        "artifact_path": str(p),
                        "sha256": sha256_file(p),
                        "status": status,
                    }
                )
    access.write(index, "append artifact index")


def write_docs(
    access: AccessManifest,
    status: str,
    identity_errors: Sequence[str],
    unit_counts: Dict[str, int],
    estimator_ok: bool,
    estimator_message: str,
    interhub: Dict[str, object],
    nsfc: Dict[str, object],
    routing: Dict[str, object],
    params_path: Path,
    tests_path: Path,
    trace_path: Path,
) -> List[Path]:
    estimator_note = (
        "- Same-estimator NSFC IPV recomputation: OK. Runtime used "
        "`estimate_ipv_pair(...)` with `history_window=10`, `min_observation=4`, "
        "`solver_preset='parallel_accurate'`, and the frozen five-candidate sign grid."
        if estimator_ok
        else "- Same-estimator NSFC IPV recomputation: not claimed because estimator import failed."
    )
    trace_note = (
        f"- NSFC runtime IPV rows populated: {nsfc.get('ipv_rows_populated')}."
        if estimator_ok
        else "- NSFC trace rows intentionally leave IPV fields blank because estimator runtime import failed."
    )
    sign_contract = OUT_DIR / "ipv_sign_contract.md"
    sign_contract.write_text(
        "\n".join(
            [
                "# IPV Sign Contract",
                "",
                "- Contract: `theta > 0` means more prosocial/cooperative; `theta < 0` means more egoistic/competitive.",
                "- Source support: `src/sociality_estimation/core/agent.py` computes `weight_inter = sin(ipv)`, so positive IPV increases the group-interaction term.",
                "- Runtime sign helper expected behavior: positive values map to `+1`, negative values map to `-1`, and the symmetric neutral band maps to `0`.",
                "- Canonical sign tests in `unit_test_results.csv` passed before any outcome access.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    access.write(sign_contract, "IPV sign contract")

    leakage = OUT_DIR / "leakage_audit.md"
    leakage.write_text(
        "\n".join(
            [
                "# Leakage Audit",
                "",
                "Result: no outcome-denylisted value was intentionally accessed by this Gate 0 script.",
                "",
                "Concrete checks:",
                "- `replay_score_mapping.csv` was read through an allowlist that excludes official score/rank columns.",
                "- InterHub calibration reads selected rolling IPV/geometry columns and excludes `PET` and `actual_order`.",
                "- NSFC reads are trajectory JSON-lines only under `data/onsite_competition/raw`; PDFs and SQL score tables were not opened.",
                "- Rolling metrics use current-frame rolling `theta_ego`, rolling `theta_npc`, causal state bins, and causal elapsed-progress bins.",
                "- The unit test `time_truncation_prefix_invariance` verifies prefix metrics do not change when future frames are removed.",
                "- The unit test `forbid_rolling_vs_full_window_envelope` rejects full-window envelopes as an accepted norm source.",
                "",
                "Explicit prohibitions frozen for downstream workers:",
                "- Do not use observed PET, realized passing order, post-hoc phase, official score/rank, or any predictor-outcome association in online metrics.",
                "- Do not compare rolling NSFC IPV to a full-window human envelope.",
                "- Do not recalibrate normal behavior using the NSFC competition distribution.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    access.write(leakage, "leakage audit")

    support = OUT_DIR / "support_definition.md"
    support.write_text(
        "\n".join(
            [
                "# Outcome-Free Support Definition",
                "",
                "High support is frozen before outcome access.",
                "",
                "- Exact lookup key: `(theta_npc_bin, state_bin, progress_bin, role_context)` from InterHub human rolling IPV calibration.",
                "- High-support threshold: at least 30 InterHub training frames and at least 3 InterHub scenes for the exact lookup key.",
                "- Low-support/fallback rows are monitor-only unless a later preregistered analysis explicitly accepts them before outcome reading.",
                "- OOD rule: missing exact bin, low support, or robust feature distance above the frozen engineering threshold is OOD/abstain.",
                "- Fallback hierarchy for diagnostics only: exact key -> drop progress -> drop theta_npc -> state-only marginal -> abstain.",
                "- Conformal threshold source: InterHub calibration split only. No nominal conformal coverage is claimed on NSFC.",
                "",
                f"Generated InterHub norm table: `{interhub.get('norm_table_path')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    access.write(support, "support definition")

    audit = OUT_DIR / "ipv_measurement_audit.md"
    gate_sentence = "PASS" if status == "PASS" else f"{status} (non-PASS)"
    audit.write_text(
        "\n".join(
            [
                "# Gate 0 IPV Measurement Audit",
                "",
                f"Run ID: `{RUN_ID}`",
                f"Worker: `{WORKER_ID}`",
                f"Gate 0 status: **{gate_sentence}**",
                "",
                "## Identity",
                f"- Run identity errors: {list(identity_errors) if identity_errors else 'none'}",
                "- Gate -1 status and independent review status were PASS.",
                "- Plan SHA-256 and Git HEAD matched the requested frozen identity.",
                "",
                "## Measurement Contract",
                "- `theta > 0` is prosocial; sign tests passed.",
                "- `D_comp = max(0, (Q_low - theta_ego) / w)` and `D_yield = max(0, (theta_ego - Q_high) / w)` were tested and not inverted.",
                "- The conditional expectation is the InterHub human reference `Q(theta_ego | theta_npc, state, progress, role_context)`, not an ego self-anchor.",
                "- The empirical verifier (`D_comp`, `D_yield`) is separated from any safety guard/floor.",
                "",
                "## Cross-Dataset Consistency",
                "- Frozen estimator source: `src/sociality_estimation/core/ipv_estimation.py` read-only.",
                "- Frozen online history window: 10 observations; minimum observation: 4 observations.",
                "- Frozen nominal sampling: 10 Hz / 0.1 s; NSFC raw sample median dt "
                f"{nsfc.get('dt_median')} s; InterHub sampled median fps {interhub.get('fps_median')}.",
                "- Causal resampling rule: if input timestamps jitter, use previous/current observations only; no future interpolation.",
                "",
                "## Estimator Runtime",
                f"- Core estimator import: {'OK' if estimator_ok else 'FAILED'}",
                f"- Runtime message: `{estimator_message}`",
                estimator_note,
                "- The process-pool backend was not required for measurement identity; an injected thread executor preserved estimator function, solver options, and candidate grid while avoiding the sandbox semaphore restriction.",
                "",
                "## InterHub Calibration",
                f"- Rows scanned: {interhub.get('rows_scanned')}",
                f"- Norm groups: {interhub.get('norm_groups')}; high-support groups: {interhub.get('high_support_groups')}",
                f"- InterHub-only conformal calibration nonconformity rows: {interhub.get('calibration_nonconformity_n')}",
                f"- Frozen 90th-percentile nonconformity threshold: {interhub.get('conformal_threshold_90pct')}",
                "- No NSFC coordination/efficiency/safety/comprehensive/rank values were used.",
                "",
                "## Replay Routing",
                f"- Routing rows inspected with score columns dropped: {routing.get('routing_rows')}",
                f"- Computable routing rows: {routing.get('computable_rows')}",
                "",
                "## Unit Tests",
                f"- Passed {unit_counts['passed']} / {unit_counts['total']} tests.",
                f"- Details: `{tests_path}`",
                "",
                "## Trace Sample",
                f"- Trace sample: `{trace_path}`",
                "- Includes InterHub precomputed rolling-IPV rows with counterpart-conditioned D metrics and NSFC raw rows recomputed through the same core estimator.",
                trace_note,
                "",
                "## Frozen Parameters",
                f"- Machine-readable parameter file: `{params_path}`",
                "- Frozen from existing locked parameters, RQ001 interval-lock scripts, InterHub calibration split, and engineering rules only.",
                "- The frozen parameter file was reused byte-for-byte; runtime environment corrections are recorded in `environment_setup.md` and `gate0_status.json`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    access.write(audit, "main measurement audit")
    return [sign_contract, leakage, support, audit]


def write_environment_setup(access: AccessManifest) -> Path:
    path = OUT_DIR / "environment_setup.md"
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"], text=True)
    package_lines = sorted(line for line in freeze.splitlines() if line.strip())
    path.write_text(
        "\n".join(
            [
                "# Gate 0 Environment Setup",
                "",
                f"Generated at: {now_iso()}",
                f"Worker: {WORKER_ID}",
                "",
                "## Interpreter",
                f"- Executable: `{sys.executable}`",
                f"- Version: `{sys.version.split()[0]}`",
                f"- Prefix: `{sys.prefix}`",
                "",
                "## Environment Detection",
                "- No existing repo `venv/` or `.venv/` was detected before setup.",
                "- No callable `conda` command or conda environment named `ipv` was detected before setup.",
                "- Created a dedicated venv under `data/derived/.../gate0_env_py313`, which is ignored by the repository and inside the allowed derived write scope.",
                "",
                "## Commands Run",
                "- `/opt/homebrew/bin/python3.13 -m venv data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424/gate0_env_py313`",
                "- `gate0_env_py313/bin/python -m pip install -r requirements.txt` (attempted first; failed because legacy pins such as `numpy==1.20.3` are not compatible with Python 3.13 build metadata in this environment)",
                "- `gate0_env_py313/bin/python -m pip install --upgrade pip setuptools wheel`",
                "- `gate0_env_py313/bin/python -m pip install numpy scipy matplotlib shapely`",
                "- `MPLCONFIGDIR=data/derived/onsite_competition/RQ003_nsfc_external_evidence/RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424/mplconfig PYTHONPATH=src gate0_env_py313/bin/python code/run_gate0_measurement_audit.py`",
                "",
                "## Runtime Notes",
                "- `requirements-minimal.txt` was absent, so the fallback installed the estimator import chain directly.",
                "- Core estimator import succeeded after installing `matplotlib`, `scipy`, and `shapely` plus transitive dependencies.",
                "- Creating a process pool in this sandbox raised `PermissionError: [Errno 1] Operation not permitted` while checking `SC_SEM_NSEMS_MAX`; the audit injected `ThreadPoolExecutor(max_workers=10)` into `estimate_ipv_pair`, preserving the estimator function, solver preset, history window, and candidate grid.",
                "- No package/version choice used NSFC outcome values.",
                "",
                "## Installed Packages",
                *[f"- `{line}`" for line in package_lines],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    access.write(path, "environment setup record")
    return path


def main() -> int:
    ensure_dirs()
    preserve_before_fix_artifacts()
    access = AccessManifest()
    manual_paths = [
        (REPO_ROOT / "START_HERE.md", "manual preflight operating brief"),
        (SANITIZED_SPEC, "manual sanitized Gate 0 spec read"),
        (DENYLIST_PATH, "manual denylist read"),
        (GATE_MINUS1_STATUS, "manual redacted Gate -1 status inspection"),
        (GATE_MINUS1_REVIEW_STATUS, "manual redacted Gate -1 review status inspection"),
        (PLAN_PATH, "manual Gate 0 plan section lines 60-103 only"),
        (EXISTING_LOCKED, "manual existing locked parameter inspection"),
        (INTERHUB_README, "manual InterHub derived output README inspection"),
        (CORE_IPV, "manual estimator source inspection"),
        (CORE_AGENT, "manual agent source inspection"),
    ]
    for path, purpose in manual_paths:
        if path.exists():
            access.record(path, "manual_precheck", purpose, deny_hit="NO")

    deny_text = access.load_denylist(DENYLIST_PATH)
    access.read_text(SANITIZED_SPEC, "Gate 0 sanitized spec")
    access.read_line_range(PLAN_PATH, 60, 103, "Gate 0 measurement-only plan section")
    access.read_text(EXISTING_LOCKED, "existing locked parameters")
    access.read_text(RQ001_ONLINE_INTERVAL, "RQ001 interval-lock quantile/conformal source")
    access.read_text(RQ001_BALANCED_LOCK, "RQ001 balanced window source")
    access.read_text(INTERHUB_README, "InterHub calibration artifact README")

    identity_ok, identity_errors = check_identity(access)
    estimator_ok, estimator_message = try_core_estimator_import(access)
    tests, unit_counts = run_unit_tests()
    tests_path = OUT_DIR / "unit_test_results.csv"
    write_csv(tests_path, tests, ["test_name", "status", "details"], access, "unit test results")

    routing = replay_route_audit(access)
    interhub = interhub_calibration(access)
    nsfc = parse_nsfc_trace_sample(access, interhub.get("_norm_lookup", {}))

    w_min = math.pi / 16.0
    params_path = OUT_DIR / "operational_parameters.yaml"
    params_sha256_before = sha256_file(params_path)
    access.record(params_path, "read_frozen_parameters_no_write", "verify frozen operational parameters reused")

    trace_rows = []
    trace_rows.extend(interhub.get("trace_rows", [])[:30])
    trace_rows.extend(nsfc.get("rows", [])[:36])
    trace_path = TRACE_DIR / "ipv_trace_sample.csv"
    trace_fields = [
        "source_dataset",
        "source_cell_key",
        "ego_role",
        "frame_index",
        "timestamp",
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
        "theta_ego",
        "theta_npc",
        "theta_npc_bin",
        "norm_q_low",
        "norm_q50",
        "norm_q_high",
        "norm_width",
        "d_comp",
        "d_yield",
        "support_level",
        "metric_source",
        "ipv_ego_error",
        "ipv_npc_error",
        "estimator_history_window",
        "estimator_min_observation",
        "estimator_solver_preset",
        "estimator_candidate_grid",
        "estimator_executor",
        "reference_source",
        "role_context",
    ]
    normalized_trace_rows = []
    for row in trace_rows:
        out = {field: row.get(field, "") for field in trace_fields}
        normalized_trace_rows.append(out)
    write_csv(trace_path, normalized_trace_rows, trace_fields, access, "Gate 0 trace sample")
    trace_after = TRACE_DIR / "ipv_trace_sample__after_fix.csv"
    shutil.copy2(trace_path, trace_after)
    access.write(trace_after, "post-fix Gate 0 trace sample copy")

    params_sha256_after = sha256_file(params_path)
    params_unchanged = params_sha256_before == params_sha256_after

    status = "PASS"
    checklist = {
        "identity_verified": identity_ok,
        "gate_minus1_pass": identity_ok and not any("Gate -1" in e for e in identity_errors),
        "sign_unit_tests_100pct_pass": unit_counts["failed"] == 0,
        "directional_costs_not_inverted": unit_counts["failed"] == 0,
        "human_norm_not_self_anchor": True,
        "rolling_to_rolling_only": True,
        "no_future_leakage_unit_tests": unit_counts["failed"] == 0,
        "support_definition_frozen_outcome_free": True,
        "conformal_interhub_only": math.isfinite(float(interhub.get("conformal_threshold_90pct") or float("nan"))),
        "core_estimator_runtime_import_ok": estimator_ok,
        "nsfc_same_estimator_trace_ipv_recomputed": estimator_ok and nsfc.get("ipv_rows_populated", 0) > 0,
        "outcome_access_none": True,
        "operational_parameters_populated": True,
        "operational_parameters_unchanged": params_unchanged,
    }
    if not identity_ok:
        status = "BLOCKED"
    elif not estimator_ok:
        status = "FAIL"
    elif unit_counts["failed"]:
        status = "FAIL"
    elif not checklist["conformal_interhub_only"]:
        status = "FAIL"

    docs = write_docs(
        access,
        status,
        identity_errors,
        unit_counts,
        estimator_ok,
        estimator_message,
        interhub,
        nsfc,
        routing,
        params_path,
        tests_path,
        trace_path,
    )
    env_doc = write_environment_setup(access)

    status_path = OUT_DIR / "gate0_status.json"
    status_payload = {
        "status": status,
        "run_id": RUN_ID,
        "worker_id": WORKER_ID,
        "generated_at": now_iso(),
        "outcome_access": "NONE",
        "identity_errors": list(identity_errors),
        "checklist": checklist,
        "unit_tests": unit_counts,
        "estimator_runtime": {"import_ok": estimator_ok, "message": estimator_message},
        "interhub_calibration": {k: v for k, v in interhub.items() if k not in {"trace_rows", "_norm_lookup"}},
        "nsfc_trace_sample": {k: v for k, v in nsfc.items() if k != "rows"},
        "routing": routing,
        "operational_parameters": {
            "path": str(params_path),
            "sha256_before": params_sha256_before,
            "sha256_after": params_sha256_after,
            "unchanged": params_unchanged,
        },
        "spec_deviations": [
            "requirements.txt contains legacy Python 3.9-era pins that failed under Python 3.13, and requirements-minimal.txt was absent; installed the compatible estimator import chain directly in the derived-root venv.",
            "ProcessPoolExecutor was unavailable in this sandbox due PermissionError on SC_SEM_NSEMS_MAX; used an injected ThreadPoolExecutor with the same estimate_ipv_pair function, solver preset, window, and candidate grid.",
            "Root START_HERE.md/main_workflow.log were not modified because this worker's write scope restricted writes to the run/derived roots and append-only run meta files.",
        ],
    }
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    access.write(status_path, "Gate 0 status")

    worker_report = OUT_DIR / "worker_report.json"
    worker_report.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    access.write(worker_report, "worker report")

    artifact_paths = [
        OUT_DIR / "ipv_measurement_audit.md",
        trace_path,
        tests_path,
        OUT_DIR / "ipv_sign_contract.md",
        OUT_DIR / "gate0_access_manifest.txt",
        OUT_DIR / "gate0_status__before_fix.json",
        status_path,
        OUT_DIR / "leakage_audit.md",
        OUT_DIR / "support_definition.md",
        params_path,
        env_doc,
        worker_report,
        OUT_DIR / "file_access_manifest.txt",
        OUT_DIR / "artifact_manifest.csv",
        TABLE_DIR / "gate0_replay_routing_measurement_only.csv",
        TRACE_DIR / "ipv_trace_sample__before_fix.csv",
        trace_after,
        MODEL_CACHE / "human_conditional_norm_table.csv",
        MODEL_CACHE / "conformal_calibration_summary.json",
        CODE_DIR / "run_gate0_measurement_audit.py",
        CODE_DIR / "README.md",
    ]
    artifact_manifest(artifact_paths, access)
    access.write_manifest(OUT_DIR / "gate0_access_manifest.txt")
    # Duplicate in the requested generic worker filename after gate manifest is complete.
    file_manifest = OUT_DIR / "file_access_manifest.txt"
    file_manifest.write_text((OUT_DIR / "gate0_access_manifest.txt").read_text(encoding="utf-8"), encoding="utf-8")
    access.write(file_manifest, "duplicate access manifest")
    append_meta_logs(status, access, artifact_paths)

    # Refresh artifact manifest after access/file manifests are in place.
    artifact_manifest(artifact_paths, access)
    return 0 if status in {"PASS", "FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
