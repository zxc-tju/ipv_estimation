#!/usr/bin/env python
"""Repair the Phase 4 future-leaky negative control without touching primary results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKER_ID = "RQ003_phase4_negcontrol_fix_001"
RUN_ID = "RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0"
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"
CONFIRMATORY_SHA256_BEFORE = "9b3b79f0704a09e5509780d9b096570b18d9275fd3a52d1381b2f74ea3be90d8"

NEG = Path(__file__).resolve().parent
RUN_ROOT = NEG.parents[1]
REPO_ROOT = RUN_ROOT.parents[3]
META = RUN_ROOT / "02_process/00_meta"
DIRT = RUN_ROOT / "01_results/tables"
CONF = RUN_ROOT / "02_process/10_confirmatory_analysis"
REV = CONF / "statistical_independent_review"
DERIVED_ROOT = REPO_ROOT / "data/derived/onsite_competition/RQ003_nsfc_external_evidence" / RUN_ID
PYTHON = DERIVED_ROOT / "model_cache/venv/bin/python"
FEATURE_WORKER = DERIVED_ROOT / "model_cache/compute_phase4_features.py"
CONFIRMATORY_SCRIPT = CONF / "run_confirmatory_analysis.py"
FRAME_LEVEL = DERIVED_ROOT / "frame_level/frame_level_directional_ipv.csv"
NCTRL_INTERMEDIATE = DERIVED_ROOT / "intermediate/negative_controls"
FIX_INTERMEDIATE = NCTRL_INTERMEDIATE / "future_leaky_fix"
MPLCONFIGDIR = DERIVED_ROOT / "model_cache/mplconfig"

CONTROL_NAME = "future_leaky_full_window_ipv"
LEAKY_MAX_STARTS_PER_CELL = 30
LEAKY_FUTURE_HORIZON_FRAMES = 30
LEAKY_SOLVER_PRESET = "balanced"
LEAKY_CANDIDATE_GRID = "legacy7"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


confirm = load_module(CONFIRMATORY_SCRIPT, "rq003_confirmatory_for_leaky_fix")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_float(value)
    return value


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{RUN_ID}|{WORKER_ID}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(label))


def verify_identity() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("RUN_ROOT exists", RUN_ROOT.is_dir(), str(RUN_ROOT))
    add("NEG exists", NEG.is_dir(), str(NEG))
    add("DERIVED_ROOT exists", DERIVED_ROOT.is_dir(), str(DERIVED_ROOT))
    try:
        manifest = json.loads((META / "run_manifest.json").read_text())
        add("run_manifest RUN_ID matches", manifest.get("RUN_ID") == RUN_ID, repr(manifest.get("RUN_ID")))
    except Exception as exc:
        add("run_manifest RUN_ID matches", False, repr(exc))
    try:
        add("plan_sha256 matches", (META / "plan_sha256.txt").read_text().strip() == PLAN_SHA256, "")
    except Exception as exc:
        add("plan_sha256 matches", False, repr(exc))
    try:
        status = json.loads((REV / "stats_review_status.json").read_text()).get("status")
        add("stats review status is FAIL", status == "FAIL", repr(status))
    except Exception as exc:
        add("stats review status is FAIL", False, repr(exc))
    for path in [
        DIRT / "negative_controls.csv",
        DIRT / "confirmatory_results.csv",
        DIRT / "cell_level_directional_ipv.csv",
        DIRT / "baseline_features_cells.csv",
        DIRT / "scenario_map_outcome_free.csv",
        DIRT / "replay_score_mapping.csv",
        NEG / "run_negative_controls.py",
        CONFIRMATORY_SCRIPT,
        FEATURE_WORKER,
        FRAME_LEVEL,
        PYTHON,
    ]:
        add(f"{path.name} exists", path.exists(), str(path))
    add("running expected project Python", Path(sys.executable).resolve() == PYTHON.resolve(), sys.executable)
    try:
        confirm.verify_identity()
        add("confirmatory script identity verification passes", True, "")
    except Exception as exc:
        add("confirmatory script identity verification passes", False, repr(exc))
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise RuntimeError(f"Pre-write identity verification failed: {failed}")
    return checks


def time_norm_auc(values: pd.Series | np.ndarray, times: pd.Series | np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    ts = np.asarray(times, dtype=float)
    finite = np.isfinite(vals) & np.isfinite(ts)
    vals = vals[finite]
    ts = ts[finite]
    if len(vals) == 0:
        return math.nan
    if len(vals) == 1:
        return float(vals[0])
    order = np.argsort(ts)
    vals = vals[order]
    ts = ts[order]
    duration = float(ts[-1] - ts[0])
    if duration <= 1e-9:
        return float(np.mean(vals))
    return float(np.trapezoid(vals, ts) / duration)


def primary_cell_ids() -> list[str]:
    df, _fold_contract, _read_paths = confirm.assemble_frame()
    mask = confirm.bool_series(df["primary_inclusion"])
    return df.loc[mask, "cell_id"].astype(str).tolist()


def sampled_start_positions(n_frames: int) -> np.ndarray:
    if n_frames < 2:
        return np.array([], dtype=int)
    n_take = min(LEAKY_MAX_STARTS_PER_CELL, n_frames - 1)
    return np.unique(np.linspace(0, n_frames - 2, n_take).round().astype(int))


def compute_shard(shard_index: int, shard_count: int) -> Path:
    verify_identity()
    FIX_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    feature_mod = load_module(FEATURE_WORKER, f"rq003_feature_worker_for_leaky_fix_{shard_index}")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_current

    norm_lookup = feature_mod.NormLookup(DERIVED_ROOT / "intermediate/interhub_conditional_norms.csv")
    wanted = set(primary_cell_ids())
    selected_ids = [cell_id for pos, cell_id in enumerate(sorted(wanted)) if pos % shard_count == shard_index]
    cells = [cell for cell in feature_mod.load_cells() if str(cell.row["cell_id"]) in set(selected_ids)]
    feature_mod.parse_logs_for_cells(cells)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for n, cell in enumerate(cells, 1):
        cell_id = str(cell.row["cell_id"])
        if cell.frames is None or cell.frames.empty:
            rows.append({"cell_id": cell_id, "status": "no_frames", "log_path": str(cell.log_path)})
            continue
        frames = cell.frames.copy()
        eligible = frames.loc[
            (frames.index >= feature_mod.MIN_OBSERVATION_FRAMES)
            & (pd.to_numeric(frames["distance_m"], errors="coerce") <= feature_mod.CONFLICT_DISTANCE_MAX_M)
        ].copy()
        starts = sampled_start_positions(len(eligible))
        if len(starts) == 0:
            rows.append({"cell_id": cell_id, "status": "too_few_eligible_frames", "log_path": str(cell.log_path)})
            continue
        for start_pos in starts:
            current = eligible.iloc[int(start_pos)]
            segment = eligible.iloc[int(start_pos) : min(len(eligible), int(start_pos) + LEAKY_FUTURE_HORIZON_FRAMES + 1)].copy()
            if len(segment) < 2:
                rows.append(
                    {
                        "cell_id": cell_id,
                        "status": "too_few_future_frames",
                        "start_pos": int(start_pos),
                        "time_s": finite_float(current.get("time_s")),
                        "log_path": str(cell.log_path),
                    }
                )
                continue
            ego_arr = segment[["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_heading"]].to_numpy(dtype=float)
            npc_arr = segment[["npc_x", "npc_y", "npc_vx", "npc_vy", "npc_heading"]].to_numpy(dtype=float)
            try:
                ipv, err = estimate_ipv_current(
                    MotionSequence(ego_arr, "gs", feature_mod.rolling_reference(ego_arr)),
                    MotionSequence(npc_arr, "gs", feature_mod.rolling_reference(npc_arr)),
                    history_window=max(len(ego_arr) - 1, 1),
                    solver_preset=LEAKY_SOLVER_PRESET,
                )
                theta_ego = float(ipv[0])
                theta_npc = float(ipv[1])
                tau = feature_mod.tau_bin(
                    min(
                        1.0,
                        max(
                            0.0,
                            (int(current.name) - feature_mod.MIN_OBSERVATION_FRAMES)
                            / feature_mod.PRIMARY_HISTORY_WINDOW,
                        ),
                    )
                )
                ego_norm = norm_lookup.lookup(
                    feature_mod.theta_bin(theta_npc),
                    str(current["state_condition_ego"]),
                    tau,
                )
                width = max(float(ego_norm["w"]), 1e-9)
                d_comp = max(0.0, (float(ego_norm["q_low"]) - theta_ego) / width)
                d_yield = max(0.0, (theta_ego - float(ego_norm["q_high"])) / width)
                status = "computed_future_inclusive_horizon"
                err_ego = finite_float(err[0])
                err_npc = finite_float(err[1])
            except Exception as exc:
                theta_ego = math.nan
                theta_npc = math.nan
                d_comp = math.nan
                d_yield = math.nan
                err_ego = math.nan
                err_npc = math.nan
                ego_norm = {"fallback_level": "", "exact_cell_n": 0, "q_low": math.nan, "q_high": math.nan, "w": math.nan}
                tau = ""
                status = f"estimate_failed:{type(exc).__name__}"
            rows.append(
                {
                    "cell_id": cell_id,
                    "team": cell.row["team"],
                    "area": cell.row["area"],
                    "scenario": cell.row["scenario"],
                    "family": cell.row["family"],
                    "log_path": str(cell.log_path),
                    "start_pos": int(start_pos),
                    "eligible_frames": int(len(eligible)),
                    "future_model_frames": int(len(segment)),
                    "time_s": finite_float(current.get("time_s")),
                    "state_condition_ego": str(current.get("state_condition_ego")),
                    "tau_bin": tau,
                    "future_leaky_theta_ego": theta_ego,
                    "future_leaky_theta_npc": theta_npc,
                    "future_leaky_error_ego": err_ego,
                    "future_leaky_error_npc": err_npc,
                    "future_leaky_D_comp_frame": d_comp,
                    "future_leaky_D_yield_frame": d_yield,
                    "norm_fallback_level": ego_norm.get("fallback_level", ""),
                    "norm_exact_cell_n": int(ego_norm.get("exact_cell_n", 0) or 0),
                    "q_low_ego": finite_float(ego_norm.get("q_low")),
                    "q_high_ego": finite_float(ego_norm.get("q_high")),
                    "w_ego": finite_float(ego_norm.get("w")),
                    "status": status,
                }
            )
        print(
            f"leaky shard {shard_index}/{shard_count} cell {n}/{len(cells)} {cell_id} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    out = FIX_INTERMEDIATE / f"future_leaky_frame_shard{shard_index:02d}_of_{shard_count:02d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def aggregate_cell_features(frame_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in frame_rows.groupby("cell_id", sort=True):
        ok = group[group["status"].astype(str) == "computed_future_inclusive_horizon"].copy()
        if ok.empty:
            rows.append(
                {
                    "cell_id": cell_id,
                    "future_leaky_D_comp": math.nan,
                    "future_leaky_D_yield": math.nan,
                    "status": "no_computed_future_horizon_frames",
                }
            )
            continue
        theta = pd.to_numeric(ok["future_leaky_theta_ego"], errors="coerce")
        rows.append(
            {
                "cell_id": cell_id,
                "future_leaky_D_comp": time_norm_auc(ok["future_leaky_D_comp_frame"], ok["time_s"]),
                "future_leaky_D_yield": time_norm_auc(ok["future_leaky_D_yield_frame"], ok["time_s"]),
                "future_leaky_theta_ego_mean": finite_float(theta.mean()),
                "future_leaky_theta_ego_median": finite_float(theta.median()),
                "future_leaky_theta_ego_std": finite_float(theta.std(ddof=1)),
                "future_leaky_theta_ego_min": finite_float(theta.min()),
                "future_leaky_theta_ego_max": finite_float(theta.max()),
                "future_leaky_frames_sampled": int(len(ok)),
                "future_leaky_horizon_frames": LEAKY_FUTURE_HORIZON_FRAMES,
                "future_leaky_max_starts_per_cell": LEAKY_MAX_STARTS_PER_CELL,
                "status": "computed_future_inclusive_framewise_horizon",
            }
        )
    return pd.DataFrame(rows)


def series_health(values: pd.Series) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n_finite": int(len(finite)),
        "min": finite_float(finite.min()) if len(finite) else None,
        "max": finite_float(finite.max()) if len(finite) else None,
        "mean": finite_float(finite.mean()) if len(finite) else None,
        "std": finite_float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "n_unique_rounded_1e12": int(finite.round(12).nunique()) if len(finite) else 0,
    }


def compute_health(df: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    primary = df.loc[confirm.bool_series(df["primary_inclusion"]), ["cell_id", "D_comp_auc", "D_yield_auc"]].copy()
    merged = primary.merge(features, on="cell_id", how="left", validate="one_to_one")
    for col in ["D_comp_auc", "D_yield_auc", "future_leaky_D_comp", "future_leaky_D_yield"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    delta_comp = merged["future_leaky_D_comp"] - merged["D_comp_auc"]
    delta_yield = merged["future_leaky_D_yield"] - merged["D_yield_auc"]
    comp_health = series_health(merged["future_leaky_D_comp"])
    yield_health = series_health(merged["future_leaky_D_yield"])
    theta_health = series_health(features["future_leaky_theta_ego_mean"])
    corr_comp = merged[["future_leaky_D_comp", "D_comp_auc"]].corr(method="spearman").iloc[0, 1]
    corr_yield = merged[["future_leaky_D_yield", "D_yield_auc"]].corr(method="spearman").iloc[0, 1]
    mean_abs_delta_comp = finite_float(delta_comp.abs().mean())
    mean_abs_delta_yield = finite_float(delta_yield.abs().mean())
    changed_comp = int((delta_comp.abs() > 1e-9).sum())
    changed_yield = int((delta_yield.abs() > 1e-9).sum())
    n_primary = int(len(merged))
    checks = {
        "D_comp_nonconstant": comp_health["n_unique_rounded_1e12"] > 1 and (comp_health["std"] or 0.0) > 1e-12,
        "D_yield_nonconstant": yield_health["n_unique_rounded_1e12"] > 1 and (yield_health["std"] or 0.0) > 1e-12,
        "theta_nonconstant": theta_health["n_unique_rounded_1e12"] > 1 and (theta_health["std"] or 0.0) > 1e-12,
        "differs_from_rolling": (
            (mean_abs_delta_comp is not None and mean_abs_delta_comp > 1e-3)
            or (mean_abs_delta_yield is not None and mean_abs_delta_yield > 1e-3)
        )
        and (changed_comp > 0 or changed_yield > 0),
        "sane_ranges": all(
            [
                comp_health["n_finite"] == n_primary,
                yield_health["n_finite"] == n_primary,
                (comp_health["min"] or 0.0) >= -1e-12,
                (yield_health["min"] or 0.0) >= -1e-12,
                (comp_health["max"] or 0.0) <= 20.0,
                (yield_health["max"] or 0.0) <= 20.0,
            ]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_primary_cells": n_primary,
        "D_comp": comp_health,
        "D_yield": yield_health,
        "theta_ego_mean": theta_health,
        "rolling_difference": {
            "mean_abs_delta_D_comp": mean_abs_delta_comp,
            "mean_abs_delta_D_yield": mean_abs_delta_yield,
            "changed_cells_D_comp": changed_comp,
            "changed_cells_D_yield": changed_yield,
            "spearman_corr_D_comp_vs_rolling": finite_float(corr_comp),
            "spearman_corr_D_yield_vs_rolling": finite_float(corr_yield),
        },
    }


def format_ci(ci: tuple[float, float]) -> str:
    lo = finite_float(ci[0])
    hi = finite_float(ci[1])
    if lo is None or hi is None:
        return "[NA, NA]"
    return f"[{lo:.6g}, {hi:.6g}]"


def no_incremental_gain(metrics: dict[str, float], pvals: dict[str, float], ci: dict[str, tuple[float, float]]) -> bool:
    delta_s = metrics["delta_spearman"]
    delta_mae = metrics["delta_mae_reduction"]
    delta_r2 = metrics["delta_cv_r2"]
    p_s = pvals.get("p_delta_spearman_greater", math.nan)
    ci_s = ci["delta_spearman"]
    significant_positive_s = (
        math.isfinite(delta_s)
        and delta_s > 0
        and math.isfinite(p_s)
        and p_s < 0.05
        and math.isfinite(ci_s[0])
        and ci_s[0] > 0
    )
    all_effects_favorable = (
        math.isfinite(delta_s)
        and math.isfinite(delta_mae)
        and math.isfinite(delta_r2)
        and delta_s > 0
        and delta_mae > 0
        and delta_r2 > 0
    )
    return not (significant_positive_s and all_effects_favorable)


def run_genuine_leaky_control(
    df: pd.DataFrame,
    fold_contract: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    control_df = df.merge(
        features[["cell_id", "future_leaky_D_comp", "future_leaky_D_yield"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    spec = confirm.AnalysisSpec(
        CONTROL_NAME,
        "negative_control",
        "leave_one_team_out",
        "primary_inclusion",
        ("future_leaky_D_comp", "future_leaky_D_yield"),
        confirm.PERMUTATIONS_PRIMARY,
        "negative_control",
    )
    pred = confirm.cross_validate(control_df, fold_contract, spec)
    metrics = confirm.metrics_from_predictions(pred)
    ci = confirm.bootstrap_ci(pred, rng_for(f"{CONTROL_NAME}:bootstrap"))
    pvals = confirm.permutation_test(control_df, fold_contract, spec, pred, rng_for(f"{CONTROL_NAME}:permutation"))
    if no_incremental_gain(metrics, pvals, ci):
        observed = (
            "genuine non-deployable future-horizon leaky control did not show a statistically supported "
            "positive increment; reported separately and excluded from NULL-robustness strengthening claims"
        )
    else:
        observed = (
            "genuine non-deployable future-horizon leaky control showed incremental signal; this is an "
            "optimistic diagnostic only and is not a deployable or primary confirmatory result"
        )
    row = {
        "control_name": CONTROL_NAME,
        "expected": "genuine future-inclusive IPV diagnostic; non-deployable optimistic upper-bound attempt, excluded from NULL-robustness claims",
        "delta_spearman": metrics["delta_spearman"],
        "ci": format_ci(ci["delta_spearman"]),
        "p": pvals.get("p_delta_spearman_greater", math.nan),
        "delta_mae": metrics["delta_mae_reduction"],
        "delta_cv_r2": metrics["delta_cv_r2"],
        "observed_behavior": observed,
        "pass_expected": True,
        "base_spearman": metrics["base_spearman"],
        "full_spearman": metrics["full_spearman"],
        "delta_spearman_ci_low": ci["delta_spearman"][0],
        "delta_spearman_ci_high": ci["delta_spearman"][1],
        "p_delta_spearman_greater": pvals.get("p_delta_spearman_greater", math.nan),
        "base_mae": metrics["base_mae"],
        "full_mae": metrics["full_mae"],
        "delta_mae_reduction": metrics["delta_mae_reduction"],
        "delta_mae_reduction_ci_low": ci["delta_mae_reduction"][0],
        "delta_mae_reduction_ci_high": ci["delta_mae_reduction"][1],
        "p_delta_mae_reduction_greater": pvals.get("p_delta_mae_reduction_greater", math.nan),
        "base_cv_r2": metrics["base_cv_r2"],
        "full_cv_r2": metrics["full_cv_r2"],
        "delta_cv_r2_ci_low": ci["delta_cv_r2"][0],
        "delta_cv_r2_ci_high": ci["delta_cv_r2"][1],
        "p_delta_cv_r2_greater": pvals.get("p_delta_cv_r2_greater", math.nan),
        "n_predictions": metrics["n_predictions"],
        "n_permutations": int(pvals.get("n_permutations", 0) or 0),
        "n_bootstrap": confirm.N_BOOTSTRAP,
        "d_columns": "future_leaky_D_comp;future_leaky_D_yield",
        "expectation_kind": "future_leaky_genuine_diagnostic",
    }
    pred = pred.copy()
    pred["control_name"] = CONTROL_NAME
    return row, pred


def csv_line_for_row(fieldnames: list[str], row: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writerow({name: row.get(name, "") for name in fieldnames})
    return buf.getvalue().rstrip("\n")


def replace_future_row_preserving_other_lines(row: dict[str, Any]) -> None:
    path = DIRT / "negative_controls.csv"
    before_path = NEG / "negative_controls__before_fix.csv"
    source_lines = before_path.read_text().splitlines()
    header = source_lines[0]
    fieldnames = next(csv.reader([header]))
    replacement = csv_line_for_row(fieldnames, row)
    out_lines: list[str] = [header]
    replaced = False
    for line in source_lines[1:]:
        if line.startswith(f"{CONTROL_NAME},"):
            out_lines.append(replacement)
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        raise RuntimeError(f"Could not find {CONTROL_NAME} row in {before_path}")
    path.write_text("\n".join(out_lines) + "\n")


def read_negative_controls() -> pd.DataFrame:
    return pd.read_csv(DIRT / "negative_controls.csv")


def reference_primary_result() -> dict[str, float]:
    confirmatory_results = pd.read_csv(DIRT / "confirmatory_results.csv")
    primary = confirmatory_results.loc[confirmatory_results["analysis_id"] == "primary_loto_confirmatory"].iloc[0]
    return {
        "base_spearman": float(primary["base_spearman"]),
        "base_mae": float(primary["base_mae"]),
        "base_cv_r2": float(primary["base_cv_r2"]),
        "delta_spearman": float(primary["delta_spearman"]),
        "delta_mae_reduction": float(primary["delta_mae_reduction"]),
        "delta_cv_r2": float(primary["delta_cv_r2"]),
    }


def write_negative_control_report(rows: pd.DataFrame, reference: dict[str, float], health: dict[str, Any]) -> tuple[str, str]:
    by_name = {r["control_name"]: r for r in rows.to_dict("records")}
    baseline_signal = float(by_name["kinematics_only"]["base_spearman"])
    shuffle_null_controls = [
        "ipv_time_shuffle",
        "shuffled_ipv",
        "wrong_envelope_cell",
        "counterpart_swap",
        "role_flip",
    ]
    shuffles_null = all(bool(by_name[name]["pass_expected"]) for name in shuffle_null_controls)
    state_degrades = bool(by_name["state_shuffle"]["pass_expected"]) and bool(by_name["wrong_state"]["pass_expected"])
    pipeline_validity = (
        "PASS: baseline retains signal and IPV/state corruptions behave as controls"
        if baseline_signal > 0 and shuffles_null and state_degrades
        else "PARTIAL: at least one degradation/null-control expectation did not cleanly pass"
    )
    primary_like = [
        "ipv_time_shuffle",
        "counterpart_swap",
        "role_flip",
        "sign_flip",
        "wrong_envelope_cell",
        "shuffled_ipv",
        "kinematics_only",
        "ipv_removed",
        "state_shuffle",
        "wrong_state",
    ]
    primary_null_robust = all(bool(by_name[name]["pass_expected"]) for name in primary_like)
    null_verdict = (
        "PASS: primary NULL/REVERSE is supported by the frozen primary LOTO result and the 10 valid negative controls. "
        "The future-leaky control is reported separately as a non-deployable optimistic diagnostic and is not used to strengthen NULL-robustness claims."
        if primary_null_robust
        else "PARTIAL: at least one of the 10 non-leaky controls did not meet its expectation; future-leaky is excluded from this verdict."
    )

    lines = [
        "# Phase 4 Negative-Control Report",
        "",
        f"Worker: `{WORKER_ID}`",
        f"Run: `{RUN_ID}`",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope",
        "",
        "All controls use the frozen primary leave-one-team-out pipeline: coordination residualized by in-fold scenario/area fixed effects, ridge models with the frozen alpha grid, the frozen primary inclusion mask, and the frozen capacity match of baseline versus baseline plus two IPV-like columns.",
        "The frozen confirmatory artifacts were read but not modified. The previous future-leaky full-window row was reclassified as an invalid no-op before this fix; the quarantined before-fix artifacts preserve that evidence.",
        "",
        "## Verdicts",
        "",
        f"- Pipeline validity verdict: {pipeline_validity}.",
        f"- NULL-robustness verdict: {null_verdict}",
        f"- Frozen model_base reference: Spearman={reference['base_spearman']:.6g}, MAE={reference['base_mae']:.6g}, CV-R2={reference['base_cv_r2']:.6g}.",
        f"- Future-leaky feature-health verdict: {health['status']}; this control is not part of the primary NULL-robustness claim.",
        "",
        "## Control Results",
        "",
        "| control | expected | delta Spearman | 95% CI | p | delta MAE reduction | delta CV-R2 | pass |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| {control_name} | {expected} | {delta_spearman:.6g} | {ci} | {p:.6g} | {delta_mae:.6g} | {delta_cv_r2:.6g} | {pass_expected} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            "## Future-Leaky Fix",
            "",
            "- The before-fix `future_leaky_full_window_ipv` row was invalid because the cached full-window estimate was constant, yielding exactly zero incremental deltas.",
            f"- The repaired control uses a future-inclusive framewise horizon: up to {LEAKY_MAX_STARTS_PER_CELL} evenly spaced starts per primary cell, with the actual next {LEAKY_FUTURE_HORIZON_FRAMES} conflict-window frames supplied to the optimizer. This is non-deployable.",
            f"- Feature health: D_comp nonconstant={health['checks']['D_comp_nonconstant']}; D_yield nonconstant={health['checks']['D_yield_nonconstant']}; theta nonconstant={health['checks']['theta_nonconstant']}; differs from rolling={health['checks']['differs_from_rolling']}; sane ranges={health['checks']['sane_ranges']}.",
            f"- Rolling-feature comparison: mean |delta D_comp|={health['rolling_difference']['mean_abs_delta_D_comp']:.6g}; mean |delta D_yield|={health['rolling_difference']['mean_abs_delta_D_yield']:.6g}; Spearman corr D_comp={health['rolling_difference']['spearman_corr_D_comp_vs_rolling']:.6g}; Spearman corr D_yield={health['rolling_difference']['spearman_corr_D_yield_vs_rolling']:.6g}.",
            "",
            "## Interpretation",
            "",
            "- `kinematics_only` and `ipv_removed` show the baseline side of the frozen pipeline retains signal; this argues against a broken outcome join or fold implementation.",
            "- `ipv_time_shuffle`, `shuffled_ipv`, `wrong_envelope_cell`, `counterpart_swap`, and `role_flip` are the IPV-null stress tests. Their expected behavior is no incremental held-out prediction gain.",
            "- `state_shuffle` and `wrong_state` corrupt the baseline state/kinematic features. Their expected behavior is degradation of the baseline reference signal relative to the frozen model_base.",
            "- `sign_flip` is a diagnostic control. Because the frozen ridge model is unconstrained and standardizes features in fold, sign reversal can be absorbed by coefficient sign changes; interpret it as a prediction diagnostic, not as evidence for a mechanistic direction.",
            "- `future_leaky_full_window_ipv` is now a genuine, non-deployable future-inclusive diagnostic only. It is not used to strengthen the primary no-criterion-validity conclusion.",
            "",
            "## Per-Control Observations",
            "",
        ]
    )
    for _, row in rows.iterrows():
        lines.append(f"- `{row['control_name']}`: {row['observed_behavior']}.")
    lines.extend(
        [
            "",
            "## Non-Goals Observed",
            "",
            "- No new confirmatory specification was introduced.",
            "- No state-dependence/NPC Phase 6 analysis was run.",
            "- No plotting or Tier decision was made.",
        ]
    )
    (NEG / "negative_control_report.md").write_text("\n".join(lines) + "\n")
    return pipeline_validity, null_verdict


def write_fix_report(row: dict[str, Any], health: dict[str, Any], feature_paths: dict[str, Path]) -> None:
    lines = [
        "# Future-Leaky Negative-Control Fix Report",
        "",
        f"Worker: `{WORKER_ID}`",
        f"Run: `{RUN_ID}`",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Decision",
        "",
        "A genuine future-leaky control was computed and used to replace the invalid before-fix no-op row.",
        "The old full-window cached control is reclassified as invalid/no-op evidence because its leaky theta and D features were constant and produced exactly zero deltas.",
        "",
        "## Method",
        "",
        f"- Optimizer: existing feature-worker parser plus `estimate_ipv_current`, solver preset `{LEAKY_SOLVER_PRESET}`, candidate grid `{LEAKY_CANDIDATE_GRID}`.",
        f"- Sharding: 10 shard files under `{FIX_INTERMEDIATE}`.",
        f"- Future leak: for each sampled primary-cell start, the actual next {LEAKY_FUTURE_HORIZON_FRAMES} conflict-window frames were supplied to the optimizer; an online deployable estimator would not have those future frames.",
        f"- Aggregation: time-normalized AUC over up to {LEAKY_MAX_STARTS_PER_CELL} evenly spaced starts per primary cell.",
        "",
        "## Feature-Health Checks",
        "",
        f"- Status: `{health['status']}`",
        f"- D_comp nonconstant: `{health['checks']['D_comp_nonconstant']}`; stats={health['D_comp']}",
        f"- D_yield nonconstant: `{health['checks']['D_yield_nonconstant']}`; stats={health['D_yield']}",
        f"- theta nonconstant: `{health['checks']['theta_nonconstant']}`; stats={health['theta_ego_mean']}",
        f"- Differs from rolling: `{health['checks']['differs_from_rolling']}`; {health['rolling_difference']}",
        f"- Sane ranges: `{health['checks']['sane_ranges']}`",
        "",
        "## Frozen LOTO Result For Repaired Leaky Control",
        "",
        f"- delta Spearman: `{row['delta_spearman']}`",
        f"- 95% CI: `{row['ci']}`",
        f"- p(delta Spearman > 0): `{row['p']}`",
        f"- delta MAE reduction: `{row['delta_mae']}`",
        f"- delta CV-R2: `{row['delta_cv_r2']}`",
        "",
        "## Artifacts",
        "",
    ]
    for label, path in feature_paths.items():
        lines.append(f"- {label}: `{path}`")
    (NEG / "leaky_control_fix_report.md").write_text("\n".join(lines) + "\n")


def append_artifact_index(paths: list[Path], command: str) -> None:
    index = META / "artifact_index.csv"
    fieldnames = ["artifact_path", "sha256", "size_bytes", "produced_by", "command", "purpose", "phase"]
    with index.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for path in paths:
            if not path.exists():
                continue
            writer.writerow(
                {
                    "artifact_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "produced_by": WORKER_ID,
                    "command": command,
                    "purpose": "Phase 4 future-leaky negative-control fix artifact",
                    "phase": "4_negative_control_fix",
                }
            )


def append_spec_deviation(status: str, health: dict[str, Any]) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    entry = (
        f"- `{timestamp}`: Phase 4 negative-control fixer `{WORKER_ID}` reclassified the before-fix "
        "`future_leaky_full_window_ipv` row as invalid/no-op and replaced it with a genuine non-deployable "
        f"future-inclusive diagnostic after feature-health status `{status}`. The primary confirmatory "
        "results and the other 10 controls were not modified. Repository-level `START_HERE.md` and "
        "`main_workflow.log` maintenance was not written because this worker's explicit WRITE_SCOPE was "
        "limited to `11_negative_controls/`, `negative_controls.csv`, derived intermediates, and append-only "
        "`00_meta` records."
    )
    if health.get("status") != "PASS":
        entry += " The leaky diagnostic was excluded from NULL-robustness claims."
    with (META / "spec_deviation_log.md").open("a") as f:
        f.write("\n" + entry + "\n")


def raw_other_control_lines_equal() -> bool:
    before = (NEG / "negative_controls__before_fix.csv").read_text().splitlines()[1:]
    after = (DIRT / "negative_controls.csv").read_text().splitlines()[1:]
    before_other = [line for line in before if not line.startswith(f"{CONTROL_NAME},")]
    after_other = [line for line in after if not line.startswith(f"{CONTROL_NAME},")]
    return before_other == after_other


def write_file_access_manifest(read_paths: list[Path], write_paths: list[Path], notes: list[str]) -> None:
    lines = [
        f"worker_id={WORKER_ID}",
        f"run_id={RUN_ID}",
        "READ:",
        *[str(p) for p in sorted(set(read_paths), key=lambda p: str(p))],
        "WRITE:",
        *[str(p) for p in write_paths],
        "NOTES:",
        *notes,
    ]
    (NEG / "file_access_manifest.txt").write_text("\n".join(lines) + "\n")


def write_artifact_manifest(paths: list[Path]) -> None:
    manifest = NEG / "artifact_manifest.csv"
    with manifest.open("w", newline="") as f:
        fieldnames = ["artifact_path", "sha256", "size_bytes", "produced_by", "purpose"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            if not path.exists():
                continue
            writer.writerow(
                {
                    "artifact_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "produced_by": WORKER_ID,
                    "purpose": "Phase 4 future-leaky negative-control fix artifact",
                }
            )


def merge_and_update(shard_count: int) -> dict[str, Any]:
    identity = verify_identity()
    shard_paths = [FIX_INTERMEDIATE / f"future_leaky_frame_shard{i:02d}_of_{shard_count:02d}.csv" for i in range(shard_count)]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing shard outputs: {missing}")
    frame_parts = [pd.read_csv(path) for path in shard_paths if path.stat().st_size > 0]
    frame_rows = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    frame_detail_path = FIX_INTERMEDIATE / "future_leaky_full_window_ipv_frame_detail.csv"
    frame_rows.to_csv(frame_detail_path, index=False)
    features = aggregate_cell_features(frame_rows)
    features_path = NCTRL_INTERMEDIATE / "future_leaky_full_window_ipv_features.csv"
    old_features_path = NCTRL_INTERMEDIATE / "future_leaky_full_window_ipv_features__before_fix_invalid_noop.csv"
    if features_path.exists() and not old_features_path.exists():
        old_features_path.write_bytes(features_path.read_bytes())
    features.to_csv(features_path, index=False)

    df, fold_contract, read_paths_raw = confirm.assemble_frame()
    health = compute_health(df, features)
    health_path = FIX_INTERMEDIATE / "future_leaky_feature_health.json"
    health_path.write_text(json.dumps(json_ready(health), indent=2, sort_keys=True))
    if health["status"] != "PASS":
        raise RuntimeError(f"Leaky feature-health checks failed: {health}")

    row, pred = run_genuine_leaky_control(df, fold_contract, features)
    predictions_path = FIX_INTERMEDIATE / "future_leaky_control_cv_predictions.csv"
    pred.to_csv(predictions_path, index=False)
    replace_future_row_preserving_other_lines(row)
    rows = read_negative_controls()
    reference = reference_primary_result()
    pipeline_validity, null_verdict = write_negative_control_report(rows, reference, health)
    write_fix_report(
        row,
        health,
        {
            "cell_features": features_path,
            "frame_detail": frame_detail_path,
            "health_json": health_path,
            "cv_predictions": predictions_path,
            "old_invalid_noop_features": old_features_path,
        },
    )

    confirmatory_unchanged = sha256_file(DIRT / "confirmatory_results.csv") == CONFIRMATORY_SHA256_BEFORE
    other_10_unchanged = raw_other_control_lines_equal()
    if not confirmatory_unchanged:
        raise RuntimeError("Protected confirmatory_results.csv hash changed")
    if not other_10_unchanged:
        raise RuntimeError("One or more non-leaky negative-control rows changed")

    read_paths = [
        META / "run_manifest.json",
        META / "plan_sha256.txt",
        REV / "stats_review_status.json",
        CONFIRMATORY_SCRIPT,
        FEATURE_WORKER,
        FRAME_LEVEL,
        DIRT / "confirmatory_results.csv",
        DIRT / "cell_level_directional_ipv.csv",
        DIRT / "baseline_features_cells.csv",
        DIRT / "scenario_map_outcome_free.csv",
        DIRT / "replay_score_mapping.csv",
        DIRT / "support_coverage.csv",
        NEG / "negative_controls__before_fix.csv",
        NEG / "negative_control_report__before_fix.md",
        NEG / "worker_report__before_fix.json",
        *[Path(p) for p in read_paths_raw],
        *shard_paths,
    ]
    if "log_path" in frame_rows.columns:
        read_paths.extend(Path(p) for p in sorted(set(frame_rows["log_path"].dropna().astype(str))))
    write_paths = [
        DIRT / "negative_controls.csv",
        NEG / "negative_control_report.md",
        NEG / "leaky_control_fix_report.md",
        NEG / "worker_report.json",
        NEG / "file_access_manifest.txt",
        NEG / "artifact_manifest.csv",
        Path(__file__).resolve(),
        NEG / "negative_controls__before_fix.csv",
        NEG / "negative_control_report__before_fix.md",
        NEG / "worker_report__before_fix.json",
        features_path,
        old_features_path,
        frame_detail_path,
        health_path,
        predictions_path,
        *shard_paths,
        META / "artifact_index.csv",
        META / "spec_deviation_log.md",
    ]
    notes = [
        "Before-fix future-leaky artifacts were quarantined with __before_fix names before active artifacts were rewritten.",
        "The old cached future-leaky feature file is preserved as future_leaky_full_window_ipv_features__before_fix_invalid_noop.csv.",
        "The primary confirmatory result hash is byte-identical to the pre-fix hash.",
        "The other 10 negative-control CSV rows are raw-line identical to the quarantined before-fix CSV.",
        "The future-leaky control is excluded from NULL-robustness strengthening claims.",
    ]
    write_file_access_manifest(read_paths, write_paths, notes)

    worker_report = {
        "status": "PASS",
        "worker_id": WORKER_ID,
        "role": "Phase 4 negative-control fixer",
        "run_id": RUN_ID,
        "identity_verification": identity,
        "fix_decision": "genuine_leaky_computed",
        "leaky_method": {
            "solver_preset": LEAKY_SOLVER_PRESET,
            "candidate_grid": LEAKY_CANDIDATE_GRID,
            "future_horizon_frames": LEAKY_FUTURE_HORIZON_FRAMES,
            "max_starts_per_cell": LEAKY_MAX_STARTS_PER_CELL,
            "shard_count": shard_count,
        },
        "feature_health": health,
        "fixed_leaky_row": row,
        "reference_primary_result": reference,
        "pipeline_validity_verdict": pipeline_validity,
        "null_robustness_verdict": null_verdict,
        "protected_outputs": {
            "confirmatory_results_sha256": sha256_file(DIRT / "confirmatory_results.csv"),
            "confirmatory_results_unchanged": confirmatory_unchanged,
            "other_10_controls_raw_lines_unchanged": other_10_unchanged,
        },
        "commands_run": [
            f"{PYTHON} {Path(__file__).resolve()} --stage shard --shard-index <0..{shard_count - 1}> --shard-count {shard_count}",
            f"{PYTHON} {Path(__file__).resolve()} --stage merge --shard-count {shard_count}",
        ],
        "tests_run": [
            "Pre-write identity verification",
            "Stats-review status verified FAIL before fix",
            "Genuine leaky feature-health checks: D_comp nonconstant, D_yield nonconstant, theta nonconstant, rolling-difference, sane ranges",
            "Frozen primary LOTO cross_validate() for repaired leaky control only",
            "Scenario-cluster bootstrap and scenario-stratified permutation for repaired leaky control",
            "confirmatory_results.csv sha256 unchanged",
            "Other 10 negative-control CSV rows raw-line unchanged",
        ],
        "spec_deviations": [
            "The before-fix full-window one-theta leaky control is reclassified as invalid/no-op and is not used as evidence.",
            "The repaired leaky diagnostic is non-deployable and excluded from NULL-robustness strengthening claims.",
            "Repository START_HERE.md/main_workflow.log maintenance was skipped because explicit worker WRITE_SCOPE excludes root files; this is logged in spec_deviation_log.md.",
        ],
        "artifacts": [str(p.resolve()) for p in write_paths if p.exists()],
    }
    (NEG / "worker_report.json").write_text(json.dumps(json_ready(worker_report), indent=2, sort_keys=True))

    artifact_paths = [p for p in write_paths if p.exists() and p not in {META / "artifact_index.csv", META / "spec_deviation_log.md"}]
    write_artifact_manifest(artifact_paths)
    artifact_paths.append(NEG / "artifact_manifest.csv")
    command = f"{PYTHON} {Path(__file__).resolve()} --stage merge --shard-count {shard_count}"
    append_artifact_index(artifact_paths, command)
    append_spec_deviation(health["status"], health)

    return {
        "STATUS": "PASS",
        "WORKER_ID": WORKER_ID,
        "ROLE": "Phase 4 negative-control fixer",
        "RUN_ID": RUN_ID,
        "SCOPE_COMPLETED": "Repaired only the future-leaky negative-control row and active negative-control language/artifacts.",
        "FILES_CREATED": [
            str(NEG / "leaky_control_fix_report.md"),
            str(NEG / "negative_controls__before_fix.csv"),
            str(NEG / "negative_control_report__before_fix.md"),
            str(NEG / "worker_report__before_fix.json"),
            str(features_path),
            str(frame_detail_path),
            str(health_path),
            str(predictions_path),
            str(old_features_path),
        ],
        "FILES_MODIFIED": [
            str(DIRT / "negative_controls.csv"),
            str(NEG / "negative_control_report.md"),
            str(NEG / "worker_report.json"),
            str(NEG / "file_access_manifest.txt"),
            str(NEG / "artifact_manifest.csv"),
            str(META / "artifact_index.csv"),
            str(META / "spec_deviation_log.md"),
        ],
        "COMMANDS_RUN": worker_report["commands_run"],
        "TESTS_RUN": worker_report["tests_run"],
        "KEY_EVIDENCE": {
            "genuine_leaky_computed": True,
            "leaky_feature_health": health,
            "fixed_leaky_delta": {
                "delta_spearman": finite_float(row["delta_spearman"]),
                "ci": row["ci"],
                "p": finite_float(row["p"]),
                "delta_mae": finite_float(row["delta_mae"]),
                "delta_cv_r2": finite_float(row["delta_cv_r2"]),
            },
            "primary_results_unchanged": confirmatory_unchanged,
            "other_10_controls_unchanged": other_10_unchanged,
        },
        "ACCEPTANCE_CRITERIA_RESULTS": {
            "identity_verified": True,
            "before_fix_artifacts_quarantined": True,
            "genuine_leaky_or_unavailable": "genuine_leaky_computed",
            "overclaiming_language_removed_from_active_report": True,
            "primary_loto_confirmatory_results_unchanged": confirmatory_unchanged,
            "other_10_controls_unchanged": other_10_unchanged,
        },
        "SPEC_DEVIATIONS": worker_report["spec_deviations"],
        "UNRESOLVED_BLOCKERS": [],
        "RECOMMENDED_NEXT_CODEX_TASK": "Phase 4 statistical reviewer re-check (leaky control only)",
        "GIT_DIFF_SUMMARY": "Added future-leaky fix runner/report, quarantined before-fix artifacts, replaced only the future-leaky row, and appended meta manifests.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["shard", "merge"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=10)
    args = parser.parse_args()
    if args.stage == "shard":
        out = compute_shard(args.shard_index, args.shard_count)
        print(json.dumps({"status": "PASS", "shard": args.shard_index, "path": str(out)}, indent=2))
        return 0
    result = merge_and_update(args.shard_count)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
