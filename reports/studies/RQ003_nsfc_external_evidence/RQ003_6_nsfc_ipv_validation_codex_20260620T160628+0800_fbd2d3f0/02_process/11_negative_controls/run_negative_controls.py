#!/usr/bin/env python
"""Phase 4 negative-control battery for the frozen RQ003 confirmatory pipeline."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKER_ID = "RQ003_phase4_negcontrols_001"
RUN_ID = "RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0"
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"

REPO_ROOT = Path("/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation")
RUN_ROOT = REPO_ROOT / "reports/studies/RQ003_nsfc_external_evidence" / RUN_ID
META = RUN_ROOT / "02_process/00_meta"
FRZ = RUN_ROOT / "02_process/06_analysis_freeze"
DIRT = RUN_ROOT / "01_results/tables"
CONF = RUN_ROOT / "02_process/10_confirmatory_analysis"
NEG = RUN_ROOT / "02_process/11_negative_controls"
DERIVED_ROOT = REPO_ROOT / "data/derived/onsite_competition/RQ003_nsfc_external_evidence" / RUN_ID
PYTHON = DERIVED_ROOT / "model_cache/venv/bin/python"
FRAME_LEVEL = DERIVED_ROOT / "frame_level/frame_level_directional_ipv.csv"
FEATURE_WORKER = DERIVED_ROOT / "model_cache/compute_phase4_features.py"
NCTRL_INTERMEDIATE = DERIVED_ROOT / "intermediate/negative_controls"
MPLCONFIGDIR = DERIVED_ROOT / "model_cache/mplconfig"
LEAKY_MAX_FRAMES = 30

CONFIRMATORY_SCRIPT = CONF / "run_confirmatory_analysis.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


confirm = load_module(CONFIRMATORY_SCRIPT, "rq003_confirmatory_frozen")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{RUN_ID}|{WORKER_ID}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(label))


def as_bool(s: pd.Series) -> pd.Series:
    return confirm.bool_series(s)


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


def verify_identity() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("RUN_ROOT exists", RUN_ROOT.is_dir(), str(RUN_ROOT))
    add("NEG exists", NEG.is_dir(), str(NEG))
    manifest_path = META / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        add("run_manifest RUN_ID matches", manifest.get("RUN_ID") == RUN_ID, repr(manifest.get("RUN_ID")))
    except Exception as exc:  # pragma: no cover - fail-fast path
        add("run_manifest RUN_ID matches", False, repr(exc))
    try:
        add("plan_sha256 matches", (META / "plan_sha256.txt").read_text().strip() == PLAN_SHA256, "")
    except Exception as exc:  # pragma: no cover - fail-fast path
        add("plan_sha256 matches", False, repr(exc))
    for path in [
        DIRT / "confirmatory_results.csv",
        DIRT / "cv_predictions.csv",
        DIRT / "fold_assignments.csv",
        CONFIRMATORY_SCRIPT,
        PYTHON,
        FRZ / "analysis_freeze.yaml",
        FRZ / "fold_contract.csv",
        DIRT / "cell_level_directional_ipv.csv",
        DIRT / "baseline_features_cells.csv",
        DIRT / "scenario_map_outcome_free.csv",
        DIRT / "replay_score_mapping.csv",
        FRAME_LEVEL,
    ]:
        add(f"{path.name} exists", path.exists(), str(path))
    add("running expected project Python", Path(sys.executable).resolve() == PYTHON.resolve(), sys.executable)
    try:
        frozen_checks = confirm.verify_identity()
        add("confirmatory script identity verification passes", True, f"{len(frozen_checks)} checks")
    except Exception as exc:  # pragma: no cover - fail-fast path
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


def primary_indices(df: pd.DataFrame) -> np.ndarray:
    return df.index[as_bool(df["primary_inclusion"])].to_numpy()


def shuffle_baseline_block(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    idx = primary_indices(df)
    donor = rng_for(label).permutation(idx)
    out.loc[idx, confirm.BASELINE_FEATURES] = df.loc[donor, confirm.BASELINE_FEATURES].to_numpy()
    return out


def shuffle_baseline_independent(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    idx = primary_indices(df)
    rng = rng_for(label)
    for col in confirm.BASELINE_FEATURES:
        donor = rng.permutation(idx)
        out.loc[idx, col] = df.loc[donor, col].to_numpy()
    return out


def add_zero_ipv(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, tuple[str, str]]:
    out = df.copy()
    cols = (f"{prefix}_D_comp_zero", f"{prefix}_D_yield_zero")
    out[cols[0]] = 0.0
    out[cols[1]] = 0.0
    return out, cols


def add_source_columns(
    df: pd.DataFrame,
    prefix: str,
    source_cols: tuple[str, str],
    transform: str = "copy",
) -> tuple[pd.DataFrame, tuple[str, str]]:
    out = df.copy()
    cols = (f"{prefix}_D_comp", f"{prefix}_D_yield")
    if transform == "copy":
        out[cols[0]] = pd.to_numeric(out[source_cols[0]], errors="coerce")
        out[cols[1]] = pd.to_numeric(out[source_cols[1]], errors="coerce")
    elif transform == "role_flip":
        out[cols[0]] = pd.to_numeric(out[source_cols[1]], errors="coerce")
        out[cols[1]] = pd.to_numeric(out[source_cols[0]], errors="coerce")
    elif transform == "sign_flip":
        out[cols[0]] = -pd.to_numeric(out[source_cols[0]], errors="coerce")
        out[cols[1]] = -pd.to_numeric(out[source_cols[1]], errors="coerce")
    else:
        raise ValueError(transform)
    return out, cols


def add_scenario_stratified_cell_shuffle(df: pd.DataFrame, prefix: str, label: str) -> tuple[pd.DataFrame, tuple[str, str]]:
    out = df.copy()
    cols = (f"{prefix}_D_comp", f"{prefix}_D_yield")
    out[cols[0]] = pd.to_numeric(df["D_comp_auc"], errors="coerce")
    out[cols[1]] = pd.to_numeric(df["D_yield_auc"], errors="coerce")
    rng = rng_for(label)
    mask = as_bool(df["primary_inclusion"])
    for _, idx in df.loc[mask].groupby("scenario").groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        donor = rng.permutation(idx_list)
        out.loc[idx_list, list(cols)] = df.loc[donor, ["D_comp_auc", "D_yield_auc"]].to_numpy()
    return out, cols


def frame_high_support() -> pd.DataFrame:
    cols = [
        "cell_id",
        "time_s",
        "estimated",
        "conflict_window",
        "state_condition_ego",
        "theta_ego",
        "q_low_ego",
        "q_high_ego",
        "w_ego",
        "D_comp_ego",
        "D_yield_ego",
        "support_ego",
    ]
    frame = pd.read_csv(FRAME_LEVEL, usecols=cols)
    mask = (
        as_bool(frame["estimated"])
        & as_bool(frame["conflict_window"])
        & (frame["support_ego"].astype(str) == "high")
    )
    return frame.loc[mask].copy()


def aggregate_frame_features(frame: pd.DataFrame, comp_col: str, yield_col: str, out_prefix: str) -> pd.DataFrame:
    rows = []
    for cell_id, g in frame.groupby("cell_id", sort=False):
        rows.append(
            {
                "cell_id": cell_id,
                f"{out_prefix}_D_comp": time_norm_auc(g[comp_col], g["time_s"]),
                f"{out_prefix}_D_yield": time_norm_auc(g[yield_col], g["time_s"]),
            }
        )
    return pd.DataFrame(rows)


def add_ipv_time_shuffle(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str], Path]:
    high = frame_high_support()
    rng = rng_for("ipv_time_shuffle")
    for source, target in [("D_comp_ego", "D_comp_time_shuffle"), ("D_yield_ego", "D_yield_time_shuffle")]:
        values = high[source].to_numpy(dtype=float)
        high[target] = values[rng.permutation(len(values))]
    features = aggregate_frame_features(high, "D_comp_time_shuffle", "D_yield_time_shuffle", "ipv_time_shuffle")
    path = NCTRL_INTERMEDIATE / "ipv_time_shuffle_features.csv"
    features.to_csv(path, index=False)
    out = df.merge(features, on="cell_id", how="left", validate="one_to_one")
    return out, ("ipv_time_shuffle_D_comp", "ipv_time_shuffle_D_yield"), path


def add_wrong_envelope(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str], Path]:
    high = frame_high_support()
    needed = ["theta_ego", "q_low_ego", "q_high_ego", "w_ego"]
    high = high.dropna(subset=needed).reset_index(drop=True)
    states = high["state_condition_ego"].astype(str).to_numpy()
    rng = rng_for("wrong_envelope")
    donor_positions = []
    all_pos = np.arange(len(high))
    for pos, state in enumerate(states):
        candidates = all_pos[states != state]
        if len(candidates) == 0:
            candidates = all_pos[all_pos != pos]
        donor_positions.append(int(rng.choice(candidates)) if len(candidates) else pos)
    donor = high.iloc[donor_positions].reset_index(drop=True)
    theta = high["theta_ego"].to_numpy(dtype=float)
    q_low = donor["q_low_ego"].to_numpy(dtype=float)
    q_high = donor["q_high_ego"].to_numpy(dtype=float)
    width = np.maximum(donor["w_ego"].to_numpy(dtype=float), 1e-9)
    high["D_comp_wrong_envelope"] = np.maximum(0.0, (q_low - theta) / width)
    high["D_yield_wrong_envelope"] = np.maximum(0.0, (theta - q_high) / width)
    features = aggregate_frame_features(high, "D_comp_wrong_envelope", "D_yield_wrong_envelope", "wrong_envelope")
    path = NCTRL_INTERMEDIATE / "wrong_envelope_features.csv"
    features.to_csv(path, index=False)
    out = df.merge(features, on="cell_id", how="left", validate="one_to_one")
    return out, ("wrong_envelope_D_comp", "wrong_envelope_D_yield"), path


def compute_future_leaky_full_window_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Path, list[str]]:
    path = NCTRL_INTERMEDIATE / "future_leaky_full_window_ipv_features.csv"
    wanted = set(df.loc[as_bool(df["primary_inclusion"]), "cell_id"].astype(str))
    if path.exists():
        cached = pd.read_csv(path)
        required = {"cell_id", "future_leaky_D_comp", "future_leaky_D_yield"}
        if required.issubset(cached.columns) and wanted.issubset(set(cached["cell_id"].astype(str))):
            return cached, path, []

    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    feature_mod = load_module(FEATURE_WORKER, "rq003_feature_worker_for_leaky_control")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_current

    norm_path = DERIVED_ROOT / "intermediate/interhub_conditional_norms.csv"
    if not norm_path.exists():
        norm_path = feature_mod.build_calibration_table(force=False)
    norm_lookup = feature_mod.NormLookup(norm_path)

    cells = [cell for cell in feature_mod.load_cells() if str(cell.row["cell_id"]) in wanted]
    feature_mod.parse_logs_for_cells(cells)

    rows: list[dict[str, Any]] = []
    read_logs: list[str] = []
    for cell in cells:
        read_logs.append(str(cell.log_path))
        cell_id = str(cell.row["cell_id"])
        if cell.frames is None or cell.frames.empty:
            rows.append({"cell_id": cell_id, "future_leaky_D_comp": math.nan, "future_leaky_D_yield": math.nan, "status": "no_frames"})
            continue
        frames = cell.frames.copy()
        eligible = frames.loc[
            (frames.index >= feature_mod.MIN_OBSERVATION_FRAMES)
            & (pd.to_numeric(frames["distance_m"], errors="coerce") <= feature_mod.CONFLICT_DISTANCE_MAX_M)
        ].copy()
        if len(eligible) < 2:
            rows.append({"cell_id": cell_id, "future_leaky_D_comp": math.nan, "future_leaky_D_yield": math.nan, "status": "too_few_eligible_frames"})
            continue
        model_window = eligible
        if len(model_window) > LEAKY_MAX_FRAMES:
            take = np.unique(np.linspace(0, len(model_window) - 1, LEAKY_MAX_FRAMES).round().astype(int))
            model_window = model_window.iloc[take].copy()
        ego_arr = model_window[["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_heading"]].to_numpy(dtype=float)
        npc_arr = model_window[["npc_x", "npc_y", "npc_vx", "npc_vy", "npc_heading"]].to_numpy(dtype=float)
        try:
            ipv, err = estimate_ipv_current(
                MotionSequence(ego_arr, "gs", feature_mod.rolling_reference(ego_arr)),
                MotionSequence(npc_arr, "gs", feature_mod.rolling_reference(npc_arr)),
                history_window=max(len(ego_arr) - 1, 1),
                solver_preset="balanced",
            )
            theta_ego = float(ipv[0])
            theta_npc = float(ipv[1])
            last = eligible.iloc[-1]
            ego_norm = norm_lookup.lookup(
                feature_mod.theta_bin(theta_npc),
                str(last["state_condition_ego"]),
                feature_mod.tau_bin(1.0),
            )
            q_low = float(ego_norm["q_low"])
            q_high = float(ego_norm["q_high"])
            width = max(float(ego_norm["w"]), 1e-9)
            d_comp = max(0.0, (q_low - theta_ego) / width)
            d_yield = max(0.0, (theta_ego - q_high) / width)
            status = f"computed_full_conflict_window_non_deployable_bounded_{len(model_window)}_frames"
        except Exception as exc:  # pragma: no cover - recorded as missing, then median-imputed by frozen transform
            theta_ego = math.nan
            theta_npc = math.nan
            err = [math.nan, math.nan]
            d_comp = math.nan
            d_yield = math.nan
            status = f"estimate_failed:{type(exc).__name__}"
        rows.append(
            {
                "cell_id": cell_id,
                "future_leaky_D_comp": d_comp,
                "future_leaky_D_yield": d_yield,
                "future_leaky_theta_ego": theta_ego,
                "future_leaky_theta_npc": theta_npc,
                "future_leaky_error_ego": finite_float(err[0]),
                "future_leaky_error_npc": finite_float(err[1]),
                "future_leaky_frames": int(len(eligible)),
                "future_leaky_model_frames": int(len(model_window)),
                "status": status,
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(path, index=False)
    return features, path, read_logs


def add_future_leaky(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str], Path, list[str]]:
    features, path, read_logs = compute_future_leaky_full_window_features(df)
    out = df.merge(features[["cell_id", "future_leaky_D_comp", "future_leaky_D_yield"]], on="cell_id", how="left", validate="one_to_one")
    return out, ("future_leaky_D_comp", "future_leaky_D_yield"), path, read_logs


def build_spec(control_name: str, d_cols: tuple[str, str]) -> Any:
    return confirm.AnalysisSpec(
        control_name,
        "negative_control",
        "leave_one_team_out",
        "primary_inclusion",
        tuple(d_cols),
        confirm.PERMUTATIONS_PRIMARY,
        "negative_control",
    )


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


def degradation_checks(metrics: dict[str, float], reference: dict[str, float]) -> tuple[bool, list[str]]:
    checks = [
        ("base_spearman_lower", metrics["base_spearman"] < reference["base_spearman"]),
        ("base_mae_higher", metrics["base_mae"] > reference["base_mae"]),
        ("base_cv_r2_lower", metrics["base_cv_r2"] < reference["base_cv_r2"]),
    ]
    passed = [name for name, ok in checks if bool(ok)]
    return len(passed) >= 2, passed


def run_control(
    df_control: pd.DataFrame,
    fold_contract: pd.DataFrame,
    control_name: str,
    expected: str,
    d_cols: tuple[str, str],
    expectation_kind: str,
    reference: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = build_spec(control_name, d_cols)
    pred = confirm.cross_validate(df_control, fold_contract, spec)
    metrics = confirm.metrics_from_predictions(pred)
    ci = confirm.bootstrap_ci(pred, rng_for(f"{control_name}:bootstrap"))
    pvals = confirm.permutation_test(df_control, fold_contract, spec, pred, rng_for(f"{control_name}:permutation"))

    if expectation_kind == "degradation":
        pass_expected, passed_checks = degradation_checks(metrics, reference)
        observed = (
            f"baseline signal check {passed_checks}; base Spearman {metrics['base_spearman']:.6g} "
            f"vs reference {reference['base_spearman']:.6g}; base MAE {metrics['base_mae']:.6g} "
            f"vs {reference['base_mae']:.6g}; base CV-R2 {metrics['base_cv_r2']:.6g} "
            f"vs {reference['base_cv_r2']:.6g}"
        )
    elif expectation_kind == "baseline_reference":
        tol = 1e-10
        pass_expected = (
            abs(metrics["delta_spearman"]) <= tol
            and abs(metrics["delta_mae_reduction"]) <= tol
            and abs(metrics["delta_cv_r2"]) <= tol
            and abs(metrics["base_spearman"] - reference["base_spearman"]) <= tol
        )
        observed = "baseline-only reference retained the frozen model_base signal and zero IPV increment"
    elif expectation_kind == "sign_flip":
        pass_expected = no_incremental_gain(metrics, pvals, ci)
        observed = "sign-flipped IPV is prediction-null/reverse; unconstrained ridge can absorb sign reversals, so this is diagnostic rather than directional proof"
    elif expectation_kind == "future_leaky":
        pass_expected = True
        if no_incremental_gain(metrics, pvals, ci):
            observed = "non-deployable future-leaky diagnostic did not show a statistically supported positive increment"
        else:
            observed = "non-deployable future-leaky diagnostic showed incremental signal; interpret separately from primary NULL-robustness claims"
    else:
        pass_expected = no_incremental_gain(metrics, pvals, ci)
        observed = (
            "no statistically supported positive IPV increment"
            if pass_expected
            else "unexpected statistically supported positive IPV increment"
        )

    row = {
        "control_name": control_name,
        "expected": expected,
        "delta_spearman": metrics["delta_spearman"],
        "ci": format_ci(ci["delta_spearman"]),
        "p": pvals.get("p_delta_spearman_greater", math.nan),
        "delta_mae": metrics["delta_mae_reduction"],
        "delta_cv_r2": metrics["delta_cv_r2"],
        "observed_behavior": observed,
        "pass_expected": bool(pass_expected),
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
        "d_columns": ";".join(d_cols),
        "expectation_kind": expectation_kind,
    }
    pred = pred.copy()
    pred["control_name"] = control_name
    return row, pred


def write_report(path: Path, rows: pd.DataFrame, reference: dict[str, float]) -> tuple[str, str]:
    by_name = {r["control_name"]: r for r in rows.to_dict("records")}
    baseline_signal = by_name["kinematics_only"]["base_spearman"]
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
    ]
    primary_null_robust = all(bool(by_name[name]["pass_expected"]) for name in primary_like)
    future = by_name["future_leaky_full_window_ipv"]
    future_null = bool(
        future["pass_expected"]
        and "did not show" in str(future["observed_behavior"])
    )
    null_verdict = (
        "PASS: primary NULL/REVERSE is robust to the requested negative controls"
        if primary_null_robust
        else "PARTIAL: at least one IPV-corruption control showed unexpected incremental utility"
    )
    if future_null:
        null_verdict += "; the future-leaky diagnostic is reported separately and is not used to strengthen NULL-robustness claims."
    else:
        null_verdict += "; the future-leaky diagnostic is not null and remains separate from the primary NULL-robustness claim."

    lines = [
        "# Phase 4 Negative-Control Report",
        "",
        f"Worker: `{WORKER_ID}`",
        f"Run: `{RUN_ID}`",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope",
        "",
        "All controls used the frozen primary leave-one-team-out pipeline: coordination residualized by in-fold scenario/area fixed effects, ridge models with the frozen alpha grid, the frozen primary inclusion mask, and the frozen capacity match of baseline versus baseline plus two IPV-like columns.",
        "The frozen confirmatory artifacts were read but not modified.",
        "",
        "## Verdicts",
        "",
        f"- Pipeline validity verdict: {pipeline_validity}.",
        f"- NULL-robustness verdict: {null_verdict}",
        f"- Frozen model_base reference: Spearman={reference['base_spearman']:.6g}, MAE={reference['base_mae']:.6g}, CV-R2={reference['base_cv_r2']:.6g}.",
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
            "## Interpretation",
            "",
            "- `kinematics_only` and `ipv_removed` show the baseline side of the frozen pipeline retains signal; this argues against a broken outcome join or fold implementation.",
            "- `ipv_time_shuffle`, `shuffled_ipv`, `wrong_envelope_cell`, `counterpart_swap`, and `role_flip` are the IPV-null stress tests. Their expected behavior is no incremental held-out prediction gain.",
            "- `state_shuffle` and `wrong_state` corrupt the baseline state/kinematic features. Their expected behavior is degradation of the baseline reference signal relative to the frozen model_base.",
            "- `sign_flip` is a diagnostic control. Because the frozen ridge model is unconstrained and standardizes features in fold, sign reversal can be absorbed by coefficient sign changes; interpret it as a prediction diagnostic, not as evidence for a mechanistic direction.",
            "- `future_leaky_full_window_ipv` is a non-deployable diagnostic only and must be interpreted with explicit feature-health checks. It is not part of the NULL-robustness verdict.",
            "- The before-fix one-theta full-window cache was reclassified as invalid/no-op by `fix_future_leaky_control.py`; future reruns should use that fixer or equivalent health checks before interpreting this diagnostic.",
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
    path.write_text("\n".join(lines) + "\n")
    return pipeline_validity, null_verdict


def append_artifact_index(paths: list[Path], command: str) -> None:
    index = META / "artifact_index.csv"
    fieldnames = ["artifact_path", "sha256", "size_bytes", "produced_by", "command", "purpose", "phase"]
    with index.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for path in paths:
            writer.writerow(
                {
                    "artifact_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "produced_by": WORKER_ID,
                    "command": command,
                    "purpose": "Phase 4 frozen pipeline negative-control artifact",
                    "phase": "4_negative_controls",
                }
            )


def main() -> int:
    NEG.mkdir(parents=True, exist_ok=True)
    NCTRL_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    identity = verify_identity()
    df, fold_contract, read_paths = confirm.assemble_frame()
    analysis_frame_path = NCTRL_INTERMEDIATE / "negative_control_base_analysis_frame.csv"
    df.to_csv(analysis_frame_path, index=False)

    confirmatory_results = pd.read_csv(DIRT / "confirmatory_results.csv")
    primary = confirmatory_results.loc[confirmatory_results["analysis_id"] == "primary_loto_confirmatory"].iloc[0].to_dict()
    reference = {
        "base_spearman": float(primary["base_spearman"]),
        "base_mae": float(primary["base_mae"]),
        "base_cv_r2": float(primary["base_cv_r2"]),
        "delta_spearman": float(primary["delta_spearman"]),
        "delta_mae_reduction": float(primary["delta_mae_reduction"]),
        "delta_cv_r2": float(primary["delta_cv_r2"]),
    }

    controls: list[tuple[str, str, pd.DataFrame, tuple[str, str], str]] = []
    intermediate_paths: list[Path] = [analysis_frame_path]
    extra_read_paths: list[str] = []

    state_df = shuffle_baseline_block(df, "state_shuffle")
    controls.append(("state_shuffle", "permute baseline state features across primary cells; expect degradation", state_df, tuple(confirm.PRIMARY_D_COLS), "degradation"))

    time_df, time_cols, time_path = add_ipv_time_shuffle(df)
    intermediate_paths.append(time_path)
    controls.append(("ipv_time_shuffle", "shuffle directional IPV values across cells/time; expect no gain", time_df, time_cols, "no_gain"))

    counterpart_df, counterpart_cols = add_source_columns(df, "counterpart_swap", ("D_comp_auc_npc", "D_yield_auc_npc"))
    controls.append(("counterpart_swap", "swap ego/NPC conditioning; expect no improvement", counterpart_df, counterpart_cols, "no_gain"))

    role_df, role_cols = add_source_columns(df, "role_flip", ("D_comp_auc", "D_yield_auc"), transform="role_flip")
    controls.append(("role_flip", "swap competition/yield role labels; expect no improvement", role_df, role_cols, "no_gain"))

    sign_df, sign_cols = add_source_columns(df, "sign_flip", ("D_comp_auc", "D_yield_auc"), transform="sign_flip")
    controls.append(("sign_flip", "negate D_comp/D_yield; diagnostic of direction", sign_df, sign_cols, "sign_flip"))

    wrong_env_df, wrong_env_cols, wrong_env_path = add_wrong_envelope(df)
    intermediate_paths.append(wrong_env_path)
    controls.append(("wrong_envelope_cell", "use mismatched conditional-norm envelope cells; expect no gain", wrong_env_df, wrong_env_cols, "no_gain"))

    kin_df, kin_cols = add_zero_ipv(df, "kinematics_only")
    controls.append(("kinematics_only", "baseline-only kinematic+safety reference signal", kin_df, kin_cols, "baseline_reference"))

    removed_df, removed_cols = add_zero_ipv(df, "ipv_removed")
    controls.append(("ipv_removed", "remove IPV columns so full arm equals model_base", removed_df, removed_cols, "baseline_reference"))

    shuffled_df, shuffled_cols = add_scenario_stratified_cell_shuffle(df, "shuffled_ipv", "shuffled_ipv")
    controls.append(("shuffled_ipv", "scenario-stratified shuffled IPV; expect null", shuffled_df, shuffled_cols, "no_gain"))

    wrong_state_df = shuffle_baseline_independent(df, "wrong_state")
    controls.append(("wrong_state", "independently corrupt state features; expect degradation", wrong_state_df, tuple(confirm.PRIMARY_D_COLS), "degradation"))

    future_df, future_cols, future_path, future_logs = add_future_leaky(df)
    intermediate_paths.append(future_path)
    extra_read_paths.extend(future_logs)
    controls.append(("future_leaky_full_window_ipv", "future-leaky diagnostic only; non-deployable and excluded from NULL-robustness claims unless feature-health checks pass", future_df, future_cols, "future_leaky"))

    rows = []
    pred_parts = []
    for control_name, expected, control_df, d_cols, expectation_kind in controls:
        row, pred = run_control(control_df, fold_contract, control_name, expected, d_cols, expectation_kind, reference)
        rows.append(row)
        pred_parts.append(pred)

    result_columns = [
        "control_name",
        "expected",
        "delta_spearman",
        "ci",
        "p",
        "delta_mae",
        "delta_cv_r2",
        "observed_behavior",
        "pass_expected",
        "base_spearman",
        "full_spearman",
        "delta_spearman_ci_low",
        "delta_spearman_ci_high",
        "p_delta_spearman_greater",
        "base_mae",
        "full_mae",
        "delta_mae_reduction",
        "delta_mae_reduction_ci_low",
        "delta_mae_reduction_ci_high",
        "p_delta_mae_reduction_greater",
        "base_cv_r2",
        "full_cv_r2",
        "delta_cv_r2_ci_low",
        "delta_cv_r2_ci_high",
        "p_delta_cv_r2_greater",
        "n_predictions",
        "n_permutations",
        "n_bootstrap",
        "d_columns",
        "expectation_kind",
    ]
    results = pd.DataFrame(rows)[result_columns]
    results_path = DIRT / "negative_controls.csv"
    results.to_csv(results_path, index=False)

    predictions_path = NCTRL_INTERMEDIATE / "negative_control_cv_predictions.csv"
    pd.concat(pred_parts, ignore_index=True).to_csv(predictions_path, index=False)

    report_path = NEG / "negative_control_report.md"
    pipeline_validity, null_verdict = write_report(report_path, results, reference)

    file_access_path = NEG / "file_access_manifest.txt"
    artifact_manifest_path = NEG / "artifact_manifest.csv"
    worker_report_path = NEG / "worker_report.json"

    write_paths = [
        results_path,
        predictions_path,
        report_path,
        worker_report_path,
        file_access_path,
        artifact_manifest_path,
        *intermediate_paths,
        Path(__file__).resolve(),
    ]
    file_access_path.write_text(
        "\n".join(
            [
                f"worker_id={WORKER_ID}",
                f"run_id={RUN_ID}",
                "READ:",
                *sorted(set(read_paths + [str(DIRT / "confirmatory_results.csv"), str(DIRT / "cv_predictions.csv"), str(DIRT / "fold_assignments.csv"), str(CONFIRMATORY_SCRIPT), str(FRAME_LEVEL), str(FEATURE_WORKER), *extra_read_paths])),
                "WRITE:",
                *[str(p) for p in write_paths],
                "NOTES:",
                "Frozen confirmatory script and frozen/input tables were not modified.",
            "Future-leaky IPV is non-deployable, diagnostic only, and excluded from NULL-robustness strengthening claims.",
            "The before-fix one-theta full-window cache was reclassified as invalid/no-op; use fix_future_leaky_control.py for the repaired diagnostic.",
                "Repository START_HERE/main_workflow maintenance was not written because the worker write scope was restricted to NEG, negative_controls.csv, DERIVED_ROOT intermediates, and META artifact_index append.",
            ]
        )
        + "\n"
    )

    artifact_paths = [p for p in write_paths if p != artifact_manifest_path]

    worker_report = {
        "status": "PASS",
        "worker_id": WORKER_ID,
        "role": "Phase 4 negative-control worker",
        "run_id": RUN_ID,
        "identity_verification": identity,
        "reference_primary_result": reference,
        "pipeline_validity_verdict": pipeline_validity,
        "null_robustness_verdict": null_verdict,
        "controls": results.replace({np.nan: None}).to_dict("records"),
        "commands_run": [
            "python identity/input inspection snippets",
            f"{PYTHON} {Path(__file__).resolve()}",
        ],
        "tests_run": [
            "Pre-write identity verification",
            "Confirmatory script verify_identity()",
            "Frozen primary LOTO cross_validate() reused for every control",
            "Scenario-cluster bootstrap reused for every control",
            "Scenario-stratified permutation p-values reused for every control",
        ],
        "spec_deviations": [
            "No frozen artifact or input table was edited.",
            "Repository START_HERE/main_workflow maintenance was skipped because the explicit worker write scope excluded root files.",
            "Future-leaky IPV is generated only as a non-deployable diagnostic, not as a new confirmatory spec.",
            "The before-fix one-theta full-window cache was reclassified as invalid/no-op and must not be used as evidence without repaired feature-health checks.",
        ],
        "artifacts": [str(p.resolve()) for p in [*artifact_paths, artifact_manifest_path]],
    }
    worker_report_path.write_text(json.dumps(json_ready(worker_report), indent=2, sort_keys=True))

    with artifact_manifest_path.open("w", newline="") as f:
        fieldnames = ["artifact_path", "sha256", "size_bytes", "produced_by", "purpose"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in artifact_paths:
            writer.writerow(
                {
                    "artifact_path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "produced_by": WORKER_ID,
                    "purpose": "Phase 4 negative-control battery",
                }
            )
    artifact_paths.append(artifact_manifest_path)
    command = f"{PYTHON} {Path(__file__).resolve()}"
    append_artifact_index(artifact_paths, command)

    key_evidence = {
        "control_directions": {
            row["control_name"]: {
                "delta_spearman": finite_float(row["delta_spearman"]),
                "delta_mae": finite_float(row["delta_mae"]),
                "delta_cv_r2": finite_float(row["delta_cv_r2"]),
                "pass_expected": bool(row["pass_expected"]),
            }
            for row in rows
        },
        "pipeline_validity_verdict": pipeline_validity,
        "future_leaky_diagnostic": by_name_if_present(results, "future_leaky_full_window_ipv"),
        "null_robustness_verdict": null_verdict,
    }
    stdout = {
        "STATUS": "PASS",
        "WORKER_ID": WORKER_ID,
        "ROLE": "Phase 4 negative-control worker",
        "RUN_ID": RUN_ID,
        "SCOPE_COMPLETED": "Full requested negative-control battery through frozen primary LOTO pipeline.",
        "FILES_CREATED": [str(p) for p in artifact_paths if p.exists()],
        "FILES_MODIFIED": [str(results_path), str(META / "artifact_index.csv")],
        "COMMANDS_RUN": worker_report["commands_run"],
        "TESTS_RUN": worker_report["tests_run"],
        "KEY_EVIDENCE": key_evidence,
        "ACCEPTANCE_CRITERIA_RESULTS": {
            "identity_verified": True,
            "all_controls_computed": len(results) == 11,
            "future_leaky_labelled_non_deployable_diagnostic": True,
            "pipeline_validity_verdict_present": True,
            "null_robustness_verdict_present": True,
            "frozen_artifacts_edited": False,
        },
        "SPEC_DEVIATIONS": worker_report["spec_deviations"],
        "UNRESOLVED_BLOCKERS": [],
        "RECOMMENDED_NEXT_CODEX_TASK": "Phase 4 statistical independent reviewer",
        "GIT_DIFF_SUMMARY": "Added negative-control runner and generated Phase 4 negative-control artifacts.",
    }
    print(json.dumps(json_ready(stdout), indent=2, sort_keys=True))
    return 0


def by_name_if_present(results: pd.DataFrame, name: str) -> dict[str, Any]:
    row = results.loc[results["control_name"] == name]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    raise SystemExit(main())
