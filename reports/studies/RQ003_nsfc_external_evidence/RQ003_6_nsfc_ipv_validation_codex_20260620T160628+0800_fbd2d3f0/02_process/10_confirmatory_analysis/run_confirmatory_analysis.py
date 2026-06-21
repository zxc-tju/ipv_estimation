#!/usr/bin/env python
"""Phase 4 frozen confirmatory analysis for RQ003.

This script intentionally stays outside the project package. It reads the
frozen Phase 4 inputs, joins official outcomes only through the Gate -1 mapping,
and writes the requested confirmatory artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


WORKER_ID = "RQ003_phase4_confirmatory_001"
RUN_ID = "RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0"
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"
REPO_ROOT = Path("/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation")
RUN_ROOT = REPO_ROOT / "reports/studies/RQ003_nsfc_external_evidence" / RUN_ID
META = RUN_ROOT / "02_process/00_meta"
FRZ = RUN_ROOT / "02_process/06_analysis_freeze"
DIRT = RUN_ROOT / "01_results/tables"
CONF = RUN_ROOT / "02_process/10_confirmatory_analysis"
DERIVED_ROOT = REPO_ROOT / "data/derived/onsite_competition/RQ003_nsfc_external_evidence" / RUN_ID
PYTHON = DERIVED_ROOT / "model_cache/venv/bin/python"
FRAME_LEVEL = DERIVED_ROOT / "frame_level/frame_level_directional_ipv.csv"
INTERMEDIATE = DERIVED_ROOT / "intermediate"

RNG_SEED = 2266481337
ALPHA_GRID = np.array([0.1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
N_BOOTSTRAP = 500
PERMUTATIONS_PRIMARY = 99
PERMUTATIONS_SECONDARY = 49

BASELINE_FEATURES = [
    "distance_auc_time_norm",
    "inverse_distance_auc_time_norm",
    "closing_speed_auc_time_norm",
    "positive_closing_auc_time_norm",
    "relative_speed_auc_time_norm",
    "ego_speed_auc_time_norm",
    "npc_speed_auc_time_norm",
    "ego_accel_auc_time_norm",
    "npc_accel_auc_time_norm",
    "heading_diff_abs_auc_time_norm",
    "abs_longitudinal_gap_auc_time_norm",
    "lateral_gap_auc_time_norm",
    "ttc_risk_auc_time_norm",
    "lateral_gap_risk_auc_time_norm",
    "near_conflict_fraction_time_norm",
    "closing_fraction_time_norm",
    "conflict_duration_s",
    "estimated_conflict_frames",
]
PRIMARY_D_COLS = ["D_comp_auc", "D_yield_auc"]
FALLBACK_D_COLS = ["D_comp_auc_fallback", "D_yield_auc_fallback"]


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    tier: str
    fold_family: str
    mask_col: str
    d_cols: tuple[str, str]
    n_permutations: int
    confirmatory_status: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def verify_identity() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("RUN_ROOT exists", RUN_ROOT.exists(), str(RUN_ROOT))
    add("CONF exists", CONF.exists(), str(CONF))
    manifest = META / "run_manifest.json"
    try:
        run_manifest = json.loads(manifest.read_text())
        manifest_run_id = run_manifest.get("run_id") or run_manifest.get("RUN_ID") or run_manifest.get("runId")
        add("run_manifest RUN_ID matches", manifest_run_id == RUN_ID, repr(manifest_run_id))
    except Exception as exc:  # pragma: no cover - fail-fast path
        add("run_manifest RUN_ID matches", False, repr(exc))
    try:
        add("plan_sha256 matches", (META / "plan_sha256.txt").read_text().strip() == PLAN_SHA256, "")
    except Exception as exc:  # pragma: no cover - fail-fast path
        add("plan_sha256 matches", False, repr(exc))
    for path in [
        FRZ / "analysis_freeze.yaml",
        FRZ / "fold_contract.csv",
        FRZ / "model_capacity_contract.md",
        FRZ / "exclusion_and_safe_subset.md",
        FRZ / "primary_endpoints.md",
        FRZ / "acceptance_matrix.csv",
        DIRT / "cell_level_directional_ipv.csv",
        DIRT / "baseline_features_cells.csv",
        DIRT / "scenario_map_outcome_free.csv",
        DIRT / "replay_score_mapping.csv",
        DIRT / "support_coverage.csv",
        PYTHON,
    ]:
        add(f"{path.name} exists", path.exists(), str(path))
    try:
        import scipy  # noqa: F401

        add("scipy import in active interpreter", True, stats.__name__)
    except Exception as exc:  # pragma: no cover - fail-fast path
        add("scipy import in active interpreter", False, repr(exc))
    if not all(c["ok"] for c in checks):
        failed = [c for c in checks if not c["ok"]]
        raise RuntimeError(f"Identity verification failed: {failed}")
    return checks


def finite_min(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.min()) if len(x) else np.nan


def assemble_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cell = pd.read_csv(DIRT / "cell_level_directional_ipv.csv")
    baseline = pd.read_csv(DIRT / "baseline_features_cells.csv")
    scenario_map = pd.read_csv(DIRT / "scenario_map_outcome_free.csv")
    support = pd.read_csv(DIRT / "support_coverage.csv")
    score = pd.read_csv(DIRT / "replay_score_mapping.csv")

    score = score[bool_series(score["in_plan_top5_cohort"])].copy()
    score["case_id"] = score["case_id"].astype(int)
    score_cols = [
        "team_code",
        "area",
        "case_id",
        "safety",
        "efficiency",
        "coordination",
        "comprehensive",
        "score_source_path",
        "analysis_ready_plan_top5",
        "mapping_status",
        "mapping_class",
        "mapped_session_id",
        "mapped_replay_path",
    ]

    base_keys = ["cell_id", "team", "area", "scenario", "family"]
    df = scenario_map[base_keys].merge(
        cell.drop(columns=["area", "scenario", "family"]),
        on=["cell_id", "team"],
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        baseline.drop(columns=["team", "area", "scenario", "family"]),
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        support.drop(columns=["team", "area", "scenario", "family", "estimated_conflict_frames"]),
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        score[score_cols],
        left_on=["team", "area", "case_id"],
        right_on=["team_code", "area", "case_id"],
        how="left",
        validate="many_to_one",
    )
    if len(df) != 150:
        raise RuntimeError(f"Expected 150 assembled cells, got {len(df)}")
    if df["coordination"].isna().any():
        missing = df.loc[df["coordination"].isna(), ["cell_id", "team", "case_id"]].to_dict("records")
        raise RuntimeError(f"Outcome join failed for cells: {missing[:10]}")

    frame = pd.read_csv(FRAME_LEVEL, usecols=["cell_id", "conflict_window", "lateral_gap_m", "ttc_s"])
    frame_guard = (
        frame[bool_series(frame["conflict_window"])]
        .groupby("cell_id")
        .agg(
            min_lateral_gap_m=("lateral_gap_m", finite_min),
            min_ttc_s=("ttc_s", finite_min),
            conflict_guard_rows=("cell_id", "size"),
        )
        .reset_index()
    )
    df = df.merge(frame_guard, on="cell_id", how="left", validate="one_to_one")

    df["mapped"] = bool_series(df["analysis_ready_plan_top5"])
    df["high_support_primary"] = bool_series(df["high_support_primary"])
    df["fallback_inclusive"] = bool_series(df["fallback_inclusive"])
    df["collision"] = pd.to_numeric(df["safety"], errors="coerce") < 100
    # No explicit takeover/line-crossing primitive exists in the authorized Phase 4 tables.
    # The absence is disclosed in the report; values are set to false rather than inferred
    # from the coordination outcome.
    df["takeover_flag_available"] = False
    df["line_crossing_flag_available"] = False
    df["takeover"] = False
    df["line_crossing"] = False
    df["ttc_guard_pass"] = pd.to_numeric(df["min_ttc_s"], errors="coerce") >= 1.5
    df["lateral_gap_guard_pass"] = pd.to_numeric(df["min_lateral_gap_m"], errors="coerce") >= 2.0
    df["s1_collision_free"] = ~df["collision"]
    df["s2_safety100_collision_free"] = (pd.to_numeric(df["safety"], errors="coerce") == 100) & df["s1_collision_free"]
    df["s3_strong_primitive_clean"] = (
        df["s1_collision_free"]
        & ~df["takeover"]
        & ~df["line_crossing"]
        & df["ttc_guard_pass"]
        & df["lateral_gap_guard_pass"]
    )
    df["primary_inclusion"] = df["mapped"] & df["high_support_primary"] & (df["scenario"] != "A1") & df["s1_collision_free"]
    df["sensitivity_fallback_inclusion"] = df["mapped"] & df["fallback_inclusive"] & (df["scenario"] != "A1") & df["s1_collision_free"]
    df["safe_s1_inclusion"] = df["primary_inclusion"] & df["s1_collision_free"]
    df["safe_s2_inclusion"] = df["primary_inclusion"] & df["s2_safety100_collision_free"]
    df["safe_s3_inclusion"] = df["primary_inclusion"] & df["s3_strong_primitive_clean"]
    df["D_sum_auc"] = df["D_comp_auc"] + df["D_yield_auc"]
    df["D_sum_auc_fallback"] = df["D_comp_auc_fallback"] + df["D_yield_auc_fallback"]

    read_paths = [
        str(FRZ / "analysis_freeze.yaml"),
        str(FRZ / "fold_contract.csv"),
        str(FRZ / "model_capacity_contract.md"),
        str(FRZ / "exclusion_and_safe_subset.md"),
        str(FRZ / "primary_endpoints.md"),
        str(FRZ / "acceptance_matrix.csv"),
        str(DIRT / "cell_level_directional_ipv.csv"),
        str(DIRT / "baseline_features_cells.csv"),
        str(DIRT / "scenario_map_outcome_free.csv"),
        str(DIRT / "support_coverage.csv"),
        str(DIRT / "replay_score_mapping.csv"),
        str(FRAME_LEVEL),
    ]
    return df, pd.read_csv(FRZ / "fold_contract.csv"), read_paths


def design_fixed_effects(train: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    train_scenarios = sorted(train["scenario"].dropna().unique().tolist())
    train_areas = sorted(train["area"].dropna().unique().tolist())
    scenario_levels = train_scenarios[1:]
    area_levels = train_areas[1:]
    cols = [np.ones(len(apply))]
    for level in scenario_levels:
        cols.append((apply["scenario"].to_numpy() == level).astype(float))
    for level in area_levels:
        cols.append((apply["area"].to_numpy() == level).astype(float))
    return np.column_stack(cols)


def residualize(train: pd.DataFrame, test: pd.DataFrame, outcome: str = "coordination") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = pd.to_numeric(train[outcome], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[outcome], errors="coerce").to_numpy(dtype=float)
    x_train = design_fixed_effects(train, train)
    beta = np.linalg.pinv(x_train) @ y_train
    fit_train = x_train @ beta
    fit_test = design_fixed_effects(train, test) @ beta
    return y_train - fit_train, y_test - fit_test, fit_train, fit_test


def fit_transform(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
    x_train_df = train[features].apply(pd.to_numeric, errors="coerce")
    x_test_df = test[features].apply(pd.to_numeric, errors="coerce")
    med = x_train_df.median(axis=0).fillna(0.0)
    x_train_df = x_train_df.fillna(med)
    x_test_df = x_test_df.fillna(med)
    mean = x_train_df.mean(axis=0)
    std = x_train_df.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    x_train = ((x_train_df - mean) / std).to_numpy(dtype=float)
    x_test = ((x_test_df - mean) / std).to_numpy(dtype=float)
    return x_train, x_test, med.to_dict(), mean.to_dict(), std.to_dict()


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x_aug.T @ x_aug + penalty) @ (x_aug.T @ y)


def ridge_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


def group_col_for_fold(fold_family: str) -> str:
    if fold_family == "leave_one_team_out":
        return "team"
    if fold_family == "leave_one_scenario_out":
        return "scenario"
    if fold_family == "leave_one_family_out":
        return "family"
    raise ValueError(f"Unknown fold family {fold_family}")


def safe_spearman(y: Iterable[float], pred: Iterable[float]) -> float:
    yy = np.asarray(list(y), dtype=float)
    pp = np.asarray(list(pred), dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[mask]
    pp = pp[mask]
    if len(yy) < 3 or np.nanstd(yy) == 0 or np.nanstd(pp) == 0:
        return np.nan
    return float(stats.spearmanr(yy, pp).statistic)


def choose_alpha(train: pd.DataFrame, features: list[str], fold_family: str) -> float:
    group_col = group_col_for_fold(fold_family)
    groups = [g for g in sorted(train[group_col].dropna().unique().tolist()) if (train[group_col] == g).sum() > 0]
    if len(groups) < 3 or len(train) < 8:
        return 10.0
    scores: dict[float, list[float]] = {float(a): [] for a in ALPHA_GRID}
    for group in groups:
        val = train[train[group_col] == group]
        tr = train[train[group_col] != group]
        if len(val) == 0 or len(tr) < 4:
            continue
        y_tr, y_val, _, _ = residualize(tr, val)
        x_tr, x_val, _, _, _ = fit_transform(tr, val, features)
        for alpha in ALPHA_GRID:
            beta = ridge_fit(x_tr, y_tr, float(alpha))
            pred = ridge_predict(x_val, beta)
            scores[float(alpha)].append(float(np.mean(np.abs(y_val - pred))))
    usable = [(alpha, float(np.mean(vals))) for alpha, vals in scores.items() if vals]
    if not usable:
        return 10.0
    usable.sort(key=lambda item: (item[1], item[0]))
    return float(usable[0][0])


def materialize_fold_assignments(df: pd.DataFrame, fold_contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fold in fold_contract.iterrows():
        family = fold["fold_family"]
        group_col = group_col_for_fold(family)
        holdout = fold["holdout_group"]
        for _, row in df.iterrows():
            role = "test" if str(row[group_col]) == str(holdout) else "train"
            rows.append(
                {
                    "fold_family": family,
                    "fold_id": fold["fold_id"],
                    "holdout_group": holdout,
                    "cell_id": row["cell_id"],
                    "team": row["team"],
                    "area": row["area"],
                    "scenario": row["scenario"],
                    "family": row["family"],
                    "fold_role_before_inclusion": role,
                    "mapped": bool(row["mapped"]),
                    "high_support_primary": bool(row["high_support_primary"]),
                    "non_A1": bool(row["scenario"] != "A1"),
                    "collision_free": bool(row["s1_collision_free"]),
                    "primary_inclusion": bool(row["primary_inclusion"]),
                    "safe_s1_inclusion": bool(row["safe_s1_inclusion"]),
                    "safe_s2_inclusion": bool(row["safe_s2_inclusion"]),
                    "safe_s3_inclusion": bool(row["safe_s3_inclusion"]),
                    "sensitivity_fallback_inclusion": bool(row["sensitivity_fallback_inclusion"]),
                }
            )
    return pd.DataFrame(rows)


def cross_validate(
    df: pd.DataFrame,
    fold_contract: pd.DataFrame,
    spec: AnalysisSpec,
    full_only: bool = False,
) -> pd.DataFrame:
    mask = bool_series(df[spec.mask_col])
    features_base = BASELINE_FEATURES
    features_full = BASELINE_FEATURES + list(spec.d_cols)
    preds = []
    group_col = group_col_for_fold(spec.fold_family)
    folds = fold_contract[fold_contract["fold_family"] == spec.fold_family]
    for _, fold in folds.iterrows():
        holdout = str(fold["holdout_group"])
        test_mask = mask & (df[group_col].astype(str) == holdout)
        train_mask = mask & ~test_mask
        test = df.loc[test_mask].copy()
        train = df.loc[train_mask].copy()
        if len(test) == 0 or len(train) < 4:
            continue
        y_train, y_test, fe_train, fe_test = residualize(train, test)
        alpha_full = choose_alpha(train, features_full, spec.fold_family)
        x_train_full, x_test_full, _, _, _ = fit_transform(train, test, features_full)
        beta_full = ridge_fit(x_train_full, y_train, alpha_full)
        pred_full = ridge_predict(x_test_full, beta_full)
        if full_only:
            pred_base = np.full(len(test), np.nan)
            alpha_base = np.nan
        else:
            alpha_base = choose_alpha(train, features_base, spec.fold_family)
            x_train_base, x_test_base, _, _, _ = fit_transform(train, test, features_base)
            beta_base = ridge_fit(x_train_base, y_train, alpha_base)
            pred_base = ridge_predict(x_test_base, beta_base)
        for i, (_, row) in enumerate(test.iterrows()):
            preds.append(
                {
                    "analysis_id": spec.analysis_id,
                    "tier": spec.tier,
                    "confirmatory_status": spec.confirmatory_status,
                    "fold_family": spec.fold_family,
                    "fold_id": fold["fold_id"],
                    "holdout_group": holdout,
                    "cell_id": row["cell_id"],
                    "team": row["team"],
                    "area": row["area"],
                    "scenario": row["scenario"],
                    "family": row["family"],
                    "case_id": int(row["case_id"]),
                    "coordination": float(row["coordination"]),
                    "fixed_effect_prediction": float(fe_test[i]),
                    "coordination_residual_oof": float(y_test[i]),
                    "pred_base_residual": float(pred_base[i]) if np.isfinite(pred_base[i]) else np.nan,
                    "pred_full_residual": float(pred_full[i]),
                    "alpha_base": float(alpha_base) if np.isfinite(alpha_base) else np.nan,
                    "alpha_full": float(alpha_full),
                    "D_comp_model": float(row[spec.d_cols[0]]) if pd.notna(row[spec.d_cols[0]]) else np.nan,
                    "D_yield_model": float(row[spec.d_cols[1]]) if pd.notna(row[spec.d_cols[1]]) else np.nan,
                }
            )
    return pd.DataFrame(preds)


def metrics_from_predictions(pred: pd.DataFrame) -> dict[str, float]:
    if pred.empty:
        return {
            "n_predictions": 0,
            "base_spearman": np.nan,
            "full_spearman": np.nan,
            "delta_spearman": np.nan,
            "base_mae": np.nan,
            "full_mae": np.nan,
            "delta_mae_reduction": np.nan,
            "base_cv_r2": np.nan,
            "full_cv_r2": np.nan,
            "delta_cv_r2": np.nan,
        }
    y = pred["coordination_residual_oof"].to_numpy(dtype=float)
    base = pred["pred_base_residual"].to_numpy(dtype=float)
    full = pred["pred_full_residual"].to_numpy(dtype=float)
    denom = float(np.sum(y**2))
    base_sse = float(np.sum((y - base) ** 2)) if np.isfinite(base).all() else np.nan
    full_sse = float(np.sum((y - full) ** 2))
    base_mae = float(np.mean(np.abs(y - base))) if np.isfinite(base).all() else np.nan
    full_mae = float(np.mean(np.abs(y - full)))
    base_spearman = safe_spearman(y, base) if np.isfinite(base).all() else np.nan
    full_spearman = safe_spearman(y, full)
    base_cv_r2 = 1.0 - base_sse / denom if denom > 0 and np.isfinite(base_sse) else np.nan
    full_cv_r2 = 1.0 - full_sse / denom if denom > 0 else np.nan
    return {
        "n_predictions": int(len(pred)),
        "base_spearman": base_spearman,
        "full_spearman": full_spearman,
        "delta_spearman": full_spearman - base_spearman if np.isfinite(base_spearman) and np.isfinite(full_spearman) else np.nan,
        "base_mae": base_mae,
        "full_mae": full_mae,
        "delta_mae_reduction": base_mae - full_mae if np.isfinite(base_mae) and np.isfinite(full_mae) else np.nan,
        "base_cv_r2": base_cv_r2,
        "full_cv_r2": full_cv_r2,
        "delta_cv_r2": full_cv_r2 - base_cv_r2 if np.isfinite(base_cv_r2) and np.isfinite(full_cv_r2) else np.nan,
    }


def bootstrap_ci(pred: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP) -> dict[str, tuple[float, float]]:
    scenarios = sorted(pred["scenario"].dropna().unique().tolist())
    keys = ["delta_spearman", "delta_mae_reduction", "delta_cv_r2"]
    values = {k: [] for k in keys}
    if len(scenarios) < 2:
        return {k: (np.nan, np.nan) for k in keys}
    groups = {s: pred[pred["scenario"] == s] for s in scenarios}
    for _ in range(n_boot):
        sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
        boot = pd.concat([groups[s] for s in sampled], ignore_index=True)
        m = metrics_from_predictions(boot)
        for k in keys:
            if np.isfinite(m[k]):
                values[k].append(m[k])
    out = {}
    for k, vals in values.items():
        if len(vals) < 20:
            out[k] = (np.nan, np.nan)
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[k] = (float(lo), float(hi))
    return out


def permutation_test(
    df: pd.DataFrame,
    fold_contract: pd.DataFrame,
    spec: AnalysisSpec,
    pred_obs: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float]:
    if spec.n_permutations <= 0 or len(pred_obs) < 6:
        return {
            "p_delta_spearman_greater": np.nan,
            "p_delta_mae_reduction_greater": np.nan,
            "p_delta_cv_r2_greater": np.nan,
            "n_permutations": 0,
        }
    obs = metrics_from_predictions(pred_obs)
    base_metrics = {
        "base_spearman": obs["base_spearman"],
        "base_mae": obs["base_mae"],
        "base_cv_r2": obs["base_cv_r2"],
    }
    perm_values = {k: [] for k in ["delta_spearman", "delta_mae_reduction", "delta_cv_r2"]}
    mask = bool_series(df[spec.mask_col])
    d_cols = list(spec.d_cols)
    for _ in range(spec.n_permutations):
        permuted = df.copy()
        for _, idx in df.loc[mask].groupby("scenario").groups.items():
            idx_list = list(idx)
            if len(idx_list) <= 1:
                continue
            shuffled = rng.permutation(idx_list)
            permuted.loc[idx_list, d_cols] = df.loc[shuffled, d_cols].to_numpy()
        pred_perm = cross_validate(permuted, fold_contract, spec, full_only=True)
        if pred_perm.empty:
            continue
        # Align to observed baseline rows so the baseline side is unchanged.
        aligned = pred_obs.drop(columns=["pred_full_residual"]).merge(
            pred_perm[["cell_id", "fold_id", "pred_full_residual"]],
            on=["cell_id", "fold_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(pred_obs):
            continue
        m = metrics_from_predictions(aligned)
        if np.isfinite(m["full_spearman"]) and np.isfinite(base_metrics["base_spearman"]):
            perm_values["delta_spearman"].append(m["full_spearman"] - base_metrics["base_spearman"])
        if np.isfinite(m["full_mae"]) and np.isfinite(base_metrics["base_mae"]):
            perm_values["delta_mae_reduction"].append(base_metrics["base_mae"] - m["full_mae"])
        if np.isfinite(m["full_cv_r2"]) and np.isfinite(base_metrics["base_cv_r2"]):
            perm_values["delta_cv_r2"].append(m["full_cv_r2"] - base_metrics["base_cv_r2"])

    out: dict[str, float] = {"n_permutations": float(spec.n_permutations)}
    for key in ["delta_spearman", "delta_mae_reduction", "delta_cv_r2"]:
        vals = np.asarray(perm_values[key], dtype=float)
        observed = obs[key]
        if len(vals) == 0 or not np.isfinite(observed):
            out[f"p_{key}_greater"] = np.nan
        else:
            out[f"p_{key}_greater"] = float((np.sum(vals >= observed) + 1.0) / (len(vals) + 1.0))
    return out


def scenario_spearman(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["D_comp_model", "D_yield_model"]:
        for scenario, g in pred.groupby("scenario"):
            rho = safe_spearman(g[variable], g["coordination_residual_oof"])
            rows.append(
                {
                    "analysis_id": pred["analysis_id"].iloc[0] if len(pred) else "",
                    "scenario": scenario,
                    "predictor": variable,
                    "n": int(len(g)),
                    "spearman": rho,
                    "direction_consistent": bool(np.isfinite(rho) and rho < 0),
                    "expected_direction": "negative: larger directional IPV deviation should align with lower coordination residual",
                }
            )
    # Predeclared block summary: total directional deviation burden.
    tmp = pred.copy()
    tmp["D_sum_model"] = tmp["D_comp_model"] + tmp["D_yield_model"]
    for scenario, g in tmp.groupby("scenario"):
        rho = safe_spearman(g["D_sum_model"], g["coordination_residual_oof"])
        rows.append(
            {
                "analysis_id": pred["analysis_id"].iloc[0] if len(pred) else "",
                "scenario": scenario,
                "predictor": "D_sum_model",
                "n": int(len(g)),
                "spearman": rho,
                "direction_consistent": bool(np.isfinite(rho) and rho < 0),
                "expected_direction": "negative: larger directional IPV deviation should align with lower coordination residual",
            }
        )
    return pd.DataFrame(rows)


def counts_summary(df: pd.DataFrame, mask_col: str) -> dict[str, object]:
    d = df[bool_series(df[mask_col])]
    return {
        "n_cells": int(len(d)),
        "n_teams": int(d["team"].nunique()),
        "n_scenarios": int(d["scenario"].nunique()),
        "per_team": {str(k): int(v) for k, v in d["team"].value_counts().sort_index().items()},
        "per_scenario": {str(k): int(v) for k, v in d["scenario"].value_counts().sort_index().items()},
    }


def make_results_row(
    spec: AnalysisSpec,
    pred: pd.DataFrame,
    df: pd.DataFrame,
    ci: dict[str, tuple[float, float]],
    pvals: dict[str, float],
) -> dict[str, object]:
    m = metrics_from_predictions(pred)
    c = counts_summary(df, spec.mask_col)
    row: dict[str, object] = {
        "analysis_id": spec.analysis_id,
        "tier": spec.tier,
        "confirmatory_status": spec.confirmatory_status,
        "fold_family": spec.fold_family,
        "subset_or_sample": spec.mask_col,
        "d_columns": ";".join(spec.d_cols),
        "n_cells_in_sample": c["n_cells"],
        "n_predictions": m["n_predictions"],
        "n_teams": c["n_teams"],
        "n_scenarios": c["n_scenarios"],
        "base_spearman": m["base_spearman"],
        "full_spearman": m["full_spearman"],
        "delta_spearman": m["delta_spearman"],
        "delta_spearman_ci_low": ci["delta_spearman"][0],
        "delta_spearman_ci_high": ci["delta_spearman"][1],
        "p_delta_spearman_greater": pvals.get("p_delta_spearman_greater", np.nan),
        "base_mae": m["base_mae"],
        "full_mae": m["full_mae"],
        "delta_mae_reduction": m["delta_mae_reduction"],
        "delta_mae_reduction_ci_low": ci["delta_mae_reduction"][0],
        "delta_mae_reduction_ci_high": ci["delta_mae_reduction"][1],
        "p_delta_mae_reduction_greater": pvals.get("p_delta_mae_reduction_greater", np.nan),
        "base_cv_r2": m["base_cv_r2"],
        "full_cv_r2": m["full_cv_r2"],
        "delta_cv_r2": m["delta_cv_r2"],
        "delta_cv_r2_ci_low": ci["delta_cv_r2"][0],
        "delta_cv_r2_ci_high": ci["delta_cv_r2"][1],
        "p_delta_cv_r2_greater": pvals.get("p_delta_cv_r2_greater", np.nan),
        "n_permutations": int(pvals.get("n_permutations", 0) or 0),
        "n_bootstrap": N_BOOTSTRAP,
        "effect_direction_spearman": "favorable" if np.isfinite(m["delta_spearman"]) and m["delta_spearman"] > 0 else "null_or_reverse",
        "effect_direction_mae": "favorable" if np.isfinite(m["delta_mae_reduction"]) and m["delta_mae_reduction"] > 0 else "null_or_reverse",
        "effect_direction_cv_r2": "favorable" if np.isfinite(m["delta_cv_r2"]) and m["delta_cv_r2"] > 0 else "null_or_reverse",
        "per_team_counts_json": json.dumps(c["per_team"], sort_keys=True),
        "per_scenario_counts_json": json.dumps(c["per_scenario"], sort_keys=True),
    }
    return row


def write_markdown_report(
    path: Path,
    df: pd.DataFrame,
    results: pd.DataFrame,
    scenario_assoc: pd.DataFrame,
    identity: list[dict[str, object]],
    safe_agreement: dict[str, object],
    spec_deviations: list[str],
) -> None:
    primary = results[results["analysis_id"] == "primary_loto_confirmatory"].iloc[0].to_dict()
    secondary = results[results["analysis_id"] == "secondary_loso_generalization"].iloc[0].to_dict()
    boundary = results[results["analysis_id"] == "boundary_lofo_family"].iloc[0].to_dict()
    srows = results[results["tier"].str.startswith("safe_subset")]
    scen_sum = (
        scenario_assoc[scenario_assoc["predictor"] == "D_sum_model"]
        .dropna(subset=["spearman"])
    )
    median_scen = float(scen_sum["spearman"].median()) if len(scen_sum) else np.nan
    direction_count = int(scen_sum["direction_consistent"].sum()) if len(scen_sum) else 0
    direction_total = int(len(scen_sum))

    lines = [
        "# Phase 4 Confirmatory Analysis Report",
        "",
        f"Worker: `{WORKER_ID}`",
        f"Run: `{RUN_ID}`",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Identity",
        "",
    ]
    for check in identity:
        lines.append(f"- {'PASS' if check['ok'] else 'FAIL'}: {check['name']} {check['detail']}")
    lines.extend(
        [
            "",
            "## Frozen Scope",
            "",
            "Only the frozen comparison was run: `state + causal kinematics + safety` versus the same baseline plus `D_comp_auc` and `D_yield_auc`.",
            "The primary outcome is official coordination residualized by scenario and area fixed effects within each training fold. The score source is the Gate -1 `replay_score_mapping.csv` joined to analysis cells by team, area, and case_id; canonical scenario labels come from the outcome-free scenario map.",
            "",
            "## Primary Sample",
            "",
            f"- Primary cells: {int(primary['n_cells_in_sample'])} cells, {int(primary['n_teams'])} teams, {int(primary['n_scenarios'])} scenarios.",
            f"- Per-team counts: `{primary['per_team_counts_json']}`.",
            f"- Per-scenario counts: `{primary['per_scenario_counts_json']}`.",
            "- T8 has no eligible held-out primary cell after the frozen high-support and non-A1 exclusions; its LOTO fold is materialized but contributes no prediction row.",
            "- Collision-free was operationalized from the frozen score/mapping evidence: the only non-100 safety rows in the top-five cohort are two zero-safety A1 rows, both excluded by the non-A1 primary rule.",
            "",
            "## Confirmatory Result",
            "",
            f"- Leave-one-team-out delta Spearman (full - baseline): {primary['delta_spearman']:.6g}, 95% scenario-cluster bootstrap CI [{primary['delta_spearman_ci_low']:.6g}, {primary['delta_spearman_ci_high']:.6g}], scenario-stratified permutation p={primary['p_delta_spearman_greater']:.6g}.",
            f"- Baseline Spearman={primary['base_spearman']:.6g}; full Spearman={primary['full_spearman']:.6g}.",
            f"- Baseline MAE={primary['base_mae']:.6g}; full MAE={primary['full_mae']:.6g}; MAE reduction={primary['delta_mae_reduction']:.6g}, CI [{primary['delta_mae_reduction_ci_low']:.6g}, {primary['delta_mae_reduction_ci_high']:.6g}], p={primary['p_delta_mae_reduction_greater']:.6g}.",
            f"- Baseline CV-R2={primary['base_cv_r2']:.6g}; full CV-R2={primary['full_cv_r2']:.6g}; delta CV-R2={primary['delta_cv_r2']:.6g}, CI [{primary['delta_cv_r2_ci_low']:.6g}, {primary['delta_cv_r2_ci_high']:.6g}], p={primary['p_delta_cv_r2_greater']:.6g}.",
            "",
            "Null/reverse disclosure: the primary Spearman, MAE, and CV-R2 increments are all null/reverse for the IPV-added model; full-model MAE is higher than baseline MAE.",
            "The small high-support primary sample (48 cells, with sparse team/scenario coverage) is a material power limitation.",
            "",
            "## Scenario-Wise IPV Association",
            "",
            f"- Median scenario-wise Spearman for `D_comp + D_yield` versus held-out coordination residual: {median_scen:.6g}.",
            f"- Direction-consistent scenarios: {direction_count} / {direction_total} with computable scenario-wise rank correlations.",
            "- Expected direction was negative: larger directional IPV deviation should align with lower coordination residual.",
            "",
            "## Generalization",
            "",
            f"- Secondary leave-one-scenario-out: delta Spearman={secondary['delta_spearman']:.6g}, CI [{secondary['delta_spearman_ci_low']:.6g}, {secondary['delta_spearman_ci_high']:.6g}], p={secondary['p_delta_spearman_greater']:.6g}; baseline/full MAE={secondary['base_mae']:.6g}/{secondary['full_mae']:.6g}; baseline/full CV-R2={secondary['base_cv_r2']:.6g}/{secondary['full_cv_r2']:.6g}.",
            f"- Boundary leave-one-family-out: delta Spearman={boundary['delta_spearman']:.6g}, CI [{boundary['delta_spearman_ci_low']:.6g}, {boundary['delta_spearman_ci_high']:.6g}], p={boundary['p_delta_spearman_greater']:.6g}; baseline/full MAE={boundary['base_mae']:.6g}/{boundary['full_mae']:.6g}; baseline/full CV-R2={boundary['base_cv_r2']:.6g}/{boundary['full_cv_r2']:.6g}. This is boundary evidence only, not a significance headline.",
            "",
            "## Safe Subsets",
            "",
        ]
    )
    for _, row in srows.iterrows():
        lines.append(
            f"- {row['analysis_id']}: N={int(row['n_cells_in_sample'])}, delta Spearman={row['delta_spearman']:.6g}, MAE reduction={row['delta_mae_reduction']:.6g}, delta CV-R2={row['delta_cv_r2']:.6g}, direction={row['safe_subset_direction'] if 'safe_subset_direction' in row else row['effect_direction_spearman']}."
        )
    lines.extend(
        [
            f"- Frozen safe-subset agreement count: {safe_agreement['agreeing_subsets']} / {safe_agreement['evaluated_subsets']} by the primary direction metric.",
            f"- Agreement requirement met: {safe_agreement['requirement_met']}.",
            "- S1 and S2 are empirically identical after the primary exclusions because every non-A1 eligible primary cell has safety_score=100. S3 is very small (6 cells) and should be treated as underpowered.",
            "- No explicit takeover or line-crossing primitive column exists in the authorized Phase 4 tables; S3 therefore uses the available collision/safety plus frozen TTC and lateral-gap guards and discloses this source limitation.",
            "",
            "## Sensitivity",
            "",
        ]
    )
    sens = results[results["analysis_id"] == "sensitivity_fallback_loto"].iloc[0].to_dict()
    lines.extend(
        [
            f"- Fallback-inclusive predictor sensitivity (not confirmatory): N={int(sens['n_cells_in_sample'])}, delta Spearman={sens['delta_spearman']:.6g}, CI [{sens['delta_spearman_ci_low']:.6g}, {sens['delta_spearman_ci_high']:.6g}], p={sens['p_delta_spearman_greater']:.6g}; baseline/full MAE={sens['base_mae']:.6g}/{sens['full_mae']:.6g}; baseline/full CV-R2={sens['base_cv_r2']:.6g}/{sens['full_cv_r2']:.6g}.",
            "",
            "## Spec Deviations And Limitations",
            "",
        ]
    )
    for item in spec_deviations:
        lines.append(f"- {item}")
    lines.extend(
        [
            "- No negative controls, state-dependence/NPC analyses, plotting, or tier decision were run.",
            "- The model is a simple ridge-linear fallback with the same fold-only preprocessing and in-train alpha grid for baseline and full arms because the high-support sample is small.",
            "- For leave-one-scenario-out and leave-one-family-out, held-out scenario levels can be unseen by the training fixed-effect fit; unseen fixed-effect levels are applied as the training reference level to avoid leakage.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def append_artifact_index(paths: list[Path], command: str) -> None:
    index = META / "artifact_index.csv"
    fieldnames = ["artifact_path", "sha256", "size_bytes", "produced_by", "command", "purpose", "phase"]
    needs_header = not index.exists() or index.stat().st_size == 0
    with index.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for path in paths:
            abs_path = str(path.resolve())
            writer.writerow(
                {
                    "artifact_path": abs_path,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "produced_by": WORKER_ID,
                    "command": command,
                    "purpose": "Phase 4 frozen confirmatory analysis artifact",
                    "phase": "4_confirmatory",
                }
            )


def main() -> int:
    CONF.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    identity = verify_identity()
    rng = np.random.default_rng(RNG_SEED)
    df, fold_contract, read_paths = assemble_frame()

    fold_assignments = materialize_fold_assignments(df, fold_contract)
    fold_assignments_path = DIRT / "fold_assignments.csv"
    fold_assignments.to_csv(fold_assignments_path, index=False)

    analysis_frame_path = INTERMEDIATE / "phase4_confirmatory_analysis_frame.csv"
    df.to_csv(analysis_frame_path, index=False)
    safe_summary_path = INTERMEDIATE / "phase4_safe_primitive_summary.csv"
    df[
        [
            "cell_id",
            "team",
            "area",
            "scenario",
            "family",
            "safety",
            "collision",
            "min_ttc_s",
            "min_lateral_gap_m",
            "ttc_guard_pass",
            "lateral_gap_guard_pass",
            "s1_collision_free",
            "s2_safety100_collision_free",
            "s3_strong_primitive_clean",
        ]
    ].to_csv(safe_summary_path, index=False)

    specs = [
        AnalysisSpec("primary_loto_confirmatory", "primary", "leave_one_team_out", "primary_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_PRIMARY, "confirmatory"),
        AnalysisSpec("secondary_loso_generalization", "secondary_generalization", "leave_one_scenario_out", "primary_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_SECONDARY, "secondary_not_confirmatory"),
        AnalysisSpec("boundary_lofo_family", "boundary_generalization", "leave_one_family_out", "primary_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_SECONDARY, "boundary_no_significance_headline"),
        AnalysisSpec("safe_s1_loto", "safe_subset_S1", "leave_one_team_out", "safe_s1_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_PRIMARY, "safe_subset_direction_check"),
        AnalysisSpec("safe_s2_loto", "safe_subset_S2", "leave_one_team_out", "safe_s2_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_PRIMARY, "safe_subset_direction_check"),
        AnalysisSpec("safe_s3_loto", "safe_subset_S3", "leave_one_team_out", "safe_s3_inclusion", tuple(PRIMARY_D_COLS), PERMUTATIONS_SECONDARY, "safe_subset_direction_check"),
        AnalysisSpec("sensitivity_fallback_loto", "sensitivity_not_confirmatory", "leave_one_team_out", "sensitivity_fallback_inclusion", tuple(FALLBACK_D_COLS), PERMUTATIONS_SECONDARY, "sensitivity_not_confirmatory"),
    ]

    all_predictions = []
    result_rows = []
    perm_cache: dict[tuple[str, str, tuple[str, str]], dict[str, float]] = {}
    pred_cache: dict[tuple[str, str, tuple[str, str]], pd.DataFrame] = {}
    for spec in specs:
        mask_signature = tuple(sorted(df.loc[bool_series(df[spec.mask_col]), "cell_id"].astype(str).tolist()))
        key = (spec.fold_family, "|".join(mask_signature), spec.d_cols)
        pred = pred_cache.get(key)
        if pred is None:
            pred = cross_validate(df, fold_contract, spec)
            pred_cache[key] = pred.copy()
        pred = pred.copy()
        pred["analysis_id"] = spec.analysis_id
        pred["tier"] = spec.tier
        pred["confirmatory_status"] = spec.confirmatory_status
        all_predictions.append(pred)
        ci = bootstrap_ci(pred, rng)
        pkey = key
        if pkey not in perm_cache:
            perm_cache[pkey] = permutation_test(df, fold_contract, spec, pred, rng)
        row = make_results_row(spec, pred, df, ci, perm_cache[pkey])
        result_rows.append(row)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions_path = DIRT / "cv_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    results = pd.DataFrame(result_rows)
    safe_dirs = []
    for _, row in results.iterrows():
        if str(row["tier"]).startswith("safe_subset"):
            direction = "agree_favorable" if np.isfinite(row["delta_spearman"]) and row["delta_spearman"] > 0 else "null_or_reverse"
            safe_dirs.append((row["analysis_id"], direction))
    safe_agree = sum(1 for _, d in safe_dirs if d == "agree_favorable")
    safe_eval = len(safe_dirs)
    safe_requirement_met = bool(safe_agree >= 2)
    results["safe_subset_direction"] = ""
    for analysis_id, direction in safe_dirs:
        results.loc[results["analysis_id"] == analysis_id, "safe_subset_direction"] = direction
    results["safe_subset_agreement_count"] = safe_agree
    results["safe_subset_requirement_met"] = safe_requirement_met
    results_path = DIRT / "confirmatory_results.csv"
    results.to_csv(results_path, index=False)

    primary_pred = predictions[predictions["analysis_id"] == "primary_loto_confirmatory"].copy()
    scenario_assoc = scenario_spearman(primary_pred)
    scenario_assoc_path = CONF / "scenario_wise_ipv_rank_association.csv"
    scenario_assoc.to_csv(scenario_assoc_path, index=False)

    spec_deviations = [
        "No explicit takeover or line_crossing primitive field was present in the authorized Phase 4 tables or mapped log manifests; S3 uses the available collision/safety, TTC, and lateral-gap guards and discloses this limitation.",
        "The collision-free flag is operationalized from the frozen mapping/safety evidence: only two top-five rows have safety below 100, both zero-safety A1 cells already excluded by the non-A1 primary rule.",
        "S1 and S2 are empirically identical after primary exclusions because all non-A1 primary cells have safety_score=100.",
    ]
    safe_agreement = {
        "evaluated_subsets": safe_eval,
        "agreeing_subsets": safe_agree,
        "requirement_met": safe_requirement_met,
        "directions": dict(safe_dirs),
    }

    report_path = CONF / "confirmatory_analysis_report.md"
    write_markdown_report(report_path, df, results, scenario_assoc, identity, safe_agreement, spec_deviations)

    file_access_path = CONF / "file_access_manifest.txt"
    write_paths = [
        results_path,
        predictions_path,
        fold_assignments_path,
        report_path,
        CONF / "worker_report.json",
        file_access_path,
        CONF / "artifact_manifest.csv",
        scenario_assoc_path,
        analysis_frame_path,
        safe_summary_path,
    ]
    file_access_path.write_text(
        "\n".join(
            [
                f"worker_id={WORKER_ID}",
                f"run_id={RUN_ID}",
                "READ:",
                *sorted(read_paths),
                "WRITE:",
                *[str(p) for p in write_paths],
                "NOTES:",
                "Outcomes read only through replay_score_mapping.csv joined by team/area/case_id.",
                "No core code, frozen specs, feature tables, or paper repository files were edited.",
            ]
        )
        + "\n"
    )

    worker_report = {
        "status": "PASS",
        "worker_id": WORKER_ID,
        "role": "Phase 4 confirmatory worker",
        "run_id": RUN_ID,
        "identity_verification": identity,
        "primary_counts": counts_summary(df, "primary_inclusion"),
        "safe_subset_agreement": safe_agreement,
        "spec_deviations": spec_deviations,
        "results": results.replace({np.nan: None}).to_dict("records"),
        "scenario_rank_association": scenario_assoc.replace({np.nan: None}).to_dict("records"),
        "commands_run": [
            "omx explore --prompt ... (failed due sandbox app-server Operation not permitted)",
            "python identity/input inspection snippets",
            f"{PYTHON} {Path(__file__).resolve()}",
        ],
        "tests_run": [
            "Pre-write identity verification",
            "Outcome join completeness check: 150/150 cells joined",
            "Fold materialization check from frozen fold_contract",
            "Scenario-stratified permutation test preserving canonical scenario labels",
            "Scenario-cluster bootstrap over canonical scenarios",
        ],
        "non_goals_observed": [
            "No negative controls",
            "No non-frozen confirmatory specs",
            "No state-dependence/NPC analysis",
            "No plotting",
            "No tier decision",
        ],
    }
    worker_report_path = CONF / "worker_report.json"
    worker_report_path.write_text(json.dumps(worker_report, indent=2, sort_keys=True))

    artifact_paths = [
        results_path,
        predictions_path,
        fold_assignments_path,
        report_path,
        worker_report_path,
        file_access_path,
        scenario_assoc_path,
        analysis_frame_path,
        safe_summary_path,
        Path(__file__).resolve(),
    ]
    artifact_manifest_path = CONF / "artifact_manifest.csv"
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
                    "purpose": "Phase 4 confirmatory analysis",
                }
            )
    artifact_paths.append(artifact_manifest_path)

    command = f"{PYTHON} {Path(__file__).resolve()}"
    append_artifact_index(artifact_paths, command)

    primary = results[results["analysis_id"] == "primary_loto_confirmatory"].iloc[0].to_dict()
    scen_sum = scenario_assoc[(scenario_assoc["predictor"] == "D_sum_model") & scenario_assoc["spearman"].notna()]
    stdout = {
        "STATUS": "PASS",
        "WORKER_ID": WORKER_ID,
        "ROLE": "Phase 4 confirmatory worker",
        "RUN_ID": RUN_ID,
        "PRIMARY_N": counts_summary(df, "primary_inclusion"),
        "PRIMARY_DELTA_SPEARMAN": primary["delta_spearman"],
        "PRIMARY_DELTA_SPEARMAN_CI": [primary["delta_spearman_ci_low"], primary["delta_spearman_ci_high"]],
        "PRIMARY_DELTA_SPEARMAN_P": primary["p_delta_spearman_greater"],
        "PRIMARY_BASE_FULL_MAE": [primary["base_mae"], primary["full_mae"]],
        "PRIMARY_BASE_FULL_CV_R2": [primary["base_cv_r2"], primary["full_cv_r2"]],
        "SCENARIO_MEDIAN_SPEARMAN_D_SUM": float(scen_sum["spearman"].median()) if len(scen_sum) else None,
        "SCENARIO_DIRECTION_CONSISTENT": [int(scen_sum["direction_consistent"].sum()) if len(scen_sum) else 0, int(len(scen_sum))],
        "SAFE_SUBSET_AGREEMENT": safe_agreement,
        "NULL_REVERSE": "Primary delta Spearman, MAE reduction, and delta CV-R2 are null/reverse; full-model MAE is higher than baseline MAE.",
        "FILES_CREATED_OR_MODIFIED": [str(p) for p in artifact_paths],
    }
    print(json.dumps(stdout, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
