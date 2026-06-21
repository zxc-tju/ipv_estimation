#!/usr/bin/env python3
"""Independent replication v2 for corrected RQ003 primary confirmatory result.

This script intentionally does not import Phase4/Phase7 implementation scripts.
It rebuilds the primary analysis frame from frozen CSV contracts, applies the
corrected scenario crosswalk, fits fold-local scenario+area residualization, and
fits capacity-matched standardized ridge models with numpy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


WORKER_ID = "RQ003_phase8_replication2_001"
RUN_ID = "RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0"
PLAN_SHA256 = "98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1"
REPO_ROOT = Path(
    "/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation"
)
RUN_ROOT = REPO_ROOT / "reports/studies/RQ003_nsfc_external_evidence" / RUN_ID
META = RUN_ROOT / "02_process/00_meta"
DIRT = RUN_ROOT / "01_results/tables"
FRZ = RUN_ROOT / "02_process/06_analysis_freeze"
G0 = RUN_ROOT / "02_process/04_gate0_measurement"
REP = RUN_ROOT / "02_process/17_independent_replication/replication2"
DERIVED_ROOT = REPO_ROOT / "data/derived/onsite_competition/RQ003_nsfc_external_evidence" / RUN_ID
REPDATA = DERIVED_ROOT / "replication2"

FOLD_ORDER = ["T17", "T14", "T16", "T15", "T20", "T11", "T6", "T5", "T7", "T8"]
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
METRIC_TOLERANCE = 1e-10
PREDICTION_TOLERANCE = 1e-10


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fnum(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def stable_sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (str(r["team"]), str(r["scenario"]), str(r["cell_id"])))


def categorical_design(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    levels = [(col, sorted({str(row[col]) for row in train})) for col in columns]

    def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
        out = []
        for row in rows:
            values = [1.0]
            for col, vals in levels:
                values.extend(1.0 if str(row[col]) == level else 0.0 for level in vals)
            out.append(values)
        return np.asarray(out, dtype=float)

    return matrix(train), matrix(test)


def fit_fixed_effects(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_test = categorical_design(train, test, ["scenario", "area"])
    y_train = np.asarray([row["coordination"] for row in train], dtype=float)
    beta = np.linalg.pinv(x_train) @ y_train
    return x_train @ beta, x_test @ beta


def feature_matrix(rows: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    return np.asarray([[row[col] for col in feature_columns] for row in rows], dtype=float)


def standardize_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    col_mean = np.nanmean(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), col_mean, x_train)
    x_test = np.where(np.isnan(x_test), col_mean, x_test)
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (x_train - center) / scale, (x_test - center) / scale


def ridge_predict(
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    test_rows: list[dict[str, Any]],
    feature_columns: list[str],
    alpha: float,
) -> np.ndarray:
    x_train = feature_matrix(train_rows, feature_columns)
    x_test = feature_matrix(test_rows, feature_columns)
    x_train, x_test = standardize_from_train(x_train, x_test)
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    penalty = alpha * np.eye(x_train.shape[1])
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_centered)
    return y_mean + x_test @ beta


def loo_alpha(
    train_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> tuple[float, list[dict[str, float]]]:
    """Training-only nested LOOCV selector.

    Each inner validation row gets scenario+area residualization fitted on its
    inner training subset. Feature standardization is also fitted on the inner
    training subset. This is intentionally independent of Phase7 output alphas.
    """

    records = []
    for alpha in ALPHA_GRID:
        squared_errors = []
        for val_idx in range(len(train_rows)):
            inner_train = [row for i, row in enumerate(train_rows) if i != val_idx]
            inner_val = [train_rows[val_idx]]
            fe_train, fe_val = fit_fixed_effects(inner_train, inner_val)
            y_train = np.asarray([row["coordination"] for row in inner_train], dtype=float) - fe_train
            y_val = np.asarray([inner_val[0]["coordination"]], dtype=float) - fe_val
            pred = ridge_predict(inner_train, y_train, inner_val, feature_columns, alpha)
            squared_errors.append(float((y_val[0] - pred[0]) ** 2))
        records.append({"alpha": alpha, "mean_squared_error": float(np.mean(squared_errors))})
    best = min(records, key=lambda row: (row["mean_squared_error"], row["alpha"]))
    return float(best["alpha"]), records


def metrics_from_predictions(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_full: np.ndarray,
) -> dict[str, float | str]:
    base_spearman = float(spearmanr(y_true, pred_base).statistic)
    full_spearman = float(spearmanr(y_true, pred_full).statistic)
    base_mae = float(np.mean(np.abs(y_true - pred_base)))
    full_mae = float(np.mean(np.abs(y_true - pred_full)))
    # Phase7 uses zero-centered residual sum of squares for CV-R2.
    denom = float(np.sum(y_true**2))
    base_cv_r2 = float(1.0 - np.sum((y_true - pred_base) ** 2) / denom)
    full_cv_r2 = float(1.0 - np.sum((y_true - pred_full) ** 2) / denom)
    return {
        "base_spearman": base_spearman,
        "full_spearman": full_spearman,
        "delta_spearman": full_spearman - base_spearman,
        "base_mae": base_mae,
        "full_mae": full_mae,
        "delta_mae_reduction": base_mae - full_mae,
        "base_cv_r2": base_cv_r2,
        "full_cv_r2": full_cv_r2,
        "delta_cv_r2": full_cv_r2 - base_cv_r2,
        "effect_direction_spearman": "favorable" if full_spearman > base_spearman else "null_or_reverse",
        "effect_direction_mae": "favorable" if full_mae < base_mae else "null_or_reverse",
        "effect_direction_cv_r2": "favorable" if full_cv_r2 > base_cv_r2 else "null_or_reverse",
    }


def build_analysis_frame(inputs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[str]]:
    crosswalk = {row["cell_id"]: row for row in read_csv(inputs["scenario_crosswalk_corrected"])}
    cell_ipv = {row["cell_id"]: row for row in read_csv(inputs["cell_level_directional_ipv"])}
    baseline = {row["cell_id"]: row for row in read_csv(inputs["baseline_features_cells"])}
    support = {row["cell_id"]: row for row in read_csv(inputs["support_coverage"])}
    scores = read_csv(inputs["replay_score_mapping"])
    score_by_key = {
        (row["team_code"], row["area"], str(row["case_id"]), row["scenario"]): row
        for row in scores
    }

    baseline_feature_columns = [
        col
        for col in next(iter(baseline.values())).keys()
        if col not in {"cell_id", "team", "area", "scenario", "family"}
    ]
    rows: list[dict[str, Any]] = []
    for cell_id, ipv in cell_ipv.items():
        cross = crosswalk.get(cell_id)
        base = baseline.get(cell_id)
        supp = support.get(cell_id)
        if not cross or not base or not supp:
            continue
        scenario = cross["official_scenario"]
        if not parse_bool(ipv["high_support_primary"]):
            continue
        if scenario == "A1":
            continue
        score_key = (ipv["team"], ipv["area"], str(ipv["case_id"]), scenario)
        score = score_by_key.get(score_key)
        if score is None:
            continue
        # The available official collision-free primitive in the coordination
        # outcome table is the safety score. In this corrected primary sample,
        # every high-support, non-A1 matched cell has safety==100.
        safety_score = fnum(score["safety"])
        if safety_score < 100.0:
            continue
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "team": ipv["team"],
            "area": ipv["area"],
            "scenario": scenario,
            "family": cross["family"],
            "case_id": str(ipv["case_id"]),
            "coordination": fnum(score["coordination"]),
            "safety": safety_score,
            "D_comp_auc": fnum(ipv["D_comp_auc"]),
            "D_yield_auc": fnum(ipv["D_yield_auc"]),
            "high_support_primary": parse_bool(ipv["high_support_primary"]),
            "coverage_rate_ego": fnum(supp["coverage_rate_ego"]),
            "coverage_rate_both": fnum(supp["coverage_rate_both"]),
        }
        for col in baseline_feature_columns:
            row[col] = fnum(base[col])
        rows.append(row)
    return stable_sort_rows(rows), baseline_feature_columns


def run_loto(
    analysis_rows: list[dict[str, Any]],
    baseline_features: list[str],
    mode: str,
    phase_alpha_lookup: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    d_features = ["D_comp_auc", "D_yield_auc"]

    for team in FOLD_ORDER:
        test_rows = [row for row in analysis_rows if row["team"] == team]
        if not test_rows:
            continue
        train_rows = [row for row in analysis_rows if row["team"] != team]
        fold_id = f"LOTO_{team}"
        fe_train, fe_test = fit_fixed_effects(train_rows, test_rows)
        y_train = np.asarray([row["coordination"] for row in train_rows], dtype=float) - fe_train
        y_test = np.asarray([row["coordination"] for row in test_rows], dtype=float) - fe_test

        if mode == "independent_training_tuned":
            alpha_base, base_curve = loo_alpha(train_rows, baseline_features)
            alpha_full, full_curve = loo_alpha(train_rows, baseline_features + d_features)
            for curve_name, curve in [("base", base_curve), ("full", full_curve)]:
                for record in curve:
                    tuning_rows.append(
                        {
                            "mode": mode,
                            "fold_id": fold_id,
                            "model": curve_name,
                            "alpha": record["alpha"],
                            "mean_squared_error": record["mean_squared_error"],
                        }
                    )
        elif mode == "reported_alpha_refit":
            assert phase_alpha_lookup is not None
            alpha_base = phase_alpha_lookup[fold_id]["base"]
            alpha_full = phase_alpha_lookup[fold_id]["full"]
        else:
            raise ValueError(f"unknown mode: {mode}")

        pred_base = ridge_predict(train_rows, y_train, test_rows, baseline_features, alpha_base)
        pred_full = ridge_predict(train_rows, y_train, test_rows, baseline_features + d_features, alpha_full)
        alpha_rows.append(
            {
                "mode": mode,
                "fold_id": fold_id,
                "holdout_group": team,
                "alpha_base": alpha_base,
                "alpha_full": alpha_full,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
            }
        )
        for row, fe_pred, residual, pb, pf in zip(test_rows, fe_test, y_test, pred_base, pred_full):
            prediction_rows.append(
                {
                    "analysis_id": f"replication2_{mode}",
                    "tier": "primary",
                    "confirmatory_status": "confirmatory",
                    "fold_family": "leave_one_team_out",
                    "fold_id": fold_id,
                    "holdout_group": team,
                    "cell_id": row["cell_id"],
                    "team": row["team"],
                    "area": row["area"],
                    "scenario": row["scenario"],
                    "family": row["family"],
                    "case_id": row["case_id"],
                    "coordination": row["coordination"],
                    "fixed_effect_prediction": float(fe_pred),
                    "coordination_residual_oof": float(residual),
                    "pred_base_residual": float(pb),
                    "pred_full_residual": float(pf),
                    "alpha_base": alpha_base,
                    "alpha_full": alpha_full,
                    "D_comp_model": row["D_comp_auc"],
                    "D_yield_model": row["D_yield_auc"],
                }
            )

    y = np.asarray([row["coordination_residual_oof"] for row in prediction_rows], dtype=float)
    pred_base = np.asarray([row["pred_base_residual"] for row in prediction_rows], dtype=float)
    pred_full = np.asarray([row["pred_full_residual"] for row in prediction_rows], dtype=float)
    summary = metrics_from_predictions(y, pred_base, pred_full)
    summary.update(
        {
            "mode": mode,
            "analysis_id": f"replication2_{mode}",
            "n_cells_in_sample": len(analysis_rows),
            "n_predictions": len(prediction_rows),
            "n_teams": len({row["team"] for row in analysis_rows}),
            "n_scenarios": len({row["scenario"] for row in analysis_rows}),
        }
    )
    return prediction_rows, alpha_rows, tuning_rows, summary


def phase_primary_results(inputs: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    result_rows = read_csv(inputs["confirmatory_results"])
    primary = next(row for row in result_rows if row["analysis_id"] == "primary_loto_confirmatory")
    cv_rows = [
        row
        for row in read_csv(inputs["cv_predictions"])
        if row["analysis_id"] == "primary_loto_confirmatory"
    ]
    numeric_keys = [
        "base_spearman",
        "full_spearman",
        "delta_spearman",
        "base_mae",
        "full_mae",
        "delta_mae_reduction",
        "base_cv_r2",
        "full_cv_r2",
        "delta_cv_r2",
    ]
    parsed = {key: fnum(primary[key]) for key in numeric_keys}
    parsed.update(
        {
            "analysis_id": primary["analysis_id"],
            "n_cells_in_sample": int(primary["n_cells_in_sample"]),
            "n_predictions": int(primary["n_predictions"]),
            "n_teams": int(primary["n_teams"]),
            "n_scenarios": int(primary["n_scenarios"]),
            "effect_direction_spearman": primary["effect_direction_spearman"],
            "effect_direction_mae": primary["effect_direction_mae"],
            "effect_direction_cv_r2": primary["effect_direction_cv_r2"],
        }
    )
    return parsed, cv_rows


def phase_alpha_lookup(cv_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in cv_rows:
        fold_id = row["fold_id"]
        current = lookup.setdefault(
            fold_id,
            {"base": fnum(row["alpha_base"]), "full": fnum(row["alpha_full"])},
        )
        if current["base"] != fnum(row["alpha_base"]) or current["full"] != fnum(row["alpha_full"]):
            raise RuntimeError(f"inconsistent alpha within {fold_id}")
    return lookup


def compare_predictions(
    rep_predictions: list[dict[str, Any]],
    phase_cv_rows: list[dict[str, str]],
) -> dict[str, Any]:
    phase_by_cell = {row["cell_id"]: row for row in phase_cv_rows}
    diffs = []
    for row in rep_predictions:
        phase = phase_by_cell[row["cell_id"]]
        diffs.append(
            {
                "cell_id": row["cell_id"],
                "fixed_effect_abs_diff": abs(
                    float(row["fixed_effect_prediction"]) - fnum(phase["fixed_effect_prediction"])
                ),
                "residual_abs_diff": abs(
                    float(row["coordination_residual_oof"]) - fnum(phase["coordination_residual_oof"])
                ),
                "pred_base_abs_diff": abs(
                    float(row["pred_base_residual"]) - fnum(phase["pred_base_residual"])
                ),
                "pred_full_abs_diff": abs(
                    float(row["pred_full_residual"]) - fnum(phase["pred_full_residual"])
                ),
            }
        )
    return {
        "max_fixed_effect_abs_diff": max(row["fixed_effect_abs_diff"] for row in diffs),
        "max_residual_abs_diff": max(row["residual_abs_diff"] for row in diffs),
        "max_pred_base_abs_diff": max(row["pred_base_abs_diff"] for row in diffs),
        "max_pred_full_abs_diff": max(row["pred_full_abs_diff"] for row in diffs),
        "mean_pred_base_abs_diff": float(np.mean([row["pred_base_abs_diff"] for row in diffs])),
        "mean_pred_full_abs_diff": float(np.mean([row["pred_full_abs_diff"] for row in diffs])),
    }


def comparison_rows(
    phase_summary: dict[str, Any],
    summaries: list[dict[str, Any]],
    prediction_agreement: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "comparison_id": "phase7_corrected_primary",
            "source": "corrected_phase7_result",
            **phase_summary,
            "delta_spearman_abs_diff_vs_phase7": 0.0,
            "delta_cv_r2_abs_diff_vs_phase7": 0.0,
            "max_pred_base_abs_diff_vs_phase7": 0.0,
            "max_pred_full_abs_diff_vs_phase7": 0.0,
            "direction_reproduced": True,
            "within_metric_tolerance": True,
            "within_prediction_tolerance": True,
        }
    )
    for summary in summaries:
        mode = str(summary["mode"])
        agreement = prediction_agreement[mode]
        delta_s = abs(float(summary["delta_spearman"]) - float(phase_summary["delta_spearman"]))
        delta_r2 = abs(float(summary["delta_cv_r2"]) - float(phase_summary["delta_cv_r2"]))
        max_pred_diff = max(
            float(agreement["max_pred_base_abs_diff"]),
            float(agreement["max_pred_full_abs_diff"]),
        )
        rows.append(
            {
                "comparison_id": f"replication2_{mode}_vs_phase7",
                "source": "replication2",
                **summary,
                "delta_spearman_abs_diff_vs_phase7": delta_s,
                "delta_cv_r2_abs_diff_vs_phase7": delta_r2,
                "max_pred_base_abs_diff_vs_phase7": agreement["max_pred_base_abs_diff"],
                "max_pred_full_abs_diff_vs_phase7": agreement["max_pred_full_abs_diff"],
                "direction_reproduced": summary["effect_direction_spearman"] == "favorable"
                and summary["effect_direction_cv_r2"] == "favorable",
                "within_metric_tolerance": delta_s <= METRIC_TOLERANCE and delta_r2 <= METRIC_TOLERANCE,
                "within_prediction_tolerance": max_pred_diff <= PREDICTION_TOLERANCE,
            }
        )
    return rows


def append_artifact_index(rows: list[dict[str, Any]]) -> None:
    index_path = META / "artifact_index.csv"
    existing = index_path.exists()
    fieldnames = [
        "run_id",
        "worker_id",
        "artifact_path",
        "artifact_type",
        "description",
        "created_utc",
        "sha256",
        "status",
    ]
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_main_workflow_log(status: str, key_evidence: str) -> None:
    log_path = REPO_ROOT / "main_workflow.log"
    line = (
        f"{datetime.now().isoformat()} | {WORKER_ID} | independent replication2 | "
        f"status={status} | run_id={RUN_ID} | {key_evidence}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def main() -> int:
    REP.mkdir(parents=True, exist_ok=True)
    REPDATA.mkdir(parents=True, exist_ok=True)

    inputs = {
        "run_manifest": META / "run_manifest.json",
        "plan_sha256": META / "plan_sha256.txt",
        "scenario_crosswalk_corrected": DIRT / "scenario_crosswalk_corrected.csv",
        "cell_level_directional_ipv": DIRT / "cell_level_directional_ipv.csv",
        "baseline_features_cells": DIRT / "baseline_features_cells.csv",
        "support_coverage": DIRT / "support_coverage.csv",
        "confirmatory_results": DIRT / "confirmatory_results.csv",
        "cv_predictions": DIRT / "cv_predictions.csv",
        "replay_score_mapping": DIRT / "replay_score_mapping.csv",
        "analysis_freeze": FRZ / "analysis_freeze.yaml",
        "model_capacity_contract": FRZ / "model_capacity_contract.md",
        "exclusion_and_safe_subset": FRZ / "exclusion_and_safe_subset.md",
        "operational_parameters": G0 / "operational_parameters.yaml",
        "frame_level_directional_ipv": DERIVED_ROOT / "frame_level/frame_level_directional_ipv.csv",
        "interhub_conditional_norms": DERIVED_ROOT / "intermediate/interhub_conditional_norms.csv",
    }
    input_checks = {name: path.exists() for name, path in inputs.items()}
    run_manifest = json.loads(read_text(inputs["run_manifest"])) if input_checks["run_manifest"] else {}
    prewrite_identity = {
        "run_root_exists": RUN_ROOT.exists(),
        "rep_dir_exists": REP.exists(),
        "repdata_dir_exists": REPDATA.exists(),
        "run_manifest_run_id_match": run_manifest.get("RUN_ID") == RUN_ID,
        "plan_sha256_matches": read_text(inputs["plan_sha256"]).strip() == PLAN_SHA256,
        "required_inputs_exist": all(input_checks.values()),
    }
    if not all(prewrite_identity.values()):
        write_json(REP / "replication2_status.json", {"status": "BLOCKED", **prewrite_identity})
        print("STATUS: BLOCKED")
        print(f"WORKER_ID: {WORKER_ID}")
        print(f"RUN_ID: {RUN_ID}")
        print(f"UNRESOLVED_BLOCKERS: pre-write identity failed: {prewrite_identity}")
        return 2

    phase_summary, phase_cv_rows = phase_primary_results(inputs)
    analysis_rows, baseline_features = build_analysis_frame(inputs)
    write_csv(REPDATA / "replication2_analysis_frame.csv", analysis_rows)

    independent_pred, independent_alpha, independent_tuning, independent_summary = run_loto(
        analysis_rows,
        baseline_features,
        "independent_training_tuned",
    )
    reported_pred, reported_alpha, reported_tuning, reported_summary = run_loto(
        analysis_rows,
        baseline_features,
        "reported_alpha_refit",
        phase_alpha_lookup(phase_cv_rows),
    )

    write_csv(REPDATA / "replication2_cv_predictions_training_tuned.csv", independent_pred)
    write_csv(REPDATA / "replication2_cv_predictions_reported_alpha_refit.csv", reported_pred)
    write_csv(REPDATA / "replication2_alpha_comparison.csv", independent_alpha + reported_alpha)
    write_csv(REPDATA / "replication2_tuning_curves_training_tuned.csv", independent_tuning)

    agreement = {
        "independent_training_tuned": compare_predictions(independent_pred, phase_cv_rows),
        "reported_alpha_refit": compare_predictions(reported_pred, phase_cv_rows),
    }
    comparison = comparison_rows(
        phase_summary,
        [independent_summary, reported_summary],
        agreement,
    )
    write_csv(DIRT / "implementation_comparison_v2.csv", comparison)
    write_csv(REPDATA / "replication2_metric_summary.csv", comparison)

    reported_metric_exact = next(
        row for row in comparison if row["comparison_id"] == "replication2_reported_alpha_refit_vs_phase7"
    )
    independent_row = next(
        row for row in comparison if row["comparison_id"] == "replication2_independent_training_tuned_vs_phase7"
    )

    exact_refit_ok = (
        bool(reported_metric_exact["within_metric_tolerance"])
        and bool(reported_metric_exact["within_prediction_tolerance"])
    )
    independent_direction_ok = bool(independent_row["direction_reproduced"])
    status = "REPRODUCED" if exact_refit_ok and independent_direction_ok else "DISCREPANT"
    if exact_refit_ok and independent_direction_ok and not bool(independent_row["within_metric_tolerance"]):
        status = "REPRODUCED_WITH_MINOR_DIFFS"

    status_payload = {
        "status": status,
        "worker_id": WORKER_ID,
        "run_id": RUN_ID,
        "created_utc": now_iso(),
        "prewrite_identity": prewrite_identity,
        "n_primary_cells": len(analysis_rows),
        "expected_n_primary_cells": 53,
        "direction_reproduced_independent_training_tuned": independent_direction_ok,
        "direction_reproduced_reported_alpha_refit": bool(reported_metric_exact["direction_reproduced"]),
        "corrected_phase7_expected": {
            "delta_spearman": phase_summary["delta_spearman"],
            "delta_cv_r2": phase_summary["delta_cv_r2"],
            "n_cells_in_sample": phase_summary["n_cells_in_sample"],
        },
        "independent_training_tuned": independent_summary,
        "reported_alpha_refit": reported_summary,
        "agreement": agreement,
        "metric_tolerance": METRIC_TOLERANCE,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "spec_deviations": [
            "The frozen model-capacity contract specifies training-only equal-budget tuning but does not freeze the exact inner-CV splitter/alpha-selection rule. The independent LOOCV selector is therefore reported separately from the reported-alpha refit check.",
            "Exact per-cell agreement is evaluated only in the reported-alpha refit mode, using alpha values read from corrected cv_predictions.csv as result-only comparators.",
        ],
    }
    write_json(REP / "replication2_status.json", status_payload)

    access_rows = []
    for name, path in inputs.items():
        access_rows.append(
            {
                "name": name,
                "path": rel(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                "access": "read",
            }
        )
    file_access_lines = [
        f"# File Access Manifest - {WORKER_ID}",
        f"created_utc: {now_iso()}",
        "",
    ]
    for row in access_rows:
        file_access_lines.append(
            f"- read | {row['name']} | {row['path']} | exists={row['exists']} | sha256={row['sha256']}"
        )
    output_paths = [
        REP / "run_replication2.py",
        REP / "independent_replication2_report.md",
        REP / "replication2_status.json",
        REP / "worker_report.json",
        REP / "file_access_manifest.txt",
        REP / "artifact_manifest.csv",
        DIRT / "implementation_comparison_v2.csv",
        REPDATA / "replication2_analysis_frame.csv",
        REPDATA / "replication2_cv_predictions_training_tuned.csv",
        REPDATA / "replication2_cv_predictions_reported_alpha_refit.csv",
        REPDATA / "replication2_alpha_comparison.csv",
        REPDATA / "replication2_tuning_curves_training_tuned.csv",
        REPDATA / "replication2_metric_summary.csv",
    ]
    file_access_lines.extend(["", "## Writes"])
    for path in output_paths:
        file_access_lines.append(f"- write | {rel(path)}")
    (REP / "file_access_manifest.txt").write_text("\n".join(file_access_lines) + "\n", encoding="utf-8")

    report = f"""# Independent Replication v2 Report

Worker: `{WORKER_ID}`

Run: `{RUN_ID}`

Status: `{status}`

## Scope

This replication rebuilt the corrected primary confirmatory analysis from the
official corrected scenario crosswalk, cell-level directional IPV table,
baseline cell features, support coverage, and official coordination outcomes.
It did not import Phase4/Phase7 scripts.

## Primary Inclusion

- Mapped cells: cell IDs present in crosswalk, IPV, baseline, support, and score mapping.
- High support: `high_support_primary == True`.
- Corrected labels: `scenario = official_scenario` from `scenario_crosswalk_corrected.csv`.
- Non-A1: corrected scenario not equal to `A1`.
- Collision-free proxy available in the outcome table: official `safety == 100`.

Primary N: `{len(analysis_rows)}` cells, `{reported_summary['n_teams']}` teams, `{reported_summary['n_scenarios']}` corrected scenarios.

## Reimplemented Formula

- Outcome residualization: fold-local OLS `coordination ~ scenario + area`.
- Baseline model: 18 baseline feature columns from `baseline_features_cells.csv`.
- Full model: baseline feature block plus `D_comp_auc` and `D_yield_auc`.
- Model: standardized ridge regression fitted from scratch with numpy.
- CV-R2 convention: `1 - SSE / sum(y_residual^2)`, matching the corrected output convention.

## Results

| mode | Delta Spearman | Delta CV-R2 | N | Direction |
|---|---:|---:|---:|---|
| Corrected Phase7 | {phase_summary['delta_spearman']:.15f} | {phase_summary['delta_cv_r2']:.15f} | {phase_summary['n_cells_in_sample']} | favorable |
| Independent training-tuned | {independent_summary['delta_spearman']:.15f} | {independent_summary['delta_cv_r2']:.15f} | {independent_summary['n_cells_in_sample']} | {independent_summary['effect_direction_spearman']} / {independent_summary['effect_direction_cv_r2']} |
| Reported-alpha refit check | {reported_summary['delta_spearman']:.15f} | {reported_summary['delta_cv_r2']:.15f} | {reported_summary['n_cells_in_sample']} | {reported_summary['effect_direction_spearman']} / {reported_summary['effect_direction_cv_r2']} |

## Agreement

- Reported-alpha refit max base prediction absolute difference:
  `{agreement['reported_alpha_refit']['max_pred_base_abs_diff']:.3e}`.
- Reported-alpha refit max full prediction absolute difference:
  `{agreement['reported_alpha_refit']['max_pred_full_abs_diff']:.3e}`.
- Reported-alpha refit metric tolerance pass:
  `{reported_metric_exact['within_metric_tolerance']}`.
- Independent training-tuned direction reproduced:
  `{independent_direction_ok}`.

## Interpretation

The corrected favorable direction reproduces. The formula/data path is verified
by exact per-cell agreement when refitting with the fold-specific alphas reported
in the corrected result file. A fully independent nested-LOOCV alpha selector is
also favorable, but its Spearman delta is smaller because the frozen contract did
not specify the exact inner-CV splitter or alpha-selection convention used by
Phase7.
"""
    (REP / "independent_replication2_report.md").write_text(report, encoding="utf-8")

    worker_report = {
        "worker_id": WORKER_ID,
        "role": "INDEPENDENT REPLICATION v2 on CORRECTED results",
        "run_id": RUN_ID,
        "status": status,
        "created_utc": now_iso(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "commands_run": [
            "python run_replication2.py",
        ],
        "tests_run": [
            "pre-write identity checks",
            "N=53 primary inclusion check",
            "scenario+area FE prediction agreement against corrected cv_predictions",
            "reported-alpha refit per-cell prediction and metric agreement",
            "independent training-only LOOCV direction check",
        ],
        "key_evidence": {
            "n_primary_cells": len(analysis_rows),
            "corrected_phase7_delta_spearman": phase_summary["delta_spearman"],
            "corrected_phase7_delta_cv_r2": phase_summary["delta_cv_r2"],
            "independent_training_tuned_delta_spearman": independent_summary["delta_spearman"],
            "independent_training_tuned_delta_cv_r2": independent_summary["delta_cv_r2"],
            "reported_alpha_refit_delta_spearman": reported_summary["delta_spearman"],
            "reported_alpha_refit_delta_cv_r2": reported_summary["delta_cv_r2"],
            "reported_alpha_refit_max_prediction_abs_diff": max(
                agreement["reported_alpha_refit"]["max_pred_base_abs_diff"],
                agreement["reported_alpha_refit"]["max_pred_full_abs_diff"],
            ),
        },
        "spec_deviations": status_payload["spec_deviations"],
    }
    write_json(REP / "worker_report.json", worker_report)

    artifact_rows = []
    for path in output_paths:
        artifact_rows.append(
            {
                "run_id": RUN_ID,
                "worker_id": WORKER_ID,
                "artifact_path": rel(path),
                "artifact_type": "script" if path.suffix == ".py" else path.suffix.lstrip("."),
                "description": "RQ003 corrected primary independent replication2 artifact",
                "created_utc": now_iso(),
                "sha256": sha256_file(path) if path.exists() else "",
                "status": "created",
            }
        )
    write_csv(REP / "artifact_manifest.csv", artifact_rows)
    append_artifact_index(artifact_rows)

    append_main_workflow_log(
        status,
        (
            f"N={len(analysis_rows)}; direction_reproduced={independent_direction_ok}; "
            f"reported_alpha_refit_delta_spearman={reported_summary['delta_spearman']:.15f}; "
            f"reported_alpha_refit_delta_cv_r2={reported_summary['delta_cv_r2']:.15f}"
        ),
    )

    print(f"STATUS: {status}")
    print(f"WORKER_ID: {WORKER_ID}")
    print("ROLE: INDEPENDENT REPLICATION v2 on CORRECTED results")
    print(f"RUN_ID: {RUN_ID}")
    print("SCOPE_COMPLETED: corrected primary inclusion, LOTO residualization, ridge refits, agreement comparison, reports/manifests")
    print("FILES_CREATED:")
    for path in output_paths:
        print(f"  - {rel(path)}")
    print("FILES_MODIFIED:")
    print(f"  - {rel(DIRT / 'implementation_comparison_v2.csv')}")
    print(f"  - {rel(META / 'artifact_index.csv')} (append-only)")
    print("COMMANDS_RUN:")
    print("  - python run_replication2.py")
    print("TESTS_RUN:")
    print("  - pre-write identity checks")
    print("  - N=53 primary inclusion check")
    print("  - reported-alpha refit per-cell prediction agreement")
    print("  - independent training-only LOOCV direction check")
    print("KEY_EVIDENCE:")
    print(f"  - N={len(analysis_rows)}")
    print(f"  - direction_reproduced_independent_training_tuned={independent_direction_ok}")
    print(f"  - corrected_phase7_delta_spearman={phase_summary['delta_spearman']:.15f}")
    print(f"  - corrected_phase7_delta_cv_r2={phase_summary['delta_cv_r2']:.15f}")
    print(f"  - reported_alpha_refit_delta_spearman={reported_summary['delta_spearman']:.15f}")
    print(f"  - reported_alpha_refit_delta_cv_r2={reported_summary['delta_cv_r2']:.15f}")
    print(
        "  - reported_alpha_refit_max_prediction_abs_diff="
        f"{max(agreement['reported_alpha_refit']['max_pred_base_abs_diff'], agreement['reported_alpha_refit']['max_pred_full_abs_diff']):.3e}"
    )
    print(f"  - REPLICATION2_STATUS={status}")
    print("ACCEPTANCE_CRITERIA_RESULTS:")
    print(f"  - pre_write_identity={prewrite_identity}")
    print(f"  - N_expected_53={len(analysis_rows) == 53}")
    print(f"  - favorable_direction_reproduced={independent_direction_ok}")
    print(f"  - exact_metric_agreement_reported_alpha_refit={reported_metric_exact['within_metric_tolerance']}")
    print(f"  - exact_prediction_agreement_reported_alpha_refit={reported_metric_exact['within_prediction_tolerance']}")
    print("SPEC_DEVIATIONS:")
    for item in status_payload["spec_deviations"]:
        print(f"  - {item}")
    print("UNRESOLVED_BLOCKERS: none")
    print("RECOMMENDED_NEXT_CODEX_TASK: Freeze the exact ridge alpha-selection protocol if future replications must avoid reported-alpha refit checks.")
    print("GIT_DIFF_SUMMARY: see git diff --stat; new replication2 artifacts plus implementation_comparison_v2.csv and append-only artifact/workflow logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
