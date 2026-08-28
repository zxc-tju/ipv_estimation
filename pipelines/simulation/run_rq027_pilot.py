#!/usr/bin/env python3
"""Run the bounded RQ027 independent-generator feasibility pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pipelines.simulation.rq027_independent_generator import (  # noqa: E402
    N_STEPS,
    build_interactive_cases,
    build_negative_control_cases,
    generate_run,
)
from sociality_estimation.core.ipv_estimation import (  # noqa: E402
    MotionSequence,
    estimate_ipv_pair,
)


HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
SIGMA_M = 0.1
MAX_WEIGHT_THRESHOLD = 0.20
MSE_SPREAD_TOL = 1e-15
PERSISTENCE_K = 3
HALF_GRID_TOL = math.pi / 16.0
ONE_GRID_TOL = math.pi / 8.0


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_finite_or_none(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def _rankdata(values: Sequence[float]) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average").to_numpy(dtype=float)


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x_rank = _rankdata(x[mask])
    y_rank = _rankdata(y[mask])
    if np.ptp(x_rank) == 0 or np.ptp(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _logdomain_measurement(diagnostic: dict[str, Any]) -> dict[str, Any]:
    observed = np.asarray(diagnostic["observed"], dtype=float)
    virtual = [np.asarray(track, dtype=float) for track in diagnostic["virtual_tracks"]]
    ipv_range = np.asarray(diagnostic["ipv_range"], dtype=float)
    if len(virtual) != len(ipv_range):
        raise ValueError("candidate grid / virtual-track count mismatch")
    mse = np.asarray(
        [np.mean(np.sum((track - observed) ** 2, axis=1)) for track in virtual],
        dtype=float,
    )
    if not np.all(np.isfinite(mse)):
        raise ValueError("non-finite candidate MSE")
    log_weights = -mse / (2.0 * SIGMA_M**2)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= float(np.sum(weights))
    mse_spread = float(np.ptp(mse))
    max_weight = float(np.max(weights))
    q_eff = float(1.0 / (len(weights) * np.sum(weights**2)))
    ipv_log = float(np.sum(ipv_range * weights))
    concentration_pass = bool(
        mse_spread > MSE_SPREAD_TOL and max_weight >= MAX_WEIGHT_THRESHOLD
    )
    return {
        "mse": mse,
        "weights": weights,
        "ipv_range": ipv_range,
        "mse_spread": mse_spread,
        "max_weight": max_weight,
        "q_eff": q_eff,
        "ipv_log": ipv_log,
        "concentration_pass": concentration_pass,
    }


def _first_persistent_index(
    rows: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> int | None:
    consecutive = 0
    for index, row in enumerate(rows):
        consecutive = consecutive + 1 if predicate(row) else 0
        if consecutive >= PERSISTENCE_K:
            return index - PERSISTENCE_K + 1
    return None


def _frame_error(ipv_log: float, true_ipv_rad: float) -> tuple[float, float]:
    signed = float(ipv_log - true_ipv_rad)
    return signed, abs(signed)


def evaluate_case(case: dict[str, Any], solver_mode: str = "exact") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated = generate_run(case)
    run_id = str(case["run_id"])
    base_run: dict[str, Any] = {
        **case,
        "n_steps": N_STEPS,
        "solver_mode": solver_mode,
        "history_window": HISTORY_WINDOW,
        "min_observation": MIN_OBSERVATION,
        "persistent_k": PERSISTENCE_K,
        "frame_rows": 0,
        "run_status": "ENGINEERING_FAILURE",
        "reason_code": "UNSET",
        "persistent_onset_step": None,
        "persistent_concentration_any": False,
        "run_estimate_ipv_log": None,
        "signed_error_rad": None,
        "abs_error_rad": None,
        "zero_abs_error_rad": None,
        "half_grid_success": False,
        "one_grid_success": False,
        "zero_half_grid_success": False,
        "zero_one_grid_success": False,
        "collision_any": bool(np.any(generated.collision_mask)),
        "opportunity_any": bool(np.any(generated.opportunity_mask)),
    }
    frame_rows: list[dict[str, Any]] = []
    try:
        target = MotionSequence(
            generated.target_motion,
            target=f"rq027_{case['template']}_target",
            reference=generated.target_reference,
        )
        counterpart = MotionSequence(
            generated.counterpart_motion,
            target=f"rq027_{case['template']}_counterpart",
            reference=generated.counterpart_reference,
        )
        ipv_values, ipv_errors, diagnostics = estimate_ipv_pair(
            target,
            counterpart,
            history_window=HISTORY_WINDOW,
            min_observation=MIN_OBSERVATION,
            return_diagnostics=True,
            solver_mode=solver_mode,
        )
        primary_diagnostics = diagnostics["primary"]
        for diagnostic in primary_diagnostics:
            step = int(diagnostic["step"])
            measurement = _logdomain_measurement(diagnostic)
            true_ipv = float(case["true_ipv_rad"])
            signed_error, abs_error = _frame_error(measurement["ipv_log"], true_ipv)
            frame: dict[str, Any] = {
                "run_id": run_id,
                "run_kind": case["run_kind"],
                "negative_control_type": case["negative_control_type"],
                "template": case["template"],
                "intensity": case["intensity"],
                "seed": case["seed"],
                "step": step,
                "history_start_index": int(diagnostic["start_index"]),
                "history_sample_count": int(len(diagnostic["observed"])),
                "true_ipv_rad": true_ipv,
                "counterpart_ipv_rad": float(case["counterpart_ipv_rad"]),
                "ipv_log": measurement["ipv_log"],
                "legacy_ipv": float(ipv_values[step, 0]),
                "legacy_ipv_error": float(ipv_errors[step, 0]),
                "signed_error_rad": signed_error,
                "abs_error_rad": abs_error,
                "max_weight": measurement["max_weight"],
                "q_eff": measurement["q_eff"],
                "mse_spread": measurement["mse_spread"],
                "concentration_pass": measurement["concentration_pass"],
                "opportunity_true": bool(generated.opportunity_mask[step]),
                "oracle_informativeness": float(
                    generated.oracle_informativeness[step]
                ),
                "collision_true": bool(generated.collision_mask[step]),
                "measurement_status": "ATTEMPTED",
                "reason_code": "OK",
            }
            for index, value in enumerate(measurement["mse"]):
                frame[f"mse_{index}"] = float(value)
            for index, value in enumerate(measurement["weights"]):
                frame[f"w_log_{index}"] = float(value)
            frame_rows.append(frame)
    except Exception as exc:  # preserve run-level failure rather than dropping it
        base_run.update(
            {
                "run_status": "ENGINEERING_FAILURE",
                "reason_code": f"{type(exc).__name__}:{exc}",
                "frame_rows": len(frame_rows),
            }
        )
        return frame_rows, base_run

    base_run["frame_rows"] = len(frame_rows)
    if len(frame_rows) != N_STEPS - MIN_OBSERVATION:
        base_run.update(
            {
                "run_status": "ENGINEERING_FAILURE",
                "reason_code": "FRAME_ROW_COUNT_MISMATCH",
            }
        )
        return frame_rows, base_run

    any_onset_index = _first_persistent_index(
        frame_rows, lambda row: bool(row["concentration_pass"])
    )
    base_run["persistent_concentration_any"] = any_onset_index is not None

    if case["run_kind"] == "negative_control":
        base_run.update(
            {
                "run_status": "NEGATIVE_CONTROL_COMPLETE",
                "reason_code": "OK",
                "persistent_onset_step": (
                    None
                    if any_onset_index is None
                    else int(frame_rows[any_onset_index]["step"])
                ),
            }
        )
        return frame_rows, base_run

    onset_index = _first_persistent_index(
        frame_rows,
        lambda row: bool(row["concentration_pass"] and row["opportunity_true"]),
    )
    if onset_index is None:
        base_run.update(
            {
                "run_status": "ABSTAIN_NO_PERSISTENT_ONSET",
                "reason_code": "NO_PERSISTENT_OPPORTUNITY_CONCENTRATION",
            }
        )
        return frame_rows, base_run

    accepted = [
        row
        for row in frame_rows[onset_index:]
        if row["concentration_pass"] and row["opportunity_true"]
    ]
    estimate = float(np.median([row["ipv_log"] for row in accepted]))
    true_ipv = float(case["true_ipv_rad"])
    signed_error, abs_error = _frame_error(estimate, true_ipv)
    zero_abs_error = abs(true_ipv)
    base_run.update(
        {
            "run_status": "RECOVERED_READING",
            "reason_code": "OK",
            "persistent_onset_step": int(frame_rows[onset_index]["step"]),
            "accepted_frame_count": len(accepted),
            "run_estimate_ipv_log": estimate,
            "signed_error_rad": signed_error,
            "abs_error_rad": abs_error,
            "zero_abs_error_rad": zero_abs_error,
            "half_grid_success": abs_error <= HALF_GRID_TOL,
            "one_grid_success": abs_error <= ONE_GRID_TOL,
            "zero_half_grid_success": zero_abs_error <= HALF_GRID_TOL,
            "zero_one_grid_success": zero_abs_error <= ONE_GRID_TOL,
        }
    )
    return frame_rows, base_run


def _recovery_metrics(run_frame: pd.DataFrame, interactive_total: int) -> dict[str, Any]:
    accepted = run_frame.loc[run_frame["run_status"] == "RECOVERED_READING"].copy()
    count = int(len(accepted))
    if count == 0:
        return {
            "accepted_runs": 0,
            "interactive_total": interactive_total,
            "coverage": 0.0 if interactive_total else None,
            "bias_rad": None,
            "mae_rad": None,
            "rmse_rad": None,
            "median_abs_error_rad": None,
            "spearman": None,
            "half_grid_success": {"numerator": 0, "denominator": 0, "value": None},
            "one_grid_success": {"numerator": 0, "denominator": 0, "value": None},
            "zero_half_grid_success": {"numerator": 0, "denominator": 0, "value": None},
            "zero_one_grid_success": {"numerator": 0, "denominator": 0, "value": None},
            "zero_mae_rad": None,
        }
    signed = accepted["signed_error_rad"].to_numpy(dtype=float)
    absolute = accepted["abs_error_rad"].to_numpy(dtype=float)
    truth = accepted["true_ipv_rad"].to_numpy(dtype=float)
    estimate = accepted["run_estimate_ipv_log"].to_numpy(dtype=float)

    def success(field: str) -> dict[str, Any]:
        numerator = int(accepted[field].astype(bool).sum())
        return {"numerator": numerator, "denominator": count, "value": numerator / count}

    return {
        "accepted_runs": count,
        "interactive_total": interactive_total,
        "coverage": count / interactive_total if interactive_total else None,
        "bias_rad": float(np.mean(signed)),
        "mae_rad": float(np.mean(absolute)),
        "rmse_rad": float(np.sqrt(np.mean(signed**2))),
        "median_abs_error_rad": float(np.median(absolute)),
        "spearman": _spearman(truth, estimate),
        "half_grid_success": success("half_grid_success"),
        "one_grid_success": success("one_grid_success"),
        "zero_half_grid_success": success("zero_half_grid_success"),
        "zero_one_grid_success": success("zero_one_grid_success"),
        "zero_mae_rad": float(np.mean(accepted["zero_abs_error_rad"].to_numpy(dtype=float))),
    }


def _build_summary(
    frame_data: pd.DataFrame,
    run_data: pd.DataFrame,
    *,
    limited: bool,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    interactive = run_data.loc[run_data["run_kind"] == "interactive"].copy()
    negative = run_data.loc[run_data["run_kind"] == "negative_control"].copy()
    recovery = _recovery_metrics(interactive, int(len(interactive)))
    accepted = interactive.loc[interactive["run_status"] == "RECOVERED_READING"]

    template_spearman: dict[str, Any] = {}
    for template, group in accepted.groupby("template", sort=True):
        template_spearman[str(template)] = _spearman(
            group["true_ipv_rad"], group["run_estimate_ipv_log"]
        )

    interactive_frames = frame_data.loc[frame_data["run_kind"] == "interactive"].copy()
    q_eff_error_spearman = _spearman(
        interactive_frames.get("q_eff", pd.Series(dtype=float)),
        interactive_frames.get("abs_error_rad", pd.Series(dtype=float)),
    )
    pass_frames = interactive_frames.loc[
        interactive_frames.get("concentration_pass", pd.Series(dtype=bool)).astype(bool)
    ]
    gate_risk = {
        "attempted_frame_count": int(len(interactive_frames)),
        "pass_frame_count": int(len(pass_frames)),
        "coverage": (
            float(len(pass_frames) / len(interactive_frames))
            if len(interactive_frames)
            else None
        ),
        "all_frame_mae_rad": (
            float(interactive_frames["abs_error_rad"].mean())
            if len(interactive_frames)
            else None
        ),
        "pass_frame_mae_rad": (
            float(pass_frames["abs_error_rad"].mean()) if len(pass_frames) else None
        ),
    }

    negative_persistent_num = int(
        negative.get("persistent_concentration_any", pd.Series(dtype=bool))
        .astype(bool)
        .sum()
    )
    negative_den = int(len(negative))
    negative_rate = negative_persistent_num / negative_den if negative_den else None
    interactive_persistent_num = int(
        (interactive["run_status"] == "RECOVERED_READING").sum()
    )
    interactive_den = int(len(interactive))
    interactive_rate = interactive_persistent_num / interactive_den if interactive_den else None

    engineering_failures = int((run_data["run_status"] == "ENGINEERING_FAILURE").sum())
    duplicate_run_ids = int(run_data["run_id"].duplicated(keep=False).sum())
    nonfinite_primary = 0
    if len(frame_data):
        primary_columns = ["ipv_log", "max_weight", "q_eff", "mse_spread"]
        nonfinite_primary = int(
            (~np.isfinite(frame_data[primary_columns].to_numpy(dtype=float))).any(axis=1).sum()
        )
    full_matrix = bool(
        not limited and len(interactive) == 240 and len(negative) == 48 and len(run_data) == 288
    )
    health = {
        "full_matrix": full_matrix,
        "run_rows": int(len(run_data)),
        "interactive_run_rows": interactive_den,
        "negative_control_run_rows": negative_den,
        "frame_rows": int(len(frame_data)),
        "engineering_failure_runs": engineering_failures,
        "duplicate_run_id_rows": duplicate_run_ids,
        "nonfinite_primary_frame_rows": nonfinite_primary,
        "collision_runs": int(run_data["collision_any"].astype(bool).sum()),
        "elapsed_seconds": elapsed_seconds,
        "row_conservation_ok": (
            len(run_data) == 288 if not limited else len(run_data) > 0
        ),
    }

    nonnegative_templates = sum(
        value is not None and value >= 0.0 for value in template_spearman.values()
    )
    recovery_gate = bool(
        recovery["spearman"] is not None
        and recovery["spearman"] > 0.0
        and nonnegative_templates >= 3
        and recovery["mae_rad"] is not None
        and recovery["zero_mae_rad"] is not None
        and recovery["mae_rad"] < recovery["zero_mae_rad"]
        and recovery["one_grid_success"]["value"] is not None
        and recovery["zero_one_grid_success"]["value"] is not None
        and recovery["one_grid_success"]["value"]
        > recovery["zero_one_grid_success"]["value"]
    )
    concentration_gate = bool(
        q_eff_error_spearman is not None and q_eff_error_spearman >= 0.0
    )
    negative_gate = bool(
        negative_rate is not None
        and interactive_rate is not None
        and negative_rate < interactive_rate
        and negative_rate <= 0.25
    )
    engineering_gate = bool(
        full_matrix
        and engineering_failures == 0
        and duplicate_run_ids == 0
        and nonfinite_primary == 0
    )
    gates = {
        "engineering_integrity": engineering_gate,
        "recovery_vs_zero_baseline": recovery_gate,
        "q_eff_error_not_reversed": concentration_gate,
        "negative_control_persistent_false_accept": negative_gate,
    }
    verdict = "SMOKE_ONLY" if limited else (
        "PILOT_GO" if all(gates.values()) else "PILOT_NO_GO"
    )
    summary = {
        "schema_version": "rq027-feasibility-summary-v1",
        "verdict": verdict,
        "scope": "limited_smoke" if limited else "full_240_plus_48",
        "recovery": recovery,
        "template_spearman": template_spearman,
        "nonnegative_template_count": nonnegative_templates,
        "q_eff_abs_error_spearman": q_eff_error_spearman,
        "fixed_policy_gate": gate_risk,
        "interactive_persistent_coverage": {
            "numerator": interactive_persistent_num,
            "denominator": interactive_den,
            "value": interactive_rate,
            "filter": "run_kind=interactive and run_status=RECOVERED_READING",
            "source_columns": "run_kind,run_status",
        },
        "negative_persistent_false_accept": {
            "numerator": negative_persistent_num,
            "denominator": negative_den,
            "value": negative_rate,
            "filter": "run_kind=negative_control and persistent_concentration_any=true",
            "source_columns": "run_kind,persistent_concentration_any",
        },
        "gates": gates,
        "claim_boundary": {
            "direct_support": "bounded independent-simulation feasibility only",
            "cannot_prove": [
                "human latent IPV",
                "causality",
                "external validity",
                "production readiness",
                "changes to accepted RQ017+ claims",
            ],
        },
    }
    return summary, health


def _evidence_rows(summary: dict[str, Any], health: dict[str, Any]) -> list[dict[str, Any]]:
    recovery = summary["recovery"]
    interactive = summary["interactive_persistent_coverage"]
    negative = summary["negative_persistent_false_accept"]
    return [
        {
            "evidence_id": "RQ027-E1-HEALTH",
            "claim": "Pilot run ledger completed with explicit engineering failures",
            "numerator": health["run_rows"] - health["engineering_failure_runs"],
            "denominator": health["run_rows"],
            "filter": "all scheduled runs",
            "source_file": "run_level_results.csv",
            "source_columns": "run_id,run_status,reason_code",
            "status": "reported",
        },
        {
            "evidence_id": "RQ027-E2-COVERAGE",
            "claim": "Interactive runs reached persistent opportunity-aware concentration",
            "numerator": interactive["numerator"],
            "denominator": interactive["denominator"],
            "filter": interactive["filter"],
            "source_file": "run_level_results.csv",
            "source_columns": interactive["source_columns"],
            "status": "reported",
        },
        {
            "evidence_id": "RQ027-E3-ONEGRID",
            "claim": "Accepted interactive runs met one-grid recovery tolerance",
            "numerator": recovery["one_grid_success"]["numerator"],
            "denominator": recovery["one_grid_success"]["denominator"],
            "filter": "run_status=RECOVERED_READING",
            "source_file": "run_level_results.csv",
            "source_columns": "run_status,abs_error_rad,one_grid_success",
            "status": "reported",
        },
        {
            "evidence_id": "RQ027-E4-NEGATIVE",
            "claim": "Negative-control runs had persistent concentration-only acceptance",
            "numerator": negative["numerator"],
            "denominator": negative["denominator"],
            "filter": negative["filter"],
            "source_file": "run_level_results.csv",
            "source_columns": negative["source_columns"],
            "status": "reported",
        },
    ]


def _write_report(path: Path, summary: dict[str, Any], health: dict[str, Any]) -> None:
    recovery = summary["recovery"]
    negative = summary["negative_persistent_false_accept"]
    interactive = summary["interactive_persistent_coverage"]
    lines = [
        "# RQ027 Independent-Generator Feasibility Pilot",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## 定位",
        "",
        "本轮检验当前冻结 IPV estimator 在不共享 planner/search/cost/likelihood 的独立轨迹生成器上能否恢复已知仿真参数，并直接检验候选集中门在无互动负对照中是否持续误报。它是 development-only feasibility，不是手稿级 confirmatory evidence。",
        "",
        "## 运行健康",
        "",
        f"- 完成 run：{health['run_rows']}；interactive={health['interactive_run_rows']}，negative-control={health['negative_control_run_rows']}。",
        f"- 工程失败：{health['engineering_failure_runs']}/{health['run_rows']}；重复 run ID 行：{health['duplicate_run_id_rows']}；非有限 primary frame：{health['nonfinite_primary_frame_rows']}/{health['frame_rows']}。",
        f"- 碰撞 run：{health['collision_runs']}/{health['run_rows']}；耗时 {health['elapsed_seconds']:.3f} s。",
        "",
        "## Recovery",
        "",
        f"- persistent opportunity-aware coverage：{interactive['numerator']}/{interactive['denominator']} = {interactive['value']}。",
        f"- accepted-run MAE：{recovery['mae_rad']} rad；zero predictor MAE：{recovery['zero_mae_rad']} rad；Spearman：{recovery['spearman']}。",
        f"- one-grid success：{recovery['one_grid_success']['numerator']}/{recovery['one_grid_success']['denominator']} = {recovery['one_grid_success']['value']}；zero predictor：{recovery['zero_one_grid_success']['numerator']}/{recovery['zero_one_grid_success']['denominator']} = {recovery['zero_one_grid_success']['value']}。",
        "",
        "## Concentration 与负对照",
        "",
        f"- q_eff 与 frame absolute error 的 Spearman：{summary['q_eff_abs_error_spearman']}（q_eff 越大表示候选越分散）。",
        f"- negative-control persistent concentration false accept：{negative['numerator']}/{negative['denominator']} = {negative['value']}。",
        "",
        "## Feasibility Gates",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in summary["gates"].items())
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- `可直接支撑`：本次独立仿真 feasibility 的运行与数值结果。",
            "- `可作旁证`：既有 S0/parity 只证明工程接线。",
            "- `待核验`：更大模板集、S2 扰动和 sealed confirmatory。",
            "- `不能证明`：真实人类心理 IPV、因果、外部有效性、生产可用性或任何 accepted RQ 的改判。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pilot(
    output_dir: Path,
    *,
    limit_runs: int | None,
    solver_mode: str,
    workers: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_interactive_cases() + build_negative_control_cases()
    if limit_runs is not None:
        if limit_runs <= 0:
            raise ValueError("--limit-runs must be positive")
        cases = cases[:limit_runs]
    start = time.perf_counter()
    all_frames: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    if workers <= 0:
        raise ValueError("workers must be positive")
    evaluator = partial(evaluate_case, solver_mode=solver_mode)
    if workers == 1:
        evaluated: Iterable[tuple[list[dict[str, Any]], dict[str, Any]]] = map(
            evaluator, cases
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        evaluated = executor.map(evaluator, cases, chunksize=1)
    for index, (frames, run_row) in enumerate(evaluated, start=1):
        all_frames.extend(frames)
        all_runs.append(run_row)
        if index % 20 == 0 or index == len(cases):
            print(f"RQ027 progress {index}/{len(cases)}", flush=True)
    if executor is not None:
        executor.shutdown(wait=True)
    elapsed = time.perf_counter() - start

    frame_data = pd.DataFrame(all_frames)
    run_data = pd.DataFrame(all_runs)
    frame_path = output_dir / "frame_level_results.csv"
    run_path = output_dir / "run_level_results.csv"
    frame_data.to_csv(frame_path, index=False)
    run_data.to_csv(run_path, index=False)
    if len(frame_data):
        frame_data.to_parquet(output_dir / "frame_level_results.parquet", index=False)

    summary, health = _build_summary(
        frame_data, run_data, limited=limit_runs is not None, elapsed_seconds=elapsed
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_finite_or_none(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "numerical_health.json").write_text(
        json.dumps(_finite_or_none(health), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_evidence_rows(summary, health)).to_csv(
        output_dir / "evidence_summary.csv", index=False
    )
    _write_report(output_dir / "REPORT.md", summary, health)

    artifact_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "run_receipt.json"
    )
    receipt = {
        "schema_version": "rq027-run-receipt-v1",
        "command": " ".join([sys.executable, *sys.argv]),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_short": _git_value("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "solver_mode": solver_mode,
        "workers": workers,
        "scheduled_run_count": len(cases),
        "limited_smoke": limit_runs is not None,
        "elapsed_seconds": elapsed,
        "artifacts": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in artifact_files
        ],
    }
    (output_dir / "run_receipt.json").write_text(
        json.dumps(_finite_or_none(receipt), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument(
        "--solver-mode", choices=("exact", "fast", "realtime"), default="exact"
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pilot(
        args.output_dir,
        limit_runs=args.limit_runs,
        solver_mode=args.solver_mode,
        workers=args.workers,
    )
    print(json.dumps({"verdict": summary["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
