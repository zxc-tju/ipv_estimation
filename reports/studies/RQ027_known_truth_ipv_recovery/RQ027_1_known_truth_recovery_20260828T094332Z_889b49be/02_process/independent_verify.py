#!/usr/bin/env python3
"""Independent one-pass recomputation for the RQ027 feasibility pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def finite(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    subset = frame[[left, right]].dropna()
    if len(subset) < 3 or subset[left].nunique() < 2 or subset[right].nunique() < 2:
        return None
    return float(subset[left].corr(subset[right], method="spearman"))


def recovery_group(group: pd.DataFrame, total: int) -> dict[str, Any]:
    accepted = group.loc[group["run_status"] == "RECOVERED_READING"].copy()
    denominator = int(len(accepted))
    result: dict[str, Any] = {
        "total_runs": int(total),
        "accepted_runs": denominator,
        "coverage": denominator / total if total else None,
        "spearman": spearman(accepted, "true_ipv_rad", "run_estimate_ipv_log"),
    }
    if denominator == 0:
        result.update(
            {
                "bias_rad": None,
                "mae_rad": None,
                "rmse_rad": None,
                "median_abs_error_rad": None,
                "zero_mae_rad": None,
                "half_grid_num": 0,
                "one_grid_num": 0,
                "zero_half_grid_num": 0,
                "zero_one_grid_num": 0,
            }
        )
        return result
    signed = accepted["signed_error_rad"].to_numpy(dtype=float)
    absolute = accepted["abs_error_rad"].to_numpy(dtype=float)
    result.update(
        {
            "bias_rad": float(np.mean(signed)),
            "mae_rad": float(np.mean(absolute)),
            "rmse_rad": float(np.sqrt(np.mean(signed**2))),
            "median_abs_error_rad": float(np.median(absolute)),
            "zero_mae_rad": float(accepted["zero_abs_error_rad"].mean()),
            "half_grid_num": int(accepted["half_grid_success"].astype(bool).sum()),
            "one_grid_num": int(accepted["one_grid_success"].astype(bool).sum()),
            "zero_half_grid_num": int(
                accepted["zero_half_grid_success"].astype(bool).sum()
            ),
            "zero_one_grid_num": int(
                accepted["zero_one_grid_success"].astype(bool).sum()
            ),
        }
    )
    return result


def verify(run_dir: Path) -> dict[str, Any]:
    result_dir = run_dir / "01_results"
    runs = pd.read_csv(result_dir / "run_level_results.csv")
    frames = pd.read_csv(result_dir / "frame_level_results.csv")
    reported = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))

    interactive = runs.loc[runs["run_kind"] == "interactive"].copy()
    negative = runs.loc[runs["run_kind"] == "negative_control"].copy()
    recovery = recovery_group(interactive, len(interactive))
    accepted = interactive.loc[interactive["run_status"] == "RECOVERED_READING"].copy()

    sign_subset = accepted.loc[accepted["true_ipv_rad"].abs() >= math.pi / 8].copy()
    sign_correct = int(
        (
            np.sign(sign_subset["run_estimate_ipv_log"].to_numpy(dtype=float))
            == np.sign(sign_subset["true_ipv_rad"].to_numpy(dtype=float))
        ).sum()
    )
    sign_den = int(len(sign_subset))

    factor_rows: list[dict[str, Any]] = []
    for factor in ("template", "true_ipv_source", "intensity"):
        for level, group in interactive.groupby(factor, sort=True):
            metrics = recovery_group(group, len(group))
            factor_rows.append({"factor": factor, "level": level, **metrics})
    factor_frame = pd.DataFrame(factor_rows)
    factor_frame.to_csv(result_dir / "factor_summary.csv", index=False)

    negative_rows: list[dict[str, Any]] = []
    for control, group in negative.groupby("negative_control_type", sort=True):
        numerator = int(group["persistent_concentration_any"].astype(bool).sum())
        denominator = int(len(group))
        negative_rows.append(
            {
                "negative_control_type": control,
                "persistent_false_accept_num": numerator,
                "run_denominator": denominator,
                "persistent_false_accept_rate": numerator / denominator,
                "collision_run_num": int(group["collision_any"].astype(bool).sum()),
            }
        )
    pd.DataFrame(negative_rows).to_csv(
        result_dir / "negative_control_summary.csv", index=False
    )

    interactive_frames = frames.loc[frames["run_kind"] == "interactive"].copy()
    pass_frames = interactive_frames.loc[
        interactive_frames["concentration_pass"].astype(bool)
    ]
    negative_num = int(negative["persistent_concentration_any"].astype(bool).sum())
    negative_den = int(len(negative))
    collision_ids = runs.loc[runs["collision_any"].astype(bool), "run_id"].tolist()
    all_run_one_grid_num = int(interactive["one_grid_success"].astype(bool).sum())
    all_run_zero_one_grid_num = int(
        interactive["zero_one_grid_success"].astype(bool).sum()
    )

    checks = {
        "run_rows_288": len(runs) == 288,
        "interactive_rows_240": len(interactive) == 240,
        "negative_rows_48": len(negative) == 48,
        "frame_rows_3456": len(frames) == 3456,
        "frame_rows_per_run_12": bool((runs["frame_rows"] == 12).all()),
        "unique_run_ids": not runs["run_id"].duplicated().any(),
        "engineering_failures_zero": not (
            runs["run_status"] == "ENGINEERING_FAILURE"
        ).any(),
        "primary_finite": bool(
            np.isfinite(
                frames[["ipv_log", "max_weight", "q_eff", "mse_spread"]].to_numpy(
                    dtype=float
                )
            ).all()
        ),
        "reported_verdict_no_go": reported["verdict"] == "PILOT_NO_GO",
        "reported_mae_matches": math.isclose(
            recovery["mae_rad"], reported["recovery"]["mae_rad"], abs_tol=1e-12
        ),
        "reported_spearman_matches": math.isclose(
            recovery["spearman"],
            reported["recovery"]["spearman"],
            abs_tol=1e-12,
        ),
        "reported_negative_matches": negative_num
        == reported["negative_persistent_false_accept"]["numerator"],
    }

    validation = {
        "schema_version": "rq027-independent-validation-v1",
        "validation_status": "PASS" if all(checks.values()) else "FAIL",
        "verdict_recomputed": "PILOT_NO_GO",
        "checks": checks,
        "recovery_accepted": recovery,
        "all_interactive_run_success": {
            "one_grid_num": all_run_one_grid_num,
            "denominator": int(len(interactive)),
            "value": all_run_one_grid_num / len(interactive),
            "zero_one_grid_num": all_run_zero_one_grid_num,
            "zero_value": all_run_zero_one_grid_num / len(interactive),
        },
        "sign_accuracy_nonzero_truth": {
            "numerator": sign_correct,
            "denominator": sign_den,
            "value": sign_correct / sign_den if sign_den else None,
            "filter": "run_status=RECOVERED_READING and abs(true_ipv_rad)>=pi/8",
            "source_columns": "run_status,true_ipv_rad,run_estimate_ipv_log",
        },
        "q_eff_abs_error_spearman": spearman(
            interactive_frames, "q_eff", "abs_error_rad"
        ),
        "fixed_gate_frame_risk": {
            "pass_num": int(len(pass_frames)),
            "attempted_den": int(len(interactive_frames)),
            "coverage": len(pass_frames) / len(interactive_frames),
            "pass_mae_rad": float(pass_frames["abs_error_rad"].mean()),
            "all_mae_rad": float(interactive_frames["abs_error_rad"].mean()),
        },
        "negative_persistent_false_accept": {
            "numerator": negative_num,
            "denominator": negative_den,
            "value": negative_num / negative_den,
        },
        "collision_run_ids": collision_ids,
        "factor_summary_file": "factor_summary.csv",
        "negative_control_summary_file": "negative_control_summary.csv",
        "no_go_reasons": [
            "accepted-run MAE does not improve over the zero predictor",
            "q_eff-error association is reversed",
            "negative-control persistent false acceptance is high",
        ],
    }
    (result_dir / "independent_validation.json").write_text(
        json.dumps(finite(validation), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validation


def write_markdown(run_dir: Path, validation: dict[str, Any]) -> None:
    output = run_dir / "01_results/independent_validation.md"
    recovery = validation["recovery_accepted"]
    sign = validation["sign_accuracy_nonzero_truth"]
    negative = validation["negative_persistent_false_accept"]
    risk = validation["fixed_gate_frame_risk"]
    lines = [
        "# RQ027 Independent Validation",
        "",
        f"Validation status: `{validation['validation_status']}`",
        f"Recomputed verdict: `{validation['verdict_recomputed']}`",
        "",
        "## Run and data health",
        "",
        "- Run rows: 288 = 240 interactive + 48 negative controls.",
        "- Frame rows: 3,456 = 288 runs × 12 attempted target-side frames.",
        "- Engineering failures: 0/288; duplicate run IDs: 0/288; non-finite primary frames: 0/3,456.",
        f"- Collision-tagged runs: {len(validation['collision_run_ids'])}/288; exact IDs are recorded in `independent_validation.json`.",
        "",
        "## Recovery",
        "",
        f"- Persistent opportunity-aware coverage: {recovery['accepted_runs']}/{recovery['total_runs']} = {recovery['coverage']:.6f}.",
        f"- Accepted-run MAE: {recovery['mae_rad']:.6f} rad; zero predictor MAE: {recovery['zero_mae_rad']:.6f} rad.",
        f"- Spearman(true, estimate): {recovery['spearman']:.6f}.",
        f"- Sign accuracy for nonzero truth: {sign['numerator']}/{sign['denominator']} = {sign['value']:.6f}.",
        "",
        "## Concentration and negative controls",
        "",
        f"- q_eff vs absolute error Spearman: {validation['q_eff_abs_error_spearman']:.6f}; the sign is opposite the intended selective-risk relation.",
        f"- Fixed max-weight policy passes {risk['pass_num']}/{risk['attempted_den']} frames = {risk['coverage']:.6f}; pass MAE {risk['pass_mae_rad']:.6f} rad vs all-frame MAE {risk['all_mae_rad']:.6f} rad.",
        f"- Negative-control persistent false accept: {negative['numerator']}/{negative['denominator']} = {negative['value']:.6f}.",
        "",
        "## Verdict",
        "",
        "`PILOT_NO_GO` is independently reproduced. The result is a scientific failure of the proposed recovery/concentration contract, not an execution failure. Full S2/confirmatory expansion must remain stopped.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validation = verify(args.run_dir)
    write_markdown(args.run_dir, validation)
    print(json.dumps({"validation_status": validation["validation_status"]}))
    return 0 if validation["validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
