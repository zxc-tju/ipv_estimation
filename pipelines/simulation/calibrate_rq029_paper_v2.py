#!/usr/bin/env python3
"""Build an RQ029 v2 overlay calibrated to the manuscript's current records.

The v1 raw replay bytes and ego trajectories are preserved. V2 replaces the
nearest-background proxy with each Shanghai scenario's fixed analysis
counterpart, recomputes three-second raw outcomes, and calibrates synthetic
run/case dependence toward the rounded intervals printed in the manuscript.
It remains synthetic and does not recover unavailable participant records.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.simulation.generate_rq029_human_synthetic import (
    CASE_TO_SCENARIO_SHANGHAI,
    DATA_STATUS,
    DEFAULT_SOURCE_SESSION,
    REPO_ROOT,
    SCENARIOS,
    _bounded_allocate,
    _calibrated_quantile_values,
    _sha256,
    _stable_seed,
)


BASE_OUTPUT = REPO_ROOT / "data/derived/rq029_human_synthetic_microdata/v1"
DEFAULT_OUTPUT = REPO_ROOT / "data/derived/rq029_human_synthetic_microdata/v2_paper_aligned"
BASE_TARGETS = REPO_ROOT / "reports/plans/RQ029_human_statistical_targets_v1.json"
PAPER_TARGETS = REPO_ROOT / "reports/plans/RQ029_paper_record_targets_v2.json"
AV_REFERENCE = (
    REPO_ROOT
    / ".codex-fleet/rq022-matched-scenario/work/T1_target_figure/av_reference_values.json"
)
SCHEMA_VERSION = "RQ029-human-synthetic-microdata-v2-paper-aligned"
CALIBRATION_SEED = 20260902

DESIGNATED_COUNTERPARTS = {
    "A1": "1200002",
    "A2": "2400041",
    "A3": "3700075",
    "A4": "4800123",
    "A5": "7000165",
    "A6": "7900187",
    "A7": "9400207",
    "B1": "10500230",
    "B2": "11700259",
    "B3": "12600290",
    "B4": "13200308",
    "C1": "14700321",
    "C2": "16300349",
    "C3": "17800415",
    "C4": "18400450",
}


def _prepare_output(base: Path, output: Path, replace: bool) -> None:
    if output.exists():
        if not replace:
            raise FileExistsError(f"output exists; pass --replace: {output}")
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("refusing replacement: output has no manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("refusing replacement: output is not RQ029 v2")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    if os.uname().sysname == "Darwin":
        result = subprocess.run(
            ["cp", "-cR", str(base), str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        copied = result.returncode == 0
    if not copied:
        shutil.copytree(base, output)
    validation = output / "validation"
    if validation.exists():
        shutil.rmtree(validation)
    validation.mkdir()


def _actor_number(actor: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = actor.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _load_designated_counterparts(source_session: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_counter = {scenario: 0 for scenario in SCENARIOS}
    source_path = source_session / "vehicle_perception_simulation_trajectory.log"
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            scenario = CASE_TO_SCENARIO_SHANGHAI[int(record["caseId"])]
            ego = next(
                actor
                for group in record["participantTrajectories"]
                for actor in group.get("value", [])
                if actor.get("isPerception") == 0
            )
            actors = [
                actor
                for group in record["participantTrajectories"]
                if group.get("role") == "mvSimulation"
                for actor in group.get("value", [])
            ]
            counterpart = next(
                (
                    actor
                    for actor in actors
                    if str(actor.get("id", "")) == DESIGNATED_COUNTERPARTS[scenario]
                ),
                None,
            )
            row = {
                "scenario_id": scenario,
                "scenario_frame_index": frame_counter[scenario],
                "background_timestamp_ms": int(ego["globalTimeStamp"]),
                "designated_counterpart_id": DESIGNATED_COUNTERPARTS[scenario],
                "counterpart_latitude": np.nan,
                "counterpart_longitude": np.nan,
                "counterpart_speed_kmh": np.nan,
                "counterpart_course_deg": np.nan,
            }
            if counterpart is not None:
                row.update(
                    {
                        "counterpart_latitude": _actor_number(
                            counterpart, ("latitude", "lat")
                        ),
                        "counterpart_longitude": _actor_number(
                            counterpart, ("longitude", "lng", "lon")
                        ),
                        "counterpart_speed_kmh": _actor_number(counterpart, ("speed",)),
                        "counterpart_course_deg": _actor_number(
                            counterpart, ("courseAngle",)
                        ),
                    }
                )
            rows.append(row)
            frame_counter[scenario] += 1
    background = pd.DataFrame(rows)
    parts: list[pd.DataFrame] = []
    for _, group in background.groupby("scenario_id", sort=False):
        group = group.sort_values("scenario_frame_index").copy()
        group["counterpart_observed"] = group.counterpart_speed_kmh.notna()
        interpolate_columns = [
            "counterpart_latitude",
            "counterpart_longitude",
            "counterpart_speed_kmh",
            "counterpart_course_deg",
        ]
        group[interpolate_columns] = group[interpolate_columns].interpolate(
            method="linear",
            limit=30,
            limit_direction="both",
        )
        group["counterpart_interpolated"] = (
            ~group.counterpart_observed & group.counterpart_speed_kmh.notna()
        )
        speed = group.counterpart_speed_kmh.to_numpy(float)
        timestamp = group.background_timestamp_ms.to_numpy(float) / 1000.0
        acceleration = np.full(len(group), np.nan)
        for index in range(1, len(group)):
            delta_t = timestamp[index] - timestamp[index - 1]
            if (
                np.isfinite(speed[index])
                and np.isfinite(speed[index - 1])
                and 0.05 <= delta_t <= 0.30
            ):
                acceleration[index] = (speed[index] - speed[index - 1]) / 3.6 / delta_t
        speed_drop = np.full(len(group), np.nan)
        speed_range = np.full(len(group), np.nan)
        brake_num = np.zeros(len(group), dtype=int)
        brake_den = np.zeros(len(group), dtype=int)
        for index in range(len(group)):
            window_speed = speed[index: index + 31]
            window_acceleration = acceleration[index: index + 31]
            valid_speed = window_speed[np.isfinite(window_speed)]
            valid_acceleration = window_acceleration[np.isfinite(window_acceleration)]
            if np.isfinite(speed[index]) and len(valid_speed):
                speed_drop[index] = max(0.0, speed[index] - float(valid_speed.min()))
                speed_range[index] = float(valid_speed.max() - valid_speed.min())
            brake_num[index] = int((valid_acceleration < -3.0).sum())
            brake_den[index] = int(len(valid_acceleration))
        group["counterpart_acceleration_mps2"] = acceleration
        group["anchor_speed_drop_kmh_raw"] = speed_drop
        group["window_speed_range_kmh_raw"] = speed_range
        group["n_frames_lt_m3_raw"] = brake_num
        group["n_frames_total_raw"] = brake_den
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _velocity(speed_kmh: np.ndarray, course_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = speed_kmh / 3.6
    angle = np.radians(course_deg)
    return speed * np.sin(angle), speed * np.cos(angle)


def _enhance_trajectories(
    trajectories: pd.DataFrame,
    background: pd.DataFrame,
) -> pd.DataFrame:
    drop_columns = [
        column
        for column in (
            "counterpart_id",
            "counterpart_speed_kmh",
            "counterpart_acceleration_mps2",
            "counterpart_distance_m",
            "closing_rate_mps",
            "current_ttc_raw_s",
        )
        if column in trajectories
    ]
    enhanced = trajectories.drop(columns=drop_columns).merge(
        background,
        on=["scenario_id", "scenario_frame_index"],
        how="left",
        validate="many_to_one",
    )
    latitude = enhanced.ego_latitude.to_numpy(float)
    longitude = enhanced.ego_longitude.to_numpy(float)
    cp_latitude = enhanced.counterpart_latitude.to_numpy(float)
    cp_longitude = enhanced.counterpart_longitude.to_numpy(float)
    dx = (cp_longitude - longitude) * 111_320.0 * np.cos(np.radians(latitude))
    dy = (cp_latitude - latitude) * 111_320.0
    distance = np.hypot(dx, dy)
    cp_vx, cp_vy = _velocity(
        enhanced.counterpart_speed_kmh.to_numpy(float),
        enhanced.counterpart_course_deg.to_numpy(float),
    )
    ego_vx = enhanced.ego_vx_mps.to_numpy(float)
    ego_vy = enhanced.ego_vy_mps.to_numpy(float)
    closing = -((cp_vx - ego_vx) * dx + (cp_vy - ego_vy) * dy) / distance
    closing[~np.isfinite(closing) | (distance <= 1e-9)] = np.nan
    current_ttc = np.full(len(enhanced), np.nan)
    np.divide(distance, closing, out=current_ttc, where=closing > 0)
    enhanced["counterpart_id"] = enhanced.designated_counterpart_id
    enhanced["counterpart_distance_m"] = distance
    enhanced["closing_rate_mps"] = closing
    enhanced["current_ttc_raw_s"] = current_ttc
    enhanced["future_min_ttc_raw_s"] = np.nan
    for _, group in enhanced.groupby("run_id"):
        indices = group.sort_values("scenario_frame_index").index
        values = enhanced.loc[indices, "current_ttc_raw_s"].to_numpy(float)
        future_min = np.asarray(
            [
                np.nanmin(values[index: index + 31])
                if np.isfinite(values[index: index + 31]).any()
                else np.nan
                for index in range(len(values))
            ]
        )
        enhanced.loc[indices, "future_min_ttc_raw_s"] = future_min
    return enhanced.sort_values(["driver_id", "scenario_id", "scenario_frame_index"])


def _join_candidate_raw(
    candidates: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    raw_columns = [
        "driver_id",
        "scenario_id",
        "scenario_frame_index",
        "designated_counterpart_id",
        "counterpart_observed",
        "counterpart_interpolated",
        "counterpart_latitude",
        "counterpart_longitude",
        "counterpart_speed_kmh",
        "counterpart_course_deg",
        "counterpart_acceleration_mps2",
        "counterpart_distance_m",
        "closing_rate_mps",
        "current_ttc_raw_s",
        "future_min_ttc_raw_s",
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_lt_m3_raw",
        "n_frames_total_raw",
    ]
    replace = [column for column in raw_columns[3:] if column in candidates]
    return candidates.drop(columns=replace).merge(
        trajectories[raw_columns],
        on=["driver_id", "scenario_id", "scenario_frame_index"],
        how="left",
        validate="many_to_one",
    )


def _bootstrap_rate_ci(
    n_both: np.ndarray,
    n_flagged: np.ndarray,
    *,
    seed: int = CALIBRATION_SEED,
    draws: int = 1000,
) -> list[float]:
    rng = np.random.default_rng(_stable_seed(seed, "alpha90_bootstrap"))
    values: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, len(n_both), len(n_both))
        values.append(float(n_flagged[indices].sum() / n_both[indices].sum()))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _case_selection_mask() -> np.ndarray:
    selected = np.zeros((len(SCENARIOS), 20), dtype=bool)
    for scenario_index in range(len(SCENARIOS)):
        count = 13 if scenario_index < 6 else 12
        indices = (np.arange(count) + scenario_index * 3) % 20
        selected[scenario_index, indices] = True
    if int(selected.sum()) != 186:
        raise AssertionError("synthetic case selection must contain 186 runs")
    return selected


def _calibrate_run_counts(
    candidates: pd.DataFrame,
    targets: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    drivers = sorted(candidates.driver_id.unique())
    scenario_order = list(SCENARIOS)
    unit = (
        candidates.groupby(["scenario_id", "driver_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_candidate"})
    )
    capacity = (
        unit.pivot(index="scenario_id", columns="driver_id", values="n_candidate")
        .reindex(index=scenario_order, columns=drivers)
        .to_numpy(int)
    )
    selected_case = _case_selection_mask()
    n_both = np.zeros_like(capacity)
    for scenario_index, scenario in enumerate(scenario_order):
        total = targets["per_scenario_alpha90"][scenario]["n_both"]
        weights = np.where(selected_case[scenario_index], 5.0, 1.0)
        n_both[scenario_index] = _bounded_allocate(
            total,
            weights,
            capacity[scenario_index],
            np.ones(len(drivers), dtype=int),
        )

    target_ci = np.asarray(targets["flag_rate_ci95_alpha90_target"], dtype=float)
    scenario_flagged = np.asarray(
        [targets["per_scenario_alpha90"][scenario]["n_flagged"] for scenario in scenario_order]
    )
    best: tuple[float, np.ndarray, list[float], dict[str, Any]] | None = None
    candidates_grid = [
        (weight, driver_sigma, run_sigma, correlation, seed)
        for weight in (1.0, 2.0, 4.0, 8.0)
        for driver_sigma in (0.2, 0.4, 0.6)
        for run_sigma in (0.2, 0.4, 0.6, 1.0)
        for correlation in (0.0, 0.5, 0.8, 0.95)
        for seed in (333, 999, 12345)
    ]
    for weight, driver_sigma, run_sigma, correlation, seed in candidates_grid:
        rng = np.random.default_rng(seed)
        driver_effect = rng.normal(size=len(drivers))
        run_effect = rng.normal(size=n_both.shape)
        flagged = np.zeros_like(n_both)
        for scenario_index in range(len(scenario_order)):
            standardized_n = (
                (n_both[scenario_index] - n_both[scenario_index].mean())
                / (n_both[scenario_index].std() + 1e-9)
            )
            latent = (
                driver_sigma * driver_effect
                + run_sigma * run_effect[scenario_index]
                + correlation * standardized_n
            )
            weights = np.exp(np.clip(latent, -5.0, 5.0))
            weights *= np.where(selected_case[scenario_index], weight, 1.0)
            flagged[scenario_index] = _bounded_allocate(
                int(scenario_flagged[scenario_index]),
                weights,
                n_both[scenario_index],
            )
        ci = _bootstrap_rate_ci(n_both.ravel(), flagged.ravel())
        driver_rate = flagged.sum(axis=0) / n_both.sum(axis=0)
        run_rate = flagged / n_both
        regularizer = 0.0
        regularizer += max(0.0, float(driver_rate.max()) - 0.15) * 10.0
        regularizer += max(0.0, float(run_rate.max()) - 0.50) * 2.0
        regularizer += max(0, int((flagged == 0).sum()) - 180) * 0.0001
        endpoint_loss = float(np.sum((np.asarray(ci) - target_ci) ** 2))
        score = endpoint_loss + regularizer
        metadata = {
            "selected_case_weight": weight,
            "driver_sigma": driver_sigma,
            "run_sigma": run_sigma,
            "n_correlation": correlation,
            "latent_seed": seed,
            "endpoint_loss": endpoint_loss,
            "regularizer": regularizer,
            "max_driver_rate": float(driver_rate.max()),
            "max_run_rate": float(run_rate.max()),
            "zero_flag_runs": int((flagged == 0).sum()),
        }
        if best is None or score < best[0]:
            best = (score, flagged.copy(), ci, metadata)
    if best is None:
        raise AssertionError("run calibration grid was empty")
    flagged = best[1]

    scenario_lower = _bounded_allocate(
        targets["flag_counts"]["90"]["n_below"],
        scenario_flagged.astype(float),
        scenario_flagged,
    )
    n_lower = np.zeros_like(flagged)
    for scenario_index in range(len(scenario_order)):
        weights = np.maximum(flagged[scenario_index], 1).astype(float)
        weights *= np.where(selected_case[scenario_index], 10.0, 1.0)
        n_lower[scenario_index] = _bounded_allocate(
            int(scenario_lower[scenario_index]),
            weights,
            flagged[scenario_index],
        )
    n_upper = flagged - n_lower
    selected_nonupper = int((n_both - n_upper)[selected_case].sum())
    required_counterpart = (
        targets["signature"]["counterpart"]["n_lower"]
        + targets["signature"]["counterpart"]["n_inside"]
    )
    if selected_nonupper < required_counterpart:
        raise AssertionError("selected synthetic cases cannot hold counterpart target rows")

    base_gate1 = targets["gates"]["n_gate1_pass"] - targets["gates"]["n_both_gates"]
    gate_room = capacity - n_both
    order = np.random.default_rng(_stable_seed(CALIBRATION_SEED, "v2_gate1")).permutation(
        gate_room.size
    )
    gate_extra_ordered = _bounded_allocate(
        base_gate1,
        gate_room.ravel()[order].astype(float),
        gate_room.ravel()[order],
    )
    gate_extra = np.empty(gate_room.size, dtype=int)
    gate_extra[order] = gate_extra_ordered
    gate_extra = gate_extra.reshape(gate_room.shape)

    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenario_order):
        for driver_index, driver in enumerate(drivers):
            rows.append(
                {
                    "driver_id": driver,
                    "scenario_id": scenario,
                    "run_id": f"human_synthetic:{driver}:{scenario}",
                    "n_candidate": int(capacity[scenario_index, driver_index]),
                    "n_gate1": int(
                        n_both[scenario_index, driver_index]
                        + gate_extra[scenario_index, driver_index]
                    ),
                    "n_both": int(n_both[scenario_index, driver_index]),
                    "n_flagged_90": int(flagged[scenario_index, driver_index]),
                    "n_below_90": int(n_lower[scenario_index, driver_index]),
                    "n_above_90": int(n_upper[scenario_index, driver_index]),
                    "high_support_run": bool(
                        selected_case[scenario_index, driver_index]
                    ),
                }
            )
    metadata = {
        **best[3],
        "alpha90_ci95": best[2],
        "target_ci95": target_ci.tolist(),
        "high_support_runs": int(selected_case.sum()),
        "selected_nonupper_capacity": selected_nonupper,
    }
    return pd.DataFrame(rows), metadata


def _raw_lower_score(frame: pd.DataFrame) -> np.ndarray:
    ttc = frame.future_min_ttc_raw_s.to_numpy(float)
    drop = frame.anchor_speed_drop_kmh_raw.to_numpy(float)
    speed_range = frame.window_speed_range_kmh_raw.to_numpy(float)
    brake_share = (
        frame.n_frames_lt_m3_raw.to_numpy(float)
        / np.maximum(frame.n_frames_total_raw.to_numpy(float), 1.0)
    )
    score = np.zeros(len(frame), dtype=float)
    score -= np.abs(np.log(np.clip(ttc, 0.2, 100.0) / 6.94))
    score -= 0.55 * np.abs(np.log((np.clip(drop, 0.0, None) + 0.2) / 3.76))
    score -= 0.45 * np.abs(np.log((np.clip(speed_range, 0.0, None) + 0.2) / 4.52))
    score -= 1.5 * np.abs(brake_share - 0.02577)
    score[~np.isfinite(score)] = -1e6
    return score


def _raw_inside_score(frame: pd.DataFrame) -> np.ndarray:
    ttc = frame.future_min_ttc_raw_s.to_numpy(float)
    drop = frame.anchor_speed_drop_kmh_raw.to_numpy(float)
    speed_range = frame.window_speed_range_kmh_raw.to_numpy(float)
    score = np.zeros(len(frame), dtype=float)
    score -= 0.35 * np.abs(np.log(np.clip(ttc, 0.2, 100.0) / 8.54))
    score -= 0.75 * np.abs(np.log((np.clip(drop, 0.0, None) + 0.2) / 1.65))
    score -= 0.75 * np.abs(
        np.log((np.clip(speed_range, 0.0, None) + 0.2) / 3.18)
    )
    score += 0.85 * (np.isfinite(ttc) & (ttc >= 2.0))
    score += 3.0 * (drop >= 1.45)
    score += 3.0 * (speed_range <= 2.98)
    score[~np.isfinite(score)] = -1e6
    return score


def _apply_run_counts(
    candidates: pd.DataFrame,
    run_counts: pd.DataFrame,
    targets: dict[str, Any],
) -> pd.DataFrame:
    out = candidates.copy()
    out["status"] = "ABSTAIN"
    out["reason_code"] = "NEAR_UNIFORM"
    out["mechanism2_gate_ok"] = False
    for alpha in ("80", "90", "95"):
        out[f"band_{alpha}"] = pd.NA
    for row in run_counts.itertuples(index=False):
        indices = out.index[out.run_id.eq(row.run_id)].to_numpy()
        rng = np.random.default_rng(_stable_seed(CALIBRATION_SEED, row.run_id, "v2_rows"))
        lower_scores = _raw_lower_score(out.loc[indices]) + 0.05 * rng.random(len(indices))
        lower_indices = indices[np.argsort(lower_scores)[::-1][: row.n_below_90]]
        after_lower_pool = np.setdiff1d(indices, lower_indices, assume_unique=False)
        inside_scores = _raw_inside_score(out.loc[after_lower_pool])
        inside_scores += 0.05 * rng.random(len(after_lower_pool))
        remaining_both_count = row.n_both - row.n_below_90
        after_lower_order = after_lower_pool[np.argsort(inside_scores)[::-1]]
        after_lower = after_lower_order[:remaining_both_count]
        both_indices = np.concatenate([lower_indices, after_lower])
        remainder = after_lower_order[remaining_both_count:]
        upper_indices = rng.permutation(after_lower)[: row.n_above_90]
        inside_indices = np.setdiff1d(after_lower, upper_indices, assume_unique=False)
        out.loc[both_indices, ["status", "reason_code", "mechanism2_gate_ok"]] = [
            "OK",
            None,
            True,
        ]
        out.loc[lower_indices, "band_90"] = "lower"
        out.loc[upper_indices, "band_90"] = "upper"
        out.loc[inside_indices, "band_90"] = "inside"
        gate1_extra = row.n_gate1 - row.n_both
        extra_indices = rng.permutation(remainder)[:gate1_extra]
        out.loc[extra_indices, ["status", "reason_code"]] = ["OK", None]

    both = out.status.eq("OK") & out.mechanism2_gate_ok
    out.loc[both, ["band_80", "band_95"]] = "inside"
    lower90 = out.index[out.band_90.eq("lower")].to_numpy()
    upper90 = out.index[out.band_90.eq("upper")].to_numpy()
    inside90 = out.index[out.band_90.eq("inside")].to_numpy()
    rng = np.random.default_rng(_stable_seed(CALIBRATION_SEED, "v2_nested_bands"))
    out.loc[lower90, "band_80"] = "lower"
    out.loc[upper90, "band_80"] = "upper"
    lower95 = rng.permutation(lower90)[: targets["flag_counts"]["95"]["n_below"]]
    upper95 = rng.permutation(upper90)[: targets["flag_counts"]["95"]["n_above"]]
    out.loc[lower95, "band_95"] = "lower"
    out.loc[upper95, "band_95"] = "upper"
    extra_lower = targets["flag_counts"]["80"]["n_below"] - len(lower90)
    extra_upper = targets["flag_counts"]["80"]["n_above"] - len(upper90)
    shuffled_inside = rng.permutation(inside90)
    out.loc[shuffled_inside[:extra_lower], "band_80"] = "lower"
    out.loc[
        shuffled_inside[extra_lower: extra_lower + extra_upper], "band_80"
    ] = "upper"
    out["ipv_log"] = np.nan
    out.loc[out.status.eq("OK"), "ipv_log"] = 0.0
    out.loc[out.band_80.eq("lower"), "ipv_log"] = -0.5
    out.loc[out.band_80.eq("upper"), "ipv_log"] = 0.5
    out.loc[out.band_90.eq("lower"), "ipv_log"] = -0.7
    out.loc[out.band_90.eq("upper"), "ipv_log"] = 0.7
    out.loc[out.band_95.eq("lower"), "ipv_log"] = -0.9
    out.loc[out.band_95.eq("upper"), "ipv_log"] = 0.9
    return out


def _select_quantile_rows(
    frame: pd.DataFrame,
    value_column: str,
    count: int,
    target_values: np.ndarray,
) -> np.ndarray:
    eligible = frame.index[frame[value_column].notna()].to_numpy()
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} finite {value_column} rows for target {count}")
    raw = frame.loc[eligible, value_column].to_numpy(float)
    order = np.argsort(raw)
    sorted_indices = eligible[order]
    sorted_raw = raw[order]
    selected: list[int] = []
    left = 0
    for position, target in enumerate(target_values):
        remaining = count - position - 1
        candidate = int(np.searchsorted(sorted_raw, target, side="left"))
        candidate = max(left, min(candidate, len(sorted_raw) - remaining - 1))
        if candidate > left:
            before = candidate - 1
            if abs(sorted_raw[before] - target) < abs(sorted_raw[candidate] - target):
                candidate = before
        selected.append(int(sorted_indices[candidate]))
        left = candidate + 1
    return np.asarray(selected, dtype=int)


def _assign_clustered_values(
    frame: pd.DataFrame,
    values: np.ndarray,
    raw_column: str,
    cluster_column: str,
    rho: float,
    seed: int,
    cluster_sign: float = 1.0,
) -> np.ndarray:
    clusters = sorted(frame[cluster_column].unique())
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    row_cluster = np.asarray([cluster_index[value] for value in frame[cluster_column]])
    rng = np.random.default_rng(seed)
    cluster_effect = rng.normal(size=len(clusters)) * cluster_sign
    raw = frame[raw_column].to_numpy(float)
    raw_rank = pd.Series(raw).rank(method="average", pct=True).fillna(0.5).to_numpy()
    raw_rank = (raw_rank - 0.5) * math.sqrt(12.0)
    noise = rng.normal(size=len(frame))
    residual = 0.85 * raw_rank + 0.15 * noise
    score = rho * cluster_effect[row_cluster] + math.sqrt(max(0.0, 1 - rho**2)) * residual
    order = np.argsort(score)
    assigned = np.empty(len(frame), dtype=float)
    assigned[order] = np.sort(values)
    return assigned


def _bootstrap_share_difference(
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    threshold: float,
    draws: int,
    seed: int,
) -> list[float]:
    grouped = []
    for cluster, group in frame.groupby(cluster_column):
        lower = group[group.band_90.eq("lower")][value_column].dropna().to_numpy(float)
        inside = group[group.band_90.eq("inside")][value_column].dropna().to_numpy(float)
        grouped.append(
            (
                cluster,
                int((lower < threshold).sum()),
                len(lower),
                int((inside < threshold).sum()),
                len(inside),
            )
        )
    values = np.asarray([row[1:] for row in grouped], dtype=int)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if sampled[1] and sampled[3]:
            estimates.append(float(sampled[0] / sampled[1] - sampled[2] / sampled[3]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _weighted_bootstrap_medians(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    draw_counts: np.ndarray,
    cluster_index: dict[str, int],
) -> np.ndarray:
    row_cluster = np.asarray([cluster_index[value] for value in cluster_ids])
    order = np.argsort(values)
    sorted_values = values[order]
    weights = draw_counts[:, row_cluster[order]]
    cumulative = np.cumsum(weights, axis=1)
    totals = weights.sum(axis=1)
    lower_position = (totals - 1) // 2
    upper_position = totals // 2
    lower_index = np.argmax(cumulative > lower_position[:, None], axis=1)
    upper_index = np.argmax(cumulative > upper_position[:, None], axis=1)
    return 0.5 * (sorted_values[lower_index] + sorted_values[upper_index])


def _bootstrap_ratio_ci(
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    draws: int,
    seed: int,
) -> list[float]:
    clusters = sorted(frame[cluster_column].unique())
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draw_counts = np.stack(
        [
            np.bincount(
                rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)
            )
            for _ in range(draws)
        ]
    ).astype(np.int16)
    lower = frame[frame.band_90.eq("lower")]
    inside = frame[frame.band_90.eq("inside")]
    lower_median = _weighted_bootstrap_medians(
        lower[value_column].to_numpy(float),
        lower[cluster_column].to_numpy(str),
        draw_counts,
        cluster_index,
    )
    inside_median = _weighted_bootstrap_medians(
        inside[value_column].to_numpy(float),
        inside[cluster_column].to_numpy(str),
        draw_counts,
        cluster_index,
    )
    ratio = lower_median / inside_median
    return [float(value) for value in np.quantile(ratio, [0.025, 0.975])]


def _select_ttc_rows(
    eligible: pd.DataFrame,
    spec: dict[str, Any],
    target_values: np.ndarray,
) -> np.ndarray:
    finite = eligible[eligible.future_min_ttc_raw_s.notna()]
    tail = finite[finite.future_min_ttc_raw_s < 2.0]
    non_tail = finite[finite.future_min_ttc_raw_s >= 2.0]
    missing = eligible[eligible.future_min_ttc_raw_s.isna()]
    tail_count = min(int(spec["lt2_num"]), len(tail))
    tail_targets = target_values[target_values < 2.0][:tail_count]
    tail_indices = _select_quantile_rows(
        tail,
        "future_min_ttc_raw_s",
        tail_count,
        tail_targets,
    )
    remaining = int(spec["n"] - tail_count)
    non_tail_count = min(remaining, len(non_tail))
    non_tail_targets = target_values[target_values >= 2.0][:non_tail_count]
    non_tail_indices = _select_quantile_rows(
        non_tail,
        "future_min_ttc_raw_s",
        non_tail_count,
        non_tail_targets,
    )
    missing_count = remaining - non_tail_count
    if missing_count > len(missing):
        raise ValueError("insufficient finite and explicitly missing TTC rows")
    missing_indices = missing.sort_index().index[:missing_count].to_numpy()
    return np.concatenate([tail_indices, non_tail_indices, missing_indices])


def _refine_ttc_tail_allocations(
    capacity: dict[str, np.ndarray],
    allocations: dict[str, np.ndarray],
    target_ci: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cluster_count = len(capacity["lower"])
    rng = np.random.default_rng(2701)
    sampled = rng.integers(0, cluster_count, size=(1000, cluster_count))
    draw_counts = np.stack(
        [np.bincount(row, minlength=cluster_count) for row in sampled]
    ).astype(np.int16)
    lower_denominator = draw_counts @ capacity["lower"]
    inside_denominator = draw_counts @ capacity["inside"]
    lower = allocations["lower"].copy()
    inside = allocations["inside"].copy()
    differences = (
        (draw_counts @ lower) / lower_denominator
        - (draw_counts @ inside) / inside_denominator
    )

    def objective(values: np.ndarray) -> tuple[np.ndarray, float]:
        interval = np.quantile(values, [0.025, 0.975])
        score = float(np.sum(((interval - target_ci) / 0.005) ** 2))
        if interval[1] >= 0:
            score += 1000.0
        return interval, score

    interval, current_score = objective(differences)
    best_score = current_score
    best_interval = interval.copy()
    best_lower = lower.copy()
    best_inside = inside.copy()
    search_rng = np.random.default_rng(123)
    iteration = -1
    for iteration in range(300_000):
        band = "lower" if search_rng.random() < 0.2 else "inside"
        values = lower if band == "lower" else inside
        band_capacity = capacity[band]
        donors = np.flatnonzero(values > 0)
        recipients = np.flatnonzero(values < band_capacity)
        if not len(donors) or not len(recipients):
            continue
        donor = int(search_rng.choice(donors))
        recipient = int(search_rng.choice(recipients))
        if donor == recipient:
            continue
        denominator = lower_denominator if band == "lower" else inside_denominator
        delta = (draw_counts[:, recipient] - draw_counts[:, donor]) / denominator
        proposed_differences = differences + (delta if band == "lower" else -delta)
        proposed_interval, proposed_score = objective(proposed_differences)
        temperature = max(0.001, 1.0 * (1.0 - iteration / 300_000))
        accept = proposed_score < current_score
        if not accept:
            accept = search_rng.random() < math.exp(
                min(0.0, (current_score - proposed_score) / temperature)
            )
        if not accept:
            continue
        values[donor] -= 1
        values[recipient] += 1
        differences = proposed_differences
        current_score = proposed_score
        if proposed_score < best_score:
            best_score = proposed_score
            best_interval = proposed_interval.copy()
            best_lower = lower.copy()
            best_inside = inside.copy()
            if (
                np.all(np.abs(best_interval - target_ci) <= 0.005)
                and best_interval[1] < 0
            ):
                break
    return {"lower": best_lower, "inside": best_inside}, {
        "optimizer_iterations": iteration + 1,
        "optimizer_score": best_score,
        "optimizer_ci95": best_interval.tolist(),
    }


def _calibrate_ttc(
    candidates: pd.DataFrame,
    targets: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = candidates.copy()
    out["future_min_ttc_s_calibrated"] = np.nan
    selected_frames: list[pd.DataFrame] = []
    values_by_band: dict[str, np.ndarray] = {}
    for band in ("lower", "inside"):
        spec = targets["signature"]["ego_ttc"][band]
        eligible = out[out.band_90.eq(band)]
        values = _calibrated_quantile_values(
            spec["n"],
            spec["q25"],
            spec["q50"],
            spec["q75"],
            spec["lt2_num"],
            2.0,
        )
        selected_indices = _select_ttc_rows(eligible, spec, values)
        frame = out.loc[selected_indices].copy()
        selected_frames.append(frame)
        values_by_band[band] = values
    selected = pd.concat(selected_frames).sort_index()
    runs = sorted(selected.run_id.unique())
    run_index = {run_id: index for index, run_id in enumerate(runs)}
    capacity: dict[str, np.ndarray] = {}
    for band in ("lower", "inside"):
        capacity[band] = np.zeros(len(runs), dtype=int)
        for run_id, group in selected[selected.band_90.eq(band)].groupby("run_id"):
            capacity[band][run_index[run_id]] = len(group)
    # Frozen after a bounded search over synthetic cluster allocations. The
    # chosen point preserves the target tail counts and excludes zero without
    # pretending to reconstruct the unavailable observed run effects.
    allocation_seed = 1835364077
    rng = np.random.default_rng(allocation_seed)
    shared_effect = rng.normal(size=len(runs))
    independent_effect = rng.normal(size=len(runs))
    tail_allocations = {
        "lower": _bounded_allocate(
            targets["signature"]["ego_ttc"]["lower"]["lt2_num"],
            np.exp(np.clip(0.437574685609432 * shared_effect, -8.0, 8.0)),
            capacity["lower"],
        ),
        "inside": _bounded_allocate(
            targets["signature"]["ego_ttc"]["inside"]["lt2_num"],
            np.exp(
                np.clip(
                    3.781234284430017
                    * (shared_effect + 0.3 * independent_effect),
                    -8.0,
                    8.0,
                )
            ),
            capacity["inside"],
        ),
    }
    target_ci = np.asarray(
        targets["signature"]["ego_ttc"]["lt2_diff_ci95_target"], dtype=float
    )
    tail_allocations, optimizer_meta = _refine_ttc_tail_allocations(
        capacity, tail_allocations, target_ci
    )
    chosen = selected.copy()
    chosen["calibrated"] = np.nan
    for band in ("lower", "inside"):
        band_frame = chosen[chosen.band_90.eq(band)]
        values = values_by_band[band]
        tail_values = np.sort(values[values < 2.0])
        other_values = np.sort(values[values >= 2.0])
        tail_rows: list[int] = []
        for run_id, group in band_frame.groupby("run_id"):
            count = int(tail_allocations[band][run_index[run_id]])
            ordered = group.sort_values("future_min_ttc_raw_s")
            tail_rows.extend(ordered.index[:count].tolist())
        other_rows = np.setdiff1d(band_frame.index.to_numpy(), np.asarray(tail_rows))
        tail_order = chosen.loc[tail_rows].sort_values("future_min_ttc_raw_s").index
        other_order = chosen.loc[other_rows].sort_values("future_min_ttc_raw_s").index
        chosen.loc[tail_order, "calibrated"] = tail_values
        chosen.loc[other_order, "calibrated"] = other_values
    out.loc[chosen.index, "future_min_ttc_s_calibrated"] = chosen.calibrated
    final_ci = _bootstrap_share_difference(
        chosen,
        "calibrated",
        "run_id",
        2.0,
        1000,
        2701,
    )
    metadata = {
        "allocation_seed": allocation_seed,
        "lower_effect_scale": 0.437574685609432,
        "inside_effect_scale": 3.781234284430017,
        **optimizer_meta,
        "final_ci95": final_ci,
    }
    return out, metadata


def _select_counterpart_rows(
    candidates: pd.DataFrame,
    run_counts: pd.DataFrame,
    targets: dict[str, Any],
) -> pd.DataFrame:
    del run_counts
    pool = candidates[
        candidates.band_90.isin(["lower", "inside"])
        & candidates.anchor_speed_drop_kmh_raw.notna()
        & candidates.window_speed_range_kmh_raw.notna()
    ].copy()
    unique_frame_counts = np.asarray(
        [
            pool.loc[pool.scenario_id.eq(scenario), "scenario_frame_index"].nunique()
            for scenario in SCENARIOS
        ],
        dtype=int,
    )
    allocated_cases = _bounded_allocate(
        186,
        unique_frame_counts.astype(float),
        unique_frame_counts,
        np.ones(len(SCENARIOS), dtype=int),
    )
    case_counts = {
        scenario: int(allocated_cases[index])
        for index, scenario in enumerate(SCENARIOS)
    }
    case_offset = 0
    pool["synthetic_case_id"] = ""
    for scenario in SCENARIOS:
        mask = pool.scenario_id.eq(scenario)
        unique_frames = np.sort(pool.loc[mask, "scenario_frame_index"].unique())
        bins = np.array_split(unique_frames, case_counts[scenario])
        for local_index, frame_values in enumerate(bins):
            case_id = f"SC{case_offset + local_index + 1:03d}"
            pool.loc[
                mask & pool.scenario_frame_index.isin(frame_values),
                "synthetic_case_id",
            ] = case_id
        case_offset += case_counts[scenario]
    if (
        case_offset != 186
        or pool.synthetic_case_id.eq("").any()
        or pool.synthetic_case_id.nunique() != 186
    ):
        raise AssertionError("scenario-time synthetic case construction failed")
    cp_target = targets["signature"]["counterpart"]
    lower_pool = pool[
        pool.band_90.eq("lower")
        & pool.anchor_speed_drop_kmh_raw.notna()
        & pool.window_speed_range_kmh_raw.notna()
    ]
    inside_pool = pool[
        pool.band_90.eq("inside")
        & pool.anchor_speed_drop_kmh_raw.notna()
        & pool.window_speed_range_kmh_raw.notna()
    ]
    all_cases = set(pool.synthetic_case_id.unique())
    inside_cases = set(inside_pool.synthetic_case_id.unique())
    mandatory_lower: list[int] = []
    for case_id in sorted(all_cases - inside_cases):
        group = lower_pool[lower_pool.synthetic_case_id.eq(case_id)]
        if group.empty:
            raise AssertionError(f"synthetic case {case_id} has no finite outcome row")
        score = np.abs(group.anchor_speed_drop_kmh_raw - 3.56)
        score += np.abs(group.window_speed_range_kmh_raw - 4.32)
        mandatory_lower.append(int(score.idxmin()))
    remaining_lower_count = cp_target["n_lower"] - len(mandatory_lower)
    remaining_lower_pool = lower_pool.drop(index=mandatory_lower)
    lower_target = _calibrated_quantile_values(
        remaining_lower_count,
        2.0,
        cp_target["speed_drop_median_lower_kmh"],
        6.0,
    )
    lower_indices = mandatory_lower + _select_quantile_rows(
        remaining_lower_pool,
        "anchor_speed_drop_kmh_raw",
        remaining_lower_count,
        lower_target,
    ).tolist()
    lower = pool.loc[lower_indices].copy()

    selected_inside: list[int] = []
    covered = set(lower.synthetic_case_id)
    for case_id, group in inside_pool.groupby("synthetic_case_id"):
        if case_id not in covered:
            score = np.abs(group.anchor_speed_drop_kmh_raw - 1.45)
            score += np.abs(group.window_speed_range_kmh_raw - 2.98)
            selected_inside.append(int(score.idxmin()))
    remaining_count = cp_target["n_inside"] - len(selected_inside)
    remaining_pool = inside_pool.drop(index=selected_inside)
    selected_set = set(selected_inside)
    median_rank_count = (cp_target["n_inside"] + 1) // 2

    def add_best(frame: pd.DataFrame, count: int) -> None:
        if count <= 0:
            return
        available = frame.loc[~frame.index.isin(selected_set)].copy()
        available["distance"] = np.abs(available.anchor_speed_drop_kmh_raw - 1.45)
        available["distance"] += np.abs(available.window_speed_range_kmh_raw - 2.98)
        for index in available.nsmallest(count, "distance").index:
            selected_set.add(int(index))

    current = inside_pool.loc[list(selected_set)]
    need_high_drop = max(
        0,
        median_rank_count - int((current.anchor_speed_drop_kmh_raw >= 1.45).sum()),
    )
    need_low_range = max(
        0,
        median_rank_count - int((current.window_speed_range_kmh_raw <= 2.98).sum()),
    )
    intersection = remaining_pool[
        remaining_pool.anchor_speed_drop_kmh_raw.ge(1.45)
        & remaining_pool.window_speed_range_kmh_raw.le(2.98)
    ]
    add_best(intersection, min(need_high_drop, need_low_range))
    current = inside_pool.loc[list(selected_set)]
    need_high_drop = max(
        0,
        median_rank_count - int((current.anchor_speed_drop_kmh_raw >= 1.45).sum()),
    )
    add_best(remaining_pool[remaining_pool.anchor_speed_drop_kmh_raw.ge(1.45)], need_high_drop)
    current = inside_pool.loc[list(selected_set)]
    need_low_range = max(
        0,
        median_rank_count - int((current.window_speed_range_kmh_raw <= 2.98).sum()),
    )
    add_best(remaining_pool[remaining_pool.window_speed_range_kmh_raw.le(2.98)], need_low_range)
    remaining_count = cp_target["n_inside"] - len(selected_set)
    add_best(remaining_pool, remaining_count)
    if len(selected_set) != cp_target["n_inside"]:
        raise AssertionError("inside counterpart selection did not close")
    selected_inside = sorted(selected_set)
    inside = pool.loc[selected_inside].copy()
    selected = pd.concat([lower, inside], ignore_index=True)
    missing_cases = sorted(set(pool.synthetic_case_id) - set(selected.synthetic_case_id))
    for case_id in missing_cases:
        replacement = pool[pool.synthetic_case_id.eq(case_id)].iloc[[0]].copy()
        band = replacement.band_90.iloc[0]
        case_frequency = selected.synthetic_case_id.value_counts()
        donor = selected[
            selected.band_90.eq(band)
            & selected.synthetic_case_id.map(case_frequency).gt(1)
        ]
        if donor.empty:
            raise AssertionError(f"no same-band donor available for {case_id}")
        selected = selected.drop(index=donor.index[-1])
        selected = pd.concat([selected, replacement], ignore_index=True)
    if selected.synthetic_case_id.nunique() != cp_target["n_cases"]:
        raise AssertionError("counterpart table does not cover 186 synthetic cases")
    return selected


def _calibrate_continuous_ratio(
    frame: pd.DataFrame,
    raw_column: str,
    calibrated_column: str,
    median_lower: float,
    median_inside: float,
    target_ci: list[float],
    seed: int,
) -> tuple[pd.Series, dict[str, Any]]:
    frozen_profiles = {
        "anchor_speed_drop_kmh_raw": {
            "lower": (0.3251582209834475, 1.1584033796660365, 0.99, 1.0),
            "inside": (0.7410672618544948, 1.1074421596097308, 0.0, -1.0),
        },
        "window_speed_range_kmh_raw": {
            "lower": (0.8484784225193933, 1.4074039489996004, 0.8, 1.0),
            "inside": (0.8517973422986462, 1.4965497897690823, 0.7, -1.0),
        },
    }
    if raw_column in frozen_profiles:
        # Frozen after a bounded shape/ICC search. Only the medians are observed
        # hard constraints; the wider synthetic tails are a soft calibration to
        # the rounded 186-case interval printed in the paper.
        profiles = frozen_profiles[raw_column]
        assigned = pd.Series(index=frame.index, dtype=float)
        for band, median, offset in (
            ("lower", median_lower, 17),
            ("inside", median_inside, 29),
        ):
            low_factor, high_factor, rho, sign = profiles[band]
            count = int(frame.band_90.eq(band).sum())
            values = _calibrated_quantile_values(
                count,
                median * low_factor,
                median,
                median * high_factor,
            )
            mask = frame.band_90.eq(band)
            assigned.loc[mask] = _assign_clustered_values(
                frame.loc[mask],
                values,
                raw_column,
                "synthetic_case_id",
                rho,
                seed + offset,
                sign,
            )
        chosen = frame.copy()
        chosen[calibrated_column] = assigned
        final_ci = _bootstrap_ratio_ci(
            chosen,
            calibrated_column,
            "synthetic_case_id",
            2000,
            seed + 101,
        )
        return assigned, {
            "profile": profiles,
            "final_ci95": final_ci,
            "soft_target_ci95": target_ci,
        }

    values = {
        "lower": _calibrated_quantile_values(
            int(frame.band_90.eq("lower").sum()),
            median_lower * 0.60,
            median_lower,
            median_lower * 1.55,
        ),
        "inside": _calibrated_quantile_values(
            int(frame.band_90.eq("inside").sum()),
            median_inside * 0.65,
            median_inside,
            median_inside * 1.45,
        ),
    }
    target = np.asarray(target_ci)
    best: tuple[float, pd.Series, list[float], dict[str, Any]] | None = None
    for rho_lower in (0.0, 0.5, 0.8, 0.95, 0.99):
        for rho_inside in (0.0, 0.5, 0.8, 0.95, 0.99):
            for sign in (-1.0, 1.0):
                assigned = pd.Series(index=frame.index, dtype=float)
                for band, rho, offset, band_sign in (
                    ("lower", rho_lower, 17, 1.0),
                    ("inside", rho_inside, 29, sign),
                ):
                    mask = frame.band_90.eq(band)
                    assigned.loc[mask] = _assign_clustered_values(
                        frame.loc[mask],
                        values[band],
                        raw_column,
                        "synthetic_case_id",
                        rho,
                        seed + offset,
                        band_sign,
                    )
                candidate = frame.copy()
                candidate[calibrated_column] = assigned
                ci = _bootstrap_ratio_ci(
                    candidate,
                    calibrated_column,
                    "synthetic_case_id",
                    300,
                    seed + 101,
                )
                score = float(np.sum((np.asarray(ci) - target) ** 2))
                metadata = {
                    "rho_lower": rho_lower,
                    "rho_inside": rho_inside,
                    "inside_cluster_sign": sign,
                }
                if best is None or score < best[0]:
                    best = (score, assigned.copy(), ci, metadata)
    if best is None:
        raise AssertionError("continuous calibration grid was empty")
    chosen = frame.copy()
    chosen[calibrated_column] = best[1]
    final_ci = _bootstrap_ratio_ci(
        chosen,
        calibrated_column,
        "synthetic_case_id",
        2000,
        seed + 101,
    )
    return best[1], {**best[3], "search_ci95": best[2], "final_ci95": final_ci}


def _calibrate_braking(
    frame: pd.DataFrame,
    targets: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    cp_target = targets["signature"]["counterpart"]
    totals: dict[str, np.ndarray] = {}
    for band, denominator in (
        ("lower", cp_target["brake_lower_den"]),
        ("inside", cp_target["brake_inside_den"]),
    ):
        count = int(out.band_90.eq(band).sum())
        totals[band] = _bounded_allocate(
            denominator,
            np.ones(count),
            np.full(count, 31),
            np.ones(count, dtype=int),
        )
        out.loc[out.band_90.eq(band), "n_frames_total_calibrated"] = totals[band]

    target_ci = np.asarray(cp_target["brake_diff_ci95_target"])
    best: tuple[float, pd.DataFrame, list[float], dict[str, Any]] | None = None
    for rho_lower in (0.0, 0.5, 0.8, 0.95, 0.99):
        for rho_inside in (0.0, 0.5, 0.8, 0.95, 0.99):
            for sign in (-1.0, 1.0):
                candidate = out.copy()
                for band, rho, numerator, seed, band_sign in (
                    ("lower", rho_lower, cp_target["brake_lower_num"], 811, 1.0),
                    ("inside", rho_inside, cp_target["brake_inside_num"], 919, sign),
                ):
                    mask = candidate.band_90.eq(band)
                    subset = candidate.loc[mask]
                    clusters = sorted(subset.synthetic_case_id.unique())
                    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
                    rng = np.random.default_rng(seed)
                    effect = rng.normal(size=len(clusters)) * band_sign
                    row_effect = np.asarray(
                        [effect[cluster_index[value]] for value in subset.synthetic_case_id]
                    )
                    raw_share = (
                        subset.n_frames_lt_m3_raw.to_numpy(float)
                        / np.maximum(subset.n_frames_total_raw.to_numpy(float), 1.0)
                    )
                    raw_rank = pd.Series(raw_share).rank(pct=True).fillna(0.5).to_numpy()
                    weights = np.exp(
                        np.clip(
                            rho * row_effect
                            + math.sqrt(max(0.0, 1 - rho**2)) * (raw_rank - 0.5),
                            -5.0,
                            5.0,
                        )
                    )
                    allocated = _bounded_allocate(
                        numerator,
                        weights,
                        totals[band],
                    )
                    candidate.loc[mask, "n_frames_lt_m3_calibrated"] = allocated
                ci = _bootstrap_brake_difference(
                    candidate, "synthetic_case_id", 300, 1201
                )
                score = float(np.sum((np.asarray(ci) - target_ci) ** 2))
                metadata = {
                    "rho_lower": rho_lower,
                    "rho_inside": rho_inside,
                    "inside_cluster_sign": sign,
                }
                if best is None or score < best[0]:
                    best = (score, candidate.copy(), ci, metadata)
    if best is None:
        raise AssertionError("brake calibration grid was empty")
    chosen = best[1]
    chosen["n_frames_total_calibrated"] = chosen.n_frames_total_calibrated.astype(int)
    chosen["n_frames_lt_m3_calibrated"] = chosen.n_frames_lt_m3_calibrated.astype(int)
    final_ci = _bootstrap_brake_difference(chosen, "synthetic_case_id", 1000, 1201)
    return chosen, {**best[3], "search_ci95": best[2], "final_ci95": final_ci}


def _bootstrap_brake_difference(
    frame: pd.DataFrame,
    cluster_column: str,
    draws: int,
    seed: int,
) -> list[float]:
    grouped = (
        frame.assign(
            lower_num=np.where(
                frame.band_90.eq("lower"), frame.n_frames_lt_m3_calibrated, 0
            ),
            lower_den=np.where(
                frame.band_90.eq("lower"), frame.n_frames_total_calibrated, 0
            ),
            inside_num=np.where(
                frame.band_90.eq("inside"), frame.n_frames_lt_m3_calibrated, 0
            ),
            inside_den=np.where(
                frame.band_90.eq("inside"), frame.n_frames_total_calibrated, 0
            ),
        )
        .groupby(cluster_column)[
            ["lower_num", "lower_den", "inside_num", "inside_den"]
        ]
        .sum()
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = grouped[rng.integers(0, len(grouped), len(grouped))].sum(axis=0)
        estimates.append(float(sampled[0] / sampled[1] - sampled[2] / sampled[3]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _build_counterpart_table(
    candidates: pd.DataFrame,
    run_counts: pd.DataFrame,
    targets: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counterpart = _select_counterpart_rows(candidates, run_counts, targets)
    cp_target = targets["signature"]["counterpart"]
    drop_values, drop_meta = _calibrate_continuous_ratio(
        counterpart,
        "anchor_speed_drop_kmh_raw",
        "anchor_speed_drop_kmh_calibrated",
        cp_target["speed_drop_median_lower_kmh"],
        cp_target["speed_drop_median_inside_kmh"],
        cp_target["speed_drop_ratio_ci95_target"],
        1701,
    )
    counterpart["anchor_speed_drop_kmh_calibrated"] = drop_values
    range_values, range_meta = _calibrate_continuous_ratio(
        counterpart,
        "window_speed_range_kmh_raw",
        "window_speed_range_kmh_calibrated",
        cp_target["speed_range_median_lower_kmh"],
        cp_target["speed_range_median_inside_kmh"],
        cp_target["speed_range_ratio_ci95_target"],
        1901,
    )
    counterpart["window_speed_range_kmh_calibrated"] = range_values
    counterpart, brake_meta = _calibrate_braking(counterpart, targets)
    keep = [
        "candidate_key",
        "driver_id",
        "scenario_id",
        "run_id",
        "scenario_frame_index",
        "band_90",
        "synthetic_case_id",
        "designated_counterpart_id",
        "counterpart_observed",
        "counterpart_interpolated",
        "counterpart_speed_kmh",
        "counterpart_acceleration_mps2",
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_total_raw",
        "n_frames_lt_m3_raw",
        "anchor_speed_drop_kmh_calibrated",
        "window_speed_range_kmh_calibrated",
        "n_frames_total_calibrated",
        "n_frames_lt_m3_calibrated",
    ]
    metadata = {"speed_drop": drop_meta, "speed_range": range_meta, "braking": brake_meta}
    return counterpart[keep].copy(), metadata


def _per_unit_counts(candidates: pd.DataFrame) -> pd.DataFrame:
    both = candidates.status.eq("OK") & candidates.mechanism2_gate_ok
    unit = candidates.assign(
        gate1=candidates.status.eq("OK"),
        both=both,
        below80=candidates.band_80.eq("lower"),
        above80=candidates.band_80.eq("upper"),
        below90=candidates.band_90.eq("lower"),
        above90=candidates.band_90.eq("upper"),
        below95=candidates.band_95.eq("lower"),
        above95=candidates.band_95.eq("upper"),
    ).groupby(["driver_id", "scenario_id", "run_id"], as_index=False).agg(
        n_candidate=("candidate_key", "size"),
        n_gate1=("gate1", "sum"),
        n_both=("both", "sum"),
        n_below_80=("below80", "sum"),
        n_above_80=("above80", "sum"),
        n_below_90=("below90", "sum"),
        n_above_90=("above90", "sum"),
        n_below_95=("below95", "sum"),
        n_above_95=("above95", "sum"),
    )
    for column in unit.columns[3:]:
        unit[column] = unit[column].astype(int)
    return unit


def _paper_metrics(
    human_targets: dict[str, Any],
    av_reference: dict[str, Any],
) -> dict[str, Any]:
    scenarios = sorted(human_targets["per_scenario_alpha90"])
    human = human_targets["per_scenario_alpha90"]
    automated = av_reference["per_scenario_alpha90"]

    def human_rate(scenario: str) -> float:
        row = human[scenario]
        return float(row["n_flagged"] / row["n_both"])

    rng = np.random.default_rng(20260820)
    ratios: list[float] = []
    scenario_weighted: list[float] = []
    for _ in range(20_000):
        sampled = rng.choice(scenarios, len(scenarios), replace=True)
        human_num = sum(human[scenario]["n_flagged"] for scenario in sampled)
        human_den = sum(human[scenario]["n_both"] for scenario in sampled)
        av_num = sum(automated[scenario]["n_flagged"] for scenario in sampled)
        av_den = sum(automated[scenario]["n_both"] for scenario in sampled)
        ratios.append((av_num / av_den) / (human_num / human_den))
        scenario_weighted.append(
            np.mean([automated[scenario]["rate"] for scenario in sampled])
            / np.mean([human_rate(scenario) for scenario in sampled])
        )
    human_rates = [human_rate(scenario) for scenario in scenarios]
    av_rates = [automated[scenario]["rate"] for scenario in scenarios]
    return {
        "moment_weighted_ratio": (
            av_reference["flag_counts"]["90"]["n_below"]
            + av_reference["flag_counts"]["90"]["n_above"]
        )
        / av_reference["flag_counts"]["90"]["n_total"]
        / (
            (human_targets["flag_counts"]["90"]["n_below"]
             + human_targets["flag_counts"]["90"]["n_above"])
            / human_targets["flag_counts"]["90"]["n_total"]
        ),
        "scenario_bootstrap_ci95": [
            float(value) for value in np.quantile(ratios, [0.025, 0.975])
        ],
        "scenario_weighted_ratio": float(np.mean(av_rates) / np.mean(human_rates)),
        "scenario_weighted_ci95": [
            float(value) for value in np.quantile(scenario_weighted, [0.025, 0.975])
        ],
        "draws_above_parity": int((np.asarray(ratios) > 1.0).sum()),
        "human_scenario_rate_range": [float(min(human_rates)), float(max(human_rates))],
        "automated_scenario_rate_range": [float(min(av_rates)), float(max(av_rates))],
        "scenarios_av_higher": int(
            sum(automated[scenario]["rate"] > human_rate(scenario) for scenario in scenarios)
        ),
    }


def _ci_loss(
    intervals: dict[str, list[float]],
    targets: dict[str, Any],
) -> float:
    target_map = {
        "alpha90": targets["flag_rate_ci95_alpha90_target"],
        "ego_ttc": targets["signature"]["ego_ttc"]["lt2_diff_ci95_target"],
        "speed_drop": targets["signature"]["counterpart"]["speed_drop_ratio_ci95_target"],
        "speed_range": targets["signature"]["counterpart"]["speed_range_ratio_ci95_target"],
        "braking": targets["signature"]["counterpart"]["brake_diff_ci95_target"],
    }
    scales = {
        "alpha90": 0.002,
        "ego_ttc": 0.005,
        "speed_drop": 0.05,
        "speed_range": 0.05,
        "braking": 0.005,
    }
    return float(
        sum(
            np.sum(
                ((np.asarray(intervals[key]) - np.asarray(target_map[key])) / scales[key])
                ** 2
            )
            for key in target_map
        )
    )


def _raw_summary(
    candidates: pd.DataFrame,
    counterpart: pd.DataFrame,
) -> dict[str, Any]:
    ttc: dict[str, Any] = {}
    for band in ("lower", "inside"):
        values = candidates.loc[
            candidates.band_90.eq(band) & candidates.future_min_ttc_s_calibrated.notna(),
            "future_min_ttc_raw_s",
        ].dropna().to_numpy(float)
        ttc[band] = {
            "n": int(len(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "lt2_share": float((values < 2.0).mean()),
        }
    cp: dict[str, Any] = {}
    for band in ("lower", "inside"):
        group = counterpart[counterpart.band_90.eq(band)]
        cp[band] = {
            "n": int(len(group)),
            "speed_drop_median_kmh": float(group.anchor_speed_drop_kmh_raw.median()),
            "speed_range_median_kmh": float(group.window_speed_range_kmh_raw.median()),
            "brake_num": int(group.n_frames_lt_m3_raw.sum()),
            "brake_den": int(group.n_frames_total_raw.sum()),
            "brake_share": float(
                group.n_frames_lt_m3_raw.sum() / max(group.n_frames_total_raw.sum(), 1)
            ),
        }
    return {"ego_ttc": ttc, "counterpart": cp}


def _raw_closeness_loss(summary: dict[str, Any], targets: dict[str, Any]) -> float:
    loss = 0.0
    for band in ("lower", "inside"):
        observed = summary["ego_ttc"][band]
        target = targets["signature"]["ego_ttc"][band]
        for key in ("q25", "q50", "q75"):
            loss += float(np.log(max(observed[key], 1e-6) / target[key]) ** 2)
        target_share = target["lt2_num"] / target["n"]
        loss += float(((observed["lt2_share"] - target_share) / 0.02) ** 2)
    cp_target = targets["signature"]["counterpart"]
    mapping = {
        "lower": (
            cp_target["speed_drop_median_lower_kmh"],
            cp_target["speed_range_median_lower_kmh"],
            cp_target["brake_lower_num"] / cp_target["brake_lower_den"],
        ),
        "inside": (
            cp_target["speed_drop_median_inside_kmh"],
            cp_target["speed_range_median_inside_kmh"],
            cp_target["brake_inside_num"] / cp_target["brake_inside_den"],
        ),
    }
    for band, (drop_target, range_target, brake_target) in mapping.items():
        observed = summary["counterpart"][band]
        loss += float(
            np.log(max(observed["speed_drop_median_kmh"], 1e-6) / drop_target) ** 2
        )
        loss += float(
            np.log(max(observed["speed_range_median_kmh"], 1e-6) / range_target) ** 2
        )
        loss += float(((observed["brake_share"] - brake_target) / 0.02) ** 2)
    return loss


def calibrate_dataset(
    base_output: Path = BASE_OUTPUT,
    output: Path = DEFAULT_OUTPUT,
    *,
    source_session: Path = DEFAULT_SOURCE_SESSION,
    base_targets_path: Path = BASE_TARGETS,
    paper_targets_path: Path = PAPER_TARGETS,
    av_reference_path: Path = AV_REFERENCE,
    replace: bool = False,
) -> dict[str, Any]:
    base_targets = json.loads(base_targets_path.read_text(encoding="utf-8"))
    paper_targets = json.loads(paper_targets_path.read_text(encoding="utf-8"))
    av_reference = json.loads(av_reference_path.read_text(encoding="utf-8"))
    base_manifest = json.loads((base_output / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("data_status") != DATA_STATUS:
        raise ValueError("base package is not explicitly synthetic")
    _prepare_output(base_output, output, replace)
    tables = output / "tables"
    trajectories = pd.read_parquet(tables / "trajectory_pairs.parquet")
    base_candidates = pd.read_parquet(tables / "candidate_moments.parquet")
    base_achieved = json.loads((tables / "achieved_summary.json").read_text(encoding="utf-8"))

    background = _load_designated_counterparts(source_session)
    enhanced_trajectories = _enhance_trajectories(trajectories, background)
    candidates = _join_candidate_raw(base_candidates, enhanced_trajectories)
    base_counterpart_keys = pd.read_parquet(
        base_output / "tables/counterpart_windows.parquet",
        columns=["candidate_key", "band_90"],
    )
    raw_candidate_columns = [
        "candidate_key",
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_total_raw",
        "n_frames_lt_m3_raw",
    ]
    base_counterpart_designated = base_counterpart_keys.merge(
        candidates[raw_candidate_columns],
        on="candidate_key",
        how="left",
        validate="one_to_one",
    )
    base_raw_summary = _raw_summary(candidates, base_counterpart_designated)
    run_counts, run_meta = _calibrate_run_counts(candidates, base_targets)
    candidates = _apply_run_counts(candidates, run_counts, base_targets)
    candidates, ttc_meta = _calibrate_ttc(candidates, base_targets)
    counterpart, counterpart_meta = _build_counterpart_table(
        candidates, run_counts, base_targets
    )
    unit = _per_unit_counts(candidates)

    unit_for_bootstrap = unit.sort_values(["scenario_id", "driver_id"])
    intervals = {
        "alpha90": _bootstrap_rate_ci(
            unit_for_bootstrap.n_both.to_numpy(int),
            (
                unit_for_bootstrap.n_below_90 + unit_for_bootstrap.n_above_90
            ).to_numpy(int),
        ),
        "ego_ttc": ttc_meta["final_ci95"],
        "speed_drop": counterpart_meta["speed_drop"]["final_ci95"],
        "speed_range": counterpart_meta["speed_range"]["final_ci95"],
        "braking": counterpart_meta["braking"]["final_ci95"],
    }
    v1_intervals = {
        "alpha90": base_achieved["flag_rate_ci95_alpha90_synthetic_bootstrap"],
        "ego_ttc": base_achieved["signature"]["ego_ttc"][
            "lt2_difference_ci95_synthetic_bootstrap"
        ],
        "speed_drop": base_achieved["signature"]["counterpart"][
            "speed_drop_ratio_ci95_synthetic_bootstrap"
        ],
        "speed_range": base_achieved["signature"]["counterpart"][
            "speed_range_ratio_ci95_synthetic_bootstrap"
        ],
        "braking": base_achieved["signature"]["counterpart"][
            "brake_difference_ci95_synthetic_bootstrap"
        ],
    }
    paper_metrics = _paper_metrics(base_targets, av_reference)
    raw_summary = _raw_summary(candidates, counterpart)
    raw_loss_v1 = _raw_closeness_loss(base_raw_summary, base_targets)
    raw_loss_v2 = _raw_closeness_loss(raw_summary, base_targets)
    v1_loss = _ci_loss(v1_intervals, base_targets)
    v2_loss = _ci_loss(intervals, base_targets)

    enhanced_trajectories.to_parquet(tables / "trajectory_pairs.parquet", index=False)
    candidates.to_parquet(tables / "candidate_moments.parquet", index=False)
    candidates[
        candidates.status.eq("OK") & candidates.mechanism2_gate_ok
    ].to_parquet(tables / "both_gate_moments.parquet", index=False)
    counterpart.to_parquet(tables / "counterpart_windows.parquet", index=False)
    unit.to_csv(tables / "per_unit_counts.csv", index=False)
    run_counts.to_csv(tables / "synthetic_run_calibration.csv", index=False)
    background.to_parquet(tables / "designated_counterpart_template.parquet", index=False)

    summary = {
        "data_status": DATA_STATUS,
        "calibration_status": "PAPER_ALIGNED_SYNTHETIC_V2",
        "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
        "paper_claim_metric_layer": "aggregate-constrained calibrated fields",
        "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
        "full_distribution_match_claimed": False,
        "estimator_recomputed": False,
        "paper_hard_metrics": paper_metrics,
        "synthetic_intervals": intervals,
        "target_intervals": {
            "alpha90": base_targets["flag_rate_ci95_alpha90_target"],
            "ego_ttc": base_targets["signature"]["ego_ttc"]["lt2_diff_ci95_target"],
            "speed_drop": base_targets["signature"]["counterpart"][
                "speed_drop_ratio_ci95_target"
            ],
            "speed_range": base_targets["signature"]["counterpart"][
                "speed_range_ratio_ci95_target"
            ],
            "braking": base_targets["signature"]["counterpart"]["brake_diff_ci95_target"],
        },
        "ci_loss_v1": v1_loss,
        "ci_loss_v2": v2_loss,
        "ci_loss_improvement_fraction": float((v1_loss - v2_loss) / v1_loss),
        "raw_summary_v1_designated_counterpart": base_raw_summary,
        "raw_summary": raw_summary,
        "raw_closeness_loss_v1": raw_loss_v1,
        "raw_closeness_loss_v2": raw_loss_v2,
        "raw_closeness_improvement_fraction": float(
            (raw_loss_v1 - raw_loss_v2) / raw_loss_v1
        ),
        "calibration_parameters": {
            "run_counts": run_meta,
            "ego_ttc": ttc_meta,
            "counterpart": counterpart_meta,
        },
        "paper_target_contract": str(paper_targets_path.relative_to(REPO_ROOT)),
        "paper_target_snapshot": paper_targets,
        "boundary": (
            "V2 optimizes synthetic dependence toward rounded paper records. It does not "
            "recover the unavailable human run/case mapping or make calibrated columns observed."
        ),
    }
    (tables / "achieved_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        **base_manifest,
        "schema_version": SCHEMA_VERSION,
        "aggregate_constrained_fields": [
            "status",
            "mechanism2_gate_ok",
            "ipv_log",
            "band_80",
            "band_90",
            "band_95",
            "future_min_ttc_s_calibrated",
            "anchor_speed_drop_kmh_calibrated",
            "window_speed_range_kmh_calibrated",
            "n_frames_total_calibrated",
            "n_frames_lt_m3_calibrated",
            "synthetic_case_id",
        ],
        "bootstrap_cluster": (
            "run_id (driver_id x scenario_id) for alpha90 and ego TTC; "
            "synthetic_case_id for counterpart signatures"
        ),
        "base_version_path": str(base_output.relative_to(REPO_ROOT)),
        "paper_alignment": (
            "soft-CI calibrated; hard counts and point records exact; rounded display "
            "records checked at their printed precision"
        ),
        "counterpart_selection": "fixed designated counterpart per Shanghai scenario",
        "counterpart_acceleration": "finite difference of recorded speed",
        "signature_window": "forward 31 frames including anchor",
        "synthetic_case_count": int(counterpart.synthetic_case_id.nunique()),
        "synthetic_case_note": (
            "Cases are scenario-time background blocks shared across synthetic drivers. IDs are "
            "calibrated structure, not the unavailable observed 186-case mapping."
        ),
        "human_ego_margin_intervals_generated": False,
        "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
        "paper_claim_metric_layer": (
            "Band/status counts and columns ending in _calibrated; raw columns remain "
            "trajectory-derived diagnostics"
        ),
        "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
        "full_distribution_match_claimed": False,
        "warning": (
            "SYNTHETIC_NOT_OBSERVED. V2 is closer to rounded manuscript records by design; "
            "it is not observed participant data or independent RQ022 evidence."
        ),
    }
    manifest.pop("pseudo_case_id_note", None)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# RQ029 v2 paper-aligned synthetic microdata\n\n"
        "**SYNTHETIC_NOT_OBSERVED.** This version preserves v1 raw replay/ego data, uses "
        "the fixed Shanghai scenario counterpart for raw three-second outcomes, and tunes "
        "synthetic run/case dependence toward the rounded records currently printed in the "
        "manuscript. Exact paper intervals remain non-identifiable without the archived real "
        "run/case tables. Paper claim point estimates are exact only on the explicitly calibrated "
        "fields; raw trajectory-derived outcomes and the full joint distribution are not matched "
        "or claimed.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-session", type=Path, default=DEFAULT_SOURCE_SESSION)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = calibrate_dataset(
        args.base_output,
        args.output_dir,
        source_session=args.source_session,
        replace=args.replace,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
