#!/usr/bin/env python3
"""Validate the paper-aligned RQ029 v2 synthetic overlay."""

from __future__ import annotations

import argparse
import copy
import filecmp
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.simulation.calibrate_rq029_paper_v2 import (  # noqa: E402
    AV_REFERENCE,
    BASE_OUTPUT,
    BASE_TARGETS,
    DEFAULT_OUTPUT,
    DESIGNATED_COUNTERPARTS,
    PAPER_TARGETS,
    SCHEMA_VERSION,
    _bootstrap_brake_difference,
    _bootstrap_rate_ci,
    _bootstrap_ratio_ci,
    _bootstrap_share_difference,
    _ci_loss,
    _paper_metrics,
    _raw_closeness_loss,
    _raw_summary,
)
from pipelines.simulation.generate_rq029_human_synthetic import (  # noqa: E402
    CASE_TO_SCENARIO_SHANGHAI,
    DATA_STATUS,
    DEFAULT_SOURCE_SESSION,
    SCENARIOS,
)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    severity: str = "critical",
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "severity": severity,
        }
    )


def _source_number(actor: dict[str, Any], names: tuple[str, ...]) -> float | None:
    """Read one numeric actor field without reusing the calibration parser."""
    for name in names:
        value = actor.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _recompute_source_counterpart_template(source_session: Path) -> pd.DataFrame:
    """Independently rebuild designated-counterpart rows from the Shanghai source log."""
    source = source_session / "vehicle_perception_simulation_trajectory.log"
    counters = {scenario: 0 for scenario in SCENARIOS}
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            scenario = CASE_TO_SCENARIO_SHANGHAI[int(record["caseId"])]
            ego_actors = [
                actor
                for group in record["participantTrajectories"]
                for actor in group.get("value", [])
                if actor.get("isPerception") == 0
            ]
            if len(ego_actors) != 1:
                raise ValueError(
                    f"source frame has {len(ego_actors)} ego actors in {scenario}"
                )
            background_actors = [
                actor
                for group in record["participantTrajectories"]
                if group.get("role") == "mvSimulation"
                for actor in group.get("value", [])
            ]
            counterpart_id = DESIGNATED_COUNTERPARTS[scenario]
            matches = [
                actor
                for actor in background_actors
                if str(actor.get("id", "")) == counterpart_id
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"source frame has duplicate counterpart {counterpart_id} in {scenario}"
                )
            counterpart = matches[0] if matches else None
            row: dict[str, Any] = {
                "scenario_id": scenario,
                "scenario_frame_index": counters[scenario],
                "background_timestamp_ms": int(ego_actors[0]["globalTimeStamp"]),
                "designated_counterpart_id": counterpart_id,
                "counterpart_latitude": np.nan,
                "counterpart_longitude": np.nan,
                "counterpart_speed_kmh": np.nan,
                "counterpart_course_deg": np.nan,
            }
            if counterpart is not None:
                row.update(
                    {
                        "counterpart_latitude": _source_number(
                            counterpart, ("latitude", "lat")
                        ),
                        "counterpart_longitude": _source_number(
                            counterpart, ("longitude", "lng", "lon")
                        ),
                        "counterpart_speed_kmh": _source_number(
                            counterpart, ("speed",)
                        ),
                        "counterpart_course_deg": _source_number(
                            counterpart, ("courseAngle",)
                        ),
                    }
                )
            rows.append(row)
            counters[scenario] += 1

    parts: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        group = pd.DataFrame(row for row in rows if row["scenario_id"] == scenario)
        group = group.sort_values("scenario_frame_index").reset_index(drop=True)
        group["counterpart_observed"] = group["counterpart_speed_kmh"].notna()
        interpolated_columns = [
            "counterpart_latitude",
            "counterpart_longitude",
            "counterpart_speed_kmh",
            "counterpart_course_deg",
        ]
        group[interpolated_columns] = group[interpolated_columns].interpolate(
            method="linear", limit=30, limit_direction="both"
        )
        group["counterpart_interpolated"] = (
            ~group["counterpart_observed"] & group["counterpart_speed_kmh"].notna()
        )

        speed = group["counterpart_speed_kmh"].to_numpy(float)
        time_s = group["background_timestamp_ms"].to_numpy(float) / 1000.0
        acceleration = np.full(len(group), np.nan)
        delta_t = np.diff(time_s)
        valid_pair = (
            np.isfinite(speed[1:])
            & np.isfinite(speed[:-1])
            & (delta_t >= 0.05)
            & (delta_t <= 0.30)
        )
        acceleration[1:][valid_pair] = (
            (speed[1:][valid_pair] - speed[:-1][valid_pair])
            / 3.6
            / delta_t[valid_pair]
        )

        speed_drop = np.full(len(group), np.nan)
        speed_range = np.full(len(group), np.nan)
        brake_num = np.zeros(len(group), dtype=int)
        brake_den = np.zeros(len(group), dtype=int)
        for index in range(len(group)):
            window_speed = speed[index : index + 31]
            window_acceleration = acceleration[index : index + 31]
            finite_speed = window_speed[np.isfinite(window_speed)]
            finite_acceleration = window_acceleration[np.isfinite(window_acceleration)]
            if np.isfinite(speed[index]) and finite_speed.size:
                speed_drop[index] = max(0.0, speed[index] - float(finite_speed.min()))
                speed_range[index] = float(finite_speed.max() - finite_speed.min())
            brake_num[index] = int((finite_acceleration < -3.0).sum())
            brake_den[index] = int(finite_acceleration.size)
        group["counterpart_acceleration_mps2"] = acceleration
        group["anchor_speed_drop_kmh_raw"] = speed_drop
        group["window_speed_range_kmh_raw"] = speed_range
        group["n_frames_lt_m3_raw"] = brake_num
        group["n_frames_total_raw"] = brake_den
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _frame_match(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
) -> tuple[bool, str]:
    """Compare exact labels/booleans and numerically tight floating-point columns."""
    try:
        pd.testing.assert_frame_equal(
            actual[columns].reset_index(drop=True),
            expected[columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as error:
        return False, str(error).splitlines()[0][:500]
    return True, f"{len(actual)} rows x {len(columns)} columns"


def _expected_trajectory_outcomes(
    base_trajectory: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute source-counterpart and TTC columns from v1 ego motion plus source rows."""
    ego_columns = [
        "driver_id",
        "scenario_id",
        "run_id",
        "scenario_frame_index",
        "ego_latitude",
        "ego_longitude",
        "ego_vx_mps",
        "ego_vy_mps",
    ]
    expected = base_trajectory[ego_columns].merge(
        template,
        on=["scenario_id", "scenario_frame_index"],
        how="left",
        validate="many_to_one",
    )
    latitude = expected["ego_latitude"].to_numpy(float)
    longitude = expected["ego_longitude"].to_numpy(float)
    cp_latitude = expected["counterpart_latitude"].to_numpy(float)
    cp_longitude = expected["counterpart_longitude"].to_numpy(float)
    dx = (cp_longitude - longitude) * 111_320.0 * np.cos(np.radians(latitude))
    dy = (cp_latitude - latitude) * 111_320.0
    distance = np.hypot(dx, dy)
    cp_speed = expected["counterpart_speed_kmh"].to_numpy(float) / 3.6
    cp_angle = np.radians(expected["counterpart_course_deg"].to_numpy(float))
    cp_vx = cp_speed * np.sin(cp_angle)
    cp_vy = cp_speed * np.cos(cp_angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = -(
            (cp_vx - expected["ego_vx_mps"].to_numpy(float)) * dx
            + (cp_vy - expected["ego_vy_mps"].to_numpy(float)) * dy
        ) / distance
    closing[~np.isfinite(closing) | (distance <= 1e-9)] = np.nan
    current_ttc = np.full(len(expected), np.nan)
    np.divide(distance, closing, out=current_ttc, where=closing > 0)
    expected["counterpart_id"] = expected["designated_counterpart_id"]
    expected["counterpart_distance_m"] = distance
    expected["closing_rate_mps"] = closing
    expected["current_ttc_raw_s"] = current_ttc
    expected["future_min_ttc_raw_s"] = np.nan
    for _, group in expected.groupby("run_id"):
        indices = group.sort_values("scenario_frame_index").index
        values = expected.loc[indices, "current_ttc_raw_s"].to_numpy(float)
        future = [
            float(np.nanmin(values[index : index + 31]))
            if np.isfinite(values[index : index + 31]).any()
            else np.nan
            for index in range(len(values))
        ]
        expected.loc[indices, "future_min_ttc_raw_s"] = future
    return expected.sort_values(
        ["driver_id", "scenario_id", "scenario_frame_index"]
    ).reset_index(drop=True)


def _nested_allclose(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _nested_allclose(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _nested_allclose(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12, equal_nan=True))
    return actual == expected


def _intervals_from_detail(
    candidates: pd.DataFrame,
    counterpart: pd.DataFrame,
    unit: pd.DataFrame,
) -> dict[str, list[float]]:
    ordered_unit = unit.sort_values(["scenario_id", "driver_id"])
    return {
        "alpha90": _bootstrap_rate_ci(
            ordered_unit.n_both.to_numpy(int),
            (ordered_unit.n_below_90 + ordered_unit.n_above_90).to_numpy(int),
        ),
        "ego_ttc": _bootstrap_share_difference(
            candidates[
                candidates.band_90.isin(["lower", "inside"])
                & candidates.future_min_ttc_s_calibrated.notna()
            ],
            "future_min_ttc_s_calibrated",
            "run_id",
            2.0,
            1000,
            2701,
        ),
        "speed_drop": _bootstrap_ratio_ci(
            counterpart,
            "anchor_speed_drop_kmh_calibrated",
            "synthetic_case_id",
            2000,
            1802,
        ),
        "speed_range": _bootstrap_ratio_ci(
            counterpart,
            "window_speed_range_kmh_calibrated",
            "synthetic_case_id",
            2000,
            2002,
        ),
        "braking": _bootstrap_brake_difference(
            counterpart,
            "synthetic_case_id",
            1000,
            1201,
        ),
    }


def _validate_raw_inheritance(
    checks: list[dict[str, Any]],
    base: Path,
    output: Path,
) -> None:
    base_runs = pd.read_csv(base / "tables/runs.csv")
    output_runs = pd.read_csv(output / "tables/runs.csv")
    _check(
        checks,
        "run_manifest_unchanged",
        base_runs.equals(output_runs),
        bool(base_runs.equals(output_runs)),
        True,
    )
    raw_files = sorted(
        path.relative_to(base)
        for path in (base / "raw").rglob("*")
        if path.is_file()
    )
    output_raw_files = sorted(
        path.relative_to(output)
        for path in (output / "raw").rglob("*")
        if path.is_file()
    )
    _check(
        checks,
        "raw_file_set_unchanged",
        raw_files == output_raw_files,
        len(output_raw_files),
        len(raw_files),
    )
    mismatches = [
        str(relative)
        for relative in raw_files
        if not filecmp.cmp(base / relative, output / relative, shallow=False)
    ]
    _check(checks, "raw_bytes_unchanged", not mismatches, mismatches, [])
    base_trajectory = pd.read_parquet(base / "tables/trajectory_pairs.parquet")
    output_trajectory = pd.read_parquet(output / "tables/trajectory_pairs.parquet")
    identity_columns = [
        "driver_id",
        "scenario_id",
        "run_id",
        "scenario_frame_index",
        "timestamp_ms",
        "ego_latitude",
        "ego_longitude",
        "ego_speed_kmh",
        "ego_course_deg",
        "source_ego_latitude",
        "source_ego_longitude",
        "source_ego_speed_kmh",
    ]
    trajectory_equal = base_trajectory[identity_columns].equals(
        output_trajectory[identity_columns]
    )
    _check(
        checks,
        "ego_trajectory_unchanged",
        trajectory_equal,
        bool(trajectory_equal),
        True,
    )


def _validate_source_counterpart_derivations(
    checks: list[dict[str, Any]],
    base: Path,
    output: Path,
    candidates: pd.DataFrame,
    counterpart: pd.DataFrame,
    source_session: Path,
) -> dict[str, Any]:
    """Lock v2 source-counterpart fields to a fresh source-log recomputation."""
    expected_template = _recompute_source_counterpart_template(source_session)
    actual_template = pd.read_parquet(
        output / "tables/designated_counterpart_template.parquet"
    )
    template_columns = list(expected_template.columns)
    actual_template = actual_template.sort_values(
        ["scenario_id", "scenario_frame_index"]
    ).reset_index(drop=True)
    expected_template = expected_template.sort_values(
        ["scenario_id", "scenario_frame_index"]
    ).reset_index(drop=True)
    template_match, template_detail = _frame_match(
        actual_template, expected_template, template_columns
    )
    _check(
        checks,
        "source_counterpart_template_recomputed",
        template_match,
        template_detail,
        f"{len(expected_template)} source-derived rows",
    )

    base_trajectory = pd.read_parquet(base / "tables/trajectory_pairs.parquet")
    actual_trajectory = pd.read_parquet(output / "tables/trajectory_pairs.parquet")
    expected_trajectory = _expected_trajectory_outcomes(
        base_trajectory, expected_template
    )
    trajectory_columns = [
        "driver_id",
        "scenario_id",
        "run_id",
        "scenario_frame_index",
        "background_timestamp_ms",
        "designated_counterpart_id",
        "counterpart_latitude",
        "counterpart_longitude",
        "counterpart_speed_kmh",
        "counterpart_course_deg",
        "counterpart_observed",
        "counterpart_interpolated",
        "counterpart_acceleration_mps2",
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_lt_m3_raw",
        "n_frames_total_raw",
        "counterpart_id",
        "counterpart_distance_m",
        "closing_rate_mps",
        "current_ttc_raw_s",
        "future_min_ttc_raw_s",
    ]
    actual_trajectory = actual_trajectory.sort_values(
        ["driver_id", "scenario_id", "scenario_frame_index"]
    ).reset_index(drop=True)
    trajectory_match, trajectory_detail = _frame_match(
        actual_trajectory, expected_trajectory, trajectory_columns
    )
    _check(
        checks,
        "trajectory_counterpart_fields_recomputed",
        trajectory_match,
        trajectory_detail,
        f"{len(expected_trajectory)} source-and-ego-derived rows",
    )

    candidate_raw_columns = [
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
    join_keys = ["driver_id", "scenario_id", "scenario_frame_index"]
    expected_candidate_raw = candidates[
        ["candidate_key", *join_keys]
    ].merge(
        expected_trajectory[[*join_keys, *candidate_raw_columns]],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    actual_candidates = candidates.sort_values("candidate_key").reset_index(drop=True)
    expected_candidate_raw = expected_candidate_raw.sort_values(
        "candidate_key"
    ).reset_index(drop=True)
    candidate_columns = ["candidate_key", *candidate_raw_columns]
    candidate_match, candidate_detail = _frame_match(
        actual_candidates, expected_candidate_raw, candidate_columns
    )
    _check(
        checks,
        "candidate_counterpart_fields_recomputed",
        candidate_match,
        candidate_detail,
        f"{len(expected_candidate_raw)} trajectory-derived rows",
    )

    counterpart_raw_columns = [
        "designated_counterpart_id",
        "counterpart_observed",
        "counterpart_interpolated",
        "counterpart_speed_kmh",
        "counterpart_acceleration_mps2",
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_total_raw",
        "n_frames_lt_m3_raw",
    ]
    expected_counterpart_raw = counterpart[["candidate_key"]].merge(
        expected_candidate_raw[["candidate_key", *counterpart_raw_columns]],
        on="candidate_key",
        how="left",
        validate="one_to_one",
    )
    actual_counterpart = counterpart.sort_values("candidate_key").reset_index(drop=True)
    expected_counterpart_raw = expected_counterpart_raw.sort_values(
        "candidate_key"
    ).reset_index(drop=True)
    counterpart_match, counterpart_detail = _frame_match(
        actual_counterpart,
        expected_counterpart_raw,
        ["candidate_key", *counterpart_raw_columns],
    )
    _check(
        checks,
        "counterpart_window_raw_fields_recomputed",
        counterpart_match,
        counterpart_detail,
        f"{len(expected_counterpart_raw)} candidate-derived rows",
    )

    base_candidates = pd.read_parquet(base / "tables/candidate_moments.parquet")
    replace_columns = [
        column for column in candidate_raw_columns if column in base_candidates
    ]
    expected_base_candidates = base_candidates.drop(columns=replace_columns).merge(
        expected_trajectory[[*join_keys, *candidate_raw_columns]],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    base_counterpart_keys = pd.read_parquet(
        base / "tables/counterpart_windows.parquet",
        columns=["candidate_key", "band_90"],
    )
    base_counterpart_designated = base_counterpart_keys.merge(
        expected_base_candidates[
            [
                "candidate_key",
                "anchor_speed_drop_kmh_raw",
                "window_speed_range_kmh_raw",
                "n_frames_total_raw",
                "n_frames_lt_m3_raw",
            ]
        ],
        on="candidate_key",
        how="left",
        validate="one_to_one",
    )
    return _raw_summary(expected_base_candidates, base_counterpart_designated)


def validate_dataset(
    output: Path = DEFAULT_OUTPUT,
    *,
    base: Path = BASE_OUTPUT,
    source_session: Path = DEFAULT_SOURCE_SESSION,
    write_report: bool = True,
) -> dict[str, Any]:
    base_targets = json.loads(BASE_TARGETS.read_text(encoding="utf-8"))
    paper_targets = json.loads(PAPER_TARGETS.read_text(encoding="utf-8"))
    automated = json.loads(AV_REFERENCE.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "tables/achieved_summary.json").read_text())
    candidates = pd.read_parquet(output / "tables/candidate_moments.parquet")
    counterpart = pd.read_parquet(output / "tables/counterpart_windows.parquet")
    unit = pd.read_csv(output / "tables/per_unit_counts.csv")
    checks: list[dict[str, Any]] = []

    _check(checks, "schema", manifest.get("schema_version") == SCHEMA_VERSION,
           manifest.get("schema_version"), SCHEMA_VERSION)
    _check(checks, "synthetic_status", manifest.get("data_status") == DATA_STATUS,
           manifest.get("data_status"), DATA_STATUS)
    _check(checks, "estimator_not_claimed", manifest.get("estimator_recomputed") is False,
           manifest.get("estimator_recomputed"), False)
    _check(
        checks,
        "human_ego_margin_intervals_not_invented",
        manifest.get("human_ego_margin_intervals_generated") is False,
        manifest.get("human_ego_margin_intervals_generated"),
        False,
    )
    _check(
        checks,
        "paper_claim_match_status",
        manifest.get("paper_claim_match_status")
        == "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL"
        and manifest.get("paper_claim_metric_layer")
        == (
            "Band/status counts and columns ending in _calibrated; raw columns remain "
            "trajectory-derived diagnostics"
        ),
        {
            "paper_claim_match_status": manifest.get("paper_claim_match_status"),
            "paper_claim_metric_layer": manifest.get("paper_claim_metric_layer"),
        },
        {
            "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
            "paper_claim_metric_layer": (
                "Band/status counts and columns ending in _calibrated; raw columns "
                "remain trajectory-derived diagnostics"
            ),
        },
    )
    _check(
        checks,
        "raw_emergent_match_not_claimed",
        manifest.get("raw_emergent_match_status") == "NOT_MATCHED_AND_NOT_CLAIMED"
        and manifest.get("full_distribution_match_claimed") is False,
        {
            "raw_emergent_match_status": manifest.get("raw_emergent_match_status"),
            "full_distribution_match_claimed": manifest.get(
                "full_distribution_match_claimed"
            ),
        },
        {
            "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
            "full_distribution_match_claimed": False,
        },
    )
    _check(
        checks,
        "summary_paper_claim_match_status",
        summary.get("paper_claim_match_status")
        == "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL"
        and summary.get("paper_claim_metric_layer")
        == "aggregate-constrained calibrated fields",
        {
            "paper_claim_match_status": summary.get("paper_claim_match_status"),
            "paper_claim_metric_layer": summary.get("paper_claim_metric_layer"),
        },
        {
            "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
            "paper_claim_metric_layer": "aggregate-constrained calibrated fields",
        },
    )
    _check(
        checks,
        "summary_raw_emergent_match_not_claimed",
        summary.get("raw_emergent_match_status") == "NOT_MATCHED_AND_NOT_CLAIMED"
        and summary.get("full_distribution_match_claimed") is False,
        {
            "raw_emergent_match_status": summary.get("raw_emergent_match_status"),
            "full_distribution_match_claimed": summary.get(
                "full_distribution_match_claimed"
            ),
        },
        {
            "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
            "full_distribution_match_claimed": False,
        },
    )
    _check(
        checks,
        "paper_target_snapshot_integrity",
        summary.get("paper_target_snapshot") == paper_targets,
        summary.get("paper_target_snapshot"),
        paper_targets,
    )
    expected_cluster_contract = (
        "run_id (driver_id x scenario_id) for alpha90 and ego TTC; "
        "synthetic_case_id for counterpart signatures"
    )
    _check(
        checks,
        "bootstrap_cluster_contract",
        manifest.get("bootstrap_cluster") == expected_cluster_contract,
        manifest.get("bootstrap_cluster"),
        expected_cluster_contract,
    )
    base_validation = json.loads(
        (base / "validation/validation_summary.json").read_text(encoding="utf-8")
    )
    _check(checks, "base_v1_validated", base_validation["validation_status"] == "PASS",
           base_validation["validation_status"], "PASS")
    _validate_raw_inheritance(checks, base, output)
    recomputed_v1_raw_summary = _validate_source_counterpart_derivations(
        checks,
        base,
        output,
        candidates,
        counterpart,
        source_session,
    )

    both = candidates.status.eq("OK") & candidates.mechanism2_gate_ok
    _check(checks, "candidate_count", len(candidates) == 78_903,
           len(candidates), 78_903)
    _check(checks, "candidate_key_unique", candidates.candidate_key.nunique() == len(candidates),
           int(candidates.candidate_key.nunique()), len(candidates))
    _check(checks, "gate1_count", int(candidates.status.eq("OK").sum()) == 40_993,
           int(candidates.status.eq("OK").sum()), 40_993)
    _check(checks, "both_gate_count", int(both.sum()) == 15_598,
           int(both.sum()), 15_598)
    for alpha in ("80", "90", "95"):
        bands = candidates.loc[both, f"band_{alpha}"]
        observed = {
            "n_below": int(bands.eq("lower").sum()),
            "n_above": int(bands.eq("upper").sum()),
            "n_inside": int(bands.eq("inside").sum()),
            "n_total": int(len(bands)),
        }
        _check(
            checks,
            f"flag_counts_{alpha}",
            observed == base_targets["flag_counts"][alpha],
            observed,
            base_targets["flag_counts"][alpha],
        )
    for scenario in SCENARIOS:
        frame = candidates[both & candidates.scenario_id.eq(scenario)]
        observed = {
            "n_both": int(len(frame)),
            "n_flagged": int(frame.band_90.isin(["lower", "upper"]).sum()),
        }
        _check(
            checks,
            f"scenario_{scenario}",
            observed == base_targets["per_scenario_alpha90"][scenario],
            observed,
            base_targets["per_scenario_alpha90"][scenario],
        )

    for band in ("lower", "inside"):
        spec = base_targets["signature"]["ego_ttc"][band]
        values = candidates.loc[
            candidates.band_90.eq(band), "future_min_ttc_s_calibrated"
        ].dropna().to_numpy(float)
        observed = {
            "n": len(values),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "lt2_num": int((values < 2.0).sum()),
        }
        expected = {key: spec[key] for key in observed}
        matches = observed["n"] == expected["n"] and observed["lt2_num"] == expected["lt2_num"]
        matches &= all(
            abs(observed[key] - expected[key]) <= 1e-9
            for key in ("q25", "q50", "q75")
        )
        _check(checks, f"ttc_point_{band}", matches, observed, expected)

    cp_target = base_targets["signature"]["counterpart"]
    cp_observed = {
        "n_lower": int(counterpart.band_90.eq("lower").sum()),
        "n_inside": int(counterpart.band_90.eq("inside").sum()),
        "n_cases": int(counterpart.synthetic_case_id.nunique()),
        "drop_lower": float(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "anchor_speed_drop_kmh_calibrated"].median()
        ),
        "drop_inside": float(
            counterpart.loc[counterpart.band_90.eq("inside"),
                             "anchor_speed_drop_kmh_calibrated"].median()
        ),
        "range_lower": float(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "window_speed_range_kmh_calibrated"].median()
        ),
        "range_inside": float(
            counterpart.loc[counterpart.band_90.eq("inside"),
                             "window_speed_range_kmh_calibrated"].median()
        ),
        "brake_lower_num": int(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "n_frames_lt_m3_calibrated"].sum()
        ),
        "brake_lower_den": int(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "n_frames_total_calibrated"].sum()
        ),
        "brake_inside_num": int(
            counterpart.loc[counterpart.band_90.eq("inside"),
                             "n_frames_lt_m3_calibrated"].sum()
        ),
        "brake_inside_den": int(
            counterpart.loc[counterpart.band_90.eq("inside"),
                             "n_frames_total_calibrated"].sum()
        ),
    }
    cp_expected = {
        "n_lower": cp_target["n_lower"],
        "n_inside": cp_target["n_inside"],
        "n_cases": cp_target["n_cases"],
        "drop_lower": cp_target["speed_drop_median_lower_kmh"],
        "drop_inside": cp_target["speed_drop_median_inside_kmh"],
        "range_lower": cp_target["speed_range_median_lower_kmh"],
        "range_inside": cp_target["speed_range_median_inside_kmh"],
        "brake_lower_num": cp_target["brake_lower_num"],
        "brake_lower_den": cp_target["brake_lower_den"],
        "brake_inside_num": cp_target["brake_inside_num"],
        "brake_inside_den": cp_target["brake_inside_den"],
    }
    cp_match = all(
        cp_observed[key] == value
        if isinstance(value, int)
        else abs(cp_observed[key] - value) <= 1e-9
        for key, value in cp_expected.items()
    )
    _check(checks, "counterpart_point_targets", cp_match, cp_observed, cp_expected)

    recomputed_intervals = _intervals_from_detail(candidates, counterpart, unit)
    stored_intervals = summary["synthetic_intervals"]
    interval_integrity = all(
        np.allclose(recomputed_intervals[key], stored_intervals[key], rtol=0.0, atol=1e-12)
        for key in recomputed_intervals
    )
    _check(checks, "interval_summary_integrity", interval_integrity,
           stored_intervals, recomputed_intervals)
    expected_target_intervals = {
        "alpha90": base_targets["flag_rate_ci95_alpha90_target"],
        "ego_ttc": base_targets["signature"]["ego_ttc"]["lt2_diff_ci95_target"],
        "speed_drop": base_targets["signature"]["counterpart"][
            "speed_drop_ratio_ci95_target"
        ],
        "speed_range": base_targets["signature"]["counterpart"][
            "speed_range_ratio_ci95_target"
        ],
        "braking": base_targets["signature"]["counterpart"][
            "brake_diff_ci95_target"
        ],
    }
    _check(
        checks,
        "target_interval_summary_integrity",
        _nested_allclose(summary["target_intervals"], expected_target_intervals),
        summary["target_intervals"],
        expected_target_intervals,
    )
    target_intervals = expected_target_intervals
    direction_match = (
        recomputed_intervals["alpha90"][1] < 0.10
        and recomputed_intervals["ego_ttc"][1] < 0.0
        and recomputed_intervals["speed_drop"][0] > 1.0
        and recomputed_intervals["speed_range"][0] > 1.0
        and recomputed_intervals["braking"][1] < 0.0
    )
    _check(checks, "paper_direction_decisions", direction_match,
           recomputed_intervals, "same side of null as paper")

    tolerances = {
        "alpha90": paper_targets["soft_tolerances"]["rate_ci_endpoint_abs"],
        "ego_ttc": paper_targets["soft_tolerances"]["difference_ci_endpoint_abs"],
        "speed_drop": paper_targets["soft_tolerances"]["ratio_ci_endpoint_abs"],
        "speed_range": paper_targets["soft_tolerances"]["ratio_ci_endpoint_abs"],
        "braking": paper_targets["soft_tolerances"]["difference_ci_endpoint_abs"],
    }
    soft_matches: dict[str, bool] = {}
    endpoint_errors: dict[str, list[float]] = {}
    for key, interval in recomputed_intervals.items():
        errors = np.abs(np.asarray(interval) - np.asarray(target_intervals[key]))
        endpoint_errors[key] = [float(value) for value in errors]
        soft_matches[key] = bool(np.all(errors <= tolerances[key]))
        _check(
            checks,
            f"soft_interval_{key}",
            soft_matches[key],
            endpoint_errors[key],
            f"each <= {tolerances[key]}",
            "soft",
        )

    v2_human_targets = copy.deepcopy(base_targets)
    alpha90 = candidates.loc[both, "band_90"]
    v2_human_targets["flag_counts"]["90"] = {
        "n_below": int(alpha90.eq("lower").sum()),
        "n_above": int(alpha90.eq("upper").sum()),
        "n_inside": int(alpha90.eq("inside").sum()),
        "n_total": int(len(alpha90)),
    }
    v2_human_targets["per_scenario_alpha90"] = {}
    for scenario in SCENARIOS:
        frame = candidates[both & candidates.scenario_id.eq(scenario)]
        v2_human_targets["per_scenario_alpha90"][scenario] = {
            "n_both": int(len(frame)),
            "n_flagged": int(frame.band_90.isin(["lower", "upper"]).sum()),
        }
    paper_metrics = _paper_metrics(v2_human_targets, automated)
    _check(
        checks,
        "paper_metric_summary_integrity",
        _nested_allclose(summary["paper_hard_metrics"], paper_metrics),
        summary["paper_hard_metrics"],
        paper_metrics,
    )
    av_target = paper_targets["automated_alpha90"]
    human_target = paper_targets["human_alpha90"]
    ratio_target = paper_targets["automated_to_human"]
    paper_display_tolerance = av_target["paper_display_tolerance"]
    hard_paper = (
        abs(paper_metrics["moment_weighted_ratio"] - ratio_target["moment_weighted_ratio"])
        <= 1e-12
        and np.allclose(
            paper_metrics["scenario_bootstrap_ci95"],
            ratio_target["scenario_bootstrap_ci95_rounded"],
            rtol=0.0,
            atol=0.005,
        )
        and np.allclose(
            paper_metrics["human_scenario_rate_range"],
            human_target["scenario_rate_range_rounded"],
            rtol=0.0,
            atol=0.0005,
        )
        and np.allclose(
            paper_metrics["automated_scenario_rate_range"],
            av_target["scenario_rate_range_from_frozen_counts"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            paper_metrics["automated_scenario_rate_range"],
            av_target["scenario_rate_range_paper_display"],
            rtol=0.0,
            atol=paper_display_tolerance,
        )
        and abs(
            paper_metrics["scenario_weighted_ratio"]
            - ratio_target["scenario_weighted_ratio_from_exact_counts"]
        )
        <= 1e-12
        and paper_metrics["draws_above_parity"]
        == ratio_target["draws_above_parity"]
        and paper_metrics["scenarios_av_higher"]
        == ratio_target["scenarios_av_higher"]
    )
    _check(checks, "paper_records_from_v2_detail", hard_paper, paper_metrics,
           paper_targets["automated_to_human"])

    v1_achieved = json.loads(
        (base / "tables/achieved_summary.json").read_text(encoding="utf-8")
    )
    v1_intervals = {
        "alpha90": v1_achieved["flag_rate_ci95_alpha90_synthetic_bootstrap"],
        "ego_ttc": v1_achieved["signature"]["ego_ttc"][
            "lt2_difference_ci95_synthetic_bootstrap"
        ],
        "speed_drop": v1_achieved["signature"]["counterpart"][
            "speed_drop_ratio_ci95_synthetic_bootstrap"
        ],
        "speed_range": v1_achieved["signature"]["counterpart"][
            "speed_range_ratio_ci95_synthetic_bootstrap"
        ],
        "braking": v1_achieved["signature"]["counterpart"][
            "brake_difference_ci95_synthetic_bootstrap"
        ],
    }
    ci_loss_v1 = _ci_loss(v1_intervals, base_targets)
    ci_loss_v2 = _ci_loss(recomputed_intervals, base_targets)
    _check(checks, "ci_closer_than_v1", ci_loss_v2 < ci_loss_v1,
           ci_loss_v2, f"< {ci_loss_v1}")

    stored_raw_v1 = summary["raw_summary_v1_designated_counterpart"]
    _check(
        checks,
        "v1_designated_raw_summary_integrity",
        _nested_allclose(stored_raw_v1, recomputed_v1_raw_summary),
        stored_raw_v1,
        recomputed_v1_raw_summary,
    )
    raw_v1 = recomputed_v1_raw_summary
    raw_v2 = _raw_summary(candidates, counterpart)
    _check(
        checks,
        "v2_raw_summary_integrity",
        _nested_allclose(summary["raw_summary"], raw_v2),
        summary["raw_summary"],
        raw_v2,
    )
    raw_loss_v1 = _raw_closeness_loss(raw_v1, base_targets)
    raw_loss_v2 = _raw_closeness_loss(raw_v2, base_targets)
    _check(checks, "raw_outcomes_closer_than_v1", raw_loss_v2 < raw_loss_v1,
           raw_loss_v2, f"< {raw_loss_v1}")
    _check(checks, "synthetic_case_count", counterpart.synthetic_case_id.nunique() == 186,
           int(counterpart.synthetic_case_id.nunique()), 186)
    _check(checks, "designated_counterpart_only",
           counterpart.designated_counterpart_id.notna().all(),
           int(counterpart.designated_counterpart_id.notna().sum()), len(counterpart))

    hard_failures = [
        row for row in checks if row["status"] == "FAIL" and row["severity"] != "soft"
    ]
    soft_match_count = sum(soft_matches.values())
    status = "PASS" if not hard_failures else "FAIL"
    soft_status = "FULL_SOFT_MATCH" if soft_match_count == len(soft_matches) else "PARTIAL_SOFT_MATCH"
    hard_checks_total = sum(row["severity"] != "soft" for row in checks)
    result = {
        "validation_status": status,
        "soft_match_status": soft_status,
        "paper_claim_match_status": "POINT_ESTIMATES_EXACT_INTERVALS_PARTIAL",
        "raw_emergent_match_status": "NOT_MATCHED_AND_NOT_CLAIMED",
        "full_distribution_match_claimed": False,
        "checks_total": len(checks),
        "checks_hard_failed": len(hard_failures),
        "hard_checks_passed": hard_checks_total - len(hard_failures),
        "hard_checks_total": hard_checks_total,
        "failed_hard_checks": [row["check"] for row in hard_failures],
        "soft_intervals_within_tolerance": soft_match_count,
        "soft_intervals_total": len(soft_matches),
        "soft_matches": soft_matches,
        "endpoint_errors": endpoint_errors,
        "ci_loss_v1": ci_loss_v1,
        "ci_loss_v2": ci_loss_v2,
        "ci_loss_improvement_fraction": float((ci_loss_v1 - ci_loss_v2) / ci_loss_v1),
        "raw_loss_v1": raw_loss_v1,
        "raw_loss_v2": raw_loss_v2,
        "raw_loss_improvement_fraction": float((raw_loss_v1 - raw_loss_v2) / raw_loss_v1),
        "recomputed_intervals": recomputed_intervals,
        "paper_hard_metrics": paper_metrics,
        "boundary": (
            "Hard paper records and point summaries are reproduced. Remaining interval endpoint "
            "differences reflect non-identifiable synthetic cluster structure and are disclosed."
        ),
    }
    if write_report:
        validation = output / "validation"
        validation.mkdir(exist_ok=True)
        pd.DataFrame(checks).to_csv(validation / "checks.csv", index=False)
        (validation / "validation_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        soft_lines = "\n".join(
            f"- {key}: {'PASS' if soft_matches[key] else 'PARTIAL'}; errors {endpoint_errors[key]}"
            for key in soft_matches
        )
        (validation / "REPORT.md").write_text(
            "# RQ029 v2 paper-alignment validation\n\n"
            f"- Overall: **{status} / {soft_status}**\n"
            f"- Hard checks: {hard_checks_total - len(hard_failures)}/"
            f"{hard_checks_total} passed\n"
            f"- CI loss improvement vs v1: {result['ci_loss_improvement_fraction']:.1%}\n"
            f"- Raw-outcome closeness improvement vs v1: "
            f"{result['raw_loss_improvement_fraction']:.1%}\n"
            "- Paper scenario bootstrap: "
            f"{paper_metrics['moment_weighted_ratio']:.6f}, "
            f"CI {paper_metrics['scenario_bootstrap_ci95']}, "
            f"above parity {paper_metrics['draws_above_parity']}/20000\n\n"
            "## Soft interval checks\n\n"
            f"{soft_lines}\n\n"
            "## Boundary\n\n"
            f"{result['boundary']} The package remains **SYNTHETIC_NOT_OBSERVED**.\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-output", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--source-session", type=Path, default=DEFAULT_SOURCE_SESSION)
    args = parser.parse_args()
    result = validate_dataset(
        args.output_dir,
        base=args.base_output,
        source_session=args.source_session,
    )
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["validation_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
