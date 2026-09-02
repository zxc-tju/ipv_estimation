#!/usr/bin/env python3
"""Validate the RQ029 Shanghai synthetic human-driving proxy package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.simulation.generate_rq029_human_synthetic import (  # noqa: E402
    DATA_STATUS,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_SESSION,
    DEFAULT_TARGETS,
    REQUIRED_LOGS,
    SCENARIOS,
    SCHEMA_VERSION,
    _sha256,
    _stable_seed,
)


def _independent_ego_actor(record: dict[str, Any]) -> dict[str, Any]:
    egos = [
        actor
        for group in record.get("participantTrajectories", [])
        for actor in group.get("value", [])
        if actor.get("isPerception") == 0
    ]
    if len(egos) != 1:
        raise ValueError(f"expected one ego object, found {len(egos)}")
    return egos[0]


def _independent_background_payload(record: dict[str, Any]) -> str:
    background = copy.deepcopy(record)
    for group in background.get("participantTrajectories", []):
        group["value"] = [
            actor for actor in group.get("value", []) if actor.get("isPerception") != 0
        ]
    return json.dumps(background, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _background_hash_and_lines(
    path: Path,
) -> tuple[str, str, int, pd.DataFrame]:
    semantic = hashlib.sha256()
    byte_hash = hashlib.sha256()
    lines = 0
    ego_rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            byte_hash.update(raw)
            record = json.loads(raw)
            ego = _independent_ego_actor(record)
            semantic.update(_independent_background_payload(record).encode("utf-8"))
            semantic.update(b"\n")
            ego_rows.append(
                {
                    "source_line_index": lines,
                    "native_case_id": int(record["caseId"]),
                    "timestamp_ms": int(ego["globalTimeStamp"]),
                    "ego_latitude": float(ego["latitude"]),
                    "ego_longitude": float(ego["longitude"]),
                    "ego_speed_kmh": float(ego["speed"]),
                    "ego_course_deg": float(ego["courseAngle"]),
                }
            )
            lines += 1
    return semantic.hexdigest(), byte_hash.hexdigest(), lines, pd.DataFrame(ego_rows)


def _vehicle_ego_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_index, raw in enumerate(handle):
            record = json.loads(raw)
            values = record.get("value", {}).get("value", [])
            if len(values) != 1 or values[0].get("isPerception") != 0:
                raise ValueError(f"invalid vehicle ego row at {path}:{line_index + 1}")
            ego = values[0]
            rows.append(
                {
                    "source_line_index": line_index,
                    "timestamp_ms": int(ego["globalTimeStamp"]),
                    "ego_latitude": float(ego["latitude"]),
                    "ego_longitude": float(ego["longitude"]),
                    "ego_speed_kmh": float(ego["speed"]),
                    "ego_course_deg": float(ego["courseAngle"]),
                }
            )
    return pd.DataFrame(rows)


def _max_polyline_distance_m(
    latitude: np.ndarray,
    longitude: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> float:
    """Independently measure point-to-source-polyline distance in local metres."""

    lat0 = float(source_latitude[0])
    lon0 = float(source_longitude[0])
    scale_x = 111_320.0 * np.cos(np.radians(lat0))
    points = np.column_stack(
        [(longitude - lon0) * scale_x, (latitude - lat0) * 111_320.0]
    )
    source = np.column_stack(
        [
            (source_longitude - lon0) * scale_x,
            (source_latitude - lat0) * 111_320.0,
        ]
    )
    segment_start = source[:-1]
    segment_vector = source[1:] - source[:-1]
    segment_norm_sq = np.sum(segment_vector * segment_vector, axis=1)
    if not len(segment_start):
        return float(np.max(np.linalg.norm(points - source[0], axis=1)))
    maximum = 0.0
    for start in range(0, len(points), 128):
        batch = points[start: start + 128]
        relative = batch[:, None, :] - segment_start[None, :, :]
        numerator = np.sum(relative * segment_vector[None, :, :], axis=2)
        denominator = np.where(segment_norm_sq > 0, segment_norm_sq, 1.0)
        fraction = np.clip(numerator / denominator[None, :], 0.0, 1.0)
        projection = segment_start[None, :, :] + fraction[:, :, None] * segment_vector
        distances = np.linalg.norm(batch[:, None, :] - projection, axis=2)
        maximum = max(maximum, float(np.min(distances, axis=1).max()))
    return maximum


def _add_check(
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


def _recompute_bootstrap_intervals(
    candidates: pd.DataFrame,
    counterpart: pd.DataFrame,
    unit: pd.DataFrame,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(_stable_seed(seed, "alpha90_bootstrap"))
    unit_values = unit[["n_both", "n_below_90", "n_above_90"]].to_numpy(int)
    flag_rates: list[float] = []
    for _ in range(1000):
        sampled = unit_values[rng.integers(0, len(unit_values), len(unit_values))]
        flag_rates.append(
            float((sampled[:, 1] + sampled[:, 2]).sum() / sampled[:, 0].sum())
        )

    ttc_rows = candidates[
        candidates.band_90.isin(["lower", "inside"])
        & candidates.future_min_ttc_s_calibrated.notna()
    ][["run_id", "band_90", "future_min_ttc_s_calibrated"]]
    ttc_by_run = {run_id: group for run_id, group in ttc_rows.groupby("run_id")}
    all_run_ids = np.asarray(sorted(candidates.run_id.unique()))
    rng = np.random.default_rng(_stable_seed(seed, "ttc_tail_bootstrap"))
    ttc_differences: list[float] = []
    for _ in range(1000):
        sampled_ids = rng.choice(all_run_ids, len(all_run_ids), replace=True)
        sampled = pd.concat(
            [ttc_by_run[run_id] for run_id in sampled_ids if run_id in ttc_by_run],
            ignore_index=True,
        )
        lower = sampled.loc[
            sampled.band_90.eq("lower"), "future_min_ttc_s_calibrated"
        ].to_numpy(float)
        inside = sampled.loc[
            sampled.band_90.eq("inside"), "future_min_ttc_s_calibrated"
        ].to_numpy(float)
        ttc_differences.append(
            float((lower < 2.0).mean() - (inside < 2.0).mean())
        )

    cp_by_run = {run_id: group for run_id, group in counterpart.groupby("run_id")}
    cp_run_ids = np.asarray(sorted(cp_by_run))
    rng = np.random.default_rng(_stable_seed(seed, "counterpart_bootstrap"))
    drop_ratios: list[float] = []
    range_ratios: list[float] = []
    brake_differences: list[float] = []
    for _ in range(1000):
        sampled_ids = rng.choice(cp_run_ids, len(cp_run_ids), replace=True)
        sampled = pd.concat([cp_by_run[run_id] for run_id in sampled_ids], ignore_index=True)
        lower = sampled[sampled.band_90.eq("lower")]
        inside = sampled[sampled.band_90.eq("inside")]
        drop_ratios.append(
            float(
                lower.anchor_speed_drop_kmh_calibrated.median()
                / inside.anchor_speed_drop_kmh_calibrated.median()
            )
        )
        range_ratios.append(
            float(
                lower.window_speed_range_kmh_calibrated.median()
                / inside.window_speed_range_kmh_calibrated.median()
            )
        )
        brake_differences.append(
            float(
                lower.n_frames_lt_m3_calibrated.sum()
                / lower.n_frames_total_calibrated.sum()
                - inside.n_frames_lt_m3_calibrated.sum()
                / inside.n_frames_total_calibrated.sum()
            )
        )

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "alpha90_flag_rate": interval(flag_rates),
        "ego_ttc_lt2_difference": interval(ttc_differences),
        "counterpart_speed_drop_ratio": interval(drop_ratios),
        "counterpart_speed_range_ratio": interval(range_ratios),
        "counterpart_brake_difference": interval(brake_differences),
    }


def validate_dataset(
    output: Path = DEFAULT_OUTPUT,
    source_session: Path = DEFAULT_SOURCE_SESSION,
    targets_path: Path = DEFAULT_TARGETS,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    _add_check(checks, "schema_version", manifest.get("schema_version") == SCHEMA_VERSION,
               manifest.get("schema_version"), SCHEMA_VERSION)
    _add_check(checks, "synthetic_status", manifest.get("data_status") == DATA_STATUS,
               manifest.get("data_status"), DATA_STATUS)
    _add_check(checks, "collection_area", manifest.get("collection_area") == "shanghai",
               manifest.get("collection_area"), "shanghai")
    _add_check(checks, "estimator_not_claimed", manifest.get("estimator_recomputed") is False,
               manifest.get("estimator_recomputed"), False)

    tables = output / "tables"
    runs = pd.read_csv(tables / "runs.csv")
    trajectories = pd.read_parquet(tables / "trajectory_pairs.parquet")
    candidates = pd.read_parquet(tables / "candidate_moments.parquet")
    counterpart = pd.read_parquet(tables / "counterpart_windows.parquet")
    unit = pd.read_csv(tables / "per_unit_counts.csv")
    achieved = json.loads((tables / "achieved_summary.json").read_text(encoding="utf-8"))
    recomputed_cis = _recompute_bootstrap_intervals(
        candidates, counterpart, unit, int(manifest["seed"])
    )

    _add_check(checks, "driver_count", runs.driver_id.nunique() == targets["n_drivers"],
               int(runs.driver_id.nunique()), targets["n_drivers"])
    _add_check(checks, "scenario_count", set(runs.scenario_id) == set(SCENARIOS),
               sorted(runs.scenario_id.unique()), list(SCENARIOS))
    _add_check(checks, "run_count", len(runs) == targets["n_runs"], len(runs), targets["n_runs"])
    _add_check(checks, "unique_runs", runs.run_id.nunique() == len(runs),
               int(runs.run_id.nunique()), len(runs))
    _add_check(checks, "all_runs_shanghai", set(runs.area) == {"shanghai"},
               sorted(runs.area.unique()), ["shanghai"])

    source_semantic, source_bytes, source_lines, _ = _background_hash_and_lines(
        source_session / "vehicle_perception_simulation_trajectory.log"
    )
    _add_check(
        checks,
        "source_background_manifest_hash",
        source_semantic == manifest["source_background_semantic_sha256"],
        source_semantic,
        manifest["source_background_semantic_sha256"],
    )
    shared_sim = output / "shared_source_logs/simulation_trajectory.log"
    source_sim_hash = _sha256(source_session / "simulation_trajectory.log")
    _add_check(checks, "shared_simulation_byte_identity", _sha256(shared_sim) == source_sim_hash,
               _sha256(shared_sim), source_sim_hash)

    for driver_id in sorted(runs.driver_id.unique()):
        session_id = runs.loc[runs.driver_id == driver_id, "synthetic_session_id"].iloc[0]
        session = output / "raw/drivers" / driver_id / "sessions" / session_id
        missing = [name for name in REQUIRED_LOGS if not (session / name).is_file()]
        _add_check(checks, f"{driver_id}_required_logs", not missing, missing, [])
        sim = session / "simulation_trajectory.log"
        sim_ok = os.path.samefile(sim, shared_sim) or _sha256(sim) == source_sim_hash
        observed_identity = (
            _sha256(sim) if not os.path.samefile(sim, shared_sim) else "samefile"
        )
        _add_check(
            checks,
            f"{driver_id}_simulation_identity",
            sim_ok,
            observed_identity,
            "samefile_or_sha",
        )
        perception = session / "vehicle_perception_simulation_trajectory.log"
        semantic, byte_hash, line_count, raw_perception_ego = _background_hash_and_lines(
            perception
        )
        _add_check(checks, f"{driver_id}_background_semantic_identity",
                   semantic == source_semantic, semantic, source_semantic)
        _add_check(checks, f"{driver_id}_perception_line_count",
                   line_count == source_lines, line_count, source_lines)
        raw_vehicle_ego = _vehicle_ego_rows(session / "vehicle_trajectory.log")
        vehicle_lines = len(raw_vehicle_ego)
        _add_check(checks, f"{driver_id}_vehicle_line_count",
                   vehicle_lines == source_lines, vehicle_lines, source_lines)
        table_ego = trajectories.loc[
            trajectories.driver_id.eq(driver_id),
            [
                "source_line_index",
                "native_case_id",
                "timestamp_ms",
                "ego_latitude",
                "ego_longitude",
                "ego_speed_kmh",
                "ego_course_deg",
            ],
        ].sort_values("source_line_index").reset_index(drop=True)
        raw_perception_ego = raw_perception_ego.sort_values(
            "source_line_index"
        ).reset_index(drop=True)
        raw_vehicle_ego = raw_vehicle_ego.sort_values("source_line_index").reset_index(
            drop=True
        )
        perception_keys_match = np.array_equal(
            raw_perception_ego[
                ["source_line_index", "native_case_id", "timestamp_ms"]
            ].to_numpy(),
            table_ego[["source_line_index", "native_case_id", "timestamp_ms"]].to_numpy(),
        )
        perception_motion_match = np.allclose(
            raw_perception_ego[
                ["ego_latitude", "ego_longitude", "ego_speed_kmh", "ego_course_deg"]
            ].to_numpy(float),
            table_ego[
                ["ego_latitude", "ego_longitude", "ego_speed_kmh", "ego_course_deg"]
            ].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        )
        _add_check(
            checks,
            f"{driver_id}_raw_perception_ego_matches_table",
            bool(perception_keys_match and perception_motion_match),
            bool(perception_keys_match and perception_motion_match),
            True,
        )
        vehicle_motion_match = len(raw_vehicle_ego) == len(raw_perception_ego) and np.allclose(
            raw_vehicle_ego[
                ["ego_latitude", "ego_longitude", "ego_speed_kmh", "ego_course_deg"]
            ].to_numpy(float),
            raw_perception_ego[
                ["ego_latitude", "ego_longitude", "ego_speed_kmh", "ego_course_deg"]
            ].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        )
        _add_check(
            checks,
            f"{driver_id}_raw_vehicle_ego_matches_perception",
            bool(vehicle_motion_match),
            bool(vehicle_motion_match),
            True,
        )
        for path in session.iterdir():
            if path.is_file():
                inventory.append(
                    {
                        "path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size,
                        "sha256": (
                            byte_hash if path == perception else "shared_or_small_not_rehashed"
                        ),
                    }
                )

    _add_check(checks, "trajectory_row_count",
               len(trajectories) == targets["n_drivers"] * source_lines,
               len(trajectories), targets["n_drivers"] * source_lines)
    required_numeric = [
        "ego_latitude", "ego_longitude", "ego_speed_kmh", "ego_course_deg",
        "source_ego_latitude", "source_ego_longitude", "source_ego_speed_kmh",
    ]
    finite = np.isfinite(trajectories[required_numeric].to_numpy(float)).all()
    _add_check(checks, "finite_ego_kinematics", bool(finite), bool(finite), True)
    monotone = all(
        np.all(np.diff(group.timestamp_ms.to_numpy()) > 0)
        for _, group in trajectories.groupby(["driver_id", "scenario_id"])
    )
    _add_check(checks, "monotone_timestamps", monotone, monotone, True)
    max_cross = float(trajectories.path_cross_track_m.max())
    _add_check(
        checks,
        "reported_path_cross_track",
        max_cross <= 0.05,
        max_cross,
        "<=0.05 m",
    )
    source_paths: dict[str, pd.DataFrame] = {}
    first_driver = sorted(trajectories.driver_id.unique())[0]
    for scenario, group in trajectories[
        trajectories.driver_id.eq(first_driver)
    ].groupby("scenario_id"):
        source_paths[scenario] = group.sort_values("scenario_frame_index")
    recomputed_cross = 0.0
    for (_, scenario), group in trajectories.groupby(["driver_id", "scenario_id"]):
        group = group.sort_values("scenario_frame_index")
        source = source_paths[scenario]
        recomputed_cross = max(
            recomputed_cross,
            _max_polyline_distance_m(
                group.ego_latitude.to_numpy(float),
                group.ego_longitude.to_numpy(float),
                source.source_ego_latitude.to_numpy(float),
                source.source_ego_longitude.to_numpy(float),
            ),
        )
    _add_check(
        checks,
        "recomputed_path_cross_track",
        recomputed_cross <= 0.05,
        recomputed_cross,
        "<=0.05 m",
    )
    endpoints_ok = True
    for _, group in trajectories.sort_values("scenario_frame_index").groupby(
        ["driver_id", "scenario_id"]
    ):
        for row in (group.iloc[0], group.iloc[-1]):
            endpoints_ok &= math_isclose(row.ego_latitude, row.source_ego_latitude)
            endpoints_ok &= math_isclose(row.ego_longitude, row.source_ego_longitude)
    _add_check(checks, "same_path_endpoints", endpoints_ok, endpoints_ok, True)
    positive = trajectories.source_ego_speed_kmh > 1.0
    speed_relative = np.abs(
        trajectories.loc[positive, "ego_speed_kmh"].to_numpy(float)
        / trajectories.loc[positive, "source_ego_speed_kmh"].to_numpy(float)
        - 1.0
    )
    speed_p95 = float(np.quantile(speed_relative, 0.95))
    _add_check(checks, "speed_diff_nonzero", float(speed_relative.max()) > 0,
               float(speed_relative.max()), ">0")
    _add_check(checks, "speed_diff_small", speed_p95 <= 0.12, speed_p95, "<=0.12")

    both = (candidates.status == "OK") & candidates.mechanism2_gate_ok
    _add_check(checks, "candidate_count",
               len(candidates) == targets["gates"]["n_candidate_moments"],
               len(candidates), targets["gates"]["n_candidate_moments"])
    _add_check(checks, "candidate_key_unique",
               candidates.candidate_key.nunique() == len(candidates),
               int(candidates.candidate_key.nunique()), len(candidates))
    _add_check(checks, "gate1_count",
               int(candidates.status.eq("OK").sum()) == targets["gates"]["n_gate1_pass"],
               int(candidates.status.eq("OK").sum()), targets["gates"]["n_gate1_pass"])
    gate1_ipv_complete = candidates.loc[candidates.status.eq("OK"), "ipv_log"].notna().all()
    _add_check(checks, "gate1_ipv_complete", bool(gate1_ipv_complete),
               bool(gate1_ipv_complete), True)
    _add_check(checks, "both_gate_count", int(both.sum()) == targets["gates"]["n_both_gates"],
               int(both.sum()), targets["gates"]["n_both_gates"])

    for alpha in ("80", "90", "95"):
        bands = candidates.loc[both, f"band_{alpha}"]
        observed = {
            "n_below": int(bands.eq("lower").sum()),
            "n_above": int(bands.eq("upper").sum()),
            "n_inside": int(bands.eq("inside").sum()),
            "n_total": int(len(bands)),
        }
        _add_check(checks, f"flag_counts_{alpha}", observed == targets["flag_counts"][alpha],
                   observed, targets["flag_counts"][alpha])
    nested_lower = (
        set(candidates.index[candidates.band_95.eq("lower")])
        <= set(candidates.index[candidates.band_90.eq("lower")])
        <= set(candidates.index[candidates.band_80.eq("lower")])
    )
    nested_upper = (
        set(candidates.index[candidates.band_95.eq("upper")])
        <= set(candidates.index[candidates.band_90.eq("upper")])
        <= set(candidates.index[candidates.band_80.eq("upper")])
    )
    _add_check(checks, "nested_reference_bands", nested_lower and nested_upper,
               nested_lower and nested_upper, True)

    for scenario in SCENARIOS:
        subset = candidates[both & candidates.scenario_id.eq(scenario)]
        observed = {
            "n_both": int(len(subset)),
            "n_flagged": int(subset.band_90.isin(["lower", "upper"]).sum()),
        }
        expected = targets["per_scenario_alpha90"][scenario]
        _add_check(checks, f"scenario_{scenario}_alpha90", observed == expected,
                   observed, expected)

    for band in ("lower", "inside"):
        spec = targets["signature"]["ego_ttc"][band]
        values = candidates.loc[
            candidates.band_90.eq(band), "future_min_ttc_s_calibrated"
        ].dropna().to_numpy(float)
        observed = {
            "n": int(len(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "lt2_num": int((values < 2.0).sum()),
        }
        expected = {key: spec[key] for key in ("n", "q25", "q50", "q75", "lt2_num")}
        passed = observed["n"] == expected["n"] and observed["lt2_num"] == expected["lt2_num"]
        passed &= all(abs(observed[key] - expected[key]) <= 1e-9 for key in ("q25", "q50", "q75"))
        _add_check(checks, f"ego_ttc_{band}", passed, observed, expected)

    cp_target = targets["signature"]["counterpart"]
    raw_columns = {
        "anchor_speed_drop_kmh_raw",
        "window_speed_range_kmh_raw",
        "n_frames_total_raw",
        "n_frames_lt_m3_raw",
    }
    _add_check(
        checks,
        "counterpart_raw_and_calibrated_separated",
        raw_columns <= set(counterpart.columns),
        sorted(raw_columns & set(counterpart.columns)),
        sorted(raw_columns),
    )
    cp_observed = {
        "n_lower": int(counterpart.band_90.eq("lower").sum()),
        "n_inside": int(counterpart.band_90.eq("inside").sum()),
        "n_pseudo_cases": int(counterpart.pseudo_case_id.nunique()),
        "speed_drop_median_lower_kmh": float(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "anchor_speed_drop_kmh_calibrated"].median()
        ),
        "speed_drop_median_inside_kmh": float(
            counterpart.loc[counterpart.band_90.eq("inside"),
                             "anchor_speed_drop_kmh_calibrated"].median()
        ),
        "speed_range_median_lower_kmh": float(
            counterpart.loc[counterpart.band_90.eq("lower"),
                             "window_speed_range_kmh_calibrated"].median()
        ),
        "speed_range_median_inside_kmh": float(
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
        **{key: cp_target[key] for key in cp_observed if key != "n_pseudo_cases"},
        "n_pseudo_cases": cp_target["n_cases"],
    }
    cp_pass = all(
        cp_observed[key] == cp_expected[key]
        if isinstance(cp_expected[key], int)
        else abs(cp_observed[key] - cp_expected[key]) <= 1e-9
        for key in cp_observed
    )
    _add_check(checks, "counterpart_signature", cp_pass, cp_observed, cp_expected)

    unit_sums = {
        "n_candidate": int(unit.n_candidate.sum()),
        "n_gate1": int(unit.n_gate1.sum()),
        "n_both": int(unit.n_both.sum()),
    }
    _add_check(
        checks,
        "per_unit_conservation",
        unit_sums == {
            "n_candidate": targets["gates"]["n_candidate_moments"],
            "n_gate1": targets["gates"]["n_gate1_pass"],
            "n_both": targets["gates"]["n_both_gates"],
        },
        unit_sums,
        targets["gates"],
    )
    _add_check(checks, "summary_marks_aggregate_constraint",
               achieved.get("estimator_recomputed") is False,
               achieved.get("estimator_recomputed"), False)
    summary_cis = {
        "alpha90_flag_rate": achieved["flag_rate_ci95_alpha90_synthetic_bootstrap"],
        "ego_ttc_lt2_difference": achieved["signature"]["ego_ttc"][
            "lt2_difference_ci95_synthetic_bootstrap"
        ],
        "counterpart_speed_drop_ratio": achieved["signature"]["counterpart"][
            "speed_drop_ratio_ci95_synthetic_bootstrap"
        ],
        "counterpart_speed_range_ratio": achieved["signature"]["counterpart"][
            "speed_range_ratio_ci95_synthetic_bootstrap"
        ],
        "counterpart_brake_difference": achieved["signature"]["counterpart"][
            "brake_difference_ci95_synthetic_bootstrap"
        ],
    }
    ci_summary_matches = all(
        np.allclose(summary_cis[key], recomputed_cis[key], rtol=0.0, atol=1e-12)
        for key in recomputed_cis
    )
    _add_check(
        checks,
        "achieved_summary_ci_integrity",
        ci_summary_matches,
        summary_cis,
        recomputed_cis,
        "critical",
    )
    synthetic_flag_ci = recomputed_cis["alpha90_flag_rate"]
    _add_check(
        checks,
        "alpha90_ci_same_nominal_decision",
        synthetic_flag_ci[1] < 0.10 and targets["flag_rate_ci95_alpha90_target"][1] < 0.10,
        synthetic_flag_ci,
        "upper bound < nominal 0.10",
        "high",
    )
    ego_ci = recomputed_cis["ego_ttc_lt2_difference"]
    _add_check(checks, "ego_tail_ci_same_direction", ego_ci[1] < 0.0, ego_ci,
               "upper bound < 0", "high")
    drop_ci = recomputed_cis["counterpart_speed_drop_ratio"]
    range_ci = recomputed_cis["counterpart_speed_range_ratio"]
    brake_ci = recomputed_cis["counterpart_brake_difference"]
    _add_check(checks, "counterpart_drop_ci_same_direction", drop_ci[0] > 1.0,
               drop_ci, "lower bound > 1", "high")
    _add_check(checks, "counterpart_range_ci_same_direction", range_ci[0] > 1.0,
               range_ci, "lower bound > 1", "high")
    _add_check(checks, "counterpart_brake_ci_same_direction", brake_ci[1] < 0.0,
               brake_ci, "upper bound < 0", "high")

    failed = [check for check in checks if check["status"] == "FAIL"]
    payload_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and output / "validation" not in path.parents
    ]
    logical_bytes = sum(path.stat().st_size for path in payload_files)
    unique_inode_bytes = sum(
        stat.st_size
        for stat in {
            (path.stat().st_dev, path.stat().st_ino): path.stat()
            for path in payload_files
        }.values()
    )
    summary = {
        "validation_status": "PASS" if not failed else "FAIL",
        "data_status": DATA_STATUS,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "failed_checks": [check["check"] for check in failed],
        "source_perception_sha256": source_bytes,
        "source_perception_lines": source_lines,
        "payload_logical_bytes": logical_bytes,
        "payload_unique_inode_bytes": unique_inode_bytes,
        "speed_relative_abs_p95": speed_p95,
        "recomputed_path_cross_track_max_m": recomputed_cross,
        "constrained_marginal_target_match": not any(
            check["status"] == "FAIL"
            for check in checks
            if check["check"].startswith(
                (
                    "candidate",
                    "gate",
                    "both",
                    "flag",
                    "scenario",
                    "ego_ttc",
                    "counterpart",
                    "per_unit",
                )
            )
        ),
        "statistical_direction_match": not any(
            check["status"] == "FAIL"
            for check in checks
            if check["check"].endswith(("same_nominal_decision", "same_direction"))
        ),
        "interval_values_forced_to_target": False,
        "synthetic_bootstrap_ci95_alpha90": synthetic_flag_ci,
        "target_bootstrap_ci95_alpha90": targets["flag_rate_ci95_alpha90_target"],
        "boundary": (
            "Counts and calibrated marginal signatures are reproduced by construction; "
            "the frozen IPV estimator was not rerun and the unknown observed joint distribution "
            "cannot be recovered from aggregate statistics."
        ),
    }
    if write_report:
        validation = output / "validation"
        validation.mkdir(exist_ok=True)
        pd.DataFrame(checks).to_csv(validation / "checks.csv", index=False)
        pd.DataFrame(inventory).to_csv(validation / "raw_file_inventory.csv", index=False)
        (validation / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        failed_text = "none" if not failed else ", ".join(check["check"] for check in failed)
        (validation / "REPORT.md").write_text(
            "# RQ029 validation report\n\n"
            f"- Overall: **{summary['validation_status']}** ({summary['checks_passed']}/"
            f"{summary['checks_total']} checks passed)\n"
            f"- Failed checks: {failed_text}\n"
            f"- Structure: {targets['n_drivers']} synthetic drivers × 15 Shanghai scenarios = "
            f"{targets['n_runs']} runs\n"
            f"- Payload storage: {unique_inode_bytes} unique-inode bytes "
            f"({logical_bytes} logical bytes with shared hardlinks)\n"
            "- Background: source simulation bytes and all non-ego perception payloads compared\n"
            f"- Recomputed maximum path cross-track error: {recomputed_cross:.12g} m\n"
            f"- Ego speed absolute relative deviation p95: {speed_p95:.6f}\n"
            f"- Constrained marginal target match: {summary['constrained_marginal_target_match']}\n"
            "- Statistical direction/null-decision match: "
            f"{summary['statistical_direction_match']}\n"
            "- Synthetic alpha-90 cluster-bootstrap CI: "
            f"{summary['synthetic_bootstrap_ci95_alpha90']}\n"
            f"- Recorded RQ022 target CI: {summary['target_bootstrap_ci95_alpha90']}\n\n"
            "## Boundary\n\n"
            f"{summary['boundary']} This package remains **SYNTHETIC_NOT_OBSERVED**.\n",
            encoding="utf-8",
        )
    return summary


def math_isclose(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-session", type=Path, default=DEFAULT_SOURCE_SESSION)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    args = parser.parse_args()
    summary = validate_dataset(args.output_dir, args.source_session, args.targets)
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if summary["validation_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
