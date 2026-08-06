#!/usr/bin/env python3
"""Build the RQ021 old/new downstream comparison from persisted JSON evidence."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
WORK = REPO / ".codex-fleet/rq021-contemporaneous-envelope/work/E1"
OUT = WORK / "rq018_rq019_comparison.json"
KEY_NUMBERS = WORK / "key_numbers.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def rq018_comparison() -> dict[str, Any]:
    old_path = WORK / "rq018_old_extended_verification.json"
    new_path = WORK / "rq018_rerun/rq018_supervisor_verification.json"
    old = read_json(old_path)
    new = read_json(new_path)
    quantiles: dict[str, Any] = {}
    for group in ("lower", "inside"):
        quantiles[group] = {}
        for q in ("25", "50", "75"):
            quantiles[group][q] = {
                "old": float(old["ttc_quantiles_by_band"][group]["q"][q]),
                "new": float(new["ttc_quantiles_by_band"][group]["q"][q]),
                "old_n": int(old["ttc_quantiles_by_band"][group]["n"]),
                "new_n": int(new["ttc_quantiles_by_band"][group]["n"]),
            }
    thresholds: dict[str, Any] = {}
    for threshold in (1.0, 1.5, 2.0, 3.0):
        key = f"ttc_lt_{threshold}"
        old_share = old["dangerous_frame_share"][key]
        new_share = new["dangerous_frame_share"][key]
        old_boot = old["dangerous_share_bootstrap"][key]
        new_boot = new["dangerous_share_bootstrap"][key]
        old_diff = float(old_share["lower"]["share"] - old_share["inside"]["share"])
        new_diff = float(new_share["lower"]["share"] - new_share["inside"]["share"])
        thresholds[str(threshold)] = {
            "old": {
                "lower": old_share["lower"],
                "inside": old_share["inside"],
                "difference_lower_minus_inside": old_diff,
                "case_bootstrap_ci95": old_boot["case_bootstrap_ci95"],
                "ci_excludes_zero": bool(old_boot["excludes_zero"]),
                "ci_source_status": "accepted_supervisor_json"
                if threshold in (2.0, 3.0)
                else "rq021_supplement_same_method",
            },
            "new": {
                "lower": new_share["lower"],
                "inside": new_share["inside"],
                "difference_lower_minus_inside": new_diff,
                "case_bootstrap_ci95": new_boot["case_bootstrap_ci95"],
                "ci_excludes_zero": bool(new_boot["excludes_zero"]),
            },
            "direction_changed": direction(old_diff) != direction(new_diff),
        }
    old_center_diff = float(
        old["ttc_quantiles_by_band"]["lower"]["q"]["50"]
        - old["ttc_quantiles_by_band"]["inside"]["q"]["50"]
    )
    new_center_diff = float(
        new["ttc_quantiles_by_band"]["lower"]["q"]["50"]
        - new["ttc_quantiles_by_band"]["inside"]["q"]["50"]
    )
    return {
        "sources": {"old": str(old_path.relative_to(REPO)), "new": str(new_path.relative_to(REPO))},
        "band_counts": {"old": old["band_counts"], "new": new["band_counts"]},
        "future_min_ttc_quantiles": quantiles,
        "dangerous_thresholds": thresholds,
        "conclusion_direction": {
            "center_lower_minus_inside_old": old_center_diff,
            "center_lower_minus_inside_new": new_center_diff,
            "center_direction_changed": direction(old_center_diff) != direction(new_center_diff),
            "all_four_danger_share_directions_unchanged": all(
                not row["direction_changed"] for row in thresholds.values()
            ),
            "ttc_lt_3_ci_exclusion_changed": bool(
                thresholds["3.0"]["old"]["ci_excludes_zero"]
                != thresholds["3.0"]["new"]["ci_excludes_zero"]
            ),
        },
    }


def fixed3_lower_braking(distribution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in distribution["alpha90_case_bootstrap_threshold_contrasts"]
        if row["window"] == "fixed3"
        and row["stratum"] == "non_scripted"
        and row["comparison"] == "lower_minus_inside"
    ]
    return {str(int(row["threshold_mps2"])): row for row in rows}


def rq019_comparison() -> dict[str, Any]:
    old_super_path = (
        REPO
        / "reports/studies/RQ019_counterpart_burden/"
        "RQ019_1_counterpart_burden_20260805T014215Z_7b9f47b/"
        "rq019_supervisor_verification.json"
    )
    new_super_path = WORK / "rq019_rerun/rq019_supervisor_verification.json"
    old_dist_path = old_super_path.with_name("distribution_results.json")
    new_dist_path = WORK / "rq019_rerun/distribution_results.json"
    old_super = read_json(old_super_path)
    new_super = read_json(new_super_path)
    old_dist = read_json(old_dist_path)
    new_dist = read_json(new_dist_path)

    medians: dict[str, Any] = {}
    for field in ("anchor_speed_drop_kmh", "speed_range_kmh", "total_heading_change_deg"):
        old = old_super["median_diff_case_bootstrap"][field]
        new = new_super["median_diff_case_bootstrap"][field]
        medians[field] = {
            "old": old,
            "new": new,
            "difference_direction_changed": direction(float(old["observed_diff"]))
            != direction(float(new["observed_diff"])),
            "ci_exclusion_changed": bool(old["excludes_zero"] != new["excludes_zero"]),
        }

    old_braking = fixed3_lower_braking(old_dist)
    new_braking = fixed3_lower_braking(new_dist)
    braking: dict[str, Any] = {}
    for threshold in ("-2", "-3", "-4"):
        old = old_braking[threshold]
        new = new_braking[threshold]
        braking[threshold] = {
            "old": old,
            "new": new,
            "direction_changed": direction(float(old["share_difference"]))
            != direction(float(new["share_difference"])),
            "case_equal_p_below_0_05_changed": bool(
                (float(old["case_equal_t_p"]) < 0.05)
                != (float(new["case_equal_t_p"]) < 0.05)
            ),
        }
    return {
        "sources": {
            "old_supervisor": str(old_super_path.relative_to(REPO)),
            "new_supervisor": str(new_super_path.relative_to(REPO)),
            "old_distribution": str(old_dist_path.relative_to(REPO)),
            "new_distribution": str(new_dist_path.relative_to(REPO)),
        },
        "band_counts": {"old": old_super["band_counts"], "new": new_super["band_counts"]},
        "non_scripted_band_counts": {
            "old": old_super["non_scripted_band_counts"],
            "new": new_super["non_scripted_band_counts"],
        },
        "median_contrasts": medians,
        "strong_braking_frame_shares": braking,
        "conclusion_direction": {
            "speed_contrasts_same_positive_direction": all(
                not medians[field]["difference_direction_changed"]
                for field in ("anchor_speed_drop_kmh", "speed_range_kmh")
            ),
            "all_three_braking_share_directions_unchanged": all(
                not row["direction_changed"] for row in braking.values()
            ),
            "all_three_new_case_bootstrap_cis_exclude_zero": all(
                not (float(row["new"]["case_bootstrap_ci_95"][0]) <= 0 <= float(row["new"]["case_bootstrap_ci_95"][1]))
                for row in braking.values()
            ),
            "case_equal_p_significance_weakened": any(
                row["case_equal_p_below_0_05_changed"] for row in braking.values()
            ),
        },
    }


def main() -> None:
    timestamp = utc_now()
    comparison = {
        "created_utc": timestamp,
        "rq018": rq018_comparison(),
        "rq019": rq019_comparison(),
    }
    OUT.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    key_numbers = read_json(KEY_NUMBERS)
    onsite_summary_path = WORK / "onsite_scoring_dryrun_summary.json"
    key_numbers["onsite_scoring_dryrun"] = read_json(onsite_summary_path)
    key_numbers["steps_4_5"] = {
        "status": "COMPLETED",
        "completed_utc": timestamp,
        "rq018_output_dir": str((WORK / "rq018_rerun").relative_to(REPO)),
        "rq019_output_dir": str((WORK / "rq019_rerun").relative_to(REPO)),
        "comparison_path": str(OUT.relative_to(REPO)),
        "comparison": comparison,
        "rq019_input_contract_adjustment": {
            "reason": "the copied script hard-coded the old alpha-90 exposure counts and rejected the new input before analysis",
            "old_expected": {"upper": 2700, "lower": 1998, "inside": 9401},
            "new_expected": {"upper": 869, "lower": 519, "inside": 12711},
            "analysis_logic_changed": False,
            "model_threshold_bootstrap_changed": False,
        },
    }
    key_numbers["created_utc"] = timestamp
    KEY_NUMBERS.write_text(json.dumps(key_numbers, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(OUT.relative_to(REPO)), "status": "COMPLETED"}))


if __name__ == "__main__":
    main()
