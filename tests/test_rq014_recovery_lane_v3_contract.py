from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "reports" / "plans" / "RQ014_recovery_lane_v3.json"
LANE_V2 = ROOT / "reports" / "plans" / "RQ014_recovery_lane_v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_collapses_only_the_envelope_axis_to_320_cells_and_960_rows() -> None:
    lane = _load(LANE)
    bank = lane["rating_blind_feature_bank"]
    assert lane["schema_version"] == "rq014-historical-recovery-lane-v3"
    assert "envelope_axis" not in bank
    assert len(bank["sampling_axis"]) == 2
    assert len(bank["temporal_axis"]["recipes"]) == 8
    assert len(bank["horizon_axis"]) == 2
    assert len(bank["readout_axis"]) == 10
    enumeration = bank["predictor_cell_enumeration"]
    assert enumeration["rule"] == "16 feature families x 2 horizons x 10 readouts"
    assert enumeration["registered_predictor_cell_count"] == 320
    cell_ids = {
        f"RR3-{sampling['sampling_id']}-{temporal['temporal_id']}-"
        f"{horizon['horizon_id']}-{readout}"
        for sampling in bank["sampling_axis"]
        for temporal in bank["temporal_axis"]["recipes"]
        for horizon in bank["horizon_axis"]
        for readout in bank["readout_axis"]
    }
    assert len(cell_ids) == 320
    associations = lane["full_data_recovery_screen"]["association_axis"]
    leaderboard_ids = {f"{cell_id}-{row['association_id']}" for cell_id in cell_ids for row in associations}
    assert len(leaderboard_ids) == 960


def test_v3_dependent_row_counts_and_domains_are_consistent() -> None:
    lane = _load(LANE)
    screen = lane["full_data_recovery_screen"]
    assert screen["registered_leaderboard_row_count"] == 960
    sensitivity = screen["association_support_contract"]["common_support_sensitivity"]
    assert sensitivity["result_artifact"]["field_rules"]["blind_cell_count"] == "exact integer 320"
    ledger = screen["append_only_ledger"]
    assert ledger["field_rules"]["row_index"] == "integer 0..959"
    assert ledger["field_rules"]["cell_id"] == "one of the 320 canonical registered predictor cell IDs"
    assert ledger["terminal_digest_artifact"]["row_count"] == 960
    assert ledger["terminal_digest_artifact"]["terminal_record_sha256"].endswith("row_index 959")
    assert "all 960 rows" in ledger["partial_leaderboard_visibility"]
    assert "all 960 unique" in screen["ranking"]["unique_rank_rule"]
    assert "ranks 1..960" in screen["ranking"]["unique_rank_rule"]
    assert "320 predictor cells and 960 association rows" in lane["adaptive_recovery_extension"]["base_ledger_rule"]
    assert "fixed 960 rows" in lane["selected_recipe_freeze"]["creation_prerequisite"]
    assert "zero of 960 rows" in lane["verdict_ladder"]["state_conditions"]["all_predictor_cells_ineligible"]
    text = LANE.read_text(encoding="utf-8")
    assert "2880" not in text
    assert "2879" not in text


def test_v3_uses_one_frozen_m3_envelope_with_extrapolation_semantics() -> None:
    lane = _load(LANE)
    frozen = lane["rating_blind_feature_bank"]["frozen_envelope"]
    assert frozen["kind"] == "RQ009_M3_CONFORMAL"
    assert frozen["scorer_sha256"] == "b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253"
    assert frozen["manifest_sha256"] == "2efbdd0c39edabc419aad815a1eb7529af3623a06c4d3a0b0a99782bcb2f40f4"
    assert frozen["feature_spec_contract_sha256"] == "3ad8ba8ab4c51422a7b2ef208683b7552b68f9e949f0087542ba208065677cce"
    assert frozen["center_output"] == "q_0p5"
    assert frozen["interval_outputs"] == ["lo_90", "hi_90"]
    assert frozen["quantile_consumption"] == (
        "consume q_0p5, lo_90, and hi_90 from the scorer's pre-OOD-mask prediction "
        "result; post-mask NaN outputs are forbidden as G2R scoring inputs"
    )
    assert "diagnostics only" in frozen["support_semantics"]
    assert "0/228" in frozen["wod_transfer_semantics"]


def test_m3_artifact_mismatch_is_a_global_pre_cell_abort() -> None:
    lane = _load(LANE)
    policy = lane["rating_blind_feature_bank"]["m3_artifact_mismatch_policy"]
    assert policy == {
        "status": "M3_ARTIFACT_MISMATCH",
        "action": "GLOBAL_ABORT",
        "must_precede": [
            "input_manifest_processing",
            "materialization_ledger_processing",
            "predictor_cell_processing",
            "rating_access",
        ],
        "predictor_cells_emitted": 0,
        "ledger_rows_emitted": 0,
        "rating_values_read": 0,
    }
    rows = lane["full_data_recovery_screen"]["upstream_terminal_rollup"]["rows"]
    assert all(row["upstream_status"] != "M3_ARTIFACT_MISMATCH" for row in rows)


def test_solver_budget_status_has_one_nonfatal_recovery_rollup() -> None:
    lane = _load(LANE)
    rows = lane["full_data_recovery_screen"]["upstream_terminal_rollup"]["rows"]
    matching = [
        row
        for row in rows
        if row["upstream_status"] == "INELIGIBLE_SOLVER_BUDGET_EXCEEDED"
    ]
    assert matching == [
        {
            "upstream_status": "INELIGIBLE_SOLVER_BUDGET_EXCEEDED",
            "stage": "F",
            "reason_priority": 52,
            "ledger_status": "INELIGIBLE_BLIND",
            "reason_code": "F_SOLVER_BUDGET_EXCEEDED",
        }
    ]


def test_v3_replay_loads_m3_and_rewrites_everything_else() -> None:
    lane = _load(LANE)
    selected = lane["selected_recipe_freeze"]
    replay = lane["clean_independent_replay"]
    assert selected["frozen_m3_binding"]["required"] is True
    assert "exact frozen M3 scorer" in replay["allowed_inputs"][2]
    assert "envelope builder" not in replay["must_reimplement"]
    assert "sole non-reimplemented scientific component" in replay["frozen_envelope_rule"]
    for component in ("resampling", "exact-window state", "deviation", "rating join", "association"):
        assert component in replay["independent_rewrite_boundary"]
    assert "InterHub" not in json.dumps(replay)
    assert "BL90" not in json.dumps(replay)


def test_v2_lane_remains_byte_identical() -> None:
    assert hashlib.sha256(LANE_V2.read_bytes()).hexdigest() == (
        "c1d3a8c4faeb04871e15d7d1d0f07edfd45b8e6904bdd5ac7e05fa3f1f412d7d"
    )
