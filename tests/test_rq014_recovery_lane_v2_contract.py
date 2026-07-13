import csv
import hashlib
import io
import json
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "reports/plans/RQ014_recovery_lane_v2.json"
ENVELOPE_PATH = ROOT / "reports/plans/RQ014_envelope_builder_contract_v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope_producer_statuses(contract: dict) -> set[str]:
    readiness = contract["scientific_input_readiness"]
    path_types = contract["path_type_contract"]
    statuses = {
        "AVAILABLE",
        readiness["schema_profile_gate"]["fail_status"],
        readiness["fixture_gate"]["fail_status"],
        path_types["interhub_mapping"]["missing_status"],
        path_types["interhub_mapping"]["duplicate_or_conflicting_status"],
        path_types["wod_mapping"]["missing_status"],
        path_types["wod_mapping"]["duplicate_or_conflicting_status"],
        path_types["unrecognized_value_status"],
        contract["wod_reference_contract"]["fixture_and_hash_gate"]["failure"],
        contract["episode_and_role_contract"]["unsupported_episode_status"],
        contract["estimator_call_contract"]["nonfinite_or_solver_failure_status"],
        contract["weighted_quantile_contract"]["zero_or_nonfinite_weight_status"],
        next(
            row["unavailable_status"]
            for row in contract["envelope_builders"]
            if row["envelope_id"] == "BL90"
        ),
    }
    statuses.update(
        contract["timeline_and_state_contract"]["failure_status_by_condition"].values()
    )
    statuses.update(contract["envelope_gate"]["failure_status_by_gate"].values())
    return statuses


def test_scientific_input_readiness_is_resolved_by_fail_closed_machine_gates():
    contract = _load(ENVELOPE_PATH)
    readiness = contract["scientific_input_readiness"]

    assert contract["authority"]["rating_values_allowed"] is False
    assert contract["authority"]["rating_derived_fields_allowed"] is False
    assert readiness["schema_profile_gate"]["required_profile"] == (
        "RQ014_SAFE_PRIMITIVE_PROFILE_V2"
    )
    assert readiness["schema_profile_gate"]["pass_status"] == "SATISFIED"
    assert readiness["schema_profile_gate"]["fail_status"] == "INELIGIBLE_BLIND"
    assert readiness["fixture_gate"]["pass_status"] == "SATISFIED"
    assert readiness["fixture_gate"]["fail_status"] == "INELIGIBLE_BLIND"
    assert readiness["candidate_native_velocity_claim"] == "FORBIDDEN"
    assert readiness["missing_policy"].startswith("INELIGIBLE_BLIND")
    assert "1/GO_STRAIGHT, 2/GO_LEFT, 3/GO_RIGHT" in readiness[
        "other_safe_source_evidence"
    ]

    wod_primitives = "\n".join(readiness["required_rating_blind_wod_primitives"])
    for required in (
        "candidate XY",
        "past velocity/acceleration",
        "tstar ego pose",
        "future ego XYZ",
        "intent integer",
        "mapping builder",
        "source_shard_id",
        "counterpart",
    ):
        assert required in wod_primitives
    assert "candidate-specific" not in wod_primitives
    assert "1/GO_STRAIGHT, 2/GO_LEFT, or 3/GO_RIGHT" in wod_primitives

    fixtures = "\n".join(readiness["required_golden_fixtures_before_g2r"])
    for required in ("R04N", "R10L", "CH/LF/HF/TP/TF", "path-type", "pseudo-anchor"):
        assert required in fixtures


def test_recovered_legacy_wod_reference_recipe_is_exact_and_hash_bound():
    contract = _load(ENVELOPE_PATH)
    reference = contract["wod_reference_contract"]
    source = reference["source_provenance"]

    assert reference["builder_id"] == "legacy_route_reference_v1"
    assert reference["candidate_scope"] == (
        "scene-level and byte-identical for all three candidates in a scene"
    )
    assert reference["candidate_specific_reference"] == "forbidden"
    assert source == {
        "absolute_path": (
            "/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/"
            "rq010b_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py"
        ),
        "size_bytes": 42665,
        "sha256": "7c60676effd0b3787cba8825790b6648a3de9d462e571677bf12dbb3920cdba2",
        "functions": ["infer_heading_direction", "build_ego_route_reference"],
        "drift": "STALE_FATAL",
    }

    heading = reference["heading_direction"]
    assert heading["back_index_iteration"] == (
        "range(max(0,n_past-6),n_past-1) in increasing index order"
    )
    assert heading["acceptance"].startswith("first delta")
    assert heading["no_accepted_delta_fallback"] == [1.0, 0.0]
    assert reference["past_reference"] == {
        "keep_every": "max(1,n_past//8)",
        "initial": "past_xy[::keep_every]",
        "ensure_p0": (
            "append p0 with numpy.vstack iff numpy.linalg.norm(past_ref[-1]-p0)>1e-9"
        ),
    }
    assert reference["intent_to_turn_sign"] == {
        "normalization": "str(scene.get('intent_name','GO_STRAIGHT')).upper()",
        "contains_LEFT": 1.0,
        "else_contains_RIGHT": -1.0,
        "else": 0.0,
    }
    assert reference["constants"] == {
        "turn_radius_m": 12.0,
        "extension_m": 80.0,
        "step_m": 1.0,
    }
    assert reference["straight_branch"]["s_values"] == (
        "numpy.arange(1.0,80.0+1.0,1.0)"
    )
    turn = reference["turn_branch"]
    assert turn["arc_steps"] == "max(8,ceil((12.0*pi/2)/1.0))"
    assert turn["phi"] == "numpy.linspace(1.0/12.0,pi/2,arc_steps)"
    assert turn["tail_s_values"] == (
        "numpy.arange(1.0,tail_len+1.0,1.0) when tail_len>0"
    )

    counterpart = reference["counterpart_reference"]
    assert "observed counterpart_state[:,0:2]" in counterpart["builder"]
    assert counterpart["extrapolation"] == "forbidden"
    gate = reference["fixture_and_hash_gate"]
    assert gate["required_fixture_ids"] == [
        "LEGACY_REF_STRAIGHT",
        "LEGACY_REF_LEFT",
        "LEGACY_REF_RIGHT",
        "LEGACY_REF_HEADING_FALLBACK",
        "LEGACY_REF_PAST_ENDPOINT_APPEND",
        "LEGACY_REF_COUNTERPART_OBSERVED",
    ]
    assert gate["pass"] == "REFERENCE_BUILDER_SATISFIED"
    assert gate["failure"] == "INELIGIBLE_BLIND"


def test_sampling_seam_and_state_derivation_are_single_valued():
    contract = _load(ENVELOPE_PATH)
    timeline = contract["timeline_and_state_contract"]
    samplings = {row["sampling_id"]: row for row in timeline["sampling_definitions"]}

    assert set(samplings) == {"R04N", "R10L"}
    assert samplings["R04N"]["dt_ns"] == 250_000_000
    assert samplings["R10L"]["dt_ns"] == 100_000_000
    assert "tstar" in samplings["R04N"]["wod_grid_phase"]
    assert "tstar" in samplings["R10L"]["wod_grid_phase"]
    assert "t=0 ego-history" in samplings["R10L"]["seam_interpolation"]

    assert timeline["wod_tstar"]["time_s"] == 0.0
    assert "candidate t=0 state is forbidden" in timeline["wod_tstar"]["seam"]
    derived = timeline["derived_state_rule"]
    assert "slice that interval's resampled position rows first" in derived["stage"]
    assert "window_boundary_rederivation" not in derived
    assert derived["derivative_halo"].startswith("FORBIDDEN")
    assert derived["cross_window_state_reuse"].startswith("FORBIDDEN")
    assert "(p[k+1]-p[k-1])/(2*dt)" == derived["velocity_interior_tick"]
    assert "same sliced window" in derived["acceleration"]
    assert "in that window" in derived["heading"]
    assert "source candidate velocity/acceleration must not be used" in timeline[
        "position_authority"
    ]
    gaps = timeline["maximum_source_gap"]
    assert gaps["R10L_wod_candidate_gap_above_s"] == 0.25
    assert "explicit exception" in gaps["R10L_wod_candidate"]
    assert gaps["interhub"] == "default 2*dt rule"
    assert timeline["failure_status_by_condition"] == {
        "no_complete_common_support_or_extrapolation_required": "INELIGIBLE_TIMELINE_SUPPORT",
        "source_gap_exceeds_registered_sampling_rule": "INELIGIBLE_TIMELINE_SOURCE_GAP",
        "timestamp_off_registered_grid_or_phase": "INELIGIBLE_TIMELINE_GRID_PHASE",
        "wod_tstar_seam_contract_failed": "INELIGIBLE_TIMELINE_SEAM",
        "position_or_derived_state_nonfinite": "INELIGIBLE_STATE_NONFINITE",
        "all_positions_stationary_heading_undefined": "INELIGIBLE_UNDEFINED_HEADING",
    }


def test_temporal_operators_cover_every_registered_boundary():
    contract = _load(ENVELOPE_PATH)
    recipes = {row["temporal_id"]: row for row in contract["temporal_recipes"]}
    expected_ids = [
        "CH-W10",
        "CH-W25",
        "LF-W10",
        "LF-W25",
        "HF-W10",
        "HF-W25",
        "TP",
        "TF",
    ]
    assert list(recipes) == expected_ids

    fixed_expected_points = {
        "R04N": {
            "CH-W10": 5,
            "CH-W25": 11,
            "LF-W10": 5,
            "LF-W25": 11,
            "HF-W10": 9,
            "HF-W25": 21,
        },
        "R10L": {
            "CH-W10": 11,
            "CH-W25": 26,
            "LF-W10": 11,
            "LF-W25": 26,
            "HF-W10": 21,
            "HF-W25": 51,
        },
    }
    rates = {"R04N": 4, "R10L": 10}
    for sampling_id, expected_by_recipe in fixed_expected_points.items():
        for temporal_id, expected_points in expected_by_recipe.items():
            operator = recipes[temporal_id]["window_operator"]
            assert operator["lower_anchor"] == "TAU"
            assert operator["upper_anchor"] == "TAU"
            assert operator["closed_endpoints"] is True
            span_ticks = round(
                (operator["upper_offset_s"] - operator["lower_offset_s"])
                * rates[sampling_id]
            )
            assert span_ticks + 1 == expected_points

    assert recipes["TP"]["window_operator"]["lower_anchor"] == "TSTAR"
    assert recipes["TP"]["window_operator"]["upper_anchor"] == "TAU"
    assert recipes["TF"]["window_operator"]["lower_anchor"] == "TSTAR"
    assert recipes["TF"]["window_operator"]["upper_anchor"] == "H_COMMON"
    assert recipes["TF"]["span_match_key"] == "h_common_tick"


def test_sixteen_feature_families_and_48_envelope_executions_are_unique():
    contract = _load(ENVELOPE_PATH)
    sampling_ids = [
        row["sampling_id"]
        for row in contract["timeline_and_state_contract"]["sampling_definitions"]
    ]
    temporal_ids = [row["temporal_id"] for row in contract["temporal_recipes"]]
    families = contract["feature_families"]

    expected_families = [
        {
            "feature_id": f"F-{sampling_id}-{temporal_id}",
            "sampling_id": sampling_id,
            "temporal_id": temporal_id,
        }
        for sampling_id, temporal_id in product(sampling_ids, temporal_ids)
    ]
    assert families == expected_families
    assert len({row["feature_id"] for row in families}) == 16

    envelope_ids = [row["envelope_id"] for row in contract["envelope_builders"]]
    execution_ids = {
        f"ENV2-{family['feature_id']}-{envelope_id}"
        for family, envelope_id in product(families, envelope_ids)
    }
    enumeration = contract["execution_enumeration"]
    assert envelope_ids == enumeration["envelope_order"] == ["BL90", "BM90", "BT90"]
    assert len(execution_ids) == enumeration["feature_envelope_execution_count"] == 48
    assert enumeration["feature_family_count"] == 16


def test_interhub_roles_path_types_and_envelope_cells_are_frozen():
    contract = _load(ENVELOPE_PATH)
    path_types = contract["path_type_contract"]
    roles = contract["episode_and_role_contract"]
    builders = {row["envelope_id"]: row for row in contract["envelope_builders"]}

    assert path_types["canonical_values_in_order"] == ["CP", "HO", "MP", "F"]
    assert "exact one-to-one lookup only" in path_types["interhub_mapping"]["rule"]
    assert "exact one-to-one lookup only" in path_types["wod_mapping"]["rule"]
    assert "materialized rating-blind before G2R" in path_types["wod_mapping"]["source"]
    builder_binding = path_types["wod_mapping"]["builder_binding"]
    assert builder_binding["registry_pointer"].endswith(
        "RQ014_config_space_v1p6.yaml#/envelope/wod_path_type_mapping"
    )
    for artifact in ("source_definition", "implementation", "mapping_table"):
        assert len(builder_binding[artifact]["sha256"]) == 64
    assert [row["role_id"] for row in roles["directed_roles_in_order"]] == [
        "A_AS_FOCAL",
        "B_AS_FOCAL",
    ]
    assert roles["directed_role_estimator_tuple_order"] == [
        "focal_state",
        "focal_raw_reference",
        "counterpart_state",
        "counterpart_raw_reference",
    ]
    assert roles["role_requirement"].startswith("both directed roles")

    assert "horizon_id" in builders["BM90"]["lookup_key"]
    assert "tau_tick" in builders["BT90"]["lookup_key"]
    assert builders["BT90"]["horizon_pooling"].startswith("none")
    assert "never pool different h_common_tick" in builders["BM90"]["pooling_domain"][
        "TF"
    ]
    assert "never pool different h_common_tick" in builders["BT90"]["TF"]

    quantile = contract["weighted_quantile_contract"]
    assert quantile["quantiles"] == [0.05, 0.5, 0.95]
    assert "left-continuous" in quantile["rule"]
    assert "episode has total weight 1/E" in quantile["episode_weight"]
    gate = contract["envelope_gate"]
    assert gate["minimum_independent_episodes"] == 50
    assert gate["minimum_distinct_ipv_values"] == 3
    assert gate["bootstrap"]["replicates"] == 1000
    assert gate["bootstrap"]["maximum_relative_se_each_half_width"] == 0.25
    assert gate["failure_status_by_gate"] == {
        "safe_primitive_or_fixture_profile_failed": "INELIGIBLE_BLIND",
        "fewer_than_50_independent_episodes": "INELIGIBLE_ENVELOPE_LOW_EPISODE_COUNT",
        "fewer_than_3_distinct_ipv_values": "INELIGIBLE_ENVELOPE_DEGENERATE",
        "quantiles_not_strictly_ordered": "INELIGIBLE_ENVELOPE_DEGENERATE",
        "either_half_width_below_1e-6_rad": "INELIGIBLE_ENVELOPE_DEGENERATE",
        "not_exactly_1000_successful_bootstrap_replicates": "INELIGIBLE_ENVELOPE_UNCERTAIN",
        "either_half_width_relative_se_above_0p25": "INELIGIBLE_ENVELOPE_UNCERTAIN",
        "missing_required_lookup_row": "INELIGIBLE_ENVELOPE_REQUIRED_CELL",
    }
    assert gate["no_pseudocount_or_ridge"] is True
    assert "retain its granular terminal status" in gate["required_query_failure"]
    assert "not a wrapper that erases" in gate["required_query_failure"]


def test_recovery_lane_resolves_960_predictors_through_the_builder_contract():
    lane = _load(RECOVERY_PATH)
    bank = lane["rating_blind_feature_bank"]
    binding = bank["envelope_builder_contract"]
    assert binding["path"] == "reports/plans/RQ014_envelope_builder_contract_v2.json"
    assert binding["schema_version"] == "rq014-envelope-builder-contract-v2"
    assert "both SATISFIED permits compute" in binding["scientific_readiness_resolution"]
    assert "INELIGIBLE_BLIND" in binding["scientific_readiness_resolution"]

    contract = _load(ENVELOPE_PATH)
    feature_ids = [row["feature_id"] for row in contract["feature_families"]]
    envelope_ids = [row["envelope_id"] for row in bank["envelope_axis"]]
    horizon_ids = [row["horizon_id"] for row in bank["horizon_axis"]]
    readout_ids = bank["readout_axis"]
    predictor_ids = {
        f"RR2-{feature_id.removeprefix('F-')}-{envelope_id}-{horizon_id}-{readout_id}"
        for feature_id, envelope_id, horizon_id, readout_id in product(
            feature_ids, envelope_ids, horizon_ids, readout_ids
        )
    }
    assert len(predictor_ids) == 16 * 3 * 2 * 10 == 960
    assert bank["predictor_cell_enumeration"]["registered_predictor_cell_count"] == 960


def test_association_methods_share_one_exact_mask_and_weights():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    support = screen["association_support_contract"]
    associations = {row["association_id"]: row for row in screen["association_axis"]}

    assert list(associations) == ["RWS", "PSP", "PPR"]
    assert "exactly the same included scenes" in support["shared_across_associations"]
    assert support["scene_weight"].startswith(
        "when n_informative_scenes>0, 1/n_informative_scenes"
    )
    assert support["candidate_weight"].startswith(
        "when n_informative_scenes>0, 1/(3*n_informative_scenes)"
    )
    assert "no candidate weights are instantiated when it is 0" in support[
        "candidate_weight"
    ]
    assert "average midranks" in support["tie_rule"]
    assert "no tolerance" in support["tie_rule"]
    assert support["between_cell_support"].startswith(
        "may differ across predictor cells only because"
    )
    common = support["common_support_sensitivity"]
    assert "non-ranking, non-gating" in common["role"]
    assert common["cell_universe"].startswith("all 960 registered predictor cell_ids")
    assert common["blind_artifact"]["exact_columns"] == ["segment_id"]
    assert common["result_artifact"]["row_order"] == ["RWS", "PSP", "PPR"]

    denominators = set(support["required_denominator_fields"])
    assert {
        "n_informative_scenes",
        "n_informative_candidates",
        "scene_weight_sum",
        "candidate_weight_sum",
        "support_id",
    } <= denominators
    assert associations["RWS"]["denominator"] == "n_informative_scenes"
    assert associations["PSP"]["denominator"] == "n_informative_candidates"
    assert associations["PPR"]["denominator"] == "n_informative_candidates"


def test_stability_and_ranking_form_a_total_order_over_2880_rows():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    stability = screen["stability"]
    ranking = screen["ranking"]

    assert stability["fold_min_informative_scenes"] == 5
    assert "all five folds" in stability["fold_validity"]
    assert stability["leave_one_scene_out_denominator"] == "n_informative_scenes"
    assert stability["eligible_cluster_min_informative_scenes"] == 5
    assert stability["leave_one_cluster_denominator"].startswith(
        "eligible_cluster_count_metric.value"
    )
    assert "source_shard_id" in stability["source_shard_leave_one_out"]
    assert "same association_support_contract mask" in stability["support_reuse_rule"]
    assert "exact scenario_cluster=NA" in stability["leave_one_scenario_cluster_out"]
    assert stability["leave_one_cluster_denominator"] == (
        "eligible_cluster_count_metric.value; it is recorded even when 0 or 1"
    )
    assert stability["leave_one_shard_denominator"] == (
        "eligible_shard_count_metric.value; it is recorded even when 0 or 1"
    )
    for field in (
        "scenario_cluster_na_scene_count_metric",
        "eligible_cluster_count_metric",
        "negative_cluster_count_metric",
        "eligible_shard_count_metric",
        "negative_shard_count_metric",
    ):
        assert field in stability["row_metric_fields"]

    assert ranking["lexicographic_key"][-1] == "leaderboard_id UTF-8 ascending"
    assert "total-order" in ranking["unique_rank_rule"]
    assert ranking["top_recipe_count"] == 1
    assert screen["same_recipe_robustness_requirement"]["historical_metric_may_select"] == [
        "RWS",
        "PSP",
        "PPR",
    ]

    contract = _load(ENVELOPE_PATH)
    feature_ids = [row["feature_id"] for row in contract["feature_families"]]
    envelope_ids = contract["execution_enumeration"]["envelope_order"]
    horizon_ids = ["H20", "HFEAS"]
    readout_ids = lane["rating_blind_feature_bank"]["readout_axis"]
    association_ids = [row["association_id"] for row in screen["association_axis"]]
    leaderboard_ids = {
        "-".join(
            [
                "RR2",
                feature_id.removeprefix("F-"),
                envelope_id,
                horizon_id,
                readout_id,
                association_id,
            ]
        )
        for feature_id, envelope_id, horizon_id, readout_id, association_id in product(
            feature_ids,
            envelope_ids,
            horizon_ids,
            readout_ids,
            association_ids,
        )
    }
    assert len(leaderboard_ids) == 16 * 3 * 2 * 10 * 3 == 2880
    assert screen["registered_leaderboard_row_count"] == 2880


def test_every_upstream_terminal_has_one_unique_ledger_rollup():
    lane = _load(RECOVERY_PATH)
    contract = _load(ENVELOPE_PATH)
    screen = lane["full_data_recovery_screen"]
    rollup = screen["upstream_terminal_rollup"]
    rows = rollup["rows"]

    by_status = {}
    for row in rows:
        by_status.setdefault(row["upstream_status"], []).append(row)
    assert all(len(matches) == 1 for matches in by_status.values())
    envelope_statuses = set(contract["terminal_statuses"])
    assert len(contract["terminal_statuses"]) == len(envelope_statuses)
    assert envelope_statuses == _envelope_producer_statuses(contract)
    expected_internal = {
        "RATING_JOIN_KEY_MISSING",
        "RATING_JOIN_KEY_AMBIGUOUS",
        "RATING_VALUE_MISSING",
        "RATING_VALUE_NONFINITE",
        "RATING_VECTOR_CONSTANT",
        "DEVIATION_VECTOR_CONSTANT",
        "INFORMATIVE_INVARIANT_FAILURE",
        "ASSOCIATION_CONSTANT",
        "ASSOCIATION_NUMERICAL_FAILURE",
        "ASSOCIATION_AVAILABLE",
    }
    assert set(rollup["recovery_internal_upstream_statuses"]) == expected_internal
    assert set(by_status) == envelope_statuses | expected_internal
    assert "subset-only validation is forbidden" in rollup["completeness_identity"]

    status_reason_pairs = [(row["ledger_status"], row["reason_code"]) for row in rows]
    assert len(status_reason_pairs) == len(set(status_reason_pairs))
    assert len({row["reason_code"] for row in rows}) == len(rows)
    allowed = set(screen["append_only_ledger"]["allowed_terminal_statuses"])
    assert {row["ledger_status"] for row in rows} <= allowed | {"CONTINUE"}
    assert by_status["AVAILABLE"][0]["ledger_status"] == "CONTINUE"
    assert by_status["ASSOCIATION_AVAILABLE"][0]["ledger_status"] == "OBSERVED"
    kernel = screen["statistic_kernel_contract"]
    assert kernel["terminal_status_namespace"] == {
        "finite": "ASSOCIATION_AVAILABLE",
        "constant": "ASSOCIATION_CONSTANT",
        "nonfinite_or_kernel_gate_failure": "ASSOCIATION_NUMERICAL_FAILURE",
        "normalization_forbidden": (
            "INELIGIBLE_ASSOCIATION_CONSTANT and NUMERICAL_FAILURE are "
            "ledger-status labels only and must never be returned as upstream "
            "association statuses"
        ),
    }
    assert kernel["midrank"]["nonfinite"] == "ASSOCIATION_NUMERICAL_FAILURE"
    assert kernel["weighted_pearson"]["constant"].endswith(
        "return ASSOCIATION_CONSTANT"
    )
    assert kernel["implementation_and_environment_gate"]["failure"] == (
        "ASSOCIATION_NUMERICAL_FAILURE"
    )
    assert by_status["INELIGIBLE_TIMELINE_SOURCE_GAP"][0] == {
        "upstream_status": "INELIGIBLE_TIMELINE_SOURCE_GAP",
        "stage": "F",
        "reason_priority": 41,
        "ledger_status": "INELIGIBLE_BLIND",
        "reason_code": "F_TIMELINE_SOURCE_GAP",
    }
    assert rollup["ledger_status_rank"] == {
        "OBSERVED": 0,
        "INELIGIBLE_BLIND": 1,
        "INELIGIBLE_REFERENCE": 2,
        "INELIGIBLE_RATING_COMPLETENESS": 3,
        "INELIGIBLE_ASSOCIATION_CONSTANT": 4,
        "NUMERICAL_FAILURE": 5,
    }
    assert rollup["row_outcome_algorithm"][-1] == (
        "exactly one terminal ledger row is emitted for every leaderboard_id"
    )


def test_observed_rows_with_stability_na_still_have_a_typed_total_order():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    stability = screen["stability"]
    schema = stability["typed_metric_schema"]
    ranking = screen["ranking"]

    assert schema["FINITE_FLOAT"]["exact_keys"] == ["kind", "value"]
    assert schema["FINITE_INT"]["exact_keys"] == ["kind", "value"]
    assert schema["NA"]["exact_keys"] == ["kind", "reason_code"]
    assert {"JSON null", "NaN", "Infinity", "bare string NA", "numeric sentinel"} <= set(
        schema["forbidden"]
    )
    assert "ledger_status remains OBSERVED" in stability["observed_with_stability_na"]
    assert "recovery_compatible is false" in stability["observed_with_stability_na"]

    projection = ranking["typed_sort_projection"]
    assert projection["association_finite"].startswith(
        "[0,association_value.value]"
    )
    assert projection["association_NA"] == "[1,0.0]"
    assert projection["fold_negative_count_NA"] == "[1,0]"
    assert projection["cluster_fraction_NA"] == "[1,0.0]"
    assert projection["n_informative_scenes_NA"] == "[1,0]"

    assert "n_informative_scenes" not in stability["row_metric_fields"]
    assert "n_informative_scenes_metric" in stability["row_metric_fields"]
    assert "separate raw n_informative_scenes" in stability["row_metric_fields"][
        "n_informative_scenes_metric"
    ]
    typed_association = {"kind": "FINITE_FLOAT", "value": -0.4}
    typed_fold_na = {"kind": "NA", "reason_code": "INSUFFICIENT_FOLD_SUPPORT"}
    typed_cluster_na = {
        "kind": "NA",
        "reason_code": "INSUFFICIENT_CLUSTER_SUPPORT",
    }
    typed_n = {"kind": "FINITE_INT", "value": 50}
    assert typed_association["value"] == -0.4
    assert typed_fold_na["kind"] == typed_cluster_na["kind"] == "NA"
    assert typed_n["value"] == 50

    observed_with_na = (0, 1, 0, -0.4, 1, 0, 1, 0.0, 0, -50, b"row-a")
    nonobserved = (1, 1, 1, 0.0, 1, 0, 1, 0.0, 1, 0, b"row-b")
    observed_tie_b = (*observed_with_na[:-1], b"row-b")
    assert observed_with_na < nonobserved
    assert observed_with_na < observed_tie_b
    assert ranking["lexicographic_key"][-1] == "leaderboard_id UTF-8 ascending"
    uncertainty = screen["association_uncertainty"]
    assert uncertainty["canonical_seed_object"]["replicates"] == 2000
    assert "terminal LF" in uncertainty["canonical_seed_object"]["serialization"]
    assert "replicate_index 0..1999" in uncertainty["draw_rule"]
    assert "no discarded or parallel draws" in uncertainty["draw_rule"]
    assert "draw_position" in uncertainty["occurrence_identity"]
    assert "never collapse or re-sort" in uncertainty["occurrence_identity"]


def test_history_only_control_cannot_receive_or_leak_candidate_future():
    lane = _load(RECOVERY_PATH)
    control = lane["rating_blind_feature_bank"]["hard_negative_control"]

    assert control["registered_executions"] == [
        "R04N-W10",
        "R04N-W25",
        "R10L-W10",
        "R10L-W25",
    ]
    for forbidden in ("candidate ID", "candidate geometry", "candidate future", "ego future"):
        assert forbidden in control["branch_isolation"]
    assert "before candidate forking" in control["branch_isolation"]
    assert "for each registered W separately slice" in control["state_derivation"]
    assert "derivative halos" in control["state_derivation"]
    assert "reuse of state from a larger history" in control["state_derivation"]
    assert "three distinct finite adversarial trajectories" in control[
        "future_perturbation_fixture"
    ]
    assert "payload SHA-256" in control["future_perturbation_fixture"]
    assert "implementation SHA-256" in control["fixture_and_hash_gate"]
    assert "environment manifest SHA-256" in control["fixture_and_hash_gate"]
    assert control["failure"] == "FATAL_CANDIDATE_ID_OR_FUTURE_LEAKAGE"


def test_interhub_role_swap_fixes_focal_and_counterpart_state_reference_semantics():
    contract = _load(ENVELOPE_PATH)
    roles = contract["episode_and_role_contract"]
    role_a, role_b = roles["directed_roles_in_order"]

    assert role_a["focal"].endswith("first")
    assert role_a["counterpart"].endswith("second")
    assert "first participant's resampled" in role_a["focal_state"]
    assert "first participant's checksum-bound route/lane" in role_a[
        "focal_raw_reference"
    ]
    assert "second participant's observed resampled window XY" == role_a[
        "counterpart_raw_reference"
    ]
    assert "second participant's resampled" in role_a["counterpart_state"]
    assert role_a["estimator_input_tuple"] == [
        "first_participant_state",
        "first_participant_route_reference",
        "second_participant_state",
        "second_participant_observed_window_xy",
    ]
    assert role_b["focal"].endswith("second")
    assert role_b["counterpart"].endswith("first")
    assert "second participant's checksum-bound route/lane" in role_b[
        "focal_raw_reference"
    ]
    assert "second participant's resampled" in role_b["focal_state"]
    assert "first participant's resampled" in role_b["counterpart_state"]
    assert "first participant's observed resampled window XY" == role_b[
        "counterpart_raw_reference"
    ]
    assert role_b["estimator_input_tuple"] == [
        "second_participant_state",
        "second_participant_route_reference",
        "first_participant_state",
        "first_participant_observed_window_xy",
    ]
    assert role_a["reported_output"] == role_b["reported_output"] == "focal IPV only"
    assert "exactly once" in roles["reference_preparation"]
    assert roles["counterpart_output"].endswith("never enters the human envelope")


def test_attrition_dag_support_bytes_and_statistic_kernel_are_frozen():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    attrition = screen["ordered_attrition_contract"]
    support = screen["association_support_contract"]
    kernel = screen["statistic_kernel_contract"]

    assert attrition["stage_order"] == ["B", "K", "F", "R", "D", "I"]
    assert attrition["candidate_order_within_scene"].startswith(
        "candidate_ordinal integer ascending exactly 1,2,3"
    )
    assert "exactly one mutually exclusive first-failure bucket" in attrition[
        "first_failure_assignment"
    ]
    assert attrition["count_identities"] == [
        "N_B=479",
        "N_B=N_K+X_K",
        "N_K=N_F+X_F",
        "N_F=N_R+X_R",
        "N_R=N_D+X_D",
        "N_D=N_I+X_I",
        "N_B=N_I+X_K+X_F+X_R+X_D+X_I",
        "candidate_slot_count_at_each_N_stage=3*N_stage",
    ]
    assert attrition["field_aliases"]["N_I"] == "n_informative_scenes"
    assert set(attrition["count_fields"]) <= set(support["required_denominator_fields"])
    field_types = support["denominator_field_types"]
    assert set(attrition["count_fields"]) <= set(field_types["raw_integer_fields"])
    assert "n_informative_scenes" in field_types["raw_integer_fields"]
    assert "typed leaderboard metric objects are separate" in field_types[
        "typed_metric_namespace"
    ]
    assert "typed leaderboard metric objects use distinct" in attrition[
        "field_namespace"
    ]
    assert attrition["violation"] == "FATAL_ATTRITION_ACCOUNTING_ERROR"
    attrition_gate = attrition["implementation_and_environment_gate"]
    assert attrition_gate["pass"] == "ATTRITION_KERNEL_SATISFIED"
    assert attrition_gate["failure"] == "FATAL_ATTRITION_ACCOUNTING_ERROR"
    assert len(attrition_gate["required_fixture_ids"]) == 4
    fatal = attrition["cell_fatal_count_policy"]
    assert "B and the geometry/rating-join-key-only K audit" in fatal["B_and_K"]
    assert "set N_F=0" in fatal["F_injection"]
    assert "N_R=N_D=N_I=X_R=X_D=X_I=0" in fatal["downstream_counts"]
    assert "both raw weight sums=0.0" in fatal["derived_zero_support"]
    assert "no typed raw-count sentinel" in fatal["row_status"]

    empty_support_id = hashlib.sha256(b"").hexdigest()
    assert support["derived_denominator_definitions"]["support_id"].endswith(
        empty_support_id
    )

    ids = ["场景-2", "scene-1"]
    payload = "".join(f"{item}\n" for item in sorted(ids, key=lambda value: value.encode()))
    assert payload.endswith("\n") and not payload.endswith("\n\n")
    support_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert len(support_id) == 64
    assert "including the final ID" in support["within_cell_support_id"]

    midrank = kernel["midrank"]
    assert "one-based occupied rank positions lo..hi" in midrank["algorithm"]
    assert "(lo+hi)/2.0" in midrank["algorithm"]
    assert "exact equality only" in midrank["ties"]
    pearson = kernel["weighted_pearson"]
    assert pearson["sum_weights"].startswith("sw=math.fsum")
    assert "math.fsum" in pearson["means"]
    assert "math.fsum" in pearson["centered_sums"]
    assert pearson["constant"].endswith("ASSOCIATION_CONSTANT")
    assert set(kernel["method_reductions"]) == {"RWS", "PSP", "PPR"}
    assert "bootstrap draw_position occurrence order" in kernel["method_reductions"][
        "RWS"
    ]
    gate = kernel["implementation_and_environment_gate"]
    assert gate["pass"] == "STATISTIC_KERNEL_SATISFIED"
    assert gate["failure"] == "ASSOCIATION_NUMERICAL_FAILURE"
    assert len(gate["required_fixture_ids"]) == 6
    input_order = kernel["input_identity_and_order"]
    assert input_order["ordinary_and_deletion_subsets"].startswith(
        "row identity is (segment_id,candidate_ordinal)"
    )
    assert "replicate_index,draw_position,source_segment_id,candidate_ordinal" in (
        input_order["bootstrap_occurrences"]
    )
    assert "never collapsed" in input_order["bootstrap_occurrences"]
    assert "preserving duplicate occurrences" in kernel["subset_recomputation"]


def test_envelope_bootstrap_has_canonical_bytes_draw_order_and_fixtures():
    contract = _load(ENVELOPE_PATH)
    timeline = contract["timeline_and_state_contract"]
    gate = contract["envelope_gate"]
    canonical = gate["bootstrap_canonical_lookup"]
    bootstrap = gate["bootstrap"]

    lookup = {
        "envelope_id": "BM90",
        "feature_id": "F-R04N-CH-W10",
        "h_common_tick_or_NA": "NA",
        "horizon_id_or_NA": "H20",
        "path_type": "CP",
        "tau_tick_or_NA": "NA",
    }
    canonical_text = (
        json.dumps(
            lookup,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    expected = (
        '{"envelope_id":"BM90","feature_id":"F-R04N-CH-W10",'
        '"h_common_tick_or_NA":"NA","horizon_id_or_NA":"H20",'
        '"path_type":"CP","tau_tick_or_NA":"NA"}\n'
    )
    assert canonical_text == expected
    canonical_bytes = canonical_text.encode("utf-8")
    assert canonical_bytes.endswith(b"\n") and not canonical_bytes.startswith(b"\xef\xbb\xbf")
    seed_material = b"RQ014-ENVBOOT|" + canonical_bytes
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big", signed=False)
    assert 0 <= seed < 2**64

    assert canonical["NA_encoding"].endswith("JSON null is forbidden")
    assert "JSON integer" in canonical["tick_encoding"]
    assert canonical["bytes"].endswith("terminal byte 0x0A")
    assert len(canonical["lookup_fixture_ids"]) == 4
    assert "raw UTF-8 byte order" in bootstrap["episode_source_order"]
    assert bootstrap["rng_constructor"] == (
        "numpy.random.Generator(numpy.random.PCG64(seed))"
    )
    assert "replicate_index 0..999" in bootstrap["draw_rule"]
    assert "exactly once" in bootstrap["draw_rule"]
    assert "no discarded" in bootstrap["draw_rule"]
    assert "draw_position" in bootstrap["draw_order"]
    assert bootstrap["occurrence_identity_schema"]["exact_keys"] == [
        "replicate_index",
        "draw_position",
        "source_episode_id",
    ]
    assert "never collapsed or re-sorted" in bootstrap["occurrence_order_rule"]
    assert "(1/E)/K_e_j" in bootstrap["occurrence_observation_weight"]
    assert "math.fsum" in bootstrap["weight_normalization_order"]
    assert "exactly the 1000 registered RNG calls" in bootstrap[
        "fixed_draw_count_and_retry_policy"
    ]
    assert "fewer than 3 distinct" in bootstrap["finite_degenerate_replicates"]
    assert "valid successful finite width observations" in bootstrap[
        "finite_degenerate_replicates"
    ]
    assert "only replicate failures" in bootstrap["replicate_failure_predicate"]
    assert bootstrap["fixture_ids"] == [
        "ENVBOOT_PCG64_DRAWS",
        "ENVBOOT_WEIGHTED_QUANTILES",
        "ENVBOOT_RELATIVE_SE",
    ]
    assert "-pi is normalized to +pi" in timeline["derived_state_rule"][
        "angle_storage"
    ]


def test_stat2_m01_support_loco_shard_and_common_support_are_single_valued():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    support = screen["association_support_contract"]
    stability = screen["stability"]

    between = support["between_cell_support"]
    for mechanism in (
        "not rating-blind AVAILABLE",
        "deviations is nonfinite",
        "three-deviation vector is constant",
    ):
        assert mechanism in between
    assert "cell-invariant" in between
    assert "NA is not a scenario cluster" in support["scenario_cluster_missing_rule"]

    common = support["common_support_sensitivity"]
    assert "all 960" in common["cell_universe"]
    assert "no globally or scene-locally unavailable cell is skipped" in common[
        "cell_universe"
    ]
    blind = common["blind_artifact"]
    assert blind["exact_columns"] == ["segment_id"]
    assert "CPython 3.9 csv.writer" in blind["encoding"]
    assert blind["empty_file_rule"] == "header plus LF only"
    result = common["result_artifact"]
    assert result["exact_top_keys"] == [
        "schema_version",
        "status",
        "blind_cell_count",
        "blind_scene_count",
        "rated_informative_scene_count",
        "support_id",
        "rows",
    ]
    assert result["row_exact_keys"] == [
        "association_id",
        "status",
        "association_value",
        "n_informative_scenes_metric",
    ]
    assert set(common["terminal_policy"]) == {
        "COMMON_SUPPORT_AVAILABLE",
        "COMMON_SUPPORT_EMPTY_BLIND",
        "COMMON_SUPPORT_EMPTY_AFTER_RATING",
        "COMMON_SUPPORT_ASSOCIATION_CONSTANT",
        "COMMON_SUPPORT_NUMERICAL_FAILURE",
        "row_status_rule",
    }
    assert "never changes recovery_compatible" in common["role"]

    metrics = stability["row_metric_fields"]
    for field in (
        "scenario_cluster_na_scene_count_metric",
        "eligible_cluster_count_metric",
        "negative_cluster_count_metric",
        "leave_one_cluster_out_negative_fraction",
        "eligible_shard_count_metric",
        "negative_shard_count_metric",
        "leave_one_shard_out_negative_fraction",
    ):
        assert field in metrics
    assert "distinct non-NA scenario_cluster" in stability[
        "eligible_cluster_definition"
    ]
    assert "negative_cluster_count_metric.value/eligible_cluster_count_metric.value" in (
        stability["leave_one_cluster_negative_definition"]
    )
    assert "negative_shard_count_metric.value/eligible_shard_count_metric.value" in (
        stability["leave_one_shard_negative_definition"]
    )
    requirements = screen["recovery_compatible_marker"]["requirements"]
    assert "eligible_cluster_count_metric.kind_equals_FINITE_INT_and_value_at_least_2" in (
        requirements
    )
    assert any(item.startswith("negative_cluster_count_metric.kind_equals") for item in requirements)


def test_stat2_m02_fixed_2880_state_machine_never_freezes_an_incompatible_row():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    adaptive = lane["adaptive_recovery_extension"]
    freeze = lane["selected_recipe_freeze"]
    verdict = lane["verdict_ladder"]

    assert screen["registered_leaderboard_row_count"] == 2880
    assert adaptive == {
        "status": "DENY_REQUIRES_NEW_CHECKSUM_BOUND_AMENDMENT",
        "enabled": False,
        "current_v2_execution": "FORBIDDEN",
        "current_v2_append_rows": 0,
        "base_ledger_rule": (
            "the current lane contains exactly 960 predictor cells and 2880 "
            "association rows; no new cell, row, leaderboard ID, rank, or "
            "hash-chain record may be appended"
        ),
        "no_compatible_base_row": (
            "terminate as NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID; do not "
            "freeze a recipe and do not replay"
        ),
        "future_reconsideration": (
            "requires a new checksum-bound amendment, new review manifest, "
            "fresh formal review, new authorization, and a separate ledger "
            "namespace; it cannot mutate or rerank this v2 ledger"
        ),
    }
    assert freeze["creation_prerequisite"].endswith(
        "ledger_status=OBSERVED and recovery_compatible=true"
    )
    assert freeze["no_artifact_states"] == [
        "RECOVERY_INCONCLUSIVE_COVERAGE",
        "NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID",
    ]
    assert "do not authorize clean replay" in freeze["no_artifact_action"]
    assert "rank 1 is necessarily such a row" in screen["ranking"][
        "freeze_candidate_rule"
    ]
    assert "zero of 2880 rows" in verdict["state_conditions"][
        "all_predictor_cells_ineligible"
    ]
    assert "zero rows have recovery_compatible=true" in verdict["state_conditions"][
        "eligible_but_none_compatible"
    ]


def test_stat2_m03_bl90_screen_winner_is_exactly_allowlisted_for_replay():
    lane = _load(RECOVERY_PATH)
    contract = _load(ENVELOPE_PATH)
    bl90 = next(
        row for row in contract["envelope_builders"] if row["envelope_id"] == "BL90"
    )
    legacy = bl90["legacy_artifact_contract"]

    assert legacy["artifact_relative_path"] == "legacy_bl90_envelope.json"
    assert legacy["manifest_relative_path"] == "legacy_bl90_envelope_manifest.json"
    assert legacy["row_order"] == ["CP", "HO", "MP", "F"]
    assert legacy["row_exact_keys"] == [
        "path_type",
        "L_binary64_be_hex",
        "M_binary64_be_hex",
        "U_binary64_be_hex",
        "row_canonical_sha256",
    ]
    assert "16 lowercase hexadecimal digits" in legacy["binary64_encoding"]
    assert legacy["manifest_serialization"].startswith(
        "the same artifact_serialization"
    )
    assert "used by the screen" in legacy["screen_binding"]
    assert "otherwise every affected BL90 predictor cell is INELIGIBLE_REFERENCE" in (
        legacy["rankability"]
    )

    binding = lane["selected_recipe_freeze"]["BL90_conditional_binding"]
    assert binding["required_when_selected_envelope_id"] == "BL90"
    assert binding["selected_rows_order"].endswith("in that canonical order")
    assert binding["selected_row_exact_keys"] == legacy["row_exact_keys"]
    replay = lane["clean_independent_replay"]
    assert "exact published-commit closed code snapshot" in replay["independence"]
    assert replay["independence"].endswith("Git worktree execution is forbidden")
    assert any(
        "exact checksum-bound legacy BL90 artifact" in item
        for item in replay["allowed_inputs"]
    )
    assert "require byte identity to the artifact used by the screen" in replay[
        "BL90_input_verification"
    ]


def test_stat2_m04_only_the_structural_control_exists_in_current_v2():
    lane = _load(RECOVERY_PATH)
    hard = lane["rating_blind_feature_bank"]["hard_negative_control"]
    controls = lane["registered_controls_and_robustness"]
    verdict = lane["verdict_ladder"]

    assert hard["current_role"].endswith("fixture and fatal gate only")
    assert hard["association_execution"] is False
    assert hard["leaderboard_rows"] == 0
    current = controls["current_structural_gate"]
    assert current["control_id"] == "NC_PRETSTAR_HISTORY_ONLY"
    assert current["association_rows"] == 0
    assert current["leaderboard_membership"] is False
    assert current["verdict_effect"].startswith("none after its structural gate passes")
    future = controls["future_optional_denied"]
    assert future["status"] == "DENY_REQUIRES_NEW_CHECKSUM_BOUND_AMENDMENT"
    assert future["current_execution"] == "FORBIDDEN"
    assert future["current_ranking_or_verdict_effect"] == "NONE"
    assert controls["final_verdict_rule"].endswith(
        "same-dataset recovery label"
    )
    assert verdict["clean_replay_pass"] == (
        "HISTORICAL_RESULT_RECOVERED_ON_SAME_DATASET"
    )
    verdict_text = json.dumps(verdict, sort_keys=True)
    assert "CONTROL_SENSITIVE" not in verdict_text
    assert "ROBUSTLY_WITHIN" not in verdict_text


def test_stat2_m05_ledger_bytes_chain_order_and_terminal_digest_are_frozen():
    lane = _load(RECOVERY_PATH)
    screen = lane["full_data_recovery_screen"]
    ledger = screen["append_only_ledger"]
    support = screen["association_support_contract"]
    stability = screen["stability"]

    assert ledger["exact_row_keys"] == [
        "schema_version",
        "row_index",
        "leaderboard_id",
        "cell_id",
        "association_id",
        "ledger_status",
        "reason_code",
        "upstream_status",
        "first_failure_stage",
        "reason_priority",
        "raw_denominators",
        "metrics",
        "recovery_compatible",
        "rank_sort_tuple",
        "prev_record_sha256",
        "record_sha256",
    ]
    assert ledger["raw_denominators_exact_keys"] == support[
        "required_denominator_fields"
    ]
    assert set(ledger["metrics_exact_keys"]) == set(
        stability["row_metric_fields"]
    )
    assert "CPython 3.9" in ledger["canonical_json"]
    assert ledger["hash_domain_prefix"].endswith("NUL byte 0x00")
    assert ledger["genesis_prev_record_sha256"] == "0" * 64
    assert "removing only record_sha256" in ledger["record_sha256_preimage"]
    assert "sampling_axis order" in ledger["base_row_order"]
    assert "association_axis order RWS, PSP, PPR" in ledger["base_row_order"]
    assert ledger["worker_completion_order"].startswith("workers never append")
    assert "every row including the last has exactly one LF" in ledger[
        "stored_jsonl"
    ]
    terminal = ledger["terminal_digest_artifact"]
    assert terminal["row_count"] == 2880
    assert terminal["terminal_record_sha256"] == "record_sha256 of row_index 2879"
    assert terminal["ledger_sha256"] == "SHA256 of exact stored_jsonl bytes"

    prefix = b"RQ014-RECOVERY-LEDGER-v2\x00"
    assert prefix.endswith(b"\x00")
    preimage_row = {"prev_record_sha256": "0" * 64, "row_index": 0}
    preimage_bytes = (
        json.dumps(
            preimage_row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    record_sha = hashlib.sha256(prefix + preimage_bytes).hexdigest()
    assert len(record_sha) == 64
    assert "record_sha256" not in preimage_row


def test_stat2_m06_bootstrap_occurrences_have_exact_weights_and_success_policy():
    contract = _load(ENVELOPE_PATH)
    weighted = contract["weighted_quantile_contract"]
    bootstrap = contract["envelope_gate"]["bootstrap"]

    assert "K_e" in weighted["within_episode_weight"]
    assert "(1/E)/K_e" in weighted["within_episode_weight"]
    assert "independent weighting group" in bootstrap["resampling"]
    assert "(1/E)/K_e_j" in bootstrap["occurrence_observation_weight"]
    assert "stable-sort observations once" in bootstrap["weight_normalization_order"]
    assert "exactly the 1000 registered RNG calls" in bootstrap[
        "fixed_draw_count_and_retry_policy"
    ]
    assert "never retried, replaced, discarded" in bootstrap[
        "fixed_draw_count_and_retry_policy"
    ]
    for valid_case in (
        "fewer than 3 distinct",
        "q05=q50",
        "q50=q95",
        "half-widths equal to 0.0",
    ):
        assert valid_case in bootstrap["finite_degenerate_replicates"]
    for invalid_case in ("empty retained observation set", "nonfinite"):
        assert invalid_case in bootstrap["replicate_failure_predicate"]
    assert "these are the only replicate failures" in bootstrap[
        "replicate_failure_predicate"
    ]
    assert "FATAL_BOOTSTRAP_WEIGHTING_ERROR" in bootstrap[
        "strict_positive_weight_invariant"
    ]
    assert bootstrap["successful_replicates_required"] == 1000
    assert "any value other than 1000" in bootstrap["successful_replicate_count"]
    assert "including valid zero widths" in bootstrap["half_width_relative_se"]
    assert "/999" in bootstrap["half_width_relative_se"]

    # Unequal source observation counts still give every drawn occurrence 1/E.
    E = 3
    occurrence_source_counts = [2, 5, 5]
    occurrence_weight_sums = [
        sum((1 / E) / count for _ in range(count))
        for count in occurrence_source_counts
    ]
    assert all(abs(value - 1 / E) < 1e-15 for value in occurrence_weight_sums)
    assert abs(sum(occurrence_weight_sums) - 1.0) < 1e-15


def test_stat2_minor_fold_seed_uses_exact_utf8_bytes():
    lane = _load(RECOVERY_PATH)
    fold = lane["full_data_recovery_screen"]["stability"]["five_fold_rule"]

    assert fold["prefix_bytes"].endswith("exactly RQ014-R2-FOLD|")
    assert "strict UTF-8" in fold["segment_bytes"]
    assert "no BOM" in fold["segment_bytes"]
    seed_material = b"RQ014-R2-FOLD|" + "场景-2".encode("utf-8")
    assert seed_material.hex() == "52513031342d52322d464f4c447ce59cbae699af2d32"
    digest = hashlib.sha256(seed_material).hexdigest()
    assert digest == "907bc9e9a1a14391fe41a4f39fb95f79fe61ed86750dce44410dbe76f5e01cd0"
    assert int.from_bytes(bytes.fromhex(digest)[:8], "big") % 5 == 0
    assert fold["fixture_id"] == "FOLD_UTF8_BYTES"


def test_stat2_minor_replay_rank_vectors_are_method_scoped():
    lane = _load(RECOVERY_PATH)
    rank_vectors = lane["clean_independent_replay"]["pass_tolerances"][
        "rank_vectors_by_association"
    ]

    assert set(rank_vectors) == {"RWS", "PSP", "PPR"}
    assert rank_vectors["RWS"].startswith("exact scene-local")
    assert rank_vectors["PSP"].startswith("exact pooled")
    assert rank_vectors["PPR"].startswith("NOT_APPLICABLE")
    assert "no diagnostic rank vector is a replay gate" in rank_vectors["PPR"]


def test_stat3_m01_derivatives_are_sliced_then_computed_without_position_halo():
    lane = _load(RECOVERY_PATH)
    contract = _load(ENVELOPE_PATH)
    timeline = contract["timeline_and_state_contract"]
    derived = timeline["derived_state_rule"]
    support = contract["temporal_position_support_contract"]

    assert contract["authority"]["window_local_state_authority"].startswith(
        "for RQ014 recovery computation"
    )
    assert "supersedes any transport-schema prose" in lane[
        "rating_blind_feature_bank"
    ]["envelope_builder_contract"]["derivative_authority"]
    assert "slice that interval's resampled position rows first" in derived["stage"]
    assert derived["position_window_input"].endswith("require n>=2")
    assert derived["velocity_first_tick"] == "(p[1]-p[0])/dt"
    assert derived["velocity_interior_tick"] == "(p[k+1]-p[k-1])/(2*dt)"
    assert derived["velocity_last_tick"] == "(p[n-1]-p[n-2])/dt"
    assert derived["derivative_halo"].startswith("FORBIDDEN")
    assert derived["cross_window_state_reuse"].startswith("FORBIDDEN")
    assert "nearest earlier defined heading in that window" in derived["heading"]
    assert support["CH"].endswith("later than tau")
    assert support["TP"].endswith("later than tau")
    for mode in ("LF", "HF", "TF"):
        assert "only" in support[mode]

    def window_velocity(position: list[float], dt: float = 1.0) -> list[float]:
        result = [(position[1] - position[0]) / dt]
        result.extend(
            (position[index + 1] - position[index - 1]) / (2 * dt)
            for index in range(1, len(position) - 1)
        )
        result.append((position[-1] - position[-2]) / dt)
        return result

    # CH/TP terminal velocity is backward one-sided and cannot see a future halo.
    causal_window = [0.0, 1.0, 4.0]
    assert window_velocity(causal_window) == [1.0, 2.0, 3.0]
    future_halo_a = causal_window + [9.0]
    future_halo_b = causal_window + [9999.0]
    assert window_velocity(future_halo_a[:3]) == window_velocity(future_halo_b[:3])
    assert window_velocity(future_halo_a)[2] != window_velocity(causal_window)[2]

    # LF first velocity is forward one-sided and cannot see a past halo.
    lookahead_window = [1.0, 4.0, 9.0]
    assert window_velocity(lookahead_window) == [3.0, 4.0, 5.0]
    assert window_velocity([-9999.0] + lookahead_window)[1] != window_velocity(
        lookahead_window
    )[0]

    envelope_text = ENVELOPE_PATH.read_text(encoding="utf-8")
    recovery_text = RECOVERY_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "derive once on each complete resampled branch before slicing any estimator window",
        "window_boundary_rederivation",
        "reconstruct velocity, acceleration, and heading once on the complete resampled branch",
        "reconstruct velocity, acceleration, and heading from the complete position grid",
    ):
        assert forbidden not in envelope_text
        assert forbidden not in recovery_text


def test_stat3_m02_hfeas_and_tf_anchor_domains_are_finite_and_shared():
    lane = _load(RECOVERY_PATH)
    contract = _load(ENVELOPE_PATH)
    lane_horizons = {
        row["horizon_id"]: row for row in lane["rating_blind_feature_bank"]["horizon_axis"]
    }
    horizon = contract["horizon_anchor_contract"]
    assert "finite nonnegative integer grid tick" in contract[
        "timeline_and_state_contract"
    ]["common_end"]

    assert lane_horizons["H20"]["candidate_tau_ticks"].startswith(
        "every integer tau_tick from rate_hz through 2*rate_hz inclusive"
    )
    assert "every tick's exact registered temporal window is complete" in (
        lane_horizons["H20"]["anchor_rule"]
    )
    assert lane_horizons["HFEAS"]["candidate_tau_ticks"].startswith(
        "finite integer closed interval rate_hz"
    )
    assert lane_horizons["HFEAS"]["maximum_tau_tick"] == (
        "h_common_tick for every temporal recipe, including TF"
    )
    assert "exactly rate_hz..h_common_tick" in lane_horizons["HFEAS"]["TF_rule"]

    assert horizon["H20"]["candidate_tau_ticks"] == (
        "integer closed interval rate_hz..2*rate_hz"
    )
    assert horizon["H20"]["requirement"].startswith(
        "require 2*rate_hz<=h_common_tick"
    )
    assert horizon["HFEAS"]["candidate_tau_ticks"] == (
        "integer closed interval rate_hz..h_common_tick, empty when "
        "h_common_tick<rate_hz"
    )
    assert horizon["HFEAS"]["maximum_tau_tick"].endswith("and TF")
    assert "never creates anchors beyond h_common_tick" in horizon["HFEAS"]["TF"]
    assert "scene_anchor_domain_contract" in horizon["domain_materialization"]

    for rate_hz, h_common_tick in ((4, 20), (10, 37)):
        h20_ticks = list(range(rate_hz, 2 * rate_hz + 1))
        hfeas_candidates = list(range(rate_hz, h_common_tick + 1))
        tf_hfeas_ticks = [
            tau_tick
            for tau_tick in hfeas_candidates
            if 0 >= 0 and h_common_tick <= h_common_tick
        ]
        assert h20_ticks[0] == rate_hz
        assert h20_ticks[-1] == 2 * rate_hz
        assert tf_hfeas_ticks[0] == rate_hz
        assert tf_hfeas_ticks[-1] == h_common_tick
        assert len(tf_hfeas_ticks) == h_common_tick - rate_hz + 1
        assert all(tau_tick <= h_common_tick for tau_tick in tf_hfeas_ticks)

    # TF cannot evade the universal tau<=H_common bound under H20 either.
    rate_hz = 4
    too_short_h_common = 6
    tf_h20_ticks = (
        list(range(rate_hz, 2 * rate_hz + 1))
        if 2 * rate_hz <= too_short_h_common
        else []
    )
    assert tf_h20_ticks == []
    assert "first require tau_tick<=h_common_tick" in horizon[
        "window_completeness"
    ]

    query = contract["wod_query_manifest"]
    assert "one unique group-by" in query["rule"]
    assert "wod_scene_anchor_domain.csv" in query["source_artifact"]
    assert "wod_scene_anchor_domain.csv directly" in query["domain_binding"]
    builders = {row["envelope_id"]: row for row in contract["envelope_builders"]}
    assert "rate_hz..h_common_tick" in builders["BM90"]["pooling_domain"]["HFEAS"]
    assert "never materialize a TF tau cell above" in builders["BT90"]["TF"]
    assert "always <=h_common_tick" in lane["rating_blind_feature_bank"][
        "readout_rules"
    ]["LAST"]

    # The temporal/horizon closure changes no registered Cartesian counts.
    assert len(contract["feature_families"]) == 16
    assert contract["execution_enumeration"]["feature_envelope_execution_count"] == 48
    assert lane["rating_blind_feature_bank"]["predictor_cell_enumeration"][
        "registered_predictor_cell_count"
    ] == 960
    assert lane["full_data_recovery_screen"]["registered_leaderboard_row_count"] == 2880


def test_stat4_m01_scene_anchor_domain_schema_groups_and_consumers_are_exact():
    lane = _load(RECOVERY_PATH)
    contract = _load(ENVELOPE_PATH)
    domain = contract["scene_anchor_domain_contract"]
    query = contract["wod_query_manifest"]

    assert domain["artifact"] == "wod_scene_anchor_domain.csv"
    assert "downstream IPV solver and envelope outcomes remain separate" in domain[
        "status_scope"
    ]
    assert domain["expected_group_count"] == (
        "479*16*2=15328 segment_id x feature_id x horizon_id groups"
    )
    assert domain["exact_columns"] == [
        "segment_id",
        "feature_id",
        "horizon_id",
        "path_type_or_NA",
        "h_common_tick_or_NA",
        "tau_tick_or_NA",
        "membership_status",
        "reason_code",
    ]
    assert domain["primary_key"] == [
        "segment_id",
        "feature_id",
        "horizon_id",
        "tau_tick_or_NA",
    ]
    assert domain["group_key"] == ["segment_id", "feature_id", "horizon_id"]
    assert "lineterminator='\\n'" in domain["encoding"]
    assert "CR is forbidden" in domain["encoding"]
    assert domain["row_order"].startswith("segment_id raw UTF-8 bytes ascending")
    assert domain["registered_membership_statuses"] == [
        "AVAILABLE",
        "INELIGIBLE_BLIND",
        "MISSING_WOD_PATH_TYPE",
        "AMBIGUOUS_WOD_PATH_TYPE",
        "UNRECOGNIZED_PATH_TYPE",
        "INELIGIBLE_TIMELINE_SUPPORT",
        "INELIGIBLE_TIMELINE_SOURCE_GAP",
        "INELIGIBLE_TIMELINE_GRID_PHASE",
        "INELIGIBLE_TIMELINE_SEAM",
        "INELIGIBLE_STATE_NONFINITE",
        "INELIGIBLE_UNDEFINED_HEADING",
    ]
    rollup_statuses = {
        row["upstream_status"]
        for row in lane["full_data_recovery_screen"]["upstream_terminal_rollup"]["rows"]
    }
    assert set(domain["registered_membership_statuses"]) <= rollup_statuses
    assert domain["reason_code_registry"].endswith(
        "checksum-bound with this contract"
    )
    assert domain["available_group"].startswith("one or more rows")
    assert "set(path_type_or_NA) has cardinality exactly 1" in domain[
        "available_group"
    ]
    assert "equals the exact checksum-bound WOD path lookup" in domain[
        "available_group"
    ]
    assert "set(h_common_tick_or_NA) cardinality exactly 1" in domain[
        "available_group"
    ]
    path_invariant = domain["path_type_group_invariant"]
    assert path_invariant["available_path_type_set_cardinality"] == 1
    assert "same checksum-bound lookup value" in path_invariant[
        "cross_group_scene_binding"
    ]
    assert path_invariant["mismatch"] == "FATAL_ANCHOR_DOMAIN_INTEGRITY_ERROR"
    assert domain["terminal_group"].startswith("exactly one row")
    assert domain["mixed_group"] == "FATAL_ANCHOR_DOMAIN_INTEGRITY_ERROR"
    assert domain["missing_or_duplicate_group"] == (
        "FATAL_ANCHOR_DOMAIN_INTEGRITY_ERROR"
    )
    manifest = domain["manifest"]
    assert manifest["artifact"] == "wod_scene_anchor_domain_manifest.json"
    assert "artifact_sha256" in manifest["exact_keys"]
    assert "generator_sha256" in manifest["exact_keys"]
    assert manifest["count_identities"] == [
        "group_count=15328",
        "group_count=available_group_count+terminal_group_count",
    ]

    binding = lane["rating_blind_feature_bank"]["scene_anchor_domain_binding"]
    assert binding["group_key"] == domain["group_key"]
    assert "consume only the checksum-verified rows" in binding["execution_rule"]
    readout = lane["rating_blind_feature_bank"]["readout_rules"]["ANCHOR_DOMAIN"]
    assert "wod_scene_anchor_domain.csv" in readout
    assert "required_envelope_queries.csv is aggregate-only" in readout

    assert query["source_artifact"].startswith("only checksum-verified AVAILABLE rows")
    assert "one unique group-by" in query["rule"]
    assert "count(distinct segment_id)" in query["rule"]
    assert "contribute only to that one checksum-bound path_type branch" in query[
        "group_by_identity"
    ]
    assert query["other_generation_paths"] == "FORBIDDEN"
    assert "contains no segment_id" in query["aggregate_only"]
    assert "cannot identify, reconstruct" in query["aggregate_only"]
    builders = {row["envelope_id"]: row for row in contract["envelope_builders"]}
    for envelope_id in ("BM90", "BT90"):
        assert builders[envelope_id]["query_source"].startswith(
            "consume only the unique checksum-bound required_envelope_queries.csv"
        )

    artifacts = contract["materialized_artifacts"]
    assert artifacts["wod_scene_anchor_domain"] == "wod_scene_anchor_domain.csv"
    assert artifacts["wod_scene_anchor_domain_manifest"] == (
        "wod_scene_anchor_domain_manifest.json"
    )
    assert artifacts["required_envelope_queries"] == "required_envelope_queries.csv"
    assert len(artifacts["manifest_required_bindings"]) >= 8
    assert lane["clean_independent_replay"]["pass_tolerances"][
        "scene_anchor_domain_group_status_reason_and_tau_membership"
    ].startswith("exact for every segment_id")
    freeze_binding = lane["selected_recipe_freeze"]["scene_anchor_domain_binding"]
    assert freeze_binding["required"] is True
    assert "selected_group_projection_sha256" in freeze_binding["exact_keys"]
    assert "without reading the screen artifact" in freeze_binding["replay_use"]
    assert "screen artifact itself remains forbidden input" in lane[
        "clean_independent_replay"
    ]["scene_anchor_domain_verification"]


def test_stat4_m01_membership_bytes_are_reversible_beyond_the_aggregate():
    contract = _load(ENVELOPE_PATH)
    domain = contract["scene_anchor_domain_contract"]
    columns = domain["exact_columns"]

    def available_row(
        segment_id: str, tau_tick: int, path_type: str = "CP"
    ) -> dict[str, str]:
        return {
            "segment_id": segment_id,
            "feature_id": "F-R04N-CH-W10",
            "horizon_id": "HFEAS",
            "path_type_or_NA": path_type,
            "h_common_tick_or_NA": "NA",
            "tau_tick_or_NA": str(tau_tick),
            "membership_status": "AVAILABLE",
            "reason_code": "F_AVAILABLE_CONTINUE",
        }

    def terminal_row(segment_id: str) -> dict[str, str]:
        return {
            "segment_id": segment_id,
            "feature_id": "F-R04N-CH-W10",
            "horizon_id": "HFEAS",
            "path_type_or_NA": "CP",
            "h_common_tick_or_NA": "NA",
            "tau_tick_or_NA": "NA",
            "membership_status": "INELIGIBLE_TIMELINE_SUPPORT",
            "reason_code": "F_TIMELINE_SUPPORT",
        }

    def encode(rows: list[dict[str, str]]) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            dialect="excel",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8")

    def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, ...], int]:
        result: dict[tuple[str, ...], set[str]] = {}
        for row in rows:
            if row["membership_status"] != "AVAILABLE":
                continue
            key = (
                row["feature_id"],
                row["path_type_or_NA"],
                row["horizon_id"],
                row["tau_tick_or_NA"],
                row["h_common_tick_or_NA"],
            )
            result.setdefault(key, set()).add(row["segment_id"])
        return {key: len(segment_ids) for key, segment_ids in result.items()}

    checksum_bound_path_lookup = {
        "scene-a": "CP",
        "scene-b": "CP",
        "scene-c": "HO",
    }

    def group_is_valid(rows: list[dict[str, str]]) -> bool:
        statuses = [row["membership_status"] for row in rows]
        if all(status == "AVAILABLE" for status in statuses):
            ticks = [int(row["tau_tick_or_NA"]) for row in rows]
            segment_ids = {row["segment_id"] for row in rows}
            path_types = {row["path_type_or_NA"] for row in rows}
            expected_path_type = (
                checksum_bound_path_lookup[next(iter(segment_ids))]
                if len(segment_ids) == 1
                and next(iter(segment_ids)) in checksum_bound_path_lookup
                else None
            )
            return (
                len(rows) >= 1
                and ticks == sorted(set(ticks))
                and all(row["reason_code"] == "F_AVAILABLE_CONTINUE" for row in rows)
                and len(segment_ids) == 1
                and len(path_types) == 1
                and path_types == {expected_path_type}
                and all(row["h_common_tick_or_NA"] == "NA" for row in rows)
            )
        return (
            len(rows) == 1
            and rows[0]["tau_tick_or_NA"] == "NA"
            and rows[0]["membership_status"]
            in set(domain["registered_membership_statuses"]) - {"AVAILABLE"}
            and rows[0]["reason_code"] == "F_TIMELINE_SUPPORT"
        )

    membership_a = [available_row("scene-a", 4), available_row("scene-b", 5)]
    membership_b = [available_row("scene-a", 5), available_row("scene-b", 4)]
    assert aggregate(membership_a) == aggregate(membership_b)
    bytes_a = encode(membership_a)
    bytes_b = encode(membership_b)
    assert bytes_a != bytes_b
    assert hashlib.sha256(bytes_a).hexdigest() != hashlib.sha256(bytes_b).hexdigest()
    assert b"\r" not in bytes_a and bytes_a.endswith(b"\n")

    assert group_is_valid([available_row("scene-a", 4), available_row("scene-a", 5)])
    assert group_is_valid(
        [available_row("scene-c", 4, "HO"), available_row("scene-c", 5, "HO")]
    )
    assert not group_is_valid(
        [available_row("scene-a", 4, "CP"), available_row("scene-a", 5, "HO")]
    )
    assert not group_is_valid(
        [available_row("scene-a", 4, "HO"), available_row("scene-a", 5, "HO")]
    )
    assert group_is_valid([terminal_row("scene-c")])
    assert not group_is_valid(
        [available_row("scene-c", 4), terminal_row("scene-c")]
    )
    assert not group_is_valid([terminal_row("scene-c"), terminal_row("scene-c")])


def test_stat4_minor_na_cluster_limitation_remains_an_accepted_residual():
    lane = _load(RECOVERY_PATH)
    marker = lane["full_data_recovery_screen"]["recovery_compatible_marker"]
    assert marker["accepted_residuals"] == {
        "STAT4-m01": (
            "scenario_cluster=NA scenes remain in native support but are not a "
            "LOCO cluster; no NA-share ceiling or leave-all-NA omission is added "
            "to recovery_compatible in this v2 contract"
        )
    }
    requirements = marker["requirements"]
    assert not any("NA-share" in item or "leave-all-NA" in item for item in requirements)
    assert "eligible_cluster_count_metric.kind_equals_FINITE_INT_and_value_at_least_2" in (
        requirements
    )

    contract = _load(ENVELOPE_PATH)
    assert len(contract["feature_families"]) == 16
    assert contract["execution_enumeration"]["feature_envelope_execution_count"] == 48
    assert lane["rating_blind_feature_bank"]["predictor_cell_enumeration"][
        "registered_predictor_cell_count"
    ] == 960
    assert lane["full_data_recovery_screen"]["registered_leaderboard_row_count"] == 2880
