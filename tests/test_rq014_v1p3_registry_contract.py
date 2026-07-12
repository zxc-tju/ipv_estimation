from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = REPO_ROOT / "reports/plans"
VALID = PLAN_DIR / "RQ014_config_space_v1p3.yaml"
FORENSIC = PLAN_DIR / "RQ014_forensic_registry_v1p3.yaml"
EXTENSION = PLAN_DIR / "RQ014_recovery_extension_registry_v1p3.yaml"


def _load_yaml(path: Path) -> dict:
    if shutil.which("ruby") is None:
        pytest.skip("Ruby stdlib YAML parser is required for registry validation")
    script = "require 'yaml'; require 'json'; puts JSON.generate(YAML.load_file(ARGV[0]))"
    output = subprocess.check_output(["ruby", "-e", script, str(path)], text=True)
    return json.loads(output)


def _duplicate_yaml_keys(path: Path) -> list[str]:
    if shutil.which("ruby") is None:
        pytest.skip("Ruby stdlib YAML parser is required for registry validation")
    script = r"""
require 'psych'
require 'json'
duplicates = []
walk = lambda do |node, trail|
  if node.is_a?(Psych::Nodes::Mapping)
    seen = {}
    node.children.each_slice(2) do |key, value|
      if key.is_a?(Psych::Nodes::Scalar)
        location = (trail + [key.value]).join('.')
        duplicates << location if seen.key?(key.value)
        seen[key.value] = true
        walk.call(value, trail + [key.value])
      else
        walk.call(value, trail)
      end
    end
  elsif node.respond_to?(:children) && node.children
    node.children.each { |child| walk.call(child, trail) }
  end
end
walk.call(Psych.parse_file(ARGV[0]), [])
puts JSON.generate(duplicates)
"""
    output = subprocess.check_output(["ruby", "-e", script, str(path)], text=True)
    return json.loads(output)


def test_v1p3_registries_are_disabled_and_resolved() -> None:
    valid, forensic, extension = map(_load_yaml, (VALID, FORENSIC, EXTENSION))
    assert valid["schema_version"] == "rq014-valid-scientific-v1p3"
    assert forensic["schema_version"] == "rq014-forensic-v1p3"
    assert extension["schema_version"] == "rq014-recovery-extension-v1p3"
    for registry in (valid, forensic, extension):
        assert registry["execution_authorized"] is False
        assert registry["amendment"].endswith("RQ014_plan_v1p3_amendment_20260711.md")
        assert registry["authorization_scopes"] == {
            "g0_readonly_forensics_authorized": False,
            "g2_ratings_blind_build_authorized": False,
            "discovery_rating_join_authorized": False,
            "confirmation_rating_join_authorized": False,
            "scientific_compute_authorized": False,
            "extension_compute_authorized": False,
            "forensic_rating_join_authorized": False,
            "forensic_compute_authorized": False,
        }
        assert registry["authorization_operation_map"]

    valid_map = valid["authorization_operation_map"]
    assert valid_map["listed_scopes_are_logical_AND"] is True
    assert valid_map["G2P_partition_freeze"] == ["g2_ratings_blind_build_authorized"]
    assert set(valid_map["G2_runtime_pilot_and_G2P_pipeline_power_simulation"]) == {
        "g2_ratings_blind_build_authorized",
        "scientific_compute_authorized",
    }

    configs = valid["configs"]
    assert len(configs) == 12
    assert len({row["config_id"] for row in configs}) == 12
    assert len(
        {
            (
                row["rate_hz"],
                row["trailing_window_s"],
                row["sequence_mode"],
                row["horizon"],
            )
            for row in configs
        }
    ) == 12


def test_v1p3_statistics_contract_closes_reviewed_ambiguities() -> None:
    valid = _load_yaml(VALID)
    contract = valid["statistical_contract_v1p3"]
    assert contract["nominal_one_sided_alpha"] == 0.025
    assert list(contract["freeze_and_execution_timeline"]) == [
        "G2_pre_rating_freeze",
        "G2P_partition_freeze",
        "authorized_discovery_join",
        "post_promotion_pre_confirmation_feature_freeze",
        "authorized_confirmation_join",
        "G4S_nuisance_only_calibration",
        "observed_tier_statistics",
    ]
    se_contract = contract["common_primitives"]["standard_error_contract"]
    assert se_contract["jackknife_SE"] == "forbidden"
    assert se_contract["reestimate_SE_inside_replicate"] is False

    extension = contract["extension_X01_to_X05"]
    assert extension["analysis_partition"]["confirmation_partition_rows_forbidden"] is True
    master = extension["scene_manifests"]["union_master"]
    assert master["self_reference_to_any_M_manifest"] == "forbidden"
    pairwise = extension["scene_manifests"]["pairwise_with_V04"]
    assert pairwise["M_Xk_V04"] == "exact_alias_of_M_Xk"
    assert "same_canonical_file" in pairwise["byte_identity_requirement"]
    assert extension["keyed_within_scene_permutation"]["key_fields"] == [
        "replicate_index_one_based",
        "scene_id",
    ]
    assert extension["association_family"]["multiplicity"]["p_adjusted_k"]
    assert extension["association_family"]["multiplicity"]["scope"].startswith(
        "complete_null_exploratory"
    )
    assert extension["recovery_flag"]["omnibus_p_never_sufficient_for_a_cell_flag"] is True
    assert extension["recovery_flag"]["label"] == "EXPLORATORY_SPEC_RECOVERY_CANDIDATE"
    assert extension["recovery_flag"]["strong_FWER_claim_allowed"] is False
    assert "delta_obs_k_less_than_or_equal_to_minus_0p20" in extension[
        "recovery_flag"
    ]["all_conditions_required"]
    assert "p_adjusted_exclusivity_k_less_than_or_equal_to_0p025" in extension[
        "recovery_flag"
    ]["all_conditions_required"]

    tiers = contract["specificity_tiers"]
    assert tiers["common_scene_manifests"]["partition_contract"][
        "analysis_partition"
    ] == "frozen_confirmation_partition_only"
    assert tiers["common_scene_manifests"]["tier_P"]["canonical_analysis_artifact"] == (
        "tier_PI_analysis_ids.csv"
    )
    feature_contract = tiers["common_scene_manifests"]["feature_master_contract"]
    assert feature_contract["selection_may_not_use"].startswith("rating_completeness")
    assert "confirmation_partition_member" in feature_contract["common_required_columns"]
    invariance = tiers["tier_C"]["invariance_diagnostics"]
    assert invariance["inferential_status"] == "DESCRIPTIVE_ONLY_NON_GATING"
    assert invariance["cannot_cap_or_upgrade_any_scientific_label"] is True
    assert invariance["structural_alias_check"]["alias_if_and_only_if"].startswith(
        "exact_vector_equality"
    )
    assert invariance["structural_alias_check"]["spearman_rank_correlation"].startswith(
        "descriptive_only"
    )
    assert tiers["tier_P"]["primary_margin"]["delta_NI"] == 0.08
    assert tiers["tier_P"]["primary_margin"]["descriptive_sensitivities"] == [0.04, 0.12]
    assert tiers["tier_P"]["primary_margin"]["empirical_effect_derivation_claimed"] is False
    assert tiers["tier_I"]["observed_collinearity_gate"]["on_hard_singular"].startswith(
        "TIER_I_UNAVAILABLE"
    )
    assert tiers["verdict_ladder"]["mapping"][
        "tier_C_and_tier_P_and_tier_I_pass"
    ].endswith("INCREMENTAL_BEYOND_FROZEN_KINEMATICS_COMPOSITE")

    dgp = contract["joint_boundary_and_power_DGP"]
    assert dgp["outer_validation"]["simulations_per_grid_cell"] == 20000
    assert dgp["outer_validation"]["fixed_denominator"] == 20000
    assert dgp["outer_validation"]["failed_outer_replicates_excluded_from_denominator"] is False
    assert dgp["outer_validation"]["worst_case_rule"]["averaging_across_cells"] == "forbidden"
    assert dgp["grids"]["tier_C_partial_null_boundary"]["cell_count"] == 18
    assert dgp["grids"]["global_zero_sanity"]["cell_count"] == 9
    assert dgp["grids"]["tier_I_beta_D_zero_boundary"]["coefficient_constraint"] == (
        "gamma_Q_exactly_zero"
    )
    ceilings = dgp["fixed_compute_feasibility"]["hard_compute_ceiling"]
    assert ceilings == {
        "aggregate_CPU_hours": 20000,
        "peak_memory_GiB_per_task": 16,
        "maximum_wall_clock_hours_after_resources_are_allocated": 96,
    }
    assert dgp["fixed_compute_feasibility"]["total_registered_outer_batches"] == 2520
    assert "outer_replicate" not in dgp["RNG_contract"]["domains"]["inner_operator"]
    pilot = dgp["fixed_compute_feasibility"]["G2_ratings_blind_runtime_pilot"]
    assert pilot["outer_test_shape_pilots"]["tier_C"]["registered_outer_batches"] == 1200
    assert pilot["outer_test_shape_pilots"]["tier_P"]["registered_outer_batches"] == 840
    assert pilot["outer_test_shape_pilots"]["tier_I"]["registered_outer_batches"] == 480
    fit_shapes = pilot["calibration_fit_validation_shape_pilots"]
    assert [fit_shapes[tier]["optimizer_parameter_count"] for tier in ("tier_C", "tier_P", "tier_I")] == [3, 2, 1]
    assert pilot["resource_pilot_manifest"]["exact_bytes_sha256_required_before_G4S"] is True


def test_x02_and_extension_flags_are_ratings_blind_and_quantitative() -> None:
    valid = _load_yaml(VALID)
    x02 = valid["statistical_contract_v1p3"]["X02_scale_eligibility"]
    assert x02["decision_time"].startswith("ratings_blind_G2")
    assert x02["support_gate"]["minimum_calculability_fraction"] == 0.70
    assert x02["support_gate"]["minimum_coverage"] == 0.70
    assert x02["support_gate"]["all_eligible_rows_denominator"].endswith(
        "before_any_X02_calculability_or_finiteness_condition"
    )
    assert x02["parity_manifest"]["required_cases"] == 12
    thresholds = x02["scale_metrics"]["hard_thresholds_all_required"]
    assert thresholds["median_z_less_than_or_equal_to"] == 0.50
    assert thresholds["empirical_q90_z_less_than_or_equal_to"] == 1.00
    assert thresholds["spearman_v_X02_vs_v_legacy_greater_than_or_equal_to"] == 0.80
    preliminary = x02["scale_metrics"]["preliminary_validity"]
    assert preliminary["finite_fraction_denominator"].startswith(
        "every_scheduled_tau_row"
    )
    assert x02["scale_metrics"]["rank_correlation_contract"][
        "NaN_or_nonfinite_correlation"
    ] == "INELIGIBLE_SCALE_INCOMPATIBLE"

    extension = _load_yaml(EXTENSION)
    assert extension["analysis_contract"]["alpha_one_sided"] == 0.025
    flag = extension["analysis_contract"]["exclusivity_flag"]["all_required"]
    assert extension["analysis_contract"]["strong_FWER_claim_allowed"] is False
    assert extension["analysis_contract"]["exclusivity_flag"]["claim_on_flag"] == (
        "EXPLORATORY_SPEC_RECOVERY_CANDIDATE"
    )
    assert "delta_k_le_minus_0.20" in flag
    assert "p_adjusted_k_le_0.025" in flag
    assert "p_adjusted_exclusivity_k_le_0.025" in flag


def test_forensic_registry_uses_v1p3_fail_closed_artifacts() -> None:
    forensic = _load_yaml(FORENSIC)
    surfaces = {row["surface_id"]: row for row in forensic["forensic_surfaces"]}
    assert surfaces["F05"]["closure_script"].endswith("hpc_pass4_v1p3.sh")
    assert surfaces["F05"]["over_budget_script"].endswith("hpc_pass4_v1p3.sbatch")
    for surface_id in ("F06", "F07", "F08"):
        assert surfaces[surface_id]["closure_script"].endswith("mac_pass4_v1p3.sh")
    assert surfaces["F07"]["required_scope_manifest"].endswith(".json")
    assert surfaces["F08"]["required_scope_manifest"].endswith(".json")
    assert surfaces["F10"]["closure_script"].endswith("fl05_indexer_v1p3.sh")
    assert surfaces["F10"]["over_budget_script"].endswith("fl05_v1p3.sbatch")

    implementation = forensic["historical_lookup_cells"][-1]["implementation"]
    for value in implementation.values():
        assert (REPO_ROOT / value).is_file()

    closure = forensic["surface_closure_contract_v1p3"]
    assert closure["missing_or_read_failure_exit_nonzero"] is True
    assert closure["inaccessible_never_relabelled_not_found"] is True
    assert closure["bundle_layout"]["current_pointer"].endswith("CURRENT")
    assert closure["F05_nonempty_contract"]["empty_surface"].startswith("INACCESSIBLE")
    assert closure["F07_F08_frozen_scope_contract"]["cutoff_exclusive_utc"] == (
        "2026-07-10T00:00:00Z"
    )
    assert closure["F07_F08_frozen_scope_contract"]["schema_version"] == (
        "rq014-frozen-scope-v1p3"
    )
    assert closure["F07_F08_frozen_scope_contract"]["snapshot_evidence_kind"] == (
        "git_blob_v1_content_integrity_only"
    )
    witness = closure["F07_F08_frozen_scope_contract"]["cutoff_time_witness_contract"]
    assert witness["registered_production_TSA_profile_v1p3"] == "none"
    assert witness["registered_pre_cutoff_whole_inventory_receipt_v1p3"] == "none"
    assert witness["current_forced_state"] == (
        "INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT"
    )
    assert witness["git_time_fallback"] == "forbidden"
    terminal_rules = closure["legal_terminal_state_rules"]
    assert "F05_F06_NOT_FOUND_ON_SCANNED_SURFACES" in terminal_rules
    assert "that_surface_whole_inventory_receipt_valid" in terminal_rules[
        "F07_F08_NOT_FOUND_ON_SCANNED_SURFACES"
    ]
    assert closure["F05_HPC_execution_contract"]["job_name"].startswith("zxc-")
    assert (REPO_ROOT / closure["authoritative_scripts"]["F05_over_200MiB"]).is_file()

    fl05 = forensic["historical_lookup_cells"][-1]
    assert fl05["implementation_requirements"]["complete_zero_candidate_policy"].endswith(
        "COMPLETE_ZERO_CANDIDATES"
    )
    terminal = fl05["adjudication_contract"]["terminal_decision"]
    assert terminal["terminal_states_mutually_exclusive"] is True
    assert terminal["precedence"][0] == "FOUND"
    assert terminal["NOT_FOUND_ON_SCANNED_SURFACES"][
        "attributable_candidate_count_exact"
    ] == 0


def test_v1p3_has_no_active_point_zero_five_or_old_closure_reference() -> None:
    active_yaml = "\n".join(path.read_text(encoding="utf-8") for path in (VALID, FORENSIC, EXTENSION))
    assert "p<0.05" not in active_yaml
    assert "within_arm_maxT_p < 0.05" not in active_yaml
    assert "RQ014_forensics_hpc_pass3_20260710.sh" not in active_yaml
    assert "RQ014_forensics_mac_pass3b_20260710.sh" not in active_yaml
    assert "claim_on_flag: SPEC_RECOVERY_CANDIDATE" not in active_yaml
    assert "annotate_CONSTRUCTION_SUSPECT" not in active_yaml

    for path in (VALID, FORENSIC, EXTENSION):
        assert _duplicate_yaml_keys(path) == []

    sbatch = (
        PLAN_DIR / "prompts/RQ014_forensics_hpc_fl05_v1p3.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --job-name=zxc-" in sbatch
