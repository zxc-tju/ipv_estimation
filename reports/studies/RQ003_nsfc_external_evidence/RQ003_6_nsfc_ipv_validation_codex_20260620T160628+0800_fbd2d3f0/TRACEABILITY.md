# RQ003 Phase 10 Traceability

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

All reader-facing figures were generated under the `nature-figure` Python
backend contract. Source result tables were read-only.

## Figure Traceability

| Figure | Primary source tables | Figure source CSV | Metadata |
|---|---|---|---|
| fig01_provenance_coverage | coverage_matrix.csv;scenario_crosswalk_corrected.csv;missingness_audit.csv;gate_minus1_status.json | 01_results/figures/fig01_provenance_coverage_source.csv | 01_results/figures/fig01_provenance_coverage_metadata.json |
| fig02_missingness_selection_bias | missingness_audit.csv;coverage_matrix.csv | 01_results/figures/fig02_missingness_selection_bias_source.csv | 01_results/figures/fig02_missingness_selection_bias_metadata.json |
| fig03_support_ood_abstention | support_coverage.csv;state_dependence_results.csv | 01_results/figures/fig03_support_ood_abstention_source.csv | 01_results/figures/fig03_support_ood_abstention_metadata.json |
| fig04_directional_ipv_signature | cell_level_directional_ipv.csv;g0r_cond_001_status.json;ipv_sign_contract.md | 01_results/figures/fig04_directional_ipv_signature_source.csv | 01_results/figures/fig04_directional_ipv_signature_metadata.json |
| fig05_scenario_fix_before_after | confirmatory_results__before_scenario_fix.csv;confirmatory_results.csv;scenario_fix_result_delta.md | 01_results/figures/fig05_scenario_fix_before_after_source.csv | 01_results/figures/fig05_scenario_fix_before_after_metadata.json |
| fig06_negative_controls | negative_controls.csv;confirmatory_results.csv | 01_results/figures/fig06_negative_controls_source.csv | 01_results/figures/fig06_negative_controls_metadata.json |
| fig07_state_dependence_boundary | state_dependence_results.csv;state_dependence_report.md | 01_results/figures/fig07_state_dependence_boundary_source.csv | 01_results/figures/fig07_state_dependence_boundary_metadata.json |
| fig08_independent_replication | implementation_comparison_v2.csv;replication2_status.json | 01_results/figures/fig08_independent_replication_source.csv | 01_results/figures/fig08_independent_replication_metadata.json |
| fig09_tier_b_evidence_map | claim_boundary_matrix.csv;tier_decision.json;paper_handoff.md | 01_results/figures/fig09_tier_b_evidence_map_source.csv | 01_results/figures/fig09_tier_b_evidence_map_metadata.json |

## Claim Traceability

See `evidence.csv` for claim-level artifact, table, figure, reviewer status,
allowed wording, and limitation fields.

## Verification Summary

- Figure manifest completeness: PASS
- Entry HTML offline check: PASS
- Compat HTML offline check: PASS
- Byte-identical HTML entries: True
- Forbidden-wording scan: PENDING
