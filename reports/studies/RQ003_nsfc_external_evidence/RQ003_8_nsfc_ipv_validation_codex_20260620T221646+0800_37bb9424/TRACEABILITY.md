# RQ003_8 Traceability

Run ID: `RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424`
Generated: `2026-06-22T10:35:56+08:00`
Plan SHA-256: `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`

## Identity

- Run manifest path: `02_process/00_meta/run_manifest.json`
- Tier decision path: `02_process/18_tier_review/tier_decision.json`
- Nature skill manifest: `02_process/19_report_build/nature_skill_manifest.json`

## Figure To Claim Map

- `fig01_provenance_coverage`: The planned top-five NSFC subset is fully mapped and computable, but inference is restricted to 10 of 20 teams and an official coordination score. Source CSV: `01_results/figures/fig01_provenance_coverage__source_data.csv`.
- `fig02_support_ood_abstention`: The primary high-support analysis is not evaluable because the frozen exact role-context key has no NSFC support, despite complete three-key overlap when role_context is ignored. Source CSV: `01_results/figures/fig02_support_ood_abstention__source_data.csv`.
- `fig03_directional_ipv_signature`: Fallback IPV predictors are sparse but non-degenerate and are not reconstructable from the frozen kinematic/safety baseline out of team. Source CSV: `01_results/figures/fig03_directional_ipv_signature__source_data.csv`.
- `fig04_capacity_matched_model_comparison`: The frozen primary comparison is not evaluable; fallback LOTO is small and uncertain, LOSO reverses, and the future-leaky control is negative. Source CSV: `01_results/figures/fig04_capacity_matched_model_comparison__source_data.csv`.
- `fig05_negative_controls`: Shuffle and removal controls do not manufacture a validated signal; IPV-removed is identical to the kinematics-only control. Source CSV: `01_results/figures/fig05_negative_controls__source_data.csv`.
- `fig06_state_dependence`: State-dependent fallback diagnostics are local and mostly abstention-boundary evidence; positive strata are exploratory and often have intervals crossing zero. Source CSV: `01_results/figures/fig06_state_dependence__source_data.csv`.
- `fig07_replication_agreement`: Independent replication reproduces all tolerance-checked quantities, the primary N=0 abstention, and the fallback LOTO estimate. Source CSV: `01_results/figures/fig07_replication_agreement__source_data.csv`.
- `fig08_tier_evidence_map`: The package satisfies provenance, measurement, controls, red-team, and replication gates, but fails primary high-support increment and transfer criteria; Tier C is the supported decision. Source CSV: `01_results/figures/fig08_tier_evidence_map__source_data.csv`.

## Claim Evidence Matrix

The top-level `evidence.csv` contains claim status, confirmatory status, artifact paths, table IDs, figure IDs, reviewer status, limitations, allowed paper wording, and forbidden wording.

## Protected Scope

- No `src/` files modified.
- No `pipelines/` files modified.
- No paper repository files modified.
- Paper repo hygiene: this run made zero paper-repo edits; a pre-existing unrelated `.gitignore` modification in the paper repo was left untouched.
- No `START_HERE.md` or `main_workflow.log` edits.

## Boundary Wording

- Use: Tier C external diagnostic boundary/stress-test; exact-support abstention; fallback sensitivity; future validation design.
- Do not use: validated criterion validity; successful domain transfer; transferable NSFC coverage guarantees from InterHub calibration; independently expert-validated coordination endpoint; NPC effect-identification claims; H3 mechanism evidence.
