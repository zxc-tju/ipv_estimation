# Phase 9 Tier Review

Worker: `RQ003_phase9_tier_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`  
Generated UTC: `2026-06-20T15:45:00Z`  
Decision: `Tier B`

## Identity Gate

PASS. The run root and tier directory exist. `02_process/00_meta/run_manifest.json` reports the requested `RUN_ID`, and `02_process/00_meta/plan_sha256.txt` equals `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1`. Red-team v3 reports `PASS_NO_BLOCKERS` and `cleared_for_tier_review=true`.

Evidence:

- `02_process/00_meta/run_manifest.json`
- `02_process/00_meta/plan_sha256.txt`
- `02_process/16_red_team_fixes/red_team3/red_team3_status.json`

## Tier Decision

The run is Tier B: interpretable diagnostic external evidence with no robust independent held-out increment. It is not Tier A because multiple required robustness criteria fail. It is not Tier C because the run passed the identity, provenance, measurement, freeze, red-team-closure, and replication-integrity gates, and the package remains useful as bounded diagnostic evidence for the approved top-five cohort.

Allowed reader-facing framing:

> no robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated (power-limited, top-five cohort, N=53; apparent favorable direction not IPV-specific)

## Criterion Assessment

| Criterion | Result | Rationale | Evidence |
|---|---|---|---|
| Gate -1 PASS | PASS | Approved top-five cohort has clean mapping and trajectory presence; full 20-team universe remains not analysis-ready. Coordination is an official/generated report score, not an independent human label. | `02_process/02_gate_minus1/gate_minus1_status.json`; `02_process/03_gate_minus1_review/gate_minus1_review_status.json` |
| Gate 0 PASS | PASS | Sign contract, outcome firewall, rolling-to-rolling slicing, and InterHub-only conditional norm boundary passed; real optimizer condition was closed in Phase 4. | `02_process/04_gate0_measurement/gate0_status.json`; `02_process/05_gate0_review/gate0_review_status.json`; `02_process/08_directional_ipv/g0r_cond_001_status.json` |
| Frozen analysis uncontaminated | PASS | Rerun freeze review is outcome-clean and quarantines prior contaminated artifacts; single confirmatory comparison is frozen. | `02_process/06_analysis_freeze/analysis_freeze.yaml`; `02_process/07_freeze_review/freeze_review_status.json` |
| Directional IPV consistent in outcome-independent safe subsets | FAIL | S1 and S2 are exactly the primary 53 cells, so agreement is duplicate rather than independent; S3 has n=6 and null/reverse behavior. | `01_results/tables/confirmatory_results.csv`; `01_results/tables/confirmatory_results_interpretation.csv` |
| At least two genuinely distinct safe subsets agree | FAIL | The mechanical agreement count is vacuous because S1=S2=primary, while S3 does not support the direction. | `02_process/10_confirmatory_analysis/confirmatory_analysis_report.md`; `02_process/16_red_team_fixes/interp_fix/interpretation_correction.md` |
| Leave-team-out stable | FAIL | Corrected primary LOTO is favorable numerically but nonsignificant: delta Spearman +0.136833, p=0.30, CI [-0.038781, +0.305797]. Replication verifies direction, not stability or robustness. | `01_results/tables/confirmatory_results.csv`; `02_process/17_independent_replication/replication2/replication2_status.json` |
| Leave-scene not contradicting | FAIL | LOSO delta Spearman is approximately zero (+0.016732), with MAE and CV-R2 deltas negative; scenario generalization is not established. | `01_results/tables/confirmatory_results.csv`; `01_results/tables/confirmatory_results_interpretation.csv` |
| Stable incremental utility over the prespecified capacity-matched baseline | FAIL | The primary increment is nonsignificant and non-generalizing; active interpretation states no robust incremental predictive utility was demonstrated. | `02_process/10_confirmatory_analysis/confirmatory_analysis_report.md`; `02_process/16_red_team_fixes/red_team3/red_team3_report.md` |
| Negative controls behave as expected | FAIL | Controls matching or exceeding the primary delta Spearman include future_leaky_full_window_ipv +0.231817, ipv_time_shuffle +0.196823, counterpart_swap +0.168441, role_flip +0.136833, and sign_flip +0.136833. State_shuffle and wrong_state failed degradation expectations. | `01_results/tables/negative_controls.csv`; `02_process/11_negative_controls/negative_control_report.md` |
| Blocking red-team findings closed | PASS | Red-team v3 closed all RT2 blockers and found no new blockers, while keeping clearance distinct from a tier decision. | `02_process/16_red_team_fixes/red_team3/red_team3_status.json`; `02_process/16_red_team_fixes/red_team3/red_team3_report.md` |
| Independent replication passed | PASS | Replication2 reproduced the corrected N=53 favorable direction exactly under reported-alpha refit and directionally under independent training-tuned analysis. | `02_process/17_independent_replication/replication2/replication2_status.json`; `02_process/17_independent_replication/replication2/independent_replication2_report.md` |
| Blind-annotation claim backed or excluded | PASS | H3 is blocked because no real two-human labels exist. No blind-annotation claim is included in the Tier basis. | `02_process/12_blind_annotation/annotation_status.json` |

## Boundary Evidence

- Scope is the approved top-five cohort only. The full 20-team universe is explicitly not analysis-ready.
- NPC matching is boundary-only. The available fields do not identify a valid pre-onset matching design; future wording is limited to "matched opportunity structure" if independent pre-onset evidence is later obtained.
- State-dependence is exploratory boundary mapping only. There are 90 interpretable rows, minimum FDR q=0.628317, and no row with q<=0.10.
- Conformal calibration is InterHub-only. The NSFC package can reference this as a boundary and abstention rule, not as a guaranteed NSFC coverage statement.

## Final Tier Rationale

Tier A is rejected because safe-subset independence, LOTO stability, LOSO generalization, stable incremental utility, and negative-control specificity all fail. Tier C is not assigned because identity/provenance/measurement/freeze gates passed, red-team blockers are closed, and independent replication verifies the corrected implementation path. The appropriate handoff is Tier B diagnostic evidence: the run can support a bounded, power-limited statement that the corrected top-five NSFC package did not demonstrate robust incremental predictive utility relative to the prespecified kinematic+safety baseline.
