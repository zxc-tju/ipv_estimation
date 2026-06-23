# RQ007 Decision: Interaction-Conditioned IPV Estimability

Status: ACCEPTED

Run ID: `RQ007_1_ipv_estimability_20260622T155229Z_289d9a99`  
GIT_HEAD: `38063a2ff9cdc717098cf3f821c2bb162a0ac1d9`  
PLAN_SHA256: `7c33f7be76cce64fe2e5d17e4cd6be72435c51216c7a779c2e29f7626912f3b8`  
Registered UTC: `2026-06-23T03:57:07Z`

Evidence matrix: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/11_final_review/claim_evidence_matrix.csv`  
Final conclusions: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/11_final_review/conclusions.md`  
Report: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/90_report/index.html`  
Entry point: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/00_entry/index.html`

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Provenance | PASS | `execution_status.json`, `02_process/02_inventory/gate_007_0_status.json` |
| Estimability | PASS | `02_process/04_estimability/` |
| Controls | PASS | `02_process/05_controls/mechanical_control_status.json` |
| Review | PASS | `02_process/08_review/` |
| Red team | PASS | `02_process/09_red_team/red_team_status.json` |
| Replication | PASS | `02_process/10_replication/replication_status.json` |
| Final no-blocker review | PASS | `02_process/11_final_review/final_no_blocker_review_status.json` |

Paper edits may use ONLY the accepted claims in this decision file. Do not use unstaged exploratory notes, held_out data, PET/intensity/order/priority/outcome fields, or paper-repository edits as claim sources for RQ007.

## Accepted Claims

### C1: Interaction-conditioned IPV estimability concentration gap

Decision: ACCEPTED  
Strength: MODERATE with strong boundaries  
Scope: proximity-bounded; development/guard only; held_out sealed; no map/lane fields used; estimator-input reruns are a sanity check only.

Frozen conclusion text:

> Within causal interaction-opportunity windows (cv_cpa_conflict), the per-frame IPV estimator CONCENTRATION INDEX is lower (IPV more identifiable) than history-length expectation by a TOTAL of about 0.13 index units (development -0.132, guard -0.129; independently replicated -0.134 / -0.133). This elevation is TIME-LOCKED (eliminated by time-shift: gap -> +0.006) and COUNTERPART-SPECIFIC (eliminated by counterpart permutation: +0.021; reversed under re-estimated counterpart switch: +0.122), so it is NOT explained by history length, arbitrary pairing, or alignment artifacts. HOWEVER, the MAJORITY (about -0.096) is attributable to spatial PROXIMITY (a nearby non-conflicting actor reproduces about -0.096); the CONFLICT-GEOMETRY-SPECIFIC increment beyond proximity is SMALL but nonzero, about -0.032 to -0.036 with case-clustered CIs excluding zero. Robust to analysis-level perturbations. Boundaries: development/guard only (held_out sealed); no map/lane fields used; estimator-input reruns are a SANITY CHECK (recompute mismatch ~0.11), not rigorous proof.

Evidence pointers:
- `02_process/11_final_review/claim_evidence_matrix.csv` rows `C1`, all `verified=true`.
- `02_process/05_controls/controls_results.csv` for total gap, time shift, counterpart permutation, nearby non-conflicting, and distant no-opportunity controls.
- `02_process/09_red_team/red_team_cluster_control_probes.csv` for proximity-bounded conflict-geometry increment and clustered CIs.
- `02_process/10_replication/replication_compare.csv` for independent replication.
- `01_results/figure_manifest.csv` figures `F1` and `S1`; report `90_report/index.html`.

### C2: Estimability is not behavioural settling

Decision: ACCEPTED  
Strength: SUPPORTED  
Scope: construct-separation claim; not a causal IPV-change or normative-truth claim.

Frozen conclusion text:

> Estimability (index concentration) is NOT equivalent to behavioural settling: under low index (estimable) the IPV current-estimate mean keeps changing (|dtheta| ~0.30 ego / ~0.31 counterpart). High index != IPV=0; full-window mean != current IPV.

Evidence pointers:
- `02_process/11_final_review/claim_evidence_matrix.csv` rows `C2`, all `verified=true`.
- `02_process/04_estimability/mean_dynamics_separation.md`.
- `01_results/figure_manifest.csv` figure `F3`; report `90_report/index.html`.

### C3: Episode IPV summary definition dependence

Decision: ACCEPTED  
Strength: SUPPORTED  
Scope: summary-definition sensitivity only; does not choose the preferred downstream summary rule.

Frozen conclusion text:

> Episode-level IPV summaries are materially definition-dependent: all-valid-frame mean vs interaction-active mean differ by ~0.26 rad on average and flip sign in ~22% of cases; estimability-weighted is closer to all-valid (~7% sign flips, corr ~0.91). The 'episode IPV' is not a definition-free quantity.

Evidence pointers:
- `02_process/11_final_review/claim_evidence_matrix.csv` rows `C3`, all `verified=true`.
- `02_process/07_summary_sensitivity/summary_sensitivity.csv`.
- `02_process/10_replication/replication_compare.csv` row `episode_summary_sensitivity`.
- `01_results/figure_manifest.csv` figure `F4`; report `90_report/index.html`.
