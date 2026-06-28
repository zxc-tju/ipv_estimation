# RQ009 Traceability

Run ID: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de`
Overall status: `COMPLETE`

## Phase Ledger

| Phase | Owner role | Key artifacts | Final status |
|---|---|---|---|
| 0_bootstrap | bootstrap / provenance-init | run skeleton, derived root, manifests, plan snapshot, denylist, execution status | PASS |
| 1_plan_review | independent plan review | plan review, addendum, rereview, PI signoff | PASS |
| 2_provenance | provenance audit | input audit, schema audit, leakage field screening | PASS |
| 3_features | feature construction | feature dictionary, hw4 target fetch, feature matrix, matrix audit | PASS |
| 4_calibration | model calibration | M0-M5 plus controls, conformal records, OOD gate, calibration audit | PASS |
| 5_evaluation | evaluation | coverage, width, pinball, Winkler, subgroup coverage, LODO | PASS |
| 6_m3_vs_m4 | formal M3-vs-M4 gate | M3-vs-M4 verdict, exploration synthesis, horizon sweep, dependency reconciliation | PASS; no escalation; robust practical null |
| 7_perturbation | robustness / transfer | perturbation sensitivity and guard-tune robustness checks | PASS |
| 8_review | independent review | end-to-end contract/leakage/statistical/null review | PASS |
| 9_red_team | red team | leakage and null-effect attack report | PASS; no exploit |
| 10_replication | replication | independent replication and null reconciliation | COMPLETE; practical null reconciled |
| 11_report | report production | bilingual offline report, figures, evidence table, report gate | PASS |
| 12_final_review | final no-blocker review | final review gate and registrar readiness | PASS; `ready_to_register=true` |
| 12b_registration | registrar | knowledge `decision.md`, registry/log updates, final run-root status | PASS |

## Final Decision Pointers

- Knowledge decision: `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
- Evidence table: `evidence.csv`
- Final review gate: `02_process/12_final_review/final_review_gate.json`
- Execution status: `execution_status.json`

## Reader-Facing Package

- Entry page: `00_entry/index.html`
- English report: `90_report/index.html`
- Chinese report: `90_report/index.zh.html`
- Figure manifest: `01_results/figures/figure_manifest.csv`
- Figures: `01_results/figures/c1_validity_envelope.*` through `c7_robustness_perturbation.*`

## Claim Trace

| Decision | Evidence rows | Primary artifacts |
|---|---|---|
| Marginal CQR validity accepted | C1, P4 | `02_process/05_evaluation/metrics_summary.csv`, `01_results/figures/c1_validity_envelope.png` |
| Counterpart-IPV practical null accepted | C2, C3, C4, C7, P9, P10 | `02_process/06_m3_vs_m4/m3_vs_m4_verdict.json`, `02_process/06_m3_vs_m4/exploration/exploration_verdict.md`, `02_process/06_m3_vs_m4/exploration/reconcile/dependency_reconcile.md`, `02_process/10_replication/replication_reconcile.md` |
| Context/kinematics dominate sharpness accepted | C2, C5 | `02_process/06_m3_vs_m4/m3_vs_m4_verdict.json`, `01_results/figures/c2_context_dominates.png`, `01_results/figures/c5_self_anchor_marginal.png` |
| Primary counterpart-conditioning mechanism not supported | C2, C3, C4, C7, P8, P9, P10 | decision record plus final report |
| Limitations retained | C6, C6_ATOM | `02_process/05_evaluation/metrics_long.csv`, `02_process/05_evaluation/lodo_results.csv`, `02_process/09_red_team/red_team_gate.json` |

## Final Headline Values

- M3 coverage 80/90/95: `0.816215 / 0.898666 / 0.949635`.
- M3 90% width/Winkler: `1.016152 / 1.422935`.
- M3-minus-`ipv_removed` 90% Winkler: `-0.000211426`, CI `[-0.001886180, 0.001504979]`, case sign p `0.862943`.
- M2 vs M0 90% width/Winkler: `-42.271% / -35.612%`.
- Subgroup rows outside +/-3 pp: `126/264`; LODO M3 90% coverage: `0.748790..0.991481`; target exact-zero atom: `0.215509`.
