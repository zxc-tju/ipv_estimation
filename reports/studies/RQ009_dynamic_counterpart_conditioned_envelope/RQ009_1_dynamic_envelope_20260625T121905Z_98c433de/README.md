# RQ009 Dynamic Counterpart-Conditioned Envelope

Run ID: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de`
Status: `COMPLETE`
Registered by: `RQ009-W12b-registrar`

## Identity

| Field | Value |
|---|---|
| Git head | `4aef4d22bb639bf003c48094607c970e55445d5f` |
| Plan SHA-256 | `b8c027a717af08cd70de6ef2b5221b387323f1131623f39fa5fd688f89baa254` |
| Run root | `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de` |
| Derived root | `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de` |

## Final Decision

Knowledge decision: `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`

- ACCEPTED: the online IPV envelope is marginally valid under CQR. M3 coverage at 80/90/95 is `0.816215 / 0.898666 / 0.949635`; M3 90% width is `1.016152` and Winkler is `1.422935`.
- ACCEPTED NEGATIVE/NULL: counterpart-conditioned IPV adds no practically meaningful, generalizing information beyond context for ego future IPV. The 90% M3-minus-`ipv_removed` Winkler effect is `-0.000211426` (`-0.014856%`), with case-cluster CI `[-0.001886180, 0.001504979]` and case sign p `0.862943`.
- ACCEPTED: context/kinematics dominate sharpness. At 90%, M2 vs M0 width and Winkler change by `-42.271%` and `-35.612%`; M4 self-anchor is marginal.
- NOT SUPPORTED: the primary hypothesis that counterpart conditioning materially sharpens or shifts the envelope.

## Reader Package

- Entry: `00_entry/index.html`
- English report: `90_report/index.html`
- Chinese report: `90_report/index.zh.html`
- Figure manifest: `01_results/figures/figure_manifest.csv`
- Evidence table: `evidence.csv`
- Report gate: `02_process/11_report/report_gate.json`
- Final review gate: `02_process/12_final_review/final_review_gate.json`

## Final Gates

| Gate | Status |
|---|---|
| Provenance/schema | PASS |
| Feature matrix and matrix audit | PASS |
| Calibration and calibration audit | PASS |
| Evaluation | PASS |
| M3-vs-M4 | PASS, no escalation |
| Exploration/horizon/reconciliation | robust practical null |
| Perturbation | PASS, validity/null robust |
| Independent review | PASS |
| Red team | PASS, no exploit |
| Replication | reconciled |
| Report gate | PASS |
| Final review | PASS, `ready_to_register=true` |

## Required Limitations

- Conditional subgroup coverage is uneven: `126/264` supported subgroup rows outside +/-3 pp.
- LODO M3 90% coverage ranges `0.748790..0.991481`.
- Exact-zero target atom is `273819/1270566 = 0.215509`, qualifying 80% endpoint-tie interpretation.
- Claims are associational and tied to this IPV operationalization, feature contract, support gate, and InterHub-derived data.
