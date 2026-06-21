# Unresolved Source Questions

Worker: `RQ003_phase1_gate_minus1_audit_001`  
Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

1. Score export generator: no script or platform export command was found that generated `archived/onsite_competition_results_legacy/score_beijing_abilities.csv` or `archived/onsite_competition_results_legacy/score_shanghai_abilities.csv` from SQL/PDF sources.
2. Formal rubric: diagnosis PDFs show metric names and submetric details, but no formal rubric/rater instruction document was found.
3. Coordination source: report evidence supports an official generated/rule/kinematic metric; no rater-level coordination records or scale document prove human expert coordination ratings.
4. Beijing vs Shanghai rubric identity: same report schema and 0-100 dimensions are visible, but a formally identical rubric cannot be established.
5. Judge comparability: SQL shows area-specific judge names (`通州跟车裁判1/2` vs `同济跟车裁判1/2`), so same judges are not established.
6. T18/T19 replay organization: SQL identifies task `6937` as `BIT_OnSite`, but the materialized session is under `北京赛区/6-223`; this requires human/source-owner resolution before using the full 20-team universe.
7. T19/223 replay: the T19 diagnosis PDF and archived CSV agree on a 29.6 total-score profile, but no `6949-*` replay folder was found and the score vector does not match any 15-scene SQL referee task vector.
8. Partial sessions: T12 official task `6953` has 13/15 case IDs in the available replay; T21 task `6934` has 14/15 case IDs. Decide whether to exclude missing cases or locate complete logs before any full-universe analysis.
9. Top-five scope: Gate -1 PASS is scoped to the top-five plan cohort. Full-universe analysis needs a separate corrected inclusion/exclusion spec.
