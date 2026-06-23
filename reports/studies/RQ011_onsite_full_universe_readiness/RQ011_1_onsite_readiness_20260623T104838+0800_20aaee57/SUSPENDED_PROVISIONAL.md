# ⚠ RUN SUSPENDED — FINDINGS PROVISIONAL (do not cite)

**Status:** SUSPENDED on 2026-06-23 at Phase 11, by user request.
**RUN_ID:** RQ011_1_onsite_readiness_20260623T104838+0800_20aaee57

## Why suspended
The OnSite source data under `data/onsite_competition/` is stored in OneDrive, and the
cloud sync was INCOMPLETE at the time of this run. The local copy was therefore missing
files. **All readiness findings in this run are PROVISIONAL and are likely confounded by
incomplete local data — they are NOT confirmed properties of the OnSite dataset.**

Corroborating evidence of incomplete local data (Phase 2 inventory `02_process/02_inventory/`):
- 127 files local-only, 16 archives not extracted,
- 10 manifest sessions with EMPTY raw session folders (data present only in the top5 subset).

## Nothing is frozen
- The final readiness status was NOT finalized. Phase 10 mechanically computed
  `BLOCKED_MAPPING`; the user approved a relabel to `RUN_LEVEL_NOT_IDENTIFIABLE`, but the
  relabel worker was stopped before completing. Treat NO leaf as final.
- Phases not run: 12 (replication), 13 (figures + HTML report), 14 (final review + registry).

## Provisional findings (suspect until re-run)
- run-level not identifiable; unit algorithm×scenario; matched-scenario only (no repeated runs).
- field sufficiency FAILED (S1=105, S2=150); blocking missing/unreliable fields:
  success_failure, collision_rule_violations, intervention_fallback_replanning,
  script_version_seed; map_route_role_scenario partial.
- high MNAR; clean set biased.

## On resume (after OneDrive sync verified complete)
1. Verify data completeness FIRST: no empty raw session folders, archives extracted,
   file/row counts reconcile with `00_manifest/` and `score_team_coverage.csv`.
2. Start a NEW run (new RUN_ID); re-do from Phase 2 inventory onward.
3. Reuse the Phase-1 plan + operational addendum in `02_process/01_plan_review/`
   (field dictionary, identity contract, mapping taxonomy, deterministic decision tree) —
   these were independently verified and are not data-dependent.

## User-supplied data semantics (CRITICAL for re-run — undercuts the "field insufficiency" verdict)
- **Per-run log** `onsite_YYYY-MM-DD_HH-MM-SS.log` records one run's progress and the
  `avalgorithm` flag: **1 = autonomous**, else manual. Competition is autonomous-only, so
  default all units to autonomous. → `intervention_fallback_replanning` is DERIVABLE/defaultable,
  not missing.
- **One run per team**, but **all teams face the EXACT SAME 15 scenarios**. → no repeated runs
  (confirms P6); cross-team scenarios are IDENTICAL → algorithm×scenario matched comparison is
  strong.
- **Per-team diagnostic PDF**, e.g. `data/onsite_competition/raw/beijing/5-BIT_Site/诊断报告.pdf`,
  gives per-scenario scores + sub-item breakdown + deduction reasons. **Scenario score 0 = a
  COLLISION in that scenario.** → `success_failure` (0 = fail) and `collision_rule_violations`
  (0 = collision; deduction reasons = violations) are DERIVABLE. Analyze the deduction reasons
  closely. P5 likely flagged these missing only because the PDFs were cloud-only/unparsed.
- Net: 3 of the 4 P5 "blocking missing" fields (success_failure, collision_rule_violations,
  intervention_fallback_replanning) are recoverable from PDF + per-run logs once sync completes;
  only `script_version_seed` remains open. Next run MUST parse `诊断报告.pdf` and `onsite_*.log`.
