# Codex Review: RQ011 OnSite Full-Universe Readiness

Status: review-complete; current rerun PASS as `READY_WITH_FROZEN_EXCLUSIONS`; no knowledge-layer decision yet.
Review date: 2026-06-24.

## Scope

Reviewed current study package:

- `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/`

Reviewed superseded package:

- `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_1_onsite_readiness_20260623T104838+0800_20aaee57/`

Primary current evidence read:

- `90_report/index.html`
- `README.md`
- `TRACEABILITY.md`
- `execution_status.json`
- `report_content.md`
- `final_review_summary.md`
- `review_summary.md`
- `final_readiness_status.json`
- `red_team_summary.md`
- `red_team_verdict.json`
- `replication_summary.md`
- `replication_compare.json`
- `analysis_unit_decision.md`
- `analysis_unit_decision.json`
- `repeated_runs_summary.md`

## Overall Verdict

RQ011_2 is strong enough to support OnSite full-universe readiness with a frozen
replay exclusion. The valid analysis unit is algorithm-scenario, with a full
outcome universe of 300 cells and a replay/trajectory/IPV universe of 285 clean
cells after excluding T19 for replay-only purposes. The suspended RQ011_1 package
must not be cited.

Paper-safe phrasing:

> The OnSite competition package supports a matched-scenario
> algorithm-scenario analysis universe. Scores cover the full 20-team x
> 15-scenario outcome universe, while replay-dependent analyses must use a
> frozen 285-cell replay universe that excludes T19 because no unique T19-owned
> vehicle-3190 replay/session can be identified.

## Claims That Can Be Carried Forward

1. RQ011_1 was suspended because the data package was incomplete and should be
   treated as non-citable.
2. RQ011_2 supersedes RQ011_1 and uses the complete OnSite competition package.
3. Official score/deduction PDFs are authoritative for outcomes. Score 0 means
   collision, the 15 scenarios are matched across teams, and each team has one
   official autonomous run.
4. The primary outcome universe is full_300: 20 teams x 15 scenarios, with no
   team excluded from score/deduction outcomes.
5. The replay/trajectory/interface/IPV universe is clean_285: T19 is excluded
   only because the needed T19-owned replay/session could not be uniquely
   identified.
6. The resolved replay mapping consists of 210 unique clean cells, 75
   conflict-resolved promoted cells, and 15 wrong-folder candidates tied to T19
   that remain excluded from replay use.
7. The T19 exclusion has moderate selection implications. T19 has 9 collisions
   in 15 scenarios, while replay_285 has 24 collisions in 285 cells and full_300
   has 33 collisions in 300 cells.
8. Repeated-run, seed-level, run-level, and causal claims are not identifiable
   from this package. The supported unit is algorithm-scenario.
9. The interface to later IPV work is partially ready, but selected-counterpart,
   opportunity, and onset thresholds are not frozen by RQ011.

## Claims To Reject Or Defer

- Do not exclude T19 from full outcome analyses. T19 is excluded only for
  replay/trajectory/IPV surfaces.
- Do not claim full_300 replay coverage or full_300 IPV coverage.
- Do not claim repeated-run effects, seed effects, run-level effects, or
  algorithm superiority.
- Do not claim any IPV-outcome association or causal relationship from RQ011.
- Do not treat the field/interface thresholds as final for downstream RQ012 or
  IPV analysis; they remain partial-readiness signals.

## Quality And Compliance Notes

The final review summary reports PASS with zero concerns or failures. The
independent replication fully agrees with the full_300 and clean_285 universes,
mapping counts, collision counts, and final status. The red-team verdict has no
blockers and supports the final readiness leaf after correcting the earlier
decision-tree treatment of run-level non-identifiability.

Two cleanup issues remain. First, `TRACEABILITY.md` still contains stale
phase-placeholder wording even though the run is complete. Second, some earlier
review material uses pre-correction wording around field sufficiency and strict
clean counts. The current source of truth is `final_readiness_status.json`,
`final_review_summary.md`, and the 90_report.

## Knowledge-Layer Action

Recommended decision state: accept RQ011_2 as
`READY_WITH_FROZEN_EXCLUSIONS`, register full_300 for outcomes and clean_285 for
replay/IPV-dependent work, mark RQ011_1 non-citable, and require all downstream
claims to preserve the T19 selection caveat.
