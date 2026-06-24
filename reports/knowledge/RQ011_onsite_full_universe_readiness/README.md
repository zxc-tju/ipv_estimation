# RQ011: OnSite Full-Universe and Run-Level Readiness

Status: readiness complete — `READY_WITH_FROZEN_EXCLUSIONS` (algorithm×scenario; T19 replay-excluded)

Execution layer: `reports/studies/RQ011_onsite_full_universe_readiness/`
Latest run: `RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5` (supersedes `RQ011_1_...20aaee57`)
Plan: `reports/plans/RQ011_plan_v0_onsite_full_universe_readiness_20260622.md`

## Research Question

Is the OnSite competition dataset ready for matched-scenario analysis, and at what analysis unit, with what
exclusions and identifiability boundaries?

## Current Interpretation

Ready at the matched-scenario unit `algorithm×scenario`. Outcome universe = `full_300` (official
scores/collisions/deductions); replay/IPV universe = `clean_285` with `T19` frozen out (moderate residual replay
selection bias). Run-level and repeated-run claims are NOT identifiable. The final readiness leaf came via a
PI-authorized RT10 decision-tree fix. `evidence.csv` is currently empty and should be populated. See
`reviews/claude_review.md`. Readiness/scope result only — not an outcome or IPV finding.
