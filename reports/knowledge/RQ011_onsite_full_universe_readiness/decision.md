# RQ011 Decision: OnSite Full-Universe Readiness

Status: ACCEPTED — `READY_WITH_FROZEN_EXCLUSIONS` (knowledge-layer freeze, human-directed 2026-06-24). Readiness/scope decision only; not an outcome or IPV finding.

Run ID: `RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5` (supersedes `RQ011_1_...20aaee57`, non-citable)
Plan SHA-256: `13142fc4ebdb8636ec099323e04e1428a09ac91a2399aa8b321f83cd5e6d3e10`
Basis for freeze: final review PASS (zero concerns); independent replication full agreement on universes/mapping/collision counts/status; red team no blockers (RT10 decision-tree fix); `reviews/claude_review.md` and `reviews/codex_review.md` both concur. Frozen at PI direction.

## Accepted Claims

| ID | Claim |
|---|---|
| RQ011-KC-UNIT | The valid primary analysis unit is `algorithm×scenario` (matched scenario; algorithm_id == team_id, case_id == scenario_id in the current inventory). |
| RQ011-KC-OUTCOME-300 | Outcome universe = `full_300` (20 teams × 15 scenarios); official score/deduction/collision fields complete; score 0 = collision. No outcome-side exclusion. |
| RQ011-KC-REPLAY-285 | Replay/trajectory/interface/IPV universe = `clean_285`; `T19` is excluded **replay-only** because no unique T19-owned vehicle-3190 replay/session can be identified (210 unique clean + 75 conflict-resolved promoted cells). |
| RQ011-KC-SELECTION | The T19 replay exclusion carries a moderate selection caveat: collisions T19 9/15, replay_285 24/285, full_300 33/300 (replay collision rate ≈8.4% vs 11.0% full). |
| RQ011-KC-IDENTIFIABILITY | Run-level, repeated-run, seed-level, independent-case, full_300 replay/IPV coverage, and causal effects are NOT identifiable. |

## Rejected Or Deferred Claims

| Claim | Reason |
|---|---|
| Exclude T19 from outcome analyses | T19 is excluded for replay/IPV only, never outcomes. |
| full_300 replay or full_300 IPV coverage | Replay universe is the 285 clean cells. |
| Repeated-run / seed / run-level effects or algorithm superiority | Not identifiable from this package. |
| Any IPV–outcome association or causal relationship | Out of scope for a readiness study. |
| Field/interface thresholds final for RQ012 / IPV work | Partial-readiness only; counterpart/opportunity/onset thresholds not frozen by RQ011. |

## Governance Flags

- The final `READY_WITH_FROZEN_EXCLUSIONS` leaf was set by a **PI-authorized RT10 fix** (2026-06-24, `pi_authorized_correction_applied: true`) re-interpreting `run_level_claims_allowed=false` from a terminal block into a scope boundary. Defensible, but it is a human-authorized re-grade of a red-team finding — keep the authorization on record.
- `evidence.csv` is currently header-only (empty); readiness checks live in process files. Recommend populating the evidence ledger.
- `git_head` (`32ebf75…`) differs from the 38063a2 baseline of the other RQ runs.

## Paper Handoff

Use as a readiness/scope decision that supplies the matched-scenario universe to RQ012, RQ011B, and RQ009. Always attach the no-run-level and T19 replay-selection caveats. RQ011_1 is non-citable.
