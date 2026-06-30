# RQ011 Knowledge Synthesis

Status: consolidated from the frozen `decision.md` (ACCEPTED — `READY_WITH_FROZEN_EXCLUSIONS`, human-directed 2026-06-24). Readiness/scope decision only; not an outcome or IPV finding. `decision.md` is the canonical claim ledger.

## What was accepted

- `RQ011-KC-UNIT`: the valid primary analysis unit is `algorithm×scenario` (matched scenario).
- `RQ011-KC-OUTCOME-300`: outcome universe = `full_300` (20 teams × 15 scenarios); official score/deduction/collision fields complete; score 0 = collision; no outcome-side exclusion.
- `RQ011-KC-REPLAY-285`: replay/trajectory/interface/IPV universe = `clean_285`; `T19` excluded **replay-only** (no uniquely identifiable T19 replay/session).
- `RQ011-KC-SELECTION`: the T19 replay exclusion carries a moderate selection caveat (replay collision rate ≈8.4% vs 11.0% full).
- `RQ011-KC-IDENTIFIABILITY`: run-level, repeated-run, seed-level, independent-case, full_300 replay/IPV coverage, and causal effects are **not** identifiable.

## Governance flags to keep on record

The final readiness leaf was set by a PI-authorized RT10 decision-tree fix (re-interpreting `run_level_claims_allowed=false` as a scope boundary, not a terminal block) — a human-authorized re-grade of a red-team finding. `evidence.csv` is header-only and should be populated. `RQ011_1` is non-citable.

## Downstream

Supplies the matched-scenario universe to RQ012, RQ011B, and RQ009; always attach the no-run-level and T19 replay-selection caveats. The RQ011B moment-level monitor close-out is now tracked in its own folder `RQ011B_onsite_moment_monitor/` (canonical record remains the `## RQ011B …` section of this RQ's `decision.md`).

Sources: `decision.md`; `reviews/claude_review.md`; `reviews/codex_review.md`.
