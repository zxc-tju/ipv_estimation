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

## RQ011B - Moment-Level Monitor Validity Close-Out

Status: `PROVISIONAL_NULL / UNDER_IDENTIFIED` (measurement-limited close-out, PI-directed 2026-06-29). This is not a frozen manuscript monitor-validity claim and is not a clean refutation of IPV-based monitoring.

Run ID: `RQ011B_1_matched_scenario_20260625T202454_8331bd49`
Worker: `RQ011B-P5-closeout-register`
Locked SAP: `reports/plans/RQ011B_SAP_v4_moment_monitor_locked_20260629.md`
Primary result: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/W1_moment/W1_moment_results.md`

### Question

Can IPV deviation from the human cooperative-norm envelope serve as a parsimonious, interpretable, directional runtime monitor of interaction failure, flagging or preceding failure moments, on OnSite competition replay?

### Verdict

`NOT_DEMONSTRATED_ON_ONSITE`: moment-level monitor validity is `UNDER_IDENTIFIED / NULL` under the pre-registered locked SAP v4, executed as registered. The result bounds the claim as "not demonstrated on OnSite, pending an adequate failure-segment-retrieval method."

Key W1 facts:

- Predictor was built outcome-blind: per-frame signed M3 deviation with 19,044 supported frames / 245 units; 4,243 frames outside the 90% norm, split 2,300 above/passive and 1,943 below/aggressive.
- Per-frame build hash: `417b00783a96893a475f6f03de4145363a622fd24a4aa1743e882a6c14e25cc4`.
- Failure ledger: 23,521 occurrences; 23,004 timestamped; 780 timestamped and pre-window IPV-eligible.
- Primary contrast, any-failure-moment vs C1 within-interaction matched controls with onset-safe pre-window, is `UNDER_IDENTIFIED`: C1 has 0 controls, so effect, efficiency, and confidence interval are not estimable; the primary gate did not pass.
- Robustness controls cannot rescue C1: C2 ROC AUC = 0.493, eff = 0.0084; C3 eff is approximately 0; C4 eff = 0.0084. Fixed alarm false alarms are 54.2 per interaction-minute with recall 0.20. Directional label is `NON_DIRECTIONAL`.
- Specificity, LOSO, and per-category BH-FDR over m=18 are all `UNDER_IDENTIFIED`; no category is BH-significant.

### Headline Limitation

**Critical PI-mandated caveat:** the binding bottleneck is the retrieval and segmentation of interaction-failure segments, not only the IPV monitor contrast. The same measurement problem recurs across the audit chain: collision-only criteria are too sparse (19/285), broad any-failure criteria are saturated (285/285 cells), and moment-level within-interaction controls vanish (`C1 = 0`) because failure markers are too dense or ill-posed. Therefore the current OnSite "interaction-failure segment" detection/segmentation is not adequate to support a clean monitor test.

The RQ011B null is therefore **provisional and measurement-limited**. It must not be written as a clean refutation of IPV monitoring. A separate future RQ should study proper interaction-failure-segment retrieval/segmentation; the RQ011B monitor verdict should be revisited only after that measurement layer is solved. No new future-RQ folder is created by this close-out.

### Manuscript Role

RQ011B does not yield a clean accepted/rejected criterion or monitor-validity claim. Its manuscript-safe role is only a bounded statement: moment-level IPV monitoring was not demonstrated on OnSite under the current failure-segment retrieval, and the result remains pending an adequate failure-segment-retrieval method. It is convergent with RQ009's counterpart-IPV practical null and RQ003's NSFC null, but the measurement limitation above must travel with any comparison.

### Reproducibility Pointers

- Locked SAP v4: `reports/plans/RQ011B_SAP_v4_moment_monitor_locked_20260629.md`
- Moment W1 results: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/W1_moment/W1_moment_results.md`
- W1 derived CSVs: `data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/W1_moment/`
- Per-frame build ledger: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/perframe_build/perframe_build_ledger.md`
- Close-out summary: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/CLOSEOUT_RQ011B.md`
- Audit chain: cell-level collision-sparse -> cell-level any-failure saturated -> moment-level C1 no controls.
