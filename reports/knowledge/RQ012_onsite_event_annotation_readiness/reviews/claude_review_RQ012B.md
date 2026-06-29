# Claude Code Review — RQ012B (OnSite automatic-event harm association)

Status: filed (2026-06-29)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ012B_2_harm_association_20260627T095847+0800_8454ad93` (COMPLETE; supersedes `RQ012B_1_event_harm_...`; run after RQ009 M3 froze).
Reader entry: `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/90_report/index.html`.

## Verdict

Concur with the bounded / power-limited null. AV deviation from the (context-conditioned) human IPV
envelope does **not** robustly, IPV-specifically predict realised harm on OnSite. The pipeline was sound
(M3 scorer reconstructed parity-perfect; pre-registered before any outcome join; cluster-aware permutation,
placebo, M2/context and exposure controls; independent replication AGREES).

## Key findings

| Endpoint / test | Result | Reading |
|---|---|---|
| InterHub→OnSite support | only 19,044/67,861 moments in-support (72% OOD-abstained) | Transfer heavily OOD-limited. |
| Primary deviation→safety | Spearman ≈ −0.12 (p≈0.06–0.08, ns); perm p=0.743 | Directionally sensible but not significant, not IPV-specific. |
| Full 64-test behavioural battery | no channel robustly SUPPORTED | Abrupt/jerk/comfort null; all at BH edge or fail a control. |
| Near-miss/contact (E09) | nominal IRR≈1.22 but **fails M2 control** | Context-explained, not IPV-specific. |
| too-passive→deadlock (E16) | IRR≈1.50, survives all controls, but **underpowered (52/243), BH edge** | Unconfirmed hint only. |
| Power | ~19 effective team clusters; harms rare (~8% units) | Only large effects detectable; null = "no detectable effect". |

## Manuscript role — DEFERRED

Per PI: **R5 / OnSite is not discussed pending proper data.** Do not put this null in the manuscript. The
`too-passive → deadlock` signal is an **internal, unconfirmed hypothesis** for a future powered test, not a
paper claim. Automatic-event counts are never a scientific outcome on their own.

## Reproducibility / process

M3 parity 0.0 vs frozen RQ009; estimator sha-pinned; analysis n=245 units; BH-FDR over 64 tests; negative
controls valid; independent clean-room recheck = AGREE_WITH_CAVEATS. Honest documentation of the
kinematic-only vs over-absorbing official-subscore baseline.

## Recommendation

Hold as a power-limited/OOD-limited null; do not feature externally. Any future OnSite consequence test
needs (a) the RQ011B failure-segment measurement solved, (b) more power (more harm events / less clustering)
or a different outcome set, and (c) on-support coverage beyond 28%. Revisit with real/adequate data, per PI.

## Source pointers

- `02_process/09_report/CONCLUSIONS.md`, `02_process/04_harm_association/summary.json`
- `02_process/08_replication/full_battery_recheck_report.md`; `02_process/10_final_review/final_review.md`
