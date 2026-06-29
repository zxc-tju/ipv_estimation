# Claude Code Review — RQ011B (OnSite matched-scenario / moment-level monitor)

Status: filed (2026-06-29)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ011B_1_matched_scenario_20260625T202454_8331bd49` (closed out 2026-06-29; verdict `PROVISIONAL_NULL_UNDER_IDENTIFIED_MEASUREMENT_LIMITED`).
Reader entry: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/90_report/index.html`.

## Verdict

Concur with the provisional / under-identified null. RQ011B did **not** demonstrate (or refute) IPV
monitor validity on OnSite — the limiting factor is that the **interaction-failure segment retrieval /
segmentation is itself inadequate**, so the test could not be properly set up. This is a measurement
bottleneck, not a clean refutation. The process was rigorous (multi-round pre-registration, PI-locked SAP
v1→v4, independent review, replication).

## Key findings

| Item | Result | Reading |
|---|---|---|
| Criterion well-posedness | collision too sparse (19 events); any-failure saturated 285/285 | Cell-level criterion ill-posed; moved to moment-level. |
| Moment-level support | 19,044 supported frames / 245 units; **C1 within-interaction controls = 0** | The primary contrast had no usable controls → under-identified. |
| Monitor signal | C2 AUC 0.49; fixed alarm 54.2 false alarms/interaction-minute, recall 0.20 | No usable runtime discrimination on this measurement. |
| Decision | gate_pass false; non-directional; no BH-significant category | Provisional null, measurement-limited. |

## Manuscript role — DEFERRED

Per PI: **OnSite / R5 is not discussed until adequate data/measurement exists.** Do not use RQ011B as an
R5 claim (positive or negative) in the manuscript. It bounds nothing for the paper yet; it identifies a
prerequisite measurement problem.

## Recommendation

Treat the bottleneck as a separate research question: **interaction-failure segment retrieval/segmentation**
on OnSite must be solved (proper onset-defined failure segments + valid within-interaction controls) before
the monitor-validity question is re-tested. Revisit the RQ011B verdict only after that. Keep the readiness
universe (RQ011 frozen) intact; this run does not change it.

## Source pointers

- `02_process/05_moment_level/CLOSEOUT_RQ011B.md`, `.../W1_moment/W1_moment_results.md`
- `execution_status.json` (`phase5_moment_monitor_closeout`); locked SAP `reports/plans/RQ011B_SAP_v4_moment_monitor_locked_20260629.md`
