# RQ023 Knowledge Layer: Benchmark Run Ledger

Status: **accepted (PI ratified 2026-08-12)**

Evidence package: `reports/knowledge/RQ023_benchmark_run_ledger/reports/run_ledger_audit_20260811.md`

GitHub issue: TBD

Paper section: Methods run accounting; Results/Discussion only after PI ratification

## Research Question

The matched-scenario real-vehicle benchmark reports 267 system-scenario cases, although 19 systems across 15 scenarios define 285 replay-eligible cells, and the consequence analyses use 175 cases. What accounts for the differences, and is the missingness related to the monitor's verdicts?

These counts and their meanings come from the copied audit report, especially sections 2–4. Its absolute source paths, filters, keys, and columns are preserved in that report; this README does not replace those provenance records.

## Why This Was Asked

Five consecutive manuscript review rounds raised the same run-accounting gap, and the manuscript could not answer it. This review-history statement comes from `INVESTIGATE.txt` under “Background”; it is context for the audit, not a newly analysed result.

## Current State

The read-only forensic audit was completed on 2026-08-11, independently verified before registration, and RATIFIED by the PI on 2026-08-12. Its report has been copied out of the disposable scratch directory and into this folder. All five knowledge claims in `decision.md` are **accepted (PI ratified 2026-08-12)** and paper-safe, subject to the two Required Qualifications.

## Manuscript Coupling

On 2026-08-11, the manuscript was corrected in two places on the strength of this audit, using only numbers already frozen in RQ017 and RQ019:

- The Methods run accounting was rewritten to print the funnel 267 -> 231 -> 204 -> 175 explicitly. RQ017 freezes 267 and 231; RQ019 freezes 27 and 175; 204 is the arithmetic difference 231 - 27, as registered in RQ023-KC-2.
- Two statements that exceeded the frozen evidence were softened: “fixed-script” counterparts were no longer said to be unable to respond to the ego, because RQ019 states only that they may not react; and the undocumented claim that counterpart control matched across the automated-systems and human-driver arms was removed because RQ023 records that equivalence under `Open / UNKNOWN`.

The manuscript does not yet cite RQ023-KC-1, RQ023-KC-3, RQ023-KC-4, or RQ023-KC-5 and cannot do so until the PI ratifies them.
