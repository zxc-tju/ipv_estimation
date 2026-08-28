# RQ027_1 Known-Truth Recovery Feasibility Pilot

Status: `PILOT_NO_GO`

Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`

Official entry: `00_entry/index.html`

## Summary

The full development-only matrix completed: `240` interactive runs and `48` negative controls. Engineering execution was healthy, but the recovery, selective-risk and negative-control gates failed. Independent recomputation reproduced the verdict. S2 and sealed confirmatory expansion are stopped.

## Files

- `00_entry/index.html`: offline reader entry
- `01_results/report.md`: formal self-contained report
- `01_results/conclusions.md`: conclusion and evidence-state summary
- `01_results/independent_validation.md/json`: independent recomputation
- `01_results/run_level_results.csv`: one row per simulation run
- `01_results/frame_level_results.parquet`: target-side frame ledger
- `02_process/`: commands, environment and traceability

## Deviations From Approved Plan

- The bounded v0 pilot completed as planned; S2, ablations, formal figures and sealed confirmatory were not run because the predeclared feasibility verdict was `PILOT_NO_GO`.
- Two post-resolution negative-control runs crossed the generator's collision tag; they remain in every denominator and are listed in `independent_validation.json`.
