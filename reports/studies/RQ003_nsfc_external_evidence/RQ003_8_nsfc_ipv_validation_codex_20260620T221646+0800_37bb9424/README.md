# RQ003_8 NSFC IPV Validation Boundary Report

Run ID: `RQ003_8_nsfc_ipv_validation_codex_20260620T221646+0800_37bb9424`

Status: PASS for Phase 10 report build.

Tier decision: **Tier C** domain-transfer / no-validated-increment result.

Canonical offline entry: `00_entry/index.html`

Compatibility offline entry: `90_report/index.html`

## What This Package Supports

- 150/150 top-five cells map from score to PDF to replay and are IPV-computable.
- Gate 0 measurement checks pass.
- Exact role-context high-support transfer is not evaluable: primary high-support N=0.
- Fallback LOTO is small and uncertain: delta_R2=+0.0094 with CI crossing zero.
- LOSO reverses direction.
- Negative controls meet expectation and IPV-removed equals kinematics-only.
- H3 blind mechanism is blocked because real two-annotator labels do not exist.
- NPC stronger matching is non-identifiable without script-version and seed fields.
- Independent replication reproduces N=0 and fallback estimates.

## What This Package Does Not Support

- Validated criterion validity.
- Successful domain transfer.
- Transferable NSFC coverage guarantees from InterHub calibration.
- Independently expert-validated coordination-endpoint wording.
- An NPC effect-identification or identical-input claim.
- A blind behavioral mechanism claim.
- Any statement that the paper repository was modified.

## Main Artifacts

- `01_results/figures/figure_manifest.csv`
- `evidence.csv`
- `TRACEABILITY.md`
- `execution_status.json`
- `02_process/19_report_build/nature_skill_manifest.json`
- `02_process/19_report_build/worker_report.json`

Note: `START_HERE.md` and `main_workflow.log` were intentionally not modified for this task per the Phase 10 worker scope.

Paper repo hygiene: this run made zero paper-repo edits; the paper repo working tree has a pre-existing unrelated `.gitignore` modification, left untouched.
