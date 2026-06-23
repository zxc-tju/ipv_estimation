# RQ010A Traceability — spec requirement → phase → deliverable → status

RUN_ID: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e`
SPEC: `02_process/00_meta/spec_snapshot.md` (PLAN_SHA256 d9988309…1778d87)
Binding addendum (Phase 1 fixes): `02_process/01_plan_review/PLAN_REVIEW_ADDENDUM.md`

| SPEC item / gate | Phase | Deliverable(s) | Result |
|---|---|---|---|
| §3 Authoritative info to verify (W0) | 2 | `01_results/tables/official_source_inventory.csv`, `02_process/02_sources/source_findings.md` | Done; official/primary only; evidence_class on every row |
| Gate 010-0 Access & licence | 3 | `02_process/03_access_license/data_access_and_license_audit.md`, `acquisition_checklist.md`, `01_results/tables/download_size_derived_estimates.csv` | **GATE_010_0_PASS** (non-commercial research) |
| §W1 Required-field crosswalk | 4 | `01_results/tables/wod_schema_requirements.csv` (+evidence_class/provenance), `field_availability_crosswalk.csv` | Done; 50 fields classified from release-schema evidence |
| §W2 Gate 010-1 Tracking-necessity | 4 | `02_process/04_schema/tracking_need_decision.json`, `crosswalk_reasoning.md` | **T2_FULL_TRACKING_REQUIRED** (m2_downgrade_guard present) |
| §W3 Technical-option comparison | 5 | `01_results/tables/tracking_options_comparison.csv`, `02_process/05_tracking_options/route_analysis.md` | Routes 1–3 infeasible/hypothesis; preferred Route 4, fallback Route 5 |
| §W4 Tracking quality gate (outcome-blind) | 6 (+11b) | `01_results/exports/tracking_quality_gate_proposal.yaml`, `02_process/05_tracking_options/quality_gate_notes.md` | 11 metrics; rating-blind 3D/BEV reference protocol; seed 2026062306; no-reference→BLOCKED |
| §W5 Compute/storage budget | 7 | `01_results/tables/compute_storage_budget.csv` | 3 scenarios; ranges + assumptions + uncertainty (no pseudo-precision) |
| §W6 Gate 010-2 HPC decision | 7 | `02_process/06_compute_budget/hpc_decision.md` | **BLOCKED_PENDING_ACCESS** (B3 rubric; size/benchmark sign-in-gated) |
| Pilot benchmark plan | 8 | `02_process/07_pilot_plan/pilot_benchmark_plan.md` | N=80, rating-blind, fixed seed, plan-only (needs authorization) |
| §W7 Candidate-future actor protocol | 9 | `01_results/exports/candidate_future_actor_protocol.md` | Shared open-loop primary + candidate-conditioned sensitivity; open-loop≠harm; N4 alignment table |
| Gate 010-3 Independent review | 10 | `02_process/08_review/independent_review.md` | PASS_WITH_FIXES (0 blocking) |
| §11 Red team (+fix→reverify) | 11/11b/11c | `02_process/09_red_team/{red_team_findings,fix_log,reverify_findings}.md` | FIX_REQUIRED → fixed → **CLEAR** |
| §6 Formal figures (Nature skill) | 12 | `01_results/figures/{fig1,fig2,fig3}.{svg,pdf,png}`, `figure_manifest.json` | 3 figures (Nature-figure skill); colorblind-safe; source CSV |
| §6 Feasibility report (HTML) | 12 | `90_report/index.html`, `00_entry/index.html`, `01_results/exports/wod_feasibility_report.html` | Offline, no external refs (verified); figures embedded |
| Final report review | 13a | `02_process/10_final_review/final_review.md` | **FINAL_REVIEW_STATUS=PASS** |
| Registration | 13b | `main_workflow.log`, `rq_progress_registry.csv`, dashboard, `STUDIES.md` | Minimal append (re-read-first; concurrent RQ008 fleet) |
| Evidence ledger | all | `evidence.csv` | Appended per phase (RQ010A-Pn-*) |

§7 Acceptance: tracking need classified T0–T3 from official evidence ✓ · missing actor tracks triggered a real tracking evaluation, not silent M2 ✓ · access/licence explicit ✓ · compute cites assumptions/uncertainty ✓ · HPC reproducible (rubric) ✓ · ratings not used to tune tracking ✓ · report states full-M3 feasibility boundary ✓.
