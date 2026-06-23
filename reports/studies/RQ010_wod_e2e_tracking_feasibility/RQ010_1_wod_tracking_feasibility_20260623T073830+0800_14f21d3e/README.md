# RQ010A — WOD-E2E Data and Tracking Feasibility (Run 1)

- **RUN_ID**: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e`
- **RQ**: RQ010A · **Type**: feasibility & infrastructure audit (Wave A)
- **GIT_HEAD at start**: `38063a2ff9cdc717098cf3f821c2bb162a0ac1d9`
- **PLAN_SHA256**: `d9988309a803dc443cfd2c9bcde2f75e078c8cd87e57a036828f0a13e1778d87`
- **Completed**: 2026-06-23 (+0800)
- **Reader-facing report**: [`90_report/index.html`](90_report/index.html) · entry [`00_entry/index.html`](00_entry/index.html) · alias [`01_results/exports/wod_feasibility_report.html`](01_results/exports/wod_feasibility_report.html)

## Question
Does the WOD-E2E release provide the actor trajectories, map, timing, candidate-future, and rating
fields required to run the full counterpart-conditioned verifier (M3); if not, what tracking/data-
alignment is required, and what compute/storage/HPC resources are needed?

## Headline verdict
- **Tracking need = `T2_FULL_TRACKING_REQUIRED`.** WOD-E2E's public release schema exposes camera
  images + calibration, ego past/future states, an intent enum, and ≤3 scored candidate ego
  trajectories — but **no surrounding-actor tracks, no map/route geometry, and no verified WOMD/WOD
  crosswalk.** M3 cannot run directly.
- **Preferred route = Route 4** (adapt an existing multi-camera 3D/BEV tracker); **fallback = Route 5**
  (custom pipeline).
- **Access/licence Gate 010-0 = PASS** (Waymo non-commercial research/publication permitted).
- **HPC decision = `BLOCKED_PENDING_ACCESS`** — official size/throughput are sign-in-gated; derived
  ranges span orders of magnitude. A rating-blind pilot (N=80, plan-only) is designed to resolve them.
- Independent review = PASS_WITH_FIXES; red team = FIX_REQUIRED → fixed → **re-verified CLEAR**.
- `rq009_interface_status = provisional_contract`.

## NOT in scope this round
No full dataset download · no tracker training · no IPV computation · no rating–deviation analysis ·
no claim that full M3 preference validity is demonstrated. Pilot execution requires user authorization.

## Phase map (0–13)
0 bootstrap · 1 plan review · 2 official sources · 3 access/licence (Gate 010-0) · 4 field crosswalk
→ tracking need (Gate 010-1) · 5 routes · 6 quality gate (outcome-blind) · 7 compute/HPC (Gate 010-2)
· 8 pilot plan · 9 actor protocol · 10 independent review (Gate 010-3) · 11 red team (+fixer+reverify)
· 12 Nature figures + offline HTML · 13 final review + registration.

## Key artifacts
- Sources: `01_results/tables/official_source_inventory.csv`, `02_process/02_sources/source_findings.md`
- Access/licence: `02_process/03_access_license/data_access_and_license_audit.md`, `acquisition_checklist.md`
- Schema/crosswalk: `01_results/tables/wod_schema_requirements.csv`, `field_availability_crosswalk.csv`, `02_process/04_schema/tracking_need_decision.json`
- Routes: `01_results/tables/tracking_options_comparison.csv`, `02_process/05_tracking_options/route_analysis.md`
- Quality gate: `01_results/exports/tracking_quality_gate_proposal.yaml`
- Compute/HPC: `01_results/tables/compute_storage_budget.csv`, `02_process/06_compute_budget/hpc_decision.md`
- Pilot: `02_process/07_pilot_plan/pilot_benchmark_plan.md`
- Actor protocol: `01_results/exports/candidate_future_actor_protocol.md`
- Figures: `01_results/figures/` (fig1 field-availability matrix, fig2 tracking decision tree, fig3 compute-budget ranges; SVG/PDF/PNG + `figure_manifest.json`)
- Review/red team: `02_process/08_review/independent_review.md`, `02_process/09_red_team/{red_team_findings,fix_log,reverify_findings}.md`, `02_process/10_final_review/final_review.md`
- Evidence ledger: `evidence.csv`; binding plan addendum: `02_process/01_plan_review/PLAN_REVIEW_ADDENDUM.md`

## Provenance note
Orchestrated via Codex CLI (gpt-5.5 xhigh) under an isolated CODEX_HOME; figures by the Nature-figure
skill (user-approved). Run-root metadata (this file, `TRACEABILITY.md`, `execution_status.json`) were
lost mid-run to OneDrive/concurrent-fleet churn and recreated at finalization; all 28 substantive
deliverables verified intact.
