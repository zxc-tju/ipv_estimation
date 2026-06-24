# Claude Code Review

Status: filed (2026-06-24)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e` (study_type feasibility-only; `overall_status: complete`).
Reader entry: `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/90_report/index.html`.

## Verdict

Concur with the feasibility-only verdict. The decisive, well-sourced finding: the **public WOD-E2E release
schema does not expose surrounding-actor tracks** (only ego states, camera images/calibration/timestamps, and
per-candidate preference scores), so Gate 010-1 is `T2_FULL_TRACKING_REQUIRED`. Consequence: the
counterpart-conditioned M3 preference validation **cannot run directly on WOD-E2E** — it first requires a
multi-camera 3D/BEV tracking pipeline (Route 4 preferred, Route 5 fallback). Access/license Gate 010-0 PASS
(Waymo non-commercial research path; no account created, no license accepted, no data downloaded). This is a
clean feasibility/route-scoping result, not an empirical validation.

## Key Findings

| Item | Result | Reading |
|---|---|---|
| Counterpart tracks in schema | Absent (REQ_COUNTERPART_* = requires_tracking) | M3 needs counterpart trajectories that WOD-E2E does not provide. |
| Tracking tier | T2_FULL_TRACKING_REQUIRED | Camera+calibration only → build detection→association→BEV→tracking. |
| Official E2E↔WOMD/WOD crosswalk | Not found (public sources) | T1 light-augmentation rejected; can't borrow Motion-dataset tracks. |
| Map conflict geometry / route ref | Missing from E2ED schema | T3 risk for any map-conditioned M3; needs RQ009 fallback. |
| Access/license (Gate 010-0) | PASS | Non-commercial research permitted; production/vehicle use forbidden. |
| HPC decision | BLOCKED_PENDING_ACCESS (rule R0) | No verified scale, no WOD-E2E benchmark; derived 0.15–17 TB full / 0.1–11 GB pilot only. |
| Candidate→rating identity | Available (preference_score metadata) | Ratings are per-candidate; usable later, ratings-blind. |

## Strengths

- Evidence is overwhelmingly **official-source / release-schema with line-level locators** (proto fields,
  Waymo terms, challenge page); benchmark/throughput claims are explicitly labelled nuScenes proxies, not
  WOD-E2E results. Strong provenance hygiene.
- Leakage discipline: `ratings_read_allowed=false`, rating-independent counterpart selection, and an explicit
  M2-downgrade guard (it did **not** silently substitute a context-only M2 for the missing M3).
- Honest abstention rules where tracking/map/transforms/critical-frame support is insufficient.

## Boundaries And Watch-Items

- The entire M3-on-WOD-E2E path is gated behind a tracker that **does not exist yet**; camera-only 3D depth
  error (LET-3D-AP) makes accuracy uncertain. RQ010B is not feasible until a pilot demonstrates adequate
  tracking + critical-frame alignment + a map/route fallback acceptable to the frozen RQ009 interface.
- Critical-frame index within the original 20 s run is unverified (alignment risk); map geometry missing
  (M3 boundary). Size/HPC unresolved because download pages are sign-in gated.
- **Provenance smell:** run-root metadata (`execution_status.json`, `README.md`, `TRACEABILITY.md`) were lost
  mid-run (OneDrive sync + concurrent RQ008 fleet) and recreated at finalization; the 28 substantive
  deliverables are reported intact, but the recreated-metadata caveat should be retained.
- `rq009_interface_status` is only `provisional_contract`.

## Reproducibility / Process Assessment

- Feasibility-only: no download, no tracker, no IPV, no rating values. Plan SHA-256 pinned. Plan review closed
  3 blocking items (B1/B2/B3); independent review PASS_WITH_FIXES; red team FIX_REQUIRED → fixed → re-verify
  CLEAR; final review PASS. Appropriate for a feasibility artifact.

## Supporting Role For The Program

- Correctly reframes RQ010 as a feasibility gate, not external validation. It tells the program that WOD-E2E
  is **not** a ready human-preference-validity set; converting `\planned{}` WOD validation into a result
  requires (1) signed-in Phase-8 pilot (official sizes, tracker FPS/accuracy, critical-frame alignment),
  (2) a frozen RQ009 interface incl. a bounded non-map fallback, and (3) sustained ratings-blind handling.

## Recommendation

Accept the feasibility verdict and the T2 tier. Do not cite WOD-E2E as a ready external-validation dataset.
Authorize RQ010B only after the Phase-8 pilot and RQ009 interface freeze; keep the analysis ratings-blind.

## Source Pointers

- `02_process/04_schema/tracking_need_decision.json`; `01_results/tables/field_availability_crosswalk.csv`
- `02_process/06_compute_budget/hpc_decision.md`; `01_results/tables/compute_storage_budget.csv`
- `evidence.csv` (P2–P11b rows); `execution_status.json`; `01_results/exports/candidate_future_actor_protocol.md`
