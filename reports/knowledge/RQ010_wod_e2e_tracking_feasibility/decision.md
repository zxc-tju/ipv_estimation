# RQ010 Decision: WOD-E2E Tracking Feasibility

Status: ACCEPTED — feasibility/route boundary (knowledge-layer freeze, human-directed 2026-06-24). RQ010B (preference validation) BLOCKED pending tracking pilot + signed-in access. Not behavioural-validity evidence.

Run ID: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e` (study_type feasibility-only)
Plan SHA-256: `d9988309a803dc443cfd2c9bcde2f75e078c8cd87e57a036828f0a13e1778d87`
Basis for freeze: plan review (3 blocking closed); independent review PASS_WITH_FIXES; red team FIX_REQUIRED → reverify CLEAR; final review PASS; `reviews/claude_review.md` and `reviews/codex_review.md` both concur. Frozen at PI direction.

## Accepted Claims

| ID | Claim |
|---|---|
| RQ010-KC-T2 | The public WOD-E2E release schema exposes ego/camera/preference context but **no surrounding-actor tracks**, so the tracking tier is `T2_FULL_TRACKING_REQUIRED`; direct T0/T1 validation is rejected (no official WOMD/WOD crosswalk verified). |
| RQ010-KC-ROUTE | Route 4 (adapt an existing multi-camera 3D/BEV tracker) is preferred; Route 5 (custom detection→association→BEV→tracking) is the fallback. Both are proposed routes, not executed pipelines. |
| RQ010-KC-ACCESS | Gate 010-0 PASS: a non-commercial research/publication access path exists (no account created, no licence accepted, no data downloaded); production/vehicle use is forbidden. |
| RQ010-KC-HPC | The compute/data-volume decision is `BLOCKED_PENDING_ACCESS` (rule R0): no verified official scale and no WOD-E2E Route-4 benchmark; derived envelope only (≈0.15–17 TB full / 0.1–11 GB pilot). |
| RQ010-KC-PROTOCOL | The RQ010B protocol is a shared open-loop, ratings-blind opportunity structure with explicit abstention; candidate-conditioned forecasting is sensitivity-only and is not realised harm. |

## Rejected Or Deferred Claims

| Claim | Reason |
|---|---|
| WOD-E2E directly supports M3 / IPV / preference validity | Counterpart tracks, map/route geometry, and a crosswalk are absent. |
| Route 4/5 has "passed" | Proposed routes; no tracker was run. |
| Any ratings/deviation/preference-score relationship | Ratings explicitly outside the feasibility result. |
| Full-dataset / cluster-budget / data-availability readiness | Sizes and manifests are sign-in gated; unresolved. |
| Downgrade to a context-only M2 substitute | Explicitly blocked (M2-downgrade guard). |

## Boundaries / Watch-Items

Feasibility-only (no download, tracker, IPV, or ratings). Map conflict geometry MISSING (M3 risk); critical-frame index within the original 20 s run unverified; official sizes sign-in gated; `rq009_interface_status = provisional_contract`. Provenance caveat: run-root metadata (`execution_status`, `README`, `TRACEABILITY`) were lost mid-run (OneDrive + concurrent RQ008 fleet) and recreated at finalization; 28 substantive deliverables reported intact.

## Paper Handoff / Next Gate

Use only as a feasibility/route contract; do **not** cite WOD-E2E as a ready external human-preference-validity set. Before RQ010B: run the signed-in Phase-8 pilot (official sizes, tracker FPS/accuracy, critical-frame alignment, map/route fallback) and freeze the RQ009 interface; keep all work ratings-blind.
