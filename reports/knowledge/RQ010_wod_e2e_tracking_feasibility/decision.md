# RQ010 Decision: WOD-E2E Tracking Feasibility

Status: ACCEPTED feasibility/route boundary; **RQ010B AUTHORIZED by PI 2026-06-24** (signed-in pilot + build multi-camera tracker (Route 4 preferred) + HPC approved). Not behavioural-validity evidence yet.

Run ID: `RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e` (feasibility-only)
Plan SHA-256: `d9988309a803dc443cfd2c9bcde2f75e078c8cd87e57a036828f0a13e1778d87`
Basis: independent review PASS_WITH_FIXES; red team reverify CLEAR; final review PASS; `reviews/claude_review.md` + `reviews/codex_review.md`; PI authorization 2026-06-24.

## PI Decision (2026-06-24): authorize RQ010B

WOD-E2E is now a priority external-validation surface (it carries the human-alignment leg jointly with InterHub after RQ012 human annotation was dropped). Authorized: (1) signed-in WOD-E2E pilot to resolve official sizes/shards/throughput; (2) build the Route 4 multi-camera 3D/BEV tracker (Route 5 fallback); (3) commit HPC for the tracking pipeline. Keep all work ratings-blind.

## Accepted Claims (feasibility)

| ID | Claim |
|---|---|
| RQ010-KC-T2 | The public WOD-E2E schema exposes ego/camera/preference context but no surrounding-actor tracks → `T2_FULL_TRACKING_REQUIRED`; T0/T1 rejected (no verified WOMD/WOD crosswalk). |
| RQ010-KC-ROUTE | Route 4 (adapt multi-camera 3D/BEV tracker) preferred; Route 5 (custom) fallback. |
| RQ010-KC-ACCESS | Gate 010-0 PASS: non-commercial research/publication path exists (production/vehicle use forbidden). |
| RQ010-KC-PROTOCOL | RQ010B is a shared open-loop, ratings-blind opportunity structure with explicit abstention; candidate-conditioned forecasting is sensitivity-only; predicted responses ≠ realised harm. |

## Open Risks Carried Into RQ010B (not resolved by authorization)

Map conflict geometry MISSING (M3 risk — needs RQ009 fallback); critical-frame index within the 20 s run unverified; official sizes were sign-in gated (pilot resolves); camera-only 3D depth error (tracker accuracy risk); HPC scale only bounded by derived estimate (0.15–17 TB full / 0.1–11 GB pilot). RQ009 interface must be frozen before the M3 preference test.

## Paper Handoff

Still feasibility/route until the pilot + tracker deliver; `\externalpending{R4}` in the manuscript. Do not cite WOD-E2E as a completed human-preference validation until RQ010B produces results.
