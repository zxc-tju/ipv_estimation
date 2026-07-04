# RQ010 Knowledge Synthesis

Status: consolidated from the frozen `decision.md` (ACCEPTED feasibility/route boundary; RQ010B authorized by PI 2026-06-24). `decision.md` is the canonical claim ledger.

## What was accepted (feasibility / route)

- `RQ010-KC-T2`: the public WOD-E2E schema exposes ego/camera/preference context but **no surrounding-actor tracks** → `T2_FULL_TRACKING_REQUIRED`; T0/T1 rejected (no verified WOMD/WOD crosswalk).
- `RQ010-KC-ROUTE`: Route 4 (adapt a multi-camera 3D/BEV tracker) preferred; Route 5 (custom) fallback.
- `RQ010-KC-ACCESS`: Gate 010-0 PASS — a non-commercial research/publication path exists (production/vehicle use forbidden).
- `RQ010-KC-PROTOCOL`: RQ010B is a shared open-loop, ratings-blind opportunity structure with explicit abstention; candidate-conditioned forecasting is sensitivity-only; predicted responses ≠ realised harm.

## What it is not (yet)

Feasibility/route only — **not** behavioural-validity evidence. WOD-E2E may not be cited as a completed human-preference validation until RQ010B delivers results (`\externalpending{R4}` in the manuscript).

## Open risks carried into RQ010B

Map conflict geometry missing (M3 risk — needs RQ009 fallback); critical-frame index unverified; camera-only 3D depth error; HPC scale only bounded by a derived estimate (0.15–17 TB full / 0.1–11 GB pilot). The RQ009 interface must be frozen before the M3 preference test. With RQ012 human annotation dropped, WOD-E2E now jointly carries the human-alignment leg with InterHub, raising RQ010B's priority.

Sources: `decision.md`; `reviews/claude_review.md`; `reviews/codex_review.md`.
