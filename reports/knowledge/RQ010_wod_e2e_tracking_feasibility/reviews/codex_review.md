# Codex Review: RQ010 WOD-E2E Tracking Feasibility

Status: review-complete; feasibility package PASS; no knowledge-layer decision yet.
Review date: 2026-06-24.

## Scope

Reviewed study package:

- `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/`

Primary study evidence read:

- `00_entry/index.html`
- `README.md`
- `TRACEABILITY.md`
- `execution_status.json`
- `final_review.md`
- `independent_review.md`
- `red_team_findings.md`
- `fix_log.md`
- `reverify_findings.md`
- `source_findings.md`
- `data_access_and_license_audit.md`
- `crosswalk_reasoning.md`
- `tracking_need_decision.json`
- `hpc_decision.md`
- `candidate_future_actor_protocol.md`

## Overall Verdict

RQ010 is a feasibility and routing result, not a behavioral validation result.
It convincingly shows that the public WOD-E2E schema is not sufficient for direct
M3/IPV validation because surrounding-actor tracks, map or route geometry, and a
verified WOMD/WOD crosswalk are absent. The correct carry-forward decision is
`T2_FULL_TRACKING_REQUIRED`, with Route 4 as the preferred multi-camera
3D/BEV-tracking path and Route 5 as the fallback custom-tracking path.

Paper-safe phrasing:

> WOD-E2E cannot directly run the current multi-actor IPV validation stack from
> the released fields alone. It can become a future validation surface only
> after an independent, rating-blind tracking layer and a frozen quality gate are
> available.

## Claims That Can Be Carried Forward

1. The public WOD-E2E release fields support ego/camera/preference context but
   do not expose surrounding-actor tracks required by the current IPV estimator.
2. The correct tracking tier is `T2_FULL_TRACKING_REQUIRED`; direct T0/T1
   validation is rejected.
3. Route 4, adapting an existing multi-camera 3D/BEV tracker, is preferred.
   Route 5, a custom tracking pipeline, is the fallback if Route 4 fails.
4. The legal/access gate for non-commercial research and publication is
   passable, but access is sign-in gated and exact official size/throughput
   details were not available locally.
5. The HPC/data-volume decision is `BLOCKED_PENDING_ACCESS`, not ready-to-run.
   The N=80 pilot is a plan only until user authorization and access details are
   available.
6. The red-team fixes materially improved the package: quality gates must use
   independent, rating-blind 3D/BEV reference annotations and canonical seed
   `2026062306` for the pass/fail pilot design.
7. Candidate future actor forecasts are allowed only as open-loop, rating-free
   opportunity structure; they cannot be treated as closed-loop actor response
   or realized harm.

## Claims To Reject Or Defer

- Do not claim WOD-E2E directly supports M3, IPV computation, or preference
  validity analysis.
- Do not claim the tracker route has passed. Route 4 and Route 5 are proposed
  routes, not executed pipelines.
- Do not claim ratings/deviation analysis or any preference-score relationship.
  Ratings are explicitly outside the feasibility result.
- Do not claim full-dataset readiness, cluster budget readiness, or final data
  availability until access is resolved.
- Do not downgrade to an M2/context-only study as a substitute for multi-actor
  tracking; the package explicitly blocks that substitution.

## Quality And Compliance Notes

The final review passes and the red-team reverify status is clear. The major
initial red-team failures were resolved by adding an operational independent
reference-annotation protocol and reconciling the seed conflict. The evidence is
source-disciplined and does not smuggle in outcome/preference claims.

The remaining risk is practical rather than interpretive: without data access,
throughput facts, and an actual tracker pilot, RQ010 remains a route-selection
and feasibility contract.

## Knowledge-Layer Action

Recommended decision state: accept RQ010 as a feasibility boundary with
`T2_FULL_TRACKING_REQUIRED`, Route 4 preferred and Route 5 fallback, and keep
HPC/pilot execution blocked pending access and explicit authorization. Do not
use RQ010 as evidence for behavioral preference validity.
