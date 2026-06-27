# RQ010B Plan v0 — WOD-E2E Tracking Build + Human-Preference Validity

Status: `approved` (PI authorized 2026-06-24) · Wave: B · Work group: Group 4B · Date: 2026-06-24

## 1. Research question

> After building a multi-camera 3D/BEV tracker to recover counterpart tracks from WOD-E2E, does a
> lower frozen-M3 IPV deviation align with higher human preference among candidate trajectories from
> the same scene (manuscript R4, the human-alignment leg)?

## 2. Two phases (B1 can start now, in parallel with RQ009)

- **B1 — Tracker + data infrastructure (start now):** signed-in pilot to resolve official sizes/shards/
  throughput; build Route 4 multi-camera 3D/BEV tracker (Route 5 custom fallback); resolve critical-frame
  index alignment; establish a map/route reference fallback; pass a rating-blind tracking quality gate.
- **B2 — Preference test (after RQ009 M3 frozen AND B1 QA passes):** run frozen M3 on tracked scenes;
  test deviation ↔ released human preference scores.

## 3. Frozen inputs / contracts

RQ010 feasibility decision (T2_FULL_TRACKING_REQUIRED; Route 4 preferred); RQ009 frozen M3 (for B2);
RQ005 leakage contract. Ratings-blind: `ratings_read_allowed=false` until the final pre-registered test;
counterpart selection rating-independent; shared open-loop opportunity structure; abstain when counterpart
tracking / history / occlusion / transforms / map / forecast support is insufficient.

## 4. Denylist

Rating values (until the final pre-registered preference test); observed PET; post-critical actor
observations; rating-tuned predictors; silent M2 (context-only) substitution for missing tracks.

## 5. Gates / quality

Tracking quality gate: rating-blind 3D/BEV reference annotation; canonical pass/fail seed `2026062306`;
report HOTA/AMOTA/ID metrics + uncertainty calibration vs threshold. M2-downgrade guard active. If
map/critical-frame cannot be resolved, route to abstain rather than a forced M3 estimate.

## 6. Endpoints + PASS/FAIL

Among same-scene candidates: lower M3 deviation ↔ higher released preference score (criterion validity);
incremental utility over a prespecified kinematic+safety baseline; abstention rate. PASS requires an
IPV-specific, baseline-beating preference association; otherwise report a bounded/negative result.

## 7. Deliverables, stop conditions, claim boundaries

Deliverables: tracker + QA report; preference-validity result OR bounded-negative; `decision.md`.
Stop: if the tracker cannot reach QA or map/critical-frame stays unresolved, freeze as a feasibility/
bounded-negative result — do NOT downgrade to context-only and call it M3. Claims: open-loop preference
alignment only; not closed-loop, not realised harm; research-use only (Waymo non-commercial licence).

## 8. Dependencies

Upstream: RQ010 (frozen), RQ009 (B2 only). B1 (tracker) is independent infrastructure and is the long
pole — start immediately so it is ready when M3 freezes.
