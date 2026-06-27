# Wave B Launch Plan — Substantive Evidence Explorations (2026-06-24)

PI decision (2026-06-24): **launch all four** substantive lines (RQ009 + RQ011B + RQ012B + RQ010B).

## Sequencing (what runs when)

```text
NOW (parallel):
  - RQ009  (M3 dynamic envelope, M0-M5)            [linchpin]
  - RQ010B-B1 (WOD multi-camera tracker build)     [long pole; independent infra]

AFTER RQ009 freezes M3 predictions:
  - RQ011B (OnSite matched-scenario)   [fast, ready -> first real external result]
  - RQ012B (OnSite automatic-event harm)
  - RQ010B-B2 (WOD preference test; also needs B1 QA pass)

LAST:
  - RQ013 (beyond-safety incremental value)  [needs R3 + R4/R5]
```

## The pivot gate

RQ009 reports an explicit **M3-vs-M4** comparison. M3 (context + counterpart IPV) drops the ego self-anchor
that RQ001/RQ002 showed was the *sharp* signal, so M3 may be less sharp than the demoted M4. If M3 is
materially worse, **pause downstream consumption and revisit the M3-primary framing with the PI** before
spending RQ010B/RQ011B/RQ012B effort on a weak M3.

## Cross-cutting contracts (every line must honour)

- Leakage (RQ005): observed PET, realized order, post-hoc phase, full-window IPV, closest frame are
  offline-only; runtime inputs need causal provenance.
- Estimability (RQ007): separate opportunity / estimability / support / deviation; abstain when not estimable.
- No temporal motifs (RQ008 negative): M3 uses context + counterpart current IPV only.
- Splits: case/scenario isolation; conformal calibrates the final interval; non-crossing quantiles.
- Negative-control discipline (from RQ003): any positive IPV increment must beat role_flip/sign_flip/
  counterpart_swap/kinematics_only/IPV_removed AND generalize (LOSO). Report nulls in full.
- Claim ceiling: empirical/probabilistic monitor, not formal proof; no planner-benefit/closed-loop claim.

## The two big scientific bets (be ready for either outcome)

1. **M3 sharpness** (RQ009): unproven; the sharp signal was the now-demoted self-anchor.
2. **External IPV–outcome association** (RQ011B/RQ010B): RQ003 prior is null/non-specific.

Minimal publishable evidence set = R1 (done) + R3 (RQ009) + R5 (RQ011B/012B). WOD (R4) is a high-value,
high-risk parallel bet.

## Per-line plans

- `RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md`
- `RQ010B_plan_v0_wod_e2e_tracking_and_preference_validity_20260624.md`
- `RQ011B_plan_v0_onsite_matched_scenario_validity_20260624.md`
- `RQ012B_plan_v0_onsite_automatic_event_harm_20260624.md`

Next: each plan needs an independent plan review before its status moves `approved -> running`; execution runs
use versioned dirs under `reports/studies/<RQ>/`. Orchestration prompts (cf. `reports/plans/prompts/`) can be
generated per line on request.
