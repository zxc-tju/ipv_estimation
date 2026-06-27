# RQ011B Plan v0 — OnSite Matched-Scenario Algorithm Validity

PI-accepted locked SAP addendum (2026-06-25) resolves Phase-1 blockers B001-B005; binding analysis spec = RQ011B_SAP_v1_locked_20260625.md; phase 3 still gated on RQ009 G1.

Status: `approved` (PI authorized 2026-06-24) · Wave: B · Work group: Group 5B · Date: 2026-06-24

## 1. Research question

> On the frozen OnSite universe, does frozen-M3 IPV deviation track official outcomes (rankings, scores,
> collisions) at the algorithm×scenario level — criterion validity and the consequence chain (manuscript R5)?

This is the fastest real external result: the data is READY; it only waits on the frozen M3.

## 2. Frozen inputs / contracts

RQ011 readiness (unit = algorithm×scenario; outcome universe `full_300`; replay/IPV universe `clean_285`
with `T19` excluded replay-only; run-level/repeated-run NOT identifiable; moderate replay selection bias);
RQ009 frozen M3; RQ005 leakage; RQ002 separation rule (compute the empirical-norm deviation and the
safety/policy guard as distinct outputs — never apply the guard then use guard-induced flags as validation).

## 3. Denylist

Exclusions or weights tuned on outcomes / IPV / IPV–outcome association; run-level / repeated-run / seed /
algorithm-superiority claims; full_300 replay or IPV coverage; observed PET on the online path.

## 4. Method / work packages

- W0: compute M3 deviation per algorithm×scenario on `clean_285` replay (outcomes read from `full_300`, with
  the replay-set selection caveat stated).
- W1: relate deviation to official ranking / per-scenario score / collisions / deductions (correlation +
  incremental regression controlling for prespecified kinematic + safety metrics).
- W2: transfer/robustness — leave-one-team-out (LOTO) and leave-one-scenario-out (LOSO).
- W3: **negative-control battery** (role_flip, sign_flip, counterpart_swap, kinematics_only, IPV_removed,
  shuffled_ipv) — the same battery that exposed RQ003's non-specificity.

## 5. Endpoints + PASS/FAIL

A supported result requires the IPV increment to (a) be statistically non-trivial, (b) **beat the negative
controls** (IPV-specific), and (c) **generalize across scenarios (LOSO)**. Anything less is reported as a
bounded/null result.

## 6. Prior / risk (read before running)

RQ003 (NSFC top-five pilot) already found **no robust, IPV-specific incremental utility** (Tier-B null). Pre-
register that RQ011B may reproduce that null; the value of the study holds either way (criterion/consequence
boundary). Do not p-hack toward a positive.

## 7. Deliverables, stop conditions, claim boundaries

Deliverables: matched-scenario validity result (or honest null); `decision.md`. Stop: report null/non-specific
results in full. Claims: bounded to algorithm×scenario; never run-level, repeated-run, or causal realised harm;
attach the T19 replay-bias caveat wherever replay/IPV results appear.

## 8. Dependencies

Upstream: RQ011 (frozen), RQ009 (M3). Feeds RQ013.
