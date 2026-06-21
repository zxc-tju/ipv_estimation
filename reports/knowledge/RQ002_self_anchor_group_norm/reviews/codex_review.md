# Codex Review: RQ002 Self-Anchor Group Norm

Status: review-complete, not yet frozen in `decision.md`.

Review date: 2026-06-21.

## Scope

Reviewed study packages:

- `reports/studies/RQ002_self_anchor_group_norm/RQ002_1_self_anchor_validation_main_20260619/`
- `reports/studies/RQ002_self_anchor_group_norm/RQ002_2_self_anchor_validation_codex_20260619/`

Both packages reach the same top-level decision: `MUST_REVISE`.

## Overall Verdict

RQ002 supports self-anchor as a sharpness and transfer component, but rejects
self-anchor alone as a validated group-norm verifier. The paper-safe route is a
hybrid verifier: self-anchor interval for sharp online estimation, plus a
situation floor, support checks, and out-of-support abstention.

Paper-safe phrasing:

> Self-anchored rolling-IPV intervals are useful for sharp online estimation,
> but they do not by themselves constitute a population social norm. A verifier
> must prevent self-consistency laundering by applying situation-conditioned
> normative floors and abstaining outside supported states.

## Claims That Can Be Carried Forward

1. **Self-anchor provides real sharpness.**
   In RQ002_1, locked causal-roll width is about `0.658` of the floor width.
   RQ002_2 independently reports TEST causal-roll coverage/width
   `0.899 / 0.591`, versus FLOOR `0.889 / 0.898`.

2. **Self-anchor transfer signal is useful but not sufficient.**
   RQ002_2 reports Leave-Waymo-Out causal-roll `0.902 / 0.628` versus FLOOR
   `0.868 / 0.871`, supporting the self-anchor as a transferable interval
   signal on the locked lane-supported slice.

3. **Out-of-support stress tests justify guardrails.**
   RQ002_1 marks E3 and its replication as supporting guarded self-anchor:
   large deviations are flagged or abstained rather than silently passed.
   RQ002_2's E3 shows flag/abstain behavior rising sharply under injected
   deltas, supporting a support-aware guard rather than a raw self-pass rule.

4. **Situation-only cannot simply replace self-anchor.**
   RQ002_1 E4 flags a concern: situation-only models lose most self-anchor
   sharpness/transfer. RQ002_2 E4 similarly shows situation-only intervals are
   wider than self-anchor intervals in both primary and LWO settings.

## Claims That Must Be Rejected Or Downgraded

1. **Reject self-anchor-only group-norm validation.**
   RQ002_1's formal verdict fails because E1 and E5 fail. The report explicitly
   says `MUST_REVISE`.

2. **Reject "self-consistency equals social compliance."**
   E1 shows the early anchor and full-window target are not independent enough
   to carry a pure group-norm claim. Self-anchor can be sharp because it tracks
   the driver's own behavior, but that is not the same as measuring the
   population norm.

3. **Reject external bad-outcome adjudication as passed.**
   E5 fails in both packages. RQ002_1 reports self-pass/situation-flag
   disagreements enriched for bad outcome proxies. RQ002_2 reports
   situation-only adjudication performing at least as well or better than
   self-anchor on the PET<=1 proxy.

4. **Downgrade individual-driver generalization.**
   Persistent driver IDs are unavailable, so E2 is only substitute evidence via
   case/source/scenario or LWO proxies.

## Required Hybrid Verifier Shape

The knowledge layer should preserve this design requirement:

- use self-anchor CQR for sharp online intervals when lane/route support exists;
- apply a situation-conditioned floor so the self-anchor cannot become more
  permissive than the population norm in risky states;
- abstain or fall back when support is sparse, lane/route is unavailable, or
  source health is poor;
- do not convert a self-pass into "socially compliant" without checking
  situation support.

## Knowledge-Layer Action

Update `synthesis.md` to replace any self-anchor-only wording with the hybrid
guardrail version. `decision.md` should reject "self-anchor alone is a group
norm" and accept only the guarded self-anchor role if the paper needs this RQ.
