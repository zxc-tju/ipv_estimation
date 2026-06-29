# ChatGPT Review — RQ012B OnSite Deviation-to-Harm Evidence

Date: 2026-06-29  
Reviewer: ChatGPT  
Status: `review-complete`  
Scope: RQ012B automatic-event harm endpoint, including the full behavioural battery.

## Verdict

RQ012B is a credible completed endpoint, but its result is a bounded/null result:

> Frozen-envelope deviation does not robustly, IPV-specifically predict realised OnSite interaction-failure harm under the current automatic-event and official-outcome battery.

This is useful boundary evidence. It is not an OnSite validation success.

## Scope revision

The two-human blind-annotation path has been deprecated by PI decision. RQ012B therefore uses:

- automatic events;
- official outcomes, collisions, deductions, and subscores;
- no human labels;
- no inter-annotator agreement;
- no human-judgment convergent leg from OnSite.

Human alignment is now carried, if at all, by WOD-E2E preference validation and InterHub.

## Pipeline reviewed

The RQ012B scientific endpoint ran after RQ009 froze the scorer. The validated pipeline includes:

- frozen RQ009 M3 scorer reconstructed with parity `0.0`;
- pinned legacy HPC IPV estimator;
- OnSite IPV and M3 anchors on HPC;
- 67,861 anchors over 267 units;
- support gate: 19,044 / 67,861 anchors in support;
- 245 / 267 usable units;
- 840 out-of-band moments across 149 units;
- pre-registered association tests;
- full behavioural battery and negative controls.

## Result summary

The first safety/collision pass showed only a weak direction:

- more out-of-band deviation associated with slightly worse safety;
- Spearman around `−0.12`, p about `0.06`, n=245;
- label permutation failed;
- placebo and context-only M2 controls were stronger or comparable.

The full behavioural battery then tested 9 automatic events, groupings, 4 official subscores, kinematic baseline, cluster-aware permutation, label/placebo/M2/exposure controls, and BH-FDR over 64 tests.

The definitive pattern:

- no channel is robustly supported;
- near-miss/contact is nominal but context-explained, failing the M2 control;
- hard braking, jerk, and comfort are null;
- too-passive to deadlock is the only hint that survives all controls, but it is underpowered and BH-edge, so it remains an unconfirmed future hypothesis.

## Interpretation

RQ012B shows that the RQ009 envelope produces a measurable out-of-band signal on OnSite, but that signal does not become a robust, IPV-specific realised-harm predictor in this battery.

This is consistent with:

- RQ009's counterpart-IPV practical null;
- RQ011B's OnSite monitor under-identification;
- earlier NSFC / OnSite Tier-B boundary evidence.

## Paper-safe wording

> In OnSite, frozen-envelope deviation yielded a measurable out-of-band signal, but it did not robustly or IPV-specifically predict realised interaction-failure harm under automatic-event and official-outcome tests.

Possible future-work note:

> A too-passive-to-deadlock channel appeared as an underpowered, BH-edge hypothesis and requires a dedicated powered test.

## Prohibited wording

Do not write:

```text
M3 deviation predicts realised harm.
IPV mismatch causes interaction failure.
OnSite validates behavioural consequences.
Automatic events are human-judgment labels.
The passivity-deadlock hint is an accepted claim.
```

## Program implications

1. RQ012B should be retained as a negative/boundary result, not hidden.
2. OnSite should be framed as a stress test, not external validation success.
3. RQ013 should not claim beyond-safety value from current OnSite evidence alone.
4. If a positive consequence chain is still desired, the program needs either WOD-E2E preference success or a future powered OnSite event/segmentation study.

## Final recommendation

Accept the RQ012B bounded/null conclusion. Use it to constrain the manuscript's R5 consequence claim and to prevent overclaiming realised-harm validation.
