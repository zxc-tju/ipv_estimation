# ChatGPT Review — RQ011B OnSite Moment-Level Monitor Validity

Date: 2026-06-29  
Reviewer: ChatGPT  
Status: `review-complete`  
Scope: RQ011B moment-level monitor validity close-out appended to the RQ011 knowledge decision.

## Verdict

RQ011B should be treated as:

```text
PROVISIONAL_NULL / UNDER_IDENTIFIED
```

It does not demonstrate OnSite moment-level monitor validity, but it also does not cleanly refute IPV monitoring. The primary finding is that the current OnSite failure-segment retrieval / segmentation layer is not adequate for a clean moment-level monitor test.

## What RQ011A already established

The RQ011A readiness decision remains valid and useful:

- outcome universe: `full_300` = 20 teams × 15 scenarios;
- replay / trajectory / IPV universe: `clean_285`;
- T19 excluded for replay-dependent analyses only;
- valid analysis unit: `algorithm×scenario`;
- no repeated-run, seed-level, run-level, or causal claims.

That readiness layer should continue to be used as the universe contract for downstream OnSite work.

## RQ011B result

RQ011B tested whether IPV deviation from the frozen human envelope could work as a parsimonious directional runtime monitor of interaction-failure moments on OnSite replay.

The primary test was under-identified:

- the any-failure-moment versus C1 within-interaction matched-control contrast had `0` C1 controls;
- the primary effect, efficiency, and confidence interval were therefore not estimable.

Robustness checks did not rescue the result:

- C2 ROC AUC about `0.493`;
- effect about `0.0084`;
- C3 approximately `0`;
- C4 about `0.0084`;
- fixed-alarm false alarms about `54.2` per interaction-minute with recall about `0.20`;
- no BH-significant category.

## Core interpretation

The bottleneck is not simply that the monitor failed. The bottleneck is that the failure-segment measurement layer is ill-posed:

- collision-only criteria are too sparse;
- broad any-failure criteria are saturated;
- moment-level matched controls vanish.

Therefore RQ011B cannot support either a positive monitor-validity claim or a clean negative refutation.

## Paper-safe wording

> OnSite moment-level monitor validity was not demonstrated under the current failure-segment retrieval; the test was under-identified and motivates a dedicated failure-segmentation layer before strong runtime-monitor claims.

## Prohibited wording

Do not write:

```text
OnSite validates runtime warnings.
OnSite refutes IPV monitoring.
IPV deviation predicts failure moments.
The monitor has high recall or low false alarm rate.
The result supports algorithm superiority.
The result is causal.
```

## Program implications

1. RQ011B should not be a main manuscript result.
2. It can appear in Extended Data or Discussion as a boundary/stress-test failure.
3. A future RQ should be created only if the program wants to solve interaction-failure segment retrieval before retrying moment-level monitoring.
4. RQ013 should not use RQ011B as positive beyond-safety evidence.
5. The clean-285 universe and full-300 outcome universe from RQ011A remain useful for other, better-defined OnSite analyses.

## Final recommendation

Register RQ011B as a measurement-limited close-out. Preserve the null and under-identification result, but do not allow it to become either a success claim or a clean refutation in the manuscript.
