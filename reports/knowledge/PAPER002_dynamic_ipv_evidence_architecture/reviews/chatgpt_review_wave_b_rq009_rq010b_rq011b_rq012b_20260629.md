# ChatGPT Review — Wave-B RQ009/RQ010B/RQ011B/RQ012B

Date: 2026-06-29  
Reviewer: ChatGPT  
Scope: recent Wave-B evidence relevant to the dynamic-IPV manuscript architecture  
Status: `review-complete`; this is a cross-RQ paper-architecture review, not an accepted claim ledger.

## Reviewed materials

This review was prepared from the current GitHub `main` state of `zxc-tju/ipv_estimation`, focusing on:

- `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
- `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/README.md`
- `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/execution_status.json`
- `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`
- `reports/plans/RQ010B_plan_v3_wod_e2e_tracking_and_preference_validity_20260625.md`
- current `START_HERE.md` RQ010B HPC handoff notes
- `reports/knowledge/RQ011_onsite_full_universe_readiness/decision.md`
- `reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`
- `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/README.md`
- current `STUDIES.md`, `RQ_PROGRESS_DASHBOARD.md`, and `rq_progress_registry.csv`

The formal claim ledgers remain the individual RQ `decision.md` files. If this review conflicts with an RQ decision file, the RQ decision file controls.

## Executive conclusion

The recent Wave-B evidence clarifies the paper direction:

```text
RQ009: method result accepted — context-conditioned conformal envelope works.
RQ010B: still running — WOD-E2E human-preference validation depends on tracking QA.
RQ011B: completed as provisional null / under-identified — OnSite moment-level monitor validity not demonstrated.
RQ012B: completed as bounded/null — OnSite M3 deviation does not robustly predict realised interaction-failure harm.
```

The manuscript should not be framed as a validated counterpart-IPV or self-anchor social-compliance verifier. The strongest supported spine is now:

> An estimability-aware, context-conditioned split-conformal envelope can serve as an empirical runtime monitor; however, the IPV-conditioning channels do not add measurable value over context, and OnSite consequence tests currently provide bounded/null evidence rather than successful external validation.

The main unresolved positive external-validity opportunity is WOD-E2E, but only if the RQ010B tracking pipeline clears its rating-blind QA gate.

## RQ009 review — dynamic envelope

### What is accepted

RQ009 accepts a sharp, near-nominal, estimability-aware **context-conditioned** split-conformal envelope. The accepted 90% result is approximately:

- coverage: `0.8987`;
- width: `1.0162`;
- Winkler: `1.4229`;
- context model versus global floor: width `−42.3%`, Winkler `−35.6%`;
- abstention: about `4.78%`.

This is the strongest current positive result for the manuscript's R3 method section.

### What is not accepted

The counterpart-IPV and ego-self-IPV channels are not manuscript claims. The RQ009 decision records a practical null:

- M3 / context + counterpart IPV is effectively the same as context-only / IPV-removed.
- M4 / ego self-history adds only marginal sharpness.
- The M3-vs-IPV-removed 90% Winkler effect is approximately `−0.0002`, with CI spanning zero and case sign p about `0.863`.

Therefore the paper should not argue that counterpart IPV is the operative mechanism, and should not revive self-anchor as the norm-defining signal.

### Boundaries

- Conditional subgroup coverage is uneven: `126/264` supported subgroup rows outside ±3 pp.
- LODO transfer coverage ranges roughly `0.749–0.991`, so transfer is not unconditional.
- The exact-zero target atom around `21.6%` qualifies tie/endpoint interpretation.
- The result is an empirical monitor, not a formal proof or planner-performance result.

### Review judgment

Use RQ009 as the R3 method result, but name the supported method:

```text
estimability-aware context-conditioned conformal envelope
```

Avoid:

```text
counterpart-conditioned IPV mechanism
self-anchor verifier
IPV contains new information beyond kinematics
validated social-compliance verifier
```

## RQ010B review — WOD-E2E tracking and human-preference validity

### Current state

RQ010B remains in progress. The v3 plan defines the right scientific question: among three same-scene candidate ego futures, does lower frozen-envelope deviation predict higher human preference, specifically beyond kinematic/safety baselines and control batteries?

The data facts are clear:

- WOD-E2E has 8 surround cameras, ego history/future, routing command, and rater feedback candidates.
- It does not provide actor tracks, LiDAR, or HD map in the E2E release.
- The human-preference test cannot run until a rating-blind multi-camera tracking layer clears QA.

### Engineering progress

The Tongji HPC setup is active. The current operating brief records:

- E2E parser access is working under `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`;
- a four-shard structural pre-flight sampled candidate-bearing frames and passed t* checks on that sample;
- StreamPETR Route 4 infrastructure is installed;
- Perception v1.4.3 dev and finetune subsets have been downloaded and crc32c verified;
- dummy, smoke, and small training runs have executed.

However, the first 64-train / 16-val detector-quality gate was poor:

- overall AP about `0.0033`;
- recall about `0.080`;
- precision about `0.033`;
- pedestrian and cyclist recall/AP near zero.

A 256-train / 16-val StreamPETR finetune is or was running as the next Route-4 attempt. Route 5 remains the fallback if full-data Route 4 still fails QA.

### Review judgment

RQ010B is the only remaining path that could provide a strong independent human-preference leg, but it is an engineering-risk path. It should remain in progress until the Route-4 QA decision is known.

Current paper-safe wording:

> WOD-E2E is being prepared as an independent human-preference validation surface, but the released fields require a rating-blind multi-camera tracking layer before the frozen envelope can be tested.

If Route 4/5 fail, WOD-E2E becomes a feasibility/bounded-negative result rather than preference-validity evidence.

## RQ011B review — OnSite moment-level monitor validity

### Result

RQ011A readiness remains accepted: full outcome universe `300`, replay/IPV universe `285`, with T19 excluded only for replay-dependent analyses.

RQ011B attempts a moment-level runtime-monitor validity test on OnSite. The close-out is:

```text
PROVISIONAL_NULL / UNDER_IDENTIFIED
```

The primary within-interaction failure-moment contrast is under-identified because the C1 matched-control set has zero controls. Robustness controls do not rescue the test:

- C2 ROC AUC around `0.493`;
- small or near-zero effects;
- no BH-significant category;
- fixed-alarm false-alarm rate about `54.2` per interaction-minute with recall about `0.20`.

### Interpretation

This is not a clean refutation of IPV monitoring. It diagnoses the OnSite failure-segment measurement layer:

- collision-only criteria are too sparse;
- broad any-failure criteria are saturated;
- moment-level matched controls vanish.

### Review judgment

RQ011B should not enter the manuscript as a main result. Its safe role is a limitation or Extended Data boundary:

> OnSite moment-level monitor validity was not demonstrated under the current failure-segment retrieval; the result is under-identified and motivates a dedicated failure-segmentation layer before strong runtime-monitor claims.

Avoid:

```text
OnSite refutes IPV monitoring
OnSite validates runtime monitor warnings
algorithm-level superiority
causal failure prediction
```

## RQ012B review — OnSite deviation-to-harm evidence

### Result

RQ012B has completed the scientific endpoint using automatic events and official outcomes, after dropping the two-human blind annotation path. The result is bounded/null:

> M3 deviation does not robustly, IPV-specifically predict realised OnSite interaction-failure harm.

The validated pipeline includes:

- frozen RQ009 M3 scorer parity `0.0`;
- pinned legacy HPC IPV estimator;
- 67,861 anchors over 267 units;
- support gate: 19,044 / 67,861 in-support anchors;
- 245 / 267 usable units;
- 840 out-of-band moments across 149 units;
- pre-registered association plus full behavioural battery and negative controls.

### Full-battery outcome

The full behavioural battery covered 9 automatic events, official subscores, groupings, kinematic baseline, cluster-aware permutation, label/placebo/M2/exposure controls, and BH-FDR over 64 tests.

The pattern is coherent but not robust:

- direction is broadly worse with more deviation, but weak;
- near-miss/contact is nominal but context-explained and fails the M2 control;
- braking, jerk, and comfort channels are null;
- too-passive to deadlock is the only hint that survives controls, but it is underpowered and BH-edge, so it is a future hypothesis, not an accepted claim.

### Review judgment

RQ012B is a credible negative/boundary result and should be preserved. The paper should not claim realised-harm validation on OnSite.

Safe wording:

> In OnSite, frozen-envelope deviation produced a measurable out-of-band signal, but it did not robustly or IPV-specifically predict realised interaction-failure harm under the current automatic-event and official-outcome battery.

Possible future-work note:

> The passivity-to-deadlock channel is a bounded, underpowered hypothesis requiring a powered test.

Avoid:

```text
M3 deviation predicts real harm
IPV mismatch causes interaction failure
OnSite validates behavioural consequences
automatic events are human-judgment labels
```

## Cross-RQ synthesis for the manuscript

### Supported

1. A context-conditioned conformal envelope is materially sharper than a global floor while maintaining near-nominal marginal coverage.
2. Estimability gating and abstention are important; IPV estimates are not equally interpretable at all moments.
3. The current OnSite evidence is useful as stress testing and boundary evidence, not as successful external validation.
4. WOD-E2E remains the main possible route to independent human-preference validity, pending tracking QA.

### Not supported

1. Counterpart-IPV conditioning as a mechanism.
2. Ego self-history / self-anchor as a group norm.
3. A robust temporal IPV motif or positive temporal law from RQ008A.
4. OnSite moment-level monitor validity.
5. OnSite realised-harm validation.
6. Beyond-safety incremental value.

## Recommended manuscript structure

### R1 — State-conditioned IPV behaviour

Use RQ004/RQ007 boundaries. Emphasize state/context dependence and measurement support, not universal law.

### R2 — Estimability and temporal-boundary result

Use RQ007 for interaction-conditioned estimability and RQ008 as a negative discovery boundary. Do not present a positive temporal process.

### R3 — Context-conditioned conformal envelope

Use RQ009 as the core method result. Emphasize sharpness, near-nominal coverage, abstention, and empirical runtime-monitor framing.

### R4 — WOD-E2E human-preference validation

Keep as ongoing or pending. It becomes a main result only after RQ010B tracking QA and rating-unlock analysis pass.

### R5 — OnSite consequence stress test

Use RQ012B/RQ011B as boundary/null evidence. Phrase as a stress test that did not demonstrate robust realised-harm prediction.

### R6 — Beyond-safety value

Do not claim yet. RQ013 should wait for RQ010B or be explicitly framed as likely bounded/null if based only on current OnSite evidence.

## Recommended program decisions

1. Continue RQ010B through the Route-4 QA decision. If the 256/full-data StreamPETR route fails, move quickly to Route 5 or close WOD as feasibility-bounded.
2. Do not run RQ013 as a positive-value synthesis until RQ010B resolves. Current OnSite-only evidence would likely yield a null/boundary synthesis.
3. Treat OnSite as stress-test/boundary evidence in the manuscript, not as validation success.
4. Drop counterpart-conditioned mechanism language from title, abstract, and headline figure captions.
5. Keep RQ012B's passivity-to-deadlock signal as a future-work hypothesis only.
6. Consider a dedicated future RQ for interaction-failure segment retrieval if the program wants another OnSite runtime-monitor test.

## Final review verdict

The recent Wave-B work improves the project by preventing overclaiming. The current evidence supports a usable empirical monitor, but not the stronger social-compliance validation story.

Recommended high-level paper-safe claim:

> We construct an estimability-aware, context-conditioned conformal envelope for online social-behaviour monitoring. It is sharper than a global norm and near-nominal marginally, but IPV-conditioning channels do not add measurable value over context, and current OnSite stress tests do not show robust realised-harm prediction. Independent human-preference validation remains pending WOD-E2E tracking quality.

Recommended prohibited claim:

> A counterpart-IPV verifier has been externally validated to predict social harm beyond safety metrics.
