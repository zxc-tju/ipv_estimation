# ChatGPT Review — RQ010B WOD-E2E Tracking + Preference Validity Status

Date: 2026-06-29  
Reviewer: ChatGPT  
Status: `review-complete` for current in-progress status  
Scope: RQ010B v3 plan, current HPC handoff notes, and RQ010 feasibility decision.

## Verdict

RQ010B remains the only active route that could provide a strong independent human-preference validation leg, but it is still an engineering-risk path. The statistical preference-validity test must not be run, cited, or interpreted until the rating-blind tracking layer clears its frozen QA gate.

## Current state

The v3 plan correctly frames the question:

> Among the three same-scene WOD-E2E candidate ego futures, does lower frozen-envelope deviation predict higher human preference, beyond a frozen kinematic+safety baseline and control battery?

The dataset facts are clear:

- WOD-E2E provides 8 surround cameras, camera calibration, ego history/future, routing command, and human-scored candidate ego futures.
- It does not provide the surrounding actor tracks, LiDAR, or HD map required to run the multi-actor verifier directly.
- Therefore full M3 testing requires a rating-blind multi-camera tracking layer.

## Engineering progress reviewed

The current operating notes indicate that Tongji HPC setup is active:

- E2E parser access is working under `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`.
- A four-shard structural pre-flight sampled candidate-bearing frames and passed the structural t* checks on those sampled frames.
- StreamPETR Route 4 infrastructure is installed.
- Waymo Perception v1.4.3 dev and finetune subsets have been downloaded and crc32c verified.
- Dummy, smoke, and small training runs have executed.

However, the first 64-train / 16-val detector-quality gate was poor:

- overall AP about `0.0033`;
- recall about `0.080`;
- precision about `0.033`;
- Pedestrian and Cyclist AP/recall/precision near zero.

A 256-train / 16-val StreamPETR finetune is or was running as the next Route-4 attempt. Route 5 remains the fallback if full-data Route 4 cannot pass the QA gate.

## Strengths

- The plan has the right lock structure: B1 tracking/QA before any rating unlock.
- Route choice is explicitly rating-blind.
- The plan blocks silent M2 substitution; if counterpart tracking fails, the study cannot pretend to have run full M3.
- The v3 plan defines a primary conditional-logit test, practical effect floor, kinematic+safety baseline, control battery, LOSO cluster structure, and rating-unlock checklist.

## Risks

1. **Tracking quality is the dominant risk.** Current low AP/recall means the first usable detector was not yet adequate for M3 or tracker QA.
2. **Class coverage is a risk.** Pedestrian/Cyclist near-zero performance would undermine long-tail interaction validity, especially for WOD-E2E scenario clusters involving pedestrians/cyclists.
3. **Map/corridor construction remains approximate.** WOD-E2E lacks HD map geometry, so corridor reconstruction and counterpart geometry must be treated as an engineered fallback.
4. **Support/abstention may shrink the denominator.** If too many scenes are OOD or fail tracking support, preference validity will be underpowered or bounded-negative.
5. **No rating-derived tuning is allowed.** Tracker QA, support gates, and inclusion must be frozen before rating values are read.

## Paper-safe wording now

> WOD-E2E is being prepared as an independent human-preference validation surface, but the released fields require a rating-blind multi-camera tracking layer before the frozen envelope can be tested.

## Prohibited wording now

Do not write:

```text
WOD-E2E validates the verifier.
WOD-E2E confirms human preference alignment.
The tracking route has passed.
M3 can be tested directly from released WOD-E2E fields.
A context-only M2 substitute validates the planned M3 claim.
```

## Decision points

Continue RQ010B through the Route-4 QA decision. If the 256/full-data StreamPETR route fails the frozen QA gate, move promptly to Route 5 or close WOD-E2E as a feasibility-bounded path.

Do not proceed to rating unlock until:

- full validation pre-flight passes;
- tracker QA passes;
- calibrated support/uncertainty gates are frozen;
- RQ009 scorer interface is frozen;
- the inclusion denominator satisfies the v3 minimum sample rules;
- all manifests are hashed and rating columns are still unread.

## Final recommendation

Keep RQ010B active. It is the most important remaining external-validity opportunity, but current evidence supports only an **in-progress tracking build**, not human-preference validation.
