# RQ004 Decision: IPV State-Space Organization

Status: ACCEPTED — bounded; supports R1 episode-level state organization (knowledge-layer freeze, human-directed 2026-06-24).

Runs: `RQ004_1/2/3` (RQ004_3 generalizable-conclusions is canonical).
Basis: Codex review (`reviews/codex_review.md`); frozen at PI direction.

## Accepted Claims

| ID | Claim |
|---|---|
| RQ004-KC-SURFACE | IPV/social compliance is a state-conditioned response surface over risk × geometry × role × time, not a single global score. |
| RQ004-KC-PRIORITY | Priority is risk-modulated, not a static label: priority-minus-nonpriority IPV +0.058 at PET≤1.0 s, ~0 mid-range, −0.034 at PET>2.0 s. |
| RQ004-KC-GEOMETRY | Coarse road geometry is a stable behavioural prior (MP vs non-MP, S-S vs non-S-S positive across all four sources); fine topology cells are too sparse for headline use. |
| RQ004-KC-AVHV | AV/HV sociality is not a fixed scalar trait (sign flips by dataset, risk, path state, priority boundary). |
| RQ004-KC-PRECONFLICT | First non-zero IPV often appears before the annotated conflict window (AV2 51.1% / Lyft 67.6% / Waymo 68.1% / nuPlan 75.1%) — descriptive/replay, not causal early-warning. |

## Rejected / Deferred

| Claim | Disposition |
|---|---|
| Generalizable cross-dataset state-space predictive law | Rejected (LODO negative; source imbalance). |
| AV/HV scalar sociality headline | Rejected (context-/source-dependent). |
| OnSite held-out validation | Deferred (protocol only; → RQ011). |
| Causal claims about social behaviour | Deferred (observational/replay). |

## Paper Handoff

Supports **R1 state-dependence** (the contextual-norm finding) as a state-conditioned response surface, not a transferable law. Cite RQ004_3 evidence + the RQ004_1 falsification table.
