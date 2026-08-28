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

## Amendment 2026-08-22 — RQ004_2 semantics claims accepted for PAPER001 (PI-directed)

Status: ACCEPTED for manuscript use. PI instruction of 2026-08-22 (Fig. 2
two-layer restructure: 先讲语义层面, 再讲具体构建) pulls the RQ004_2
multi-agent conclusions into the paper; the 2026-06-24 freeze above predates
that package and never registered it. Basis: RQ004_2 close-out
(`reports/studies/RQ004_ipv_state_space/RQ004_2_nature_conclusions_multiagent_20260618/`,
round-3 findings endorsed 4:0), frozen tables under
`02_process/agent_pair_asymmetry/tables/`.

### Accepted claims (numbers frozen; no recomputation permitted)

| ID | Claim | Frozen evidence |
|---|---|---|
| RQ004-KC-COMPLEMENT | Within one interaction the two partners' preferences are complementary: one gives way as the other presses. | Exchangeable ICC −0.338 [−0.351, −0.325] against a scenario-matched shuffled-partner null +0.041 [+0.030, +0.052], n = 34,757 matched-support dyads (99.7% of 34,850 valid cases); independent second implementation −0.339 (n = 34,645); per-source ICC negative in every corpus (nuPlan −0.405, Waymo −0.357, Lyft −0.311, AV2 −0.226). `round3_c1_static_implementation_comparison.csv`, `dataset_dyad_metrics.csv`. |
| RQ004-KC-EARLYLOCK | The role division is set early: the eventual asserter is legible from the first 5% of the interaction from IPV alone (AUC 0.75, n = 24,872), rising to 0.81 by half-way, while kinematics alone stays ≈ 0.56–0.57 at every window. | `round3_c1_role_lock_auc_source.csv` (cluster bootstrap over scenes; n grows 24,872 → 31,831 with window because short cases drop out). |
| RQ004-KC-PREYIELD | Yielding is pre-encoded in the preference: turning vehicles sit +0.079 [+0.063, +0.095] rad above their straight-driving partners (n = 3,565 pairs, BH p = 4.2e-21), direction replicated in all four sources; the nuPlan interval admits zero and is drawn as no-claim grey. | `table_turning_pairs.csv`, `table_turn_dataset_replication.csv`. |

### Presentation decisions

| ID | Decision |
|---|---|
| RQ004-PRES-PLANE | The paired-preference plane — each valid case's (agent-1 mean IPV, agent-2 mean IPV) from `round3_case_level.csv`, n = 34,850, each case plotted in both orders because partners are exchangeable, summarised by mass contours — is a pure presentation of an existing frozen table. Every statistic quoted on or around it comes from the frozen tables above; the panel computes no new estimate. The plane shows all 34,850 valid cases while the ICC is evaluated on the 34,757 with matched shuffling support; panels state their own n. |
| RQ004-PRES-BORROW | From the PNAS sibling draft (202508 Sociality Pattern Analysis) the manuscript borrows two idioms only: the paired-IPV plane presentation and the "two-sided script" concept prose. No number, estimate, or dataset split from that draft may enter PAPER001 (different estimator configuration and corpus; no decision record there). |

### Standing prohibitions unaffected

The role-gap panel stays out ("carrying rhetoric, not evidence"); the
PET-binned complementarity dose-response stays out of the manuscript; the
priority-left-turn reversal row is not cited in the paper (PET-adjacent
directional claim).

### Paper handoff

Feeds PAPER001 Fig. 2 layer 1 (semantics) and the §2.2/§2.4/Discussion prose;
manuscript-side registration in the paper repo `claims_register.md` cites this
amendment.
