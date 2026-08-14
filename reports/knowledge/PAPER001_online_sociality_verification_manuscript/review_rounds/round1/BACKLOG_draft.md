# Backlog draft — items requiring new data, experiments, or analyses (PI ruled: log, do not execute)

Ordering: by leverage on the referees' accept/reject probability, highest first. Each item: what it needs, which reviewer demands it resolves, what claim it would upgrade if done. Cross-refs are AGGREGATION item IDs (R1-xx) and reviewer item IDs.

## B01 — Independent construct validation of the IPV / flags
- Needs: an external anchor for the monitored scalar — blinded human ratings of flagged vs situation-matched unflagged clips, convergent/discriminant validity against SVO or courtesy constructs, or planted-preference / known-ground-truth simulation recovery; plus estimator error calibration curves linking the reliability statistic to estimation error.
- Resolves: A-M1/P1/q1-q2; B-M3/P3/q2; C-M9/q12; D-M1/P7 (R1-30).
- Upgrades: the central framing from "conditional trajectory atypicality under this estimator" toward "social preference monitoring"; without it the title/abstract framing decision (E4) stands in for it.
- Constraint memo: the human-preference route via released ratings was TERMINATED with a bounded null (register C4W; RQ020: n=75, ρ=+0.148, permutation p=1.0; two hard data walls). Any new attempt needs a different instrument (e.g., purpose-collected judgements), not a rerun; and the register bans arguing the preference rung in either direction meanwhile.

## B02 — Situation-conditioned kinematic-baseline monitor (incremental-value control)
- Needs: build a reference range on plain kinematic targets (closing speed, gap-closure rate) under the IDENTICAL conformal pipeline, frozen before comparison; test whether IPV flags carry information about the Fig. 5 signature beyond it (matched/weighted comparison, incremental prediction).
- Resolves: A-M3/P3/q7; B-M5/q8-q9; D-M2/P2/q4 (R1-11, R1-38).
- Upgrades: would either license a true "registers what kinematics do not" incremental claim (currently only the dissociation is claimed, per register C6) or cap the contribution — D marks this as the experiment "on which the paper's 'beyond safety' thesis stands or falls".
- Constraint memo: register C6 currently rules the increment NOT claimed; running this experiment is a PI-level strategic decision because a null result would be discoverable.

## B03 — Prospective landmark redesign of the consequence analysis
- Needs: first-eligible-alert-per-run analysis with a washout longer than the estimation window; non-overlapping exposure/outcome windows; pre-trend and lead/lag plots; placebo alert times; exclusion of cases with counterpart braking already under way; sensitivity to horizon/persistence definitions; hierarchical inference at team/run/scenario levels.
- Resolves: A-M2/P2/q4; B-M5/P5/q7; C-M5/P3/q7; D-M2 (R1-31).
- Upgrades: "is followed by" from a per-moment descriptive association to a prospectively-timed claim; would also answer the ED2 optics (deceleration already under way).
- Partial existing evidence to reuse: placebo-exposure permutation (passes, p=0.0199) and the event-level companion tests (null — RQ019 B8) already exist frozen.

## B04 — Sequential/episode-level false-alarm calibration + persistence-rule disclosure
- Needs: retrieve the frozen persistence/warning rule parameters (they exist in configs, not in the knowledge layer), then calibrate and report an episode-level estimand (probability of any false flag per interaction, flag-cluster rate, run length) on independent data; report flagged-stretch durations and verdict flicker.
- Resolves: C-M3/P1/q6; B-M2 (sequence part), B-q; D-q5 (R1-32).
- Upgrades: "runtime monitor"/planner-warning claims to a validated sequential operating point; currently only the pointwise 90 % level is calibrated.

## B05 — Cluster-aware conformal validity + per-source/state-stratified test coverage
- Needs: case/scene-level coverage uncertainty (cluster bootstrap or analytical), coverage/width/abstention stratified by source, geometry, risk, progress on the untouched test fold; optionally a cluster-conformal or one-anchor-per-case construction as a sensitivity.
- Resolves: A-M4/P4; B-M2/P2/q5; C-M1/M2/P1-P2/q3-q4; C-m1 (R1-02, R1-36).
- Upgrades: the calibration claim from "marginal over accepted moments" to group-conditional statements with honest uncertainty; complements the LODO numbers being surfaced now (P21).

## B06 — Full reproducibility specification of the reference model
- Needs: document the quantile-regression model family, features/timestamps, hyperparameters, seeds, and training details (the model exists only as a frozen pickle + manifest; the knowledge layer does not describe the family); produce the participant-to-window flow diagram and a leakage-audit note.
- Resolves: C-M4; B-M1/P1; A-M7; D-M7(v) (R1-19-part).
- Upgrades: reproducibility from "frozen artefact exists" to "independently reimplementable"; low scientific risk, high review value.

## B07 — Estimator sensitivity analyses (σ, grid density, window, reference geometry, map error)
- Needs: unfreeze-safe sensitivity sweeps on held-out data: likelihood scale σ (RQ006 exists but is `pending-review — no claims frozen`), candidate-grid density (0.20 vs 1/7 threshold behaviour), window length, lane-reference perturbations, counterpart mis-assignment.
- Resolves: A-q2; B-M3/q2, B-m1-part; C-M9 (R1-47).
- Upgrades: the readability gate from "frozen convention" to "validated measurement property"; also feeds B01's error-calibration curves.
- Note: RQ006 must pass its own review before anything from it is citable.

## B08 — Non-inferiority/equivalence analysis for "no added emergencies" + multiplicity handling
- Needs: pre-specified primary emergency endpoint(s), substantive non-inferiority margins, one-sided simultaneous cluster-aware bounds; multiplicity-adjusted intervals across the threshold battery.
- Resolves: C-M6/P4/q9; D-M6; A-m8; B-m10 (R1-35, R1-44).
- Upgrades: "no increase detected" (current, descriptive) to "increase excluded up to margin δ"; the abstract's "without added emergencies" (E3) can only return in its strong form after this.

## B09 — Uncertainty for the remaining descriptive panels
- Needs: case-clustered CIs for Fig. 2b (movement of the reading) and Fig. 2c (episode-summary disagreement; per-pair flip rates with n), CIs for Fig. 3c LODO R² per fold, CI + formal interaction test for the Fig. 3a middle-risk band.
- Resolves: A-m3/m4/m5; B-m6; C-m2 (R1-24, R1-25-part).
- Upgrades: descriptive panels to inferentially interpretable ones; cheap analyses, but they touch frozen figure data so they are new computations by definition.

## B10 — Formal equivalence margin for the null ablations
- Needs: cross-validated incremental performance of counterpart-IPV / self-history channels with confidence intervals against a pre-specified equivalence margin (the frozen paired difference −0.0002, p=0.863 becomes the point estimate inside a TOST-style statement).
- Resolves: C-M10(i)/q13 (R1-29).
- Upgrades: "adds no measurable value" from a null-result narrative to a bounded equivalence claim, which is what makes the headline event-vs-case tension referee-proof.

## B11 — C7 completion: real human-arm data, verification, and RQ022 decision
- Needs (already specified end-to-end in `.codex-fleet/rq022-matched-scenario/work/T1_target_figure/DATA_INTERFACE.md`): offline-server measured values for every `human_arm_data.json` field; moment-level and per-unit parquet tables returned; supervising analysis blind-recomputes every number; `data_status: REAL_VERIFIED`; figure regenerated without watermark; `\targetnum` digits swapped; claims accepted in `reports/knowledge/RQ022_*/decision.md`.
- Resolves: the entire human-arm statistical battery — A-M5/P5/q10-q11; B-M6/P6/q11-q12, B-m17; C-M7/M8/P5/q10-q11; D-M4/M6/q7 (R1-33, R1-34, R1-45).
- Upgrades: §2.5 and Fig. 6 from PI-attested target prose with synthetic digits to citable results; unlocks per-driver dispersion, flag-rate CI (field already in the interface), zero-cell accounting, and the per-scenario paired contrast with real numbers.
- Note: endpoints are pre-registered (calibration transfer, frozen C5b battery, per-scenario contrast); adding referee-demanded endpoints (equivalence margins, driver-type × flag interaction tests, scenario-standardised rates) is an endpoint change → PI decision first (E2/E10).

## B12 — Crossed-design inference for the AV–human contrast
- Needs: once C7 real data exist — multilevel or two-way (driver/system × scenario) bootstrap for the flag-rate ratio and per-scenario ordering; scenario-standardised rates; sensitivity to equal-weighting choices; report effective independent units (19 team clusters on the AV side).
- Resolves: C-M7/q10; A-M5 (2.1-fold CI); B-M6; D-M4 (R1-34).
- Upgrades: the "flagged about twice as often" contrast (currently a synthetic placeholder) to an uncertainty-qualified, weighting-explicit statement.

## B13 — Online implementation study (latency, throughput, faults)
- Needs: end-to-end causal implementation on representative hardware at 10 Hz; median/p99/worst-case latency; deadline-miss and stale-input behaviour; fault injection (identity switches, map error, missing tracks); an assume–guarantee statement for what holds during abstention.
- Resolves: B-M4/P4/q6, B-q14; A-M7/q13 (R1-37).
- Upgrades: "online/runtime" from computability (non-anticipative inputs — already demonstrated) to demonstrated real-time operation; determines whether E9's positioning language can strengthen later.

## B14 — Open the sealed confirmatory split (PI-gated, irreversible)
- Needs: explicit PI authorization under the frozen governance (RQ007 held-out confirmation; RQ008 Wave-B amendment first); then confirmatory replication of the Fig. 2 readability results.
- Resolves: A-M8-part/q14; B-q (dev/confirmatory); C-M4-part (R1-43).
- Upgrades: Fig. 2's claims from development/guard-provisional to held-out-confirmed. Do NOT do this casually: single-shot evidence.

## B15 — Leakage-audit document for the split/feature pipeline
- Needs: an auditable note tracing that no future information, outcome-adjacent transformation, or cross-fold scene leakage enters the features (the frozen feature-contract checks exist; the narrative audit does not), including the timestamp table B-M1 asks for.
- Resolves: C-M4; B-M1 (timing table); A-q8 (R1-19-part).
- Upgrades: "frozen and held sealed" from assertion to audit trail.

## B16 — Benchmark documentation pack
- Needs: scenario-template descriptions (A1–C4: geometry, counterpart script), per-template run counts, system categories (production vs research prototypes) — requires benchmark-organiser material not in the workspace.
- Resolves: D-M5/q2; A-m10 (labels A3/A5/B1); B-q13-part (R1-46-part).
- Upgrades: benchmark external validity assessment; also feeds E6 (edition/year).

## B17 — Abstention–kinematics dependence check ("can a planner evade the monitor by staying unreadable?")
- Needs: association analysis between unreadability/abstention reasons and assertive kinematics on the benchmark; report abstention-by-reason per scenario/system.
- Resolves: D-q6; C-M2-part (abstention masking) (R1-36-part).
- Upgrades: the abstention discipline from "auditable as such" to "audited"; closes a gaming loophole argument reviewers will press on resubmission.
