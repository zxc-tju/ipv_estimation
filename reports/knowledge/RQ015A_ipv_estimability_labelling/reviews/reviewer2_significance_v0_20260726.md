# RQ015A v0 Independent Review — Reviewer 2 (Originality and Scientific Significance Emphasis)

## Review setup

- **Input scope:** Frozen plan `reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md`, SHA-256 `cd352390d816c12c77942211ad73479a5d3b71c43c8c801150ab3e25cfa9fea8`.
- **Integrity check:** `reports/plans/RQ015AB_split_checksums_20260726.sha256` verified `5/5 OK` before review.
- **Assessment boundary:** Independent, read-only review of the plan and cited RQ007/RQ009/RQ015 historical evidence. No RQ015A review written by another reviewer was read; no plan, code, data, or status index was modified.
- **Emphasis:** originality and scientific importance/significance, with shorter checks of technical soundness, interdisciplinary interest, and readability.
- **Shared claim presented by the plan:** Existing `ipv_error` values can be transformed into a row-level, auditable estimability state, then summarized at frame and case levels without changing the estimator or revising downstream results (`RQ015A plan:18-32,55-70,115-129`).
- **Visible evidence base:** the legacy warm-up initialization (`src/sociality_estimation/core/ipv_estimation.py:247-252,270-272`); the frozen RQ007 concentration-index findings and boundaries (`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:3,14-16,18-30`); the RQ007 split freeze (`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/split_freeze.json:36-48,76-114`); and the accepted RQ009 result and limitations (`reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md:8-25,29-37`).
- **Missing evidence affecting confidence:** no latent-IPV truth, recovery experiment, or calibrated mapping from concentration to estimation success/failure is supplied; the plan deliberately defers D1/D2/D3 separation to RQ015B (`RQ015A plan:29-32`).
- **Exact citation aliases used below:** `RQ015A plan` = `reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md`; `RQ007 decision` = `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`; `split_freeze.json` = `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/split_freeze.json`; `RQ007 final conclusions` = the same run's `02_process/11_final_review/conclusions.md`; `tau_selection.md` = the same run's `02_process/04_estimability/tau_selection.md`; `summary_sensitivity_method.md` = the same run's `02_process/07_summary_sensitivity/summary_sensitivity_method.md`; `RQ009 decision` = `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`; `RQ009 anchor_audit.json` = `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/anchor_audit.json`.

## Reviewer 2

### Overall assessment

RQ015A is a worthwhile **measurement-audit and research-governance study**. Its originality lies less in a new estimator or scientific mechanism than in turning a widespread, previously implicit numerical-quality signal into a corpus-wide provenance ledger, keeping frame-level and case-level exposure separate, and forcing downstream studies to state whether their estimands survive selection on measurement quality. That is valuable internal science and could become a reusable methodological pattern.

The present plan nevertheless overstates what its observed variable establishes. Its opening question asks whether each row has “actually measured IPV”, and its categories are named `ESTIMABLE` and `NOT_ESTIMABLE` (`RQ015A plan:18-21,55-65`). Yet the only row-level evidence is the dispersion of candidate weights. RQ007 explicitly freezes this quantity as an **identifiability proxy**, separates it from the current IPV and episode summaries, and defers latent-IPV truth (`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:15-16,24-25,30`). The plan also admits that the same near-uniform pattern cannot distinguish numerical underflow, flatness under the current grid/model, and model misfit (`RQ015A plan:29-32`). Therefore the current design can answer “how concentrated are the legacy candidate weights?” but not “was IPV measured successfully?”. This is a construct-validity blocker, not a wording-only preference.

A second blocker is the claim that frame/anchor products are “directly damaged” while case-level products are damaged less (`RQ015A plan:94-105`). The planned C0 phase correctly says that no downstream study will be re-estimated and that selection may change the estimand (`RQ015A plan:115-129`). Those safeguards are incompatible with declaring harm in advance. RQ007 established summary-definition sensitivity but explicitly did not adjudicate which summary is scientifically preferred (`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/11_final_review/conclusions.md:52-69`). RQ009, meanwhile, retained near-nominal marginal coverage in its original population and found no measurable advantage from the counterpart-IPV channel over the context-only envelope (`reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md:12,16-25`). RQ015A may identify **exposure to low-concentration proxy states** and prioritize audits; it cannot yet establish downstream damage or rank frame- versus case-level harm.

### Who would be interested in the results, and why

- Researchers in autonomous-driving interaction modelling and inverse planning would care because candidate-weight concentration is often silently treated as estimator confidence.
- Measurement-science and uncertainty-quantification readers would care about the separation between numerical completion, candidate discrimination, model adequacy, and latent-truth recovery.
- Runtime-monitoring and safety-verification researchers would care because abstention and selection can change the population on which a monitor is interpreted.
- Reproducible-computing readers may value the proposed artifact-level ledger and provenance reconstruction.

Interest outside the immediate IPV programme is currently **potential rather than demonstrated**. To support broader significance, the final report would need to present the result as a general measurement-audit pattern and show which conclusions are invariant to operational cutoffs, datasets, candidate-grid sizes, and aggregation rules. A project-specific relabelling of one error column, without construct validation or a generalizable audit result, is important quality control but not yet outstanding scientific importance.

### Major strengths

1. **The scope split is scientifically disciplined.** RQ015A does not modify the estimator, deploy an abstention gate, retrain M3, overwrite frozen artifacts, or reopen accepted decisions (`RQ015A plan:8-12,23-32,144-150`). This makes a bounded descriptive audit possible before RQ015B.
2. **The D0 correction is well grounded.** The active estimator initializes IPV to zero and the reliability field to one before writing estimates from the minimum-observation index onward (`src/sociality_estimation/core/ipv_estimation.py:247-252,270-272`), matching the plan's warm-up interpretation (`RQ015A plan:34-40`).
3. **The sealed-split boundary is appropriate.** The plan prohibits using RQ007 sealed cases in the final portrait (`RQ015A plan:74-78`), and a concrete outcome-blind case assignment already exists with 19,258 development, 7,628 guard, and 11,342 held-out cases (`split_freeze.json:36-48,76-114`).
4. **Frame and case units are kept visible.** Requiring both prevents millions of correlated frames from being mistaken for millions of independent scientific units (`RQ015A plan:92-105`).
5. **The C0 matrix is preferable to silently reopening downstream claims.** It asks for analysis unit, aggregation, estimand change, and selection-bias risk, then returns decisions to each owning RQ (`RQ015A plan:115-129`). The plan also correctly preserves the bounded status of RQ003 and RQ011B (`RQ015A plan:117-119`).

### Major concerns

#### B1 — The central success/failure interpretation exceeds the observable construct (`BLOCKER`)

**Evidence.** The research question uses “到底有没有测出 IPV” and calls the task a measurement study (`RQ015A plan:18-21`). The three-state rule converts weight dispersion directly into `ESTIMABLE / WEAK / NOT_ESTIMABLE` (`RQ015A plan:55-65`). However, the plan itself says RQ015A cannot distinguish D1 underflow, D2 flatness under the current grid/model, and D3 model misfit (`RQ015A plan:29-32`). RQ007 further states that the index is only an identifiability proxy, that concentration/current estimate/episode summary are distinct, and that latent IPV truth was not established (`RQ007 decision:15-16,24-25,30`).

**Why this matters.** A peaked weight vector may still be confidently wrong because all candidates fit poorly or the true value lies outside the grid; a diffuse vector may arise from a numerical defect, model insensitivity, or genuinely weak discrimination. Neither state alone validates or falsifies the estimated IPV. The label therefore measures **legacy candidate-weight concentration**, not measurement success.

**Required closure.** Either:

- rename the research question, ledger field, states, figures, and conclusions to operational concentration language such as `CONCENTRATED_PROXY / INTERMEDIATE_PROXY / NEAR_UNIFORM_PROXY`, with a prominent “not estimation success/failure” prohibition; or
- add independent construct validation against a known-truth or recovery benchmark, which would expand beyond the declared RQ015A scope and is better left to RQ015B.

Retaining the shorthand “estimability” is defensible only if every public-facing use is explicitly qualified as `operational_estimability_proxy`; “measured/not measured”, “successful/failed”, and equivalent language must be prohibited.

#### B2 — Downstream damage is asserted without a downstream estimand or reanalysis (`BLOCKER`)

**Evidence.** The plan labels RQ009/RQ014/RQ011B frame-level products “directly damaged” and case products “less damaged”, conditional on retaining only estimable frames (`RQ015A plan:94-105`). Yet Phase C0 expressly performs no downstream re-estimation and recognizes that selection on estimability may change both the estimand and exposure (`RQ015A plan:115-129`). RQ007 C3 establishes that summary choices differ, but its frozen conclusion says it does not determine which rule is scientifically preferred (`RQ007 final conclusions:52-69`). RQ009's accepted record reports original-population 90% coverage near 0.899 and a null counterpart-IPV increment over context-only (`RQ009 decision:12,16-25`); this evidence does not show damage from low-concentration target rows.

**Why this matters.** Prevalence is not damage. Damage requires a task-specific estimand and a contrast: prediction validity, calibration, association, ranking, or decision performance under a prespecified alternative measurement rule. “At least one/five concentrated frames” also does not establish that an episode summary is reliable, because those frames can be sparse and selectively located.

**Required closure.** Replace all harm language in RQ015A with **audit exposure / qualification risk**. The C0 matrix may state `potentially affected`, identify how the estimand would change, and route a sensitivity or redesign requirement to the owning RQ. It must not issue `damaged / less damaged / invalid` conclusions without the corresponding reanalysis.

#### M1 — The cutoffs are new operational policy, not an RQ007-validated success boundary (`MAJOR`)

**Evidence.** RQ015A uses `K_eff<=4` and `K_eff>=0.93K` without calibration (`RQ015A plan:55-69`). RQ007 did freeze the concentration metric, but its selected event-onset rule was a different development-derived threshold, `tau=0.292859481` with three consecutive frames (`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/04_estimability/tau_selection.md:3-8`), and its episode weighting was continuous rather than based on the proposed three bins (`.../02_process/07_summary_sensitivity/summary_sensitivity_method.md:19-31`).

**Why this matters.** The headline 24%/53% portrait is cutoff-dependent. PI adoption can legitimately freeze an operational rule, but it cannot convert that rule into validated measurement truth or claim that the cutoffs themselves were inherited from RQ007.

**Required closure.** State separately that (a) the metric is inherited from RQ007 and (b) the two cutoffs are new PI-governed operational cutoffs. Retain continuous `K_eff`, publish threshold-sensitivity curves/tables, and prohibit scientific interpretation of category names beyond candidate-weight concentration.

#### M2 — The proximity and “fixable versus intrinsic” interpretations outrun the planned evidence (`MAJOR`)

**Evidence.** The plan concludes from marginal distance-bin shares that estimability is not mainly proximity-driven and says predictor exploration will inform whether failures are fixable or intrinsic (`RQ015A plan:107-113`). But the same plan states it cannot distinguish D1/D2/D3 (`RQ015A plan:29-32`). RQ007's accepted C1 says most of its interaction-aligned concentration gap is proximity-compatible, with only a small conflict-geometry residual (`RQ007 decision:14,22-24,30,34`).

**Why this matters.** “Only 3.8% of frames are within 5 m” is a population-composition statement; it does not estimate the conditional effect or explanatory share of proximity. Predictor association with the proxy cannot separate a numerical defect from grid/model flatness or misfit. These are different estimands and mechanisms.

**Required closure.** Reframe Section 5.4 as outcome-blind exploration of **associations with the concentration proxy**. Separate population share, conditional rate, and incremental prediction. Do not infer reparability, intrinsic identifiability, or mechanism. Reconcile any proximity wording explicitly with RQ007's different interaction-gap estimand.

#### M3 — The exploratory and episode-summary analyses are not reproducibly specified (`MAJOR`)

**Evidence.** Section 5.4 lists candidate predictors but specifies no response parameterization, model family, case-grouped validation, missingness rule, interaction handling, multiplicity control, or effect-size/uncertainty output (`RQ015A plan:107-113`). Section 5.3 says it will “inherit and extend” RQ007 weighting but does not freeze the exact weighting formula or the handling of zero denominators and sparse cases (`RQ015A plan:100-105`). RQ007's actual continuous rule is explicit (`summary_sensitivity_method.md:19-35`) and its result is only a summary-definition sensitivity, not a preferred estimator (`RQ007 final conclusions:52-69`).

**Required closure.** Before Formal G1, freeze:

- the exact proxy outcome(s), including continuous `K_eff` as primary and categories as descriptive;
- case-grouped split/validation and case-clustered uncertainty;
- missingness and unknown-K handling;
- a minimal prespecified model/effect set plus multiplicity policy;
- the exact RQ007 weighting formula and a minimum effective-frame/support rule;
- interpretation limited to predictive/descriptive association and summary sensitivity.

### Technical failings that need to be addressed before the case is established

1. **Category overlap for small grids.** `ESTIMABLE: K_eff<=4` and `NOT_ESTIMABLE: K_eff>=0.93K` overlap when `K<=4`, but the plan accepts an arbitrary row-specific K (`RQ015A plan:57-64`). Freeze `K>=5` for eligible classification or define non-overlapping precedence and an unsupported-grid state.
2. **The split is already locatable and should be pinned, not optionally rebuilt.** The frozen artifact gives the case key, exact hash rule, bucket ranges, counts, and unit-ID artifact (`split_freeze.json:24-48,67-114`). RQ015A should name and checksum this artifact; rebuilding should be an exceptional byte-equivalence procedure, not an equal default (`RQ015A plan:74-78`).
3. **RQ009 unit language should be exact.** Its feature audit contains 6,397,266 anchors across all folds, of which 2,558,374 are train and 1,270,566 test (`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/anchor_audit.json:8-23`). Calling all 6.4 million “training and envelope” rows obscures the partition and may overstate the affected inferential unit (`RQ015A plan:100-101`).
4. **“No numerical values change” needs a boundary.** Row-level legacy IPV is not overwritten, but `K_eff`, operational states, filtered summaries, and weighted summaries are newly derived numbers (`RQ015A plan:21,55-65,100-105,131-138`). Say “no legacy IPV is re-estimated or overwritten.”

### Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| **Originality** | **Moderate as an audit pattern; low as a standalone scientific advance in the current form.** Corpus-wide provenance labelling plus unit-aware downstream qualification is useful. However, the metric and RQ007 weighting already exist; RQ015A currently adds operational binning and bookkeeping rather than a validated new construct or mechanism. |
| **Scientific importance / significance** | **High internal importance, potentially broader methodological importance, but not yet outstanding.** It can prevent invalid interpretation of numerical completion as measurement. Broader significance requires construct-honest labels, cutoff sensitivity, and evidence that the audit changes or bounds a scientifically meaningful conclusion; present downstream-harm claims are not demonstrated. |
| **Interdisciplinary readership** | **Potentially meaningful.** Measurement science, inverse problems, uncertainty quantification, autonomous systems, and runtime verification share this failure mode. The current IPV-specific vocabulary and unvalidated success/failure semantics limit reach. |
| **Technical soundness** | **Not yet established.** D0 and fold governance are strong, but the central proxy-to-success mapping is unsupported; cutoff provenance, predictor analysis, and summary comparison are insufficiently specified. |
| **Readability for nonspecialists** | **Structurally clear but semantically hazardous.** The plan is easy to navigate, yet “measured/not measured”, “directly damaged”, and “fixable versus intrinsic” make a proxy sound like ground truth. A simple mechanism schematic should separate latent IPV, candidate-model fit, normalized weights, concentration proxy, operational label, and downstream selection risk. |

### Recommendation posture

**Currently not established from the supplied evidence.** I support the bounded RQ015A audit after the two blockers are removed and the three major specifications are frozen. The most defensible scientific product is a **legacy candidate-weight concentration and measurement-risk portrait**, not a row-wise verdict on whether IPV was successfully measured and not a finding that downstream studies were already damaged.

## Risk / unsupported claims

- **Unsupported:** “This row did/did not measure IPV” from `ipv_error` alone (`RQ015A plan:18-21,55-69`).
- **Unsupported:** `NOT_ESTIMABLE` as a validated failure state rather than a near-uniform legacy-weight state (`RQ015A plan:61-69`).
- **Unsupported:** frame/anchor studies are already “directly damaged” and case studies are less damaged (`RQ015A plan:94-105`).
- **Unsupported:** predictor associations can distinguish fixable defects from intrinsic identification limits despite D1/D2/D3 being unavailable (`RQ015A plan:29-32,107-113`).
- **Overstated:** proximity is not a main predictor, based only on marginal bin composition; RQ007 supports a different, proximity-dominated conditional contrast (`RQ007 decision:14,22-24`).
- **Not assessable from the present plan:** generalization of category prevalences outside the legacy InterHub estimator/domain; correctness of any IPV value; impact on planner performance, human preference, safety, or downstream validity.

## Minimum closure conditions for re-review

1. Replace success/failure semantics with explicitly operational concentration-proxy semantics throughout the research question, states, figures, ledger, and acceptance criteria.
2. Replace all downstream-damage rankings with exposure/qualification-risk language and reserve damage/validity conclusions for owning-RQ amendments with task-specific reanalysis.
3. Identify the two cutoffs as new operational policy, retain continuous `K_eff`, and freeze sensitivity reporting.
4. Bound Section 5.4 to predictive association with the proxy and freeze a case-aware, reproducible analysis contract.
5. Freeze the exact episode-summary formulas and state that comparisons test definition sensitivity rather than identify the correct summary.
6. Pin the existing RQ007 split artifact/checksum and close K-domain/precedence ambiguity.

## Verdict

**`BLOCKED`**

- Blockers: **2**
- Major concerns: **3**
- Minor/technical clarifications: **4**
- `formal_g1_eligible=false`
- `execution_authorized=false`

The blockers are scientific claim-boundary defects, not objections to performing the audit. Once the study is reframed as a concentration-proxy portrait and downstream consequences are left as qualification risks, RQ015A becomes a coherent, valuable, and appropriately bounded measurement-governance study.

## Uniform post-review evidence challenge

### Challenge protocol

- This is an append-only post-review challenge. The original report, original verdict `BLOCKED`, and original counts `2 blocker / 3 major / 4 minor` above are preserved unchanged.
- I re-read only this Reviewer 2 report and the specified first-party contract, plan, code, manifest, and data artifacts. I did not read Reviewer 1, Reviewer 3, or any synthesis of their RQ015A opinions.
- The challenge asks whether evidence A–D adds or upgrades findings; it does not retroactively rewrite the original review.

### A + B — The RQ007 binding contract upgrades B1 from an inferred construct-validity defect to an explicit contract violation

**New evidence.** RQ007's binding execution contract separates five first-class concepts: current IPV, concentration index `c_i(t)`, opportunity `o(t)`, estimability `g_i(t)`, and behavioural dynamics (`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/binding_execution_contract.md:53-61`). It defines `g_i(t)=1` only as the conjunction of sustained low concentration, warm-up exclusion, active opportunity, surviving mechanical controls, and acceptable case health (`binding_execution_contract.md:71-79`). Most decisively, it requires concentration-index-only output to be reported as a weaker diagnostic and forbids naming or interpreting it as estimability (`binding_execution_contract.md:81`). RQ015A nevertheless asks whether IPV was “measured” and labels the `K_eff`-only bins `ESTIMABLE` and `NOT_ESTIMABLE` (`reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md:18-21,55-69`).

**Independent adjudication.** This does not create a separate finding from original B1; it **upgrades its evidentiary basis and severity rationale**. The original review inferred that concentration alone cannot establish success/failure. The binding contract now directly prohibits the proposed naming. B1 remains one blocker because double-counting the same semantic defect would inflate the tally.

**Minimum closure.** RQ015A must choose one of two non-hybrid contracts:

1. **Concentration-only audit:** rename the RQ, ledger field, states, figures, predictors, and conclusions to `concentration_proxy` language and forbid `estimability`, `measured/not measured`, and `success/failure`; or
2. **RQ007 estimability audit:** reconstruct the complete `g_i(t)` conjunction, including opportunity, sustained-window rule, controls, and case health. That is materially larger than the current RQ015A scope.

Adding a footnote that `ipv_error` is a proxy is insufficient while the state itself remains named `ESTIMABLE`.

### C — Sealed-inclusive threshold selection upgrades M1 to a blocker

**New evidence.** The plan says its displayed priors are full-corpus statistics containing the RQ007 sealed set (`RQ015A plan:39-53`), then immediately freezes the `K_eff<=4` and `K_eff>=0.93K` category cutoffs by PI decision without new calibration (`RQ015A plan:55-69`). It later says no threshold may be frozen until the split is located (`RQ015A plan:74-78`) and makes “sealed never participated in any threshold” an acceptance condition (`RQ015A plan:140-142`). RQ007's binding definition independently requires its concentration threshold to be justified from development/guard only (`binding_execution_contract.md:73-76`).

**Independent adjudication.** The cutoff values were adopted after the sealed-inclusive portrait was visible. Calling them “policy” rather than “calibrated” does not restore outcome-blind or sealed-blind selection. This is not merely the original M1 provenance ambiguity: it makes the plan's own acceptance statement unachievable on its stated history. I therefore **upgrade original M1 from `MAJOR` to `BLOCKER`**; it is not counted in both categories.

**Minimum closure.** The revised plan must state honestly that the legacy full-corpus aggregate informed the current proposal and must not claim the existing RQ007 sealed set was untouched for RQ015A threshold selection. It must then either:

- treat all RQ015A portraits as descriptive audits with no held-out confirmation claim, while reporting continuous `K_eff` and cutoff sensitivity; or
- freeze a new rule using only an admissible development set and reserve a genuinely untouched, newly defined confirmation split. The already viewed RQ007 sealed aggregate cannot be made unseen.

### D1 — M3 prediction rows are not anchors; this adds a major unit-of-analysis finding

**New evidence.** The frozen M3 test prediction manifest records alpha levels 80/90/95, `1,270,566` anchors, and `3,811,698` prediction rows—exactly three rows per anchor (`data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/prediction_manifest.json:135-145`). RQ015A only says that each artifact must report its true analysis unit (`RQ015A plan:89-90`) and elsewhere loosely describes 6.4 million M3 anchors (`RQ015A plan:100-101`); it does not freeze deduplication keys or prohibit treating alpha-expanded prediction rows as independent measurement rows.

**Independent adjudication.** This is a **new `MAJOR`** finding. It does not by itself invalidate the concentration-only audit, because the same anchor label can be joined to all three alphas. But without a canonical anchor grain, any prediction-row portrait triples denominators, produces pseudoreplication, and can make alpha expansion look like additional evidence.

**Minimum closure.** Freeze the canonical M3 estimability unit as the unique anchor key (at minimum `fold + case_key + anchor_frame_index + perspective`, with tier/version where needed). Compute and count the concentration label once per anchor. Treat `alpha` as a repeated interval level, not a new IPV measurement; any alpha-specific table must preserve anchor clustering and show both unique anchors and expanded rows.

### D2 — OnSite D0 is estimator-local, not global `frame_index`; this adds a blocker

**New evidence.** The active estimator initializes arrays by the local input-sequence length and writes results only for local loop position `t >= min_observation` (`src/sociality_estimation/core/ipv_estimation.py:247-252,270-272`). OnSite's recorded estimator contract uses `min_observation=4` (`data/derived/onsite_competition/RQ011B_matched_scenario/RQ011B_1_matched_scenario_20260625T202454_8331bd49/onsite_ipv_channel_target_exact_hw10_summary.json:55-60,72-85`). The named OnSite time-series artifact begins cases at native/global frame indices other than zero while its first four estimator-local rows retain the zero/one sentinel; therefore global `frame_index<4` is not the D0 predicate. RQ015A nevertheless freezes exactly that predicate (`RQ015A plan:36-38,61-70`) while explicitly including RQ012B OnSite in scope (`RQ015A plan:89-90`).

**Independent adjudication.** This is a **new `BLOCKER`**. Applying the frozen rule as written would misclassify OnSite's warm-up sentinels as attempted estimates whenever the extracted/replayed sequence begins at native frame index greater than three. Conversely, global frame number cannot prove estimator-local warm-up after cropping, resampling, or rolling reconstruction.

**Minimum closure.** Define D0 as `estimator_local_position < min_observation` within the exact estimator invocation, never as an unqualified global `frame_index` rule. The ledger must store or reconstruct `estimator_sequence_id`, local position, history/window contract, and estimator version. If these are not recoverable for an artifact, assign `unknown`/L2-L4 rather than infer D0. Add an OnSite fixture whose native frame index starts above four but whose first four local outputs are correctly labelled D0.

### Post-challenge verdict and count reconciliation

| Item | Original disposition | Post-challenge disposition | Count effect |
|---|---|---|---:|
| A+B full RQ007 `g_i(t)` contract | B1 blocker | B1 blocker, evidence upgraded | 0 |
| C sealed-inclusive cutoff history | M1 major | M1 upgraded to blocker | blocker +1; major -1 |
| D1 M3 alpha-expanded predictions | not separately recorded | new major | major +1 |
| D2 OnSite estimator-local D0 | not separately recorded | new blocker | blocker +1 |

- **Original preserved verdict/counts:** `BLOCKED`; `2 blocker / 3 major / 4 minor`.
- **Final post-challenge verdict:** **`BLOCKED`** (unchanged).
- **Final post-challenge counts:** **`4 blocker / 3 major / 4 minor`**.
- `formal_g1_eligible=false`.
- `execution_authorized=false`.

The final blocker set is: (1) concentration-only output is misnamed/interpreted as estimability or measurement success; (2) downstream damage is asserted without task-specific reanalysis; (3) sealed-inclusive evidence makes the claimed threshold-freeze history internally impossible; and (4) D0 is not portable across artifacts under the plan's global-frame predicate. Formal G1 should not begin until all four are closed on a newly checksum-frozen plan.
