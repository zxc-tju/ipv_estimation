# RQ015A v1 Independent Review — Reviewer 2 (Originality, Scientific Significance, and Claim Boundaries)

## Review setup

- **Frozen object:** `reports/plans/RQ015A_plan_v1_attempt_status_and_weight_concentration_audit_20260726.md`.
- **Verified SHA-256:** `3c77f9713153a22772d92adfa7841f48a919ba10782b15baea3ecdc3e6367b04` (exact match to the assigned review object).
- **Checksum bundle:** `reports/plans/RQ015A_plan_v1_checksums_20260726.sha256` verified **6/6 OK** before review.
- **Permitted context used:** the v0 three-reviewer synthesis, the sealed-exposure disclosure and PI ruling, the RQ007 binding execution contract, and accepted RQ007 decisions. No new v1 review by Reviewer 1 or Reviewer 3 was read, and no communication with another review lane occurred.
- **Review emphasis:** originality and scientific significance, with particular attention to construct narrowing; the scientific objective, selection rule, guard role, sensitivity and interpretation of the two provisional thresholds; proximity; episode summaries; downstream qualification risk; the sealed waiver; and the exact claims this audit can establish.
- **Citation aliases:** `v1 plan` = the frozen RQ015A v1 plan above; `sealed disclosure` = `reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md`; `RQ007 contract` = `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/binding_execution_contract.md`; `RQ007 decision` = `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`; `v0 synthesis` = `reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_three_reviewer_synthesis_v0_20260726.md`.

## Reviewer 2

### Overall assessment

RQ015A v1 makes a substantial and scientifically important correction. It no longer equates candidate-weight concentration with successful IPV measurement or with RQ007 estimability. The revised title, research question, labels, explicit prohibition, and limitation section consistently define a retrospective audit of **estimation attempt provenance** and **candidate-weight concentration** (`v1 plan:14-40,71-75,187-192`). This closes the central construct-substitution blocker from v0 in substance. The study can now be coherent without latent-IPV truth or a reconstruction of the full RQ007 conjunction.

The resulting contribution is valuable primarily as measurement science and research governance. Its originality is not a new estimator, a new interaction mechanism, or a validated confidence score. It is the cross-artifact separation of numerical attempt state, recoverability, effective candidate-weight support, analysis unit, and downstream selection exposure. That is high-value internal science and could become a transferable audit pattern for inverse-planning pipelines.

However, two central deliverables remain scientifically underdetermined. First, the plan requires `c_lo` and `c_hi` to be “rederived” from development plus guard data but gives no scientific objective, estimand, selection algorithm, sampling weights, source/K transport rule, validation criterion, or sensitivity decision rule (`v1 plan:77-88`). A checksum can freeze an algorithm after it is written; it cannot supply the missing scientific rationale. Second, the proposed downstream matrix still permits the conclusion `低资格风险` without a task-level reanalysis and provides no deterministic rubric or indeterminate state despite explicit `unknown` artifacts (`v1 plan:157-170`). These are Formal-G1 blockers because every final portrait and downstream qualification statement depends on them.

### Who would be interested in the results, and why

- **Inverse-planning and autonomous-driving researchers** would benefit from seeing how often a legacy estimator was not run, produced concentrated candidate support, or remained near uniform under its frozen grid and model.
- **Measurement-science and uncertainty-quantification readers** would value the explicit distinction between execution, discrimination among candidates, model fit, accuracy, and latent-truth recovery.
- **Runtime-verification and safety researchers** would care because filtering on a measurement-quality proxy can change both the analytic population and the estimand.
- **Reproducible-research readers** may find the artifact-level ledger, unit conservation, missingness states, and downstream ownership boundary generalizable.

The wider scientific significance will depend on whether continuous concentration distributions, source/configuration heterogeneity, dependency-aware uncertainty, and threshold sensitivity are treated as primary evidence. A project-specific three-bin count alone would remain useful quality control, but would not constitute a broad scientific advance.

### Major strengths

1. **The construct has been honestly narrowed.** The plan explicitly states that `K_eff` cannot establish whether IPV was truly measured, replaces `ESTIMABLE/NOT_ESTIMABLE` with concentration-only labels, and prohibits restoration of estimability language without the complete RQ007 conjunction (`v1 plan:14-29`). This now conforms to RQ007's separation of current IPV, concentration index, opportunity, estimability, and behavioural dynamics (`RQ007 contract:53-81`).
2. **Attempt status and concentration status are orthogonal.** `NOT_ATTEMPTED` is routed before the three concentration bins, preventing warm-up sentinels from being interpreted as near-uniform estimates (`v1 plan:50-69`).
3. **The small-K overlap and rounded-boundary defects are corrected.** Normalization by actual `K`, mutually exclusive intervals, exact transformation back to the stored index, and `unknown` when K is unavailable are all meaningful improvements (`v1 plan:42-69`).
4. **Cross-artifact hazards are visible.** The plan recognizes M3 alpha-row triplication, OnSite estimator-local warm-up, missing WOD concentration fields, and the need for ledger keys and row conservation (`v1 plan:90-105`).
5. **The sealed exposure is no longer hidden.** The plan records that the original cutoffs were written after sealed-inclusive aggregate inspection, demotes them to provisional values, excludes sealed data from RQ015A conclusions, and links the PI ruling (`v1 plan:77-88,107-125`; `sealed disclosure:7-29,62-84`).
6. **Downstream damage language has been withdrawn.** The plan correctly states that RQ015A will not declare a downstream study damaged or less damaged without task-specific re-estimation (`v1 plan:157-170`).

### Major concerns

#### B1 — The two-threshold rederivation has no scientific selection contract (`BLOCKER`)

**Evidence.** The existing values `4/7` and `0.93` are correctly marked `PROVISIONAL_PENDING_DEVGUARD_REDERIVATION`. The plan says that a rule will be frozen before computation and that both boundaries will be rederived on deduplicated development plus guard rows (`v1 plan:60-61,77-88`). It does not state:

- what scientific property `c_lo` and `c_hi` are intended to identify;
- what loss, stability target, density feature, quantile, mixture model, or semantic anchor selects them;
- whether rows, cases, perspectives, artifacts, configurations, K values, or sources receive equal weight;
- whether one global pair of cutoffs is assumed transportable across grids, estimator windows, sources, and sampling rates;
- what role remains for guard after it is pooled into cutoff selection;
- what uncertainty, robustness, reassignment rate, or failure criterion would reject unstable cutoffs.

**Why this matters.** The audit has no external truth label or downstream outcome that can define an optimal boundary, and downstream outcomes are correctly forbidden as threshold criteria (`v1 plan:194-198`). Therefore an empirical marginal distribution cannot by itself reveal the scientifically “correct” transition from `CONCENTRATED` to `INTERMEDIATE` or from `INTERMEDIATE` to `NEAR_UNIFORM`. Raw-row pooling would also allow long cases and the dominant artifact/configuration to determine the policy. Normalizing `K_eff` by K removes a scale difference, but does not prove measurement invariance across candidate grids or estimators.

Pooling development and guard for selection also removes the guard's independent validation role. This conflicts with the logic of an outcome-blind guard: once guard influences the cutoffs, stability on guard is no longer external evidence. The PI condition to rederive from dev+guard is a valid governance requirement, but it is not yet a complete scientific method (`sealed disclosure:71-84`).

**Required closure.** Before Formal G1, freeze a threshold SAP that does all of the following:

1. makes continuous `K_eff/K` the primary audit result and the three bins explicitly secondary operational summaries;
2. defines the scientific semantics of each boundary independently of desired prevalence or downstream findings;
3. specifies one reproducible selection rule, eligibility universe, case/source/configuration weighting, missingness handling, tie/boundary handling, and seed if applicable;
4. uses **development for selection and guard only for locked validation**, or explicitly states that pooled dev+guard yields no independent guard validation and creates a separate admissible validation design;
5. prespecifies guard pass/fail checks for cutoff stability, label reassignment, source/K heterogeneity, case-level prevalence, and episode-summary sensitivity;
6. publishes a threshold grid or perturbation analysis around both boundaries and states what instability would cause bins to be withheld while retaining the continuous portrait;
7. treats any cross-source/global cutoff as an assumption to be tested, not as a consequence of K normalization.

If the PI requires literal pooled dev+guard rederivation, the honest interpretation is a **post-exposure, descriptive policy binning with no independent confirmation**. It may still be useful, but it must not be described as scientifically validated or as guard-confirmed.

#### B2 — The qualification-risk verdicts remain unsupported and non-reproducible (`BLOCKER`)

**Evidence.** Phase C0 requests exposure proportions, estimand-change checks, selection-bias risk, and one of three conclusions: `低资格风险 / 存在资格风险，需任务级重估 / 不适用` (`v1 plan:157-170`). No deterministic mapping from the evidence fields to those conclusions is given. Meanwhile the same plan recognizes artifacts with unknown K, missing `ipv_error`, pending RQ015B recovery, and source-specific unresolved attempt provenance (`v1 plan:69,94-105,127-136`). There is no `INDETERMINATE/UNKNOWN_REQUIRES_OWNER_REVIEW` verdict.

**Why this matters.** Exposure prevalence is not validity. Even a small selected subset can have high influence, and a large near-uniform subset can be irrelevant to a downstream estimand that does not use that measurement. Conversely, the absence of an audit trigger cannot establish “low qualification risk” without task-specific influence, calibration, association, or decision analysis. The plan explicitly declines that reanalysis, so `低资格风险` exceeds what RQ015A can prove. The missing indeterminate state would force unknown evidence into a falsely reassuring, adverse, or not-applicable category.

**Required closure.** Replace the three unconstrained verdicts with a machine-decidable audit-routing rubric. At minimum distinguish:

- `NOT_APPLICABLE` — the downstream estimand does not consume the audited IPV field;
- `NO_AUDIT_TRIGGER_DETECTED` — all required provenance is available and the prespecified exposure/estimand-change trigger is absent, explicitly **not** evidence of low scientific risk;
- `OWNER_REANALYSIS_REQUIRED` — selection changes the analytic population/estimand or crosses a prespecified exposure trigger;
- `INDETERMINATE_UNKNOWN_PROVENANCE` — required concentration, K, attempt, join, or aggregation evidence is unavailable.

The rubric must freeze thresholds, denominators, precedence, unknown handling, and evidence required for each state. RQ015A may route work; it may not certify downstream validity.

#### M1 — The proximity statement still exceeds the evidence and conflicts with the accepted boundary (`MAJOR`)

**Evidence.** The plan states that prior evidence shows concentration is **not mainly determined by proximity**, reasoning that the `<5 m` bin has a higher rate but covers only 3.8% of frames (`v1 plan:145-149`). Population share does not estimate the conditional association or explanatory contribution of proximity. RQ007's accepted C1 addresses a different, interaction-aligned estimand and concludes that most of its total concentration gap is proximity-compatible, with only a small conflict-geometry-specific residual (`RQ007 decision:14,22,28-34`).

**Required closure.** Treat proximity as an open descriptive covariate. Separate (i) frame-population composition, (ii) conditional concentration prevalence, and (iii) incremental association after source, case, observation history, configuration, and geometry. State explicitly that RQ015A does not adjudicate mechanism and that its whole-corpus marginal portrait is not a refutation of RQ007's opportunity-window contrast. “Provides hypotheses for RQ015B” is acceptable; “shows proximity is not main” is not.

#### M2 — The episode-summary comparison is neither frozen nor interpretation-safe (`MAJOR`)

**Evidence.** The plan proposes “only `CONCENTRATED` frames” and “concentration-weighted” summaries while saying it inherits and extends RQ007-KC-C3 (`v1 plan:151-155`). It does not define the exact continuous weight function, whether the weight increases or decreases with the newly defined `c`, the analysis unit, perspective/configuration separation, minimum support, zero denominator, unknown-state handling, case inclusion, or dependence-aware uncertainty. RQ007 C3 established **definition sensitivity**, not that weighting or hard filtering yields a scientifically preferable episode IPV (`RQ007 decision:15-16,24-25`).

**Required closure.** Freeze the formulas and eligibility rules before execution. Include the unfiltered attempted-frame summary as a reference, treat hard-filtering and continuous weighting as alternative definitions, and report support loss plus case-level changes. State that the analysis measures **summary-definition sensitivity** only; a lower sign-flip rate is not validation, correction, or evidence that one episode summary is more accurate.

#### M3 — The sealed waiver must remain a governance decision, not a scientific “unaffected” result (`MAJOR`)

**Evidence.** The disclosure accurately states that sealed-inclusive aggregate statistics informed the original thresholds (`sealed disclosure:7-29`) and the PI adopted interpretation A with an attached rederivation condition (`sealed disclosure:62-84`). The plan repeats that RQ007's held-out confirmation path is “unaffected” (`v1 plan:107-125`). PI authority can waive a governance breach and authorize a future operation; it cannot make previously observed aggregate information scientifically unobserved.

**Required closure.** In every report and handoff, phrase this as: **“PI governance waiver granted after aggregate sealed exposure; no row-level held-out values or held-out inference were used; RQ015A excludes sealed; any future RQ007 confirmation proceeds under this disclosed waiver.”** Do not use “unaffected” as an empirical claim of pristine independence. The waiver is sufficient to continue once the attached conditions and the new scientific SAP are satisfied, but it is not evidence that contamination risk is exactly zero.

#### M4 — “Two directly observable questions for every row” is not yet true across the declared artifact scope (`MAJOR`)

**Evidence.** The research question promises a two-state attempt answer and a concentration bin for existing artifacts (`v1 plan:31-40`). Yet the execution table leaves OnSite local-position semantics to be confirmed before Phase A, leaves WOD rules to per-artifact confirmation, and explicitly identifies artifacts whose concentration field is unavailable (`v1 plan:90-105,127-136`). The three concentration states allow `unknown`, but the attempt-state contract itself remains only `ATTEMPTED / NOT_ATTEMPTED`.

**Required closure.** Qualify the research question as “where artifact provenance supports the determination” and add orthogonal `attempt_status=UNKNOWN` and `concentration_status=UNKNOWN/NOT_APPLICABLE` states. Freeze the authoritative source and reconstruction rule for every artifact before that artifact enters the denominator. An unresolved artifact may remain in the inventory, but must not be counted as a negative or positive observation.

### Technical and interpretive clarifications

#### m1 — The symbol and name `c` are directionally confusing (`MINOR`)

`c=K_eff/K` approaches 1 as weights become uniform and becomes smaller as they concentrate (`v1 plan:50-57`). Calling it “归一化集中度” invites readers to assume that larger means more concentrated. It also reuses `c_i(t)`, which the RQ007 contract uses for the stored concentration/identifiability index (`RQ007 contract:57-60`). Rename it `normalized_effective_candidate_fraction`/`u_eff`, or define a monotone concentration score such as `1-K_eff/K`, and keep the direction explicit in every figure.

#### m2 — The acceptance ban is literal rather than output-scoped (`MINOR`)

The plan says the entire text must contain no estimability wording (`v1 plan:27,183`) while necessarily using the term to explain the historical defect and the RQ007 boundary (`v1 plan:5-6,16-18,27-29,73-75,187-192`). Change the acceptance condition to “no RQ015A metric, state, figure, or conclusion is named or interpreted as estimability or measurement success.” Historical explanation and explicit prohibitions should remain allowed.

#### m3 — The RQ007-C2 “tension” is not defined (`MINOR`)

The plan sends `P(IPV=0 | c>=0.93)` back to RQ007 as a tension with C2 (`v1 plan:198`). RQ007 C2 rejects equivalence between high index and IPV zero; it does not assert statistical independence (`RQ007 decision:15`). A high conditional probability therefore need not contradict C2. State the exact hypothesis to be checked and preserve the distinction between exact zero, warm-up sentinel zero, numeric underflow/rounding, and intervals that contain zero.

### What this audit can and cannot establish

| Claim layer | Supportable under a closed v1 contract? | Boundary |
|---|---:|---|
| Whether an estimator invocation was attempted for a row/anchor | **Yes, conditionally** | Only when estimator-local provenance is recoverable; otherwise `UNKNOWN`. |
| The stored candidate-weight concentration or normalized effective-candidate fraction | **Yes** | Artifact/configuration/K specific; continuous value is primary. |
| Membership in operational concentration bins | **Yes, descriptively** | Only after a frozen, transparent rule and sensitivity analysis; not a truth label. |
| Frame/anchor/case exposure to attempt, concentration, and unknown states | **Yes** | Requires canonical units, deduplication, denominators, case-aware uncertainty, and source stratification. |
| Sensitivity of episode IPV summaries to alternative inclusion/weighting rules | **Yes** | Does not identify the correct or most accurate summary. |
| Whether IPV was successfully measured or is accurate | **No** | No latent truth or calibrated recovery benchmark. |
| RQ007 estimability | **No** | The full opportunity + sustained concentration + controls + case-health conjunction is out of scope. |
| Cause of near-uniform weights | **No** | D1/D2/D3 separation is deferred to RQ015B. |
| Downstream damage, validity, safety, preference, or low qualification risk | **No** | Requires owning-RQ, task-specific reanalysis. |
| A causal mechanism or that proximity is/is not the main driver | **No** | Exploratory associations can generate hypotheses only. |

### Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| **Originality** | **Moderate as a reusable audit framework; low-to-moderate as a standalone discovery.** The important novelty is harmonized provenance, recoverability, unit-aware concentration diagnostics, and downstream routing. The underlying concentration metric and summary-sensitivity insight are inherited. |
| **Scientific importance / significance** | **High internal importance and plausible methodological relevance.** It can prevent numerical completion or diffuse candidate support from being mistaken for valid measurement. Broader significance requires continuous/source-stratified results and robust sensitivity, not just three category prevalences. |
| **Interdisciplinary readership** | **Potentially meaningful.** The construct separation generalizes to inverse problems, model-based inference, runtime monitoring, and measurement governance. |
| **Technical soundness** | **Not yet established for the central binned and qualification outputs.** The construct correction and arithmetic are strong, but threshold selection, guard validation, risk routing, episode formulas, and cross-artifact unknown states remain incomplete. |
| **Readability for nonspecialists** | **Substantially improved, with residual terminology risk.** The mechanism boundary is now explicit, but the inverted `c` direction, “low qualification risk,” and unconditional observability language could still mislead. |

### Recommendation posture

The bounded audit should proceed after the two blockers and four major concerns are closed on a new checksum-frozen revision. The construct-narrowing decision itself should be retained. The scientifically defensible product is a **provenance-aware, continuous candidate-weight concentration audit with operational-bin sensitivity and downstream reanalysis routing**. It is not a validation study, not a repair study, and not a downstream-validity verdict.

## Minimum closure conditions for re-review

1. Freeze a scientifically explicit threshold SAP: objective, derivation population, unit/weighting, dev selection, guard validation, cross-source/K policy, uncertainty, sensitivity grid, and fail/withhold rule.
2. Replace `低资格风险` with bounded audit-routing language and add a deterministic indeterminate/unknown state.
3. Rewrite the proximity sentence as an open, descriptive association question reconciled with RQ007's different estimand.
4. Freeze every episode-summary formula, denominator, support rule, missingness rule, and interpretation as definition sensitivity only.
5. Preserve the sealed event as a disclosed PI governance waiver, not scientific proof of untouched independence.
6. Add `UNKNOWN` to attempt provenance, qualify “directly observable,” and freeze each artifact's authoritative reconstruction rule before inclusion.
7. Rename the normalized effective-candidate fraction or otherwise remove the directional/symbol collision.

## Risk / unsupported claims

- Unsupported: development-plus-guard data can “rederive” scientifically meaningful cutoffs without a declared objective and selection rule.
- Unsupported: a cutoff tuned using guard is independently validated by guard.
- Unsupported: one global normalized cutoff is automatically comparable across K, estimator configurations, sources, and sampling rates.
- Unsupported: low concentration-bin exposure establishes low downstream qualification risk.
- Unsupported: marginal `<5 m` frame share shows proximity is not a main explanatory factor.
- Unsupported: hard filtering or concentration weighting improves episode IPV rather than merely changes its definition.
- Unsupported: the PI waiver proves that the prior aggregate sealed exposure had exactly no scientific information effect.
- Not assessable: IPV accuracy, latent preference truth, RQ007 estimability, D1/D2/D3 cause, downstream performance/validity, or a preferred repair.

## Verdict

**`BLOCKED`**

- Blockers: **2**
- Major concerns: **4**
- Minor clarifications: **3**
- `formal_g1_eligible=false`
- `execution_authorized=false`

The v1 construct correction is accepted and should not be reopened. Formal G1 remains blocked because the two provisional thresholds and the downstream qualification verdicts still lack a scientifically bounded, reproducible decision contract.
