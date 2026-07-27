# RQ015A v2 Independent Review — Reviewer 2 (Originality, Scientific Significance, and Claim Boundaries)

## Review setup

- **Input scope:** frozen plan `reports/plans/RQ015A_plan_v2_concentration_audit_20260726.md`.
- **Verified plan SHA-256:** `9186c95eb6d84ee56626f6e96cb75d2f0422297446824ea20848e34258ab9a67`.
- **Integrity check:** `reports/plans/RQ015A_plan_v2_checksums_20260726.sha256` verified **6/6 OK** before review.
- **Assessment boundary:** one independent Reviewer 2 report, as explicitly requested, rather than the skill's default multi-review package. I used only the frozen v2 plan, the permitted v1 synthesis, the PI sealed-exposure disclosure, and accepted/binding RQ007 context. I did not read or communicate with Reviewer 1 or Reviewer 3 about any v2 output, and did not read a v2 synthesis.
- **Review emphasis:** originality, scientific significance, and claim boundaries, with targeted checks of the continuous-primary `q_eff` construct; the policy bins and C0 thresholds; sensitivity/withheld consistency; the standing PI dev+guard rederivation condition; proximity/factor analysis; episode summaries; and downstream qualification language.
- **Shared claim presented by v2:** a retrospective, cross-artifact audit can report estimation-attempt provenance and the continuous normalized effective-candidate fraction `q_eff` without claiming IPV accuracy or RQ007 estimability; operational bins are secondary, and downstream RQs receive deterministic audit-routing actions rather than validity verdicts (`v2 plan:11-48,77-114,116-145,180-190`).
- **Visible evidence base:** the v1 consensus that construct narrowing is closed and continuous `q_eff` should be primary (`v1 synthesis:30-66,142-155`); the standing PI disclosure and mandatory rederivation condition (`sealed disclosure:62-84,89-108`); and RQ007's binding separation of concentration, current IPV, opportunity, estimability, and behavioural dynamics (`RQ007 contract:53-81`; `RQ007 decision:14-30`).
- **Missing materials affecting confidence:** the frozen bundle contains no append-only PI decision superseding the mandatory dev+guard rederivation condition; no route-level sensitivity/withheld rule for C0; no complete exploratory-factor SAP; and no fully specified formula/eligibility contract for the two episode summaries.

## Reviewer 2

### Overall assessment

RQ015A v2 is a substantial scientific improvement over v1. Making the **continuous distribution of `q_eff = K_eff/K` the primary result** is the correct response to the absence of a naturally identified semantic cutoff. It converts the study from an attempted classifier without truth labels into a defensible descriptive measurement audit. The construct boundary is also maintained: v2 does not call `q_eff` measurement success or RQ007 estimability, and explicitly preserves accuracy, cause, and full-estimability exclusions (`v2 plan:11-39,180-185`).

This gives RQ015A a coherent contribution. Its originality is a provenance-aware, unit-aware audit framework spanning products with different roles, missingness, warm-up semantics, and lineage—not a new estimator or a newly validated confidence construct. Its scientific importance is high within the programme because it can prevent silent reuse of numerically weak or unrecoverable measurement channels. Broader value is plausible for inverse-planning and model-based measurement pipelines if the final report foregrounds continuous distributions, unknown states, lineage, and selection consequences rather than project-specific policy labels.

The plan is nevertheless not ready for Formal G1. The historical cutoffs have not actually left the decision chain: `q_lo` controls one episode summary and `q_hi` controls the C0 owner-routing verdict (`v2 plan:41-48,111-113,116-130`). More importantly, v2 declares that dev+guard rederivation is no longer needed, while the checksum-bound disclosure still calls that rederivation a mandatory, unconditionally executed PI condition (`v2 plan:16-25`; `sealed disclosure:71-84`). No superseding PI ruling is included. A plan-level relabelling from “scientific threshold” to “policy threshold” does not itself cancel an active governance decision, especially while the same historically exposed values still determine actions.

The second central problem is internal inconsistency between bins, sensitivity, withholding, and C0. The plan says bins cannot be used for downstream determination, yet C0 thresholds the continuous measure at `q_hi=0.93`; it withholds bin summaries when prevalence varies by more than 10 percentage points, but does not withhold or qualify a C0 route that can change at a 5% exposure boundary under a much smaller sensitivity shift. Thus the continuous-primary scientific result is viable, but the policy-action layer remains neither authority-complete nor sensitivity-safe.

### Who would be interested in the results, and why

- **Autonomous-driving interaction and inverse-planning researchers** would care because the audit separates whether an estimator ran from how concentrated its candidate support was.
- **Measurement-science and uncertainty-quantification readers** would care because `q_eff` is kept distinct from fit quality, accuracy, confidence intervals, and latent-truth recovery.
- **Runtime-monitoring and safety-verification researchers** would care because selection on a measurement-quality proxy can alter the population to which a downstream monitor or envelope applies.
- **Reproducibility and research-governance readers** may value the product lineage, canonical units, explicit unknown states, deduplication, row conservation, and owner-routing design.

Interest outside the IPV programme remains conditional. The audit would have transferable methodological value if it demonstrates how conclusions change across measurement units, provenance states, K/configuration strata, and policy choices. A fixed `4/7`, `0.93`, `5%`, and `20%` administrative scheme without route-stability evidence would remain programme-specific governance rather than broad scientific insight.

### Major strengths

1. **Continuous `q_eff` is now primary.** The plan correctly acknowledges that a distribution does not reveal two unique semantic boundaries and demotes bins to secondary policy summaries (`v2 plan:11-39`). This closes the core scientific-threshold misconception identified in v1 (`v1 synthesis:84-88,142-150`).
2. **Construct narrowing remains intact.** Attempt provenance and candidate-weight concentration are the only intended constructs; accuracy, latent truth, causes of near-uniformity, and the full RQ007 estimability conjunction remain out of scope (`v2 plan:13-14,180-190`).
3. **Lineage and analysis units are materially stronger.** The plan prohibits cross-product pooling of duplicated observations, distinguishes L1/L2/L3 units, propagates unknown/support states, and requires case-clustered intervals (`v2 plan:65-114`). These changes protect the scientific denominator from artifact expansion and pseudoreplication.
4. **The policy-bin limitation is visible.** Fixed cutoffs are explicitly labelled policy choices, a nine-cell sensitivity table is required, and an unstable-bin withholding path exists (`v2 plan:41-48`). This is directionally correct even though it is not yet connected consistently to C0 and episode outputs.
5. **Downstream language is substantially safer.** `低资格风险` is removed; the route names state audit actions rather than damage or validity, and owning RQ decisions are not automatically changed (`v2 plan:116-130`).
6. **Episode claims are narrowed to definition sensitivity.** The plan explicitly prohibits interpreting either hard filtering or `1-q_eff` weighting as more accurate (`v2 plan:111-114`). This conforms to RQ007-KC-C3, which establishes definition dependence but not a preferred summary (`RQ007 decision:16,24-25`).
7. **The held-out record is more accurate.** v2 states that the scan programmatically parsed and aggregated held-out fields while no single-row values were displayed, exported, or manually inspected, and it prohibits pristine-untouched wording (`v2 plan:132-145`; `sealed disclosure:89-108`).

### Major concerns

#### B1 — The standing PI rederivation condition is abandoned without a superseding authority record (`BLOCKER`)

**Evidence.** The PI disclosure states that the two concentration cutoffs **must** be rederived from development plus guard data, labels this an unconditional attached condition, and says the provisional values cannot produce conclusions before rederivation (`sealed disclosure:71-84`). The v2 plan instead states that scientific rederivation is no longer required and reinstates the same `4/7` and `0.93` values as policy choices (`v2 plan:16-25,41-48`). The v2 checksum bundle includes the unchanged disclosure, and no append-only PI amendment or superseding decision is present.

The policy reframing does not make the condition moot. `q_lo` still determines the hard-filtered episode summary, and `q_hi` still enters C0 routing; therefore these historically exposed values continue to affect report outputs and owner actions (`v2 plan:111-113,116-130`). The disclosure itself records that the values were written after sealed-inclusive aggregate inspection (`sealed disclosure:24-29`).

**Why this matters.** This is an authority-chain defect, not a disagreement about whether descriptive policy bins are scientifically permissible. The plan author is identified as acting in a PI role, but the proposed plan is not an append-only amendment to the standing disclosure and does not explicitly supersede its mandatory condition. Reviewers cannot infer revocation from a conflicting draft sentence.

**Required closure.** Before Formal G1, choose and checksum-bind one of two paths:

1. **Comply with the standing PI condition:** freeze an authorized dev+guard rederivation rule and execute it only after the plan passes review; or
2. **Explicitly supersede it:** add an append-only PI decision stating that the mandatory rederivation condition is withdrawn because bins are now purely administrative, specifying whether the historically exposed values may be reused, where they may be used, and what sensitivity/withholding conditions replace rederivation.

The second path must also address why the policy values may control episode and C0 actions. Until one path is recorded, `formal_g1_eligible=false` regardless of the scientific merit of continuous `q_eff`.

#### B2 — Policy bins, C0 routing, and the withheld path are internally inconsistent (`BLOCKER`)

**Evidence.** Section 1.1 says bins may not be used for downstream determination and that unstable bin summaries will be withheld (`v2 plan:41-48`). C0 nevertheless defines exposure using `q_eff >= q_hi`, where `q_hi=0.93` is exactly the upper policy-bin boundary, and turns a composite exposure below or above `5%` into different owner actions; `unknown >=20%` creates a third threshold (`v2 plan:116-130`). The nine-cell sensitivity varies `q_hi`, but the C0 contract does not say whether the route is recomputed, what happens if it changes, or how the `5%` and `20%` action thresholds are stress-tested.

The bin-withholding rule also does not protect C0. It withholds only if a three-bin prevalence range exceeds 10 percentage points (`v2 plan:44-47`). A route can cross the C0 `5%` boundary while the sensitivity range remains below 10 points; the plan would then call the bins stable enough to publish while emitting a policy-dependent owner action. Similarly, `BINS_WITHHELD_UNSTABLE` has no stated cascade to the hard-threshold episode summary or C0 route.

**Why this matters.** Calling C0 “continuous” does not remove threshold dependence. Thresholding `q_eff` at `q_hi` is binning, and the additional `5%`/`20%` rules are unvalidated policy cutoffs. Administrative thresholds are permissible if explicitly authorized and reported as policy, but they cannot silently acquire scientific force or produce a seemingly deterministic route when plausible policy choices disagree.

**Required closure.** Freeze a route-level policy contract that:

1. states the authority and rationale for `q_hi`, `5%`, and `20%` separately;
2. reports the continuous exposure curve as a function of `q_hi`, not only a single thresholded percentage;
3. recomputes every C0 terminal state across the prespecified `q_hi` sensitivity grid and a justified sensitivity set for the `5%`/`20%` action thresholds;
4. emits `ROUTE_WITHHELD_POLICY_SENSITIVE` or `INDETERMINATE_POLICY_SENSITIVE` whenever the owner action changes across admissible policies;
5. makes `BINS_WITHHELD_UNSTABLE` cascade to every hard-threshold derivative, including the `q_lo` episode summary and any `q_hi`-based C0 route;
6. states explicitly that `NO_AUDIT_TRIGGER_DETECTED` is not evidence of validity, safety, accuracy, or low qualification risk, and that `OWNER_REANALYSIS_REQUIRED` is an administrative routing action rather than evidence of damage.

#### M1 — The continuous-primary variable is conceptually sound but its scientific contract is not fully closed (`MAJOR`)

**Evidence.** The plan defines `K_eff in (0,K]` and `q_eff in (0,1]` (`v2 plan:27-39`). For normalized nonnegative weights, the mathematical support is `K_eff in [1,K]` and `q_eff in [1/K,1]`; the lower support depends on K. The warm-up sentinel `ipv_error=1` would make the displayed inverse formula singular, so `attempt_status` must be resolved before `q_eff` is evaluated. The plan's product table records D0 rules, but the primary definition does not explicitly state this gating or define invalid/out-of-domain/rounding behavior.

The plan also promises distributions at L1, L2, and L3, but does not define what statistic moves `q_eff` from L1 to a case-perspective-configuration unit and then to a case (`v2 plan:35-39,95-114`). Mean, median, pooled measurements, equal-L2 weighting, and equal-case weighting answer different questions. Per-product reporting prevents duplicate pooling, but it does not by itself establish comparability across K, estimator configurations, or sampling regimes.

**Required closure.** Rename the quantity `normalized_effective_candidate_fraction` (or equivalent), state that larger values mean more diffuse/near-uniform support, define its exact K-dependent range, and compute it only for `ATTEMPTED` rows with valid `ipv_error` and K. Freeze tolerance/clipping/rejection rules. For L2/L3, specify the aggregation function, weighting order, and unit-specific denominators; stratify or explicitly qualify comparisons across K/configuration rather than treating normalization as measurement invariance.

#### M2 — The proximity/factor analysis is bounded in wording but lacks a reproducible scientific question (`MAJOR`)

**Evidence.** v2 removes the unsupported v1 assertion that proximity is not a main determinant and now lists only an exploratory “concentration-related factors” deliverable with no causal conclusion (`v2 plan:158-168`). This is an important correction. However, no predictors, primary response, unit, model/contrast, source handling, missingness rule, multiplicity policy, or uncertainty output is frozen. RQ007's accepted C1 concerns a specific interaction-opportunity contrast and concludes that most of that gap is proximity-compatible (`RQ007 decision:14,22,28-34`). A new whole-corpus association would be a different estimand.

**Required closure.** Either remove the factor analysis from Formal G1 or freeze a minimal exploratory SAP: continuous `q_eff` as the response; named covariates fixed before inspection; source/configuration stratification; case-aware uncertainty; missingness and multiplicity handling; and effect sizes rather than significance-only ranking. Any proximity result must be described as an association under the new estimand and must not be framed as confirming or refuting RQ007 C1, identifying mechanism, or separating fixable from intrinsic causes.

#### M3 — Episode-summary boundaries are improved, but the two definitions remain incompletely specified and policy-sensitive (`MAJOR`)

**Evidence.** The plan names two summaries—frames with `q_eff <= q_lo` and weights `1-q_eff`—and correctly limits interpretation to definition sensitivity (`v2 plan:111-114`). It does not state the complete normalized weighted-mean formula, whether only `ATTEMPTED`/known rows are eligible, how zero total weight is handled, how perspective/configuration units are preserved, what unfiltered attempted-frame reference is used, or how q-lo sensitivity affects the hard-filtered result.

The first summary also depends on the historically exposed `q_lo=4/7`, yet the sensitivity/withheld contract is defined only for bin proportions, not episode-summary conclusions (`v2 plan:41-48,111-114`).

**Required closure.** Freeze the exact numerator, denominator, eligibility, minimum support, zero-weight, unknown, perspective/configuration, and case aggregation rules. Include an all-attempted reference. Recompute the hard-filtered summary across the full `q_lo` sensitivity set and withhold any qualitative episode conclusion that changes materially; if bin summaries are withheld, the q-lo hard-filtered episode summary must also be withheld. Preserve the claim as definition sensitivity only—never improved accuracy, correction, or preferred episode IPV.

#### M4 — C0 is safer in name but is not yet a truly mutually exclusive, exhaustive, claim-bounded routing algorithm (`MAJOR`)

**Evidence.** The table gives four apparently necessary-and-sufficient states, then assigns the priority `INDETERMINATE > OWNER_REANALYSIS_REQUIRED > NO_AUDIT_TRIGGER_DETECTED > NOT_APPLICABLE` (`v2 plan:116-130`). Applicability should be evaluated first: if an owning RQ does not consume IPV/error-derived quantities, missing mapping or unknown provenance should not override `NOT_APPLICABLE`. Conversely, for a consuming RQ, an unavailable 1:1 mapping should force indeterminate before any exposure calculation.

The plan also refers to “analysis rows” without freezing whether the denominator is source rows, unique measurements, anchors, L2 units, or cases. The same unknown proportion can differ substantially by unit. Finally, the route names are safer than `低资格风险`, but the acceptance and report contract do not explicitly require the no-validity/no-safety interpretation to accompany every route.

**Required closure.** Freeze the evaluation order as: (1) applicability, (2) mapping/provenance sufficiency, (3) exact canonical denominator and mutually exclusive terminal measurement states, (4) administrative policy trigger, and (5) route-sensitivity/withholding. Require every route row to carry its analysis unit, numerator, denominator, policy version, sensitivity status, reason code, and the statement that the route does not revise or validate the owning RQ conclusion.

#### m1 — The acceptance ban remains literally impossible (`MINOR`)

The acceptance criterion requires no occurrence of estimability language (`v2 plan:170-178`) although the plan must use the term to explain the forbidden construct and known limitation (`v2 plan:11-14,180-185`). Scope the check to names and interpretations of RQ015A outputs: no metric, state, figure, or conclusion may be called or interpreted as estimability or successful measurement. Historical explanation and explicit prohibitions should remain permissible.

#### m2 — The fixed 50-bin histogram is a display choice, not scientific resolution (`MINOR`)

The plan fixes 50 equal-width bins on `[0,1]` (`v2 plan:35-36`). This is acceptable for reproducible display, but it should be labelled a visualization setting and paired with ECDF/quantiles; no modality, threshold, or mechanism claim should be inferred from the chosen bin width. The impossible interval below `1/K` should remain visibly empty rather than being interpreted as absent empirical support.

### Technical failings that need to be addressed before the case is established

| Failing | Consequence | Minimum closure evidence |
|---|---|---|
| Standing PI condition conflicts with v2 | Execution would violate the checksum-bound governance record | Append-only PI supersession, or a compliant rederivation contract, included in a new checksum bundle |
| `q_eff` support/gating and L2/L3 aggregation are incomplete | Primary continuous distributions can include invalid sentinels or answer different unit-level questions | Exact domain, attempt-first gating, invalid handling, aggregation/weighting formulas, and fixtures |
| Bin sensitivity does not propagate to episode/C0 | Secondary policy can change owner actions while still being called publishable | Route-level sensitivity table plus automatic `POLICY_SENSITIVE/WITHHELD` cascade |
| C0 uses ungrounded `q_hi/5%/20%` thresholds | An administrative choice can be mistaken for a scientific qualification result | Explicit policy authority/version, continuous exposure curve, threshold sensitivities, and bounded route language |
| Factor analysis lacks an SAP | Post-hoc associations can be narrated as mechanism | Fixed response/covariates/unit/models, case-aware uncertainty, missingness/multiplicity rules, and noncausal claim template |
| Episode summaries lack complete formulas | Two implementations can produce different summaries and support sets | Formula, reference summary, eligibility/support/zero-weight rules, q-lo sensitivity, and fixtures |
| C0 evaluation order and denominator are ambiguous | The four states are not reliably mutually exclusive/exhaustive across owning RQs | Applicability-first decision tree, canonical unit, precedence tests, and one fixture for each terminal state |

### Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| **Originality** | **Moderate as an audit framework; limited as a standalone discovery.** The novel value is the harmonized provenance/lineage and unit-aware measurement audit. `q_eff` is a transformation of an existing concentration index, and the policy bins are not discoveries. |
| **Scientific importance / significance** | **High internal importance, potentially broader methodological importance.** Preventing numerical/provenance states from being mistaken for validated measurements can materially improve research reliability. Broader importance requires policy-robust conclusions and a transferable audit logic rather than fixed programme-specific triggers. |
| **Interdisciplinary readership** | **Plausible but not yet demonstrated.** Inverse problems, measurement science, uncertainty quantification, autonomous systems, and runtime verification share the underlying failure mode. The final presentation must explain the evidence boundary without IPV-specific shorthand. |
| **Technical soundness** | **Continuous-primary direction is sound; the complete case is not yet established.** Authority conflict, invalid-domain/aggregation gaps, route instability, and incomplete factor/episode contracts still affect central outputs. |
| **Readability for nonspecialists** | **Substantially improved.** The primary/secondary distinction is clear. Remaining risk comes from calling `q_eff` “normalized concentration” when larger values indicate more diffuse support, and from presenting administrative thresholds as deterministic scientific qualification. |

### Recommendation posture

The continuous-primary redesign should be retained. It makes RQ015A a coherent and worthwhile measurement-governance study. I would support progression after the two blockers are removed and the four major specifications are frozen on a new checksum-bound object. The scientifically defensible headline is:

> RQ015A audits where legacy IPV estimators were attempted and how broadly their candidate weights were distributed, while preserving unknown states and routing policy-sensitive downstream reanalysis without judging IPV accuracy, estimability, or downstream validity.

This review does not make an editorial decision or claim that the work has demonstrated broad Nature-level importance; it assesses whether the supplied plan establishes its bounded scientific case.

## Risk / unsupported claims

- Unsupported: that continuous-primary reporting automatically makes the historically exposed `4/7` and `0.93` values governance-compliant.
- Unsupported: that the standing mandatory dev+guard rederivation condition can be withdrawn by implication rather than an append-only PI decision.
- Unsupported: that bins are not used downstream when `q_hi` directly controls C0 routing.
- Unsupported: that a bin summary stable within 10 percentage points guarantees a stable C0 action at a 5% boundary.
- Unsupported: that `q_hi=0.93`, `5%`, or `20%` is a scientifically validated qualification boundary.
- Unsupported: that `NO_AUDIT_TRIGGER_DETECTED` demonstrates low risk, accuracy, safety, or validity.
- Unsupported: that `q_eff` normalization alone creates comparable physical meaning across K, estimators, products, sources, or sampling rates.
- Unsupported: that exploratory proximity/factor associations establish mechanism or refute RQ007's proximity-bounded result.
- Unsupported: that q-lo filtering or `1-q_eff` weighting produces a more accurate episode IPV.
- Not assessable from RQ015A: IPV truth/accuracy, causes of near-uniform support, the full RQ007 estimability conjunction, downstream performance/damage, or a preferred estimator repair.

## Verdict

**`BLOCKED`**

- Blockers: **2**
- Major concerns: **4**
- Minor concerns: **2**
- `formal_g1_eligible=false`
- `execution_authorized=false`

The continuous `q_eff` primary result and the narrowed construct are scientifically acceptable. Formal G1 remains blocked because v2 conflicts with the active PI rederivation condition and because the secondary policy layer still drives episode/C0 actions without a coherent sensitivity-to-withholding contract.
