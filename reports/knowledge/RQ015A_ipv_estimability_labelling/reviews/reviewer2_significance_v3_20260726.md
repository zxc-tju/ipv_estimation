# RQ015A v3 Independent Review — Reviewer 2 (Originality, Scientific Significance, and Claim Boundaries)

## Review setup

- **Input scope:** frozen plan `reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`.
- **Verified plan SHA-256:** `75912bc1433a5efb5b0520af492e27579e9a1f6652074d3f37eb3a77befff264`.
- **Integrity check:** `reports/plans/RQ015A_plan_v3_checksums_20260726.sha256` verified **6/6 OK** before review.
- **Assessment boundary:** one independent Reviewer 2 report, as explicitly requested, rather than the skill's default multi-review package. I used only the frozen v3 bundle, the permitted v2 synthesis, the PI sealed-exposure disclosure, and accepted/binding RQ007 context. I did not read or communicate with Reviewer 1 or Reviewer 3 about any v3 output, and did not read any v3 synthesis.
- **Review emphasis:** originality, scientific significance, and claim boundaries, with targeted checks of the continuous-primary `q_eff` construct; the claimed decoupling of policy bins from downstream outputs; the standing PI rederivation condition; C0 policy-action semantics; factor-analysis scope; and cross-configuration / cross-artifact comparability.
- **Shared claim presented by v3:** RQ015A can now freeze a real, executable audit package that reports attempt provenance and continuous `q_eff` distributions across products without claiming IPV accuracy or full RQ007 estimability; report bins are descriptive only, while episode summaries and C0 routing consume only continuous quantities (`v3 plan:7-18,20-35,51-79,80-111`).
- **Visible evidence base:** the frozen v3 plan; the bound schema, run spec, contracts implementation, and fixtures (`manifest:1-6`; `ledger schema:1-169`; `run spec:1-78`; `contracts:1-328`; `tests:1-244`); the standing sealed-exposure disclosure and RQ007 README governance addendum (`sealed disclosure:62-108`; `RQ007 README:3-12,40-50`); the still-frozen RQ007 concept and estimability contract (`RQ007 binding contract:53-81`; `RQ007 decision:3,14-30`); and the v2 consensus on what needed closure (`v2 synthesis:34-105,239-260`).
- **Missing materials affecting confidence:** no append-only PI / RQ007 authority record superseding the mandatory dev+guard rederivation condition; no checksum-bound split-assignment artifact in the v3 bundle; no executable factor-analysis contract despite keeping that output in scope; and no explicit `policy-sensitive / withheld` terminal when C0 sensitivity flips.

## Reviewer 2

### Overall assessment

RQ015A v3 is scientifically better than v2 in the way that matters most. The central methodological correction is now real rather than merely promised: the primary result is the continuous `q_eff` distribution, and the executable contract no longer feeds the report bins into episode summaries or C0 routing (`v3 plan:20-35`; `contracts:11-12,219-313`; `tests:136-201`). This is the right bounded contribution. It turns RQ015A into a provenance-aware measurement audit rather than a disguised threshold-discovery exercise.

That improvement is substantial. It preserves the accepted construct boundary: `q_eff` remains a concentration/effective-support measure, not accuracy, not “measured IPV”, and not the full RQ007 estimability conjunction (`v3 plan:82-85,101-105`; `RQ007 binding contract:53-81`; `RQ007 decision:14-25`). The new schema, run-spec, and tested helper module also materially improve reproducibility and make the work more legible to downstream consumers (`v3 plan:10-18,37-79`; `ledger schema:40-169`; `run spec:38-78`).

Formal G1 is still not established. The decisive remaining blocker is authority-chain, not scientific taste: the checksum-bound disclosure and the RQ007 README both still state that the historical `4/7` and `0.93` values carry a **mandatory dev+guard-only rederivation condition** (`sealed disclosure:71-84`; `RQ007 README:6-12`). The v3 plan declares that condition “formally lifted” because bins are now descriptive only (`v3 plan:31-34,96-99`), but no append-only PI or RQ007 governance artifact in the frozen bundle actually performs that supersession. Reviewers cannot infer revocation from a conflicting plan sentence.

The second remaining problem is that the continuous-primary scientific object is still not fully closed at the comparability layer. The schema correctly forbids cross-artifact pooling (`ledger schema:156-160`), but the executable aggregation code groups L1 rows only by `(case_id, perspective, configuration)` and then collapses to case level, ignoring `artifact_id` and `measurement_role` (`contracts:165-213`). The run spec also references `RQ007 case_split_assignment.csv` only by symbolic name, without binding its exact path/hash into the v3 checksum bundle (`run spec:26-35`). Those gaps mean that the nominally primary `q_eff` distributions are still vulnerable to unresolved split provenance and accidental cross-artifact / cross-configuration aggregation.

### Who would be interested in the results, and why

- **Autonomous-driving interaction and inverse-planning researchers** would care because the audit separates whether a legacy estimator produced a usable attempt trail from how concentrated its candidate support was.
- **Measurement-science and uncertainty-governance readers** would care because the study treats unknown, unavailable, and concentrated-support states as distinct outcomes rather than collapsing them into zero or success.
- **Runtime-monitoring and validation researchers** would care because downstream qualification can change when the measurement-quality selection rule changes, even if the underlying estimator is untouched.
- **Reproducibility / research-governance readers** may value the attempt to freeze schema, provenance, exposure boundaries, and audit actions as first-class artifacts.

Interest outside the immediate IPV programme is still conditional. The transferable contribution is the audit logic, not the specific thresholds. If the final report presents `0.05 / 0.20 / 0.80` as programme-local triage policy and foregrounds continuous distributions, lineage, and sensitivity, the paper has methodological value. If it instead foregrounds named categories or route labels as if they were natural scientific states, the contribution becomes much narrower.

### Major strengths

1. **The primary scientific correction is now executable.** Episode summaries use only all-attempted and `1-q_eff` continuous weighting, and C0 routing consumes only `unknown_share`, `unavailable_share`, and `mean_q_eff_attempted` (`contracts:219-313`; `tests:136-201`).
2. **Construct narrowing remains intact.** v3 still treats RQ015A as an attempt/concentration audit only, leaving accuracy, mechanism, and full estimability outside scope (`v3 plan:82-85,101-105`).
3. **The closure package is more real.** Schema, contracts code, tests, and run spec exist and are checksum-bound instead of being future promises (`v3 plan:10-18,70-79`; `manifest:1-6`).
4. **Downstream wording is safer.** C0 is framed as administrative audit routing rather than as a validity, safety, or damage verdict (`v3 plan:25-35,66-69`).
5. **The held-out record is more precise.** The disclosure and README now consistently state that held-out single-row values were not shown/exported/manually inspected, while programmatic parsing and aggregation did occur (`sealed disclosure:89-108`; `RQ007 README:6-12,40-50`).

### Major concerns

#### B1 — The mandatory PI rederivation condition is still active in the frozen governance record (`BLOCKER`)

**Evidence.** The checksum-bound disclosure states that the two concentration cutoffs must be rederived from development plus guard data, that `4/7` and `0.93` are provisional, and that no conclusion portrait may use them before rederivation (`sealed disclosure:71-84`). The RQ007 README governance addendum repeats that any future held-out confirmation proceeds under this waiver **with a mandatory dev/guard-only threshold rederivation condition** (`RQ007 README:6-12`). The v3 plan instead says that the condition is “formally lifted” because the values are now only report bins (`v3 plan:31-34,96-99`). No append-only superseding PI/RQ007 artifact is bound in the v3 manifest.

**Why this matters.** v3's scientific direction is acceptable; the authority chain is not yet. The plan cannot silently rewrite a standing checksum-bound governance condition that survives in both the disclosure and the authoritative RQ007 README.

**Required closure.** Before Formal G1, add one checksum-bound append-only authority artifact that does one of two things:

1. explicitly supersedes the rederivation condition and defines where the historical values may still appear, with what policy-sensitive caveats; or
2. retains the condition and removes all v3 language implying it has already been lifted.

Until then, `formal_g1_eligible=false`.

#### B2 — The primary continuous audit still lacks a fully frozen comparability boundary (`BLOCKER`)

**Evidence.** The plan claims cross-artifact pooling is forbidden and that sigma01 is the only corpus-level source (`v3 plan:89-92`; `ledger schema:156-160`). But the only executable aggregation surface groups rows by `(case_id, perspective, configuration)` and then by `case_id`, ignoring `artifact_id` and `measurement_role` (`contracts:165-213`). A caller that passes mixed rows from duplicated/intermediate products can therefore produce a pooled L2/L3 summary without violating any code-level guard. The run spec also names the split source generically as `RQ007 case_split_assignment.csv`, while the v3 checksum bundle does not bind the exact split-assignment artifact that determines the dev/guard population (`run spec:26-35`; `manifest:1-6`).

**Why this matters.** This is not just an execution nicety. The primary scientific claim is a lineage-aware, unit-aware audit. If the split-defining artifact is not immutable and the executable aggregation path can collapse across artifacts or heterogeneous configurations, then the “continuous-primary” result is not yet a uniquely defined scientific object.

**Required closure.** Either:

1. checksum-bind the exact split-assignment artifact and restrict the executable aggregation contract to one artifact namespace at a time; or
2. extend the code-level grouping contract so `artifact_id`, `measurement_role`, and any K-defining configuration remain explicit through every reported aggregation layer.

#### M1 — Bins are decoupled from downstream logic, but C0 still lacks an explicit policy-sensitive output state (`MAJOR`)

**Evidence.** v3 correctly removes bins from episode summaries and C0 (`contracts:219-313`; `tests:183-201`). However, `c0_route_with_sensitivity` still emits a single primary terminal plus a `stable` boolean (`contracts:309-313`). When admissible triage cuts disagree, the output remains a determinate owner action with an instability flag rather than a policy-sensitive or withheld terminal. The plan likewise promises a `stable` marker but does not state how an unstable route is to be narrated in the final report (`v3 plan:27-35`).

**Required closure.** Add explicit report language or a contract-level terminal such as `ROUTE_POLICY_SENSITIVE` / `ROUTE_WITHHELD_POLICY_SENSITIVE` whenever sensitivity flips the owner action. A lone boolean is too easy to treat as metadata while retaining a headline route.

#### M2 — The factor-analysis claim remains descriptive in wording but not frozen in substance (`MAJOR`)

**Evidence.** v3 narrows factor analysis to descriptive Spearman correlations plus case-cluster bootstrap CIs (`v3 plan:63-65`), which is a correct downgrade. But the plan still does not freeze the factor inventory, unit of analysis, stratification rules, missingness handling, or multiplicity posture, and the bound implementation file contains no factor-analysis function at all (`contracts:1-328`; `tests:1-244`; `run spec:44-50`).

**Required closure.** Either remove factor analysis from the Formal-G1 execution package, or freeze a minimal exploratory SAP and executable implementation: exact factors, response, unit, strata, missingness rule, cluster bootstrap procedure, and noncausal interpretation template.

#### M3 — Cross-configuration comparability is still only partially defended (`MAJOR`)

**Evidence.** v3 is correct to report actual K and forbid silent substitution of `K=7` (`v3 plan:82-85`). But the normalization `q_eff = K_eff / K` does not by itself make different candidate sets, estimator configurations, or products scientifically interchangeable. The code-level L3 aggregation averages across all `OK` L2 units in a case (`contracts:198-213`), which can collapse heterogeneous configurations into one case-level mean even when their K or role semantics differ.

**Required closure.** The final report must either keep configuration-specific summaries separate, or explicitly justify and freeze any cross-configuration case aggregation rule. The current package is good enough to say “normalized scale exists”; it is not yet good enough to imply full comparability.

#### m1 — The quantity is still named like concentration even though larger values mean more diffuse support (`MINOR`)

The plan and schema still risk reader confusion by describing `q_eff` under concentration language while larger values indicate more near-uniform candidate support (`v3 plan:82-85`; `ledger schema:23-27`). The final report should either rename it `effective_candidate_fraction` or repeatedly state the direction in figures and captions.

#### m2 — The execution environment and the fixture story are not stated in the same contract vocabulary (`MINOR`)

The run spec declares a pure-stdlib environment (`run spec:18-24`), yet the bound fixture file is a `pytest` suite (`tests:1-15`). That does not invalidate the scientific case, but the execution package should state clearly whether `pytest` is a review-time requirement or whether the “16/16 pass” evidence is historical rather than re-runnable in the minimal environment.

### Technical failings that need to be addressed before the case is established

| Failing | Consequence | Minimum closure evidence |
|---|---|---|
| Standing PI rederivation condition remains active | The v3 plan conflicts with the checksum-bound governance record | Append-only PI/RQ007 supersession or explicit withdrawal of the “formally lifted” claim |
| Split provenance and aggregation scope are not fully bound | The primary `q_eff` result can vary with unresolved split or mixed-artifact aggregation choices | Checksum-bound split assignment plus artifact-scoped aggregation contract |
| C0 sensitivity has no explicit policy-sensitive terminal | Administrative route can be read as determinate despite admitted threshold dependence | Contracted unstable-route output state and report wording |
| Factor analysis is in scope but not executable or fully frozen | Exploratory associations can be narrated without a unique scientific question | Remove from Formal G1 or freeze + implement a minimal SAP |
| Cross-configuration case aggregation is not fully justified | “Normalized” summaries may imply comparability beyond what the package proves | Frozen per-configuration reporting rule or justified aggregation rule |

### Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| **Originality** | **Moderate and now clearer.** The novelty is not a new statistical quantity but the conversion of a fragile estimator by-product into a provenance-aware audit framework with explicit unknown states and lineage. |
| **Scientific importance / significance** | **High internal importance; moderate broader methodological importance.** The work can materially improve downstream validity discipline inside the programme. Broader importance depends on presenting a transferable measurement-governance lesson rather than project-specific policy labels. |
| **Interdisciplinary readership** | **Plausible but bounded.** Inverse problems, measurement science, reproducibility, and autonomous-systems validation share the underlying issue. The narrative must stay on evidence provenance and policy-sensitive selection, not on IPV-specific jargon. |
| **Technical soundness** | **Much improved but not yet fully established.** The continuous-primary redesign and bin decoupling are sound; the authority-chain supersession and comparability/aggregation contract remain incomplete. |
| **Readability for nonspecialists** | **Improved.** The plan is more understandable than v2 because it distinguishes primary continuous results from descriptive bins. Remaining readability risk comes from concentration-style naming and from administrative routes that may still look more determinate than they are. |

### Recommendation posture

The core v3 redesign should be retained. It closes the most important scientific error from v2 by making the bounded construct executable and by removing report bins from downstream mechanics. I would support progression after the two blockers are closed and the three major issues above are frozen on a new checksum-bound package.

The scientifically defensible headline is now narrower and cleaner:

> RQ015A audits where legacy IPV estimators were attempted and how diffuse their candidate-weight support was, while preserving unknown states and exposing when downstream audit routing depends on administrative policy rather than scientific validation.

This review does not make an editorial decision or claim Nature-level fit. It addresses whether the supplied v3 package establishes its own bounded scientific case.

## Risk / unsupported claims

- Unsupported: that the standing mandatory dev+guard rederivation condition has already been formally lifted by the v3 plan itself.
- Unsupported: that historical `4/7` and `0.93` values are now governance-compliant merely because they are called report bins.
- Unsupported: that the primary `q_eff` summaries are fully immutable while the split-defining artifact is not checksum-bound in the v3 bundle.
- Unsupported: that the current executable aggregation path mechanically prevents cross-artifact or cross-configuration pooling.
- Unsupported: that a primary C0 route remains scientifically or administratively determinate when admitted sensitivity sets disagree and only a `stable=false` flag is emitted.
- Unsupported: that `0.05 / 0.20 / 0.80` are scientific qualification thresholds rather than programme-local triage policy.
- Unsupported: that factor associations, if reported under the current package, would have a uniquely defined estimand or mechanism interpretation.
- Not assessable from RQ015A: IPV truth/accuracy, causes of near-uniform support, the full RQ007 estimability conjunction, downstream safety/performance damage, or a preferred estimator repair.

## Verdict

**`BLOCKED`**

- Blockers: **2**
- Major concerns: **3**
- Minor concerns: **2**
- `formal_g1_eligible=false`
- `execution_authorized=false`

v3 successfully fixes the central scientific design error from v2: continuous `q_eff` is primary and report bins are no longer consumed downstream. Formal G1 remains blocked because the frozen governance record still carries a mandatory rederivation condition that v3 has not validly superseded, and because the supposedly primary continuous audit is not yet fully closed against unresolved split provenance and aggregation scope.
