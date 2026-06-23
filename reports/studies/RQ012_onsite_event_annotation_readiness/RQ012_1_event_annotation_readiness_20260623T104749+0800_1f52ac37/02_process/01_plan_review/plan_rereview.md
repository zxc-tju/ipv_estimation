# RQ012A Plan Re-Review

Worker: RQ012-W03-plan-rereview
Role: independent plan re-review
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Prior review: plan_review.md
Resolution addendum: plan_resolution_v0_1.md
Reviewed SPEC: reports/plans/RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md

## SPEC Hash Check

Expected SPEC SHA256:

```text
921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e
```

Observed SPEC SHA256:

```text
921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e
```

Result: PASS. The original SPEC is unedited relative to the required hash.

## Per-Finding Re-Review

### B01 - Construct-Proximal Motif Labels May Become Circular Consequence Endpoints

Verdict: CLOSED

The addendum's B01 section introduces the mandatory `endpoint_eligibility` field for every event or label and defines four distinct tiers: `independent_consequence_endpoint`, `construct_proximal_descriptor`, `planner_system_event`, and `annotation_quality_label`. This directly separates human-observable consequence endpoints from construct-proximal descriptors, planner/system events, and annotation-quality labels.

The closure is substantive, not nominal. The B01 binding rule explicitly names aggressive intrusion, over-yielding/freeze, and appropriate assertiveness as examples of `construct_proximal_descriptor`, forbids that tier as a primary validation endpoint, and states that any later event-IPV association built on a construct-proximal descriptor is circular and inadmissible as primary evidence. The binding constraints reiterate that only `independent_consequence_endpoint` items may be primary event-IPV validation endpoints in later RQ012B and that construct-proximal descriptors are forbidden as primary endpoints. This removes the circularity pathway identified in the prior review rather than merely renaming the labels.

### B02 - Development-Subset Threshold Source Is Under-Specified

Verdict: CLOSED

The addendum's B02 section establishes a real dual-track threshold design. The confirmatory track requires any threshold used by the extractor, pilot, blind package, readiness report, or later RQ012B analysis to be derived from a pre-frozen, outcome-blind development subset with documented sample frame, random or scenario-stratified selection method, recorded seed/hash, no-overlap rule, prohibited outcome-correlated proxies, authorized builder role, and a frozen manifest produced before viewing more than outcome-blind data-health summaries.

The exploratory track is allowed rather than banned. It permits intuition, candidate thresholds, sensitivity ideas, and failure-mode discovery, but requires explicit leakage-risk annotation covering subset choice, potential outcome-correlated proxies, and leakage-risk rating. It also requires quarantine labelled `exploratory - not leakage-protected` and forbids using exploratory outputs as binding thresholds.

The firewall is sufficient: exploratory-derived thresholds cannot silently become confirmatory thresholds. They must be re-derived or confirmed on the frozen outcome-blind confirmatory subset, pass confirmatory provenance rules, and be recorded with exploratory source, confirmatory subset manifest, decision rationale, sensitivity check, and approval record. The addendum also forbids choosing thresholds by later IPV association strength, sign, significance, or narrative usefulness. This closes the leakage risk while preserving controlled exploration.

### B03 - Gate 012B Is Not Explicitly Dependent On All Wave-A Gates

Verdict: CLOSED

The addendum's B03 section makes Gate 012B non-bypassable. It requires documented PASS for Gates 012-0, 012-1, 012-2, and 012-3 before event-IPV analysis can begin. It also requires frozen ontology including `endpoint_eligibility`, frozen threshold rationale including confirmatory-track provenance and promotion records, frozen blind-package audit, frozen agreement-analysis protocol saved before any agreement results are viewed, two real independent human annotation files, agreement computed under the frozen protocol, cited RQ007/RQ009/RQ011 freezes, and explicit recorded authorization.

The original literal-only Gate 012B path is explicitly declared invalid: any path satisfying only the original literal Gate 012B list is not enough, and a worker must not start event-IPV analysis if any listed item is missing, provisional, failed, undocumented, or assumed. This directly closes the bypass risk identified in the prior review.

## Advisory Re-Review

### A01 - Additional Consequence Classes

Verdict: FOLDED_IN

The addendum's A01 section requires W0/W1 to audit or explicitly document as unavailable/non-goals: collision/contact, off-route or no-progress timeout, human/operator takeover, emergency stop, abrupt lateral comfort events, and yielding to a non-counterpart actor. It also requires required signals, availability status, actor identity rule, outcome-blind extractability, and `endpoint_eligibility` tier for each candidate.

### A02 - Global Indirect Leakage Proxies

Verdict: FOLDED_IN

The addendum's A02 section promotes indirect leakage proxies to a global denylist across event-definition, threshold-development, extraction-pilot, blind-package, annotation-preparation, merge-validation, and pre-analysis work. It names area IDs, scenario IDs, run IDs, filenames, paths, item ordering, thumbnails, manifest-derived strata, and prior or borrowed annotation files, unless specifically approved in writing as outcome-blind design variables for a particular use.

### A03 - Reproducibility-Auditable Blind Issuance

Verdict: FOLDED_IN

The addendum's A03 section requires a blind-package issuance manifest before annotator release, including neutral item IDs, source-to-neutral mapping custody location, per-file checksums, randomized order seed, package version, evidence of embedded-metadata stripping, and auditor sign-off. It also requires the manifest to be frozen before formal annotation files are accepted.

### A04 - Simulated-Label Rejection Provenance

Verdict: FOLDED_IN

The addendum's A04 section adds provenance controls beyond content checks: annotator attestation, file-provenance checks, controlled submission channel, and metadata/audit logs for receipt, custody, and merge. It explicitly forbids accepting a label file as real human annotation solely because CSV content is non-empty, non-duplicate, and schema-valid.

## Internal Consistency Checks

- `plan_review_findings.csv` marks B01, B02, B03, and A01-A04 as `resolved_by_addendum`, with notes pointing to the corresponding addendum sections. This is consistent with the re-review findings above.
- The addendum states it supplements and does not replace or edit the SPEC, and the SPEC hash check confirms the original SPEC was not edited.
- There is no substantive contradiction with the SPEC. The addendum is more specific where the SPEC was under-specified, and it states downstream workers must follow the addendum where it is more specific.
- Wave-A scope is preserved. The addendum repeats that Wave A is readiness work only, with no event-IPV association, no causal claims, no fabricated labels or agreement, and no use of official coordination to train or tune event detection.
- The behavioural reference is constrained to outcome-blind behavioural evidence, not fully independent truth. This resolves the prior wording concern without expanding the Wave-A claim.

## Final Verdict

RE-REVIEW VERDICT: CLEARED
