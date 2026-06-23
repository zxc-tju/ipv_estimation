# RQ012A Plan-Review Resolution Addendum v0.1

Worker: RQ012-W02-plan-resolution
Role: implementer, governance addendum authoring
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Supplements: reports/plans/RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md
Supplemented SPEC SHA256: 921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e
Git HEAD at review: 38063a2ff9cdc717098cf3f821c2bb162a0ac1d9
Status: binding addendum; does not replace or edit the SPEC.

This addendum is binding on all downstream RQ012A workers. Where this addendum is more specific than the original SPEC, workers must follow this addendum. The original Wave-A scope remains unchanged: no event-IPV association, no causal claims, no use of official coordination to train or tune event detection, no simulated/model substitute for human annotation, and no claim that the behavioural reference is fully independent truth. The reference should be described as outcome-blind behavioural evidence unless a later approved plan defines a stricter independence tier.

## B01 - Construct-Proximal Motif Labels May Become Circular Consequence Endpoints

Finding: Human motif labels such as aggressive intrusion and over-yielding/freezing could be used as circular consequence endpoints for competitive or over-yielding IPV deviations.

Binding resolution: W1 event ontology and W5 annotation codebook v2 must include a mandatory ontology/codebook field named `endpoint_eligibility` for every event or label. Allowed values are:

- `independent_consequence_endpoint`: Conceptually separable from the IPV construct, such as counterpart hard braking, near miss, deadlock, oscillation/stop-go, or safety intervention. Only this tier is eligible as a primary event-IPV validation endpoint in later RQ012B, subject to all gates and later authorization.
- `construct_proximal_descriptor`: Semantically restates competitive, assertive, or over-yielding behaviour, such as aggressive intrusion, over-yielding/freeze, or appropriate assertiveness. This tier is forbidden as a primary validation endpoint. It may be used only as secondary descriptive context, and any later use must be explicitly flagged as construct-proximal.
- `planner_system_event`: Planner fallback, replanning, trajectory rejection, mission failure, or comparable planner/system-status event. Use must be kept analytically distinct from human-observable consequence endpoints.
- `annotation_quality_label`: Insufficient evidence, unrelated failure, unusable item, or other label about annotation quality, exclusion, or interpretability rather than interaction consequences.

A later event-IPV association built on a `construct_proximal_descriptor` endpoint is circular and is not admissible as primary evidence. Any report, protocol, ontology, or downstream contract that refers to a primary behavioural consequence endpoint must filter out `construct_proximal_descriptor`, `planner_system_event`, and `annotation_quality_label` unless an approved later protocol explicitly defines a separate secondary analysis.

Modified SPEC work packages/gates: W1 Event ontology; W5 Annotation codebook v2; W7 Agreement and downstream-analysis protocol; Gate 012-1 Outcome-blind ontology and thresholds; Gate 012B Later analysis authorization.

Rationale: The endpoint tier prevents a validation endpoint from semantically restating the IPV construct it is meant to test.

## B02 - Development-Subset Threshold Source Is Under-Specified

Finding: The development-subset threshold source lacks pre-frozen selection, provenance, separation, and proxy-protection rules.

Binding resolution: Threshold development is governed by a two-track rule.

Confirmatory track: Any threshold actually used by the extractor, pilot, blind package, readiness report, or any later RQ012B analysis must be derived from a pre-frozen, outcome-blind development subset. Before threshold work begins, the confirmatory subset must have documented provenance covering sample frame, random or scenario-stratified selection method, recorded seed/hash, no overlap with formal annotation or validation items unless explicitly justified, prohibited outcome-correlated proxies, authorized builder role, and a frozen manifest produced before viewing anything beyond outcome-blind data-health summaries. Confirmatory subset construction may not use IPV values, official coordination scores/ranks/team identities, later event-IPV associations, event outcomes, filenames/paths/orderings/thumbnails, area/scenario/run IDs, manifest-derived strata, or prior/borrowed annotation files unless the specific field is explicitly approved in writing as an outcome-blind design variable.

Exploratory track: Analysts may explore any interesting subset for intuition, candidate thresholds, sensitivity ideas, and failure-mode discovery. Every exploratory threshold note or artifact must carry an explicit leakage-risk annotation stating how the subset was chosen, which potential outcome-correlated proxies were involved, and a leakage-risk rating. Outputs from this track must be stored in a clearly marked quarantine labelled `exploratory - not leakage-protected` and must not be used as binding thresholds.

Firewall: An exploratory-derived threshold may not silently become a confirmatory threshold. To be adopted, it must be re-derived or confirmed on the frozen outcome-blind confirmatory subset and must pass all confirmatory provenance rules. The explore-to-confirm promotion must be recorded explicitly, including the exploratory source, confirmatory subset manifest, decision rationale, sensitivity check, and approval record. Choosing thresholds by the strength, sign, significance, or narrative usefulness of any later IPV association remains forbidden.

Modified SPEC work packages/gates: W2 Outcome-blind threshold rationale; W3 Automatic extractor prototype; W7 Agreement and downstream-analysis protocol; Gate 012-1 Outcome-blind ontology and thresholds; Gate 012B Later analysis authorization.

Rationale: The two-track rule preserves productive exploration while preventing leakage-contaminated thresholds from entering confirmatory extraction or later validation.

## B03 - Gate 012B Is Not Explicitly Dependent On All Wave-A Gates

Finding: Gate 012B could be read as allowing event-IPV analysis after only the original literal 012B list, bypassing Wave-A gates and frozen deliverables.

Binding resolution: Event-IPV analysis in RQ012B may begin only after all of the following are documented and frozen:

- Gate 012-0 has documented PASS status.
- Gate 012-1 has documented PASS status.
- Gate 012-2 has documented PASS status.
- Gate 012-3 has documented PASS status.
- The ontology is frozen, including `endpoint_eligibility`.
- The threshold rationale is frozen, including confirmatory-track provenance and any explore-to-confirm promotion records.
- The blind-package audit is frozen.
- The agreement-analysis protocol is frozen and saved before any agreement results are viewed.
- Two real annotation files exist from independent human annotators.
- Agreement has been computed under the frozen protocol.
- RQ007, RQ009, and RQ011 freezes are complete and cited.
- An explicit recorded authorization permits RQ012B event-IPV analysis.

Any path satisfying only the original literal Gate 012B list is invalid. A worker must not start event-IPV analysis if any item above is missing, provisional, failed, undocumented, or merely assumed.

Modified SPEC work packages/gates: W1 through W8 as prerequisites; Gates 012-0, 012-1, 012-2, 012-3, and 012B.

Rationale: Gate 012B must be a non-bypassable authorization chain, not a partial checklist that can be satisfied while upstream readiness remains unresolved.

## A01 - Candidate List Should Include Or Explicitly Exclude Additional Consequence Classes

Finding: W0 does not explicitly include or exclude collision/contact, off-route or no-progress timeout, takeover, emergency stop, abrupt lateral comfort events, or yielding to non-counterpart actors.

Binding resolution: W0 and W1 must treat the following as mandatory audit candidates unless documented as unavailable or non-goals under W0/W1: collision/contact, off-route or no-progress timeout, human/operator takeover, emergency stop, abrupt lateral comfort events, and yielding to a non-counterpart actor. For each candidate, record required signals, availability status, actor identity rule, whether it can be outcome-blindly extracted, and its `endpoint_eligibility` tier. If a candidate is excluded, the exclusion must state whether it is unavailable, outside Wave-A scope, not derivable from current materials, or not conceptually appropriate as a consequence endpoint.

Modified SPEC work packages/gates: W0 Event-signal availability audit; W1 Event ontology; Gate 012-0 Signal feasibility; Gate 012-1 Outcome-blind ontology and thresholds.

Rationale: Mandatory candidate accounting prevents quiet omission of relevant consequence classes and makes exclusions auditable.

## A02 - Denylist Should Name Indirect Leakage Proxies Globally

Finding: Indirect leakage proxies are audited in W4 but not globally denied for all pre-analysis work.

Binding resolution: The global denylist applies to all event-definition, threshold-development, extraction-pilot, blind-package, annotation-preparation, merge-validation, and pre-analysis work. In addition to explicit outcomes already listed in the SPEC, the following are prohibited leakage channels unless explicitly approved in writing as outcome-blind design variables: area IDs, scenario IDs, run IDs, filenames, paths, item ordering, thumbnails, manifest-derived strata, and prior or borrowed annotation files. Approval must specify the variable, why it is outcome-blind for the intended use, who authorized it, and where the decision is recorded. Approval for one use does not generalize to other uses.

Modified SPEC work packages/gates: Denylist; W0 through W8; Gates 012-0, 012-1, 012-2, and 012B.

Rationale: Leakage can enter through metadata and provenance fields even when explicit IPV or score columns are hidden.

## A03 - Blind Issuance Should Be Reproducibility-Auditable

Finding: W4 requires a leakage audit but does not require auditable issuance artifacts for neutral IDs, ordering, checksums, or metadata stripping.

Binding resolution: Blind issuance must be reproducibility-auditable before annotator release. The blind-package audit must include an issuance manifest with neutral item IDs, source-to-neutral mapping custody location, per-file checksums, randomized order seed, package version, evidence of embedded-metadata stripping from annotator-facing materials, and auditor sign-off before release. The manifest must be frozen before any formal annotation files are accepted.

Modified SPEC work packages/gates: W4 Existing blind-package audit; W6 Two-human annotation coordination; Gate 012-2 Blind-package integrity; Gate 012-3 Human coordination readiness.

Rationale: A reproducible issuance manifest makes blinding claims inspectable without exposing protected identities to annotators.

## A04 - Simulated-Label Rejection Needs Provenance Evidence

Finding: W8 requires rejection of simulated labels, but content-only checks cannot reliably prove labels came from real independent annotators.

Binding resolution: Merge validation must include provenance controls in addition to content tests. Required controls are annotator attestation, file-provenance checks, a controlled submission channel, and metadata/audit logs for receipt, custody, and merge. A label file may not be accepted as a real human annotation solely because its CSV content is non-empty, non-duplicate, and schema-valid. Merge validation must record which provenance checks passed, who verified them, and any exceptions or rejected submissions.

Modified SPEC work packages/gates: W6 Two-human annotation coordination; W8 Merge-script validation; Gate 012-3 Human coordination readiness; Gate 012B Later analysis authorization.

Rationale: Human-label validity is a provenance claim as well as a file-content claim.

## Binding Constraints

Downstream RQ012A contracts may cite the following checklist verbatim:

- `endpoint_eligibility` is mandatory for every event or label in the ontology/codebook.
- Only `independent_consequence_endpoint` items may be primary event-IPV validation endpoints in later RQ012B.
- `construct_proximal_descriptor` items are forbidden as primary validation endpoints; any event-IPV association built on them is circular and inadmissible as primary evidence.
- `planner_system_event` and `annotation_quality_label` items must remain analytically distinct from primary behavioural consequence endpoints unless an approved later protocol defines a secondary analysis.
- Confirmatory thresholds used by extractors, pilots, reports, or RQ012B must come from a pre-frozen, outcome-blind development subset with documented sample frame, selection method, seed/hash, no-overlap rule, proxy denylist, builder role, and frozen manifest.
- Exploratory threshold work is allowed only when explicitly labelled with leakage-risk details and quarantined as `exploratory - not leakage-protected`.
- Exploratory-derived thresholds may be adopted only after re-derivation or confirmation on the frozen outcome-blind confirmatory subset, with an explicit explore-to-confirm promotion record.
- Thresholds must never be chosen by the strength, sign, significance, or narrative usefulness of later IPV associations.
- Gate 012B is non-bypassable: event-IPV analysis requires documented PASS for Gates 012-0, 012-1, 012-2, and 012-3; frozen ontology; frozen threshold rationale; frozen blind-package audit; frozen agreement-analysis protocol saved before viewing agreement results; two real annotation files; computed agreement; RQ007/RQ009/RQ011 freezes; and explicit recorded authorization.
- Collision/contact, off-route or no-progress timeout, human/operator takeover, emergency stop, abrupt lateral comfort events, and yielding to a non-counterpart actor must be audited or explicitly documented as unavailable/non-goals under W0/W1.
- Global leakage denylist includes area/scenario/run IDs, filenames, paths, item ordering, thumbnails, manifest-derived strata, and prior or borrowed annotation files unless explicitly approved as outcome-blind design variables.
- Blind issuance requires a frozen manifest with checksums, neutral item IDs, randomized order seed, embedded-metadata stripping evidence, and auditor sign-off before release.
- Simulated/borrowed-label rejection requires provenance evidence: annotator attestation, file-provenance checks, controlled submission channel, and metadata/audit logs.
- Wave A remains readiness work only: no event-IPV association, no causal claims, no fabricated labels or agreement, and no claim that the behavioural reference is fully independent truth.
