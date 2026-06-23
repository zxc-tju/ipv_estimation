# RQ012 Plan v0 — OnSite Event Ontology and Blind-Annotation Readiness

Status: `planning`  
Wave: A  
Work group: Group 6A  
Date: 2026-06-22

## 1. Research question

> Which automatically extractable and human-observable interaction events can serve as an
independent behavioural consequence reference for later testing of competitive and
over-yielding IPV deviations?

This Wave A study defines, audits, and coordinates evidence collection. It must not fabricate
human labels or calculate event–IPV associations before the required gates pass.

## 2. Existing foundation

RQ003 prepared an anonymized mechanism sample, a scenario-stratified random validation sample,
an annotation codebook, two annotator templates, a controlled identity map, and a merge script.
The package remains incomplete because no real two-human annotations exist. Those materials
may be reused only after version and leakage review.

## 3. Intended temporal order for later analysis

A later RQ012B study should distinguish:

```text
interaction-opportunity onset
→ IPV-estimability onset
→ first persistent deviation t0
→ subsequent event or harm
```

Deviation onset must not be defined during pre-interaction high uncertainty, after interaction
resolution, or while counterpart identity is unstable.

## 4. Denylist

During event-definition, threshold-development, extraction-pilot, and annotation preparation,
workers and human annotators must not use:

- IPV values or deviation labels;
- official coordination scores, ranks, or team identities;
- final RQ011/RQ013 associations;
- outcome-dependent threshold selection;
- simulated or model-generated human labels;
- another annotator's completed labels.

## 5. Work packages

### W0 — Event-signal availability audit

For every candidate event, identify required signals, units, sampling rate, time alignment,
actor identity, derivability, and missingness. Candidate events include:

- counterpart hard braking;
- high deceleration;
- high jerk;
- forced yielding;
- yield-role reversal;
- repeated stop–go;
- unnecessary stop;
- conflict escalation;
- near miss;
- safety-controller intervention;
- planner fallback;
- repeated replanning;
- trajectory rejection;
- mission failure.

Classify each event as `direct`, `derivable`, `partially observable`, `human-only`, or
`unavailable`.

### W1 — Event ontology

For every event, define:

```text
event_id
behavioural interpretation
required signals
actor
threshold or rule
minimum duration
merge-gap rule
onset and end
missing-data rule
online/offline status
known confounds
candidate direction: competitive / over-yielding / non-specific
```

The ontology must distinguish physical safety events, interaction-quality events, planner
system events, and human-judged behavioural motifs.

### W2 — Outcome-blind threshold rationale

Thresholds may derive from:

- engineering or safety standards;
- primary literature;
- measurement resolution;
- existing platform thresholds;
- distributions in a development subset that excludes IPV and official outcomes.

For each threshold, record alternatives and sensitivity bands. Do not choose thresholds based
on the strength or sign of later IPV associations.

### W3 — Automatic extractor prototype

Implement a pilot on a small outcome-blind sample and report:

- computable fraction;
- event frequency;
- impossible values;
- duplicate/overlapping events;
- sampling-rate sensitivity;
- threshold sensitivity;
- actor attribution failures;
- missing-data failures.

No IPV or official outcome association may be computed in this stage.

### W4 — Existing blind-package audit

Review and version-lock the existing:

- mechanism sample;
- random validation sample;
- controlled identity map;
- anonymized media/trajectory references;
- annotation codebook;
- annotator templates;
- merge script.

Confirm that filenames, metadata, thumbnails, paths, and ordering do not reveal team, area,
score, rank, or IPV information.

### W5 — Annotation codebook v2

Retain or refine labels such as:

```text
aggressive intrusion
appropriate assertiveness
over-yielding / freeze
oscillation
deadlock
smooth reciprocal negotiation
unrelated failure
insufficient evidence
```

For each label include inclusion/exclusion criteria, onset rule, examples, counterexamples,
and confidence level. Training examples must be separate from formal validation items.

### W6 — Two-human annotation coordination

Freeze a protocol requiring:

- at least two independent human annotators;
- blinded materials only;
- no mutual access to labels;
- separate training and formal samples;
- version-locked codebook and media;
- raw labels preserved permanently;
- predefined disagreement/adjudication process;
- no adjudication before independent agreement statistics are saved.

### W7 — Agreement and downstream-analysis protocol

Before labels are opened, freeze:

- primary agreement statistic;
- prevalence-aware secondary statistic;
- clip-level and event-level agreement;
- minimum usable agreement;
- missing/uncertain-label handling;
- disagreement adjudication;
- criteria for allowing later event–IPV analysis.

A later analysis may use Cohen's kappa, Krippendorff's alpha, or another appropriate measure,
but the choice must be justified and frozen before viewing agreement results.

### W8 — Merge-script validation

Test that the merge pipeline rejects:

- empty templates;
- copied duplicate annotator files;
- simulated labels;
- labels with wrong item IDs;
- files that reveal protected identities;
- incomplete required fields.

## 6. Gates

### Gate 012-0 — Signal feasibility

PASS requires a documented signal path for each automatic event retained in the ontology.

### Gate 012-1 — Outcome-blind ontology and thresholds

PASS requires thresholds and event rules selected without IPV, scores, or ranks.

### Gate 012-2 — Blind-package integrity

PASS requires a leakage audit over all annotator-facing files.

### Gate 012-3 — Human coordination readiness

PASS requires named roles, locked materials, independent annotation procedures, and a tested
merge pipeline. Until two real annotators complete the task, status remains:

```text
BLOCKED_FOR_HUMAN_LABELS
```

### Gate 012B — Later analysis authorization

Event–IPV analysis may begin only after:

- two real annotation files exist;
- agreement is computed;
- RQ007 freezes estimability onset;
- RQ009 freezes deviation and persistence definitions;
- RQ011 freezes the analysis universe and run identifiers.

## 7. Deliverables

```text
event_signal_availability.csv
event_ontology.yaml
event_threshold_rationale.md
automatic_event_extractor_spec.md
automatic_event_pilot.csv
automatic_event_pilot_report.md
blind_package_leakage_audit.md
annotation_codebook_v2.md
blind_annotation_protocol.md
annotator_training_package/
annotator_01_template.csv
annotator_02_template.csv
agreement_analysis_protocol.yaml
annotation_merge_validation_tests.md
human_coordination_checklist.md
annotation_readiness_status.json
onsite_event_annotation_readiness_report.html
```

Formal report figures must use the Nature skill and include SVG, PDF, PNG, source-data CSV,
and a figure manifest.

## 8. Acceptance criteria

- Automatic events have auditable signal and threshold definitions.
- Event extraction is piloted without viewing IPV or official outcomes.
- Existing annotation materials pass identity and information-leakage checks.
- The validation sample remains random/scenario-stratified rather than IPV-extreme selected.
- At least two truly independent human annotators are required.
- No simulated labels or agreement are generated.
- Merge validation prevents copied or empty submissions.
- The final report clearly separates readiness from completed behavioural validation.

## 9. Non-goals

- Do not calculate event–IPV association in Wave A.
- Do not estimate causal effects.
- Do not use official coordination to train or tune event detection.
- Do not let Codex or another model substitute for human annotators.
- Do not call the behavioural reference a fully independent ground truth; it remains based on
  the same observed trajectory behaviour.

## 10. Dependency and handoff

RQ012A may start immediately. RQ012B requires real human annotations, the reviewed RQ007
estimability contract, the frozen RQ009 deviation interface, and the RQ011 analysis-unit and
mapping decision.
