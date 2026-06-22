# RQ008 Plan v0 — InterHub Temporal IPV Discovery

Status: `planning`  
Wave: A  
Work group: Group 2A, followed by independent Group 2B confirmation  
Date: 2026-06-22

## 1. Research question

> What reproducible temporal structures appear in the joint mean and uncertainty of human
IPV as interactions form, roles emerge, risk changes, negotiations resolve, and interaction
information disappears?

This is an open discovery study. The first-stage findings are exploratory and must not be
presented as confirmatory evidence.

## 2. Inputs

Primary input:

```text
data/derived/interhub/20260612_sigma_0_1_full_rerun/
00_hpc_outputs/sigma01_ipv_timeseries.csv
```

RQ008 may also use outcome-free InterHub metadata, geometry, role, source, map/reference,
risk descriptions, and the provisional field definitions produced by RQ007 W0.

RQ008A may start once RQ007 establishes the schema. Formal confirmation must wait until
RQ007 freezes the estimability/valid-window contract.

## 3. Discovery–confirmation separation

Before any temporal exploration:

- split complete scenes/cases into a discovery subset and an untouched confirmation subset;
- never split frames from the same case across subsets;
- record the split seed and file hash;
- prevent discovery workers from reading the confirmation outcomes;
- retain a source-held-out or scene-held-out confirmation option.

Group 2A may explore freely inside the discovery subset. Group 2B may receive only frozen
hypotheses, fixed code, and held-out data.

## 4. Denylist

RQ008A must not read:

- WOD-E2E ratings;
- OnSite scores, ranks, or outcomes;
- downstream verifier performance;
- RQ009 coverage or deviation results;
- the untouched RQ008B confirmation outputs.

## 5. Work packages

### W0 — Temporal data atlas

Build source-, geometry-, role-, and duration-stratified summaries of:

- ego/counterpart IPV mean;
- ego/counterpart uncertainty;
- pair sum and pair difference;
- causal risk proxies;
- interaction progress;
- role availability and stability;
- valid-frame and estimability indicators.

### W1 — Multiple temporal alignments

Compare at least:

- causal interaction progress;
- interaction-opportunity onset;
- provisional estimability onset;
- map conflict-point progress;
- causal closing-time alignment;
- offline oracle phase for discovery only;
- resolution onset.

Oracle phase must never be promoted to a deployed input.

### W2 — Early role assignment

Explore:

- whether roles form soon after interaction onset;
- which actor becomes estimable first;
- role persistence and reversal;
- whether role formation precedes visible braking or yielding;
- whether early role assignment differs by source or geometry.

### W3 — Stage-dependent adjustment

Explore:

- IPV changes as risk rises or falls;
- ego–counterpart lead–lag relations;
- complementarity versus mutual competition;
- asymmetric response;
- hysteresis;
- stability versus ongoing negotiation;
- post-resolution ambiguity.

Do not assume that continuous feedback adjustment exists. An early stable assignment is an
acceptable and potentially important finding.

### W4 — Dynamic motif discovery

Permitted methods include:

```text
functional PCA
trajectory clustering
change-point detection
hidden-state or state-space models
dynamic time warping
motif discovery
joint ego–counterpart sequence clustering
```

Candidate motifs may include early stable role assignment, gradual reciprocal negotiation,
late forced yielding, competitive escalation, over-yielding freeze, oscillation, deadlock,
smooth resolution, or post-resolution ambiguity. These labels are hypotheses, not required
outcomes.

### W5 — Mechanical and compositional controls

Run at least:

- time shuffle;
- within-case reversed time;
- pseudo-pair controls;
- duration-matched nulls;
- random alignment points;
- source-composition balancing;
- uncertainty-only clustering;
- estimability-matched controls.

### W6 — Cross-source and subgroup reproducibility

For each candidate pattern, inspect:

- Waymo;
- nuPlan;
- Lyft;
- Argoverse-2;
- drop-Waymo;
- geometry family;
- role;
- interaction duration;
- high- and low-estimability slices.

Source-specific dynamics are permitted. The study must not force a universal temporal law.

### W7 — Freeze a limited confirmation set

Group 2A may nominate only a small number of candidate hypotheses. Each nomination must
include:

- exact operational definition;
- endpoint and analysis unit;
- expected direction;
- exclusion rules;
- alignment rule;
- subgroup scope;
- held-out test;
- failure criterion.

All unsuccessful searches, rejected motifs, and reverse results remain in `tried.md`.

## 6. Gates

### Gate 008-0 — Protected split

PASS requires a complete-case/scene discovery–confirmation split with no frame leakage.

### Gate 008-1 — Discovery reproducibility

PASS requires rerunnable pipelines, source-linked figures, and a complete attempt log.

### Gate 008-2 — Candidate-hypothesis freeze

An independent reviewer must approve a finite, non-overlapping confirmation list. Exploratory
motif definitions may not be edited after held-out data are opened.

### Gate 008B — Held-out confirmation

This is a later Wave B gate. Confirmation requires fixed code, frozen RQ007 valid-window
rules, held-out data, clustered uncertainty, and explicit pass/fail criteria.

## 7. Deliverables

```text
discovery_confirmation_split.csv
temporal_ipv_atlas.html
mean_uncertainty_state_map.csv
temporal_alignment_comparison.csv
temporal_motif_catalog.csv
candidate_temporal_hypotheses.md
negative_discoveries.md
tried.md
frozen_confirmation_protocol_proposal.yaml
90_report/index.html
```

All reader-facing figures must use the Nature skill and include SVG, PDF, PNG, source-data
CSV, and a figure manifest.

## 8. Acceptance criteria

- Discovery and confirmation data are isolated before exploration.
- Temporal claims distinguish mean dynamics from uncertainty/estimability dynamics.
- At least one mechanical time-trend control and one pseudo-pair control accompany every
  proposed motif class.
- Source-specific and null patterns are reported.
- Oracle phase remains discovery-only.
- The output is a candidate-hypothesis package, not a claim of confirmed temporal norms.

## 9. Non-goals

- Do not train the dynamic human envelope.
- Do not use external ratings or OnSite outcomes.
- Do not label temporal clusters as social norms without held-out confirmation.
- Do not force a reciprocal-feedback narrative.
- Do not modify manuscript claims based only on discovery-stage results.

## 10. Dependency and handoff

RQ008A depends only on RQ007 schema audit. RQ008B depends on the reviewed RQ007 estimability
contract. RQ009 may use only RQ008 variables that survive the frozen confirmation process or
are explicitly labelled exploratory/sensitivity inputs.
