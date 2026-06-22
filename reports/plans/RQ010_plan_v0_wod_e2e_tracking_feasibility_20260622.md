# RQ010 Plan v0 — WOD-E2E Data and Tracking Feasibility

Status: `planning`  
Wave: A  
Work group: Group 4A, followed by Group 4B preference validity  
Date: 2026-06-22

## 1. Research question

> Does the WOD-E2E release provide the actor trajectories, map, timing, candidate-future,
and rating fields required to run the full counterpart-conditioned verifier; if not, what
tracking or data-alignment solution is required, and what compute/storage resources are
needed?

This Wave A study is a feasibility and infrastructure audit. It does not yet download the
full dataset, train a tracker, calculate IPV deviations, or test rating validity.

## 2. Intended downstream role

A later RQ010B study may test whether lower IPV deviation aligns with higher human preference
among multiple candidate trajectories from the same scene. The desired primary analysis uses
the frozen RQ009 M3 verifier:

```text
context + counterpart current IPV
```

If surrounding actor trajectories are missing, the preferred response is to evaluate an
additional tracking pipeline rather than silently redefining the primary study as context-only
M2.

## 3. Information to verify from authoritative sources

- dataset release and access status;
- licence and permitted research use;
- download mechanism and authentication;
- total compressed and expanded size;
- train/validation/test segment counts;
- segment IDs and possible links to other Waymo datasets;
- critical-frame definition;
- candidate future trajectories and identity mapping;
- rating files, aggregation, and rater-level availability;
- ego states, route, map, calibration, and camera streams;
- surrounding actor tracks, IDs, classes, histories, and uncertainty;
- timestamps and coordinate frames;
- overlap or crosswalk to WOMD/WOD segments.

All factual conclusions in the final feasibility report must cite official documentation,
release files, schemas, or primary papers.

## 4. Work packages

### W0 — Official-source inventory

Create a source table containing:

```text
source_id
source_type
publisher
version/date
claim supported
access URL or repository path
licence
retrieval status
```

Prefer official documentation, official repositories, release schemas, and the primary paper.

### W1 — Required-field crosswalk

Map every field needed by RQ009/RQ010B to the released data:

- ego prehistory and candidate future;
- surrounding actor IDs and histories;
- map/lane/route reference;
- timestamps and coordinate transforms;
- candidate-to-rating identity;
- critical frame;
- scenario/segment identity;
- rating and rater metadata;
- fields required for counterpart selection and estimability.

Classify each field as `available`, `derivable`, `requires_alignment`, `requires_tracking`,
`restricted`, or `missing`.

### W2 — Tracking-necessity decision

Assign one status:

```text
T0_NO_TRACKING_NEEDED
T1_LIGHT_AUGMENTATION
T2_FULL_TRACKING_REQUIRED
T3_BLOCKED
```

Definitions:

- `T0`: actor trajectories, IDs, timing, map, and quality are sufficient directly;
- `T1`: released tracks exist but need lane association, counterpart selection, smoothing,
  uncertainty, or minor gap filling;
- `T2`: full multi-camera detection/association/BEV tracking is required;
- `T3`: access, licence, calibration, timing, or other critical information prevents a valid
  study.

### W3 — Technical-option comparison

Evaluate in order:

1. direct released actor tracks;
2. official segment crosswalk to WOMD/WOD tracks;
3. released or official perception outputs;
4. existing multi-camera 3D/BEV tracking system;
5. custom detection, association, ego-motion compensation, BEV fusion, tracking, map matching,
   and uncertainty propagation.

For each option report:

- data prerequisites;
- expected accuracy and failure modes;
- map/lane compatibility;
- reproducibility;
- engineering effort;
- licence constraints;
- CPU/GPU/storage needs;
- suitability for critical-frame counterpart selection.

### W4 — Tracking quality-gate proposal

Before reading ratings, propose outcome-blind acceptance metrics for a future pilot:

- actor detection recall;
- position and velocity error;
- ID-switch rate;
- track continuity;
- critical-frame actor coverage;
- available pre-critical history;
- occlusion rate;
- map/lane association accuracy;
- counterpart-selection agreement;
- uncertainty calibration.

The manual QA subset must be random or scenario-stratified and independent of ratings and IPV.

### W5 — Compute and storage budget

Estimate three scenarios:

```text
A. direct tracks / alignment only
B. light track augmentation and map matching
C. full eight-camera tracking
```

For each estimate:

- download size;
- expanded data size;
- intermediate feature/BEV/track storage;
- CPU-hours;
- GPU-hours;
- recommended GPU memory;
- parallelism;
- local workstation runtime;
- HPC runtime;
- pilot and full-run cost ranges.

Document assumptions and confidence intervals. Do not present point estimates without their
basis.

### W6 — HPC decision

Return exactly one recommendation:

```text
LOCAL_CPU_OK
SINGLE_GPU_WORKSTATION_OK
HPC_RECOMMENDED
HPC_REQUIRED
BLOCKED_PENDING_ACCESS
```

The decision must follow from verified data scale and either published benchmarks or a small,
licence-compliant pilot benchmark. Full-dataset download is not required during Wave A.

### W7 — Candidate-future actor protocol

Assess how the three candidate ego futures will share surrounding actor futures.

Primary candidate protocol:

```text
shared open-loop opportunity structure
```

- use tracked actor history before the critical frame;
- produce one frozen actor forecast shared across ego candidates;
- compare candidates under the same opportunity structure;
- do not claim closed-loop actor response.

Sensitivity candidate protocol:

```text
candidate-conditioned actor forecast
```

This may be considered later, but forecast uncertainty, model provenance, and outcome-blind
training must be explicit. Generated actor responses cannot be labelled realised harm.

## 5. Gates

### Gate 010-0 — Access and licence

PASS requires verified access conditions and a licence-compatible research path.

### Gate 010-1 — Schema sufficiency

PASS requires a field-level determination of whether full M3 is directly possible, possible
after alignment, or dependent on tracking.

### Gate 010-2 — Tracking and compute decision

PASS requires a preferred technical route, fallback route, quality gate, pilot design, and
resource estimate.

### Gate 010-3 — Independent feasibility review

An independent reviewer must confirm that the tracking/HPC recommendation is evidence-based
and that no rating information influenced the tracking-quality gate.

## 6. Deliverables

```text
official_source_inventory.csv
data_access_and_license_audit.md
wod_schema_requirements.csv
field_availability_crosswalk.csv
tracking_need_decision.json
tracking_options_comparison.csv
tracking_quality_gate_proposal.yaml
compute_storage_budget.csv
pilot_benchmark_plan.md
hpc_decision.md
candidate_future_actor_protocol.md
wod_feasibility_report.html
```

Formal figures in the HTML report must use the Nature skill and include source-data files and
a figure manifest.

## 7. Acceptance criteria

- Tracking need is classified T0–T3 using official evidence.
- Missing actor trajectories trigger a real tracking/alignment evaluation, not an unrecorded
  M2 substitution.
- Access and licence constraints are explicit.
- Compute and storage estimates cite assumptions or pilot measurements.
- The HPC recommendation is reproducible.
- Rating data are not used to tune tracking quality.
- The final report states whether full M3 preference validation is feasible and under what
  boundary.

## 8. Non-goals

- Do not download the full dataset in Wave A.
- Do not train a tracker in Wave A.
- Do not calculate IPV deviation or rating association.
- Do not use human ratings to select tracking methods or thresholds.
- Do not claim realised interaction harm from candidate trajectories.
- Do not silently fall back to M2 while describing the full M3 as validated.

## 9. Dependency and handoff

RQ010A may run immediately. RQ010B begins only after:

- Gate 010 passes;
- required data are acquired;
- any tracking pipeline passes its frozen quality gate;
- RQ009 produces a frozen leave-Waymo-out verifier and prediction interface.
