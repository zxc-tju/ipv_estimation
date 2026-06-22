# RQ011 Plan v0 — OnSite Full-Universe and Run-Level Readiness

Status: `planning`  
Wave: A  
Work group: Group 5A  
Date: 2026-06-22

## 1. Research question

> Can the full OnSite competition universe be represented as a reliable
scenario–algorithm–run dataset with traceable replay, score, trajectory, and event fields for
later matched-scenario validity and interaction-consequence studies?

This is a data-readiness and provenance study. It must not test IPV–outcome associations.

## 2. Existing evidence and boundary

RQ003 established a high-quality top-five pilot and a clean 150-cell mapping for its approved
cohort, but the full scored universe contains unmatched, wrong-folder, multi-session, and
partial-case problems. The top-five result is therefore a pilot and boundary reference, not a
substitute for a full-universe audit.

## 3. Inputs

Potential local inputs include:

- `data/onsite_competition/00_manifest/`;
- `data/onsite_competition/raw/`;
- `data/onsite_competition/top5_research_subset/`;
- archived score tables, SQL exports, diagnosis reports, and replay logs;
- RQ003 provenance tables and mapping audits.

Every input must be registered with path, hash, size, date, source, and field-level authority.

## 4. Denylist

During readiness auditing, workers must not:

- compute IPV–score or IPV–rank associations;
- select mappings based on favorable IPV or outcome relationships;
- tune exclusions to improve statistical significance;
- describe coordination as expert-rated unless direct source evidence proves it;
- treat post-interaction NPC trajectories as matching variables;
- infer repeated runs without an explicit run/session identity contract.

## 5. Work packages

### W0 — Full-universe inventory

Build a master inventory across:

```text
area
team / algorithm
session
task
scenario
case
run / repetition
replay and log files
score tables
diagnosis reports
event and intervention logs
```

Classify items as score+replay, score-only, replay-only, media-only, partial, duplicate, or
ambiguous.

### W1 — Canonical identity contract

Define and document:

```text
area_id
algorithm_id
team_id
session_id
task_id
scenario_id
case_id
run_id
actor_id
counterpart_id
```

Resolve or flag:

- one algorithm with multiple sessions;
- multiple candidate sessions for one score vector;
- wrong-folder sessions;
- partial scenario coverage;
- duplicated material;
- cross-area name collisions;
- score-team naming mismatches.

No candidate mapping may be silently promoted to clean.

### W2 — Replay–score–run mapping

Create a row-level crosswalk and assign each row one mapping status:

```text
unique_clean
unique_sql_disambiguated
one_to_many_unresolved
wrong_folder_candidate
score_only
replay_only
partial_case
unmatched
excluded_with_reason
```

Each resolution must cite the authoritative evidence used, such as SQL task ownership,
manifest fields, score vectors, PDF reports, or timestamps.

### W3 — Run-level field availability

For every clean run/cell, audit whether the following are available and usable:

- synchronized ego trajectory;
- surrounding actor trajectories and IDs;
- timestamps and sampling rate;
- map, route, role, and scenario semantics;
- official per-scenario scores;
- success/failure and mission state;
- collision and rule violations;
- TTC/APET/minimum distance or inputs required to derive them;
- intervention, fallback, replanning, or safety-controller logs;
- initial conditions;
- simulator script/version/seed if applicable.

### W4 — Repeated-run and matched-scenario feasibility

Determine whether true repeated runs exist and whether they are comparable. Audit:

- explicit run IDs;
- same algorithm under repeated initial conditions;
- same scenario definition across algorithms;
- traffic/NPC configuration comparability;
- script/version/seed availability;
- route and actor identity comparability;
- whether the appropriate analysis unit is run, case, cell, session, or scenario aggregate.

If repeated runs are not identifiable, downstream work must be described as matched-scenario
cross-algorithm comparison rather than repeated-run analysis.

### W5 — Interaction-estimability interface readiness

Without defining the final RQ007 thresholds, check whether the data can support derivation of:

- interaction opportunity;
- ego and counterpart estimability;
- interaction onset and resolution;
- counterpart identity stability;
- map/role confidence;
- abstention reasons.

Document which fields are observed, derivable, unavailable, or unreliable.

### W6 — Missingness and selection-bias audit

Compare clean, partial, ambiguous, and missing units across:

- area;
- team/algorithm;
- scenario/family;
- score dimensions;
- efficiency and coordination;
- success/failure;
- data quality.

The audit is descriptive and must not include IPV predictors.

### W7 — Analysis-unit recommendation

Recommend one of:

```text
run-level analysis
case-level analysis
team×scenario cell analysis
session-level analysis
scenario-aggregate analysis
not identifiable
```

State required exclusions, weighting, clustering, and fixed/random effects for later RQ011B.

## 6. Gates

### Gate 011-0 — Source provenance

PASS requires authoritative mapping sources and an explicit status for every scored unit.

### Gate 011-1 — Identity and run readiness

PASS requires a canonical ID contract and an evidence-based conclusion about repeated runs.

### Gate 011-2 — Field sufficiency

PASS requires enough trajectory, map, counterpart, score, and event fields for at least one
well-defined downstream analysis unit.

### Gate 011-3 — Missingness boundary

PASS requires explicit selection-bias analysis and frozen exclusions. Non-random missingness
must not be hidden.

### Gate 011-4 — Independent audit review

An independent reviewer must verify joins, duplicates, exclusions, and the recommended
analysis unit.

## 7. Final readiness statuses

Return exactly one:

```text
READY_FULL_UNIVERSE
READY_WITH_FROZEN_EXCLUSIONS
TOP5_ONLY
RUN_LEVEL_NOT_IDENTIFIABLE
BLOCKED_MAPPING
```

## 8. Deliverables

```text
full_universe_inventory.csv
source_provenance_manifest.csv
canonical_id_contract.md
run_level_crosswalk.csv
replay_score_run_mapping.csv
run_field_availability.csv
repeated_run_feasibility.md
initial_condition_comparability.csv
interaction_estimability_field_audit.csv
missingness_selection_audit.csv
analysis_unit_recommendation.md
onsite_readiness_status.json
onsite_readiness_report.html
```

Formal figures must use the Nature skill and include source-data exports and a figure
manifest.

## 9. Acceptance criteria

- Every official scored unit has a mapping status.
- Ambiguous and wrong-folder mappings remain visibly flagged.
- Repeated-run claims are supported by explicit identifiers.
- The recommended analysis unit is justified.
- Missingness and area/team concentration are quantified.
- Coordination remains labelled official/generated unless provenance proves otherwise.
- No IPV–outcome association is performed.
- The final readiness status is independently reviewed.

## 10. Non-goals

- Do not rerun the RQ003 confirmatory analysis.
- Do not search for statistical significance.
- Do not estimate beyond-safety value.
- Do not perform post-treatment NPC matching.
- Do not generate manuscript validity claims.

## 11. Dependency and handoff

RQ011A may start immediately. RQ011B matched-scenario validity requires:

- a PASS readiness gate;
- frozen exclusions and analysis unit;
- the reviewed RQ007 estimability contract;
- frozen RQ009 predictions and abstention reasons.
