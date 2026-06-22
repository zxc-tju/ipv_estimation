# RQ007 Plan v0 — Interaction-Conditioned IPV Estimability

Status: `planning`  
Wave: A  
Work group: Group 1  
Date: 2026-06-22

## 1. Research question

> During an interaction, when does an online IPV estimate become behaviourally identifiable,
how does its uncertainty evolve as interaction information accumulates, and when does the
estimate become ambiguous or no longer applicable after interaction resolution?

The plan treats IPV as an interaction-conditioned behavioural measurement, not as a fixed
personality trait and not as a quantity that is meaningful at every frame.

## 2. Primary input

```text
data/derived/interhub/20260612_sigma_0_1_full_rerun/
00_hpc_outputs/sigma01_ipv_timeseries.csv
```

This existing file is the primary time-series input. The study must not regenerate the full
IPV corpus by default. Targeted estimator reruns are allowed only when required fields are
missing or a prespecified perturbation test requires recomputation.

Additional outcome-free inputs may include InterHub case metadata, map/reference-line data,
geometry, role, source, timestamps, sampling-rate metadata, and estimator-generation scripts.

## 3. Core distinctions

The study must keep four objects separate:

1. `theta_hat_i(t)`: current IPV estimate;
2. `sigma_hat_i(t)`: current estimator uncertainty or dispersion;
3. `g_i(t)`: whether the current interaction contains enough information for IPV to be
   estimable;
4. `o(t)`: whether an interaction opportunity exists.

High uncertainty before interaction does not mean IPV equals zero. Increased ambiguity after
resolution does not mean behaviour has returned to a normative baseline. Behavioural change
in the IPV mean must be separated from increased certainty in the estimate.

## 4. Denylist

This plan and its threshold-development work must not read or tune against:

- WOD-E2E human ratings;
- OnSite official scores, rankings, or outcome labels;
- downstream interaction-harm labels;
- later RQ009 interval coverage or deviation outcomes;
- manuscript headline preferences.

## 5. Work packages

### W0 — Provenance and schema audit

- Record SHA-256, file size, row count, modification time, Git commit, and generation path.
- Identify primary keys, duplicate rows, source/case/agent/counterpart identifiers,
  timestamps, frames, sampling rates, missingness, and units.
- Establish the actual definitions of IPV mean/value, IPV standard deviation/error, valid
  frame ratio, optimizer status, and any confidence fields.
- Audit joins to geometry, role, route/reference, source, and episode metadata.
- Document the exact formula currently used for any full-window or episode-level IPV summary.

### W1 — Interaction-opportunity candidates

Develop outcome-free candidate definitions using geometry and causal kinematics, including:

- possible path conflict;
- map conflict point and route overlap;
- closing relation;
- relative position and velocity;
- role availability;
- counterpart stability.

Distance alone must not define interaction existence. Candidate definitions must allow early
anticipatory response before vehicles are physically close.

### W2 — Estimability lifecycle

Describe and model the joint trajectory of IPV mean and uncertainty across the interaction:

- pre-opportunity period;
- opportunity onset;
- early mutual influence;
- negotiation or role establishment;
- resolution;
- post-resolution ambiguity.

Test, without presupposing, whether uncertainty follows a U-shaped, monotonic, or
motif-specific profile.

### W3 — Time-to-estimability

Evaluate outcome-free candidate definitions of sustained estimability, for example:

```text
t_est = first t such that uncertainty < sigma_0 for K consecutive frames
```

Compare candidate `sigma_0` and `K` values only on development/guard data. Do not select them
using external outcomes. Report:

- delay from interaction-opportunity onset to estimability;
- ego/counterpart ordering;
- differences by source, geometry, role, and interaction duration;
- encounters that never become estimable.

### W4 — Non-interaction and mechanical controls

Run controls that can reveal a mechanical history-length effect:

- random pseudo-pairs;
- time-shifted counterparts;
- nearby actors on non-conflicting paths;
- distant pairs with no interaction opportunity;
- counterpart-ID permutation.

If uncertainty contracts similarly under these controls, the convergence claim must be
rejected or reformulated.

### W5 — Targeted perturbation tests

On a representative, outcome-blind subset, rerun the estimator under:

- trajectory noise;
- downsampling;
- dropped frames;
- map/reference offset;
- wrong-lane assignment;
- counterpart switching;
- observation-window sensitivity.

Record failure rates, bias, stability, and uncertainty response.

### W6 — Episode-summary sensitivity

Compare at least:

```text
all-valid-frame mean
interaction-active mean
estimability-weighted mean
```

Assess whether existing episode-level risk/geometry/role patterns depend materially on
including low-information frames.

## 6. Gates

### Gate 007-0 — Input provenance

PASS requires a traceable schema, generation path, and join contract. Unknown uncertainty
semantics are blocking.

### Gate 007-1 — Estimability interpretation

PASS requires explicit separation of interaction opportunity, estimator certainty, and
behavioural IPV dynamics.

### Gate 007-2 — Mechanical-control falsification

PASS requires evidence that the proposed estimability pattern is not explained solely by
longer observation history or arbitrary pairing.

### Gate 007-3 — Provisional contract review

An independent reviewer must approve a provisional:

- interaction-opportunity rule;
- uncertainty definition;
- valid-window rule;
- candidate estimability mask;
- abstention reason taxonomy.

The contract remains provisional until independently reviewed and confirmed on held-out
cases.

## 7. Deliverables

```text
timeseries_data_dictionary.md
timeseries_provenance_manifest.csv
interaction_opportunity_candidates.md
estimability_candidate_metrics.csv
estimability_trace.parquet
estimability_onset_candidates.csv
noninteraction_controls.csv
perturbation_results.csv
full_window_summary_sensitivity.csv
interaction_estimability_contract_provisional.yaml
tried.md
90_report/index.html
```

Formal figures in the HTML report must be generated through the Nature skill and accompanied
by SVG, PDF, PNG, source-data CSV, and a figure manifest.

## 8. Acceptance criteria

- Existing time-series data are used as the primary source rather than unnecessarily
  regenerated.
- The uncertainty field has a verified definition.
- Interaction opportunity is not reduced to distance alone.
- Mean dynamics and uncertainty contraction are analysed separately.
- Non-interaction controls are complete.
- Estimability thresholds are outcome-free.
- Null, reverse, source-specific, and failed patterns are retained.
- The final claim is limited to measurement identifiability, not psychological truth or
  verifier validity.

## 9. Non-goals

- Do not construct the human reasonable interval.
- Do not relate IPV to WOD or OnSite outcomes.
- Do not claim that full-window IPV is the truth to which online IPV must converge.
- Do not claim that IPV is meaningful at every frame.
- Do not write manuscript performance claims.

## 10. Dependency and handoff

RQ008A may begin after W0 establishes the schema and a protected confirmation split.
RQ008B, RQ009, RQ011B, and RQ012B depend on the reviewed RQ007 valid-window and estimability
contract.
