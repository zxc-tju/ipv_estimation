# RQ011B Locked SAP v4 Moment-Level Monitor Amendment

Status: **LOCKED - PI-approved 2026-06-29; supersedes the cell-level criterion of SAP v2/v3**  
RQ: `RQ011B`  
Run: `RQ011B_1_matched_scenario_20260625T202454_8331bd49`  
Worker: `RQ011B-P5-moment-lock`  
Role: registrar / moment-level pre-registration lock  
Prior criteria superseded:
`reports/plans/RQ011B_SAP_v2_interaction_conditioned_locked_20260628.md`,
`reports/plans/RQ011B_SAP_v3_broad_interaction_failure_locked_20260628.md`  
Accepted v2 packet:
`reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/`  
Independent re-review packet:
`reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_rereview/`

This locked amendment is the binding RQ011B moment/event-level monitor
pre-registration. It supersedes the cell-level binary criterion approach in SAP
v2/v3 because the cell-level binary criterion was ill-posed for this run:
collision was too sparse, while any-failure was saturated at 285/285 cells. The
analysis now moves to onset-defined failure events and exposure-matched
non-failure pseudo-onsets.

No analysis, scorer run, event/control join, outcome read, model fit, plot,
knowledge decision edit, registry/dashboard edit, paper-repository edit, or
claim acceptance is authorized by this lock. The next gate is the moment-level
pre-outcome build: per-frame M3/M2/IPV_removed deviations, failure-event
timestamp ledger, and C1-C4 controls, followed by moment W1 as the first
outcome-touching run.

## 1. Ultimate RQ And Claim Boundary

Ultimate RQ:

```text
Does local IPV deviation from the human cooperative-norm envelope, strictly
before the onset of a failure event, flag and precede real interaction-failure
events as a parsimonious, interpretable, directional runtime monitor?
```

The intended claim is bounded to an associational, exposure-conditioned,
interpretable runtime-monitor claim. This lock does not support causal,
planner-benefit, safety-benefit, legal-fault, human-normative, or
independence-from-physical-exposure claims.

## 2. Unit, Events, And Timing

The primary unit is one timestamped **failure event occurrence** from the 18
confirmatory categories. Each occurrence has:

```text
t_onset       = earliest frame of the event-defining buildup or precursor
t_fail        = online confirmation/emission timestamp
event_interval = [t_onset, t_resolution]
```

The 18-category timing rules are frozen in:

```text
reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/category_timing_table.csv
```

Rows without a defensible `t_onset` are
`PRE_WINDOW_DESCRIPTIVE_ONLY`. Rows without a defensible timestamp are
`TIMESTAMP_UNAVAILABLE`. These rows remain in coverage accounting but cannot
support confirmatory onset-safe inference.

## 3. Primary And Secondary Windows

The primary onset-safe pre-window is:

```text
W_pre = [t_onset - 2.0s, t_onset - 0.5s]
```

The primary event/control scores are medians over valid frames in `W_pre`:

```text
local_signed_pre = median(M3_signed_dev_f over W_pre)
local_abs_pre    = median(M3_abs_dev_f over W_pre)
local_aggr_pre   = median(M3_aggressive_tail_f over W_pre)
local_pass_pre   = median(M3_passive_tail_f over W_pre)
```

Support requires at least 3 valid scored frames and at least 0.5 s elapsed span
inside `W_pre`. Missing pre-window values may not be imputed from post-window
frames. If support fails, the row is `IPV_PRE_UNAVAILABLE`.

The at-moment secondary value is the closest valid scored frame with timestamp
`<= t_fail` and `t_fail - timestamp <= 0.25s`. It is coincident context only,
cannot be described as predictive, and cannot rescue a failed onset-safe
pre-window result.

Sensitivity windows are report-only and non-confirmatory:

```text
[t_onset - 1.5s, t_onset - 0.25s]
[t_onset - 3.0s, t_onset - 0.5s]
nearest valid value at t_onset - 0.5s within 0.25s tolerance
```

## 4. Per-Frame M3/M2/IPV_Removed Build

The per-frame build manifest is frozen at:

```text
reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/per_frame_m3_m2_build_manifest.md
```

Locked defaults:

```text
primary current IPV column = ipv_ego_hw10
primary envelope level = 90%
deviation_scale_floor = 0.05
M3 scorer = frozen RQ009 M3 scorer
M2 scorer = frozen RQ009 M2/context-only scorer
IPV_removed = frozen RQ009 IPV-removed tier with counterpart-IPV channel zeroed
```

Per-frame deviations must be materialized and hashed before reading authorized
failure timestamps or constructing controls. The manifest hard-deny-lists
target/future fields, official scores, failure labels, event counts, event
intervals, controls, and post-event or closest-approach fields. Any build-order
deviation is a W1 protocol deviation.

## 5. Failure And Control Moment Sets

The primary composite is:

```text
any_failure_moment =
  any timestamped, onset-defined event occurrence from the 18 confirmatory
  categories
```

Directional composites:

```text
aggressive_failure_moment =
  any timestamped, onset-defined AGGRESSIVE category occurrence

passive_failure_moment =
  any timestamped, onset-defined PASSIVE category occurrence

undirected_failure_moment =
  any timestamped, onset-defined SEVERE_UNDIRECTED category occurrence
```

All non-failure controls use all-category full-interval exclusion plus guard.
For every failure event in the same trajectory/cell:

```text
G_event = 3.0s
forbidden_event_guard = [event_onset - G_event, event_resolution + G_event]
```

A candidate control pseudo-onset `t_control` is eligible only if:

```text
[t_control - 2.0s, t_control] is disjoint from every forbidden_event_guard
```

The exclusion applies to C1-C4, every category, and all event intervals in the
same trajectory/cell. C1/C3 exposure matching uses pre-window exposure only.

## 6. Required Control Constructions

The control rules are frozen in:

```text
reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/control_constructions_v2.csv
```

Required constructions:

- `C1_within_interaction_matched`: primary confirmatory control. Same
  `algorithm_id x scenario_id` cell controls, preferably same actor pair,
  matched on pre-window exposure bins; up to 5:1 controls per event.
- `C2_all_non_failure_baseline`: all eligible non-failure interaction
  pseudo-onsets after all-category interval-plus-guard exclusion. Used for
  runtime prevalence, PR context, and false alarms per interaction minute.
- `C3_exposure_matched`: non-failure pseudo-onsets matched on pre-window
  distance, closing rate, relative speed, and TTC bins.
- `C4_random_non_failure`: seeded non-failure sample from the C2 universe,
  primary seed `20260629`, target 5:1 controls.

C1 is the primary support and decision construction. C2-C4 are robustness and
sensitivity checks and cannot rescue a failed C1 result.

## 7. Primary Test And Parsimony

Primary confirmatory contrast:

```text
any_failure_moment vs C1 within-interaction matched controls
predictor = local_abs_pre
window = [t_onset - 2.0s, t_onset - 0.5s]
```

Primary descriptive effect sizes:

- matched-set standardized mean difference in `local_abs_pre`;
- matched-set median shift with bootstrap CI;
- odds ratio per one training-set SD increase in `local_abs_pre`;
- matched-set ROC AUC and precision-recall AUC.

Parsimony statistic:

```text
DeltaR2_ipv     = R2_Tjur(base + local_abs_pre) - R2_Tjur(base)
DeltaR2_physics = R2_Tjur(base + physics_full) - R2_Tjur(base)
eff             = DeltaR2_ipv / DeltaR2_physics
```

`physics_full` is closed. It includes only features computed from `W_pre` or
history ending at `t_onset - 0.5s`: relative distance, log-distance, closing
rate, relative speed, ego/counterpart speed, deceleration pressure, TTC and
inverse-TTC risk with missingness, longitudinal jerk, lateral acceleration,
lateral jerk, heading-rate, steering-rate, exposure bins, close-distance
fractions, TTC-risk fractions, ego stop/no-progress duration, 5 s and 10 s
displacement, stop/go state, stop/go run counts and alternations, longest stop
and go runs, open-gap duration/fraction, encroachment-pressure
duration/fraction, corridor-intersection and corridor-entry flags, non-ego
hard-brake/evasion response persistence, prior response flags, near-miss and
contact margins, and active E09/E15 proxy flags.

No component may be added, dropped, tuned, or reweighted after this lock.
Missing components are represented by frozen missingness flags and
training-fold median imputation. If an entire feature family cannot be computed
because required raw fields are absent, parsimony is
`UNDER_IDENTIFIED_OR_DEGENERATE`, not silently reduced.

Locked defaults:

```text
tau_physics = 0.03
tau_ratio   = 0.60
B_bootstrap = 10000
bootstrap_seed = 20260625
```

Gate 1 passes only if C1 has support, `DeltaR2_physics_C1 >= tau_physics` with
CI lower bound greater than 0, `DeltaR2_ipv_C1` CI lower bound greater than 0,
the `local_abs_pre` coefficient is positive with CI excluding zero, and
`eff_C1 >= tau_ratio` by point estimate.

## 8. Monitor Metrics

Required monitor outputs:

- ROC AUC and PR AUC for `local_abs_pre`;
- fixed threshold tables for thresholds `0.0`, `0.5`, `1.0`, and `2.0`;
- primary fixed alarm:

```text
abs(M3_signed_dev) >= 1.0 for >=0.3s of consecutive valid frames
```

- lead-to-onset distribution, with primary horizon
  `[t_onset - 5.0s, t_onset]`;
- C2 false alarms per interaction minute;
- directional alarm metrics for aggressive and passive classes.

Lead-to-confirmation may be reported only as descriptive context for rows that
lack onset-safe timing. Thresholds may be cross-validated only inside training
folds for calibration plots or operational sensitivity tables; no threshold may
be selected after W1 curves are visible.

## 9. Directional Inverted-U

Directional C1 models use onset-safe pre-window tails:

```text
aggressive_failure_moment vs matched controls:
  label ~ matched_set fixed effects + local_aggr_pre_scaled + local_pass_pre_scaled

passive_failure_moment vs matched controls:
  label ~ matched_set fixed effects + local_aggr_pre_scaled + local_pass_pre_scaled
```

Expected signs:

- AGGRESSIVE categories: below-norm/aggressive tail is positive and dominant.
- PASSIVE categories: above-norm/passive tail is positive and dominant.
- SEVERE_UNDIRECTED categories: no directional sign claim.

Sign-flip specificity is required. Aggressive KIN categories remain kinematic
encroachment proxies, not fault or right-of-way labels. Official task and
efficiency categories remain passive only under bounded mission-logistics
language.

## 10. Structure-Preserving Specificity

Required specificity controls:

1. shuffled IPV within exact `scenario_id x exposure_bin x control_construction`
   strata when possible;
2. wrong-cell / wrong-envelope;
3. sign-flip for directional tails;
4. M2 with identical timestamps, windows, support filters, and controls;
5. IPV_removed with identical timestamps, windows, support filters, and
   controls.

`counterpart_swap` remains descriptive only under prior amendment A1. Locked
margins:

```text
delta_beta = 0.05
delta_R2   = 0.01
B_perm     = 10000
perm_seed  = 20260625
```

## 11. Generalization

Primary generalization is leave-one-scenario-out (LOSO). Scaling, optional
model parameters, and learned calibration are fit only on the other scenarios.
Held-out scenario intercepts are not fit. LOSO reports log-loss skill, ROC AUC
where both classes exist, PR AUC, fixed-threshold alarm performance, fold
validity, and no-single-scenario dominance checks.

Leave-one-team-out (LOTO) is secondary when at least 10 teams have both event
and control support.

## 12. Sparse Handling And Multiplicity

Sparse tiers:

- `>=20` failure events and `>=20` eligible controls: full analyses;
- smaller class `10-19`: M3-only exact or blocked-permutation effect and
  descriptive monitor metrics; no positive support;
- smaller class `<10`, all-event, all-control, timestamp unavailable, onset
  unavailable, or predictor unavailable: `UNDER_IDENTIFIED`.

Single primary confirmatory test:

```text
any_failure_moment vs C1 within-interaction matched controls
predictor = local_abs_pre
window = [t_onset - 2.0s, t_onset - 0.5s]
```

C2-C4 cannot rescue failed C1. Across-control consistency requires all valid
C1-C4 contrasts to have positive point estimates and no reliable opposite
effect, with C1 plus at least two of C2-C4 showing positive CI support.

Secondary per-category tests use BH-FDR with fixed denominator:

```text
m = 18
```

The denominator does not shrink after support filtering, sparse fallback,
timestamp unavailability, onset unavailability, or model failure.

## 13. Interpretation Labels And Ceiling

Aggregate monitor labels:

- `SUPPORTED_MONITOR`
- `NON_PARSIMONIOUS`
- `NON_SPECIFIC`
- `NON_GENERALIZING`
- `UNDER_IDENTIFIED`

Directional labels:

- `DIRECTIONALLY_INTERPRETABLE`
- `NON_DIRECTIONAL`
- `UNDER_IDENTIFIED_DIRECTIONAL`

The strongest positive conclusion requires `SUPPORTED_MONITOR` and
`DIRECTIONALLY_INTERPRETABLE`, and remains only an interpretable runtime-monitor
claim. This lock forbids causal, planner, deployment, legal-fault, and
normative-authority claims.

## 14. Leakage Firewall And Required W1 Outputs

Forbidden before W1 analysis:

- endpoint values/counts/prevalence except authorized W1 timestamp and support
  accounting;
- IPV-outcome associations;
- W1 scorer outputs or fitted outcome models;
- official score, rank, or team identity for tuning;
- confirmation-time or event-interval kinematics for primary matching;
- event-buildup frames in the primary predictor;
- closest or critical approach frame selected using a future failure;
- realized future minimum distance, realized passing order, PET, intensity, or
  post-failure trajectories;
- full-cell or full-window mean IPV;
- post-hoc category/window/control/threshold/physics/support selection.

W1 must produce the category timing ledger, per-frame M3/M2/IPV_removed signed
deviation tables with hash manifests, C1-C4 control tables with exclusion and
pre-window exposure diagnostics, primary C1 effect/parsimony table, C2-C4
sensitivity tables, monitor tables, directional/sign-flip tables,
structure-preserving specificity tables, LOSO and optional LOTO tables,
fixed-denominator per-category BH-FDR table, coverage table, and deviations
log.

## Acceptance Record

Audit chain:

```text
cell-level W1 saturation
  -> moment redesign
  -> independent review
  -> v2 moment packet
  -> independent re-review PASS
  -> PI accepted v2 with recommended defaults on 2026-06-29
  -> this locked SAP v4 moment-level monitor amendment
```

Cell-level rationale: collision was too sparse, while cell-level any-failure
was saturated at 285/285 cells. Therefore, the current RQ011B criterion moves
from cell-level binary classification to onset-defined moment/event-level
monitor validation.

Locked source pointers:

- v2 preregistration:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/prereg_moment_v2.md`
- category timing table:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/category_timing_table.csv`
- control constructions:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/control_constructions_v2.csv`
- per-frame build manifest:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/per_frame_m3_m2_build_manifest.md`
- PI decision brief:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_draft/pi_decision_brief_moment_v2.md`
- independent re-review PASS:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_rereview/prereg_moment_v2_rereview.md`
- PI acceptance brief:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/05_moment_level/prereg_moment_v2_rereview/pi_acceptance_brief_moment.md`

