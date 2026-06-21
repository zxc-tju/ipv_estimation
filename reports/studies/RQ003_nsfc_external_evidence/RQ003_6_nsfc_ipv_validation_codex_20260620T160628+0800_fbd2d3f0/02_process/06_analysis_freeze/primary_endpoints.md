# RQ003 Frozen Primary Endpoints

Worker: `RQ003_phase3_freeze_002`

## Primary Unit and Scope

- Unit: one `team x scenario` cell.
- Scope: approved top-five cohort only.
- Team universe: the ten teams in `data/onsite_competition/top5_research_subset/tables/top5_session_manifest.csv`.
- Area source: outcome-free session manifests and top-five directory names.
- Scenario universe: the canonical fifteen scenario labels `A1`-`A5`, `B1`-`B5`, and `C1`-`C5` described in the frozen plan.
- Cell membership rule: Phase 4 must derive `team x scenario` cell membership from an outcome-free scenario/session mapping or replay metadata. If scenario membership exists only in a score-joined table, Phase 4 must stop before model fitting.

## Primary Sample

Primary cells must satisfy all conditions:

1. `mapped == true`: the replay/session can be mapped to an approved top-five team and a scenario cell using outcome-free structure.
2. `high_support == true`: all primary IPV summaries use only high-support frames under Gate 0 support rules.
3. `scenario != A1`: A1 is excluded from the primary continuous analysis.
4. `collision == 0`: cells with any collision primitive are excluded from the primary continuous analysis.

Cells failing support or mapping are abstained from the primary model. Catastrophic safety failures are reported separately from the continuous coordination-residual model.

## Primary Outcome

Outcome: official coordination residual after `scenario + area` fixed effects.

Residualization contract:

- Do not compute now.
- Fit fixed effects inside each training fold only.
- Apply training-fold fixed-effect centering to held-out cells without using held-out outcome values for model fitting.
- Keep the outcome name as official coordination score/residual until provenance supports a stronger label.

## Primary Predictor

Predictor block added by the full model:

- `D_comp_auc_conflict_time_norm`
- `D_yield_auc_conflict_time_norm`

Definitions:

- `D_comp(t) = max(0, (Q_low(theta_npc, s, tau) - theta_ego) / w(t))`.
- `D_yield(t) = max(0, (theta_ego - Q_high(theta_npc, s, tau)) / w(t))`.
- Use conflict-window, time-normalized AUC only.
- Conflict-window frames are non-abstained high-support frames satisfying the Gate 0 current-frame inter-agent distance cap `<= 41.31042720375975 m`.
- Time-normalized AUC means trapezoidal area over valid conflict-window time divided by valid conflict-window duration, so longer replays do not dominate by duration.
- If a cell has no valid high-support conflict-window frames, it is abstained from the primary sample.

Gate 0 frozen parameters:

- Primary rolling window: 10 frames.
- Minimum observation: 4 frames.
- Short/medium/long sensitivity windows: 5/10/20 frames.
- `Q_low = 0.25`, `m = 0.5`, `Q_high = 0.75`.
- `w_min = 0.19634954084936207 rad`.
- High support requires exact InterHub cell count `>= 30`, estimator error `<= 0.6216308869824523`, `theta_npc` within `[-1.178096549046999, 1.178097245096148] rad`, and distance `<= 41.31042720375975 m`.

## Single Confirmatory Comparison

Only one confirmatory comparison is frozen:

`state + causal kinematics + safety`

versus

`state + causal kinematics + safety + D_comp_auc_conflict_time_norm + D_yield_auc_conflict_time_norm`

The two models must differ only by the two IPV predictors. Preprocessing, folds, imputation, scaling, tuning budget, training data, loss, and evaluation metrics must be identical.

## Generalization

- Primary: leave-one-team-out.
- Secondary: leave-one-scenario-out.
- Boundary: leave-one-family-out for A/B/C family transfer. This is not a significance headline because only three folds exist.

## Metrics

Primary evaluation may report rank correlation, MAE, and CV-R2, but the confirmatory decision is based on the prespecified full-vs-baseline improvement under leave-one-team-out plus the safe-subset agreement rule.

## Non-Confirmatory Endpoints

The following are not confirmatory endpoints:

- comprehensive score,
- area rank,
- overall rank,
- p90/max/onset/latency/gain/phase summaries,
- alternative conflict windows,
- full-window IPV,
- observed PET or realized order,
- post-hoc phase labels,
- all safe-subset combinations beyond S1/S2/S3.
