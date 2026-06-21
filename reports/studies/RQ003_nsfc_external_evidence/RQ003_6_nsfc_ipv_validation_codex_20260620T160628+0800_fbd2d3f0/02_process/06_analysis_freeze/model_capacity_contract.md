# RQ003 Frozen Model Capacity Contract

Worker: `RQ003_phase3_freeze_002`

## Goal

The confirmatory comparison must test incremental predictive utility relative to a prespecified capacity-matched kinematic and safety baseline. It must not be described as new information beyond kinematics in an information-theoretic sense.

## Model Pair

Baseline:

```text
state + causal kinematics + safety
```

Full:

```text
state + causal kinematics + safety + D_comp_auc_conflict_time_norm + D_yield_auc_conflict_time_norm
```

The two models differ only by the two IPV predictors.

## Online Information Budget

All features must be available under the same online causal budget:

- current and past ego/counterpart states only;
- rolling prefix history only;
- primary history window of 10 frames;
- minimum observation of 4 frames;
- the same 10 Hz rolling contract required by Gate 0;
- no held-out outcome values in fitting, preprocessing, tuning, residualization, or imputation.

## Forbidden Features

The following are forbidden from confirmatory baseline or full models:

- full-window IPV as a deployed value;
- observed PET as an online feature;
- realized order after the decision point;
- post-hoc phase labels;
- future frames;
- outcome-derived ranks, official scores, or score residuals as predictors;
- any feature selected after reading predictor-outcome results;
- any scenario membership source that exists only in a score-joined table.

## Training-Fold Contract

For each fold:

1. Split by the frozen fold contract before preprocessing.
2. Fit imputers, scalers, encoders, fixed effects, and model hyperparameters using training data only.
3. Apply fitted transforms to held-out data.
4. Use identical preprocessing and tuning for baseline and full models.
5. Save the train/test row identifiers and fold metadata before model fitting.

## Tuning Budget

- Tuning is allowed only within training data.
- Baseline and full models receive the same hyperparameter grid, same validation protocol, and same random seed.
- No extra tuning pass may be granted to the full model or to the baseline.
- If sample size is too small for a flexible model, fall back to the simplest prespecified regularized linear/rank model for both arms.

## Negative Controls

Negative controls are not confirmatory but should be run after the primary comparison:

- IPV removed;
- IPV time-shuffled inside training folds;
- state-shuffled/wrong-envelope cell;
- counterpart swap;
- role flip;
- sign flip;
- kinematics-only baseline;
- future-leaky full-window IPV as an optimistic upper bound only.

All negative controls are discovery-family or red-team checks and must be labelled as such.
