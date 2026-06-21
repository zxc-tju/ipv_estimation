# Phase 4 Capacity-Matched Baseline Report

Worker: `RQ003_phase4_features_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Status: PARTIAL

The baseline feature table contains 150 cells and 18 manifest rows. Features are computed from state, safety, and causal kinematics over the same conflict-window/time-normalization as the IPV AUC features. The baseline table contains no IPV columns and no outcome columns; the confirmatory full model should differ only by adding `D_comp_auc` and `D_yield_auc` from `cell_level_directional_ipv.csv`.

Forbidden features excluded: full-window IPV, observed PET, realized order, post-hoc phase labels, future-frame predictors, official scores/ranks, and outcome-tuned thresholds.
