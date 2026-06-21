# Phase 8 Independent Replication Report

Worker: `RQ003_phase8_replication_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Verdict: `DISCREPANT`

## Method

I reimplemented the pipeline from the frozen contracts and data contracts without importing or reading the Phase 4 worker scripts. Directional deviations were recomputed as:

- `D_comp = max(0, (Q_low - theta_ego) / w)`
- `D_yield = max(0, (theta_ego - Q_high) / w)`

Support gating used the frozen exact-cell count, estimator-error, theta-OOD, and conflict-distance thresholds. Cell features used time-normalized trapezoidal AUC over valid conflict-window timestamps. Fold assignment was materialized directly from `fold_contract.csv`.

The independent statistical model uses fold-local scenario+area fixed-effect residualization followed by a capacity-matched standardized ridge residual model with `alpha=0.1` for both baseline and full arms. This is a clean implementation of the model-capacity contract, but it does not reproduce the stored Phase 4 residual predictions.

## Tolerances

- Cell metric/support/baseline floating values: `1e-09`
- Fold and Boolean/string assignments: exact match
- CV residual predictions: `1e-06`
- Primary key statistics: `1e-06`

## Agreements

- Cell-level directional IPV, support coverage, fallback features, onset/persistence, simultaneous competition, and reciprocity mismatch reproduce within tolerance.
- Baseline kinematic/safety features reproduce within tolerance.
- Fold assignment reproduces exactly.
- Fold-local fixed-effect predictions and OOF residual targets reproduce within tolerance.

## Discrepancies

- Material CV residual prediction discrepancies remain for `pred_base_residual` and `pred_full_residual`.
- Primary Spearman and MAE effect directions remain `null_or_reverse`; the independent CV-R2 delta is favorable while Phase 4's CV-R2 delta is `null_or_reverse`.

Phase 4 primary stats:

```json
{
  "base_spearman": 0.3541033434650456,
  "full_spearman": 0.243812418584455,
  "delta_spearman": -0.1102909248805905,
  "base_mae": 7.044133658371353,
  "full_mae": 7.783041155316771,
  "delta_mae_reduction": -0.7389074969454175,
  "base_cv_r2": 0.0262230270093748,
  "full_cv_r2": -0.0218113262658128,
  "delta_cv_r2": -0.0480343532751876,
  "effect_direction_spearman": "null_or_reverse",
  "effect_direction_mae": "null_or_reverse",
  "effect_direction_cv_r2": "null_or_reverse"
}
```

Replication primary stats:

```json
{
  "base_spearman": 0.34129396439426835,
  "full_spearman": 0.32957012592270946,
  "delta_spearman": -0.01172383847155889,
  "base_mae": 6.920215991329486,
  "full_mae": 7.180724665758951,
  "delta_mae_reduction": -0.2605086744294649,
  "base_cv_r2": 0.03014824548034034,
  "full_cv_r2": 0.07989493875335296,
  "delta_cv_r2": 0.04974669327301262,
  "effect_direction_spearman": "null_or_reverse",
  "effect_direction_mae": "null_or_reverse",
  "effect_direction_cv_r2": "favorable",
  "model_spec": "independent_standardized_ridge_alpha_0.1"
}
```

## Verdict

Metric construction, support gating, conflict-window AUC, cell aggregation, fold assignment, and fixed-effect outcome residualization reproduce independently. The primary LOTO residual model does not reproduce Phase 4 per-cell predictions under the independently specified ridge implementation, so no Tier A interpretation should proceed until the Phase 4 model specification is reconciled without reading forbidden worker scripts.
