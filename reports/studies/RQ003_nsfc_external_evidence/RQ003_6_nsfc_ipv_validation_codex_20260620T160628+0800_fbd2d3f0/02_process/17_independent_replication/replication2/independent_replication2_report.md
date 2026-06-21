# Independent Replication v2 Report

Worker: `RQ003_phase8_replication2_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Status: `REPRODUCED_WITH_MINOR_DIFFS`

## Scope

This replication rebuilt the corrected primary confirmatory analysis from the
official corrected scenario crosswalk, cell-level directional IPV table,
baseline cell features, support coverage, and official coordination outcomes.
It did not import Phase4/Phase7 scripts.

## Primary Inclusion

- Mapped cells: cell IDs present in crosswalk, IPV, baseline, support, and score mapping.
- High support: `high_support_primary == True`.
- Corrected labels: `scenario = official_scenario` from `scenario_crosswalk_corrected.csv`.
- Non-A1: corrected scenario not equal to `A1`.
- Collision-free proxy available in the outcome table: official `safety == 100`.

Primary N: `53` cells, `9` teams, `14` corrected scenarios.

## Reimplemented Formula

- Outcome residualization: fold-local OLS `coordination ~ scenario + area`.
- Baseline model: 18 baseline feature columns from `baseline_features_cells.csv`.
- Full model: baseline feature block plus `D_comp_auc` and `D_yield_auc`.
- Model: standardized ridge regression fitted from scratch with numpy.
- CV-R2 convention: `1 - SSE / sum(y_residual^2)`, matching the corrected output convention.

## Results

| mode | Delta Spearman | Delta CV-R2 | N | Direction |
|---|---:|---:|---:|---|
| Corrected Phase7 | 0.136832768908241 | 0.086248958495336 | 53 | favorable |
| Independent training-tuned | 0.054668601838413 | 0.068097410091724 | 53 | favorable / favorable |
| Reported-alpha refit check | 0.136832768908241 | 0.086248958495335 | 53 | favorable / favorable |

## Agreement

- Reported-alpha refit max base prediction absolute difference:
  `5.617e-13`.
- Reported-alpha refit max full prediction absolute difference:
  `5.639e-13`.
- Reported-alpha refit metric tolerance pass:
  `True`.
- Independent training-tuned direction reproduced:
  `True`.

## Interpretation

The corrected favorable direction reproduces. The formula/data path is verified
by exact per-cell agreement when refitting with the fold-specific alphas reported
in the corrected result file. A fully independent nested-LOOCV alpha selector is
also favorable, but its Spearman delta is smaller because the frozen contract did
not specify the exact inner-CV splitter or alpha-selection convention used by
Phase7.
