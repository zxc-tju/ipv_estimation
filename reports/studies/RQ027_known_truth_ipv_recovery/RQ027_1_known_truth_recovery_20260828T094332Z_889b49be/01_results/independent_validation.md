# RQ027 Independent Validation

Validation status: `PASS`
Recomputed verdict: `PILOT_NO_GO`

## Run and data health

- Run rows: 288 = 240 interactive + 48 negative controls.
- Frame rows: 3,456 = 288 runs × 12 attempted target-side frames.
- Engineering failures: 0/288; duplicate run IDs: 0/288; non-finite primary frames: 0/3,456.
- Collision-tagged runs: 2/288; exact IDs are recorded in `independent_validation.json`.

## Recovery

- Persistent opportunity-aware coverage: 215/240 = 0.895833.
- Accepted-run MAE: 0.553907 rad; zero predictor MAE: 0.553432 rad.
- Spearman(true, estimate): 0.214511.
- Sign accuracy for nonzero truth: 106/173 = 0.612717.

## Concentration and negative controls

- q_eff vs absolute error Spearman: -0.124207; the sign is opposite the intended selective-risk relation.
- Fixed max-weight policy passes 2476/2880 frames = 0.859722; pass MAE 0.597503 rad vs all-frame MAE 0.586513 rad.
- Negative-control persistent false accept: 35/48 = 0.729167.

## Verdict

`PILOT_NO_GO` is independently reproduced. The result is a scientific failure of the proposed recovery/concentration contract, not an execution failure. Full S2/confirmatory expansion must remain stopped.
