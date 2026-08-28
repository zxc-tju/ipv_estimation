# RQ027 Conclusions

Verdict: `PILOT_NO_GO`

## Conclusion 1 — Engineering execution passed

`288/288` scheduled runs completed with `0/288` engineering failures, `0/288` duplicate run IDs and `0/3,456` non-finite primary frames. Independent recomputation returned `PASS`.

## Conclusion 2 — Independent point recovery did not clear feasibility

Persistent target-side readings existed for `215/240` interactive runs, but accepted-run MAE was `0.553907 rad`, slightly worse than the same-run zero predictor (`0.553432 rad`). Spearman was `0.214511`; nonzero-truth sign accuracy was `106/173=61.2717%`.

## Conclusion 3 — Candidate concentration failed as accuracy selection

Spearman(`q_eff`, absolute error) was `-0.124207`, opposite the intended direction. The fixed max-weight policy retained `2,476/2,880` frames and increased MAE from `0.586513` to `0.597503 rad`.

## Conclusion 4 — Negative controls were frequently accepted

Persistent concentration-only false acceptance was `35/48=72.9167%`; every control family exceeded `58%`.

## Required action

Stop S2 and sealed confirmatory expansion. Preserve all results; do not retune the gate and rerun within this contract. No accepted manuscript claim is created.
