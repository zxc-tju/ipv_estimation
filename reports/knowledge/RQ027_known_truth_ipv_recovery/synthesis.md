# RQ027 Synthesis

Status: `PILOT_NO_GO / AWAITING PI ACCEPTANCE`

## Problem and stage

RQ027 tests whether the frozen IPV estimator recovers a simulation-controlled target IPV when the generator does not share its planner, search, cost implementation or likelihood, and whether candidate concentration supports selective abstention. The development-only `240 interactive + 48 negative-control` feasibility pilot is complete; no sealed test was opened.

## What the reports establish

- Engineering execution completed: `288/288` runs, `0/288` engineering failures, `0/288` duplicate IDs and `0/3,456` non-finite primary frames.
- Persistent opportunity-aware readings existed for `215/240` interactive runs.
- Accepted-run MAE was `0.553907 rad`, not better than the same-run zero predictor (`0.553432 rad`); Spearman was `0.214511`.
- `q_eff` versus absolute error Spearman was `-0.124207`; the fixed concentration policy raised frame MAE from `0.586513` to `0.597503 rad`.
- Negative-control persistent false acceptance was `35/48=72.9167%`, with every control family above `58%`.
- Independent recomputation matched the executor summary and returned `validation_status=PASS`.

## What the reports do not establish

- They do not prove that all possible IPV estimators or generators fail.
- They do not establish or refute a stable human psychological IPV.
- They do not change accepted RQ017/RQ018/RQ019/RQ021/RQ024/RQ025 claims.
- They do not support causal, external-validity, production or deployment language.

## Boundary and action

The evidence directly supports `PILOT_NO_GO` for this frozen estimator/gate under this independent-generator feasibility domain. Per the approved contract, S2, the 3,120/14,040-run expansion and sealed confirmatory testing remain stopped. No retuned rerun is authorized inside RQ027 v0.

## Manuscript-safe language

> In an independent-generator feasibility pilot, the frozen IPV estimator showed weak pooled ordering but did not improve mean absolute recovery error over a zero predictor; its candidate-concentration rule did not reduce error and frequently persisted in negative controls.

This wording remains provisional until PI acceptance; no `decision.md` exists.
