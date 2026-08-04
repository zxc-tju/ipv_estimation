# RQ009 Interface Note for K2 L1 Gate Ledger

Status: **PASS, waiting on commander**. The full row ledger was materialized and fetched locally. K2-1's old FAIL was caused by comparing the G-anchor canary to the wrong Mac baseline and by an uncalibrated solver-failure tripwire that the supervisor withdrew. K2-2 reran only the corrected G-HPC anchor check and the missing RQ009 join-key uniqueness check; both passed.

## Purpose

This ledger supplies the first gate for online verification. The gate decides whether a frame-level IPV value carries discriminating information among the seven candidate IPV values. IPV means Interaction Preference Value, a scalar parameter for social interaction tendency.

Rows with near-uniform candidate weights are abstentions: the IPV scalar is not informative for that frame.

## Storage

L1 schema version: `rq015k_logdomain_gate_l1_v1`.

Candidate order: `[-3,-2,-1,0,1,2,3] * pi/8`, with `candidate_grid_id = legacy7_pi_over_8`, `K = 7`, and `sigma = 0.1`.

The writer uses scalar array columns because the parquet writer rejected fixed-size list columns with null row values:

- `candidate_ipv_0` ... `candidate_ipv_6`
- `mse_0` ... `mse_6`
- `log_score_0` ... `log_score_6`
- `w_log_0` ... `w_log_6`

For `rq009_feature_matrix` rows, these array columns are intentionally null. RQ009 rows store `interhub_canonical_key` plus gate scalars: `status`, `reason_code`, `ipv_log`, `max_w_log`, `mse_spread`, `k_eff_log`, and `gate_applicable`.

To restore arrays for an RQ009 row, join:

```text
rq009_feature_matrix.interhub_canonical_key
  -> interhub_sigma01_hw4_timeseries.canonical_key
```

Then read the scalar array columns from the matched InterHub A-class row. Corrected validation sampled 1,000 RQ009 rows, covering 500 unique InterHub keys, and resolved all 1,000 rows to A-class array columns.

## Gate Fields

Use only these fields to determine gate state:

- `status`
- `reason_code`

Reason values:

- `NO_IPV_EFFECT`: exact-zero MSE spread, so the seven candidates have no IPV-dependent MSE difference.
- `NEAR_UNIFORM`: log-domain softmax maximum weight is below `0.20`, so the IPV scalar does not carry candidate-level discriminating information.

Engineering statuses:

- `NON_FINITE_INPUT`
- `SOLVER_FAILURE`

Rows outside this K2 materializer scope have `gate_applicable = false` and must not be counted under the two scientific reasons.

## Zero IPV Warning

`ipv_log = 0` is a legal and frequent `status = OK` value. It represents neutral social interaction tendency after the gate has passed.

**Corrected 2026-08-03 by the supervisor. Read this before using any earlier copy of this file.**
An earlier version of this section stated that `ipv_log = 0` occurs in `23.40%` of passing-gate
estimates. **That figure was wrong as a property of this product.** `23.40%` is `238/1,017` from the
J-track anchor sample, which is stratified and enriched for zero-heavy signatures; it was carried
into this interface contract as if it described the corpus. The error originated with the supervisor
and was caught by the track-L leader on 2026-08-03.

Census values, each with its denominator. Both were measured on the delivered product, and the
supervisor reproduced the RQ009-domain figures independently with `pyarrow` on the local L1 files:

| Domain | Denominator (`status = OK` rows) | `ipv_log` exactly `0` | `abs(ipv_log) <= 1e-9` |
|---|---:|---:|---:|
| InterHub canonical solve units | 3,502,340 | 175,458 = `5.0097%` | 348,539 = `9.9516%` |
| RQ009 ledger rows | 6,405,292 | 314,636 = `4.9121%` | 628,276 = `9.8087%` |

Source: `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L3b_ipvlog_zero_census.json`.

Pick the row matching your denominator. Do not quote a rate without its denominator, and do not
reuse the `23.40%` figure.

**The qualitative warning is unchanged and still binding:** zero is a legal, common passing value,
roughly one passing row in twenty is exactly zero and about one in ten is within `1e-9` of zero.
The discriminator is `status` plus `reason_code`, not the numeric value of `ipv_log`.

Downstream code must not treat numeric zero as an abstention.
