# K2R_B Phase 1 Independent Notes

Created before reading `board/K2-leader-kickoff.md`, `board/reports/K1b_memory_pilot.md`, or `board/commander_notes.md`.

## Scope

InterHub full-corpus solve scope is `4,981,984` canonical solve units, from `.codex-fleet/rq015k-fullcorpus-gate/work/k1_t1_t6_local_analysis.json`, `per_artifact[artifact=interhub_sigma01_hw4_timeseries].canonical_solve_units_charged_to_this_artifact`.

The four-ledger attempted-row total is `13,980,600`, but K2 should not treat that as solve units. RQ009 has `8,994,736` attempted rows and `0` new solve rows; K1 records `joined_rows_to_sigma01=8,994,736`, `attempted_rows=8,994,736`, and `new_rows_not_in_sigma01=0` in `per_artifact[artifact=rq009_feature_matrix]`.

OnSite and WOD add `2,974 + 906 = 3,880` attempted rows, but K1 found no current materializer that emits the required per-candidate MSE/log fields. They must be either out of the InterHub-only K2 scope or represented as explicitly non-scientific engineering rows after supervisor approval. They must not be recoded as `NO_IPV_EFFECT` or `NEAR_UNIFORM`.

## Resource Recommendation

K1b single-PKL pilot raw fields:

- P6: `workers=6`, `interhub_rows=1120`, `driver_elapsed_seconds=390.383811712265`, worker peak RSS sum `16,728.84765625 MB`.
- P10: `workers=10`, `interhub_rows=1120`, `driver_elapsed_seconds=248.7333538532257`, worker peak RSS sum `27,881.87890625 MB`.
- P16: `workers=16`, `interhub_rows=1120`, `driver_elapsed_seconds=173.2081036567688`, worker peak RSS sum `44,629.078125 MB`.

Throughput per allocated core is best at P6:

- P6: `1120 / 390.383811712265 / 6 = 0.4781619 rows/s/core`.
- P10: `1120 / 248.7333538532257 / 10 = 0.4502814 rows/s/core`.
- P16: `1120 / 173.2081036567688 / 16 = 0.4041381 rows/s/core`.

Recommended first K2 shape: `665` single-PKL row-range shards, `6` workers per shard, `--mem=32G`, Slurm array concurrency `%665`.

Reasoning:

- `665 * 6 = 3,990` cores, just under the QOS name `cpu-4000_core-l40-16_card-a800-16_card` from live `sacctmgr show assoc`.
- `ceil(4,981,984 / 665) = 7,492` units per shard on average.
- Pure driver-scaled one-wave wall is `(4,981,984 / (1120 / 390.383811712265)) / 665 = 0.7254 h`.
- Total core-hours are `(4,981,984 / (1120 / 390.383811712265) / 3600) * 6 = 2,894.18 core-hours`.
- Use an operational wall budget of about `1.0 h` to absorb Slurm dispatch, PKL cold-load imbalance, and tail shards.

Memory rule: sum per-worker peak RSS for the selected worker count, multiply by `1.30`, then round up to the next common Slurm memory bin. For P6 this is `16,728.84765625 MB * 1.30 = 21,747.50 MB`; `--mem=32G` is the rounded request. The K1b target PKL was `waymo_0-299.pkl`, `310,197,719 bytes`, the largest PKL in `hpc_frozen_pkl_listing.tsv`, so the margin is not based on a small-file pilot.

Live cluster constraint: if all CPU partitions shown by `sinfo` are allowed, QOS is the binding constraint. Live `sinfo` showed `amd` `19,088` idle CPU cores, `intel` `2,517` idle CPU cores, and `fata` `321` idle CPU cores, total `21,926`, above the `3,990` cores used by the P6 plan and above the QOS cap. Memory is not binding: the `amd` idle line alone reports `52` idle nodes with `192` CPUs and at least `682,931 MB` free per node, enough for the `32G` shard request under the QOS cap. Single-node memory is not binding because `32G` is far below `644,000 MB` on intel, `772,000 MB` on amd, and `3,094,000 MB` on fata.

Worst case under this configuration: the array queues or stretches because the scheduler will not place `%665` small CPU jobs, or concurrent PKL reads create storage tail latency. This should increase wall time, not corrupt results, if shard manifests and atomic completion are enforced. OOM is unlikely with `32G`, but any OOM must trigger a bounded retry and not a silent partial ledger.

## Sharding and Idempotence

Shard key should be single PKL plus deterministic row-key range. The canonical key is `scene_unique_id | frame_index | measurement_role | candidate_grid_id`, with `pkl_file` included in the shard manifest so no shard crosses a PKL. A preflight manifest must list every canonical key exactly once, sort keys deterministically, assign each key to one `shard_id`, and record input ledger SHA, PKL SHA, code SHA, command, `sigma=0.1`, `candidate_grid_id=legacy7_pi_over_8`, `K=7`, expected row count, and created UTC.

A shard is complete only when all of these are true: final parquet exists; final manifest exists; temp files are absent; manifest input SHA/code SHA/command/sigma/grid/K match the launch manifest; output row count equals expected row count; canonical key count equals expected count; canonical keys are unique; output SHA matches manifest; status/reason counts are present; validator result is PASS. File existence alone is not completion.

Retry classes:

- `SCHEMA_MISMATCH`, missing input, bad SHA, duplicate key, missing key: retry `0`; stop the full wave.
- `OOM`: retry once with the same row range and lower workers or higher memory; second OOM stops the wave.
- `TIMEOUT`: retry once after splitting the row range; second timeout stops the wave.
- `SOLVER_FAILURE`: retry failed rows once; if still failing, write engineering-failure rows, but stop the wave if a shard exceeds `100` failed rows or `2.0%`, whichever is smaller.
- `NON_FINITE_INPUT`: do not blind retry; classify rows. Stop if any shard exceeds `0.1%`, because that indicates input corruption rather than isolated bad rows.

## Acceptance

This is a census. Success is not a sampling estimate.

Required census checks:

- InterHub output covers exactly `4,981,984` canonical solve units from `k1_t1_t6_local_analysis.json`, with duplicate canonical keys `0` and missing canonical keys `0`.
- RQ009 join output covers exactly `8,994,736` rows from `rq009_feature_matrix`, with exact-one join for each row and new solve rows `0`.
- `held_out_parsed_rows=0` remains true for K2 preflight and final manifest.
- Every scientific row has `K=7`, `candidate_grid_id=legacy7_pi_over_8`, arrays of length `7`, finite `mse_per_candidate`, `sum(w_log)` within tolerance of `1`, `max_w_log in [1/7, 1]`, and `k_eff_log in [1, 7]`.
- Reason order is exact: engineering failure first; then `mse_spread == 0` gives `ABSTAIN/NO_IPV_EFFECT`; then `max_w_log < 0.20` gives `ABSTAIN/NEAR_UNIFORM`; otherwise `OK`.
- `ipv_log` is null for non-OK statuses and finite for OK.
- K1/K1b overlap rows match previous `mse_per_candidate[7]` and `w_log[7]` exactly where overlap exists.
- Summary counts reconcile to shard counts and to source ledger `attempt_status` counts.

The J1 interval must not be a K2 success criterion. J1 is a design-weighted estimate over a `2,646,058` HT denominator, `2,300` anchors, `1,909` clusters, `B=2,000`, seed `20260731`. K2 is a row-level InterHub census with `4,981,984` canonical solve units plus RQ009 join rows. The denominators and domains are not the same. A large discrepancy can trigger investigation, but passing or failing K2 cannot be decided by whether the K2 row-level census falls inside J1's interval.

## Delivery

Deliver three durable artifacts:

1. `interhub_gate_ledger.parquet`: one row per InterHub canonical solve unit, with canonical key columns, original `product_row_key`, PKL/source fields, `candidate_ipv[7]`, `mse_per_candidate[7]`, `log_score[7]`, `w_log[7]`, `max_w_log`, `mse_spread`, `k_eff_log`, `status`, `reason_code`, `ipv_log`, `K`, `candidate_grid_id`, `sigma`, and provenance fields.
2. `rq009_gate_join.parquet`: one row per RQ009 feature-matrix row, preserving RQ009 row keys and adding the joined InterHub gate fields plus `context_cell_key`.
3. `manifest_and_summary/`: shard manifests, checksum manifest, status/reason/source/context summaries, validator report, overlap report, and command/environment record.

Mandatory warning to ship with the ledger: `ipv_log == 0` is a valid OK point value. RQ009 must use `status` and `reason_code` to distinguish OK from ABSTAIN; it must never infer ABSTAIN, missingness, or imputation need from the numeric value `0`. RQ009 rows are joined projections of InterHub solve rows, not independent solves, so RQ009 denominators must not be mixed with InterHub canonical solve denominators or J1 HT denominators.
