# K1 preflight and resource plan for the log-domain IPV gate

## 0. Position and main conclusion

This work supports online verification: before comparing an autonomous-driving vehicle with human IPV distributions, the pipeline must first decide whether the current frame's IPV value carries discriminating information among the 7 candidate IPV values. IPV means Interaction Preference Value, a scalar parameter for social interaction tendency. If the 7 candidate trajectories receive nearly uniform log-domain weights, the IPV point value is not informative for the first gate and should be recorded as an abstention, not as `ipv = 0`.

The overall research state is: the RQ009 human-envelope gate is already accepted; the first gate specification is already frozen by track J; the existing RQ015A L1 ledgers have 14,473,982 rows but do not contain `mse_per_candidate[7]`, `log_score[7]`, or `w_log[7]`. K1 is only a preflight and resource plan for materializing those missing fields. K1 did not submit a full-scale job.

本报告中“全语料”只指「RQ015A 当前 4 份本地可审计 L1 parquet ledger 的全行」，不指 WOD 全量或项目全量。

Main conclusion: do not launch a 4-ledger K2 run as if all four artifacts are already equally rebuildable. InterHub can be recomputed from the frozen HPC PKL snapshot, and all `rq009_feature_matrix` rows join back to InterHub solved units. OnSite and WOD currently have only legacy IPV/error or anchor/candidate tables in the checked local/HPC paths; K2 needs either a new materializer for those two artifacts or an explicit engineering-failure/pass-through decision before full 4-ledger materialization.

Terminology used below:

- `product_row_key`: the stable row key string in the current L1 ledger, encoded as escaped `field=value` parts.
- `context_cell_key`: the RQ009 context cell used downstream; in the checked RQ009 feature matrix it can be reconstructed from `source_dataset`, `geometry_path_category`, and `priority_role`.
- HT weight: Horvitz-Thompson design weight used by J1 for sample-to-domain estimates. K1 did not recompute J1's HT estimates.

## 1. T1 canonical solve units

Source files and columns: `.codex-fleet/rq015k-fullcorpus-gate/work/k1_t1_t6_local_analysis.json` and `t1_solve_units_by_artifact.csv`, derived from the four RQ015A L1 parquet ledgers using columns `artifact_id`, `product_row_key`, `measurement_role`, `attempt_status`, `candidate_grid_id`, and first-pass `rq007_split` counts. `held_out_parsed_rows = 0`.

Canonical key definition:

| Artifact | Canonical solve key |
|---|---|
| `interhub_sigma01_hw4_timeseries` | `scene_unique_id | frame_index | measurement_role | candidate_grid_id` |
| `rq009_feature_matrix` | maps to InterHub as `case_key | anchor_frame_index | mapped sigma01 agent role | candidate_grid_id` |
| `onsite_dense_timeseries` | `case_key | frame_index | timestamp_ms | measurement_role | candidate_grid_id` |
| `wod_rq010b_full479_audited` | `segment_key | candidate_index | measurement_role | candidate_grid_id` |

RQ009 role mapping: `key_agent_1/target_future -> agent_1`, `key_agent_1/counterpart_current -> agent_2`, `key_agent_2/target_future -> agent_2`, `key_agent_2/counterpart_current -> agent_1`.

| Artifact | L1 rows | `ATTEMPTED` rows | canonical solve units charged to K2 | RQ009 rows joined to InterHub | RQ009 new rows |
|---|---:|---:|---:|---:|---:|
| `interhub_sigma01_hw4_timeseries` | 5,197,072 | 4,981,984 | 4,981,984 | n/a | n/a |
| `onsite_dense_timeseries` | 281,268 | 2,974 | 2,974 | n/a | n/a |
| `rq009_feature_matrix` | 8,994,736 | 8,994,736 | 0 | 8,994,736 | 0 |
| `wod_rq010b_full479_audited` | 906 | 906 | 906 | n/a | n/a |
| Total | 14,473,982 | 13,980,600 | 4,985,864 | 8,994,736 | 0 |

RQ009 join rate is 8,994,736 / 8,994,736 = 100.00%, with numerator and denominator both from `rq009_feature_matrix` `ATTEMPTED` rows in `k1_t1_t6_local_analysis.json`, fields `joined_rows_to_sigma01` and `attempted_rows`.

Compression ratio: 13,980,600 applicable L1 rows / 4,985,864 canonical solve units = 2.8040. Applying only this compression to the G-track 35 / 49 / 61 day extrapolations changes them to:

| Time basis | Original G extrapolation for 13,980,600 rows | K1 compressed extrapolation for 4,985,864 units |
|---|---:|---:|
| Solve loop | 35 days | 12.48 days |
| Driver | 49 days | 17.47 days |
| Slurm wall | 61 days | 21.75 days |

These compressed numbers use the G-track 2,300-anchor timing and only change the denominator. K1 pilot timing below is a separate, more conservative cold-load measurement.

## 2. T2 HPC-side artifact search

Search scope: the requested HPC roots under `/share/home/u25310231/ZXC/`, recorded in `.codex-fleet/rq015k-fullcorpus-gate/work/hpc_t2_filename_hits.tsv`. The search matched file names with `*mse*`, `*log_score*`, `*w_log*`, `*candidate*`, and `*anchor*`. It was read-only. RQ014 filename hits were not opened for table contents.

Selected headers are recorded in `hpc_t2_selected_headers.tsv`; they include RQ010B WOD candidate outputs, the WOD projection used by RQ015A, and an RQ012B OnSite anchor table. Those selected headers do not contain `mse_per_candidate`, `w_log`, or `log_score`. The only confirmed `mse_per_candidate[7]` / `w_log[7]` tables on HPC are repeated G-track 2,300-row sample files under `rq015g_anchor_resolve_*`, including the known `anchor_mse_hpc.csv`.

Conclusion: no full or large-batch HPC table with per-candidate MSE/log weights was found in K1. This does not change the K2 cost structure.

## 3. T3 rebuild entry by artifact

| Artifact | Current rebuild status | Evidence |
|---|---|---|
| InterHub sigma01 | Rebuild entry exists | Frozen HPC PKL root has 15 PKL files, total 1,856,362,564 bytes; pilot input scan saw 4,981,984 development/guard post-warm anchors and `remote_pkl_missing_rows = 0`; source split count read only `scene_unique_id` before filtering, with `held_out_parsed_rows = 0`. |
| RQ009 feature matrix | Reuse InterHub solves | T1 joined 8,994,736 / 8,994,736 RQ009 applicable rows to InterHub units; no new solve units. |
| OnSite dense | No current K-compatible materializer | Local L1 source has kinematics plus legacy IPV/error columns; HPC selected OnSite anchor header has context and legacy current/future IPV/error columns but no per-candidate MSE/log fields. Pilot classified all 2,974 applicable OnSite rows as contract-only `SCHEMA_MISMATCH`. |
| WOD full479 | No current K-compatible materializer | The RQ015A projection has only `segment_key,candidate_index,ego_ipv,ego_ipv_error`; checked RQ010B candidate headers have candidate IPV/spread style fields but no `mse_per_candidate[7]`, `log_score[7]`, or `w_log[7]`. Pilot classified all 906 applicable WOD rows as contract-only `SCHEMA_MISMATCH`. |

The OnSite/WOD result is a scope conclusion, not a compute failure.

## 4. T4 stratified pilot

Pilot input source: `.codex-fleet/rq015k-fullcorpus-gate/work/pilot_input_summary.json`, `pilot_units.csv`, and `pilot_strata_counts.csv`.

Pilot design:

- Total units: 5,000.
- InterHub actual solves: 1,120 units, stratified by `source`, `measurement_role`, `n_obs` band, and PKL file.
- OnSite contract-check units: 2,974 / 2,974 applicable rows.
- WOD contract-check units: 906 / 906 applicable rows.
- G-track anchor overlap seeded for validation: 24 known anchors were forced into the pilot; result overlap after de-duplication was 27 rows.

InterHub source coverage in the 1,120 solve units:

| Source | Units |
|---|---:|
| av2 | 76 |
| lyft | 76 |
| nuplan | 680 |
| waymo | 288 |

HPC submission:

- Work dir: `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_pilot_20260802T090801Z`.
- Job id: `2068610`.
- Partition/node: `fata`, node `fata02`.
- Workers: 6.
- Job name: `zxc-rq015k-pilot`.
- Threads: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
- Requested memory: `--mem=160G`.

Pilot timing:

| Basis | Value | Source |
|---|---:|---|
| Solve loop | 499.6056067943573 s | `k1_pilot_summary.json`, field `solve_loop_elapsed_seconds` |
| Driver | 499.7747611999512 s | `k1_pilot_summary.json`, field `driver_elapsed_seconds` |
| Slurm wall | 518 s (`00:08:38`) | `k1_pilot_sacct_final.txt`, job `2068610` |

Pilot InterHub solve success rate: 1,120 / 1,120 = 100.00%, source `k1_pilot_summary.json`, fields `interhub_rows` and `failure_counts_interhub.OK`.

Contract-only failure rate in the 5,000-unit pilot: 3,880 / 5,000 = 77.60%, source `k1_pilot_contract_outcomes.csv` rows and `pilot_units.csv` rows, filter `pilot_action != SOLVE`, failure type `SCHEMA_MISMATCH`. This percentage is not a scientific abstention rate; it records missing materializer scope for OnSite/WOD in K1.

G-anchor acceptance check: 27 overlapping rows had max absolute difference 0.0 for `mse_per_candidate[7]` or `w_log[7]`, source `k1_pilot_summary.json`, field `g_anchor_overlap`.

Pilot timing per InterHub solve unit, source `k1_pilot_interhub_results.csv`, column `elapsed_seconds`, all 1,120 rows:

| Statistic | Seconds |
|---|---:|
| mean | 2.663829145686967 |
| P50 | 0.852176308631897 |
| P90 | 2.2621901750564577 |
| P99 | 52.0956428027153 |
| max | 72.83289957046509 |

Peak worker RSS:

- Min / median / max per worker: 15,480.9766 / 15,526.9590 / 15,574.5586 MB.
- PKL disk size seen per worker: 1,770.3653 MB.
- RSS / seen-PKL-disk amplification min / median / max: 8.7445 / 8.7705 / 8.7974.
- Source: `k1_pilot_summary.json`, field `worker_memory`.

Slurm memory recommendation from K1: keep `--mem=160G` for any shard that may allow each of 6 workers to load all 15 PKLs. The measured 6-worker peak RSS sum is about 93.2 GB; adding a 30% margin gives about 121.2 GB, and `160G` matches the successful pilot while leaving scheduler/accounting headroom. A later PKL-scoped shard design may reduce memory, but that should be proven by another authorized pilot before lowering `--mem`.

## 5. T5 resource plan and failure recovery rules

Recommended K2 compute lane for InterHub/RQ009:

- Partition: `fata` primary, `intel` acceptable fallback because G-track already established bitwise agreement for this computation across AMD and Intel.
- Nodes: one node per shard.
- Workers per node: 6.
- Memory: `--mem=160G` until a PKL-scoped memory pilot proves a lower value.
- Shard size: 50,000 canonical InterHub solve units per shard, fixed by `(pkl_file, source, row-key range)`. This yields about 100 InterHub solve shards for 4,981,984 units.
- RQ009: no solver shard; it is a join/materialization phase from InterHub outputs using the T1 mapping.

Resource budget:

| Basis | Wall days for 4,985,864 units | Worker-hours at 6 workers |
|---|---:|---:|
| G compressed solve-loop basis | 12.48 | 1,797.40 |
| G compressed driver basis | 17.47 | 2,516.36 |
| G compressed Slurm basis | 21.75 | 3,132.61 |
| K1 pilot conservative solve-loop basis | 25.74 | 3,706.79 |
| K1 pilot conservative driver basis | 25.75 | 3,708.05 |
| K1 pilot conservative Slurm basis | 26.69 | 3,843.27 |

Operational recommendation: budget by the conservative K1 pilot basis for scheduling and cluster reservation, but design shards by PKL and row-key range so cold PKL load is amortized and retry scope is small. Do not use the optimistic compressed G number as the only resource commitment.

Concrete shard manifest rule:

Each shard manifest must include `shard_id`, `artifact_scope`, `pkl_file_list`, `source_dataset`, `row_key_min`, `row_key_max`, `canonical_key_count`, `expected_output_rows`, `input_ledger_sha256`, `input_pkl_sha256`, `code_sha`, `command`, `sigma`, `candidate_grid_id`, and `created_utc`. The same canonical key must appear in exactly one shard. RQ009 rows must reference InterHub solve outputs and must not trigger duplicate solves.

Concrete checkpoint rule:

Each shard writes `<shard>.tmp.parquet` and `<shard>.tmp.manifest.json`; after validation, the writer atomically renames them to final names. A shard is considered complete only if the final manifest matches input SHA, code SHA, command, canonical row count, expected output row count, and output SHA. File existence alone is not completion. Resubmission must be idempotent: a matching completed manifest is skipped; a non-matching final file is a hard `SCHEMA_MISMATCH` stop.

Concrete retry rule:

Failure types must include `OOM`, `TIMEOUT`, `SOLVER_FAILURE`, `NON_FINITE_INPUT`, and `SCHEMA_MISMATCH`.

- `SCHEMA_MISMATCH`: stop the whole K2 wave immediately.
- `OOM`: retry the failed shard once with 3 workers and unchanged input; if it OOMs again, stop and report.
- `TIMEOUT`: retry the failed shard once with doubled wall time or half-sized row range; if it times out again, stop and report.
- `SOLVER_FAILURE`: retry only failed rows once; stop if a shard has more than 100 failed rows or more than 2.0% failed rows, whichever is smaller.
- `NON_FINITE_INPUT`: do not retry blindly; classify rows and stop if the rate exceeds 0.1% within any shard.

Concrete product validation rule:

Before accepting K2 output, validate: primary key uniqueness; no missing shard ranges; no duplicate shard ranges; `K=7`; arrays length exactly 7; finite `mse_per_candidate[7]` before scientific reason assignment; engineering failure rows have null `ipv_log` and null `w_log[7]`; reason assignment order is exact-zero MSE spread before near-uniform; `ipv_log` null only where required; state counts reconcile to source `attempt_status`; random and G-anchor overlap reruns match; all input and PKL SHA lists match; `held_out_parsed_rows = 0`.

## 6. T6 RQ009 join dry run

Source: `.codex-fleet/rq015k-fullcorpus-gate/work/k1_t1_t6_local_analysis.json`, field `t6_rq009_context_join_dry_run`.

Dry run sample: 1,024 row-level RQ009 ledger rows, yielding 512 distinct feature keys because each feature key has two measurement roles. The join back to the RQ009 feature matrix returned 512 / 512 exact-one matches, 0 misses, and 0 duplicate keys. The selected feature-matrix files count was 138. The parsed key fields were `case_key`, `anchor_frame_index`, `perspective`, and `source_dataset`; measurement role came from the ledger column.

Go/no-go implication: RQ009 join is not blocking K2. A K2 materializer can reconstruct `context_cell_key` by joining from the parsed key fields into the feature matrix and then forming `source_dataset|geometry_path_category|priority_role`.

## 7. Appendix: RQ009 counterpart IPV fill behavior

The RQ009 OOD gate uses `counterpart_ipv_current`, `counterpart_ipv_error_current`, and `counterpart_ipv_slope_pre_anchor`. The already-established rule is: `build_features.py` lines 774-776 directly use upstream legacy IPV / error values; slope is computed by `theil_sen_slope()`, which returns NaN when fewer than 2 valid historical points exist; later calibration absorbs this through median imputation in `Preprocessor` and the gate numeric preprocessing. K1 did not re-quantify this and did not recompute any RQ009 abstention rate.

## 8. Decisions pending supervisor approval

### Decision 1: OnSite/WOD handling in K2

Option A: build new K-compatible materializers for OnSite and WOD, then run full 4-ledger K2.

Basis: K1 found no current OnSite/WOD path that emits `mse_per_candidate[7]`, `log_score[7]`, or `w_log[7]`. This is the cleanest path if K2 must produce row-level gate fields for every applicable row in the current four ledgers.

Consequence if not done: OnSite 2,974 rows and WOD 906 rows cannot receive the same scientific gate fields as InterHub/RQ009.

Option B: explicitly approve engineering-failure/pass-through rows for OnSite/WOD in K2.

Basis: K2 contract already requires engineering failures to remain separate from scientific reasons. OnSite/WOD are small in count but cannot be silently recoded as `NO_IPV_EFFECT` or `NEAR_UNIFORM`.

Consequence if not done: a full 4-ledger output would either be incomplete or would mix missing rebuild scope into scientific abstention reasons.

Recommendation: choose A if those 3,880 rows must carry full log-domain gate fields; choose B only if supervisor accepts that K2's first materialization is InterHub/RQ009-complete but OnSite/WOD are explicitly non-scientific engineering rows.

### Decision 2: Resource basis for K2 scheduling

Option A: schedule by the conservative K1 pilot basis, about 26.69 Slurm days at 6 workers if run serially.

Basis: K1 pilot covered all 15 InterHub PKLs and measured 15.6 GB peak RSS per worker.

Consequence if not done: using only the G compressed basis may under-reserve time/memory for full-source cold-load behavior.

Option B: first authorize a smaller PKL-scoped resource pilot before K2 to lower `--mem` or time budgets.

Basis: current pilot intentionally let each worker see all 15 PKLs; a PKL-scoped shard should lower memory, but K1 did not prove that with Slurm.

Consequence if not done: K2 should retain `--mem=160G`.

Recommendation: for K2 launch planning, keep `--mem=160G`, 6 workers, and 50,000-unit shards. Lower memory only after another authorized pilot.

## 9. K2 output contract

### 9.1 Gate applicability

Only rows with `attempt_status == ATTEMPTED` receive this gate. That is 13,980,600 rows, source current four L1 parquet ledgers, column `attempt_status`.

The 219,360 `NOT_ATTEMPTED` rows and 274,022 `UNKNOWN` rows must write `gate_applicable = false`, all gate fields null, and preserve `source_attempt_status` plus `source_reason_code`.

Never write upstream missing input as `NO_IPV_EFFECT` or `NEAR_UNIFORM`.

### 9.2 Engineering failures

`OK` and `ABSTAIN` are not enough. Reuse existing mutually exclusive engineering statuses such as `NON_FINITE_INPUT` and `SOLVER_FAILURE`. Engineering-failure rows must have `ipv_log = null` and `w_log[7]` all null, and must not be counted under the two scientific reasons.

### 9.3 Sigma and log score

Use:

```text
log_score_i = -mse_i / (2 * sigma^2), sigma = 0.1
```

The candidate grid is `legacy7_pi_over_8`. Output must persist `log_score[7]`, or record sigma and formula so `w_log[7]` is reproducible.

### 9.4 Exact MSE spread

`mse_spread == 0` is exact float64 equality after asserting seven finite float64 MSE values. Do not replace it with `np.isclose`. Non-finite rows use engineering status.

### 9.5 Softmax implementation

Use the existing stable `weights_from_mse()` implementation: subtract max log weight, exponentiate, and normalize. Do not reimplement this separately.

### 9.6 Ordered mutually exclusive reason assignment

Assign `NO_IPV_EFFECT` first for exact-zero MSE spread. Then assign `NEAR_UNIFORM` only where reason is still null and `max_w_log < 0.20`. Do not let two boolean masks overwrite each other in the wrong order.

### 9.7 Frozen gate spec

```text
Input: frame_id, candidate_grid_id, K=7, candidate_ipv[7],
       mse_per_candidate[7], log_score[7], context_cell_key

w_log      = softmax(log_score) using log-sum-exp
mse_spread = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log  = max(w_log)
k_eff_log  = 1 / sum(w_log_i^2)

if mse_spread == 0:      status=ABSTAIN, reason_code=NO_IPV_EFFECT,  ipv_log=null
elif max_w_log < 0.20:   status=ABSTAIN, reason_code=NEAR_UNIFORM,   ipv_log=null
else:                    status=OK,      reason_code=null,
                         ipv_log = sum(candidate_ipv_i * w_log_i)
```

The computation must be log-domain. `theta = 0.20` is a policy threshold and must not be tuned or scanned in K2. The two reasons are ordered and mutually exclusive. When status is `ABSTAIN`, `ipv_log` must be null, not 0, NaN, or missing.

### 9.8 Materialized columns and schema decision

K2 should emit two parquet products:

1. L1 row-level gate ledger, partitioned by `artifact_id` and `shard_id`.
2. Aggregation table, partitioned by `artifact_id` and aggregation/context keys.

L1 schema version: `rq015k_logdomain_gate_l1_v1`.

Array columns should be Arrow fixed-size lists of length 7 where supported: `mse_per_candidate`, `log_score`, `w_log`, and `candidate_ipv`. If fixed-size list support blocks a writer, use scalar columns `mse_0`...`mse_6`, `log_score_0`...`log_score_6`, `w_log_0`...`w_log_6`, `candidate_ipv_0`...`candidate_ipv_6`, but the schema must declare candidate order as `[-3,-2,-1,0,1,2,3] * pi/8`.

Required L1 columns: `artifact_id`, canonical row key, original `product_row_key`, `measurement_role`, `case_id`, `rq007_split`, `frame_id`, `context_cell_key`, `candidate_grid_id`, `K`, `candidate_ipv[7]`, `mse_per_candidate[7]`, `log_score[7]`, `w_log[7]`, `max_w_log`, `mse_spread`, `k_eff_log`, `status`, `reason_code`, `ipv_log`, `gate_applicable`, `source_attempt_status`, `source_reason_code`, legacy `ipv_error`, legacy `k_eff`, legacy `q_eff`, `solver_status`, `failure_type`, `shard_id`, `input_sha256`, `code_sha`, `created_utc`.

Nullability: arrays and gate numeric fields are non-null only when `gate_applicable = true` and status is scientific `OK` or `ABSTAIN`; `ipv_log` is non-null only for `status = OK`; all gate fields are null for `NOT_ATTEMPTED`, `UNKNOWN`, and engineering failure rows except status/failure metadata.

`gate_pass_rate` belongs in the aggregation table, not repeated into every L1 row.

### 9.9 Downstream warning

The interface documentation must warn downstream users: `status` and `reason_code` are the only valid discriminators. `ipv_log = 0` is a legal and frequent `OK` value. J1 reported that among `status=OK` rows, 23.40% had `|ipv_log| <= 1e-9`; numerator/denominator details are in the J1 report, not recomputed by K1. Therefore downstream code must not treat numeric zero as abstention.

## 10. K2 go / no-go judgment

Judgment: recommend not launching full 4-ledger K2 until Decision 1 is resolved. Recommend launching InterHub/RQ009 K2 only after supervisor explicitly accepts that OnSite/WOD are either handled by new materializers or represented as separate engineering rows.

Basis:

- T1 removed duplicate RQ009 solves: 8,994,736 / 8,994,736 RQ009 applicable rows join to InterHub, and 0 rows are new.
- T6 join dry run passed: 512 / 512 sampled feature keys had exact-one matches to RQ009 context inputs.
- T4 InterHub pilot succeeded: 1,120 / 1,120 InterHub solve rows OK, and 27 G-anchor overlap rows matched exactly.
- T3 found no current K-compatible OnSite/WOD materializer.

If K2 is launched for InterHub/RQ009 after that decision:

- Use `fata`, 1 node per shard, 6 workers per node, `--mem=160G`, 50,000 canonical units per shard.
- Budget by conservative K1 pilot: 25.74 / 25.75 / 26.69 serial wall days for solve-loop / driver / Slurm basis, or 3,706.79 / 3,708.05 / 3,843.27 worker-hours.
- Use the T5 manifest, retry, and validation rules above.

K1 stops here. No full-scale job was submitted, no protected estimator files were modified, and no git commit was made.
