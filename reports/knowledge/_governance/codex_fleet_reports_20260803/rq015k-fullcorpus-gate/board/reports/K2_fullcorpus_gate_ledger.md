# K2 Full-Corpus Log-Domain Gate Ledger

Report timestamp: `2026-08-03T00:28:54Z`

final_status: `PASS`
state: `WAITING_ON_COMMANDER`
previous_final_status: `FAIL`
previous_blockers: `g_anchor, solver_failure_threshold`

## 1. Position And Final Status

This work supports online verification: before comparing an autonomous-driving vehicle with human IPV distributions, the pipeline needs a row-level gate that decides whether the frame-level IPV scalar carries discriminating information among seven candidate IPV values. IPV means Interaction Preference Value, a scalar parameter for social interaction tendency.

K2 materialized the requested row ledger for the current four RQ015A local-auditable L1 artifacts. K2 did not change the second RQ009 envelope gate, did not rerun RQ009 joins in K2-2, and did not change any scientific threshold.

**Final status: PASS, waiting on commander.** K2-1 originally closed as `FAIL` with blockers `g_anchor` and `solver_failure_threshold`. The supervisor ruling at `2026-08-02T19:12:54Z` found both blockers invalid: `g_anchor` compared against the wrong Mac baseline, and `solver_failure_threshold` was an uncalibrated tripwire that the supervisor withdrew. K2-2 then ran only the two missing local checks and the requested failure characterization:

- Correct G-HPC baseline anchor check: `PASS`, `anchor_rows=2300`, `compared_rows=2300`, `max_abs_diff=0.0`, `first_mismatch=null`, source `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json`.
- RQ009 join `canonical_key` uniqueness: `PASS`, `rows=8,994,736`, `unique_keys=8,994,736`, `duplicates=0`, source `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/rq009_join_key_uniqueness.json`.
- SOLVER_FAILURE characterization written to `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/solver_failure_characterization.json`; this is characterization only, not a rerun and not a FAIL criterion.

The board state is `WAITING_ON_COMMANDER`, not `DONE`.

## 2. Output Locations

- Remote authoritative work dir: `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k2_fullcorpus_finalize_20260802T175006Z/`
- Local L1 copy: `data/derived/rq015k_logdomain_gate/l1_v1/`
- Local aggregates: `data/derived/rq015k_logdomain_gate/aggregates_v1/`
- Manifest directory: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/shard_manifests/`
- Validation directory: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/`
- Interface note: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/INTERFACE_NOTE.md`
- Progress log: `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_progress.log`

Remote product size was `1,781,292,967` bytes, below the 8 GB threshold, so the full product was fetched locally. Local L1 has 510 parquet files and 510 manifests.

## 3. Scope And Row Accounting

The phrase full corpus here means only the current four RQ015A local-auditable L1 artifacts. It does not mean all WOD data or the whole project.

Machine source: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/final_validation_summary.json`.

| Class | Rows | Handling | Result |
|---|---:|---|---|
| A. InterHub solve rows | 4,981,984 | Solved on HPC | Present |
| B. RQ009 join rows | 8,994,736 | Exact-one join from A | Present |
| C. non-`ATTEMPTED` rows | 493,382 | `gate_applicable=false` | Present |
| D. OnSite/WOD `ATTEMPTED` pass-through | 3,880 | `gate_applicable=false` | Present |
| Total | 14,473,982 | L1 row count | Present |

Validation counts:

- InterHub coverage: `4,981,984 / 4,981,984` canonical keys, filter `artifact_id=interhub_sigma01_hw4_timeseries` and `gate_applicable=true`, source `final_validation_summary.json` fields `interhub_coverage.canonical_keys / expected`; missing `0`, duplicates `0`.
- RQ009 join row count: `8,994,736 / 8,994,736`, source `final_validation_summary.json` field `rq009_join.rows`; misses `0`. `new_solve_rows=0` is by construction from B-class join-only materialization, not a K2-2 measurement.
- RQ009 join key uniqueness: `8,994,736 / 8,994,736` unique `canonical_key`, duplicates `0`, filter local `l1_v1/artifact_id=rq009_feature_matrix`, source `rq009_join_key_uniqueness.json` fields `rows / unique_keys / duplicates`.
- Non-solve rows: `497,262` rows, source `final_validation_summary.json` field `non_solve_rows.rows`; includes InterHub `NOT_ATTEMPTED=215,088`, OnSite `NOT_ATTEMPTED=4,272`, `UNKNOWN=274,022`, OnSite `ATTEMPTED=2,974`, and WOD `ATTEMPTED=906`.
- Held-out evidence: `held_out_parsed_rows = 0`, source `final_validation_summary.json` field `held_out_parsed_rows`.

## 4. InterHub Scientific Counts

A-class InterHub solve-unit counts come from `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/a_class_manifest_rollup.json`, fields `a_status_counts`, `a_reason_counts`, and `a_rows`, filter `shard_id=A_*`.

| A-class result | Numerator | Denominator | Rate | Source field |
|---|---:|---:|---:|---|
| `status=OK` | 3,502,340 | 4,981,984 | 70.3001% | `a_status_counts.OK / a_rows` |
| `reason_code=NEAR_UNIFORM` | 1,457,746 | 4,981,984 | 29.2604% | `a_reason_counts.NEAR_UNIFORM / a_rows` |
| `reason_code=NO_IPV_EFFECT` | 19,964 | 4,981,984 | 0.4007% | `a_reason_counts.NO_IPV_EFFECT / a_rows` |
| `status=SOLVER_FAILURE` | 1,934 | 4,981,984 | 0.0388% | `a_status_counts.SOLVER_FAILURE / a_rows` |

J-track design-based estimate is an explanatory comparison, not an acceptance criterion. J used 2,300 anchors, HT weights, and 1,909 clusters to estimate `71.2695%`, CI `[67.1729%, 75.2135%]`, over an HT-weight denominator of `2,646,058`. The K2 InterHub census denominator is `4,981,984` solve units. The census value `70.3001%` falls inside the J interval and differs from the point estimate by `0.97` percentage points, but this must not be written as "validation passed" because the domain and denominator differ.

### 4.1 Supervisor addendum: which domain the J estimand corresponds to

Added by the supervisor at `2026-08-03T00:24:36Z`, after K2-2 had already written this report. The
underlying counts were measured independently by the supervisor by staging all 45 local RQ009 join
shards into a separate container and reading `canonical_key`, `status`, and `gate_applicable` with
`pyarrow 25.0.0`. That same independent scan is what produced `rows=8,994,736`,
`unique_keys=8,994,736`, `duplicates=0`, agreeing exactly with K2-2's measurement in section 6.

The same gate applied to the same product yields two different pass rates, because the two domains
weight solve units differently. One canonical solve unit can back several ledger rows
(compression ratio `2.804x`), and units are not referenced an equal number of times.

| Domain | Passing | Denominator | Rate | Distance from J point estimate `71.2695%` |
|---|---:|---:|---:|---:|
| InterHub canonical solve units | 3,502,340 | 4,981,984 | `70.3001%` | `0.9694` pp |
| RQ009 ledger rows | 6,405,292 | 8,994,736 | `71.2116%` | `0.0579` pp |

Both fall inside the J interval. The ledger-row domain is about 17x closer to the J point estimate.
This is consistent with J being a Horvitz-Thompson, row-weighted estimator, whose estimand is
therefore a row-weighted pass rate rather than a deduplicated solve-unit pass rate.

Two limits on this observation, both mandatory to carry forward:

1. The J HT denominator is `2,646,058` weight units, which is not equal to `8,994,736` RQ009 ledger
   rows. Whether these are the same domain under different weighting conventions is **not yet
   established**. This is an open item; it must not be described as "the domains agree".
2. Even if the domains were shown to align, J is a sample-based estimate and K2 is a census. The
   defensible statement is bounded to: "on the ledger-row domain the design-based estimate and the
   census differ by `0.06` percentage points." It is still not a validation of either.

Both rates must be reported with their denominators. Any comparison against J should use the
ledger-row rate and state the reason; the solve-unit rate should be reported separately.

## 5. Corrected G-Anchor Check

K2-1 compared HPC-produced K2 rows against `.codex-fleet/rq015b-repair/work/anchor_mse.csv`, which is the RQ015B Mac baseline. The K2 task required `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`, the G-track HPC baseline.

K2-2 changed only the `validate_g_anchor()` baseline path in `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py`. The comparison code stayed unchanged: float64 conversion, canonical-key alignment, and `diff != 0.0` failure.

Machine source: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json`.

| Field | Value |
|---|---:|
| `status` | `PASS` |
| `anchor_rows` | 2,300 |
| `compared_rows` | 2,300 |
| `max_abs_diff` | 0.0 |
| `first_mismatch` | null |

The earlier `max_abs_diff=0.013332352186283258` is preserved as failure history: it was K2-HPC compared to B-Mac. Supervisor spot check on `ipv_007137|46|1` showed K2-HPC matched G-HPC exactly and differed from B-Mac by exactly that old validator value. Therefore the old G-anchor FAIL was a wrong-baseline comparison, not a K2 numeric defect.

## 6. RQ009 Join Checks

The old `finalize()` value `join_counts["duplicates"] = 0` was hard-coded, and `validate_outputs()` did not check B-class join duplicates. K2-2 added the missing measurement outside finalize.

Machine source: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/rq009_join_key_uniqueness.json`.

| Field | Value |
|---|---:|
| `rows` | 8,994,736 |
| `unique_keys` | 8,994,736 |
| `duplicates` | 0 |
| `duplicate_examples` | `[]` |

`new_solve_rows = 0` remains valid only by construction: B-class rows are materialized from the join mapping and store `interhub_canonical_key`; K2-2 did not remeasure it as a data-derived count.

Array restoration uses the corrected 18:44 file `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/rq009_array_restore_1000_corrected.json`, not the original `final_validation_summary.json` field. The original summary says `FAIL` because it compared `found_in_A=500` unique InterHub keys against `sampled_rows=1000` B rows. The corrected version compares resolved sampled B rows and reports `resolved_sample_rows=1000 / sampled_rows=1000`, missing `0`, status `PASS`.

## 7. SOLVER_FAILURE Characterization

The supervisor withdrew the old single-shard tripwire because it was calibrated from a pilot with `0/1,120` failures and never calibrated on nuPlan Vegas. K2-2 did not rerun any failed row. It characterized the existing `1,934` engineering-failure rows.

Machine source: `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/solver_failure_characterization.json`.

| Distribution | Counts |
|---|---|
| Source | `nuplan=1,934` |
| Dataset | `nuplan_train=1,934` |
| PKL | `train_vegas2.pkl=130`, `train_vegas3.pkl=634`, `train_vegas4.pkl=384`, `train_vegas5.pkl=392`, `train_vegas6.pkl=394` |
| Shard | `A_train_vegas2_0018=130`, `A_train_vegas3_0010=392`, `A_train_vegas3_0014=242`, `A_train_vegas4_0005=384`, `A_train_vegas5_0009=392`, `A_train_vegas6_0019=394` |
| `n_obs` | `11=1,862`; `5,6,7,8,9,10=12` each |
| `n_band` | `FULL=1,862`, `RAMP=72` |
| `signature` | `N=1,693`, `Z=220`, `U=21` |
| measurement role | `agent_1=967`, `agent_2=967` |

All 1,934 rows have `source_attempt_status=ATTEMPTED`, `failure_type=SOLVER_FAILURE`, `reason_code=SOLVER_FAILURE`, `solver_status=SOLVER_FAILURE`, and null `mse_spread`, `max_w_log`, `k_eff_log`, and `ipv_log`. `non_finite` is 0 in the old threshold source.

The 1,934 rows collapse to 967 case-frame pairs, exactly two rows per pair, across six `scene_unique_id` values. The largest cases are `ipv_012600=394`, `ipv_000875=392`, `ipv_009708=392`, `ipv_009097=384`, `ipv_001328=242`, and `ipv_008344=130`.

Relationship to the earlier 400 degenerate anchors: not the same mechanism. The earlier G result was 400 `spread(mse)==0` degenerate anchors, all nuPlan, with signature `U=399` and `N=1`, and platform-stable MSE vectors. K2 SOLVER_FAILURE rows are all nuPlan Vegas and mostly `n_obs=11`, so they are in a related nuPlan stress regime, but they are mostly `signature=N` (`1,693/1,934`) and have no MSE vector because the solver failed before gate metrics were materialized. Using the existing J/G proxy `signature in {U,Z}` for source-collinear/zero-postwarm scope, only `241/1,934` are source-collinearity proxy true. Therefore these failures are engineering failures adjacent to the nuPlan geometry regime, not evidence of the same `NO_IPV_EFFECT` degenerate-anchor mechanism.

## 8. Failure History Preserved

K2 did have real execution failures before the final product. They are preserved here because they matter operationally, but none touched the numeric path: sigma, candidate grid, solver formula, gate formula, and K2 L1 numerical values are not changed by these failures or their fixes.

Solve submissions:

| Job | Role | Outcome | Root cause | Numeric path touched? |
|---|---|---|---|---|
| `2069424` | first solve array, `1-460%450` | cancelled | Matplotlib font-cache concurrent lock | No |
| `2069818` | retry solve array, `1-460%450` | cancelled | PyArrow fixed-size-list parquet writer could not write null array rows | No, storage layout only |
| `2070433` | final solve array, `1-460%427` | completed `460/460` | no solve failure in final submission | No |

Finalize submissions:

| Job | Role | Outcome | Root cause | Numeric path touched? |
|---|---|---|---|---|
| `2071368` | first finalize | cancelled after diagnosis | per-row recomputation of source parquet SHA-256 was too slow | No |
| `2072466` | patched finalize | exited code 2 after validation | old validation compared against wrong G-anchor baseline and included withdrawn solver-failure threshold | No |

All submitted jobs used `--partition=intel,fata`; `amd` was not used. Slurm evidence is in `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/slurm_jobs_summary.json` and `slurm_sacct_all.tsv`.

## 9. Interface Warning

This warning is also written to `INTERFACE_NOTE.md`.

`ipv_log = 0` is a legal and frequent passing-gate estimate.

> **Correction, 2026-08-03, supervisor.** This section originally read: "after the gate, `23.40%` of
> passing rows have `ipv_log = 0`". That number is `238/1,017` from the J anchor sample, not a
> property of this product, and the supervisor carried it into the interface contract in error.
> Census values: InterHub solve units, denominator 3,502,340 — exactly zero `5.0097%`, within
> `1e-9` `9.9516%`. RQ009 ledger rows, denominator 6,405,292 — `4.9121%` and `9.8087%`.
> The original wording is preserved above the correction rather than overwritten, per this
> round's own lesson about silent edits. `INTERFACE_NOTE.md` has been corrected in place.
> Evidence: `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L3b_ipvlog_zero_census.json`. The discriminator is `status` plus `reason_code`, not the numeric value of `ipv_log`. Downstream code must not treat numeric zero as abstention.

Rows outside this K2 materializer scope have `gate_applicable=false` and must not be counted under `NO_IPV_EFFECT` or `NEAR_UNIFORM`.

## 10. Manifest Notes

Shard ownership is determined by the continuous input CSV row-order block used to create each shard. The manifest fields `row_key_min` and `row_key_max` are observed key extrema inside that shard; they are not proof that shards are globally non-overlapping, because row keys are strings and the shard cut is by input CSV order. Future manifest schemas should include explicit source row index interval fields, such as `input_csv_row_index_start_inclusive` and `input_csv_row_index_end_exclusive`, to make the real ownership boundary auditable.

## 11. Methodological Lessons

One-row canaries did not exercise two real paths: multi-worker concurrency and engineering-failure row writeout. The failures that actually occurred were in those paths, so future canaries for this workflow must include concurrent workers and at least one row that writes as an engineering failure.

Every acceptance criterion needs one deliberate failing test proving it can fail for the intended reason. This K2 closeout found two checks that looked active but were not checking the intended thing: RQ009 `duplicates` was hard-coded at 0, and the G-anchor canary compared against the wrong baseline.

## 12. Completion Self-Evidence

Protected file SHA-256:

```text
bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7  src/sociality_estimation/core/agent.py
e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4  src/sociality_estimation/core/ipv_estimation.py
8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830  src/sociality_estimation/core/reliability_logdomain.py
2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b  pipelines/interhub/process_interhub.py
3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522  configs/ipv_sigma01_exact.json
```

No git commit was made. No Slurm job was submitted by K2-2. No solve, join, threshold scan, RQ007 held-out parsing, or RQ014 blinded-field read was performed by K2-2.

`git --no-optional-locks status --porcelain`:

```text
 M START_HERE.md
 M main_workflow.log
?? nohup.out
?? reports/studies/RQ015A_ipv_estimability_labelling/_to_delete/
```
