# K1b memory and concurrency pilot for K2 InterHub sharding

## 0. Position and main conclusion

This is a measurement-only follow-up to K1. K1 already established the K2
InterHub/RQ009 rebuild scope and measured the 6-worker all-PKL baseline. K1b
only measures the missing resource point: one shard limited to one PKL file,
with 6, 10, and 16 workers. It does not authorize or run full K2.

Main conclusion: single-PKL sharding reduces peak worker RSS from the K1
all-PKL level of about 15,522 MB/worker to about 2,789 MB/worker on the largest
PKL (`waymo_0-299.pkl`). Among the three measured settings, P16 is the best
K2 shard shape: one PKL plus row-key range, 16 workers, and `--mem=64G`.
Under the requested planning snapshot of 36 eligible intel nodes plus fata02,
this gives 228 concurrent shards, 3,648 active CPU cores, no 4,000-core QOS
binding, and an estimated wall time of 1.02 hours for 3,704 core-hours.

K2 remains unauthorized. This report only supplies the resource measurement.

## 1. Inputs and boundary

Target PKL selection source: `.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/k1b_sample_summary.json`, fields `candidate_pkls_from_k1_worker_memory`, `target_pkl`, and `target_pkl_bytes`; disk sizes came from `.codex-fleet/rq015k-fullcorpus-gate/work/hpc_frozen_pkl_listing.tsv`.

- Target PKL: `waymo_0-299.pkl`.
- Target PKL size: 310,197,719 bytes = 295.828 MB.
- Sample rows: 1,120 solve units, source `k1b_sample_summary.json`, field `sample_rows`; CSV line count is 1,121 including header.
- K1 forced overlap rows: 72, source `k1b_sample_summary.json`, field `forced_k1_overlap_rows`.
- Split filter: 780 `development` rows and 340 `guard` rows, source `k1b_single_pkl_sample.csv`, column `split`.
- `held_out_parsed_rows = 0`, source `k1b_sample_summary.json` and each `results/<P*>/k1b_pilot_summary.json`, field `held_out_parsed_rows`.

HPC submission:

- Remote work dir: `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k1b_memory_pilot_20260802T112244Z`, source `latest_hpc_workdir.txt`.
- Slurm job: `2068976`, job name `zxc-rq015k-k1b`, source `latest_slurm_job.txt` and `k1b_sacct_final.txt`.
- Job size: one combined Slurm allocation, `AllocCPUS=16`, `ReqMem=160G`, node `fata02`, elapsed `00:14:09`, state `COMPLETED`, exit `0:0`, source `k1b_sacct_final.txt`.
- Internal measured configs: P6/P10/P16, each exactly 1,120 solve units, source `results/<config>/k1b_pilot_summary.json`, field `interhub_rows`.
- Thread variables: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, source `results/<config>/k1b_pilot_summary.json`, field `thread_env`.

Live pre-submit recheck was captured before `sbatch` at `2026-08-02T11:22:56Z`, source `cluster_snapshot_pre_submit.txt`, field `snapshot_utc`. The QOS association line was `cpu-4000_core-l40-16_card-a800-16_card`, source `cluster_snapshot_pre_submit.txt`, `sacctmgr association`.

## 2. Measurement table

K1 baseline for speedup: 1,120 units / 499.6056067943573 s = 2.241768 units/s, source `.codex-fleet/rq015k-fullcorpus-gate/work/k1_pilot_summary.json`, fields `interhub_rows` and `solve_loop_elapsed_seconds`.

For P6/P10/P16, RSS values are from `results/<config>/k1b_pilot_summary.json`, field `worker_memory.*.peak_rss_mb`; throughput is `interhub_rows / solve_loop_elapsed_seconds`; config elapsed is the in-allocation start/complete interval from `k1b_progress.log`. The combined Slurm job elapsed is `00:14:09` from `k1b_sacct_final.txt`.

| Config | PKL scope | Workers | Per-worker peak RSS MB, min/median/max | RSS sum | Suggested `--mem` | Throughput, units/s | Speedup vs K1 | Config elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P6 | `waymo_0-299.pkl` | 6 | 2788.129 / 2788.145 / 2788.145 | 16,728.848 MB = 16.337 GiB | 24G | 2.869391 | 1.280x | 00:06:30 |
| P10 | `waymo_0-299.pkl` | 10 | 2788.180 / 2788.188 / 2788.199 | 27,881.879 MB = 27.228 GiB | 40G | 4.503571 | 2.009x | 00:04:09 |
| P16 | `waymo_0-299.pkl` | 16 | 2789.301 / 2789.316 / 2789.332 | 44,629.078 MB = 43.583 GiB | 64G | 6.467854 | 2.885x | 00:02:54 |

Suggested memory rule used in this table: `ceil_to_8G(RSS_sum_GiB * 1.30)`. The formula source is this K1b report; its inputs are `worker_memory.*.peak_rss_mb` from each summary JSON. P16 has 43.583 GiB measured RSS sum; 30% margin gives 56.658 GiB, rounded to `--mem=64G`.

All three configs saw only `["waymo_0-299.pkl"]` in `worker_memory.*.pkl_files_seen`; each worker's `pkl_disk_mb_seen` was 295.828 MB.

## 3. Bitwise consistency

Result: 通过.

Source: `.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/k1b_consistency_summary.json`.

- P6 vs P10: `status=PASS`, `compared_rows=1120`, field `config_pair_checks[0]`.
- P6 vs P16: `status=PASS`, `compared_rows=1120`, field `config_pair_checks[1]`.
- K1 overlap: `status=PASS`, `overlap_rows=72`, `compared_rows=72`, field `k1_overlap`.
- Compared field: exact CSV string equality of `mse_per_candidate[7]`, source field `comparison_field`.
- First mismatch: `null`, source field `first_mismatch`.

All three config summaries also report `failure_counts_interhub.OK = 1120`, so there were no OOM, solver, or non-finite failures in the measured sample.

## 4. Recommended K2 shard configuration

Recommendation: use one PKL per shard, split within that PKL by row-key range, with 16 workers and `--mem=64G`.

Basis:

- P16 is the fastest measured setting: 6.467854 units/s, source `results/P16/k1b_pilot_summary.json`, field `throughput_units_per_second`.
- P16 peak RSS sum is 43.583 GiB; `--mem=64G` leaves the 30% margin used above.
- P16 passed exact `mse_per_candidate[7]` equality against P6 and P10 over all 1,120 rows and against K1 over 72 overlapping rows.
- Thread variables were pinned to 1 in all config summaries.
- The measured sample used only `waymo_0-299.pkl`, the largest K1-seen PKL by disk size.

Do not infer authorization from this recommendation. It only defines the measured resource shape for a future K2 run.

## 5. Scheduling recomputation under the requested snapshot

This calculation uses the snapshot specified in the task text: intel 36 eligible nodes, 629 GiB and 96 CPU per node; fata02 one node, 2.95 TiB and 192 CPU; QOS cap 4,000 CPU cores; total work 3,704 core-hours.

Recommended shard shape: P16, `--mem=64G`, 16 CPU cores per shard.

| Resource | Slots per node formula | Nodes | Slots | Cores |
|---|---:|---:|---:|---:|
| intel | `min(floor(96/16), floor(629/64)) = min(6, 9) = 6` | 36 | 216 | 3,456 |
| fata02 | `min(floor(192/16), floor((2.95*1024)/64)) = min(12, 47) = 12` | 1 | 12 | 192 |
| Total | n/a | 37 | 228 | 3,648 |

QOS check: 3,648 active CPU cores < 4,000-core QOS cap, so the QOS is not binding under this requested 36+fata02 snapshot.

Estimated wall time: `3,704 core-hours / 3,648 active cores = 1.015 hours`, rounded to 1.02 hours.

The live pre-submit recheck is archived separately in `cluster_snapshot_pre_submit.txt`. It showed 48 non-down/drain/inval intel nodes, 40 live intel nodes satisfying P16+64G by idle CPU/free-memory fields, and 1 live fata node satisfying that same shape. The scheduling calculation above intentionally follows the task's requested 36+fata02 snapshot.

## 6. Closeout self-check

Protected file SHA-256 at close:

| File | SHA-256 |
|---|---|
| `src/sociality_estimation/core/agent.py` | `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` |
| `src/sociality_estimation/core/ipv_estimation.py` | `e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4` |
| `pipelines/interhub/process_interhub.py` | `2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b` |
| `src/sociality_estimation/core/reliability_logdomain.py` | `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` |
| `configs/ipv_sigma01_exact.json` | `3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522` |

`git --no-optional-locks status --porcelain` at close:

```text
 M START_HERE.md
 M main_workflow.log
?? nohup.out
?? reports/studies/RQ015A_ipv_estimability_labelling/_to_delete/
```

Slurm jobs submitted by K1b:

| Job ID | Job name | Scope | Rows/config | Worker configs | Slurm request | State |
|---|---|---|---:|---|---|---|
| 2068976 | `zxc-rq015k-k1b` | single PKL `waymo_0-299.pkl` only | 1,120 | P6, P10, P16 | `--cpus-per-task=16 --mem=160G --time=04:00:00` | `COMPLETED`, exit `0:0` |

No full-scale K2 job was submitted. No git commit was made. OnSite and WOD were not touched in this K1b measurement.
