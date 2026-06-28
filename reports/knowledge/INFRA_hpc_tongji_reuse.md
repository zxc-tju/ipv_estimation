# Tongji HPC InterHub IPV Reuse Reference

Last verified: 2026-06-27 by `INFRA-hpc-reuse-doc`.

Shared cross-project HPC usage guide:
`/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/HPC_TONGJI_USAGE_GUIDE.md`.
Use that file for global access, safety, directory, and Slurm conventions; keep
this file as the InterHub/IPV-specific reuse reference.

This is a cross-RQ operating reference for reusable Tongji HPC assets created or
validated during the InterHub/RQ009 IPV work. Verification was read-only on HPC:
`ssh`, `stat`, `ls`/`find`, `wc`, `git rev-parse`, conda import smoke, and
`sinfo`. No HPC files were written.

## Access And Safety

- SSH alias: `tongji-hpc`.
- User working root on HPC: `/share/home/u25310231/ZXC`.
- All durable HPC project work should live under that `/ZXC` root unless the
  user explicitly says otherwise.
- Slurm job names must start with `zxc-` for newly submitted jobs.
- Connection pattern:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc '<read-only command>'
```

- The alias uses ProxyJump through `zt-desktop` and key-based passwordless auth.
  If any password prompt appears, stop and report it; do not type, store, or
  print passwords.
- Full local HPC usage guide, outside this repo:
  `/Users/xiaocong/Documents/Codex/2026-06-25/ben/outputs/claude-hpc-usage-guide.md`.
- Heavy compute must run through Slurm `sbatch`, not on the login node.
- This reference concerns the research repo only. Do not touch the paper repo.

## Key Consistency Rule

For IPV values intended to be consistent with the project's existing
InterHub-derived sigma01 datasets, use the pinned legacy flat checkout on Tongji
HPC:

```text
/share/home/u25310231/ZXC/ipv_estimation
git HEAD 5edd28104bf5989e2dc258c9405ce897d7523cc4
```

Do not use the current local `src/sociality_estimation` estimator for those
canonical target-generation jobs unless a separate exact migration-parity proof
has been completed. RQ009 code parity showed that the pinned legacy code
reproduces stored `sigma01_ipv_timeseries.csv` to floating-point tolerance,
while current local `src/` had material drift.

## Verified Identity

```text
hostname: logini04
whoami:   u25310231
pwd:      /share/home/u25310231
```

## Reusable Assets

| Asset | Exact path | What it is | Verified facts and reuse note |
|---|---|---|---|
| Raw InterHub full-dataset index | `/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data/full_datasets/index.csv` | Full InterHub case index used by the sigma01 and RQ009 hw=4 runs. | Exists; `10,532,834` bytes. Use with the pinned legacy estimator. |
| Raw InterHub PKLs | `/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data/full_datasets/batches/20260611_fullset_param_rerun/pkl/` | Full trajectory PKL bundle. | Exists; 15 `.pkl` files; `du -sh` reports `1.8G`. The convenience symlink `/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data/full_datasets/pkl` points to this batch. |
| Pinned legacy IPV code | `/share/home/u25310231/ZXC/ipv_estimation` | Canonical flat-layout estimator for sigma01-compatible IPV generation. | Git HEAD `5edd28104bf5989e2dc258c9405ce897d7523cc4`; `process_interhub.py` `81,543` bytes, `agent.py` `41,196` bytes, `ipv_estimation.py` `11,596` bytes. |
| Conda env | `conda activate ipv` via `/share/apps/miniconda3/etc/profile.d/conda.sh` | Runtime env for the pinned legacy code. | Import smoke passed: `import numpy, scipy, pandas, shapely; print("ok")`. |
| RQ009 hw=4 full-frame IPV output | `/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260626_sigma_0_1_hw4_rq009/` | Reusable frame-level `history_window=4`, `sigma=0.1` IPV time series and summaries. | `sigma01_hw4_ipv_timeseries.csv` exists, `2,208,674,429` bytes, `3,695,982` physical lines / `3,695,981` data rows, SHA-256 `cf970f01455905000dac4f24909e69f532e21014987a52a541466a2748fd34fc`; key columns include `scene_unique_id` and `frame_index`. |
| RQ009 hw=4 submit/consolidation scripts | `/share/home/u25310231/ZXC/rq009_hw4_submit_20260626/` | Working Slurm array, serial retry, merge, and consolidation scripts used to produce the hw=4 output. | Exists; key scripts listed below. Use as a working example, but stage a new run space/output root for new jobs. |
| Original sigma01 generation-script archive | `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/` | Local repo-tracked process archive for the original sigma01 full rerun. | Contains `hpc_run_files/submit_full_datasets_sigma01_array.sh`, `hpc_run_files/submit_full_datasets_sigma01_merge.sh`, and `consolidation/consolidate_sigma01_batch.py`. |
| Slurm partitions | Tongji Slurm, checked by `sinfo -h -o "%P %D %c %t"` | Available cluster capacity snapshot. | CPU partitions include `intel` 96 CPU/node, `amd` 192 CPU/node, `fata` 192 CPU/node; GPU partitions `L40`, `L40-fbb`, and `A800` were also visible. RQ009 original array saw tasks pending with `(MaxCpuPerAccount)`, so expect account CPU throttling. |

## Raw PKL File Sizes

| File | Size |
|---|---:|
| `av2_motion_forecasting.pkl` | `48,881,183` bytes |
| `lyft_train_full.pkl` | `281,931,715` bytes |
| `train_boston.pkl` | `9,794,208` bytes |
| `train_pittsburgh.pkl` | `6,201,187` bytes |
| `train_singapore.pkl` | `4,967,922` bytes |
| `train_vegas1.pkl` | `10,725,804` bytes |
| `train_vegas2.pkl` | `93,647,450` bytes |
| `train_vegas3.pkl` | `116,669,298` bytes |
| `train_vegas4.pkl` | `66,704,310` bytes |
| `train_vegas5.pkl` | `98,288,174` bytes |
| `train_vegas6.pkl` | `100,851,877` bytes |
| `waymo_0-299.pkl` | `310,197,719` bytes |
| `waymo_300-499.pkl` | `216,924,220` bytes |
| `waymo_500-799.pkl` | `283,794,317` bytes |
| `waymo_800-999.pkl` | `206,783,180` bytes |

## RQ009 hw=4 Output Files

Important top-level files under
`/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260626_sigma_0_1_hw4_rq009/`:

| File | Size / fact |
|---|---:|
| `sigma01_hw4_ipv_timeseries.csv` | `2,208,674,429` bytes; `3,695,981` data rows |
| `sigma01_hw4_ipv_timeseries.csv.gz` | `426,184,890` bytes |
| `sigma01_hw4_ipv_timeseries.csv.sha256` | `cf970f01455905000dac4f24909e69f532e21014987a52a541466a2748fd34fc` |
| `sigma01_hw4_ipv_timeseries.csv.wc_l.txt` | `3,695,982 sigma01_hw4_ipv_timeseries.csv` |
| `index_with_ipv.csv` | `21,543,062` bytes |
| `sigma01_hw4_case_summary.csv` | `23,531,988` bytes |
| `sigma01_hw4_timeseries_health.csv` | `8,033,701` bytes |
| `sigma01_hw4_consolidation_summary.json` | `19,212` bytes |
| `merge_summary.json` | `1,203` bytes |
| `cases/` | Per-case output directory |
| `_consolidation_parts/` | Consolidation part files |

The verified header starts with index/case metadata and includes:
`scene_unique_id`, `frame_index`, `ipv_key_agent_1`,
`ipv_key_agent_1_error`, `ipv_key_agent_2`, and
`ipv_key_agent_2_error`.

## RQ009 Reusable Script Directory

Verified files under `/share/home/u25310231/ZXC/rq009_hw4_submit_20260626/`:

| File | Size |
|---|---:|
| `submit_full_datasets_sigma01_hw4_rq009_array.sh` | `6,343` bytes |
| `retry_failed_cases_serial.py` | `7,601` bytes |
| `merge_full_datasets_sigma01_hw4_rq009.sh` | `2,335` bytes |
| `run_consolidate_sigma01_hw4_batch.sh` | `2,812` bytes |

That directory also contains Slurm logs from jobs `1703214`, `1703215`, and
`1705151`.

Local provenance copies are in:

```text
reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/
```

## Slurm Snapshot

Verified `sinfo -h -o "%P %D %c %t"` output on 2026-06-27:

```text
intel* 182 96 down*
intel* 12 96 mix
intel* 4 96 alloc
intel* 34 96 idle
amd 1 192 drain*
amd 160 192 down*
amd 8 192 drng
amd 22 192 drain
amd 26 192 mix
amd 13 192 alloc
amd 50 192 idle
fata 2 192 inval
fata 13 192 drain*
fata 1 192 mix
fata 2 192 idle
L40 1 56 drng
L40 3 56 drain
L40 15 56 mix
L40 8 56 alloc
L40 7 56 idle
L40-fbb 2 56 alloc
A800 2 56 drng
A800 2 56 drain
A800 11 56 mix
A800 1 56 alloc
A800 3 56 idle
```

RQ009 submit provenance recorded that array job `1703214_[2-3]` initially
waited with `PENDING (MaxCpuPerAccount)`. Plan array sizes accordingly; an
account-level CPU throttle can make large CPU arrays start in waves.

## How To Reuse: Recompute sigma01-Compatible IPV

Use this recipe for a new `history_window=<N>` target that must be consistent
with the existing sigma01-derived datasets.

1. Verify access and pinned code before any compute:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'hostname; whoami; pwd'

ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'repo=/share/home/u25310231/ZXC/ipv_estimation; git -C "$repo" rev-parse HEAD; stat -c "%n|%s bytes" "$repo/process_interhub.py" "$repo/agent.py" "$repo/ipv_estimation.py"'

ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'source /share/apps/miniconda3/etc/profile.d/conda.sh && conda activate ipv && python -c "import numpy,scipy,pandas,shapely; print(\"ok\")"'
```

2. Start from the RQ009 working scripts, but stage a new run space and output
   root for any new job. Do not overwrite:

```text
/share/home/u25310231/ZXC/rq009_hw4_submit_20260626/
/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260626_sigma_0_1_hw4_rq009/
/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun/
```

3. Preserve these estimator settings unless the RQ explicitly changes the
   estimand:

```text
agent.sigma = 0.1
--min-observation 4
--reference-clip-margin-m 60
--reference-max-points 40
--reference-smooth-points 40
nuPlan sampling: 20Hz -> 10Hz
legacy accurate/default SLSQP path in pinned flat checkout
```

4. Adapt the RQ009 array script for the new `history_window`:

```text
Base example:
  /share/home/u25310231/ZXC/rq009_hw4_submit_20260626/submit_full_datasets_sigma01_hw4_rq009_array.sh

Inputs:
  CSV_PATH=/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data/full_datasets/index.csv
  PKL_ROOT=/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data/full_datasets/batches/20260611_fullset_param_rerun/pkl

Change for the new run:
  HISTORY_WINDOW=<N>
  OUTPUT_ROOT=interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>
  RUN_SPACE=/share/home/u25310231/ZXC/<new_run_space>
  #SBATCH --output and --error paths
  job names/comments and consolidation rename suffixes
  #SBATCH --job-name=zxc-<short-task-name>
```

The RQ009 array used 4 shards (`#SBATCH --array=0-3`) and
`--cpus-per-task=192` on `amd`. That is a proven pattern, but account throttles
may delay some array tasks.

5. Submit by Slurm only:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'cd /share/home/u25310231/ZXC/<new_run_space> && sbatch --job-name=zxc-<task> submit_full_datasets_sigma01_hw<N>_<rq>_array.sh'
```

6. Submit merge after the array succeeds:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'cd /share/home/u25310231/ZXC/<new_run_space> && sbatch --job-name=zxc-<task>-merge --dependency=afterok:<array_job_id> merge_full_datasets_sigma01_hw<N>_<rq>.sh'
```

7. Submit consolidation after merge succeeds:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'cd /share/home/u25310231/ZXC/<new_run_space> && sbatch --job-name=zxc-<task>-consolidate run_consolidate_sigma01_hw<N>_batch.sh'
```

8. Verify rows, checksum, and key columns before reuse:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'out=/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>; wc -l "$out"/sigma01_hw<N>_ipv_timeseries.csv; sha256sum "$out"/sigma01_hw<N>_ipv_timeseries.csv; head -1 "$out"/sigma01_hw<N>_ipv_timeseries.csv'
```

9. Fetch back to a repo-local derived-data target:

```bash
mkdir -p '<local_target_dir>'

scp -p -o BatchMode=yes -o ConnectTimeout=15 \
  tongji-hpc:'/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>/sigma01_hw<N>_ipv_timeseries.csv.gz' \
  tongji-hpc:'/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>/sigma01_hw<N>_ipv_timeseries.csv.sha256' \
  tongji-hpc:'/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>/sigma01_hw<N>_case_summary.csv' \
  tongji-hpc:'/share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/<new_batch_id>/sigma01_hw<N>_timeseries_health.csv' \
  '<local_target_dir>/'
```

## General SSH / Slurm Templates

Read-only identity check:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'hostname; whoami; pwd'
```

Queue check:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'squeue -u u25310231'
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'squeue -u u25310231 -j <job_id>'
```

Completed-job status:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList'
```

Partition snapshot:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'sinfo -h -o "%P %D %c %t"'
```

Submit:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc \
  'cd /share/home/u25310231/ZXC/<run_space> && sbatch --job-name=zxc-<task> <script>.sh'
```

Fetch:

```bash
scp -p -o BatchMode=yes -o ConnectTimeout=15 \
  tongji-hpc:'<remote_path>' \
  '<local_path_or_dir>'
```

## Cross-Links

- RQ009 decision:
  `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md`
- RQ009 run root:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/`
- RQ009 HPC provenance:
  - `02_process/03_features/hpc_recon.md`
  - `02_process/03_features/code_parity.md`
  - `02_process/03_features/hpc_submit.md`
  - `02_process/03_features/target_hw4_fetch.md`
- RQ009 reusable script copies:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/submit_full_datasets_sigma01_hw4_rq009_array.sh`
  and neighboring `retry_failed_cases_serial.py`,
  `merge_full_datasets_sigma01_hw4_rq009.sh`,
  `run_consolidate_sigma01_hw4_batch.sh`.
- Original sigma01 generation archive:
  `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/`
- Local fetched RQ009 hw=4 target copy:
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/`
- Current operating pointer:
  `START_HERE.md`
