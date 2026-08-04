# G2 Cross-Node Determinism Gate

UTC closeout: 2026-08-01T03:13:41Z

## Scope

Single-case cross-node probe only. The `2022477` cpui158 output for `ipv_000001`
was used as the reference. A new `fata02` Slurm job recomputed the same copied
input under the frozen sigma01-compatible environment and compared the resulting
xlsx/csv values.

No source code, managed checkout, frozen reference work directory, or other
track directory was modified.

## Slurm Job

| Field | Value |
|---|---|
| Job ID | `2024766` |
| Job name | `zxc-rq015g-crossnode-fata` |
| Partition | `fata` |
| Node | `fata02` |
| State | `COMPLETED` |
| ExitCode | `0:0` |
| Elapsed | `00:03:23` |
| New work directory | `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_fata_crossnode_20260801T030403Z` |
| Reference work directory | `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/sigma01_onecase_20260801T0710CST` |

## Input SHA-256 Check

| Input | 2022477 source SHA-256 | fata02 copied SHA-256 | Result |
|---|---:|---:|---|
| `input/code_parity_sample_cases.csv` | `fe81e166ecbba4be6956aea9a4fe41e81841ec6d6040075cb142a9d2bc55d26d` | `fe81e166ecbba4be6956aea9a4fe41e81841ec6d6040075cb142a9d2bc55d26d` | match |
| `input/pkl/train_vegas3.pkl` | `a0de6c3d41c8b00aacdc7445ef6a1ed0e51e8cc40c0afa251a0648fa6b89af4f` | `a0de6c3d41c8b00aacdc7445ef6a1ed0e51e8cc40c0afa251a0648fa6b89af4f` | match |

The reference pkl was a symlink; the G2 directory copied the target file as a
regular file.

## Parameter Alignment

| Parameter | 2022477 cpui158 reference | G2 fata02 run |
|---|---|---|
| code checkout | `/share/home/u25310231/ZXC/sociality_estimation/code/repo @ 6bdcc2e6` | same checkout, `6bdcc2e64bacd75d02741aa18ef5d61eef5a2962` |
| Python env | `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01` | same |
| entrypoint | `pipelines/interhub/process_interhub.py` | same |
| `--execution-profile` | `configs/ipv_sigma01_exact.json` | same |
| profile SHA-256 | `3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522` | same |
| `--workers` | `1` | `1` |
| `--solver-mode` | `exact` | `exact` |
| solver preset | `null` | `null` |
| `--mp-start-method` | `auto` | `auto` |
| `--case-timeout-seconds` | `600` | `600` |
| `--reference-clip-margin-m` | `60.0` | `60.0` |
| `--reference-max-points` | `40` | `40` |
| `--reference-smooth-points` | `40` | `40` |
| `--limit` | `1` | `1` |
| plots | `save_plots=false` | `--no-plots` |
| dataset filter | `[]` | `[]` |
| Slurm CPUs | `NCPUS=1` | `--cpus-per-task=1` |

Expected differences only: partition/node/log/output paths.

## Environment Fingerprint

Recorded in
`/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_fata_crossnode_20260801T030403Z/analysis/env_fingerprint.txt`.

| Field | Value |
|---|---|
| Hostname | `fata02` |
| `sys.executable` | `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python` |
| Python | `3.9.24 (main, Oct 21 2025, 20:11:42) [GCC 11.2.0]` |
| numpy | `1.21.6` |
| scipy | `1.7.3` |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `NUMEXPR_NUM_THREADS` | `1` |

## xlsx Value Comparison

Compared files:

- Reference:
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/sigma01_onecase_20260801T0710CST/output/cases/nuplan_train/train_vegas3/scenario_2400/row_00000_3131c9ba61a9/data/ipv_results.xlsx`
- G2 fata02:
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_fata_crossnode_20260801T030403Z/output/cases/nuplan_train/train_vegas3/scenario_2400/row_00000_3131c9ba61a9/data/ipv_results.xlsx`

Key alignment used ordered `timestamp`, `key_agent_1`, `key_agent_2`.

| Metric | Result |
|---|---:|
| Reference rows | `87` |
| fata02 rows | `87` |
| Value columns | `ipv_key_agent_1`, `ipv_key_agent_1_error`, `ipv_key_agent_2`, `ipv_key_agent_2_error` |
| Value count | `348` |
| Ordered key/timestamp equality | `true` |
| `max|Δ|` | `0.0` |
| `mean|Δ|` | `0.0` |
| Exact equal values | `348 / 348` |
| Float64 bitwise equal values | `348 / 348` |

## CSV Value Comparison

Compared `output/code_parity_sample_cases_with_ipv_limit.csv` from both runs.

| Metric | Result |
|---|---:|
| Reference rows | `1` |
| fata02 rows | `1` |
| Numeric columns | `ipv_key_agent_1_mean`, `ipv_key_agent_1_error_mean`, `ipv_key_agent_2_mean`, `ipv_key_agent_2_error_mean` |
| Numeric `max|Δ|` | `0.0` |
| Numeric `mean|Δ|` | `0.0` |
| Numeric exact equal values | `4 / 4` |
| Numeric float64 bitwise equal values | `4 / 4` |
| Non-path result/source columns equal | `true` |
| Absolute `ipv_result_case_dir` equal | `false` |
| Relative case path after `/output/` equal | `true` |

The absolute case-dir difference is the expected new-run-directory difference;
the relative case identity matches.

Comparison artifact:
`/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_fata_crossnode_20260801T030403Z/analysis/crossnode_compare_result.json`.

## 判定

指定判据：

```text
max|Δ| == 0（逐位相同）      → fata02 与 cpui158 对本计算逐位确定，跨节点无差异
max|Δ| ≤ 1e-15              → 同一数值等价类，本轮 2,300 锚点结果直接采信
max|Δ| 显著大于 1e-15        → 分区切换本身引入数值差异，【立刻停下报 leader】，
                                  不要自行扩大规模、不要换第三个分区
```

Observed `max|Δ| = 0.0` and `348 / 348` xlsx values are float64 bitwise
identical.

**Final G2 gate verdict: `max|Δ| == 0`; fata02 与 cpui158 对本计算逐位确定，跨节点无差异。**
