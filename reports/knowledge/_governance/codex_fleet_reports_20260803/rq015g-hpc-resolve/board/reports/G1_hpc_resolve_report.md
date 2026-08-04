# G1 HPC Resolve Report

## Run Environment And Job History

- final_scientific_job: `2023332` (`COMPLETED`, exit `0:0`, elapsed `00:14:22`).
- final_hpc_workdir: `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_anchor_resolve_20260801T014419Z`.
- final_partition_node_cpus: `fata` / `fata02` / `6`; job name kept `zxc-rq015g-anchor-resolve`.
- Python: `/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python3.9` / `3.9.24`; numpy/scipy/pandas `1.21.6` / `1.7.3` / `1.4.4`.
- BLAS/thread pins: `{"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}`.
- note: initial reviewed sbatch needed runtime repairs before final compute: Slurm spool cwd, staged sample SHA path, PKL symlink resolution, process-pool memory peak, and queue partition. Only job `2023332` produced scientific artifacts.

| JobID | JobName | Partition | State | Elapsed | NCPUS | ExitCode | NodeList |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023137 | zxc-rq015g-anchor-resolve | intel | FAILED | 00:00:01 | 24 | 1:0 | cpui192 |
| 2023140 | zxc-rq015g-anchor-resolve | intel | FAILED | 00:00:40 | 24 | 1:0 | cpui192 |
| 2023141 | zxc-rq015g-anchor-resolve | intel | FAILED | 00:00:03 | 24 | 2:0 | cpui192 |
| 2023247 | zxc-rq015g-anchor-resolve | intel | OUT_OF_MEMORY | 00:01:49 | 24 | 0:125 | cpui152 |
| 2023274 | zxc-rq015g-anchor-resolve | intel | CANCELLED by 10451 | 00:00:00 | 24 | 0:0 | None assigned |
| 2023291 | zxc-rq015g-anchor-resolve | intel | CANCELLED by 10451 | 00:00:00 | 6 | 0:0 | None assigned |
| 2023332 | zxc-rq015g-anchor-resolve | fata | COMPLETED | 00:14:22 | 6 | 0:0 | fata02 |

## SHA And Input Equality Checks

| artifact | actual | expected | status |
| --- | --- | --- | --- |
| `.codex-fleet/rq015b-repair/work/sample_v1.csv` | `d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e` | `d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e` | OK |
| `configs/ipv_sigma01_exact.json` | `3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522` | `3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522` | OK |
| `pipelines/interhub/process_interhub.py` | `2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b` | `2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b` | OK |
| `src/sociality_estimation/core/agent.py` | `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` | `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` | OK |
| `src/sociality_estimation/core/ipv_estimation.py` | `e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4` | `e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4` | OK |
| `src/sociality_estimation/core/reliability_logdomain.py` | `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` | `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` | OK |

- staged PKL symlink target: `/share/home/u25310231/ZXC/sociality_estimation/data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/pkl`; resolved target: `/share/home/u25310231/ZXC/sociality_estimation/data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/batches/20260611_fullset_param_rerun/pkl`.
| PKL | local SHA | HPC SHA | size match | SHA match | anchors |
| --- | --- | --- | --- | --- | --- |
| `train_singapore.pkl` | `b3dfb5d331281fdbad5b24a135a74fb9d587d006486d6e999dae2105030627cf` | `b3dfb5d331281fdbad5b24a135a74fb9d587d006486d6e999dae2105030627cf` | True | True | 5 |
| `train_vegas1.pkl` | `284b20daf76da80b3124962caa7d92c7546da67430348ee04799b1f83b63067f` | `284b20daf76da80b3124962caa7d92c7546da67430348ee04799b1f83b63067f` | True | True | 23 |
| `train_vegas2.pkl` | `0d43a930219e1b0c1a8ea80207d83cac694ef3265a13a66837acc93bd9e77e0e` | `0d43a930219e1b0c1a8ea80207d83cac694ef3265a13a66837acc93bd9e77e0e` | True | True | 219 |
| `train_vegas3.pkl` | `a0de6c3d41c8b00aacdc7445ef6a1ed0e51e8cc40c0afa251a0648fa6b89af4f` | `a0de6c3d41c8b00aacdc7445ef6a1ed0e51e8cc40c0afa251a0648fa6b89af4f` | True | True | 275 |
| `train_vegas4.pkl` | `78fe06c28c60fabf07fb8027ab7a9ff480975543abd93796ca37556ac699b5e3` | `78fe06c28c60fabf07fb8027ab7a9ff480975543abd93796ca37556ac699b5e3` | True | True | 154 |
| `train_vegas5.pkl` | `52b2a134b2ee0e2481a175b5ed83c6ea5ee94df4ed74b2ea51c7dab41d497e72` | `52b2a134b2ee0e2481a175b5ed83c6ea5ee94df4ed74b2ea51c7dab41d497e72` | True | True | 236 |
| `train_vegas6.pkl` | `14e5e6ad7dca31fbf626c3f18fc591a5d6e039f5805cca90f5f72ffc949ce4a0` | `14e5e6ad7dca31fbf626c3f18fc591a5d6e039f5805cca90f5f72ffc949ce4a0` | True | True | 238 |
| `waymo_0-299.pkl` | `f4b1b8ba03f514964674c9a69c49655a4126baa0694347ed3db2a69594892eee` | `f4b1b8ba03f514964674c9a69c49655a4126baa0694347ed3db2a69594892eee` | True | True | 676 |
| `waymo_800-999.pkl` | `d96669710f2c9b3e26af4defd2b33f62372672dbd89c9caf2ffcdd2456d187ab` | `d96669710f2c9b3e26af4defd2b33f62372672dbd89c9caf2ffcdd2456d187ab` | True | True | 474 |

## Coverage And Health

- post-fetch validation: `all_passed=True`; failed checks: `[]`.
- rows: `2300`; solve_errors: `0`; nonfinite_rows: `0`; mse_nonfinite_cells: `0`.
- serial cross-check: n=`24`, max_abs_diff=`0.0`, exact_zero=`True`.
- split distribution: `{'development': 1647, 'guard': 653}`; held_out_parsed_rows=`0`.
- structural support for held_out seal: sample has no `held_out`; `run_b2.ALLOWED_SPLITS=['development', 'guard']`; `load_sample()` rejects splits outside that set before solving. Files read by the driver: `run_b1_rq015b.py`, `run_b2_rq015b.py`, `sample_v1.csv`, `sample_v1.sha256`, `b2_summary.json`, Mac `anchor_mse.csv`, `d1_sigma_analysis.py`, local manifest, staged `src/`, `configs/`, and the 9 PKL files listed above.
- quotas: U300/Z150/N125 x 4 source/n_band cells confirmed: `{'nuplan|FULL|N': {'case_count': 123, 'drawn': 125}, 'nuplan|FULL|U': {'case_count': 226, 'drawn': 300}, 'nuplan|FULL|Z': {'case_count': 143, 'drawn': 150}, 'nuplan|RAMP|N': {'case_count': 124, 'drawn': 125}, 'nuplan|RAMP|U': {'case_count': 213, 'drawn': 300}, 'nuplan|RAMP|Z': {'case_count': 145, 'drawn': 150}, 'waymo|FULL|N': {'case_count': 124, 'drawn': 125}, 'waymo|FULL|U': {'case_count': 288, 'drawn': 300}, 'waymo|FULL|Z': {'case_count': 149, 'drawn': 150}, 'waymo|RAMP|N': {'case_count': 123, 'drawn': 125}, 'waymo|RAMP|U': {'case_count': 284, 'drawn': 300}, 'waymo|RAMP|Z': {'case_count': 144, 'drawn': 150}}`.
- min_mse p0/p50/p100: `0` / `0.05248295094` / `655.53318`.

## Mac Vs HPC Anchor Comparison

- `anchor_mse_hpc.csv` columns: 36 and exactly match Mac `ANCHOR_FIELDNAMES`/`anchor_mse.csv` order.
- argmin_candidate changed: `686` / `2300` = `29.83%`.
- argmin direction crosstab: `{'0->1': 41, '0->2': 37, '0->3': 40, '0->4': 3, '0->5': 4, '0->6': 4, '1->0': 66, '1->2': 37, '1->3': 36, '1->4': 1, '1->5': 4, '1->6': 4, '2->0': 44, '2->1': 43, '2->3': 54, '2->4': 3, '2->5': 6, '2->6': 2, '3->0': 35, '3->1': 41, '3->2': 50, '3->4': 13, '3->5': 4, '3->6': 4, '4->0': 3, '4->1': 5, '4->2': 1, '4->3': 14, '4->5': 10, '4->6': 6, '5->0': 4, '5->1': 3, '5->2': 1, '5->3': 6, '5->4': 9, '5->6': 11, '6->0': 7, '6->1': 4, '6->2': 5, '6->3': 3, '6->4': 6, '6->5': 12}`.
- legacy_fallback_triggered flips: `10`; Mac true / HPC false = `0`; Mac false / HPC true = `10`.
- fallback direction crosstab: `{'False->True': 10}`.

| metric | group | level | abs max | abs p99 | abs median |
| --- | --- | --- | --- | --- | --- |
| `ipv_error_legacy` | ALL | ALL | 0.62203553 | 0.24561147 | 1.5979871e-07 |
| `ipv_error_legacy` | source | nuplan | 0.62203553 | 0.32479281 | 2.5871704e-06 |
| `ipv_error_legacy` | source | waymo | 0.62203553 | 0.093028268 | 6.4329875e-11 |
| `ipv_error_legacy` | signature | N | 0.54580317 | 0.26056482 | 0.0034279008 |
| `ipv_error_legacy` | signature | U | 0.62203553 | 0.016724913 | 0 |
| `ipv_error_legacy` | signature | Z | 0.49326077 | 0.22181669 | 0.00046028703 |
| `ipv_error_legacy` | n_band | FULL | 0.62203553 | 0.2599773 | 1.4762177e-08 |
| `ipv_error_legacy` | n_band | RAMP | 0.62203553 | 0.17187556 | 2.0727256e-07 |
| `ipv_error_log` | ALL | ALL | 0.56425342 | 0.27229572 | 0.00011568215 |
| `ipv_error_log` | source | nuplan | 0.56425342 | 0.29073114 | 3.7064927e-05 |
| `ipv_error_log` | source | waymo | 0.48216249 | 0.26025012 | 0.0001981369 |
| `ipv_error_log` | signature | N | 0.54580317 | 0.26056482 | 0.0034279008 |
| `ipv_error_log` | signature | U | 0.56425342 | 0.28762667 | 2.7722269e-12 |
| `ipv_error_log` | signature | Z | 0.49326077 | 0.22181669 | 0.00046028703 |
| `ipv_error_log` | n_band | FULL | 0.54580317 | 0.2599773 | 0.00044972274 |
| `ipv_error_log` | n_band | RAMP | 0.56425342 | 0.28385122 | 2.6572535e-05 |
| `ipv_legacy` | ALL | ALL | 1.2084997 | 0.30889665 | 0 |
| `ipv_legacy` | source | nuplan | 1.1780972 | 0.33296173 | 0 |
| `ipv_legacy` | source | waymo | 1.2084997 | 0.19670145 | 0 |
| `ipv_legacy` | signature | N | 1.2084997 | 0.61795142 | 0.01095008 |
| `ipv_legacy` | signature | U | 1.1780972 | 5.5511151e-17 | 0 |
| `ipv_legacy` | signature | Z | 0.30877861 | 0.04238299 | 1.5612511e-17 |
| `ipv_legacy` | n_band | FULL | 1.1780972 | 0.35687382 | 0 |
| `ipv_legacy` | n_band | RAMP | 1.2084997 | 0.24575398 | 0 |
| `ipv_log` | ALL | ALL | 1.8374682 | 0.6132145 | 2.7755576e-17 |
| `ipv_log` | source | nuplan | 1.8374682 | 0.55584192 | 6.9388939e-18 |
| `ipv_log` | source | waymo | 1.2084997 | 0.6252473 | 5.5511151e-17 |
| `ipv_log` | signature | N | 1.2084997 | 0.61795142 | 0.01095008 |
| `ipv_log` | signature | U | 1.8374682 | 0.78218377 | 0 |
| `ipv_log` | signature | Z | 0.30877861 | 0.04238299 | 2.7755576e-17 |
| `ipv_log` | n_band | FULL | 1.2395361 | 0.6225937 | 2.7755576e-17 |
| `ipv_log` | n_band | RAMP | 1.8374682 | 0.59196846 | 2.7755576e-17 |
| `min_mse` | ALL | ALL | 3.4798327 | 0.20544398 | 0.00017936676 |
| `min_mse` | source | nuplan | 0.50419493 | 0.13179019 | 3.8636525e-05 |
| `min_mse` | source | waymo | 3.4798327 | 0.23556594 | 0.00034476425 |
| `min_mse` | signature | N | 0.12951226 | 0.068047653 | 0.00045087751 |
| `min_mse` | signature | U | 3.4798327 | 0.26680003 | 2.9609781e-05 |
| `min_mse` | signature | Z | 0.13775477 | 0.058165266 | 0.00025988333 |
| `min_mse` | n_band | FULL | 0.46117029 | 0.20721364 | 0.00034593445 |
| `min_mse` | n_band | RAMP | 3.4798327 | 0.17815347 | 7.2640214e-05 |
| `min_rms` | ALL | ALL | 0.61273768 | 0.10179764 | 0.00026063594 |
| `min_rms` | source | nuplan | 0.26147386 | 0.11361419 | 0.00021777859 |
| `min_rms` | source | waymo | 0.61273768 | 0.06668134 | 0.00027409918 |
| `min_rms` | signature | N | 0.22687884 | 0.093047565 | 0.0011387355 |
| `min_rms` | signature | U | 0.61273768 | 0.075860547 | 1.797817e-05 |
| `min_rms` | signature | Z | 0.26147386 | 0.10348083 | 0.00063454542 |
| `min_rms` | n_band | FULL | 0.22687884 | 0.11297079 | 0.00044395822 |
| `min_rms` | n_band | RAMP | 0.61273768 | 0.079825463 | 0.00013311333 |

### Argmin And Fallback Changes By Group

| group | level | n | argmin changed | argmin changed % | Mac fallback true / HPC false | Mac fallback false / HPC true |
| --- | --- | --- | --- | --- | --- | --- |
| n_band | FULL | 1150 | 357 | 31.04% | 0 | 6 |
| n_band | RAMP | 1150 | 329 | 28.61% | 0 | 4 |
| signature | N | 500 | 159 | 31.80% | 0 | 0 |
| signature | U | 1200 | 224 | 18.67% | 0 | 10 |
| signature | Z | 600 | 303 | 50.50% | 0 | 0 |
| source | nuplan | 1150 | 396 | 34.43% | 0 | 6 |
| source | waymo | 1150 | 290 | 25.22% | 0 | 4 |

## D0-D4 Mechanism Split On HPC

| scope | source | denominator | D1 | D2 | D3 | D4 | OK | D1 CI | D2 CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined | all | 534939.0 | 43.17% | 40.32% | 0.00% | 0.00% | 16.52% | 43.17% [39.51%, 46.99%] | 40.32% [36.42%, 44.04%] |
| source | nuplan | 145821.0 | 1.10% | 62.54% | 0.00% | 0.00% | 36.36% | 1.10% [0.70%, 1.57%] | 62.54% [55.60%, 69.44%] |
| source | waymo | 389118.0 | 58.93% | 31.99% | 0.00% | 0.00% | 9.08% | 58.93% [54.30%, 63.56%] | 31.99% [27.65%, 36.24%] |

- bootstrap: B=`2000`, seed=`20260731`, clusters=`1459`.
- Mac baseline for comparison: D1 `43.01%` CI `39.35%-46.83%`, D2 `39.48%` CI `35.69%-43.08%`, OK `17.51%`, legacy_fallback_total `603`.
- HPC legacy_fallback_total `613`; non-U fallback count `0`.

## Sigma Scan

- copied script: `.codex-fleet/rq015g-hpc-resolve/work/d1_sigma_analysis_hpc.py`; post-fetch check confirms only line 17 differs from the D-track source.
- supplemental fixed-point check: `.codex-fleet/rq015g-hpc-resolve/work/sigma_02347_check.json`; used only because the copied script's HPC rederived sigma is `0.2290908968`, not fixed `0.2347`.
| sigma | source | n | k_eff median | near-uniform | hard-argmax |
| --- | --- | --- | --- | --- | --- |
| 0.02 | ALL | 2300 | 2.55005 | 31.65% | 29.87% |
| 0.02 | nuplan | 1150 | 6.86611 | 53.13% | 13.74% |
| 0.02 | waymo | 1150 | 1.6873 | 10.17% | 46.00% |
| 0.1 | ALL | 2300 | 6.77865 | 53.26% | 13.13% |
| 0.1 | nuplan | 1150 | 6.99978 | 69.57% | 2.00% |
| 0.1 | waymo | 1150 | 4.38538 | 36.96% | 24.26% |
| 0.2347 | ALL | 2300 | 6.99249 | 65.78% | 6.87% |
| 0.2347 | nuplan | 1150 | 6.99999 | 82.17% | 0.26% |
| 0.2347 | waymo | 1150 | 6.39825 | 49.39% | 13.48% |

- rederived sigma ALL `0.2290908968`: near-uniform `65.57%`, hard-argmax `7.04%`.
| source | near-uniform non-decreasing | hard-argmax non-increasing | near-uniform range | hard-argmax range |
| --- | --- | --- | --- | --- |
| ALL | True | True | 26.61% -> 81.91% | 35.26% -> 0.17% |
| nuplan | True | True | 48.17% -> 96.00% | 19.48% -> 0.00% |
| waymo | True | True | 5.04% -> 67.83% | 51.04% -> 0.35% |

## Degenerate Spread Check

- Mac-defined `spread(mse)==0` count: `400`.
- HPC still `spread==0`: `400`.
- Mac/HPC MSE JSON strings exactly equal: `400`.
- all_hpc_still_zero=`True`; all_mse_strings_equal=`True`; anchor_ids_sha256=`870dc417ff489364a3e6f5902c916def4ae735bd4e42209bc822bd68dcd53d0e`.

## 判定表

| # | 结论 | 判定 | 支撑数字 |
| --- | --- | --- | --- |
| B2-1 | 平价门：log 域 == 连乘 | 存活 | Mac max 3.75e-15; HPC eligible_count 1528, max 3.2196468e-15, pass_1e-12=True |
| B2-2 | log 域下兜底不可达 | 存活 | HPC shifted softmax denominator 1, finite=True, k_eff=1 |
| B2-3 | 400 行 MSE 逐位相同（无交互退化） | 存活 | Mac spread==0 count 400; HPC still zero 400; exact JSON strings 400 |
| B2-4 | D1/D2 机制拆分 | 数值需更新 | Mac D1 43.01% / D2 39.48% / OK 17.51%; HPC D1 43.17% / D2 40.32% / OK 16.52%; waymo 58.93%, nuplan 1.10% |
| D-1 | 近均匀占比 | 数值需更新 | Mac log 53.17%; HPC log 53.26%; legacy 75.91% |
| D-2 | 硬 argmax 占比 | 数值需更新 | Mac log 12.87%; HPC log 13.13%; legacy 1.78% |
| D-3 | sigma 扫描 59% 地板 | 存活 | HPC ALL near-uniform: sigma=0.15 is 60.00%, rederived sigma=0.2290909 is 65.57%, sigma=0.2347 is 65.78% |
| D-4 | 两曲线反向单调 | 存活 | ALL grid near-uniform non-decreasing=True; hard-argmax non-increasing=True |

## Leader 复核清单

1. 分区和资源修正：最终作业从 `intel/24` 调整为 `fata/6`，原因是 `intel` 队列窗口过晚且 24 worker OOM；需要复核这是否仍满足“受管 HPC 冻结环境”的解释边界。
2. sigma=0.2347：复制版 `d1_sigma_analysis_hpc.py` 只改 line 17，但原 sweep 不含固定 0.2347；我用 `sigma_02347_check.json` 做补充单点计算，需复核是否接受这个补充口径。
3. 权重重构：D split 复用 B2 函数，但 G1 在允许输入边界内从 `b2_summary.json` 重构 T2 权重，没有读取未列入白名单的 `t2_summary.json`。

## Artifact Paths

- `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv` sha256 `dcce61d304b8550cf6a8f039cc25cc0c7c1ef09679206fea441fc7c912c6edfc`
- `.codex-fleet/rq015g-hpc-resolve/work/g1_compare.json` sha256 `cd3befa80fbc70e630aae45d413a56eca0672bc66b83306e6b26de742ece1ff5`
- `.codex-fleet/rq015g-hpc-resolve/work/g1_hpc_summary.json` sha256 `7334634c0ae1569ef2a7422b5e20a173a5de7166f9f6e9ff847b5dd975815d24`
- `.codex-fleet/rq015g-hpc-resolve/work/hpc_environment_manifest.json` sha256 `08e6a9a9730aded06458751e5bfa82768ec2378f6e9f781ea596318adcc83f05`
- `.codex-fleet/rq015g-hpc-resolve/work/post_fetch_validation.json` sha256 `ed44a7da7d5a2b5323fd91cd9ca16e35549431caa943ade3baa83d46ec8208f5`
- `.codex-fleet/rq015g-hpc-resolve/work/sigma_02347_check.json` sha256 `7efd5c501c48ea2b9fbff4ddcc94ada2cbfbbcb9782a8cf02997f4e4830ae416`
- `.codex-fleet/rq015g-hpc-resolve/work/slurm_attempts_sacct.txt` sha256 `e1c7b25b383676b524fdfda14a5cf5ac845d5c23f588f3b41006f015b4e736f8`
