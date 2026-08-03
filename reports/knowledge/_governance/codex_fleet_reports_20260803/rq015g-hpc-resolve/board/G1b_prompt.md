# G1（第 2 段）— 执行 HPC 重解并完成对照分析

你是 track G 的唯一执行 agent。仓库根：
`.`

**这一段有网络。** 上一段你把准备工作全部做完了，但沙箱禁网导致 `ssh` 返回
`Operation not permitted`，所以卡在提交前。leader 已复核你写的脚本并放行；
本次启动已开启 `network_access=true`，`ssh tongji-hpc` 现在可用。

**先自检网络**：`ssh -o BatchMode=yes -o ConnectTimeout=15 tongji-hpc 'hostname; squeue -u u25310231 | head'`
若仍失败，写 BLOCKED 报告并停下，**不要**用 `--dangerously-bypass-approvals-and-sandbox` 绕。
**若出现任何密码提示：立刻停下并上报，不要输入/存储/打印密码。**

---

## 一句话任务

RQ015B 的 T5 在 macOS 上解了 2,300 个冻结锚点；macOS 已被证明不是严格数值复现环境
（同一 case 对存档 max diff = 1.1244582，与网格端点 3π/8=1.178 同量级）。
把**同一批锚点、同一份代码、同一组参数**放进已证可复现的受管 HPC 冻结环境重解，
逐锚点对照，给出修正后的 D1/D2 机制拆分与 σ 扫描数字。

**描述性/诊断性产出。一轮做完，自查数值健康与覆盖，出报告。不做盲审、不出第二版规格。**

---

## 你上一段已经产出的东西（leader 已复核，**直接用，不要重写**）

```
work/run_g1_hpc.py                        薄驱动（复用 run_b2 的 solve_anchor_task）
work/submit_rq015g_anchor_resolve.sbatch  24 核 / 4h / BLAS 全部钉 1 线程 / 冻结环境 python
work/stage_and_submit_g1_hpc.sh           staging + sbatch（leader 已逐行复核，放行）
work/fetch_g1_hpc_outputs.sh              回捞产物
work/local_input_manifest.json            本地输入 SHA 清单（9 个 PKL + 源码 + sample）
board/reports/BLOCKED_G1_transport_20260801T010553Z.md   上一段的阻塞记录
```

leader 复核结论：staging 脚本只写**新建的** `work_dirs/INFRA/rq015g_anchor_resolve_<UTC戳>/`，
PKL 符号链接指向正确的 `full_datasets/pkl`，作业名 `zxc-` 前缀，仅走 `sbatch`，
`BatchMode=yes`，不碰 managed checkout。**符合硬边界，可以执行。**

---

## 现在做什么

### 第 1 步 — 提交
从仓库根执行 `.codex-fleet/rq015g-hpc-resolve/work/stage_and_submit_g1_hpc.sh`。
记下 `HPC_WORKDIR` 与 `JOB_ID`（脚本会写进 `work/latest_hpc_workdir.txt` /
`work/latest_slurm_job.txt` / `work/latest_sbatch_output.txt`）。

### 第 2 步 — 轮询（**不要提交完就走**）
用 `squeue -u u25310231 -j <JOB_ID>` 与 `sacct -j <JOB_ID> --format=JobID,JobName%28,State,Elapsed,NCPUS`
轮询，同时看 `<HPC_WORKDIR>/logs/*.out|*.err` 增长。
**每 5 分钟**往 `.codex-fleet/rq015g-hpc-resolve/board/progress.log` 追加一行
（格式 `<UTC> | <阶段> | 做了什么 | 结论`，时间戳用 `date -u +%Y-%m-%dT%H:%M:%SZ`，
**不要前瞻估计**）。
预算参考：Mac 约 3.2 CPU-s/锚点 × 2300 ≈ 2 CPU-小时；24 核下计算部分约 5–15 分钟，
另加 PKL 载入（约 1.1 GB）。若 30 分钟仍 PD，照常每 5 分钟记一行，不要放弃。

### 第 3 步 — 回捞 + 校验
用 `fetch_g1_hpc_outputs.sh` 把产物取回本地 `work/`。
**运行时校验（任一条不过 ⇒ 停下并在报告里写明，不要将就）**：
```
□ sample_v1.csv SHA-256 == d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e
□ HPC 侧 agent.py                  == bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7
       ipv_estimation.py           == e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4
       reliability_logdomain.py    == 8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830
       process_interhub.py         == 2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b
□ 9 个 PKL 的 SHA-256：HPC 侧 == work/local_input_manifest.json 里的本地值
  （**这是"同一份输入数据"的唯一证明**）
□ 实际 Python 来自 envs/ipv-exact-sigma01 且为 3.9.24 —— 这就是本轮的全部意义；
  **若因任何原因用了别的环境，本轮结论作废，如实上报而不是将就**
□ 行数 == 2300，solve_errors == 0，anchor_id 集合与 sample_v1 完全一致
□ 所有 split ∈ {development, guard}；held_out 行数 == 0
□ 串行交叉校验（≥24 个代表性锚点，单进程重解）max_abs_diff == 0.0
  若不是 0.0，**如实报告并把结论相应降级**，不要掩盖
```

### 第 4 步 — 对照与重算（本轮的科学产出）

**4.1 `work/anchor_mse_hpc.csv`** —— 36 列，列名与顺序与 Mac 版
`.codex-fleet/rq015b-repair/work/anchor_mse.csv` 完全一致（即 `ANCHOR_FIELDNAMES`）。
一列都不许改名或增删——这是能自动比对的前提。

**4.2 `work/g1_compare.json` + 报告中的对照表**（Mac vs HPC 逐锚点）：
- `min_mse`、`min_rms`、`ipv_log`、`ipv_legacy`、`ipv_error_log`、`ipv_error_legacy`
  的绝对差分布：**max / p99 / median**（再给相对差更好）
- **`argmin_candidate` 变了多少个锚点**：计数 + 占比 + 变化方向交叉表
  ← "落到不同候选"的直接证据
- **`legacy_fallback_triggered` 翻转了多少**：分别给
  「Mac 触发 / HPC 不触发」与「Mac 不触发 / HPC 触发」两个方向的计数
  ← **直接决定 D1 被高估了多少，是本轮最重要的单个数字**
- 以上全部**分源给**（waymo / nuplan），另按 `signature`(U/Z/N) 与 `n_band`(FULL/RAMP) 分组

**4.3 在 HPC 数字上重算，与 Mac 版并列**：
- **D0–D4 机制拆分**：复用 `run_b2_rq015b.py` 的 `classify_mechanism` /
  `threshold_and_classify` / `weighted_counts` / `bootstrap_ci` / `group_table` /
  `sensitivity_summary`（同样 B=2000、seed=20260731），保证口径与 Mac 一致。
  **必须分源（waymo / nuplan）**；**合并值不得单独呈现**，要与分源值同表并列。
- **σ 扫描**：把 `d1_sigma_analysis.py` 复制为 `work/d1_sigma_analysis_hpc.py`，
  **只改第 17 行 `INPUT_CSV`** 指向 `anchor_mse_hpc.csv`，其余不动。
  至少报 **σ = 0.02 / 0.1 / 0.2347**，看 ① 59% 地板是否仍成立
  ② 两条曲线反向单调是否仍成立。
- **`spread(mse) == 0` 的退化锚点**（Mac 侧约 400 行）：先按该定义在 Mac 版里数出实际条数，
  再看它们在 HPC 侧是否仍 `spread==0` 且与 Mac **逐位相同**。
  leader 预期成立（无交互时目标退化为 `cos(ipv)·interior + 常数`，正标量不改极小点，解析性质）。
  **若不成立，这是重点发现，要专门解释，不要一笔带过。**

**4.4 判定表** —— 每条给 **存活 / 数值需更新 / 结论改变** 三选一 + 支撑数字：

| # | 结论 | Mac 数字 |
|---|---|---|
| B2-1 | 平价门：log 域 ≡ 连乘 | 3.75e-15 |
| B2-2 | log 域下兜底不可达 | 解析 + 极端输入 + pytest |
| B2-3 | 400 行 MSE 逐位相同（无交互退化） | — |
| B2-4 | D1/D2 机制拆分 | D1 43.01% / D2 39.48% / OK 17.51%；waymo 58.73%、nuplan 1.06% |
| D-1 | 近均匀占比 | 53.17% (log) |
| D-2 | 硬 argmax 占比 | 12.87% (log) |
| D-3 | σ 扫描 59% 地板 | 59% |
| D-4 | 两曲线反向单调 | — |

B2-1/B2-2/B2-3 按 leader 判断**应当不受影响**（同一组 MSE 上的两种权重公式之比，
或解析性质）；你要在 HPC 数字上**确认**这一点，而不是默认它成立。

**Mac 基线数字**（`b2_summary.json`）：
```
weighted_main  D1=0.43010748889125155 D2=0.3948001734777231 D3=0.0 D4=0.0 OK=0.17509233763101836
bootstrap CI   D1 [0.3935451234499348, 0.46826223927974564]  D2 [0.3568659082008827, 0.4307511112323502]
               B=2000 seed=20260731 clusters=1459
health         rows=2300 solve_errors=0 nonfinite_rows=0 legacy_fallback_total=603
               legacy_fallback_non_U_count=0  min_mse p0=0.0 p50=0.0551034557 p100=655.5329262812
t5             workers=6 executor=thread elapsed=1240.3s serial_check_n=24 serial_check_max_abs_diff=0.0
parity         eligible_count=1526 eligible_max=3.747002708109903e-15 pass_1e_minus_12=True
```
**D 轨 Mac 基线**（`d1_sigma_stats.json`，σ=0.1）：
```
frac_near_uniform_log=0.5317  frac_hard_argmax_log=0.1287
frac_near_uniform_legacy=0.753 frac_hard_argmax_legacy=0.0217
k_eff_log_mean=5.111 median=6.795 ; k_eff_legacy_mean=6.219 median=7.0
legacy_fallback_triggered_true=603  partial_underflow_true=171
```

### 第 5 步 — 报告
`board/reports/G1_hpc_resolve_report.md`，包含：
运行环境与 Slurm 作业号、全部 SHA 校验结果、4.2/4.3/4.4 的表、
数值健康自查（行数、nonfinite、solve_errors、覆盖、配额是否仍为 U300/Z150/N125 × 4 组）、
以及 `held_out_parsed_rows = 0` 的**结构佐证**（不是一句声明：给 sample 的 split 分布、
代码里的 `ALLOWED_SPLITS` 闸门、你读过的文件清单）。
报告末尾写「leader 复核清单」：列出你自己认为最可能被挑战的 3 个点。

---

## 只读这些路径（**不要对 `reports/` 做全仓库 `rg`**
宽泛检索会把 RQ003 `12_blind_annotation/controlled_identity_map.csv` 的
controlled-access 行整行拉进上下文）

```
.codex-fleet/rq015b-repair/work/{sample_v1.csv,sample_v1.sha256,anchor_mse.csv,b2_summary.json,run_b1_rq015b.py,run_b2_rq015b.py}
.codex-fleet/rq015b-repair/board/sampling_contract_v1.md
.codex-fleet/rq015d-sigma-rederive/work/{d1_sigma_analysis.py,d1_sigma_stats.json}
.codex-fleet/rq015d-sigma-rederive/board/reports/D1_sigma_report.md
.codex-fleet/rq015g-hpc-resolve/work/*   （你自己上一段的产物）
src/sociality_estimation/core/*.py   pipelines/interhub/process_interhub.py   configs/
```

## 硬边界

```
□ 【绝不覆盖】sigma01 任何冻结产物、pinned legacy checkout、RQ009/RQ015A 的 run 目录
   HPC 侧只写【新建的】 work_dirs/INFRA/rq015g_anchor_resolve_<UTC戳>/
   本地只写 .codex-fleet/rq015g-hpc-resolve/{work,board}/ 下的新文件
   【不得修改】.codex-fleet/rq015b-repair/ 与 .codex-fleet/rq015d-sigma-rederive/ 下任何文件
□ dev+guard only；held_out 封条守住
□ 不重抽样、不出合同 v2、不改估计器算法
□ 不接线生产：src/sociality_estimation/core/agent.py 一字不动
□ 必须用冻结环境 envs/ipv-exact-sigma01 (Python 3.9.24)
□ 重计算只走 sbatch，不在登录节点跑；作业名 zxc- 前缀
□ 描述性，不作因果主张；只给证据不给建议
   （"要不要申请全量重跑"是 PI 的决定，你给的是修正后的 D1/D2 数字）
□ 全文禁用 `estimability` 与"测出/未测出 IPV"。
   可辩护表述：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
□ 三条 track 并发在同一工作区，铁律：
   禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
   禁止 git checkout 任何历史提交到主工作区
   禁止 git commit
   工作区非空是【预期状态】；你只对自己创建的文件负责，
   清洁性检查只查自己的文件清单，不看全仓库 git status
□ 本地解释器钉死 <local-rq009-venv>/bin/python（系统 python3 缺 pytest）
```
