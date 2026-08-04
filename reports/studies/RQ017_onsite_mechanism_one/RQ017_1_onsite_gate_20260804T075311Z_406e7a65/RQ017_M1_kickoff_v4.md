# RQ017-M1 任务书 v4（执行版）：OnSite materializer —— 为 67,861 个自动驾驶车锚点产出机制一判据

> **版本沿革**：v1 未执行。v2 经第一轮双盲复审新增「测量合同 preflight」，但把 venue 写成
> 「本机跑」——错误，未执行。v3 改为 HPC，经第二轮双盲复审发现三处硬缺陷，未执行。
> **v4 是执行版**，修补了 v3 的三处缺陷并补入监督方后续实测。v1–v3 原文保留，不删不改。
>
> **v3 的三处缺陷（v4 已修）**：
> 1. v3 引用 K2 的 sbatch 模板时**漏掉了 `PYTHONPATH` 与 `pydeps` 安排**。冻结环境
>    **没有 pyarrow**，照 v3 写第一次写 parquet 就崩。
> 2. v3 的「环境同源」只要求「记录并比对」，太软。**版本不钉死，「同一软件栈」就是空话。**
> 3. v3 只说「与 RQ016C 支持门交叉」，**没限定连接列**；而现成打分脚本是整表 `read_parquet`，
>    anchor 表里带 `target_ipv_future` 等目标值，照做会违反输入列白名单且数值检查抓不到。
>
> **另补入**：HPC 上**已经有一份 OnSite 输入**且与本地 sha256 相同（见 §4.2），不必上传。

你是本轮唯一的执行 agent。执行顺序**不得颠倒**：

```
§3 测量合同 preflight（blocker 脚本，本机/登录节点）
  → 全绿才 §4 staging
  → §5 环境同源硬断言（blocker）
  → §7 canary（必须 sbatch）
  → canary 全绿才投全量
  → §8 取回 + 自查 → 出报告
```

仓库根即当前工作目录；相对路径相对仓库根，HPC 路径为绝对路径。

---

## 0. 这件事在哪一步（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由**两道串联的弃权机制**构成：

- **机制一**：这一帧的 IPV 数值能不能估？若七个候选的权重近均匀，则该数值不携带候选间的
  判别信息，直接弃权，不进机制二。规格已由 RQ015 冻结。
- **机制二**：当前场景收集到的人类样本够不够判断这辆车是否偏离？依据是人类参照分布（envelope）。

**已完成**：RQ015 冻结机制一并在 InterHub 全语料（同济 HPC）跑出台账；RQ016C 用纯人-人样本
建好供 OnSite 使用的人类参照 envelope，并验证过打分管线可在真实 OnSite 行上运行。

**唯一还缺的**：OnSite 一行都没有机制一判据——K2 台账
`artifact_id == onsite_dense_timeseries` 的 281,268 行中 `mse_0..mse_6`、`max_w_log`、
`mse_spread`、`status`、`reason_code` 非空计数**全部为 0**，`gate_applicable` 全为 False。

**本轮补上这一块。本轮不做机制二打分，不下任何关于「某辆车像不像人」的结论。**

**为什么必须在 HPC 上跑**：K2 那份 InterHub 台账（本轮全部对照的基准）在 HPC 上产出；
实测 Mac 与 HPC 在 2,300 个重合锚点上 `mse_per_candidate[7]` 字符串不同的有
**1,867/2,300 = 81.17%**，最大逐元素绝对差 **70.4**，`argmin_candidate` 翻转
**686/2,300 = 29.83%**，差异来自软件栈而非 CPU（同一 HPC 栈跨节点逐位相同 348/348）。

## PI 已裁定（冻结，不得重新讨论）

1. **范围 = B：全 timing-valid anchor frames，67,861 行**
2. **参考线合同 = 沿用观测轨迹 fallback**（OnSite dense 源表真实 map/lane/route/reference-line
   字段实测 0/274,022）
3. **venue = 同济 HPC**，分区限 `intel,fata`；**`amd` 不得使用**（未做该分区确定性 canary，
   K2 全程亦未使用）

---

## 1. 冻结配置（`configs/ipv_sigma01_exact.json`，只读，不得改）

```
solver_mode exact ｜ sigma 0.1 ｜ min_observation 4
reference_clip_margin_m 60.0 ｜ reference_max_points 40 ｜ reference_smooth_points 40
current_ipv_history_window 10 ｜ future_target_history_window 4 ｜ future_target_final_offset 6
```

67,861 行表的溯源记录（`feature_history_window=10`、`target_history_window=4`、
`min_observation=4`、`target_final_offset=6`）与之一致。
旧通道 `ONSITE_CHANNEL_EXACT_HW10`（285 行、`target_history_window=10`）**列入 denylist**。

## 2. 复用什么、绝对不要复用什么

### 2.1 只复用这两段

| 复用对象 | 位置 | 说明 |
|---|---|---|
| 求解 | `src/sociality_estimation/core/ipv_estimation.py` 的 `MotionSequence`(:37)、`estimate_ipv_pair`(:181) | 受保护，只读调用，`solver_mode="exact"` |
| 门判据 | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py` 的 `gate_from_mse()`(:649) | 冻结规格逐字实现；`weights_from_mse` 来自受保护的 `reliability_logdomain.py`(:172) |

另可参考 K2 的 **L1 parquet 写出与原子落盘**（同文件 `:776-817`、`:840-876`）——
七候选数组展开、null scalar 写法、manifest 哈希。

### 2.2 **不要**做这几件事

- ❌ **不要用 `scripts/hpc/submit_research_run.sh` 与 `research_run_spec_v2.schema.json`。**
  该 schema 是 **RQ014 专用**（字段含 `wod_path_type_mapping_manifest`、`ratings_source`、
  `g2r_*`、`declassification_export_*`、`recovery_contract`）。按 §4 的 K2 先例自建 staged array job。
- ❌ **不要把 OnSite 行送进 `run_b2_rq015b.py` 的 `solve_anchor_task()`（:268）。**
  它 :271-272 按 InterHub split 拒绝非 development/guard 行，下游从 InterHub PKL 构造序列——
  OnSite 没有 PKL。**OnSite 必须有自己的入口。**
- ❌ **不要整体复用 K2 的 `validate_outputs()`**：硬编码 `4_981_984`、`8_994_736`、
  `14_473_982`（`:1327-1338`），对 OnSite 会产生假 blocker。只可复用 row-level invariant。
- ❌ **不要修改 `run_b2_rq015b.py` 的 `ALLOWED_SPLITS` 护栏**（:72、:271）。见 §6.2。
- ❌ **不要调用 `.codex-fleet/rq016c-human-only-envelope/work/H2/score_external_rows.py`
  或任何整表 `read_parquet` 的重新打分入口。** 见 §8.1。
- ❌ **不要使用 `/share/home/u25310231/ZXC/ipv_estimation`**（已退役）。

### 2.3 本轮实质要新写的，只有一件

**OnSite 行 → `MotionSequence` 的入口**（`data` / `target` / `reference` 三件，
`reference` 用观测轨迹 fallback）。求解与门判据两段原样调用。

## 2.4 门判据（冻结规格，一字不改，不得调参，不得阈值扫描）

```
log_score_i = -mse_i / (2 * sigma^2)      sigma = 0.1
w_log       = softmax(log_score)           用 log-sum-exp（即 weights_from_mse）
mse_spread  = max(mse) - min(mse)

输入非有限 / 缺列 / 求解失败  → NON_FINITE_INPUT 或 SOLVER_FAILURE（工程失败）
elif mse_spread == 0          → ABSTAIN, NO_IPV_EFFECT
elif max(w_log) < 0.20        → ABSTAIN, NEAR_UNIFORM
else                          → OK, ipv_log = sum(candidate_ipv_i * w_log_i)
```

网格 `legacy7_pi_over_8`（7 点 `[-3..3]·π/8`）、`K=7`、`sigma=0.1`：**不得改**。
`theta=0.20` 是**政策阈值不是数据断点**。`mse_spread == 0` 是**精确浮点相等，不得用
`np.isclose`**（在 InterHub 台账上这条正好卡住 30 行：19,994 → 19,964）。
两条科学 reason 互斥且有序，先 `NO_IPV_EFFECT`。**工程失败绝不允许被记成两个科学 reason 之一。**

---

## 3. 【blocker】测量合同 preflight

**实现为独立可执行脚本，提交前必跑，任一条不过即退出非零并停止全流程。**
输出 `measurement_contract.json`。**不是报告里的一段话。**

两轮双盲复审的四位复审独立指出：这一层错了，产出的七个 MSE 仍然有限、权重和为 1、
`max_w_log`/`k_eff_log` 落在合法范围、门判据可复算、`ipv_log` 恒等式成立——
**所有既有检查都会通过**，但对应的是错的行、错的车或错的时刻。

- **C1 键的一对一**：输出的 `product_row_key` 必须与
  `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`
  （带 `product_row_key` 列）逐行一对一：anchor 表 67,861 行、构造键去重 67,861、
  dry-run 67,861 行、**交集 67,861/67,861 = 100.0000%**，无重复无缺失无多余。键构造式写进审计文件。
- **C2 输出粒度冻结**：一行一 anchor，对应自动驾驶车的当前角色，`输出行数 == 67,861`。
  若判断必须四角色展开，**停下上报，不得自行决定**（展开改变分母且与 C1 冲突）。
- **C3 窗口是行位置语义**，不是绝对 frame_index。见
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:836-847` 与 `:930-997`：

  ```python
  for pos in range(n):
      if pos + TARGET_FINAL_OFFSET >= n: continue
      wx_start = max(0, pos - FEATURE_HISTORY_WINDOW + 1)
      valid = frame_index[wx_start : pos + 1] >= MIN_OBSERVATION
  ```

  求解输入为历史窗口 + 当前行。断言你的窗口切法与上式等价。
- **C4 短历史必须显式处理**：`history_row_count` 实测分布（分母 67,861）
  `10→66,289 ｜ 9→257 ｜ 8→258 ｜ 7→261 ｜ 6→264 ｜ 5→265 ｜ 4→267`，
  即 **1,572 行历史短于 10**。按冻结 `min_observation = 4` 处理，**不得在输入端剔除**；
  求解失败按工程失败记录。
- **C5 主体方向冻结**：断言 ego 是自动驾驶车、counterpart 是对手，且与 anchor 表的
  `ego_key_agent` / `counterpart_key_agent` 一致。
- **C6 帧语义事后可审**：输出必须记录 `solve_frame_index`、`anchor_frame_index`、
  `target_window_end_frame_index`、`history_window_used`。
- **C7 输入列白名单**：**用显式 `columns=[...]` 读取**，只读 motion / key / source / window 字段。
  白名单写进审计文件，并断言**实际读取的列集合是白名单的子集**。
  **禁读清单（监督方实测该 anchor 表中确实存在的 9 列，一个都不许进）**：

  ```
  target_ipv_future                             target_ipv_error_future
  counterpart_ipv_current                       counterpart_ipv_error_current
  counterpart_ipv_slope_pre_anchor              counterpart_ipv_history_count
  counterpart_ipv_history_fraction
  M4_ONLY_ego_self_anchor_ipv_current           M4_ONLY_ego_self_anchor_ipv_error_current
  ```

  这些是旧估计器算出的通道与目标值。本轮要产出的是**新的**机制一判据，
  读入它们既无必要，也会让「本轮结果独立于旧估计器」这一点无法主张。
- **C8【v4 新增】CSV 与 parquet 等价**：HPC 上现成的是 **CSV**，而本轮全部既有测量
  （67,861 行、C1 的键一对一、C4 的分布、§6.4 的 7 行异常）都基于**本地 parquet**。
  必须断言：由 CSV 读入后的**行数与键集合与 parquet 完全一致**。不一致即停。

---

## 4. HPC staging 与作业

### 4.1 运行目录

```
/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/
  repo_stage/  pydeps/  inputs/  logs/  outputs/  process_cache/
```

`<run_id>` = `rq017_onsite_materializer_<UTC 时间戳>`。durable 产物、日志、脚本**必须**留在该树下。

### 4.2 【v4 修正】输入已在 HPC，不必上传

监督方实测：以下两个文件已在 HPC，且与本地同名 CSV **字节数与 sha256 完全相同**：

```
/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/
  onsite_m3_av_anchors_multi_allvalid.csv    71,137,488 B  sha256 4ff857c80d84f5e8aae1cb1bbf4ef0d1…
  onsite_ipv_timeseries_multi_allvalid.csv   52,088,770 B  sha256 e49c226bafb950125a69f8b5dc90df02…
```

⚠ 该目录在 `sociality_estimation` 项目树**之外**，**只读引用，不得写入**。
使用前必须重新核验 sha256 并写进审计文件；并执行 §3 的 **C8**（CSV 与 parquet 等价断言）。
若 C8 不通过，改为上传本地 parquet 并说明理由。

### 4.3 【v4 修正】sbatch 与依赖（照 K2 完整模板，**不得再漏 PYTHONPATH**）

**冻结环境没有 pyarrow**（实测 Python 3.9.24 / numpy 1.21.6 / scipy 1.7.3 / pandas 1.4.4，
`import pyarrow` 失败）。K2 的做法是装一份到 run-dir 的 `pydeps/` 再用 `PYTHONPATH` 接入
（见 `stage_and_submit_k2_fullcorpus.sh:65-69`）。**照抄，并把版本钉死为 `pyarrow==12.0.1`。**

staging 阶段：

```bash
PYTHON=/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python
PYTHONPATH="${HPC_WORKDIR}/pydeps" "$PYTHON" -c 'import pyarrow' \
  || "$PYTHON" -m pip install --no-input --no-deps --target "${HPC_WORKDIR}/pydeps" 'pyarrow==12.0.1'
```

sbatch：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=zxc-rq017-m1          # 必须 zxc- 前缀
#SBATCH --partition=intel,fata            # amd 不得使用
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-<N>%<M>
#SBATCH --output=logs/zxc-rq017-m1_%A_%a.out
#SBATCH --error=logs/zxc-rq017-m1_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# matplotlib font-cache 并发锁与 XDG 缓存：每个 task 独立目录（K2 曾因此整轮作业被取消）
export RQ017_PROCESS_CACHE_BASE="${SCRIPT_DIR}/process_cache"
export MPLCONFIGDIR="${SCRIPT_DIR}/process_cache/mpl/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
export XDG_CACHE_HOME="${SCRIPT_DIR}/process_cache/xdg/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "${SCRIPT_DIR}/repo_stage"
PYTHON="/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python"
export PYTHONPATH="${SCRIPT_DIR}/pydeps:${PWD}/src:${PWD}/.codex-fleet/rq015b-repair/work:${PYTHONPATH:-}"
```

**分片数与并发上限自己现算**，不要照抄 K2 的 `1-460%427`（那是 4,981,984 个单元，本轮只有
67,861，是其 **1/73**）。并发上限必须**逐节点装箱现算**
（`sum_nodes min(floor(idle_cpu/workers), floor(free_mem/mem_per_shard))`），
**禁止用「分区空闲核总数 ÷ 每片 worker 数」**。分片依据写进报告。

⚠ 提交前查 `sinfo`：监督方 2026-08-04 观察到 `intel` 4 个空闲、183 个 down，`fata` 0 个空闲。
**排队等待属正常，不得因此改投 amd。**

### 4.4 登录节点纪律

重计算一律 `sbatch`；preflight 与校验类轻量脚本可在登录节点跑。
**出现任何密码提示立即停止并报告**，不得输入、存储或打印密码。

---

## 5. 【blocker，v4 强化】环境同源硬断言

**必须给出正面证据证明与 K2 同栈，不是仅仅「也在 HPC 上跑」。任一条不符即停，不得跑全量。**

**5.1 版本逐项断言**（实测值，不符即停）：

```
python  3.9.24 ｜ numpy 1.21.6 ｜ scipy 1.7.3 ｜ pandas 1.4.4 ｜ pyarrow 12.0.1（经 pydeps）
```

**5.2 import origin 断言**：打印并断言 `numpy`、`scipy`、`pandas`、`pyarrow`、
`sociality_estimation.core.ipv_estimation` 的 `__file__` 实际来自冻结 env / pydeps / repo_stage，
**不是来自登录节点的系统 python 或用户 site-packages**。

**5.3 受保护文件 SHA blocker**：`agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、
`process_interhub.py`、`ipv_sigma01_exact.json` 在 repo_stage 中的 sha256 必须与本地一致。

**5.4 G 锚点逐位复算**：在本轮环境下重算一小批锚点，
**要求 `mse_per_candidate[7]` 逐位相同、`max_abs_diff = 0.0`**。两份基线的**确切路径**：

```
✅ 用这个（HPC 版）  .codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv
                     .codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json
❌ 不要用（Mac 版）  .codex-fleet/rq015b-repair/work/anchor_mse.csv
```

⚠ **上一轮有人比错过基线，把 Mac 版当成 HPC 版，导致误判为数值缺陷。**
比对前先断言你打开的是上面那条 `rq015g-hpc-resolve` 路径。

**5.5 事后分区断言**：全量跑完用 `sacct` 断言所有 array task 的分区**不含 `amd`**，
节点清单写进运行回执。

---

## 6. 安全边界

### 6.1 冻结事实（直接引用，不要重算后报略不同的数）

```
OnSite anchor 表   67,861 行 × 66 列，全部 av_included == "AV"
OnSite K2 台账     281,268 行；status/reason_code/mse_0..6 非空全为 0；gate_applicable 全 False
InterHub 求解单元  4,981,984：OK 3,502,340（70.3001%）/ NEAR_UNIFORM 1,457,746（29.2604%）
                   / NO_IPV_EFFECT 19,964（0.4007%）/ SOLVER_FAILURE 1,934（0.0388%）
RQ016C 支持门      OnSite 通过 21,936/67,861 = 32.3249%
```

### 6.2 held_out 护栏**只能收紧不能放松**

`run_b2_rq015b.py:271` 的 `ALLOWED_SPLITS = {"development","guard"}` 会拒绝 OnSite 行
（OnSite 不在 RQ007 划分体系里）。**不得删除、注释或放宽。** OnSite 走 §2.3 独立入口。
**该断言必须是脚本 blocker**，实测：参与求解的行中来自 InterHub 的为 **0 行**、
携带 RQ007 `held_out` 标记的为 **0 行**。

### 6.3 参考线 fail-closed

去重后观测轨迹点数 **< 2 时不得继续写 OK**。既有实现
`build_onsite_m3_anchors_hpc.py:705-710` 的 `prepared_reference()` 抛
`ValueError("observed reference has fewer than two unique points")`。保留并列为 canary 用例。

### 6.4 7 行坐标系异常必须定性，不得静默丢弃

67,861 行中 **7 行（0.0103%）** 的 `relative_distance_anchor` ≈ 570,761.6 米，
由单侧 `relative_dx_anchor` ≈ −570,761.6 导致，`relative_dy_anchor` 正常（约 −8.8 米）。
查清是坐标原点不一致还是缺失值哨兵，单独成节说明。**照常参与求解、必须出现在正式产物中**；
求解失败按工程失败记录。

---

## 7. canary（§3 与 §5 全绿后才做；canary 全绿后才投全量）

**canary 必须走 sbatch，且使用与全量完全相同的 wrapper、`PYTHONPATH`、cache 设置与
parquet writer。** 必须覆盖：

1. **四种状态各至少一行**：`OK` / `NEAR_UNIFORM` / `NO_IPV_EFFECT` / 一种工程失败
2. **至少 2 个并发 array task**（K2 的失败全在并发路径上）
3. **写出后读回校验**，含 null scalar 的落盘/读回模式
4. **§6.4 的 7 行真实坐标异常行**
5. **§6.3 的参考线 fail-closed 路径**（构造去重后点数 < 2 的输入）

自然样本不出现的状态，**构造合成输入**触发（七候选 MSE 完全相同 → `NO_IPV_EFFECT`；
含 NaN → 工程失败）。**合成 sentinel 必须与自然样本分文件输出，正式产物 manifest 明确排除
sentinel**，并在报告中声明。

**K2 的三次失败必须逐一验证已规避**（作业 `2069424`、`2069818`、`2071368`）：
Matplotlib font-cache 并发锁、PyArrow fixed-size-list 无法写 null array 行、
逐行重算源文件 SHA 过慢。**canary 不全绿不得投全量。**

用 canary 实测分片速率再外推全量墙钟，写进报告。

---

## 8. 交付物与自查

产物先落 HPC `outputs/`，取回后落 `data/derived/rq017_onsite_gate/l1_v1/`（parquet，可分片），
schema 与 K2 台账 `data/derived/rq015k_logdomain_gate/l1_v1/` 对齐，另加 §C6 四个帧语义字段。

- 报告：`.codex-fleet/rq017-onsite-materializer/board/reports/RQ017_1_onsite_materializer.md`
- `work/M1/measurement_contract.json`（§3）｜`env_parity.json`（§5）
- `work/M1/key_numbers.json`｜`run_receipt.json`（作业号、分区、节点、array 形状、
  逐分片行数与耗时、失败计数、输入与代码哈希、`sacct` 输出）
- 脚本放 `work/M1/`，可复跑

### 8.1 【v4 新增】机制二交叉的连接合同

报告要给出「与 RQ016C 支持门交叉后最终可判行数」。**做法限定如下**：

- **只从 `.../H2/onsite_scoring_dryrun.parquet` 读两列：`product_row_key`、`mechanism2_gate_ok`**，
  用显式 `columns=[...]`。
- **禁止调用 `score_external_rows.py` 或任何整表 `read_parquet` 的重新打分入口**——
  anchor 表含 `target_ipv_future` 等目标值，整表读入会违反 C7 白名单，
  而机制一的数值检查抓不到这种违规。
- 交叉只做计数，不重算 envelope、不产出区间。

### 8.2 自查（一轮，但必须有牙齿）

1. **行数守恒**：产物行数 == 67,861；**且** `product_row_key` 唯一、§C1 三方交集成立
2. **状态守恒**：`OK + ABSTAIN + 工程失败 == 总行数`；`NEAR_UNIFORM + NO_IPV_EFFECT == ABSTAIN`
3. **门判据可复算**：从落盘 `mse_0..6` 用冻结规格重推 `status`/`reason_code`，零处不一致
4. **恒等式**：OK 行上 `ipv_log == sum(candidate_ipv_i * w_log_i)`、`k_eff_log == 1/sum(w^2)`；
   **另须单独断言 `K == 7` 与网格 ID == `legacy7_pi_over_8`**（恒等式抓不到整体错用 5 点网格）
5. **工程失败隔离**：工程失败行被记成两个科学 reason 之一的行数为 0
6. **blocker 齐备**：§3 测量合同、§5 环境同源、§6.2 护栏，三者都必须是会退出非零的脚本
7. **负对照（强制两条，必须真的 FAIL）**：把 `mse_spread == 0` 换成 `np.isclose(atol=1e-12)`
   → 必须 FAIL；把 `theta` 从 0.20 改成 0.22 → 必须 FAIL。
   ⚠ **自然输出不保证含敏感行，必须注入合成 sentinel** 保证两条都能 FAIL。
   任一条没 FAIL 说明检查是坏的，**先修检查再继续**，并说明怎么修的。
8. **数值健康**：NaN/inf 计数；`max_w_log ∈ [1/7, 1]`；`mse_spread >= 0`；`k_eff_log ∈ [1, 7]`
9. **取回完整性**：远端与本地逐分片行数与 sha256 一致
10. **不覆盖检查**：本地与远端产物目录若已存在内容，停下报告，不得覆盖

### 8.3 报告要回答

67,861 行的四类状态分布（各带分子分母）；与 InterHub 全语料对照；
与 RQ016C 支持门交叉后**最终可判的行数**。

⚠ **禁止对任何一辆车下「像不像人」的结论。**

---

## 9. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py ｜ configs/ipv_sigma01_exact.json
不改 .codex-fleet/rq015b-repair/ 与 rq015k-fullcorpus-gate/ 下任何文件（只读复用）
不改 RQ009 / RQ016 / RQ016C 已落盘 run 目录（只读）
不改 data/derived/ 已有内容；本轮只新建 data/derived/rq017_onsite_gate/
HPC 侧只在 work_dirs/RQ017/<run_id>/ 下写；rq012b_onsite_ipv_* 目录只读引用
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析；护栏只能收紧不能放松（§6.2）
RQ014 致盲相关的评分字段不得读取；输入列白名单见 §3 C7 与 §8.1
不得使用 amd 分区｜不得在登录节点跑重计算｜作业名必须 zxc- 前缀
出现密码提示立即停止并报告
不得静默覆盖已冻结产物或已接受的 decision.md
不要对 reports/ 做全仓库 rg；用定向 ls/grep
git status 一律用 git --no-optional-locks status --porcelain
时间戳一律实取 date -u +%Y-%m-%dT%H:%M:%SZ
```

**措辞禁令**：禁用 `estimability` 与「测出/未测出 IPV」。可辩护表述是
**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。描述性结果不得写成因果主张。
不用比喻、不用自造简称。

**分母纪律**：每个百分数必须紧跟分子、分母、筛选条件、来源文件与列名。
在流通的分母：67,861（OnSite anchor）、281,268（OnSite K2 台账行）、
4,981,984（InterHub 求解单元）、8,994,736（RQ009 台账行）。

本机环境：python3 已有 pyarrow 21.0.0 / pandas 2.3.3（**与 HPC 不同，见 §5.1**）。

## 10. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```

报告开头先定位，写给完全没跟进过程的读者。需要监督方拍板的事单独成节。
