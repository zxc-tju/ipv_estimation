# K2-1 — 全语料重算：物化 InterHub 的 log 域门判据台账

你是 codex agent **K2-1**。这是 track K2 的**唯一执行体**。
仓库根（已是你的 `--cd`）：
`.`

**PI 已授权全量执行。** 本 prompt 是你的唯一依据；与任何早前文档冲突时以本文为准。

---

## 第 0 节 铁律（先读这一节，违反即为任务失败）

```
禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
禁止 git checkout 任何历史提交到主工作区（要看旧代码用 git worktree add）
禁止 git commit / git add / git push（本轮产物由 PI 统一提交）
工作区非空是【预期状态】——此前 A/B/…/K1b 各轨留下的文件仍在。
你只对自己创建/修改的文件负责；清洁性检查只查自己的文件清单，不看全仓库 git status。
```

**不得修改这五个受保护文件**（新增 writer/materializer 可以，改计算不行）：

```
src/sociality_estimation/core/agent.py
src/sociality_estimation/core/ipv_estimation.py
src/sociality_estimation/core/reliability_logdomain.py
pipelines/interhub/process_interhub.py
configs/ipv_sigma01_exact.json
```

四条效度硬约束（与流程无关，不得放松）：

```
1. RQ007 held_out 不得被解析（污染不可恢复）——最终必须自证 held_out_parsed_rows = 0
2. RQ014 致盲相关的评分字段不得读取
3. 不得静默覆盖冻结产物或已接受的 decision.md
4. 描述性结果不得写成因果主张
```

其他：

- **报告与代码注释全文禁用 `estim` + `ability` 拼成的那个词，以及「测出 / 未测出 IPV」的说法。**
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
  （注意：L1 ledger 的磁盘路径里含该词，路径本身照用，但**不要把它写进报告正文**；
  取路径的正确做法见第 2 节。）
- 时间戳一律现调 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不得取整、不得前瞻估计**。
  此前轨道出现过比墙钟早 12 分钟的日志行。
- **不要对 `reports/` 做全仓库 `rg`。** 宽泛检索会把 RQ003
  `12_blind_annotation/controlled_identity_map.csv` 的 controlled-access 行整行拉进上下文。
  只读第 2 节列出的路径。
- HPC 上若出现任何**密码提示，立刻停止并上报**，不得输入、存储或打印口令。
- 每次投递用**新建** work_dir，**不得覆盖任何既有目录**。

---

## 第 1 节 这项工作是什么（先定位，再干活）

最终用途是 **online verification**：判断一辆自动驾驶车的 IPV 是否落在人类分布内。
IPV = Interaction Preference Value，社会交互倾向的标量参数。

判断链路上有**两个串联的弃权机制**：

- **机制一（本任务要物化的）**：判「这一帧的 IPV 到底能不能算出有意义的值」。
  弃权则直接结束，不进机制二。
- **机制二**：RQ009 已 accepted 的 envelope 支持度判据。本轮**不动它**。

机制一的规格（下称「门」）已由 track J 冻结。现在的问题是：
RQ015A 的 4 份 L1 parquet ledger 共 14,473,982 行，**里面没有** `mse_per_candidate[7]`、
`log_score[7]`、`w_log[7]` 这三组逐候选量，所以门算不了。

**K2 就是把这三组量在全语料上重算出来，落成一张行级台账。**
有了它，RQ009 才能筛出可用的人类样本建 envelope，在线检测才能逐帧调用同一判据。

前置工作已完成并经独立复算：

- **K1**（勘察）：确定了求解单元口径、重建入口、输出契约。报告见第 2 节。
- **K1b**（内存 pilot）：确定了分片形状与吞吐。报告见第 2 节。
- **K2R-A / K2R-B**（对任务书的双盲复审）：已修正三处监督方错误，结论已并入本 prompt。

**本轮编制**：只有你一个 agent，一轮自查，出报告，结束。
**不做盲审、不做多路复审、不出第二版规格、不加授权闸门。**
发现自己在写「规格 v2」或第 8 版计划就是跑偏了，停下——上一轮一个描述性审计走了
8 个计划版本、7 轮盲审、32 个 agent，科学结论产出为零。那是反面案例。

---

## 第 2 节 必读文件清单（**只读这些，不要自己摸索仓库**）

前一轨有 agent 把 31 次 exec 全花在摸索仓库上，一行计算都没跑就被杀了。**不要重蹈。**

**必读（按顺序）**：

| 路径（相对仓库根） | 你要从中取什么 |
|---|---|
| `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md` | §1 canonical key 定义与行数表、§5 分片/续算/重投规则、**§9 完整输出契约 9.1–9.9（schema 以此为准）** |
| `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md` | §1 HPC 冻结路径与作业形状、§2 P6/P10/P16 实测表、§3 逐位一致性口径 |
| `src/sociality_estimation/core/reliability_logdomain.py` | **第 36-49 行**状态枚举、**第 172-188 行** `weights_from_mse(mse, sigma)`。**只读，不改** |
| `.codex-fleet/rq015k-fullcorpus-gate/work/k1_local_preflight.py` | **第 18-30 行** `LEDGER_PARENT` 与 `ARTIFACT_FILES` 的构造方式。**直接复用这段常量定义**，不要自己拼路径 |
| `.codex-fleet/rq015k-fullcorpus-gate/work/k1_build_pilot_inputs.py` | K1 怎么从 ledger + 冻结 PKL 构造求解输入（**这是你 solver driver 的直接前身，尽量复用**） |
| `.codex-fleet/rq015k-fullcorpus-gate/work/build_k1b_single_pkl_sample.py` | 单 PKL 取样与 split 过滤写法 |
| `.codex-fleet/rq015k-fullcorpus-gate/work/fetch_k1b_memory_pilot_outputs.sh` | 回取产物的 rsync/ssh 写法（`tongji-hpc`、`BatchMode=yes`） |
| `.codex-fleet/rq015k-fullcorpus-gate/work/hpc_frozen_pkl_listing.tsv` | 15 个冻结 PKL 的文件名与字节数 |
| `<local-projects-root>/1_Codes/HPC_TONGJI_USAGE_GUIDE.md` | 提交前检查表、sbatch 模板、目录规范 |

**按需读（不要通读）**：

- `.codex-fleet/rq015b-repair/work/anchor_mse.csv` — G 轨锚点 `mse_per_candidate[7]` 参考表，
  第 6 节数值 canary 用。若本地不存在，HPC 副本在
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_anchor_resolve_20260801T012113Z/repo_stage/.codex-fleet/rq015b-repair/work/anchor_mse.csv`
- `.codex-fleet/rq015k-fullcorpus-gate/work/k1_pilot_summary.json` — K1 pilot 的
  `g_anchor_overlap` / `worker_memory` / `thread_env` 口径
- `.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/k1b_consistency_summary.json`

**不要读**：`board/K2R_*.log`、`board/K1*.log`（都是几 MB 的 agent 日志，没有你要的东西）。

---

## 第 3 节 门的规格（**冻结，一个字不许改**）

```text
输入（单帧内可得）: frame_id, candidate_grid_id, K=7, candidate_ipv[7],
                    mse_per_candidate[7], log_score[7]

log_score_i = -mse_i / (2 * sigma^2)     其中 sigma = 0.1   ← 必须钉死
w_log       = softmax(log_score)          用 log-sum-exp（复用现成实现）
mse_spread  = max(mse_per_candidate) - min(mse_per_candidate)
max_w_log   = max(w_log)
k_eff_log   = 1 / sum(w_log_i^2)

if 输入非有限 / 缺列 / 求解失败:
        status=ENGINEERING_FAILURE, reason_code=NON_FINITE_INPUT 或 SOLVER_FAILURE,
        ipv_log=null, w_log=null
elif mse_spread == 0:   status=ABSTAIN, reason_code=NO_IPV_EFFECT, ipv_log=null
elif max_w_log < 0.20:  status=ABSTAIN, reason_code=NEAR_UNIFORM,  ipv_log=null
else:                   status=OK, reason_code=null,
                        ipv_log = sum(candidate_ipv_i * w_log_i)
```

逐条强制：

- `sigma = 0.1`、候选网格 `legacy7_pi_over_8`（7 点 `[-3..3]·π/8`）、`K = 7`：**全部不得改**
- `theta = 0.20` 是**政策阈值，不是数据断点**。**不得调参、不得做阈值扫描、不得提替代判据**
- **必须在 log 域算。** 连乘域下溢会把本可算的行错判为不可算
- `mse_spread == 0` 是**精确浮点相等**：先断言 7 个 MSE 均为**有限 float64 且长度恰为 7**，
  再判 `max - min == 0.0`。**不得用 `np.isclose`**
- 两条科学 reason **互斥且有序**：先 `NO_IPV_EFFECT`，否则再 `NEAR_UNIFORM`。
  向量化实现必须用**有序 `np.select`**，或「先写 1，再只在仍为 null 处写 2」。
  ⚠ **两个布尔掩码顺序覆盖会写反，而且看不出来**：J 轨样本内 400 个 `mse_spread==0` 的行
  **全部也满足** `max_w_log<0.20`，写反了两个计数都还是「合理」的。**必须写一个针对性单测。**
- **工程失败绝不允许被记成两个科学 reason 之一**
- softmax **复用** `src/sociality_estimation/core/reliability_logdomain.py`
  第 172-188 行的 `weights_from_mse()`（已是稳定实现：减最大值后最大项恒为 exp(0)=1，
  分母 ≥ 1，永不为零）。**不要另写一个。**
  该函数签名 `weights_from_mse(mse, sigma)` 两个参数**皆必填、无默认值**，
  所以 **`sigma=0.1` 必须由你显式传入**，并写进每一份 manifest。
- 状态枚举与 `reliability_logdomain.py` 第 36-49 行的 `STATUS_NON_FINITE_INPUT` /
  `STATUS_SOLVER_FAILURE` **对齐**，不得在批处理侧另起一套写法。

---

## 第 4 节 范围与四类行（PI 已裁定，不要重新讨论）

「全语料」在本轮**一律限定为「RQ015A 当前 4 份本地可审计 L1 parquet ledger 的 InterHub 部分」**。
**不得**在任何地方写成「全 WOD」或「全项目全语料」。

| 项 | 决定 |
|---|---|
| InterHub | **做**，4,981,984 个求解单元（`attempt_status == ATTEMPTED`） |
| RQ009 feature matrix | **不重解**，8,994,736 行**全部由 InterHub 结果 join 回填**（K1 干跑 512/512 精确一对一，0 漏 0 重） |
| OnSite 2,974 / WOD 906（`ATTEMPTED`） | **本轮不做**。写 `gate_applicable = false`，保留 `source_attempt_status`/`source_reason_code` |
| 非 `ATTEMPTED` 行 219,360 `NOT_ATTEMPTED` + 274,022 `UNKNOWN` | 同样写 `gate_applicable = false` |

**四类行必须在最终 L1 产物里全部有归属分片和预期行数**（复审 A 判定这是最可能造成实际损失的一处：
只按 solve shard 判完成，产物会「看起来完成」而缺失后三类）：

| 类 | 行数 | 处理 |
|---|---:|---|
| A. InterHub solve 行 | 4,981,984 | 真算门 |
| B. RQ009 join 回填行 | 8,994,736 | 从 A join 回填，**不新增求解** |
| C. 非 `ATTEMPTED` 行（InterHub 215,088 + OnSite 278,294） | 493,382 | `gate_applicable=false` |
| D. OnSite/WOD `ATTEMPTED` pass-through | 3,880 | `gate_applicable=false` |
| **合计** | **14,473,982** | = 5,197,072 + 281,268 + 8,994,736 + 906 |

**C、D 两类绝不允许有任何一行落入 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。**

⚠ **一处必须区分清楚，否则你会误触发整体停止**：
K1 pilot 把 OnSite/WOD 记成了 `SCHEMA_MISMATCH`（contract-only）。
但第 5.6 节「表级缺列 → `SCHEMA_MISMATCH` 整体停止」**只适用于 InterHub solve 通道**。
OnSite/WOD 是 **PI 已裁定的范围外项**，必须用一个**独立的、非科学的**标记
（建议 `gate_applicable=false` + `source_reason_code` 保留原值 +
一个明确的 `out_of_scope_reason = NO_MATERIALIZER_IN_SCOPE`），
**不得触发整体停止，也不得记成两个科学 reason 之一**。

### 4.1 leader 裁定：RQ009 分区的数组存储（**必须在报告中单列一节说明**）

B 类 8,994,736 行**默认不复制** 4 个 7 元数组（`candidate_ipv` / `mse_per_candidate` /
`log_score` / `w_log`），改为存 **`interhub_canonical_key` 外键 + 全部门标量**
（`status`/`reason_code`/`ipv_log`/`max_w_log`/`mse_spread`/`k_eff_log`/`gate_applicable`）。

理由：join 是 exact-one，数组是可精确还原的冗余，复制约多占 2.0 GB。
**要求**：报告里单列一节写明这个取舍、给出从外键还原数组的确切方法，
并做一次 **1,000 行的还原验证**（还原出的数组与 A 类对应行逐位相同）。
若你发现该裁定会破坏第 6 节任一验收判据，**停下来在 `K2_progress.log` 写明并改为全量复制**。

---

## 第 5 节 资源、分片与投递

### 5.1 固定的部分

- **每片 6 worker（P6）。** 每 worker 吞吐 **0.4782319 units/s**（P16 为 0.4042，核效率低 18%）。
  **核·小时与分片形状无关，恒为 2,893.75 核·小时**（4,981,984 ÷ 0.4782319 ÷ 3600），
  所以在核数受限时 P6 严格优于 P16。
- **`--mem=48G`。这是明确的保守策略，不是公式结果，照抄，不要自己重算。**
  P6 实测 6 worker 合计峰值 RSS 16.337 GiB（每 worker 2,789.332 MB），48G 是 **2.94 倍余量**。
  （备注供你知情，**不要据此改数**：任务书同节给的自适应公式
  `ceil_to_8G(2.789 GiB × workers × 3.0)` 代入 P6 得 56G，与固定值 48G 差 8G。
  本轮以 **48G** 为准；把这处 8G 差异**如实写进报告的偏差节**，不要自行调整。）
  **OOM 是本项目唯一真正炸过的失败模式，不要压缩这个余量。**
- **分片粒度固定、并发不固定。** 按 `(单个 PKL, 行键区间)` 切，
  每片目标 **10,000–12,000 canonical units**（约 415–498 片）。
  一个节点可承载多片；**不得沿用 K1 报告的 `one node per shard` 写法**（那是已被推翻的）。
- 单片预计 11,000 / 2.869391 ≈ 3,834 s ≈ 1.07 h。**`--time=04:00:00`**（3.75 倍余量）。
- 线程钉死 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
  （**不设会破坏确定性**）。

### 5.2 投递时必须现算的并发（**不得从任何文档抄片数**）

投递前重查集群快照，按**逐节点装箱**计算：

```
slots       = sum_over_nodes  min( floor(idle_cpus_node / workers),
                                   floor(free_mem_MB_node / mem_MB_per_shard) )
concurrency = min(slots, floor(4000 / workers))      # QOS 合计核上限 4000
```

只统计 state **不含** `down` / `drain` / `inval` / `drng` 的节点。
Slurm array 用 `%concurrency` 限流；并发不足只会多跑几波，**不影响任何数值结果**。

🚫 **禁止用「分区空闲核总数 ÷ 每片 worker 数」**——那正是被两位复审方独立判定为错误的算法。
Slurm 单个作业步必须落在**单个节点**上，所以必须逐节点 `floor` 后求和。
朴素法在实况下高估并发 5.8%。

参考量级（2026-08-02T12:15:01Z 实况，`intel`+`fata`，51 个可用节点、2,838 空闲核，
三方独立算得同值）：**P6/48G → cpu_only=457 / mem_only=664 / slots=447 / 2,682 活跃核 / 1.079 h**。
**这是参考，不是你要用的数。你必须自己现算并把快照原文存档。**

⚠ 实况风险：`intel` 的 232 个节点中曾有 183 个 `down`，可用只有 51 个。
投递时若可用节点更少，按上式现算即可，**墙钟相应拉长、核·小时不变**。

### 5.3 分区范围：**只用 `intel` + `fata`，不得放开 `amd`**（监督方裁定）

集群另有 `amd` 分区（空闲核约 19,088），放开后墙钟可降到约 0.72 h。**本轮不放开**，两条理由：

1. G 轨的逐位一致性证据（Slurm 2024766，348/348）只覆盖 `fata` 与 `intel` 两类节点。
   放开第三个分区属于把确定性证据**静默外推**。
2. 收益约 21 分钟墙钟，代价是一个未经验证的数值通道加一轮 canary。不划算。

sbatch 里必须显式写 `--partition`，且**不得**包含 `amd`。

### 5.4 HPC 通道（**复用，不要另造**）

- 冻结环境：`/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python`
  （Python 3.9.24 / numpy 1.21.6 / scipy 1.7.3 / OpenBLAS）
- 冻结 PKL 快照：`.../snapshots/interhub_legacy_20260711_v1/full_datasets/pkl`
  （15 个文件，共 1,856,362,564 字节）。**禁用 `subsets_for_yiru/pkl`**
- managed checkout：`6bdcc2e6`
- SSH alias `tongji-hpc`；HPC 工作根 `/share/home/u25310231/ZXC`
- **重计算只经 `sbatch`，绝不在登录节点直跑**
- job name 必须以 `zxc-` 开头（建议 `zxc-rq015k-k2`），log 名与之对应放 `logs/`
- 新建 work_dir：`/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k2_fullcorpus_<UTC>/`
- 提交前先 `scontrol show config | grep -i MaxArraySize`，确认片数不超过 array 上限；
  超了就分多个 array 作业，**不要靠减少片数来规避**

### 5.5 分片 manifest（**每一项都必须有**）

`shard_id`、`artifact_scope`、`pkl_file_list`、`source_dataset`、`row_key_min`、`row_key_max`、
`canonical_key_count`、`expected_output_rows`、`input_ledger_sha256`、`input_pkl_sha256`、
`code_sha`、`command`、`sigma`、`candidate_grid_id`、`created_utc`。

**同一 canonical key 只能出现在一个分片里。**

### 5.6 断点续算与重投

先写 `<shard>.tmp.parquet` 与 `<shard>.tmp.manifest.json`，校验后**原子 rename**。

- **rename 顺序：先 parquet、后 manifest。manifest 是唯一完成标记。**
  两个 rename 不构成一个事务；下游必须先核 manifest 才认 parquet。
- 一个分片算完成，必须 manifest 的**输入 SHA、PKL SHA、代码 SHA、命令、sigma、grid、
  canonical 行数、预期输出行数、输出 SHA 全部匹配**。**文件存在不等于完成。**
- 重投必须幂等：已完成且 manifest 匹配则跳过；
  最终文件存在但不匹配 → 按 `SCHEMA_MISMATCH` **硬停**。

失败分类与重投：

| 类型 | 处理 |
|---|---|
| `SCHEMA_MISMATCH` | **立即整体停止**，报监督方（写进 `K2_progress.log` 与 `STATUS` 段） |
| `OOM` | 该片降到 3 worker 重投**一次**；再 OOM 则停止并报 |
| `TIMEOUT` | 该片加倍墙钟**或**减半行区间重投**一次**；再超时则停止并报 |
| `SOLVER_FAILURE` | 只重跑失败行**一次**；单片失败行 > 100 行或 > 2.0%（**取小**）则停止 |
| `NON_FINITE_INPUT` | **不盲目重试**；分类记录，单片 > 0.1% 则停止并报 |

### 5.7 输出 schema 强制要求

- **7 元数组的 element dtype 必须是 double（float64）。**
  `mse_spread == 0` 是精确浮点相等，任何 float32 降精度都会改变判定。
- 数组列优先用 Arrow fixed-size list(7)；若 writer 不支持，退化为
  `mse_0..mse_6` / `log_score_0..6` / `w_log_0..6` / `candidate_ipv_0..6` 标量列，
  但 schema 必须声明候选顺序为 `[-3,-2,-1,0,1,2,3] * pi/8`。
- **表级缺列 → `SCHEMA_MISMATCH` 整体停止**，不得降级成逐行工程失败继续产出。
- **行级 list 为 null / 为空 / 长度 ≠ 7 → 明确归入 `NON_FINITE_INPUT`**，不得留作未定义。
- L1 schema version：`rq015k_logdomain_gate_l1_v1`。
  完整列清单照 **K1 报告 §9.8**（含 `artifact_id`、canonical row key、`product_row_key`、
  `measurement_role`、`case_id`、`rq007_split`、`frame_id`、`context_cell_key`、
  `candidate_grid_id`、`K`、四个 7 元数组、`max_w_log`、`mse_spread`、`k_eff_log`、
  `status`、`reason_code`、`ipv_log`、`gate_applicable`、`source_attempt_status`、
  `source_reason_code`、legacy `ipv_error`/`k_eff`/`q_eff`、`solver_status`、`failure_type`、
  `shard_id`、`input_sha256`、`code_sha`、`created_utc`）。
- `gate_pass_rate` 放聚合表，**不要重复进每一行 L1**。

---

## 第 6 节 验收判据

### 6.1 普查完整性（**任一条不过即为未完成**）

1. **InterHub 覆盖**：输出 canonical key 数 = **4,981,984**，缺失 0，重复 0
2. **RQ009 回填**：**8,994,736** 行全部 exact-one join，misses 0，duplicates 0，**新增求解行 0**
3. **非科学范围行**：`NOT_ATTEMPTED` 219,360 + `UNKNOWN` 274,022 + OnSite 2,974 + WOD 906
   全部有归属分片、全部 `gate_applicable = false`，**无一行落入两个科学 reason**
4. **manifest 对账**：每片的输入 SHA / PKL SHA / 代码 SHA / 命令 / sigma / grid /
   canonical 行数 / 预期输出行数 / 输出 SHA **全部匹配**
5. **逐行门不变量（必须从产物重算，不是相信 writer）**：
   `K = 7`；数组长度 7 且 dtype 为 double；`|sum(w_log) - 1| ≤ 1e-12`；
   `max_w_log ∈ [1/7, 1]`；`k_eff_log ∈ [1, 7]`；reason 顺序正确；
   非 OK 行的 `ipv_log` 与 `w_log` **均为 null**；OK 行的 `ipv_log` 有限
6. **`held_out_parsed_rows = 0`**

### 6.2 数值 canary（必过，但不充分）

**G 锚点重叠逐位一致。** 比较口径必须是：按 canonical key 对齐后
**解析为 float64 逐位比较**，**不得用 CSV 字符串相等**（字符串比较会把格式差异误报成数值差异）。
不同则**停止并报**。

**worker 数不变性**：本轮固定 P6，K1b 已证 P6/P10/P16 在 1,120 行上零不一致，无需重跑。
**若投递时你改了 worker 数，必须重跑同样的抽查。**

### 6.3 解释性对照（**不是验收判据，不触发停止**）

J 轨的全域可算率 **71.2695%**、CI **[67.1729%, 75.2135%]** 是**设计基抽样估计**，
分母是 2,646,058 的 HT 权重、2,300 个锚点、1,909 个 cluster。
K2 是**行级普查**，分母是 4,981,984 个 canonical solve unit（另有 8,994,736 个 join 行）。
**两边的域与分母不是同一个东西，落不落在该 CI 内都不能判定 K2 成功或失败。**

正确用法：把 K2 普查结果与该区间**并列报出，差异只需解释，不触发停止**。
两条 reason 的全域权重占比参照：`NO_IPV_EFFECT` **0.5095%**、`NEAR_UNIFORM` **28.2210%**。

---

## 第 7 节 必须随台账交付给 RQ009 的一条警告（**写进接口说明，不能只留在报告正文**）

J 轨实测：**门后（`status=OK`）的行里有 23.40% 的 `ipv_log` 恰好为零**
（判定口径 `|ipv_log| <= 1e-9`；分 signature 为 N 12/363、U 91/511、Z 135/143）。

本门保证「弃权」不再写成 `ipv=0`，但**反过来不成立**：
`ipv_log = 0` 仍是**合法且高频的通过门的估计值**（中性社会倾向）。

> **判别字段只能是 `status` 与 `reason_code`，不是 `ipv_log` 的数值。**
> 下游代码不得把数值 0 当作弃权。

这条必须同时出现在：产物旁的 `INTERFACE_NOTE.md`、L1 schema 的字段描述、以及报告正文。
（旁证：RQ009 `decision.md` 的 Boundaries 记有 `Target exact-zero atom ~21.6%`，量级一致。）

---

## 第 8 节 交付物

**本地工作区（新建，不得覆盖）**：
`.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/`

必须产出：

1. **`.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md`** — 主报告
2. `work/k2_fullcorpus/shard_manifests/` — 全部分片 manifest（含四类行的归属账）
3. `work/k2_fullcorpus/validation/` — 第 6 节每一条判据的机器可读输出（JSON/CSV）
4. `work/k2_fullcorpus/cluster_snapshot_pre_submit.txt` — 投递前快照原文 + 你的逐节点装箱现算过程
5. `work/k2_fullcorpus/INTERFACE_NOTE.md` — 交给 RQ009 的接口说明（含第 7 节警告）
6. **L1 产物**：权威副本留在 HPC work_dir；
   本地落到 `data/derived/rq015k_logdomain_gate/l1_v1/`（该目录已 gitignore）。
   先测产物总字节数：**≤ 8 GB 则全量 rsync 回本地**；
   **> 8 GB 则回取 A 类 + 聚合表 + manifest + validation**，并在报告里
   **明确写出总字节数、已回取部分、未回取部分与其远端权威路径**。
   ⚠ **不得静默截断**——「没回取」必须写出来。
7. 聚合表：按 `artifact_id` × context 键的 `gate_pass_rate` 等汇总
8. `board/reports/K2_progress.log` — 见第 9 节

**报告写作要求（PI ruling 2026-08-01，硬性）**：

1. **先定位，再讲进度。** 开头交代三件事：这项工作要解决什么问题、整体走到哪一步、本次是哪一环。
2. **不用黑话、不用比喻。** 必须用项目专有名词时，当场一句话说明它是什么。
3. **结论与待决事项分开成节。** 需要上级拍板的写清选项、判断依据、不做的后果。
4. **数字自带口径。** 任何百分比必须同时给**分子、分母、筛选条件、来源文件与列名**。
   读者无法自行复算的数字等于没给。

---

## 第 9 节 汇报与结项

**每完成一个阶段追加一行到 `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_progress.log`**：

```
<现调的 UTC 时间戳> | <阶段> | 做了什么 | 结论
```

阶段建议：`prep` → `manifest` → `submit` → `monitor` → `join` → `validate` → `report`。
**投递后每轮 squeue 轮询也写一行**，让 leader 能看到你还活着。

**预计墙钟约 1.1 小时**（按你现算的并发为准，不是固定值）。
**若超过 4 小时仍未完成，在 `K2_progress.log` 写明原因并停下上报，不要无限重投。**

### 结项自证（**报告最后一节，缺一项即为未结项**）

1. 五个受保护文件的 **SHA-256 原样列出**
2. `git --no-optional-locks status --porcelain` **输出原样贴出**
3. 本轮**全部 Slurm 作业号与各自规模**（片数、单元数、`--mem`、`--time`、分区、状态、Elapsed）逐个列出
4. `held_out_parsed_rows = 0` 的证据来源（文件与字段）
5. 确认**未做任何 git commit**、**未修改任何受保护文件**、**未放开 `amd` 分区**

---

## 第 10 节 明确的反模式（出现即停下来问自己是否在浪费时间）

- 计划连出 2 版以上却没有新的事实进来
- 为一次尚未运行的分析做多轮盲审
- 为描述性产物建多重授权闸门
- 把「程序对不对」的验证做到远超「结论对不对」的验证
- 用治理文书替代实际产出

遇到「要不要再加一道保险」的犹豫，**默认选不加，先把结果跑出来**。
唯一的例外是第 0 节那四条效度硬约束和第 5.6 节的 OOM 内存余量——那些是边界，不是流程。

**现在开始。第一步：读第 2 节的必读清单，然后写 `K2_progress.log` 的第一行。**
