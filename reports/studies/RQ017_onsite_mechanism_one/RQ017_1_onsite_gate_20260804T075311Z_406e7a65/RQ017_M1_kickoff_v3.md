# RQ017-M1 任务书 v3：OnSite materializer（HPC 版）——为 67,861 个自动驾驶车锚点产出机制一判据

> **版本沿革**：v1 从未执行。v2 经一轮独立双盲复审后新增「测量合同 preflight」，
> 但 v2 把 venue 写成「本机跑，不投 HPC」——**这是错的**，v2 因此也未执行。
> v3 = v2 的科学与验收内容**原样保留**，venue 改为 HPC 并按 K2 先例落实。
> v1、v2 原文保留，不删不改。

**venue 改为 HPC 的理由不是快，是产物来源一致性。** 实测：K2 那份 InterHub 台账
（本轮全部对照的基准）在 HPC 上产出；而 Mac 与 HPC 的求解结果在 2,300 个重合锚点上
`mse_per_candidate[7]` 字符串不同的有 **1,867/2,300 = 81.17%**，最大逐元素绝对差 **70.4**，
`argmin_candidate` 翻转 **686/2,300 = 29.83%**，差异来自软件栈而非 CPU
（同一 HPC 栈下跨节点逐位相同 348/348）。在 Mac 上产出的 OnSite 台账与 InterHub 台账
不可比，本轮的全部对照都会失去意义。

你是本轮唯一的执行 agent。执行顺序**不得颠倒**：

```
测量合同 preflight（§3，轻量，本机或登录节点即可）
  → 全绿才 staging（§4）
  → 环境同源核验（§5）——证明与 K2 同一软件栈
  → canary（§7，必须走 sbatch）
  → canary 全绿才投全量
  → 取回 + 自查（§8）→ 出报告
```

仓库根即当前工作目录，以下相对路径都相对仓库根；HPC 路径均为绝对路径。

---

## 0. 这件事在哪一步（不要跳过）

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。
IPV（Interaction Preference Value）是表示交互倾向的标量。判定由**两道串联的弃权机制**构成：

- **机制一**：这一帧的 IPV 数值能不能估？若七个候选的权重近均匀，则该数值不携带候选间的
  判别信息，直接弃权，不进机制二。规格已由 RQ015 冻结。
- **机制二**：当前场景收集到的人类样本够不够判断这辆车是否偏离？依据是人类参照分布（envelope）。

**已完成**：RQ015 冻结机制一并在 InterHub 全语料（HPC）跑出台账；RQ016C 用纯人-人样本
建好了供 OnSite 使用的人类参照 envelope，并验证过打分管线可在真实 OnSite 行上运行。

**唯一还缺的**：OnSite 一行都没有机制一判据——K2 台账
`artifact_id == onsite_dense_timeseries` 的 281,268 行中 `mse_0..mse_6`、`max_w_log`、
`mse_spread`、`status`、`reason_code` 非空计数**全部为 0**，`gate_applicable` 全为 False。

**本轮补上这一块。本轮不做机制二打分，不下任何关于「某辆车像不像人」的结论。**

## PI 已裁定（冻结，不得重新讨论）

1. **范围 = B：全 timing-valid anchor frames，67,861 行**
2. **参考线合同 = 沿用观测轨迹 fallback**（OnSite dense 源表真实 map/lane/route/reference-line
   字段实测 0/274,022）
3. **venue = HPC**，分区 `intel,fata`；**`amd` 不得使用**（未做该分区的确定性 canary，
   且 K2 全程也未使用 amd）

---

## 1. 冻结配置（`configs/ipv_sigma01_exact.json`，只读，不得改）

```
solver_mode exact ｜ sigma 0.1 ｜ min_observation 4
reference_clip_margin_m 60.0 ｜ reference_max_points 40 ｜ reference_smooth_points 40
current_ipv_history_window 10 ｜ future_target_history_window 4 ｜ future_target_final_offset 6
```

那张 67,861 行表的溯源记录（`feature_history_window=10`、`target_history_window=4`、
`min_observation=4`、`target_final_offset=6`）与之一致。
旧通道 `ONSITE_CHANNEL_EXACT_HW10`（285 行、`target_history_window=10`）**列入 denylist**。

## 2. 复用什么、绝对不要复用什么

### 2.1 只复用这两段

| 复用对象 | 位置 | 说明 |
|---|---|---|
| 求解 | `src/sociality_estimation/core/ipv_estimation.py` 的 `MotionSequence`(:37) 与 `estimate_ipv_pair`(:181) | 受保护，只读调用，`solver_mode="exact"` |
| 门判据 | `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py` 的 `gate_from_mse()`(:649) | 冻结规格逐字实现；`weights_from_mse` 来自受保护的 `reliability_logdomain.py`(:172) |

### 2.2 **不要**做这几件事

- ❌ **不要用 `scripts/hpc/submit_research_run.sh` 与 `configs/run_specs/research_run_spec_v2.schema.json`。**
  该 schema 是 **RQ014 专用**的（必填/可选字段含 `wod_path_type_mapping_manifest`、
  `ratings_source`、`g2r_*`、`declassification_export_*`、`recovery_contract`），
  与本轮无关。**按 §4 的 K2 先例自建 staged array job。**
- ❌ **不要把 OnSite 行送进 `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py` 的
  `solve_anchor_task()`（:268）。** 它 :271-272 会按 InterHub split 拒绝非 development/guard 行，
  且下游从 InterHub PKL 构造序列——OnSite 没有 PKL。**OnSite 必须有自己的入口。**
- ❌ **不要整体复用 K2 的 `validate_outputs()`**：它硬编码 `4_981_984`、`8_994_736`、
  `14_473_982`（`k2_fullcorpus_materializer.py:1327-1338`），对 OnSite 会产生假 blocker。
  只可复用其中的 row-level invariant。
- ❌ **不要修改 `run_b2_rq015b.py` 的 `ALLOWED_SPLITS` 护栏**（:72、:271）。见 §6.2。
- ❌ **不要使用 `/share/home/u25310231/ZXC/ipv_estimation`**（已退役，仅存兼容链接）。

### 2.3 本轮实质要新写的，只有一件

**OnSite 行 → `MotionSequence` 的入口**（`data` / `target` / `reference` 三件，
`reference` 用观测轨迹 fallback）。求解与门判据两段原样调用。

## 2.4 门判据（冻结规格，一字不改，不得调参，不得做阈值扫描）

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

## 3. 【最重要】测量合同 preflight（轻量，staging 之前完成）

**全部实现为会 FAIL 的脚本断言，输出机器可读审计文件。任一条不过，停下报告。**

上一轮双盲复审的两位复审独立指出：这一层错了，产出的七个 MSE 仍然有限、权重和为 1、
`max_w_log`/`k_eff_log` 落在合法范围、门判据可复算、`ipv_log` 恒等式成立——
**所有既有检查都会通过**，但对应的是错的行、错的车或错的时刻。

**C1 键的一对一**（监督方已实测基础成立，本轮必须继承）：输出的 `product_row_key` 必须与
`.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`
（该表带 `product_row_key` 列）逐行一对一：anchor 表 67,861 行、构造键去重 67,861、
dry-run 67,861 行、**交集 67,861/67,861 = 100.0000%**，无重复无缺失无多余。
键的构造式写进审计文件。

**C2 输出粒度冻结**：一行一 anchor，对应自动驾驶车的当前角色，`输出行数 == 67,861`。
若你判断必须做四角色展开，**停下上报，不得自行决定**——展开会改变分母且与 C1 冲突。

**C3 窗口是行位置语义**，不是绝对 frame_index。`valid_anchor_positions()` 与
`build_anchor_rows()` 都按 `pos` 取窗，见
`reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/build_onsite_m3_anchors_hpc.py:836-847`
与 `:930-997`：

```python
for pos in range(n):
    if pos + TARGET_FINAL_OFFSET >= n: continue
    wx_start = max(0, pos - FEATURE_HISTORY_WINDOW + 1)
    valid = frame_index[wx_start : pos + 1] >= MIN_OBSERVATION
```

求解输入为历史窗口 + 当前行。断言你的窗口切法与上式等价。

**C4 短历史必须显式处理**：`history_row_count` 实测分布（分母 67,861）为
`10→66,289 ｜ 9→257 ｜ 8→258 ｜ 7→261 ｜ 6→264 ｜ 5→265 ｜ 4→267`，
即 **1,572 行历史短于 10**。按冻结 `min_observation = 4` 处理，**不得在输入端剔除**；
若因此求解失败，按工程失败记录。

**C5 主体方向冻结**：显式冻结并断言 ego 是自动驾驶车、counterpart 是对手，
且与 anchor 表的 `ego_key_agent` / `counterpart_key_agent` 一致。

**C6 帧语义事后可审**：输出必须记录 `solve_frame_index`、`anchor_frame_index`、
`target_window_end_frame_index`、`history_window_used`。

**C7 输入列白名单**：只读 motion / key / source / window 字段；**禁读** M3 context 特征、
`target_ipv_future` 等目标值、任何 outcome 或评分字段。白名单写进审计文件并断言实际读取
的列集合是它的子集。

---

## 4. HPC staging 与作业（照 K2 先例）

### 4.1 运行目录

```
/share/home/u25310231/ZXC/sociality_estimation/work_dirs/RQ017/<run_id>/
  repo_stage/   代码
  pydeps/       依赖
  inputs/       上传的 OnSite 输入
  logs/         zxc-rq017-m1_%A_%a.out / .err
  outputs/      分片产物
  process_cache/
```

`<run_id>` 用 `rq017_onsite_materializer_<UTC 时间戳>`。**durable 产物、日志、脚本必须留在
该目录树下**；不得写入其它项目目录。

### 4.2 要上传的输入（约 30 MB，scp 即可）

```
data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/
  onsite_m3_av_anchors_multi_allvalid.parquet          16.5 MB   ← 67,861 行 anchor 表
  onsite_ipv_timeseries_multi_allvalid.parquet         12.6 MB   ← 轨迹时序
```

上传后**逐文件核对 sha256 与本地一致**，写进审计文件。若你判断还需要别的输入，
在报告中列出并说明理由。

### 4.3 sbatch 模板（照 K2 的 `submit_k2_solve_array.sbatch`）

```bash
#SBATCH --job-name=zxc-rq017-m1        # 必须 zxc- 前缀
#SBATCH --partition=intel,fata          # amd 不得使用
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-<N>%<M>
#SBATCH --output=logs/zxc-rq017-m1_%A_%a.out
#SBATCH --error=logs/zxc-rq017-m1_%A_%a.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# matplotlib font-cache 并发锁：必须给每个 task 独立 cache 目录（K2 曾因此整轮作业被取消）
PYTHON="/share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01/bin/python"
```

K2 用的是 `--array=1-460%427` 处理 4,981,984 个求解单元。本轮只有 67,861 个
（**73 倍小**），**分片数自己按每片行数与内存现算，不要照抄 460**，并在报告中给出
分片依据。**并发上限必须按逐节点装箱现算**，禁止用「分区空闲核总数 ÷ 每片 worker 数」。

⚠ 提交前查 `sinfo`：`intel` 与 `fata` 当前空闲节点很少（监督方 2026-08-04 观察到
intel 4 个空闲、183 个 down，fata 0 个空闲）。**排队等待属正常，不得因此改投 amd。**

### 4.4 登录节点纪律

重计算一律 `sbatch`，**不得在登录节点跑求解**。preflight 与校验类轻量脚本可以在登录节点跑。
**若出现任何密码提示，立即停止并报告**，不得输入、存储或打印密码。

---

## 5. 环境同源核验（本轮新增，**这是 venue 改为 HPC 的直接验收**）

必须给出**正面证据**证明本轮与 K2 处于同一软件栈，而不是仅仅"也在 HPC 上跑"：

1. 记录并比对冻结环境路径与其 python 版本、关键依赖版本（numpy / scipy / pyarrow），
   与 K2 运行记录中的环境清单对照。
2. **取 K2 已有的 HPC 锚点基线做逐位复算**：
   `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/g_anchor_hpc_baseline.json`
   与 G 轨 `anchor_mse_hpc.csv`。在本轮环境下重算其中一小批锚点，
   **要求 `mse_per_candidate[7]` 逐位相同、`max_abs_diff = 0.0`**。
   ⚠ 基线必须认准 **HPC 版**（`anchor_mse_hpc.csv`）；
   `.codex-fleet/rq015b-repair/work/anchor_mse.csv` 是 **Mac 版**，
   **上一轮有人比错过，导致误判为数值缺陷**。
3. 若逐位不同，**停下报告，不得继续跑全量**——那说明栈不同源，本轮对照不成立。

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
（OnSite 不在 RQ007 的划分体系里）。**不得删除、注释或放宽。**
OnSite 走 §2.3 的独立入口，InterHub 那条路径一字不动。
**该断言必须是脚本 blocker**，并实测：参与求解的行中来自 InterHub 的为 **0 行**、
携带 RQ007 `held_out` 标记的为 **0 行**。

### 6.3 参考线 fail-closed

去重后观测轨迹点数 **< 2 时不得继续写 OK**。既有实现已有该异常：
`build_onsite_m3_anchors_hpc.py:705-710` 的 `prepared_reference()` 抛
`ValueError("observed reference has fewer than two unique points")`。保留该行为并列为 canary 用例。

### 6.4 那 7 行坐标系异常必须定性，不得静默丢弃

67,861 行中 **7 行（0.0103%）** 的 `relative_distance_anchor` ≈ 570,761.6 米，
由单侧 `relative_dx_anchor` ≈ −570,761.6 导致，`relative_dy_anchor` 正常（约 −8.8 米）。
查清是坐标原点不一致还是缺失值哨兵，单独成节说明。**照常参与求解、必须出现在正式产物中**；
求解失败按工程失败记录。

---

## 7. canary（preflight 与 §5 全绿后才做；canary 全绿后才投全量）

**canary 必须走 sbatch，不得在登录节点跑。** 必须覆盖：

1. **四种状态各至少一行**：`OK` / `NEAR_UNIFORM` / `NO_IPV_EFFECT` / 一种工程失败
2. **至少 2 个并发 array task**（K2 的失败全在并发路径上）
3. **写出后读回校验**，含 null scalar 的落盘/读回模式
4. **§6.4 的 7 行真实坐标异常行**
5. **§6.3 的参考线 fail-closed 路径**（构造去重后点数 < 2 的输入）

自然样本里不出现的状态，**构造合成输入**触发（七候选 MSE 完全相同 → `NO_IPV_EFFECT`；
含 NaN → 工程失败），报告中声明该行是合成的、**未混入正式产物**。

**K2 实际遇到的三次失败必须逐一验证已规避**（作业 `2069424`、`2069818`、`2071368`）：
Matplotlib font-cache 并发锁、PyArrow fixed-size-list 无法写 null array 行、
逐行重算源文件 SHA 过慢。**canary 不全绿不得投全量。**

用 canary 实测本机分片速率再外推全量墙钟，写进报告。

---

## 8. 交付物与自查

产物先落 HPC 运行目录 `outputs/`，取回后落
`data/derived/rq017_onsite_gate/l1_v1/`（parquet，可分片），schema 与 K2 台账
`data/derived/rq015k_logdomain_gate/l1_v1/` 对齐，另加 §C6 的四个帧语义字段。

- 报告：`.codex-fleet/rq017-onsite-materializer/board/reports/RQ017_1_onsite_materializer.md`
- 测量合同审计：`.codex-fleet/rq017-onsite-materializer/work/M1/measurement_contract.json`
- 环境同源核验：`.../work/M1/env_parity.json`
- 机器数字：`.../work/M1/key_numbers.json`｜运行回执：`.../work/M1/run_receipt.json`
  （含作业号、分区、节点、array 形状、逐分片行数与耗时、失败计数、输入与代码哈希）
- 脚本放 `.../work/M1/`，可复跑

**报告要回答**：67,861 行的四类状态分布（各带分子分母）；与 InterHub 全语料对照；
与 RQ016C 支持门（21,936/67,861 = 32.3249%）**交叉后最终可判的行数**。

⚠ **禁止对任何一辆车下「像不像人」的结论。**

### 自查（一轮，但必须有牙齿）

1. **行数守恒**：产物行数 == 67,861；**且** `product_row_key` 唯一、§C1 三方交集成立
2. **状态守恒**：`OK + ABSTAIN + 工程失败 == 总行数`；`NEAR_UNIFORM + NO_IPV_EFFECT == ABSTAIN`
3. **门判据可复算**：从落盘 `mse_0..6` 用冻结规格重推 `status`/`reason_code`，零处不一致
4. **恒等式**：OK 行上 `ipv_log == sum(candidate_ipv_i * w_log_i)`、`k_eff_log == 1/sum(w^2)`；
   **另须单独断言 `K == 7` 与网格 ID == `legacy7_pi_over_8`**（恒等式抓不到整体错用 5 点网格）
5. **工程失败隔离**：工程失败行被记成两个科学 reason 之一的行数为 0
6. **护栏 blocker**（§6.2）与**环境同源**（§5）必须都是脚本 blocker
7. **负对照（强制两条，必须真的 FAIL）**：把 `mse_spread == 0` 换成
   `np.isclose(atol=1e-12)` → 必须 FAIL；把 `theta` 从 0.20 改成 0.22 → 必须 FAIL。
   ⚠ **自然输出不保证含敏感行，必须注入合成 sentinel 行**保证两条都能 FAIL。
   任一条没 FAIL 说明检查是坏的，**先修检查再继续**，并说明怎么修的。
8. **数值健康**：NaN/inf 计数；`max_w_log ∈ [1/7, 1]`；`mse_spread >= 0`；`k_eff_log ∈ [1, 7]`
9. **取回完整性**：远端与本地逐分片行数与 sha256 一致
10. **不覆盖检查**：本地与远端产物目录若已存在内容，停下报告，不得覆盖

---

## 9. 硬边界

```
不改：src/sociality_estimation/core/{agent,ipv_estimation,reliability_logdomain}.py
      pipelines/interhub/process_interhub.py ｜ configs/ipv_sigma01_exact.json
不改 .codex-fleet/rq015b-repair/ 与 rq015k-fullcorpus-gate/ 下任何文件（只读复用）
不改 RQ009 / RQ016 / RQ016C 已落盘 run 目录（只读）
不改 data/derived/ 已有内容；本轮只新建 data/derived/rq017_onsite_gate/
HPC 侧只在 work_dirs/RQ017/<run_id>/ 下写；不得动其它项目目录
不做 git commit / 不碰 git 的任何写操作
禁止 git checkout -- . / restore . / stash / reset --hard / clean -fd
RQ007 held_out 不得被解析；护栏只能收紧不能放松（§6.2）
RQ014 致盲相关的评分字段不得读取；输入列白名单见 §C7
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

本机环境：python3 已有 pyarrow 21.0.0 / pandas 2.3.3。缺依赖直接装上继续。

## 10. 报告结尾必须带状态行

```
state: WAITING_ON_COMMANDER
timestamp_utc: <实取>
```

报告开头先定位，写给完全没跟进过程的读者。需要监督方拍板的事单独成节。
