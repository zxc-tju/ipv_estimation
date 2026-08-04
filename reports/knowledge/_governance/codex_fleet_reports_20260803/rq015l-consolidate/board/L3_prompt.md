# L3 — RQ015 收官成文

你是 RQ015 收官轮（track L）的第三个也是最后一个执行 agent。仓库根就是你的 `--cd`。
L1 与 L2 已结项，leader 已做完自查并修正了 L1 的一处口径。**你的任务是成文，不是重算。**

**唯一交付物：** `.codex-fleet/rq015l-consolidate/board/reports/RQ015_consolidated_report.md`

---

## 零、先读这四份（按顺序，不读完不要动笔）

1. `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L1_report_section.md`
2. **`.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L1b_leader_selfcheck.md`**
   ← **这份是 leader 对 L1 的修正与补充，与 L1 冲突处以这份为准**
3. `.codex-fleet/rq015l-consolidate/work/L2_onsite_unknown/L2_report_section.md`
4. `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md`

需要引用别的轨道时，可读（都是已结项的 markdown 报告）：
`.codex-fleet/rq015b-repair/board/reports/B2_report.md`、
`.codex-fleet/rq015c-drift-forensics/board/reports/C1_drift_report.md`、
`.codex-fleet/rq015d-sigma-rederive/board/reports/D1_sigma_report.md`、
`.codex-fleet/rq015f-estimability-contract/board/reports/F1_contract_recheck.md`、
`.codex-fleet/rq015g-hpc-resolve/board/reports/G1_hpc_resolve_report.md` 与 `G2_crossnode_gate.md`、
`.codex-fleet/rq015h-abstain-gate/board/reports/H_FINAL_leader_synthesis.md`、
`.codex-fleet/rq015i-underflow-regimes/board/reports/I1_underflow_regimes.md`、
`.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md`、
`.codex-fleet/rq015e-rq009-dependency/board/reports/E1_dependency_report.md`。

**读报告即可，不要重跑任何分析、不要重算任何数字。**

---

## 一、读者是谁（决定全文写法）

读者是**一个没有跟进过 RQ015 任何过程的人**，隔了很久才看这一份，
并且会跳过所有中间汇报直接看这一份。**上下文重建的成本由你承担，不由读者承担。**

四条硬性要求，会被逐条核：

1. **先定位，再讲进度。** 开头必须交代：这项工作要解决什么问题、整体走到哪一步、
   本轮是哪一环。不得直接从增量讲起。
2. **不用黑话，不用比喻。** 必须使用项目专有名词时**当场用一句话说明它是什么**
   （IPV、canonical 求解单元、台账行、envelope、HT 权重、anchor……）。
   自造的形象化说法一律换成直白描述。
3. **结论与待决事项分开成节。** 需要上级拍板的事**必须单独成节**，
   写清选项、判断依据、以及不做的后果，**不得藏在叙述中间当作陈述句带过**。
4. **数字自带口径。** 每个百分数必须同时给分子、分母、筛选条件、来源文件与列名。
   **一个读者无法自行复算的数字等于没给。**

---

## 二、必须覆盖的十节（顺序可微调，内容不可缺）

### 1. 位置与问题
最终用途是 online verification：判断一辆自动驾驶车的 IPV
（Interaction Preference Value，刻画社会交互倾向的标量参数）是否落在人类分布内。
判据由**两个弃权机制串联**：机制一判「这一帧的 IPV 数值到底携不携带信息」，
弃权则直接结束、不进机制二；机制二是 RQ009 已 accepted 的 envelope（人类分布覆盖区间）支持度判据。
**RQ015 整条线做的是机制一。**

### 2. 缺陷的性质
原实现在数值下溢时退回「七个候选等权」的兜底；候选网格对称，
于是必然算出 `ipv` **恰为 0**、`ipv_error` **恰为 `1 − 1/√7 = 0.6220355269907728`**。
后果：**「算失败了」与「该个体完全自利」在数据里不可区分。**

必须解释**为什么七个候选会给出逐位相同的轨迹**：
目标函数在无交互时退化为 `cos(ipv) · 内项 + 常数`，
而候选网格 ⊂ (−π/2, π/2) 使 `cos(ipv) > 0`，**正标量不移动 argmin**。

### 3. 修复
改到 log 域，`w = softmax(−MSE/(2σ²))`。
**必须写明这是精确恒等，不是近似**——它与线性域公式在数学上等价，只是避免了下溢。

### 4. 门的规格（已冻结）
- `mse_spread == 0` → `NO_IPV_EFFECT`
- `max(w_log) < 0.20` → `NEAR_UNIFORM`
- 否则 → `OK`

**`θ = 0.20` 是政策阈值，不是数据断点。必须这样写**，不得暗示它由数据中的某个拐点决定。

### 5. 确定性证据
- 同一软件栈下 **AMD EPYC 与 Intel 逐位相同**：348/348，Slurm job `2024766`
- **Mac 与 HPC 不同**：2,300 个锚点中 1,867 个不同，最大差 70.4；
  **差异来自软件栈而非 CPU**
- 方法学结论：**曲面越平，argmin 越不可复现，但由曲面形状定义的量反而更可复现**

### 6. 全语料普查（**照抄，不要重算**）
InterHub 全量 **4,981,984** 个 canonical 求解单元
（canonical 求解单元 = 去重后的一次 IPV 求解，一个求解单元可支撑多条台账行）：

| 结果 | 计数 | 占比（分母 4,981,984） |
|---|---:|---:|
| `OK` | 3,502,340 | 70.3001% |
| `NEAR_UNIFORM` | 1,457,746 | 29.2604% |
| `NO_IPV_EFFECT` | 19,964 | 0.4007% |
| `SOLVER_FAILURE`（工程失败） | 1,934 | 0.0388% |

RQ009 台账行域：`OK` **6,405,292 / 8,994,736 = 71.2116%**。
G 锚点（正确 HPC 基线）：`compared_rows = 2300`、`max_abs_diff = 0.0`。
RQ009 回填：`rows = unique_keys = 8,994,736`、`duplicates = 0`（实测，非硬编码）。
总台账 **14,473,982 = InterHub 5,197,072 + RQ009 8,994,736 + OnSite 281,268 + WOD 906**。

### 7a. L1：RQ009 精确零点原子的拆分
**这是本轮最主要的科学产出，要给足篇幅。**

先讲清 RQ009 自己的那条警告：RQ009 报告 Limitations 一节记有，
其打分目标存在 **~21.5% 的精确零点原子（273,819 / 1,270,566）**，
并明言这造成 80% boundary-tie / 1e-10 endpoint-nudge 的覆盖脆弱、削弱相关性，
限定了 interval-tie 行为与 practical null 的解释。RQ009 当时无法判断这些 0 是什么。

然后给 join 可行性（**这一段不能省**）：
- 键：`case_key + anchor_frame_index + perspective + source_dataset` + `|role=target_future`
- `1,270,566` = RQ009 M3 `fold=test` 预测文件在**单个 alpha 层**上的目标行；
  `3,811,698 = 1,270,566 × 3`（三个 alpha 层）
- `8,994,736` = `4,497,368` 个 product 行 × 2 个 measurement role
  （`target_future` 与 `counterpart_current`）
- 连接是**精确、无重复的左连接**，但**不是全覆盖**

**未命中 381,674 行（30.0399%，381,674/1,270,566）的定性以 `L1b_leader_selfcheck.md` 为准**：
整案级排除，涉及 2,270 个 case，其中出现在台账 `case_id` 里的为 **0**，
被部分覆盖的 case 为 **0**；命中的 5,306 个 case 与 RQ015E 独立记录的 dev+guard case 集吻合。
**因此推断这些 case 属 RQ007 held_out。必须写明这是推断，不是直读标签**
（未打开受保护的 confirmation 划分文件）。

**主口径分母是 192,271**（零点原子中落在台账覆盖域内的行），不是 273,819：

| 类别 | 行数 | 占台账覆盖域（分母 192,271） | 占全零点原子（分母 273,819） |
|---|---:|---:|---:|
| 过门的真中性零（`status=OK`） | 99,938 | 51.9777% | 36.4978% |
| 弃权而被记成 0（`status≠OK`） | 92,333 | 48.0223% | 33.7205% |
| ├ `NEAR_UNIFORM` | 90,490 | 47.0638% | 33.0474% |
| ├ `NO_IPV_EFFECT` | 1,796 | 0.9341% | 0.6559% |
| └ `SOLVER_FAILURE` | 47 | 0.0244% | 0.0172% |
| 台账未覆盖（整案级排除） | 81,548 | 不适用 | 29.7817% |

**结论（描述性，不得写成因果）：** 在台账覆盖域上，RQ009 那个精确零点原子里
**约一半（48.0223%，92,333/192,271）不是中性的 IPV 估计值**，而是弃权情形下被写成 0 的数值；
主体是权重近均匀（90,490/192,271 = 47.0638%）。**这直接回应了 RQ009 自己那条警告。**

**旧指纹不是好判据**（支撑第 8 节）：在台账覆盖域的零点行里，
指纹（`ipv_error` 命中 `0.6220355269907728`，容差 1e-12）命中行中
**74.8911%（22,358/29,854）状态是 `OK`**；非 OK 行中只有 **8.1184%（7,496/92,333）** 命中指纹。
机制解读（描述性）：那 22,358 行（11.6284%，22,358/192,271）是旧实现线性域下溢退回等权、
而 log 域恢复出非均匀权重（`max(w_log) ≥ 0.20`）的行，即 log 域改写实际救回的那批。
**结论：旧的 `ipv_error` 数值既不充分也不必要。**

### 7b. L2：OnSite 274,022 行 `UNKNOWN` 的定性
以 `L2_report_section.md` 为准，要点：
- `UNKNOWN` 是 `scripts/rq015a/build_ledger.py:1219-1233` 的**显式分支，不是隐式 else 兜底**
- OnSite 合计 281,268 行：`UNKNOWN` 97.4238%（274,022/281,268）、
  `ATTEMPTED` 1.0574%（2,974/281,268）、`NOT_ATTEMPTED` 1.5188%（4,272/281,268）
- 全部 `UNKNOWN` 行 100.0000%（274,022/274,022）为 `source_reason_code=EMPTY_CELL_UNEXPLAINED`
- **判断：属「流水线没走到」，不属「数据确实不支持」。** 依据是这些行的
  `case_key/frame_index/timestamp_ms`、ego/counterpart 位置速度 heading、配对 ID、
  距离与相对速度字段均 100.0000%（274,022/274,022）非空；
  而 OnSite stage3plus 生成脚本默认 `--max-anchors-per-unit 1`，
  只对选中 anchor 的支撑帧/target 帧填 `ipv_*`（`build_onsite_m3_anchors.py:776-831`）
- **保留的输入边界**：dense 源表没有真实地图/车道/route/reference-line 字段
  （0.0000%，0/274,022），当时用 observed trajectory fallback 作参考线
- 与 RQ015A 口径一致：同一组来源状态计数，仅列名由 `attempt_status` 改为 `source_attempt_status`
- **本轮不补齐。** WOD 906 行与 OnSite 2,974 行继续保持不适用状态

### 8. 交付给下游的接口约束
`ipv_log = 0` 是**合法且高频**的通过门估计值（门后 23.40% 的通过行取该值）。
**判别只能用 `status` 与 `reason_code`，不能用数值。**
下游代码不得把数值 0 当作弃权。台账覆盖范围之外的行 `gate_applicable=false`，
不得被计入 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。
第 7a 节的指纹结果是这条约束的正面证据，请交叉引用。

### 9. 方法学教训（K2 换来的，必须写）
- (a) **1 行 canary 测不到两条真实路径**：多 worker 并发、工程失败行写盘。
  K2 实际发生的失败恰好都在这两条路径上。
- (b) **每条验收判据都要有一次「故意让它失败」的验证。** 本轮出现两例
  「看起来在检查、实际没检查该检查的东西」：RQ009 `duplicates` 硬编码为 0、
  G 锚点比错基线（拿 K2-HPC 去比 RQ015B 的 Mac 基线）。
- (c)（本轮新增，必须写）**L1 报了 29.78% 的 join miss 却没查明它是什么**，
  由 leader 补查才定性为整案级排除。**「未命中」不是一个可以留白的类别**，
  留白会让整张拆分表无法解读。

### 10. 遗留项
- **J 轨 HT 权重分母 2,646,058 与 RQ009 台账行 8,994,736 的关系尚未确立**
  （K2 报告 §4.1 明写 `not yet established`）
- OnSite/WOD materializer 未做（PI 已裁定本轮不做）
- 第 7a 节 held_out 归属为推断，未直读受保护划分文件

---

## 三、**分母纪律（监督方硬约束，违反即退回）**

至少五个分母在流通，**每一个比率后面必须紧跟它的分母，不许出现无分母的百分数，
不许在不同分母之间搬运比率**：

| 分母 | 含义 |
|---:|---|
| 2,646,058 | J 轨 HT（Horvitz-Thompson，一种按抽样概率倒数加权的估计方法）权重的全域分母 |
| 4,981,984 | InterHub canonical 求解单元 |
| 8,994,736 | RQ009 台账行 |
| 1,270,566 | RQ009 打分目标行（单 alpha 层） |
| 192,271 | 零点原子中落在台账覆盖域内的行 |

- 求解单元与台账行的压缩比 **2.804×** 已知；
  **2,646,058 与 8,994,736 的关系仍未确立**，照 `not yet established` 表述，
  **不得称「域一致」**。
- 与 J 的抽样估计对照**只用台账行域**（差 **0.0579** 个百分点）并说明理由
  （J 是按行加权的 HT 估计，其估计目标是行加权通过率，不是去重求解单元通过率）；
  求解单元域（差 **0.9694** 个百分点）**单独列**。
- **仍不得写成「验证通过」。** 可辩护的表述上限是：
  「在台账行域上，设计基估计与普查相差 0.06 个百分点」。J 是抽样估计，K2 是普查。

---

## 四、硬边界（违反即退回）

- **不改** `src/sociality_estimation/core/agent.py`、`ipv_estimation.py`、`reliability_logdomain.py`、
  `pipelines/interhub/process_interhub.py`、`configs/ipv_sigma01_exact.json`
- **不改 RQ009 目录下任何文件**，不重算它的 envelope、不重算它的 4.78% 弃权率、
  不改它已 accepted 的任何结论
- **不改 `scripts/rq015a/` 下任何文件**
- **不修改 L1/L2 已落盘的证据文件**（包括那个 81,548 的计数，原样保留交监督方裁定）
- 不投 Slurm，不重解任何锚点，不重跑 K2 的 join，**不做任何重算**
- 不 `git commit`；**禁止** `git checkout -- .` / `git restore` / `git stash` /
  `git reset --hard` / `git clean -fd`
- **不解析 RQ007 held_out**，**不打开** `data/derived/interhub/RQ008_*/01_split/confirmation_PROTECTED/`；
  不读 RQ014 致盲相关评分字段
- **不要对 `reports/` 做全仓库 `rg`/`grep -r`**（会把 RQ003 controlled-access 内容拉进上下文）
- **全文禁用 `estimability` 一词，禁用「测出 / 未测出 IPV」的说法。**
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
- **描述性结果不得写成因果主张**
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ` 实取，**不要前瞻估计、不要编**

---

## 五、格式与过程纪律

- 输出 **Markdown**（不是 HTML），写到
  `.codex-fleet/rq015l-consolidate/board/reports/RQ015_consolidated_report.md`
- 篇幅以讲清楚为准，不要为凑长度注水，也不要为省事压缩掉分母和口径
- **一轮写完 + 一轮自查，就结束。** 不做盲审、不出第二版、不写治理文书。
  本项目此前一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，科学结论产出为零，
  那是反面案例不是标准。
- 自查只查四件事：
  (1) 是否存在无分母的百分数；
  (2) 是否在不同分母之间搬运了比率；
  (3) 第 7a 节主口径分母是否为 192,271，held_out 归属是否写明为「推断」；
  (4) 是否出现 `estimability` 或「测出/未测出 IPV」字样。
- 文末写一节 `## 状态`，内含 `state: WAITING_ON_COMMANDER` 与实取 UTC 时间戳。
  **不要自称 DONE。**
