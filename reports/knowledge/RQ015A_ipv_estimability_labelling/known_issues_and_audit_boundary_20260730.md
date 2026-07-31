# RQ015A 已知问题清单与审计边界声明 — 2026-07-30

随复审包提交｜起草：Claude Code CLI（指挥者）
状态（2026-07-30 定稿时）：`BUILD_WHILE_DENY`，`execution_authorized = false`
**状态更新（2026-07-31，复审通过之后）**：PI 已批准并翻转 `execution_authorized = true`；但**审计仍未运行且当前无法运行**——见新增的 **§7**。§1–§6 为复审时点的原文，未作改动。

本文件的存在本身是一项方法学选择。PI 于 2026-07-30 裁定停止条件为
**零 blocker**（而非零发现），因此剩余的 major / minor 以本清单形式随包提交，
而不是继续无限修下去。理由：审计永远无法证明"已经查完"，
但可以陈述"查了哪些、没查哪些、未查的为何不阻断"。请复审方据此评估，而非据此猜测。

---

## 1. 已知未修问题

### 1.1 同形字符列名可绕过结构列 denylist（major）

**现象（已实测）**：结构列 denylist 已对零宽与格式字符（Unicode `Cf`/`Cc`）做剥离，
再经 NFKC + strip + lower 比对，因此 `ra​ting`、`ipv‌_error`、`sco﻿re`
均被正确拒绝。但**同形字符未被覆盖**：

```
'rаting'   （а = U+0430 西里尔小写 а，冒充拉丁 a）   → 通过
'scоre'    （о = U+043E 西里尔小写 о，冒充拉丁 o）   → 通过
```

**为何不阻断**：

1. 列名来源是 `RQ015A_ledger_schema_v2.json` 的 `row_key_fields` 等字段，
   属受控合同文件，非外部输入。威胁模型不含"攻击者构造列名"。
2. 现实失效路径是：未来某产物**恰好**含一个同形字符拼写的评分列名，
   且被写进 schema。这需要两个独立的偶然同时发生。
3. `--execute` 路径另有 R5 的其它防线（角色列白名单由 schema 的
   `role_source_columns` 显式枚举，而非"凡不在 denylist 者皆可读"）。

**为何不现在修**：覆盖同形字符需要定义 confusable 映射的范围
（Unicode 官方 confusables 表有数千条目，全量引入会带来新的误拒风险，
且需要决定是否引入外部数据文件——这与 run spec 的环境声明相关）。
该范围是独立决策，不宜在收尾轮临时定。

**建议处置**：若 PI 要求覆盖，应作为独立小任务定义映射范围与误拒回归集，
而非在本轮扩张。

### 1.2 D0 的 `NOT_ATTEMPTED` 行保留非空 `q_eff` / `k_eff`（minor）

**现象（已实测）**：OnSite 每 case 5 行、前 4 行落入 local D0 且 `error = 0.5` 时，
32 个 `NOT_ATTEMPTED` 行的 `q_eff` 全部非空。

**为何不阻断**：所有下游消费者均按 `attempt_status` 过滤
（`aggregate_l2` 只对 `ATTEMPTED` 且 `q_eff` 非 None 的行取均值；
`factor_analysis` 同）。独立审计未观察到结论污染。

**为何不现在修**：将"D0 行必须 `q_eff = None`"提升为台账不变量，
会改变 L1 行的语义并可能影响 identity_2/identity_3 的计数口径。
在零 blocker 已达成的收尾阶段，该改动的风险高于收益。

**建议处置**：作为台账 schema v3 的候选条目，与 §2.1 的 recoverability 归一一并考虑。

---

## 2. 已冻结的临时判定（非缺陷，但复审方应知悉）

### 2.1 `rq014_g2r_anchor_scores` 的 recoverability 歧义

schema v2 中该条目缺 `rq007_split_applicable`、`rq007_split_value`、
`expansion_factor`、`collapse_factor`，且 `recoverability` 写作 `L4_UNRECOVERABLE`，
而另两个同类条目写 `ARTIFACT_NOT_PRESENT_LOCALLY`。
identity_3 是 `measurement_rows = Σ_over_recoverability n(recoverability)`，
同一批行有两个竞争值即求和歧义。

**冻结判定**：`ARTIFACT_NOT_PRESENT_LOCALLY` 优先于 `L4_UNRECOVERABLE`，
identity_3 按前者求和；当前实现已强制取前者。
schema v3 的正式修正随 WOD 取回决定一并进行。

### 2.2 `c0_routing_stability` 含 `false` 不使 run receipt FAIL

路由不稳定是**合法的审计结论**，不是审计机制失败。
把它判为 FAIL 会使"发现了不稳定"与"程序坏了"无法区分。
该判定被三轮审计各自追问过一次，三次维持原议。

### 2.3 `Decimal` 不被接受为合法数值

R7 的确定性求和建立在 `float` + `math.fsum` 上；接受 `Decimal`
会开出第二条数值路径，破坏"唯一算法"。故一律拒绝，且在 docstring 中写明系有意为之。

---

## 3. 信任边界声明

```
scripts/rq015a/run_rq015a.py 是唯一受信入口。
直接调用内部函数，或用 object.__new__ / pickle / 元类伪造对象，不在信任模型内。
```

**理由**：Python 无法阻止对象伪造。把它当作威胁面会使每轮审计都产出一条
"还能这样伪造"，且无收敛终点——第 3、6 两轮审计各出现过一次这类报告。

**边界内仍必须成立的两件事，已由审计 7 独立验证**：

- **(a) 公开 CLI 不可绕过**：从 `--execute` / `--validate-only` 出发的任何可达路径
  都不得绕过 permit 校验、allowlist 回源核对、或结构列 denylist。
  审计 7 结论：`--execute` 在 permit 前拒绝且回执记
  `measurement_reader_constructed=false`；合成双键全 true 时仍被 BUILD_WHILE_DENY
  后置硬拒；`--validate-only` 不触达 permit 与 reader。
- **(b) 注入防护**：外部对象不得**替换**掉一个已经过校验的对象。
  由 `open_measurement_reader()` 的 `WeakSet` 身份登记实现
  （非对象上的公开标记属性——那只会变成又一个可伪造的标记）。
  审计 7 结论：duck-typed reader 被拒；同 reader 复用有效；reader 回收后登记不保留。

**一处诚实的限制**（写于 2026-07-30；2026-07-31 已由实测取代，见 §7.4）：
`--execute` 当时是**故意不执行**的（`execution_authorized = false`）。因此"授权翻转后的端到端行为"是从内部路径
**推断**的，不是实测的。审计 7 自己也标注了这一点。
翻转授权后应重跑一次信任边界检验。

---

## 4. 审计覆盖边界

七轮独立健壮性审计，每轮由独立 agent 执行并**对此前所有审计与修复轮保持盲态**
（禁读前序报告、`main_workflow.log`、`git log` / `git diff`）。

累计闭合 **40 条**缺陷，其中 **9 条 blocker**：

| 轮次 | blocker | 其它 | 备注 |
|---|---|---|---|
| 1 | 3 | 6 | |
| 2 | 2 | 5 | 2 条 blocker 由修复轮 1 引入 |
| 3 | 1 | 4 | |
| 4 | 1 | 5 | 首次触及 q 的消费端 |
| 5 | 1 | 3 | 首次交出覆盖面清单 |
| 6 | 1 | 4 | 覆盖面由 12 项"深入"扩至 30 项 |
| 7 | **0** | 6 | 收敛；含信任边界检验 |

### 4.1 公开面覆盖（审计 7 的清单）

**深入**（≥5 类非法/边界输入）26 项，含 `q_eff`、`k_eff_from_error`、
`check_conservation`、`local_positions`、`aggregate_l2`/`l3`、`episode_summaries`、
`band_shares`、`bins_stability`、`c0_route`(+sensitivity)、
`normalize_structural_column_name`、`is_forbidden_human_field`、
`is_measurement_like_field`、`assert_structural_columns_are_safe`、
`CaseAllowlist` 及其两个校验函数、`AllowlistedArtifactScope`、`L1LedgerRow`、
`SortedL1LedgerRows`、`SortedL3Units`、`StructuralColumnSet`、`ExecutePermit`。

**浅**（冒烟或类型检查）15 项。

**未攻击** 1 项：`StructuralReader`（纯 Protocol，无实现体）。

### 4.2 明确未覆盖的方面

- 全量数据扫描（sigma01 2.1 GB / feature matrix 138 个 parquet part）
  —— 受 `BUILD_WHILE_DENY` 约束，一律未做。
- 真实数据的端到端 `LedgerBuildResult` 路径 —— 同上，仅用小型合成输入与
  单个 parquet part 的表头验证。
- 授权翻转后的实际执行行为 —— 见 §3 的诚实限制。
- 任何 rating / preference / human-score 字段 —— 全程零读取（R5）。
- 任何 held_out 行的 measurement 列 —— 全程零解析（R3）。

---

## 5. 方法学 caveat（自我披露）

**审计 1–6 的 prompt 是逐代 `sed` 派生并在尾部追加"本轮补充"章节生成的，
未经重写。** 到第七代时该 prompt 已由 9.6 KB 膨胀至 24.7 KB，
内含 **6 个各自声称"覆盖以上全部、最高优先级"的补充章节**，
而 `OUTPUT CONTRACT`（内容为"最终消息必须严格是此结构、之后不要再写任何东西"）
位于第 152 行、**后面还有 300 行**。该 prompt 使 agent 在
`turn.started` 后 46 分钟零输出、彻底挂死。

发现后已重写为 8.5 KB / 178 行（0 个补充章节，contract 位于末尾），重发即正常完成。
旧稿归档于 `board/prompts/W23-reaudit7-BLOATED-superseded.md` 作为证据。

**对复审方的含义**：审计 1–6 使用的是同一派生链上的较早世代，
虽未膨胀到挂死的临界点，但**其指令一致性可能已被稀释，产出质量可能受影响而未被察觉**。
指挥者无法排除这一影响。审计 7（零 blocker 那一轮）使用的是重写后的干净 prompt。

如复审方认为需要，可要求用干净 prompt 重跑其中任意一轮以校验。

---

## 6. 其它已披露事项（详见计划 v5）

- **可审计范围 3/6**：本地存在 `interhub_sigma01_hw4_timeseries`、
  `rq009_feature_matrix`、`onsite_dense_timeseries`；三个 WOD/RQ014 产物本地缺失。
  若最终未取回，报告须在**标题级**披露 WOD 一支零覆盖，**不得表述为"全语料"**。
- **`wod_rq010b_full479_audited` 不是 L4**：HPC 只读探测确认其 906 行
  （与 schema v2 的 `expected_unverified.rows` 吻合）含 `ego_ipv_error` /
  `ego_ipv_driven_error`；取回并在 HPC 侧投影掉 `rating` 列后可为 `L1_DIRECT`。
  取回需单独授权，PI 已裁定分两步进行。
- **暴露裁定 §8**：`4/7` 与 `0.93` 的 dev+guard 重导条件已由 PI 于 2026-07-27 解除，
  理由是二者已由科学阈值降为报告用 policy bins 且 R6 已在代码层强制 bins 不进入判定。
  §6 原文未改，解除不构成任何执行授权。
- **既有全量测试失败**：约 21 条（RQ014 / launcher / shortcut），
  已用 `19b28024` 干净检出比对证实为既有状态（同批文件 23 failed），与本实现无关。

---

## 7. 本包**不**包含什么（2026-07-31 追加，复审之后）

> 本节在三路最终复审通过之后追加，记录授权翻转当天查明的一项范围事实。
> 它不改变任何已复审的结论，但**对接手者是最重要的一句**。

### 7.1 审计的执行体不存在；审计从未运行

`scripts/rq015a/run_rq015a.py` 的 `--execute` 路径在授权许可签发**成功之后**，
仍无条件抛出：

```
execution permit unexpectedly succeeded in BUILD_WHILE_DENY;
refusing to run audit without PI-reviewed post-authorization handoff
```

函数到此结束。**没有任何代码把 `build_ledger` / `factor_analysis` / `receipt`
串联成一次真实的审计运行。**

**这是原设计，不是缺陷。** 交接手册的范围 T1–T11 依次是：台账构建器、执行前校验、
回执写入器、因素分析、唯一入口、授权对象、运行合同、计划正文、暴露裁定、
校验清单、HPC 只读探测——**没有一项是"运行审计"**。
本包的交付物自始至终是**一份可被复审的实现**，而不是审计结果。

### 7.2 因此，"已授权"不等于"可运行"

截至 2026-07-31：

- `execution_authorized` 已由 PI 批准并翻为 `true`（六道检查全部 PASS，
  `load_execute_permit()` 直接调用可签发许可）；
- **但 `--execute` 仍会拒绝**，因为上述硬阻断；
- **审计从未运行过**：held_out 的 measurement 列零解析、评分字段零读取、
  未产出任何台账、画像或结论。

接手者若期望"翻转即可跑"，会在这里卡住——**那不是环境问题，是范围问题**。

### 7.3 运行审计属于新范围，且需要独立复审

要真正运行，至少需要：

1. 把已交付的各组件接成一条执行路径（读取 → 逐产物建 L1 → 三恒等式 →
   L1→L2→L3 → 因素分析 → C0 路由 → 写 run receipt 与输出）；
2. 对**该执行路径本身**做独立审计与复审——它是全新代码，
   不因组件已通过复审而自动可信；
3. 移除 `run_rq015a.py` 的硬阻断，并明确记录移除的授权依据。

**强烈建议先在最小产物上打通再扩到全量**：`onsite_dense_timeseries`
物理行 70,317，而 `interhub_sigma01_hw4_timeseries` 为 2.1 GB、
`rq009_feature_matrix` 为 138 个 parquet 分片。

理由是本包已有的经验：**从未被执行过的路径会藏东西**。
仅 2026-07-31 一天，授权链上就暴露了两个存在已久但不可能被测到的缺陷
（`authorized_package_commit` 的不可满足语义、`execution_authorized` 实为六道而非双键），
二者成因相同——开关一直关着，那条链从未走到全绿。执行路径尚未走过任何一次。

### 7.4 信任边界已由实测确认（不再是推断）

复审方曾记录"授权翻转后的端到端行为是**推断的、不是实测的**"。
2026-07-31 翻转后已做专项复检，结论：**边界成立**。
六道检查逐道构造反例均独立拒绝；伪造 `CaseAllowlist` 映射在取得 reader **之前**被回源核对拒绝；
split 源的 TOCTOU 替换在首行 yield 前被 `source_sha256` 不符拦下；
duck-typed reader 注入被内部类型要求拒绝；结构列 denylist 判定入口未因授权开启而改变。

**仍未覆盖**：真实大文件的完整读取路径、真实 validate receipt 的 input-root 全量摘要、
以及 permit 成功后那条硬阻断分支的实跑。这三项要等执行路径存在之后才能覆盖。
