# RQ015A v2 独立技术复审——Reviewer 1

## Review setup

- **Input scope:** `reports/plans/RQ015A_plan_v2_concentration_audit_20260726.md`。
- **Frozen SHA-256:** `9186c95eb6d84ee56626f6e96cb75d2f0422297446824ea20848e34258ab9a67`，现场复算一致。
- **Baseline manifest:** `reports/plans/RQ015A_plan_v2_checksums_20260726.sha256`，现场执行 `shasum -a 256 -c` 得到 **6/6 OK**。
- **Assessment boundary:** 本路只复审技术正确性、唯一可执行性和主张边界。可读证据限定为 v2 冻结计划、v1 综合、PI exposure disclosure、RQ007 binding/split 与相关一手 schema/code。未读取 Reviewer 2 / Reviewer 3 的任何 v2 输出，未读取任何 v2 synthesis，也未与其他复审路线交换判断。
- **Execution boundary:** 未运行 RQ015A 计算，未解析新的 held-out 测量值，未改计划/代码/数据/决策/状态，未连接 HPC。
- **Review posture:** 本报告按 Nature-style referee 证据标准评估“作者的案例是否已被技术建立”，不代替编辑作期刊录用判断。

## Overall assessment

**Verdict: `BLOCKED`**

**Finding counts: 5 blocker / 3 major / 2 minor.**

v2 的方向是正确的，而且比 v1 更接近一个可审计的测量产品：它把 continuous `q_eff` 设为主产物，把三档降级为 policy summary（计划 `:11-48`）；修正了 RQ007 split 标签与 case-first allowlist（`:50-64`）；补上 M3 `source_dataset` join key、OnSite retained-sequence 时钟、WOD replay-out-of-scope 和 RQ014 attempt-unknown 事实（`:65-75`）；禁止跨产物 pooling（`:77-84`）；还给出三层单位、minimum/zero support、case bootstrap、C0 有序路由、精确 exposure 措辞和 run-contract 所需字段。

然而，v2 仍未关闭 Formal-G1 的决定性自由度。五个问题会直接改变行数、分母、episode 摘要或 C0 终态：

1. 仍在使用被 PI 明令“重导出前不得用于结论画像”的 `4/7` 与 `0.93`，但没有新 PI amendment 撤销/替代该条件；
2. `input_rows = sum terminal_rows` 与本计划自己的 1→2、1→4 宽表展开及 3→1 M3 去重数学上不可同时成立；
3. L2/L3 键丢掉 `artifact_id` 和 measurement role，会重新混合计划明令禁止 pooling 的产物/current/target 测量；
4. 计划说 policy bins 不得用于任何下游判定，但 C0 恰好以 `q_eff >= q_hi` 作为 owner-reanalysis 触发器；
5. inventory/schema/crosswalk/SAP/run spec/validator 仍只是未来交付物，不在本次 6 项 checksum manifest 内，执行者仍可在本轮复审后自行冻结关键规则。

### G2–G7 closure audit

| Gate | v2 status | Technical judgment |
|---|---|---|
| **G2 policy bins / threshold contract** | **PARTIAL / BLOCKING** | continuous `q_eff` 作 primary 的中心修正成立；但旧 PI rederivation condition 未被替代，v2 又用同两个值生成 policy summary、episode filter 和 C0 trigger，且三档不等式/边界与 instability 函数未写死。 |
| **G3 input / split / D0 / ledger** | **OPEN / BLOCKING** | split 白名单、M3 key 和 OnSite 时钟已显著改进；但无 checksum-frozen inventory/schema/crosswalk/fixtures，守恒式错误，主键不足，WOD D0 仍写“逐产物确认”。 |
| **G4 units / SAP / episode summary** | **PARTIAL / BLOCKING** | 三层概念、support 与 bootstrap 已有；但 L2/L3 会跨产物/角色混合，层间聚合函数、因素分析 SAP、episode 零分母/unknown/时间权重未冻结。 |
| **G5 deterministic C0 routing** | **OPEN / BLOCKING** | 路由名称和优先级有进步；但与 `:48` 直接矛盾，暴露度“合计”可重复计数，条件非本质互斥，通用 1:1 mapping 不适用于聚合型 owning-RQ 分析。 |
| **G6 held-out governance** | **PARTIAL / major residual** | disclosure 已精确订正，README pointer 已创建且纳入 manifest；但 `decision.md` 仍字面声称 untouched，而 disclosure 要求 decision + README 均加 pointer，v2 边界又禁止修改任何 decision。 |
| **G7 run / authorization / validator** | **OPEN / BLOCKING** | operation ID、validate-only/no-overwrite/single-use 意图已有；但没有实际 immutable run spec、exact command/env/output root、validator schema 或绑定 SHA，当前 manifest 无法审查这些对象。 |

## Who would be interested in the results, and why

- **自动驾驶与人–机交互研究者：** 它可把“估计器未运行”、“已运行但候选权重近均匀”、“产物丢失了集中度字段”和“下游只剩聚合值”分开。
- **参数识别、逆向规划和不确定性研究者：** `q_eff` 是一个清楚的 effective-support 量，适合说明“候选权重集中”不等于“参数正确”。
- **RQ003/RQ009/RQ010B/RQ011B/RQ012B/RQ014 的 owning analysts：** 一个真正冻结的 ledger 可用来判断哪些原分析需要 owner review，而不是由 RQ015A 代替 owning RQ 宣布结论有效或失效。
- **更广的测量学与数据治理读者：** 宽表展开、测量角色、历史产物可恢复性与 held-out exposure 的组合，具有超出单一 IPV 任务的共性。

## Major strengths

1. **构念边界继续保持正确。** v2 不回退到 `K_eff-only = estimability`，并再次明确禁止“是否测出 IPV”语义（`:11-15`），符合 RQ007 对五个 first-class concepts 和 concentration-only 较弱诊断的绑定分离（`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/binding_execution_contract.md:53-81`）。
2. **continuous-primary 是对 v1 G2 最好的科学回应。** 数据分布本身不能自动生成两个有语义的真实分界；把分位数、直方图和三层分布作为主产物（`:16-39`），比将操作性分档伪装成科学边界更稳健。
3. **split 泄漏路径讲清了。** 真实标签、白名单、RQ009 另一套 fold 不得与 RQ007 混用，以及“先按 case ID 过滤，后读 measurement fields”都已准确写明（`:50-64`），与 RQ007 `split_freeze.json:24-45,76-114` 一致。
4. **跨产物事实不再被压成一个错误通用规则。** M3 的 `source_dataset` 键、OnSite filtering-after-sort 局部序号、WOD replay 可恢复但超范围、RQ014 21,576 行 attempt unknown，都是会真实改变分母的修正（`:65-75`）。
5. **禁止 pooling 正确保护了 observation weight。** 计划识别 RQ009 current/target/M3 是 sigma01/hw4 的复制或 join，要求逐产物报告，语料级数字仅来自上游原表（`:77-84`）。
6. **held-out 暴露事实已经被精确化。** 更新后 disclosure 明确区分“程序解析/聚合”与“未展示/导出/人工检视”（`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md:89-108`），RQ007 README 也已加入不得再称 pristine untouched 的指针（`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/README.md:3-12,40-50`）。

## Major concerns

### B1 — G2：“降级为 policy”没有撤销 PI 的强制重导出条件，且 policy bins 本身未完整定义（blocker）

v2 把 `q_lo=4/7` 和 `q_hi=0.93` 改称 policy choice，并写“不再需要科学导出”（`:16-25,41-48`）。这一科学定位本身可以成立，但它没有自动改写被本轮 manifest 绑定的 PI disclosure。该 disclosure 仍明确规定：

- 两个集中度阈值“**必须**从 dev+guard 分布重新导出”；
- `4/7` 与 `0.93` 为 `PROVISIONAL_PENDING_DEVGUARD_REDERIVATION`；
- 重导出前“**不得**用当前占位值产出任何结论画像”。

证据：`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md:71-84`。新增的措辞精确化章节仅更正“是否逐行解析”的事实（`:89-108`），没有撤销上述附加条件。因而 v2 仍用同两个值生成九组分档表、用 `q_lo` 过滤 episode（计划 `:43-47,111-114`），并以 `q_hi` 触发 C0 owner action（`:118-130`），与现存 PI 条件不能同时成立。

即使不考虑治理冲突，v2 也没有在取代 v1 后自包含地重写三档 label、不等式与等号归属。“任一产物的三档占比极差 >10pp”也有多个可实现解释：可以是每一档在 9 组 grid 上的 max-min 后取最大，也可以是单组三档内 max-min，两者会产生不同 withheld 结果。

**Closure required:** 由 PI 作出 checksum-bound append-only amendment，明确以下二选一：

1. 撤销/替代旧 rederivation condition，授权同两个值只作 policy sensitivity，且不得驱动 C0/owner action；或
2. 完成被明令的 dev+guard 重导出，再按新的 provenance 使用。

同时必须写死三档不等式/等号、精确 instability 函数（例如 `max_class(max_grid p - min_grid p) > 0.10`），并规定 `BINS_WITHHELD_UNSTABLE` 时 episode/C0 是否同步 fail closed。

### B2 — G3：ledger 守恒式数学上错误，主键不足，冻结 inventory/schema 仍未交付（blocker）

v2 把 ledger 定为 long format，主键为 `(artifact_id, product_row_key, measurement_role)`，并要求 `input_rows = sum terminal_rows`（`:86-93`）。该守恒式与计划自己的产物事实直接冲突：

- sigma01 hw4 是 `2,490,992` 宽表 physical frame rows 展开成 `4,981,984` agent-values（`:67-70`）；
- OnSite 一个 physical row 含 ego/counterpart x hw4/hw10 四通道（`:71-72`）；
- M3 三个 alpha physical rows 要去重为一个 anchor measurement（`:70`）。

因此 physical `input_rows`、expanded measurement cells、intentional duplicate rows 和 terminal ledger rows 不可用一个 1:1 等式混在一起。验收 `:175-178` 要求 OnSite D0=`1,068`，这是 physical-row 计数；在四通道 L1 ledger 中，对应的 D0 measurement count 应明确是 `4,272`（若四通道全在 scope）。两个分母必须并列，不能用同一个“D0 行数”名称。

主键也未足以保证唯一：当同一 OnSite row 同时含 hw4/hw10，或同一 RQ009 anchor/perspective 同时含 current/target role 时，键必须显式带 `configuration_id` 与 `measurement_time_role`；不能依赖未定义的 `measurement_role` 是否暗中把两者编进字符串。`product_row_key` 也未逐产物写出确切列。WOD RQ010B full479 的 D0 判据在所谓“已冻结清单”中仍是“逐产物确认”（`:73`）。

最后，v1 综合明确要求下一版把 inventory、schema、crosswalk、split SHA 和 fixtures 直接纳入 checksum manifest（`reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_three_reviewer_synthesis_v1_20260726.md:159-178`）。v2 manifest 实际只绑定 plan、RQ015B plan、disclosure、RQ007 README、RQ015B prototype 代码与测试（`reports/plans/RQ015A_plan_v2_checksums_20260726.sha256:1-6`），未绑定任何 RQ015A inventory/schema/crosswalk/fixture，甚至未绑定已存在的 RQ007 split assignment。

**Closure required:** 将守恒链重写为可复核的多阶段等式，例如：

```text
expanded_measurement_cells
= terminal_ledger_rows + explicitly_rejected_cells

source_physical_rows * frozen_crosswalk_expansion_factor
= expanded_measurement_cells + structurally_absent_cells

M3_alpha_rows
= unique_anchor_measurements * 3 + malformed_or_incomplete_alpha_rows
```

实际公式必须逐产物冻结而非照抄上述示意。L1 主键至少加入 `configuration_id` 与 `measurement_time_role`，每个 `product_row_key` 给出具体列。把 authoritative input path/hash/producer/version/K/grid/D0 rule/expected physical rows/expected measurement cells、schema、crosswalk、fixtures 与 validator 实际创建并纳入新 manifest 后，再重启复审。

### B3 — G4：L2/L3 键重新引入跨产物/角色 pooling，且层间 SAP 与 episode 边界仍未冻结（blocker）

v2 定义：

```text
L1 = (artifact_id, product_row_key, measurement_role)
L2 = (case_id, perspective, configuration)
L3 = case_id
```

证据：计划 `:95-101`。但同一计划 `:77-84` 明令每个产物分布分开报告、禁止跨产物 pooling。L2 去掉 `artifact_id`，L3 连 perspective/configuration/measurement role 也全部去掉；如果按字面 group-by，它们必然把 sigma01 原表、RQ009 复制/join、current/target 和不同 window 重新合并。即使执行者准备“先分 artifact 再 group-by”，这个关键前置操作也没有写进键或公式，所以仍有多种合法实现。

层间统计也未定义。计划没有说 L2 代表 mean/median/quantile/empirical distribution 中的哪一个，也没有说 L3 是对 L2 等权、对 measurement 等权，还是先对 perspective 再对 configuration 聚合。`>=5` minimum support 和 `ZERO_SUPPORT` 分支是好的元规则（`:103-110`），但它们不能代替 estimand 本身。“集中度相关因素探索性分析”仍是交付物（`:158-168`），而所谓 §3 SAP 完全没有冻结该分析的 endpoint、预测子、当前/未来时间支撑、source/configuration 混杂、missingness、多重性或效应量。

episode summary 虽然终于写出“`q_eff<=q_lo` 硬筛”和“`1-q_eff` 加权”（`:111-114`），但仍没有定义：

- D0、attempt unknown、q unknown 是否先排除；
- 零个 `q_eff<=q_lo` 帧时的 terminal status；
- 所有有效帧 `q_eff=1` 使 `sum(1-q_eff)=0` 时的处置；
- 不规则采样/多采样率是帧等权还是时间积分；
- perspective/configuration 是否先分开；
- `BINS_WITHHELD_UNSTABLE` 时依赖 `q_lo` 的 episode 定义是否也 withheld。

**Closure required:** 修正 L2/L3 键，至少在所有层保留 `artifact_id` 和 `measurement_time_role`，并把 configuration/perspective 聚合顺序写成精确函数。如不愿冻结因素分析 SAP，就将该交付物从 scope 删除。episode 两套摘要须给出有效行集、时间权重、零支持/零分母/unknown 终态与唯一 fixture 输出。

### B4 — G5：C0 路由与“bins 不得下游使用”直接矛盾，其暴露分子也不是互斥集合（blocker）

计划 `:48` 写：分档“**不得用于任何下游判定**；C0 只用连续量与暴露度”。但 C0 的两个主终态恰好由 `q_eff>=q_hi` 这一 policy-bin 边界与 5% 触发线决定（`:118-130`）。这不是边界案例，而是文本内部直接矛盾。

“`q_eff>=q_hi` + `NOT_ATTEMPTED` + `unknown` 的合计占比”也未定义成互斥集合。对 D0 行，`q_eff` 必然不可计算；如果 `unknown` 表示所有 q-unknown，则同一行会同时进入 `NOT_ATTEMPTED` 与 `unknown`。对 RQ014，21,576 行 attempt unknown 且全产物无 error，attempt-unknown、q-unknown 与 provenance-unknown 是否为同一“unknown”完全未说明。这会直接改变 5%/20% 触发结果。

四个终态虽然有优先级，但表中自称“必要充分条件”的条件本身重叠：`unknown>=20%` 必然也使“合计暴露”至少 20%，同时满足 `OWNER_REANALYSIS_REQUIRED`与 `INDETERMINATE`。优先级可以把输出变成单值，但应写成 ordered decision function，不应声称原始条件互斥。`NOT_APPLICABLE` 必须先于 mapping/unknown 检查，否则一个根本不使用 IPV 的 RQ 会因无法建立 ledger mapping 而误进 `INDETERMINATE`。

此外，“分析行与 ledger 1:1”不是通用正确条件：case/episode 聚合型 owning RQ 可能合法地以一个 analysis row 对应多个 L1 measurements。所需的是按 owning estimand 冻结的完整 mapping/cardinality/coverage，不是强制 1:1。当 `q_hi` 在 `0.90/0.93/0.96` 之间使 owner action 翻转时，计划也没有 routing-withheld 或 policy-sensitive 终态。

**Closure required:** 在以下两路中选一：

- 真正保持 `:48`：C0 不使用 `q_hi`，只输出 continuous-distribution exposure 与 owner-readable evidence，不自动路由；或
- 承认 C0 是一个明示 policy routing，以新 PI amendment 授权 `q_hi`, 5%, 20% 三个 policy cutoff，用互斥 row-state union 定义 numerator，冻结 ordered pseudocode、逐 RQ mapping cardinality 和 policy-sensitivity-withheld 终态。

### B5 — G7：run contract 仍是字段清单，不是本轮可复审的 immutable object（blocker）

v2 已给出 `operation_id=rq015a_concentration_audit`、validate-only 先行、no-overwrite、single-use receipt 和 fail-closed 类别（`:146-156`）。但 plan 没有给出实际：

- SAP/inventory/schema/crosswalk/code/env manifest 路径与 SHA；
- exact command；
- output root；
- validate-only 和 execution 的独立 operation/receipt 身份；
- receipt/validator 的 machine-readable schema 和唯一 PASS 规则；
- 谁在什么 gate 对未来冻结的对象复审并签发单次授权。

v2 manifest `:1-6` 也印证了这些对象尚未存在于复审包。当前清单只能作为 run-spec 的验收要求，不能证明某个具体 run spec 已经唯一且可安全授权。RQ007 的绑定输入闸明确规定：必需 artifact 缺失、不完整或不可 trace 必须 fail closed（`binding_execution_contract.md:20-23,93-123`）。

**Closure required:** 创建实际的 inventory/schema/crosswalk/SAP/run-spec/validator/fixture 文件，将其全部纳入新 checksum manifest，并在解析任何 measurement fields 之前重启独立复审。复审通过后再由 scoped single-use authorization 先允许 validate-only；validate receipt PASS 后才可授权一次 compute。

### M1 — `q_eff` 代数变换正确，但计划给出的理论域不精确，invalid 分支缺失（major）

对 `K` 个非负归一化权重，有

```text
1/K <= sum(w_i^2) <= 1
1 <= K_eff = 1/sum(w_i^2) <= K
1/K <= q_eff = K_eff/K <= 1
0 <= ipv_error <= 1 - 1/sqrt(K)
```

因此计划 `:29-33` 的两个下界应从 `(0,K]` / `(0,1]` 改为 `[1,K]` / `[1/K,1]`。该差异不改变已有 K=5/7 正常行的计算，但它是 invalid-domain validator 的核心依据。

legacy `estimate_ipv_pair` 先把 D0 行预填为 `IPV=0,error=1`，再从局部 `t=min_observation` 开始估计（`src/sociality_estimation/core/ipv_estimation.py:247-272`）。若不先按 attempt status 分流，`q_eff=1/[K(1-error)^2]` 将除以 0。v2 计划没有冻结统一的计算顺序和 `ipv_error=NaN/inf/out-of-domain`、K 非正整数、grid 重复的 invalid reason codes。K=1 时 `q_eff=1` 是数学上的 trivial support，但不能证明“多个候选近均匀”，secondary bins 应将 singleton grid 设为 unsupported 或单列。

**Closure required:** 更正理论域；冻结 `attempt routing -> finite/domain validation -> q_eff derivation -> policy summary` 顺序；定义 K/grid/duplicate-candidate/error-domain 验证、浮点容差、singleton 分支和 invalid reason codes。

### M2 — attempt status、`q_eff` availability 和 recovery feasibility 仍未被冻结为正交状态（major）

v2 的逐产物表正确识别了三种不同事实：

- WOD Phase1/Phase1b/10Hz/SchemeB 的 error 丢失，但 replay 可恢复且超出本 RQ scope（`:74`）；
- RQ014 全产物无 error，其中 21,576 行 attempt status 也不可由 scene-wide 终态恢复（`:75`）；
- K 不确定时 `q_eff=unknown`（`:35-39`）。

但 ledger 章节只说“`unknown` 的显式取值”，同时又把 `K unknown` 与 duplicate/unmapped/守恒不符一起写为“全部 fail closed”（`:86-93`）。这无法判定 K unknown 是一个允许的 terminal row，还是导致全 product/run 中止。C0 的单一 `unknown` 更混合了 attempt-unknown、q-unknown、K-unknown、mapping-unknown 和 error-not-retained。

**Closure required:** 冻结至少三个正交字段：

- `attempt_status = ATTEMPTED | NOT_ATTEMPTED | UNKNOWN`；
- `q_eff_status = AVAILABLE | ERROR_NOT_RETAINED | K_UNKNOWN | INVALID_ERROR | NOT_APPLICABLE_D0 | UNKNOWN_PROVENANCE`；
- `recovery_status = DIRECT | JOIN | REPLAY_OUT_OF_SCOPE | UNRECOVERABLE | NOT_NEEDED`。

必须为 expected unknown terminal 与意外 schema/provenance defect 定义不同 fail-closed 级别，并说明 C0 的 unknown numerator 取哪些互斥状态的并集。

### M3 — G6 的 disclosure 和 README 已闭合，但 `decision.md` 指针要求与“不修改 decision”仍相互冲突（major）

更新后 disclosure 与 README 已经足以证明 PI 判读 A 不是一个被隐藏的 exposure；这部分是实质关闭。但 disclosure 的补充治理动作明文要求 `decision.md` **与** knowledge README 都加 append-only pointer（`sealed_exposure_disclosure_20260726.md:100-108`），v2 也写“`decision.md` / knowledge README 须加 pointer”（计划 `:132-144`）。现有 README 已加指针，但 `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md:3,24,30` 仍使用 `sealed/untouched`。同时 v2 `:187-191` 又禁止修改任何 owning-RQ `decision.md`。

这个剩余问题不需要重新打开 PI 判读 A，但需要一个唯一治理实现：要么授权只在 `decision.md` 增加不改历史 claims 的 append-only pointer，要么明确指定 manifest-bound README addendum 在 exposure status 上 supersede 旧 decision 措辞，并撤销“两个文件都必须改”的要求。

## Technical failings that need to be addressed before the case is established

| ID | Severity | Technical failing | Verifiable closure condition |
|---|---|---|---|
| B1 | blocker | 旧 PI rederivation condition 未被替代，却继续以同两个值生成 policy/episode/C0；bins 与 instability 函数不完整 | checksum-bound PI amendment 在“只做非下游 policy sensitivity”与“完成重导出”中二选一；冻结不等式、边界和 exact withheld 函数 |
| B2 | blocker | long-ledger 守恒式与 1→2/1→4/3→1 转换矛盾；主键和 product row key 不足；无实际 inventory/schema/fixtures | 冻结逐产物多阶段守恒链、完整 L1 键和具体 input path/hash/schema/crosswalk/fixtures/validator，纳入 manifest 再复审 |
| B3 | blocker | L2/L3 键违反 no-pooling，层间 estimand、因素 SAP 和 episode edge cases 未冻结 | 所有层保留 artifact/measurement-time role，冻结 exact aggregation、SAP、有效行集、时间权重、zero/unknown/withheld 终态及 fixtures |
| B4 | blocker | C0 用 `q_hi` 驱动 owner action，与 bins 禁止下游使用矛盾；numerator 可重复计数，路由条件非本质互斥 | 删除 cutoff 自动路由，或以 PI policy amendment 授权后冻结 disjoint row-state union、ordered pseudocode、逐-RQ cardinality 和 routing-withheld |
| B5 | blocker | run contract 只是未来字段清单，本轮未复审 actual spec/command/env/output/validator | 实际创建并 checksum-bind inventory/schema/SAP/run-spec/validator/fixtures，重启独立复审；复审通过后 scoped validate-only → PASS receipt → single compute authorization |
| M1 | major | `q_eff` 理论下界写错，D0/invalid/K=1 域与计算顺序缺失 | 改为 `K_eff in [1,K]`, `q_eff in [1/K,1]`；冻结 attempt-first 顺序、error/K/grid 验证、容差、singleton/invalid reason codes |
| M2 | major | attempt、q availability、recovery feasibility 未正交，单一 `unknown` 不可解释 | 冻结三个独立 enum、expected-vs-defect fail-closed 级别和 C0 unknown union |
| M3 | major | README pointer 已有，但 decision pointer 必须/禁止两条规则相冲突 | 授权 append-only decision pointer，或正式声明 README addendum supersedes exposure wording 并撤销双文件要求 |

## Minor concerns

### m1 — “全文无 estimability”仍是不可通过的字面验收（minor）

验收 `:170-178` 要求“全文无 estimability”，但计划为了声明禁令与已知限制，已在 `:13-15,38-39,180-185` 合理使用该词。应改为：任何 label、主图、结果、metadata 或结论不得把 attempt/concentration-only 解释为 estimability；否定性边界说明除外。

### m2 — 图文套件和文档自包含性仍未进入硬验收（minor）

v2 要求分位数和 50-bin 直方图（`:35-36`），但 v1 关闭包要求的 mechanism boundary、source x feasibility count/%、case-level 完整分布、带 n/分母/聚类区间的分层热图、evidence-to-owner-action matrix 和 PNG+PDF/SVG 尚未出现在 v2 验收中（`rq015a_three_reviewer_synthesis_v1_20260726.md:159-178`）。计划 `:35-36` 还误将三层单位指向“§4”，实际定义在 §3。这不改变数值 verdict，但会影响非专业读者是否能分清 D0、q-unknown、近均匀和 owner action。

## Assessment against Nature-style criteria

| Axis | Assessment |
|---|---|
| **Originality** | **Moderate and bounded.** 将历史多产物正规化为 attempt / continuous effective-support / recoverability ledger，再将暴露路由回 owning RQ，有实用的测量治理新意。但 `ipv_error`、`K_eff` 和 episode-definition sensitivity 的核心理论已来自 RQ007/RQ015B，不应声称为新的行为真值发现。 |
| **Scientific importance** | **High internal importance; broad importance not yet demonstrated.** 该审计可能改变多个下游 RQ 对 IPV 产物的使用资格，对信息完整性很重要。它不验证估计准确度、行为机制或部署改进，应保持为描述性 measurement audit。 |
| **Interdisciplinary readership** | **Potentially relevant.** “数值存在不等于测量有效”、“数据产物的可恢复性与估计状态正交”、“政策 cutoff 不是自然边界”对逆问题、计量测量和数据治理都有共性。 |
| **Technical soundness** | **Not yet established.** continuous-primary、split 白名单、OnSite 时钟与 no-pooling 是正确设计；但 PI condition、ledger 守恒、L2/L3 estimand、C0 numerator/routing 和 actual run spec 未闭合，会改变中心数字与 owner action。 |
| **Readability for nonspecialists** | **Improved but not standalone.** 主产物改成 continuous distribution 很容易解释；但 physical rows、measurement cells、attempt status、q availability、recovery feasibility 与 C0 action 仍需一幅机制/流程图和一个显式状态表才能避免混淆。 |

## Recommendation posture

**`BLOCKED / REQUEST_CHANGES`**

这不是对 v2 的 continuous-primary 转向或 no-pooling 原则的否定。相反，这两点应在下一版原样保留。阻断来自三类仍会改变结果的问题：旧 PI 命令与新 policy 使用冲突；宽表到 long ledger 的数学与三层 estimand 不唯一；实际 frozen execution objects 未被本轮复审。

下一版应作为一个**真实的 Formal-G1 review bundle**，不再只把 inventory/schema/SAP/run spec 写成未来交付物。修正后须重启至少两路独立复审；在此之前仅允许 metadata/schema/case-ID-only 准备，不允许读取新的 measurement fields。

## Risk / unsupported claims

- 不能由 `ATTEMPTED`、`ipv_error`、`K_eff` 或 `q_eff` 单独证明 IPV 已正确测量、未测出，或满足 RQ007 estimability conjunction。
- 不能由 `q_eff` 归一化自动推出不同 grid、sigma、history window、sampling rate 与 data source 的物理可比性；v2 的 no-pooling 约束应保留。
- 不能将 `4/7` / `0.93` 另行命名为 policy 就视为自动满足或撤销 PI 已冻结的 dev+guard rederivation condition。
- 不能在 `BINS_WITHHELD_UNSTABLE` 时仍无条件使用同一 `q_lo/q_hi` 产生 episode 或 C0 owner action。
- 不能把 physical-row D0 count 与 long-ledger measurement D0 count 当成同一分母。
- 不能把 `unknown` 作为 attempt、q availability、recovery 和 mapping provenance 的共用终态后直接相加。
- 不能以 policy exposure prevalence 证明 owning RQ 的结论低风险、失效、受损或安全性；C0 最多是预先授权的 audit-action policy。
- 不能由 README addendum 的存在隐藏 `decision.md` 仍字面声称 untouched 的剩余冲突；同时也不应重新打开 PI 判读 A。
- 不能在 immutable inventory/schema/SAP/run-spec/validator 未出现于复审 manifest 时宣称运行已唯一可复现或已获执行授权。

## Final verdict and execution status

- **Verdict:** `BLOCKED`
- **Finding counts:** `5 blocker / 3 major / 2 minor`
- **formal_g1_eligible:** `false`
- **execution_authorized:** `false`
- **RQ015A compute:** `NOT AUTHORIZED`
- **HPC:** `NOT NEEDED / NOT USED`

在 B1–B5 关闭并对新 checksum-frozen 执行包完成新一轮独立复审前，不得生成 policy bins、episode summary、C0 routing、最终 ledger 或任何新的 measurement-field 画像。
