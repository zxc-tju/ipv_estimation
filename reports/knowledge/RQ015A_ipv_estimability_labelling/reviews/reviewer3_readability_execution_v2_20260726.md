# RQ015A v2 独立复审 — Reviewer 3（跨学科可读性、可复现执行与治理）

状态：`COMPLETE / BLOCKED / REQUEST_CHANGES`  
`formal_g1_eligible=false`｜`execution_authorized=false`｜未运行 RQ015A 计算

## 1. Review setup

- 冻结对象：`reports/plans/RQ015A_plan_v2_concentration_audit_20260726.md`
- 对象 SHA-256：`9186c95eb6d84ee56626f6e96cb75d2f0422297446824ea20848e34258ab9a67`
- baseline manifest：`reports/plans/RQ015A_plan_v2_checksums_20260726.sha256`
- manifest SHA-256：`c63f7564269fb98ccca776e5cb52bdbfdc9760b7d06a948b1440164b3917b96e`
- 现场校验：manifest 内 `6/6 OK`。
- 输入范围：v2 plan、v1 三路综合、PI held-out 暴露登记、RQ007 契约/split/README/decision、实际产物 schema/manifest，以及对 v2 支持件的限定范围搜索。
- 独立性：本路未读取 Reviewer 1/2 的任何 v2 输出，未读取未来 v2 synthesis，未与其他 v2 Reviewer 交流。
- 评估边界：只复审计划是否可授权、可复现、可机器验收；不运行分析，不修改 plan/code/data/status/decision。

v2 的中心主张是：连续 `q_eff=K_eff/K` 分布为主产物，三档降为政策性 secondary summary；并以跨产物合同、三层单位、C0 路由和 fail-closed 运行链声称逐条关闭 G2-G7（plan `:6-25`）。

可见冻结包只有 plan 与六项 manifest。manifest `:1-6` 绑定 v2 plan、RQ015B plan、held-out disclosure、RQ007 README、RQ015B log-domain 代码及测试；没有绑定 artifact inventory、ledger schema、measurement-role crosswalk、per-product fixtures、SAP、C0 rubric、run spec、environment、exact command、validator 或 receipt template。限定范围搜索也未找到可替代这些对象的 RQ015A v2 实现。

## 2. Overall assessment

v2 的科学表述明显优于 v1。连续分布优先，避免强迫数据产生并不存在的天然语义阈值；禁止跨产物 pooling、区分 RQ007 split 与 RQ009 fold、修正 held-out 暴露措辞、作废“低资格风险”也都是实质进步。

但它仍把应在复审前存在的冻结对象写成未来交付物。“schema 须附 fixture”“run spec 将绑定 exact command”和“receipt 机器判 PASS/FAIL”不等于这些对象已经存在。更严重的是，冻结文字内部有三项可执行冲突：

1. manifest-bound PI/RQ007 文件仍强制 dev/guard 重导阈值，v2 却保留旧阈值并宣布无需重导；
2. ledger 要求 `input_rows = sum(terminal_rows)`，但 sigma01/OnSite 必须一行展开多 measurement，M3 又必须三行去重为一 anchor；
3. plan `:48` 禁止 policy bins 用于下游判定，C0 `:123-125` 却用同一 `q_hi` 触发 `OWNER_REANALYSIS_REQUIRED`。

因此当前不能进入 Formal G1。`6/6 OK` 只证明列出的六个文件未漂移，不证明 G2-G7 已实质关闭。

## 3. Who would be interested in the results, and why

- 自动驾驶交互与社会行为估计研究者：可把“估计器运行过”“权重有多少候选支持”和“是否真正准确/可估计”分开。
- 可靠性、abstention 与验证工程人员：provenance、explicit unknown、no-overwrite 与 owner routing 可形成通用审计模式。
- 使用潜变量作下游特征的机器学习研究者：数值存在不等于数值携带候选间判别信息。
- 研究治理与可复现性读者：held-out 暴露、split 隔离、append-only 治理和机器验收具有可迁移性。

目前只建立了高价值的项目内部 measurement audit，尚未提供相对既有工作的原创性定位，也未证明跨系统的广泛科学影响。

## 4. Major strengths

1. Plan `:11-25,180-185` 保持 attempt/concentration 与 accuracy/estimability 的推断断点。
2. `:27-48` 以连续 `q_eff` 为 primary，并为 policy summary 加 3×3 sensitivity 与 `BINS_WITHHELD_UNSTABLE`。
3. `:77-84` 禁止把 sigma01 的复制/join 产物 pooling 成伪语料级证据。
4. `:65-75` 明示 M3 缺 error、OnSite 需局部位置、WOD replay 超范围、RQ014 终态不能还原逐行 attempt。
5. `:95-130` 引入 three-level units、minimum/zero support、unknown、case bootstrap，并以 `NO_AUDIT_TRIGGER_DETECTED` 取代无证据的低风险结论。
6. Disclosure `:89-108` 与 RQ007 README `:3-12,40-50` 已改为程序解析聚合但未显示/导出/人工检视单行值。

## 5. Major concerns

### 5.1 Blockers

#### B1. v2 与 manifest-bound PI/RQ007 条件冲突，G2 未治理性关闭

Plan `:16-25,41-48` 固定旧 `4/7` 和 `0.93` 为政策边界并取消重导。可是 manifest-bound disclosure `:71-84` 仍规定两阈值必须从 dev+guard 重导，重导前不得用于结论画像；manifest-bound RQ007 README `:6-12` 也仍称其为 mandatory condition。把阈值改名 policy 不会自动撤销 PI 条件，尤其 C0 仍让 `q_hi=0.93` 产生 owner-action 后果。

**关闭条件**: PI/RQ007 治理层冻结 addendum，明确是否以 continuous-primary + policy-summary 取代旧重导条件。若保留政策阈值，须登记 `4/7`、`0.93`、`10 pp`、`5%`、`20%` 均为治理选择，并冻结 sensitivity 改变 C0 路由时的 withhold/escalation 规则。

#### B2. authoritative inventory/schema/crosswalk/fixtures 不在冻结包中，守恒式也无法成立

Plan `:65-93,160` 的 Markdown 表没有逐产物 checksum-bound path/hash、producer/run/version、schema path/hash、field/type/nullability、raw key、K/grid/window/rate、split scope、expected raw rows 与 normalized measurements。M3 未绑定 prediction/feature partitions 与 1:1 fixture；OnSite `:71-72` 仍写 role 未冻结，filter/group/tie/duplicate/断帧/重启规则均缺；WOD full479 D0 仍是逐产物确认，其他 WOD/RQ014 也无 exact path/hash/field/key。

而 `:88-93` 的 `input_rows = sum(terminal_rows)` 与本计划自身冲突：sigma01 的 2,490,992 宽表 rows 要展开成 4,981,984 agent-values；OnSite 每 physical row 有四 channels；M3 每 anchor 三个 alpha rows 要去重成一个 measurement。raw row 不是共同守恒单位。

**关闭条件**: 将 machine-readable inventory/schema/crosswalk/fixtures 纳入新 manifest；守恒拆成 `raw physical rows -> expanded role rows -> deduplicated normalized measurements`，逐产物冻结 expansion/dedup/cardinality 与期望计数。预期 unavailable/replay-out-of-scope/attempt-unknown 是可审计终态，不能与 run-fatal unknown 混为一类。

#### B3. 三层 key 没有对应三层统计量，SAP 与公式 fixtures 缺失

Plan `:95-115` 定义了 L1/L2/L3 key，却未定义 `q_eff` 从 L1 到 L2/L3 是均值、中位数、ECDF、尾部占比，还是 L2 等权 case 摘要。不同实现会对长 episode 产生不同权重。bootstrap 未绑定 estimand/strata/空层/重复 case ID/quantile 实现；hard-filtered summary 在无入选帧时无定义，`1-q_eff` 在全 `q_eff=1` 时分母为零；交付物 `:164` 的因素探索没有 response、predictor、时间有效性、模型、聚类、missingness 或否证检查；`:114` 要求的 fixtures 也不在 manifest。

**关闭条件**: checksum-bind 独立 SAP；每个输出冻结 unit、numerator、denominator、aggregation/equal-weighting、missing/unknown、support、uncertainty、primary/sensitivity 与字段；每个公式附可手算 input-to-expected-output fixture，并定义 zero-support/zero-weight 终态。

#### B4. C0 四终态并未互斥穷尽，且与禁止 bins 下游判定冲突

Plan `:116-130` 仍有以下不唯一处：

- `:48` 禁止 policy bins 下游判定，`:123-125` 却用同一 `q_hi` 触发 owner action；
- `q_eff>=q_hi`、`NOT_ATTEMPTED`、`unknown` 是集合并集还是占比相加未说明；正交字段直接相加会双重计数；
- `INDETERMINATE` 优先于 `NOT_APPLICABLE`，而不使用 IPV 的 RQ 自然无法建立 IPV 1:1 映射，若不先判 applicability 会错路由；
- owning-RQ rows 可能是 anchor×alpha、case aggregate 或窗口摘要，不一定与 L1 ledger 1:1；
- `5%/20%` 未明确登记为 policy；产品若触发 `BINS_WITHHELD_UNSTABLE`，C0 是否仍用 0.93 未规定。

**关闭条件**: checksum-bind machine C0 rubric 与 owning-RQ link crosswalk。先判 `uses_ipv`；对使用 IPV 者用明确行集合并集，冻结 link cardinality、分母、unmapped/duplicate 与 priority；全部阈值标 policy；路由随 sensitivity 改变时进入显式 policy-sensitive/owner-review 终态；用穷尽 fixture 覆盖重叠和边界。

#### B5. held-out leakage 指标不精确，RQ007 decision pointer 与禁止修改 decision 冲突

Plan `:57-63,132-144,172-178` 的 ID-first 原则正确，但逐行源必须先解析 case ID 才能判断 held_out。字面 `held_out parsed rows=0` 无法区分 ID-only 与 measurement-field 解析，会让正确实现误失败，或让错误实现用含混计数过门。

另外 `:139-140` 要求 RQ007 `decision.md / knowledge README` 加 pointer，`:129,187-190` 又禁止修改 owning-RQ decision。实际 README 已有 pointer，但 `decision.md:3,24,30` 仍写 `sealed/untouched`，且 decision 未在 v2 manifest 中。

**关闭条件**: counters 至少拆为 `held_out_id_only_rows_seen`、`held_out_measurement_fields_parsed_rows`、`held_out_normalized_measurements`、`held_out_conclusion_rows`，后三者必须为 0。将 assignment/split-freeze 的 SHA `90d8bb91e68f9b5e0596cf1ae915eb22b01a5c4ccffbad00c0b446efa46d537d` / `99d81147c761981c87bb6a052044a92db35a76003033d2ff46cd462a79f88570` 纳入 manifest。由 PI 授权只追加治理 pointer 的窄例外，或建立不改 frozen decision bytes 的权威 addendum/overlay。

#### B6. operation ID 不等于 immutable run spec，G7 执行链不存在

Plan `:146-156` 只有 operation ID 与未来属性，没有 exact command/entrypoint/workdir、Python/environment 与包版本、input allowlist/hashes、read-only policy、output root、expected outputs、resource bound、validate-only command、receipt schema/template 或 validator implementation。manifest 中 `reliability_logdomain.py` 及测试属于 RQ015B，不是 RQ015A ledger/portrait/validator。

**关闭条件**: checksum-bind run spec、environment manifest、receipt schema、validator 及 validate-only/full-run exact commands；run spec 绑定 plan/SAP/inventory/schema/crosswalk/fixtures/code/env/command/output-root SHAs。validate-only 通过后，另由有权人为精确 run-spec SHA 签发 single-use receipt；复审通过本身不产生执行权。

### 5.2 Major issues

#### M1. `q_eff` 理论域仍写错，invalid/tolerance 分支未定义

Plan `:29-39` 写 `K_eff in (0,K]`、`q_eff in (0,1]`。对 K 个归一化非负权重，正确域为 `sum(w_i^2) in [1/K,1]`、`K_eff in [1,K]`、`q_eff in [1/K,1]`。还需冻结 K 为整数且 `K>=1`、error finite/range、边界容差、越界/无穷 reason code，以及从 stored error 计算还是与原权重交叉校验。

#### M2. 图文交付只有 50-bin 直方图，没有 claim-indexed 证据套件

Plan `:35-36,158-168` 未冻结每个主张对应的 figure/table、panel source、n/分母、unknown/pending、CI、配色/图例、caption/alt text、source-table hash、figure manifest 或 PNG+PDF/SVG。

最小图文合同应包括：

1. attempt provenance → `q_eff` proxy → product exposure → owner action 的机制/推断断点图；
2. `source × feasibility/terminal state` count+% 图，区分 unknown 与 replay-out-of-scope；
3. product/config/K 分面的 continuous ECDF/密度+分位图，并显示 policy sensitivity/withheld；
4. L2/L3 完整分布，显示变长 episode、minimum/zero support 与 case-clustered CI；
5. `source × geometry × window` 热图，每格显示 n、分母、tail metric 与 CI；
6. owning-RQ evidence-to-action matrix，分列 applicability、mapping coverage、exposure、policy sensitivity、reason 与 action。

每图应输出 PNG 与 PDF/SVG，并附 panel-level source table 和 checksum manifest。

#### M3. 状态入口不一致，“G2-G7 已冻结”超过证据

`STUDIES.md:50` 已指向 v2，但称 G2-G7 已逐条冻结；这与 closure artifacts 缺失不符。`START_HERE.md:15-64`、knowledge README `:3-49` 和 study README `:1-14` 仍以 v1 BLOCKED 为现行状态，没有登记 v2 plan/manifest 与 awaiting-review、no-authorization 状态。应统一为“v2 已起草并 freeze；是否关闭 gates 待独立复审”，不能在复审前写已关闭。

#### M4. 无 pooling 不等于跨配置可比

Plan `:77-84` 正确禁止合并派生观测，但 `K_eff/K` 无量纲化不证明不同 grid、sigma/hw/window/rate、model 或 source 具有相同测量语义。可并列描述分布，但在无 matched-config/invariance 证据前，不得把高低排名解释为更可辨识、更可靠或数据质量更高；计划应区分 matched comparison 与 descriptive side-by-side。

### 5.3 Minor issues

#### m1. “全文无 estimability”不是合理 lint

Plan `:13-14,172,182-184` 自身必须在限制和 RQ007 边界中提到该词。应限定为 output labels、figure/table titles、headline claims 与 result interpretation 不得把 RQ015A concentration-only 命名为 estimability；否定、历史和边界语境允许出现。

#### m2. quantile 与 histogram 的跨平台细节未锁定

Plan `:35-36` 未规定 quantile interpolation、bin 左/右闭区间（尤其 `q_eff=1`）、显示精度、empty-bin/unknown 计数及是否加权。应写入 SAP 并以边界 fixture 锁定。

## 6. Technical failings and exact closure package

| ID | Technical failing | 直接影响 | 可验收关闭件 |
|---|---|---|---|
| B1 | PI 重导条件与 v2 policy reuse 冲突 | 同一冻结包有两套授权规则 | PI/RQ007 addendum + SHA；policy/sensitivity/withhold 一致 |
| B2 | 无 inventory/schema/crosswalk/fixtures，raw-row 守恒错误 | 输入、键、类型、分母与 D0 不唯一 | 全部入 manifest；raw→expanded→dedup 三段守恒金标 PASS |
| B3 | SAP 无三层 estimands/公式/fixtures | 实现者对长 case 权重不同 | checksum-bound SAP + 手算 fixtures + zero-support/weight 状态 |
| B4 | C0 重叠、映射未定、bins 禁令自相矛盾 | owner action 可被实现选择改变 | machine rubric + link crosswalk + 穷尽边界 fixtures |
| B5 | leakage counter 含混；decision pointer 冲突 | 过滤验收和权威事实不唯一 | 分层 counters + split SHAs + PI-authorized pointer/addendum |
| B6 | 无 run spec/env/command/validator/receipt | 无对象可授权、无机器完成证据 | 对象全 checksum-bound；validate-only PASS；再签 single-use receipt |
| M1 | `q_eff` domain/invalid 规则错缺 | 边界行与守恒不一致 | domain/finite/tolerance/reason codes + fixtures |
| M2 | 无 claim-indexed 图文套件 | 非专家看不到机制、分母、unknown | figure contract + source tables + PNG/PDF/SVG + manifest |
| M3 | 状态入口不一致 | 新 agent 可误判当前权威状态 | START_HERE/STUDIES/两层 README 统一 |
| M4 | dimensionless 被潜在误读为跨配置等义 | 产物高低可被误读为可靠性排名 | matched vs descriptive 标识；禁止 accuracy/quality 解读 |

## 7. Assessment against Nature-style criteria

| Axis | Assessment | Evidence-bounded readout |
|---|---|---|
| Originality | **Not assessable** | 无与 effective-support、abstention、measurement audit 或 provenance validation 既有工作的对比。 |
| Scientific importance | **内部重要，广泛意义未证明** | 可防止 non-informative 数值被当成正常测量，但 A 不识别成因、不评准确度、不重估下游效应。 |
| Interdisciplinary readership | **有潜力，尚未形成 broad-interest case** | audit 链可迁移，但文本仍依赖 RQ/D0/C0/alpha 等内部语言。 |
| Technical soundness | **不通过** | 六个 blocker 会改变阈值权限、输入 universe、守恒、聚合、C0 路由和运行合法性。 |
| Readability for nonspecialists | **较 v1 改善，仍需大修** | continuous-first 清楚，但权威文件冲突、术语密集且无机制与 claim-indexed 图。 |
| Reproducibility / governance | **不通过** | manifest 锁定了意图文本和旁路对象，没有锁定决定数字与授权的 closure package。 |

## 8. Recommendation posture

**Currently not established from the frozen package; supportive of the scientific direction after the execution and governance contract is rebuilt.**

应保留 continuous-primary、no pooling、explicit unknown 和 no accuracy/estimability claim。下一版必须把实际 inventory/schema/crosswalk/fixtures/SAP/C0 rubric/run spec/validator 一并放入 checksum manifest，再对整个 closure package 重启独立复审。

## 9. Risk / unsupported claims

- 不能声称 v2 已关闭 G2-G7；当前只描述了拟关闭方向。
- `6/6 OK` 不证明 ledger/SAP/C0/operation 可复现，因为它们不在 manifest。
- 不能无 addendum 地重用旧 `4/7/0.93` 产生 owner-action 后果。
- 不能声称 raw `input_rows = terminal_rows` 已适用于展开和去重产物。
- 不能声称 C0 四终态已互斥穷尽。
- 不能把 `held_out parsed rows=0` 当无歧义 leakage 证据。
- 不能继续无条件引用 RQ007 decision 的 `sealed/untouched` 历史文字。
- 不能由 attempt/error/`K_eff`/`q_eff` 推出准确度、真实 estimability 或 downstream damage。
- 不能由无量纲 `q_eff` 推出不同配置或 source 的物理等义。
- 不能声称 RQ015A 已获执行授权；本复审不是 receipt。

## 10. Final verdict and counts

| Item | Result |
|---|---:|
| Verdict | **`BLOCKED / REQUEST_CHANGES`** |
| Blocker | **6** |
| Major | **4** |
| Minor | **2** |
| `formal_g1_eligible` | **`false`** |
| `execution_authorized` | **`false`** |

在新的 checksum-bound closure package 通过独立复审、且有权人对精确 run-spec SHA 签发 scoped single-use receipt 前：不得构建最终 ledger/画像，不得用 policy bins 产生 C0 owner action，不得解析新的 held-out measurement fields，不得 replay WOD/RQ014，不得修改 accepted claim/decision，也不得以本报告作为执行授权。

