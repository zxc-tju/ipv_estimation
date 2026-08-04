# Reviewer 1（技术正确性主审）— RQ015A v3 独立复审

Review setup
- Input scope: `reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`，`reports/plans/RQ015A_ledger_schema_v1.json`，`reports/plans/RQ015A_run_spec_v1.json`，`scripts/rq015a/rq015a_contracts.py`，`tests/test_rq015a_contracts.py`，`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md`，以及一手治理边界 `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/README.md`。
- Assessment boundary: 仅审查 RQ015A v3 的技术合同、治理一致性、可执行性与 fail-closed 性质；不评价 RQ015B 数值修复，不运行任何 RQ015A 数据分析、held-out measurement 读取、replay 或 HPC。
- Shared manuscript claim summary: v3 试图把 v2 中“承诺将来冻结”的对象变成实文件，并通过 schema、唯一算法实现、fixtures 与 run spec 关闭 v2 的主要 blocker，核心主张是：RQ015A 只审计 attempt provenance 与 candidate-weight concentration，不触及完整 estimability、accuracy 或 RQ007 confirmatory inference。
- Visible evidence base: manifest `6/6 OK`；合同代码语法可编译；本地测试 `16/16 passed`；RQ007 README 已追加暴露登记 pointer。
- Missing materials affecting confidence: 仍缺少可执行 command/entrypoint、真实 split_source 路径绑定、central authorization object、逐 artifact 精确 input file/path/hash/source-column 绑定。

Overall technical verdict
- Review outcome: `BLOCKED / REQUEST_CHANGES`
- Findings: `4 blocker / 3 major / 1 minor`
- `formal_g1_eligible=false`
- `execution_authorized=false`

Who would be interested in the results, and why
- 直接相关方是 RQ015A / RQ007 / owning-RQ 的执行与治理人员，因为这版在“contracts 已落地成文件”这一步上相对 v2 有真实进展，能显著缩小 closure gap。
- 但现阶段有兴趣不等于可执行：技术合同虽然更清楚，仍未形成可授权、唯一、无歧义的 Formal-G1 package。

Major strengths
- v3 正确保留了 RQ015A 的构念边界：只讨论 attempt status 与 concentration proxy，不越界到 accuracy 或完整 estimability。
- v3 相比 v2 的最大实质进展是把三类承诺对象写成了真实文件：ledger schema、contracts 实现、run spec。
- 下游不再消费 report bins，这一点在正文、代码和测试三处一致。
- 守恒从错误的单条等式改成三条恒等式，能覆盖 1→2、1→4 与 3→1 的单位变换。
- `sorted + math.fsum` 的逐位确定性是合理的唯一算法设计，避免输入顺序改变结果。

Major concerns
- v3 仍未形成真正可执行的 immutable run spec；当前更多是“运行合同描述”，而不是“唯一可运行对象”。
- v3 在“禁止跨产物 pooling”的原则上仍然是文本闭合、实现未闭合。
- v3 对 PI 阈值重导条件的“正式解除”没有得到同层级治理对象的同步更新，导致计划、暴露登记与 RQ007 README 三方不一致。

Technical failings that need to be addressed before the case is established

1. [BLOCKER] PI 条件解除与既有治理对象冲突，不能视为已闭合。  
   计划在 `RQ015A_plan_v3_concentration_audit_20260726.md` 第 31–34、98–99 行声称 dev+guard 重导 `4/7` 与 `0.93` 的条件已正式解除；但暴露登记第 73–84 行仍明确要求两阈值必须从 dev+guard 重导，RQ007 README 第 6–12 行仍把它写成 mandatory condition。现有证据不足以证明“同一治理命令已被正式 supersede”。  
   Fix: 增加 checksum-bound append-only PI addendum，明确 supersede 原条件；或撤回 v3 中“已解除”的表述，恢复为“未闭合待治理”。

2. [BLOCKER] run spec 不具备唯一可执行性。  
   `RQ015A_run_spec_v1.json` 声称已冻结 operation、环境、output root 与授权对象，但并没有实际 command / entrypoint / launcher 字段；`phases` 只有步骤描述。`split_source` 只写成抽象名 `RQ007 case_split_assignment.csv`，不是 repo 内可解析路径；`authorization_object` 指向 `configs/research_authorization.json#rq015a_concentration_audit`，而 central config 当前没有该对象。  
   Fix: 冻结精确 command、entrypoint、参数、真实 split file path，以及 central authorization registry 中的实际 operation 条目；缺一不可。

3. [BLOCKER] 禁止跨产物 pooling 的 invariant 没有在实现中锁死。  
   schema 宣布 `cross_artifact_pooling.policy = FORBIDDEN`，但 `aggregate_l2()` 只按 `(case_id, perspective, configuration)` 分组，`aggregate_l3()` 只按 `case_id` 分组，不保留 `artifact_id` 或 `measurement_role`。这允许不同 artifact 的 L1 行在同一 case 内被合并。该问题可以通过构造 `sigma01 + M3 + OnSite` 同 case 的输入直接复现。  
   Fix: 在 L2/L3 key 中显式保留 `artifact_id` 与必要 role namespace；或把聚合 API 改为逐 artifact 执行并以类型系统/测试禁止混合输入。

4. [BLOCKER] 逐 artifact 绑定仍不足以支持“唯一对象”执行。  
   schema 只给出了 artifact inventory 与部分 source mapping，但未冻结 exact input file path/hash；`onsite_dense_timeseries` 与 `wod_rq010b_full479_audited` 没有完整 `ipv_error_source` / role-source-column 绑定；M3 虽给出 join keys，但没有把 join 两侧的具体文件对象写死。  
   Fix: 每个 artifact 至少补齐 exact input file(s)、SHA-256、role-source-column 或 join-source binding；否则 execution 仍存在人为自由裁量。

5. [MAJOR] `held_out_parsed_rows = 0` 与“先按 case ID allowlist 过滤再读 measurement 字段”的流程不完全匹配。  
   若先解析 ID 再过滤，`held_out` 的 ID-only 行在流程上是可见的，但不应与 measurement-field parsing 混为同一计数器。现在的字段名容易把合法 ID-only 读取与违规 measurement 解析混淆。  
   Fix: 拆成 `held_out_id_only_rows_seen`、`held_out_measurement_rows_parsed`、`held_out_conclusion_rows` 三个计数器。

6. [MAJOR] `q_eff` 的数学域描述仍不精确。  
   schema 将 `q_eff` 范围写为 `(0, 1]`。对固定正整数 `K` 和规范化非负权重，精确下界应为 `1/K`。当前写法不会立刻导致实现错误，但会使数学说明比真实定义更松，影响合同可审计性。  
   Fix: 将范围改写为 `[1/K, 1]`（条件化说明），或至少在 note 中明确 exact lower bound 取决于 K。

7. [MAJOR] 测试只证明“当前合同代码自洽”，没有证明“执行包已闭合”。  
   `16/16` tests 覆盖了守恒、local position、deterministic mean、bins 不进路由与 schema 自检，但没有覆盖真实 split path 可解析、authorization object 存在、逐 artifact source binding 完整、以及 anti-pooling invariant。  
   Fix: 增加四类 fixture / contract tests：authorization registry presence、split_source path resolution、artifact binding completeness、cross-artifact aggregation rejection。

8. [MINOR] `run spec` 中“精确命令已冻结”的正文表述超出了 JSON 当前内容。  
   计划正文第 74–76 行说 run spec 冻结了“精确命令”，但 JSON 未体现。  
   Fix: 要么补 command 字段，要么收窄正文表述为“运行步骤已冻结，命令尚待补齐”。

Assessment against Nature-style criteria
- Originality: 这里的“原创性”不是科研发现，而是合同闭合方式。v3 的实质进步是把 blocker 从 prose 转为 machine object，这一点是新的、有效的。
- Scientific importance: 对 RQ015A/RQ007 的治理价值较高，因为它把“机械通过”与“正式可执行”区分得更清楚；但重要性不等于当前已经可执行。
- Interdisciplinary readership: 广义上有限，主要面向本研究仓库的研究治理与执行链路；不是对外部广泛科学读者的结果性贡献。
- Technical soundness: 相比 v2 明显提升，但仍未建立唯一执行对象，且 anti-pooling 与治理 supersede 两个关键点未闭合，因此技术上不能放行。
- Readability for nonspecialists: 对本仓库熟悉者可读；对非本项目读者仍较依赖上下文，不影响本次技术结论。

Recommendation posture
- promising and materially improved, but currently not established from the provided evidence
- 具体判定：`REQUEST CHANGES`

Risk / unsupported claims
- 不支持“PI 的 dev+guard 阈值重导条件已正式解除”这一说法；现有绑定对象相互冲突。
- 不支持“run spec 已冻结精确命令并可执行”这一说法；JSON 中没有 command/entrypoint，authorization object 也不存在。
- 不支持“禁止跨产物 pooling 已被代码层强制”这一说法；当前聚合 key 允许混合 artifact。
- 不支持“逐产物输入对象已唯一绑定”这一说法；仍缺少 exact file path/hash/source binding。
- `16/16 passed` 只能证明合同代码在当前 fixtures 下自洽，不能证明 Formal G1 达成，也不能证明 execution 可以授权。
