# RQ015A v0 独立复审 — Reviewer 3（跨学科可读性 × 可复现执行/治理）

## Review setup

- **Input scope**：冻结计划 `reports/plans/RQ015A_plan_v0_ipv_estimability_labelling_20260726.md`，审查 SHA-256 为 `cd352390d816c12c77942211ad73479a5d3b71c43c8c801150ab3e25cfa9fea8`；`reports/plans/RQ015AB_split_checksums_20260726.sha256`（自身 SHA-256 `06f4d7718db251190dfab294dff66f86712cd447a828abbd21fb2dd7feec9229`）逐项校验 `5/5 OK`。
- **Independent-review boundary**：本路线未读取任何 RQ015A 新复审文件，未与其他 Reviewer 交换意见；只读核查冻结计划、其直接引用物、RQ007 split provenance、RQ009 schema/manifest、现行估计器接口及现存数据文件的 metadata/schema。未运行 RQ015A 分析，未改计划、代码、数据或状态索引。
- **Assessment boundary**：这是计划级复审，不是对尚未生成的非 sealed 画像、预测子模型或 C0 资格结论的结果复核。此处可以判断研究问题、规格与执行链是否足以无歧义实施；不能确认剔除 sealed 后的任何比例或效应。
- **Shared claim summary**：RQ015A 拟把历史 IPV 数值的“是否实际尝试估计”与基于权重集中度的三态可估计性显式化，分别在 agent-frame 与 case 层面画像，并向六个下游 RQ 给出资格判断；它明确不改估计器、不部署闸门、不重估既有结论（计划 `:16-27`, `:115-129`, `:144-150`）。
- **Visible evidence base**：计划给出 D0、全语料先验比例、`K_eff` 变换、四层可行性框架、双口径画像和边界（计划 `:34-70`, `:80-113`）。RQ007 已存在可精确复用的 outcome-blind case split 契约及 case-ID 表，而非只能“定位或重建”：`split_freeze.json:24-41,45-53,76-114`。
- **Missing materials affecting confidence**：RQ015A 专属输入清单、逐产物字段语义 crosswalk、ledger schema/主键、聚合与预测子统计规格、本地 operation authority、可执行命令/环境、machine-readable validator 与验收状态均未提供。RQ015A 执行层目前只说明 Formal G1 后才建立 RUN_ID（`reports/studies/RQ015A_ipv_estimability_labelling/README.md:1`）。

## Reviewer 3

### Overall assessment

**Verdict: `BLOCKED`**  
**Finding count: 4 blocker / 5 major / 2 minor.**

研究问题本身清楚且值得做：把“未测量”“测量但权重平坦”“测量且集中”从同一个数值 0 中拆开，是下游统计解释的必要前提。然而，当前 v0 仍是研究意图与范围说明，不是可复现、可授权的执行契约。最严重的问题是：不同产物中的“一行”并非同一分析单位，`ipv_error` 也可能指当前 counterpart、未来 target 或两个 history-window 版本；计划既未冻结 authoritative inventory，也未冻结长表键、去重和 join 规则。按现文执行，两个诚实实现者可以得到不同分母、不同状态比例和不同下游资格结论。

因此，本复审不支持进入 Formal G1；这不是对研究方向的否定，而是对当前冻结字节尚不能唯一决定执行与验收的判断。

### Who would be interested in the results, and why

- 自动驾驶交互建模与运行时验证研究者：他们需要知道模型输入中的“中性 0”是行为中性还是测量失败。
- 计算行为科学、人机交互与测量学研究者：该问题展示了代理量、缺失机制和聚合单位如何共同改变行为解释。
- 不确定性量化、可靠性工程与 ML 治理读者：RQ015A 把逐行可追溯状态与下游资格审计连接起来，若执行契约闭合，可成为比单纯展示误差分布更一般的方法示例。
- 数据工程与科学可复现性读者：同一潜变量在 CSV 双列、anchor matrix、三 nominal prediction rows、OnSite 多窗口和 WOD 局部产物中的跨工件 lineage，是一个具有普遍性的 provenance 问题。

目前的跨学科吸引力仍是“潜在的”：计划没有把内部 RQ 编号、D/L 分层和 M3 术语翻译成一个无需项目背景即可理解的机制图，也尚未给出能够证明下游结论受影响程度的已执行结果。

### Major strengths

1. **边界诚实**：明确限定为测量学描述研究，不提出行为因果主张（计划 `:16-21`），并承认历史产物无法拆分 D1/D2/D3（计划 `:29-32`）。
2. **避免把集中度误称为误差真值**：计划说明该量只是 identifiability proxy，不得表述为 IPV 估计误差的直接度量（计划 `:67-70`）。
3. **认识到分析单位会改变结论**：要求帧级和 case 级并列，且指出帧/锚点消费者与 episode 消费者受损不同（计划 `:92-105`）。
4. **下游选择偏倚意识充分**：C0 要求逐 RQ 说明筛选是否改变估计量/暴露，并明确可估计性与结果相关时的选择偏倚风险（计划 `:121-127`）。
5. **保留执行边界**：不改估计器、不部署闸门、不重训 M3、不覆盖冻结产物，并保持 `execution_authorized=false`（计划 `:3`, `:144-160`）。

### Major concerns

#### B-R3-01 — authoritative input universe 与字段语义未冻结（BLOCKER）

计划只用自然语言列出 “sigma01 时间序列、RQ009 目标/特征矩阵与 M3 预测、RQ012B OnSite、RQ010B/RQ014 WOD” （计划 `:89-90`），没有逐项给出 exact path、SHA-256、size、producer commit/version、schema、measurement field、主键、预期行数或本地可得性。它同时声称 `ipv_error` 已存在于“全部主要产物”（计划 `:23-27`），却又承认有些产物需要 RQ015B 重算或根本无法恢复（计划 `:80-87`）；这两段不能共同充当输入合同。

这不是形式问题。RQ009 的 feature dictionary 同时包含：当前 counterpart 的 `counterpart_ipv_error_current`、未来 target 的 `target_ipv_error_future` 以及 M4-only ego self-anchor error（`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/feature_dictionary.csv:52-59`）。三者回答不同问题，计划没有规定 RQ015A 对哪一个打标。M3 test prediction manifest 又记录 `1,270,566` anchors 但 `3,811,698` rows（每 anchor 对应 80/90/95 三个 nominal level；`data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/prediction_manifest.json:135-145`）；只读 schema 检查还确认 prediction 表本身没有 `ipv_error`。若按“每一行”扫描，会把同一个 target 测量三计。

**Closure required**：冻结一份 machine-readable `input_inventory`。每个 artifact 至少绑定 `artifact_id/path/sha256/bytes/producer_commit/estimator_version/row_unit/measurement_fields/primary_key/K/grid_id/min_observation/split_scope/expected_rows/local_availability`，并逐字段声明 `current counterpart / future target / ego self-anchor / not applicable`。路径须使用当前仓库可解析的 repo-relative authority，不得靠名称搜索选择最新版。

#### B-R3-02 — `estimability_ledger` 没有 schema、主键与行守恒合同（BLOCKER）

计划的研究问题说“每一行”是否测出 IPV（计划 `:16-19`），但其先验统计的实际单位是两个 agent-value/源行（计划 `:39-50`）；sigma01 源 schema 也确实在一条 frame row 内保存两套 IPV/error（`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/02_inventory/timeseries_provenance_manifest.csv:6-9`）。RQ009 则冻结了 `(case_key, anchor_frame_index, perspective)` 一类 anchor key（`feature_dictionary.csv:2-10`）。OnSite 现存时间序列还在同一行保存 hw4/hw10 × ego/counterpart 四套 IPV/error。当前计划仅写“schema 先冻结”（计划 `:131-138`），没有列任何字段、类型、枚举、唯一键或 consumer join。

**Closure required**：先冻结版本化长表 schema，明确一行必须是一个 `artifact × case/unit × timestamp/anchor × perspective × measurement_role × estimator_version × window` 测量；至少包含 source artifact hash、source row locator、原始 IPV/error、K/grid provenance、attempt status、estimability status、feasibility layer、split status/provenance、reason code 和 RQ015B enrichment status。须规定唯一键、允许的一对多关系、prediction nominal 去重、双 agent 展开、missing/nonfinite 处理及逐 artifact 行守恒式。任何 duplicate key、unmapped measurement 或 conservation mismatch 都应 fail closed。

#### B-R3-03 — RQ007 sealed 排除虽已有权威工件，但计划没有绑定它（BLOCKER）

计划把“定位或重建” split 作为第一步（计划 `:74-78`），但现有 RQ007 `split_freeze.json` 已冻结：键为 `scene_unique_id`（`:24-32`）、总数与三 split 计数（`:37-41`）、固定 salt（`:51-53`）和逐 split hash-bucket 规则及 assignment path（`:76-114`）。只读核查确认实际表：

- `data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/case_split_assignment.csv`
- SHA-256 `90d8bb91e68f9b5e0596cf1ae915eb22b01a5c4ccffbad00c0b446efa46d537d`
- `38,228` unique case IDs、duplicate `0`、development/guard/held_out = `19,258/7,628/11,342`。

RQ009 也明确 `case_key = scene_unique_id`（`feature_dictionary.csv:2-3`），所以 InterHub join 可以现在就冻结。相反，OnSite/WOD 不属于 RQ007 case universe；它们应显式为 `RQ007_SPLIT_NOT_APPLICABLE`，不能与 `unknown` 或 join failure 混在一起。当前验收只写“sealed 全程未参与任何阈值”（计划 `:140-142`），没有 overlap=0、unmatched=0/有界、hash match 或重算一致性等机器门。

**Closure required**：直接绑定上述 split artifact + `split_freeze.json` 的 hash；“重建”只允许作为按 frozen salt 的 byte/row-equivalent verifier，不得生成新 assignment authority。冻结 join key、cardinality 和范围规则，并输出 machine receipt：split hash、counts、duplicate/unmatched、每个 InterHub 输出与 held_out 的 overlap、进入每张统计表的 held_out count（必须 0）。外部数据集使用明确的 not-applicable 状态。

#### B-R3-04 — 没有可执行的本地 operation、环境、授权与机器验收路径（BLOCKER）

计划将资源描述为“本地 CPU 分钟级”（计划 `:152-155`），然后只给出 `scoped decision → allowlist → 单次执行`（计划 `:157-160`）。它未命名 operation ID、authority 文件、local runner、exact command、Python/environment lock、RUN_ID/output root、immutable spec、validate-only 行为或 single-use receipt。执行层 README 也只是说这些要在 Formal G1 与 scoped authorization 后建立（`reports/studies/RQ015A_ipv_estimability_labelling/README.md:1`）。

冻结 bundle 中唯一相关扫描脚本只接受一个 CSV、硬编码列号并把全语料摘要打印到 stdout；它没有读取 split、没有生成 ledger/manifest/validator，也不覆盖 RQ009/OnSite/WOD（`reports/plans/prompts/RQ015_portrait_scan_v1.sh:2-18`）。因此它只能复现立项先验，不能作为 RQ015A 执行入口。

**Closure required**：冻结独立于 RQ014 HPC launcher 的本地 authorization contract：固定 operation ID、reviewed input/spec/code/env hashes、唯一 exact command、repo-relative output root、dry-run/validate-only、禁止覆盖、RUN_ID allocation、terminal receipt 和一次性授权状态。再提供 machine-readable acceptance JSON/schema 与 validator；Formal G1 只证明 reviewed bytes，无权自动执行。

#### M-R3-01 — K 的定义域与异常值合同不完整（MAJOR）

三态判据在 K≤4 时重叠：例如 K=4、K_eff=4 同时满足 `ESTIMABLE (≤4)` 与 `NOT_ESTIMABLE (≥0.93K=3.72)`（计划 `:55-65`）。现行接口允许调用者显式传入候选网格，而不仅限默认 K=5/7（`src/sociality_estimation/core/ipv_estimation.py:220-236`），所以这不是纯理论边界。计划也未规定 `ipv_error` 非有限、<0、>理论上界 `1-1/sqrt(K)` 或浮点超界时的状态。

**Closure required**：冻结允许 K 集合（若仅 {5,7} 就明确拒绝其他 K），或给出无重叠的优先级/阈值；同时冻结 finite/range/tolerance 验证和 `INVALID_SOURCE_VALUE` 处置。K/grid provenance 不可从当前代码猜测历史产物；必须来自绑定的 producer config/commit，否则为 `unknown`。

#### M-R3-02 — 帧级、case 级和预测子分析缺少统计规格（MAJOR）

计划要求双口径、两种 episode 摘要和预测子探索（计划 `:92-113`），但没有定义：两个 agent 是否分开或合并、case 的分母、minimum support、unknown/pending/D0 的分母处理、可估计性权重公式、数据源权重、case/source 聚类、区间/效应量、预测模型、验证方式、missingness 和多重探索标记。RQ007 只冻结了一个历史 all-valid 等权平均公式（`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/02_inventory/summary_formula.md:5-23`），不能自动定义 RQ015A 新承诺的“仅可估计帧”与“可估计性加权”对照。

**Closure required**：在执行前冻结 frame/person/case 分母与 row-weighting、每个摘要的公式、minimum estimable support、cluster-aware uncertainty，以及探索性预测子的 outcome、features、model/validation、effect-size 和 reporting-only 边界。探索性可以不做 confirmatory p-value gate，但不能没有可重算规格。

#### M-R3-03 — 与 RQ015B 的终态与 enrichment 协议不闭合，且“固有识别极限”越界（MAJOR）

RQ015A 正确承认 D1/D2/D3 需 `min_mse` 等新统计量并归 RQ015B（计划 `:29-32`, `:80-87`）；RQ015B 也写明 L3 重算服务于 A，且 B3/覆盖审计依赖 A 的 fold provenance（`reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md:14-16`）。但 A 的验收要求“四层分类完整”（计划 `:140-142`），没有定义 `pending_RQ015B` 是否允许 A 完成，也没有规定 B 以后如何按稳定键追加而不覆盖 A v0 ledger。

同时，A 仍说其预测子分析帮助判断“可修 vs 固有识别极限”（计划 `:111-113`）；B 已明确 D2 只能称为“当前网格与模型下平坦”，没有网格/模型敏感性就不得称为固有不可辨识（RQ015B 计划 `:126-127`）。

**Closure required**：定义 A 的独立终态（例如 `COMPLETE_BOUNDED_L1_L2_L4_WITH_L3_PENDING`）与 B 的 append-only enrichment key/schema/version；明确 A 不等待 B 也不声称机制拆分完成。删除“固有识别极限”，改为“当前历史配置下的可估计性相关因素/需 B 做机制鉴别”。

#### M-R3-04 — C0 资格矩阵没有可重算的判定规则（MAJOR）

C0 要覆盖六个 RQ 并输出 `可重估 / 需重新设计 / 不适用`（计划 `:115-129`），但输入覆盖清单并未逐一列出 RQ003/RQ011B 的具体消费工件（计划 `:89-90`），也没有类别判定树、evidence columns、unknown/pending 的优先级或 owner handoff。不同 Reviewer 可对同一选择偏倚风险给出不同类别，无法独立复算。

**Closure required**：冻结逐 RQ consumer inventory 和 decision table，至少包含 `consumer_artifact_hash / measurement_role / analysis_unit / aggregation / estimability_join_coverage / estimand_changed / exposure_changed / selection_risk / required_sensitivity / eligibility_class / reason_code / owner`；给出类别的确定性规则，无法判定必须输出 `UNKNOWN_REQUIRES_OWNER_REVIEW`，不得强塞三类之一。

#### M-R3-05 — 跨学科机制叙事与图形合同不足（MAJOR）

开头的“到底有没有测出 IPV”非常直观（计划 `:16-21`），但之后连续引入 D0–D3、L1–L4、M3、C0、Formal G1、sealed、proxy 和多个 RQ 编号，没有一张图说明这些层次的关系。对非本项目读者，最容易误读的三点是：`0` 不是一个单一机制、`ipv_error` 不是误差真值、帧级低可估计率不等于 case 级完全不可用。

**Closure required**：将一张机制图列为正式交付，而非装饰：`frozen artifact → measurement-role adapter → D0 first split → K/grid provenance → L1/L2 label or L3 pending/L4 unknown → ESTIMABLE/WEAK/NOT_ESTIMABLE → frame portrait → case aggregation → downstream qualification`。数据图至少应有：(a) source×feasibility-layer 的 count/% 堆叠图；(b) 带阈值与 n 的 K_eff ECDF/密度；(c) 每 case 可估计比例分布而非只给均值；(d) source×geometry×window 热图并标 n；(e) 六个下游 RQ 的 evidence-to-decision matrix。所有百分比必须同时展示 numerator/denominator 和 unknown/pending。

#### m-R3-01 — D0 的代码文档与实际初始化相互矛盾（MINOR）

计划正确指出 warm-up 是 `0/1` 占位（计划 `:34-38`）；但现行函数 docstring 仍说早期行填 `np.nan`（`src/sociality_estimation/core/ipv_estimation.py:210-215`），实际却初始化为 zeros/ones（同文件 `:247-252`）。虽然历史工件可直接证明 D0，执行文档若不钉死 producer semantics，后续实现者可能按 docstring 误判。应修正文档或在 inventory 中声明以 frozen artifact + producer implementation 为权威。

#### m-R3-02 — 立项扫描脚本缺少最小 schema guard（MINOR）

`RQ015_portrait_scan_v1.sh` 的注释用法名称与实际文件名不同，并以固定列号 `$22/$26/$27/$33/$34` 解析（脚本 `:2-10`）；没有验证 header、非有限值或零分母（`:12-19`）。它不是正式执行器，因此不构成独立 blocker；但作为计划引用的复现证据，应至少做 exact header/hash guard，并把结构化结果与 denominator 写入 JSON/CSV receipt。

### Technical failings that need to be addressed before the case is established

1. 冻结逐产物 authoritative inventory、measurement-role crosswalk 与 producer/version/K/grid provenance。
2. 冻结 long-form ledger schema、唯一键、join/cardinality、dedup 和逐 artifact 行守恒。
3. 绑定已有 RQ007 split bytes，而不是重新发现 authority；对 sealed exclusion 给出 machine proof。
4. 冻结本地 operation ID、authority、exact command、环境、output root、validate-only 和一次性 receipt。
5. 把自然语言验收转成 validator 可判的 PASS/FAIL JSON：hash/schema/key/range/conservation/sealed-overlap/expected-output checks。
6. 补齐 frame/case 聚合与探索性预测子的统计规格。
7. 定义 A 的有界终态及 B 的 append-only enrichment，不再使用“固有识别极限”。
8. 增加一张机制图与一组 denominator-aware 数据图，使非专业读者能区分测量、状态、聚合和下游决策四层。

### Assessment against Nature-style criteria

- **Originality**：从给定材料只能确认这是对现有 IPV 证据链的内部测量审计；没有 prior-work positioning，无法判断方法学原创性。`Not assessable from provided material`。
- **Scientific importance / significance**：对当前项目的内部有效性可能很重要，因为计划触及六个下游 RQ（计划 `:121-129`）。但 A 明确不重估既有研究，因此它能直接建立的是数据健康与资格边界，而非更广泛的行为学结论。是否具有 outstanding/broad scientific importance 尚未由结果建立。
- **Interdisciplinary readership**：潜在读者面覆盖自动驾驶、测量学、不确定性量化和计算行为科学；一般化的核心是“把未测量从中性数值中分离”。目前内部缩写和 artifact-specific 细节遮蔽了这一核心，需要上述机制图和非项目化术语。
- **Technical soundness**：`K_eff` 变换、D0 先分流、proxy 边界、sealed 排除意图和选择偏倚警告是合理基础；但输入、单位、key、split binding、异常处理、统计规格和执行授权未闭合，作者的可复现性主张当前未建立。
- **Readability for nonspecialists**：研究问题首段清楚，后续却缺少术语表、机制图、统一分析单位和从 measurement 到 downstream decision 的叙事桥。尤其“每一行”与“每个 agent-frame”、RQ009 target/current/self-anchor 三种 error 未区分，会误导专业读者，更会阻断非专业读者。

### Recommendation posture

**`BLOCKED` / currently not established from the provided execution contract.**  
关闭四个 blocker 并对五个 major 给出冻结规格后，方向上可支持重新复审。该判断是 Reviewer 3 对技术与可读性就绪度的意见，不是期刊编辑决定，也不等于否定 RQ015A 的研究价值。

## Risk / unsupported claims

- “`ipv_error` 已存在于全部主要产物”不被当前覆盖范围支持；计划自身的 L3/L4 已承认例外（计划 `:23-32`, `:80-90`）。
- “无需重算”只能对 L1/L2 成立；L3 明确依赖 RQ015B，L4 无法恢复。应改成按 artifact 分层的有界陈述。
- “固有识别极限”不受 A 的历史 proxy/预测子分析支持；没有网格/模型敏感性时只能讨论当前配置下的平坦或相关因素。
- 剔除 RQ007 sealed 后的任何帧级/case 级比例、source/geometry/window 差异和 predictor importance 均尚未执行，不能由全语料先验数字替代。
- “本地 CPU 分钟级”没有绑定输入总量、实现、硬件或基准；2.1 GB sigma01 加多组 Parquet/WOD provenance 的全 inventory 扫描成本尚不可从计划验证。
- C0 的三类资格结论在没有 deterministic decision rule 和 consumer artifact binding 前不可独立复算。

## Read-only verification record

- Frozen plan SHA-256: `cd352390d816c12c77942211ad73479a5d3b71c43c8c801150ab3e25cfa9fea8` — match.
- Split bundle manifest: `5/5 OK`.
- RQ007 split authority: `38,228` rows, duplicate case ID `0`, counts `19,258 / 7,628 / 11,342`; assignment SHA-256 `90d8bb91e68f9b5e0596cf1ae915eb22b01a5c4ccffbad00c0b446efa46d537d`.
- sigma01 primary CSV exists, size约 `2.1 GB`, SHA-256 `a60404fb1cad14a2eb49a9a5e6d7dee6a4038234d18e1567733c023e42ab2df6`; the same hash is recorded by RQ009 input provenance (`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/00_meta/input_manifest.json:4-13`).
- RQ009 M3 test prediction metadata: `1,270,566` anchors / `3,811,698` rows; read-only schema contains no `ipv_error`, confirming that it requires a key-bound join rather than direct L1 labelling.
- OnSite read-only schema check found 70,317 time-series rows with four distinct IPV/error roles (ego/counterpart × hw4/hw10) and 67,861 M3 anchors, reinforcing the need for `measurement_role × window` in the ledger key.

## Uniform post-review evidence challenge

### Challenge protocol and preservation rule

- 本节是统一后置证据质询的独立追加判断；未读取 Reviewer 1/2 文件，也未改动本报告前文。
- 前文原始判定与计数继续作为首次复审记录：**`BLOCKED`; 4 blocker / 5 major / 2 minor**。
- 后置证据揭示两个此前未单列的中心 blocker；M3/OnSite 证据则实质强化既有 input/schema finding，但为避免重复计数，不另增 finding ID。

### Evidence A + B — RQ015A 把 concentration-only state 错命名为 RQ007 “estimability”（NEW BLOCKER）

**New finding `B-R3-05 — construct substitution`。** RQ007 的 binding contract 明确区分五个概念：当前 IPV、集中度指数 `c_i(t)`、交互机会 `o(t)`、可估计性 `g_i(t)`、行为动力学（`reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_process/00_meta/binding_execution_contract.md:53-61`）。其中 `g_i(t)=1` 必须同时满足 sustained low concentration、warm-up、active opportunity、mechanical controls 和 case health（同文件 `:71-79`）；concentration-index-only output 必须作为较弱 diagnostic 单列，且不得命名或解释为 estimability（`:81`）。

RQ015A 却用 `K_eff`/单帧 `ipv_error` 单指标把尝试后的行分为 `ESTIMABLE / WEAK / NOT_ESTIMABLE`（计划 `:55-69`），并据此回答“到底有没有测出 IPV”（计划 `:18-21`）。这不是局部命名瑕疵，而是中心构念替换：权重集中只说明当前候选网格下的相对 likelihood concentration；它本身既不证明场景存在揭示社会性行为的机会，也不证明 mechanical controls 与 case health 已通过。反过来，diffuse weights 也不能单独证明“没有测出”。因此当前研究问题、状态标签、画像标题和 C0 资格链都超出提供的证据。

**最小关闭条件只能二选一并冻结：**

1. **推荐的窄化路线**：将 RQ015A 政名为“IPV estimation-attempt and candidate-weight concentration labelling”；研究问题改成“是否尝试估计、候选权重有多集中”；状态改为 `NOT_ATTEMPTED / CONCENTRATED / INTERMEDIATE / DIFFUSE`，全篇禁止把后三者称为 `ESTIMABLE/NOT_ESTIMABLE` 或“测出/没测出”。C0 只能审计 concentration-state sensitivity。
2. **保留 estimability 路线**：逐行实现并绑定 RQ007 的完整 `g_i(t)` conjunction，包括 frozen sustained rule/threshold、time-valid opportunity、mechanical-control evidence 和 case-health gate；不能提供这些字段的 artifact 必须标 `ESTIMABILITY_NOT_ASSESSABLE`。只靠 `K_eff` 的跨数据集标签不得升级为 estimability。

### Evidence C — cutoff provenance 与 sealed 声明自相矛盾（NEW BLOCKER）

**New finding `B-R3-06 — sealed-informed cutoff provenance unresolved`。** 计划先公开含 RQ007 sealed 的全语料分布，包含 `ipv_error≥0.61` 比例（计划 `:42-53`）；紧接着冻结 `K_eff≥0.93K`，并明确 K=7 时就是 `ipv_error≥0.61`（`:55-64`）。计划没有提供任何早于该全语料扫描的 timestamped/hashed cutoff 决策，因而无法证明 `0.93K`/`0.61` 未受 sealed 分布影响。与此同时，它又要求 split 定位前不得冻结任何阈值（`:74-78`），验收声称 sealed 全程未参与任何阈值（`:140-142`）。RQ007 的原始 contract 还要求 concentration threshold 只能在 development/guard 上选择（`binding_execution_contract.md:73-76`）。当前四组陈述无法同时成立。

这项缺陷不能通过最终画像时删除 sealed 行追溯修复：若 cutoff 是在观察全语料分布后确定，信息已经进入规则。它也进一步削弱 “直接采用 RQ007 已冻结判据” 的说法，因为 RQ007 冻结的是另一套 `tau + sustained + conjunction` 构念，而不是 `K_eff≤4 / ≥0.93K`。

**最小关闭条件：**

- 提供可验证、早于全语料画像的 cutoff authority（exact bytes/hash/timestamp/rationale），证明 `4` 与 `0.93K` 是预先指定且未看 sealed；否则不得声称 sealed 未参与阈值。
- 若不存在该证据，将 cutoffs 明确降级为 **post-hoc policy bins**，登记 RQ007 sealed 对本次 cutoff 设计已发生 aggregate-level contamination；不得把 sealed 用作 RQ015A 的 untouched/confirmatory split，也不得做阈值有效性升级。
- 在后一情况下，所有统计仍应剔除 sealed；development 用于探索，guard 单独报告稳定性，但必须承认这不能恢复一个已经被查看过 aggregate distribution 的 sealed confirmation claim。
- machine receipt 必须同时记录 cutoff authority hash、首次可见时间、使用 split、sealed-access disclosure 和最终 claim class。

### Evidence D — artifact multiplicity and estimator-local D0 clock（CONFIRMS/UPGRADES EXISTING FINDINGS; NO NEW COUNT）

1. **M3 predictions = 3 alpha rows/anchor**：该证据已由前文 `B-R3-01/B-R3-02` 直接捕获；它确认 prediction row 不能作为 measurement row，必须按 `(case_key, anchor_frame_index, perspective)` 连接一次 target measurement，再把 alpha 作为 consumer dimension。此项不新增计数。
2. **OnSite D0 depends on estimator-local sequence position**：这实质升级 `B-R3-01/B-R3-02` 的 closure。计划把 D0 写成 `frame_index < MIN_OBSERVATION`（计划 `:58-65`），但跨 artifact 的原生 `frame_index` 不一定等于一次 estimator call 内的 sequence position；OnSite 又同时保存 hw4/hw10 与 ego/counterpart 多套结果。用全局/native frame index 会把已尝试与 warm-up 错分。ledger 必须为每个 `measurement_role × window × estimator invocation` 绑定 `local_sequence_position`、`min_observation` 和 producer semantics；若历史 artifact 无法恢复 local clock，则 `attempt_status=UNKNOWN`，不得推断 D0。此项属于既有 input/schema blocker 的具体失败模式，不重复增加 blocker 数。

### Post-challenge final verdict and counts

- **Original record（保留）**：`BLOCKED`; **4 blocker / 5 major / 2 minor**。
- **After uniform evidence challenge**：`BLOCKED`; **6 blocker / 5 major / 2 minor**。
- 新增 blocker：`B-R3-05 construct substitution`、`B-R3-06 sealed-informed cutoff provenance unresolved`。
- Evidence D 不另计数，但其 estimator-local clock 要求成为 `B-R3-01/B-R3-02` 的必要关闭条件。

### Minimum closure set after challenge

RQ015A 下一版至少须同时完成：

1. 在 concentration-only 窄化路线与完整 RQ007 `g_i(t)` 路线中明确二选一；字段、标签、标题、报告语言和 C0 结论全部一致。
2. 解决 cutoff authority 与 sealed disclosure；没有 pre-scan authority 就承认 post-hoc/contaminated，取消 untouched-confirmation 语言。
3. 冻结 authoritative input inventory、长表 schema/唯一键与 measurement-role crosswalk；M3 每 anchor 只连接一次测量，alpha 不进入测量分母。
4. D0 只由 estimator-local sequence clock + producer-bound `min_observation` 判定；不可恢复时 fail closed 为 unknown。
5. 完成前文 `B-R3-01` 至 `B-R3-04` 其余 machine-readable inventory/split/local-authority/validator 条件，以及 M-R3-01 至 M-R3-05 的统计、A/B 边界、C0 与可读性规格。
