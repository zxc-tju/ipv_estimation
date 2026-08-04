# RQ014 研究结果独立审稿意见

Review date: 2026-07-25  
Reviewer mode: independent, read-only, rating-blind  
Review target: `reports/studies/RQ014_wod_e2e_rating_recovery/03_results/RQ014_final_report_20260725.html`

## Review setup

- **Input scope**：最终 HTML 报告、冻结的 recovery-lane-v3 合同、R3/G2 实现、D4 授权记录、现有测试、R10L rating-free probe 源码，以及 RQ014 知识库/索引状态。
- **Assessment boundary**：本 review 未读取任何原始 rating/preference 明细，未重新计算评分关联，也未接触 HPC 评分源。最终报告中的聚合统计按其持久化表述接受，但检查其合同一致性、解释边界和可复核性。
- **Shared claim summary**：R04N 臂出现一个通过冻结 `recovery_compatible` 门的次要 NMD/RWS 行；预登记 primary NEX 无通过行；R10L 因已确认实现缺陷未被有效评估。
- **Visible evidence base**：最终报告给出 `1/960`、`r=-0.384`、`n=42`、fold/LOO/LOCO、NEX `0` 通过和 R10L probe 汇总（最终报告 `:49-55, 91-135, 137-188`）。v3 合同明确这是同数据集、已知结果驱动的 specification recovery，而不是独立验证或假设检验（`reports/plans/RQ014_recovery_lane_v3.json:6-21`）。
- **Missing materials affecting confidence**：
  1. 当前 R3 产物合同和测试只列出 ledger、terminal digest、attrition、rank index、common-support sensitivity 和两个 kernel manifests，没有 `selected_recovery_recipe.json`（`scripts/rq014/run_managed_g3.py:1282-1291, 1599-1633`；`tests/test_rq014_g3r_managed.py:392-400`）。
  2. 未见 G4R clean independent replay 的 PASS 产物；D4 明确 R4 仍未授权（`reports/plans/RQ014_PI_decision_D4_G3R_authorize_20260723.md:15-18, 26-31`）。
  3. R10L probe 的本地副本自述：除两个桶外，多数 observed counts 只来自任务 brief，未找到仓库内 CODE/SPEC 载体（`.codex-fleet/rq014-execution-v1p6/board/staging/probe_r10l_recovery.py:62-74`）；该目录又被 `.gitignore:23` 忽略。
  4. 最终 HTML 被 `.gitignore:62` 忽略；本 review 已同步知识库 README 和 `STUDIES.md` 的状态，但底层结果包仍未成为 tracked evidence。

## Overall evidence verdict

| Layer | Verdict | Meaning |
|---|---|---|
| R3 对冻结 bank、join、association 和 recovery gate 的机械执行 | `FAITHFUL` | 代码实现与 v3 的 RWS/PSP/PPR、fold/LOO/LOCO 和 gate 规则一致；现有聚焦测试通过。 |
| R10L as-run 科学可解释性 | `DEFECT` | 上游将任意全时间线 gap 升级为整采样臂终结，宽于冻结的 per-window 语义。 |
| R04N 唯一兼容行的科学结论 | `UNCERTAIN / BOUNDED` | 是同数据集、已知方向驱动搜索中的次要 readout、场景内候选排序信号；未做 clean replay，也不是独立数据验证。 |
| “M3 已实现 WOD 外部效度迁移” | `NOT ESTABLISHED` | primary NEX 为 null，M3 在 WOD 被合同明示为 out-of-support extrapolation，且无新数据确认。 |
| RQ014 当前合同终态 | `PENDING_REPLAY` | 至多对应 v3 的 `RECOVERY_RECIPE_FROZEN_PENDING_REPLAY`；实际 selected-recipe artifact 也尚未在当前 R3 产物集中闭环。 |
| Manuscript claim acceptance | `NOT ACCEPTED` | 不应创建接受性 `decision.md`，也不应向论文仓库传递 confirmed-transfer 表述。 |

## Reviewer 1 — technical soundness / technical failings emphasis

### Overall assessment

盲态治理、单次 join、哈希链和失败关闭设计很强；R3 对既有 bank 的消费也在代码层可追踪。但“R3 机械 PASS”不能覆盖两个不同层级的问题：上游 R10L bank 生成存在方法缺陷，且兼容行出现后要求的 recipe freeze 与 clean replay 没有完成。因此，当前材料支持一个可审计的 **screen result**，不支持已经完成的 historical recovery。

### Who would be interested in the results, and why

做自动驾驶行为评价、human-aligned planning metric、跨数据集模型迁移，以及可审计科研计算治理的读者会关心。科学上关心的是候选轨迹的 IPV-envelope deviation 是否与人类偏好排序一致；方法上关心的是如何在历史结果已知、配置遗失时限制结果依赖的搜索。

### Major strengths

1. v3 明确冻结单位、方向和 claim boundary：观测单位是 candidate，依赖簇是三候选 segment；同数据集 recovery、clean replay、独立 replication 和因果主张被严格分开（`reports/plans/RQ014_recovery_lane_v3.json:15-27`）。
2. R3 association 和 stability 代码与冻结门一致。`_recovery_compatible` 实现 `n>=40`、`r<=-0.30`、fold、LOO、cluster 数和 LOCO 比例门（`scripts/rq014/run_managed_g3.py:408-430`），与合同要求一致（`reports/plans/RQ014_recovery_lane_v3.json:666-681`）。
3. R10L 缺陷被透明披露，而没有把 480 个空行解释成科学 null（最终报告 `:137-168`）。

### Major concerns

1. **兼容行后的合同链没有闭环。** v3 规定兼容行出现时生成唯一 `selected_recovery_recipe.json`，clean replay 通过后才能达到 `HISTORICAL_RESULT_RECOVERED_ON_SAME_DATASET`（`reports/plans/RQ014_recovery_lane_v3.json:731-802, 824-843`）。当前 R3 实现和测试的产物集合不包含 selected recipe（`scripts/rq014/run_managed_g3.py:1282-1291, 1599-1633`；`tests/test_rq014_g3r_managed.py:392-400`），D4 还明确禁止 R4（`reports/plans/RQ014_PI_decision_D4_G3R_authorize_20260723.md:15-18, 26-31`）。这不是措辞小问题，而是结论等级的硬边界。
2. **R10L 为确认的上游方法缺陷。** `run_managed_g2.py` 在全分支重采样阶段捕获 `SourceGapError` 后写入 terminal sampling（`scripts/rq014/run_managed_g2.py:189-216`）；随后 anchor-domain 在逐 window 检查前把整分支全部 terminalize（`scripts/rq014/build_wod_scene_anchor_domain.py:251-275`）。冻结 v3 则要求先切 exact window、再在窗内派生状态（`reports/plans/RQ014_recovery_lane_v3.json:91-108`）。因此 R10L as-run 只能判 `DEFECT`。
3. **R10L “修复亦无用”尚缺 durable evidence closure。** 最终报告给出 2,740 recovered、最大每格 `n=29`（最终报告 `:150-168`），但本地 probe 自己说明大多数 run-specific counts 没有仓库 CODE/SPEC 来源，只有两个桶是 fail-closed assertions（probe `:62-74`）。在 probe JSON、输入 hashes 和完整运行收据进入可跟踪 evidence package 前，这条结论只能是 `UNCERTAIN` 的执行决策依据，不能作为强科学结论。
4. **最终结果包未进入 durable source of truth。** 最终 HTML 和 probe 都被忽略；本 review 仅同步了知识层状态，不能代替底层 aggregate artifacts、hashes 和 receipts。这使审稿人仍无法仅从 tracked repository 完成 read-back replication。

### Technical failings that need to be addressed before the case is established

- 机械冻结 rank-1 recipe，并生成合同要求的完整 selected-recipe binding artifact。
- 由未参与 G2/G3 的 fresh implementation 完成 G4R clean replay；原 screen 代码、非 selected rows 和 screen caches 不得复用（`reports/plans/RQ014_recovery_lane_v3.json:763-802`）。
- 将 R3 aggregate evidence、R10L probe JSON、输入/输出 hash 和审计收据纳入 tracked report package。
- 在任何 scientific decision 中把 R3 screen fidelity、R10L upstream defect 和 G4R reproducibility 分开报告。

### Assessment against Nature-style criteria

- **Originality**：结果本身的新颖性有限；高价值部分更可能是失落结果 recovery 的强治理框架。
- **Scientific importance**：当前是局部、同数据集、次要 readout 信号，尚不构成高影响外部验证。
- **Interdisciplinary readership**：human-aligned AV metrics 有跨领域潜力，但需把候选排序和场景质量预测区分开。
- **Technical soundness**：R3 screen 机械 sound；总体 claim chain 因 R10L defect 和缺失 clean replay 未闭环。
- **Readability for nonspecialists**：结论页可读，但 `R04N/R10L/NEX/NMD/RWS` 密度较高，且没有用一张简单图解释“screen → freeze → replay → confirmation”。

### Recommendation posture

`Currently not established as recovered or externally validated; technically promising after contract closure.`

## Reviewer 2 — statistical evidence / reproducibility emphasis

### Overall assessment

当前统计结果最适合定义为 **bounded historical recovery candidate**。唯一通过行为 `RR3-R04N-CH-W25-H20-NMD_MEAN-RWS`，报告值为 `r=-0.384, n=42`，但它来自 960 行已知方向驱动搜索中的 secondary predictor；primary NEX `0` 行通过（最终报告 `:106-135`）。fold、LOO、LOCO 显示的是 selected cell 的内部删样稳定性，不是独立复制。

### Who would be interested in the results, and why

关心小样本偏好评价、分层稳定性检查、迁移 OOD 风险和 specification-search bias 的统计/ML 读者会关心，尤其是这种“三候选/场景内排序 + 跨场景平均”的评价设计。

### Major strengths

1. association 方向没有翻转：NEX/NMD/AMD 越大表示偏离越大，因此负相关是预期方向；RWS、PSP、PPR 返回普通 signed association（`reports/plans/RQ014_recovery_lane_v3.json:203-218, 456-490`）。
2. 唯一兼容行通过预先机器化的 n/effect/fold/LOO/LOCO 组合门，而不是只挑一个最负的点估计（最终报告 `:106-121`）。
3. 报告主动承认 primary NEX null、样本衰减和不做 confirmatory multiplicity claim（最终报告 `:123-135, 184-193`）。

### Major concerns

1. **这是 known-result specification recovery，不是 hypothesis test。** v3 目标就是从已知负方向中找一个复现 recipe，p 值和 prospective power 都不控制 recovery（`reports/plans/RQ014_recovery_lane_v3.json:6-21, 666-678`）。因此 `1/960` secondary pass 的选择后 CI 不能作为普通、未选择的 95% coverage 来解释。
2. **跨 cell 的负中心不是 160 次独立复现。** cells 共享同一 479-scene universe、相近 temporal/readout transforms，且允许 support 在 cell 间变化；合同称其为 operational recovery ordering，而非 equal-cohort causal comparison（`reports/plans/RQ014_recovery_lane_v3.json:263-275, 684-714`）。最终报告的 RWS `19/20` 负号和三类关联负中心（`:93-103, 178-181`）只能作为 descriptive directional pattern。
3. **fold/LOO/LOCO 不是独立验证。** 合同明确 fold 不是 train/test（`reports/plans/RQ014_recovery_lane_v3.json:639-650`）；LOO/LOCO 都从同一 selected support 做删样重算。只有 4 个 eligible clusters 的 4/4 负号说明该小支持内没有单一 cluster 翻转结果，但不能建立跨域泛化（最终报告 `:112-120`）。
4. **唯一正面结果是 within-scene candidate ordering。** RWS 对每个场景的三候选 rating/deviation 做 midrank correlation，再跨场景等权平均（`reports/plans/RQ014_recovery_lane_v3.json:456-476`）。这可支持“在该 recipe 下，人类较偏好的候选往往 deviation 更低”，不能单独支持“模型预测 479 个场景的绝对人类评分”。PSP/PPR 提供 pooled descriptive patterns，但无 compatible pass（最终报告 `:97-100, 109-121`）。
5. **M3 对 WOD 是合同允许的 OOD extrapolation。** v3 明示历史 pilot `0/228 in-support`，并把 support/abstain 只当诊断而不屏蔽行（`reports/plans/RQ014_recovery_lane_v3.json:50-53`）。这不使计算失效，却显著限制 external-validity 解释。
6. **primary endpoint failed.** NEX `0` 通过而 NMD `1` 通过，不能事后把 NMD 升格为确认性 primary（最终报告 `:123-135`）。
7. **冻结 gate 缺少经验性 null calibration。** `1/960` 是观察到的通过计数，不是多重搜索校正后的错误率。若要把该候选提升到描述性 recovery 之外，需要在独立授权和预先冻结的方案下，对同一场景内 rating labels 做置换，并重新运行完整选择流程；否则不能把单个 gate pass 当作小概率偶然性已被排除。

### Technical failings that need to be addressed before the case is established

- 对 selected recipe 做 clean replay，报告点估计、support、fold/LOO/LOCO 和 gate 是否逐项复现。
- 对新数据做真正的 prospective confirmation；同一 rated479 上继续切分或加统计量不能消除 full-data selection。
- 若要解释 gate pass 的偶然性，另行冻结并授权完整 selection-pipeline 的 within-scene permutation calibration；该分析不得事后改变 gate。
- 若讨论 160-cell sign pattern，应提供明确的 joint/hierarchical descriptive summary，而不是把 cell tally 当 replication count。
- 保留 NEX 为 primary null；NMD 只能标为 secondary recovery candidate。

### Assessment against Nature-style criteria

- **Originality**：严格 recovery protocol 有方法学原创性；单个 secondary cell 的统计发现不强。
- **Scientific importance**：效应可能有意义，但 `n=42`、4 clusters、选择后 secondary endpoint 和 OOD scorer 共同限制重要性。
- **Interdisciplinary readership**：偏好建模与安全评价读者会关心；更广泛读者需要新的、未参与 recovery 的评分数据。
- **Technical soundness**：统计计算可追踪；确认性解释不成立。
- **Readability for nonspecialists**：应直说“42 个有信息场景内比较各自 3 个候选”，不要写成对 479 个场景绝对评分的普遍预测。

### Recommendation posture

`Bounded descriptive recovery candidate; independent confirmation and clean replay are required before a strong scientific claim.`

## Reviewer 3 — originality / significance / interdisciplinary readability emphasis

### Overall assessment

报告的诚实边界优于结果强度：它明确写出不能声称“迁移已确认”，也不能把 R10L 空结果当 null（最终报告 `:176-193`）。但标题和开篇研究问题仍容易让非专业读者把唯一 RWS 结果理解为广义 scene-level prediction 或 external validation。当前最有价值的叙事是“在严格盲态治理下得到一个有限的历史 recovery candidate，并发现一个会清空半网格的方法缺陷”，而不是“M3 已跨数据集验证”。

### Who would be interested in the results, and why

- 自动驾驶规划/评价研究者：candidate preference 与 social-behavior deviation 的关联。
- 人机交互与认知建模研究者：模型量化是否贴近人类候选偏好。
- 科研软件与可复现性研究者：已知结果 recovery 中如何限制评分泄漏、方法漂移和 partial leaderboard exposure。

### Major strengths

1. “可以说/不可以说”边界明确（最终报告 `:176-193`）。
2. R10L defect、可恢复量估计和不重跑决定被完整披露，没有沉没负面技术证据（最终报告 `:137-173`）。
3. rating-blind governance、一次性 join 和 aggregate-only disclosure 具有独立方法学价值（最终报告 `:67-88`）。

### Major concerns

1. **科学叙事比证据宽。** 报告标题与开篇问“M3 是否预测 479 个场景的人类评分”（最终报告 `:43-46, 67-75`），而唯一 load-bearing positive result 是一个 R04N/NMD/RWS 的 within-scene candidate-ordering result。
2. **广泛意义尚未建立。** same-dataset known-result recovery、secondary endpoint、primary null、M3 OOD extrapolation 和缺失 G4R 共同使结果难以支撑跨数据集 external validation。
3. **可复核报告包不足。** reader-facing HTML 和 probe 被忽略；虽然本 review 已同步 tracked knowledge index，但底层 aggregate evidence 仍不可从仓库独立 read back。对外部审稿人而言，这降低了现有结果的可达性和可信度。
4. **缩写和执行治理压过科学问题。** R04N/R10L、CH/HF/LF/TP/TF、NEX/NMD/AMD、RWS/PSP/PPR 在短段内集中出现（最终报告 `:71-75`），缺少一张面向非专业读者的 estimand 示意图。

### Technical failings that need to be addressed before the case is established

- 将标题和摘要改为 “historical specification recovery”，并用一句话说明不是前瞻验证。
- 把唯一正面项写成 within-scene candidate ordering-compatible signal；PSP/PPR 仅作 pooled descriptive support。
- 对论文级 claim 要求 clean replay 和新数据确认；否则不进入 accepted claim ledger。
- 发布 tracked Markdown report、evidence manifest 和完整 review/synthesis/decision 路由。

### Assessment against Nature-style criteria

- **Originality**：治理/恢复框架可能新颖；科学信号本身不足以单独构成高新颖性。
- **Scientific importance**：尚未显示 immediate and far-reaching implication。
- **Interdisciplinary readership**：主题有潜力，当前证据仍主要吸引本领域方法读者。
- **Technical soundness**：screen 层扎实，claim 层不完整。
- **Readability for nonspecialists**：需要减缩写、加示意图、收窄标题。

### Recommendation posture

`Promising as a transparent recovery case study; broad-interest scientific case remains underdeveloped.`

## Cross-review synthesis

### Consensus strengths

1. rating-blind、single-join、aggregate-only、fail-closed 的治理链是本项目最扎实的部分。
2. R3 对冻结 ledger/gate 的机械执行是可追踪的。
3. 最终报告诚实披露 primary NEX null、R10L defect 和样本衰减，没有把空臂包装为阴性科学结果。
4. 唯一 R04N/NMD/RWS 行在其小支持内具有 fold/LOO/LOCO 符号稳定性。

### Consensus technical risks

1. rank-1 recipe freeze 和 G4R clean replay 未完成，故不能达到 `HISTORICAL_RESULT_RECOVERED_ON_SAME_DATASET`。
2. R10L 是真实方法缺陷；其 probe ceiling 目前缺少 tracked、可 read-back 的完整证据包。
3. 唯一兼容行为 secondary NMD；primary NEX 为 null。
4. 960-cell known-result search、cell 间依赖和选择后 CI 不支持确认性统计解释。
5. 唯一正面结果是 within-scene candidate ordering-compatible，不是 479-scene absolute rating prediction。
6. M3 在 WOD 上是明确 out-of-support extrapolation，不能据此宣称强 external validity。
7. 当前没有完整 selection-pipeline 的经验性 null calibration；`1/960` 本身不是 multiplicity-adjusted significance。
8. G0 历史指纹不可得，因此未来即使 clean replay 通过，也只能证明冻结配置被计算复现，不能证明它与记忆中的历史结果完全同一。

### Where emphasis differs across reviewers

- Reviewer 1 最重视合同闭环：selected-recipe artifact、clean replay 和 durable evidence package。
- Reviewer 2 最重视选择偏差、cell 依赖、primary/secondary 层级和 OOD/scarce-support 解释。
- Reviewer 3 最重视标题、跨学科意义和读者是否会把 recovery candidate 误读为 external validation。

### Broad-interest / significance readout

当前结果对 AV human-alignment 方法研究和科研治理有价值，但尚未达到“跨领域读者会因一个已建立的普遍科学结论而关心”的程度。可发表价值更多来自透明的 recovery case study，而不是单个 secondary cell 本身。

### Most important issues to resolve before a strong case is established

1. 冻结唯一 recipe，并完成 independent clean replay。
2. 将 R3 和 R10L probe 证据纳入 tracked report package。
3. 保留 primary NEX null，不把 NMD 事后升格。
4. 将唯一正面结论收窄为 within-scene candidate ordering-compatible。
5. 未来使用未参与 recovery 的新评分数据做 prospective confirmation。
6. 若需要评估 screen gate 的偶然通过率，预先冻结并独立授权全流程的场景内置换校准。

## Risk / unsupported claims

| Claim | Review verdict | Supported replacement |
|---|---|---|
| “M3 向 WOD 的迁移已确认” | Unsupported | “R04N 出现一个同数据集、次要 NMD/RWS 的 bounded recovery candidate。” |
| “历史强负相关已恢复” | Premature | “一个兼容行已在 screen 中识别；clean replay 尚未完成。” |
| “模型预测 479 个场景的人类评分” | Overbroad | “在 42 个 informative scenes 内，三候选的 deviation ordering 与人类偏好 ordering 出现负向一致性。” |
| “160 cells 的负中心构成重复验证” | Unsupported | “相关 cells 上出现 descriptive directional consistency。” |
| “R10L 证明无迁移” | Unsupported | “R10L as-run 为 DEFECT，不提供方向性科学结论。” |
| “修复 R10L 必然无用” | Not fully assessable from tracked evidence | “rating-free probe 报告几何支持上限低于门槛；需纳入完整 probe receipt 后再冻结。” |
| “结果具有强 external validity” | Unsupported | “结果来自合同允许但明确 out-of-support 的 WOD extrapolation。” |
| “当前成果包可独立复现” | Unsupported | “screen 代码可测试；selected recipe、clean replay 和 durable result/probe package 尚缺。” |
| “G4R 通过即可证明与记忆中的历史结果同一” | Unsupported | “G4R 只能证明冻结配置可被独立计算复现；缺失的历史指纹不能被事后重建。” |

## Final reviewer conclusion

RQ014 已完成一个治理严格、机械可审计的全数据 recovery screen，并在 R04N 中识别出一个方向正确、删样稳定的 secondary NMD/RWS 候选；这是真实但有限的结果。它**不能**被表述为 confirmed transfer、independent validation 或 historical result fully recovered。

当前最稳妥的项目状态是：

> `SCREEN_COMPLETE_WITH_ONE_SECONDARY_CANDIDATE; R10L_DEFECT; SELECTED_RECIPE_AND_CLEAN_REPLAY_PENDING; NO_ACCEPTED_MANUSCRIPT_CLAIM`

在 selected-recipe artifact、G4R clean replay 和 durable evidence package 完成前，不建议创建接受性 `decision.md`，也不建议把该结果用于论文中的确认性主张。即使未来 clean replay 通过，它也只闭合 v3 冻结配置的计算复现；对新数据的 prospective confirmation，以及与不可得历史指纹的同一性，仍是不同命题。
