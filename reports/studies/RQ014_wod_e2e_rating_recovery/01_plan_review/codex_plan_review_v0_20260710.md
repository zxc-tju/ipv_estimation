# RQ014 Plan v0 — Codex independent review

Review date: 2026-07-10  
Reviewed plan: `reports/plans/RQ014_plan_v0_wod_e2e_rating_ipv_deviation_recovery_20260710.md`  
Plan SHA-256: `4a7d49504beeb50d09aae24a9c931d940dc5fd6e49d5f9ca7584bbbccdf55baf`  
Forensics report: `reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/forensics_report.md`  
Forensics SHA-256: `e6f1ce6e136cdf7355e18e38b5853b934658d462f72e61acb3bddf282ca16c88`  
Review type: independent statistical, execution/HPC, governance, and adversarial review  
Nature-writing use: claim–evidence–boundary discipline for the review narrative  

## Verdict

**BLOCKED_PENDING_MAJOR_REVISION.**

RQ014 的研究目的合理，且 v0 中“先取证、再冻结 registry、复现 RQ010B anchors、最后才加入评分、保留全部失败配置、red team、clean-room 重实现、禁止静默改写 RQ010B”等治理思想值得保留。但是，v0 当前还不是一个估计对象唯一、配置空间可枚举、确认检验自洽、功效可接受的可执行计划。

本 review **不批准启动 Phase R/H/S/C**，也不批准提交全网格 HPC 作业。当前允许继续的工作仅限于：

1. 完成或正式关闭 Phase F 剩余高价值取证面；
2. 编写取代 v0 的 v1/amendment；
3. 做 rating-blind 输入清单、eligibility 和资源基准设计；
4. v1 完成后重新进行一次独立计划复审。

## Executive assessment

计划的核心风险不是“算力太大”，而是把三类不同任务放进了同一个搜索和显著性 family：

- **科学有效性问题**：同一场景中，评分更高的候选是否具有更小的预先定义 IPV-envelope deviation；
- **历史结果重建问题**：旧结果是否来自 pooled、scene-level 或不同窗口定义；
- **缺陷取证问题**：dt 错配、自参考、CV 外推或 OOD envelope 是否能制造类似负相关。

这三类问题具有不同的分析单位、统计量、证据等级与结论边界。若继续混合，即使某个 cell 显著，也无法判断它是候选级科学关系、场景间混杂、配置选择过拟合，还是已知缺陷的产物。

## Plan strengths to preserve

| Strength | Assessment |
|---|---|
| 如实保留 RQ010B bounded null | PASS。没有把记忆中的阳性结果当作已知真理。 |
| 三种终局均可记录 | Directionally good，但 verdict 名称和证据门槛需重写。 |
| Forensics first | Good principle；当前 Phase F 尚未真正闭环。 |
| Registry freeze + hashes | Good；但 v0 的 registry 仍不可枚举。 |
| G2 anchor parity | Essential；需要精确 adapter lineage、fixtures 和容差。 |
| Ratings 最后 join | Good；confirmation 还需文件级访问控制。 |
| Red team + clean-room implementation | Good；只能证明内部计算复现，不能称独立数据 replication。 |
| 不静默修改 RQ010B | PASS；后续最多 scoped addendum。 |
| HPC `/ZXC` 路径与 `zxc-` 前缀 | PASS。 |

## Blocking findings

| ID | Finding | Evidence | Minimum resolution |
|---|---|---|---|
| B1 | Phase F 未闭环，却给出无限定 `NOT_FOUND` | Forensics report lines 27–57 | 先检查 6-29 pilot 中间结果；其余面完成或标记 `INACCESSIBLE` |
| B2 | valid scientific estimand 不唯一 | Plan lines 106–122 | 冻结 candidate-level, scene-grouped, within-scene estimand；其他单位移出 valid family |
| B3 | 配置空间与 `<=~96` 不相容 | Plan lines 95–125 | 用 `config_space.yaml` 逐行枚举；valid 与 forensic 两个 registry 分开 |
| B4 | verifier/envelope 没有可编码的测量合同 | Plan lines 114–120 | 冻结 rolling window、共同时间支持、missingness、归一化、唯一主 endpoint 和 envelope 同尺度规则 |
| B5 | confirmation 与 whole-grid max-stat 自相矛盾 | Plan lines 74–84, 124–125 | 最多晋级一个 valid recipe；confirmation 只检验该 recipe 的一个主统计量 |
| B6 | 50/50 split 与 `n>=30` 不能保证确认功效 | Plan lines 127–147；RQ010B N=75/98/47 | 先做 rating-blind eligibility 和 simulation-power gate，再冻结 split 与研究上限 |
| B7 | estimator lineage/G2 anchor contract 不完整 | Plan lines 63–72 | 分开 pinned legacy estimator 与 WOD patched adapter；锁代码/adapter/env/input hashes、fixtures 与容差 |
| B8 | claim tiers 可把弱内部信号升级为 “recovered/replication” | Plan lines 17–40, 134–147 | 重写互斥 verdict；同数据只能称 internal locked confirmation；IPV-specific 必须过物理/负控制 |

### B1. Phase F is not complete under its own “forensics first” rule

`00_forensics/forensics_report.md` 把 verdict 写为 “NOT_FOUND（已扫面全部阴性）”，但同一文档又明确列出：

- 6 月 29 日四个 pilot 结果目录的中间相关数尚未检查；该项直接检验当前最高优先级 H-A；
- Mac sibling archive / paper Git pass3b 未完成；
- `WODE2E.md` 正文和 Cowork 候选会话未完成；
- 旧 Windows 与 OneDrive 历史不可达。

因此当前严谨表述应是：**未在已完成的扫描面找到旧结果**。在 Phase R 之前至少应完成 6-29 pilot 链、mac pass3b 和 `WODE2E.md`；若连接或介质不可用，记录时间、命令和 `INACCESSIBLE`。旧 Windows/OneDrive 可设定时间盒后不阻塞，但不能被写成“已扫描阴性”。

### B2. The plan mixes incompatible estimands

v0 同时注册 pooled candidates、within-scene rank、scene-mean、history-only、candidate vs driven trajectory。这些量不回答同一个问题：

- pooled candidates 忽略三候选嵌套，可能由场景难度、path type 或评分基线产生 Simpson 型关系；
- scene-mean 检验场景级难度，不检验同场候选排序；
- history-only 和 driven trajectory 对同一场景的三个 rated candidates 是共享量；
- candidate-set-internal envelope 用评价集合本身构造 norm，测到的是候选集合中心性，不是外部 human IPV envelope。

科学 valid family 必须固定为：**candidate-level unit、segment group、同场候选比较、scene-clustered inference**。计划需明确场景等权/候选等权、两候选与三候选处理、rating/deviation ties、全相同 deviation、NaN/fallback 和主统计量。pooled/scene-mean/history-only/driven/candidate-internal envelope 只能进入 forensic 或描述性分支。

### B3. The registered grid is not enumerable

v0 的 `<= ~96 expensive configs` 没有从 E1–E4 的轴定义中推导出来。独立实现审查在不同合理解释下得到 **672–1600** 个昂贵组合；简单字面乘法也可得到约 960。cheap layer 的自然解释约为 **486–648 readouts per expensive config**。即使强行截断到 96 个昂贵配置，也可能产生约 4.7–6.2 万个结果 cell。

真正的 blocker 不是这些数量必然跑不动，而是计划没有说明：

- 哪些组合合法；
- 哪些组合被删除；
- smoothing 的 window/order；
- matched/mismatched 的具体配对；
- future horizon；
- undefined/degenerate cell 是否进入 family；
- `<=96` 是何种 deterministic design，而非看结果后的裁剪。

v1 必须逐行冻结 `config_id`。建议把先前 reviewed handoff 的 16 core / 48 fixed readout 作为收窄起点，但仍需解决 future-only 2 s 与 2.5 s trailing window 的 early-window/min-observation 冲突。support readouts 不能都参与 recipe 选择。

### B4. The rolling verifier and human-envelope scale are under-specified

v0 只列出 signed/absolute exceedance、80/90/95%、fraction outside 和 max exceedance，没有冻结：

- 每时刻 IPV 对哪个物理窗口 `[t-W,t]` 计算；
- `history_window` 和 `min_observation` 如何随 4/10 Hz 变换；
- 三候选是否共享同一个 rating-blind counterpart；
- 三候选共同有限时间支持；
- 一条候选缺失时是否同步删除三条；
- counterpart extrapolation；
- 可变 horizon 的时间权重/有效时长归一化；
- 上下不对称 band 的归一化；
- 唯一 confirmatory endpoint；
- verifier-level non-degeneracy。

此外，RQ010B path-type norm 来自 **InterHub pure HV–HV**，不是 WOD 内部 human norm；M3 在 WOD 上 `<=15%` in-support。更重要的是 IPV 对 rate/window 敏感，4 Hz vs 10 Hz candidate IPV 排序相关仅约 0.29–0.31。若不同 dt/window 的 IPV 都与同一个旧 envelope 比较，相关可能只是 estimator scale 与 norm scale 不一致。

v1 必须二选一：

1. 对每个 valid timing recipe，用相同 estimator/dt/window/preprocessing 在 rating-blind human source 上重建同尺度 envelope；或
2. 只保留与现有 envelope 构建合同同尺度的 timing recipe，其余作为 sensitivity/forensic。

如采用静态 path-type band，应准确称为 **rolling WOD candidate IPV against an InterHub-derived static path-type human envelope**，不能称 WOD dynamic M3 envelope。

### B5. Confirmation and max-stat are internally inconsistent

Phase S 允许 PI 从 discovery leaderboard 主观选 `<=5` 个 cell；Phase C 说只运行晋级 cell，但又要求 confirmation max-stat 覆盖整个 registered grid，包括 defect、失败、不同统计单位和不同 subset。

whole-grid max-stat 要求每次 permutation 对每个 cell 都有可定义且同尺度的统计量。当前 family 不满足这一条件；若真的在 confirmation 运行整个 grid，又等于把全部 cell 在 confirmation 打开。

推荐的最小修复：

- discovery 用冻结的 deterministic gate/tie-breaker 晋级 **一个** valid recipe；
- confirmation 只检验该 recipe 的唯一主统计量；
- 不再声称 whole-grid confirmation max-stat；
- 若坚持多个 recipe，只对实际确认的同估计对象 recipes 用 Holm 或 synchronized maxT；
- forensic family 不进入 valid family 的 leaderboard、alpha 或 multiplicity denominator。

### B6. Power must be a gate, not an afterthought

RQ010B 已知有效场景约为 75、98；10 Hz Scheme 2 为 47。50/50 后 confirmation 约为 37、49 或 23 个场景。在忽略 ties、聚类和多重校正的乐观 Fisher-z 近似下，真实 `|rho|=0.30`、双侧 alpha=0.05 的功效约为：

| Confirmation N | Approximate power |
|---:|---:|
| 23 | 0.28 |
| 37 | 0.44 |
| 49 | 0.56 |

因此 `n>=30` 不是功效设计。v1 必须在 rating-blind eligibility 后，用真实 candidate count、ties、missingness、cluster 与完整 selection-to-confirmation 过程进行模拟，并冻结最小有意义效应、单/双侧检验、最低功效、split 和低功效终局。若门不过，RQ014 的上限只能是 historical specification recovery；confirmation 失败不能被解释为加强 RQ010B null。

### B7. The pinned estimator is not the full RQ010B WOD adapter lineage

v0 把 `/ZXC/ipv_estimation` 的 pinned `5edd2810` 与 RQ010B 已修复的 WOD adapter 当成同一套实现。独立实现审查发现：legacy lineage 本身仍采用概率域 trajectory reliability，且没有 WOD 的 timing/log-domain adapter；RQ010B fixed path 额外设置 sample dt 并替换稳定 log-domain reliability。仅声明 `HEAD=5edd2810` 不足以复现 RQ010B anchors。

G2 需要拆成明确的 anchor suite：

- legacy estimator commit；
- WOD adapter/monkeypatch SHA；
- environment lock；
- exact input row IDs/hashes；
- eligibility and join contract；
- expected deterministic values；
- point estimate/N exact tolerance；
- bootstrap/permutation seed and p-value tolerance；
- rolling-verifier golden fixtures、inside-envelope zero、越界单调性、common-support 和 no-extrapolation tests。

`full479 rho=-0.0384` 是 raw ego-IPV pooled forensic result，不是 C1/C2 envelope-deviation anchor，应与 accepted Scheme 1/2 anchors 分开。

### B8. Claim tiers and RQ010B amendment boundary are unsafe

同一 WOD-E2E 评分已经在 RQ010B 全量 join；新 hash split 只能提供程序性的 **same-dataset internal locked confirmation**，不是独立数据 replication。clean-room agent 可以证明 computational reproducibility，不能证明独立科学复现。

v0 的 T1 只要求负号和 p<0.05，T2 又明确“不要求”超越 physics baseline，却允许形成 manuscript-grade/revision 叙述。这不足以改变 RQ010B 的“IPV 不优于物理特征”结论。

建议冻结互斥 verdict：

Scientific family:

- `VALID_INTERNAL_CONFIRMATION_ASSOCIATION`
- `VALID_INTERNAL_CONFIRMATION_IPV_SPECIFIC`
- `DISCOVERY_ONLY_OVERFIT`
- `INCONCLUSIVE_LOW_POWER_OR_COVERAGE`
- `NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY`

Forensic family:

- `HISTORICAL_SIGNATURE_RECONSTRUCTED`：必须匹配旧 rho/N/figure/file fingerprint；
- `ARTIFACT_MECHANISM_SUPPORTED`：同一 common support 的 defect-on/off matched twin 产生预注册的 paired effect；
- `FORENSIC_SIGNAL_FOUND_UNATTRIBUTED`：找到负相关但无法证明是旧结果。

没有旧 fingerprint 时，某个 bug cell 变负只能支持 artifact-compatible mechanism，不能称旧结果已恢复。`NOT_RECOVERED` 也只能说明 frozen family 内未恢复，不能在低功效条件下“加强 RQ010B null”。

## Major findings

1. **Split stratification is not frozen.** “abstention status” 会随 recipe 变化；split 应使用 config-independent segment hash/cluster 或明确一个 rating-blind baseline eligibility。
2. **Discovery promotion is subjective.** `PI judgment selects <=5` 应替换为 deterministic gate、排序指标和 tie-breaker，最多一个 valid recipe。
3. **Defect twins lack one-factor/common-support contracts.** 必须同 scenes/candidates、同 endpoint，仅改变一个 defect，并报告 paired effect、覆盖和数值退化。
4. **Confirmation access is procedural, not truly sealed.** 全量 ratings 已存在；需分离 discovery/confirmation manifests、权限/loader denylist、冻结统计脚本与一次性 join log。
5. **Cache reuse is overclaimed.** 现有 final-IPV CSV 可用于 zero-recompute fingerprint 与 G2，但不能支持新的 window/smoothing/reference/counterpart axes；需先建 input/cache manifest 与 hashes。
6. **Compute plan needs an actual benchmark contract.** 12-scene stratified pilot 应报告 per-candidate/timepoint wall time、peak RSS、cache hit、ProcessPool oversubscription、atomic shard/DONE/retry/job manifest；CPU-first/no-GPU 方向合理。
7. **Execution record needs a run ID.** Phase H/S/C 应在 `reports/studies/RQ014.../RQ014_1_<topic>_<timestamp>_<sha>/` 下记录，而不是把所有执行产物直接放在 RQ root。
8. **Model routing does not match the intended economy.** v0 写 `codex exec, xhigh` 用于全部执行；确定性索引、网格、Slurm 和聚合应交给低成本执行单元，高级模型只负责 registry、gate、red team 和结论裁决。

## Required v1 structure

v1/amendment 至少应包含以下冻结合同：

1. Phase F closure matrix：DONE / NOT_FOUND_ON_SCANNED_SURFACES / INACCESSIBLE。
2. 一个逐行枚举的 valid registry 与独立 forensic registry。
3. 唯一 candidate-level within-scene estimand 和唯一主统计量。
4. rolling verifier 数学定义、timing、共同支持、missingness、normalization 和 envelope scale contract。
5. rating-blind eligibility + simulation-power gate，再决定 split。
6. deterministic 单-recipe promotion；confirmation 只检验一个 valid recipe。
7. 明确的 control family 和 `IPV-specific` 门槛。
8. 互斥且完备的 scientific/forensic verdict taxonomy。
9. G2 anchor suite 的 code/adapter/env/input hashes、expected values 和 tolerances。
10. HPC input/cache/job manifest、benchmark budget、retry/DONE semantics 和 run ID。
11. confirmation loader/access denylist 与一次性 join audit。
12. 对 RQ010B 的 scoped-addendum 规则：不删除已执行 operationalizations 的 bounded null。

## Re-review acceptance checklist

v1 只有全部满足下列条件才建议 PASS：

- [ ] 6-29 pilot 中间结果已检查或正式标记不可达；
- [ ] plan 明确 supersede v0，SHA 已登记；
- [ ] valid/forensic configs 均逐行可枚举；
- [ ] 不再存在 `approximately <=96` 或 PI 看榜后选 cell；
- [ ] primary estimand/endpoint/ties/missingness/permutation 完整；
- [ ] envelope 与 estimator timing 同尺度，或 timing family 已收窄；
- [ ] blind eligibility/power gate 已定义；
- [ ] confirmation family、alpha、最低效应和低功效状态明确；
- [ ] G2 anchors 与新 verifier fixtures 均有 hash/tolerance；
- [ ] clean-room reimplementation 与 independent replication 用词分开；
- [ ] `IPV-specific` 必须击败冻结的 physics/negative controls；
- [ ] HPC run layout、job prefix、benchmark、manifest 和 restart 语义完整。

## Evidence reviewed

- `reports/plans/RQ014_plan_v0_wod_e2e_rating_ipv_deviation_recovery_20260710.md`
- `reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/forensics_report.md`
- `START_HERE.md`, `STUDIES.md`, `main_workflow.log`
- RQ010B governing preregistration, phase-3 result, 10 Hz sensitivity and accepted `decision.md`
- `src/sociality_estimation/core/ipv_estimation.py`
- Tongji HPC shared usage guide
- `.codex-fleet/wod-e2e-ipv-recovery-design/board/validation.md`
- `.codex-fleet/wod-e2e-ipv-recovery-design/final/analysis.html`

Independent review lanes: statistics/method, implementation/HPC, scientific governance, plus lead adjudication. No reviewer modified the plan or submitted compute.

## Final ruling

RQ014 的问题值得继续，但 **v0 不是可执行 SAP**。最短正确路径不是运行宽网格，而是：完成最高价值取证 → 发布收窄 v1 → 冻结一个候选级 valid estimand 和一个主 endpoint → 完成功效与 adapter parity 门 → 最多晋级一个 valid recipe → 内部锁定确认 → clean-room 重实现与控制电池。

在这些修订完成前，任何 discovery 阳性都只能是探索性信号，任何 defect-cell 阳性都只能是 artifact-compatible，任何 confirmation 阴性都不能用来加强 RQ010B null。
