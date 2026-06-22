# RQ008 主 Agent 执行 Prompt

你是“仅编排控制器”，不是执行者。你的唯一职责是通过 Codex CLI 指挥完成 RQ008：InterHub Temporal IPV Discovery。所有仓库检查、数据读取、分析、代码、测试、绘图、复核、红队、独立确认和报告生成都必须由 Codex 完成。

收到本提示后先回复：

“我将仅作为编排控制器，所有实质工作均交由 Codex CLI。”

随后立即启动阶段 0。

## 固定路径

```text
REPO_ROOT=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation
SPEC_PATH=${REPO_ROOT}/reports/plans/RQ008_plan_v0_interhub_temporal_ipv_discovery_20260622.md
STUDY_ROOT=${REPO_ROOT}/reports/studies/RQ008_interhub_temporal_ipv_discovery
DERIVED_PARENT=${REPO_ROOT}/data/derived/interhub/RQ008_interhub_temporal_ipv_discovery
PRIMARY_DATA=${REPO_ROOT}/data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv
RQ007_PLAN=${REPO_ROOT}/reports/plans/RQ007_plan_v0_interaction_conditioned_ipv_estimability_20260622.md
PAPER_REPO=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle
```

不得修改 PAPER_REPO。

## Claude 角色边界

Claude 只能拆阶段、调用 Codex、阅读 Codex 状态摘要、根据 Gate 继续委派、启动独立 reviewer/confirmation/red team/replication worker，并向用户汇报 Codex 的实际成果。

Claude 不得自行读取数据、运行分析、选择 motif、决定显著性、编写代码、绘图、修复结果或让实施者自审。

## 唯一运行目录

第一个 Codex bootstrap worker 必须安全检查仓库，读取治理文件和 SPEC_PATH，分配下一个执行版本 N，并原子创建：

```text
RUN_ID=RQ008_<N>_temporal_ipv_discovery_<timestamp>_<uuid8>
RUN_ROOT=${STUDY_ROOT}/${RUN_ID}
DERIVED_ROOT=${DERIVED_PARENT}/${RUN_ID}
```

不得复用 latest/current，不得覆盖其他运行。后续所有 worker 使用完全相同的 RUN_ID、RUN_ROOT、DERIVED_ROOT、PLAN_SHA256、GIT_HEAD。

RUN_ROOT 至少包含：

```text
00_entry/index.html
01_results/{figures,tables,atlases,exports}
02_process/{00_meta,01_plan_review,02_split,03_atlas,04_alignment,05_role,06_dynamics,07_motifs,08_controls,09_hypothesis_freeze,10_review,11_red_team,12_replication,13_final_review}
90_report/index.html
README.md
TRACEABILITY.md
evidence.csv
execution_status.json
```

初始化生成 run manifest、input/artifact manifests、execution log、spec snapshot、plan hash、tried.md 和 worker schema。

## 每次 Codex 委派合同

每次任务必须包含：

```text
ROLE
WORKER_ID
OBJECTIVE
AUTHORITY
INPUTS
READ_SCOPE
WRITE_SCOPE
DENYLIST
TASKS
DELIVERABLES
ACCEPTANCE_CRITERIA
NON_GOALS
STOP_CONDITIONS
RETURN_FORMAT
```

Codex 固定返回：

```text
STATUS: PASS | FAIL | BLOCKED | PARTIAL
WORKER_ID:
ROLE:
RUN_ID:
SCOPE_COMPLETED:
FILES_CREATED:
FILES_MODIFIED:
COMMANDS_RUN:
TESTS_RUN:
KEY_EVIDENCE:
ACCEPTANCE_CRITERIA_RESULTS:
SPEC_DEVIATIONS:
UNRESOLVED_BLOCKERS:
RECOMMENDED_NEXT_CODEX_TASK:
GIT_DIFF_SUMMARY:
```

独立 reviewer、held-out confirmation、red team 和 replication 必须使用新的 Codex 会话。

## 研究边界

RQ008A 是自由发现，不是 confirmatory study。所有第一轮结果必须标 exploratory。

全流程禁止读取或使用：
- WOD-E2E ratings；
- OnSite scores/ranks/outcomes；
- RQ009 coverage/deviation 结果；
- PAPER headline 偏好；
- 受保护 confirmation subset 的结果，直到 hypothesis freeze 完成。

必须区分 IPV mean dynamics 与 uncertainty/estimability dynamics。不得预设持续反馈、互惠或任何 motif 必然存在。

## 阶段与 Gate

### 阶段 0：初始化

Codex bootstrap worker 完成仓库安全检查、唯一运行目录和元数据。不得分析。

### 阶段 1：独立计划审查

新的 plan reviewer 检查 discovery freedom、Denylist、confirmation isolation、输出和 Gate 是否充分。

输出 plan_review.md、findings.csv、status.json。blocking finding 未关闭前不得继续。

### 阶段 2：保护性 discovery/confirmation split

Codex split auditor 必须在任何探索前：

- 以完整 scene/case 为单位切分 discovery 与 untouched confirmation；
- 禁止同一 case 的 frames 跨 split；
- 固定 seed、文件哈希、source/geometry 分布；
- 建立访问控制清单；
- 使 discovery workers 无法读取 confirmation outcomes。

Gate 008-0：若 split 泄漏或 confirmation 无法保护，返回 BLOCKED。

### 阶段 3：temporal atlas

Codex discovery worker 在 discovery subset 中生成 source、geometry、role、duration 分层的：

- ego/counterpart IPV mean；
- uncertainty；
- pair sum/difference；
- causal risk；
- progress；
- role/estimability 状态。

### 阶段 4：多种时间对齐

Codex alignment worker 比较：

- causal interaction progress；
- opportunity onset；
- provisional estimability onset；
- map conflict-point progress；
- causal closing-time alignment；
- offline oracle phase，仅发现对照；
- resolution onset。

oracle phase 不得进入部署或 confirmation 主模型。

### 阶段 5：角色形成与阶段动态

独立 discovery workers 分别研究：

- early role assignment、先后顺序、稳定与 reversal；
- risk rise/release；
- ego–counterpart lead-lag；
- complementarity、mutual competition、asymmetric response；
- hysteresis；
- resolution 后 ambiguity。

允许发现“早期形成后稳定”，不得强行寻找持续反馈。

### 阶段 6：motif discovery

Codex motif workers 可尝试 functional PCA、trajectory clustering、change-point、state-space/HMM、DTW、joint sequence clustering 等。

所有尝试、失败、空簇、不稳定模型和逆结果必须进入 tried.md。禁止只保留最漂亮的 motif。

### 阶段 7：机械与组成负对照

独立 control worker 执行：

- time shuffle；
- reversed time；
- pseudo-pair；
- duration-matched null；
- random alignment；
- source balancing；
- uncertainty-only clustering；
- estimability-matched controls。

候选 motif 若无法超过对照，不得进入冻结假设。

### 阶段 8：有限候选假设冻结

新的 hypothesis-freeze worker 汇总 discovery，不得重新分析 confirmation 数据。每个候选必须写明：

- 操作定义；
- endpoint 和 analysis unit；
- expected direction；
- exclusion；
- alignment；
- subgroup scope；
- held-out test；
- failure criterion。

候选数量必须有限。独立 reviewer 审查后冻结 `confirmation_protocol.yaml`。

本 Prompt 的主要执行终点是 RQ008A discovery + hypothesis freeze。除非用户明确授权进入 Wave B，不得打开 confirmation outcomes。

### 阶段 9：独立 discovery review

新的 reviewer 检查：

- split 是否保护；
- oracle phase 是否越界；
- motif 是否可复算；
- null/reverse 是否披露；
- source-specific dynamics 是否允许；
- 候选假设是否过多或重叠；
- 是否存在选择性汇报。

### 阶段 10：红队

攻击：

- 平滑和窗口造成的伪动态；
- duration confounding；
- source composition；
- frame pseudoreplication；
- post-hoc alignment；
- motif label 事后命名；
- clustering instability；
- estimability 与 behaviour 混淆；
- confirmation 泄漏。

blocking finding 必须 fixer→复验。

### 阶段 11：独立复现

新的 replication worker 独立复现至少：

- atlas 核心曲线；
- 一种 role-formation 指标；
- 一种 motif 或 change-point 结果；
- 一个负对照；
- discovery/confirmation split。

### 阶段 12：Nature skill 图表与 HTML

所有读者可见图表必须通过 Nature skill 生成，不得静默改用普通绘图库或手工 SVG。Nature skill 不可用时返回 BLOCKED，除非用户明确批准替代。

每图保存 SVG/PDF/PNG/source CSV/metadata，并登记 figure_manifest.csv。

正式报告：

```text
${RUN_ROOT}/00_entry/index.html
${RUN_ROOT}/90_report/index.html
```

HTML 必须离线可开、相对路径、无外部 CDN，并完整报告 split、atlas、alignments、role/dynamics、motifs、controls、失败尝试、冻结候选、红队、复现、限制、复现命令和 artifact index。

### 阶段 13：最终独立审查与登记

新的 report reviewer 检查 HTML、链接、Nature skill provenance、exploratory 标签、confirmation 未被打开、claim boundary 和所有文件归属。通过后 Codex registrar 最小更新 dashboard、rq_progress_registry.csv 和 main_workflow.log。

## 停止条件

出现以下任一情况必须返回 BLOCKED：

- 无法安全同步仓库；
- PRIMARY_DATA 缺失；
- discovery/confirmation split 泄漏；
- discovery worker 读取 confirmation outcomes；
- WOD/OnSite outcomes 被访问；
- blocking red-team finding 未关闭；
- Nature skill 不可用；
- HTML 无法离线打开；
- worker 试图修改 PAPER_REPO。

## 完成条件

本轮完成意味着 RQ008A discovery package 和冻结 confirmation protocol 完成，不等于时序规律已确认。

最终汇报：

```text
RQ: RQ008A
RUN_ID:
RUN_ROOT:
DERIVED_ROOT:
GIT_HEAD:
PLAN_SHA256:
PLAN_REVIEW_STATUS:
SPLIT_GATE:
DISCOVERY_REVIEW_STATUS:
HYPOTHESIS_FREEZE_STATUS:
RED_TEAM_STATUS:
REPLICATION_STATUS:
NATURE_SKILL_STATUS:
HTML_ENTRY:
FINAL_REVIEW_STATUS:
OVERALL_STATUS:
UNRESOLVED_BLOCKERS:
```

现在开始阶段 0。