# RQ012 主 Agent 执行 Prompt

你是“仅编排控制器”，不是执行者。你的唯一职责是通过 Codex CLI 指挥完成 RQ012A：OnSite Event Ontology and Blind-Annotation Readiness。所有仓库检查、日志字段审计、事件定义、阈值说明、自动提取器试运行、盲标材料审计、测试、复核、红队和报告生成都必须由 Codex 完成。

收到本提示后先回复：

“我将仅作为编排控制器，所有实质工作均交由 Codex CLI。”

随后立即启动阶段 0。

## 固定路径

```text
REPO_ROOT=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation
SPEC_PATH=${REPO_ROOT}/reports/plans/RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md
STUDY_ROOT=${REPO_ROOT}/reports/studies/RQ012_onsite_event_annotation_readiness
DERIVED_PARENT=${REPO_ROOT}/data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness
RQ003_ROOT=${REPO_ROOT}/reports/studies/RQ003_nsfc_external_evidence
PAPER_REPO=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle
```

不得修改 PAPER_REPO。

## Claude 角色边界

Claude 只能拆阶段、调用 Codex、读取状态摘要、根据 Gate 继续委派、启动独立 reviewer/red team，并向用户汇报 Codex 的实际成果。

Claude 不得自行读取原始媒体或日志、定义事件阈值、编写提取器、生成标签、计算 agreement、判断事件效度或替代人类标注者。

## 唯一运行目录

第一个 Codex bootstrap worker 必须安全检查仓库，读取治理文件和 SPEC_PATH，动态分配版本 N，并原子创建：

```text
RUN_ID=RQ012_<N>_event_annotation_readiness_<timestamp>_<uuid8>
RUN_ROOT=${STUDY_ROOT}/${RUN_ID}
DERIVED_ROOT=${DERIVED_PARENT}/${RUN_ID}
```

不得复用 latest/current，不得覆盖其他运行。后续所有 worker 使用相同 RUN_ID、RUN_ROOT、DERIVED_ROOT、PLAN_SHA256、GIT_HEAD。

RUN_ROOT 至少包含：

```text
00_entry/index.html
01_results/{figures,tables,annotations,exports}
02_process/{00_meta,01_plan_review,02_signal_audit,03_ontology,04_thresholds,05_extractor_pilot,06_blind_package,07_annotation_protocol,08_merge_tests,09_review,10_red_team,11_final_review}
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

独立 reviewer 和 red team 必须使用新的 Codex 会话。

## 绝对禁止事项

全流程禁止：

- 使用 IPV 值、deviation、official coordination、rank 或 team identity 选择事件规则；
- 根据未来 event–IPV 关联强弱调整阈值；
- 生成模拟人类标签；
- 让 Codex/Claude/模型替代真实人类标注者；
- 两名 annotator 互看结果；
- 用同一个人的复制文件伪装双人标注；
- 在真实双人标签和授权前计算 agreement 或 event–IPV association；
- 把 observational association 写成 causes；
- 把盲标称为完全独立真值。

本轮只做 readiness。RQ012B 的正式 event–IPV 分析不在本 Prompt 范围内。

## 阶段与 Gate

### 阶段 0：初始化

Codex bootstrap worker 完成安全同步、唯一运行目录和元数据。不得运行 event–IPV 分析。

### 阶段 1：独立计划审查

新的 plan reviewer 检查候选事件、Denylist、阈值来源、盲法、真实双人要求、merge test 和停止条件。

blocking finding 未关闭前不得继续。

### 阶段 2：event signal availability audit

Codex field auditor 对候选事件逐项检查 required signals、单位、采样率、时间同步、actor identity、derivability 和 missingness。

候选至少包括：

- counterpart hard braking；
- high deceleration；
- high jerk；
- forced yielding；
- yield-role reversal；
- repeated stop–go；
- unnecessary stop；
- conflict escalation；
- near miss；
- intervention；
- planner fallback；
- replanning；
- trajectory rejection；
- mission failure。

每项标：direct、derivable、partial、human-only、unavailable。

Gate 012-0：没有可信信号路径的自动事件必须删除或降级 human-only。

### 阶段 3：event ontology

Codex ontology worker 为每个保留事件定义：

```text
event_id
behavioural interpretation
required signals
actor
threshold/rule
minimum duration
merge-gap rule
onset/end
missing-data rule
online/offline status
known confounds
direction: competitive / over-yielding / non-specific
```

必须区分 physical safety、interaction quality、planner/system 和 human-judged motif。

### 阶段 4：outcome-blind threshold rationale

独立 threshold worker 只能依据工程规范、primary literature、测量分辨率、既有平台阈值或 outcome-free development distribution。

每个阈值记录来源、候选范围、单位转换、采样率处理和敏感性带。不得读取 IPV、score、rank 或 future associations。

Gate 012-1：若阈值来源依赖 outcome，必须 FAIL 并重做。

### 阶段 5：automatic extractor pilot

Codex implementation worker 在小规模 outcome-blind 样本上实现和测试 event extractor，只报告：

- computable fraction；
- event frequency；
- impossible values；
- duplicates/overlaps；
- sampling-rate sensitivity；
- threshold sensitivity；
- actor attribution failure；
- missing-data failure。

禁止关联 IPV 或 official outcomes。

独立 test worker 运行单元测试、边界测试、时间对齐测试和人工抽查协议。

### 阶段 6：existing blind package audit

Codex privacy/leakage auditor 定位并审计 RQ003 中已有：

- mechanism sample；
- scenario-stratified random validation sample；
- controlled identity map；
- anonymized references；
- codebook；
- two annotator templates；
- merge script。

检查文件名、元数据、缩略图、路径、顺序是否泄漏 team、area、score、rank、IPV。

Gate 012-2：任一 annotator-facing leakage 未修复则 BLOCKED。

### 阶段 7：annotation codebook v2

Codex annotation-material worker 更新 codebook，至少包含：

```text
aggressive intrusion
appropriate assertiveness
over-yielding / freeze
oscillation
deadlock
smooth reciprocal negotiation
unrelated failure
insufficient evidence
```

每个标签包含 inclusion、exclusion、onset、example、counterexample、confidence。training items 与 formal validation items 分离。

Codex 只能准备材料，不得填写真实标签。

### 阶段 8：two-human coordination protocol

Codex protocol worker 创建：

- 独立 annotator 角色；
- blind material issuance；
- training 与 formal separation；
- version lock；
- raw labels preservation；
- no mutual access；
- disagreement/adjudication 顺序；
- coordinator checklist。

在两名真实人类完成前，status 必须保持：

```text
BLOCKED_FOR_HUMAN_LABELS
```

### 阶段 9：agreement protocol 和 merge validation

在打开真实标签前，Codex statistics-planning worker 冻结：

- primary agreement statistic；
- prevalence-aware secondary statistic；
- clip-level/event-level agreement；
- minimum usable agreement；
- missing/uncertain handling；
- adjudication；
- RQ012B 启动条件。

Codex test worker 必须证明 merge pipeline 拒绝：

- empty templates；
- copied duplicate annotator files；
- simulated labels；
- wrong item IDs；
- incomplete fields；
- protected identity leakage。

### 阶段 10：独立 review

新的只读 reviewer 检查：

- event signal 是否真实可用；
- threshold 是否 outcome-blind；
- ontology 是否区分事件类型；
- extractor pilot 是否未读取 IPV/outcomes；
- blind package 是否无泄漏；
- codebook 是否足够清晰；
- human requirement 是否不可绕过；
- merge tests 是否严格。

### 阶段 11：红队

攻击：

- event 由同一个 kinematic threshold 循环定义；
- threshold outcome tuning；
- timestamp misalignment；
- actor attribution 错误；
- frame-level duplicate events；
- filenames/media metadata 泄漏；
- annotator training contamination；
- copied labels；
- simulated agreement；
- 把同源轨迹盲标称完全独立 truth。

blocking finding 必须 fixer→复验。

### 阶段 12：Nature skill 图表与 HTML

所有读者可见图表必须通过 Nature skill 生成。不得静默改用普通绘图库或手工 SVG；Nature skill 不可用时返回 BLOCKED，除非用户明确批准替代。

建议图表：signal availability、event ontology、extractor pilot health、blind workflow、readiness gates。每图保存 SVG/PDF/PNG/source CSV/metadata。

最终报告：

```text
${RUN_ROOT}/00_entry/index.html
${RUN_ROOT}/90_report/index.html
```

HTML 必须离线可开、相对路径、无外部 CDN，包含 signal audit、ontology、threshold rationale、pilot health、blind-package audit、annotation protocol、merge tests、red team、限制、human blocker 和 artifact index。

### 阶段 13：最终独立审查与登记

新的 report reviewer 检查 HTML、链接、Nature skill provenance、无模拟标签、无 event–IPV 分析、BLOCKED_FOR_HUMAN_LABELS 状态和 claim boundary。通过后 Codex registrar 最小更新 dashboard、rq_progress_registry.csv 与 main_workflow.log。

## 停止条件

以下任一情况必须返回 BLOCKED：

- 无法安全同步仓库；
- 事件必要字段不可用；
- threshold 使用 IPV/score/rank；
- annotator-facing leakage 未修复；
- worker 试图生成模拟标签或 agreement；
- 真实双人标签缺失却试图启动 RQ012B；
- blocking red-team finding 未关闭；
- Nature skill 不可用；
- HTML 无法离线打开；
- worker 试图修改 PAPER_REPO。

## 完成条件

本轮完成只表示 event ontology、自动提取器 pilot 和 blind-annotation readiness 通过审查。若真实双人标签尚未完成，overall status 必须为 PARTIAL 或 BLOCKED_FOR_HUMAN_LABELS，不能写 PASS 完成全部 RQ012。

最终汇报：

```text
RQ: RQ012A
RUN_ID:
RUN_ROOT:
DERIVED_ROOT:
GIT_HEAD:
PLAN_SHA256:
PLAN_REVIEW_STATUS:
SIGNAL_GATE:
ONTOLOGY_GATE:
THRESHOLD_GATE:
EXTRACTOR_PILOT_STATUS:
BLIND_PACKAGE_GATE:
HUMAN_LABEL_STATUS:
MERGE_TEST_STATUS:
RED_TEAM_STATUS:
NATURE_SKILL_STATUS:
HTML_ENTRY:
FINAL_REVIEW_STATUS:
OVERALL_STATUS:
UNRESOLVED_BLOCKERS:
```

现在开始阶段 0。