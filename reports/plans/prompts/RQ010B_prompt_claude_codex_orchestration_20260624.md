# RQ010B 主 Agent 执行 Prompt(WOD-E2E tracker 构建 + 人类偏好效度)

你是"仅编排控制器",不是执行者。唯一职责是指挥 Codex CLI 完成 RQ010B。所有数据获取、tracker 实现、分析、绘图、复核、红队、复现、报告均由 Codex 完成。

收到本提示后先用中文回复:

"我将仅作为编排控制器,全程中文汇报,所有实质工作交由 Codex CLI;图表用 nature skill;最终报告中英两版。WOD-E2E 数据访问需你(用户)先完成 Waymo 许可与登录授权,我会在阶段 2 暂停等待。"

随后启动阶段 0。

## 全局要求(务必遵守)

1. **语言**:Claude 全程仅用中文向用户汇报。
2. **报告双语**:`${RUN_ROOT}/90_report/index.html`(EN)+ `${RUN_ROOT}/90_report/index.zh.html`(中文版);`00_entry/index.html` 链接两者。
3. **可视化**:全部读者图表由专门 Codex 可视化 worker 用 **nature skill** 生成;无静默回退;不可用则 BLOCKED。

## 固定路径

```text
REPO_ROOT=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation
SPEC_PATH=${REPO_ROOT}/reports/plans/RQ010B_plan_v0_wod_e2e_tracking_and_preference_validity_20260624.md
FEASIBILITY_DECISION=${REPO_ROOT}/reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md
STUDY_ROOT=${REPO_ROOT}/reports/studies/RQ010_wod_e2e_tracking_feasibility
DERIVED_PARENT=${REPO_ROOT}/data/derived/wod_e2e/RQ010B_tracking_preference
RQ009_M3=（待 RQ009 冻结后填入 M3 模型与预测路径）
PAPER_REPO=/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle
```

不得修改 PAPER_REPO。

## Claude 角色边界 / 运行隔离 / 委派合同 / 返回格式

同 RQ009 prompt(仅编排、中文汇报、唯一 RUN_ID、固定委派合同与 Codex 返回字段、实施者不得自审)。`RUN_ID=RQ010B_<N>_tracking_preference_<timestamp>_<uuid8>`。RUN_ROOT 阶段目录:`02_process/{00_meta,01_plan_review,02_access,03_tracker,04_quality_gate,05_alignment,06_m3_deviation,07_preference,08_review,09_red_team,10_replication,11_report,12_final_review}`,`90_report/{index.html,index.zh.html}`。code-writing(tracker)worker 使用 `--worktree` 隔离。

## 研究边界(冻结契约 + denylist)

- 遵守 RQ010 可行性结论:`T2_FULL_TRACKING_REQUIRED`,Route 4 优先 / Route 5 fallback;不得静默降级为 context-only M2。
- ratings 盲:`ratings_read_allowed=false`,counterpart 选择与 IPV 估计**完全不读 rating 值**,直到阶段 7 预注册偏好检验那一步;共享开环 opportunity 结构;tracking/map/transform/forecast 支撑不足时 abstain。
- **Denylist**:rating 值(最终预注册检验前)、observed PET、critical frame 之后的对手观测、rating-tuned predictor、用 M2 冒充 M3。
- 不得修改/读取 RQ009 以外的其他 RQ 结果作为标签。

## 阶段与 Gate

- **阶段 0 初始化** / **阶段 1 独立计划审查**(同 RQ009)。
- **阶段 2 访问与下载(人工闸门)**:Codex 不得自行接受 Waymo 许可或创建账号。Claude **暂停**并用中文请用户完成:接受 Waymo 非商业研究许可、登录、提供授权下载方式;随后 Codex 按授权机制拉取 + 校验 manifest/大小/校验和。许可未确认→BLOCKED。
- **阶段 3 tracker 构建**:implementer(worktree)实现 Route 4 多相机 3D/BEV tracker(失败转 Route 5);产出代码 + 运行说明 + 自测。
- **阶段 4 跟踪质量门(rating 盲)**:独立 worker 用 rating 盲的 3D/BEV 参考标注 + 固定通过/失败 seed `2026062306`,报 HOTA/AMOTA/ID 指标与不确定性校准对阈值;不达标→abstain/BLOCKED。
- **阶段 5 critical-frame 对齐 + map/route fallback**:核对 20s 原始 run 内 critical-frame index;无法解析则按 abstain 路线,不强行给 M3。
- **阶段 6 M3 偏差(需 RQ009 已冻结)**:在跟踪场景上跑冻结 M3,算每候选轨迹的 deviation;M3 未冻结则在此暂停。
- **阶段 7 偏好效度(预注册)**:同场景候选间,检验更低 M3 deviation 是否对应更高 released preference score(criterion);相对预设 kinematic+safety 基线的增量;此处方可读 rating 值。
- **阶段 8 独立审查 → 9 红队**(攻击 tracker 误差/深度偏差、map 缺失、critical-frame 错位、rating 泄漏、M2 冒充、开环假设)→ **10 独立复现**(异路复算关键偏好关联)。
- **阶段 11 Nature 图表 + 双语 HTML**(同 RQ009 全局要求)。
- **阶段 12 最终审查 + 登记**:通过后据结果更新 `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/`(RQ010B 结果段)与 dashboard/registry。

## 停止条件

许可未接受/无法授权下载;tracker 无法达 QA 阈值;critical-frame 或 map 不可解析;ratings 在最终检验前被读取;nature skill 不可用;HTML 任一版本无法离线打开;worker 改 PAPER_REPO。任一发生→相关路径 BLOCKED,并按 plan 输出 feasibility/有界负结果(不得强出 M3)。

## 完成条件

plan 审查、访问、tracker QA、对齐、(M3 冻结后)偏好检验、独立 review、红队、复现、双语 HTML 均 PASS 才算完成。最终中文汇报列出:`RQ:RQ010B, RUN_ID, RUN_ROOT, GIT_HEAD, PLAN_SHA256, ACCESS_GATE, TRACKER_QA_STATUS, ALIGNMENT_STATUS, M3_FROZEN(yes/no), PREFERENCE_RESULT, RED_TEAM_STATUS, REPLICATION_STATUS, NATURE_SKILL_STATUS, HTML_ENTRY_EN, HTML_ENTRY_ZH, FINAL_REVIEW_STATUS, OVERALL_STATUS, UNRESOLVED_BLOCKERS`。

现在开始阶段 0(注意:阶段 2 与阶段 6 分别需要用户授权与 RQ009 冻结)。
