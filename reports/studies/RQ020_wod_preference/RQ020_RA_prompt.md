# RQ020 复审 A：从任务书出发的独立审查

你是复审 A。**只读审查，不执行任何计算、不写任何仓库产物。**

## 你要审什么

任务书：`.codex-fleet/rq020-wod-preference/board/RQ020_kickoff_v1.md`

它规划的是 RQ020：在 WOD-E2E 上检验「强势侧非典型是否对应人类不认可」，
分三个阶段（抬高样本量 → 验证参照迁移 → 检验有向假设），每阶段末尾有 STOP 闸。

## 你的立场

**从任务书出发**，逐条核对它的设定是否站得住。你可以并且应该去读它引用的一切依据，
自己去验证那些数字，不要采信任务书的转述。

建议核对的依据（不限于此）：

- `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`（RQ010B 的 null）
- `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f/`
  下的预注册与 phase3 结果
- `reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`（人类对齐为何落到 WOD）
- `reports/knowledge/RQ016C_human_only_envelope/`、`reports/knowledge/RQ018_*`、`reports/knowledge/RQ019_*`
- `reports/studies/RQ016C_human_only_envelope/*/envelope_model/feature_contract.json`

## 禁读

- **`.codex-fleet/rq020-review-b/` 整个目录**（另一位复审的工作区）
- RQ014 致盲相关的评分字段内容：遇到 rating/preference/score 字段先停下，不要读内容


---

## 收尾问题（必须逐条回答，按编号，不得合并）

**Q1** 三阶段分解（抬功效 → 验参照迁移 → 检验假设）是否正确？闸是否设在对的位置？
若你认为顺序或闸位应当不同，给出你的方案与理由。

**Q2** 两个闸的阈值（阶段一 n ≥ 150；阶段二支持门通过率 ≥ 30%）是否站得住？
**给出你自己独立算出的数字**，不要只评论。若你算出的阈值不同，说明依据。

**Q3** 「本轮不是重跑 RQ010B 的 null，而是在足够功效下检验一个新假设」——
这个定位是否站得住？如果你认为它实质上仍是同一检验，直接说。

**Q4** 评分接触纪律（全程 ratings-blind、只在最终检验连接一次、连接前冻结预注册）
是否足以防止评分泄漏进上游？指出任何可能的泄漏路径。

**Q5** **这份任务书最可能以什么方式产出一个「看起来成立、实际不成立」的结果？**
给出最具体的那一条，包括它会怎样通过现有的全部检查。

**Q6** 有没有哪一样东西缺失，会导致某个阶段的产物**无法解释**（不是不好看，是无法解释）？

**Q7** 三个阶段各自的失败分支是否充分？指出任何一个「失败了但任务书没说怎么办」的情形。

**Q8** 总判定：`GO` / `GO_WITH_CHANGES` / `NO_GO`。
若非 GO，给出**单一最重要**的修改（只给一条，不要列清单）。

---

## 输出纪律

- 报告写到 `.codex-fleet/rq020-review-a/board/reports/RQ020_RA_review.md`。
- 只读、不写任何仓库产物；**不执行任何 git 写操作**；不跑估计器、不投 Slurm 作业。
- 结尾写 `state: WAITING_ON_COMMANDER` 与真实 UTC 时间戳
  （`date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计）。
- 不要对 `reports/` 做全仓库 `rg`（宽泛检索会把 RQ003 controlled-access 行拉进上下文）。
- 禁用词：`estimability`、「测出/未测出 IPV」。
