# RQ020-v3 复审 B：两阶段——先独立推导，再对照

你是复审 B。**只读审查，不执行任何计算、不写任何仓库产物。**

背景：要在 WOD-E2E 数据集上推进「人类不认可」这一级的验证。
已有两块材料：RQ010B 的欠功效零结果，和 RQ014 的一次规格恢复搜索。

## 第一阶段（独立推导）——**此阶段严禁打开任何任务书**

**完成第一阶段前，不得读取**：

- `.codex-fleet/rq020-wod-preference/`（整个目录，所有版本任务书都在里面）
- `reports/studies/RQ020_wod_preference/`（归档副本）
- `.codex-fleet/rq020-review2-a/`、`.codex-fleet/rq020-review-a/`、`.codex-fleet/rq020-review-b/`（其他复审）

在这个限制下，**自己从记录推导出**：

1. RQ014 那次搜索到底找到了什么、它的效力边界是什么
2. 它的配置里，哪些部分是可以复用的，哪些复用了就会继承选择效应
3. 若要把参照换成更新的纯人类 envelope，应该怎么设计这次检验
4. 偏离量应该怎么定义；越界与「到中心距离」哪个更该作主
5. 支持门该开还是该关；两者各自的代价
6. 会卡在哪里；哪些地方必须设停止点
7. 你会怎样防止评分泄漏进上游

起点（不限于此）：`reports/plans/RQ014_recovery_lane_v3.json`、
`reports/knowledge/RQ014_wod_e2e_rating_recovery/`、
`reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`、
`reports/knowledge/RQ016C_human_only_envelope/`、
`reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`。

**把推导结论完整写下来后，再进入第二阶段。**
第一阶段必须独立成节，且要能看出它是在未见任务书的情况下得出的。

## 第二阶段（对照）

现在可以打开 `.codex-fleet/rq020-wod-preference/board/RQ020_kickoff_v3.md`，
逐条对照：哪些一致；哪些**你的方案更好**、为什么；哪些任务书考虑到而你没有；
任务书有没有**你独立推导时就已预见的坑**。

## 禁读（全程）

- `.codex-fleet/rq020-review2-a/`、`.codex-fleet/rq020-review-a/`、`.codex-fleet/rq020-review-b/`
- `reports/studies/RQ020_wod_preference/RQ020_RA_review.md`、`RQ020_RB_review.md`
- RQ014 致盲相关的评分字段内容：遇到 rating/preference/score 字段先停下，不要读内容

报告写到 `.codex-fleet/rq020-review2-b/board/reports/RQ020_v3_RB_review.md`，两阶段分成两节。

---

## 收尾问题（必须逐条回答，按编号，不得合并）

**Q1** 沿用 RQ014 已冻结的配置（R04N / CH-W25 / H20 / terminal row）是否正当？
该配置本身是从 320 格搜索中**被选出来的**——把它固定下来复用，是否把那次搜索的选择效应
一并继承了？如果是，还有没有办法补救？

**Q2** 四个格（NEX/NMD × 支持门开/关）是否是正确的家族？把 **NEX + 支持门开** 定为主格
是否正确？**给出你自己的排序与理由**，不要只评论。

**Q3** 「本轮不是搜索，所以 p 值有意义，多重比较只需在 4 个上校正」——这个论证是否站得住？

**Q4** P2 与 P4 关闭支持门、沿用 pre-OOD-mask 外推。**在未验证支持的区间上算出的越界量，
报告它是否正当？** 还是说这等于给一个无效量套上合法外观？

**Q5** **这份任务书最可能以什么方式产出一个「看起来成立、实际不成立」的结果？**
给出最具体的那一条，包括它会怎样通过现有的全部检查与对照。

**Q6** 有没有哪一样东西缺失，会导致某个阶段的产物**无法解释**（不是不好看，是无法解释）？

**Q7** 功效：**给出你自己独立算出的数字**——各格在何种 n 下能识别何种效应。
任务书要求「每格报出可识别的最小效应」，这个要求**够不够**防止把零结果写成「无关联」？

**Q8** 总判定：`GO` / `GO_WITH_CHANGES` / `NO_GO`。
若非 GO，给出**单一最重要**的修改（只给一条）。

---

## 输出纪律

- 只读、不写任何仓库产物；**不执行任何 git 写操作**；不跑估计器、不投 Slurm 作业。
- 结尾写 `state: WAITING_ON_COMMANDER` 与真实 UTC 时间戳
  （`date -u +%Y-%m-%dT%H:%M:%SZ`，不要前瞻估计）。
- 不要对 `reports/` 做全仓库 `rg`。
- 禁用词：`estimability`、「测出/未测出 IPV」。
