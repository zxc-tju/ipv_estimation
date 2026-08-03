# A2 — 把审计结果写成科学报告

> 派发前把 `<REPO_ROOT>` 替换为仓库绝对路径，并确认 A1 的产出已通过指挥者自查。

## 角色与唯一目标

你是 `experimenter`。唯一目标：把 A1 跑出的数字变成 **`bounded_report.md`** ——
这是整个 RQ015A 的**科学交付物**，前面十天的一切都是为了这一份。

**你不重跑审计、不改代码。** 你的输入是已落盘的结果文件。

## 过程强度

描述性 / 诊断性产出，按 `AGENTS.md` 的 Research Velocity Principle：**一轮即可**。
不做多轮复审、不出多个报告版本。写一版好的，交付。

## 输入

```
<REPO_ROOT>/reports/studies/RQ015A_ipv_estimability_labelling/<最新 run 目录>/
    concentration_ledger*        逐行台账（parquet）
    concentration_ledger_summary.csv
    portraits.json
    c0_routing.json
    run_receipt.json
背景：reports/plans/RQ015A_plan_v8_concentration_audit_20260731.md
      reports/knowledge/RQ015A_ipv_estimability_labelling/known_issues_and_audit_boundary_20260730.md
```

## 这份报告要回答什么

RQ015A 的主量是连续的 `q_eff = K_eff/K`：
`q_eff → 1` 表示 7 个候选的权重摊平，**这一帧的 IPV 数值不携带候选间的判别信息**；
`q_eff = 1/7` 表示锁定单一候选。

报告要交付六件东西：

**1. 主分布** — 逐产物、逐层的连续 `q_eff` 分布。给分位数与直方图，不要只给均值。
同时报 `ATTEMPTED / NOT_ATTEMPTED / UNKNOWN` 三态占比。

**2. 分层：坏在哪儿** — 按数据源（nuplan / waymo / OnSite / WOD）、
按配置（hw4 vs hw10）、按视角（key_agent_1/2、ego/counterpart）、按 case、按 episode。
此前只知道"Waymo 最差"这一句，现在要给出差多少、是全程差还是只在某类场景差。

**3. 可用子集清单** — **最有实操价值的一项**。哪些 case / episode 的 IPV 携带判别信息。
输出一份可直接拿去筛样本的名单（csv），并给出它占全集的比例。
RQ009 / RQ013 / RQ014 可以据此改成"只在有信息的子集上做"。

**4. 协变因素** — Spearman 秩相关 + case-cluster bootstrap 95% CI（B=2000, seed 20260726）。
回答"什么条件下这个估计器会失效"。**只描述，不下因果**：
禁止"驱动""导致""预测""解释了"这类词，用"与……共变""在……条件下更高"。

**5. C0 路由** — 逐下游消费者（M3 训练标签 / RQ009 包络 / RQ014 筛选）给出四态之一：
`NOT_APPLICABLE` / `NO_AUDIT_TRIGGER_DETECTED` / `OWNER_REANALYSIS_REQUIRED` /
`INDETERMINATE_UNKNOWN_PROVENANCE`，每态附 reason code 与敏感性。
这是唯一带**行动后果**的产物——它可能会说 RQ009 的某条结论需要重估。

**6. 覆盖与限制披露** — 见下节，必须在标题级位置，不能塞进脚注。

## 必须遵守的表述边界

```
1. 覆盖是 4/6 产物。WOD 一支为**部分覆盖**（已取回 wod_rq010b_full479_audited 906 行；
   phase1b/schemeB 与 rq014_g2r_anchor_scores 经 PI 裁定不取回）。
   **禁止表述为"全语料"或"全 WOD"。**
2. 全文禁止 estimability 与"测出/未测出 IPV"。
   可辩护的说法：权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息。
   完整 estimability 是 RQ007 的合取条件，不在本 RQ 范围。
3. 本 RQ **不衡量 IPV 估得准不准**（只量权重集中度），
   **不区分近均匀的成因**（下溢 / 当前网格下真平坦 / 模型失配 —— 那是 RQ015B）。
4. 报告用 policy bins（4/7 与 0.93）是**政策选择**，不是数据中发现的边界；
   必须与九组敏感性一并披露；任一档占比极差 > 10pp 即标 BINS_WITHHELD_UNSTABLE、
   只发连续分布。
5. warm-up 占位（ipv_error = 1.0，估计器从未运行）与均匀回退
   （ipv_error ≈ 0.6220355269907728，跑了但权重摊平）是**两件不同的事**，
   报告中必须分开呈现，不得合并成"零值率"。
```

## 图

按 `AGENTS.md` 的图表标准：每个结论配自己的证据图，不放"顺便有意思"的图。
标注单位与样本量，给不确定性，色盲友好，同时导出 PNG 与 PDF。
数量以说清六件事为限，不追求多。

## 交付

```
reports/studies/RQ015A_ipv_estimability_labelling/<run 目录>/bounded_report.md
    + figures/（PNG + PDF）
    + usable_subset.csv（第 3 项的名单）
```

## 结项报告格式（给指挥者看的，不是那份 bounded_report）

```markdown
## 状态
SUCCESS | PARTIAL | FAILED

## 六个核心数字（我最想先看到这个）
1. 全语料 ATTEMPTED / NOT_ATTEMPTED / UNKNOWN 三态占比
2. q_eff 中位数与四分位（逐产物）
3. 近均匀（q_eff >= 0.93）占比（逐产物）
4. 可用子集：多少 case / 占比
5. 最强的三个协变因素（Spearman + CI）
6. C0 四态判定（逐下游消费者）

## bins 稳定性
BINS_REPORTABLE | BINS_WITHHELD_UNSTABLE（附极差）

## 交付文件
路径清单

## 我对结果的判读（≤15 行）
最值得 PI 注意的一两件事；如果数字与此前的探针估计（约四成零值、
过半近均匀）不一致，明确指出并给出可能原因

## 表述边界自查
5 条逐条打勾

## 未做 / 存疑
```

## 禁止

不重跑审计、不改代码、不写因果措辞、不出多版报告、
不把覆盖说成全语料、不使用 estimability 表述。
