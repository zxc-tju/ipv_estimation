# 在线 IPV 合理区间估计报告（可部署版）

入口页：`00_entry/index.html`（自包含 HTML，4 张内嵌 SVG 图 + 锁定数字表，离线可开）。

本报告回答的问题是：online sociality verification 时，如何在线、准确地估计"人类驾驶人在相同交互场景下的 IPV 合理区间"，以取代现行 verifier 基于 **预测 PET → 风险分箱 → 经验包络** 的查询（不准）。

本报告是 `3-online_ipv_interval-1`（2026-06-18 interval-query 探索版）的**后继与升级**：它把当时"最佳但需前缀因果重建后才可部署"的 rolling-IPV 自锚方案，真正完成了**严格前缀因果重建**并**锁定了生产数字**。

## 指标含义

| 指标 | 含义 | 怎么判断好坏 |
|---|---|---|
| P90 interval [q05,q95] | 人类 IPV 的 90% 合理区间 | 约 90% 真实 IPV 应落入 |
| 覆盖率 Coverage | 真实 IPV 落入区间的比例 | 应接近 0.90；低于即欠覆盖、不安全 |
| 平均宽度 Mean width | 区间上限减下限 | coverage 达标时越小越精准 |
| Winkler 区间分数 | 同时惩罚过宽与漏覆盖 | 越低越好 |
| false-flag 误报率 | 把合规人类轨迹误判为偏离的比例 | 越低越好 |
| CQR / conformal | 条件分位数回归 + 保形校准 | 用于把覆盖率校准到名义值 |
| Leave-Waymo-Out | 留出 Waymo 做跨数据源压力测试 | 检验跨源迁移稳健性 |

## 主要结论

1. **risk/PET 不是主要杠杆**：即使用 oracle（真实）PET，条件包络也只比全局区间窄 ~3%（宽度 0.833 vs 0.857）。问题不在"PET 预测不准"。
2. **真正有效的信号是驾驶人自身早窗因果 rolling-IPV 自锚** + split-conformal 校准（"看他头 1–2 秒怎么开，反推全程 IPV 范围"）。
3. **跨数据集稳健性是决定性测试**：所有非自锚方法在 Waymo 留出上欠覆盖；唯有 causal rolling-IPV 达名义 0.90。
4. **因果落定**：离线 IPV 用全程参照看似泄漏，但用**地图车道中心线**参照重建的严格前缀因果 IPV 与离线 corr = 0.993 ⇒ 真可部署（route-conditioned）。

推荐方案：**lane-referenced causal rolling-IPV self-anchor + split-conformal**。

锁定生产数字（平衡 5,000 cases / 10,000 rows，Wilson 95% CI）：

| 场景 | 方法 | 覆盖率 [95% CI] | 平均宽度 |
|---|---|---|---:|
| 同分布 TEST | oracle PET（上限参照） | 0.889 [.874,.901] | 0.867 |
| | no-roll 运动学（兜底） | 0.896 [.882,.909] | 0.738 |
| | **causal rolling-IPV（推荐）** | **0.899 [.885,.911]** | **0.591** |
| Waymo 留出 | oracle PET | 0.860 [.850,.870] | 0.840 |
| | no-roll 运动学 | 0.857 [.847,.867] | 0.743 |
| | **causal rolling-IPV** | **0.902 [.894,.910]** | **0.628** |

即：推荐方案比 oracle PET 区间窄 **−31.8%**（TEST）、比 no-roll 窄 **−19.9%**，且是**唯一在 Waymo 留出达名义 0.90 覆盖**的方法。无已知车道的 ~26% 案例退回 no-roll 运动学 CQR（−25~30%，仍优于 PET）。

## 目录结构

- `00_entry/index.html`：报告入口（先看这个）。
- `01_results/`：可复用结果——`figures/`(4 图 png+svg)、`metrics_balanced_lock.csv`(锁定数字)、`ab_metrics.csv`(verifier A/B)、`baseline_metrics.csv`、`model_balanced_lock_cqr.joblib`(生产模型)、`final_summary.md`、图注/清单。
- `02_process/`：过程产物——`scripts/`(特征/方法/锁定/接入/绘图脚本)、`agent_reports/`(15 个 Codex agent 的简报)、`board/`(研究计划/发现/数据契约/内容包)、`INTEGRATION.md`(verifier 接线说明)。
- `TRACEABILITY.md`：入口 → 结果 → 过程的可追溯映射。

完整研究舰队归档（含所有 agent 工作区与日志）已移到
`archived/report_local_state/interhub_20260620/codex_fleet/ipv-online-interval/`。
