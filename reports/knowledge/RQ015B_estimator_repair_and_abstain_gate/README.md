# RQ015B — IPV 估计器数值修复与 verifier 弃权闸（知识层）

计划：`reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md`
状态：PROPOSED / 待独立双路复审；无计算授权；未创建 `decision.md`。

由 PI 决策 2026-07-26 从原 RQ015 拆分而来（另一半为 RQ015A）。

## 核心

- 缺陷：概率域连乘下溢 + `sum(var)==0 → 均匀权重` 兜底，使"无信息"与"中性"不可区分。
- 修法：`w = softmax(−MSE_i/(2σ²))`（与现行计算数学等价、永不下溢）。
- 下溢临界 RMS：n=5 为 1.6915 m（subnormal）/ 1.7336 m（舍入为 0）；
  n=11 为 1.1470 / 1.1752 m。
- 实现现状 `BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`：
  `src/sociality_estimation/core/reliability_logdomain.py` +
  `tests/test_rq015_reliability_logdomain.py`（36/36 通过）。
  **生产兼容层三项未交付，不得接线。**

## 部署前置

verifier 弃权闸的部署**必须**先通过 gate-pass 条件覆盖审计（计划 §7）。
RQ009 的约 0.899 覆盖只能作为历史 ungated 边际结果保留。
既有基线：冻结 M3 test fold 90% nominal 支持域内，`|y|<1e-6` 近零行有
520,826/522,219 = 99.7333% 的区间包含 0（约 0.27% 不含），
因此"每个测不出的帧都判合规"是**错误的全称表述**。

## 边界

不重训 M3；不覆盖冻结产物；新估计器不得进入 RQ014 lane。
