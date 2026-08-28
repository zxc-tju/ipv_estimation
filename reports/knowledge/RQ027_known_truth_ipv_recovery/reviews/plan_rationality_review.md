# RQ027 研究计划合理性审查

日期：2026-08-28
审查对象：`/Users/xiaocong/Downloads/IPV_recovery_agent_research_plan.md`
对象 SHA-256：`2a35861eecaff4b03fe0bb58724a8bb9a88ae673e467315f8ef1a497c4fac367`
结论：`REVISE -> GO FOR BOUNDED FEASIBILITY PILOT`

## 1. 问题与当前阶段

这项工作要解决的是审稿意见反复指出的测量缺口：在线/离线 estimator 一致、候选权重集中或同模型 synthetic 成功，都不能证明 estimator 恢复了已知参数，也不能证明集中度代表低误差。

仓库已有 WP2 同模型 pilot 和 RQ024 失败诊断，但没有真正独立生成器下的 known-truth recovery。RQ026 是 runtime-only，不能承载本问题。因此新建 RQ027 合理且必要。

## 2. 合理且应保留的设计

- 分开 interaction opportunity、IPV estimate 与 estimability；无信息时不得把零值当中性真值。
- S0/S1/S2/S3 分层，并把同模型 S0 降级为工程 sanity。
- 独立生成器、off-grid 真值、响应式 counterpart、负对照与模型失配。
- simulation run 为主统计单位；帧只服务于 onset 与时序。
- 风险与 coverage 同时报；拒绝窗口不能从分母消失。
- 明确禁止把仿真 recovery 外推为真实人类心理参数 recovery。

## 3. 原计划不能直接执行的原因

### 3.1 关键量尚未冻结

原稿没有给出 persistent `K`、accepted frame、opportunity onset、oracle `Δ/H`、false-accept 分母和 concentration 的精确计算顺序。缺少这些定义时，最终 PASS/FAIL 可被结果后口径改变。

### 3.2 规模与流程过早膨胀

最低 `3,120`、推荐 `14,040` runs，再叠加全量扰动、消融、九 worker、多轮 review 与 replication，不适合一个尚未证明独立生成器能产生可识别行为的新问题。它违反本仓库“探索/诊断一轮自查，手稿级主张才多轮复审”的速度原则。

### 3.3 split 的 cluster 数不足

若只有 `8-12` 个 scenario templates，再按约 `30%` 划 sealed test，test 只有约 `2-4` 个 template clusters；此时 scenario-clustered bootstrap 不稳定。完整 confirmatory 需先扩大模板数或重新冻结推断单位，不能直接沿用草案比例。

### 3.4 默认阈值缺少依据

`80%` half-grid、bias CI `±π/32`、sign `90%`、Spearman `0.90` 和 negative false accept `5%` 都是可讨论的目标，但目前不是历史性能、功效分析或用途容差推导出的标准。它们不能在 guard 后再被包装成事前科学标准。

### 3.5 旧 Gate A 不可复用

RQ024 已接受：旧 Gate A 的相邻严格阈值 `36/42` 违反 MAE 非升合同，且 `q_eff/k_eff/ipv_error` 几乎同构。有限样本 risk-coverage 不保证逐点单调；RQ027 应检验整体关联、固定政策门与 baseline 差，而不是再次要求所有相邻点机械单调。

### 3.6 独立性需由代码合同保证

既有 WP2 generator 调用 `Agent.solve_optimization()`，属于 S0。真正 S1 不得共享 `Agent`、SLSQP、cost 实现、candidate tracks 或 likelihood；只允许共享状态输出格式、量纲、采样率和 IPV 方向约定。

### 3.7 核心 estimator 与现行弃权门不是同一实现层

`estimate_ipv_pair()` 的 legacy 概率域权重在全部 likelihood 下溢时会回退为均匀权重并继续输出零 IPV；near-uniform / tie 的 `ABSTAIN` 来自后续 log-domain materializer，而不是 core 函数本身。因此 RQ027 需从 diagnostics 重算 log-domain MSE 权重作为 primary，并把 legacy reading 仅作 sensitivity。否则 `max(weight)>=0.20` 会与它原本对应的权重实现错配。

## 4. 计划修订

- 单开 `RQ027_known_truth_ipv_recovery`，不挂入 RQ026。
- 首轮只做 `240` 个 S1 interactive runs 与 `48` 个 S3 negative-control runs。
- 预先冻结五个真值为 `{-2,-1.5,0,1.5,2}×π/8`，避免零值 predictor 在 one-grid 容差下机械命中大多数 off-grid 点。
- 固定 `K=3`、`max(weight)>=0.20` 政策门、run-level median、opportunity 与 oracle 定义。
- 冻结 target-side 为每个 simulation run 的唯一 primary directed estimate；counterpart-side 不改变 `240/48` 分母。
- 同时报告 concentration-only false accept 与 opportunity-aware reading；不能让 opportunity gate 通过定义把负对照误报变成零。
- pilot 失败即停止，不扩规模、不同轮调门重跑。

## 5. 与现有 RQ 的关系

- RQ007：held-out 继续封存；RQ027 不读任何 RQ007 数据。
- RQ015A/B：沿用“concentration 不等于 accuracy/estimability”的术语边界，不部署 prototype gate。
- RQ024：继承其已接受的失败事实，新合同不推翻它。
- RQ026：保持 runtime-only，不共享产物或结论。
- RQ017/RQ018/RQ019/RQ021/RQ025：不重开已接受结论，不用于 recovery 训练或验收。

## 6. 当前判断

科学方向合理，原稿结构完整但过度流程化、关键判据冻结过晚。经上述收窄后，先做独立 S1+S3 feasibility pilot 是高信息量、可证伪且与现有治理兼容的下一步。

证据状态：

- `可直接支撑`：新建 RQ027 并执行 bounded pilot 的合理性。
- `可作旁证`：既有 WP2 同模型 pilot 的接口与工程健康。
- `待核验`：独立生成器下 recovery、concentration-error 关系、negative-control false accept。
- `不能证明`：真实人类心理 IPV、外部有效性、因果、生产可用性。
