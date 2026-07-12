# RQ014 Plan v1.2 Amendment — 响应 codex 复审（BLOCKED_PENDING_MAJOR_REVISION）

状态：`DRAFT_FOR_INDEPENDENT_REVIEW`
日期：2026-07-11（仍在任何评分 join / 新结果之前）
基底：v1 计划（不动）；本文取代 v1.1 amendment（v1.1 保留为历史记录）。
响应评审：`reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/codex_plan_review_v1p1_20260711.md`
（全部 8 项阻断 + 7 项 major 逐条接受，无抗辩项）。

配套文件（生效后取代 v1p1 同名 registry）：
`RQ014_config_space_v1p2.yaml`、`RQ014_forensic_registry_v1p2.yaml`、
`RQ014_recovery_extension_registry_v1p2.yaml`、
`prompts/RQ014_forensics_hpc_fl05_indexer_v1p2.sh`、
`RQ014_plan_v1p2_checksums_20260711.sha256`。

未修改的 v1 / v1.1 条款照旧。当前项目状态维持
`BLOCKED_PENDING_MAJOR_REVISION / NO_COMPUTE_AUTHORIZED`，直至本 amendment 通过复审且 G0 关闭。

---

## R1（评审 B1）FL05 重写为 fail-closed 结构化索引器

旧 grep 脚本废止（保留文件仅作历史）。新实现
`prompts/RQ014_forensics_hpc_fl05_indexer_v1p2.sh`：

- canonical 产物 = STDOUT 上的**未截断** CSV，schema 与 registry 完全一致（10 列：
  statistic_name, value, n, unit, config_fields_as_recorded, source_file,
  source_file_sha256, mtime, parse_status, raw_locator）；审计摘要走 STDERR；
- fail-closed：`set -euo pipefail`；任一输入根目录缺失 → 非零退出并打印 FATAL（缺目录
  不可能再伪装成阴性扫描）；CSV 解析崩溃 → 整体失败；非空树提取 0 行 → 失败；
- 按格式解析：CSV 表头列匹配、JSON 递归遍历（含兄弟键提取 n 与 config 上下文）、MD 按行；
  无法解析的候选行以 `parse_status=UNPARSED_CANDIDATE` **入表可见**，不丢弃；
- 每个源文件登记 SHA-256 与 mtime；statistic 行引用其源文件哈希；
- 预检树体积：冻结登录节点预算 200 MB，超出即中止并要求转 `zxc-` CPU sbatch；
- 脚本本身与产物 CSV/audit log 均纳入 v1.2 checksums（R8）。

F10 关闭条件：索引器成功运行 + `historical_stats_index.csv` 与 audit log 归档 + 哈希登记。

## R2（评审 B2）扩展臂重写为可执行 verifier 合同；X06 移出

`RQ014_recovery_extension_registry_v1p2.yaml` 要点：

- **X01–X05 每格为完整合同**（不再是单字段差量）：X02 静态包络登记为多字段 bundle，
  含状态变量（risk_bin×geometry，role 边缘化）、WOD 特征映射表、tau 规则（time-pooled，
  horizon 入 envelope ID）、分位数、权重、尺度兼容性声明（legacy full-window IPV 与
  rolling 估计尺度差异必须记录为 caveat）、builder 哈希与盲支持门（状态格覆盖率 ≥0.70
  否则该格 `INELIGIBLE`）；可用性条件不满足 → `INACCESSIBLE`；
- X03 登记为 envelope+endpoint 联合 bundle：无 L/M/U，自带 endpoint 公式
  `D_sc = trapz(|v_c(tau) − median3(tau)|)/duration`，明示主 exceedance 公式不适用；
- X04/X05 冻结 `q10/q50/q90`、`q025/q50/q975`，新 envelope ID（…-B80/…-B95），复用 v1
  human sample/权重/tau 合同与全部 half-width 门；
- **X06（pooled Spearman）移出扩展臂**，转入 forensic registry 作 FD01 pooled 诊断
  （scene-block permutation，保留场景间结构；report-only）；
- 家族推断只覆盖同质的 X01–X05（同一 `rho_WS`）：同步 within-scene permutation
  （master seed，同一置换抽样贯穿五格）、studentized lower-tail maxT、
  `p=(1+#(minT_b<=minT_obs))/(B+1)`、B=9,999；
- 冻结收缩政策：不可用格按登记原因收缩家族，可用格 <3 时不做家族 maxT、仅出单格
  descriptive 行；
- "独占"旗标显式定义：`rho_WS(X_k) ≤ −0.30` 且同场景集上 V04 clean 基线
  `rho_WS > −0.10`，且臂内 maxT p<0.05 → `SPEC_RECOVERY_CANDIDATE`；
- alpha 措辞更正（评审 major 7）：扩展臂**拥有自己的探索性 family**，与
  scientific/forensic 不共享 alpha。

## R3（评审 B3）required-cell 的非循环定义

`primary-eligible` 一词废止。新定义（入 valid registry）：

> `pre_envelope_trajectory_eligible`：恰好三条完整候选；冻结 path type 已指派；
> 三候选在 scheduled tau grid 上 solver-finite IPV 覆盖 ≥0.80 且内部缺口 ≤2·dt；
> 共享 counterpart common support 成立；**未查询任何 envelope、未读取任何评分**。

required cells = 此类场景 scheduled `(path_type, tau)` 的并集；物化为
`required_envelope_cells.csv` + SHA 后**不可变**；此后 envelope gate 才判定 config 资格。
非 required cell 失败仅记录。

## R4（评审 B4）Tier C 拆分：安慰剂优越性 vs 不变性诊断

- **安慰剂优越性 family（必须赢，family=2）**：`shuffled_ipv`、`counterpart_swap`
  （二者真正破坏候选对应关系）。统计量 `A_m=S_IPV−S_placebo`，paired scene bootstrap
  studentized maxT one-sided lower bounds，FWER 0.025，两个 lower bound >0 ⇒ PASS；
- **不变性诊断（不设"必须赢"）**：`sign_flip`、`role_flip`。先做 ratings-blind 非别名
  检查（翻转后 deviation 向量与基线 rank 相关 ≥0.999 或最大绝对差 <1e-9 ⇒ 记
  `STRUCTURAL_INVARIANCE`，属包络对称性的正常性质，PASS-中性）；若非别名，做冻结
  等价边界检验：`|S_flip−S_IPV| ≤ 0.10` ⇒ `INVARIANT`；`S_flip−S_IPV` 的 one-sided
  lower bound >0（翻转反而更强）⇒ `DIRECTION_ARTIFACT_SUSPECT`，升级 forensic 并给
  association 结论加注 `CONSTRUCTION_SUSPECT`；其余记 `ASYMMETRY_OBSERVED`（信息性）；
- **universal common-scene manifest**：主 endpoint 与两个安慰剂均完整的场景 master-ID
  清单在检验前冻结；全部 bootstrap 以 master-ID 同步抽样；诊断格支持不足时按登记
  原因收缩，不影响安慰剂 family；
- role_flip 的 role envelope 未过盲门 ⇒ 该诊断 `UNAVAILABLE`（记录，不封顶标签）。

## R5（评审 B5）Tier P/C 校准换临界零假设 + 统一 alpha

- **全案统一 alpha 合同**：所有 one-sided 检验名义 alpha=0.025（97.5% 单侧下界）；
  G2 empirical size/FWER 容忍 ≤0.03（名义值 + 蒙特卡洛裕度，写死不再含糊）；
- **Tier P size 校准在 NI 临界上**：模拟 `S_IPV−S_K = −delta_NI`，且 `S_K` 取
  {0.10, 0.20, 0.30} 网格（覆盖 S_K>0 的真实情形），叠加实测 tie/missingness 结构；
  empirical size ≤0.03 方可用；
- **Tier C 校准补等强边界**：`S_IPV=S_placebo>0`（取 0.10/0.20/0.30）下的 FWER ≤0.03，
  不再只用全零 global null；
- 全零 global null 保留为附加 sanity 场景，不作为唯一校准。

## R6（评审 B6）Tier P 比较器、边际、标签、功效对齐

- 比较器更名：`frozen_rq010b_kinematics_composite`（单一冻结标量；精确公式 + 代码
  SHA 必须在 G2 从 RQ010B 工件恢复，恢复不了 ⇒ Tier P `UNAVAILABLE`，禁止临时公式）；
- **标签收窄**：PASS 语义 = "单指标不劣于冻结 RQ010B 运动学合成分数"；标签改为
  `PARSIMONY_VS_FROZEN_KINEMATICS_COMPOSITE`；"顶得上七特征组合 / 单指标表达综合
  倾向"类句式**不得**由本检验支持（需另行建立交叉拟合多变量比较器后才可讨论，
  本轮不做）；
- **margin 重定并给依据**：`delta_NI = 0.08` = RQ010B 已发表最弱物理特征效应
  （ρ≈0.16）的一半；即 IPV 至多允许损失"最弱有效物理信号的一半"仍称不劣。
  0.10 旧值废止（其依据是 discovery 门槛，与科学可接受损失无关，评审判定正确）；
- **NI 功效前置**：G2P 增加 Tier P 专项功效项——在 `S_IPV=S_K`（等强）与实测结构下，
  NI 检验通过概率 ≥0.80；达不到 ⇒ Tier P 预先登记 `UNDERPOWERED_UNAVAILABLE`，
  标签梯子最高停在 `ASSOCIATION_ROBUST`（避免用弱检验发强标签）。

## R7（评审 B7 + major 1–3）执行规则补全

- `sequence_contract.local_window_estimator_rule` 增加
  `local_span_trailing_only_s: min(W, tau)`，并显式注明 W=1.0 时 trailing_only 即
  future-only/history+future 的规范别名（V01/V02/V07/V08 由此唯一可产）；
- **层级执行顺序入合同**：Tier C → P → I 严格顺序，前层未 PASS 后层不运行；
- Tier I 标签改为 `INCREMENTAL_BEYOND_FROZEN_KINEMATICS_COMPOSITE`；
- FT01 字段精确化：`estimator_internal_integration_dt_s: 0.10`（输入轨迹 grid dt=0.25
  不变——缺陷在估计器内部积分步长，不在数据重采样）；
- FT04 改登记为**两字段历史 bundle**（counterpart_source=single_frame_detection +
  extrapolation=constant_velocity_allowed），并注明：该 bundle 即 6-29 pilot 的历史
  通路，一因素纯度在此格显式豁免、豁免理由入档。

## R8（评审 major 4–6）范围与索引卫生

- checksums v1.2 覆盖：v1 计划、本 amendment、三个 v1p2 registry、FL05 索引器脚本、
  三个 pass3/pass3b 取证脚本；FL05 运行产物（CSV+audit log）在运行后追加登记；
- START_HERE 与 STUDIES.md 增记："RQ014 v1.2 candidate amendment 起草，v1.1 复审
  BLOCKED，无计算授权"；
- workflow log 已于 2026-07-11 追加 superseding 状态条目（不改写历史行）。

---

## 生效条件（不变 + 收紧）

1. 本 amendment + 三个 v1p2 registry + FL05 索引器通过新一轮独立复审；
2. G0 全部 surface 达合法终态（F05–F08 用 pass3/pass3b；F10 用新索引器）；
3. checksums 登记完毕。

在此之前：`G0=OPEN`，`G1=RE-REVIEW_BLOCKED`，`execution_authorized=false`。
