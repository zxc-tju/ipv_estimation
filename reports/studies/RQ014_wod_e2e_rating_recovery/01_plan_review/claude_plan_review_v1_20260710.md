# RQ014 Plan v1 — Claude (PI 角色) 独立评审

日期：2026-07-10。评审对象：
`reports/plans/RQ014_plan_v1_wod_e2e_rating_ipv_deviation_recovery_20260710.md` +
`RQ014_config_space_v1.yaml` + `RQ014_forensic_registry_v1.yaml`。

## 结论：PASS_WITH_FIXES

统计与治理设计显著优于 v0（本人所拟）；在冻结前必须修复 4 项阻塞问题，其中 B1/B2 直接
关系到"找回"这一首要目标能否达成。

## 已核实项

1. **Anchor 真实性 PASS**：A1–A4 的全精度值逐一命中仓库结果文件——
   `phase3_effect_summary.csv`（0.14767623020869206 / 0.03129843185743807）、
   `tenhz_sensitivity_summary.json`（0.16481712868606582 / 0.12753396435827072）。非虚构。
2. **Alias 数学 PASS**：W=1.0 且 tau≥1.0 时 future-only 窗 `[max(0,tau-1),tau]` 与
   history+future 窗 `[tau-1,tau]` 逐点恒等；16→12 的削减成立。W=2.5 在 tau<2.5 时两模式
   分歧（history+future 触及 t<0 真实历史），保留为独立 cell 正确。
3. **Registry 一致性 PASS**：YAML 12 configs × 3 readouts = 36 行可机器物化；正文
   §5–§11 的合同项均有对应字段；forensic registry 的 FT01–FT04 单因素 twin 与 §2.2 一致。
4. **两条 estimator lineage 分离**：正确吸收了 INV-ipv-code-diff 的教训（本地漂移
   `a0fee535`），LEGACY_SIGMA01_CORE 与 RQ010B_WOD_FIXED 分开登记。
5. **功效诚实性**：明确 N≈23–49 时确认功效仅 0.28–0.56，不预承诺 confirmation，
   INCONCLUSIVE_LOW_POWER 作为一等公民终态。这是 v1 最重要的品质。

## 阻塞修复项（冻结前必须完成）

### B1 恢复覆盖面缺口 — 增设第三个 registry 臂（最重要）

valid family 把 envelope 固定为 matched time-indexed path-type HV、endpoint 固定为 rolling
exceedance。但 G1 取证是 NOT_FOUND——没有任何证据把旧实验钉在这个形态上。若旧实验用的是
(a) terminal/final-IPV 偏离（非 rolling）、(b) sigma01 静态 risk×geometry×role 包络、或
(c) candidate-internal 包络，则冻结的 valid family **原则上不可能找回旧设置**，而这三者
既不是 defect（不在 forensic FT 目录）也不是 pilot（不在 FL 目录）——它们落在两个 family
的夹缝里。后果：`NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY` 将在"记忆有误"与"family 太窄"
之间不可辨——这恰是 §0 自己禁止的混淆。

**修复**：新增 `RQ014_recovery_extension_registry_v1.yaml`（第三臂，地位同 forensic：
不可晋级、不进 alpha family、只产 recovery/discovery 信号），以 V04 类比基线做单因素变体：
terminal-IPV endpoint、sigma01 静态包络（若可 ratings-blind 重建）、candidate-internal
包络、80/95 band levels。每行注明"若该行独占强负相关 ⇒ 旧设置候选，但证据等级仅为
specification-recovery"。

### B2 F05 范围过窄 — 补 FL05：reframed 阶段中间统计全索引

forensic lookup 只列了 6-29 的 4 个 pilot 目录。但 G1 之后，记忆来源最高概率的宿主是
`/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/**` 与 `results/**` 里 RQ010B 执行期产生的
**全部中间相关性表格**（phase1c/2c/2e/2f/3、schemeB、10hz 各阶段）。PI 在执行期看过这些
中间数；任何一次 rho≤-0.30 的记录行 + 其配置指纹，就是 historical fingerprint 的直接候选。

**修复**：forensic registry 增加 FL05：只读索引两目录下所有 csv/json/md 中的相关统计量
（值、N、配置字段、文件 SHA、mtime），产出 `historical_stats_index.csv`。一个 grep 级
Slurm-free 作业即可完成。

### B3 envelope gate "所需 cell" 语义歧义

§6.2 规则 8："任一 config 所需的任一 path-type×tau cell 未过 gate ⇒ 整个 config 失效"。
"所需"必须 ratings-blind 地定义为：**冻结分析 manifest 中 ≥1 个 primary-eligible 场景实际
查询的 cell 集合**，在 feature build 期间物化并写入 envelope contract。否则一个无场景使用的
稀疏 cell（如 HO × tau=4.75s，InterHub HO 样本很可能 <50 episodes）会无谓杀死全部 HMAX5
configs。

### B4 G0 闭环实操 — 立即补跑两个既备脚本

F05–F08 仍为 OPEN，而 §17 把 OPEN surface 列为 stop condition。`hpc_pass3`（上次仅 ssh
超时）与 `mac_pass3b` 均已备好、各约 1 分钟。跑完即可把 F05–F08 全部转为合法终态，且
F05/FL01–04 的输出直接喂 forensic fingerprint。无理由带着 OPEN 进 G1。

## 非阻塞意见

- N1：V01/V02（4Hz, W=1.0 → 4 intervals/5 points）估计脆弱性高，预计 fallback 率显著；
  已有健康度 gate 覆盖，但 benchmark 必须按 config 报告 fallback 率（§15 已要求）。
- N2：§8.3 功效模拟中 12 configs × 4 splits × 3 tie 档 × 20k sims 的机器成本不小，但
  deviation 向量冻结后每次 sim 只是 rank 运算，CPU 可承受；建议先跑 1k sims 冒烟校准
  bisection 收敛再放大。
- N3：§1.1 断言"RQ010B 主量接近 terminal IPV 绝对偏离"——与 worker prompt 证据一致，
  但 G2 时应对 phase3 脚本做一次事实核对并引用行号。
- N4：checksums 文件若含 html 版本，修复后需同步重生成，避免 md/html 漂移。
- N5：`rho_WS` 在 3 候选/场景下取值粗（r_s ∈ {±1, ±0.5, 0} 附近），within-scene 置换每场
  仅 6 种排列——置换分布合法但离散；置换 p 的 granularity 应在 G2 报告中说明。

## 评审裁决

- B1–B4 修复并生成 v1.1（或 v1+amendment）后，本评审升级为 PASS，G1 关卡可关闭。
- B1/B2 不修复则本评审为 FAIL：计划将系统性地无法达成其首要目标（找回），只能证伪一个
  从未被证据支持的特定形态。
- 修复不需要重做任何统计设计；valid family 的 12-config 结构、promotion、confirmation、
  specificity 合同原样保留。
