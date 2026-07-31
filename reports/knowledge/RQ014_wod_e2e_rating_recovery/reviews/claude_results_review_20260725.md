# RQ014 结果评审 — Claude（PI 角色）独立复核

日期：2026-07-25
评审对象：`reports/studies/RQ014_wod_e2e_rating_recovery/03_results/RQ014_final_report_20260725.html`
（R3 解盲运行 `RQ014_3_full_rating_join_and_rank_20260724T053954Z_49dcd5c0`，Slurm 1974225，DONE=PASS）
对照机器权威：`reports/plans/RQ014_recovery_lane_v3.json`、base v1 plan、v1.6 execution handoff、
`reports/knowledge/RQ014_wod_e2e_rating_recovery/README.md`、`main_workflow.log`。

## 裁定：`ACCEPT_RESULTS / REJECT_FINAL_STATUS_LABEL`

结果本身可信、盲态合规、缺陷披露诚实，**科学内容我接受**。但这份文件不能以"最终研究报告"
的身份存在：按冻结合同，存在 `recovery_compatible=true` 行意味着必须先走 D5A（G4R 干净复现）
再走 D6，才可能形成最终结论。另有两处措辞越界、一处解释缺口需在归档前修正。

---

## 一、已核验且应予肯定的部分

1. **盲态链条完整**。全流程仅一次评分接触（R3 运行时内存中），收据只含计数/哈希，
   独立 codex 只读裁决 E 为 CONFIRMED，持久产物零逐条评分值。此前 9 次 G2R 构建评分读取数为 0。
   与 lane v3 的 rating boundary 条款一致。
2. **网格 960→320 是登记变更，不是静默缩水**。lane v3 明文
   `registered_predictor_cell_count: 320`（16 特征族 × 2 horizon × 10 readout；envelope 不再作为轴、
   改为 M3 固定），经 PR #10–#14 双人复审并重新生成 FORMAL_G1_PASS。我逐字核对 JSON 后确认。
3. **恢复门是冻结的、readout 无关的、机器判定的**。报告 §3.2 的七项判据与 lane v3
   `recovery_compatible_marker.requirements` 十项逐条对应无遗漏、无放宽。唯一通过行不是人眼挑选。
4. **自限诚实**。开篇即写明主 readout（NEX）零通过、唯一通过行落在次要 readout 上，
   并同时拒绝"迁移已确认"与"无迁移"两种越界表述。这是本报告最值得肯定之处。
5. **R10L 缺陷披露方式正确**。缺陷经两轮独立裁决定性为"实现宽于冻结方法"，
   由 rating-independent 的 attrition 结构发现，探针 rating-free、stdlib-only 且交叉校验可复算桶精确命中；
   因此"不修复不重跑"的 PI 决定不构成结果依赖的方法修改。这一判断我同意。

## 二、阻断项（归档前必须处理）

### B1 状态标签越权（治理）

日志中不存在任何 G4R/clean replay 记录，`knowledge/…/decision.md` 正确地不存在——即研究**尚未**
走完 D5A/D6。但文件名与标题自称"最终研究报告"，与治理状态直接冲突。
另外该文件位于 `03_results/`，而 v1.6 §14 要求每个执行 wave 建 `RQ014_<n>_<op>_<UTC>_<sha8>/`
执行报告目录（前两波均已遵守）。

要求：改名/改标题为 R3 恢复筛查有界报告，显著标注
`PENDING_D5A_G4R_REPLAY / NOT_A_FINAL_CLAIM`，并迁入与前两波一致的 RUN_ID 执行目录
（`03_results/` 可保留为指针页）。

### B2 标题把"规格恢复"表述为"外部效度/跨数据集迁移"（措辞）

lane v3 `claim_boundary` 冻结为：`historical specification recovery on the same dataset`、
`independent_replication: not provided`、`causal_claim: forbidden`、`p_values_gate_recovery: false`。
正文 §1 已正确写明"已知结果的规格恢复筛查、非前瞻性假设检验"，但标题与副标题
（"外部效度 / 跨数据集迁移的单次受控解盲评估"）会被独立引用，且与 RQ010B 已冻结的
`RQ010B-KC-M3NOTRANSFER`（M3 在 WOD-E2E 数值支持 ≤15%，不得作为 valid primary envelope）
表面冲突。

要求：标题回归恢复框架；并在方法节明确交代 v3 中实际使用的 M3 artifact 与包络形态，
以及它与 RQ010B M3-不迁移结论的关系（v3 已把 envelope 移出轴、固定 M3，需说明这是否规避了
当初的 in-support 问题，还是把该限制转化为了 support 衰减）。这一条不澄清，任何读者都会
在两份冻结结论之间产生矛盾感。

### B3 唯一阳性缺少零假设标定（解释）

"960 行中恰 1 行通过"这句话在缺少门的 null 通过率时不可解释：多条件门在 960 行、
中位 n=23 的稀疏格局下的偶然通过概率未知，可能远大于也可能远小于 1。
恢复框架不以 p 值为门（我同意），但**可解释性**要求知道这个数。

要求（择一）：
- 优先：申请一个新的受限 operation，对已访问的评分做 within-scene 置换 B 次（如 B=999），
  在完全相同的冻结门下重跑，报告"≥1 行通过"的经验概率与逐行通过率；这不引入新假设检验，
  只标定既有门。
- 退而求其次：在报告中显式写明"该单行的偶然通过概率未标定"，并禁止在任何摘要中使用
  "仅 1/960 通过"暗示稀有性。

## 三、重要意见（非阻断，须入档）

- **C1 方向证据只覆盖半个网格。** 探针只证明 R10L 无格能过 n≥40 的门，**未**证明 R10L 不会改变
  分布中心；修复后 120 格有 ≥5 场景、80 格有 ≥20 场景，这些本可提供分布证据。
  §5"方向一致性"已 scoped 到 R04N，但应补一句显式边界："另一半注册网格的分布贡献未知"。
- **C2 recipe 层面的收敛是被低估的正面证据。** 唯一通过格
  `CH-W25-H20-NMD_MEAN` 与最强主 readout 格 `CH-W25-H20-NEX_MEAN`（r=−0.351，n=37）
  是同一 recipe，仅 readout 不同。这比"1/960"更能说明信号定位在特定 recipe 上，
  建议在报告中升格为独立证据行，而不仅作为功效不足的注脚。
- **C3 NEX/NMD 的信息量差异应作为方法学发现记录。** NEX 在半数场景为 0 → 秩并列 → informative
  场景更少，这是可事前诊断的结构性问题。报告"重新审视主 readout"的建议方向正确，
  但必须写死：只能前瞻预登记，禁止事后换主终点。
- **C4 衰减链条应成表。** 479 → 476 评分键 → path-type 映射 254 → 每格中位 23 informative。
  最大单一衰减项是 path-type 映射覆盖率（254/476），应作为后续工作的首要攻坚对象，
  并在报告中以完整 attrition 表呈现。

## 四、对原始"找回"目标的 PI 判定

- 记忆中的结果是"方向为负的强相关"。唯一通过行 r=−0.384、n=42，在 5 折 / 全部 LOO / 4 簇 LOCO
  下方向不翻转，属**中等偏强且稳健**，构成一个合格的 `SPEC_RECOVERY_CANDIDATE`。
- 但因 G0 历史取证被 PI 豁免（2026-07-11 决定），**历史指纹不可得**：即使 G4R 复现通过，
  最强表述也只能是"在同一数据集上计算复现了一个方向正确、稳健的负相关配置"，
  **不能**声称"这就是当年那个结果"。当初豁免取证换取开工速度的代价，在此正式兑现，
  应在最终结论中作为永久残余风险声明随行。
- 主 readout 零通过意味着：若严格按 base v1 的主终点定义，本次为**未恢复**；
  若按 lane v3 的 readout 无关门，则为**单格恢复候选**。两种表述都要写进结论，
  不得只取其一。

## 五、建议的下一步顺序

1. 修 B1/B2（文件状态与标题），补 C1–C4 的措辞与表格 —— 纯文档工作，无需授权。
2. D5A：授权 G4R 干净复现（唯一 rank-1 recipe，全新 agent、不得复用原实现函数）。
3. 与 2 同批申请 B3 的门置换标定 operation（同一次评分数据，仅置换，不新增网格）。
4. D6 结论审查：接受则写 `knowledge/…/decision.md`，措辞按 §四 的双表述 + 残余风险；
   同时把 R10L 缺陷与 path-type 衰减作为后续 RQ 的输入登记。
