# STATUS — track C（rq015c-drift-forensics）

state: DONE
updated_at: 2026-07-31T17:15:00Z
phase: 结项（监督方第 2 条已放行，4 项表述调整 + 2 条具名下一步已落地）

summary:
  **结论有两句，第一句才是本轮买到的东西。**

  1. **【证伪，确定性正面结果】PI 的"本地代码漂移"假设不成立。**
     用 5edd2810 的代码回放同一批 40 个冻结锚点，与当前 HEAD **逐位相同**：
     max|Δipv| = max|Δipv_error| = max|Δweights| = **0.000e+00**，
     gate_a 通过的是**同一批 12 个锚点**。
     ⇒ agent.py 983→1244、ipv_estimation.py 313→675 加一次目录重构，
     在这条计算路径上对这 40 个锚点**严格保行为**。
     ⇒ 关掉了"重构可能悄悄改了行为"这个悬念；
     **RQ015B 基于当前代码得到的 D1/D2 机制结论不会因代码版本用错而失效。**
  2. **【存档来源未定论】verdict = LOCAL_FORENSICS_INCONCLUSIVE。**
     legacy 代码 gate_a 同为 12/40（阈值 39/40）。差异源既不在 legacy 代码、也不在 current 代码。
     按 PI 裁定不接 HPC，本地取证到此为止。

  **附带发现：**
  - 失配分**两类**：Z 类 14/14 IPV 精确一致、**仅二阶矩 Σw² 失配**
    （已实测：`ipv_error = 1 − sqrt(Σw²)`，14/14 逐位吻合；重解后一阶矩仍落回 2^-55
    ⇒ argmin 未变 ⇒ 排除轨迹生成/参考线差异）；N 类 0/12 一阶矩即变。两类应分开追。
  - **【数据完整性，独立于 RQ015，需 PI 处理】** `waymo_300-499.pkl` 于
    2026-07-31T02:42:06Z 被改写且传输不完整，至今 `pickle data was truncated`、n_events=0；
    其余 9 个 pkl mtime 为 2026-06-09（早于存档）。B1 已排除该 folder，未污染本轮结论。
    此事证明**原始输入是会被改动的** ⇒ "输入 pkl 版本"从并列第三升为**最可疑候选**。
  - lyft/av2：本地 full_datasets **无 lyft、无 av2**（已复核）；
    `5edd2810:process_argoverse.py` 读的是**预处理 CSV 而非原始 AV2 parquet**
    （决定取数代价量级，PI 决策最关键的一条）。定量体量推断**未复核**，已在报告中分级标注。

  **两条具名下一步（本轮不做，交 PI）：**
  A. 比对 sigma01 时代是否记录过输入 pkl 的 SHA256/行数/case 数与本地现有 pkl
     —— 最便宜，且**唯一可能不接 HPC 就出结论**的一条。
  B. 重新同步修复 `waymo_300-499.pkl`。

  **仓库影响：无。** 未 commit、未 checkout 主工作区、**未创建 worktree**
  （`git worktree add` 被 sandbox 拒后改用 git 对象提取，leader 与监督方各自逐 blob 验签
  与 5edd2810 一致；该替代方案 provenance 更强且对 `.git/` 零改动，故**无遗留清理项**）。
  `src/ pipelines/ configs/ scripts/` 无 diff；B 轨冻结产物 mtime 未变；禁用术语 0 命中。

  START_HERE.md **未改**：C/D/E 并发，覆写式共享根文件有互冲风险；
  待登记的已知弱点（waymo_300-499.pkl 截断）已写入 `main_workflow.log` 与报告 §11.1，
  建议 PI 在三条 track 收束后统一并入。

next:
  无。track C 收工。
  主报告：board/reports/C1_drift_report.md
          SHA-256 a173b47cb40ea46ad0f2e1aaf7f8c0e3e9e3d2576195a4ebc6a74d43cd14331f
          （§9 内列的本报告自身哈希是 §10/§11 追加前的旧值，属固有现象，已在报告末尾附注说明；
           §9 中其余 artifact 哈希仍有效）
