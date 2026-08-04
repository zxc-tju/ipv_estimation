# STATUS — track D（rq015d-sigma-rederive）

state: DONE
updated_at: 2026-07-31T17:12:40Z
phase: 结项（监督方 17:08Z 第 3 条已放行，收尾修订完成）

## 判定

**B3：不需要重定全局 σ。σ = 0.1 保持不动。** 监督方 2026-07-31T17:08Z 已接受。

## 编制与成本

一个 codex agent（D1, gpt-5.5, xhigh）+ 一轮 leader 自查 + 一轮收尾文字修订。
D1 用时约 7 分钟。**未出规格 v2、未加审计轮次、未派第二个 agent、未重跑。**
全轮无新实验：所有结论在既有 `anchor_mse.csv` 上完成。

## 三条结论（按重要性排序，已全部进正文）

**1. 第三类失效：400 行（17.4%）对任何 σ 免疫**（报告 §1b）
   7 个候选 MSE 逐位相同 ⇒ 改变 IPV 候选完全不改变预测轨迹。
   nuplan 内 34.78%，**waymo 0 行**；占 "约 59% 失效地板" 的 29.2%；
   σ=0.01→1.0 全程 k_eff=7.000；`ipv_log` 恒为 `0.0`（pathological constant）；
   连带 `at_grid_boundary` 假阳性 400/1,185 = 33.8%。
   σ 与 log 域改写**都无效**，全量重跑也救不回来。
   与 B 轨 "nuplan D2 63.56% / D1 1.06%" 画像结构一致（两条 track 独立收敛）。
   → 已立为具名候选 **`RQ015-NEXT-CAND-01`**，本轮不立项，是否立项由 PI 定。

**2. 不存在单一 σ 能同时缓解两端**（报告 Q2；监督方选项 2）
   两条曲线全扫描区间严格反向单调、无共同低谷：
   near% 25.00→81.96 单调升，hard% 35.57→0.17 单调降。
   合计失效地板约 **59%**，任何 σ 打不穿。σ=0.02 是**内部极小值但极浅**
   （6.56 pp 改善换硬 argmax +16.56 pp 翻倍），且合并最优对 nuplan 净损失 12.61 pp。
   ⇒ 修法是 log 域改写 + 弃权闸，不是重定 σ。

**3. 下溢兜底掩盖的是"过锐"而非"过平"**（报告 §1c）
   603 兜底行 k_eff_legacy 精确=7（貌似完全中性），但 log 域 median 仅 1.441、41.63% 是硬 argmax。
   ⇒ 存档里那批 ipv=0 中有一部分是**强偏好被抹平**的结果。
   RQ015A 只能观察到"权重近均匀"，本轮第一次给出被掩盖掉的是什么。

附：计划 §5 预设方向被否——`median(sqrt(min_mse))=0.2347` 比 0.1 **大 2.3 倍**（权重更平），
与 §5 担心的"过锐"相反；分源 nuplan 0.0941 vs waymo 1.1153 **跨 11.9 倍**。

## 收尾修订（按监督方第 3 条，纯文字/版面，无重算）

- [x] 改掉"名义最优 σ=0.02 落在扫描边界"——**不准确**，改为"内部极小值但极浅"，
      并换用监督方指出的两条更强理由（浅极小 + 分源冲突）
- [x] σ_rederived 方向提升到 §1 摘要（含 11.9 倍分源差）
- [x] 603 行兜底对照从 Step 3 提升为正文 §1c
- [x] 400 行从附录提升为正文 §1b 单独成节 + `at_grid_boundary` 假阳性写进正文
- [x] 立具名候选 `RQ015-NEXT-CAND-01` + known-issues 可复制片段（报告 §8）
- [x] 附录 §B 保留原始核查记录，加交叉引用指向 §1b

## 完整性核验（leader 独立执行）

- D1 全部关键数字经 leader 独立重算复现（k_eff 定义 1/Σw² err=0；603 行 median 1.441 / 硬 41.63%；c=50）
- 输入 `anchor_mse.csv` sha256 = `b0f6202501ea738b1ae6d49f83af1877bee85b391d5db6a44375d67b552eb114`，
  与 `min_mse_misfit_threshold.json` 冻结值一致 ⇒ **全轮输入未被改动**
- `git status --porcelain src/` 为空；`agent.py` 一字未动；未新建 estimator_version；
  未覆盖任何冻结产物；无任何 git 写操作
- 两个禁用术语全文 0 次；分源结论齐备；产物仅落在授权的 5 个路径
- RQ007：split 仅 development(1647)+guard(653)，**无 sealed/held_out 行进入统计**

## 交付物

```
board/reports/D1_sigma_report.md    主报告（正文 + leader 自查附录 + 监督方问答）315 行
board/reports/D1.log               D1 完整执行日志
work/d1_sigma_analysis.py          可复跑脚本
work/d1_sigma_stats.json           全部数字
work/sigma_sweep.csv               σ 扫描 36 行（12 σ × ALL/nuplan/waymo）
work/mode_by_min_rms_decile.csv    Step 2 十分位交叉表
```
复跑：`<local-rq009-venv>/bin/python .codex-fleet/rq015d-sigma-rederive/work/d1_sigma_analysis.py`

## 移交 PI

本轮产物**未提交**（按舰队规则由 PI 统一提交）。
待 PI 裁决的开放项：是否立项 `RQ015-NEXT-CAND-01`（nuplan 候选求解退化）。
