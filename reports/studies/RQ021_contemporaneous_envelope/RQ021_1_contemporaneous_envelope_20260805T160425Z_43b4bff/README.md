# RQ021-E1 执行记录：人类参照区间改为同期目标量

Run ID: `RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff`
Repo HEAD at run time: `43b4bff`
执行 agent: codex `gpt-5.6-sol`，reasoning effort `xhigh`，本机运行，约 38 分钟
Session: `019fd28a-e849-7611-956d-fcbb007c6346`

## 这一轮解决了什么

在线验证的第二道弃权机制（机制二）把自动驾驶车的 IPV 与「人类参照区间（envelope）」比较。
RQ016C-H2 建的 envelope 的目标量是**锚点之后** `[t+3, t+6]` 的 IPV（列 `target_ipv_future`），
而监控器在线时实际算出并拿去比较的是**锚点当下** `[t-9, t]` 的 IPV（列 `ipv_log`）。
两者在纯人-人 test fold 486,660 行上相关仅 r = 0.3488，中位绝对差 0.3724——不是同一个量。

PI 于 2026-08-05 裁定 envelope 不做预测性，改用同期目标量。本轮执行该裁定。

## 唯一的规格改动

目标列 `target_ipv_future` → `ipv_log`。**行集不变**：实测 K2 台账全部 8,994,736 行中
`status == "OK"` 6,405,292 行、`ipv_log` 非空 6,405,292 行、两种不匹配各 0 行，即
`ipv_log` 有定义 ⟺ 机制一通过；而 RQ016C-H2 参照池本已筛 `status == "OK"`。

特征（22 数值 + 4 类别）、fold、支持门、alpha 层、split-conformal 流程全部沿用 H2。

## 主要结果

- **循环性诊断未触发事前阈值**：D1 宽度比 0.795（停止线 < 0.25）、D2 out-of-fold R² 0.209
  （停止线 ≥ 0.60）。对照 `target_ipv_future` 的 D1 为 0.590、D2 为 0.220——
  context 对同期 IPV 的解释力**更低**，不是更高。
- **新 envelope 更宽**：人类 test fold 90% 层 mean width 1.242394 → 1.865128；coverage
  0.898272 → 0.902798。
- **OnSite 分组重划**（α=90，两门交集 14,099 帧）：下侧/区间内/上侧由 1,998/9,401/2,700
  变为 519/12,711/869；越界合计由 33.32% 降至 9.84%，与人类自身在新 envelope 下的
  9.720%（下侧 4.798% + 上侧 4.922%）基本一致。
- **支持门不变**（与目标列无关）：OnSite 支持门通过 21,936/67,861、两门交集 14,099/67,861，
  新旧精确相同。RQ017 的三条主张不受影响。

## 目录内容

| 路径 | 内容 |
|---|---|
| `RQ021_1_contemporaneous_envelope.md` | 执行方报告 |
| `RQ021_kickoff_v1.md` | 监督方任务书（含事前阈值） |
| `key_numbers.json` | 全部机器数字 |
| `run_rq021_e1_contemporaneous_envelope.py` | 主脚本（由 H2 脚本改最小必要处而来） |
| `score_external_rows.py` | 外部行打分脚本 |
| `envelope_model/` | 持久化产物的元数据部分 |
| `onsite_scoring_dryrun_summary.json` | OnSite 全量 67,861 行打分汇总 |
| `rq018_rerun/`、`rq019_rerun/` | RQ018/RQ019 在新 envelope 上的完整重跑 |
| `rq018_rq019_comparison.json`、`build_rq021_downstream_comparison.py` | 新旧对照 |
| `rq018_old_extended_verification.{py,json}` | 旧口径缺失 CI 的补算（TTC<1、<1.5） |

## 未随本目录落盘的大文件

以下留在 `.codex-fleet/rq021-contemporaneous-envelope/work/E1/`（与 RQ016C-H2 的处理一致）：

- `envelope_model/rq016c_h2_envelope.pkl`（172 MB，条件分位数模型 + 编码器 + 支持门 kNN 树）
- `onsite_scoring_dryrun.parquet`（2.8 MB，OnSite 67,861 行逐行区间）
- `selftest_*.parquet`

**已知命名瑕疵**：pkl 文件名仍为 `rq016c_h2_envelope.pkl`（脚本由 H2 改写而来时未改文件名）。
`manifest.json` 的 `version` 字段已正确写为 `RQ021-E1-contemporaneous-human-only-envelope-v1`，
以该字段为准。

## 监督方复核记录（未采信执行方报告）

监督方独立复算的项目与结果：

- 从两份 `onsite_scoring_dryrun.parquet` 与 `data/derived/rq017_onsite_gate/l1_v1` 直接重建分组，
  新旧三组计数逐项复现（旧 1,998/9,401/2,700，新 519/12,711/869）；
  支持门 21,936/67,861、两门交集 14,099/67,861 新旧精确相同。
- 位置对齐核验：打分表与锚点表逐行 `geometry_path_category`/`priority_role` 完全一致，
  新旧打分表 `product_row_key` 顺序完全一致。
- case/team 计数（监督方补算，执行方未做）：下侧 175 → 120 case，上侧 182 → 129 case，
  区间内 223 → 229 case；总 case 数 231、team 数 19 不变。
- RQ018-KC-C3 的三项数字（监督方补算，执行方未做）：见
  `reports/knowledge/RQ018_abnormal_ipv_degradation/decision.md`。
- 两份重跑脚本 diff：RQ018 只改 3 行（1 处输入路径 + 2 处输出目录）；
  RQ019 改 5 行（4 处路径 + 1 处 `expected_90` 输入计数断言）。执行方已主动申报后者。
- 执行方一处归因不准确：它用旧 pkl 重算 D3 得 90% mean width 1.238468，而 RQ016C 冻结报告为
  1.242394（差 0.32%），而 conformal 半径逐位相同。执行方将差异归因为「口径不同」，
  实为重新预测的浮点精度差异。不影响任何判断。

## 状态

`WAITING_ON_COMMANDER` → PI 于 2026-08-05 裁定：采用新 envelope；
RQ018/RQ019 主张按帧层面口径重写（见各自 `decision.md`）。
