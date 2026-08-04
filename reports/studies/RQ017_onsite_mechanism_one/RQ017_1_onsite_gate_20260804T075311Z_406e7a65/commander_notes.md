# RQ017 监督方裁定与预注册记录

追加式，不覆盖历史。每条带 UTC 时间戳。

---

## 2026-08-04 — 双盲复审裁定

对 v1 任务书（`RQ017_M1_kickoff.md`，**从未执行**）做了一轮独立双盲复审：
A 号从方案入手逐条审；B 号两阶段，阶段一禁止打开方案、先从代码与数据独立推导，
阶段二才对照。两份 prompt 的收尾问答逐字节相同（各 1,172 字节，
sha256 `b2c011d222745fb8d5997ef9…`）。

**致盲核验**：在 B 号 240 万字符的日志里检索"同时出现读取动作与禁读路径的命令行"，
零命中；其报告自述"阶段一到此冻结；阶段二若改变判断，只在下一节说明，不修改本节"。
该检查能覆盖 shell 读取，不能覆盖 agent 内部工具读取——强度以此为限，不作更强声明。

**两人独立收敛于同一条主要意见**（这是采纳的主要依据）：

- **Q7（只能提一条改动）**：两人都提「把 OnSite 输入/测量合同提到 canary 之前，
  写成会 FAIL 的断言」。
- **Q2（哪一步错了不会自己暴露）**：两人都指向输入适配层——行位置窗口、
  ego/counterpart 方向、参考线。接错仍会产出七个有限 MSE、权重和为 1、
  `max_w_log`/`k_eff_log` 在合法范围、门判据可复算、`ipv_log` 恒等式成立，
  **v1 的 8 条自查全会通过**，但解的是错的行/错的车/错的时刻。
- **Q8**：两人都判 `GO_WITH_CHANGES`。

**监督方独立复核了四条吃重的断言，全部属实**：

1. anchor 构造键与 RQ016C dry-run 的 `product_row_key` 交集 67,861/67,861 = 100%，
   且 dry-run 确实带该列 → 一对一合同可直接查（A 号提出）
2. 窗口是行位置语义：`valid_anchor_positions()` 按 `pos` 取窗
   （`build_onsite_m3_anchors_hpc.py:836-847`）；`history_row_count` 实测
   10→66,289，另有 **1,572 行为 4..9**（B 号提出）
3. K2 `validate_outputs()` 硬编码 4,981,984 / 8,994,736 / 14,473,982
   （`k2_fullcorpus_materializer.py:1327-1338`），整体复用会对 OnSite 产生假 blocker（B 号提出）
4. `prepared_reference()` 在去重后点数 < 2 时 raise
   （`build_onsite_m3_anchors_hpc.py:705-710`），可作 fail-closed 用例（B 号提出）

A 号另对 v1 引用的 6 个既有数字全部抽查，全部吻合。

**据此产出 v2**（`RQ017_M1_kickoff_v2.md`）：新增第 3 节测量合同 preflight（7 条会 FAIL
的断言，执行顺序钉死为 preflight → canary → 全量）；修正 v1「照抄求解链路」这一会引起
误读的措辞，改为正面清单加否定清单（不得调 `solve_anchor_task`、不得整体复用
`validate_outputs`、不得动 `ALLOWED_SPLITS`）；负对照改为必须注入合成 sentinel；
护栏断言升为脚本 blocker；canary 增加 7 行真实坐标异常行、参考线 fail-closed 路径、
写后读回的 null scalar 模式；自查增加 `K==7` 与网格 ID 单独断言、不覆盖检查。
**v1 原文保留，不删不改。**

---

## 2026-08-04T06:22:47Z — 【预注册】结果预测（在 v2 派出之前写下，供事后证伪）

**方法**：OnSite 有 2,974 行带旧估计器的 `q_eff`
（`q_eff = 1/((1-ipv_error)^2 * K)`；≈0.143 表示旧权重高度集中，1.0 表示均匀）。
在 InterHub 上旧 `q_eff` 与新机制一通过率有强关系，把该映射按 OnSite 的 `q_eff`
分布重加权外推。**该方法回代 InterHub 自身得 70.3%，与实际 70.30% 吻合。**

| q_eff 区间 | InterHub 占比 | OnSite 占比 | 该区间 OK 率 |
|---|---:|---:|---:|
| [0.00,0.20) | 9.33% | 10.46% | 98.17% |
| [0.20,0.30) | 4.05% | 9.11% | 95.05% |
| [0.30,0.50) | 7.16% | 21.32% | 94.96% |
| [0.50,0.80) | 15.14% | 21.15% | 92.68% |
| [0.80,1.01) | 64.32% | 37.96% | 56.69% |

**预注册预测**：

```
机制一通过率        ≈ 80%（合理区间 65%–85%）
过机制一行数        ≈ 54,500 / 67,861
再过 RQ016C 支持门  ≈ 17,600（合理区间 11,000–18,000，占 67,861 的 16%–26%）
```

**预测的三条弱点（先说在前）**：

1. 那 2,974 行只占 67,861 的 4.4%，且来自 `max_anchors_per_unit=1` 年代——
   被选中的锚点可能系统性更有交互，**会让预测偏高**。
2. 旧 `q_eff` 在旧估计器口径下算得；新一轮用冻结配置，参考线与窗口若有差异，映射会平移。
3. 支持门 32.3249% 是在全部 67,861 行上测的，不是过门子集；两道门若正相关则最终数更高。

**判为异常、需监督方出手的阈值（预先设定）**：

- OK 率 **> 90% 或 < 50%** → 输入适配层多半接错
- `NO_IPV_EFFECT` 显著高于 InterHub 的 0.4007% → 怀疑观测轨迹作参考线导致前向目标退化。
  **这是「参考线 fallback」这条 PI 裁定的主要风险点，重点盯**
- `SOLVER_FAILURE` > 5%（InterHub 为 0.0388%）→ 输入构造有系统问题
- 键一对一不成立 → 立即停

**预期的正常中断**：最可能卡在 C2（输出粒度）——anchor 表是一行一锚点，而 K2 schema 带
`measurement_role`，是否需要四角色展开是真开放问题。v2 已明令此时停下上报、不得自行决定。
**该中断属预期内，不是故障。**

---

## 2026-08-04T06:48:35Z — 更正：OnSite 输入已在 HPC 上，且与本地逐位相同

监督方此前向 PI 报「OnSite 数据不在 HPC 上」，**该判断错误**。原因是只查了
`sociality_estimation/work_dirs/`（其下确实只有 INFRA 与 RQ014），而 OnSite 的工作目录
在 ZXC 顶层、位于 `sociality_estimation` 项目树之外。

**实测**：`/share/home/u25310231/ZXC/rq012b_onsite_ipv_20260627T202508/outputs/onsite_anchors_multi/`
已存在两个输入，与本地 `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/`
的同名 CSV **字节数与 sha256 均完全相同**：

```
onsite_m3_av_anchors_multi_allvalid.csv   71,137,488 bytes  sha256 4ff857c80d84f5e8aae1cb1bbf4ef0d1…
onsite_ipv_timeseries_multi_allvalid.csv  52,088,770 bytes  sha256 e49c226bafb950125a69f8b5dc90df02…
```

**对 v3 的影响**（复审结束后一并修订，**现在不改 v3，避免复审读到移动的靶子**）：

1. §4.2「上传约 30 MB」可简化——输入已在 HPC，只需核验 sha256 后引用，不必上传。
2. 但**必须先解决一个口径问题**：HPC 上是 **CSV**，本地测量（67,861 行、与 RQ016C dry-run
   的键一对一、`history_row_count` 分布、7 行坐标异常）全部基于本地 **parquet**
   （`onsite_m3_av_anchors_multi_allvalid.parquet`，16.5 MB）。CSV 与 parquet 同源但不同格式，
   **执行方必须断言由 CSV 读入后的行数与键集合与 parquet 一致**，否则 C1 的一对一合同可能悄悄失效。
3. 该 HPC 目录在 `sociality_estimation` 项目树之外，属只读引用；本轮产物仍只写
   `work_dirs/RQ017/<run_id>/`。

**留给复审的观察点**：本轮两位复审的 Q1 都要求给出 HPC 上的既有先例与绝对路径。
若他们能独立发现这个目录并指出 CSV/parquet 的口径问题，说明复审确实在独立核查；
若都没发现，说明 Q1 的问法还不够指向性。**监督方不提前告知他们。**
