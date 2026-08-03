# STATUS — track B（rq015b-repair）

state: DONE
updated_at: 2026-07-31T11:02Z
phase: B2（收尾完成）

summary: 两问均已回答。(a) **平价门过**——未触发兜底且无部分下溢的 1,526/2,300 个锚点上，
legacy 连乘与 log 域最大权重差 `3.75e-15`；log 域 `logw -= logw.max()` 使分母恒 ≥1，
**兜底不可达**（极端 RMS 10–15 m 实测 `finite=True`、权重 `[1.0, 2.66e-223, 0…]`，pytest 1 passed）。
(b) 机制拆分见下，**但合并值不得整体外推**：两个来源方向相反。
监督方已于 10:57Z 验收 B 轨科学产出；本回合只做了第 7 条指定的两件收尾，未派 agent、未重跑、未改报告数字。

## D0–D4 构成（本地可达 post-warm U∪Z，HT 分母 534,939；case-cluster bootstrap B=2000, seed=20260731, cluster=1459）

| 机制 | 合并 | waymo（占框 72.7%） | nuplan（占框 27.3%） |
|---|---:|---:|---:|
| D1 数值下溢 | **43.01%** [39.35, 46.83] | **58.73%** | **1.06%** |
| D2 当前网格与模型下真平坦 | **39.48%** [35.69, 43.08] | 30.45% | **63.56%** |
| D3 模型失配 | 0.00% | — | — |
| D4 求解器/输入失败 | 0.00% | — | — |
| OK（本就非均匀） | 17.51% | 10.81% | 35.38% |

D0（warm-up）为独立普查、不在上述分母内：nuplan 42,432 + waymo 130,624 = 173,056。
log 域改写后权重实质非均匀（"修得好的"）= **53.62%** [50.12, 57.46]。

**分叉结论：waymo 上 D1 占主导，改写能让大量零值锚点获得候选间判别信息；
nuplan 上 D1 仅 1.06%，多数零值锚点在当前网格与模型下候选间差异本就极小，改数值对其不起作用。
是否申请全量重跑预算属 PI 决策——本轮只给证据与分源边界，不给建议。**

## 五条一等边界（不是脚注）

1. **复现门未过**：存档主表（2026-06-12）逐锚点值在当前 HEAD 上未复现，`gate_a 12/40`（阈值 39/40）。
   上述比例**不得读作存档 M3 标签的成分构成**。
2. **Z 层（524,231 行）复现率 0/14**，N 层 0/12；U 层 12/14 因均匀兜底是不动点，不构成强复现证据。
3. **`D3 = 0.00%` 是冻结优先级 `D4 > D1 > D3 > D2 > OK` 的产物**，23 个超阈锚点全被 D1 先吸收，
   不得读作"不存在模型失配"；三行相同的敏感性表对本轮口径**无区分力**。
4. **CI 半宽最大 3.74 pp，未达 ≤3 pp 精度上限。**
5. **覆盖边界三项**：lyft/av2 无本地 pkl（1,019,626 锚点完全不可达）、waymo 缺 500-799 分片、
   `waymo_300-499.pkl` 截断已排除。两个分母：24.10%（精确零/后热身）与 38.65%（`|ipv|<1e-6`/dev+guard 全体，
   即既有认知"约四成"的口径）。

## 硬约束终检（全过）

- held_out **未被解析**：只判 split 不读行，non_devguard_rows_not_parsed=1,097,445
- rating/preference 字段：未读取
- 生产估计器**未接线**：`agent.py` 与 `reliability_logdomain.py` SHA 均匹配派发前基线；HEAD 仍 `e82091ce`
- 未覆盖任何冻结产物；全程**未发生回滚**（`ROLLBACK_ALERT.md` 不存在）
- 无因果措辞；未使用 estimability 与"测出/未测出 IPV"表述

## 产物

- `board/reports/B2_report.md`（结项报告，140 行）
- `board/reports/B2_leader_adjudication.md`（leader 一轮自查）
- `work/anchor_mse.csv`（2,300 行，非有限值 0）
- `board/sampling_contract_v1.md`（跑前冻结，运行时 `sample_sha_ok` 复核 2,300 行）
- 已并入 `reports/knowledge/RQ015A_ipv_estimability_labelling/known_issues_and_audit_boundary_20260730.md` **§9**

next: 无。B 轨收口。两件收尾均已完成——rollback_watchdog(pid 90240) 已停、快照目录保留；
复现门片段已作 §9 并入 RQ015A 知识库。追查存档复现失败的代码漂移**留给独立一轮**
（需工作区干净、无并发 track，方可回放历史提交）。root `main_workflow.log` 由 A 轨 leader 统一写。
