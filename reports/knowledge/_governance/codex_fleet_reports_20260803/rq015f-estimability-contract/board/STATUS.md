# STATUS — track F（rq015f-estimability-contract）

state: DONE
updated_at: 2026-08-01T00:44:00Z
leader_pid: 61918
phase: 监督方第 5 条已放行；第 2/3/4/5/6/7 条修订完成；第 6 条 track G 已启动。结项。

## 编制

一个 codex agent（F1, pid 64166，约 26 分钟）+ leader 一轮自查。
无第二轮、无盲审、无规格 v2、无授权闸门。符合速度原则。

## 交付物

- `board/reports/F1_contract_recheck.md` —— F1 五问报告
- `board/reports/F1_leader_adjudication.md` —— **leader 裁定书（先读这份）**
  - 正文：独立复核 + C3 机制 + C1 口径错配
  - **附录 A**：回应监督方第 3、4 条（hw4/hw10、−0.13 不可复现、方向表更正）
  - **附录 B**：按第 5 条完成的修订（水平差机制、披露句、共线警告、Q5 分母表、版本口径、下一轮候选、预判留档）
- `work/f1_results.json` + 23 个中间件；脚本 `work/run_f1_contract_recheck.py`

## 三行结论

| 主张 | 判定 | 依据 |
|---|---|---|
| RQ007-KC-C1 | **需重述（基于 raw pooled 代理量）** | 扣除下溢 gap **加深** 4.9%(dev)/6.1%(guard)，方向未翻；headline −0.132 的 Phase-5 邻近性匹配口径**本轮未重建**，−0.1497/−0.17108 标注为另一口径的独立估计，**不是对 −0.13 的修正** |
| RQ007-KC-C2 | **NOT ASSESSED（本轮未取证）** | 未重算 \|dθ\| / settling，不得计入"扛得住" |
| RQ007-KC-C3 | **扛得住 + 必须补一句披露** | mean\|Δ\| 0.2647→0.2919（不塌反升 10.3%）；22% 不动是 strict-sign 规则的结构性免疫，代价是静默丢弃 628/13,214 = 4.75% |

## 本轮四条实质发现

1. **strict-sign 规则 ≡ 一个下溢过滤器**（最有价值）。663 个零均值 case-agent 被排除在
   C3 分母外，其中 **628 个（94.7%）是 active 帧全为下溢**。C3 的稳健不是偶然，
   是规则恰好挡住了污染；代价是 4.75% 被静默丢弃，原文未声明。
2. **RQ007 的因变量就是被下溢直接钉死的那个量**。`c1/c2 ≡ ipv_key_agent_N_error`
   （200 帧×2 全部 ≤1e-12），均匀兜底把它精确固定在上界 `0.6220355269907728`。
   定义层 `ipv_error = 1 − 1/√k_eff`（残差 2.29e-16）⇒ 与 RQ015A 的 q_eff 同一个量，
   **两个 RQ 的叙事可合并**（但须标注 hw10 vs hw4）。
3. **Q5 分母修正是对 PI 论点的直接量化证据**：近均匀 conf0/conf1 =
   13.93%/9.84%(strict)、37.56%/11.25%(ε=.01)、47.51%/13.15%(ε=.05)。
   代价：机会窗只占 ATTEMPTED 的 **4.00%**。
4. **conflict=1 侧的非下溢基线本身低得多**（0.2448 vs 0.4018，距上界 1.71×）——
   这个水平差本身就是 C1 的一个更强证据，且不依赖下溢。

## 合规（监督方已独立复核通过）

`held_out_parsed_rows = 0`，**跨来源**佐证：F1 的 `dropped_rows_never_parsed = 1,097,445`
与 `warmup = 107,544` 分别等于 RQ007 自身 `summary_sensitivity_counts.json` 的独立计数。
未重解轨迹；未改任何冻结产物；无 git 写操作；禁用词零命中；F1 只写本轨目录。

## 下一轮具名候选（本轮均不做，PI 一句话可触发）

1. 重建 Phase-5 邻近性匹配对照 → 把 C1 判定换成 headline 本身（差异已定位到帧集合：
   RQ007 dev 118,212 帧 vs 本轮 143,706）。
2. **全量检验 `spread(mse)==0` 与 conflict 窗的对应**，摆脱 nuplan/waymo 共线
   （监督方评价此项价值最高）。

## 附带完成（第 6 条，非研究范围）

track G 已启动**恰好一次**：pid **70604**，PPID=1、PGID=70595，
38 秒内写出自己的 preflight 行（`通道=6bdcc2e6+envs/ipv-exact-sigma01`）。
未替 G 做任何研究工作，未读其 kickoff。
