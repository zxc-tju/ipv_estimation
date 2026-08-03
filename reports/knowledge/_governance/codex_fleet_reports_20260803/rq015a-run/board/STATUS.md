# STATUS — track A
state: DONE
updated_at: 2026-07-31T10:44Z
phase: closed
leader_pid: 71554

summary: **RQ015A 的审计跑出来了，报告写完了。** 两个 codex agent、一轮，无新增规格版本、
  无新增授权闸门。1,447 万 measurement 行，`run_receipt = PASS`，`bounded_report.md`
  + 4 组图（PNG/PDF）+ `usable_subset.csv` 已交付。`STUDIES.md`、`START_HERE.md`、
  `main_workflow.log` 均已同步。
next: 无。等 PI 决定是否提交（见下方唯一待办）。

## 六个核心数字

| artifact | measurement | ATTEMPTED | NOT_ATTEMPTED | UNKNOWN | q_eff 中位数 | q_eff≥0.93 |
|---|---:|---:|---:|---:|---:|---:|
| onsite_dense_timeseries | 281,268 | 2,974 | 4,272 | 274,022 | 0.5860 | 32.72% |
| wod_rq010b_full479_audited | 906 | 906 | 0 | 0 | 0.3719 | 26.60% |
| interhub_sigma01_hw4_timeseries | 5,197,072 | 4,981,984 | 215,088 | 0 | 0.9657 | 53.49% |
| rq009_feature_matrix | 8,994,736 | 8,994,736 | 0 | 0 | 0.8651 | 46.10% |

1. 三态合计：ATTEMPTED **13,980,600 (96.59%)** / NOT_ATTEMPTED 219,360 (1.52%) / UNKNOWN 274,022 (1.89%)
2. q_eff 中位数见上表；四产物值域两端**逐位命中理论界** [1/7, 1.0]
3. 近均匀占比见上表（**三个产物 bins 不稳定，仅作敏感性附注**）
4. 可用子集：**19,778 个 case/episode key**、34,283 unit row、3,049,608 ATTEMPTED 行 = 全集的 **21.27%**
5. 最强三个共变因素：`counterpart_current` ρ=−0.266 [−0.270,−0.261]；
   `target_future` ρ=0.155 [0.150,0.160]；`hw4` ρ=0.115 [0.108,0.121]（B=2000, seed 20260726）
6. C0：M3 训练标签 `NO_AUDIT_TRIGGER_DETECTED` **stable=false**；RQ009 包络 同上 **stable=false**；
   RQ014 筛选 `NOT_APPLICABLE`

## 三个最值得 PI 注意的结论

1. **C0 路由不稳定（最具行动后果）**：sigma01 与 rq009_feature_matrix 主判据"未触发"，
   但 `stable=false`，一档敏感性 → `OWNER_REANALYSIS_REQUIRED`。
   **rq009_feature_matrix 正是 RQ009 自己的输入。** 是否重估 RQ009 属 PI 决策，
   报告只给证据、不给建议。
2. **OnSite 覆盖度 1.06%**（2,974/281,268），UNKNOWN 274,022 (97.42%) 为
   `EMPTY_CELL_UNEXPLAINED`。空串未读作 0、未被强解释为 warm-up——保守处置，是优点。
3. **bins 四中三不稳定**（onsite 极差 21.02pp / sigma01 13.62pp / fm 15.29pp；仅 WOD 8.06pp 可报）。

## 交付物
```
reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/
    bounded_report.md            15 KB，14 个标题级小节
    figures/fig1–fig4            PNG + PDF 各 4
    usable_subset.csv            34,283 行 + 表头
    portraits.json / c0_routing.json / run_receipt.json / concentration_ledger_summary.csv
    concentration_ledger/*.parquet   374 MB（已被 .gitignore:67 忽略）
```

## 唯一待办（需 PI/监督方定，我没有自作主张）
**是否提交这 12 项改动。** 我未提交（未获指示）。关键事实：
**374 MB 的 parquet 台账已被 `.gitignore:67`（`reports/studies/**/*.parquet`）忽略**，
所以提交进 git 的只有报告、图、JSON、CSV 与代码改动，**体积很小、安全**。
我建议提交——否则这些成果的唯一保险是 B 轨的 `rollback_watchdog.sh`(pid 90240)，
而监督方说它要等 A 轨结项才可停，两者互相等待。
**给一句话我就提交。**

## 本轮的过程账（对照速度原则）
2 个 agent、1 轮、0 个新规格版本、0 个新授权闸门、0 轮盲审。
唯一的"额外保险"是 55 个受保护文件的 sha256 基线——那是硬约束 3 的效度边界，不是流程。
