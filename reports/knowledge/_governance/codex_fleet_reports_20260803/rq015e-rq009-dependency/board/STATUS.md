# STATUS — track E（rq015e-rq009-dependency）

state: DONE
updated_at: 2026-07-31T17:09:43Z
phase: 监督方已放行；三条修改已落实；本轨结束

## summary

**编制：1 个 codex agent（E1），1 轮 leader 独立复算，0 次盲审，0 版规格 v2。**
E1 16:43:26Z 派出 → 16:52:45Z 结项（9.5 分钟）。

### 结论口径（按监督方 17:08 放行指示定稿）

> RQ009-KC-R3 的**组成性质在受限子集上仍然成立**
> （`coverage_within_3pp = True`、`directional_tails_ok = True`、M2 仍显著窄于 M0）；
> 但**两个幅度数字站不住**：M2 相对 M0 的收窄从 −42.41% 降到 −29.65%，
> Winkler 从 −35.47% 降到 −29.51%。该差值 **+12.76 pp 是下界**，因为可用性口径是
> case 级，受限集内仍含 33.27% 的近均匀权重行。
> **是否重估 RQ009 属 PI 决策，本报告只给证据与边界，不给建议。**

| 量 | A（冻结，已发表） | B（参照系） | **C（受限）** | D（补集） | Δ(C−B) |
|---|---:|---:|---:|---:|---:|
| M2 vs M0 mean_width | −42.27% | −42.41% | **−29.65%** | −59.05% | **+12.76 pp（下界）** |
| M2 vs M0 Winkler | −35.61% | −35.47% | **−29.51%** | −44.71% | **+5.96 pp（下界）** |
| M2 coverage | 0.898889 | 0.898949 | **0.882762** | 0.920065 | −1.62 pp |
| abstention | 4.78% | 4.94% | **6.29%** | 3.14% | +1.34 pp |

差异已定位到唯一一侧：`mean_width(M0) = 1.748666` 在 B/C/D 上完全相同（全局 floor），
全部变化来自 M2：`1.007080 (B) → 1.230155 (C) → 0.716080 (D)`。
被可用性口径排除的 D（`share(q_eff≥0.93)` = 62.77%，`k_eff` 中位数 6.96/7）正是 M2
收窄幅度看起来最大的部分。**只报共现，不作因果归因。**

### 放行后落实的三条修改（全部已写入报告）

1. §1 判定段按监督方给定口径重写，"**+12.76 pp 是下界**"已进结论段（不只留在边界小节）。
2. 附录 A.3-1 强化为"稀释而非清除 ⇒ 行级过滤只会让差距更大"。
3. 新增 **附录 A.4 具名候选下一轮 E2**：按 `q_eff` 连续分层的 width/coverage 剖面
   （可同时压掉"下界"的定性论证与成分混淆），**本轮不做**，PI 一句话可触发。

### 硬边界（leader 自查 + 监督方复核，均通过）

- `held_out_rows_entering_any_statistic = 0`，且为**结构佐证**而非声明：
  台账 `development 6,459,684 + guard 2,535,052 = 8,994,736` 精确等于台账总行数；
  每 tier join 恒等式 `2,666,676 + 1,145,022 = 3,811,698`；`C ⊆ B` = True；
  `usable(rq009) ⊆ 台账` 派工前已核验。机制为先建 dev+guard case 集再下推过滤。
- 未读取任何 RQ014 致盲相关字段。
- 未修改/覆盖任何 RQ009 / RQ015A 冻结产物或任何 `decision.md`；未写入 `data/derived/`；
  未执行任何 git 写操作。产物全部在本 board 下。
- 措辞：无 `estimability` / "测出 IPV" 类表述；无因果语言；无重估建议。

### 交付物

```
board/reports/E1_dependency_report.md      （E1 正文 + leader 验收附注附录 A.1–A.4）
board/reports/restricted_metrics.csv       （set × tier × alpha × 全部度量）
board/reports/E1_numbers.json
board/scripts/e1_restricted_recompute.py
board/reports/E1.log
```

## next

无。本轨关闭。候选下一轮 E2 见报告附录 A.4，等 PI 决定是否触发。
