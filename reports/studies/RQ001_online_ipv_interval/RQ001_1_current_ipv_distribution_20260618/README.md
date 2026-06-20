# 在线当前 IPV 合理性区间估计报告

入口页：`00_entry/index.html`

本报告回答的问题是：online sociality verification 时，如何判断当前自身 IPV 是否落在人类驾驶人在相同交互场景下的合理 IPV 分布内。

可用在线信息包括：

- 实时估计到的自身 IPV 历史序列；
- 交互双方历史轨迹；
- 交互双方参考线/道路语义信息的在线代理变量；
- paired IPV 分布规律，包括 pair-sum、pair-diff、符号互补和联合合理性。

## 指标含义

| 指标 | 含义 | 怎么判断好坏 |
|---|---|---|
| P50/P80/P90/P95 interval | 对应概率下的人类 IPV 合理区间。例如 P90 区间表示人类 IPV 应有约 90% 落在其中。 | 概率越高区间越宽；用于输出不同严格程度的合理性。 |
| 覆盖率 Coverage | 真实 IPV 落入预测区间的比例。 | P90 coverage 应接近 0.90；低于目标说明欠覆盖，不够安全。 |
| 平均宽度 Mean width | 区间平均宽度，即上限减下限。 | 在 coverage 接近目标时越小越精准。 |
| Winkler 区间分数 | 区间评分，同时惩罚区间过宽和真实值落在区间外。 | 越低越好，适合综合比较。 |
| CQR | Conformalized Quantile Regression，先预测条件分位数，再校准覆盖率。 | 适合直接输出多概率 IPV 分布区间。 |
| Source guard | 数据源/域校准保护。 | 遇到未知源或 source shift 时牺牲部分宽度换覆盖率。 |

## 主要结论

正常在线运行时，推荐使用：

`full_context_paired_source_guard_cqr_hgb`

它综合使用自身 lagged IPV、对方 lagged IPV、paired IPV 结构、轨迹历史、参考线/场景代理变量，并使用 CQR 输出多概率区间。

Primary split 的 P90 结果：

| 方法 | 覆盖率 | 平均宽度 | Winkler 分数 |
|---|---:|---:|---:|
| Full-context paired CQR + source guard | 0.904 | 0.859 | 1.243 |
| Full-context paired CQR | 0.901 | 0.863 | 1.244 |
| Paired-history CQR | 0.903 | 0.837 | 1.251 |
| Self-history CQR | 0.904 | 0.832 | 1.254 |
| Residual source guard | 0.901 | 1.082 | 1.579 |
| Global floor | 0.897 | 1.876 | 2.255 |

跨域或未知 source 时，推荐 fallback：

`full_context_paired_hgb_source_guard_conformal`

Leave-Waymo-Out stress test：

| 方法 | P90 覆盖率 | 平均宽度 |
|---|---:|---:|
| Residual full-context source guard | 0.906 | 1.203 |
| Baseline residual full-context paired | 0.856 | 0.960 |

## Paired IPV 的作用

Paired IPV 应纳入合理性计算，但不应简单替代 self-history 主模型。

H2 paired 实验发现，paired-history-only 相对 self-history 在 P90 coverage 上只提升约 0.46 个百分点，同时 mean width 增加约 0.57%，interval score 也略变差。因此 paired IPV 更适合作为：

- joint reasonableness 输出；
- source/volatility gating；
- 条件分布修正特征；
- pair-sum 和 pair-diff 分布检查。

Primary split 下 paired 分布 P90 覆盖：

| Paired 检查 | 覆盖率 |
|---|---:|
| Pair-sum | 0.898 |
| Pair-diff | 0.896 |
| Joint sum-and-diff | 0.802 |

## 推荐在线输出

| 字段 | 含义 |
|---|---|
| `ipv_interval_p50/p80/p90/p95` | 自身当前 IPV 的多概率合理区间。 |
| `ipv_reasonable_p90` | 自身当前 IPV 是否落在 P90 合理区间内。 |
| `ipv_tail_score` | 当前 IPV 在人类分布中的尾部程度。 |
| `pair_sum_interval_p90` | 自身 + 对方 IPV 的合理区间。 |
| `pair_diff_interval_p90` | 自身 - 对方 IPV 的合理区间。 |
| `pair_joint_reasonable_p90` | pair-sum 与 pair-diff 是否同时合理。 |
| `calibration_mode` | 当前使用 CQR、source guard 还是 fallback。 |
| `source_health` | source 是否已知、是否漂移、历史长度是否足够。 |

## 关键文件

- `00_entry/index.html`
- `01_results/final_summary.md`
- `01_results/current_ipv_interval_metrics.csv`
- `01_results/current_ipv_selected_cqr_metrics.csv`
- `01_results/paired_ipv_reasonableness_metrics.csv`
- `01_results/figures/`
- `02_process/run_current_ipv_distribution_experiment.py`
- `02_process/run_selected_cqr_methods.py`
- `02_process/agents/`
