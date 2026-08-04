# D1 — σ 特征化与反事实（RQ015B B3）

你是 track D 的**唯一执行 agent**。任务是**纯统计分析**：所有原料已经落盘，
**不要重解任何轨迹、不要跑任何 IPV 求解、不要调用任何数据管线**。

工作目录（下面所有相对路径都相对它）：
`.`

Python 解释器**钉死**：`<local-rq009-venv>/bin/python`
（系统 python3 缺包，不要用它。numpy/pandas 在这个 venv 里可用。）

---

## 0. 你要读的文件（就这些，不要在仓库里做宽泛检索）

**唯一数据输入：**
```
.codex-fleet/rq015b-repair/work/anchor_mse.csv        (2,301 行含表头 = 2,300 数据行, 2.4 MB)
```
**可选的上下文输入（只读，用于对齐口径，不是数据源）：**
```
.codex-fleet/rq015b-repair/work/b2_summary.json
.codex-fleet/rq015b-repair/work/min_mse_misfit_threshold.json
```
**禁止**：`rg` / `grep` 全仓库；禁止读 `reports/` 目录下的任何文件（有 controlled-access 内容）；
禁止读 `.codex-fleet/rq015b-repair/work/anchor_meta.csv`（373 MB，本任务用不到）；
禁止读 `sample_candidates.sqlite`、`frame_index.csv`。

`anchor_mse.csv` 的列（已核实）：
```
sample_order anchor_id scene_unique_id dataset source folder n_band signature split
frame_index agent_slot n_obs K
mse_per_candidate[7] rms_per_candidate[7] legacy_var[7] legacy_density_product[7]
min_mse min_rms argmin_candidate legacy_prod_sum legacy_fallback_triggered
w_legacy[7] w_log[7] max_abs_diff manual_legacy_weight_diff
ipv_legacy ipv_log ipv_error_legacy ipv_error_log
k_eff_legacy k_eff_log at_grid_boundary any_nonfinite partial_underflow solve_error
```
`[7]` 列是 JSON 数组字符串，用 `json.loads` 解析。

已核实的样本构成（你要在报告里复核并复述这些数字）：
`split`: development 1647 / guard 653（**没有 held_out / sealed 行**）；
`source`: nuplan 1150 / waymo 1150；`n_band`: FULL 1150 / RAMP 1150；`K`=7 全部；
`legacy_fallback_triggered`=True 603 行；`partial_underflow`=True 171 行；
`solve_error` 全空，`any_nonfinite` 全 False。

---

## 1. 要回答的问题（RQ015B 计划 §5）

**σ = 0.1 是否需要重新推导？** —— **"不需要"是完全可接受、甚至是预期之一的结论。**
不要为了让报告"有结论"而去改 σ。你的任务是让证据说话。

权重锐度由 `Δlog w = ΔMSE / (2σ²)` 决定，σ=0.1 ⇒ 系数 50。
坍缩与否取决于**候选之间实际的 MSE 间距**：
间距 RMS ~0.1 m (ΔMSE~0.01) ⇒ K_eff≈2.2（未坍缩）；
间距 RMS ~1.0 m (ΔMSE~1.0) ⇒ K_eff<1.05（**坍缩为硬 argmax = 虚假自信**）。

### 必须正面回答的张力（本轮的核心）

计划 §5 担心 **σ 太小 ⇒ 权重过锐 ⇒ 硬 argmax ⇒ 虚假自信**（K_eff→1）。
但 RQ015A 全量实测画像**方向相反**：sigma01 的 k_eff 中位数 **6.76/7**，
53.49% 的 ATTEMPTED 行权重**近均匀**（q_eff = k_eff/K ≥ 0.93）。
且已知 RQ015A 全量里有 676,405 行是**精确均匀兜底**（legacy 连乘下溢 ⇒ Σ=0 ⇒ 强制 1/K）。

**用 anchor_mse.csv 把这个张力说清楚。**

---

## 2. 分析步骤（按顺序做，每一步的数字都要进 JSON 落盘）

### Step 0 — 反推并核验权重公式（先做，后面全部依赖它）
不要假设公式，**从数据里反推**：对每一行，用 `w_log[7]` 与 `mse_per_candidate[7]`
拟合 `log w_i = -c · mse_i + const`（对每一对候选取 `log(w_i/w_j)/(mse_j - mse_i)`，
取中位数得该行的 c；剔除 w 下溢到 0 的候选对）。
报告 c 的分布，并与理论值 `1/(2σ²) = 50`（σ=0.1）对比。
**报出 c 的中位数、p1/p99、以及相对偏差 |c-50|/50 的最大值。**
如果 c ≈ 50，就确认了 `w ∝ exp(-mse/(2σ²))`，MSE 已经是逐观测均值、**不再乘 n_obs**；
如果不是 50，如实报出实际系数，并据此修正后续所有反事实计算的公式。
（同时报出：`k_eff` 的定义核验 —— 用 `w_log` 算 `exp(-Σ w ln w)` 或 `1/Σw²`，
两种都算，看哪个能复现 `k_eff_log` 列，报出你确认的定义。）

### Step 1 — 两种模式各占多少
定义（**在报告里写死这三个阈值**）：
- **近均匀** `q_eff = k_eff / K ≥ 0.93`
- **硬 argmax** `k_eff ≤ 1.05`
- 其余为中间态

对 **legacy 权重（`k_eff_legacy`）** 和 **log-domain 权重（`k_eff_log`）** 分别统计：
计数、占比、k_eff 的 5 数概括。
**必须分源报（nuplan / waymo 各一套）**，再报合并值；另外按 `n_band`(FULL/RAMP) 交叉一次。
已核实的合并参考值（你要复现、不一致就说明为什么）：
`k_eff_legacy` median 7.000 / mean 6.219；`k_eff_log` median 6.795 / mean 5.111；
近均匀占比 legacy 0.7530 / log 0.5317；硬 argmax 占比 legacy 0.0217 / log 0.1287。

### Step 2 — 模式出现在什么 MSE 尺度上
对每行计算候选间距的两个刻画：
- `gap12_mse` = 次小 mse − 最小 mse（决定 argmax 置信度的那个间隙）
- `spread_mse` = max(mse) − min(mse)；以及 `spread_rms` = max(rms) − min(rms)
按 `min_rms` 的十分位分箱，报每箱里近均匀 / 中间态 / 硬 argmax 的占比，
以及 `gap12_mse`、`spread_mse` 的中位数。**分源做。**
**判读要正面回答：是不是同一个 σ 在不同 MSE 尺度上同时制造了两端失效？**
（即：低 MSE 尺度的行因间距远小于 2σ² 而近均匀；高 MSE 尺度的行因间距远大于 2σ² 而硬 argmax。
如果是，报出两端各自的行数与所在的 min_rms 区间；如果不是，如实说不是。）

### Step 3 — 扣掉兜底行之后的近均匀是什么原因
`legacy_fallback_triggered == True`（603 行）是 legacy 连乘下溢导致的**精确均匀**，
它是 B2 正在修的缺陷，不是 σ 的性质。
- 报出这 603 行的 `k_eff_legacy`（应为精确 7）与它们在 log-domain 下的 `k_eff_log` 分布——
  **这是关键对照**：下溢兜底把"过锐"伪装成了"均匀"吗？给出这 603 行 `k_eff_log` 的分位数与
  硬 argmax 占比。
- 扣掉这 603 行后，剩余 1,697 行里 legacy / log 的近均匀占比各是多少？
- 对**剩余的近均匀行**，检查其 `gap12_mse` 与 `spread_mse`：
  近均匀是因为候选间距真的小于分辨尺度（`spread_mse ≪ 2σ² = 0.02`），
  还是因为别的原因（例如多个候选 mse 几乎相同 = 候选网格在该场景下不可分辨）？
  给出 `spread_mse / (2σ²)` 这个无量纲比值的分布，这是最直接的判据。
- 同样分源报。

### Step 4 — 反事实 σ
先计算 `σ_rederived = median(sqrt(min_mse))`（**只在 development + guard 上，即全部 2,300 行；
报出参与计算的行数**）。已核实合并值 ≈ 0.2347，**分源也要各报一个**。

然后做 **σ 扫描**（不是只算一个点）：
`σ ∈ {0.01, 0.02, 0.03, 0.05, 0.075, 0.1(现状), 0.15, 0.2, σ_rederived, 0.3, 0.5, 1.0}`
对每个 σ，用 log-domain 数值稳定的 softmax（`w ∝ exp(-(mse_i - min_mse)/(2σ²))`，先减最小值）
重算每行权重，输出：
- `k_eff` 的中位数 / IQR
- 近均匀占比、硬 argmax 占比
- 若 legacy 连乘实现保持不变，该 σ 下**会不会加剧下溢**（用 `legacy_density_product` /
  `legacy_prod_sum` 的量级做定性判断即可，报一句话结论）
- **分源各一套**
落盘成 `sigma_sweep.csv`（列：sigma, source, n, k_eff_median, k_eff_p25, k_eff_p75,
frac_near_uniform, frac_hard_argmax）。

**明确回答**：把 σ 从 0.1 改成 σ_rederived≈0.235 是让权重更锐还是更平？
它把计划 §5 担心的"虚假自信"问题解决了，还是走向了反方向？
存在单一 σ 能同时把两端都修好吗？如果不存在，如实写"不存在"，并说明原因
（例如各行 MSE 尺度跨越数量级，单一全局 σ 无法同时匹配）。

### Step 5 — 结论
基于 Step 1–4 给出 **B3 的判定**：**需要重定 / 不需要重定**，并给出理由链。
如果判"不需要重定"，必须同时说明：那么 RQ015A 观察到的近均匀现象应该归因于什么
（可辩护的表述见下），以及这对下游意味着什么。

---

## 3. 硬边界（逐条遵守，报告里要有一节逐条声明）

```
□ 拟合只在 development + guard 上进行；RQ007 sealed 禁止参与。
   按 split 列过滤并在报告里报出过滤后的行数（预期 1647+653=2300，若出现其它 split 值，
   立即剔除并在报告里显著标注）。
□ σ 是 heuristic，不是观测噪声标准差 —— 它混合了真实观测噪声、候选网格的离散化误差、
   与模型偏差。报告必须显式声明这一点，以及选择目标是
   "使权重锐度与候选可分辨性相称"，而不是"估计真实噪声水平"。
□ 若证据显示未坍缩，则【不动 σ】，如实写"不需要重定"。不要为了有结论而改 σ。
□ 本轮【不接线生产】：src/sociality_estimation/core/agent.py 一字不动。
   任何 src/ 下的文件都不要改。不要新建 estimator_version、不要改任何冻结产物。
□ 结论只在"本地可达 nuplan + waymo 子集、2,300 锚点样本"这个框内成立。
   已知 waymo 与 nuplan 结构差异极大 —— 分源报，不要只报合并值。
□ 全文禁用 "estimability" 一词，禁用"测出/未测出 IPV"这类说法。
   可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。
□ 描述性结果不得写成因果主张。
□ git：禁止 git commit / git checkout / git restore / git stash / git reset / git clean。
   工作区非空是预期状态（另外两条 track 的 agent 正在同一仓库并发工作）。
   你只负责你自己新建的文件，不要碰任何你没创建的文件。
```

---

## 4. 交付物（只写这几个文件，不要写别的地方）

```
.codex-fleet/rq015d-sigma-rederive/board/reports/D1_sigma_report.md   ← 主报告
.codex-fleet/rq015d-sigma-rederive/work/d1_sigma_analysis.py          ← 可复跑分析脚本
.codex-fleet/rq015d-sigma-rederive/work/d1_sigma_stats.json           ← 所有数字（报告引用它）
.codex-fleet/rq015d-sigma-rederive/work/sigma_sweep.csv               ← σ 扫描表
.codex-fleet/rq015d-sigma-rederive/work/mode_by_min_rms_decile.csv    ← Step 2 交叉表
```
（图不是必需的。如果做图，放 `.codex-fleet/rq015d-sigma-rederive/work/figs/`，
用 matplotlib Agg 后端，不要弹窗，不要装新包。）

报告结构：
1. 一句话结论（需要 / 不需要重定 σ）
2. 样本与过滤（行数、split、source、n_band 构成；σ 是 heuristic 的显式声明）
3. Step 0 公式核验
4. Step 1 两种模式的占比（分源）
5. Step 2 模式 vs MSE 尺度（分源）
6. Step 3 扣掉兜底行之后（分源）——含 603 行的 legacy-vs-log 关键对照
7. Step 4 反事实 σ 扫描（分源）
8. Step 5 判定与理由链
9. 边界与不成立范围（本地子集、2,300 锚点、nuplan/waymo 结构差异、不可外推）
10. 硬边界逐条声明

**报告要短而硬**：数字优先，每个论断后面跟得上它的数字来源（JSON 里的键名）。
不要写工作日志式的"我先尝试了什么"。不要写规格 v2。**一轮做完，不要自己加审计轮次。**

## 5. 自查（在报告末尾加一节 "Self-check"）
- [ ] 所有数字可由 `d1_sigma_analysis.py` 重跑复现（脚本能独立运行，写明命令行）
- [ ] Step 1 的合并参考值复现一致（不一致则解释）
- [ ] 没有 sealed/held_out 行进入任何统计
- [ ] 没有修改任何 `src/` 文件、没有 git 写操作
- [ ] 全文无 "estimability"、无"测出/未测出 IPV"
- [ ] 分源结论都在，没有只报合并值
- [ ] 若判定"需要重定"，是否真的是证据驱动而不是为了有结论
