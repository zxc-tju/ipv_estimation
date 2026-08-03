# RQ015B 结项报告 — 平价门与机制拆分（本地可达 nuplan + waymo 子集，分层抽样 2,300 锚点）
## 1. 两句话结论（(a) 平价门过/不过、兜底是否不可达；(b) D0–D4 构成与"修得好的"占多少）
平价门过：未触发兜底且无部分下溢的 1526/2300 个锚点上，legacy 与 log 域最大权重差 3.75e-15；log 域分母不可归零的 pytest 返回码 0，极端 RMS 实验分母 1。
本地可达 post-warm 零值框（U∪Z，HT 分母 534,939）中，合并机制构成为 D1 43.01% [39.35%, 46.83%]、D2 39.48% [35.69%, 43.08%]。**但该合并值几乎完全由 waymo 抬起，两个来源方向相反**：waymo（占框 72.7%）D1 58.73% / D2 30.45%；nuplan（占框 27.3%）**D1 仅 1.06% / D2 63.56%**。即：log 域改写在 **waymo** 上能让大量零值锚点获得候选间判别信息；在 **nuplan** 上，多数零值锚点在当前网格与模型下候选间差异本就极小，改数值对其不起作用。**是否申请全量重跑预算属 PI 决策，本报告只给证据与分源边界，不给建议。**（分源分解见 §6 逐层表；合并值的 95% CI 半宽最大 3.74 pp，未达 ≤3 pp 精度上限。）

本轮结论刻画的是**当前代码在抽样锚点上的行为**；存档主表（2026-06-12）的
逐锚点值在本轮未能复现（复现门 gate_a 12/40，阈值 39/40），
故上述比例**不得读作存档 M3 标签的成分构成**。

## 2. 覆盖边界与口径（三项边界 + 两个分母）
- 结论外推范围：本地可达的 nuplan + waymo 子集；抽样样本 2,300 个锚点，split 仅 development/guard。
- 两个分母：精确零、dev+guard 后热身为 1,200,636 / 4,981,984 = 24.10%；`|ipv| < 1e-6`、dev+guard 全体含 warm-up 为 2,008,902 / 5,197,072 = 38.65%。
- 38.65% 是既有认知里“约四成”的口径，与 24.10% 分母不同，不是冲突。
- 覆盖边界 1：`lyft_train_full` 785,080 锚点与 `av2_motion_forecasting` 234,546 锚点无本地原始数据，本轮完全不可达。
- 覆盖边界 2：waymo 缺 `500-799` 分片。
- 覆盖边界 3：`waymo_300-499.pkl` 存在但读取报 `pickle data was truncated`，已按 `pkl_available=False` 排除。

## 3. 平价门（4 条判据逐条给数字）
- T5 执行记录：executor=thread，workers=6，anchors=2300，solve_errors=0，elapsed=1240.3s；串行一致性检查 n=24，max_diff=0。
1. 无兜底且无部分下溢：n=1526/2300，max=3.75e-15，p99=1.42e-15，median=2.78e-17；全样本 max_abs_diff=0.857。
2. 已触发兜底：legacy 为均匀权重；log 域 k_eff 分布见 §4。
3. log 域不可走除零分支：`logw -= logw.max()` 后最大项为 0，`exp(0)=1`，分母至少为 1；极端 RMS 实验 finite=True，k_eff=1；pytest 返回码 0。
4. 临界 RMS 复核：n=5 与 n=11 均无早于闭式阈值的实测归零；细节见 §5。

| source | n_band | sig | n | max | p99 | median |
|---|---|---:|---:|---:|---:|---:|
| nuplan | FULL | N | 93 | 8.33e-16 | 5.77e-16 | 5.55e-17 |
| nuplan | FULL | U | 273 | 2.22e-16 | 1.11e-16 | 0 |
| nuplan | FULL | Z | 150 | 3.61e-16 | 2.22e-16 | 2.78e-17 |
| nuplan | RAMP | N | 106 | 2.5e-15 | 8.6e-16 | 5.55e-17 |
| nuplan | RAMP | U | 268 | 4.72e-16 | 1.11e-16 | 0 |
| nuplan | RAMP | Z | 148 | 1.64e-15 | 4.95e-16 | 2.78e-17 |
| waymo | FULL | N | 73 | 2.89e-15 | 2.61e-15 | 1.39e-16 |
| waymo | FULL | U | 2 | 1.67e-16 | 1.65e-16 | 9.71e-17 |
| waymo | FULL | Z | 143 | 1.72e-15 | 1.54e-15 | 1.11e-16 |
| waymo | RAMP | N | 79 | 3.75e-15 | 1.6e-15 | 1.11e-16 |
| waymo | RAMP | U | 43 | 1.3e-15 | 1.13e-15 | 2.78e-17 |
| waymo | RAMP | Z | 148 | 2.25e-15 | 2.03e-15 | 8.33e-17 |

## 4. 兜底行上 log 域的表现（k_eff 分布 —— 修得好还是本就平坦）
- 兜底样本：603/2300；签名分布 {'U': 603}。
- log 域 k_eff：min=1，Q25=1，median=1.44096，Q75=4.73094，p95=6.97423，p99=6.99768，max=7。
- 判据 k_eff/K < 0.93 下，兜底行中 log 域实质非均匀：509/603；近均匀：94/603。
- 权重近均匀表示该 IPV 数值不携带候选间的判别信息。

## 5. 临界 RMS 复核（闭式 vs 实测，含已批准的容差口径）
| n_obs | boundary | closed RMS m | np.prod onset m | delta mm | observed candidate onset m | observed row fallback min_rms m | early count |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | subnormal | 1.691526 | 1.691526 | 0.000 | 1.71962 | 1.7459 | 0 |
| 5 | zero | 1.733619 | 1.734418 | 0.799 | 1.7459 | 1.7459 | 0 |
| 11 | subnormal | 1.147025 | 1.147025 | 0.000 | 1.14715 | 1.17759 | 0 |
| 11 | zero | 1.175245 | 1.175781 | 0.536 | 1.17759 | 1.17759 | 0 |
- 逐项 `np.prod` 在 subnormal 区间累积舍入；本机实测归零点晚于闭式 0.536-0.799 mm，方向为晚且量级为毫米级，按已批准口径不算失败；本轮未见负向提前归零。

## 6. 机制拆分 D0–D4（框加权占比 + 95% CI + 逐层表）
- D0 warm-up 普查：nuplan 42,432 / waymo 130,624，合计 173,056；D1-D4 主估计量分母为本地可达 post-warm U∪Z HT 框 534939。
- `min_mse_misfit=Q0.99(min_mse)=18.695342`，阈值文件 SHA-256 `8ec2cdc106fced6a663e062418e01ac0d816b6c0a6c6faa6b9aa87e461c10345`；这是本抽样样本估计的本轮临时口径，不得据此冻结生产阈值。
- case-cluster bootstrap：B=2000，seed=20260731，cluster=1459；最大 95% CI 半宽 3.74 pp，≤3 pp 判据 未通过。
- **`D3 = 0.00%` 是冻结优先级 `D4 > D1 > D3 > D2 > OK` 的产物，不得读作"不存在模型失配"。** 23 个 `min_mse > Q0.99` 的 in-scope 锚点全部先被 D1 吸收，因此 D3 桶在本口径下不可能非零；本轮对模型失配的发生率**无判别力**。
- **合并值只在"本地可达 nuplan + waymo 子集"这一框内成立，且分源方向相反**（waymo D1 58.73% / D2 30.45%，nuplan D1 1.06% / D2 63.56%）。合并 D1 43.01% 由占框 72.7% 的 waymo 抬起，**不得作为整体性质外推**。原始计数方向一致，非加权造成：U 层触发兜底 waymo 551/600 = 91.8%，nuplan 52/600 = 8.7%。

| mechanism | HT count / denominator | estimate [95% CI] |
|---|---:|---:|
| D1_NUMERICAL_UNDERFLOW | 230081.3 / 534939.0 | 43.01% [39.35%, 46.83%] |
| D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL | 211194.0 / 534939.0 | 39.48% [35.69%, 43.08%] |
| D3_MODEL_MISFIT | 0.0 / 534939.0 | 0.00% [0.00%, 0.00%] |
| D4_SOLVER_OR_INPUT_FAILURE | 0.0 / 534939.0 | 0.00% [0.00%, 0.00%] |

| threshold | D1 | D2 | D3 | D4 | OK |
|---|---:|---:|---:|---:|---:|
| p0.95 | 43.01% | 39.48% | 0.00% | 0.00% | 17.51% |
| p0.99 | 43.01% | 39.48% | 0.00% | 0.00% | 17.51% |
| p0.999 | 43.01% | 39.48% | 0.00% | 0.00% | 17.51% |

**这张敏感性表三行完全相同，不构成"结论对阈值稳健"的证据。** 原因同上：在冻结优先级 `D4 > D1 > D3 > D2 > OK` 下，`min_mse` 阈值只影响 D3 桶，而所有超阈锚点都已被 D1 先行吸收，因此改变 p0.95/p0.99/p0.999 在本样本上**不可能**改变任何一格——该表对本轮口径**无区分力**。

| source | n_band | sig | drawn/cases | HT denom | D1 | D2 | D3 | D4 | OK | repair_good |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nuplan | FULL | U | 300/226 | 16510 | 8.33% | 90.33% | 0.00% | 0.00% | 1.33% | 9.00% |
| nuplan | FULL | Z | 150/143 | 124198 | 0.00% | 59.33% | 0.00% | 0.00% | 40.67% | 40.67% |
| nuplan | RAMP | U | 300/213 | 1630 | 10.00% | 89.67% | 0.00% | 0.00% | 0.33% | 10.33% |
| nuplan | RAMP | Z | 150/145 | 3483 | 0.00% | 75.33% | 0.00% | 0.00% | 24.67% | 24.67% |
| waymo | FULL | U | 300/288 | 222963 | 99.00% | 0.67% | 0.00% | 0.00% | 0.33% | 83.33% |
| waymo | FULL | Z | 150/149 | 149289 | 0.67% | 72.67% | 0.00% | 0.00% | 26.67% | 27.33% |
| waymo | RAMP | U | 300/284 | 7915 | 85.33% | 14.33% | 0.00% | 0.00% | 0.33% | 71.33% |
| waymo | RAMP | Z | 150/144 | 8951 | 0.67% | 82.67% | 0.00% | 0.00% | 16.67% | 17.33% |

| 层 | 复现门 | 能支持的说法 | **不能**支持的说法 |
|---|---|---|---|
| U（676,405） | 12/14 | 兜底触发本身可复现；当前代码下该层的 D1/D2 拆分 | "存档这 676,405 行里 X% 是 D1"——不动点不承载权重信息 |
| Z（**524,231** = 1,200,636 − 676,405） | **0/14** | 仅"当前代码下这批锚点的机制构成" | 任何向存档 Z 层的外推 |
| N | 0/12 | 同上 | 同上 |

Z 层（524,231 行，精确零但无均匀签名）是 D1/D2 的关键分界，但该层复现率为 0/14；因此本轮只报告当前代码在抽样锚点上的机制构成。

## 7. 复现门未过：一等边界（不是脚注）
本轮结论刻画的是**当前代码在抽样锚点上的行为**；存档主表（2026-06-12）的
逐锚点值在本轮未能复现（复现门 gate_a 12/40，阈值 39/40），
故上述比例**不得读作存档 M3 标签的成分构成**。

## 8. 数值健康自查（没有问题就写"无"）
- 非有限/异常：row=0/2300，solve_error=0/2300，mse 非有限 cell=0/16100。
- mse_per_candidate 非病态常数：候选列 std 范围 16.3266 到 34.0359。
- min_mse 分布：min=0，p50=0.0551035，p95=6.52572，p99=18.6953，max=655.533。
- 同一案例贡献锚点数最大 6；分布 {1: 1620, 2: 216, 3: 52, 5: 6, 4: 14, 6: 1}。
- 兜底与 U 签名一致性：兜底但非 U = 0；U 但未兜底 = 597/1200。
- 除上述边界信号外，无。

## 9. known issues（代码漂移假说，留待独立一轮）
- 存档逐锚点值未过复现门；求解器差异、参考线处理或代码漂移均未在本轮追查。
- 可合并片段：`.codex-fleet/rq015b-repair/board/reports/known_issue_snippet_repro_gate.md`。

## 10. 产物清单（路径 + sha256 + 每列一行说明）+ 完工自检三项 SHA
- `.codex-fleet/rq015b-repair/work/anchor_mse.csv` sha256 `b0f6202501ea738b1ae6d49f83af1877bee85b391d5db6a44375d67b552eb114`
- `.codex-fleet/rq015b-repair/work/mechanism_split.csv` sha256 `d227eae153369c916f1515cd93a5f754fed81b50e2b2d5d387a760cc5c3d8aad`
- `.codex-fleet/rq015b-repair/work/min_mse_misfit_threshold.json` sha256 `8ec2cdc106fced6a663e062418e01ac0d816b6c0a6c6faa6b9aa87e461c10345`
- `.codex-fleet/rq015b-repair/work/fallback_unreachable_experiment.json` sha256 `95c7be63af7002b4e224e03111eabf202a17194e2e0da3c3f2cf608801c9dc60`
- `.codex-fleet/rq015b-repair/work/test_b2_fallback_unreachable.py` sha256 `1e8e02f34e2dda11567b47b328981f5c578d90601d9e2e78d789b4f5ee431ccc`
- `.codex-fleet/rq015b-repair/work/b2_pytest_output.txt` sha256 `35d86279d96ed65491ba8ff0748e1c8e7891711531404dc5d4b9cd6ae33de3ec`
- `.codex-fleet/rq015b-repair/board/reports/known_issue_snippet_repro_gate.md` sha256 `be9970c110a26c919304fbb44b3ccd64e3ccc2704d519769628900e33015f170`
- `.codex-fleet/rq015b-repair/work/run_b2_rq015b.py` sha256 `5b5c02ba9e30b3b0c987af453555707744ff610a5be5afa6718a26e32003ff64`

anchor_mse.csv 列说明：
- `sample_order`：冻结样本顺序。
- `anchor_id`：锚点唯一键。
- `scene_unique_id`：case-cluster bootstrap 的案例键。
- `dataset/source/folder`：本地可达数据来源与 pkl 分片。
- `n_band/signature/split`：抽样层、零值签名、development/guard split。
- `frame_index/agent_slot/n_obs/K`：锚点帧、角色、观测步数、候选数。
- `mse_per_candidate[7]/rms_per_candidate[7]`：同一组重解候选轨迹对应的 MSE/RMS。
- `legacy_var[7]/legacy_density_product[7]`：legacy 概率域连乘诊断量。
- `min_mse/min_rms/argmin_candidate`：最佳候选位置与误差。
- `legacy_prod_sum/legacy_fallback_triggered/partial_underflow`：legacy 分母与下溢标志。
- `w_legacy[7]/w_log[7]/max_abs_diff/manual_legacy_weight_diff`：生产 legacy 权重、log 域权重及差异。
- `ipv_legacy/ipv_log/ipv_error_legacy/ipv_error_log`：两套权重下的 IPV 数值与误差显示值。
- `k_eff_legacy/k_eff_log/at_grid_boundary/any_nonfinite/solve_error`：有效候选数、边界与异常记录。

- HEAD actual `e82091ceaa2586bdb09b6153dfbed3be24d6bf98` expected `e82091ceaa2586bdb09b6153dfbed3be24d6bf98` => OK
- `src/sociality_estimation/core/agent.py` actual `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` expected `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` => OK
- `src/sociality_estimation/core/reliability_logdomain.py` actual `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` expected `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` => OK

## 11. 偏差与未做的事（有就写，没有写"无"；不得静默截断）
- 未追查复现门失败根因；按监督裁定作为边界记录。
- 未追加抽样、未改配额、未接线生产估计器、未修改 tracked 文件。
- 未写 root `main_workflow.log`：本轮铁律限定全部输出只能写入 `.codex-fleet/rq015b-repair/`。
- **结项后文本更正（不静默）**：2026-07-31T10:43Z 依监督方第 6 条对本报告做过三处**纯文本**更正——
  §1 第二句改写（分源结构提到结论层、删去"支持申请全量重跑预算"）、§6 增补 D3=0 的优先级成因与
  敏感性表无区分力标注、CI 半宽超限重申。**未重跑 T5、未重派 B2、未改动任何数字或产物 SHA**；
  §1 下方与 §7 的复现门边界段逐字保留。更正前后判定过程见 `board/reports/B2_leader_adjudication.md`。
