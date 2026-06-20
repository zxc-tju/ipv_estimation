# 可追溯记录

## 数据来源

- `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv`（2.2GB，3.7M 帧：逐帧 px/py/vx/vy/heading + 逐帧因果 IPV）
- `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_case_summary.csv`（38,228 case：几何/路权/PET + 全程 IPV 均值=标签）
- 原始 pkl 运动/车道源：经 `process_interhub.py` 的 `lane_ids → 车道中心线` 逻辑读取
- 在线因果 IPV 估计器：仓库 `ipv_estimation.py` 的 `RealtimeIPVEstimator`（`docs/realtime_ipv_estimator.md`）
- 既有报告：`reports/studies/RQ001_online_ipv_interval/RQ001_2_interval_query_20260618/`（2026-06-18 interval-query 探索版，本报告的前身）

## 标签与无泄漏契约

- 标签 `y` = 该 agent 全程 IPV 均值（`ipv_key_agent_{1,2}_mean`），长表化为 one-row-per-(case,agent)。
- 只用早窗（前 1–2s）因果特征：几何、路权角色（`priority_label`）、AV/HV 身份、运动学 risk proxy、自身因果 rolling-IPV。
- 禁用泄漏量：observed PET、actual_order、intensity、min_distance/closest_frame、全程均值（=标签）、事后 phase 等——仅 `oracle_pet` 作上限参照与诊断，绝不作特征。
- 划分按 case 分组、按数据集分层；另设 Leave-Waymo-Out 跨源压力测试。详见 `02_process/board/DATA_BRIEF.md` 与 `01_results/`/`02_process/scripts/build_features.py`、`metrics.py`。

## 主分析脚本（`02_process/scripts/`）

- `build_features.py`：流式读 2.2GB 时序，构建无泄漏 case-agent 特征表 + 划分 + 基线（FLOOR / oracle CEILING）。
- `metrics.py` / `baselines.py`：coverage/width/pinball/Winkler/CRPS 指标与上下界基线 → `01_results/baseline_metrics.csv`。
- 四方法：`run_hA_*`（直接运动学分位数）、`run_hb_*`（rolling-IPV 自锚）、`run_causal_risk_conformal*`（causal risk+conformal）、`run_distributional*`（条件密度）。
- `run_w3b_lane_causal_rebuild.py`：用 `RealtimeIPVEstimator`+地图车道参照做严格前缀因果重建（Gate-A corr 0.993）。
- `run_balanced_lock.py`：平衡 5,000-case / 10,000-row 因果重建 + 锁定评估 → `01_results/metrics_balanced_lock.csv`、`predict_interval_balanced_lock.py`、`model_balanced_lock_cqr.joblib`。
- `online_ipv_interval.py` + `run_ab.py`：verifier 即插即用估计器 + deviation_score + A/B vs PET-bin 包络 → `01_results/ab_metrics.csv`、`02_process/INTEGRATION.md`。
- `make_nature_figures.py`：经 `nature-figure` skill 产出 `01_results/figures/`。

## Fleet 子实验（15 个 Codex agent，简报见 `02_process/agent_reports/`）

- **scout-priorart**：实时联网 SOTA（CQR/加权 conformal、分布/PIT conformal、有界变换、SVO 在线估计）。
- **harness**：共享特征底座/划分/指标/基线；独立无泄漏审计通过。
- **hA / hB / hC / hD**：直接运动学分位数 / rolling-IPV 自锚 / causal-risk+conformal / 条件密度。结论：rolling-IPV 同时最锐且最稳；oracle PET ≈ 全局。
- **redteam-rolling-causality**：红队。判定 rolling-IPV 非标签复制（corr 0.58–0.72），但离线版用全程参照 → PARTIAL 因果（需用正确参照重建）。
- **replicate-rolling-headline**：独立 QRF+conformal 路线复现头条（cov 0.904 / width 0.478），AGREE。
- **w3-causal-rebuild**：首次严格前缀因果重建（用观测前缀参照）→ corr 0.28（崩），定位"参照泄漏"。
- **w3b-lane-causal-rebuild**：改用**地图车道参照** → corr 0.993，证明 rolling-IPV 可部署（route-conditioned）。
- **w4-verifier-integration**：verifier A/B，新包络全面优于 PET-bin（更窄、误报不升/降）。
- **w4-full-causal-rebuild**：全量重建成本探针（2.6s/case → ~20.5h），确立需平衡抽样。
- **w4c-balanced-driver**：平衡 5,000-case 因果重建驱动 → 锁定生产数字（本报告 §结果）。
- **w5a-nature-figures**：经 `nature-figure` skill 出 4 张 publication 图。
- **w5b-nature-writing-html**：经 `nature-writing` skill 写中文 Nature 风格正文并组装自包含 HTML（`00_entry/index.html`）。

## 在线可用性边界

- 允许：当前/历史轨迹状态；自身 lagged 因果 IPV（时间戳早于 `t`，由 `RealtimeIPVEstimator` 前缀估计）；地图车道中心线（静态、在线可得）；几何/路权/身份。
- 假设：决策时已知 agent 的车道/路线（route-conditioned）；适用于 74% `lane_ids` 案例；其余 ~26% 退回 no-roll 运动学。
- 局限：Waymo 覆盖虽达 0.90，但一般跨源漂移（标签漂移）仍建议小样本目标域再校准；锁定数字基于平衡 5k 抽样（Av2/nuPlan 本地车道可用性受限，CI 较宽）；sigma=0.1。
