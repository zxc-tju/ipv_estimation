# 2_sociality_estimation 项目结构文档

## 1. 项目定位

本项目围绕“交互偏好值”（IPV, Interaction Preference Value）展开，当前保留两条主线：

- 轨迹数据离线/在线估计：从 InterHub CSV/pkl 等轨迹数据中估计双车 IPV 序列并导出结果。
- 交互行为仿真分析：基于博弈/规则控制器进行双车交互仿真与分析。

核心能力只放在 `src/sociality_estimation/`；数据管线、仿真入口和报告脚本调用这个公共包。

---

## 2. 当前目录总览

```text
.
├─ src/sociality_estimation/
│  ├─ core/
│  │  ├─ agent.py              # 核心智能体：规划、博弈、IPV 估计
│  │  └─ ipv_estimation.py     # 面向数据管线/在线调用的 IPV 估计封装
│  ├─ verifier/
│  │  ├─ model.py              # 冻结 M3 quantile/CQR/OOD 推理对象
│  │  ├─ scorer.py             # 便携 scorer 加载、校验与推理入口
│  │  ├─ deviation.py          # 规范化前的原始 envelope exceedance
│  │  └─ features.py           # RQ009/RQ012 共用因果特征公式
│  └─ planning/
│     ├─ utility.py            # 几何、平滑、运动学通用工具
│     ├─ lattice_planner.py    # Lattice 规划入口封装
│     └─ Lattice.py            # Lattice 规划核心实现
├─ pipelines/
│  ├─ interhub/
│  │  ├─ process_interhub.py   # 当前 InterHub CSV/pkl 批处理主入口
│  │  └─ tools/                # InterHub rerun、合并、报告辅助脚本
│  └─ simulation/
│     └─ simulator.py          # 交互仿真框架（Scenario/Simulator）
├─ docs/                       # 轻量说明文档
├─ reports/
│  ├─ studies/                 # 执行层：RQ/执行版本/report package
│  └─ knowledge/               # 判断层：review、synthesis、decision
├─ data/                       # 数据入口：README/manifest 可跟踪，raw payload 被忽略
├─ models/rq009_m3/            # Git 同步的冻结 M3 scorer、contract 与 SHA manifest
├─ configs/                    # 版本化实验参数 profile
├─ environments/               # estimator/verifier 分离环境锁
├─ archived/
│  ├─ compat_wrappers_20260619 # 已归档的旧根入口/旧 tools 兼容层
│  └─ legacy_scripts/          # 旧版数据脚本归档
├─ requirements.txt
├─ STUDIES.md
└─ main_workflow.log
```

根目录不再保留 `agent.py`、`ipv_estimation.py`、`process_interhub.py`、`simulator.py` 或 `tools/`。这些短期兼容包装已归档到 `archived/compat_wrappers_20260619/`，不再是活跃入口。

论文/Overleaf 工作区已拆分为独立仓库，不再是本项目的 `paper/` 子目录：

- 本地 clean clone：`../9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`
- 远端：`https://github.com/zxc-tju/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`

主项目只保留数据、代码、报告和分析管线；不要重新创建顶层 `paper/` 作为活跃论文入口。

研究知识库保留在本仓库，分两层；`reports/` 第一层只保留
`studies/` 与 `knowledge/`：

- `reports/studies/`：执行层，记录每个 RQ 的 execution、report package
  路径、evidence.csv、命令、环境和偏离计划情况。
- `reports/knowledge/`：判断层，记录 ChatGPT/Claude/Codex/human review、综合判断
  和最终 `decision.md`。

根目录 `STUDIES.md` 是所有研究问题的总索引。

大型原始数据已归到 `data/` 入口，但 raw 子目录继续被 git 忽略：

- InterHub 原始数据在 `data/interhub/raw/`。
- InterHub 派生大结果数据在 `data/derived/interhub/`。
- Onsite competition 原始回放/队伍材料在 `data/onsite_competition/raw/` 和
  `data/onsite_competition/top5_research_subset/teams/`。
- Argoverse 历史数据/结果在 `archived/argoverse/`。

---

## 3. 分层架构

### 3.1 核心算法层

- `src/sociality_estimation/core/agent.py`
  - 轨迹优化、IBR、IDM、IPV 估计、成本函数与约束构建。
- `src/sociality_estimation/core/ipv_estimation.py`
  - `MotionSequence`
  - `estimate_ipv_pair(...)`
  - `estimate_ipv_current(...)`
  - `RealtimeIPVEstimator`

新代码使用包路径导入：

```python
from sociality_estimation.core.agent import Agent
from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_pair
```

### 3.2 规划与工具层

- `src/sociality_estimation/planning/utility.py`
- `src/sociality_estimation/planning/lattice_planner.py`
- `src/sociality_estimation/planning/Lattice.py`

新代码使用：

```python
from sociality_estimation.planning.utility import smooth_ployline
from sociality_estimation.planning.Lattice import TrajPoint
```

### 3.3 数据管线层

- `pipelines/interhub/process_interhub.py`
  - 当前 InterHub CSV/pkl 主入口。
  - 默认 CSV/pkl/output 路径以仓库根目录为基准。
- `pipelines/interhub/tools/`
  - `build_missing_ipv_rerun_input.py`
  - `merge_subsets_for_yiru_ipv_archives.py`
  - `update_ipv_distribution_report.py`

推荐命令：

```bash
python pipelines/interhub/process_interhub.py --csv <input.csv> --pkl-root <pkl_dir>
```

### 3.4 IPV verifier 层

- `sociality_estimation.verifier.score_anchors(...)`：输出七个 quantile、
  80/90/95% CQR envelope 与 OOD/support abstention。
- `sociality_estimation.verifier.score_verifier(...)`：在 envelope 基础上输出
  规范的 raw signed/absolute exceedance。
- 默认 scorer bundle 位于 `models/rq009_m3/`；加载前按 `manifest.json`
  校验 SHA-256。模型类使用稳定包路径，不再依赖研究报告目录或本机绝对路径。
- IPV 生成与 verifier 使用隔离环境；见 `environments/README.md`。

### 3.5 仿真层

- `pipelines/simulation/simulator.py`
  - `Scenario`
  - `Simulator`
  - T-intersection、ramp merge、NDS 分析入口函数

注意：该模块仍引用外部 `NDS_analysis`，当前仓库快照不包含该模块；直接运行相关仿真入口可能失败。这是迁移前已有风险。

---

## 4. 核心对象与职责

### 4.1 `MotionSequence`

位置：`src/sociality_estimation/core/ipv_estimation.py`

- 数据结构：`[x, y, vx, vy, heading]`
- 额外字段：`target`（如 `lt_argo`、`gs`、`rt` 等），`reference`（可选参考线）
- 用途：作为数据脚本与 `Agent` 之间的标准输入容器。

### 4.2 `Agent`

位置：`src/sociality_estimation/core/agent.py`

- 规划相关：
  - `lp_ibr_interact(...)`
  - `solve_linear_programming(...)`
  - `ibr_interact(...)`
  - `solve_optimization(...)`
  - `idm_plan(...)`
  - `cruise_plan(...)`
- 状态维护：
  - `update_state(...)`
  - `trj_solution`
  - `observed_trajectory`
  - `action_collection`
- IPV 估计：
  - `estimate_self_ipv(...)`

### 4.3 `Scenario` / `Simulator`

位置：`pipelines/simulation/simulator.py`

- `Scenario`：双车初始状态与控制器类型容器。
- `Simulator`：执行交互循环、调用 `Agent` 控制器、可视化、语义结果判定、元数据导出。

---

## 5. 两条主业务流水线

### 5.1 InterHub 数据估计流水线

1. `pipelines/interhub/process_interhub.py`
2. `sociality_estimation.core.ipv_estimation.estimate_ipv_pair(...)`
3. `Agent.estimate_self_ipv(...)`
4. `Agent.solve_optimization(...)`
5. `utility_fun`、`cal_individual_cost`、`cal_group_cost`
6. planning utilities / bicycle model / reference-line tools

归档的 Argoverse CSV、InterHub JSON、subset 兼容包装和 `mean_ipv` metadata 后处理脚本仍在 `archived/legacy_scripts/`。恢复前必须核对硬编码路径和导入路径。

### 5.2 仿真分析流水线

1. 构造 `Scenario`。
2. `Simulator.initialize(...)` 实例化双车 `Agent` 并互设估计对手。
3. `Simulator.interact(...)` 按步循环调用控制器：
   - `linear-gt`
   - `gt` / `opt`
   - `idm`
   - `lattice`
   - `replay`
4. `get_semantic_result(...)` 判定语义结果并落盘可视化/表格。

---

## 6. 输入输出与路径约定

### 输入

- InterHub：CSV 索引 + pkl 事件数据。
- Onsite competition：本地大数据包与轻量 manifest 均在 `data/onsite_competition/`，其中 raw/team payload 被 git 忽略。
- 归档 Argoverse：旧脚本按脚本目录旁的 `0_souce_data/` 读取，当前历史数据位置需要按实际路径核对。
- 仿真附属资源：`background_pic/*`、`NDS_analysis`（当前仓库未包含）。

### 输出

- InterHub 派生大结果：`data/derived/interhub/...`
- 研究执行报告：`reports/studies/RQxxx_topic/RQxxx_n_topic_date/...`
- 研究判断记录：`reports/knowledge/RQxxx_topic/...`
- 报告过程归档：`archived/report_process/...`
- Argoverse 历史结果：`archived/argoverse/...`
- 仿真输出：仍按 `pipelines/simulation/simulator.py` 内部函数配置写入，部分旧函数使用相对 `../outputs/...`。

---

## 7. 并行与集群

- 本地并行：`pipelines/interhub/process_interhub.py` 使用 `ProcessPoolExecutor`。
- 集群并行：历史 Slurm 脚本保留在对应结果包的 `01_process/hpc_run_files/` 归档目录。
- 当前 full/subset CSV+pkl 任务应通过 `pipelines/interhub/process_interhub.py` 新入口恢复。

---

## 8. 当前工程注意点

- `pipelines/simulation/simulator.py` 仍依赖不在仓库内的 `NDS_analysis`。
- `archived/legacy_scripts/batch_process_ipv.py` 默认根目录是旧的 `interhub_traj_lane/ipv_estimation_results`，与当前输出结构不一致。
- `src/sociality_estimation/core/agent.py` 的全局参数（如 `dt`、`TRACK_LEN`）会影响估计与仿真行为，切换实验场景时应统一配置。
- 不要重新创建根目录 `agent.py`、`ipv_estimation.py`、`process_interhub.py`、`simulator.py` 或 `tools/` 作为活跃代码入口。

---

## 9. 建议阅读顺序

1. `src/sociality_estimation/core/ipv_estimation.py`
2. `src/sociality_estimation/core/agent.py`
3. `pipelines/interhub/process_interhub.py`
4. `pipelines/simulation/simulator.py`
5. `src/sociality_estimation/planning/Lattice.py`
