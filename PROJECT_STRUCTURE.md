# 2_sociality_estimation 项目结构文档

## 1. 项目定位
本项目围绕“交互偏好值（IPV, Interaction Preference Value）”展开，包含两条主线：
- 轨迹数据离线估计：从数据集轨迹中估计双车 IPV 序列并导出结果。
- 交互行为仿真分析：基于博弈/规则控制器进行双车交互仿真与分析。

核心是 `Agent` 的规划与 IPV 估计能力，数据脚本和仿真脚本都在调用这一核心能力。

---

## 2. 目录与文件总览（当前仓库）
```text
.
├─ agent.py                     # 核心智能体：规划、博弈、IPV估计
├─ simulator.py                 # 交互仿真框架（Scenario/Simulator）
├─ ipv_estimation.py            # 面向数据管线的 IPV 估计封装
├─ process_argoverse.py         # Argoverse 批处理入口
├─ process_interhub.py          # Interhub JSON 批处理入口
├─ batch_process_ipv.py         # 二次统计：回写 mean_ipv 到 metadata
├─ submit.sh                    # SLURM 集群批处理脚本（Interhub）
├─ requirements.txt             # 依赖列表
├─ main_workflow.log            # 工作流日志（仓库要求持续追加）
└─ tools
   ├─ utility.py                # 几何/平滑/运动学通用工具
   ├─ lattice_planner.py        # Lattice 规划入口封装
   └─ Lattice.py                # Lattice 规划核心实现
```

---

## 3. 分层架构

### 3.1 入口层（脚本）
- `process_argoverse.py`
- `process_interhub.py`
- `batch_process_ipv.py`
- `simulator.py`
- `submit.sh`

负责数据加载、任务分发、并行执行、结果落盘。

### 3.2 核心算法层
- `ipv_estimation.py`：把 `Agent.estimate_self_ipv` 封装成可复用的双车估计函数 `estimate_ipv_pair(...)`。
- `agent.py`：包含轨迹优化、IBR、IDM、IPV 估计、成本函数与约束构建。

### 3.3 工具与底层规划层
- `tools/utility.py`：参考线平滑、几何操作、简化运动学模型。
- `tools/lattice_planner.py` + `tools/Lattice.py`：Lattice 路径规划与避障。

---

## 4. 核心对象与职责

### 4.1 `MotionSequence`（`ipv_estimation.py`）
- 数据结构：`[x, y, vx, vy, heading]`
- 额外字段：`target`（如 `lt_argo`、`gs`、`rt` 等），`reference`（可选参考线）
- 用途：作为数据脚本与 Agent 之间的标准输入容器。

### 4.2 `Agent`（`agent.py`）
- 规划相关
  - `lp_ibr_interact(...)`：线性化 IBR
  - `solve_linear_programming(...)`：线性规划求解
  - `ibr_interact(...)` + `solve_optimization(...)`：非线性博弈求解
  - `idm_plan(...)` / `cruise_plan(...)`：规则控制
- 状态维护
  - `update_state(...)`
  - `trj_solution`、`observed_trajectory`、`action_collection`
- IPV 估计
  - `estimate_self_ipv(...)`：枚举虚拟 IPV 候选，比较观测轨迹与虚拟轨迹相似性，输出 `ipv` 与 `ipv_error`。

### 4.3 `Scenario` / `Simulator`（`simulator.py`）
- `Scenario`：双车初始状态与控制器类型容器。
- `Simulator`：执行交互循环、调用 Agent 控制器、可视化、语义结果判定、元数据导出。

---

## 5. 两条主业务流水线

## 5.1 数据估计流水线（Argoverse / Interhub）

### 公共核心（`ipv_estimation.py`）
1. `estimate_ipv_pair(primary, counterpart, ...)`
2. 对每个时刻 `t`：
   - 用历史窗口切片轨迹；
   - 分别构造 primary/counterpart 两个 `Agent`；
   - 调用 `Agent.estimate_self_ipv(...)`；
   - 记录 `ipv_values[t, :]` 与 `ipv_errors[t, :]`。
3. 可选返回 diagnostics（虚拟轨迹、权重、选定步）。

### Argoverse 管线（`process_argoverse.py`）
- 数据组织：按 `ARGO_CONFIG` 的数据版本/场景/子集迭代。
- 处理步骤：
  - 读取 case CSV（含轨迹与参考线）；
  - 构建 `MotionSequence`；
  - 调用 `estimate_ipv_pair`；
  - 导出 `*_ipv_results.xlsx` 与 `*_ipv_curve.png`；
  - 若启用调试，导出虚拟轨迹图。
- 并行：`ProcessPoolExecutor`（`max_workers` 控制）。

### Interhub 管线（`process_interhub.py`）
- 输入：`interhub_traj_lane/trajectory_data_*.json`
- 处理步骤：
  - `_load_dataset`：清洗与按 scenario/vehicle 重组；
  - `_select_vehicle_pair`：优先 AV-HV，其次 HV-HV；
  - `_classify_heading`：按航向变化分类 `lt/rt/gs`；
  - `_build_motion_sequence` + `estimate_ipv_pair`；
  - 输出 Excel、IPV 曲线、metadata；
  - 对跳过样本生成诊断图。
- 并行：`ProcessPoolExecutor`（支持 `--workers`）。

---

## 5.2 仿真分析流水线（`simulator.py`）
1. 构造 `Scenario`（初始位置/速度/航向/IPV/控制器类型）。
2. `Simulator.initialize(...)` 实例化双车 `Agent` 并互设估计对手。
3. `Simulator.interact(...)` 按步循环：
   - 每车按 `conl_type` 选择控制器：
     - `linear-gt`（线性 IBR）
     - `gt` / `opt`（非线性博弈/单次优化）
     - `idm`
     - `lattice`
     - `replay`
   - `update_state(...)` 更新状态与轨迹历史。
4. 后处理：
   - `get_semantic_result(...)` 判定语义（yield/rush/crashed 等）；
   - 结果可视化与 Excel 落盘。

---

## 6. 调用关系（核心链路）
```text
process_argoverse.py / process_interhub.py
            -> ipv_estimation.estimate_ipv_pair
            -> Agent.estimate_self_ipv
            -> Agent.solve_optimization
            -> utility_fun + (cal_individual_cost, cal_group_cost)
            -> bicycle_model / reference-line tools
```

```text
simulator.py
   -> Simulator.interact
      -> Agent.lp_ibr_interact / Agent.ibr_interact / Agent.idm_plan / lattice_planning
      -> Agent.update_state
```

---

## 7. 输入输出与路径约定

### 7.1 输入
- Argoverse：代码当前按脚本目录下 `0_souce_data/` 读取。
- Interhub：`interhub_traj_lane/trajectory_data_*.json`。
- 仿真附属资源：`background_pic/*`、`NDS_analysis`（当前仓库未包含该模块文件）。

### 7.2 输出
- Argoverse：`1_experiment_result/ipv_estimation/...`
- Interhub：`interhub_traj_lane/ipv_estimation/...`
- 选择失败诊断：`interhub_traj_lane/diagnostics_selection_skipped/...`
- 仿真输出：`simulator.py` 中按函数内路径写入 `../outputs/...`

---

## 8. 并行与集群
- 本地并行：`process_argoverse.py`、`process_interhub.py`、`batch_process_ipv.py` 都使用 `ProcessPoolExecutor`。
- 集群并行：`submit.sh`
  - SLURM array（`--array=0-6`）映射多个 JSON；
  - 每任务 `--cpus-per-task=96`；
  - 调用 `python process_interhub.py --workers "$WORKERS" "$TARGET"`。

---

## 9. 当前可见的工程注意点
- `simulator.py` 依赖 `NDS_analysis`，但该文件不在当前仓库内，直接运行相关入口可能失败。
- `batch_process_ipv.py` 默认根目录是 `interhub_traj_lane/ipv_estimation_results`，与 `process_interhub.py` 的 `OUTPUT_ROOT=interhub_traj_lane/ipv_estimation` 命名不一致，使用前需核对目录。
- `agent.py` 的全局参数（如 `dt`、`TRACK_LEN`）会影响估计与仿真行为，切换实验场景时应统一配置。

---

## 10. 建议的阅读顺序
1. 先看 `ipv_estimation.py`（最清晰的公共估计接口）。
2. 再看 `agent.py`（理解 IPV 的本体逻辑）。
3. 然后看 `process_interhub.py` / `process_argoverse.py`（数据管线）。
4. 最后看 `simulator.py` + `tools/Lattice.py`（仿真与规划细节）。
