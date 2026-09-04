# RQ029 人类驾驶模板包整理与真实采集补录清单

## 技术摘要

本项工作要解决的是：RQ029 的 20 名驾驶人 × 15 场景微观数据由聚合统计约束生成，旧包的文件夹、车辆名、run ID、状态列、共享日志和重复 recordId 都会触发模板导入拒绝。本轮不修改用于论文统计校准的 v1/v2 历史包，而是新建一个模板专用包：

`data/derived/rq029_human_template/v1/`

新包仅在显式 `TEMPLATE_MODE` 下验证通过，生产模式返回 `TEMPLATE_MODE_REQUIRED`。根 README 穷举全部子目录，并明确说明原始轨迹、派生表和分析表全部为合成数据。机器可读根来源文件使用 `GENERATED_NOT_OBSERVED` 和 `contains_observed_human_trajectories=false`，子目录、占位 ID 和逐行数据中不再使用 `synthetic` 作为标签或占位值。

## 新模板包的两个数据层

1. `01_collection_raw/`：20 个驾驶人会话，每个会话只有 `monitor.log`、`simulation_trajectory.log`、`vehicle_perception_simulation_trajectory.log` 和 `vehicle_trajectory.log` 四类真实 schema 兼容日志；另含 raw 规范表。
2. `02_analysis_support/`：保留分析链演练所需的 candidate、gate、band、counterpart 和汇总表。这些表在真实逐帧数据上重算之前，不是论文实证支撑材料。

`00_control/` 是模板控制元数据，不是第三个数据层。它包含占位符注册表、必填真实字段合同、导入模式合同、文件清单、SHA-256 和验证报告。

## 旧包的明确异常已修复

修复前后对比见 `template_quality_comparison.csv`：

- 20 个 session 的唯一 `taskId`：`1/20 → 20/20`。
- 唯一 `recordId`：`1/20 → 20/20`。
- 唯一 ego vehicle ID：`1/20 → 20/20` 保留占位值。
- 唯一 `monitor.log` 哈希：`1/20 → 20/20`。
- 唯一 `simulation_trajectory.log` 哈希：`1/20 → 20/20`；差异来自会话时间轴，背景运动数值不变。
- 含明显生成标签的 session 目录：`20/20 → 0/20`。
- session 通知文件：`20/20 → 0/20`。
- 含明显生成前缀的 run ID：`300/300 → 0/300`。
- 含明显生成前缀的 ego name：`20/20 → 0/20`。
- `runs.csv` 的逐行数据状态列已移除；包级来源边界仅放在根目录控制文件。

## 需要用真实采集信息填写的内容

完整机器可读清单位于：

`data/derived/rq029_human_template/v1/00_control/required_real_collection_fields.csv`

### A. 可以通过真实台账填充的标识

- 20 名驾驶人的匿名 `driver_id`；或者由研究负责人正式确认继续使用 `D01–D20`。
- 每个 session 的真实 `taskId`、`recordId` 和 `<taskId>-<recordId>` 目录名。
- 驾驶人与 session 的对应关系，包括补采或重跑。
- 真实车辆或注入平台的 ego `id` 和 `name`。
- 真实车辆的 `vehicleType`、`driveType`、长宽高、牌照地区/类型与颜色配置。
- 采集场地、路线版本、设备版本和软件版本。

### B. 不能只改标签，必须用真实日志整体替换

- 真实会话开始/结束时间，以及全部逐帧 `timestamp/globalTimeStamp/frameId`。
- `monitor.log` 全文，包括 CPU、内存、磁盘、网络、FPS、系统状态和 AV monitor 运动值。
- `vehicle_trajectory.log` 的 ego 位置、速度、加速度、航向、方向盘、踏板、刹车和档位。
- `vehicle_perception_simulation_trajectory.log` 的真实 ego 和感知记录。
- 若真实人类实验没有复用同一回放背景，`simulation_trajectory.log` 和背景交通流也必须整体替换。

只替换 A 类 ID，不能把 B 类生成轨迹和监控量登记为真实采集。

### C. 必须由试验负责人确认的映射

- `scenario_id ↔ native_case_id ↔ caseName` 权威口径。现有项目字典与 15 个上海原始 AV session 的非空 `caseName` 在 `11/15` 个场景上语义不一致；详见 `scenario_label_crosscheck.csv`。
- 论文统计中 186 个真实 case 的逐行映射。该映射无法从现有聚合统计恢复。
- 是否真实共用同一辆车和同一回放背景；如果有台账证据，vehicle ID 或背景运动重复可以是合理的。

### D. 必须在真实行级数据上重算

- `run_id`、`candidate_key` 及所有规范化原始表。
- candidate/gate/band 状态、`ipv_log`、TTC、对手运动签名和急刹统计。
- `per_unit_count.csv`、bootstrap 分组、置信区间、论文统计表和图源数据。

## 现有真实数据能提供哪些值

### 真实人类数据

当前仓库只有已验证的人类聚合结果，没有人类逐帧采集档案。已有值包括：

- `n_drivers=20`、`n_runs=300`、20×15 口径和 0 缺失。
- candidate/gate/band 聚合计数、15 场景统计和论文运动签名。
- provenance：`generated_by=X. Zhao`、`generation_date=2026.08.19`、`source_machine=offline-svr-510`。
- 受控档案引用：`per_unit_count.csv` 与 `run_human_arm_ana_.py`；这两个文件当前未在仓库中找到。

人类驾驶人 ID、task/record/session ID、车辆信息、采集时间、monitor 和逐帧轨迹均不可用。逐字段状态见 `human_observed_data_availability.csv`。

### 真实上海 AV 数据

现有实际上海 AV 包提供 15 个 session，覆盖 T1–T12 队伍，其中 T4、T10 和 T12 各有两个 session。这些值可用于确认格式，不能作为人类驾驶会话的真实值。完整列表位于 `existing_observed_shanghai_sessions.csv`。已验证的格式规律是：

- session 目录名为 `<4 位 taskId>-<10 位 recordId>`。
- 目录前半段与 `monitor.log.taskId` 一致，后半段与 `monitor.log.recordId` 一致。
- vehicle ID 是数字字符串，vehicle name 是自由文本。
- 原始日志使用 `taskId`、`recordId`、`caseId` 和 `frameId`；`run_id` 是后续派生键。

## 验证结果

- 完整深度验证：`74/74 PASS`。
- 包体：20 sessions、300 unique runs、81,880 frames。
- 四类日志的唯一哈希：每类 `20/20`。
- 背景车运动：去除 ego、会话标识和绝对时间后，20 个会话与上海 T11 参考背景的语义哈希一致。
- 主车运动：经纬度、速度、航向和速度分量与 v2 逐行一致；仅身份和绝对时间轴变化。
- 清单：100 个 managed files，actual files、`file_inventory.csv` 和 `MANIFEST.sha256` 三方双向一致。
- 回归测试：RQ029 基础生成、模板、分层包和论文对齐测试合计 `29 passed`。
- 独立逐帧复算：frame gap run `0/300`，非正 dt run `0/300`，最大相邻位移 `1.053354 m`，12 个 motion 列与 v2 差异均为 0。
- 独立完整性复算：actual/inventory/MANIFEST 均为 100 项，互差和哈希 mismatch 均为 0。
- 代码复审：首轮发现 `--replace` 可删除未登记人工文件；修复为删除前校验 actual/inventory/MANIFEST/逐文件哈希，并新增 fresh-build 回归后，复审结论 `APPROVE`。
- 生产模式：显式拒绝，退出码 1，状态 `TEMPLATE_MODE_REQUIRED`。

## 使用边界

新包解决的是 schema、路径、占位符、日志唯一性和模板模式问题，不是人类实测数据恢复。仅根据聚合统计无法恢复真实个体轨迹、真实个体效应、联合分布、真实 186-case 映射或 frozen estimator 在真实行上的输出。

后续填充脚本的具体输入表头、处理顺序和失败关闭检查见 `FILL_SCRIPT_SPEC.md`。
