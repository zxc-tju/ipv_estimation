# RQ029 真实采集补录脚本合同

本文件用于指导后续填充脚本。目标不是通过替换标签将生成轨迹改名为实测数据，而是将真实采集系统导出的身份、会话、逐帧日志和分析结果按模板结构物化到一个新目录。

## 建议输入文件

### `session_identity_mapping.csv`

每个真实 session 一行，至少包含：

```text
driver_id,task_id,record_id,session_id,vehicle_id,vehicle_name,
collection_start_timestamp_ms,collection_end_timestamp_ms,
vehicle_type,drive_type,length,width,height,pic_license,
site,route_version,device_version,software_version,real_log_directory
```

其中 `session_id` 必须等于 `<task_id>-<record_id>`；`real_log_directory` 必须指向真实采集导出目录，且必须含四类日志。

### `scenario_mapping.csv`

```text
scenario_id,native_case_id,case_name,mapping_authority,protocol_version
```

当前项目字典与上海原始日志的 `caseName` 有 11/15 语义冲突，因此填充脚本不得默认其中任一方为真值，必须要求显式的 `mapping_authority` 和 `protocol_version`。

## 四类处理动作

### 1. 可根据真实台账替换的标识

- `driver_id`
- `taskId`
- `recordId`
- session 目录名
- ego `id`
- ego `name`

这些字段应同步写入 session 目录、四类日志、`runs.csv`、逐帧表和分析表，不能只改某一份文件。

### 2. 必须用真实导出整体替换的流

- `monitor.log`：系统帧率、CPU、内存、磁盘、网络、位置和速度监控。
- `vehicle_trajectory.log`：ego 位置、速度、加速度、航向、方向盘、踏板、刹车和档位。
- `vehicle_perception_simulation_trajectory.log` 的 ego 与感知内容。
- 如果真实协议没有重用固定回放背景，`simulation_trajectory.log` 和背景 actor 也必须整体替换。

补录脚本不得将模板日志中的坐标、速度、控制量或监控量仅通过更名保留为“真实采集值”。

### 3. 需要试验负责人确认的口径

- 上海的具体场地、路线与版本。
- 20 名驾驶人是否全部完成同一组 15 场景。
- `scenario_id ↔ native_case_id ↔ caseName` 权威映射。
- 20 人是否共用同一辆车；如果共用，vehicle ID 和长宽高可以合理重复。
- 是否存在补采或重跑；脚本不得强行压缩为每人一个 session。
- 背景交通是否确实使用同一回放流。

### 4. 必须在真实逐帧数据上重算的字段

- `run_id` 和 `candidate_key`。
- candidate/gate/band 状态。
- `ipv_log`、TTC、对手减速、速度极差和急刹统计。
- 真实 186-case 聚类映射与 bootstrap 分组。
- `per_unit_count.csv`、论文统计表、图源数据和置信区间。

## 补录脚本的建议顺序

1. 只读加载模板，向一个新的输出根目录物化，不覆盖模板包。
2. 校验真实 session 台账的必填列、主键、唯一性和目录存在性。
3. 验证每个真实 session 的四类日志都是合法 JSONL，并校验目录名与 `taskId/recordId` 一致。
4. 整体复制真实日志，不把生成坐标、监控量或控制量混入。
5. 从真实日志重建规范化原始表，再衍生 `run_id` 和 frame key。
6. 在真实行级数据上重跑冻结 estimator 与统计分析，不复制模板的 calibrated 列。
7. 重新生成 `file_inventory.csv`、`MANIFEST.sha256`、导入收据和数据字典。
8. 只有在所有 `FULL_STREAM`、`CONFIRMATION`、`RECOMPUTE` 门均通过后，才能在新包中声明来源为真实采集；同时保留模板到真实包的处理收据。

## 生产导入前的失败关闭检查

- 保留占位值 `7901–7920`、对应的保留 `recordId`、`8001–8020`、`VEHICLE_Dxx` 或 `NEEDS_REAL_INPUT` 任一项：失败。
- 会话目录与日志 `taskId/recordId` 不一致：失败。
- 时间戳不在真实采集窗口，或逐帧时间不单调：失败。
- 真实日志缺失、空文件或 JSONL 无法解析：失败。
- 应独立的 `recordId/session_id` 重复：失败。
- 车辆 ID 重复但没有共用同一实车的台账证据：失败。
- 背景日志完全相同但协议没有记录固定回放：失败。
- 仍复制模板 analysis/calibrated 列而未在真实行级数据上重算：失败。
- actual files、inventory 和 SHA manifest 不双向一致：失败。
