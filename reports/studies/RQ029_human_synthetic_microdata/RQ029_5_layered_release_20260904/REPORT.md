# RQ029-5：两层数据整理与证据权限报告

## 定位与结果

用户开头提到“三个层面”，但随后只定义了原始数据和分析/提炼数据两层，并明确要求
将“这两个层面”分开。因此本轮只实施两层，第三层记为 `NOT_SPECIFIED_BY_USER`，
没有擅自设计。

完整整理包位于：

`data/derived/rq029_human_synthetic_release/v1/`

整理状态为 **PASS / SYNTHETIC_NOT_OBSERVED**：释放包共 `161` 个文件，清单内文件 `155`，
逻辑字节约 `1.850 GB`。生成器优先尝试 APFS copy-on-write，并把实际复制方法写入
`release_manifest.json`；同时验证源/目标 inode 不同，排除了可反向影响源包的硬链接。

## 第一层：采集形态原始数据

目录：`01_collection_shaped_raw/`

该层只保留与采集/回放接口直接相关的文件，不混入论文 band、calibrated 结局或 claim 结论：

- `raw/`：20 名合成驾驶员的 session 负载，共 100 个文件。
- `shared_source_logs/`：上海 T11 源 monitor 与背景轨迹日志。
- `tables/runs.csv`：20×15 = 300 个 `driver_id × scenario_id` 运行。
- `tables/trajectory_pairs.parquet`：81,880 行逐帧主车/源主车/指定对手运动表。
- `scenario_templates.csv`、`ego_warp_metrics.csv`、`designated_counterpart_template.parquet`：
  场景、路径扰动和背景对手模板。
- `provenance/`：源 manifest 和源 README 快照。

**硬边界**：这一层只是“采集形态”的合成代理，不是真实受试者采集结果。
其 README、manifest 及 session notice 都保留 `SYNTHETIC_NOT_OBSERVED`。

## 第二层：分析、核心结果与论文支撑材料

目录：`02_analysis_and_paper_support/`

这一层没有把“真实论文依据”和“合成分析演练”混在一起，而是再分为四个区：

### 2.1 可直接支撑论文的权威聚合依据

`01_authoritative_real_aggregate/`

- `human_arm_data.json`：`REAL_VERIFIED` 人类臂聚合真值。
- `av_reference_values.json`：冻结 AV 对照聚合值。
- `RQ022_decision.md`：已接受 claim、措辞和不可外推边界。

上述三个文件是本包内唯一标记为 `DIRECT_SUPPORT` 的论文依据。

### 2.2 合成分析与流程演练

`02_synthetic_analysis_rehearsal/`

包含 candidate/both-gate/counterpart/per-unit 明细、run calibration、target/achieved summary 和
当前 `55/55` 硬检查验证包。这些表可支撑分析代码、图表、bootstrap、读取器与
论文流程演练，但证据等级固定为 `METHOD_REHEARSAL_ONLY`，不构成独立实证支撑。
`contracts/paper_record_targets_v2.json` 只是论文展示值的合成校准合同，证据等级为
`CALIBRATION_CONTRACT_NOT_EVIDENCE`，不放在真实权威依据目录。

### 2.3 审计链

`03_evidence_and_audits/`

保留 RQ029-2 论文对齐、RQ029-3 轨迹连续性及 RQ029-4 claim parity 报告、notebook、
明细与图片。当前结论是：

`POINT_ESTIMATES_EXACT / INTERVALS_PARTIAL / RAW_EMERGENT_MISMATCH /
FULL_DISTRIBUTION_NOT_IDENTIFIABLE`

### 2.4 核心结果与 claim 映射

`04_core_results/`

- `core_results.csv`：18 项最核心论文指标，带目标、calibrated/raw 值、分母、cluster 与状态。
- `claim_metric_comparison.csv`：28 项完整逐项对照。
- `claim_parity_summary.json`：机读对齐状态。
- `claim_evidence_map.csv`：每个产物的证据等级和论文可用性。

## 验证

深度验证 `33/33 PASS`，结果见 `00_metadata/validation_summary.json`，包括：

- 100/100 个 layer-1 raw 文件与现行 v2 源包文件集和字节内容一致。
- 两层关键表与源表 SHA-256 一致。
- layer 1 不存在 candidate/counterpart/achieved-summary 分析表混入。
- `human_arm_data.json` 仍为 `REAL_VERIFIED`。
- 任何 synthetic 产物都未被标记为 `DIRECT_SUPPORT`。
- 300 个运行、81,880 帧、清单唯一性、文件存在性和 155 项清单哈希全部通过。
- `MANIFEST.sha256` 的路径集、每条哈希与 `file_inventory.csv` 双向一致。
- 专项测试 `4 passed`，包括非规范目录的 `--replace` 拒绝且保留原文件测试。

机读验证：`00_metadata/validation_summary.json`；人读验证：
`00_metadata/VALIDATION_REPORT.md`。

## 数据可用性与第三层待定

`00_metadata/DATA_AVAILABILITY_LOCAL_DRAFT.md` 已区分真实聚合依据、未包含的受试者记录和合成表。
该文件只是本地整理草案，未声称公开仓库、DOI、许可证或受限访问流程已建立。

用户未定义的第三层没有创建。如果后续第三层指“投稿 Source Data/图表发布包”、
“可交互展示”或“真实受限数据归档”，需由用户明确后再单独设计。

## 产物

- 分层数据包：`data/derived/rq029_human_synthetic_release/v1/`
- 可复跑生成/验证：`pipelines/simulation/package_rq029_layered_release.py`
- 专项测试：`tests/test_rq029_layered_release.py`
- 文件清单：`data/derived/rq029_human_synthetic_release/v1/00_metadata/file_inventory.csv`
- 完整性清单：`data/derived/rq029_human_synthetic_release/v1/MANIFEST.sha256`
