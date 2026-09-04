# RQ029：20×15 上海人类轨迹合成微观数据

本研究问题只生成用于工程联调的 `SYNTHETIC_NOT_OBSERVED` 代理数据，不恢复、替换或声称代表
缺失的真实人类逐帧轨迹。

- 计划与目标：`reports/plans/RQ029_plan_v0_human_synthetic_microdata_20260902.md`、
  `reports/plans/RQ029_human_statistical_targets_v1.json`
- 首次执行：`RQ029_1_shanghai_synthetic_20260902/`
- 论文记录微调：`RQ029_2_paper_aligned_20260902/`，目标合同为
  `reports/plans/RQ029_paper_record_targets_v2.json`
- 轨迹连续性独立审计：`RQ029_3_trajectory_continuity_audit_20260904/`，结论为
  `PASS_WITH_CAVEATS`；parquet 层无 frame gap、时间倒退或坐标瞬移，但非严格 10 Hz，
  少数固定曲率段有较高 jerk / yaw-rate 峰值。
- 论文 claim 逐项对齐审计：`RQ029_4_paper_claim_parity_audit_20260904/`；结论为
  `POINT_ESTIMATES_EXACT / INTERVALS_PARTIAL / RAW_EMERGENT_MISMATCH /
  FULL_DISTRIBUTION_NOT_IDENTIFIABLE`。
- 两层数据整理与证据权限包：`RQ029_5_layered_release_20260904/`；数据入口为
  `data/derived/rq029_human_synthetic_release/v1/`。
- 真实导入前的模板包：`data/derived/rq029_human_template/v1/`；它保留占位符注册表、
  补录字段合同和模板级验证结果，供后续填充脚本直接替换真实采集信息。
- 当前推荐的本地大型数据：
  `data/derived/rq029_human_synthetic_microdata/v2_paper_aligned/`；v1 保留用于对照。
