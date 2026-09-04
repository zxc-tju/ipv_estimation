# RQ029-4 Chart Map

| 文件 | 问题 | 图形 | 字段 | 支持的结论 | 编码/QA |
|---|---|---|---|---|---|
| `figures/claim_gap.png` 左 | 5 组 CI 离事前容差边界有多远 | 分组柱图 | metric, lower/upper endpoint error, tolerance | 只有 alpha90 两端超容差；其他 4 组在容差内但不是端点精确 | 蓝/橙区分下/上端，黑色虚线=1倍容差；轴从0起始，已目视确认标签无截断 |
| `figures/claim_gap.png` 右 | 论文点估计的一致性是否从 raw 轨迹自然产生 | 水平柱图 | metric, raw/calibrated relative error | calibrated 点误差为0，raw TTC 与急刹差异明确 | 橙色=raw，蓝色=calibrated；指标名直接标注，已目视确认图例与尺度正确 |
