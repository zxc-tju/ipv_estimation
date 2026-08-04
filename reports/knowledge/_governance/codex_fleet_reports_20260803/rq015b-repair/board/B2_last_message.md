完成。主要产物已写入：

- [B2_report.md](<.codex-fleet/rq015b-repair/board/reports/B2_report.md>)
- [anchor_mse.csv](<.codex-fleet/rq015b-repair/work/anchor_mse.csv>)
- [known_issue_snippet_repro_gate.md](<.codex-fleet/rq015b-repair/board/reports/known_issue_snippet_repro_gate.md>)

关键结果：T5 解算 2,300/2,300，0 异常，6-thread executor，用时 1240.3s；串行一致性检查 n=24，max diff=0。平价门通过：eligible 1526/2300，max weight diff `3.75e-15`。log 域不可归零 pytest 通过。

机制拆分主结果（本地可达 post-warm U∪Z，HT 分母 534,939）：D1 `43.01% [39.35%, 46.83%]`，D2 `39.48% [35.69%, 43.08%]`，D3/D4 均 `0%`；log 域后权重实质非均匀 `53.62% [50.12%, 57.46%]`。CI 半宽最大 `3.74 pp`，已在报告中标注未过 `≤3 pp` 精度上限。

完工自检通过：HEAD 仍为 `e82091ceaa2586bdb09b6153dfbed3be24d6bf98`；`agent.py` 与 `reliability_logdomain.py` SHA 均匹配指定值。未写 root `main_workflow.log`，因为本轮铁律限定全部输出只能写入 `.codex-fleet/rq015b-repair/`。

