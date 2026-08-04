已按硬停止条件停在 T3，没有进入 T4/T5。

T1/T2 已完成并落盘：
- `work/frame_index.csv`: 4,981,984 个 dev+guard 后热身锚点，SHA `6c7bd6...cf675`
- `work/sample_v1.csv`: 2,300/2,300 抽样完成，12 格无 shortfall，SHA `d27f10...66759e`
- `work/sample_v1.sha256` 已写入 frame/sample 两个 SHA
- `work/t1_summary.json`, `work/t2_summary.json`, `work/t3_t4_summary.json` 均已保留

T1 对齐结果：
- dev+guard 后热身精确零值：1,200,636 / 4,981,984 = 24.10%
- dev+guard 全体 `|ipv| < 1e-6`: 2,008,902 / 5,197,072 = 38.65%，这是最接近“四成”的口径
- `waymo_300-499.pkl` 文件存在但读取失败：`pickle data was truncated`，已按 `pkl_available=False` 排除抽样

T3 复现门失败，已写：
- `.codex-fleet/rq015b-repair/board/BLOCKED_B1.md`
- BLOCKED SHA `c0d79015a8f67ebffdeb835ea55be8cda7c14065ece60c5fd949e99d3ddbda13`

硬门数字：
- 门 A：12/40，通过标准是 >=39/40，所以失败
- 门 B：40/40，通过，最大权重差为 0
- 额外只读定位显示，完整 pair-call 与窗口单帧调用在失败锚点上给出相同重解结果，所以不是窗口化调用导致的失配
- 最可能分歧点已写入 `BLOCKED_B1.md`：当前本地生产路径/环境与存档主表之间存在求解器、参考线处理或生成 checkout 差异

最终自检：
- HEAD 仍是 `e82091ceaa2586bdb09b6153dfbed3be24d6bf98`
- `agent.py` SHA 未变
- `reliability_logdomain.py` SHA 未变
- 但当前 `git status --short` 非空，显示 RQ015A 相关并发/外部改动；这些不在 B1 输出范围内，我没有回滚它们。由于这一点，派发要求里的“git status 必须为空”不能认证为通过。

`anchor_mse.parquet/csv` 和 `B1_report.md` 未生成，因为 T3 是明确硬停门。

