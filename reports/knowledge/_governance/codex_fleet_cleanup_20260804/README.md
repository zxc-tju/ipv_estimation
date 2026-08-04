# `.codex-fleet/` 大块中间数据清理记录（2026-08-04）

基线提交 `102b094`。本次把 `.codex-fleet/` 从 **6.5 GB 降到约 0.3 GB**，
删除的全部是 `.gitignore` 覆盖的中间产物，**没有删除任何脚本、任务书、报告或证据 JSON**。

## 为什么这些可以删

删除前对每一条做了引用检查（`rg -l --fixed-strings`，只取文件名不取内容行）。
结论：**所有引用都出现在归档的治理记录里**（历史 prompt 与报告在描述"当时跑了什么"），
**没有一条是活脚本的运行时依赖**。两处需要单独说明：

- `frame_index.csv` / `anchor_meta.csv` / `sample_candidates.sqlite`
  只被 `.codex-fleet/rq015b-repair/work/run_b1_rq015b.py` 触及，而该脚本是它们的
  **生成者**（`run_b1_rq015b.py:282-283` 以 `open("w")` 写出，:472-473 再读回自校验）。
  即它们是 B1 阶段的产物，不是外部输入。
- `k2_fullcorpus/inputs/` 是 K2 那次 Slurm array 作业（`--array=1-460%427`）的
  460 个分片投喂目录。**该作业已完成，产物完好**：
  `data/derived/rq015k_logdomain_gate/l1_v1/` 共 510 个 parquet 分片、1.7 GB。

## 删除清单

| 路径 | 大小 | 文件数 | 再生方式 |
|---|---:|---:|---|
| `.codex-fleet/rq015b-repair/work/sample_candidates.sqlite` | 2.6 GB | 1 | 重跑 `run_b1_rq015b.py` |
| `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/inputs/` | 2.4 GB | 460 | 由 `k2_fullcorpus_materializer.py` 重新 staging |
| `.codex-fleet/rq015b-repair/work/frame_index.csv` | 634 MB | 1 | 重跑 `run_b1_rq015b.py` |
| `.codex-fleet/rq015b-repair/work/anchor_meta.csv` | 356 MB | 1 | 重跑 `run_b1_rq015b.py` |
| `.codex-fleet/rq016c-human-only-envelope/work/H1/envelope_model/` | 164 MB | 8 | 重跑 `H1/run_rq016c_h1_human_only_envelope.py` |
| `.codex-fleet/rq015b-repair/work/pycache/` | 27 MB | 1,269 | Python 自动重建 |

## 明确保留（承重，不可删）

- `.codex-fleet/rq016c-human-only-envelope/work/H2/` —— **RQ017 依赖**。
  `envelope_model/rq016c_h2_envelope.pkl`（172 MB）是机制二的持久化模型；
  `onsite_scoring_dryrun.parquet`（2.7 MB）是 RQ017 机制二交叉的连接源。
- `.codex-fleet/rq015b-repair/work/run_b1_rq015b.py` 与 `run_b2_rq015b.py`
  —— 后者的 `loads_array` / `solve_anchor_task` 被 K2 materializer 直接 import。
- `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/k2_fullcorpus_materializer.py`
  —— 冻结的机制一门实现。
- `.codex-fleet/git_cleanup_protected_sha_before.txt` —— 受保护文件基线校验清单。
- `.codex-fleet/rq015a-run/board/detach_launch.py` —— 派发脱离进程组用（macOS 无 `setsid`）。
- 所有 `board/` 下的报告、prompt、任务书；所有 `work/M1/*.json` 证据文件。

## H1 的特殊说明

H1 是 RQ016C 的**第一次尝试，已被判定不可用**：它把 `vehicle_type_list` 留在类别 context 里，
而纯人-人参照池没有 `AV` 取值、真实 OnSite 行全部带 `['AV','HV']`，
所以 H1 的持久化产物无法用于它唯一的外部打分用途（见
`reports/studies/RQ016C_human_only_envelope/.../RQ016C_2_human_only_envelope_fixed.md:5`）。

因此只删除 H1 的 164 MB `envelope_model/`，**保留** H1 的脚本、`key_numbers.json`
和 selftest parquet（合计约 130 KB），错误历史的证据链不受影响。
H1 报告 `RQ016C_1_human_only_envelope.md` 的「持久化模型」一节已加注说明产物已删除。
