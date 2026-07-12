# RQ014 PI scoped decision — v1.5 G2 staged start

日期：2026-07-12。来源：PI/用户在 RQ014 任务中明确指示“开始执行”。

## 首个 operation

在且仅在 v1.5 formal G1 artifact 为 `FORMAL_G1_PASS`、reviewed bytes 未漂移、中央授权文件列出
同名 operation、managed launcher 的全部 fail-closed 检查通过时，授权：

```text
rq014_g2_declassification_export
```

该 operation 只允许 checksum-bound 地读取以下三个 source classes：八个
`phase1_post_scene_bundle.pkl`、`rated479_segment_readiness.tsv`、
`selected_counterpart_tracks.csv`。它不得读取 raw TFRecord、scored target、ratings CSV 或任意评分值；
输出只能是 `RQ014_score_stripped_schema_v1.json` 允许的 canonical CSV/JSON bundle。

## 第二个 operation

`rq014_g2_contract_preflight` 已在 reviewed execution contract 中以 exact export receipt、`DONE.json` 与
run-spec refs 为 predicate 条件式注册，但中央 allowlist 当前不列该 operation，故仍 DENY。只有 export job
`COMPLETED/0:0`、score-stripped validator PASS、output/receipt hashes 封存、bounded report 无 blocker 后，
才可把它加入中央 allowlist；authority byte 变化必须重建 candidate manifest、双路 review、formal G1 与 final
bundle，但不再修改 execution-contract status。本决定不能替代中央授权更新和上述 fresh review。

## 共同允许

- 校验 v1.5 方案、registry、review、checksum 与 exact published Git commit；
- 只通过 `configs/run_specs/README.md` 与 `RQ014_execution_contract_v1p5.json` 共同冻结的单条
  checksum-bound clean-environment bootstrap 使用固定 entrypoint/resource/path；任何 direct-wrapper
  shorthand 均不构成授权；
- bootstrap 必须锁定 inherited fd8 runtime lock、以 fd9 打开并 hash/执行同一个 exact wrapper；wrapper
  不得 fallback 创建 descriptor，launcher 必须在 RQ014 dependency preload 前核验 fd8/fd9 target 与
  path/fd identity。该 gate 是 local provenance check，不是抵御同账号主动仿造的密码学秘密；
- 写入托管 score-stripped input root 或一次 bounded preflight receipt/report；
- 记录 source artifact hashes，但不记录其 rating-bearing parent absolute paths 或评分内容。

## 共同禁止

- 读取、挂载、传输或统计 row-level rating/preference/ranking 字段；
- 在 G2 重算 A1–A4 rho；
- 让 G2 挂载 raw `rated479_segments`、pickle 或 TFRecord；
- 使用 `/share/home/u25310231/ZXC/ipv_estimation`、`/share/home/u25310231/ZXC/RQ010B_wod_e2e/code`
  或任意非 commit-addressed checkout 执行代码；
- 直接调用 `sbatch`；
- resource pilot、G2R feature build、G3R full-rating recovery screen、G4R clean replay、旧 blind
  build/G2P power、optional validation、extension 或 forensic compute。

## 后续授权

`rq014_g2_resource_pilot`、`rq014_r2_blind_feature_build`、
`rq014_r3_full_rating_join_and_rank` 与 `rq014_r4_clean_replay` 均维持 `DENY`。它们必须分别在
preflight/pilot/budget、完整评分盲 feature freeze、独立评分授权、唯一 recipe freeze 后形成新的 PI 决定
和中央 allowlist 变更。旧 `rq014_g2_blind_build`/`rq014_g2p_power_simulation` 退为 optional non-gating；
任何 rating-bearing operation 当前均未获中央授权。

## 与 2026-07-11 决定的关系

`RQ014_PI_decision_G0_waiver_launch_20260711.md` 的 G0 waiver 意图及 residual-risk 接受继续有效；
其中非法组合状态、self-declared formal G1、宽泛布尔授权和旧 HPC 路径由 v1.5 取代。
