# RQ014 v1.5 formal G1 round-4 remediation

日期：2026-07-12  
状态：第四轮整改已闭合，等待第五轮 fresh formal review；本文不是 G1 PASS，也不授权 HPC。

## 冻结的 BLOCKED 证据

第四轮 fresh formal review 读取 60-row candidate manifest
`RQ014_plan_v1p5_review_manifest_round4_blocked_20260712.sha256`，其 SHA-256 为
`fe07f81aea9c4b2afe8d5437370a337bd970f566a954eb4b0ad32169d7987391`。该轮原字节结论为：

- statistics：`BLOCKED`，0 blocker / 1 major / 1 minor；artifact SHA-256
  `09f5908beeabb329e0bc708c928d48a3f90ee6d220709d889f414ee1f77b1899`；
- execution/governance：`BLOCKED`，0 blocker / 1 major；artifact SHA-256
  `fc2b3cbd08b90f194ded9b1775bc73d841d8656eb36adfd02542758278b15a85`。

两份 review 与 manifest 是 append-only provenance；不得改写成 PASS，第四轮 reviewer 不参与下一轮
formal G1 计票。

## Finding-to-fix map

| Finding | 冻结风险 | 本轮整改目标 |
|---|---|---|
| STAT4-M01 | `required_envelope_queries.csv` 只有 aggregate key/count，却被合同宣称为逐 scene 冻结 anchor 域；不同 scene→tau 分配可产生同一聚合表 | 增加 checksum-bound per-scene anchor-membership artifact，冻结 `segment_id/feature_id/horizon_id/tau_tick/h_common_tick` exact key、排序、编码、terminal/empty 规则；aggregate query 表只能由该 artifact 唯一 group-by 生成。WOD feature/readout、BM90、BT90 均只消费同一 per-scene artifact，禁止独立重算。 |
| EG4-M01 | 调用者可直接传 `--rq014-only` 进入 v2 submit gate，Python launcher 未验证 wrapper/fd8 来源；materializer/preflight 在后续正式校验前已 import | 在任何 RQ014 dependency preload 前验证由唯一 wrapper 继承的 fixed descriptor capability：runtime lock fd 与 exact wrapper fd 的路径、regular-file identity、dev/inode/mode 必须匹配固定 managed paths；缺失/替换/直接 flag 均在 dependency import 前拒绝。`_validate_cli_entry_mode` 也要求已验证 capability，不能让 boolean 自行授权。 |

第四轮的 `STAT4-m01` 作为接受的 same-dataset stability residual 保留：`scenario_cluster=NA` scenes
仍进入主 association，但不属于具名 LOCO cluster，source-shard stability 也不是 recovery-compatible
硬门。合同已要求至少两个 eligible named clusters、fold 与 leave-one-scene 稳定性，并把最终标签限制为
same-dataset recovery；本轮不把该 residual 静默升级成新的筛选规则。

第四轮 reviewers 同时确认 STAT3-M01/M02 与 EG3-M01/M02 已实质闭合。

## 整改闭包证据

- 新 `wod_scene_anchor_domain.csv` 是逐 scene membership 的唯一权威，共固定
  `479*16*2=15,328` 个 `segment_id × feature_id × horizon_id` groups。每个 AVAILABLE group
  只含其唯一有序 tau rows；每个 ineligible group 恰好一个 `tau_tick_or_NA=NA` terminal row；混合、
  缺失或重复 group 均为 fatal。exact columns、primary/group keys、UTF-8 LF encoding、排序、11 个
  membership statuses、reason mapping、manifest keys/count identities 和 generator/environment hashes
  全部冻结。
- `required_envelope_queries.csv` 明确为 aggregate-only：它只能从上述 AVAILABLE rows 做唯一
  group-by，`wod_scene_count=count(distinct segment_id)`；terminal rows 不参与。WOD feature/readout
  直接消费 per-scene artifact，BM90/BT90 只消费其唯一 aggregate，其他生成或重算路径均禁止。
  Recovery、envelope 与专属 test SHA-256 分别为
  `e3a6821a16b4e0c19c8af5a0c4577165f95781695b6c130a15d1f4cff75d4e1b`、
  `6b4046ba71e4cd365edbcdb70bde38819c5465545fed8531003cb6b45bddf877`、
  `2fa6d552c3ddcd5eeac491e2f03b671027576c62881509f13f45ca4bb4773b37`。
- Clean operator bootstrap 现在先锁定 fd8，再以 fd9 打开 exact managed wrapper，hash retained
  `/proc/$$/fd/9` 并执行同一 fd9；wrapper 禁止 fallback 创建 descriptor，并复核 readlink、regular
  file 与 descriptor/path 的 device/inode/mode。Launcher 在任何 RQ014 materializer/preflight import
  前复核同一 capability，并只返回 module-private identity token；普通 `True`、boolean 或 caller object
  不能通过 `_validate_cli_entry_mode` 或 `prepare_and_submit`。
- 真实 copied-launcher 恶意 `--rq014-only` 子进程在 dependency marker 执行前被拒绝；missing fd、wrong
  target、wrong inode、symlink、directory 与正向 inherited fd8/fd9 均有测试。Tongji login-node
  只读兼容探针又确认 `/bin/sh → /usr/bin/env -i → /bin/sh` 保留 fd8/fd9，输出
  `FD8_FD9_PRESERVED`；未访问项目数据、未创建 run root、未调用 Slurm。
- Round-4 manifest、两份 BLOCKED reviews 与本文进入 required set；当前 required=53，与 v1.3
  inheritance union=64。Launcher 为 115,810 bytes、SHA-256
  `5402bc9873ea89d0488381725ec472390546cc21468a7a38b0db501539243530`；wrapper 为 8,108 bytes、
  SHA-256 `9f6492094a041f4aa4c34c021a009a122f2f71a6e34dee6e4e08e3803b340676`。

主代理独立复跑 8-file RQ014 聚焦套件为 `194 passed`。全仓 non-shortcut 回归在排除本地未同步、
git-ignored 的 RQ009 scorer bundle 所专属 `test_verifier_runtime.py` 后为
`202 passed, 1 skipped, 2 deselected`；测试产生的 RQ012 fixture 输出已恢复。Python compilation、
wrapper/operator shell syntax、strict JSON parsing、exact hash assertions 与 `git diff --check` 均通过。

Capability 是本机 fail-closed provenance gate，不是抵御同一账号主动仿造 descriptor 的密码学秘密；
wrapper byte trust anchor 仍是 clean bootstrap 的 reviewed digest。该边界在合同中显式记录，不扩张为
系统级对抗性安全声明。

## 下一轮 gate

整改完成后必须重建新的 canonical review manifest，并交给两名未参与前四轮审查或实现的 fresh
reviewers。只有两路均为
`NO_BLOCKER` 且 0 unresolved blocker / major 才可生成 formal G1。当前仍未读取 rating value、未创建
production run root、未提交 Slurm。
