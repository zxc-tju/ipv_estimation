# RQ014 v1.5 formal G1 round-3 remediation

日期：2026-07-12  
状态：第三轮整改已闭合，等待第四轮 fresh formal review；本文不是 G1 PASS，也不授权 HPC。

## 冻结的 BLOCKED 证据

第三轮 fresh formal review 读取 56-row candidate manifest
`RQ014_plan_v1p5_review_manifest_round3_blocked_20260712.sha256`，其 SHA-256 为
`f32ef9104ebdbf97e424cb7d69f0f9f191224ffeb11362a5f458c043f7c2ddb9`。该轮结论原字节保留：

- statistics：`BLOCKED`，0 blocker / 2 major；artifact SHA-256
  `83d0377b953b84915987830212b908fb2455062b8a8bba3224a14bf38b0f33c0`；
- execution/governance：`BLOCKED`，0 blocker / 2 major；artifact SHA-256
  `9c592f4ce72f7cd3aa191d789acb28f7b26b474304077653452349a1db5cf189`。

两份 review 与 manifest 是 append-only provenance；不得改写成 PASS，也不得让两名第三轮 reviewer
参与下一轮 formal G1 计票。

## Finding-to-fix map

| Finding | 冻结风险 | 本轮整改目标 |
|---|---|---|
| STAT3-M01 | 在完整分支先做中心差分再切窗，使 CH/TP 端点使用声明窗口之外、甚至 `tau` 之后的位置；LF/HF 边界也消费区间外 halo | 先切出每个 exact closed position window，再仅在该窗口内用冻结的一侧/中心/一侧算子独立派生 velocity、acceleration、heading；任何 derivative halo 均禁止，CH/TP 的 causal 标签由输入支持事实保证。 |
| STAT3-M02 | TF 窗口与 `tau` 无关，而 HFEAS 只写 `tau>=1` 和窗口完整，未给有限上界 | 将所有 HFEAS anchor 冻结为 `tau_tick` 从 1 秒对应 tick 到 `H_common_tick` 的有限整数闭区间，再按该 temporal recipe 的完整窗口筛选；同步 WOD query、BT90 与 readout 规则和对抗测试。 |
| EG3-M01 | production exact-path bootstrap 置空 package path 且只预载 preflight；ledger validator 的 late import 必然找不到 `materialize_registry` | 把 exact reviewed `materialize_registry.py` 路径作为 bootstrap 参数并在执行 entrypoint 前显式装入 `scripts.rq014.materialize_registry`；继续禁止 checkout/root/src 与任意 shadow import，增加真实 `-I -S -B` 动态导入回归。 |
| EG3-M02 | formal G1 分别核验两个 role，却不要求 `reviewer_agent` identity 不同 | formal G1 validator 收集两个已验证 artifact 的 reviewer identity 并强制二者不同；基础 fixture 使用两个 identity，另加同 identity 必拒绝的对抗测试。 |

第三轮 statistics reviewer 另确认：WOD CP/HO/MP/F mapping 仍是未来 fail-closed gate，而非当前
declassification export blocker。G2R、rating join 与 replay 保持 `DENY`；mapping implementation、table、
safe inputs、fixtures 与 environment 必须在任何 feature build 前另行冻结、重新双审并进入新的 formal G1/
final bundle。

## 整改闭包证据

- `RQ014_envelope_builder_contract_v2.json` 与 `RQ014_recovery_lane_v2.json` 现在统一要求：先切出
  exact closed position window，再只在窗内以 first one-sided / interior centered / last one-sided
  算子派生 velocity，对窗内 velocity 用同算子派生 acceleration，并只在窗内计算/填充 heading。
  derivative halo、跨窗状态复用和 source dynamics 替代均为 `FORBIDDEN`；CH/TP 不再消费 `tau`
  之后的位置。
- HFEAS 的候选域是有限整数闭区间 `rate_hz..h_common_tick`，随后按 exact temporal-window
  completeness 筛选；H20 固定为 `rate_hz..2*rate_hz` 且整组完整。TF 虽然 IPV window 对 tau
  不变，anchor 仍满足 `tau_tick<=h_common_tick`。WOD query manifest、BM90、BT90 和四类 readout
  共用同一冻结域。16 families、48 feature-envelope executions、960 predictor cells、2,880
  association rows 不变。两份科学合同与专属测试 SHA-256 分别为
  `96e0aeb72c91590cf9494aa94628a224761767fdfbddd274da72b254ff7a453c`、
  `91077a4883932b9f016b767711ce27d6ca54d0a81c7c47dbe535339cc8f1bad3`、
  `eeedb5118dd70ae18d38ca8c8d3894569ed6a5a0de6e18199e145949b9977fc8`。
- 首次 isolated launcher 与 Slurm exact-path bootstrap 都在空 package path 下，以
  `spec_from_file_location` 显式预载 exact `scripts.rq014.materialize_registry`；wrapper 在首次
  managed Python 前校验 materializer 11,502 bytes、SHA-256
  `d8cac79f19e07296fef03415cca76474b48ca7b122d733ab2c3ec7146bbdecc4`。hostile cwd/
  `PYTHONPATH`、shadow/hook、missing file 与 wrong API 回归均 fail closed。
- formal G1 validator 先完整核验两份 review artifact，再要求两个去除首尾空白后的
  `reviewer_agent` identity 不同；同 identity fixture 现在被拒绝。真实主体独立仍由 fresh-review
  governance 与保存的 canonical task identity 共同保证，不能由字符串测试单独证明。
- 第三轮 manifest、两份 BLOCKED review 与本文均进入 required set；当前 required set 为 49 个
  文件，与 v1.3 inheritance 的 union 为 60 个文件。最终 launcher 为 111,039 bytes、SHA-256
  `ebcfb9696dd9ea64a349de77f1536504b5c70441064efc312b9237f557109d98`；wrapper 为 7,478
  bytes、SHA-256 `faa00e4297e63d9fef87318b0b1e07222981e17fc1954ce00a5ee95bc55b4c9b`。

主代理独立复跑的 8-file RQ014 聚焦套件为 `179 passed`。全仓 non-shortcut 回归在排除本地
未同步、git-ignored 的 RQ009 scorer bundle 所专属 `test_verifier_runtime.py` 后为
`187 passed, 1 skipped, 2 deselected`；测试产生的 RQ012 fixture 输出已恢复。Python compilation、
shell/bootstrap syntax、strict JSON parsing 与 `git diff --check` 均通过。

## 下一轮 gate

整改完成后必须重建新的 canonical review manifest，并交给两名不同且未参与前三轮实现或审查的
fresh reviewers。只有 statistics 与
execution/governance 两路均为 `NO_BLOCKER`、0 unresolved blocker、0 unresolved major，主代理才可
生成 formal G1。当前仍未读取 rating value、未创建 production run root、未提交 Slurm。
