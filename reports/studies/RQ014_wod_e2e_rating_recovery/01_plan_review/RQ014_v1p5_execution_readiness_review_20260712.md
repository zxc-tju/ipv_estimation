# RQ014 v1.5 执行就绪审查：Formal G1

日期：2026-07-12  
审查对象：WOD E2E 人类评分—IPV envelope 偏离关系的已知结果恢复方案  
结论：**FORMAL_G1_PASS；允许进入首个受管去密步骤的发布准备，但尚未产生任何经验结果，也尚未提交 HPC 作业。**

## 结论先行

RQ014 v1.5 的科学合同、评分隔离边界和受管 HPC 执行合同已经通过第六轮 fresh 双路正式审查。
Statistics 与 execution/governance reviewer 对同一份 68-file manifest 均返回 `NO_BLOCKER`，未闭合
blocker/major 均为 0。Formal G1 artifact 已通过项目内机器 validator。

本结论的含义严格受限：

- 已证明的是“合同字节足够单值、边界足够 fail closed，可以准备执行首个
  `rq014_g2_declassification_export`”；
- 尚未读取任何人类评分值，尚未计算评分—偏离相关系数，尚未选出恢复 recipe；
- 本地专用分支尚未发布到 `origin/main`，因此 managed launcher 的 published-commit gate 仍会拒绝运行；
- `rq014_g2_contract_preflight`、rating-blind feature build、rating join/rank 与 clean replay 仍未获得当前
  中央授权；preflight 的未来解锁还要求 export receipt、authority byte refreeze、fresh 双审、Formal G1
  与新 final bundle。

因此，这是一份**执行就绪结论**，不是“历史相关性已经复现”的结果报告。

## 已冻结的恢复问题

研究目标不是重新发现是否存在关系，而是在不查看评分的前提下冻结候选规范，再恢复已知的负向关系：
人类评分越高，轨迹偏离人类 IPV envelope 的程度越低；评分越低，偏离程度越高。

冻结设计包含：

- 2 个采样率：4 Hz 原生相位与 10 Hz t* 锚定线性插值；
- 8 个时间语义：CH-W10/W25、LF-W10/W25、HF-W10/W25、TP、TF；
- 3 个 envelope：BL90、BM90、BT90；
- 2 个 horizon：H20、HFEAS；
- 10 个偏离 readout；
- 共 960 个 rating-blind predictor cells；
- 每 cell 使用 RWS、PSP、PPR 三种 association，总计 2,880 个必须终结的 leaderboard rows；
- 只允许一个按冻结 total order 机械选出的、`OBSERVED` 且 `recovery_compatible=true` 的 rank-1 recipe
  进入 clean replay，不允许回退到 rank 2。

通过第五轮整改后，时间域与状态域具有唯一实现：每个 exact closed position window 先切片，再只在窗内
派生 velocity、acceleration 与 heading；derivative halo 和跨窗状态复用均禁止。HFEAS 是有限
`rate_hz..h_common_tick` 域。逐场景 anchor membership 由 15,328 个固定 groups 的
`wod_scene_anchor_domain.csv` 合同承载，aggregate envelope query 只能从它唯一生成；每个 AVAILABLE
group 的 path type 必须组内单值并匹配 checksum-bound scene lookup。

## 正式审查证据

| 证据 | 结果 |
|---|---|
| 第六轮 review manifest | 68 files；SHA-256 `e8cda2e2fbe7e5a1f79f27ba64eb751b02d462ff5147d593245fd2c860bccdec` |
| Statistics review | `NO_BLOCKER`；0 blocker / 0 major；SHA-256 `9ce860f0511962cf5a0b5f98697ba1e06e2704bbedc9d7ccef053aa5742066f5` |
| Execution/governance review | `NO_BLOCKER`；0 blocker / 0 major；SHA-256 `41749fbfba243ecd8cd9992f8eb371a264b8ccd24aae75248e29fb63859ff4eb` |
| Formal G1 | `FORMAL_G1_PASS`；SHA-256 `a24e69c240700840d908514580ba6f43cb6555d1de02d825121475bdeda88713` |
| 聚焦回归 | 194 passed |
| 广泛 non-shortcut 回归 | 202 passed, 1 skipped, 2 deselected；排除本地未同步的 ignored RQ009 scorer 专属模块 |
| 静态验证 | Python compile、strict JSON、shell/operator syntax、exact hash、retained-fd assertions、`git diff --check` 全部通过 |
| Tongji 兼容探针 | fd8/fd9 经过 `/usr/bin/env -i` 保留；输出 `FD8_FD9_PRESERVED`；未调用 Slurm |

Formal G1 validator 还机器核验了：两名 reviewer identity 不同；两份 review 绑定同一 manifest；所有
v1.3 inherited bytes 保持原 hash；F01–F04 阴性取证证据与 registry 三方 hash 一致；前五轮 BLOCKED
manifest、reviews 与 remediation 均作为 append-only provenance 保留。

## 六轮审查如何收敛

| Round | Statistics | Execution/governance | 主要闭合项 |
|---|---|---|---|
| 1 | BLOCKED | BLOCKED | 去密科学原语、时间 seam、envelope/statistic 单值性、环境与 import surface |
| 2 | BLOCKED | BLOCKED | support/stability、rank/freeze、BL90、ledger/bootstrap、source TOCTOU、spec one-read、G0 evidence |
| 3 | BLOCKED | BLOCKED | 窗口边界导数、TF-HFEAS 有界性、isolated late import、双 reviewer identity |
| 4 | BLOCKED | BLOCKED | per-scene anchor membership、direct launcher wrapper capability |
| 5 | BLOCKED | NO_BLOCKER | AVAILABLE group 的 path-type 组内单值性 |
| 6 | NO_BLOCKER | NO_BLOCKER | 0 unresolved blocker / major，进入 Formal G1 PASS |

旧轮次没有被覆盖或“改写成通过”。每次发现 major 后，旧 manifest 与 review 原字节改名保存，修复后的
新字节重新生成 manifest，并使用新的 reviewer identity。

## 执行边界

当前唯一中央 allowlisted operation 是 `rq014_g2_declassification_export`。它只能读取 8 个 exact
score-omitting Phase-1 bundles、structural readiness TSV 和 selected counterpart CSV；每个源由
role/path/size/SHA 绑定，并以单个 `O_NOFOLLOW` descriptor 读取、核验和解析 retained bytes。Raw rated
TFRecord、ratings CSV、scored targets 和 joined rating tables 均禁止进入 G2。

Operator bootstrap 使用 clean environment，锁定 fd8，以 fd9 打开 exact wrapper，hash retained
`/proc/$$/fd/9` 后执行同一 fd。Wrapper 不得 fallback 创建 descriptor，并在首次 managed Python 前
验证 launcher、preflight、materializer、完整 stdlib、managed native closure 与 isolated sys.path。
Launcher 在任何 RQ014 dependency preload 前核验 fd8/fd9 path/device/inode/mode，并生成 module-private
capability token；普通 boolean、直接 flag、缺失或替换 descriptor 均不能进入 validate/submit。

运行代码只能从 final checksum bundle 登记的 exact published Git commit blobs 物化到 closed snapshot；
dirty/untracked bytes、Git worktree、hooks、filters、sitecustomize、PYTHONPATH shadows 和未登记文件均不进入
运行面。Slurm directive 与 submission command 均使用 `--export=NIL`。

## 接受的残余边界

- fd capability 是本机 provenance gate，不是抵御同一账号主动仿造 descriptor 的密码学秘密；真正的
  wrapper byte trust anchor 是 clean-bootstrap reviewed digest。
- `/lib64` 是明确声明的操作系统 ABI trust boundary。
- `scenario_cluster=NA` scenes 进入主 association 但不成为具名 LOCO cluster；fold、leave-one-scene 与
  至少两个 eligible named clusters 仍是 recovery compatibility gate。最终 claim 只允许 same-dataset
  historical recovery，不允许泛化或因果语言。
- WOD CP/HO/MP/F mapping 尚未物化；在任何 G2R feature build 前必须冻结 implementation、table、inputs、
  fixtures 与 environment，并重新经过 authority/manifest/review/G1/final-bundle 门。
- 本地 ignored RQ009 model bundle 未同步导致其专属两项全仓测试不可运行；该模块不在 RQ014 聚焦执行面。

## 下一步

1. 生成包含 review manifest、两份正式 review、Formal G1 与本报告的 final checksum bundle。
2. 按 Lore commit protocol 提交专用分支；不在本任务内推送或合并到 `origin/main`。
3. 合并并发布 exact commit 后，在 managed HPC checkout 同步该 commit，创建只读 canonical run spec。
4. 先运行 validate-only；只有所有 snapshot/submission/runtime plan 与 hash gate 通过后，才可提交首个
   `zxc-` 前缀的 declassification export job。
5. Export 完成并产生 exact receipt 后，若要启用 contract preflight，必须修改中央 authority byte，并
   重新生成 manifest、fresh 双审、Formal G1 与 final bundle。

在上述步骤完成前，不得声称已经复现“评分越高、IPV envelope 偏离越低”的经验关系。
