# RQ014 Plan v1.3 — Codex targeted independent design review

日期：2026-07-11
评审对象：不可改写的 base v1 + `RQ014_plan_v1p3_amendment_20260711.md` + 三套 resolved registries +
FL05/pass4 implementations + fixtures。
最终裁定：`PASS_AS_NONEXECUTABLE_DESIGN_CONTRACT / G0_OPEN / G1_NOT_REACHED / NO_EXECUTION_AUTHORIZED`

## 裁定含义

v1.3 已把 v1.2 的统计、取证和执行歧义收敛为可机器检查的设计合同，可以取代 v1.2 作为当前方案。
这不是 formal G1 PASS，更不是实验结果复现：base plan 要求先关闭 G0；当前 F05–F08/F10 仍为
`OPEN`，G2 code/data hashes 尚未冻结，八项授权全部为 `false`。因此不得读取评分、运行科学/取证
统计、连接 HPC 执行 RQ014，或提交 Slurm 作业。

## 独立证据角色

- statistics reviewer：复核 extension、Tier C/P/I、X02、resource projection 与 authorization map，裁定
  `NO_BLOCKER_PENDING_CHECKSUM`；
- execution reviewer：反证 FL05、F05–F08、两套 Slurm wrappers、atomic publication 与 interpreter
  provenance，最终裁定 `NO_BLOCKER_PENDING_CHECKSUM`；
- cutoff-provenance red team：证明 Git committer date 可回填，否决其 cutoff 证明能力；
- lead adjudication：选择 fail-closed 边界，不制造截止日前 receipt 的虚假正例。

## v1.2 阻断项的处理

| 评审项 | v1.3 冻结的修复 | 裁定 |
|---|---|---|
| FL05 formats/audit | wide/long CSV、numeric/long JSON、statistic-local Markdown；finite/range validation；逐文件 audit；immutable generation + atomic `CURRENT` | 通过 fixtures |
| FL05 HPC fallback | 固定 `zxc-rq014-fl05`、durable root、200 MiB/4 GiB lanes、checksum-bound parser/shell/sbatch、Python realpath/version/SHA receipt | 通过静态与 fixture review |
| F05–F08 closure | 逐 surface bundle、完整 byte-stream、O_NOFOLLOW/containment/identity、fail-closed status、atomic pointer | 通过 fixtures |
| F07/F08 cutoff | Git 仅作 `content_integrity_only`；无截止日前 whole-inventory receipt 时固定 `INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT` | 通过，保留可用性边界 |
| Extension X01–X05 | rating-blind direct feature universe、同一 `M_k`、scene-keyed single-step minT、cell-adjusted p、V04 paired contrast、complete-null exploratory wording | 统计合同通过 |
| Tier C/P/I | frozen common sets、latent-rank DGP、CORE_REJECT、20,000 fixed denominator、2,520 batches、worst-cell gates、numerical failures retained | 统计合同通过 |
| X02/invariance | 全量 denominator、coverage/scale eligibility、average-midranks policy；alias exact rank-vector；invariance 完全 descriptive/non-gating | 统计合同通过 |
| Governance | 八项授权、逐 operation 逻辑 AND、G2/G2P/discovery/post-promotion/confirmation/G4S 时序分离 | 通过；全部授权仍 false |

## 截止时间证据的关键边界

本次 red team 发现，测试可在当前新建 Git repo 并回填 `GIT_COMMITTER_DATE`，因此 commit hash + blob hash
只能证明内容一致，不能证明整份扫描 inventory 在 2026-07-10 前已经存在。v1.3 不注册生产 TSA
profile，也没有已知的截止日前 RFC 3161 whole-inventory receipt。脚本仍可保存 F07/F08 的候选线索，
但 terminal state 必须为 `INACCESSIBLE`、`complete_scan=false`、非零退出；不得据此写
`NOT_FOUND_ON_SCANNED_SURFACES`。未来若找到真实历史 receipt，必须通过新的 checksum-bound amendment
冻结 OpenSSL 3、root、exact signer leaf、intermediates、policy、accuracy/revocation contract，并重新
独立复审。

## 验证证据

- `42 passed`：`test_rq014_fl05_indexer.py`、`test_rq014_g0_closure_scripts.py`、
  `test_rq014_v1p3_registry_contract.py`；
- 五个 shell/sbatch wrappers 均通过 `bash -n`；indexer 与三份 tests 均通过 `py_compile`；
- 三个 YAML registries 可解析、无 duplicate keys，八项 authorization scopes 全为 `false`；
- config 中 statistical contract 与独立 reviewer artifact 内容一致，仅有 code-block 结尾空行差异；
- `RQ014_plan_v1p3_checksums_20260711.sha256` 共 14 项，`shasum -a 256 -c` 为 14/14 `OK`；
- 未读取 ratings、真实 HPC 或 OneDrive 数据；未运行 RQ014 compute；未提交 Slurm job。

## 当前 gate 与合法下一步

| Gate/arm | 当前状态 | 下一项合法动作 |
|---|---|---|
| G0 | `OPEN` | 仅在 `g0_readonly_forensics_authorized=true` 后运行 checksum-bound 只读 F05–F10 closure |
| F07/F08 | 只能 `INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT` | 记录 residual risk；不得用新 token、Git backdate 或测试 CA补证 |
| G1 | `NOT_REACHED` | G0 全部成为合法 terminal state 后，再进行 formal plan/registry gate |
| G2/G2P | 未开始 | formal G1 与 scoped authorization 后才可冻结 code/data/env hashes 并做 ratings-blind build/power |
| rating joins / compute | 未授权 | 继续禁止；不同 arm 的 authorization 不得互相替代 |

## 评审产物

- [v1.3 amendment](../../../plans/RQ014_plan_v1p3_amendment_20260711.md)
- [v1.3 checksum manifest](../../../plans/RQ014_plan_v1p3_checksums_20260711.sha256)
- [HTML review](codex_plan_review_v1p3_20260711.html)

最终结论：v1.3 作为**非执行设计合同**通过 targeted review。它提高了未来复现实验的可识别性与
审计性，但没有产生、恢复或确认任何 rating↔IPV-deviation 经验结果。
