# RQ014 v1.5 formal G1 round-1 remediation

日期：2026-07-12  
状态：第一轮与第二轮预审问题均已闭合，adversarial pre-review 为 `NO_BLOCKER/APPROVE`；等待冻结新
manifest 并交 fresh formal reviewers。本文不是 G1 PASS，也不授权 HPC。

## 结论

第一轮 formal G1 的 `BLOCKED` 判定有效且已原字节保留。其 1 个 blocker、5 个 majors 均已转化为
机器合同与对抗测试；旧 review manifest 已原 SHA 保存，不能沿用第一轮 reviewer 计票。第二轮预审又
闭合 runtime/native closure、commit-blob snapshot、association typed-ledger/status namespace、bootstrap
occurrence 与 clean operator bootstrap。冻结前聚焦套件 `141 passed`；广泛 non-shortcut 回归
`204 passed, 1 skipped, 2 deselected`；py_compile、shell syntax 与 diff check 通过。仍未读取 rating
value、未创建 RQ014 production run root、未提交 Slurm。

## Finding-to-fix map

| Finding | Round-1 风险 | 整改与机器证据 |
|---|---|---|
| STAT-M01 | 去密 bundle 丢失安全科学原语，G2 又禁止重开 parent pickle | schema/export/preflight 现无损保留 candidate/history/actual-future 的 7-state arrays、4×4 pose、exact route intent、source shard；候选源中空 dynamics 只编码 `NA`，legacy 7-field geometry SHA 不变。新增 two CSV 与 adversarial validators/tests。 |
| STAT-M02 | t*=0 seam、4→10 Hz phase 与导数端点不唯一 | 唯一 seam 为 history≤0/candidate>0；duplicate raw0 只作 t*=0 audit；R04/R10 均 t*-anchored；R10 0→0.25 仅支持内线性插值；完整 branch 只差分一次再切 window。 |
| STAT-M03 | CH/LF/HF/TP/TF 的 BM90/BT90 构建不唯一 | 新 `RQ014_envelope_builder_contract_v2.json` 枚举 16 families/48 envelope executions，冻结 InterHub pseudo anchors、双向 HV roles、support、reference、BM90/BT90 lookup/pooling、episode weights、weighted ECDF、gates 与 terminal states。恢复并绑定旧 scene-level WOD route reference 源码 SHA。 |
| STAT-M04 | RWS/PSP/PPR mask、weight、denominator 与 stability 不唯一 | 三种 association 对每个 predictor cell 共享 exact informative-scene mask；冻结 scene/candidate weights、ties/constants、support ID、分母、fold/LOO/LOCO/shard 重算，以及 2,880-row total order。 |
| EG-B01 | wrapper/Slurm 可在 reviewed checks 前继承环境 hook | 唯一 operator 命令从 `/usr/bin/env -i` 的 clean shell 原子校验 wrapper SHA 后执行；wrapper 再于首个 managed Python 前绑定 launcher/preflight/runtime；job directive 与 `/usr/bin/sbatch` 均 `--export=NIL`；`NONE` 明确禁止；工具路径绝对化。 |
| EG-M01 | 未审查 `sitecustomize`/shadow module 可进入 Python import surface | launcher 与 job 均 `-I -S -B -X utf8`；preflight/entrypoint 只按 exact reviewed path 装入，不添加 checkout/root/src；hostile cwd/PYTHONPATH/sitecustomize tests 通过。v3 environment receipt 绑定完整 stdlib checksum manifest、文件数/总字节数、0 symlink、无 special entry、zip absence，以及 20-row managed-native loader closure，并在第一次 managed Python 前校验。 |
| EG-m01 | submit failure 可留下不可审计、不可重试 namespace | submit 前失败回滚 partial code snapshot/run root 并写 retryable `FAILED` receipt；进入 sbatch 后状态不明则保留 namespace、原始响应及已知 job ID（若可解析），并写 non-retryable `SUBMISSION_STATE_UNKNOWN`。 |

## Round-2 pre-review additions

第二轮预审不是 formal review，也不能参与 PASS 计票；它新增并要求闭合以下问题：

- 托管 Python 的授权边界从单一 executable 扩展为完整 stdlib regular-file closure；wrapper 必须在任何
  managed Python 或 checkout Python 模块执行前，以绝对系统工具校验 v3 receipt、stdlib 与 managed-native
  checksum manifests。
- 运行代码只从 declared published commit 的 Git tree blobs 按 final checksum bundle 物化为 closed
  snapshot；不得建 Git worktree，也不得让 dirty/untracked bytes、hooks、filters、fsmonitor 或未登记文件
  进入 run。
- recovery ledger 必须统一 association terminal-status namespace，分离 raw attrition counts 与 typed
  ledger metrics，并冻结 early-fatal counts、typed `.value` projection 及 bootstrap 重复抽样 occurrence
  identity/order。
- `--export=NIL` 的 Slurm 语义和 Tongji 上 GNU `sha256sum --check --strict` 的只读兼容探针已确认；探针
  未创建作业。最终 formal review 仍须对重新冻结的 exact bytes 独立判定。

## 仍属未来 gate、不是本轮经验结果

- Declassification export 是当前唯一条件式 allowlisted operation；G2R、rating join 与 replay 仍为 `DENY`。
- CP/HO/MP/F WOD mapping、safe-primitive/reference/temporal golden-fixture manifest、InterHub source
  manifest 与 960-cell feature bank 必须在评分不可访问时另行物化、hash-freeze 并通过 G2R preflight。
- 本轮只建立“可以安全执行第一步”的合同；没有相关系数、没有 leaderboard、没有 recovered recipe。

## Round-2 acceptance rule

新的 statistics 与 execution/governance reviewer 必须读取同一份重新生成的 checksum manifest，逐项验证
上述修复并给出 `NO_BLOCKER`、0 unresolved blocker、0 unresolved major。任何受审字节变化均使 round-2
review stale；formal G1 只有在两路均通过后才可由 lead 生成。
