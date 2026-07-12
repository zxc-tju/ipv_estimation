# RQ014 v1.5 G2 staged kickoff — managed declassification first

你是 RQ014 的执行编排主代理。机器权威为 v1.5 amendment、三份 v1.5 registry、
`RQ014_recovery_lane_v2.json`、`RQ014_envelope_builder_contract_v2.json`、
`RQ014_execution_contract_v1p5.json`、formal G1 artifact、final checksum bundle、
`configs/research_authorization.json` 与 managed launcher。任一冲突、缺失或 SHA 漂移都停止。

## 当前唯一可执行 operation

```text
rq014_g2_declassification_export
```

只有以下前件全部成立才有效：formal G1=`FORMAL_G1_PASS`；exact reviewed commit 已发布到
`origin/main`；managed checkout HEAD 等于 run commit（dirty/untracked bytes 不是执行源，由 exact commit
blob snapshot 排除）；中央 allowlist 只列该 operation；source/env/spec hashes 全部通过。
formal G1 绑定的 statistics 与 execution/governance review 必须均通过结构验证，并声明不同的非空
`reviewer_agent` identity。
`rq014_g2_contract_preflight` 已在 reviewed contract 中条件式注册，但中央 allowlist 尚未列入，故仍 DENY。

## 不可触碰边界

- 不打开或挂载 `$W/data/rated479_segments`；其中 TFRecord 仍嵌入 `preference_score`。
- 不读取 scored target、ratings CSV、joined table、rating order/tie/completeness 或任意 observed statistic。
- 不从 `$W/code`、`/share/home/u25310231/ZXC/ipv_estimation` 或 unmanaged checkout 执行。
- 不直接调用 `sbatch`，不创建 `/share/home/u25310231/ZXC/RQ014_recovery`。
- 不运行 preflight/resource pilot/G2R feature build/G3R rating recovery/G4R clean replay/optional
  validation/extension/forensic。
- 不修改 checksum-bound source registries；本阶段也不物化 G2 bindings。

`$W=/share/home/u25310231/ZXC/RQ010B_wod_e2e` 只作为三个精确 declassification source classes 的
只读根：8 个 `phase1_post_scene_bundle.pkl`、structural readiness TSV、selected counterpart CSV。
路径/大小/SHA 已冻结在
`reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/RQ014_declassification_source_inventory_20260712.json`。

## 执行路径

唯一 base：`/share/home/u25310231/ZXC/sociality_estimation`。唯一 launcher：

```bash
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c 'wrapper=/share/home/u25310231/ZXC/sociality_estimation/code/repo/scripts/hpc/submit_research_run.sh; lock=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock; test ! -L "$wrapper" && test -f "$wrapper" && test ! -L "$lock" && exec 8>"$lock" && /usr/bin/flock -s 8 && exec 9<"$wrapper" && test "$(/usr/bin/readlink /proc/$$/fd/8)" = "$lock" && test "$(/usr/bin/readlink /proc/$$/fd/9)" = "$wrapper" && /usr/bin/printf "%s  %s\n" d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d /proc/$$/fd/9 | (cd / && /usr/bin/sha256sum --check --strict -) && exec /bin/sh /proc/$$/fd/9 "$@"' rq014-bootstrap --spec /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/REPLACE_EXPORT_RUN_ID.json
```

先运行 validate-only。输出必须显示：operation、exact commit、formal G1、contract bundle、10 个 source
path/size/hash expectations、v3 environment manifest、完整 stdlib 与 managed-native checksum closure、exact
`code_snapshot_files`/commit-blob materialization plan、固定 exporter entrypoint、
`zxc-rq014-export-<specsha12>`、1 CPU/8 GiB/2 h、全部 thread limits=1、固定 output root，且
rating_access=`FORBIDDEN`。validate-only 不创建 run root、snapshot receipt 或 rendered sbatch；实际 snapshot
receipt 与 directive/submit-command 双重 `--export=NIL` 仅在 `--submit` 内生成并重新 fail closed 校验。
bootstrap 必须先取得并锁定 fd8 runtime lock，再打开 exact wrapper 为 fd9，hash 并执行 retained
`/proc/$$/fd/9`；wrapper 只能继承两个 descriptor，不得 fallback 创建。launcher 在 materializer/preflight
import 前复核 exact target 与 path/fd device-inode-mode identity；direct `--rq014-only` 或 caller boolean/token
替代值一律拒绝。该 local machine provenance gate is not a cryptographic secret against deliberate
same-account descriptor emulation；真正的 wrapper byte trust anchor 仍是 clean-bootstrap digest。拆成两条
命令、重新执行 wrapper path、漏掉任一 fd 或使用其他 wrapper SHA 均失败。

只有 validate-only bounded evidence 无差异时才允许：

```bash
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c 'wrapper=/share/home/u25310231/ZXC/sociality_estimation/code/repo/scripts/hpc/submit_research_run.sh; lock=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock; test ! -L "$wrapper" && test -f "$wrapper" && test ! -L "$lock" && exec 8>"$lock" && /usr/bin/flock -s 8 && exec 9<"$wrapper" && test "$(/usr/bin/readlink /proc/$$/fd/8)" = "$lock" && test "$(/usr/bin/readlink /proc/$$/fd/9)" = "$wrapper" && /usr/bin/printf "%s  %s\n" d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d /proc/$$/fd/9 | (cd / && /usr/bin/sha256sum --check --strict -) && exec /bin/sh /proc/$$/fd/9 "$@"' rq014-bootstrap --spec /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/REPLACE_EXPORT_RUN_ID.json --submit
```

## Export 合同

固定 tracked entrypoint：`scripts/rq014/export_score_stripped_bundle.py`。它必须：

1. 使用 RestrictedUnpickler，拒绝 pickle global/class；要求 `ratings_blind=true`、8 shard 的
   `prep_abstentions=[]`，并递归拒绝任何 rating-semantic source key；
2. 对 bundle/top/scene/state、readiness、counterpart header 做 exact allowlist；
   launcher 必须传入 10 个 reviewed role/size/SHA-256；每个源以单次
   `O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK` descriptor 读取，前后 `fstat` 连续、exact bytes digest
   通过后，只从同一 retained `BytesIO`/`StringIO` 解析，receipt 记录同一 digest；
3. geometry hash 逐字节复用 7-field `.17g` 历史规则；
4. past[-1] 为 t*=0；candidate optional duplicate drop threshold `<0.75 m`；effective points 最多 20，
   时间为 `0.25,0.50,... s`；保留 raw-time 解释并冻结 history<=0/candidate>0 seam；
5. 无损导出安全的 `source_shard_id`、route intent、history 7-state、actual future 7-state 与 4x4 pose；
   candidate 源中为空的 state arrays 必须编码为整列 `NA`，不得伪造 native dynamics；
6. 输出 fixed nine-file canonical bundle（7 CSV + 2 JSON）到
   `/share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1`；
7. 在 atomic publish 前运行完整 score-stripped validator；unexpected field、非有限值、duplicate key、
   rating semantic、TFRecord/pickle/protobuf、symlink 或 parent path 泄漏任一出现即失败；
8. run root outputs 写 export receipt + `DONE.json`，不得只靠 stdout 宣称完成；receipt 写失败时回滚
   output root，进程崩溃留下的无 receipt root 必须人工隔离而不得直接重用。

在第一次 managed Python 前，clean operator shell 与 wrapper 必须只用绝对系统工具依次验证 wrapper、
launcher、preflight、registry materializer、managed Python executable、`managed_python_environment_v3.json`、14,326 个 stdlib
regular files（307,357,072 bytes）、0 symlink、无 special entry、`python39.zip` absent、20 个 pinned
managed-native libraries 及 isolated `sys.path`。trusted launcher 再于 scientific entrypoint 前校验 final
checksum bundle 文件自身与全部登记 bytes；运行代码只能从 exact Git commit blobs 物化到 run root 的
closed snapshot。禁止 Git worktree、hooks、filters、fsmonitor、dirty/untracked 与 unregistered files。
两个 isolated Python 阶段均保持 `scripts`/`scripts.rq014.__path__` 为空，并在 entrypoint 前通过
`spec_from_file_location` 将 exact closed-snapshot `materialize_registry.py` 显式预载为
`scripts.rq014.materialize_registry`；不得使用普通 path import 或本地 shadow。

## 完成证据

只在 Slurm `COMPLETED/0:0`、job name/workdir/commit/spec SHA 精确一致、bundle validator PASS、
`sanitization_receipt.json` 与 `file_manifest.json` hashes 封存、479 universe identity 通过时，形成 bounded
report。报告必须写 `rating_access=NONE`、`observed_statistics=NONE`。

随后停止。只有在该 report/receipt/DONE 被接受后，才可把已条件式注册的
`rq014_g2_contract_preflight` 单独加入中央 allowlist；该 reviewed authority byte 的变化必须重建 candidate
manifest、双路 review、formal G1 与 final bundle，但不再改 execution-contract status。完成这些步骤前不得
准备或执行下一张 spec。

## Bounded report（≤100 行）

STATUS / OPERATION / RUN_ID / GIT_COMMIT / SPEC_SHA256 / JOB_ID_AND_STATE /
SOURCE_HASH_VERDICT / OUTPUT_BUNDLE_PATH_AND_HASHES / UNIVERSE_AND_GEOMETRY_COUNTS /
FORBIDDEN_FIELD_AND_SCHEMA_SCANS / RATING_ACCESS / DEVIATIONS / NEXT_CENTRAL_AUTH_DECISION。
