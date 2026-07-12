# RQ014 plan v1.5 amendment — rating-blind execution and historical-recovery contract

日期：2026-07-12  
状态：formal G1 的最终状态由独立 artifact 决定；本文本身不授权执行。  
适用顺序：base v1 → v1.3 scientific hardening → 本 v1.5 amendment → `RQ014_recovery_lane_v2.json`。

## 1. 目的与不变项

v1.5 同时修复 bounded preflight 暴露的执行合同矛盾，以及复核发现的科学目标错配。RQ014 的主目标是
找回一个已经被观察到、但具体设置遗失的负相关 recipe，不是重新发现零假设。因此原 v1.3 的 G0、
declassification、评分隔离、provenance、estimator 数值核与 managed-HPC 约束继续有效；原 12-config
split/power/promotion/confirmation 路线则不再控制历史恢复。

新的机器科学权威 `RQ014_recovery_lane_v2.json` 冻结 16 个 rate × temporal feature families，其中
`CH` 是真正的因果历史窗 `[tau-W,tau]`，`LF` 是真正的未来 look-ahead `[tau,tau+W]`，`HF` 是双侧
组合窗，同时保留最可能的旧实现 `TP=[0,tau]` 与 `TF=[0,H_common]`。它再交叉 3 种 envelope、2 个
horizon 与 10 个 deviation readout，形成 960 个评分盲 predictor cells；评分访问后一次性计算 RWS、
pooled Spearman、pooled Pearson 三种关联，共 2880 个 append-only terminal rows。

全量 rated479 的恢复排序不以 p value 或 prospective power 为门槛。全部行终结后只机械选择 rank 1，
冻结 recipe，并由 fresh agent 在独立、commit-addressed 的干净执行面中重写 resampling、window、
envelope、deviation、join 与 statistic 后重放。它能支持“同一数据集内历史结果已计算复现”，不能称为
新数据 confirmation；托管 HPC 上的该执行面必须由最终 checksum bundle 物化为 closed code snapshot，
不得用 Git worktree 代替。
旧 power/split/Tier C/P/I 合同仅保留为 non-gating provenance、post-selection stability 与未来新样本规划。
frontier lead model 只管理 registry/gate/aggregate interpretation；低成本 workers 只执行冻结 cell batches，
不得修改 axes/ranking/ledger，也不得看到 partial leaderboard。

机器权威为：

- `RQ014_config_space_v1p5.yaml`；
- `RQ014_forensic_registry_v1p5.yaml`；
- `RQ014_recovery_extension_registry_v1p5.yaml`；
- `RQ014_recovery_lane_v2.json`（历史恢复的主科学权威）；
- `RQ014_envelope_builder_contract_v2.json`（16 个 feature families 的 reference、InterHub pseudo-anchor、BM90/BT90 与 association-support 执行语义）；
- `RQ014_execution_contract_v1p5.json`；
- `configs/research_authorization.json` 与 managed launcher。

任一层缺失、冲突、hash 漂移或为 `DENY`，最终决定均为 `DENY`。

## 2. v1.4 attempted launch 的处置

2026-07-11 PI decision 的 G0 waiver 意图继续有效，但 attempted v1.4 不构成合法版本：

1. 它把 `INACCESSIBLE_PI_WAIVED` 写进关闭枚举，而 base v1 只允许 `FOUND`、
   `NOT_FOUND_ON_SCANNED_SURFACES`、`INACCESSIBLE`；
2. 它在旧 review 明示 `G1_NOT_REACHED` 后自行宣告 formal G1 PASS；
3. 它同时保留 `execution_authorized=false` 与局部 scope=true，优先级未定义；
4. 它原地改写 checksum-bound `*_v1p3.yaml`，使同一路径被两个版本绑定到不同 bytes；
5. kickoff 使用已经退役的 `/share/home/u25310231/ZXC/ipv_estimation`、非托管 run root 和 direct sbatch。

因此 v1.3 文件与测试恢复为原字节；v1.5 全部使用新文件名。attempted v1.4 的 PI decision 与 bounded
preflight report 作为 provenance 保留，但其执行条款被本文取代。

## 3. G0 合法关闭

### 3.1 状态与原因分离

F01–F04 保持 `NOT_FOUND_ON_SCANNED_SURFACES`。F05、F06、F07、F08、F10 统一为
`status: INACCESSIBLE`，另用 `closure_basis: PI_WAIVER`、reason codes 与 checksum-bound decision
表达 waiver；F09 为独立的 legacy-storage inaccessible，不属于 waiver。F07/F08 额外登记
`NO_PRE_CUTOFF_WHOLE_INVENTORY_RECEIPT`，但该原因码不得冒充 terminal state。

聚合终态为：

```text
CLOSED_WITH_INACCESSIBLE_SURFACES
FOUND=0, NOT_FOUND_ON_SCANNED_SURFACES=4, INACCESSIBLE=6, OPEN=0
```

负面结论只覆盖 F01–F04；不得写“所有可能档案均未找到”。

### 3.2 条件式 freeze

waiver surface 只要求合法状态、`complete_scan=false`、waiver decision/hash、reason codes 与 residual
risk；不得要求没有生成的 pass4/FL05 outputs，也不得保留 `closure_evidence`。exact input manifest、
forensic twin common support 与 rating join 属于未来 forensic compute 前置条件，不能反向阻断 G0。

## 4. Formal G1

formal G1 必须在 G0 与 v1.5 candidate review manifest 冻结后执行。至少需要两份 fresh independent
review：statistics 和 execution/governance。两者读取同一个 review manifest，逐项复核 G0、科学合同、
blind boundary、materialization、managed HPC 与 tests，并只给 `NO_BLOCKER` 或 `BLOCKED`。两份通过
结构验证的 review payload 必须声明不同的非空 `reviewer_agent` identity；同一 identity 承担两路角色
一律 fail closed。

唯一 lead artifact 只允许：

```text
FORMAL_G1_PASS
FORMAL_G1_BLOCKED
```

只有两个 reviewer 均 `NO_BLOCKER`、unresolved blocker=0、unresolved major=0、checksums/tests 全部通过
才可 PASS。reviewed plan/registry/test/launcher/authorization bytes 任一改变，artifact 立即 stale。G1 PASS
仅说明设计可执行，不自动授权 operation。

第一轮冻结 manifest（SHA-256 `85bc636f...d0853`）已按此规则执行并正确得到 `BLOCKED`：statistics
review 指出四个 major（安全科学原语丢失、t*=0 seam/10 Hz 相位冲突、BM90/BT90 未完全定义、三种
association 的 mask/weight/denominator 未冻结）；execution/governance review 指出一个 blocker 与一个
major（提交/Slurm 环境可注入、Python import surface 未封闭）。原审查字节以
`*_round1_blocked_20260712.json` 保留，绝不改写为 PASS。后续修订必须生成新的 review manifest，并由
不同的 fresh reviewers 重新审查；第一轮结论只作为整改证据，不能参与 PASS 计票。

## 5. Operation-level 授权

废弃聚合 `execution_authorized` 与八个重复布尔作为机器权限源。唯一机器权限是 operation registry 与
`configs/research_authorization.json` 的交集，并叠加 exact gate/hash/prohibition checks。

v1.5 首轮只允许条件式注册：

```text
rq014_g2_declassification_export
```

它只读八个 score-omitting Phase-1 bundles、结构 readiness 表与 counterpart 表，生成 canonical
score-stripped bundle。`rq014_g2_contract_preflight` 现在以 exact export receipt/DONE/run-spec refs 为 predicate
条件式注册，但等待 receipt 被接受且中央 allowlist 更新前仍 DENY；resource
pilot 与新的 `rq014_r2_blind_feature_build` 分别等待前序 receipt。`rq014_r3_full_rating_join_and_rank`
需要评分盲 feature bank 完整封存及新的 PI/中央评分授权；`rq014_r4_clean_replay` 需要唯一 recipe freeze。
旧 blind build/G2P power 退为 optional non-gating，所有 rating、replay、optional operation 目前均为
`DENY`。未知 operation fail closed。

## 6. Score-stripped rated479 输入

只读代码核验确认 `$W/data/rated479_segments/<segment>/frames.tfrecord` **仍嵌入
`preference_score`**：旧 `slim_e2ed_frame` 先 `CopyFrom(frame)`，随后只裁剪相机数据，没有清除
`preference_trajectories[*].preference_score`。旧 `read_target_rows_no_scores` 也先读入含
`candidate_scores` 的完整 CSV 行再丢列，不能视为“未读取评分”；旧 geometry gate 更显式查询
`HasField("preference_score")`。因此 raw TFRecord、scored target、旧 pickle 及其 reader 全部不得成为
G2 输入。

reviewed source inventory 逐项绑定上述 8 个 bundle、readiness 与 counterpart 的绝对路径、size、SHA，
launcher 把 10 个 reviewed role/size/SHA 传给 exporter；exporter 对每个源只开一个 no-follow descriptor，
以 `fstat` 连续性与 expected size/SHA 验证一次读取的 exact bytes，只从 retained `BytesIO/StringIO` 解析，
并在 receipt 记录同一 digest。任何 symlink、short/growing read、identity/timestamp drift 或 digest 漂移均失败。
run spec 自报的新 hash 不能替代 inventory。静态数据流审计另绑定三个 legacy producer 源文件：target
CSV 的 `candidate_scores` 在形成 row dictionary 前被移除；cohort/t* 只使用 scored-frame count 与结构
字段；candidate geometry 按 protobuf repeated-field 原顺序序列化，没有 score sort/top-k；score-field
presence 只作结构门槛。candidate ordinal 因而只能作 opaque join key，禁止作为 predictor；rating join
必须用 geometry SHA。

先通过独立审计的 declassification exporter，把八个已经不含 score value 的 Phase-1 scene bundles、
479-ID structural readiness 与 selected counterpart table 重序列化为严格 allowlist 的 canonical CSV/JSON：

```text
/share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1/
  blind_scene_manifest.csv
  candidate_states.csv
  ego_history_states.csv
  ego_future_states.csv
  tstar_ego_pose.csv
  counterpart_tracks.csv
  structural_attrition.csv
  sanitization_receipt.json
  file_manifest.json
```

schema 由 `RQ014_score_stripped_schema_v1.json` 冻结：G2 只接受 RFC4180 CSV 与 canonical JSON，
不接受 pickle/TFRecord/protobuf；任何 unexpected column/key 或 normalized rating semantic field 都 fatal。
旧 `score` detector-confidence 列改名为 `detector_confidence`。sanitization receipt 可登记 rating-bearing
parent 的 artifact ID/hash，但不得携带评分值、order、tie 或 G2 可解析的 parent absolute path。

candidate geometry hash 必须逐字节复用历史 G3 的 7-field `.17g` 规则，并在任何去重/截断之前计算。
Phase-1 bundle 的 frame 轴为 0.1 s，但 candidate/ego estimator 轴为 0.25 s：ego past 最后一项固定 t*=0；
若 candidate raw point 0 到 past[-1] 的 XY 距离 `<0.75 m`，只把该点标为 t* duplicate；此后最多 20 点
依次映射到 `+0.25,+0.50,... s`。export 同时保留 raw index、drop flag、effective index/time，禁止从
bundle key 顺序或 frame cadence 猜 candidate 时间。

round-1 整改后，去密边界还必须无损保留所有已审计且安全的科学原语：`source_shard_id`、route intent
的 exact `1=GO_STRAIGHT, 2=GO_LEFT, 3=GO_RIGHT` 映射、16 点 ego history 的 7 个源状态数组、20 点
actual ego future、finite 4x4 t* pose，以及候选的全部 7 个源数组（源中为空者逐行编码为 `NA`）。
candidate 的原生 velocity/acceleration 在 1,428 条源轨迹中均为空，因此不得声称存在 native candidate
dynamics；恢复计算只从位置按冻结算子重建。`path_type` 在 export 中诚实保留为 `UNMAPPED`，必须在
G2R 之前只用这些评分盲安全原语另行物化 CP/HO/MP/F mapping 并冻结 SHA。

唯一 seam 为 history `t<=0`、candidate `t>0`，t*=0 只由 history 拥有。candidate raw0 与 history t*=0
距离严格 `<0.75 m` 时，raw0 仅作 t*=0 audit row 并从 effective future 删除；否则 raw0 是 `+0.25 s`。
R04N 与 R10L 均以 t*=0 定相；R10L 的 `0<t<0.25` 只在线性连接 history t*=0 与首个 positive
candidate anchor 的观测支持内插值。velocity/acceleration/heading 在每条完整重采样 branch 上只重建
一次，再切 CH/LF/HF/TP/TF window；禁止在 window 边界重新差分。

选择 479 个 segment 及历史 t* 可作为 declassified structural metadata 保留，但必须明确：它们源于旧
scored-frame selection，只是不泄露 score value/order/tie。exporter 是受信任的去密边界，必须独立审计；
G2 只能只读挂载 export root，不能看见或解析它的父输入。

## 7. 分阶段 input manifest

- `input_manifest.g2.json`：只含 canonical score-stripped WOD bundle/receipt、严格 schema 的 InterHub
  pure-HV/HV source manifest 与 blind anchor receipt；rating/split/pickle/TFRecord roles 禁止。
- `G2R`：先通过 safe-primitive/reference/temporal/InterHub-role golden fixtures，再封存 960-cell
  rating-blind feature bank、envelopes、masks、controls 与 producer hashes；当前仍未授权。
- `G3R`：单次 full-rated479 rating join 与 2880-row recovery ledger；当前未授权。
- `G4R`：只读取 frozen rank-1 recipe 的 clean independent replay；当前未授权。
- `G5_NEW_DATA`：未来真正未参与 recovery 的新评分批次；当前未授权。

因此 base v1 §13.1 的“单一 manifest 一次包含 ratings/splits”被 staged-finalization 取代。

## 8. A1–A4 anchor 的盲态处理

G2 不重算 A1–A4 rho。它只校验 `RQ014_blind_anchor_receipt_v1p5.json` 中已经在 base v1 公开的四个
aggregate constants、receipt/source hashes 与 rating-free estimator fixtures。任何新 rho、N/key join 或
row-level parity 都需要未来 checksum-bound rating-join operation，并且必须在新 discovery statistic 前
完成。receipt 通过只能说明公开 anchor provenance 未漂移，不能声称 numerical rho 已复算。

## 9. Registry binding，不改写 source YAML

三个 v1.5 source registries 均 immutable。G2 不“填回” `TO_FREEZE_AT_G2`，而生成
`registry_bindings.g2.json`：它绑定三份 source registry SHA 与 17 个明确 binding IDs，每个值必须为
64 位小写 SHA。X02 在 valid/extension registry 间的 source-definition、WOD-mapping、implementation
hash 必须相等。

materialized registry manifest 是 source hashes + complete binding ledger 的联合 receipt；不是改写后的
YAML。source SHA 漂移、缺 binding、未知 binding、值非 SHA 或 cross-registry equality 失败均 fatal。

## 10. X02 双 artifact

X02 不再把 `X02_source_definition.yaml` 和 `X02_wod_mapping.csv` 塞进一个未定义的
`artifact_sha256`。两者分别登记 SHA；如需要 aggregate digest，固定为 sorted-key canonical JSON 对象
`{source_definition_sha256, wod_mapping_sha256}` 的 SHA-256。旧 X02 scale gate 不得再使可能的历史
static-envelope recipe 从恢复屏幕消失：exact legacy artifact 可作为 `BL90`，不可达则固定
`INELIGIBLE_REFERENCE`；matched static `BM90` 与 matched time-indexed `BT90` 仍完整执行。

`RQ014_envelope_builder_contract_v2.json` 进一步逐项冻结 16 个 sampling×temporal families、InterHub
pseudo-t*=0 integer grids、两个 directed HV roles、complete support、episode-equal weighted ECDF、BM90
horizon pooling、BT90 exact-tau cells、TF 的 `H_common` key、50-episode/ordered-quantile/half-width/bootstrap
gates 与所有 terminal states。WOD focal reference 恢复为评分盲旧实现 `legacy_route_reference_v1`：三条
candidate 共用由 16 点 past 与 route intent 生成的 scene-level 参考线（12 m 转弯半径、80 m 延伸、1 m
步长）；counterpart reference 为同一 estimator window 上的 observed/resampled XY。源代码路径、size、SHA
与 exact 算法均已登记，后续实现/fixture 必须 hash-match，不能在看评分后替换 reference。

三种 association 对同一 predictor cell 必须共享完全相同的 informative-scene mask：每场三条 rating 与
三条 deviation 均 finite，且两组三元向量各自非恒定。RWS 场景等权；PSP/PPR 以每候选
`1/(3*n_scene)` 等权；ties 用 exact-binary64 average midranks。每行携带 support ID、完整分母、fold/LOO/
LOCO/shard 重算；2,880 行全部 terminal 后按冻结 total order 机械选唯一 rank 1。

## 11. 托管 HPC

唯一 durable base：

```text
/share/home/u25310231/ZXC/sociality_estimation
```

唯一 managed checkout：`<base>/code/repo`；它只提供受审 wrapper 及 exact published commit 的来源。
每个 run 只能把 final checksum bundle 登记的 regular files 从 exact commit blobs 物化到
`<base>/work_dirs/RQ014/<RUN_ID>/code` closed snapshot，先绑定 bundle 自身 SHA，再逐文件校验，并把
snapshot receipt 封存为 `<base>/work_dirs/RQ014/<RUN_ID>/manifests/code_snapshot.json`。Git
checkout/worktree、hooks、filters、fsmonitor、symlink 与未登记 commit 文件均不得进入执行面。
输出仍位于 `<base>/work_dirs/RQ014/<RUN_ID>`。validate-only 与 submit 分别只可使用以下同形的单条
clean-environment bootstrap 命令；wrapper SHA 校验、wrapper 执行不能拆开：

```bash
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c 'wrapper=/share/home/u25310231/ZXC/sociality_estimation/code/repo/scripts/hpc/submit_research_run.sh; lock=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock; test ! -L "$wrapper" && test -f "$wrapper" && test ! -L "$lock" && exec 8>"$lock" && /usr/bin/flock -s 8 && exec 9<"$wrapper" && test "$(/usr/bin/readlink /proc/$$/fd/8)" = "$lock" && test "$(/usr/bin/readlink /proc/$$/fd/9)" = "$wrapper" && /usr/bin/printf "%s  %s\n" d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d /proc/$$/fd/9 | (cd / && /usr/bin/sha256sum --check --strict -) && exec /bin/sh /proc/$$/fd/9 "$@"' rq014-bootstrap --spec /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/REPLACE_RUN_ID.json
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c 'wrapper=/share/home/u25310231/ZXC/sociality_estimation/code/repo/scripts/hpc/submit_research_run.sh; lock=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock; test ! -L "$wrapper" && test -f "$wrapper" && test ! -L "$lock" && exec 8>"$lock" && /usr/bin/flock -s 8 && exec 9<"$wrapper" && test "$(/usr/bin/readlink /proc/$$/fd/8)" = "$lock" && test "$(/usr/bin/readlink /proc/$$/fd/9)" = "$wrapper" && /usr/bin/printf "%s  %s\n" d8036336c354b202f388c5fd9dbc05a80a1e8292a574cc4c47400d0a772d7a1d /proc/$$/fd/9 | (cd / && /usr/bin/sha256sum --check --strict -) && exec /bin/sh /proc/$$/fd/9 "$@"' rq014-bootstrap --spec /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/REPLACE_RUN_ID.json --submit
```

production spec 必须是上述 `run_specs/` 的 direct-child、regular、non-symlink、read-only canonical JSON；
launcher 通过单个 no-follow fd 读取一次并把 retained bytes 贯穿 validation/sealing，不重开路径。第一条只
输出 exact commit-blob snapshot files plan、job/resource/thread plan 与 runtime metadata，不创建 run root、
snapshot receipt 或 rendered sbatch；第二条仅在前者证据通过且 operation 有效授权时物化并复核这些
submit-only artifacts。禁止 direct
sbatch，禁止执行 retired `/ZXC/ipv_estimation`、unmanaged `/ZXC/RQ014_recovery` 或外部 RQ010B code。
job name 固定含 spec SHA digest；declassification export 以 `zxc-rq014-export-` 开头，contract preflight
以 `zxc-rq014-pre-` 开头。Python 必须是 inventory 与
`managed_python_environment_v3.json` 共同绑定的 exact `python3.9` bytes。operator 的 clean shell 先校验
wrapper（8,108 bytes，SHA-256 `d80363...2d7a1d`）；clean bootstrap 先取得并锁定 fd8 runtime
lock，再把 exact managed wrapper 打开为 fd9，hash `/proc/$$/fd/9` 并执行同一个 fd9，禁止重新打开
wrapper path。wrapper 只能继承 fd8/fd9，缺失时不得自行补建；launcher 在任何 RQ014 materializer/
preflight preload 前复核两个 `/proc/self/fd` exact target、固定路径的 regular/non-symlink lstat，以及
descriptor/path 的 device、inode、mode identity。这个 local machine provenance gate is not a cryptographic secret
against deliberate same-account descriptor emulation；真正的 wrapper byte trust anchor 仍是 reviewed clean
bootstrap digest。caller 自行传入 `--rq014-only`、`True` 或任意对象均不能授权。wrapper 在第一次 managed
Python 启动前再用绝对系统工具校验 launcher（116,287 bytes，`6b3cf6...2bd91f`）、preflight（69,166 bytes，
`f91bbd...4102`）、registry materializer（11,502 bytes，`d8cac7...decc4`）、v3
environment manifest、完整 stdlib checksum manifest（14,326 个 regular files、307,357,072 bytes、0
symlink、无 special entry、`python39.zip` absent）与 20-row managed-native closure（16 个单级 loader
symlink、4 个 regular loader paths、14,656,296 resolved bytes）。`/lib64` 是明确的 OS ABI trust boundary，
不伪称由 native TSV 冻结。trusted launcher 随后在任何 scientific entrypoint 前校验 formal/final bundle，
并直接从 exact commit tree blobs 物化 snapshot；dirty/untracked worktree bytes 不能进入执行面。所有
BLAS/OpenMP thread limits 为 1。RQ014 v2 禁止覆盖 managed base。

operator 唯一支持的入口是上面的单条空环境 bootstrap；不得先在继承环境中运行 checksum、再另行调用
wrapper，也不得跳过 wrapper SHA trust anchor，且不得用 `$wrapper` path 替代已 hash 的 fd9 执行。
wrapper 拒绝 `HPC_SOCIALITY_ROOT`、`BASH_ENV`、`ENV`、`LD_PRELOAD`、`PYTHONHOME`、`PYTHONPATH`
与所有 `SBATCH_*`；Slurm directive 和提交命令均使用 `--export=NIL`，因为 `NONE` 可能隐式重建登录
环境，故明确禁止。2026-07-12 对 Tongji GNU `sha256sum --check --strict` 的只读兼容探针已通过；该探针
未提交 Slurm。RQ014 Python 只以 `-I -S -B -X utf8` 启动；`scripts` 与 `scripts.rq014` 的 `__path__`
保持为空，先用 `spec_from_file_location` 把 closed-snapshot exact reviewed
`materialize_registry.py` 预载为 `scripts.rq014.materialize_registry`，再按 exact reviewed path 装入
preflight 与固定 entrypoint，普通 path import 被禁止且 checkout/root/src 均不进入 `sys.path`。提交前失败回滚 partial snapshot/run root 并写
retryable `FAILED` receipt；`sbatch` 已开始后若状态不明则保留 namespace、原始响应与已知 job ID（若可
解析），写 non-retryable `SUBMISSION_STATE_UNKNOWN`，禁止盲重试。

## 12. v1.5 preflight 顺序

1. 本地 tests、schema、duplicate-key 与 checksum 验证；
2. fresh statistics + execution/governance review；
3. 生成 formal G1 artifact；
4. 将 exact reviewed commit 发布到 `origin/main` 并同步 managed checkout；
5. 对 declassification-export immutable v2 spec 运行 launcher validate-only；
6. 证据一致后，只提交 `rq014_g2_declassification_export`，生成并验证 canonical score-stripped bundle；
7. export receipt/DONE/Slurm completion/bounded report 通过后，另改中央 allowlist 开启已经条件式注册的
   `rq014_g2_contract_preflight`；中央 authority byte 变化后必须重建 candidate manifest、双路 review、
   formal G1 与 final bundle，但不再改 reviewed execution-contract status；
8. 物化 G2 input manifest 与 registry bindings，对 preflight v2 spec validate-only 后再提交；
9. 输出 bounded preflight report，等待 resource-pilot scoped decision；
10. 另行授权后，完整构建并封存 960-cell G2R feature bank；
11. 另行评分授权后执行一次 G3R full-data recovery screen，全部 2880 行 terminal 后冻结 rank 1；
12. 另行授权 fresh G4R worker 对唯一 frozen recipe 做 clean replay。

本方案编写/复核阶段不得创建 production run root 或提交 Slurm。

## 13. Stop conditions

除 base v1 条件外，任一以下情况立即停止：非法 terminal state；formal G1 missing/stale；中央 allowlist
缺失；run commit 不等于 managed HEAD 或未在 `origin/main`；spec/manifest duplicate key；路径越界、
symlink 或命中 forbidden root/pattern；任何 `contains_rating=true`；forbidden field audit 非 0；raw rated479
路径出现在 spec；17 个 registry bindings 不完整或不一致；A1–A4 被尝试在 G2 重算；direct sbatch；
retired checkout；job/resource/thread contract 漂移。

## 14. Residual risk

<a id="residual-risk"></a>

- F05/F06/F07/F08/F09/F10 未形成完整阴性扫描，因此旧设置可能仍在不可达档案；
- FL05 未执行，historical fingerprint 类结论在当前 run 不可达；
- score-stripped contract 能隔离 rating-bearing artifacts，但 rated479 universe 的选择本身来自既有评分
  项目；这不会泄露 rating values，却限制外推范围；
- A1–A4 在 G2 仅做 receipt integrity，真正数值 parity 延后；
- 当前只实现 declassification export 与 contract-preflight 边界，不代表 G2R、G3R、G4R 已获授权或具备预算；
- 同一 Unix 账户不能提供强 ACL 隔离，当前 blind boundary 依赖 exact reviewed entrypoint、无 rating 参数、
  source allowlist/hash 与 receipt chain；若未来执行通用 worker，必须另加容器/ACL 或等价隔离；
- 即使未来复现负相关，同一 WOD-E2E 数据上的 recovery 仍是内部关联证据，不是外部独立复制。

## 15. 通过定义

v1.5 当前首轮的成功不是得到相关系数，而是：合法关闭 G0；formal G1 对同一字节给出无 blocker 结论；
评分隔离、staged manifests、registry bindings 与 managed launcher 可机器验证；首个 allowlist 只允许
declassification export，随后才可按 receipt 开启 bounded preflight。主恢复路线已经冻结为完整全量屏幕
与 clean replay，但其 feature/rating/replay operations 均保持 DENY；任何经验结果仍为未运行。
