# RQ014 Plan v1.3 Amendment — execution-contract hardening

状态：`DESIGN_REVIEW_PASS / G0_OPEN / G1_NOT_REACHED / NO_EXECUTION_AUTHORIZED`
日期：2026-07-11（仍在任何 rating join、恢复计算或新结果之前）
基底：不可改写的 v1 计划 + v1.2 amendment；v1.1/v1.2 均保留为历史记录。
响应评审：`reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/codex_plan_review_v1p2_20260711.md`。

配套生效文件：

- `RQ014_config_space_v1p3.yaml`；
- `RQ014_forensic_registry_v1p3.yaml`；
- `RQ014_recovery_extension_registry_v1p3.yaml`；
- `prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.py`；
- `prompts/RQ014_forensics_hpc_fl05_indexer_v1p3.sh`；
- `prompts/RQ014_forensics_hpc_fl05_v1p3.sbatch`；
- `prompts/RQ014_forensics_hpc_pass4_v1p3.sh`；
- `prompts/RQ014_forensics_hpc_pass4_v1p3.sbatch`；
- `prompts/RQ014_forensics_mac_pass4_v1p3.sh`；
- `RQ014_plan_v1p3_checksums_20260711.sha256`。

三个 v1p3 registry 是完整 resolved registries，不要求执行者运行时合并 YAML。冲突优先级固定为：

| 旧条款 | v1.3 处置 |
|---|---|
| base v1 §5–10 valid grid、endpoint、split、promotion、confirmation association | 保留，除 v1p3 registry 明示字段外不改 |
| base v1 §11 specificity family | 由 `RQ014_config_space_v1p3.yaml#statistical_contract_v1p3.specificity_tiers` 整体取代 |
| base v1 §12 discovery/association verdict rows | 保留到 `VALID_INTERNAL_CONFIRMATION_ASSOCIATION`；旧 `...IPV_SPECIFIC` row 由 v1p3 Tier C/P/I ladder 取代 |
| v1.2 R2 extension inference/X02 | 由 v1p3 extension registry + statistical contract 整体取代 |
| v1.2 R4–R6 invariance、Tier C/P/I、calibration、margin | 由 v1p3 statistical contract 整体取代 |
| v1.2 R1/R8 FL05 与 pass3/pass3b closure | 由 v1p3 Python/shell/sbatch + pass4 整体取代 |
| 其余不冲突条款 | 继续有效 |

若 prose 与 resolved registry 冲突，以 v1p3 registry 为唯一机器权威并触发 review failure；执行者不得
自行选版本。本文是通过 targeted review 的**非执行设计合同**；G0 尚未运行，正式 G1 因 gate 顺序
尚未到达，所有授权均保持 false。

---

## R1 — FL05 改为 fixture-tested、原子发布的结构化索引器

v1.2 shell 内嵌 parser 废止，仅保留为历史。v1.3 使用独立 stdlib Python CLI，并由薄 shell wrapper
调用：

1. 同时支持 wide CSV、`statistic_name/value` long CSV、numeric-key JSON、long-record JSON 与
   Markdown statistic-local `key=value`；
2. value 必须是有限数值且相关系数范围为 `[-1,1]`；无法安全解释的候选必须输出
   `UNPARSED_CANDIDATE`，不得静默丢弃或标为 parsed；
3. Markdown 只读取统计关键词邻域内显式绑定的数值，禁止取整行第一个数字；
4. 每个扫描文件必须进入 audit，包括零命中文件；audit 至少含 path、bytes、SHA-256、mtime、
   parser status、parsed/unparsed row counts 与 error；
5. 每次运行写入 `bundle_root/generations/.staging-<generation_id>/`；完成 CSV、audit、manifest、DONE
   的 schema/hash/fsync 后，先原子 rename 为 immutable generation，最后只原子替换一个 `CURRENT`
   pointer；pointer 失败只允许留下不可见 orphan generation，旧 CURRENT bundle 必须保持一致；
6. root 缺失、没有至少一份非空 supported file、文件 stat/read 失败、unsupported encoding/format
   ambiguity 超出登记规则、output/audit 不完整均非零退出；bundle root 与任一 input root 在 lexical
   或 resolved path 上重叠也必须在任何写入前拒绝；
7. tests 必须覆盖 v1.2 review 的 long CSV/JSON、nonnumeric CSV、malformed row、`4 Hz` Markdown、
   missing root、zero-hit audit 与 atomic-failure 反例。

非空输入树若所有 supported files 都被完整读取并进入 audit，但没有任何 statistic candidate，合法发布
`COMPLETE_ZERO_CANDIDATES`；“零候选”本身不是 parser failure。空树、读取不完整或 audit 不完整仍
必须 fail closed。这样 F10 可对“完整扫描且确实全零”形成受限阴性结论。

Indexer 成功不等于 F10 已关闭。`COMPLETE_WITH_UNPARSED_CANDIDATES` 必须生成逐行
`fl05_adjudication.csv`：每条 unparsed candidate 要么回到 exact source locator 形成结构化解释，要么因
无法解析而把 F10 置为 `INACCESSIBLE`。只有 complete index、全部 unparsed 已处置且零 attributable
rating↔IPV fingerprint 时，F10 才可记 `NOT_FOUND_ON_SCANNED_SURFACES`。

登录节点只允许总输入 `<=200 MiB` 的只读索引。超过预算必须由 checksum-bound Slurm wrapper
`RQ014_forensics_hpc_fl05_v1p3.sbatch` 执行；job name 固定以 `zxc-` 开头，durable run root 固定为
`/share/home/u25310231/ZXC/RQ014_recovery/<SAFE_RUN_ID>/`。sbatch 只接受 safe RUN_ID，不接受任意
output/预算覆盖；运行前必须校验 v1.3 checksum manifest 与 parser/shell/sbatch bytes，独立写
Slurm preflight logs `/share/home/u25310231/ZXC/RQ014_recovery/zxc-rq014-fl05-%j.{out,err}` 与
run logs `<SAFE_RUN_ID>/logs/zxc-rq014-fl05-${SLURM_JOB_ID}.{out,err}`。wrapper 还必须固定
`/share/home/u25310231/.conda/envs/ipv/bin/python`（Python >=3.9），记录 realpath/version/SHA，并拒绝
继承的解释器 override。

## R2 — F05–F08 使用逐 surface、fail-closed closure

旧 pass3/pass3b 只能作为 discovery probes，不能产生合法 terminal state。v1.3 pass4 合同：

- F05 HPC：四个 pilot directories、phase3 root 与历史 job log 分别做 required-root/readability check；
- F06 Mac：sibling archived roots 与 paper Git history/tree；
- F07 Mac：登记的 Obsidian note candidates；
- F08 Mac：登记的 Cowork session roots/fulltext candidates；
- F05 每个 pilot dir/phase3 至少一个完整读取的非空 regular file，job log 也必须非空；空 inventory
  不能形成阴性 terminal state；
- F07/F08 只接受 schema `rq014-frozen-scope-v1p3` 的非空 JSON scope manifest；每个 entry 必须含
  absolute `source_path`、lowercase SHA-256、descriptive `snapshot_at_utc`，以及
  `git_blob_v1_content_integrity_only` 锚点（absolute repo、full commit ID、safe Git path）。脚本验证
  commit/blob 与 source bytes 的一致性，但 Git author/committer date 可回填，**不得**作为 cutoff 证据；
  只有在 `2026-07-10T00:00:00Z` 前由独立 TSA 对整份 scope manifest 原始字节签发的 RFC 3161
  `TimeStampResp`，并由 checksum-bound 的 OpenSSL 3、root、exact signer leaf、intermediates、policy、
  accuracy 与 revocation contract 完整验证后，才可能允许 `FOUND` 或
  `NOT_FOUND_ON_SCANNED_SURFACES`。manifest/CLI/environment 不得覆盖 trust material 或 cutoff，也不得
  回退到 Git 时间。v1.3 **没有**登记生产 TSA profile，也没有一份已知的 cutoff 前 whole-inventory
  receipt；因此 F07/F08 在本版本只能合法发布
  `INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT`（仍可保存已扫描候选作为非终局线索）。启用可信 receipt
  必须另发 checksum-bound amendment 并重新独立复审。两份 scope manifest 中任一份、当前 output tree、
  symlink-resolved alias 与当前 RQ014 repository 均不得进入扫描域；
- 每个 surface 写 `OUTPUT/<surface>/generations/<RUN_ID>/{manifest.csv,evidence.txt,status.json,DONE}`，
  完整校验/fsync 并原子 rename generation 后，最后只原子替换该 surface 的 `CURRENT` 文本指针；
  `--run-id` 必须为 safe ID，pointer failure 只能留下不可见的完整 orphan generation；
- canonical search 不使用 `head`/字节上限截断；大文件若只登记 hash 而未完整搜索，surface 不能判
  `NOT_FOUND_ON_SCANNED_SURFACES`；
- required root/read/search 失败时整体非零，status 保持 `OPEN` 或显式 `INACCESSIBLE`，不得伪装阴性；
- `FOUND` 需要命中内容、locator 与 source SHA；`NOT_FOUND_ON_SCANNED_SURFACES` 需要全部 required
  file `read_status=FULL_READ_OK`；`INACCESSIBLE` 需要原因与 residual-risk statement；
- tests 用 root override fixtures 验证 missing、clean、hit 三种路径，禁止连接真实 HPC/OneDrive。

F05 pass4 在 login node 只允许 preflight 总量 `<=200 MiB`；更大 inventory 必须使用 checksum-bound
`RQ014_forensics_hpc_pass4_v1p3.sbatch`，固定 job name `zxc-rq014-g0-f05`、durable output
`/share/home/u25310231/ZXC/RQ014_recovery/<RUN_ID>/g0_pass4`、4 GiB compute ceiling、固定输入 roots、
固定 Python 与 preflight/run 两级 durable logs。wrapper 只接受 safe RUN_ID；本文不提交该作业。

## R3 — Extension family 使用 scene-keyed single-step maxT

对 rating-blind eligible 的 available cells `K`：

1. G2P 在任何 rating join 前冻结 extension analysis partition：若 blind split/power gate 选出有效 split，
   只用其 discovery partition；若无 split 通过，则使用 full rated479 recovery-only partition，并永久
   禁用 confirmation wording。随后直接物化 `extension_union_master_scene_ids.csv`，记录 X01–X05
   与 V04 feature-complete flags；不得通过尚未存在的 `M_k` 反向定义，也不得混入 confirmation rows。
   授权 discovery join 时，每格唯一
   `M_k` = feature master 中 Xk+V04 完整且 rating/Xk/V04 均 informative 的 pairwise common set；
   association 与 exclusivity 共用这个 byte-identical `M_k`；`M_k` 在首个 statistic 前一次性
   hash/freeze，之后禁止增删 rows；
2. 第 `b` 次置换中，scene `s` 的三候选 permutation 由登记 domain、master seed、replicate 与
   `scene_id` 的 SHA-256 keyed draw 决定：取 64-bit word，经对 6 的 rejection sampling 后按六种
   lexicographic permutations unrank；同一 scene 在所有 cells 使用同一 permutation，cell 只读取
   自己的 subset，禁止用依赖库默认 shuffle 语义替代；
3. `theta_k=rho_WS,k`；`SE_k=SD_b(rho_perm,bk)`，只估计一次、replicate 内不重估；
   `T_obs,k=theta_obs,k/SE_k`，`T_bk=theta_bk/SE_k`；
4. 单格 single-step adjusted p：
   `p_adj,k=(1 + #b[min_j(T_bj) <= T_obs,k])/(B+1)`，`B=9,999`；omnibus p 同时报告但不能
   替代 `p_adj,k`；
5. 任一 active cell 的 observed SE<=1e-12、undefined replicate>1% 或 keyed synchronization audit
   失败，整个 extension family 为 `EXTENSION_NUMERICAL_INVALID`，禁止 outcome-driven shrink；
6. rating-blind unavailable cells 可在 join 前按登记原因收缩；active family 少于 3 格只做 descriptive。

## R4 — “exclusive versus V04” 改为配对对比，而非复用 omnibus p

每格在同一 `M_k` 上计算：

```text
Delta_k = rho_WS(Xk) - rho_WS(V04)
```

以 scene-keyed Poisson(1) paired bootstrap、固定 bootstrap SD 与 across-cell centered maxT 构造
97.5% simultaneous upper bound `U_Delta,k`。一个 cell 只有同时满足下列条件才可标
`EXPLORATORY_SPEC_RECOVERY_CANDIDATE`：

- `rho_WS(Xk) <= -0.30`；
- 同一 `M_k` 上 `rho_WS(V04) > -0.10`；
- association complete-null exploratory `p_adj,k <= 0.025`；
- paired exclusivity `p_adj_delta,k <= 0.025`；
- `Delta_k <= -0.20` 且 `U_Delta,k < 0`；

两个条件族以交集方式发 flag。因 varying supports 未主张 subset pivotality，association minT 只保证
complete-null exploratory adjustment，不宣称 cell-level strong FWER；flag 仍仅为 specification-recovery
候选，extension 不晋级、不确认、不修改 RQ010B。

## R5 — 单一 alpha 合同

全案所有 one-sided test（scientific、forensic、extension）名义 alpha 固定为 `0.025`。所有单侧
下/上界使用 97.5%。v1.3 不运行 inferential equivalence test；sign/role diagnostics 已按 R8 降级为
descriptive。任何 `p<0.05` 的 v1.2 extension wording 废止。G2 empirical size/FWER gate 的容忍
仍为 `<=0.03`，但此处只指 R6 登记的 scientific specificity DGP cells，必须按每格 worst-case
规则判断。forensic twins 不再声称一个未定义的 equality-boundary DGP gate；其 maxT 合同、fixture 与
numerical invalidation 单独登记，不能借用 R6 calibration label。

## R6 — Tier C/P/I 使用冻结联合 latent-rank DGP 校准

共同原则：保留真实 scene blocks、predictor joint dependence、missingness 与 ties；只模拟 ratings。
对每个 tier 的冻结 common set，将 predictor 转为 within-scene centered midranks并标准化，生成：

```text
U_sc = - X_sc beta + epsilon_sc,  epsilon ~ N(0,1)
rating = within-scene rank/tie_operator(U)
```

`beta` 用冻结版本的 `scipy.optimize.least_squares(method=trf)` 与 common-random residuals 校准到登记的
raw `rho_WS` target vector：100,000 fit draws + 100,000 independent validation draws，参数 bounds
`[-8,8]`、`max_nfev=5000`、validation 最大绝对 target error `<=0.005`。不收敛或 target infeasible
则对应 tier `CALIBRATION_UNAVAILABLE`，禁止以最近可行 target 替代。正式 size/power 使用独立 RNG
domains。

- strength grid：`g in {0.10,0.20,0.30}`；tie profiles：`{observed_exact, stress_0p20,
  stress_0p40}`；
- Tier C equality boundary：`S_IPV=S_shuffle=S_swap=g`；
- Tier C partial-null：分别令一个 advantage 在 0 边界、另一个为 +0.10 alternative，两个成员轮换；
- Tier P NI boundary：`S_K=g, S_IPV=max(0,g-0.08)`；当 `g<0.08` 的未登记外推不得执行；
- Tier P equal-strength power：`S_IPV=S_K=g`；
- Tier I null：latent DGP 中 Q coefficient 精确为 0，K-only rating strength `S_K=g`，保持实测
  Q–K dependence，不强迫 marginal Q–rating correlation 为 0；
- global-zero sanity：Tier C/P/I 三个 frozen manifests × 三个 tie profiles，共 9 个 report-only cells，
  每格同样 20,000 outer 且运行 exact registered tier test；不替代 boundary gate，但计入 compute 投影。
  其中 Tier P 明确标为 `equal_zero_NI_behavior_sanity`，不是 Type-I null cell；
- DGP 只统计不含任何 calibration gate 的 `CORE_REJECT`（C：两 placebo core tests；P：p/LB；I：
  beta sign+p/LB）；final tier PASS 才在 CORE_REJECT 之外叠加 size/power gate，禁止递归；
- 每个 boundary/power cell 固定 20,000 independent evaluations，每次运行完全相同的登记检验；
- seed 由 registry 中冻结的 SHA-256 domain separator 与 tier/grid/tie/draw/replicate fields 生成，
  完整写入 manifest，fit、validation、outer 与 inner streams 禁止重叠；
- size/FWER：每个 cell 的 point estimate 与 one-sided 95% binomial upper bound均 `<=0.03`；
- NI power：每个 cell 的 one-sided 95% Clopper–Pearson lower bound `>=0.80`；
- **worst-case gate**：所有登记 `g × tie` cells 均通过，禁止以平均值过门。

Timing 拆为三段：G2 评分盲阶段冻结全部 predictor arrays、代码、seed domains、tie operators/stress
profiles 与 runtime pilot；discovery 完成并 deterministic promotion 后、confirmation rating access 前，
只在 frozen confirmation partition 上选择已冻结的 promoted/control/composite arrays，物化并 hash
Tier C 与 Tier P/I feature masters；confirmation join 获单独授权后，再由 nuisance-only loader 一次性
物化 completeness/tie analysis manifests，并在任何 Tier C/P/I observed statistic 前运行冻结的
`G4S_PRE_SPECIFICITY_CALIBRATION`。loader 禁止输出 observed rating order、rho 或 effect direction。

计算合同：每个 tier/manifest 的 9,999 个 scene-resample operators 由
`tier_id + manifest_sha256 + bootstrap_replicate` 唯一生成，预计算一次，并在 observed test 与所有
DGP outer evaluations 中复用同一冻结 operator matrix；inner seed 不含 outer replicate。
numerical failure 在 size grid 保守计为
false positive、在 power grid 计为 failure，固定 denominator 20,000，禁止失败后补抽或缩分母。评分盲
resource pilot 分别跑 Tier C/P/I 三种真实 test shape：每种 3 次 `500 outer × exact 9,999 inner`，并按
C/P/I 各自 1,200/840/480 batches 分层投影。fit/validation 也按 3/2/1 参数的 C/P/I optimizer shapes
分别做 deterministic 479-scene fixture；fixture 的 predictor 值由登记 SHA-256 generator 生成并保证
对应维度 full rank。每个 shape 强制执行 5,000 次 exact objective/Jacobian cost path（禁止 early stop），
另加同 fixture 的 SciPy wrapper overhead；每种 3 次，分别乘 27/18/9 targets。所有 9,999-size operator
types 也完整生成/hash 一次。
54 个 boundary/power cells 加 9 个 global-zero cells 共 63 cells、每格 40 个 500-outer batches，固定
2,520 outer batches。总 CPU 还必须加上全部 100,000-draw fit、100,000-draw validation、operator
generation 与 manifest validation 的实测/保守投影；每个 outer-batch/fit/validation/operator task 的
memory 直接用全部 shapes 的最大 RSS。wall time 按每 tier 的 max elapsed ×
`ceil(task_count/frozen slots)` 求和，再加 serial critical path。
`G2_compute_resource_pilot_manifest.json` 必须登记 registry/code/env/generator/fixture/operator hashes、
各 shape 三次 CPU/elapsed/RSS、outer/fit slots、cores/memory/BLAS=1、scheduler allocation 与完整投影公式；
manifest exact bytes 在 G4S 前 hash。只有 aggregate
CPU `<=20,000 CPU-hours`、每 task peak RSS `<=16 GiB`、资源分配后 projected wall time `<=96 h`
才可运行 full 20,000；超预算不得减少 replicates、早停或 sequential peeking，相关 tier 记
`CALIBRATION_COMPUTE_INFEASIBLE`。

## R7 — X02 增加定义、支持与量化尺度资格门

X02 保留“sigma01 static envelope + registered rolling-local candidate IPV”这一 old-spec 候选，但只有在
评分盲 G2 同时通过下列 gate 后才能进入 extension family：

- exact recovery：sigma01 risk-bin edges/edge semantics、PET proxy formula/units/frame、CP/HO/MP/F
  geometry mapping、legacy estimator/window semantics、static builder 与 source manifest 全部 hash；任一
  模糊 ⇒ `INACCESSIBLE_DEFINITION`，禁止 analyst-designed substitute；
- source risk bins 固定为 PET `<=1.0`、`(1.0,1.5]`、`(1.5,2.0]`、`>2.0 s`；
- WOD mapping 必须使用 exact recovered、评分盲的同语义 proxy；
- coverage denominator = **全部** pre-envelope trajectory-eligible scenes 的 scheduled candidate-tau
  rows；先要求 recovered PET proxy+geometry calculable fraction `>=0.70`，再要求映射到有限 static
  L/M/U 与有限 X02 IPV 的 fraction（同一全量 denominator）`>=0.70`；
- parity manifest：CP/HO/MP/F 每类按
  `SHA256("RQ014-v1p3-X02-parity|scene_id|candidate_id")` 取前三个，共 12 cases，并覆盖其全部
  scheduled tau；不足 12 ⇒ `INELIGIBLE_SUPPORT`；
- 不允许 rescaling/affine correction。finite paired fraction 在任何 complete-pair conditioning 前、以
  12-case manifest 的全部 scheduled rows 为 denominator，必须 `>=90%`；同时至少 10 个
  distinct rows、每个 static half-width `>1e-6 rad`、median normalized difference `<=0.50`、
  q90 `<=1.00`、absolute normalized signed bias `<=0.25`、legacy-vs-rolling Spearman `>=0.80`；
- Spearman 只在上述 finite paired rows 上按 average midranks 计算；任一向量常数、少于 3 个 distinct
  pairs、NaN/nonfinite result 均直接 `INELIGIBLE_SCALE_INCOMPATIBLE`，实现版本/SHA 在 G2 冻结；
- 任一尺度门失败 ⇒ `INELIGIBLE_SCALE_INCOMPATIBLE`，在 rating access 前收缩 family。

## R8 — Invariance diagnostics 降级为完全描述性、非门控

结构 alias 必须逐 scene 的三候选 deviation rank vector 完全相同（浮点 value 可另报 max difference）；
全局 Spearman 只作描述，不得单独判 alias。非 alias 时，
在各自 pairwise common scenes 上报告 `delta_flip=S_flip-S_IPV`、9,999 次 paired scene bootstrap
ordinary percentile 95% descriptive interval，并将 `[-0.10,0.10]` 仅作为 reference margin：

- interval 完全落入 margin ⇒ `DESCRIPTIVELY_COMPATIBLE_WITH_MARGIN`；
- interval 完全高于 0 ⇒ `FLIP_STRONGER_DESCRIPTIVE`；
- 其余 ⇒ `ASYMMETRY_DESCRIPTIVE`；
- 禁止使用 `INVARIANT`、`EQUIVALENT`、`DIRECTION_ARTIFACT_SUSPECT`；
- diagnostics 不能升级、降级或封顶任何 scientific label，也不能自动触发 forensic；
- role envelope unavailable ⇒ `UNAVAILABLE_NON_GATING`。

## R9 — Tier P/I 共用冻结 paired manifest

deterministic promotion 后、任何 confirmation rating access 前，在 frozen confirmation partition 上
物化 `tier_PI_feature_master_ids.csv`：promoted endpoint 与 frozen composite 均 feature-complete 的 scene
IDs。授权 confirmation join 后只按预先登记的 rating-informative 规则生成唯一
`tier_PI_analysis_ids.csv` 与 attrition ledger；config 中的 `tier_P_common_scene_manifest.csv` 与
`tier_I_common_scene_manifest.csv` 必须是该 analysis file 的 byte-identical aliases。discovery partition
rows 禁止进入上述文件。P/I 均在这一集合上重算 `S_IPV`、`S_K` 和 centered midranks。composite
精确公式或 code SHA 恢复不了时，P/I 同时 `UNAVAILABLE`。

## R10 — Tier I null calibration 与共线性政策

Tier I 按 R6 在 `beta_D=0` 的 K-only boundary 校准。Observed 若 `rank<2` 或任一标准化设计列
variance `<=1e-12`，记 `TIER_I_UNAVAILABLE_SINGULAR`；否则若 `|corr(Q,K)|>=0.95` 或 condition
number `>40`，记 `TIER_I_UNAVAILABLE_COLLINEAR`。Bootstrap 若
`rank<2`、variance `<=1e-12` 或 condition number `>1e8` 才记 undefined；`40<cond<=1e8` 仍计算但
逐 replicate 审计。undefined 比例 `>1%` 或 empirical size 上界 `>0.03` ⇒
`TIER_I_NUMERICAL_INVALID`。SSE check 继续仅作 sanity。

## R11 — Specificity label ladder 与 overall verdict 唯一化

base v1 的 `VALID_INTERNAL_CONFIRMATION_IPV_SPECIFIC` 永久退役。只有 base v1 confirmation association
先通过，才进入以下唯一 ladder：

| Tier 终态 | 唯一 overall verdict |
|---|---|
| Tier C unavailable / numerical invalid | `VALID_INTERNAL_CONFIRMATION_ASSOCIATION`，注记 `TIER_C_NOT_ASSESSED` |
| Tier C 有效运行但未通过 | `VALID_INTERNAL_CONFIRMATION_ASSOCIATION`，注记 `PLACEBO_SUPERIORITY_NOT_ESTABLISHED` |
| Tier C pass；Tier P unavailable / underpowered / fail | `VALID_INTERNAL_CONFIRMATION_ASSOCIATION_ROBUST` |
| Tier C+P pass；Tier I unavailable / invalid / fail | `VALID_INTERNAL_CONFIRMATION_PARSIMONY_VS_FROZEN_KINEMATICS_COMPOSITE` |
| Tier C+P+I 全 pass | `VALID_INTERNAL_CONFIRMATION_INCREMENTAL_BEYOND_FROZEN_KINEMATICS_COMPOSITE` |

Tier C 的 pass label 固定为 `VALID_INTERNAL_CONFIRMATION_ASSOCIATION_ROBUST`；Tier I fail 不撤销 Tier P。
R8 invariance diagnostics 完全非门控，不得改变 ladder。Tier C 未评估或未拒绝均不得自动推断
`CONSTRUCTION_SUSPECT`；只有另有登记的阳性反证才允许使用该措辞。

## R12 — Historical fingerprint 与 NI margin 措辞收紧

FL05 的 `rho<=-0.30` 只先生成 candidate。升级为
`HISTORICAL_RATING_IPV_FINGERPRINT_CANDIDATE` 还必须同时满足：

- variable pair 明确包含 rating/preference/score 与 IPV/deviation/envelope；
- WOD-E2E candidate-level 或 within-scene scope；
- statistic direction、N、unit、source locator、source SHA 与 config context 完整；
- parse status 为结构化 parsed，而非 unparsed text。

否则仅记 `UNATTRIBUTED_NEGATIVE_CORRELATION_CANDIDATE`，不能关闭“旧结果已找到”。

`delta_NI=0.08` 保留，但依据改为**评分前 PI 选择的 normative maximum acceptable loss**，不是
“RQ010B 最弱物理效应的一半”的经验估计。0.04/0.12 只作登记 sensitivity，不改变 primary margin，
不发 primary label。

## R13 — 版本、checksums 与授权

v1.3 checksum manifest 必须覆盖：base v1、本文、三套 v1p3 registry、FL05 Python/shell/sbatch、
HPC pass4 shell/sbatch、Mac pass4、三份 fixture/contract tests。运行后再追加
output/audit/status/manifests 的 hashes。

本 amendment 只有在以下条件全部满足后才可取代 v1.2 成为当前**非执行设计合同**：

1. YAML/CLI/schema/tests/checksums 全部通过；
2. fresh statistics reviewer 与 fresh execution reviewer 均无 blocker；
3. final no-blocker review 通过；
4. START_HERE/STUDIES/workflow log 登记 v1.3 verdict。

这四项通过不等于 formal G1 PASS，也不产生执行权限。成为 executable candidate 还必须先由授权的
只读取证把 G0 所有 OPEN surface 变为合法 terminal state，再按 base gate 顺序完成 formal G1，并由
后续 scoped decision 明确开启相应授权；F07/F08 缺少 cutoff 前 whole-inventory receipt 时只能以
`INACCESSIBLE_NO_PRE_CUTOFF_SCOPE_RECEIPT` 关闭，不能声称阴性扫描。

授权拆成八个独立布尔状态：`g0_readonly_forensics_authorized`、`g2_ratings_blind_build_authorized`、
`discovery_rating_join_authorized`、`confirmation_rating_join_authorized`、`scientific_compute_authorized`、
`extension_compute_authorized`、`forensic_rating_join_authorized`、`forensic_compute_authorized`。
本文发布时八者均为 false；G1 PASS 不自动翻转任何状态，必须由后续 scoped decision 逐项授权。
operation map 中列出的 scopes 一律为逻辑 AND：G0 pass4/FL05 只需要 G0 scope；G2 feature/envelope/X02
build 与 G2P partition freeze 需要 G2 build；G2 resource pilot 与 G2P pipeline-power simulation 需要
G2 build + scientific compute；valid discovery 需要 discovery join + scientific compute；post-promotion
pre-confirmation feature freeze 需要 G2 build + discovery join + scientific compute；extension statistics
需要 discovery join + extension compute；confirmation association/G4S/C/P/I 需要 confirmation join +
scientific compute；FD01/FT01–FT04 需要 forensic rating join + forensic compute。任一所需 scope 为 false
即 fail closed，且不同 arm 的 compute scope 不得互相替代。
G0/F05–F10、G2 hashes/parity 与 G2P 必须按 base gate 顺序关闭；G4S calibration 只能在 confirmation
join 单独授权后、tier observed tests 前执行。本文不授权 rating join、HPC execution 或结果计算。
