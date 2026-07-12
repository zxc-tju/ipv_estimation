# RQ014 Plan v1 — WOD-E2E 评分与 rolling IPV-envelope 偏离结果找回

状态：`DRAFT_FOR_INDEPENDENT_REVIEW`  
日期：2026-07-10  
适用范围：研究计划与注册表制定；**尚未授权计算、评分 join 或 HPC 作业提交**  
取代关系：本文件取代 v0 作为未来执行的候选计划；v0 保留为历史记录，不删除、不覆盖。  
配套注册表：

- `reports/plans/RQ014_config_space_v1.yaml`：12 个 non-alias scientific-valid core configs；
- `reports/plans/RQ014_forensic_registry_v1.yaml`：取证面与小型 one-factor forensic catalog；
- v0 independent review：`reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/codex_plan_review_v0_20260710.md`。

## 0. Executive decision

RQ014 的直接目标不是再次提出“IPV 偏离可能与评分有关”这一假设，而是找回一个据 PI
记忆确实曾经出现过的实验结果及其可执行设置：**同一 WOD-E2E 场景中，评分越高的候选轨迹，
其 rolling IPV 越少偏离 human IPV envelope；评分越低，偏离越大。**

v1 把工作拆成两个互不借用证据的任务：

1. **Specification recovery**：在一个很小、逐行可枚举、物理时间一致的 valid family 中，
   找回可能产生旧信号的 rate/window/sequence/horizon 设置；
2. **Scientific validation**：只有在 rating-blind coverage 与完整 pipeline power 过门时，
   才允许 discovery → 单 recipe → same-dataset internal locked confirmation。

历史缺陷、pooled statistics、旧 pilot 和实现异常只进入独立 forensic family。它们不能参与
valid leaderboard、promotion、alpha family 或 RQ010B claim amendment。

v1 对三种情况都给出合法终局：找回并内部确认；只在 discovery/forensic 中出现；在冻结范围内
未找回或因功效/覆盖不足无法判断。**任何阴性结果都不得被描述为证明记忆错误，也不得自动加强
RQ010B 的 bounded null。**

## 1. 已知事实与研究边界

### 1.1 当前可复现基线

RQ010B 已接受的 WOD-E2E 结果为 bounded/underpowered null：

- 4 Hz future-only：`n=75`, deviation↔rating `rho=+0.147676...`；
- 4 Hz history+future：`n=98`, `rho=+0.031298...`；
- 10 Hz future-only：`n=75`, `rho=+0.164817...`；
- 10 Hz history+future：`n=47`, `rho=+0.127534...`；
- IPV 未胜过弱运动学控制；
- 4 Hz 与 10 Hz 的 candidate IPV rank correlation 仅约 `0.29–0.31`；
- M3 在 WOD-E2E 上的 in-support 比例不超过约 15%，不能作为本研究 valid primary envelope。

这些数是 G2 parity anchors，不是 RQ014 的 rolling-envelope endpoint。RQ010B 当前主要 quantity
接近 terminal/final candidate IPV 的绝对偏离；RQ014 要恢复的是**沿候选未来逐时估计 IPV，随后
对 envelope exceedance 做物理时间聚合**的 verifier。

### 1.2 允许与禁止的结论

即使最高 gate 全部通过，RQ014 也只能声称：

> WOD-E2E 同一评分数据上的 specification recovery 与程序性锁定内部确认。

不得声称：independent-data replication、因果验证、闭环安全验证，或对 RQ010B 已执行
operationalizations 的静默推翻。clean-room 重实现只证明 computational reproducibility。

RQ014 若产生不同结果，只能在新的 `reports/knowledge/RQ014_*/decision.md` 中提出 scoped
addendum：说明新的 rolling endpoint、有效配置和边界，同时保留 RQ010B 原有 bounded null。

## 2. 两个隔离的 analysis families

### 2.1 Scientific-valid family

唯一问题：

> 在同一场景的三条候选轨迹中，评分较高的候选是否具有更低的、预先定义的 rolling
> IPV-envelope deviation？

固定条件：

- unit = candidate；group/resampling unit = segment；
- primary scene 必须有同组三条完整候选；
- 三候选共享一个 rating-blind、真实观测的 counterpart；
- counterpart 只允许在观测支持内插值，禁止外推；
- 三候选使用同步 common time support；
- estimator dt 必须与 trajectory dt 匹配；
- reference、solver、sigma、IPV grid、reliability domain 全部固定；
- envelope 必须与 estimator rate/window/sequence/time scale 匹配；
- pooled、scene mean、history-only、driven trajectory、candidate-internal envelope、M3 OOD
  均禁止进入本 family。

### 2.2 Forensic family

只回答旧结果或假信号可能如何形成。允许检查：

- 2026-06-29 四个 pilot 目录及其中间统计；
- dt mismatch；
- candidate-as-own-reference；
- probability-domain likelihood underflow / uniform fallback；
- CV-extrapolated counterpart；
- pooled raw-IPV 与旧 mean-within-scene statistics。

Forensic family 使用单独的 registry、common-support twin、multiplicity 和 verdict。无论效应多强，
forensic cell 都不得晋级 scientific confirmation。

FT01–FT04 的机制统计量固定为
`Delta_m = rho_WS(defect_m) - rho_WS(clean baseline)`。负值表示 defect 使关系更负。每一对必须
使用完全相同的 scenes、candidates、timestamps 和 endpoint，只改变 registry 声明的一个字段。
四对 twins 各做 9,999 次 paired scene bootstrap，并同步重算 clean/defect。定义
`se_m=SD(Delta_m^b)`、`Z_bm=(Delta_m^b-Delta_m_obs)/se_m`；以四项
`max_m Z_bm` 的 95th percentile 为 `c_0.95`，simultaneous upper bound 为
`U_m=Delta_m_obs+c_0.95*se_m`。

某 pair 只有同时满足以下条件才能得到 `ARTIFACT_MECHANISM_SUPPORTED`：defect
`rho_WS<=-0.30`、`Delta_m<=-0.20`、`U_m<0`、common-support N 不低于 clean valid analysis
的 80%，且 solver/fallback health 变化完整报告。bootstrap `se_m=0` 或无定义比例超过 1% 时为
numerical failure。RNG 使用 NumPy `PCG64`，seed=`20260710+forensic_pair_index`。该证据只支持
defect mechanism，不证明它就是历史结果。

## 3. Gate architecture

| Gate | 必需产物 | PASS 后允许 | FAIL/不足后的合法动作 |
|---|---|---|---|
| G0 Forensic closure | closure matrix + hashes | 冻结 forensic fingerprint | 写明 `INACCESSIBLE`；不得写无条件 NOT_FOUND |
| G1 Plan/registry | v1 + 两 registry + independent review PASS | 开始 G2 ratings-blind build | 不得提交搜索作业 |
| G2 Provenance/parity | input/env/code hashes、legacy anchors、verifier fixtures、benchmark | 构建 valid features/envelopes | 修复 harness；不得读取评分 |
| G2P Eligibility/power | blind coverage、结构 alias、pipeline power | 选择 confirmatory split 或 recovery-only branch | 只允许 historical/specification recovery |
| G3 Discovery | 完整 ledger + deterministic promotion | 最多冻结一个 valid recipe | 无晋级则停止 scientific branch |
| G4 Confirmation | one-shot access audit + primary result | 运行 specificity battery | 主检验失败即停止 scientific claim |
| G5 Validation | red team + clean-room verifier + decision review | 形成 RQ014 decision | 降级 verdict，不救援主结果 |

任何 gate 的规则只能通过**看新评分结果之前**的版本化 amendment 修改，并再次独立复审。

## 4. G0：Phase F 闭环

G0 terminal states 只允许：`FOUND`、`NOT_FOUND_ON_SCANNED_SURFACES`、`INACCESSIBLE`；执行中的
transient state 为 `OPEN`。关闭时不得残留 `OPEN`。

| ID | Surface | v1 起草时状态 | 闭环要求 |
|---|---|---|---|
| F01 | 本仓库、Git、archived、workflow log | `NOT_FOUND_ON_SCANNED_SURFACES` | 记录扫描命令、日期、输出 SHA |
| F02 | 旧 paper snapshots/history | `NOT_FOUND_ON_SCANNED_SURFACES` | 记录 commit 范围与结果 SHA |
| F03 | HPC accounting/history/home/ZXC | `NOT_FOUND_ON_SCANNED_SURFACES` | 明确 sacct 保留窗边界 |
| F04 | Mac Codex/Claude/history | `NOT_FOUND_ON_SCANNED_SURFACES` | 保存候选清单 SHA |
| F05 | 2026-06-29 四个 pilot 中间目录 | `OPEN` | 分时只读重试；失败后正式转为 `INACCESSIBLE` 并记录错误与优先恢复动作 |
| F06 | sibling archived + paper Git pass3b | `OPEN` | 定向 Git/`rg` 扫描，避免 OneDrive 全盘遍历 |
| F07 | Obsidian `WODE2E.md` 正文 | `OPEN` | 只读审计或记为 `INACCESSIBLE` |
| F08 | Cowork 候选会话全文 | `OPEN` | 时间盒扫描或记为 `INACCESSIBLE` |
| F09 | 旧 Windows / OneDrive history/trash | `INACCESSIBLE` | 非阻塞；保留 residual-risk statement |

G0 可以 `CLOSED_WITH_INACCESSIBLE_SURFACES`，但报告只能说“已扫描面未找到”。若 F05 找到旧
rho/N/figure/config fingerprint，它只能更新 forensic fingerprint 和优先级；不得在看到新结果后
修改 valid registry。

## 5. Valid configuration registry

### 5.1 12 个非别名 core configs

`RQ014_config_space_v1.yaml` 逐行冻结：

初始 16-cell 析因设计中，`W=1.0 s` 且从 `tau=1.0 s` 开始时，future-only 与
history+future 都严格等于 `[tau-1,tau]`。四对 rate×horizon cells 是 deterministic aliases，
在查看评分前删除。最终 registry 为：

```text
2 rates × [W1.0 trailing-only × 2 horizons] = 4
+ 2 rates × [W2.5 × 2 sequence modes × 2 horizons] = 8
= 12 non-alias core configs
```

每个 config 固定三个 readouts，因此 materialized registry 严格为 36 行；只有
`mean_normalized_exceedance_90` 是 recipe selection 与 confirmation 的 primary readout。
`frac_outside_90`、`max_normalized_exceedance_90` 仅作机制与健康度支持，不能救援主结果。

### 5.2 物理时间而非固定样本数

`trailing_window_s` 是**最大回看上限**。评分从候选未来 `tau=1.0 s` 开始：

- future-only window：`[max(0, tau-W), tau]`；
- history+future window：`[tau-W, tau]`，`t<0` 使用 real ego history，`t>=0` 使用 candidate；
- counterpart 在两种模式下都使用同一个真实观测 track；
- `W=2.5, H2, future_only` 是 1–2 s 的 expanding prefix，**不是完整 2.5 s window**。

样本语义：

| rate | W | `history_window_steps` | 点数 | 物理跨度 |
|---:|---:|---:|---:|---:|
| 4 Hz | 1.0 s | 4 | 5 | 1.0 s |
| 4 Hz | 2.5 s | 10 | 11 | 2.5 s |
| 10 Hz | 1.0 s | 10 | 11 | 1.0 s |
| 10 Hz | 2.5 s | 25 | 26 | 2.5 s |

`evaluation_start_s=1.0` 只定义哪些 terminal times 被评分，不直接传给 estimator。每个 tau 都
独立构造一个 local physical window：

- `local_span_s = min(W,tau)`（future-only）或 `W`（history+future）；
- `n_intervals = round(local_span_s*rate_hz)`；`n_points=n_intervals+1`；
- 调用 `estimate_ipv_pair` 时，`history_window=n_intervals`，
  `min_observation=n_intervals`；
- 只读取 local output 的 terminal row `index=n_intervals`。

这避免把全局 tau index 与 rolling local-window index 混为一谈。golden fixtures 必须覆盖
4 Hz 的 1 s→4 intervals/5 points、2.5 s→10/11，以及 10 Hz 的 10/11、25/26。

### 5.3 固定实现条件

- sigma = 0.1；
- seven-point grid = `[-3,-2,-1,0,1,2,3] * pi/8`；
- solver = exact legacy-accurate path；
- likelihood = stable log-domain；
- smoothing = none；
- ego reference = frozen WOD route-based constant-curvature reference；
- input 4 Hz 保持 native；10 Hz 只对位置在观测支持内插值，再重算 velocity/heading；
- 不插值 IPV，不外推 counterpart，不重跑 StreamPETR。

core commit、WOD adapter、preprocessing 和 reference builder 的 SHA 在 G2 填入。未填入前
registry 保持 `PROPOSED_NOT_EXECUTABLE`。

## 6. Rolling verifier measurement contract

对 config `k`、scene `s`、candidate `c`，在每个 scheduled future time `tau` 独立构造精确物理
window，得到 candidate ego IPV `v_sck(tau)`。不得把 terminal IPV 重复当成一条 time series。

### 6.1 Common support

scheduled grid 为 `tau = 1.0, 1.0+dt, ..., H_s`：

- H2：`H_s=2.0 s`；
- HMAX5：`H_s=min(5.0, 三候选与共享 counterpart 的共同观测终点)`；
- 两个 horizon 都要求 `H_s>=2.0 s`；
- interpolation 只允许在源轨迹支持内；
- 若任一候选在某 tau 为 NaN、solver failure、uniform fallback 或无 counterpart support，
  同一 tau 从三候选全部同步删除；
- common grid 覆盖率必须 `>=0.80`，最大内部缺口 `<=2*dt`；否则整场对该 config 缺失；
- H2 必须保留 1.0 s 与 2.0 s 两端；HMAX5 必须保留 1.0 s 与实际共同终点；
- 两候选可用的场景只能进入 coverage appendix，不能进入 primary statistic。

### 6.2 Matched time-indexed human envelope

旧的单一 `pathtype_hv_norm.csv` 仅作 G2 anchor，不能自动用于 12 个 configs。valid family 必须从
InterHub pure HV-HV 重建同尺度 envelope：

```text
E(path_type, rate_hz, W, sequence_mode, tau,
  estimator_tree, adapter, solver, grid, preprocessing)
```

human builder 的确定性合同如下：

1. input eligibility：pure HV-HV、冻结 path-type mapping、两车有限位置、源轨迹内部 gap
   `<=2*source_dt`，且完整覆盖所需 local span；禁止外推；
2. terminal anchors：把每个 eligible human episode 在目标 rate 的观测支持内重采样，使用所有能
   提供完整 local window 的 grid timestamps；这些 timestamps 即 pseudo decision anchors；
3. span：W1 trailing-only 为 1 s；W2.5 future-only 在 `tau<W` 为 tau、之后为 W；W2.5
   history+future 始终为 W；
4. weighting：若 episode `e` 在某 path-type×tau cell 有 `m_e` 个 terminals，每个 terminal 权重
   `1/m_e`，使每个 episode 总权重为 1；
5. weighted quantile：按 `(ipv_value, episode_id, terminal_timestamp)` stable sort，取加权经验 CDF
   首次达到 q 的最小值（left-continuous inverse）；
6. bootstrap：seed `20260710`，在 path type 内对 episodes 有放回抽 1,000 次，每次重建 terminal
   weights 和 q05/q50/q95；half-width relative SE = bootstrap sample SD / 原始 half-width；
7. tau grid 与对应 config 完全一致。H2 与 HMAX5 共用 time-indexed envelope，H2 只读取至 2 s；
8. 任一 config 所需的任一 path-type×tau cell 未过 gate，则**整个 config**在 rating join 前标为
   `INELIGIBLE_ENVELOPE_*`，不做 outcome-dependent scene exclusion。

以上 builder spec materialize 后生成 `envelope_builder_contract.json` 并登记 SHA-256。若实现改为
time-pooled static band，则 horizon 必须进入 envelope ID，且需按 12 个非别名 configs 重建，不能混用。

每个 cell 冻结 central 90% band：

```text
L = weighted q05, M = weighted q50, U = weighted q95
```

Blind envelope gate：

- 每个 path type × tau 至少 50 个独立 human episodes；
- `L < M < U`；
- `M-L` 与 `U-M` 均大于 `1e-6 rad`；
- 1,000 次 episode-cluster bootstrap 后，每个 half-width 的 relative SE `<=0.25`；
- 不支持的 path-type×tau 使对应 config 整体在 rating join 前失效；
- 任一 recipe 若无法建同尺度 envelope，状态为 `INELIGIBLE_ENVELOPE_*`，保留失败行，不得在
  看评分后换 envelope 或偷偷转入 forensic。

### 6.3 唯一 primary endpoint

非对称 normalized exceedance：

```text
v < L:        e(tau) = (L-v)/(M-L)
L <= v <= U: e(tau) = 0
v > U:        e(tau) = (v-U)/(U-M)
```

候选级 primary deviation：

```text
D_sc = trapezoid_integral[e_sc(tau), tau] / common_duration_seconds
```

支持 readouts：

- `frac_outside_90`：按物理时间加权的 `e(tau)>0` 比例；
- `max_normalized_exceedance_90`：common support 上的最大 `e(tau)`。

## 7. Primary estimand and statistic

对 config `k` 依次报告：

- `N_input`；
- `N_feature_complete`：三候选 endpoint 全部存在；
- `N_rating_complete`：三候选评分存在；
- `N_informative`：三候选评分至少两个不同值，且 deviation 至少两个不同值。

primary analysis 只使用 `N_informative`。全 rating tie 或全 deviation tie 的场景进入 attrition 表；
不补值，不降为两候选主分析。partial ties 使用 midrank。

对每个 informative scene：

1. 将三候选 rating 升序 midrank 为 `R_sc`；
2. 将三候选 deviation 升序 midrank 为 `Q_sc`；
3. 分别在场景内中心化；
4. 计算该场景的 Pearson correlation `r_s`；
5. 主统计量为场景等权平均：

```text
rho_WS = mean_s(r_s)
```

高 rating 对应高 rank，高 deviation 对应高 rank，因此记忆方向为 `rho_WS < 0`。拼接全部 centered
ranks 的相关、pooled candidate correlation、Kendall、Bradley–Terry 和 top-hit 只能作支持分析。

## 8. Rating isolation, split and power

### 8.1 程序性隔离

项目历史上已打开过全量 ratings，因此“sealed”只表示本次 loader 隔离。G2 feature/envelope worker
不得挂载或读取任何 `contains_rating=true` 文件。每次 rating access 写入 append-only
`rating_access_log.jsonl`。

split 在任何 deviation 与评分值 join 前生成：

- split universe = frozen rated479 segment manifest 中的全部 segment IDs；缺失 scenario cluster 的
  segment 进入固定 `UNMAPPED` stratum；
- key = frozen `segment_id`；
- stratum = 既存且 config-independent 的 `scenario_cluster`；
- sort key = `SHA256("RQ014-v1|20260710|" + segment_id)`；
- 所有 valid configs 使用同一 manifest；
- 禁止按 config eligibility、abstention、effect direction 或 rating distribution 重分层。

### 8.2 Blind split selection

候选 discovery proportions 为 `{0.50, 0.33, 0.25, 0.20}`。对每个比例 `p`，使用 deterministic
stratified largest-remainder allocation：每个 scenario stratum 初始 discovery quota 为
`floor(p*n_h)`；全局目标为 `round_half_up(p*N)`；剩余名额按 fractional remainder 降序分配，
tie 以 stratum ID 字典序打破；stratum 内按 frozen segment hash 排序取前若干。discovery 五折按
`SHA256("RQ014-v1-fold|"+segment_id)` 全局排序后 round-robin 分配。

每个 config 独立标记 `power_eligible`。从大到小选择“至少一个非别名 config 通过完整 pipeline
power gate”的第一个比例；只有在该比例下自身通过 power gate 的 configs 可进入 promotion。
该选择不由 PI 看 leaderboard 决定。

### 8.3 完整 pipeline power gate

记忆中的最低“强相关”目标冻结为 `rho_WS=-0.30`。G2P 对每个非别名 config `j` 和每个候选
split 至少运行 20,000 次 simulation。DGP 固定为：

1. 保留实际 12 维 deviation vectors、config-specific missingness 和 configs 间相关结构；
2. 轮流把 config `j` 作为 data-generating recipe；对其每个 complete scene/candidate 生成
   `U_sc=-beta_j*Q_jsc+epsilon_sc`，`epsilon~N(0,1)`，再对 U 做场景内 rank；
3. 对 0%、20%、40% 三种 tie mechanism 分别校准 `beta_{j,tie}`；每个 beta 用 outcome-free
   Monte Carlo bisection，使经过对应 tie operator 后的期望 `rho_WS=-0.30`，容差 `0.005`；
4. tie operator 分别运行 0%、20%、40% scenarios：按 frozen RNG 选定相应比例 scenes，把 latent
   utility 最接近的一对候选赋为平均 midrank；其余 scenes 保持严格 1/2/3 ranks；
5. 同一模拟 ratings 同时与全部 configs 的固定 deviations 进入真实 discovery promotion；
6. pipeline success 仅当晋级 config 为 `j`（或 G2 rating-blind 证明的 exact structural alias）且
   confirmation association gate 通过；
7. RNG = NumPy `Generator(PCG64)`；`j_index` 是按 config ID 排序后的 zero-based index，
   `split_index` 按 `{0.50,0.33,0.25,0.20}` 顺序为 0–3；seed 为
   `20260710 + 1000*j_index + split_index`；所有 RNG streams 与校准结果写入 manifest。

Simulation 必须包含：

- 每 config 的实际 missingness、deviation ties 和三候选结构；
- 0%、20%、40% 三档 rating-tie sensitivity；
- G2 后的全部非别名 eligible configs、deterministic promotion 和单-recipe confirmation；
- one-sided alpha `0.025` 的 permutation test。

PASS 要求：

- 20% tie scenario 下 `pipeline_power>=0.80`；
- 40% tie scenario 单独报告，不得隐藏；
- discovery expected `N_informative>=40`；
- 五个 discovery folds 每折 expected `N>=8`；
- confirmation numerical-degeneration probability `<0.01`。

已知 50/50 confirmation N 约 23/37/49 时，对 `|rho|=0.30` 的乐观功效仅约
0.28/0.44/0.56，实际 ties 和 selection 只会更低。v1 因此**不预承诺现有缓存足以形成科学确认**。

若没有 split/config 过 power gate：

1. scientific verdict 先锁为 `INCONCLUSIVE_LOW_POWER_OR_COVERAGE`；
2. 不开放“confirmation”措辞；
3. 仍可用全量数据运行冻结的 12-config specification-recovery leaderboard；
4. 可运行同步 within-scene permutations 的 12-config maxT 作为 multiplicity-aware diagnostic，
   但其结果仍标为 recovery/discovery signal，不升级为 independent confirmation；
5. 可继续 forensic branch；
6. 阴性不得加强 RQ010B null。

## 9. Discovery and deterministic promotion

仅 power/coverage eligible 的 valid configs 可参加 scientific promotion。每个 config 报告 full-discovery
`rho_WS`、5 个 frozen hash folds、leave-one-scene-out 最大影响、N、tie/fallback/zero 比例。

晋级门：

- full-discovery `rho_WS<=-0.10`；
- 至少 4/5 folds 为负；
- fold median `<=-0.05`；
- leave-one-scene-out 不得翻为正；
- 所有 numerical/envelope/common-support gates PASS。

多个 config 通过时，只选一个，排序顺序固定为：

1. negative folds 数量降序；
2. fold-median rho 升序；
3. full-discovery rho 升序；
4. expected confirmation N 降序；
5. `config_id` 字典序升序。

PI 不得看榜后换门槛或挑次优 cell。没有 config 通过时按 §12 决策表裁决：只有全部 12 个
non-alias configs 都 power-eligible 才能写 `NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY`；否则为
`INCONCLUSIVE_LOW_POWER_OR_COVERAGE`，并另报“未在 power-eligible subset 找回”。scientific
branch 随后停止。

## 10. One-recipe confirmation

只检验晋级的一个 recipe 和一个 primary endpoint：

- `H0: rho_WS >= 0`；`H1: rho_WS < 0`；
- 9,999 次 within-scene rating-label permutation；
- 每次置换原始评分并重算 midranks；
- one-sided `p=(1+#(rho_b<=rho_obs))/(B+1)`；
- 9,999 次 scene bootstrap，报告 percentile 95% CI；
- 无定义 bootstrap 比例超过 1% 则 numerical failure；
- seed、permutation count 和 output hash 固定。

association confirmation 同时要求：

- `rho_WS<=-0.10`；
- one-sided permutation `p<0.025`；
- bootstrap 95% CI upper `<0`；
- clean-room verifier reimplementation 的 `|delta rho_WS|<=0.03`。

只有 `rho_WS<=-0.30` 才能写“与记忆中的强效应相容”。较小但显著的负效应只能称内部关联。

## 11. IPV-specificity controls

所有 controls 必须在 discovery 前冻结，并在 promoted recipe 的相同 scenes、candidates、common
support 上运行。

### 11.1 Kinematic cost family

沿用 RQ010B 定义与方向：

- `-min_gap_m`；
- `-min_ttc_capped_s`；
- `accel_rms`；
- `jerk_rms`；
- `-progress_m`；
- `curvature_mean_abs`；
- frozen `kinematics_combined_cost`。

若 combined cost 的精确公式/code hash 无法恢复，只允许 association verdict，不允许 IPV-specific。

### 11.2 Negative controls

- deterministic within-scene `shuffled_ipv`：按 frozen hash 对三候选 deviation 做一个非恒等循环位移；
- path/support-matched `counterpart_swap`：在 path type × observed-support quartile 内按 frozen hash
  排序，使用下一 segment 的 counterpart，末项循环到首项；仍禁止外推；
- `role_flip`：交换 ego/counterpart 角色后用对应 human-role envelope 重算；若对应 envelope 无法通过
  blind gate，此 control 记为 unavailable，最高 verdict 降为 association；
- `sign_flip`：仅将 candidate IPV 变为 `-v`，保持原 path-type asymmetric envelope 不变，用于检查
  sign convention 是否能机械制造信号；
- `IPV_removed`：不是伪造一个 deviation，而是从预注册的 within-scene rank model 中删除 deviation，
  形成 kinematics-only nested baseline。

前四个 deviation controls 用同一个 `rho_WS`。定义 expected-direction strength
`S=max(0,-rho_WS)`；`IPV_removed` 只进入 nested-model incremental test。

IPV-specific 必须：

1. association gate 已通过；
2. `S_IPV` 严格高于每个 individual kinematic 和 negative control；
3. 在所有 primary、7 个 kinematic costs 与 4 个 deviation controls 均完整的同一 universal
   confirmation scene set 上，构造 within-scene centered midranks：rating `R~`、deviation `Q~`、
   combined kinematic cost `K~`；每个 candidate 权重 `1/3`，不设截距，拟合
   `R~=beta_D*Q~+beta_K*K~+error`；baseline 为 `R~=beta_K*K~+error`；
4. nested increment 的唯一 signed statistic 为 `T_inc=-beta_D`，并额外要求
   `SSE_full<SSE_baseline`；每次 bootstrap 都必须重新生成 midranks、重拟合 full/baseline；
5. specificity family 固定为 12 个 one-sided statistics：7 个
   `S_IPV-S_kinematic_j`、4 个 `S_IPV-S_deviation_control_j` 和 `T_inc`；
6. 优越性不使用 rating-label permutation。固定 9,999 次 paired scene bootstrap：从 universal
   common scene set 有放回抽场景，每次重算 11 个 advantages 并重拟合 nested model；令观测统计
   `A_m` 为 11 个 `S_IPV-S_control` 与 `T_inc`，`se_m=SD(A_m^b)`，
   `Z_bm=(A_m^b-A_m_obs)/se_m`；
7. 以每次 bootstrap 的 `max_m Z_bm` 的 97.5th percentile 为 `c_0.975`，构造 one-sided
   simultaneous lower bounds `L_m=A_m_obs-c_0.975*se_m`，控制 family-wise alpha `0.025`；
   不允许执行时改回 label permutation 或 Holm；
8. 所有 12 个 `L_m>0`，且 combined-model `beta_D<0`、`SSE_full<SSE_baseline`；
9. leave-one-scenario-cluster-out 方向稳定。

Specificity 是 association 通过后的层级检验，不与 discovery grid 混为一个 maxT family。
`SSE_full<=SSE_baseline` 对嵌套 OLS 基本是代数性质，只作为实现 sanity check，不能单独解释为
specificity evidence；真正的增量证据是 `T_inc` 的 simultaneous lower bound。

由于 `S=max(0,-rho_WS)` 在 0 处非光滑，G2 必须用实际 missingness/ties 进行至少 10,000 次 global-null
simulation，验证上述 12 项 paired-bootstrap bounds 的 empirical FWER `<=0.03`，并报告 positive-control
coverage。任一 statistic bootstrap `se=0`、无定义 replicate 比例 `>0.01`，或 empirical FWER 超过
0.03，specificity 状态为 `SPECIFICITY_NUMERICAL_INVALID`，最高 verdict 降为 association；不得换检验救援。

## 12. Verdict taxonomy

Scientific family 使用以下机器决策表：

| 条件 | 唯一 verdict |
|---|---|
| 无 config power-eligible；或任一非别名 valid config 因 envelope/common-support/numerical gate 不可判 | `INCONCLUSIVE_LOW_POWER_OR_COVERAGE` |
| 全部 12 个非别名 valid configs 均 power-eligible，但没有 config 通过 discovery promotion | `NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY` |
| 仅部分 configs power-eligible，且其中没有 config 晋级 | `INCONCLUSIVE_LOW_POWER_OR_COVERAGE`；另报“未在 power-eligible subset 找回” |
| 一个 recipe 晋级，但 confirmation association gate 失败 | `DISCOVERY_ONLY_OVERFIT` |
| association 通过，但 specificity 不完整或任一项失败 | `VALID_INTERNAL_CONFIRMATION_ASSOCIATION` |
| association 与 12 项 specificity maxT family 全部通过 | `VALID_INTERNAL_CONFIRMATION_IPV_SPECIFIC` |

因此，`NOT_RECOVERED_WITHIN_FROZEN_VALID_FAMILY` 只有在完整 valid family 都有充分功效时才允许使用。

Forensic family 单独裁决：

- `HISTORICAL_SIGNATURE_RECONSTRUCTED`：匹配旧 rho/N/figure/file fingerprint 的冻结容差；
- `ARTIFACT_MECHANISM_SUPPORTED`：同一 common support 的一因素 defect-on/off twin 通过预注册配对检验；
- `FORENSIC_SIGNAL_FOUND_UNATTRIBUTED`：出现负相关但无旧 fingerprint 或无法单因素归因；
- `FORENSIC_NOT_RECOVERED_WITHIN_FROZEN_CATALOG`。

没有旧 fingerprint 时，bug cell 变负不能称旧结果已恢复。

## 13. G2 provenance and parity contract

### 13.1 Input manifest

`input_manifest.csv` 至少包含：`input_id, role, absolute_path, size_bytes, sha256, row_count,
primary_keys, key_unique, sampling_rate_hz, coordinate_frame, time_support, contains_rating,
producer_code_sha, source_run_id`。

必须锁定 rated479 manifest、score-stripped bundles、selected counterpart tracks、candidate geometry
mapping、path-type human source、4/10 Hz anchor tables、ratings manifest 和 split manifests。

### 13.2 两条 estimator lineage

不得再把 `5edd2810` 与 WOD fixed adapter 统称为同一实现：

- `LEGACY_SIGMA01_CORE`：commit `5edd28104bf5989e2dc258c9405ce897d7523cc4`，legacy
  probability-domain lineage，用于历史/InterHub provenance；
- `RQ010B_WOD_FIXED`：实际 core tree + exact WOD adapter bytes + timing patch + stable log-domain
  reliability + route reference + no-extrapolation，用于 accepted anchors 和 valid candidate build。

G2 必须登记 core commit/tree/status、adapter SHA、dt mutation、sigma、grid、solver、reference parameters、
history/min-observation rule。adapter 修改 module-global timing/reliability 时，不同 rate/config 必须在
独立进程运行；任务内部只允许一层并行。

### 13.3 Legacy anchors

| Anchor | 必须复现 |
|---|---|
| A1 | 4 Hz Scheme 1：N/key exact；rho `0.14767623020869206` |
| A2 | 4 Hz Scheme 2：N/key exact；rho `0.03129843185743807` |
| A3 | 10 Hz Scheme 1：N/key exact；rho `0.16481712868606582` |
| A4 | 10 Hz Scheme 2：N/key exact；rho `0.12753396435827072` |
| A5 | full479 raw IPV pooled anchor，仅 forensic；不得冒充 envelope anchor |

容差：input SHA/keys/N/abstain exact；no-extrapolation violation=0；同环境 row IPV max abs diff
`<=1e-8`；锁定环境跨 CPU host `<=1e-6` 且 candidate rank exact；rho delta `<=1e-6`；固定
permutation/bootstrap exceedance count exact，或 p tolerance `<=1/(B+1)`。

### 13.4 Rolling-verifier fixtures

至少验证：band 内与边界为 0、上下越界单调、非对称归一化、同步删除、4/10 Hz 人工连续轨迹
物理时间一致、no extrapolation、history/future 拼接不重复 t*、window-to-sample 映射、fallback/tie
显式 abstain。

## 14. Execution model: advanced lead, low-cost deterministic workers

| Role | 允许 | 禁止 |
|---|---|---|
| Advanced lead | estimand、registry、gate、claim boundary、异常裁决 | 手挑 leaderboard、结果后改 filter/threshold |
| Low-cost forensic worker | 索引、hash、mtime、closure matrix | 解释相关性或扩配置 |
| Low-cost build worker | 按 registry 构建 pair/IPV/envelope/verifier cache | 读取 ratings、改参数 |
| Low-cost HPC worker | 生成/提交/监控 Slurm、幂等 retry、资源 ledger | 改 config、登录节点重计算 |
| Low-cost stats worker | 运行冻结脚本、输出完整 long table | 决定晋级、隐藏失败 cell |
| Independent validator | 仅按书面 spec 重实现 top verifier/statistic | 复用作者 verifier 或看 discovery leaderboard |
| Advanced red team | 查泄漏、scale/dt/reference/counterpart twins、controls | 新增 outcome-driven endpoint |
| Final adjudicator | 按 taxonomy 裁决 | 静默覆盖 RQ010B |

## 15. HPC and reproducibility contract

第一轮 CPU-only；不申请 GPU、不重跑 StreamPETR。HPC durable root：

```text
/share/home/u25310231/ZXC/RQ014_recovery/<RUN_ID>/
```

`RUN_ID = RQ014_1_wod_rating_recovery_<timestamp>_<plan_sha8>`。本地执行报告镜像到：

```text
reports/studies/RQ014_wod_e2e_rating_recovery/<RUN_ID>/
data/derived/wod_e2e/RQ014_recovery/<RUN_ID>/
```

HPC 子目录：`code/configs/env/inputs/manifests/scripts/logs/work_dirs/checkpoints/cache/results/audit`。
所有新 job name 以 `zxc-` 开头，例如 `zxc-rq014-g2-anchor`、`zxc-rq014-bench`、
`zxc-rq014-ipv`、`zxc-rq014-score`。重计算用 `sbatch`；登录节点仅做 identity/path/hash/queue/manifest。

12-scene benchmark 必须覆盖 4/10 Hz、W=1/2.5、future/history+future、短/长支持和边界案例，输出
wall time、peak RSS、cache hit、fallback rate、projected CPU hours/storage 和 array concurrency。
冻结资源预算前不得提交 full run。设置 BLAS/OpenMP threads=1；一个 task 只允许一层并行。

cache key 使用 canonical JSON + SHA-256：trajectory pair → IPV → verifier → statistics 四层；rating
SHA 只能出现在 statistics key。shard 先写 `.partial`，校验后 atomic rename；只有有效 `DONE.json`
可被 aggregation 读取；retry 保留所有 attempt，不能覆盖失败记录。

## 16. Required artifacts

每个 run 至少有：

- `run_manifest.json`；
- `input_manifest.csv`；
- frozen `valid_registry.yaml` 与 `forensic_registry.yaml`；
- materialized `spec_registry.csv`（36 valid readout rows）；
- `environment_manifest.json` + conda/pip/platform/BLAS records；
- `job_manifest.csv`、`cache_manifest.csv`、`artifact_manifest.csv`；
- `rating_access_log.jsonl`；
- `tried.md`、`deviations_from_plan.md`；
- candidate/scene attrition tables；
- claim-level `evidence.csv`；
- discovery full ledger、confirmation one-shot audit、red-team 与 clean-room reports。

## 17. Stop conditions

任一条件触发即停止当前阶段：

- G0 有未分类 `OPEN` surface；
- plan/registry SHA 漂移；
- input keys/hash 不一致；
- core、adapter 或 env 不能唯一识别；
- required anchor/fixture 超容差；
- feature build 前访问 rating；
- counterpart extrapolation violation >0；
- estimator/envelope scale 不兼容；
- common-support、numerical-health 或 power gate 失败；
- projected compute/storage 超预算；
- discovery 无 recipe 通过；
- confirmation 已成功生成一次统计量；
- confirmation 主检验失败；
- 结果后提出改 split/filter/endpoint/recipe；
- 需要 GPU、新 detector 或 360° tracker 但无 amendment；
- SSH identity/work root 不符或出现密码提示。

Power gate 不通过时允许继续 specification/forensic recovery，但不得运行或宣称 confirmation。

## 18. Independent re-review acceptance checklist

- [ ] G0 所有 surface 已为三种合法终态；
- [ ] v1 与两个 registry SHA 已登记；
- [ ] 12 non-alias core / 36 readout rows 可机器 materialize；
- [ ] primary estimand、ties、missingness、common support 唯一；
- [ ] time-indexed envelope 与 estimator scale 兼容；
- [ ] blind split/power rule 可执行；
- [ ] promotion 唯一且无需 PI judgment；
- [ ] confirmation 只有一个 recipe/endpoint；
- [ ] specificity family 和 FWER 完整；
- [ ] legacy core 与 WOD adapter lineage 分离；
- [ ] anchors、fixtures、hashes、tolerances 完整；
- [ ] HPC run/cache/checkpoint/retry 语义完整；
- [ ] recovery、internal confirmation、independent replication 用词严格分开；
- [ ] RQ010B scoped-addendum boundary 明确。

## 19. v1 的预期价值与剩余风险

这个设计首先保证能回答“哪种物理窗口、频率和 sequence 定义最接近旧设置”，同时把结果后挑
配置的自由度压到 12 个非别名、同估计对象 recipes。它也明确接受一个现实：前向相机 counterpart 覆盖将
N 限制在约 75–98 附近时，真正的 split confirmation 很可能达不到 80% power。此时 v1 仍可找回
一个候选设置和 artifact mechanism，但不会用不足的样本制造过强结论。

主要 residual risks：旧 fingerprint 可能只在不可达介质；matched InterHub envelope 重建可能失败；
WOD fixed adapter 与 legacy sigma01 scale 可能不兼容；tracker 误差进入 counterpart state；三候选
scene-level statistic 离散；同一数据无法提供外部独立复现。

因此，v1 的成功定义不是“必须得到负相关”，而是：**以冻结、可审计、不会混淆缺陷与科学证据的
方式，找回最可能的实验设置，并对任何出现或未出现的信号给出正确证据等级。**
