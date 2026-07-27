# RQ015A 执行交接手册 — 2026-07-27

接手方：Claude Code CLI（新线程）｜交出方：Claude（Cowork 会话，PI 角色代理）
状态：`BUILD_WHILE_DENY`｜`execution_authorized = false`｜`formal_g1_eligible = false`

> **你的角色：指挥者，不是执行者。**
> PI 裁定（2026-07-27）：接手方**只做编排与判断**——澄清目标、分解工作流、写 prompt、
> 判定结果可信度、决定方向、做最终综合。**任何边界明确的执行任务一律交给 codex CLI**
> （最高等级模型 + 最高推理档），并行后台跑。详见 §1.5 与
> `.codex-fleet/rq015a-implementation/board/plan.md`。
> HPC 相关工作也归你（你有权限），同样按"指挥 codex 执行"的方式做，见 T11。

## 0. 三分钟读懂你要做什么

RQ015A 是一次**回溯审计**：把语料里每一条 IPV 记录标上两件事——

1. 估计器到底有没有真的跑（`ATTEMPTED` / `NOT_ATTEMPTED` / `UNKNOWN`）；
2. 如果跑了，7 个候选的权重有多集中（连续量 `q_eff = K_eff / K`）。

`q_eff → 1` 表示权重摊平、这一帧实际没提取到信息；`q_eff = 1/K` 表示锁定单一候选。
其余产物（分层分布、可用子集清单、协变因素、对下游 RQ 的路由判定）全部由这张台账推出。

**它不做的事**：不衡量 IPV 估得准不准；不区分近均匀的成因（那是 RQ015B）；
不构成 RQ007 意义上的完整 estimability（那是个合取条件）。
**全文禁止 `estimability` 与"测出 / 未测出 IPV"表述。**

计划已过 **6 轮独立复审**，全部 BLOCKED。复审已收敛到一个结论：
**计划正文不再是瓶颈，缺的是实现。** PI 于 2026-07-26 裁定按路径 A 推进——把实现写完，
再连同计划一起送最后一轮复审。你接手的就是这件事。

## 1. 铁律（违反任何一条即作废重来）

| # | 规则 | 为什么 |
|---|---|---|
| R1 | **先按 `case_id` 白名单过滤，再读任何 measurement 列** | held_out 边界。见 R2。 |
| R2 | **`fold` 不是 `split`。禁止用任何 fold 名近似 RQ007 split** | RQ009 的 4 个 fold 与 RQ007 的 3 个 split 正交，每个 fold 都含约 29% held_out；按 fold 过滤会解析 1,899,898 行 held_out |
| R3 | `held_out_parsed_rows` 必须为 0，且写进 receipt | 同上 |
| R4 | **禁止跨产物 pooling** | feature matrix 由 sigma01 派生，合并会对同一原始 observation 重复加权。已由 `assert_single_artifact()` 代码强制 |
| R5 | **禁止读取任何 `rating` / `preference` / human-score 字段** | RQ014 的致盲体系；也是 run spec 的中止条件 |
| R6 | 报告用 policy bins **不得进入任何判定**（episode 摘要、C0 路由只吃连续量） | 前几版的自相矛盾点，已由 `test_c0_routing_never_consumes_report_bins` 强制 |
| R7 | 所有均值用 `sorted + math.fsum`，禁止朴素 `sum()/n` | 朴素求和在不同输入顺序下给出 `0.3` 与 `0.30000000000000004`，违反"唯一算法" |
| R8 | 空串**绝不**读作 0；`NOT_ATTEMPTED` 优先于 `UNKNOWN` | warm-up 占位写的是 `error = 1.0`；让退化规则优先会把已完全解释的机制吞进"成因不明" |
| R9 | 遇到不一致 **fail closed**，不要静默截断/补零 | v3 复审抓出的三个 fail-open 就是这么来的 |
| R10 | 不改估计器、不部署闸门、不重训 M3、不覆盖任何冻结产物、不改任何 owning RQ 的 `decision.md`、不做 replay | 范围边界 |
| R11 | 新版本**另存新文件**，不覆写既有 vN | PI 明确要求 |

## 1.5 编排合同（PI 裁定 2026-07-27）

### 你自己做什么

只有四件事：**分解工作流**、**写 agent prompt**、**判定结果是否可信**、**最终综合**。
外加编排脚本、board 文件、以及 §8 触发时向 PI 上报。

### 你不做什么

不写 T1–T5 的实现代码，不跑数据管线，不手动改大段文件。
**如果你发现自己正要动手写一段非平凡代码——停下，那是 agent 的活。**

### codex 调用规格

```bash
scripts/codex_agent.sh \
  --fleet-dir   ./.codex-fleet/rq015a-implementation \
  --name        <agent-name> \
  --role        implementer|reviewer|replicator|experimenter|designer \
  --workdir     <repo-root> \
  --prompt-file ./.codex-fleet/rq015a-implementation/board/prompts/<name>.md \
  --model gpt-5.5 -c model_reasoning_effort="xhigh" \
  --sandbox workspace-write --ask-for-approval never \
  [--worktree]        # 所有写代码的 agent 都要，避免并行改同一仓库互撞
```

**独立 agent 一次全部发出**（同一批调用），让它们真正并行；不要一个一个等。
轮询用 `scripts/fleet_status.sh <fleet-dir> --results`，**只读 bounded report，不读原始日志**
（除非要 debug 失败）。

### prompt 的硬要求

codex agent **只看得到 prompt 文件**，看不到本手册、看不到你的会话。所以每个 prompt 必须自包含：

1. 角色与唯一目标（一句话）
2. **把 §1 铁律里与该任务相关的条目原文抄进去**——不要写"见手册 §1"
3. **把 §3 冻结事实里该任务要用的数字原文抄进去**——不要让 agent 自己去推
4. 输入文件的**绝对路径**
5. 禁止事项（尤其：不得翻转 `execution_authorized`、不得真跑审计、不得读 held_out）
6. **有界的结构化结项报告格式**（让你少读字）

模板见 skill 的 `references/agent_prompts.md`；不要凭空自创 prompt。

### 交叉验证（不可省）

单个 agent 的产出是草稿不是事实。合并任何实现前，至少过一轮 **红队**
（`reviewer` 角色，任务是**把它弄坏**：找 fail-open、找 split 过滤顺序被绕过、
找跨产物 pooling、找非确定性）。守恒数字这类要害结果另加一次
**独立复算**（`replicator`，走不同路径、对原实现盲）。
结论写进 `.codex-fleet/rq015a-implementation/board/validation.md`。

### 开工前先自检

```bash
codex --version && codex exec --help >/dev/null && echo "codex OK"
```
codex 未安装或未认证 → 告诉 PI 怎么修，然后停下。没有 codex 就没有这套编排。

## 2. 当前状态快照

### 已完成

- 预执行合同核验（对真实文件），得 **C1–C14** 共 14 项修正 →
  `reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_20260726.md`
- 可复现核验脚本（内建 measurement 列守卫）→ `scripts/rq015a/preflight_structural_scan.py`
- 台账 schema v2（吸收全部 14 项）→ `reports/plans/RQ015A_ledger_schema_v2.json`
- 唯一算法实现 → `scripts/rq015a/rq015a_contracts.py`（v3 三个 fail-open 已修）
- fixtures **20/20 通过** → `tests/test_rq015a_contracts.py`
- WOD 只读取回规格 + HPC 探测脚本 →
  `reports/plans/RQ015A_wod_retrieval_spec_v1.json`、`scripts/rq015a/hpc_probe_wod_targets.sh`

### 未完成（= 你的任务）

| # | 交付物 | 阻断了什么 |
|---|---|---|
| T1 | `scripts/rq015a/build_ledger.py` | 一切 |
| T2 | `scripts/rq015a/validate_only.py` | Formal G1 |
| T3 | `scripts/rq015a/receipt.py` | 运行合同 |
| T4 | `scripts/rq015a/factor_analysis.py` | 主产物之一 |
| T5 | `scripts/rq015a/run_rq015a.py`（唯一入口） | run spec 缺"确切命令" |
| T6 | `configs/research_authorization.json#rq015a_concentration_audit` | 授权对象不存在 |
| T7 | run spec v2（修环境段 + 绑定 schema v2 + 确切命令） | 复审 blocker |
| T8 | 计划 v4（吸收 C1–C14、§9 修订、M4 裁定） | 复审 blocker |
| T9 | 裁定书内正式解除"重导 4/7 与 0.93"条件 | **需 PI 签字** |
| T10 | 重生成 checksum manifest + 红队 + 独立复算 | 送审前提 |
| T11 | **HPC 只读探测与（视结果）取回** | WOD 一支的覆盖 |

### 工作流分解（发给 codex 的分工）

| Wave | agent | role | worktree | 产出 | 依赖 |
|---|---|---|---|---|---|
| 0 | `W0-hpc-probe` | experimenter | 否 | `rq015a_wod_probe.json` + 判读建议 | 无 |
| 0 | `W0-iface-freeze` | designer | 否 | `board/module_interface_v1.md`（T1–T5 的函数签名与数据契约） | 无 |
| 1 | `W1-ledger-builder` | implementer | **是** | `build_ledger.py` + fixtures（T1） | W0-iface |
| 1 | `W2-validate-receipt` | implementer | **是** | `validate_only.py` + `receipt.py` + fixtures（T2/T3） | W0-iface |
| 1 | `W3-factor-analysis` | implementer | **是** | `factor_analysis.py` + fixtures（T4） | W0-iface |
| 2 | `W4-entrypoint-authz` | implementer | **是** | `run_rq015a.py` + 授权对象 + run spec v2（T5/T6/T7） | W1–W3 合并 |
| 2 | `W5-plan-v4` | designer | 否 | 计划 v4 正文（T8） | 无（可与 W4 并行） |
| 3 | `W6-red-team` | reviewer | 否 | 缺陷清单（专找 fail-open / 过滤顺序 / pooling / 非确定性） | W4 |
| 3 | `W7-replicate-conservation` | replicator | 否 | 独立复算三产物守恒数字，盲于原实现 | W1 |

`W0-iface-freeze` 先行的理由：让三个并行 implementer 对着**同一份已冻结的接口**写，
否则合并时接口必然打架。这份接口由你（指挥者）审定后才放行 Wave 1。

**T9 不进 fleet**——需 PI 签字，你只起草待签条款然后停。

## 3. 已冻结的事实（直接用，不要重新推导）

### 3.1 常量

```text
候选网格 legacy7_pi_over_8   = [-3,-2,-1,0,1,2,3] * pi/8    K = 7
候选网格 realtime5_pi_over_8 = [-3,-1,0,1,3]      * pi/8    K = 5   ← 代码库中确实存在
                                                             （src/.../core/agent.py:63-64）

warm-up 占位          ipv_error = 1.0（精确值，估计器从未运行）
均匀回退 K=7          ipv_error = 0.6220355269907728  ( = 1 - 1/sqrt(7) )
均匀回退 K=5          ipv_error = 0.5527864045000421
q_eff = 1 / ((1-e)^2 * K)
K=7 时 q_eff=4/7  对应 ipv_error = 0.5
K=7 时 q_eff=0.93 对应 ipv_error = 0.608069099165
K=7 时 one-hot     q_eff = 1/7 = 0.14285714285714285

MIN_SUPPORT_L1_PER_L2 = 5
bootstrap: B = 2000, seed = 20260726, 按 case_id 聚类
```

**占位（error=1.0）与均匀回退（error≈0.622）是两件不同的事**，不得混为一谈：
前者估计器从未运行，后者运行了但权重摊平。

### 3.2 逐产物（全部实测，来源见 preflight 文档）

| artifact | 格式 | E | C | 主键 | split |
|---|---|---|---|---|---|
| `interhub_sigma01_hw4_timeseries` | csv | 2 | 1 | `(scene_unique_id, frame_index)` | 适用 |
| `rq009_feature_matrix` | parquet ×138 | **2** | **1** | `(case_key, anchor_frame_index, perspective, source_dataset)` | 适用 |
| `onsite_dense_timeseries` | csv/parquet | 4 | 1 | `(case_key, frame_index, timestamp_ms)` | **不适用** |
| `rq009_m3_predictions` | — | — | — | — | **不入台账**（无 ipv_error） |
| WOD ×2 + `rq014_g2r_anchor_scores` | — | — | — | — | **本地缺失** |

```text
sigma01（实测）
  全量数据行            3,695,981   （dev 1,865,625 / guard 732,911 / held_out 1,097,445，0 未映射）
  dev+guard 物理行      2,598,536
  其中 D0(frame<4)        107,544
  measurement_rows      5,197,072   （= 2,598,536 × 2）
  NOT_ATTEMPTED           215,088   （= 107,544 × 2）
  ⚠ 2,490,992 是**已排除 D0** 的数，不可用作 identity_1 基数

rq009_feature_matrix（实测）
  全量                  6,397,266   （138 个 parquet part）
  dev 3,229,842 / guard 1,267,526 / held_out 1,899,898
  dev+guard             4,497,368
  measurement_rows      8,994,736   （= 4,497,368 × 2）
  perspective 取值      key_agent_1 / key_agent_2
  角色列                counterpart_current -> counterpart_ipv_error_current
                        target_future       -> target_ipv_error_future
  ⚠ M4_ONLY_ego_self_anchor_ipv_error_current：**PI 已裁定排除**，E 固定为 2

onsite（实测）
  物理行 70,317  measurement_rows 281,268（= ×4）
  首行 frame_index = 101 且四个 ipv 字段均为空串
```

### 3.3 join / 过滤

```text
case_split_assignment.csv 的 case_id
  ≡ sigma01 的 scene_unique_id
  ≡ feature matrix 的 case_key（该表 scene_unique_id 与 case_key 同值）
形如 ipv_000001；实测 3,695,981 行 0 未映射。unmapped != 0 即 fail closed。

split 文件：data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/
            RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/
            case_split_assignment.csv
            （38,228 cases：development 19,258 / guard 7,628 / held_out 11,342）
白名单：{development, guard}
```

### 3.4 D0 规则逐产物不同

- **sigma01**：全局 `frame_index < 4` **有效**。证据：每 split 恰有 `case 数`个
  `frame_index == 0` 的行（19,258 / 7,628 / 11,342），说明逐 case 从 0 起且连续。
- **OnSite**：全局规则**无效**，必须用局部序号 ——
  冻结 filtering → 每 `case_key` 内按 `(timestamp_ms, frame_index)` 稳定升序 →
  `local_position = row_number - 1`。
  **禁止 `frame_index - min(frame_index)`**（会给出 13 而正确答案是 3）。
  已实现：`rq015a_contracts.local_positions()`。
- **feature matrix**：断言 `n(NOT_ATTEMPTED) == 0`（anchor 由构造起于 min_observation 之后）。

## 4. 可直接调用的既有实现

`scripts/rq015a/rq015a_contracts.py`（**唯一算法**，不要另写一份）：

```python
k_eff_from_error(ipv_error)                  # error>=1 -> None，不除零
q_eff(ipv_error, K)                          # fail-closed：越界/非有限/bool/K<1/k_eff>K 一律抛错
check_conservation(artifact_id, physical_rows, expansion, collapse,
                   status_counts, recoverability_counts) -> ConservationReport
local_positions(rows)                        # OnSite D0 判据
assert_single_artifact(rows)                 # 跨产物 pooling 守卫（代码强制）
aggregate_l2(l1_rows) / aggregate_l3(l2)     # 逐位确定（sorted + fsum）
episode_summaries(ipvs, q_effs)              # 只吃连续量，w = 1 - q_eff
band_shares(q, lo, hi) / bins_stability(q)   # 仅描述性
c0_route(...) / c0_route_with_sensitivity(...)
ContractViolation                            # 所有 fail-closed 的异常类型
```

⚠ 该文件顶部 `SCHEMA_VERSION = "rq015a-concentration-ledger-v1"`，**需随 schema v2 更新**。

## 5. 剩余工作项与验收标准

### T1 `build_ledger.py`

按 schema v2 的 `artifacts[]` 逐产物产出 L1 台账行。硬性要求：

- 顺序必须是 **先 split 过滤 → 后读 measurement 列**（R1）；用代码结构保证，不是靠注释
- 每行带 `artifact_id`（否则 `assert_single_artifact` 会拒）
- `attempt_status` 按 §3.4 的逐产物规则 + R8 的优先级
- 三条守恒恒等式逐产物断言，不成立即 `ContractViolation`
- parquet 读取用 pandas + `pyarrow` 或 `fastparquet`，engine 名写进 receipt

验收：新增 fixture 覆盖 (a) split 过滤发生在 measurement 读取之前，(b) 三个产物各自的
D0 规则，(c) 守恒恒等式失败路径，(d) unmapped case fail closed。

### T2 `validate_only.py`

`must_precede_execute = true`。检查项见 run spec `phases[0].checks`，且必须
`reads_measurement_fields = false`。产出 `validate_receipt.json`。

### T3 `receipt.py`

字段见 run spec `receipt_required_fields`，**另加**：`parquet_engine`、`schema_version`、
`m4_only_channel_excluded: true`、`artifacts_absent_locally: [...]`。
`machine_verdict` 必须是程序算出来的 PASS/FAIL，不是人写的。

### T4 `factor_analysis.py`

**仅描述性**：逐候选因素报 `q_eff` 的 Spearman 秩相关 + case-cluster bootstrap
（B=2000，seed 20260726，按 `case_id` 聚类）95% CI。不下因果结论、不做变量选择。
输出须对输入顺序逐位确定。

### T5 `run_rq015a.py`

```
python3 scripts/rq015a/run_rq015a.py --validate-only
python3 scripts/rq015a/run_rq015a.py --execute      # 需授权对象存在且 execution_authorized=true
```

`--execute` 在授权缺失时必须**拒绝运行并退出非零**。

### T6 授权对象

`configs/research_authorization.json` 增 `rq015a_concentration_audit` 条目，
初始 `execution_authorized: false`。翻转只能由 PI 手动做。

### T7 run spec v2

修正三处：环境段（**"纯标准库"是误述**，目标含 parquet，需 pandas/numpy + 一个 parquet
engine）、绑定 `RQ015A_ledger_schema_v2.json`、写入 T5 的确切命令。
另：`fixtures 16/16` 应改为当前实际数（20）。

### T8 计划 v4

另存 `RQ015A_plan_v4_concentration_audit_2026MMDD.md`。吸收 C1–C14；
§9 按 PI 2026-07-26 裁定修订为"允许对三个 WOD/RQ014 产物做只读取回"；
记录 M4_ONLY 排除裁定；更新可审计范围与覆盖缺口披露要求。

### T9 **需 PI 签字**

PI 曾附加条件"从 dev+guard 重导 4/7 与 0.93"。v3 计划正文写了"解除"，但**正式约束文本在裁定书里**：
`reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md` §6。
只改计划正文在治理上不成立。
解除理由：那两个数已从"科学阈值"降为"报告用 policy bins"，不进入任何判定。
**不要替 PI 签。** 起草解除条款、标记待签，然后停在这里。

### T10 送审前

重生成 checksum manifest（覆盖计划 v4、schema v2、contracts、fixtures、run spec v2、
取回规格、裁定书）；跑全量测试；做一轮对抗式自审，专找 fail-open
（"非法输入被伪装成合法结果"是这个项目已经犯过三次的错）。

## 6. T11 — HPC 探测与取回（**现归你，不再等 PI**）

PI 裁定 2026-07-27：HPC 工作交接手方执行（你有权限）。仍按"指挥 codex"的方式做——
派 `W0-hpc-probe`（`experimenter` 角色，Wave 0，与 `W0-iface-freeze` 并行）。

**第一步是探测，不是取数：**

```bash
bash scripts/rq015a/hpc_probe_wod_targets.sh > rq015a_wod_probe.json
```

只做 `ls / head -1 / wc -l / sha256sum`；不写入 HPC 任何路径、不提交作业、不复制数据。
sandbox 需要外网/ssh 时用 `--sandbox danger-full-access`，且**该 agent 的 prompt 必须
明写它只被授权做只读探测**；取回是下一个独立操作、需要单独授权。

探测结果回来前，按"WOD 三产物缺失"实现（schema v2 已如此标注）——T1–T10 不被 T11 阻断。
结果回来后由**你**判读：

- 若 `phase1_ipv_build/candidate_estimability_audit.csv` 等确含 error 列 →
  schema v1 的 "ipv_error_source = absent" 定性被推翻，recoverability 升级为 `L1_DIRECT`；
- `rq014_g2r_anchor_scores` **建议不取回**：其行 schema 23 个字段中没有任何 error 字段，
  取回不改变 `UNKNOWN` 定性，只扩大致盲风险面；
- 任何取回都必须 HPC 侧先做列投影 + sanitization receipt，本地永不出现评分列（R5）。

## 7. 环境事实（实测）

```text
python 3.10 / pandas 2.3.3 / numpy 2.2.6 / pytest 9.1.1
pyarrow    ：pip 下载 46.9 MB 超时失败
fastparquet：安装成功，可读全部目标 parquet   ← 目前的可用路径
scipy      ：缺失（故 import src/.../core/agent.py 会失败；
             preflight 脚本改用正则读候选网格常量绕开）

测试：python3 -m pytest tests/test_rq015a_contracts.py -q      # 当前 20/20
核验：python3 scripts/rq015a/preflight_structural_scan.py --repo-root .
      （约 60s；只读结构，不读 measurement）
```

## 8. 立即停止条件

出现以下任一情况，停下来找 PI，不要自行决定：

1. 任何 `held_out` 行被解析
2. 需要翻转 `execution_authorized`
3. 需要改动任何 owning RQ 的 `decision.md`
4. 发现的事实与本手册 §3 冻结事实冲突（说明我核验错了，需重新核验并出修订记录）
5. 需要读取任何 rating / preference 字段
6. 要动 RQ014 致盲体系相关的任何产物

## 9. 起步指令

```
1. codex --version && codex exec --help >/dev/null   # 没有 codex 就先解决它，见 §1.5
2. 读 START_HERE.md、本手册、preflight_contract_verification_20260726.md、
   RQ015A_ledger_schema_v2.json、.codex-fleet/rq015a-implementation/board/plan.md
3. 跑 pytest 确认 20/20 基线（这一步自己跑，不用派 agent）
4. 一次性发出 Wave 0 两个 agent：W0-hpc-probe 与 W0-iface-freeze
5. 审定 W0-iface-freeze 的接口冻结文件 —— 这是你的判断，不能外包
6. 接口放行后，一次性发出 Wave 1 三个 implementer（各自 --worktree）
7. 收报告 → 合并 → Wave 2 → Wave 3 红队与复算 → 送最后一轮独立复审
8. 每个 wave 结束按 AGENTS.md 写 main_workflow.log，并同步 START_HERE.md
9. T9 停在待签状态，不要替 PI 签字
```

**贯穿全程的一条**：你的 token 是稀缺资源，花在判断上，不要花在打字上。
读 bounded report，不读 transcript。

## 10. 相关文档

| 文档 | 作用 |
|---|---|
| `reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_20260726.md` | C1–C14 全部证据 |
| `reports/plans/RQ015A_ledger_schema_v2.json` | 台账合同（当前有效） |
| `reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md` | 计划正文（待出 v4） |
| `reports/plans/RQ015A_run_spec_v1.json` | 运行合同（待出 v2） |
| `reports/plans/RQ015A_wod_retrieval_spec_v1.json` | WOD 取回规格 |
| `reports/knowledge/RQ015A_ipv_estimability_labelling/sealed_exposure_disclosure_20260726.md` | 暴露裁定（T9 要改这里） |
| `reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md` | 姊妹 RQ（修估计器），待独立双路复审 |
| `main_workflow.log` | 时序记录 |

## 11. 一句话总结

计划已经吵够了；**现在缺的是把 T1–T5 写出来**。写的时候记住两件事：
先过滤再读数（R1/R2），以及不确定就 fail closed（R9）——
这个项目迄今为止所有被抓出来的错，都是这两条里的一条没守住。
