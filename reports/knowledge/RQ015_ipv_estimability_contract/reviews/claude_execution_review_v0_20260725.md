# RQ015 Plan v0 — 独立执行/治理复审（Claude）

角色：独立执行、治理、可复现性复审员（**非**计划作者；统计推断由另一名复审员独立负责，
本文不涉及统计推断，且未与其通信）。
复审对象：`reports/plans/RQ015_plan_v0_ipv_estimability_contract_and_estimator_repair_20260725.md`
（SHA-256 `e200cbaa…`，复审时校验通过，计划未被修订）。
日期：2026-07-26 ｜ 性质：只读；未提交任何 Slurm 作业、未连接 HPC、未接触任何 rating/preference_score 字段。
唯一写入文件：本文。

---

## 1. VERDICT

`BLOCKED` — 5 项 BLOCKING（含一项：计划 §2.5 的核心异常诊断被数据直接证伪，且该错误已扩散进 `START_HERE.md` 与 RQ015 README）。

---

## 2. 验证执行记录

### 2.1 校验和与产物完整性

```
$ sha256sum -c reports/plans/RQ015_plan_v0_checksums_20260725.sha256
reports/plans/RQ015_plan_v0_ipv_estimability_contract_and_estimator_repair_20260725.md: OK
reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/claude_center_collapse_diagnostic_20260725.md: OK
reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/m3_center_dispersion_probe.py: OK
```

3/3 通过。清单覆盖 3 个文件。

### 2.2 治理登记状态

| 检查 | 命令/位置 | 结果 |
|---|---|---|
| 三层结构 | `ls reports/` | `knowledge` / `plans` / `studies`（见 F-N1） |
| RQ015 studies 层 | `reports/studies/RQ015_ipv_estimability_contract/README.md` | 存在，仅 README |
| RQ015 knowledge 层 | `reports/knowledge/RQ015_ipv_estimability_contract/README.md` | 存在，仅 README |
| **`decision.md` 未被创建** | `find reports/knowledge/RQ015_*` | **确认不存在** ✅ |
| STUDIES.md 登记 | `STUDIES.md:49` | `planning / 待独立复审；无计算授权` ✅ 与计划 §状态一致 |
| START_HERE 登记 | `START_HERE.md:11-30` | `PROPOSED，待独立复审，**无计算授权**` ✅ |
| 计算授权 | `configs/research_authorization.json` | 仅含 `INFRA` 与 `RQ014`；**无 RQ015 条目** ✅ |
| 未同时改论文仓库 | `git status --porcelain` | 改动仅在本仓库；无 paper 仓库路径 ✅ |
| 工作流日志 | `main_workflow.log` | 4 条 RQ015 条目（drafted / unit-correction / sigma-history / independent-review） ✅ |

### 2.3 触发证据数字的独立重算（只读全量）

数据：`data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv`
（2,208,674,429 B，3,695,981 数据行）。

**先做了一次负向对照**：用 `awk -F,` 朴素逗号切分复算，得到 zero_rate=9.38%、`err≥0.61`=17.50%——
全部错误。原因：该 CSV 含带逗号的引号字段（`"['HV', 'HV']"`），
`head -200000 | awk '{print NF}'` 显示每行字段数在 **39 / 40 / 41 / 42** 之间浮动。
**任何基于 `awk -F,` / `cut -d,` 的一次性命令在这份产物上都会静默错位。** 见 F-B4。

改用严格 CSV 解析（pandas `read_csv`，分块，`usecols`）复算：

| 计划陈述 | 位置 | 我的复算 | 判定 |
|---|---|---:|---|
| 7,391,962 agent-值 | §2 | 7,391,962 | ✅ |
| `\|IPV\|<1e-9` = 43.71% | §2.1 | 43.7088%（3,230,936） | ✅ |
| 零值行中 92% `err≥0.61` | §2.1 | 92.10% | ✅ |
| 全样本 `err≥0.61` = 54.5% | §2.2 | 54.54% | ✅ |
| 全样本 `err≤0.5` = 23.2% | §2.2 | 23.17% | ✅ |
| `P(IPV=0\|err≥0.61)` = 73.8% | §2.2 | 73.80% | ✅ |
| waymo 48.86 / av2 42.57 / nuplan 37.47 / lyft 37.20 | §2.3 | 48.86 / 42.57 / 37.47 / 37.20 | ✅ |
| 各源 `err≥0.61` / `err≤0.5` 四行 | §2.3 | 55.40·27.42 / 57.21·9.61 / 57.86·10.35 / 45.29·34.50 | ✅ |
| one-hot 自信零 = 1,703 行 | §2.1 | 1,703（`\|err\|<1e-9`） | ✅ |
| **32.4% 精确等于 0.622036** | §2.1 | **不复现**，见 F-B2 | ❌ |
| **9.47% 零值行 `err>0.62204` ⇒ K≥9 网格混入** | §2.5 | 数量对（9.47%），**成因诊断被证伪**，见 F-B1 | ❌ |

### 2.4 §2.5 异常行的定向取证

```
rows with ipv_error > 1-1/sqrt(7):        305,824
  of which |ipv| >= 1e-9 (non-zero):            0
  distinct rounded ipv_error values:   {1.000000: 305,824}     ← 全部恰为 1.0
  by dataset: waymo 185,744 / nuplan 59,992 / lyft 40,840 / av2 19,248
frame_index distribution (agent-1 侧，152,912 行):
  frame_index=0  152,912/4 = 38,228  占该 frame_index 全部行的 100.00%
  frame_index=1  38,228   100.00%
  frame_index=2  38,228   100.00%
  frame_index=3  38,228   100.00%
  max frame_index with sentinel = 3
```

38,228 case × 4 帧 × 2 agent = **305,824**，与异常行数完全相等。
`ipv_error` 直方图另证：`[0.6220356, 0.70)` 区间为**空**，没有任何介于 K=7 上限与 1.0 之间的值。

源码定位（`src/sociality_estimation/core/ipv_estimation.py:251-252`）：

```python
ipv_values = np.zeros((steps, 2), dtype=float)
ipv_errors = np.ones((steps, 2), dtype=float)
```

估计循环仅对 `t >= min_observation`（= 4）写回（同文件 `:335-338`）。
因此 `(ipv=0.0, ipv_error=1.0)` 是**未估计的热身帧初始化哨兵**，不是任何候选网格的产物。
`ipv_error = 1 − √Σw² = 1` ⟺ `Σw² = 0` ⟺ 全零权重，
而 `cal_traj_reliability`（`core/agent.py:1136-1141`）**永远不可能**返回全零权重
（它要么返回归一化 `var`，要么返回 `ones(K)/K`）。K≥9 假设在数学上即被排除。

### 2.5 log 域实现的实际位置与形态

```
$ grep -rn "cal_traj_reliability" --include="*.py" . | grep -v .venv
scripts/rq014/wod_ipv_adapter.py:24,43                       ← stable_traj_reliability（log 域）
archived/ipv_rt_final/scripts/run_full_reference_validation.py:33,63,79  ← cal_traj_reliability_log_distance（log 域）
reports/studies/RQ012_.../vendored_estimator/agent.py:813     ← 概率空间副本（RQ012B 冻结用）
src/sociality_estimation/core/agent.py:247,1078               ← 概率空间核心
```

`archived/report_process/RQ010B_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py`
（35,672 B，SHA-256 `92826fab…`）中 `grep -n "cal_traj_reliability|log|sigma"` **零命中**。

### 2.6 HPC 与授权面

```
configs/run_specs/research_run_spec_v2.schema.json:  "rq_id": {"const": "RQ014"}
                                                     "operation": {"enum": [rq014_* × 5]}
scripts/hpc/submit_research_run.sh:179:              "$PYTHON" -I -S -B -X utf8 "$LAUNCHER" --rq014-only "$@"
docs/reproducible_ipv_pipeline.md:6-10:              /share/home/u25310231/ZXC/ipv_estimation 是 retired stub，仅含 TOMBSTONE.md
START_HERE.md:786:                                   同上（"execution surface is retired"）
```

### 2.7 既有平价基础设施（计划未引用）

```
tests/test_ipv_estimator_parity.py:25-29
  STRICT_LOCAL_ATOL       = 1e-10
  SIGMA01_LOCAL_ENV_ATOL  = 0.1
  SIGMA01_HPC_STRICT_ATOL = 1e-12
  STRICT_SIGMA01_ENV_VAR  = "RQ_IPV_PARITY_STRICT"
tests/test_ipv_estimator_parity.py:3-5（docstring）
  "IPV bit-exact reproducibility of sigma01 requires the generation environment
   (HPC / pinned scipy); locally exact differs ~0.06 due to SLSQP/platform."
tests/fixtures/{ipv_estimator_parity_fixture.json, ipv_exact_local_golden.json}
configs/ipv_sigma01_exact.json（solver_mode=exact, sigma=0.1, min_observation=4, 60/40/40, nuplan 20→10Hz）
```

### 2.8 其他

```
$ git status --porcelain
 M START_HERE.md
 M STUDIES.md
 M main_workflow.log
 M reports/knowledge/RQ014_wod_e2e_rating_recovery/README.md
?? reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/
?? reports/knowledge/RQ015_ipv_estimability_contract/
?? reports/plans/RQ015_plan_v0_checksums_20260725.sha256
?? reports/plans/RQ015_plan_v0_ipv_estimability_contract_and_estimator_repair_20260725.md
$ git rev-parse --abbrev-ref HEAD → main @ 6bdcc2e6
$ grep -rn "estimability_ledger|K_eff|NOT_ESTIMABLE" --include=*.py --include=*.json --include=*.yaml .
  （计划文本之外零命中）
```

---

## 3. BLOCKING findings

### F-B1 §2.5 的异常诊断被数据证伪；Phase A 步骤 5 与 §5 冻结定义均建立在该错误之上（严重度：CRITICAL）

**证据**
- 计划 `reports/plans/RQ015_plan_v0_…md:159-163`：
  「9.47% 的零值行 `error > 0.62204`，超过 K=7 的理论上限，说明时间序列中混入了
  候选网格数不同（K≥9）的行。」
- 实测（§2.4）：这 305,824 行的 `ipv_error` **无一例外恰为 1.0**，`ipv` 恰为 0.0，
  `frame_index ∈ {0,1,2,3}`，数量恰为 38,228 case × 4 帧 × 2 agent。
- 源码 `src/sociality_estimation/core/ipv_estimation.py:251-252` 与 `:335-338`：
  这是 `min_observation=4` 之前的**未估计热身帧初始化值**。
- `ipv_error=1` 数学上要求全零权重，`core/agent.py:1136-1141` 的两个分支都不可能产生。

**为什么阻断（三重后果）**

1. **Phase A 步骤 5**（`:189`「查清 §2.5 的 K>7 异常来源，必要时按 K 分组重算阈值」）
   被定向到一个不存在的问题；执行者会去追查候选网格版本混用，而真实成因已经确定。
2. **§5 冻结定义在这批行上除以零**：`K_eff = 1/(1−ipv_error)²`，`ipv_error=1.0` ⇒ `K_eff = +inf`。
   计划未给出任何处理规则。执行者必须自行发明（当作 NOT_ESTIMABLE？丢弃？报错？），
   而这正是 §5 声称已冻结、不容执行期裁量的部分。
3. **§2.4「两条路径产生同一个 0」的前提不完整**：产物中已经存在**第三条**路径——
   `(ipv=0, ipv_error=1)` 的未估计哨兵。它既不是 D1 数值下溢也不是 D2 固有不可辨识，
   却被 §2.1/§2.2 的全部统计当作真实零值与 `err≥0.61` 计入分母，
   使 D1/D2 的规模被系统性高估。剔除后（仅计 `frame_index≥4` 的有效行）：
   `n = 7,086,138`，零值率 **41.28%**（非 43.71%），`err≥0.61` **52.58%**（非 54.54%）。
   头条结论方向不变，但 §3-H2 要求的 D1/D2 定量拆分基线是错的。

**污染范围（已扩散出计划文件）**
- `START_HERE.md:24-25`：「待查异常：9.47% 的零值行 `error>0.62204`，超过 K=7 上限
  ⇒ 时间序列混入 K≥9 的候选网格。」
- `reports/knowledge/RQ015_ipv_estimability_contract/README.md:24`：同一断言。
两处均为**当前操作简报与知识层**，按 AGENTS.md「若事实不确定须显式写出不确定性」，
不得保留一条已被证伪的机制断言。

**最小修复要求**
1. 修订 §2.5：改述为「已定位：305,824 行 = 38,228 case × 4 帧 × 2 agent 的
   `min_observation` 热身哨兵 `(ipv=0.0, ipv_error=1.0)`，源自
   `core/ipv_estimation.py:251-252`；非候选网格混用」，并给出上述取证命令的落盘脚本（见 F-B4）。
2. §2.1–§2.4 全部比例改为**双口径**（含哨兵 / 仅 `frame_index≥min_observation` 有效行），
   并以后者为 D1/D2 拆分的基线。
3. §5 增加显式前置过滤规则：`ipv_error ≥ 1 − 1e-12` ⇒ `status=NOT_ATTEMPTED`，
   在计算 `K_eff` 之前剔除，永不进入 ESTIMABLE/WEAK/NOT_ESTIMABLE 三态或任何分母。
4. 同一 PR 内更正 `START_HERE.md` 与 RQ015 knowledge README 的对应断言。

---

### F-B2 §2.1「32.4% 精确等于 1−1/√7」不可复现，且该量是 D1 规模的唯一直接证据（严重度：HIGH）

**证据**
- 计划 `:36-37`：「零值行中 92% 的 `error ≥ 0.61`；**其中 32.4% 精确等于
  `1 − 1/√7 = 0.622036`**，即 `sum(var)==0` 的均匀兜底分支。」
- 我的复算（`T = 1 − 1/√7 = 0.6220355269907728`，分母 = 3,230,936 零值行）：

| 判据 | 计数 | 占零值行 |
|---|---:|---:|
| `\|e−T\| < 1e-12` | 273,884 | 8.48% |
| `\|e−T\| < 1e-9` | 319,246 | 9.88% |
| 直方图桶 `[T, T+1e-7)` | 410,718 | 12.71% |
| `\|e−T\| < 1e-6` | 714,244 | 22.11% |
| `\|e−T\| < 1e-4` | 1,846,564 | 57.15% |
| **`e ≥ T − 1e-6`（单边）** | 1,020,068 | **31.57%** |

没有任何「精确等于」的判据得到 32.4%；唯一接近的是**单边** `e ≥ T − 1e-6` = 31.57%，
那不是「精确等于」。且直方图显示 `ipv_error` 在 T 附近**不是尖峰而是连续堆积**
（`[0.6150, T)` 一档就占 68.42%），意味着「兜底分支占比」对容差高度敏感，
在 12.7%–57% 之间浮动，量级不确定超过 4 倍。

**为什么阻断**
`sum(var)==0` 兜底分支的规模是 §2.4-D1 与 §2.4e「D1 真实且量级很大」的唯一直接计数证据，
也是 §3-H2 预期（「改 log 域后 D1 类零值大幅减少」）的定量锚。
一个 4 倍不确定、且与写入 `START_HERE.md:18-19` 的公开数字不一致的量，不能作为冻结前提。

**最小修复要求**
1. 声明精确判据（建议：`ipv_error` 的 IEEE-754 位模式恰等于
   `np.float64(1 - 1/np.sqrt(7))`，或落盘时保留的十进制位数下的严格相等），
   并在计划中记录该判据与结果计数。
2. 因为兜底分支**无法从 `ipv_error` 可靠反推**（连续堆积、无尖峰），
   把「D1 兜底分支占比」从 §2 的既成事实降级为 Phase A 的待测量项，
   并要求 Phase A 通过**重跑带分支埋点的估计器**（记录 `sum(var)==0` 布尔标志）直接计数，
   而不是靠 `ipv_error` 的浮点近似匹配。
3. 同步更正 `START_HERE.md:18-19` 的 32.4%。

---

### F-B3 `estimability_ledger` 无 schema；`0.93·K` 在既有产物上不可计算（严重度：HIGH）

**证据**
- 计划 `:182-183`：Phase A 步骤 2「对全部现存 IPV 产物…逐行打标，产出 `estimability_ledger`」。
- 全仓 `grep -rn "estimability_ledger"`：**计划文本之外零命中**。无字段定义、无键、无落盘位置、
  无与源产物的连接键、无版本字段、无「一行 = 一个 (产物, case, frame, agent)」的粒度声明。
- 计划 `:221`：`K = 该行实际候选数（须显式记录，见 §2.5）`。
  但目标产物的表头（`head -1 …sigma01_hw4_ipv_timeseries.csv`，39 列）
  **不含任何候选数/网格列**：
  `…,ipv_key_agent_1,ipv_key_agent_1_error,key_agent_1_px,…`。
  RQ009 特征矩阵、RQ012B OnSite 侧同理未见该字段。

**为什么阻断**
Phase A 明确定位为「**回溯**打标，无需重算」。但三态阈值中的 `NOT_ESTIMABLE : K_eff ≥ 0.93·K`
需要逐行的 K，而 K 在既有产物中根本不存在也不可反推
（`ipv_error` 是 K 与权重分布的合成量，不可分离）。执行者只能：
(a) 对全部行硬编码 K=7 —— 但这正是 §2.5 想排除的假设；或
(b) 重算 —— 与「无需重算、成本最低」的 Phase A 定位矛盾且需要计算授权。
两条路都要求执行者自行发明规则。

**最小修复要求**
1. 在计划内（或作为随计划一同 checksum 的独立 JSON schema 文件）给出
   `estimability_ledger` 的完整 schema：主键（`product_id, run_id, scene_unique_id, frame_index, agent_slot`）、
   `ipv`、`ipv_error`、`K_source`（`recorded` / `assumed`）、`K`、`K_eff`、
   `status ∈ {ESTIMABLE, WEAK, NOT_ESTIMABLE, NOT_ATTEMPTED}`、`ledger_version`、
   源产物 SHA-256、生成脚本 SHA-256。
2. 明确写出：对不记录 K 的既有产物，`K_source="assumed"`, `K=7`，
   并**要求 Phase A 交付一份「K=7 假设的验证证据」**
   （例如从 RQ009/sigma01 生成配置 `configs/ipv_sigma01_exact.json` 与
   `candidate_ipv_values` 的取值链路证明全部行同网格），否则该假设不得进入 §5 冻结。
3. 说明 `ledger` 的落盘根（`data/derived/…` 或 `reports/studies/RQ015_…/<RUN_ID>/`）
   与「不得覆写任何既有产物」的写入边界。

---

### F-B4 触发证据不可复现：核心数字无落盘脚本，且该 CSV 对朴素逗号切分不安全（严重度：HIGH）

**证据**
- 校验和清单（`RQ015_plan_v0_checksums_20260725.sha256`）只覆盖 3 个文件，
  其中唯一的脚本 `m3_center_dispersion_probe.py` 读取的是
  `04_calibration/predictions/tier=M3/fold=test/predictions.parquet`（M3 目标侧），
  **与 §2.1–§2.4c 的数据源完全不同**。
- §2.1/§2.2/§2.3（sigma01 全量时间序列扫描）、§2.4b（38,228 case 级 8 行表）、
  §2.4c（7 档距离分箱表）、§2.4d（sigma 历史两口径表）**均无对应脚本**。
  全仓 `grep -rn "0.622036|K_eff"` 只命中 `START_HERE.md` 与原始 JSON 数据。
- 该 CSV 存在带逗号的引号字段（第 9 列 `"['HV', 'HV']"`），每行字段数 39–42 浮动。
  我用 `awk -F,` 复算得到 zero_rate=9.38%（真值 43.71%）、`err≥0.61`=17.50%（真值 54.54%）
  —— **朴素切分会静默产生量级错误**。

**为什么阻断**
1. §5 声称阈值「在本文件冻结，SHA 登记于计划校验和」，但阈值所依据的画像数字
   不能由任何被 checksum 的产物重建。冻结的是结论，不是证据链。
2. F-B1 与 F-B2 已经证明：至少有两个 §2 数字是错的/不可复现的。
   这不是理论隐忧，而是已经发生的事故，且事故的技术形态（CSV 引号字段）
   与「一次性 awk」高度吻合。
3. AGENTS.md `:39`「Keep report packages reproducible」与
   `:37`「Before accepting any result set, check numerical health, coverage, and data integrity」。

**最小修复要求**
1. 新增一个只读、rating-free 的落盘脚本（建议
   `reports/knowledge/RQ015_ipv_estimability_contract/reviews/rq015_estimability_scan.py`），
   一次性产出 §2.1–§2.4c 的全部表格，**必须使用严格 CSV 解析器**（`pandas.read_csv` 或 `csv` 模块），
   并在脚本头部显式禁止 `awk -F,` / `cut -d,`。
2. 脚本输出须包含输入文件 SHA-256、行数、库版本，并把 §2 的每个数字与脚本输出逐一对应。
3. 校验和清单扩充为覆盖：计划、该扫描脚本、其输出摘要 JSON、
   `configs/ipv_sigma01_exact.json`、`src/sociality_estimation/core/agent.py`、
   `src/sociality_estimation/core/ipv_estimation.py`（Phase B 的修改对象与基线）、
   以及（若采纳 F-B3）`estimability_ledger` schema。

---

### F-B5 §8 的 HPC 计划与现行执行面不兼容；无资源估算（严重度：HIGH）

**证据**
- 计划 `:248-252`（全部内容）：「Phase B 涉及估计器重跑，需 HPC（`zxc-rq015-*`）；
  Phase C 复用各 RQ 的既有脚本，只加样本限定。执行沿用项目既有治理：
  独立双人复审 → Formal G1 → 冻结 spec → validate-only → 提交。」
- `configs/research_authorization.json`：仅 `INFRA`（2 个操作）与 `RQ014`（5 个操作）。**无 RQ015**。
- `configs/run_specs/research_run_spec_v2.schema.json`：`"rq_id": {"const": "RQ014"}`；
  `operation` 为 5 个 `rq014_*` 的闭合 enum。
- `scripts/hpc/submit_research_run.sh:179`：`"$LAUNCHER" --rq014-only "$@"`。
- `docs/reproducible_ipv_pipeline.md:6-10` / `START_HERE.md:786`：
  生成 sigma01 的 pinned legacy checkout `/share/home/u25310231/ZXC/ipv_estimation`
  **已退役为 tombstone**，仅剩 `TOMBSTONE.md` 与兼容软链；`5edd2810` 的字节存于
  `archives/legacy-code/`。
- `reports/knowledge/_analysis/INFRA_hpc_tongji_reuse.md:21,33`：
  作业名须以 `zxc-` 开头（计划已遵守 ✅）；重计算须走 `sbatch` 而非登录节点（计划未提及）。
- `docs/reproducible_ipv_pipeline.md:20-23`：
  「Production runs must go through `scripts/hpc/submit_research_run.sh`」。

**为什么阻断**
「沿用既有治理」在字面上不可执行：现行 managed 提交面在 schema 与 launcher 两层都硬编码为
RQ014-only，RQ015 的任何 spec 都会在 `rq_id` const 上被拒。计划没有说明它打算
(a) 扩展 schema 到 v3 并新增 `rq015_*` 操作 enum（需自身的双审 + Formal G1 + 契约级联），
还是 (b) 走 managed 面之外的路径（则与 `docs/reproducible_ipv_pipeline.md:20-23` 冲突）。
同时，Phase B 的「修复前后对比」需要一个 sigma01 兼容基线，而该基线的原始执行面已退役，
计划未指出替代基线（`code/repo` + `envs/ipv-exact-sigma01` + `configs/ipv_sigma01_exact.json`）。
另外 §8 完全没有资源估算：无 CPU 数、无节点数、无 wall-time、无输出体量、
无 `resource_profile_id`——而 RQ014 历史已多次因资源/预算问题触发 PI 决策
（`main_workflow.log` D2/D3 条目）。

**最小修复要求**
1. §8 明确二选一并写清后果：扩展 `research_run_spec` 到 v3（列出新增 `rq015_*` 操作名、
   `resource_profile_id`、environment manifest 绑定、以及这本身需要独立的契约级联与 Formal G1），
   或声明 RQ015 不使用 managed 面并给出替代的合规提交路径。
2. 明确 Phase B 的 sigma01 兼容基线为
   `/share/home/u25310231/ZXC/sociality_estimation/code/repo` @ 某个已发布 `origin/main` commit +
   `envs/ipv-exact-sigma01` + `--execution-profile configs/ipv_sigma01_exact.json`，
   并显式声明**不使用**已退役的 `ipv_estimation` checkout。
3. 给出 Phase A / B3（4 个 σ）/ B4 各自的 CPU·hr、wall-time 上限、输出字节量估算，
   与 `sbatch`（非登录节点）、`zxc-rq015-*` 作业名的对应。

---

## 4. MAJOR findings

### F-M1 §6 的「平价测试」不可判定；既有平价基础设施被忽略（严重度：MAJOR）

**证据**
- 计划 `:234-235`：「B 通过：log 域实现通过平价测试（**在无下溢样本上与旧实现数值一致**）」。
- 未定义：容差、「旧实现」指哪一份、「无下溢样本」如何机械判定、在哪个平台/ABI 上比。
- 仓库已有 `tests/test_ipv_estimator_parity.py`，其中已经解决了这些问题：
  `STRICT_LOCAL_ATOL=1e-10`、`SIGMA01_LOCAL_ENV_ATOL=0.1`、`SIGMA01_HPC_STRICT_ATOL=1e-12`、
  `STRICT_SIGMA01_ENV_VAR="RQ_IPV_PARITY_STRICT"`，配套 fixture 两份。计划零引用。
- 该文件 docstring `:3-5` 与 `START_HERE.md:765-770` 均记载：
  **本地 exact 与 sigma01 相差约 0.06（SLSQP/平台差异）**，HPC 生成环境下才是 `4.44e-16`。
  因此「与旧实现数值一致」在本地任何合理容差下都会失败，原因与 log 域无关。
- 「无下溢样本」不可从产物机械判定：`ipv_error` 无法区分 `sum(var)==0` 与近均匀（见 F-B2 直方图）。

**最小修复要求**
§6-B 改写为可判定条款：
(i) 基线 = `configs/ipv_sigma01_exact.json` + 现有 `tests/fixtures/ipv_estimator_parity_fixture.json`；
(ii) 复用现有三档容差常量，并要求在 `RQ_IPV_PARITY_STRICT=1` 的 HPC exact 环境跑严格档；
(iii)「无下溢样本」定义为**由带埋点的旧实现在运行时报告 `sum(var) > 0` 的样本**
（而非事后按 `ipv_error` 筛），埋点结果落盘为 parity 判据的一部分。

### F-M2 §Phase B-B1 对 log 域既有实现的归属错误；且两份既有实现都保留了均匀兜底（严重度：MAJOR）

**证据**
- 计划 `:196-198`：「该修复在 **RQ010B 的 WOD adapter** 中已存在，但未回流核心估计器」。
- 实测：log 域实现在 `scripts/rq014/wod_ipv_adapter.py`（**RQ014** lane）与
  `archived/ipv_rt_final/scripts/run_full_reference_validation.py:63-55`。
  `archived/report_process/RQ010B_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py`
  中 `cal_traj_reliability` / `log` / `sigma` 全部零命中（35,672 B，SHA-256 `92826fab…`）。
- 附带发现（provenance 不符）：`scripts/rq014/wod_ipv_adapter.py:4-6` 的 docstring 声称
  字节来源为 `analyze_wod_e2e_ipv_rating_pilot.py`「42,665 bytes, SHA-256 `7c60676e…`」，
  但仓库内该文件名唯一实例是 35,672 B / `92826fab…`。全仓无 42,665 B 的同名文件。
- **两份 log 域实现都保留了均匀兜底**，与 B2 的叙事相反：
  `wod_ipv_adapter.py:64`（σ 非法）、`:84`（全部非有限）、`:91`（total≤0）三处
  `return np.ones(candidates_num)/candidates_num`；
  `run_full_reference_validation.py:53` 同样 `return np.ones(...)/len(...)`。

**为什么重要**
执行者按 §Phase B-B1 会去 RQ010B 归档里找一个不存在的实现；
更实质的是，若把 `wod_ipv_adapter.py` 当作「已修复」的对照基线，
它的三处兜底会把 `ipv=0` 的病理**重新注入**对照组，使 B4 的前后对比失效。
B2「删除均匀兜底」的工作量也被低估：需要同时改核心 + adapter + 归档路径的语义。

**最小修复要求**
更正 §Phase B-B1 的路径归属；显式列出三处兜底的行号并纳入 B2 的删除清单；
若 adapter 用作对照基线，须先声明其兜底行为已被禁用或已被计数。

### F-M3 Phase B 修改核心估计器与 RQ014 在途 lane 的时序冲突未被处理（严重度：MAJOR）

**证据**
- 计划 `:196-198` 要求把 log 域「回流并统一」到 `src/sociality_estimation/core/agent.py`；
  `:245-246` 断言「RQ014 冻结 lane 不受影响」，但未说明机制。
- `docs/reproducible_ipv_pipeline.md:11-13`：HPC 部署是 commit-addressed，
  「Each run uses a clean detached worktree at an exact commit published on `origin/main`」。
- `main_workflow.log`（2026-07-24 条目）：RQ014 R3 重跑状态为
  `COMPLETE_LOCAL_AWAITING_MERGE_SPEC_VALIDATE_RERUN`——**明确等待一个新的 main commit**。
- `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md:13`：
  历史教训——核心估计器一次「纯提速/重构」（`a0fee535` 的 `cal_individual_cost` /
  `cal_group_cost` 向量化）就把 sigma01 数值改动了 `0.281`，且当时无人预期。
- `reports/knowledge/_analysis/PROGRAM_REVIEW_20260707_claude.md:98` 已把
  「pinned 数值基线 + 兼容性开关」列为常设约定：
  「任何未来估计器加速都应先过 sigma01 parity 测试再合入」。

**为什么重要**
若 Phase B 先落到 `main`，RQ014 等待的那个「fresh commit」就会携带一个改变的估计器，
其 science bytes 与已冻结的 execution/output contract 不再对应。
计划的「不受影响」断言在 commit-addressed 部署下对**历史**运行成立，对**在途**重跑不成立。
同时，计划没有采纳仓库自己的教训：它描述的是**就地替换**（「回流并统一」），
而非新旧并存 + 版本标识。

**最小修复要求**
1. §Phase B-B1/B2 改为**新旧并存**：新增 `likelihood_mode ∈ {"prob_legacy","log_stable"}`
   （默认 `prob_legacy`）与 `abstain_policy ∈ {"uniform_legacy","explicit"}`（默认 `uniform_legacy`），
   使既有全部调用方在不改参数时字节不变；`configs/ipv_sigma01_exact.json` 显式钉住 legacy 组合。
2. 产物侧增加 `estimator_variant` / `likelihood_mode` 列或 run-manifest 字段，
   使任何新旧混合的 ledger 可被区分。
3. §7 增加显式时序条款：Phase B 合入 `main` 必须在 RQ014 R3 重跑的 spec 冻结之后，
   或经 RQ014 lane 的 PI 决策显式豁免。

### F-M4 §Phase B 与 §7 未列出「不得写入」的冻结产物清单（严重度：MAJOR）

**证据**
- 计划 `:199-205` 描述 B2/B4 会产生新的估计输出，`:203-204` 说「在冻结的子样本上比较修复前后」，
  但**全文没有一处列出受保护路径**。
- 应受保护而未被点名的至少包括：
  - `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/`
    （sigma01 hw4 时间序列 + 03_features + 04_calibration/predictions）
  - `checkpoints/rq009_m3/`（checksum-bound 私有 scorer，SHA-256 `b04999ab…`）
  - `tests/fixtures/{ipv_estimator_parity_fixture.json, ipv_exact_local_golden.json,
    rq014_g2r_v1/, rq014_g3r_v1/, m3_verifier_portable_fixture.json}`
  - `configs/ipv_sigma01_exact.json`、`configs/run_specs/*`、`reports/plans/RQ014_*contract*.json`
  - `reports/studies/RQ012_.../vendored_estimator/agent.py`（RQ012B 冻结用的估计器副本）
  - HPC 侧 `data/interhub/snapshots/`、`archives/historical-results/`（immutable snapshots）
- 对照：`main_workflow.log` 中 RQ014 的每一条都逐项声明了「未触碰 X」，本计划没有对应条款。

**最小修复要求**
§7 增加「受保护路径清单 + 只读断言」，并要求 Phase B 的每次运行输出到
新的 `RUN_ID` 目录，附一条「未修改上述任一路径」的 sha256 复验收据。

### F-M5 Phase C「唯一变化是样本限定」不可机械执行（严重度：MAJOR）

**证据**
- 计划 `:211`：「不修改任何原检验的估计量、统计单位或阈值；唯一变化是样本限定」。
- 但四项目标的 IPV 来源并不同构：
  - RQ012B 使用的是**自带的估计器副本**
    `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_.../02_process/03_event_deviation/onsite_ipv/vendored_estimator/agent.py:813`，
    其 `cal_traj_reliability` 与核心版本一样保留 `sum(var)` 兜底（该文件第 871-874 行区域）。
    若 Phase B 只改 `src/`，RQ012B 的重估仍跑旧估计器。
  - RQ003 / RQ011B 的 OnSite 侧产物在
    `data/derived/onsite_competition/{RQ003_nsfc_external_evidence, RQ011B_matched_scenario, RQ012B_event_harm}/`，
    计划未核实这些产物是否逐行携带 `ipv_error`；若不携带，样本限定无法施加。
- 计划也未说明样本限定施加在**哪一级**：帧、锚点、case，还是分析单位本身
  ——而 §2.4b 刚刚论证过这个区分至关重要。

**最小修复要求**
Phase C 增加一个前置步骤 C0：逐 RQ 核实
(i) IPV 产物是否逐行含 `ipv_error`；(ii) 该 RQ 的分析单位；
(iii) 其估计器代码路径（核心 / vendored / 归档）及其是否受 Phase B 影响；
并对任何 (i) 不满足或 (iii) 不一致的 RQ，明确标记为「Phase C 不适用」而非默认可执行。

### F-M6 §5 的冻结未落在可信位置（严重度：MAJOR）

**证据**
- 计划 `:229`：「阈值 4 与 0.93·K 在本文件冻结，SHA 登记于计划校验和。」
- `git status --porcelain`：计划与 checksum 文件均为 `??`（未跟踪），
  `START_HERE.md` / `STUDIES.md` 为 `M`（未提交）。当前 HEAD = `main @ 6bdcc2e6`。
- AGENTS.md `:42`：「**Merged GitHub files are the source of truth.**」

**为什么重要**
「冻结」目前只存在于工作区。任何人（含计划作者）都能在无痕迹的情况下修改阈值并重算 checksum。
§3 明令「阈值必须在看到任何重估结果之前冻结」，这条禁令目前没有技术支撑。

**最小修复要求**
在 Phase A 启动之前，把计划 + checksum + registry 更新合入 `origin/main`，
并在 §5 记录冻结 commit SHA；此后任何阈值变更必须走版本化 amendment（如 RQ014 的 v1p1/v1p2 模式）。

---

## 5. NOTES（非阻断）

- **N1 `reports/` 首层目录与 AGENTS.md 不符（既存状况，非本计划引入）。**
  `AGENTS.md:9`：「`reports/` must keep only `studies/` and `knowledge/` as first-level directories.」
  实际为 `knowledge` / `plans` / `studies`，`reports/plans/` 已含 60+ 文件并有自己的 README 与
  「Planning rules」，`START_HERE.md:1032` 也把它登记为「Centralized plans/prompts」。
  RQ015 遵循了事实惯例，判为合规；但 `AGENTS.md:9` 已过期，建议在某次治理维护中更正
  （不应由 RQ015 承担）。

- **N2 与另一份独立复审的收敛。** 我在完成上述取证后才读到 `main_workflow.log`
  最后一条（2026-07-26T12:57:21，`RQ015-plan-v0-independent-review`，`REQUEST_CHANGES`），
  其关于 warm-up padding（305,824 / frame_index<4）与修正后画像（41.2794% / 52.5810%）的
  结论与我独立复算完全一致。两条独立路径收敛，F-B1 的可信度应视为很高。
  该条同时指出一项本文未覆盖的数值细节：3.8 m 下溢悬崖是 `n=1` 的算法，
  几何均值实现的实际悬崖约在 hw=4 时 1.734 m、hw=10 时 1.175 m——若属实，
  §2.4-D1 与 §2.4e 的临界点数字也需修订。**注意：该复审已完成一天，计划文件的 SHA-256
  仍与原始 checksum 一致，即 v0 未被修订，`START_HERE.md` 与 RQ015 README 的错误断言仍在原位。**

- **N3 §2.5 异常行的另一层含义（正面）。** 既然 `(ipv=0, ipv_error=1)` 已经是一个
  事实上的「未估计」哨兵，那么 Phase B-B2 想引入的「显式弃权」并非全新概念，
  而是把一个已经存在但未文档化的状态**提升为一等公民**。建议 §Phase B-B2 直接复用并扩展
  该语义（`NOT_ATTEMPTED` / `NOT_ESTIMABLE` 两态），而不是新造一套与之并行的状态机。

- **N4 §2.5 的一个可直接采用的观察。** `ipv_error > 1−1/√7` 的行**全部**是零值行
  （305,824 中非零行 = 0），且 `[0.6220356, 0.70)` 区间为空。这是一个干净的、
  零成本的完整性判据，建议写入 ledger 的健康检查：
  任何 `ipv_error ∈ (1−1/√7, 1)` 的行都应视为数据损坏并 fail-closed。

- **N5 `zxc-` 作业名前缀已遵守**（`:251`），这一点合规。缺的是资源估算与 sbatch 约定（见 F-B5）。

- **N6 rating 隔离。** 计划全文未涉及 rating/preference_score；Phase C 涉及 RQ010B
  时应显式重申 rating-free 边界或指明其属于 RQ014/RQ010B 的受控 rating 面。
  当前 §7 未提及，建议补一句。本复审全程未接触任何评分字段。

---

## 6. 未能验证的部分

| 项 | 原因 |
|---|---|
| §2.4b 的 case 级 8 行表（96.01% / 89.27% / … / 23.37%） | 无落盘脚本；重建需自行定义 case 聚合规则与是否剔除 `frame_index<4` 哨兵，而后者恰是争议点（F-B1）。自行定义会引入我自己的口径，无法构成对计划的独立校验。 |
| §2.4c 的距离分箱表 | 同上；且分箱边界、距离定义（瞬时中心距 vs 包围盒距）、以及是否含哨兵帧均未在计划中说明。 |
| §2.4d 的 sigma 历史两口径表（7,318/38,228 等） | 依赖 `σ=0.02` 与 `σ=0.1, hw=10` 的历史产物，未在本会话中定位到可读路径；`main_workflow.log` 2026-07-26 的 `RQ009-M3-zero-rate-audit` 条目复述了同一组数字，但同样无落盘脚本。 |
| §2.4e 的「3.8 m 下溢悬崖」 | 需要运行估计器做数值实验；本复审为只读且无计算授权。见 N2：另一复审员报告该数字口径有误。 |
| Phase B 在 HPC 上的实际可行性与资源量 | 硬约束禁止连接 HPC；仅能从仓库内的 schema/launcher/文档推断（见 F-B5）。 |
| RQ003 / RQ011B OnSite 产物是否逐行携带 `ipv_error` | 需要打开 `data/derived/onsite_competition/` 下的大体量产物并推断其字段语义；超出本次只读复审的取证预算，已转为 F-M5 的 C0 前置步骤要求。 |
| `wod_ipv_adapter.py` docstring 声称的 42,665 B / `7c60676e…` 源文件 | 全仓 `find -name analyze_wod_e2e_ipv_rating_pilot.py` 仅一个实例（35,672 B / `92826fab…`）。该 provenance 不符可能是 RQ014 lane 的既存问题，不在 RQ015 范围内，仅在 F-M2 中记录。 |

---

## 7. 通过条件（BLOCKING 关闭清单）

| ID | 关闭条件 |
|---|---|
| F-B1 | §2.5 重写为 warm-up 哨兵；§2.1–2.4 给出双口径；§5 增加 `NOT_ATTEMPTED` 前置过滤；同 PR 更正 `START_HERE.md:24-25` 与 RQ015 README:24 |
| F-B2 | 声明精确判据并复算；或把「兜底分支占比」降级为 Phase A 待测项（带埋点重跑）；更正 `START_HERE.md:18-19` |
| F-B3 | 交付 `estimability_ledger` schema（随计划 checksum）；明确 `K_source=assumed, K=7` 及其验证证据要求 |
| F-B4 | 交付严格 CSV 解析的落盘扫描脚本；扩充 checksum 清单至 ≥8 项 |
| F-B5 | §8 给出 managed 面扩展路线或替代合规路径、sigma01 兼容基线的现行位置、以及三段资源估算 |

关闭后建议流程：修订为 **v1** → 重新生成 checksums → **重新走独立双审**
（本复审针对 v0，不自动继承）→ Formal G1 → 冻结 spec → validate-only → 提交。

本复审为只读。除本文件外未修改任何文件；未创建或修改任何 `decision.md`；
未触碰计划、registry、数据或源码；未提交 Slurm 作业；未连接 HPC；未接触任何评分字段。
