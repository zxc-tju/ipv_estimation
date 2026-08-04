# C1 — sigma01 存档为何复现不出来：legacy 代码回放取证

你是 track C 的唯一执行 agent（代号 **C1**）。仓库根：
`.`
（下称 `$ROOT`，你的 `--cd` 已经是它）。

这是一次**取证**，不是修复。产出是一份报告 + 一个机器可读 summary。

---

## 0. 铁律（违反即本轮作废；每条都是前几轮的事故换来的）

**git 禁令**（C/D/E 三条 track 正在同一个工作区并发工作，工作区非空是预期状态）：

```
禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
禁止 git checkout 任何历史提交到主工作区（看旧代码只能用 git worktree add）
禁止 git commit / git add
禁止修改主仓库 .git/info/exclude、.gitignore
禁止删除或改动 5edd2810 相关的任何历史对象（不许 gc/prune/reflog expire）
```

清洁性只对**你自己创建的文件**负责。**不要看全仓库 `git status` 并据此做任何清理动作。**

**不得修改现行代码。** `src/`、`pipelines/`、`configs/`、`scripts/` 一律只读。
你的所有新文件只能写在这两个目录下：

```
$ROOT/.codex-fleet/rq015c-drift-forensics/work/      # 脚本、中间产物、JSON
$ROOT/.codex-fleet/rq015c-drift-forensics/board/reports/   # 报告
```

**四条硬约束**（与流程无关，不得放松）：
1. RQ007 held_out 集不得被解析。你只用 `split ∈ {development, guard}` 的行——
   B1 冻结样本已经过滤过了，别自己再去碰 held_out。
2. RQ014 致盲相关的评分字段不得读取。
3. 不得静默覆盖冻结产物或已接受的 `decision.md`。
   **`.codex-fleet/rq015b-repair/` 下的一切是冻结的，只读，不得写入。**
4. 描述性结果不得写成因果主张。

**术语禁令（全文，含代码注释、变量名、报告）**：禁用 `estimability`，
禁用"测出 / 未测出 IPV"。可辩护的表述是：
**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。

**不要对 `reports/` 做全仓库 `rg`/`grep`**——宽泛检索会把 RQ003
`12_blind_annotation/controlled_identity_map.csv` 的 controlled-access 行整行拉进上下文。
本任务不需要读 `reports/`。

---

## 1. 环境

- 解释器**钉死**：`<local-rq009-venv>/bin/python`
  （系统 python3 缺包，会把基线判错）。所有 python 调用都用它的绝对路径。
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`。**不要前瞻估计、不要手写时间。**
- 如缺 Python 依赖，直接在该 venv 里 `pip install` 并在报告里记一行，不要停下来问。
- 设 `export MPLBACKEND=Agg`（legacy `agent.py` 顶层 `import matplotlib.pyplot`）。
- 心跳：每完成一个步骤，追加一行到
  `$ROOT/.codex-fleet/rq015c-drift-forensics/board/C1_heartbeat.log`，格式
  `<UTC> | <步骤> | <一句话结论>`。**长步骤中途也要写**，让 leader 能看到你在动。

---

## 2. 背景与已核实事实（直接用，不要重新确认）

RQ015B 用**当前 HEAD 代码**重解锚点，复现不出 2026-06-12 存档的逐锚点值：
**复现门 gate_a 12/40，阈值 39/40**。PI 假设：存档由 HPC 上那份代码产出，
它与本地现行代码有差异。**本轮只做本地 git 取证，不接 HPC，不 ssh。**

已核实（leader 已验，别重复验）：

```
legacy commit 5edd28104bf5989e2dc258c9405ce897d7523cc4
  "Run full InterHub IPV with sigma 0.1"，提交日期 2026-06-12 00:32:37 +0800
  是 HEAD 的祖先，对象全部在本地（非浅克隆）
  当时是【扁平布局】：agent.py / ipv_estimation.py / process_interhub.py 在仓库根，
                      辅助模块在 tools/（tools/utility.py, tools/Lattice.py, tools/lattice_planner.py）
  现在：src/sociality_estimation/core/{agent,ipv_estimation}.py
        pipelines/interhub/process_interhub.py
        src/sociality_estimation/planning/（tools/ 的去处）
行数   agent.py 983 -> 1244     ipv_estimation.py 313 -> 675
兜底段落  legacy agent.py:871-874 == current agent.py:1136-1139 **逐字未变**
          （if sum(var): weight = var/(sum(var)) else: np.ones(n)/n）
          => 缺陷本身没漂移，要找的是它【周围】的差异
API   legacy ipv_estimation.py 有 class MotionSequence 与 def estimate_ipv_pair —— 与当前同名，
      但**签名/语义可能已变**，必须自己读 legacy 源码确定，不要假设兼容
```

**产出存档的作业脚本就在这个 commit 里**（leader 已读，参数已核对一致）：

```
git show 5edd2810:submit_full_datasets_sigma01_array.sh
  -> python process_interhub.py --skip-preflight --csv ... --pkl-root ... --output-root ... \
       --shard-index/--shard-count 4 --workers 96 --mp-start-method fork \
       --case-timeout-seconds 1800 \
       --reference-clip-margin-m 60 --reference-max-points 40 --reference-smooth-points 40 \
       --no-plots --only-incomplete
     环境：conda activate ipv；OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/SCIPY_NUM_THREADS 全 =1
     nuPlan 采样：20Hz -> 10Hz
```

这些参数与 `$ROOT/configs/ipv_sigma01_exact.json` 完全一致
⇒ **参数口径已被排除，不是候选解释。别再花时间在这上面。**

---

## 3. Leader 已定位的关键结构（这是你的搜索起点）

gate 判据在 B1 harness `run_t3_t4` 里，对 40 个 pilot 锚点：

```python
ipv_diff = abs(out["ipv"]        - row["legacy_ipv"])
err_diff = abs(out["ipv_error"]  - row["legacy_ipv_error"])
a_ok = ipv_diff <= 1e-6 and err_diff <= 1e-6     # gate_a
```

**40 个锚点的 gate 结果全在** `$ROOT/.codex-fleet/rq015b-repair/board/BLOCKED_B1.md`
（`gate_rows` 那一行 JSON，含每个 anchor 的 `ipv_diff` / `err_diff` / `signature` / `source` / `n_band`）。

Leader 从中读出的分层事实（你要在报告里复算确认）：

- **12 个通过者全部是 `signature=U`**（14 个 U 中过 12）；`Z` 0/14；`N` 0/12。
- `U` 是均匀兜底的不动点：权重被强制成 `ones(n)/n`，IPV 只由网格决定，
  对权重细节不敏感 ⇒ **一切依赖权重的锚点都对不上**。
- ⇒ 漂移落在**产生 `rel_dis` 的那条链路**（虚拟轨迹集合 / 观测轨迹 / 参考线），
  而不在兜底段落本身。
- 另注意：多个 gate_a 失败行 **`ipv_diff` 恰为 0.0 或 ~1e-17，但 `err_diff` 明显非零**
  （例：`ipv_001269|5|1` ipv_diff=0.0, err_diff=3.4e-05；
   `ipv_007762|90|1` ipv_diff=0.0, err_diff=0.0499）。
  这类行说明 **argmax/IPV 值一致但误差量一致性被破坏**——单独成一类，要在分层里体现。

---

## 4. 输入清单（**路径已给全，不要在仓库里摸索**）

只读，全部已确认存在：

```
冻结样本（复用，绝不重新抽样）
  $ROOT/.codex-fleet/rq015b-repair/work/sample_v1.csv        2300 行 + 表头
  $ROOT/.codex-fleet/rq015b-repair/work/sample_v1.sha256
  $ROOT/.codex-fleet/rq015b-repair/work/anchor_mse.csv       2300 行，当前代码的逐锚点解
  $ROOT/.codex-fleet/rq015b-repair/board/BLOCKED_B1.md       40 锚点 gate_rows

B1 harness（**读它，照抄它的取数与调用口径**，别自己另起一套）
  $ROOT/.codex-fleet/rq015b-repair/work/run_b1_rq015b.py
    关键函数：build_sequences(606) / legacy_weights_from_rel_dis(673) /
              diagnostic_for_anchor(687) / solve_current_details(742) /
              pilot_rows(832) / run_t3_t4(877)
    ROOT 定义在第 24 行是 parents[3]；你若把脚本放到别处，路径要自己改对

数据（全部在主仓库，gitignored）
  $ROOT/data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/sigma01_ipv_timeseries.csv
  $ROOT/data/derived/interhub/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/02_outputs/splits/case_split_assignment.csv
  $ROOT/data/interhub/raw/full_datasets/pkl/          (10 个 pkl：train_*.pkl, waymo_*.pkl)
  $ROOT/configs/ipv_sigma01_exact.json

当前代码（只读）
  $ROOT/src/sociality_estimation/core/agent.py
  $ROOT/src/sociality_estimation/core/ipv_estimation.py
  $ROOT/src/sociality_estimation/core/reliability_logdomain.py
  $ROOT/src/sociality_estimation/planning/
  $ROOT/pipelines/interhub/process_interhub.py

legacy 代码（用 worktree 检出，见下）
```

---

## 5. 第 0 步 — legacy worktree（**方法已定，照做**）

```bash
git worktree add --detach <local-codex-fleet>/rq015c-legacy-5edd2810 5edd2810
```

放在**仓库外**，且用 `--detach`（不建分支）。这只往 `.git/worktrees/` 加东西，
是**新增**，不动其它 track 的任何文件。

**不要在 worktree 里软链 data/，不要改 `.git/info/exclude`。** 理由：
legacy `agent.py` 只 `import` `tools.utility` / `tools.Lattice` / numpy / scipy / matplotlib，
`ipv_estimation.py` 只 `from agent import Agent`——**纯计算，无仓库相对的数据访问**。
所以回放脚本放在主仓库 `work/` 下，数据路径天然解析到主仓库，零共享状态改动。

**导入必须在独立进程里做，不许把 legacy 与当前实现混进同一个解释器。**
legacy 有顶层 `tools` 包，当前有 `src/sociality_estimation`——同进程混装会出难查的遮蔽问题。
做法：

```
进程 A（legacy）：sys.path 只含 worktree 根；import agent, ipv_estimation；
                  逐锚点算 -> 写 JSON（含 ipv、ipv_error，以及诊断量：weights、rel_dis 摘要）
进程 B（当前）  ：sys.path 含 $ROOT 与 $ROOT/src；照 B1 口径算 -> 写 JSON
进程 C（比较）  ：读两个 JSON + 存档值，算 gate
```
每个进程开头 `assert` 一下自己 import 到的模块 `__file__` 落在预期目录，写进 JSON 存证。

**回退配方**（仅当回放确实报出仓库相对路径缺失时才用）：
在 worktree 内建软链 `ln -s "$ROOT/data" <worktree>/data`，
**只软链、不改主仓库任何 git 配置**，并在报告里记录这次回退及原因。

---

## 6. 决定性实验（本轮核心，别被周边分析淹没）

### 6.1 先复现 B1 的"当前代码"侧（对齐基线）

用 B1 的 40 个 pilot 锚点（`pilot_rows` 的选法照抄），在**当前 HEAD 代码**上重算，
确认你能复现 `gate_a=12/40`。**这一步不通过就不要往下走**——说明你的取数口径与 B1 不一致，
先把口径对齐（对比 `anchor_mse.csv` 里同 anchor 的 `ipv_legacy` / `ipv_error_legacy` 列）。

写心跳：`baseline-replicated gate_a=<n>/40`。

### 6.2 主实验：legacy 代码回放同一批 40 锚点

关键：**legacy 的调用口径要从 legacy 源码本身推出来**，不要把当前 API 套上去。
必读（用 `git show 5edd2810:<path>` 或直接读 worktree 里的文件）：

```
5edd2810:ipv_estimation.py     313 行，看 MotionSequence 字段与 estimate_ipv_pair 签名/默认值
5edd2810:agent.py              983 行，重点 Agent.estimate_self_ipv(562)、
                               cal_traj_reliability(813)、get_cost_param(608)、
                               ibr_interact_with_virtual_agents(538)、模块级 sigma
5edd2810:process_interhub.py   看它怎么从 pkl 构造 MotionSequence、怎么切窗口、
                               nuplan 20->10Hz 在哪一步做、参考线三个参数怎么传下去
5edd2810:tools/utility.py      get_central_vertices / CalcRefLine 等参考线相关
5edd2810:tools/Lattice.py      TrajPoint
5edd2810:tools/lattice_planner.py
```

把 `--reference-clip-margin-m 60 / --reference-max-points 40 / --reference-smooth-points 40`
按 legacy `process_interhub.py` 的传递路径原样传下去。

对 40 个锚点各算一遍，得到 `ipv_legacy_code` / `ipv_error_legacy_code`，
用**同一判据**（both ≤ 1e-6）对存档值算 `gate_a_legacycode`。

写心跳：`legacy-replay gate_a=<n>/40`。

### 6.3 三种结局，都要能干净地说出来

**(1) legacy 代码能复现（gate_a ≥ 39）⇒ drift 确认。**
   接着必须回答"**哪些函数变了、变化如何改变逐锚点结果**"。方法：
   **定位到函数级，靠替换实验，不靠读代码猜。**
   在 legacy 侧逐个把某个函数换成当前实现（或反过来），看 gate 计数如何移动。
   建议按 leader 的分层证据排优先级（但要用证据裁决，不要预设答案）：
   - H2（最高优先）：产生虚拟轨迹集合的链路（lattice / `ibr_interact_with_virtual_agents` /
     `get_cost_param` / `solve_optimization`）——因为**只有 U 不动点过关**，说明权重整体变了
   - H1：误差量 `ipv_error` 的定义/归一化——因为存在 `ipv_diff=0 但 err_diff>0` 的一类
   - H3：参考线构造（`get_central_vertices` / `CalcRefLine` / clip / smooth 的实现细节）
   - H4：模块级 `sigma`、IPV 网格、窗口切法
   给出"换掉 X ⇒ gate_a 从 a 变到 b"的表格。这是本轮最有价值的产出。

**(2) legacy 代码也复现不出 ⇒ 不是本地 drift。**
   **如实报"本地取证不足以定论"，不要硬凑结论。** 但要把剩余候选**排序并各自附证据**：
   HPC 侧未提交改动 / 环境依赖版本（numpy·scipy 版本对 `linprog`、`minimize` 的影响）/
   输入数据版本（pkl 是否与 2026-06-12 同一份——可查 mtime、大小、以及存档 CSV 里
   是否含可交叉核对的场景计数）。并给出"要定论还需要什么"（哪怕它在 HPC 上）。

**(3) 部分复现 ⇒ 给分层**：哪一类锚点能对上、哪一类不能
   （至少按 `signature` U/Z/N × `source` nuplan/waymo × `n_band` FULL/RAMP 交叉，
   并单列"ipv 对得上但 error 对不上"那一类）。

**数值健康自查**（不可省）：非有限值计数、`solve_error` 计数、
40 个锚点是否都真的算出来了（不要把异常吞掉当"不通过"）、
legacy 与当前两侧的候选数 K 是否都是 7。

---

## 7. 附带交付：lyft / av2 可达性盘点（**只盘点，不搬运**）

PI 要决定后续要不要取这两个源的原始数据。**只回答"能不能拿、多大代价"。**
**不要真去搬数据、不要 ssh、不要改执行面。**

要读的文件（路径已给全）：

```
$ROOT/scripts/hpc/migrate_legacy_payloads.sh + .sbatch
$ROOT/scripts/hpc/inventory_legacy_layout.sh + .sbatch
$ROOT/scripts/hpc/archive_legacy_checkout.sh + .sbatch
$ROOT/scripts/hpc/finalize_legacy_inventory.sh + .sbatch
$ROOT/scripts/hpc/attest_migrated_snapshots.sh + .sbatch
$ROOT/scripts/hpc/ensure_interhub_data_topology.sh
$ROOT/scripts/hpc/sync_tongji_checkout.sh
$ROOT/data/interhub/README.md
$ROOT/data/interhub/raw/full_datasets/BATCH_CURRENT.txt
git show 5edd2810:process_argoverse.py          # av2：揭示了什么输入格式要求
git show 5edd2810:HPC_FULL_DATASETS_NUPLAN_AGV_COMMANDS.md
git show 5edd2810:PROJECT_STRUCTURE.md
<local-projects-root>/1_Codes/HPC_TONGJI_USAGE_GUIDE.md
```

输出四点，写成"**选项 + 代价**"给 PI：
1. 仓库里现成的取回/迁移路径是什么（哪个脚本、怎么调、需要什么前置）
2. 这两个源在 HPC 上的**预期路径与体量**（从脚本/文档/manifest **推断**；
   推断就标"推断"，不要写成已知事实）
3. 本地是否有任何可用的部分数据或中间产物
   （已知：`data/interhub/raw/full_datasets/pkl/` 只有 10 个 pkl，
    全是 nuplan `train_*` 与 `waymo_*`，**没有 lyft/av2**——你要复核并说明还有没有别处）
4. `process_argoverse.py` 揭示的 av2 输入格式要求（要什么字段、什么坐标口径、
   与现行 InterHub pkl 口径差多远）

**本节不要做成研究，控制在报告一节的篇幅。**

---

## 8. 交付物

```
$ROOT/.codex-fleet/rq015c-drift-forensics/board/reports/C1_drift_report.md   # 主报告（中文）
$ROOT/.codex-fleet/rq015c-drift-forensics/work/c1_summary.json               # 机器可读
$ROOT/.codex-fleet/rq015c-drift-forensics/work/gate_legacy_vs_current.csv    # 40 锚点逐行对照
$ROOT/.codex-fleet/rq015c-drift-forensics/work/run_c1_*.py                   # 你写的脚本
$ROOT/.codex-fleet/rq015c-drift-forensics/board/C1_heartbeat.log             # 心跳
```

`c1_summary.json` **必须**含：

```json
{
  "created_utc": "...",
  "legacy_commit": "5edd28104bf5989e2dc258c9405ce897d7523cc4",
  "head_commit": "...",
  "worktree_path": "...",
  "module_provenance": {"legacy_agent__file__": "...", "current_agent__file__": "..."},
  "python": {"version": "...", "numpy": "...", "scipy": "...", "pandas": "..."},
  "gate": {
    "n_anchors": 40,
    "gate_a_current": 12,
    "gate_a_legacy_code": 0,
    "criterion": "ipv_diff<=1e-6 and err_diff<=1e-6"
  },
  "stratified": {"by_signature": {}, "by_source": {}, "by_n_band": {},
                 "ipv_match_but_err_mismatch_count": 0},
  "health": {"nonfinite": 0, "solve_errors": 0, "anchors_actually_solved": 40, "K_legacy": 7, "K_current": 7},
  "localization": [{"swap": "...", "gate_a_after": 0, "note": "..."}],
  "verdict": "DRIFT_CONFIRMED | LOCAL_FORENSICS_INCONCLUSIVE | PARTIAL",
  "artifact_sha256": {"<relpath>": "<sha>"}
}
```

主报告结构（按此顺序，不要加治理章节）：
1. 结论（**一句话**给 verdict；然后 3–6 条要点）
2. 方法与口径（worktree、双进程隔离、模块 provenance、参数来源）
3. 基线对齐（6.1，是否复现出 12/40）
4. 主实验结果（6.2，含 40 行对照表要点 + 分层表）
5. 定位（6.3 分支，替换实验表格；或不足以定论时的候选排序与"还缺什么"）
6. 数值健康自查
7. lyft / av2 可达性盘点（选项 + 代价）
8. 局限与不做的事（明确写出：本轮未接 HPC；哪些结论只在这 40 个锚点上成立）
9. artifact 清单 + sha256

**写作要求**：
- 只写描述性结论，不写因果主张。
- 全文禁用 `estimability` 与"测出/未测出 IPV"。
- 数字必须来自你实际跑出来的产物，不许估计、不许从上文复述当结果。
- 如果某一步没跑成，**明说没跑成**，不要用文字绕过去。

---

## 9. 自检（结项前必做，写进报告第 9 节）

1. 列出**你自己创建/修改的全部文件**清单，逐个 `shasum -a 256`。
   **只查这份清单，不看全仓库 `git status`。**
2. 确认 `.codex-fleet/rq015b-repair/` 下没有任何文件被你写过
   （`ls -la` 比对 mtime 即可；那批文件的 mtime 应停在 7月31 17:3x–18:3x）。
3. 确认 `src/` `pipelines/` `configs/` `scripts/` 未被修改：`git diff --stat -- src pipelines configs scripts`
   应为空（只读这四个路径，不要 `git diff` 全仓库）。
4. `grep -ni "estimability" ` 你的报告与脚本 → 必须 0 命中。
5. 报告里每个数字都能在 `work/` 下某个产物里找到出处。

跑完自检，在心跳里写最后一行 `DONE <verdict> gate_a_legacy=<n>/40`，然后结束。

---

## 10. 节奏（速度原则，最高准则）

本轮是**诊断性产出**：一个 agent，一轮自查，出报告，结束。
**不做盲审、不做多路复审、不出第二版规格、不加授权闸门、不用治理文书替代实际产出。**
发现自己在写"计划 v2"就是跑偏了，停下来去跑数。

时间预算：主实验（6.1+6.2）优先保证跑完。若定位实验（6.3 分支 1）时间不够，
**先把已得的替换实验结果落盘并在报告里标注"未穷尽"**，不要为了做完而不写报告。
第 7 节（lyft/av2）如果时间紧，压缩到最短，但**必须有**。
