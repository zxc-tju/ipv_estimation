# Track L leader — RQ015 收官：零点拆分、未知态查证、成文

仓库根：`<REPO_ROOT>`
看板：`<REPO_ROOT>/.codex-fleet/rq015l-consolidate/board/`
工作区：`<REPO_ROOT>/.codex-fleet/rq015l-consolidate/work/`

**PI 已授权。** RQ015 的 11 条轨道（A–K）已全部结项，`STATUS.md` 均为 `DONE`。
本轨是收官轮，**不做任何重算，不投 Slurm**，全部工作在本地已有产物上完成。

## 一、位置（不要跳过，写报告时也要先交代这一段）

最终用途是 online verification：判断自动驾驶车辆的 IPV 是否落在人类分布内。
**两个弃权机制串联**：机制一判「这一帧能不能估」，弃权则直接结束、不进机制二；
机制二是 RQ009 已 accepted 的 envelope 支持度判据。

RQ015 做的是**机制一**。起点是一个缺陷：数值下溢时代码退回「七个候选等权」的兜底，
而候选网格对称，于是必然算出 `ipv` 恰为 0、`ipv_error` 恰为 `1-1/√7 = 0.6220355269907728`。
**「算失败了」与「该个体完全自利」在数据里不可区分。** 整条线就是为了把这两件事分开。

K2 已交付全语料台账：`data/derived/rq015k_logdomain_gate/l1_v1/`
（510 个 L1 parquet 分片 + 510 个 manifest，总 14,473,982 行）。

## 二、本轮三件事

### L1：把 RQ009 的精确零点拆成两类（**PI 裁定：只拆分，不动 RQ009**）

RQ009 自己的报告
`reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/90_report/index.html`
**第 127 行**记有：打分目标存在 **~21.5% 的精确零点原子（273,819 / 1,270,566）**，
并明言这造成「80% boundary-tie / 1e-10 endpoint-nudge 覆盖脆弱」、削弱相关性，
限定了 interval-tie 行为与 practical null 的解释。

K2 台账能逐行区分「过门的真中性零」与「弃权而被记成 0」。**任务：把这 273,819 个零拆开。**

**第一步是判定 join 是否成立，不是直接 join。**
- RQ009 打分目标的行键是什么？K2 L1 的 `artifact_id=rq009_feature_matrix` 分区里
  `canonical_key`（= `product_row_key` + `|role=` + `measurement_role`）与
  `interhub_canonical_key` 各是什么口径？
- **1,270,566 与 8,994,736 是什么关系？**（子集？不同粒度？不同过滤？）
- **如果无法精确一对一 join，就如实写「不可精确 join」并说明卡在哪里，
  不许用近似匹配、不许放宽键、不许"大致对得上"。** 这一条比结果本身重要。

join 成立时给出：
`过门且 ipv_log==0` 的行数与占比、`弃权而记为 0` 的行数与占比、
以及后者按 `reason_code`（`NEAR_UNIFORM` / `NO_IPV_EFFECT`）与工程失败的拆分。
**这直接回答 RQ009 自己那条警告：那 21.5% 里有多少是真中性、多少是伪零。**

产出 `work/L1_rq009_zero_atom_split/` 下的机器证据 + 报告一节。
**不得修改 RQ009 的任何文件、不得重算它的 envelope、不得改它已 accepted 的结论。**

### L2：查清 OnSite 那 274,022 行为何是 `UNKNOWN`（**只查证，不重算**）

K2 台账里 OnSite 的构成是：`ATTEMPTED` 2,974 / `NOT_ATTEMPTED` 4,272 / **`UNKNOWN` 274,022**。
**未知态比真正尝试过的行多两个量级，这个比例本身可疑。**

要回答：
1. `attempt_status` 的 `UNKNOWN` 是在**哪一段代码、依据什么条件**写出来的？给文件与行号
2. 这 274,022 行**有没有**尝试估计所需的输入（观测长度、交互对、参考线等）？
   即：它是「数据确实不支持」还是「流水线没走到」？
3. 与 RQ015A 审计当时的口径是否一致？（RQ015A 记录 OnSite 仅 **1.06%** 的行携带 IPV 数值）
4. 若属于「流水线没走到」，说明补齐需要什么——**只给判断，不要动手补**

产出 `work/L2_onsite_unknown/` 下的机器证据 + 报告一节。
**WOD 906 行与 OnSite 2,974 行本轮不处理，继续保持不适用状态。**

### L3：成文（L1/L2 回来之后才动笔）

写一份完整交付：`board/reports/RQ015_consolidated_report.md`。必须覆盖：

1. **位置与问题**：两个弃权机制、机制一要解决什么
2. **缺陷的性质**：均匀兜底 → `ipv` 恰 0、`ipv_error` 恰 0.6220355269907728；
   为何七个候选会给出逐位相同的轨迹（目标函数在无交互时退化为 `cos(ipv)·内项 + 常数`，
   而候选网格 ⊂ (−π/2, π/2) 使 `cos(ipv) > 0`，正标量不移动 argmin）
3. **修复**：改到 log 域，`w = softmax(−MSE/(2σ²))` 是**精确恒等，不是近似**
4. **门的规格（冻结）**：`mse_spread == 0 → NO_IPV_EFFECT`；`max(w_log) < 0.20 → NEAR_UNIFORM`；
   否则 OK。**θ=0.20 是政策阈值，不是数据断点**，必须这样写
5. **确定性证据**：同一软件栈下 AMD EPYC 与 Intel **逐位相同**（348/348，Slurm 2024766）；
   Mac 与 HPC 不同，差异来自软件栈而非 CPU（2,300 个锚点中 1,867 个不同，最大差 70.4）。
   方法学结论：**曲面越平，argmin 越不可复现，但由曲面形状定义的量反而更可复现**
6. **全语料普查**：见下方数字表
7. **L1 的零点拆分结论**、**L2 的未知态结论**
8. **交付给下游的接口约束**：`ipv_log = 0` 是合法且高频的通过门估计值（门后 23.40%），
   判别只能用 `status` 与 `reason_code`
9. **方法学教训**（K2 换来的，必须写）：
   (a) 1 行 canary 测不到「多 worker 并发」与「工程失败行写盘」两条真实路径；
   (b) 每条验收判据都要有一次**故意让它失败**的验证——本轮出现两例
       「看起来在检查、实际没检查该检查的东西」（RQ009 `duplicates` 硬编码为 0、
       G 锚点比错基线）
10. **遗留项**：J 的 HT 分母与台账行的关系尚未确立

## 三、必须照抄的数字（已由监督方核定，**不要重算，直接引用**）

InterHub 全量 4,981,984 个 canonical 求解单元：

| | 计数 | 占比 |
|---|---:|---:|
| `OK` | 3,502,340 | 70.3001% |
| `NEAR_UNIFORM` | 1,457,746 | 29.2604% |
| `NO_IPV_EFFECT` | 19,964 | 0.4007% |
| `SOLVER_FAILURE`（工程） | 1,934 | 0.0388% |

RQ009 台账行域：`OK` 6,405,292 / 8,994,736 = **71.2116%**。
G 锚点（正确 HPC 基线）：`compared_rows=2300`、`max_abs_diff=0.0`。
RQ009 回填：`rows = unique_keys = 8,994,736`、`duplicates = 0`（实测）。
总台账 14,473,982 = InterHub 5,197,072 + RQ009 8,994,736 + OnSite 281,268 + WOD 906。

## 四、**分母纪律（监督方硬约束，违反即退回）**

现在至少有四个分母在流通：

| 分母 | 含义 |
|---|---|
| 2,646,058 | J 轨 HT 权重的全域分母 |
| 4,981,984 | InterHub canonical 求解单元 |
| 8,994,736 | RQ009 台账行 |
| 1,270,566 | RQ009 打分目标行 |

- **每一个比率后面必须紧跟它的分母**，不许出现无分母的百分数
- **不许在不同分母之间搬运比率**
- 求解单元与台账行的压缩比 2.804× 已知；**2,646,058 与 8,994,736 的关系仍未确立**，
  K2 报告 §4.1 已明写 `not yet established`，成文时照此表述，**不得称"域一致"**
- 与 J 的抽样估计对照时**只用台账行域**（差 0.0579 个百分点）并说明理由，
  求解单元域（差 0.9694）单独列。**仍不得写成"验证通过"**——
  可辩护的表述上限是「在台账行域上设计基估计与普查相差 0.06 个百分点」

## 五、硬边界

- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`
  与 `configs/ipv_sigma01_exact.json`
- **不改 RQ009 的任何文件**，不重算它的 envelope、不重算它的 4.78% 弃权率
- 不投 Slurm，不重解任何锚点，不重跑 K2 的 join
- 不提交 git commit；禁止 `git checkout -- .` / `restore` / `stash` / `reset --hard` / `clean -fd`
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）
- 全文禁用 `estimability` 与「测出/未测出 IPV」的说法。
  可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
- 描述性结果不得写成因果主张
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**

## 六、编制与汇报

**L1 与 L2 并行派两个 codex agent；两者都回来之后再派第三个写 L3。**
一轮自查，转 `WAITING_ON_COMMANDER`，**不得自行 DONE**。
**不做盲审、不做多路复审、不出第二版规格。**
反面案例：上一轮一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，科学结论产出为零。

派发用 `.codex-fleet/rq015a-run/board/detach_launch.py`：
```bash
python3 .codex-fleet/rq015a-run/board/detach_launch.py \
  --log <board>/reports/<AGENT>.log --pidfile <board>/<AGENT>.pid \
  -- codex exec --cd "$PWD" --model gpt-5.5 -c model_reasoning_effort="xhigh" \
     --sandbox workspace-write -c sandbox_workspace_write.network_access=true \
     "$(cat <你的 prompt 文件>)"
```
派完立刻自检 `ps -o pid,ppid,pgid -p <新pid>` → **PPID 必须是 1**。
⚠ `codex exec` **没有** `--ask-for-approval` 参数。**不得用 `danger-full-access`。**

**`claude -p` 是单回合的**：派完 codex 不要结束回合，在本回合内阻塞轮询
（`sleep 60` 看日志增长，每 5 分钟写一行 `progress.log`），
等 codex 真正结项 → 自查 → 写 `WAITING_ON_COMMANDER` → 才结束回合。

你必须维护三个文件：`board/STATUS.md`（覆写）、`board/progress.log`（追加）、
`board/commander_notes.md`（监督方写给你的，**每完成一个阶段读一次**）。

本轮全部工作在本地完成，**预计 1-2 小时**。若超过 4 小时仍未完成，写明原因并报监督方。
