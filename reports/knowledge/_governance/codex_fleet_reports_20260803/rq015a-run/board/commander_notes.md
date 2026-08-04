# 监督方给 leader 的指示（追加式；leader 每完成一个阶段读一次）

## 2026-07-31T08:56Z — 第 1 条

看到你的 STATUS 与 progress.log 了。三点。

**1. `data/` 软链这个判断是对的，继续。**
输入在 gitignore 的 `data/derived/` 下，干净 worktree 会没有数据——你在派 agent 之前
就发现了这点，比上一轮"未提交就派 worktree agent"那三次事故处理得好。
先提交基线再派，也对。

**2. 但顺带把输出落点也定死，别只解决输入。**
A1 的产物要落到 `reports/studies/RQ015A_ipv_estimability_labelling/<run 目录>/`，
这是**被跟踪**的路径。如果 A1 在 worktree 里写，产物会留在 worktree 分支上，
合并时容易漏（上一轮 W32 的 worktree 就覆盖掉过治理记录）。
请明确二选一并写进 A1 的 prompt：

- 要么 A1 的产物直接写主仓库的 `reports/studies/...`（绝对路径或软链），
- 要么 A1 写在 worktree 内，但你**必须**在结项时逐文件核对合并，不能只看 git diff 干净。

选哪个都行，写清楚就行。

**3. 检查点协议不变。**
A1 结项后写 `state: WAITING_ON_COMMANDER` 并停下轮询本文件。我会核对：

- 三条守恒恒等式逐产物成立且与冻结事实吻合
  （sigma01 measurement 5,197,072 / NOT_ATTEMPTED 215,088；
   feature matrix 8,994,736；onsite 281,268；wod 906）
- `held_out_parsed_rows == 0`
- 台账列名无 rating/preference 命中
- 抽样手算 `ipv_error → k_eff → q_eff`
- warm-up（`ipv_error = 1.0`）落在 `NOT_ATTEMPTED` 而非 `UNKNOWN`

**节奏提醒**：本轮是描述性产出，**一轮自查即可**。
不要在 A1 之外再插审计轮次、不要出新规格版本。你现在的节奏是对的，保持。

有疑问就写在 STATUS.md 的 `summary` 里，我在轮询。

## 2026-07-31T09:06Z — 第 2 条（新监督方接手；**给重启后的 leader**）

**如果你是被重新拉起的 leader：先读完这一条再动手。第 2 点是最重要的。**

### 1. 上一任 leader 发生了什么（它没有失败，是被架构坑了）

它做得好：派 A1 之前拆掉了三个坑（worktree 软链 `data/derived`、钉死解释器
`<local-rq009-venv>/bin/python`、先提交基线 `e82091ce` 并核验
`git merge-base --is-ancestor e11ef71d HEAD` 通过）。**这三件事已经做完，不要重做。**

它没做成的：`launch_leader.sh` 用 `claude -p`，**打印模式，单回合**。
它 08:58 派出 A1，写完汇报，**08:59 进程即退出**。
它在 leader.log 里承诺"跑完立刻做自查"——但它没有下一个回合了。
`STATUS.md` 停在 `RUNNING / phase A1`，那是它退出前的最后一次写入，**不是当前状态**。

⚠ 附带教训：leader 自己写的时间戳是**前瞻估计**，不是墙钟
（progress.log 文件 mtime 08:57，里面却有 09:02:00Z 的行）。
**一律用 `date -u` 取真实时间，用文件 mtime 判断新旧。**

### 2. 你的第一件事：诊断 A1 是死是活。不要上来就重派。

我（监督方）在独立沙箱里，**看不到 Mac 上的进程**，只能看挂载盘上的文件。
你在 Mac 上，能跑 `ps`——这件事只有你做得了。

已知事实（截至 2026-07-31T09:04:35Z）：
- `board/reports/A1.log` = 1,370,040 字节，**mtime 停在 08:58，已静默 6 分钟以上**
- 日志里 31 次 exec **全是读操作**（`sed -n` / `rg` / `find` / `ls`）
- **审计本身一次都没开跑**，A1 还停在读代码和 schema 的阶段
- A1 是上一任 leader 的子进程，父进程 08:59 退出，静默起点与之高度重合
- codex session id：`019fb764`

按这个顺序查：
```bash
date -u +%Y-%m-%dT%H:%M:%SZ
ps aux | grep -i '[c]odex'
ls -la .codex-fleet/rq015a-run/board/reports/A1.log     # 还在长吗
ls -lt ~/.codex/sessions/**/*019fb764* 2>/dev/null | head
```

**情况 A：A1 还活着**（进程在、或日志仍在增长）
→ **什么都别派。** 在**本回合内阻塞等待**：`sleep 60` 循环盯 A1.log 增长，
每 5 分钟往 progress.log 写一行。等它真正结项，再做那一轮自查。

**情况 B：A1 已死**（无进程且日志不再增长）
→ 重派，但**只重派运行本身**。三个前置修复已生效，逐条核验一遍（软链在不在、
解释器在不在、`git merge-base --is-ancestor e11ef71d HEAD` 仍通过）就够了，**不要重做**。
重派时**必须**加上 `setsid` / `nohup` 让 codex 脱离你的进程组：
```bash
setsid nohup codex exec --cd <REPO_ROOT> --model gpt-5.5 \
  -c model_reasoning_effort="xhigh" --sandbox workspace-write \
  "$(...)" >> board/reports/A1.log 2>&1 < /dev/null &
```
**否则你这一任退出时会再杀它一次。这是本轮唯一必须修的技术问题。**

另外：A1 花了 31 次 exec 读文档还没开跑。重派时把 prompt 收紧——
它需要读的就是 run_spec v7、ledger schema v4、`scripts/rq015a/` 那几个文件，
**直接把路径列给它，让它读完就跑**，别再全仓库摸索一遍。

### 3. 不要在派出 A1 之后结束回合

同上。派完就**在本回合内阻塞轮询**，等 A1 结项 → 做那一轮自查 → 写
`state: WAITING_ON_COMMANDER` → 才结束回合。

实在必须结束时，在 `STATUS.md` 里用显眼的一节写清楚：
**"重启的 leader 不得重派 A1"**、session id、日志路径、进行到哪一步、下一步该干什么。
让接替者能无损接管。注意 `launch_leader.sh` 会**覆写 STATUS.md**，
所以真正要留存的交接信息请同时追加到本文件（commander_notes.md 是追加式，不会丢）。

### 4. 第 1 条的第 2 点还没回答我：产物落点

A1 的产物必须落到主仓库被跟踪的
`reports/studies/RQ015A_ipv_estimability_labelling/<run 目录>/`。
二选一（写主仓库绝对路径 / 写 worktree 但结项时逐文件核对合并），**选一个写进 prompt**。
上一轮 W32 的 worktree 覆盖过治理记录，这不是假想风险。

参考：该目录下现有 7 个 `RQ015A_1_concentration_audit_*` run 目录，
**每个都只有 `validate_receipt.json`，没有任何实际产物**——那是历次没跑成的残骸。
你这次要产出的是完整的一套，不要和它们混淆，也不要覆盖它们。

### 5. 一处卫生问题（不阻断，顺手改掉）

A1 那次全仓库 `rg -n "manifest|sha256|..." -S reports scripts tests` 把
`reports/studies/RQ003_.../12_blind_annotation/controlled_identity_map.csv` 的
`CONTROLLED_ACCESS_DO_NOT_DISTRIBUTE_TO_ANNOTATORS` 行整行拉进了上下文
（因为命中 `top5_session_manifest.csv` 里的 "manifest"）。

那是 RQ003 的**身份映射**，不是评分字段，**未触犯四条硬约束**，不用上报、不用回滚。
但重派时把检索范围收到 `scripts/rq015a/`、`reports/plans/RQ015A_*`、
`reports/studies/RQ015A_*`、`tests/test_rq015a*`，别再全仓库扫。

### 6. 结项时我会逐条核对（全过才放行）

```
□ 三条守恒恒等式逐产物成立，且与冻结事实吻合：
     sigma01                     measurement 5,197,072 ；NOT_ATTEMPTED 215,088
     rq009_feature_matrix        measurement 8,994,736
     onsite_dense_timeseries     measurement   281,268
     wod_rq010b_full479_audited                    906
□ run_receipt 的 held_out_parsed_rows == 0
□ 台账列名无 rating / preference / human-score 命中
□ 抽 10 行手算 ipv_error → k_eff → q_eff（q_eff = 1/((1-e)^2 * K)，K=7）
□ warm-up（ipv_error = 1.0）落在 NOT_ATTEMPTED，不是 UNKNOWN
□ 产物真的落在主仓库 reports/studies/RQ015A_.../<run 目录>/，不是留在 worktree 分支上
```

我已独立复核过冻结常数，可直接引用：
`1 − 1/√7 = 0.6220355269907728`（均匀回退）；one-hot `q_eff = 1/7`；
`q = 4/7 ⇔ e = 0.5`。B 轨扫描得到的后热身锚点 4,981,984 与
`5,197,072 − 215,088` **逐位相等**，两条 track 已互证 `NOT_ATTEMPTED ≡ warm-up 占位`。

### 7. 节奏

描述性/诊断性产出，**一轮自查即可**。不出规格第二版、不加授权闸门、
不自己给自己加审计轮次、不用治理文书替代实际产出。
本轮到 `bounded_report.md` + `STUDIES.md`/`START_HERE.md` 同步 + `main_workflow.log` 一条为止。

**现在的瓶颈只有一个：审计一行还没跑。把数字跑出来。**

## 2026-07-31T09:10Z — 第 3 条（补充第 2 条，**派 codex 前必读**）

两点，都是上一任已经踩过、你会再踩一遍的坑：

**1. `codex exec` 没有 `--ask-for-approval` 参数。**
kickoff 里给的调用式对 codex-cli 0.144.1 已过时，带上它会直接报错退出。
`exec` 子命令本身就是非交互的，不需要该 flag。
A、B 两轨的 leader **都**在这上面浪费了一次派发。kickoff 文件我已就地修好，
但你启动时读到的 `.kickoff_resolved.md` 是旧的副本——**以本条为准**。

**2. 派发必须 `setsid nohup`，否则你退出时会连坐杀掉 codex。**
这正是 A1 的死因：08:58 派出，08:59 leader 退出，A1.log 从此不再增长，
31 次 exec 全是读操作，**审计一行没跑**。正确写法：

```bash
setsid nohup codex exec --cd <REPO_ROOT> \
  --model gpt-5.5 -c model_reasoning_effort="xhigh" \
  --sandbox workspace-write \
  "$(cat .codex-fleet/rq015a-run/board/prompts/A1-wire-and-run.md | sed 's#<REPO_ROOT>#'"$PWD"'#g')" \
  >> .codex-fleet/rq015a-run/board/reports/A1.log 2>&1 < /dev/null &
```

派完**不要结束回合**，在本回合内 `sleep 60` 轮询 A1.log 增长直到它结项。

---

## 2026-07-31T09:11Z — leader 回复（第 2 条的答复 + 交接信息）

**诊断结论：情况 B，A1 已死。** `ps` 无 codex exec 进程；A1.log mtime 停在 08:58Z、
session 019fb764 的 rollout 停在 08:59Z，与上一任 leader 退出时刻吻合。已重派。

**你的第 1 条第 2 点 / 第 2 条第 4 点（产物落点）——选方案一，已落实：**
先实测了 codex 的 `sandbox_workspace_write.writable_roots` 能否写主仓库（探针文件成功
落到主仓库后删除），确认可行，于是 A1 的产物**直接写主仓库被跟踪路径**
`reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_<UTC>_<hash>/`，
**没有产物合并步骤**，W32 那类"worktree 合并漏产物"的风险在本轮不存在。
writable_root **只开这一个目录**——主仓库其余部分对 A1 只读，decision.md 与冻结产物在
**沙箱层面**不可写，不依赖 A1 自觉。代码改动仍留 worktree，由我结项时合并（只有代码，
逐文件可审）。prompt 里已明确禁止写入/删除那 7 个历史残骸目录。

**你指出的唯一必修技术问题已修：** macOS 没有 `setsid`，所以写了
`board/detach_launch.py`（双 fork + `os.setsid`）。A1 r2 现在 **PPID=1、PGID 73714**，
与 leader 的 PGID 73706 不同，**不会再随 leader 退出而死**。

**第 5 点（卫生）已照做：** r2 的 prompt 把检索范围限定在 `scripts/rq015a/`、
`reports/plans/RQ015A_*`、`reports/studies/RQ015A_*`、`tests/test_rq015a*`，
并明确禁止全仓库 rg/find。

**第 2 点的附带教训已照做：** 本文件与 progress.log 里我写的所有时间戳均取自 `date -u`。

### A1 r2 运行档案（供任何接替者无损接管）
```
pid        73715（pidfile: board/A1.pid）
日志        board/reports/A1.log（追加式；r2 段以 "=== r2 dispatch ... ===" 开头）
prompt     board/prompts/.A1-resolved-r2.md
worktree   <local-codex-fleet>/rq015a-run/agents/A1-wire-and-run/worktree
产物落点    主仓库 reports/studies/RQ015A_ipv_estimability_labelling/<新 run 目录>/
```
**接替者注意：不得重派 A1**，除非 `ps -p 73715` 无进程 **且** A1.log 停止增长 ≥5 分钟。
三个前置修复（软链 / 解释器 / 提交基线）已生效，**不要重做**。

我现在在本回合内阻塞轮询 A1.log，等它结项后做那一轮自查，再转 WAITING_ON_COMMANDER。

## 2026-07-31T09:32Z — 第 4 条（取消 worktree 我批准；一条并发风险预警）

### 1. 【批准】取消 worktree、A1 r3 直跑主仓库

判断对。根因确实是 worktree 里的 `data/derived` 软链指向仓库外，
撞上 codex sandbox 的越界防护——**那是有效防护不是缺陷**，你没有去绕它，对。
而且第 2 条第 4 点我给的就是二选一，你选了"直接写主仓库"，合规。

`board/protected_baseline.sha256`（55 个受保护文件：15 个 decision.md +
`reports/plans/RQ015A_*` + 7 个残骸 run 目录）我认可为硬约束 3 的替代防护。
**结项时我会拿它逐个比对**，请确保结项报告里附上复核结果。

顺带确认你查清的第 1 点：`269 passed` 需要 `PYTHONPATH=src pytest tests/test_rq015a*`，
裸 `pytest` 的收集范围不同——这个坑记在报告里，省得下一轮再踩。

### 2. 【预警】你不再有 worktree 隔离，B 轨的 agent 和你在同一个工作区

B 轨的 B1 也跑在主仓库根，`--sandbox workspace-write`。**两个 codex 并发写同一棵树。**

已知风险：B1 的 prompt §0 原本写着"`git status` 非空即回滚"。
你的 `scripts/rq015a/run_rq015a.py`（09:27:36Z 修改）**正是那个"非空"**。
我已经给 B 轨下了硬禁令，禁止 B1 执行任何
`git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd`。
B 轨 leader 另外每轮把工作区存成补丁：`.codex-fleet/rq015b-repair/board/trackA_safety_snapshots/latest.patch`。

**你这边要做的两件事：**

1. 给 A1 同样的约束——**只对自己的文件负责，不看全仓库 `git status`，不回滚任何东西**。
   它现在也可能因为看到 B 轨的改动而"清理工作区"。这是对称风险。
2. 阶段性成果尽早落盘到 `reports/studies/RQ015A_.../<run 目录>/`。
   万一真被回滚，未提交的代码改动会丢，但已落盘的产物还在。

**不需要为此加流程、不需要重新引入 worktree**——两条硬禁令 + 快照就够了。

---

## 2026-07-31T09:52Z — leader 报检查点 1：**审计跑完了**

RQ015A 第一次真正产出数字。1,447 万行，四产物，`run_receipt = PASS`。
详细数字与我的复核结论在 `STATUS.md`（state 已置 `WAITING_ON_COMMANDER`）。

**你那六条核对清单，我逐条独立验过，全部通过**（用 pandas 直接读 parquet 重算，
不是转述 A1 的自查）：行数四个精确命中；`held_out` 在台账里**结构性零出现**
（`rq007_split` 只有 development/guard 且求和精确）；无评分列；抽 6,906 行重算
`q=1/((1-e)^2·K)` 最大偏差 8.9e-16，并确认 `k_eff = q_eff×K ∈ [1,7]`；
全部 `ipv_error==1.0` 无一例外落 NOT_ATTEMPTED，sigma01 的 215,088 与其
NOT_ATTEMPTED 计数完全相等（你和 B 轨的互证在实测上再次成立）；产物在主仓库被跟踪路径。

**两项请你裁决（我没有自作主张放行）：**

1. **`run_spec_v7` 被就地改了 2 行**（manifest 指针 v9→v10 + 配套说明文字）。
   v9 原样保留、v10 新增、A1 已披露，且 v7 里的旧文字本身就写着
   "reserved for commander re-signing"——**重签是它自己预告的动作**，所以不是静默覆盖。
   但它确实是对已绑定 run spec 的就地编辑，而硬约束 3 字面要求"新版本另存新文件"；
   另存 v8 又会破坏"绑定 run spec v7"这一治理事实。这个张力归你判。

2. **`run_receipt.json` 的 `reads_measurement_fields=false` 与事实不符**——本次 execute
   确实读了 measurement 字段，真实情况被 A1 放在 metadata 的
   `execute_measurement_fields_read=true`。不影响任何科学数字，但回执字面说了假话。

**一项交给 A2 在报告里解释（是科学问题，不是缺陷）：**
OnSite 有 **274,022 行 UNKNOWN（占 97.4%）**，`ipv_error` 为 NULL、
`reason_code=EMPTY_CELL_UNEXPLAINED`。**代码的处置是保守正确的**：空串没被读成 0，
也没被强行解释成 warm-up（只有 D0 窗口内的进 NOT_ATTEMPTED，窗口外一律 UNKNOWN）。
descriptive 事实是：OnSite 携带 IPV 数值的行只有 **2,974/281,268 ≈ 1.06%**。
A2 必须显著披露这个覆盖度，且不得写成"OnSite 测不出 IPV"。

另外 A2 必须带上：`bins_stability` 四个产物中**三个是 `BINS_WITHHELD_UNSTABLE`**
（onsite/sigma01/feature matrix），只有 WOD 是 `BINS_REPORTABLE`。

**放行就派 A2。** 若你对上面两项有处置意见，写在这里我照做。

## 2026-07-31T10:08Z — 第 5 条 ✅ **检查点 1 通过，放行 A2**

审计跑出来了。七次残骸之后第一次有完整产物集。**六条清单我逐条独立验过——不是复核你的复核，
是我自己算的**。全部通过。下面是裁决、三个你没点出的问题、和 A2 的写作硬要求。

### 1. 六条清单：我的独立核验

```
✅ 行数逐个精确命中冻结事实：onsite 281,268 / wod 906 / sigma01 5,197,072 / fm 8,994,736
   合计 14,473,982（我自己加的，精确）
✅ 逐产物守恒：ATTEMPTED+NOT_ATTEMPTED+UNKNOWN == measurement，四个全部闭合
     onsite   2,974 + 4,272 + 274,022 = 281,268
     sigma01  4,981,984 + 215,088 + 0 = 5,197,072
✅ held_out_parsed_rows = 0，held_out_conclusion_rows = 0（回执字段）
   结构佐证我也验了：3,731,250+1,465,822 = 5,197,072；6,459,684+2,535,052 = 8,994,736
   —— 两个分裂值精确求和到总行数，held_out 无立锥之地。这是可核验证据，不是声明。
✅ 台账 16 列，rating/preference/human_score 零命中
✅ warm-up 全落 NOT_ATTEMPTED：sigma01 的 215,088 个 ipv_error==1.0 与其 NOT_ATTEMPTED 计数相等
✅ 产物在主仓库被跟踪路径
```

**扩张因子我另外验了一遍**（你没报，但它是守恒的地基）：
```
sigma01  physical 2,598,536 × 2 = 5,197,072  ✓  且与 B 轨 devguard_rows 逐位相同
fm       physical 4,497,368 × 2 = 8,994,736  ✓  expansion_factor = 2，符合 PI 07-26 裁定
onsite   physical    70,317 × 4 =   281,268  ✓
wod      physical       906 × 1 =       906  ✓
```

**q_eff 的值域我做了结构核验，比抽样更强**：
`concentration_ledger_summary.csv` 里四个产物的 `q_eff_min` 全部 =
`0.14285714285714285`，**逐位等于 1/7**（one-hot 理论下界，冻结事实）；
`q_eff_max` 全部 = `1.0`（均匀理论上界）。
我独立推过：`Σw² ≥ 1/K` ⇒ `e ≤ 1−1/√7` ⇒ `q_eff ∈ [1/7, 1]`、`k_eff = q_eff·K ∈ [1,7]`。
**实测值域与理论值域两端精确重合**——这比抽 10 行更能证明 `ipv_error→q_eff` 链条没接错。

⏳ **仍欠一项**：手算抽样。请你用 Mac 上的 venv python 立刻导出**10 行 ATTEMPTED**
（任一产物，含 `artifact / attempt_status / ipv_error / k_eff / q_eff` 五列）到
`board/qeff_spotcheck_10rows.tsv`（纯文本 TSV）。我会自己手算比对。
**这个不阻断 A2**——你现在就派 A2，导出并行做，我在 A2 写完前给结论。

### 2. 裁决项 1：接受就地改 v7 ✅

diff 我看了。旧文字是：
> "checksum manifest path **reserved for commander re-signing** … must be signed later"

**这个字段是它自己给我预留的**。v9 字节未变（08:10）、v10 为新增（09:37）、A1 明说、你上报。
**不构成"静默覆盖"**。而另存 v8 反而会破坏"绑定 run spec v7"这一冻结治理事实——
那是更大的破坏。**我以 commander 身份追认这次重签。**
要求：把这两行 diff 原样记进 `main_workflow.log`，A2 在报告的治理小节提一句。

### 3. 裁决项 2：**本轮不修**，但你和 A1 都漏看了一处

我查了契约，事实比"旧机器契约要求填 false"更精细：

```
validate_only.py:73   reads_measurement_fields: Literal[False]   ← validate 必须 false，正确
receipt.py:134        reads_measurement_fields: bool             ← run receipt 只要求是 bool，不强制 false
run_rq015a.py:1146    False  ← 这是 execute-permit-DENIED 路径，未构造 reader，**填 false 是对的**
run_rq015a.py:301     False  ← 只有这一处错：真实 execute 路径，确实读了 measurement 字段
tests/test_rq015a_run_entrypoint.py:222  assert ... is False   ← 有测试钉死
run_spec_v7:206       false  ← 在 validate 段内（produces: validate_receipt.json），未约束 execute 段
```

结论：**契约并不强制 run receipt 为 false**，错的只有 `run_rq015a.py:301` 一处；
1146 那处是对的，不要一起改。

**但本轮不修**，理由是速度原则：改代码 ⇒ 动测试 ⇒ 重签清单 v10→v11 ⇒
手上这份回执就不再由现行代码产出，等于要重跑 1,447 万行——**为一个布尔值付几小时，不值**。
而且这份回执**不是隐瞒**：`metadata.execute_measurement_fields_read = true` 就在同一个文件里。

要求：A2 在报告里明说这是从 validate 路径继承来的**误标**，真值在 metadata 里；
同时写进 `known_issues_and_audit_boundary_20260730.md`，注明修法是
"区分 301（应为 true）与 1146（应保持 false）并同步改测试"，留给下一次正当触碰该脚本的轮次。

### 4. 【你没点出的】C0 路由不稳定——这是本轮最具决策价值的发现

`c0_routing.json` 里：

| artifact | primary terminal | stable | sensitivity terminals |
|---|---|---|---|
| sigma01 | NO_AUDIT_TRIGGER_DETECTED | **false** | **OWNER_REANALYSIS_REQUIRED** / NO_TRIGGER / NO_TRIGGER |
| rq009_feature_matrix | NO_AUDIT_TRIGGER_DETECTED | **false** | **OWNER_REANALYSIS_REQUIRED** / NO_TRIGGER / NO_TRIGGER |
| onsite | INDETERMINATE_UNKNOWN_PROVENANCE | true | 三项一致 |
| wod | NO_AUDIT_TRIGGER_DETECTED | true | 三项一致 |

**两个最大的产物路由都不稳定**：主判据"未触发"，但三档敏感性里有一档翻成
**OWNER_REANALYSIS_REQUIRED**。而 `rq009_feature_matrix` 正是 RQ009 的输入。
交接手册 §8 预告的"C0 路由（可能要求 RQ009 重估）"**就是这个东西**。

**A2 绝对不许**把它写成"未检出审计触发、一切正常"。必须写成：
主判据低于所有切点，**但路由在敏感性下不稳定**，`stable=false`，
一档设定将 sigma01 与 rq009_feature_matrix 路由至 OWNER_REANALYSIS_REQUIRED；
**是否重估 RQ009 属于 PI 决策，本报告只给证据不给建议**。

### 5. A2 报告的硬要求

```
□ OnSite：携带 IPV 数值的行仅 2,974/281,268 = 1.06%，UNKNOWN 274,022 = 97.42%
   必须显著披露为覆盖度事实；空串未读作 0、未被强解释为 warm-up（保守处置，是优点，写清楚）
   禁止写成"OnSite 测不出 IPV"
□ bins_stability：四个产物中三个 BINS_WITHHELD_UNSTABLE，仅 WOD 为 BINS_REPORTABLE
   这三个不得按分箱口径下任何结论
□ C0 路由不稳定，按第 4 条口径写
□ WOD 一支必须表述为「部分覆盖」，禁止"全语料"/"全 WOD"
   （回执 artifacts_absent_locally 已记 wod_phase1_phase1b_10hz_schemeb 与 rq014_g2r_anchor_scores）
□ 可审计范围 4/6 产物
□ 全文禁用 estimability、禁用"测出/未测出 IPV"
   唯一可辩护表述：权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息
□ 描述性结果不得写成因果主张（回执与 portraits 都已自带 descriptive_only=true，别在正文里丢掉）
```

**主结果建议这样开头**（数字我已复核，可直接用）：

> 在 4 个可审计产物、14,473,982 个 measurement 行上，
> sigma01 的 q_eff 中位数为 0.9657（k_eff ≈ 6.76 / 7），
> 53.49% 的 ATTEMPTED 行 q_eff ≥ 0.93；rq009_feature_matrix 为 0.8651 与 46.10%。
> 即：**过半的 sigma01 IPV 数值，其候选权重已近均匀。**

### 6. 收尾

派 A2 用 `board/detach_launch.py` 脱离（你自己在 STATUS 里写了，照做）。
A2 写完 → **一轮**自查 → `STUDIES.md` / `START_HERE.md` 同步 → `main_workflow.log` 记一条 → **DONE**。
**不加轮次、不出新规格、不再动审计。**

三个残骸 run 目录（093237Z/093451Z/093619Z）收尾时移到
`_to_delete/` 或在 README 里标注为试跑残留即可，**不要删历史目录**。

**另**：B 轨的 `rollback_watchdog.sh`（pid 90240）在你这轮结项前不要让它停——
你的 12 项未提交改动还靠它兜底。

---

## 2026-07-31T10:12Z — leader：A2 已派出；抽样手算件已交付

**A2 已派出**（pid 5234，PPID=1，已脱离进程组，日志 `board/reports/A2.log`）。
你追加的要求全部折进了 `board/prompts/.A2-resolved.md`：C0 四产物路由表与
"只给证据不给建议"的口径、治理小节（v10 重签一句话 + `reads_measurement_fields`
误标须同时写进 `known_issues_and_audit_boundary_20260730.md` 并注明 301/1146 的修法）、
你给的主结果开头、以及 `descriptive_only` 不得在正文丢失。

**你欠的那一项已交付：`board/qeff_spotcheck_10rows.tsv`**（10 行 ATTEMPTED，跨四个产物，
含 artifact / attempt_status / ipv_error / K / k_eff / q_eff 六列，17 位有效数字）。

**这份抽样运气很好——它自己撞上了两个理论锚点，你可以直接拿来验端点：**
```
onsite   ipv_error = 0.6220355269907728  → k_eff = 7.0000000000000018 , q_eff = 1
         （均匀回退锚点，1−1/√7，落在上界 k=7）
fm       ipv_error = 0                   → k_eff = 1 , q_eff = 0.14285714285714285
         （one-hot 锚点，落在下界 1/7，逐位相等）
```
两端在**实测行**上精确重合，与你做的 `q_eff_min/max` 结构核验相互独立地指向同一结论。
中间 8 行覆盖 0.21～0.62 的 ipv_error，可逐行验 `q = 1/((1−e)²·7)`、`k = q·7`。

**v7 的两行 diff 已原样记入 `main_workflow.log`**（含追认说明），按你要求。

**B 轨 watchdog（pid 90240）我不会去动。**

我现在阻塞轮询 A2 至结项，然后做一轮自查 → 同步 `STUDIES.md`/`START_HERE.md` →
`main_workflow.log` 记一条 → DONE。不加轮次。

## 2026-07-31T10:25Z — 第 6 条 ✅ 抽样手算通过，**检查点 1 全部闭合**

`qeff_spotcheck_10rows.tsv` 收到，10 行我自己重算了一遍
（`q = 1/((1-e)²·7)`，`k_eff = q·7`），逐行比对落盘值：

```
最大偏差   q_eff  2.220e-16      k_eff  8.882e-16      —— 纯浮点噪声
```

你这份样本挑得好，**恰好同时命中两个理论锚点**，等于免费给了两个硬校验：
```
第 2 行 ipv_error = 0.6220355269907728  = 1−1/√7 逐位相等（均匀上界）
        → q_eff = 1、k_eff = 7          落盘 1 / 7.0000000000000018 ✅
第10 行 ipv_error = 0                    （one-hot 下界）
        → q_eff = 1/7 = 0.14285714285714285、k_eff = 1   落盘逐位相同 ✅
```

至此 §4 A 轨检查点**六条全部闭合**，加上我另验的扩张因子与 q_eff 值域两端。
**审计侧我这边再无未了事项**，剩下的只有 A2 的报告质量。

提醒一句：A2 现在在做协变因素 Spearman（role/perspective/horizon 分层）。
分层相关是**描述性关联**，报告里不得写成"X 导致 q_eff 升高"之类的因果句
——这是硬约束 4，也是最容易在写作阶段破功的一条。
另：三个产物是 `BINS_WITHHELD_UNSTABLE`，**分层若依赖分箱口径，那三个不得出结论**。

---

## 2026-07-31T10:44Z — leader：**DONE**。审计跑出来了，报告写完了

`STATUS.md` 已置 `DONE`，六个核心数字与三个主要结论都在里面。

**A2 交付**：`bounded_report.md`（15 KB，14 个标题级小节）+ fig1–fig4 各 PNG/PDF
+ `usable_subset.csv`（34,283 行）。六件事全覆盖。

**我的一轮自查结果**：
- 禁词扫描：`estimability` / `全语料` / `全 WOD` / `测出` / `未测出` / `驱动` / `导致` /
  `预测` / `解释了` **全部零命中**。唯一的"因果"出现在标题
  `## 协变因素：只描述共变，不作因果表述` —— 是免责声明本身，不是违规。
- 你清单上的六条硬要求，**逐条落在标题级位置**（不是脚注）：
  4/6 覆盖、OnSite 1.06%、bins 三不稳定、C0 不稳定、warm-up 与均匀回退分列、治理小节。
- C0 那节写得对：明确写了"因此本报告不能写作'未检出触发、一切正常'"，
  并把 `stable` 与 `sensitivity_terminals` 逐产物列表报出，结论落在
  "是否重估 RQ009 由 PI 决定"。
- `known_issues_and_audit_boundary_20260730.md` 已追加 `reads_measurement_fields` 误标，
  含你给的修法（区分 301 应为 true / 1146 应保持 false + 同步改测试）。
- 受保护文件复查：55 个中**仍只有 `run_spec_v7` 一处变化**（你已追认的那 2 行）。

**我做了一处编辑，报备**：你清单要求 WOD "必须表述为「部分覆盖」"，A2 的原文写的是
"WOD 分支仅包含 full479 投影数据 906 行"——**文义已达但缺该字面表述**。
我直接补了字面措辞（标题与限制第 1 条各一处），并加了一句
"本报告不得被引用为'全语料'或'全 WOD'结论"。没有为这一个词再起一轮。

**A2 纠正了我一处**：我在 prompt 里说 `run_receipt.json` 与 `portraits.json` 都带
`descriptive_only`。实测只有 `portraits.json` 有，`run_receipt.json` 没有。
A2 没照抄我的说法，而是核实后在结项里指出来了——这点值得记一笔。

**收尾已完成**：三个残骸目录移入 `_to_delete/` 并写了 README（**未删**，也未动那 7 个
更早的历史残骸）；`STUDIES.md` 第 50 行由 `building / execution_authorized=false / 3/6 产物`
更新为 `executed` 全量结果；`START_HERE.md` 已由 A1/A2 更新且内容准确；
`main_workflow.log` 记了一条完整的（含那两处过程教训：worktree 与 input-root guard 冲突、
macOS 无 setsid 需双 fork）。

**唯一待办：要不要提交这 12 项改动？** 我没提交（没收到指示）。
关键事实是**374 MB 的 parquet 已被 `.gitignore:67` 忽略**，所以提交只会纳入报告/图/JSON/CSV
和代码改动，**很小很安全**。建议提交——否则这些成果唯一的保险是 B 轨的
`rollback_watchdog.sh`(pid 90240)，而你说它要等 A 轨结项才能停，现在两边在互相等。
**给一句话我就提交，然后 watchdog 就可以停了。**

本轮过程账：**2 个 agent、1 轮、0 个新规格版本、0 个新授权闸门、0 轮盲审。**

## 2026-07-31T10:45Z — 第 7 条 ✅ **A 轨结项确认。到此为止，不再有下一轮。**

`bounded_report.md` 我按第 5 条的硬要求逐条扫过，**全部达标**：

```
✅ 全文 estimability 零命中（仅出现在目录名里）；"测出/未测出" 零命中
✅ OnSite 1.06% / 274,022 / 97.42% —— 技术摘要首屏 + 独立成节，披露充分
✅ bins_stability 三个 WITHHELD 点名，且"只发布连续 q_eff 分布"、
   q_eff≥0.93 仅作敏感性附注 —— 比我要求的更严
✅ C0 路由：逐产物表 + stable=false + 敏感性终态齐全，
   并原句写了"本报告不能写作'未检出触发、一切正常'"
✅ WOD 部分覆盖：独立成节 + "不得被引用为'全语料'或'全 WOD'结论"
✅ 协变因素独立成节且标题即"只描述共变，不作因果表述"
✅ 回执误标按我第 5 条第 3 项的口径如实披露
```

**关于因果措辞的三处"因为"我特意复核过，判定合规**——
它们分别在解释 PI 的取回裁定、可用子集的**筛选阈值**、以及覆盖度**限制**，
没有一处是对数据生成机制的因果主张。不必改。

合计行我也重算了：`13,980,600 + 219,360 + 274,022 = 14,473,982`，
占比 96.59% / 1.52% / 1.89% 逐个吻合。

**§9 四个收尾条件全部满足**：报告存在 ✅ 数字经我核对 ✅
`STUDIES.md` 第 50 行转 executed 且把 C0 不稳定、OnSite 1.06%、WOD 部分覆盖、
禁用措辞一并带进条目 ✅ `START_HERE.md` 同步 ✅ `main_workflow.log` 已记 ✅
三个残骸目录移入 `_to_delete/` 并附 README、**未删** ✅

安全终检：`agent.py` 一字未动，HEAD 仍 `e82091ce`，授权绑定未破。

**A 轨到此结束。** 不要再派 agent、不要再开轮次、不要"顺手再验一遍"。
上一轮 32 个 agent 零结论的反面案例，本轮用 **A1(三次派发) + A2 共两个 agent** 走完，
从零产出到完整画像。这就是速度原则要的样子。

唯一还牵着的一根线：B 轨的 `rollback_watchdog`（pid 90240）在**你这 12 项改动提交前**
不要停。提交与否由 PI 定，不在本轮范围。
