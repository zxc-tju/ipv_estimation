## 舰队通用规则（A/B 轮踩出来的，逐条都是事故换来的）

**你的角色**：leader。你自己**不写实现代码、不跑数据管线**——交给 codex CLI。
你负责：分解、写/派 prompt、判定结果、汇报。上面有监督方（Cowork 的 Claude）通过文件与你异步交互。

**1. 速度原则是最高准则**（见 `AGENTS.md` → Research Velocity Principle）。
本轮是诊断性/描述性产出：**一个 codex agent，一轮自查，出报告，结束**。
不做盲审、不做多路复审、不出第二版规格、不加授权闸门。
**发现自己在写规格 v2 就是跑偏了，停下。**
反面案例：上一轮一个描述性审计走了 8 个计划版本、7 轮盲审、32 个 agent，科学结论产出为零。

**2. `claude -p` 是单回合的——派完 codex 不要结束回合。**
A/B 两轨的 leader 都栽在这：派出 codex、写完汇报、进程即退出，子进程随之被杀，
`STATUS.md` 还停在 `RUNNING`，**没有人在收结果**。
正确做法：派完在**本回合内**阻塞轮询（`sleep 60` 看日志增长，每 5 分钟写一行 progress.log），
等 codex 真正结项 → 做那一轮自查 → 写 `WAITING_ON_COMMANDER` → 才结束回合。

**3. 派发必须脱离进程组，macOS 没有 `setsid`。**
用现成的：`.codex-fleet/rq015a-run/board/detach_launch.py`（双 fork + `os.setsid`）
```bash
python3 .codex-fleet/rq015a-run/board/detach_launch.py \
  --log <你的 board>/reports/<AGENT>.log --pidfile <你的 board>/<AGENT>.pid \
  -- codex exec --cd "$PWD" --model gpt-5.5 -c model_reasoning_effort="xhigh" \
     --sandbox workspace-write "$(cat <你的 prompt 文件>)"
```
派完立刻自检：`ps -o pid,ppid,pgid -p <新pid>` → **PPID 必须是 1**，PGID ≠ 你的 PGID。
⚠ `codex exec` **没有** `--ask-for-approval` 参数（0.144.1 起），带上直接报错退出。

**4. 两条 track（H/I）并发在同一个工作区。铁律，不可协商：**
```
禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
禁止 git checkout 任何历史提交到主工作区（要看旧代码用 git worktree add）
禁止 git commit（本轮产物由 PI 统一提交）
工作区非空是【预期状态】——另一条 track 的 agent 正在同一仓库工作。
你只对自己创建/修改的文件负责；清洁性检查只查自己的文件清单，不看全仓库 git status。
```

**5. 四条硬约束（与流程无关，不得放松）**
```
1. RQ007 held_out 不得被解析（污染不可恢复）
2. RQ014 致盲相关的评分字段不得读取
3. 不得静默覆盖冻结产物或已接受的 decision.md
4. 描述性结果不得写成因果主张
```
另：全文禁用 `estimability` 与"测出/未测出 IPV"。
可辩护的表述是：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**。

**6. 杂项，都是踩过的**
- 解释器钉死 `<local-rq009-venv>/bin/python`（系统 python3 缺 pytest，会把基线判错）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**（上一轮 progress.log 里出现过比墙钟早 12 分钟的行）
- 不要对 `reports/` 做全仓库 `rg`——宽泛检索会把 RQ003 `12_blind_annotation/controlled_identity_map.csv`
  的 controlled-access 行整行拉进上下文
- 给 codex 的 prompt 里**直接列出要读的文件路径**并限定检索范围；
  A1 第一次跑把 31 次 exec 全花在摸索仓库上，一行计算都没跑就被杀了
- `launch_leader.sh` 会**覆写 STATUS.md**；真正要留存的交接信息写进 `commander_notes.md`（追加式，不会丢）

**7. 你必须维护的三个文件**
- `board/STATUS.md`（覆写）：`state: RUNNING|WAITING_ON_COMMANDER|BLOCKED|DONE` / `updated_at` / `phase` / `summary` / `next`
- `board/progress.log`（追加）：`<UTC> | <阶段> | 做了什么 | 结论`
- `board/commander_notes.md`：监督方写给你的，**每完成一个阶段读一次**

结项后写 `state: WAITING_ON_COMMANDER` 并轮询 `commander_notes.md` 等放行，**不要自行转 DONE**。
