# 独立审查 A（K2 执行方案）：从任务书出发，逐条验证

你是**独立审查方 A**。同时另有一位审查方在独立工作，你们互不可见——
**不要试图寻找或读取对方的产物。** 最终由监督方比对你们的分歧。

被审对象：`.codex-fleet/rq015k-fullcorpus-gate/board/K2-leader-kickoff.md`
这是一份**已获 PI 授权、但尚未派发**的执行任务书。本轮复审出结论、监督方裁定之后才会启动。
所以你的判断是有后果的：说「可执行」，它就会带着你没抓到的错误跑到集群上去。

## 背景材料（属既有记录，你可以读）

- `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md` —— K1 勘察：范围、单元数、RQ009 join 干跑、成本基线
- `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md` —— K1b 内存/并发实测（P6/P10/P16）与它自己的建议
- `.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/` 与 `.codex-fleet/rq015k-fullcorpus-gate/work/k1_pilot_summary.json`
  —— **原始实测产物。数字以这里为准，不以任何报告的转述为准。**
- `.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md` —— 门规格定稿与设计基估计
- `.codex-fleet/rq015g-hpc-resolve/board/reports/G_leader_adjudication.md` —— HPC 通道、确定性证据、成本参照
- `AGENTS.md`

## 你要审的五件事

### 1. 第四节的资源裁定（**本次复审的第一重点**）

第四节是**监督方推翻 K1b 建议之后自己算的**：P6 / `--mem=48G` / 418 片 / 2,508 核 / 约 1.15 小时。
监督方在本项目里已经出过多次算术与口径错误（分母口径、行数、计数），**它的算术不享有任何信任**。

请**完全独立地重算一遍**：
从 K1b 的原始 summary JSON 取每 worker 峰值 RSS 与吞吐，从 K1 报告与台账取全量求解单元数，
自己查一次当前集群实况（只读，方法见下），给出**你自己的**
（片数 / 每片 worker 数 / `--mem` / 总核·小时 / 墙钟）。

逐项对照第四节。**不一致就直接写「计划错了」并给出正确值。**
另外检查：
- 约束到底是哪一个在起作用——QOS 的 4,000 核合计上限、实际空闲核、实际空闲内存、还是单节点内存上限？第四节挑的那个约束对吗？
- 第四节内部是否自洽？它同时给了一个固定的 `--mem` 值和一条自适应公式，把该配置的 worker 数代进那条公式，得到的是不是同一个值？
- 墙钟外推用的是哪个吞吐口径（每 worker 还是每片），有没有把 pilot 的启动/载入开销算进去？

### 2. 第二节的门规格搬到批处理上是否可无歧义执行

规格本身**冻结、一个字不许改**；你审的是它的**批处理实现要求**够不够。至少：
- `weights_from_mse()`（`src/sociality_estimation/core/reliability_logdomain.py` 第 172-188 行）——
  **去读这段代码**，确认它的入参与第二节写的 `log_score_i = -mse_i / (2*sigma^2)`、`sigma = 0.1`
  是否严格一致：σ 是它的显式参数还是内部常量？有没有默认值会在调用方不传时悄悄接管？
- `mse_spread == 0` 的精确浮点相等，经过 parquet 往返、类型推断、或任何 float32 降精度之后还成立吗？
- 互斥 reason 的顺序、以及工程失败与两个科学 reason 的隔离——第二节的要求足以防住实现写反吗？
- 缺列、NaN、数组长度不为 7、以及「该行压根没有 `mse_per_candidate`」这几种情况，规格覆盖全了吗？

### 3. 第五节的分片、续算、重投、校验

对每一条问同一个问题：**这条能不能被机械地判定，不需要人来解释？**
- `expected_output_rows` 从哪来？在分片那一刻算得出来吗？
- 「同一 canonical key 只能出现在一个分片里」——按 `(单个 PKL, 行键区间)` 切，这一条是自动成立，还是需要额外证明？
- 各失败类型的重投阈值（`SOLVER_FAILURE` 100 行或 2.0%、`NON_FINITE_INPUT` 0.1%）
  放到 418 个分片、约五百万单元的规模上，是太松还是太紧？
  **会不会在一切正常的情况下就被触发、把整轮停掉？** 用 K1/K1b 实测的失败率去算。
- `.tmp` 写入 + 校验 + 原子 rename，在集群的共享文件系统上成立吗？

### 4. 第六节的验收判据

三条：对照设计基 CI、G 锚点逐位一致、worker 数不改变数值。
对每一条给出两件事：**它在什么情况下会漏判（真出错却判过），什么情况下会误报（没出错却判失败）。**
如果你认为某一条在统计上或口径上根本不成立，直接说，并说明该换成什么。

### 5. 范围与数字

第三节的每一个数字（InterHub 求解单元数、RQ009 行数、OnSite/WOD、
`NOT_ATTEMPTED` 与 `UNKNOWN` 计数）**回到源头核一遍**，不要采信任务书的转述。
第七节的 23.40% 与它的分 signature 拆分同样核。

## 共同的三个收尾问题（两位审查方措辞一致，便于比对）

**Q1. 第四节的资源配置正确吗？** 给出你独立算出的
（片数 / 每片 worker / `--mem` / 总核·小时 / 墙钟），并指明监督方错在哪一步（若有）。
另说明：**按你的配置投出去，最坏情况是什么。**

**Q2. 第六节的三条验收判据，能不能真正判定 K2 成功？**
逐条给出漏判情形与误报情形。缺哪条？多哪条？

**Q3. 明确判定：`可执行` / `需修改后执行` / `不应执行`。** 给理由。
另单列一条：**你认为这份任务书里最可能造成实际损失的一处。**

## 硬约束（违反即为审查失败）

- **你是审查方，不执行 K2。** 不得 `sbatch` / `srun` / `salloc`，不得重解任何锚点，不得建 K2 的 work_dir
- 允许**只读**的集群查询：`ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "<命令>"`，
  命令限于 `sinfo` / `squeue` / `sacct` / `sacctmgr show assoc` / `ls` / `du`。
  查到的原始输出连同 `date -u +%Y-%m-%dT%H:%M:%SZ` 存进你自己的工作文件，报告里引用它
- **禁止读取另一位审查方的任何文件**：不得打开 `board/reviews/` 下不属于你的报告，不得读不属于你的 `.log`
- 只读；除你自己的报告与 `work/` 下你自己的临时脚本外不写任何文件
- **不改** `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py` / `configs/ipv_sigma01_exact.json`
- 不提交 git commit；禁止 `git checkout -- .` / `git restore .` / `git stash` / `git reset --hard` / `git clean -fd`
- 工作区非空是**预期状态**（此前轨道留下的文件仍在），清洁性只查你自己创建的文件
- 不解析 RQ007 held_out，不读 RQ014 致盲字段
- 不对 `reports/` 做全仓库 `rg`（会把 RQ003 controlled-access 行拉进上下文）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**
- 解释器钉死 `<local-rq009-venv>/bin/python`
- 全文禁用 `estimability` 与「测出/未测出 IPV」的说法

写作要求：不用比喻和行话；每个数字带来源文件与字段名或行号；
**与任务书或监督方不一致时，直接写「计划错了」并给出正确值，不要客气。**
一轮到底：不写第二版规格，不提替代判据，不做阈值扫描。

产出写到 `.codex-fleet/rq015k-fullcorpus-gate/board/reviews/K2R_A_review.md`。
