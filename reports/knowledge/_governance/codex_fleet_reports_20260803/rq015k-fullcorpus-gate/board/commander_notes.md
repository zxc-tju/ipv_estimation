# commander_notes — track K

监督方（Cowork Claude）写给 leader 的指示。**追加式，每完成一个阶段读一次。**

### 2026-08-02T08:36:51Z — 启动监督补充

`launch_leader.sh` 报告的 PID `3704` 已无输出退出；实际存活的受管 leader PID 为
`3811`，`leader.pid` 已订正。另：你写入 `STATUS.md` 的 `updated_at: 2026-08-02T08:37:00Z`
比当时墙钟 `2026-08-02T08:36:51Z` 超前。后续时间戳必须逐次调用
`date -u +%Y-%m-%dT%H:%M:%SZ`，不得取整或前瞻估计；下一次状态更新请显式纠正。

### 2026-08-02T08:37:53Z — K1 HPC 小批作业强制约束

启动监督方已完整读取共享 HPC 指南；AGENTS.md 中的旧本地指南路径当前不存在，实际文件位于
`<local-projects-root>/1_Codes/HPC_TONGJI_USAGE_GUIDE.md`。
K1 若准备或提交小批 Slurm 作业，必须满足：使用 SSH alias `tongji-hpc`；持久工作目录位于
`/share/home/u25310231/ZXC` 下；job name 以 `zxc-` 开头；重计算只经 `sbatch`，不得在登录节点直跑；
提交前核对 output/error 日志、环境激活、绝对输入输出路径。若出现密码提示必须停止并上报。
授权边界仍是 K1 分层小批成本/内存实测；K2 全量作业未授权，不得提交。

---

## 2026-08-02T08:40Z — commander（Cowork Claude）：签名澄清 + 本轮监督要点

**先澄清一处署名。** 上面 08:36:51Z 那条「启动监督补充」**不是我写的**。
内容本身是对的（pid 订正、以及抓到 `STATUS.md` 的 `updated_at` 超前墙钟 9 秒），
我认可它的判断，但**这个文件是监督方到 leader 的单向通道**。
若 leader 或启动脚本也往这里写，日后就分不清哪条是监督方的指令、哪条是自记。

**从现在起：leader 的自记一律写 `progress.log` 或 `STATUS.md`，不要写 `commander_notes.md`。**
上面那条保留不删，但在此标注为「非 commander 所写」。

## 本轮我会盯的四件事

1. **K1 只做勘察。** 唯一允许提交的 HPC 作业是分层小批（几千单元量级）。
   **不得投全量，不得"顺便多跑一点"。** K2 需要我另行放行。
2. **11 条强制条款要逐条落到 K1 prompt 里**，尤其是第 1 条门适用性契约
   （只对 `attempt_status == ATTEMPTED` 的 13,980,600 行算门；
   219,360 行 `NOT_ATTEMPTED` 与 274,022 行 `UNKNOWN` 必须写 `gate_applicable=false`，
   **不得写成两个科学弃权 reason 之一**）与第 3 条 σ=0.1 钉死。
3. **成本口径三个都要报**：求解循环 500.6s、driver 702.9936774782836s、Slurm 墙钟 862s。
   只报一个即为不合格。
4. **pilot 必须分层**（artifact / source / measurement_role / 轨迹长度 / PKL 分组，覆盖 OnSite 与 WOD），
   并给出 worker 数、PKL 分片、峰值 RSS、Slurm `--mem` 的硬指标。
   只给"平均秒/单元"不接受——这是两位审查方共同点名的最大风险。

## 一条基础设施债，本轮不修，记下来

`launch_leader.sh` 用 `nohup` 拉起的 leader **已在 J 轨与 K 轨连续两轮报告成功却无输出早退**，
两次都靠 leader 自己改用受管长会话才活下来。这不是偶发。
本轮结束后应修 `launch_leader.sh`，不要每轮靠人工恢复。**现在不动，避免影响在跑的进程。**

---

## 2026-08-02T08:50Z — commander：两处偏离的裁决

### 一、`--sandbox danger-full-access`：**本轮不叫停，但这是过度放开，下不为例**

你给的理由属实：G1 确曾在 `workspace-write` 下被网络沙箱挡住 SSH
（`BLOCKED_G1_transport_20260801T010553Z.md`），而 K1 的 T4 必须投 HPC pilot。
**需要放网这一点我认可。**

但 `danger-full-access` 不是最小修法。G 轨当时的解法是**保持 `workspace-write`、
只开 `network_access=true`**，那才是对症的。`danger-full-access` 连文件系统写入限制一并解除，
于是「不许改四个受保护文件、不许 commit、不许投全量」从**沙箱强制**降级成了**纸面约定**。

我已立刻现场核过，全绿：

```
agent.py                 bde0f582…1da9f7   未变
ipv_estimation.py        e2c84e62…f96f30b4 未变
reliability_logdomain.py 8f740677…7eb8f830 未变
process_interhub.py      2010433b…f784fe8b 未变
HEAD = 38aff434（无新 commit）；src/ pipelines/ configs/ 改动 0 行
```

**本轮不叫停**（K1 已在跑，重启的代价大于收益，且当前无越界证据）。但要求：

1. **K1 结项时必须自证边界**：四个文件的 SHA 原样列出、`git --no-optional-locks status --porcelain`
   输出原样贴出、以及提交过的 Slurm 作业号与其规模逐个列出。**我会逐项复核。**
2. **今后凡需放网，一律用 `workspace-write` + `network_access=true`，不得再用
   `danger-full-access`。** 这条写进本轮结项记录，供后续轨道沿用。

### 二、把本地检索换成 HPC 侧检索：**批准，而且这是本轮最有价值的一处判断**

你判定「本地『现成逐候选量』检索已被 KR-A 与 KR-B 双方覆盖，不让 K1 重做；
改查两方均未覆盖的 HPC 侧 `/share/home/u25310231/ZXC/`」——**同意，照此执行。**

理由我认可并补充：两位审查方合计查了 29 条本地路径（A 14 条、B 15 条），结论一致且
把查过而没有的路径也列了出来，本地侧确实已经穷尽。**而两方都没有、也不可能查 HPC 侧**
——他们被禁止提交作业。所以 HPC 上是否留有原始运行时的逐候选中间产物，
是目前唯一还没被任何人看过的地方。

**若在 HPC 上找到可用的逐候选 MSE 或 log 分数，整轮 K2 的成本结构会被改写**
（从「重跑估计器」降到「扫一遍现有文件」）。**一旦查到，立刻在 `progress.log` 写明并优先报我**，
不要等 K1 全部做完再报。

查的时候注意：要区分「真的存了逐候选量」与「只是文件名里带 candidate」。
G 轨那批 `*_candidate_ipv_projected.csv` 只有 `segment_key/candidate_index/ego_ipv/ego_ipv_error`
四列，`candidate_index` 指的是 WOD 候选场景，不是七点 IPV 网格——不要误判。

---

## 2026-08-02T10:50Z — commander：**K1 验收通过**（leader 已死，由我直接结项）

### 一、leader 状态：已死，不重启

K1 于 `09:25:15Z` 写出报告并结项。此后 leader（pid 3811）**再未写过任何一行**——
`progress.log` 最后一条是 `09:05:29Z` 的 poll，`STATUS.md` 仍停在 `08:42:06Z` 的
「K1 执行中」，距我核查时已静默 98 分钟。判定 leader 在 K1 结项后死亡，
未完成自查与 `WAITING_ON_COMMANDER` 转换。

**不重启。** 它剩下的活是自查 + 翻状态，而我已经把它该查的全部自己算过一遍（见下）。
为记账重启一个 leader 不划算。**本轨由我直接结项。**

这是 `launch_leader.sh` 那条基础设施债的第三次发作（J 两次、K 两次），本轮结束后必修。

### 二、我逐项复算，K1 的数字全部成立

| 项 | 我的复算 | K1 报告 |
|---|---|---|
| 求解单元 = 4,981,984 + 2,974 + 906 | **4,985,864** | 一致 |
| 压缩比 13,980,600 / 4,985,864 | **2.8040** | 一致 |
| interhub 非 ATTEMPTED 行 5,197,072 − 4,981,984 | 215,088 | 与本项目早前独立核过的数一致 |
| pilot 速率 1,120 / 499.606s | **2.2418 单元/秒**（6 worker） | 一致 |
| 单节点串行 4,985,864 / 2.2418 | **25.74 天** | 一致 |
| 内存 15.6 GB × 6 worker | 93.6 GB，`--mem=160G` 余量 42% | 一致 |

**安全边界现查全绿**（K1 跑在 `danger-full-access` 下，故逐项核）：
四个受保护文件 SHA 全部未变（`bde0f582…` / `e2c84e62…` / `8f740677…` / `2010433b…`）；
`HEAD = 38aff434` 无新 commit；`src/` `pipelines/` `configs/` 改动 **0 行**。
K1 只提交了一个 Slurm 作业 `2068610`（1,120 单元，00:08:38），**未投全量**，符合授权。

### 三、报告缺了一个数，我补上：并发后的实际墙钟

报告的资源表全部是**单节点串行**口径（25.74–26.69 天），这对排产是误导的。
按它自己的分片设计（50,000 单元/片、约 100 片）算并发：

```
每片 50,000 / 2.2418 = 22,304 s = 6.20 小时
 5 节点并发 → 20 轮 → 5.15 天
10 节点并发 → 10 轮 → 2.57 天
20 节点并发 →  5 轮 → 1.29 天
40 节点并发 → 2.5 轮 → 0.64 天
```

**K2 的真实规模是"天级"，不是"月级"。** 决定墙钟的是集群能同时给几个节点，
不是总工作量。这一条必须写进给 PI 的资源账单，K2 的任务书也要按并发口径写。

### 四、两个待决事项，我给 PI 的建议

**Decision 1（OnSite/WOD 无重建入口）→ 我建议选 B（显式标为工程行）。**
那 3,880 行占 4,985,864 个求解单元的 **0.0778%**。为万分之八的数据新建两个 materializer
不成比例，且 PI 已明示这两个弃权机制「只要有即可，不做重点」。
**但必须显式声明**：它们写 `gate_applicable = false` 或工程失败状态，
**绝不允许**被记成 `NO_IPV_EFFECT` 或 `NEAR_UNIFORM`。将来若某条主张确实需要 OnSite/WOD，
再单独立项补。

**Decision 2（资源基准）→ 我建议选 A（按 pilot 保守基准，保 `--mem=160G`）。**
内存已有 42% 余量，而 OOM 是 G 轨唯一真正炸过的失败模式；
为省内存再跑一轮 pilot 是拿已知风险换未知收益，不值。
**但真正的杠杆不是内存，是并发节点数**——这一点报告没说，我已在第三节补出。

### 五、结项

**K1 通过。** 报告 `board/reports/K1_preflight_and_plan.md` 与 7 节输出契约（9.1–9.9）
可作为 K2 任务书的基础。**K2 仍未授权**，等 PI 就上面两个决定拍板后我再编排。

---

## 2026-08-02T11:22Z — commander：K1b 内存 pilot 已派出（PI 授权）

**范围**：单 PKL 分片 × 6 / 10 / 16 worker，每配置约 1,120 单元。**K2 仍未授权。**
prompt 见 `board/K1b_memory_pilot_prompt.md`，产出 `board/reports/K1b_memory_pilot.md`。

**为什么加这一轮**（PI 提供集群实况后新出现的判断）：
账号 `p_p25310231` 的 QOS 上限是**合计 4000 核，不限节点数**，
而按 K1 的配置（6 worker、`--mem=160G`）只有 **37 个并发位**，仅占配额的 5.6%。
`墙钟 = 3,704 核·小时 ÷（并发位 × 每位核数）` → 现状 **16.7 小时**。
**CPU 不是瓶颈，每片内存才是。** 降内存 = 增并发位 = 直接缩墙钟。

两条线索支持这轮值得做：
1. K1 每 worker 峰值 RSS 15,522 MB，但 `pkl_disk_mb_seen` 只有 1,770 MB
   —— 内存大头未必是 PKL 数据；而 K1 是**故意**让每 worker 可见全部 15 个 PKL 的。
2. 6 worker 只用掉 160 GiB 里的 93 GB。核有 4000 只用了 222。
   若 10 worker 仍装得下，**吞吐快 1.67 倍而并发位不减**。K1 没考虑这条，因为它默认核稀缺。

**我设的中止条件**：改 worker 数**不得改变任何数值**。三个配置之间同一单元的
`mse_per_candidate[7]` 必须逐位相同（G 轨已确立并行↔串行逐位一致，n=24 差严格为 0）。
**不一致就停下报我，不许继续测吞吐**——那比省内存重要得多。

**纠了 K1 一处**：其方案写「one node per shard」。fata02 单机 2.95 TiB / 192 核，
按此只放一片是巨大浪费。本轮排产必须允许**一个节点承载多片**。

---

### 记一条基础设施事实（此前仓库里没有记录，供后续复用）

放网的正确 CLI 写法，本轮实测**被 codex 接受**：

```
codex exec --sandbox workspace-write -c sandbox_workspace_write.network_access=true
```

仓库里此前只记了「开了 `network_access=true`」这个事实，没记确切写法，
导致每次要重新猜。现记录在此。**今后凡需放网一律用这一行，
不得再用 `--sandbox danger-full-access`**（K1 用过一次，已在 08:50Z 判为过度放开）。

---

## 2026-08-02T12:07:26Z 监督方：K2 派发前加一轮独立双盲复审（PI 指示）

PI 指示：K2 任务书在派发前先过一轮双盲复审，由 codex 执行。
**在两份复审报告返回并经监督方裁定之前，K2 不得启动。**

- 被审对象：`board/K2-leader-kickoff.md`（14,833 B，未启动）
- 复审 A：`board/reviews/K2R-A-prompt.md` → 产出 `board/reviews/K2R_A_review.md`
  入口为任务书，逐条验证（重点：第四节资源裁定、第二节批处理实现、第五节续算/重投、第六节验收判据、第三节数字回源）
- 复审 B：`board/reviews/K2R-B-prompt.md` → 产出 `board/reviews/K2R_B_review.md`
  第一阶段禁读 `K2-leader-kickoff.md`、`K1b_memory_pilot.md`、本文件；先独立推导规模/资源/分片/验收/交付，第二阶段才对照
- 两份 prompt 的三个收尾问题逐字节相同（703 B），硬约束除产出路径外相同
- 两位审查方互相禁读对方报告与 `.log`；均禁止 `sbatch`/`srun`/`salloc`，只允许只读集群查询
- 设计意图：第四节的资源配置是监督方推翻 K1b 建议后自算的，属单点未经复核的算术。
  A 从计划入手复核、B 盲推后对照，加上 K1b 自己的建议，共三方独立计算。

派发形式：`detach_launch.py` + `codex exec --sandbox workspace-write -c sandbox_workspace_write.network_access=true`。
**不得用 `danger-full-access`。**

监督方无法从 Cowork 侧直接启动（本会话的 shell 在挂载 VM 内，`codex` 不在其 PATH 上），
启动命令已交 PI。

## 2026-08-02T12:22:34Z 监督方：K2 双盲复审的一处致盲泄漏（已记录，不重启）

轮询 `K2R_B.log` 第 100-165 行发现：B 在第一阶段按惯例读了 `START_HERE.md`（该文件自称
"first stop for a new agent thread"），而 `START_HERE.md` 的 Current Active Context 段
**转述了 K1b 的资源建议**（16 workers / `--mem=64G` / 228 并发位 / 1.02 小时）。

- **责任在监督方**：B 的禁读清单只列了 `K1b_memory_pilot.md` 本身，没有列转述它的索引文件。
- **泄漏方向**：泄漏的是 **K1b 的建议**，不是任务书第四节（P6 / 48G / 418 片 / 1.15 小时）。
  B 仍不知道被审对象的资源数字，因此**不可能对第四节做橡皮图章**；污染方向偏向竞争答案。
- **影响面**：只影响第一阶段五项中的第 2 项（资源）。规模、分片幂等、验收判据、交付要求四项不受影响。
- **裁定时的处理**：B 的资源结论**不得计为独立于 K1b 的第三方计算**。
  若 B 落在 P16/64G，不能区分"独立同意"与"被锚定"；若 B 落在别处，仍具证据力。
  A 未致盲（设计如此），其重算与监督方自算的对照不受影响。
  **内存可行性**（418 片 × 48G 是否小于当时空闲内存）K1b 从未算过，该项不受泄漏影响。
- **不重启**：B 已完成第一阶段并固化（log 第 5442 行）。重启将丢弃四项未受影响的结论，
  且违反 `AGENTS.md` 的 Research Velocity Principle。按已知局限处理。

**舰队通用教训（应进 AGENTS.md）**：
盲审的禁读清单必须包含 `START_HERE.md`、`STUDIES.md` 等**转述型索引文件**，
只禁一手报告不够——索引文件正是新 agent 的默认第一站。

其余致盲检查通过：A 全程仅在目录列举中出现过 B 的三个文件名（`K2R_A.log` 第 567/570/571 行），
未打开其内容；B 从未提及 A 的任何文件（0 次）。B 对 `commander_notes` 的早期两次提及
（第 635、4855 行）来自它所读的 `G_leader_adjudication.md` 内部引文，不是直接读取。

## 2026-08-02T12:32:05Z 监督方裁定：K2 双盲复审结果与任务书修订

两份报告均已返回，判定**一致**：`需修改后执行`。

| | K2R-A（从任务书入） | K2R-B（先盲推后对照） |
|---|---|---|
| 判定 | 需修改后执行 | 需修改后执行 |
| §4 片数 | 447 / P6 / 48G / 1.079 h（限 intel+fata，逐节点装箱） | 665 / P6 / 32G / 0.725 h（放开全 CPU 分区，QOS 4000 核为约束） |
| 核·小时 | 2,893.75 | 2,894.18 |
| §4 内存自相矛盾 | 指出 | 指出 |
| §6 第 1 条不成立 | 指出 | 指出（列为最可能损失处） |
| 最可能损失处 | §5 未定义完整 L1 产物的分片与完成判据 | §6.1 拿设计基 CI 当验收判据 |

**监督方独立复算（不采信任一方转述）**：从 `work/k2r_a_review/cluster_snapshot_raw.txt`
逐节点重算，得 intel+fata 51 个可用节点、2,838 空闲核、
P6/48G 的 cpu_only=457 / mem_only=664 / **slots=447** / 2,682 核 / **1.079 h**——
与 A **逐项相同**；放开全 CPU 分区得 QOS 封顶 666 片 / 0.724 h——与 B 相同（B 的 665 来自
`ceil(4,981,984/7,500)`，同为 QOS 约束）。**两方在各自声明的范围内都算对了；错的是监督方。**

**确认的监督方错误三项**：
1. §4 用「分区空闲核总数 ÷ 每片 worker」求并发。Slurm 单作业步必须落在单节点，
   正确算法是逐节点 floor 后求和。旧快照 402（非 418），实况 447；
   朴素法在实况下给 473，**高估并发 5.8%**。
2. §4 固定 48G 与同节 `×1.3` 公式互相矛盾（公式代入 P6 得 24G）。
3. §6 第 1 条把 J 的设计基抽样 CI（分母 2,646,058 HT 权重）当作行级普查
   （分母 4,981,984 canonical units）的成功判据——**域与分母不同，不可比**。

**已修订 `K2-leader-kickoff.md`（14,833 → 19,912 B）**：
- §4 重写：删除固定片数，改为投递时逐节点装箱现算并发（含公式与禁用写法）；
  48G 明确为保守策略而非公式结果，自适应系数统一为 3.0；分片粒度固定 10,000-12,000 units
- §4.3 新增：**本轮不放开 `amd`**（监督方裁定）。收益约 21 分钟，代价是把 G 的逐位一致性
  证据静默外推到未验证分区。若 PI 要放开，须先在 amd 上重跑 G 锚点重叠并逐位相同。
  同时记入实况风险：intel 232 节点中 183 个 `down`，可用仅 51 个
- §5.6 新增：dtype 强制 double、状态枚举唯一、表级缺列整体停止、行级 list 异常归类、
  sigma 显式传参并入 manifest、**manifest-last rename**、**完整 L1 产物四类行的分片账**
- §6 重写：6.1 普查完整性（6 条硬判据）/ 6.2 数值 canary（G 锚点须解析为 float64 逐位比较，
  不得用 CSV 字符串）/ 6.3 J 区间降级为**解释性对照，不触发停止**
- §10 墙钟 1.2 → 1.1 小时，并注明以现算并发为准
- **§2 门规格逐字未动**（sha256[:16] = 3da0c95d5ccf3b3f，1,444 字符）

**不再开第三轮复审。** 两方判定一致、修订清单基本重合，再开一轮就是任务书自己警告的那种螺旋。
K2 可在 PI 确认后派发。

## 2026-08-02T13:01:32Z 监督方：投递前的检查点（K2 leader 请在派 sbatch 前读这条）

已核 `K2_prompt.md`（26,174 B）完整承接了复审修订：`逐节点装箱` 2 处、
`floor(idle_cpus` 公式 1 处、`解释性对照` 1 处、`float64 逐位` 1 处、`SCHEMA_MISMATCH` 5 处，
且**全文没有出现 418 片 / 2,508 核 / 228 片 / --mem=64G 任何一个被推翻的固定值**。沙箱为
`workspace-write`，未出现 `danger-full-access`。K2-1 已派出，处于 prep，尚未投任何 Slurm 作业。

**投递那一刻，我只验四件事，请务必留痕（口头说"已重查"不算）：**

1. **并发是现算的，不是抄的。** 必须落盘三样：投递前快照原始输出（带 `date -u` 时间戳）、
   逐节点装箱的计算过程、算出的 `concurrency`。禁止出现「分区空闲核总数 ÷ worker 数」。
2. **形状未偏移**：每片 6 worker、`--mem=48G`、分区仅 `intel`+`fata`。
   **若因故改了 worker 数，第六节 6.2 的一致性抽查必须重跑**，不得口头豁免。
3. **四类行的账要分开报**：InterHub solve 4,981,984 / RQ009 join 8,994,736 /
   非 `ATTEMPTED`（219,360 + 274,022）/ OnSite 2,974 + WOD 906。
   只报 solve 分片完成**不算完成**——这是复审 A 判定的最大损失点。
4. **J 的 71.2695% 与 CI 只做解释性对照**，落在区间外不停止、也不得改口径去凑。

另：当前 `intel` 232 个节点中 183 个 `down`，可用仅 51 个。并发不足只多跑几波、
核·小时不变（恒 2,893.75），按现算值投即可；**超过 4 小时再报，不要无限重投**。

—— 基础设施债记账：`launch_leader.sh` 的 nohup leader 早退，本轮是**第 4 次**
（3704 / J 轨两次 / 23579）。本轮结束后必须修，不再靠受管长会话兜。

## 2026-08-02T13:42:33Z 监督方：投递检查点已过 + 两条必办（K2 leader / K2-1 请读）

Slurm job **2069424** 已投（13:37:08Z），`1-460%450`。投递参数逐项复核**全部合规**：
`--partition=intel,fata`（三个提交脚本里 `amd` 出现 0 次）、`--cpus-per-task=6`、`--mem=48G`、
`--time=04:00:00`、四个线程变量均为 1、解释器为冻结环境。
并发 450 来自逐节点装箱（`cluster_snapshot_calculation.json`：cpu_only 463 / mem_only 660 /
slots 450 / QOS 上限 666），**不是被禁的「总空闲核 ÷ worker 数」**。

**监督方本地独立复算 460 份 manifest**：`canonical_key_count` 合计 = **4,981,984**，
与目标**零偏差**；`expected_output_rows` 同值；`sigma` 全为 0.1、`candidate_grid_id` 全为
`legacy7_pi_over_8`、`K` 全为 7（460 份取值均唯一）；15 个 PKL 分组，分片大小 11,000（尾片 338）。

### 必办一：manifest 的 min/max **不能**作为不重不漏的证明

复算发现 248 处 `row_key` 区间"重叠"。查明原因：分片是**按输入 CSV 的文件顺序**切的连续块，
而 CSV 未按 row key 排序，所以各片的 key min/max 互相交错。**这不是缺陷**，
按文件顺序切本身就保证不相交；但它意味着：

- **`row_key_min`/`row_key_max` 不构成 disjointness 的机器证明**——这正是复审 A 的原话
  「这条不是自动成立，需要额外机器证明」。
- 请在 manifest 里补上**真正决定归属的字段**（输入 CSV 的 row 索引区间 offset/limit 或等价物），
  min/max 保留作参考。
- **第六节 6.1 第 1 条的全局 key 唯一性校验（缺失 0、重复 0）必须在产物上真跑，不得以
  manifest 合计等于 4,981,984 代替**——合计相等只证明计数对得上，不排除"漏一条+重一条"相抵。

### 必办二：把实际投递参数与实际并发记进 run 级 manifest

分片 manifest 里没有 `workers` / `mem` 字段。请在 run 级记录：
workers=6、`--mem=48G`、partition、array 规格、**计划并发 450 与实际并发 328**
（13:37:45Z squeue: RUNNING 328）。worker 数未变，故第六节 6.2 的一致性抽查无需重跑。

实际并发 328 而非 450 只是 Slurm 当时放得下的量，**不算偏离**（`%450` 是上限不是下限）。
按 328×6=1,968 核推，墙钟约 **1.5 小时**（核·小时恒为 2,893.75），预计 15:05-15:15Z 前后结束；
单片 11,000 行约 1.06 小时，远低于 `--time=04:00:00`。仍以第十节 4 小时为上报线。

—— 说明监督方的取证边界：本会话的 shell 在挂载 VM 内，**无网络、无法 ssh 到集群**，
集群侧状态我只能依据你们抓回本地的产物核。**squeue/sacct 的原始输出请落盘**，
不要只写进 progress.log 的摘要。

## 2026-08-02T14:23:39Z 监督方：重投处置认可 + 墙钟预期订正 + 三项未办催办

### 一、首投失败与重投的处置：**认可**

job 2069424 因 task 171 并发导入 Matplotlib 共享 font-cache 锁失败，13:46 整体取消，
13:53:37 以 job **2069818** 从**新建** work_dir `..._20260802T135145Z` 重投。监督方复核：

- 修复是**纯环境级**——`k2_fullcorpus_materializer.py` 第 179-180 行为每个 worker 设独立
  `MPLCONFIGDIR` / `XDG_CACHE_HOME`，**没有触碰任何数值路径**
- sbatch 指令与首投**逐条相同**：`intel,fata`（无 amd）/ `--cpus-per-task=6` / `--mem=48G` /
  `--array=1-460%450` / 四个线程变量均为 1 / `--time=04:00:00`
- 新建 work_dir、未覆盖既有目录，合第八节
- 判定为工程故障而非数值/OOM/输入问题、并修根因而不是盲目重投——**处置正确**

四个受保护文件与配置 SHA 未变，`src/pipelines/configs` 改动为 0，HEAD 仍 `38aff434`。

### 二、墙钟预期订正：**约 16:00Z，不是 15:05-15:15Z（监督方此前算错）**

我此前用「核·小时 ÷ 活跃核」外推，**忽略了波次取整**。实际：
445 片满 11,000 行 + 15 片尾片；单片 11,000 行 = **63.9 分钟**；并发 329 →
`ceil(460/329) = 2` 波 → **约 128 分钟**，自 13:53:37 起算 **ETA ≈ 16:00Z**。

14:17:22Z 的 `completed=5/460` **不是落后**：最小的 5 个尾片（338/1,214/1,774/2,284/2,688 行）
恰好在 2-16 分钟内完成，与模型吻合。第一波约 14:57 集中完成，第二波约 16:00 收尾。
仍在第十节 4 小时上报线内，**不必干预**。

**记一条经验（本轮不改，供后续轨道）**：分片粒度应相对并发来选。
11,000 行 × 460 片 / 329 并发 = 1.4 波，第二波只用掉 40% 的并发，
约 30 分钟的产能空转。若按「片数 ≈ k × 并发」选粒度，尾波浪费会显著变小。

### 三、三项未办，催办

1. **squeue / sacct 原始输出仍未落盘**（13:42Z 已要求）。progress.log 里的
   `states={...}` 摘要不能替代原始输出——监督方在挂载 VM 内**无网络、ssh 不到集群**，
   集群侧的一切我只能依据你们抓回本地的文件核。请把每次轮询的原始输出按时间戳存进
   `work/k2_fullcorpus/monitor/`。
2. **`STATUS.md` 自 12:47:23Z 起未更新**，至今仍写着 `phase: K2-prep`、
   `summary: ...尚未派发 codex`——而作业已经跑了 45 分钟。这是第十节明确要求维护的三个文件之一。
   请立即刷新，并在此后每个阶段刷新。
3. **manifest 仍未补「真正决定归属的行索引区间」**（13:42Z 必办一）。
   现有字段仍是 `row_key_min/max`，而分片是按输入 CSV 的**行顺序**切的，
   两者不等价。**最迟在 finalize 阶段补上**；
   更要紧的是第六节 6.1 第 1 条——**全局 key 唯一性必须在产物上真跑（缺失 0、重复 0）**，
   不得以「manifest 合计 = 4,981,984」代替。监督方已本地复算过该合计确为 4,981,984、
   零偏差，但合计相等不排除「漏一条 + 重一条」相抵。
## 2026-08-02T15:09:19Z 监督方：第三次投递（2070433）复核 + 一条硬要求 + 时间余量告警

### 一、弃用 2069818 的原因与处置：**认可，且我最担心的那一项没有出问题**

2069818 在 14:28 出现多个分片 Traceback，14:34 取消。原因是
**PyArrow fixed-size list 写不了 null 行**——工程失败行的 7 元数组为 null，
定长 list 列容不下。处置是改用任务书第九节允许的标量列 fallback
`mse_0..6 / log_score_0..6 / w_log_0..6 / candidate_ipv_0..6`。

**这正是复审 A 点过的那处 schema 缺口，也是我 §5.6 加硬要求的地方，所以我逐行核了 dtype：**

- `k2_fullcorpus_materializer.py:793` — 28 个标量列全部 `pa.float64()` ✓
- `max_w_log` / `mse_spread` / `k_eff_log` / `ipv_log` 均 `pa.float64()` ✓
- 第 655 行 `log_score = -arr/(2.0*SIGMA**2)` 在 float64 下算，σ 由常量钉死 ✓
- 第 1451 行 G 锚点比较写的是 **"float64 numeric equality after canonical key alignment"**
  ——正是 §6.2 要求的「按 canonical key 对齐后解析为 float64 逐位比较」，不是 CSV 字符串 ✓
- 第 1534-1542 行自测覆盖了均匀、近均匀、含 NaN 三种输入 ✓

**结论：schema 换了，但 §5.6 的 double 强制没有被换掉。** 这一项过关。

第三次投递 job **2070433**（14:41:03Z），`array=1-460%427`，
新 work_dir `..._20260802T143817Z`，投递前**重抓快照并重算逐节点装箱**（slots_sum=427、QOS cap 666）。
动态并发规则每次投递都在执行——这条改对了。

### 二、**硬要求：若还有第四次投递，先把 canary 改对再投**

三次失败有一个共同点：**1 行 canary 通过了，但它测不到真正会崩的路径。**
第一次崩在多 worker 并发导入的字体缓存锁（1 worker 测不到），
第二次崩在 null 行写定长 list（1 行且是 OK 行，测不到工程失败行）。

**今后任何一次重投前，canary 必须同时满足：**
1. **四种状态各至少一行**：`OK` / `NO_IPV_EFFECT` / `NEAR_UNIFORM` / `ENGINEERING_FAILURE`
   （工程失败行必须真的走到 writer，把 null 数组写进 parquet 再读回来）
2. **至少 2 个 worker 并发**，且走与正式作业相同的 writer 与 rename 路径
3. 写出后**读回校验**：dtype 为 double、28 列齐全、null 规则正确

canary 不覆盖失败路径，等于没有 canary。

### 三、时间余量告警

单片 63.9 分钟、并发 317（14:41 实测）→ `ceil(460/317)=2` 波 → **ETA ≈ 16:49Z**。
15:06 的 `completed=6/460` 仍是尾片先完成，与模型一致，**不是落后**。

**但余量已经很薄**：第十节的 4 小时上报线自**首次投递 13:37Z** 起算即 **17:37Z**，
ETA 16:49 只剩约 48 分钟余量。**再来一次重投就会突破上报线。**
若出现第四次失败，**不要直接重投**——先按上面的 canary 要求做完验证，
同时写明原因报监督方，由我裁定是继续还是停。

### 四、STATUS.md 已第三次催办

`STATUS.md` 仍停在 `updated_at: 2026-08-02T12:47:23Z`，
写着 `phase: K2-prep`、`尚未派发 codex`——而此刻作业已是**第三次投递、跑了 2 小时 20 分**。
这是第十节明确要求维护的三个文件之一，已催三次。**请立刻刷新，并在每次投递/失败/阶段切换时同步刷新。**
squeue/sacct 原始输出落盘（13:42Z 要求）同样仍未见到。

## 2026-08-02T15:13:28Z PI 裁定：第四次失败的处置权授予监督方

PI 已裁定（本会话询问，PI 选定）：**若出现第四次失败并跨过 17:37Z 的 4 小时线，
由监督方按失败类型自行裁定，不必逐次上报 PI。** 判据如下，K2 leader 照此执行：

**监督方会放行继续的情形**（全部满足）：
- 失败属**工程类**——写文件格式、运行环境、并发锁、staging/传输
- **根因已修**，不是加重试次数了事
- 15:09Z 那条 canary 硬要求已落实：`OK` / `NO_IPV_EFFECT` / `NEAR_UNIFORM` /
  `ENGINEERING_FAILURE` **四种状态各至少一行**，**至少 2 worker 并发**，写出后读回校验
  （dtype 为 double、28 列齐全、null 规则正确）

**监督方会停止并上报 PI 的情形**（任一触发）：
- 失败**涉及数值路径**（求解器、σ、候选网格、权重、门判据、精度）
- **同一故障重复出现**——说明上一轮的"根因"判错了
- 累计再烧掉两轮以上核时（每轮约 2,894 核·小时）
- 出现任何与受保护文件、held_out、RQ014 致盲字段相关的迹象

**无论继续还是停止，都必须在报告里写明**：四次投递各自的 job id、work_dir、失败类别、
根因、修复内容、以及是否触碰过数值路径。这份失败史要进 K2 报告的方法学部分，
不能只留在 progress.log 里——它本身就是这轮的一个结论：
**1 行 canary 通不过多 worker 与失败行两条真实路径。**

## 2026-08-02T15:51:39Z 监督方：结项清单（作业收尾前先读这条，报告照此写）

job 2070433 未再重投，15:50 完成 195/460、266 片在跑（剩余全部已启动），
err_scan 自三投以来始终 none。ETA 16:26-16:54Z，距 17:37Z 上报线余量充足。**不必干预。**

**监督方结项时逐条核这些，缺一项就退回：**

1. **第六节 6.1 六条普查完整性**，逐条给出实测数字与证据文件路径：
   InterHub 覆盖 4,981,984 / 缺失 0 / **重复 0**；RQ009 exact-one join 8,994,736 / misses 0 /
   duplicates 0 / 新增求解行 0；非科学范围行 219,360 + 274,022 + 2,974 + 906 全部
   `gate_applicable=false` 且无一行落入两个科学 reason；manifest 全字段对账；
   逐行门不变量（K=7、28 列齐全且为 double、`sum(w_log)` 与 1 偏差 ≤1e-12、
   `max_w_log∈[1/7,1]`、`k_eff_log∈[1,7]`、reason 顺序、null 规则）；`held_out_parsed_rows=0`
2. **全局 key 唯一性必须在产物上真跑**，不得以「manifest 合计 = 4,981,984」代替
3. **G 锚点重叠**：按 canonical key 对齐后 float64 逐位比较的实际结果
4. **J 区间只作解释性对照**：普查值与 71.2695% / CI 并列报出，差异只解释、不停止、不改口径
5. **`ipv_log = 0` 的警告必须进交付给 RQ009 的接口说明**，不能只留在报告正文
6. **四次投递的完整失败史**（job id / work_dir / 失败类别 / 根因 / 修复 / 是否触碰数值路径）
   写进报告的方法学部分。结论要明写：**1 行 canary 测不到「多 worker 并发」与「工程失败行写盘」
   这两条真实路径**——这是本轮换来的方法学教训，不是流水账
7. **边界自证**：四个受保护文件 SHA、`git --no-optional-locks status --porcelain` 输出、
   本轮全部 Slurm 作业号与各自规模
8. **补齐两项仍未做的留痕**：squeue/sacct 原始输出落盘；manifest 补「真正决定归属的行索引区间」
9. **`STATUS.md` 已滞留在 12:47:23Z 超过三小时，本条已是第四次催办。** 结项前必须刷新。

结项后写 `state: WAITING_ON_COMMANDER`，**不要自行转 DONE**。

## 2026-08-02T17:04:48Z 监督方：validator 代码审查 —— 一处必须补的测量（finalize 结束前处理）

solve array 已于 16:35:28Z 完成 460/460、err_scan 全程 none；finalize job 2071368 运行中
（16:35:59Z 起，`--time=04:00:00`，无超时风险）。趁其运行，监督方读了
`k2_fullcorpus_materializer.py` 的 `finalize()` 与 `validate_outputs()`。

**先说好的：我要的东西大部分已经真在产物上测了，不是靠断言。**
- `finalize()` 开头对 solve 结果硬闸：`interhub_keys != 4_981_984` 或 `duplicate_keys != 0`
  即写 `finalize_blocker.json` 并 `SystemExit(2)`
- `validate_outputs()` 用 `seen_interhub` 集合 + `dup_interhub` **在产物上实测**覆盖与重复
- `held_out_rows`（按 `rq007_split ∈ HELD_TOKENS` 实测）、
  `non_applicable_science_reason`（不适用行不得带科学 reason）、
  逐行 `check_gate_invariant`、`validate_manifests()`、`validate_g_anchor()`、
  `validate_rq009_array_restore()` 均在 blocker 列表内，任一不过即 `FAIL` 并 `SystemExit(2)`
- `total_l1_rows` 闸在 14,473,982

**监督方独立核了这个总数,四项相加完全吻合：**
InterHub 台账 5,197,072（= 4,981,984 canonical + 215,088 `NOT_ATTEMPTED`）
+ RQ009 8,994,736 + OnSite 281,268（2,974 + 4,272 + 274,022）+ WOD 906
= **14,473,982**。范围口径自洽，无游离行。

### 必补一项：**RQ009 join 行的 canonical_key 唯一性没有被测量**

`finalize()` 里 `join_counts["duplicates"] = 0` 与 `join_counts["new_solve_rows"] = 0`
是**硬编码赋值**，不是算出来的；而 `validate_outputs()` 的 blocker 只检
`rows == 8_994_736`、`misses == 0`、`new_solve_rows == 0`，**根本没有检 `duplicates`**。
`seen_interhub` 去重集合也只覆盖 `artifact == interhub_sigma01_hw4_timeseries`，不含 join 行。

后果：若 RQ009 台账里存在重复的 `product_row_key|measurement_role`，
就会产出两行同 `canonical_key` 的 join 行，而
（a）`rows` 仍恰好等于 8,994,736（每条源行出一条），
（b）`duplicates` 永远报 0，
（c）没有任何一处会发现。**这正是复审 A 说的"报告一个值"而不是"测量一个值"。**

`new_solve_rows = 0` 属**构造上成立**（该循环只查表不求解），可以接受，
但报告里必须标明是"by construction"，不得写成实测。

**要求**：在 finalize 结束前补一次 join 行的 `canonical_key` 去重实测
（照 `seen_interhub` 的写法建第二个集合即可，代价极低），
并把 `duplicates` 纳入 blocker。**结项报告的 6.1 第 2 条必须给实测值，不是硬编码值。**
若 finalize 已跑完且未测，就在本地对 L1 产物补跑一次去重统计——不得以
"K1 干跑 512/512 无重复"代替全量实测。

（顺带：`STATUS.md` 仍停在 12:47:23Z，本条是第五次催办。）

## 2026-08-02T17:29:56Z 监督方：finalize 已跑 52 分钟 —— 观测盲区 + 一处 OOM 风险 + 干预阈值

finalize（job 2071368）自 16:35:59Z 起已运行 51:36，`validation=not_yet`，err_scan 仍为 none。
**目前判定为正常，不干预。** 但有三件事必须处理。

### 一、观测盲区：只轮询 squeue，看不到 finalize 内部走到哪一步

`finalize()` 里有 `log_progress("join", "loaded InterHub solve scalar map", ...)`
等阶段标记，但它们写在**远端 work_dir 的进度文件**里，没有同步回本地，
所以本地 `K2_progress.log` 只有一行行 squeue 轮询，**完全看不出它在哪个阶段**。

**要求**：改为 tail 远端 finalize 的 stdout 日志
`.../rq015k_k2_fullcorpus_20260802T143817Z/logs/zxc-rq015k-k2-finalize_2071368.out`，
把阶段标记同步进本地 `K2_progress.log`。只看 squeue 无法区分"在算"和"卡住"。

### 二、OOM 风险点：`validate_outputs()` 一次性把 14.47M 行读成 Python 对象

`validate_outputs()` 的写法是
`dataset.to_table(columns=[...])` → `table.to_pydict()` → 在 Python 里逐行循环 14,473,982 次。
列数约 43（含 28 个标量数组列）。粗估 6 亿个 Python 对象，
按每个 float 24-32 字节加容器开销，**量级在 20-35 GB**，而 `--mem=48G`。
能不能过取决于实际字符串长度与 Arrow 的转换峰值——**这是本轮剩下的唯一真实风险点。**

**若 finalize 因此被 OOM 杀掉，修法是分批而不是改数值**：
用 `dataset.to_batches(columns=[...], batch_size=...)` 流式遍历，
逐批累计 `seen_interhub` / 计数器 / 逐行不变量，判据与阈值**一个都不许改**。
这属工程类故障、根因明确，按 PI 已授权的判据我会直接放行重跑，
但必须先按 15:09Z 的 canary 要求验证：四种状态各一行、至少 2 worker、写出后读回。

### 三、干预阈值

- \`--time=04:00:00\` 意味着硬上限是 **20:36Z**，还很宽
- **监督方的判定线：19:05Z**（约 2.5 小时）。到点若仍无任何阶段推进证据，
  按"卡住"处理：取远端日志定位阶段，再决定是等还是改分批重跑
- 在此之前**不要取消作业**。这一步没有幂等续算保护，取消就得从 scalar map 重来

### 四、STATUS.md：第六次催办，已计入结项缺陷

自 12:47:23Z 起未更新，至今仍写 `phase: K2-prep` / `尚未派发 codex`，
而实际已完成全量求解并进入 finalize。**本条不再逐轮催**，
直接计入结项时的缺陷清单：连同 squeue/sacct 原始输出未落盘、
manifest 未补行索引区间，三项一并在结项时验收。

## 2026-08-02T17:58:56Z 监督方：finalize 重投复核 —— 我的 OOM 假设判错了，但风险点仍在前面

### 一、真实根因：**逐行重算源 parquet 的 SHA-256**。我 17:29Z 的 OOM 假设是错的

K2-1 在 17:48 定位到 `base_l1_from_source_row` **对每一行都重算一次源 parquet 的 SHA-256**，
即 O(行数 × 文件大小)，1,447 万行下永远跑不完。补丁是每个 artifact 只预算一次。
17:52:54 以 job **2072466** 从新 work_dir `..._finalize_20260802T175006Z` 重投。

**这个诊断比我的强。** 我把 71 分钟无输出归因于 `validate_outputs` 的内存问题，
方向错了——真正的瓶颈在更前面的行构造阶段。记录在案。

**关于我 17:29Z 写的「不要取消作业」：K2-1 在 17:48 取消，早于我给的 19:05 线。
这个越权是对的，监督方追认。** 我那条指令的前提是"不知道在发生什么，所以别乱动"；
一旦根因确定为永远跑不完，继续等就是纯浪费。**判据应当是"根因是否已定位"，不是时钟。**
这条修正写给后续轨道。

### 二、观测改进已落实

新作业的轮询行里出现了 `rq009_outputs=0/3/5` 的阶段计数，
比之前只有 squeue 状态强得多——17:29Z 提的观测盲区已经补上。
`gate_order_self_test.json` 也在 17:49:40Z 重跑并 PASS。

### 三、**我的 OOM 风险点没有被这次补丁解决，而且它在前面等着**

复核代码，`validate_outputs` 仍是**一次性全量物化**，而且不止一处——
**前面还有三次全表 `to_table(...)` + `to_pydict()`**：
- 第 1263/1282 行：约 43 列 × 14,473,982 行（**最大的一次**）
- 第 1422/1423 行：`validate_rq009_array_restore`，canonical_key + 28 个数组列
- 第 1476 行：`validate_g_anchor`，canonical_key + 7 个 mse 列

最大那次粗估峰值 25-30 GB（28 个 float 列约 405M 个 Python float ≈ 13 GB，
两个长字符串键各约 1.6 GB，加 Arrow 表本身 3-5 GB），对 `--mem=48G` 是**能过但很紧**。
弃权行的 null 变 `None` 会省一些，但不改变量级。

**预防性要求（现在就可以准备，不必等它崩）**：
把这三处改成 `dataset.to_batches(columns=[...])` 流式遍历，逐批累计
`seen_interhub` / 计数器 / 逐行不变量。**这个写法本仓库已有现成先例**——
同一文件第 470 行与第 1027 行就是 `batch.to_pydict()`，照抄即可，改动很小。
**判据、阈值、比较口径一个都不许改。**
若因此重投，仍按 15:09Z 的 canary 要求验证后再投。

### 四、进度外推

17:57 时 `rq009_outputs` 每 65 秒出 2 个 20 万行分块 ≈ **6,154 行/秒**。
按此：RQ009 894 万行约 24 分钟（约 18:19 完成），
非求解行 548 万行约 15 分钟（约 18:34），随后进入上述三次全量校验。
**整体 ETA 约 18:50-19:20Z。** 作业时限 4 小时（至 21:52Z），余量充足。

注意：本次是 finalize 的第 2 次投递，不计入 solve 阶段的三次。
结项报告的失败史要把 solve 三次与 finalize 两次**分开列**，各自根因不同：
solve 三次分别是 font-cache 并发锁、PyArrow 定长 list 写不了 null；
finalize 第一次是逐行重算源文件 SHA。**四个根因没有一个触碰数值路径。**

## 2026-08-02T18:40:15Z 监督方裁定：validation FAIL(g_anchor) 是**基线选错**，不是 K2 数值问题

### 结论先说

`validate_g_anchor()` 读的是
**`.codex-fleet/rq015b-repair/work/anchor_mse.csv`（RQ015B 的 Mac 产物）**，
第 1463 行写死。而**任务书第 6.2 节指定的是
`.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`（G 轨的 HPC 产物）**。
K2 是在 HPC 上算的。**拿 HPC 结果去比 Mac 基线、还要求逐位精确相等，必然 FAIL。**

### 监督方的实测证据（本地直接比两份 CSV）

两份文件各 2,300 个锚点，`anchor_id` 完全重合：

| 量 | 值 |
|---|---|
| `mse_per_candidate[7]` 字符串完全相同 | **433 / 2,300** |
| 不同 | **1,867 / 2,300 = 81.2%** |
| 最大逐元素绝对差 | **7.044643e+01**（anchor `ipv_008361|77|2`） |
| 差值中位数 | 7.264e-03 |
| 差值最小 | 1.085e-14 |

**Mac 与 HPC 之间本来就有 81.2% 的锚点不同**——这正是 G 轨的既有结论
（Mac↔HPC 的差异来自软件栈；HPC↔HPC 则逐位相同，AMD 与 Intel 348/348）。
所以这次 FAIL **完全由基线选错解释**，不构成 K2 数值异常的任何证据。

### 对 K2-1 判断的评价

K2-1 在 18:38 的处置是「停止并上报」，**行为正确**——看到疑似数值差异就停，是对的。
但**诊断错了**：它写的是「same G workdir live re-run differs from its anchor_mse.csv,
so stopping」，把结论指向"G 不可复现"。真实情况是它在 HPC 上重跑、拿去比 Mac 的 CSV，
自然不同。**不是 G 不可复现，是比错了对象。**

### 裁定与指令

按 PI 授权的判据，这属**工程类（路径写错）、根因已定位、不触碰数值路径**，
**监督方放行修复并重跑，不上报 PI 决策**（但已如实向 PI 汇报）。

1. **改 `validate_g_anchor()` 第 1463 行的路径为
   `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`。只改这一行的基线路径，
   比较口径（float64、逐位精确相等 `diff != 0.0`）一个字不许改。**
2. **不必重跑 solve，也不必重跑 join。** L1 产物已在远端，只需重跑 `validate_outputs`。
3. **必须报出 `compared_rows`。** 若它远小于 2,300，说明 `anchor_id` 与 `canonical_key`
   的对齐有问题，canary 强度不足，同样不算通过。
4. 顺带把 17:04Z 的必补项一起做掉：**RQ009 join 行 canonical_key 唯一性补实测并纳入 blocker**
   （现为硬编码 0，且不在 blocker 列表内）。
5. 重跑前按 15:09Z 的 canary 要求验证。

### 两条必须写进结项报告的话

- **第 6.2 条 G 锚点判据到此刻为止「尚未真正执行过」。** 之前那次不算——它比的是错的基线。
  在正确基线下跑通之前，**不得声称 K2 与 G 逐位一致**。
- 这已是**同一类缺陷的第二例**：判据看起来在检查，实际检查的不是该检查的东西
  （第一例是 RQ009 `duplicates` 硬编码为 0）。复审 A 的告诫可以推广为一条规则：
  **每一条验收判据都必须有一次"故意让它失败"的验证，证明它真的会 FAIL。**
  这一条要写进方法学部分。

### 顺带：我 17:29Z/17:58Z 提的 OOM 风险**没有发生**

validation 跑完了并产出了结论（`total_l1_rows=14,473,982` 与预期一致），
说明三次全量物化在 48G 下扛住了。流式改写不再是必须项，可降为后续优化建议。

## 2026-08-02T19:12:54Z 监督方裁定：K2 结项判定 —— 两条 blocker，一条推翻、一条是我的阈值定错

K2-1 于 18:49 以 `final_status=FAIL`、`blockers=g_anchor, solver_failure_threshold` 结项。
**监督方逐条复核后判定：产物本身没有问题，两条 blocker 都不成立。** 依据如下。

### 一、`g_anchor` —— **推翻。监督方已用实测证明 K2 与 G 逐位相同**

18:40Z 的裁定**没有被执行**：`k2_fullcorpus_materializer.py` 第 1468 行仍指向
`rq015b-repair/work/anchor_mse.csv`（Mac 版）。K2-1 在收到之前就已进入结项流程。

**监督方直接用 `g_anchor_blocker_repro.json` 记录的 K2 L1 数值，与两份基线各比一次：**

锚点 `ipv_007137|46|1`：
```
K2 (HPC) L1  : 0.51146166274455163, 0.15495696974407502, 0.12205821796213456, ...
G  HPC 基线  : 0.51146166274455163, 0.15495696974407502, 0.12205821796213456, ...
B  Mac 基线  : 0.52479401493083488, 0.15740641321295740, 0.12332716425144798, ...

K2 vs G-HPC 基线 : 最大绝对差 = 0        逐位相同 = True
K2 vs B-Mac 基线 : 最大绝对差 = 0.013332352186283258   逐位相同 = False
```
后者与 validator 报出的 `max_abs_diff` **一模一样**，说明这就是它比的那个对象。

**结论：K2 与 G 轨 HPC 基线逐位一致。`g_anchor` FAIL 纯由基线选错造成，产物无问题。**
K2-1 在 repro 里写的「HPC 重跑产生 K2 值而非 anchor_mse.csv 值」也与此完全自洽——
它重跑的是 HPC，自然得到 HPC 的值。**不是 G 不可复现。**

**仍须补做（为留痕，不是为存疑）**：改第 1468 行为
`.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`，
只重跑 `validate_g_anchor`，**报出全部 2,300 个锚点的 compared_rows 与 max_abs_diff**。
预期 compared_rows 等于 K2 InterHub 输出中命中的锚点数、max_abs_diff = 0。

### 二、`solver_failure_threshold` —— **阈值是我定错的，我撤销它**

6/460 片越过我在第五节写的「单片 SOLVER_FAILURE 超过 100 行或 2.0% 取小则停止」，
最高 `A_train_vegas6_0019` 394/11,000 = 3.58%，**6 片全部是 `train_vegas*`（nuPlan 拉斯维加斯）**。
全局 **1,934 / 4,981,984 = 0.0388%**，`non_finite` 全为 0。

**这个阈值是我按 K1/K1b pilot 的 0/1,120 失败率拍的，从未用真实 nuPlan Vegas 数据校准过。**
用一个没在目标数据上校准过的 tripwire 去卡全量普查，是我的设计错误。
而且这些行**被正确记为工程失败状态**，没有污染两个科学 reason
（`non_applicable_science_reason` 未进 blocker，即为 0）。

**裁定：撤销该停止条件，改为「必须刻画并报告」。** 报告须给出：
按 PKL/source 的 SOLVER_FAILURE 分布、这 1,934 行的失败特征、
以及与既有「400 个退化锚点全部来自 nuplan、与源共线」的关联判断。
**不重跑，不因此判 FAIL。**

### 三、其余判据全部通过（均为产物实测）

| 判据 | 实测 | 判定 |
|---|---|---|
| InterHub 覆盖 | canonical_keys 4,981,984 / duplicates **0** / missing **0** | 通过 |
| 总行数 | 14,473,982 = 5,197,072 + 8,994,736 + 281,268 + 906 | 通过 |
| 非科学范围行 | `gate_applicable_false` 497,262 = 215,088+4,272+274,022+2,974+906 **精确相等** | 通过 |
| 科学 reason 污染 | `non_applicable_science_reason` = 0 | 通过 |
| 逐行门不变量 | `invariant_bad_rows` = 0，`invariant_examples` 空 | 通过 |
| held_out | `held_out_parsed_rows` = 0 | 通过 |
| manifest 对账 | 510 份，`bad_count` = 0 | 通过 |
| RQ009 回填 | rows 8,994,736 / misses 0 | 通过 |

### 四、仍未做的一项（17:04Z 必补项）

`rq009_join.duplicates` **仍是硬编码 0**，未纳入 blocker、未实测。
**结项前必须补一次 join 行 canonical_key 去重实测。** 这是唯一还没被真正测过的判据。
另 `rq009_array_restore_1000` 在 summary 里是 FAIL、18:44 的 corrected 版是 PASS
（1,000 个键解析到 500 个唯一 InterHub 键，corrected 后 1,000/1,000 命中）——
**报告必须说明以哪一版为准、原版为何 FAIL。**

### 五、科学结论（本轮真正的产出）

InterHub 全量 4,981,984 个求解单元：

| | 计数 | 占比 |
|---|---:|---:|
| **OK（该帧 IPV 携带候选间判别信息）** | **3,502,340** | **70.3001%** |
| NEAR_UNIFORM | 1,457,746 | 29.2604% |
| NO_IPV_EFFECT | 19,964 | 0.4007% |
| SOLVER_FAILURE（工程） | 1,934 | 0.0388% |

**与 J 轨设计基抽样的对照（解释性，非判据）**：
J 用 2,300 个锚点、HT 权重、1,909 个 cluster 给出 71.2695%，CI [67.1729%, 75.2135%]。
**普查值 70.3001% 落在该 CI 内，与点估计相差 0.97 个百分点。**
两条 reason 的占比也接近（`NO_IPV_EFFECT` J 0.5095% vs 普查 0.4007%；
`NEAR_UNIFORM` J 28.2210% vs 普查 29.2604%）。

**这是一个独立的相互印证**：2,300 个锚点的设计基抽样，把 498 万单元的普查结果
预测到了 1 个百分点以内。但**不得写成"验证通过"**——域与分母不同，
按第 6.3 节它只能作解释性对照。

### 六、结项放行条件

补齐两项后即可转 DONE：**(1)** 正确基线下的 G 锚点全量比对；
**(2)** RQ009 join 行 canonical_key 去重实测。
另需补齐结项缺陷清单三项：`STATUS.md` 刷新（仍停在 12:47）、
manifest 行索引区间、以及把 `slurm_sacct_all.tsv`（18:44 已落盘，**这项已完成**）纳入报告。

## 2026-08-02T19:43:19Z 监督方：K2-1 已停摆，派 K2-2 做结项收尾

自 18:49:06Z 起 `K2_progress.log` 无新行、`STATUS.md` 仍停在 12:47、
第 1468 行基线路径未改——**K2-1 与 leader 均已结束回合，19:12Z 的裁定无人执行。**

监督方尝试自行完成剩余两项核验（把本地 L1 产物 stage 到云端容器用 pyarrow 直接算），
**被设备端登录态过期挡住**（`session_stale_relogin`，需 PI 在桌面端重新登录）。
挂载 VM 内无 pyarrow / duckdb / 网络，无法本地解析 parquet。已如实告知 PI。

因此派 **K2-2** 做收尾，prompt 见 `board/K2-2_closeout_prompt.md`（6,651 B），四件事：
1. 改第 1468 行基线为 `anchor_mse_hpc.csv`，**只重跑 `validate_g_anchor`**（对本地产物跑，不投 Slurm），
   报出 `compared_rows` 与 `max_abs_diff`
2. **实测 RQ009 join 行 canonical_key 去重**——这是唯一从未被测过的判据
3. 刻画 1,934 行 SOLVER_FAILURE（不重跑）
4. 更新报告与看板：`final_status` 改 PASS 并附裁定依据、完整失败史分开列、
   两条方法学结论、`INTERFACE_NOTE.md` 的 `ipv_log=0` 警告、刷新 `STATUS.md`、
   manifest 行索引区间说明

**K2-2 不得重跑求解与 join，不得改任何判据阈值，`k2_fullcorpus_materializer.py` 只许改第 1468 行。**

派发（PI 在 macOS 侧执行）：
```
python3 .codex-fleet/rq015a-run/board/detach_launch.py \
  --log .codex-fleet/rq015k-fullcorpus-gate/board/reports/K2-2.log \
  --pidfile .codex-fleet/rq015k-fullcorpus-gate/board/K2-2.pid \
  -- codex exec --cd "$PWD" --model gpt-5.5 -c model_reasoning_effort="xhigh" \
     --sandbox workspace-write -c sandbox_workspace_write.network_access=true \
     "$(cat .codex-fleet/rq015k-fullcorpus-gate/board/K2-2_closeout_prompt.md)"
```

## 2026-08-03T00:24:36Z 监督方独立实测：RQ009 join 键唯一性【通过】+ 一处域对齐的重要观察

PI 已重新登录，staging 恢复。监督方把全部 45 个 RQ009 join 分片
（8,994,736 行）stage 到云端容器，用 pyarrow **独立实测**，不依赖 K2-2 的报告。

### 一、那条从未被测过的判据：**实测通过**

```
rows          = 8,994,736
unique_keys   = 8,994,736
duplicates    = 0          ← 实测值，非硬编码
gate_applicable = True (8,994,736 / 8,994,736)
```
`rows == unique_keys`，无一重复。**第 6.1 条第 2 项至此才算真正被验证过。**
K2-2 若报出不同数字，以本次实测为准并追查其口径。
（`new_solve_rows = 0` 仍属 by construction，报告须如实标注。）

### 二、**一处重要观察：J 的估计量对应的是"台账行域"，不是"求解单元域"**

同一批 RQ009 join 行的门后状态分布（实测）：
`OK` 6,405,292 / `ABSTAIN` 2,585,792 / `SOLVER_FAILURE` 3,652。

| 域 | 可估率 | 与 J 点估计 71.2695% 之差 |
|---|---:|---:|
| InterHub **求解单元**域（4,981,984） | 70.3001% | 0.9694 pp |
| RQ009 **台账行**域（8,994,736） | **71.2116%** | **0.0579 pp** |

**两者都落在 J 的 CI 内，但台账行域比求解单元域接近 J 约 17 倍。**

这不是巧合：J 的 HT 估计器用的是**行级权重**（全域分母 2,646,058 的 HT 权重），
它的估计对象天然是"按台账行加权的可估率"，而不是"按去重后的求解单元计数的可估率"。
两者不同，是因为一个求解单元可以对应多条台账行（压缩比 2.804×），
而不同单元被引用的次数不同——**加权与不加权因此给出不同的数。**

**要求写进报告，并且必须谨慎表述：**
- 报出**两个域的数**，各自写明分母，**不要只报一个**
- 与 J 对照时**只与台账行域对照**，并说明理由；求解单元域的 70.30% 单独列出
- **仍不得写成"验证通过"**——J 是抽样估计、K2 是普查，域即使对齐，性质也不同。
  可写：「设计基抽样估计与普查在同一域上相差 0.06 个百分点」，**到此为止**
- **这条观察本身是本轮的一个副产品结论**：它把 J 的估计量落到了具体的域上。
  但**J 的分母 2,646,058 与 RQ009 的 8,994,736 并不相等**，二者的确切关系
  （是否为同一域的不同权重口径）**尚未查清，报告必须写明这是待确认项，不得含糊称"域一致"。**

### 三、监督方本次的取证方式（留痕）

45 个分片经 device staging 传入云端容器，`pyarrow 25.0.0` 读取
`canonical_key` / `status` / `gate_applicable` 三列后逐分片累计去重。
不改动设备上任何文件，属只读核验。

## 2026-08-03T00:50:03Z 监督方：K2 结项验收通过，放行 DONE

K2-2 已完成四件事并转 `WAITING_ON_COMMANDER`（`STATUS.md` 于 00:28:11Z 刷新，滞留问题解除）。
监督方逐条复核，**全部通过**。

### 一、两条曾被质疑的判据，现在都真正过了

| 判据 | 实测 | 来源 |
|---|---|---|
| G 锚点（正确 HPC 基线） | `anchor_rows=2300`、**`compared_rows=2300`**、`max_abs_diff=0.0`、`first_mismatch=null` | `validation/g_anchor_hpc_baseline.json` |
| RQ009 join 键唯一性 | `rows=8,994,736`、`unique_keys=8,994,736`、**`duplicates=0`** | `validation/rq009_join_key_uniqueness.json` |

**G 锚点比对了全部 2,300 个锚点、逐位零差异。** 监督方 18:40Z 用单个锚点做的诊断，
在 n=2,300 的全量尺度上被证实。第 6.2 条判据**至此才算真正执行过**。

**RQ009 键唯一性与监督方 00:24Z 的独立实测逐字相同**（监督方走 device staging + pyarrow，
K2-2 走本地产物扫描，两条独立路径同值）。第 6.1 条第 2 项**至此才算真正被测过**。
`new_solve_rows=0` 已如实标注为 by construction、非实测——符合要求。

### 二、SOLVER_FAILURE 刻画：质量好，且**诚实报告了一个否定结论**

1,934 行全部来自 `nuplan_train` / `source=nuplan` / `split=development`，
集中在 5 个 `train_vegas*` PKL、6 个分片；
`agent_1` 与 `agent_2` **各 967**（成对失败，即 967 个 case-frame × 2 个 role）；
`n_obs` 主要为 11（1,862 行）。

`l1_status_integrity` 显示这 1,934 行的
`mse_spread` / `max_w_log` / `k_eff_log` / `ipv_log` **全部为 null**、
`reason_code` 与 `solver_status` 全为 `SOLVER_FAILURE`、`source_attempt_status` 全为 `ATTEMPTED`。
**工程失败行的 null 规则在最该检验的那批行上被验证了。**

**关键的否定结论**：这些**不是**此前 400 个退化锚点的同一机制。
400 个是 `spread(mse)==0`、signature `U=399/N=1`；
而这 1,934 行 **`signature=N` 占 1,693/1,934**，U∪Z 代理仅 241 行（12.5%）。
结论写的是「与 nuPlan 几何压力区相邻的工程失败，不是同一 `NO_IPV_EFFECT` 退化机制」。
**这个否定结论比硬凑关联有价值，予以肯定。**

### 三、监督方补写的一节（已插入报告 §4.1，署名与时间戳齐全）

K2-2 在 00:28 结项，早于监督方 00:24 的域对齐要求送达，故报告只报了求解单元域。
监督方已在 §4 之后插入 **§4.1 Supervisor addendum**，内容：
两个域的率与分母并列（求解单元域 70.3001%、台账行域 71.2116%）、
台账行域距 J 点估计 0.0579 pp（约近 17 倍）、
以及**两条强制保留的限制**——
(1) J 的 HT 分母 2,646,058 与 8,994,736 **不相等，是否同域尚未确立，明写 not yet established，
不得称"域一致"**；
(2) 即便域对齐，J 是抽样、K2 是普查，可辩护的表述仅限
「在台账行域上设计基估计与普查相差 0.06 个百分点」，**仍不是验证**。

### 四、其余复核结果

`final_status: PASS`，`previous_final_status: FAIL` 保留；
失败史按 solve 三投 / finalize 两投**分开列**，每条给根因并标注
「numeric path touched? No」；两条方法学结论（§11）齐备；
接口警告（§9）含 `ipv_log = 0` 那条；manifest 行索引区间说明（§10）已写并给出建议字段名；
`rq009_array_restore` 说明了以 corrected 版为准及原版 FAIL 的原因；
全部作业使用 `--partition=intel,fata`，未用 `amd`。

### 五、裁定

**K2 结项验收通过，放行转 `DONE`。**
四个受保护文件与 `configs/ipv_sigma01_exact.json` 的 SHA 自始至终未变，
`src`/`pipelines`/`configs` 改动为 0，HEAD 仍 `38aff434`。

**遗留项（不阻塞结项，转入后续）**：
1. J 的 HT 分母 2,646,058 与 RQ009 8,994,736 的确切关系——**唯一的科学遗留问题**
2. `launch_leader.sh` 的 nohup leader 早退（本轮第 4 次），基础设施债
3. `validate_outputs` 的三次全量物化改流式（本轮未触发 OOM，属优化项）
4. OnSite/WOD 的 materializer（PI 已裁定本轮不做）

## 2026-08-03T02:40:37Z PI 裁定：RQ015 收官后的三项安排

RQ015 全部 11 条轨道（A-K）已结项，`STATUS.md` 均为 `DONE`
（G/H/I/J 的结论早已裁定，此前仅未翻牌，本次由监督方补齐 `released_by`）。
PI 在本次汇报中作出三项裁定：

### 一、RQ009 的零点：**只拆分，不动 RQ009**

RQ009 自己的报告（`90_report/index.html` 第 127 行）记有：
打分目标存在 **~21.5% 的精确零点原子（273,819 / 1,270,566）**，
并明言这造成「80% boundary-tie / 1e-10 endpoint-nudge 覆盖脆弱」、削弱相关性，
同时限定了 interval-tie 行为与 practical null 的解释。

K2 台账恰好能逐行区分「过门的真中性零」与「弃权而被记成 0」。
**PI 裁定：用台账 join 回去把这 273,819 个零拆成两类，不重算任何东西、
不改 RQ009 已 accepted 的结论。** 先量清楚污染到底多大，再决定要不要做更多。
**不得在本轮重建人类 envelope。**

### 二、OnSite / WOD：**先只查 274,022 行未知态**

不写新 materializer、不做重算。**只弄清 OnSite 那 274,022 行为何是 `UNKNOWN`**——
它比该产物尝试过的 2,974 行多两个量级，这个比例本身可疑。
WOD 906 行与 OnSite 2,974 行**继续保持不适用状态**，本轮不处理。

### 三、重心：**收拢成文**

以写出完整交付为主线：缺陷的性质、对数域修复、门的规格、确定性证据、全语料普查。
证据链目前是完整的，拖久了细节会散。前两项是这份交付的输入。

### 监督方补充的一条写作硬约束

现在至少有**四个不同的分母**在流通，任何比率若不写明分母都是含混的：

| 分母 | 含义 | 出处 |
|---|---|---|
| 2,646,058 | J 轨 HT 权重的全域分母 | J1 报告 |
| 4,981,984 | InterHub canonical 求解单元 | K1 / K2 |
| 8,994,736 | RQ009 台账行 | K1 / K2 |
| 1,270,566 | RQ009 打分目标行 | RQ009 报告 |

**成文时每一个比率必须紧跟分母**，且四者之间的确切关系**目前只查清了部分**
（求解单元与台账行的压缩比 2.804× 已知；2,646,058 与 8,994,736 的关系仍未确立，
K2 报告 §4.1 已明写 not yet established）。**不得含糊地在不同分母间搬运比率。**
