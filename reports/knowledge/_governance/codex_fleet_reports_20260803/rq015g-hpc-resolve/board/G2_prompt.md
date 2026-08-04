# G2 — 跨节点确定性闸门：在 fata02 上重算 2022477 的同一个 case，与 cpui158 的结果逐位比对

你是 track G 的第二个执行 agent（G2）。仓库根：
`.`

**你有网络**，`ssh -o BatchMode=yes tongji-hpc` 可用。
**若出现任何密码提示：立刻停下并上报，不要输入/存储/打印密码。**

---

## 为什么做这一件事（只做这一件）

Track G 已在受管 HPC 冻结环境把 2,300 个冻结锚点重解完毕（Slurm `2023332`，COMPLETED）。
但它跑在 **`fata` 分区的 `fata02`**，而 PI 当初证明 sigma01 严格可复现（max diff `1.11e-16`）
的作业 `2022476` / `2022477` 跑在 **`intel` 分区的 `cpui158`**。

leader 已查明两者**微架构不同**：
```
fata02   AMD EPYC 9654 96-Core Processor   avx2=yes avx512f=yes fma=yes   CoresPerSocket=96
cpui158  CoresPerSocket=48, RealMemory=644000                             （非 96 核 EPYC）
```

因此必须回答：**fata02 与 cpui158 对这套计算是否属于同一数值等价类？**
若不是，"HPC 的答案"这个说法本身就没有定义，本轮 2,300 锚点交付物失去意义。

`intel` 分区当前 184/232 节点 offline，新作业预计 `2026-08-01T21:13Z` 才起跑，等不起。
**但我们不需要等**——`2022477` 已经在 cpui158 上把真实 case `ipv_000001` 算完并留下了产物。
所以：**在 fata02 上用完全相同的输入与冻结环境重算同一个 case，与 2022477 的产物逐位比对。**
这就是跨节点确定性探针，只是拿现成的 intel 侧结果当参照，不必重排队。

---

## 参照作业 2022477 的完整配置（照抄，不要改）

工作目录：
```
/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/sigma01_onecase_20260801T0710CST
├── input/code_parity_sample_cases.csv      12 行，取 limit=1 即 ipv_000001
├── input/pkl/train_vegas3.pkl
├── logs/zxc-sigma01-onecase_2022477.{out,err}
└── output/
    ├── cases/nuplan_train/train_vegas3/scenario_2400/row_00000_3131c9ba61a9/data/ipv_results.xlsx   ← 参照产物
    ├── cases/.../data/metadata.json
    ├── code_parity_sample_cases_with_ipv_limit.csv
    └── processing_summary_limit.json
```

`2022477` 的 `.out` 里记录的运行参数（**逐项照抄**）：
```
workers=1              solver_mode=exact        solver_preset=null
mp_start_method=auto   case_timeout_seconds=600
reference_clip_margin_m=60.0   reference_max_points=40   reference_smooth_points=40
limit=1                save_plots=false         dataset_filter=[]
execution_profile: <repo>/configs/ipv_sigma01_exact.json
                   sha256=3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522
结果：status=ok，87 帧行 / 348 个 IPV-or-error 值，对存档 max diff 1.11e-16
```

代码与环境：
```
managed checkout  /share/home/u25310231/ZXC/sociality_estimation/code/repo  @ 6bdcc2e6
conda env         /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01   Python 3.9.24
入口              pipelines/interhub/process_interhub.py
```

---

## 你要做的

### 第 1 步 — 建新工作目录并复制输入
```
新目录（自己建，不得覆盖 2022477 的任何东西）：
/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_fata_crossnode_<UTC戳>/
```
把 `2022477` 的 `input/`（CSV + pkl）**原样复制**过来（复制，不是软链，避免误改只读源）。
复制后校验 CSV 与 pkl 的 SHA-256 与源一致。

### 第 2 步 — 提交到 fata02
写一个 sbatch，**除分区/节点/日志路径外，一切与 2022477 相同**：
```
#SBATCH --job-name=zxc-rq015g-crossnode-fata     ← zxc- 前缀必须
#SBATCH --partition=fata
#SBATCH --nodelist=fata02                         ← 必须钉在 fata02，与 2023332 同节点
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                         ← 与 2022477 的 NCPUS=1 一致
#SBATCH --time=00:30:00
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
用 `envs/ipv-exact-sigma01/bin/python` 跑 `process_interhub.py`，参数逐项对齐上面那张表
（`--workers 1`、`--limit 1`、reference `60/40/40`、`solver_mode exact` 等；
具体 CLI 参数名以 `code/repo` 里 `process_interhub.py --help` 为准）。
**重计算只走 sbatch，不得在登录节点跑。**

`fata` 分区当前有空闲节点，应该很快起跑。提交后**在本回合内轮询**
`squeue -j <id>` / `sacct -j <id>`，每 5 分钟往
`.codex-fleet/rq015g-hpc-resolve/board/progress.log` 追加一行
（格式 `<UTC> | <阶段> | 做了什么 | 结论`，时间戳用 `date -u +%Y-%m-%dT%H:%M:%SZ`）。

### 第 3 步 — 逐位比对（本任务的全部产出）
把 fata02 的 `ipv_results.xlsx` 与 `2022477` 的对应文件逐值比对：
- 对齐 key 与 timestamp，确认行数同为 **87**、值个数同为 **348**
- 报 `max|Δ|`、`mean|Δ|`、以及**逐位相同的值个数 / 348**
- 同时比对 `code_parity_sample_cases_with_ipv_limit.csv` 的 IPV 相关列

**判据**（直接照抄进结论）：
```
max|Δ| == 0（逐位相同）      → fata02 与 cpui158 对本计算逐位确定，跨节点无差异
max|Δ| ≤ 1e-15              → 同一数值等价类，本轮 2,300 锚点结果直接采信
max|Δ| 显著大于 1e-15        → 分区切换本身引入数值差异，【立刻停下报 leader】，
                              不要自行扩大规模、不要换第三个分区
```

### 第 4 步 — 报告
写 `.codex-fleet/rq015g-hpc-resolve/board/reports/G2_crossnode_gate.md`，包含：
Slurm 作业号与节点、输入 SHA 校验、运行参数逐项对照表（fata02 vs 2022477）、
环境指纹（`sys.executable`、Python/numpy/scipy 版本、线程环境变量）、
比对结果（max/mean/逐位相同计数）、以及按上面判据给出的**明确判定**。

---

## 硬边界

```
□ 【绝不覆盖】2022477 / 2022476 的工作目录、sigma01 任何冻结产物、managed checkout、
   RQ009/RQ015A 的 run 目录。HPC 侧只写你新建的 rq015g_fata_crossnode_<UTC戳>/
   本地只写 .codex-fleet/rq015g-hpc-resolve/{work,board}/ 下的新文件
   【不得修改】.codex-fleet/rq015b-repair/ 与 .codex-fleet/rq015d-sigma-rederive/ 下任何文件
□ 必须用冻结环境 envs/ipv-exact-sigma01 (Python 3.9.24)；改了环境本次结论作废
□ 不改算法、不改估计器；src/sociality_estimation/core/agent.py 一字不动
□ 【致盲纪律】input CSV 含 RQ014 致盲相关字段
   （intensity / priority_label / turn_label / path_category / path_relation / actual_order 等）。
   它们只作为管线输入原样传递，**你不得读取、不得统计、不得在报告里引用这些列的取值**
□ RQ007 held_out 不得被解析
□ 描述性，不作因果主张；只给证据不给建议
□ 全文禁用 `estimability` 与"测出/未测出 IPV"
□ 三条 track 并发在同一工作区，铁律：
   禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
   禁止 git commit；工作区非空是【预期状态】，你只对自己创建的文件负责
□ 本地解释器钉死 <local-rq009-venv>/bin/python
□ 时间戳一律 date -u +%Y-%m-%dT%H:%M:%SZ，不要前瞻估计
```

**只做这一件事。** 不要顺带重跑 2,300 锚点、不要改 σ 分析、不要写第二版规格。
