# G1 — 在冻结 HPC 环境里重解 2,300 个锚点，并与 Mac 版逐锚点对照

你是 track G 的唯一执行 agent（G1）。仓库根：
`.`

**一句话任务**：RQ015B 的 T5 在 macOS 上把 2,300 个冻结锚点解了一遍；macOS 已被证明
不是严格数值复现环境（同一 case 对存档 max diff = 1.1244582，与网格端点 3π/8=1.178 同量级）。
把**同一批锚点、同一份代码、同一组参数**放到已证可复现的受管 HPC 冻结环境里再解一遍，
然后逐锚点对照，给出修正后的 D1/D2 机制拆分与 σ 扫描数字。

**这是描述性/诊断性产出。一轮做完，自查一遍数值健康与覆盖，出报告。不做盲审、不出第二版规格。**

---

## 0. leader 已完成的 preflight（**直接采信，不要重做，但运行时仍要按第 3 节校验 SHA**）

leader 已经跑过下面这些检查，结论全部为绿。你**不需要**再去摸索仓库结构或到处 find：

**(a) 代码在两侧逐位相同** —— 这是本轮实验成立的关键：本轮唯一变量是**环境**，不是代码。

| 文件 | 本地 SHA-256 | HPC `code/repo` @ `6bdcc2e6` |
|---|---|---|
| `src/sociality_estimation/core/agent.py` | `bde0f58258e915feb90eeb89d716632db95051d5b9d0a98abe9898cacd1da9f7` | **相同** |
| `src/sociality_estimation/core/ipv_estimation.py` | `e2c84e62fe35668912d09f76dc5c076caa2913cb10d95add473ed4def96f30b4` | **相同** |
| `pipelines/interhub/process_interhub.py` | `2010433b6ed72a85f45d0fdc5ad1e6414e5113605f1e0f65f9cb7d4cf784fe8b` | **相同** |
| `src/sociality_estimation/core/reliability_logdomain.py` | `8f740677eb2c3cfd0cba7e9785db9b1fba5cd4a40c0f6e0584bab5747eb8f830` | **HPC 侧不存在**（RQ015B 新增文件），**必须由你 stage 过去** |

**(b) HPC 上有本轮需要的全部 PKL，且体积与本地逐字节一致**

HPC PKL 根目录（**用这个**）：
```
/share/home/u25310231/ZXC/sociality_estimation/data/interhub/snapshots/interhub_legacy_20260711_v1/full_datasets/pkl
```
（它是指向 `batches/20260611_fullset_param_rerun/pkl` 的符号链接，只读。）

sample_v1 需要 9 个 PKL，每个的 HPC 体积都与本地相同：
`train_singapore.pkl`(4,967,922, 5 锚点) `train_vegas1.pkl`(10,725,804, 23)
`train_vegas2.pkl`(93,647,450, 219) `train_vegas3.pkl`(116,669,298, 275)
`train_vegas4.pkl`(66,704,310, 154) `train_vegas5.pkl`(98,288,174, 236)
`train_vegas6.pkl`(100,851,877, 238) `waymo_0-299.pkl`(310,197,719, 676)
`waymo_800-999.pkl`(206,783,180, 474)

⚠ **不要**用 `.../interhub_legacy_20260711_v1/subsets_for_yiru/pkl`——那是更小的 legacy 子集，
缺 vegas4/5/6 与 waymo_800-999，且同名文件体积不同。用错了本轮作废。

ℹ 本地 `waymo_300-499.pkl` 是 truncated 的，但 **sample_v1 一个锚点都不用它**，
所以本地那个缺陷不影响本轮。HPC 侧该文件反而是完整的。

**(c) 已证可用的受管通道**
```
managed checkout   /share/home/u25310231/ZXC/sociality_estimation/code/repo  @ 6bdcc2e64bacd75d02741aa18ef5d61eef5a2962
conda env          /share/home/u25310231/ZXC/sociality_estimation/envs/ipv-exact-sigma01   (Python 3.9.24)
execution profile  <repo>/configs/ipv_sigma01_exact.json  sha256=3add56c2785c4b11cdb5baf75e2505fe3ebb49c407c9f7f7c226652ca1e78522
先例作业           Slurm 2022476 zxc-sigma01-fixture  (夹具, max diff 4.44e-16)
                   Slurm 2022477 zxc-sigma01-onecase (真实 ipv_000001, max diff 1.11e-16)
先例作业配置       partition=intel, NCPUS=1, workers=1, solver_mode=exact, 87 帧耗时 212 s
先例作业目录       /share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/sigma01_onecase_20260801T0710CST
```
SSH 别名 `tongji-hpc`。**重计算只能走 `sbatch`，不得在登录节点跑。**
作业名必须以 `zxc-` 开头。**若出现任何密码提示：立刻停下并上报，不要输入/存储/打印密码。**

---

## 1. 输入（冻结，**不得重抽样**）

```
.codex-fleet/rq015b-repair/work/sample_v1.csv       2,300 锚点；SHA-256 已由 leader 现场核过 = 
                                                     d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e
.codex-fleet/rq015b-repair/work/sample_v1.sha256    校验文件
.codex-fleet/rq015b-repair/board/sampling_contract_v1.md   抽样合同（已冻结，不得出 v2）
.codex-fleet/rq015b-repair/work/anchor_mse.csv      【Mac 对照基线】sha256=b0f6202501ea738b1ae6d49f83af1877bee85b391d5db6a44375d67b552eb114
.codex-fleet/rq015b-repair/work/b2_summary.json     Mac 版 B2 汇总（D 拆分、bootstrap、health 都在里面）
.codex-fleet/rq015b-repair/work/run_b1_rq015b.py    含 diagnostic_for_anchor / legacy_weights_from_rel_dis
.codex-fleet/rq015b-repair/work/run_b2_rq015b.py    含 solve_anchor_task / 机制分类 / 加权 / bootstrap
.codex-fleet/rq015d-sigma-rederive/work/d1_sigma_analysis.py   D 轨 σ 扫描脚本（INPUT_CSV 硬编码在第 17 行）
.codex-fleet/rq015d-sigma-rederive/work/d1_sigma_stats.json    D 轨 Mac 基线数字
.codex-fleet/rq015d-sigma-rederive/board/reports/D1_sigma_report.md
```

**只读上面这些路径。不要对 `reports/` 做全仓库 `rg`**——宽泛检索会把 RQ003
`12_blind_annotation/controlled_identity_map.csv` 的 controlled-access 行整行拉进上下文。

Mac 基线关键数字（供你对照，取自 `b2_summary.json`）：
```
weighted_main/proportions   D1=0.43010748889125155  D2=0.3948001734777231
                            D3=0.0  D4=0.0  OK=0.17509233763101836
bootstrap CI                D1 [0.3935451234499348, 0.46826223927974564]
                            D2 [0.3568659082008827, 0.4307511112323502]   B=2000 seed=20260731 clusters=1459
health                      rows=2300  solve_errors=0  nonfinite_rows=0
                            legacy_fallback_total=603  legacy_fallback_non_U_count=0
                            min_mse p0=0.0  p50=0.0551034557  p100=655.5329262812
t5                          anchors=2300 workers=6 executor=thread elapsed=1240.3s
                            serial_check_n=24  serial_check_max_abs_diff=0.0
parity                      eligible_count=1526  eligible_max=3.747002708109903e-15  pass_1e_minus_12=True
```
D 轨 Mac 基线（取自 `d1_sigma_stats.json`，σ=0.1）：
```
frac_near_uniform_log=0.5317   frac_hard_argmax_log=0.1287
frac_near_uniform_legacy=0.753 frac_hard_argmax_legacy=0.0217
k_eff_log_mean=5.111 median=6.795 ; k_eff_legacy_mean=6.219 median=7.0
legacy_fallback_triggered_true=603  partial_underflow_true=171
```

---

## 2. 实验设计（**已定，照做**）

> 保持**代码不变、样本不变、参数不变**，只让**环境**从 macOS 换成冻结的 Linux/SciPy/BLAS ABI。
> 因此：**不要重写估计器，不要"顺手改进"数值实现**。你要 stage 的是与 Mac 跑的那份逐位相同的代码。

具体做法：在 HPC 上搭一个**仓库形状的 staging 树**，让 `run_b1_rq015b.py` / `run_b2_rq015b.py`
不改一个字节就能跑（它们用 `ROOT = Path(__file__).resolve().parents[3]` 定位仓库根）：

```
<HPC_WORKDIR>/repo_stage/
  ├── src/sociality_estimation/...                 ← 从本地 rsync（含 reliability_logdomain.py）
  ├── pipelines/interhub/process_interhub.py       ← 从本地 rsync
  ├── configs/                                     ← 从本地 rsync
  ├── data/interhub/raw/full_datasets/pkl  ──符号链接──►  上面 (b) 的 HPC PKL 根目录
  └── .codex-fleet/rq015b-repair/work/
        ├── run_b1_rq015b.py      ← 逐位复制，不改
        ├── run_b2_rq015b.py      ← 逐位复制，不改
        ├── sample_v1.csv         ← 逐位复制
        ├── sample_v1.sha256
        └── run_g1_hpc.py         ← 【你写的唯一新代码】薄驱动
```

`run_g1_hpc.py` 必须：
- `from run_b2_rq015b import solve_anchor_task, ANCHOR_FIELDNAMES, load_sample, ...`
  —— **直接复用 Mac 那份 `solve_anchor_task` 函数对象**，这样"数学完全一致"是结构上保证的，
  而不是靠你手抄。（`run_b2` 的自检闸门在 `main()` 里，import 不会触发；若 import 真的触发了
  任何 HEAD 校验，**不要去改 `run_b2`**，改为在驱动里绕过 HEAD 检查并在报告里写明。）
- 输出 `anchor_mse_hpc.csv`，**列名与 Mac 版 `anchor_mse.csv` 完全一致、顺序一致**
  （即 `ANCHOR_FIELDNAMES`，共 36 列）。这是能自动比对的前提，一列都不许改名或增删。
- 记录：本机 HEAD（当前是 `511b936c`，与 `run_b2` 里写死的 `EXPECTED_HEAD=e82091ce` 不同，
  **这是预期的，不是错误**——本轮的不变量是**文件 SHA**，不是 HEAD）、
  `agent.py` / `ipv_estimation.py` / `reliability_logdomain.py` 的实际 SHA-256、
  `sample_v1.csv` 的 SHA-256、Python/NumPy/SciPy 版本与 BLAS 信息（`numpy.show_config()`）、
  每个用到的 PKL 的 SHA-256。

**并行与数值稳定性（重要）**：
- 先例作业 2022477 是 `NCPUS=1 / workers=1`。为避免 BLAS 多线程引入数值差异，
  每个 worker 进程必须设 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。
- 用**进程级**并行（锚点之间互相独立）。申请 16–32 核，`--job-name=zxc-rq015g-anchor-resolve`。
- **必须做串行交叉校验**：另取 ≥24 个代表性锚点，用单进程串行再解一遍，
  要求与并行结果 `max_abs_diff == 0.0`（Mac 版 `serial_check_n=24` 就是 `0.0`）。
  **若不是 0.0，如实报告，不要掩盖**——那说明并行本身在动数值，结论要相应降级。
- 预算参考：Mac 约 3.2 CPU-s/锚点 × 2300 ≈ 2 CPU-小时。walltime 给 4 小时足够富余。

---

## 3. 运行时必须校验（任一条不过 ⇒ 停下并在报告里写明，不要将就）

```
□ sample_v1.csv SHA-256 == d27f10907b7ca8da5815a6b832859d64a40b7fbf41aa0e5587c51bec8466759e
□ staging 后 HPC 侧 agent.py / ipv_estimation.py / reliability_logdomain.py / process_interhub.py
  的 SHA-256 == 第 0(a) 节表格里的本地值（逐位相同才算数）
□ 9 个 PKL 的 SHA-256：HPC 侧 == 本地侧（**这是"同一份输入数据"的唯一证明**）
□ 实际使用的 Python 来自 envs/ipv-exact-sigma01 且版本为 3.9.24
  —— 这就是本轮的全部意义；**若因任何原因用了别的环境，本轮结论作废，如实上报**
□ 解出的行数 == 2300，solve_errors == 0，与 sample_v1 的 anchor_id 集合完全一致
□ 所有 split ∈ {development, guard}；held_out 行数 == 0
```

---

## 4. 交付物

写到 `.codex-fleet/rq015g-hpc-resolve/work/` 与 `board/reports/`：

**4.1 `work/anchor_mse_hpc.csv`** —— 36 列，列名/顺序与 Mac 版完全一致。

**4.2 `work/g1_compare.json` + 报告中的对照表** —— Mac vs HPC 逐锚点对照，至少给：
- `min_mse`、`min_rms`、`ipv_log`、`ipv_legacy`、`ipv_error_log`、`ipv_error_legacy`
  的差异分布：**max / p99 / median**（绝对差；再给相对差更好）
- **`argmin_candidate` 变了多少个锚点**（直接计数 + 占比 + 变化方向的交叉表）
  ← "落到不同候选"的直接证据
- **`legacy_fallback_triggered` 翻转了多少**：分别给
  「Mac 触发 / HPC 不触发」与「Mac 不触发 / HPC 触发」两个方向的计数
  ← **这一条直接决定 D1 被高估了多少，是本轮最重要的单个数字**
- 上述全部**分源给**（waymo / nuplan），另按 `signature`(U/Z/N) 与 `n_band`(FULL/RAMP) 给分组

**4.3 在 HPC 数字上重算，与 Mac 版并列呈现**：
- **D0–D4 机制拆分**：复用 `run_b2_rq015b.py` 里的 `classify_mechanism` /
  `threshold_and_classify` / `weighted_counts` / `bootstrap_ci` / `group_table` /
  `sensitivity_summary`（同样 B=2000、seed=20260731），保证口径与 Mac 版一致。
  **必须分源（waymo / nuplan）**；**合并值不得单独呈现**——要给必须与分源值同表并列。
  Mac 版的 waymo 58.73% / nuplan 1.06% 是本轮要修正的核心对象。
- **σ 扫描**：把 `d1_sigma_analysis.py` 复制为 `work/d1_sigma_analysis_hpc.py`，
  **只改第 17 行 `INPUT_CSV` 指向 `anchor_mse_hpc.csv`**，其余不动。
  至少报告 **σ = 0.02 / 0.1 / 0.2347** 三个点，看
  ① 59% 地板是否仍成立 ② 两条曲线的反向单调是否仍成立。
- **那 400 行 `spread(mse) == 0` 的退化锚点**是否仍然逐位相同。
  （先在 Mac 版里把这批锚点按 `spread(mse)==0` 定义出来并报出实际条数，
  再看它们在 HPC 侧是否仍 `spread==0` 且与 Mac 逐位相同。
  **leader 预期成立**——无交互时目标退化为 `cos(ipv)·interior + 常数`，正标量不改极小点，
  是解析性质。**若不成立，这是重点发现，要专门解释，不要一笔带过。**）

**4.4 判定表** —— 对下列每条结论给 **存活 / 数值需更新 / 结论改变** 三选一 + 支撑数字：

| # | 结论 | Mac 数字 |
|---|---|---|
| B2-1 | 平价门：log 域 ≡ 连乘 | 3.75e-15 |
| B2-2 | log 域下兜底不可达 | 解析 + 极端输入 + pytest |
| B2-3 | 400 行 MSE 逐位相同（无交互退化） | — |
| B2-4 | D1/D2 机制拆分 | D1 43.01% / D2 39.48% / OK 17.51%；waymo 58.73%、nuplan 1.06% |
| D-1 | 近均匀占比 | 53.17% (log) |
| D-2 | 硬 argmax 占比 | 12.87% (log) |
| D-3 | σ 扫描 59% 地板 | 59% |
| D-4 | 两曲线反向单调 | — |

B2-1 / B2-2 / B2-3 按 leader 的判断**应当不受影响**（它们比的是同一组 MSE 上的两种权重公式，
或是解析性质）；你要做的是**在 HPC 数字上确认这一点**，而不是默认它成立。

**4.5 报告 `board/reports/G1_hpc_resolve_report.md`** —— 包含：
运行环境与 Slurm 作业号、全部 SHA 校验结果、4.2/4.3/4.4 的表、
数值健康自查（行数、nonfinite、solve_errors、覆盖、配额是否仍为 U300/Z150/N125×...）、
以及 `held_out_parsed_rows = 0` 的**结构佐证**（不是一句声明，要给出你是怎么保证的：
sample 的 split 分布、代码里的 `ALLOWED_SPLITS` 闸门、你读过的文件清单）。

---

## 5. 硬边界

```
□ 【绝不覆盖】sigma01 任何冻结产物、pinned legacy checkout、RQ009/RQ015A 的 run 目录
   HPC 侧只写【新建的】 work_dir：
   /share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015g_anchor_resolve_<UTC时间戳>/
   本地只写 .codex-fleet/rq015g-hpc-resolve/{work,board}/ 下的新文件
   【不得修改】.codex-fleet/rq015b-repair/ 与 .codex-fleet/rq015d-sigma-rederive/ 下的任何文件
□ dev+guard only；held_out 封条守住
□ 不重抽样、不出合同 v2、不改估计器算法
□ 不接线生产：src/sociality_estimation/core/agent.py 一字不动
□ 必须用冻结环境 envs/ipv-exact-sigma01 (Python 3.9.24)
□ 描述性，不作因果主张；只给证据不给建议
   （"要不要申请全量重跑"是 PI 的决定，你给的是修正后的 D1/D2 数字）
□ 全文禁用 `estimability` 与"测出/未测出 IPV"。
   可辩护表述：**权重近均匀 ⇒ 该 IPV 数值不携带候选间的判别信息**
□ 三条 track 并发在同一工作区，铁律：
   禁止 git checkout -- . / git restore . / git stash / git reset --hard / git clean -fd
   禁止 git checkout 任何历史提交到主工作区
   禁止 git commit
   工作区非空是【预期状态】；你只对自己创建的文件负责，
   清洁性检查只查自己的文件清单，不看全仓库 git status
```

## 6. 杂项

- 本地解释器钉死 `<local-rq009-venv>/bin/python`（系统 python3 缺 pytest）
- 时间戳一律 `date -u +%Y-%m-%dT%H:%M:%SZ`，**不要前瞻估计**
- HPC 作业异步：`sbatch` 后用 `squeue -u u25310231` / `sacct -j <id>` 轮询，
  别提交完就走。作业排队时每 5 分钟往
  `.codex-fleet/rq015g-hpc-resolve/board/progress.log` 追加一行队列状态
  （格式 `<UTC> | <阶段> | 做了什么 | 结论`）
- 每完成一个阶段也往 `progress.log` 追加一行，让 leader 能看到你在动
- 完成后在报告末尾写一个「leader 复核清单」：列出你自己认为最可能被挑战的 3 个点
