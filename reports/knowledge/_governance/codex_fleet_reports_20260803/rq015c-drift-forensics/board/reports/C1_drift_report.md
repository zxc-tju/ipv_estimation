# C1 drift report

## 1. 结论

**第一句（确定性正面结果）：本地代码漂移假设被证伪。**
用 `5edd2810` 的代码回放同一批 40 个冻结锚点，结果与当前 HEAD **逐位相同**
（max|Δipv| = max|Δipv_error| = max|Δweights| = **0.000e+00**，gate_a 通过的是同一批 12 个锚点）。
即：`agent.py` 983→1244 行、`ipv_estimation.py` 313→675 行，外加一次扁平布局→`src/` 的目录重构，
**在这条计算路径上、对这 40 个锚点是严格保行为的**。

这条结论关掉了一个长期悬念——"重构可能悄悄改了行为"。它同时意味着
**RQ015B 基于当前代码得到的 D1/D2 机制结论，不会因为用错代码版本而失效**。

**第二句（存档来源）：verdict = LOCAL_FORENSICS_INCONCLUSIVE。**
legacy 代码 gate_a 也是 12/40（阈值 39/40）。差异源同时**不在** legacy 代码、**也不在** current 代码；
本地取证到此为止，按 PI 裁定不接 HPC。剩余候选见 §5，并已被 §11 的新证据重新排序。

- 当前侧基线 gate_a=12/40；B1 冻结记录 gate_a=12/40。
- legacy 代码本地回放 gate_a=12/40。
- current 侧 `ipv` 对齐但 `ipv_error` 失配的锚点数为 15；legacy 侧为 15。
- 数值健康：current solved=40/40，legacy solved=40/40，nonfinite=0，solve_errors=0。
- `git worktree add` 因 sandbox 无法写 `.git/worktrees/`，本轮改用本地 git 对象归档提取 legacy 源码到 C1 work；这是方法偏差，未触碰现行代码。

## 2. 方法与口径

- legacy commit: `5edd28104bf5989e2dc258c9405ce897d7523cc4`。
- head commit: `511b936c84e30805b765bf0fe157a3faad418414`。
- 隔离方式：current 与 legacy 分别由独立 Python 进程导入；JSON 产物后再由比较进程合并。
- legacy module provenance: `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/agent.py`。
- current module provenance: `src/sociality_estimation/core/agent.py`。
- 参数来源：`configs/ipv_sigma01_exact.json`；history_window=10，min_observation=4，reference clip=60m，max_points=40，smooth_points=40。
- 40 个 pilot 锚点来自冻结 `sample_v1.csv` 的 B1 分层抽取；脚本断言 split 只含 `development`/`guard`。

## 3. 基线对齐

- C1 current 进程重算 gate_a=12/40。
- B1 `BLOCKED_B1.md` gate 行读取结果 gate_a=12/40。
- 二者一致，说明 C1 的 pilot 抽取与 current 调用口径对齐。

## 4. 主实验结果

- legacy 代码本地回放 gate_a=12/40。
| signature | total | current_pass | legacy_code_pass | current_ipv_same_err_diff | legacy_ipv_same_err_diff |
| --- | --- | --- | --- | --- | --- |
| N | 12 | 0 | 0 | 0 | 0 |
| U | 14 | 12 | 12 | 1 | 1 |
| Z | 14 | 0 | 0 | 14 | 14 |

| source | total | current_pass | legacy_code_pass |
| --- | --- | --- | --- |
| nuplan | 20 | 5 | 5 |
| waymo | 20 | 7 | 7 |

| n_band | total | current_pass | legacy_code_pass |
| --- | --- | --- | --- |
| FULL | 18 | 5 | 5 |
| RAMP | 22 | 7 | 7 |

逐行对照已写入 `work/gate_legacy_vs_current.csv`。代表性失配行：
| anchor_id | sig | src | band | cur_ipv_diff | cur_err_diff | leg_ipv_diff | leg_err_diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ipv_007762|90|1 | Z | nuplan | FULL | 0.000e+00 | 0.0498660195 | 0.000e+00 | 0.0498660195 |
| ipv_008146|185|1 | Z | nuplan | FULL | 0.000e+00 | 0.0291652952 | 0.000e+00 | 0.0291652952 |
| ipv_010732|5|2 | Z | nuplan | RAMP | 0.000e+00 | 0.0260467792 | 0.000e+00 | 0.0260467792 |
| ipv_007906|9|1 | Z | nuplan | RAMP | 2.776e-17 | 0.0149102358 | 2.776e-17 | 0.0149102358 |
| ipv_007753|34|1 | Z | nuplan | FULL | 2.776e-17 | 0.00244307212 | 2.776e-17 | 0.00244307212 |
| ipv_031967|9|1 | Z | waymo | RAMP | 0.000e+00 | 0.00129254479 | 0.000e+00 | 0.00129254479 |
| ipv_034118|10|1 | Z | waymo | FULL | 2.776e-17 | 6.435e-05 | 2.776e-17 | 6.435e-05 |
| ipv_001269|5|1 | Z | nuplan | RAMP | 0.000e+00 | 3.431e-05 | 0.000e+00 | 3.431e-05 |
| ipv_018362|90|2 | Z | waymo | FULL | 2.776e-17 | 3.393e-05 | 2.776e-17 | 3.393e-05 |
| ipv_033558|6|2 | Z | waymo | RAMP | 2.776e-17 | 3.092e-05 | 2.776e-17 | 3.092e-05 |

## 5. 定位

- legacy 代码本地回放没有改善 gate，因此本地取证不足以把差异归到已提交代码变更。剩余候选按证据排序如下：
- HPC 侧未提交改动：与本轮现象兼容；本地对象只能证明 5edd2810 存在，不能证明当时作业目录无未提交文件。
- 环境依赖版本：与本轮现象兼容；本轮 Python/numpy/scipy/pandas 版本记录在 summary，但无法代表 2026-06-12 HPC 环境。
- 输入 pkl 版本：与本轮现象兼容；本轮使用当前本地 pkl，未连接 HPC 核对当时数据快照。
- 要定论还缺：HPC 当时 checkout 的工作区快照或 attest、当时环境包版本、当时 pkl 快照 hash。

| swap | gate_a_after | note |
| --- | --- | --- |
| not_applicable | 12 | legacy 代码本地回放也未接近 39/40，优先排序剩余候选而非函数替换。 |

## 6. 数值健康自查

- current solved=40/40；legacy solved=40/40；异常计数=0。
- 非有限值计数=0。
- K_current=7；K_legacy=7。
- 40 个锚点实际求解计数=40。

## 7. lyft / av2 可达性盘点

| 选项 | 路径 | 代价 |
| --- | --- | --- |
| HPC snapshot inventory | `inventory_legacy_layout.sh/.sbatch` + `finalize_legacy_inventory.sh/.sbatch` | 需要 HPC 登录、Slurm 队列、只读盘点；本轮未执行。 |
| payload migration | `migrate_legacy_payloads.sh/.sbatch` + `attest_migrated_snapshots.sh/.sbatch` | 需要先完成 inventory 和快照 attest；体量按脚本文档推断为多源 full_datasets 级别。 |
| checkout sync/topology | `sync_tongji_checkout.sh` + `ensure_interhub_data_topology.sh` | 用于把 Tongji checkout 与本地 InterHub 拓扑对齐；不等于搬运 lyft/av2 原始数据。 |

- 本地 full_datasets pkl 目录 `data/interhub/raw/full_datasets/pkl`：10 个 pkl，文件名为 `train_singapore.pkl, train_vegas1.pkl, train_vegas2.pkl, train_vegas3.pkl, train_vegas4.pkl, train_vegas5.pkl, train_vegas6.pkl, waymo_0-299.pkl, waymo_300-499.pkl, waymo_800-999.pkl`。
- 本地 lyft 可用性：False；本地 av2/argoverse 可用性：False。
- 预期 HPC 路径与体量（推断）：legacy full dataset 文档指向 `interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all`；记录 CSV 10335 行、pkl_events=10335、matched_rows=10335；其中 nuplan_train=7825、av2_motion_forecasting=2510。排除 subset 后推断需算 5335 行，其中 av2 只剩 10 行。
- lyft：本轮指定脚本/文档与本地 full_datasets 文件名未发现 lyft 路径或文件；若 PI 需要 lyft，先走 HPC inventory，而不是假设 managed snapshot 已含 lyft。
- `process_argoverse.py` 暴露的 av2/Argoverse 输入格式：它读的是预处理 CSV，不是原始 AV2 scenario parquet。argo1 要 `<case_id>_lt.csv`、`<case_id>_gs.csv` 的 `x,y` 并用 sample_time=0.1 推导速度/航向，同时要 `<case_id>_reflinelt.csv` 与 `<case_id>_reflinegs.csv` 的 `x,y`；argo2 要 `<case_id>_ego*.csv` 与 `<case_id>_agent*.csv` 的 `x,y,vx,vy,heading`，以及 `refline*_lt*.csv` / `refline*_gs*.csv` 的 `x,y`。这与现行 InterHub pkl 的 normalized vehicles/road_info/timestamps 口径不同，需要先做 CSV 转换。

## 8. 局限与不做的事

- 本轮未接 HPC、未 ssh、未搬运数据。
- 结论只覆盖冻结 40 个 pilot 锚点，不外推到全量样本。
- 未读取 RQ014 相关评分字段；未重新抽样；未写 `.codex-fleet/rq015b-repair/`。
- 未修改 `src/`、`pipelines/`、`configs/`、`scripts/`。

## 9. artifact 清单 + sha256

自检结果：
- `git diff --stat -- src pipelines configs scripts`: ``。
- 禁用术语 grep：``。
- `.codex-fleet/rq015b-repair/` 仅执行 `ls -la` 只读检查；未写入。
```
total 0
drwxr-xr-x@  4 xiaocong  staff  128 Jul 31 17:02 .
drwx------  20 xiaocong  staff  640 Aug  1 00:23 ..
drwxr-xr-x@ 29 xiaocong  staff  928 Jul 31 18:59 board
drwxr-xr-x@ 23 xiaocong  staff  736 Jul 31 18:26 work
```

- `.codex-fleet/rq015c-drift-forensics/board/C1_heartbeat.log` SHA-256 `fc4412bf1af6d87809cc7d9baab06ffd00e5b0b9e2c11a56318f03c5df88a3db`
- `.codex-fleet/rq015c-drift-forensics/board/reports/C1_drift_report.md` SHA-256 `eab4004d34a9c85af6a69d779557c677280f92a45aa419f2056e5a9270c93f29`
- `.codex-fleet/rq015c-drift-forensics/work/c1_summary.json` SHA-256 `58ace05220d56fc7ba25e15c0dbcba7f0e2c24c9202927b33bf4abb0953a16db`
- `.codex-fleet/rq015c-drift-forensics/work/current_40.json` SHA-256 `33b96e4764dfd0224b78523848a643f6f0c47dc70ade2664e100049107f80ee9`
- `.codex-fleet/rq015c-drift-forensics/work/gate_legacy_vs_current.csv` SHA-256 `836ac8b5a2f0e13ffb9103a40064a8fe10df0d8f7a89f761fe8aa8caa439933a`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_40.json` SHA-256 `2fc1850ef359a02b3cdd5c2b890d22d285df6a41643d98b9bc2e12de46907d45`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/HPC_FULL_DATASETS_NUPLAN_AGV_COMMANDS.md` SHA-256 `65b2d7c82d2839cc069c04b41e9071b9469b130ea91a07df381e5e9e7ed76e2e`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/PROJECT_STRUCTURE.md` SHA-256 `904dd01a6d8bc74de0639e9efd7d24bc95274e417a6ff7722ac463f48aac3a7b`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/agent.py` SHA-256 `8c5c633846cb07c88c31c19770a6f69a7bcfde058c0dad291d1cc79ab4663d08`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/ipv_estimation.py` SHA-256 `30b9fd0fbf615b737d201387710d0ebd986e468e49baa1b92b6cc7ca5e827dfe`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/process_argoverse.py` SHA-256 `a057d561b4272b52fede9d5f097dfc78e8be05dcf68a8dac9f5aad2ee15a14f1`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/process_interhub.py` SHA-256 `0a08606200c97fc4b340444b2cf56317c8905a24ea7a0712ea2b3f50f22b334d`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/submit_full_datasets_sigma01_array.sh` SHA-256 `b01d1e2507c7032ccad604610ae1b8116e5d50b939cc6d1e9e41403103e1d491`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/Lattice.py` SHA-256 `f3ca6075748e77d15e790316aaf28dfcb67710d2ef0091f52f64f301d926e70a`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/build_missing_ipv_rerun_input.py` SHA-256 `9b726f6a62c548e89abdd8f6e5bad59330b255ca58fdbfa18e2c718f6d2a58b4`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/lattice_planner.py` SHA-256 `0626acbb747fc9753f2e7af5a1aa2cc9307ba3ba8c64ccc84c93d84a9b508934`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/merge_subsets_for_yiru_ipv_archives.py` SHA-256 `0d3afba6c1d08261fd62985759919f6fd850b0a1f79be1403abb629b3b789c20`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/update_ipv_distribution_report.py` SHA-256 `c193018895568087cb0e98340de82db73bca11938ced4ffb64278f3adeba318f`
- `.codex-fleet/rq015c-drift-forensics/work/legacy_5edd2810_src/tools/utility.py` SHA-256 `46d58d2dfc13f2bbcf3acf4a2986dc4145d52fbd2d06e40e3f8a7a59fdc154c5`
- `.codex-fleet/rq015c-drift-forensics/work/pilot_40.json` SHA-256 `69228579e2e436f50fbe3fa0c648199e5fe5ccd516d83a8beed90e04bead42c6`
- `.codex-fleet/rq015c-drift-forensics/work/run_c1_compare.py` SHA-256 `004492ee9f2a1ae12dc64b9ef00de1faac16e42abf61a712bab6e831f6ad3ae5`
- `.codex-fleet/rq015c-drift-forensics/work/run_c1_current.py` SHA-256 `0b13e6931ec140c043cf1247f6140e32ee1e87cc661389c72e2e356c3983d41a`
- `.codex-fleet/rq015c-drift-forensics/work/run_c1_legacy.py` SHA-256 `0309077b9aec0c637d122ac783349955282b697e41ec9f6100c532cf4018a5a2`
- `.codex-fleet/rq015c-drift-forensics/work/run_c1_prepare.py` SHA-256 `97739125b8f60299e3d3546af2ea33125e0da78efa883450eef700200b17878c`

---

## 10. Leader 复核（track C leader 独立核算，2026-07-31）

C1 的数字全部复核通过。以下三条是 leader 独立算出、**C1 报告未明说**的内容，
它们改变了本轮结论的强度，须与上文一并阅读。

### 10.1 legacy 源码提取是逐字节忠实的（方法偏差已闭合）

`git worktree add` 被 sandbox 拒后，C1 改为从本地 git 对象提取 5edd2810 源码。
leader 对提取结果逐文件重算 git blob hash 并与 `git ls-tree -r 5edd2810` 比对：

```
计算路径全部命中且逐字节一致（13/13，mismatch=0）：
  agent.py  ipv_estimation.py  process_interhub.py
  tools/utility.py  tools/Lattice.py  tools/lattice_planner.py
  submit_full_datasets_sigma01_array.sh  process_argoverse.py  等
导入闭包完整：present 集合中未出现指向"未提取仓库文件"的 import
（唯一未命中的 import 名为 zipfile/base64/html/seaborn/shapely，
  均为标准库或第三方包，不是本仓库文件）
```

⇒ 该方法偏差不削弱结论。就 provenance 而言它反而**强于** worktree：
每个 blob 都被逐一验签，且对主仓库 `.git/` 零改动。

### 10.2 legacy 与 current 的逐锚点结果**逐位相同**（本轮最强的一条事实）

leader 绕开 C1 的比较脚本，直接从 `current_40.json` / `legacy_40.json` 重算：

```
40/40 锚点  ipv 与 ipv_error 完全相同
max |ipv_current  - ipv_legacy|      = 0.000e+00
max |err_current  - err_legacy|      = 0.000e+00
max |weights_current - weights_legacy| = 0.000e+00   (权重向量也逐位相同)
gate_a 通过集合：current 与 legacy **是同一批 12 个锚点**
```

因此本轮的结论应当分成两句，而不是一句：

1. **PI 的"本地代码漂移"假设被证伪**（这是**肯定性**结论，不是"没查出来"）。
   2026-06-12 至 HEAD 之间的目录重构与改写，在这条计算路径上、对这 40 个锚点
   是**严格保行为**的：agent.py 983→1244、ipv_estimation.py 313→675 的全部改动，
   没有移动任何一个锚点的数值。
2. **存档究竟由什么产出，本地证据不足以定论**（这才是 `LOCAL_FORENSICS_INCONCLUSIVE`
   所指的部分）。差异源同时不在 legacy 代码、也不在 current 代码。

C1 报告 §1 只写了第 2 句。第 1 句是本轮实际买到的东西，不应丢失。

### 10.3 与存档的失配是**两类**，不是一类（后续追查的直接抓手）

leader 按 signature 重算"IPV 是否对上 / 误差是否对上"：

| signature | n | IPV 对上存档 | 误差对上存档 | IPV 对上但误差对不上 | 失配幅度 |
| --- | --- | --- | --- | --- | --- |
| U | 14 | 13 | 12 | 1 | — |
| **Z** | **14** | **14** | **0** | **14** | 误差差 2.6e-06 .. 5.0e-02 |
| **N** | **12** | **0** | **0** | 0 | IPV 差 3.8e-04 .. 2.8e-01 |

两类失配的形态完全不同：

- **Z 类（14/14）**：先说清楚**什么不算信息**——Z 类的 `archived_ipv` 按构造即为 `0.0`
  （已核：14 行存档值取值集合 = {0.0}），所以"两边都是 0"本身是废话。
  **有信息量的是重解之后发生了什么**：

  ```
  重解后 current_ipv 取值集合 = {0.0, ±2.775558e-17}，而 2.775558e-17 = 2^-55，
  是对称项求和的浮点残渣，数值上就是零
  ⇒ 权重的【一阶矩】Σwᵢxᵢ 仍然落回零，即重解后权重仍然保持对称，argmin 落点未变
  ```

  而误差标量的定义（本轮实测核对，14/14 逐位吻合）是

  ```
  ipv_error = 1 − sqrt(Σwᵢ²)        ← 只依赖【二阶矩】，与一阶矩无关
  ```

  ⇒ **Z 类的失配只发生在 Σwᵢ² 上，不在权重的一阶矩上。**
  这正是它排除"轨迹生成 / 参考线差异"的原因：那类差异会同时移动一阶矩，
  从而移动 IPV；而这里 IPV 没动。
- **N 类（0/12）**：IPV 本身就差（3.8e-04 .. 2.8e-01），一阶矩也变了，与 Z 类不是同一回事。

⇒ 后续若要继续追，这两类应当**分开追**。本轮**不做**进一步验证，仅登记观察到的结构。

### 10.4 清洁性（只查 track C 自己的文件清单）

```
git diff --stat -- src pipelines configs scripts   -> 空
.codex-fleet/rq015b-repair/ 下 mtime 仍停在 7月31 17:31–18:31（冻结未被写）
禁用术语 grep（estimability / 测出 IPV / 未测出）-> 0 命中
                （仅 RQ007 目录名本身含该词，属路径引用）
未创建 git worktree（被 sandbox 拒），故无 worktree 需要在结项时移除
```

### 10.5 leader 对 §7（lyft/av2）的保留意见

§7 的口径与结构可用，但第 90 行那段行数推断（10335 / 7825 / 2510 / 5335 / av2 只剩 10 行）
是从 legacy 文档反推的，**leader 未独立复核**，且表述本身不够清楚。
PI 若要据此决策，建议只采信其定性部分：
**本地 full_datasets 无 lyft、无 av2**（已复核为真：10 个 pkl 全是 nuplan `train_*` 与 `waymo_*`），
以及 `process_argoverse.py` 读的是**预处理 CSV 而非原始 AV2 parquet**（这决定了取数代价）。
定量体量以 HPC inventory 实跑为准。

---

## 11. 监督方复核后的补充（2026-07-31 放行时并入）

### 11.1【数据完整性问题，独立于 RQ015，需报 PI】一个原始输入在今天凌晨被改动且传输不完整

监督方查本地原始输入 mtime，leader 已独立复核（`stat` + B1 冻结的 `pkl_status.csv`）：

```
train_singapore / train_vegas1..6      2026-06-09T01:08:30Z .. 01:08:42Z
waymo_0-299                            2026-06-09T06:26:26Z
waymo_800-999                          2026-06-09T06:42:02Z
   ⇒ 以上 9 个都【早于】存档日期 2026-06-12，与"就是当年那份输入"一致

waymo_300-499.pkl                      2026-07-31T02:42:06Z   ← 【今天被写过】
   B1 冻结记录：exists=True, loadable=False, n_events=0,
                error="_pickle.UnpicklingError: pickle data was truncated"
   ⇒ 至今仍不可读
```

02:42 UTC 早于本 Cowork 会话（约 08:30 UTC 起），**不是 C/D/E 三条 track 造成的**。

**这是一个需要单独处理的数据完整性问题**：一个原始输入被替换且传输不完整，
当前仍无法读取。B1 当时按 `pkl_available=False` 把它排除了，所以它没有污染本轮结论，
但它**独立于 RQ015 需要 PI 知晓并修复**（重新同步该 pkl）。

### 11.2 剩余候选重新排序：「输入 pkl 版本」升为最可疑

§5 原本把三个剩余候选并列。11.1 证明了一件事：**原始输入是会被改动的**。
据此重排：

| 排序 | 候选 | 依据 | 能否不接 HPC 验证 |
| --- | --- | --- | --- |
| **1** | **输入 pkl 版本** | 已证实原始输入会被改动（11.1 是一个实例） | **可能可以**（见 11.3 第 1 条） |
| 2 | HPC 侧未提交改动 | 与现象兼容；本地对象只能证明 5edd2810 存在，不能证明当时作业目录干净 | 否 |
| 3 | 环境依赖版本 | 与现象兼容；无法用本轮环境代表 2026-06-12 的 HPC 环境 | 否 |

注意这三条是**并存的可能解释**，本轮没有任何一条被证实；此处只是按"可验证性 × 已有证据"排序。

### 11.3 两条具名下一步（**本轮不做**，交 PI 决定是否启动）

**下一步 A — 输入 pkl 版本比对（最便宜，且唯一可能不接 HPC 就出结论的一条）**
> 找出 sigma01 时代是否记录过输入 pkl 的 SHA256 / 行数 / case 数
> （候选出处：run manifest、`main_workflow.log`、HPC 提交脚本的 `.out` 日志、
> `data/interhub/raw/full_datasets/BATCH_CURRENT.txt`），与本地现有 pkl 逐个比对。
> - 若能对上 ⇒ 输入未变，矛头转向 HPC 侧未提交改动或环境版本；
> - 若对不上 ⇒ 谜底揭晓，且**不必动 HPC**。

**下一步 B — 修复 `waymo_300-499.pkl`（数据完整性，与 RQ015 无关）**
> 重新同步该文件并校验可读性与 `n_events`。当前状态下任何用到该 folder 的分析
> 都在无声地少一批数据（B1 已显式排除，但后续分析未必会）。

### 11.4 §7（lyft/av2）可信度分级 —— 供 PI 决策时区分对待

- **[已复核，可采信]** 本地 `data/interhub/raw/full_datasets/pkl/` 共 10 个 pkl，
  全部是 nuplan `train_*` 与 waymo `*`，**无 lyft、无 av2**。
- **[已复核，可采信，且这是决定取数代价量级的关键一条]**
  `5edd2810:process_argoverse.py` 读的是**预处理 CSV，不是原始 AV2 scenario parquet**
  （argo1 要 `<case>_lt.csv` / `_gs.csv` / `_reflinelt.csv` / `_reflinegs.csv`；
  argo2 要 `_ego*.csv` / `_agent*.csv` / `refline*_lt*.csv` / `refline*_gs*.csv`）。
  ⇒ **若那批中间产物还在，取数可能只需搬中间产物，不必搬整个原始数据集**，
  代价量级差一个数量级。这是 PI 决定要不要取 lyft/av2 时最关键的信息。
- **[未复核，仅供参考，请勿与上面两条同等采信]**
  §7 第 90 行的体量推断（10335 / 7825 / 2510 / 5335 / av2 只剩 10 行）
  由 C1 从 legacy 文档反推，leader **未独立复核**，且表述本身不够清楚。
  定量体量应以 HPC inventory 实跑为准。

---

### 附注：§9 中本报告自身的 SHA-256 已过期（预期行为，非完整性失败）

§9 由 C1 在结项时生成，其中 `C1_drift_report.md` 自身的哈希
`eab4004d…` 是**追加 §10/§11 之前**的值。文件无法包含自身的哈希，这是固有的。
**§9 中其余所有 artifact 的哈希仍然有效**（那些文件在 §10/§11 追加后未被改动）。
本报告的最终哈希记录在 `board/progress.log` 与 `board/STATUS.md` 中。
