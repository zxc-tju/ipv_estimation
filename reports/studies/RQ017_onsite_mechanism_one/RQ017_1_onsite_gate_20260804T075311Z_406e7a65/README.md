# RQ017-1：为 OnSite 自动驾驶车锚点产出机制一判据（执行记录）

执行日期：2026-08-04｜基线提交：`406e7a65`｜执行方：单个 codex agent（M1，在同济 HPC 上）｜监督方：Claude

## 这一轮在整个研究里的位置

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。判定由两道
串联的弃权机制构成——**机制一**判断某一帧的 IPV（Interaction Preference Value，
表示交互倾向的标量）数值是否携带七个候选之间的判别信息，不携带则弃权、不进机制二；
**机制二**用人类参照分布（envelope）判断当前情境是否有足够的人类样本可比。

RQ015 冻结了机制一并在 InterHub 全语料上跑出台账；RQ016C 用纯人-人样本建好了供 OnSite
使用的人类参照 envelope。**但在本轮之前，OnSite 的自动驾驶车数据一行都没有机制一判据**
——K2 台账里 `artifact_id == onsite_dense_timeseries` 的 281,268 行中 `mse_0..mse_6`、
`status`、`reason_code` 非空计数全为 0。**也就是说这套方法从未真正对准过一辆自动驾驶车。**

本轮补上这一块：为 67,861 个自动驾驶车锚点求解七候选 MSE 并给出机制一判据。

## 为什么必须在 HPC 上跑

K2 那份 InterHub 台账（本轮全部对照的基准）在同济 HPC 上产出。实测 Mac 与 HPC 的求解结果
在 2,300 个重合锚点上 `mse_per_candidate[7]` 字符串不同的有 1,867/2,300 = 81.17%，
最大逐元素绝对差 70.4，`argmin_candidate` 翻转 686/2,300 = 29.83%，差异来自软件栈而非 CPU。
在 Mac 上产出的 OnSite 台账与 InterHub 台账不可比，本轮全部对照会失去意义。

**同源已验证**：`env_parity.json` 状态 `PASS`，G 锚点在本轮环境下重算 2,300 个，
`max_abs_diff = 0.0`。

## 主要结果（监督方已独立复算，逐位一致）

帧级（分母 67,861）：`OK` 37,520 = 55.2971%；`ABSTAIN` 30,341 = 44.7029%
（**全部为 `NEAR_UNIFORM`**）；`NO_IPV_EFFECT` 0；工程失败 0。
与 RQ016C 支持门交叉后**两门都过 14,099 = 20.7763%**。

Case 级（分母 267 个 case）：**至少 1 帧两关都过的 case 有 231 个 = 86.5169%**；
全程无一帧可判的 36 个 = 13.4831%，**其中因机制一全程无解的为 0 个 = 0.0000%**。
即**没有任何一个 case 是全程无法估计 IPV 的**，36 个不可用 case 全部死于机制二无参照
（其机制一通过率 53.78%，与全体 55.30% 基本相同；机制二通过率 0.07%，对全体 32.32%）。

与 InterHub 对照：同一套机制一规则在 InterHub 4,981,984 个求解单元上 `OK` 为 70.3001%，
在 OnSite 上为 55.2971%——**自动驾驶车的轨迹比人类轨迹更难反推出交互倾向**。
成因本轮未验证，不提供解释。

## 目录内容

平铺布局（`reports/studies/**/` 下 `00_entry/`、`01_results/`、`02_process/`、`scripts/`
等子目录名被 `.gitignore` 忽略）。

| 文件 | 内容 |
|---|---|
| `RQ017_1_onsite_materializer.md` | 交付报告。前半 M1 原文，后半监督方附录 A–E |
| `case_level_availability.json` | Case 级可用性分析（本轮由监督方补做） |
| `key_numbers.json` | 全部关键数字 |
| `measurement_contract.json` | 测量合同 preflight 的 8 条断言结果 |
| `env_parity.json` | 环境同源硬断言（版本、import origin、G 锚点逐位复算） |
| `canary_validation.json` | canary 与两条负对照结果 |
| `run_receipt.json` / `retrieval_integrity.json` | Slurm 作业回执、取回完整性 |
| `rq017_onsite_materializer.py` | 可复跑主脚本 |
| `stage_rq017_hpc.sh` / `submit_rq017_array.sbatch` | HPC staging 与 array 作业 |
| `RQ017_M1_kickoff_v4.md` | 执行版任务书 |
| `commander_notes.md` | 监督方裁定与**预注册预测**记录 |

### 未入库的产物

正式台账未入库（`data/derived/` 整体被 `.gitignore` 忽略）：

```
data/derived/rq017_onsite_gate/l1_v1/artifact_id=onsite_dense_timeseries/    约 19 MB，67,861 行
```

HPC 侧运行目录见 `run_receipt.json` 的 `work_dirs/RQ017/<run_id>/`。
用 `rq017_onsite_materializer.py` 可从本目录已入库的脚本与源数据重新生成。

## 效度边界（引用本轮结果必须一并带上）

1. **`NO_IPV_EFFECT` 在 OnSite 上实际不可达**：OnSite 恰为 0 的行为 0/67,861，
   最小非零 `mse_spread` 为 2.32e-08；InterHub 为 19,964/4,981,984 = 0.4007%，
   最小非零 4.77e-15，相差七个数量级。这与观测轨迹 fallback 这条参考线合同一致。
   **拿 OnSite 的弃权理由构成去与 InterHub 对比不成立，只能比总弃权率。**
2. **机制二的缺口是重叠不是数量**：`MP|yield`/`MP|priority` 各有逾百万行人类支撑而通过率
   仅 14.58%/13.48%；`F|priority` 仅 45,283 行却达 47.03%。限制因素是自动驾驶车的运动学状态
   是否落在人类样本附近，不是人类样本数量。
3. **机制二比的是运动学邻域（12 项距离特征），不是 IPV 数值本身。
   「机制二不通过」只意味着无法判定，不得解读为「该车不像人」。**
4. **本轮不对任何车辆作出判断。** 产物只提供机制一判据；车辆层在线验证属后续任务。
5. **参考线用观测轨迹 fallback**，与 InterHub 不同源；可比性由「同一估计器、同一冻结配置、
   同一软件栈」保证，不由参考线来源保证。
6. 7 行坐标系异常（`relative_distance_anchor` ≈ 570,761 米，全部来自
   `onsite:shanghai:T10:C4:native_case:2311`）照常参与求解并入库，状态均为 `OK`，未静默剔除。
7. **未解释的观察**：短历史行（1,572 行）机制一通过率 73.92%，高于满历史行的 54.85%，
   方向与直觉相反，成因未验证。
8. 描述性结果，不构成因果主张。禁用 `estimability` 与「测出/未测出 IPV」表述。

## 合规自证

- 测量合同 preflight `PASS`（8 条断言，含键一对一 67,861/67,861）。
- 环境同源 `PASS`，G 锚点 `max_abs_diff = 0.0`；`sacct` 断言所有 array task 分区不含 `amd`。
- 两条负对照真的 FAIL（`isclose_atol_1e_12`、`theta_0_22`），合成 sentinel 覆盖四种状态
  且未混入正式产物。
- 参与求解行中来自 InterHub 的为 0 行；未打开受保护 confirmation 划分文件；
  未读取 RQ014 致盲相关评分字段；输入列白名单排除了 anchor 表中 9 个旧估计器通道与目标值列。
- 未修改五个受保护文件、RQ009/RQ016/RQ016C 已落盘 run 目录、`data/derived/` 已有内容。
