# RQ016C-1：只用纯人-人样本重建供 OnSite 使用的人类 envelope（执行记录）

执行日期：2026-08-04｜基线提交：`7480c173`｜执行方：两个 codex agent（H1 首轮、H2 修正重跑）｜监督方：Claude

## 这一轮在整个研究里的位置

最终目标是**在线验证**：判断一辆自动驾驶车表现出的社会交互倾向像不像人。判定由两道串联
弃权机制构成——机制一判断某一帧的 IPV（Interaction Preference Value，表示交互倾向的标量）
数值能否估出（RQ015 已冻结）；机制二拿通过机制一的数值与人类参照分布（envelope）比。

RQ016 重建过一次 envelope，但 RQ016B 查实其中 **10.9009% 的目标值是自动驾驶车自己的 IPV**
（69,288/635,618）。拿一个含自动驾驶车行为的参照去判自动驾驶车，会削弱「与人类比较」这一主张。

**PI 于 2026-08-04 裁定**：envelope 是建在数据分布之上的查询机制，针对不同目标可以建不同的
envelope；本研究要问「OnSite 的自动驾驶车的 IPV 是否落在人类的分布范围内」，参照就应当是
**纯人类**的分布。本轮执行这条裁定。

## 做了什么

在**只含纯人-人交互**（`agent_type_pair == "HV;HV"`）、台账覆盖（只含 development + guard）、
机制一通过（`status == "OK"`）的 **2,442,625 行**上重拟 context-conditioned split-conformal
envelope，并持久化成可以给外部行打分的产物。

### 特征集的必要修改

在纯人-人池内，`agent_type_pair` 恒为 `HV;HV`、`av_included` 恒为 `all_HV`、
`vehicle_type_list` 恒为全 HV 列表——三列都编码「双方是不是自动驾驶车」。它们**必须移除**，
理由两条：

1. 车辆是否为自动驾驶车是**被检验的对象**，不是它所处的情境，本就不该作为 context 变量。
2. OnSite 行在这三列上的取值（`AV;HV` / `AV` / `['AV','HV']`）**在训练池中从未出现**，
   保留任何一列都会使打分时撞上未知类别；且 `agent_type_pair` 曾是支持门分格键之一，
   保留它会让 OnSite 全部 67,861 行落不进任何格、机制二全量弃权。

最终 context：22 项数值 + **4 项类别**（`geometry_path_category`、`geometry_path_relation`、
`turn_pair_label`、`priority_role`）；支持门分格键 `geometry_path_category + priority_role`；
支持门距离特征 12 项。

### H1 与 H2 的关系（错误历史，原样保留）

H1 是首轮，只移除了 `agent_type_pair` 与 `av_included`，**漏掉 `vehicle_type_list`**——
这是监督方任务书的规格错误，不是执行方的执行错误。监督方复核实测：纯人-人训练池
`vehicle_type_list` 含 `AV` 的行为 0/2,442,625，而 OnSite 全部 67,861 行都是 `['AV','HV']`，
故 **H1 的产物对 OnSite 的可打分行数为 0/67,861**。H1 自己的负对照当时报 `UNEXPECTED_PASS`
（扰动后断言仍通过），说明那条断言没有在检查它声称检查的东西。

H2 是修正重跑：移除 `vehicle_type_list`，并新增三条验收——类别词表覆盖断言、真实 OnSite
全量打分演练、两条必须真的 FAIL 的负对照。**H1 的报告与产物原样保留在本目录，不删不改。**

## 主要结果

纯人-人参照池 2,442,625 行（development 1,752,509 + guard 690,116，held_out 实测 0），
分 fold：train 974,984 / calibration 481,088 / guard_tune 499,893 / test 486,660。

| alpha | coverage | 覆盖/过支持门行 | 平均宽度 | 中位宽度 | 机制二弃权率 |
|---|---:|---:|---:|---:|---:|
| 80 | 0.796299 | 367,840/461,937 | 0.782266 | 0.757047 | 5.0801%（24,723/486,660） |
| 90 | 0.898038 | 414,837/461,937 | 1.238468 | 1.265956 | 同上 |
| 95 | 0.948623 | 438,204/461,937 | 1.714635 | 1.759300 | 同上 |

### 真实 OnSite 全量打分演练

只加载持久化产物、不重新拟合，对全部 67,861 行跑通打分路径，产出三层区间与支持门判定。
**支持门通过率 21,936/67,861 = 32.3249%**，逐格差异极大：

| 格 | OnSite 行 | 通过 | 通过率 |
|---|---:|---:|---:|
| `F\|priority` | 29,677 | 13,958 | 47.03% |
| `F\|yield` | 14,537 | 4,783 | 32.90% |
| `MP\|priority` | 10,291 | 1,387 | 13.48% |
| `MP\|yield` | 7,590 | 1,107 | 14.58% |
| `CP\|priority` | 2,336 | 50 | 2.14% |
| `F\|equal` | 1,535 | 368 | 23.97% |
| `CP\|yield` | 1,488 | 275 | 18.48% |
| `MP\|equal` | 291 | 8 | 2.75% |
| `CP\|equal` | 116 | 0 | 0.00% |

即：落进哪个格按几何与优先权粗分，通过与否还要看运动学上有没有足够近的人类样本；
**OnSite 约三分之二的帧在人类数据里找不到足够相近的情境。**

**这一演练只证明打分管线在真实 OnSite 行上可运行，不构成对任何一辆自动驾驶车的判定。**
OnSite 一行都没有机制一的判据——K2 台账 `artifact_id == onsite_dense_timeseries` 的
281,268 行中 `status`、`reason_code`、`mse_0..mse_6` 非空计数全部为 0。机制一未通过之前
不得进入机制二。

## 目录内容

平铺布局（`reports/studies/**/` 下的 `00_entry/`、`01_results/`、`02_process/`、`scripts/`
等子目录名被 `.gitignore` 忽略，放进去不会被跟踪）。

| 文件 | 内容 |
|---|---|
| `RQ016C_2_human_only_envelope_fixed.md` | **H2 最终报告** |
| `RQ016C_1_human_only_envelope.md` | H1 首轮报告（错误历史，保留） |
| `key_numbers.json` | H2 的全部关键数字 |
| `onsite_scoring_dryrun_summary.json` | OnSite 全量打分演练汇总 |
| `run_rq016c_h2_human_only_envelope.py` | 可复跑主脚本 |
| `score_external_rows.py` | 外部行打分脚本 |
| `envelope_model/` | 特征合同、支持门规则、逐格与全局 conformal 半径、逐格支撑量、打分说明 |
| `RQ016C_H1_kickoff.md` / `RQ016C_H2_kickoff.md` | 两轮任务书（H2 的任务书含缺陷说明） |

### 未入库的产物

拟合好的模型本体 **未入库**（体积 171,911,135 字节，约 164 MB，超出仓库对跟踪文件的体积约定）：

```
路径   .codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl
sha256 bc25302b4a7a307e3c73b3429b880e3cfda59074fc80850a732a93a67ef75de2
大小   171911135 bytes
```

同样未入库的还有 `onsite_scoring_dryrun.parquet`（67,861 行，约 2.7 MB）与三个自测 parquet，
均在 `.codex-fleet/rq016c-human-only-envelope/work/H2/` 下。
用 `run_rq016c_h2_human_only_envelope.py` 可从本目录已入库的合同与源数据重新生成模型。

## 效度边界

1. **本轮结果与 RQ009 已发表数不构成复现关系**——RQ009 的 test 域含 RQ007 held_out，
   本轮受红线约束仅 development + guard；且特征集已修改。
2. **无同源迁移证据**：RQ009 的 LODO 只含 4 个留出源，OnSite 不在其中。
3. **OnSite 的支持门通过率只有 32.3249%**，两道门叠加后可判比例会更低。
4. **OnSite 存在坐标系异常**：监督方实测 7 行（0.0103%）的 `relative_distance_anchor`
   ≈ 570,761.6 米，`relative_dx_anchor` ≈ −570,761.6 而 `relative_dy_anchor` 正常，
   典型的单侧坐标原点不一致。量极小，不影响本轮结论，但**真正做 OnSite 分析前须处理**。
   除这 7 行外分布正常（p99 = 119.56 米、p99.9 = 200.87 米，人类池上限 223.083 米）。
5. **`apet_online_proxy` 填充率 OnSite 7.90% 对 InterHub 40.26%**，系统性差异。
6. 描述性结果，不构成因果主张。禁用 `estimability` 与「测出/未测出 IPV」表述。

## 合规自证

- 参与计算行中 `rq007_split` 不在 `{development, guard}` 的实测计数为 **0**。
- 未打开受保护 confirmation 划分文件；未读取 RQ014 致盲相关评分字段。
- 未修改 `data/derived/`、RQ009 与 RQ016 原 run 目录、五个受保护文件。
- 本目录已做绝对路径与密钥扫描，命中的用户机器路径已改为仓库相对路径或 `~`。
