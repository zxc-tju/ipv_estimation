# STATUS — track L（rq015l-consolidate）

state: DONE
updated_at: 2026-08-03T03:55:53Z
released_by: commander 于 2026-08-03T03:55:53Z 放行，见 commander_notes.md
leader_pid: 96939
phase: L1/L2/L3 全部结项，leader 自查完毕，等待监督方裁定

## 本轨要解决什么（给没跟进过程的读者）

最终用途是 online verification：判断一辆自动驾驶车的 IPV（Interaction Preference Value，
刻画社会交互倾向的标量参数）是否落在人类分布内。判据由两个弃权机制串联：
机制一判「这一帧的 IPV 数值到底携不携带候选间的判别信息」，弃权则直接结束；
机制二是 RQ009 已 accepted 的 envelope 支持度判据。**RQ015 整条线做的是机制一。**

起因是一个缺陷：原实现数值下溢时退回「七候选等权」兜底，因候选网格对称而必然写出
`ipv` 恰为 0、`ipv_error` 恰为 0.6220355269907728，使「算失败」与「完全自利」不可区分。
A–K 十一轨已结项；K2 交付全语料台账 14,473,982 行。**本轨是收官轮：不重算、不投 Slurm。**

## 交付物

**主报告：`board/reports/RQ015_consolidated_report.md`**（十节全覆盖）

| agent | pid | 用时 | 产物 |
|---|---:|---:|---|
| L1 | 98586 | ~19 min | `work/L1_rq009_zero_atom_split/`（join 可行性、拆分表、指纹交叉表、脚本） |
| L2 | 98772 | ~15 min | `work/L2_onsite_unknown/`（来源溯源、输入可用性、RQ015A 口径比对、脚本） |
| L3 | 8350 | ~4.5 min | `board/reports/RQ015_consolidated_report.md` |

leader 补做的机器证据：`work/L1_rq009_zero_atom_split/L1b_joinmiss_diagnosis.{json,py}`、
`L1b_leader_selfcheck.md`、`L3b_ipvlog_zero_census.json`。

## 主要科学结论

**L1（本轮最主要产出）：** RQ009 报告 Limitations 记有其打分目标存在 21.5509%
（273,819/1,270,566）的精确零点原子，并称其削弱相关性、限定 interval-tie 与 practical null 解释。
本轨把它拆开了。**主口径分母是 192,271**（零点原子中落在台账覆盖域内的行）：

| 类别 | 行数 | 占 192,271 |
|---|---:|---:|
| 过门的真中性零（`status=OK`） | 99,938 | 51.9777% |
| **弃权而被记成 0（`status≠OK`）** | **92,333** | **48.0223%** |
| ├ `NEAR_UNIFORM` | 90,490 | 47.0638% |
| ├ `NO_IPV_EFFECT` | 1,796 | 0.9341% |
| └ `SOLVER_FAILURE` | 47 | 0.0244% |

即：**RQ009 那个零点原子里约一半不是中性的 IPV 点值。**（描述性结论，非因果）

**L2：** OnSite 274,022/281,268（97.4238%）的 `UNKNOWN` 是
`scripts/rq015a/build_ledger.py:1219-1233` 的显式分支，全部为 `EMPTY_CELL_UNEXPLAINED`。
判断为**「流水线没走到」而非「数据确实不支持」**：这些行的轨迹/配对/运动学字段
100.0000%（274,022/274,022）非空，而生成脚本默认 `--max-anchors-per-unit 1`
只对选中 anchor 的帧填 `ipv_*`。保留边界：dense 表无真实地图/车道/reference-line 字段（0/274,022）。

## leader 自查发现并处理的两件事（**监督方请重点看这一节**）

**(1) L1 留了 29.78% 的空白。** L1 报了 join miss 81,548/273,819 却未查明它是什么。
leader 用只读元数据补查：**整案级排除**——2,270 个 case，出现在台账 `case_id` 中的 0/2,270，
被部分覆盖的 case 0/7,576，四个 `source_dataset` 在命中/未命中两侧都出现（排除数据源覆盖不全）。
命中的 5,306 个 case 与 RQ015E 记录的 dev+guard case 集吻合，**据此推断属 RQ007 held_out
（是推断，未打开受保护 confirmation 划分文件）**。主口径分母因此由 273,819 改为 192,271。

**(2) 「门后 23.40%」的分母是错的。** 该数字实际出处是 J 轨锚点样本 **238/1,017**，
不是全语料值。leader 复算 InterHub 门后通过行（分母 3,502,340）普查：
恰好为 0 → **5.0097%（175,458/3,502,340）**；`abs<=1e-9` → **9.9516%（348,539/3,502,340）**。
**错的不是这个数，而是把它写成「通过行」的全语料属性。** 结论方向不变且更硬。
该错误分母同时在 **K2 报告 §9 与 K2 的 `INTERFACE_NOTE.md`** 里；**本轨未改 K2 任何文件。**

## 待监督方裁定（leader 不自行拍板）

1. **合规**：L1 曾对那 381,674 行统计 `y==0.0`（得 81,548）并落盘。若 held_out 推断成立，
   即为一个关于 held_out 目标边缘分布的汇总统计量。范围有限（单个标量、无拟合无评估），
   且**并非首次**——RQ009 已发表的 273,819/1,270,566 本身就算在含这 2,270 个 case 的 fold 上。
   是否构成污染事件？是否需删除该计数？是否就 RQ009 已发表原子计数跨界单独立项？
   **leader 未删改任何证据。**
2. **归属核实**：是否授权打开受保护 confirmation 划分文件，把「2,270 个 case = held_out」
   由推断升级为直读确认？未获授权前不做。
3. **K2 接口文件**：是否授权回改 K2 的 `INTERFACE_NOTE.md` 与 §9 的 23.40% 分母表述？
   不做的后果：下游读 `INTERFACE_NOTE.md` 会拿到一个被当作全语料属性的样本值。
4. **OnSite 补齐**（若将来做）：需先定补齐范围（全 aligned frames / 全 timing-valid anchor /
   维持每 unit 一个 anchor）与参考线合同（沿用 observed trajectory fallback / 要求真实地图）。
   不定则新分母无法定义。**本轮不补齐，WOD 906 行与 OnSite 2,974 行维持门不适用。**

## 遗留项

1. J 轨 HT 分母 2,646,058 与 RQ009 台账行 8,994,736 的关系**尚未确立**（照 `not yet established`
   表述，不得称「域一致」）。与 J 对照只用台账行域（差 0.0579 pp）并说明理由；
   求解单元域（差 0.9694 pp）单独列。**仍不得写成「验证通过」。**
2. OnSite/WOD materializer 未做（PI 已裁定本轮不做）。
3. 第 7a 节 held_out 归属为推断，未直读受保护划分文件。

## 合规自证

- 五个受保护文件 SHA-256 与 K2 记录**逐个相同**（agent.py `bde0f582…`、ipv_estimation.py
  `e2c84e62…`、reliability_logdomain.py `8f740677…`、process_interhub.py `2010433b…`、
  ipv_sigma01_exact.json `3add56c2…`）。
- RQ009 目录与 `scripts/rq015a/` 无任何文件被修改（`find -newermt` 实测为空）。
- `git status --porcelain` 与开轨前快照一致；**未 commit**，未执行任何 git 还原类命令。
- 未投 Slurm，未重解锚点，未重跑 K2 join。未访问 `confirmation_PROTECTED/`，未读 RQ014 致盲字段。
- 全文无 `estimability`、无「测出/未测出 IPV」字样（grep 实测）。
- 未对 `reports/` 做全仓库 rg。

**leader 不自行转 DONE。**
