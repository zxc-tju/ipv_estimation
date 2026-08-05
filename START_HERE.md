# START_HERE: Current Operating Brief

Last reviewed: 2026-08-05.

Use this file as the first stop for a new agent thread. Keep durable policy in
`AGENTS.md`, architecture notes in `PROJECT_STRUCTURE.md`, and the compact research
question index in `STUDIES.md`.

## Current Active Context

- **RQ018 已完成并经监督方独立复算（2026-08-04T22:44:27Z）。这条替代了 A1 那条
  「等待 commander 复核」；A1 原始交付完整保存在执行记录目录里。
  监督方复算后改写了 A1 的结论——引用本轮务必以本条为准。**
  **背景一句话**：在线验证串联两道弃权机制——机制一判断某一帧的 IPV（Interaction
  Preference Value，表示交互倾向的标量）数值是否携带七个候选间的判别信息，机制二用
  人类参照分布判断当前情境是否有足够人类样本可比。RQ015 冻结机制一、RQ016C 建纯人-人
  参照、RQ017 在自动驾驶车上算出机制一判据后，**本轮问：IPV 超出人类范围时行为有没有劣化。**
  分析集为机制一 `status == OK` 且机制二 `mechanism2_gate_ok == True` 的
  **14,099/67,861 = 20.7763%**（来源 `data/derived/rq017_onsite_gate/l1_v1/` 的
  `status,ipv_log` 与 `.codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet`
  的 `mechanism2_gate_ok,lo_90,hi_90,width_90`，连接列 `product_row_key`），
  覆盖 231 个 case、19 个 team；上侧/下侧/区间内 = 2,700/1,998/9,401。
  **结论：超出人类参照的 IPV 表达显著压缩了安全裕度的整体分布，但不制造极端危险情形。**
  **PI 2026-08-05 定性**：IPV 是社会交互倾向属性、本就不是安全代理指标，越界不带来更极端危险
  符合预期；有价值的是它把原本比较安全的情形变得没那么安全。**「未观察到更高风险」是边界条件，
  不是结论本身，引用不得只截后半句。** 分布整体左移：中位 7.51 s 对 8.81 s（−14.8%），
  75% 分位 12.75 s 对 22.38 s（−43.0%）。
  IPV 低于该情境人类参照下界（**比人类更激进**）的帧，后续最小 TTC 中位更短
  （7.51 s 对区间内 8.81 s），**但危险阈值以下的帧占比一致更低**——
  TTC<2 s 为 5.28%(96/1,819) 对 9.85%(861/8,739)，TTC<3 s 为 12.53%(228/1,819) 对
  16.90%(1,477/8,739)；case 层 bootstrap 1,000 次，TTC<2 s 占比差 −0.0457、
  95% CI [−0.0696, −0.0227] 不含 0。**中位数下降来自安全端长 TTC 的减少，不是危险端的增加**
  （25% 分位两组几乎相同，4.105 对 4.089）。上侧越界（更合作让行）无任何劣化对应
  （系数 −0.0146、p_case=0.8564）。unit 级四个非安全子分数无一致模式；
  事故类结果功效不足（全 267 unit 中 `official_safety < 100` 21 个、碰撞/接管扣分非零 18 个）。
  **IPV 符号语义（极易读反）**：`agent.py:1193` 为
  `util = cos(ipv)×自身代价 + sin(ipv)×交互代价`，**IPV 越负 = 越竞争激进**，
  故下侧越界是「比人类更激进」**不是「更消极」**；且**下侧越界 ≠ `IPV<0`**
  （区间内另有 3,611/9,401 = 38.41% 的行也是负 IPV，因为人类在那些情境下同样取负值）。
  **引用红线**：禁用「导致」等因果表述；**不得把「TTC 中位更短」读成「更危险」**；
  主口径无结果（原始 `future_min_ttc_s` 全部不显著 p_case 0.31–0.37，`log1p` 是事后变换）；
  TTC 缺失与曝露相关（上侧 13.70%/下侧 8.96%/区间内 7.04%）；结果变量互相矛盾
  （最小距离方向相反且显著 +1.0396、p_case=0.0064）；共 288 个 p 值，
  标签置换 p=0.0149 承受不了多重校正；与 RQ012B 冻结结论 `RQ012-KC-HARM-NULL`
  是不同曝露定义与分析单元，**不构成推翻**且方向不矛盾。**尚无 `decision.md`。**
  **产物**：`reports/studies/RQ018_abnormal_ipv_degradation/RQ018_1_association_20260804T224427Z_276cf4c/`
  （报告前半 A1 原文、后半监督方附录 A–E；含监督方独立复核脚本与 JSON）。
  **设计要点（后续复用）**：锚点时刻的 TTC、PET proxy、相对距离、接近率**全部是 envelope 的
  22 项 `numeric_context` 与 12 项支持门距离特征**，用作同期结果变量构成循环论证；
  帧级结果必须取锚点之后的未来窗口。
- **RQ017 已完成并经监督方独立复算放行（2026-08-04T13:12:45Z）。这条替代了 M1 那条「等待 commander 复核」，M1 原始交付完整保存在执行记录目录里。**
  **背景一句话**：在线验证串联两道弃权机制——机制一判断某一帧的 IPV（表示交互倾向的标量）
  数值是否携带七个候选间的判别信息，机制二用人类参照分布判断当前情境是否有足够人类样本可比。
  RQ015 冻结机制一、RQ016C 建好纯人类参照，**但在 RQ017 之前这套方法从未真正对准过一辆
  自动驾驶车**（OnSite 台账 281,268 行的机制一判据非空计数为 0）。RQ017 补上了这一块。
  **产物落位**：`reports/studies/RQ017_onsite_mechanism_one/RQ017_1_onsite_gate_20260804T075311Z_406e7a65/`；
  知识层 `reports/knowledge/RQ017_onsite_mechanism_one/`；
  正式台账 `data/derived/rq017_onsite_gate/l1_v1/`（约 19 MB、67,861 行，**未入库**，
  `data/derived/` 整体被 gitignore）。轨道原始工作区 `.codex-fleet/rq017-onsite-materializer/`。
  **venue = 同济 HPC**（分区 intel,fata，未用 amd），理由是产物来源一致性而非速度：
  Mac 与 HPC 的求解结果在 1,867/2,300 = 81.17% 的锚点上不同。同源已验证：G 锚点重算
  **max_abs_diff = 0.0**。
  **帧级结果**（分母 67,861）：`OK` 37,520 = 55.2971%、`ABSTAIN` 30,341 = 44.7029%
  （**全部 `NEAR_UNIFORM`**）、`NO_IPV_EFFECT` 0、工程失败 0；与 RQ016C 支持门交叉后
  **两门都过 14,099 = 20.7763%**。
  **Case 级结果**（分母 267 个 case）：至少 1 帧可判的 **231 个 = 86.5169%**；全程不可判 36 个，
  **其中因机制一全程无解的为 0 个 = 0.0000%**——**没有任何一个 case 是全程无法估计 IPV 的**，
  36 个全部死于机制二无参照。**真正的约束是人类参照覆盖，不是可估计性。**
  **必须一并引用的边界**：(1) `NO_IPV_EFFECT` 在 OnSite 上**实际不可达**
  （0/67,861，最小非零 `mse_spread` 2.32e-08 对 InterHub 的 4.77e-15），
  **弃权理由构成不可与 InterHub 对比，只能比总弃权率**；(2) 机制二比的是运动学邻域
  （12 项距离特征）**不是 IPV 数值**，「机制二不通过」**只意味着无法判定，不得解读为
  「该车不像人」**；(3) 机制二缺口是**重叠不是数量**——`MP` 两格逾百万行人类支撑而通过率
  仅 13–15%，`F|priority` 仅 45,283 行却 47.03%；(4) **本轮不对任何车辆作出判断**；
  (5) 未解释观察：短历史行（1,572 行）机制一通过率 73.92%，高于满历史行 54.85%。
  **一次公开失败的预测**：监督方预注册（时间戳 2026-08-04T06:22:47Z，早于派发）预测机制一
  通过率 ≈ 80%（区间 65–85%），**实测 55.2971%，落在区间外，预测失败**。原因是校准样本来自
  `max_anchors_per_unit=1` 年代、锚点是被选出来的，选择效应把预测整体抬高——该弱点在预注册时
  已写明。**下次从「被挑选的子集」外推到全集时须先处理选择效应。**
- **RQ016B / RQ016C 两轮已完成并经监督方独立复算放行（2026-08-04T02:03:12Z）。**
  这一条替代了 RQ016C-H2 / RQ016C-H1 / RQ016B-F2 / RQ016B-F1 四条 `WAITING_ON_COMMANDER`
  状态条目，四个 agent 的原始交付内容完整保存在下方执行记录目录里。
  **背景一句话**：在线验证的判定串联两道弃权机制——机制一判断某一帧的 IPV 数值能否估出
  （RQ015 已冻结），机制二拿它与人类参照分布（envelope）比。RQ015/RQ016 只用了 InterHub
  的人类数据，而要判的自动驾驶车在 WOD 与 OnSite 里。这两轮回答「能不能用、怎么用」。
  **产物落位**：`reports/studies/RQ016B_wod_onsite_feasibility/RQ016B_1_feasibility_20260804T001351Z_7480c173/`
  与 `reports/studies/RQ016C_human_only_envelope/RQ016C_1_human_only_envelope_20260804T005716Z_7480c173/`；
  知识层 `reports/knowledge/RQ016B_wod_onsite_feasibility/` 与 `reports/knowledge/RQ016C_human_only_envelope/`。
  **RQ016B 结论**：直接套用**不可行**——WOD 与 OnSite 一行都没有七候选 MSE
  （`mse_0..6`/`status`/`reason_code` 非空计数全为 0），机制一判不了。WOD 本地只有 4 列 906 行、
  29 个 M2 特征全 MISSING，需重做脱敏投影且触及 RQ014 致盲边界，**PI 已裁定本轮放弃 WOD**。
  OnSite 可行：67,861 行 AV 锚点、29 个 M2 特征一个不缺、类别取值 100% 被 InterHub 覆盖，
  缺的只有 materializer。**另查实 RQ016 的 envelope 里 10.9009%（69,288/635,618）的目标值
  是自动驾驶车自己的 IPV**（`target_ipv_future` 取自 ego 一侧，而 `ego` 是 AV 的专属 track id）。
  **RQ016C 结论**：据 PI 2026-08-04 裁定（envelope 是查询机制，不同目标可建不同 envelope），
  只用纯人-人 2,442,625 行重建了供 OnSite 使用的参照 envelope。90% 层 coverage 0.898038
  （414,837/461,937）、平均宽度 1.238468、机制二弃权 5.0801%（24,723/486,660）。
  特征集较 RQ009 M2 移除 `agent_type_pair` / `av_included` / `vehicle_type_list` 三列——
  车辆是否为自动驾驶车是被检验对象而非情境，且 OnSite 在这三列的取值在人类训练池中从未出现。
  **产物已在真实 OnSite 全量 67,861 行上跑通打分（只加载不重拟），支持门通过
  21,936/67,861 = 32.3249%**，逐格从 `F|priority` 47.03% 到 `CP|equal` 0.00%。
  ⚠ **该演练只证明管线可运行，不构成对任何一辆自动驾驶车的判定**——OnSite 无机制一判据。
  **未入库产物**：拟合模型本体 164 MB 在
  `.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl`
  （sha256 `bc25302b4a7a307e3c73b3429b880e3cfda59074fc80850a732a93a67ef75de2`），可由已入库脚本重生成。
  **后续状态更新**：materializer 动工前的范围与参考线合同已由 RQ017 v4 裁定并执行：
  范围选 B 全 timing-valid anchor 67,861，参考线合同沿用 observed-trajectory fallback
  （OnSite dense 源表真实 map/lane/route/reference-line 字段为 0/274,022）。
  **已知边界**：无同源迁移证据（RQ009 LODO 4 个留出源均不含 OnSite 与该 WOD 产物，
  90% coverage 波动 0.7484–0.9921）；OnSite 有 7 行坐标系异常
  （`relative_distance_anchor` ≈ 570,762 米）真正分析前须处理；
  `apet_online_proxy` 填充率 OnSite 7.90% vs InterHub 40.26%。
- **RQ016 机制二 envelope 重建已完成并经监督方独立复算放行（2026-08-03T15:34Z）。**
  这条替代了同日 13:48Z 那条「等待监督方核数」的状态，A1 原始交付内容仍完整保存在
  执行记录目录里。
  **背景一句话**：在线验证的判定由两道串联弃权机制构成——机制一判断某一帧的 IPV
  数值能否估出（RQ015 已冻结），机制二判断人类样本是否足以判断该车偏离。机制二依赖的
  人类参照分布此前建在含伪零的样本上（旧估计器数值下溢时退回七候选等权，写出 IPV
  恰为 0，使「没估出来」与「恰为中性」不可区分）。本轮重建它。
  **产物落位**：`reports/studies/RQ016_human_envelope_rebuild/RQ016_1_envelope_rebuild_20260803T134808Z_d23fa836/`
  （报告、`01_results/key_numbers.json`、`02_process/` 下的可复跑脚本与任务书与裁定记录）；
  知识层 `reports/knowledge/RQ016_human_envelope_rebuild/`；
  轨道原始工作区 `.codex-fleet/rq016-envelope-rebuild/`。
  **主要结果**：在 `development + guard` 域同法跑两臂，唯一变量是样本口径。90% 名义层
  coverage 由 0.898832（758,857/844,270）变为 0.902689（545,159/603,928），
  平均区间宽度由 1.016189 变为 1.300967，即 **+28.02%**；覆盖基本不变而区间宽近三成。
  两道门串联合并弃权 **32.0583%（284,964/888,892）**，机制一贡献 28.4932%
  （253,274/888,892）、机制二贡献 3.5651%（31,690/888,892）。
  **变宽的机制已由监督方独立证实**（独立于执行方的 conformal 实现）：目标
  `target_ipv_future` 的四分位距由 0.0493 变为 0.2017（约 4.09 倍）；恰为 0 的行由
  192,221/888,892 = 21.6248% 降为 99,908/635,618 = 15.7182%；另以完全不训练模型的
  边际分位数宽度复核，B/A 比值 80/90/95 三层为 1.3763/1.1896/1.0130，同向同量级。
  **实际含义**：旧 envelope 偏窄，会比人类数据本身所支持的更频繁地把一辆车判为「不像人」。
  **必须一并引用的边界**：(1) 与 RQ009 已发表数**不构成复现关系**——其 test 域含
  RQ007 held_out，本轮受红线约束仅 dev+guard；(2) 零点聚集只被减半未消除，
  `|y| < 1e-6` 占比 A 臂 42.39%、B 臂 29.63%；(3) 支持门用 12 项距离特征而非 RQ009
  原门 15 项（排除 3 个由旧估计器算出的 `counterpart_ipv_*` 列，监督方已裁定接受）；
  (4) 描述性结果，不得写成因果主张；(5) 尚无 `decision.md`，无已接受手稿主张。
  **合规**：参与计算行中 `rq007_split` 不在 `{development, guard}` 的实测计数为 0
  （A1 与监督方各自独立测得）；未打开受保护 confirmation 划分文件；未改 `data/derived/`
  与 RQ009 原 run；未改五个受保护的估计器/管线/配置文件。
- **注意：RQ015 收官报告 §10 第 3 项「K2 `INTERFACE_NOTE.md` 的 23.40% 待订正」已过期。**
  该订正在 2026-08-03 就已执行完毕，`INTERFACE_NOTE.md` 与 K2 报告 §9 都已带订正块、
  原文保留。仅 `K1_preflight_and_plan.md:303` 仍有旧措辞，属历史计划文档，按「保留错误
  历史」原则不动。引用 `ipv_log = 0` 比例时用普查值 5.0097%（175,458/3,502,340）等，
  不要引用 23.40%（那是 J 轨锚点样本 238/1,017）。
- **RQ015L 收官轮 L1/L2/L3 均已交付，全轨转 `WAITING_ON_COMMANDER`（2026-08-03T03:47Z）。**
  合并报告为 `.codex-fleet/rq015l-consolidate/board/reports/RQ015_consolidated_report.md`
  （L3 撰写，leader 复核后就地纠正两处，见下）。leader 派出 L1 pid 98586、L2 pid 98772
  并行执行（各约 19/15 分钟），归队后派 L3 pid 8350 成文（约 4.5 分钟）。
  **leader 复核发现并处理的两件事：**
  (1) L1 报了 29.78%（81,548/273,819）的 join miss 却未定性。leader 用只读元数据补查得：
  **整案级排除**——涉及 2,270 个 case，出现在 K2 台账 `case_id` 中的为 0/2,270，
  被部分覆盖的 case 为 0/7,576；命中的 5,306 个 case 与 RQ015E 记录的 dev+guard case 集吻合。
  **据此推断（非直读标签，未打开受保护 confirmation 划分文件）这些 case 属 RQ007 held_out。**
  因此**主口径分母改为 192,271**（零点原子中落在台账覆盖域内的行），不再用 273,819。
  证据：`work/L1_rq009_zero_atom_split/L1b_leader_selfcheck.md` 与 `L1b_joinmiss_diagnosis.json`。
  **须监督方裁定**：L1 曾对那 381,674 行统计 `y==0.0` 计数（81,548）并落盘，若 held_out
  推断成立即为跨界统计；且 RQ009 已发表的原子计数 273,819/1,270,566 本身就算在含这 2,270 个
  case 的 fold 上。leader 未删改证据，原样上报。
  (2) **「门后 23.40% 取 `ipv_log=0`」的分母是错的**，详见下方 RQ015K 条目的修订。
  **L1 主结论**：在台账覆盖域上，RQ009 精确零点原子里 48.0223%（92,333/192,271）不是中性
  IPV 点值，而是弃权情形下被写成 0 的数值，主体为 `NEAR_UNIFORM` 47.0638%（90,490/192,271）；
  真中性零为 51.9777%（99,938/192,271）。这直接回应 RQ009 自己关于零点原子的 Limitations 警告。
- **（历史记录，保留）L1/L2 交付当时的看板状态：**
  L 轨解决 RQ015 最后两项查证并成文：L1 判定 RQ009 的 273,819/1,270,566 个精确零点能否与
  K2 台账精确一对一连接并拆分真中性/弃权伪零；L2 查清 OnSite 274,022 行 `UNKNOWN` 的代码来源
  与输入支持状态；两者完成后才派 L3 写 `board/reports/RQ015_consolidated_report.md`。
  `bash .codex-fleet/launch_leader.sh L` 已通过修订后的 `detach_launch.py` 路径启动，leader PID
  `96939`、PPID `1`、PGID `96938`，核验 69 秒后仍存活，未复发旧 nohup 早退；`leader.log`
  仍为 0 字节符合 `claude -p` 完成前不输出的已知行为。`STATUS.md` 最近读取仍为 `LAUNCHING`。
  本线程已直接完成 L1 本地只读分析，交付目录为
  `.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/`：RQ009 `y == 0.0`
  精确零点 273,819/1,270,566；K2 `target_future` 精确左连接为 192,271/273,819 命中、
  81,548/273,819 未命中；命中行中 `status=OK` 为 99,938/273,819，`status` 非 OK 为
  92,333/273,819（`NEAR_UNIFORM` 90,490，`NO_IPV_EFFECT` 1,796，`SOLVER_FAILURE` 47）。
  `L1_report_section.md` 末尾状态为 `state: WAITING_ON_LEADER`。
  L2 本地只读分析也已交付，目录为
  `.codex-fleet/rq015l-consolidate/work/L2_onsite_unknown/`：OnSite K2 台账 281,268 行中
  `UNKNOWN` 为 274,022/281,268，全部带 `source_reason_code=EMPTY_CELL_UNEXPLAINED`；
  这些行的 dense 源表轨迹、配对 ID、位置、速度、heading、距离与相对速度字段均为
  274,022/274,022 非空，但真实地图/车道/reference-line 字段为 0/274,022。L2 判断为既有
  OnSite 生成合同下默认 bounded-anchor 流程未覆盖大多数 dense role 行，不是轨迹或配对字段普遍缺失；
  RQ015A 旧口径的分子分母与 K2 来源状态一致，均为 2,974/281,268。`L2_report_section.md`
  末尾状态为 `state: WAITING_ON_LEADER`。L3 当前不由本线程判定。
  本轨仅使用本地既有产物，不投 Slurm、不重算 K2 join、不修改 RQ009。
- **RQ015K K2 全语料收尾已由 K2-2 完成，当前等待 commander 复核（2026-08-03）。**
  K2 materializer 已生成并回取本地：`data/derived/rq015k_logdomain_gate/`（约 1.7G，
  510 个 L1 parquet 分片、510 个 manifest）。远端权威目录为
  `/share/home/u25310231/ZXC/sociality_estimation/work_dirs/INFRA/rq015k_k2_fullcorpus_finalize_20260802T175006Z/`。
  K2-1 曾以 `final_status=FAIL`、`blockers=g_anchor, solver_failure_threshold` 结项；监督方
  `2026-08-02T19:12:54Z` 逐项复核后裁定两条 blocker 都不成立：`g_anchor` 是把 HPC 产物错比到
  RQ015B Mac 基线，`solver_failure_threshold` 是未在 nuPlan Vegas 校准的单片 tripwire 并已撤销。
  K2-2 只做四项收尾，不重跑求解、不重跑 join、不改阈值：指定基线路径改为
  `.codex-fleet/rq015g-hpc-resolve/work/anchor_mse_hpc.csv`；正确 G-HPC anchor 校验
  `anchor_rows=2300`、`compared_rows=2300`、`max_abs_diff=0.0`、`first_mismatch=null`；
  RQ009 join `canonical_key` 去重实测 `rows=8,994,736`、`unique_keys=8,994,736`、`duplicates=0`；
  1,934 行 `SOLVER_FAILURE` 已刻画为工程失败。主报告为
  `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K2_fullcorpus_gate_ledger.md`；
  机器证据新增在 `.codex-fleet/rq015k-fullcorpus-gate/work/k2_fullcorpus/validation/{g_anchor_hpc_baseline.json,rq009_join_key_uniqueness.json,solver_failure_characterization.json}`。
  `board/STATUS.md` 已刷新为 `state: WAITING_ON_COMMANDER`，**不得由 K2-2 自行转 DONE**。
  **当前边界**：下游只能用 `status` / `reason_code` 判别门状态；`ipv_log=0` 是合法且高频的通过门估计值，
  不能把数值 0 当作弃权。
  **⚠ 分母订正（track L leader，2026-08-03T03:44:10Z）**：此前写作「门后 23.40%」的说法**分母是错的**。
  23.40% 的实际出处是 **J 轨锚点样本 238/1,017**，不是全语料普查值。
  **InterHub 门后通过行（分母 3,502,340）的普查值是：`ipv_log` 恰好为 0 → 5.0097%（175,458/3,502,340）；
  `abs(ipv_log)<=1e-9` → 9.9516%（348,539/3,502,340）。**
  机器证据：`.codex-fleet/rq015l-consolidate/work/L1_rq009_zero_atom_split/L3b_ipvlog_zero_census.json`。
  该错误分母同时存在于 **K2 报告 §9 与 K2 的 `INTERFACE_NOTE.md`**（下游要读的那份）；
  **track L 未改 K2 任何文件**，是否回改待监督方裁定。结论方向不变、且更硬：即便按最严口径，
  门后仍有 5.0097% 的通过行取 `ipv_log=0`。
- **RQ015K K1 勘察与 K1b 单-PKL内存/并发 pilot 已完成（2026-08-02）；K2 后续已由 PI 单独授权。**
  K1 交付报告为
  `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md`，
  支撑脚本与证据在 `.codex-fleet/rq015k-fullcorpus-gate/work/`。K1 只提交了一个
  小批 Slurm pilot，job id `2068610`，未提交全量作业、未提交 git commit、未改受保护估计器文件。
  报告结论是：InterHub/RQ009 可进入下一步规划，但 OnSite/WOD 需要先明确新 materializer
  或工程状态处理规则；K2 千万行级重算需监督方另行放行。
  K1b 交付报告为
  `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md`，
  证据目录为 `.codex-fleet/rq015k-fullcorpus-gate/work/k1b_memory_pilot/`。K1b 只提交了
  一个合并小批 Slurm job `2068976`，在 `waymo_0-299.pkl` 上各跑 1,120 单元的 P6/P10/P16；
  三配置 `mse_per_candidate[7]` 逐位一致性通过，且与 K1 pilot 重叠 72 行通过。推荐的 K2
  InterHub shard 形状为单 PKL + row-key range、16 workers、`--mem=64G`；在 36 个 intel
  节点 + fata02 的题面快照下为 228 个并发位、3,648 核、预计 1.02 小时。K1b 未提交全量作业、
  未提交 git commit、未改受保护估计器或 `configs/ipv_sigma01_exact.json`；K1b 的推荐本身不构成
  K2 放行，当前授权来自后续 PI 明示与修订后的 `K2-leader-kickoff.md`。
- **RQ015J J1 弃权门规格与全域影响 design-based estimate 已完成（2026-08-02T04:11:51Z）。**
  权威复审报告为
  `.codex-fleet/rq015j-gate-spec/board/reports/J_plan_review.md`，复算脚本与机器证据为
  `.codex-fleet/rq015j-gate-spec/work/{j_plan_review_compute.py,j_plan_review_compute.json}`。
  J1 交付报告为
  `.codex-fleet/rq015j-gate-spec/board/reports/J1_gate_spec_and_impact.md`，
  J1 复算脚本与机器证据为
  `.codex-fleet/rq015j-gate-spec/work/{j1_gate_spec_compute.py,j1_gate_spec_evidence.json}`。
  J1_DONE 时间戳为 `2026-08-02T04:11:51Z`；本次未提交 git commit、未跑 HPC、
  未改 `agent.py` / `ipv_estimation.py` / `process_interhub.py` / `reliability_logdomain.py`。
  两项必须订正的计划事实：(1) `anchor_mse.csv` 全 2,300 行按 `k_eff_log` 统计，
  6.75--7.00 格为 1,166/2,300=50.6957%；766 仅在额外排除 400 个
  `mse_per_candidate[7]` spread=0 锚点后成立；(2) 门后 `at_grid_boundary` 为
  nuplan 79/312=25.3205%、waymo 398/705=56.4539%，原 1%/20% 实为
  `ipv_log` 精确命中 ±3π/8 端点的 3/312=0.9615%、143/705=20.2837%。
  样本内未加权保留率为 1,017/2,300=44.2174%；按 `mechanism_split.csv` 的
  `ht_weight`、全域分母 2,646,058 做设计基估计后为
  1,885,831.096/2,646,058=71.2695%，cluster bootstrap 95% CI
  [67.1729%, 75.2135%]。复审确认 2,300 锚点文件不含 RQ009 的 29 个 context-only
  变量；修订后的 J 任务已取消本轮上下文分格，改为只按锚点自带的
  `signature` / `n_band` / `n_obs` 汇报，并把上下文分格留给 RQ009 应用门时处理。
  这表示不可执行要求已删除，不表示缺失字段已经补齐。复审未提交 commit、未跑 HPC、
  未改受保护估计器代码。
- **RQ015A A1(r3) concentration audit 已执行完成（2026-07-31）。**
  当前 canonical run 目录：
  `reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/`。
  `run_receipt.json` 机器判定 `PASS`，`held_out_parsed_rows=0`，
  四个本地可审计产物全部落盘 parquet L1 台账：
  OnSite `281,268` measurement 行、WOD full479 `906` 行、sigma01 `5,197,072` 行、
  RQ009 feature matrix `8,994,736` 行，合计 `14,473,982` 行。A2 科学交付物
  `bounded_report.md` 已写入同一 run 目录，并同步生成 `figures/fig1`–`fig4`
  的 PNG/PDF 与 `usable_subset.csv`（主判据 19,778 个 case/episode key，
  3,049,608 个 ATTEMPTED 行，占 14,473,982 行分母的 21.27%）。后续若只需复核报告，
  不要重跑审计。
  当前执行绑定为 run spec v7 + schema v4 + 新清单
  `reports/plans/RQ015A_plan_v10_checksums_20260731.sha256`；v9 是执行接线前的历史清单，
  不要覆盖。最终验证：
  `PYTHONPATH=src /Users/xiaocong/.rq009_codex_fleet/venv/bin/python -m pytest -q tests/test_rq015a*`
  → `269 passed in 6.57s`。
  本轮修复：`run_rq015a.py` 接出授权成功后的真实执行路径；`build_ledger.py`
  最小修复 OnSite CSV 整数字符串 local-position 解析与 WOD provenance `K_source.value=7` 绑定。
  A2 追加披露：`run_receipt.json` 顶层 `reads_measurement_fields=false` 是从 validate 路径继承的误标，
  真值在 `metadata.execute_measurement_fields_read=true`；已追加到已知问题清单，留待下一次正当触碰
  `run_rq015a.py` 时修。
- **RQ015A pre-execute implementation package historical note（已被 A1(r3) 执行结果取代）。**
  A1(r3) 之前的状态是：`execution_authorized` 已翻为 `true`，但审计仍无法运行，
  因为 `run_rq015a.py` 的 `--execute` 在许可签发成功后仍无条件抛出
  `refusing to run audit without PI-reviewed post-authorization handoff`。
  该历史状态不再是当前事实；保留下方背景仅用于理解为什么需要 A1(r3) 接线。
  v3 复审的核心 blocker 是"完整 ledger builder / factor / bootstrap / validator / receipt 不存在"，
  现已全部交付并经**七轮独立健壮性审计**（每轮 agent 对此前所有审计与修复轮盲）收敛至零 blocker，
  再经**三路最终独立复审**（技术／显著性／可执行性）全部 `PASS_WITH_CONDITIONS`，条件已闭合。
  **`rq015a_concentration_audit` 的 `execution_authorized` 仍为 `false`、`allowed_operations` 为空、
  `authorized_package_commit` 为 `null`；审计从未运行，held_out 的 measurement 列零解析。**
  - **当前权威制品**：计划 `RQ015A_plan_v7_concentration_audit_20260730.md`；
    运行合同 `RQ015A_run_spec_v6_20260731.json`；台账 `RQ015A_ledger_schema_v4_20260731.json`；
    清单 `RQ015A_plan_v8_checksums_20260731.sha256`（20 项，自校验 OK）；
    已知问题清单与审计边界声明 `reports/knowledge/RQ015A_ipv_estimability_labelling/known_issues_and_audit_boundary_20260730.md`。
    **旧版本 v1–v6 一律未被改动**（R11）。
  - **验证命令**：
    `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python -m pytest tests/test_rq015a_contracts.py
    tests/test_rq015a_build_ledger.py tests/test_rq015a_validate_receipt.py
    tests/test_rq015a_factor_analysis.py tests/test_rq015a_run_entrypoint.py -q`
    → **256 passed**（干净检出无 gitignored `data/` 时为 255 passed + 1 skipped，属预期）。
    `run_rq015a.py --validate-only` → exit 0 `machine_verdict=PASS fixture_total_passed=256`；
    `--execute` → exit 1。**`--run-spec` / `--schema` 不再硬编码**，
    分别从授权对象的 `run_spec_path` 与该 spec 的 `bound_artifacts.ledger_schema` 推导。
  - **执行授权是三重条件**：双键（`execution_authorized` + `allowed_operations`）、
    run spec 路径绑定核对、以及 `authorized_package_commit` 必须等于当时的 git HEAD。
    **翻转只能由 PI 手动做，且必须是最后一个动作**——其后任何提交都会使 commit 绑定失配。
    PI 已选择方式 **B**（指挥者准备、PI 执行）。翻转后应重跑一次信任边界检验：
    复审方明确记录"授权翻转后的端到端行为是**推断的、不是实测的**"。
  - **WOD 取回已完成（PI 2026-07-31 批准）**：HPC 侧按 4 列白名单
    （`segment_key` / `candidate_index` / `ego_ipv` / `ego_ipv_error`）投影 → 净化凭证 →
    传输前五项校验 → 传输 → 本地四项复核，全部通过。落地
    `data/derived/wod_e2e/rq015a_full479_projected/`（906 行，CSV SHA `d10c3a6f…30b7d1`，
    被丢弃 61 列含 `rating`，禁词扫描命中 0，数据在 gitignore 内不进版本控制）。
    `wod_rq010b_full479_audited` 因此升为 **`L1_DIRECT`**；K = 7 由三环证据链确定
    （运行 `stats.json` 记 `ipv_solver_mode: fast` → `ipv_estimation.py:220-223` fast 为七候选
    → `agent.py:63-64` 七点网格），故 `q_eff` 可算。
    **另两个 WOD/RQ014 产物仍未取回**，报告须在标题级披露该覆盖缺口，不得表述为"全语料"。
  - **已知未修（随包提交，复审方评价披露充分）**：同形字符列名仍可绕过结构列 denylist
    （实测西里尔 `rаting` / `scоre` 通过；覆盖需定义 confusable 映射范围，属独立决策）；
    D0 的 `NOT_ATTEMPTED` 行保留非空 `q_eff`/`k_eff`（下游按 `attempt_status` 过滤，未见污染）。
    **方法学 caveat 已自我披露**：审计 1–6 的 prompt 为逐代 sed 派生，第七代膨胀致 agent 挂死，
    故其指令一致性可能已被稀释且无法排除影响；零 blocker 那轮用的是重写后的干净稿。
- **RQ015 已按 PI 决策 2026-07-26 拆分为 RQ015A / RQ015B；合并版 v1/v1.1/v1.2 仅作历史记录，不再是执行依据。**
  拆分依据：合并计划三轮独立复审均 BLOCKED，规格面积扩张快于闭合（见
  `reports/plans/RQ015_plan_v1p2_amendment_20260726.md` §A6）。历史复审记录保留于
  `reports/knowledge/RQ015_ipv_estimability_contract/reviews/`。
  - **RQ015A v3 三路独立复审完成（2026-07-26）— `BLOCKED / REQUEST_CHANGES`**：
    计划 `reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`（SHA-256
    `75912bc1433a5efb5b0520af492e27579e9a1f6652074d3f37eb3a77befff264`），基线 manifest
    `RQ015A_plan_v3_checksums_20260726.sha256` **6/6 OK**。三路为 R1 4B/3M/1m、
    R2 2B/3M/2m、R3 2B/3M/0m；综合在
    `reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_three_reviewer_synthesis_v3_20260726.md`。
    **已接受并须保留**：continuous `q_eff` primary、bins 不进入 episode/C0、三恒等式骨架、
    OnSite local-position、`sorted + math.fsum` 和 L3 `ZERO_SUPPORT` 不填 0。
    **仍阻断 Formal G1**：PI rederivation condition 未被 append-only supersede；run spec 无 exact
    command/entrypoint、授权 fragment 不存在、split 未绑定；逐产物 path/hash/key/role 与真实数据
    不一致（含 RQ014 wrong key、OnSite `case_key`、M3 collapse/role 混淆）；invalid
    `ipv_error` 可产生有效 q；L2/L3 可跨 artifact pooling；C0 无 q 可返回 NO_TRIGGER；完整
    ledger builder/factor/bootstrap/validator/receipt 不存在。显式外部 venv 下合并测试 **52 passed**，
    但 declared stdlib environment 不含 tests 所需 `pytest`，故该通过不等于 validate-only 可复现。
  - **PI 裁定 2026-07-26 — 按路径 A 推进（把实现写完再送最后一轮复审）**，执行中可调用 codex
    做边界明确的任务。另两项裁定：(a) 修订 plan §9，**授权对三个 WOD/RQ014 产物做只读取回**；
    (b) feature matrix 的 **M4_ONLY_ego_self_anchor 通道排除**，`expansion_factor` 固定为 2。
  - **预执行合同核验完成（2026-07-26，对真实文件；只读结构，未解析任何 `ipv_*` 数值）**：
    `reports/knowledge/RQ015A_ipv_estimability_labelling/preflight_contract_verification_20260726.md`，
    可复现脚本 `scripts/rq015a/preflight_structural_scan.py`。得 **C1–C14 共 14 项修正**，
    已吸收进 `reports/plans/RQ015A_ledger_schema_v2.json`（不覆写 v1）。三条要害：
    **C6（安全）** RQ009 的 fold `{train,guard_tune,calibration,test}` 与 RQ007 的 split 正交，
    每个 fold 都含约 29% held_out；按 fold 过滤会解析 **1,899,898 行 held_out**——必须先按
    `case_id` 白名单过滤再读 measurement。
    **C3/C4/C5（产物指认错）** `rq009_m3_predictions` 的 15 列**无任何 `ipv_error`**；三角色实为
    feature matrix 的列；3× alpha 折叠只属于 predictions，feature matrix 为 `E=2 / C=1`；
    v1 的 `anchors_dev_guard=1778594` 无法复现已删除，实测 dev+guard **4,497,368** 行。
    **C1（恒等式）** sigma01 的 2,490,992 是**已排除 D0** 的数，用作 identity_1 基数会使
    identity_2 的 `NOT_ATTEMPTED` 恒为 0；改为 physical **2,598,536** / measurement **5,197,072** /
    NOT_ATTEMPTED **215,088**。
    另：**C14** WOD 三产物（RQ010B full479、phase1b schemeB、RQ014 anchor scores）本地全部缺失，
    可审计范围实为 3/6。fixtures 已修至 **20/20**。
  - **交接手册（新线程从这里开始）**：`reports/plans/RQ015A_execution_handoff_20260727.md`。
    含 12 条铁律、已冻结常量与逐产物实测事实、T1–T11 待办与验收标准、立即停止条件。
  - **编排合同（PI 裁定 2026-07-27）**：接手方**只做指挥者**——分解工作流、写 prompt、
    判定可信度、最终综合；**任何边界明确的执行任务一律交给 codex CLI**
    （`gpt-5.5` + `xhigh`，并行后台，写代码的 agent 用 `--worktree`）。
    **fleet 目录已移出 OneDrive** → `~/.codex-fleet-local/rq015a-implementation/board/`
    （原 `.codex-fleet/rq015a-implementation/` 已不存在）。移动原因：worktree 建在
    OneDrive 同步目录内会使路径超长（实测 333 字符，OneDrive 拒绝同步），
    **后续所有 `--worktree` agent 的 fleet-dir 都必须用该本地路径**。
    同理 `.codex-fleet/rq014-execution-v1p6/agents/` 也已移至
    `~/.codex-fleet-local/rq014-execution-v1p6/`；该 fleet 的 `board/`（含 `w4g_evidence/`）
    **仍留在仓库内**，本文件其它位置对它的引用依然有效。
    board 内容：`plan.md`、`module_interface_v1.md` 与指挥者裁定
    `module_interface_v1_commander_addendum.md`（6 条强制修正）、
    `prompts/`（W0–W7 共 9 份）、`reports/`（各 agent 的有界结项报告）。
    交叉验证已执行：`W7-replicate-conservation` 盲算守恒数字**与冻结事实零分歧**；
    `W6-red-team`（专找 fail-open / 过滤顺序 / pooling / 非确定性）进行中。
  - **HPC 工作已移交接手方**（PI 裁定 2026-07-27，接手方有权限）：T11 只读探测
    `bash scripts/rq015a/hpc_probe_wod_targets.sh > rq015a_wod_probe.json`，
    规格 `reports/plans/RQ015A_wod_retrieval_spec_v1.json`。**致盲危险 HIGH**：
    传输前须在 HPC 侧做列投影 + sanitization receipt；**探测≠取回，取回需单独授权**。
  - **T9 已由 PI 裁定解除（2026-07-27）**：`sealed_exposure_disclosure_20260726.md`
    新增 **§8**（append-only supersede，§6 原文一字未改），正式解除 §6「附加条件」
    的三条——即"两阈值须从 dev+guard 重导出 / 导出规则须先冻结登记 SHA / 重导出前
    不得产出结论画像"，并撤销 `PROVISIONAL_PENDING_DEVGUARD_REDERIVATION` 标记。
    解除理由：`4/7`（⇔`ipv_error=0.5`）与 `0.93`（⇔`ipv_error=0.608069099165`）
    已由科学阈值降为**报告用 policy bins**，而 R6 + `test_c0_routing_never_consumes_report_bins`
    已在代码层强制 bins 不进入任何判定，该条件已丧失保护对象。
    **不在解除范围**：§6 判读 A 与记录豁免、§7 措辞精确化（扫描程序确实解析并聚合过
    held_out 逐行字段）、§7 三条治理动作、R1/R2/R3、以及 `execution_authorized` 仍为 `false`
    ——本次解除**不构成任何执行授权**。
    文件 SHA-256 由 `aabbd0d6…4ab24` 变为 `6c904b806e28bb4d940db145bd365287fa23287ddd22881caa41bc8c44439f54`；
    v1/v2/v3 manifest 为各自复审时点的历史快照，v4 包须由 T10 重新登记。
    签署状态 **`RECORDED_ON_PI_RULING`（已生效，无待签事项）**——PI 于 2026-07-27 明确选择
    以「会话裁定 + 指挥者记录」形式生效，不留签署栏，避免空置签署栏成为后续复审的未闭合项。
  - **RQ015B — 估计器修复与 verifier 弃权闸**：
    `reports/plans/RQ015B_plan_v0_estimator_repair_and_abstain_gate_20260726.md`。
    B1 log 域改写（`w=softmax(−MSEᵢ/2σ²)`，平价门 ≤1e-12）；B2 正交结果契约 +
    **生产兼容层三项（未交付，不得接线）**；D1–D4 可执行分类器（D1 定义为
    "legacy 结果是否被改变"，D2 更名 `D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL`、
    禁用"固有不可辨识"）；`min_mse_misfit := Q_0.99(min_mse)` 在 dev+guard 上冻结
    （sealed 禁止参与）；B3 σ 仅在证据支持时执行；**部署前必须通过 gate-pass 条件
    覆盖审计**（RQ009 test fold 独用、4 分层、case-cluster bootstrap B=2000
    seed 20260726、点估计 ±3pp 且 CI 下界 ≥ nominal−5pp）。
    实现现状 `BUILD_WHILE_DENY`：`src/sociality_estimation/core/reliability_logdomain.py`
    + `tests/test_rq015_reliability_logdomain.py` **36/36 通过**；legacy 未改、未接线生产。
  - RQ009 hw4 的含-sealed 立项基线：有效 agent-value（7,086,138）近零 **41.2794%**、
    `err≥0.61` **52.5810%**、`err≤0.50` **24.1688%**；不得泛化为所有估计器配置或
    最终画像。该 InterHub 产物的 D0 warm-up 占位为 305,824 个 agent-value
    （=38,228×4×2，原"K≥9 网格混入"结论已证伪）；下溢临界 RMS
    n=5 为 1.6915/1.7336 m、n=11 为 1.1470/1.1752 m；复现脚本
    `reports/plans/prompts/RQ015_portrait_scan_v1.sh`。
  - **措辞订正**：并非"每个测不出的帧都判合规"——冻结 M3 test fold 90% nominal 支持域内，
    `|y|<1e-6` 近零行有 **520,826/522,219 = 99.7333%** 的区间包含 0（约 0.27% 不含）。
  - 两个 RQ 均 `execution_authorized=false`；RQ015A v3 已三路独立复审并 BLOCKED，
    `formal_g1_eligible=false`；RQ015B 仍待其自己的独立复审。RQ015A v3 复审包为
    `reports/knowledge/RQ015A_ipv_estimability_labelling/reviews/rq015a_review_manifest_v3_20260726.sha256`。
- **RQ010B COMPLETE (2026-07-03; 10Hz sensitivity closed 2026-07-04) = bounded NULL.** Reframed WOD-E2E human-preference
  validity: candidate IPV does NOT predict human preference and is not comparable to
  physics (Scheme 1 future-only n=75 rho=0.148 p=0.10; Scheme 2 history+future >=1s
  n=98 rho=0.031 p=0.69; max-stat permutation p=1.0 both). M3 does NOT transfer to
  WOD-E2E (<=15% in-support) -> path-type HV norm. Review PASS, red-team null ROBUST,
  replication exact. Report `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f/90_report_reframed_preference/index.html` (+`.zh.html`);
  decision `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`. Full
  pipeline on HPC `/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/` (retained). The PI-flagged
  4Hz->10Hz caveat is now checked under
  `/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase_10hz_sensitivity/`: re-estimated
  candidate IPV at dt=0.1 with no counterpart extrapolation and joined ratings only for
  the final test. Null holds at 10Hz (Scheme 1 n=75 rho=0.165 p=0.0626; Scheme 2 10Hz
  effective n=47 rho=0.128 p=0.241; max-stat p=1.0 both; IPV-vs-4Hz Spearman 0.308/0.289).
  Deliverables: `candidate_ipv_10hz.csv` and `tenhz_sensitivity_report.md`. No active
  RQ010B compute; token relay stopped.
- RQ012B Stage 4/5 deviation-to-harm association and negative-control battery is
  complete for the expanded all-valid frozen-M3 OnSite deviation table. Analysis
  set is the pre-registered gate-passing units: `n=245` units across 19 teams;
  exclusions are 18 replay-eligible units that failed IPV/anchor build before
  deviation plus 22 built units with no gate-passing anchor. Final verdict is
  `NULL`: no primary objective harm co-primary deviation effect is reliable
  after BH-FDR or label-permutation control, and none passes the full
  stage-5 battery. Primary co-primary effects: official_safety
  `frac_outside_90` increment `1.1595e-05`, 95% CI
  `[9.728e-08, 0.0013195]`, permutation p `0.7429`, q `0.999`;
  official_safety `max_abs_exceedance_90` increment `0.0001303`, 95% CI
  `[6.687e-07, 0.0013279]`, p `0.6485`, q `0.9947`;
  collision/intervention indicator increments were effectively zero with p
  `0.3845`/`0.6941` and q `0.9422`/`0.999`. W0 event-count associations are
  secondary only; E16 sparse event rows included some low p-values, but
  automatic-event counts alone are not a scientific outcome and several
  controls failed. Stage 4b full interaction-failure consequence battery is now
  complete over the 8 non-inert automatic behavioural manifestations, 4
  behavioural groupings, and 4 official subscores with kinematic-only +
  exposure baseline, seed `20260628`, 5,000 team-block permutations, and 300
  team-cluster bootstraps. Full-battery verdict is `BOUNDED` with `0`
  SUPPORTED endpoints: strongest powered channel is NEAR-MISS/CONTACT
  `max_abs_exceedance_90` IRR `1.2239`, 95% CI `[1.0314, 1.3450]`,
  permutation p `0.0018`, BH q `0.05119`, baseline-incremental and beating
  placebo/label but failing M2; E09 near-miss similarly has IRR `1.2329`, p
  `0.0018`, q `0.05119` but fails placebo and M2. E16 no-progress/deadlock is
  bounded and control-passing (IRR `1.4967`, p `0.002599`, q `0.05119`) but is
  explicitly UNDERPOWERED. No official subscore or abrupt/discomfort channel
  passes BH-FDR/control requirements, and no interaction-failure channel is
  IPV-specifically supported by deviation. Full-battery artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4b/full_battery/{full_battery_results.csv,endpoint_summary.csv,negative_control_results_full_battery.csv}`
  and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/{harm_association_full_battery_report.md,results_full_battery.json}`.
  Earlier Stage 4/5 artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4plus/` and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/{harm_association_report.md,results.json}`;
  stage-5 detail:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/05_negative_controls/negative_control_detail.csv`.
  Publication figure package for HA-1/HA-2/HA-3 plus the intuitive Stage-4b
  partial-rank Fig. 4:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/01_results/figures/`
  with PNG/PDF/SVG figure groups, per-panel source CSVs, manifest, and plotting
  scripts. Latest added figure:
  `fig4_deviation_vs_failures_intuitive.{png,pdf,svg}`, computed from
  `unit_analysis_table.parquet` as exposure-controlled partial Spearman
  correlations with event rates and `100 - official_score`; all point estimates
  are positive but weak, and too-passive lower-tail deviation is larger than
  too-aggressive upper-tail deviation in 9/10 displayed consequences.
  Bilingual offline-openable report package:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/90_report/index.html`,
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/90_report/index.zh.html`,
  and entry page
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/00_entry/index.html`.
  Independent blind replication by a different route also reproduces NULL:
  team-block outcome-profile permutation plus exposure-controlled rank/logistic
  tests gave official_safety p `0.0762`/`0.2529` for
  `frac_outside_90`/`max_abs_exceedance_90`, collision/intervention p
  `0.3421`/`0.8956`, all `AGREE`; artifacts:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/08_replication/`.
  Independent full-battery replication/red-team recheck also `AGREE`s with the
  bounded/null story while adding two wording caveats: the displayed/powered
  consequences have uniform positive worse-direction partial-r signs, but the
  full 16-endpoint family has sparse underpowered E18/E19 exceptions; and the
  strict `partial r <= 0.17` shorthand holds for the displayed simple-rank view
  but not for the NEAR-MISS/CONTACT grouping (`r=0.205`, still small). E16
  lower-tail deadlock remains the only all-control-passing row in the
  independent Poisson check (M3 increment `0.03817`, M2 `0.03606`, placebo
  `0.01534`, within-team permutation p `0.0010`, 52 units with E16>0), but is
  underpowered and published BH q is `0.05119`; near-miss and
  NEAR-MISS/CONTACT max-exceedance lose to M2 (`0.04335` vs `0.04536`, and
  `0.04879` vs `0.05145`). Recheck artifacts:
  `data/derived/onsite_competition/RQ012B_event_harm/stage4b/recheck/` and
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/08_replication/full_battery_recheck_report.md`.
  Reproduce the original Stage 4/5 run with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/run_harm_association.py --seed 20260628 --n-permutations 5000 --bootstrap 300`.
  Reproduce the full-battery Stage 4b run with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/04_harm_association/run_harm_association_full_battery.py --seed 20260628 --n-permutations 5000 --bootstrap 300`.
  Reproduce the full-battery independent recheck with
  `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python data/derived/onsite_competition/RQ012B_event_harm/stage4b/recheck/recheck_full_battery.py --seed 20260628 --n-permutations 5000`.
- RQ012B Stage 3+ OnSite all-valid M3-anchor enabling build now has
  AV-perspective clean_285 anchors under
  `/Users/xiaocong/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet`:
  267/285 units covered, 67,861 anchors, 29/32 M3 required inputs fully
  populated, dense IPV rows at
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet`.
  This HPC build used pinned legacy estimator HEAD `5edd2810` with
  process-pool Slurm job `1710800` on one AMD 192-core node. The expanded
  frozen-M3 OOD/support gate and deviation scan is complete: gate pass
  19,044/67,861 anchors and 245/267 units; abstain 48,817 anchors. At the 90%
  band, 840 gate-passing anchors across 149 units are out-of-band; 80%/95%
  counts are 2,475/447 anchors and 193/116 units. Per-unit max absolute 90%
  exceedance is >0 for 149 units (nonzero min/median/max
  0.00158/0.24593/1.06895). Abstention remains structural: distance over
  threshold for 47,166 category-eligible anchors and unsupported joint cells
  for 1,651 anchors; imputed-NaN distance features are common (64,040 anchors,
  45,301 abstainers) but are not a separate frozen-gate hard-fail. Stage 4/5
  harm association has now been run from this expanded deviation table; see the
  current RQ012B Stage 4/5 bullet above. Stage-3 gate report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/ood_gate_multi/ood_gate_multi_report.md`.
  Scored data:
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate_multi/`.
- Prior RQ012B Stage 3+ frozen-M3 OOD/support gate is complete only for the
  earlier one-anchor-per-unit 267-anchor OnSite AV build. As-is gate pass is
  51/267 anchors/units, with 216 abstentions;
  units with usable deviation are 51. Dominant frozen hard-fail causes are
  k=25 distance over threshold 1.6072176695 for 136 category-eligible anchors
  and unsupported joint cells for 80 anchors (`F|equal|AV;HV`=60,
  `CP|equal|AV;HV`=20). `priority_role=equal`, geometry levels CP/F/MP, and
  `agent_type_pair=AV;HV` are individually supported in RQ009
  `ood_gate.json`; `apet_online_proxy` NaN is common (184 abstaining anchors)
  but the frozen distance gate already uses RQ009 train-median imputation.
  Sensitivities: drop `apet_online_proxy` from distance gives 61/267 usable;
  train-median imputation and literal equal-supported are no-ops; drop-apet
  plus equal-supported is also 61/267. Report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/ood_gate/ood_gate_report.md`.
  Scored data:
  `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate/`.
- Primary technical context: realtime IPV estimator validation and InterHub
  CSV/pkl motion-data pipelines.
- Recommended online sign mode: `RealtimeIPVEstimator.for_realtime_sign(...)`
  with `history_window=10`, `max_workers=10`, and the five-candidate sign grid.
- Accuracy-preserving online value mode: `solver_preset="parallel_accurate"`
  with the legacy seven-candidate grid.
- The 20260612 sigma 0.1 full-rerun data source is now under
- **Primary active research:** RQ009 estimability-aware dynamic counterpart-conditioned human
  envelope. PI authorized launch; independent plan review is the first gate.
- RQ009 plan:
  `reports/plans/RQ009_plan_v0_dynamic_counterpart_conditioned_envelope_20260624.md`.
- RQ009 main-agent prompt:
  `reports/plans/prompts/RQ009_prompt_claude_codex_orchestration_20260624.md`.
- **RQ014 current execution recovery:** v1.5 was merged to `origin/main` by PR #5 at
  `a738de44715abb118e5571eec42af30d9b1c6786` (contract commit `24be08278adf43371fda14e7ec23a95b986b2fb1`).
  It restores immutable v1.3 bytes, legally closes G0
  with five PI-waived surfaces represented as `status=INACCESSIBLE`, and replaces ambiguous
  booleans with fail-closed managed operation authorization. Sixth-round fresh statistics and
  execution/governance reviews both returned `NO_BLOCKER`; machine validation accepts
  `RQ014_formal_G1_v1p5_20260712.yaml` as `FORMAL_G1_PASS`. The exact contract is published, and the
  authorized rating-blind declassification export has now been executed and dual-reviewed PASS
  (2026-07-13; see the v1.6 execution bullet below). PI accepted D1, and the subsequently authorized
  `rq014_g2_contract_preflight` completed PASS; no later RQ014 operation may run pending D2.
- RQ014 primary science authority is now
  `reports/plans/RQ014_recovery_lane_v2.json`: a fixed 960-cell rating-blind feature grid followed by a
  one-time, separately authorized 2,880-row full-data recovery screen and a clean independent replay of
  (2026-07-13; see the v1.6 execution bullet below). PI accepted D1 and authorized starting the
  authority-change loop for only `rq014_g2_contract_preflight`; no further RQ014 production operation may run yet.
- RQ014 review-candidate primary science authority is now
  `reports/plans/RQ014_recovery_lane_v3.json`: the PI-identified checksum-bound RQ009 M3 model is one
  frozen envelope input, giving a fixed 320-cell rating-blind feature grid followed by a one-time,
  separately authorized 960-row full-data recovery screen and a clean independent replay of
  the mechanically frozen rank-1 recipe. True causal-history, look-ahead-future, two-sided combined,
  t*-prefix and full-future semantics are distinct. Window-local state derivation forbids derivative
  halos; a checksum-bound 15,328-group per-scene anchor-domain contract is the sole membership authority.
  Legacy split/power/confirmation is optional and does not gate historical recovery.
- RQ014 first staged operation was `rq014_g2_declassification_export`, not scientific compute and
  not contract preflight. Raw rated479 TFRecords still embed `preference_score`, so the operation
  may read only the eight exact score-omitting Phase-1 bundles, structural readiness TSV, and
  selected counterpart CSV registered in
  `reports/studies/RQ014_wod_e2e_rating_recovery/02_g2_preflight/RQ014_declassification_source_inventory_20260712.json`.
  It emits only the canonical CSV/JSON schema under the managed input root. That export gate and
  the subsequent contract preflight are now complete; the current stop gate is D2 as recorded below. Historical kickoff:
  `reports/plans/prompts/RQ014_G2_kickoff_prompt_v1p5_20260712.md`.
- RQ014 multi-agent execution handoff is
  `reports/plans/RQ014_plan_v1p6_execution_handoff_20260712.md` (SHA-256
  `f007c290ea6bb1130b2df1b49c63e482e34cfc7147716f8d68dd4c918e81de0c`). It is an
  append-only operational supplement, not a new authorization and not part of the v1.5 Formal-G1
  bundle. At handoff publication, the Tongji managed checkout was `b1476bd0` and lacked the v1.5 contract; the Lead
  Agent first had to sync published Git history and detach at exact reviewed commit `24be0827`, then
  create/dual-review an immutable export spec and run validate-only. Waves 0–3 through the already
  authorized rating-blind export/bounded report could proceed without a new user decision. The first
  mandatory user/PI decision was D1 after export PASS evidence; preflight, compute budget, 960-cell
  bundle. At handoff time the Tongji managed checkout was `b1476bd0` and lacked the v1.5 contract; that
  historical sync instruction was completed before the later W4b/W4c work. The Lead then had to
  create/dual-review an immutable export spec and run validate-only. Waves 0–3 through the already
  authorized rating-blind export/bounded report may proceed without a new user decision. The first
  mandatory user/PI decision is D1 after export PASS evidence; preflight, compute budget, 320-cell
  feature build, rating join, clean replay and claim acceptance each retain later explicit stop gates.
  Independent execution/HPC and science/governance reviewers both returned `NO_BLOCKER` after
  remediation; durable review:
  `reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_v1p6_execution_handoff_review_20260712.md`.
- **RQ014 v1.6 Waves 0–3 EXECUTED (2026-07-13): rating-blind export PASS; D1 accepted.** Managed checkout
  synced via reviewed incremental bundle (HPC HEAD detached at exact `24be0827`; remote-main CAS
  `b1476bd0`→`eb1ade2b`; 4-round red-teamed sync script; attestation archived). Immutable spec
  `RQ014_0_score_stripped_export_20260712T154921Z_1ee1e1d1.json` (SHA-256 `0e6ca13094ad…31f62b`, 0444,
  inode 95871301641, W1-A/W1-B byte-identical dual derivation) published by staging hard-link no-replace.
  Validate-only evidence parsed independently by W1-A and fresh W1-D (14/14). Single authorized submit →
  Slurm `1919412` `zxc-rq014-export-0e6ca13094ad` COMPLETED 0:0 (3m52s, amd/cpua102, 1CPU/8G, --export=NIL
  on directive and submit line). Output: 9-file score-stripped bundle at
  `/share/home/u25310231/ZXC/sociality_estimation/inputs/RQ014/wod_rated479_score_stripped/v1` — universe 479,
  geometry 476, structural attrition 3, candidate distribution {0:3,3:476}, all forbidden/unexpected/duplicate/
  nonfinite scans 0, receipts hash-chained DONE→export→{sanitization,file_manifest}. W3 statistics and
  execution/governance reviewers (fresh, distinct) both `NO_BLOCKER`; bounded report + evidence manifest:
  `reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_1_declassification_export_20260712T165224Z_0e6ca130/`.
  PI accepted the export through the Lead session's interactive D1 prompt on 2026-07-13 and authorized starting
  the `rq014_g2_contract_preflight` authorization loop only. The scoped decision, exact two-operation candidate
  allowlist and W4b candidate review manifest implement §8.1 steps 1–3. W4b fixed the cross-commit
  exporter-provenance defect with a required preflight-only
  `declassification_export_commit`; round-4 fresh statistics and execution/governance reviews both returned
  `NO_BLOCKER` on the exact 70-row manifest `4e06316aa35a95c85a330bf6d82a6ba87642f0d3e47a12810519c34f284b17a7`.
  The same-named Formal G1 is regenerated at SHA-256 `6bbdcd08107f0d93119191177bdf53419b5b2fb7fc511a16f0cad1870e30fcd7`
  with `FORMAL_G1_PASS`; the 74-row final bundle is regenerated at SHA-256
  `999ad5529241ca1a8197b525ba84abde9c570d298c0478d0e1e78e8b8d136d3c`. Formal G1 does not grant execution:
  publication/sync, immutable-spec validate-only, fresh validation and explicit user confirmation remain mandatory.
  No rating value was read.
  Fleet evidence: `.codex-fleet/rq014-execution-v1p6/board/`.
- **RQ014 preflight COMPLETE, D2 accepted (2026-07-14).** Authorization loops PR #10–#14 (each fresh-dual-reviewed,
  Formal G1 regenerated) delivered: preflight allowlisting + explicit `declassification_export_commit`; science
  amendment v1.7 + `RQ014_recovery_lane_v3.json` (PI-identified frozen M3 envelope, 320 cells / 960 terminal rows,
  out-of-support extrapolation semantics normative); registries v1p6 (12 active bindings; X02 LEGACY_INACTIVE_UNBOUND);
  WOD path-type mapping freeze (254 mapped CP115/HO90/MP48/F1; 222 excluded@F, 3@K; installed at
  `inputs/RQ014/wod_path_type_mapping/v1`); blind-anchor fixed root + shared cross-phase validator. First preflight
  submission failed fail-closed (reviewed cross-phase defect; RUN_ID burned, root preserved); after PR #14, job
  `1924193` `zxc-rq014-pre-72dd4362f954` COMPLETED 0:0 — 12/12 bindings materialized, M3 delivery verified
  pre-deserialization (immutable receipt), receipts + bounded report dual NO_BLOCKER. Report:
  `reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_2_contract_preflight_20260714T003336Z_72dd4362/`.
  W5d is published on `origin/main` at merge `c3036fce`: export/preflight remain on v3, while
  `rq014_g2_resource_pilot` binds the checksum-closed v4 runtime for rating-blind
  `source_load`, `window_assembly`, `feature_prep`, and M3 measurements on frozen light/heavy cells.
  Its `amd` profile is 1 node/1 task/16 CPU/32G/04:00:00; the endpoint cells run concurrently in separate
  single-threaded processes under a 16-worker ceiling after one separately measured parent source load whose
  read-only payload is inherited copy-on-write by the fork workers. Ordered
  axes/320 IDs fail closed on digest drift; joint three-candidate H-common, exact H20/HFEAS eligibility, and frozen
  heading boundaries govern the measured windows. Native-10Hz counterpart positions are support-only interpolated
  to the R04N 0.25 s grid; gaps above the exact inclusive `2*dt=0.5 s` boundary are ineligible. Per-cell serial,
  worker-pool, and aggregate wall-clock evidence
  is recorded for D3. The v4 gate verifies the pinned stdlib (1,849/40,860,773/0),
  site-packages (12,206/487,535,728/0), and 94-row native closure before M3 loads once in the parent
  and scores per cell on a deterministic rating-free cost vector. Any v4/M3 mismatch aborts with no DONE;
  numeric M3/combined projections are emitted only on PASS. The first pilot validate-only attempt failed
  before submission because the M3 validator still required the pre-W5d 12-key delivery block while the reviewed
  contract correctly carried 14 keys. W5f repairs that authority wiring with a real-contract end-to-end regression.
  W5g clarifies that preflight is the verification-only operation, while the pilot spec still requires M3 and its
  job prelude reverifies the scorer before v4 deserialization. A pre-runtime closure mismatch exits non-zero with
  a deterministic `RQ014_CLOSURE_GATE_FAIL <stable-identity>` stderr line from every emitted export,
  preflight, and pilot gate (digest, output-root, M3, stdlib, site-packages, and native), captured by Slurm
  with no receipt/DONE;
  runtime-detected M3 failures write a FAIL receipt with no DONE or numeric cost. The W5f-W5j fresh mini-review
  and Formal G1/final-bundle regeneration completed, and sync v15 plus a newly pinned pilot
  spec reached validate-only. That validation then failed before submission because the Python v4 native-manifest
  validator expected literal `<TAB>` tokens while the correctly pinned immutable TSV contains real tab bytes.
  W5l changes the v4 Python-side expectation and adds a real-format full-closure integration regression. Its
  follow-up review found the emitted v4 Slurm prelude retained the same literal-token copy; W5m aligns that
  shell-only expectation with the published real-tab header and executes all five emitted native-header gates
  against those bytes. Both fixes leave the v3 literal-`<TAB>` contract unchanged. W5n then completed fresh
  dual review, Formal G1, final-bundle regeneration, and publication through sync v16. Re-anchored pilot job
  `1929952` reached runtime with source loading and M3 model loading both PASS, but both selected cells failed
  window assembly because the pilot harness required exact float equality at observed-support endpoints that
  differ by exactly `1.7763568394002505e-15 s`: target `3.5` is about four binary64 ULPs above stored
  `3.4999999999999982`, and target `-4.9` is about two ULPs below stored `-4.899999999999999`. W5o mirrors
  the frozen preflight resampler's bounded `1e-12` endpoint-equivalence tolerance; this is not characterized
  as zero extrapolation. Frozen science and the `>2*dt` gap exclusion are unchanged. Fresh dual review, Formal
  G1/final-bundle regeneration and sync produced the corrected immutable rerun: job `1930942` completed PASS
  with all eight stages passing, every failure count zero, and receipt SHA-256 `0f192b4e…cc184`. D3 subsequently
  accepted its projection of about 2.8 minutes parallel wall time and `0.6670795` CPU-hours for all 320 cells.
- **RQ014 v1.6 contract preflight EXECUTED PASS (receipt-verified 2026-07-14; this supersedes
  earlier preflight-pending statements above).** Exact authority commit `b06a243eea7e1418622f89e5ea80d3da4fe3bc58`,
  Formal G1 `755e6a34…`, final bundle `41ac5280…`, and immutable run spec `72dd4362…` produced run
  `RQ014_1_wod_rating_recovery_20260713T161542Z_41ac5280`. Slurm `1924193`
  `zxc-rq014-pre-72dd4362f954` completed `0:0` in 3m26s on `cpua041`. The immutable preflight
  receipt is SHA-256 `1e2d0cf6…0bb2e23`, status `PASS`, with `rating_access=NONE`,
  `rating_join=NONE`, `observed_statistics=NONE`, all sanitization scans zero, and the exact
  M3 artifact verified without deserialization. `rq014_g2_resource_pilot` now has a local W5b implementation
  candidate but remains non-submittable pending fresh review, G1/bundle, validate-only, and explicit pre-submit PI
  confirmation; stop and preserve. Local receipt/log copies are
  under `.codex-fleet/rq014-execution-v1p6/board/w4g_evidence/pf_*`.
- **RQ014 G2R W1-W5a merged; W5b authorizes only the rating-blind r2 build
  (local authority wave, 2026-07-17).** Base `origin/main` `7441f27f` contains the frozen output schemas,
  WOD-to-M3/anchor/NC kernel, scoring/readouts, and 320-cell rating-blind orchestration.
  W5a adds the schema-v2 template, exact v4 closed-snapshot launcher path, profile
  `rq014-g2r-cpu-v1` (`amd`, 1/1/16 CPU, 32G, 04:00:00, thread caps 1), atomic
  rating-blind output publication, immutable receipt, and PASS-only DONE. D1 preflight,
  D2 pilot PASS (`1930942`; receipt `0f192b4e…cc184`), and D3 budget approval are satisfied;
  `RQ014_PI_decision_D3_G2R_authorize_20260717.md` therefore adds only
  `rq014_r2_blind_feature_build` to the central allowlist. At that W5b checkpoint,
  rating access/join/statistics remained forbidden, R3/D4 remained denied, and
  G2R ended before leaderboard/recovery-ledger construction; the D4 Wave-B entry
  below supersedes only that R3/D4 status. This local authority wave performs no HPC action: operators must still
  publish/sync the reviewed commit, rebuild fresh upstream lineage against its contract,
  materialize an immutable G2R spec, pass validate-only, and use the explicit submit step.
- **RQ014 G2R run-8 stale-pin repair handoff (2026-07-23; no commit or HPC
  submit in this repair wave).** The rating-blind W2/IPV prepass keeps spawned workers capped at
  126 under the 128-CPU, 256G, 72:00:00 `rq014-g2r-cpu-v1` profile, while the parent
  preserves canonical merge order and the single unchanged M3 batch. Run #8 completed all
  479 scene prepasses, proving the deterministic budget/terminalization fix, then failed
  post-assembly because the output contract still embedded an obsolete execution-contract
  SHA. A complete audit of its 39 file-backed SHA/size pairs plus the intrinsic grid hash
  found no second stale binding; the execution contract at that repair checkpoint was `47bf9e48...92fcf` at
  36,039 bytes. Each scene retains a deterministic 40,000,000-objective-evaluation budget.
  Exhaustion emits `INELIGIBLE_SOLVER_BUDGET_EXCEEDED` /
  `F_SOLVER_BUDGET_EXCEEDED`; enumerable numerical/source-gap candidate failures use the
  existing scene-terminal channel, while structural drift remains globally fail-closed.
  The output contract at that checkpoint was SHA-256
  `36f5bbd089627e4e1e9cd5e45599d890529fc6313b793e98a108d95c2f0328ca`;
  it is superseded by the D4 Wave-A source-contract re-anchor below.
- **RQ014 R3 result review complete (2026-07-25; this supersedes the execution-state
  wording in the D4/R3 bullet below for scientific status).** Managed R3 run
  `RQ014_3_full_rating_join_and_rank_20260724T053954Z_49dcd5c0` produced 960
  terminal association rows and one compatible secondary
  `RR3-R04N-CH-W25-H20-NMD_MEAN-RWS` row (`r=-0.384`, `n=42`); primary NEX has
  zero compatible rows. R10L is `DEFECT`, not a scientific null, because a
  whole-branch source-gap terminalization cleared the arm before the frozen
  per-window semantics could apply. Its rating-free support ceiling is
  `UNCERTAIN` until the full probe receipt is tracked. The current R3 artifact
  set has no `selected_recovery_recipe.json`, and no G4R clean replay PASS is
  present. Current status is therefore `PENDING_REPLAY / NOT ACCEPTED`; do not
  create an accepting `decision.md` or route a confirmed-transfer claim to the
  paper repository. Independent review:
  `reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/codex_review.md`.
- **RQ014 D4/R3 Wave B conditionally authorizes the single managed R3 operation
  (2026-07-23; no rating read, HPC action, or commit).** The immutable blind bank
  `RQ014_2_blind_feature_build_20260722T210000Z_e41c8792` is complete with
  `BANK_VERIFY=PASS` and receipt prefix `b74bb0e2`. The new fixed operation
  `rq014_r3_full_rating_join_and_rank` verifies that umbrella/PASS-receipt/DONE
  chain and all eight bank artifacts, then under env-v5 and profile
  `rq014-g3r-cpu-v1` (`amd`, 1/1/16 CPU, 32G, 02:00:00) performs one exact
  geometry-keyed rating join and the recovery-lane-v3 RWS/PSP/PPR screen. It
  produces exactly 960 terminal rows, counts-and-hashes-only rating-access and
  operation receipts, and PASS-only DONE through a mode-0700 private staging
  directory and atomic publication. The governed rating source is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase3_preference_test/ratings_extracted.csv`;
  its bytes were not opened or hashed in Wave A or Wave B. The immutable spec
  binds the stat-only size `337416` with `sha256: null`; managed R3 establishes
  and records the governed source digest at first controlled contact. R3 is now
  in `allowed_operations`; its
  contract status is
  `CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1_AND_SCOPED_D4_DECISION`. Current anchors:
  launcher `22a9c41c...b3e0` (213,848 bytes), wrapper `902186a7...6e57`,
  execution contract `91f86e6d...084f` (39,508 bytes), and G2R output contract
  `d3736c16...6e87`. Future G2R specs must bind the latter; future G3R specs must
  additionally bind the final Wave-B G1/bundle and exact bank/rating references.
  First managed R3 job `1969820` failed closed at `RATING_ACCESS` because the
  local parser expected the synthetic Wave-A column names instead of the real
  13-column ratings schema. The source opened and matched the bound size; its
  controlled-contact SHA-256 is `2bbd7d721591b4756108285ae869c2fb4d6dc7bbe45077870787381929ec3e4d`.
  Non-disclosure held (`rating_value_read_count=0`; empty keyset hash). The
  uncommitted local parser repair maps `segment_key`, `tstar_context_step`,
  one-based `candidate_index`, and `geom_hash` onto the frozen four-key bank
  interface. A fresh merge commit and regenerated G3R spec are required before
  any rerun; no bank re-anchor is required.
  <!-- G3R_WAVE_B_BINDING_STATUS:START -->
  Bindings finalized: bank manifest `2b4da1df4a5328b80d88b815ac3cdb71546952bac4638b29f4fa263b527d4515`, bank receipt `b74bb0e2ab5966b9eaaab164130bd50791b5ceee5743030b0bb26719d79c37b9`, DONE `256750c71902e31e46335c331369c256b7b7d13a4fb08758f1b8234b6229efdb`, and ratings size `337416`; first controlled contact established source SHA-256 `2bbd7d721591b4756108285ae869c2fb4d6dc7bbe45077870787381929ec3e4d` before the fail-closed parser rejection. The ratings file was not opened or hashed by the build/finalizer, and no rating value was consumed by failed job `1969820`.
  <!-- G3R_WAVE_B_BINDING_STATUS:END -->
- **RQ014 NC gate portability repair passed Linux preverification and entered its
  Phase 2 governance freeze (2026-07-20).** The third
  rating-blind 320-cell attempt reached the NC gate after source registration, then
  failed because macOS and managed-Linux SLSQP IPV bytes differed while state,
  M3-context, and both reference components stayed byte-identical. Managed-Linux
  preverification then rejected the first hybrid patch 5/5: R10L-W25 and its future
  control reached different IPV point estimates with errors equal to `1.9e-13`,
  exposing a degenerate equal-error solution set. The PI-final error-anchored option
  keeps those four component hashes and both same-process controls exact, compares
  only the two committed IPV errors at `rtol=0`, `atol=1e-5`, and treats IPV values as
  non-anchored provenance that must be finite and within the exact solver candidate
  hull `[-3*pi/8,3*pi/8]`. Revised Mac and managed-Linux env-v5 replays both pass
  5/5. The Phase 2 change set binds the final output-contract hash through all three
  code consumers and regenerates the 141-row review manifest, two formal reviews,
  Formal G1, and the 145-row checksum bundle. It remains an uncommitted Lead handoff;
  no submit or rating-access authority is added.
- RQ014 focused verification command uses the existing verifier environment:
  `.venv_ipv_verifier/bin/python -m pytest -q tests/test_rq014_v1p5_contract.py
  tests/test_rq014_score_stripped_export.py tests/test_hpc_run_launcher.py
  tests/test_rq014_managed_hpc_contract.py` plus the G0/FL05/v1p3/recovery-contract suites. Current
  W5b focused resource-pilot suite result is recorded in `main_workflow.log`; prior W5a result: `244 passed`.
  Python compilation, shell syntax and `git diff --check` also pass. No rating value was read,
  no production run root was created, and no Slurm job was submitted.
- RQ014 last adjudicated W4h preflight authority bundle is
  `reports/plans/RQ014_plan_v1p6_checksums_20260713.sha256`: 105 rows, SHA-256
  `41ac52808cba5eb729829bc031053c49fb49583691ff24f7e2662c38b5ee2f19`; the v1.5 baseline remains at
  `reports/plans/RQ014_plan_v1p5_checksums_20260712.sha256`.
- **RQ014 contract preflight PASS; D2 accepted; W5a is an authority candidate, not submit authority.** The W4h
  correction was reviewed/published, then the single authorized preflight submit completed as Slurm job `1924193`
  with `COMPLETED/0:0`. The bounded report
  `reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_2_contract_preflight_20260714T003336Z_72dd4362/report.md`
  records rating access `NONE`, observed statistics `NONE`, verified receipt chain and dual `NO_BLOCKER`. D2 accepts
  that evidence and authorizes only the §8.1 loop for `rq014_g2_resource_pilot`. The W5a candidate adds it as the
  exact third central operation, binds decision `RQ014_PI_decision_D2_resource_pilot_20260714.md`, and requires
  receipt schemas `rq014-g2-contract-preflight-receipt-v1` plus `rq014-managed-operation-done-v1`. W5b now adds
  the pilot profile/schema/template/entrypoint and freezes light cell `RR3-R04N-CH-W10-H20-NEX_MEAN` plus heavy
  cell `RR3-R10L-TF-HFEAS-NEX_MEAN`; M3 is verification-only and cannot be deserialized under v3. Fresh dual
  review and post-review G1/bundle regeneration are next;
  an explicit PI stop remains before pilot submit, and D3 separately gates the full G2R compute budget. No rating
  value was read and no HPC write/job occurred in W5a.
- RQ007 held-out remains sealed. RQ009 must freeze all rules and stop at
  `READY_FOR_SEALED_TEST` until a new PI authorization opens it.
- RQ008B is not authorized; no RQ008 motif may enter RQ009.
- External-validation priority after RQ009: **OnSite first**, WOD-E2E tracking pilot in
  parallel.
- Two-human RQ012 annotation is deferred; RQ012 remains `BLOCKED_FOR_HUMAN_LABELS`.
- The current paper baseline is paper-repository `main` merge `c6783577`; `structure.md` is
  v4.1 estimability-aware dynamic norm and must supersede v3 self-anchor round-trips.
- The 20260612 sigma 0.1 full-rerun data source is under
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`.
- HPC reuse: shared Tongji HPC usage guide for all local projects →
  `../HPC_TONGJI_USAGE_GUIDE.md`; InterHub/IPV-specific reusable assets remain
  in `reports/knowledge/INFRA_hpc_tongji_reuse.md`. On HPC, durable work lives
  under `/share/home/u25310231/ZXC`, and newly submitted Slurm job names must
  start with `zxc-`.
- RQ010B WOD-E2E Tongji HPC basic parser access is now working under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`. The 2026-06-26 four-shard
  ratings-sealed structural pre-flight sampled all 12 candidate-bearing
  scene frames found in shards 00000..00003 and passed the five t* structural
  checks on those 12; the full 479-scene gate remains pending.
- RQ010B StreamPETR Route 4 Tongji HPC setup is now available under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`: code at `code/StreamPETR`,
  env at `envs/streampetr`, checkpoint at
  `checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, and
  full R50 flash checkpoint dummy 6-camera forward passed on an L40 GPU node
  (`logs/streampetr_checkpoint_forward_flash_l40_20260626.log`, output
  `boxes_tensor_shape=[20, 9]`). Key versions: torch 1.13.0+cu117,
  CUDA module 11.8, mmcv 1.6.0, mmdet 2.28.2, mmdet3d 1.0.0rc6.
- RQ010B StreamPETR Route 4 real Waymo Perception lead-config smoke now passes
  on Tongji HPC L40: converter/dataset/config/smoke scripts live under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/`; one training
  segment was converted to
  `data/waymo_stream_petr/waymo_infos_train_1seg.pkl` with five forward cameras
  (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `SIDE_LEFT`, `SIDE_RIGHT`), sample
  shape `img=[1,5,3,256,704]`, forward output `boxes=[300,9]`, and two-step
  forward/backward smoke loss decreased `400.0814 -> 316.3276`. Runbook:
  `code/StreamPETR/tools/waymo_perception/RQ010B_ROUTE4_WAYMO_STREAM_PETR_SMOKE.md`;
  key logs:
  `logs/waymo_sample_shape_l40_20260626.log`,
  `logs/waymo_forward_l40_20260626.log`,
  `logs/waymo_train_overfit_l40_20260626.log`.
- RQ010B Waymo Perception v1.4.3 small dev subset is now available on Tongji
  HPC under `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/perception/`: first
  4 deterministic-sorted training segments plus first 2 validation segments
  from `gs://waymo_open_dataset_v_1_4_3/individual_files/`, exact-size and
  crc32c verified. Manifest:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/perception_dev.tsv`.
- RQ010B Waymo Perception v1.4.3 finetune subset is now 256 training plus
  16 validation `.tfrecord` segments on Tongji HPC under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/perception/`, crc32c verified
  with manifest
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/perception_256.tsv`
  (272/272 ok). This supersedes the earlier 64-train/16-val
  `perception_finetune.tsv` subset for current StreamPETR finetuning.
- RQ010B StreamPETR Route 4 dev6 dry-run finetune now passes end-to-end on
  Tongji HPC L40. Converted infos are
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_train_4seg.pkl`
  (794 train samples, 4 scenes) and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_val_2seg.pkl`
  (397 val samples, 2 scenes). Dry-run config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_dev6_dryrun.py`;
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_dev6_dryrun_20260626/`.
  The clean 40-iter run saved `iter_20.pth` and `iter_40.pth`, loss decreased
  `73.2736 -> 40.4284`, and the lightweight Waymo center-distance smoke eval
  completed with `waymo_center_recall_2m=0.0` on 397 val samples. This metric
  validates the eval path only; it is not an accuracy claim.
- RQ010B StreamPETR Route 4 lead-config 64-train/16-val finetune was stopped
  early after best 16-val smoke recall plateaued around epoch 4. Frozen best
  checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/best_waymo_center_recall_2m_iter_50732.pth`.
  The original Slurm job was `1707389` (L40 node `gpu4037`), launched
  2026-06-27 02:48 CST after job `1707307` failed at MMCV config parse with
  `TypeError: cannot pickle '_io.BufferedReader' object`. The config fix
  deletes the closed `_handle` left by the parse-time train-info `with open`
  block, and login-node `Config.fromfile(...)` returns `CONFIG_PARSE_OK`.
  Sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`;
  config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`;
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`;
  Slurm logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune64_1707389.out`
  and `.err`. The job first converts
  `waymo_infos_train_64seg.pkl` and `waymo_infos_val_16seg.pkl`.
- RQ010B §5 detector-quality/error-model gate is complete for the frozen
  Route 4 best checkpoint on 16 Perception validation segments. Evaluation job
  `1710088` ran on one L40 for 00:28:41 (0.478 GPU-h allocated; script runtime
  estimate 0.460 GPU-h). Method: StreamPETR single-GPU inference, classwise
  rotated BEV NMS (`score>=0.05`, IoU `0.25`, max `100` detections/frame),
  center-distance AP/matching at 2 m because official Waymo LET-3D-AP metrics
  ops were unavailable in the current env. Validation-selected operating point
  is score threshold `0.15` (max micro-F1 on the same 16-val segments). Result:
  overall AP `0.00328`, recall `0.08034`, precision `0.03276`; Vehicle AP
  `0.00432`, recall `0.10585`, precision `0.03276`; Pedestrian and Cyclist AP,
  recall, and precision all `0.0`. Verdict: this 64-segment detector is not
  adequate for tracker + HOTA/AMOTA QA; add the remaining 734 Perception
  training segments (full 798 total) with class-balanced checks and retrain
  before retesting Route 4. Route 5 remains fallback if full-data Route 4 still
  leaves Pedestrian/Cyclist near zero. Outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_best50732_20260627_summary.json`,
  `_metrics_by_class_range.csv`, `_error_model.json`, and `_error_model.csv`.
- RQ010B improved StreamPETR Waymo Perception 256-train/16-val finetune is
  complete. Train-only 4-L40 DDP Slurm job `1712698` completed on `gpu4011`
  in `20:27:18` (about 81.8 allocated L40 GPU-h) and saved all 12 raw-equivalent
  epoch checkpoints through `iter_152124.pth` (`latest.pth -> iter_152124.pth`).
  Previous jobs failed only in in-loop distributed
  evaluation: `1712416` on `gpu4009` failed after epoch-1 eval with NCCL
  watchdog timeout, and `1712590` on `gpu4025` saved `iter_25354.pth` then
  failed at DistEvalHook with `TypeError: 'NoneType' object is not iterable` in
  `projects/mmdet3d_plugin/datasets/waymo_dataset.py evaluate()`. Training was
  healthy with loss around 21 and checkpoints `iter_12677.pth` and
  `iter_25354.pth`. Current no-eval fix applied on HPC: `tools/train.py` keeps
  `timeout=datetime.timedelta(hours=4)` for `init_dist`; the active config sets
  `evaluation=None`, `custom_hooks=[]`, `raw_equivalent_epochs=12`,
  `max_iters=152124`, `checkpoint_config.interval=12677`, and
  `max_keep_ckpts=-1`; `projects/mmdet3d_plugin/core/apis/mmdet_train.py`
  skips eval-hook registration when `evaluation is None`; the resume sbatch
  also passes `--no-validate` and resumes explicitly from `iter_25354.pth`.
  Saved checkpoints must still be evaluated separately on 1 GPU because the
  DDP eval path has the known `NoneType` bug. Config:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune256_balanced_warminit.py`;
  warm-init checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/checkpoints/stream_petr_waymo3_warminit_nusc_car_ped_bicycle.pth`.
  Recipe uses ClassBalancedDataset `oversample_thr=0.70`, nuScenes class-row
  warm init (`car->Vehicle`, `pedestrian->Pedestrian`, `bicycle->Cyclist`),
  5x LR for `pts_bbox_head.cls_branches`, grid mask plus resize/flip and
  BEV rot/scale augmentation. The config hard-points to
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_train_256seg.pkl`
  and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/waymo_stream_petr/waymo_infos_val_16seg.pkl`
  and fails if they are absent. Converted-info assertion:
  train 50,708 samples/256 scenes/360,505,501 bytes; val 3,151 samples/16
  scenes/22,943,129 bytes; JSON
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/waymo_infos_256_16_assert_1712385.json`.
  `Config.fromfile` passes with 4-GPU schedule
  `raw_iters_per_epoch=12677`, `checkpoint_interval=12677`,
  `eval_interval=164801` (disabled sentinel greater than `max_iters`),
  `max_iters=152124`, `evaluation is None`, and zero custom hooks. Quick
  4-L40 DDP smoke job `1712408` on `gpu4011` passed warm-init +
  class-balanced sampler
  + DDP backprop with averaged loss decreasing
  `78.1665 -> 66.1600 -> 59.0420`;
  logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp_smoke_1712408.log`
  and `.jsonl`. Original full-run sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628.sbatch`;
  failed timeout-fix resume sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_timeoutfix_20260628.sbatch`;
  completed no-eval resume sbatch:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_noeval_20260628.sbatch`;
  original resume checkpoint:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/iter_25354.pth`
  (`latest.pth` now points to final `iter_152124.pth`);
  work dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/`;
  Slurm logs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_1712416.out`
  and `.err` for the first failed run,
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_resume_1712590.out`
  and `.err` for the failed eval-resume job, and
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/logs/streampetr_waymo_finetune256_ddp4_resume_noeval_1712698.out`
  and `.err` for the completed train-only resume job. Separate single-GPU
  detector-quality Slurm job `1745613` (`zxc-rq010b-eval256`) evaluated
  checkpoints `iter_76062`, `iter_101416`, `iter_126770`, and `iter_152124`
  on the 16 Perception validation segments in `01:42:26` (about 1.71 allocated
  L40 GPU-h; script runtime sum about 1.64 GPU-h). Best by mean AP over the
  9 class x range cells is ep12 `iter_152124`: `mAP_9=0.08454`, pooled
  center-distance AP `0.10835`, overall recall `0.21916`, precision `0.23675`
  at score threshold `0.225`. Class all-range matched-TP recall/precision:
  Vehicle `0.24363`/`0.23469`, Pedestrian `0.14515`/`0.24725`, Cyclist
  `0.05644`/`0.40000`. Pedestrian and Cyclist are nonzero, so warm init plus
  class balance worked enough to clear the zero-class failure; `grad_norm:nan`
  during training was benign for detector output. The detector is now adequate
  to proceed to tracker + HOTA/AMOTA QA on the 16-val pilot, while still weak
  at 50+ m and not yet a final detector-quality solution. Best outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_256_balwarm_ep12_iter_152124_20260629_summary.json`,
  `_metrics_by_class_range.csv`, `_error_model.json`, `_error_model.csv`,
  `_matched_tp_errors.csv`, `_threshold_sweep.csv`, and `_post_nms_records.pkl`;
  checkpoint ranking summary:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_route4_detector_quality_256_balwarm_20260629_checkpoint_summary.{json,csv}`.
- RQ010B WOD-E2E IPV-rating pilot degeneracy investigation/fix is complete on
  Tongji HPC. The original
  `results/rq010b_wod_e2e_ipv_rating_pilot_20260629/` IPV-rating result is
  invalid for IPV conclusions: WOD-E2E state sequences were sampled at
  `dt=0.25` s while the legacy IPV estimator still integrated with global
  `dt=0.1` s; probability-space trajectory likelihoods underflowed to all-zero
  candidate weights and forced the uniform `ego_ipv=0.0` fallback; and the
  adapter used each evaluated candidate as its own reference line instead of
  the RQ010B §6 scene-level route reference. The patched HPC adapter is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_ipv_rating_pilot_20260629/analyze_wod_e2e_ipv_rating_pilot.py`
  with backup `.bak_20260629_dtfix`; it sets estimator `dt=0.25`, uses
  log-domain trajectory-likelihood normalization, and builds the §6
  past-pose-plus-routing constant-curvature ego reference. Final fixed outputs:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_ipv_rating_pilot_routefix_20260629T124941/`.
  Distribution is now finite and varied but still partially uninformative:
  `n=33`, range `[-1.1781, 1.1781]`, 17/33 `abs(ego_ipv)>1e-6`,
  16 rounded-distinct IPV values, and 4 uniform-fallback rows. Re-run
  IPV-rating association remains weak/null: pooled Spearman `rho=0.123`,
  95% bootstrap CI `[-0.224, 0.452]`, `p=0.495`; mean within-scene Spearman
  `-0.0787` over 11 usable scenes; IPV single-feature R2 `0.0110`, below the
  best physics feature `driven_ade_m` R2 `0.0634`. Reproducer artifacts:
  `debug_reproducer_dt_route_log_20260629.{md,json}` in the final result dir.
  Applicability caveat remains: this is an exploratory one-frame StreamPETR
  velocity-extrapolated counterpart pilot, not the full RQ010B validated
  tracker/M3 preference test.
- RQ010B WOD-E2E multi-frame ceiling investigation is complete on Tongji HPC
  and overturns the four-shard hard-ceiling interpretation. A single
  `E2EDFrame` contains one timestamp's 8 camera JPEGs, one per surround camera,
  not an internal recent-frame sequence; ego `past_states`/`future_states` are
  16/20 samples at 4 Hz. The sparse four-shard finding is an interleaved
  shard-access artifact: adding four CRC-clean validation probe shards (`00004`,
  `00005`, `00007`, `00010`) increased unique pre-t* frame coverage for all 12
  rated pilot segments (min/median/max `2/6/9` -> `6/12/16`). The 8 clean
  shards still did not reconstruct a 10-frame contiguous run ending at t*
  (end-contiguous max 2; best pre-t* contiguous max 3), so the concrete next
  path is full/targeted validation shard indexing over
  `val_202504211843.tfrecord-00000..00092-of-00093`. Four attempted extra probe
  shards (`00006`, `00008`, `00009`, `00011`) were excluded after CRC failures
  from an interrupted transfer. Artifacts:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/wod_e2e_temporal_ceiling_probe_20260629/{rated_record_structure_probe.json,shard_growth_probe_clean_extra.json,shard_growth_probe_contiguous_clean_extra.json}`
  and manifest
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/validation_probe_00004_00011.tsv`.
- RQ010B full WOD-E2E validation rated479 streaming ingest/extract is complete
  on Tongji HPC. Slurm job `1746449` (`zxc-rq010b-full93`, `amd`, `cpua277`)
  completed with exit `0:0` in `11:47:32`; it was not cancelled. The apparent
  post-loop hang was useful finalizer work: after `all-shard loop complete
  receipts_ok=93/93` at 2026-06-30 04:37 CST, the job wrote sorted
  per-segment `frames.tfrecord` and `frames.index.tsv`, logged
  `segments_finalized=479`, and exited at 05:11. The buggy final summarize
  left `manifests/rated479_segment_counts.tsv` at 0 bytes; it was regenerated
  atomically from the independent readiness table and is now nonzero.
  Extracted data is under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/data/rated479_segments/<segment_key>/`
  with exactly 479 rated segment directories and zero leftover raw validation
  shards. The stray 480th directory was empty `_tmp` and was removed; the stale
  GCS token at
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/secrets/gcs_token` was deleted
  after finalization.
  Readiness artifacts:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/manifests/rated479_segment_readiness.tsv`
  and `.json`. Gate result: 479/479 segments have at least 10 strict
  contiguous pre-t* frames; min/median/max pre-t* contiguous history is
  91/228/229 frames, histogram `50-99:3`, `100-199:100`, `>=200:376`, and
  there are no short/abstain-worthy segments. All 479 segments have the five
  forward-arc cameras/calibrations (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`,
  `SIDE_LEFT`, `SIDE_RIGHT`) on every indexed frame. Native cadence is 10 Hz
  from adjacent `context_step` deltas (`mode=1` for all segments). Current
  directory size is about 354.8 GB apparent / 330.5 GiB because final
  tracker-facing TFRecords and shard TFRecords are both retained; the
  tracker-facing final `frames.tfrecord` set is 177.4 GB / 165.2 GiB, median
  374,013,480 bytes (356.7 MiB) per segment. Downstream tracker should read
  each segment through `frames.index.tsv` sorted by `context_step`, then stream
  the matching records from `frames.tfrecord`; each E2EDFrame contains the
  pruned five-camera images/calibrations plus ego past/future states and
  preference trajectories. Shard archives remain under `shards/` for audit or
  rebuild only.
- RQ010B WOD-E2E 12-segment dense multiframe tracking -> IPV counterpart
  selection repair is complete on Tongji HPC, using cached detections/tracks
  from
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_pilot_20260630T053507/`
  and the fixed IPV adapter under
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_ipv_rating_pilot_20260629/`.
  The active selector code is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/code/rq010b_multiframe_tracking_ipv_20260630/analyze_multiframe_tracking_ipv.py`
  with backups `.bak_counterpart_gates_20260630T0603` and
  `.bak_vehicle_class_gate_20260630T0610`; focused regression tests are in
  `test_counterpart_selection_gates.py` and pass (`4 passed`). Final cached
  L40 Slurm rerun `1751326` (`zxc-rq010b-cp-gates`) completed with exit `0:0`
  in `00:01:58`, writing
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_pilot_counterpart_gates_20260630T060925/`.
  Gates now require vehicle class, >=10 hits, >=1.0 s span, <=0.5 s stale,
  >=2.0 m displacement, >=0.5 m/s path speed, jitter ratio <=4, step p95
  <=1.75 m, observation gap <=0.5 s, history coverage >=0.4, current distance
  <=35 m, and predicted ego-path conflict gap <=8 m / TTC <=6 s or compatible
  crossing/leading/opposing geometry. Retained interaction rate on the 12 pilot
  segments is 8/12 vehicle counterparts (24 candidate IPV rows) and 4
  abstentions: two no quality-passing tracks, one no interacting vehicle after
  conflict gate, and one pedestrian-only/conflict-gate case after the vehicle
  class gate. Selected vehicle tracks are all moving/interacting
  (displacement 2.51-6.36 m, 14-48 observed points, min predicted gap
  0.221-7.71 m, geometries crossing/opposing/leading-or-merging). Ego IPV on
  retained candidates is finite with range `[-1.17810, 1.14898]`, median `0`,
  13/24 nonzero above 1e-6, and pooled rating association remains small/null
  (Spearman rho `0.1269`, 95% bootstrap CI `[-0.3147, 0.5425]`, p `0.5547`).
  This pilot gate has now been scaled to all 479; use the audited full-run
  result below for current RQ010B operating facts.
- RQ010B WOD-E2E full479 scored-target multiframe tracking -> gated
  counterpart -> fixed ego-IPV audit is complete under Tongji HPC `/ZXC`
  boundaries, status `audited_not_frozen` pending review. Canonical result dir:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/`.
  Detector array `1751377_[0-7]` used one L40 per shard on
  `gpu4006/gpu4008/gpu4010` and completed cleanly; CPU job `1751378` wrote the
  full analysis artifacts then failed only in posthoc audit markdown formatting,
  which was patched and rerun successfully on the completed CSV/JSON outputs.
  Frozen gates retained 302/479 scenes and abstained 177/479. Abstention
  reasons: no interacting track after conflict gate 92, no track after motion
  gate 36, no track after quality filter 34, no track after history-coverage
  gate 9, no track after smoothness gate 4, and 2 data-level target abstentions
  with no scored preference frame plus 50-frame history. Selected counterparts
  are all `Vehicle`, all pass real-moving gates, and all pass interaction gates;
  selected-track displacement median 3.34 m, mean speed median 1.75 m/s, and
  predicted min-gap median 3.37 m. Ego IPV distribution over 906 retained
  candidate rows has mean 0.00799, median 0, q25/q75 -0.0938/0.1041, and range
  [-1.1781, 1.1781]. Primary IPV-rating association is weak/null: all-retained
  pooled Spearman rho -0.0384, 95% CI [-0.1016, 0.0256], p=0.2477; fresh
  confirmatory subset excluding the 12 pilot segments has 294 scenes / 882 rows,
  pooled Spearman rho -0.0445, 95% CI [-0.1078, 0.0183], p=0.1872. Within-scene
  rank correlations are also weak (fresh mean Spearman -0.0678, p=0.1198;
  mean Kendall -0.0554, p=0.1590). Shape check is only suggestive: quadratic
  term is negative but not conventionally significant (fresh quadratic p=0.0623,
  delta R2=0.00394). Comparability verdict: IPV is not comparable to the best
  single physical feature in this audited run; `driven_fde_m` is best with fresh
  R2=0.02085 and Spearman -0.2461, while ego IPV has fresh R2=0.00265 and
  Spearman -0.0445. Open-loop/closed-loop bias summary: driven trajectory is
  closest to the top-rated candidate in 181/302 retained scenes (59.9%) and
  driven IPV lies inside the candidate IPV range in 214/302 scenes (70.9%).
  Main audit table:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/results/rq010b_wod_e2e_multiframe_tracking_ipv_full479_scored_audited_20260630T063600/rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_selected_counterpart_summary.csv`;
  audit summary JSON/MD:
  `rq010b_wod_e2e_multiframe_tracking_ipv_full479_audited_audit_summary.{json,md}`
  in the same result directory.
- RQ009 Phase 3 features gate is now PASS. The hw=4 target source remains
  the verified frame-level `sigma01_hw4_ipv_timeseries.csv` with 3,695,981
  data rows, exact key overlap with sigma01, SHA-256
  `cf970f01455905000dac4f24909e69f532e21014987a52a541466a2748fd34fc`,
  and 12-case hw=4 parity `max_abs_diff=0.0`; the assembled feature matrix
  has 6,397,266 perspective-anchor rows under
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/`.
  Independent Phase 3.5 matrix audit `RQ009-W3-matrix-audit` is PASS with
  target t*+6 re-derivation `max_abs_diff=0.0` on 400 anchors, full-scale
  case-split no-bleed, max numeric M3 feature-target |corr| `0.1146074993`,
  and leakage-probe test R2 `0.2811922275`; report:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.md`.
- The 2026-06-27 `INV-ipv-code-diff` finding describes the pre-fix state after
  `a0fee535`: the first vectorized cost helpers drifted from the pinned
  sigma01-generation estimator. Commit `67f4c543` later restored the legacy
  loop backend as default `solver_mode="exact"` and repaired the vectorized
  `fast` backend. Current local estimator/profile tests pass (`6 passed`, one
  Linux-only strict check skipped), and verifier tests pass `8/8`. Final HPC job
  `1912947` reproduces sigma01 with `exact` at `max_abs_diff=4.44e-16`; the
  non-canonical `fast` backend differs from `exact` by `0.0016531` on that ABI.
  Cross-platform SLSQP still moves local exact output by about `0.0587`, so
  formal production uses the cloned sigma01 binary ABI and `solver_mode=exact`.
  Reproduction preserves `sigma=0.1`, `history_window`, `min_observation=4`,
  reference clip/max/smooth `60/40/40`, NuPlan 20-to-10 Hz downsampling, and the
  tracked `configs/ipv_sigma01_exact.json`; InterHub CLI reference defaults are
  now aligned to `60/40/40`.
- **sigma01 reproduction spot-check refreshed 2026-08-01.** On the clean managed
  HPC checkout at `6bdcc2e64bacd75d02741aa18ef5d61eef5a2962` with
  `envs/ipv-exact-sigma01` (Python 3.9.24), Slurm job `2022476`
  (`zxc-sigma01-fixture`) passed the strict two-case fixture with
  `sigma01_max_abs_diff=4.44e-16`. Slurm job `2022477`
  (`zxc-sigma01-onecase`) then reran real NuPlan case `ipv_000001` end to end:
  87 frame rows / 348 IPV-or-error values matched the archived sigma01 rows at
  max/mean absolute difference `1.11e-16` / `6.58e-18`; keys and timestamps were
  identical. The same case on macOS produced `max_abs_diff=1.12446`, so strict
  reproduction requires the pinned Linux/SciPy/BLAS ABI, not merely the same
  source and parameters. Current local full-PKL replay is also structurally
  incomplete: AV2/Lyft PKLs are absent and `waymo_300-499.pkl` remains truncated;
  9 of the 10 visible local PKLs deserialize successfully. These local defects do
  not affect the immutable managed-HPC snapshot used by the passing spot-check.
- Git-based HPC deployment is active at
  `/share/home/u25310231/ZXC/sociality_estimation/code/repo`; the 2026-07-11/12
  root-cure cutover was validated at `47f79685`, and deployment follows
  published `main` commits. Exact and verifier environments are isolated under
  `envs/ipv-exact-sigma01` (Python 3.9.24) and
  `envs/ipv-verifier` (Python 3.9.6); their conda/pip locks are tracked under
  `environments/`. Portable private scorer SHA-256
  `b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253`
  is checksum-bound at `checkpoints/rq009_m3/`; final verifier job `1912948`
  passed `8/8`. The historical `/share/home/u25310231/ZXC/ipv_estimation`
  execution surface is retired: it contains only a tombstone and raw/results
  compatibility links. Its Git bundle, dirty patch, source/tests/tools, and
  manifests are checksum-archived under
  `/share/home/u25310231/ZXC/sociality_estimation/archives/legacy-code/`.
  InterHub raw/results are immutable snapshots registered by 51 and 173,034
  fresh SHA-256 checks respectively; managed post-switch preflight jobs
  `1915718` and `1915764` both matched all 7,500 CSV/PKL events. Deployment guide:
  `docs/reproducible_ipv_pipeline.md`. Historical investigation notes:
  `reports/knowledge/_analysis/ipv_estimator_divergence_investigation.md` and
  `reports/knowledge/_analysis/ipv_accel_hyperparam_finding.md`.
- RQ014 must use this managed topology and the single checksum-bound, clean-environment bootstrap
  frozen in `configs/run_specs/README.md` and `RQ014_execution_contract_v1p5.json`; invoking
  `scripts/hpc/submit_research_run.sh` without that wrapper-hash gate, direct `sbatch`,
  `/share/home/u25310231/ZXC/RQ014_recovery`, the retired `ipv_estimation` checkout, and external
  RQ010B code execution are forbidden. The central allowlist used for job `1924193` contained exactly
  the staged declassification export and contract preflight. Preflight completed PASS; resource pilot
  and all later operations remain machine-denied pending their separate gates and authorization.
- RQ009 Phase 4 calibration and independent Phase 4.5 calibration-integrity
  audit are now PASS. M3 test coverage reproduces at 80=`0.8162154701`,
  90=`0.8986657101`, and 95=`0.9496345436`; M3 conformal radii reproduce
  from calibration only at 80=`-0.0041994299`, 90=`-0.0080911424`, and
  95=`-0.0054183006`; no test-fold leakage was detected and calibration/test
  scene/case overlap is zero. Audit caveat: the `1e-10` endpoint nudge changes
  exact M3 80% boundary-tie coverage but only for rows within the 1e-10
  tolerance; this is recorded as nonblocking. Reports:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/04_calibration/calibration_report.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/04_calibration/calibration_audit.md`.
- RQ012B Stage 3+ callable frozen-M3 scorer build is PASS. The scorer reuses
  RQ009 calibration code, refits only M3 per-quantile HGB models with the
  frozen selected hyperparameters and seed `20260626`, uses saved M3 conformal
  radii and saved OOD gate threshold/support parameters, and touches no OnSite
  outcomes. The original serialized scorer/helper/contract are provenance-only
  and were moved out of the active derived topology to
  `data/derived/_provenance_archive/rq009_m3_legacy_source_20260711/`.
  Runtime code must use the manifest-verified portable bundle under
  `models/rq009_m3/` locally or `checkpoints/rq009_m3/` on HPC.
  Refit/parity/provenance:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/m3_refit/`.
  Saved RQ009 M3 calibration+test prediction parity is exact for materialized
  q/interval/abstain columns (`max_abs_diff=0.0`, fraction within `1e-5` =
  `1.0`); scorer SHA-256
  `bf9a0c7ae41ba9efcb2ad997aaac1b7881d7788cf8dadd01252c17ed7a6b0ba5`.
- RQ009 Phase 6 M3-vs-M4 gate is PASS with no stable incremental
  counterpart-IPV interval value over `ipv_removed` (M3 90% Winkler
  difference `-0.000211426`, case-cluster 95% CI
  `[-0.0018861798657293647, 0.00150497909450504]`). Phase 6.5b
  exploration group G5/C15 is complete on guard_tune only: dependency is not
  approximately zero by the partial-Pearson threshold (`0.0315120479`), but
  point and interval screens do not improve the matched control
  (`dR2=-0.003274902`, `dMAE=-0.0001594282`,
  `dWinkler90%=0.1898183`, coverage delta `0.00001536`). Phase 6.5b
  exploration group G2/C05,C08,C09 is complete on guard_tune only:
  dependencies were nonzero (`partial_r=-0.1411410`, `-0.1160929`,
  `-0.1009250`), all point screens were worse than matched controls
  (least-bad `C09 dR2=-0.0068356`, `dMAE=0.0002454`), and the best interval
  screen was C05 with a small Winkler reduction (`dWinkler90%=-0.9070%`, 95%
  CI `[-1.0933%, -0.7357%]`, coverage rule OK) below the 5% meaningful-effect
  threshold. Phase 6.5b
  exploration group G3/C06-C07/C10-C12 is also complete on guard_tune only:
  C10/C11 were sparse/ineligible with zero matching guard rows; C06/C07/C12
  dependencies were nonzero (`partial_r=-0.1492908`, `-0.1591437`,
  `-0.1182023`), point screens were worse than matched controls (least-bad
  `C12 dR2=-0.0132819`; least-bad MAE `C07 dMAE=0.0008816`), and only C06
  showed a small interval Winkler reduction (`dWinkler90%=-0.5748%`, 95% CI
  `[-0.7061%, -0.4280%]`, coverage rule OK) below the 5% meaningful-effect
  threshold. Phase 6.5b
  exploration group G1/C01-C04 is complete on guard_tune only: all dependency
  probes were nonzero (`partial_r=-0.1390962`, `-0.2381406`,
  `-0.2212075`, `-0.0319065`), all point screens were worse than matched
  controls (least-bad `C04 dR2=-0.0009120`, `dMAE=0.0007248`), and the best
  interval screen was C03 with a small Winkler reduction
  (`dWinkler90%=-1.1902%`, 95% CI `[-1.5128%, -0.8373%]`, coverage rule OK)
  below the 5% meaningful-effect threshold. Phase 6.5b
  exploration group G4/C13-C14 is complete on guard_tune only: both dependency
  probes were nonzero (`partial_r=-0.1623032` for both C13 and C14); C13 was
  worse on point metrics (`dR2=-0.0075118`, `dMAE=0.0004988`), while C14 had a
  tiny MAE reduction but worse R2/MSE (`dR2=-0.0009504`,
  `dMAE=-0.0001637`, `dMSE=0.0001786`). Both interval screens showed small
  Winkler reductions with coverage rule OK (`C13 dWinkler90%=-0.4868%`, 95%
  CI `[-0.6499%, -0.3213%]`; `C14 dWinkler90%=-0.3591%`, 95% CI
  `[-0.4368%, -0.2757%]`), below the 5% meaningful-effect threshold. Phase
  6.5c exploration synthesis is PASS and formalizes the guard_tune verdict as
  `null_confirmed`: all 15 candidates were aggregated, no point or interval
  screen was both BH-significant and pre-registered meaningful, the best point
  dR2 remained negative (`C04=-0.0009120`), the best interval Winkler reduction
  was small (`C03=-1.1902%`, below the 5% bar), and no test confirmation was
  triggered. The test fold remains untouched. Dependency caveat: raw engineered
  dependency probes were often nonzero (max absolute partial r `C02=0.2381406`);
  C15's orthogonalized probe was small (`partial_r=0.0315120`,
  `spearman=0.0142350`, `CMI=0.0006338`) but above the strict preregistered
  `<0.02` partial-r approximate-zero threshold, so the robust-null claim is a
  performance/adaptation null rather than a literal independence claim.
  Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/m3_vs_m4_verdict.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G5.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G2.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G3.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G1.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/results/results_G4.md`
  plus
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/exploration_verdict.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/exploration_master_table.csv`.
- RQ009 Phase 6.5d longer-horizon sweep is complete under
  `02_process/06_m3_vs_m4/exploration/horizon/`. Registered horizons
  `h={6,8,11,13,16,18,21}` were run on train/calibration/guard_tune only with
  lookup targets from the frame-level hw=4 time series; the test fold remains
  untouched because no horizon/encoding cleared the guard_tune bar. Decision:
  `null_across_horizons` for point/interval adaptation. Eligibility shrank from
  5,126,700 analysis anchors / 30,566 cases at h=6 to 4,215,974 anchors /
  29,972 cases at h=21. Best point screen was h=6 C08
  (`dR2=0.001715`, below the 0.02 bar and MAE worse); best interval screen was
  h=6 C08 (`dWinkler90%=-0.2405%`, far below the 5% bar). Dependency caveat:
  the registered raw/ego-controlled counterpart-current probe did not reproduce
  the prior C15-style h=6 near-zero partial-r sanity (`h6=-0.1090`, max abs
  `0.1254` at h=18), so treat the horizon conclusion as a guarded
  point/interval adaptation null rather than a literal independence result.
  Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/06_m3_vs_m4/exploration/horizon/horizon_verdict.md`,
  `horizon_verdict.json`, `horizon_sweep.csv`, and
  `horizon_partial_r_curve.csv`/`.svg`.
- RQ009 Phase 6.5e dependency reconciliation is complete under
  `02_process/06_m3_vs_m4/exploration/reconcile/`. Canonical residualization
  uses `counterpart_ipv_current` only, with base context plus M4 ego
  self-anchor current/error/slope, and HistGradientBoosting residualizers for
  ego target and counterpart current. The h=6 canonical partial r is
  in-sample `-0.039023`, train-to-guard row-level `-0.037680`, and held-out
  case-level `0.057521`; across h={6,11,16,21}, train-to-guard row-level r
  stays small (`-0.0345` to `-0.0393`) and does not grow with horizon, while
  held-out case-level r flips positive and is not monotone (`0.0226` to
  `0.0575`). C15 vs horizon differed because C15 used an orthogonalized
  five-column counterpart-block PCA component screen with max-absolute
  component correlations, while the horizon sweep used signed raw
  `counterpart_ipv_current` with Ridge residualization and ego-self controls.
  Predictive reconciliation: Ridge dR2 is small but positive
  (`0.010609`, `0.014090`, `0.015037` for h=6,16,21), below the 2% bar, while
  flexible HGB dR2 is negative at all checked horizons (`-0.005250`,
  `-0.005854`, `-0.012110`). Verdict: `robust_null` for a real,
  cross-case-generalizing counterpart-current dependency; row-level negative
  sign, where present, would be compensatory under the local convention
  `theta>0` prosocial/yielding. The test fold remains untouched. Artifacts:
  `dependency_reconcile.md`, `dependency_reconcile.json`, and
  `run_dependency_reconcile.py`.
- RQ009 Phase 7 perturbation sensitivity is complete under
  `02_process/07_perturbation/`. The worker used outcome-blind,
  source-stratified case subsamples from train/calibration/guard_tune only
  (targets 50k/35k/45k rows before full-case overshoot; guard_tune reporting;
  test fold not read) and refit/recalibrated M2/M3/M4/`ipv_removed` for
  feature-window, counterpart-noise, missingness, OOD-gate, subsample-seed,
  and target-horizon perturbations. M3 90% guard_tune coverage ranged
  `0.8754..0.9119`, mean width `0.9703..1.1144`, Winkler `1.5265..1.7217`,
  and paired M3-vs-`ipv_removed` relative Winkler gain ranged
  `-0.541%..1.193%`. No validity break outside +/-3 pp and no meaningful
  counterpart-null flip were found; gate booleans are
  `validity_robust=true`, `null_robust=true`. Artifacts:
  `perturbation_results.csv`, `perturbation_report.md`,
  `perturbation_gate.json`, and `perturbation.py`.
- RQ009 Phase 8 independent end-to-end review (`RQ009-W8-review`) is PASS
  under `02_process/08_review/`. The review found no blocking or major issues:
  contract adherence is OK, leakage controls are clean, M3 marginal
  gate-passing conformal validity is sound, and the counterpart-IPV null is
  defensible as a bounded performance/adaptation null rather than a literal
  independence claim. Independent spot checks reproduced the feature matrix
  counts (`6,397,266` rows, no case split bleed), target timing (`+2/+6`),
  M3 calibration radii and test coverage, calibration/test case overlap `0`,
  and the effective scored target zero atom (`273,819 / 1,270,566`). Two minor
  hygiene findings remain for final packaging: top-level run status/index
  artifacts are stale, and exploration p-values were reconstructed from saved
  CIs because raw bootstrap draws/case-level paired differences were not
  retained. Artifacts: `independent_review.md`, `review_findings.csv`,
  `review_gate.json`, and `execution_log.md`.
- RQ009 Phase 10 clean-room replication (`RQ009-W10-replication`) is complete
  under `02_process/10_replication/` with status `FAIL` for a documented
  divergence rather than a leakage/blocking failure. The independent route used
  HGB quantile CQR, linear conformal quantiles, and a train/guard robust-distance
  support gate without importing original calibration/evaluation code. M3
  coverage agrees at 90% (`0.898762` vs original `0.898666`) with 90% width
  `1.067353` vs original `1.016152`; M3-vs-M4 agrees in direction and scale
  (`width +4.929%`, `Winkler +1.979%`, both below PI escalation bars, vs
  original `+2.960%` and `+2.784%`). The paired counterpart-null diverges:
  M3-minus-`ipv_removed` interval-score difference is small in practical terms
  (`-0.004664`, `-0.334%`) but case sign-test p=`1.72e-10`, unlike the original
  near-zero/sign p=`0.8629`; held-out row-level dependency remains small and
  agrees (`r=-0.0234` vs canonical about `-0.04`). Artifacts:
  `replicate.py`, `replication_results.csv`, `replication_report.md`,
  `replication_gate.json`, and `execution_log.md`.
- RQ009 Phase 10b replication-null reconciliation (`RQ009-W10b-recon`) is
  complete under `02_process/10_replication/` with the practical null
  reconciled. Frozen M3 and `ipv_removed` test predictions reproduce the
  original 90% paired row-weighted Winkler difference
  `-0.000211426` (`-0.014856%` of `ipv_removed`), case sign-test
  p=`0.862943`, case Wilcoxon p=`0.522202`, and case-cluster CI containing
  zero (recomputed `[-0.001848, 0.001454]`; original Phase 6
  `[-0.001886, 0.001505]`). Row-level sign/Wilcoxon tests are tiny-p because
  they count 1,209,857 autocorrelated anchors; the naive paired t-test is not
  significant (`p=0.534857`). The clean-room `-0.333852%` result remains far
  below the 5% meaningful-effect bar, and its saved p-value `1.7158e-10` is
  labeled `paired_case_sign_p` in the replication code/results, not row-level.
  Artifacts: `replication_reconcile.md` and `replication_reconcile.json`.
- RQ009 Phase 11 visualization/report package (`RQ009-W11-report`) is PASS.
  Nature-skill figure generation was available and used; the offline bilingual
  report package has seven conclusion-owned figure groups, 14 evidence rows,
  and a PASS `report_gate.json` with `offline_ok=true`. Entry point:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/00_entry/index.html`.
  English/Chinese reports:
  `90_report/index.html` and `90_report/index.zh.html`. Figure manifest:
  `01_results/figures/figure_manifest.csv`. The package headline is: marginally
  valid CQR envelope, counterpart-IPV practically null, and adaptation encoded
  mainly in kinematics/context.
- RQ009 Phase 12a final independent report review
  (`RQ009-W12a-final-review`) is PASS and `ready_to_register=true` after the
  W11b report fix. The final review verified offline EN/ZH link resolution,
  bilingual correspondence, C1-C7 figure binding, headline number consistency,
  honest practical-null/marginal-validity boundaries, and `evidence.csv`
  consistency with zero blocking, major, or minor findings. Artifacts:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/12_final_review/final_review.md`,
  `final_review_findings.csv`, and `final_review_gate.json`. No report,
  registry, contract, or paper-repository files were edited by the final
  reviewer.

## Canonical Code Entrypoints

- Core IPV package: `src/sociality_estimation/core/`.
- Planning and geometry helpers: `src/sociality_estimation/planning/`.
- Active InterHub CSV/pkl pipeline: `pipelines/interhub/process_interhub.py`.
- Active simulation entrypoint: `pipelines/simulation/simulator.py`.
- InterHub helper/report scripts: `pipelines/interhub/tools/`.
- Old root wrappers are archived under `archived/compat_wrappers_20260619/`.

## Convenience Launchers

- macOS launchers: `scripts/launch_claude.command` and `scripts/launch_codex.command`.
- They enter the project root and start the corresponding CLI through the current team launcher.
- If the launcher or CLI is unavailable, leave the Terminal window open for diagnosis.

## Canonical Research Paths

- Compact index: `STUDIES.md`.
- Program dashboard: `reports/knowledge/RQ_PROGRESS_DASHBOARD.md`.
- Machine registry: `reports/knowledge/rq_progress_registry.csv`.
- Centralized plans/prompts: `reports/plans/`.
- Execution/report layer: `reports/studies/`.
- Interpretation/review/decision layer: `reports/knowledge/`.
- `reports/` has three governed first-level directories: `plans/`, `studies/`, `knowledge/`.
- Large derived outputs: `data/derived/`.
- Report-linked process archives and local agent state:
  `archived/report_process/` and `archived/report_local_state/`.
- Manuscript drafting lives in the standalone paper repository:
  `../9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`.
  Do not recreate a top-level `paper/` directory here.

## Repository State

对齐时间 2026-08-04T15:50Z（每次分支或远端状态变化时更新本节）。

- **RQ015A / RQ015K/L / RQ016 / RQ016B / RQ016C / RQ017 全部已合入 `main`**，
  经 PR #46（38 个提交，337 文件，零冲突，merge commit `caa4f09`）。
  这几轮的成果不再只存在于某条特性分支上。
- 工作分支 `rq018-next-round`，从合并后的 `main` 开出，尚未推送。
  **分支名是占位**：下一轮 RQ 编号确定后应改名（推送前改名无成本）。
- 分支已整理：远端 22 条降到 3 条，本地 6 条降到 4 条。
  删除依据是「相对 `main` 领先 0 个提交」，即无独有内容。
  - 远端保留：`main`、`rq007-estimability-run`、`codex/rq014-spawn-path-fix`
    （后者 PR #36 已关闭未合并，尚有 1 个独有提交，故未删）。
  - 本地额外保留两条历史分支：`archive/rq014-wod-e2e-recovery-20260712`、
    `codex/backup-rq012-diverged-20260623T201226`。三条历史分支各落后 `main`
    136–190 个提交，内容主体已在 `main` 内但仍有零星差异，**2026-08-04 PI 裁定暂留**，
    未做逐文件核查。
- `reports/` 的三个一级目录 `plans/` / `studies/` / `knowledge/` 为治理层，
  `AGENTS.md`、`STUDIES.md`、本文件三处表述已一致（2026-08-04 PI 裁定保留 `plans/`）。
- `.git` 已做过一次 `git gc`：5.0 GB → 19 MB，`git fsck` 无错，六条分支与工作区
  逐一核对无损。**未做任何历史改写。** 44 个 `tmp_obj_*` 垃圾文件已手动清除。
- `.codex-fleet/` 已清理大块中间数据：**6.5 GB → 384 MB**。删除项、引用检查依据与再生方式
  见 `reports/knowledge/_governance/codex_fleet_cleanup_20260804/`。
  **承重产物全部保留**，其中 RQ017 依赖的
  `.codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl`
  与 `H2/onsite_scoring_dryrun.parquet` 清理后已功能性复验
  （67,861 行 / 67,861 唯一键 / `mechanism2_gate_ok` 为真 21,936，与记录一致）。
  RQ016C 的 H1 产物（已被 H2 取代且判定不可用）已删除，H1 报告内已加注。
- 受保护文件基线校验清单：`.codex-fleet/git_cleanup_protected_sha_before.txt`
  （`shasum -a 256 -c` 应全部 `OK`）。

## Active Study Map

| RQ | Study folder | Knowledge folder |
|---|---|---|
| RQ001 online IPV interval | `reports/studies/RQ001_online_ipv_interval/` | `reports/knowledge/RQ001_online_ipv_interval/` |
| RQ002 self-anchor group norm | `reports/studies/RQ002_self_anchor_group_norm/` | `reports/knowledge/RQ002_self_anchor_group_norm/` |
| RQ003 NSFC external evidence | `reports/studies/RQ003_nsfc_external_evidence/` | `reports/knowledge/RQ003_nsfc_external_evidence/` |
| RQ004 IPV state space | `reports/studies/RQ004_ipv_state_space/` | `reports/knowledge/RQ004_ipv_state_space/` |
| RQ005 NMI evidence gap | `reports/studies/RQ005_nmi_evidence_gap/` | `reports/knowledge/RQ005_nmi_evidence_gap/` |
| RQ006 sigma sensitivity | `reports/studies/RQ006_sigma_sensitivity/` | `reports/knowledge/RQ006_sigma_sensitivity/` |
| RQ007 interaction-conditioned IPV estimability | `reports/studies/RQ007_interaction_conditioned_ipv_estimability/` | `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/` |
| RQ008 temporal IPV discovery | `reports/studies/RQ008_interhub_temporal_ipv_discovery/` | `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/` |
| RQ009 dynamic counterpart envelope | `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/` | `reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/` |
| RQ010 WOD-E2E tracking feasibility | `reports/studies/RQ010_wod_e2e_tracking_feasibility/` | `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/` |
| RQ011 OnSite readiness | `reports/studies/RQ011_onsite_full_universe_readiness/` | `reports/knowledge/RQ011_onsite_full_universe_readiness/` |
| RQ012 event/annotation readiness | `reports/studies/RQ012_onsite_event_annotation_readiness/` | `reports/knowledge/RQ012_onsite_event_annotation_readiness/` |
| RQ013 beyond-safety utility | `reports/studies/RQ013_beyond_safety_incremental_validity/` | `reports/knowledge/RQ013_beyond_safety_incremental_validity/` |

For parallel agent runs under one RQ, the number after the RQ stem is the execution version.
Each execution must create a unique atomically locked RUN_ID/RUN_ROOT.

## Current PI Decisions

- Launch RQ009 now.
- Do not run RQ008B.
- Keep RQ007 held-out sealed until RQ009 reaches its pre-opening freeze; request a new PI
  authorization before any read.
- Defer two-human RQ012 annotation.
- Authorize WOD-E2E signed-in manifest/pilot work in principle; account/licence/login must be
  completed by the user.
- Prioritize OnSite RQ011B after RQ009; WOD proceeds in parallel.
- Use paper `main` commit `c6783577` as the current v4.1 baseline.

## Canonical Data Paths

- InterHub subset CSV:
  `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- InterHub subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- InterHub full-dataset raw data: `data/interhub/raw/full_datasets/`
- InterHub sigma 0.1 time-series and full-rerun outputs:
  `data/derived/interhub/20260612_sigma_0_1_full_rerun/00_hpc_outputs/`
- RQ009 dynamic envelope hw=4 target source:
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/target_hw4/sigma01_hw4_ipv_timeseries.csv`
  (verification report:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/target_hw4_fetch.md`).
- RQ009 dynamic envelope Phase 3 feature matrix:
  `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/matrix/`
  with gate:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/features_gate.json`.
  Independent Phase 3.5 audit:
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.md`
  and
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/matrix_audit.json`.
- RQ010B WOD-E2E Tongji HPC work root:
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/`; parser env at
  `envs/e2e`, structural pre-flight code at `src/e2e_structural_preflight.py`,
  and latest four-shard result at
  `results/e2e_structural_preflight_4shards_20260626.json`. StreamPETR Route 4
  setup is at `code/StreamPETR` with env `envs/streampetr`, checkpoint
  `checkpoints/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, and latest
  checkpoint smoke log `logs/streampetr_checkpoint_forward_flash_l40_20260626.log`.
  Waymo Perception v1.4.3 dev and finetune subsets for StreamPETR
  dataloader/calibration work are at `data/perception/{training,validation}/`
  with manifests `manifests/perception_dev.tsv` (6 files, all crc32c matched on
  2026-06-26), `manifests/perception_finetune.tsv` (64 training plus 16
  validation files, all crc32c matched on 2026-06-27; total bytes
  80,523,139,102), and current `manifests/perception_256.tsv` (256 training
  plus 16 validation files, 272/272 crc32c ok).
  Latest Route 4 real-Waymo StreamPETR smoke artifacts are
  `data/waymo_stream_petr/waymo_infos_train_1seg.pkl`,
  `checkpoints/stream_petr_waymo3_reinit_cls.pth`, and
  `code/StreamPETR/tools/waymo_perception/RQ010B_ROUTE4_WAYMO_STREAM_PETR_SMOKE.md`.
  Latest Route 4 dev6 train/eval dry-run artifacts are
  `data/waymo_stream_petr/waymo_infos_train_4seg.pkl`,
  `data/waymo_stream_petr/waymo_infos_val_2seg.pkl`,
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_dev6_dryrun.py`,
  `work_dirs/streampetr_waymo_dev6_dryrun_20260626/iter_40.pth`, and summary
  `logs/streampetr_waymo_dev6_dryrun_summary_20260626.md`.
  Latest Route 4 64/16 finetune artifacts are config
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune64_leadcfg.py`,
  script `scripts/streampetr_waymo_finetune64_leadcfg_20260627.sbatch`, work
  dir `work_dirs/streampetr_waymo_finetune64_leadcfg_20260627/`, logs
  `logs/streampetr_waymo_finetune64_1707389.out`/`.err`, and existing converted
  infos `data/waymo_stream_petr/waymo_infos_train_64seg.pkl` plus
  `data/waymo_stream_petr/waymo_infos_val_16seg.pkl`. Latest §5 detector
  quality/error-model outputs are under `results/` with prefix
  `rq010b_route4_detector_quality_best50732_20260627`, especially
  `_summary.json`, `_metrics_by_class_range.csv`, `_error_model.json`,
  `_error_model.csv`, `_matched_tp_errors.csv`, and `_threshold_sweep.csv`.
  Latest improved recipe config/smoke/train/eval artifacts are
  `projects/configs/StreamPETR/stream_petr_r50_flash_704_waymo_5cam_finetune256_balanced_warminit.py`,
  warm-init checkpoint
  `checkpoints/stream_petr_waymo3_warminit_nusc_car_ped_bicycle.pth`,
  converted infos `data/waymo_stream_petr/waymo_infos_train_256seg.pkl` plus
  `data/waymo_stream_petr/waymo_infos_val_16seg.pkl`,
  support files
  `projects/mmdet3d_plugin/datasets/waymo_ap_dataset.py`,
  `projects/mmdet3d_plugin/core/evaluation/waymo_early_stopping.py`,
  `tools/waymo_perception/make_waymo_warminit_checkpoint.py`, and
  `tools/waymo_perception/smoke_train_waymo_ddp.py`, DDP smoke logs
  `logs/streampetr_waymo_finetune256_ddp_smoke_1712408.log`/`.jsonl`,
  failed Slurm jobs `1712416` and `1712590`, completed train-only resume Slurm
  job `1712698`, single-GPU detector-quality eval job `1745613`, no-eval resume sbatch
  `scripts/streampetr_waymo_finetune256_balanced_warminit_ddp4_resume_noeval_20260628.sbatch`,
  work dir
  `work_dirs/streampetr_waymo_finetune256_balanced_warminit_ddp4_20260628/`,
  final checkpoint `latest.pth -> iter_152124.pth`, checkpoint ranking outputs
  `results/rq010b_route4_detector_quality_256_balwarm_20260629_checkpoint_summary.{json,csv}`,
  best detector-quality/error-model outputs under prefix
  `results/rq010b_route4_detector_quality_256_balwarm_ep12_iter_152124_20260629`,
  single-GPU eval sbatch
  `scripts/run_rq010b_detector_quality_256_eval.sbatch`, and Slurm logs
  `logs/streampetr_waymo_finetune256_ddp4_resume_noeval_1712698.out`/`.err`
  (failed run logs remain
  `logs/streampetr_waymo_finetune256_ddp4_1712416.out`/`.err` and
  `logs/streampetr_waymo_finetune256_ddp4_resume_1712590.out`/`.err`).
- Onsite competition current all-team package, generated locally and ignored:
  `data/onsite_competition/all_teams_dataset/` (rebuild with
  `scripts/build_onsite_all_teams_dataset.py`)
- Onsite competition lightweight manifests: `data/onsite_competition/00_manifest/`
- Onsite competition archived raw/top-five subset payload:
  `archived/onsite_competition_raw_and_top5_subset_20260623/`
- Legacy Argoverse source data:
  `archived/argoverse/0_souce_data/` (typo is historical).

## Key Report Entries

- RQ001 deployable online interval report:
  `reports/studies/RQ001_online_ipv_interval/RQ001_3_online_interval_lock_20260619/00_entry/index.html`
- RQ002 main self-anchor validation:
  `reports/studies/RQ002_self_anchor_group_norm/RQ002_1_self_anchor_validation_main_20260619/00_entry/index.html`
- RQ002 parallel Codex validation:
  `reports/studies/RQ002_self_anchor_group_norm/RQ002_2_self_anchor_validation_codex_20260619/00_entry/index.html`
- RQ003 core NSFC evidence:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_1_nsfc_core_evidence_20260618/00_entry/core_results_nature.html`
- RQ003 detailed synthesis:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_2_nsfc_detailed_synthesis_20260619/00_entry/index.html`
- RQ003 parallel Codex open exploration:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_5_nsfc_open_explore_codex_20260619/00_entry/index.html`
- RQ003 Tier B NSFC IPV validation final reader:
- OnSite all-team package, generated locally and ignored:
  `data/onsite_competition/all_teams_dataset/`
  (rebuild with `scripts/build_onsite_all_teams_dataset.py`).
- OnSite lightweight manifests: `data/onsite_competition/00_manifest/`.
- OnSite archived raw/top-five payload:
  `archived/onsite_competition_raw_and_top5_subset_20260623/`.
- Legacy Argoverse source data: `archived/argoverse/0_souce_data/`.

## Key Report And Decision Entries

- RQ003 Tier B validation:
  `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/00_entry/index.html`
- RQ007 estimability report:
  `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/00_entry/index.html`
- RQ008 InterHub temporal IPV discovery report (negative discovery-only result;
  knowledge `decision.md` frozen 2026-06-24; 0/24 candidates survived,
  confirmation split remains unopened):
  `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/00_entry/index.html`
- RQ009 dynamic counterpart-conditioned envelope bilingual report (Nature-style
  conclusion-owned figures; Phase 11 report gate PASS, offline EN/ZH):
  `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/00_entry/index.html`
- RQ010 WOD-E2E tracking feasibility report (`T2_FULL_TRACKING_REQUIRED`;
  knowledge `decision.md` frozen 2026-06-24; Route 4 preferred,
  Route 5 fallback; basic Tongji HPC parser/pre-flight access verified on
  four validation shards 2026-06-26, full gate pending):
  `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/00_entry/index.html`
- RQ012A OnSite event annotation readiness Wave-A package (9 automatic events;
  gates 012-0/012-1 pass, 012-2 text-cleared, 012-3 ready-pending-humans,
  012B blocked; knowledge `decision.md` freezes the deferral, not a full PASS):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html`
- RQ012B W0 frozen automatic extractor health bilingual report (no
  outcome/IPV/deviation association; scientific endpoint
  `BLOCKED_PENDING_M3`; clean_285 attempted 285, succeeded 280; precedence
  suppression 2.6569%, identity stability 100% raw intervals; coarser sampling
  unstable):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/00_entry/index.html`
- RQ012B W0 independent blind replication of frozen extractor health (no
  outcome/IPV/deviation association; native counts near but not exact versus
  W0, computability 280/285 agrees; principled uniform-grid resampling remains
  unstable at 5 Hz +88.1% and 20 Hz -30.4% total primary events):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/02_process/08_replication/replication_report.md`
- RQ012B W0 extractor-health publication figures (Nature-style, extractor
  evidence only; PNG/PDF/SVG plus source data and manifest; no outcome/IPV/
  ranking/harm endpoint plotted):
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_1_event_harm_20260625T202307+0800_38f47437/01_results/figures/figure_manifest.md`
- RQ011A OnSite full-universe readiness (re-run on complete data; `READY_WITH_FROZEN_EXCLUSIONS`:
  outcome universe full 300 / replay 285 with T19 excluded; run-level & repeated-run not identifiable
  by design; knowledge `decision.md` frozen 2026-06-24; supersedes the suspended
  RQ011_1 incomplete-data run):
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`
- RQ011B OnSite matched-scenario run paused after phases 0-2 at the phase-3 gate (P1 FAIL blockers B001-B005; P2 PASS; resume requires RQ009 M3 downstream clearance plus PI-approved SAP/controls): `reports/studies/RQ011_onsite_full_universe_readiness/RQ011B_1_matched_scenario_20260625T202454_8331bd49/02_process/00_meta/PAUSE_STATE.md`

## Latest Review Packets

- RQ001 Codex review:
  `reports/knowledge/RQ001_online_ipv_interval/reviews/codex_review.md`
- RQ002 Codex review:
  `reports/knowledge/RQ002_self_anchor_group_norm/reviews/codex_review.md`
- RQ004 Codex review:
  `reports/knowledge/RQ004_ipv_state_space/reviews/codex_review.md`
- RQ005 Codex review:
  `reports/knowledge/RQ005_nmi_evidence_gap/reviews/codex_review.md`
- RQ006 Codex review:
  `reports/knowledge/RQ006_sigma_sensitivity/reviews/codex_review.md`
- RQ007 Codex review:
  `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/reviews/codex_review.md`
- RQ008 Codex review:
  `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/reviews/codex_review.md`
- RQ010 Codex review:
  `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/reviews/codex_review.md`
- RQ011 Codex review:
  `reports/knowledge/RQ011_onsite_full_universe_readiness/reviews/codex_review.md`
- RQ012 Codex review:
  `reports/knowledge/RQ012_onsite_event_annotation_readiness/reviews/codex_review.md`

These review packets are evidence-boundary reviews, not accepted
`decision.md` freezes.

## Latest Decision Packets

- RQ007 accepted development/guard estimability boundary:
- RQ007 decision:
  `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`
- RQ008 negative temporal-discovery report:
  `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/00_entry/index.html`
- RQ008 decision:
  `reports/knowledge/RQ008_interhub_temporal_ipv_discovery/decision.md`
- RQ010 tracking-feasibility report:
  `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010_1_wod_tracking_feasibility_20260623T073830+0800_14f21d3e/00_entry/index.html`
- RQ010 decision:
  `reports/knowledge/RQ010_wod_e2e_tracking_feasibility/decision.md`
- RQ011 complete-data readiness report:
  `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`
- RQ011 decision:
  `reports/knowledge/RQ011_onsite_full_universe_readiness/decision.md`
- RQ012 readiness report:
  `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/90_report/index.html`
- RQ012 decision:
  `reports/knowledge/RQ012_onsite_event_annotation_readiness/decision.md`

## How To Run Tests

- Launcher checks: `python3 -m unittest tests.test_shortcut_scripts -q`.
- Broader suite when available: `python -m pytest tests -q`.
- Syntax check:
  `python -m py_compile src/sociality_estimation/core/agent.py src/sociality_estimation/core/ipv_estimation.py src/sociality_estimation/planning/Lattice.py src/sociality_estimation/planning/lattice_planner.py src/sociality_estimation/planning/utility.py pipelines/interhub/process_interhub.py pipelines/simulation/simulator.py`.
- One-case InterHub smoke:
  `python pipelines/interhub/process_interhub.py --limit 1 --workers 1 --solver-preset realtime --no-plots --output-root data/derived/interhub/_codex_runtime_smoke`.
- Record any durable dependency change in requirements or `main_workflow.log`.

## What Not To Delete

- Raw/local data under `data/interhub/raw/`, `data/onsite_competition/all_teams_dataset/`,
  `archived/onsite_competition_raw_and_top5_subset_20260623/`, and
  `archived/argoverse/0_souce_data/`.
- Derived InterHub full-rerun outputs under `data/derived/interhub/`.
- Plans/prompts under `reports/plans/`.
- Reader-facing study report packages under `reports/studies/`.
- Knowledge decisions and manuscript context under `reports/knowledge/`.
- Report-linked process archives under `archived/report_process/`.
- `main_workflow.log`, `AGENTS.md`, `START_HERE.md`, `PROJECT_STRUCTURE.md`, and `STUDIES.md`.

## Known Weak Spots

- NuPlan remains the weakest realtime IPV slice; no dataset-specific >90% guarantee.
- Self-anchor remains M4 ablation only, not normative authority.
- RQ007 is a development/guard estimability boundary; held-out is sealed and most gross
  concentration is proximity-driven.
- RQ008 supports a negative directional temporal-discovery boundary, not proof that all temporal
  dynamics are absent; RQ008B is currently not authorized.
- RQ009 must not read RQ007 sealed data until all rules/code/thresholds are frozen and the PI
  explicitly authorizes opening.
- RQ010 requires full tracking; exact data/HPC scale remains sign-in gated.
- RQ010B Route 4 64-segment StreamPETR is not tracker-ready: §5 detector
  quality on 16 Perception validation segments has overall 2 m center-distance
  AP `0.00328`, recall `0.08034`, precision `0.03276`, and zero
  Pedestrian/Cyclist detections at the selected operating point. The improved
  256-seg balanced/warm-init checkpoint ep12 `iter_152124` is the current best
  Route 4 pilot detector and is tracker-QA-ready for the 16-val pilot:
  `mAP_9=0.08454`, pooled AP `0.10835`, recall `0.21916`, precision `0.23675`,
  with nonzero Pedestrian and Cyclist detections. Remaining weak spots are
  far-range quality, small Cyclist sample size, and the fact that this is still
  a 16-val pilot rather than a final full-data detector validation.
- RQ011 supports full_300 outcomes and clean_285 replay/IPV with T19 replay-only exclusion;
  run-level/repeated-run/causal claims are unavailable.
- RQ012 is readiness-only and human annotation is deferred.
- Paper `main` is v4.1 but still carries evidence/external-pending markers and is not submission-ready.
