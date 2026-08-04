# RQ015A Plan v8（覆盖范围与取回结论定稿正文）— IPV 估计尝试状态与候选权重集中度回溯审计

状态：`POST_FINAL_REVIEW_REVISION / BUILD_WHILE_DENY`｜`formal_g1_eligible=false`｜`execution_authorized=false`  
日期：2026-07-31｜另存自 v7（v1/v2/v3/v4/v5/v6/v7 均保留为历史，不覆写）

本版在 v7 基础上只修正 WOD 取回后的覆盖披露与执行记录：`wod_rq010b_full479_audited` 已由 PI 于
2026-07-31 批准并成功取回；另两个 WOD 产物与 `rq014_g2r_anchor_scores` 已由 PI 裁定不取回。
其余已通过复审的骨架继承 v7。RQ015A 的对象仍是 `attempt_status` 与连续
`q_eff = K_eff / K` 的集中度画像；它不衡量 IPV 精度，不区分近均匀权重的成因（那是 RQ015B），
也不构成 RQ007 的完整合取条件。

v6 已吸收最终三路复审、复审后机器闸门修复、七轮独立健壮性审计、信任边界声明、schema v2 待决项的 PI 冻结判定、
实现收敛状态与已知问题清单指针。RQ015A 的对象仍是 `attempt_status` 与连续
`q_eff = K_eff / K` 的集中度画像；它不衡量 IPV 精度，不区分近均匀权重的成因（那是 RQ015B），
也不构成 RQ007 的完整合取条件。

铁律不变：不改估计器；不覆盖冻结产物；不改任何 `decision.md`；不 replay；不真正运行本审计；
不读取、转移、登记任何人工评价类字段；不翻转 `execution_authorized`。

## 0. 绑定来源与 v8 的写作边界

本正文以以下实文件和裁定为依据：v7 计划正文、ledger schema v4、run spec v6、封存暴露登记、
预执行合同核验、2026-07-31 WOD 取回记录、PI 对 phase1b / schemeB / RQ014 不取回的裁定，
以及 `scripts/rq015a/` 下当前实现。本文只写计划正文，不代写已知问题清单、checksum manifest、
PI 授权对象或任何执行回执。

v8 是取回完成后的覆盖定稿正文，不是执行授权。run spec v6 的 `bound_artifacts.plan` 仍指向 v7；
本文件不更新绑定、不重签清单、不改变授权状态。后续若进入送审包，包身份仍须由另行重签的 checksum
manifest 承载，validate-only 必须逐行校验其成员集合与 SHA-256。正文若出现测试数，只能标为
“截至清单签署时的实测值”；机器权威始终是 validate-only 回执里的
`metadata.contract_fixtures.total_passed` / `fixture_total_passed`，不是本正文。

设计决定：一个 commit 无法在自己内部记录自己的 SHA；若正文硬写 HEAD，提交完成时该值天然落后。
因此 package identity 由 checksum manifest 承载，commit 号仅记录在 `main_workflow.log` 与提交信息里。
任何后续运行必须另有 checksum-bound package、
validate-only 通过、独立复审无 blocker，以及 PI 对精确 package/run-spec 的 scoped single-use
authorization。

## 1. v3/v4 已接受骨架：保留但收紧

- 主量继续是连续 `q_eff`，报告用 policy bins 仅作描述性展示，不进入 episode 摘要、C0 路由、
  `machine_verdict` 或任何判定。
- episode 摘要继续用连续量；C0 路由继续用 `unknown_share`、`unavailable_share`、
  `mean_q_eff_attempted` 与稳定性结论。
- 三恒等式、OnSite 局部序号、`sorted + math.fsum`、L3 `ZERO_SUPPORT` 不填 0、禁止跨产物 pooling
  均保留为必须实现的机器约束。
- `ATTEMPTED`、`NOT_ATTEMPTED`、`UNKNOWN`、q availability、recoverability 与 owning-RQ action
  是不同状态轴，不得互相替代。

## 2. C1-C14 预执行核验修正

| 项 | v6 合同 |
|---|---|
| C1 | sigma01 的 `2,490,992` 是已排除 D0 后的数，不能作 identity_1 基数。冻结为 physical `2,598,536`、measurement `5,197,072`、`NOT_ATTEMPTED` `215,088`。 |
| C2 | split join key 明确为 `case_id == scene_unique_id == case_key`，运行时必须断言未映射为 0。 |
| C3 | `rq009_m3_predictions` 的 15 列无任何 `ipv_error`，从 ledger 移除，仅作 provenance。 |
| C4 | 3× alpha 折叠只属于 predictions；feature matrix 为 `expansion_factor=2`、`collapse_factor=1`。 |
| C5 | v1 的 `anchors_dev_guard=1,778,594` 与 `cases_dev_guard=10,580` 无法复现，删除；feature matrix dev+guard 为 `4,497,368` 行。 |
| C6 | RQ009 fold `{train, guard_tune, calibration, test}` 与 RQ007 split `{development, guard, held_out}` 正交；每个 fold 含约 29% held_out。逐 fold held_out 行：train `751,826`、guard_tune `389,818`、calibration `376,580`、test `381,674`，合计 `1,899,898`。必须先按 `case_id` 白名单过滤，再读 measurement。 |
| C7 | OnSite 主键列是 `case_key`，不是 `case_id`；局部序号必须在 case 内按 `(timestamp_ms, frame_index)` 稳定排序后生成。 |
| C8 | OnSite 的 K 由 provenance 与 vendored estimator 确定为 `7`，grid 为 `legacy7_pi_over_8`。 |
| C9 | `rq014_g2r_anchor_scores` 本地无数据，v1 三字段 key 错；真实逻辑 key 来自 schema 的多字段顺序。 |
| C10 | run 环境不得再写“纯标准库”：parquet 产物要求 pandas/numpy 与 pyarrow 或 fastparquet。 |
| C11 | warm-up 占位 `error=1.0` 与均匀回退不同；`NOT_ATTEMPTED` 优先于 `UNKNOWN`。 |
| C12 | K 不是全局常量；代码库同时存在 K=7 与 K=5，未经 provenance 确认不得外推。 |
| C13 | OnSite 空串不能读作 0；空串且 `local_position < 4` 为 D0，空串且之后为 `UNKNOWN`。 |
| C14 | 可审计范围为 4/6：`interhub_sigma01_hw4_timeseries`、`rq009_feature_matrix`、`onsite_dense_timeseries`、`wod_rq010b_full479_audited`（906 行，已取回）。不可审计范围为 2/6：`wod_phase1_phase1b_10hz_schemeb`、`rq014_g2r_anchor_scores`；二者不是“尚未取回”，而是 PI 已裁定不取回，因为确认无 error 列，取回也无法产出本 RQ 主量。 |

## 3. RQ009 feature matrix：排除 M4_ONLY 通道

feature matrix 的 `M4_ONLY_ego_self_anchor_ipv_current` 与
`M4_ONLY_ego_self_anchor_ipv_error_current` 在本 RQ 中排除。`M4_ONLY` 前缀即禁用标记；
纳入会把 M4 自锚通道混入 M3 语境。`expansion_factor` 固定为 2，只包含
`counterpart_current` 与 `target_future` 两条通道。若未来需要分析 M4 自锚，必须另建独立
`artifact_id`，绝不与这两条通道合并聚合。

## 4. §9 修订：HPC 只读取回边界

v3 §9 的“无 HPC”改为：允许对三个 WOD/RQ014 产物做只读取回。取回是独立操作，必须有自己的
授权对象、SHA 清单、no-overwrite 目标目录与 sanitization receipt。2026-07-31，PI 已批准并完成
`wod_rq010b_full479_audited` 的只读取回；phase1b / schemeB 与 `rq014_g2r_anchor_scores` 则由 PI
裁定不取回。探测不等于取回；本 RQ 只取回 full479 投影后的 4 列，未取回其它 WOD/RQ014 数据。

强制顺序：HPC 侧先做列投影与列名 denylist 检查，确认无受禁列后，才允许传输。
本地永不出现受禁列。任何投影 receipt 缺失、SHA 不一致、目标目录已存在、列名越界或写入既有 HPC
产物的迹象，均立即 abort。

## 5. HPC 只读探测结果与 PI 裁定

2026-07-27 的只读探测确认六个 WOD 目标在 HPC 上全部存在；2026-07-31 的 PI 裁定与执行结论如下。

| 目标 | 行数 | error 列 | 取回边界 |
|---|---:|---|---|
| RQ010B full479 audited candidate CSV | 906 | 有：`ego_ipv_error` | 已按 4 列白名单投影、取回并复核 |
| phase1 candidate CSV | 1,428 | 无 | 不纳入本 RQ 主量 |
| phase1 audit CSV | 142 | 无 | 不纳入本 RQ 主量 |
| phase1b subwindow candidate CSV | 1,428 | 无 | PI 裁定不取回 |
| phase1b subwindow audit CSV | 1,428 | 无 | PI 裁定不取回 |
| schemeB effective-N candidate CSV | 476 | 无 | PI 裁定不取回 |

结论：

- `wod_rq010b_full479_audited` 不是 L4。其 906 行已按授权取回，schema v4 已升为
  ledger-bearing / `L1_DIRECT`，`not_attempted_rule = none_expected`。
- `wod_phase1_phase1b_10hz_schemeb` 的 `ipv_error_source: absent` 定性确认无误；phase1b 与
  schemeB 确无 error 列，取回也无法产出本 RQ 主量，故 PI 裁定不取回。
- `rq014_g2r_anchor_scores` 的行 schema 确认无任何 error 字段，仅有候选值字段，故 PI 同样裁定不取回。

## 6. 取回的执行记录与边界（2026-07-31）

PI 于 2026-07-31 批准并成功执行 WOD full479 取回。落地目录为
`data/derived/wod_e2e/rq015a_full479_projected/`，CSV 文件为
`rq010b_wod_full479_audited_candidate_ipv_projected.csv`，SHA-256 为
`d10c3a6f903dae19bf14de6b895c72f6c4ffc92688d97baa9a7c26684930b7d1`。该 CSV 有 906 个数据行，
表头恰为 4 列：`segment_key`、`candidate_index`、`ego_ipv`、`ego_ipv_error`。

执行链路为：HPC 侧按列名白名单投影，生成 sanitization receipt，完成传输前五项校验，传输后再做
本地四项复核，全部通过。`sanitization_receipt.json` 的 SHA-256 为
`f2435ed9190cac0f4b64e20a4f10edc1412d8bd1f965124f63684abf672c0fdf`，使本次取回可被独立复核。

边界必须随报告披露：源文件 65 列中有 61 列在 HPC 侧即被丢弃，本地从未出现；本地与 HPC 侧禁词扫描
均命中 0。driven 两列是对照基线，有意未取，不能被当作第二测量通道；`expansion_factor` 保持 1。
K = 7、`grid_id = legacy7_pi_over_8` 由三环证据链确定，因此 `q_eff` 可算，该产物是完整
`L1_DIRECT`，不是半截。schema v4 对该 artifact 的实测规则为
`not_attempted_rule = none_expected`，906 行 `ego_ipv_error` 无空串、占位或非有限值。

## 7. 独立盲算复核（2026-07-27）

独立 agent 对原实现与冻结数字保持盲态，使用 `pyarrow.csv` 列投影流式读与 `pyarrow.dataset`
重算三个本地产物的行数口径，未用 pandas。结果与冻结事实逐位吻合，零分歧。

两项新增证据继续保留：

- feature matrix 的 `anchor_frame_index` 最小值为 `7`，已在全部 138 parts / `6,397,266` 行上验证；
  schema v2 中“抽样实测”升级为全量结论。
- OnSite 的 `local_position < 4` 行数为 `1,068`（`267 cases × 4`）。错误使用全局
  `frame_index < 4` 只有 `53` 行，两者相差 `1,015` 行。该结果同时冻结 OnSite identity_2 的
  `NOT_ATTEMPTED` 基数，并量化支持“禁止全局 frame_index 规则”。

## 8. schema v2 内部歧义的冻结判定与 schema v4 现状

`rq014_g2r_anchor_scores` 与另两个 `ARTIFACT_NOT_PRESENT_LOCALLY` 条目不一致：缺
`rq007_split_applicable`、`rq007_split_value`、`expansion_factor`、`collapse_factor`；
recoverability 又写为 `L4_UNRECOVERABLE`。identity_3 定义为
`measurement_rows = Σ_over_recoverability n(recoverability)`，同一批行同时有两个竞争的
recoverability 值会造成求和歧义。

冻结判定（PI 已裁定）：`ARTIFACT_NOT_PRESENT_LOCALLY` 优先于 `L4_UNRECOVERABLE`；
identity_3 按前者求和。当前实现已强制取前者：schema-derived absent-local 条目在构造
`AbsentArtifactCoverage` 与 absent coverage 时统一落到 `ARTIFACT_NOT_PRESENT_LOCALLY`。
schema v4 已吸收 WOD full479 取回后的状态：`wod_rq010b_full479_audited` 为 ledger-bearing /
`L1_DIRECT`，且不再是 absent-local 或待取回条目。本计划不改 schema 文件，只记录该事实。

## 9. 暴露裁定 §8（2026-07-27）

封存暴露登记新增 §8，作为 append-only supersede，§6 原文未改。§8 正式解除 §6 附加条件的三条：
两阈值须从 dev+guard 重导出、导出规则须先冻结登记 SHA、重导出前不得产出结论画像；同时撤销
`PROVISIONAL_PENDING_DEVGUARD_REDERIVATION` 标记。

解除理由：`4/7`（对应 `ipv_error=0.5`）与 `0.93`（对应 `ipv_error=0.608069099165`）已由科学阈值
降为报告用 policy bins；R6 加上 `test_c0_routing_never_consumes_report_bins` 已在代码层强制 bins
不进入任何判定，该条件已丧失保护对象。

不在解除范围内：§6 判读 A 与记录豁免；§7 措辞精确化，即扫描程序确实解析并聚合过 held_out 逐行字段，
不得回退为“未读取任何 held-out 逐行测量值”；§7 三条治理动作；R1/R2/R3；以及
`execution_authorized=false`。本解除不构成任何执行授权，不替 PI 签署。签署状态为
`RECORDED_ON_PI_RULING`（已生效，无待签事项）。

## 10. 实现现状（仅记录，不授权运行）

当前分支仍为 `rq015a-implementation`。为避免 provenance 反复漂移，本正文不记录当前 HEAD；
run spec v6 仍绑定 v7，v8 的绑定更新与清单重签由后续指挥者处理。本文不改变任何执行授权。

五个 RQ015A 测试文件的实测通过数不在正文冻结；进场基线、复审后新增 fixture、清单签署时实测值可以不同。
若签署材料需要显示测试数，必须标为“截至该清单签署时的实测值”。权威来源是 validate-only 回执里的
`metadata.contract_fixtures.total_passed`（CLI 摘要中同义显示为 `fixture_total_passed`），不得以正文数值覆盖回执。
验证命令形状：

```bash
<PYTHON> -m pytest tests/test_rq015a_contracts.py tests/test_rq015a_build_ledger.py tests/test_rq015a_validate_receipt.py tests/test_rq015a_factor_analysis.py tests/test_rq015a_run_entrypoint.py -q
```

全量 pytest 仍有约 21 条既有失败（RQ014 / launcher / shortcut）。已用 `19b28024` 的干净检出比对：
同批文件在基线为 23 failed，故这些失败与本实现无关。测试通过不构成执行授权。

当前实现的关键受信对象：

- `scripts/rq015a/run_rq015a.py`：唯一受信入口；`--execute` 在 permit 前拒绝。
- `scripts/rq015a/build_ledger.py`：schema-derived scope、allowlist 源回核、真实路径两阶段读取、
  `open_measurement_reader()` 身份登记、L1/L2/L3 守恒与聚合。
- `scripts/rq015a/validate_only.py` 与 `receipt.py`：validate-only 结构列检查、no-overwrite receipt、
  机器 verdict 与运行环境记录。
- `scripts/rq015a/rq015a_contracts.py` 与 `factor_analysis.py`：连续量合同、排序 + `math.fsum`、
  C0 路由与描述性因素关联。

## 11. 七轮独立健壮性审计与收敛证据

七轮均由独立 agent 执行，且对此前所有审计与修复轮保持盲态：禁读前序报告、
`main_workflow.log`、`git log` 与 `git diff`。累计闭合 40 条缺陷，其中 9 条 blocker。

| 轮次 | 结果 | 代表性 blocker / 审计价值 |
|---|---|---|
| 审计 1 | 3 blocker + 6 其它 | `q_eff(1.1, 7)` 曾静默返回 1.0，非法输入被包装成合法集中度上界。 |
| 审计 2 | 2 blocker + 5；2 条 blocker 由修复轮 1 引入 | 真实文件路径曾 `pd.read_parquet(...).to_dict("records")` 先读入全部测量列再过滤；R1 在真实路径上不成立，fixtures 因全走注入路径未覆盖。 |
| 审计 3 | 1 blocker + 4 | 授权双键只实现一半：未读授权条目自身的 `execution_authorized`，且拍平所有条目的 `allowed_operations`。 |
| 审计 4 | 1 blocker + 5 | 两阶段读取中 null join key 在 Python 侧变成字符串 `"None"`，而 pyarrow 按 null 语义过滤，静默产出 0 行。 |
| 审计 5 | 1 blocker + 3 | `local_positions()` 作为 OnSite D0 判据，五轮未被攻击，曾接受空输入、负数、bool、float。 |
| 审计 6 | 1 blocker + 4；覆盖面由 12 项“深入”扩至 30 项 | 结构列 denylist 曾未做 Unicode 规范化，列名伪装风险未进入 fail-closed 面。 |
| 审计 7 | 0 blocker + 6；收敛 | 信任边界复核确认边界内路径已 fail-closed，剩余问题转入已知问题清单。 |

反面证据同样写入复审正文：审计 7 确认 `--execute` 在 permit 前拒绝；拒绝回执记录
`measurement_reader_constructed=false`；合成双键全 true 时仍被 BUILD_WHILE_DENY 后置硬拒；
duck reader 被拒；allowlist 源替换在 build 前 recheck 失败。

## 12. 最终复审与复审后修复（2026-07-30）

最终三路独立复审裁决为：技术 `BLOCKED`，显著性 `PASS_WITH_CONDITIONS`，可执行性 `BLOCKED`。
两路 `BLOCKED` 各自独立指出同样三条 blocker：run spec 绑定核对未由入口实际执行；
execute 未强制消费 validate receipt 并核对 input digest；checksum manifest 的逐行 SHA-256 与行集一致性
未进入机器 verdict。

这些 blocker 的性质是：规格声称的闸门代码未执行。复审方“手工复现通过不能替代机器合同自身成立”的批评，
已被指挥方全盘接受。

代码侧已完成修复并由指挥者实测确认：

- 伪造 run spec 指过去：exit 1，报 `run_spec_path mismatch`。
- 合法 v3 但缺 validate receipt：exit 1，报 `validate receipt is required`。
- 校验清单中文件漂移：validate-only `machine_verdict=FAIL`。
- v7 清单重签后的测试总数必须由 validate-only 回执记录；正文不作为机器权威。

## 13. 信任边界声明

`scripts/rq015a/run_rq015a.py` 是唯一受信入口。直接调用内部函数，或用 `object.__new__` / pickle /
元类伪造对象，不在信任模型内。

理由：Python 无法阻止对象伪造；把它当作威胁面会使每轮审计都产出一条“还能这样伪造”，且无收敛终点。
本计划不把 hostile same-process object fabrication 当作 RQ015A 的闭合要求。

边界内仍必须成立的两件事：

1. 从公开 CLI（`--execute` / `--validate-only`）出发的任何可达路径都不得绕过 permit 校验、
   allowlist 回源核对、或结构列 denylist。
2. 外部对象不得替换掉一个已经过校验的对象；注入防护已由 `open_measurement_reader()` 的 WeakSet
   身份登记实现。

## 14. 已知问题清单指针

随包提交的已知问题清单由指挥方另行成文，复审方必须与本 v8 同读。本文点名其中两条实质项：

- 同形字符列名仍可绕过 denylist：受禁列名的同形字符伪装样例曾通过。
  这是已知剩余风险，不得在 v8 中改写成已闭合。复审方要求的限定是：同形字符问题披露充分，
  理由在 schema 受控前提下可接受，但不得再声称“全 Unicode 防护闭合”。
- D0 的 `NOT_ATTEMPTED` 行保留非空 `q_eff` / `k_eff`。下游按 `attempt_status` 过滤，未见结论污染；
  但复审必须确认该残留是否仅为表示层问题，还是需在执行前改为强制 null。

## 15. 可审计范围与覆盖披露

可审计产物为 4 个：`interhub_sigma01_hw4_timeseries`、`rq009_feature_matrix`、
`onsite_dense_timeseries`、`wod_rq010b_full479_audited`（906 行，已取回）。

不可审计产物为 2 个：`wod_phase1_phase1b_10hz_schemeb`、`rq014_g2r_anchor_scores`。这不是
“尚未取回”，而是 PI 已裁定不取回；理由是二者确无 error 列，取回也无法产出本 RQ 主量。

覆盖缺口披露从条件句改为确定要求：报告必须在标题级披露 WOD 一支为部分覆盖。它不再是零覆盖，
因为 full479 已有 906 行可审计；也不是全覆盖，因为 phase1b / schemeB 未纳入。报告明确禁止表述为
“全语料”或“全 WOD”。

建议标题级披露措辞：

> WOD 分支为部分覆盖：本报告纳入已取回且可审计的 full479 投影数据（906 行），不包含经 PI 裁定不取回的 phase1b / schemeB 产物。

覆盖披露要求：

- 报告必须逐产物列出 present / PI-ruled-out / not-in-scope 的状态。
- 所有 WOD/RQ014 缺口必须进入 coverage limitations，不得静默从分母消失。
- `M4_ONLY` 排除、policy bins 历史来源、held_out 解析边界、D0 规则逐产物差异，均须随报告披露。

## 16. 复审后使用边界

v8 可作为覆盖定稿正文，但仍不授权执行。任何后续 package/run-spec 必须至少保留以下判定面：

1. §11 的 40 条缺陷闭合叙述是否与当前实现、测试和已知问题清单一致；
2. §13 的信任边界是否足够明确，是否避免把 Python 对象伪造误列为无限 blocker；
3. §14 两条已知问题是否允许带入最终 package，或必须在执行授权前修复；
4. WOD 部分覆盖、v8 checksum package 与 PI 单次授权是否被清楚隔离。

只有当复审后机器合同无 blocker、已知问题处置被 PI 明确接受或修复、并且精确 package/run-spec 另行授权后，
才可能进入实际执行。
