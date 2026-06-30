# `reports/knowledge/` 治理方案

生成日期：2026-06-30  
范围：仅 `reports/knowledge/`  目标结构档位：**中等（推荐）**  
状态：**方案稿，待审阅后执行**（本文档不移动任何文件）

---

## 1. 一句话结论

`knowledge/` 的内容是健康的，乱的是**结构纪律**：模板和现实脱节、评审文件命名出现三套写法、子问题（B 系）和交接文件无处安放、根目录堆了 7 个无归属的工程/分析文件、还有 12 个 `.DS_Store`。问题都属于"约定缺失"，可以低风险地分阶段收敛，不影响任何已冻结的结论。

---

## 2. 现状盘点

当前共 12 个 RQ 文件夹（RQ001–RQ012）、2 个 PAPER 文件夹、1 个 `_template`，外加根目录散落文件。标准件由 `_template` 与 `README.md` 定义为：`README.md` / `report_index.md` / `synthesis.md` / `decision.md` / `plan.md`，以及 `reviews/` 下的角色评审。`RQ_PROGRESS_DASHBOARD.md` + `rq_progress_registry.csv` 是程序级登记，机制本身设计良好（有 source-of-truth 优先级），问题出在文件层没有对应的命名与归位规范。

---

## 3. 问题诊断

按严重度与处理优先级排列：

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| P1 | **模板与现实脱节** | `_template` 含 `plan.md`，但 12 个 RQ **无一**有 `plan.md`（plan 实际在 `reports/plans/`）；模板 `reviews/` 只发 `codex_response.md`，而现实 10 个 RQ 用 `codex_review.md`、仅 3 个用 `codex_response.md` | 新建 RQ 一开始就和模板不一致，越铺越歪 |
| P2 | **标准件缺失** | `synthesis.md` 在 RQ001–008 都有，RQ009/010/011/012（最新的 4 个 accepted）**全缺**；PAPER002 只有 `reviews/` 一个文件，无 README/decision/synthesis 却已是 `writing` 状态 | "已接受"的 RQ 缺少综合解释层，paper 取证时无单一入口 |
| P3 | **评审命名三套并存** | ①角色名：`claude_review.md`；②带日期：`chatgpt_review_20260629.md`、`chatgpt_review_rq010b_20260629.md`；③子问题后缀＋大小写不一：`claude_review_RQ011B.md` 与同目录 `claude_review.md` 并存 | 无法用统一规则定位"某 RQ 的某角色评审"，脚本化困难 |
| P4 | **`codex_review` 与 `codex_response` 语义混用** | 二者实为不同角色（review=Codex 做评审；response=Codex 回应评审），但模板只发 response、现实多用 review，规范从未写明 | 同名不同义，易误删或误并 |
| P5 | **子问题（B 系）安放方式临时** | `RQ011B`（registry：`closed-out`）、`RQ012B` 仅作为登记 ID 或文件名后缀，**无独立文件夹**，产物塞进父 RQ；`RQ013`（`planning`）连文件夹都没有 | 子问题证据散落，难追溯 |
| P6 | **交接文件无家** | `RQ012/RQ012B_paper_handoff.md`、`RQ012/RQ012B_to_RQ013_handoff.md` 直接躺在 RQ 根，不在模板内 | 交接物与标准件混放 |
| P7 | **7 个游离根文件（解释层混入机器产物）** | `INFRA_hpc_tongji_reuse.md`、`ipv_accel_hyperparam.{json,md}`、`ipv_estimator_api_map.{json,md}`、`ipv_estimator_divergence.{json,md}`；其中 `INFRA_*` 与 `ipv_estimator_api_map` 被 0 个文件引用 | README 称 knowledge 为"解释层"，却混入裸 `.json`；半数已成孤儿 |
| P8 | **1:1 不变式已被打破且无文档** | README 称"每个 RQ 一个 knowledge 文件夹，且与 `reports/studies/` 同名对应"，但 studies 只有 RQ001–012，没有 PAPER、没有 B 系；knowledge 反而多出 PAPER002、B 系登记 | 规则与现实矛盾，新人无所适从 |
| P9 | **12 个 `.DS_Store`（含 `_template` 内）** | `find . -name .DS_Store` = 12；`.gitignore` 第 18 行已列出但磁盘仍残留，且模板带着它，一复制就扩散 | 目录列表噪声，污染模板 |

---

## 4. 治理原则

1. **模板即现实**：`_template` 必须等于"新建一个 RQ 应有的样子"。模板错了就改模板，而不是让每个 RQ 去迁就一个没人用的文件。
2. **解释层保持纯净**：`knowledge/` 只放可读的解释/综合/决策；裸机器产物（`.json` 配置、缓存）原则上属于 `reports/studies/`，中等档位先就近收纳、标注来源。
3. **一个角色一个文件，轮次写进正文**：评审默认每角色一个文件，多轮在文件内追加小节，不靠文件名堆日期。
4. **登记驱动结构**：凡是 registry 里独立登记的 ID（含 B 系）才配独立文件夹；未登记的子问题留在父 RQ 内、但命名规范化。
5. **状态靠登记，不靠移动文件夹**：生命周期（accepted / archived-review 等）由 registry + `decision.md` 表达，文件夹不因状态而搬家。
6. **不变式要么遵守、要么写明例外**：把 PAPER 与 B 系的例外明确写进 README，消除"已违反但没文档"的灰色地带。

---

## 5. 目标结构（中等档位）

```text
reports/knowledge/
├── README.md                      # 更新：标准件清单、1:1 例外、指向 CONVENTIONS
├── CONVENTIONS.md                 # 【新增】命名与生命周期规范（治理核心）
├── RQ_PROGRESS_DASHBOARD.md       # 保留
├── rq_progress_registry.csv       # 保留
├── _template/                     # 修正为与现实一致（见 §6）
│   ├── README.md
│   ├── report_index.md
│   ├── synthesis.md
│   ├── decision.md
│   └── reviews/
│       ├── chatgpt_review.md
│       ├── claude_review.md
│       ├── codex_review.md
│       └── codex_response.md      # 可选，仅在确有回应时建
├── _analysis/                     # 【新增】跨 RQ 的工程/分析笔记归位
│   ├── INFRA_hpc_tongji_reuse.md
│   ├── ipv_accel_hyperparam_finding.md        (+ ipv_accel_hyperparam.json)
│   ├── ipv_estimator_api_map.md               (+ ipv_estimator_api_map.json)
│   └── ipv_estimator_divergence_investigation.md  (+ ipv_estimator_divergence.json)
├── PAPER001_online_sociality_verification_manuscript/   # 保留
├── PAPER002_dynamic_ipv_evidence_architecture/          # 补齐标准件骨架
│   ├── README.md  decision.md  synthesis.md
│   └── reviews/…（现有 wave_b 评审并入并改名）
├── RQ001 … RQ012/                 # RQ009–012 补 synthesis.md；reviews 命名归一
│   └── （RQ012 增设 handoffs/ 收纳交接文件）
├── RQ011B_onsite_moment_monitor/  # 【新增】已登记 B 系 → 独立文件夹
└── RQ013_beyond_safety_increment/ # 可选占位（仅 README，离开 planning 时补全）
```

要点：`RQ011B` 在 registry 已独立登记（`closed-out`），故升格为独立文件夹；`RQ012B` **未**单独登记，保留在 `RQ012/` 内，仅做命名规范化。

---

## 6. 命名与生命周期规范（拟写入 `CONVENTIONS.md`）

**标准件（每个已登记 RQ/PAPER 文件夹）**

| 文件 | 含义 | 何时必需 |
|---|---|---|
| `README.md` | 问题、范围、现状 | 建夹即有 |
| `report_index.md` | 该 RQ 全部执行报告索引 | 建夹即有 |
| `synthesis.md` | 跨版本/跨报告的综合解释 | 进入 `review` 及以后必需 |
| `decision.md` | 接受/拒绝/搁置的 claim | 建夹即有 |

`plan.md` **不进 knowledge**：计划统一在 `reports/plans/`，knowledge 内只在 README 里给链接。（据此从 `_template` 删除 `plan.md`。）

**评审文件（`reviews/` 内，全部小写）**

| 文件 | 角色语义 |
|---|---|
| `chatgpt_review.md` | ChatGPT 对结果/实现的评审 |
| `claude_review.md` | Claude 评审 |
| `codex_review.md` | Codex 评审 |
| `codex_response.md` | Codex 对上述评审的**回应**（仅在有回应时建） |

命名规则：

- **一角色一文件**；第二轮评审在文件内追加 `## Round 2 — 2026-06-29` 小节，**日期写进正文，不写进文件名**。
- 仅当子问题需要**并行独立留档**时，才用后缀 `_rqNNNx`（小写、与登记 ID 一致），例如 `claude_review_rq012b.md`；后缀里**不再附日期**。
- 据此规范化现有文件：`chatgpt_review_20260629.md` → 并入 `chatgpt_review.md`；`chatgpt_review_rq011b_20260629.md` → 随 RQ011B 迁入新文件夹后回归 `chatgpt_review.md`；`claude_review_RQ011B.md` → 同上；`RQ012B` 相关 → `*_rq012b.md`（留在 RQ012）。

**交接文件**：放进该 RQ 下的 `handoffs/` 子目录，命名 `to_rqNNN_handoff.md` 或 `paper_handoff.md`。即 RQ012 两个交接文件 → `RQ012_*/handoffs/`。

**`_analysis/`**：跨 RQ、不归属单一问题的工程/分析笔记集中此处，命名 `<topic>.md`（可带同名 `.json`）。每个文件顶部加 front-matter 注明关联 RQ 与是否孤儿。（严格档位下裸 `.json` 应迁 `reports/studies/`；中等档位先就近收纳。）

**生命周期**：不建 archive 目录；状态以 registry 的 `program_status` + `decision.md` 为准，文件夹不随状态搬家（如 RQ006 `archived-review` 仍留原位）。

---

## 7. 分阶段迁移步骤

每阶段独立可回滚；所有移动用 `git mv` 保留历史，移动期间**不改文件内容**。

**阶段 0 — 低风险清扫（先做，零结构影响）**
- 删除 12 个 `.DS_Store`（含 `_template/.DS_Store`），确认未被 git 跟踪。
- 修正 `_template`：删 `plan.md`，`reviews/` 增 `codex_review.md`、把 `codex_response.md` 标注为可选。
- 新建 `CONVENTIONS.md`，落地 §6。

**阶段 1 — 补齐标准件**
- 为 RQ009/010/011/012 各补 `synthesis.md`（先建骨架，内容从现有 review/decision 提炼）。
- 为 PAPER002 补 `README.md` / `decision.md` / `synthesis.md` 骨架，使其与 `writing` 状态匹配。

**阶段 2 — 评审命名归一**（详见 §8 映射表）
- 去日期、统一大小写、角色归位；多轮并入正文小节。

**阶段 3 — 结构化**
- 建 `_analysis/`，迁入 7 个游离根文件（`.md` 与同名 `.json` 一起），各加 front-matter；对 0 引用的 `INFRA_*`、`ipv_estimator_api_map` 标注孤儿待确认是否归档。
- 建 `RQ011B_onsite_moment_monitor/` 独立文件夹，迁入其 `*_rq011b_*` 评审并改回标准名。
- 建 `RQ012_*/handoffs/`，迁入两个交接文件；`RQ012B` 评审改名 `*_rq012b.md` 留在 RQ012。
- （可选）建 `RQ013_*/README.md` 占位。

**阶段 4 — 登记同步与守护**
- 更新 `README.md`：标准件清单去掉 `plan.md`、写明 PAPER 与 B 系对 1:1 不变式的例外。
- 加一个提交前检查清单（见 §9），可选写成校验脚本。

---

## 8. 关键改名/迁移映射（最易错处）

| 现状路径 | 目标路径 | 动作 |
|---|---|---|
| `RQ009_*/reviews/chatgpt_review_20260629.md` | `RQ009_*/reviews/chatgpt_review.md` | 改名（日期入正文） |
| `RQ010_*/reviews/chatgpt_review_rq010b_20260629.md` | `RQ010_*/reviews/chatgpt_review_rq010b.md` | 去日期 |
| `RQ011_*/reviews/chatgpt_review_rq011b_20260629.md` | `RQ011B_onsite_moment_monitor/reviews/chatgpt_review.md` | 迁入新夹＋改名 |
| `RQ011_*/reviews/claude_review_RQ011B.md` | `RQ011B_onsite_moment_monitor/reviews/claude_review.md` | 迁入新夹＋小写 |
| `RQ011_*/reviews/claude_review.md` / `codex_review.md` | 留在 `RQ011_*`（属母问题） | 不动 |
| `RQ012_*/reviews/claude_review_RQ012B.md` | `RQ012_*/reviews/claude_review_rq012b.md` | 仅小写归一（不单建夹） |
| `RQ012_*/reviews/chatgpt_review_rq012b_20260629.md` | `RQ012_*/reviews/chatgpt_review_rq012b.md` | 去日期 |
| `RQ012_*/RQ012B_paper_handoff.md` | `RQ012_*/handoffs/paper_handoff.md` | 迁入 handoffs/ |
| `RQ012_*/RQ012B_to_RQ013_handoff.md` | `RQ012_*/handoffs/to_rq013_handoff.md` | 迁入 handoffs/ |
| `PAPER002_*/reviews/chatgpt_review_wave_b_rq009_rq010b_rq011b_rq012b_20260629.md` | `PAPER002_*/reviews/chatgpt_review_wave_b.md` | 去日期＋缩短 |
| 根 `INFRA_hpc_tongji_reuse.md` | `_analysis/INFRA_hpc_tongji_reuse.md` | 迁入（标注孤儿） |
| 根 `ipv_accel_hyperparam.json` + `ipv_accel_hyperparam_finding.md` | `_analysis/` 同名 | 成对迁入 |
| 根 `ipv_estimator_api_map.json` + `ipv_estimator_api_map.md` | `_analysis/` 同名 | 成对迁入（标注孤儿） |
| 根 `ipv_estimator_divergence.json` + `ipv_estimator_divergence_investigation.md` | `_analysis/` 同名 | 成对迁入 |

> 迁移前需全仓 `grep` 一遍被引路径并同步更新（已知 `ipv_accel_hyperparam`、`ipv_estimator_divergence` 各有 1 处内部引用）。

---

## 9. 长效维护机制

- **`CONVENTIONS.md` 为单一规范源**，README 顶部链接它；新建 RQ 一律 `cp -r _template`。
- **提交前检查清单**（可脚本化）：①每个已登记 RQ/PAPER 至少有 `README/report_index/synthesis/decision` 四件；②`reviews/` 文件名匹配 `^(chatgpt|claude|codex)_(review|response)(_rq\d+[a-z])?\.md$`；③根目录无游离 `.md/.json`（除看板、登记、README、CONVENTIONS）；④registry 的 ID 集合与 knowledge 文件夹集合一致（B 系/ PAPER 例外列在 README）。
- **`.gitignore`** 已含 `.DS_Store`，保持即可；阶段 0 清掉残留。
- **registry 同步**：任何新增/升格文件夹（如 RQ011B、RQ013）必须同时在 `rq_progress_registry.csv` 与看板登记，反之亦然。

---

## 10. 风险与回滚

改动均为"移动 + 改名 + 补骨架"，**不触碰已冻结的 `decision.md` 结论**，因此对 paper 取证链无实质风险。所有操作经 `git mv`，可整阶段 `git revert`。唯一需要人工确认的判断点有三处：(a) 两个 0 引用的孤儿文件（`INFRA_*`、`ipv_estimator_api_map`）是迁入 `_analysis/` 还是直接归档；(b) 裸 `.json` 是否现在就升级到"严格档位"迁去 `reports/studies/`；(c) `RQ013` 是否现在建占位夹。建议这三处在执行阶段逐个点头即可。

---

## 附：建议执行顺序一览

阶段 0（清扫＋修模板＋建 CONVENTIONS）→ 阶段 1（补 synthesis / PAPER002 骨架）→ 阶段 2（评审改名）→ 阶段 3（_analysis / RQ011B / handoffs）→ 阶段 4（README 与检查清单）。每阶段一个提交，便于回滚与审阅。
