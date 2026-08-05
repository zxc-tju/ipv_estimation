# Repository Guidelines

## Quick Orientation & START_HERE Maintenance
Begin each new agent thread by reading `START_HERE.md` beside this file, then use `AGENTS.md` for durable rules and `PROJECT_STRUCTURE.md` for deeper architecture notes. Keep `START_HERE.md` as the short current operating brief: current active batch, canonical data paths, canonical output paths, test command, app/server start instructions, protected files/directories, latest stable report/result, and known weak spots.

Whenever a workflow changes any of those current-operating facts, update `START_HERE.md` in the same task before finishing. If the facts are uncertain, write the uncertainty explicitly instead of leaving stale guidance. Log the maintenance outcome in `main_workflow.log` together with the normal workflow summary.

## Project Structure & Module Organization
Reusable estimation logic lives under `src/sociality_estimation/`: `core/agent.py` and `core/ipv_estimation.py` hold the IPV model/estimator, while `planning/` holds lattice and geometry helpers. Dataset and experiment entrypoints live under `pipelines/`, with the active InterHub CSV/pkl pipeline at `pipelines/interhub/process_interhub.py` and simulation code at `pipelines/simulation/simulator.py`. Raw/local data lives under ignored `data/` subdirectories such as `data/interhub/raw/` and `data/onsite_competition/raw/`; track only README/manifest/index files from `data/` unless explicitly requested. Large derived data lives under ignored `data/derived/`. The research knowledge base has three governed layers under `reports/`: approved plans, locked SAPs, and PI decision records under `reports/plans/`; execution reports under `reports/studies/`; reviewer synthesis/decisions under `reports/knowledge/`. `STUDIES.md` is the root index. `reports/` must keep only `plans/`, `studies/`, and `knowledge/` as first-level directories. The old root-level compatibility files (`agent.py`, `ipv_estimation.py`, `process_interhub.py`, `simulator.py`, and `tools/`) were archived under `archived/compat_wrappers_20260619/` and are not active entrypoints. Legacy dataset scripts live under `archived/legacy_scripts/`; review and restore them only if you need old Argoverse CSV, old InterHub JSON, subset wrapper, or old `mean_ipv` metadata post-processing workflows. Report-linked process archives live under `archived/report_process/`; local agent state lives under `archived/report_local_state/`. Manuscript drafting is now in the separate paper repository at `../../2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`; do not recreate a top-level `paper/` directory here.

## Build, Test, and Development Commands
Create environments with `python -m venv venv && venv\Scripts\activate` on Windows or `conda create -n ipv python=3.9`. Install dependencies via `pip install -r requirements-minimal.txt` for clusters or `requirements.txt` locally. Invoke current InterHub CSV/pkl jobs through `python pipelines/interhub/process_interhub.py --csv <input.csv> --pkl-root <pkl_dir>`; default InterHub inputs are under `data/interhub/raw/`. Archived compatibility wrappers and legacy scripts are not active commands until restored and path-checked. Legacy HPC Slurm command files are retained in report-linked process archives or `archived/` rather than as root-level tracked scripts.

If a required Python dependency is missing during runtime or verification, install it directly in the active project environment and continue rather than stopping to ask. Prefer the current venv/conda environment and existing `requirements.txt`; when a missing package becomes a durable project requirement, update the appropriate requirements file or document the environment change in the workflow log.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` functions, and `CamelCase` classes. Modules favour explicit type hints and docstrings; mirror that when extending APIs. Prefer `logging` over `print` (see existing `LOGGER` usage) and keep plotting helpers under the dataset modules. Place reusable algorithm utilities inside `src/sociality_estimation/`; do not recreate root-level `tools/` wrappers.

## Testing Guidelines
There is no automated suite yet; validate changes by rerunning the relevant pipeline with a representative dataset and comparing the generated Excel/plot artifacts. When adding functionality, include lightweight `pytest`-style checks under a new `tests/` directory that exercise `estimate_ipv_pair` or trajectory transforms, and document any sample input you rely on. Capture key diagnostic figures (`--diagnostics`) for regression tracking.

## Commit & Pull Request Guidelines
Commits in this repo use short, imperative subjects (for example, `Enhance process_interhub.py...`). Group related edits and avoid bundling data drops. Pull requests should outline the scenario touched, CLI commands executed, and notable output locations; attach diffs or screenshots of plots when behaviour shifts. Cross-link issues or task IDs where applicable.

## Data & Configuration Notes
Large trajectory files stay outside version control; if you must share samples, provide download instructions instead. Keep environment-specific tweaks in separate config files or guarded by CLI flags, and sanitise paths before publishing job scripts.

## Research Velocity Principle (PI ruling 2026-07-31) — read this before the section below

**默认推进方式是高效、粗放、以结果产出为先。** 本项目此前出现过严重的过程膨胀：
一个描述性数据质量审计走了 8 个计划版本、7 轮独立盲审、3 路最终复审、32 个 agent，
最终科学结论产出为零。这不是标准，是需要避免的反面案例。

**过程强度必须与主张风险成比例：**

| 产出类型 | 该有的过程 |
|---|---|
| 探索性 / 诊断性 / 描述性结果（多数工作） | 跑出来 → 自查一遍数值健康与覆盖 → 出报告。**一轮即可** |
| 进入手稿的主张 | 才启用多轮独立复审与证据冻结 |
| 触及已冻结 `decision.md` 的改动 | 才需要完整治理闭环 |

**验证策略**：本研究所用数据是**用于科研发现**的数据；结论的可靠性由**独立数据集上的复现**
承担，不由对单次分析的层层加固承担。因此不要为一次描述性分析反复加固，
把精力投到"换一个数据集/换一个口径看结论还在不在"。

**明确的反模式**（出现即停下来问自己是否在浪费时间）：

- 计划连出 2 版以上却没有新的事实进来
- 为一次**尚未运行**的分析做多轮盲审
- 为描述性产物建多重授权闸门
- 把"程序对不对"的验证做到远超"结论对不对"的验证
- 用治理文书替代实际产出

**不属于"过程"、不得放松的少数几条**（它们是效度边界，不是流程）：

1. RQ007 held-out 集不得被解析——它是确认性路径的唯一保障，一旦污染无法恢复
2. RQ014 致盲相关的评分字段不得读取
3. 不得静默覆盖已冻结产物或已接受的 `decision.md`
4. 描述性结果不得写成因果主张

除这四条外，遇到"要不要再加一道保险"的犹豫，**默认选不加，先把结果跑出来**。

## Scientific Analysis & Reporting Guidelines
Start every research analysis by making the research question, unit of analysis, variable meanings, inclusion/exclusion criteria, and data provenance explicit. When semantics are unclear, inspect local dictionaries, source documentation, or generation scripts before interpreting a field. Distinguish descriptive patterns, predictive associations, and causal claims; do not use causal language unless the design supports it.

Treat publication-oriented analyses as claim-indexed research, not as a dashboard of interesting plots. Prioritize conclusions with broad relevance, practical or theoretical guidance value, and some generalizability across contexts, while stating boundaries and limitations explicitly. A conclusion is not stable unless it is supported by multiple views of the data, including positive evidence, boundary cases, uncertainty, and robustness or negative checks.

**（2026-07-31 依速度原则收窄）** 多轮独立复审只适用于**即将写进手稿的主张**，不适用于探索性、诊断性或描述性产出，也不适用于尚未运行的分析。对手稿级主张，默认序列仍是独立探索 → 交叉复审与证伪 → 最终无 blocker 复审；对其余一切，一轮自查即可。Aim for a small set of stable conclusions only when the data genuinely support them; do not inflate weak, local, or sampling-driven patterns into publication-level claims.

Every conclusion in a report artifact must have its own dedicated evidence and figure bundle. Figures under a conclusion must directly explain, test, qualify, or falsify that conclusion; remove charts that are merely adjacent or generally interesting. Use publication-grade figure standards: clean multi-panel layouts, readable labels, explicit units and sample sizes, uncertainty intervals or effect sizes where appropriate, colorblind-safe palettes, and export both PNG for viewing and PDF/SVG for publication editing when possible.

Before accepting any result set, check numerical health, coverage, and data integrity. Look for pathological constants, impossible values, duplicate keys, failed or missing cases, leakage, extreme sparsity, context imbalance, and sensitivity to preprocessing or model parameters. When comparing experimental batches, keep searchable summary tables and long-form detail tables, then compare distributions, signs, effect sizes, uncertainty, and data-health indicators.

Keep report packages reproducible but tidy. The reader-facing report folder should retain the final report, figure exports, figure manifest or chart map, conclusions summary, and evidence summary. Move rebuild inputs, scripts, review notes, old report iterations, obsolete figures, and audit files into a report-linked process archive with a README and integrity-check record. Delete only reproducible caches unless the user explicitly approves deeper cleanup.

## 自带上下文的汇报（PI ruling 2026-08-01）

**原则：每一份向上的汇报，都必须能被一个没有跟进过程的读者一次读懂。**
上下文重建的成本由写的人承担，不由读的人承担。

这条适用于本项目中**每一层**向上的汇报——codex agent 交给 leader、leader 交给监督方、
监督方交给 PI。每一层都倾向于假设读者一直在跟着看，而这个假设在每一层都不成立：
读者往往隔很久才看一次，并且会跳过中间的过程性汇报，直接看最新的一份。

四条硬性要求：

1. **先定位，再讲进度。** 开头必须交代三件事：这项工作要解决什么问题、整体已经走到哪一步、
   本次是其中哪一环。不得直接从增量讲起。
2. **不用黑话，不用比喻。** 必须使用项目专有名词时，当场用一句话说明它是什么。
   自造的形象化说法一律换成直白描述——例如「失效地板」应写成「失效比例的下限，
   任何 σ 取值都压不下去」；「救回率」应写成「原本判为不可估、改用 log 域计算后变为可估的比例」。
3. **结论与待决事项分开。** 需要上级拍板的事必须单独成节，写清选项、判断依据、以及不做的后果，
   不得藏在叙述中间当作陈述句带过。
4. **数字自带口径。** 给出百分比必须同时给分子、分母、筛选条件、来源文件与列名。
   一个读者无法自行复算的数字，等于没给。

反面基准（均为本项目真实发生过的）：把「D1 占 43%」单独引用而不带分源拆分（waymo 58.93% vs
nuplan 1.10%，合并值会误导）；把自造词当作读者已知的术语直接使用；把需要 PI 决策的事
写成叙述句混在段落里，导致它被当成已经定了的事。

## Shared Research Protocol
Merged GitHub files are the source of truth. Every durable research task should have an RQ ID, a GitHub issue when practical, and a row in `STUDIES.md`. Keep the execution layer and interpretation layer separate: `reports/studies/RQxxx_topic/RQxxx_n_topic_date/` records what was run, where reports/artifacts live, commands, environment, deviations, and claim-level evidence; `reports/knowledge/RQxxx_topic/` records ChatGPT/Claude/Codex/human reviews, synthesis, and final accepted/rejected claims. One RQ may have multiple execution reports, but it should have one knowledge folder. Never silently change an approved plan. Every manuscript-relevant report claim must have an evidence row. Do not edit the paper repository and this research repository in the same PR. Paper edits may only use claims accepted in `reports/knowledge/RQxxx_topic/decision.md`.

## Agent Workflow & Logging
Every time an agent workflow finishes, a summary of the task and its status must be logged in the `main_workflow.log` file located at the repository root. This log should capture the completion status, key outcomes, and any persistent artifacts generated.

## Imported Claude Cowork project instructions
