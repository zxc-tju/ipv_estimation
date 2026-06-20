# Repository Guidelines

## Quick Orientation & START_HERE Maintenance
Begin each new agent thread by reading `START_HERE.md` beside this file, then use `AGENTS.md` for durable rules and `PROJECT_STRUCTURE.md` for deeper architecture notes. Keep `START_HERE.md` as the short current operating brief: current active batch, canonical data paths, canonical output paths, test command, app/server start instructions, protected files/directories, latest stable report/result, and known weak spots.

Whenever a workflow changes any of those current-operating facts, update `START_HERE.md` in the same task before finishing. If the facts are uncertain, write the uncertainty explicitly instead of leaving stale guidance. Log the maintenance outcome in `main_workflow.log` together with the normal workflow summary.

## Project Structure & Module Organization
Reusable estimation logic lives under `src/sociality_estimation/`: `core/agent.py` and `core/ipv_estimation.py` hold the IPV model/estimator, while `planning/` holds lattice and geometry helpers. Dataset and experiment entrypoints live under `pipelines/`, with the active InterHub CSV/pkl pipeline at `pipelines/interhub/process_interhub.py` and simulation code at `pipelines/simulation/simulator.py`. Raw/local data lives under ignored `data/` subdirectories such as `data/interhub/raw/` and `data/onsite_competition/raw/`; track only README/manifest/index files from `data/` unless explicitly requested. Large derived data lives under ignored `data/derived/`. The research knowledge base has two layers under `reports/`: execution reports under `reports/studies/`, and reviewer synthesis/decisions under `reports/knowledge/`; `STUDIES.md` is the root index. `reports/` must keep only `studies/` and `knowledge/` as first-level directories. The old root-level compatibility files (`agent.py`, `ipv_estimation.py`, `process_interhub.py`, `simulator.py`, and `tools/`) were archived under `archived/compat_wrappers_20260619/` and are not active entrypoints. Legacy dataset scripts live under `archived/legacy_scripts/`; review and restore them only if you need old Argoverse CSV, old InterHub JSON, subset wrapper, or old `mean_ipv` metadata post-processing workflows. Report-linked process archives live under `archived/report_process/`; local agent state lives under `archived/report_local_state/`. Manuscript drafting is now in the separate paper repository at `../9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`; do not recreate a top-level `paper/` directory here.

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

## Scientific Analysis & Reporting Guidelines
Start every research analysis by making the research question, unit of analysis, variable meanings, inclusion/exclusion criteria, and data provenance explicit. When semantics are unclear, inspect local dictionaries, source documentation, or generation scripts before interpreting a field. Distinguish descriptive patterns, predictive associations, and causal claims; do not use causal language unless the design supports it.

Treat publication-oriented analyses as claim-indexed research, not as a dashboard of interesting plots. Prioritize conclusions with broad relevance, practical or theoretical guidance value, and some generalizability across contexts, while stating boundaries and limitations explicitly. A conclusion is not stable unless it is supported by multiple views of the data, including positive evidence, boundary cases, uncertainty, and robustness or negative checks.

For high-stakes or full-dataset conclusion reports, use multiple independent evidence roles and at least three rounds of discussion/review before freezing major claims. A good default sequence is independent exploration, cross-review and falsification, then final no-blocker review. Aim for a small set of stable conclusions only when the data genuinely support them; do not inflate weak, local, or sampling-driven patterns into publication-level claims.

Every conclusion in a report artifact must have its own dedicated evidence and figure bundle. Figures under a conclusion must directly explain, test, qualify, or falsify that conclusion; remove charts that are merely adjacent or generally interesting. Use publication-grade figure standards: clean multi-panel layouts, readable labels, explicit units and sample sizes, uncertainty intervals or effect sizes where appropriate, colorblind-safe palettes, and export both PNG for viewing and PDF/SVG for publication editing when possible.

Before accepting any result set, check numerical health, coverage, and data integrity. Look for pathological constants, impossible values, duplicate keys, failed or missing cases, leakage, extreme sparsity, context imbalance, and sensitivity to preprocessing or model parameters. When comparing experimental batches, keep searchable summary tables and long-form detail tables, then compare distributions, signs, effect sizes, uncertainty, and data-health indicators.

Keep report packages reproducible but tidy. The reader-facing report folder should retain the final report, figure exports, figure manifest or chart map, conclusions summary, and evidence summary. Move rebuild inputs, scripts, review notes, old report iterations, obsolete figures, and audit files into a report-linked process archive with a README and integrity-check record. Delete only reproducible caches unless the user explicitly approves deeper cleanup.

## Shared Research Protocol
Merged GitHub files are the source of truth. Every durable research task should have an RQ ID, a GitHub issue when practical, and a row in `STUDIES.md`. Keep the execution layer and interpretation layer separate: `reports/studies/RQxxx_topic/RQxxx_n_topic_date/` records what was run, where reports/artifacts live, commands, environment, deviations, and claim-level evidence; `reports/knowledge/RQxxx_topic/` records ChatGPT/Claude/Codex/human reviews, synthesis, and final accepted/rejected claims. One RQ may have multiple execution reports, but it should have one knowledge folder. Never silently change an approved plan. Every manuscript-relevant report claim must have an evidence row. Do not edit the paper repository and this research repository in the same PR. Paper edits may only use claims accepted in `reports/knowledge/RQxxx_topic/decision.md`.

## Agent Workflow & Logging
Every time an agent workflow finishes, a summary of the task and its status must be logged in the `main_workflow.log` file located at the repository root. This log should capture the completion status, key outcomes, and any persistent artifacts generated.

## Imported Claude Cowork project instructions
