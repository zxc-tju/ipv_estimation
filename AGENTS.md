# Repository Guidelines

## Quick Orientation & START_HERE Maintenance
Begin each new agent thread by reading `START_HERE.md` beside this file, then use `AGENTS.md` for durable rules and `PROJECT_STRUCTURE.md` for deeper architecture notes. Keep `START_HERE.md` as the short current operating brief: current active batch, canonical data paths, canonical output paths, test command, app/server start instructions, protected files/directories, latest stable report/result, and known weak spots.

Whenever a workflow changes any of those current-operating facts, update `START_HERE.md` in the same task before finishing. If the facts are uncertain, write the uncertainty explicitly instead of leaving stale guidance. Log the maintenance outcome in `main_workflow.log` together with the normal workflow summary.

## Project Structure & Module Organization
Core estimation logic lives in `agent.py`, `simulator.py`, and `ipv_estimation.py`; keep shared helpers in `tools/`. The active dataset pipeline is `process_interhub.py`, which targets InterHub CSV/pkl datasets, including subset and full-dataset runs. Legacy root-level scripts have been moved to `archived/legacy_scripts/`; review and restore them only if you need old Argoverse CSV, old InterHub JSON, backward-compatible subset wrapper, or old `mean_ipv` metadata post-processing workflows. Generated reports land in `argoverse/1_experiment_result/` and `interhub_traj_lane/1_ipv_estimation_results/`; these paths are gitignored, so stage only code and lightweight configs.

## Build, Test, and Development Commands
Create environments with `python -m venv venv && venv\Scripts\activate` on Windows or `conda create -n ipv python=3.9`. Install dependencies via `pip install -r requirements-minimal.txt` for clusters or `requirements.txt` locally. Invoke current InterHub CSV/pkl jobs through `python process_interhub.py --csv <input.csv> --pkl-root <pkl_dir>`. Archived legacy scripts are not active commands until restored and path-checked. Legacy HPC Slurm command files are retained in report-linked process archives under `interhub_traj_lane/` rather than as root-level tracked scripts.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` functions, and `CamelCase` classes. Modules favour explicit type hints and docstrings; mirror that when extending APIs. Prefer `logging` over `print` (see existing `LOGGER` usage) and keep plotting helpers under the dataset modules. Place reusable utilities inside `tools/` rather than the dataset scripts.

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

## Agent Workflow & Logging
Every time an agent workflow finishes, a summary of the task and its status must be logged in the `main_workflow.log` file located at the repository root. This log should capture the completion status, key outcomes, and any persistent artifacts generated.
