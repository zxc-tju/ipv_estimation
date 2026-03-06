# Repository Guidelines

## Project Structure & Module Organization
Core estimation logic lives in `agent.py`, `simulator.py`, and `ipv_estimation.py`; keep shared helpers in `tools/`. Dataset-specific pipelines are `process_argoverse.py` (expects folders under `argoverse/0_souce_data/`) and `process_interhub.py` (targets JSONs in `interhub_traj_lane/`). Generated reports land in `argoverse/1_experiment_result/` and `interhub_traj_lane/ipv_estimation/`; these paths are gitignored, so stage only code and lightweight configs.

## Build, Test, and Development Commands
Create environments with `python -m venv venv && venv\Scripts\activate` on Windows or `conda create -n ipv python=3.9`. Install dependencies via `pip install -r requirements-minimal.txt` for clusters or `requirements.txt` locally. Run Argoverse processing with `python process_argoverse.py`; invoke Interhub jobs through `python process_interhub.py --diagnostics trajectory_data_interaction_single.json`. For batch HPC runs submit `sbatch submit.sh`, which parallelises across JSON datasets and propagates the `--workers` value.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` functions, and `CamelCase` classes. Modules favour explicit type hints and docstrings; mirror that when extending APIs. Prefer `logging` over `print` (see existing `LOGGER` usage) and keep plotting helpers under the dataset modules. Place reusable utilities inside `tools/` rather than the dataset scripts.

## Testing Guidelines
There is no automated suite yet; validate changes by rerunning the relevant pipeline with a representative dataset and comparing the generated Excel/plot artifacts. When adding functionality, include lightweight `pytest`-style checks under a new `tests/` directory that exercise `estimate_ipv_pair` or trajectory transforms, and document any sample input you rely on. Capture key diagnostic figures (`--diagnostics`) for regression tracking.

## Commit & Pull Request Guidelines
Commits in this repo use short, imperative subjects (for example, `Enhance process_interhub.py...`). Group related edits and avoid bundling data drops. Pull requests should outline the scenario touched, CLI commands executed, and notable output locations; attach diffs or screenshots of plots when behaviour shifts. Cross-link issues or task IDs where applicable.

## Data & Configuration Notes
Large trajectory files stay outside version control; if you must share samples, provide download instructions instead. Keep environment-specific tweaks in separate config files or guarded by CLI flags, and sanitise paths before publishing job scripts.

## Agent Workflow & Logging
Every time an agent workflow finishes, a summary of the task and its status must be logged in the `main_workflow.log` file located at the repository root. This log should capture the completion status, key outcomes, and any persistent artifacts generated.
