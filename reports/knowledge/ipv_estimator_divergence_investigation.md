# IPV Estimator Divergence Investigation

Worker: `INV-ipv-code-diff`  
Date: 2026-06-27  
Scope: current local `src/sociality_estimation` estimator versus the pinned legacy flat checkout that generated `sigma01_ipv_timeseries.csv`.

## Verdict

There is genuine local estimator logic drift from the sigma01-generation code. The current package layout was introduced later as a mostly layout/import refactor on 2026-06-19 (`0827f504c32cb0322979d27c4fe6c964113569b3`), but the first post-sigma01 behavior-changing estimator commit is 2026-06-15 (`a0fee5354ecafe5cb8dbcfc6507ec69efa973e14`, `Add realtime IPV sign estimator`).

The responsible logic is the `Agent` cost-helper rewrite in `a0fee535`: `cal_individual_cost` and `cal_group_cost` were changed from loop-based calculations to vectorized calculations used inside `solve_optimization`. Restoring those two old helper implementations inside the current package reproduced the legacy diagnostic frame exactly.

Corrected status: current local `src/sociality_estimation` is not sigma01-numeric-compatible; use the pinned HPC legacy estimator for sigma01/RQ009-target-compatible IPV unless a migration patch restores exact parity.

## Local Git Timeline

Commands run included `git log --follow --date=iso --stat` for:

- `src/sociality_estimation/core/agent.py`
- `src/sociality_estimation/core/ipv_estimation.py`
- `src/sociality_estimation/planning/Lattice.py`
- `src/sociality_estimation/planning/lattice_planner.py`
- `src/sociality_estimation/planning/utility.py`
- `src/sociality_estimation/planning/__init__.py`

Estimator-relevant commits:

| Commit | ISO date | Author | Subject | Classification |
|---|---:|---|---|---|
| `93d5fa7efc2c8621f9360ae460c9846c3f488fbd` | 2025-10-30 15:36:21 +0800 | Xiaocong Zhao | Initial project setup | Initial estimator/planning implementation. |
| `005b6dfed1bf017157c3cc1f14449d6ee36249b9` | 2025-10-31 09:37:12 +0800 | Xiaocong Zhao | Add IPV diagnostic helpers and visualization | Logic-affecting pre-sigma01 change: switched active global setting from T-intersection to Argoverse (`dt=0.1`, `TRACK_LEN=8`, `MIN_DIS=5`) and fixed per-call virtual-track reliability collection; also added diagnostics. |
| `a03b3570d8278ababc81444fe065a00855c2bf98` | 2025-10-31 16:59:55 +0800 | Xiaocong Zhao | Add interhub IPV processing pipeline | Estimator-adjacent reference support in `solve_linear_programming`; not the nonlinear `estimate_ipv_pair` SLSQP path responsible for the sigma01 drift. |
| `e14b1039cc3d043d7273d5386766dce4ccac8ca0` | 2026-06-07 22:22:14 +0800 | Xiaocong Zhao | Speed up remaining nuPlan IPV reruns | Pre-sigma01 reference-preparation speedup: accepts prepared `(cv, s)` tuples and avoids repeated smoothing in `utility_fun`. Included in sigma01 baseline. |
| `5edd28104bf5989e2dc258c9405ce897d7523cc4` | 2026-06-12 00:32:37 +0800 | Xiaocong Zhao | Run full InterHub IPV with sigma 0.1 | Sigma01 generation commit: changed `agent.sigma` from `0.02` to `0.1`. This is the pinned baseline. |
| `a0fee5354ecafe5cb8dbcfc6507ec69efa973e14` | 2026-06-15 12:52:29 +0800 | Xiaocong Zhao | Add realtime IPV sign estimator | Post-sigma01 genuine logic drift: added solver presets/current/realtime APIs and rewrote `estimate_self_ipv`, candidate helpers, `solve_optimization`, `cal_individual_cost`, `cal_group_cost`, and `utility_fun`. The cost-helper rewrite changes numeric optimizer results. |
| `0827f504c32cb0322979d27c4fe6c964113569b3` | 2026-06-19 23:01:38 +0800 | xiaocong | Make the repository navigable around canonical roots | Flat-to-package refactor: `agent.py` -> `src/sociality_estimation/core/agent.py`, `ipv_estimation.py` -> `src/sociality_estimation/core/ipv_estimation.py`, `tools/*` -> `src/sociality_estimation/planning/*`, `process_interhub.py` -> `pipelines/interhub/process_interhub.py`. Diff is imports/default paths/newline plus package marker; no additional estimator math drift found here. |

Local last estimator-file change: 2026-06-19 23:01:38 +0800, `0827f504`, package layout/import refactor.  
Local last estimator logic change: 2026-06-15 12:52:29 +0800, `a0fee535`, cost-helper/vectorization and realtime estimator changes.

## Pinned HPC Code Identity

Read-only HPC command used: `ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc ...`; no HPC writes or sbatch were performed.

HPC checkout:

- Path: `/share/home/u25310231/ZXC/ipv_estimation`
- Git HEAD: `5edd28104bf5989e2dc258c9405ce897d7523cc4`
- HEAD subject/date: 2026-06-12 00:32:37 +0800, `Run full InterHub IPV with sigma 0.1`

SHA-256:

| File | SHA-256 |
|---|---|
| `process_interhub.py` | `0a08606200c97fc4b340444b2cf56317c8905a24ea7a0712ea2b3f50f22b334d` |
| `agent.py` | `8c5c633846cb07c88c31c19770a6f69a7bcfde058c0dad291d1cc79ab4663d08` |
| `ipv_estimation.py` | `30b9fd0fbf615b737d201387710d0ebd986e468e49baa1b92b6cc7ca5e827dfe` |
| `tools/utility.py` | `46d58d2dfc13f2bbcf3acf4a2986dc4145d52fbd2d06e40e3f8a7a59fdc154c5` |
| `tools/Lattice.py` | `f3ca6075748e77d15e790316aaf28dfcb67710d2ef0091f52f64f301d926e70a` |
| `tools/lattice_planner.py` | `0626acbb747fc9753f2e7af5a1aa2cc9307ba3ba8c64ccc84c93d84a9b508934` |

The local planning helpers are layout-only different from the legacy `tools/` copies: `Lattice.py` is byte-identical; `utility.py` and `lattice_planner.py` differ only in import paths/newline.

## Function-Level Diff

No responsible differences found in:

- Sigma kernel: both use `sigma=0.1`, `sigma2=0.05`.
- Candidate IPV grid for the matched call: both use seven candidates `[-3,-2,-1,0,1,2,3] * pi/8`.
- Likelihood/weight aggregation: `cal_traj_reliability` source hash is unchanged; `ipv = sum(ipv_range * weight)` and `ipv_error = 1 - sqrt(sum(weight ** 2))` are unchanged.
- Entry-point parameters for the matched call: `history_window=10`, `min_observation=4`, reference clip margin 60 m, reference max points 40, reference smooth points 40, NuPlan 20 Hz -> 10 Hz downsample, and legacy/default SLSQP (`solver_preset="accurate"` locally).
- Planning/lattice helpers used by this path: layout/import-only differences.

Responsible differences:

- `src/sociality_estimation/core/agent.py:862-914` versus legacy `agent.py:746-810`: `cal_individual_cost` and `cal_group_cost` were vectorized in `a0fee535`. Although intended as a speed/refactor change, this changes the numeric SLSQP objective path enough to alter candidate trajectories and weights.
- `src/sociality_estimation/core/agent.py:600-646`: `solve_optimization` added optional `solver_options`; with `solver_options=None`, this is not the matched-run cause.
- `src/sociality_estimation/core/agent.py:672-721`: `estimate_self_ipv` now delegates through candidate-task helpers. Isolation test showed this is not the cause: current helper path and a legacy-style `copy.deepcopy` loop inside the current package had `track_max_abs=0.0` and `weight_max_abs=0.0`.
- `src/sociality_estimation/core/ipv_estimation.py:57-335`: adds reference pre-preparation, solver presets, parallel/current/realtime APIs. For the sigma01-compatible entrypoint, reference was already prepared as a `(cv, s)` tuple and `solver_preset="accurate"` resolves to default SLSQP; these are not the decisive cause.

Isolation evidence for first differing case/frame (`ipv_000001`, primary, frame 10):

- Current helper path versus current legacy-style deepcopy loop: `track_max_abs=0.0`, `weight_max_abs=0.0`.
- Current package with the old loop-based `cal_individual_cost` and `cal_group_cost` monkeypatched back in: `patched_weight_max_abs_vs_legacy=0.0`, `patched_virtual_last_max_abs_vs_legacy=0.0`.

## Controlled Re-Parity

A direct estimator-level A/B was run on the first four real cases from the original RQ009 parity sample, using the same entrypoint preprocessing and identical matched parameters:

- Cases: `ipv_000001`, `ipv_000004`, `ipv_000005`, `ipv_001788`
- Real pkl root: `data/interhub/raw/full_datasets/pkl`
- Frames: first 12 downsampled frames per case
- Frame rows: 48
- Estimated rows: 32 (`frame_index >= 4`)
- Values compared: 192 (`ipv_key_agent_1`, `ipv_key_agent_1_error`, `ipv_key_agent_2`, `ipv_key_agent_2_error`)

Result:

| Comparison | Max abs diff | Mean abs diff | Agree at 1e-9 |
|---|---:|---:|---|
| current local vs pinned legacy, matched params | `0.28116810052219554` | `0.01111504310902534` | no |
| estimated rows only | `0.28116810052219554` | `0.01667256466353801` | no |

Per-column max abs diffs:

| Column | Max abs diff |
|---|---:|
| `ipv_key_agent_1` | `0.28116810052219554` |
| `ipv_key_agent_1_error` | `0.2771876941733831` |
| `ipv_key_agent_2` | `0.16081548973998638` |
| `ipv_key_agent_2_error` | `0.060912816553648774` |

This matched-parameter rerun rules out a pure parameter/default/entrypoint mismatch. The earlier full 12-case parity result remains directionally correct (`local_vs_hpc` max abs diff about `1.1244582089284632`), but the stronger conclusion is now grounded in a controlled same-local-data A/B and a function-level isolation test.

## Commands Run

Key commands:

- `git log --follow --date=iso --stat -- <estimator files>`
- `git show --date=iso --stat --patch a0fee535... -- agent.py ipv_estimation.py`
- `ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'cd /share/home/u25310231/ZXC/ipv_estimation && git rev-parse HEAD && sha256sum ...'`
- `scp -o BatchMode=yes -o ConnectTimeout=12 ...` to copy read-only source into `/tmp/rq009_ipv_legacy_hpc_inv/`
- Direct Python A/B scripts using `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python`, `PYTHONPATH=/tmp/rq009_ipv_legacy_hpc_inv` for legacy and `PYTHONPATH=src` for current.

No paper repository files were read or edited. No analysis outputs were modified; scratch rerun artifacts were written under `/tmp`.
