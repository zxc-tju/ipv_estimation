# IPV Acceleration Hyperparameter Finding

Worker: `INV-ipv-accel-hyperparam`  
Date: 2026-06-27  
Question: is the post-sigma01 IPV drift from `a0fee535` an acceleration with a tunable accuracy hyperparameter that can restore sigma01 parity without reverting the speedup?

## Verdict

No. I found speed and solver-effort knobs, but no accuracy hyperparameter that makes the vectorized objective reproduce the sigma01/HPC legacy estimator.

The current vectorized estimator at its default accurate setting differs from the archived HPC/sigma01 baseline on the 4-case, first-12-frame parity subset with `max_abs_diff=0.3141348682582335`, `mean_abs_diff=0.023990428243039392` over 192 compared IPV/error values. Tightening SLSQP tolerances did not reduce the difference: the strictest tested setting, `solver_options={"ftol": 1e-14, "maxiter": 2000}`, had `max_abs_diff=0.626191780704079`. Denser candidate IPV grids also did not restore parity.

Recommendation: do not treat `solver_preset`, `solver_options`, or `candidate_ipv_values` as a sigma01-compatibility switch. For sigma01-consistent local output, use the pinned HPC legacy estimator or patch the vectorized helpers to be numerically equivalent to the legacy objective. There is no current config-only fix.

## Acceleration Diff Summary

Inspected command:

```bash
git -C "$REPO_ROOT" show --date=iso --stat --patch a0fee535 -- src/sociality_estimation/core/agent.py agent.py
```

`a0fee5354ecafe5cb8dbcfc6507ec69efa973e14` (`Add realtime IPV sign estimator`, 2026-06-15 12:52:29 +0800) changed the flat `agent.py` before the later package refactor.

Relevant changes:

- Added candidate-task helpers: `_solve_candidate_ipv_track`, `_build_candidate_ipv_tasks`, `_apply_candidate_ipv_tracks`, and optional `candidate_executor`.
- Added `candidate_ipv_values`; default still resolves to the legacy seven-candidate grid `[-3,-2,-1,0,1,2,3] * pi/8`.
- Added `solve_optimization(..., solver_options=None)` and passes options through to SciPy `minimize(..., method="SLSQP")`.
- Added `solver_preset` in `ipv_estimation.py`: `accurate=None`, `parallel_accurate=None`, `balanced={"maxiter":20,"ftol":1e-3}`, `realtime={"maxiter":8,"ftol":1e-2}`.
- Rewrote `cal_individual_cost`: legacy loops over each planned point and calls `amin(norm(cv - point))`; current broadcasts a full point-by-reference distance matrix and takes row minima.
- Rewrote `cal_group_cost`: legacy accumulates nearness in a Python loop; current uses `np.where`, vector dot products, and `np.sum`.
- `utility_fun` now prepares the reference once through `_prepare_reference`.

There is no parameter inside the vectorized `cal_individual_cost` or `cal_group_cost` that controls approximation granularity, integration steps, sample count, truncation, clipping, or tolerance. The vectorized objective body is hard-coded.

## Candidate Knobs Checked

| Candidate | Location | Current default | Legacy-equivalent value | Finding |
|---|---|---:|---:|---|
| `solver_preset` / `solver_options` | `src/sociality_estimation/core/ipv_estimation.py:22-27`, `:65-79`, `:158-170`; `src/sociality_estimation/core/agent.py:600-639` | `solver_preset="accurate"`, `solver_options=None` | Same as current default | Controls SLSQP effort/latency, not vectorization parity. Tightening worsened drift. |
| `candidate_ipv_values` | `src/sociality_estimation/core/agent.py:63`, `:109-112`; `src/sociality_estimation/core/ipv_estimation.py:170` | `None -> [-3,-2,-1,0,1,2,3] * pi/8` | Same seven-candidate grid | Changing the grid changes the estimator and did not restore sigma01 parity. |
| `parallel_accurate` / `max_workers` | `src/sociality_estimation/core/ipv_estimation.py:22-27`, `:231-235` | off for `estimate_ipv_pair`, on by default for realtime wrapper | Numerically same as `accurate` | Throughput knob only; same diff as default. |
| Vectorized helper resolution | `src/sociality_estimation/core/agent.py:862-914` | no knob | none | Genuine non-equivalent numeric objective path. |

## Baseline Truth

Archived RQ009 code-parity evidence verifies that the HPC legacy run reproduces sigma01:

- Full archived parity sample: 12 cases, 986 rows, 3,944 values.
- `hpc_hw10_vs_sigma01`: `max_abs_diff=1.1102230246251565e-16`, `mean_abs_diff=4.753700101329432e-18`.
- Archived current local `local_hw10_vs_sigma01`: `max_abs_diff=1.1244582089284632`, `mean_abs_diff=0.024589097602074055`.

For the high-accuracy sweep I used the same first four parity cases highlighted in the prior divergence investigation (`ipv_000001`, `ipv_000004`, `ipv_000005`, `ipv_001788`), first 12 downsampled frames per case, `history_window=10`, `min_observation=4`, reference clip 60 m, max reference points 40, smooth points 40. This gives 64 perspective-anchors and 192 compared IPV/error values. The archived HPC case files match sigma01 on this subset at `max_abs_diff=9.020562075079397e-17`.

## Speed/Accuracy Sweep

All timings are local scratch timings under `/Users/xiaocong/.rq009_codex_fleet/venv/bin/python`; they are useful for relative local speed only. Diff is versus archived HPC/sigma01 values.

| Setting | Max abs diff | Mean abs diff | Median sec / 64 anchors | Sec / anchor | Notes |
|---|---:|---:|---:|---:|---|
| scratch current with loop helper monkeypatch | `0.45851989560527673` | `0.023104985237703546` | `16.707061125` | `0.2610478301` | Local control only; not used as truth because archived HPC is the verified sigma01 producer. |
| `solver_preset="accurate"` | `0.3141348682582335` | `0.023990428243039392` | `10.503153000` | `0.1641117656` | Current default; not parity. |
| `solver_preset="parallel_accurate", max_workers=4` | `0.3141348682582335` | `0.023990428243039392` | `7.717459209` | `0.1205853001` | Same numerics as default; faster, still not parity. |
| `accurate`, `{"ftol":1e-12,"maxiter":1000}` | `0.6732598991849208` | `0.02712463647119546` | `41.375543875` | `0.6464928730` | High accuracy setting; worse and slower. |
| `accurate`, `{"ftol":1e-14,"maxiter":2000}` | `0.626191780704079` | `0.02681366643959845` | `50.684495666` | `0.7919452448` | Finest tested tolerance; still not parity. |
| `accurate`, `{"ftol":1e-12,"maxiter":1000,"eps":1e-9}` | `1.0346763307607156` | `0.06003802378397707` | `22.618367084` | `0.3534119857` | Finite-difference step change worsened drift. |
| `solver_preset="balanced"` | `2.3561900603011603` | `0.09656862802380371` | `1.343641375` | `0.0209943965` | Faster but much less sigma01-compatible. |
| `solver_preset="realtime"` | `2.2432316599877025` | `0.1370229429974629` | `0.592178041` | `0.0092527819` | Fastest tested; not parity. |
| 13-candidate IPV grid, `pi/16` step | `0.4446384691199836` | `0.051331565062312345` | `19.621541541` | `0.3065865866` | Denser grid did not help. |
| 25-candidate IPV grid, `pi/32` step | `0.4300682637893677` | `0.08332207671140929` | `37.973214125` | `0.5933314707` | Denser grid did not help. |

Fastest setting with the default vectorized seven-candidate model was `parallel_accurate` at about `2.17x` faster than the scratch loop-helper control, but it is not parity-accurate. There is no parity-accurate vectorized setting in the sweep, so `speedup_at_parity` is not defined.

## Decisive Test Result

The PI hypothesis predicts that increasing an accuracy knob should drive the vectorized estimator back to legacy/sigma01 output.

Observed:

- Current default accurate: `max_abs_diff=0.3141348682582335`.
- Current high-accuracy SLSQP, `ftol=1e-14,maxiter=2000`: `max_abs_diff=0.626191780704079`.
- Dense candidate grids: best tested max diff `0.4300682637893677`.
- `parallel_accurate`: exact same values as default, only faster.

Therefore the drift is not a tunable resolution/accuracy tradeoff. The responsible change remains the non-parameterized vectorized objective helper rewrite in `cal_individual_cost` / `cal_group_cost`.

## Files And Scratch

- Scratch script: `/tmp/ipv_accel_hyperparam_INV/run_experiment.py`
- Scratch result JSON: `/tmp/ipv_accel_hyperparam_INV/results.json`
- HPC access was read-only: verified `/share/home/u25310231/ZXC/ipv_estimation` at git `5edd28104bf5989e2dc258c9405ce897d7523cc4` and streamed source into `/tmp`.
- No tracked `src/` files and no paper repository files were edited.
