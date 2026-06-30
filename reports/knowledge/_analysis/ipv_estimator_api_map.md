# IPV Estimator API Map And 3-Mode Refactor Proposal

Worker: `SURV-ipv-api`  
Role: API surveyor / designer  
Date: 2026-06-27  
Scope: current local `src/sociality_estimation` estimator, current InterHub caller, pinned sigma01-generation legacy checkout at `/share/home/u25310231/ZXC/ipv_estimation`, and local archive `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/`.

Read-only boundary: no tracked `src/` files and no paper repository files were edited. HPC access used `ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc '<read-only commands>'` only.

## Executive Summary

The current public estimator surface is centered on `estimate_ipv_pair(...)`, with online wrappers `estimate_ipv_current(...)` and `RealtimeIPVEstimator`. The current acceleration knobs (`solver_preset`, `solver_options`, `candidate_ipv_values`, `parallel_accurate`) do not restore sigma01 parity because the drift is not a solver/grid setting. The responsible drift is the post-sigma01 vectorization of `cal_individual_cost` and `cal_group_cost` in `a0fee535`; the pinned sigma01 producer used loop-based helpers.

Recommended selector: add `solver_mode: str = "exact"` as a new composite estimator selector, and keep `solver_preset` as a deprecated compatibility alias/low-level override. Reason: the PI's three modes select cost backend, candidate grid, solver effort, and parallelism; that is broader than a solver preset.

Recommended mode map:

| Mode | Cost backend | Candidate grid | Solver settings | Intended parity |
|---|---|---|---|---|
| `exact` | legacy loop helpers | 7 values `[-3,-2,-1,0,1,2,3] * pi/8` | SciPy SLSQP defaults (`solver_options=None`) | sigma01/HPC parity |
| `fast` | equivalence-fixed vectorized helpers | same 7 values | SciPy SLSQP defaults | same answer as `exact`, faster |
| `realtime` | equivalence-fixed vectorized helpers | 5 values `[-3,-1,0,1,3] * pi/8` | `{"maxiter": 8, "ftol": 1e-2}` by default | approximate |

## Public API Map

### `MotionSequence`

Location: `src/sociality_estimation/core/ipv_estimation.py:35-55`

Signature:

```python
@dataclass
class MotionSequence:
    data: np.ndarray
    target: str
    reference: Optional[np.ndarray] = None
```

Behavior:

- `data` must be a 2D array with at least five columns: `[x, y, vx, vy, heading]`.
- `target` is consumed by `Agent` and reference lookup logic.
- `reference` can be `None`, raw `(N, 2)` points, or in practice a prepared `(cv, s)` tuple after pipeline smoothing.

### `estimate_ipv_pair`

Location: `src/sociality_estimation/core/ipv_estimation.py:158-335`

Full current signature:

```python
def estimate_ipv_pair(
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    history_window: int = 10,
    min_observation: int = 4,
    return_diagnostics: bool = False,
    diagnostic_steps: Optional[Sequence[int]] = None,
    solver_preset: str = "accurate",
    solver_options: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
    candidate_executor=None,
    candidate_ipv_values: Optional[Sequence[float]] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
]:
```

Return shape:

- If `return_diagnostics=False`: `(ipv_values, ipv_errors)`.
- If `return_diagnostics=True`: `(ipv_values, ipv_errors, diagnostics)`.
- `ipv_values.shape == (T, 2)` and `ipv_errors.shape == (T, 2)`, where `T = min(len(primary.data), len(counterpart.data))`.
- Column 0 is `primary`; column 1 is `counterpart`.
- Current code initializes pre-estimation rows as `ipv_values=0.0`, `ipv_errors=1.0` (`src/.../ipv_estimation.py:223-224`), despite the docstring saying earlier rows are filled with `np.nan`.
- `diagnostics` is `{"primary": [...], "counterpart": [...]}`. Each item includes `step`, `start_index`, `observed`, `interacting`, `virtual_tracks`, `weights`, `ipv_range`, `ipv`, and `ipv_error` (`src/.../ipv_estimation.py:303-330`).

Current control flow:

- For each `t` from `min_observation` to `steps - 1`, use `start = max(0, t - history_window)` (`src/.../ipv_estimation.py:237-239`).
- Construct fresh `Agent` objects from the start-state of each window (`src/.../ipv_estimation.py:241-260`).
- Prepare references once before the loop via `_prepare_reference_for_repeated_use` (`src/.../ipv_estimation.py:229-230`).
- Estimate each side serially with `_estimate_agent_ipv`, or in a combined parallel candidate map when `active_candidate_executor` exists (`src/.../ipv_estimation.py:263-291`).

### `estimate_ipv_current`

Location: `src/sociality_estimation/core/ipv_estimation.py:338-388`

Full current signature:

```python
def estimate_ipv_current(
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    history_window: int = 10,
    return_diagnostics: bool = False,
    solver_preset: str = "parallel_accurate",
    solver_options: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
    candidate_executor=None,
    candidate_ipv_values: Optional[Sequence[float]] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
]:
```

Return shape:

- If `return_diagnostics=False`: `(ipv_pair, err_pair)` with both arrays shape `(2,)`.
- If `return_diagnostics=True`: `(ipv_pair, err_pair, diagnostics)`.
- It calls `estimate_ipv_pair(..., min_observation=steps - 1, diagnostic_steps=[steps - 1])` and returns the last row (`src/.../ipv_estimation.py:369-388`).

### `RealtimeIPVEstimator`

Location: `src/sociality_estimation/core/ipv_estimation.py:402-492`

Constructor:

```python
class RealtimeIPVEstimator:
    def __init__(
        self,
        *,
        history_window: int = 10,
        solver_preset: str = "parallel_accurate",
        solver_options: Optional[Dict[str, float]] = None,
        max_workers: Optional[int] = None,
        candidate_executor=None,
        candidate_ipv_values: Optional[Sequence[float]] = None,
    ):
```

Classmethod:

```python
@classmethod
def for_realtime_sign(cls, **kwargs):
```

- Sets `solver_preset="parallel_accurate"` if absent.
- Sets `candidate_ipv_values = SIGN_REALTIME_CANDIDATE_IPV_VALUES.copy()` if absent (`src/.../ipv_estimation.py:430-435`).

Instance methods:

```python
def close(self) -> None

def estimate_current(
    self,
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    return_diagnostics: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, np.ndarray]]]],
]

def estimate_sign_current(
    self,
    primary: MotionSequence,
    counterpart: MotionSequence,
    *,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Return shape:

- `estimate_current(...)` returns the same shape as `estimate_ipv_current(...)`.
- `estimate_sign_current(...)` returns `(signs, ipv_values, ipv_errors)`, all shape `(2,)`; signs are in `{-1, 0, 1}`.

Parallel behavior:

- Reuses an external executor if supplied.
- Otherwise opens a persistent `ProcessPoolExecutor(max_workers=max_workers)` only when `solver_preset == "parallel_accurate"` (`src/.../ipv_estimation.py:445-452`).

### `Agent` estimator-facing methods

Location: `src/sociality_estimation/core/agent.py`

Constructor:

```python
class Agent:
    def __init__(self, position, velocity, heading, target, acceleration=None)
```

Candidate optimizer:

```python
def solve_optimization(self, inter_track, *, solver_options=None)
```

- Uses zero controls as `u0`, acceleration/steering bounds, objective `utility_fun(...)`, and SciPy `minimize(..., method="SLSQP")` (`src/.../agent.py:600-646`).
- If `solver_options is not None`, passes them into `minimize(..., options=dict(solver_options))` (`src/.../agent.py:630-639`).
- Returns `self.trj_solution`, a full trajectory with columns `[x, y, vx, vy, heading]`.

Self-IPV estimator:

```python
def estimate_self_ipv(
    self,
    self_actual_track,
    inter_track,
    *,
    return_details=False,
    solver_options=None,
    candidate_executor=None,
    candidate_ipv_values=None,
)
```

- Mutates `self.ipv`, `self.ipv_error`, and `self.virtual_track_collection`.
- Returns `None` unless `return_details=True`.
- With details, returns `{"virtual_tracks": ..., "weights": ..., "ipv_range": ...}` (`src/.../agent.py:672-721`).

## Cost Objective, Candidate Generation, Likelihood, And Aggregation

### Constants and grids

Location: `src/sociality_estimation/core/agent.py:29-67`

- `dt = 0.1`
- `TRACK_LEN = 8`
- `MAX_DELTA_UT = 1e-4`
- `MIN_DIS = 5`
- `WEIGHT_DELAY = 0.3`, `WEIGHT_DEVIATION = 0.8`, normalized into `weight_metric`
- `WEIGHT_INT = 1`, `WEIGHT_GRP = 0.22`
- `MAX_STEERING_ANGLE = pi / 6`
- `MAX_ACCELERATION = 1.0`
- `INITIAL_IPV_GUESS = 0`
- `virtual_agent_IPV_range = np.array([-3, -2, -1, 0, 1, 2, 3]) * pi / 8`
- `sigma = 0.1`, `sigma2 = 0.05`

Realtime sign grid:

- `SIGN_REALTIME_CANDIDATE_IPV_VALUES = np.array([-3, -1, 0, 1, 3]) * pi / 8` (`src/.../ipv_estimation.py:29-32`).

### Candidate trajectory generation

Current package locations:

- `_resolve_candidate_ipv_values(candidate_ipv_values=None)`: `src/.../agent.py:109-112`
- `_build_candidate_ipv_tasks(...)`: `src/.../agent.py:115-135`
- `_solve_candidate_ipv_track(task)`: `src/.../agent.py:84-106`
- `_apply_candidate_ipv_tracks(...)`: `src/.../agent.py:138-164`
- `Agent.estimate_self_ipv(...)`: `src/.../agent.py:672-721`

Current behavior:

1. Candidate grid is resolved from explicit `candidate_ipv_values` or `virtual_agent_IPV_range`.
2. One task is built per candidate IPV, carrying the subject initial state, target, reference, counterpart track, and solver options.
3. Each task constructs a fresh `Agent`, sets `candidate_agent.ipv`, calls `solve_optimization(inter_track, solver_options=...)`, and returns only `virtual_track[:, 0:2]`.
4. Tracks are compared with the observed xy track by `cal_traj_reliability([], self_actual_track, virtual_tracks_recent, subject.target)`.
5. Aggregation sets:

```python
subject.ipv = sum(ipv_range * ipv_weight)
subject.ipv_error = 1 - np.sqrt(sum(ipv_weight ** 2))
```

Location: `src/.../agent.py:150-157`.

Legacy sigma01 behavior:

- `Agent.estimate_self_ipv(self_actual_track, inter_track, *, return_details=False)` deep-copied `self` inside a Python loop for each `ipv_temp` and called `solve_optimization` (`git show 5edd2810:agent.py:562-606`).
- Prior isolation found the new candidate-task helper path is not the drift source when cost helpers are held fixed.

### Lattice planning distinction

The estimator path does not call `lattice_planning(...)`. The active estimator candidate tracks use SLSQP plus `bicycle_model`.

- SLSQP path: `Agent.solve_optimization(...)` -> `utility_fun(...)` -> `bicycle_model(...)`.
- `bicycle_model` is in `src/sociality_estimation/planning/utility.py:70-93`.
- `lattice_planning(...)` is in `src/sociality_estimation/planning/lattice_planner.py:28-83` and is used by simulation controller branches (`pipelines/simulation/simulator.py:154-232`), not by `estimate_ipv_pair`.

### Objective construction

`Agent.solve_optimization(...)`:

- Builds `u0` as zero acceleration and steering controls for `track_len - 1` steps (`src/.../agent.py:618-620`).
- Bounds acceleration by `[-1.0, 1.0]` and steering by `[-pi/6, pi/6]` (`src/.../agent.py:622-625`).
- Objective is `fun = utility_fun(self_info, inter_track)` (`src/.../agent.py:627-628`).
- Uses `minimize(fun, u0, bounds=bds, method="SLSQP", options=solver_options_if_any)` (`src/.../agent.py:630-639`).

`utility_fun(...)`:

- Location: `src/.../agent.py:1012-1036`.
- Prepares reference once using `_prepare_reference`.
- For each control vector `u`, rolls out `track_self = bicycle_model(..., dt=0.1)[:, 0:2]`.
- Computes:

```python
interior_cost = cal_individual_cost(track_self, target=self_info[4], ref=prepared_ref)
group_cost = cal_group_cost([track_self, track_inter[:, 0:2]])
util = np.cos(self_info[3]) * interior_cost + np.sin(self_info[3]) * group_cost
```

### `cal_individual_cost`

Current vectorized helper:

- Location: `src/sociality_estimation/core/agent.py:862-892`.
- Reference is prepared via `_prepare_reference(target, track[0, :], ref)`.
- Converts `track` and reference to arrays, broadcasts all point-reference deltas, computes row-wise minimum distance:

```python
track_xy = np.asarray(track)[:, 0:2]
cv_xy = np.asarray(cv)[:, 0:2]
diff = cv_xy[None, :, :] - track_xy[:, None, :]
dis2cv = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
```

- Travel cost: `-norm(track[-1] - track[0]) / len(track)`.
- Lane deviation cost: `max(0.2, dis2cv.mean())`.
- Overall: `weight_metric.dot([cost_travel_distance, cost_mean_deviation]) * WEIGHT_INT`.

Legacy loop helper:

- Location: `git show 5edd2810:agent.py:746-783`; also `git show a0fee535^:agent.py:746-783`.
- It used explicit reference preparation logic and then:

```python
dis2cv = np.zeros([np.size(track, 0), 1])
for i in range(np.size(track, 0)):
    dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))
```

Diff scope:

- Responsible replacement introduced by `a0fee535`: legacy loop lines `746-783` -> new vectorized lines `856-886` in flat `agent.py`; current packaged equivalent is `src/.../agent.py:862-892`.
- Equivalence fix must either restore this loop path for `exact` or implement a vectorized path proven identical against it.

### `cal_group_cost`

Current vectorized helper:

- Location: `src/sociality_estimation/core/agent.py:895-914`.
- Converts tracks to arrays, computes relative position/distance and relative velocity arrays.
- Computes all collision factors and near-progress terms in vector form:

```python
collision_factor = np.where(dis_rel[1:] > 3, 0.5, 1.5)
nearness = collision_factor * np.sum(pos_rel[1:, :] * vel_rel, axis=1) / dis_rel[1:]
vel_rel_along_sum = np.sum((nearness + np.abs(nearness)) * 0.5)
cost_group = vel_rel_along_sum / TRACK_LEN
```

Legacy loop helper:

- Location: `git show 5edd2810:agent.py:786-810`; also `git show a0fee535^:agent.py:786-810`.
- It accumulated in Python loop order:

```python
vel_rel_along_sum = 0
for i in range(np.size(vel_rel, 0)):
    if dis_rel[i + 1] > 3:
        collision_factor = 0.5
    else:
        collision_factor = 1.5
    nearness_temp = collision_factor * pos_rel[i + 1, :].dot(vel_rel[i, :]) / dis_rel[i + 1]
    vel_rel_along_sum = vel_rel_along_sum + (nearness_temp + np.abs(nearness_temp)) * 0.5
cost_group = vel_rel_along_sum / TRACK_LEN
```

Diff scope:

- Responsible replacement introduced by `a0fee535`: legacy loop lines `786-810` -> vectorized lines `889-908` in flat `agent.py`; current packaged equivalent is `src/.../agent.py:895-914`.
- This loop-order difference is the more obvious floating-point sensitivity point because `np.sum` need not match Python accumulation order.

### Likelihood kernel and IPV/error aggregation

`cal_traj_reliability(...)`:

- Location: `src/sociality_estimation/core/agent.py:917-980`.
- The estimator calls this with `inter_track=[]`, so the active branch is trajectory similarity, not cost-preference similarity.
- For each candidate `i`, current and legacy compute:

```python
rel_dis = np.linalg.norm(virtual_track - act_track, axis=1)
var[i] = np.power(
    np.prod((1 / sigma / np.sqrt(2 * math.pi)) * np.exp(- rel_dis ** 2 / (2 * sigma ** 2))),
    1 / np.size(act_track, 0),
)
```

- If `sum(var) != 0`, weights are `var / sum(var)`; otherwise uniform over candidates (`src/.../agent.py:975-979`).
- `sigma=0.1`, `sigma2=0.05` are unchanged between current and sigma01 legacy.
- Aggregation unchanged: `ipv = sum(ipv_range * weight)`, `error = 1 - sqrt(sum(weight ** 2))`.

## Existing Knobs And Effects

### `solver_preset`

Location:

- Definition: `src/sociality_estimation/core/ipv_estimation.py:22-27`
- Resolution: `src/sociality_estimation/core/ipv_estimation.py:65-79`
- Main pair signature default: `src/.../ipv_estimation.py:166`
- Online current default: `src/.../ipv_estimation.py:344`
- Realtime class default: `src/.../ipv_estimation.py:416`
- InterHub CLI choices/default: `pipelines/interhub/process_interhub.py:1887-1897`

Accepted current values and effects:

| Value | Options resolved | Parallel effect | Candidate grid effect | Current default location |
|---|---|---|---|---|
| `"accurate"` | `None`, so SciPy SLSQP defaults | none unless `candidate_executor` supplied | none, defaults to 7-grid | `estimate_ipv_pair`, InterHub pipeline |
| `"parallel_accurate"` | `None`, same SLSQP defaults | opens/reuses a `ProcessPoolExecutor` when no external executor is supplied | none, defaults to 7-grid | `estimate_ipv_current`, `RealtimeIPVEstimator` |
| `"balanced"` | `{"maxiter": 20, "ftol": 1e-3}` | none unless executor supplied | none | explicit only |
| `"realtime"` | `{"maxiter": 8, "ftol": 1e-2}` | none unless executor supplied | none | explicit only |

Important current semantic mismatch:

- `solver_preset="realtime"` does not reduce the IPV candidate grid.
- The five-candidate realtime sign path is only selected by `RealtimeIPVEstimator.for_realtime_sign(...)`, which sets `candidate_ipv_values`, while leaving `solver_preset="parallel_accurate"`.
- `parallel_accurate` is a speed/parallelism knob only, not a numeric accuracy knob.

### `solver_options`

Location:

- Public parameter: `src/.../ipv_estimation.py:167`, `:345`, `:417`
- Resolution: `src/.../ipv_estimation.py:65-79`
- SLSQP pass-through: `src/.../agent.py:630-639`

Behavior:

- Optional dict merged over the selected preset options.
- No code-level validation of accepted keys; keys are passed through to SciPy SLSQP `options`.
- If merged options are empty, resolved value is `None`.
- Prior sweep found strict settings such as `{"ftol": 1e-14, "maxiter": 2000}` do not restore sigma01 parity.

### `candidate_ipv_values`

Location:

- Public parameter: `src/.../ipv_estimation.py:170`, `:348`, `:420`
- Resolution: `src/.../agent.py:109-112`

Behavior:

- `None` -> `virtual_agent_IPV_range = [-3,-2,-1,0,1,2,3] * pi/8`.
- Explicit sequence -> `np.asarray(candidate_ipv_values, dtype=float)`.
- `RealtimeIPVEstimator.for_realtime_sign(...)` sets `[-3,-1,0,1,3] * pi/8`.
- Changing candidate grid changes the estimator and is not a sigma01 parity fix.

### `max_workers` and `candidate_executor`

Location:

- Public parameters: `src/.../ipv_estimation.py:168-169`, `:346-347`, `:418-419`
- Pair-level executor setup: `src/.../ipv_estimation.py:231-235`
- Realtime persistent executor: `src/.../ipv_estimation.py:445-452`

Behavior:

- If `candidate_executor` is supplied, it is used for candidate maps.
- In `estimate_ipv_pair`, `solver_preset=="parallel_accurate"` with no supplied executor opens a local `ProcessPoolExecutor(max_workers=max_workers)`.
- In `RealtimeIPVEstimator`, a persistent pool is created only when `solver_preset=="parallel_accurate"` and no external executor exists.
- Parallelism changes throughput, not candidate grid or solver tolerances.

### `history_window` and `min_observation`

Locations:

- Estimator defaults: `history_window=10`, `min_observation=4` in `estimate_ipv_pair` (`src/.../ipv_estimation.py:162-163`).
- `estimate_ipv_current` has `history_window=10`; it internally uses `min_observation=steps-1`.
- InterHub constants: `pipelines/interhub/process_interhub.py:46-47`.

Behavior:

- `history_window` controls rolling prefix length: window start is `max(0, t - history_window)`.
- `min_observation` controls first estimated timestep.

### Pipeline-only preprocessing knobs

Current InterHub caller locations:

- `DATASET_DOWNSAMPLE_FACTORS = {"nuplan_train": 2}` at `pipelines/interhub/process_interhub.py:51`.
- Reference clipping/subsampling/smoothing functions at `pipelines/interhub/process_interhub.py:411-461`.
- CLI defaults for reference knobs are `0`, but sigma01 used explicit `60`, `40`, and `40`.

Behavior:

- For `nuplan_train`, aligned tracks are downsampled by factor 2 (`process_interhub.py:544-568`), documented as 20 Hz -> 10 Hz in the sigma01 Slurm script.
- `reference_clip_margin_m` clips references to the observed-motion bounding box plus margin.
- `reference_max_points` uniformly down-samples clipped references.
- `reference_smooth_points > 0` calls `smooth_ployline(reference, point_num=...)`.

## Sigma01 Generation Parameter Set

Pinned code:

- HPC path: `/share/home/u25310231/ZXC/ipv_estimation`
- Git HEAD: `5edd28104bf5989e2dc258c9405ce897d7523cc4`
- Subject/date: `2026-06-12 00:32:37 +0800 Run full InterHub IPV with sigma 0.1`
- Entrypoint: root-level `process_interhub.py`
- Core files: root-level `agent.py`, `ipv_estimation.py`

Exact generation parameters to reproduce in `exact` mode:

| Parameter | Value | Evidence |
|---|---|---|
| `agent.sigma` | `0.1` | HPC `agent.py:60`; archive Slurm prints `agent.sigma` |
| `agent.sigma2` | `0.05` | HPC/current `agent.py:61` |
| `history_window` | `10` | HPC `process_interhub.py:40`, CLI default `:1863`; archive command did not override |
| `min_observation` | `4` | HPC `process_interhub.py:41`, CLI default `:1864`; archive command did not override |
| Candidate IPV grid | `[-3,-2,-1,0,1,2,3] * pi/8` | HPC `agent.py:57`; legacy `estimate_self_ipv` loops over this |
| Solver | SciPy SLSQP defaults | HPC `agent.py:527-530`; no `solver_preset` or `solver_options` existed |
| Cost helpers | legacy Python loops | HPC `agent.py:746-810` |
| Reference clip margin | `60.0 m` | archive `submit_full_datasets_sigma01_array.sh:40,114`; summaries show `60.0` |
| Reference max points | `40` | archive `submit_full_datasets_sigma01_array.sh:41,115`; summaries show `40` |
| Reference smooth points | `40` | archive `submit_full_datasets_sigma01_array.sh:42,116`; summaries show `40` |
| Reference smoothing API | `smooth_ployline(reference, point_num=40)` | HPC `process_interhub.py:455-461` |
| nuPlan downsample | factor 2, 20 Hz -> 10 Hz | HPC `process_interhub.py:45,544-568`; archive script line 82 |
| Heading classifier | threshold 12 deg; labels made unique | HPC `process_interhub.py:571-587` |
| Plots | disabled | archive command `--no-plots` line 117 |
| Case timeout | `1800` seconds | archive script line 39 and command line 113 |
| Outer parallelism | 4 shards, 96 workers per shard, `fork` | archive script lines 35-37 and command lines 108-112 |
| Source CSV | `interhub_traj_lane/0_raw_data/full_datasets/index.csv` | archive script line 31 |
| pkl root | `interhub_traj_lane/0_raw_data/full_datasets/pkl` | archive script line 32 |
| Output root | `interhub_traj_lane/1_ipv_estimation_results/full_datasets/batches/20260612_sigma_0_1_full_rerun` | archive script line 33 |

Archive confirmation:

- `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/hpc_audit/processing_summary_shard_00_of_04.json` records workers `96`, start method `fork`, timeout `1800`, reference knobs `60/40/40`, no plots, 4 shards, and all 9557 shard-0 rows OK.
- `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/.../02_process/03_features/code_parity.md` records `HPC legacy hw=10 vs stored sigma01 max_abs_diff=1.1102230246251565e-16`.

## Call Sites

### Active local package internal calls

1. `src/sociality_estimation/core/ipv_estimation.py:103-107`
   - `_estimate_agent_ipv(...)` calls `agent.estimate_self_ipv(self_track, inter_track, return_details=..., solver_options=..., candidate_executor=..., candidate_ipv_values=...)`.

2. `src/sociality_estimation/core/ipv_estimation.py:122-155`
   - `_estimate_agent_pair_ipv_parallel(...)` uses `agent_module._build_candidate_ipv_tasks`, `candidate_executor.map(agent_module._solve_candidate_ipv_track, ...)`, and `_apply_candidate_ipv_tracks(...)`.

3. `src/sociality_estimation/core/ipv_estimation.py:369-381`
   - `estimate_ipv_current(...)` calls `estimate_ipv_pair(...)` with `min_observation=steps-1`, optional diagnostics on final step, and forwards solver/grid/parallel knobs.

4. `src/sociality_estimation/core/ipv_estimation.py:469-478`
   - `RealtimeIPVEstimator.estimate_current(...)` calls `estimate_ipv_current(...)` with its stored knobs.

### Primary active pipeline caller

`pipelines/interhub/process_interhub.py:1162-1168`

```python
ipv_values, ipv_errors = estimate_ipv_pair(
    seq_primary,
    seq_secondary,
    history_window=task.history_window,
    min_observation=task.min_observation,
    solver_preset=task.solver_preset,
)
```

Upstream:

- `CaseTask.solver_preset` default is `"accurate"` (`process_interhub.py:102`).
- `_build_tasks(...)`, `run_worker_benchmark(...)`, `_resolve_workers(...)`, and `run_processing(...)` all default/pass `solver_preset="accurate"` (`process_interhub.py:1267-1313`, `1612-1683`, `1702-1739`, `1742-1848`).
- CLI `--solver-preset` choices are `"accurate"`, `"parallel_accurate"`, `"balanced"`, `"realtime"` with default `"accurate"` (`process_interhub.py:1887-1897`).
- The pipeline does not expose `solver_options`, `candidate_ipv_values`, or `max_workers` for inner candidate parallelism.

### Active simulation direct callers

1. `pipelines/simulation/simulator.py:854-858`
   - Creates NDS-targeted `Agent(..., 'lt_nds')` and `Agent(..., 'gs_nds')`.
   - Calls `agent_lt.estimate_self_ipv(track_lt_temp, track_gs_temp)` and `agent_gs.estimate_self_ipv(track_gs_temp, track_lt_temp)`.
   - Passes no return details, solver options, executor, or candidate grid.

2. `pipelines/simulation/simulator.py:2206-2207`
   - Deep-copied simulation agents call `estimate_self_ipv(nds_trj_lt, nds_trj_gs)` and the reverse.
   - Passes no knobs.

### Archived legacy callers

1. `archived/legacy_scripts/process_argoverse.py:200-210`
   - Calls `estimate_ipv_pair(artifacts.motion_lt, artifacts.motion_gs, return_diagnostics=True, diagnostic_steps=debug_steps)` for debug, or `estimate_ipv_pair(artifacts.motion_lt, artifacts.motion_gs)` otherwise.
   - No solver/grid knobs.

2. `archived/legacy_scripts/process_interhub_json_legacy.py:575-582`
   - Calls `estimate_ipv_pair(seq_primary, seq_secondary, history_window=HISTORY_WINDOW, min_observation=MIN_OBSERVATION, return_diagnostics=enable_diagnostics, diagnostic_steps=diag_steps)`.
   - No solver/grid knobs.

### Tests and docs

- No active test currently calls `estimate_ipv_pair`; `tests/test_shortcut_scripts.py` is unrelated.
- `docs/realtime_ipv_estimator.md` documents `RealtimeIPVEstimator.for_realtime_sign(...)`, `RealtimeIPVEstimator(..., solver_preset="parallel_accurate")`, and the current sign/latency validation snapshot.

## Proposed 3-Mode Mapping

### Selector recommendation

Add a new public selector:

```python
solver_mode: str = "exact"
```

Keep `solver_preset` temporarily for compatibility, but deprecate it.

Why not reuse `solver_preset` directly:

- The PI's modes select cost helper backend, candidate grid, solver settings, and parallelism.
- Current `solver_preset` only resolves SLSQP `options` plus one special parallel branch.
- Current `solver_preset="realtime"` does not select the realtime five-candidate grid, so reusing the old name without a semantic break would keep a confusing API.

Recommended compatibility rule:

- If `solver_mode` is provided, it is authoritative.
- If `solver_mode` is omitted and `solver_preset` is provided, translate the old preset with a warning.
- If both are provided, allow only compatible combinations or raise `ValueError` with a clear message.
- In a later cleanup, remove `solver_preset` from docs and keep it as an alias only.

### Mode config table

Proposed internal config:

```python
IPV_MODE_CONFIGS = {
    "exact": {
        "cost_backend": "legacy_loop",
        "candidate_ipv_values": None,  # None resolves to 7-grid
        "solver_options": None,
        "parallel_candidates": False,
    },
    "fast": {
        "cost_backend": "legacy_equiv_vectorized",
        "candidate_ipv_values": None,  # same 7-grid
        "solver_options": None,
        "parallel_candidates": "auto_or_explicit",
    },
    "realtime": {
        "cost_backend": "legacy_equiv_vectorized",
        "candidate_ipv_values": SIGN_REALTIME_CANDIDATE_IPV_VALUES,
        "solver_options": {"maxiter": 8, "ftol": 1e-2},
        "parallel_candidates": "prefer_persistent_executor",
    },
}
```

Concrete behavior:

- `exact`
  - Use legacy loop helpers.
  - Use seven-candidate grid.
  - Use default SLSQP (`solver_options=None`) unless caller explicitly overrides, but parity tests should use no override.
  - Default `estimate_ipv_pair` should be `exact`.

- `fast`
  - Use equivalence-fixed vectorized helpers that have golden parity to `exact`.
  - Use seven-candidate grid.
  - Use default SLSQP.
  - For online wrappers, allow candidate parallelism/persistent pool. For batch InterHub, avoid implicit nested process pools unless requested because `process_interhub.py` already parallelizes over cases.

- `realtime`
  - Use equivalence-fixed vectorized helpers.
  - Use five-candidate grid `[-3,-1,0,1,3] * pi/8` unless caller explicitly provides `candidate_ipv_values`.
  - Use default realtime solver options `{"maxiter": 8, "ftol": 1e-2}` unless caller overrides.
  - `RealtimeIPVEstimator.for_realtime_sign(...)` becomes a thin alias for `RealtimeIPVEstimator(solver_mode="realtime", ...)`, with optional `threshold` still applied by `estimate_sign_current`.

### Where cost backends plug in

Minimal-churn path:

1. In `src/sociality_estimation/core/agent.py`, split current helpers into named backends near the existing helper locations:

```python
def _cal_individual_cost_legacy_loop(track, target, ref=None): ...
def _cal_group_cost_legacy_loop(track_packed): ...
def _cal_individual_cost_vectorized_equiv(track, target, ref=None): ...
def _cal_group_cost_vectorized_equiv(track_packed): ...

def cal_individual_cost(track, target, ref=None, *, cost_backend="legacy_loop"): ...
def cal_group_cost(track_packed, *, cost_backend="legacy_loop"): ...
```

2. Add a `cost_backend` argument through:

- `_solve_candidate_ipv_track(task)` task dict.
- `_build_candidate_ipv_tasks(..., cost_backend=...)`.
- `_estimate_agent_pair_ipv_parallel(..., cost_backend=...)`.
- `Agent.estimate_self_ipv(..., cost_backend="legacy_loop")`.
- `Agent.solve_optimization(..., cost_backend="legacy_loop")`.
- `utility_fun(self_info, track_inter, *, cost_backend="legacy_loop")`.

3. Do not pass Python function objects through `ProcessPoolExecutor` tasks. Pass a short string `cost_backend` because task dicts need to remain pickle-safe.

4. Preserve public `cal_individual_cost(...)` and `cal_group_cost(...)` names as exact-loop defaults so any direct external imports become sigma01-compatible by default.

Equivalence-fixed vectorization guidance:

- `cal_group_cost` must preserve legacy accumulation order or prove that the alternative sum is bit/near-bit identical. A naive `np.sum` is not sufficient for the golden test.
- `cal_individual_cost` can use vectorization for per-point distances, but the final shape/mean and distance tie behavior must be tested against the legacy loop.
- If true vectorized parity fails, keep `fast` behind a feature flag or make it use exact loops plus candidate parallelism until a parity-proof vectorized helper exists. Do not ship a mode called `fast` as "same answer" without the golden test.

### Backward compatibility plan

Public defaults:

- `estimate_ipv_pair(..., solver_mode="exact")` by default.
- `estimate_ipv_current(..., solver_mode="fast")` is acceptable for online speed, but if strict default consistency matters, use `solver_mode="exact"` and let `RealtimeIPVEstimator.for_realtime_sign` opt into realtime. The PI request specifically says default = exact, so the safer API default is exact everywhere except explicit realtime factory aliases.
- `RealtimeIPVEstimator(..., solver_mode="fast")` may preserve the old online default intent, but document that the no-argument scientific default is exact in `estimate_ipv_pair`.

Old `solver_preset` alias mapping:

| Old value | Proposed mapping | Notes |
|---|---|---|
| `"accurate"` | `solver_mode="exact"` | Aligns with old documentation promise of legacy-equivalent accurate output. |
| `"parallel_accurate"` | `solver_mode="fast"`, seven-grid, default SLSQP, candidate parallel enabled/reused | Preserves speed-only intent while making math exact-equivalent. |
| `"balanced"` | `solver_mode="fast"`, seven-grid, `solver_options={"maxiter": 20, "ftol": 1e-3}` | Deprecated compatibility path; not one of the PI's final modes. |
| `"realtime"` | `solver_mode="realtime"` | New composite realtime should use five-grid. To reproduce old seven-grid realtime, caller can pass explicit seven-grid `candidate_ipv_values`. |

Pipeline migration:

- Add `--solver-mode {exact,fast,realtime}` default `exact`.
- Keep `--solver-preset` hidden or documented as deprecated. If supplied, translate to the table above.
- Record both `solver_mode` and any deprecated `solver_preset` in metadata for traceability.
- For sigma01-compatible reruns, use `--solver-mode exact --history-window 10 --min-observation 4 --reference-clip-margin-m 60 --reference-max-points 40 --reference-smooth-points 40`.

Docs migration:

- Update `docs/realtime_ipv_estimator.md` so "accuracy-preserving" means `solver_mode="fast"` only after golden parity passes.
- Make `for_realtime_sign(...)` explicitly approximate.

## Golden Parity Regression Test Plan

Add tests under:

```text
tests/test_ipv_estimator_parity.py
tests/fixtures/ipv_sigma01_parity/
```

Recommended fixtures:

1. A tiny frozen fixture derived from existing RQ009 parity artifacts:
   - Source cases: `data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/03_features/parity/code_parity_sample_cases.csv`
   - Stored sigma01 expected rows: `.../03_features/parity/sigma01_sample_timeseries.csv`
   - Existing report: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/code_parity.md`

2. To keep the unit test lightweight and not dependent on large ignored pkl files, materialize a compact fixture with:
   - `primary_motion`, `secondary_motion`
   - prepared estimation references after clip/smooth (`60/40/40`)
   - expected `ipv_values` and `ipv_errors` from pinned HPC legacy for a few representative cases/frames
   - metadata with HPC git head `5edd28104bf5989e2dc258c9405ce897d7523cc4`

Test assertions:

- `estimate_ipv_pair(..., solver_mode="exact")` equals frozen legacy expected values with `max_abs_diff <= 1e-16` or `np.testing.assert_allclose(..., atol=1e-16, rtol=0)`.
- `estimate_ipv_pair(..., solver_mode="fast")` equals `exact` on the same fixture with the same tolerance before exposing `fast` as same-answer.
- `solver_mode="realtime"` is allowed to differ, but should return finite values in the candidate range and errors in `[0, 1]`.
- Alias tests:
  - `solver_preset="accurate"` resolves to `solver_mode="exact"`.
  - `solver_preset="parallel_accurate"` resolves to `solver_mode="fast"`.
  - `candidate_ipv_values` explicit override still wins.

Optional lower-level tests:

- Directly compare loop and vectorized-equivalent `cal_individual_cost` and `cal_group_cost` on deterministic tracks/references.
- Include a group-cost case that stresses accumulation order and near-zero/negative nearness clipping.

## Risks And Complications

- Default output change: current local `estimate_ipv_pair()` default is vectorized-drifted `"accurate"`; changing default to exact will intentionally change local outputs back to sigma01-compatible values.
- `solver_preset` name is misleading for composite modes; keeping it as the only selector risks future confusion.
- Candidate parallelism can oversubscribe CPUs when `process_interhub.py` already parallelizes over cases.
- `ProcessPoolExecutor` tasks must remain pickle-safe; use string backend identifiers, not function closures.
- `agent.py` uses module-level globals (`dt`, `TRACK_LEN`, `sigma`, `sigma2`, `virtual_agent_IPV_range`) that are not formal API parameters.
- `estimate_ipv_pair` docstring currently says early rows are `np.nan`, but code returns `0.0` IPV and `1.0` error before `min_observation`.
- The current `realtime` preset is solver-only, while the desired `realtime` mode is solver plus reduced grid. This needs explicit migration notes.
- The fast vectorized-equivalent backend may be hard to make exactly equal at `1e-16` if it changes reduction order. Gate it on tests, not intention.
- The active simulation code calls `Agent.estimate_self_ipv(...)` directly, so adding a backend/mode only to `estimate_ipv_pair` is insufficient.
- Archived legacy scripts import old root-layout modules and may not be runnable without path restoration, but their call style shows the no-knob default expectation.

## Key Evidence References

- Current `SOLVER_PRESETS`: `src/sociality_estimation/core/ipv_estimation.py:22-27`
- Current `estimate_ipv_pair` signature and return docs: `src/sociality_estimation/core/ipv_estimation.py:158-217`
- Current pair loop and parallel branch: `src/sociality_estimation/core/ipv_estimation.py:231-291`
- Current `estimate_ipv_current`: `src/sociality_estimation/core/ipv_estimation.py:338-388`
- Current `RealtimeIPVEstimator`: `src/sociality_estimation/core/ipv_estimation.py:402-492`
- Current candidate grid and sigma globals: `src/sociality_estimation/core/agent.py:61-67`
- Current candidate task helpers and IPV/error aggregation: `src/sociality_estimation/core/agent.py:84-164`
- Current `solve_optimization`: `src/sociality_estimation/core/agent.py:600-646`
- Current `estimate_self_ipv`: `src/sociality_estimation/core/agent.py:672-721`
- Current vectorized cost helpers: `src/sociality_estimation/core/agent.py:862-914`
- Current likelihood kernel: `src/sociality_estimation/core/agent.py:917-980`
- Current objective `utility_fun`: `src/sociality_estimation/core/agent.py:1012-1036`
- Legacy loop cost helpers: `git show 5edd2810:agent.py:746-810`
- Legacy default SLSQP and `estimate_self_ipv`: `git show 5edd2810:agent.py:527-606`
- Legacy `estimate_ipv_pair` no solver/grid knobs: `git show 5edd2810:ipv_estimation.py:42-176`
- Legacy InterHub constants and downsample: `git show 5edd2810:process_interhub.py:40-45`, `:544-568`
- Sigma01 Slurm parameters: `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/hpc_run_files/submit_full_datasets_sigma01_array.sh:30-42`, `:103-118`
- Sigma01 processing summary: `archived/report_process/interhub_20260612_sigma_0_1_full_rerun/01_process/hpc_audit/processing_summary_shard_00_of_04.json`
- RQ009 code parity: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/03_features/code_parity.md`
- Prior divergence note: `reports/knowledge/ipv_estimator_divergence_investigation.md`
- Prior hyperparameter note: `reports/knowledge/ipv_accel_hyperparam_finding.md`
