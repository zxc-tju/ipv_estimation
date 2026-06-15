# Realtime IPV Parallel Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a realtime IPV estimation path that preserves the accurate candidate-trajectory model while parallelizing per-candidate work so sign accuracy can stay above 90% in validation.

**Architecture:** Keep the existing serial estimator as the reference path. Add an optional parallel candidate executor inside `Agent.estimate_self_ipv`, then expose it through `estimate_ipv_pair` and `estimate_ipv_current` with a realtime-accurate preset. The first guaranteed mode should compute the same IPV candidates as the accurate mode; approximate modes remain explicit fallback choices.

**Tech Stack:** Python 3.9+, NumPy, SciPy SLSQP, `concurrent.futures`, pytest.

---

### Task 1: Parallel Candidate Estimation API

**Files:**
- Modify: `agent.py`
- Modify: `ipv_estimation.py`
- Test: `tests/test_ipv_runtime_optimizations.py`

- [x] **Step 1: Write the failing tests**

Add tests proving that `estimate_self_ipv(..., candidate_executor=...)` submits one task per IPV candidate, returns the same weighted IPV shape as the serial path, and that `estimate_ipv_current(..., solver_preset="parallel_accurate", max_workers=...)` forwards the executor choice.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ipv_runtime_optimizations.py -q`

Expected: failures because `candidate_executor`, `parallel_accurate`, and `max_workers` are not implemented yet.

- [x] **Step 3: Implement minimal candidate execution hook**

Add a small top-level worker helper in `agent.py` that constructs a temporary `Agent`, solves one candidate IPV trajectory, and returns `(track_xy, ipv_value)`. Keep serial behavior as default when no executor is provided.

- [x] **Step 4: Expose parallel options in wrappers**

Add `parallel_accurate` to `SOLVER_PRESETS`, add optional `max_workers` to `estimate_ipv_pair` and `estimate_ipv_current`, and create a bounded `ProcessPoolExecutor` only around the current estimation call.

- [x] **Step 5: Verify unit tests**

Run: `python -m pytest tests/test_ipv_runtime_optimizations.py -q`

Expected: all tests pass.

### Task 2: Small-Batch Accuracy And Latency Check

**Files:**
- No production files unless Task 1 exposes a bug.
- Output: ignored analysis folder under `interhub_traj_lane/1_ipv_estimation_results/`.

- [x] **Step 1: Run current stratified 6-case benchmark**

Compare original/accurate labels against `parallel_accurate` on the same 6-case sample with `history_window=8`, `min_observation=4`, `reference_max_points=40`, and `reference_smooth_points=40`.

- [x] **Step 2: Check gates**

Required for this small-batch gate: three-class sign accuracy at threshold `0.05` must be at least 90%. Report pair-step latency and speedup. If latency is not realtime, keep optimizing executor lifetime and worker granularity before changing the scientific objective.

- [x] **Step 3: Record workflow log**

Append a summary to `main_workflow.log` with status, accuracy, latency, and artifacts.
