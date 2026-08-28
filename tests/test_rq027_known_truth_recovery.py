from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.simulation.rq027_independent_generator import (
    COUNTERPART_IPV_LEVELS,
    NEGATIVE_CONTROL_TYPES,
    N_STEPS,
    build_interactive_cases,
    build_negative_control_cases,
    generate_run,
)
from pipelines.simulation.run_rq027_pilot import (
    _logdomain_measurement,
    run_pilot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generator_import_contract_is_stdlib_plus_numpy() -> None:
    source_path = REPO_ROOT / "pipelines/simulation/rq027_independent_generator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {"__future__", "dataclasses", "math", "typing", "numpy"}
    assert not ({"sociality_estimation", "scipy"} & imported_roots)


def test_case_matrices_are_frozen_and_unique() -> None:
    interactive = build_interactive_cases()
    negative = build_negative_control_cases()
    all_cases = interactive + negative
    assert len(interactive) == 240
    assert len(negative) == 48
    assert len({case["run_id"] for case in all_cases}) == 288
    assert {case["negative_control_type"] for case in negative} == set(
        NEGATIVE_CONTROL_TYPES
    )
    counts = pd.Series([case["negative_control_type"] for case in negative]).value_counts()
    assert set(counts.tolist()) == {12}

    paired = [
        case
        for case in interactive
        if case["template"] == "ambiguous_priority_crossing"
        and case["counterpart_ipv_rad"] == COUNTERPART_IPV_LEVELS[0]
        and case["intensity"] == "strong"
        and case["seed_index"] == 0
    ]
    assert len(paired) == 5
    assert len({case["seed"] for case in paired}) == 1


def test_generator_is_deterministic_responsive_and_finite() -> None:
    cases = [
        case
        for case in build_interactive_cases()
        if case["template"] == "ambiguous_priority_crossing"
        and case["counterpart_ipv_rad"] == COUNTERPART_IPV_LEVELS[0]
        and case["intensity"] == "strong"
        and case["seed_index"] == 0
    ]
    cases.sort(key=lambda item: item["true_ipv_rad"])
    generated = [generate_run(case) for case in cases]
    repeated = generate_run(cases[0])
    assert np.array_equal(generated[0].target_motion, repeated.target_motion)
    assert np.array_equal(
        generated[0].oracle_informativeness, repeated.oracle_informativeness
    )
    assert generated[0].target_motion.shape == (N_STEPS, 5)
    assert generated[0].counterpart_motion.shape == (N_STEPS, 5)
    assert all(np.all(np.isfinite(run.target_motion)) for run in generated)
    assert all(np.all(np.isfinite(run.counterpart_motion)) for run in generated)
    maximum_pairwise_change = max(
        np.linalg.norm(left.target_motion - right.target_motion)
        for left in generated
        for right in generated
    )
    assert maximum_pairwise_change > 0.5
    assert sum(float(np.max(run.oracle_informativeness)) > 0 for run in generated) >= 4


def test_negative_control_geometry_and_collision_health() -> None:
    cases = build_negative_control_cases()
    selected = {
        control: next(
            case for case in cases if case["negative_control_type"] == control
        )
        for control in NEGATIVE_CONTROL_TYPES
    }
    generated = {name: generate_run(case) for name, case in selected.items()}
    assert not generated["no_conflict_neighbour"].opportunity_mask.any()
    assert not generated["wrong_run_pseudo_pair"].opportunity_mask.any()
    assert not generated["post_resolution_window"].opportunity_mask.any()
    assert all(not run.collision_mask.any() for run in generated.values())


def test_logdomain_measurement_recomputes_stable_primary_weights() -> None:
    observed = np.zeros((5, 2), dtype=float)
    virtual = [np.full((5, 2), float(index) * 0.05) for index in range(7)]
    diagnostic = {
        "observed": observed,
        "virtual_tracks": virtual,
        "ipv_range": np.arange(-3, 4, dtype=float) * np.pi / 8.0,
    }
    result = _logdomain_measurement(diagnostic)
    assert np.isclose(np.sum(result["weights"]), 1.0)
    assert result["weights"][0] == np.max(result["weights"])
    assert result["mse_spread"] > 0
    assert 1 / 7 <= result["q_eff"] <= 1


def test_exact_smoke_writes_complete_artifact_set(tmp_path: Path) -> None:
    summary = run_pilot(
        tmp_path,
        limit_runs=1,
        solver_mode="exact",
        workers=1,
    )
    assert summary["verdict"] == "SMOKE_ONLY"
    expected = {
        "frame_level_results.csv",
        "frame_level_results.parquet",
        "run_level_results.csv",
        "summary.json",
        "numerical_health.json",
        "evidence_summary.csv",
        "REPORT.md",
        "run_receipt.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    runs = pd.read_csv(tmp_path / "run_level_results.csv")
    frames = pd.read_csv(tmp_path / "frame_level_results.csv")
    assert len(runs) == 1
    assert len(frames) == N_STEPS - 4
    assert runs["run_id"].is_unique
    assert np.isfinite(frames[["ipv_log", "max_weight", "q_eff"]]).all().all()
