"""Fixtures for RQ015A descriptive factor association.

All rows are synthetic. scipy is used only as a test oracle.
"""

from __future__ import annotations

import ast
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

from factor_analysis import (  # noqa: E402
    STATUS_INSUFFICIENT_SUPPORT,
    STATUS_OK,
    STATUS_UNDEFINED_CONSTANT_INPUT,
    analyze_factor,
    spearman_rank_correlation,
)
from rq015a_contracts import ATTEMPTED, NOT_ATTEMPTED, UNKNOWN, ContractViolation  # noqa: E402


FACTOR_PATH = ROOT / "scripts" / "rq015a" / "factor_analysis.py"


def _row(case_id, status, q_eff, factor, artifact_id="synthetic_artifact"):
    return SimpleNamespace(
        artifact_id=artifact_id,
        case_id=case_id,
        attempt_status=status,
        q_eff=q_eff,
        factor=factor,
    )


def _base_rows():
    return [
        _row("c1", ATTEMPTED, 0.20, 5.0),
        _row("c1", ATTEMPTED, 0.35, 5.0),
        _row("c2", ATTEMPTED, 0.30, 3.0),
        _row("c2", ATTEMPTED, 0.62, 4.0),
        _row("c3", ATTEMPTED, 0.62, 4.0),
        _row("c3", ATTEMPTED, 0.71, 2.0),
        _row("c4", ATTEMPTED, 0.80, 1.0),
        _row("c5", ATTEMPTED, 0.92, 1.0),
    ]


def test_input_order_is_bitwise_deterministic_for_five_permutations():
    rows = _base_rows()
    base = analyze_factor(rows, "factor")
    assert base.status == STATUS_OK
    for seed in range(5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        out = analyze_factor(shuffled, "factor")
        assert out.spearman_rho == base.spearman_rho
        assert out.ci95_low == base.ci95_low
        assert out.ci95_high == base.ci95_high


def test_spearman_matches_scipy_with_and_without_ties():
    cases = [
        ([1, 2, 3, 4, 5, 6], [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]),
        ([1, 1, 2, 3, 5, 8], [0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
        ([1, 2, 2, 2, 4, 5, 5], [0.2, 0.2, 0.5, 0.5, 0.6, 0.9, 0.9]),
        ([3, 1, 3, 2, 4, 4, 5, 6], [0.4, 0.8, 0.4, 0.7, 0.1, 0.3, 0.2, 0.2]),
    ]
    deltas = []
    for xs, ys in cases:
        ours = spearman_rank_correlation(xs, ys)
        expected = float(stats.spearmanr(xs, ys).statistic)
        deltas.append(abs(ours - expected))
    assert max(deltas) <= 1e-12


def test_bootstrap_repeats_exactly_with_frozen_seed():
    first = analyze_factor(_base_rows(), "factor")
    second = analyze_factor(_base_rows(), "factor")
    assert first.status == second.status == STATUS_OK
    assert first.ci95_low == second.ci95_low
    assert first.ci95_high == second.ci95_high
    assert first.n_bootstrap_defined == second.n_bootstrap_defined


def test_case_cluster_bootstrap_honors_cluster_row_multiplicity():
    base = [
        _row("c1", ATTEMPTED, 0.10, 9.0),
        _row("c2", ATTEMPTED, 0.25, 2.0),
        _row("c3", ATTEMPTED, 0.40, 4.0),
        _row("c4", ATTEMPTED, 0.55, 6.0),
        _row("c5", ATTEMPTED, 0.70, 8.0),
        _row("c6", ATTEMPTED, 0.85, 10.0),
    ]
    inflated = list(base) + [
        _row("c1", ATTEMPTED, 0.15, 1.0),
        _row("c1", ATTEMPTED, 0.18, 1.5),
        _row("c1", ATTEMPTED, 0.22, 2.0),
        _row("c1", ATTEMPTED, 0.26, 2.5),
    ]
    base_out = analyze_factor(base, "factor")
    inflated_out = analyze_factor(inflated, "factor")
    base_width = base_out.ci95_high - base_out.ci95_low
    inflated_width = inflated_out.ci95_high - inflated_out.ci95_low
    assert inflated_width != base_width


def test_cross_artifact_pooling_raises_contract_violation():
    rows = _base_rows()
    rows.append(_row("c6", ATTEMPTED, 0.44, 2.0, artifact_id="other_artifact"))
    with pytest.raises(ContractViolation):
        analyze_factor(rows, "factor")


def test_effective_rows_below_min_support_returns_marker_without_numbers():
    rows = [_row("c%d" % i, ATTEMPTED, 0.2 + i * 0.1, float(i)) for i in range(4)]
    out = analyze_factor(rows, "factor")
    assert out.status == STATUS_INSUFFICIENT_SUPPORT
    assert out.spearman_rho is None and out.ci95_low is None and out.ci95_high is None


def test_constant_factor_or_q_eff_returns_undefined_marker_without_numbers():
    factor_constant = [_row("c%d" % i, ATTEMPTED, 0.1 + i * 0.1, 1.0) for i in range(5)]
    q_constant = [_row("c%d" % i, ATTEMPTED, 0.5, float(i)) for i in range(5)]
    for rows in (factor_constant, q_constant):
        out = analyze_factor(rows, "factor")
        assert out.status == STATUS_UNDEFINED_CONSTANT_INPUT
        assert out.spearman_rho is None and out.ci95_low is None and out.ci95_high is None


def test_nonfinite_factor_values_are_excluded_and_counted():
    rows = _base_rows() + [
        _row("c6", ATTEMPTED, 0.51, math.nan),
        _row("c7", ATTEMPTED, 0.52, math.inf),
    ]
    out = analyze_factor(rows, "factor")
    assert out.status == STATUS_OK
    assert out.n_excluded_nonfinite_factor == 2
    assert out.n_used == len(_base_rows())


def test_only_attempted_rows_with_finite_q_eff_participate():
    attempted = _base_rows()
    noisy = attempted + [
        _row("c9", NOT_ATTEMPTED, 0.01, 1000.0),
        _row("c10", UNKNOWN, 0.99, -1000.0),
        _row("c11", ATTEMPTED, None, 5.0),
        _row("c12", ATTEMPTED, math.nan, 5.0),
    ]
    base = analyze_factor(attempted, "factor")
    out = analyze_factor(noisy, "factor")
    assert out.spearman_rho == base.spearman_rho
    assert out.ci95_low == base.ci95_low
    assert out.ci95_high == base.ci95_high
    assert out.n_excluded_status == 2
    assert out.n_excluded_nonfinite_q_eff == 2


def test_unknown_attempt_status_fails_closed():
    rows = _base_rows() + [_row("bad", "BOGUS", 0.4, 2.0)]
    with pytest.raises(ContractViolation):
        analyze_factor(rows, "factor")


def test_none_case_id_sorts_before_strings_and_is_deterministic():
    rows = [
        _row(None, ATTEMPTED, 0.20, 6.0),
        _row("b", ATTEMPTED, 0.30, 5.0),
        _row("a", ATTEMPTED, 0.40, 4.0),
        _row(None, ATTEMPTED, 0.60, 3.0),
        _row("b", ATTEMPTED, 0.80, 2.0),
        _row("a", ATTEMPTED, 0.90, 1.0),
    ]
    base = analyze_factor(rows, "factor")
    assert base.status == STATUS_OK
    assert base.n_clusters == 3
    for seed in range(5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        assert analyze_factor(shuffled, "factor") == base


def test_result_has_no_decision_routing_surface():
    out = analyze_factor(_base_rows(), "factor")
    payload = asdict(out)
    serialized = json.dumps(payload, sort_keys=True)
    assert "c0_route" not in serialized
    assert "machine_verdict" not in serialized
    assert "verdict" not in payload


def test_production_code_does_not_import_scipy_or_neighbor_modules():
    tree = ast.parse(FACTOR_PATH.read_text())
    forbidden_modules = {"scipy", "rq015a_types", "build_ledger"}
    forbidden_names = {"c0_route", "c0_route_with_sensitivity"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_modules
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] not in forbidden_modules
            for alias in node.names:
                assert alias.name not in forbidden_names
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_names
