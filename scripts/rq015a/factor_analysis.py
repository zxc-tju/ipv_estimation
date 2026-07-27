#!/usr/bin/env python3
"""RQ015A descriptive factor association for q_eff.

This module is deliberately narrow: it consumes already-built L1 ledger-like
rows, reads only artifact_id, case_id, attempt_status, q_eff, and the requested
numeric factor fields, and returns descriptive Spearman associations. It does
not read data files and does not route any operational decision.

Spearman ties are ranked with the average 1-based rank for each exact-equality
tie group after sorting by numeric value. Pearson correlation is then computed
on those ranks with deterministic sorted + math.fsum reductions.

The case-cluster bootstrap enumerates clusters by sorted case_id, with None
ordered before strings via (case_id is not None, case_id or ""), then samples
cluster indices with random.Random(20260726). The 95% CI uses percentile
positions p*(n-1) with linear interpolation between adjacent sorted bootstrap
statistics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from rq015a_contracts import (
    ATTEMPTED,
    MIN_SUPPORT_L1_PER_L2,
    NOT_ATTEMPTED,
    UNKNOWN,
    ContractViolation,
    assert_single_artifact,
)

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_DEFINED_SUPPORT_FLOOR = 0.90
CI_LOW_QUANTILE = 0.025
CI_HIGH_QUANTILE = 0.975

STATUS_OK = "OK"
STATUS_INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
STATUS_INSUFFICIENT_SUPPORT_BOOTSTRAP_DEGENERATE = "INSUFFICIENT_SUPPORT_BOOTSTRAP_DEGENERATE"
STATUS_UNDEFINED_CONSTANT_INPUT = "UNDEFINED_CONSTANT_INPUT"

_FORBIDDEN_FACTOR_FRAGMENTS = ("rating", "preference", "human", "score", "label")
_ALLOWED_ATTEMPT_STATUSES = (ATTEMPTED, NOT_ATTEMPTED, UNKNOWN)


class L1FactorRow(Protocol):
    artifact_id: str
    case_id: Optional[str]
    attempt_status: str
    q_eff: Optional[float]


@dataclass(frozen=True)
class FactorAssociationResult:
    artifact_id: str
    factor_name: str
    status: str
    n_input_rows: int
    n_attempted_rows: int
    n_excluded_status: int
    n_excluded_nonfinite_q_eff: int
    n_excluded_nonfinite_factor: int
    n_used: int
    n_clusters: int
    spearman_rho: Optional[float]
    ci95_low: Optional[float]
    ci95_high: Optional[float]
    bootstrap_iterations: int
    bootstrap_seed: int
    n_bootstrap_defined: int
    n_bootstrap_undefined: int
    descriptive_only: bool = True


@dataclass(frozen=True)
class _Observation:
    case_id: Optional[str]
    factor_value: float
    q_eff: float


@dataclass(frozen=True)
class _PreparedRows:
    observations: Tuple[_Observation, ...]
    n_input_rows: int
    n_attempted_rows: int
    n_excluded_status: int
    n_excluded_nonfinite_q_eff: int
    n_excluded_nonfinite_factor: int


def _det_mean(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    return math.fsum(vals) / len(vals)


def _get_value(row: object, field_name: str) -> object:
    if isinstance(row, Mapping):
        if field_name not in row:
            raise ContractViolation("L1 row missing field %s" % field_name)
        return row[field_name]
    if not hasattr(row, field_name):
        raise ContractViolation("L1 row missing field %s" % field_name)
    return getattr(row, field_name)


def _finite_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _case_id(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ContractViolation("case_id must be str or None")


def _case_sort_key(case_id: Optional[str]) -> Tuple[bool, str]:
    return (case_id is not None, case_id or "")


def _validate_factor_name(factor_name: str) -> None:
    if not isinstance(factor_name, str) or not factor_name:
        raise ContractViolation("factor_name must be a non-empty string")
    lowered = factor_name.lower()
    for fragment in _FORBIDDEN_FACTOR_FRAGMENTS:
        if fragment in lowered:
            raise ContractViolation("forbidden outcome-like factor field %r" % factor_name)


def _artifact_guard(rows: Sequence[object]) -> str:
    guard_rows = []
    for row in rows:
        guard_rows.append({"artifact_id": _get_value(row, "artifact_id")})
    return assert_single_artifact(guard_rows)


def _prepare_rows(rows: Sequence[object], factor_name: str) -> _PreparedRows:
    observations = []
    n_attempted_rows = 0
    n_excluded_status = 0
    n_excluded_nonfinite_q_eff = 0
    n_excluded_nonfinite_factor = 0

    for row in rows:
        status = _get_value(row, "attempt_status")
        if status not in _ALLOWED_ATTEMPT_STATUSES:
            raise ContractViolation("unknown attempt_status %r" % status)
        if status != ATTEMPTED:
            n_excluded_status += 1
            continue
        n_attempted_rows += 1

        q_value = _finite_float(_get_value(row, "q_eff"))
        if q_value is None:
            n_excluded_nonfinite_q_eff += 1
            continue
        if q_value <= 0.0 or q_value > 1.0:
            raise ContractViolation("q_eff outside (0, 1]")

        factor_value = _finite_float(_get_value(row, factor_name))
        if factor_value is None:
            n_excluded_nonfinite_factor += 1
            continue

        observations.append(_Observation(
            case_id=_case_id(_get_value(row, "case_id")),
            factor_value=factor_value,
            q_eff=q_value,
        ))

    observations.sort(key=lambda obs: (
        _case_sort_key(obs.case_id),
        obs.factor_value,
        obs.q_eff,
    ))
    return _PreparedRows(
        observations=tuple(observations),
        n_input_rows=len(rows),
        n_attempted_rows=n_attempted_rows,
        n_excluded_status=n_excluded_status,
        n_excluded_nonfinite_q_eff=n_excluded_nonfinite_q_eff,
        n_excluded_nonfinite_factor=n_excluded_nonfinite_factor,
    )


def _has_variation(values: Sequence[float]) -> bool:
    if not values:
        return False
    first = values[0]
    for value in values[1:]:
        if value != first:
            return True
    return False


def _average_ranks(values: Sequence[float]) -> List[float]:
    pairs = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(pairs)
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = math.fsum(sorted((float(start + 1), float(end)))) / 2.0
        for pair_index in range(start, end):
            ranks[pairs[pair_index][1]] = average_rank
        start = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    mean_x = _det_mean(xs)
    mean_y = _det_mean(ys)
    if mean_x is None or mean_y is None:
        return None
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    ss_x = math.fsum(sorted(x * x for x in dx))
    ss_y = math.fsum(sorted(y * y for y in dy))
    if ss_x == 0.0 or ss_y == 0.0:
        return None
    numerator = math.fsum(sorted(x * y for x, y in zip(dx, dy)))
    rho = numerator / math.sqrt(ss_x * ss_y)
    if rho > 1.0 and rho <= 1.0 + 1e-15:
        return 1.0
    if rho < -1.0 and rho >= -1.0 - 1e-15:
        return -1.0
    if rho == 0.0:
        return 0.0
    return rho


def spearman_rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return Spearman rank correlation with average ranks for exact ties.

    Ties are identified by exact equality after conversion to float. Each tied
    block receives the arithmetic average of its occupied 1-based ranks, and
    Pearson correlation is computed over the two rank vectors. Constant input
    returns None instead of 0.0 or NaN.
    """
    if len(xs) != len(ys):
        raise ContractViolation("Spearman inputs have different lengths")
    clean_x = []
    clean_y = []
    for x_value, y_value in zip(xs, ys):
        x_float = _finite_float(x_value)
        y_float = _finite_float(y_value)
        if x_float is None or y_float is None:
            raise ContractViolation("Spearman inputs must be finite")
        clean_x.append(x_float)
        clean_y.append(y_float)
    if len(clean_x) < 2:
        return None
    if not _has_variation(clean_x) or not _has_variation(clean_y):
        return None
    return _pearson(_average_ranks(clean_x), _average_ranks(clean_y))


def _cluster_rows(observations: Sequence[_Observation]) -> List[Tuple[_Observation, ...]]:
    groups: Dict[Optional[str], List[_Observation]] = {}
    for obs in observations:
        if obs.case_id not in groups:
            groups[obs.case_id] = []
        groups[obs.case_id].append(obs)

    clusters = []
    for case_id in sorted(groups, key=_case_sort_key):
        cluster_rows = sorted(groups[case_id], key=lambda obs: (obs.factor_value, obs.q_eff))
        clusters.append(tuple(cluster_rows))
    return clusters


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ContractViolation("percentile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ContractViolation("quantile outside [0, 1]")
    position = quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    value = lower + (upper - lower) * fraction
    if value == 0.0:
        return 0.0
    return value


def _bootstrap_ci(
    observations: Sequence[_Observation],
    iterations: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float], int, int]:
    clusters = _cluster_rows(observations)
    rng = random.Random(seed)
    bootstrapped = []
    undefined = 0
    for _ in range(iterations):
        sample = []
        for _ in range(len(clusters)):
            sample.extend(clusters[rng.randrange(len(clusters))])
        rho = spearman_rank_correlation(
            [obs.factor_value for obs in sample],
            [obs.q_eff for obs in sample],
        )
        if rho is None:
            undefined += 1
        else:
            bootstrapped.append(rho)
    bootstrapped.sort()
    if len(bootstrapped) < 2:
        return None, None, len(bootstrapped), undefined
    return (
        _percentile(bootstrapped, CI_LOW_QUANTILE),
        _percentile(bootstrapped, CI_HIGH_QUANTILE),
        len(bootstrapped),
        undefined,
    )


def _result(
    artifact_id: str,
    factor_name: str,
    status: str,
    prepared: _PreparedRows,
    n_clusters: int,
    spearman_rho: Optional[float],
    ci95_low: Optional[float],
    ci95_high: Optional[float],
    n_bootstrap_defined: int,
    n_bootstrap_undefined: int,
) -> FactorAssociationResult:
    return FactorAssociationResult(
        artifact_id=artifact_id,
        factor_name=factor_name,
        status=status,
        n_input_rows=prepared.n_input_rows,
        n_attempted_rows=prepared.n_attempted_rows,
        n_excluded_status=prepared.n_excluded_status,
        n_excluded_nonfinite_q_eff=prepared.n_excluded_nonfinite_q_eff,
        n_excluded_nonfinite_factor=prepared.n_excluded_nonfinite_factor,
        n_used=len(prepared.observations),
        n_clusters=n_clusters,
        spearman_rho=spearman_rho,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
        n_bootstrap_defined=n_bootstrap_defined,
        n_bootstrap_undefined=n_bootstrap_undefined,
    )


def analyze_factor(rows: Iterable[L1FactorRow], factor_name: str) -> FactorAssociationResult:
    """Analyze one candidate factor against q_eff descriptively.

    Inclusion is limited to rows where attempt_status == "ATTEMPTED", q_eff is
    finite and in (0, 1], and the factor value is finite. Non-finite factor
    exclusions are counted explicitly.
    """
    _validate_factor_name(factor_name)
    row_list = list(rows)
    artifact_id = _artifact_guard(row_list)
    prepared = _prepare_rows(row_list, factor_name)
    clusters = _cluster_rows(prepared.observations)

    if len(prepared.observations) < MIN_SUPPORT_L1_PER_L2:
        return _result(
            artifact_id, factor_name, STATUS_INSUFFICIENT_SUPPORT, prepared,
            len(clusters), None, None, None, 0, 0,
        )
    if len(clusters) < 2:
        return _result(
            artifact_id, factor_name, STATUS_INSUFFICIENT_SUPPORT, prepared,
            len(clusters), None, None, None, 0, 0,
        )

    factor_values = [obs.factor_value for obs in prepared.observations]
    q_values = [obs.q_eff for obs in prepared.observations]
    if not _has_variation(factor_values) or not _has_variation(q_values):
        return _result(
            artifact_id, factor_name, STATUS_UNDEFINED_CONSTANT_INPUT, prepared,
            len(clusters), None, None, None, 0, 0,
        )

    rho = spearman_rank_correlation(factor_values, q_values)
    if rho is None:
        return _result(
            artifact_id, factor_name, STATUS_UNDEFINED_CONSTANT_INPUT, prepared,
            len(clusters), None, None, None, 0, 0,
        )

    ci_low, ci_high, n_defined, n_undefined = _bootstrap_ci(
        prepared.observations, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED,
    )
    if n_defined < BOOTSTRAP_DEFINED_SUPPORT_FLOOR * BOOTSTRAP_ITERATIONS:
        return _result(
            artifact_id, factor_name, STATUS_INSUFFICIENT_SUPPORT_BOOTSTRAP_DEGENERATE,
            prepared, len(clusters), None, None, None, n_defined, n_undefined,
        )
    if ci_low is None or ci_high is None:
        return _result(
            artifact_id, factor_name, STATUS_INSUFFICIENT_SUPPORT, prepared,
            len(clusters), None, None, None, n_defined, n_undefined,
        )
    return _result(
        artifact_id, factor_name, STATUS_OK, prepared, len(clusters),
        rho, ci_low, ci_high, n_defined, n_undefined,
    )


def analyze_factors(
    rows: Iterable[L1FactorRow],
    factor_names: Sequence[str],
) -> Tuple[FactorAssociationResult, ...]:
    """Analyze candidate factors in lexicographic order for stable output."""
    row_list = list(rows)
    names = sorted(factor_names)
    if not names:
        raise ContractViolation("at least one factor name is required")
    return tuple(analyze_factor(row_list, factor_name) for factor_name in names)
