#!/usr/bin/env python3
"""Verify exact IPV output against the frozen sigma01 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sociality_estimation.core.ipv_estimation import MotionSequence, estimate_ipv_pair


DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "ipv_estimator_parity_fixture.json"
)


def _sequence(motion: list, target: list, reference: dict) -> MotionSequence:
    return MotionSequence(
        np.asarray(motion, dtype=float),
        target=target,
        reference=(
            np.asarray(reference["cv"], dtype=float),
            np.asarray(reference["s"], dtype=float),
        ),
    )


def _max_abs(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def verify(fixture_path: Path) -> dict[str, float]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    sigma01_max_abs_diff = 0.0
    fast_exact_max_abs_diff = 0.0

    for case in fixture["cases"]:
        primary = _sequence(
            case["primary_motion"], case["targets"][0], case["primary_reference"]
        )
        counterpart = _sequence(
            case["counterpart_motion"],
            case["targets"][1],
            case["counterpart_reference"],
        )
        kwargs = {
            "history_window": int(fixture["history_window"]),
            "min_observation": int(fixture["min_observation"]),
        }
        exact_ipv, exact_error = estimate_ipv_pair(
            primary, counterpart, solver_mode="exact", **kwargs
        )
        fast_ipv, fast_error = estimate_ipv_pair(
            primary, counterpart, solver_mode="fast", **kwargs
        )
        sigma01_max_abs_diff = max(
            sigma01_max_abs_diff,
            _max_abs(exact_ipv, np.asarray(case["expected_ipv"], dtype=float)),
            _max_abs(exact_error, np.asarray(case["expected_error"], dtype=float)),
        )
        fast_exact_max_abs_diff = max(
            fast_exact_max_abs_diff,
            _max_abs(fast_ipv, exact_ipv),
            _max_abs(fast_error, exact_error),
        )

    return {
        "sigma01_max_abs_diff": sigma01_max_abs_diff,
        "fast_exact_max_abs_diff": fast_exact_max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--sigma-tolerance", type=float, default=1e-12)
    parser.add_argument("--fast-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    result = verify(args.fixture)
    print(json.dumps(result, sort_keys=True))
    if result["sigma01_max_abs_diff"] > args.sigma_tolerance:
        raise SystemExit("sigma01 strict parity failed")
    if result["fast_exact_max_abs_diff"] > args.fast_tolerance:
        raise SystemExit("fast/exact parity failed")


if __name__ == "__main__":
    main()
