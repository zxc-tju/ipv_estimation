#!/usr/bin/env python3
"""Dependency-free Spearman correlation with deterministic average midranks."""
from __future__ import annotations

import math
from typing import Sequence


def average_midranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks with exact-value tie groups."""

    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Spearman inputs must be finite")
    order = sorted(range(len(numeric)), key=lambda index: (numeric[index], index))
    ranks = [0.0] * len(numeric)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and numeric[order[stop]] == numeric[order[start]]:
            stop += 1
        midrank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = midrank
        start = stop
    return ranks


def spearman_average_midranks(x: Sequence[float], y: Sequence[float]) -> float:
    """Return Pearson correlation of deterministic average-midranks."""

    if len(x) != len(y):
        raise ValueError("Spearman vectors must have equal length")
    if len(x) < 3:
        raise ValueError("Spearman vectors require at least three rows")
    numeric_x = [float(value) for value in x]
    numeric_y = [float(value) for value in y]
    if len(set(zip(numeric_x, numeric_y))) < 3:
        raise ValueError("Spearman vectors require at least three distinct pairs")
    rank_x = average_midranks(numeric_x)
    rank_y = average_midranks(numeric_y)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    centered_x = [value - mean_x for value in rank_x]
    centered_y = [value - mean_y for value in rank_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        raise ValueError("Spearman vectors must both be nonconstant")
    result = sum(a * b for a, b in zip(centered_x, centered_y)) / denominator
    if not math.isfinite(result):
        raise ValueError("Spearman result must be finite")
    return result
