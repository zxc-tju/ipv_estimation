"""Canonical raw IPV-envelope deviation calculations."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def raw_envelope_deviation(
    observed_ipv: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed exceedance, absolute exceedance, and outside-band mask.

    Signed deviation is zero inside the interval, negative below the lower
    bound, and positive above the upper bound. Missing/abstained bounds produce
    NaN deviations and a false outside flag.
    """
    observed = np.asarray(observed_ipv, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    observed, lo, hi = np.broadcast_arrays(observed, lo, hi)
    valid = np.isfinite(observed) & np.isfinite(lo) & np.isfinite(hi)
    signed = np.full(observed.shape, np.nan, dtype=float)
    signed[valid] = 0.0
    below = valid & (observed < lo)
    above = valid & (observed > hi)
    signed[below] = observed[below] - lo[below]
    signed[above] = observed[above] - hi[above]
    absolute = np.abs(signed)
    outside = below | above
    return signed, absolute, outside
