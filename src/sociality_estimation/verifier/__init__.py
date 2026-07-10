"""Portable IPV human-envelope verifier."""

from sociality_estimation.verifier.anchors import build_m3_anchor_features
from sociality_estimation.verifier.deviation import raw_envelope_deviation
from sociality_estimation.verifier.features import (
    apet_constant_velocity_proxy,
    closing_ttc,
    relative_state,
    theil_sen_slope,
    wrap_angle,
)


def load_scorer(*args, **kwargs):
    from sociality_estimation.verifier.scorer import load_scorer as implementation

    return implementation(*args, **kwargs)


def score_anchors(*args, **kwargs):
    from sociality_estimation.verifier.scorer import score_anchors as implementation

    return implementation(*args, **kwargs)


def score_verifier(*args, **kwargs):
    from sociality_estimation.verifier.scorer import score_verifier as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "apet_constant_velocity_proxy",
    "build_m3_anchor_features",
    "closing_ttc",
    "load_scorer",
    "raw_envelope_deviation",
    "relative_state",
    "score_anchors",
    "score_verifier",
    "theil_sen_slope",
    "wrap_angle",
]
