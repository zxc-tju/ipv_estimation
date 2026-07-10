"""Frozen RQ009 M3 verifier runtime definitions.

This module contains only the inference-time model and support-gate logic.
Training, tuning, and report generation remain in their research run archive.
Keeping these classes under a stable package path makes serialized scorers
portable across Git checkouts and machines.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


QUANTILE_LEVELS: Tuple[float, ...] = (0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975)
ALPHAS: Tuple[float, ...] = (0.20, 0.10, 0.05)
ALPHA_LABEL = {0.20: "80", 0.10: "90", 0.05: "95"}
QUANTILE_BY_ALPHA = {
    0.20: (0.10, 0.90),
    0.10: (0.05, 0.95),
    0.05: (0.025, 0.975),
}
Q_INDEX = {quantile: index for index, quantile in enumerate(QUANTILE_LEVELS)}
INTERVAL_EPS = 1e-10
GATE_K = 25


@dataclass
class FeatureSpec:
    name: str
    numeric: List[str]
    categorical: List[str]
    zero_numeric: List[str] = field(default_factory=list)
    channel_slot_names: List[str] = field(default_factory=list)


@dataclass
class Preprocessor:
    numeric: List[str]
    categorical: List[str]
    zero_numeric: List[str] = field(default_factory=list)
    imputer: Optional[SimpleImputer] = None
    encoder: Optional[OrdinalEncoder] = None
    categorical_mask: Optional[List[bool]] = None

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("Preprocessor not fitted")
        numeric_values = self.imputer.transform(
            frame[self.numeric].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32, copy=False)
        for column in self.zero_numeric:
            if column in self.numeric:
                numeric_values[:, self.numeric.index(column)] = 0.0
        if not self.categorical:
            return numeric_values
        if self.encoder is None:
            raise RuntimeError("Categorical encoder not fitted")
        categorical_values = self.encoder.transform(
            categorical_frame(frame, self.categorical)
        ).astype(np.float32, copy=False)
        return np.hstack([numeric_values, categorical_values]).astype(
            np.float32, copy=False
        )


@dataclass
class TierModel:
    spec: FeatureSpec
    preprocessor: Preprocessor
    models: Dict[float, HistGradientBoostingRegressor]


@dataclass
class GateModel:
    numeric: List[str]
    categorical: List[str]
    support_columns: List[str]
    joint_cell_columns: List[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    encoder: OneHotEncoder
    tree: cKDTree
    threshold: float
    train_rows: int
    train_reference_rows: int
    guard_rows: int
    guard_eligible_rows: int
    support_levels: Dict[str, Dict[str, Dict[str, int]]]
    unsupported_levels: Dict[str, List[str]]
    supported_joint_cells: Sequence[str]
    unsupported_joint_cells: List[str]


def categorical_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, columns].astype("string").fillna("__MISSING__")


def predict_tier_quantiles(
    tier_model: TierModel,
    frame: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Predict and monotonically rearrange the seven frozen M3 quantiles."""
    x = tier_model.preprocessor.transform(frame)
    raw = np.column_stack(
        [tier_model.models[quantile].predict(x) for quantile in QUANTILE_LEVELS]
    ).astype(np.float32)
    rearranged = np.sort(raw, axis=1).astype(np.float32)
    changed = np.abs(raw - rearranged) > 1e-7
    health = {
        "rows": int(len(frame)),
        "changed_by_quantile": {
            str(quantile): int(changed[:, index].sum())
            for index, quantile in enumerate(QUANTILE_LEVELS)
        },
        "changed_fraction_by_quantile": {
            str(quantile): float(changed[:, index].mean()) if len(frame) else 0.0
            for index, quantile in enumerate(QUANTILE_LEVELS)
        },
        "any_crossing_before_rearrangement": (
            bool(np.any(np.diff(raw, axis=1) < -1e-7)) if len(frame) else False
        ),
    }
    del x, raw
    gc.collect()
    return rearranged, changed, health


def joint_cell_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    cat = categorical_frame(frame, columns)
    values = cat.iloc[:, 0].astype(str)
    for column in columns[1:]:
        values = values + "|" + cat[column].astype(str)
    return values


def category_support_mask(frame: pd.DataFrame, gate: GateModel) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in gate.support_columns:
        values = frame[column].astype("string").fillna("__MISSING__").to_numpy()
        supported = set(gate.support_levels[column].keys())
        mask &= np.fromiter(
            (value in supported for value in values), dtype=bool, count=len(values)
        )
    cells = joint_cell_series(frame, gate.joint_cell_columns).to_numpy()
    mask &= np.fromiter(
        (value in gate.supported_joint_cells for value in cells),
        dtype=bool,
        count=len(cells),
    )
    return mask


def transform_gate_matrix(frame: pd.DataFrame, gate: GateModel) -> np.ndarray:
    numeric = gate.imputer.transform(frame[gate.numeric]).astype(np.float32)
    numeric = gate.scaler.transform(numeric).astype(np.float32)
    categorical = gate.encoder.transform(categorical_frame(frame, gate.categorical))
    return np.hstack([numeric, categorical]).astype(np.float32)


def mean_knn_distance(tree: cKDTree, x: np.ndarray, k: int = GATE_K) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    distances, _ = tree.query(x, k=k, workers=-1)
    if distances.ndim == 1:
        return distances.astype(np.float32)
    return distances.mean(axis=1).astype(np.float32)


def apply_gate(
    frame: pd.DataFrame,
    gate: GateModel,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the frozen categorical-support and kNN OOD abstention gate."""
    category_ok = category_support_mask(frame, gate)
    distance_ok = np.zeros(len(frame), dtype=bool)
    distances = np.full(len(frame), np.nan, dtype=np.float32)
    if category_ok.any():
        x = transform_gate_matrix(frame, gate)
        eligible_distances = mean_knn_distance(gate.tree, x[category_ok])
        distances[category_ok] = eligible_distances
        distance_ok[category_ok] = eligible_distances <= gate.threshold
        del x, eligible_distances
        gc.collect()
    gate_ok = category_ok & distance_ok
    diagnostics = {
        "rows": int(len(frame)),
        "category_pass_rows": int(category_ok.sum()),
        "distance_pass_rows": int(distance_ok.sum()),
        "gate_pass_rows": int(gate_ok.sum()),
        "abstain_rows": int((~gate_ok).sum()),
        "abstention_rate": float(1.0 - gate_ok.mean()) if len(frame) else 0.0,
        "distance_mean": (
            float(np.nanmean(distances)) if np.isfinite(distances).any() else None
        ),
        "distance_p95": (
            float(np.nanquantile(distances, 0.95))
            if np.isfinite(distances).any()
            else None
        ),
    }
    return gate_ok, diagnostics


def calibrated_bounds(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    c_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_lo = q_lo.astype(np.float64, copy=False) - float(c_alpha)
    raw_hi = q_hi.astype(np.float64, copy=False) + float(c_alpha)
    lo_cal = np.minimum(raw_lo, raw_hi) - INTERVAL_EPS
    hi_cal = np.maximum(raw_lo, raw_hi) + INTERVAL_EPS
    return lo_cal, hi_cal
