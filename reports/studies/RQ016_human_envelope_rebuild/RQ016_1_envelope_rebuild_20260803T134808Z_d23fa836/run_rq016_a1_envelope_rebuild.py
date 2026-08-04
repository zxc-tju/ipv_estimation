#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ016-A1 envelope rebuild on the K2-covered development+guard domain."""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "RQ009_1_dynamic_envelope_20260625T121905Z_98c433de"
RQ009_RUN_ROOT = (
    REPO_ROOT
    / "reports"
    / "studies"
    / "RQ009_dynamic_counterpart_conditioned_envelope"
    / RUN_ID
)
FEATURE_DICTIONARY_PATH = RQ009_RUN_ROOT / "02_process" / "03_features" / "feature_dictionary.csv"
HYPERPARAMETER_PATH = RQ009_RUN_ROOT / "02_process" / "04_calibration" / "hyperparameter_tuning.json"
METRICS_SUMMARY_PATH = RQ009_RUN_ROOT / "02_process" / "05_evaluation" / "metrics_summary.csv"
RQ009_DECISION_PATH = REPO_ROOT / "reports" / "knowledge" / "RQ009_dynamic_counterpart_conditioned_envelope" / "decision.md"
MATRIX_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "interhub"
    / "RQ009_dynamic_counterpart_conditioned_envelope"
    / RUN_ID
    / "03_features"
    / "matrix"
)
K2_LEDGER_ROOT = REPO_ROOT / "data" / "derived" / "rq015k_logdomain_gate" / "l1_v1"
WORK_DIR = REPO_ROOT / ".codex-fleet" / "rq016-envelope-rebuild" / "work" / "A1"
REPORT_PATH = REPO_ROOT / ".codex-fleet" / "rq016-envelope-rebuild" / "board" / "reports" / "RQ016_1_envelope_rebuild.md"
KEY_NUMBERS_PATH = WORK_DIR / "key_numbers.json"

KEY_COLUMNS = ["case_key", "anchor_frame_index", "perspective", "source_dataset"]
TARGET_COLUMN = "target_ipv_future"
FOLD_COLUMN = "fold"
FOLDS = ["train", "guard_tune", "calibration", "test"]

QUANTILE_LEVELS: Tuple[float, ...] = (0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975)
ALPHAS: Tuple[float, ...] = (0.20, 0.10, 0.05)
NOMINAL_BY_ALPHA = {0.20: 0.80, 0.10: 0.90, 0.05: 0.95}
ALPHA_LABEL = {0.20: "80", 0.10: "90", 0.05: "95"}
QUANTILE_BY_ALPHA = {0.20: (0.10, 0.90), 0.10: (0.05, 0.95), 0.05: (0.025, 0.975)}
Q_INDEX = {q: i for i, q in enumerate(QUANTILE_LEVELS)}
INTERVAL_EPS = 1e-10

M2_NUMERIC_CONTEXT = [
    "elapsed_time_s",
    "history_row_count",
    "ego_vx_anchor",
    "ego_vy_anchor",
    "ego_heading_anchor",
    "counterpart_vx_anchor",
    "counterpart_vy_anchor",
    "counterpart_heading_anchor",
    "relative_dx_anchor",
    "relative_dy_anchor",
    "relative_distance_anchor",
    "relative_dvx_anchor",
    "relative_dvy_anchor",
    "relative_speed_anchor",
    "closing_rate_anchor",
    "heading_difference_anchor",
    "relative_distance_mean_wx",
    "relative_distance_std_wx",
    "relative_speed_mean_wx",
    "closing_rate_mean_wx",
    "closing_ttc_anchor",
    "apet_online_proxy",
]
M2_CATEGORICAL_CONTEXT = [
    "geometry_path_category",
    "geometry_path_relation",
    "turn_pair_label",
    "agent_type_pair",
    "vehicle_type_list",
    "av_included",
    "priority_role",
]
IPV_CONDITIONING_COLUMNS = [
    "counterpart_ipv_current",
    "counterpart_ipv_error_current",
    "counterpart_ipv_slope_pre_anchor",
    "M4_ONLY_ego_self_anchor_ipv_current",
    "M4_ONLY_ego_self_anchor_ipv_error_current",
]

GATE_DISTANCE_NUMERIC = [
    "elapsed_time_s",
    "history_row_count",
    "relative_distance_anchor",
    "relative_speed_anchor",
    "closing_rate_anchor",
    "heading_difference_anchor",
    "relative_distance_mean_wx",
    "relative_distance_std_wx",
    "relative_speed_mean_wx",
    "closing_rate_mean_wx",
    "closing_ttc_anchor",
    "apet_online_proxy",
]
GATE_SUPPORT_CATEGORICAL = ["geometry_path_category", "priority_role", "agent_type_pair"]
GATE_JOINT_CELL = GATE_SUPPORT_CATEGORICAL
MIN_SUPPORT_L1_PER_L2 = 5


@dataclass
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]


@dataclass
class Preprocessor:
    numeric: List[str]
    categorical: List[str]
    imputer: SimpleImputer | None = None
    encoder: OrdinalEncoder | None = None
    categorical_mask: List[bool] | None = None

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(numeric_frame(frame, self.numeric))
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            dtype=np.float32,
        )
        self.encoder.fit(categorical_frame(frame, self.categorical))
        self.categorical_mask = [False] * len(self.numeric) + [True] * len(self.categorical)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.encoder is None:
            raise RuntimeError("preprocessor_not_fitted")
        numeric = self.imputer.transform(numeric_frame(frame, self.numeric)).astype(np.float32, copy=False)
        categorical = self.encoder.transform(categorical_frame(frame, self.categorical)).astype(np.float32, copy=False)
        return np.hstack([numeric, categorical]).astype(np.float32, copy=False)


@dataclass
class TierModel:
    spec: FeatureSpec
    preprocessor: Preprocessor
    models: Dict[float, HistGradientBoostingRegressor]


@dataclass
class GateModel:
    imputer: SimpleImputer
    scaler: StandardScaler
    encoder: OneHotEncoder
    tree: cKDTree
    threshold: float
    support_levels: Dict[str, Dict[str, Dict[str, int]]]
    unsupported_levels: Dict[str, List[str]]
    supported_joint_cells: set[str]
    unsupported_joint_cells: List[str]


def utc_now_from_date() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def pct_text(numerator: int, denominator: int, digits: int = 4) -> str:
    return f"{pct(numerator, denominator) * 100:.{digits}f}% ({numerator:,}/{denominator:,})"


def categorical_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, columns].astype("string").fillna("__MISSING__")


def numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def product_row_key(frame: pd.DataFrame) -> pd.Series:
    return (
        "case_key="
        + frame["case_key"].astype("string")
        + "|anchor_frame_index="
        + frame["anchor_frame_index"].astype("int64").astype("string")
        + "|perspective="
        + frame["perspective"].astype("string")
        + "|source_dataset="
        + frame["source_dataset"].astype("string")
    )


def read_feature_dictionary() -> Dict[str, Any]:
    dictionary = pd.read_csv(FEATURE_DICTIONARY_PATH)
    role_col = "role(feature|target|M4_only|key|fold)"
    features = dictionary.loc[dictionary[role_col] == "feature", "name"].astype(str).tolist()
    source_row = dictionary.loc[dictionary["name"] == "source_dataset"].iloc[0].to_dict()
    return {
        "path": str(FEATURE_DICTIONARY_PATH.relative_to(REPO_ROOT)),
        "role_feature_count": int(len(features)),
        "role_feature_columns": features,
        "source_dataset_role": str(source_row[role_col]),
        "source_dataset_definition": str(source_row["definition"]),
    }


def load_selected_hgb_params() -> Dict[str, Any]:
    payload = json.loads(HYPERPARAMETER_PATH.read_text(encoding="utf-8"))
    selected = dict(payload["selected"])
    return {
        "max_iter": int(selected["max_iter"]),
        "learning_rate": float(selected["learning_rate"]),
        "max_leaf_nodes": int(selected["max_leaf_nodes"]),
        "min_samples_leaf": int(selected["min_samples_leaf"]),
        "l2_regularization": float(selected["l2_regularization"]),
        "max_bins": int(selected["max_bins"]),
        "early_stopping": bool(selected["early_stopping"]),
    }


def load_k2_target_ledger() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset = ds.dataset(K2_LEDGER_ROOT, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=[
            "artifact_id",
            "product_row_key",
            "canonical_key",
            "measurement_role",
            "status",
            "reason_code",
            "rq007_split",
            "context_cell_key",
            "gate_applicable",
        ],
        filter=(ds.field("artifact_id") == "rq009_feature_matrix")
        & (ds.field("measurement_role") == "target_future"),
    )
    frame = table.to_pandas()
    for column in ["product_row_key", "canonical_key", "status", "reason_code", "rq007_split", "context_cell_key"]:
        frame[column] = frame[column].astype("string")
    duplicates = int(frame.duplicated("product_row_key").sum())
    invalid_split_rows = int((~frame["rq007_split"].isin(["development", "guard"])).sum())
    status_counts = frame["status"].value_counts(dropna=False).to_dict()
    reason_counts = frame["reason_code"].fillna("__NONE__").value_counts(dropna=False).to_dict()
    diag = {
        "source_path": str(K2_LEDGER_ROOT.relative_to(REPO_ROOT)),
        "rows": int(len(frame)),
        "product_row_key_duplicates": duplicates,
        "canonical_key_duplicates": int(frame.duplicated("canonical_key").sum()),
        "invalid_rq007_split_rows": invalid_split_rows,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "reason_code_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "gate_applicable_false_rows": int((frame["gate_applicable"] != True).sum()),
    }
    return frame.drop(columns=["artifact_id", "measurement_role"]), diag


def matrix_columns() -> List[str]:
    columns = set(KEY_COLUMNS + [FOLD_COLUMN, TARGET_COLUMN])
    columns.update(M2_NUMERIC_CONTEXT)
    columns.update(M2_CATEGORICAL_CONTEXT)
    columns.update(GATE_DISTANCE_NUMERIC)
    columns.update(GATE_SUPPORT_CATEGORICAL)
    return sorted(columns)


def load_joined_folds(ledger: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    dataset = ds.dataset(MATRIX_ROOT, format="parquet", partitioning="hive")
    columns = matrix_columns()
    folds: Dict[str, pd.DataFrame] = {}
    matrix_total_rows = 0
    matrix_key_duplicates = 0
    joined_rows = 0
    fold_counts: Dict[str, Dict[str, int]] = {}
    ledger_lookup = ledger.set_index("product_row_key", drop=False)
    for fold in FOLDS:
        table = dataset.to_table(columns=columns, filter=ds.field("fold") == fold)
        matrix = table.to_pandas()
        matrix_total_rows += int(len(matrix))
        matrix["anchor_frame_index"] = matrix["anchor_frame_index"].astype("int64")
        matrix["product_row_key"] = product_row_key(matrix)
        matrix_key_duplicates += int(matrix.duplicated("product_row_key").sum())
        joined = matrix.merge(
            ledger_lookup[["status", "reason_code", "rq007_split", "context_cell_key"]],
            left_on="product_row_key",
            right_index=True,
            how="inner",
            validate="one_to_one",
        )
        joined_rows += int(len(joined))
        for column in KEY_COLUMNS + [FOLD_COLUMN, "status", "reason_code", "rq007_split", "context_cell_key"]:
            joined[column] = joined[column].astype("string")
        folds[fold] = joined.reset_index(drop=True)
        fold_counts[fold] = {
            "matrix_rows_before_k2_filter": int(len(matrix)),
            "joined_dev_guard_rows": int(len(joined)),
            "status_ok_rows": int((joined["status"] == "OK").sum()),
            "status_not_ok_rows": int((joined["status"] != "OK").sum()),
        }
        del table, matrix, joined
        gc.collect()
    diag = {
        "matrix_source_path": str(MATRIX_ROOT.relative_to(REPO_ROOT)),
        "matrix_rows_read_all_folds": matrix_total_rows,
        "matrix_product_row_key_duplicates": matrix_key_duplicates,
        "joined_rows": joined_rows,
        "ledger_rows": int(len(ledger)),
        "ledger_left_join_hits": joined_rows,
        "ledger_left_join_misses": int(len(ledger) - joined_rows),
        "one_to_zero_or_one_ok": bool(
            joined_rows == len(ledger)
            and int(ledger.duplicated("product_row_key").sum()) == 0
            and matrix_key_duplicates == 0
        ),
        "fold_counts": fold_counts,
    }
    return folds, diag


def arm_frames(joined_folds: Mapping[str, pd.DataFrame], status_ok_only: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for fold, frame in joined_folds.items():
        selected = frame.loc[frame["status"] == "OK"].copy() if status_ok_only else frame.copy()
        out[fold] = selected.reset_index(drop=True)
    return out


def hgb_params(selected_params: Mapping[str, Any], random_state: int, quantile: float, categorical_mask: Sequence[bool]) -> Dict[str, Any]:
    return {
        "loss": "quantile",
        "quantile": float(quantile),
        "max_iter": int(selected_params["max_iter"]),
        "learning_rate": float(selected_params["learning_rate"]),
        "max_leaf_nodes": int(selected_params["max_leaf_nodes"]),
        "min_samples_leaf": int(selected_params["min_samples_leaf"]),
        "l2_regularization": float(selected_params["l2_regularization"]),
        "max_bins": int(selected_params["max_bins"]),
        "early_stopping": bool(selected_params["early_stopping"]),
        "random_state": int(random_state + round(quantile * 1000)),
        "categorical_features": list(categorical_mask),
    }


def fit_tier_model(frame: pd.DataFrame, selected_params: Mapping[str, Any], random_state: int) -> TierModel:
    spec = FeatureSpec(M2_NUMERIC_CONTEXT, M2_CATEGORICAL_CONTEXT)
    prep = Preprocessor(spec.numeric, spec.categorical).fit(frame)
    x_train = prep.transform(frame)
    y_train = frame[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
    models: Dict[float, HistGradientBoostingRegressor] = {}
    for quantile in QUANTILE_LEVELS:
        model = HistGradientBoostingRegressor(
            **hgb_params(selected_params, random_state, quantile, prep.categorical_mask or [])
        )
        model.fit(x_train, y_train)
        models[quantile] = model
    del x_train
    gc.collect()
    return TierModel(spec=spec, preprocessor=prep, models=models)


def predict_quantiles(model: TierModel, frame: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = model.preprocessor.transform(frame)
    raw = np.column_stack([model.models[q].predict(x) for q in QUANTILE_LEVELS]).astype(np.float32)
    rearranged = np.sort(raw, axis=1).astype(np.float32)
    changed = np.abs(raw - rearranged) > 1e-7
    health = {
        "rows": int(len(frame)),
        "any_crossing_before_rearrangement": bool(np.any(np.diff(raw, axis=1) < -1e-7)) if len(frame) else False,
        "changed_fraction_by_quantile": {
            str(q): float(changed[:, i].mean()) if len(frame) else 0.0 for i, q in enumerate(QUANTILE_LEVELS)
        },
    }
    del x, raw
    gc.collect()
    return rearranged, health


def joint_cell_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    cat = categorical_frame(frame, columns)
    values = cat.iloc[:, 0].astype(str)
    for column in columns[1:]:
        values = values + "|" + cat[column].astype(str)
    return values


def category_support_mask(
    frame: pd.DataFrame,
    support_levels: Mapping[str, Mapping[str, Mapping[str, int]]],
    supported_joint_cells: set[str],
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in GATE_SUPPORT_CATEGORICAL:
        values = frame[column].astype("string").fillna("__MISSING__").to_numpy()
        supported = set(support_levels[column].keys())
        mask &= np.fromiter((str(value) in supported for value in values), dtype=bool, count=len(values))
    cell_values = joint_cell_series(frame, GATE_JOINT_CELL).to_numpy()
    mask &= np.fromiter((str(value) in supported_joint_cells for value in cell_values), dtype=bool, count=len(cell_values))
    return mask


def transform_gate_matrix(frame: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler, encoder: OneHotEncoder) -> np.ndarray:
    numeric = imputer.transform(numeric_frame(frame, GATE_DISTANCE_NUMERIC)).astype(np.float32, copy=False)
    numeric = scaler.transform(numeric).astype(np.float32, copy=False)
    categorical = encoder.transform(categorical_frame(frame, GATE_SUPPORT_CATEGORICAL)).astype(np.float32, copy=False)
    return np.hstack([numeric, categorical]).astype(np.float32, copy=False)


def mean_knn_distance(tree: cKDTree, x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    distances, _ = tree.query(x, k=k, workers=-1)
    if distances.ndim == 1:
        return distances.astype(np.float32)
    return distances.mean(axis=1).astype(np.float32)


def fit_gate(train_frame: pd.DataFrame, guard_frame: pd.DataFrame) -> Tuple[GateModel, Dict[str, Any], np.ndarray, Dict[str, Any]]:
    support_levels: Dict[str, Dict[str, Dict[str, int]]] = {}
    unsupported_levels: Dict[str, List[str]] = {}
    for column in GATE_SUPPORT_CATEGORICAL:
        counts = train_frame.groupby(column, dropna=False).agg(anchors=("case_key", "size"), cases=("case_key", "nunique"))
        supported: Dict[str, Dict[str, int]] = {}
        unsupported: List[str] = []
        for value, row in counts.iterrows():
            key = str(value)
            anchors = int(row["anchors"])
            cases = int(row["cases"])
            if anchors >= 50 and cases >= 10:
                supported[key] = {"anchors": anchors, "cases": cases}
            else:
                unsupported.append(key)
        support_levels[column] = supported
        unsupported_levels[column] = unsupported

    joint_counts = (
        pd.DataFrame({"cell": joint_cell_series(train_frame, GATE_JOINT_CELL), "case_key": train_frame["case_key"]})
        .groupby("cell")
        .agg(anchors=("case_key", "size"), cases=("case_key", "nunique"))
    )
    supported_joint_cells = {
        str(cell)
        for cell, row in joint_counts.iterrows()
        if int(row["anchors"]) >= 50 and int(row["cases"]) >= 10
    }
    unsupported_joint_cells = [str(cell) for cell in joint_counts.index if str(cell) not in supported_joint_cells]
    train_category_ok = category_support_mask(train_frame, support_levels, supported_joint_cells)
    guard_category_ok = category_support_mask(guard_frame, support_levels, supported_joint_cells)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_num = imputer.fit_transform(numeric_frame(train_frame.loc[train_category_ok], GATE_DISTANCE_NUMERIC)).astype(np.float32)
    train_num = scaler.fit_transform(train_num).astype(np.float32)
    train_cat = encoder.fit_transform(categorical_frame(train_frame.loc[train_category_ok], GATE_SUPPORT_CATEGORICAL))
    train_x = np.hstack([train_num, train_cat]).astype(np.float32)
    tree = cKDTree(train_x)

    guard_x = transform_gate_matrix(guard_frame, imputer, scaler, encoder)
    guard_distances = mean_knn_distance(tree, guard_x[guard_category_ok], k=25)
    threshold = float(np.quantile(guard_distances, 0.95, method="linear"))
    guard_distance_ok = np.zeros(len(guard_frame), dtype=bool)
    guard_distance_ok[guard_category_ok] = guard_distances <= threshold
    guard_gate_ok = guard_category_ok & guard_distance_ok
    model = GateModel(
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
        tree=tree,
        threshold=threshold,
        support_levels=support_levels,
        unsupported_levels=unsupported_levels,
        supported_joint_cells=supported_joint_cells,
        unsupported_joint_cells=unsupported_joint_cells,
    )
    payload = {
        "definition": {
            "distance": "mean Euclidean distance to k=25 nearest train anchors after train-fit median imputation, standardization, and one-hot support categoricals",
            "threshold": "95th percentile of otherwise category-eligible guard_tune distances",
            "category_support": "train support >=50 anchors and >=10 cases for individual support levels and the joint cell",
            "category_columns": GATE_SUPPORT_CATEGORICAL,
            "joint_cell_columns": GATE_JOINT_CELL,
            "distance_numeric": GATE_DISTANCE_NUMERIC,
            "distance_categorical_one_hot": GATE_SUPPORT_CATEGORICAL,
        },
        "params": {
            "k": 25,
            "threshold": threshold,
            "threshold_percentile": 0.95,
            "category_min_anchors": 50,
            "category_min_cases": 10,
        },
        "train_rows": int(len(train_frame)),
        "train_reference_rows_after_category_support": int(train_category_ok.sum()),
        "guard_rows": int(len(guard_frame)),
        "guard_category_eligible_rows": int(guard_category_ok.sum()),
        "guard_gate_passing_rows": int(guard_gate_ok.sum()),
        "guard_abstention_rate": float(1.0 - guard_gate_ok.mean()) if len(guard_frame) else math.nan,
        "unsupported_levels": unsupported_levels,
        "unsupported_joint_cells": unsupported_joint_cells,
    }
    del train_num, train_cat, train_x, guard_x, guard_distances
    gc.collect()
    return model, payload, guard_gate_ok, {
        "category_pass_rows": int(guard_category_ok.sum()),
        "distance_pass_rows": int(guard_distance_ok.sum()),
        "gate_pass_rows": int(guard_gate_ok.sum()),
    }


def apply_gate(frame: pd.DataFrame, gate: GateModel) -> Tuple[np.ndarray, Dict[str, Any]]:
    category_ok = category_support_mask(frame, gate.support_levels, gate.supported_joint_cells)
    distance_ok = np.zeros(len(frame), dtype=bool)
    distances = np.full(len(frame), np.nan, dtype=np.float32)
    if category_ok.any():
        x = transform_gate_matrix(frame, gate.imputer, gate.scaler, gate.encoder)
        d = mean_knn_distance(gate.tree, x[category_ok], k=25)
        distances[category_ok] = d
        distance_ok[category_ok] = d <= gate.threshold
        del x, d
        gc.collect()
    gate_ok = category_ok & distance_ok
    return gate_ok, {
        "rows": int(len(frame)),
        "category_pass_rows": int(category_ok.sum()),
        "category_fail_rows": int((~category_ok).sum()),
        "distance_pass_rows": int(distance_ok.sum()),
        "distance_fail_rows": int((category_ok & ~distance_ok).sum()),
        "gate_pass_rows": int(gate_ok.sum()),
        "abstain_rows": int((~gate_ok).sum()),
        "abstention_rate": float(1.0 - gate_ok.mean()) if len(frame) else math.nan,
        "distance_mean": float(np.nanmean(distances)) if np.isfinite(distances).any() else None,
        "distance_p95": float(np.nanquantile(distances, 0.95)) if np.isfinite(distances).any() else None,
    }


def calibrated_bounds(q_lo: np.ndarray, q_hi: np.ndarray, c_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    raw_lo = q_lo.astype(np.float64, copy=False) - float(c_alpha)
    raw_hi = q_hi.astype(np.float64, copy=False) + float(c_alpha)
    lo = np.minimum(raw_lo, raw_hi) - INTERVAL_EPS
    hi = np.maximum(raw_lo, raw_hi) + INTERVAL_EPS
    return lo, hi


def conformal_radius(scores: np.ndarray, alpha: float) -> Tuple[float, int, int]:
    finite_scores = scores[np.isfinite(scores)]
    n = int(len(finite_scores))
    if n == 0:
        raise RuntimeError("no_calibration_scores")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    rank = min(rank, n)
    value = float(np.partition(finite_scores, rank - 1)[rank - 1])
    return value, n, rank


def compute_radii(q_pred: np.ndarray, y: np.ndarray, gate_ok: np.ndarray) -> Dict[str, Dict[str, Any]]:
    radii: Dict[str, Dict[str, Any]] = {}
    for alpha in ALPHAS:
        label = ALPHA_LABEL[alpha]
        q_lo_level, q_hi_level = QUANTILE_BY_ALPHA[alpha]
        q_lo = q_pred[:, Q_INDEX[q_lo_level]]
        q_hi = q_pred[:, Q_INDEX[q_hi_level]]
        scores = np.maximum(q_lo[gate_ok] - y[gate_ok], y[gate_ok] - q_hi[gate_ok])
        radius, n, rank = conformal_radius(scores, alpha)
        radii[label] = {
            "alpha": float(alpha),
            "nominal": float(NOMINAL_BY_ALPHA[alpha]),
            "c_alpha": radius,
            "calibration_n": n,
            "rank": rank,
            "formula": "ceil((n+1)*(1-alpha))-th smallest nonconformity score",
        }
    return radii


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    width = hi - lo
    lower_penalty = (2.0 / alpha) * (lo - y) * (y < lo)
    upper_penalty = (2.0 / alpha) * (y - hi) * (y > hi)
    return width + lower_penalty + upper_penalty


def score_test_frame(frame: pd.DataFrame, q_pred: np.ndarray, radii: Mapping[str, Mapping[str, Any]], gate_ok: np.ndarray) -> Dict[str, Dict[str, Any]]:
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
    result: Dict[str, Dict[str, Any]] = {}
    for alpha in ALPHAS:
        label = ALPHA_LABEL[alpha]
        q_lo_level, q_hi_level = QUANTILE_BY_ALPHA[alpha]
        q_lo = q_pred[:, Q_INDEX[q_lo_level]].astype(np.float64, copy=False)
        q_hi = q_pred[:, Q_INDEX[q_hi_level]].astype(np.float64, copy=False)
        lo, hi = calibrated_bounds(q_lo, q_hi, float(radii[label]["c_alpha"]))
        ok = gate_ok
        y_ok = y[ok]
        lo_ok = lo[ok]
        hi_ok = hi[ok]
        width_ok = hi_ok - lo_ok
        covered = (y_ok >= lo_ok) & (y_ok <= hi_ok)
        lower_tail = y_ok < lo_ok
        upper_tail = y_ok > hi_ok
        result[label] = {
            "alpha": float(alpha),
            "nominal": float(NOMINAL_BY_ALPHA[alpha]),
            "total_n": int(len(frame)),
            "n": int(ok.sum()),
            "abstained_n": int((~ok).sum()),
            "abstention": float(1.0 - ok.mean()) if len(frame) else math.nan,
            "coverage": float(covered.mean()) if len(y_ok) else math.nan,
            "covered_n": int(covered.sum()) if len(y_ok) else 0,
            "mean_width": float(np.mean(width_ok)) if len(width_ok) else math.nan,
            "median_width": float(np.median(width_ok)) if len(width_ok) else math.nan,
            "width_p05": float(np.quantile(width_ok, 0.05)) if len(width_ok) else math.nan,
            "width_p95": float(np.quantile(width_ok, 0.95)) if len(width_ok) else math.nan,
            "width_min": float(np.min(width_ok)) if len(width_ok) else math.nan,
            "width_max": float(np.max(width_ok)) if len(width_ok) else math.nan,
            "width_std": float(np.std(width_ok)) if len(width_ok) else math.nan,
            "negative_width_rows": int(np.sum(width_ok < 0)) if len(width_ok) else 0,
            "lower_tail": float(lower_tail.mean()) if len(y_ok) else math.nan,
            "upper_tail": float(upper_tail.mean()) if len(y_ok) else math.nan,
            "winkler": float(np.mean(interval_score(y_ok, lo_ok, hi_ok, float(alpha)))) if len(y_ok) else math.nan,
            "conformal_radius": float(radii[label]["c_alpha"]),
            "calibration_n": int(radii[label]["calibration_n"]),
        }
    return result


def frame_numeric_health(frame: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        out[column] = {
            "nan": int(np.isnan(values).sum()),
            "pos_inf": int(np.isposinf(values).sum()),
            "neg_inf": int(np.isneginf(values).sum()),
        }
    return out


def context_support(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    all_frame = pd.concat([frames[fold] for fold in FOLDS], ignore_index=True)
    train_frame = frames["train"]
    def counts_for(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame({"context_cell": joint_cell_series(frame, GATE_JOINT_CELL), "case_key": frame["case_key"]})
            .groupby("context_cell")
            .agg(rows=("case_key", "size"), cases=("case_key", "nunique"))
            .reset_index()
        )

    all_counts = counts_for(all_frame)
    train_counts = counts_for(train_frame)
    return {
        "context_columns": GATE_JOINT_CELL,
        "min_support_l1_per_l2": MIN_SUPPORT_L1_PER_L2,
        "all_rows": {
            "cells": int(len(all_counts)),
            "below_min_support_cells": int((all_counts["rows"] < MIN_SUPPORT_L1_PER_L2).sum()),
            "min_rows": int(all_counts["rows"].min()) if len(all_counts) else 0,
            "counts": all_counts.sort_values("context_cell").to_dict(orient="records"),
        },
        "train_rows_for_gate_fit": {
            "cells": int(len(train_counts)),
            "below_min_support_cells": int((train_counts["rows"] < MIN_SUPPORT_L1_PER_L2).sum()),
            "min_rows": int(train_counts["rows"].min()) if len(train_counts) else 0,
            "counts": train_counts.sort_values("context_cell").to_dict(orient="records"),
        },
    }


def assert_no_invalid_split(frame: pd.DataFrame) -> None:
    invalid = int((~frame["rq007_split"].isin(["development", "guard"])).sum())
    if invalid:
        raise AssertionError(f"FAIL held_out_guard: invalid_split_rows={invalid}")


def negative_control(joined_folds: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    sample = joined_folds["test"].head(10).copy()
    if sample.empty:
        return {"status": "SKIPPED", "reason": "test_frame_empty"}
    sample.loc[sample.index[0], "rq007_split"] = "held_out"
    try:
        assert_no_invalid_split(sample)
    except AssertionError as exc:
        return {"status": "EXPECTED_FAIL", "failure_output": str(exc)}
    return {"status": "UNEXPECTED_PASS", "failure_output": ""}


def validate_two_arms(a_frames: Mapping[str, pd.DataFrame], b_frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    fold_subset_ok: Dict[str, bool] = {}
    fold_counts: Dict[str, Dict[str, int]] = {}
    for fold in FOLDS:
        a_keys = set(a_frames[fold]["product_row_key"].astype(str))
        b_keys = set(b_frames[fold]["product_row_key"].astype(str))
        fold_subset_ok[fold] = b_keys.issubset(a_keys)
        fold_counts[fold] = {"A_rows": len(a_keys), "B_rows": len(b_keys), "A_minus_B": len(a_keys - b_keys)}
    return {
        "only_variable": "row_filter",
        "row_filter_A": "all K2-covered target_future rows",
        "row_filter_B": "status == OK",
        "m2_numeric_context": M2_NUMERIC_CONTEXT,
        "m2_categorical_context": M2_CATEGORICAL_CONTEXT,
        "excluded_old_ipv_conditioning_columns": IPV_CONDITIONING_COLUMNS,
        "support_gate_context_columns": GATE_JOINT_CELL,
        "alpha_layers": [ALPHA_LABEL[a] for a in ALPHAS],
        "folds": FOLDS,
        "fold_subset_ok": fold_subset_ok,
        "fold_counts": fold_counts,
        "all_b_rows_are_subset_of_a": bool(all(fold_subset_ok.values())),
    }


def numeric_checks(
    arm_name: str,
    frames: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    predict_health: Mapping[str, Any],
) -> Dict[str, Any]:
    y_counts = {}
    numeric_counts = {}
    for fold, frame in frames.items():
        y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=False)
        y_counts[fold] = {
            "rows": int(len(frame)),
            "target_nan": int(np.isnan(y).sum()),
            "target_inf": int(np.isinf(y).sum()),
        }
        numeric_counts[fold] = frame_numeric_health(frame, M2_NUMERIC_CONTEXT)
    coverage_ok = {
        label: bool(0.0 <= float(row["coverage"]) <= 1.0)
        for label, row in metrics.items()
        if np.isfinite(float(row["coverage"]))
    }
    width_ok = {
        label: {
            "negative_width_rows": int(row["negative_width_rows"]),
            "pathological_constant_width": bool(
                np.isfinite(float(row["width_std"])) and float(row["width_std"]) <= 1e-12 and int(row["n"]) > 1
            ),
        }
        for label, row in metrics.items()
    }
    return {
        "arm": arm_name,
        "target_numeric_counts": y_counts,
        "m2_numeric_feature_counts": numeric_counts,
        "coverage_in_0_1": coverage_ok,
        "width_health": width_ok,
        "prediction_health": predict_health,
    }


def run_arm(
    name: str,
    frames: Mapping[str, pd.DataFrame],
    selected_params: Mapping[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    start = time.time()
    print(f"[{utc_now_from_date()}] {name}: fit support gate", flush=True)
    gate, gate_payload, _, guard_diag = fit_gate(frames["train"], frames["guard_tune"])
    cal_gate_ok, cal_gate_diag = apply_gate(frames["calibration"], gate)
    test_gate_ok, test_gate_diag = apply_gate(frames["test"], gate)
    print(f"[{utc_now_from_date()}] {name}: fit M2 quantile model", flush=True)
    model = fit_tier_model(frames["train"], selected_params, random_state)
    print(f"[{utc_now_from_date()}] {name}: predict calibration/test", flush=True)
    q_cal, cal_pred_health = predict_quantiles(model, frames["calibration"])
    y_cal = frames["calibration"][TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False)
    radii = compute_radii(q_cal, y_cal, cal_gate_ok)
    del q_cal, y_cal
    gc.collect()
    q_test, test_pred_health = predict_quantiles(model, frames["test"])
    metrics = score_test_frame(frames["test"], q_test, radii, test_gate_ok)
    del q_test, model
    gc.collect()
    return {
        "arm": name,
        "elapsed_s": round(time.time() - start, 3),
        "row_counts": {fold: int(len(frame)) for fold, frame in frames.items()},
        "gate": {
            "fit": gate_payload,
            "guard_tune": guard_diag,
            "calibration": cal_gate_diag,
            "test": test_gate_diag,
        },
        "conformal_radii": radii,
        "metrics": metrics,
        "prediction_health": {"calibration": cal_pred_health, "test": test_pred_health},
    }


def combined_abstention(joined_test: pd.DataFrame, b_test_gate_diag: Mapping[str, Any]) -> Dict[str, Any]:
    denominator = int(len(joined_test))
    mechanism1 = int((joined_test["status"] != "OK").sum())
    mechanism2_after_ok = int(b_test_gate_diag["abstain_rows"])
    combined = mechanism1 + mechanism2_after_ok
    return {
        "definition": "mechanism1_abstain_rows + mechanism2_abstain_rows_among_status_OK, denominator is K2-covered target_future rows in RQ009 fold=test",
        "denominator": denominator,
        "mechanism1_abstain_rows": mechanism1,
        "mechanism1_contribution_rate": pct(mechanism1, denominator),
        "mechanism2_abstain_after_mechanism1_ok_rows": mechanism2_after_ok,
        "mechanism2_contribution_rate": pct(mechanism2_after_ok, denominator),
        "combined_abstain_rows": combined,
        "combined_abstention_rate": pct(combined, denominator),
        "source_file": str(K2_LEDGER_ROOT.relative_to(REPO_ROOT)),
        "source_columns": ["status", "fold", "measurement_role"],
    }


def comparison(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label in ["80", "90", "95"]:
        a_m = a["metrics"][label]
        b_m = b["metrics"][label]
        out[label] = {
            "coverage_delta_B_minus_A": float(b_m["coverage"] - a_m["coverage"]),
            "mean_width_delta_B_minus_A": float(b_m["mean_width"] - a_m["mean_width"]),
            "mean_width_pct_B_vs_A": float((b_m["mean_width"] / a_m["mean_width"] - 1.0) * 100.0),
            "mechanism2_abstention_delta_B_minus_A": float(b_m["abstention"] - a_m["abstention"]),
        }
    return out


def source_reference() -> Dict[str, Any]:
    metrics = pd.read_csv(METRICS_SUMMARY_PATH)
    m2 = metrics[(metrics["tier"] == "M2") & (metrics["alpha_label"].astype(str) == "90")].iloc[0].to_dict()
    m0 = metrics[(metrics["tier"] == "M0") & (metrics["alpha_label"].astype(str) == "90")].iloc[0].to_dict()
    return {
        "decision_path": str(RQ009_DECISION_PATH.relative_to(REPO_ROOT)),
        "metrics_summary_path": str(METRICS_SUMMARY_PATH.relative_to(REPO_ROOT)),
        "scope_note": "external reference only; original RQ009 fold=test includes held_out rows and is not a reproduction target for this run",
        "m2_90": {
            "coverage": float(m2["coverage"]),
            "covered_n": int(round(float(m2["coverage"]) * int(m2["n"]))),
            "mean_width": float(m2["mean_width"]),
            "winkler": float(m2["winkler"]),
            "abstention": float(m2["abstention"]),
            "abstained_n": int(m2["abstained_n"]),
            "n": int(m2["n"]),
            "total_n": int(m2["total_n"]),
        },
        "m0_90": {
            "mean_width": float(m0["mean_width"]),
            "winkler": float(m0["winkler"]),
            "n": int(m0["n"]),
            "total_n": int(m0["total_n"]),
        },
        "m2_vs_m0_90": {
            "mean_width_pct": float((float(m2["mean_width"]) / float(m0["mean_width"]) - 1.0) * 100.0),
            "winkler_pct": float((float(m2["winkler"]) / float(m0["winkler"]) - 1.0) * 100.0),
        },
    }


def format_metric_table(arm_results: Mapping[str, Any]) -> str:
    lines = [
        "| alpha | coverage | covered/through-mechanism2 | mean width | median width | mechanism2 abstention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["80", "90", "95"]:
        row = arm_results["metrics"][label]
        lines.append(
            "| {label} | {coverage:.6f} | {covered:,}/{n:,} | {mean_width:.6f} | {median_width:.6f} | {abstention:.4%} ({abstained:,}/{total:,}) |".format(
                label=label,
                coverage=float(row["coverage"]),
                covered=int(row["covered_n"]),
                n=int(row["n"]),
                mean_width=float(row["mean_width"]),
                median_width=float(row["median_width"]),
                abstention=float(row["abstention"]),
                abstained=int(row["abstained_n"]),
                total=int(row["total_n"]),
            )
        )
    return "\n".join(lines)


def format_comparison_table(comp: Mapping[str, Any]) -> str:
    lines = [
        "| alpha | coverage B-A | mean width B-A | mean width B vs A formula | mechanism2 abstention B-A |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["80", "90", "95"]:
        row = comp[label]
        lines.append(
            f"| {label} | {row['coverage_delta_B_minus_A']:+.6f} | {row['mean_width_delta_B_minus_A']:+.6f} | B/A - 1 = {row['mean_width_pct_B_vs_A']:+.2f}% | {row['mechanism2_abstention_delta_B_minus_A']:+.4%} |"
        )
    return "\n".join(lines)


def compact_source_line(numerator: int, denominator: int, filter_text: str, source: str, columns: Sequence[str]) -> str:
    return f"{pct_text(numerator, denominator)}；筛选条件：{filter_text}；来源：`{source}`；列：`{', '.join(columns)}`。"


def build_report(payload: Mapping[str, Any], timestamp_utc: str) -> str:
    a = payload["arms"]["A_domain_baseline"]
    b = payload["arms"]["B_status_ok"]
    comp = payload["comparison"]
    combined = payload["combined_abstention_test"]
    join = payload["join_health"]
    ledger = payload["k2_ledger"]
    two_arm = payload["two_arm_invariance"]
    b90 = b["metrics"]["90"]
    a90 = a["metrics"]["90"]
    support_a = payload["context_support"]["A_domain_baseline"]
    support_b = payload["context_support"]["B_status_ok"]
    neg = payload["negative_control"]
    heldout_invalid = payload["held_out_assertion"]["invalid_split_rows_joined_all_folds"]
    source_k2 = payload["k2_ledger"]["source_path"]
    source_matrix = payload["join_health"]["matrix_source_path"]
    ext = payload["external_reference"]
    ext_m2 = ext["m2_90"]
    ext_m0 = ext["m0_90"]
    ext_cmp = ext["m2_vs_m0_90"]

    report = f"""# RQ016-A1 envelope rebuild

本轮要解决的问题是：机制二的人类 envelope 过去建在含伪零的 RQ009 样本上，非 OK 行在旧目标列里可能表现为精确 0，从而把“权重近均匀，IPV 数值不携带候选间判别信息”和“IPV 恰为中性”混在一起。整体研究链路已经有 RQ015 冻结机制一；本次是机制二重建环节，只在 K2 台账覆盖的 `development + guard` 域内做 A/B 两臂描述性计算。

人类 envelope 是 RQ009 已接受的 context-conditioned split-conformal 区间：先用上下文变量给人类目标 IPV 建条件分位数，再用 calibration fold 的 split-conformal 半径扩展区间；测试行落在支持门外时，机制二弃权。

## 结论

**重建后 coverage。** B 臂是只保留 `status == "OK"` 的重建结果。90% 名义层下，coverage = {float(b90['coverage']):.6f}，分子分母为 {int(b90['covered_n']):,}/{int(b90['n']):,}；筛选条件：K2 `artifact_id == rq009_feature_matrix`、`measurement_role == target_future`、`rq007_split in {{development, guard}}`、RQ009 `fold == test`、`status == OK`、机制二支持门通过；来源：`{source_k2}` 的 `status/rq007_split/measurement_role` 与 `{source_matrix}` 的 `fold/target_ipv_future`。A 臂同域基线 coverage = {float(a90['coverage']):.6f}，分子分母为 {int(a90['covered_n']):,}/{int(a90['n']):,}；B-A = {comp['90']['coverage_delta_B_minus_A']:+.6f}。

**重建后区间宽度。** B 臂 90% mean width = {float(b90['mean_width']):.6f}，median width = {float(b90['median_width']):.6f}，分母为机制二支持门通过的 {int(b90['n']):,} 行；来源同上，区间列由脚本按 `q_lo/q_hi/c_alpha` 在内存生成。A 臂 90% mean width = {float(a90['mean_width']):.6f}，median width = {float(a90['median_width']):.6f}；B 相对 A 的 mean width 变化为 {float(b90['mean_width']):.6f}/{float(a90['mean_width']):.6f} - 1 = {comp['90']['mean_width_pct_B_vs_A']:+.2f}%。

**机制二自身弃权率。** B 臂 90% 层的机制二弃权率与所有 alpha 层相同，因为支持门不随 alpha 变化：{float(b90['abstention']):.4%}，分子分母为 {int(b90['abstained_n']):,}/{int(b90['total_n']):,}；筛选条件：同上但分母为 `status == OK` 的 test 行，分子为机制二支持门未通过；来源：`{source_k2}` 的 `status` 与 `{source_matrix}` 的 RQ009 M2 上下文字段。A 臂同域基线机制二弃权率为 {float(a90['abstention']):.4%}，分子分母为 {int(a90['abstained_n']):,}/{int(a90['total_n']):,}。

**两道门串联后的合并弃权率。** 在 K2 覆盖的 RQ009 test 目标行上，合并弃权率 = {float(combined['combined_abstention_rate']):.4%}，分子分母为 {int(combined['combined_abstain_rows']):,}/{int(combined['denominator']):,}。其中机制一贡献 {int(combined['mechanism1_abstain_rows']):,}/{int(combined['denominator']):,} = {float(combined['mechanism1_contribution_rate']):.4%}，筛选条件为 `status != OK`；机制二贡献 {int(combined['mechanism2_abstain_after_mechanism1_ok_rows']):,}/{int(combined['denominator']):,} = {float(combined['mechanism2_contribution_rate']):.4%}，筛选条件为 `status == OK` 且机制二支持门未通过。来源：`{source_k2}` 的 `status/measurement_role/rq007_split` 与 `{source_matrix}` 的 `fold`。

## 两臂结果

A 臂（域内基线，保留未过门行，含伪零）：

{format_metric_table(a)}

B 臂（处理组，只保留 `status == "OK"`）：

{format_metric_table(b)}

A/B 差值：

{format_comparison_table(comp)}

RQ009 已发表数只能作外部参照：`reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md` 的 RQ009-KC-R3 写的是原 RQ009 域。原 RQ009 `metrics_summary.csv` 中 M2 90% coverage = {float(ext_m2['coverage']):.6f}，分子分母按 `coverage * n` 取整为 {int(ext_m2['covered_n']):,}/{int(ext_m2['n']):,}；M2 90% 弃权率 = {float(ext_m2['abstention']):.4%}，分子分母为 {int(ext_m2['abstained_n']):,}/{int(ext_m2['total_n']):,}；M2 相对 M0 的 mean_width 变化 = {float(ext_m2['mean_width']):.6f}/{float(ext_m0['mean_width']):.6f} - 1 = {float(ext_cmp['mean_width_pct']):+.2f}%，Winkler 变化 = {float(ext_m2['winkler']):.6f}/{float(ext_m0['winkler']):.6f} - 1 = {float(ext_cmp['winkler_pct']):+.2f}%。筛选条件：`tier in {{M2, M0}}`、`alpha_label == 90`、原 RQ009 `fold == test`；来源：`{ext['metrics_summary_path']}` 的 `coverage/n/abstained_n/total_n/mean_width/winkler` 和 `{ext['decision_path']}`。这个 test 域包含 held_out，因此本轮结果不写作复现或未复现 RQ009。

## 自查

**连接健康。** K2 target_future 台账左连接到 RQ009 feature matrix：命中 {join['ledger_left_join_hits']:,}/{join['ledger_rows']:,}，未命中 {join['ledger_left_join_misses']:,}/{join['ledger_rows']:,}；K2 `product_row_key` 重复 {ledger['product_row_key_duplicates']}，K2 `canonical_key` 重复 {ledger['canonical_key_duplicates']}，matrix `product_row_key` 重复 {join['matrix_product_row_key_duplicates']}，one-to-zero-or-one 检查结果为 `{join['one_to_zero_or_one_ok']}`。来源：`{source_k2}` 的 `product_row_key/canonical_key` 与 `{source_matrix}` 的 `case_key/anchor_frame_index/perspective/source_dataset`。

**held_out 断言。** 本轮所有参与计算的连接后行中，`rq007_split` 不在 `{{development, guard}}` 的实测计数为 {heldout_invalid} 行；来源：`{source_k2}` 的 `rq007_split`。本轮没有打开任何受保护 confirmation 划分文件。

**两臂只差一个变量。** 两臂共用 alpha 层 `{two_arm['alpha_layers']}`、RQ009 fold 结构 `{two_arm['folds']}`、M2 数值上下文字段 `{two_arm['m2_numeric_context']}`、M2 类别上下文字段 `{two_arm['m2_categorical_context']}`、支持门联合格字段 `{two_arm['support_gate_context_columns']}`。`source_dataset` 仅作为连接键/报告键，未作为预测变量。B 臂所有 fold 的行键都是 A 臂子集：`{two_arm['all_b_rows_are_subset_of_a']}`。唯一变量是行过滤：A 为 K2 覆盖的全部 target_future 行，B 为 `status == OK` 行。

**每格支撑量。** 支持门联合格为 `geometry_path_category + priority_role + agent_type_pair`，不按数据源分格。A 臂全样本 {support_a['all_rows']['cells']} 格，最小格样本数 {support_a['all_rows']['min_rows']}，低于 `MIN_SUPPORT_L1_PER_L2 = 5` 的格数 {support_a['all_rows']['below_min_support_cells']}；B 臂全样本 {support_b['all_rows']['cells']} 格，最小格样本数 {support_b['all_rows']['min_rows']}，低于 5 的格数 {support_b['all_rows']['below_min_support_cells']}。完整逐格计数在 `key_numbers.json` 的 `context_support`。

**负对照。** 我故意把 10 行 test 样本中的一行 `rq007_split` 改成 `held_out` 后重跑 held_out 断言，输出为：`{neg['failure_output']}`；负对照状态 `{neg['status']}`。

**数值健康。** B 臂 test 目标列 NaN/inf 计数为 {payload['numeric_health']['B_status_ok']['target_numeric_counts']['test']['target_nan']}/{payload['numeric_health']['B_status_ok']['target_numeric_counts']['test']['target_inf']}；A 臂 test 目标列 NaN/inf 计数为 {payload['numeric_health']['A_domain_baseline']['target_numeric_counts']['test']['target_nan']}/{payload['numeric_health']['A_domain_baseline']['target_numeric_counts']['test']['target_inf']}。B 臂 90% 区间负宽度行数 {payload['numeric_health']['B_status_ok']['width_health']['90']['negative_width_rows']}，病态常数宽度标记 `{payload['numeric_health']['B_status_ok']['width_health']['90']['pathological_constant_width']}`；所有 alpha 的 coverage 均落在 [0, 1]：`{payload['numeric_health']['B_status_ok']['coverage_in_0_1']}`。

## 待监督方拍板

本轮没有请求新的执行授权或阈值选择。唯一需要监督方确认的是是否接受本报告采用的 M2-only 支持门：判断依据是任务书禁止 M3/M4 旧 IPV-conditioning 通道，本脚本因此在训练模型和支持门距离特征中均排除了 `counterpart_ipv_current/counterpart_ipv_error_current/counterpart_ipv_slope_pre_anchor`；不接受的后果是需要另开一轮，明确允许使用 RQ009 原支持门中的旧 counterpart IPV 通道，但那会引入任务书要求避免的第二条污染路径。

state: WAITING_ON_COMMANDER
timestamp_utc: {timestamp_utc}
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-state", type=int, default=20260626)
    parser.add_argument("--report-only", action="store_true", help="rewrite the Markdown report from existing key_numbers.json")
    args = parser.parse_args()
    started = utc_now_from_date()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        payload = json.loads(KEY_NUMBERS_PATH.read_text(encoding="utf-8"))
        payload["external_reference"] = source_reference()
        timestamp_utc = utc_now_from_date()
        payload["created_utc"] = timestamp_utc
        write_json(KEY_NUMBERS_PATH, payload)
        REPORT_PATH.write_text(build_report(payload, timestamp_utc), encoding="utf-8")
        print(f"[{timestamp_utc}] rewrote {REPORT_PATH.relative_to(REPO_ROOT)} from {KEY_NUMBERS_PATH.relative_to(REPO_ROOT)}", flush=True)
        return 0

    print(f"[{started}] load feature dictionary and K2 ledger", flush=True)
    feature_dict = read_feature_dictionary()
    selected_params = load_selected_hgb_params()
    ledger, ledger_diag = load_k2_target_ledger()
    print(f"[{utc_now_from_date()}] join RQ009 matrix folds to K2 target ledger", flush=True)
    joined_folds, join_diag = load_joined_folds(ledger)
    all_joined = pd.concat([joined_folds[fold][["rq007_split"]] for fold in FOLDS], ignore_index=True)
    invalid_split_rows = int((~all_joined["rq007_split"].isin(["development", "guard"])).sum())
    if invalid_split_rows:
        raise RuntimeError(f"held_out_guard_failed invalid_split_rows={invalid_split_rows}")
    neg = negative_control(joined_folds)

    a_frames = arm_frames(joined_folds, status_ok_only=False)
    b_frames = arm_frames(joined_folds, status_ok_only=True)
    two_arm = validate_two_arms(a_frames, b_frames)
    support = {
        "A_domain_baseline": context_support(a_frames),
        "B_status_ok": context_support(b_frames),
    }

    arm_a = run_arm("A_domain_baseline", a_frames, selected_params, args.random_state)
    arm_b = run_arm("B_status_ok", b_frames, selected_params, args.random_state)
    comp = comparison(arm_a, arm_b)
    combined = combined_abstention(joined_folds["test"], arm_b["gate"]["test"])
    health = {
        "A_domain_baseline": numeric_checks("A_domain_baseline", a_frames, arm_a["metrics"], arm_a["prediction_health"]),
        "B_status_ok": numeric_checks("B_status_ok", b_frames, arm_b["metrics"], arm_b["prediction_health"]),
    }
    timestamp_utc = utc_now_from_date()
    payload: Dict[str, Any] = {
        "created_utc": timestamp_utc,
        "started_utc": started,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "feature_dictionary": feature_dict,
        "selected_hgb_params": selected_params,
        "m2_feature_contract": {
            "numeric_context": M2_NUMERIC_CONTEXT,
            "categorical_context": M2_CATEGORICAL_CONTEXT,
            "support_gate_distance_numeric": GATE_DISTANCE_NUMERIC,
            "support_gate_categorical": GATE_SUPPORT_CATEGORICAL,
            "excluded_old_ipv_conditioning_columns": IPV_CONDITIONING_COLUMNS,
            "source_dataset_used_as_predictor": False,
        },
        "k2_ledger": ledger_diag,
        "join_health": join_diag,
        "held_out_assertion": {
            "invalid_split_rows_joined_all_folds": invalid_split_rows,
            "allowed_splits": ["development", "guard"],
        },
        "two_arm_invariance": two_arm,
        "context_support": support,
        "negative_control": neg,
        "arms": {
            "A_domain_baseline": arm_a,
            "B_status_ok": arm_b,
        },
        "comparison": comp,
        "combined_abstention_test": combined,
        "numeric_health": health,
        "external_reference": source_reference(),
    }
    write_json(KEY_NUMBERS_PATH, payload)
    REPORT_PATH.write_text(build_report(payload, timestamp_utc), encoding="utf-8")
    print(f"[{timestamp_utc}] wrote {KEY_NUMBERS_PATH.relative_to(REPO_ROOT)}", flush=True)
    print(f"[{timestamp_utc}] wrote {REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
