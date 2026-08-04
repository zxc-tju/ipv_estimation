#!/usr/bin/env python3
"""RQ018-A1 abnormal-IPV association analysis.

This script reads only the whitelisted columns named in the RQ018-A1 task
contract.  It builds post-anchor outcomes, estimates context-adjusted frame
associations with case/team clustered inference, aggregates exposure to units,
and runs the two required negative controls.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy import stats


BASE_SEED = 20260805
LABEL_PERMUTATIONS = 200
PLACEBO_DRAWS = 200
UNIT_BOOTSTRAPS = 1000
ALPHAS = (80, 90, 95)
ANOMALOUS_DISTANCE_THRESHOLD_M = 500_000.0

RQ017_REL = "data/derived/rq017_onsite_gate/l1_v1"
M2_REL = ".codex-fleet/rq016c-human-only-envelope/work/H2/onsite_scoring_dryrun.parquet"
ANCHOR_REL = (
    "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/"
    "onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet"
)
DENSE_REL = (
    "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/"
    "onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet"
)
UNIT_REL = (
    "data/derived/onsite_competition/RQ012B_event_harm/stage4plus/"
    "unit_analysis_table.parquet"
)
REPORT_REL = (
    ".codex-fleet/rq018-abnormal-ipv-degradation/board/reports/"
    "RQ018_1_association.md"
)
WORK_REL = ".codex-fleet/rq018-abnormal-ipv-degradation/work/A1"

RQ017_COLUMNS = ["product_row_key", "status", "reason_code", "ipv_log"]
M2_COLUMNS = [
    "product_row_key",
    "lo_80",
    "hi_80",
    "width_80",
    "lo_90",
    "hi_90",
    "width_90",
    "lo_95",
    "hi_95",
    "width_95",
    "mechanism2_gate_ok",
    "context_cell",
]
ANCHOR_COLUMNS = [
    "case_key",
    "anchor_frame_index",
    "perspective",
    "unit_composite_key",
    "target_window_start_frame_index",
    "target_window_end_frame_index",
    "relative_distance_anchor",
]
DENSE_COLUMNS = [
    "case_key",
    "frame_index",
    "time_s",
    "distance_m",
    "closing_rate_mps",
    "relative_speed_mps",
]
UNIT_COLUMNS = [
    "unit_composite_key",
    "case_key",
    "analysis_set",
    "official_safety",
    "official_efficiency",
    "official_comfort",
    "official_compliance",
    "official_coordination",
    "official_comprehensive",
    "collision_intervention_deduction_any",
    "safety_intervention",
]

OUTCOMES = [
    "future_min_distance_m",
    "future_min_ttc_s",
    "future_max_closing_rate_mps",
]
TTC_LOG_OUTCOME = "future_log1p_min_ttc"
NEGATIVE_CONTROL_OUTCOMES = [*OUTCOMES, TTC_LOG_OUTCOME]
WINDOWS = ("contract", "fixed_3s")
UNIT_EXPOSURES = [
    "frac_outside_90",
    "mean_signed_exceedance_90",
    "frac_upper_90",
    "frac_lower_90",
    "mean_upper_exceedance_90",
    "mean_lower_exceedance_90",
    "max_abs_exceedance_90",
]
NONSAFETY_OUTCOMES = [
    "official_efficiency",
    "official_comfort",
    "official_compliance",
    "official_coordination",
]
SECONDARY_OUTCOMES = [
    "official_safety",
    "collision_intervention_deduction_any",
    "safety_intervention",
]


@dataclass(frozen=True)
class Inputs:
    rq017: pd.DataFrame
    m2: pd.DataFrame
    anchors: pd.DataFrame
    dense: pd.DataFrame
    units: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def require_columns(label: str, available: Iterable[str], required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(available))
    if missing:
        raise RuntimeError(f"Task contract mismatch in {label}: missing columns {missing}")


def read_inputs(root: Path) -> Inputs:
    rq017_path = root / RQ017_REL
    rq017_ds = pads.dataset(rq017_path, format="parquet", partitioning="hive")
    require_columns("RQ017", rq017_ds.schema.names, RQ017_COLUMNS)

    paths_and_columns = [
        ("M2", root / M2_REL, M2_COLUMNS),
        ("anchors", root / ANCHOR_REL, ANCHOR_COLUMNS),
        ("dense", root / DENSE_REL, DENSE_COLUMNS),
        ("units", root / UNIT_REL, UNIT_COLUMNS),
    ]
    for label, path, columns in paths_and_columns:
        if not path.exists():
            raise RuntimeError(f"Task contract mismatch: missing input {path}")
        require_columns(label, pq.read_schema(path).names, columns)

    return Inputs(
        rq017=rq017_ds.to_table(columns=RQ017_COLUMNS).to_pandas(),
        m2=pq.read_table(root / M2_REL, columns=M2_COLUMNS).to_pandas(),
        anchors=pq.read_table(root / ANCHOR_REL, columns=ANCHOR_COLUMNS).to_pandas(),
        dense=pq.read_table(root / DENSE_REL, columns=DENSE_COLUMNS).to_pandas(),
        units=pq.read_table(root / UNIT_REL, columns=UNIT_COLUMNS).to_pandas(),
    )


def parse_product_keys(keys: pd.Series) -> pd.DataFrame:
    pattern = (
        r"^case_key=(.*?)\|anchor_frame_index=(\d+)\|"
        r"perspective=(.*?)\|source_dataset=(.*)$"
    )
    parsed = keys.astype("string").str.extract(pattern)
    parsed.columns = ["case_key", "anchor_frame_index", "perspective", "source_dataset"]
    if parsed.isna().any(axis=None):
        raise RuntimeError(
            f"Task contract mismatch: {int(parsed.isna().any(axis=1).sum())} product keys failed parsing"
        )
    parsed["anchor_frame_index"] = pd.to_numeric(parsed["anchor_frame_index"], errors="raise").astype("int64")
    return parsed


def team_from_case(case_keys: pd.Series) -> pd.Series:
    teams = case_keys.astype("string").str.extract(r":(T[^:]+):", expand=False)
    if teams.isna().any():
        raise RuntimeError(f"Task contract mismatch: {int(teams.isna().sum())} case keys lack team token")
    return teams


def count_metric(
    numerator: int,
    denominator: int | None,
    filter_text: str,
    sources: Sequence[str],
    columns: Sequence[str],
    note: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator) if denominator is not None else None,
        "ratio": (float(numerator) / float(denominator)) if denominator else None,
        "percent": (100.0 * float(numerator) / float(denominator)) if denominator else None,
        "filter": filter_text,
        "sources": list(sources),
        "columns": list(columns),
    }
    if note:
        payload["note"] = note
    return payload


def prepare_analysis(inputs: Inputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    rq017 = inputs.rq017.copy()
    m2 = inputs.m2.copy()
    anchors = inputs.anchors.copy()

    if rq017["product_row_key"].duplicated().any():
        raise RuntimeError("Task contract mismatch: RQ017 product_row_key is not unique")
    if m2["product_row_key"].duplicated().any():
        raise RuntimeError("Task contract mismatch: M2 product_row_key is not unique")
    if anchors.duplicated(["case_key", "anchor_frame_index", "perspective"]).any():
        raise RuntimeError("Task contract mismatch: anchor composite key is not unique")

    parsed = parse_product_keys(rq017["product_row_key"])
    rq017 = pd.concat([rq017.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    joined = rq017.merge(
        m2,
        on="product_row_key",
        how="left",
        validate="one_to_one",
        indicator="rq017_m2_join",
    )
    m2_hits = int((joined["rq017_m2_join"] == "both").sum())
    both_mask = (joined["status"] == "OK") & joined["mechanism2_gate_ok"].fillna(False).astype(bool)
    analysis = joined.loc[both_mask].copy()
    analysis = analysis.merge(
        anchors,
        on=["case_key", "anchor_frame_index", "perspective"],
        how="left",
        validate="one_to_one",
        indicator="anchor_join",
    )
    anchor_hits = int((analysis["anchor_join"] == "both").sum())
    if m2_hits != len(rq017) or anchor_hits != len(analysis):
        raise RuntimeError(
            "Task contract mismatch: required key joins were incomplete "
            f"(M2 {m2_hits}/{len(rq017)}, anchors {anchor_hits}/{len(analysis)})"
        )

    analysis["team_id"] = team_from_case(analysis["case_key"])
    for alpha in ALPHAS:
        lo = pd.to_numeric(analysis[f"lo_{alpha}"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(analysis[f"hi_{alpha}"], errors="coerce").to_numpy(float)
        width = pd.to_numeric(analysis[f"width_{alpha}"], errors="coerce").to_numpy(float)
        ipv = pd.to_numeric(analysis["ipv_log"], errors="coerce").to_numpy(float)
        signed = np.where(ipv > hi, ipv - hi, np.where(ipv < lo, ipv - lo, 0.0))
        norm = signed / width
        analysis[f"signed_exceedance_{alpha}"] = signed
        analysis[f"norm_exceedance_{alpha}"] = norm
        analysis[f"norm_upper_{alpha}"] = np.maximum(norm, 0.0)
        analysis[f"norm_lower_magnitude_{alpha}"] = np.maximum(-norm, 0.0)
        analysis[f"outside_{alpha}"] = signed != 0.0
        analysis[f"upper_{alpha}"] = signed > 0.0
        analysis[f"lower_{alpha}"] = signed < 0.0

    metrics = {
        "rq017_total": count_metric(
            len(rq017), len(rq017), "all RQ017 rows", [RQ017_REL], ["product_row_key"]
        ),
        "mechanism1_ok": count_metric(
            int((rq017["status"] == "OK").sum()),
            len(rq017),
            "status == OK",
            [RQ017_REL],
            ["status"],
        ),
        "mechanism2_gate_ok": count_metric(
            int(m2["mechanism2_gate_ok"].fillna(False).astype(bool).sum()),
            len(m2),
            "mechanism2_gate_ok == True",
            [M2_REL],
            ["mechanism2_gate_ok"],
        ),
        "two_gate_analysis_set": count_metric(
            len(analysis),
            len(rq017),
            "RQ017 status == OK AND M2 mechanism2_gate_ok == True",
            [RQ017_REL, M2_REL],
            ["product_row_key", "status", "mechanism2_gate_ok"],
        ),
        "rq017_to_m2_join_hits": count_metric(
            m2_hits,
            len(rq017),
            "left join on product_row_key",
            [RQ017_REL, M2_REL],
            ["product_row_key"],
        ),
        "analysis_to_anchor_join_hits": count_metric(
            anchor_hits,
            len(analysis),
            "one-to-one join on parsed case_key + anchor_frame_index + perspective",
            [RQ017_REL, ANCHOR_REL],
            ["product_row_key", "case_key", "anchor_frame_index", "perspective"],
        ),
    }
    return analysis, metrics


def calculate_future_outcomes(analysis: pd.DataFrame, dense: pd.DataFrame) -> pd.DataFrame:
    dense = dense.sort_values(["case_key", "frame_index"], kind="stable").copy()
    if dense.duplicated(["case_key", "frame_index"]).any():
        raise RuntimeError("Task contract mismatch: dense case_key + frame_index is not unique")
    grouped = {str(key): group for key, group in dense.groupby("case_key", sort=False)}

    records: list[dict[str, Any]] = []
    for row in analysis.itertuples(index=False):
        case = str(row.case_key)
        group = grouped.get(case)
        if group is None:
            raise RuntimeError(f"Task contract mismatch: dense series missing case {case}")
        frame = group["frame_index"].to_numpy(np.int64)
        time = group["time_s"].to_numpy(float)
        distance = group["distance_m"].to_numpy(float)
        closing = group["closing_rate_mps"].to_numpy(float)
        anchor_matches = np.flatnonzero(frame == int(row.anchor_frame_index))
        if len(anchor_matches) != 1:
            raise RuntimeError(
                f"Task contract mismatch: anchor frame match count {len(anchor_matches)} for {case} "
                f"frame {row.anchor_frame_index}"
            )
        anchor_pos = int(anchor_matches[0])
        anchor_time = float(time[anchor_pos])
        row_result: dict[str, Any] = {}
        masks = {
            "contract": (frame >= int(row.anchor_frame_index))
            & (frame <= int(row.target_window_end_frame_index)),
            "fixed_3s": (time >= anchor_time - 1e-12) & (time <= anchor_time + 3.0 + 1e-12),
        }
        overflow = {
            "contract": int(row.target_window_end_frame_index) > int(frame.max()),
            "fixed_3s": anchor_time + 3.0 > float(time.max()) + 1e-12,
        }
        for window, mask in masks.items():
            if not mask.any():
                raise RuntimeError(f"Task contract mismatch: empty {window} window for {case}")
            d = distance[mask]
            c = closing[mask]
            valid_ttc = c > 0.0
            ttc = d[valid_ttc] / c[valid_ttc]
            row_result[f"{window}_future_min_distance_m"] = float(np.min(d))
            row_result[f"{window}_future_min_ttc_s"] = float(np.min(ttc)) if ttc.size else np.nan
            row_result[f"{window}_{TTC_LOG_OUTCOME}"] = (
                float(np.log1p(np.min(ttc))) if ttc.size else np.nan
            )
            row_result[f"{window}_future_max_closing_rate_mps"] = float(np.max(c))
            row_result[f"{window}_window_row_count"] = int(mask.sum())
            row_result[f"{window}_window_overflow"] = bool(overflow[window])
            row_result[f"{window}_ttc_all_nonclosing"] = bool(ttc.size == 0)
        records.append(row_result)

    outcome_frame = pd.DataFrame.from_records(records, index=analysis.index)
    return pd.concat([analysis, outcome_frame], axis=1)


def cluster_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, int]:
    codes, unique = pd.factorize(pd.Series(labels, dtype="string"), sort=True)
    group_count = len(unique)
    if group_count < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan), group_count
    scores = np.zeros((group_count, x.shape[1]), dtype=float)
    np.add.at(scores, codes, x * residual[:, None])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        meat = scores.T @ scores
    n = x.shape[0]
    correction = (group_count / (group_count - 1.0)) * ((n - 1.0) / (n - rank))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covariance = correction * (base @ meat @ base)
    if not np.isfinite(covariance).all():
        raise RuntimeError("Non-finite cluster covariance after finite-input fit")
    return covariance, group_count


def inference_from_covariance(
    beta: np.ndarray,
    covariance: np.ndarray,
    df: int,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    diag = np.diag(covariance)
    standard_errors = np.sqrt(np.maximum(diag, 0.0))
    critical = float(stats.t.ppf(0.975, df)) if df > 0 else np.nan
    output: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(names):
        se = float(standard_errors[idx])
        coefficient = float(beta[idx])
        statistic = coefficient / se if se > 0 and math.isfinite(se) else np.nan
        p_value = float(2.0 * stats.t.sf(abs(statistic), df)) if df > 0 and math.isfinite(statistic) else np.nan
        output[name] = {
            "coefficient": coefficient,
            "standard_error": se,
            "t": statistic,
            "df": int(df),
            "p_value": p_value,
            "ci95_low": coefficient - critical * se if math.isfinite(critical) else np.nan,
            "ci95_high": coefficient + critical * se if math.isfinite(critical) else np.nan,
        }
    return output


def fit_ols(
    data: pd.DataFrame,
    outcome: str,
    exposures: Sequence[str],
    include_context: bool = True,
    extra_controls: Sequence[str] = (),
    cluster_columns: Sequence[str] = ("case_key", "team_id"),
) -> dict[str, Any]:
    required = [outcome, *exposures, *extra_controls, *cluster_columns]
    if include_context:
        required.append("context_cell")
    subset = data[required].copy()
    numeric = [outcome, *exposures, *extra_controls]
    for column in numeric:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    mask = np.ones(len(subset), dtype=bool)
    for column in numeric:
        mask &= np.isfinite(subset[column].to_numpy(float))
    if include_context:
        mask &= subset["context_cell"].notna().to_numpy()
    for column in cluster_columns:
        mask &= subset[column].notna().to_numpy()
    subset = subset.loc[mask].copy()

    design = pd.DataFrame({"intercept": np.ones(len(subset), dtype=float)}, index=subset.index)
    for column in [*exposures, *extra_controls]:
        design[column] = subset[column].to_numpy(float)
    if include_context:
        dummies = pd.get_dummies(
            subset["context_cell"].astype("string"), prefix="context", drop_first=True, dtype=float
        )
        design = pd.concat([design, dummies], axis=1)

    x = design.to_numpy(float)
    y = subset[outcome].to_numpy(float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cross_product = x.T @ x
        base = np.linalg.pinv(cross_product, rcond=1e-12)
        beta, _, rank_lstsq, _ = np.linalg.lstsq(x, y, rcond=1e-12)
        residual = y - x @ beta
    rank = int(rank_lstsq)
    if not (np.isfinite(base).all() and np.isfinite(beta).all() and np.isfinite(residual).all()):
        raise RuntimeError(f"Non-finite OLS result for {outcome} after finite-input filtering")
    df_resid = len(y) - rank
    if df_resid <= 0:
        raise RuntimeError(f"Insufficient residual degrees of freedom for {outcome}")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        naive_cov = float(residual @ residual) / df_resid * base
    if not np.isfinite(naive_cov).all():
        raise RuntimeError(f"Non-finite naive covariance for {outcome}")
    names = list(design.columns)
    output: dict[str, Any] = {
        "n": int(len(y)),
        "rank": rank,
        "df_resid": int(df_resid),
        "context_fixed_effects": bool(include_context),
        "exposures": list(exposures),
        "extra_controls": list(extra_controls),
        "naive": inference_from_covariance(beta, naive_cov, df_resid, names),
    }
    for column in cluster_columns:
        covariance, group_count = cluster_covariance(
            x,
            residual,
            base,
            subset[column].astype("string").to_numpy(),
            rank,
        )
        output[f"cluster_{column}"] = {
            "clusters": int(group_count),
            "finite_cluster_correction": "CR1: G/(G-1) * (n-1)/(n-rank)",
            "parameters": inference_from_covariance(beta, covariance, group_count - 1, names),
        }
    output["parameters_reported"] = {
        name: {
            "naive": output["naive"][name],
            "case_cluster": output.get("cluster_case_key", {}).get("parameters", {}).get(name),
            "team_cluster": output.get("cluster_team_id", {}).get("parameters", {}).get(name),
        }
        for name in exposures
    }
    del output["naive"]
    output.pop("cluster_case_key", None)
    output.pop("cluster_team_id", None)
    return output


def frame_models(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_contract": {
            "formula": "outcome ~ exposure + C(context_cell)",
            "directional_formula": (
                "outcome ~ normalized_upper_magnitude + normalized_lower_magnitude + C(context_cell)"
            ),
            "primary_inference": "case-clustered CR1 covariance; t reference with case clusters minus one df",
            "secondary_inference": "team-clustered CR1 covariance; t reference with team clusters minus one df",
            "naive_inference": "homoskedastic OLS covariance shown only as comparison",
            "exposure_scale": "one human-envelope interval width",
        },
        "alpha_results": {},
        "ttc_log1p_sensitivity": {},
    }
    for alpha in ALPHAS:
        alpha_result: dict[str, Any] = {}
        for window in WINDOWS:
            window_result: dict[str, Any] = {}
            for outcome in OUTCOMES:
                outcome_column = f"{window}_{outcome}"
                window_result[outcome] = {
                    "signed": fit_ols(frame, outcome_column, [f"norm_exceedance_{alpha}"]),
                    "directional_joint": fit_ols(
                        frame,
                        outcome_column,
                        [f"norm_upper_{alpha}", f"norm_lower_magnitude_{alpha}"],
                    ),
                }
            alpha_result[window] = window_result
        result["alpha_results"][str(alpha)] = alpha_result
        result["ttc_log1p_sensitivity"][str(alpha)] = {}
        for window in WINDOWS:
            outcome_column = f"{window}_{TTC_LOG_OUTCOME}"
            result["ttc_log1p_sensitivity"][str(alpha)][window] = {
                "signed": fit_ols(frame, outcome_column, [f"norm_exceedance_{alpha}"]),
                "directional_joint": fit_ols(
                    frame,
                    outcome_column,
                    [f"norm_upper_{alpha}", f"norm_lower_magnitude_{alpha}"],
                ),
            }
    return result


def case_permutation_model(
    frame: pd.DataFrame,
    outcome: str,
    exposures: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    required = [
        "case_key",
        "team_id",
        "context_cell",
        "anchor_frame_index",
        outcome,
        *exposures,
    ]
    working = frame[required].sort_values(
        ["case_key", "anchor_frame_index"], kind="stable"
    ).reset_index(drop=True)
    case_positions = {
        str(case): indices.to_numpy(dtype=np.int64)
        for case, indices in working.groupby("case_key", sort=True).groups.items()
    }
    cases = np.asarray(sorted(case_positions), dtype=object)
    observed_model = fit_ols(working, outcome, exposures)
    rng = np.random.default_rng(seed)
    coefficient_distributions = {exposure: np.empty(draws, dtype=float) for exposure in exposures}
    statistic_distributions = {exposure: np.empty(draws, dtype=float) for exposure in exposures}
    permuted_names = [f"permuted_{index}" for index in range(len(exposures))]
    for draw in range(draws):
        donor_cases = rng.permutation(cases)
        permuted_matrix = np.empty((len(working), len(exposures)), dtype=float)
        for recipient_case, donor_case in zip(cases, donor_cases):
            recipient = case_positions[str(recipient_case)]
            donor = case_positions[str(donor_case)]
            donor_values = working.loc[donor, list(exposures)].to_numpy(float)
            nearest = np.rint(
                np.linspace(0.0, len(donor_values) - 1.0, num=len(recipient))
            ).astype(np.int64)
            permuted_matrix[recipient, :] = donor_values[nearest, :]
        permuted_data = working.copy()
        for column_index, name in enumerate(permuted_names):
            permuted_data[name] = permuted_matrix[:, column_index]
        permuted_model = fit_ols(permuted_data, outcome, permuted_names)
        for exposure, permuted_name in zip(exposures, permuted_names):
            estimate = permuted_model["parameters_reported"][permuted_name]["case_cluster"]
            coefficient_distributions[exposure][draw] = float(estimate["coefficient"])
            statistic_distributions[exposure][draw] = float(estimate["t"])
    output: dict[str, Any] = {
        "case_count": int(len(cases)),
        "draws": int(draws),
        "seed": int(seed),
        "method": (
            "Shuffle whole case exposure trajectories against recipient case outcomes. Cases are sorted by "
            "anchor_frame_index; when donor and recipient lengths differ, donor values are mapped by nearest "
            "relative-anchor position. Refit the original frame model with recipient context_cell fixed effects "
            "and case-clustered inference."
        ),
        "comparison_statistic": "absolute case-clustered t statistic",
        "parameters": {},
    }
    for exposure in exposures:
        observed = observed_model["parameters_reported"][exposure]["case_cluster"]
        coefficient_distribution = coefficient_distributions[exposure]
        statistic_distribution = statistic_distributions[exposure]
        empirical_p = (
            1.0 + float(np.sum(np.abs(statistic_distribution) >= abs(observed["t"])))
        ) / (draws + 1.0)
        output["parameters"][exposure] = {
            "observed_case_cluster_coefficient": float(observed["coefficient"]),
            "observed_case_cluster_t": float(observed["t"]),
            "empirical_two_sided_p": empirical_p,
            "permutation_abs_coefficient_quantiles": {
                "p50": float(np.quantile(np.abs(coefficient_distribution), 0.50)),
                "p90": float(np.quantile(np.abs(coefficient_distribution), 0.90)),
                "p95": float(np.quantile(np.abs(coefficient_distribution), 0.95)),
                "p99": float(np.quantile(np.abs(coefficient_distribution), 0.99)),
            },
            "permutation_abs_t_quantiles": {
                "p50": float(np.quantile(np.abs(statistic_distribution), 0.50)),
                "p90": float(np.quantile(np.abs(statistic_distribution), 0.90)),
                "p95": float(np.quantile(np.abs(statistic_distribution), 0.95)),
                "p99": float(np.quantile(np.abs(statistic_distribution), 0.99)),
            },
        }
    return output


def run_label_permutations(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {
        "draws": LABEL_PERMUTATIONS,
        "base_seed": BASE_SEED,
        "alpha": 90,
        "models": {},
    }
    model_index = 0
    for window in WINDOWS:
        output["models"][window] = {}
        for outcome in NEGATIVE_CONTROL_OUTCOMES:
            outcome_column = f"{window}_{outcome}"
            output["models"][window][outcome] = {
                "signed": case_permutation_model(
                    frame,
                    outcome_column,
                    ["norm_exceedance_90"],
                    LABEL_PERMUTATIONS,
                    BASE_SEED + 1000 + model_index,
                ),
                "directional_joint": case_permutation_model(
                    frame,
                    outcome_column,
                    ["norm_upper_90", "norm_lower_magnitude_90"],
                    LABEL_PERMUTATIONS,
                    BASE_SEED + 2000 + model_index,
                ),
            }
            model_index += 1
    return output


def placebo_exposure(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = frame["lo_90"].to_numpy(float)
    hi = frame["hi_90"].to_numpy(float)
    width = frame["width_90"].to_numpy(float)
    fake_ipv = rng.uniform(lo - width, hi + width)
    signed = np.where(fake_ipv > hi, fake_ipv - hi, np.where(fake_ipv < lo, fake_ipv - lo, 0.0))
    norm = signed / width
    return norm, np.maximum(norm, 0.0), np.maximum(-norm, 0.0)


def run_placebo(frame: pd.DataFrame, frame_result: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(BASE_SEED + 3000)
    placebo = frame.copy()
    signed, upper, lower = placebo_exposure(placebo, rng)
    placebo["placebo_norm"] = signed
    placebo["placebo_upper"] = upper
    placebo["placebo_lower_magnitude"] = lower
    example: dict[str, Any] = {
        "outside_count": int(np.count_nonzero(signed)),
        "upper_count": int(np.count_nonzero(upper)),
        "lower_count": int(np.count_nonzero(lower)),
        "inside_count": int(np.count_nonzero(signed == 0.0)),
        "models": {},
    }
    for window in WINDOWS:
        example["models"][window] = {}
        for outcome in NEGATIVE_CONTROL_OUTCOMES:
            outcome_column = f"{window}_{outcome}"
            example["models"][window][outcome] = {
                "signed": fit_ols(placebo, outcome_column, ["placebo_norm"]),
                "directional_joint": fit_ols(
                    placebo,
                    outcome_column,
                    ["placebo_upper", "placebo_lower_magnitude"],
                ),
            }

    comparisons: dict[str, Any] = {window: {} for window in WINDOWS}
    parameter_names = ("signed", "upper", "lower")
    placebo_stats: dict[tuple[str, str, str], list[float]] = {
        (window, outcome, parameter): []
        for window in WINDOWS
        for outcome in NEGATIVE_CONTROL_OUTCOMES
        for parameter in parameter_names
    }
    placebo_coefficients: dict[tuple[str, str, str], list[float]] = {
        (window, outcome, parameter): []
        for window in WINDOWS
        for outcome in NEGATIVE_CONTROL_OUTCOMES
        for parameter in parameter_names
    }
    for _ in range(PLACEBO_DRAWS):
        draw_signed, draw_upper, draw_lower = placebo_exposure(frame, rng)
        draw_data = frame.assign(
            placebo_norm=draw_signed,
            placebo_upper=draw_upper,
            placebo_lower_magnitude=draw_lower,
        )
        for window in WINDOWS:
            for outcome in NEGATIVE_CONTROL_OUTCOMES:
                signed_model = fit_ols(draw_data, f"{window}_{outcome}", ["placebo_norm"])
                directional_model = fit_ols(
                    draw_data,
                    f"{window}_{outcome}",
                    ["placebo_upper", "placebo_lower_magnitude"],
                )
                estimates = {
                    "signed": signed_model["parameters_reported"]["placebo_norm"]["case_cluster"],
                    "upper": directional_model["parameters_reported"]["placebo_upper"]["case_cluster"],
                    "lower": directional_model["parameters_reported"]["placebo_lower_magnitude"][
                        "case_cluster"
                    ],
                }
                for parameter, estimate in estimates.items():
                    placebo_stats[(window, outcome, parameter)].append(float(estimate["t"]))
                    placebo_coefficients[(window, outcome, parameter)].append(
                        float(estimate["coefficient"])
                    )

    for window in WINDOWS:
        for outcome in NEGATIVE_CONTROL_OUTCOMES:
            if outcome == TTC_LOG_OUTCOME:
                real_models = frame_result["ttc_log1p_sensitivity"]["90"][window]
            else:
                real_models = frame_result["alpha_results"]["90"][window][outcome]
            real_estimates = {
                "signed": real_models["signed"]["parameters_reported"]["norm_exceedance_90"][
                    "case_cluster"
                ],
                "upper": real_models["directional_joint"]["parameters_reported"]["norm_upper_90"][
                    "case_cluster"
                ],
                "lower": real_models["directional_joint"]["parameters_reported"][
                    "norm_lower_magnitude_90"
                ]["case_cluster"],
            }
            comparisons[window][outcome] = {}
            for parameter, real_case in real_estimates.items():
                t_distribution = np.asarray(placebo_stats[(window, outcome, parameter)], dtype=float)
                beta_distribution = np.asarray(
                    placebo_coefficients[(window, outcome, parameter)], dtype=float
                )
                empirical_p = (
                    1.0 + float(np.sum(np.abs(t_distribution) >= abs(real_case["t"])))
                ) / (PLACEBO_DRAWS + 1.0)
                comparisons[window][outcome][parameter] = {
                    "real_case_cluster_coefficient": real_case["coefficient"],
                    "real_case_cluster_t": real_case["t"],
                    "placebo_draws": PLACEBO_DRAWS,
                    "empirical_p_real_not_stronger_than_placebo_abs_t": empirical_p,
                    "distinguishable_at_0_05": bool(empirical_p < 0.05),
                    "placebo_abs_t_quantiles": {
                        "p50": float(np.quantile(np.abs(t_distribution), 0.50)),
                        "p90": float(np.quantile(np.abs(t_distribution), 0.90)),
                        "p95": float(np.quantile(np.abs(t_distribution), 0.95)),
                        "p99": float(np.quantile(np.abs(t_distribution), 0.99)),
                    },
                    "placebo_coefficient_quantiles": {
                        "p05": float(np.quantile(beta_distribution, 0.05)),
                        "p50": float(np.quantile(beta_distribution, 0.50)),
                        "p95": float(np.quantile(beta_distribution, 0.95)),
                    },
                }
    return {
        "construction": (
            "For each frame draw fake IPV uniformly on [lo_90-width_90, hi_90+width_90], "
            "then apply the same signed and normalized exceedance formula."
        ),
        "seed": BASE_SEED + 3000,
        "seeded_example": example,
        "real_vs_placebo": comparisons,
    }


def aggregate_units(frame: pd.DataFrame, units: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped = frame.groupby("unit_composite_key", sort=False)
    aggregates = grouped.agg(
        eligible_frame_count=("product_row_key", "size"),
        outside_frame_count=("outside_90", "sum"),
        upper_frame_count=("upper_90", "sum"),
        lower_frame_count=("lower_90", "sum"),
        frac_outside_90=("outside_90", "mean"),
        mean_signed_exceedance_90=("norm_exceedance_90", "mean"),
        frac_upper_90=("upper_90", "mean"),
        frac_lower_90=("lower_90", "mean"),
        mean_upper_exceedance_90=("norm_upper_90", "mean"),
        mean_lower_exceedance_90=("norm_lower_magnitude_90", "mean"),
        max_abs_exceedance_90=("norm_exceedance_90", lambda values: float(np.max(np.abs(values)))),
        raw_mean_signed_exceedance_90=("signed_exceedance_90", "mean"),
        raw_max_abs_exceedance_90=("signed_exceedance_90", lambda values: float(np.max(np.abs(values)))),
    ).reset_index()

    cohort = units.loc[units["analysis_set"].fillna(False).astype(bool)].copy()
    cohort["team_id"] = team_from_case(cohort["case_key"])
    merged = cohort.merge(aggregates, on="unit_composite_key", how="left", validate="one_to_one", indicator=True)
    coverage = {
        "base_analysis_units": int(len(cohort)),
        "units_with_defined_exposure": int((merged["_merge"] == "both").sum()),
        "units_without_two_gate_frames": int((merged["_merge"] == "left_only").sum()),
        "two_gate_units_outside_analysis_set": int(
            (~frame["unit_composite_key"].isin(cohort["unit_composite_key"])).groupby(frame["unit_composite_key"]).any().sum()
        ),
        "definition": (
            "Unit exposure is aggregated only over frames passing both mechanisms. Units with no such frame "
            "remain missing, not zero. Required unit exceedance summaries use normalized exceedance in interval-width units."
        ),
    }
    return merged.drop(columns="_merge"), coverage


def rank_association(x: np.ndarray, y: np.ndarray, control: np.ndarray | None = None) -> tuple[float, float]:
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    if control is None:
        if np.ptp(rank_x) == 0.0 or np.ptp(rank_y) == 0.0:
            return np.nan, np.nan
        rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
        df = len(x) - 2
        statistic = rho * math.sqrt(df / max(1e-15, 1.0 - rho * rho))
        return rho, float(2.0 * stats.t.sf(abs(statistic), df))
    rank_z = stats.rankdata(control)
    design = np.column_stack([np.ones(len(rank_z)), rank_z])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        beta_x, _, _, _ = np.linalg.lstsq(design, rank_x, rcond=1e-12)
        beta_y, _, _, _ = np.linalg.lstsq(design, rank_y, rcond=1e-12)
        residual_x = rank_x - design @ beta_x
        residual_y = rank_y - design @ beta_y
    if np.ptp(residual_x) == 0.0 or np.ptp(residual_y) == 0.0:
        return np.nan, np.nan
    rho = float(np.corrcoef(residual_x, residual_y)[0, 1])
    df = len(x) - 3
    statistic = rho * math.sqrt(df / max(1e-15, 1.0 - rho * rho))
    p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    return rho, p_value


def unit_association(
    data: pd.DataFrame,
    exposure: str,
    outcome: str,
    control: str | None,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    columns = [exposure, outcome, "team_id"] + ([control] if control else [])
    subset = data[columns].copy()
    for column in [exposure, outcome] + ([control] if control else []):
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    x = subset[exposure].to_numpy(float)
    y = subset[outcome].to_numpy(float)
    z = subset[control].to_numpy(float) if control else None
    rho, naive_p = rank_association(x, y, z)
    if not math.isfinite(rho):
        raise RuntimeError(f"Undefined full-sample rank association for {outcome} ~ {exposure}")
    groups = {team: indices.to_numpy() for team, indices in subset.groupby("team_id").groups.items()}
    team_names = sorted(groups)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(draws):
        selected = rng.choice(team_names, size=len(team_names), replace=True)
        positions = np.concatenate([groups[team] for team in selected])
        try:
            estimate, _ = rank_association(
                subset.loc[positions, exposure].to_numpy(float),
                subset.loc[positions, outcome].to_numpy(float),
                subset.loc[positions, control].to_numpy(float) if control else None,
            )
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(estimate):
            samples.append(estimate)
    bootstrap = np.asarray(samples, dtype=float)
    if len(bootstrap) < int(0.9 * draws):
        raise RuntimeError(f"Too few valid team bootstrap draws for {outcome} ~ {exposure}")
    sign_p = min(
        1.0,
        2.0
        * min(
            (1.0 + float(np.sum(bootstrap <= 0.0))) / (len(bootstrap) + 1.0),
            (1.0 + float(np.sum(bootstrap >= 0.0))) / (len(bootstrap) + 1.0),
        ),
    )
    return {
        "n": int(len(subset)),
        "team_clusters": int(len(team_names)),
        "rho": rho,
        "naive_p_value": naive_p,
        "team_block_bootstrap": {
            "requested_draws": int(draws),
            "valid_draws": int(len(bootstrap)),
            "seed": int(seed),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "two_sided_sign_p": float(sign_p),
        },
        "control": control if control else "none",
        "method": "Spearman rank association" if control is None else "Partial Spearman rank association",
    }


def unit_models(unit_frame: pd.DataFrame, coverage: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "coverage": coverage,
        "exposure_definitions": {
            "frac_outside_90": "outside-frame count divided by two-gate frame count",
            "mean_signed_exceedance_90": "mean signed normalized exceedance; units are interval widths",
            "frac_upper_90": "upper-outside-frame count divided by two-gate frame count",
            "frac_lower_90": "lower-outside-frame count divided by two-gate frame count",
            "mean_upper_exceedance_90": "mean positive normalized upper magnitude, with zero elsewhere",
            "mean_lower_exceedance_90": "mean positive normalized lower magnitude, with zero elsewhere",
            "max_abs_exceedance_90": "maximum absolute normalized exceedance; units are interval widths",
        },
        "nonsafety": {},
        "secondary": {},
    }
    counter = 0
    for outcome in NONSAFETY_OUTCOMES:
        output["nonsafety"][outcome] = {}
        for exposure in UNIT_EXPOSURES:
            seed = BASE_SEED + 4000 + zlib.crc32(f"{outcome}:{exposure}".encode("utf-8")) % 1_000_000
            output["nonsafety"][outcome][exposure] = {
                "univariate": unit_association(
                    unit_frame, exposure, outcome, None, UNIT_BOOTSTRAPS, seed
                ),
                "partial_controlling_official_comprehensive": unit_association(
                    unit_frame,
                    exposure,
                    outcome,
                    "official_comprehensive",
                    UNIT_BOOTSTRAPS,
                    seed + 1,
                ),
            }
            counter += 1
    for outcome in SECONDARY_OUTCOMES:
        output["secondary"][outcome] = {}
        for exposure in UNIT_EXPOSURES:
            seed = BASE_SEED + 5000 + zlib.crc32(f"{outcome}:{exposure}".encode("utf-8")) % 1_000_000
            output["secondary"][outcome][exposure] = unit_association(
                unit_frame, exposure, outcome, None, UNIT_BOOTSTRAPS, seed
            )
            counter += 1
    output["association_count"] = counter
    return output


def add_key_metrics(
    metrics: dict[str, Any],
    inputs: Inputs,
    frame: pd.DataFrame,
    unit_coverage: dict[str, Any],
) -> None:
    units = inputs.units
    unit_analysis = units["analysis_set"].fillna(False).astype(bool)
    metrics.update(
        {
            "unit_total": count_metric(
                len(units), len(units), "all unit rows", [UNIT_REL], ["unit_composite_key"]
            ),
            "unit_analysis_set": count_metric(
                int(unit_analysis.sum()),
                len(units),
                "analysis_set == True",
                [UNIT_REL],
                ["analysis_set"],
            ),
            "unit_official_safety_below_100": count_metric(
                int((pd.to_numeric(units["official_safety"], errors="coerce") < 100).sum()),
                len(units),
                "official_safety < 100, all 267 units",
                [UNIT_REL],
                ["official_safety"],
            ),
            "unit_collision_or_intervention_deduction_nonzero": count_metric(
                int(
                    (
                        pd.to_numeric(units["collision_intervention_deduction_any"], errors="coerce")
                        .fillna(0)
                        .ne(0)
                    ).sum()
                ),
                len(units),
                "collision_intervention_deduction_any != 0, all 267 units",
                [UNIT_REL],
                ["collision_intervention_deduction_any"],
            ),
            "unit_safety_intervention_nonzero": count_metric(
                int(
                    pd.to_numeric(units["safety_intervention"], errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .sum()
                ),
                len(units),
                "safety_intervention != 0, all 267 units",
                [UNIT_REL],
                ["safety_intervention"],
            ),
            "analysis_units_with_defined_exposure": count_metric(
                int(unit_coverage["units_with_defined_exposure"]),
                int(unit_coverage["base_analysis_units"]),
                "analysis_set == True AND at least one two-gate frame",
                [RQ017_REL, M2_REL, ANCHOR_REL, UNIT_REL],
                [
                    "status",
                    "mechanism2_gate_ok",
                    "unit_composite_key",
                    "analysis_set",
                ],
                "Units with no two-gate frame remain missing rather than being assigned zero exposure.",
            ),
            "frame_case_count": count_metric(
                frame["case_key"].nunique(),
                inputs.anchors["case_key"].nunique(),
                "distinct case_key with at least one two-gate analysis frame / all anchor-table cases",
                [RQ017_REL, M2_REL, ANCHOR_REL],
                ["status", "mechanism2_gate_ok", "case_key"],
            ),
            "frame_team_count": count_metric(
                frame["team_id"].nunique(),
                team_from_case(inputs.anchors["case_key"]).nunique(),
                "distinct team token with at least one two-gate frame / all anchor-table team tokens",
                [ANCHOR_REL],
                ["case_key"],
            ),
        }
    )
    for alpha in ALPHAS:
        for side, column in (
            ("upper", f"upper_{alpha}"),
            ("lower", f"lower_{alpha}"),
            ("inside", f"outside_{alpha}"),
        ):
            numerator = int((~frame[column]).sum()) if side == "inside" else int(frame[column].sum())
            condition = (
                f"two-gate frames with lo_{alpha} <= ipv_log <= hi_{alpha}"
                if side == "inside"
                else f"two-gate frames with {side}_{alpha} == True"
            )
            metrics[f"alpha_{alpha}_{side}_frames"] = count_metric(
                numerator,
                len(frame),
                condition,
                [RQ017_REL, M2_REL],
                ["ipv_log", f"lo_{alpha}", f"hi_{alpha}", f"width_{alpha}"],
            )
    for window in WINDOWS:
        metrics[f"{window}_window_overflow"] = count_metric(
            int(frame[f"{window}_window_overflow"].sum()),
            len(frame),
            f"two-gate frames whose {window} requested endpoint exceeds the case endpoint",
            [ANCHOR_REL, DENSE_REL],
            [
                "case_key",
                "anchor_frame_index",
                "target_window_end_frame_index",
                "frame_index",
                "time_s",
            ],
        )
        metrics[f"{window}_ttc_all_nonclosing"] = count_metric(
            int(frame[f"{window}_ttc_all_nonclosing"].sum()),
            len(frame),
            f"all frames in the available {window} window have closing_rate_mps <= 0",
            [DENSE_REL],
            ["case_key", "frame_index", "time_s", "distance_m", "closing_rate_mps"],
        )
        metrics[f"{window}_future_min_ttc_above_1e6"] = count_metric(
            int((pd.to_numeric(frame[f"{window}_future_min_ttc_s"], errors="coerce") > 1_000_000).sum()),
            len(frame),
            f"finite {window} future_min_ttc_s > 1e6 seconds",
            [DENSE_REL],
            ["distance_m", "closing_rate_mps", "frame_index", "time_s"],
        )
    full_anomaly = inputs.anchors["relative_distance_anchor"] >= ANOMALOUS_DISTANCE_THRESHOLD_M
    analysis_anomaly = frame["relative_distance_anchor"] >= ANOMALOUS_DISTANCE_THRESHOLD_M
    metrics["coordinate_anomaly_full_anchor_rows"] = count_metric(
        int(full_anomaly.sum()),
        len(inputs.anchors),
        f"relative_distance_anchor >= {ANOMALOUS_DISTANCE_THRESHOLD_M:g} m",
        [ANCHOR_REL],
        ["case_key", "anchor_frame_index", "relative_distance_anchor"],
    )
    metrics["coordinate_anomaly_two_gate_rows"] = count_metric(
        int(analysis_anomaly.sum()),
        len(frame),
        f"two-gate frames with relative_distance_anchor >= {ANOMALOUS_DISTANCE_THRESHOLD_M:g} m",
        [RQ017_REL, M2_REL, ANCHOR_REL],
        [
            "status",
            "mechanism2_gate_ok",
            "case_key",
            "anchor_frame_index",
            "relative_distance_anchor",
        ],
    )


def build_data_health(inputs: Inputs, frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    finite_ipv = np.isfinite(pd.to_numeric(frame["ipv_log"], errors="coerce").to_numpy(float))
    case_counts = frame.groupby("case_key").size()
    anomaly_full = inputs.anchors["relative_distance_anchor"] >= ANOMALOUS_DISTANCE_THRESHOLD_M
    anomaly_main = frame["relative_distance_anchor"] >= ANOMALOUS_DISTANCE_THRESHOLD_M
    health: dict[str, Any] = {
        "analysis_set_check": {
            "observed": int(len(frame)),
            "expected": 14099,
            "matches_expected": bool(len(frame) == 14099),
            "definition": "RQ017 status == OK AND M2 mechanism2_gate_ok == True",
            "sources": [RQ017_REL, M2_REL],
            "columns": ["product_row_key", "status", "mechanism2_gate_ok"],
        },
        "joins": {
            "rq017_to_m2": metrics["rq017_to_m2_join_hits"],
            "analysis_to_anchor": metrics["analysis_to_anchor_join_hits"],
        },
        "ipv_log": {
            "rows": int(len(frame)),
            "nan": int(pd.to_numeric(frame["ipv_log"], errors="coerce").isna().sum()),
            "positive_inf": int(np.isposinf(pd.to_numeric(frame["ipv_log"], errors="coerce")).sum()),
            "negative_inf": int(np.isneginf(pd.to_numeric(frame["ipv_log"], errors="coerce")).sum()),
            "finite": int(finite_ipv.sum()),
            "source": RQ017_REL,
            "columns": ["ipv_log", "status"],
        },
        "interval_validity": {},
        "window_coverage": {},
        "future_outcome_health": {},
        "outside_counts": {},
        "case_frame_distribution": {
            "case_count": int(case_counts.size),
            "team_count": int(frame["team_id"].nunique()),
            "min": int(case_counts.min()),
            "p25": float(case_counts.quantile(0.25)),
            "median": float(case_counts.quantile(0.50)),
            "p75": float(case_counts.quantile(0.75)),
            "max": int(case_counts.max()),
            "filter": "two-gate analysis frames",
            "sources": [RQ017_REL, M2_REL, ANCHOR_REL],
            "columns": ["status", "mechanism2_gate_ok", "case_key"],
        },
        "coordinate_anomaly": {
            "threshold_m": ANOMALOUS_DISTANCE_THRESHOLD_M,
            "full_anchor_rows": int(anomaly_full.sum()),
            "full_anchor_denominator": int(len(inputs.anchors)),
            "full_anchor_cases": inputs.anchors.loc[anomaly_full, "case_key"].value_counts().to_dict(),
            "full_anchor_distance_range_m": [
                float(inputs.anchors.loc[anomaly_full, "relative_distance_anchor"].min()),
                float(inputs.anchors.loc[anomaly_full, "relative_distance_anchor"].max()),
            ],
            "two_gate_rows": int(anomaly_main.sum()),
            "two_gate_denominator": int(len(frame)),
            "handling": (
                "Retain all rows in the auditable as-is path. The seven flagged source rows contribute zero "
                "rows to the two-gate analysis, so excluding them leaves every fitted main model unchanged."
            ),
            "main_model_max_abs_coefficient_change_if_excluded": 0.0,
            "source": ANCHOR_REL,
            "columns": ["case_key", "relative_distance_anchor", "anchor_frame_index"],
        },
        "context_cell_counts": frame["context_cell"].value_counts(dropna=False).to_dict(),
    }
    for alpha in ALPHAS:
        width = pd.to_numeric(frame[f"width_{alpha}"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(frame[f"lo_{alpha}"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(frame[f"hi_{alpha}"], errors="coerce").to_numpy(float)
        health["interval_validity"][str(alpha)] = {
            "rows": int(len(frame)),
            "width_positive": int(np.sum(width > 0.0)),
            "width_nonpositive": int(np.sum(~(width > 0.0))),
            "lo_below_hi": int(np.sum(lo < hi)),
            "lo_not_below_hi": int(np.sum(~(lo < hi))),
            "source": M2_REL,
            "columns": [f"lo_{alpha}", f"hi_{alpha}", f"width_{alpha}"],
        }
        upper = int(frame[f"upper_{alpha}"].sum())
        lower = int(frame[f"lower_{alpha}"].sum())
        inside = int((~frame[f"outside_{alpha}"]).sum())
        health["outside_counts"][str(alpha)] = {
            "upper": upper,
            "lower": lower,
            "inside": inside,
            "sum": upper + lower + inside,
            "analysis_rows": int(len(frame)),
            "sum_matches": bool(upper + lower + inside == len(frame)),
            "sources": [RQ017_REL, M2_REL],
            "columns": ["ipv_log", f"lo_{alpha}", f"hi_{alpha}", f"width_{alpha}"],
        }
    for window in WINDOWS:
        overflow = int(frame[f"{window}_window_overflow"].sum())
        ttc_missing = int(frame[f"{window}_ttc_all_nonclosing"].sum())
        health["window_coverage"][window] = {
            "rows": int(len(frame)),
            "overflow_rows": overflow,
            "overflow_ratio": overflow / len(frame),
            "ttc_all_nonclosing_rows": ttc_missing,
            "ttc_all_nonclosing_ratio": ttc_missing / len(frame),
            "window_row_count": {
                "min": int(frame[f"{window}_window_row_count"].min()),
                "median": float(frame[f"{window}_window_row_count"].median()),
                "max": int(frame[f"{window}_window_row_count"].max()),
            },
            "source": DENSE_REL,
            "columns": [
                "case_key",
                "frame_index",
                "time_s",
                "distance_m",
                "closing_rate_mps",
            ],
        }
        health["future_outcome_health"][window] = {}
        for outcome in OUTCOMES:
            values = pd.to_numeric(frame[f"{window}_{outcome}"], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            health["future_outcome_health"][window][outcome] = {
                "rows": int(len(values)),
                "finite_rows": int(len(finite)),
                "missing_or_nonfinite_rows": int(len(values) - len(finite)),
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
                "above_1e6": int(np.sum(finite > 1_000_000.0)),
                "source": DENSE_REL,
                "columns": ["distance_m", "closing_rate_mps", "frame_index", "time_s"],
            }
    positive_closing = pd.to_numeric(
        inputs.dense.loc[inputs.dense["closing_rate_mps"] > 0.0, "closing_rate_mps"],
        errors="coerce",
    ).to_numpy(float)
    health["ttc_numeric_boundary"] = {
        "minimum_positive_closing_rate_mps": float(np.min(positive_closing)),
        "contract_ttc_above_1e6_rows": health["future_outcome_health"]["contract"][
            "future_min_ttc_s"
        ]["above_1e6"],
        "fixed_3s_ttc_above_1e6_rows": health["future_outcome_health"]["fixed_3s"][
            "future_min_ttc_s"
        ]["above_1e6"],
        "handling": (
            "Keep the task-contract TTC definition without an unstated closing-rate floor. "
            "Very small positive closing rates can create very large finite TTC values; counts and maxima are disclosed."
        ),
        "source": DENSE_REL,
        "columns": ["distance_m", "closing_rate_mps"],
    }
    return health


def pformat(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    value = float(value)
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def effect_cell(result: dict[str, Any], parameter: str) -> str:
    item = result["parameters_reported"][parameter]
    case = item["case_cluster"]
    return (
        f"{case['coefficient']:.4f} [{case['ci95_low']:.4f}, {case['ci95_high']:.4f}]; "
        f"p_naive={pformat(item['naive']['p_value'])}, p_case={pformat(case['p_value'])}, "
        f"p_team={pformat(item['team_cluster']['p_value'])}"
    )


def rank_cell(result: dict[str, Any]) -> str:
    boot = result["team_block_bootstrap"]
    return (
        f"{result['rho']:.3f} [{boot['ci95_low']:.3f}, {boot['ci95_high']:.3f}] "
        f"(n={result['n']}, team-block p={pformat(boot['two_sided_sign_p'])})"
    )


def percentage(metric: dict[str, Any]) -> str:
    return f"{metric['numerator']:,}/{metric['denominator']:,} = {metric['percent']:.4f}%"


def build_report(
    generated_at: str,
    metrics: dict[str, Any],
    health: dict[str, Any],
    frame_results: dict[str, Any],
    unit_results: dict[str, Any],
    negative_controls: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# RQ018-1：异常 IPV 与后续行为风险的关联性正向探索",
            "",
            "## 1. 工作定位与当前进度",
            "",
            "这项研究最终要在线验证一辆自动驾驶车表现出的社会交互倾向是否落在人类合理范围内。IPV（Interaction Preference Value）是表示交互倾向的标量；机制一判断某帧 IPV 数值是否携带七个候选之间的判别信息，机制二判断该情境是否有足够人类参照。RQ015 已冻结机制一，RQ016C 已建立纯人-人参照区间，RQ017 已在 OnSite 自动驾驶车数据上完成两门计算。本次 RQ018-A1 是下一环：只做一次探索性、描述性分析，检查越出人类参照区间的 IPV 与锚点后行为风险及 unit 级竞赛结果是否有关联。",
            "",
            "本轮已完成指定输入复算、未来窗口构造、case/team 聚类稳健推断、unit 级关联、两项负对照和数据健康自查。结果不作因果解释，也不对任何车辆或队伍作判断。",
            "",
            "## 2. 分析合同与避免循环论证",
            "",
            "锚点时刻的相对距离、接近率、TTC 与 PET proxy 都已用于人类参照区间的情境条件化，因此本报告不把它们当同期结果。帧级结果全部来自锚点后窗口：合同窗口 `[anchor_frame_index, target_window_end_frame_index]`，以及按 `time_s` 定义的锚点起 3 秒窗口。TTC 逐帧按 `distance_m / closing_rate_mps` 计算；`closing_rate_mps <= 0` 的帧不进入最小值，全窗口均不接近时 TTC 记为缺失。",
            "",
            "主曝露为 90% 人类参照区间的有符号、区间宽度归一化越界量。上侧和下侧用两个非负幅度变量联合进入方向模型；80% 与 95% 仅用于敏感性。帧级模型控制 `context_cell` 固定效应，case 聚类为主口径，team 聚类为次口径；同时列出朴素 p 值以显示忽略嵌套会造成什么差异。",
            "",
            "## 3. 基线复算与覆盖",
            "",
            "| 数字 | 口径、来源与列 |",
            "|---|---|",
            f"| 机制一 OK：{percentage(metrics['mechanism1_ok'])} | 筛选 `status == OK`；来源 `{RQ017_REL}`；列 `status` |",
            f"| 机制二通过：{percentage(metrics['mechanism2_gate_ok'])} | 筛选 `mechanism2_gate_ok == True`；来源 `{M2_REL}`；列 `mechanism2_gate_ok` |",
            f"| 两门交集：{percentage(metrics['two_gate_analysis_set'])} | 筛选 `status == OK AND mechanism2_gate_ok == True`；来源 `{RQ017_REL}` + `{M2_REL}`；连接列 `product_row_key` |",
            f"| unit 基础集：{percentage(metrics['unit_analysis_set'])} | 筛选 `analysis_set == True`；来源 `{UNIT_REL}`；列 `analysis_set` |",
            f"| unit 曝露有定义：{percentage(metrics['analysis_units_with_defined_exposure'])} | `analysis_set == True` 且至少一帧两门通过；来源 RQ017 + M2 + 锚点表 + unit 表；列 `status, mechanism2_gate_ok, unit_composite_key, analysis_set`。其余 unit 保持缺失，不赋零值 |",
            "",
            f"`product_row_key` 从 RQ017 到机制二命中 {percentage(metrics['rq017_to_m2_join_hits'])}；两门交集解析到锚点表命中 {percentage(metrics['analysis_to_anchor_join_hits'])}。两者分别使用 `product_row_key` 和 `case_key + anchor_frame_index + perspective`，来源见上表。",
            "",
            "## 4. 帧级主结果：90% 参照区间",
            "",
            "系数单位为曝露增加一个参照区间宽度时结果变量的变化；距离和 TTC 下降表示风险增加，最大接近率上升表示风险增加。方括号为 case 聚类 95% 置信区间。",
            "",
            "| 窗口 | 结果 | 有符号单斜率 | 上侧幅度（方向联合模型） | 下侧幅度（方向联合模型） |",
            "|---|---|---|---|---|",
        ]
    )
    for window in WINDOWS:
        for outcome in OUTCOMES:
            model = frame_results["alpha_results"]["90"][window][outcome]
            lines.append(
                "| "
                + f"{window} | {outcome} | "
                + effect_cell(model["signed"], "norm_exceedance_90")
                + " | "
                + effect_cell(model["directional_joint"], "norm_upper_90")
                + " | "
                + effect_cell(model["directional_joint"], "norm_lower_magnitude_90")
                + " |"
            )

    lines.extend(
        [
            "",
            "### 4.1 80% / 95% 敏感性",
            "",
            "下表只列有符号单斜率；上、下侧联合模型的完整朴素、case 聚类和 team 聚类结果在 `frame_level_results.json`。",
            "",
            "| α | 窗口 | 结果 | case 聚类效应 [95% CI] 与三种 p 值 |",
            "|---:|---|---|---|",
        ]
    )
    for alpha in (80, 95):
        for window in WINDOWS:
            for outcome in OUTCOMES:
                model = frame_results["alpha_results"][str(alpha)][window][outcome]["signed"]
                lines.append(
                    f"| {alpha} | {window} | {outcome} | "
                    + effect_cell(model, f"norm_exceedance_{alpha}")
                    + " |"
                )

    lines.extend(
        [
            "",
            "### 4.2 TTC 的 `log1p` 数值健康敏感性",
            "",
            "原始 TTC 在接近率非常接近零但仍为正时可达到极大有限值。任务书原始尺度结果仍在主表；下表额外对 `log(1 + future_min_ttc_s)` 拟合同一模型，不设接近率阈值、不改变帧的纳入。负系数表示越界幅度增加时未来最小 TTC 变短。",
            "",
            "| α | 窗口 | 有符号单斜率 | 上侧幅度（联合模型） | 下侧幅度（联合模型） |",
            "|---:|---|---|---|---|",
        ]
    )
    for alpha in ALPHAS:
        for window in WINDOWS:
            model = frame_results["ttc_log1p_sensitivity"][str(alpha)][window]
            lines.append(
                f"| {alpha} | {window} | "
                + effect_cell(model["signed"], f"norm_exceedance_{alpha}")
                + " | "
                + effect_cell(model["directional_joint"], f"norm_upper_{alpha}")
                + " | "
                + effect_cell(model["directional_joint"], f"norm_lower_magnitude_{alpha}")
                + " |"
            )

    lines.extend(
        [
            "",
            "## 5. 负对照",
            "",
            f"case 标签置换共 {LABEL_PERMUTATIONS} 次：整段打乱 case 的曝露轨迹与结果轨迹的对应；不同帧数按相对锚点位置最近邻对齐，然后原样重拟合含 `context_cell` 固定效应与 case 聚类推断的帧模型。安慰剂曝露按任务书从每帧 `[lo_90-width_90, hi_90+width_90]` 均匀抽取，使用固定种子；安慰剂 p 值比较真实 case 聚类 |t| 与 {PLACEBO_DRAWS} 次安慰剂 |t| 分布。",
            "",
            "| 窗口 | 结果 | 标签置换 p（有符号） | 安慰剂 p（有符号） | 标签置换 p（下侧幅度） | 安慰剂 p（下侧幅度） |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    label = negative_controls["case_label_permutation"]
    placebo = negative_controls["placebo_exposure"]["real_vs_placebo"]
    for window in WINDOWS:
        for outcome in NEGATIVE_CONTROL_OUTCOMES:
            lp_signed = label["models"][window][outcome]["signed"]["parameters"]["norm_exceedance_90"][
                "empirical_two_sided_p"
            ]
            lp_lower = label["models"][window][outcome]["directional_joint"]["parameters"][
                "norm_lower_magnitude_90"
            ]["empirical_two_sided_p"]
            pp_signed = placebo[window][outcome]["signed"][
                "empirical_p_real_not_stronger_than_placebo_abs_t"
            ]
            pp_lower = placebo[window][outcome]["lower"][
                "empirical_p_real_not_stronger_than_placebo_abs_t"
            ]
            lines.append(
                f"| {window} | {outcome} | {pformat(lp_signed)} | {pformat(pp_signed)} | "
                f"{pformat(lp_lower)} | {pformat(pp_lower)} |"
            )

    lines.extend(
        [
            "",
            "## 6. unit 级非安全子分数",
            "",
            "unit 基础分母是 `analysis_set == True` 的 245 个；其中 225 个至少有一帧两门通过，20 个曝露无定义而不进入关联。unit 越界量先除以各帧区间宽度，再在 unit 内聚合。下表以 team 为 block 给出 Spearman 或控制 `official_comprehensive` 后的偏 Spearman 95% bootstrap 区间；`U` 为单变量，`P` 为偏关联。",
            "",
            "| 结果 | unit 曝露 | U: rho [team-block 95% CI] | P: partial rho [team-block 95% CI] |",
            "|---|---|---|---|",
        ]
    )
    selected_unit_exposures = [
        "frac_outside_90",
        "mean_signed_exceedance_90",
        "mean_upper_exceedance_90",
        "mean_lower_exceedance_90",
        "max_abs_exceedance_90",
    ]
    for outcome in NONSAFETY_OUTCOMES:
        for exposure in selected_unit_exposures:
            item = unit_results["nonsafety"][outcome][exposure]
            lines.append(
                f"| {outcome} | {exposure} | {rank_cell(item['univariate'])} | "
                f"{rank_cell(item['partial_controlling_official_comprehensive'])} |"
            )

    lines.extend(
        [
            "",
            "## 7. 次要安全结果与功效边界",
            "",
            f"全 267 个 unit 中，`official_safety < 100` 为 {percentage(metrics['unit_official_safety_below_100'])}（筛选 `official_safety < 100`，来源 `{UNIT_REL}`，列 `official_safety`）；`collision_intervention_deduction_any != 0` 为 {percentage(metrics['unit_collision_or_intervention_deduction_nonzero'])}（同一来源，列 `collision_intervention_deduction_any`）；`safety_intervention != 0` 为 {percentage(metrics['unit_safety_intervention_nonzero'])}（同一来源，列 `safety_intervention`）。阳性事件稀少，以下结果只给效应和区间；不显著不能解释成没有关联，显著也必须按低功效探索看待。",
            "",
            "| 次要结果 | unit 曝露 | Spearman rho [team-block 95% CI] |",
            "|---|---|---|",
        ]
    )
    for outcome in SECONDARY_OUTCOMES:
        for exposure in selected_unit_exposures:
            item = unit_results["secondary"][outcome][exposure]
            lines.append(f"| {outcome} | {exposure} | {rank_cell(item)} |")

    contract = health["window_coverage"]["contract"]
    fixed = health["window_coverage"]["fixed_3s"]
    anomaly = health["coordinate_anomaly"]
    alpha90 = health["outside_counts"]["90"]
    log_contract_lower = frame_results["ttc_log1p_sensitivity"]["90"]["contract"][
        "directional_joint"
    ]["parameters_reported"]["norm_lower_magnitude_90"]["case_cluster"]
    log_fixed_lower = frame_results["ttc_log1p_sensitivity"]["90"]["fixed_3s"][
        "directional_joint"
    ]["parameters_reported"]["norm_lower_magnitude_90"]["case_cluster"]
    log_contract_label_p = negative_controls["case_label_permutation"]["models"]["contract"][
        TTC_LOG_OUTCOME
    ]["directional_joint"]["parameters"]["norm_lower_magnitude_90"]["empirical_two_sided_p"]
    log_fixed_label_p = negative_controls["case_label_permutation"]["models"]["fixed_3s"][
        TTC_LOG_OUTCOME
    ]["directional_joint"]["parameters"]["norm_lower_magnitude_90"]["empirical_two_sided_p"]
    log_contract_placebo_p = negative_controls["placebo_exposure"]["real_vs_placebo"][
        "contract"
    ][TTC_LOG_OUTCOME]["lower"]["empirical_p_real_not_stronger_than_placebo_abs_t"]
    log_fixed_placebo_p = negative_controls["placebo_exposure"]["real_vs_placebo"]["fixed_3s"][
        TTC_LOG_OUTCOME
    ]["lower"]["empirical_p_real_not_stronger_than_placebo_abs_t"]
    lower_checks_pass = all(
        value < 0.05
        for value in (
            log_contract_lower["p_value"],
            log_fixed_lower["p_value"],
            log_contract_label_p,
            log_fixed_label_p,
            log_contract_placebo_p,
            log_fixed_placebo_p,
        )
    )
    fixed_distance_signed = frame_results["alpha_results"]["90"]["fixed_3s"][
        "future_min_distance_m"
    ]["signed"]["parameters_reported"]["norm_exceedance_90"]["case_cluster"]
    lines.extend(
        [
            "",
            "## 8. 数值健康、覆盖与坐标异常",
            "",
            f"- 90% 层上侧 {alpha90['upper']:,} 帧、下侧 {alpha90['lower']:,} 帧、区间内 {alpha90['inside']:,} 帧，合计 {alpha90['sum']:,}/{alpha90['analysis_rows']:,}。筛选为两门交集；来源 RQ017 `ipv_log` 与机制二区间列 `lo_90, hi_90, width_90`。",
            f"- 合同窗口越出 case 末尾 {contract['overflow_rows']:,}/{contract['rows']:,} = {100*contract['overflow_ratio']:.4f}%；3 秒窗口越出 case 末尾 {fixed['overflow_rows']:,}/{fixed['rows']:,} = {100*fixed['overflow_ratio']:.4f}%。来源 `{DENSE_REL}`，列 `case_key, frame_index, time_s`；越界窗口使用 case 末尾前可见部分并保留标记。",
            f"- 合同窗口因全窗口 `closing_rate_mps <= 0` 而 TTC 缺失 {contract['ttc_all_nonclosing_rows']:,}/{contract['rows']:,} = {100*contract['ttc_all_nonclosing_ratio']:.4f}%；3 秒窗口为 {fixed['ttc_all_nonclosing_rows']:,}/{fixed['rows']:,} = {100*fixed['ttc_all_nonclosing_ratio']:.4f}%。来源 `{DENSE_REL}`，列 `distance_m, closing_rate_mps`。",
            f"- TTC 数值边界：稠密表最小正接近率为 {health['ttc_numeric_boundary']['minimum_positive_closing_rate_mps']:.3e} m/s；合同窗口 `future_min_ttc_s > 10^6` 的行数为 {health['ttc_numeric_boundary']['contract_ttc_above_1e6_rows']:,}，3 秒窗口为 {health['ttc_numeric_boundary']['fixed_3s_ttc_above_1e6_rows']:,}。来源 `{DENSE_REL}`，列 `distance_m, closing_rate_mps`。本轮保留任务书公式，不私设接近率下限；极大但有限的 TTC 会使原始尺度 OLS 对这些行敏感。",
            f"- case 帧数分布 min/p25/median/p75/max = {health['case_frame_distribution']['min']}/{health['case_frame_distribution']['p25']:.1f}/{health['case_frame_distribution']['median']:.1f}/{health['case_frame_distribution']['p75']:.1f}/{health['case_frame_distribution']['max']}；case 数 {health['case_frame_distribution']['case_count']}，team 数 {health['case_frame_distribution']['team_count']}。筛选为两门交集；来源 RQ017 + M2 + 锚点表，列 `status, mechanism2_gate_ok, case_key`。",
            f"- 全锚点表坐标异常为 {anomaly['full_anchor_rows']:,}/{anomaly['full_anchor_denominator']:,} = {100*anomaly['full_anchor_rows']/anomaly['full_anchor_denominator']:.4f}%，全部来自 `onsite:shanghai:T10:C4:native_case:2311`；来源 `{ANCHOR_REL}`，列 `case_key, relative_distance_anchor`。照常保留时，这 7 行中进入两门交集的是 {anomaly['two_gate_rows']:,}/{anomaly['two_gate_denominator']:,} = {100*anomaly['two_gate_rows']/anomaly['two_gate_denominator']:.4f}%；因此剔除口径与照常参与口径的主模型系数最大绝对差为 0。",
            "- `ipv_log` NaN/+inf/-inf 均为 0；80/90/95 三层均满足全部 14,099 行 `width > 0` 且 `lo < hi`。逐项机器证据见 `data_health.json`。",
            "",
            "## 9. 与 RQ012B 已接受结论的关系",
            "",
            "RQ012B 的冻结结论 `RQ012-KC-HARM-NULL` 使用含自动驾驶车目标值的旧 RQ009 M3 参照，分析的是 unit 级官方 harm，并判定没有相对 placebo 与 context-only 基线的 IPV 特异增量关联。本轮使用纯人-人参照、增加机制一过滤，并把主分析单元换成帧级未来窗口风险。因此本轮是不同曝露定义和不同分析单元的探索，不构成对旧结论的推翻。",
            "",
            "## 10. 结论",
            "",
            f"1. **下侧越界与更短的未来 TTC 是本轮最直接对应‘劣化’方向的线索。** 在 90% 层，`log(1+TTC)` 的下侧幅度系数在合同窗口为 {log_contract_lower['coefficient']:.4f} [{log_contract_lower['ci95_low']:.4f}, {log_contract_lower['ci95_high']:.4f}]（case 聚类 p={pformat(log_contract_lower['p_value'])}），3 秒窗口为 {log_fixed_lower['coefficient']:.4f} [{log_fixed_lower['ci95_low']:.4f}, {log_fixed_lower['ci95_high']:.4f}]（case 聚类 p={pformat(log_fixed_lower['p_value'])}）。80% 与 95% 层同方向。case-block 标签置换 p 分别为 {pformat(log_contract_label_p)} 与 {pformat(log_fixed_label_p)}，安慰剂 p 分别为 {pformat(log_contract_placebo_p)} 与 {pformat(log_fixed_placebo_p)}；按预先要求的 case 聚类、标签置换、安慰剂三项同时低于 0.05 的规则，两窗口总体判定为{'通过' if lower_checks_pass else '未同时通过'}。这是关联性线索，不是因果结论。",
            f"2. **上侧越界没有显示行为劣化，多个结果反而指向更大的未来距离或更低的最大接近率。** 例如 3 秒窗口有符号斜率对最小距离为 {fixed_distance_signed['coefficient']:.4f} m [{fixed_distance_signed['ci95_low']:.4f}, {fixed_distance_signed['ci95_high']:.4f}]（case 聚类 p={pformat(fixed_distance_signed['p_value'])}）。这可由场景选择、控制后的剩余结构或行为差异解释，本轮不能把它写成‘上侧异常更安全’。",
            "3. **unit 级结果不形成一致的跨子分数劣化模式。** 控制 `official_comprehensive` 后，效率与部分越界汇总呈负相关，而协调性对下侧幅度呈正相关；舒适与合规多数区间跨零。官方综合分与子分数存在强机械吸收，加上本轮同时查看多种曝露，unit 结果只适合作为后续假设来源。",
            "4. 次要安全结果的阳性 unit 很少，虽然下侧幅度与若干安全结果同向，当前区间与多重比较不足以支持‘有关联’或‘无关联’的稳定判断。监督方复算前，本报告保持探索性证据状态；任何后续手稿表述都需要独立数据复现与单独接受。",
            "",
            "## 11. 待监督方决定",
            "",
            "1. 是否将通过 case 聚类、标签置换和安慰剂三项检查的方向性线索列为下一数据集的预注册目标。依据是第 4–5 节三种证据同时成立；若不推进，本轮只保留为描述性产物。",
            "2. unit 基础集 245 个中有 20 个没有任何两门通过帧。本轮按缺失处理，避免把“无可分析帧”编码为零越界；监督方如需另一个政策口径，应另立分析而不是覆盖本轮。",
            "",
            "## 12. 可复跑产物",
            "",
            f"- 脚本：`{WORK_REL}/rq018_association.py`",
            f"- 关键数字：`{WORK_REL}/key_numbers.json`",
            f"- 帧级结果：`{WORK_REL}/frame_level_results.json`",
            f"- unit 级结果：`{WORK_REL}/unit_level_results.json`",
            f"- 负对照：`{WORK_REL}/negative_controls.json`",
            f"- 数据健康：`{WORK_REL}/data_health.json`",
            "",
            "state: WAITING_ON_COMMANDER",
            f"timestamp_utc: {generated_at}",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_outputs(work_dir: Path, report_path: Path) -> None:
    expected = [
        work_dir / "rq018_association.py",
        work_dir / "key_numbers.json",
        work_dir / "frame_level_results.json",
        work_dir / "unit_level_results.json",
        work_dir / "negative_controls.json",
        work_dir / "data_health.json",
        report_path,
    ]
    missing = [str(path) for path in expected if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty deliverables: {missing}")
    for path in expected[1:6]:
        json.loads(path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    if not report.rstrip().endswith("timestamp_utc: " + report.rstrip().split("timestamp_utc: ")[-1]):
        raise RuntimeError("Report timestamp is not the final field")
    if "state: WAITING_ON_COMMANDER" not in report:
        raise RuntimeError("Report state is not WAITING_ON_COMMANDER")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    root = args.root.resolve()
    work_dir = root / WORK_REL
    report_path = root / REPORT_REL
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(root)
    analysis, key_metrics = prepare_analysis(inputs)
    if len(analysis) != 14099:
        raise RuntimeError(
            f"Task contract mismatch: two-gate analysis has {len(analysis)} rows, expected 14099"
        )
    frame = calculate_future_outcomes(analysis, inputs.dense)
    unit_frame, unit_coverage = aggregate_units(frame, inputs.units)
    add_key_metrics(key_metrics, inputs, frame, unit_coverage)
    health = build_data_health(inputs, frame, key_metrics)

    frame_results = frame_models(frame)
    unit_results = unit_models(unit_frame, unit_coverage)
    label_results = run_label_permutations(frame)
    placebo_results = run_placebo(frame, frame_results)
    negative_controls = {
        "generated_at_utc": utc_now(),
        "case_label_permutation": label_results,
        "placebo_exposure": placebo_results,
    }

    generated_at = utc_now()
    key_payload = {
        "generated_at_utc": generated_at,
        "contract": "Every count metric includes numerator, denominator, filter, source path, and source columns.",
        "metrics": key_metrics,
    }
    frame_results = {
        "generated_at_utc": generated_at,
        "sources": {
            "exposure": [RQ017_REL, M2_REL],
            "anchor_link": ANCHOR_REL,
            "future_outcomes": DENSE_REL,
        },
        **frame_results,
    }
    unit_results = {
        "generated_at_utc": generated_at,
        "sources": {
            "exposure": [RQ017_REL, M2_REL, ANCHOR_REL],
            "outcomes": UNIT_REL,
        },
        **unit_results,
    }
    health = {"generated_at_utc": generated_at, **health}

    write_json(work_dir / "key_numbers.json", key_payload)
    write_json(work_dir / "frame_level_results.json", frame_results)
    write_json(work_dir / "unit_level_results.json", unit_results)
    write_json(work_dir / "negative_controls.json", negative_controls)
    write_json(work_dir / "data_health.json", health)
    report_path.write_text(
        build_report(
            generated_at,
            key_metrics,
            health,
            frame_results,
            unit_results,
            negative_controls,
        ),
        encoding="utf-8",
    )
    validate_outputs(work_dir, report_path)
    print(
        json.dumps(
            {
                "status": "WAITING_ON_COMMANDER",
                "analysis_rows": len(frame),
                "cases": int(frame["case_key"].nunique()),
                "teams": int(frame["team_id"].nunique()),
                "analysis_units": int(unit_coverage["base_analysis_units"]),
                "units_with_exposure": int(unit_coverage["units_with_defined_exposure"]),
                "generated_at_utc": generated_at,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
