"""Canonical standardized-history to M3 anchor feature assembly."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from sociality_estimation.verifier.features import (
    apet_constant_velocity_proxy,
    closing_ttc,
    relative_state,
    theil_sen_slope,
    wrap_angle,
)


HISTORY_WINDOW = 10
MIN_OBSERVATION = 4
SLOPE_MAX_POINTS = 5
HISTORY_COLUMNS = (
    "timestamp_s",
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "ego_heading",
    "counterpart_x",
    "counterpart_y",
    "counterpart_vx",
    "counterpart_vy",
    "counterpart_heading",
    "counterpart_ipv",
    "counterpart_ipv_error",
)
CATEGORY_COLUMNS = (
    "geometry_path_category",
    "geometry_path_relation",
    "turn_pair_label",
    "agent_type_pair",
    "vehicle_type_list",
    "av_included",
    "priority_role",
)


def build_m3_anchor_features(
    history: pd.DataFrame,
    categories: Mapping[str, object],
    *,
    case_start_timestamp_s: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Build one M3-ready anchor row from a standardized causal history.

    `history` must contain only observations available at or before the anchor
    and be ordered oldest to newest. For exact elapsed-time parity, callers
    pass the interaction/case start timestamp even when supplying a cropped
    ten-row history.
    """
    missing_history = [column for column in HISTORY_COLUMNS if column not in history]
    missing_categories = [column for column in CATEGORY_COLUMNS if column not in categories]
    if missing_history or missing_categories:
        raise ValueError(
            f"Missing history columns={missing_history}; categories={missing_categories}"
        )
    if len(history) < MIN_OBSERVATION:
        raise ValueError(f"At least {MIN_OBSERVATION} history rows are required")
    ordered = history.sort_values("timestamp_s", kind="stable")
    window = ordered.tail(HISTORY_WINDOW).reset_index(drop=True)
    anchor = window.iloc[-1]
    case_start = (
        float(case_start_timestamp_s)
        if case_start_timestamp_s is not None
        else float(ordered.iloc[0]["timestamp_s"])
    )

    relative = relative_state(
        window["ego_x"].to_numpy(float),
        window["ego_y"].to_numpy(float),
        window["ego_vx"].to_numpy(float),
        window["ego_vy"].to_numpy(float),
        window["counterpart_x"].to_numpy(float),
        window["counterpart_y"].to_numpy(float),
        window["counterpart_vx"].to_numpy(float),
        window["counterpart_vy"].to_numpy(float),
    )
    last = -1
    ego_position = anchor[["ego_x", "ego_y"]].to_numpy(dtype=float)
    ego_velocity = anchor[["ego_vx", "ego_vy"]].to_numpy(dtype=float)
    counterpart_position = anchor[["counterpart_x", "counterpart_y"]].to_numpy(dtype=float)
    counterpart_velocity = anchor[["counterpart_vx", "counterpart_vy"]].to_numpy(dtype=float)
    slope_window = window.tail(SLOPE_MAX_POINTS)

    row = dict(metadata or {})
    row.update(
        {
            "elapsed_time_s": float(anchor["timestamp_s"] - case_start),
            "history_row_count": int(len(window)),
            "ego_vx_anchor": float(anchor["ego_vx"]),
            "ego_vy_anchor": float(anchor["ego_vy"]),
            "ego_heading_anchor": float(anchor["ego_heading"]),
            "counterpart_vx_anchor": float(anchor["counterpart_vx"]),
            "counterpart_vy_anchor": float(anchor["counterpart_vy"]),
            "counterpart_heading_anchor": float(anchor["counterpart_heading"]),
            "relative_dx_anchor": float(relative["dx"][last]),
            "relative_dy_anchor": float(relative["dy"][last]),
            "relative_distance_anchor": float(relative["distance"][last]),
            "relative_dvx_anchor": float(relative["dvx"][last]),
            "relative_dvy_anchor": float(relative["dvy"][last]),
            "relative_speed_anchor": float(relative["rel_speed"][last]),
            "closing_rate_anchor": float(relative["closing_rate"][last]),
            "heading_difference_anchor": wrap_angle(
                float(anchor["counterpart_heading"] - anchor["ego_heading"])
            ),
            "relative_distance_mean_wx": float(np.mean(relative["distance"])),
            "relative_distance_std_wx": float(np.std(relative["distance"])),
            "relative_speed_mean_wx": float(np.mean(relative["rel_speed"])),
            "closing_rate_mean_wx": float(np.mean(relative["closing_rate"])),
            "closing_ttc_anchor": closing_ttc(
                float(relative["distance"][last]),
                float(relative["closing_rate"][last]),
            ),
            "apet_online_proxy": apet_constant_velocity_proxy(
                ego_position,
                ego_velocity,
                counterpart_position,
                counterpart_velocity,
            ),
            "counterpart_ipv_current": float(anchor["counterpart_ipv"]),
            "counterpart_ipv_error_current": float(anchor["counterpart_ipv_error"]),
            "counterpart_ipv_slope_pre_anchor": theil_sen_slope(
                slope_window["timestamp_s"].to_numpy(float),
                slope_window["counterpart_ipv"].to_numpy(float),
            ),
        }
    )
    row.update({column: categories[column] for column in CATEGORY_COLUMNS})
    return pd.DataFrame([row])
