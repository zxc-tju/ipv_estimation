#!/usr/bin/env python3
"""Build the IPV verifier visual explainer and its panel-level source data.

The three figures deliberately separate:

1. how a verifier decision is produced (including a real OnSite application),
2. what reference data support the frozen RQ009 M3 verifier, and
3. what the held-out evidence validates and where generalisation is limited.

All figure rendering is performed with Python/matplotlib.  The script reads the
frozen research artifacts but never mutates them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# Mandatory editable-text rules from the publication-figure contract.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

# Arial Unicode provides local CJK glyph coverage while Arial remains the Latin
# face.  PDF text is embedded as editable TrueType text.
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Arial Unicode MS",
    "DejaVu Sans",
    "Liberation Sans",
]
# Matplotlib does not reliably fall back within ``font.sans-serif`` for CJK
# glyphs.  Use the installed Unicode Arial face explicitly after recording the
# mandatory sans-serif/editable-SVG contract above.
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7
plt.rcParams["savefig.facecolor"] = "white"


INK = "#263746"
MUTED = "#687684"
GRID = "#DCE3E8"
ENVELOPE = "#69B7A7"
ENVELOPE_LIGHT = "#DDF1EC"
OBSERVED = "#315C9B"
PASS = "#2A8C7B"
UPPER = "#C74E45"
LOWER = "#7A5A9E"
ABSTAIN = "#B18A42"
ABSTAIN_LIGHT = "#F2E7CF"
UNSUPPORTED = "#C7CDD2"
PALE_BLUE = "#DCE8F5"
PALE_GREY = "#F2F4F6"

SOURCE_COLORS = {
    "waymo_train": "#4D6C9C",
    "nuplan_train": "#7895BC",
    "lyft_train_full": "#B2C4DB",
    "av2_motion_forecasting": "#DDE6F0",
}

SOURCE_LABELS = {
    "waymo_train": "Waymo",
    "nuplan_train": "nuPlan",
    "lyft_train_full": "Lyft",
    "av2_motion_forecasting": "AV2",
}

CASE_KEY = "onsite:beijing:T15:B3:native_case:2335"
ANNOTATED_FRAMES = (63, 80, 210, 246)


def comma(value: float | int) -> str:
    """Format a count without scientific notation."""
    return f"{int(round(float(value))):,}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.06, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
    )


def quiet_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.tick_params(labelsize=6.4, colors=INK, length=2.8, pad=2)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)


def rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = GRID,
    linewidth: float = 0.9,
    radius: float = 0.018,
    hatch: str | None = None,
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        hatch=hatch,
    )
    ax.add_patch(box)
    return box


def arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color=INK,
            shrinkA=1,
            shrinkB=1,
        ),
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    # Matplotlib writes trailing spaces in multi-line SVG path data.  Strip
    # them deterministically so regenerated editable figures stay Git-clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def matrix_fold_counts(matrix_root: Path) -> pd.DataFrame:
    rows = []
    for fold in ("train", "guard_tune", "calibration", "test"):
        files = sorted((matrix_root / f"fold={fold}").rglob("*.parquet"))
        count = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        rows.append({"fold": fold, "raw_rows": int(count), "file_count": len(files)})
    return pd.DataFrame(rows)


def aggregate_train_matrix(matrix_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exact train composition and joint-cell support."""
    dimensions = (
        "source_dataset",
        "geometry_path_category",
        "agent_type_pair",
        "priority_role",
    )
    counts: Dict[str, Counter] = {dimension: Counter() for dimension in dimensions}
    joint_counts: Counter = Counter()
    joint_cases: Dict[str, set[str]] = defaultdict(set)

    columns = ["case_key", *dimensions]
    for path in sorted((matrix_root / "fold=train").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=131_072):
            values = batch.to_pydict()
            for dimension in dimensions:
                counts[dimension].update(values[dimension])
            for index in range(batch.num_rows):
                cell = "|".join(
                    (
                        str(values["geometry_path_category"][index]),
                        str(values["priority_role"][index]),
                        str(values["agent_type_pair"][index]),
                    )
                )
                joint_counts[cell] += 1
                joint_cases[cell].add(str(values["case_key"][index]))

    total = sum(counts["source_dataset"].values())
    composition_rows = []
    for dimension, counter in counts.items():
        for level, anchors in counter.items():
            composition_rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "anchors": int(anchors),
                    "share": float(anchors / total),
                    "train_total": int(total),
                }
            )

    joint_rows = []
    for geometry, role, agent_pair in product(
        ("MP", "CP", "F", "HO"),
        ("priority", "yield", "equal"),
        ("HV;HV", "AV;HV"),
    ):
        cell = f"{geometry}|{role}|{agent_pair}"
        anchors = int(joint_counts.get(cell, 0))
        cases = int(len(joint_cases.get(cell, set())))
        anchor_gate = anchors >= 50
        case_gate = cases >= 10
        supported = anchor_gate and case_gate
        if supported:
            failure = ""
        elif anchors == 0:
            failure = "not_observed"
        elif not anchor_gate and not case_gate:
            failure = "anchors_and_cases_below_threshold"
        elif not anchor_gate:
            failure = "anchors_below_threshold"
        else:
            failure = "cases_below_threshold"
        joint_rows.append(
            {
                "geometry_path_category": geometry,
                "priority_role": role,
                "agent_type_pair": agent_pair,
                "joint_cell": cell,
                "anchors": anchors,
                "unique_cases": cases,
                "anchor_gate_ge_50": anchor_gate,
                "case_gate_ge_10": case_gate,
                "supported": supported,
                "failure_reason": failure,
            }
        )
    return pd.DataFrame(composition_rows), pd.DataFrame(joint_rows)


def build_gate_flow(folds: pd.DataFrame, ood_gate: Mapping[str, object]) -> pd.DataFrame:
    raw = dict(zip(folds["fold"], folds["raw_rows"]))
    application = ood_gate["application"]
    rows = [
        {
            "fold": "train",
            "raw_rows": raw["train"],
            "category_pass_rows": int(ood_gate["train_reference_rows_after_category_support"]),
            "gate_pass_rows": int(ood_gate["train_reference_rows_after_category_support"]),
            "gate_definition": "category support reference",
        },
        {
            "fold": "guard_tune",
            "raw_rows": raw["guard_tune"],
            "category_pass_rows": int(ood_gate["guard_category_eligible_rows"]),
            "gate_pass_rows": int(ood_gate["guard_gate_passing_rows"]),
            "gate_definition": "category + kNN distance",
        },
        {
            "fold": "calibration",
            "raw_rows": raw["calibration"],
            "category_pass_rows": int(application["calibration"]["category_pass_rows"]),
            "gate_pass_rows": int(application["calibration"]["gate_pass_rows"]),
            "gate_definition": "category + kNN distance",
        },
        {
            "fold": "test",
            "raw_rows": raw["test"],
            "category_pass_rows": int(application["test"]["category_pass_rows"]),
            "gate_pass_rows": int(application["test"]["gate_pass_rows"]),
            "gate_definition": "category + kNN distance",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["abstain_rows"] = frame["raw_rows"] - frame["gate_pass_rows"]
    frame["gate_pass_rate"] = frame["gate_pass_rows"] / frame["raw_rows"]
    return frame


def load_onsite_case(scored_path: Path, trajectory_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = [
        "case_key",
        "unit_composite_key",
        "anchor_frame_index",
        "elapsed_time_s",
        "target_ipv_future",
        "pred_human_ipv_point",
        "lo_90",
        "hi_90",
        "support_gate_pass",
        "gate_category_pass",
        "gate_distance_k25_mean",
        "gate_distance_threshold",
        "deviation_signed_exceedance_90",
        "deviation_abs_exceedance_90",
        "deviation_outside_90",
        "deviation_upper_tail_90",
        "deviation_lower_tail_90",
        "abstention_reasons",
        "gate_joint_cell",
    ]
    scores = pd.read_parquet(
        scored_path,
        columns=score_columns,
        filters=[("case_key", "==", CASE_KEY)],
    ).sort_values("anchor_frame_index")

    trajectory_columns = [
        "case_key",
        "unit_composite_key",
        "frame_index",
        "time_s",
        "ego_x",
        "ego_y",
        "counterpart_x",
        "counterpart_y",
        "distance_m",
        "closing_rate_mps",
        "ipv_ego_hw10",
        "ipv_counterpart_hw10",
    ]
    trajectory = pd.read_parquet(
        trajectory_path,
        columns=trajectory_columns,
        filters=[("case_key", "==", CASE_KEY)],
    ).sort_values("frame_index")

    positions = trajectory.rename(columns={"frame_index": "anchor_frame_index"})[
        [
            "anchor_frame_index",
            "time_s",
            "ego_x",
            "ego_y",
            "counterpart_x",
            "counterpart_y",
            "distance_m",
            "closing_rate_mps",
        ]
    ]
    scores = scores.merge(positions, on="anchor_frame_index", how="left", validate="one_to_one")

    scores["status"] = "in_band"
    scores.loc[scores["deviation_upper_tail_90"].fillna(False), "status"] = "upper"
    scores.loc[scores["deviation_lower_tail_90"].fillna(False), "status"] = "lower"
    scores.loc[~scores["support_gate_pass"].fillna(False), "status"] = "abstain"
    scores["observed_future_ipv"] = scores["target_ipv_future"]
    return scores, trajectory


def load_validation_sources(
    metrics_path: Path,
    c1_source_path: Path,
    lodo_path: Path,
    ood_gate: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    coverage = metrics[metrics["tier"].eq("M3")].copy()
    coverage["coverage_delta_pp"] = (coverage["coverage"] - coverage["nominal"]) * 100.0

    c1 = pd.read_csv(c1_source_path)
    width_hist = c1[c1["panel"].eq("m3_width_histogram_90")].copy()
    width_quantiles = c1[c1["panel"].eq("m3_width_quantile_90")].copy()

    subgroup_rows = []
    subgroup_map = ood_gate["application"]["test"]["subgroups"]
    axis_order = (
        "source_dataset",
        "geometry_path_category",
        "priority_role",
        "agent_type_pair",
    )
    for axis in axis_order:
        for level, values in subgroup_map[axis].items():
            subgroup_rows.append(
                {
                    "axis": axis,
                    "level": level,
                    "anchors": int(values["anchors"]),
                    "cases": int(values["cases"]),
                    "abstention_rate": float(values["abstention_rate"]),
                }
            )
    abstention = pd.DataFrame(subgroup_rows)
    abstention["overall_abstention_rate"] = float(ood_gate["abstention_overall_test"])

    lodo = pd.read_csv(lodo_path)
    lodo = lodo[(lodo["tier"].eq("M3")) & (lodo["alpha_label"].eq(90))].copy()
    return coverage, width_hist, width_quantiles, abstention, lodo


def figure_1_mechanism(case: pd.DataFrame, trajectory: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 6.55))
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.55, 1.25, 0.82],
        hspace=0.42,
        wspace=0.72,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1:, :4])
    ax_c = fig.add_subplot(grid[1, 4:])
    ax_d = fig.add_subplot(grid[2, 4:], sharex=ax_c)

    # Panel a: the mechanism.  Observed IPV deliberately bypasses the envelope
    # model and enters only the final comparison, preserving the runtime logic.
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    add_panel_label(ax_a, "a", x=-0.015, y=1.0)
    ax_a.text(
        0.02,
        1.02,
        "先判“能不能比”，再判“偏离多少”",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )

    boxes = {
        "context": (0.02, 0.14, 0.20, 0.72),
        "model": (0.27, 0.14, 0.20, 0.72),
        "gate": (0.52, 0.14, 0.20, 0.72),
        "verdict": (0.77, 0.14, 0.21, 0.72),
    }
    rounded_box(ax_a, boxes["context"][:2], boxes["context"][2], boxes["context"][3], facecolor="#F4F7FA")
    rounded_box(ax_a, boxes["model"][:2], boxes["model"][2], boxes["model"][3], facecolor="#EDF4FB", edgecolor="#A9C3DE")
    rounded_box(ax_a, boxes["gate"][:2], boxes["gate"][2], boxes["gate"][3], facecolor="#F4F0F8", edgecolor="#C7B6D5")
    rounded_box(ax_a, boxes["verdict"][:2], boxes["verdict"][2], boxes["verdict"][3], facecolor="#F7F8F9")

    # Context sketch.
    ax_a.text(0.035, 0.79, "1  当前交互情境", fontsize=8.2, fontweight="bold", color=INK)
    ax_a.plot([0.045, 0.19], [0.36, 0.66], color=OBSERVED, linewidth=2.0)
    ax_a.plot([0.04, 0.20], [0.66, 0.36], color=MUTED, linewidth=1.6)
    ax_a.add_patch(patches.Rectangle((0.093, 0.45), 0.026, 0.053, angle=22, color=OBSERVED, ec="white", lw=0.5))
    ax_a.add_patch(patches.Rectangle((0.142, 0.47), 0.026, 0.053, angle=-22, color="#7A8792", ec="white", lw=0.5))
    ax_a.text(0.035, 0.26, "仅使用 t ≤ t* 的历史", fontsize=6.7, color=INK)
    ax_a.text(0.035, 0.205, "25 数值 + 7 类别特征", fontsize=6.7, color=INK)
    ax_a.text(0.035, 0.155, "含对手 IPV 历史通道", fontsize=6.2, color=MUTED)

    # Conditional envelope model.
    ax_a.text(0.285, 0.79, "2  条件人类包络", fontsize=8.2, fontweight="bold", color=INK)
    x = np.linspace(0.30, 0.44, 100)
    centre = 0.50
    density = np.exp(-0.5 * ((x - 0.37) / 0.036) ** 2)
    density = 0.11 * density / density.max()
    ax_a.fill_between(x, centre - density, centre + density, color=ENVELOPE_LIGHT, edgecolor=ENVELOPE, linewidth=0.8)
    for xpos, label in ((0.31, "q05"), (0.37, "q50"), (0.43, "q95")):
        ax_a.plot([xpos, xpos], [0.37, 0.63], color=INK if label == "q50" else ENVELOPE, linewidth=0.8, linestyle="--" if label != "q50" else "-")
        ax_a.text(xpos, 0.33, label, ha="center", va="top", fontsize=5.8, color=INK)
    ax_a.text(0.285, 0.235, "7 个条件分位点", fontsize=6.7, color=INK)
    ax_a.text(0.285, 0.185, "CQR 校准后形成 80/90/95% 带", fontsize=6.2, color=MUTED)
    ax_a.text(0.285, 0.145, "90%: n = 1,205,609; c = −0.008091", fontsize=5.9, color=MUTED)

    # Two-stage gate.
    ax_a.text(0.535, 0.79, "3  支持门：可比吗？", fontsize=8.2, fontweight="bold", color=INK)
    rounded_box(ax_a, (0.545, 0.50), 0.15, 0.16, facecolor="white", edgecolor="#C7B6D5", radius=0.012)
    ax_a.text(0.56, 0.606, "类别支持", fontsize=6.9, fontweight="bold", color=INK)
    ax_a.text(0.56, 0.555, "19 / 24 联合单元", fontsize=6.3, color=INK)
    ax_a.text(0.56, 0.515, "≥50 anchors 且 ≥10 cases", fontsize=5.8, color=MUTED)
    rounded_box(ax_a, (0.545, 0.27), 0.15, 0.16, facecolor="white", edgecolor="#C7B6D5", radius=0.012)
    ax_a.text(0.56, 0.376, "数值 OOD", fontsize=6.9, fontweight="bold", color=INK)
    ax_a.text(0.56, 0.325, "25-NN 平均距离", fontsize=6.3, color=INK)
    ax_a.text(0.56, 0.286, "15 数值 + 3 类别", fontsize=5.5, color=MUTED)
    ax_a.text(0.56, 0.252, "d ≤ 1.6072", fontsize=5.5, color=MUTED)

    # Final comparison and states.
    ax_a.text(0.785, 0.79, "4  比较 observed IPV", fontsize=8.2, fontweight="bold", color=INK)
    state_specs = (
        (0.79, 0.57, "带内", "超出量 = 0", PASS, "white"),
        (0.89, 0.57, "上界超出", "+ 超出量", UPPER, "white"),
        (0.79, 0.34, "下界超出", "− 超出量", LOWER, "white"),
        (0.89, 0.34, "弃权", "不作判断", ABSTAIN_LIGHT, INK),
    )
    for xpos, ypos, title, subtitle, color, text_color in state_specs:
        rounded_box(ax_a, (xpos, ypos), 0.078, 0.14, facecolor=color, edgecolor=color if color != ABSTAIN_LIGHT else ABSTAIN, radius=0.010)
        ax_a.text(xpos + 0.039, ypos + 0.088, title, ha="center", fontsize=6.3, fontweight="bold", color=text_color)
        ax_a.text(xpos + 0.039, ypos + 0.040, subtitle, ha="center", fontsize=5.3, color=text_color if color != ABSTAIN_LIGHT else MUTED)
    ax_a.text(0.785, 0.19, "观测 IPV 只进入最终比较器", fontsize=6.2, color=INK)
    ax_a.text(0.785, 0.15, "外部应用须保持同一时间对齐合同", fontsize=5.8, color=MUTED)

    arrow(ax_a, (0.225, 0.50), (0.265, 0.50))
    arrow(ax_a, (0.475, 0.50), (0.515, 0.50))
    arrow(ax_a, (0.725, 0.50), (0.765, 0.50))
    # Observed-IPV bypass.
    ax_a.annotate(
        "观测到的自车窗口 IPV",
        xy=(0.795, 0.245),
        xytext=(0.12, 0.06),
        fontsize=5.8,
        color=OBSERVED,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color=OBSERVED,
            linewidth=0.9,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
    )

    # Panel b: real trajectory.
    add_panel_label(ax_b, "b")
    ax_b.set_title("真实 OnSite 交互轨迹", loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=5)
    x0 = float(np.nanmean(np.r_[trajectory["ego_x"], trajectory["counterpart_x"]]))
    y0 = float(np.nanmean(np.r_[trajectory["ego_y"], trajectory["counterpart_y"]]))
    ax_b.plot(trajectory["counterpart_x"] - x0, trajectory["counterpart_y"] - y0, color="#9EA8B0", linewidth=1.3, alpha=0.85, label="交互对手")
    ax_b.plot(trajectory["ego_x"] - x0, trajectory["ego_y"] - y0, color=OBSERVED, linewidth=1.5, alpha=0.9, label="AV")
    status_colors = {"in_band": PASS, "upper": UPPER, "lower": LOWER, "abstain": ABSTAIN}
    for status, status_frame in case.groupby("status", observed=True):
        ax_b.scatter(
            status_frame["ego_x"] - x0,
            status_frame["ego_y"] - y0,
            s=7 if status != "abstain" else 5,
            color=status_colors[status],
            alpha=0.85 if status != "abstain" else 0.45,
            edgecolor="none",
            zorder=4,
        )
    for number, frame_index in enumerate(ANNOTATED_FRAMES, start=1):
        row = case.loc[case["anchor_frame_index"].eq(frame_index)].iloc[0]
        ax_b.scatter(row["ego_x"] - x0, row["ego_y"] - y0, s=35, facecolor="white", edgecolor=status_colors[row["status"]], linewidth=1.1, zorder=7)
        ax_b.text(row["ego_x"] - x0, row["ego_y"] - y0, str(number), ha="center", va="center", fontsize=5.8, fontweight="bold", color=status_colors[row["status"]], zorder=8)
    ax_b.set_aspect("equal", adjustable="datalim")
    ax_b.set_xlabel("Δx (m)", fontsize=6.7, color=INK)
    ax_b.set_ylabel("Δy (m)", fontsize=6.7, color=INK)
    quiet_axis(ax_b)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PASS, markeredgecolor="none", markersize=4, label="带内"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=UPPER, markeredgecolor="none", markersize=4, label="上界超出"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LOWER, markeredgecolor="none", markersize=4, label="下界超出"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ABSTAIN, markeredgecolor="none", markersize=4, label="弃权"),
    ]
    ax_b.legend(handles=legend_handles, loc="lower left", fontsize=5.8, ncol=2, handletextpad=0.4, columnspacing=0.8)

    # Panel c: dynamic envelope and observed future IPV.
    add_panel_label(ax_c, "c")
    ax_c.set_title("动态 90% 包络：随情境变化；弃权时撤回", loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=5)
    time = case["elapsed_time_s"].to_numpy(float)
    lo = case["lo_90"].to_numpy(float)
    hi = case["hi_90"].to_numpy(float)
    observed_values = case["observed_future_ipv"].to_numpy(float)
    q50 = case["pred_human_ipv_point"].to_numpy(float)
    ax_c.fill_between(time, lo, hi, where=np.isfinite(lo) & np.isfinite(hi), color=ENVELOPE_LIGHT, alpha=1.0, linewidth=0, label="90% 人类包络")
    ax_c.plot(time, q50, color=ENVELOPE, linewidth=0.8, linestyle="--", alpha=0.8, label="条件中位数")
    ax_c.plot(time, observed_values, color=OBSERVED, linewidth=0.65, alpha=0.55, zorder=2)
    for status in ("in_band", "upper", "lower", "abstain"):
        frame = case[case["status"].eq(status)]
        marker = "x" if status == "abstain" else "o"
        ax_c.scatter(
            frame["elapsed_time_s"],
            frame["observed_future_ipv"],
            s=8 if status != "abstain" else 7,
            marker=marker,
            color=status_colors[status],
            alpha=0.9 if status != "abstain" else 0.65,
            linewidth=0.55,
            zorder=4,
        )
    for number, frame_index in enumerate(ANNOTATED_FRAMES, start=1):
        row = case.loc[case["anchor_frame_index"].eq(frame_index)].iloc[0]
        ax_c.annotate(
            str(number),
            xy=(row["elapsed_time_s"], row["observed_future_ipv"]),
            xytext=(0, 9 if number in (2, 3) else -10),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=status_colors[row["status"]],
            bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=status_colors[row["status"]], lw=0.75),
            arrowprops=dict(arrowstyle="-", color=status_colors[row["status"]], lw=0.6),
        )
    ax_c.axhline(0, color=GRID, linewidth=0.6, zorder=0)
    ax_c.set_ylabel("观测 / 包络 IPV（rad）", fontsize=6.7, color=INK)
    ax_c.tick_params(labelbottom=False)
    quiet_axis(ax_c, grid_axis="y")
    ax_c.legend(loc="upper right", fontsize=5.7, ncol=2, handlelength=1.8, columnspacing=0.8)

    # Panel d: distance gate.
    add_panel_label(ax_d, "d")
    distance = case["gate_distance_k25_mean"].to_numpy(float)
    threshold = float(case["gate_distance_threshold"].dropna().iloc[0])
    ax_d.plot(time, distance, color="#8A6BA5", linewidth=0.9, alpha=0.8)
    ax_d.fill_between(time, 0, threshold, color="#EEE7F3", alpha=0.75, zorder=0)
    ax_d.axhline(threshold, color=ABSTAIN, linewidth=1.05, linestyle="--")
    ax_d.text(time[-1], threshold + 0.08, f"阈值 = {threshold:.4f}", ha="right", va="bottom", fontsize=5.8, color=ABSTAIN)
    pass_mask = case["support_gate_pass"].to_numpy(bool)
    ax_d.scatter(time[pass_mask], distance[pass_mask], s=5, color=PASS, alpha=0.65, edgecolor="none")
    ax_d.scatter(time[~pass_mask], distance[~pass_mask], s=5, color=ABSTAIN, alpha=0.55, edgecolor="none")
    for number, frame_index in enumerate(ANNOTATED_FRAMES, start=1):
        row = case.loc[case["anchor_frame_index"].eq(frame_index)].iloc[0]
        ax_d.text(row["elapsed_time_s"], row["gate_distance_k25_mean"], str(number), ha="center", va="center", fontsize=5.3, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.14", fc=status_colors[row["status"]], ec="white", lw=0.4), zorder=7)
    ax_d.set_ylim(0, max(4.55, float(np.nanpercentile(distance, 99)) * 1.05))
    ax_d.set_xlabel("距首个有效锚点的时间（s）", fontsize=6.7, color=INK)
    ax_d.set_ylabel("25-NN 平均\n距离", fontsize=6.3, color=INK)
    quiet_axis(ax_d, grid_axis="y")
    ax_d.text(0.0, 1.04, "距离超过阈值 → ABSTAIN；不是“正常”", transform=ax_d.transAxes, fontsize=6.4, fontweight="bold", color=INK, va="bottom")

    fig.suptitle("IPV verifier 的运行机制与一条真实外部应用序列", fontsize=13.0, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, output_dir, "fig1_verifier_mechanism")


def figure_2_reference_data(
    gate_flow: pd.DataFrame,
    composition: pd.DataFrame,
    joint_support: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.25))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.88, 1.05, 1.55], hspace=0.62, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])

    # Gate-flow bars.
    add_panel_label(ax_a, "a")
    labels = ["训练参考", "阈值选择", "CQR 校准", "冻结测试"]
    y = np.arange(len(labels))[::-1]
    total = gate_flow["raw_rows"].to_numpy(float)
    usable = gate_flow["gate_pass_rows"].to_numpy(float)
    category_pass = gate_flow["category_pass_rows"].to_numpy(float)
    ax_a.barh(y, total / 1e6, color=PALE_GREY, edgecolor=GRID, linewidth=0.6, height=0.58, label="原始 rows")
    ax_a.barh(y, category_pass / 1e6, color="#C8D7E8", edgecolor="none", height=0.58, label="类别支持")
    ax_a.barh(y, usable / 1e6, color="#7895BC", edgecolor="none", height=0.58, label="进入参考/通过 gate")
    for yi, raw_value, usable_value in zip(y, total, usable):
        ax_a.text(raw_value / 1e6 + 0.035, yi, f"{comma(usable_value)} / {comma(raw_value)}  ({100*usable_value/raw_value:.2f}%)", va="center", fontsize=6.4, color=INK)
    ax_a.set_yticks(y, labels)
    ax_a.set_xlim(0, 3.05)
    ax_a.set_xlabel("视角-锚点记录（百万行）", fontsize=6.7, color=INK)
    ax_a.set_title("数据流：只有支持门通过的行才能校准或验证包络", loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_a, grid_axis="x")
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, -0.22),
        fontsize=5.7,
        ncol=3,
        columnspacing=0.8,
        handlelength=1.4,
    )
    test_row = gate_flow[gate_flow["fold"].eq("test")].iloc[0]
    ax_a.text(
        0.995,
        1.04,
        f"测试折：类别不支持 {comma(test_row['raw_rows'] - test_row['category_pass_rows'])}；数值 OOD {comma(test_row['category_pass_rows'] - test_row['gate_pass_rows'])}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )

    # Dataset sources.
    add_panel_label(ax_b, "b")
    source = composition[composition["dimension"].eq("source_dataset")].copy()
    source["sort"] = source["level"].map({name: index for index, name in enumerate(SOURCE_COLORS)})
    source = source.sort_values("sort")
    yy = np.arange(len(source))[::-1]
    colors = [SOURCE_COLORS[level] for level in source["level"]]
    ax_b.barh(yy, source["share"] * 100, color=colors, height=0.62, edgecolor="white", linewidth=0.45)
    for yi, (_, row) in zip(yy, source.iterrows()):
        ax_b.text(row["share"] * 100 + 1.0, yi, f"{row['share']*100:.2f}%  ({comma(row['anchors'])})", va="center", fontsize=6.2, color=INK)
    ax_b.set_yticks(yy, [SOURCE_LABELS[level] for level in source["level"]])
    ax_b.set_xlim(0, 62)
    ax_b.set_xlabel("训练折占比", fontsize=6.7, color=INK)
    ax_b.set_title("四个来源共同构成人类参考", loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_b, grid_axis="x")

    # Context imbalance strips.
    add_panel_label(ax_c, "c")
    strips = [
        ("geometry_path_category", "路径", ["MP", "CP", "F", "HO"], ["#4D6C9C", "#8FA8C7", "#BBC9DB", "#E1E7EE"]),
        ("agent_type_pair", "主体", ["HV;HV", "AV;HV"], ["#5D7D8B", "#B5CBD1"]),
        ("priority_role", "角色", ["priority", "yield", "equal"], ["#796A9B", "#A494BE", "#D7CEE3"]),
    ]
    strip_y = [2, 1, 0]
    for (dimension, label, levels, colors), ypos in zip(strips, strip_y):
        subset = composition[composition["dimension"].eq(dimension)].set_index("level")
        left = 0.0
        for level, color in zip(levels, colors):
            share = float(subset.loc[level, "share"])
            ax_c.barh(ypos, share * 100, left=left, color=color, edgecolor="white", linewidth=0.45, height=0.55)
            if share >= 0.08:
                ax_c.text(left + share * 50, ypos, f"{level}\n{share*100:.2f}%", ha="center", va="center", fontsize=5.7, color="white" if mpl_colors.rgb_to_hsv(mpl_colors.to_rgb(color))[2] < 0.72 else INK, fontweight="bold" if share > 0.4 else "normal")
            left += share * 100
    ax_c.set_yticks(strip_y, ["路径类别", "主体类型", "观察角色"])
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("训练折占比", fontsize=6.7, color=INK)
    ax_c.set_title("MP 路径占训练折 91.02%", loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_c, grid_axis="x")
    ax_c.text(99, 2.33, "CP 4.89%  ·  F 3.66%  ·  HO 0.43%", ha="right", va="bottom", fontsize=5.4, color=MUTED)
    ax_c.text(99, -0.33, "equal 1.35%", ha="right", va="top", fontsize=5.4, color=MUTED)

    # Joint-cell heatmap: anchors and cases are both visible because support
    # requires both thresholds.
    add_panel_label(ax_d, "d")
    row_order = [
        (geometry, agent_pair)
        for geometry in ("MP", "CP", "F", "HO")
        for agent_pair in ("HV;HV", "AV;HV")
    ]
    col_order = ["priority", "yield", "equal"]
    values = np.zeros((len(row_order), len(col_order)), dtype=float)
    for row_index, (geometry, agent_pair) in enumerate(row_order):
        for col_index, role in enumerate(col_order):
            match = joint_support[
                joint_support["geometry_path_category"].eq(geometry)
                & joint_support["agent_type_pair"].eq(agent_pair)
                & joint_support["priority_role"].eq(role)
            ].iloc[0]
            values[row_index, col_index] = np.log10(max(1, int(match["anchors"])))
    cmap = mpl_colors.LinearSegmentedColormap.from_list("support", ["#F4F7FA", "#B9CBE0", "#4D6C9C"])
    image = ax_d.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=6)
    for row_index, (geometry, agent_pair) in enumerate(row_order):
        for col_index, role in enumerate(col_order):
            match = joint_support[
                joint_support["geometry_path_category"].eq(geometry)
                & joint_support["agent_type_pair"].eq(agent_pair)
                & joint_support["priority_role"].eq(role)
            ].iloc[0]
            anchors = int(match["anchors"])
            cases = int(match["unique_cases"])
            supported = bool(match["supported"])
            luminance = values[row_index, col_index] / 6.0
            text_color = "white" if luminance > 0.60 else INK
            symbol = "✓" if supported else ("—" if anchors == 0 else "×")
            ax_d.text(col_index, row_index - 0.12, f"{symbol}  {comma(anchors)}", ha="center", va="center", fontsize=6.2, fontweight="bold", color=text_color)
            ax_d.text(col_index, row_index + 0.23, f"{comma(cases)} 场景", ha="center", va="center", fontsize=5.2, color=text_color)
            if not supported:
                rect = patches.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1, fill=False, hatch="////", edgecolor="#8C969D", linewidth=0.75)
                ax_d.add_patch(rect)
    ax_d.set_xticks(np.arange(len(col_order)), ["priority", "yield", "equal"])
    ax_d.set_yticks(np.arange(len(row_order)), [f"{geometry} | {agent_pair}" for geometry, agent_pair in row_order])
    ax_d.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, labelsize=6.1)
    ax_d.set_title("24 个路径×角色×主体联合单元：19 支持、4 稀疏不支持、1 未出现", loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=18)
    ax_d.text(0.0, 1.09, "单元内显示锚点数 / 唯一场景数；支持条件：锚点 ≥ 50 且场景 ≥ 10；斜纹 = 不支持", transform=ax_d.transAxes, fontsize=6.2, color=MUTED, va="bottom")
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.018, pad=0.02)
    colorbar.set_label("log10（锚点数）", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.3, length=2)

    fig.suptitle("冻结 RQ009 M3 verifier 的数据基础与支持空间", fontsize=13.0, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, output_dir, "fig2_reference_data")


def figure_3_evidence_limits(
    coverage: pd.DataFrame,
    width_quantiles: pd.DataFrame,
    abstention: pd.DataFrame,
    lodo: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.52], hspace=0.48, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    sub = grid[1, 1].subgridspec(1, 2, wspace=0.18)
    ax_d1 = fig.add_subplot(sub[0, 0])
    ax_d2 = fig.add_subplot(sub[0, 1], sharey=ax_d1)

    # Overall held-out coverage.
    add_panel_label(ax_a, "a")
    x = coverage["nominal"].to_numpy(float) * 100
    y = coverage["coverage"].to_numpy(float) * 100
    xx = np.linspace(78, 97, 100)
    ax_a.fill_between(xx, xx - 3, xx + 3, color=PALE_GREY, alpha=1.0, zorder=0, label="±3 pp")
    ax_a.plot(xx, xx, color=MUTED, linestyle="--", linewidth=0.9, label="nominal")
    ax_a.plot(x, y, color=OBSERVED, linewidth=1.2, marker="o", markersize=4.6)
    for _, row in coverage.iterrows():
        ax_a.annotate(
            f"{row['coverage']*100:.2f}%\n({row['coverage_delta_pp']:+.2f} pp)",
            xy=(row["nominal"] * 100, row["coverage"] * 100),
            xytext=(0, 8 if row["alpha_label"] != 95 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=5.8,
            color=INK,
        )
    ax_a.set_xlim(78, 97)
    ax_a.set_ylim(78, 97)
    ax_a.set_xticks([80, 90, 95])
    ax_a.set_yticks([80, 85, 90, 95])
    ax_a.set_xlabel("标称覆盖率（%）", fontsize=6.7, color=INK)
    ax_a.set_ylabel("留出集实测覆盖率（%）", fontsize=6.7, color=INK)
    ax_a.set_title("门后边际覆盖：接近标称水平", loc="left", fontsize=8.3, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_a, grid_axis="both")
    ax_a.text(
        0.02,
        0.03,
        "门通过记录：89.87%（1,209,857 行；7,550 场景）\n"
        "全测试集（弃权计为未覆盖）：85.57%",
        transform=ax_a.transAxes,
        fontsize=5.45,
        color=MUTED,
        va="bottom",
    )

    # Why context conditioning matters: M0 versus frozen M3 at 90%.
    add_panel_label(ax_b, "b")
    global_values = np.array([1.7486657502267058, 2.210260664657108, 0.8986359544971018])
    m3_values = np.array([1.0161520998427664, 1.4229346736299133, 0.8986657100797862])
    ratios = 100 * m3_values / global_values
    metric_labels = ["平均带宽", "Winkler 分数", "覆盖率"]
    yy = np.arange(3)[::-1]
    ax_b.hlines(yy, ratios, 100, color=GRID, linewidth=5.0, zorder=1)
    ax_b.scatter(np.full(3, 100), yy, s=34, color="#9EA8B0", zorder=3, label="global envelope")
    ax_b.scatter(ratios, yy, s=38, color=OBSERVED, zorder=4, label="M3 context envelope")
    absolute_labels = [
        f"1.749 → 1.016  ({(ratios[0]-100):+.1f}%)",
        f"2.210 → 1.423  ({(ratios[1]-100):+.1f}%)",
        f"89.86% → 89.87%  ({(m3_values[2]-global_values[2])*100:+.2f} pp)",
    ]
    annotation_specs = [
        (63.0, yy[0] - 0.30, "left"),
        (69.0, yy[1] + 0.25, "left"),
        (103.0, yy[2] + 0.25, "right"),
    ]
    for label, (x_text, y_text, align) in zip(absolute_labels, annotation_specs):
        ax_b.text(x_text, y_text, label, ha=align, fontsize=5.6, color=INK)
    ax_b.set_yticks(yy, metric_labels)
    ax_b.set_xlim(50, 105)
    ax_b.set_xlabel("M3 相对全局包络（%）", fontsize=6.7, color=INK)
    ax_b.set_title("情境化：更窄，覆盖几乎不变", loc="left", fontsize=8.3, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_b, grid_axis="x")
    quantile_map = dict(zip(width_quantiles["quantile"], width_quantiles["width"]))
    ax_b.text(
        0.02,
        0.03,
        "M3 90% 带宽：P10 / 中位数 / P90\n"
        f"{quantile_map[0.1]:.3f} / {quantile_map[0.5]:.3f} / {quantile_map[0.9]:.3f} rad",
        transform=ax_b.transAxes,
        fontsize=5.6,
        color=MUTED,
        va="bottom",
    )

    # Test abstention heterogeneity.
    add_panel_label(ax_c, "c")
    order = [
        ("source_dataset", "waymo_train"),
        ("source_dataset", "nuplan_train"),
        ("source_dataset", "lyft_train_full"),
        ("source_dataset", "av2_motion_forecasting"),
        ("geometry_path_category", "MP"),
        ("geometry_path_category", "F"),
        ("geometry_path_category", "HO"),
        ("geometry_path_category", "CP"),
        ("priority_role", "priority"),
        ("priority_role", "yield"),
        ("priority_role", "equal"),
        ("agent_type_pair", "HV;HV"),
        ("agent_type_pair", "AV;HV"),
    ]
    axis_colors = {
        "source_dataset": "#5C7EA8",
        "geometry_path_category": "#8A6BA5",
        "priority_role": "#B18A42",
        "agent_type_pair": "#4E8D84",
    }
    label_map = {
        **SOURCE_LABELS,
        "priority": "priority",
        "yield": "yield",
        "equal": "equal",
        "HV;HV": "HV–HV",
        "AV;HV": "AV–HV",
    }
    plot_rows = []
    for axis, level in order:
        row = abstention[abstention["axis"].eq(axis) & abstention["level"].eq(level)].iloc[0]
        plot_rows.append(row)
    y_positions = np.arange(len(plot_rows))[::-1]
    for index, (ypos, row) in enumerate(zip(y_positions, plot_rows)):
        if index in (0, 4, 8, 11):
            group_end = y_positions[index] + 0.46
            group_start_index = {0: 3, 4: 7, 8: 10, 11: 12}[index]
            group_start = y_positions[group_start_index] - 0.46
            ax_c.axhspan(group_start, group_end, color=PALE_GREY if index in (0, 8) else "#FAFAFB", zorder=0)
        rate = float(row["abstention_rate"]) * 100
        ax_c.hlines(ypos, 0, rate, color=GRID, linewidth=1.1, zorder=1)
        ax_c.scatter(rate, ypos, s=25, color=axis_colors[row["axis"]], zorder=3)
        ax_c.text(rate + 0.9, ypos, f"{rate:.1f}%", va="center", fontsize=5.7, color=INK)
    overall = float(abstention["overall_abstention_rate"].iloc[0]) * 100
    ax_c.axvline(overall, color=INK, linewidth=0.8, linestyle="--")
    ax_c.text(overall + 0.4, y_positions[0] + 0.52, f"总体 {overall:.2f}%", fontsize=5.6, color=INK, va="bottom")
    ylabels = [f"{label_map.get(row['level'], row['level'])}  (n={comma(row['anchors'])})" for row in plot_rows]
    ax_c.set_yticks(y_positions, ylabels)
    ax_c.set_xlim(0, 36)
    ax_c.set_xlabel("测试集弃权率（%）", fontsize=6.7, color=INK)
    ax_c.set_title("总体 4.78% 会掩盖情境差异：CP 达 31.68%", loc="left", fontsize=8.3, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_c, grid_axis="x")
    ax_c.text(0.0, -0.18, "支持门通过率不等于每个子组的条件覆盖都可靠；264 个已报告子组中 126 个超出 ±3 pp。", transform=ax_c.transAxes, fontsize=5.7, color=MUTED, va="top")

    # Leave-one-dataset-out boundary.
    add_panel_label(ax_d1, "d", x=-0.18)
    lodo_order = ["av2_motion_forecasting", "lyft_train_full", "nuplan_train", "waymo_train"]
    lodo_indexed = lodo.set_index("heldout_source").loc[lodo_order]
    yy = np.arange(len(lodo_order))[::-1]
    coverage_values = lodo_indexed["coverage"].to_numpy(float) * 100
    abstain_values = lodo_indexed["abstention"].to_numpy(float) * 100
    coverage_colors = [PASS if value >= 87 else UPPER for value in coverage_values]
    ax_d1.axvline(90, color=INK, linestyle="--", linewidth=0.8)
    ax_d1.scatter(coverage_values, yy, s=28, color=coverage_colors, zorder=3)
    for value, ypos in zip(coverage_values, yy):
        ax_d1.text(value + (0.7 if value < 97 else -0.7), ypos, f"{value:.1f}%", ha="left" if value < 97 else "right", va="center", fontsize=5.6, color=INK)
    ax_d1.set_yticks(yy, [SOURCE_LABELS[level] for level in lodo_order])
    ax_d1.set_xlim(70, 101)
    ax_d1.set_xlabel("90% 覆盖率", fontsize=6.2, color=INK)
    ax_d1.set_title("跨来源留一验证", loc="left", fontsize=8.3, fontweight="bold", color=INK, pad=5)
    quiet_axis(ax_d1, grid_axis="x")

    ax_d2.barh(yy, abstain_values, color="#D7C6A4", height=0.52)
    for value, ypos in zip(abstain_values, yy):
        ax_d2.text(value + 0.7, ypos, f"{value:.1f}%", va="center", fontsize=5.6, color=INK)
    ax_d2.set_xlim(0, 46)
    ax_d2.set_xlabel("弃权率", fontsize=6.2, color=INK)
    ax_d2.tick_params(axis="y", left=False, labelleft=False)
    quiet_axis(ax_d2, grid_axis="x")
    ax_d2.text(0.5, -0.19, "边际有效 ≠ 无条件跨域有效", transform=ax_d2.transAxes, ha="center", va="top", fontsize=6.2, fontweight="bold", color=UPPER)

    fig.suptitle("验证证据与适用边界：哪些结论成立，哪些不能外推", fontsize=13.0, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, output_dir, "fig3_evidence_and_limits")


def write_manifest(output_dir: Path) -> None:
    rows = [
        ("fig1_verifier_mechanism", "a", "一条记录如何得到带内/超出/弃权？", "fig1_onsite_case.csv", "机制示意；参数来自冻结 M3 合同"),
        ("fig1_verifier_mechanism", "b", "真实车辆如何交互？", "fig1_onsite_case.csv", "OnSite 外部应用；单场景"),
        ("fig1_verifier_mechanism", "c", "动态包络如何随时间变化？", "fig1_onsite_case.csv", "observed 为 target_ipv_future；不是人工意图标签"),
        ("fig1_verifier_mechanism", "d", "为何某些时刻弃权？", "fig1_onsite_case.csv", "25-NN 平均距离与冻结阈值"),
        ("fig2_reference_data", "a", "各折有多少行真正进入参考或判定？", "fig2_gate_flow.csv", "perspective-anchor rows"),
        ("fig2_reference_data", "b", "训练参考来自哪些数据集？", "fig2_train_composition.csv", "训练折精确聚合"),
        ("fig2_reference_data", "c", "训练情境是否均衡？", "fig2_train_composition.csv", "训练折精确聚合"),
        ("fig2_reference_data", "d", "哪些联合情境被支持？", "fig2_joint_support.csv", "anchors>=50 且 unique cases>=10"),
        ("fig3_evidence_and_limits", "a", "门后边际覆盖是否接近标称？", "fig3_m3_coverage.csv", "held-out gate-passing rows"),
        ("fig3_evidence_and_limits", "b", "情境化包络是否更有信息？", "fig3_context_gain.csv", "M0 global 与 M3 frozen context"),
        ("fig3_evidence_and_limits", "c", "测试弃权是否因子组而异？", "fig3_test_abstention.csv", "冻结测试折"),
        ("fig3_evidence_and_limits", "d", "跨来源外推是否稳定？", "fig3_lodo_m3.csv", "leave-one-dataset-out"),
    ]
    pd.DataFrame(rows, columns=["figure", "panel", "question", "source_csv", "note"]).to_csv(output_dir / "figure_manifest.csv", index=False)


def write_source_data(
    output_dir: Path,
    case: pd.DataFrame,
    gate_flow: pd.DataFrame,
    composition: pd.DataFrame,
    joint_support: pd.DataFrame,
    coverage: pd.DataFrame,
    width_hist: pd.DataFrame,
    width_quantiles: pd.DataFrame,
    abstention: pd.DataFrame,
    lodo: pd.DataFrame,
) -> None:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    case.to_csv(source_dir / "fig1_onsite_case.csv", index=False)
    gate_flow.to_csv(source_dir / "fig2_gate_flow.csv", index=False)
    composition.to_csv(source_dir / "fig2_train_composition.csv", index=False)
    joint_support.to_csv(source_dir / "fig2_joint_support.csv", index=False)
    coverage.to_csv(source_dir / "fig3_m3_coverage.csv", index=False)
    width_hist.to_csv(source_dir / "fig3_m3_width_histogram.csv", index=False)
    width_quantiles.to_csv(source_dir / "fig3_m3_width_quantiles.csv", index=False)
    abstention.to_csv(source_dir / "fig3_test_abstention.csv", index=False)
    lodo.to_csv(source_dir / "fig3_lodo_m3.csv", index=False)

    context_gain = pd.DataFrame(
        [
            {"metric": "mean_width", "global_M0": 1.7486657502267058, "frozen_M3": 1.0161520998427664},
            {"metric": "winkler", "global_M0": 2.210260664657108, "frozen_M3": 1.4229346736299133},
            {"metric": "coverage", "global_M0": 0.8986359544971018, "frozen_M3": 0.8986657100797862},
        ]
    )
    context_gain["relative_percent"] = context_gain["frozen_M3"] / context_gain["global_M0"] * 100
    context_gain.to_csv(source_dir / "fig3_context_gain.csv", index=False)

    # Manifest source paths are relative to the source_data directory.
    manifest = pd.read_csv(output_dir / "figure_manifest.csv")
    manifest["source_csv"] = manifest["source_csv"].map(lambda value: f"source_data/{value}")
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)


def verify_sources(
    folds: pd.DataFrame,
    composition: pd.DataFrame,
    joint_support: pd.DataFrame,
    gate_flow: pd.DataFrame,
    coverage: pd.DataFrame,
    case: pd.DataFrame,
) -> None:
    assert int(folds["raw_rows"].sum()) == 6_397_266
    assert int(composition.loc[composition["dimension"].eq("source_dataset"), "anchors"].sum()) == 2_558_374
    assert int(joint_support["anchors"].sum()) == 2_558_374
    assert int(joint_support["supported"].sum()) == 19
    assert int((joint_support["anchors"].gt(0) & ~joint_support["supported"]).sum()) == 4
    assert int(joint_support["anchors"].eq(0).sum()) == 1
    assert int(gate_flow.loc[gate_flow["fold"].eq("test"), "gate_pass_rows"].iloc[0]) == 1_209_857
    m3_90 = coverage[coverage["alpha_label"].eq(90)].iloc[0]
    assert np.isclose(float(m3_90["coverage"]), 0.8986657100797862)
    assert len(case) == 374
    assert int(case["support_gate_pass"].sum()) == 189
    assert int(case["deviation_upper_tail_90"].sum()) == 14
    assert int(case["deviation_lower_tail_90"].sum()) == 8


def build(repo_root: Path, output_dir: Path) -> None:
    rq009_data = repo_root / "data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de"
    rq009_report = repo_root / "reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de"
    rq012_report = repo_root / "reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93"

    matrix_root = rq009_data / "03_features/matrix"
    ood_gate_path = rq009_report / "02_process/04_calibration/ood_gate.json"
    metrics_path = rq009_data / "05_evaluation/metrics_summary.csv"
    c1_source_path = rq009_report / "01_results/figures/c1_validity_envelope.source.csv"
    lodo_path = rq009_data / "05_evaluation/lodo_results.csv"
    scored_path = repo_root / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/ood_gate_multi/onsite_m3_av_anchors_multi_allvalid_scored_ood_gate.parquet"
    trajectory_path = repo_root / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_ipv_timeseries_multi_allvalid.parquet"

    ood_gate = json.loads(ood_gate_path.read_text(encoding="utf-8"))
    folds = matrix_fold_counts(matrix_root)
    composition, joint_support = aggregate_train_matrix(matrix_root)
    gate_flow = build_gate_flow(folds, ood_gate)
    case, trajectory = load_onsite_case(scored_path, trajectory_path)
    coverage, width_hist, width_quantiles, abstention, lodo = load_validation_sources(
        metrics_path,
        c1_source_path,
        lodo_path,
        ood_gate,
    )

    verify_sources(folds, composition, joint_support, gate_flow, coverage, case)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir)
    write_source_data(
        output_dir,
        case,
        gate_flow,
        composition,
        joint_support,
        coverage,
        width_hist,
        width_quantiles,
        abstention,
        lodo,
    )
    figure_1_mechanism(case, trajectory, output_dir)
    figure_2_reference_data(gate_flow, composition, joint_support, output_dir)
    figure_3_evidence_limits(coverage, width_quantiles, abstention, lodo, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: docs/assets/ipv_verifier).",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (args.output_dir or repo_root / "docs/assets/ipv_verifier").resolve()
    build(repo_root, output_dir)
    print(f"Wrote IPV verifier explainer assets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
