#!/usr/bin/env python3
"""手稿 Figure 1：When social behaviour can be meaningfully measured.

核心结论（图必须为其辩护）：
    这个交互倾向读数只在交互真正发生时才变得可读，而「可读」既不等于
    行为已经定型，也不意味着可以用单一数值概括整段交互。

面板：
    (a) hero —— 可读性是交互条件化的：真实缺口 vs 三条对照
    (b)      —— 可读 ≠ 已定型：可读帧里读数反而移动得更多
    (c)      —— 概括规则会改变结论：三种 episode 概括口径的分布与符号翻转

**刻意不画的**：可读性缺口的来源分解（空间邻近 vs 冲突几何）。
手稿的 claims register 对该主张明令 "write NO source attribution"——正反两个方向
都不作归因，因此相关对照（history_matched_residual、nearby_nonconflicting）不进图，
以免图替正文作出被禁止的归因。

archetype: quantitative grid（hero + 从属）。全部绘图与导出均为 Python/matplotlib。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = HERE / "RQ007_1_ipv_estimability_20260622T155229Z_289d9a99"
CONTROLS = RUN / "02_process/05_controls/controls_results.csv"
F3 = RUN / "01_results/traces/rq007_f3_estimable_not_settled_source.csv"
F4D = RUN / "01_results/traces/rq007_f4_summary_distribution_source.csv"
F4S = RUN / "01_results/traces/rq007_f4_summary_sensitivity_source.csv"
PAPER_FIGS = (
    REPO.parent.parent / "2_PaperWriting"
    / "NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/figures"
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "lines.linewidth": 1.1,
})

C_REAL = "#42949E"      # 真实交互条件（与 Fig 3 的方法色一致）
C_CTRL = "#A8A8A8"      # 对照
C_EGO = "#484878"       # 自车
C_CP = "#B4C0E4"        # 对手方
C_RULE = "#606060"
C_ACCENT = "#B64342"

# 只保留确立「交互条件化 / 时间锁定 / 对手方特异」的对照。
# 归因类对照按 claims register 的禁令排除，见模块 docstring。
CONTROL_ROWS = [
    ("time_shift_alignment", "Same interaction,\nmisaligned in time"),
    ("pseudo_pair_alignment", "Same moment,\na different partner"),
    ("distant_no_opportunity", "No interaction\nopportunity"),
]


def panel_a(ax):
    d = pd.read_csv(CONTROLS)
    d = d[d.split == "development"].set_index("control_id")
    real = d.iloc[0]

    labels = ["Real interaction"] + [lab for _, lab in CONTROL_ROWS]
    vals = [real.real_gap] + [d.loc[cid, "control_gap"] for cid, _ in CONTROL_ROWS]
    los = [real.real_ci_low] + [d.loc[cid, "ci_low"] for cid, _ in CONTROL_ROWS]
    his = [real.real_ci_high] + [d.loc[cid, "ci_high"] for cid, _ in CONTROL_ROWS]
    cols = [C_REAL] + [C_CTRL] * len(CONTROL_ROWS)

    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color=C_RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for yi, v, lo, hi, c in zip(y, vals, los, his, cols):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.plot([v], [yi], marker="o", ms=4.6, color=c, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.text(max(hi, v) + 0.004, yi, f"{v:+.3f}", ha="left", va="center", fontsize=6.2,
                color=c, fontweight="bold" if c == C_REAL else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.set_xlim(-0.15, 0.052)
    ax.set_xlabel("Change in how sharply the reading is identified (negative = more sharply identified)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("The reading becomes legible only during a real interaction",
                 loc="left", fontweight="bold", pad=6)
    ax.text(0.985, -0.34, "bars: 95% CI over case clusters", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=C_RULE, style="italic")


def panel_b(ax):
    d = pd.read_csv(F3)
    d = d[d.statistic == "mean_abs_dtheta"]
    zones = [("event_window_low_index", "Legible"), ("higher_index_gt_tau", "Less legible")]
    x = np.arange(len(zones))
    w = 0.34
    for role, color, off, lab in (("ego", C_EGO, -w / 2, "Ego"),
                                  ("counterpart", C_CP, +w / 2, "Counterpart")):
        v = [float(d[(d.role == role) & (d.concentration_zone == z)].value.iloc[0]) for z, _ in zones]
        ax.bar(x + off, v, w, color=color, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.008, f"{vi:.2f}", ha="center", va="bottom", fontsize=6, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in zones])
    ax.set_ylabel("Movement of the reading, |$\\Delta\\theta$| (rad)")
    ax.margins(y=0.24)
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("Legible does not mean settled", loc="left", fontweight="bold", pad=6)


def panel_c(ax):
    d = pd.read_csv(F4D)
    d = d[d.split == "development"]
    order = [("all_valid_mean", "Every valid frame"),
             ("interaction_active_mean", "Only while interacting"),
             ("estimability_weighted_mean", "Weighted by legibility")]
    y = np.arange(len(order))[::-1]
    ax.axvline(0, color=C_RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for yi, (rule, lab) in zip(y, order):
        r = d[d.rule == rule].iloc[0]
        ax.plot([r["q25"], r["q75"]], [yi, yi], color=C_EGO, lw=3.4,
                solid_capstyle="butt", alpha=0.45, zorder=2)
        ax.plot([r["median"]], [yi], marker="|", ms=9, mew=1.4, color=C_EGO, zorder=3)
        ax.text(r["q75"] + 0.03, yi, f"median {r['median']:+.2f}", va="center",
                fontsize=6, color=C_EGO)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.42, 1.02)
    ax.set_xlabel("Episode-level summary of the reading (rad)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("The summary rule changes the answer", loc="left", fontweight="bold", pad=6)

    s = pd.read_csv(F4S)
    s = s[(s.split == "development") &
          (s.rule_pair == "all_valid_mean__vs__interaction_active_mean")].iloc[0]
    ax.text(0.5, -0.40,
            f"the first two rules disagree by {s.mean_abs_delta:.2f} rad on average and flip the "
            f"sign in {s.frac_sign_change*100:.0f}% of episodes",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=C_ACCENT, style="italic")


def main():
    fig = plt.figure(figsize=(7.09, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.78, 1.0], height_ratios=[1.0, 0.98],
                          wspace=0.62, hspace=0.72,
                          left=0.145, right=0.975, top=0.905, bottom=0.135)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    for ax, lab, dx in ((ax_a, "a", -0.185), (ax_b, "b", -0.235), (ax_c, "c", -0.40)):
        ax.text(dx, 1.08, lab, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(PAPER_FIGS / f"fig1_measurable.{ext}", bbox_inches="tight", **kw)
    print(f"wrote {PAPER_FIGS/'fig1_measurable.pdf'}")
    print(f"wrote {PAPER_FIGS/'fig1_measurable.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
