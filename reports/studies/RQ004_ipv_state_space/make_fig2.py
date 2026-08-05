#!/usr/bin/env python3
"""手稿 Figure 2：Human social behaviour is context-dependent.

核心结论（图必须为其辩护）：
    人类的交互倾向不是一个可以外推的固定数值，而是随情境状态变化的响应面：
    同一个「优先权」标签在高风险与低风险下含义相反；粗几何是跨来源稳定的先验；
    但把这些拼成一条可跨数据集迁移的定律并不成立。

面板：
    (a) hero —— 优先权差随风险反转（PET 三段）
    (b)      —— 粗几何先验在四个来源上一致为正
    (c)      —— 边界：留一数据集外推不成立（这一格是限制，不是卖点）

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
RUN = HERE / "RQ004_1_state_space_law_nature_20260618"
GAP = RUN / "02_process/agent_I_figures/claim3_F3_case_priority_gap.csv"
GEO = RUN / "02_process/agent_E_geometry/round3_geometry_prior_results.csv"
LODO = RUN / "02_process/agent_I_figures/F7_lodo_summary_source_data.csv"
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

C_YIELD = "#42949E"     # 让行方向
C_ASSERT = "#B64342"    # 争取方向
C_MP = "#484878"
C_SS = "#7884B4"
C_RULE = "#606060"
C_NEG = "#A8A8A8"

BANDS = [("PET<=1.0", "High risk\nPET $\\leq$ 1 s"),
         ("1.0<PET<=2.0", "Middle\n1–2 s"),
         ("PET>2.0", "Low risk\nPET > 2 s")]


def panel_a(ax):
    """优先权差随风险反转——本节最锐利的结果。"""
    d = pd.read_csv(GAP)
    x = np.arange(len(BANDS))
    means, los, his, ns = [], [], [], []
    rng = np.random.default_rng(0)
    for key, _ in BANDS:
        v = d.loc[d.pet_bin == key, "priority_minus_nonpriority_ipv"].values
        m = v.mean()
        boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        means.append(m); los.append(lo); his.append(hi); ns.append(len(v))

    ax.axhline(0, color=C_RULE, lw=0.8, zorder=1)
    for xi, m, lo, hi, n in zip(x, means, los, his, ns):
        col = C_YIELD if m > 0 else C_ASSERT
        ax.plot([xi, xi], [lo, hi], color=col, lw=1.3, zorder=2, solid_capstyle="butt")
        ax.plot([xi], [m], marker="o", ms=6, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.text(xi, hi + 0.004, f"{m:+.3f}", ha="center", va="bottom",
                fontsize=6.8, color=col, fontweight="bold")
        ax.text(xi, lo - 0.004, f"n={n:,}", ha="center", va="top", fontsize=5.8, color=C_RULE)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in BANDS])
    ax.set_ylabel("Priority $-$ non-priority (rad)")
    ax.set_xlim(-0.55, len(BANDS) - 0.45)
    ax.margins(y=0.30)
    ax.set_title("What holding priority means depends on the risk",
                 loc="left", fontweight="bold", pad=6)
    ax.text(0.02, 0.955, "more accommodating", transform=ax.transAxes, fontsize=6,
            color=C_YIELD, style="italic", va="top")
    ax.text(0.02, 0.045, "more assertive", transform=ax.transAxes, fontsize=6,
            color=C_ASSERT, style="italic", va="bottom")
    ax.text(0.985, -0.30, "bars: 95% bootstrap CI", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=C_RULE, style="italic")


def panel_b(ax):
    """粗几何先验：四个来源方向一致。"""
    d = pd.read_csv(GEO)
    d = d[d.contrast_id.notna()]
    sources = ["AV2", "Lyft", "Waymo", "nuPlan"]
    x = np.arange(len(sources))
    w = 0.34
    for cid, color, lab, off in (("MP_minus_nonMP", C_MP, "Merge/pass geometry", -w / 2),
                                 ("SS_minus_nonSS", C_SS, "Same-direction pair", +w / 2)):
        sub = d[d.contrast_id == cid].set_index("dataset_short")
        eff = [sub.loc[s, "effect"] for s in sources]
        lo = [sub.loc[s, "effect"] - sub.loc[s, "ci_low"] for s in sources]
        hi = [sub.loc[s, "ci_high"] - sub.loc[s, "effect"] for s in sources]
        ax.bar(x + off, eff, w, color=color, label=lab,
               yerr=[lo, hi], error_kw=dict(lw=0.7, ecolor=C_RULE, capsize=1.6))
    ax.axhline(0, color=C_RULE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Difference in reading (rad)")
    ax.margins(y=0.30)
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("Coarse geometry points the same way in every source",
                 loc="left", fontweight="bold", pad=6)


def panel_c(ax):
    """边界：把状态面拼成可迁移定律不成立。"""
    d = pd.read_csv(LODO)
    d = d[d.outcome_spec == "case_mean_ipv"].sort_values("full_state_space_r2")
    y = np.arange(len(d))[::-1]
    ax.axvline(0, color=C_RULE, lw=0.8, zorder=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        v = r.full_state_space_r2
        col = C_NEG if v <= 0 else C_MP
        ax.plot([0, v], [yi, yi], color=col, lw=1.1, zorder=2)
        ax.plot([v], [yi], marker="o", ms=4.4, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(v - 0.012 if v < 0 else v + 0.012, yi, f"{v:+.3f}",
                va="center", ha="right" if v < 0 else "left", fontsize=6, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(d.holdout_dataset)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlim(-0.40, 0.16)
    ax.set_xlabel("Variance explained on the held-out source ($R^2$)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("It does not become a transferable law", loc="left",
                 fontweight="bold", pad=6)
    ax.text(0.5, -0.46,
            "fitting every source but one and predicting the last leaves $R^2$ at best barely above zero",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=C_RULE, style="italic")


def main():
    fig = plt.figure(figsize=(7.09, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.88],
                          wspace=0.40, hspace=0.62,
                          left=0.105, right=0.975, top=0.875, bottom=0.145)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    for ax, lab, dx in ((ax_a, "a", -0.175), (ax_b, "b", -0.135), (ax_c, "c", -0.088)):
        ax.text(dx, 1.13, lab, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(PAPER_FIGS / f"fig2_context.{ext}", bbox_inches="tight", **kw)
    print(f"wrote {PAPER_FIGS/'fig2_context.pdf'}")
    print(f"wrote {PAPER_FIGS/'fig2_context.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
