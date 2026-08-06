#!/usr/bin/env python3
"""手稿 Figure 3：条件化的人类参照区间是一个可用且已标定的在线监控器。

核心结论（图必须为其辩护）：
    不按情境条件化，就没有参照可言——全局参照在 95% 水平上正好等于 IPV 的整个取值域；
    条件化之后的参照区间既收窄又保持标定；
    但情境只解释了读数方差的两成，所以交互偏好必须在线测量，不能由情境推出。

面板：
    (a) 三个名义水平上的平均区间宽度：全局 vs 情境条件化，附取值域上限参考线
    (b) 实际覆盖率减名义覆盖率（百分点）
    (c) 情境能解释在线读数方差的比例（out-of-fold R²）

数据来源（唯一）：同目录 `key_numbers.json` 的
    human_only_envelope.metrics                                  条件化包络
    human_only_envelope.circularity_diagnostics.marginal_envelopes.ipv_log.metrics   全局包络
    human_only_envelope.circularity_diagnostics.D2_contemporaneous_test_r2           R²

archetype: quantitative grid。所有绘图、预览与导出均为 Python（matplotlib）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PAPER_FIGS = (
    REPO.parent.parent
    / "2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/figures"
)
KEY_NUMBERS = HERE / "key_numbers.json"

# IPV 候选网格 [-3..3] x pi/8 ⇒ 取值域宽度 6*pi/8
IPV_DOMAIN_WIDTH = 6.0 * np.pi / 8.0

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

C_GLOBAL = "#A8A8A8"     # 全局参照（基线）
C_CONTEXT = "#42949E"    # 情境条件化（本文的监控器）
C_REST = "#DCDCDC"       # 未被情境解释的部分
C_RULE = "#606060"

LBL_G = "Global range"
LBL_C = "Conditioned on the situation"
ALPHAS = ["80", "90", "95"]


def panel_label(ax, text: str) -> None:
    ax.text(-0.22, 1.06, text, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def load() -> dict:
    k = json.loads(KEY_NUMBERS.read_text(encoding="utf-8"))["human_only_envelope"]
    diag = k["circularity_diagnostics"]
    return {
        "cond": {a: k["metrics"][a] for a in ALPHAS},
        "marg": {a: diag["marginal_envelopes"]["ipv_log"]["metrics"][a] for a in ALPHAS},
        "r2": float(diag["D2_contemporaneous_test_r2"]["r2"]),
        "r2_n": int(diag["D2_contemporaneous_test_r2"]["rows"]),
        "n_eval": int(k["metrics"]["90"]["n"]),
    }


def panel_a(ax, d: dict) -> None:
    """宽度以「占 IPV 取值域的百分比」表示：全局参照在 95% 水平正好等于 100%。"""
    x = np.arange(len(ALPHAS))
    w = 0.36
    g = [100 * d["marg"][a]["mean_width"] / IPV_DOMAIN_WIDTH for a in ALPHAS]
    c = [100 * d["cond"][a]["mean_width"] / IPV_DOMAIN_WIDTH for a in ALPHAS]

    ax.axhline(100, color=C_RULE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(-0.48, 102, "entire admissible range", fontsize=6.3, color=C_RULE,
            ha="left", va="bottom")

    ax.bar(x - w / 2, g, w, color=C_GLOBAL, label=LBL_G, zorder=2)
    ax.bar(x + w / 2, c, w, color=C_CONTEXT, label=LBL_C, zorder=2)

    for xi, (gi, ci) in enumerate(zip(g, c)):
        ax.text(xi - w / 2, gi - 4, f"{gi:.0f}", ha="center", va="top", fontsize=6.3,
                color="white", fontweight="bold")
        ax.text(xi + w / 2, ci - 4, f"{ci:.0f}", ha="center", va="top", fontsize=6.3,
                color="white", fontweight="bold")
        ax.annotate(f"−{100 * (1 - ci / gi):.0f}%", xy=(xi + w / 2, ci), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.4,
                    color=C_CONTEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}%" for a in ALPHAS])
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Width of the reference range\n(% of the admissible range)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    panel_label(ax, "a")


def panel_b(ax, d: dict) -> None:
    x = np.arange(len(ALPHAS))
    w = 0.36
    g = [100 * (d["marg"][a]["coverage"] - int(a) / 100) for a in ALPHAS]
    c = [100 * (d["cond"][a]["coverage"] - int(a) / 100) for a in ALPHAS]

    ax.axhline(0, color=C_RULE, lw=0.8, zorder=1)
    ax.bar(x - w / 2, g, w, color=C_GLOBAL, zorder=2)
    ax.bar(x + w / 2, c, w, color=C_CONTEXT, zorder=2)

    for xi, (gi, ci) in enumerate(zip(g, c)):
        for xoff, v, col in ((-w / 2, gi, C_RULE), (w / 2, ci, C_CONTEXT)):
            off = 0.09 if v >= 0 else -0.09
            ax.text(xi + xoff, v + off, f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.2, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}%" for a in ALPHAS])
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Achieved − nominal coverage (pp)")
    ax.set_ylim(-0.85, 3.35)
    panel_label(ax, "b")


def panel_c(ax, d: dict) -> None:
    r2 = d["r2"]
    ax.barh([0], [100 * r2], height=0.42, color=C_CONTEXT, zorder=2)
    ax.barh([0], [100 * (1 - r2)], left=[100 * r2], height=0.42, color=C_REST, zorder=2)

    ax.text(100 * r2 / 2, 0, f"{100 * r2:.1f}%", ha="center", va="center",
            fontsize=7.2, color="white", fontweight="bold")
    ax.text(100 * r2 + 100 * (1 - r2) / 2, 0, f"{100 * (1 - r2):.1f}%", ha="center",
            va="center", fontsize=7.2, color=C_RULE, fontweight="bold")

    ax.annotate("explained by\nthe situation", xy=(100 * r2, 0.21),
                xytext=(100 * r2 + 6, 0.66), fontsize=6.3, color=C_CONTEXT,
                ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_CONTEXT,
                                shrinkA=0, shrinkB=1))

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Variance of the online reading (%)")
    ax.spines["left"].set_visible(False)
    panel_label(ax, "c")


def main() -> None:
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.26, top=0.90, wspace=0.44)
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    panel_c(axes[2], d)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.30, -0.015),
               handlelength=1.1, columnspacing=1.6, borderpad=0.2)

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = PAPER_FIGS / f"fig3_monitor.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print("wrote", out)

    print("\n--- caption numbers ---")
    for a in ALPHAS:
        g, c = d["marg"][a], d["cond"][a]
        print(f"  alpha={a}: global width {g['mean_width']:.4f} cov {g['coverage']:.4f} | "
              f"conditioned width {c['mean_width']:.4f} cov {c['coverage']:.4f} | "
              f"narrowing {100 * (1 - c['mean_width'] / g['mean_width']):.2f}%")
    print(f"  admissible domain width = {IPV_DOMAIN_WIDTH:.6f} rad; "
          f"global 95% width = {d['marg']['95']['mean_width']:.6f} "
          f"({100 * d['marg']['95']['mean_width'] / IPV_DOMAIN_WIDTH:.2f}% of domain)")
    print(f"  out-of-fold R2 = {d['r2']:.6f} on n = {d['r2_n']:,}")
    print(f"  evaluated moments n = {d['n_eval']:,}")
    print(f"  mechanism-two abstention = {d['cond']['90']['abstention']:.6f} "
          f"({d['cond']['90']['abstained_n']:,}/{d['cond']['90']['total_n']:,})")


if __name__ == "__main__":
    main()
