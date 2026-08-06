#!/usr/bin/env python3
"""手稿 Figure 5：From atypicality to interaction consequences.

核心结论（图必须为其辩护）：
    当监控器判定「强势侧非典型」时，随后的交互在双方都被压缩；
    但两侧的紧急事件都变得更少，而不是更多。

面板矩阵：行 = 交互的哪一侧，列 = 分布的哪一部分
    (a) 自车裕度分布左移        (b) 自车危险阈值占比更低
    (c) 对手方常规反应约 2×      (d) 对手方强制动占比更低

archetype: quantitative grid。所有绘图、预览与导出均为 Python（matplotlib）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PAPER_FIGS = (
    REPO.parent.parent
    / "2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/figures"
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

# 一个中性族（典型）+ 一个信号族（非典型）+ 一个弱化族（未获支持）
C_TYPICAL = "#7884B4"   # 区间内 = 人类参照范围内
C_ATYPICAL = "#B64342"  # 下侧越界 = 比人类更强势
C_REJECT = "#A8A8A8"    # 未获支持的量
C_RULE = "#606060"

LBL_TYP = "Within human range"
LBL_ATY = "Atypical (assertive side)"


def load():
    data = json.loads((HERE / "fig5_data.json").read_text(encoding="utf-8"))
    ego = pd.read_parquet(HERE / "fig5_ego_ttc.parquet")
    cp = pd.read_parquet(HERE / "fig5_counterpart_outcomes.parquet")
    return data, ego, cp


def panel_a(ax, ego, data):
    """自车未来最小 TTC 的 ECDF：主体左移，低分位几乎重合。标注全部从数据读取。"""
    q = data["ego_ttc_quantiles"]
    med_in = float(q["inside"]["q"]["50"])
    med_lo = float(q["lower"]["q"]["50"])
    q25_in = float(q["inside"]["q"]["25"])
    q25_lo = float(q["lower"]["q"]["25"])
    med_drop_pct = 100.0 * (med_lo / med_in - 1.0)

    for band, color, label in ((\
            "inside", C_TYPICAL, LBL_TYP), ("lower", C_ATYPICAL, LBL_ATY)):
        v = np.sort(ego.loc[ego.band == band, "future_min_ttc_s"].values)
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=color, label=f"{label} (n={len(v):,})")
        med = np.median(v)
        ax.plot([med], [0.5], marker="o", ms=3.2, color=color, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlim(0.5, 200)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Ego margin to counterpart after the verdict (s, log scale)")
    ax.set_ylabel("Cumulative fraction of moments")
    ax.axhline(0.25, color=C_RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.axhline(0.5, color=C_RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)

    ax.annotate(f"lower quartile\nnearly unchanged\n({q25_lo:.2f} vs {q25_in:.2f} s)",
                xy=(min(q25_lo, q25_in), 0.25), xytext=(0.72, 0.40), fontsize=6,
                color=C_RULE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_RULE,
                                shrinkA=0, shrinkB=1.5))
    ax.annotate(f"median\n$-${abs(med_drop_pct):.1f}%", xy=(med_in * 0.86, 0.5),
                xytext=(17, 0.30),
                fontsize=6, color=C_ATYPICAL, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_ATYPICAL,
                                shrinkA=0, shrinkB=1.5))
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3,
              labelspacing=0.35)
    ax.set_title("Ordinary interaction compresses", loc="left", fontweight="bold", pad=6)


def _threshold_bars(ax, thresholds, lower, inside, sig, xlabel):
    """(b)/(d) 共用：成对柱 + 显著性标记。"""
    x = np.arange(len(thresholds))
    w = 0.36
    ax.bar(x - w / 2, np.array(inside) * 100, w, color=C_TYPICAL, label=LBL_TYP)
    ax.bar(x + w / 2, np.array(lower) * 100, w, color=C_ATYPICAL, label=LBL_ATY)
    for i, (lo, ins, s) in enumerate(zip(lower, inside, sig)):
        top = max(lo, ins) * 100
        if s:
            ax.text(i, top + 0.55, "$\\ast$", ha="center", va="bottom", fontsize=7,
                    color=C_RULE)
        ax.annotate("", xy=(i + w / 2, lo * 100 + 0.15), xytext=(i - w / 2, ins * 100 + 0.15),
                    arrowprops=dict(arrowstyle="->", lw=0.55, color=C_RULE,
                                    connectionstyle="arc3,rad=-0.25"))
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Moments in that state (%)")
    ax.margins(y=0.22)


def panel_b(ax, data):
    ks = ["ttc_lt_1.0", "ttc_lt_1.5", "ttc_lt_2.0", "ttc_lt_3.0"]
    labels = ["< 1 s", "< 1.5 s", "< 2 s", "< 3 s"]
    lower = [data["ego_danger_shares"][k]["lower"]["share"] for k in ks]
    inside = [data["ego_danger_shares"][k]["inside"]["share"] for k in ks]
    boot = data["ego_danger_bootstrap_all"]   # 四个阈值全部检验过
    sig = [boot[k.replace("ttc_lt_1.0", "ttc_lt_1.0")]["excludes_zero"] for k in ks]
    _threshold_bars(ax, labels, lower, inside, sig, "Ego margin below threshold")
    ax.set_title("Emergency events do not increase", loc="left", fontweight="bold", pad=6)
    ax.text(0.98, 0.95, "fewer, not more", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=C_ATYPICAL, style="italic")


def panel_c(ax, data):
    """对手方反应的 lower/inside 中位数比值 + case 层 bootstrap CI。"""
    keys = ["anchor_speed_drop_kmh", "speed_range_kmh",
            "total_heading_change_deg", "max_abs_yaw_rate_dps"]
    names = ["Speed reduction\nfrom the verdict", "Speed range\nover the window",
             "Net heading\nchange", "Peak yaw rate"]
    rb = data["counterpart_ratio_bootstrap"]
    y = np.arange(len(keys))[::-1]

    xmax = 20.0
    label_x = 46.0        # 数值标签固定成一列，避免右侧参差与空白
    for yi, k in zip(y, keys):
        r = rb[k]
        ok = r["excludes_one"]
        stable = (r["ci95"][1] - r["ci95"][0]) < 5.0
        color = C_ATYPICAL if (ok and stable) else C_REJECT
        lo, hi = r["ci95"]
        clipped = hi > xmax
        ax.plot([lo, min(hi, xmax)], [yi, yi], color=color, lw=1.1,
                solid_capstyle="butt", zorder=2)
        for b in (lo,) + ((hi,) if not clipped else ()):
            ax.plot([b, b], [yi - 0.11, yi + 0.11], color=color, lw=1.0, zorder=2)
        if clipped:
            ax.annotate("", xy=(xmax, yi), xytext=(xmax - 2.2, yi),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=color))
        ax.plot([r["ratio"]], [yi], marker="o", ms=4.2, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        note = f"{r['ratio']:.2f}×"
        if not (ok and stable):
            note += "\nnot supported"
        ax.text(label_x, yi, note, va="center", ha="right", fontsize=6,
                color=color if (ok and stable) else C_RULE, linespacing=1.25)

    ax.axvline(1.0, color=C_RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.82, 52)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1×", "2×", "5×", "10×", "20×"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlabel("Counterpart response, atypical ÷ within-range (median ratio)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_d(ax, data):
    rows = sorted(data["counterpart_braking_thresholds"], key=lambda r: -r["threshold_mps2"])
    labels = [f"$<\\!-${abs(int(r['threshold_mps2']))} m s$^{{-2}}$" for r in rows]
    lower = [r["lower_share"] for r in rows]
    inside = [r["inside_share"] for r in rows]
    sig = [(r["ci95"][0] < 0 and r["ci95"][1] < 0) for r in rows]
    _threshold_bars(ax, labels, lower, inside, sig, "Counterpart braking below threshold")
    ax.text(0.98, 0.95, "fewer, not more", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=C_ATYPICAL, style="italic")


def main():
    data, ego, cp = load()
    fig = plt.figure(figsize=(7.09, 4.9))  # 180 mm double column
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.92],
                          wspace=0.34, hspace=0.52,
                          left=0.115, right=0.975, top=0.90, bottom=0.095)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    panel_a(ax_a, ego, data)
    panel_b(ax_b, data)
    panel_c(ax_c, data)
    panel_d(ax_d, data)

    for ax, lab in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        ax.text(-0.155 if ax in (ax_a, ax_c) else -0.135, 1.06, lab,
                transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    # 行标签：交互的哪一侧
    fig.text(0.012, 0.735, "E G O   V E H I C L E", rotation=90, va="center",
             ha="center", fontsize=6.5, fontweight="bold", color=C_RULE)
    fig.text(0.012, 0.275, "C O U N T E R P A R T", rotation=90, va="center",
             ha="center", fontsize=6.5, fontweight="bold", color=C_RULE)

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(PAPER_FIGS / f"fig5_consequence.{ext}", bbox_inches="tight", **kw)
    print(f"wrote {PAPER_FIGS/'fig5_consequence.pdf'}")
    print(f"wrote {PAPER_FIGS/'fig5_consequence.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
