#!/usr/bin/env python3
"""⚠ SUPERSEDED 2026-08-06 —— 本脚本产出的是**旧** Figure 3，不要再跑它来出图。

手稿现行的 Figure 3 由 RQ021 的同期 envelope 重建，脚本在
`reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/make_fig3.py`。

本脚本读的是 RQ009 的 `calibration_gate.json`，其 envelope 的目标量是锚点之后 [t+3,t+6] 的
IPV，而机制二在线时比较的是锚点当下 [t-9,t] 的读数——不是同一个量
（见 `reports/knowledge/RQ021_contemporaneous_envelope/decision.md`）。

输出路径已改为本目录，**不再写入论文仓库**，以免静默覆盖现行 Figure 3。
保留本文件是为了让旧图可复现，作为证据链的一部分。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = HERE / "RQ009_1_dynamic_envelope_20260625T121905Z_98c433de"
GATE = RUN / "02_process/04_calibration/calibration_gate.json"
LODO = RUN / "02_process/05_evaluation/lodo_results.csv"
# SUPERSEDED：不再写入论文仓库，只在本目录留存旧图，见文件顶部说明。
PAPER_FIGS = HERE / "superseded_fig3_rq009_envelope"

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

# 一个中性族（被改进的基线）+ 一个方法族 + 一个警示族
C_GLOBAL = "#A8A8A8"    # 全局参照范围（基线）
C_CONTEXT = "#42949E"   # 情境条件化（本文的监控器）
C_RULE = "#606060"
C_WARN = "#B64342"      # 落在容许带之外的留出源

LBL_G = "Global range"
LBL_C = "Conditioned on the situation"
ALPHAS = ["80", "90", "95"]


def load():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    mc = gate["marginal_coverage"]
    lodo = pd.read_csv(LODO)
    lodo = lodo[(lodo.tier == "M2") & (lodo.alpha_label == 90)].copy()
    return mc, gate["abstention_overall"], lodo


def panel_a(ax, mc, abstention):
    """hero：区间宽度。变窄多少，是这一节的主结果。"""
    x = np.arange(len(ALPHAS))
    w = 0.34
    g = [mc["M0"][a]["mean_width"] for a in ALPHAS]
    c = [mc["M2"][a]["mean_width"] for a in ALPHAS]
    ax.bar(x - w / 2, g, w, color=C_GLOBAL, label=LBL_G)
    ax.bar(x + w / 2, c, w, color=C_CONTEXT, label=LBL_C)

    for i, (gi, ci) in enumerate(zip(g, c)):
        pct = 100 * (ci / gi - 1)
        ax.annotate("", xy=(i + w / 2, ci + 0.05), xytext=(i - w / 2, gi + 0.05),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color=C_RULE,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text(i, max(gi, ci) + 0.14, f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=6.8, color=C_CONTEXT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}%" for a in ALPHAS])
    ax.set_xlabel("Nominal coverage of the reference range")
    ax.set_ylabel("Mean interval width (rad)")
    ax.margins(y=0.28)
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3, labelspacing=0.35)
    ax.set_title("Conditioning on the situation sharpens the range",
                 loc="left", fontweight="bold", pad=6)
    ax.text(0.985, 0.055,
            f"abstains on {abstention*100:.2f}% of moments\nwhere the situation is unsupported",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            color=C_RULE, style="italic", linespacing=1.3)


def panel_b(ax, mc):
    """校准：变窄没有以覆盖为代价。用相对名义的偏差，比对角线图在小面板里更可读。"""
    x = np.arange(len(ALPHAS))
    w = 0.34
    for tier, color, label, off in (("M0", C_GLOBAL, LBL_G, -w / 2),
                                    ("M2", C_CONTEXT, LBL_C, +w / 2)):
        dev = [(mc[tier][a]["coverage"] - float(mc[tier][a]["nominal"])) * 100 for a in ALPHAS]
        ax.bar(x + off, dev, w, color=color, label=label)
        for xi, d in zip(x + off, dev):
            ax.text(xi, d + (0.055 if d >= 0 else -0.055), f"{d:+.2f}",
                    ha="center", va="bottom" if d >= 0 else "top", fontsize=5.8, color=color)
    ax.axhline(0, color=C_RULE, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}%" for a in ALPHAS])
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Achieved $-$ nominal (pp)")
    ax.set_ylim(-0.75, 2.4)
    ax.set_title("Coverage stays near nominal", loc="left", fontweight="bold", pad=6)
    ax.text(0.985, 0.97, "positive = conservative", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=C_RULE, style="italic")


def panel_c(ax, lodo):
    """迁移边界：跨来源不是无条件成立。这一格是限制，不是卖点。"""
    lodo = lodo.sort_values("coverage")
    names = [s.replace("_train_full", "").replace("_train", "").replace("_motion_forecasting", "")
             for s in lodo.heldout_source]
    y = np.arange(len(lodo))[::-1]
    cov = lodo.coverage.values
    inside = np.abs(cov - 0.90) <= 0.03

    ax.axvspan(0.87, 0.93, color=C_RULE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(0.90, color=C_RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for yi, cv, ok, n in zip(y, cov, inside, lodo.n_cases.values):
        col = C_CONTEXT if ok else C_WARN
        ax.plot([0.90, cv], [yi, yi], color=col, lw=0.9, zorder=2)
        ax.plot([cv], [yi], marker="o", ms=4.2, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(cv + (0.012 if cv > 0.90 else -0.012), yi, f"{cv:.3f}",
                va="center", ha="left" if cv > 0.90 else "right", fontsize=6, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(lodo) - 0.05)
    ax.set_xlim(0.70, 1.045)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Coverage when that source is held out (90% nominal)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Transfer across sources is not unconditional",
                 loc="left", fontweight="bold", pad=6)
    ax.text(0.015, 0.995,
            "shaded band: within 3 pp of nominal — no source lands inside it",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            color=C_RULE, style="italic")


def main():
    mc, abstention, lodo = load()
    fig = plt.figure(figsize=(7.09, 4.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.05],
                          wspace=0.42, hspace=0.78,
                          left=0.078, right=0.965, top=0.90, bottom=0.105)
    ax_a = fig.add_subplot(gs[:, 0])   # hero，占满左列
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, mc, abstention)
    panel_b(ax_b, mc)
    panel_c(ax_c, lodo)

    for ax, lab, dx in ((ax_a, "a", -0.095), (ax_b, "b", -0.155), (ax_c, "c", -0.155)):
        ax.text(dx, 1.09, lab, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(PAPER_FIGS / f"fig3_monitor.{ext}", bbox_inches="tight", **kw)
    print(f"wrote {PAPER_FIGS/'fig3_monitor.pdf'}")
    print(f"wrote {PAPER_FIGS/'fig3_monitor.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
