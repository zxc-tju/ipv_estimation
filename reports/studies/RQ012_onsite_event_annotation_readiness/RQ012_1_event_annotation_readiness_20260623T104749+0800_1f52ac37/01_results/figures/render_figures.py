#!/usr/bin/env python3
"""RQ012A readiness figures — Nature-figure-skill render (Python/matplotlib).

Outcome-blind readiness figures only (no IPV / score / rank / team).
Renders FIG1-FIG5 to PNG (view) + SVG (editable text) + PDF (editable text)
from the prepared source CSVs in source_data/. Run with the project .venv_fig.
"""
import csv
import os
import textwrap
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- MANDATORY editable-text rules (nature-figure api.md) -------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"   # text stays as <text> in SVG
plt.rcParams["pdf.fonttype"] = 42       # editable TrueType in PDF
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

HERE = Path(__file__).resolve().parent
SRC = HERE / "source_data"

# Colorblind-safe, semantically-encoded palette (color + text + position redundancy)
C_KEEP = "#0F4D92"      # keep_automatic / pass (deep blue)
C_DEMOTE = "#E28E2C"    # demote_human_only / partial / pending (amber)
C_REMOVE = "#B64342"    # remove / blocked (brick red)
C_NEUTRAL = "#767676"
C_PASS = "#2E7D46"      # gate pass (green = go cue)
C_PARTIAL = "#E28E2C"
C_PENDING = "#9A7BD0"   # pending humans (violet, distinct from blocked)
C_BLOCK = "#B64342"
C_INDEP = "#3775BA"     # independent endpoint
C_PROX = "#E28E2C"      # construct-proximal (not primary)
C_PLANNER = "#A8A8A8"   # planner/system removed
GRID = "#D8D8D8"


def read_csv(name):
    with open(SRC / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, stem):
    out = HERE / stem
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", stem, "(png/svg/pdf)")


# ===========================================================================
# FIG1 — Signal availability and Gate 012-0 action
# ===========================================================================
def fig1():
    rows = read_csv("fig1_signal_availability_source.csv")
    classes = ["direct", "derivable", "partially_observable", "unavailable"]
    actions = ["keep_automatic", "demote_human_only", "remove"]
    acolor = {"keep_automatic": C_KEEP, "demote_human_only": C_DEMOTE, "remove": C_REMOVE}
    alabel = {"keep_automatic": "Keep automatic", "demote_human_only": "Demote human-only", "remove": "Remove"}
    counts = {c: {a: 0 for a in actions} for c in classes}
    for r in rows:
        counts[r["signal_class"]][r["gate_012_0_action"]] += 1

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = range(len(classes))
    left = [0] * len(classes)
    for a in actions:
        vals = [counts[c][a] for c in classes]
        ax.barh(list(y), vals, left=left, color=acolor[a], edgecolor="white",
                linewidth=0.8, label=alabel[a], height=0.62)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.replace("_", " ") for c in classes])
    ax.set_xlabel("Number of event candidates")
    ax.set_xlim(0, 8)
    ax.invert_yaxis()
    ax.set_title("Gate 012-0 signal availability  (n = 20 candidates)", fontsize=8, loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.5, ncol=1)
    ax.text(0, -0.30, "Kept automatic = 9   ·   Demoted human-only = 5   ·   Removed = 6.  "
            "Retention does not authorize event–IPV analysis.",
            transform=ax.transAxes, fontsize=6, color=C_NEUTRAL)
    save(fig, "fig1_signal_availability")


# ===========================================================================
# FIG2 — Event ontology by class x extraction mode x endpoint tier
# ===========================================================================
def fig2():
    rows = read_csv("fig2_event_ontology_source.csv")
    classes = ["physical_safety", "interaction_quality", "human_motif", "planner_system"]
    modes = ["automatic", "human_only", "removed"]
    mcolor = {"automatic": C_KEEP, "human_only": C_DEMOTE, "removed": C_PLANNER}
    mlabel = {"automatic": "Automatic", "human_only": "Human-only", "removed": "Removed"}
    counts = {c: {m: 0 for m in modes} for c in classes}
    prox = {c: 0 for c in classes}      # construct-proximal (not primary) count
    for r in rows:
        counts[r["event_class"]][r["extraction_mode"]] += 1
        if r["endpoint_eligibility"] == "construct_proximal_descriptor":
            prox[r["event_class"]] += 1

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    y = range(len(classes))
    left = [0] * len(classes)
    for m in modes:
        vals = [counts[c][m] for c in classes]
        ax.barh(list(y), vals, left=left, color=mcolor[m], edgecolor="white",
                linewidth=0.8, label=mlabel[m], height=0.6)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.replace("_", " ") for c in classes])
    ax.set_xlabel("Number of ontology events")
    ax.set_xlim(0, 8)
    ax.invert_yaxis()
    ax.set_title("Event ontology by class and extraction mode  (n = 20 events)",
                 fontsize=8, loc="left", fontweight="bold")
    # annotate construct-proximal (B01: not primary endpoints)
    for i, c in enumerate(classes):
        if prox[c]:
            ax.text(left[i] + 0.12, i, f"[{prox[c]}] not-primary", va="center",
                    ha="left", fontsize=6, color=C_PROX, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.text(0, -0.30, "B01 endpoint tiers: construct-proximal descriptors [in brackets] are secondary only, "
            "forbidden as primary event–IPV endpoints; planner/system removed (no signal path).",
            transform=ax.transAxes, fontsize=6, color=C_NEUTRAL)
    save(fig, "fig2_event_ontology")


# ===========================================================================
# FIG3 — Extractor pilot health (4 panels), hero = raw vs primary de-overlap
# ===========================================================================
def fig3():
    rows = read_csv("fig3_extractor_pilot_health_source.csv")
    ev = [r["event_id"] for r in rows]
    f = lambda k: [float(r[k]) for r in rows]
    i = lambda k: [int(float(r[k])) for r in rows]
    comp = f("computable_fraction")
    raw = i("raw_event_count")
    prim = i("primary_event_count")
    supp = i("suppressed_by_precedence")
    tlo, tce, thi = i("threshold_low_count"), i("threshold_central_count"), i("threshold_high_count")
    so = i("sampling_original_count")
    sd = i("sampling_decimate2_count")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32)
    x = range(len(ev))

    # (A) computable fraction
    axa = fig.add_subplot(gs[0, 0])
    axa.bar(list(x), comp, color=C_KEEP, edgecolor="white", linewidth=0.6)
    axa.set_xticks(list(x)); axa.set_xticklabels(ev, rotation=45, ha="right", fontsize=6)
    axa.set_ylabel("Computable fraction (0–1)")
    axa.set_ylim(0, 1.05)
    axa.axhline(1.0, color=GRID, lw=0.6, zorder=0)
    axa.set_title("a  Computable fraction by event", fontsize=7.5, loc="left", fontweight="bold")
    axa.text(0.0, -0.42, "E01 = 0: deferred (counterpart relation unavailable).",
             transform=axa.transAxes, fontsize=5.6, color=C_NEUTRAL)

    # (B) HERO raw vs primary
    axb = fig.add_subplot(gs[0, 1])
    w = 0.4
    xb = list(x)
    axb.bar([v - w/2 for v in xb], raw, width=w, color=C_NEUTRAL, edgecolor="white",
            linewidth=0.5, label="Raw intervals")
    axb.bar([v + w/2 for v in xb], prim, width=w, color=C_KEEP, edgecolor="white",
            linewidth=0.5, label="Primary (de-overlapped)")
    axb.set_xticks(xb); axb.set_xticklabels(ev, rotation=45, ha="right", fontsize=6)
    axb.set_ylabel("Event intervals (central band)")
    axb.set_title("b  Raw vs de-overlapped primary counts", fontsize=7.5, loc="left", fontweight="bold")
    axb.legend(fontsize=6)
    # annotate the E09 suppression
    for j, e in enumerate(ev):
        if supp[j] > 0:
            axb.annotate(f"−{supp[j]}\n(E15 precedence)", xy=(j, prim[j]),
                         xytext=(j, max(raw) * 0.62), fontsize=5.6, ha="center",
                         color=C_REMOVE, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=C_REMOVE, lw=0.7))

    # (C) threshold sensitivity low/central/high
    axc = fig.add_subplot(gs[1, 0])
    w3 = 0.27
    axc.bar([v - w3 for v in xb], tlo, width=w3, color="#AADCA9", edgecolor="white", linewidth=0.4, label="Low")
    axc.bar(xb, tce, width=w3, color=C_KEEP, edgecolor="white", linewidth=0.4, label="Central")
    axc.bar([v + w3 for v in xb], thi, width=w3, color=C_DEMOTE, edgecolor="white", linewidth=0.4, label="High")
    axc.set_xticks(xb); axc.set_xticklabels(ev, rotation=45, ha="right", fontsize=6)
    axc.set_ylabel("Event intervals")
    axc.set_yscale("symlog")
    axc.set_title("c  Threshold-band sensitivity", fontsize=7.5, loc="left", fontweight="bold")
    axc.legend(fontsize=6, ncol=3)

    # (D) sampling sensitivity original vs decimate2
    axd = fig.add_subplot(gs[1, 1])
    axd.bar([v - w/2 for v in xb], so, width=w, color=C_KEEP, edgecolor="white", linewidth=0.5, label="Original cadence")
    axd.bar([v + w/2 for v in xb], sd, width=w, color="#9A4D8E", edgecolor="white", linewidth=0.5, label="Decimate ×2")
    axd.set_xticks(xb); axd.set_xticklabels(ev, rotation=45, ha="right", fontsize=6)
    axd.set_ylabel("Event intervals")
    axd.set_yscale("symlog")
    axd.set_title("d  Sampling-rate sensitivity", fontsize=7.5, loc="left", fontweight="bold")
    axd.legend(fontsize=6)

    fig.suptitle("Extractor pilot health — outcome-blind audit (9 automatic events, 5 pilot sessions; no event–IPV association)",
                 fontsize=8.5, fontweight="bold", x=0.01, ha="left", y=1.005)
    save(fig, "fig3_extractor_pilot_health")


# ===========================================================================
# FIG4 — Blind workflow / readiness pipeline (schematic)
# ===========================================================================
def fig4():
    rows = [r for r in read_csv("fig4_blind_workflow_readiness_pipeline_source.csv") if r["row_type"] == "node"]
    rows.sort(key=lambda r: int(r["stage_order"]))
    statecolor = {
        "text_issuance_surfaces_cleared": C_PARTIAL,
        "blocked_for_human_labels": C_BLOCK,
        "protocol_ready_pending_label_receipt": C_PENDING,
        "frozen_not_computed": C_PENDING,
        "not_started": C_NEUTRAL,
        "blocked": C_BLOCK,
    }
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    n = len(rows)
    bw, bh, gap = 1.7, 1.0, 0.55
    for k, r in enumerate(rows):
        xx = k * (bw + gap)
        col = statecolor.get(r["state"], C_NEUTRAL)
        box = FancyBboxPatch((xx, 0), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.0, edgecolor="white", facecolor=col)
        ax.add_patch(box)
        ax.text(xx + bw/2, bh*0.70, f"S{k+1}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
        wrapped = textwrap.fill(r["stage_label"], width=14)
        ax.text(xx + bw/2, bh*0.30, wrapped, ha="center", va="center",
                color="white", fontsize=5.4, linespacing=1.05)
        ax.text(xx + bw/2, -0.28, r["state"].replace("_", " "), ha="center", va="top",
                fontsize=5.2, color=C_NEUTRAL)
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((xx + bw, bh/2), (xx + bw + gap, bh/2),
                         arrowstyle="-|>", mutation_scale=9, color=C_NEUTRAL, lw=1.1))
    total = n * (bw + gap) - gap
    # BLOCKED_FOR_HUMAN_LABELS band over S2..S5
    ax.annotate("", xy=(bw + gap + 0.0, bh + 0.45), xytext=((n-1)*(bw+gap) + bw, bh + 0.45),
                arrowprops=dict(arrowstyle="-", color=C_BLOCK, lw=1.2))
    ax.text((bw + gap + (n-1)*(bw+gap) + bw)/2, bh + 0.62, "BLOCKED_FOR_HUMAN_LABELS",
            ha="center", va="bottom", fontsize=6.5, color=C_BLOCK, fontweight="bold")
    ax.set_xlim(-0.2, total + 0.2)
    ax.set_ylim(-0.7, bh + 1.0)
    ax.axis("off")
    ax.set_title("Blind annotation workflow — readiness pipeline (Gate 012B authorization blocked)",
                 fontsize=8, loc="left", fontweight="bold")
    save(fig, "fig4_blind_workflow_readiness_pipeline")


# ===========================================================================
# FIG5 — Readiness gate status ladder
# ===========================================================================
def fig5():
    rows = read_csv("fig5_readiness_gates_source.csv")
    statecolor = {
        "pass": C_PASS,
        "text_issuance_surfaces_cleared": C_PARTIAL,
        "ready_pending_humans": C_PENDING,
        "blocked": C_BLOCK,
    }
    statelabel = {
        "pass": "PASS",
        "text_issuance_surfaces_cleared": "TEXT-CLEARED",
        "ready_pending_humans": "READY – PENDING HUMANS",
        "blocked": "BLOCKED",
    }
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = list(range(len(rows)))[::-1]
    for yy, r in zip(y, rows):
        col = statecolor.get(r["status"], C_NEUTRAL)
        ax.barh(yy, 1.0, color=col, edgecolor="white", height=0.62)
        ax.text(0.02, yy, f"{r['gate_id']}  {r['gate_label']}", va="center", ha="left",
                color="white", fontsize=7, fontweight="bold")
        ax.text(1.04, yy, statelabel.get(r["status"], r["status"]), va="center", ha="left",
                color=col, fontsize=6.5, fontweight="bold")
    ax.set_xlim(0, 1.9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("RQ012A readiness gates  (n = 5)", fontsize=8.5, loc="left", fontweight="bold")
    ax.text(0, -0.22,
            "012B remains BLOCKED until: RQ011 frozen universe · final neutral media/card issuance · "
            "auditor sign-off ·\ntwo accepted human labels · κ + Gwet AC1 agreement · required RQ007/009/011 freezes · "
            "explicit authorization.",
            transform=ax.transAxes, fontsize=5.8, color=C_NEUTRAL)
    save(fig, "fig5_readiness_gates")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("ALL FIGURES RENDERED to", HERE)
