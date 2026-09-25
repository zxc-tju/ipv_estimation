#!/usr/bin/env python3
"""Render six accepted RQ004 panels without statistical recomputation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style import ROOT, RUN, DATA, HUMAN, NEUTRAL, TEXT, W, apply_style, export, source_record

BASE = ROOT / "reports/studies/RQ004_ipv_state_space"
R1 = BASE / "RQ004_1_state_space_law_nature_20260618/02_process"
R2 = BASE / "RQ004_2_nature_conclusions_multiagent_20260618/02_process"
SOURCES = {
    "a": R2 / "agent_pair_asymmetry/tables/round3_case_level.csv",
    "b_source": R2 / "agent_pair_asymmetry/tables/dataset_dyad_metrics.csv",
    "b_pooled": R2 / "agent_dynamics/tables/round3/round3_c1_static_implementation_comparison.csv",
    "c": R2 / "agent_dynamics/tables/round3/round3_c1_role_lock_auc_source.csv",
    "d_pooled": R2 / "agent_context_safety/tables/table_turning_pairs.csv",
    "d_source": R2 / "agent_context_safety/tables/table_turn_dataset_replication.csv",
    "e": R1 / "agent_B_evidence/priority_gap_pet_bins.csv",
    "f": R1 / "agent_E_geometry/round3_geometry_prior_results.csv",
}
SOURCE_LABELS = {
    "av2_motion_forecasting": "AV2", "lyft_train_full": "Lyft",
    "waymo_train": "Waymo", "nuplan_train": "nuPlan",
}
SOURCE_ORDER = ["AV2", "Lyft", "Waymo", "nuPlan"]


def panel_title(ax, letter, title):
    ax.text(-.23, 1.075, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")
    ax.set_title(title, fontsize=8.8, loc="left", pad=15)


def point_ci(ax, value, low, high, y, color=HUMAN, marker="o", fill=True, size=4.5):
    if np.isfinite(low) and np.isfinite(high):
        ax.plot([low, high], [y, y], color=color, linewidth=1.1, solid_capstyle="butt", zorder=2)
    ax.plot(value, y, marker, ms=size, mec=color, mfc=color if fill else "white",
            mew=1.0, linestyle="none", zorder=3)


def main():
    apply_style()
    DATA.mkdir(parents=True, exist_ok=True)
    inputs = {key: pd.read_csv(path, low_memory=False) for key, path in SOURCES.items()}
    records = []

    def record(panel_id, key, fields, unit, filters, notes=""):
        records.append(source_record("Figure 2", panel_id, SOURCES[key], fields, unit, filters, notes))

    # This is a display aggregation only, not a new statistical estimate.
    cases = inputs["a"]
    v1 = cases["ipv_key_agent_1_mean"].to_numpy()
    v2 = cases["ipv_key_agent_2_mean"].to_numpy()
    assert len(cases) == 34850 and cases.source_row.nunique() == 34850
    assert np.isfinite(v1).all() and np.isfinite(v2).all()
    edges = np.linspace(-1.2, 1.2, 49)
    hist, xedges, yedges = np.histogram2d(np.r_[v1, v2], np.r_[v2, v1], bins=[edges, edges])
    assert int(hist.sum()) == len(cases) * 2
    bins = pd.DataFrame([
        {"x_low_rad": xedges[i], "x_high_rad": xedges[i + 1],
         "y_low_rad": yedges[j], "y_high_rad": yedges[j + 1],
         "symmetric_display_count": int(hist[i, j]),
         "percent_of_symmetric_points": hist[i, j] / hist.sum() * 100,
         "n_independent_cases": len(cases)}
        for i in range(hist.shape[0]) for j in range(hist.shape[1])
    ])
    bins.to_csv(DATA / "fig2a_joint_distribution_bins.csv", index=False)
    record("a", "a", "source_row;ipv_key_agent_1_mean;ipv_key_agent_2_mean", "independent case",
           "all 34850 valid rows; symmetrize partners for display only",
           "48x48 fixed bins in [-1.2,1.2] rad; no smooth or synthetic points; RQ004-PRES-PLANE")

    b_rows = []
    for _, r in inputs["b_source"].iterrows():
        low, high = json.loads(r.icc_ci)
        b_rows.append(dict(label=SOURCE_LABELS[r.dataset], kind="source_paired", n=int(r.n),
                           estimate=r.icc, ci_low=low, ci_high=high,
                           interval="95% case bootstrap CI; 1000 samples"))
    for _, r in inputs["b_pooled"].iterrows():
        independent = r.implementation == "DYAD"
        b_rows.append(dict(label="Independent" if independent else "All paired",
                           kind="independent_paired" if independent else "pooled_paired", n=int(r.n_cases),
                           estimate=r.observed_icc, ci_low=r.observed_ci_low, ci_high=r.observed_ci_high,
                           interval="unavailable" if independent else "95% scene-cluster bootstrap CI; 300 samples"))
        b_rows.append(dict(label="Independent null" if independent else "Re-paired",
                           kind="independent_null" if independent else "pooled_null", n=int(r.n_cases),
                           estimate=r.null_mean, ci_low=r.null_ci_low, ci_high=r.null_ci_high,
                           interval="central 95% of matched permutation null"))
    b = pd.DataFrame(b_rows)
    b.to_csv(DATA / "fig2b_complementarity.csv", index=False)
    record("b", "b_source", "dataset;n;icc;icc_ci", "independent case",
           "four frozen source rows", "per-source support totals 34850; intervals are 1000 case-bootstrap CIs")
    record("b", "b_pooled", ";".join(inputs["b_pooled"].columns), "matched-support case",
           "DYNAMICS pooled estimates and DYAD independent estimates", "independent observed interval is absent and not invented")

    auc = inputs["c"]
    assert len(auc) == 18 and set(auc.window) == {.05, .1, .2, .3, .4, .5}
    auc.to_csv(DATA / "fig2c_early_role_auc.csv", index=False)
    record("c", "c", ";".join(auc.columns), "case, clustered by scene",
           "plot models ipv and kinematics; all three models preserved in Source Data",
           "role target is final-quarter mean-IPV difference sign; original grouped OOF predictions and 300 scene bootstrap CIs")

    overall = inputs["d_pooled"].query("turn_group == 'all_turn_vs_straight'").copy()
    overall["source"] = "All pairs"
    d = inputs["d_source"].copy()
    d["source"] = d.dataset.map(SOURCE_LABELS)
    d = pd.concat([overall, d], ignore_index=True)
    d.to_csv(DATA / "fig2d_turn_straight.csv", index=False)
    record("d", "d_pooled", "turn_group;n_cases;mean_turn_minus_straight_ipv;ci_low;ci_high;p_bh",
           "paired case", "turn_group=all_turn_vs_straight", "frozen paired-mean Student-t interval; excludes nonaccepted reversal subgroup")
    record("d", "d_source", "dataset;n_cases;mean_turn_minus_straight_ipv;ci_low;ci_high",
           "paired case", "all four source rows", "frozen Student-t interval; nuPlan crosses zero")

    e = inputs["e"].query("dataset == 'ALL'").copy()
    assert e.n.tolist() == [9799, 22833, 4863]
    e.to_csv(DATA / "fig2e_priority_pet.csv", index=False)
    record("e", "e", "dataset;pet_bin;n;estimate;ci_low;ci_high", "paired priority/nonpriority case",
           "dataset=ALL, all three pre-existing PET strata", "frozen B=1000 seed=42 case-bootstrap CIs; no rebootstrap; offline PET only")

    f = inputs["f"].query("table == 'primary_contrast' and scope == 'dataset'").copy()
    f = f[f.contrast_id.isin(["MP_minus_nonMP", "SS_minus_nonSS"])].copy()
    assert len(f) == 8 and f.effect.notna().all()
    f[["dataset_short", "contrast_id", "n", "n_a", "n_b", "effect", "ci_low", "ci_high"]].to_csv(DATA / "fig2f_geometry.csv", index=False)
    record("f", "f", "table;scope;dataset_short;contrast_id;n;n_a;n_b;effect;ci_low;ci_high", "interaction case",
           "table=primary_contrast;scope=dataset;contrast_id in MP_minus_nonMP,SS_minus_nonSS",
           "eight source-specific original two-group bootstrap CIs; B=1200; no pooling")
    pd.DataFrame(records).to_csv(RUN / "process/fig2_sources.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 160 / 25.4))
    fig.subplots_adjust(left=.095, right=.98, bottom=.16, top=.91, wspace=.70, hspace=.75)
    aa, ab, ac, ad, ae, af = axes.ravel()

    panel_title(aa, "a", "Paired event readings")
    cmap = LinearSegmentedColormap.from_list("paired_density", ["#EEF0F6", "#99ACCA", "#4B5C9B", "#27334E"])
    fractions = hist.T / hist.sum() * 100
    img = aa.pcolormesh(xedges, yedges, np.ma.masked_where(hist.T == 0, fractions),
                        norm=LogNorm(vmin=100 / hist.sum(), vmax=fractions.max()), cmap=cmap, rasterized=True)
    aa.axhline(0, color=NEUTRAL, lw=.75, ls="--")
    aa.axvline(0, color=NEUTRAL, lw=.75, ls="--")
    aa.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel="Partner 1 mean IPV (rad)", ylabel="Partner 2 mean IPV (rad)")
    aa.set_xticks([-1, 0, 1]); aa.set_yticks([-1, 0, 1])
    aa.set_aspect("equal", adjustable="box", anchor="N")
    for x, y, text in [(.04, .94, "Mixed signs"), (.65, .94, "Both +"), (.04, .04, "Both −"), (.60, .04, "Mixed signs")]:
        aa.text(x, y, text, transform=aa.transAxes, fontsize=6.5, color=TEXT,
                ha="left", va="top" if y > .5 else "bottom")
    aa.text(.5, -.35, "n = 34,850 independent cases", ha="center", transform=aa.transAxes, fontsize=7.2)
    cbax = aa.inset_axes([.14, 1.03, .72, .04])
    cb = fig.colorbar(img, cax=cbax, orientation="horizontal", ticks=[.01, .1, 1])
    cb.ax.set_xticklabels(["0.01", "0.1", "1%"])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.ax.minorticks_off()
    cb.outline.set_visible(False)

    panel_title(ab, "b", "Partner complementarity")
    order = SOURCE_ORDER + ["All paired", "Re-paired", "Independent"]
    selected = b.set_index("label").loc[order]
    for y, (label, r) in enumerate(selected.iterrows()):
        color = NEUTRAL if r.kind == "pooled_null" else HUMAN
        marker = "D" if r.kind == "pooled_null" else "o"
        point_ci(ab, r.estimate, r.ci_low, r.ci_high, y, color, marker,
                 fill=r.kind != "independent_paired", size=4.4)
    ab.axvline(0, color=NEUTRAL, ls="--", lw=.7)
    ab.axhline(3.5, color="#DADDE1", lw=.6)
    ab.set_yticks(range(len(order)), order)
    ab.set(xlim=(-.49, .09), ylim=(6.6, -.7), xlabel="Exchangeable ICC")
    ab.set_xticks([-.4, -.2, 0]); ab.spines["left"].set_visible(False); ab.tick_params(axis="y", length=0)
    ab.text(.5, -.26, "Pooled n = 34,757\nIndependent n = 34,645", transform=ab.transAxes,
            ha="center", va="top", fontsize=7.2, linespacing=1.3)

    panel_title(ac, "c", "Roles are legible early")
    for model, label, color, marker in [("ipv", "Early IPV", HUMAN, "o"), ("kinematics", "Kinematics", NEUTRAL, "s")]:
        sub = auc[auc.model == model].sort_values("window")
        ac.fill_between(sub.window * 100, sub.ci_low, sub.ci_high, color=color, alpha=.16, linewidth=0)
        ac.plot(sub.window * 100, sub.auc, marker=marker, color=color, ms=3.5, label=label)
    ac.axhline(.5, color=NEUTRAL, ls="--", lw=.7)
    ac.set(xlim=(2, 53), ylim=(.48, .86), xlabel="Elapsed interaction (%)", ylabel="Final-IPV-role AUC")
    ac.set_xticks([5, 20, 35, 50]); ac.set_yticks([.5, .6, .7, .8])
    ac.legend(loc="center right", bbox_to_anchor=(1.06, .48), fontsize=7, handlelength=1.5)
    ac.text(.5, -.26, "n = 24,872 → 31,831 cases", transform=ac.transAxes, ha="center", va="top", fontsize=7.2)

    panel_title(ad, "d", "Turning and straight roles")
    order_d = ["All pairs"] + SOURCE_ORDER
    for y, name in enumerate(order_d):
        r = d[d.source == name].iloc[0]
        col = NEUTRAL if r.ci_low <= 0 <= r.ci_high else HUMAN
        point_ci(ad, r.mean_turn_minus_straight_ipv, r.ci_low, r.ci_high, y, col,
                 marker="D" if name == "All pairs" else "o")
    ad.axvline(0, color=NEUTRAL, ls="--", lw=.7)
    ad.axhline(.5, color="#DADDE1", lw=.6)
    ad.set_yticks(range(5), order_d)
    ad.set(xlim=(-.025, .20), ylim=(4.6, -.6), xlabel="Turn − straight IPV (rad)")
    ad.set_xticks([0, .1, .2]); ad.spines["left"].set_visible(False); ad.tick_params(axis="y", length=0)
    ad.text(.5, -.25, "3,565 paired cases", transform=ad.transAxes, ha="center", va="top", fontsize=7.2)

    panel_title(ae, "e", "Priority varies with PET")
    for x, (_, r) in enumerate(e.iterrows()):
        ae.errorbar(x, r.estimate, yerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]],
                    fmt="o", color=HUMAN, ms=4.7, capsize=2, elinewidth=1.1)
        ae.text(x, r.ci_high + .009, f"{r.estimate:+.3f}", ha="center", fontsize=7.5)
    ae.axhline(0, color=NEUTRAL, ls="--", lw=.7)
    ae.set(xlim=(-.45, 2.45), ylim=(-.061, .098), ylabel="Priority − nonpriority (rad)")
    ae.set_xticks([0, 1, 2], ["≤1 s", "1–2 s", ">2 s"])
    ae.set_yticks([-.04, 0, .04, .08])
    ae.set_xlabel("Offline PET stratum")
    ae.text(.5, -.25, "n = 9,799 / 22,833 / 4,863", transform=ae.transAxes, ha="center", va="top", fontsize=7.2)

    panel_title(af, "f", "Geometry across sources")
    for y, name in enumerate(SOURCE_ORDER):
        for cid, color, marker, offset in [("MP_minus_nonMP", HUMAN, "o", -.13), ("SS_minus_nonSS", W, "s", .13)]:
            r = f[(f.dataset_short == name) & (f.contrast_id == cid)].iloc[0]
            point_ci(af, r.effect, r.ci_low, r.ci_high, y+offset, color, marker, size=4.2)
    af.axvline(0, color=NEUTRAL, ls="--", lw=.7)
    af.set_yticks(range(4), SOURCE_ORDER)
    af.set(xlim=(-.005, .20), ylim=(3.5, -.65), xlabel="IPV difference (rad)")
    af.set_xticks([0, .1, .2]); af.spines["left"].set_visible(False); af.tick_params(axis="y", length=0)
    af.legend(handles=[Line2D([], [], color=HUMAN, marker="o", label="MP − non-MP", ms=4),
                       Line2D([], [], color=W, marker="s", label="S–S − non-S–S", ms=4)],
              loc="upper center", bbox_to_anchor=(.5, -.24), fontsize=7, handlelength=1.4, labelspacing=.5)

    export(fig, "fig2_context")
    print("Figure 2 exported: 6 panels; 8 source records; no statistic recomputation")


if __name__ == "__main__":
    main()
