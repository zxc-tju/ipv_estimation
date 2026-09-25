"""Render the supplied aggregate subjective results without rerunning statistics."""
from decimal import Decimal, ROUND_HALF_UP
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from figure_style import (
    ROOT, RUN, ASSETS, DATA, COLORS, ORDER, TEXT, HUMAN, NEUTRAL,
    apply_style, export, source_record, sha256,
)

SOURCE = ROOT / "data/manually_organized/RQ028_v1_主观评分_20260914/论文可用结果提取_20260925"
TABLES = SOURCE / "tables"
METRICS = ["q1_rev", "q4", "q5_cost_cp"]
TITLES = ["Perceived atypicality", "Comfort", "Reported interaction cost"]
SUBTITLES = ["Q1 reversed (8 − Q1)", "Q4", "Q5 · counterpart only"]
CONTRASTS = ["A-W", "C-W", "A-C"]
FILES = [
    "rating_descriptives", "participant_paired_contrasts", "paper_core_results",
    "design_counts", "common_scenario_source", "common_scenario_sensitivity",
    "alternative_code_sensitivity", "source_label_sensitivity", "scenario_class_coverage",
]


def three_decimals(value, signed=False):
    """Round labels conventionally without changing any exported source value."""
    rounded = Decimal(str(value)).quantize(Decimal(".001"), rounding=ROUND_HALF_UP)
    label = f"{rounded:+.3f}" if signed else f"{rounded:.3f}"
    return label.replace("-", "−")


def check_inputs(tables):
    """Check existing aggregates and internal numerical agreement only."""
    desc = tables["rating_descriptives"]
    contrasts = tables["participant_paired_contrasts"]
    scene = tables["common_scenario_sensitivity"]
    assert len(desc) == 30 and len(contrasts) == 30 and len(scene) == 30
    assert not desc.duplicated(["class", "seat", "metric"]).any()
    for table in [contrasts, scene]:
        assert not table.duplicated(["seat", "metric", "contrast"]).any()
        assert np.isfinite(table[["estimate", "ci_low", "ci_high"]]).all().all()
        assert (table.ci_low <= table.estimate).all()
        assert (table.ci_high >= table.estimate).all()
    assert (desc.n_subjects == 40).all() and (desc.n_pairs == 20).all()
    assert not ((desc.metric == "q5_cost_cp") & (desc.seat == "ego")).any()
    assert (desc.ci_low <= desc.subject_equal_mean).all()
    assert (desc.ci_high >= desc.subject_equal_mean).all()
    design = tables["design_counts"]
    assert design.n_ratings.sum() == 1192
    assert design.loc[design.seat == "cp", "n_trials"].sum() == 596
    assert design.loc[design.seat == "cp", "n_segments"].sum() == 90
    assert tables["scenario_class_coverage"].scenario_id.nunique() == 15
    for row in tables["paper_core_results"].itertuples(index=False):
        original = contrasts[
            (contrasts.metric == row.metric) & (contrasts.seat == row.seat)
            & (contrasts.contrast == row.contrast)
        ].iloc[0]
        for key in ["estimate", "ci_low", "ci_high", "p_holm_30"]:
            assert np.isclose(getattr(row, key), original[key], rtol=1e-12, atol=1e-14)
    scenario_rows = tables["common_scenario_source"]
    for row in scene.itertuples(index=False):
        values = scenario_rows[
            (scenario_rows.metric == row.metric) & (scenario_rows.seat == row.seat)
            & (scenario_rows.contrast == row.contrast)
        ].difference
        assert len(values) == row.n_common_scenarios
        assert (values > 0).sum() == row.n_positive
        assert (values < 0).sum() == row.n_negative
        assert (values == 0).sum() == row.n_zero
    return {
        "descriptive_rows": len(desc), "paired_comparison_rows": len(contrasts),
        "scenario_comparison_rows": len(scene), "paper_core_comparisons_matched": 8,
        "participants": 40, "participant_pairs": 20, "segments": 90,
        "scenarios": 15, "completed_trials": 596, "position_rating_records": 1192,
        "source_aggregate_checks": "pass", "statistics_rerun": False,
        "individual_or_pair_data_read": False,
        "ego_atypicality_A-W_Holm30_p": float(contrasts[
            (contrasts.seat == "ego") & (contrasts.metric == "q1_rev")
            & (contrasts.contrast == "A-W")].p_holm_30.iloc[0]),
    }


def main_figure(tables):
    desc, contrasts = tables["rating_descriptives"], tables["participant_paired_contrasts"]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 115 / 25.4))
    fig.subplots_adjust(left=.07, right=.985, top=.755, bottom=.395, wspace=.30)
    fig.add_artist(Rectangle((.025, .916), .96, .068, transform=fig.transFigure,
                             facecolor="#F1F3F4", edgecolor="none", zorder=0))
    fig.text(.505, .959, "40 participants / 20 pairs   ·   90 segments / 15 scenarios",
             ha="center", va="center", fontsize=8)
    fig.text(.505, .931, "596 completed trials   ·   1,192 position-rating records",
             ha="center", va="center", fontsize=8)
    handles = [Line2D([], [], color=TEXT, marker="s", linestyle="none", markerfacecolor="white",
                      markersize=4.5, label="Ego position"),
               Line2D([], [], color=TEXT, marker="o", linestyle="none", markerfacecolor=TEXT,
                      markersize=4.5, label="Counterpart position")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.505, .875),
               ncol=2, columnspacing=2, handletextpad=.5)
    for j, (ax, metric, title, subtitle) in enumerate(zip(axes, METRICS, TITLES, SUBTITLES)):
        seats = ["cp"] if metric == "q5_cost_cp" else ["ego", "cp"]
        for seat in seats:
            offset = 0 if len(seats) == 1 else (-.13 if seat == "ego" else .13)
            d = desc[(desc.metric == metric) & (desc.seat == seat)].set_index("class")
            for i, category in enumerate(ORDER):
                row = d.loc[category]
                mean = row.subject_equal_mean
                ax.errorbar(i + offset, mean,
                            yerr=[[mean - row.ci_low], [row.ci_high - mean]],
                            marker="s" if seat == "ego" else "o", markersize=4.5,
                            markerfacecolor="white" if seat == "ego" else COLORS[category],
                            markeredgecolor=COLORS[category], ecolor=COLORS[category],
                            linestyle="none", capsize=3, capthick=.9, elinewidth=1, zorder=3)
        ax.set_xlim(-.50, 2.50)
        ax.set_ylim(.9, 7.1)
        ax.set_yticks(range(1, 8))
        ax.set_xticks(range(3), ORDER)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.grid(axis="y", color="#E2E4E6", linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        pos = ax.get_position()
        fig.text(pos.x0 - .034, .815, "abc"[j], fontsize=11, weight="bold")
        fig.text(pos.x0, .815, title, fontsize=8.6, weight="bold")
        fig.text(pos.x0, .782, subtitle, fontsize=7.5, color="#555555")
        table_ax = fig.add_axes([pos.x0 - .005, .125, pos.width + .010, .192])
        table_ax.axis("off")
        table_ax.text(0, .96, "Counterpart paired differences", fontsize=7.5, weight="bold")
        table_ax.text(0, .73, "Contrast", fontsize=7.1, color="#555555")
        table_ax.text(.42, .73, "Δ", fontsize=7.1, color="#555555", ha="right")
        table_ax.text(.995, .73, "95% CI", fontsize=7.1, color="#555555", ha="right")
        d = contrasts[(contrasts.metric == metric) & (contrasts.seat == "cp")].set_index("contrast")
        for i, name in enumerate(CONTRASTS):
            row = d.loc[name]
            y = .51 - i * .23
            table_ax.text(0, y, name.replace("-", " − "), fontsize=7.3)
            estimate_label = (f"{row.estimate:+.4f}" if metric == "q1_rev" and name == "C-W"
                              else three_decimals(row.estimate, signed=True))
            table_ax.text(.42, y, estimate_label,
                          fontsize=7.3, ha="right")
            table_ax.text(.995, y, f"[{three_decimals(row.ci_low)}, {three_decimals(row.ci_high)}]",
                          fontsize=7.3, ha="right")
    axes[0].set_ylabel("Rating (1–7)")
    fig.text(.51, .070, "A: assertive side     C: accommodating side     W: within range",
             ha="center", fontsize=7.5)
    fig.text(.51, .035, "Points: equal-participant means. Bars: 95% pair-bootstrap intervals.",
             ha="center", fontsize=7.5)
    export(fig, "fig6_subjective")


def sensitivity_figure(tables):
    contrasts = tables["participant_paired_contrasts"]
    scene = tables["common_scenario_sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 95 / 25.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.28, top=.70, wspace=.39)
    handles = [Line2D([], [], color=HUMAN, marker="o", linestyle="none", markersize=4.5,
                      label="Equal participants; resample pairs"),
               Line2D([], [], color=TEXT, marker="s", markerfacecolor="white", linestyle="none",
                      markersize=4.5, label="Equal common scenarios; resample scenarios")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.515, .995),
               ncol=1, handletextpad=.5, labelspacing=.55, fontsize=7.7)
    for j, (ax, metric, title) in enumerate(zip(axes, METRICS, TITLES)):
        for df, off, color, mark, fill in [
            (contrasts, -.105, HUMAN, "o", HUMAN), (scene, .105, TEXT, "s", "white")
        ]:
            d = df[(df.metric == metric) & (df.seat == "cp")].set_index("contrast").loc[CONTRASTS]
            y = d.estimate.to_numpy()
            ax.errorbar(y, np.arange(3) + off,
                        xerr=np.vstack([y - d.ci_low, d.ci_high - y]), marker=mark,
                        markersize=4.5, markerfacecolor=fill, markeredgecolor=color,
                        ecolor=color, linestyle="none", capsize=3, capthick=.9, elinewidth=1)
        ax.axvline(0, color=NEUTRAL, linewidth=.75, linestyle=(0, (3, 3)))
        ax.set_yticks(range(3), [x.replace("-", " − ") for x in CONTRASTS])
        ax.set_ylim(2.45, -.45)
        ax.set_xlabel("Difference (rating points)")
        ax.grid(axis="x", color="#E2E4E6", linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim([(-.6, 1.85), (-1.4, .15), (-.15, 2.5)][j])
        ax.set_xticks([[-.5, 0, .5, 1, 1.5], [-1.0, -.5, 0], [0, 1, 2]][j])
        pos = ax.get_position()
        fig.text(pos.x0 - .042, .790, "abc"[j], fontsize=11, weight="bold")
        fig.text(pos.x0, .790, title, fontsize=8.6, weight="bold")
        fig.text(pos.x0, .749, "Counterpart position", fontsize=7.5, color="#555555")
    fig.text(.51, .16, "Common scenarios: 8 (A − W), 9 (C − W), 12 (A − C).",
             ha="center", fontsize=7.6)
    fig.text(.51, .102, "95% intervals use separate resampling units; fixed stimuli or fixed observed raters, respectively.",
             ha="center", fontsize=7.3)
    fig.text(.51, .045, "A: assertive side     C: accommodating side     W: within range",
             ha="center", fontsize=7.5)
    export(fig, "figS2_subjective_sensitivity")


def provenance(tables):
    records = []
    for j, metric in enumerate(METRICS):
        seats = "cp only" if metric == "q5_cost_cp" else "ego and cp"
        records.append(source_record(
            "6", "abc"[j], TABLES / "rating_descriptives.csv",
            "class;seat;metric;subject_equal_mean;ci_low;ci_high;n_ratings;n_subjects;n_pairs;ci_method",
            "rating points (1-7)", f"metric={metric}; seat={seats}; class order A,C,W",
            "Equal-participant mean; supplied pair-cluster percentile bootstrap intervals; no raw-weighted mean plotted."))
        records.append(source_record(
            "6", "abc"[j], TABLES / "participant_paired_contrasts.csv",
            "seat;metric;contrast;estimate;ci_low;ci_high;p_holm_30;estimand;ci_method",
            "within-subject difference in rating points", f"metric={metric}; seat=cp; A-W,C-W,A-C",
            "All three counterpart contrasts shown; original intervals retained; no inference from mean-interval overlap."))
        for table in ["participant_paired_contrasts", "common_scenario_sensitivity"]:
            records.append(source_record(
                "S2", "abc"[j], TABLES / f"{table}.csv",
                "seat;metric;contrast;estimate;ci_low;ci_high;ci_method", "difference in rating points",
                f"metric={metric}; seat=cp; A-W,C-W,A-C", "Supplied estimates and intervals; two estimands and resampling units stay separate."))
    records.append(source_record(
        "6", "design strip", TABLES / "design_counts.csv",
        "class;arm;seat;n_ratings;n_trials;n_subjects;n_pairs;n_segments;n_scenarios",
        "counts", "both seats for rating records; cp rows for trials and segments",
        "40 participants and 20 pairs are shared counts, not summed across rows; trials=596; segments=90; ratings=1192."))
    records.append(source_record(
        "6", "design strip", TABLES / "scenario_class_coverage.csv",
        "scenario_id", "scenarios", "unique scenario_id", "15 observed scenarios."))
    records.append(source_record(
        "S2", "direction-count check", TABLES / "common_scenario_source.csv",
        "scenario_id;seat;metric;contrast;difference", "scenario-level difference",
        "check against all 30 common_scenario_sensitivity rows", "Aggregate scenario counts checked; no new estimation."))
    for name in FILES:
        records.append(source_record(
            "6/S2", "supplementary source table", TABLES / f"{name}.csv", "all existing fields",
            "source-defined", "complete aggregate table; no filtering", f"Byte-identical copy: source_data/fig6_{name}.csv"))
    records.append(source_record(
        "6/S2", "prior plotting entry point", SOURCE / "process/make_figures.py", "input and weighting logic",
        "not applicable", "read only", "Original figures migrated without reading the individual-level source used for faint dots."))
    pd.DataFrame(records).to_csv(RUN / "process/fig6_sources.csv", index=False)


def main():
    apply_style()
    DATA.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_csv(TABLES / f"{name}.csv") for name in FILES}
    checks = check_inputs(tables)
    # Copy aggregate tables without rewriting their values or metadata.
    for name in FILES:
        target = DATA / f"fig6_{name}.csv"
        shutil.copyfile(TABLES / f"{name}.csv", target)
        assert sha256(target) == sha256(TABLES / f"{name}.csv")
    tables["rating_descriptives"][tables["rating_descriptives"].metric.isin(METRICS)].to_csv(
        DATA / "fig6_plotted_means.csv", index=False)
    for name, label in [("participant_paired_contrasts", "equal_participant_pair_bootstrap"),
                        ("common_scenario_sensitivity", "equal_scenario_scenario_bootstrap")]:
        d = tables[name]
        d[(d.seat == "cp") & d.metric.isin(METRICS)].assign(estimation_frame=label).to_csv(
            DATA / f"figS2_{label}.csv", index=False)
    provenance(tables)
    main_figure(tables)
    sensitivity_figure(tables)
    qa_path = RUN / "process/fig6_qa.json"
    prior = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    prior.update({"numerical_checks": checks,
                  "output_sha256": {p.name: sha256(p) for p in sorted(ASSETS.glob("fig6_subjective.*"))}
                  | {p.name: sha256(p) for p in sorted(ASSETS.glob("figS2_subjective_sensitivity.*"))},
                  "source_copies_sha256_match": True})
    qa_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
