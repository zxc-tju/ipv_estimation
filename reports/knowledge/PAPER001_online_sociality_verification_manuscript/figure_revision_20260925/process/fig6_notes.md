# Figure 6 and supplementary sensitivity figure

## Figure contract

The supplied fixed-stimulus results show higher counterpart atypicality ratings for both outside-range categories; comfort and reported interaction cost distinguish their directions. The figure visualizes these existing descriptive and paired results without adding causal, safety, population-loss, response-time, or equivalence claims.

The main figure is a quantitative triptych: (a) Q1 reversed as 8−Q1, ego and counterpart; (b) Q4 comfort, both positions; (c) Q5 reported interaction cost, counterpart only. The study-design strip reports the supplied cohort. Means are equal-participant means and intervals are the supplied pair-cluster bootstrap intervals. A compact counterpart contrast display includes all three contrasts per metric, including the small C−W comfort contrast and the uncertain A−C atypicality contrast. Supplementary sensitivity panels compare the existing paired-participant and equal-common-scenario estimates, preserving their different resampling units.

Python/matplotlib only; 183 mm wide, editable SVG/PDF and 300 dpi PNG; shared Arial style and A/C/W category colours. Open squares encode ego and filled circles counterpart. The full 1–7 rating axis is retained. No individual or pair records are read or exported. All source tables and statistics remain unchanged; no estimation, testing, or bootstrap is rerun. The full 30-comparison table retains Holm-30 p values. The missing full Q5 wording remains a protocol gap.

## Source and numerical verification

`fig6_sources.csv` records every panel's exact input, fields, filters, units and SHA-256. `make_fig6.py` reads only nine supplied aggregate tables. The original `make_figures.py` was inspected for input and weighting semantics but not executed. The new renderer does not read `ratings_analysis_rows.csv`, `participant_contrast_source.csv`, `pair_contrast_source.csv`, or raw rating records.

The nine unfiltered aggregate tables are copied byte for byte to `source_data/fig6_*.csv`. The complete 30-row descriptive table retains raw-row means separately from participant-equal means; only the latter are plotted. The complete 30-row paired table retains all tested outcomes and all Holm-30 p values. The source and alternative-code sensitivity tables retain 60 and 30 rows respectively. The scenario-level aggregate table retains 290 rows and all 30 common-scenario summaries. Plot-only CSVs add convenient filtered views without changing estimates.

Checks passed: 15 plotted means; 9 main-panel counterpart contrasts; 18 supplementary plotted estimates across the two estimands; all 8 `paper_core_results.csv` comparisons match the paired table's estimate, CI and adjusted p value. Every common-scenario direction count matches the supplied scenario-aggregate rows. All 9 source copies match input SHA-256. There are no non-finite plotted values, duplicate aggregate keys, inverted intervals, or out-of-axis intervals.

The design strip uses shared counts of 40 participants and 20 pairs, 90 segments summed over counterpart-position class/arm rows, 15 unique scenarios in the scenario-coverage table, 596 completed trials from counterpart-position rows, and 1,192 rating records over both positions. Per position, A/C/W have 158/239/199 rating records. These are repeated observations, not 1,192 independent people. Counts of 600 attempts and 4 aborts belong to Methods as specified by the plan and are not reconstructed from completed-trial tables.

### Source-number cross-check

Values below reproduce the existing counterpart-position paired table (CI bounds rounded only for presentation). Figure labels use conventional half-up rounding to three decimals except counterpart atypicality C−W: its exact source estimate is +0.6825 and is printed with four decimals to avoid the +0.682 versus +0.683 tie-rounding ambiguity. Source CSV precision is unchanged. The manuscript and its existing core-results table should likewise retain +0.6825 for this one estimate.

| Panel | Contrast | Estimate | Existing 95% CI | Existing Holm-30 p |
|---|---|---:|---|---:|
| a · atypicality | A−W | +0.639167 | [0.317906, 0.992094] | 0.02453828 |
| a · atypicality | C−W | +0.682500 | [0.548333, 0.816667] | 1.992454e−7 |
| a · atypicality | A−C | −0.043333 | [−0.302083, 0.227510] | 1.000000 |
| b · comfort | A−W | −1.090833 | [−1.226250, −0.961250] | 6.114536e−11 |
| b · comfort | C−W | −0.221250 | [−0.351250, −0.100000] | 0.03419828 |
| b · comfort | A−C | −0.869583 | [−0.973750, −0.765823] | 5.179370e−11 |
| c · reported cost | A−W | +2.070417 | [1.885000, 2.262510] | 3.945733e−13 |
| c · reported cost | C−W | +0.907917 | [0.770833, 1.050833] | 3.494437e−9 |
| c · reported cost | A−C | +1.162500 | [1.018750, 1.300000] | 6.932545e−11 |

Ego-position atypicality A−W has existing Holm-30 p = 0.07038382684. It is not labelled significant. A−C counterpart atypicality is not described as equality or equivalence. Mean-CI overlap is not used to infer any paired contrast. The full Q5 item wording is still unverified; the figure preserves only the existing item name, “reported interaction cost”.

## Caption suggestions

**Figure 6 | Outside-range categories differ in perceived atypicality, comfort and reported interaction cost.** a, Perceived atypicality (8−Q1); b, comfort (Q4); c, reported interaction cost (Q5, counterpart position only). A, assertive-side segments; C, accommodating-side segments; W, within-range segments. Open squares denote ego-position ratings and filled circles denote counterpart-position ratings. Points show equal-participant means; bars show the existing 95% percentile intervals from 10,000 resamples of 20 participant pairs, conditional on the fixed stimulus set. The 40 participants contributed 1,192 position-rating records from 596 completed trials involving 90 segments and 15 scenarios; each position contributed 158 A, 239 C and 199 W records. Below each panel, all three counterpart within-participant differences and their corresponding pair-bootstrap intervals are shown. Paired comparisons use the existing pair-level tests with Holm adjustment across 30 comparisons; mean-interval overlap does not determine significance. Ego atypicality A−W was not significant after adjustment (p=0.0704), and the counterpart A−C atypicality comparison does not establish equivalence. Source Data include all 30 comparisons.

**Supplementary Figure S2 | Common-scenario sensitivity of the counterpart-position contrasts.** a, Perceived atypicality (8−Q1); b, comfort (Q4); c, reported interaction cost (Q5). Filled circles reproduce equal-participant paired contrasts with 95% percentile intervals from 10,000 participant-pair resamples, conditional on the fixed stimulus set. Open squares reproduce equal-common-scenario contrasts with 95% percentile intervals from 10,000 scenario resamples, conditional on the observed raters. A−W, C−W and A−C share 8, 9 and 12 scenarios, respectively. The two analyses have different estimands and resampling units; the scenario analysis is a descriptive sensitivity check within the selected scenario population. Category definitions follow Fig. 6. Source Data are provided.

## Rendering and QA

Run from any directory with the selected Python runtime:

```sh
python3 reports/knowledge/PAPER001_online_sociality_verification_manuscript/figure_revision_20260925/process/make_fig6.py
```

Outputs are `assets/fig6_subjective.{pdf,svg,png}` and `assets/figS2_subjective_sensitivity.{pdf,svg,png}`. Main figure size is 183×115 mm; supplementary size is 183×95 mm. The 300 dpi PNGs were inspected directly against the two supplied source PNGs. No panel, label, interval, or text is clipped or overlapping. Mean plots retain the complete 1–7 scale; counterpart-only Q5 has no invented ego record. Lowercase a/b/c labels, A/C/W order, the shared three-category palette, and separate position markers are consistent. The contrast numbers are included without stars or new tests. SVG text remains text and PDFs retain extractable, embedded-font labels. The final numerical checks, output hashes and structured visual verdicts are in `fig6_qa.json`.

The figure work leaves questionnaire, consent, ethics and stimulus-version evidence gaps unchanged. Rendering these supplied results does not independently establish empirical provenance or amend an accepted decision.
