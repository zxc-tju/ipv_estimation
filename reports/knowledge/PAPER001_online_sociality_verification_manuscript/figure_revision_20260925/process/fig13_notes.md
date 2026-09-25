# Figure 1 / 3 source verification and figure contract

This task removes duplicate information from the monitor definition and calibration figures while preserving frozen measurements and all migration destinations. Python and the shared `figure_style.py` govern all drawings and exports. No model fitting or empirical reanalysis is performed.

## Contract before drawing

- Figure 1: A trajectory-derived reading is compared with a situation-conditioned human range, with explicit abstention. Schematic-led composite: a two-input mechanism, b the existing case trajectories, c the same case's 90% range and unsmoothed readings. Export 183 mm wide, editable PDF/SVG and 300 dpi PNG.
- Figure 3: Conditioning sharpens the reference at near-nominal marginal coverage, while gates delimit use. Quantitative grid: width, empirical coverage deviation, nested counts, and all-readable R². Source denominators remain distinct.
- Supplementary estimator detail: move the angle interpretation and the two frozen candidate-weight examples without deleting the existing measurement supplement.
- Extended Data reference distribution: move the readable human histogram and three frozen example ranges.
- Extended Data source transfer: keep episode-summary prediction R² separate from reference-range coverage; show support abstention with coverage for all four sources.

## Verified inputs

1. `S1_scoring/all_candidates_scored.parquet`: explicitly select case `ipv_004992`, perspective `key_agent_1`, exactly 221 consecutive frames. It is the case named in the previous Figure 1 trajectory/timeline generator, not a new outcome- or rating-based selection. The S1 README states that these are existing human-human test-fold cases scored by the frozen RQ021 model. The main timeline retains the old 11.0–22.7 s display; the local trajectory view retains the old start at 13.2 s. Coordinates are translated only to the existing closest-path origin; the 5 m conflict-zone circle is a schematic locator, not a measured road boundary. No map geometry exists in this source and no roadway is invented.
2. RQ021_1 `key_numbers.json`: `human_only_envelope.metrics`, `circularity_diagnostics.marginal_envelopes.ipv_log.metrics`, and `D2_contemporaneous_test_r2`. Accepted by RQ021 decision. Width and coverage use 461,937 supported/readable moments; R² uses all 486,660 readable moments.
3. S1 `fig1_three_layer_data.json`: exact candidate grid, two exemplar weight vectors, human histogram, example conditional range summaries, and pure-human test-fold funnel. The separate 4,497,368-row readability universe is not used in Figure 3.
4. RQ004_1 `F7_lodo_summary_source_data.csv`: `outcome_spec == case_mean_ipv`, full-state-space R², source test_n; migrated original Figure 2c information.
5. RQ021_2 `key_numbers_e2.json`: `lodo.<source>.metrics.90` contains complete coverage and support-abstention counts; `insample_by_source` is not substituted. The RQ021 accepted decision explicitly preserves the source-transfer boundary.

## Statistical and integrity boundaries

- Width unit is percentage of the seven-candidate span, 3π/4 rad, not the parameterisation domain. Reductions are computed from unrounded means.
- Main Figure 1 values failing either gate are masked as NaN; neither line nor shaded range bridges them. Status strip separately shows abstention. No smoothing or interpolation is used.
- Figure 3 support abstention uses 24,723 / 486,660, and not all 663,282 candidates.
- R² uses `1 − SSE/SST`; the residual component is not labelled personality, free will, or preference.
- Frozen point estimates without uncertainty are shown without fabricated intervals.

## Corrected scope of the moved reference examples

The three old `deployed_ranges` examples are NOT a summary of the human test-fold histogram. Their counts sum with the other cells to **67,861 OnSite candidate moments**. For each cell, the old lower number equals the median of `lo_90`, the upper number equals the median of `hi_90`, and `w` is their difference; `w` is not the median of `width_90`. Exact verification against the frozen deployment score table passed:

| Context | Candidate n | Median lower (rad) | Median upper (rad) | Difference of the medians (rad) |
|---|---:|---:|---:|---:|
| Crossing / priority | 2,336 | -0.49388012300000916 | 0.9005043507622315 | 1.3943844737622406 |
| Merging / priority | 10,291 | -0.7724621296928955 | 1.0143455267952515 | 1.786807656488147 |
| Merging / equal | 291 | -1.0097757578896118 | 1.0663764477776123 | 2.076152205667224 |

There is no readability/support filter for these summaries. The previously quoted width range 1.05–2.37 rad is also the all-candidate deployment distribution: `width_90` min 1.047387033900943, max 2.372843861779895, median 1.7131542565438416 over 67,861 rows. Among 21,936 support-passing rows the min/max instead are 1.1106675567242767 / 2.372558355531421. These are model-output descriptions, not evidence that every plotted candidate receives a verdict.

## Caption suggestions (figure numbering to be assigned during manuscript integration)

### Figure 1 — fig1_monitoring

**Trajectory-derived readings are compared with a situation-conditioned human range, with explicit abstention.** **a**, A causal trajectory window supplies the IPV reading, whereas the observable situation supplies the human reference range. Readability and human-support gates precede comparison; a failed gate returns no verdict. A, assertive-side; C, accommodating-side; W, within range. Objective interaction outcomes and subjective ratings are downstream evaluations. **b**, Local trajectories from the same previously displayed interaction (`ipv_004992`, first-agent perspective); circles and squares distinguish the monitored vehicle and counterpart, and three times match panel c. Coordinates are translated to the closest-path midpoint. The 5 m circle locates the conflict region schematically and is not a measured road boundary. **c**, Unsmoothed per-frame readings and the frozen dynamic 90% reference from that interaction. Teal dots are within range and light-red triangles lie above its accommodating boundary. This case contains no assertive-side departure. The top strip encodes states; grey and gaps indicate failed gates. Neither readings nor reference lines connect across those gaps. Source data are provided as a Source Data file.

Integration detail: the source has 221 frames, with 59 W, 2 C, and 160 abstentions under the intersection of both gates. Main c retains the previous 11.0–22.7 s display; b retains the previous local view from 13.2 s. Numbers above refer to all 221 source frames, not the display-window count.

### Figure 3 — fig3_monitor

**Conditioning narrows the human reference while retaining near-nominal empirical marginal coverage.** **a**, Global and situation-conditioned mean interval widths at 80%, 90% and 95% nominal coverage, expressed as percentages of the seven-candidate span, 3π/4 rad. Reductions of 21%, 20% and 8% are computed from unrounded means; the 90% conditional mean is 1.87 rad. **b**, Empirical coverage minus nominal coverage in percentage points. Panels a and b use the same 461,937 readable and supported naturalistic human test moments. **c**, Nested gates on 663,282 candidate moments: 486,660 readable, 461,937 also supported, then 417,036 inside and 44,901 outside the 90% range. Support abstention is 24,723/486,660 = 5.08% of readable moments. **d**, The conditional median model explains 20.9% of reading variance (R² = 1 − SSE/SST) on all 486,660 readable moments, including those excluded by the support gate. The remainder is residual variation. All quantities are frozen point estimates; no uncertainty intervals are shown. Source data are provided as a Source Data file.

### Supplementary estimator detail — figS4_estimator_details

**The fixed estimator maps candidate weights to a reading or an abstention.** **a**, The seven evaluated candidates span −3π/8 to 3π/8; the angle weights own-progress and interaction costs by cos(θ) and sin(θ), respectively. **b**, The first frozen example (frame 10) has concentrated candidate weights and a weighted-mean reading of −0.843 rad. **c**, The second frozen example (frame 15) has nearly uniform weights and returns an abstention. The dotted line is equal weight, 1/7. The two examples are retained from the earlier estimator illustration and are not claimed to be frames from the monitored case in Figure 1. The angle panel is a conceptual explanation; b and c reproduce the supplied weight vectors without fitting or uncertainty estimates. Source data are provided as a Source Data file.

**Keep the existing measurement-diagnostic Supplementary Figure.** This is an additional estimator-detail figure (three panels a/b/c), not a replacement.

### Extended Data source transfer — figED3_source_transfer

**Source-held-out diagnostics distinguish episode-level prediction from reference-range transfer.** **a**, R² for prediction of case-mean IPV when each source is held out; test-set independent-case counts are 23,218 (Waymo), 7,499 (nuPlan), 5,105 (Lyft) and 2,406 (AV2). **b**, Empirical coverage of the nominal 90% range and support-gate abstention when the human reference is refitted without each source. Coverage denominators are accepted moments; abstention denominators are readable moments before the support gate. Coverage counts are 143,380/193,096 (Waymo), 149,068/150,587 (nuPlan), 68,270/91,069 (Lyft), and 11,315/12,576 (AV2); abstention counts are 12,892/205,988, 7,351/157,938, 9,077/100,146 and 10,012/22,588, respectively. The dotted line is 90% nominal coverage. Both panels show frozen point estimates without uncertainty intervals, and concern distinct targets and analysis populations. Source data are provided as a Source Data file.

### Extended Data reference distribution — figED5_reference_distribution

**Readable human readings and deployment-context range summaries describe different populations.** **a**, Histogram of 486,660 readable human test-fold moments; bars give the percentage per bin across the seven-candidate span. **b**, Three retained deployment-context examples: each line joins the separate medians of the frozen 90% lower and upper bounds over all candidate moments in that context. Counts are 2,336 crossing/priority, 10,291 merging/priority and 291 merging/equal moments. These summaries come from the 67,861-candidate deployment benchmark without readability or support filtering; a line is not an observed single-frame interval or an empirical coverage interval. Source data are provided as a Source Data file.

## Delivered files and verification

- `process/make_fig1.py`: Figure 1, supplementary estimator details, and ED reference distribution.
- `process/make_fig3.py`: Figure 3 and ED source transfer.
- `process/fig13_sources.csv`: 10 per-panel source records including SHA-256, fields, population and transformations.
- `process/fig13_qa.json`: numeric checks, all visual-verdict iterations, export checks and hashes.
- Five bundles in `assets/`: `fig1_monitoring`, `fig3_monitor`, `figS4_estimator_details`, `figED3_source_transfer`, `figED5_reference_distribution`, each PDF/SVG/PNG.
- Ten CSVs in `source_data/`: fig1b trajectory, fig1c timeline, fig1 supplementary weight vectors, fig3 width/coverage, fig3 gates, fig3 R², ED3 prediction, ED3 reference metrics, ED5 histogram, ED5 context summaries.
- Both scripts executed with numeric assertions and passed `py_compile`. All five PDFs are one page with extractable text; SVGs preserve editable `<text>` elements. PNG visual QA passed after spacing corrections. Main figure 1/3 and ED3 final verdict scores are 95; supplemental details 94 and ED5 95.
- Final page-level manuscript QA is the integration lane's responsibility. No existing manuscript asset, frozen input, decision, or main.tex was modified by this lane.
