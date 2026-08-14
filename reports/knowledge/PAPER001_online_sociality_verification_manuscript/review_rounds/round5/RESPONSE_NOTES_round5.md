# Round-5 response-letter material

## V08 — Absolute-width attack

We thank the reviewers for asking us to clarify the absolute width of the human reference range. The disclosure that the mean conditioned $90\%$ range occupies $1.87$ rad of the $2.36$ rad candidate span is the paper's own: we added it deliberately in round 4, and it now appears in both Section 2.3 and the Fig. 4 caption.

A wide human range is a property of human behavioural diversity under the available situation description, not a defect of the instrument. The monitor's value claim is the relative sharpening of the human reference by $21\%$, $20\%$ and $8\%$ at the $80\%$, $90\%$ and $95\%$ levels, respectively, together with calibration. Because the reference remains wide, a flag is a conservative clear-departure event rather than a fine discrimination; this is stated directly in the sentence quoted by the reviewer.

The manuscript also reports the width distribution rather than only its mean: the 5th--95th percentiles are $1.35$--$2.28$ rad. Per-source width distributions are logged as backlog item B09 and are not presented as completed analysis in this revision.

## V09 — E11 companion attacks

1. We thank the reviewers for asking us to clarify how the two negative-control tests bear on the reported association. The tests answer different questions. The case-label permutation destroys case composition rather than flag timing and yielded $p=0.1493$. The exposure placebo preserves composition while reassigning whole flag sequences across scenario runs and yielded an empirical $p=0.0199$. Its role is therefore timing-specific, as Methods 4.5 states explicitly: "the exposure placebo above is the test specific to flag timing."

2. The magnitude evidence does not rest on either of these $p$-values. It rests on the pre-specified fixed three-second-window analysis, for which the case-clustered intervals exclude zero at all three nominal levels, together with agreement in sign across all six level-by-window combinations. We also retain the acknowledged boundary in the manuscript: at the $90\%$ level, the open-ended contract-window interval crosses zero ($[-2.6100,+0.1372]$). Thus, the complete record distinguishes the supported fixed-window magnitude result from the weaker contract-window result without claiming that every negative control is significant.

3. We have kept the full inferential record in Methods 4.5 because moving it into Results would invert the manuscript's two-layer design. Results states the qualitative association, whereas Methods reports the resampling unit, placebo construction, draw count, statistic, interval pattern and boundary. To make that record immediately auditable, we have added a direct "(Methods 4.5)" pointer to the Results sentence describing the whole-sequence placebo.

4. The reason the exposure placebo is the confirmatory test for timing is already stated in print: Methods 4.5 says that "the exposure placebo above is the test specific to flag timing." Reassigning whole flag sequences across scenario runs tests whether the observed timing association survives while run composition is preserved; the case-label permutation instead destroys case composition and is therefore not a timing-specific test.

5. The frozen negative-control record also makes the empirical counts explicit: the exposure-placebo exceedance count is $4/201$, whereas the case-label-permutation count is $30/201$, as recorded in `rq018_rerun/negative_controls.json`. We quote these counts here for auditability but do not add them to the PI-ratified manuscript block.

6. Any refinement of draw-count precision is logged as backlog item B27 and will require a pre-commitment note before analysis; it is not presented as completed work in this revision.
