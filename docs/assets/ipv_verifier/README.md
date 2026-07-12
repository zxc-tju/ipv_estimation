# IPV verifier figure bundle

## Figure contract

- Core claim: the frozen RQ009 M3 verifier first checks whether the current
  context is supported, and only then compares the observed ego-window IPV
  against a context-conditioned human reference envelope.
- `ABSTAIN` means that no comparison is licensed; it is not an in-envelope or
  normal result.
- Figure 1 is schematic-led and uses one real OnSite external-application
  sequence. Figures 2 and 3 are data-led summaries of the frozen RQ009
  reference, evaluation evidence, and limits.

## Files

- `fig1_verifier_mechanism.*`: mechanism, real trajectory, dynamic envelope,
  and numerical support distance.
- `fig2_reference_data.*`: fold flow, source composition, context imbalance,
  and the 24 joint support cells.
- `fig3_evidence_and_limits.*`: marginal coverage, context-conditioning gain,
  subgroup abstention, and leave-one-dataset-out evidence.
- `source_data/`: panel-level source CSV files.
- `figure_manifest.csv`: figure/panel-to-source mapping.

Each figure is exported as a 300 dpi PNG, PDF, and editable SVG with live text.

## Rebuild

From the repository root:

```bash
.venv_ipv_local_test/bin/python scripts/build_ipv_verifier_explainer.py
```

The builder reads the frozen local RQ009/RQ012B artifacts, validates the core
row counts and support-cell totals, writes the panel source tables, and then
regenerates all figure formats.

## Interpretation boundaries

- The plotted OnSite `observed` value is `target_ipv_future`, an estimator
  output, not a manually annotated human-intent ground truth.
- The OnSite sequence demonstrates external application mechanics; it is not
  the RQ009 held-out human-reference validation domain.
- Signed exceedance is distance beyond the nearest envelope boundary, not a
  safety or causal effect measure.
- RQ009's temporal contract targets the future-window IPV around `t*+6`; an
  external runtime must preserve that alignment.
- The distance gate uses frozen imputation rules for missing distance features;
  missingness is retained in the diagnostic reasons rather than treated as an
  automatic engineering failure.

## QA record

- Exact fold, train-composition, joint-cell, test-gate, coverage, and selected
  OnSite case counts are asserted in the build script.
- The three PNG exports were visually inspected for legibility, overlap,
  contrast, panel labeling, and color consistency.
- SVG exports preserve text objects; PNG/PDF/SVG and all relative HTML links
  are checked in the accompanying workflow verification.
