# Backlog delta — round 4 (NEW items only; B01–B17 in ../round1/BACKLOG_draft.md, B18–B22 in ../round2/BACKLOG_delta.md, B23–B26 in ../round3/BACKLOG_delta.md)

Round-4 reviewers re-demanded, without new facts, the standing items B01 (construct validation —
A-M1/B-M10/C-M1, fourth round), B02 (baseline incremental-value control — A-M2/D-M7), B03 (consequence
episode/landmark redesign — A-M3/B-M6/C-M6/D-M4; D's matched-kinematics control formulation logged as
the sharpest statement of the existing matching facet), B04 (sequential/persistence operating
characteristics — B-M1/C-M8/A-M7), B05 (cluster-aware conformal — B-M2/C-M2/A-M4), B06
(reproducibility spec — A-m1/m2, B-m8/m10/m12, C-m3/m5, plus the new paired-anchor-row documentation
note from B-m8), B07 (estimator sensitivity incl. frozen constants sigma/0.20/k=25 — A-M6/B-M3/D-m5),
B08 (multiplicity + non-inferiority — A-m7/B-M7/C-M7), B09 (uncertainty for descriptive panels —
B-m4/C-m1/C-m6; new facet: per-pair Fig-2c flip rates with CIs), B10/B21 (equivalence margins + oracle
ablation — D-M6/C-q10), B11–B12 (C7 completion + crossed inference — A-M4/B-M5/C-M4/D-M3; D's
per-system distribution facet also feeds E10), B13 (online implementation study — B-M9/B-m13/D-m7),
B14/B15 (sealed-split + freeze artifacts — C-M9/C-q4), B16/B19 (benchmark documentation + missing-run
audit — D-M5/B-q8/C-q8), B17 (abstention–kinematics dependence; carries D-q8's 70.3-vs-55.3 diagnostic
demand), B18 (over-yielding outcome characterisation — B-m7), B20 (alternative denominators — C-M4
facet), B22 (two-sided acceptance band + per-cell human-arm diagnostics, E-gated — B-M4/C-M3/D-M3),
B23 (absolute-fit gate audit), B24 (freeze the partner-IPV correlation — D-M6 explicitly: "no
coefficient, no CI, no method, anywhere"), B25 (held-out-source ablation replication), B26
(per-endpoint placebo battery). Do not re-log those. The genuinely new items:

## B27 — Permutation-precision upgrade (≥1,000 draws for both negative controls)
- Needs: rerun the frozen exposure-timing placebo AND the case-level label permutation with ≥1,000
  whole-trajectory reassignment draws (same frozen spec: statistic = absolute case-clustered t, same
  seed discipline, same endpoint battery), reporting each empirical p with its Monte-Carlo standard
  error; update `negative_controls.json`-style ledger and freeze before any manuscript digit changes.
- Resolves: D-M4 ("raise placebo draws to ≥1,000"), D-m4 (200 draws inconsistent with 1,000–2,000
  bootstrap resamples elsewhere), C-M7 ("more permutation draws and Monte Carlo uncertainty are needed
  for a reported probability near 0.02").
- Upgrades if done: the printed p=0.0199 (currently 4/201, resolution ~0.005) becomes
  resolution-defensible; removes the cheapest quantitative jab at the consequence battery's control.
- Constraint memo: this is a recomputation of a frozen quantity → PI freeze-discipline sign-off first;
  pre-commit to publishing the new p regardless of direction (deciding after seeing it voids the
  control); the E11 companion wording would then be updated in one pass under the same register rule.
  Cheap to run; strictly sequenced with (not blocked by) B26's episode-level battery.

## B28 — Temporal-provenance audit of the situation vector z_t (+ lagged-covariate sensitivity)
- Needs: (i) a feature-by-feature table for every z_t component: measurement timestamp(s), derivation
  window, whether it can be influenced by the ego action inside the one-second estimation window, and
  training status (extends the B06/B15 feature contract); (ii) a time-order diagram separating
  pre-decision context from potentially behaviour-generated state; (iii) a sensitivity refit of the
  reference conditioning on suitably lagged pre-window covariates (and map/role-only conditioning),
  reporting width/coverage deltas and verdict-overlap rates against the frozen reference.
- Resolves: C-M5 (conditioning on behaviour-generated state can normalise part of an assertive
  manoeuvre into its own reference range; collider concern; "list every feature and its temporal
  offset"), C-q5; strengthens the answer to R4-19 and narrows what remains of the "same situation"
  estimand objection.
- Upgrades if done: "conditioned on the current observable situation" acquires an explicit
  pre-decision reading, or the paper gains an honest bounded statement about which components are
  contemporaneous with the judged action — either way the collider attack loses its open-endedness.
- Constraint memo: the audit table is documentation (fast, no endpoint change); the lagged refit is a
  NEW analysis over frozen splits → log, do not execute; freeze the lag convention before computing.

## B29 — Fig-1c abstention-gap reason annotation
- Needs: extract the per-frame abstention reason codes for the concept-figure interaction from the
  frozen verdict series, add them to the fig0 restyle input, and regenerate panel c with the grey
  gaps labelled (or glyph-coded) by reason — plus a legibility pass reconciling the "reading issued"
  strip with the gap semantics at display size.
- Resolves: D-m10 (grey gaps vs top strip hard to reconcile; "consider annotating the abstention
  reasons on the gaps").
- Upgrades if done: the concept figure then demonstrates the reason-code discipline the text
  advertises (auditable abstention), at zero claim risk — a display upgrade fully in the paper's
  abstention-first spirit.
- Constraint memo: the current frozen restyle input (`restyle_fig0_concept.py` parquet) carries no
  reason fields (verified), so this needs a data-plumbing step → new computation, log only. Once the
  enriched input exists the change itself is FIX-FIGURE-eligible (label-only, acceptance harness,
  figH untouched).
