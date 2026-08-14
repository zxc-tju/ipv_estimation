# Backlog delta — round 3 (NEW items only; B01–B17 in ../round1/BACKLOG_draft.md, B18–B22 in ../round2/BACKLOG_delta.md)

Round-3 reviewers re-demanded, without new facts, the standing items B01 (construct validation — A-M1/B-M1/C-M1),
B02 (baseline incremental-value control — A-M8/B-M7), B03 (consequence episode/landmark redesign — A-M3/B-M4/C-M7;
new facet noted below under B26), B04 (persistence operating characteristics + run-length structure — D-M6/A-q4/B-q2;
new facet: recompute Fig-5/6b signatures under minimum-persistence thresholds, logged inside B04),
B05 (cluster-aware conformal + group-conditional coverage — B-M2/C-M2/C-M3), B06 (reproducibility spec — A-m1/m2,
B-m10/m11, C-m1/m2/m9; new notes: bootstrap interval type + seeds, the 70.3%-vs-71.2% readable-rate universe
disambiguation, per-source flow), B07 (estimator sensitivity incl. grid endpoints and the 0.20 cutoff — A-M2/B-M6/C-m5/m6),
B08 (multiplicity/non-inferiority — B-M5/C-M8), B09 (uncertainty for descriptive panels incl. Fig-4c per-source R² —
A-m6/B-m4/m5/C-m3), B11–B12 (C7 completion + crossed-design inference — A-M4/B-M5/C-M6/D-M2), B13 (online
implementation study — A-q12/B-q11/C-q12), B14 (sealed-split unseal, PI-gated — B-q8/C-m7), B15 (freeze-artifact/
provenance — A-q13/B-m12/C-q10), B16 (benchmark documentation pack — D-m3/q9), B17 (abstention–kinematics
dependence — B-M3 facet), B19 (missing-run audit — D-m2/q8, C-q7, A-q9), B20 (alternative-denominator sensitivity —
C-m11 facet, B-m8 facet), B21 (oracle-counterpart + equivalence-grade ablation reporting — D-M5/q4, C-q11),
B22 (two-sided acceptance band + per-cell human-arm diagnostics, E-gated — A-q11, C-q5 facet). Do not re-log those.
The genuinely new items:

## B23 — Absolute-fit gate audit (machine-vs-human candidate-manifold fit)
- Needs: per-moment best-candidate MSE (the quantity already computed inside the likelihood step)
  summarised as distributions for (i) human-arm judgeable moments, (ii) automated-arm judgeable
  moments, (iii) natural-corpus accepted moments; then a pre-specified absolute-fit gate sweep
  (e.g., MSE percentile cutoffs) showing how the AV-vs-human flag-rate ratio moves; freeze before
  looking at outcomes.
- Resolves: D-M4, D-q5; A-q1 (part); the sharpest remaining measurement-artifact objection to the
  §2.5 comparison ("the reliability gate is relative — a trajectory poorly explained by all seven
  candidates can still be confidently mapped to the extreme candidate").
- Upgrades if done: removes model-misfit as an alternative explanation for the 2.1x contrast, or
  bounds its contribution honestly; either outcome converts D-M4 into a strength (the audit itself
  is in the paper's instrument-audit spirit).
- Constraint memo: AV-side computation is C7-independent; the human-arm mirror waits for
  REAL_VERIFIED. Any gate added to the monitor definition is a freeze-order change → PI ruling
  first; run it as a diagnostic, not a new gate.

## B24 — Freeze the event-level partner-IPV correlation
- Needs: the correlation between interacting partners' episode-level IPVs (the quantity §2.2
  asserts qualitatively), computed on the frozen corpus with case-clustered CI, per source and
  pooled; freeze into an RQ decision (extends the RQ004 family). No manuscript number exists today:
  claims register C1/C1b sanction the qualitative claim only.
- Resolves: A-m11 (quantify the partner correlation); also strengthens the headline event-vs-case
  tension (§2.3) by giving its premise a magnitude.
- Upgrades if done: "partners' preferences co-vary" becomes citable with a number; the mediation
  story ("the correlation is carried by the shared observable situation") becomes testable rather
  than narrative.
- Constraint memo: purely descriptive addition; no endpoint change; but the number must clear the
  knowledge-layer freeze before any digit enters the text.

## B25 — Ablation replication on held-out sources (counterpart channel and self-history)
- Needs: rerun the frozen counterpart-IPV and self-history ablations with one source held out
  (mirroring the LOSO protocol), reporting interval-score differences with cluster-aware CIs and a
  pre-declared equivalence margin on each held-out source.
- Resolves: C-M9 ("test the ablation on untouched sources"); complements B10 (equivalence margin)
  and B21 (oracle-counterpart attenuation bound).
- Upgrades if done: "the situation suffices" becomes robust to the objection that the null is a
  within-mixture artefact; protects the Discussion's "no reading of hidden intention" claim.
- Constraint memo: uses only frozen splits and the LOSO machinery; still NEW analysis → log, do not
  execute; freeze margins before computation (deciding them after seeing results voids them).

## B26 — Per-endpoint timing-placebo battery under the episode redesign
- Needs: once B03's episode-level redesign exists, a pre-specified timing-placebo (whole-trajectory
  reassignment) for EACH primary endpoint and side, reported as a table rather than a single
  omnibus p; interim: the existing per-endpoint placebo entries in
  `rq018_rerun/negative_controls.json` (p ranging 0.005–0.985 across endpoint × window × side)
  are already frozen — their DISCLOSURE is the E11 decision, not new analysis.
- Resolves: B-M4 ("whole-sequence timing placebos should be reported for every primary endpoint,
  not only as a single omnibus statement"); C-M8 (selective-summary concern).
- Upgrades if done: the consequence battery's negative-control story becomes endpoint-resolved and
  immune to the selective-disclosure objection that E7 data access would otherwise expose.
- Constraint memo: strictly sequenced AFTER the E11 ruling (what may be printed now) and WITH B03
  (the redesign defines the endpoints the battery attaches to).
