# Backlog delta — round 5 (NEW items only; B01–B17 in ../round1/BACKLOG_draft.md, B18–B22 in ../round2/BACKLOG_delta.md, B23–B26 in ../round3/BACKLOG_delta.md, B27–B29 in ../round4/BACKLOG_delta.md)

Round-5 reviewers re-demanded, without new facts, the standing items:
B01 (construct validation — A-M1/B-M4/C-M7/D-M6, FIFTH round; D again proposes the small rater study),
B02 (baseline incremental value — A-M3; plus the named formal-safety-monitor comparator facet, see B30 note below),
B03 (consequence landmark/matching redesign — A-M2/B-M8/C-M5; C-M5's "situation cells insufficiently specified" logged under B28),
B04 (sequential operating characteristics — B-M2/C-M3/A-M9; A-m7's flag-stretch/episode descriptives logged as a facet),
B05 (cluster-aware conformal — A-M4/B-M1/C-M1),
B06 (reproducibility documentation — B-m6/m11, C-m1/m2, D-m6; new facets: conformal-boundary tie handling B-m11, scene counts per fold C-m2, non-lane-referenced-slice agreement D-m6; NOTE: the quantile-model class/hyperparameters and feature contract are REMOVED from this cluster by plan item V06, which prints them from the frozen manifest),
B07 (estimator sensitivity — A-M8/B-M5/C-q13/D-q7; grid-boundary artefact C-m8),
B08 (multiplicity + non-inferiority — A-M7/C-M6/D-M3b),
B09 (uncertainty for descriptive panels — A-m3/B-m2/C-m3/C-m5; new facet: per-source width distributions, B-m8),
B10/B21 (equivalence margins for the ablation nulls — C-M7 part),
B11/B12/B20 (crossed inference + per-system distributions + denominators — A-M5/B-M7/C-M4/D-M2c/B-q10),
B13 (runtime implementation + assurance case — B-M3/B-M12/A-M9),
B14/B15 (sealed-split consumption + freeze artifacts — D-M7 in its sharpest form yet: "consume and report the confirmatory split before publication, or mark exploratory"; B-m4/C-q15),
B16/B19 (benchmark documentation + run ledger incl. the 65 outcome-absent runs — D-M8/D-m3/C-m9/A-m8/A-q6),
B17 (abstention–kinematics/conflict-time dependence — A-q7),
B22 (two-sided acceptance band + human-arm per-cell diagnostics, E-gated — B-M6/C-M2/D-M2; B-M6 now demands a pre-specified transfer estimand and equivalence margin, which is exactly the B22 band — ratify BEFORE unblinding),
B24 (freeze the partner-IPV correlation coefficient — C-m4),
B26 (per-endpoint placebo battery — C-M5 part),
B27 (permutation-precision upgrade ≥1,000 draws — A-M2/C-m7/D-m7, second round),
B28 (z_t temporal-provenance audit — C-M5/C-q9; V06's printed feature contract completes the documentation half; the lagged-refit half stays here),
B29 (Fig-1c gap-reason annotation — no round-5 re-raise; stands).
Do not re-log those. The genuinely new items:

## B30 — Nesting audit: do the 80/90/95 intervals stay nested after conformal widening?
- Needs: a one-pass audit over the frozen test-fold interval series checking, per accepted moment,
  that the 80% interval ⊆ 90% ⊆ 95% after adding the level-specific conformal radii
  (c_80=1.4435e-3, c_90=1.1921e-6, c_95=0.0 — `RQ021 key_numbers.json` human_only_envelope);
  report the count and location of any violations. The quantile levels are rearranged non-crossing
  BEFORE widening (printed, Methods 4.3); because the radii are non-monotone across levels
  (c_80 > c_90 > c_95), nesting after widening is plausible but not guaranteed and has never been
  checked or frozen.
- Resolves: B-m7 (round-5, NEW).
- Upgrades if done: a one-sentence Methods statement ("nesting verified on all accepted test
  moments" or an honest count), closing a small formal hole in the monitor's level semantics.
- Constraint memo: new (tiny) computation over frozen outputs → PI sign-off, log-don't-execute;
  freeze the audit output before any manuscript sentence cites it.

## B31 — Knowledge-layer registration + surfacing of the per-source deployed-reference coverage
- Needs: NO computation. The accepted RQ021_2 study already contains the per-source test-fold
  coverage of the DEPLOYED (all-source-fitted) reference:
  `reports/studies/RQ021_contemporaneous_envelope/RQ021_2_lodo_transfer_20260807T114305Z_0c4d280/key_numbers_e2.json`
  → `insample_by_source`, 90% level: Waymo 0.8824 (173,745/196,905), nuPlan 0.9589
  (145,846/152,093), Lyft 0.8588 (80,218/93,410), AV2 0.8821 (17,227/19,529); pooled 0.9028;
  per-source abstention 3.7–13.5%. What is needed is a one-line PI addendum in
  `reports/knowledge/RQ021_contemporaneous_envelope/decision.md` (mirroring the existing B2 LODO
  addendum) registering these as citable, then a one-sentence Methods 4.4 insertion.
- Resolves: D-q3 exactly ("per-source achieved coverage of the deployed reference — pooled coverage
  could mask per-source miscalibration"); bounds C-M1's "marginal coverage can coexist with
  arbitrarily poor coverage in particular sources" at the source level (observed floor 0.859, not
  arbitrary); complements, and is milder than, the printed LOSO boundary (0.743–0.990).
- Upgrades if done: converts a hostile hypothetical into a bounded, disclosed heterogeneity
  statement at zero analysis cost — the cheapest remaining evidence-surfacing move in the whole
  backlog.
- Constraint memo: study-layer numbers without a knowledge-layer claim id are not citable under the
  round-4 precedent (R4-37/B18); hence the registration step. Pre-commit to printing the numbers
  regardless of how they read (they are already fixed), consistent with freeze discipline.

## Facet note (no new id) — formal safety-monitor comparators
B-M9 demands running representative causal safety monitors (RSS-style safe distance, reachable-set
intersection, barrier-function robustness, predicted collision risk, STL robustness) on the same
timestamped streams and quantifying overlap, lead time and incremental value at flagged moments.
This is the sharpest statement yet of the comparator family B raised in rounds 1–3 and lives under
**B02** (incremental-value control), whose register memo (C6: increment not-claimed; a null would be
discoverable) applies unchanged. Logged here so the spec is not lost: same streams, matched
abstention/coverage budgets, overlap/disagreement/lead-time table, incremental value after
conditioning on the robustness score. §2.6's printed claim is threshold-scoped and needs no edit
while the increment stays unclaimed.
